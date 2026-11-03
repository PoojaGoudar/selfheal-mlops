"""FastAPI inference service. Models are loaded from the MLflow registry BY ALIAS, never from a file path:
promotion = moving an alias; the API hot-reloads within a couple of seconds."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from selfheal import registry, store
from selfheal.config import settings
from selfheal.features import FEATURES, features_from_request
from selfheal.serving.router import route

app = FastAPI(title="Self-healing AQI forecaster", version="1.0")


class PredictRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = Field(description="ISO timestamp of the latest observation")
    pm25_history: list[float] = Field(description="Hourly PM2.5, oldest -> newest (>=24 values)")
    temp: float
    humidity: float
    wind: float
    pressure: float
    precip: float = 0.0


class FeedbackRequest(BaseModel):
    request_id: str
    y_true: float


class _Runtime:
    def __init__(self):
        self.lock = threading.Lock()
        self.models: dict[str, tuple[int, object] | None] = {"champion": None, "challenger": None}
        self.last_refresh = 0.0
        self.interval = float(os.getenv("MODEL_REFRESH_S", "2"))

    def refresh(self, force: bool = False) -> None:
        if not force and time.time() - self.last_refresh < self.interval:
            return
        with self.lock:
            self.last_refresh = time.time()
            for role, alias in (("champion", registry.CHAMPION), ("challenger", registry.CHALLENGER)):
                v = registry.alias_version(alias)
                cur = self.models[role]
                if v is None:
                    self.models[role] = None
                elif cur is None or cur[0] != v:
                    self.models[role] = (v, registry.load_model(v))


RT = _Runtime()


@app.on_event("startup")
def _startup() -> None:
    store.init_db()
    try:
        RT.refresh(force=True)
    except Exception as e:  # service starts even if bootstrap has not run yet
        print("startup: no model yet:", e)


@app.get("/health")
def health():
    RT.refresh()
    ch, cl = RT.models["champion"], RT.models["challenger"]
    return {"status": "ok" if ch else "no_champion", "champion": ch[0] if ch else None,
            "challenger": cl[0] if cl else None, "rollout_stage": store.get_control("rollout_stage", "none")}


@app.post("/reload")
def reload():
    RT.refresh(force=True)
    return health()


@app.post("/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    RT.refresh()
    champ, chall = RT.models["champion"], RT.models["challenger"]
    if champ is None:
        raise HTTPException(503, "No champion model registered. Run `make bootstrap` first.")
    try:
        feats = features_from_request(req.pm25_history, req.model_dump(), req.ts)
    except ValueError as e:
        raise HTTPException(422, str(e))
    X = pd.DataFrame([feats])[FEATURES]
    champ_pred = float(champ[1].predict(X)[0])
    chall_pred = float(chall[1].predict(X)[0]) if chall else None
    cfg = settings()
    stage = store.get_control("rollout_stage", "none")
    served_by = route(req.request_id, stage, int(store.get_control("canary_pct", cfg.canary_pct)), chall is not None)
    served = chall_pred if served_by == "challenger" else champ_pred
    latency = time.perf_counter() - t0
    store.insert_prediction({
        "request_id": req.request_id, "created_at": store.now_iso(),
        "target_ts": str(pd.to_datetime(req.ts) + pd.Timedelta(hours=1)), "features": json.dumps(feats),
        "champion_version": champ[0], "champion_pred": champ_pred,
        "challenger_version": chall[0] if (chall and stage in ("shadow", "canary")) else None,
        "challenger_pred": chall_pred if stage in ("shadow", "canary") else None,
        "served_by": served_by, "served_pred": served, "latency_ms": latency * 1000})
    return {"request_id": req.request_id, "prediction_pm25_next_hour": round(served, 2),
            "model_version": (chall if served_by == "challenger" else champ)[0], "served_by": served_by}


@app.post("/feedback")
def feedback(fb: FeedbackRequest):
    if not store.record_feedback(fb.request_id, fb.y_true):
        raise HTTPException(404, "unknown request_id")
    return {"ok": True}


@app.get("/stats")
def stats():
    """Lightweight service monitoring: traffic, latency and rollout split over the last 500 requests."""
    df = store.recent_predictions(500)
    if df.empty:
        return {"requests": 0}
    return {"requests_last_500": int(len(df)), "labeled": int(df["y_true"].notna().sum()),
            "latency_ms_p50": round(float(df["latency_ms"].quantile(0.5)), 2),
            "latency_ms_p95": round(float(df["latency_ms"].quantile(0.95)), 2),
            "share_served_by_challenger": round(float((df["served_by"] == "challenger").mean()), 3)}
