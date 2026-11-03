"""Live-stream simulator: one tick = one simulated hour. Calls the API like a real client would.

The 'chaos' regime is read from the control table every tick (the dashboard buttons set it)."""
from __future__ import annotations

import argparse
import time
import uuid
from pathlib import Path

import pandas as pd
import requests

from selfheal import store
from selfheal.config import settings
from selfheal.data.synthetic import SyntheticAir

WEATHER = ["temp", "humidity", "wind", "pressure", "precip"]


class ReplaySource:
    """Replays real (Open-Meteo) history and applies the chaos regimes as transformations."""

    def __init__(self, csv: Path):
        self.df = pd.read_csv(csv, parse_dates=["ts"]).reset_index(drop=True)
        self.i = 0

    def next(self, ts, regime: str = "normal") -> dict:
        row = self.df.iloc[self.i % len(self.df)].to_dict()
        self.i += 1
        row["ts"], row["regime"] = pd.Timestamp(ts), regime
        if regime == "smoke_event":
            row["pm2_5"] = row["pm2_5"] * 1.5 + 40
            row["temp"] += 5
            row["humidity"] = max(10.0, row["humidity"] - 15)
        elif regime == "sensor_fault":
            row["wind"] = 0.0
        return row


class HTTPClient:
    def __init__(self, base: str):
        self.base, self.s = base.rstrip("/"), requests.Session()

    def predict(self, payload: dict) -> dict:
        r = self.s.post(f"{self.base}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()

    def feedback(self, request_id: str, y_true: float) -> None:
        self.s.post(f"{self.base}/feedback", json={"request_id": request_id, "y_true": y_true}, timeout=10)


class TestClientAdapter:
    """Wrap a FastAPI TestClient (used by tests and the headless end-to-end run)."""

    def __init__(self, tc):
        self.tc = tc

    def predict(self, payload: dict) -> dict:
        r = self.tc.post("/predict", json=payload)
        r.raise_for_status()
        return r.json()

    def feedback(self, request_id: str, y_true: float) -> None:
        self.tc.post("/feedback", json={"request_id": request_id, "y_true": y_true})


class Simulator:
    def __init__(self, client, source=None):
        cfg = settings()
        store.init_db()
        hist = store.read_observations(48)
        self.hist = hist["pm2_5"].tolist()
        self.ts = (hist["ts"].iloc[-1] if len(hist) else pd.Timestamp.now().floor("h")) + pd.Timedelta(hours=1)
        replay = cfg.data_dir / "replay.csv"
        if source is not None:
            self.source = source
        elif cfg.sim_source == "replay" and replay.exists():
            self.source = ReplaySource(replay)
        else:
            self.source = SyntheticAir(seed=int(time.time()) % 10_000, pm0=self.hist[-1] if self.hist else 35.0)
        self.client, self.prev = client, None

    def tick(self) -> dict:
        regime = store.get_control("regime", "normal")
        row = self.source.next(self.ts, regime)
        store.insert_observation(row)
        if self.prev:
            self.client.feedback(self.prev, row["pm2_5"])  # the truth for last hour's forecast has arrived
        self.hist = (self.hist + [row["pm2_5"]])[-48:]
        out = None
        if len(self.hist) >= 24:
            rid = uuid.uuid4().hex
            out = self.client.predict({"request_id": rid, "ts": str(self.ts), "pm25_history": self.hist,
                                       **{k: row[k] for k in WEATHER}})
            self.prev = rid
        self.ts += pd.Timedelta(hours=1)
        return {"ts": str(row["ts"]), "pm2_5": row["pm2_5"], "regime": regime, "prediction": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=0, help="0 = run forever")
    a = ap.parse_args()
    cfg = settings()
    while True:  # do not start streaming before the API has a champion model (bootstrap may still be running)
        try:
            if requests.get(f"{cfg.api_url}/health", timeout=5).json().get("champion"):
                break
        except (requests.RequestException, ValueError):
            pass
        print("waiting for the API and a champion model...", flush=True)
        time.sleep(3)
    sim = Simulator(HTTPClient(cfg.api_url))
    n = 0
    while not a.ticks or n < a.ticks:
        try:
            sim.tick()
        except requests.RequestException as e:
            print("API not reachable yet:", e, flush=True)
            time.sleep(2)
            continue
        n += 1
        time.sleep(cfg.sim_seconds_per_hour)


if __name__ == "__main__":
    main()
