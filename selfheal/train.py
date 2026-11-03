"""Training / retraining with full MLflow tracking and Model Registry registration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature

from selfheal import registry
from selfheal.config import settings
from selfheal.features import FEATURES, TARGET, make_features

DEFAULT_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=10,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=0, verbose=-1, n_jobs=1)


def git_sha() -> str:
    if os.getenv("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def recency_weights(ts: pd.Series, half_life_h: float | None) -> np.ndarray | None:
    if not half_life_h:
        return None
    age_h = (ts.max() - ts).dt.total_seconds() / 3600.0
    return np.power(0.5, age_h / half_life_h).to_numpy()


def split_time(feat: pd.DataFrame, holdout_hours: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split. NEVER random for time series (that would leak the future)."""
    return feat.iloc[:-holdout_hours].copy(), feat.iloc[-holdout_hours:].copy()


def fit(train: pd.DataFrame, half_life_h: float | None, params: dict | None = None) -> LGBMRegressor:
    model = LGBMRegressor(**{**DEFAULT_PARAMS, **(params or {})})
    model.fit(train[FEATURES], train[TARGET], sample_weight=recency_weights(train["ts"], half_life_h))
    return model


def train_and_register(obs: pd.DataFrame, *, reason: str, holdout_hours: int, half_life_h: float | None,
                       reference_rows: int, register: bool = True, params: dict | None = None,
                       extra_tags: dict | None = None) -> dict:
    """Train on ``obs`` (raw observations), log everything to MLflow, register a new model version."""
    registry.init()
    feat = make_features(obs)
    train, hold = split_time(feat, holdout_hours)
    model = fit(train, half_life_h, params)
    pred = model.predict(hold[FEATURES])
    metrics = {
        "val_mae": mae(hold[TARGET], pred),
        "val_rmse": float(np.sqrt(np.mean((hold[TARGET] - pred) ** 2))),
        "persistence_mae": mae(hold[TARGET], hold["pm25_lag0"]),
        "n_train": float(len(train)), "n_holdout": float(len(hold)),
    }
    metrics["gain_vs_persistence"] = 1 - metrics["val_mae"] / max(metrics["persistence_mae"], 1e-9)
    data_hash = hashlib.md5(pd.util.hash_pandas_object(obs[["ts", "pm2_5"]], index=False).values.tobytes()).hexdigest()[:12]

    with mlflow.start_run(run_name=f"train-{reason}") as run:
        mlflow.log_params({**DEFAULT_PARAMS, **(params or {}), "holdout_hours": holdout_hours,
                           "half_life_hours": half_life_h or 0, "n_features": len(FEATURES)})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"reason": reason, "git_sha": git_sha(), "data_hash": data_hash,
                         "window_start": str(feat["ts"].iloc[0]), "window_end": str(feat["ts"].iloc[-1]),
                         "holdout_start": str(hold["ts"].iloc[0]), **(extra_tags or {})})
        # Reference distribution = what "normal" looks like for THIS model (used by the drift monitor).
        with tempfile.TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "reference.csv"
            feat[FEATURES].tail(reference_rows).to_csv(ref_path, index=False)
            mlflow.log_artifact(str(ref_path))
        sig = infer_signature(hold[FEATURES], pred)
        kwargs = {"registered_model_name": settings().model_name} if register else {}
        mlflow.sklearn.log_model(model, name="model", signature=sig, serialization_format="cloudpickle", **kwargs)
        run_id = run.info.run_id

    version = None
    if register:
        mvs = registry.init().search_model_versions(f"run_id='{run_id}'")
        version = int(mvs[0].version)
    return {"version": version, "run_id": run_id, "metrics": metrics, "model": model, "holdout": hold}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train from a CSV (used by the DVC pipeline).")
    ap.add_argument("--data", default="data/raw/air.csv")
    ap.add_argument("--holdout-hours", type=int, default=168)
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--metrics-out", default="models/metrics.json")
    a = ap.parse_args()
    df = pd.read_csv(a.data, parse_dates=["ts"])
    res = train_and_register(df, reason="dvc-pipeline", holdout_hours=a.holdout_hours, half_life_h=None,
                             reference_rows=1500, register=not a.no_register)
    Path(a.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.metrics_out).write_text(json.dumps({k: round(v, 4) for k, v in res["metrics"].items()}, indent=2))
    print(json.dumps(res["metrics"], indent=2))


if __name__ == "__main__":
    main()
