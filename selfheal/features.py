"""Feature engineering shared by training AND serving (one code path = no train/serve skew).

At time t we know PM2.5 up to and including t, plus the weather at t.
The target is PM2.5 at t+1 (next hour).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

WEATHER = ["temp", "humidity", "wind", "pressure", "precip"]
PM_LAGS = [1, 2, 5, 11, 23]  # lag k = value k hours before "now"
FEATURES = (
    ["pm25_lag0"]
    + [f"pm25_lag{k}" for k in PM_LAGS]
    + ["pm25_roll_mean_6", "pm25_roll_mean_24", "pm25_roll_std_6"]
    + WEATHER
    + ["hour_sin", "hour_cos", "is_weekend"]
)
# Continuous features that are meaningful to test for distribution drift.
DRIFT_FEATURES = ["pm25_lag0", "pm25_roll_mean_24", "pm25_roll_std_6", "temp", "humidity", "wind", "pressure"]
MIN_HISTORY = 24
TARGET = "target"


def make_features(df: pd.DataFrame, with_target: bool = True) -> pd.DataFrame:
    """Batch feature builder used by training/retraining. ``df`` needs ts, pm2_5 and weather columns."""
    d = df.sort_values("ts").reset_index(drop=True)
    out = pd.DataFrame({"ts": pd.to_datetime(d["ts"])})
    pm = d["pm2_5"].astype(float)
    out["pm25_lag0"] = pm
    for k in PM_LAGS:
        out[f"pm25_lag{k}"] = pm.shift(k)
    out["pm25_roll_mean_6"] = pm.rolling(6).mean()
    out["pm25_roll_mean_24"] = pm.rolling(24).mean()
    out["pm25_roll_std_6"] = pm.rolling(6).std()  # ddof=1, matched in features_from_request
    for c in WEATHER:
        out[c] = d[c].astype(float)
    hour = out["ts"].dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["is_weekend"] = (out["ts"].dt.dayofweek >= 5).astype(float)
    if with_target:
        out[TARGET] = pm.shift(-1)
        return out.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return out.dropna(subset=FEATURES).reset_index(drop=True)


def features_from_request(pm25_history: list[float], weather: dict, ts: str | datetime) -> dict:
    """Single-row feature builder for the API. History is oldest -> newest; last value = now."""
    if len(pm25_history) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} hourly PM2.5 values, got {len(pm25_history)}")
    h = np.asarray(pm25_history, dtype=float)
    t = pd.to_datetime(ts)
    f = {"pm25_lag0": h[-1]}
    for k in PM_LAGS:
        f[f"pm25_lag{k}"] = h[-1 - k]
    f["pm25_roll_mean_6"] = h[-6:].mean()
    f["pm25_roll_mean_24"] = h[-24:].mean()
    f["pm25_roll_std_6"] = h[-6:].std(ddof=1)
    for c in WEATHER:
        f[c] = float(weather[c])
    f["hour_sin"] = float(np.sin(2 * np.pi * t.hour / 24))
    f["hour_cos"] = float(np.cos(2 * np.pi * t.hour / 24))
    f["is_weekend"] = float(t.dayofweek >= 5)
    return {k: float(v) for k, v in f.items()}
