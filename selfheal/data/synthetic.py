"""Synthetic hourly air-quality generator with switchable regimes.

Regimes
-------
normal        stable relationships (wind and rain clean the air, humidity traps particles)
smoke_event   CONCEPT + DATA DRIFT: baseline jumps, wind now *carries* smoke (sign flips),
              temperature up, humidity down
sensor_fault  DATA-QUALITY problem: the wind sensor reports a constant 0.0 while the real
              air quality is unchanged. The right fix is the sensor, NOT retraining.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ["normal", "smoke_event", "sensor_fault"]


class SyntheticAir:
    def __init__(self, seed: int = 42, pm0: float = 35.0):
        self.rng = np.random.default_rng(seed)
        self.pm = pm0
        self.wind = 2.2
        self.hum = 65.0

    def next(self, ts, regime: str = "normal") -> dict:
        ts = pd.Timestamp(ts)
        r, h = self.rng, ts.hour
        doy = ts.dayofyear
        temp = 25 + 4 * np.sin(2 * np.pi * (h - 14) / 24) + 2 * np.sin(2 * np.pi * doy / 365) + r.normal(0, 0.7)
        self.wind = float(np.clip(0.8 * self.wind + 0.2 * 2.2 + r.normal(0, 0.45), 0.1, 9.0))
        self.hum = float(np.clip(0.85 * self.hum + 0.15 * (65 - 15 * np.sin(2 * np.pi * (h - 14) / 24)) + r.normal(0, 2.0), 15, 100))
        pressure = 1008 + 2 * np.sin(2 * np.pi * h / 12) + r.normal(0, 0.8)
        rain = float(r.random() < 0.03)
        precip = rain * float(r.exponential(1.5))
        diurnal = 10 * np.sin(2 * np.pi * (h - 9) / 24)

        if regime == "smoke_event":
            target = 80 + 1.5 * diurnal + 0.25 * (self.hum - 65) + 4.0 * (self.wind - 2.2)
        else:  # normal and sensor_fault share the same physics
            target = 35 + diurnal + 0.25 * (self.hum - 65) - 3.0 * (self.wind - 2.2) - 4 * rain
        self.pm = float(np.clip(0.8 * self.pm + 0.2 * target + r.normal(0, 2.5), 2, 500))

        obs_temp, obs_hum, obs_wind = temp, self.hum, self.wind
        if regime == "smoke_event":
            obs_temp, obs_hum = temp + 5, float(np.clip(self.hum - 15, 10, 100))
        if regime == "sensor_fault":
            obs_wind = 0.0
        return {
            "ts": ts, "pm2_5": self.pm, "pm10": self.pm * 1.6 + r.normal(0, 3), "no2": 20 + 0.2 * self.pm + r.normal(0, 2),
            "o3": 40 - 0.1 * self.pm + r.normal(0, 3), "temp": float(obs_temp), "humidity": float(obs_hum),
            "wind": float(obs_wind), "pressure": float(pressure), "precip": precip, "regime": regime,
        }


def generate(n_hours: int, start, seed: int = 42, regime_fn=None) -> pd.DataFrame:
    gen = SyntheticAir(seed)
    idx = pd.date_range(pd.Timestamp(start), periods=n_hours, freq="h")
    rows = [gen.next(t, regime_fn(i) if regime_fn else "normal") for i, t in enumerate(idx)]
    return pd.DataFrame(rows)
