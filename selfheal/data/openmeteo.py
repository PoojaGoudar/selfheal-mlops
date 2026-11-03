"""Live history from Open-Meteo (free, no API key). Network access is required.

Endpoints used (see https://open-meteo.com/en/docs/air-quality-api and /docs):
  * air-quality-api.open-meteo.com/v1/air-quality  -> pm2_5, pm10, nitrogen_dioxide, ozone
  * api.open-meteo.com/v1/forecast                 -> temperature, humidity, wind, pressure, precipitation
Both accept ``past_days`` (max 92). Check Open-Meteo's terms for non-commercial use.
"""
from __future__ import annotations

import pandas as pd
import requests

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_history(lat: float, lon: float, past_days: int = 90, timeout: int = 30) -> pd.DataFrame:
    common = {"latitude": lat, "longitude": lon, "past_days": min(past_days, 92), "forecast_days": 1, "timezone": "Asia/Kolkata"}
    aq = requests.get(AQ_URL, params={**common, "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone"}, timeout=timeout)
    wx = requests.get(WX_URL, params={**common, "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,surface_pressure,precipitation"}, timeout=timeout)
    aq.raise_for_status()
    wx.raise_for_status()
    a, w = pd.DataFrame(aq.json()["hourly"]), pd.DataFrame(wx.json()["hourly"])
    df = a.merge(w, on="time", how="inner").rename(columns={
        "time": "ts", "nitrogen_dioxide": "no2", "ozone": "o3", "temperature_2m": "temp",
        "relative_humidity_2m": "humidity", "wind_speed_10m": "wind", "surface_pressure": "pressure", "precipitation": "precip"})
    df["ts"] = pd.to_datetime(df["ts"])
    df = df[df["ts"] <= pd.Timestamp.now().floor("h")]  # drop forecast hours
    df = df.dropna(subset=["pm2_5", "temp", "humidity", "wind", "pressure"]).fillna({"pm10": 0.0, "no2": 0.0, "o3": 0.0, "precip": 0.0})
    df["regime"] = "normal"
    return df.sort_values("ts").reset_index(drop=True)
