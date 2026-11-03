"""Central configuration. Every value can be overridden with an environment variable.

Settings are re-read on every call to ``settings()`` so tests (and the dashboard)
can change environment variables without re-importing modules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default, cast=str):
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    return cast(raw)


@dataclass(frozen=True)
class Settings:
    # --- storage -----------------------------------------------------------
    data_dir: Path
    model_name: str
    experiment: str
    # --- location (default: Bengaluru) ------------------------------------
    lat: float
    lon: float
    # --- services ---------------------------------------------------------
    api_url: str
    # --- simulator --------------------------------------------------------
    sim_seconds_per_hour: float
    sim_source: str  # "synthetic" | "replay"
    # --- drift / alerting -------------------------------------------------
    drift_window: int
    psi_threshold: float
    psi_min_features: int
    mae_ratio_threshold: float
    mae_window: int
    min_labeled_for_alert: int
    # --- healing pipeline -------------------------------------------------
    min_new_hours: int
    retrain_window_hours: int
    holdout_hours: int
    half_life_hours: float
    min_improvement: float
    shadow_min: int
    canary_min: int
    canary_pct: int
    auto_approve: bool
    monitor_interval_s: float
    # --- LLM copilot ------------------------------------------------------
    anthropic_api_key: str
    anthropic_model: str
    llm_price_in: float  # USD per million input tokens (estimate, set to your plan)
    llm_price_out: float

    @property
    def db_path(self) -> Path:
        return self.data_dir / "selfheal.db"

    @property
    def tracking_uri(self) -> str:
        return os.getenv("MLFLOW_TRACKING_URI") or f"sqlite:///{(self.data_dir / 'mlflow.db').as_posix()}"

    @property
    def artifact_root(self) -> str:
        return (self.data_dir / "mlartifacts").resolve().as_uri()


def settings() -> Settings:
    return Settings(
        data_dir=Path(_env("SELFHEAL_DATA_DIR", "./runtime")).resolve(),
        model_name=_env("MODEL_NAME", "aqi-forecaster"),
        experiment=_env("MLFLOW_EXPERIMENT", "aqi-forecasting"),
        lat=_env("LOCATION_LAT", 12.9716, float),
        lon=_env("LOCATION_LON", 77.5946, float),
        api_url=_env("API_URL", "http://localhost:8000"),
        sim_seconds_per_hour=_env("SIM_SECONDS_PER_HOUR", 0.5, float),
        sim_source=_env("SIM_SOURCE", "synthetic"),
        drift_window=_env("DRIFT_WINDOW", 168, int),
        psi_threshold=_env("PSI_THRESHOLD", 0.25, float),
        psi_min_features=_env("PSI_MIN_FEATURES", 3, int),
        mae_ratio_threshold=_env("MAE_RATIO_THRESHOLD", 1.5, float),
        mae_window=_env("MAE_WINDOW", 48, int),
        min_labeled_for_alert=_env("MIN_LABELED_FOR_ALERT", 24, int),
        min_new_hours=_env("MIN_NEW_HOURS", 72, int),
        retrain_window_hours=_env("RETRAIN_WINDOW_HOURS", 336, int),
        holdout_hours=_env("HOLDOUT_HOURS", 36, int),
        half_life_hours=_env("HALF_LIFE_HOURS", 72.0, float),
        min_improvement=_env("MIN_IMPROVEMENT", 0.05, float),
        shadow_min=_env("SHADOW_MIN", 24, int),
        canary_min=_env("CANARY_MIN", 12, int),
        canary_pct=_env("CANARY_PCT", 25, int),
        auto_approve=_env("AUTO_APPROVE", False, bool),
        monitor_interval_s=_env("MONITOR_INTERVAL_S", 2.0, float),
        anthropic_api_key=_env("ANTHROPIC_API_KEY", ""),
        anthropic_model=_env("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        llm_price_in=_env("LLM_PRICE_IN", 3.0, float),
        llm_price_out=_env("LLM_PRICE_OUT", 15.0, float),
    )
