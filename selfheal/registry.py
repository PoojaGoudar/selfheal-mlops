"""Thin wrapper around MLflow tracking + Model Registry (aliases: ``champion`` / ``challenger``)."""
from __future__ import annotations

import logging
import warnings
from functools import lru_cache

import mlflow
import pandas as pd
from mlflow import MlflowClient

from selfheal.config import settings

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
logging.getLogger("mlflow").setLevel(logging.ERROR)
CHAMPION, CHALLENGER = "champion", "challenger"


def init() -> MlflowClient:
    cfg = settings()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_registry_uri(cfg.tracking_uri)
    client = MlflowClient()
    if client.get_experiment_by_name(cfg.experiment) is None:
        try:
            client.create_experiment(cfg.experiment, artifact_location=cfg.artifact_root)
        except Exception:  # another container created it a moment earlier: that is fine
            if client.get_experiment_by_name(cfg.experiment) is None:
                raise
    mlflow.set_experiment(cfg.experiment)
    return client


def alias_version(alias: str) -> int | None:
    try:
        return int(init().get_model_version_by_alias(settings().model_name, alias).version)
    except Exception:
        return None


def set_alias(alias: str, version: int) -> None:
    init().set_registered_model_alias(settings().model_name, alias, str(version))


def delete_alias(alias: str) -> None:
    try:
        init().delete_registered_model_alias(settings().model_name, alias)
    except Exception:
        pass


def load_model(version: int):
    init()
    return mlflow.sklearn.load_model(f"models:/{settings().model_name}/{version}")


def version_info(version: int) -> dict:
    c = init()
    mv = c.get_model_version(settings().model_name, str(version))
    run = c.get_run(mv.run_id)
    return {"version": int(version), "run_id": mv.run_id, "metrics": dict(run.data.metrics),
            "params": dict(run.data.params), "tags": dict(run.data.tags), "created": mv.creation_timestamp}


@lru_cache(maxsize=16)
def _reference_cached(tracking_uri: str, model: str, version: int) -> pd.DataFrame:
    c = init()
    mv = c.get_model_version(model, str(version))
    path = c.download_artifacts(mv.run_id, "reference.csv")
    return pd.read_csv(path)


def reference_df(version: int) -> pd.DataFrame:
    cfg = settings()
    return _reference_cached(cfg.tracking_uri, cfg.model_name, int(version))


def list_versions() -> pd.DataFrame:
    c = init()
    cfg = settings()
    rows = []
    aliases: dict[str, list[str]] = {}
    for a in (CHAMPION, CHALLENGER):
        v = alias_version(a)
        if v:
            aliases.setdefault(str(v), []).append(a)
    try:
        versions = c.search_model_versions(f"name='{cfg.model_name}'")
    except Exception:
        versions = []
    for mv in versions:
        run = c.get_run(mv.run_id)
        rows.append({"version": int(mv.version), "aliases": ",".join(aliases.get(str(mv.version), [])),
                     "val_mae": run.data.metrics.get("val_mae"), "reason": run.data.tags.get("reason", ""),
                     "git_sha": run.data.tags.get("git_sha", ""), "created": pd.to_datetime(mv.creation_timestamp, unit="ms")})
    df = pd.DataFrame(rows)
    return df.sort_values("version", ascending=False).reset_index(drop=True) if not df.empty else df
