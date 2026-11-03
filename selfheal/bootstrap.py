"""One-time setup: create data, validate it, train v1 and set it as the champion."""
from __future__ import annotations

import argparse
import shutil

import pandas as pd

from selfheal import registry, store
from selfheal.config import settings
from selfheal.data import synthetic
from selfheal.data.validate import validate
from selfheal.train import train_and_register


def run(days: int = 120, source: str = "synthetic", reset: bool = False, seed: int = 42) -> int:
    cfg = settings()
    if reset and cfg.data_dir.exists():
        shutil.rmtree(cfg.data_dir)
    store.init_db()
    registry.init()
    if registry.alias_version(registry.CHAMPION) is not None:
        print("champion already exists - nothing to do (use --reset to start over)")
        return registry.alias_version(registry.CHAMPION)

    if source == "live":
        from selfheal.data.openmeteo import fetch_history
        full = fetch_history(cfg.lat, cfg.lon, past_days=90)
        if len(full) < 24 * 40:
            raise SystemExit("live data too short; use --source synthetic")
        cut = len(full) - 24 * 25  # last 25 days are kept aside and replayed as the 'live' stream
        df, replay = full.iloc[:cut], full.iloc[cut:]
        replay.to_csv(cfg.data_dir / "replay.csv", index=False)
        print(f"live data: {len(df)} training hours, {len(replay)} replay hours (set SIM_SOURCE=replay to use them)")
    else:
        start = pd.Timestamp.now().floor("h") - pd.Timedelta(hours=24 * days)
        df = synthetic.generate(24 * days, start, seed=seed)

    validate(df)
    (cfg.data_dir / "raw").mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.data_dir / "raw" / "air.csv", index=False)
    store.insert_observations(df)
    res = train_and_register(df, reason="bootstrap", holdout_hours=168, half_life_h=None, reference_rows=1500)
    registry.set_alias(registry.CHAMPION, res["version"])
    store.set_control("regime", "normal")
    store.set_control("rollout_stage", "none")
    store.audit("bootstrap", "initial_champion", {"version": res["version"], "val_mae": res["metrics"]["val_mae"], "source": source})
    print(f"champion = v{res['version']}  val_mae={res['metrics']['val_mae']:.2f}  persistence_mae={res['metrics']['persistence_mae']:.2f}")
    return res["version"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--source", choices=["synthetic", "live"], default="synthetic")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    run(a.days, a.source, a.reset)


if __name__ == "__main__":
    main()
