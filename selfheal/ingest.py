"""DVC stage 1: produce the raw dataset CSV (synthetic by default, live Open-Meteo optionally)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from selfheal.data import synthetic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/air.csv")
    ap.add_argument("--params", default="params.yaml")
    a = ap.parse_args()
    p = yaml.safe_load(Path(a.params).read_text())["ingest"]
    if p["source"] == "live":
        from selfheal.data.openmeteo import fetch_history
        df = fetch_history(p["lat"], p["lon"], past_days=p["days"])
    else:
        df = synthetic.generate(24 * p["days"], pd.Timestamp(p["start"]), seed=p["seed"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
