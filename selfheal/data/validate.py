"""Data validation (pandera). Used by the DVC pipeline, bootstrap and unit tests."""
from __future__ import annotations

import sys

import pandas as pd

try:  # pandera >= 0.24
    import pandera.pandas as pa
except ImportError:  # older versions
    import pandera as pa

SCHEMA = pa.DataFrameSchema(
    {
        "ts": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "pm2_5": pa.Column(float, pa.Check.in_range(0, 1000), coerce=True),
        "temp": pa.Column(float, pa.Check.in_range(-30, 60), coerce=True),
        "humidity": pa.Column(float, pa.Check.in_range(0, 100), coerce=True),
        "wind": pa.Column(float, pa.Check.ge(0), coerce=True),
        "pressure": pa.Column(float, pa.Check.in_range(300, 1100), coerce=True),
        "precip": pa.Column(float, pa.Check.ge(0), coerce=True),
    },
    unique=["ts"],
    strict=False,
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"]).astype("datetime64[ns]")
    return SCHEMA.validate(d)


if __name__ == "__main__":
    path = sys.argv[1]
    frame = pd.read_csv(path, parse_dates=["ts"])
    validate(frame)
    print(f"OK: {path} passed validation ({len(frame)} rows)")
