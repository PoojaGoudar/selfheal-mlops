"""Drift statistics implemented by hand so every number can be explained in a viva.

PSI (Population Stability Index): bin the reference data into quantile bins, compare the share of
current rows per bin:  PSI = sum((cur% - ref%) * ln(cur% / ref%)).
Rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from selfheal.features import DRIFT_FEATURES

EPS = 1e-4


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 5) -> float:
    ref, cur = np.asarray(ref, float), np.asarray(cur, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (almost) constant reference
        return 0.0 if np.allclose(cur, ref.mean(), atol=1e-6) else 10.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_report(ref: pd.DataFrame, cur: pd.DataFrame, features=DRIFT_FEATURES) -> dict:
    out = {}
    for f in features:
        r, c = ref[f].dropna().to_numpy(), cur[f].dropna().to_numpy()
        out[f] = {"psi": round(psi(r, c), 4), "ks_p": float(ks_2samp(r, c).pvalue),
                  "ref_mean": round(float(r.mean()), 3), "cur_mean": round(float(c.mean()), 3),
                  "ref_std": round(float(r.std()), 4), "cur_std": round(float(c.std()), 4)}
    return out


def flat_features(report: dict) -> list[str]:
    """Features that went constant while the reference varied: classic broken-sensor / broken-pipeline signature."""
    return [f for f, v in report.items() if v["ref_std"] > 1e-3 and v["cur_std"] < 0.02 * v["ref_std"]]


def assess(report: dict, psi_threshold: float, psi_min_features: int) -> dict:
    drifted = sorted([f for f, v in report.items() if v["psi"] > psi_threshold], key=lambda f: -report[f]["psi"])
    return {"drifted_features": drifted, "n_drifted": len(drifted), "drift_alert": len(drifted) >= psi_min_features,
            "flat_features": flat_features(report), "max_psi": max((v["psi"] for v in report.values()), default=0.0)}
