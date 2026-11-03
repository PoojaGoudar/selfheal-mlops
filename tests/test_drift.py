import numpy as np
import pandas as pd

from selfheal.monitoring import drift


def test_psi_same_distribution_is_small():
    r = np.random.default_rng(0)
    assert drift.psi(r.normal(0, 1, 2000), r.normal(0, 1, 500)) < 0.1


def test_psi_shifted_distribution_is_large():
    r = np.random.default_rng(0)
    assert drift.psi(r.normal(0, 1, 2000), r.normal(3, 1, 500)) > 0.25


def test_flat_feature_detected():
    r = np.random.default_rng(0)
    ref = pd.DataFrame({f: r.normal(5, 1, 500) for f in drift.DRIFT_FEATURES})
    cur = pd.DataFrame({f: r.normal(5, 1, 50) for f in drift.DRIFT_FEATURES})
    cur["wind"] = 0.0
    rep = drift.feature_report(ref, cur)
    assert drift.flat_features(rep) == ["wind"]
    assert drift.assess(rep, 0.25, 3)["drift_alert"] is False or "wind" in drift.assess(rep, 0.25, 3)["drifted_features"]
