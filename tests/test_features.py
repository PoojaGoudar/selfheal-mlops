import pytest

from selfheal.data.synthetic import generate
from selfheal.data.validate import validate
from selfheal.features import FEATURES, WEATHER, features_from_request, make_features


def test_train_serve_parity():
    """The serving feature path must equal the training path (no train/serve skew)."""
    df = generate(24 * 10, "2026-01-01")
    feat = make_features(df)
    row = feat.iloc[100]
    j = df.index[df["ts"] == row["ts"]][0]
    req = features_from_request(df["pm2_5"].iloc[j - 23:j + 1].tolist(), df.iloc[j][WEATHER].to_dict(), row["ts"])
    assert max(abs(req[k] - row[k]) for k in FEATURES) < 1e-9


def test_target_is_next_hour():
    df = generate(100, "2026-01-01")
    f = make_features(df)
    j = df.index[df["ts"] == f.iloc[0]["ts"]][0]
    assert f.iloc[0]["target"] == pytest.approx(df["pm2_5"].iloc[j + 1])


def test_short_history_rejected():
    with pytest.raises(ValueError):
        features_from_request([1.0] * 10, {k: 1.0 for k in WEATHER}, "2026-01-01 00:00")


def test_validation_catches_bad_data():
    import pandera
    df = generate(48, "2026-01-01")
    validate(df)
    df.loc[3, "humidity"] = 140
    with pytest.raises(pandera.errors.SchemaError):
        validate(df)
