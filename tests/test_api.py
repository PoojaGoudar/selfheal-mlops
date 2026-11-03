import pytest
from fastapi.testclient import TestClient

from selfheal import bootstrap, registry, store
from selfheal.data.synthetic import generate


@pytest.fixture()
def client(env):
    bootstrap.run(days=30)
    from selfheal.serving.app import app
    return TestClient(app)


def _payload(request_id="r1"):
    df = generate(48, "2026-03-01")
    last = df.iloc[-1]
    return {"request_id": request_id, "ts": str(last["ts"]), "pm25_history": df["pm2_5"].tolist(),
            **{k: float(last[k]) for k in ["temp", "humidity", "wind", "pressure", "precip"]}}


def test_health_predict_feedback(client):
    assert client.get("/health").json()["champion"] == 1
    r = client.post("/predict", json=_payload()).json()
    assert 0 < r["prediction_pm25_next_hour"] < 200 and r["served_by"] == "champion"
    assert client.post("/feedback", json={"request_id": "r1", "y_true": 33.0}).json() == {"ok": True}
    assert client.post("/feedback", json={"request_id": "nope", "y_true": 1}).status_code == 404
    st = client.get("/stats").json()
    assert st["requests_last_500"] >= 1 and st["latency_ms_p95"] > 0


def test_short_history_is_422(client):
    p = _payload()
    p["pm25_history"] = p["pm25_history"][:5]
    assert client.post("/predict", json=p).status_code == 422


def test_alias_move_changes_served_model(client):
    """Promotion = moving an alias. The API must pick it up without restarting."""
    from selfheal.healing.pipeline import start_shadow
    from selfheal.train import train_and_register
    obs = store.read_observations()
    v2 = train_and_register(obs, reason="test", holdout_hours=48, half_life_h=24, reference_rows=100)["version"]
    start_shadow(v2)
    store.set_control("rollout_stage", "canary")
    store.set_control("canary_pct", 100)
    r = client.post("/predict", json=_payload("r2")).json()
    assert r["served_by"] == "challenger" and r["model_version"] == v2
    registry.set_alias("champion", v2)
    registry.delete_alias("challenger")
    store.set_control("rollout_stage", "none")
    assert client.post("/predict", json=_payload("r3")).json()["model_version"] == v2
