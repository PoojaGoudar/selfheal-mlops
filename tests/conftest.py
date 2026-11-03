
import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SELFHEAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MODEL_REFRESH_S", "0")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    from selfheal import registry
    registry._reference_cached.cache_clear()
    return tmp_path
