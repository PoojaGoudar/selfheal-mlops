import numpy as np

from selfheal.healing.pipeline import evaluate_gate
from selfheal.serving.router import bucket, route


def test_router_is_deterministic_and_respects_percentage():
    ids = [f"req-{i}" for i in range(4000)]
    assert [bucket(i) for i in ids] == [bucket(i) for i in ids]
    share = np.mean([route(i, "canary", 25, True) == "challenger" for i in ids])
    assert 0.22 < share < 0.28


def test_router_never_serves_challenger_in_shadow_or_without_one():
    assert all(route(f"r{i}", "shadow", 100, True) == "champion" for i in range(200))
    assert all(route(f"r{i}", "canary", 100, False) == "champion" for i in range(200))


def test_gate_pass_and_fail():
    y = np.array([10.0, 20, 30, 40])
    assert evaluate_gate(y + 5, y + 1, y, 0.05)["passed"] is True
    assert evaluate_gate(y + 1, y + 5, y, 0.05)["passed"] is False
    assert evaluate_gate(y + 1, y + 0.98, y, 0.05)["passed"] is False  # better, but not by enough
