import json
from types import SimpleNamespace as NS

from selfheal import store
from selfheal.copilot import copilot

REPORT = {"kind": "model_drift", "champion_version": 1, "baseline_mae": 2.0, "rolling_mae": 30.0, "mae_ratio": 15.0, "n_labeled": 48,
          "drift": {"wind": {"psi": 1.2, "ks_p": 0.0, "ref_mean": 2.2, "cur_mean": 2.3, "ref_std": 0.9, "cur_std": 0.9}},
          "assessment": {"drifted_features": ["wind"], "flat_features": [], "n_drifted": 1, "drift_alert": True, "max_psi": 1.2}}


def test_rule_based_without_key(env):
    out = copilot.diagnose(REPORT)
    assert out["source"] == "rule_based" and out["retrain_recommended"] is True


def test_rule_based_flat_feature_says_do_not_retrain():
    r = json.loads(json.dumps(REPORT))
    r["assessment"]["flat_features"] = ["wind"]
    r["drift"]["wind"].update(cur_std=0.0)
    out = copilot.rule_based(r)
    assert out["retrain_recommended"] is False and "Do NOT retrain" in out["recommended_action"]


class FakeClient:
    """Mimics anthropic.Anthropic: first answer asks for a tool, second answer is the final JSON."""
    def __init__(self):
        self.calls = 0
        self.messages = NS(create=self.create)

    def create(self, **kw):
        self.calls += 1
        usage = NS(input_tokens=100, output_tokens=50)
        if self.calls == 1:
            return NS(stop_reason="tool_use", usage=usage, content=[NS(type="tool_use", id="t1", name="get_drift_report", input={})])
        final = {"summary": "s", "probable_cause": "c", "confidence": "high", "evidence": ["e"], "recommended_action": "a", "retrain_recommended": True}
        return NS(stop_reason="end_turn", usage=usage, content=[NS(type="text", text="Here you go: " + json.dumps(final))])


def test_llm_tool_loop_and_cost_logging(env):
    store.init_db()
    out = copilot.diagnose(REPORT, client=FakeClient())
    assert out["source"].startswith("llm:") and out["confidence"] == "high"
    calls = store.read_llm_calls(5)
    assert len(calls) == 1 and int(calls.iloc[0]["input_tokens"]) == 200 and int(calls.iloc[0]["ok"]) == 1


def test_llm_failure_falls_back(env):
    store.init_db()

    class Broken:
        messages = NS(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))

    out = copilot.diagnose(REPORT, client=Broken())
    assert out["source"] == "rule_based_fallback" and "boom" in out["llm_error"]
