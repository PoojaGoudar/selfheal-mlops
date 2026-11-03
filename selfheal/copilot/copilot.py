"""LLM incident copilot (Claude API with tool use). ADVISORY ONLY: tools are read-only.

Works without an API key via a deterministic rule-based fallback, so demos never break offline.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from selfheal import registry, store
from selfheal.config import settings

PROMPT_VERSION = "incident_copilot_v1"
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / f"{PROMPT_VERSION}.md"

TOOLS = [
    {"name": "get_drift_report", "description": "Per-feature PSI / KS drift statistics comparing the recent serving window with the champion model's reference data, plus the monitor's assessment.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_error_breakdown", "description": "Live forecast error of the champion model: overall MAE, MAE in the older vs newer half of the recent window, and mean signed bias (prediction - truth).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_recent_runs", "description": "The most recent registered model versions with validation MAE, aliases and why they were trained.",
     "input_schema": {"type": "object", "properties": {}}},
]


# ------------------------------------------------------------------ tools --
def _tool(name: str, ctx: dict) -> dict:
    if name == "get_drift_report":
        return {"drift": ctx.get("drift"), "assessment": ctx.get("assessment")}
    if name == "get_error_breakdown":
        df = store.recent_predictions(400, labeled_only=True)
        df = df[df["champion_version"] == ctx.get("champion_version")]
        if df.empty:
            return {"note": "no labeled predictions yet"}
        err = (df["champion_pred"] - df["y_true"])
        half = len(df) // 2
        return {"n": len(df), "mae": round(float(err.abs().mean()), 3), "bias_pred_minus_true": round(float(err.mean()), 3),
                "mae_older_half": round(float(err.abs().iloc[:half].mean()), 3) if half else None,
                "mae_newer_half": round(float(err.abs().iloc[half:].mean()), 3), "baseline_val_mae": ctx.get("baseline_mae")}
    if name == "get_recent_runs":
        v = registry.list_versions()
        return {"versions": json.loads(v.head(5).to_json(orient="records", date_format="iso"))} if not v.empty else {"versions": []}
    return {"error": f"unknown tool {name}"}


# --------------------------------------------------------------- fallback --
def rule_based(report: dict) -> dict:
    a, d = report.get("assessment", {}), report.get("drift", {})
    flat, drifted = a.get("flat_features", []), a.get("drifted_features", [])
    ratio = report.get("mae_ratio")
    if flat:
        f = flat[0]
        return {"summary": f"Feature '{f}' has gone constant (std {d[f]['cur_std']} vs {d[f]['ref_std']} in reference): this looks like a broken sensor or pipeline, not a real change in the world.",
                "probable_cause": f"Data-quality failure on '{f}' (stuck/flat values).", "confidence": "high",
                "evidence": [f"{f}: reference mean {d[f]['ref_mean']}, current mean {d[f]['cur_mean']}, current std {d[f]['cur_std']}"],
                "recommended_action": f"Fix the '{f}' sensor / ingestion. Do NOT retrain: the model would learn the broken values.",
                "retrain_recommended": False, "source": "rule_based"}
    ev = [f"{f}: PSI {d[f]['psi']} (mean {d[f]['ref_mean']} -> {d[f]['cur_mean']})" for f in drifted[:4]]
    if ratio:
        ev.append(f"Rolling MAE {report.get('rolling_mae'):.2f} is {ratio:.1f}x the validation baseline {report.get('baseline_mae'):.2f}")
    return {"summary": f"{len(drifted)} feature(s) drifted ({', '.join(drifted[:3]) or 'none above threshold'}) and forecast error is {'elevated' if ratio and ratio > 1.5 else 'normal'}.",
            "probable_cause": "Input distribution and/or the relationship between inputs and PM2.5 changed (data + concept drift).",
            "confidence": "medium", "evidence": ev,
            "recommended_action": "Retrain on the recent window, evaluate on a recent hold-out, then shadow -> canary -> human approval.",
            "retrain_recommended": True, "source": "rule_based"}


# -------------------------------------------------------------------- LLM --
def _extract_json(text: str) -> dict:
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("no JSON object in model output")
    return json.loads(text[a:b + 1])


def _llm(report: dict, client=None, max_turns: int = 6) -> dict:
    cfg = settings()
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    system = PROMPT_PATH.read_text()
    msgs = [{"role": "user", "content": "Incident raised by the monitor:\n" + json.dumps(
        {k: report.get(k) for k in ("kind", "champion_version", "baseline_mae", "rolling_mae", "mae_ratio", "n_labeled")}, default=str)
        + "\nInvestigate with the tools and give your diagnosis as the JSON object described."}]
    t0, in_tok, out_tok = time.perf_counter(), 0, 0
    try:
        for _ in range(max_turns):
            resp = client.messages.create(model=cfg.anthropic_model, max_tokens=1500, system=system, tools=TOOLS, messages=msgs)
            in_tok += resp.usage.input_tokens
            out_tok += resp.usage.output_tokens
            if resp.stop_reason == "tool_use":
                msgs.append({"role": "assistant", "content": resp.content})
                results = [{"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(_tool(b.name, report), default=str)}
                           for b in resp.content if b.type == "tool_use"]
                msgs.append({"role": "user", "content": results})
                continue
            text = "".join(b.text for b in resp.content if b.type == "text")
            out = _extract_json(text)
            out["source"] = f"llm:{cfg.anthropic_model}"
            cost = in_tok * cfg.llm_price_in / 1e6 + out_tok * cfg.llm_price_out / 1e6
            store.log_llm_call(cfg.anthropic_model, PROMPT_VERSION, in_tok, out_tok, (time.perf_counter() - t0) * 1000, cost, True)
            return out
        raise RuntimeError("copilot exceeded max tool turns")
    except Exception:
        store.log_llm_call(cfg.anthropic_model, PROMPT_VERSION, in_tok, out_tok, (time.perf_counter() - t0) * 1000, 0.0, False)
        raise


def diagnose(report: dict, client=None) -> dict:
    """Return a structured diagnosis. Never raises: falls back to rules on any LLM problem."""
    cfg = settings()
    if not (cfg.anthropic_api_key or client):
        return rule_based(report)
    try:
        return _llm(report, client=client)
    except Exception as e:
        out = rule_based(report)
        out["source"] = "rule_based_fallback"
        out["llm_error"] = f"{type(e).__name__}: {e}"[:300]
        return out
