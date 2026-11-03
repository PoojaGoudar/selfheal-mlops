"""The 'healing' actions: retrain a candidate, gate it, roll it out, promote or roll back, and document it."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from selfheal import registry, store
from selfheal.config import settings
from selfheal.features import FEATURES, TARGET
from selfheal.train import mae, train_and_register


def evaluate_gate(champion_pred, candidate_pred, y_true, min_improvement: float) -> dict:
    """Offline gate: candidate must beat the champion on the SAME recent hold-out by at least min_improvement."""
    mc, mn = mae(y_true, champion_pred), mae(y_true, candidate_pred)
    improvement = 1 - mn / max(mc, 1e-9)
    return {"champion_mae": round(mc, 3), "candidate_mae": round(mn, 3), "improvement": round(float(improvement), 3),
            "min_improvement": min_improvement, "passed": bool(improvement >= min_improvement)}


def retrain_candidate(incident_id: int) -> dict:
    cfg = settings()
    obs = store.read_observations(cfg.retrain_window_hours)
    res = train_and_register(obs, reason=f"retrain-incident-{incident_id}", holdout_hours=cfg.holdout_hours,
                             half_life_h=cfg.half_life_hours, reference_rows=120, extra_tags={"incident": str(incident_id)})
    champ_v = registry.alias_version(registry.CHAMPION)
    hold = res["holdout"]
    gate = evaluate_gate(registry.load_model(champ_v).predict(hold[FEATURES]), res["model"].predict(hold[FEATURES]),
                         hold[TARGET], cfg.min_improvement)
    gate.update(champion_version=champ_v, candidate_version=res["version"], holdout_rows=len(hold))
    store.audit("pipeline", "candidate_trained", {"incident": incident_id, **gate})
    return {"version": res["version"], "gate": gate}


def compare_live(df) -> dict:
    """Live comparison on rows where BOTH models predicted and the truth is known."""
    d = df.dropna(subset=["challenger_pred", "y_true"])
    if d.empty:
        return {"n": 0}
    mc = float(np.mean(np.abs(d["champion_pred"] - d["y_true"])))
    mn = float(np.mean(np.abs(d["challenger_pred"] - d["y_true"])))
    return {"n": len(d), "champion_mae": round(mc, 3), "challenger_mae": round(mn, 3), "improvement": round(1 - mn / max(mc, 1e-9), 3)}


def start_shadow(version: int) -> None:
    registry.set_alias(registry.CHALLENGER, version)
    store.set_control("rollout_stage", "shadow")
    store.set_control("rollout_started_at", store.now_iso())
    store.audit("pipeline", "shadow_started", {"challenger": version})


def start_canary() -> None:
    store.set_control("rollout_stage", "canary")
    store.set_control("rollout_started_at", store.now_iso())
    store.audit("pipeline", "canary_started", {"canary_pct": store.get_control("canary_pct", settings().canary_pct)})


def rollback(reason: str, actor: str = "pipeline") -> None:
    v = registry.alias_version(registry.CHALLENGER)
    registry.delete_alias(registry.CHALLENGER)
    store.set_control("rollout_stage", "none")
    store.audit(actor, "rolled_back", {"challenger": v, "reason": reason})


def _reanchor_reference(version: int) -> int:
    """After a regime change the new 'normal' is what the challenger just proved itself on (shadow + canary).
    Re-anchor the drift baseline of the promoted model to that live traffic window."""
    import tempfile

    cfg = settings()
    feats = store.recent_features(cfg.drift_window)
    if len(feats) < 48:
        return 0
    c = registry.init()
    run_id = c.get_model_version(cfg.model_name, str(version)).run_id
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "reference.csv"
        feats[FEATURES].to_csv(p, index=False)
        c.log_artifact(run_id, str(p))
    registry._reference_cached.cache_clear()
    return len(feats)


def promote(incident: dict, actor: str) -> int:
    v = registry.alias_version(registry.CHALLENGER)
    if v is None:
        raise RuntimeError("no challenger to promote")
    n_ref = _reanchor_reference(v)
    store.audit("pipeline", "drift_baseline_reanchored", {"version": v, "rows": n_ref})
    old = registry.alias_version(registry.CHAMPION)
    registry.set_alias(registry.CHAMPION, v)
    registry.delete_alias(registry.CHALLENGER)
    store.set_control("rollout_stage", "none")
    store.set_control("cooldown_since", store.now_iso())
    store.audit(actor, "promoted", {"new_champion": v, "old_champion": old, "incident": incident["id"]})
    _model_card(v, old, incident, actor)
    return v


def _model_card(version: int, old: int | None, incident: dict, actor: str) -> Path:
    info = registry.version_info(version)
    out = Path(settings().data_dir) / "model_cards"
    out.mkdir(parents=True, exist_ok=True)
    gate = incident.get("gate") or {}
    text = f"""# Model card: aqi-forecaster v{version}

| Field | Value |
|---|---|
| Task | Next-hour PM2.5 forecast |
| Promoted at | {store.now_iso()} |
| Replaced | v{old} |
| Approved by | {actor} |
| Trigger | incident #{incident['id']} ({incident.get('kind')}) |
| Code version | git `{info['tags'].get('git_sha', 'unknown')}` |
| Data window | {info['tags'].get('window_start')} -> {info['tags'].get('window_end')} (hash `{info['tags'].get('data_hash')}`) |
| MLflow run | `{info['run_id']}` |
| Hold-out MAE | {info['metrics'].get('val_mae', float('nan')):.2f} (persistence baseline {info['metrics'].get('persistence_mae', float('nan')):.2f}) |
| Offline gate | {gate} |

## Intended use
Demo-grade forecasting for operational dashboards. Not for health or safety decisions.

## Known limitations
Trained on a recent window only; may underperform on regimes not yet observed. Monitored continuously; see audit log.
"""
    p = out / f"model_v{version}.md"
    p.write_text(text)
    return p
