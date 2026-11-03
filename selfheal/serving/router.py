"""Traffic routing for shadow / canary rollouts. Deterministic: same request id -> same bucket."""
from __future__ import annotations

import hashlib


def bucket(request_id: str) -> int:
    return int(hashlib.md5(request_id.encode()).hexdigest(), 16) % 100


def route(request_id: str, stage: str, canary_pct: int, has_challenger: bool) -> str:
    """Return which model SERVES the request: 'champion' or 'challenger'.
    shadow stage: challenger is only evaluated silently; canary stage: canary_pct% of users get it."""
    if has_challenger and stage == "canary" and bucket(request_id) < canary_pct:
        return "challenger"
    return "champion"
