"""Tiny SQLite persistence layer shared by API, simulator, monitor and dashboard.

SQLite (WAL mode) keeps the whole platform runnable on a laptop with zero extra
services. Swap for Postgres later by re-implementing this one module.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime

import pandas as pd

from selfheal.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations(
  ts TEXT PRIMARY KEY, pm2_5 REAL, pm10 REAL, no2 REAL, o3 REAL,
  temp REAL, humidity REAL, wind REAL, pressure REAL, precip REAL, regime TEXT);
CREATE TABLE IF NOT EXISTS predictions(
  request_id TEXT PRIMARY KEY, created_at TEXT, target_ts TEXT, features TEXT,
  champion_version INTEGER, champion_pred REAL,
  challenger_version INTEGER, challenger_pred REAL,
  served_by TEXT, served_pred REAL, y_true REAL, latency_ms REAL);
CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_at);
CREATE TABLE IF NOT EXISTS control(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS incidents(
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, updated_at TEXT,
  kind TEXT, status TEXT, report TEXT, diagnosis TEXT,
  candidate_version INTEGER, gate TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, actor TEXT, action TEXT, details TEXT);
CREATE TABLE IF NOT EXISTS llm_calls(
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, model TEXT, prompt_version TEXT,
  input_tokens INTEGER, output_tokens INTEGER, latency_ms REAL, cost_usd REAL, ok INTEGER);
"""

OBS_COLS = ["ts", "pm2_5", "pm10", "no2", "o3", "temp", "humidity", "wind", "pressure", "precip", "regime"]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


@contextmanager
def conn():
    cfg = settings()
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(cfg.db_path, timeout=30)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    with conn() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)


# ---------------------------------------------------------------- control --
def set_control(key: str, value) -> None:
    with conn() as c:
        c.execute("INSERT INTO control(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, json.dumps(value)))


def get_control(key: str, default=None):
    with conn() as c:
        row = c.execute("SELECT value FROM control WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def all_control() -> dict:
    with conn() as c:
        return {r["key"]: json.loads(r["value"]) for r in c.execute("SELECT key,value FROM control")}


# ----------------------------------------------------------- observations --
def insert_observation(row: dict) -> None:
    insert_observations(pd.DataFrame([row]))


def insert_observations(df: pd.DataFrame) -> None:
    d = df.copy()
    if "regime" not in d:
        d["regime"] = "normal"
    d["ts"] = pd.to_datetime(d["ts"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    rows = [tuple(r) for r in d[OBS_COLS].itertuples(index=False, name=None)]
    with conn() as c:
        c.executemany(f"INSERT OR REPLACE INTO observations({','.join(OBS_COLS)}) VALUES({','.join('?' * len(OBS_COLS))})", rows)


def read_observations(last_hours: int | None = None) -> pd.DataFrame:
    q = "SELECT * FROM observations ORDER BY ts"
    if last_hours:
        q = f"SELECT * FROM (SELECT * FROM observations ORDER BY ts DESC LIMIT {int(last_hours)}) ORDER BY ts"
    with conn() as c:
        df = pd.read_sql_query(q, c)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def last_observation_ts():
    with conn() as c:
        row = c.execute("SELECT MAX(ts) AS m FROM observations").fetchone()
    return pd.to_datetime(row["m"]) if row and row["m"] else None


# ------------------------------------------------------------ predictions --
def insert_prediction(rec: dict) -> None:
    cols = ["request_id", "created_at", "target_ts", "features", "champion_version", "champion_pred",
            "challenger_version", "challenger_pred", "served_by", "served_pred", "latency_ms"]
    with conn() as c:
        c.execute(f"INSERT OR REPLACE INTO predictions({','.join(cols)}) VALUES({','.join('?' * len(cols))})",
                  [rec.get(k) for k in cols])


def record_feedback(request_id: str, y_true: float) -> bool:
    with conn() as c:
        cur = c.execute("UPDATE predictions SET y_true=? WHERE request_id=?", (float(y_true), request_id))
        return cur.rowcount > 0


def recent_predictions(n: int, labeled_only: bool = False, since: str | None = None) -> pd.DataFrame:
    where = []
    if labeled_only:
        where.append("y_true IS NOT NULL")
    if since:
        where.append(f"created_at >= '{since}'")
    w = ("WHERE " + " AND ".join(where)) if where else ""
    with conn() as c:
        df = pd.read_sql_query(
            f"SELECT * FROM (SELECT * FROM predictions {w} ORDER BY created_at DESC LIMIT {int(n)}) ORDER BY created_at", c)
    return df


def recent_features(n: int) -> pd.DataFrame:
    df = recent_predictions(n)
    if df.empty:
        return pd.DataFrame()
    return pd.DataFrame([json.loads(f) for f in df["features"]])


def count_labeled_since(since: str, served_by: str | None = None) -> int:
    q, p = "SELECT COUNT(*) AS n FROM predictions WHERE y_true IS NOT NULL AND created_at >= ?", [since]
    if served_by:
        q += " AND served_by=?"
        p.append(served_by)
    with conn() as c:
        return int(c.execute(q, p).fetchone()["n"])


# -------------------------------------------------------------- incidents --
def create_incident(kind: str, report: dict) -> int:
    ts = now_iso()
    with conn() as c:
        cur = c.execute("INSERT INTO incidents(created_at,updated_at,kind,status,report,notes) VALUES(?,?,?,?,?,?)",
                        (ts, ts, kind, "detected", json.dumps(report), json.dumps([])))
        return int(cur.lastrowid)


_JSON_FIELDS = {"report", "diagnosis", "gate", "notes"}


def update_incident(incident_id: int, **fields) -> None:
    fields["updated_at"] = now_iso()
    sets, vals = [], []
    for k, v in fields.items():
        sets.append(f"{k}=?")
        vals.append(json.dumps(v) if k in _JSON_FIELDS else v)
    with conn() as c:
        c.execute(f"UPDATE incidents SET {','.join(sets)} WHERE id=?", [*vals, incident_id])


def add_note(incident_id: int, text: str) -> None:
    inc = get_incident(incident_id)
    notes = (inc.get("notes") or []) + [{"ts": now_iso(), "text": text}]
    update_incident(incident_id, notes=notes)


def _inc(row) -> dict:
    d = dict(row)
    for k in _JSON_FIELDS:
        if d.get(k):
            d[k] = json.loads(d[k])
    return d


def get_incident(incident_id: int) -> dict:
    with conn() as c:
        return _inc(c.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone())


CLOSED = ("resolved", "dismissed")


def open_incident() -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM incidents WHERE status NOT IN ('resolved','dismissed') ORDER BY id DESC LIMIT 1").fetchone()
    return _inc(row) if row else None


def list_incidents(n: int = 20) -> list[dict]:
    with conn() as c:
        return [_inc(r) for r in c.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (n,))]


# ------------------------------------------------------------ audit / llm --
def audit(actor: str, action: str, details: dict | None = None) -> None:
    with conn() as c:
        c.execute("INSERT INTO audit(ts,actor,action,details) VALUES(?,?,?,?)",
                  (now_iso(), actor, action, json.dumps(details or {}, default=str)))


def read_audit(n: int = 50) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql_query("SELECT ts,actor,action,details FROM audit ORDER BY id DESC LIMIT ?", c, params=(n,))


def log_llm_call(model, prompt_version, in_tok, out_tok, latency_ms, cost, ok) -> None:
    with conn() as c:
        c.execute("INSERT INTO llm_calls(ts,model,prompt_version,input_tokens,output_tokens,latency_ms,cost_usd,ok) VALUES(?,?,?,?,?,?,?,?)",
                  (now_iso(), model, prompt_version, in_tok, out_tok, latency_ms, cost, int(ok)))


def read_llm_calls(n: int = 20) -> pd.DataFrame:
    with conn() as c:
        return pd.read_sql_query("SELECT * FROM llm_calls ORDER BY id DESC LIMIT ?", c, params=(n,))
