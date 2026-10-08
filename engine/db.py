"""SQLite storage for the discovery engine.

Every stage writes to two places: this database (for the pipeline) and a CSV in
data/ (so you can open it). Section 6 of AGENTS.md defines the first four tables;
the rest are plumbing that AGENTS.md requires: the LLM cache, the call log, the
run log and the error file.

Run `python run.py doctor` to create the database.
"""

import csv
import hashlib
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "engine.db"


def label_id(post_id: str) -> str:
    """Opaque key for one row of a hand-labelling file.

    Derived from post_id so it survives a re-export of the sample, and hashed so the
    blind file cannot be read as a source name: post_ids start with the source.
    """
    return hashlib.sha1(post_id.encode("utf-8")).hexdigest()[:10]

SCHEMA = """
-- Section 6: raw_posts. One row per collected post, deduplicated on post_id.
CREATE TABLE IF NOT EXISTS raw_posts (
    post_id         TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    myntra_explicit INTEGER DEFAULT 0,
    parent_context  TEXT,
    author_hash     TEXT,
    date            TEXT,
    text            TEXT NOT NULL,
    rating          REAL,
    engagement      INTEGER,
    url             TEXT,
    lang_hint       TEXT,
    collected_at    TEXT,
    raw_file        TEXT
);

-- Module 2 step 1: what survived cleaning, with the reason for every removal kept
-- separately in data/exports/clean_dropped.csv.
CREATE TABLE IF NOT EXISTS clean_posts (
    post_id         TEXT PRIMARY KEY REFERENCES raw_posts(post_id),
    source          TEXT,
    date            TEXT,
    myntra_explicit INTEGER,
    parent_context  TEXT,
    text_clean      TEXT,
    text_en         TEXT,
    is_relevant     INTEGER,
    -- Written by the relevance stage. They live here rather than only in
    -- relevant_posts so a re-judgement can overwrite them without touching that table.
    relevance_reason     TEXT,
    relevance_confidence REAL,
    model                TEXT,
    judged_by            TEXT,
    judged_at            TEXT,
    -- A second opinion, kept beside the final judgement rather than replacing it.
    -- Populated when a post was judged by groq and then re-judged by gemini, so the
    -- cross-model comparison survives and the final label is still single-judge.
    groq_is_relevant     INTEGER,
    groq_reason          TEXT,
    groq_confidence      REAL,
    groq_model           TEXT,
    groq_judged_at       TEXT,
    engagement      INTEGER,
    rating          REAL,
    lang_hint       TEXT,
    url             TEXT,
    cleaned_at      TEXT
);

-- Embedding cache, keyed by post_id, so near-duplicate detection never re-embeds
-- text it has already seen. Vectors are stored as comma-separated floats.
CREATE TABLE IF NOT EXISTS embeddings (
    key       TEXT PRIMARY KEY,
    vector    TEXT,
    text_hash TEXT
);

-- Section 6: relevant_posts. raw_posts plus the clean, translated and judged fields.
CREATE TABLE IF NOT EXISTS relevant_posts (
    post_id              TEXT PRIMARY KEY REFERENCES raw_posts(post_id),
    text_clean           TEXT,
    text_en              TEXT,
    is_relevant          INTEGER,
    relevance_reason     TEXT,
    relevance_confidence REAL,
    model                TEXT,
    judged_by            TEXT,   -- which provider answered: gemini or groq
    judged_at            TEXT
);

-- Section 6: tagged_posts. One row per post, matching config/schema.yaml.
-- segment_cues from the spec is stored as separate columns so they are easy to filter.
-- in_scope is the first field of the schema and the second stage of the relevance
-- filter: it is decided by the locked rubric in docs/labelling_guide.md. Rows with
-- in_scope = 0 stay in this table and are excluded from every analysis table.
CREATE TABLE IF NOT EXISTS tagged_posts (
    post_id                 TEXT PRIMARY KEY REFERENCES relevant_posts(post_id),
    in_scope                INTEGER,
    in_scope_reason         TEXT,
    journey_stage           TEXT,
    saving_behaviour_text   TEXT,
    saving_behaviour_code   TEXT,
    codes                   TEXT,   -- JSON list
    primary_code            TEXT,
    stated_reason           TEXT,
    underlying_reason       TEXT,
    barrier_type            TEXT,
    external_sources        TEXT,   -- JSON list
    comparison_behaviour    TEXT,
    seg_gender              TEXT,
    seg_life_stage          TEXT,
    seg_city_tier           TEXT,
    seg_price_sensitivity   TEXT,
    seg_shopping_frequency  TEXT,
    seg_category            TEXT,
    unmet_need              TEXT,
    severity                INTEGER,
    evidence_quote          TEXT,
    confidence              REAL,
    model                   TEXT,
    tagged_at               TEXT
);

-- Module 3: open codes, one row per (post, code). Written by the pilot open-coding
-- step; clustering reads them and the codebook is built from the clusters.
CREATE TABLE IF NOT EXISTS open_codes (
    post_id       TEXT NOT NULL REFERENCES tagged_posts(post_id),
    code          TEXT NOT NULL
);
-- PK-free on purpose: _migrate splices each column line into ALTER TABLE, which cannot
-- take a table-level PRIMARY KEY. Dedup is enforced at write time.
CREATE UNIQUE INDEX IF NOT EXISTS open_codes_pk ON open_codes (post_id, code);

-- Section 6: labels. Your hand labels. Same fields as tagged_posts.
-- UNIQUE(post_id, labeller) so re-labelling a post overwrites rather than duplicates.
CREATE TABLE IF NOT EXISTS labels (
    label_id                INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id                 TEXT NOT NULL,
    labeller                TEXT NOT NULL DEFAULT 'pm',
    labelled_at             TEXT,
    journey_stage           TEXT,
    saving_behaviour_text   TEXT,
    saving_behaviour_code   TEXT,
    codes                   TEXT,
    primary_code            TEXT,
    stated_reason           TEXT,
    underlying_reason       TEXT,
    barrier_type            TEXT,
    external_sources        TEXT,
    comparison_behaviour    TEXT,
    seg_gender              TEXT,
    seg_life_stage          TEXT,
    seg_city_tier           TEXT,
    seg_price_sensitivity   TEXT,
    seg_shopping_frequency  TEXT,
    seg_category            TEXT,
    unmet_need              TEXT,
    severity                INTEGER,
    evidence_quote          TEXT,
    confidence              REAL,
    UNIQUE(post_id, labeller)
);

-- Every LLM response is cached here, keyed by prompt hash + input id, so a re-run
-- costs nothing (AGENTS.md hard constraint).
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key     TEXT PRIMARY KEY,
    provider      TEXT,
    model         TEXT,
    stage         TEXT,
    input_id      TEXT,
    prompt_hash   TEXT,
    response_json TEXT,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    created_at    TEXT
);

-- Call counter: how many calls we have used today against the daily quota.
CREATE TABLE IF NOT EXISTS llm_call_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT,
    day           TEXT,
    stage         TEXT,
    provider      TEXT,
    model         TEXT,
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    from_cache    INTEGER DEFAULT 0,
    status        TEXT,
    error         TEXT,
    attempts      INTEGER DEFAULT 1
);

-- AGENTS.md: log counts at every stage. Mirrored to logs/run_log.csv.
CREATE TABLE IF NOT EXISTS run_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    stage     TEXT,
    source    TEXT,
    rows_in   INTEGER,
    rows_out  INTEGER,
    timestamp TEXT,
    notes     TEXT
);

-- AGENTS.md: output that failed validation after 3 retries.
CREATE TABLE IF NOT EXISTS errors (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    stage      TEXT,
    post_id    TEXT,
    error      TEXT,
    raw_output TEXT,
    timestamp  TEXT
);

CREATE INDEX IF NOT EXISTS idx_raw_source ON raw_posts(source);
CREATE INDEX IF NOT EXISTS idx_relevant_flag ON relevant_posts(is_relevant);
CREATE INDEX IF NOT EXISTS idx_tagged_primary ON tagged_posts(primary_code);
CREATE INDEX IF NOT EXISTS idx_tagged_barrier ON tagged_posts(barrier_type);
CREATE INDEX IF NOT EXISTS idx_calls_day ON llm_call_log(day);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def today() -> str:
    """The current quota day, in Pacific time.

    Google resets the Gemini and Groq daily request quota at midnight Pacific, not at
    midnight UTC. Counting against a UTC day makes our counter roll over up to seven
    hours before the provider's does, which looks like a fresh allowance and produces a
    burst of HTTP 429s in the evening. This is used only for quota accounting, never for
    post dates.
    """
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001 - no tzdata on this machine, fall back to UTC
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns that were introduced after a database was first created.

    CREATE TABLE IF NOT EXISTS silently does nothing when the table already exists, so
    a new column in SCHEMA would never reach an older database and every write to it
    would fail with "no such column". This walks SCHEMA and adds anything missing.
    """
    for match in re.finditer(
            r"CREATE TABLE IF NOT EXISTS (\w+)\s*\((.*?)\n\s*\);", SCHEMA, re.S):
        name, columns = match.group(1), match.group(2)
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                            (name,)).fetchone():
            continue
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({name})")}
        for line in columns.splitlines():
            line = line.strip()
            if not line or line.startswith("--"):
                continue
            # ALTER TABLE rejects the trailing comma that separates columns.
            definition = line.rstrip(",")
            # ...and any trailing SQL comment, which is fine in the schema file but is a
            # syntax error once spliced into ALTER TABLE. Strip it first.
            definition = re.sub(r"\s*--.*$", "", definition).strip().rstrip(",")
            if definition.split()[0].isidentifier() and definition.split()[0] not in have:
                conn.execute(f"ALTER TABLE {name} ADD COLUMN {definition}")
    conn.commit()


def connect(path=None) -> sqlite3.Connection:
    """Open the database, creating it and its tables on first use."""
    path = Path(path) if path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


# ---------- generic helpers ----------

def upsert(conn: sqlite3.Connection, table: str, rows: list[dict]) -> int:
    """Insert or update rows by primary key. Returns the number of rows written.

    Uses INSERT OR REPLACE so re-running a stage after a crash does not duplicate
    rows or fail on existing keys.
    """
    if not rows:
        return 0
    cols = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in cols)
    sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({placeholders})"
    conn.executemany(sql, [tuple(r.get(c) for c in cols) for r in rows])
    conn.commit()
    return len(rows)


def export_csv(rows, path) -> int:
    """Write rows (dicts or sqlite3.Row) to a CSV for you to open."""
    rows = [dict(r) for r in rows]
    if not rows:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def log_run(conn, stage: str, source: str, rows_in: int, rows_out: int, notes: str = "") -> None:
    """Record stage counts in the database and in logs/run_log.csv (AGENTS.md)."""
    conn.execute(
        "INSERT INTO run_log (stage, source, rows_in, rows_out, timestamp, notes) VALUES (?,?,?,?,?,?)",
        (stage, source, rows_in, rows_out, utc_now(), notes),
    )
    conn.commit()
    log_path = ROOT / "logs" / "run_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["stage", "source", "rows_in", "rows_out", "timestamp", "notes"])
        writer.writerow([stage, source, rows_in, rows_out, utc_now(), notes])


def log_error(conn, stage: str, error: str, post_id=None, raw_output=None) -> None:
    """Record output that failed validation after retries (AGENTS.md -> data/errors.csv)."""
    conn.execute(
        "INSERT INTO errors (stage, post_id, error, raw_output, timestamp) VALUES (?,?,?,?,?)",
        (stage, post_id, str(error)[:500], str(raw_output)[:2000] if raw_output else None, utc_now()),
    )
    conn.commit()
    csv_path = ROOT / "data" / "errors.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(["stage", "post_id", "error", "raw_output", "timestamp"])
        writer.writerow([stage, post_id, str(error)[:500], str(raw_output)[:2000] if raw_output else "",
                         utc_now()])


def rows_by_source(conn: sqlite3.Connection, table: str, source: str,
                   order_by: str = "date DESC") -> list[sqlite3.Row]:
    """Every row held for one source, so a CSV always mirrors the whole table rather
    than only the rows added by the most recent run."""
    return conn.execute(
        f"SELECT * FROM {table} WHERE source = ? ORDER BY {order_by}", (source,)).fetchall()


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]


def env_value(name: str) -> str:
    """Read a key from .env without ever printing it."""
    value = os.environ.get(name)
    return value.strip() if value else ""