"""Helpers shared by the collectors.

These exist so every source handles the data contract in exactly one place. If the
anonymisation rule ever changes, it changes here and nowhere else.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from engine import db

ROOT = Path(__file__).resolve().parent.parent.parent
BRAND_TERMS = ("myntra",)
DEVAANAGARI = re.compile(r"[\u0900-\u097F]")


def hash_author(name: str) -> str:
    """Hash a username so it is never written to disk or sent to the LLM.

    The prefix means the same username hashes differently here than it would in an
    unrelated system, so the value cannot be looked up in a precomputed table.
    """
    if not name:
        return ""
    return hashlib.sha256(f"discovery-engine::{name.strip().lower()}".encode()).hexdigest()[:16]


def lang_hint(text: str) -> str:
    """'hi' only when the text is actually in Devanagari script.

    Romanised Hinglish such as "best hai" reads as English by this test. That is
    known and acceptable: the relevance stage deals with Hinglish anyway.
    """
    return "hi" if DEVAANAGARI.search(text or "") else "en"


def mentions_myntra(text: str) -> int:
    return int(any(term in (text or "").lower() for term in BRAND_TERMS))


def existing_ids(prefix: str) -> set[str]:
    """Post IDs already stored for a source, so a re-run resumes instead of redoing
    work that is already finished."""
    conn = db.connect()
    try:
        return {r["post_id"] for r in conn.execute(
            "SELECT post_id FROM raw_posts WHERE post_id LIKE ?", (prefix + "%",))}
    finally:
        conn.close()


def save_raw(rows: list[dict], label: str) -> list[dict]:
    """Write the collected rows to data/raw/ and point every row at that file, so any
    finding can be traced back to exactly what the source served."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = ROOT / "data" / "raw" / f"{label}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for row in rows:
        row["raw_file"] = path.name
    return rows