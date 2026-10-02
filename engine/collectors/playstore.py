"""Play Store collector for the Myntra app.

Driven entirely by config/sources.yaml. No API key needed.

Two things worth knowing about the library:
  - `reviews()` is NOT a generator. It returns one tuple: (list of review dicts,
    continuation_token). Each call follows its own token internally and can pull up
    to 4500 reviews in a burst with no delay, so we ask for a small page at a time
    and sleep between calls.
  - `reviews_all()` ignores any limit, so it cannot be used when you want a cap.
    The loop below is ours for that reason.

Data hygiene:
  - Myntra attaches its own canned marketing reply to many reviews (`replyContent`).
    It is dropped here, before anything reaches the LLM, or it would be tagged as if
    it were the customer's opinion.
  - Usernames are hashed, never stored (AGENTS.md data contract).
  - Nothing is filtered here. Module 2 (clean) owns short posts, duplicates and spam,
    so that what lands in raw_posts is what Google actually served.
"""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from google_play_scraper import Sort, reviews as gp_reviews

from engine.db import utc_now

ROOT = Path(__file__).resolve().parent.parent.parent
BRAND_TERMS = ("myntra",)
DEVAANAGARI = re.compile(r"[\u0900-\u097F]")
# Safety net only: the library fetches this many per internal request. We never
# ask for the whole allowance at once, so a crawl never bursts.
MAX_PAGE = 4500


def _hash_author(name: str) -> str:
    """Hash a username so it is never written to disk in the clear."""
    if not name:
        return ""
    return hashlib.sha256(f"discovery-engine::{name.strip().lower()}".encode()).hexdigest()[:16]


def _lang_hint(text: str) -> str:
    return "hi" if DEVAANAGARI.search(text or "") else "en"


def _to_row(raw: dict, app_id: str, lang: str, country: str) -> dict | None:
    """Map one API review onto the raw_posts schema from AGENTS.md section 6."""
    text = (raw.get("content") or "").strip()
    if not text:
        return None
    created = raw.get("at")
    if isinstance(created, datetime):
        created = created.replace(tzinfo=timezone.utc).isoformat(timespec="seconds")
    return {
        "post_id": f"playstore:{raw.get('reviewId')}",
        "source": "playstore",
        "myntra_explicit": int(any(term in text.lower() for term in BRAND_TERMS)),
        "parent_context": f"Myntra on Google Play ({country})",
        "author_hash": _hash_author(raw.get("userName", "")),
        "date": created,
        "text": text,
        "rating": raw.get("score"),
        "engagement": raw.get("thumbsUpCount"),
        "url": (f"https://play.google.com/store/apps/details?id={app_id}"
                f"&reviewId={raw.get('reviewId')}&hl={lang}"),
        "lang_hint": _lang_hint(text),
        "collected_at": utc_now(),
        "raw_file": None,
    }


def _existing_ids(app_id_prefix: str = "playstore:") -> set[str]:
    """Already-collected IDs, so a re-run resumes instead of redoing finished work."""
    from engine import db
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT post_id FROM raw_posts WHERE post_id LIKE ?", (app_id_prefix + "%",)).fetchall()
        return {r["post_id"] for r in rows}
    finally:
        conn.close()


def fetch_sample(cfg: dict, n: int = 5) -> tuple[list[dict], list[str]]:
    """Fetch one small page and report the raw field names, so you can see exactly what
    the endpoint returns before anything is collected at scale."""
    ps = cfg["playstore"]
    result, _ = gp_reviews(app_id=ps["app_id"], lang=ps.get("lang", "en"),
                           country=ps.get("country", "in"),
                           sort=Sort.NEWEST, count=min(n, MAX_PAGE))
    rows = [_to_row(r, ps["app_id"], ps.get("lang", "en"), ps.get("country", "in")) for r in result]
    fields = sorted(result[0].keys()) if result else []
    return [r for r in rows if r], fields


def collect(cfg: dict, progress=None) -> list[dict]:
    """Collect newest-first up to the config limit. Resumable and polite."""
    ps = cfg["playstore"]
    app_id = ps["app_id"]
    lang = ps.get("lang", "en")
    country = ps.get("country", "in")
    limit = int(ps.get("limit", 10000))
    page_size = max(1, min(int(ps.get("page_size", 100)), MAX_PAGE))
    delay = float(ps.get("delay_seconds", 1.0))
    max_pages = int(ps.get("max_pages", 10000))

    already = _existing_ids()
    rows: list[dict] = []
    seen: set[str] = set()
    token = None
    pages = 0
    duplicates = 0

    while len(rows) + len(already) < limit and pages < max_pages:
        pages += 1
        try:
            result, token = gp_reviews(app_id=app_id, lang=lang, country=country,
                                       sort=Sort.NEWEST, count=page_size,
                                       continuation_token=token)
        except Exception as exc:
            # A transient failure must not throw away what we already have.
            print(f"   page {pages} failed ({type(exc).__name__}); keeping {len(rows)} rows collected so far")
            break

        if not result:
            print(f"   page {pages} returned no reviews: reached the end of what Google serves")
            break

        for raw in result:
            row = _to_row(raw, app_id, lang, country)
            if row is None:
                continue
            if row["post_id"] in seen or row["post_id"] in already:
                duplicates += 1
                continue
            seen.add(row["post_id"])
            rows.append(row)

        if progress:
            progress(len(rows), limit, pages, duplicates)

        if token is None or token.token is None:
            print(f"   Google offered no further pages after {pages}")
            break
        time.sleep(delay)

    if not rows:
        return []
    return _save_raw(rows, ROOT / "data" / "raw", app_id)


def _save_raw(rows: list[dict], raw_dir: Path, app_id: str) -> list[dict]:
    """Keep the untouched API payload for traceability, as AGENTS.md requires."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = raw_dir / f"playstore_{app_id}_{stamp}.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for row in rows:
        row["raw_file"] = path.name
    return rows