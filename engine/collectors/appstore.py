"""App Store collector for the Myntra iOS app.

Source: Apple's public customer-reviews RSS JSON feed. No API key needed.

The feed's ceiling, measured rather than assumed:
    https://itunes.apple.com/{storefront}/rss/customerreviews/page={n}/id={app_id}/sortby={sort}/json
  - 50 reviews per page
  - page 11 and beyond return HTTP 400, so 10 pages is the hard limit: 500 reviews
  - the whole feed covers only the last few days, so this source adds a second
    platform's perspective, not extra history

An earlier note in this repo claimed the feed was dead and returned nothing for any
app. That was wrong: it was tested and it works. The old collector is replaced.

Data hygiene: usernames are hashed, never stored. Apple attaches no brand reply to
reviews, unlike Google Play.
"""

import time
from datetime import datetime, timezone

import requests

from engine.collectors.common import (existing_ids, hash_author, lang_hint, mentions_myntra,
                                      save_raw)
from engine import db
from engine.db import log_error

FEED = ("https://itunes.apple.com/{storefront}/rss/customerreviews/page={page}"
        "/id={app_id}/sortby={sort}/json")
LOOKUP = "https://itunes.apple.com/lookup"
HEADERS = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}


def _get(url: str, params: dict | None = None) -> requests.Response:
    return requests.get(url, params=params, headers=HEADERS, timeout=30)


def lookup_app(cfg: dict) -> dict:
    """Confirm the configured ID still points at the app we think it does.

    App IDs get reused and the Myntra app has changed hands in the past, so this is
    checked rather than trusted. Used by `doctor`.
    """
    ps = cfg["appstore"]
    r = _get(LOOKUP, {"id": ps["app_id"], "country": ps.get("storefront", "in")})
    r.raise_for_status()
    results = r.json().get("results") or []
    return results[0] if results else {}


def _to_row(entry: dict, app_id: str, storefront: str) -> dict | None:
    """Map one RSS entry onto the raw_posts schema from AGENTS.md section 6."""
    content = (entry.get("content", {}).get("label") or "").strip()
    title = (entry.get("title", {}).get("label") or "").strip()
    text = content or title
    if not text:
        return None

    stamp = entry.get("updated", {}).get("label")
    try:
        date = datetime.fromisoformat(stamp).astimezone(timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        date = None

    return {
        "post_id": f"appstore:{entry.get('id', {}).get('label')}",
        "source": "appstore",
        "myntra_explicit": mentions_myntra(text),
        "parent_context": f"Myntra on the {storefront.upper()} App Store",
        "author_hash": hash_author(entry.get("author", {}).get("name", {}).get("label", "")),
        "date": date,
        "text": text,
        "rating": int(entry["im:rating"]["label"]) if "im:rating" in entry else None,
        "engagement": entry.get("im:voteSum", {}).get("label"),
        "url": f"https://apps.apple.com/{storefront}/app/id{app_id}?see-all=reviews",
        "lang_hint": lang_hint(text),
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_file": None,
    }


def _entries(response: requests.Response) -> list[dict]:
    """Reviews only. Page 1 sometimes carries the app itself as the first entry, which
    has no rating and must not be stored as if it were a customer's review."""
    feed = response.json().get("feed", {})
    entries = feed.get("entry", []) or []
    if isinstance(entries, dict):
        entries = [entries]
    return [e for e in entries if isinstance(e, dict) and "im:rating" in e]


def collect(cfg: dict, progress=None) -> list[dict]:
    """Walk every page the feed offers, newest first. Resumable and polite."""
    ps = cfg["appstore"]
    app_id = str(ps["app_id"])
    storefront = ps.get("storefront", "in")
    sort = ps.get("sort", "mostrecent")
    delay = float(ps.get("delay_seconds", 3))
    max_pages = int(ps.get("pages", 10))
    limit = int(ps.get("limit", 500))

    already = existing_ids("appstore:")
    rows: list[dict] = []
    seen: set[str] = set()
    duplicates = 0
    pages_used = 0

    for page in range(1, max_pages + 1):
        url = FEED.format(storefront=storefront, page=page, app_id=app_id, sort=sort)
        try:
            response = _get(url)
        except requests.RequestException as exc:
            print(f"   page {page} request failed ({type(exc).__name__}); "
                  f"keeping the {len(rows)} reviews collected so far")
            break

        if response.status_code != 200:
            # This is how the feed says "no more pages", so it is not an error.
            print(f"   page {page} returned HTTP {response.status_code}: "
                  f"that is the end of the feed")
            break

        try:
            entries = _entries(response)
        except ValueError:
            print(f"   page {page} was not valid JSON; stopping here")
            break

        if not entries:
            print(f"   page {page} held no reviews: end of feed")
            break

        pages_used = page
        for entry in entries:
            row = _to_row(entry, app_id, storefront)
            if row is None:
                continue
            if row["post_id"] in seen or row["post_id"] in already:
                duplicates += 1
                continue
            seen.add(row["post_id"])
            rows.append(row)

        if progress:
            progress(len(rows), limit, page, duplicates)
        if len(rows) + len(already) >= limit:
            break
        time.sleep(delay)

    if not rows:
        # Per config: never save an empty file, record why, and let the pipeline carry on.
        message = (f"App Store feed returned no reviews for app {app_id} in the "
                   f"{storefront} storefront after {pages_used} pages")
        conn = db.connect()
        try:
            log_error(conn, "collect", "ZeroReviews", None, message)
        finally:
            conn.close()
        raise RuntimeError(message)

    print(f"   used {pages_used} of {max_pages} pages the feed offers")
    return save_raw(rows, f"appstore_{app_id}")