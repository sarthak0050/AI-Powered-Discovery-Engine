"""YouTube comments collector via YouTube Data API v3.

Quota is the real constraint, so this collector is built around three facts measured
against the live API rather than assumed:

  - ``search.list`` costs 100 units per call. It is by far the most expensive thing
    here, so results are cached to disk and a re-run pays nothing for them.
  - ``commentThreads.list`` costs 1 unit per call and returns up to 100 top-level
    comments. 300 top-level comments therefore costs 3 units, not 300.
  - ``comments.list`` costs 1 unit per call for replies to a thread.

The daily allowance is 10,000 units shared across anything using the key. Spending is
tracked locally in a small state file, because the API will not tell you what you have
used, and the run stops with headroom rather than discovering the ceiling the hard way.

Data hygiene: display names are hashed, never stored. Comment text is requested as
plainText so no HTML is carried into the corpus.
"""

import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

from engine.collectors.common import (ROOT, existing_ids, hash_author, lang_hint,
                                      mentions_myntra, save_raw)
from engine import db
from engine.db import log_error

API = "https://www.googleapis.com/youtube/v3"
CACHE_DIR = ROOT / "data" / "cache" / "youtube"
QUOTA_STATE = CACHE_DIR / "quota.json"
# Enough context that a query with different settings is not served a stale answer.
CACHE_VERSION = "v1"

SEARCH_COST = 100
THREAD_COST = 1
REPLY_COST = 1

# A row must say something. These are YouTube's own placeholders for deleted or
# held-back comments, and storing them put noise straight into the corpus.
# A row must say something. These are YouTube's own stand-ins for comments that were
# deleted, held back, or left on a video the viewer cannot see.
# Matching is deliberately exact-plus-bracketed: a loose "starts with deleted" rule
# would throw away genuine comments like "deleted my cart".
PLACEHOLDER_EXACT = {
    "[removed]", "[deleted]", "[hidden]", "[comment removed]", "[comment deleted]",
    "[comment removed by user]", "[comment removed by channel owner]",
    "[comment deleted by user]", "[deleted comment]", "[removed comment]",
    "this comment was removed", "this comment has been deleted",
    "comment removed by user",
}
PLACEHOLDER_BRACKETED = re.compile(
    r"^\[[^\]]{0,60}(removed|deleted|hidden|unavailable|disabled)[^\]]{0,60}\]$", re.I)
# Long-form placeholder YouTube substitutes for a comment on a members-only video.
PLACEHOLDER_PREFIXES = (
    "link is mentioned in my community",
    "this comment is not available",
)
LINK_ONLY = re.compile(r"^\s*(https?://\S+\s*)+$")


class BudgetReached(Exception):
    """Raised before a call would push the run past its own safety budget."""


class Quota:
    """Local estimate of today's usage, persisted so a second run in the same day
    knows what the first one already spent."""

    def __init__(self, limit: int, budget: int):
        self.limit = limit
        self.budget = budget
        self.today = datetime.now(timezone.utc).date().isoformat()
        self.spent = 0
        self.calls = 0
        self.loads()

    def loads(self) -> None:
        if QUOTA_STATE.exists():
            try:
                state = json.loads(QUOTA_STATE.read_text())
                if state.get("date") == self.today:
                    self.spent = state.get("units", 0)
                    self.calls = state.get("calls", 0)
            except (ValueError, OSError):
                pass    # a corrupt state file must not stop a collection

    def save(self) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        QUOTA_STATE.write_text(json.dumps(
            {"date": self.today, "units": self.spent, "calls": self.calls,
             "daily_limit": self.limit, "budget": self.budget}, indent=1))

    def spend(self, units: int, what: str) -> None:
        if self.spent + units > self.budget:
            raise BudgetReached(
                f"{self.spent}/{self.budget} units used; refusing to spend {units} more on {what}")
        self.spent += units
        self.calls += 1
        self.save()

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.spent)


class YouTube:
    def __init__(self, key: str, quota: Quota):
        self.key = key
        self.quota = quota
        self.last_call = 0.0
        self.errors: list[str] = []

    def get(self, path: str, cost: int, what: str, **params) -> dict:
        self.quota.spend(cost, what)
        # A small pause keeps the collector from looking like a scraper.
        gap = 0.25 - (time.time() - self.last_call)
        if gap > 0:
            time.sleep(gap)
        try:
            response = requests.get(f"{API}/{path}", params={**params, "key": self.key},
                                    headers={"Accept": "application/json"}, timeout=45)
        except requests.RequestException as exc:
            self.errors.append(f"{what}: {type(exc).__name__}")
            return {}
        self.last_call = time.time()

        if response.status_code == 200:
            return response.json()
        message = response.json().get("error", {}).get("message", "")
        if response.status_code in (403, 429) and "quota" in message.lower():
            raise BudgetReached(f"Google says the daily quota is exhausted: {message[:140]}")
        # Keep Google's own wording: a bare status code hides which parameter was wrong.
        self.errors.append(f"{what}: HTTP {response.status_code} {message[:160]}")
        return {}

    def search(self, query: str, cfg: dict) -> list[dict]:
        """Top videos for one query, served from disk cache when we already have them."""
        params = {
            # search.list only accepts the snippet part. Asking for
            # contentDetails or statistics returns HTTP 400 unknownPart.
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": int(cfg["videos_per_query"]),
            "order": cfg.get("order", "relevance"),
            "publishedAfter": _published_after(cfg),
            "relevanceLanguage": cfg.get("relevance_language", "en"),
            "regionCode": cfg.get("region_code", "IN"),
            "safeSearch": cfg.get("safe_search", "moderate"),
        }
        key = hashlib.sha256(
            json.dumps({"v": CACHE_VERSION, **params}, sort_keys=True).encode()).hexdigest()[:20]
        path = CACHE_DIR / f"search_{key}.json"

        if cfg.get("cache_search_results", True) and path.exists():
            try:
                cached = json.loads(path.read_text())
                if cached.get("params") == params:
                    return cached["items"]
            except (ValueError, KeyError, OSError):
                pass

        data = self.get("search", SEARCH_COST, f"search {query!r}", **params)
        items = data.get("items") or []
        if items:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"params": params, "items": items}, ensure_ascii=False))
        return items


def _published_after(cfg: dict) -> str:
    months = int(cfg.get("months_back", 24))
    start = datetime.now(timezone.utc) - timedelta(days=30 * months)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(text: str) -> str | None:
    """Reject anything that is not actually words a person wrote."""
    text = (text or "").strip()
    if not text or LINK_ONLY.match(text):
        return None
    if text.lower() in PLACEHOLDER_EXACT or PLACEHOLDER_BRACKETED.match(text):
        return None
    if text.lower().startswith(PLACEHOLDER_PREFIXES):
        return None
    return text


def _iso(stamp: str | None) -> str | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).isoformat(timespec="seconds")
    except ValueError:
        return None


def _row(post_id: str, text: str, snippet: dict, video_id: str, title: str) -> dict:
    return {
        "post_id": post_id,
        "source": "youtube",
        "myntra_explicit": mentions_myntra(text),
        "parent_context": title,              # the video title, as specified
        "author_hash": hash_author(snippet.get("authorDisplayName", "")),
        "date": _iso(snippet.get("publishedAt")),
        "text": text,
        "rating": None,                        # YouTube comments are not rated
        "engagement": snippet.get("likeCount"),
        "url": f"https://www.youtube.com/watch?v={video_id}&lc={post_id.rsplit(':', 1)[-1]}",
        "lang_hint": lang_hint(text),
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_file": None,
    }


def fetch_video_comments(yt: YouTube, video_id: str, cfg: dict, title: str) -> list[dict]:
    """Top-level comments for one video, then replies to the threads that have any."""
    rows: list[dict] = []
    cap = int(cfg.get("max_top_level_comments", 300))
    collected_threads = 0
    page_token = None
    reply_units = 0
    reply_cap = int(cfg.get("max_reply_units_per_video", 80))

    while collected_threads < cap:
        data = yt.get("commentThreads", THREAD_COST, f"comments {video_id}",
                      part="snippet", videoId=video_id, maxResults=100,
                      order=cfg.get("order", "relevance"), textFormat="plainText",
                      **({"pageToken": page_token} if page_token else {}))
        items = data.get("items") or []
        if not items:
            break

        for thread in items:
            top = thread.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            text = _clean(top.get("textDisplay", ""))
            if text:
                rows.append(_row(f"youtube:comment:{thread['id']}", text, top, video_id,
                                 title))
            collected_threads += 1

            replies = thread.get("snippet", {}).get("totalReplyCount") or 0
            if cfg.get("include_replies", True) and replies and reply_units < reply_cap:
                reply_rows, spent = _fetch_replies(yt, thread["id"], cfg, video_id, title,
                                                   reply_cap - reply_units)
                rows += reply_rows
                reply_units += spent

            if collected_threads >= cap:
                break

        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return rows


def _fetch_replies(yt: YouTube, thread_id: str, cfg: dict, video_id: str, title: str,
                   units_left: int) -> tuple[list[dict], int]:
    """Replies to one thread. Returns the rows and how many quota units they cost,
    so the caller can keep one comment-heavy video from eating the whole budget."""
    rows: list[dict] = []
    spent = 0
    page_token = None
    for _ in range(int(cfg.get("max_reply_pages_per_thread", 2))):
        if units_left - spent <= 0:
            break
        data = yt.get("comments", REPLY_COST, f"replies {thread_id}",
                      part="snippet", parentId=thread_id, maxResults=100,
                      textFormat="plainText",
                      **({"pageToken": page_token} if page_token else {}))
        spent += REPLY_COST
        for item in data.get("items") or []:
            snippet = item.get("snippet", {})
            text = _clean(snippet.get("textDisplay", ""))
            if text:
                rows.append(_row(f"youtube:reply:{item['id']}", text, snippet, video_id,
                                 title))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return rows, spent


def collect(cfg: dict, progress=None) -> list[dict]:
    yc = cfg["youtube"]
    load_dotenv(dotenv_path=str(ROOT / ".env"))
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("YOUTUBE_API_KEY is not set in .env, so YouTube cannot be collected")

    quota = Quota(int(yc.get("daily_quota_limit", 10000)), int(yc.get("quota_budget", 8000)))
    print(f"   quota: {quota.spent} units already spent today, budget {quota.budget} "
          f"of Google's {quota.limit} daily limit")
    yt = YouTube(key, quota)

    already = existing_ids("youtube:")
    rows: list[dict] = []
    seen: set[str] = set()
    per_query: dict[str, dict] = {}
    videos_total = 0
    stopped_early = None

    for query in yc.get("queries", []):
        try:
            videos = yt.search(query, yc)
        except BudgetReached as exc:
            stopped_early = str(exc)
            break

        q_rows = 0
        for item in videos:
            vid = item["id"]["videoId"]
            snippet = item.get("snippet", {})
            title = snippet.get("title", "")
            videos_total += 1

            if yc.get("include_videos", True):
                text = _clean(" ".join([title, snippet.get("description", "")[:500]]))
                if text:
                    video_row = _row(f"youtube:video:{vid}", text, snippet, vid, title)
                    if video_row["post_id"] not in seen and video_row["post_id"] not in already:
                        seen.add(video_row["post_id"])
                        rows.append(video_row)
                        q_rows += 1

            try:
                for row in fetch_video_comments(yt, vid, yc, title):
                    if row["post_id"] in seen or row["post_id"] in already:
                        continue
                    seen.add(row["post_id"])
                    rows.append(row)
                    q_rows += 1
            except BudgetReached as exc:
                stopped_early = str(exc)
                break

        per_query[query] = {"videos": len(videos), "rows": q_rows,
                            "units": quota.spent}
        print(f"   {query:24} {len(videos):2} videos -> {q_rows:4} rows "
              f"({quota.spent} units used so far)")
        if stopped_early:
            break
        if progress:
            progress(len(rows), 0, 0, 0)

    if stopped_early:
        print(f"   stopped early on purpose: {stopped_early}")
        print(f"   {quota.spent} units used, {quota.remaining} left of the budget")

    if yt.errors:
        print(f"   {len(yt.errors)} request errors, first few: {yt.errors[:3]}")

    conn = db.connect()
    try:
        for error in yt.errors[:50]:
            log_error(conn, "collect", error[:400], None, "YouTube request failed")
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("no YouTube comments were collected")

    archive = save_raw(rows, "youtube")
    (CACHE_DIR / "per_query.json").write_text(json.dumps(
        {"queries": per_query, "units": quota.spent, "calls": quota.calls,
         "videos": videos_total, "rows": len(rows)}, indent=1))
    return archive