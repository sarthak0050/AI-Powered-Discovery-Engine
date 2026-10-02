"""Reddit collector for Myntra discussion, via the Arctic Shift API.

Free, no key, no uptime guarantee. Arctic Shift is a volunteer project, so this
collector is deliberately slow and always willing to give up rather than hammer it.

What the API actually does (measured, not assumed):
  - Keyword search REQUIRES a subreddit. There is no global text search, which is why
    config lists subreddits rather than letting the API find them.
  - Passing `sort` to a keyword query returns HTTP 422 "Timeout". Never sorted here.
  - Omitting `after` also times out. A date floor is always sent.
  - Deep pagination is unreliable, so instead of crawling pages we walk the date range
    and bisect it when the server times out.
  - HTTP 429 arrives after a handful of quick calls. `X-RateLimit-Reset` carries the
    number of seconds to wait; it is obeyed, not guessed.

What counts as a match: the text must mention Myntra AND at least one neutral search
term. Comments under a matching post are kept even when they do not repeat the brand,
because the surrounding conversation is where the reason behind a complaint appears.
"""

import time
from datetime import datetime, timedelta, timezone

import requests

from engine import db
from engine.collectors.common import (existing_ids, hash_author, lang_hint, mentions_myntra,
                                      save_raw)
from engine.db import log_error

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; discovery-engine/1.0; +local research)"}
# Fields worth keeping. Everything else the API sends is stripped before archiving.
POST_KEEP = ("id", "created_utc", "author", "subreddit", "title", "selftext", "score",
             "num_comments", "permalink")
COMMENT_KEEP = ("id", "created_utc", "author", "subreddit", "body", "score", "link_id",
                "parent_id", "permalink")


class DownForNow(Exception):
    """The API is refusing us; wait and try once more."""


def _iso(epoch) -> str | None:
    try:
        return datetime.fromtimestamp(float(epoch), timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def _trim(item: dict, keep: tuple) -> dict:
    return {k: v for k, v in item.items() if k in keep}


def mentions(text: str, terms: list[str]) -> bool:
    """True when the text names Myntra and at least one neutral search term."""
    low = (text or "").lower()
    return "myntra" in low and any(t in low for t in terms)


class ArcticShift:
    """Thin client that knows how this particular API misbehaves."""

    def __init__(self, cfg: dict):
        self.base = cfg["base_url"].rstrip("/")
        self.delay = float(cfg.get("delay_seconds", 6))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.last_call = 0.0
        self.requests_made = 0
        self.rate_limit_waits = 0

    def get(self, path: str, **params):
        """One polite GET. Raises DownForNow when the service says no."""
        for attempt in range(1, self.max_retries + 1):
            wait = self.delay - (time.time() - self.last_call)
            if wait > 0:
                time.sleep(wait)

            try:
                response = requests.get(f"{self.base}{path}", params=params,
                                        headers=HEADERS, timeout=60)
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise DownForNow(f"{type(exc).__name__}: {exc}") from exc
                time.sleep(3 * attempt)
                continue

            self.last_call = time.time()
            self.requests_made += 1

            if response.status_code == 429:
                # The service tells us exactly how long to wait. Believe it.
                reset = response.headers.get("X-RateLimit-Reset")
                pause = int(reset) + 2 if (reset or "").isdigit() else 20 * attempt
                self.rate_limit_waits += 1
                if attempt == self.max_retries:
                    raise DownForNow(f"rate limited, gave up after {pause}s")
                time.sleep(pause)
                continue

            if response.status_code >= 500:
                if attempt == self.max_retries:
                    raise DownForNow(f"HTTP {response.status_code}")
                time.sleep(5 * attempt)
                continue

            return response

        raise DownForNow("exhausted retries")

    @staticmethod
    def timed_out(response) -> bool:
        """The API signals an unoptimised query with 422 and this exact message."""
        if response.status_code != 422:
            return False
        try:
            return "timeout" in (response.json().get("error") or "").lower()
        except ValueError:
            return False

    def data(self, response) -> list[dict]:
        try:
            return response.json().get("data") or []
        except ValueError:
            return []


def verify_subreddits(api: ArcticShift, names: list[str]) -> dict[str, dict]:
    """Check each configured subreddit really exists before spending requests on it.

    One request per name, because the bulk /ids endpoint wants internal IDs rather
    than the names people actually recognise.
    """
    found = {}
    for name in names:
        try:
            response = api.get("/subreddits/search", subreddit=name)
        except DownForNow as exc:
            print(f"   {name}: could not verify ({exc}); will still try it")
            continue
        rows = api.data(response)
        if rows:
            found[name] = rows[0]
        else:
            print(f"   {name}: not found in Arctic Shift, skipping")
    return found


def fetch_posts(api: ArcticShift, sub: str, cfg: dict, after: int, before: int,
                budget: int | None = None, field: str = "query") -> list[dict]:
    """Posts in a date window that mention Myntra.

    The API's own advice is to retry first, because a cold query often succeeds on a
    second attempt, and to narrow the window only if that fails. So: retry, then bisect,
    then give up on that subreddit rather than grinding through the whole timeline.

    `field` picks the keyword target: "query" searches titles and body text,
    "title" only titles. Title-only search is markedly cheaper for this API, so it is
    used as a fallback when full-text times out.
    """
    page_size = int(cfg.get("page_size", 100))
    max_pages = int(cfg.get("max_pages_per_subreddit", 5))
    max_retries = int(cfg.get("max_warmup_retries", 2))
    min_split = int(cfg.get("min_split_days", 60))
    collected: list[dict] = []
    cursor = after
    started = api.requests_made

    def out_of_budget() -> bool:
        return budget is not None and (api.requests_made - started) >= budget

    for _ in range(max_pages):
        batch, retry = [], 0
        while True:
            try:
                response = api.get("/posts/search", subreddit=sub,
                                   **{field: cfg["query"]},
                                   after=cursor, before=before, limit=page_size)
            except DownForNow as exc:
                print(f"   {sub}: {exc}")
                return collected
            if not api.timed_out(response):
                break
            # Cold query: the docs say a second attempt often works.
            retry += 1
            if retry > max_retries or out_of_budget():
                break
            time.sleep(5 * retry)

        if api.timed_out(response):
            span_days = (before - cursor) / 86400
            if span_days > min_split and not out_of_budget():
                middle = cursor + int((before - cursor) / 2)
                print(f"   {sub}: still timing out over {span_days:.0f} days, "
                      f"narrowing to two halves")
                collected += fetch_posts(api, sub, cfg, cursor, middle, budget, field)
                collected += fetch_posts(api, sub, cfg, middle, before, budget, field)
                return collected
            print(f"   {sub}: timed out on a {span_days:.0f}-day window; "
                  f"recording partial coverage for this subreddit")
            break

        rows = api.data(response)
        if not rows:
            break
        batch += [_trim(r, POST_KEEP) for r in rows]
        collected += batch

        # No sort order is available, so walk forward using the newest timestamp seen.
        newest = max((r.get("created_utc") or 0) for r in batch)
        if newest <= cursor:
            break                      # no forward progress: stop rather than loop
        cursor = newest + 1
        if len(rows) < page_size:
            break                      # short page means the window is exhausted
        if out_of_budget():
            break

    return collected


def flatten_comments(nodes: list[dict], keep: int) -> list[dict]:
    """Flatten the tree the API returns.

    Two shapes arrive here: each node is {"kind": ..., "data": {...comment...}} and
    collapsed nodes are {"kind": "more"} carrying only child IDs, which have nothing
    to read and are skipped. Reading the wrapper instead of "data" yields blank bodies,
    so the unwrap is the whole trick here.
    """
    out: list[dict] = []

    def walk(items):
        for item in items or []:
            if not isinstance(item, dict):
                continue
            if item.get("kind") == "more":
                continue
            comment = item.get("data")
            if not isinstance(comment, dict) or not (comment.get("body") or "").strip():
                continue
            out.append(_trim(comment, COMMENT_KEEP))
            if len(out) >= keep:
                return
            walk(item.get("replies"))

    walk(nodes)
    return out[:keep]


def fetch_comments(api: ArcticShift, post_id: str, cfg: dict) -> list[dict]:
    try:
        response = api.get("/comments/tree", link_id=post_id,
                           limit=int(cfg.get("max_comments_per_post", 300)))
    except DownForNow as exc:
        print(f"   comments for {post_id}: {exc}")
        return []
    if api.timed_out(response):
        return []
    return flatten_comments(api.data(response), int(cfg.get("max_comments_per_post", 300)))


def _row(item: dict, kind: str, sub: str) -> dict | None:
    """Map one post or comment onto the raw_posts schema."""
    if kind == "post":
        title = (item.get("title") or "").strip()
        body = (item.get("selftext") or "").strip()
        text = f"{title}\n{body}".strip() if body else title
    else:
        text = (item.get("body") or "").strip()
    if not text or text.lower() in ("[removed]", "[deleted]"):
        # A deleted comment carries no text at all. Storing the placeholder would put
        # a fifth of the corpus into rows that say nothing about Myntra.
        return None
    return {
        "post_id": f"reddit:{kind}:{item.get('id')}",
        "source": "reddit",
        "myntra_explicit": mentions_myntra(text),
        "parent_context": sub,           # the subreddit, as specified
        "author_hash": hash_author(item.get("author", "")),
        "date": _iso(item.get("created_utc")),
        "text": text,
        "rating": None,                   # Reddit has no stars
        "engagement": item.get("score"),
        "url": f"https://reddit.com{item.get('permalink')}" if item.get("permalink") else None,
        "lang_hint": lang_hint(text),
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_file": None,
    }


def collect(cfg: dict, progress=None) -> list[dict]:
    """Walk every configured subreddit. A subreddit that fails is logged and skipped."""
    r = cfg["reddit"]
    api = ArcticShift(r)
    terms = [t.lower() for t in r.get("search_terms", [])]
    now = datetime.now(timezone.utc)
    before = int(now.timestamp())
    after = int((now - timedelta(days=30 * int(r.get("months_back", 24)))).timestamp())
    subs = r.get("subreddits", [])

    print(f"   window: {datetime.fromtimestamp(after, timezone.utc).date()} to "
          f"{datetime.fromtimestamp(before, timezone.utc).date()}")
    print(f"   verifying {len(subs)} subreddits")
    verify_subreddits(api, subs)

    already = existing_ids("reddit:")
    rows: list[dict] = []
    seen: set[str] = set()
    per_sub: dict[str, tuple[int, int]] = {}
    skipped: list[tuple[str, str]] = []

    for sub in subs:
        budget = int(r.get("requests_per_subreddit", 8))
        try:
            posts = fetch_posts(api, sub, r, after, before, budget)
            if not posts and r.get("title_fallback", True):
                # Title-only search is much cheaper here and often succeeds where the
                # full-text query times out.
                posts = fetch_posts(api, sub, r, after, before, max(2, budget // 2), "title")
        except DownForNow as exc:
            skipped.append((sub, str(exc)))
            continue

        posts_kept = comments_kept = 0
        for post in posts:
            text = f"{post.get('title') or ''} {post.get('selftext') or ''}"
            if not mentions(text, terms):
                continue
            row = _row(post, "post", sub)
            if row is None or row["post_id"] in seen or row["post_id"] in already:
                continue
            seen.add(row["post_id"])
            rows.append(row)
            posts_kept += 1

            if r.get("include_comments", True):
                # Every comment under a matching post is kept: the reply is where the
                # reason behind the complaint usually appears.
                for comment in fetch_comments(api, post["id"], r):
                    crow = _row(comment, "comment", sub)
                    if crow is None or crow["post_id"] in seen or crow["post_id"] in already:
                        continue
                    seen.add(crow["post_id"])
                    rows.append(crow)
                    comments_kept += 1

        per_sub[sub] = (posts_kept, comments_kept)
        if posts_kept or comments_kept:
            print(f"   {sub:22} {posts_kept:3} posts, {comments_kept:3} comments")

    conn = db.connect()
    try:
        for sub, reason in skipped:
            log_error(conn, "collect", "RedditUnavailable", None, f"{sub}: {reason}")
    finally:
        conn.close()

    print(f"   {api.requests_made} requests, {api.rate_limit_waits} rate-limit waits, "
          f"{len(skipped)} subreddits skipped")

    if not rows:
        raise RuntimeError("no Reddit posts matched; see data/errors.csv for skipped subreddits")

    return save_raw(rows, "reddit")