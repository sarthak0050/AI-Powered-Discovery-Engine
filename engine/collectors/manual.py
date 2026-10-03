"""Manual collector for sources that must not be scraped.

Quora, X, Instagram and blogs have no usable public API for this, so per AGENTS.md
they are copied by hand into data/manual/*.csv and loaded from here. Nothing is
fetched over the network: this collector only reads your files.

Required columns: source_name, url, date, text

    source_name   where the post came from, e.g. quora, x, instagram, blog
                  This becomes post_id and parent_context, so keep it consistent
                  across rows ("x" not "X" on some rows and "twitter" on others).
    url           link to the post
    date          any format Excel understands; anything ISO-ish is stored as-is
    text          the post body, in your own words if you prefer to keep it short

Optional columns, used when present and ignored when not:
    author        hashed on import and never stored in the clear
    engagement    likes or upvotes, if you noted them
    rating        star rating, if the source has one
    notes         anything you want to remember about the row; never sent to the LLM

A few things this handles that are worth knowing about:
  - The template file and any file starting with an underscore are skipped, so you
    can keep working examples in the folder without them being collected.
  - Blank rows and rows with no text are skipped and counted, not silently dropped.
  - A missing or misspelled column is a hard error listing what was found, because
    silently importing a wrong file would poison the corpus.
  - Deduplication is on a hash of the text, so pasting the same post twice is a no-op
    even if the URL differs.
  - An unparseable date is kept with date=None rather than discarded. Losing a whole
    post because of a date format is worse than losing its date.
"""

import csv
import hashlib
import re
from datetime import datetime, timezone

from engine.collectors.common import (ROOT, hash_author, lang_hint, mentions_myntra,
                                      save_raw)

REQUIRED = ("source_name", "url", "date", "text")
# Read when present, ignored when absent. Kept in one place so the template header
# and this collector cannot drift apart.
OPTIONAL = ("author", "engagement", "rating", "notes")

MANUAL_DIR = ROOT / "data" / "manual"
TEMPLATE = "TEMPLATE_manual_posts.csv"
TEMPLATE_HEADER = ",".join(REQUIRED + OPTIONAL)

# A row has to say something. This is not a length limit: one-word posts are real
# sentiment and are kept. This only catches rows that carry no readable content,
# such as a cell that got filled with "n/a" or a stray emoji.
MEANINGLESS = re.compile(r"^\s*(n/?a|none|tbd|todo|test|asdf|\.+)\s*$", re.I)


class ManualFormatError(Exception):
    """Raised when a CSV is missing the columns this collector needs."""


def _slug(value: str) -> str:
    """Normalise a source name so 'X ', 'x' and 'Twitter' cannot become two sources."""
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or "manual"


def _parse_date(value: str) -> str | None:
    """Return an ISO timestamp, or None when the value cannot be read.

    Kept permissive on purpose: your CSVs will come out of Excel and browsers in
    several formats, and a post with a missing date is still worth having.
    """
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d-%m-%Y",
                "%d/%m/%Y", "%m/%d/%Y", "%d %b %Y", "%b %d, %Y", "%d-%b-%Y", "%d-%B-%Y"):
        try:
            return datetime.strptime(value[:len(fmt) + 6].strip(), fmt).isoformat(
                timespec="seconds")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat(timespec="seconds")
    except ValueError:
        return None


def _number(value) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def ensure_template(force: bool = False) -> "object":
    """Create the template CSV for the user to copy and fill in.

    Written with a few example rows so the format is obvious. It is skipped by
    collection because it lives under TEMPLATE and files starting with _ are ignored.
    """
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    path = MANUAL_DIR / TEMPLATE
    if path.exists() and not force:
        return path
    rows = [
        # Real examples, kept short. Replace every one of these with your own posts.
        TEMPLATE_HEADER,
        "quora,https://www.quora.com/Some-Question-about-online-shopping-answer-123,"
        "2025-08-14,I keep saving Myntra dresses for the sale but never buy them because "
        "I cannot tell if the fabric is good,",
        "x,https://x.com/someone/status/1234567890,2025-11-02,Myntra app had the same dress "
        "cheaper than the website so I switched,,12,,price check",
        "instagram,https://www.instagram.com/p/ABC123/,2026-01-20,Myntra haul - the jeans "
        "fit well but the top was loose around the waist,,45,,video is mostly product shots",
        "blog,https://someblog.com/posts/online-shopping-tips,2025-06-30,I compared three "
        "Myntra orders and the sizing chart was wrong every single time,,,,useful for trend "
        "filtering",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _files(glob: str) -> list:
    """Every CSV to read, skipping the template and working files."""
    found = []
    for path in sorted(MANUAL_DIR.glob(glob)):
        if path.name == TEMPLATE or path.name.startswith("_"):
            continue
        if path.name.startswith("."):
            continue
        found.append(path)
    return found


def read_csv(path) -> list[dict]:
    """Read one manual CSV into raw_posts rows, validating the header first."""
    with open(path, newline="", encoding="utf-8-sig") as fh:      # utf-8-sig strips the BOM
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ManualFormatError(f"{path.name} is empty, so it has no header row")
        header = {f.strip().lower() for f in reader.fieldnames if f}
        missing = [c for c in REQUIRED if c not in header]
        if missing:
            raise ManualFormatError(
                f"{path.name} is missing the column(s): {', '.join(missing)}\n"
                f"  found instead: {', '.join(sorted(header))}\n"
                f"  needed: {', '.join(REQUIRED)}\n"
                f"  copy data/manual/{TEMPLATE} and paste your posts under its header")

        rows = []
        for line in reader:
            clean = {(k or "").strip().lower(): (v or "") for k, v in line.items() if k}
            text = clean.get("text", "").strip()
            if not text or MEANINGLESS.match(text):
                continue
            source = _slug(clean.get("source_name", ""))
            # Hash the text rather than the URL: the same post pasted from two
            # different links should land once, not twice.
            text_key = hashlib.sha256(f"{source}::{text}".encode()).hexdigest()[:16]
            engagement = _number(clean.get("engagement", ""))
            rows.append({
                "post_id": f"manual:{source}:{text_key}",
                "source": "manual",
                # The hand-copied source name is the context, the same way a subreddit
                # or a video title is for the other collectors.
                "myntra_explicit": mentions_myntra(text),
                "parent_context": clean.get("source_name", "").strip() or source,
                "author_hash": hash_author(clean.get("author", "")),
                "date": _parse_date(clean.get("date", "")),
                "text": text,
                "rating": _number(clean.get("rating", "")),
                "engagement": engagement,
                "url": clean.get("url", "").strip(),
                "lang_hint": lang_hint(text),
                "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "raw_file": None,
            })
        return rows


def collect(cfg: dict, progress=None) -> list[dict]:
    mc = cfg["manual"]
    ensure_template()
    paths = _files(mc.get("glob", "*.csv"))

    if not paths:
        print(f"   no CSVs to read in data/manual/ yet.")
        print(f"   copy {TEMPLATE}, fill it in, and run this again.")
        print(f"   columns needed: {', '.join(REQUIRED)}")
        return []

    all_rows: list[dict] = []
    seen: set[str] = set()
    duplicates = 0
    blank = 0

    for path in paths:
        try:
            rows = read_csv(path)
        except ManualFormatError as exc:
            raise                      # a bad file must stop the run, not be half-imported
        kept = 0
        for row in rows:
            if row["post_id"] in seen:
                duplicates += 1
                continue
            seen.add(row["post_id"])
            all_rows.append(row)
            kept += 1
        print(f"   {path.name:38} {kept:4} posts")
        if progress:
            progress(len(all_rows), 0, 0, 0)

    if not all_rows:
        raise RuntimeError(
            "every CSV in data/manual/ was blank or held only placeholder text")

    print(f"   {duplicates} duplicate post(s) skipped")
    missing_date = sum(1 for r in all_rows if not r["date"])
    if missing_date:
        print(f"   {missing_date} post(s) had no readable date and were kept with date=None")

    by_source: dict[str, int] = {}
    for row in all_rows:
        by_source[row["parent_context"].lower()] = by_source.get(
            row["parent_context"].lower(), 0) + 1
    print("   per source: " + ", ".join(f"{k} {v}" for k, v in sorted(by_source.items())))

    archive = save_raw(all_rows, "manual")
    return archive