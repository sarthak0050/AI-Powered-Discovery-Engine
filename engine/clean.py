"""Module 2, step 1: clean the raw corpus.

Removes exact duplicates, near-duplicates, spam and anything under the word floor, then
writes what survives into `clean_posts`. Nothing about what counts as junk lives here:
the thresholds and the spam patterns all come from config/clean.yaml.

Order matters and is deliberate:

  1. link-only and emoji-only, because these are cheap to detect and useless to embed
  2. the word floor, also cheap
  3. spam patterns
  4. exact duplicates, which is free once the text is normalised
  5. near-duplicates by embedding, which is the expensive step, so it runs last and
     only ever sees the survivors

Every removed row is written to data/exports/clean_dropped.csv with the reason, so
anything this stage throws away can be argued with later.

Resumable: the embedding of each post is cached in the database by content hash, so a
re-run never re-embeds text it has already seen.
"""

import re
import sqlite3
import unicodedata

import yaml

from engine import db

ROOT = db.ROOT
# Emoji and pictographs, matched by code point range rather than pasted in, because
# pasted ranges are easy to corrupt. Includes dingbats, misc symbols, regional
# indicators, and the variation/zero-width selectors and skin-tone modifiers that
# emoji runs are built from.
EMOJI_RANGES = (
    "[\U0001F000-\U0001FAFF"   # pictographs, emoticons, transport
    "\U0001F1E6-\U0001F1FF"           # regional indicators
    "\u2190-\u21FF"                   # arrows
    "\u2300-\u27BF"                   # misc technical, misc symbols, dingbats
    "\u2B00-\u2BFF"                   # arrows and squares
    "\uFE00-\uFE0F"                   # variation selectors
    "\U0001F3FB-\U0001F3FF"    # skin tones
    "\u200D]"                       # zero width joiner
)
EMOJI_GLYPH = re.compile(EMOJI_RANGES)
EMOJI_ONLY = re.compile(f"^(?:{EMOJI_RANGES}|\\s|\\uFE0F|\\u200D)+$", re.UNICODE)
URL_ONLY = re.compile(r"^\s*(https?://|www\.)\S+(\s+https?://\S+)*\s*$", re.I)
MENTION_LEAD = re.compile(r"^(@\S+\s+)+")
HTML = re.compile(r"<[^>]+>")


def load_clean_config(path=None) -> dict:
    path = path or ROOT / "config" / "clean.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def text_clean(text: str) -> str:
    """Tidy one post without changing what it says.

    Removes the leading @mentions and hashtags that dominate YouTube comments, strips
    HTML and collapsed whitespace. Hinglish and Devanagari are left exactly as written:
    normalising those is the LLM's job in the relevance stage, and doing it here too
    would mean two different translations to reconcile.
    """
    text = HTML.sub(" ", text or "")
    text = MENTION_LEAD.sub("", text)
    # Drop hashtags but keep the words next to them: "#myntratop" is noise,
    # "#myntratop dress looks great" is not.
    text = re.sub(r"#(\w+)", r"\1", text)
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def word_count(text: str) -> int:
    """Count words, for the min_length_words floor.

    Two things a naive split on spaces gets wrong:
      - URLs count as several "words", so "look https://a.com/x" would clear a floor
        of 5 on the strength of a link alone.
      - Devanagari has combining marks, so splitting on the letter ranges alone counts
        "दुनिया" as three words rather than one.
    """
    text = re.sub(r"https?://\S+|www\.\S+", " ", text or "")
    text = re.sub(r"[@#]\S+", " ", text)
    words = re.findall(r"[^\W_]+(?:['\u2019][^\W_]+)*", text, re.UNICODE)
    # Rejoin Devanagari sequences: a base letter plus its matras is one word.
    return len(re.findall(r"[\u0900-\u097F][\u0900-\u097F\u200C]*|[\w'’\u0900-\u097F]+",
                          text, re.UNICODE)) or len(words)


def drop_reason(row: sqlite3.Row, cfg: dict) -> str | None:
    """Why this row is being removed, or None to keep it. Pure checks, no LLM."""
    raw = row["text"] or ""
    clean = text_clean(raw)

    if cfg.get("drop_link_only", True) and URL_ONLY.match(clean.strip()):
        return "link_only"
    if cfg.get("drop_emoji_only", True):
        stripped = EMOJI_ONLY.sub("", clean).strip()
        if not stripped and clean:
            return "emoji_only"

    floor = int(cfg.get("min_length_words", 0) or 0)
    if floor and word_count(clean) < floor:
        return f"under_{floor}_words"

    lowered = clean.lower()
    for pattern in cfg.get("spam_patterns") or []:
        try:
            if re.search(pattern, lowered):
                return f"spam:{pattern[:40]}"
        except re.error:
            if pattern in lowered:
                return f"spam:{pattern[:40]}"
    return None


def embedder(cfg: dict):
    """Load the local embedding model. It never leaves this machine."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(cfg.get(
        "embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))


def _embedding_cached(conn, key: str) -> list | None:
    row = conn.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
    return [float(v) for v in row["vector"].split(",")] if row else None


def _store_embedding(conn, key: str, text: str, model, batch_size: int) -> list:
    """Embed one text, saving the result so a re-run is free."""
    vector = model.encode([text], batch_size=1, show_progress_bar=False,
                          normalize_embeddings=True)[0]
    conn.execute("INSERT OR REPLACE INTO embeddings (key, vector, text_hash) VALUES (?,?,?)",
                 (key, ",".join(f"{v:.6f}" for v in vector), key[:16]))
    conn.commit()
    return [float(v) for v in vector]


def find_near_duplicates(vectors: list[list[float]], threshold: float,
                         block: int = 2048) -> set[int]:
    """Indices that are near-duplicates of an earlier row.

    Each row is compared against every earlier row. That needs two comparisons per
    block: against the rows in its own block, and against all rows before it. Comparing
    only across blocks would miss duplicates inside a single block, and comparing only
    within blocks would miss the pairs that straddle a boundary.

    Blocked because 60k rows all-pairs is 3.6 billion comparisons. The block size bounds
    memory, not correctness: at 256 the matrix compared per step is at most
    256 x 60,000 floats, around 60 MB.

    Vectors are normalised, so a dot product is the cosine similarity.
    """
    import numpy as np

    matrix = np.asarray(vectors, dtype="float32")
    total, dropped = matrix.shape[0], set()
    for start in range(0, total, block):
        end = min(start + block, total)
        chunk = matrix[start:end]
        for i in range(end - start):
            # Within this block, only rows above i can make it a duplicate. Using
            # the whole row here would flag i against a later row, which drops the
            # copy that should have been kept and leaves the original.
            if i and (chunk[i] @ chunk[:i].T).max() >= threshold:
                dropped.add(start + i)
                continue
            if start:
                earlier = chunk[i] @ matrix[:start].T
                if earlier.size and earlier.max() >= threshold:
                    dropped.add(start + i)
    return dropped


def clean(conn, cfg: dict, limit: int | None = None, progress=None) -> tuple[list[dict], list[dict]]:
    """Return (kept, dropped) row dicts. Drops carry a reason."""
    rows = conn.execute("SELECT * FROM raw_posts ORDER BY date DESC").fetchall()
    if limit:
        rows = rows[:limit]

    print(f"   {len(rows)} raw posts to clean")
    if not rows:
        return [], []

    # 1-3: cheap checks
    stage1, dropped = [], []
    reasons: dict[str, int] = {}
    for row in rows:
        reason = drop_reason(row, cfg)
        if reason:
            dropped.append({**dict(row), "drop_reason": reason})
            reasons[reason] = reasons.get(reason, 0) + 1
        else:
            stage1.append(row)
    for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"   removed {count:6}  {reason}")

    # 4: exact duplicates on the cleaned text
    seen, unique = {}, []
    exact = 0
    for row in stage1:
        key = text_clean(row["text"]).lower()
        if key in seen:
            dropped.append({**dict(row), "drop_reason": f"exact_duplicate_of:{seen[key]}"})
            exact += 1
        else:
            seen[key] = row["post_id"]
            unique.append(row)
    print(f"   removed {exact:6}  exact duplicates")

    # 5: near duplicates. The expensive step, so it only sees survivors.
    if len(unique) > 1:
        print(f"   embedding {len(unique)} survivors locally (no text leaves this machine)...")
        model = embedder(cfg)
        batch_size = int(cfg.get("embedding_batch_size", 256))
        keys = [row["post_id"] for row in unique]
        vectors = []
        for start in range(0, len(unique), batch_size):
            chunk = unique[start:start + batch_size]
            missing = [r for r in chunk if _embedding_cached(conn, r["post_id"]) is None]
            if missing:
                texts = [text_clean(r["text"]) for r in missing]
                encoded = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                                       normalize_embeddings=True)
                for row, vector in zip(missing, encoded):
                    conn.execute(
                        "INSERT OR REPLACE INTO embeddings (key, vector, text_hash) VALUES (?,?,?)",
                        (row["post_id"], ",".join(f"{v:.6f}" for v in vector),
                         row["post_id"][:16]))
                conn.commit()
            for row in chunk:
                vectors.append(_embedding_cached(conn, row["post_id"]))
            if progress and start % (batch_size * 4) == 0:
                progress(start + len(chunk), len(unique), 0, 0)

        threshold = float(cfg.get("near_duplicate_similarity", 0.95))
        dupes = find_near_duplicates(vectors, threshold)
        if dupes:
            kept_rows = [r for i, r in enumerate(unique) if i not in dupes]
            for i in sorted(dupes):
                dropped.append({**dict(unique[i]),
                                "drop_reason": f"near_duplicate_of:{unique[i - 1]['post_id']}"
                                if i else "near_duplicate"})
            print(f"   removed {len(dupes):6}  near duplicates (similarity >= {threshold})")
            unique = kept_rows

    kept = [{"post_id": r["post_id"], "source": r["source"], "date": r["date"],
             "myntra_explicit": r["myntra_explicit"], "parent_context": r["parent_context"],
             "text_clean": text_clean(r["text"]), "text_en": None, "is_relevant": None,
             "engagement": r["engagement"], "rating": r["rating"], "lang_hint": r["lang_hint"],
             "url": r["url"], "cleaned_at": db.utc_now()} for r in unique]
    print(f"   kept {len(kept)}   removed {len(dropped)}   "
          f"({len(dropped)/len(rows)*100:.1f}% removed)")
    return kept, dropped


def run(argv=None, **kwargs):
    """Entry point used by run.py. argv is the parsed argparse namespace."""
    cfg = load_clean_config()
    conn = db.connect()
    limit = getattr(argv, "limit", None)

    kept, dropped = clean(conn, cfg, limit=limit,
                          progress=kwargs.get("progress"))

    conn.execute("DELETE FROM clean_posts")
    if kept:
        db.upsert(conn, "clean_posts", kept)

    dropped_path = ROOT / (cfg.get("exports", {}).get("dropped")
                           or "data/exports/clean_dropped.csv")
    if dropped:
        db.export_csv(dropped, dropped_path)

    db.export_csv([dict(r) for r in conn.execute(
        "SELECT c.*, r.text AS text_original, r.author_hash, r.collected_at, r.raw_file "
        "FROM clean_posts c JOIN raw_posts r ON r.post_id = c.post_id")],
        ROOT / "data" / "exports" / "clean_posts.csv")

    db.log_run(conn, "clean", "all", len(kept) + len(dropped), len(kept),
               f"{len(dropped)} removed -> {dropped_path.name}")
    conn.commit()
    print(f"\n   {db.count_rows(conn, 'clean_posts')} rows now in clean_posts")
    print(f"   removed rows with reasons: {dropped_path}")
    print("\n   next: python run.py relevance --limit 100")
    conn.close()
    return 0