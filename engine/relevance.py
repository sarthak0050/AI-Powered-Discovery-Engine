"""Module 2, step 2: the LLM relevance filter.

Reads clean_posts, sends batches of 20 to the model with prompts/relevance.txt, and
writes the judgement into relevant_posts. The same call does the Hinglish
translation, so text and text_en are decided together and cannot disagree.

Two properties this stage is built around:

  - Resumable. Every judgement is written as soon as its batch returns, and every call
    is cached by prompt hash, so a re-run costs nothing and a crash loses at most one
    batch. At 64k posts a full run needs about 3,200 calls, which is more than one
    day's free quota, so stopping and resuming is the normal case rather than an edge
    case.

  - Auditable. Nothing is deleted. relevant_posts holds every post with is_relevant set
    to 0 or 1, so a wrong judgement is one UPDATE away from being corrected.

Relevance precision is measured from your own labels in data/labels/relevance_check.csv,
never from the model's own opinion of itself.
"""

import csv
import json
import random
import re
import sys

from pydantic import BaseModel, Field

from engine import db
from engine.llm import LLM, LLMError, QuotaExhausted, render_prompt

PROMPT = db.ROOT / "prompts" / "relevance.txt"
LABEL_EXPORT = db.ROOT / "data" / "labels" / "relevance_check.csv"
# Same rows, shuffled, with the engine's verdict stripped out. You label this one so
# your answers are not anchored by what the engine said.
LABEL_BLIND = db.ROOT / "data" / "labels" / "relevance_check_blind.csv"

# Truncate very long posts before sending them. A 5,000-word YouTube essay would cost
# a fortune in tokens and adds nothing to a relevance decision that a sentence answers.
MAX_CHARS = 1200

DEVAANAGARI = re.compile("[\u0900-\u097F]")

# Romanised Hindi written with Latin letters, which is how most of this corpus is
# written. These are the markers, chosen because they are not ordinary English words,
# so an English review containing "link" or "order" is not mistaken for Hinglish.
HINGLISH_MARKERS = re.compile(
    r"\b(karo|kya|kis|kis|kahan|hai|hain|nahi|nhi|karna|karna|krna|krwa|kar|raha|rahi|rahe|"
    r"chahiye|waala|wali|bhai|yaar|accha|khub|matlab|abhi|jaldi|jldi|pichli|dusra|bhi|hi|"
    r"kripya|dhanyavaad|shukriya|nahi|kaisa|kaisi|kyu|kyun|kyunki|isliye|phir|bhi)\b"
    r"|\b(dijiye|dejiye|bhejo|bhej|kro|krdo|krdijiye)\b",
    re.IGNORECASE)


def looks_non_english(text: str) -> bool:
    """True when the post needs translating, whether in Devanagari or Romanised Hindi.

    Used to decide whether a model's text_en is worth keeping. A model asked to
    translate plain English will sometimes paraphrase it anyway, and a paraphrase of
    the original stored next to it is worse than no translation at all.
    """
    text = text or ""
    if DEVAANAGARI.search(text):
        return True
    # Hinglish only counts when the Romanised markers outnumber the ordinary English
    # ones, otherwise a single "order" in an English review trips the test.
    hits = len(HINGLISH_MARKERS.findall(text))
    words = max(len(text.split()), 1)
    return hits >= 2 or (hits >= 1 and words <= 12)


class Judgement(BaseModel):
    id: int
    text_en: str | None = None
    is_relevant: bool
    relevance_reason: str = Field(max_length=240)
    relevance_confidence: float = Field(ge=0, le=1)


class BatchJudgement(BaseModel):
    results: list[Judgement]


def load_posts(conn, limit: int | None = None, only_unjudged: bool = True,
               only_judged_by: str | None = None) -> list:
    """Posts waiting to be judged.

    With a limit, the sample is spread across sources and dates rather than taking the
    first N rows. The first N are all from whichever source happens to be earliest, so a
    pilot judged that way tells you how the model behaves on Reddit and nothing about
    how it behaves on app reviews.

    only_judged_by narrows to posts a particular provider judged, so they can be
    re-judged by the main provider. --rejudge on its own re-judges everything, which is
    the right thing after a prompt change but an expensive mistake when the only rows
    that need redoing are the handful groq answered.
    """
    if only_judged_by:
        where = f"WHERE c.judged_by = '{only_judged_by}'"
    elif only_unjudged:
        where = "WHERE c.is_relevant IS NULL"
    else:
        where = ""
    if not limit:
        return conn.execute(
            f"SELECT c.*, r.text AS text_original FROM clean_posts c "
            f"JOIN raw_posts r ON r.post_id = c.post_id {where} "
            f"ORDER BY c.date ASC").fetchall()

    # Oldest first within each source, then round-robin across sources. Every source
    # contributes what it has rather than the largest one contributing everything.
    by_source: dict[str, list] = {}
    for row in conn.execute(
            f"SELECT c.*, r.text AS text_original FROM clean_posts c "
            f"JOIN raw_posts r ON r.post_id = c.post_id {where} "
            f"ORDER BY c.date ASC"):
        by_source.setdefault(row["source"], []).append(row)

    picked, sources = [], sorted(by_source)
    index = 0
    while len(picked) < limit and any(by_source.values()):
        source = sources[index % len(sources)]
        if by_source[source]:
            picked.append(by_source[source].pop(0))
        index += 1
        if index > limit * len(sources) * 4:
            break
    return picked


def judge(conn, posts: list, progress=None) -> tuple[int, int]:
    """Send the posts through the model. Returns (kept, dropped)."""
    template = PROMPT.read_text(encoding="utf-8")
    llm = LLM(stage="relevance")

    items = [{"id": i + 1,
              "text": " ".join((p["text_clean"] or "").split())[:MAX_CHARS]}
             for i, p in enumerate(posts)]
    by_id = {i + 1: p for i, p in enumerate(posts)}

    kept = dropped = failed = 0
    batches = call_json_batched(llm, template, items, BatchJudgement, by_id)

    for index, (batch_index, chunk, outcome, stats) in enumerate(batches):
        if isinstance(outcome, QuotaExhausted):
            # call_json_batched returns after yielding this, so nothing is lost and
            # everything already committed stays committed.
            break
        if isinstance(outcome, Exception):
            failed += len(chunk)
            # call_json already wrote one row per real post_id for an LLMError, so
            # logging again here would only duplicate it. Anything else (a bug in this
            # file rather than a failed API call) has not been recorded yet.
            if not isinstance(outcome, LLMError):
                for item in chunk:
                    row = by_id.get(item["id"])
                    if row:
                        db.log_error(conn, "relevance",
                                     f"batch {batch_index}: {outcome}"[:400],
                                     post_id=row["post_id"])
            print(f"   batch {batch_index} failed: {str(outcome)[:80]}")
            conn.commit()
            continue

        for result in outcome.results:
            row = by_id.get(result.id)
            if row is None:
                continue                      # the model invented an id; ignore it
            is_relevant = 1 if result.is_relevant else 0
            # text_en is only trusted when the post actually looks non-English. A model
            # asked to translate English will sometimes paraphrase it anyway, and a
            # paraphrased original is worse than none.
            #
            # The test has to be the post text, not lang_hint. lang_hint is set at
            # collection time by checking for the Devanagari script only, so it says "en"
            # for "Mam myntra glam clan open nhi ho rha" and the translation was being
            # thrown away. That lost 22 of the 23 non-English posts in the first pilot.
            text_en = (result.text_en or "").strip() or None
            if text_en and not looks_non_english(row["text_clean"] or ""):
                text_en = None
            # judged_by and model both come from the provider that actually answered
            # this batch, not the configured primary. On a daily-quota 429 the batch is
            # handed to groq, and that has to be visible in the data afterwards.
            answered_by = (stats or {}).get("provider") or llm._provider
            answered_model = (stats or {}).get("model") or llm.model
            model_name = f"{answered_by}/{answered_model}"
            conn.execute(
                "UPDATE clean_posts SET text_en=?, is_relevant=?, relevance_reason=?, "
                "relevance_confidence=?, model=?, judged_by=?, judged_at=? WHERE post_id=?",
                (text_en, is_relevant, result.relevance_reason,
                 float(result.relevance_confidence), model_name, answered_by,
                 db.utc_now(), row["post_id"]))
            kept += is_relevant
            dropped += not is_relevant
        conn.commit()

        if progress:
            progress(index * len(chunk), len(posts), 0, 0)
        # Every 25 batches rather than every batch, so the log stays readable over a
        # 1,700-batch run and a long run can be watched while it works.
        if index % 25 == 0 or index == len(items) // max(
                int((llm.config.get("batch") or {}).get("relevance", 20)) or 20, 1):
            print(f"   batch {index:4}  {len(chunk):3} posts judged  "
                  f"(running: {kept} relevant, {dropped} not)")
            sys.stdout.flush()

    return kept, dropped


def call_json_batched(llm, template, items, schema, by_id=None):
    """Yield (batch_index, chunk, parsed_or_exception) one batch at a time.

    This is a generator on purpose. Collecting every batch into a list first meant
    nothing was written to the database until all ~1,700 calls had finished, so a
    crash or a quota stop lost the whole run and progress looked frozen. Yielding
    lets the caller commit each batch as it lands.

    by_id maps the prompt's 1..N numbering to the rows behind it, so a batch that
    fails is recorded against real post_ids rather than numbers that mean nothing
    outside this run. Without it the prompt numbering is logged as before.
    """
    batch_size = int((llm.config.get("batch") or {}).get("relevance", 20)) or 20
    chunks = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    for index, chunk in enumerate(chunks, start=1):
        try:
            real_ids = [by_id[i["id"]]["post_id"] for i in chunk
                        if by_id and i["id"] in by_id]
            parsed, stats = llm.call_json(
                render_prompt(template, {"posts": json.dumps(chunk, ensure_ascii=False)}),
                schema, input_id=",".join(str(i["id"]) for i in chunk),
                post_ids=real_ids or None)
            # stats carries the provider that actually answered (there is no fallback
            # now, but judged_by must still come from stats, never from the configured
            # primary).
            yield index, chunk, parsed, stats
        except QuotaExhausted as exc:
            # Out of calls for today. Stop cleanly and keep everything judged so far.
            yield index, chunk, exc, None
            print(f"\n   daily quota reached after {index} batches. "
                  f"Re-run the same command tomorrow to continue; finished batches are kept.")
            return
        except Exception as exc:                       # noqa: BLE001 - recorded below
            yield index, chunk, exc, None


def build_relevant_table(conn) -> int:
    """Copy the judged rows into relevant_posts (AGENTS.md section 6).

    That table stores only the judgement fields, so it is filled with those alone.
    Source, date and the original text stay reachable by joining raw_posts on post_id,
    which keeps one copy of every post rather than two that can drift apart.
    """
    rows = conn.execute(
        "SELECT post_id, text_clean, text_en, is_relevant, relevance_reason, "
        "relevance_confidence, model, judged_by, judged_at FROM clean_posts "
        "WHERE is_relevant IS NOT NULL").fetchall()
    conn.execute("DELETE FROM relevant_posts")
    if rows:
        db.upsert(conn, "relevant_posts", [dict(r) for r in rows])
    return len(rows)


def export_label_check(conn, kept: int = 50, dropped: int = 50, seed: int = 7) -> int:
    """Write kept and dropped posts for your own review, to measure real precision.

    Sampling is stratified by source so the file is not dominated by whichever source
    happens to have more rows, and the seed is fixed so a re-run produces the same file
    and your judgements stay attached to the same posts.
    """
    rng = random.Random(seed)
    chosen = []
    for flag, want in ((1, kept), (0, dropped)):
        by_source: dict[str, list] = {}
        for row in conn.execute(
                "SELECT c.*, r.source, r.text AS text_original FROM clean_posts c "
                "JOIN raw_posts r ON r.post_id = c.post_id WHERE c.is_relevant = ?",
                (flag,)):
            by_source.setdefault(row["source"], []).append(dict(row))

        # Round-robin across sources, so a rare source is represented before a common
        # one is exhausted.
        pool = {src: rng.sample(rows, len(rows)) for src, rows in by_source.items()}
        order = sorted(pool, key=lambda s: -len(pool[s]))
        picked = []
        index = 0
        while len(picked) < want and any(pool.values()):
            src = order[index % len(order)]
            if pool[src]:
                picked.append(pool[src].pop())
            index += 1
            if index > want * len(order) * 4:
                break
        chosen.extend((flag, r) for r in picked[:want])

    LABEL_EXPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(LABEL_EXPORT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["post_id", "source", "date", "engine_said_relevant",
                         "relevance_confidence", "relevance_reason",
                         "text_original", "text_en", "my_judgement"])
        for flag, row in chosen:
            writer.writerow([row["post_id"], row["source"], row["date"], flag,
                             row["relevance_confidence"], row["relevance_reason"],
                             " ".join((row["text_original"] or "").split()),
                             row["text_en"] or "", ""])

    # Blind copy: the same rows, shuffled, carrying only an opaque id, the text and an
    # empty verdict. Written here rather than by hand so the two files cannot drift —
    # the id is derived from post_id, so your answers always land on the same post.
    blind = list(chosen)
    random.Random(seed + 1).shuffle(blind)
    with open(LABEL_BLIND, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "text", "my_judgement"])
        for _flag, row in blind:
            writer.writerow([db.label_id(row["post_id"]),
                             " ".join((row["text_original"] or "").split()), ""])
    return len(chosen)


def report(conn) -> None:
    rows = conn.execute(
        "SELECT r.source, COUNT(*) AS n, SUM(c.is_relevant) AS rel FROM clean_posts c "
        "JOIN raw_posts r ON r.post_id = c.post_id WHERE c.is_relevant IS NOT NULL "
        "GROUP BY r.source ORDER BY n DESC").fetchall()
    print("\n   relevance by source")
    print(f"     {'source':12} {'judged':>7} {'relevant':>9} {'rate':>7}")
    for row in rows:
        rate = (row["rel"] / row["n"] * 100) if row["n"] else 0
        print(f"     {row['source']:12} {row['n']:7} {row['rel'] or 0:9} {rate:6.1f}%")

    confidence = conn.execute(
        "SELECT AVG(relevance_confidence) AS avg_c FROM clean_posts "
        "WHERE is_relevant = 1").fetchone()
    unsure = conn.execute(
        "SELECT COUNT(*) AS n FROM clean_posts "
        "WHERE is_relevant IS NOT NULL AND relevance_confidence < 0.6").fetchone()
    total = conn.execute(
        "SELECT COUNT(*) AS n FROM clean_posts WHERE is_relevant IS NOT NULL").fetchone()
    print(f"\n   mean confidence on kept posts: {(confidence['avg_c'] or 0):.2f}")
    print(f"   kept posts below 0.6 confidence: {unsure['n']} of {total['n']}")


def show_samples(conn, n: int = 10) -> None:
    for flag, label in ((1, "KEPT as relevant"), (0, "DROPPED as not relevant")):
        rows = conn.execute(
            "SELECT r.source, r.date, c.text_clean, c.text_en, c.relevance_reason, "
            "c.relevance_confidence FROM clean_posts c "
            "JOIN raw_posts r ON r.post_id = c.post_id WHERE c.is_relevant = ? "
            "ORDER BY RANDOM() LIMIT ?", (flag, n)).fetchall()
        print(f"\n   {n} random {label}:")
        for row in rows:
            text = " ".join((row["text_clean"] or "").split())
            print(f"     [{row['source']} | {str(row['date'])[:10]} | "
                  f"conf {row['relevance_confidence']:.2f}] {text[:78]}")
            if row["text_en"]:
                print(f"        EN: {row['text_en'][:74]}")
            print(f"        why: {row['relevance_reason'][:74]}")


def run(argv=None, **kwargs):
    """Entry point used by run.py. argv is the parsed argparse namespace."""
    limit = getattr(argv, "limit", None)
    sample = getattr(argv, "sample", 10)
    # --rejudge exists so a prompt change can be applied to posts already judged;
    # without it those posts would keep their old answer forever.
    only_unjudged = not getattr(argv, "rejudge", False)
    conn = db.connect()

    if db.count_rows(conn, "clean_posts") == 0:
        print("\n   clean_posts is empty. Run: python run.py clean\n")
        conn.close()
        return 1

    print("\n== relevance")
    # --rejudge-by groq means "redo only what groq judged", which is a couple of
    # hundred rows, not the whole corpus.
    rejudge_by = getattr(argv, "rejudge_by", None)
    if rejudge_by:
        only_unjudged = False
    posts = load_posts(conn, limit=limit, only_unjudged=only_unjudged,
                       only_judged_by=rejudge_by)
    if rejudge_by:
        print(f"   re-judging only the {len(posts)} posts judged by {rejudge_by}")
    if not posts:
        already = conn.execute(
            "SELECT COUNT(*) AS n FROM clean_posts WHERE is_relevant IS NOT NULL").fetchone()["n"]
        print(f"   nothing new to judge. {already} posts already judged.")
    else:
        remaining = conn.execute(
            "SELECT COUNT(*) AS n FROM clean_posts WHERE is_relevant IS NULL").fetchone()["n"]
        batch = int((LLM(stage="relevance").config.get("batch") or {}).get("relevance", 20))
        print(f"   {len(posts)} posts to judge in batches of {batch}"
              f"  ({remaining} unjudged in total)")
        kept, dropped = judge(conn, posts, progress=kwargs.get("progress"))
        print(f"\n   {kept} relevant, {dropped} not relevant")

    total = build_relevant_table(conn)
    conn.commit()
    print(f"   relevant_posts now holds {total} rows")

    # Joined on the way out so the CSV you open has the source and date next to the
    # text, which is what makes it reviewable.
    db.export_csv([dict(r) for r in conn.execute(
        "SELECT p.*, r.source, r.date, r.myntra_explicit, r.parent_context, "
        "r.text AS text_original, r.rating, r.engagement, r.url FROM relevant_posts p "
        "JOIN raw_posts r ON r.post_id = p.post_id ORDER BY r.date DESC")],
        db.ROOT / "data" / "exports" / "relevant_posts.csv")
    db.log_run(conn, "relevance", "all", total,
               conn.execute("SELECT COUNT(*) AS n FROM relevant_posts "
                            "WHERE is_relevant = 1").fetchone()["n"],
               "-> relevant_posts.csv")
    conn.commit()

    report(conn)
    show_samples(conn, n=sample)

    picked = export_label_check(conn)
    print(f"\n   {picked} posts written to data/labels/relevance_check.csv")
    print(f"   ...and the same {picked}, shuffled and without the verdict, to")
    print("   data/labels/relevance_check_blind.csv")
    print("   Judge them with: python run.py label-relevance")
    conn.close()
    return 0