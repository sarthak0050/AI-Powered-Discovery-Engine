"""Stage 2 of the relevance filter: the rubric-based in_scope check.

Stage 1 (prompts/relevance.txt) is deliberately broad — it keeps anything that might
describe a fashion buying journey, because a wrong drop is invisible downstream. This
check applies the locked rule in docs/labelling_guide.md (the base rule plus the
pure-complaint exception) to the posts stage 1 kept. It is the first field of the
extraction schema, config/schema.yaml.

A post that fails is not deleted: the verdict lands in tagged_posts, the row stays in
the database, and every analysis table reads WHERE in_scope = 1.

The rule itself is never copied into this file or into a prompt. It is read out of
docs/labelling_guide.md at run time, so "defined exactly by the locked rubric" means
the rubric text itself, and a change to the guide is a change to this check.

Run:
    python run.py in-scope                the 50 kept rows of relevance_check.csv
    python run.py in-scope --pilot        draw the 300 Module 3 posts, check them first
    python run.py in-scope --force        re-check rows that already have a verdict
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter

from pydantic import BaseModel, Field

from engine import db, llm

GUIDE = db.ROOT / "docs" / "labelling_guide.md"
PROMPT = db.ROOT / "prompts" / "in_scope.txt"
CHECK = db.ROOT / "data" / "labels" / "relevance_check.csv"
PILOT_CSV = db.ROOT / "data" / "exports" / "pilot_sample.csv"

MAX_CHARS = 1200
PILOT_SIZE = 300
SEED = 7

IN_SCOPE = "in_scope"
IN_SCOPE_REASON = "in_scope_reason"
IN_SCOPE_MODEL = "in_scope_model"
IN_SCOPE_STATUS = "in_scope_status"
VALID = "valid"
INVALID = "INVALID"


def expected_model() -> str:
    """The provider/model this stage is configured to use.

    A verdict answered by anything else is marked INVALID in in_scope_status: the
    check is only a measurement when it runs on the provider config named for it,
    and one answered by the judge's own model cannot be quoted as a validation.
    """
    client = llm.LLM(stage="in_scope")
    return f"{client._provider}/{client.model}"


class ScopeFlag(BaseModel):
    value: bool
    reason: str = Field(max_length=240)


class ScopeItem(BaseModel):
    id: int
    in_scope: ScopeFlag


class ScopeBatch(BaseModel):
    results: list[ScopeItem]


def scope_rule() -> str:
    """The locked rubric: everything from '## The rule' to the first worked example.

    That slice holds the base rule, the journey list, the short-post note, the
    pure-complaint exception, the "links to a buying decision" list and the returns
    section — and stops before the worked examples, which quote posts this check may
    itself be deciding. Injecting the guide's own text rather than a copy of it is the
    point: if the two ever disagree, the guide wins.
    """
    guide = GUIDE.read_text(encoding="utf-8")
    start = guide.find("## The rule")
    end = guide.find("## 3 examples that are NOT relevant")
    if start < 0 or end <= start:
        raise SystemExit(
            f"{GUIDE.name} no longer has a '## The rule' section before the worked "
            f"examples — the in_scope check cannot find the rubric.")
    rule = guide[start:end].strip().rstrip("- \n").strip()
    for marker in ("A post is relevant if it describes any part of a fashion buying journey",
                   "A pure post-purchase complaint is NOT relevant"):
        if marker not in rule:
            raise SystemExit(
                f"{GUIDE.name} has changed: the rubric no longer contains {marker!r}. "
                f"The in_scope check would silently use a different rule — nothing run.")
    return rule


def _text(row: dict) -> str:
    """The original text, which is what the guide's worked examples quote."""
    text = " ".join((row.get("text_original") or row.get("text_en") or "").split())
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rstrip() + " …[truncated]"
    return text


def _answered_label(configured: str, answered: dict[str, int]) -> str:
    """Who actually answered — the configured model when it did, else the answerer."""
    if not answered:
        return configured
    if len(answered) == 1:
        return next(iter(answered))
    return "+".join(sorted(answered))


def run_check(items: list[dict], on_batch) -> tuple[int, str]:
    """Apply the locked rubric to items, in batches, calling on_batch after each one.

    items: [{"id": int, "text": str, "post_id": str}] — ids are positions in the
    caller's own file or sample, so a batch's cache key is stable and a resume never
    re-bills a finished call. on_batch(result_by_id, chunk, model) writes each batch as
    it lands, which is what makes a crash or a quota stop lose at most one batch.
    """
    if not items:
        return 0, ""
    template = PROMPT.read_text(encoding="utf-8")
    rule = scope_rule()
    client = llm.LLM(stage="in_scope")
    batch_size = int((client.config.get("batch") or {}).get("in_scope", 20)) or 20
    model = f"{client._provider}/{client.model}"
    print(f"   in_scope: {client.describe()}   batches of {batch_size}   "
          f"rubric {len(rule):,} chars")

    done = 0
    answered: dict[str, int] = {}
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        payload = [{"id": i["id"], "text": i["text"]} for i in chunk]
        prompt = llm.render_prompt(
            template, {"rubric": rule, "posts": json.dumps(payload, ensure_ascii=False)})
        input_id = ",".join(str(i["id"]) for i in chunk)
        post_ids = [i["post_id"] for i in chunk if i.get("post_id")] or None
        try:
            parsed, stats = client.call_json(
                prompt, ScopeBatch, input_id=input_id, post_ids=post_ids)
        except (llm.LLMError, llm.QuotaExhausted) as exc:
            print(f"   stopped: {exc}")
            print(f"   {done} row(s) written. Re-run the same command to continue.")
            return done, _answered_label(model, answered)

        who = f"{stats['provider']}/{stats['model']}"
        answered[who] = answered.get(who, 0) + 1
        if who != model:
            print(f"   WARNING: {model} unavailable — this batch was answered by {who}.")

        got = {r.id: r for r in parsed.results}
        missing = [i["id"] for i in chunk if i["id"] not in got]
        if missing:
            raise SystemExit(
                f"in_scope returned no answer for id(s) {missing}. Nothing was written "
                f"for this batch — re-run the same command.")
        on_batch(got, chunk, who)
        done += len(chunk)
        print(f"   batch {start // batch_size + 1}: {len(chunk)} checked  "
              f"({done}/{len(items)})")
        sys.stdout.flush()
    label = _answered_label(model, answered)
    if label != model:
        print(f"   answered by: {label}  (configured: {model})")
    return done, label


def _report_sample(rows: list[dict]) -> None:
    """Precision after stage 2, measured against judge_label on the kept rows.

    Suppressed when any verdict came from the wrong provider. The reference answers
    are the judge's, and the judge is Groq: a Groq in_scope run grades the judge with
    its own model, so agreement would be circular. Those rows carry INVALID in
    in_scope_status and no number is printed until the rerun on the stage's own
    provider lands.
    """
    from engine import label_report as lr

    kept = [r for r in rows if r["engine_said_relevant"] == "1"]
    checked = [r for r in kept if str(r.get(IN_SCOPE) or "").strip() in {"0", "1"}]
    skipped = [r for r in kept if r not in checked]
    bad = [r for r in checked if (r.get(IN_SCOPE_STATUS) or "") != VALID]

    print()
    print("-" * 74)
    print(f"STAGE 2 — in_scope on the kept rows (rubric = {GUIDE.name})")
    print("-" * 74)
    print(f"  rows checked                     {len(checked)} of {len(kept)} kept")
    if bad:
        answered = sorted({(r.get(IN_SCOPE_MODEL) or "unknown") for r in bad})
        print(f"  VALIDATION INVALID               {len(bad)} of {len(checked)} verdicts "
              f"answered by {', '.join(answered)}")
        print("    The reference judge is the same model, so this would be the judge")
        print("    grading itself — not a measurement. Nothing is reported, and these")
        print(f"    rows stay marked INVALID until they are re-run on {expected_model()}:")
        print("      python run.py in-scope --force        (~3 calls, after the reset)")
        return
    if skipped:
        print(f"   {len(skipped)} kept row(s) have no in_scope verdict yet — "
              f"the numbers below cover the {len(checked)} that do.")

    conn = db.connect()
    corpus = lr.corpus_counts(conn)
    conn.close()
    annotated = lr.annotate(rows, source="judge")
    stage1 = lr.report(annotated)

    per_source = Counter(r["source"] for r in kept)
    est_pass = est_rel = est_false = 0.0
    pass_n = rel_n = dropped_n = 0
    for r in checked:
        weight = corpus.get((1, r["source"]), 0) / per_source.get(r["source"], 1)
        if r[IN_SCOPE] == "1":
            pass_n += 1
            est_pass += weight
            if r["mine"] == 1:
                rel_n += 1
                est_rel += weight
        else:
            dropped_n += 1
            if r["mine"] == 1:
                est_false += weight          # judge-relevant, excluded by stage 2
    precision2 = est_rel / est_pass if est_pass else 0.0
    raw2 = rel_n / pass_n if pass_n else 0.0
    false_excluded = [r for r in checked
                      if r[IN_SCOPE] == "0" and r["mine"] == 1]
    false_kept = [r for r in checked if r[IN_SCOPE] == "1" and r["mine"] == 0]

    print(f"  pass in_scope                    {pass_n}   drop {dropped_n}")
    print(f"  precision after stage 2 (raw)    {rel_n}/{pass_n} = {raw2*100:.0f}%")
    print(f"  precision after stage 2 (weighted to the kept mix)  {precision2*100:.1f}%")
    print(f"  stage 1 precision, same kept rows {stage1['precision']*100:.1f}%")
    print(f"  judge-relevant rows stage 2 dropped  {len(false_excluded)} of "
          f"{len(checked)}  (weights to about {est_false:,.0f} corpus rows)")
    print()
    print("  Rows where the two disagree:")
    disagreements = false_excluded + false_kept
    if not disagreements:
        print("    none — in_scope agrees with the judge on every checked row.")
    for r in sorted(disagreements, key=lambda x: x[IN_SCOPE], reverse=True):
        direction = "pass, judge says 0" if r[IN_SCOPE] == "1" else "drop, judge says 1"
        print(f"    {r['post_id']}   in_scope={r[IN_SCOPE]}  ({direction})")
        print(f"      in_scope — {r.get(IN_SCOPE_REASON, '')}")
        print(f"      judge    — {r.get('judge_reason', '')}")
    print()
    print("  METHODOLOGY — two-stage filtering.")
    print("    Stage 1 kept these rows broadly; stage 2 applies the locked rubric and")
    print("    drops out-of-scope posts, which stay in the database and are excluded")
    print("    from every analysis table. The reference is the judge's label from the")
    print("    same rubric, so this measures how consistently the rule is applied, not")
    print("    whether the rule is right.")


def run_sample(force: bool = False) -> int:
    """Check the rows stage 1 kept in relevance_check.csv, then score stage 2."""
    from engine.judge import read_rows, write_rows

    fieldnames, rows = read_rows()
    if "judge_label" not in fieldnames:
        raise SystemExit(
            f"{CHECK.name} has no judge_label column. Run: python run.py judge-relevance")
    for column in (IN_SCOPE, IN_SCOPE_REASON, IN_SCOPE_MODEL, IN_SCOPE_STATUS):
        if column not in fieldnames:
            fieldnames.append(column)
    for r in rows:
        r.setdefault(IN_SCOPE, "")
        r.setdefault(IN_SCOPE_REASON, "")
        r.setdefault(IN_SCOPE_MODEL, "")
        r.setdefault(IN_SCOPE_STATUS, "")

    def checked(r: dict) -> bool:
        return str(r.get(IN_SCOPE) or "").strip() in {"0", "1"}

    kept = [i for i, r in enumerate(rows) if r["engine_said_relevant"] == "1"]
    pending = [i for i in kept if force or not checked(rows[i])]

    print(f"\n== in-scope: {len(pending)} of {len(kept)} kept rows to check "
          f"({len(rows)} rows in {CHECK.name})")
    if pending:
        expected = expected_model()
        items = [{"id": p + 1, "text": _text(rows[p]), "post_id": rows[p]["post_id"]}
                 for p in pending]

        def on_batch(got, chunk, who):
            for item in chunk:
                answer = got[item["id"]]
                row = rows[item["id"] - 1]
                row[IN_SCOPE] = "1" if answer.in_scope.value else "0"
                row[IN_SCOPE_REASON] = answer.in_scope.reason
                row[IN_SCOPE_MODEL] = who
                row[IN_SCOPE_STATUS] = VALID if who == expected else INVALID
                if who != expected:
                    print(f"   marked INVALID: answered by {who}, this stage is "
                          f"pinned to {expected}.")
            write_rows(fieldnames, rows)

        done, model = run_check(items, on_batch)
        conn = db.connect()
        db.log_run(conn, "in_scope", "relevance_check", len(items), done, notes=model)
        conn.close()
        if done < len(items):
            _report_sample(rows)
            return 1
    else:
        print("   all kept rows already checked (use --force to check them again)")

    _report_sample(rows)
    print(f"\n   verdicts written to {CHECK.name} as in_scope / in_scope_reason.")
    return 0


def _draw_pilot(conn, want: int = PILOT_SIZE, seed: int = SEED) -> list[dict]:
    """A fixed, source-balanced sample of the posts stage 1 kept.

    Round-robin across sources with a fixed seed: playstore is 70% of the kept pool,
    so taking the first N or sampling flat would hand the pilot a Reddit-free or
    app-store-free codebook. Same shape as the relevance label export.
    """
    rng = random.Random(seed)
    by_source: dict[str, list] = {}
    for row in conn.execute(
            "SELECT c.post_id, c.text_clean, r.text AS text_original, r.source, r.date "
            "FROM clean_posts c JOIN raw_posts r ON r.post_id = c.post_id "
            "WHERE c.is_relevant = 1 ORDER BY c.date ASC"):
        by_source.setdefault(row["source"], []).append(dict(row))
    pool = {src: rng.sample(rows, len(rows)) for src, rows in by_source.items()}
    order = sorted(pool, key=lambda s: -len(pool[s]))
    picked: list[dict] = []
    index = 0
    while len(picked) < want and any(pool.values()):
        source = order[index % len(order)]
        if pool[source]:
            picked.append(pool[source].pop())
        index += 1
        if index > want * len(order) * 4:
            break
    return picked


def _write_verdict(conn, post_id: str, value: int, reason: str, model: str) -> None:
    """Land the verdict in tagged_posts without disturbing any other column.

    UPDATE-then-INSERT rather than INSERT OR REPLACE: a REPLACE would blank the row
    for a post that extraction had already tagged in full.
    """
    now = db.utc_now()
    cur = conn.execute(
        "UPDATE tagged_posts SET in_scope=?, in_scope_reason=?, model=?, tagged_at=? "
        "WHERE post_id=?", (value, reason, model, now, post_id))
    if cur.rowcount == 0:
        conn.execute(
            "INSERT INTO tagged_posts (post_id, in_scope, in_scope_reason, model, tagged_at) "
            "VALUES (?,?,?,?,?)", (post_id, value, reason, model, now))
    conn.commit()


def _read_registry() -> list[dict]:
    """The rows already drawn and screened for the pilot (pilot_sample.csv)."""
    import csv
    if not PILOT_CSV.exists():
        return []
    with open(PILOT_CSV, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _topup_draws(pass_counts: dict[str, int], rates: dict[str, float],
                 cand_counts: dict[str, int], target_total: int) -> dict[str, int]:
    """How many fresh candidates to draw per source to reach target_total in-scope.

    Each source that can still grow asks for the same share of the missing passes
    it already contributes, converted to draws with its own pass rate — so a
    low-pass source like playstore is oversampled and keeps its share of the
    codebook instead of being crowded out by Reddit's higher pass rate.
    """
    need = max(0, target_total - sum(pass_counts.values()))
    if need <= 0:
        return {}
    usable = {s: cand_counts[s] for s in pass_counts if cand_counts.get(s, 0) > 0}
    if not usable:
        return {}
    share_total = sum(pass_counts.get(s, 0) for s in usable) or 1
    draws = {}
    for source, available in usable.items():
        expected = need * pass_counts.get(source, 0) / share_total
        draws[source] = min(available, int(round(expected / rates.get(source, 0.5))))
    return {s: n for s, n in draws.items() if n > 0}


def run_topup(force: bool = False) -> int:
    """Screen more posts so the pilot pool grows toward ~300 in-scope posts.

    Reads the existing registry (150 in-scope posts), measures each source's pass
    rate on it, then draws as many fresh candidates as each source needs to reach
    the target — oversampling low-pass sources. New verdicts land in tagged_posts
    and the new rows are appended to PILOT_CSV. Resumable: already-answered rows
    are skipped.
    """
    registry = _read_registry()
    if not registry:
        print("\n   no pilot registry yet. Run the first tranche first:\n"
              "       python run.py in-scope --pilot")
        return 1
    pass_counts = Counter(r["source"] for r in registry
                          if str(r.get("in_scope") or "").strip() == "1")
    drawn_counts = Counter(r["source"] for r in registry)
    rates = {s: pass_counts[s] / drawn_counts[s] for s in drawn_counts if drawn_counts[s]}
    print(f"\n== in-scope top-up: {sum(pass_counts.values())} in-scope now, "
          f"target {PILOT_SIZE}")
    for source in sorted(rates):
        print(f"   {source:11} pass {pass_counts[source]:>3}/{drawn_counts[source]:>3} "
              f"({rates[source]*100:.0f}%)")

    conn = db.connect()
    registered = {r["post_id"] for r in registry}
    by_source: dict[str, list] = {}
    for row in conn.execute(
            "SELECT c.post_id, c.text_clean, r.text AS text_original, r.source, r.date "
            "FROM clean_posts c JOIN raw_posts r ON r.post_id = c.post_id "
            "WHERE c.is_relevant = 1 ORDER BY c.date ASC"):
        d = dict(row)
        if d["post_id"] not in registered:
            by_source.setdefault(d["source"], []).append(d)
    cand_counts = {s: len(rows) for s, rows in by_source.items()}
    plan = _topup_draws(pass_counts, rates, cand_counts, PILOT_SIZE)
    if not plan:
        print("   no source has candidates left — the pool is already as big "
              "as the corpus allows.")
        conn.close()
        return 1
    for source, n in sorted(plan.items()):
        print(f"   draw {n:>4} {source:11} (~{n * rates[source] / 20:.1f} calls at 20/batch)")
    total = sum(plan.values())
    print(f"   = {total} candidates, ~{(total + 19) // 20} calls")

    drawn = [row for source in sorted(plan) for row in by_source[source][:plan[source]]]
    existing = {r["post_id"] for r in conn.execute(
        "SELECT post_id FROM tagged_posts WHERE in_scope IS NOT NULL")}
    pending = [d for d in drawn if force or d["post_id"] not in existing]
    if not pending:
        print("   all drawn rows already have a verdict (use --force to re-check them)")
    items = [{"id": i + 1, "text": _text(d), "post_id": d["post_id"]}
             for i, d in enumerate(pending)]
    new_rows: dict[str, dict] = {}

    def on_batch(got, chunk, who):
        for item in chunk:
            row = pending[item["id"] - 1]
            answer = got[item["id"]]
            value = 1 if answer.in_scope.value else 0
            _write_verdict(conn, row["post_id"], value, answer.in_scope.reason, who)
            new_rows[row["post_id"]] = {
                "post_id": row["post_id"], "source": row["source"], "date": row["date"],
                "in_scope": value, "in_scope_reason": answer.in_scope.reason,
                "in_scope_model": who,
                "text": " ".join((row["text_original"] or "").split()),
            }

    done, model = run_check(items, on_batch)
    db.log_run(conn, "in_scope", "pilot_topup", len(items), done, notes=model)
    if done < len(items):
        print(f"\n   {done} of {len(items)} checked. Re-run the same command to finish.")
        conn.close()
        return 1

    # Rebuild the registry from tagged_posts, not from this run's slices: a crash or
    # quota stop leaves finished verdicts in the database but the CSV only grows at
    # completion, so a resumed run would silently undercount the pool.
    export = [{
        "post_id": r["post_id"], "source": r["source"], "date": r["date"],
        "in_scope": r["in_scope"], "in_scope_reason": r["in_scope_reason"],
        "in_scope_model": r["in_scope_model"], "text": " ".join((r["text"] or "").split()),
    } for r in conn.execute(
        "SELECT t.post_id, r.source, r.date, t.in_scope, t.in_scope_reason, "
        "        t.model AS in_scope_model, r.text "
        "FROM tagged_posts t JOIN raw_posts r ON r.post_id = t.post_id "
        "WHERE t.in_scope IS NOT NULL ORDER BY t.post_id")]
    db.export_csv(export, PILOT_CSV)

    passed = sum(1 for r in export if str(r.get("in_scope") or "").strip() == "1")
    print(f"\n   in scope now: {passed} of {len(export)} registry rows")
    for source in sorted({r["source"] for r in export}):
        group = [r for r in export if r["source"] == source]
        ok = sum(1 for r in group if str(r.get("in_scope") or "").strip() == "1")
        print(f"     {source:11} {ok}/{len(group)}")
    print(f"\n   pilot pool = {passed} posts that pass the rubric")
    print("   Next: python run.py pilot  (open coding on the posts that passed)")
    conn.close()
    return 0


def run_pilot(force: bool = False) -> int:
    """Draw the 300 Module 3 posts and put them through the rubric before coding.

    The pilot builds the codebook, so a batch of pure complaints in it would teach the
    codebook the wrong categories. Checking first means open coding only ever sees
    posts that pass the same rule as everything downstream.
    """
    conn = db.connect()
    drawn = _draw_pilot(conn)
    if not drawn:
        print("\n   no relevant posts to sample. Run: python run.py relevance")
        conn.close()
        return 1

    verdicts: dict[str, tuple[int, str, str]] = {}
    if not force:
        for row in conn.execute(
                "SELECT post_id, in_scope, in_scope_reason, model FROM tagged_posts "
                "WHERE in_scope IS NOT NULL"):
            verdicts[row["post_id"]] = (row["in_scope"], row["in_scope_reason"] or "",
                                        row["model"] or "")

    pending = [d for d in drawn if d["post_id"] not in verdicts]
    sources = sorted({d["source"] for d in drawn})
    print(f"\n== in-scope pilot: {len(drawn)} posts drawn across "
          f"{len(sources)} sources (seed {SEED})")
    print(f"   {len(pending)} still to check, {len(drawn) - len(pending)} already done")
    if not pending and not force:
        print("   nothing to do (use --force to check them again)")
    else:
        wanted = {d["post_id"] for d in pending}
        items = [{"id": i + 1, "text": _text(drawn[i]), "post_id": drawn[i]["post_id"]}
                 for i, d in enumerate(drawn) if d["post_id"] in wanted]

        def on_batch(got, chunk, who):
            for item in chunk:
                answer = got[item["id"]]
                row = drawn[item["id"] - 1]
                value = 1 if answer.in_scope.value else 0
                verdicts[row["post_id"]] = (value, answer.in_scope.reason, who)
                _write_verdict(conn, row["post_id"], value, answer.in_scope.reason, who)

        done, model = run_check(items, on_batch)
        db.log_run(conn, "in_scope", "pilot", len(items), done, notes=model)
        if done < len(items):
            print(f"\n   {done} of {len(items)} checked. Re-run to finish.")
            conn.close()
            return 1

    rows = []
    for d in drawn:
        value, reason, who = verdicts.get(d["post_id"], (None, "", ""))
        rows.append({
            "post_id": d["post_id"], "source": d["source"], "date": d["date"],
            "in_scope": "" if value is None else value, "in_scope_reason": reason,
            "in_scope_model": who,
            "text": " ".join((d["text_original"] or "").split()),
        })
    db.export_csv(rows, PILOT_CSV)

    passed = [r for r in rows if r["in_scope"] == 1]
    print(f"\n   in scope: {len(passed)} of {len(rows)}   "
          f"(out of scope: {len(rows) - len(passed)})")
    print("   by source (in scope / drawn):")
    for source in sorted({r["source"] for r in rows}):
        group = [r for r in rows if r["source"] == source]
        ok = sum(1 for r in group if r["in_scope"] == 1)
        print(f"     {source:11} {ok}/{len(group)}")
    print(f"\n   pilot pool = {len(passed)} posts that pass the rubric")
    print(f"   all {len(rows)} rows written to {PILOT_CSV.relative_to(db.ROOT)} with the flag")
    if len(passed) < PILOT_SIZE:
        print(f"   {PILOT_SIZE - len(passed)} short of {PILOT_SIZE} — say the word and I'll "
              f"draw the same strata again to top the pool up (more calls).")
    print("   Next: python run.py pilot  (open coding on the posts that passed)")
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="apply the locked rubric as the second-stage scope check")
    ap.add_argument("--pilot", action="store_true",
                    help="draw the 300 Module 3 pilot posts and check them first")
    ap.add_argument("--top-up", action="store_true",
                    help="draw and check more posts to grow the pilot pool toward 300 in-scope")
    ap.add_argument("--force", action="store_true",
                    help="re-check rows that already have an in_scope verdict")
    args = ap.parse_args(argv)
    try:
        if args.top_up:
            return run_topup(force=args.force)
        return run_pilot(force=args.force) if args.pilot else run_sample(force=args.force)
    except KeyboardInterrupt:
        print("\n   stopped; everything already written is kept.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
