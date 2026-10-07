"""Report precision and recall for the relevance filter.

Two ways to supply the reference answers:

1. --judge (the default path used now): the 100 rows are judged by an independent
   model against docs/labelling_guide.md, stored in judge_label / judge_reason by
   `python run.py judge-relevance`. my_judgement stays blank. This is cross-model
   validation against a written rubric, not human validation.
2. Hand labels: you fill in data/labels/relevance_check_blind.csv (id, text,
   my_judgement) and the answers are merged onto relevance_check.csv by id.

The sample is stratified: 50 kept and 50 dropped, split evenly across sources. The
corpus is not split that way (playstore is 70% of kept, appstore 1%), so averaging
agreement across the 100 rows would describe the sample rather than the corpus.
Every number below is post-stratified back to the real source mix.

Run:
    python -m engine.label_report --judge
    python -m engine.label_report            # hand labels, once my_judgement is filled
    python -m engine.label_report --selftest
"""
from __future__ import annotations

import argparse
import csv
import re
import textwrap
from collections import defaultdict

from engine import db

LABELS = db.ROOT / "data" / "labels" / "relevance_check.csv"
BLIND = db.ROOT / "data" / "labels" / "relevance_check_blind.csv"
GUIDE = db.ROOT / "docs" / "labelling_guide.md"
BLIND_COLUMNS = ["id", "text", "my_judgement"]

JUDGE_LABEL = "judge_label"
JUDGE_REASON = "judge_reason"

METHODOLOGY = """\
METHODOLOGY — cross-model validation against a written rubric
  The engine's verdicts were produced by Gemini. The reference answers come from an
  independent judge (a different model, run on Groq) applying docs/labelling_guide.md
  as the rubric. The judge never saw the engine's verdict, confidence, source or date.
  This is NOT human validation: no hand labels were used, my_judgement stays empty.
  Metrics are post-stratified to the real source mix of the corpus, not to the sample."""


DECISION = re.compile(
    r"stop|never (buy|order)|no longer|won'?t (buy|order)|uninstall|switch|"
    r"delay|postpone|wait|hold off|before (you )?order|check your|verify|"
    r"compare|comparison|compar|shortlist|wish ?list|cart|choose|choosing|"
    r"recommend|suggest|worth|avoid|think twice|warn",
    re.I,
)
COMPLAINT = re.compile(
    r"deliver|refund|cancel|customer care|support|service|bug|crash|replace|"
    r"return|payment|track|damaged|scam|late|wrong (product|item|size)|"
    r"disappointed|worst|uninstalled|garbage",
    re.I,
)


def corpus_counts(conn) -> dict:
    """{(verdict, source): rows in the whole corpus} for post-stratification."""
    out = defaultdict(int)
    for r in conn.execute(
            "SELECT source, is_relevant, COUNT(*) AS n FROM clean_posts "
            "WHERE is_relevant IS NOT NULL GROUP BY source, is_relevant"):
        out[(int(r["is_relevant"]), r["source"])] = r["n"]
    return out


def read_blind() -> tuple[dict[str, int], dict[str, str]]:
    """Your labels and texts, keyed by id. Fails loudly rather than guessing.

    Both dicts come back empty when the blind file does not exist yet, so the report
    can still run off answers merged on an earlier run.
    """
    if not BLIND.exists():
        return {}, {}
    with open(BLIND, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        if header != BLIND_COLUMNS:
            raise SystemExit(
                f"{BLIND.name}: expected columns {BLIND_COLUMNS}, got {header}. "
                f"Restore the file to exactly id, text, my_judgement.")
        labels: dict[str, int] = {}
        texts: dict[str, str] = {}
        for line, row in enumerate(reader, start=2):
            rid = (row.get("id") or "").strip()
            value = (row.get("my_judgement") or "").strip()
            if not rid:
                raise SystemExit(f"{BLIND.name}: line {line} has an empty id.")
            if rid in labels:
                raise SystemExit(f"{BLIND.name}: line {line} repeats id {rid}.")
            if value not in {"0", "1"}:
                found = value or "(empty)"
                raise SystemExit(
                    f"{BLIND.name}: line {line} (id {rid}) has my_judgement="
                    f"{found} — every row must be 0 or 1.")
            labels[rid] = int(value)
            texts[rid] = row.get("text") or ""
    return labels, texts


def merge_labels(fieldnames: list[str], rows: list[dict]) -> None:
    """Copy your judgement from the blind file onto the answer key, by id.

    Any id missing on either side, or any text that does not match, aborts: a silent
    merge onto the wrong post would corrupt the whole quality report.
    """
    labels, texts = read_blind()
    if not labels:
        # No blind file: trust whatever was merged on an earlier run, still checking
        # that each value is 0, 1 or blank.
        for row in rows:
            value = str(row.get("my_judgement", "")).strip()
            if value not in {"", "0", "1"}:
                raise SystemExit(
                    f"{LABELS.name}: id {row['post_id']} has my_judgement="
                    f"{value!r} — must be 0 or 1.")
        return

    by_id = {db.label_id(r["post_id"]): r for r in rows}
    unknown = sorted(set(labels) - set(by_id))
    if unknown:
        raise SystemExit(
            f"{BLIND.name}: {len(unknown)} id(s) are not in {LABELS.name} "
            f"({', '.join(unknown[:5])}{'…' if len(unknown) > 5 else ''}). "
            f"The two files come from different samples — re-export them together.")
    unlabelled = sorted(set(by_id) - set(labels))
    if unlabelled:
        raise SystemExit(
            f"{BLIND.name}: {len(unlabelled)} row(s) are missing an id present in "
            f"{LABELS.name} ({', '.join(unlabelled[:5])}"
            f"{'…' if len(unlabelled) > 5 else ''}). Nothing was merged.")

    bad_text = [rid for rid, row in by_id.items()
                if " ".join(texts[rid].split())
                != " ".join((row["text_original"] or "").split())]
    if bad_text:
        raise SystemExit(
            f"{BLIND.name}: text does not match {LABELS.name} for "
            f"{len(bad_text)} id(s) ({', '.join(bad_text[:5])}"
            f"{'…' if len(bad_text) > 5 else ''}). Nothing was merged.")

    for rid, value in labels.items():
        by_id[rid]["my_judgement"] = str(value)

    # Write the merged answers back so every other tool reads one file.
    with open(LABELS, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_check() -> tuple[list[str], list[dict]]:
    """The answer key as it stands: engine verdict, plus your judgement once merged."""
    with open(LABELS, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def annotate(rows: list[dict], source: str = "human") -> list[dict]:
    column = JUDGE_LABEL if source == "judge" else "my_judgement"
    for r in rows:
        r["engine"] = int(r["engine_said_relevant"])
        value = str(r.get(column, "")).strip()
        if value and value not in {"0", "1"}:
            raise SystemExit(
                f"{LABELS.name}: id {r['post_id']} has {column}={value!r} — "
                f"must be 0, 1 or blank.")
        r["mine"] = int(value) if value in {"0", "1"} else None
    return rows


def load_labels(source: str = "human") -> list[dict]:
    """Answer key with reference answers in r['mine'].

    source='judge' reads judge_label straight from the file (no blind merge) and
    refuses to report on a half-judged sample. source='human' merges your answers
    from the blind file as before.
    """
    fieldnames, rows = read_check()
    if source == "judge":
        if JUDGE_LABEL not in fieldnames:
            raise SystemExit(
                f"{LABELS.name} has no {JUDGE_LABEL} column. "
                f"Run: python run.py judge-relevance")
        unjudged = [r["post_id"] for r in rows
                    if not str(r.get(JUDGE_LABEL, "")).strip()]
        if unjudged:
            raise SystemExit(
                f"{len(unjudged)} of {len(rows)} row(s) have no judge_label yet "
                f"({', '.join(unjudged[:3])}{'…' if len(unjudged) > 3 else ''}). "
                f"Run: python run.py judge-relevance")
        return annotate(rows, source="judge")
    merge_labels(fieldnames, rows)
    return annotate(rows)


def guide_example_ids(rows: list[dict]) -> set[str]:
    """Post ids the guide quotes as worked examples.

    The rubric handed to the judge contains these posts with their answers, so the
    judge is not deciding them independently. The report therefore shows the numbers
    with and without them.
    """
    guide = GUIDE.read_text(encoding="utf-8")
    quoted = re.findall(
        r"`((?:playstore|appstore|reddit|youtube|manual):[^`]+)`", guide)
    # Prose references are often truncated ("playstore:e504636a…"); only full ids
    # that are actually in this sample can be excluded from it.
    sample = {r["post_id"] for r in rows}
    found = {q for q in quoted if "..." not in q and q in sample}
    if not found:
        raise SystemExit(
            f"none of the ids quoted in {GUIDE.name} appear in {LABELS.name} — "
            f"the guide and the sample have drifted apart.")
    absent = {q for q in quoted if "..." not in q and q not in sample}
    if absent:
        raise SystemExit(
            f"{GUIDE.name} quotes {len(absent)} id(s) that are not in {LABELS.name} "
            f"({', '.join(sorted(absent))}) — re-export the sample or fix the guide.")
    return found


def strata(rows: list[dict], verdict: int) -> dict:
    """{source: [row, ...]} for one engine verdict."""
    out = defaultdict(list)
    for r in rows:
        if r["engine"] == verdict and r["mine"] is not None:
            out[r["source"]].append(r)
    return out


def post_stratify(group: dict, corpus: dict, verdict: int):
    """Estimate a corpus-wide rate by weighting each source back to its real size.

    Each sampled row from source s stands for corpus[s]/sampled[s] rows. Returns the
    estimated count of positives in the whole corpus group, plus what was sampled.
    """
    est = 0.0
    sampled_total = corpus_total = 0
    for source, rows in group.items():
        pop = corpus.get((verdict, source), 0)
        corpus_total += pop
        sampled_total += len(rows)
        if not rows or not pop:
            continue
        agree = sum(r["mine"] for r in rows)
        est += agree * pop / len(rows)
    return est, sampled_total, corpus_total


def report(rows: list[dict] | None = None) -> dict:
    conn = db.connect()
    corpus = corpus_counts(conn)
    conn.close()
    if rows is None:
        rows = load_labels()

    kept_rows = [r for r in rows if r["mine"] is not None and r["engine"] == 1]
    drop_rows = [r for r in rows if r["mine"] is not None and r["engine"] == 0]
    if not kept_rows or not drop_rows:
        raise SystemExit("Fill in the reference answers on both kept and dropped rows first.")

    # Precision: of what the engine kept, how much did you agree is relevant.
    tp, _, kept_pop = post_stratify(strata(rows, 1), corpus, 1)
    precision = tp / kept_pop if kept_pop else 0.0

    # Missed: of what the engine dropped, how much you say was relevant after all.
    fn, _, drop_pop = post_stratify(strata(rows, 0), corpus, 0)
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    raw_p = sum(r["mine"] for r in kept_rows) / len(kept_rows)
    raw_fn = sum(r["mine"] for r in drop_rows) / len(drop_rows)

    return {
        "precision": precision, "recall": recall,
        "tp": tp, "fn": fn,
        "raw_precision": raw_p, "raw_missed_rate": raw_fn,
        "kept_pop": kept_pop, "drop_pop": drop_pop,
        "missed_total": fn,
        "kept_rows": kept_rows, "drop_rows": drop_rows,
        "corpus": corpus, "rows": rows,
    }


def disagreements(res: dict) -> dict:
    """Every row where your label differs from the engine, grouped by why."""
    out = {"engine_wrong_keep": [], "engine_wrong_drop": []}
    for r in res["kept_rows"]:
        if r["mine"] == 0:
            out["engine_wrong_keep"].append(r)
    for r in res["drop_rows"]:
        if r["mine"] == 1:
            out["engine_wrong_drop"].append(r)
    return out


def _theme(r: dict) -> str:
    """Short label for why a row went wrong, used to group the disagreement list."""
    text = r["text_original"]
    engine_reason = r["relevance_reason"]
    if r["engine"] == 1:
        # Engine kept it; you did not. Was it a pure complaint?
        if DECISION.search(text) and not COMPLAINT.search(text):
            return "engine kept it as a decision, you saw none"
        if COMPLAINT.search(text) and not DECISION.search(text):
            return "engine kept a pure post-purchase complaint"
        return "engine kept it as a decision, you saw none"
    # Engine dropped it; you saw a decision.
    if DECISION.search(text):
        return "engine missed a decision link"
    if COMPLAINT.search(text):
        return "complaint that also states a buying decision"
    return "engine saw no decision, you did"


def selftest() -> None:
    """Feed the engine's own answers back as if they were yours.

    Agreement should be perfect: precision 100%, recall 100%, zero missed. Anything
    else means the weighting maths is wrong, and the real report cannot be trusted.
    """
    conn = db.connect()
    corpus = corpus_counts(conn)
    conn.close()
    # Reads the answer key only — no merge, no file writes, no dependence on the
    # blind file being filled in yet.
    rows = annotate(read_check()[1])
    for r in rows:
        r["mine"] = r["engine"]          # pretend you agreed with the engine

    tp, _, kept_pop = post_stratify(strata(rows, 1), corpus, 1)
    fn, _, drop_pop = post_stratify(strata(rows, 0), corpus, 0)
    precision = tp / kept_pop
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print("SELF-TEST: engine answers used as the human labels")
    print(f"  precision        {precision*100:.1f}%   (expect 100.0)")
    print(f"  recall           {recall*100:.1f}%     (expect 100.0)")
    print(f"  missed relevant  {fn:.1f}              (expect 0.0)")
    print(f"  kept sampled     {sum(1 for r in rows if r['engine']==1)} of {kept_pop}")
    print(f"  dropped sampled  {sum(1 for r in rows if r['engine']==0)} of {drop_pop}")
    ok = abs(precision - 1.0) < 1e-9 and abs(recall - 1.0) < 1e-9 and abs(fn) < 1e-6
    print()
    print("  MATHS OK" if ok else "  *** MATHS WRONG — do not trust the real report ***")
    return ok


def _print_metrics(title: str, res: dict, detail: bool = True,
                   who: str = "you") -> None:
    P, R = res["precision"], res["recall"]
    print("-" * 74)
    print(title)
    print("-" * 74)
    print(f"  precision            {P*100:.1f}%")
    print(f"  recall               {R*100:.1f}%")
    print(f"  relevant posts wrongly dropped (estimated)   {res['missed_total']:,.0f}")
    print(f"  rows behind this view: {len(res['kept_rows'])} kept + "
          f"{len(res['drop_rows'])} dropped")
    print()
    print(f"  raw sample figures: precision {res['raw_precision']*100:.0f}%, "
          f"{res['raw_missed_rate']*100:.0f}% of dropped rows {who} called relevant")
    if detail:
        print()
        print("  by source (precision):")
        for source, srows in sorted(strata(res["rows"], 1).items()):
            pop = res["corpus"].get((1, source), 0)
            if not pop:
                continue
            a = sum(r["mine"] for r in srows)
            print(f"    {source:11} {a}/{len(srows)} = {a*100//len(srows):>3}%   "
                  f"(weights to {pop:,} corpus rows)")
        print()
        print(f"  by source (share of dropped rows {who} called relevant):")
        for source, srows in sorted(strata(res["rows"], 0).items()):
            pop = res["corpus"].get((0, source), 0)
            if not pop:
                continue
            a = sum(r["mine"] for r in srows)
            print(f"    {source:11} {a}/{len(srows)} = {a*100//len(srows):>3}%   "
                  f"(weights to {pop:,} corpus rows)")
    print()
    for threshold, label in ((0.85, "precision"), (0.85, "recall")):
        val = P if label == "precision" else R
        if val < threshold:
            print(f"  BELOW {threshold:.0%} — {label} needs a revised prompt "
                  f"(propose, test on these 100 rows, show before/after)")
    print()


def judge_report() -> int:
    """Metrics from judge_label, with and without the rows the rubric quoted."""
    rows = load_labels("judge")
    examples = guide_example_ids(rows)

    _print_metrics(
        f"ALL {len(rows)} ROWS", report(rows), detail=True, who="the judge")
    kept_out = [r for r in rows if r["post_id"] not in examples]
    _print_metrics(
        f"EXCLUDING THE {len(examples)} GUIDE EXAMPLES "
        f"(the rubric was shown with their answers)",
        report(kept_out), detail=False, who="the judge")

    disagreements = [r for r in rows if r["engine"] != r["mine"]]
    print("=" * 74)
    print(f"GEMINI vs JUDGE — {len(disagreements)} disagreement(s) of {len(rows)} rows")
    print("=" * 74)
    if not disagreements:
        print("  none — the two models agree on every row.")
    for r in sorted(disagreements, key=lambda x: (x["engine"], x["mine"])):
        direction = "engine kept, judge dropped" if r["engine"] == 1 \
            else "engine dropped, judge kept"
        tag = "  [GUIDE EXAMPLE]" if r["post_id"] in examples else ""
        print(f"\n  {r['post_id']}   ({direction}){tag}")
        print(f"    gemini ({r['engine']}) — {r['relevance_reason']}")
        print(f"    judge  ({r['mine']}) — {r[JUDGE_REASON]}")
        print(textwrap.fill("    text: " + " ".join((r["text_original"] or "").split()),
                            width=74, subsequent_indent="             "))
    print()
    print(METHODOLOGY)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="verify the weighting maths on the engine's own answers")
    ap.add_argument("--judge", action="store_true",
                    help="report against judge_label (independent model + written "
                         "rubric) instead of hand labels")
    args = ap.parse_args()
    if args.selftest:
        return 0 if selftest() else 1
    if args.judge:
        return judge_report()

    res = report()
    print("=" * 74)
    print("RELEVANCE QUALITY — post-stratified to the real source mix")
    print("=" * 74)
    _print_metrics("HAND LABELS", res, detail=True, who="you")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
