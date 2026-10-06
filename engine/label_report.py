"""Report precision and recall from your hand labels in data/labels/relevance_check.csv.

The sample is stratified: 50 kept and 50 dropped, split evenly across sources. The
corpus is not split that way (playstore is 70% of kept, appstore 1%), so averaging
agreement across the 100 rows would describe the sample rather than the corpus.
Every number below is post-stratified back to the real source mix.

Run:
    python -m engine.label_report
    python -m engine.label_report --selftest
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict

from engine import db

LABELS = db.ROOT / "data" / "labels" / "relevance_check.csv"

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


def load_labels() -> list[dict]:
    with open(LABELS, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["engine"] = int(r["engine_said_relevant"])
        r["mine"] = int(r["my_judgement"]) if str(r["my_judgement"]).strip() else None
    return rows


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


def report() -> dict:
    conn = db.connect()
    corpus = corpus_counts(conn)
    conn.close()
    rows = load_labels()

    kept_rows = [r for r in rows if r["mine"] is not None and r["engine"] == 1]
    drop_rows = [r for r in rows if r["mine"] is not None and r["engine"] == 0]
    if not kept_rows or not drop_rows:
        raise SystemExit("Fill in my_judgement on both kept and dropped rows first.")

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
    rows = [r for r in load_labels() if str(r["my_judgement"]).strip() != ""]
    if not rows:
        rows = load_labels()
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="verify the weighting maths on the engine's own answers")
    args = ap.parse_args()
    if args.selftest:
        return 0 if selftest() else 1

    res = report()
    P, R = res["precision"], res["recall"]
    print("=" * 74)
    print("RELEVANCE QUALITY — post-stratified to the real source mix")
    print("=" * 74)
    print(f"  precision            {P*100:.1f}%")
    print(f"  recall               {R*100:.1f}%")
    print(f"  relevant posts wrongly dropped (estimated)   {res['missed_total']:,.0f}")
    print()
    print(f"  raw sample figures: precision {res['raw_precision']*100:.0f}%, "
          f"{res['raw_missed_rate']*100:.0f}% of dropped rows you called relevant")
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
    print("  by source (share of dropped rows you called relevant):")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
