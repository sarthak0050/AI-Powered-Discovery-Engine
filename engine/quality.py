"""Engine quality report — one row per metric (AGENTS.md "Engine quality report").

Reads what earlier stages already wrote (the database plus the saved validation
CSVs) and measures how much of the pipeline can be trusted: relevance precision and
recall against the independent judge, in_scope cross-model agreement, extraction
agreement against both the human labels and the Groq judge, the other_emerging rate,
the low-confidence share, the stated-vs-underlying divergence, coverage gaps in the
decision subset, and the LLM error / retry rate and call counts from the call log.

Every number is computed from stored data — nothing is hard-coded. The file or table a
figure comes from is named in the method column so it can be traced back. Deterministic:
no LLM calls.

Columns: metric, value, target, status (pass / below / info), method, n.

Run:
    python run.py quality
    python -m engine.quality
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter

from engine import db, validate

EXPORTS = db.ROOT / "data" / "exports"
OUT = EXPORTS / "engine_quality.csv"
PILOT = EXPORTS / "pilot_sample.csv"
HUMAN = db.ROOT / "data" / "labels" / "human_labels.csv"
JUDGE = db.ROOT / "data" / "labels" / "extraction_judge.csv"

# The decision subset is defined by journey stage alone (Track A in score.py).
from engine.analyze import DECISION_STAGES  # noqa: E402

# Targets from AGENTS.md: other_emerging < 10%, low-confidence < 15%, coverage >= 100
# posts per source, tagging agreement >= 80%. Relevance targets are the 85% bar the
# label report itself prints.
PRECISION_TARGET = 85.0
AGREEMENT_TARGET = 80.0
OTHER_EMERGING_TARGET = 10.0
LOW_CONF_TARGET = 15.0
COVERAGE_MIN = 100


def _row(metric: str, value, target: str, status: str, method: str, n) -> dict:
    return {"metric": metric, "value": value, "target": target, "status": status,
            "method": method, "n": n}


def _outcome(value: float, target: float, ge: bool = True) -> str:
    """pass/below against a target. ge=True means higher is better."""
    ok = value >= target if ge else value < target
    return "pass" if ok else "below"


# --------------------------------------------------------------------------- relevance

def relevance_metrics() -> list[dict]:
    """Precision / recall of the relevance filter vs the independent judge.

    Computed live by label_report.report on the judge labels in
    data/labels/relevance_check.csv — this is cross-model validation against a written
    rubric, not human validation. Post-stratified to the real source mix, plus the raw
    sample figure.
    """
    from engine import label_report as lr

    rows = lr.load_labels("judge")
    res = lr.report(rows)
    n = len(res["kept_rows"]) + len(res["drop_rows"])
    method = ("cross-model, written rubric, not human; judge_label in "
              "data/labels/relevance_check.csv")
    p, r, raw = res["precision"] * 100, res["recall"] * 100, res["raw_precision"] * 100
    return [
        _row("relevance_precision_poststratified", f"{p:.1f}%", f">={PRECISION_TARGET:.0f}%",
             _outcome(p, PRECISION_TARGET), method, n),
        _row("relevance_recall_poststratified", f"{r:.1f}%", f">={PRECISION_TARGET:.0f}%",
             _outcome(r, PRECISION_TARGET), method, n),
        _row("relevance_precision_raw", f"{raw:.0f}%", f">={PRECISION_TARGET:.0f}%",
             _outcome(raw, PRECISION_TARGET), method, n),
    ]


# --------------------------------------------------------------------------- in_scope

def inscope_agreement() -> dict | None:
    """Cross-model agreement on in_scope: pilot screen vs extraction-stage verdict.

    The pilot registry (pilot_sample.csv) carries the in_scope verdict from the Module 3
    screening; tagged_posts carries the extraction-stage verdict. Joining on post_id
    gives the posts answered twice, across providers.
    """
    if not PILOT.exists():
        return None
    with open(PILOT, newline="", encoding="utf-8") as fh:
        pilot = list(csv.DictReader(fh))
    conn = db.connect()
    current = {r["post_id"]: int(r["in_scope"]) for r in conn.execute(
        "SELECT post_id, in_scope FROM tagged_posts WHERE in_scope IS NOT NULL")}
    conn.close()
    pairs = [(int(r["in_scope"]), current[r["post_id"]])
             for r in pilot
             if str(r.get("in_scope") or "").strip() in {"0", "1"}
             and r["post_id"] in current]
    if not pairs:
        return None
    agree = sum(1 for a, b in pairs if a == b)
    return {"n": len(pairs), "agree": agree,
            "agreement": agree / len(pairs)}


def _inscope_row() -> dict | None:
    m = inscope_agreement()
    if not m:
        return None
    pct = m["agreement"] * 100
    return _row("in_scope_cross_model_agreement", f"{pct:.1f}%", "-", "info",
                "pilot screen vs extraction in_scope, joined on post_id; "
                "data/exports/pilot_sample.csv", m["n"])


# --------------------------------------------------------------------------- extraction

def _human_rows() -> tuple[list[dict], int]:
    """Human labels matched to the engine's tags, as validate.compare consumes them."""
    labelled = validate._load_human()
    conn = db.connect()
    engine_rows = validate._engine_rows(conn, [r["post_id"] for r in labelled])
    conn.close()
    matched = [r for r in labelled if r["post_id"] in engine_rows]
    merged = [{"post_id": r["post_id"],
               "human": {k: (r.get(k) or "") for k, _ in validate.COMPARE_FIELDS},
               "engine": {k: (engine_rows[r["post_id"]].get(k) or "") for k, _ in
                          validate.COMPARE_FIELDS}}
              for r in matched]
    return merged, len(matched)


def extraction_human_metrics() -> list[dict]:
    """primary_code and barrier_type agreement vs the hand labels (n=100)."""
    merged, n = _human_rows()
    if not merged:
        return []
    res = validate.compare(merged)
    method = "data/labels/human_labels.csv vs tagged_posts; agreement + Cohen's kappa"
    out = []
    for f in res["fields"]:
        if f["field"] not in ("primary_code", "barrier_type"):
            continue
        pct = f["agreement"] * 100
        kappa = res["kappa"].get(f["field"], 0.0)
        out.append(_row(f"extraction_vs_human_{f['field']}", f"{pct:.0f}% (kappa {kappa:.2f})",
                        f">={AGREEMENT_TARGET:.0f}%", _outcome(pct, AGREEMENT_TARGET),
                        method, n))
    return out


def extraction_judge_metrics() -> list[dict]:
    """primary_code and barrier_type agreement vs the independent Groq judge (n=50)."""
    if not JUDGE.exists():
        return []
    with open(JUDGE, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    method = "data/labels/extraction_judge.csv; agreement + Cohen's kappa"
    out = []
    for field in ("primary_code", "barrier_type"):
        a = [validate._normalise(r.get(f"engine_{field}")) for r in rows]
        b = [validate._normalise(r.get(f"judge_{field}")) for r in rows]
        agree = sum(1 for x, y in zip(a, b) if x == y)
        pct = agree / len(rows) * 100 if rows else 0.0
        kappa = validate._cohen_kappa(a, b)
        out.append(_row(f"extraction_vs_judge_{field}", f"{pct:.0f}% (kappa {kappa:.2f})",
                        f">={AGREEMENT_TARGET:.0f}%", _outcome(pct, AGREEMENT_TARGET),
                        method, len(rows)))
    return out


def _done_rows(conn) -> list[dict]:
    """Fully extracted in-scope posts (the population the rates below describe)."""
    return [dict(r) for r in conn.execute(
        "SELECT codes, primary_code, confidence, underlying_reason "
        "FROM tagged_posts WHERE in_scope = 1 AND extract_status = 'done'")]


def rate_metrics() -> list[dict]:
    """other_emerging rate, low-confidence share, stated-vs-underlying divergence."""
    conn = db.connect()
    done = _done_rows(conn)
    conn.close()
    n = len(done)
    if not n:
        return []
    other = sum(1 for r in done
                if "other_emerging" in (r["codes"] or "")
                or r["primary_code"] == "other_emerging")
    low = sum(1 for r in done
              if r["confidence"] is not None and float(r["confidence"]) < 0.5)
    # underlying_reason is only populated when it differs from stated_reason, so its
    # share of the population is the divergence rate.
    diverged = sum(1 for r in done if (r["underlying_reason"] or "").strip())
    oe = other / n * 100
    lc = low / n * 100
    dv = diverged / n * 100
    return [
        _row("other_emerging_rate", f"{oe:.1f}%", f"<{OTHER_EMERGING_TARGET:.0f}%",
             _outcome(oe, OTHER_EMERGING_TARGET, ge=False),
             "tagged_posts, in_scope=1 & extract_status='done'", n),
        _row("low_confidence_share", f"{lc:.1f}%", f"<{LOW_CONF_TARGET:.0f}%",
             _outcome(lc, LOW_CONF_TARGET, ge=False),
             "tagged_posts, in_scope=1 & extract_status='done'; confidence < 0.5", n),
        _row("stated_vs_underlying_divergence", f"{dv:.1f}%", "-", "info",
             "tagged_posts, in_scope=1 & extract_status='done'; underlying_reason "
             "populated (schema sets it only when it differs from stated_reason)", n),
    ]


# --------------------------------------------------------------------------- coverage

def coverage_metrics() -> list[dict]:
    """Posts per source in the pre-purchase decision subset; flag any source < 100."""
    conn = db.connect()
    rows = list(conn.execute(
        "SELECT r.source AS source, COUNT(*) AS n "
        "FROM tagged_posts t JOIN raw_posts r ON r.post_id = t.post_id "
        "WHERE t.in_scope = 1 AND t.journey_stage IN (%s) "
        "GROUP BY r.source ORDER BY n DESC"
        % ",".join("?" for _ in DECISION_STAGES), tuple(DECISION_STAGES)))
    conn.close()
    total = sum(r["n"] for r in rows)
    out = []
    for r in rows:
        status = "pass" if r["n"] >= COVERAGE_MIN else "below"
        out.append(_row(f"coverage_{r['source']}_decision_subset", r["n"],
                        f">={COVERAGE_MIN} posts", status,
                        "tagged_posts x raw_posts; pre-purchase decision subset "
                        "(journey_stage only)", total))
    return out


# --------------------------------------------------------------------------- llm log

def llm_metrics() -> list[dict]:
    """Total calls by provider/model, plus the error and retry rate, from the log."""
    conn = db.connect()
    total = conn.execute("SELECT COUNT(*) FROM llm_call_log").fetchone()[0]
    if not total:
        conn.close()
        return []
    non_ok = conn.execute(
        "SELECT COUNT(*) FROM llm_call_log WHERE status IS NULL OR status <> 'ok'"
    ).fetchone()[0]
    retried = conn.execute(
        "SELECT COUNT(*) FROM llm_call_log WHERE attempts > 1").fetchone()[0]
    by_model = list(conn.execute(
        "SELECT provider, model, COUNT(*) AS n FROM llm_call_log "
        "GROUP BY provider, model ORDER BY n DESC"))
    statuses = list(conn.execute(
        "SELECT status, COUNT(*) AS n FROM llm_call_log GROUP BY status ORDER BY n DESC"))
    conn.close()

    err = non_ok / total * 100
    retry = retried / total * 100
    breakdown = "; ".join(f"{r['provider']}/{r['model']} {r['n']}" for r in by_model)
    status_note = ", ".join(f"{r['status']} {r['n']}" for r in statuses)
    method = f"llm_call_log; breakdown: {breakdown}; statuses: {status_note}"
    out = [
        _row("llm_calls_total", total, "-", "info", method, total),
        _row("llm_error_rate", f"{err:.1f}%", "-", "info",
             f"llm_call_log, status <> 'ok' (mostly http_429 rate limits); {status_note}",
             total),
        _row("llm_retry_rate", f"{retry:.1f}%", "-", "info",
             "llm_call_log, attempts > 1 (invalid JSON / transient error retried)",
             total),
    ]
    for r in by_model:
        out.append(_row(f"llm_calls_{r['provider']}_{r['model']}", r["n"], "-", "info",
                        "llm_call_log, grouped by provider/model", total))
    return out


# --------------------------------------------------------------------------- assemble

def collect() -> list[dict]:
    """Every metric row, in report order."""
    rows: list[dict] = []
    rows += relevance_metrics()
    row = _inscope_row()
    if row:
        rows.append(row)
    rows += extraction_human_metrics()
    rows += extraction_judge_metrics()
    rows += rate_metrics()
    rows += coverage_metrics()
    rows += llm_metrics()
    return rows


def _print(rows: list[dict]) -> None:
    print("=" * 78)
    print("ENGINE QUALITY REPORT")
    print("=" * 78)
    print(f"  {'metric':<44}{'value':>12}  {'status':<6}")
    for r in rows:
        value = str(r["value"])
        print(f"  {r['metric']:<44}{value:>12}  {r['status']:<6}")
    counts = Counter(r["status"] for r in rows)
    print()
    print(f"  {len(rows)} metric(s): "
          f"{counts.get('pass', 0)} pass, {counts.get('below', 0)} below, "
          f"{counts.get('info', 0)} info")
    print(f"  -> {OUT.relative_to(db.ROOT)}")


def run(args=None) -> int:
    rows = collect()
    if not rows:
        print("no metrics to report — run the earlier stages first.")
        return 2
    db.export_csv(rows, OUT)
    _print(rows)
    conn = db.connect()
    db.log_run(conn, "quality", "pipeline", len(rows), len(rows),
               notes=f"{Counter(r['status'] for r in rows)}")
    conn.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="engine quality report (no LLM calls)")
    ap.parse_args(argv)
    return run()


if __name__ == "__main__":
    sys.exit(main())
