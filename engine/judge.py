"""Independent judge for the 100-row relevance check.

Replaces the hand-labelling step. An LLM that never saw the engine's verdict judges
each of the 100 rows against the written rubric (docs/labelling_guide.md), and the
answers are stored in judge_label / judge_reason on data/labels/relevance_check.csv.
my_judgement is left untouched and stays blank — that column remains yours if you ever
want to hand-label instead.

The judge runs on Groq and only on Groq: the engine that produced the verdict being
checked is Gemini, so a different model checking it is cross-model validation against
a written rubric, rather than one model grading its own homework. It is pinned in
config/llm.yaml (stage_providers.judge) — there is no automatic fallback anywhere, and
an unavailable provider stops the stage instead of silently switching models.

Later, extraction validation uses the same shape: ~50 posts, judged on primary_code
and barrier_type against config/codebook.yaml instead of this guide.

Run:
    python run.py judge-relevance
    python run.py judge-relevance --force     # re-judge every row
    python -m engine.label_report --judge     # measure the result
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

from pydantic import BaseModel, Field

from engine import db, llm

CHECK = db.ROOT / "data" / "labels" / "relevance_check.csv"
GUIDE = db.ROOT / "docs" / "labelling_guide.md"
PROMPT = db.ROOT / "prompts" / "judge_relevance.txt"

# Same cap the relevance stage uses. A 5,000-word YouTube essay costs a fortune in
# tokens and a sentence answers the rule just as well.
MAX_CHARS = 1200

JUDGE_LABEL = "judge_label"
JUDGE_REASON = "judge_reason"


class JudgeItem(BaseModel):
    id: int
    label: int = Field(ge=0, le=1)
    reason: str = Field(max_length=240)


class JudgeBatch(BaseModel):
    results: list[JudgeItem]


def judge_llm() -> llm.LLM:
    """Groq only, no fallback: the whole point is that this is not the engine's model.

    Pinned in config/llm.yaml — stage_providers.judge = groq, stage_models.judge =
    openai/gpt-oss-120b — so the pin can be seen and edited in one place instead of
    buried in code.
    """
    return llm.LLM(stage="judge")


def read_rows() -> tuple[list[str], list[dict]]:
    with open(CHECK, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for column in (JUDGE_LABEL, JUDGE_REASON):
        if column not in fieldnames:
            fieldnames.append(column)
    if not rows:
        raise SystemExit(f"{CHECK.name} is empty — re-export the sample first.")
    for r in rows:
        r.setdefault(JUDGE_LABEL, "")
        r.setdefault(JUDGE_REASON, "")
    return fieldnames, rows


def write_rows(fieldnames: list[str], rows: list[dict]) -> None:
    with open(CHECK, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _text(row: dict) -> str:
    """The original text, which is what the guide's worked examples quote."""
    text = " ".join((row.get("text_original") or row.get("text_en") or "").split())
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rstrip() + " …[truncated]"
    return text


def _judged(row: dict) -> bool:
    return (row.get(JUDGE_LABEL) or "").strip() in {"0", "1"}


def run(force: bool = False) -> int:
    fieldnames, rows = read_rows()
    guide = GUIDE.read_text(encoding="utf-8")
    template = PROMPT.read_text(encoding="utf-8")
    judge = judge_llm()
    batch_size = int((judge.config.get("batch") or {}).get("judge", 10)) or 10

    pending = [i for i, r in enumerate(rows) if force or not _judged(r)]
    print(f"\nJudge: {judge.describe()}")
    print(f"Rubric: {GUIDE.name} ({len(guide):,} chars)   "
          f"rows to judge: {len(pending)} of {len(rows)}   batch: {batch_size}")
    if not pending:
        print("\nAll rows already judged. Next: python -m engine.label_report --judge")
        return 0

    made = 0
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        positions = list(range(start, start + len(chunk)))
        if not force and not any(p in pending for p in positions):
            continue
        # ids are the row's position in the file, so the cache key for a batch is
        # stable across runs and a resume never re-bills a finished call.
        items = [{"id": p + 1, "text": _text(rows[p])} for p in positions]
        prompt = llm.render_prompt(
            template, {"rubric": guide, "posts": json.dumps(items, ensure_ascii=False)})
        input_id = ",".join(str(i["id"]) for i in items)
        try:
            parsed, _stats = judge.call_json(
                prompt, JudgeBatch, input_id=input_id,
                post_ids=[rows[p]["post_id"] for p in positions])
        except (llm.LLMError, llm.QuotaExhausted) as exc:
            print(f"\n  stopped: {exc}")
            print(f"  {made} row(s) written so far. Re-run the same command to continue.")
            return 1

        got = {item.id: item for item in parsed.results}
        missing = [i["id"] for i in items if i["id"] not in got]
        if missing:
            raise SystemExit(
                f"judge returned no answer for id(s) {missing}. Nothing was written "
                f"for this batch — re-run the same command.")

        for p in positions:
            item = got.get(p + 1)
            if item is None:
                continue
            rows[p][JUDGE_LABEL] = str(item.label)
            rows[p][JUDGE_REASON] = item.reason
            made += 1
        write_rows(fieldnames, rows)
        done = sum(1 for r in rows if _judged(r))
        print(f"  batch {start // batch_size + 1}: +{len(chunk)} rows  "
              f"({done}/{len(rows)} judged)")

    conn = db.connect()
    db.log_run(conn, "judge", "relevance_check", len(rows), made,
               notes=f"{judge._provider}/{judge.model}")
    conn.close()
    print(f"\n{made} row(s) judged with {judge._provider}/{judge.model}.")
    print(f"{CHECK.name}: judge_label / judge_reason filled, "
          f"my_judgement left as it was.")
    print("Next: python -m engine.label_report --judge")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="judge the 100-row sample against the rubric")
    ap.add_argument("--force", action="store_true",
                    help="re-judge rows that already have a judge_label")
    args = ap.parse_args(argv)
    try:
        return run(force=args.force)
    except KeyboardInterrupt:
        print("\n  stopped; everything already written is kept.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
