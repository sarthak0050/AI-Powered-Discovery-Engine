"""Terminal labeller for the relevance check.

    python run.py label-relevance

Shows one post at a time from data/labels/relevance_check_blind.csv — just the text
and an opaque id, never the engine's verdict or the source — and writes your 0/1 into
that file as you go.

Every answer is saved the moment you enter it, so quitting mid-way loses nothing:
run the same command again and it picks up at the first post you have not judged.
The merge back onto data/labels/relevance_check.csv happens later, in
engine/label_report.py.
"""

import csv
import shutil
import sys
import textwrap

from engine import db

BLIND = db.ROOT / "data" / "labels" / "relevance_check_blind.csv"
COLUMNS = ["id", "text", "my_judgement"]

HELP = "  1 relevant    0 not relevant    s skip    b back    q quit (already saved)"
RULE = ("Rule: a pure complaint (delivery, refund, cancellation, service, bugs) is 0, "
        "unless the post says it changed how they will buy — 1.\n"
        "Full guide: docs/labelling_guide.md")


def _labelled(row: dict) -> bool:
    return str(row.get("my_judgement", "")).strip() in {"0", "1"}


def _first_unlabelled(rows: list[dict], after: int = -1) -> int | None:
    for i in range(after + 1, len(rows)):
        if not _labelled(rows[i]):
            return i
    return None


def _read() -> tuple[list[str], list[dict]]:
    if not BLIND.exists():
        raise SystemExit(
            f"{BLIND} not found. There is nothing to label yet — tell me and I will "
            f"export a fresh blind sample.")
    with open(BLIND, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        if header != COLUMNS:
            raise SystemExit(f"{BLIND.name} has columns {header}, expected {COLUMNS}.")
        return header, list(reader)


def _save(fieldnames: list[str], rows: list[dict]) -> None:
    """Written after every single answer, so no answer can be lost to a crash."""
    with open(BLIND, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _show(rows: list[dict], idx: int, width: int, is_tty: bool) -> None:
    row = rows[idx]
    done = sum(1 for r in rows if _labelled(r))
    if is_tty:
        print("\033[2J\033[H", end="")          # clear screen, cursor home
    print(f"  {idx + 1} of {len(rows)}    {done} labelled, {len(rows) - done} to go")
    if _labelled(row):
        print(f"  (already {row['my_judgement']} — enter 1 or 0 to change it)")
    print(f"  id {row['id']}\n")
    text = " ".join((row.get("text") or "").split()) or "(empty post)"
    for line in textwrap.wrap(text, width=max(width - 6, 40)) or ["(empty post)"]:
        print(f"  {line}")
    print()


def run() -> int:
    if not sys.stdin.isatty():
        print("This needs an interactive terminal — run `python run.py label-relevance` "
              "directly in a window, not through a pipe.")
        return 1

    fieldnames, rows = _read()
    if not rows:
        print(f"{BLIND.name} is empty.")
        return 1

    print(f"\n{RULE}\n")
    if all(_labelled(r) for r in rows):
        print(f"All {len(rows)} posts are already labelled.")
        print("Next: python -m engine.label_report")
        return 0

    width = shutil.get_terminal_size((100, 24)).columns
    idx = _first_unlabelled(rows)

    while True:
        _show(rows, idx, width, sys.stdout.isatty())
        try:
            choice = input(f"{HELP}\n  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  saved, nothing lost.")
            return 0

        if choice in {"1", "0"}:
            rows[idx]["my_judgement"] = choice
            _save(fieldnames, rows)
            nxt = _first_unlabelled(rows, idx)
            if nxt is None:
                nxt = _first_unlabelled(rows)      # anything skipped earlier
            if nxt is None:
                _show(rows, idx, width, True)
                print(f"\n  All {len(rows)} posts labelled.\n"
                      f"\n  Next: python -m engine.label_report\n")
                return 0
            idx = nxt
        elif choice == "s":
            nxt = _first_unlabelled(rows, idx)
            if nxt is None:
                left = sum(1 for r in rows if not _labelled(r))
                print(f"\n  {left} left unlabelled (all before this point). "
                      f"Run the command again to reach them.")
                return 0
            idx = nxt
        elif choice == "b":
            idx = max(0, idx - 1)
        elif choice == "q":
            done = sum(1 for r in rows if _labelled(r))
            print(f"\n  saved: {done} of {len(rows)} labelled. "
                  f"Run the same command to continue.")
            return 0
        else:
            print("  Type 1, 0, s, b or q.")
