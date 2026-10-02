#!/usr/bin/env python3
"""Single entry point for every stage.

    python run.py collect --source playstore      get posts from one source
    python run.py clean                           remove duplicates and junk
    python run.py relevance                       decide which posts are about buying
    python run.py pilot                           build the codebook (then stop for you)
    python run.py extract                         tag posts against the codebook
    python run.py label                           hand-label 100 posts in the browser
    python run.py validate                        compare the LLM against your labels
    python run.py analyze                         build the analysis tables
    python run.py score                           score the opportunity areas
    python run.py dashboard                       open the dashboard
    python run.py doctor                          check keys and sources

Everything you may want to change lives in config/*.yaml or prompts/*.txt.
Run this with no arguments to see the list.
"""

import argparse
import importlib
import random
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine import db                     # noqa: E402
from engine.llm import LLM                # noqa: E402

SOURCES = ["playstore", "appstore", "reddit", "youtube", "manual"]
STAGES = {
    "clean": "Milestone 2", "relevance": "Milestone 2", "pilot": "Milestone 3",
    "extract": "Milestone 4", "label": "Milestone 4", "validate": "Milestone 4",
    "analyze": "Milestone 5", "score": "Milestone 6", "dashboard": "Milestone 5",
}


def load_yaml(name: str) -> dict:
    with open(ROOT / "config" / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def report_shape(rows):
    """The counts that tell you whether the run looks sane, before you read any text."""
    dates = sorted(r.get("date") or "" for r in rows if r.get("date"))
    ratings = {}
    langs = {}
    for row in rows:
        if row.get("rating") is not None:
            ratings[row["rating"]] = ratings.get(row["rating"], 0) + 1
        langs[row.get("lang_hint", "?")] = langs.get(row.get("lang_hint", "?"), 0) + 1
    explicit = sum(r.get("myntra_explicit") or 0 for r in rows)
    words = [len((r.get("text") or "").split()) for r in rows]
    print(f"   dates: {dates[0][:10]} to {dates[-1][:10]}")
    if ratings:
        print("   ratings: " + ", ".join(f"{k} star {v}" for k, v in
                                        sorted(ratings.items(), key=lambda x: str(x[0]))))
    else:
        print("   ratings: not used by this source")
    print("   languages: " + ", ".join(f"{k} {v}" for k, v in sorted(langs.items())))
    print(f"   mention Myntra by name: {explicit} of {len(rows)}")
    if words:
        print(f"   length: {min(words)}-{max(words)} words (mean {sum(words)/len(words):.1f})")


def show_samples(rows, n=10, width=150):
    """Show a random sample of what we collected, in plain English."""
    if not rows:
        print("   (no rows)")
        return
    picks = rows if len(rows) <= n else random.sample(rows, n)
    for row in picks:
        text = (row.get("text") or "").replace("\n", " ").strip()
        bits = [f"★{row['rating']}" if row.get("rating") else "", row.get("source", ""),
                str(row.get("date") or "")[:10], row.get("lang_hint", "")]
        print(f"   - [{' | '.join(b for b in bits if b)}] {text[:width]}")
    if len(rows) > n:
        print(f"   ({n} of {len(rows)} rows shown, picked at random)")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_collect(args) -> int:
    cfg = load_yaml("sources.yaml")
    chosen = [args.source] if args.source else SOURCES
    conn = db.connect()
    total = 0

    for name in chosen:
        if name not in SOURCES:
            print(f"Unknown source '{name}'. Choose from: {', '.join(SOURCES)}")
            return 2
        if not cfg.get(name, {}).get("enabled", True):
            print(f"\n== {name}: disabled in config/sources.yaml, skipping")
            continue

        print(f"\n== {name}")
        try:
            module = importlib.import_module(f"engine.collectors.{name}")
        except Exception as exc:
            print(f"   collector unavailable: {exc}")
            continue

        rows = []
        try:
            # One page first, so you can see the raw fields before a full run.
            if name == "playstore":
                sample, fields = module.fetch_sample(cfg, n=5)
                print(f"   one page fetched: {len(sample)} reviews")
                print(f"   raw fields from the API: {', '.join(fields)}")
                if sample:
                    show_samples(sample, n=2, width=110)
                if args.preview_only:
                    print("   --preview-only given, stopping before the full collection.")
                    continue
            if args.limit:
                cfg[name]["limit"] = args.limit

            def progress(got, target, pages, duplicates):
                print(f"   ...{got} new (page {pages}, {duplicates} already seen)")

            rows = module.collect(cfg, progress if args.verbose else None) if name == "playstore" \
                else module.collect(cfg)
        except NotImplementedError as exc:
            print(f"   not built yet: {exc}")
            continue
        except Exception as exc:
            print(f"   failed: {exc}")
            db.log_run(conn, "collect", name, 0, 0, f"failed: {type(exc).__name__}")
            continue

        if not rows:
            print("   no rows collected (nothing was saved)")
            db.log_run(conn, "collect", name, 0, 0, "no rows")
            continue

        written = db.upsert(conn, "raw_posts", rows)
        # Export everything held for this source, not just this run, so the CSV and
        # the database never drift apart after a resumed collection.
        stored = [dict(r) for r in db.rows_by_source(conn, "raw_posts", name)]
        csv_path = ROOT / "data" / "exports" / f"raw_posts_{name}.csv"
        db.export_csv(stored, csv_path)
        db.log_run(conn, "collect", name, 0, written, f"-> {csv_path.name}")
        total += written
        print(f"\n   {written} new rows added to raw_posts")
        print(f"   {len(stored)} rows for {name} now in the database and in {csv_path.name}")
        report_shape(stored)
        show_samples(stored, n=args.sample)

    if total:
        print(f"\nDone. raw_posts now holds {db.count_rows(conn, 'raw_posts')} rows in total.")
    elif not args.preview_only:
        print("\nNothing new was collected.")
    return 0


def cmd_doctor(_args) -> int:
    """One screen telling you what works before you spend anything."""
    print("Fashion Shopping Discovery Engine — status check\n")
    conn = db.connect()
    print(f"  database      data/engine.db ({db.count_rows(conn, 'raw_posts')} raw posts)")
    print(f"  venv          python {sys.version.split()[0]}")

    print("\n  keys (present / missing — values are never printed)")
    import os
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "YOUTUBE_API_KEY"):
        value = os.environ.get(key, "")
        state = f"found, {len(value)} chars" if value else "MISSING — paste into .env"
        print(f"    {key:18} {state}")

    llm = LLM(stage="relevance")
    print(f"\n  LLM           {llm.describe()}")
    print(f"                 precision stages use {llm.config['per_stage_models'].get('precision')}")

    print("\n  sources")
    cfg = load_yaml("sources.yaml")
    for name in SOURCES:
        section = cfg.get(name, {})
        note = "enabled" if section.get("enabled", True) else "disabled in config"
        if name == "appstore":
            note += "  (Apple RSS verified empty — will self-mark unavailable)"
        if name == "reddit":
            note += "  (fragile service, best-effort)"
        print(f"    {name:12} {note}")
    print("\n  Run a stage:  python run.py collect --source playstore --preview-only")
    return 0


def cmd_stage(name: str, args) -> int:
    if name in ("label", "dashboard"):
        page = ROOT / "dashboard" / f"{name}.py"
        # A stub file exists long before the stage is built, so check its contents.
        if not page.exists() or "NotImplementedError" in page.read_text():
            print(f"\nThe {name} stage ({STAGES.get(name, 'not built yet')}) has not been built yet. "
                  f"Nothing was changed.\n")
            return 1
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(page)], cwd=ROOT)
        return 0

    try:
        module = importlib.import_module(f"engine.{name}")
        return module.run() or 0
    except NotImplementedError as exc:
        print(f"\n{exc}\n")
        print(f"{STAGES.get(name, 'That stage')} is not built yet. Nothing was changed.")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command")

    collect = subs.add_parser("collect", help="collect posts from the sources")
    collect.add_argument("--source", choices=SOURCES, help="only this source")
    collect.add_argument("--limit", type=int, help="override the limit in config/sources.yaml")
    collect.add_argument("--preview-only", action="store_true", help="fetch one page, show fields, stop")
    collect.add_argument("--verbose", action="store_true", help="progress while collecting")
    collect.add_argument("--sample", type=int, default=10, help="how many random rows to show (default 10)")
    collect.set_defaults(func=cmd_collect)

    doctor = subs.add_parser("doctor", help="check keys, sources and the database")
    doctor.set_defaults(func=cmd_doctor)

    for name, help_text in [
        ("clean", "remove duplicates, spam and very short posts"),
        ("relevance", "keep only posts about fashion purchase decisions"),
        ("pilot", "build the codebook from a 300-post sample"),
        ("extract", "tag every post against the codebook"),
        ("label", "hand-label posts in the browser"),
        ("validate", "compare the LLM against your labels"),
        ("analyze", "build the analysis tables"),
        ("score", "score the opportunity areas"),
        ("dashboard", "open the dashboard"),
    ]:
        sub = subs.add_parser(name, help=help_text)
        sub.set_defaults(func=lambda a, n=name: cmd_stage(n, a))
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())