"""Module 3 — pilot and codebook builder.

Draws nothing itself: the pilot pool is whatever `python run.py in-scope --pilot`
drew and screened, and only the posts that pass the locked rubric are coded. Open
coding invents short free-text labels (`prompts/open_coding.txt`), embeddings +
HDBSCAN group the labels into clusters, an LLM names each cluster and writes its
inclusion/exclusion rules (`prompts/theme_naming.txt`), and the result is a draft
`config/codebook.yaml` plus `data/exports/codebook_review.csv` for you to edit.

Built bottom-up and problem-agnostic: no theme, barrier or category is assumed in
this file or in the prompts. `other_emerging` always exists so no post can be
untagged while you review the draft.

The pipeline stops here until you approve the codebook (set approved_by_user in
config/codebook.yaml).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from engine import db, llm

PROMPT = db.ROOT / "prompts" / "open_coding.txt"
NAMING_PROMPT = db.ROOT / "prompts" / "theme_naming.txt"
PILOT_CSV = db.ROOT / "data" / "exports" / "pilot_sample.csv"
CODES_CSV = db.ROOT / "data" / "exports" / "open_codes.csv"
CODEBOOK_YAML = db.ROOT / "config" / "codebook.yaml"
REVIEW_CSV = db.ROOT / "data" / "exports" / "codebook_review.csv"

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_CODES = 3            # the prompt asks for 1-3 codes per post
MIN_CLUSTER_POSTS = 3    # a cluster must show in at least this many posts
MAX_EXAMPLES = 2


class CodeItem(BaseModel):
    id: int
    codes: list[str] = Field(default_factory=list, max_length=MAX_CODES)


class CodeBatch(BaseModel):
    results: list[CodeItem]


class ThemeItem(BaseModel):
    cluster_id: int
    label: str = Field(min_length=1, max_length=40)
    definition: str
    include_when: str
    exclude_when: str
    example_quotes: list[str] = Field(default_factory=list, max_length=4)


class ThemeBatch(BaseModel):
    clusters: list[ThemeItem]


def _pilot_pool() -> list[dict]:
    """The pilot registry posts that passed in_scope, in screening order."""
    if not PILOT_CSV.exists():
        raise SystemExit(
            f"{PILOT_CSV.relative_to(db.ROOT)} does not exist. Run the screening first:\n"
            f"    python run.py in-scope --pilot")
    import csv
    with open(PILOT_CSV, newline="", encoding="utf-8") as fh:
        registered = [r["post_id"] for r in csv.DictReader(fh)]
    if not registered:
        raise SystemExit("the pilot registry is empty — nothing to code.")
    conn = db.connect()
    marks = ",".join("?" * len(registered))
    rows = [dict(r) for r in conn.execute(
        f"SELECT t.post_id, r.text AS text_original, r.source, r.engagement "
        f"FROM tagged_posts t JOIN raw_posts r ON r.post_id = t.post_id "
        f"WHERE t.in_scope = 1 AND t.post_id IN ({marks}) "
        f"ORDER BY t.post_id", registered)]
    conn.close()
    return rows


def _open_coding(pool: list[dict], force: bool) -> int:
    """Code every pool post with 1-3 invented labels, batch by batch (resumable)."""
    from engine.llm import LLMError, QuotaExhausted

    template = PROMPT.read_text(encoding="utf-8")
    client = llm.LLM(stage="open_coding")
    batch_size = int((client.config.get("batch") or {}).get("open_coding", 20)) or 20
    who = f"{client._provider}/{client.model}"
    print(f"== open coding {len(pool)} posts on {who}   batches of {batch_size}")

    conn = db.connect()
    existing = {r["post_id"] for r in conn.execute("SELECT post_id FROM open_codes")}
    pending = [p for p in pool if force or p["post_id"] not in existing]
    print(f"   {len(pending)} to code, {len(pool) - len(pending)} already coded")

    done = 0
    coded_total = 0
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        payload = [{"id": i + 1, "text": (p["text_original"] or "")[:1200]}
                   for i, p in enumerate(chunk)]
        prompt = llm.render_prompt(
            template, {"posts": json.dumps(payload, ensure_ascii=False)})
        ids = ",".join(str(p["post_id"]) for p in chunk)
        try:
            parsed, stats = client.call_json(
                prompt, CodeBatch, input_id=ids,
                post_ids=[p["post_id"] for p in chunk])
        except (LLMError, QuotaExhausted) as exc:
            print(f"   stopped: {exc}")
            print(f"   {done} batch(es) written. Re-run the same command to continue.")
            return 1
        real = {i + 1: chunk[i] for i in range(len(chunk))}
        for item in parsed.results:
            post = real.get(item.id)
            if not post:
                continue
            for code in item.codes:
                clean = " ".join(str(code).strip().lower().split())
                if clean and len(clean) <= 60:
                    conn.execute(
                    "INSERT INTO open_codes (post_id, code) "
                    "SELECT ?, ? WHERE NOT EXISTS "
                    "(SELECT 1 FROM open_codes WHERE post_id = ? AND code = ?)",
                    (post["post_id"], clean, post["post_id"], clean))
        conn.commit()
        done += 1
        coded_total += len(chunk)
        print(f"   batch {done}: {len(chunk)} posts coded   ({coded_total}/{len(pending)})")
        sys.stdout.flush()

    db.log_run(conn, "pilot", "pilot", len(pending), coded_total,
               notes=f"{who} ({len(existing)} already coded)")
    conn.close()
    _export_codes()
    return 0


def _export_codes() -> None:
    conn = db.connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT post_id, code FROM open_codes ORDER BY post_id, code")]
    conn.close()
    db.export_csv(rows, CODES_CSV)
    print(f"   {len(rows)} codes written to {CODES_CSV.relative_to(db.ROOT)}")


def _cluster(pool: list[dict]) -> list[dict]:
    """Embed the distinct codes, cluster them, and annotate each cluster.

    Cluster: {cluster_id, members: [{code, posts}], post_ids: [...],
              examples: [2 text snippets], posts: n}
    """
    conn = db.connect()
    rows = list(conn.execute(
        "SELECT post_id, code FROM open_codes ORDER BY post_id, code"))
    conn.close()
    if not rows:
        raise SystemExit("no open codes to cluster — run open coding first.")

    by_post = defaultdict(list)
    for r in rows:
        by_post[r["post_id"]].append(r["code"])
    unique = sorted({r["code"] for r in rows})
    code_users = defaultdict(set)
    for r in rows:
        code_users[r["code"]].add(r["post_id"])

    print(f"== clustering {len(unique)} distinct codes from {len(by_post)} posts "
          f"({EMBED_MODEL})")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(unique, show_progress_bar=True, batch_size=64)

    import hdbscan
    labels = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1,
                             metric="euclidean", cluster_selection_method="eom",
                             prediction_data=False).fit_predict(vectors)

    source_of = {p["post_id"]: p["source"] for p in pool}
    text_of = {p["post_id"]: (p["text_original"] or "") for p in pool}
    clusters: dict[int, dict] = {}
    for code, label in zip(unique, labels):
        if label < 0:
            continue                       # noise: not a real cluster
        cluster = clusters.setdefault(int(label), {
            "cluster_id": int(label), "members": [], "post_ids": set()})
        users = code_users[code]
        cluster["members"].append({"code": code, "posts": len(users)})
        cluster["post_ids"] |= users

    kept = []
    for cluster in sorted(clusters.values(), key=lambda c: -len(c["post_ids"])):
        cluster["members"].sort(key=lambda m: -m["posts"])
        cluster["post_ids"] = sorted(cluster["post_ids"])
        if len(cluster["post_ids"]) < MIN_CLUSTER_POSTS:
            continue
        cluster["examples"] = _examples(cluster["post_ids"], text_of)
        kept.append(cluster)

    for i, cluster in enumerate(kept):
        cluster["cluster_id"] = i + 1
    print(f"   {len(kept)} clusters kept (noise folded into other_emerging); "
          f"posts covered by a cluster: {sum(len(c['post_ids']) for c in kept)}")
    for c in kept:
        print(f"   {c['cluster_id']:>2}  {len(c['post_ids']):>3} posts  "
              f"[{', '.join(m['code'] for m in c['members'][:3])}]")
    return kept


def _examples(post_ids: list[str], text_of: dict) -> list[str]:
    """Up to two readable, representative verbatim excerpts for a cluster."""
    usable = []
    for pid in sorted(post_ids, key=lambda p: len(text_of.get(p, ""))):
        text = " ".join(text_of.get(pid, "").split())
        if 15 <= len(text) <= 600:
            usable.append(text[:600])
        if len(usable) >= MAX_EXAMPLES:
            break
    return usable


def _theme_apply(clusters: list[dict]) -> dict:
    from engine.llm import LLMError, QuotaExhausted
    template = NAMING_PROMPT.read_text(encoding="utf-8")
    client = llm.LLM(stage="theme_naming")
    batch_size = int((client.config.get("batch") or {}).get("theme_naming", 8)) or 8
    who = f"{client._provider}/{client.model}"
    print(f"== naming {len(clusters)} clusters on {who}   batches of {batch_size}")

    named: dict[int, ThemeItem] = {}
    for start in range(0, len(clusters), batch_size):
        chunk = clusters[start:start + batch_size]
        group = [{
            "cluster_id": c["cluster_id"],
            "members": c["members"],
            "example_excerpts": c["examples"],
        } for c in chunk]
        prompt = llm.render_prompt(
            template, {"clusters": json.dumps(group, ensure_ascii=False)})
        ids = ",".join(str(c["cluster_id"]) for c in chunk)
        try:
            parsed, stats = client.call_json(
                prompt, ThemeBatch, input_id=ids,
                post_ids=[c["post_ids"][0] for c in chunk])
        except (LLMError, QuotaExhausted) as exc:
            print(f"   stopped: {exc}")
            print(f"   re-run the same command to continue; finished batches are kept.")
            return {}
        for item in parsed.clusters:
            named[item.cluster_id] = item
    return named


def _slugs(labels: dict[int, str]) -> dict[int, str]:
    taken: set[str] = set()
    out: dict[int, str] = {}
    for cid, label in labels.items():
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or f"cluster-{cid}"
        if slug in taken:
            suffix = 2
            while f"{slug}-{suffix}" in taken:
                suffix += 1
            slug = f"{slug}-{suffix}"
        taken.add(slug)
        out[cid] = slug
    return out


def _write_codebook(clusters: list[dict], named: dict[int, ThemeItem],
                    pool: list[dict]) -> int:
    import yaml
    labelled = [c for c in clusters if c["cluster_id"] in named]
    missing = [c for c in clusters if c["cluster_id"] not in named]
    if missing:
        print(f"   {len(missing)} clusters not named yet — re-run to finish.")
        return 1

    source_counts = Counter(p["source"] for p in pool)
    source_of = {p["post_id"]: p["source"] for p in pool}
    slug_of = _slugs({c["cluster_id"]: named[c["cluster_id"]].label for c in labelled})

    codes = []
    covered_posts: set[str] = set()
    review_rows = []
    for c in sorted(labelled, key=lambda c: -len(c["post_ids"])):
        theme = named[c["cluster_id"]]
        slug = slug_of[c["cluster_id"]]
        codes.append({
            "code": slug,
            "label": theme.label,
            "definition": theme.definition,
            "include_when": theme.include_when,
            "exclude_when": theme.exclude_when,
            "example_quotes": theme.example_quotes[:2],
        })
        covered_posts |= set(c["post_ids"])
        sources = Counter(source_of.get(pid, "?") for pid in c["post_ids"])
        review_rows.append({
            "code_id": slug, "label": theme.label,
            "definition": theme.definition,
            "include_when": theme.include_when, "exclude_when": theme.exclude_when,
            "example_quotes": " || ".join(theme.example_quotes[:2]),
            "posts": len(c["post_ids"]),
            "top_sources": ", ".join(f"{s} ({n})" for s, n in sources.most_common(3)),
        })

    codes.append({
        "code": "other_emerging",
        "label": "Other emerging theme",
        "definition": (
            "Posts (or codes the clusterer could not group) that fit no named code. "
            "Used sparingly: if it exceeds 10% of posts the codebook is missing something."),
        "include_when": "Nothing else clearly fits.",
        "exclude_when": "Another code applies.",
        "example_quotes": [],
    })
    review_rows.append({
        "code_id": "other_emerging", "label": "Other emerging theme",
        "definition": codes[-1]["definition"],
        "include_when": "Nothing else clearly fits.",
        "exclude_when": "Another code applies.",
        "example_quotes": "",
        "posts": len(pool) - len(covered_posts),
        "top_sources": ", ".join(f"{s} ({n})" for s, n in source_counts.most_common(3)),
    })

    doc = {
        "version": 1,
        "generated_at": db.utc_now(),
        "approved_by_user": False,
        "pilot_pool": len([p for p in pool]),
        "pilot_sources": dict(source_counts),
        "codes": codes,
    }
    head = ((CODEBOOK_YAML.read_text(encoding="utf-8").split("codes:")[0]
             if CODEBOOK_YAML.exists() else "")
            or "# GENERATED IN MILESTONE 3 — DO NOT HAND-WRITE.\n#\n"
            "# Produced by the pilot run, then you edit it. Set approved_by_user to\n"
            "# true only when you are happy with every code.\n\n")
    CODEBOOK_YAML.write_text(head + yaml.safe_dump(doc, sort_keys=False,
                                                   allow_unicode=True),
                             encoding="utf-8")
    db.export_csv(review_rows, REVIEW_CSV)

    conn = db.connect()
    db.log_run(conn, "pilot", "codebook", len(pool), len(codes))
    conn.close()

    print(f"\n   codebook: {len(codes)} codes "
          f"({len(codes) - 1} named + other_emerging)")
    print(f"   posts covered by a named cluster: {len(covered_posts)} of {len(pool)} "
          f"({100 * len(covered_posts) // len(pool)}%)")
    print(f"   draft written to {CODEBOOK_YAML.relative_to(db.ROOT)} and "
          f"{REVIEW_CSV.relative_to(db.ROOT)}")
    print("   REVIEW AND APPROVE config/codebook.yaml (approved_by_user: true)\n"
          "   The pipeline stops here until you do.")
    return 0


def run(args=None) -> int:
    force = bool(getattr(args, "force", False))
    try:
        pool = _pilot_pool()
    except SystemExit as exc:
        print(f"\n{exc}\n")
        return 1
    print(f"\nPilot pool: {len(pool)} posts that passed in-scope "
          f"({Counter(p['source'] for p in pool)})")

    if _open_coding(pool, force) != 0:
        return 1
    clusters = _cluster(pool)
    if not clusters:
        print("\n   no stable clusters (all labels were noise). Try again after the "
              "top-up, or relax MIN_CLUSTER_POSTS.")
        return 1
    named = _theme_apply(clusters)
    if not named:
        return 1
    return _write_codebook(clusters, named, pool)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="build the draft codebook (Module 3)")
    ap.add_argument("--force", action="store_true",
                    help="re-code posts that already have open codes")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())