"""Module 3 — pilot and codebook builder.

Draws nothing itself: the pilot pool is whatever `python run.py in-scope --pilot`
drew and screened, and only the posts that pass the locked rubric are coded. Open
coding invents short free-text labels (`prompts/open_coding.txt`), embeddings +
HDBSCAN group the labels into clusters, an LLM names each cluster and writes its
inclusion/exclusion rules (`prompts/theme_naming.txt`), and an LLM consolidates
the named clusters into a two-level codebook of parent themes and child codes
(`prompts/codebook_consolidate.txt`). The pilot pool is then re-tagged against the
consolidated codebook (`prompts/codebook_recode.txt`) so the true other_emerging
rate and per-code / per-source counts measure the codebook you will approve.

The result is a draft `config/codebook.yaml` (version 2, with parent_themes and a
merge_map saying which old code became which new code) plus
`data/exports/codebook_review.csv` and `data/exports/codebook_counts.csv`.

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
CONSOLIDATION_PROMPT = db.ROOT / "prompts" / "codebook_consolidate.txt"
RECODE_PROMPT = db.ROOT / "prompts" / "codebook_recode.txt"
PILOT_CSV = db.ROOT / "data" / "exports" / "pilot_sample.csv"
CODES_CSV = db.ROOT / "data" / "exports" / "open_codes.csv"
CODEBOOK_YAML = db.ROOT / "config" / "codebook.yaml"
REVIEW_CSV = db.ROOT / "data" / "exports" / "codebook_review.csv"
COUNTS_CSV = db.ROOT / "data" / "exports" / "codebook_counts.csv"
MERGE_CSV = db.ROOT / "data" / "exports" / "merge_map.csv"
TAGS_CSV = db.ROOT / "data" / "exports" / "pilot_tags.csv"

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_CODES = 3            # the prompt asks for 1-3 codes per post
MIN_CLUSTER_POSTS = 3    # a cluster must show in at least this many posts
MAX_EXAMPLES = 2
MIN_THEMES, MAX_THEMES = 8, 10
MIN_CHILDREN, MAX_CHILDREN = 25, 35

_embed_client = None     # lazy singleton, reused by consolidation + folding


def _embed():
    global _embed_client
    if _embed_client is None:
        from sentence_transformers import SentenceTransformer
        _embed_client = SentenceTransformer(EMBED_MODEL)
    return _embed_client


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


class ThemePlan(BaseModel):
    id: int
    label: str = Field(min_length=1, max_length=60)


class ChildPlan(BaseModel):
    id: int
    label: str = Field(min_length=1, max_length=40)
    parent_theme_id: int
    anchor_cluster_id: int
    member_cluster_ids: list[int]


class ConsolidationPlan(BaseModel):
    themes: list[ThemePlan]
    children: list[ChildPlan]


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


def _label_sims(model, label: str, labels: list[str]) -> list[float]:
    """Cosine similarity of one label against a list, higher is closer."""
    import numpy as np
    vecs = model.encode([label] + labels, show_progress_bar=False, batch_size=64)
    q = vecs[0]
    q = q / (np.linalg.norm(q) + 1e-9)
    rest = vecs[1:]
    norms = np.linalg.norm(rest, axis=1)
    norms[norms == 0] = 1e-9
    rest = rest / norms[:, None]
    return (rest @ q).tolist()


def _normalize_plan(clusters: list[dict], named: dict[int, ThemeItem],
                    plan: ConsolidationPlan) -> list[dict]:
    """Make the LLM's merge plan structurally sound and cover every cluster.

    The model is asked for 8-10 themes and 25-35 children and to cover every
    cluster once, but it can drift. This layer enforces it deterministically:
    drops junk, de-duplicates members, folds unassigned clusters into their nearest
    child (by label embedding), folds children with fewer than three pilot posts
    into their nearest child, and caps the total at 35. It never invents codes.
    """
    known = {c["cluster_id"] for c in clusters if c["cluster_id"] in named}
    posts_of = {c["cluster_id"]: len(c["post_ids"]) for c in clusters}
    label_of = {cid: named[cid].label for cid in known}
    theme_ids = {t.id for t in plan.themes}

    kids: list[dict] = []
    for ch in plan.children:
        if ch.parent_theme_id not in theme_ids:
            print(f"   plan: child {ch.id} points at an unknown theme; dropped")
            continue
        members = [m for m in ch.member_cluster_ids if m in known]
        if not members:
            print(f"   plan: child {ch.id} covers no known cluster; dropped")
            continue
        kids.append({
            "id": ch.id, "label": ch.label, "parent_theme_id": ch.parent_theme_id,
            "anchor": ch.anchor_cluster_id if ch.anchor_cluster_id in members else members[0],
            "members": members,
        })

    assigned: dict[int, int] = {}
    deduped: list[dict] = []
    for kid in kids:
        keep = [m for m in kid["members"] if m not in assigned]
        if not keep:
            print(f"   plan: child {kid['id']} only re-lists other children's clusters; dropped")
            continue
        for m in keep:
            assigned[m] = len(deduped)
        kid["members"] = keep
        if kid["anchor"] not in keep:
            kid["anchor"] = keep[0]
        deduped.append(kid)

    if not plan.themes:
        print("   plan: no themes returned; forcing a single theme")
        deduped = [dict(kid, parent_theme_id=1) for kid in deduped]
        plan.themes = [ThemePlan(id=1, label="All themes")]

    orphans = sorted(known - set(assigned))
    if orphans:
        model = _embed()
        print(f"   plan: {len(orphans)} clusters were left out — folding them into the "
              f"nearest child by label similarity")
        for cid in orphans:
            sims = _label_sims(model, label_of[cid], [k["label"] for k in deduped])
            if sims:
                best = int(max(range(len(sims)), key=lambda i: sims[i]))
                deduped[best]["members"].append(cid)
                assigned[cid] = best
            elif plan.themes:
                deduped.append({"id": cid + 100000, "label": label_of[cid],
                                "parent_theme_id": plan.themes[0].id,
                                "anchor": cid, "members": [cid]})

    if len(deduped) > 1:
        model = _embed()
        _fold_kids(deduped, model, posts_of)

    if len(deduped) < MIN_CHILDREN:
        print(f"   warning: only {len(deduped)} children after consolidation "
              f"(target {MIN_CHILDREN}-{MAX_CHILDREN}) — re-run to retry the plan.")
    for i, kid in enumerate(deduped):
        kid["id"] = i + 1
    return deduped


def _posts_of(kid: dict, posts_of: dict) -> int:
    return sum(posts_of[m] for m in kid["members"])


def _fold_kids(deduped: list[dict], model, posts_of: dict) -> None:
    """Fold small and surplus children into their nearest neighbour (in place)."""

    def _nearest_other(i: int) -> int | None:
        sims = _label_sims(model, deduped[i]["label"], [k["label"] for k in deduped])
        best, best_score = None, -2.0
        for j, score in enumerate(sims):
            if j != i and score > best_score:
                best, best_score = j, score
        return best

    def _fields(i: int) -> int:
        return sum(posts_of[m] for m in deduped[i]["members"])

    changed = True
    while changed:
        changed = False
        # Fold children with fewer than MIN_CLUSTER_POSTS posts into the nearest child.
        for j in list(range(len(deduped))):
            if len(deduped) <= 1:
                return
            if _fields(j) >= MIN_CLUSTER_POSTS:
                continue
            target = _nearest_other(j)
            if target is None:
                continue
            deduped[target]["members"] += deduped.pop(j)["members"]
            changed = True
            break
        # Cap at MAX_CHILDREN by folding the smallest child into its nearest neighbour.
        if len(deduped) > MAX_CHILDREN:
            small = min(range(len(deduped)), key=lambda i: _fields(i))
            target = _nearest_other(small)
            if target is not None:
                deduped[target]["members"] += deduped.pop(small)["members"]
                changed = True


def _consolidate(clusters: list[dict], named: dict[int, ThemeItem]) -> ConsolidationPlan | None:
    """One LLM call groups the named clusters into themes + child codes."""
    from engine.llm import LLMError, QuotaExhausted
    template = CONSOLIDATION_PROMPT.read_text(encoding="utf-8")
    client = llm.LLM(stage="codebook_consolidate")
    who = f"{client._provider}/{client.model}"
    labelled = [c for c in clusters if c["cluster_id"] in named]
    print(f"== consolidating {len(labelled)} named clusters on {who} (one call)")

    group = []
    for c in sorted(labelled, key=lambda c: -len(c["post_ids"])):
        theme = named[c["cluster_id"]]
        group.append({
            "cluster_id": c["cluster_id"], "label": theme.label,
            "definition": theme.definition, "include_when": theme.include_when,
            "exclude_when": theme.exclude_when, "posts": len(c["post_ids"]),
            "example_quotes": theme.example_quotes[:2],
        })
    prompt = llm.render_prompt(template, {"clusters": json.dumps(group, ensure_ascii=False)})
    ids = ",".join(str(c["cluster_id"]) for c in labelled)
    try:
        plan, stats = client.call_json(prompt, ConsolidationPlan, input_id=ids,
                                       post_ids=[c["post_ids"][0] for c in labelled])
    except (LLMError, QuotaExhausted) as exc:
        print(f"   stopped: {exc}\n   re-run the same command to continue.")
        return None
    print(f"   plan from the model: {len(plan.themes)} themes, {len(plan.children)} children")
    return plan


def _build_codebook_v2(clusters: list[dict], named: dict[int, ThemeItem],
                       plan: ConsolidationPlan, pool: list[dict]) -> dict:
    """Normalise the plan and write config/codebook.yaml (version 2)."""
    import yaml
    kids = _normalize_plan(clusters, named, plan)
    if not kids:
        print("   consolidation returned no usable children — re-run after the quota "
              "resets, or inspect prompts/codebook_consolidate.txt")
        return {}

    source_counts = Counter(p["source"] for p in pool)
    posts_of_cluster = {c["cluster_id"]: len(c["post_ids"]) for c in clusters}
    slug_of_cluster = _slugs(
        {c["cluster_id"]: named[c["cluster_id"]].label for c in clusters
         if c["cluster_id"] in named})
    theme_ids_in_use = {kid["parent_theme_id"] for kid in kids}
    themes = [t for t in plan.themes if t.id in theme_ids_in_use]
    theme_slug = _slugs({t.id: t.label for t in themes})
    child_slug = _slugs({kid["id"]: kid["label"] for kid in kids})

    used_themes: dict[int, list[dict]] = defaultdict(list)
    for kid in sorted(kids, key=lambda k: -_posts_of(k, posts_of_cluster)):
        used_themes[kid["parent_theme_id"]].append(kid)

    codes = []
    for theme in themes:
        for kid in used_themes.get(theme.id, []):
            anchor = named[kid["anchor"]]
            codes.append({
                "code": child_slug[kid["id"]],
                "label": kid["label"],
                "parent_theme": theme_slug[theme.id],
                "definition": anchor.definition,
                "include_when": anchor.include_when,
                "exclude_when": anchor.exclude_when,
                "example_quotes": anchor.example_quotes[:2],
            })

    codes.append({
        "code": "other_emerging",
        "label": "Other emerging theme",
        "parent_theme": None,
        "definition": (
            "Posts that fit no named code. Used sparingly: if it exceeds 10% of posts "
            "the codebook is missing something."),
        "include_when": "Nothing else clearly fits.",
        "exclude_when": "Another code applies.",
        "example_quotes": [],
    })

    merge_rows = []
    for kid in kids:
        to = child_slug[kid["id"]]
        for mid in kid["members"]:
            old = slug_of_cluster[mid]
            if len(kid["members"]) == 1 and old == to:
                reason = "kept"
            elif len(kid["members"]) == 1:
                reason = "renamed"
            else:
                reason = "merged"
            merge_rows.append({
                "from": old, "to": to, "reason": reason,
                "posts": posts_of_cluster[mid],
            })

    doc = {
        "version": 2,
        "generated_at": db.utc_now(),
        "approved_by_user": False,
        "pilot_pool": len(pool),
        "pilot_sources": dict(source_counts),
        "parent_themes": [{"theme_id": theme_slug[t.id], "label": t.label}
                          for t in themes],
        "codes": codes,
        "merge_map": merge_rows,
    }
    head = "\n".join(line for line in CODEBOOK_YAML.read_text(encoding="utf-8")
                     .splitlines() if line.lstrip().startswith("#"))
    if not head.strip():
        head = ("# GENERATED IN MILESTONE 3 — DO NOT HAND-WRITE.\n#\n"
                "# Produced by the pilot run, then you edit it. Set approved_by_user to\n"
                "# true only when you are happy with every code.\n\n")
    else:
        head = head.rstrip() + "\n\n"
    CODEBOOK_YAML.write_text(head + yaml.safe_dump(doc, sort_keys=False,
                                                   allow_unicode=True),
                             encoding="utf-8")

    covered: set[str] = set()
    clusters_by_id = {c["cluster_id"]: c for c in clusters}
    for kid in kids:
        for mid in kid["members"]:
            covered |= set(clusters_by_id[mid]["post_ids"])

    print(f"\n   consolidated codebook: {len(themes)} parent themes, "
          f"{len(codes) - 1} child codes (+ other_emerging)")
    print(f"   merge map: {len(merge_rows)} old code -> new code rows")
    print(f"   posts covered by a named child: {len(covered)} of {len(pool)} "
          f"({100 * len(covered) // len(pool)}%) before re-coding")
    print(f"   written to {CODEBOOK_YAML.relative_to(db.ROOT)}")
    return doc


def _recode(pool: list[dict], doc: dict, force: bool) -> dict | None:
    """Re-tag the pilot pool against the consolidated codebook (resumable).

    Returns counts keyed by code, plus the true other_emerging share.
    """
    from engine.llm import LLMError, QuotaExhausted
    template = RECODE_PROMPT.read_text(encoding="utf-8")
    client = llm.LLM(stage="codebook_recode")
    batch_size = int((client.config.get("batch") or {}).get("codebook_recode", 20)) or 20
    who = f"{client._provider}/{client.model}"
    known = {c["code"] for c in doc["codes"]}
    listing = [{
        "code": c["code"], "label": c["label"], "definition": c["definition"],
        "include_when": c.get("include_when", ""), "exclude_when": c.get("exclude_when", ""),
    } for c in doc["codes"] if c["code"] != "other_emerging"]
    print(f"== re-coding {len(pool)} pilot posts against {len(listing)} child codes "
          f"on {who}   batches of {batch_size}")

    conn = db.connect()
    if force:
        conn.execute("DELETE FROM pilot_tags")
        conn.commit()
    existing = {r["post_id"] for r in conn.execute("SELECT post_id FROM pilot_tags")}
    pending = [p for p in pool if force or p["post_id"] not in existing]
    print(f"   {len(pending)} to tag, {len(pool) - len(pending)} already tagged")

    done = 0
    tagged_total = 0
    for start in range(0, len(pending), batch_size):
        chunk = pending[start:start + batch_size]
        payload = [{"id": i + 1, "text": (p["text_original"] or "")[:1200]}
                   for i, p in enumerate(chunk)]
        prompt = llm.render_prompt(template, {
            "codes": json.dumps(listing, ensure_ascii=False),
            "posts": json.dumps(payload, ensure_ascii=False)})
        ids = ",".join(str(p["post_id"]) for p in chunk)
        try:
            parsed, stats = client.call_json(
                prompt, CodeBatch, input_id=ids,
                post_ids=[p["post_id"] for p in chunk])
        except (LLMError, QuotaExhausted) as exc:
            print(f"   stopped: {exc}")
            print(f"   {done} batch(es) written. Re-run the same command to continue.")
            conn.close()
            return None
        real = {i + 1: chunk[i] for i in range(len(chunk))}
        written = 0
        for item in parsed.results:
            post = real.get(item.id)
            if not post:
                continue
            valid = [str(code) for code in item.codes if code in known]
            if not valid:
                valid = ["other_emerging"]
            for code in valid[:4]:
                conn.execute(
                    "INSERT INTO pilot_tags (post_id, code) "
                    "SELECT ?, ? WHERE NOT EXISTS "
                    "(SELECT 1 FROM pilot_tags WHERE post_id = ? AND code = ?)",
                    (post["post_id"], code, post["post_id"], code))
                written += 1
        conn.commit()
        done += 1
        tagged_total += len(chunk)
        print(f"   batch {done}: {len(chunk)} posts tagged   ({tagged_total}/{len(pending)})")
        sys.stdout.flush()

    db.log_run(conn, "pilot", "codebook_recode", len(pending), tagged_total,
               notes=f"{who} ({len(existing)} already tagged)")
    rows = [dict(r) for r in conn.execute("SELECT post_id, code FROM pilot_tags "
                                          "ORDER BY post_id, code")]
    conn.close()
    db.export_csv(rows, TAGS_CSV)
    print(f"   {len(rows)} tags written to {TAGS_CSV.relative_to(db.ROOT)}")

    source_of = {p["post_id"]: p["source"] for p in pool}
    code_posts: dict[str, set] = defaultdict(set)
    for row in rows:
        code_posts[row["code"]].add(row["post_id"])
    by_code = {}
    for entry in doc["codes"]:
        ps = code_posts.get(entry["code"], set())
        by_code[entry["code"]] = {
            "posts": len(ps),
            "sources": Counter(source_of.get(pid, "?") for pid in ps),
        }
    other = code_posts.get("other_emerging", set())
    return {"by_code": by_code, "other_posts": len(other),
            "other_rate": len(other) / len(pool) if pool else 0.0}


def _finish_outputs(doc: dict, counts: dict, pool: list[dict]) -> int:
    """Write the review CSV (parent theme, then count), counts CSV and merge map."""
    sources = dict(Counter(p["source"] for p in pool))
    source_cols = [s for s, _ in sorted(sources.items(), key=lambda kv: -kv[1])]
    theme_order = {t["theme_id"]: i for i, t in enumerate(doc["parent_themes"])}
    by_code = counts["by_code"]

    def rank(entry):
        parent = entry.get("parent_theme")
        return (theme_order.get(parent, 99),
                -by_code.get(entry["code"], {}).get("posts", 0),
                entry["label"].lower())

    ordered = sorted(doc["codes"], key=rank)
    parents_by_id = {t["theme_id"]: t["label"] for t in doc["parent_themes"]}

    review_rows = []
    count_rows = []
    for entry in ordered:
        name = entry["code"]
        info = by_code.get(name, {"posts": 0, "sources": Counter()})
        parent = parents_by_id.get(entry.get("parent_theme"), "")
        review_rows.append({
            "code_id": name, "parent_theme": parent, "label": entry["label"],
            "definition": entry["definition"],
            "include_when": entry.get("include_when", ""),
            "exclude_when": entry.get("exclude_when", ""),
            "example_quotes": " || ".join(entry.get("example_quotes") or []),
            "posts": info["posts"],
            "top_sources": ", ".join(f"{s} ({n})" for s, n in info["sources"].most_common(3)),
        })
        row = {"code": name, "label": entry["label"], "parent_theme": parent,
               "posts": info["posts"]}
        for s in source_cols:
            row[s] = info["sources"].get(s, 0)
        count_rows.append(row)

    db.export_csv(review_rows, REVIEW_CSV)
    db.export_csv(count_rows, COUNTS_CSV)
    db.export_csv(doc["merge_map"], MERGE_CSV)

    conn = db.connect()
    db.log_run(conn, "pilot", "codebook", len(pool), len(doc["codes"]))
    conn.close()

    print(f"\n   review CSV ({REVIEW_CSV.relative_to(db.ROOT)}): sorted by parent "
          f"theme, then by count")
    print(f"   per-code + per-source counts: {COUNTS_CSV.relative_to(db.ROOT)}")
    print(f"   merge map: {MERGE_CSV.relative_to(db.ROOT)}")
    print(f"   other_emerging: {counts['other_posts']} of {len(pool)} posts "
          f"({100 * counts['other_rate']:.1f}%) — target under 10%")
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
    plan = _consolidate(clusters, named)
    if plan is None:
        return 1
    doc = _build_codebook_v2(clusters, named, plan, pool)
    if not doc:
        return 1
    counts = _recode(pool, doc, force)
    if counts is None:
        return 1
    return _finish_outputs(doc, counts, pool)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="build the draft codebook (Module 3)")
    ap.add_argument("--force", action="store_true",
                    help="re-code posts that already have open codes")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())