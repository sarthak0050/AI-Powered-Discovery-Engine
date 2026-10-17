# Methodology notes

## The decision subset is defined by journey stage, not by `in_scope`
The pre-purchase decision subset is the posts whose `journey_stage` is one of browsing, saved_or_wishlisted, in_cart, comparing, postponed, abandoned. We do not gate on the `in_scope` flag, because `in_scope` is model-sensitive: on the 619 posts answered by both providers, Gemini and Groq agreed on `in_scope` only 61.6% of the time, and Gemini is the more permissive of the two (it kept 232 posts Groq dropped). `journey_stage` is a structured field read from the post text and is stable across re-runs, so the subset is reproducible. The Module 5 `_decision` tables use this same stage-only definition.

## Extraction validation
- Human labels, 100 posts: `primary_code` 72% agreement (Cohen's kappa 0.70), below the 80% target but accepted because the disagreements are within-family (adjacent codes under the same parent theme); `barrier_type` 87% (kappa 0.81).
- Independent Groq judge, 50 posts: `primary_code` 66% (κ 0.62) / `barrier_type` 86% (κ 0.77).

## Stated vs underlying reason
`stated_reason` and `underlying_reason` differ in only 2.2% of posts, so the underlying-reason field carries limited extra depth and is not a scoring input.

## Coverage caveat
App Store contributes 69 in-scope posts (below 100), so any App Store split is low-confidence and should not be read on its own.

## Why two tracks
The corpus splits by journey stage into two decision contexts that pull in opposite directions, so we score them separately instead of blending them:
- **Track A — Pre-purchase decision** (n=1998; browsing..abandoned). Scores all four factors, including `proximity_to_purchase`, because the question is how close to buying the friction bites.
- **Track B — Trust & retention** (n=4435; purchased/returned, excluding the strength/context codes). These posts are past the decision, so `proximity_to_purchase` is dropped and the remaining factor weights are renormalised to sum to 1.

Both tracks use the same qualification rule (>= 50 posts from >= 2 sources, 95% Wilson CI on frequency) and the same traceability rule (every opportunity lists its member codes).

## Shares overlap because posts are multi-coded
Each post carries 1–4 codes, so a post can count toward several opportunities. Opportunity shares therefore overlap and do not sum to 100%; read each share against its own denominator, not as a partition of the corpus.

## Bridge check (reported as a range)
Within Track A we count posts that cite a past bad experience (delivery, refund, quality, fake product, support) as a reason for hesitating or not buying now. It is a range, not one number: **31% by code** (strict — the post carries one of the five bad-experience codes) to **36% by keyword** (looser — the text mentions a category term, which can include a neutral mention). The union is **40%**. `bridge_check.csv` keeps the code-matched and keyword-matched signals in separate columns.

## Abandoned-stage circularity check
The bridge concentrates in the abandoned stage: 82.4% by code there versus 7.4% across the other pre-purchase stages (91.3% vs 17.1% on the union). Because `abandoned` posts are by construction ones that did not convert, that slice is partly circular — it cannot be read as independent evidence that past bad experiences cause abandonment. We therefore split Track A so the item-level signal can be read without the trust signal.

## A1 / A2 split
We split Track A into **A1 — item-level decision friction** (n=1198: posts that do not match the bridge) and **A2 — trust-driven hesitation** (n=800: the bridge posts). A1 is re-scored with the existing Track A opportunity definitions (no new synthesis), so its shares use A1 as the denominator. A2 is summarised by size and top codes only.
