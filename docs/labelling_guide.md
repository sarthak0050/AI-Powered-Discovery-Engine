# Labelling guide: relevance

This document **is** the rubric. It is what an independent judge reads to decide
whether each of the 100 checked posts is relevant, and it is also the rule the engine's
prompt is written against. If the two disagree, this guide wins.

The 100 posts are 50 the engine kept and 50 it dropped. Their answers measure relevance
precision and recall. There are no wrong answers here — disagreement with the engine is
the useful result.

---

## The rule

> **A post is relevant if it describes any part of a fashion buying journey.**
> A **pure post-purchase complaint is NOT relevant** — unless it links back to that
> journey. The complaint test below is an *exception* to the rule, not the whole rule.

### What counts as part of the buying journey

Relevant when the text describes any one of these, in any direction, whether or not the
purchase happened. This is the engine's own definition from `prompts/relevance.txt`,
written out here so the rubric can be read on its own:

- looking for something to wear, or searching what a platform or brand offers
- saving, wishlisting, shortlisting, or bookmarking an item
- putting something in a cart
- comparing options, or comparing one platform against another
- choosing between alternatives
- deciding not to buy, holding off, or postponing
- abandoning a cart or an order
- buying, ordering, or paying
- returning, refunding, or exchanging something already bought
- describing how such a decision was made, or what happened as a result

**Short, low-information posts about choosing still count as relevant.** *"which one?"*,
*"keep the last one, pl"*, *"full sleeve available or not?"*, *"Buy a Birkenstock"* —
all part of the journey. A post can be relevant while three words long and irrelevant
while three paragraphs long: length and detail are not evidence either way, and a post
does not become `0` for lack of detail.

Scope is unchanged from the prompt: fashion shopping online. Anything outside it —
phones, cars, food, gym, careers — is `0`, and so is chatter, greetings or jokes with
no shopping decision in them.

### The exception: a pure complaint

> **A pure post-purchase complaint is NOT relevant.**
> A complaint becomes **relevant** only when the post links that experience to a
> buying decision.

Two halves, and the second one is what makes a post relevant:

| | |
|---|---|
| **NOT relevant** | The author is only reporting a problem that already happened: delivery, refund, cancellation, customer service, app bugs. The story stops at the complaint. |
| **Relevant** | The author reports a problem *and* says it changed what they will do next time they shop. |

## What counts as "links to a buying decision"

Any one of these is enough — this list is what separates a relevant complaint from a
support ticket, and it is not a test applied to posts that are not complaints:

- **Stop buying** — "I won't order from here again", "I'm deleting the app"
- **Delay** — "I'll wait for the sale", "not buying until this is fixed", postponing an order
- **Switch** — moving to another app, brand or store because of the experience
- **Add checks before buying** — reading size charts carefully, checking seller ratings, comparing prices first, switching to COD, verifying measurements
- **Wishlist / cart behaviour** — something that changes what gets saved, carted or abandoned

The complaint is the *cause*. The changed buying behaviour is the *relevance*. Without
that second half, it is a support ticket, not a shopping decision.

## Returns and refunds

Returning and refunding are on the list above, so a post about them follows the same
two-part shape as everything else: a stated change to future buying makes it `1`, the
process on its own is `0`.

- **Relevant** — the return or exchange drives an explicit buying decision: *"Even when
  an exchange is eligible, they don't resolve the issue despite calling customer care
  multiple times. I'm deleting the app and will never shop from Myntra again"*
  (`appstore:14610049074`); or the author now orders two sizes to keep one, checks the
  return window before adding to cart, or buys somewhere *because* returns are easy.
- **NOT relevant** — a pure return/refund process complaint with no buying link:
  *"I have not received the full refund and they are saying that the platform fee is
  not refunded"* (`playstore:fae1d562-b6cc-49dc-af3c-7081cf685dfd`).

Same shape as every other complaint: the process alone is `0`; a stated change to
future buying is `1`.

---

## 3 examples that are NOT relevant

Pure post-purchase complaints. The author reports a problem and stops there.

### 1. `playstore:e5f62f5c-1f77-4bae-ae0b-b84c9d8bc400`

> product not delivered on time and cancel my order without any reason and update me.

Delivery failure and a cancellation. The author wants an update — not a change in how
they shop next time. **No stated change to future buying.**

### 2. `playstore:d7d552c3-83d6-40b3-af90-48384e34e43f`

> The worst possible experience with the delivery partner. I ordered a shirt and it
> didn't arrive the delivery partner never contacted me I tried calling him but he
> kept giving me excuses and finally cancelled it by himself without informing me.
> The second time I ordered the same shirt the delivery date showed 12 sept and the
> delivery partner did call me and said yes I'll delivery it today by night time and
> suddenly I got a msg that I've requested a return when I didn't.

A detailed account of two failed deliveries. Long and vivid, but it ends at the
complaint — no mention of where they will shop instead, or what they will do
differently. **Length is not evidence of relevance.**

### 3. `youtube:reply:UgwLw1CY3tlccoflkVZ4AaABAg.ANo_sqkolDfANw5lCNPBna`

> 😅😅bhai delevery nhi hoti aise trick karne ke baad

Hinglish: *"Bro, delivery doesn't happen after doing this trick."* A delivery
complaint in a video comment. No buying decision, no change of behaviour. **Being in
Hinglish does not change the rule.**

---

## 3 examples that ARE relevant

The same kind of experience, but each one closes the loop back to a decision about
future buying.

### 1. Stop buying — `appstore:14611182525`

> Myntra is flooded lately with lot of scammers selling low quality products .
> I don’t choose to buy from Myntra anymore

Still a complaint about scams and quality. But **"I don't choose to buy from Myntra
anymore"** is an explicit decision to stop. Relevant.

### 2. Add checks before buying — `appstore:14602860806`

> Very beautiful size is perfect if you are thin take small size but before order
> check your exact measurements

Positive about the product, and it hands the reader a **rule to apply before the next
order**: check your measurements first. Relevant.

### 3. Comparing and choosing — `reddit:post:1wechpb`

> Dudes and Dudettes Plz help me choose 😭 I’ve shortlisted a few black kitten
> heels from Myntra. I want something feminine, dressy and party-friendly,
> since I’d mostly wear them for dinners, with friends or events rather than
> everyday use. Which one would you pick? And if you own any of these, please
> tell me how they are in terms of comfort too.

No complaint at all — a straightforward **comparison and choice** among shortlisted
options. Relevant.

---

## What this changes

The engine's current prompt has an internal contradiction. It lists *"returning,
refunding, or exchanging something already bought"* as relevant (line 20), while
excluding *"delivery, packaging, courier behaviour, customer support"* (line 29).
Posts about returns and refunds fall into both, so the engine decided them
inconsistently.

Your rule settles it: **returning and refunding are relevant only when the post also
says what the author will do next time they buy.**

**The prompt will not be touched until the reference answers are in and the filter has
been measured.** We measure it as it stands first, so precision and recall describe the
system that actually ran. The contradiction is fixed in `prompts/relevance.txt` only
afterwards, and only on your approval.

That is a real tightening, so expect some disagreement with the engine. In
particular, the engine keeps complaints like *"Full of scam... They
cancelled my order... haven't processed a refund"* (`playstore:e504636a...`) or a bare
refund complaint about platform fees (`playstore:fae1d562...`) — under this rule both
should be `0`, because neither author says they will change how they shop.

Disagreements are the point of this exercise. They will drive the revised prompt if
precision or recall comes in under 85%.

---

## How the answers are produced

```
python run.py judge-relevance            # judge the 100 rows against this guide
python -m engine.label_report --judge    # measure precision and recall
```

The judge reads this guide plus the post text — nothing else. Answers land in
`judge_label` and `judge_reason` on `data/labels/relevance_check.csv`. The
`my_judgement` column stays empty, and the blind file plus `python run.py
label-relevance` remain available whenever you want a human cross-check.

## Methodology

The relevance check is a **cross-model validation against a written rubric**, not human
validation.

- **System under test:** Gemini's verdict on each of the 100 rows
  (`engine_said_relevant`, with its `relevance_reason`).
- **Reference answers:** an independent judge — a different model (Groq,
  `openai/gpt-oss-120b`) — applying this guide as the rubric.
- **Independence:** the judge never sees the engine's verdict, confidence, source or
  date, so agreement is two independent reads of the same written rule, not one model
  grading its own homework. The judge is hard-pinned to Groq with no fallback, because
  a fallback to Gemini would silently break that.
- **Reported twice:** across all 100 rows, and excluding the 8 rows quoted as worked
  examples in this guide — the judge was handed those answers inside the rubric, so
  they are not independent evidence.
- **Weighting:** precision, recall and the estimated count of wrongly dropped posts are
  post-stratified to the real source mix of the corpus (playstore is 70% of kept rows,
  appstore 1%), never averaged over the sample.
- **Two-stage filtering:** stage 1 is the broad relevance pass (`prompts/relevance.txt`
  over the whole corpus) — that is what `engine_said_relevant` holds and what the
  metrics above measure. Stage 2 is an `in_scope` check applying this guide to the posts
  stage 1 kept, run during extraction (`python run.py in-scope`). Posts failing stage 2
  are excluded from every analysis table but kept in the database, so a later ruling on
  them is one UPDATE away rather than a lost post.
- **Which model runs which stage:** stage 1 (relevance) and stage 2 (`in_scope` default)
  run on Gemini. The judge, and the Module 3 pilot, ran on Groq
  (`openai/gpt-oss-120b`) because Gemini's daily quota was exhausted on the day the
  pilot was screened — pilot screening is sampling, not measurement, so the swap is
  recorded rather than hidden. Consequence: the 50-row stage-2 validation also came
  back on Groq, the judge's own model, so it would be the judge grading itself. Those
  rows are marked `INVALID` in `in_scope_status` in `data/labels/relevance_check.csv`,
  no stage-2 precision number is reported from them, and they are re-run on Gemini
  (`python run.py in-scope --force`, ~3 calls) after the quota reset.
- **Observed stage-2 pass rate:** on the 300-post pilot draw (seed 7, balanced by
  source), 150 posts passed `in_scope` — 50% overall; by source appstore 36/71,
  playstore 28/77, reddit 45/76, youtube 41/76. Pass rate is expected to sit near the
  stage-1 precision range, not near 100%: stage 1 is deliberately broad.
- **Recall limitation, reported as an interval:** the filter mainly misses short,
  low-information posts — length is not evidence, so terse posts are the ones it drops
  by mistake. The estimated count of wrongly dropped posts is therefore printed as a
  95% confidence interval, not a point estimate: 12-13 sampled rows per source stand
  for strata of thousands, the interval is wide, and that width is the honest form of
  the number.
- **Human labels are still possible:** `my_judgement` is untouched by the judge, so a
  hand-labelled run can be added later as a third opinion.
- **Extraction validation (Module 4) uses the same shape:** about 50 posts, judged on
  `primary_code` and `barrier_type` against `config/codebook.yaml` instead of this
  guide, again by a model that has not seen the system's answer.
