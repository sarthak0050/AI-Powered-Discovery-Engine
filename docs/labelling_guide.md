# Labelling guide: relevance

**File to fill in:** `data/labels/relevance_check.csv`
**Column to fill:** `my_judgement` — put `1` if you agree the post is relevant, `0` if not.
Leave the other columns alone.

You are labelling 100 posts: 50 the engine kept, 50 it dropped. Your answers are how
relevance precision and recall get measured. There are no wrong answers here — if you
disagree with the engine, that is the useful result.

---

## The rule

> **A pure post-purchase complaint is NOT relevant.**
> A complaint becomes **relevant** only when the post links that experience to a
> buying decision.

Two halves, and the second one is what makes a post relevant:

| | |
|---|---|
| **NOT relevant** | The author is only reporting a problem that already happened: delivery, refund, cancellation, customer service, app bugs. The story stops at the complaint. |
| **Relevant** | The author reports a problem *and* says it changed what they will do next time they shop. |

## What counts as "links to a buying decision"

Any one of these is enough:

- **Stop buying** — "I won't order from here again", "I'm deleting the app"
- **Delay** — "I'll wait for the sale", "not buying until this is fixed", postponing an order
- **Switch** — moving to another app, brand or store because of the experience
- **Add checks before buying** — reading size charts carefully, checking seller ratings, comparing prices first, switching to COD, verifying measurements
- **Wishlist / cart behaviour** — something that changes what gets saved, carted or abandoned

The complaint is the *cause*. The changed buying behaviour is the *relevance*. Without
that second half, it is a support ticket, not a shopping decision.

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

That is a real tightening, so expect to disagree with the engine on some rows. In
particular, watch for the engine keeping complaints like *"Full of scam... They
cancelled my order... haven't processed a refund"* — under this rule that should be
`0`, because the author never says they will change how they shop.

Disagreements are the point of this exercise. They will drive the revised prompt if
precision or recall comes in under 85%.

---

## When you are done

Save the file and tell me. Do not fill in any column other than `my_judgement`.
