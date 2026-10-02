# AGENTS.md — Fashion Shopping Discovery Engine

> Put this file in the root of the project folder. OpenCode reads it automatically at the start of every session, so it does not need to be pasted again.

## 1. What we are building

An **AI-powered discovery engine** that collects public conversations about online fashion shopping (**scope: Myntra only**; other platforms appear only when Myntra users mention checking them), and turns them into **structured, quantified, evidence-backed insight** about how people decide what to buy, save, compare, postpone or abandon.

The engine is NOT a sentiment analyser and NOT a review summariser. It must:
1. Collect raw posts from multiple public sources.
2. Keep only posts relevant to fashion purchase decisions.
3. Discover themes inductively (bottom-up) from a pilot sample and turn them into a codebook.
4. Tag every post against that codebook into a fixed, structured schema.
5. Quantify: frequencies, segment splits, source splits, co-occurrence.
6. Group findings into opportunity areas and score them with a configurable formula.
7. Report its own quality (relevance precision, tagging accuracy, coverage).

**The engine must be problem-agnostic.** Do not hard-code any assumed user problem, theme, barrier or category anywhere in code or prompts. All categories come from the pilot (Module 3) and live in editable config files.

## 2. Who you are working with

I am a product manager, not a developer. Therefore:
- Explain what you did in plain English after each step: what was built, how to run it, what it produced.
- After every milestone, show me: row counts, 10 sample rows, and anything that looks wrong.
- Never ask me to edit code. Anything I may want to change must live in `config/` (YAML) or `prompts/` (plain text).
- Give me a single command to run each stage (e.g. `python run.py collect --source playstore`).
- Stop at the end of each milestone and wait for my approval before starting the next.
- If something fails, explain the cause in one or two sentences and propose a fix.

## 3. Hard constraints

- **100% free.** Use only free libraries, free API tiers and free hosting. Never add a paid service, a paid scraper or anything requiring a credit card. If a step seems to need one, stop and tell me.
- **Secrets:** API keys live only in a `.env` file (git-ignored). Provide `.env.example`. Never print or commit keys.
- **Respect rate limits and terms:** use official or openly offered endpoints, add delays and exponential backoff, and identify a polite user agent.
- **Resumable:** every long job saves progress and can resume after a crash or quota limit without redoing finished work.
- **Cache every LLM call** (keyed by prompt hash + input id) so re-runs cost nothing.
- **Every stage writes its output** to `data/` as both CSV (for me) and to the SQLite database (for the pipeline).
- **Log counts** at every stage to `logs/run_log.csv`: stage, source, rows in, rows out, timestamp.
- **Deterministic LLM settings:** temperature 0, JSON-only output, validated with Pydantic. Invalid output is retried up to 3 times, then sent to `data/errors.csv`.
- Keep the code simple and readable. Prefer a few clear files over clever abstractions.

## 4. Tech stack (all free)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | Runs locally on my Mac |
| Env and packages | `uv` (or `venv` + `pip`) | Pin versions in `requirements.txt` |
| Play Store reviews | `google-play-scraper` | No key needed |
| App Store reviews | Apple public customer-reviews RSS (JSON) feed | No key needed; only the ~500 most recent reviews per app per country |
| Reddit | Arctic Shift API (`arctic-shift.photon-reddit.com`) | Free, no key; no uptime guarantee, so handle failures gracefully. Reddit's official API no longer offers self-serve keys. |
| YouTube comments | YouTube Data API v3 | Free key from Google Cloud; daily quota, so cache search results |
| Other sources (Quora, X, Instagram, blogs) | Manual import: I paste posts into `data/manual/*.csv` | Do not scrape these |
| LLM (primary) | Google Gemini API free tier, a Flash or Flash-Lite model | Model name lives in config, not code |
| LLM (fallback) | Groq free tier (e.g. `openai/gpt-oss-120b`) | Low daily token cap, so use only as a fallback |
| LLM access layer | One small `llm.py` wrapper with a provider switch | So the provider and model can be changed in config |
| Validation | `pydantic` | Validates every LLM JSON response |
| Embeddings | `sentence-transformers`, a multilingual MiniLM model, run locally | Free; works offline |
| Clustering | `hdbscan` + `umap-learn` (or `BERTopic`) | For the pilot and for clustering unmet needs |
| Storage | SQLite (`data/engine.db`) + CSV exports | No server needed |
| Analysis | `pandas` | |
| Dashboard | `streamlit` + `plotly` | |
| Hosting | GitHub (code) + Streamlit Community Cloud (dashboard) | Both free; the hosted app reads only processed CSVs, never calls APIs |
| Optional automation | GitHub Actions | Free for public repos; only if I ask for scheduled refreshes |

## 5. Project structure

```
discovery-engine/
├── AGENTS.md
├── README.md                # plain-English how-to-run guide
├── run.py                   # single entry point: python run.py <stage> [options]
├── .env.example
├── requirements.txt
├── config/
│   ├── sources.yaml         # apps, subreddits, YouTube queries, search terms, date range, limits
│   ├── llm.yaml             # provider, model, batch size, rate limits
│   ├── schema.yaml          # extraction fields and allowed values
│   ├── codebook.yaml        # generated in Module 3, edited by me
│   └── scoring.yaml         # scoring factors and weights
├── prompts/                 # every LLM prompt as an editable .txt file
│   ├── relevance.txt
│   ├── open_coding.txt
│   ├── theme_naming.txt
│   ├── extraction.txt
│   └── opportunity_synthesis.txt
├── engine/
│   ├── collectors/          # playstore.py, appstore.py, reddit.py, youtube.py, manual.py
│   ├── clean.py
│   ├── relevance.py
│   ├── pilot.py
│   ├── extract.py
│   ├── validate.py
│   ├── analyze.py
│   ├── score.py
│   ├── llm.py
│   └── db.py
├── dashboard/
│   └── app.py               # Streamlit app, reads data/exports only
├── data/
│   ├── raw/  processed/  manual/  labels/  exports/
└── logs/
```

## 6. Data contracts

**raw_posts**: `post_id` (source-prefixed, unique), `source` (playstore | appstore | reddit | youtube | manual), `myntra_explicit` (bool: does the post explicitly mention Myntra), `parent_context` (app name, subreddit, video title), `author_hash` (hashed, never the raw username), `date`, `text`, `rating` (if any), `engagement` (likes or upvotes), `url`, `lang_hint`.

**relevant_posts**: raw_posts + `text_clean`, `text_en` (English normalisation of Hinglish/Hindi), `is_relevant` (bool), `relevance_reason`, `relevance_confidence`.

**tagged_posts**: one row per post, with the fields defined in `config/schema.yaml`. The default schema is below; it describes decision behaviour generically and contains no assumed problem:
- `journey_stage`: browsing | saved_or_wishlisted | in_cart | comparing | postponed | abandoned | purchased | returned | other
- `saving_behaviour`: free text + code from codebook (why the item was saved, if mentioned)
- `codes`: list of codebook codes (1–4 per post)
- `primary_code`
- `stated_reason`: what the user says
- `underlying_reason`: the likely deeper reason, if different (else null)
- `barrier_type`: informational | confidence | social | occasion_timing | decision_comparison | trust | financial | functional | none | other
- `external_sources`: list of places the user mentions checking outside the app
- `comparison_behaviour`: free text or null
- `segment_cues`: { gender, life_stage, city_tier, price_sensitivity, shopping_frequency, category } (null when unknown; never guess)
- `unmet_need`: one normalised sentence, or null
- `severity`: 1–5
- `evidence_quote`: a verbatim quote of 25 words or fewer from the post
- `confidence`: 0–1

**labels** (my hand labels): same fields as tagged_posts, for about 100 posts.

## 7. Modules and definition of done

### Module 1 — Collectors
- One collector per source, all driven by `config/sources.yaml`.
- Config holds the target app (Myntra only), subreddits, YouTube search queries, neutral search terms, date range and per-source maximums.
- Deduplicate on `post_id`. Save raw API responses to `data/raw/` for traceability.
- **Done when** each source writes `raw_posts` rows, the run log shows counts, and I've seen 10 samples per source.

### Module 2 — Clean + relevance filter
- Remove exact and near duplicates (embedding similarity > 0.95) and spam.
- Do **not** remove short posts. A one-word review like "good" is real sentiment from a real customer, and the volume of them is itself a signal. Length is not a quality test.
- Use the LLM to normalise Hinglish to English, keeping the original text.
- Run the LLM relevance check in batches of 20 posts per call, using `prompts/relevance.txt`. Relevant means the post discusses considering, saving, comparing, choosing, delaying, abandoning or buying fashion items online.
- Export 50 kept and 50 dropped posts to `data/labels/relevance_check.csv` for me to review.
- **Done when** `relevant_posts` exists and relevance precision has been measured from my review.

### Module 3 — Pilot and codebook builder
- Draw a stratified sample of 300 relevant posts (balanced across sources).
- Open coding: for each post, the LLM writes 1–3 short free-text descriptive codes with no predefined list (`prompts/open_coding.txt`).
- Embed and cluster the open codes. The LLM names each cluster and drafts a definition, inclusion rule, exclusion rule and 2 example quotes (`prompts/theme_naming.txt`).
- Write a draft `config/codebook.yaml` (target 20–40 codes) plus `data/exports/codebook_review.csv` for me to edit. Always include an `other_emerging` code.
- **Done when** I've approved the codebook. Do not proceed without my approval.

### Module 4 — Extraction + validation
- Tag every relevant post against `schema.yaml` and `codebook.yaml`, using `prompts/extraction.txt`, in batches of 10–15 posts per call.
- Validate every response with Pydantic. `evidence_quote` must be a verbatim substring of the post; reject and retry otherwise.
- Build a simple Streamlit **labelling page** (`dashboard/label.py`) that shows me one post at a time with dropdowns, so I can hand-label 100 posts. It saves to `data/labels/human_labels.csv`.
- `validate.py` compares the LLM against my labels and reports per-field agreement (and Cohen's kappa for `primary_code`) plus a confusion table of the most-confused codes.
- **Done when** agreement on `primary_code` and `barrier_type` is at least 80%, or I accept a lower figure with a note.

### Module 5 — Analysis
Produce these tables in `data/exports/`:
- Code frequency (overall, by source, by segment), as count and % of relevant posts
- Journey stage distribution, and codes by journey stage
- Barrier type distribution, stated vs underlying reason (how often they differ, and the top pairs)
- External sources frequency
- Code co-occurrence matrix (top 20 pairs)
- Unmet needs clustered into themes, with counts and representative quotes
- Coverage table: posts per source and segment, with groups under 100 posts flagged as low-confidence
- Trend by month, where dates allow

### Module 6 — Opportunity synthesis + scoring
- The LLM groups codes and unmet-need clusters into 6–12 opportunity areas (`prompts/opportunity_synthesis.txt`). Each one lists its member codes, so every number is traceable to posts.
- Score each opportunity with factors and weights from `config/scoring.yaml`. Defaults:
  - **frequency:** % of relevant posts
  - **severity:** mean 1–5
  - **proximity_to_purchase:** weight by the journey stage where it occurs; later stages weigh more
  - **non_monetary_addressability:** share of posts whose barrier_type is not financial
  
  Normalise each factor to 0–1, then take the weighted sum. I must be able to add a factor later (e.g. a business metric) by editing YAML only.
- Export `opportunity_scorecard.csv` and `evidence_pack.md` (per opportunity: score breakdown, segments, sources, 5–8 quotes with links).

### Engine quality report
`data/exports/engine_quality.csv` and a dashboard tab showing: relevance precision, tagging agreement, `other_emerging` rate (target < 10%), low-confidence share (target < 15%), coverage gaps, LLM error and retry rate, and total LLM calls used.

### Dashboard (`dashboard/app.py`)
Tabs: Overview (funnel of counts: raw → relevant → tagged) · Themes · Segments · Sources · Unmet needs · Opportunities · Evidence explorer (filter posts by code, segment, source; show quotes and links) · Engine quality.
Reads only from `data/exports/`. Deployable to Streamlit Community Cloud.

## 8. LLM rules
- All prompts live in `prompts/`; code only fills placeholders.
- Every prompt tells the model: use only the text given; do not infer segments without evidence; return null when unknown; return JSON matching the schema exactly.
- Batch requests, respect RPM and RPD from `llm.yaml`, sleep when the quota is hit, and resume the next day if needed.
- Track and print calls used vs the daily quota.

## 9. Working style
- Build one milestone at a time, in the order given in `PROMPTS.md`.
- Write a tiny smoke test for each module (e.g. run on 20 posts) before the full run.
- Keep `README.md` updated with how to run each stage in plain English.