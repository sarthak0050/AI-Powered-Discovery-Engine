# Fashion Shopping Discovery Engine

Collects public conversations about shopping for clothes online (Myntra) and turns
them into structured, counted, evidence-backed insight — with a quote and a link
behind every number.

It is **not** a sentiment tool and **not** a review summariser. It does not start
from any assumed problem. It finds themes from the data, you approve them, and only
then does it tag and count.

---

## 1. What you need, and it is all free

| What | Why | Cost | Card needed? |
|---|---|---|---|
| Python 3.11+ | runs the engine | free | no |
| Google Gemini API key | does all the reading and tagging | free tier, no expiry | no |
| Groq API key | only used if Gemini is failing | free tier | no |
| YouTube API key | only for YouTube comments (optional) | free, 10,000 units/day | no |

Nothing here needs a paid plan. If any step ever looks like it needs one, that is a
bug — tell me and I will find a free route.

---

## 2. Getting each key, step by step

### Gemini (required — this one does the real work)

1. Go to **https://aistudio.google.com/apikey** and sign in with Google.
2. Click **Create API key**, pick or create a project, confirm.
3. Copy the key. It is a long string.

### Groq (optional, used only as a backup)

1. Go to **https://console.groq.com/keys**, sign in.
2. Click **Create API Key**, copy it. It starts with `gsk_`.

### YouTube (optional, only for YouTube comments)

1. Go to **https://console.cloud.google.com** and create a project.
2. **APIs & Services → Library**, search **YouTube Data API v3**, click **Enable**.
3. **APIs & Services → Credentials → Create Credentials → API key**.
4. Copy it. It starts with `AIza`.

---

## 3. Where to paste them

Put them in the file called **`.env`** in this folder, one per line:

```
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
YOUTUBE_API_KEY=your_key_here
```

That file is already git-ignored, and the code never prints a key — not in errors,
not in logs. If you ever see `***` in an error message, that is the code protecting
you.

`.env.example` is the same file with the values blanked out, for reference.

---

## 4. Running it

Always use the virtual environment. From this folder:

```bash
source .venv/bin/activate          # every time you open a new terminal
```

Check everything is wired up:

```bash
python run.py doctor
```

This prints which keys were found, which sources are enabled, and how many LLM
calls you have used today. Run it any time something looks wrong.

### The stages

| Command | What it does |
|---|---|
| `python run.py collect --source playstore` | Fetch posts. Always shows you one page of raw fields first, so you can see exactly what came back before a full run |
| `python run.py collect --source playstore --preview-only` | Just the one page, then stop |
| `python run.py clean` | Remove duplicates, spam and very short posts; translate Hinglish |
| `python run.py relevance` | Keep only posts about a fashion buying decision (batches of 20) |
| `python run.py pilot` | Build the codebook from 300 posts. **Stops for your approval** |
| `python run.py extract` | Tag every post against the codebook you approved |
| `python run.py label` | Open a page in your browser to hand-label ~100 posts |
| `python run.py validate` | Compare the engine's tags to your labels |
| `python run.py analyze` | Build the analysis tables |
| `python run.py score` | Score the opportunity areas |
| `python run.py dashboard` | Open the dashboard |

Stages are built one milestone at a time. If you run one that is not built yet, it
says so and changes nothing.

---

## 5. Files you can edit without touching code

| File | What it controls |
|---|---|
| `config/sources.yaml` | Which app, which subreddits, which YouTube searches, date range, per-source limits |
| `config/llm.yaml` | Which model, which provider, batch sizes, rate limits, daily quota |
| `config/schema.yaml` | The fields the engine fills in, and the allowed values |
| `config/codebook.yaml` | **The themes.** Generated in Milestone 3, then yours to edit |
| `config/scoring.yaml` | How opportunity areas are scored, and the weights |
| `prompts/*.txt` | Every instruction given to the model. Plain English, editable |

To switch models, edit `config/llm.yaml`. To add a scoring factor, edit
`config/scoring.yaml`.

---

## 6. Where the outputs go

```
data/raw/          untouched API responses, kept for traceability (not committed)
data/engine.db     the pipeline's database
data/exports/      CSV tables you can open in Excel
data/labels/       files for you to review and label
data/errors.csv    anything the model got wrong, so you can see it
logs/run_log.csv   rows in and rows out at every stage
```

You can delete `data/` at any time and re-run: every LLM answer is cached, so a
re-run costs you nothing.

---

## 7. Models, and why these ones

`config/llm.yaml` picks the model. These were tested against the free tier from this
machine, and the ones listed are the ones that actually answered:

| Stage | Model | Why |
|---|---|---|
| Relevance filtering, open coding | `gemini-3.5-flash-lite` | Cheapest and fastest; this is high-volume work |
| Theme naming, extraction, opportunities | `gemini-3.5-flash` | Stronger, and accuracy matters more here |
| Fallback if Gemini fails | Groq `openai/gpt-oss-120b` | Free, used only when Gemini is down |

Two models are deliberately **not** used, both verified from this machine:
`gemini-3.8-flash` returns "high demand" errors on the free tier, and
`gemini-2.5-flash` returns "no longer available to new users". If a future model
name looks better, put it in `config/llm.yaml` and run one stage to test it.

### About "temperature 0"

Your spec asks for temperature 0 so results are repeatable. Newer Gemini endpoints
have dropped the temperature knob, so the engine does something stronger: it sends
the model a **JSON schema** and the API will not return anything that does not match
it. Structure is enforced by the API, not by hoping. The temperature setting is still
sent where the endpoint accepts it.

---

## 8. If something goes wrong

| What you see | What it means | What to do |
|---|---|---|
| `GEMINI_API_KEY is missing` | `.env` is empty or not found | Re-check step 3 above |
| `Daily quota reached` | You used today's free requests | Nothing is lost. Run the same command tomorrow |
| `HTTP 503 ... high demand` | Google is busy, usually temporary | It retries automatically 3 times. If it persists, switch to `gemini-3.5-flash-lite` in `config/llm.yaml` |
| `HTTP 429` | Too fast | It waits and retries by itself. Lower `rpm` in `config/llm.yaml` if it keeps happening |
| iOS shows "unavailable" | Apple's free review feed is empty for every app | Expected. The rest of the pipeline carries on without it |
| Reddit looks empty or errors | That free service has no uptime guarantee | Expected. Play Store is the backbone; lower `page_size` in `config/sources.yaml` |

Nothing in this engine deletes or overwrites your hand labels.

---

## 9. Why the LLM layer calls REST instead of using an SDK

Both Gemini and Groq are plain JSON-over-HTTP, so the engine calls them with
`requests`. That keeps the dependency list smaller and makes the provider switch
easy to read in one file: `engine/llm.py`. If you ever want streaming or async,
swapping in an SDK is a contained change to that one file.

---

## 10. Respecting the sources

- Free, openly offered endpoints only. Nothing here bypasses a login or a paywall.
- Polite user agent, delays between requests, exponential backoff on errors.
- Usernames are hashed before storage — the raw name is never written to disk.
- Google's free tier may use prompts for model training. Your inputs are public app
  reviews and public Reddit posts, so nothing private is sent. Tell me if you would
  rather not accept that and we will route it through Groq instead.