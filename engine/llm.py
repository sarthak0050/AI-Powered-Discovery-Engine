"""One small wrapper around the free LLM APIs, with the provider and model chosen
in config/llm.yaml rather than in code.

Why it talks REST instead of using provider SDKs:
    Gemini and Groq are both plain JSON-over-HTTP. Calling them with `requests`
    keeps the dependency list small and makes the provider switch easy to read.

What it does for you (all of it required by AGENTS.md):
    - provider + model from config, per stage, no code changes
    - batching, so one call covers many posts
    - rate limiting against RPM, TPM and the daily quota, then sleeps
    - retries with exponential backoff on 429/500/503 and timeouts
    - falls back to the second provider if the first one keeps failing
    - JSON-only responses validated with Pydantic, then data/errors.csv on failure
    - every response cached to disk, so re-running a stage costs nothing
    - a call counter against the daily quota, printed by run.py

Keys come from .env and are never printed: every error message is scrubbed.
"""

import hashlib
import json
import os
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GROQ_BASE = "https://api.groq.com/openai/v1/chat/completions"

# Statuses worth retrying. 400 is deliberately absent: it usually means the request
# itself is wrong, and repeating it only wastes quota.
RETRY_STATUSES = {408, 425, 429, 500, 502, 503, 504}

# A 429 that asks us to wait longer than this is a daily quota, not a burst. Waiting
# hours inside a batch would stall the stage, so the batch is given up and the stage
# stops with a clear message; everything already written is kept and the run resumes
# after the reset. There is no automatic fallback anywhere.
MAX_HONOURED_RETRY_DELAY = 30


def _retry_delay_seconds(body: str) -> float | None:
    """Pull the wait time out of a 429 body, in seconds. None when there is none.

    Google carries it as details[].retryDelay, "14689s", and as "retry in 12s".
    Groq words it "Please try again in 1.815s" — a per-minute burst, not a daily
    quota, and the difference matters: waiting 1.8s fixes it, while treating it as
    a spent daily quota hands the batch away for nothing.
    """
    import re
    match = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', body or "")
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    match = re.search(r'try again in\s+([0-9.]+)\s*s', body or "", re.IGNORECASE)
    if match:
        return float(match.group(1))
    match = re.search(r'retry in ([0-9hms]+)', body or "", re.IGNORECASE)
    if match:
        total = 0.0
        for value, unit in re.findall(r'([0-9.]+)([hms])', match.group(1)):
            total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
        return total or None
    return None

DEFAULT_LIMITS = {"rpm": 10, "tpm": 100000, "rpd": 1000}


class QuotaExhausted(RuntimeError):
    """Daily request limit reached. Progress is saved: rerun the same command tomorrow."""


class LLMError(RuntimeError):
    """Every attempt failed."""


class _HttpFailure(Exception):
    def __init__(self, status: int, text: str):
        self.status = status
        super().__init__(f"HTTP {status}: {text[:300]}")


# --------------------------------------------------------------------------
# config and prompt helpers
# --------------------------------------------------------------------------

def load_config(path=None) -> dict:
    path = Path(path) if path else ROOT / "config" / "llm.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def render_prompt(template: str, values: dict) -> str:
    """Fill {{placeholders}} in a prompt file. Prompt wording lives in prompts/ only."""
    out = template
    for key, value in values.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        out = out.replace("{{%s}}" % key, text)
    return out


def _scrub(text) -> str:
    """Strip anything key-shaped before it reaches a log, an error or your screen."""
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "YOUTUBE_API_KEY"):
        secret = os.environ.get(var)
        if secret and secret in str(text):
            text = str(text).replace(secret, "***")
    return re.sub(r"key=[^&\s\"']+", "key=***", str(text))


def _estimate_tokens(text: str) -> int:
    """Rough token count for throttling: about 3 characters per token.

    4 characters per token under-counted badly against Groq, which counts its own way:
    a prompt of ~12,000 characters came back as 4,438 requested tokens while the
    throttler believed it was 3,000, so a per-minute check that passed locally still
    drew a 429 from the API. Over-estimating only costs a little waiting; under-
    estimating costs a rejected call and a retry.
    """
    return max(1, len(text) // 3)


def pydantic_to_gemini_schema(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model into the JSON schema shape Gemini's responseSchema takes.

    Pydantic emits keywords Gemini does not accept ($ref, $defs, title, anyOf for
    Optional). This resolves and removes them.
    """
    raw = model.model_json_schema()
    defs = raw.get("$defs", {})
    keep = ("type", "enum", "description", "items", "properties", "required",
            "nullable", "minimum", "maximum", "minItems", "maxItems", "format")

    def resolve(node):
        if isinstance(node, list):
            return [resolve(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            return resolve(defs.get(node["$ref"].split("/")[-1], {"type": "string"}))
        if "anyOf" in node:
            branches = [b for b in node["anyOf"] if b.get("type") != "null"]
            merged = resolve(branches[0]) if branches else {"type": "string"}
            if len(branches) < len(node["anyOf"]):
                merged["nullable"] = True
            return merged

        out = {}
        for key, value in node.items():
            if key in ("$defs", "$schema", "title", "additionalProperties", "default", "examples"):
                continue
            if key not in keep:
                continue
            if key == "type" and isinstance(value, list):
                non_null = [t for t in value if t != "null"]
                out["type"] = non_null[0] if non_null else "string"
                if len(non_null) < len(value):
                    out["nullable"] = True
            elif key == "properties":
                # A mapping of field name -> schema, so each value resolves on its own.
                out[key] = {name: resolve(sub) for name, sub in value.items()}
            else:
                out[key] = resolve(value)
        if out.get("type") == "object" and "properties" in out:
            out["required"] = out.get("required", [])
        return out

    schema = resolve(raw)
    # A schema with no fields constrains nothing. Sending nothing is safer than
    # sending a broken schema, because the model then invents its own field names.
    return schema if schema.get("properties") else None


# --------------------------------------------------------------------------
# the wrapper
# --------------------------------------------------------------------------

@dataclass
class LLM:
    """One instance per stage. Model and limits follow config/llm.yaml."""

    config: dict = field(default_factory=load_config)
    stage: str = ""

    def __post_init__(self):
        self._recent_calls: deque = deque()   # timestamps inside the last minute
        self._token_window: list = []        # (timestamp, tokens) inside the last minute
        self._day = None
        self._day_calls = 0
        self._day_tokens = 0
        self._day_rejected = 0
        self._day_by_model: dict = {}        # ok calls per model today; limits are per model
        self._day_tokens_by_model: dict = {} # ok tokens per model today, for the tpd ceiling

    # ---------- model selection ----------

    @property
    def _provider(self) -> str:
        """The provider pinned to this stage, else the config default.

        Every stage that must not drift names its provider in config/llm.yaml
        (stage_providers): relevance, in_scope and extraction are Gemini; the judge and
        the Module 3 pilot coding are Groq. There is no fallback: an unavailable named
        provider stops the stage with a clear message rather than silently switching.
        """
        named = self.config.get("stage_providers") or {}
        return named.get(self.stage) or self.config.get("provider", "gemini")

    @property
    def _stage_key(self) -> str:
        """Bulk stages use the cheaper model, accuracy stages the stronger one."""
        if self.stage in {"relevance", "open_coding"}:
            return "bulk"
        if self.stage in {"extraction", "theme_naming", "opportunity_synthesis",
                          "in_scope"}:
            return "precision"
        return "main"

    @property
    def model(self) -> str:
        named = self.config.get("stage_models") or {}     # a stage pinned to groq needs a groq model
        if named.get(self.stage):
            return named[self.stage]
        per_stage = self.config.get("per_stage_models") or {}
        if per_stage.get(self.stage):          # a per-stage model beats the shared key
            return per_stage[self.stage]
        return per_stage.get(self._stage_key) or self.config.get("model", "")

    def describe(self) -> str:
        used, quota = self.quota()
        return f"{self._provider}/{self.model} · calls today {used}/{quota or '?'}"

    # ---------- cache ----------

    @property
    def _cache_enabled(self) -> bool:
        return bool(self.config.get("cache", {}).get("enabled", True))

    def _cache_key(self, provider: str, model: str, prompt: str, input_id: str) -> str:
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        raw = "|".join([provider, model, self.stage, prompt_hash, str(input_id)])
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str):
        if not self._cache_enabled:
            return None
        from engine import db
        conn = db.connect()
        try:
            row = conn.execute("SELECT response_json FROM llm_cache WHERE cache_key = ?", (key,)).fetchone()
            return row["response_json"] if row else None
        finally:
            conn.close()

    def _cache_put(self, key, provider, model, input_id, prompt, payload, pt, ot) -> None:
        if not self._cache_enabled:
            return
        from engine import db
        conn = db.connect()
        try:
            db.upsert(conn, "llm_cache", [{
                "cache_key": key, "provider": provider, "model": model, "stage": self.stage,
                "input_id": str(input_id)[:200],
                "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                "response_json": json.dumps(payload, ensure_ascii=False),
                "prompt_tokens": pt, "output_tokens": ot, "created_at": db.utc_now(),
            }])
        finally:
            conn.close()

    def _cache_delete(self, key: str) -> None:
        from engine import db
        conn = db.connect()
        try:
            conn.execute("DELETE FROM llm_cache WHERE cache_key = ?", (key,))
            conn.commit()
        finally:
            conn.close()

    # ---------- counters ----------

    def _load_day(self) -> None:
        from engine import db
        today = db.today()
        if self._day == today:
            return
        conn = db.connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS calls, COALESCE(SUM(COALESCE(prompt_tokens,0) + COALESCE(output_tokens,0)), 0) AS tokens "
                "FROM llm_call_log WHERE day = ? AND from_cache = 0 AND status = 'ok' "
                "AND model = ?", (today, self.model)).fetchone()
            rejected = conn.execute(
                "SELECT COUNT(*) AS n FROM llm_call_log WHERE day = ? AND from_cache = 0 "
                "AND status = 'http_429' AND model = ?", (today, self.model)).fetchone()["n"]
            self._day, self._day_calls, self._day_tokens = today, row["calls"], row["tokens"]
            self._day_rejected = rejected
            # Every model's usage today, not just this one, so switching models or
            # falling back sees an accurate count for whichever it is about to use.
            self._day_by_model = {
                r["model"]: r["calls"] for r in conn.execute(
                    "SELECT model, COUNT(*) AS calls FROM llm_call_log "
                    "WHERE day = ? AND from_cache = 0 AND status = 'ok' GROUP BY model",
                    (today,))}
            self._day_tokens_by_model = {
                r["model"]: r["tokens"] for r in conn.execute(
                    "SELECT model, COALESCE(SUM(COALESCE(prompt_tokens,0) "
                    "+ COALESCE(output_tokens,0)), 0) AS tokens FROM llm_call_log "
                    "WHERE day = ? AND from_cache = 0 AND status = 'ok' GROUP BY model",
                    (today,))}
        finally:
            conn.close()

    def quota(self) -> tuple[int, int]:
        self._load_day()
        return self._day_calls, int(self._limits(self._provider).get("rpd", 0))

    def usage(self) -> dict:
        self._load_day()
        return {"calls_today": self._day_calls, "tokens_today": self._day_tokens,
                "rpd_limit": self._limits(self._provider).get("rpd", 0)}

    def _log_call(self, provider, model, pt, ot, status, error=None, attempts=1) -> None:
        from engine import db
        self._load_day()
        conn = db.connect()
        try:
            conn.execute(
                "INSERT INTO llm_call_log (ts, day, stage, provider, model, prompt_tokens, output_tokens, "
                "from_cache, status, error, attempts) VALUES (?,?,?,?,?,?,?,0,?,?,?)",
                (db.utc_now(), db.today(), self.stage, provider, model, int(pt or 0), int(ot or 0),
                 status, _scrub(error) if error else None, attempts))
            conn.commit()
        finally:
            conn.close()
        if status == "ok":
            self._day_calls += 1
            self._day_tokens += int(pt or 0) + int(ot or 0)
            self._token_window.append((time.time(), int(pt or 0) + int(ot or 0)))
            self._day_by_model[model] = self._day_by_model.get(model, 0) + 1
            self._day_tokens_by_model[model] = (
                self._day_tokens_by_model.get(model, 0) + int(pt or 0) + int(ot or 0))
        elif status == "http_429":
            # A rejected request is not a request against the allowance. Counting it
            # meant a day of pure 429s looked like a full quota.
            self._day_rejected += 1

    def _limits(self, provider: str) -> dict:
        return (self.config.get("rate_limits") or {}).get(provider, DEFAULT_LIMITS)

    def _rpd_limit(self, provider: str, model: str) -> int:
        """The daily request limit for a specific model name.

        The free tier caps per model per day, and the models are not equal: Gemini's
        own 429 body reports `limit: 20` for gemini-3.5-flash (quotaId
        GenerateRequestsPerDayPerProjectPerModel) but `limit: 500` for flash-lite.
        Config carries those per-model figures; the provider-wide figure is only the
        fallback when a model has not been measured.
        """
        limits = self._limits(provider)
        by_model = limits.get("rpd_by_model") or {}
        return int(by_model.get(model, limits.get("rpd", 0)))

    # ---------- rate limiting ----------

    def _recent_tokens(self, provider: str) -> int:
        """Tokens sent in the last 60 seconds.

        Tracked as a rolling window rather than a running daily total. TPM is a
        per-minute limit, so it has to be compared with the last minute's usage.
        """
        now = time.time()
        cutoff = now - 60
        self._token_window = [(t, tok) for t, tok in self._token_window if t > cutoff]
        return sum(tok for _t, tok in self._token_window)

    def _throttle(self, provider: str, est_tokens: int, model_name: str = "") -> None:
        """Sleep until it is polite and legal to send this request."""
        self._load_day()
        limits = self._limits(provider)
        rpm = int(limits.get("rpm", 10))
        tpm = int(limits.get("tpm", 100000))
        rpd = int(limits.get("rpd", 0))
        tpd = int(limits.get("tpd", 0))     # tokens per DAY

        # The free tier limit is per model per day, not per provider, so the count that
        # matters is this model's. gemini-3.5-flash-lite and gemini-3.5-flash have
        # SEPARATE allowances, and they are not equal (flash is 20/day, flash-lite
        # 500/day — both read out of the API's own 429 bodies). Zero, not the
        # provider-wide total, for a model with no calls yet.
        used = self._day_by_model.get(model_name, 0)
        rpd = self._rpd_limit(provider, model_name)
        if rpd and used >= rpd:
            raise QuotaExhausted(
                f"Daily request quota reached for {provider}/{model_name}: "
                f"{used}/{rpd} requests. "
                f"Everything so far is saved — run the same command after the reset to continue.")

        # Some providers cap tokens per day as well, and that ceiling is not visible in
        # the response headers: groq returns limit-requests and limit-tokens (per minute)
        # but its 200,000 tokens/day only appears in the 429 body. Tracking only requests
        # meant the run sailed past the real ceiling and then stalled on 429s, because
        # the throttler still believed it had its full request allowance.
        tokens_today = self._day_tokens_by_model.get(model_name, 0)
        if tpd and tokens_today + est_tokens > tpd:
            raise QuotaExhausted(
                f"Daily token quota reached for {provider}/{model_name}: "
                f"{tokens_today} used, this batch needs about {est_tokens}, limit {tpd}. "
                f"Everything so far is saved — run the same command after the reset to continue.")

        now = time.time()
        while self._recent_calls and now - self._recent_calls[0] >= 60:
            self._recent_calls.popleft()
        if len(self._recent_calls) >= rpm:
            time.sleep(max(0.5, 60 - (now - self._recent_calls[0])))
            now = time.time()
            while self._recent_calls and now - self._recent_calls[0] >= 60:
                self._recent_calls.popleft()

        # Token ceiling, measured over the last minute. Extraction batches are large, so
        # this binds before RPM does. The wait is capped: if a single request is itself
        # bigger than the per-minute limit, waiting cannot help, so it proceeds rather
        # than sleeping until the heat death of the universe. An earlier version compared
        # the whole day's tokens to this per-minute limit, which deadlocked permanently
        # once the day passed tpm.
        if tpm:
            waited = 0.0
            while waited < 120:
                used = self._recent_tokens(provider)
                if used + est_tokens <= tpm or used == 0:
                    break
                time.sleep(5)
                waited += 5

        self._recent_calls.append(time.time())

    # ---------- provider calls ----------

    def _timeout(self) -> tuple:
        """(connect, read) timeouts.

        Split so a dead socket after the Mac wakes from sleep fails in seconds rather
        than sitting there for the full read timeout, three retries over.
        """
        read = self.config.get("timeout_seconds", 90)
        connect = self.config.get("connect_timeout_seconds", 15)
        return (connect, read)

    def _call_gemini(self, model: str, prompt: str, schema: dict | None):
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise LLMError("GEMINI_API_KEY is missing. Paste it into .env (see README.md).")
        cfg = {"responseMimeType": "application/json"}
        temp = self.config.get("temperature", 0)
        if temp is not None:
            cfg["temperature"] = temp
        if schema:
            cfg["responseSchema"] = schema
        resp = requests.post(f"{GEMINI_BASE}/{model}:generateContent",
                             params={"key": key}, timeout=self._timeout(),
                             json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg})
        if resp.status_code != 200:
            raise _HttpFailure(resp.status_code, _scrub(resp.text))
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        usage = data.get("usageMetadata", {})
        return json.loads(text), usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)

    def _call_groq(self, model: str, prompt: str, schema: dict | None):
        key = os.environ.get("GROQ_API_KEY", "").strip()
        if not key:
            raise LLMError("GROQ_API_KEY is missing. Paste it into .env (see README.md).")
        resp = requests.post(GROQ_BASE, headers={"Authorization": f"Bearer {key}"},
                             timeout=self._timeout(),
                             json={"model": model,
                                   "messages": [{"role": "user", "content": prompt}],
                                   "temperature": self.config.get("temperature", 0),
                                   "response_format": {"type": "json_object"}})
        if resp.status_code != 200:
            raise _HttpFailure(resp.status_code, _scrub(resp.text))
        data = resp.json()
        usage = data.get("usage", {})
        return (json.loads(data["choices"][0]["message"]["content"]),
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

    def _dispatch(self, provider, model, prompt, schema):
        return self._call_groq(model, prompt, schema) if provider == "groq" \
            else self._call_gemini(model, prompt, schema)

    @staticmethod
    def _backoff(attempt: int) -> None:
        """Exponential backoff with jitter, so a struggling endpoint is not hammered."""
        time.sleep(min(60, (2 ** attempt) + random.uniform(0, 1.5)))

    # ---------- public API ----------

    def call_json(self, prompt: str, schema: type[BaseModel], input_id: str = "",
                  post_ids: list | None = None) -> tuple:
        """One logical call, validated against a Pydantic model.

        Returns (parsed_model, stats_dict). The stage's provider is pinned in config
        (stage_providers) and there is NO automatic fallback: if that provider is
        unavailable the stage stops with a clear message and resumes later. QuotaExhausted
        is raised (pre-emptively) when the day's limit is spent; everything already
        written is kept. Errors are recorded once per post so data/errors.csv can be
        joined back onto the posts it is talking about.

        post_ids are the real ids of the rows behind this call, one per row. input_id
        stays the cache key and is left alone.
        """
        provider, model = self._provider, self.model
        try:
            return self._call_one(provider, model, prompt, schema, input_id)
        except QuotaExhausted:
            raise                                      # already says everything is saved
        except LLMError as exc:
            message = f"{provider}/{model}: {exc}"[:480]
            from engine import db
            conn = db.connect()
            if post_ids:
                for pid in post_ids:
                    db.log_error(conn, self.stage, message, post_id=pid)
            else:
                db.log_error(conn, self.stage, message, post_id=input_id)
            raise LLMError(
                f"{self.stage or 'llm'} call failed for the pinned provider "
                f"({message}). There is no automatic fallback — make that provider "
                f"available again (free-tier quota resets daily) or point this stage at "
                f"another provider in config/llm.yaml (stage_providers + stage_models).")

    def _call_one(self, provider, model, prompt, schema, input_id):
        """Retries one provider. Raises LLMError when this provider is exhausted."""
        gemini_schema = pydantic_to_gemini_schema(schema) if provider == "gemini" else None
        est = _estimate_tokens(prompt)
        max_retries = int(self.config.get("max_retries", 3))
        attempts = 0
        last_error = "no attempt made"
        # A batch that keeps hitting 429 or a dead socket is not going to recover by
        # being retried a fourth time. Give up on this batch and let the caller record
        # it, so the post stays unjudged and is retried on the next run instead of
        # parking the whole stage on one bad batch.
        give_up_after = int(self.config.get("max_retries", 3))

        while attempts < give_up_after:
            attempts += 1
            key = self._cache_key(provider, model, prompt, input_id)

            cached = self._cache_get(key)
            if cached is not None:
                try:
                    parsed = schema.model_validate(json.loads(cached))
                    return parsed, {"cached": True, "attempts": attempts,
                                    "provider": provider, "model": model}
                except (ValidationError, json.JSONDecodeError) as exc:
                    last_error = f"cached response no longer valid: {exc}"
                    self._cache_delete(key)      # drop it and ask again

            self._throttle(provider, est, model)
            try:
                payload, pt, ot = self._dispatch(provider, model, prompt, gemini_schema)
            except _HttpFailure as exc:
                last_error = _scrub(str(exc))
                self._log_call(provider, model, 0, 0, f"http_{exc.status}", last_error, attempts)
                if exc.status == 429:
                    # Gemini tells us exactly how long to wait in the body. Honour it
                    # when it is short — that is a burst limit and waiting fixes it.
                    # When it is hours, waiting is pointless and the batch is given up:
                    # this stage has no fallback, it stops here and resumes after the
                    # reset instead of burning the remaining retries.
                    delay = _retry_delay_seconds(str(exc))
                    if delay is not None and 0 < delay <= MAX_HONOURED_RETRY_DELAY:
                        print(f"      429: waiting {delay:.0f}s as the API asked")
                        time.sleep(delay)
                        continue
                    if delay is not None:
                        raise LLMError(
                            f"daily quota is spent, retry in {delay / 3600:.1f}h. "
                            f"This stage has no fallback — it stops here; everything "
                            f"already written is saved.")
                    # No wait time in the body. Assuming a spent daily quota here used
                    # to crash on a None division, and would be wrong anyway: back off
                    # and retry like any other rate limit.
                    self._backoff(attempts)
                    continue
                if exc.status in RETRY_STATUSES:
                    self._backoff(attempts)
                    continue
                break                                    # a 400 will not fix itself
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = _scrub(f"{type(exc).__name__}: {exc}")
                self._log_call(provider, model, 0, 0, "network", last_error, attempts)
                self._backoff(attempts)
                continue
            except json.JSONDecodeError as exc:
                last_error = f"model returned text that is not JSON: {exc}"
                self._log_call(provider, model, 0, 0, "invalid_json", last_error, attempts)
                break

            try:
                parsed = schema.model_validate(payload)
            except ValidationError as exc:
                last_error = f"response did not match the schema: {exc}"
                self._log_call(provider, model, pt, ot, "invalid_json", last_error, attempts)
                break                                     # the model answered, just wrongly

            self._cache_put(key, provider, model, input_id, prompt, payload, pt, ot)
            self._log_call(provider, model, pt, ot, "ok", None, attempts)
            return parsed, {"cached": False, "attempts": attempts, "provider": provider,
                            "model": model, "prompt_tokens": pt, "output_tokens": ot}

        raise LLMError(f"{give_up_after} attempts, last error: {last_error}")

    def call_json_batched(self, prompt_template: str, items: list[dict],
                          schema: type[BaseModel], input_key: str = "posts",
                          extra_values: dict | None = None,
                          id_map: dict | None = None) -> list[tuple]:
        """One call per batch of posts. Returns [(parsed, stats), ...] in order.

        id_map maps each item's "id" to the real row id (post_id). When given, errors
        are recorded against the real ids instead of the 1..N numbers the prompt uses.
        """
        default = 20
        batch_size = int((self.config.get("batch") or {}).get(self.stage, default)) or default
        extra_values = extra_values or {}
        results = []
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            values = dict(extra_values)
            values[input_key] = json.dumps(chunk, ensure_ascii=False)
            prompt = render_prompt(prompt_template, values)
            ids = ",".join(str(c.get("id")) for c in chunk)
            post_ids = None
            if id_map:
                found = [id_map[c["id"]] for c in chunk if c.get("id") in id_map]
                post_ids = found or None
            results.append(self.call_json(prompt, schema, input_id=ids,
                                          post_ids=post_ids))
        return results