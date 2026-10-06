"""Smoke test for engine/llm.py.

Run it with:

    python tests/test_llm_smoke.py

It makes one real (free) call to prove the key works end to end, then proves the
response is cached so a second identical call costs nothing. The offline tests run
without touching the network.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel

from engine import db
from engine.llm import LLM, QuotaExhausted, _scrub, load_config, pydantic_to_gemini_schema, render_prompt

PROMPT = """Does each post discuss deciding whether to buy clothes online? JSON only.

POSTS:
{"results": [
  {"id": 1, "text": "Saved three dresses on Myntra but keep delaying because I cannot tell if the fabric is real"},
  {"id": 2, "text": "The battery on my phone died today"}
]}"""


class RelevanceItem(BaseModel):
    id: int
    text_en: str | None = None
    is_relevant: bool
    relevance_reason: str
    relevance_confidence: float


class RelevanceBatch(BaseModel):
    results: list[RelevanceItem]


class TestOffline(unittest.TestCase):
    """No network. These catch most mistakes before an API call is wasted."""

    def test_daily_quota_429_hands_batch_to_fallback(self):
        """A 429 that asks us to wait hours must not burn retries or stop the run."""
        from engine import llm as llm_mod

        self.assertEqual(llm_mod.MAX_HONOURED_RETRY_DELAY, 30)
        self.assertEqual(
            llm_mod._retry_delay_seconds('{"retryDelay": "14689s"}'), 14689.0)
        self.assertEqual(
            llm_mod._retry_delay_seconds("* Quota exceeded. Please retry in 4h4m49s."), 14689.0)
        self.assertEqual(llm_mod._retry_delay_seconds("Please retry in 20s."), 20.0)
        self.assertIsNone(llm_mod._retry_delay_seconds("no delay in this body"))

        # Build the chain explicitly rather than reading today's config: this test is
        # about the gemini->groq handoff and must keep testing it even while the live
        # config has relevance pointed at groq for the day.
        llm = LLM(stage="relevance")
        llm.config = dict(llm.config, provider="gemini", model="gemini-3.5-flash-lite",
                          per_stage_models={}, fallback_provider="groq",
                          fallback_model="openai/gpt-oss-120b",
                          fallback_on_daily_quota=True)
        llm._day_by_model = {llm.model: 500}       # daily allowance already spent

        def fake_call_one(provider, model, prompt, schema, input_id):
            if provider == "gemini":
                raise llm_mod.QuotaExhausted("gemini spent for today")
            return {"results": []}, {"provider": provider, "model": model}

        llm._call_one = fake_call_one
        parsed, stats = llm.call_json("x", None)
        # The primary being exhausted must not abort the stage.
        self.assertEqual(stats["provider"], "groq")

        llm.config["fallback_on_daily_quota"] = False
        with self.assertRaises(llm_mod.QuotaExhausted):
            llm.call_json("x", None)

        # With no fallback left in the chain, an exhausted quota must stop the stage
        # even when the flag is on, or the run would spin with nowhere to go.
        llm.config = dict(llm.config, fallback_provider=None, fallback_on_daily_quota=True)
        with self.assertRaises(llm_mod.QuotaExhausted):
            llm.call_json("x", None)

    def test_daily_quota_counts_only_ok_calls_and_is_per_model(self):
        """429s are rejections, not usage, and each model has its own 500."""
        llm = LLM(stage="relevance")
        # Zero the counters so the assertion is about this call, not today's real usage
        # already on disk. Start with a known state.
        llm._day_calls, llm._day_rejected = 0, 0
        llm._day_by_model = {}
        llm._day = db.today()          # already "loaded", so _log_call does not re-read
        llm._log_call("gemini", "gemini-3.5-flash-lite", 0, 0, "http_429", "quota", 1)
        self.assertEqual(llm._day_calls, 0)          # not charged
        self.assertEqual(llm._day_rejected, 1)        # but still visible
        self.assertNotIn("gemini-3.5-flash-lite", llm._day_by_model)

        # An unused model must not be blocked by another model's usage.
        llm._day_by_model = {"gemini-3.5-flash-lite": 500}
        llm._throttle("gemini", 500, "gemini-3.5-flash")   # 0/450, must pass
        with self.assertRaises(QuotaExhausted):
            llm._throttle("gemini", 500, "gemini-3.5-flash-lite")

    def test_judged_by_records_the_provider_that_answered(self):
        """A fallback batch must be recorded as groq, not as the configured primary."""
        from engine import relevance as R

        class FakeLLM:
            _provider = "gemini"
            model = "gemini-3.5-flash-lite"
            config = {"batch": {"relevance": 20}}

            def call_json(self, prompt, schema, input_id=None):
                return (R.BatchJudgement(results=[R.Judgement(
                        id=1, is_relevant=False, relevance_reason="x",
                        relevance_confidence=0.9, text_en=None)]),
                        {"provider": "groq", "model": "openai/gpt-oss-120b"})

        posts = [{"post_id": "p1", "text_clean": "some post", "text_en": None}]
        judged = []
        for _i, _c, outcome, stats in R.call_json_batched(
                FakeLLM(), "{{posts}}", [{"id": 1, "text": "some post"}], R.BatchJudgement):
            answered_by = stats["provider"] if stats else None
            self.assertEqual(answered_by, "groq")
            model = f"{answered_by}/{stats['model']}"
            judged.append((answered_by, model))
        self.assertEqual(judged, [("groq", "groq/openai/gpt-oss-120b")])
        # The primary alone must never be what gets recorded.
        self.assertNotIn("gemini", judged[0][0])

    def test_daily_token_quota_is_enforced(self):
        """rpd alone does not describe groq: it also caps tokens per day (200,000)."""
        cfg = load_config()
        groq = cfg["rate_limits"]["groq"]
        # 200000 is groq's real ceiling, from its 429 body. 180000 is that paced down.
        self.assertEqual(groq["tpd"], 180000)
        self.assertEqual(cfg["rate_limits"]["gemini"]["rpd"], 430)

        llm = LLM(stage="relevance",
                  config=dict(cfg, provider="groq", model="openai/gpt-oss-120b",
                              per_stage_models={}, fallback_provider=None))
        model = "openai/gpt-oss-120b"
        llm._day = db.today()
        # Plenty of requests left, but the token pool is nearly gone: must still stop.
        llm._day_by_model = {model: 10}
        llm._day_tokens_by_model = {model: 179_000}
        with self.assertRaises(QuotaExhausted):
            llm._throttle("groq", 4000, model)
        # With room in the token pool it must pass.
        llm._day_tokens_by_model = {model: 1000}
        llm._throttle("groq", 4000, model)

    def test_schema_conversion_strips_pydantic_extras(self):
        schema = pydantic_to_gemini_schema(RelevanceBatch)
        self.assertEqual(schema["type"], "object")
        self.assertIn("results", schema["properties"])
        item = schema["properties"]["results"]["items"]
        self.assertEqual(item["type"], "object")
        self.assertNotIn("title", item)
        self.assertNotIn("$defs", schema)
        self.assertNotIn("anyOf", item["properties"]["text_en"])
        # Optional[str] must reach Gemini as a nullable string, not a union it rejects.
        self.assertEqual(item["properties"]["text_en"]["type"], "string")
        self.assertTrue(item["properties"]["text_en"].get("nullable"))

    def test_prompt_rendering_fills_placeholders(self):
        out = render_prompt("Posts: {{posts}} limit {{n}}", {"posts": "abc", "n": 20})
        self.assertEqual(out, "Posts: abc limit 20")

    def test_secrets_are_scrubbed_from_errors(self):
        import os
        secret = os.environ.get("GEMINI_API_KEY", "")
        if not secret:
            self.skipTest("no key in .env")
        self.assertNotIn(secret, _scrub(f"failed for url .../models?key={secret}&x=1"))

    def test_config_points_at_working_models(self):
        cfg = load_config()
        self.assertIn(cfg["provider"], ("gemini", "groq"))
        self.assertTrue(cfg["model"])
        # The API's own 429 names it: generate_content_free_tier_requests, limit 500 per
        # model per day, and 500 is where it starts 429ing. We sit at 430.
        self.assertEqual(cfg["rate_limits"]["gemini"]["rpd"], 430)
        # Groq's numbers come from its response headers: 1000 requests, 8000 tokens.
        self.assertEqual(cfg["rate_limits"]["groq"]["rpd"], 900)
        self.assertEqual(cfg["rate_limits"]["groq"]["tpm"], 7200)


class TestLive(unittest.TestCase):
    """One real call, then a repeat to prove the disk cache works."""

    def test_live_call_and_cache(self):
        # Pin gemini: this test checks that a gemini key and model still work end to end.
        # Left on the live config it would follow whatever relevance happens to use
        # today, and would stop testing gemini the moment that is switched to groq.
        cfg = dict(load_config())
        cfg.update(provider="gemini", model="gemini-3.5-flash",
                   per_stage_models={"bulk": "gemini-3.5-flash"})
        llm = LLM(stage="relevance", config=cfg)
        print(f"\n  using: {llm.describe()}")

        parsed, stats = llm.call_json(PROMPT, RelevanceBatch, input_id="smoke-1")
        print(f"  call: cached={stats['cached']} attempts={stats['attempts']} "
              f"tokens={stats.get('prompt_tokens')}->{stats.get('output_tokens')}")
        for item in parsed.results:
            print(f"    id={item.id} relevant={item.is_relevant} conf={item.relevance_confidence}")
            print(f"       {item.relevance_reason}")

        self.assertEqual(len(parsed.results), 2)
        by_id = {item.id: item for item in parsed.results}
        self.assertTrue(by_id[1].is_relevant, "the Myntra wishlist post must be judged relevant")
        self.assertFalse(by_id[2].is_relevant, "the phone battery post must be judged irrelevant")
        if stats["cached"]:
            print("  (first call served from cache: an earlier run already asked this exact question)")

        again, stats2 = llm.call_json(PROMPT, RelevanceBatch, input_id="smoke-1")
        self.assertTrue(stats2["cached"], "the second identical call must come from cache")
        self.assertEqual([i.is_relevant for i in again.results],
                         [i.is_relevant for i in parsed.results])
        print("  second identical call served from cache: no quota used")


if __name__ == "__main__":
    result = unittest.main(argv=[sys.argv[0], "-v"], exit=False)
    usage = LLM(stage="relevance").usage()
    print(f"\n  calls used today: {usage['calls_today']}/{usage['rpd_limit']} "
          f"(tokens {usage['tokens_today']})")
    sys.exit(0 if result.result.wasSuccessful() else 1)