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

from engine.llm import LLM, _scrub, load_config, pydantic_to_gemini_schema, render_prompt

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
        self.assertGreaterEqual(cfg["rate_limits"]["gemini"]["rpd"], 1000)


class TestLive(unittest.TestCase):
    """One real call, then a repeat to prove the disk cache works."""

    def test_live_call_and_cache(self):
        llm = LLM(stage="relevance")
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