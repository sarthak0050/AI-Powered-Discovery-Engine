"""Tag every relevant post against config/schema.yaml and config/codebook.yaml
using prompts/extraction.txt. The batch size comes from config/llm.yaml
(batch.extraction = 20): 20 was chosen so the 6,663-post run (~334 calls at
batch 20 vs ~556 at 12) fits one flash-lite daily allowance (450 paced) — the
quota maths in llm.yaml depend on it.

every evidence_quote is checked to be a verbatim substring of the post.


Milestone 4 of AGENTS.md. Not built yet.
"""

from engine.db import connect


def run(*args, **kwargs):
    raise NotImplementedError(
        "extract is Milestone 4. It has not been built yet — "
        "run the earlier stages first, or ask for this milestone."
    )
