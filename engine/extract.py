"""Tag every relevant post against config/schema.yaml and config/codebook.yaml
using prompts/extraction.txt, batches of 10-15.

every evidence_quote is checked to be a verbatim substring of the post.


Milestone 4 of AGENTS.md. Not built yet.
"""

from engine.db import connect


def run(*args, **kwargs):
    raise NotImplementedError(
        "extract is Milestone 4. It has not been built yet — "
        "run the earlier stages first, or ask for this milestone."
    )
