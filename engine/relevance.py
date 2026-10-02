"""LLM relevance filter, batches of 20 posts, using prompts/relevance.txt.

Also exports 50 kept and 50 dropped posts to data/labels/relevance_check.csv so you can
measure relevance precision yourself.


Milestone 2 of AGENTS.md. Not built yet.
"""

from engine.db import connect


def run(*args, **kwargs):
    raise NotImplementedError(
        "relevance is Milestone 2. It has not been built yet — "
        "run the earlier stages first, or ask for this milestone."
    )
