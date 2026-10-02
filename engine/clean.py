"""Remove duplicates, spam and very short posts, then translate Hinglish.

Exact dedupe, then near-duplicate removal by embedding similarity > 0.95 using a
multilingual MiniLM model run locally (free, offline).


Milestone 2 of AGENTS.md. Not built yet.
"""

from engine.db import connect


def run(*args, **kwargs):
    raise NotImplementedError(
        "clean is Milestone 2. It has not been built yet — "
        "run the earlier stages first, or ask for this milestone."
    )
