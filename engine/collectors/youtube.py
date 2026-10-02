"""YouTube comments collector via YouTube Data API v3.

Needs YOUTUBE_API_KEY in .env (verified working). Quota is the constraint:
search.list costs 100 units per call, commentThreads.list costs 1 unit per 100 comments,
and the default daily allowance is 10,000 units. Search results must be cached.


Milestone 1 of AGENTS.md. Not built yet.
"""

from engine.db import connect


def run(*args, **kwargs):
    raise NotImplementedError(
        "collect is Milestone 1. It has not been built yet — "
        "run the earlier stages first, or ask for this milestone."
    )
