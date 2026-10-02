"""Reddit collector via the Arctic Shift API (free, no key).

Verified on this machine: it works but is fragile. It requires a subreddit (it cannot
search across all of Reddit), it degrades to 422 "Timeout. Maybe slow down a bit" and
429 under repeated calls, and limit=100 times out server-side while limit=5 works.
Treat as best-effort: small pages, long delays, exponential backoff, resume from cache.


Milestone 1 of AGENTS.md. Not built yet.
"""

from engine.db import connect


def run(*args, **kwargs):
    raise NotImplementedError(
        "collect is Milestone 1. It has not been built yet — "
        "run the earlier stages first, or ask for this milestone."
    )
