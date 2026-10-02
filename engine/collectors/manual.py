"""Manual import: reads any CSV you drop into data/manual/.

Expected columns: text,date,author,source_note,url. This is the route for Quora, X,
Instagram and blog posts — AGENTS.md forbids scraping those.


Milestone 1 of AGENTS.md. Not built yet.
"""

from engine.db import connect


def run(*args, **kwargs):
    raise NotImplementedError(
        "collect is Milestone 1. It has not been built yet — "
        "run the earlier stages first, or ask for this milestone."
    )
