"""Apple App Store collector for the Myntra iOS app.

Verified on this machine: Apple's public customer-reviews RSS feed returns an empty
feed for every app tested (Myntra, Instagram, Meesho). Per the rules you set, this
collector must therefore: use the India storefront, sort most recent, wait 3s between
requests, stop at config limit, raise a clear error on zero reviews rather than saving
an empty file, and mark iOS unavailable so the rest of the pipeline continues.


Milestone 1 of AGENTS.md. Not built yet.
"""

from engine.db import connect


def run(*args, **kwargs):
    raise NotImplementedError(
        "collect is Milestone 1. It has not been built yet — "
        "run the earlier stages first, or ask for this milestone."
    )
