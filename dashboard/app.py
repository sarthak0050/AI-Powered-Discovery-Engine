"""Streamlit dashboard. Reads only from data/exports/ so it can be hosted on
Streamlit Community Cloud without any API keys.

Tabs: Overview, Themes, Segments, Sources, Unmet needs, Opportunities, Evidence
explorer, Engine quality.


Milestone 5 of AGENTS.md. Not built yet.
"""

from engine.db import connect


def run(*args, **kwargs):
    raise NotImplementedError(
        "app is Milestone 5. It has not been built yet — "
        "run the earlier stages first, or ask for this milestone."
    )
