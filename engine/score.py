"""Group codes into opportunity areas and score them using config/scoring.yaml.

Each factor is normalised to 0-1 and combined by its configured weight. Adding a new
factor means editing scoring.yaml and adding one function here.


Milestone 6 of AGENTS.md. Not built yet.
"""

from engine.db import connect


def run(*args, **kwargs):
    raise NotImplementedError(
        "score is Milestone 6. It has not been built yet — "
        "run the earlier stages first, or ask for this milestone."
    )
