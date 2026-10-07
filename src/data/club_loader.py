"""Adapter stub for a club's own tracking data. NOT YET IMPLEMENTED.

This file intentionally does not connect to any real data source. It exists
to make the project's extension path concrete rather than just asserted in
the README: given access to a club's own tracking feed (SkillCorner,
Second Spectrum, TRACAB, or whatever vendor the club uses), implement the
two functions below to output the exact same standardized schema that
metrica_loader.py already produces. Every function in src/features/ (entropy,
voronoi) and src/viz/ operates on that standardized schema and requires no
changes at all to work with this loader instead.

What you'd actually need to do:
1. Get the raw export format from the club/vendor and read it here.
2. Add a corresponding `<vendor>_to_meters()` conversion function to
   src/data/cleaning.py (same pattern as metrica_to_meters/
   statsbomb_to_meters; usually under 10 lines).
3. Add any needed access config (API credentials, file paths) to
   config.yaml under a new `club:` section.
4. Implement load_tracking() and load_events() below to match the exact
   return schema documented in metrica_loader.py's corresponding functions.

What you would NOT need to touch: src/features/entropy.py,
src/features/voronoi.py, src/viz/*.py, or app/streamlit_app.py; they
already operate on the standardized schema, not on any vendor's raw format.

One real limitation this stub does NOT address: if "club data" means
GPS/wearable training-load data (e.g. Catapult, StatSports) rather than
optical player tracking, that is a different data type entirely (workload
totals, not x/y positions) and would need new feature modules alongside a
new loader, not just a new loader. This stub is specifically for optical
tracking data (x/y positions of all players), the same type as the Metrica
sample data this project currently uses.
"""
from __future__ import annotations

import pandas as pd


def load_tracking(match_id: str, team: str) -> pd.DataFrame:
    """Load a club's tracking data for one match/team. NOT IMPLEMENTED.

    Must return the same schema as metrica_loader.load_tracking(): one row
    per frame, with each player's (x, y) and the ball's (x, y), in whatever
    raw coordinate convention the vendor uses (convert via a new function
    in cleaning.py before this data reaches any feature module).

    Raises
    ------
    NotImplementedError
        Always, until a real data access path exists. Do not stub this out
        with synthetic/fabricated data (see METHODOLOGY.md, principle 1).
    """
    raise NotImplementedError(
        "No club tracking data source is connected. This is an "
        "intentional placeholder for future extension; see this module's "
        "docstring for exactly what implementing it would involve."
    )


def load_events(match_id: str) -> pd.DataFrame:
    """Load a club's event data for one match. NOT IMPLEMENTED.

    Must return the same schema as metrica_loader.load_events().
    """
    raise NotImplementedError(
        "No club event data source is connected. See this module's "
        "docstring for the extension path."
    )
