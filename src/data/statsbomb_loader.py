"""Loads StatsBomb open event data for the 2023 NWSL season.

IMPORTANT (see METHODOLOGY.md, principle 3): each event gives the position
of only the ONE player on the ball. Shot events also carry a freeze frame
(players in camera view at the instant of the shot), but there is no
continuous position data for all 22 players. Do not use this module's
output to compute per-instant, multi-player entropy or Voronoi control.
It's the right source for season-level, real-team aggregate analysis
(rolling-window average position, shape over time) via
src/features/formation.py.

Source: https://github.com/statsbomb/open-data
competition_id=49, season_id=107 is the free 2023 NWSL season. (The free
release also has NWSL 2018, season_id=3, which this project doesn't use.)
Do not point this at other competition/season IDs without first confirming
they're actually part of the free release, and do not add teams that
didn't exist in the covered season.

Files are read from the local sparse clone made by scripts/download_data.sh
(data/external/statsbomb/data/...), not fetched by statsbombpy at runtime:
the open-data repo layout is stable, local reads are reproducible and
offline, and the pipeline doesn't depend on network access.

Coordinates: StatsBomb's 120 x 80 pitch units, origin top-left, and every
event's location is given from the perspective of the team performing it
(that team attacks toward x = 120 in both halves). A team's own events are
therefore already direction-normalized; no halftime flip is needed, unlike
Metrica tracking data. Verified on the data: both teams' goalkeepers
average x ≈ 7-16 in both halves of 2023 NWSL matches.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

EVENT_COLUMNS = [
    "match_id", "index", "period", "minute", "second", "type", "team",
    "possession_team", "player_id", "player_name", "position", "x", "y",
]


def _data_dir() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["statsbomb_dir"]) / "data"


def get_2023_nwsl_matches() -> pd.DataFrame:
    """Return the match list for the 2023 NWSL season.

    Returns
    -------
    pd.DataFrame
        One row per match, sorted by date: match_id, match_date (datetime),
        match_week, competition_stage, home_team, away_team, home_score,
        away_score. Use match_id to pull events via get_match_events().
    """
    cfg = load_config()
    comp_id = cfg["statsbomb"]["competition_id"]
    season_id = cfg["statsbomb"]["season_id"]
    with open(_data_dir() / "matches" / str(comp_id) / f"{season_id}.json") as f:
        raw = json.load(f)
    matches = pd.DataFrame(
        {
            "match_id": [m["match_id"] for m in raw],
            "match_date": pd.to_datetime([m["match_date"] for m in raw]),
            "match_week": [m["match_week"] for m in raw],
            # StatsBomb's stage names carry stray whitespace ("1st Round ").
            "competition_stage": [m["competition_stage"]["name"].strip() for m in raw],
            "home_team": [m["home_team"]["home_team_name"] for m in raw],
            "away_team": [m["away_team"]["away_team_name"] for m in raw],
            "home_score": [m["home_score"] for m in raw],
            "away_score": [m["away_score"] for m in raw],
        }
    )
    return matches.sort_values(["match_date", "match_id"], ignore_index=True)


def flatten_events(raw_events: list[dict], match_id: int) -> pd.DataFrame:
    """Flatten StatsBomb's nested event JSON into the columns this project uses.

    Events without a location (e.g. Starting XI, Half Start) are kept, with
    NaN x/y, so event counts still match the raw file.
    """
    rows = []
    for ev in raw_events:
        location = ev.get("location") or [None, None]
        rows.append(
            {
                "match_id": match_id,
                "index": ev["index"],
                "period": ev["period"],
                "minute": ev["minute"],
                "second": ev["second"],
                "type": ev["type"]["name"],
                "team": ev["team"]["name"],
                "possession_team": ev["possession_team"]["name"],
                "player_id": ev.get("player", {}).get("id"),
                "player_name": ev.get("player", {}).get("name"),
                "position": ev.get("position", {}).get("name"),
                "x": location[0],
                "y": location[1],
            }
        )
    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    events["x"] = events["x"].astype(float)
    events["y"] = events["y"].astype(float)
    events["player_id"] = events["player_id"].astype("Int64")
    return events


def get_match_events(match_id: int) -> pd.DataFrame:
    """Return full event data for one match.

    Parameters
    ----------
    match_id : int
        From get_2023_nwsl_matches().

    Returns
    -------
    pd.DataFrame
        One row per event (pass, carry, shot, pressure, etc.), with columns
        EVENT_COLUMNS: acting player/team, possession team, period, clock,
        event type, and location (x, y) in StatsBomb's 120x80 pitch units,
        from the acting team's attacking perspective (see module docstring).

    Notes
    -----
    Convert coordinates with cleaning.statsbomb_to_meters() before combining
    with anything derived from Metrica data (different units/pitch size
    conventions); don't mix raw coordinate systems across sources.
    """
    with open(_data_dir() / "events" / f"{match_id}.json") as f:
        return flatten_events(json.load(f), match_id)


def get_team_matches(team_name: str) -> pd.DataFrame:
    """Convenience wrapper: all 2023 NWSL matches involving a given team.

    Parameters
    ----------
    team_name : str
        Must match StatsBomb's team naming exactly. Gotham is
        "NJ/NY Gotham FC" in this release (25 matches: 22 regular season +
        3 playoff).

    Raises
    ------
    ValueError
        If the name matches no team, so a typo fails loudly instead of
        returning an empty frame.
    """
    matches = get_2023_nwsl_matches()
    mask = (matches["home_team"] == team_name) | (matches["away_team"] == team_name)
    if not mask.any():
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        raise ValueError(f"No 2023 NWSL matches for '{team_name}'. Teams: {teams}")
    return matches[mask].reset_index(drop=True)
