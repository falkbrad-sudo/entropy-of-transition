"""Tests for src/data/*_loader.py.

These are integration tests; they require the real datasets to be present
(via scripts/download_data.sh). Marked so they're skipped by default in
environments without that data, but still runnable explicitly with `pytest -m integration`.

Add `markers = ["integration: requires downloaded data or network access"]`
to pytest.ini / pyproject.toml's [tool.pytest.ini_options] so this marker
doesn't warn on every run.
"""
import pandas as pd
import pytest

from src.data import metrica_loader, statsbomb_loader


@pytest.mark.integration
def test_metrica_tracking_loads_expected_columns():
    df = metrica_loader.load_tracking(game_id=1, team="Home")
    assert list(df.columns) == metrica_loader.TRACKING_COLUMNS
    assert len(df) > 0


@pytest.mark.integration
@pytest.mark.parametrize("game_id", [1, 2, 3])
def test_metrica_player_ids_match_between_tracking_and_events(game_id):
    """Every player named in the events must exist in the tracking data
    (catches spelling quirks like game 2's 'Player 26')."""
    tracking = metrica_loader.load_match_tracking(game_id)
    events = metrica_loader.load_events(game_id)
    event_players = set(events["from_player"].dropna()) | set(events["to_player"].dropna())
    assert event_players <= set(tracking["player_id"])


@pytest.mark.integration
def test_statsbomb_2023_nwsl_matches_returns_gotham_fc():
    matches = statsbomb_loader.get_2023_nwsl_matches()
    all_teams = set(matches["home_team"]) | set(matches["away_team"])
    # Exact naming confirmed against the data: "NJ/NY Gotham FC".
    assert "NJ/NY Gotham FC" in all_teams
    assert len(statsbomb_loader.get_team_matches("NJ/NY Gotham FC")) == 25


@pytest.mark.integration
def test_statsbomb_matches_are_all_from_2023():
    """Executable scope guard: the season analysis is about 2023 only. If
    this ever fails, StatsBomb has changed what this season_id covers and
    METHODOLOGY.md principle 2 should be revisited, not silently ignored.
    """
    matches = statsbomb_loader.get_2023_nwsl_matches()
    assert (matches["match_date"].dt.year == 2023).all()


# --- Unit tests (no data needed): Metrica raw-format parsing helpers ---

def test_metrica_header_pairs_named_and_unnamed_columns_as_x_y():
    raw = ["Period", "Frame", "Time [s]", "Player11", "Unnamed: 4",
           "Player 26", "Unnamed: 6", "Ball", "Unnamed: 8"]
    assert metrica_loader._parse_tracking_columns(raw) == [
        "period", "frame", "time_s", "Player11|x", "Player11|y",
        "Player26|x", "Player26|y", "ball|x", "ball|y",
    ]


def test_metrica_header_rejects_unknown_columns():
    with pytest.raises(ValueError):
        metrica_loader._parse_tracking_columns(["Period", "Frame", "Time [s]", "Speed"])


def test_metrica_wide_to_long_shape_and_ball_team():
    wide = pd.DataFrame({"frame": [1, 2], "period": [1, 1], "time_s": [0.04, 0.08],
                         "Player1|x": [0.1, 0.2], "Player1|y": [0.3, 0.4],
                         "ball|x": [0.5, 0.5], "ball|y": [0.5, 0.6]})
    long = metrica_loader._wide_to_long(wide, "Home", include_ball=True)
    assert len(long) == 4
    assert set(long.loc[long.player_id == "ball", "team"]) == {"Ball"}
    row = long[(long.frame == 2) & (long.player_id == "Player1")].iloc[0]
    assert (row.x, row.y) == (0.2, 0.4)
    assert len(metrica_loader._wide_to_long(wide, "Home", include_ball=False)) == 2


def test_epts_subtypes_join_like_csv_games():
    assert metrica_loader._subtype_name(None) is None
    assert metrica_loader._subtype_name({"name": "BLOCKED", "id": 25}) == "BLOCKED"
    assert metrica_loader._subtype_name(
        [{"name": "ON TARGET", "id": 28}, {"name": "SAVED", "id": 26}]
    ) == "ON TARGET-SAVED"


def test_epts_events_flatten_to_csv_schema():
    raw = {"data": [{
        "team": {"name": "Team B", "id": "FIFATMB"},
        "type": {"name": "SHOT", "id": 2},
        "subtypes": [{"name": "ON TARGET", "id": 28}, {"name": "GOAL", "id": 30}],
        "start": {"frame": 100, "time": 4.0, "x": 0.9, "y": 0.5},
        "end": {"frame": 110, "time": 4.4, "x": None, "y": None},
        "period": 1,
        "from": {"name": "Player 20", "id": "P3587"},
        "to": None,
    }]}
    events = metrica_loader._parse_epts_events(raw, {"FIFATMA": "Home", "FIFATMB": "Away"})
    assert list(events.columns) == list(metrica_loader._EVENT_COLUMN_NAMES.values())
    row = events.iloc[0]
    assert (row.team, row.type, row.subtype) == ("Away", "SHOT", "ON TARGET-GOAL")
    assert row.from_player == "P3587" and row.to_player is None
    assert row.start_frame == 100 and pd.isna(row.end_x)


def test_statsbomb_flatten_events_extracts_location_and_handles_missing():
    raw = [
        {"index": 1, "period": 1, "minute": 0, "second": 0, "type": {"name": "Starting XI"},
         "team": {"name": "NJ/NY Gotham FC"}, "possession_team": {"name": "NJ/NY Gotham FC"}},
        {"index": 2, "period": 1, "minute": 3, "second": 7, "type": {"name": "Pressure"},
         "team": {"name": "NJ/NY Gotham FC"}, "possession_team": {"name": "Angel City"},
         "player": {"id": 42, "name": "A Player"}, "position": {"name": "Center Back"},
         "location": [30.5, 41.0]},
    ]
    events = statsbomb_loader.flatten_events(raw, match_id=123)
    assert list(events.columns) == statsbomb_loader.EVENT_COLUMNS
    assert pd.isna(events.loc[0, "x"]) and pd.isna(events.loc[0, "player_id"])
    row = events.iloc[1]
    assert (row.x, row.y, row.player_id, row.possession_team) == (30.5, 41.0, 42, "Angel City")
    assert (events["match_id"] == 123).all()
