"""Tests for src/data/sequences.py, on small hand-built event lists."""
import pandas as pd
import pytest

from src.data.sequences import find_control_recoveries, find_counter_attacks

# Home attacks +x in period 1; Away attacks -x.
DIRECTIONS = {"Home": {1: 1}, "Away": {1: -1}}


def _events(rows: list[tuple]) -> pd.DataFrame:
    """rows: (team, type, start_time_s, start_x_m)."""
    return pd.DataFrame(
        [
            {
                "team": team,
                "type": typ,
                "subtype": "ON TARGET-GOAL" if typ == "SHOT" else None,
                "period": 1,
                "start_frame": round(t * 25),
                "start_time_s": t,
                "start_x": x,
            }
            for team, typ, t, x in rows
        ]
    )


def test_clean_counter_attack_is_found_with_correct_frames():
    events = _events(
        [
            ("Away", "PASS", 9.0, -10.0),
            ("Home", "RECOVERY", 10.0, -30.0),  # Home's own half
            ("Home", "PASS", 12.0, -20.0),
            ("Away", "CHALLENGE", 14.0, 10.0),  # contest without turnover
            ("Home", "SHOT", 18.0, 40.0),
        ]
    )
    result = find_counter_attacks(events, DIRECTIONS)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["attacking_team"] == "Home"
    assert row["defending_team"] == "Away"
    assert (row["start_frame"], row["end_frame"]) == (250, 450)
    assert row["duration_s"] == pytest.approx(8.0)
    assert row["recovery_x_m"] == pytest.approx(-30.0)


def test_recovery_in_opponent_half_is_excluded():
    events = _events([("Home", "RECOVERY", 10.0, 20.0), ("Home", "SHOT", 15.0, 40.0)])
    assert find_counter_attacks(events, DIRECTIONS).empty


def test_away_recovery_uses_away_attack_direction():
    """Away attacks -x, so x=+30 is deep in Away's own half."""
    events = _events([("Away", "RECOVERY", 10.0, 30.0), ("Away", "SHOT", 16.0, -40.0)])
    result = find_counter_attacks(events, DIRECTIONS)
    assert len(result) == 1
    assert result.iloc[0]["recovery_x_m"] == pytest.approx(-30.0)


@pytest.mark.parametrize(
    "breaker",
    [("Home", "SET PIECE"), ("Home", "BALL OUT"), ("Away", "RECOVERY"), ("Home", "BALL LOST")],
)
def test_possession_breaks_disqualify_sequence(breaker):
    events = _events(
        [
            ("Home", "RECOVERY", 10.0, -30.0),
            (breaker[0], breaker[1], 12.0, 0.0),
            ("Home", "SHOT", 16.0, 40.0),
        ]
    )
    assert find_counter_attacks(events, DIRECTIONS).empty


def test_duration_bounds():
    too_fast = _events([("Home", "RECOVERY", 10.0, -5.0), ("Home", "SHOT", 11.0, 40.0)])
    too_slow = _events([("Home", "RECOVERY", 10.0, -30.0), ("Home", "SHOT", 30.0, 40.0)])
    assert find_counter_attacks(too_fast, DIRECTIONS).empty
    assert find_counter_attacks(too_slow, DIRECTIONS).empty


def test_control_ends_at_possession_break():
    events = _events(
        [
            ("Home", "RECOVERY", 10.0, -30.0),
            ("Home", "PASS", 12.0, -20.0),
            ("Home", "BALL LOST", 16.0, 0.0),
            ("Away", "PASS", 17.0, 0.0),
        ]
    )
    result = find_control_recoveries(events, DIRECTIONS)
    assert len(result) == 1
    row = result.iloc[0]
    assert (row["attacking_team"], row["defending_team"]) == ("Home", "Away")
    assert (row["start_frame"], row["end_frame"]) == (250, 400)
    assert row["duration_s"] == pytest.approx(6.0)
    assert row["end_reason"] == "possession_break"


def test_control_capped_at_max_seconds():
    events = _events(
        [
            ("Home", "RECOVERY", 10.0, -30.0),
            ("Home", "PASS", 20.0, -20.0),
            ("Home", "BALL LOST", 40.0, 0.0),
        ]
    )
    row = find_control_recoveries(events, DIRECTIONS, max_seconds=15.0).iloc[0]
    assert row["end_reason"] == "max_seconds"
    assert row["end_time_s"] == pytest.approx(25.0)
    assert row["end_frame"] == 250 + 15 * 25


def test_recoveries_followed_by_a_shot_are_not_controls():
    """Any own shot before the break disqualifies it; even one too slow for a counter."""
    quick = _events([("Home", "RECOVERY", 10.0, -30.0), ("Home", "SHOT", 18.0, 40.0)])
    slow = _events([("Home", "RECOVERY", 10.0, -30.0), ("Home", "PASS", 14.0, 0.0),
                    ("Home", "SHOT", 27.0, 40.0)])
    assert find_control_recoveries(quick, DIRECTIONS).empty
    # The slow shot falls after the 15 s cap, so this one *is* a control.
    assert len(find_control_recoveries(slow, DIRECTIONS)) == 1
    assert find_control_recoveries(slow, DIRECTIONS, max_seconds=20.0).empty


def test_control_rules_match_counter_rules():
    events = _events(
        [
            ("Home", "RECOVERY", 10.0, 20.0),   # opponent's half: excluded
            ("Away", "PASS", 11.0, 0.0),
            ("Away", "BALL OUT", 12.0, 0.0),
            ("Away", "RECOVERY", 30.0, 30.0),   # own half for Away; superseded
            ("Away", "RECOVERY", 32.0, 25.0),   # latest recovery starts the window
            ("Away", "SET PIECE", 33.0, 0.0),   # only 1 s: too short
            ("Home", "RECOVERY", 50.0, -30.0),
            ("Home", "PASS", 55.0, -10.0),      # period's last event
        ]
    )
    result = find_control_recoveries(events, DIRECTIONS)
    assert len(result) == 1
    assert result.iloc[0]["start_time_s"] == 50.0
    assert result.iloc[0]["end_reason"] == "period_end"
    assert result.iloc[0]["duration_s"] == pytest.approx(5.0)
