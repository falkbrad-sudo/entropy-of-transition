"""Tests for src/features/formation.py, on hand-built event tables."""
import numpy as np
import pandas as pd
import pytest

from src.features.formation import (
    compactness_by_match,
    compare_windows,
    compute_average_positions,
    compute_compactness,
    window_difference_test,
)

TEAM = "Gotham"


def _events(rows: list[tuple]) -> pd.DataFrame:
    """rows: (match_id, team, possession_team, player_id, position, x, y)."""
    df = pd.DataFrame(rows, columns=["match_id", "team", "possession_team", "player_id",
                                     "position", "x", "y"])
    df["player_name"] = "P" + df["player_id"].astype(str)
    df["player_id"] = df["player_id"].astype("Int64")
    return df


def test_average_position_is_mean_of_touches_and_phase_filters():
    events = _events([
        (1, TEAM, "Opp", 7, "Center Back", -30.0, 0.0),   # out of possession
        (1, TEAM, "Opp", 7, "Center Back", -20.0, 10.0),  # out of possession
        (1, TEAM, TEAM, 7, "Center Back", 10.0, 0.0),     # in possession: excluded
        (1, "Opp", "Opp", 99, "Center Forward", 0.0, 0.0),  # other team: excluded
    ])
    avg = compute_average_positions(events, TEAM, phase="out_of_possession", min_touches=1)
    assert len(avg) == 1
    row = avg.iloc[0]
    assert (row.x, row.y, row.n_touches) == (-25.0, 5.0, 2)
    all_phases = compute_average_positions(events, TEAM, phase="all", min_touches=1)
    assert all_phases.iloc[0].n_touches == 3


def test_few_touches_flagged_unreliable():
    events = _events([(1, TEAM, "Opp", 7, "Center Back", 0.0, 0.0)])
    avg = compute_average_positions(events, TEAM, min_touches=2)
    assert not avg.iloc[0].reliable


def test_unknown_phase_raises():
    with pytest.raises(ValueError):
        compute_average_positions(_events([]), TEAM, phase="defending")


def _square_positions(half: float, with_gk: bool = True) -> pd.DataFrame:
    rows = [{"player_id": i, "x": sx * half, "y": sy * half, "n_touches": 50,
             "is_goalkeeper": False, "reliable": True}
            for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)])]
    if with_gk:
        rows.append({"player_id": 99, "x": -50.0, "y": 0.0, "n_touches": 80,
                     "is_goalkeeper": True, "reliable": True})
    return pd.DataFrame(rows)


def test_compactness_of_known_square_excludes_goalkeeper():
    """4 outfielders on a 20 m square centred at the origin (+ a GK far back)."""
    result = compute_compactness(_square_positions(10.0))
    assert result["team_length_m"] == pytest.approx(20.0)
    assert result["team_width_m"] == pytest.approx(20.0)
    assert result["centroid_spread_m"] == pytest.approx(np.sqrt(200))
    assert result["block_height_m"] == pytest.approx(0.0)
    assert result["n_players"] == 4


def test_compactness_uses_only_n_most_involved_outfielders():
    positions = _square_positions(10.0, with_gk=False)
    positions.loc[0, "n_touches"] = 1  # least involved player at (-10, -10)
    result = compute_compactness(positions, n_outfield=3)
    assert result["n_players"] == 3
    # Remaining: (-10, 10), (10, -10), (10, 10) -> still 20 x 20 extent
    assert result["team_length_m"] == pytest.approx(20.0)
    assert result["block_height_m"] == pytest.approx(10 / 3)


def test_compactness_with_no_qualifying_players_is_nan():
    positions = _square_positions(10.0).assign(reliable=False)
    result = compute_compactness(positions)
    assert np.isnan(result["team_length_m"]) and result["n_players"] == 0


def _two_match_events() -> pd.DataFrame:
    """Match 1: 4 players on a 20 m square. Match 2: same players on a 40 m square."""
    rows = []
    for match_id, half in [(1, 10.0), (2, 20.0)]:
        for pid, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
            rows += [(match_id, TEAM, "Opp", pid, "Midfield", sx * half, sy * half)] * 3
    return _events(rows)


def test_compare_windows_returns_one_row_per_window():
    result = compare_windows(_two_match_events(), TEAM, {"early": [1], "late": [2]},
                             min_touches=1)
    assert list(result["window"]) == ["early", "late"]
    assert result["team_width_m"].tolist() == pytest.approx([20.0, 40.0])
    assert result["n_matches"].tolist() == [1, 1]
    assert result["n_touches"].tolist() == [12, 12]


def test_compactness_by_match_matches_per_window_result():
    result = compactness_by_match(_two_match_events(), TEAM, min_touches=1)
    assert result["match_id"].tolist() == [1, 2]
    assert result["team_length_m"].tolist() == pytest.approx([20.0, 40.0])


def test_window_difference_matches_hand_computed_welch():
    """a = [1, 2, 3], b = [4, 6, 8]: means 2 and 6, sample variances 1 and 4.
    t = (6 - 2) / sqrt(1/3 + 4/3) = 3.098; Welch df = (5/3)^2 / ((1/3)^2/2 + (4/3)^2/2)
    = 2.941 -> two-sided p = 0.0548 (2 * scipy.stats.t.sf(3.098, 2.941)).
    """
    by_match = pd.DataFrame({"match_id": [1, 2, 3, 4, 5, 6], "m": [1, 2, 3, 4, 6, 8]})
    result = window_difference_test(by_match, "m", [1, 2, 3], [4, 5, 6])
    assert result["mean_a"] == pytest.approx(2.0) and result["mean_b"] == pytest.approx(6.0)
    assert result["difference"] == pytest.approx(4.0)
    assert result["sem_b"] == pytest.approx(2 / np.sqrt(3))
    assert result["p_value"] == pytest.approx(0.0548, abs=1e-4)


def test_window_difference_needs_two_matches_per_window():
    by_match = pd.DataFrame({"match_id": [1, 2, 3], "m": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        window_difference_test(by_match, "m", [1], [2, 3])
