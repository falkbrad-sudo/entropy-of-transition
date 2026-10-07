"""Tests for src/features/voronoi.py, on hand-constructable configurations."""
import numpy as np
import pandas as pd
import pytest

from src.features.voronoi import (
    compute_territory_timeseries,
    compute_voronoi_areas,
    compute_voronoi_cells,
    compute_voronoi_territory,
)

PITCH_AREA = 105.0 * 68.0


def _frame(points: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(points, columns=["team", "player_id", "x", "y"])


def test_two_players_symmetric_about_halfway_split_pitch_in_half():
    """Bisector of (-20, 5) and (20, 5) is the halfway line x = 0."""
    frame = _frame([("Home", "H", -20.0, 5.0), ("Away", "A", 20.0, 5.0)])
    result = compute_voronoi_territory(frame)
    assert result["home_area_m2"] == pytest.approx(PITCH_AREA / 2)
    assert result["away_area_m2"] == pytest.approx(PITCH_AREA / 2)
    assert result["home_fraction"] == pytest.approx(0.5)


def test_asymmetric_pair_gives_exact_bisector_split():
    """Players at x=-40 and x=0 (same y): bisector at x=-20, so Home owns
    the strip x in [-52.5, -20] -> 32.5 m x 68 m.
    """
    frame = _frame([("Home", "H", -40.0, 0.0), ("Away", "A", 0.0, 0.0)])
    result = compute_voronoi_territory(frame)
    assert result["home_area_m2"] == pytest.approx(32.5 * 68.0)


def test_four_quadrant_players_each_own_a_quarter():
    frame = _frame([
        ("Home", "H1", -26.25, -17.0), ("Home", "H2", -26.25, 17.0),
        ("Away", "A1", 26.25, -17.0), ("Away", "A2", 26.25, 17.0),
    ])
    areas = compute_voronoi_areas(frame)
    assert areas.tolist() == pytest.approx([PITCH_AREA / 4] * 4)
    assert list(areas.index) == ["H1", "H2", "A1", "A2"]


def test_random_configuration_tiles_the_pitch():
    rng = np.random.default_rng(0)
    xy = rng.uniform([-52.5, -34], [52.5, 34], size=(22, 2))
    frame = _frame([("Home" if i < 11 else "Away", f"P{i}", x, y)
                    for i, (x, y) in enumerate(xy)])
    areas = compute_voronoi_areas(frame)
    assert areas.sum() == pytest.approx(PITCH_AREA)
    assert (areas > 0).all()


def test_ball_and_nan_rows_are_ignored():
    frame = _frame([("Home", "H", -20.0, 5.0), ("Away", "A", 20.0, 5.0),
                    ("Ball", "ball", 30.0, 0.0), ("Away", "sub", np.nan, np.nan)])
    assert compute_voronoi_territory(frame)["home_fraction"] == pytest.approx(0.5)


def test_player_outside_pitch_is_clamped_and_area_still_tiles():
    frame = _frame([("Home", "H", -55.0, 0.0), ("Away", "A", 10.0, 36.0),
                    ("Away", "B", 52.5, -34.0)])
    assert compute_voronoi_areas(frame).sum() == pytest.approx(PITCH_AREA)


def test_coincident_players_split_their_cell():
    frame = _frame([("Home", "H1", -20.0, 0.0), ("Home", "H2", -20.0, 0.0),
                    ("Away", "A", 20.0, 0.0)])
    areas = compute_voronoi_areas(frame)
    assert areas["H1"] == pytest.approx(PITCH_AREA / 4)
    assert areas["H2"] == pytest.approx(PITCH_AREA / 4)


def test_territory_timeseries_per_frame():
    rows = []
    for frame, home_x in [(1, -20.0), (2, -40.0)]:
        rows += [{"frame": frame, "time_s": frame * 0.04, "team": "Home",
                  "player_id": "H", "x": home_x, "y": 0.0},
                 {"frame": frame, "time_s": frame * 0.04, "team": "Away",
                  "player_id": "A", "x": 20.0 if frame == 1 else 0.0, "y": 0.0}]
    result = compute_territory_timeseries(pd.DataFrame(rows), 1, 2)
    assert list(result["frame"]) == [1, 2]
    assert result["home_area_m2"].tolist() == pytest.approx([PITCH_AREA / 2, 32.5 * 68.0])



def test_territory_timeseries_matches_single_frame_territory():
    """The timeseries skips per-frame DataFrames; values must match
    compute_voronoi_territory frame by frame (NaN and ball rows included)."""
    rng = np.random.default_rng(4)
    rows = []
    for frame in range(1, 11):
        xy = rng.uniform([-52.5, -34], [52.5, 34], size=(22, 2))
        if frame == 3:
            xy[5] = np.nan
        rows += [{"frame": frame, "time_s": frame * 0.04, "team": "Home" if i < 11 else "Away",
                  "player_id": f"P{i}", "x": x, "y": y} for i, (x, y) in enumerate(xy)]
        rows.append({"frame": frame, "time_s": frame * 0.04, "team": "Ball",
                     "player_id": "ball", "x": 0.0, "y": 0.0})
    tracking = pd.DataFrame(rows)
    result = compute_territory_timeseries(tracking, 1, 10)
    for frame, positions in tracking.groupby("frame"):
        expected = compute_voronoi_territory(positions)
        row = result[result["frame"] == frame].iloc[0]
        for key, value in expected.items():
            assert row[key] == pytest.approx(value, rel=1e-12)

def test_cells_are_pitch_halves_for_symmetric_pair():
    frame = _frame([("Home", "H", -20.0, 5.0), ("Away", "A", 20.0, 5.0)])
    cells = compute_voronoi_cells(frame)
    home = cells["H"]
    assert home[:, 0].min() == pytest.approx(-52.5) and home[:, 0].max() == pytest.approx(0.0)
    assert home[:, 1].min() == pytest.approx(-34.0) and home[:, 1].max() == pytest.approx(34.0)


def test_cell_polygons_agree_with_reported_areas():
    rng = np.random.default_rng(1)
    xy = rng.uniform([-52.5, -34], [52.5, 34], size=(10, 2))
    frame = _frame([("Home", f"P{i}", x, y) for i, (x, y) in enumerate(xy)])
    areas = compute_voronoi_areas(frame)
    for pid, poly in compute_voronoi_cells(frame).items():
        x, y = poly[:, 0], poly[:, 1]
        shoelace = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        assert shoelace == pytest.approx(areas[pid])  # positive => counter-clockwise
