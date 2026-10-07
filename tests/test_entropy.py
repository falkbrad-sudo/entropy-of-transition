"""Tests for src/features/entropy.py.

Per METHODOLOGY.md conventions: use small, hand-constructable examples with a
known correct answer, not real match data (which no one could hand-verify).
"""
import numpy as np
import pandas as pd
import pytest

from src.features.entropy import (
    compute_dispersion,
    compute_entropy_timeseries,
    compute_grid_entropy,
)


class TestComputeDispersion:
    def test_four_corners_of_square_has_known_rms_distance(self):
        """4 players at the corners of a known square: RMS distance from
        centroid is analytically computable, so this is a real check, not
        just 'does it run'.
        """
        # Corners of a 10m x 10m square centered at the origin.
        positions = pd.DataFrame(
            {"x": [-5, -5, 5, 5], "y": [-5, 5, -5, 5]}
        )
        result = compute_dispersion(positions, attack_axis="x")
        # Distance from center (0,0) to each corner is sqrt(5^2 + 5^2) = 7.071
        expected_rms = np.sqrt(5**2 + 5**2)
        assert result["rms_distance"] == pytest.approx(expected_rms, rel=1e-6)

    def test_all_players_at_same_point_has_zero_dispersion(self):
        """Degenerate case: no spread at all should give exactly zero,
        not a near-zero floating point artifact or a divide-by-zero error.
        """
        positions = pd.DataFrame({"x": [10, 10, 10], "y": [20, 20, 20]})
        result = compute_dispersion(positions)
        assert result["rms_distance"] == pytest.approx(0.0, abs=1e-9)


    def test_axis_variances_separate_stretch_directions(self):
        """A flat line of players across the pitch (constant x) has all its
        spread perpendicular to the attack axis and none along it.
        """
        positions = pd.DataFrame({"x": [0, 0, 0, 0], "y": [-15, -5, 5, 15]})
        result = compute_dispersion(positions, attack_axis="x")
        # Population variance of [-15, -5, 5, 15] = (225 + 25 + 25 + 225) / 4
        assert result["along_axis_variance"] == pytest.approx(0.0, abs=1e-9)
        assert result["perpendicular_variance"] == pytest.approx(125.0)
        assert result["rms_distance"] == pytest.approx(np.sqrt(125.0))

    def test_rms_squared_equals_sum_of_axis_variances(self):
        positions = pd.DataFrame({"x": [-20, -3, 7, 11, 30], "y": [4, -12, 0, 25, -8]})
        result = compute_dispersion(positions)
        assert result["rms_distance"] ** 2 == pytest.approx(
            result["along_axis_variance"] + result["perpendicular_variance"]
        )

    def test_nan_positions_are_ignored(self):
        """Unused substitutes appear as NaN rows in Metrica tracking."""
        positions = pd.DataFrame(
            {"x": [-5, -5, 5, 5, np.nan], "y": [-5, 5, -5, 5, np.nan]}
        )
        result = compute_dispersion(positions)
        assert result["rms_distance"] == pytest.approx(np.sqrt(50))


class TestComputeGridEntropy:
    def test_all_players_in_one_cell_has_zero_entropy(self):
        """If every player occupies the same grid cell, there's no
        uncertainty about where players are -> entropy should be exactly 0.
        """
        # Points clustered around (4, 4): inside the single cell
        # x in [0, 8.75), y in [0, 8.5) of the 12 x 8 grid. (Points straddling
        # the origin would NOT work here; x=0 and y=0 are both bin edges.)
        positions = pd.DataFrame({"x": [4.0, 4.1, 3.9], "y": [4.0, 4.1, 3.9]})
        result = compute_grid_entropy(positions, n_bins_x=12, n_bins_y=8)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_players_spread_across_distinct_cells_has_positive_entropy(self):
        """Players in different, well-separated grid cells should give
        entropy > 0; a coarse sanity check on the direction of the metric,
        not an exact value (since the exact value depends on bin count).
        """
        positions = pd.DataFrame(
            {"x": [-40, -20, 0, 20, 40], "y": [-30, -15, 0, 15, 30]}
        )
        result = compute_grid_entropy(positions, n_bins_x=12, n_bins_y=8)
        assert result > 0.0

    def test_k_players_in_k_distinct_cells_gives_ln_k(self):
        """Uniform occupancy over k cells: H = -k * (1/k) ln(1/k) = ln k."""
        positions = pd.DataFrame(
            {"x": [-40, -20, 0.5, 20, 40], "y": [-30, -15, 0.5, 15, 30]}
        )
        result = compute_grid_entropy(positions, n_bins_x=12, n_bins_y=8)
        assert result == pytest.approx(np.log(5))

    def test_two_to_one_split_gives_known_entropy(self):
        """2 players in one cell, 1 in another: p = (2/3, 1/3)."""
        positions = pd.DataFrame({"x": [4.0, 4.5, -30.0], "y": [4.0, 4.5, -20.0]})
        result = compute_grid_entropy(positions, n_bins_x=12, n_bins_y=8)
        expected = -(2 / 3) * np.log(2 / 3) - (1 / 3) * np.log(1 / 3)
        assert result == pytest.approx(expected)

    def test_point_just_outside_pitch_is_clipped_not_dropped(self):
        positions = pd.DataFrame({"x": [53.0, 52.0], "y": [0.5, 0.5]})
        # Both land in the last x bin -> one occupied cell -> zero entropy.
        assert compute_grid_entropy(positions) == pytest.approx(0.0, abs=1e-9)


def _two_frame_tracking() -> pd.DataFrame:
    """Frame 1: 4 Home players on a 10m square. Frame 2: on a 20m square.
    Plus one Away player and the ball, which must be filtered out.
    """
    rows = []
    for frame, half_side in [(1, 5.0), (2, 10.0)]:
        for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
            rows.append(
                {"frame": frame, "time_s": frame * 0.04, "team": "Home",
                 "player_id": f"H{i}", "x": sx * half_side, "y": sy * half_side}
            )
        rows.append({"frame": frame, "time_s": frame * 0.04, "team": "Away",
                     "player_id": "A0", "x": 40.0, "y": 30.0})
        rows.append({"frame": frame, "time_s": frame * 0.04, "team": "Ball",
                     "player_id": "ball", "x": 0.0, "y": 0.0})
    return pd.DataFrame(rows)


class TestComputeEntropyTimeseries:
    def test_dispersion_per_frame_matches_known_squares(self):
        result = compute_entropy_timeseries(_two_frame_tracking(), "Home", 1, 2)
        assert list(result["frame"]) == [1, 2]
        assert list(result["n_players"]) == [4, 4]
        assert result["rms_distance"].tolist() == pytest.approx(
            [np.sqrt(50), np.sqrt(200)]
        )

    def test_frame_range_is_inclusive_and_filters(self):
        result = compute_entropy_timeseries(_two_frame_tracking(), "Home", 2, 2)
        assert list(result["frame"]) == [2]

    def test_excluded_players_are_dropped(self):
        result = compute_entropy_timeseries(
            _two_frame_tracking(), "Home", 1, 1, exclude_players=["H0"]
        )
        assert result["n_players"].iloc[0] == 3

    def test_grid_entropy_method(self):
        result = compute_entropy_timeseries(
            _two_frame_tracking(), "Home", 1, 2, method="grid_entropy"
        )
        # Frame 2's 20m square puts all 4 players in distinct cells.
        assert result["grid_entropy"].iloc[1] == pytest.approx(np.log(4))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            compute_entropy_timeseries(_two_frame_tracking(), "Home", 1, 2, method="nope")

    def test_vectorized_timeseries_matches_single_frame_functions(self):
        """The timeseries is computed for all frames at once; it must give
        the same values as compute_dispersion / compute_grid_entropy frame by
        frame, including points on bin edges, off the pitch and NaN."""
        rng = np.random.default_rng(3)
        rows = []
        for frame in range(1, 21):
            xy = rng.uniform([-60, -40], [60, 40], size=(10, 2))
            xy[0] = [-52.5 + 105 / 12 * 3, 34.0]   # exactly on an x edge and the far y edge
            if frame % 5 == 0:
                xy[1] = np.nan
            rows += [{"frame": frame, "time_s": frame * 0.04, "team": "Home",
                      "player_id": f"H{i}", "x": x, "y": y} for i, (x, y) in enumerate(xy)]
        tracking = pd.DataFrame(rows)
        shape = compute_entropy_timeseries(tracking, "Home", 1, 20)
        grid = compute_entropy_timeseries(tracking, "Home", 1, 20, method="grid_entropy")
        for frame, positions in tracking.groupby("frame"):
            expected = compute_dispersion(positions)
            row = shape[shape["frame"] == frame].iloc[0]
            for key, value in expected.items():
                assert row[key] == pytest.approx(value, rel=1e-12)
            assert row["n_players"] == positions[["x", "y"]].notna().all(axis=1).sum()
            assert grid.loc[grid["frame"] == frame, "grid_entropy"].iloc[0] == pytest.approx(
                compute_grid_entropy(positions), rel=1e-12)

    def test_same_occupancy_pattern_gives_identical_entropy(self):
        """Cell order must not matter, so equal patterns tie exactly."""
        a = pd.DataFrame({"x": [-50.0, -50.0, 0.0, 0.0, 0.0, 30.0], "y": [0.0] * 6})
        b = pd.DataFrame({"x": [-50.0, 0.0, 0.0, 30.0, 30.0, 30.0], "y": [0.0] * 6})
        assert compute_grid_entropy(a) == compute_grid_entropy(b)  # counts {2,3,1} vs {1,2,3}
