"""Tests for src/data/cleaning.py.

These functions are already implemented (not stubs); this test file should
pass immediately and serves as a working example of the project's testing
convention for whoever (or whatever) picks up the remaining TODOs.
"""
import pandas as pd
import pytest

from src.data.cleaning import (
    attack_direction_by_period,
    infer_goalkeeper,
    metrica_to_meters,
    normalize_attack_direction,
    statsbomb_to_meters,
    validate_coordinates,
)


class TestMetricaToMeters:
    def test_kickoff_point_maps_to_origin(self):
        df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        out = metrica_to_meters(df, "x", "y")
        assert out["x"].iloc[0] == pytest.approx(0.0)
        assert out["y"].iloc[0] == pytest.approx(0.0)

    def test_corner_maps_to_pitch_edge(self):
        df = pd.DataFrame({"x": [1.0], "y": [1.0]})
        out = metrica_to_meters(df, "x", "y")
        assert out["x"].iloc[0] == pytest.approx(52.5)
        assert out["y"].iloc[0] == pytest.approx(34.0)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        metrica_to_meters(df, "x", "y")
        assert df["x"].iloc[0] == 0.5  # original untouched


class TestStatsbombToMeters:
    def test_center_of_statsbomb_pitch_maps_to_origin(self):
        df = pd.DataFrame({"x": [60.0], "y": [40.0]})
        out = statsbomb_to_meters(df, "x", "y")
        assert out["x"].iloc[0] == pytest.approx(0.0)
        assert out["y"].iloc[0] == pytest.approx(0.0)

    def test_corner_maps_to_pitch_edge(self):
        df = pd.DataFrame({"x": [120.0], "y": [80.0]})
        out = statsbomb_to_meters(df, "x", "y")
        assert out["x"].iloc[0] == pytest.approx(52.5)
        assert out["y"].iloc[0] == pytest.approx(34.0)


class TestValidateCoordinates:
    def test_valid_coordinates_do_not_raise(self):
        df = pd.DataFrame({"x": [0.0, 50.0, -50.0], "y": [0.0, 30.0, -30.0]})
        validate_coordinates(df, "x", "y")  # should not raise

    def test_out_of_bounds_x_raises(self):
        df = pd.DataFrame({"x": [1000.0], "y": [0.0]})
        with pytest.raises(ValueError):
            validate_coordinates(df, "x", "y")

    def test_margin_is_configurable(self):
        df = pd.DataFrame({"x": [58.0], "y": [0.0]})  # 5.5 m past the goal line
        with pytest.raises(ValueError):
            validate_coordinates(df, "x", "y")
        validate_coordinates(df, "x", "y", margin_m=10.0)  # should not raise

    def test_catches_unconverted_metrica_coordinates(self):
        """A realistic bug this check should catch: forgetting to convert
        Metrica's 0-1 coordinates before validating against meter bounds.
        0.5 is within [-53.5, 53.5] so this actually would NOT raise --
        this test documents that limitation rather than asserting a raise,
        so a future reader doesn't assume validate_coordinates catches
        every unit-conversion mistake.
        """
        df = pd.DataFrame({"x": [0.5], "y": [0.5]})
        validate_coordinates(df, "x", "y")  # does not raise -- known gap


def _kickoff_tracking() -> pd.DataFrame:
    """Home starts in the -x half in period 1 and the +x half in period 2
    (teams swap ends at halftime). Second frame of each period has Home's
    deepest player 'H_gk' near its own goal.
    """
    rows = []
    for period, side in [(1, -1), (2, 1)]:
        base = 0 if period == 1 else 100
        for frame in (base + 1, base + 2):
            for pid, x in [("H_gk", 50.0), ("H_a", 10.0), ("H_b", 20.0)]:
                rows.append({"period": period, "frame": frame, "team": "Home",
                             "player_id": pid, "x": side * x, "y": side * 5.0})
            rows.append({"period": period, "frame": frame, "team": "Away",
                         "player_id": "A_a", "x": -side * 10.0, "y": 0.0})
    return pd.DataFrame(rows)


class TestAttackDirection:
    def test_infers_swap_at_halftime(self):
        assert attack_direction_by_period(_kickoff_tracking(), "Home") == {1: 1, 2: -1}
        assert attack_direction_by_period(_kickoff_tracking(), "Away") == {1: -1, 2: 1}

    def test_normalized_team_defends_negative_x_in_both_periods(self):
        tracking = _kickoff_tracking()
        directions = attack_direction_by_period(tracking, "Home")
        out = normalize_attack_direction(tracking, "x", "y", directions)
        home = out[out["team"] == "Home"]
        assert (home["x"] < 0).all()

    def test_flip_is_a_rotation_preserving_pairwise_distances(self):
        df = pd.DataFrame({"period": [2, 2], "x": [3.0, -7.0], "y": [4.0, 1.0]})
        out = normalize_attack_direction(df, "x", "y", {2: -1})
        assert out["x"].tolist() == [-3.0, 7.0]
        assert out["y"].tolist() == [-4.0, -1.0]  # y flips too: rotation, not reflection

    def test_missing_period_raises(self):
        df = pd.DataFrame({"period": [3], "x": [0.0], "y": [0.0]})
        with pytest.raises(ValueError):
            normalize_attack_direction(df, "x", "y", {1: 1, 2: -1})

    def test_ambiguous_kickoff_raises(self):
        df = pd.DataFrame({"period": [1, 1], "frame": [1, 1], "team": ["Home"] * 2,
                           "x": [-5.0, 5.0]})
        with pytest.raises(ValueError):
            attack_direction_by_period(df, "Home")


class TestInferGoalkeeper:
    def test_deepest_player_after_normalization_is_goalkeeper(self):
        tracking = _kickoff_tracking()
        out = normalize_attack_direction(
            tracking, "x", "y", attack_direction_by_period(tracking, "Home")
        )
        assert infer_goalkeeper(out, "Home") == "H_gk"
