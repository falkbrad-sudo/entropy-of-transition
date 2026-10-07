"""Tests for src/features/baseline.py, on small hand-built time series."""
import numpy as np
import pandas as pd
import pytest

from src.features.baseline import (
    ball_depth_matched,
    drop_thin_counters,
    matched_changes,
    matched_percentile_test,
    matched_percentiles,
    shape_change,
)


def _series(t, perp, along, entropy=None, fraction=None, ball=None) -> pd.DataFrame:
    n = len(t)
    return pd.DataFrame({
        "t_rel_s": t,
        "perpendicular_variance": perp,
        "along_axis_variance": along,
        "grid_entropy": entropy if entropy is not None else [1.0] * n,
        "defending_fraction": fraction if fraction is not None else [0.5] * n,
        "ball_x_m": ball if ball is not None else [0.0] * n,
    })


def test_shape_change_from_recovery_to_end_and_horizon():
    series = _series(
        t=[-0.04, 0.0, 1.0, 2.0],
        perp=[99.0, 8.0, 4.0, 2.0],      # pre-recovery frame must be ignored
        along=[99.0, 4.0, 6.0, 8.0],
        entropy=[0.0, 2.0, 1.8, 1.5],
        fraction=[0.0, 0.5, 0.45, 0.4],
        ball=[0.0, 10.0, -5.0, -30.0],
    )
    end = shape_change(series)
    assert end["horizon_s"] == 2.0
    assert end["width_log2_ratio"] == pytest.approx(-2.0)   # 8 -> 2: quartered
    assert end["length_log2_ratio"] == pytest.approx(1.0)   # 4 -> 8: doubled
    assert end["grid_entropy_change"] == pytest.approx(-0.5)
    assert end["defending_fraction_change"] == pytest.approx(-0.1)
    assert end["ball_x_m"] == -30.0

    at_1s = shape_change(series, horizon_s=1.0)
    assert at_1s["horizon_s"] == 1.0
    assert at_1s["width_log2_ratio"] == pytest.approx(-1.0)  # 8 -> 4
    assert at_1s["ball_x_m"] == -5.0


def test_shape_change_requires_a_frame_after_recovery():
    with pytest.raises(ValueError, match="t_rel_s"):
        shape_change(_series(t=[-0.08, -0.04], perp=[1.0, 1.0], along=[1.0, 1.0]))


def _window_tables():
    """One counter (2 s) and controls lasting 1, 2 and 3 s."""
    counters = pd.DataFrame({"game_id": [1], "sequence_id": [1], "duration_s": [2.0]})
    counter_series = _series(t=[0.0, 2.0], perp=[8.0, 2.0], along=[4.0, 8.0]).assign(
        game_id=1, sequence_id=1
    )
    controls = pd.DataFrame({"game_id": [1, 1, 2], "control_id": [1, 2, 1],
                             "duration_s": [1.0, 2.0, 3.0]})
    control_series = pd.concat([
        _series(t=[0.0, 1.0], perp=[8.0, 8.0], along=[4.0, 4.0]).assign(game_id=1, control_id=1),
        _series(t=[0.0, 2.0], perp=[8.0, 4.0], along=[4.0, 4.0]).assign(game_id=1, control_id=2),
        # Lasts 3 s but is measured at the counter's 2 s: width 8 -> 8, not 8 -> 1.
        _series(t=[0.0, 2.0, 3.0], perp=[8.0, 8.0, 1.0], along=[4.0, 2.0, 2.0]).assign(
            game_id=2, control_id=1
        ),
    ])
    return counters, counter_series, controls, control_series


def test_matched_changes_uses_only_long_enough_controls_at_the_counter_duration():
    matched = matched_changes(*_window_tables())
    counter = matched[matched["role"] == "counter"]
    ctl = matched[matched["role"] == "control"].sort_values("control_game_id")
    assert len(counter) == 1 and counter["width_log2_ratio"].iloc[0] == pytest.approx(-2.0)
    # The 1 s control is ineligible for a 2 s counter.
    assert list(zip(ctl["control_game_id"], ctl["control_id"])) == [(1, 2), (2, 1)]
    assert (ctl["horizon_s"] == 2.0).all()
    assert ctl["width_log2_ratio"].tolist() == pytest.approx([-1.0, 0.0])
    assert ctl["length_log2_ratio"].tolist() == pytest.approx([0.0, -1.0])


def test_matched_percentiles_midrank():
    matched = matched_changes(*_window_tables())
    result = matched_percentiles(matched, metrics=("width_log2_ratio", "grid_entropy_change"))
    row = result.iloc[0]
    assert row["n_controls"] == 2
    assert row["width_log2_ratio_percentile"] == 0.0             # narrower than both
    assert row["width_log2_ratio_control_median"] == pytest.approx(-0.5)
    assert row["grid_entropy_change_percentile"] == 0.5          # tied with both


def _matched_rows(counter_values, control_values) -> pd.DataFrame:
    rows = []
    for i, (c, refs) in enumerate(zip(counter_values, control_values)):
        rows.append({"game_id": 1, "sequence_id": i, "role": "counter", "m": c})
        rows += [{"game_id": 1, "sequence_id": i, "role": "control", "m": v} for v in refs]
    return pd.DataFrame(rows)


def test_permutation_test_matches_hand_computed_null():
    """3 counters, each below all 4 of its controls -> mean percentile 0.

    Null: a random control ranked among the other 3 has percentile
    0, 1/3, 2/3 or 1 with probability 1/4 each, so a mean as far from 0.5 as
    0 (i.e. all three 0 or all three 1) has probability 2 * (1/4)^3 = 1/32.
    """
    matched = _matched_rows([0.0] * 3, [[1.0, 2.0, 3.0, 4.0]] * 3)
    result = matched_percentile_test(matched, "m", n_permutations=200_000, seed=1)
    assert result["n_counters"] == 3
    assert result["mean_percentile"] == 0.0
    assert result["p_value"] == pytest.approx(1 / 32, abs=0.002)


def test_permutation_test_is_not_significant_for_typical_counters():
    rng = np.random.default_rng(0)
    refs = [rng.normal(size=30) for _ in range(8)]
    matched = _matched_rows([float(np.median(r)) for r in refs], refs)
    result = matched_percentile_test(matched, "m", n_permutations=5_000)
    assert result["mean_percentile"] == pytest.approx(0.5, abs=0.05)
    assert result["p_value"] > 0.5


def test_permutation_test_needs_two_controls_per_counter():
    with pytest.raises(ValueError, match="at least 2"):
        matched_percentile_test(_matched_rows([0.0], [[1.0]]), "m")


def test_ball_depth_matched_keeps_controls_within_caliper():
    matched = pd.DataFrame({
        "game_id": 1, "sequence_id": [1, 1, 1, 1, 1, 2, 2],
        "role": ["counter", "control", "control", "control", "control", "counter", "control"],
        "ball_x_m": [-40.0, -30.0, -29.9, -50.0, np.nan, -20.0, -30.0],
    })
    kept = ball_depth_matched(matched, caliper_m=10.0)
    # Counter 1 at -40: keeps -30 (10 m) and -50 (10 m), drops -29.9 (10.1 m) and NaN.
    # Counter 2 at -20: keeps -30 (10 m).
    assert kept["ball_x_m"].tolist() == [-40.0, -30.0, -50.0, -20.0, -30.0]


def test_drop_thin_counters():
    matched = _matched_rows([0.0, 0.0, 0.0], [[1.0, 2.0], [1.0], []])
    kept = drop_thin_counters(matched, min_controls=2)
    assert kept["sequence_id"].unique().tolist() == [0]
    assert len(kept) == 3
