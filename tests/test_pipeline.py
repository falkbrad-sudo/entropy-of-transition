"""Tests for src/pipeline.py's per-sequence analysis, on synthetic tracking."""
import pandas as pd
import pytest

from src.pipeline import analyze_sequence


def _tracking() -> pd.DataFrame:
    """Frames 1-4. Away (defending) attacks -x in period 1, so its goal is at
    +x: goalkeeper 'A_gk' at x=50, outfielders on a square that widens from
    frame 1 to frame 4. One Home attacker at x=-10. Kickoff-style first frame:
    Away's mean x is positive (own half), Home's negative.
    """
    rows = []
    for frame in range(1, 5):
        half = 5.0 * frame
        common = {"frame": frame, "period": 1, "time_s": frame * 0.04}
        rows.append({**common, "team": "Away", "player_id": "A_gk", "x": 50.0, "y": 0.0})
        for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
            rows.append({**common, "team": "Away", "player_id": f"A{i}",
                         "x": 20.0 + sx * half, "y": sy * half})
        rows.append({**common, "team": "Home", "player_id": "H0", "x": -10.0, "y": 0.0})
        rows.append({**common, "team": "Ball", "player_id": "ball", "x": 10.0, "y": 0.0})
    # Standing still (pitch control needs velocities).
    return pd.DataFrame(rows).assign(vx=0.0, vy=0.0)


def test_analyze_sequence_metrics_window_and_goalkeeper_exclusion():
    tracking = _tracking()
    directions = {"Home": {1: 1}, "Away": {1: -1}}
    sequence = pd.Series({"defending_team": "Away", "start_frame": 3, "end_frame": 4,
                          "start_time_s": 0.12})
    result = analyze_sequence(tracking, sequence, directions, pre_frames=1)

    assert list(result["frame"]) == [2, 3, 4]  # 1 frame of pre-recovery context
    assert result["t_rel_s"].tolist() == pytest.approx([-0.04, 0.0, 0.04])
    assert (result["n_players"] == 4).all()  # goalkeeper excluded from shape metrics
    assert result.attrs["goalkeeper"] == "A_gk"
    # Square of half-side 5*frame: rms distance = half * sqrt(2).
    assert result["rms_distance"].tolist() == pytest.approx(
        [10 * 2**0.5, 15 * 2**0.5, 20 * 2**0.5]
    )
    # Territory includes the goalkeeper and both teams, fractions in (0, 1).
    assert result["defending_fraction"].between(0, 1).all()
    assert result["defending_area_m2"].gt(0).all()
    # Ball at raw x=+10 is 10 m into Away's own half (Away attacks -x).
    assert (result["ball_x_m"] == -10.0).all()
    # Pitch control for the defending team, a share of the pitch.
    assert result["pitch_control_fraction"].between(0, 1).all()
    assert result["own_third_control"].between(0, 1).all()


def test_analyze_sequence_pitch_control_only_at_requested_frames():
    sequence = pd.Series({"defending_team": "Away", "start_frame": 1, "end_frame": 4,
                          "start_time_s": 0.04})
    directions = {"Home": {1: 1}, "Away": {1: -1}}
    result = analyze_sequence(_tracking(), sequence, directions, 0, pitch_control_frames=[2, 4])
    assert result["pitch_control_fraction"].notna().tolist() == [False, True, False, True]
    none = analyze_sequence(_tracking().drop(columns=["vx", "vy"]), sequence, directions, 0,
                            pitch_control_frames=[])
    assert none["own_third_control"].isna().all()
    with pytest.raises(ValueError, match="vx"):
        analyze_sequence(_tracking().drop(columns=["vx", "vy"]), sequence, directions, 0)


def test_counter_pitch_control_frames_and_window_frames():
    from src.pipeline import counter_pitch_control_frames, window_frames

    frame_times = pd.Series([f * 0.04 for f in range(1, 101)], index=range(1, 101))
    seq = pd.Series({"start_frame": 50, "start_time_s": 2.0, "end_frame": 63})
    # Every 5th frame from the recovery, 10 frames of context, plus the shot.
    assert counter_pitch_control_frames(frame_times, seq, pre_frames=12, step=5) == [
        40, 45, 50, 55, 60, 63]
    windows = pd.DataFrame({"start_frame": [5, 8, 20], "end_frame": [9, 10, 21]})
    assert window_frames(windows).tolist() == [5, 6, 7, 8, 9, 10, 20, 21]
    assert window_frames(windows, pre_frames=1).tolist() == [4, 5, 6, 7, 8, 9, 10, 19, 20, 21]


def test_sequence_positions_window_and_nan_drop():
    from src.pipeline import sequence_positions

    tracking = _tracking()
    tracking.loc[(tracking.frame == 3) & (tracking.player_id == "H0"), ["x", "y"]] = float("nan")
    sequence = pd.Series({"start_frame": 3, "end_frame": 4})
    result = sequence_positions(tracking, sequence, pre_frames=1)
    assert sorted(result["frame"].unique()) == [2, 3, 4]
    assert result[["x", "y"]].notna().all().all()
    assert len(result) == 3 * 7 - 1  # 7 objects per frame, one NaN row dropped


def test_baseline_control_sets_keep_counters_and_filter_controls_by_ball():
    from src.pipeline import baseline_control_sets

    matched = pd.DataFrame({
        "game_id": 1, "sequence_id": 1,
        "role": ["counter", "control", "control", "control", "control"],
        "ball_x_m": [-40.0, -30.0, -17.5, -45.0, float("nan")],  # NaN: ball not tracked
    })
    sets = baseline_control_sets(matched, ball_x_max_m=-17.5, depth_caliper_m=5.0)
    assert len(sets["all"]) == 5
    assert sets["ball_in_defending_third"]["ball_x_m"].tolist() == [-40.0, -30.0, -17.5, -45.0]
    # Only -45 is within 5 m of the counter's -40: 1 control, so the counter is dropped.
    assert sets["ball_depth_matched"].empty
    sets = baseline_control_sets(matched, ball_x_max_m=-17.5, depth_caliper_m=10.0)
    assert sets["ball_depth_matched"]["ball_x_m"].tolist() == [-40.0, -30.0, -45.0]


def test_horizon_frames_pick_what_shape_change_would_read():
    from src.pipeline import horizon_frames

    frame_times = pd.Series([f * 0.04 for f in range(100, 201)], index=range(100, 201))
    # Recovery event at 4.01 s falls between frames 100 (4.00 s) and 101 (4.04 s).
    window = pd.Series({"start_frame": 100, "start_time_s": 4.01, "end_frame": 150})
    frames = horizon_frames(frame_times, window, horizons_s=[1.0, 0.5, 99.0])
    # First frame with t_rel >= 0 is 101; t_rel <= 0.5 -> frame 112 (4.48 s; 113 is 0.51 s);
    # t_rel <= 1.0 -> frame 125 (5.00 s); 99 s and the window end -> 150.
    assert frames == [101, 112, 125, 150]


def test_analyze_sequence_frames_subset_matches_full_series():
    tracking = _tracking()
    directions = {"Home": {1: 1}, "Away": {1: -1}}
    sequence = pd.Series({"defending_team": "Away", "start_frame": 1, "end_frame": 4,
                          "start_time_s": 0.04})
    full = analyze_sequence(tracking, sequence, directions, pre_frames=0)
    sparse = analyze_sequence(tracking, sequence, directions, pre_frames=0, frames=[1, 4])
    assert sparse["frame"].tolist() == [1, 4]
    pd.testing.assert_frame_equal(
        sparse.reset_index(drop=True),
        full[full["frame"].isin([1, 4])].reset_index(drop=True),
    )


def test_window_end_changes_one_row_per_window_at_its_own_end():
    from src.pipeline import window_end_changes

    def series(perp_end):
        return pd.DataFrame({"t_rel_s": [0.0, 1.0, 2.0],
                             "perpendicular_variance": [8.0, 6.0, perp_end],
                             "along_axis_variance": 4.0, "grid_entropy": 1.0,
                             "defending_fraction": 0.5, "ball_x_m": [0.0, -10.0, -40.0]})

    sequences = pd.DataFrame({"game_id": [1], "sequence_id": [1], "duration_s": [2.0]})
    controls = pd.DataFrame({"game_id": [1, 2], "control_id": [1, 1], "duration_s": [2.0, 2.0]})
    result = window_end_changes(
        sequences, series(2.0).assign(game_id=1, sequence_id=1),
        controls, pd.concat([series(4.0).assign(game_id=1, control_id=1),
                             series(8.0).assign(game_id=2, control_id=1)]),
    )
    assert result["role"].tolist() == ["counter", "control", "control"]
    assert result["width_log2_ratio"].tolist() == pytest.approx([-2.0, -1.0, 0.0])
    assert (result["ball_x_m"] == -40.0).all()
