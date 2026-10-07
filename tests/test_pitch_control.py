"""Tests for src/features/pitch_control.py, on hand-checkable configurations."""
import numpy as np
import pandas as pd
import pytest

from src.features.pitch_control import (
    PitchControlParams,
    compute_pitch_control_surface,
    compute_velocities,
    intercept_probability,
    onside_attackers,
    pitch_control_at_targets,
    pitch_grid,
    summarize_surface,
    time_to_intercept,
)

PARAMS = PitchControlParams()
STILL = np.zeros((1, 2))


def test_time_to_intercept_reaction_then_straight_run():
    targets = np.array([[10.0, 0.0], [0.0, 0.0]])
    # Standing still 10 m away: 0.7 s reaction + 10 m / 5 m/s = 2.7 s.
    standing = time_to_intercept(np.zeros((1, 2)), STILL, targets, PARAMS)
    assert standing[:, 0] == pytest.approx([2.7, 0.7])
    # Running at 5 m/s toward the target: 3.5 m covered while reacting, 6.5 m left.
    running = time_to_intercept(np.zeros((1, 2)), np.array([[5.0, 0.0]]), targets, PARAMS)
    assert running[0, 0] == pytest.approx(0.7 + 6.5 / 5)
    # NaN velocity counts as standing still.
    nan_v = time_to_intercept(np.zeros((1, 2)), np.full((1, 2), np.nan), targets, PARAMS)
    assert nan_v[:, 0] == pytest.approx([2.7, 0.7])


def test_intercept_probability_is_a_logistic_centred_on_arrival_time():
    assert intercept_probability(np.array(2.0), np.array(2.0), 0.45) == pytest.approx(0.5)
    early = intercept_probability(np.array(1.5), np.array(2.0), 0.45)
    late = intercept_probability(np.array(2.5), np.array(2.0), 0.45)
    assert early + late == pytest.approx(1.0)
    assert early < 0.5 < late


def test_pitch_grid_matches_shaw_construction():
    xgrid, ygrid = pitch_grid()
    assert (len(xgrid), len(ygrid)) == (50, 32)  # int(50 * 68 / 105) = 32
    assert xgrid[0] == pytest.approx(-52.5 + 1.05)
    assert np.diff(xgrid) == pytest.approx(np.full(49, 2.1))
    assert ygrid[0] == pytest.approx(-34 + 68 / 64)


def test_far_ahead_team_takes_the_point_outright():
    target = np.array([[0.0, 0.0]])
    att, dfd = pitch_control_at_targets(
        target, np.array([[0.0, 0.0]]), STILL, np.array([[50.0, 0.0]]), STILL, None
    )
    assert (att[0], dfd[0]) == (1.0, 0.0)


def test_symmetric_one_on_one_splits_evenly_and_converges():
    targets = np.array([[0.0, y] for y in (-20.0, 0.0, 15.0)])
    att, dfd = pitch_control_at_targets(
        targets, np.array([[-10.0, 0.0]]), STILL, np.array([[10.0, 0.0]]), STILL,
        ball_xy=np.array([0.0, 30.0]),
    )
    assert att == pytest.approx(dfd, abs=1e-12)
    assert np.all(att + dfd >= 1 - PARAMS.converge_tol)
    assert np.all(att + dfd <= 1 + 1e-12)


def test_goalkeeper_wins_a_tie():
    target = np.array([[0.0, 0.0]])
    args = (target, np.array([[-10.0, 0.0]]), STILL, np.array([[10.0, 0.0]]), STILL, None)
    _, outfielder = pitch_control_at_targets(*args)
    _, keeper = pitch_control_at_targets(*args, def_goalkeeper=0)
    assert keeper[0] > outfielder[0]
    assert keeper[0] == pytest.approx(0.75, abs=0.01)  # 3x the rate: ~3/4 of the ball


def test_offside_line_is_second_deepest_defender_ball_or_halfway():
    defenders = np.array([-50.0, -20.0, -10.0])  # goalkeeper, two outfielders
    attackers = np.array([-25.0, -20.1, -19.0, 5.0])
    # Line at the second-deepest defender, -20 (ball at -5 is not as deep).
    assert onside_attackers(attackers, defenders, ball_x=-5.0).tolist() == [
        False, True, True, True]
    # Ball at -30 is deeper: everyone level with or behind it is onside.
    assert onside_attackers(attackers, defenders, ball_x=-30.0).all()
    # Defenders all in the other half: halfway line (0) is the line.
    assert onside_attackers(np.array([-1.0, 1.0]), np.array([10.0, 20.0]),
                            ball_x=np.nan).tolist() == [False, True]


def test_velocities_of_uniform_motion_with_a_gap():
    frames = np.arange(1, 31)
    x = 2.0 * frames / 25.0  # 2 m/s
    x[14] = np.nan           # gap at frame 15 splits the run into 14 + 15 frames
    tracking = pd.DataFrame({
        "frame": frames, "period": 1, "player_id": "p", "x": x, "y": 3.0,
    })
    # A second player tracked for only 5 frames (too short for the 7-frame window).
    short = pd.DataFrame({"frame": frames[:5], "period": 1, "player_id": "q",
                          "x": 0.0, "y": 0.0})
    # A third moving at 20 m/s (impossible): flagged as a tracking error.
    fast = pd.DataFrame({"frame": frames, "period": 1, "player_id": "r",
                         "x": 20.0 * frames / 25.0, "y": 0.0})
    result = compute_velocities(pd.concat([tracking, short, fast], ignore_index=True))
    p = result[result["player_id"] == "p"]
    valid = p["x"].notna()
    assert p.loc[valid, "vx"].to_numpy() == pytest.approx(np.full(valid.sum(), 2.0))
    assert p.loc[valid, "vy"].to_numpy() == pytest.approx(np.zeros(valid.sum()), abs=1e-12)
    assert p.loc[~valid, "vx"].isna().all()
    assert result.loc[result["player_id"] == "q", "vx"].isna().all()
    assert result.loc[result["player_id"] == "r", "vx"].isna().all()
    assert list(result.index) == list(range(len(result)))  # row order kept


def _frame(att_x: float, def_x: float) -> pd.DataFrame:
    return pd.DataFrame({
        "team": ["Home", "Away", "Ball"], "player_id": ["H", "A", "ball"],
        "x": [att_x, def_x, 0.0], "y": [0.0, 0.0, 0.0],
        "vx": [0.0, 0.0, np.nan], "vy": [0.0, 0.0, np.nan],
    })


def test_surface_of_mirror_image_players_is_antisymmetric():
    xgrid, ygrid, surface = compute_pitch_control_surface(
        _frame(att_x=10.0, def_x=-10.0), attacking_team="Home", offsides=False
    )
    assert surface.shape == (len(ygrid), len(xgrid))
    # Mirror symmetry about x = 0: defending control at -x equals attacking
    # control at x, so surface + mirrored surface is each cell's total
    # probability, which is within converge_tol of 1.
    total = surface + surface[:, ::-1]
    assert np.all(total >= 1 - PARAMS.converge_tol) and np.all(total <= 1 + 1e-12)
    assert surface[:, xgrid < -20].mean() > 0.9   # defender's side
    assert surface[:, xgrid > 20].mean() < 0.1


def test_offside_attacker_is_ignored():
    # Defenders: goalkeeper at -50 and one at -10; ball at 0. The offside
    # line is -10, so a lone attacker at -40 is offside and the defending
    # team controls everywhere.
    frame = pd.concat([_frame(att_x=-40.0, def_x=-10.0), pd.DataFrame({
        "team": ["Away"], "player_id": ["GK"], "x": [-50.0], "y": [0.0],
        "vx": [0.0], "vy": [0.0]})], ignore_index=True)
    _, _, surface = compute_pitch_control_surface(frame, "Home", defending_goalkeeper="GK")
    assert surface.min() == 1.0
    _, _, with_offside = compute_pitch_control_surface(frame, "Home", offsides=False)
    assert with_offside.min() < 0.5


def test_summarize_surface_means_over_cells():
    xgrid = np.array([-30.0, -10.0, 10.0, 30.0])
    surface = np.tile([1.0, 0.5, 0.0, 0.0], (2, 1))
    summary = summarize_surface(xgrid, surface)
    assert summary["pitch_control_fraction"] == pytest.approx(0.375)
    assert summary["own_third_control"] == 1.0  # only x = -30 is at or behind -17.5
