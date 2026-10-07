"""Pitch control: which team would win the ball at each point of the pitch.

REQUIRES Metrica tracking data; same constraint as entropy.py (positions
and velocities of all 22 players at one instant).

Voronoi territory (voronoi.py) gives each point to the nearest player. It
ignores which way players are running, how fast they can get there and
how long the ball takes to arrive. Spearman's (2018) pitch-control model
adds those factors:

    W. Spearman, "Beyond Expected Goals", MIT Sloan Sports Analytics
    Conference 2018,
    https://www.sloansportsconference.com/research-papers/beyond-expected-goals

Each player keeps their current velocity for a reaction time, then runs
at top speed to the target, which gives an expected arrival time
(time_to_intercept). Their actual arrival time is uncertain: the
probability that they have arrived by time T is a logistic function of
T minus the expected arrival time (intercept_probability). Once a player
has arrived, they gain control at rate lambda. The probability that each
team controls a point is then found by integrating Spearman's equation 3,
starting when the ball arrives (distance / average ball speed). The result
is a probability between 0 and 1 rather than a hard boundary.

The parameters, the shortcuts and the forward-Euler integration follow
Laurie Shaw's reference implementation for the Metrica sample data
(Friends of Tracking, Metrica_PitchControl.py,
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking).
Shaw's code evaluates one target point at a time; here the same integration
runs over every grid cell at once with numpy. Each cell stops integrating
in the step it converges, exactly as Shaw's per-target loop does, so the
two give the same values (checked against Shaw's code on real frames
while this module was written; tests/test_pitch_control.py checks small
hand-computable cases).

Differences from Shaw's notebook:
  - The pitch is 105 x 68 m (this project's convention), not 106 x 68.
  - Velocities come from compute_velocities: the least-squares slope over
    the same 7-frame Savitzky-Golay window. Shaw's notebook smooths
    frame-to-frame differences with that window instead.
  - The ball's start position is its tracked position at the frame (a
    pitch-control surface "now"), not the start of a pass event.

Conventions: positions must be in the defending team's attacking frame
(cleaning.normalize_attack_direction for the defending team), so the
defending team's own goal is at x = -pitch_length / 2. The offside rule and
the "own third" summary rely on that. "Attacking" means the team in
possession.

Pure functions; no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

PLAYER_TEAMS = ("Home", "Away")


@dataclass(frozen=True)
class PitchControlParams:
    """Model parameters; defaults are Shaw's default_model_params().

    lambda_def equals lambda_att (Shaw sets Spearman's kappa_def = 1, so
    both teams control the ball equally well). lambda_gk = 3 * lambda_def
    applies to the defending goalkeeper, who can catch the ball.
    time_to_control_veto sets the shortcuts: a team that arrives this many
    "control times" ahead of the other is given the point outright, and
    players that far behind their own team's fastest player are left out.
    """

    max_player_speed: float = 5.0       # m/s
    reaction_time: float = 0.7          # s
    tti_sigma: float = 0.45             # s, spread of the arrival-time logistic
    lambda_att: float = 4.3             # 1/s, control rate
    lambda_def: float = 4.3
    lambda_gk: float = 4.3 * 3.0
    average_ball_speed: float = 15.0    # m/s
    int_dt: float = 0.04                # s
    max_int_time: float = 10.0          # s
    converge_tol: float = 0.01
    time_to_control_veto: float = 3.0
    offside_tol_m: float = 0.2

    def time_to_control(self, lam: float) -> float:
        """Time for a player to reach 99.9 % control (with veto 3), as in Shaw."""
        return self.time_to_control_veto * np.log(10) * (
            np.sqrt(3) * self.tti_sigma / np.pi + 1 / lam
        )


def compute_velocities(
    tracking: pd.DataFrame,
    frame_rate_hz: float = 25.0,
    window_frames: int = 7,
    max_speed_ms: float = 12.0,
) -> pd.DataFrame:
    """Add smoothed per-player velocities (vx, vy in m/s) to long tracking.

    Each player's x and y, within each period, are differentiated with a
    Savitzky-Golay filter (polynomial order 1, so the velocity at a frame
    is the least-squares slope of position over `window_frames` frames;
    at the ends of a run the slope of the end window is used). Each run of
    consecutive frames with a valid position is differentiated separately;
    runs shorter than the window get NaN velocity, and so do speeds above
    `max_speed_ms`, which are tracking errors (Shaw uses the same 12 m/s
    cut-off).

    Parameters
    ----------
    tracking : pd.DataFrame
        Long tracking with frame, period, player_id, x, y, with one row per
        player for each consecutive frame (as from metrica_loader).
    frame_rate_hz : float
    window_frames : int
        Odd window length.
    max_speed_ms : float

    Returns
    -------
    pd.DataFrame
        A copy of `tracking` with vx and vy columns, in the same row order.
    """
    ordered = tracking.sort_values(["player_id", "period", "frame"], kind="stable")
    vx = np.full(len(ordered), np.nan)
    vy = np.full(len(ordered), np.nan)
    keys = ordered[["player_id", "period"]].to_numpy()
    xs = ordered["x"].to_numpy(float)
    ys = ordered["y"].to_numpy(float)
    # Runs of consecutive rows with the same player and period and finite
    # positions; each is differentiated separately.
    finite = np.isfinite(xs) & np.isfinite(ys)
    new_run = np.r_[True, (keys[1:] != keys[:-1]).any(axis=1) | (finite[1:] != finite[:-1])]
    starts = np.flatnonzero(new_run)
    for lo, hi in zip(starts, np.r_[starts[1:], len(ordered)]):
        if not finite[lo] or hi - lo < window_frames:
            continue
        for src, dst in ((xs, vx), (ys, vy)):
            dst[lo:hi] = savgol_filter(src[lo:hi], window_frames, 1, deriv=1,
                                       delta=1.0 / frame_rate_hz)
    too_fast = np.hypot(vx, vy) > max_speed_ms
    vx[too_fast] = np.nan
    vy[too_fast] = np.nan
    out = tracking.copy()
    out["vx"] = pd.Series(vx, index=ordered.index)
    out["vy"] = pd.Series(vy, index=ordered.index)
    return out


def pitch_grid(
    n_cells_x: int = 50, pitch_length: float = 105.0, pitch_width: float = 68.0
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre coordinates of the evaluation grid (Shaw's construction).

    n_cells_y = int(n_cells_x * width / length), so cells are close to
    square (2.1 m x 2.1 m by default). Returns (xgrid, ygrid).
    """
    n_cells_y = int(n_cells_x * pitch_width / pitch_length)
    dx, dy = pitch_length / n_cells_x, pitch_width / n_cells_y
    return (np.arange(n_cells_x) * dx - pitch_length / 2 + dx / 2,
            np.arange(n_cells_y) * dy - pitch_width / 2 + dy / 2)


def time_to_intercept(
    xy: np.ndarray, velocity: np.ndarray, targets: np.ndarray, params: PitchControlParams
) -> np.ndarray:
    """Expected arrival time of each player at each target, in seconds.

    The player keeps their velocity for params.reaction_time, then runs in
    a straight line at params.max_player_speed (Spearman 2018 / Shaw's
    simple_time_to_intercept). NaN velocities are treated as 0.

    Parameters
    ----------
    xy, velocity : (n_players, 2) arrays
    targets : (n_targets, 2) array

    Returns
    -------
    (n_targets, n_players) array
    """
    reaction = xy + np.nan_to_num(velocity) * params.reaction_time
    distance = np.linalg.norm(targets[:, None, :] - reaction[None, :, :], axis=2)
    return params.reaction_time + distance / params.max_player_speed


def intercept_probability(t: np.ndarray, tti: np.ndarray, tti_sigma: float) -> np.ndarray:
    """Probability a player has arrived by time t: logistic in t - tti
    with standard deviation tti_sigma (Spearman 2018, eq. 4)."""
    return 1.0 / (1.0 + np.exp(-np.pi / np.sqrt(3.0) / tti_sigma * (t - tti)))


def onside_attackers(
    att_x: np.ndarray, def_x: np.ndarray, ball_x: float, tol_m: float = 0.2
) -> np.ndarray:
    """Boolean mask of attackers who are not offside.

    Positions in the defending team's frame (own goal at -x), so attackers
    run toward -x. As in Shaw's check_offsides, the offside line is
    whichever of the second-deepest defender (goalkeeper included), the
    ball and the halfway line is nearest the defending goal. An attacker
    more than `tol_m` beyond it is offside. A NaN ball x is ignored.
    """
    if len(def_x) < 2:
        return np.ones(len(att_x), dtype=bool)
    second_deepest = np.sort(def_x)[1]
    line = min(second_deepest, 0.0, ball_x if np.isfinite(ball_x) else np.inf)
    return att_x >= line - tol_m


def pitch_control_at_targets(
    targets: np.ndarray,
    att_xy: np.ndarray,
    att_v: np.ndarray,
    def_xy: np.ndarray,
    def_v: np.ndarray,
    ball_xy: np.ndarray | None,
    params: PitchControlParams = PitchControlParams(),
    def_goalkeeper: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Spearman pitch control at each target point.

    Parameters
    ----------
    targets : (n_targets, 2) array
    att_xy, att_v : (n_att, 2) arrays
        Attacking (in-possession) players' positions and velocities, with
        no NaN positions (offside players already removed if wanted). If
        one team has no players, the other controls every target.
    def_xy, def_v : (n_def, 2) arrays
        Defending players, likewise.
    ball_xy : (2,) array or None
        Ball position. None or NaN means the ball is already at every
        target (ball travel time 0), as in Shaw's code.
    params : PitchControlParams
    def_goalkeeper : int, optional
        Row of def_xy that is the goalkeeper (uses params.lambda_gk).

    Returns
    -------
    ppcf_att, ppcf_def : (n_targets,) arrays
        Control probabilities. They sum to between 1 - converge_tol and 1
        at every target.
    """
    n = len(targets)
    if len(att_xy) == 0 or len(def_xy) == 0:
        return (np.full(n, float(len(def_xy) == 0)), np.full(n, float(len(att_xy) == 0)))
    if ball_xy is None or np.isnan(ball_xy).any():
        ball_time = np.zeros(n)
    else:
        ball_time = np.linalg.norm(targets - ball_xy, axis=1) / params.average_ball_speed
    tti_att = time_to_intercept(att_xy, att_v, targets, params)
    tti_def = time_to_intercept(def_xy, def_v, targets, params)
    tau_att = tti_att.min(axis=1)
    tau_def = tti_def.min(axis=1)
    ttc_att = params.time_to_control(params.lambda_att)
    ttc_def = params.time_to_control(params.lambda_def)

    ppcf_att, ppcf_def = np.zeros(n), np.zeros(n)
    def_wins = tau_att - np.maximum(ball_time, tau_def) >= ttc_def
    att_wins = ~def_wins & (tau_def - np.maximum(ball_time, tau_att) >= ttc_att)
    ppcf_def[def_wins] = 1.0
    ppcf_att[att_wins] = 1.0

    active = np.flatnonzero(~def_wins & ~att_wins)
    lam_def = np.full(len(def_xy), params.lambda_def)
    if def_goalkeeper is not None:
        lam_def[def_goalkeeper] = params.lambda_gk
    # Players too far behind their team's fastest player are left out.
    rate_att = (tti_att[active] - tau_att[active, None] < ttc_att) * params.lambda_att
    rate_def = (tti_def[active] - tau_def[active, None] < ttc_def) * lam_def[None, :]
    tti_att, tti_def = tti_att[active], tti_def[active]
    p_att, p_def = np.zeros_like(tti_att), np.zeros_like(tti_def)
    sum_att, sum_def = np.zeros(len(active)), np.zeros(len(active))
    t = ball_time[active] - params.int_dt
    n_steps = len(np.arange(-params.int_dt, params.max_int_time, params.int_dt))
    # Forward Euler on Spearman's eq. 3, one step for every unconverged cell
    # at a time. Converged cells are written out and dropped from the arrays.
    for _ in range(1, n_steps):
        t = t + params.int_dt
        free = (1.0 - sum_att - sum_def)[:, None]
        p_att += (free * intercept_probability(t[:, None], tti_att, params.tti_sigma)
                  * rate_att * params.int_dt)
        p_def += (free * intercept_probability(t[:, None], tti_def, params.tti_sigma)
                  * rate_def * params.int_dt)
        sum_att, sum_def = p_att.sum(axis=1), p_def.sum(axis=1)
        done = 1.0 - sum_att - sum_def <= params.converge_tol
        if done.any():
            ppcf_att[active[done]] = sum_att[done]
            ppcf_def[active[done]] = sum_def[done]
            keep = ~done
            active, t, sum_att, sum_def = active[keep], t[keep], sum_att[keep], sum_def[keep]
            tti_att, tti_def = tti_att[keep], tti_def[keep]
            rate_att, rate_def = rate_att[keep], rate_def[keep]
            p_att, p_def = p_att[keep], p_def[keep]
            if active.size == 0:
                break
    ppcf_att[active] = sum_att  # cells that hit max_int_time unconverged
    ppcf_def[active] = sum_def
    return ppcf_att, ppcf_def


def compute_pitch_control_surface(
    frame_positions: pd.DataFrame,
    attacking_team: str,
    defending_goalkeeper: str | None = None,
    params: PitchControlParams = PitchControlParams(),
    n_cells_x: int = 50,
    offsides: bool = True,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The defending team's pitch-control surface for one frame.

    Parameters
    ----------
    frame_positions : pd.DataFrame
        One frame: team, player_id, x, y, vx, vy for both teams' players
        and optionally a ball row (team not in PLAYER_TEAMS), in the
        defending team's attacking frame (see module docstring). Players
        with NaN positions are ignored.
    attacking_team : str
        "Home" or "Away": the team in possession.
    defending_goalkeeper : str, optional
        player_id of the defending goalkeeper (gets params.lambda_gk).
    params : PitchControlParams
    n_cells_x : int
        Grid resolution (see pitch_grid).
    offsides : bool
        Leave out offside attackers (onside_attackers), as Shaw does.

    Returns
    -------
    xgrid, ygrid : 1-D arrays
        Cell centres.
    ppcf_def : (len(ygrid), len(xgrid)) array
        Probability that the defending team controls each cell.
    """
    players = frame_positions[frame_positions["team"].isin(PLAYER_TEAMS)].dropna(
        subset=["x", "y"]
    )
    att = players[players["team"] == attacking_team]
    dfd = players[players["team"] != attacking_team]
    if att.empty or dfd.empty:
        raise ValueError("Need at least one player on each team.")
    ball = frame_positions[~frame_positions["team"].isin(PLAYER_TEAMS)]
    ball_xy = ball[["x", "y"]].to_numpy(float)[0] if len(ball) else None
    if offsides:
        ball_x = ball_xy[0] if ball_xy is not None else np.nan
        att = att[onside_attackers(att["x"].to_numpy(float), dfd["x"].to_numpy(float),
                                   ball_x, params.offside_tol_m)]
    gk = None
    if defending_goalkeeper is not None:
        hits = np.flatnonzero(dfd["player_id"].to_numpy() == defending_goalkeeper)
        gk = int(hits[0]) if len(hits) else None

    xgrid, ygrid = pitch_grid(n_cells_x, pitch_length, pitch_width)
    gx, gy = np.meshgrid(xgrid, ygrid)
    targets = np.column_stack([gx.ravel(), gy.ravel()])
    _, ppcf_def = pitch_control_at_targets(
        targets,
        att[["x", "y"]].to_numpy(float), att[["vx", "vy"]].to_numpy(float),
        dfd[["x", "y"]].to_numpy(float), dfd[["vx", "vy"]].to_numpy(float),
        ball_xy, params, gk,
    )
    return xgrid, ygrid, ppcf_def.reshape(gx.shape)


def summarize_surface(
    xgrid: np.ndarray, ppcf_def: np.ndarray, own_third_x_m: float = -17.5
) -> dict[str, float]:
    """Defending team's mean control over the whole pitch and its own third.

    Cells are equal-area, so a mean over cells is an area-weighted share.
    "Own third" is the cells whose centre is at x <= own_third_x_m (in the
    defending frame).

    Returns
    -------
    dict[str, float]
        {"pitch_control_fraction": ..., "own_third_control": ...}
    """
    return {
        "pitch_control_fraction": float(ppcf_def.mean()),
        "own_third_control": float(ppcf_def[:, xgrid <= own_third_x_m].mean()),
    }


def compute_pitch_control_timeseries(
    tracking: pd.DataFrame,
    attacking_team: str,
    start_frame: int,
    end_frame: int,
    defending_goalkeeper: str | None = None,
    params: PitchControlParams = PitchControlParams(),
    n_cells_x: int = 50,
    offsides: bool = True,
) -> pd.DataFrame:
    """summarize_surface for every frame in [start_frame, end_frame].

    `tracking` as for compute_pitch_control_surface (with vx, vy), any
    number of frames. Returns one row per frame: frame, time_s,
    pitch_control_fraction, own_third_control.
    """
    window = tracking[tracking["frame"].between(start_frame, end_frame)]
    rows = []
    for (frame, time_s), positions in window.groupby(["frame", "time_s"], sort=True):
        xgrid, _, surface = compute_pitch_control_surface(
            positions, attacking_team, defending_goalkeeper, params, n_cells_x, offsides
        )
        rows.append({"frame": frame, "time_s": time_s, **summarize_surface(xgrid, surface)})
    return pd.DataFrame(rows)
