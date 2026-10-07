"""Baseline comparison: counter-attacks vs. own-half recoveries with no shot.

The counter-attack findings ("the defense gets narrow and long") mean
little on their own: a defense might do that whenever it loses the ball
upfield. This module compares each counter-attack's shape change with the
same change in control windows (src/data/sequences.find_control_recoveries)
measured at the same elapsed time since the recovery.

Matching on elapsed time matters because shape changes accumulate: a
counter that lasts 14 s should be compared with controls 14 s after their
recovery, not with controls that ended after 4 s. So for a counter of
duration d, the eligible controls are those whose window lasts at least d,
each evaluated at t = d.

Ball position is the obvious confounder: a counter-attack ends with a shot,
so its ball is near goal, while most controls never get there.
ball_depth_matched keeps only controls whose ball, at the matched time, was
within a caliper of where the counter's ball was at the shot.

Each counter gets a percentile within its own eligible controls (midrank:
ties count half). If counters behaved like controls, those percentiles
would be uniform on (0, 1) and average 0.5; matched_percentile_test turns
the mean percentile into a permutation p-value.

Pure functions on the time-series tables written by src/pipeline.py; no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANGE_METRICS = (
    "width_log2_ratio",
    "length_log2_ratio",
    "grid_entropy_change",
    "defending_fraction_change",
    "pitch_control_change",
    "own_third_control_change",
)
# Optional per-frame columns -> their change metric (NaN if the column is absent).
_OPTIONAL_CHANGES = {
    "pitch_control_fraction": "pitch_control_change",
    "own_third_control": "own_third_control_change",
}
COUNTER_KEY = ("game_id", "sequence_id")
CONTROL_KEY = ("game_id", "control_id")
_TIME_TOL_S = 1e-9


def shape_change(series: pd.DataFrame, horizon_s: float | None = None) -> pd.Series:
    """Change in the defending shape from the recovery to a later time.

    Parameters
    ----------
    series : pd.DataFrame
        One window's per-frame metrics, as from pipeline.analyze_sequence:
        t_rel_s, along_axis_variance, perpendicular_variance, grid_entropy,
        defending_fraction, and optionally ball_x_m, pitch_control_fraction
        and own_third_control.
    horizon_s : float, optional
        Seconds after the recovery to measure at (the last frame at or
        before it). None means the window's last frame.

    Returns
    -------
    pd.Series
        horizon_s (the time actually used), width_log2_ratio and
        length_log2_ratio (log2 of the end / start variance across and
        along the attack axis: -1 = halved, +1 = doubled),
        grid_entropy_change, defending_fraction_change,
        pitch_control_change and own_third_control_change (end - start; the
        last two NaN if absent), and ball_x_m at the end (NaN if absent).
    """
    after = series[series["t_rel_s"] >= -_TIME_TOL_S]
    if after.empty:
        raise ValueError("Series has no frame at or after the recovery (t_rel_s >= 0).")
    start = after.iloc[0]
    if horizon_s is not None:
        after = after[after["t_rel_s"] <= horizon_s + _TIME_TOL_S]
    end = after.iloc[-1]
    optional = {
        change: float(end[col] - start[col]) if col in end else np.nan
        for col, change in _OPTIONAL_CHANGES.items()
    }
    return pd.Series(
        {
            "horizon_s": float(end["t_rel_s"]),
            "width_log2_ratio": float(
                np.log2(end["perpendicular_variance"] / start["perpendicular_variance"])
            ),
            "length_log2_ratio": float(
                np.log2(end["along_axis_variance"] / start["along_axis_variance"])
            ),
            "grid_entropy_change": float(end["grid_entropy"] - start["grid_entropy"]),
            "defending_fraction_change": float(
                end["defending_fraction"] - start["defending_fraction"]
            ),
            **optional,
            "ball_x_m": float(end["ball_x_m"]) if "ball_x_m" in end else np.nan,
        }
    )


def matched_changes(
    counters: pd.DataFrame,
    counter_series: pd.DataFrame,
    controls: pd.DataFrame,
    control_series: pd.DataFrame,
) -> pd.DataFrame:
    """Shape changes for each counter and its duration-matched controls.

    Parameters
    ----------
    counters, controls : pd.DataFrame
        Window tables with duration_s, keyed by COUNTER_KEY / CONTROL_KEY.
    counter_series, control_series : pd.DataFrame
        Per-frame metrics for those windows, same keys (see shape_change).

    Returns
    -------
    pd.DataFrame
        One row per counter (role "counter", measured at its end) plus one
        row per (counter, eligible control) pair (role "control", measured
        at the counter's duration). Columns: game_id, sequence_id, role,
        control_game_id, control_id (NaN on counter rows), then the
        shape_change fields.
    """
    counter_groups = dict(list(counter_series.groupby(list(COUNTER_KEY))))
    control_groups = dict(list(control_series.groupby(list(CONTROL_KEY))))
    rows = []
    for counter in counters.itertuples(index=False):
        key = (counter.game_id, counter.sequence_id)
        base = {"game_id": counter.game_id, "sequence_id": counter.sequence_id}
        rows.append({**base, "role": "counter", "control_game_id": np.nan,
                     "control_id": np.nan, **shape_change(counter_groups[key])})
        eligible = controls[controls["duration_s"] >= counter.duration_s - _TIME_TOL_S]
        for control in eligible.itertuples(index=False):
            series = control_groups[(control.game_id, control.control_id)]
            rows.append({**base, "role": "control", "control_game_id": control.game_id,
                         "control_id": control.control_id,
                         **shape_change(series, horizon_s=counter.duration_s)})
    return pd.DataFrame(rows)


def ball_depth_matched(matched: pd.DataFrame, caliper_m: float) -> pd.DataFrame:
    """Keep only controls whose ball depth matches their counter's.

    Parameters
    ----------
    matched : pd.DataFrame
        Output of matched_changes.
    caliper_m : float
        A control row is kept if its ball_x_m (at the matched time) is
        within this many meters of the counter's ball_x_m at the shot.
        Controls with an untracked ball (NaN) are dropped.

    Returns
    -------
    pd.DataFrame
        Counter rows unchanged, plus the matching control rows.
    """
    counter_ball = (
        matched[matched["role"] == "counter"]
        .set_index(list(COUNTER_KEY))["ball_x_m"]
        .rename("counter_ball_x_m")
    )
    joined = matched.join(counter_ball, on=list(COUNTER_KEY))
    keep = (joined["role"] == "counter") | (
        (joined["ball_x_m"] - joined["counter_ball_x_m"]).abs() <= caliper_m
    )
    return matched[keep.to_numpy()]


def drop_thin_counters(matched: pd.DataFrame, min_controls: int = 2) -> pd.DataFrame:
    """Drop counters (and their controls) with fewer than `min_controls` controls.

    matched_percentile_test needs at least 2 controls per counter; a strict
    control set (e.g. a narrow ball_depth_matched caliper) can leave some
    counters with fewer. Callers should report how many counters remain.
    """
    n_controls = (
        matched[matched["role"] == "control"].groupby(list(COUNTER_KEY)).size()
    )
    enough = n_controls[n_controls >= min_controls].index
    keys = pd.MultiIndex.from_frame(matched[list(COUNTER_KEY)])
    return matched[keys.isin(enough)]


def _midrank_percentile(value: float, reference: np.ndarray) -> float:
    """Fraction of `reference` below `value`, counting ties as half."""
    return float(((reference < value).sum() + 0.5 * (reference == value).sum()) / len(reference))


def matched_percentiles(
    matched: pd.DataFrame, metrics: tuple[str, ...] = CHANGE_METRICS
) -> pd.DataFrame:
    """Each counter's percentile among its own matched controls.

    Parameters
    ----------
    matched : pd.DataFrame
        Output of matched_changes (optionally with control rows filtered,
        e.g. by ball_x_m).
    metrics : tuple of str
        Columns to rank.

    Returns
    -------
    pd.DataFrame
        One row per counter: game_id, sequence_id, n_controls, and for each
        metric `<m>` (counter value), `<m>_control_median` and
        `<m>_percentile` (0 = below every control, 1 = above every one).
    """
    rows = []
    for (game_id, sequence_id), group in matched.groupby(list(COUNTER_KEY), sort=True):
        counter = group[group["role"] == "counter"].iloc[0]
        ctl = group[group["role"] == "control"]
        row = {"game_id": game_id, "sequence_id": sequence_id, "n_controls": len(ctl)}
        for m in metrics:
            reference = ctl[m].to_numpy()
            row[m] = counter[m]
            row[f"{m}_control_median"] = float(np.median(reference)) if len(ctl) else np.nan
            row[f"{m}_percentile"] = (
                _midrank_percentile(counter[m], reference) if len(ctl) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def matched_percentile_test(
    matched: pd.DataFrame, metric: str, n_permutations: int = 10_000, seed: int = 0
) -> dict:
    """Permutation test: do counters' matched percentiles average 0.5?

    Null hypothesis: each counter is exchangeable with its matched
    controls. Under it, the observed mean percentile is distributed like
    the mean obtained by replacing each counter with one of its own
    controls drawn at random (ranked against the remaining controls).
    Simulating that keeps the discreteness and ties of the real data, which
    a uniform-distribution approximation would not.

    Parameters
    ----------
    matched : pd.DataFrame
        Output of matched_changes. Every counter needs >= 2 controls.
    metric : str
        One of the shape_change columns.
    n_permutations : int
        Null draws.
    seed : int
        RNG seed, so p-values are reproducible.

    Returns
    -------
    dict
        n_counters, mean_percentile, p_value (two-sided, distance from 0.5,
        with the +1 correction so it is never exactly 0).
    """
    observed, references = [], []
    for _, group in matched.groupby(list(COUNTER_KEY), sort=True):
        reference = group.loc[group["role"] == "control", metric].to_numpy()
        if len(reference) < 2:
            raise ValueError("Every counter needs at least 2 matched controls.")
        counter_value = group.loc[group["role"] == "counter", metric].iloc[0]
        if np.isnan(counter_value) or np.isnan(reference).any():
            raise ValueError(f"NaN values in {metric}; percentiles would be meaningless.")
        observed.append(_midrank_percentile(counter_value, reference))
        references.append(reference)
    mean_observed = float(np.mean(observed))

    rng = np.random.default_rng(seed)
    null = np.zeros(n_permutations)
    for reference in references:
        picks = rng.integers(len(reference), size=n_permutations)
        values = reference[picks]
        below = (reference[None, :] < values[:, None]).sum(axis=1)
        ties = (reference[None, :] == values[:, None]).sum(axis=1) - 1  # minus the pick itself
        null += (below + 0.5 * ties) / (len(reference) - 1)
    null /= len(references)

    extreme = np.abs(null - 0.5) >= abs(mean_observed - 0.5) - _TIME_TOL_S
    return {
        "n_counters": len(observed),
        "mean_percentile": mean_observed,
        "p_value": float((extreme.sum() + 1) / (n_permutations + 1)),
    }
