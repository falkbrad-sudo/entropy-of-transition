"""Positional entropy: quantifying how organized (or scattered) a team's
defensive shape is at a given instant.

REQUIRES Metrica tracking data (true simultaneous multi-player positions).
Do not feed this module StatsBomb event data; see METHODOLOGY.md principle 3
and src/features/formation.py for the correct StatsBomb-appropriate proxy.

Two candidate approaches, both legitimate biophysics-inspired techniques.
Pick one (or implement both and compare; that comparison could itself be
an interesting section of the writeup):

1. Discrete positional entropy (Shannon-style):
   Bin the defensive half of the pitch into a grid. At each frame, compute
   the fraction of players in each cell, then Shannon entropy of that
   occupancy distribution: H = -sum(p_i * log(p_i)) over non-empty cells.
   A tightly-organized line concentrates players in few cells (low H); a
   scattered/broken shape spreads them out (high H). Simple, fast,
   interpretable; a reasonable default to implement first.

2. Continuous spatial dispersion (physically closer to your biophysics
   background; analogous to a radius-of-gyration or pairwise-distance
   variance calculation for a set of interacting particles):
   Compute the centroid of the defending team's outfield players, then the
   variance (or RMS distance) of each player's position from that centroid.
   Rising variance during a transition = the shape stretching/breaking.
   Can be extended to compute this separately along the axis of attack vs.
   perpendicular to it, to distinguish "getting stretched longways" from
   "getting pulled apart sideways"; likely more tactically meaningful than
   a single scalar.

Either way: compute this frame-by-frame over identified counter-attack
sequences (compute_entropy_timeseries), so the output is a time series
showing the metric rising (or not) as the transition develops. No smoothing
is applied: every value is a raw per-frame measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_grid_entropy(
    positions: pd.DataFrame,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    n_bins_x: int = 12,
    n_bins_y: int = 8,
) -> float:
    """Shannon entropy of a team's occupancy across a pitch grid, for one frame.

    Parameters
    ----------
    positions : pd.DataFrame
        One row per player, with 'x' and 'y' columns in standard meter
        coordinates (see src/data/cleaning.py). Exactly one frame's worth of
        positions for one team; do not pass multiple frames or both teams
        at once.
    pitch_length, pitch_width : float
        Standard pitch dimensions in meters.
    n_bins_x, n_bins_y : int
        Grid resolution. Coarser grids (fewer bins) are more robust to
        tracking noise; finer grids capture more structure but are noisier
        frame-to-frame; worth trying a couple of resolutions and checking
        sensitivity before picking a final value.

    Returns
    -------
    float
        Shannon entropy in nats (natural log). Higher = more scattered.

    Notes
    -----
    With N players the entropy is bounded by ln(N) (every player in a
    different cell), e.g. ln(10) ≈ 2.30 nats for 10 outfield players. On the
    default 12 x 8 grid (8.75 m x 8.5 m cells), a normally spread team
    already sits near that ceiling, so this metric mostly counts how many
    distinct cells are occupied. It's sensitive to clustering (several
    players sharing a cell), not to how far apart the occupied cells are;
    use compute_dispersion() for the latter.

    Points slightly outside the pitch (tracking noise near the touchlines)
    are clipped onto the boundary cells rather than dropped. Rows with NaN
    coordinates are ignored. An empty frame returns NaN.
    """
    xy = positions[["x", "y"]].dropna().to_numpy(dtype=float)
    if len(xy) == 0:
        return float("nan")
    half_l, half_w = pitch_length / 2, pitch_width / 2
    x = np.clip(xy[:, 0], -half_l, half_l)
    y = np.clip(xy[:, 1], -half_w, half_w)
    counts, _, _ = np.histogram2d(
        x, y, bins=[n_bins_x, n_bins_y], range=[[-half_l, half_l], [-half_w, half_w]]
    )
    return _entropy_from_counts(counts[counts > 0])


def _entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy (nats) of occupancy counts, as ln N - sum(c ln c) / N.

    Equal to -sum(p ln p) with p = c / N. Summing over the *sorted* counts
    makes the result depend only on the multiset of counts, so two frames
    with the same occupancy pattern get bit-identical entropies; an entropy
    change is then exactly 0, which keeps ties as ties in rank-based
    comparisons (features/baseline.py).
    """
    c = np.sort(np.asarray(counts, dtype=float))
    n = c.sum()
    return float(np.log(n) - (c * np.log(c)).sum() / n)


def compute_dispersion(
    positions: pd.DataFrame,
    attack_axis: str = "x",
) -> dict[str, float]:
    """Centroid-relative dispersion of a team's shape, for one frame.

    Treats the players as equal-mass particles: rms_distance is the radius
    of gyration of the configuration, and the two axis variances are the
    diagonal elements of its gyration tensor in the pitch frame, so
    rms_distance**2 == along_axis_variance + perpendicular_variance exactly.

    Parameters
    ----------
    positions : pd.DataFrame
        One row per player, 'x'/'y' columns, one frame, one team. Rows with
        NaN coordinates are ignored.
    attack_axis : str
        Which standard-coordinate axis represents the direction of attack
        ('x' in this project's convention; see cleaning.py).

    Returns
    -------
    dict[str, float]
        {"rms_distance": ..., "along_axis_variance": ...,
         "perpendicular_variance": ...}; three complementary measures of
        how stretched/compact the shape is, and along which direction.
        rms_distance is in m; the variances are in m^2 (population
        variance, i.e. divided by N, not N-1; these are descriptive
        statistics of one configuration, not estimates from a sample).
        All values are NaN for an empty frame.
    """
    if attack_axis not in ("x", "y"):
        raise ValueError(f"attack_axis must be 'x' or 'y', got '{attack_axis}'.")
    perpendicular_axis = "y" if attack_axis == "x" else "x"
    valid = positions[["x", "y"]].dropna()
    if len(valid) == 0:
        nan = float("nan")
        return {"rms_distance": nan, "along_axis_variance": nan, "perpendicular_variance": nan}
    along_var = float(np.mean((valid[attack_axis] - valid[attack_axis].mean()) ** 2))
    perp_var = float(
        np.mean((valid[perpendicular_axis] - valid[perpendicular_axis].mean()) ** 2)
    )
    return {
        "rms_distance": float(np.sqrt(along_var + perp_var)),
        "along_axis_variance": along_var,
        "perpendicular_variance": perp_var,
    }


def compute_entropy_timeseries(
    tracking: pd.DataFrame,
    team: str,
    start_frame: int,
    end_frame: int,
    method: str = "dispersion",
    exclude_players: list[str] | None = None,
) -> pd.DataFrame:
    """Apply an entropy/dispersion metric across every frame in a sequence.

    This is the function that actually produces the "entropy spikes during
    the counter-attack" result; call this on a frame range you've
    identified (via Metrica event data) as a real transition moment.

    Parameters
    ----------
    tracking : pd.DataFrame
        Long-format tracking data (from metrica_loader.load_tracking or
        load_match_tracking), already converted to standard meter
        coordinates, with frame, time_s, team, player_id, x, y columns.
    team : str
        Which team's shape to analyze; typically the defending team during
        the sequence of interest.
    start_frame, end_frame : int
        Inclusive frame range defining the sequence (e.g., from the moment
        possession is lost to the moment the resulting shot/clearance
        happens).
    method : str
        "grid_entropy" or "dispersion"; which function above to apply.
    exclude_players : list[str], optional
        player_ids to leave out, typically the goalkeeper (see
        cleaning.infer_goalkeeper), whose position far behind the outfield
        shape would otherwise dominate the dispersion.

    Returns
    -------
    pd.DataFrame
        One row per frame, with frame, time_s, n_players, and the computed
        metric column(s); ready to hand directly to src/viz for plotting.
        Frames in which the team has no valid positions are omitted.
    """
    if method not in ("dispersion", "grid_entropy"):
        raise ValueError(f"method must be 'dispersion' or 'grid_entropy', got '{method}'.")
    mask = (
        (tracking["team"] == team)
        & tracking["frame"].between(start_frame, end_frame)
        & tracking["x"].notna()
        & tracking["y"].notna()
    )
    if exclude_players:
        mask &= ~tracking["player_id"].isin(exclude_players)
    sequence = tracking.loc[mask, ["frame", "time_s", "x", "y"]]
    if sequence.empty:
        metric_cols = (["rms_distance", "along_axis_variance", "perpendicular_variance"]
                       if method == "dispersion" else ["grid_entropy"])
        return pd.DataFrame(columns=["frame", "time_s", "n_players", *metric_cols])
    # Vectorized over frames; same values as calling compute_dispersion /
    # compute_grid_entropy on each frame (tests/test_entropy.py checks this).
    groups = sequence.groupby(["frame", "time_s"], sort=True)
    result = groups.size().rename("n_players").reset_index()
    if method == "dispersion":
        centred = sequence[["x", "y"]] - groups[["x", "y"]].transform("mean")
        variances = (centred**2).groupby([sequence["frame"], sequence["time_s"]],
                                         sort=True).mean()
        result["along_axis_variance"] = variances["x"].to_numpy()
        result["perpendicular_variance"] = variances["y"].to_numpy()
        result.insert(3, "rms_distance", np.sqrt(result["along_axis_variance"]
                                                 + result["perpendicular_variance"]))
    else:
        cells = _grid_cells(sequence["x"].to_numpy(float), sequence["y"].to_numpy(float))
        counts = pd.Series(1, index=pd.MultiIndex.from_arrays(
            [sequence["frame"].to_numpy(), cells])).groupby(level=[0, 1]).sum()
        frame_of = counts.index.get_level_values(0).to_numpy()
        bounds = np.flatnonzero(np.r_[True, frame_of[1:] != frame_of[:-1], True])
        values = counts.to_numpy()
        result["grid_entropy"] = [_entropy_from_counts(values[lo:hi])
                                  for lo, hi in zip(bounds[:-1], bounds[1:])]
    return result


def _grid_cells(
    x: np.ndarray,
    y: np.ndarray,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    n_bins_x: int = 12,
    n_bins_y: int = 8,
) -> np.ndarray:
    """Flat grid-cell index of each point, binned exactly as compute_grid_entropy
    (np.histogram2d: half-open bins, the last one closed, after clipping)."""
    half_l, half_w = pitch_length / 2, pitch_width / 2
    index = []
    for v, half, n in ((x, half_l, n_bins_x), (y, half_w, n_bins_y)):
        v = np.clip(v, -half, half)
        edges = np.linspace(-half, half, n + 1)
        i = np.searchsorted(edges, v, side="right") - 1
        index.append(np.where(v == edges[-1], n - 1, i))
    return index[0] * n_bins_y + index[1]
