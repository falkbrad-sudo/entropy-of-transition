"""Aggregate team-shape proxy for StatsBomb event data.

This module exists specifically because StatsBomb's free event data CANNOT
support the per-instant entropy/Voronoi calculations in entropy.py and
voronoi.py (see METHODOLOGY.md principle 3 for why). This is the honest
alternative: instead of a team's exact shape at one instant, compute each
player's AVERAGE position over a rolling window of their own touches, and
track how that average shape's compactness changes over the course of a
match or season.

This is coarser than true tracking-based analysis and the module/docstrings
should say so wherever this feature is presented (plots, README, app);
don't let it get described as equivalent to the Metrica-based entropy
calculation. In particular:

- An "average position" is where a player tends to be *when involved in an
  event*, not where they stand. Players who rarely touch the ball (often
  centre-backs, out of possession) are under-sampled.
- Averaging over many matches blurs genuinely different shapes (formation
  changes, opponents, game state) into one point per player; and pulls
  every player toward their long-run mean, so a pooled multi-match window
  reports a systematically *smaller* shape than the average single match
  (2023 Gotham, out of possession: pooled length ~30 m vs ~37-43 m per
  match). Compare pooled windows with pooled windows, and per-match values
  (compactness_by_match) with per-match values, never one against the other.
- Phase matters. "out_of_possession" (events by the team while the opponent
  has possession; pressures, recoveries, interceptions, clearances, ...)
  is the closest event-data proxy for a *defensive* shape; "in_possession"
  describes the shape with the ball. They are not interchangeable.

Windows are sets of matches (e.g. by match week), because the comparison
this module exists for is across a season, not within a match.

Intended real use case: Gotham FC's 2023 "worst-to-first" NWSL title run;
compare their average defensive compactness early in the season vs. during
their title-winning stretch, using only real, available data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PHASES = ("all", "in_possession", "out_of_possession")
GOALKEEPER_POSITION = "Goalkeeper"


def _filter_phase(events: pd.DataFrame, team: str, phase: str) -> pd.DataFrame:
    """The team's own located, player-attributed events in the given phase."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got '{phase}'.")
    own = events[
        (events["team"] == team)
        & events["x"].notna()
        & events["y"].notna()
        & events["player_id"].notna()
    ]
    if phase == "in_possession":
        own = own[own["possession_team"] == team]
    elif phase == "out_of_possession":
        own = own[own["possession_team"] != team]
    return own


def compute_average_positions(
    events: pd.DataFrame,
    team: str,
    phase: str = "out_of_possession",
    min_touches: int = 20,
) -> pd.DataFrame:
    """Average (x, y) position per player, from their events in `events`.

    Parameters
    ----------
    events : pd.DataFrame
        Event data from statsbomb_loader.get_match_events(); one match or
        several pre-concatenated (the window is whatever you pass in);
        already converted to standard meters via cleaning.statsbomb_to_meters().
    team : str
        Team name to filter to (must match StatsBomb's exact naming).
    phase : str
        "all", "in_possession", or "out_of_possession" (see module docstring).
    min_touches : int
        Players with fewer events than this are kept but flagged
        reliable=False: an average of 2 touches isn't a meaningful "shape"
        data point.

    Returns
    -------
    pd.DataFrame
        One row per player, sorted by n_touches descending: player_id,
        player_name, position (their most frequent listed position), x, y,
        n_touches, is_goalkeeper, reliable.
    """
    own = _filter_phase(events, team, phase)
    columns = ["player_id", "player_name", "position", "x", "y", "n_touches",
               "is_goalkeeper", "reliable"]
    if own.empty:
        return pd.DataFrame(columns=columns)
    grouped = own.groupby("player_id")
    result = pd.DataFrame(
        {
            "player_name": grouped["player_name"].first(),
            "position": grouped["position"].agg(
                lambda s: s.mode().iloc[0] if s.notna().any() else None
            ),
            "x": grouped["x"].mean(),
            "y": grouped["y"].mean(),
            "n_touches": grouped.size(),
        }
    ).reset_index()
    result["is_goalkeeper"] = result["position"] == GOALKEEPER_POSITION
    result["reliable"] = result["n_touches"] >= min_touches
    return result.sort_values("n_touches", ascending=False, ignore_index=True)[columns]


def select_outfield_shape(average_positions: pd.DataFrame, n_outfield: int = 10) -> pd.DataFrame:
    """The `n_outfield` most-involved reliable outfield players.

    Fixing the number of points makes shapes comparable across windows: a
    window in which 16 different outfielders appeared would otherwise look
    "wider" just because it has more points.
    """
    eligible = average_positions[
        average_positions["reliable"] & ~average_positions["is_goalkeeper"]
    ]
    return eligible.nlargest(n_outfield, "n_touches")


def compute_compactness(average_positions: pd.DataFrame, n_outfield: int = 10) -> dict[str, float]:
    """Summary compactness metrics from a set of average player positions.

    Parameters
    ----------
    average_positions : pd.DataFrame
        Output of compute_average_positions(); one row per player.
    n_outfield : int
        Shape is computed from the n most-involved reliable outfield players
        (see select_outfield_shape). Fewer are used if fewer qualify; the
        count is reported.

    Returns
    -------
    dict[str, float]
        {"team_length_m", "team_width_m", "centroid_spread_m",
        "block_height_m", "n_players"}:
        - team_length_m / team_width_m: extent (max - min) of the average
          positions along / across the attack axis;
        - centroid_spread_m: RMS distance of the average positions from
          their centroid;
        - block_height_m: centroid x (standard coordinates: 0 = halfway
          line, negative = own half); how high the shape sits.
        Named differently from entropy.compute_dispersion()'s outputs on
        purpose, so a reader can't mistake one proxy for the other. All NaN
        if no players qualify.
    """
    shape = select_outfield_shape(average_positions, n_outfield)
    if shape.empty:
        nan = float("nan")
        return {"team_length_m": nan, "team_width_m": nan, "centroid_spread_m": nan,
                "block_height_m": nan, "n_players": 0}
    x = shape["x"].to_numpy(float)
    y = shape["y"].to_numpy(float)
    return {
        "team_length_m": float(x.max() - x.min()),
        "team_width_m": float(y.max() - y.min()),
        "centroid_spread_m": float(np.sqrt(np.mean((x - x.mean()) ** 2 + (y - y.mean()) ** 2))),
        "block_height_m": float(x.mean()),
        "n_players": len(shape),
    }


def compare_windows(
    events: pd.DataFrame,
    team: str,
    windows: dict[str, list[int]],
    phase: str = "out_of_possession",
    min_touches: int = 20,
    n_outfield: int = 10,
) -> pd.DataFrame:
    """Compare a team's compactness between windows of matches.

    Convenience function for the headline comparison this module exists
    for: e.g., Gotham FC early-2023 vs. Gotham FC during their title run.

    Parameters
    ----------
    events : pd.DataFrame
        Multi-match events (standard meters) with a match_id column.
    team : str
    windows : dict[str, list[int]]
        {label: [match_id, ...]}; windows may differ in size.
    phase, min_touches, n_outfield
        As in compute_average_positions / compute_compactness. min_touches
        applies to the whole window, so scale it with window length.

    Returns
    -------
    pd.DataFrame
        One row per window, in the given order: window, n_matches,
        n_touches (team events used), then the compactness metrics.
    """
    rows = []
    for label, match_ids in windows.items():
        window_events = events[events["match_id"].isin(match_ids)]
        avg = compute_average_positions(window_events, team, phase, min_touches)
        rows.append(
            {
                "window": label,
                "n_matches": int(window_events["match_id"].nunique()),
                "n_touches": int(avg["n_touches"].sum()) if not avg.empty else 0,
                **compute_compactness(avg, n_outfield),
            }
        )
    return pd.DataFrame(rows)


def compactness_by_match(
    events: pd.DataFrame,
    team: str,
    phase: str = "out_of_possession",
    min_touches: int = 5,
    n_outfield: int = 10,
) -> pd.DataFrame:
    """compare_windows with each match as its own window; a season trend line.

    Single-match averages rest on far fewer events than season windows, so
    expect much more match-to-match noise; read the trend, not single points.
    """
    match_ids = list(dict.fromkeys(events["match_id"]))  # preserve given order
    result = compare_windows(
        events, team, {str(m): [m] for m in match_ids}, phase, min_touches, n_outfield
    )
    return result.rename(columns={"window": "match_id"}).assign(
        match_id=lambda d: d["match_id"].astype(int)
    )


def window_difference_test(
    by_match: pd.DataFrame,
    metric: str,
    window_a: list[int],
    window_b: list[int],
) -> dict[str, float]:
    """Welch's t-test of a per-match compactness metric between two match sets.

    Uses per-match values (compactness_by_match output), so the test sees
    the real match-to-match spread; pooled window values hide it. Welch's
    version does not assume equal variances. With small windows (e.g. 3
    playoff matches) the test has little power; with several metrics tested
    at once, correct for multiple comparisons before calling anything
    significant.

    Parameters
    ----------
    by_match : pd.DataFrame
        Needs 'match_id' and `metric`; one phase only.
    metric : str
    window_a, window_b : list[int]
        match_ids in each window.

    Returns
    -------
    dict[str, float]
        mean, standard error (sem) and n for each window, the difference
        (b - a), and Welch's two-sided p-value.
    """
    from scipy import stats

    a = by_match.loc[by_match["match_id"].isin(window_a), metric].dropna()
    b = by_match.loc[by_match["match_id"].isin(window_b), metric].dropna()
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each window needs at least 2 matches for a t-test.")
    return {
        "mean_a": float(a.mean()), "sem_a": float(a.sem()), "n_a": len(a),
        "mean_b": float(b.mean()), "sem_b": float(b.sem()), "n_b": len(b),
        "difference": float(b.mean() - a.mean()),
        "p_value": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
    }
