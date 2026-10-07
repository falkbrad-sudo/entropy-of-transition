"""Voronoi territory: the part of the pitch nearest to each player.

REQUIRES Metrica tracking data; same constraint as entropy.py.

Given all 22 players' positions at one frame, each player "controls" the
region closer to them than to any other player (the Voronoi diagram,
scipy.spatial.Voronoi, clipped to the pitch). Summing each team's regions
gives a simple, fast "who covers more of the pitch" metric per frame.

It ignores velocity, reaction time and the ball, so it measures spatial
coverage, not control. src/features/pitch_control.py implements
Spearman's (2018) pitch-control model, which accounts for those; the two
are reported side by side so the difference is visible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi

PLAYER_TEAMS = ("Home", "Away")
# Points are kept this far inside the pitch so that no player lies exactly on
# an edge, where it would coincide with its own mirror image (Qhull rejects
# coincident input points).
_EDGE_EPSILON_M = 1e-6


def _clamp_unique(xy: np.ndarray, pitch_length: float, pitch_width: float):
    """Clamp points onto the pitch and deduplicate coincident ones.

    Returns (unique_xy, owner): owner[i] is the row of unique_xy for point i.
    """
    max_x = pitch_length / 2 - _EDGE_EPSILON_M
    max_y = pitch_width / 2 - _EDGE_EPSILON_M
    clamped = np.column_stack([np.clip(xy[:, 0], -max_x, max_x),
                               np.clip(xy[:, 1], -max_y, max_y)])
    unique_xy, owner = np.unique(clamped, axis=0, return_inverse=True)
    if len(unique_xy) < 2:
        raise ValueError("Need at least 2 distinct player positions for a Voronoi diagram.")
    return unique_xy, owner.ravel()


def _clipped_cells(unique_xy: np.ndarray, pitch_length: float, pitch_width: float) -> list:
    """Pitch-clipped Voronoi cell of each point, as a counter-clockwise (k, 2) array."""
    mirrored = [
        unique_xy,
        np.column_stack([-pitch_length - unique_xy[:, 0], unique_xy[:, 1]]),  # x = -L/2 edge
        np.column_stack([pitch_length - unique_xy[:, 0], unique_xy[:, 1]]),  # x = +L/2 edge
        np.column_stack([unique_xy[:, 0], -pitch_width - unique_xy[:, 1]]),  # y = -W/2 edge
        np.column_stack([unique_xy[:, 0], pitch_width - unique_xy[:, 1]]),  # y = +W/2 edge
    ]
    vor = Voronoi(np.vstack(mirrored))
    cells = []
    for i in range(len(unique_xy)):
        vertices = vor.vertices[vor.regions[vor.point_region[i]]]
        # Cells are convex, so sorting by angle about the vertex mean orders them.
        centre = vertices.mean(axis=0)
        order = np.argsort(np.arctan2(vertices[:, 1] - centre[1], vertices[:, 0] - centre[0]))
        cells.append(vertices[order])
    return cells


def _polygon_area(vertices: np.ndarray) -> float:
    """Shoelace area of a simple polygon given in order."""
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _point_areas(xy: np.ndarray, pitch_length: float, pitch_width: float) -> np.ndarray:
    """Clipped Voronoi area of each point in xy (no NaNs); coincident points
    split their shared cell equally."""
    unique_xy, owner = _clamp_unique(xy, pitch_length, pitch_width)
    cells = _clipped_cells(unique_xy, pitch_length, pitch_width)
    unique_areas = np.array([_polygon_area(c) for c in cells])
    shares = np.bincount(owner)  # points stacked on each unique position
    return unique_areas[owner] / shares[owner]


def compute_voronoi_areas(
    positions: pd.DataFrame,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> pd.Series:
    """Area of each player's pitch-clipped Voronoi cell, for one frame.

    Method: every player is reflected across each of the four pitch edges
    and the Voronoi diagram is built over the original + mirrored points.
    Each edge is then the perpendicular bisector between a player and its
    mirror image, so the original players' cells are finite and are exactly
    their unbounded cells clipped to the pitch rectangle; no polygon-
    clipping library needed. Cells are convex, so ordering each cell's
    vertices by angle gives its polygon, whose area is the shoelace
    formula. The cells tile the pitch, so the areas sum to
    pitch_length * pitch_width.

    Parameters
    ----------
    positions : pd.DataFrame
        One frame, one row per player, with 'player_id', 'x', 'y' in standard
        meter coordinates. Rows with NaN coordinates are ignored. Positions
        outside the pitch (tracking noise near the lines) are clamped onto
        it first, since the method assumes every point is inside.
    pitch_length, pitch_width : float

    Returns
    -------
    pd.Series
        Area in m^2, indexed by player_id. Players at exactly the same
        position split their shared cell equally.

    Raises
    ------
    ValueError
        If fewer than 2 distinct positions remain.
    """
    valid = positions.dropna(subset=["x", "y"])
    areas = _point_areas(valid[["x", "y"]].to_numpy(float), pitch_length, pitch_width)
    return pd.Series(areas, index=valid["player_id"].to_numpy(), name="area_m2")


def compute_voronoi_cells(
    positions: pd.DataFrame,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> dict[str, np.ndarray]:
    """Polygon of each player's pitch-clipped Voronoi cell, for one frame.

    Same construction as compute_voronoi_areas (use this for drawing, so the
    picture shows exactly the cells whose areas are reported).

    Returns
    -------
    dict[str, np.ndarray]
        {player_id: (k, 2) array of cell vertices in counter-clockwise
        order}, standard meter coordinates. Coincident players share one
        polygon.
    """
    valid = positions.dropna(subset=["x", "y"])
    unique_xy, owner = _clamp_unique(valid[["x", "y"]].to_numpy(float), pitch_length,
                                     pitch_width)
    cells = _clipped_cells(unique_xy, pitch_length, pitch_width)
    return {pid: cells[o] for pid, o in zip(valid["player_id"].to_numpy(), owner)}


def compute_voronoi_territory(
    positions: pd.DataFrame,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> dict[str, float]:
    """Simple Voronoi-based territory control for one frame, both teams.

    Parameters
    ----------
    positions : pd.DataFrame
        One row per player (all 22, both teams, for one frame), with 'x',
        'y' in standard meter coordinates, a 'player_id' column, and a
        'team' column identifying which side each player is on ("Home" /
        "Away"). Rows for any other team value (e.g. the ball) are ignored.
    pitch_length, pitch_width : float

    Returns
    -------
    dict[str, float]
        {"home_area_m2": ..., "away_area_m2": ..., "home_fraction": ...}

    Notes
    -----
    This is pure nearest-player territory: it ignores velocity, reaction
    time and the ball's position, so it measures *spatial* coverage, not
    control in the sense of Spearman's (2018) pitch-control model.
    Including the goalkeepers is deliberate; they genuinely are the nearest
    player to the space behind their defensive line.
    """
    players = positions[positions["team"].isin(PLAYER_TEAMS)]
    areas = compute_voronoi_areas(players, pitch_length, pitch_width)
    team_of = players.dropna(subset=["x", "y"])["team"].to_numpy()
    home_area = float(areas[team_of == "Home"].sum())
    away_area = float(areas[team_of == "Away"].sum())
    return {
        "home_area_m2": home_area,
        "away_area_m2": away_area,
        "home_fraction": home_area / (home_area + away_area),
    }


def compute_territory_timeseries(
    tracking: pd.DataFrame,
    start_frame: int,
    end_frame: int,
) -> pd.DataFrame:
    """Apply compute_voronoi_territory across every frame in a sequence.

    Same pattern as entropy.compute_entropy_timeseries; this is the
    function that produces a plottable time series of territory control
    shrinking/growing across a real transition sequence.

    Parameters
    ----------
    tracking : pd.DataFrame
        Long-format tracking for both teams (metrica_loader.load_match_tracking),
        in standard meter coordinates.
    start_frame, end_frame : int
        Inclusive frame range.

    Returns
    -------
    pd.DataFrame
        One row per frame: frame, time_s, home_area_m2, away_area_m2,
        home_fraction.
    """
    mask = (
        tracking["frame"].between(start_frame, end_frame)
        & tracking["team"].isin(PLAYER_TEAMS)
        & tracking["x"].notna()
        & tracking["y"].notna()
    )
    # Same computation as compute_voronoi_territory per frame, on plain
    # arrays (per-frame DataFrame overhead dominated the run time).
    players = tracking.loc[mask, ["frame", "time_s", "team", "x", "y"]].sort_values(
        "frame", kind="stable"
    )
    frames = players["frame"].to_numpy()
    times = players["time_s"].to_numpy()
    xy = players[["x", "y"]].to_numpy(float)
    is_home = (players["team"] == "Home").to_numpy()
    bounds = np.flatnonzero(np.r_[True, frames[1:] != frames[:-1], True])
    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        areas = _point_areas(xy[lo:hi], 105.0, 68.0)
        home_area = float(areas[is_home[lo:hi]].sum())
        away_area = float(areas[~is_home[lo:hi]].sum())
        rows.append({"frame": frames[lo], "time_s": times[lo], "home_area_m2": home_area,
                     "away_area_m2": away_area,
                     "home_fraction": home_area / (home_area + away_area)})
    return pd.DataFrame(rows, columns=["frame", "time_s", "home_area_m2", "away_area_m2",
                                       "home_fraction"])
