"""Shared cleaning and coordinate-conversion utilities.

Every data source in this project uses a different coordinate convention.
Converting to a single standard (meters, origin at pitch center, x positive
toward the away team's goal) here means every feature/viz module downstream
can assume one consistent system and never has to know which raw source a
DataFrame originally came from.

Standard convention used everywhere past this module:
- Units: meters
- Origin: center of the pitch
- x: -52.5 (defending goal) to +52.5 (attacking goal), for a 105m-long pitch
- y: -34 (one touchline) to +34 (other touchline), for a 68m-wide pitch

Two details of that convention that the converters alone do NOT handle:
- Both raw sources put their origin at the top-left corner, so after
  conversion +y points toward the bottom touchline (as drawn in the raw
  data's frame of reference). This is a reflection-free shift + scale, so
  distances and areas are unaffected.
- Teams swap ends at halftime. "x toward the attacking goal" is only true
  for a given team after normalize_attack_direction(), which flips each
  period as needed so the chosen team attacks +x throughout the match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# StatsBomb's pitch-unit convention: 120 x 80, origin at top-left corner.
STATSBOMB_LENGTH_UNITS = 120.0
STATSBOMB_WIDTH_UNITS = 80.0


def metrica_to_meters(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Convert Metrica's normalized (0-1) coordinates to standard meters.

    Metrica's raw data has (0, 0) at the top-left, (1, 1) at the bottom-right,
    (0.5, 0.5) at kickoff, for a 105m x 68m pitch.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain x_col and y_col with values in [0, 1].
    x_col, y_col : str
        Column names to convert (function returns a copy with these columns
        replaced; does not mutate the input).

    Returns
    -------
    pd.DataFrame
        Copy of df with x_col, y_col converted to the standard meter
        convention described in this module's docstring.
    """
    out = df.copy()
    out[x_col] = (out[x_col] - 0.5) * PITCH_LENGTH_M
    out[y_col] = (out[y_col] - 0.5) * PITCH_WIDTH_M
    return out


def statsbomb_to_meters(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Convert StatsBomb's 120x80 pitch-unit coordinates to standard meters.

    StatsBomb's raw data has (0, 0) at the top-left corner, (120, 80) at the
    bottom-right, regardless of actual pitch dimensions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain x_col and y_col with values in [0, 120] / [0, 80]
        respectively.
    x_col, y_col : str

    Returns
    -------
    pd.DataFrame
        Copy of df with x_col, y_col converted to the standard meter
        convention described in this module's docstring.
    """
    out = df.copy()
    out[x_col] = (out[x_col] / STATSBOMB_LENGTH_UNITS - 0.5) * PITCH_LENGTH_M
    out[y_col] = (out[y_col] / STATSBOMB_WIDTH_UNITS - 0.5) * PITCH_WIDTH_M
    return out


def validate_coordinates(
    df: pd.DataFrame, x_col: str, y_col: str, margin_m: float = 1.0
) -> None:
    """Raise if any (x, y) pair falls meaningfully outside pitch bounds.

    A cheap sanity check to run right after any coordinate conversion;
    catches unit-conversion bugs (e.g. forgetting a source uses 0-1 instead
    of 0-100) before they silently corrupt downstream feature calculations.

    Parameters
    ----------
    df : pd.DataFrame
    x_col, y_col : str
    margin_m : float
        Allowed distance beyond the pitch lines. 1 m suits event data; real
        tracking data legitimately has players several meters outside the
        lines (corners, throw-ins), so pass a larger margin for it (see
        config.yaml: cleaning.tracking_margin_m). NaN coordinates are
        ignored.

    Raises
    ------
    ValueError
        If any point falls more than margin_m outside the standard pitch
        bounds defined in this module.
    """
    bad_x = (df[x_col].abs() > PITCH_LENGTH_M / 2 + margin_m).any()
    bad_y = (df[y_col].abs() > PITCH_WIDTH_M / 2 + margin_m).any()
    if bad_x or bad_y:
        raise ValueError(
            f"Coordinates in columns '{x_col}'/'{y_col}' fall outside "
            f"expected pitch bounds after conversion; check units."
        )


def attack_direction_by_period(
    tracking: pd.DataFrame,
    team: str,
) -> dict[int, int]:
    """Infer which way a team attacks in each period, from kickoff positions.

    At the first frame of each period every player is in their own half, so
    the sign of the team's mean x at that frame tells us which goal they are
    defending: a team whose players sit at negative x defends the -x goal and
    therefore attacks toward +x.

    Parameters
    ----------
    tracking : pd.DataFrame
        Long-format tracking data in standard meter coordinates, with
        'period', 'frame', 'team', 'x' columns (see
        metrica_loader.load_tracking).
    team : str
        Team whose attack direction to infer.

    Returns
    -------
    dict[int, int]
        {period: +1 or -1}, where +1 means the team already attacks toward
        +x in that period and -1 means its coordinates need flipping.

    Raises
    ------
    ValueError
        If the team has no valid positions at the first frame of a period,
        or its mean x there is exactly 0 (direction genuinely ambiguous).
    """
    team_rows = tracking[(tracking["team"] == team) & tracking["x"].notna()]
    directions: dict[int, int] = {}
    for period, period_rows in team_rows.groupby("period"):
        first_frame = period_rows["frame"].min()
        mean_x = period_rows.loc[period_rows["frame"] == first_frame, "x"].mean()
        if not np.isfinite(mean_x) or mean_x == 0:
            raise ValueError(
                f"Cannot infer attack direction for team '{team}' in period "
                f"{period}: mean kickoff x is {mean_x}."
            )
        directions[int(period)] = 1 if mean_x < 0 else -1
    return directions


def normalize_attack_direction(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    directions: dict[int, int],
    period_col: str = "period",
) -> pd.DataFrame:
    """Rotate coordinates 180 degrees in periods where the team attacks -x.

    Flipping both x and y (a rotation about the center spot, not a
    reflection) preserves handedness, so "left back" stays on the same side
    of the team's shape before and after the flip. Apply the same
    `directions` to tracking and event data for the same match so they stay
    in one consistent frame of reference.

    Parameters
    ----------
    df : pd.DataFrame
        Data in standard meter coordinates with a period column.
    x_col, y_col : str
        Coordinate columns to transform (a copy is returned; input is not
        mutated).
    directions : dict[int, int]
        Output of attack_direction_by_period().
    period_col : str
        Name of the period column.

    Returns
    -------
    pd.DataFrame
        Copy of df in which the reference team attacks toward +x in every
        period.
    """
    out = df.copy()
    sign = out[period_col].map(directions)
    if sign.isna().any():
        missing = sorted(out.loc[sign.isna(), period_col].unique())
        raise ValueError(f"No attack direction provided for period(s) {missing}.")
    out[x_col] = out[x_col] * sign
    out[y_col] = out[y_col] * sign
    return out


def infer_goalkeeper(tracking: pd.DataFrame, team: str) -> str:
    """Identify a team's goalkeeper as the player deepest toward their own goal.

    Metrica's sample data does not label positions, so this is inferred:
    after normalize_attack_direction(), the goalkeeper is the player with the
    lowest mean x over the match. Only the starting goalkeeper is returned;
    a goalkeeper substitution would need separate handling.

    Parameters
    ----------
    tracking : pd.DataFrame
        Long-format tracking data, direction-normalized for `team`, with
        'team', 'player_id', 'x' columns.
    team : str

    Returns
    -------
    str
        player_id of the inferred goalkeeper.
    """
    team_rows = tracking[tracking["team"] == team]
    return str(team_rows.groupby("player_id")["x"].mean().idxmin())
