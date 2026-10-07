"""Static pitch visualizations using mplsoccer.

Keep all raw matplotlib/mplsoccer boilerplate here; feature and pipeline
modules should never import matplotlib directly, so the visual style stays
consistent and easy to change in one place.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from mplsoccer import Pitch

from src.features.voronoi import compute_voronoi_cells

# Shared style. Series colors are the first slots of a categorical palette
# validated for color-vision-deficiency separation; text never uses them.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
LINE_WIDTH = 2.0
PITCH_LINES = "#c9c8c3"
TEAM_COLORS = {"Home": SERIES_COLORS[0], "Away": SERIES_COLORS[1]}
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
MARKER_SIZE = 90  # points^2 (~9.5 px diameter at 100 dpi)


def to_pitch_xy(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> tuple:
    """Standard meter coordinates -> mplsoccer 'custom' pitch coordinates.

    mplsoccer's custom pitch runs 0..105 x 0..68 with +y pointing up. This
    project's standard coordinates are centred, with +y pointing toward the
    bottom touchline as drawn in the raw data (see cleaning.py), so y is
    flipped here to keep the raw sources' top-of-image at the top.
    """
    return np.asarray(x) + PITCH_LENGTH_M / 2, PITCH_WIDTH_M / 2 - np.asarray(y)


def _draw_pitch(ax: plt.Axes | None = None) -> plt.Axes:
    pitch = Pitch(pitch_type="custom", pitch_length=PITCH_LENGTH_M, pitch_width=PITCH_WIDTH_M,
                  pitch_color=SURFACE, line_color=PITCH_LINES, linewidth=1, line_zorder=1)
    if ax is None:
        _, ax = pitch.draw(figsize=(8, 5.6))
        ax.figure.set_facecolor(SURFACE)
    else:
        pitch.draw(ax=ax)
    return ax


def _scatter_players(ax: plt.Axes, positions: pd.DataFrame, color: str, label: str | None,
                     alpha: float = 1.0) -> None:
    px, py = to_pitch_xy(positions["x"], positions["y"])
    # White ring separates overlapping markers.
    ax.scatter(px, py, s=MARKER_SIZE, color=color, edgecolors=SURFACE, linewidths=1.5,
               zorder=4, label=label, alpha=alpha)


# Candidate label offsets (pitch meters, ha, va): below, above, right, left.
_LABEL_SLOTS = [((0.0, -2.2), "center", "top"), ((0.0, 2.2), "center", "bottom"),
                ((2.0, 0.0), "left", "center"), ((-2.0, 0.0), "right", "center")]
_LABEL_CHAR_WIDTH_M = 1.15  # approx. width of one 7-pt character on a 105 m pitch panel
_LABEL_HEIGHT_M = 2.6


def _label_box(x: float, y: float, text: str, slot: tuple) -> tuple[float, float, float, float]:
    (dx, dy), ha, va = slot
    w, h = len(text) * _LABEL_CHAR_WIDTH_M, _LABEL_HEIGHT_M
    left = {"center": x + dx - w / 2, "left": x + dx, "right": x + dx - w}[ha]
    bottom = {"top": y + dy - h, "bottom": y + dy, "center": y + dy - h / 2}[va]
    return left, bottom, left + w, bottom + h


def _overlaps(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _place_labels(ax: plt.Axes, px: np.ndarray, py: np.ndarray, texts: list[str]) -> None:
    """Greedy label placement: first candidate slot that overlaps no earlier label
    (falls back to 'below'). Approximate; sized for 7-pt text on a full pitch."""
    placed: list[tuple] = []
    for x, y, text in zip(px, py, texts):
        chosen = _LABEL_SLOTS[0]
        for slot in _LABEL_SLOTS:
            if not any(_overlaps(_label_box(x, y, text, slot), box) for box in placed):
                chosen = slot
                break
        placed.append(_label_box(x, y, text, chosen))
        (dx, dy), ha, va = chosen
        ax.text(x + dx, y + dy, text, ha=ha, va=va, fontsize=7, color=TEXT_PRIMARY, zorder=5)


def plot_team_positions(
    positions: pd.DataFrame,
    title: str | None = None,
    ax: plt.Axes | None = None,
    color: str = SERIES_COLORS[0],
    label_col: str | None = None,
) -> plt.Axes:
    """Scatter plot of one team's positions for a single frame.

    Works equally for a tracking frame or for average positions from
    formation.compute_average_positions; the caller's title should say
    which, since they mean very different things.

    Parameters
    ----------
    positions : pd.DataFrame
        One row per player, 'x'/'y' in standard meter coordinates.
    title : str, optional
    ax : matplotlib.axes.Axes, optional
        Pass an existing axes to draw into (e.g., for a subplot grid);
        otherwise creates a new pitch.
    color : str
        Marker color (identity only; labels stay in text ink).
    label_col : str, optional
        Column to annotate each marker with (e.g. a player surname).

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _draw_pitch(ax)
    valid = positions.dropna(subset=["x", "y"])
    _scatter_players(ax, valid, color, None)
    if label_col:
        px, py = to_pitch_xy(valid["x"], valid["y"])
        _place_labels(ax, px, py, valid[label_col].astype(str).tolist())
    if title:
        ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=10)
    return ax


def plot_entropy_timeseries(
    entropy_df: pd.DataFrame,
    metric_col: str | list[str],
    event_markers: pd.DataFrame | None = None,
    labels: dict[str, str] | None = None,
    y_label: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Line plot of an entropy/dispersion metric over a sequence, with
    optional markers for key events (e.g., "possession lost", "shot taken").

    This is likely the single most important plot in the whole project;
    it's the direct visual evidence for how the defense's shape changes
    during the transition.

    Parameters
    ----------
    entropy_df : pd.DataFrame
        Output of entropy.compute_entropy_timeseries (needs 'time_s').
        Time is plotted relative to the first row, in seconds.
    metric_col : str or list[str]
        Column(s) to plot. Several columns share one y-axis, so pass only
        metrics with the same units (e.g. the two axis variances, both m^2).
        At most 3, to keep series colors distinguishable. NaN rows are
        skipped, so a column computed on every Nth frame is drawn through
        those frames.
    event_markers : pd.DataFrame, optional
        Rows with 'time_s' (same clock as entropy_df) and 'label'; drawn as
        labeled vertical rules.
    labels : dict[str, str], optional
        Display name per metric column (defaults to the column name).
    y_label, title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    cols = [metric_col] if isinstance(metric_col, str) else list(metric_col)
    if len(cols) > len(SERIES_COLORS):
        raise ValueError(f"At most {len(SERIES_COLORS)} metrics per plot, got {len(cols)}.")
    labels = labels or {}
    t0 = entropy_df["time_s"].iloc[0]
    t = entropy_df["time_s"] - t0

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ends = []
    for col, color in zip(cols, SERIES_COLORS):
        name = labels.get(col, col)
        # Metrics sampled on fewer frames (NaN elsewhere) are drawn through
        # their sampled points.
        has = entropy_df[col].notna().to_numpy()
        ax.plot(t[has], entropy_df[col][has], color=color, linewidth=LINE_WIDTH, label=name,
                solid_capstyle="round")
        ends.append((name, t[has].iloc[-1], entropy_df[col][has].iloc[-1]))
    # Direct labels at the line ends, in text ink next to a colored mark,
    # nudged vertically so close line ends don't print on top of each other.
    _end_labels(ax, ends)
    if event_markers is not None:
        for _, ev in event_markers.iterrows():
            x = ev["time_s"] - t0
            ax.axvline(x, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
            # Labels in the right half sit left of their rule, so they don't
            # collide with the direct labels at the line ends.
            right_half = x > t.iloc[-1] / 2
            ax.annotate(ev["label"], (x, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(-4 if right_half else 4, -4), textcoords="offset points",
                        ha="right" if right_half else "left", va="top",
                        fontsize=9, color=TEXT_SECONDARY)

    ax.set_xlabel("Time since sequence start (s)", color=TEXT_SECONDARY)
    if y_label:
        ax.set_ylabel(y_label, color=TEXT_SECONDARY)
    if title:
        ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if len(cols) > 1:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.15), ncol=len(cols),
                  labelcolor=TEXT_PRIMARY, fontsize=9)
    fig.tight_layout()
    return fig


_END_LABEL_GAP_PT = 12.0


def _end_labels(ax: plt.Axes, ends: list[tuple[str, float, float]]) -> None:
    """Annotate (text, x, y) line ends, spreading labels >= _END_LABEL_GAP_PT apart."""
    if not ends:
        return
    fig = ax.figure
    fig.canvas.draw()  # finalize limits so data -> display is accurate
    to_pt = 72.0 / fig.dpi
    ys = [ax.transData.transform((x, y))[1] * to_pt for _, x, y in ends]
    order = np.argsort(ys)
    placed = np.array(ys, dtype=float)
    for prev, cur in zip(order[:-1], order[1:]):
        placed[cur] = max(placed[cur], placed[prev] + _END_LABEL_GAP_PT)
    for (text, x, y), y_pt, p_pt in zip(ends, ys, placed):
        ax.annotate(text, (x, y), xytext=(8, p_pt - y_pt),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT_PRIMARY)


def plot_voronoi_diagram(
    positions: pd.DataFrame,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Full-pitch Voronoi diagram, both teams, for a single frame.

    Cells come from voronoi.compute_voronoi_cells (the same construction
    that produces the reported areas) rather than mplsoccer's
    Pitch.voronoi, so the picture and the numbers can't disagree.

    Parameters
    ----------
    positions : pd.DataFrame
        One tracking frame (long format): 'team', 'player_id', 'x', 'y' in
        standard meters. Home/Away rows are players; a 'Ball' row, if
        present, is drawn as the ball.
    title : str, optional
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax = _draw_pitch(ax)
    players = positions[positions["team"].isin(TEAM_COLORS)].dropna(subset=["x", "y"])
    cells = compute_voronoi_cells(players)
    team_of = dict(zip(players["player_id"], players["team"]))
    for pid, poly in cells.items():
        px, py = to_pitch_xy(poly[:, 0], poly[:, 1])
        # Light fill + surface-colored edge = a gap between adjacent cells.
        ax.add_patch(Polygon(np.column_stack([px, py]), closed=True,
                             facecolor=TEAM_COLORS[team_of[pid]], alpha=0.22,
                             edgecolor=SURFACE, linewidth=2, zorder=2))
    for team, color in TEAM_COLORS.items():
        _scatter_players(ax, players[players["team"] == team], color, team)
    ball = positions[(positions["team"] == "Ball")].dropna(subset=["x", "y"])
    if not ball.empty:
        bx, by = to_pitch_xy(ball["x"], ball["y"])
        ax.scatter(bx, by, s=40, color=TEXT_PRIMARY, edgecolors=SURFACE, linewidths=1.5,
                   zorder=5, label="Ball")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0), ncol=3, frameon=False, fontsize=9,
              labelcolor=TEXT_PRIMARY)
    if title:
        ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=10)
    return ax


def plot_pitch_control(
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    defending_control: np.ndarray,
    positions: pd.DataFrame,
    defending_team: str,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Pitch-control surface for one frame, with players, velocities and ball.

    Parameters
    ----------
    xgrid, ygrid, defending_control : arrays
        From pitch_control.compute_pitch_control_surface, in the same
        orientation as `positions` (standard meters).
    positions : pd.DataFrame
        One frame: team, x, y, and optionally vx, vy (drawn as 1-second
        arrows) and a Ball row.
    defending_team : str
        "Home" or "Away". Cells shade from the attacking team's color
        (attackers control) through a neutral gray (contested) to the
        defending team's color.
    title : str, optional
    ax : matplotlib.axes.Axes, optional
    """
    from matplotlib.colors import LinearSegmentedColormap

    ax = _draw_pitch(ax)
    attacking_team = next(t for t in TEAM_COLORS if t != defending_team)
    cmap = LinearSegmentedColormap.from_list(
        "control", [TEAM_COLORS[attacking_team], "#f0efec", TEAM_COLORS[defending_team]]
    )
    # Grid y runs top (-34) to bottom (+34) as drawn; pitch coordinates flip it.
    left, right = to_pitch_xy(np.array([xgrid[0], xgrid[-1]]), np.zeros(2))[0]
    dx, dy = xgrid[1] - xgrid[0], ygrid[1] - ygrid[0]
    top, bottom = to_pitch_xy(np.zeros(2), np.array([ygrid[0], ygrid[-1]]))[1]
    image = ax.imshow(defending_control, cmap=cmap, vmin=0, vmax=1, alpha=0.75, zorder=1.5,
                      extent=(left - dx / 2, right + dx / 2, bottom - dy / 2, top + dy / 2),
                      origin="upper", interpolation="nearest")
    players = positions[positions["team"].isin(TEAM_COLORS)].dropna(subset=["x", "y"])
    for team, color in TEAM_COLORS.items():
        team_rows = players[players["team"] == team]
        if {"vx", "vy"} <= set(team_rows.columns):
            moving = team_rows.dropna(subset=["vx", "vy"])
            px, py = to_pitch_xy(moving["x"], moving["y"])
            ax.quiver(px, py, moving["vx"], -moving["vy"], color=TEXT_SECONDARY,
                      angles="xy", scale_units="xy", scale=1, width=0.0025, zorder=3)
        _scatter_players(ax, team_rows, color, team)
    ball = positions[positions["team"] == "Ball"].dropna(subset=["x", "y"])
    if not ball.empty:
        bx, by = to_pitch_xy(ball["x"], ball["y"])
        ax.scatter(bx, by, s=40, color=TEXT_PRIMARY, edgecolors=SURFACE, linewidths=1.5,
                   zorder=5, label="Ball")
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.03, pad=0.01)
    bar.set_label(f"Probability {defending_team} controls the ball there", color=TEXT_SECONDARY,
                  fontsize=8)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    bar.outline.set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0), ncol=3, frameon=False, fontsize=9,
              labelcolor=TEXT_PRIMARY)
    if title:
        ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=10)
    return ax


def plot_formation_comparison(
    shapes: dict[str, pd.DataFrame],
    title: str | None = None,
    label_col: str | None = None,
) -> plt.Figure:
    """Side-by-side average-position shapes, one pitch per window.

    For the StatsBomb aggregate proxy (formation.py): each panel is a set of
    *average event positions* over a window of matches, not a snapshot of
    the team; the caller's title should say so.

    Parameters
    ----------
    shapes : dict[str, pd.DataFrame]
        {panel title: average positions ('x', 'y', optional label_col)}.
    title : str, optional
    label_col : str, optional
        Column to annotate markers with.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, len(shapes), figsize=(5.2 * len(shapes), 4.2),
                             facecolor=SURFACE)
    for ax, (panel_title, positions) in zip(np.atleast_1d(axes), shapes.items()):
        plot_team_positions(positions, title=panel_title, ax=ax, label_col=label_col)
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def plot_season_trend(
    by_match: pd.DataFrame,
    metric_col: str,
    window_breaks: list[float] | None = None,
    window_labels: list[str] | None = None,
    y_label: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Per-match values of one compactness metric across a season.

    Parameters
    ----------
    by_match : pd.DataFrame
        formation.compactness_by_match output joined to match info (needs
        'match_week' and metric_col); one phase only.
    metric_col : str
    window_breaks : list[float], optional
        match_week positions of window boundaries (e.g. [11.5, 22.5]),
        drawn as dashed rules.
    window_labels : list[str], optional
        One label per window (len(window_breaks) + 1), drawn at the top;
        the mean of each window is drawn as a short horizontal segment.
    y_label, title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = by_match.sort_values("match_week")
    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(data["match_week"], data[metric_col], color=SERIES_COLORS[0],
            linewidth=LINE_WIDTH, marker="o", markersize=6, markeredgecolor=SURFACE,
            markeredgewidth=1.5)
    breaks = window_breaks or []
    edges = [data["match_week"].min() - 0.5, *breaks, data["match_week"].max() + 0.5]
    for b in breaks:
        ax.axvline(b, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)))
    if window_labels:
        for (lo, hi), label in zip(zip(edges[:-1], edges[1:]), window_labels):
            in_window = data[(data["match_week"] > lo) & (data["match_week"] < hi)]
            mean = in_window[metric_col].mean()
            ax.hlines(mean, lo + 0.3, hi - 0.3, color=TEXT_SECONDARY, linewidth=1.5)
            ax.annotate(f"{label}\nmean {mean:.1f} (n={len(in_window)})", ((lo + hi) / 2, 1.0),
                        xycoords=("data", "axes fraction"), ha="center", va="bottom",
                        fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("Match week", color=TEXT_SECONDARY)
    if y_label:
        ax.set_ylabel(y_label, color=TEXT_SECONDARY)
    if title:
        ax.set_title(title, color=TEXT_PRIMARY, loc="left", fontsize=11, pad=34)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    fig.tight_layout()
    return fig


def plot_baseline_comparison(
    matched: pd.DataFrame,
    metrics: dict[str, str],
    row_labels: dict[tuple, str] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Each counter-attack's shape change against its matched controls.

    One panel per metric, one row per counter: grey dots are that counter's
    duration-matched controls, the grey tick their median, the blue marker
    the counter itself. The dashed rule marks "no change".

    Parameters
    ----------
    matched : pd.DataFrame
        baseline.matched_changes output (role, game_id, sequence_id, metrics).
    metrics : dict[str, str]
        {column: axis label}, one panel each.
    row_labels : dict[tuple, str], optional
        {(game_id, sequence_id): label}; defaults to "Game g, #s".
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    keys = sorted(matched.groupby(["game_id", "sequence_id"]).groups)
    labels = row_labels or {k: f"Game {k[0]}, #{k[1]}" for k in keys}
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 0.55 * len(keys) + 1.8),
                             sharey=True, facecolor=SURFACE, squeeze=False)
    rng = np.random.default_rng(0)  # vertical jitter only, for legibility
    for ax, (col, axis_label) in zip(axes[0], metrics.items()):
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        for row, key in enumerate(keys):
            group = matched[(matched["game_id"] == key[0]) & (matched["sequence_id"] == key[1])]
            ctl = group.loc[group["role"] == "control", col]
            ax.scatter(ctl, row + rng.uniform(-0.18, 0.18, len(ctl)), s=10,
                       color=TEXT_SECONDARY, alpha=0.3, linewidths=0, zorder=2,
                       label="Matched controls" if row == 0 else None)
            ax.vlines(ctl.median(), row - 0.3, row + 0.3, color=TEXT_PRIMARY, linewidth=LINE_WIDTH,
                      zorder=3, label="Control median" if row == 0 else None)
            counter = group.loc[group["role"] == "counter", col].iloc[0]
            ax.scatter([counter], [row], s=MARKER_SIZE, color=SERIES_COLORS[0],
                       edgecolors=SURFACE, linewidths=1.5, zorder=4,
                       label="Counter-attack" if row == 0 else None)
        ax.set_xlabel(axis_label, color=TEXT_SECONDARY)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_SECONDARY)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0][0].set_yticks(range(len(keys)), [labels[k] for k in keys])
    axes[0][0].invert_yaxis()
    axes[0][0].tick_params(axis="y", length=0)
    axes[0][-1].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
                       fontsize=8, labelcolor=TEXT_SECONDARY)
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def plot_depth_response(
    changes: pd.DataFrame,
    metrics: dict[str, str],
    bin_edges_m: list[float],
    third_line_m: float | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Shape change against how deep the ball got, counters vs. controls.

    One panel per metric. x is the ball's position at the window end in the
    defending team's frame (its own goal line at -52.5, so left = deeper);
    grey dots are controls, the black step line their median per depth bin,
    blue markers the counter-attacks.

    Parameters
    ----------
    changes : pd.DataFrame
        pipeline.window_end_changes output (role, ball_x_m, metric columns).
    metrics : dict[str, str]
        {column: axis label}, one panel each.
    bin_edges_m : list[float]
        Depth-bin edges for the control medians (increasing).
    third_line_m : float, optional
        Draw a dashed rule here (e.g. -17.5, the edge of the defending third).
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = changes.dropna(subset=["ball_x_m"])
    controls = data[data["role"] == "control"]
    counters = data[data["role"] == "counter"]
    bins = pd.cut(controls["ball_x_m"], bin_edges_m)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.6 * len(metrics), 4.4),
                             facecolor=SURFACE, squeeze=False)
    for ax, (col, axis_label) in zip(axes[0], metrics.items()):
        ax.set_facecolor(SURFACE)
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        if third_line_m is not None:
            ax.axvline(third_line_m, color=GRID, linewidth=1.5, zorder=1)
            ax.annotate("defending third", (third_line_m, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(-4, -4), textcoords="offset points", ha="right", va="top",
                        fontsize=8, color=TEXT_SECONDARY)
        ax.scatter(controls["ball_x_m"], controls[col], s=12, color=TEXT_SECONDARY, alpha=0.3,
                   linewidths=0, zorder=2, label="Controls (no shot)")
        medians = controls.groupby(bins, observed=False)[col].median().dropna()
        for i, (interval, value) in enumerate(medians.items()):
            ax.hlines(value, interval.left, interval.right, color=TEXT_PRIMARY,
                      linewidth=LINE_WIDTH, zorder=3,
                      label="Control median per depth bin" if i == 0 else None)
        ax.scatter(counters["ball_x_m"], counters[col], s=MARKER_SIZE, color=SERIES_COLORS[0],
                   edgecolors=SURFACE, linewidths=1.5, zorder=4, label="Counter-attacks")
        ax.set_xlim(-52.5, 52.5)
        ax.set_xlabel("Ball x at window end (m; defending goal line at −52.5)",
                      color=TEXT_SECONDARY)
        ax.set_ylabel(axis_label, color=TEXT_SECONDARY)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_SECONDARY)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0][-1].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False,
                       fontsize=8, labelcolor=TEXT_SECONDARY)
    if title:
        fig.suptitle(title, color=TEXT_PRIMARY, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def add_logo(fig: plt.Figure, image: np.ndarray, height_in: float = 0.2,
             margin_in: float = 0.05) -> plt.Axes:
    """Place a logo image in the figure's bottom-left corner.

    For data-provider attribution (StatsBomb's open-data terms ask for their
    logo on published work). The logo keeps a fixed physical height, so it
    is the same size on every figure; the source caption sits bottom-right.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    image : np.ndarray
        (h, w, 3 or 4) image array, e.g. from matplotlib.image.imread.
    height_in, margin_in : float
        Logo height and corner margin, in inches.

    Returns
    -------
    matplotlib.axes.Axes
        The (axis-less) axes holding the logo.
    """
    fig_w, fig_h = fig.get_size_inches()
    width_in = height_in * image.shape[1] / image.shape[0]
    ax = fig.add_axes((margin_in / fig_w, margin_in / fig_h, width_in / fig_w, height_in / fig_h))
    ax.imshow(image)
    ax.set_axis_off()
    return ax
