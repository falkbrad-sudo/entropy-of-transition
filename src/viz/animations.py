"""Frame-by-frame animations of a tracking sequence.

An animated GIF/MP4 of a counter-attack, with the defending team's shape
metrics drawn alongside and a cursor marking the current frame. Written
directly against matplotlib.animation.FuncAnimation (the same general
pattern as the Friends-of-Tracking LaurieOnTracking tutorials,
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking, but not
adapted from their code). Styling and coordinate handling are shared with
pitch_plots.py so animations and static figures look the same.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from src.viz import pitch_plots as pp

FRAME_RATE_HZ = 25.0


def _offsets(rows: pd.DataFrame) -> np.ndarray:
    """(n, 2) mplsoccer-coordinate array for a scatter, (0, 2) if empty."""
    rows = rows.dropna(subset=["x", "y"])
    px, py = pp.to_pitch_xy(rows["x"], rows["y"])
    return np.column_stack([px, py]) if len(rows) else np.empty((0, 2))


def animate_sequence(
    tracking: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    entropy_series: pd.DataFrame | None = None,
    output_path: str = "output.gif",
    metric_cols: list[str] | None = None,
    labels: dict[str, str] | None = None,
    frame_step: int = 1,
    title: str | None = None,
) -> None:
    """Render an animation of player positions over a frame range.

    Parameters
    ----------
    tracking : pd.DataFrame
        Long-format tracking, both teams + ball, standard meter coordinates
        (team, player_id, frame, x, y). Only [start_frame, end_frame] is used.
    start_frame, end_frame : int
    entropy_series : pd.DataFrame, optional
        If provided (e.g. pipeline output with 'frame' and 'time_s'), render
        as a synced side panel with a moving cursor at the current frame.
    output_path : str
        ".gif" is written with Pillow (no extra dependency); ".mp4" requires
        ffmpeg on the PATH.
    metric_cols : list[str], optional
        Columns of entropy_series to plot (same units; at most 3). Defaults
        to the along/perpendicular axis variances.
    labels : dict[str, str], optional
        Display names for metric_cols.
    frame_step : int
        Render every n-th tracking frame (2 -> 12.5 fps playback at real
        speed). Larger steps give smaller files.
    title : str, optional
    """
    window = tracking[tracking["frame"].between(start_frame, end_frame)]
    frames = sorted(window["frame"].unique())[::frame_step]
    if not frames:
        raise ValueError(f"No tracking frames in [{start_frame}, {end_frame}].")
    by_frame = {f: g for f, g in window.groupby("frame")}

    with_panel = entropy_series is not None
    fig, axes = plt.subplots(
        1, 2 if with_panel else 1, figsize=(13, 4.8) if with_panel else (8, 5.4),
        facecolor=pp.SURFACE, gridspec_kw={"width_ratios": [1.45, 1]} if with_panel else None,
    )
    pitch_ax = axes[0] if with_panel else axes
    pp._draw_pitch(pitch_ax)
    scatters = {
        team: pitch_ax.scatter([], [], s=pp.MARKER_SIZE, color=color, edgecolors=pp.SURFACE,
                               linewidths=1.5, zorder=4, label=team)
        for team, color in pp.TEAM_COLORS.items()
    }
    ball = pitch_ax.scatter([], [], s=40, color=pp.TEXT_PRIMARY, edgecolors=pp.SURFACE,
                            linewidths=1.5, zorder=5, label="Ball")
    pitch_ax.legend(loc="upper left", bbox_to_anchor=(0, 0), ncol=3, frameon=False,
                    fontsize=9, labelcolor=pp.TEXT_PRIMARY)
    clock = pitch_ax.text(0.99, 1.01, "", transform=pitch_ax.transAxes, ha="right",
                          va="bottom", fontsize=9, color=pp.TEXT_SECONDARY)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=pp.TEXT_PRIMARY)

    cursor = None
    if with_panel:
        metric_cols = metric_cols or ["along_axis_variance", "perpendicular_variance"]
        series = entropy_series[entropy_series["frame"].between(start_frame, end_frame)]
        t0 = series["time_s"].iloc[0]
        panel = axes[1]
        panel.set_facecolor(pp.SURFACE)
        for col, color in zip(metric_cols, pp.SERIES_COLORS):
            panel.plot(series["time_s"] - t0, series[col], color=color,
                       linewidth=pp.LINE_WIDTH, label=(labels or {}).get(col, col))
        panel.set_xlabel("Time since window start (s)", color=pp.TEXT_SECONDARY)
        panel.set_ylim(bottom=0)
        panel.grid(axis="y", color=pp.GRID, linewidth=0.8)
        panel.tick_params(colors=pp.TEXT_SECONDARY)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            panel.spines[side].set_color(pp.GRID)
        panel.legend(frameon=False, fontsize=9, labelcolor=pp.TEXT_PRIMARY, loc="upper left")
        cursor = panel.axvline(0, color=pp.TEXT_SECONDARY, linewidth=1)
        frame_time = dict(zip(series["frame"], series["time_s"] - t0))
    fig.tight_layout()

    def update(frame: int):
        rows = by_frame[frame]
        for team, scatter in scatters.items():
            scatter.set_offsets(_offsets(rows[rows["team"] == team]))
        ball.set_offsets(_offsets(rows[rows["team"] == "Ball"]))
        clock.set_text(f"frame {frame}  ·  t = {frame / FRAME_RATE_HZ:.2f} s")
        artists = [*scatters.values(), ball, clock]
        if cursor is not None and frame in frame_time:
            cursor.set_xdata([frame_time[frame]] * 2)
            artists.append(cursor)
        return artists

    fps = FRAME_RATE_HZ / frame_step
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps) if output_path.endswith(".gif") else FFMpegWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=80)
    plt.close(fig)
