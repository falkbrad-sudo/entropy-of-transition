"""Render the project's report figures from data/processed/ (no raw data needed).

Run as `python -m src.figures` after `python -m src.pipeline`. Writes PNGs and
GIFs to reports/figures/. Every figure carries a data-source caption: the
Metrica sample data asks to be acknowledged, and StatsBomb's open-data terms
ask that published work state StatsBomb as the source *and* show their logo
(from https://statsbomb.com/media-pack/), so the StatsBomb-based figures also
carry the logo in reports/assets/.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402
from src.features.pitch_control import (  # noqa: E402
    compute_pitch_control_surface,
    summarize_surface,
)
from src.features.voronoi import compute_voronoi_territory  # noqa: E402
from src.pipeline import (  # noqa: E402
    AVERAGE_POSITIONS_FILE,
    BASELINE_MATCHED_FILE,
    FORMATION_BY_MATCH_FILE,
    FORMATION_WINDOWS_FILE,
    POSITIONS_FILE,
    SEQUENCES_FILE,
    TIMESERIES_FILE,
    WINDOW_CHANGES_FILE,
    baseline_control_sets,
)
from src.viz import pitch_plots as pp  # noqa: E402
from src.viz.animations import animate_sequence  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"
METRICA_SOURCE = "Data: Metrica Sports open sample data (anonymized teams)"
STATSBOMB_SOURCE = "Data: StatsBomb open data, 2023 NWSL"
STATSBOMB_LOGO = PROJECT_ROOT / "reports/assets/hudl-statsbomb-logo-default.png"
SHAPE_LABELS = {
    "along_axis_variance": "Length (along attack axis)",
    "perpendicular_variance": "Width (across pitch)",
}
CONTROL_LABELS = {
    "defending_fraction": "Nearest-player share (Voronoi)",
    "pitch_control_fraction": "Pitch-control share",
    "own_third_control": "Pitch control of own third",
}
WINDOW_NAMES = {
    "early_regular_season": "Weeks 1–11",
    "late_regular_season": "Weeks 12–22",
    "playoffs": "Playoffs",
}


def _caption(fig: plt.Figure, text: str) -> None:
    fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7, color=pp.TEXT_SECONDARY)


def load_headlines(cfg: dict, processed: Path) -> pd.DataFrame:
    """Join config's curated headline list to the detected sequence table.

    Raises if a curated sequence is no longer detected (e.g. after changing
    the counter-attack thresholds) instead of silently dropping it.
    """
    sequences = pd.read_csv(processed / SEQUENCES_FILE)
    curated = pd.DataFrame(cfg["headlines"]["sequences"])
    headlines = curated.merge(sequences, on=["game_id", "start_frame"], how="left")
    missing = headlines[headlines["sequence_id"].isna()]
    if not missing.empty:
        raise ValueError(
            "Curated headline sequence(s) not in the detected set: "
            f"{missing[['game_id', 'start_frame']].to_dict('records')}"
        )
    headlines["sequence_id"] = headlines["sequence_id"].astype(int)
    return headlines


def sequence_title(seq: pd.Series) -> str:
    """e.g. 'Game 1: Home counter-attack vs Away, 374–385 s (shot ON TARGET-SAVED)'."""
    return (f"Game {seq.game_id}: {seq.attacking_team} counter-attack vs {seq.defending_team}, "
            f"{seq.start_time_s:.0f}–{seq.end_time_s:.0f} s (shot {seq.shot_subtype})")


def render_headline(seq: pd.Series, series: pd.DataFrame, positions: pd.DataFrame,
                    out_dir: Path, frame_step: int) -> None:
    """Time series, Voronoi and pitch control at recovery + shot, and a GIF
    for one headline sequence."""
    stem = f"game{seq.game_id}_seq{seq.sequence_id}"
    markers = pd.DataFrame({
        "time_s": [seq.start_time_s, seq.end_time_s],
        "label": [f"{seq.attacking_team} recovers", f"{seq.attacking_team} shot"],
    })
    fig = pp.plot_entropy_timeseries(
        series, list(SHAPE_LABELS), markers, labels=SHAPE_LABELS,
        y_label=f"{seq.defending_team} outfield positional variance (m²)",
        title=sequence_title(seq),
    )
    _caption(fig, METRICA_SOURCE)
    fig.savefig(out_dir / f"{stem}_shape.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=pp.SURFACE)
    for ax, (frame, label) in zip(axes, [(seq.start_frame, "Recovery"), (seq.end_frame, "Shot")]):
        frame_positions = positions[positions["frame"] == frame]
        territory = compute_voronoi_territory(frame_positions)
        share = territory[f"{seq.defending_team.lower()}_area_m2"]
        pitch_share = share / (105 * 68)
        pp.plot_voronoi_diagram(
            frame_positions, ax=ax,
            title=f"{label}: {seq.defending_team} nearest to {pitch_share:.0%} of the pitch",
        )
    fig.suptitle(sequence_title(seq), x=0.01, ha="left", fontsize=11, color=pp.TEXT_PRIMARY)
    fig.tight_layout()
    _caption(fig, METRICA_SOURCE)
    fig.savefig(out_dir / f"{stem}_voronoi.png", dpi=110)
    plt.close(fig)

    render_pitch_control(seq, series, positions, out_dir, stem)

    animate_sequence(
        positions, int(series["frame"].min()), seq.end_frame, series,
        str(out_dir / f"{stem}.gif"), labels=SHAPE_LABELS, frame_step=frame_step,
        title=sequence_title(seq),
    )


def defending_frame_surface(
    frame_positions: pd.DataFrame, seq: pd.Series
) -> tuple:
    """Pitch control for one saved frame, returned in the raw orientation.

    pitch_control expects the defending team's frame (own goal at -x). The
    saved positions are raw, so they are flipped (x, y, vx, vy -> negated)
    when the defending goalkeeper is on the +x side, and the surface is
    flipped back for drawing; the grid is symmetric about the centre.
    """
    gk_x = frame_positions.loc[frame_positions["player_id"] == seq.defending_goalkeeper, "x"]
    sign = -1.0 if gk_x.mean() > 0 else 1.0
    flipped = frame_positions.assign(**{c: frame_positions[c] * sign
                                        for c in ("x", "y", "vx", "vy")})
    xgrid, ygrid, surface = compute_pitch_control_surface(
        flipped, seq.attacking_team, seq.defending_goalkeeper
    )
    summary = summarize_surface(xgrid, surface)
    return xgrid, ygrid, surface if sign > 0 else surface[::-1, ::-1], summary


def render_pitch_control(seq: pd.Series, series: pd.DataFrame, positions: pd.DataFrame,
                         out_dir: Path, stem: str) -> None:
    """Pitch-control surface at recovery + shot, and control shares over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=pp.SURFACE)
    for ax, (frame, label) in zip(axes, [(seq.start_frame, "Recovery"), (seq.end_frame, "Shot")]):
        frame_positions = positions[positions["frame"] == frame]
        xgrid, ygrid, surface, summary = defending_frame_surface(frame_positions, seq)
        pp.plot_pitch_control(
            xgrid, ygrid, surface, frame_positions, seq.defending_team, ax=ax,
            title=f"{label}: {seq.defending_team} controls "
                  f"{summary['pitch_control_fraction']:.0%} of the pitch, "
                  f"{summary['own_third_control']:.0%} of its own third",
        )
    fig.suptitle(sequence_title(seq) + ": pitch control (Spearman 2018; arrows: 1 s of motion)",
                 x=0.01, ha="left", fontsize=11, color=pp.TEXT_PRIMARY)
    fig.tight_layout()
    _caption(fig, METRICA_SOURCE)
    fig.savefig(out_dir / f"{stem}_pitch_control.png", dpi=110)
    plt.close(fig)

    markers = pd.DataFrame({
        "time_s": [seq.start_time_s, seq.end_time_s],
        "label": [f"{seq.attacking_team} recovers", f"{seq.attacking_team} shot"],
    })
    fig = pp.plot_entropy_timeseries(
        series, list(CONTROL_LABELS), markers, labels=CONTROL_LABELS,
        y_label=f"{seq.defending_team} share", title=sequence_title(seq),
    )
    fig.axes[0].set_ylim(0, 1)
    _caption(fig, METRICA_SOURCE + "; pitch control sampled 5 times a second")
    fig.savefig(out_dir / f"{stem}_control.png", dpi=130)
    plt.close(fig)


def render_gotham(processed: Path, out_dir: Path, team: str) -> None:
    """Shape comparison and per-match trend figures for the StatsBomb case study."""
    positions = pd.read_csv(processed / AVERAGE_POSITIONS_FILE)
    windows = pd.read_csv(processed / FORMATION_WINDOWS_FILE)
    by_match = pd.read_csv(processed / FORMATION_BY_MATCH_FILE)
    phase = "out_of_possession"
    logo = mpimg.imread(STATSBOMB_LOGO)

    shape = positions[(positions["phase"] == phase) & positions["in_shape"]].copy()
    shape["surname"] = shape["player_name"].str.split().str[-1]
    w = windows[windows["phase"] == phase].set_index("window")
    panels = {
        f"{WINDOW_NAMES[k]} ({int(w.loc[k, 'n_matches'])} matches): "
        f"length {w.loc[k, 'team_length_m']:.0f} m, width {w.loc[k, 'team_width_m']:.0f} m": g
        for k, g in shape.groupby("window", sort=False)
    }
    fig = pp.plot_formation_comparison(
        panels, label_col="surname",
        title=f"{team} 2023: average event positions out of possession "
              "(10 most-involved outfielders; attacking left → right)",
    )
    _caption(fig, STATSBOMB_SOURCE + "; event-position averages, not tracking")
    pp.add_logo(fig, logo)
    fig.savefig(out_dir / "gotham_shapes_out_of_possession.png", dpi=120)
    plt.close(fig)

    oop = by_match[by_match["phase"] == phase]
    for metric, label in [("block_height_m", "Block height (m from halfway; − = own half)"),
                          ("team_length_m", "Team length (m)")]:
        fig = pp.plot_season_trend(
            oop, metric, window_breaks=[11.5, 22.5],
            window_labels=list(WINDOW_NAMES.values()), y_label=label,
            title=f"{team} 2023, out of possession: per-match {label.split(' (')[0].lower()}",
        )
        _caption(fig, STATSBOMB_SOURCE)
        pp.add_logo(fig, logo)
        fig.savefig(out_dir / f"gotham_trend_{metric}.png", dpi=130)
        plt.close(fig)


BASELINE_METRICS = {
    "width_log2_ratio": "Width change (log₂ ratio; −1 = halved)",
    "length_log2_ratio": "Length change (log₂ ratio; +1 = doubled)",
}
CONTROL_SET_TITLES = {
    "all": "all own-half recoveries with no shot",
    "ball_in_defending_third": "own-half recoveries with no shot, ball in the defending third "
                               "by then",
    "ball_depth_matched": "own-half recoveries with no shot, ball within ±{caliper:.0f} m of "
                          "the counter's ball at the shot",
}
DEPTH_BIN_EDGES_M = [-52.5, -35.0, -17.5, 0.0, 17.5, 35.0, 52.5]


def render_baseline(processed: Path, out_dir: Path, baseline_cfg: dict) -> None:
    """Counter-attacks vs. duration-matched controls (one figure per control
    set), and shape change against ball depth."""
    matched = pd.read_csv(processed / BASELINE_MATCHED_FILE)
    sequences = pd.read_csv(processed / SEQUENCES_FILE).set_index(["game_id", "sequence_id"])
    row_labels = {k: f"Game {k[0]}, #{k[1]} ({row.duration_s:.0f} s)"
                  for k, row in sequences.iterrows()}
    control_sets = baseline_control_sets(
        matched, baseline_cfg["ball_x_max_m"], baseline_cfg["depth_caliper_m"]
    )
    for name, subset in control_sets.items():
        controls_title = CONTROL_SET_TITLES[name].format(caliper=baseline_cfg["depth_caliper_m"])
        fig = pp.plot_baseline_comparison(
            subset, BASELINE_METRICS, row_labels,
            title="Defending shape change, recovery to shot, vs. controls at the same elapsed "
                  f"time\nControls: {controls_title}",
        )
        _caption(fig, METRICA_SOURCE)
        fig.savefig(out_dir / f"baseline_{name}.png", dpi=130)
        plt.close(fig)

    fig = pp.plot_depth_response(
        pd.read_csv(processed / WINDOW_CHANGES_FILE), BASELINE_METRICS, DEPTH_BIN_EDGES_M,
        third_line_m=baseline_cfg["ball_x_max_m"],
        title="Defending shape change from recovery to window end, by how deep the ball got",
    )
    _caption(fig, METRICA_SOURCE + "; controls measured at their own possession end (3–15 s)")
    fig.savefig(out_dir / "baseline_depth_response.png", dpi=130)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    processed = resolve_path(cfg["paths"]["data_processed"])
    out_dir = resolve_path(FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    series_all = pd.read_csv(processed / TIMESERIES_FILE)
    positions_all = pd.read_csv(processed / POSITIONS_FILE)
    for _, seq in load_headlines(cfg, processed).iterrows():
        key = (series_all["game_id"] == seq.game_id) & (
            series_all["sequence_id"] == seq.sequence_id
        )
        pkey = (positions_all["game_id"] == seq.game_id) & (
            positions_all["sequence_id"] == seq.sequence_id
        )
        logger.info("Rendering %s", sequence_title(seq))
        render_headline(seq, series_all[key], positions_all[pkey], out_dir,
                        cfg["headlines"]["animation_frame_step"])
    render_baseline(processed, out_dir, cfg["features"]["baseline"])
    render_gotham(processed, out_dir, cfg["features"]["formation"]["team"])
    logger.info("Figures written to %s", out_dir)


if __name__ == "__main__":
    main()
