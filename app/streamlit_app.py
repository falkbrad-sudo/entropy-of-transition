"""Interactive demo: The Entropy of Transition.

Run with: streamlit run app/streamlit_app.py
(after `python -m src.pipeline`; `python -m src.figures` adds the animations)

Goal: let a non-technical viewer (a coach, a GM, a fan) step through real
counter-attacks and see how the defending team's shape changes, without
reading any code. The plots and one-line plain-language takeaways matter
more here than exposing every parameter. Everything shown is read from data/processed/ (no
live recomputation from raw data), and every takeaway sentence is computed
from those numbers rather than written by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` puts app/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config, resolve_path  # noqa: E402
from src.features.voronoi import compute_voronoi_territory  # noqa: E402
from src.figures import (  # noqa: E402
    CONTROL_LABELS,
    FIGURES_DIR,
    SHAPE_LABELS,
    STATSBOMB_LOGO,
    WINDOW_NAMES,
    sequence_title,
)
from src.pipeline import (  # noqa: E402
    AVERAGE_POSITIONS_FILE,
    FORMATION_BY_MATCH_FILE,
    FORMATION_WINDOWS_FILE,
    POSITIONS_FILE,
    SEQUENCES_FILE,
    TIMESERIES_FILE,
)
from src.viz import pitch_plots as pp  # noqa: E402

st.set_page_config(page_title="The Entropy of Transition", layout="wide")

PHASE_NAMES = {"out_of_possession": "Without the ball", "in_possession": "With the ball"}


@st.cache_data
def load_processed(name: str) -> pd.DataFrame:
    cfg = load_config()
    return pd.read_csv(resolve_path(cfg["paths"]["data_processed"]) / name)


def shape_takeaway(seq: pd.Series, series: pd.DataFrame) -> str:
    """One plain-language sentence, computed from the recovery and shot frames."""
    at_recovery = series.loc[series["frame"] == seq.start_frame].iloc[0]
    at_shot = series.iloc[-1]
    length_ratio = at_shot.along_axis_variance / at_recovery.along_axis_variance
    width_ratio = at_shot.perpendicular_variance / at_recovery.perpendicular_variance

    def change(ratio: float, grew: str, shrank: str) -> str:
        if ratio >= 1:
            return f"{grew} by {ratio:.1f}×"
        return f"{shrank} by {1 / ratio:.1f}×"

    return (
        f"In the {seq.duration_s:.0f} s between the recovery and the shot, {seq.defending_team}'s "
        f"outfield shape {change(length_ratio, 'stretched front-to-back', 'shortened')} "
        f"and {change(width_ratio, 'spread across the pitch', 'narrowed')} "
        f"(positional variance), and the share of the pitch closest to one of "
        f"{seq.defending_team}'s players went from {at_recovery.defending_fraction:.0%} "
        f"to {at_shot.defending_fraction:.0%}."
    )


def tracking_tab(cfg: dict) -> None:
    sequences = load_processed(SEQUENCES_FILE)
    series_all = load_processed(TIMESERIES_FILE)
    positions_all = load_processed(POSITIONS_FILE)
    curated = pd.DataFrame(cfg["headlines"]["sequences"])
    sequences = sequences.merge(curated, on=["game_id", "start_frame"], how="left")
    sequences["headline"] = sequences["reason"].notna()

    show_all = st.toggle("Show all detected counter-attacks (not just the 3 headlines)")
    options = sequences if show_all else sequences[sequences["headline"]]
    labels = {sequence_title(row): idx for idx, row in options.iterrows()}
    seq = sequences.loc[labels[st.selectbox("Counter-attack", list(labels))]]
    if seq.headline:
        st.caption(f"Why this one: {seq.reason}")

    key = (series_all["game_id"] == seq.game_id) & (series_all["sequence_id"] == seq.sequence_id)
    series = series_all[key]
    pkey = (positions_all["game_id"] == seq.game_id) & (
        positions_all["sequence_id"] == seq.sequence_id
    )
    positions = positions_all[pkey]
    st.markdown(f"**{shape_takeaway(seq, series)}**")
    st.caption(
        "For context: defenses narrow like this whenever the ball gets near their goal, "
        "counter-attack or not. Compared with other possessions whose ball got as deep, "
        "these counters are not clearly different (README, Findings 1b–1c)."
    )

    left, right = st.columns(2)
    with left:
        markers = pd.DataFrame({
            "time_s": [seq.start_time_s, seq.end_time_s],
            "label": [f"{seq.attacking_team} recovers", f"{seq.attacking_team} shot"],
        })
        fig = pp.plot_entropy_timeseries(
            series, list(SHAPE_LABELS), markers, labels=SHAPE_LABELS,
            y_label=f"{seq.defending_team} outfield positional variance (m²)",
        )
        st.pyplot(fig)
        plt.close(fig)
    with right:
        frames = sorted(positions["frame"].unique())
        frame = st.select_slider(
            "Frame (25 per second)", options=frames, value=int(seq.start_frame),
            format_func=lambda f: f"{(f - seq.start_frame) / 25:+.1f} s from recovery",
        )
        frame_positions = positions[positions["frame"] == frame]
        territory = compute_voronoi_territory(frame_positions)
        share = territory[f"{seq.defending_team.lower()}_area_m2"] / (105 * 68)
        fig, ax = plt.subplots(figsize=(8, 5.6), facecolor=pp.SURFACE)
        pp.plot_voronoi_diagram(
            frame_positions, ax=ax,
            title=f"Each player's nearest region: {seq.defending_team}: {share:.0%}",
        )
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Space control over time: nearest player vs. pitch control"):
        markers = pd.DataFrame({
            "time_s": [seq.start_time_s, seq.end_time_s],
            "label": [f"{seq.attacking_team} recovers", f"{seq.attacking_team} shot"],
        })
        fig = pp.plot_entropy_timeseries(
            series, list(CONTROL_LABELS), markers, labels=CONTROL_LABELS,
            y_label=f"{seq.defending_team} share",
        )
        fig.axes[0].set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Pitch control (computed 5 times a second) also accounts for which way "
                   "players are running and how long the ball takes to arrive.")

    gif = resolve_path(FIGURES_DIR) / f"game{seq.game_id}_seq{seq.sequence_id}.gif"
    if gif.exists():
        with st.expander("Animation"):
            st.image(str(gif))
    elif seq.headline:
        st.caption("Run `python -m src.figures` to render this sequence's animation.")

    with st.expander("How these numbers are defined"):
        st.markdown(
            "- **Length / width**: variance of the defending outfield players' positions along "
            "and across the direction of attack (goalkeeper excluded). Together they are the "
            "squared radius of gyration of the shape.\n"
            "- **Nearest region (Voronoi)**: the part of the pitch closer to a given player than "
            "to anyone else. It ignores speed and the ball, so it measures coverage, not control. "
            "A team pushed back toward its own goal loses area almost by construction.\n"
            "- **Pitch control**: the probability that the defending team would win a ball "
            "played to each point, given every player's position and velocity and the "
            "ball's travel time (Spearman 2018, with Laurie Shaw's published parameters). "
            "Shown for the whole pitch and for the defending team's own third.\n"
            "- **Counter-attack**: a ball recovery in the team's own half followed by its shot "
            f"{cfg['features']['sequences']['min_seconds']}–"
            f"{cfg['features']['sequences']['max_seconds']} s later, with no stoppage or "
            "turnover in between (see src/data/sequences.py)."
        )


def formation_tab(cfg: dict) -> None:
    team = cfg["features"]["formation"]["team"]
    st.subheader(f"{team}: early season vs. late season vs. playoffs")
    st.caption(
        "Built from where players were when they made an event (pass, pressure, tackle, …), "
        "averaged over many matches. This is a coarser proxy than the tracking data in the "
        "other tab: StatsBomb's free 2023 NWSL data records the player on the ball for each "
        "event, plus a snapshot of nearby players only at shots."
    )
    phase = st.radio("Phase", list(PHASE_NAMES), format_func=PHASE_NAMES.get, horizontal=True)

    positions = load_processed(AVERAGE_POSITIONS_FILE)
    windows = load_processed(FORMATION_WINDOWS_FILE)
    by_match = load_processed(FORMATION_BY_MATCH_FILE)

    shape = positions[(positions["phase"] == phase) & positions["in_shape"]].copy()
    shape["surname"] = shape["player_name"].str.split().str[-1]
    w = windows[windows["phase"] == phase].set_index("window")
    panels = {
        f"{WINDOW_NAMES[k]}: length {w.loc[k, 'team_length_m']:.0f} m, "
        f"width {w.loc[k, 'team_width_m']:.0f} m": g
        for k, g in shape.groupby("window", sort=False)
    }
    fig = pp.plot_formation_comparison(panels, label_col="surname")
    st.pyplot(fig)
    plt.close(fig)

    metric = st.selectbox(
        "Per-match trend",
        ["block_height_m", "team_length_m", "team_width_m"],
        format_func={"block_height_m": "How high the shape sits (m from halfway)",
                     "team_length_m": "Team length (m)",
                     "team_width_m": "Team width (m)"}.get,
    )
    fig = pp.plot_season_trend(
        by_match[by_match["phase"] == phase], metric, window_breaks=[11.5, 22.5],
        window_labels=list(WINDOW_NAMES.values()),
    )
    st.pyplot(fig)
    plt.close(fig)
    st.info(
        "Read the spread, not just the means: single matches vary far more than the "
        "differences between windows, and the playoffs are only 3 matches. In this data, "
        "none of the early-vs-late differences is robust once you account for that "
        "(see README, Findings)."
    )
    st.image(str(STATSBOMB_LOGO), width=180)
    st.caption("Data: StatsBomb open data, 2023 NWSL.")


def main() -> None:
    cfg = load_config()

    st.title("The Entropy of Transition")
    st.caption(
        "How a defense's shape changes during a counter-attack, treated the way a "
        "physicist treats a system of interacting particles."
    )

    st.warning(
        "Demo uses public sample data (Metrica Sports open tracking data; "
        "StatsBomb open 2023 NWSL data), not any club's proprietary "
        "tracking/GPS data. See README.md for details.",
        icon="ℹ️",
    )

    tab_tracking, tab_formation = st.tabs(
        ["Counter-attacks (tracking data)", "Season shape comparison (event data)"]
    )
    with tab_tracking:
        tracking_tab(cfg)
    with tab_formation:
        formation_tab(cfg)


main()
