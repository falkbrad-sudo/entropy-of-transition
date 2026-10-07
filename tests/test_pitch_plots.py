"""Smoke tests for src/viz/pitch_plots.py (rendering, not visual correctness)."""
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.viz.pitch_plots import (  # noqa: E402
    _label_box,
    _overlaps,
    add_logo,
    plot_baseline_comparison,
    plot_depth_response,
    plot_entropy_timeseries,
    plot_formation_comparison,
    plot_pitch_control,
    plot_season_trend,
    plot_team_positions,
    plot_voronoi_diagram,
    to_pitch_xy,
)


def _series() -> pd.DataFrame:
    return pd.DataFrame({"time_s": [10.0, 10.04, 10.08], "a": [1.0, 2.0, 3.0],
                         "b": [3.0, 2.0, 1.0]})


def test_plots_one_line_per_metric_on_relative_time():
    fig = plot_entropy_timeseries(_series(), ["a", "b"])
    lines = fig.axes[0].get_lines()
    assert len(lines) == 2
    assert lines[0].get_xdata()[0] == pytest.approx(0.0)


def test_event_markers_are_drawn():
    markers = pd.DataFrame({"time_s": [10.04], "label": ["Shot"]})
    fig = plot_entropy_timeseries(_series(), "a", event_markers=markers)
    assert len(fig.axes[0].get_lines()) == 2  # metric line + marker rule


def test_too_many_metrics_raises():
    df = _series().assign(c=0.0, d=0.0)
    with pytest.raises(ValueError):
        plot_entropy_timeseries(df, ["a", "b", "c", "d"])


def test_to_pitch_xy_maps_centre_and_corners():
    px, py = to_pitch_xy([0.0, -52.5, 52.5], [0.0, -34.0, 34.0])
    assert px.tolist() == pytest.approx([52.5, 0.0, 105.0])
    # Standard +y is the bottom touchline -> mplsoccer y = 0 (bottom).
    assert py.tolist() == pytest.approx([34.0, 68.0, 0.0])


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "team": ["Home", "Home", "Away", "Away", "Ball"],
        "player_id": ["H1", "H2", "A1", "A2", "ball"],
        "x": [-30.0, -10.0, 10.0, 30.0, 0.0],
        "y": [-10.0, 10.0, -10.0, 10.0, 0.0],
    })


def test_voronoi_diagram_draws_one_cell_per_player():
    ax = plot_voronoi_diagram(_frame())
    assert len(ax.patches) >= 4  # pitch markings may add patches too
    cells = [p for p in ax.patches if p.get_alpha() == pytest.approx(0.22)]
    assert len(cells) == 4  # ball excluded


def test_team_positions_with_labels():
    ax = plot_team_positions(_frame().iloc[:2].assign(name=["A", "B"]), label_col="name")
    assert {t.get_text() for t in ax.texts} >= {"A", "B"}


def test_formation_comparison_makes_one_panel_per_window():
    shape = _frame().iloc[:4]
    fig = plot_formation_comparison({"early": shape, "late": shape})
    assert len(fig.axes) == 2


def test_label_slots_detect_overlap():
    below = _label_box(50.0, 30.0, "Name", ((0.0, -2.2), "center", "top"))
    above = _label_box(50.0, 30.0, "Name", ((0.0, 2.2), "center", "bottom"))
    assert not _overlaps(below, above)
    assert _overlaps(below, below)


def test_season_trend_draws_window_means():
    by_match = pd.DataFrame({"match_week": [1, 2, 3, 4], "m": [1.0, 3.0, 10.0, 20.0]})
    fig = plot_season_trend(by_match, "m", window_breaks=[2.5], window_labels=["a", "b"])
    ax = fig.axes[0]
    means = sorted(seg[0][1] for coll in ax.collections for seg in coll.get_segments())
    assert means == pytest.approx([2.0, 15.0])
    assert any("mean 15.0 (n=2)" in t.get_text() for t in ax.texts)


def test_baseline_comparison_one_panel_per_metric_one_row_per_counter():
    rows = []
    for seq in (1, 2):
        rows.append({"game_id": 1, "sequence_id": seq, "role": "counter", "w": -1.0, "l": 1.0})
        rows += [{"game_id": 1, "sequence_id": seq, "role": "control", "w": v, "l": -v}
                 for v in (-0.5, 0.0, 0.5)]
    fig = plot_baseline_comparison(pd.DataFrame(rows), {"w": "width", "l": "length"},
                                   title="t")
    assert len(fig.axes) == 2
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["Game 1, #1", "Game 1, #2"]
    assert len(fig.axes[0].collections) == 2 * 3  # 2 rows x (controls, median, counter)


def test_depth_response_draws_bin_medians_and_both_roles():
    changes = pd.DataFrame({
        "role": ["control"] * 4 + ["counter"],
        "ball_x_m": [-40.0, -30.0, 10.0, float("nan"), -45.0],
        "w": [-1.0, -0.5, 0.0, 9.0, -1.2],
    })
    fig = plot_depth_response(changes, {"w": "width"}, [-52.5, -17.5, 52.5], third_line_m=-17.5)
    ax = fig.axes[0]
    labels = [h.get_label() for h in ax.get_legend_handles_labels()[0]]
    assert labels == ["Controls (no shot)", "Control median per depth bin", "Counter-attacks"]
    # Two non-empty bins -> two median segments (the NaN-ball control is dropped).
    medians = [c for c in ax.collections if c.get_label() != "Controls (no shot)"
               and c.get_label() != "Counter-attacks"]
    assert len(medians) == 2


def test_sparse_metric_is_drawn_through_its_sampled_frames():
    series = _series().assign(c=[0.5, float("nan"), 0.7])
    line = plot_entropy_timeseries(series, "c").axes[0].get_lines()[0]
    assert line.get_ydata().tolist() == [0.5, 0.7]


def test_pitch_control_draws_surface_players_and_arrows():
    from src.features.pitch_control import compute_pitch_control_surface

    frame = pd.DataFrame({
        "team": ["Home", "Away", "Ball"], "player_id": ["H", "A", "ball"],
        "x": [10.0, -10.0, 0.0], "y": [5.0, -5.0, 0.0],
        "vx": [-3.0, 2.0, np.nan], "vy": [0.0, 1.0, np.nan],
    })
    xgrid, ygrid, surface = compute_pitch_control_surface(frame, "Home", offsides=False)
    ax = plot_pitch_control(xgrid, ygrid, surface, frame, "Away", title="t")
    assert len(ax.images) == 1 and ax.images[0].get_array().shape == surface.shape
    left, right, bottom, top = ax.images[0].get_extent()
    assert (left, right, bottom, top) == pytest.approx((0.0, 105.0, 0.0, 68.0))
    assert len(ax.collections) >= 3  # quivers + player scatters + ball


def test_end_labels_of_close_lines_are_spread_apart():
    series = pd.DataFrame({"time_s": [0.0, 1.0], "a": [1.0, 2.0], "b": [1.0, 2.001]})
    ax = plot_entropy_timeseries(series, ["a", "b"]).axes[0]
    offsets = sorted(t.xyann[1] for t in ax.texts)
    # Line ends are a fraction of a point apart; the upper label is pushed up
    # to make a 12 pt gap (minus that fraction).
    assert offsets[0] == 0.0 and 11.0 < offsets[1] <= 12.0


def test_logo_keeps_its_aspect_and_fixed_height_in_the_corner():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 4))
    logo = np.ones((50, 200, 4))  # 4:1 image
    ax = add_logo(fig, logo, height_in=0.25, margin_in=0.1)
    x0, y0, w, h = ax.get_position().bounds
    assert (x0 * 10, y0 * 4) == pytest.approx((0.1, 0.1))
    assert (w * 10, h * 4) == pytest.approx((1.0, 0.25))  # 0.25 in tall -> 1 in wide
    assert not ax.axison
    plt.close(fig)
