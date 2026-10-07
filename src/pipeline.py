"""End-to-end orchestration: load -> clean -> compute features -> save.

Run as `python -m src.pipeline`. Intended to be the single command that
regenerates everything in data/processed/ from scratch (given
data/external/ already populated via scripts/download_data.sh); useful
both for development and as proof, if asked, that nothing here is hand-
massaged or cherry-picked after the fact.
"""
from __future__ import annotations

import logging
from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data import metrica_loader, statsbomb_loader
from src.data.cleaning import (
    attack_direction_by_period,
    infer_goalkeeper,
    metrica_to_meters,
    normalize_attack_direction,
    statsbomb_to_meters,
    validate_coordinates,
)
from src.data.sequences import find_control_recoveries, find_counter_attacks
from src.features import baseline, entropy, formation, pitch_control, voronoi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEAMS = ("Home", "Away")
SEQUENCES_FILE = "metrica_counter_attacks.csv"
TIMESERIES_FILE = "metrica_sequence_timeseries.csv"
POSITIONS_FILE = "metrica_sequence_positions.csv"
CONTROLS_FILE = "metrica_control_recoveries.csv"
CONTROL_TIMESERIES_FILE = "metrica_control_timeseries.csv"
BASELINE_MATCHED_FILE = "metrica_baseline_matched.csv"
BASELINE_PERCENTILES_FILE = "metrica_baseline_percentiles.csv"
BASELINE_TESTS_FILE = "metrica_baseline_tests.csv"
WINDOW_CHANGES_FILE = "metrica_window_end_changes.csv"
FORMATION_WINDOWS_FILE = "statsbomb_formation_windows.csv"
FORMATION_BY_MATCH_FILE = "statsbomb_formation_by_match.csv"
AVERAGE_POSITIONS_FILE = "statsbomb_average_positions.csv"
WINDOW_TESTS_FILE = "statsbomb_window_tests.csv"
WINDOW_TEST_METRICS = ("block_height_m", "team_length_m", "team_width_m")


def prepare_metrica_game(
    game_id: int, tracking_margin_m: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[int, int]]]:
    """Load one Metrica game and convert tracking + events to standard meters.

    Returns
    -------
    tracking, events : pd.DataFrame
        In standard meter coordinates, raw (un-flipped) orientation.
        Tracking also has per-player velocities vx, vy in m/s
        (pitch_control.compute_velocities, over the whole match so that
        any frame's velocity is the same however the match is sliced).
    directions : dict[str, dict[int, int]]
        {team: {period: +1/-1}} from cleaning.attack_direction_by_period.
    """
    tracking = metrica_to_meters(metrica_loader.load_match_tracking(game_id), "x", "y")
    validate_coordinates(tracking, "x", "y", margin_m=tracking_margin_m)
    tracking = pitch_control.compute_velocities(tracking, metrica_loader.FRAME_RATE_HZ)
    events = metrica_loader.load_events(game_id)
    events = metrica_to_meters(events, "start_x", "start_y")
    events = metrica_to_meters(events, "end_x", "end_y")
    directions = {team: attack_direction_by_period(tracking, team) for team in TEAMS}
    return tracking, events, directions


def defending_goalkeepers(
    tracking: pd.DataFrame, directions: dict[str, dict[int, int]]
) -> dict[str, str]:
    """{team: goalkeeper player_id}, inferred from the whole match's tracking."""
    return {
        team: infer_goalkeeper(
            normalize_attack_direction(tracking, "x", "y", directions[team]), team
        )
        for team in TEAMS
    }


def analyze_sequence(
    tracking: pd.DataFrame,
    sequence: pd.Series,
    directions: dict[str, dict[int, int]],
    pre_frames: int,
    goalkeeper: str | None = None,
    frames: Collection[int] | None = None,
    pitch_control_frames: Collection[int] | None = None,
) -> pd.DataFrame:
    """Defensive-shape and territory time series for one counter-attack.

    Tracking is direction-normalized for the *defending* team (so its
    along-axis variance is along the direction it attacks), its goalkeeper
    is excluded from the shape metrics, and the frame window starts
    `pre_frames` before the recovery to give a baseline.

    `goalkeeper` is the defending team's goalkeeper; if None it is inferred
    from the whole match (cleaning.infer_goalkeeper). Pass it when analyzing
    many windows from one match; inference scans every frame.

    `frames`, if given, restricts the output to those frames (within the
    window). Use with horizon_frames when only a few time points are
    needed: every metric is computed per frame, so the values are identical
    to the same rows of the full series.

    `pitch_control_frames` are the output frames at which pitch control is
    computed (it costs ~30 ms a frame, ten times the other metrics); None
    means every output frame, and other rows get NaN. Pitch control needs
    vx, vy columns in `tracking` (prepare_metrica_game adds them). The team
    in possession is the non-defending team.

    Returns
    -------
    pd.DataFrame
        One row per frame: frame, time_s, t_rel_s (seconds relative to the
        recovery), n_players, rms_distance, along_axis_variance,
        perpendicular_variance, grid_entropy, defending_area_m2,
        defending_fraction, ball_x_m (ball's x in the defending team's
        attacking frame, so its own goal line is at -52.5; NaN when the
        ball isn't tracked), pitch_control_fraction and own_third_control
        (pitch_control.summarize_surface, for the defending team).
    """
    defending = sequence["defending_team"]
    attacking = next(team for team in TEAMS if team != defending)
    if goalkeeper is None:
        goalkeeper = defending_goalkeepers(tracking, directions)[defending]
    start = max(int(sequence["start_frame"]) - pre_frames, int(tracking["frame"].min()))
    end = int(sequence["end_frame"])
    window = tracking[tracking["frame"].between(start, end)]
    if frames is not None:
        window = window[window["frame"].isin(frames)]
    norm = normalize_attack_direction(window, "x", "y", directions[defending])

    shape = entropy.compute_entropy_timeseries(
        norm, defending, start, end, method="dispersion", exclude_players=[goalkeeper]
    )
    grid = entropy.compute_entropy_timeseries(
        norm, defending, start, end, method="grid_entropy", exclude_players=[goalkeeper]
    )
    territory = voronoi.compute_territory_timeseries(norm, start, end)
    territory["defending_area_m2"] = territory[f"{defending.lower()}_area_m2"]
    territory["defending_fraction"] = territory["defending_area_m2"] / (
        territory["home_area_m2"] + territory["away_area_m2"]
    )
    ball = norm.loc[norm["team"] == metrica_loader.BALL_TEAM, ["frame", "x"]]
    result = (
        shape.merge(grid[["frame", "grid_entropy"]], on="frame")
        .merge(territory[["frame", "defending_area_m2", "defending_fraction"]], on="frame")
        .merge(ball.rename(columns={"x": "ball_x_m"}), on="frame", how="left")
    )

    control_cols = ["pitch_control_fraction", "own_third_control"]
    pc_frames = result["frame"] if pitch_control_frames is None else (
        result.loc[result["frame"].isin(pitch_control_frames), "frame"])
    if len(pc_frames):
        if not {"vx", "vy"} <= set(tracking.columns):
            raise ValueError("Pitch control needs vx, vy (pitch_control.compute_velocities).")
        moving = normalize_attack_direction(
            norm[norm["frame"].isin(pc_frames)], "vx", "vy", directions[defending]
        )
        control = pitch_control.compute_pitch_control_timeseries(
            moving, attacking, start, end, defending_goalkeeper=goalkeeper
        )
        result = result.merge(control[["frame", *control_cols]], on="frame", how="left")
    else:
        result[control_cols] = float("nan")
    result.insert(2, "t_rel_s", result["time_s"] - float(sequence["start_time_s"]))
    result.attrs["goalkeeper"] = goalkeeper
    return result


def horizon_frames(
    frame_times: pd.Series,
    window: pd.Series,
    horizons_s: Collection[float],
    tol_s: float = 1e-9,
) -> list[int]:
    """The frames baseline.shape_change reads, for a set of horizons.

    Parameters
    ----------
    frame_times : pd.Series
        time_s indexed by frame, for the whole match (sorted by frame).
    window : pd.Series
        A counter or control row: start_frame, start_time_s, end_frame.
    horizons_s : collection of float
        Seconds after the recovery. For each one, the last frame at or
        before it is returned (as shape_change(series, horizon_s) would
        pick); the window's last frame is always included too.

    Returns
    -------
    list[int]
        Sorted frames: the first frame at or after the recovery, the
        horizon frames, and the window's last frame.
    """
    times = frame_times.loc[int(window["start_frame"]):int(window["end_frame"])]
    t_rel = times - float(window["start_time_s"])
    after = t_rel[t_rel >= -tol_s]
    frames = {int(after.index[0]), int(after.index[-1])}
    for h in horizons_s:
        upto = after[after <= h + tol_s]
        if not upto.empty:
            frames.add(int(upto.index[-1]))
    return sorted(frames)


def sequence_positions(
    tracking: pd.DataFrame, sequence: pd.Series, pre_frames: int
) -> pd.DataFrame:
    """All players' + ball positions over a sequence's analysis window.

    Raw (un-flipped) standard-meter orientation, NaN rows (players not on
    the pitch) dropped; vx, vy are kept if present (for pitch-control
    figures). Saved so the app and animations can draw sequences
    without reloading full-match tracking.
    """
    start = max(int(sequence["start_frame"]) - pre_frames, int(tracking["frame"].min()))
    window = tracking[tracking["frame"].between(start, int(sequence["end_frame"]))]
    columns = ["frame", "time_s", "team", "player_id", "x", "y"]
    columns += [c for c in ("vx", "vy") if c in window.columns]
    return window.dropna(subset=["x", "y"])[columns].reset_index(drop=True)


def frame_time_index(tracking: pd.DataFrame) -> pd.Series:
    """time_s indexed by frame, sorted (the input horizon_frames expects)."""
    return (tracking[["frame", "time_s"]].drop_duplicates("frame")
            .set_index("frame")["time_s"].sort_index())


def window_frames(windows: pd.DataFrame, pre_frames: int = 0) -> np.ndarray:
    """Every frame inside any window [start_frame - pre_frames, end_frame]."""
    if windows.empty:
        return np.array([], dtype=int)
    return np.unique(np.concatenate([
        np.arange(start - pre_frames, end + 1)
        for start, end in zip(windows["start_frame"], windows["end_frame"])
    ]))


def counter_pitch_control_frames(
    frame_times: pd.Series, sequence: pd.Series, pre_frames: int, step: int
) -> list[int]:
    """Frames of a counter's series that get pitch control: every `step`-th
    frame from the recovery (both directions), plus the frames
    baseline.shape_change reads (horizon_frames)."""
    start, end = int(sequence["start_frame"]), int(sequence["end_frame"])
    sampled = range(start - (pre_frames // step) * step, end + 1, step)
    return sorted(set(sampled) | set(horizon_frames(frame_times, sequence, [])))


def run_metrica_pipeline(cfg: dict) -> None:
    """Run the tracking-based entropy/Voronoi analysis across all Metrica games.

    For each game: load + convert, find rule-based counter-attacks and
    control recoveries (src/data/sequences.py), and compute each
    counter-attack's full defensive-shape, territory and pitch-control time
    series. Pitch control is computed every
    features.pitch_control.counter_frame_step frames (and at the recovery
    and the shot).

    Control windows are only ever read at a few times (the recovery, each
    counter duration they are matched at, and their end; see
    features/baseline.py), so only those frames are computed
    (horizon_frames). Every metric is per-frame, so the values are the same
    as in a full series. This needs every counter's duration, so each
    game's control-window tracking is kept until all games are scanned.

    Writes to data/processed/:
      - SEQUENCES_FILE, TIMESERIES_FILE, POSITIONS_FILE: the counter-attacks,
        their per-frame metrics and per-frame positions, joined on
        (game_id, sequence_id);
      - CONTROLS_FILE, CONTROL_TIMESERIES_FILE: the control windows and
        their metrics at the frames above, joined on (game_id, control_id);
      - the baseline comparison (run_baseline_comparison).
    """
    logger.info("Running Metrica tracking-based pipeline...")
    seq_cfg = cfg["features"]["sequences"]
    rules = {k: seq_cfg[k] for k in ("min_seconds", "max_seconds", "max_recovery_x_m")}
    pre_frames = round(seq_cfg["pre_seconds"] * metrica_loader.FRAME_RATE_HZ)
    pc_step = cfg["features"]["pitch_control"]["counter_frame_step"]
    out_dir = resolve_path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    available = metrica_loader.list_available_games()
    tables: dict[str, list[pd.DataFrame]] = {
        name: [] for name in ("sequences", "series", "positions", "controls", "control_series")
    }
    held = {}  # game_id -> (control-window tracking, frame times, directions, goalkeepers)
    for game_id in cfg["metrica"]["game_ids"]:
        if game_id not in available:
            raise FileNotFoundError(
                f"Metrica game {game_id} not found; run scripts/download_data.sh."
            )
        logger.info("Game %d: loading...", game_id)
        tracking, events, directions = prepare_metrica_game(
            game_id, cfg["cleaning"]["tracking_margin_m"]
        )
        goalkeepers = defending_goalkeepers(tracking, directions)
        frame_times = frame_time_index(tracking)

        sequences = find_counter_attacks(events, directions, **rules)
        sequences.insert(0, "game_id", game_id)
        sequences.insert(1, "sequence_id", range(1, len(sequences) + 1))
        sequences["defending_goalkeeper"] = sequences["defending_team"].map(goalkeepers)
        for _, seq in sequences.iterrows():
            series = analyze_sequence(
                tracking, seq, directions, pre_frames, goalkeepers[seq["defending_team"]],
                pitch_control_frames=counter_pitch_control_frames(
                    frame_times, seq, pre_frames, pc_step),
            )
            series.insert(0, "game_id", game_id)
            series.insert(1, "sequence_id", seq["sequence_id"])
            tables["series"].append(series)
            positions = sequence_positions(tracking, seq, pre_frames)
            positions.insert(0, "game_id", game_id)
            positions.insert(1, "sequence_id", seq["sequence_id"])
            tables["positions"].append(positions)
        tables["sequences"].append(sequences)

        controls = find_control_recoveries(
            events, directions, frame_rate_hz=metrica_loader.FRAME_RATE_HZ, **rules
        )
        controls.insert(0, "game_id", game_id)
        controls.insert(1, "control_id", range(1, len(controls) + 1))
        tables["controls"].append(controls)
        logger.info("Game %d: %d counter-attacks, %d control recoveries",
                    game_id, len(sequences), len(controls))
        in_controls = tracking[tracking["frame"].isin(window_frames(controls))]
        held[game_id] = (in_controls, frame_times, directions, goalkeepers)
        del tracking

    durations = pd.concat(tables["sequences"])["duration_s"].tolist()
    for controls in tables["controls"]:
        if controls.empty:
            continue
        game_id = int(controls["game_id"].iloc[0])
        tracking, frame_times, directions, goalkeepers = held.pop(game_id)
        logger.info("Game %d: analyzing %d control recoveries...", game_id, len(controls))
        for _, ctl in controls.iterrows():
            series = analyze_sequence(
                tracking, ctl, directions, 0, goalkeepers[ctl["defending_team"]],
                frames=horizon_frames(frame_times, ctl, durations),
            )
            series.insert(0, "game_id", game_id)
            series.insert(1, "control_id", ctl["control_id"])
            tables["control_series"].append(series)

    combined = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    for name, file in (("sequences", SEQUENCES_FILE), ("series", TIMESERIES_FILE),
                       ("positions", POSITIONS_FILE), ("controls", CONTROLS_FILE),
                       ("control_series", CONTROL_TIMESERIES_FILE)):
        combined[name].to_csv(out_dir / file, index=False)
    logger.info("Wrote %s, %s, %s, %s and %s to %s", SEQUENCES_FILE, TIMESERIES_FILE,
                POSITIONS_FILE, CONTROLS_FILE, CONTROL_TIMESERIES_FILE, out_dir)
    run_baseline_comparison(
        combined["sequences"], combined["series"], combined["controls"],
        combined["control_series"], cfg["features"]["baseline"], out_dir,
    )


def baseline_control_sets(
    matched: pd.DataFrame, ball_x_max_m: float, depth_caliper_m: float
) -> dict[str, pd.DataFrame]:
    """The three control sets the baseline is reported against.

    - "all": every duration-matched control.
    - "ball_in_defending_third": only controls whose ball, at the matched
      time, is at or behind `ball_x_max_m` in the defending team's frame.
    - "ball_depth_matched": only controls whose ball, at the matched time,
      is within `depth_caliper_m` of the counter's ball at the shot
      (baseline.ball_depth_matched). Counters left with fewer than 2 such
      controls are dropped (baseline.drop_thin_counters).

    The second and third ask whether counters differ from other possessions
    that also got the ball close to goal, or only from possessions that
    didn't. Counter rows are kept in every set.
    """
    near_goal = (matched["role"] == "counter") | (matched["ball_x_m"] <= ball_x_max_m)
    return {
        "all": matched,
        "ball_in_defending_third": matched[near_goal],
        "ball_depth_matched": baseline.drop_thin_counters(
            baseline.ball_depth_matched(matched, depth_caliper_m)
        ),
    }


def baseline_tests(
    control_sets: dict[str, pd.DataFrame], n_permutations: int, seed: int
) -> pd.DataFrame:
    """Permutation test per (control set, metric); one row each.

    `control_sets` as from baseline_control_sets. bonferroni_alpha = 0.05 /
    (number of tests in the table). A set with no testable counter (every
    counter dropped for having < 2 controls) gets NaN rows, not an error.
    """
    rows = []
    for name, subset in control_sets.items():
        n_ctl = subset[subset["role"] == "control"].groupby(list(baseline.COUNTER_KEY)).size()
        if n_ctl.empty:
            rows += [{"control_set": name, "metric": metric, "n_counters": 0}
                     for metric in baseline.CHANGE_METRICS]
            continue
        for metric in baseline.CHANGE_METRICS:
            result = baseline.matched_percentile_test(subset, metric, n_permutations, seed)
            rows.append({"control_set": name, "metric": metric,
                         "min_controls_per_counter": int(n_ctl.min()),
                         "median_controls_per_counter": float(n_ctl.median()), **result})
    tests = pd.DataFrame(rows)
    tests["bonferroni_alpha"] = 0.05 / len(tests)
    tests["significant"] = tests["p_value"] < tests["bonferroni_alpha"]
    return tests


def window_end_changes(
    sequences: pd.DataFrame,
    series: pd.DataFrame,
    controls: pd.DataFrame,
    control_series: pd.DataFrame,
) -> pd.DataFrame:
    """Shape change from recovery to window end, one row per counter and control.

    Unlike matched_changes, each window is measured at its own end (the
    shot, or the control's possession end), so this is descriptive: it
    shows how shape change varies with where the ball got to.

    Returns
    -------
    pd.DataFrame
        role ("counter" / "control"), game_id, window_id (sequence_id or
        control_id), duration_s, then the baseline.shape_change fields.
    """
    rows = []
    for role, windows, frames, id_col in (
        ("counter", sequences, series, "sequence_id"),
        ("control", controls, control_series, "control_id"),
    ):
        groups = dict(list(frames.groupby(["game_id", id_col])))
        for w in windows.itertuples(index=False):
            window_id = getattr(w, id_col)
            rows.append({"role": role, "game_id": w.game_id, "window_id": window_id,
                         "duration_s": w.duration_s,
                         **baseline.shape_change(groups[(w.game_id, window_id)])})
    return pd.DataFrame(rows)


def run_baseline_comparison(
    sequences: pd.DataFrame,
    series: pd.DataFrame,
    controls: pd.DataFrame,
    control_series: pd.DataFrame,
    baseline_cfg: dict,
    out_dir: Path,
) -> None:
    """Compare counter-attacks with duration-matched controls (features/baseline.py).

    Writes BASELINE_MATCHED_FILE (every counter and matched-control shape
    change), BASELINE_PERCENTILES_FILE (each counter's percentile among its
    controls, per control set), BASELINE_TESTS_FILE and WINDOW_CHANGES_FILE
    (window_end_changes).
    """
    matched = baseline.matched_changes(sequences, series, controls, control_series)
    control_sets = baseline_control_sets(
        matched, baseline_cfg["ball_x_max_m"], baseline_cfg["depth_caliper_m"]
    )
    percentiles = pd.concat([
        baseline.matched_percentiles(subset).assign(control_set=name)
        for name, subset in control_sets.items()
    ], ignore_index=True)
    tests = baseline_tests(control_sets, baseline_cfg["n_permutations"], baseline_cfg["seed"])
    matched.to_csv(out_dir / BASELINE_MATCHED_FILE, index=False)
    percentiles.to_csv(out_dir / BASELINE_PERCENTILES_FILE, index=False)
    tests.to_csv(out_dir / BASELINE_TESTS_FILE, index=False)
    window_end_changes(sequences, series, controls, control_series).to_csv(
        out_dir / WINDOW_CHANGES_FILE, index=False
    )
    logger.info("Wrote %s, %s, %s and %s to %s", BASELINE_MATCHED_FILE,
                BASELINE_PERCENTILES_FILE, BASELINE_TESTS_FILE, WINDOW_CHANGES_FILE, out_dir)


def window_difference_tests(
    by_match: pd.DataFrame, windows: dict[str, list[int]], metrics: tuple[str, ...]
) -> pd.DataFrame:
    """Welch t-test for every (phase, metric, window pair); one row each.

    bonferroni_alpha = 0.05 / (number of tests in the table), so a reader
    can see at a glance which differences survive multiple-comparison
    correction (significant column).
    """
    labels = list(windows)
    rows = []
    for phase, phase_rows in by_match.groupby("phase", sort=False):
        for metric in metrics:
            for i, a in enumerate(labels):
                for b in labels[i + 1:]:
                    result = formation.window_difference_test(
                        phase_rows, metric, windows[a], windows[b]
                    )
                    rows.append({"phase": phase, "metric": metric, "window_a": a,
                                 "window_b": b, **result})
    tests = pd.DataFrame(rows)
    tests["bonferroni_alpha"] = 0.05 / len(tests)
    tests["significant"] = tests["p_value"] < tests["bonferroni_alpha"]
    return tests


def run_statsbomb_pipeline(cfg: dict) -> None:
    """Run the aggregate formation-compactness analysis on 2023 NWSL data.

    For the configured team (Gotham FC): load all its matches, convert to
    standard meters, and for each configured phase write
      - FORMATION_WINDOWS_FILE: compactness per match-week window
        (early/late regular season, playoffs),
      - FORMATION_BY_MATCH_FILE: compactness per match (season trend),
      - AVERAGE_POSITIONS_FILE: per-player average positions per window,
        for plotting the shapes themselves,
      - WINDOW_TESTS_FILE: Welch t-tests of per-match metrics between every
        pair of windows, with a Bonferroni threshold over all tests run.
    These are rolling/window aggregates over events, never per-instant
    team shapes (METHODOLOGY.md, principle 3).
    """
    logger.info("Running StatsBomb formation-aggregate pipeline...")
    fcfg = cfg["features"]["formation"]
    team = fcfg["team"]
    matches = statsbomb_loader.get_team_matches(team)
    events = pd.concat(
        [statsbomb_loader.get_match_events(m) for m in matches["match_id"]], ignore_index=True
    )
    events = statsbomb_to_meters(events, "x", "y")
    validate_coordinates(events.dropna(subset=["x", "y"]), "x", "y")
    logger.info("%s: %d matches, %d events", team, len(matches), len(events))

    windows = {
        label: matches.loc[matches["match_week"].between(lo, hi), "match_id"].tolist()
        for label, (lo, hi) in fcfg["windows"].items()
    }
    window_rows, match_rows, position_rows = [], [], []
    for phase in fcfg["phases"]:
        window_rows.append(
            formation.compare_windows(
                events, team, windows, phase, fcfg["min_touches_window"], fcfg["n_outfield"]
            ).assign(phase=phase)
        )
        match_rows.append(
            formation.compactness_by_match(
                events, team, phase, fcfg["min_touches_match"], fcfg["n_outfield"]
            ).assign(phase=phase)
        )
        for label, match_ids in windows.items():
            avg = formation.compute_average_positions(
                events[events["match_id"].isin(match_ids)], team, phase,
                fcfg["min_touches_window"],
            )
            shape_ids = set(formation.select_outfield_shape(avg, fcfg["n_outfield"])["player_id"])
            position_rows.append(
                avg.assign(window=label, phase=phase, in_shape=avg["player_id"].isin(shape_ids))
            )

    out_dir = resolve_path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(window_rows, ignore_index=True).to_csv(out_dir / FORMATION_WINDOWS_FILE, index=False)
    by_match = pd.concat(match_rows, ignore_index=True).merge(
        matches[["match_id", "match_date", "match_week", "competition_stage",
                 "home_team", "away_team", "home_score", "away_score"]],
        on="match_id",
    )
    by_match.to_csv(out_dir / FORMATION_BY_MATCH_FILE, index=False)
    window_tests = window_difference_tests(by_match, windows, WINDOW_TEST_METRICS)
    window_tests.to_csv(out_dir / WINDOW_TESTS_FILE, index=False)
    pd.concat(position_rows, ignore_index=True).to_csv(
        out_dir / AVERAGE_POSITIONS_FILE, index=False
    )
    logger.info("Wrote %s, %s, %s and %s to %s", FORMATION_WINDOWS_FILE,
                FORMATION_BY_MATCH_FILE, AVERAGE_POSITIONS_FILE, WINDOW_TESTS_FILE, out_dir)


def main() -> None:
    cfg = load_config()
    run_metrica_pipeline(cfg)
    run_statsbomb_pipeline(cfg)
    logger.info("Pipeline complete. See data/processed/ for output.")


if __name__ == "__main__":
    main()
