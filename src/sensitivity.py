"""Robustness of the counter-attack findings to the detection thresholds.

Run as `python -m src.sensitivity` (after scripts/download_data.sh; it reads
raw tracking, not data/processed/). The counter-attack rule in
src/data/sequences.py has three tunable thresholds (min_seconds,
max_seconds, max_recovery_x_m). This re-runs detection, the per-counter
shape-change counts and the full baseline comparison (pipeline.
baseline_control_sets / baseline_tests) for each variant in config.yaml
(features.sensitivity.variants), so a reader can see which conclusions
depend on the particular thresholds chosen.

Only the frames the comparison reads are computed (pipeline.horizon_frames):
the recovery, each matched horizon and the window end. The metrics are
per-frame, so the values are the same as from the full series. The
"default" variant should therefore reproduce data/processed/
metrica_baseline_tests.csv exactly; tests/test_sensitivity.py checks that.

Writes to data/processed/:
  - SENSITIVITY_COUNTS_FILE: per variant, how many counters / controls were
    found and how many counters narrowed, lengthened, etc.
  - SENSITIVITY_TESTS_FILE: per variant x control set x metric, the
    baseline permutation test (Bonferroni within each variant).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data import metrica_loader
from src.data.sequences import find_control_recoveries, find_counter_attacks
from src.features import baseline
from src.pipeline import (
    analyze_sequence,
    baseline_control_sets,
    baseline_tests,
    defending_goalkeepers,
    frame_time_index,
    horizon_frames,
    prepare_metrica_game,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SENSITIVITY_COUNTS_FILE = "metrica_sensitivity_counts.csv"
SENSITIVITY_TESTS_FILE = "metrica_sensitivity_tests.csv"
RULE_KEYS = ("min_seconds", "max_seconds", "max_recovery_x_m")
WINDOW_KEY = ["game_id", "role", "defending_team", "start_frame", "start_time_s", "end_frame"]
_TOL_S = 1e-9


def variant_rules(seq_cfg: dict, variants: dict[str, dict]) -> dict[str, dict]:
    """{variant: {min_seconds, max_seconds, max_recovery_x_m}}, defaults filled in."""
    defaults = {k: seq_cfg[k] for k in RULE_KEYS}
    unknown = {k for overrides in variants.values() for k in overrides} - set(RULE_KEYS)
    if unknown:
        raise ValueError(f"Unknown sensitivity rule key(s): {sorted(unknown)}")
    return {name: {**defaults, **(overrides or {})} for name, overrides in variants.items()}


def find_windows(
    events: pd.DataFrame, directions: dict[str, dict[int, int]], rules: dict[str, dict]
) -> pd.DataFrame:
    """Counters and controls for every variant, one row per (variant, window)."""
    parts = []
    for name, rule in rules.items():
        counters = find_counter_attacks(events, directions, **rule)
        controls = find_control_recoveries(
            events, directions, frame_rate_hz=metrica_loader.FRAME_RATE_HZ, **rule
        )
        parts += [counters.assign(variant=name, role="counter"),
                  controls.assign(variant=name, role="control")]
    return pd.concat(parts, ignore_index=True)


def window_horizons(windows: pd.DataFrame) -> dict[int, list[float]]:
    """{uid: horizons}; for a control, every counter duration (from any
    variant it belongs to) that it lasts long enough to be matched at."""
    durations = {
        name: g["duration_s"].to_numpy()
        for name, g in windows[windows["role"] == "counter"].groupby("variant")
    }
    horizons: dict[int, set[float]] = {}
    for w in windows[windows["role"] == "control"].itertuples(index=False):
        d = durations.get(w.variant, np.array([]))
        horizons.setdefault(w.uid, set()).update(d[d <= w.duration_s + _TOL_S].tolist())
    return {uid: sorted(h) for uid, h in horizons.items()}


def counter_counts(name: str, rule: dict, counters: pd.DataFrame, controls: pd.DataFrame,
                   series: pd.DataFrame) -> dict:
    """How many counters narrowed / lengthened / lost control etc., recovery to shot."""
    groups = dict(list(series.groupby(list(baseline.COUNTER_KEY))))
    changes = pd.DataFrame([baseline.shape_change(groups[(c.game_id, c.sequence_id)])
                            for c in counters.itertuples(index=False)])
    n = len(counters)
    row = {"variant": name, **rule, "n_counters": n, "n_controls": len(controls)}
    if n == 0:
        return row
    return {
        **row,
        "width_fell": int((changes["width_log2_ratio"] < 0).sum()),
        "length_rose": int((changes["length_log2_ratio"] > 0).sum()),
        "grid_entropy_fell_or_flat": int((changes["grid_entropy_change"] <= 0).sum()),
        "voronoi_share_fell": int((changes["defending_fraction_change"] < 0).sum()),
        "pitch_control_fell": int((changes["pitch_control_change"] < 0).sum()),
        "own_third_control_fell": int((changes["own_third_control_change"] < 0).sum()),
        "median_width_log2_ratio": float(changes["width_log2_ratio"].median()),
        "median_length_log2_ratio": float(changes["length_log2_ratio"].median()),
    }


def run_sensitivity(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detection + baseline for every configured variant; see module docstring.

    Returns
    -------
    counts, tests : pd.DataFrame
        The SENSITIVITY_COUNTS_FILE and SENSITIVITY_TESTS_FILE tables.
    """
    rules = variant_rules(cfg["features"]["sequences"], cfg["features"]["sensitivity"]["variants"])
    margin = cfg["cleaning"]["tracking_margin_m"]
    game_ids = cfg["metrica"]["game_ids"]

    # Pass 1: windows for every variant. Horizons depend on every game's
    # counters, so frames can only be chosen once all games are scanned.
    found = []
    for game_id in game_ids:
        _, events, directions = prepare_metrica_game(game_id, margin)
        found.append(find_windows(events, directions, rules).assign(game_id=game_id))
        logger.info("Game %d: windows found for %d variants", game_id, len(rules))
    windows = pd.concat(found, ignore_index=True)
    windows["uid"] = windows.groupby(WINDOW_KEY, sort=False).ngroup()
    horizons = window_horizons(windows)

    # Pass 2: metrics at the needed frames of each distinct window.
    series_parts = []
    for game_id in game_ids:
        tracking, _, directions = prepare_metrica_game(game_id, margin)
        goalkeepers = defending_goalkeepers(tracking, directions)
        frame_times = frame_time_index(tracking)
        unique = windows[windows["game_id"] == game_id].drop_duplicates("uid")
        logger.info("Game %d: computing %d distinct windows", game_id, len(unique))
        for w in unique.itertuples(index=False):
            row = pd.Series(w._asdict())
            frames = horizon_frames(frame_times, row, horizons.get(w.uid, []))
            series = analyze_sequence(tracking, row, directions, 0,
                                      goalkeepers[w.defending_team], frames=frames)
            series_parts.append(series.assign(game_id=game_id, uid=w.uid))
        del tracking
    all_series = pd.concat(series_parts, ignore_index=True)

    base_cfg = cfg["features"]["baseline"]
    count_rows, test_parts = [], []
    for name, rule in rules.items():
        v = windows[windows["variant"] == name]
        counters = v[v["role"] == "counter"].rename(columns={"uid": "sequence_id"})
        controls = v[v["role"] == "control"].rename(columns={"uid": "control_id"})
        counter_series = all_series[all_series["uid"].isin(counters["sequence_id"])].rename(
            columns={"uid": "sequence_id"})
        control_series = all_series[all_series["uid"].isin(controls["control_id"])].rename(
            columns={"uid": "control_id"})
        count_rows.append(counter_counts(name, rule, counters, controls, counter_series))
        if counters.empty:
            continue
        matched = baseline.matched_changes(counters, counter_series, controls, control_series)
        sets = baseline_control_sets(matched, base_cfg["ball_x_max_m"],
                                     base_cfg["depth_caliper_m"])
        tests = baseline_tests(sets, base_cfg["n_permutations"], base_cfg["seed"])
        test_parts.append(tests.assign(variant=name, **rule))
        logger.info("Variant %s: %d counters, %d controls", name, len(counters), len(controls))

    counts = pd.DataFrame(count_rows)
    tests = pd.concat(test_parts, ignore_index=True)
    out_dir = resolve_path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out_dir / SENSITIVITY_COUNTS_FILE, index=False)
    tests.to_csv(out_dir / SENSITIVITY_TESTS_FILE, index=False)
    logger.info("Wrote %s and %s to %s", SENSITIVITY_COUNTS_FILE, SENSITIVITY_TESTS_FILE, out_dir)
    return counts, tests


def main() -> None:
    run_sensitivity(load_config())


if __name__ == "__main__":
    main()
