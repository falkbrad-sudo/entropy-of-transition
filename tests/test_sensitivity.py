"""Tests for src/sensitivity.py."""
import pandas as pd
import pytest

from src.sensitivity import (
    SENSITIVITY_TESTS_FILE,
    variant_rules,
    window_horizons,
)

SEQ_CFG = {"min_seconds": 3, "max_seconds": 15, "max_recovery_x_m": 0, "pre_seconds": 2}


def test_variant_rules_fill_defaults_and_reject_unknown_keys():
    rules = variant_rules(SEQ_CFG, {"default": {}, "max_20s": {"max_seconds": 20}})
    assert rules["default"] == {"min_seconds": 3, "max_seconds": 15, "max_recovery_x_m": 0}
    assert rules["max_20s"] == {"min_seconds": 3, "max_seconds": 20, "max_recovery_x_m": 0}
    with pytest.raises(ValueError, match="pre_seconds"):
        variant_rules(SEQ_CFG, {"bad": {"pre_seconds": 1}})


def test_window_horizons_union_over_variants_and_duration_limit():
    windows = pd.DataFrame([
        # Variant a: counters of 5 s and 12 s. Variant b: a counter of 8 s.
        {"variant": "a", "role": "counter", "uid": 0, "duration_s": 5.0},
        {"variant": "a", "role": "counter", "uid": 1, "duration_s": 12.0},
        {"variant": "b", "role": "counter", "uid": 2, "duration_s": 8.0},
        # Control 10 lasts 10 s and is in both variants: matched at 5 (a) and 8 (b), not 12.
        {"variant": "a", "role": "control", "uid": 10, "duration_s": 10.0},
        {"variant": "b", "role": "control", "uid": 10, "duration_s": 10.0},
        # Control 11 (variant b only) lasts 4 s: too short for every counter in b.
        {"variant": "b", "role": "control", "uid": 11, "duration_s": 4.0},
    ])
    assert window_horizons(windows) == {10: [5.0, 8.0], 11: []}


@pytest.mark.integration
def test_default_variant_reproduces_pipeline_baseline():
    """Sparse-frame sensitivity run vs. the full-series pipeline (same seed)."""
    from src.config import load_config, resolve_path
    from src.pipeline import BASELINE_TESTS_FILE

    processed = resolve_path(load_config()["paths"]["data_processed"])
    if not (processed / SENSITIVITY_TESTS_FILE).exists():
        pytest.skip("Run `python -m src.pipeline` and `python -m src.sensitivity` first.")
    pipeline = pd.read_csv(processed / BASELINE_TESTS_FILE)
    sensitivity = pd.read_csv(processed / SENSITIVITY_TESTS_FILE)
    default = sensitivity[sensitivity["variant"] == "default"].reset_index(drop=True)
    cols = ["control_set", "metric", "n_counters", "mean_percentile", "p_value"]
    pd.testing.assert_frame_equal(default[cols], pipeline[cols])
