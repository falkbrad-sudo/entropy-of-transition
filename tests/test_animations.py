"""Smoke test for src/viz/animations.py: renders a tiny synthetic GIF."""
import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from src.viz.animations import animate_sequence  # noqa: E402


def _tracking() -> pd.DataFrame:
    rows = []
    for frame in range(1, 7):
        rows += [
            {"frame": frame, "team": "Home", "player_id": "H", "x": -10.0 + frame, "y": 0.0},
            {"frame": frame, "team": "Away", "player_id": "A", "x": 10.0, "y": float(frame)},
        ]
        if frame != 3:  # ball missing in one frame must not break rendering
            rows.append({"frame": frame, "team": "Ball", "player_id": "ball", "x": 0.0, "y": 0.0})
    return pd.DataFrame(rows)


def test_gif_has_one_image_per_rendered_frame(tmp_path):
    series = pd.DataFrame({"frame": range(1, 7), "time_s": [f * 0.04 for f in range(1, 7)],
                           "along_axis_variance": range(6), "perpendicular_variance": range(6)})
    out = tmp_path / "seq.gif"
    animate_sequence(_tracking(), 1, 6, series, str(out), frame_step=2)
    with Image.open(out) as gif:
        assert gif.n_frames == 3  # frames 1, 3, 5


def test_empty_range_raises(tmp_path):
    with pytest.raises(ValueError):
        animate_sequence(_tracking(), 100, 200, output_path=str(tmp_path / "x.gif"))
