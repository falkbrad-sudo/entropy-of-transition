"""Integration test: the Streamlit app runs end to end on data/processed/.

Requires `python -m src.pipeline` to have been run (hence the marker).
"""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def _run() -> AppTest:
    return AppTest.from_file(APP, default_timeout=120).run()


@pytest.mark.integration
def test_app_renders_every_sequence_without_errors():
    at = _run()
    assert not at.exception
    assert len(at.selectbox[0].options) == 3  # curated headlines by default
    at.toggle[0].set_value(True).run()
    for i in range(len(at.selectbox[0].options)):
        at.selectbox[0].select_index(i).run()
        assert not at.exception


@pytest.mark.integration
def test_frame_slider_and_formation_controls():
    at = _run()
    # AppTest lists options as display strings; set the underlying frame number
    # (the game-1 headline's shot frame, the slider's last option).
    at.select_slider[0].set_value(9628).run()
    assert not at.exception
    at.radio[0].set_value("in_possession").run()
    at.selectbox[1].set_value("team_width_m").run()
    assert not at.exception


@pytest.mark.integration
def test_takeaway_is_computed_from_data():
    at = _run()
    takeaway = at.markdown[0].value
    assert "stretched front-to-back by 3.1×" in takeaway  # game 1 headline, 141 -> 445 m²
    assert "narrowed by 2.7×" in takeaway
