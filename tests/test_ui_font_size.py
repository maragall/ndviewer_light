"""
Tests for IMA-179: larger, configurable viewer UI font sizes.

Covers the two failure modes found in the eng review:
- NDV_SLIDER_STYLE losing its slider chrome in the f-string conversion
- T/FOV labels clipping (they use a shared setFixedWidth for alignment;
  push-API FOV labels arrive after setup, so the width must re-sync)

No ndv import anywhere — these must run in environments without ndv.
Qt-widget tests follow the QApplication precedent in test_3d_visualization.py
and the bind-real-method-to-mock pattern in test_refresh.py.
"""

import os
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ndviewer_light import (  # noqa: E402
    NDV_SLIDER_STYLE,
    SLIDER_VALUE_FONT_SIZE_PX,
    UI_FONT_SIZE_PX,
    LightweightViewer,
)


@pytest.fixture(scope="module")
def qapp():
    """Session QApplication (offscreen) for font-metrics and QLabel tests."""
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _make_viewer_stub(fov_labels=(), time_max=0, fov_max=0):
    """Mock viewer carrying real QLabels, with the real width-sync method bound."""
    from PyQt5.QtWidgets import QLabel

    stub = MagicMock(spec=LightweightViewer)
    stub._time_label = QLabel("T")
    stub._fov_label = QLabel("FOV")
    stub._fov_labels = list(fov_labels)
    stub._time_slider = MagicMock()
    stub._time_slider.maximum.return_value = time_max
    stub._fov_slider = MagicMock()
    stub._fov_slider.maximum.return_value = fov_max
    stub._sync_slider_label_widths = (
        lambda *args: LightweightViewer._sync_slider_label_widths(stub)
    )
    return stub


def _advance(text: str) -> int:
    """Pixel width of `text` at the configured UI font size."""
    from PyQt5.QtGui import QFont, QFontMetrics
    from PyQt5.QtWidgets import QLabel

    font = QFont(QLabel().font())
    font.setPixelSize(UI_FONT_SIZE_PX)
    return QFontMetrics(font).horizontalAdvance(text)


class TestStyleConstants:
    def test_style_contains_configured_font_sizes(self):
        """The constants must actually drive the applied stylesheet."""
        assert f"font-size: {UI_FONT_SIZE_PX}px" in NDV_SLIDER_STYLE
        assert f"font-size: {SLIDER_VALUE_FONT_SIZE_PX}px" in NDV_SLIDER_STYLE

    def test_regression_slider_chrome_rules_retained(self):
        """CRITICAL: f-string conversion must not drop the slider chrome."""
        for rule in (
            "QSlider::groove:horizontal",
            "QSlider::handle:horizontal",
            "QSlider::sub-page:horizontal",
        ):
            assert rule in NDV_SLIDER_STYLE

    def test_exactly_one_font_size_source(self):
        """Only the two configured rules — no stray hardcoded font-size."""
        assert NDV_SLIDER_STYLE.count("font-size") == 2


class TestLabelWidthSync:
    def test_width_fits_widest_fov_label(self, qapp):
        """Width must fit long push-API region labels, not just 'FOV: 999'."""
        stub = _make_viewer_stub(fov_labels=["A1:0", "cell_0_0:3"])
        stub._sync_slider_label_widths()
        needed = _advance("FOV: cell_0_0:3")
        assert stub._fov_label.minimumWidth() >= needed

    def test_width_fits_time_slider_maximum(self, qapp):
        """Width must fit the largest time index the slider can show."""
        stub = _make_viewer_stub(time_max=99999)
        stub._sync_slider_label_widths()
        assert stub._time_label.minimumWidth() >= _advance("T: 99999")

    def test_labels_get_equal_width(self, qapp):
        """T and FOV labels share one width so the sliders stay aligned."""
        stub = _make_viewer_stub(fov_labels=["region_10:12"], time_max=3)
        stub._sync_slider_label_widths()
        assert stub._time_label.minimumWidth() == stub._fov_label.minimumWidth()
        # setFixedWidth pins min == max
        assert stub._time_label.minimumWidth() == stub._time_label.maximumWidth()

    def test_width_resyncs_when_fov_labels_arrive_after_setup(self, qapp):
        """Push-API path: labels set post-setup must widen the labels."""
        stub = _make_viewer_stub()
        stub._sync_slider_label_widths()
        width_before = stub._fov_label.minimumWidth()

        stub._fov_labels = ["a_very_long_region_name_0:12"]
        stub._sync_slider_label_widths()

        assert stub._fov_label.minimumWidth() > width_before
        assert stub._fov_label.minimumWidth() >= _advance(
            "FOV: a_very_long_region_name_0:12"
        )

    def test_width_never_below_floor(self, qapp):
        """Empty dataset keeps the 'FOV: 999' floor for a stable layout."""
        stub = _make_viewer_stub()
        stub._sync_slider_label_widths()
        assert stub._fov_label.minimumWidth() >= _advance("FOV: 999")
