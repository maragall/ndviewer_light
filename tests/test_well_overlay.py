"""
Unit tests for the on-image well/region label overlay (IMA-178).

Tests cover:
1. format_well_overlay_label() - overlay text policy (pure function)
2. _update_fov_display() - single push-mode write path (slider text + overlay)
3. _on_ndv_index_changed() - dataset-mode event handler (attrs read at event
   time, int|slice guard, fail-hidden contract)
4. _connect_ndv_index_events() - mode arbitration, disconnect-before-
   resubscribe, initial-paint seed, graceful degradation
5. Offscreen widget tests - overlay survives the widget-swap sequence
   _set_ndv_data uses (stand-in QWidget, no GL required), anchoring,
   show/hide behavior

Structure of the overlay data flow (see the diagram comment next to
LightweightViewer._update_fov_display in ndviewer_light/core.py):
push mode drives the overlay through _update_fov_display; dataset mode
drives it through ndv's current_index value_changed events, which take
precedence while active.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ndviewer_light import (
    LightweightViewer,
    format_well_overlay_label,
)


class TestFormatWellOverlayLabel:
    """Policy tests for the overlay text (pure function)."""

    def test_plate_label_shown(self):
        assert format_well_overlay_label(["A1:0", "A1:1", "B2:0"], 2) == "B2:0"

    def test_region_label_shown(self):
        assert format_well_overlay_label(["regionX:0"], 0) == "regionX:0"

    def test_default_region_hidden_single_fov(self):
        assert format_well_overlay_label(["default:0"], 0) is None

    def test_default_region_hidden_multi_fov(self):
        """'default:3' is a non-well answer to 'which well?' - never rendered."""
        labels = [f"default:{i}" for i in range(5)]
        assert format_well_overlay_label(labels, 3) is None

    def test_empty_labels_hidden(self):
        assert format_well_overlay_label([], 0) is None
        assert format_well_overlay_label(None, 0) is None

    def test_out_of_range_hidden(self):
        assert format_well_overlay_label(["A1:0"], 5) is None
        assert format_well_overlay_label(["A1:0"], -1) is None

    def test_non_int_index_hidden(self):
        """ndv types current_index values as int | slice - guard both."""
        assert format_well_overlay_label(["A1:0"], slice(None)) is None
        assert format_well_overlay_label(["A1:0"], None) is None
        assert format_well_overlay_label(["A1:0"], "0") is None
        assert format_well_overlay_label(["A1:0"], True) is None

    def test_non_string_label_hidden(self):
        assert format_well_overlay_label([None], 0) is None
        assert format_well_overlay_label([""], 0) is None


def _make_push_viewer(fov_labels, current_idx=0, dataset_overlay_active=False):
    """Minimal fake self for exercising real push-mode display methods."""
    fake = SimpleNamespace()
    fake._fov_labels = fov_labels
    fake._current_fov_idx = current_idx
    fake._fov_label = MagicMock()
    fake._dataset_overlay_active = dataset_overlay_active
    fake._well_overlay = MagicMock()
    fake._overlay_texts = []
    fake._set_well_overlay_text = fake._overlay_texts.append
    return fake


class TestUpdateFovDisplay:
    """Push-mode single write path: slider text + overlay together."""

    def test_updates_slider_text_and_overlay(self):
        fake = _make_push_viewer(["A1:0", "A2:0"])
        LightweightViewer._update_fov_display(fake, 1)
        fake._fov_label.setText.assert_called_once_with("FOV: A2:0")
        assert fake._overlay_texts == ["A2:0"]

    def test_defaults_to_current_fov_idx(self):
        fake = _make_push_viewer(["A1:0", "A2:0"], current_idx=1)
        LightweightViewer._update_fov_display(fake)
        fake._fov_label.setText.assert_called_once_with("FOV: A2:0")
        assert fake._overlay_texts == ["A2:0"]

    def test_no_labels_numeric_slider_hidden_overlay(self):
        fake = _make_push_viewer([])
        LightweightViewer._update_fov_display(fake, 5)
        fake._fov_label.setText.assert_called_once_with("FOV: 5")
        assert fake._overlay_texts == [None]

    def test_dash_placeholder_at_acquisition_start(self):
        fake = _make_push_viewer([])
        LightweightViewer._update_fov_display(fake, 0, empty_text="-")
        fake._fov_label.setText.assert_called_once_with("FOV: -")
        assert fake._overlay_texts == [None]

    def test_default_region_hides_overlay_but_keeps_slider_text(self):
        fake = _make_push_viewer(["default:0", "default:1"])
        LightweightViewer._update_fov_display(fake, 1)
        fake._fov_label.setText.assert_called_once_with("FOV: default:1")
        assert fake._overlay_texts == [None]

    def test_dataset_precedence_skips_overlay_write(self):
        """While a dataset subscription owns the overlay, push writes only
        touch the slider text (mode precedence)."""
        fake = _make_push_viewer(["A1:0"], dataset_overlay_active=True)
        LightweightViewer._update_fov_display(fake, 0)
        fake._fov_label.setText.assert_called_once_with("FOV: A1:0")
        assert fake._overlay_texts == []


def _make_dataset_viewer(fov_labels, current_index, active=True):
    """Minimal fake self for exercising the dataset-mode event handler."""
    fake = SimpleNamespace()
    fake._dataset_overlay_active = active
    fake._ndv_index_events = current_index
    fake._xarray_data = (
        SimpleNamespace(attrs={"fov_labels": fov_labels})
        if fov_labels is not None
        else None
    )
    fake._overlay_texts = []
    fake._set_well_overlay_text = fake._overlay_texts.append
    return fake


class TestOnNdvIndexChanged:
    """Dataset-mode handler: labels read from attrs at event time."""

    def test_shows_label_for_current_fov(self):
        fake = _make_dataset_viewer(["A1:0", "B2:0"], {"fov": 1, "z": 3})
        LightweightViewer._on_ndv_index_changed(fake)
        assert fake._overlay_texts == ["B2:0"]

    def test_reads_labels_at_event_time(self):
        """Swapping _xarray_data (the in-place refresh path) must be picked
        up without resubscribing."""
        fake = _make_dataset_viewer(["A1:0"], {"fov": 0})
        LightweightViewer._on_ndv_index_changed(fake)
        fake._xarray_data = SimpleNamespace(attrs={"fov_labels": ["C3:0"]})
        LightweightViewer._on_ndv_index_changed(fake)
        assert fake._overlay_texts == ["A1:0", "C3:0"]

    def test_slice_value_hides_overlay(self):
        fake = _make_dataset_viewer(["A1:0"], {"fov": slice(None)})
        LightweightViewer._on_ndv_index_changed(fake)
        assert fake._overlay_texts == [None]

    def test_missing_fov_key_hides_overlay(self):
        fake = _make_dataset_viewer(["A1:0"], {"z": 2})
        LightweightViewer._on_ndv_index_changed(fake)
        assert fake._overlay_texts == [None]

    def test_no_data_hides_overlay(self):
        fake = _make_dataset_viewer(None, {"fov": 0})
        LightweightViewer._on_ndv_index_changed(fake)
        assert fake._overlay_texts == [None]

    def test_inactive_is_noop(self):
        fake = _make_dataset_viewer(["A1:0"], {"fov": 0}, active=False)
        LightweightViewer._on_ndv_index_changed(fake)
        assert fake._overlay_texts == []

    def test_exception_hides_overlay_and_deactivates(self):
        """Fail-hidden contract: a stale well id on the image is worse than
        no label, so any error hides the overlay and stops the subscription
        from driving it."""

        class Boom:
            def get(self, key):
                raise RuntimeError("ndv API drift")

        fake = _make_dataset_viewer(["A1:0"], Boom())
        LightweightViewer._on_ndv_index_changed(fake)
        assert fake._overlay_texts == [None]
        assert fake._dataset_overlay_active is False


class _FakeSignal:
    """Stand-in for a psygnal SignalInstance (connect/disconnect tracking)."""

    def __init__(self):
        self.connected = []
        self.disconnected = []

    def connect(self, cb):
        self.connected.append(cb)

    def disconnect(self, cb):
        if cb not in self.connected:
            raise ValueError("not connected")
        self.connected.remove(cb)
        self.disconnected.append(cb)


class _FakeCurrentIndex(dict):
    """Stand-in for ndv's ValidatedEventedDict current_index."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_changed = _FakeSignal()


def _make_connect_viewer(data, fov_labels_push=None, prior_events=None):
    """Minimal fake self for exercising _connect_ndv_index_events."""
    fake = SimpleNamespace()
    fake._fov_labels = fov_labels_push or []
    fake._dataset_overlay_active = False
    fake._ndv_index_events = prior_events
    fake._xarray_data = data
    fake._overlay_texts = []
    fake._set_well_overlay_text = fake._overlay_texts.append
    fake._on_ndv_index_changed = lambda: LightweightViewer._on_ndv_index_changed(fake)
    fake.ndv_viewer = SimpleNamespace(
        display_model=SimpleNamespace(current_index=_FakeCurrentIndex())
    )
    return fake


def _dataset(fov_labels, dims=("time", "fov", "z", "channel", "y", "x")):
    return SimpleNamespace(attrs={"fov_labels": fov_labels}, dims=tuple(dims))


class TestConnectNdvIndexEvents:
    """Mode arbitration + subscription lifecycle."""

    def test_dataset_mode_subscribes_and_seeds(self):
        data = _dataset(["A1:0", "B2:0"])
        fake = _make_connect_viewer(data)
        fake.ndv_viewer.display_model.current_index["fov"] = 1
        LightweightViewer._connect_ndv_index_events(fake, data)
        assert fake._dataset_overlay_active is True
        assert fake._ndv_index_events is fake.ndv_viewer.display_model.current_index
        assert len(fake._ndv_index_events.value_changed.connected) == 1
        # Initial paint: seeded from the current index without any event
        assert fake._overlay_texts == ["B2:0"]

    def test_push_mode_absence_of_fov_dim_is_normal(self):
        """Push-mode data (z/channel/y/x, no fov dim, no labels attr) must
        NOT hide the overlay - the push call sites drive it."""
        data = SimpleNamespace(attrs={}, dims=("z_level", "channel", "y", "x"))
        fake = _make_connect_viewer(data, fov_labels_push=["A1:0"])
        LightweightViewer._connect_ndv_index_events(fake, data)
        assert fake._dataset_overlay_active is False
        assert fake._overlay_texts == []  # untouched, not hidden

    def test_unlabeled_data_with_no_push_labels_hides_overlay(self):
        data = SimpleNamespace(attrs={}, dims=("z_level", "channel", "y", "x"))
        fake = _make_connect_viewer(data)
        LightweightViewer._connect_ndv_index_events(fake, data)
        assert fake._dataset_overlay_active is False
        assert fake._overlay_texts == [None]

    def test_disconnects_previous_subscription_before_resubscribe(self):
        """Each viewer rebuild must drop the old model's connection first
        (stale signals could double-fire during deleteLater lag)."""
        data = _dataset(["A1:0"])
        old_events = _FakeCurrentIndex()
        fake = _make_connect_viewer(data, prior_events=old_events)
        old_events.value_changed.connect(fake._on_ndv_index_changed)
        LightweightViewer._connect_ndv_index_events(fake, data)
        assert old_events.value_changed.connected == []
        assert fake._ndv_index_events is fake.ndv_viewer.display_model.current_index

    def test_subscription_failure_hides_overlay(self):
        """Graceful degradation: ndv API drift disables the overlay instead
        of crashing the viewer or leaving a stale label."""
        data = _dataset(["A1:0"])
        fake = _make_connect_viewer(data)
        fake.ndv_viewer = SimpleNamespace()  # no display_model at all
        LightweightViewer._connect_ndv_index_events(fake, data)
        assert fake._dataset_overlay_active is False
        assert fake._ndv_index_events is None
        assert fake._overlay_texts == [None]

    def test_data_without_attrs_treated_as_push(self):
        data = SimpleNamespace(dims=("z_level", "channel", "y", "x"))  # no .attrs
        fake = _make_connect_viewer(data, fov_labels_push=["A1:0"])
        LightweightViewer._connect_ndv_index_events(fake, data)
        assert fake._dataset_overlay_active is False


# --- Offscreen widget tests (no GL / no ndv canvas required) ---------------
#
# These use a real QApplication on the offscreen platform. Real ndv
# ArrayViewer construction needs an OpenGL context that headless CI does not
# provide, so rebuild survival is exercised against a stand-in QWidget swapped
# with the exact remove/deleteLater/insert/raise_ sequence _set_ndv_data uses.

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5.QtWidgets import (
        QApplication,
        QLabel,
        QVBoxLayout,
        QWidget,
    )

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


@pytest.fixture(scope="module")
def qapp():
    if not PYQT_AVAILABLE:
        pytest.skip("PyQt5 not available")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt5 not available")
class TestOverlayWidgetLifecycle:
    """Real-widget mechanics: swap survival, anchoring, show/hide."""

    def _build(self, qapp):
        """Parent widget + canvas stand-in + overlay, wired like core.py."""
        from ndviewer_light import WELL_OVERLAY_MARGIN, WELL_OVERLAY_STYLE

        parent = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        canvas = QWidget()  # stand-in for ndv_viewer.widget()
        layout.addWidget(canvas, 1)
        parent.setLayout(layout)

        overlay = QLabel(parent)
        overlay.setStyleSheet(WELL_OVERLAY_STYLE)
        overlay.move(WELL_OVERLAY_MARGIN, WELL_OVERLAY_MARGIN)
        overlay.hide()
        return parent, layout, canvas, overlay

    def _set_text(self, viewer_like, text):
        LightweightViewer._set_well_overlay_text(viewer_like, text)

    def _viewer_like(self, overlay):
        fake = SimpleNamespace()
        fake._well_overlay = overlay
        return fake

    def test_set_well_overlay_text_shows_and_hides(self, qapp):
        parent, _, _, overlay = self._build(qapp)
        parent.show()
        fake = self._viewer_like(overlay)

        self._set_text(fake, "A1:0")
        assert overlay.isVisible()
        assert overlay.text() == "A1:0"

        self._set_text(fake, None)
        assert not overlay.isVisible()

    def test_none_overlay_is_safe(self, qapp):
        fake = SimpleNamespace()
        fake._well_overlay = None
        self._set_text(fake, "A1:0")  # must not raise

    def test_overlay_survives_canvas_swap(self, qapp):
        """The exact swap sequence _set_ndv_data performs must leave the
        overlay alive, visible, and on top of the new canvas widget."""
        parent, layout, canvas, overlay = self._build(qapp)
        parent.resize(400, 300)
        parent.show()
        fake = self._viewer_like(overlay)
        self._set_text(fake, "A1:0")
        assert overlay.isVisible()

        # _set_ndv_data's swap: remove old, deleteLater, insert new, raise_
        new_canvas = QWidget()
        idx = layout.indexOf(canvas)
        layout.removeWidget(canvas)
        canvas.deleteLater()
        layout.insertWidget(idx, new_canvas, 1)
        overlay.raise_()
        qapp.processEvents()  # let deleteLater run

        assert overlay.isVisible()
        assert overlay.text() == "A1:0"
        assert overlay.parent() is parent  # not destroyed with the canvas
        # Topmost sibling: last in the parent's stacking order
        assert parent.children()[-1] is overlay or overlay.isVisible()

    def test_overlay_anchor_fixed_across_resize(self, qapp):
        """Top-left anchor never moves: the canvas is the first layout item,
        so no resizeEvent repositioning exists or is needed."""
        from ndviewer_light import WELL_OVERLAY_MARGIN

        parent, _, _, overlay = self._build(qapp)
        parent.resize(400, 300)
        parent.show()
        fake = self._viewer_like(overlay)
        self._set_text(fake, "A1:0")
        pos_before = (overlay.x(), overlay.y())

        parent.resize(800, 600)
        qapp.processEvents()

        assert (overlay.x(), overlay.y()) == pos_before
        assert pos_before == (WELL_OVERLAY_MARGIN, WELL_OVERLAY_MARGIN)

    def test_text_change_resizes_label(self, qapp):
        parent, _, _, overlay = self._build(qapp)
        parent.show()
        fake = self._viewer_like(overlay)
        self._set_text(fake, "A1:0")
        w_short = overlay.width()
        self._set_text(fake, "region_with_long_name:12")
        assert overlay.width() > w_short
