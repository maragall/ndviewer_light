"""Tests for slider playback: play buttons and playback speed.

Characterization tests that pin the current play-button behavior so the
playback refactor and speed-control feature (IMA-190) land on a green
baseline.

Follows the house pattern: no QApplication / pytest-qt. ViewerDouble
inherits the real playback methods from LightweightViewer but skips Qt
construction; QTimer is patched at the module level per-test.
"""

import sys
from unittest.mock import MagicMock, patch

from ndviewer_light import LightweightViewer

# The core module, resolved via the class so this works both when the repo
# root shim is the ndviewer_light package (CI, Squid submodule) and when the
# inner package is imported directly.
core = sys.modules[LightweightViewer.__module__]


class ViewerDouble(LightweightViewer):
    """Real playback methods, mocked collaborators, no Qt construction."""

    def __init__(self):  # deliberately does NOT call super().__init__()
        self._time_play_timer = None
        self._fov_play_timer = None
        self._time_play_btn = MagicMock()
        self._fov_play_btn = MagicMock()
        self._time_slider = MagicMock()
        self._fov_slider = MagicMock()


class TestTimePlayButton:
    """Characterize _on_time_play_clicked timer lifecycle."""

    def test_play_creates_and_starts_timer_at_current_interval(self):
        viewer = ViewerDouble()
        with patch.object(core, "QTimer") as timer_cls:
            viewer._on_time_play_clicked(True)

        timer_cls.assert_called_once_with(viewer)
        assert viewer._time_play_timer is timer_cls.return_value
        viewer._time_play_timer.timeout.connect.assert_called_once()
        viewer._time_play_timer.start.assert_called_once_with(100)

    def test_play_connects_timer_to_time_step(self):
        viewer = ViewerDouble()
        with patch.object(core, "QTimer") as timer_cls:
            viewer._on_time_play_clicked(True)

        (connected_cb,) = timer_cls.return_value.timeout.connect.call_args.args
        assert connected_cb.__func__ is LightweightViewer._time_play_step

    def test_play_reuses_existing_timer(self):
        viewer = ViewerDouble()
        existing = MagicMock()
        viewer._time_play_timer = existing
        with patch.object(core, "QTimer") as timer_cls:
            viewer._on_time_play_clicked(True)

        timer_cls.assert_not_called()
        assert viewer._time_play_timer is existing
        existing.timeout.connect.assert_not_called()
        existing.start.assert_called_once_with(100)

    def test_pause_stops_timer(self):
        viewer = ViewerDouble()
        viewer._time_play_timer = MagicMock()
        viewer._on_time_play_clicked(False)

        viewer._time_play_timer.stop.assert_called_once()

    def test_pause_with_no_timer_is_noop(self):
        viewer = ViewerDouble()
        viewer._on_time_play_clicked(False)  # must not raise
        assert viewer._time_play_timer is None

    def test_text_fallback_without_iconify(self):
        viewer = ViewerDouble()
        with (
            patch.object(core, "QTimer"),
            patch.object(core, "ICONIFY_AVAILABLE", False),
        ):
            viewer._on_time_play_clicked(True)
            viewer._on_time_play_clicked(False)

        texts = [c.args[0] for c in viewer._time_play_btn.setText.call_args_list]
        assert texts == ["⏸", "▶"]

    def test_no_text_swap_with_iconify(self):
        viewer = ViewerDouble()
        with (
            patch.object(core, "QTimer"),
            patch.object(core, "ICONIFY_AVAILABLE", True),
        ):
            viewer._on_time_play_clicked(True)

        viewer._time_play_btn.setText.assert_not_called()


class TestFovPlayButton:
    """Characterize _on_fov_play_clicked timer lifecycle."""

    def test_play_creates_and_starts_timer_at_current_interval(self):
        viewer = ViewerDouble()
        with patch.object(core, "QTimer") as timer_cls:
            viewer._on_fov_play_clicked(True)

        timer_cls.assert_called_once_with(viewer)
        assert viewer._fov_play_timer is timer_cls.return_value
        viewer._fov_play_timer.start.assert_called_once_with(100)

    def test_play_connects_timer_to_fov_step(self):
        viewer = ViewerDouble()
        with patch.object(core, "QTimer") as timer_cls:
            viewer._on_fov_play_clicked(True)

        (connected_cb,) = timer_cls.return_value.timeout.connect.call_args.args
        assert connected_cb.__func__ is LightweightViewer._fov_play_step

    def test_pause_stops_timer(self):
        viewer = ViewerDouble()
        viewer._fov_play_timer = MagicMock()
        viewer._on_fov_play_clicked(False)

        viewer._fov_play_timer.stop.assert_called_once()

    def test_both_timers_are_independent(self):
        viewer = ViewerDouble()
        with patch.object(core, "QTimer", side_effect=lambda *a: MagicMock()):
            viewer._on_time_play_clicked(True)
            viewer._on_fov_play_clicked(True)

        assert viewer._time_play_timer is not viewer._fov_play_timer
        viewer._time_play_timer.start.assert_called_once_with(100)
        viewer._fov_play_timer.start.assert_called_once_with(100)


class TestPlayStep:
    """Characterize the looping step callbacks."""

    def test_time_step_advances(self):
        viewer = ViewerDouble()
        viewer._time_slider.maximum.return_value = 3
        viewer._time_slider.value.return_value = 1
        viewer._time_play_step()

        viewer._time_slider.setValue.assert_called_once_with(2)

    def test_time_step_wraps_to_zero(self):
        viewer = ViewerDouble()
        viewer._time_slider.maximum.return_value = 3
        viewer._time_slider.value.return_value = 3
        viewer._time_play_step()

        viewer._time_slider.setValue.assert_called_once_with(0)

    def test_time_step_noop_for_single_timepoint(self):
        viewer = ViewerDouble()
        viewer._time_slider.maximum.return_value = 0
        viewer._time_play_step()

        viewer._time_slider.setValue.assert_not_called()

    def test_fov_step_wraps_to_zero(self):
        viewer = ViewerDouble()
        viewer._fov_slider.maximum.return_value = 2
        viewer._fov_slider.value.return_value = 2
        viewer._fov_play_step()

        viewer._fov_slider.setValue.assert_called_once_with(0)

    def test_fov_step_noop_for_single_fov(self):
        viewer = ViewerDouble()
        viewer._fov_slider.maximum.return_value = 0
        viewer._fov_play_step()

        viewer._fov_slider.setValue.assert_not_called()


class TestStopPlayAnimation:
    """Characterize the shared stop helper."""

    def test_stops_active_timer_and_resets_button(self):
        viewer = ViewerDouble()
        timer = MagicMock()
        timer.isActive.return_value = True
        button = MagicMock()
        with patch.object(core, "ICONIFY_AVAILABLE", False):
            viewer._stop_play_animation(timer, button)

        timer.stop.assert_called_once()
        button.setChecked.assert_called_once_with(False)
        button.setText.assert_called_once_with("▶")

    def test_inactive_timer_untouched(self):
        viewer = ViewerDouble()
        timer = MagicMock()
        timer.isActive.return_value = False
        button = MagicMock()
        viewer._stop_play_animation(timer, button)

        timer.stop.assert_not_called()
        button.setChecked.assert_not_called()

    def test_none_timer_is_noop(self):
        viewer = ViewerDouble()
        viewer._stop_play_animation(None, MagicMock())  # must not raise
