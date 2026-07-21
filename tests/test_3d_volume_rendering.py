"""3D volume rendering: the monkey-patches, the z-axis guess, and the voxel scale.

Three separate failures had to be fixed to make ndv's built-in 3D button work on a
z-stack, and each has a test here:

1. THE PATCH MUST STAY BOUND. ``core`` monkey-patches ``VispyArrayCanvas.add_volume``
   and ``VolumeVisual.__init__`` / ``._create_vertex_data``. If a future ndv or vispy
   renames or restructures either target, the ``try/except ImportError`` around the patch
   swallows it and 3D silently reverts to isotropic — a class of failure that has already
   cost this project a full day once (a viewer missing ``register_array`` went black and
   said nothing). ``TestMonkeyPatchesAreBound`` makes that loud.

2. THE Z AXIS MUST NOT BE THE CHANNEL AXIS. ndv's ``guess_z_axis`` returned the CHANNEL
   axis for our ``(z_level, channel, y, x)`` arrays, so the 3D button built a volume out
   of the channels. See ``Downsampling3DXarrayWrapper.guess_z_axis``.

3. THE VOXEL SCALE MUST BE PHYSICAL. Push-mode acquisitions built their arrays in code
   and never carried ``pixel_size_um`` / ``dz_um``, so volumes rendered isotropic — 2x
   wrong in z on a dz=1.5um / pixel=0.752um stack. And when the wrapper downsamples xy to
   fit the GL texture limit, the stretch has to be corrected by that factor too.
"""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("ndv")
pytest.importorskip("vispy")

from ndviewer_light import core  # noqa: E402

# The real tissue z-stack this feature was built against.
TISSUE_PIXEL_SIZE_UM = 0.752
TISSUE_DZ_UM = 1.5
TISSUE_ASPECT = TISSUE_DZ_UM / TISSUE_PIXEL_SIZE_UM  # ~1.995


def _instance_attrs(cls):
    """Attribute names any method of ``cls`` assigns to self — vispy sets these in
    __init__ rather than declaring them on the class, so hasattr(cls, ...) misses them."""
    names = set()
    for obj in vars(cls).values():
        code = getattr(obj, "__code__", None)
        if code is not None:
            names.update(code.co_names)
    return names


def _stack(n_z=10, n_c=3, h=64, w=64):
    """An array shaped exactly like the ones the viewer hands ndv."""
    return xr.DataArray(
        np.zeros((n_z, n_c, h, w), dtype=np.uint16),
        dims=["z_level", "channel", "y", "x"],
        coords={"z_level": list(range(n_z)), "channel": list(range(n_c))},
    )


class TestMonkeyPatchesAreBound:
    """A stale monkey-patch must never again fail silently."""

    def test_patch_targets_still_exist_in_ndv_and_vispy(self):
        """The attributes core.py patches must still be there to patch."""
        from ndv.views._vispy._array_canvas import VispyArrayCanvas
        from vispy.visuals.volume import VolumeVisual

        assert hasattr(VispyArrayCanvas, "add_volume")
        assert hasattr(VolumeVisual, "_create_vertex_data")
        # The patched vertex builder reads these off the instance; a vispy rename would
        # make it raise and the volume would never appear.
        for attr in ("_vol_shape", "_vertices", "_index_buffer"):
            assert hasattr(VolumeVisual, attr) or attr in _instance_attrs(VolumeVisual)

    def test_patches_are_actually_installed(self):
        """Importing core must leave OUR functions bound, not the originals.

        This is the guard: the patch block is wrapped in try/except ImportError, so an
        upstream rename makes it a no-op with no traceback. Asserting on __name__ is what
        turns that silence into a red test.
        """
        from ndv.views._vispy._array_canvas import VispyArrayCanvas
        from vispy.visuals.volume import VolumeVisual

        assert VispyArrayCanvas.add_volume.__name__ == "_patched_add_volume"
        assert VolumeVisual.__init__.__name__ == "_patched_init"
        assert VolumeVisual._create_vertex_data.__name__ == "_patched_create_vertex_data"
        assert getattr(VolumeVisual, "_voxel_scale_patch", False) is True
        assert getattr(VispyArrayCanvas, "_camera_scale_patch", False) is True

    def test_add_volume_signature_still_accepts_data(self):
        """_patched_add_volume calls through as (self, data); keep that contract."""
        import inspect

        from ndv.views._vispy._array_canvas import VispyArrayCanvas

        params = list(inspect.signature(VispyArrayCanvas.add_volume).parameters)
        assert params == ["self", "data"]


class TestFailuresAreNotSilent:
    """IMA-257: a volume that cannot be built must SAY so, not render empty.

    The whole 3D path is defensive — a rename, a frozen visual, a failed downsample, a
    camera that will not frame — and every one of those used to end in `pass` or a debug
    log. The viewer then showed an unscaled or blank box that looks plausible and is
    wrong. Each swallow point now records a reason that a caller can ask for.
    """

    @pytest.fixture(autouse=True)
    def _clean(self):
        core._clear_volume_problem()
        yield
        core._clear_volume_problem()

    def test_patches_applied_cleanly_on_this_install(self):
        """The import-time guard: None means the patch block really ran."""
        assert core.VOLUME_PATCH_ERROR is None, (
            f"3D volume patches did not apply: {core.VOLUME_PATCH_ERROR}"
        )

    def test_no_problem_reported_when_nothing_is_wrong(self):
        assert core.last_volume_problem() is None

    def test_a_problem_is_readable_by_the_caller(self):
        core._note_volume_problem("volume too large to upload")
        assert "too large" in core.last_volume_problem()

    def test_a_problem_is_cleared_by_a_good_volume(self):
        core._note_volume_problem("something broke")
        assert core.last_volume_problem() is not None
        core._clear_volume_problem()
        assert core.last_volume_problem() is None

    def test_no_except_in_the_volume_path_is_silent(self):
        """Structural guard: every handler in the patch block must report or re-raise.

        Written structurally on purpose. The individual failures need a GL context or a
        vispy API break to provoke, so a behavioural test would have to fake them — and
        a faked test of a `pass` proves nothing. What actually regressed here was someone
        adding a quiet `except`, and that is visible in the source.
        """
        import inspect

        src = inspect.getsource(core).splitlines()
        # The volume patch block, from the vispy import to the end of add_volume patching.
        start = next(i for i, ln in enumerate(src) if "Monkeypatch vispy VolumeVisual" in ln)
        end = next(i for i, ln in enumerate(src) if "_camera_scale_patch = True" in ln)

        silent = []
        for i in range(start, end):
            if src[i].strip().startswith("except"):
                body = "\n".join(src[i + 1 : i + 8])
                reports = "_note_volume_problem" in body or "logger.error" in body
                reraises = "raise" in body
                if not (reports or reraises):
                    silent.append(f"line {i + 1}: {src[i].strip()}")
        assert not silent, (
            "silent except in the 3D volume path — a failed volume would render blank "
            f"or isotropic with nothing said (IMA-257): {silent}"
        )

    def test_the_reason_names_the_consequence_not_just_the_exception(self):
        """A message that only says 'ValueError' does not tell the user what they see."""
        core._note_volume_problem(
            "could not downsample a (10, 4000, 4000) volume to fit the 2048px GL "
            "texture limit (boom) — the volume is too large to upload and will "
            "render blank"
        )
        msg = core.last_volume_problem()
        assert "render blank" in msg  # what the USER observes
        assert "2048" in msg  # the actionable number


class TestGuessZAxis:
    """ndv's 3D button asks the wrapper which axis to make the third visible one."""

    def test_z_axis_is_z_level_not_channel(self):
        """The bug: ndv returned the channel axis, so 3D stacked CHANNELS."""
        w = core.Downsampling3DXarrayWrapper(_stack())
        assert w.guess_channel_axis() == 1
        assert w.guess_z_axis() == 0

    def test_z_axis_survives_a_single_channel(self):
        w = core.Downsampling3DXarrayWrapper(_stack(n_c=1))
        assert w.guess_z_axis() == 0

    def test_z_axis_never_equals_channel_axis(self):
        for n_c in (1, 2, 4):
            w = core.Downsampling3DXarrayWrapper(_stack(n_c=n_c))
            assert w.guess_z_axis() != w.guess_channel_axis()


class TestVoxelScaleReachesTheRenderer:
    """Every micrometre value that decides the volume's geometry, checked as a number."""

    @pytest.fixture(autouse=True)
    def _reset_globals(self):
        yield
        core._current_voxel_scale = None
        core._current_volume_zoom_ratio = 1.0

    def test_effective_scale_is_none_without_metadata(self):
        core._current_voxel_scale = None
        assert core._effective_voxel_scale() is None

    def test_effective_scale_is_the_physical_aspect(self):
        core._current_voxel_scale = (1.0, 1.0, TISSUE_ASPECT)
        core._current_volume_zoom_ratio = 1.0
        assert core._effective_voxel_scale()[2] == pytest.approx(1.99468, abs=1e-4)

    def test_downsampling_xy_corrects_the_stretch(self):
        """2800px xy shrunk to 2048 while z is untouched => 0.731x correction.

        Without this the volume renders 1.37x too tall: the physical aspect describes the
        data on disk, but what reaches the GPU has had xy (and only xy) shrunk.
        """
        core._current_voxel_scale = (1.0, 1.0, TISSUE_ASPECT)
        core._current_volume_zoom_ratio = 2048 / 2800
        assert core._effective_voxel_scale()[2] == pytest.approx(1.4592, abs=1e-3)

    def test_zoom_ratio_from_dim_info(self):
        dim_info = [("z_level", 10), ("channel", 3), ("y", 2800), ("x", 2800)]
        zooms = [1.0, 1.0, 2048 / 2800, 2048 / 2800]
        ratio = core._zoom_ratio(dim_info, zooms, {"z", "z_level", "depth", "focus"})
        assert ratio == pytest.approx(2048 / 2800)

    def test_zoom_ratio_is_one_when_nothing_downsampled(self):
        dim_info = [("z_level", 10), ("channel", 3), ("y", 64), ("x", 64)]
        ratio = core._zoom_ratio(dim_info, [1.0, 1.0, 1.0, 1.0], {"z_level"})
        assert ratio == 1.0


class TestPatchedVertexGeometry:
    """The patched _create_vertex_data is what actually stretches the volume in z."""

    class _Buffer:
        def __init__(self):
            self.data = None

        def set_data(self, data):
            self.data = data

    class _StubVolume:
        """Enough of a VolumeVisual for _create_vertex_data, without an GL context."""

        def __init__(self, shape, scale):
            self._vol_shape = shape
            self._voxel_scale = scale
            self._vertices = TestPatchedVertexGeometry._Buffer()
            self._index_buffer = TestPatchedVertexGeometry._Buffer()

    def test_z_extent_matches_the_physical_aspect(self):
        """10 planes at dz=1.5um over 256px at 0.752um must render ~2x tall per plane."""
        from vispy.visuals.volume import VolumeVisual

        stub = self._StubVolume((10, 256, 256), (1.0, 1.0, TISSUE_ASPECT))
        VolumeVisual._create_vertex_data(stub)
        pos = stub._vertices.data

        assert pos[:, 2].min() == pytest.approx(-0.5 * TISSUE_ASPECT)
        assert pos[:, 2].max() == pytest.approx(9.5 * TISSUE_ASPECT)
        # x/y are left in pixels, so the rendered box must have the physical proportions.
        rendered = np.ptp(pos[:, 2]) / np.ptp(pos[:, 0])
        physical = (10 * TISSUE_DZ_UM) / (256 * TISSUE_PIXEL_SIZE_UM)
        assert rendered == pytest.approx(physical, rel=1e-6)

    def test_unscaled_volume_is_left_alone(self):
        """No metadata => fall through to vispy's own vertices, z in plain plane units."""
        from vispy.visuals.volume import VolumeVisual

        stub = self._StubVolume((10, 256, 256), None)
        VolumeVisual._create_vertex_data(stub)
        assert stub._vertices.data[:, 2].max() == pytest.approx(9.5)

    def test_voxel_scale_survives_vispy_freeze(self):
        """vispy Visuals self.freeze() in __init__, rejecting new attribute names.

        The patch used to assign self._voxel_scale straight after __init__ and swallow the
        resulting AttributeError, so the scale never stuck and every volume rendered
        isotropic. Anything that re-freezes before the assignment must fail here.
        """

        class Frozen:
            __isfrozen = False

            def freeze(self):
                self.__isfrozen = True

            def unfreeze(self):
                self.__isfrozen = False

            def __setattr__(self, name, value):
                if getattr(self, "_Frozen__isfrozen", False) and not hasattr(self, name):
                    raise AttributeError(f"{self} is frozen; cannot set {name!r}")
                object.__setattr__(self, name, value)

        obj = Frozen()
        obj.freeze()
        with pytest.raises(AttributeError):
            obj._voxel_scale = (1.0, 1.0, TISSUE_ASPECT)
        # ...which is exactly why the patch unfreezes first.
        obj.unfreeze()
        obj._voxel_scale = (1.0, 1.0, TISSUE_ASPECT)
        obj.freeze()
        assert obj._voxel_scale[2] == pytest.approx(TISSUE_ASPECT)

    def test_vispy_visuals_still_freeze_themselves(self):
        """If vispy ever drops freeze/unfreeze, the patch's dance needs revisiting.

        Asserted on BaseVisual, not on VolumeVisual.__init__, because __init__ is the
        thing we monkey-patched — reading its source would just read our own patch back.
        """
        from vispy.visuals.visual import BaseVisual

        assert hasattr(BaseVisual, "freeze")
        assert hasattr(BaseVisual, "unfreeze")


class TestPushModeCarriesVoxelSize:
    """start_acquisition is push mode's ONLY chance to declare the voxel size."""

    def test_start_acquisition_accepts_um_keywords(self):
        import inspect

        from ndviewer_light import LightweightViewer

        params = inspect.signature(LightweightViewer.start_acquisition).parameters
        assert "pixel_size_um" in params
        assert "dz_um" in params
        # Optional: existing callers that omit them must keep working.
        assert params["pixel_size_um"].default is None
        assert params["dz_um"].default is None

    def test_stamp_writes_um_attrs(self):
        from ndviewer_light import LightweightViewer

        viewer = LightweightViewer.__new__(LightweightViewer)
        viewer._pixel_size_um = TISSUE_PIXEL_SIZE_UM
        viewer._dz_um = TISSUE_DZ_UM
        xarr = LightweightViewer._stamp_voxel_size_um(viewer, _stack())
        assert xarr.attrs["pixel_size_um"] == TISSUE_PIXEL_SIZE_UM
        assert xarr.attrs["dz_um"] == TISSUE_DZ_UM
        # Everything in this project is micrometres and every key ends in _um.
        assert all(k.endswith("_um") for k in xarr.attrs)

    def test_stamp_is_a_noop_without_sizes(self):
        from ndviewer_light import LightweightViewer

        viewer = LightweightViewer.__new__(LightweightViewer)
        viewer._pixel_size_um = None
        viewer._dz_um = None
        xarr = LightweightViewer._stamp_voxel_size_um(viewer, _stack())
        assert "pixel_size_um" not in xarr.attrs
        assert "dz_um" not in xarr.attrs
