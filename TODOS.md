# TODOS

## Persist playback fps across sessions
- **What:** Remember the user's chosen playback fps between app launches (QSettings).
- **Why:** After IMA-190, fps resets to `DEFAULT_PLAYBACK_FPS` (5) every launch; a user who always wants 2 fps re-sets it each session.
- **Pros:** ~10 lines; QSettings is the boring Qt built-in; control "learns".
- **Cons:** Introduces the app's first persistent-state pattern — an architectural precedent that deserves its own review moment; no user has asked yet.
- **Context:** After IMA-190 lands, fps lives in `self._playback_fps` (set from `DEFAULT_PLAYBACK_FPS` at construction, updated by the spinbox's `valueChanged`). Persistence = one QSettings read at construction + one write in `_on_playback_fps_changed`. Decided in the IMA-190 eng review (2026-07-04) to keep that PR spec-scoped.
- **Depends on:** IMA-190 landing.

## Playback tick outruns synchronous frame loads (pre-existing)
- **What:** Decouple playback ticks from synchronous frame loading (skip-ahead or async load) so requested fps ≈ actual fps.
- **Why:** Each play-timer tick does `setValue` → `valueChanged` → `_load_current_position()` (`core.py:1654`) → synchronous TIFF/zarr load on the main thread. On slow storage the UI stutters and actual fps falls below the setting. This is the deeper cause of janky playback, independent of the "too fast" default fixed by IMA-190.
- **Pros:** Diagnosis is already done (verified by two independent reviewers in the IMA-190 eng review); fixes root jank.
- **Cons:** Real fix means threading/async work in a 4000-line single-file app — meaningful scope; speculative until a user formally reports stutter.
- **Context:** Note `ZARR_LOAD_DEBOUNCE_MS` (`core.py:87`) does NOT cover playback — it only debounces live-acquisition frame registration (`_on_zarr_frame_registered` → `_schedule_zarr_debounced_load`, `core.py:2572/2576`). Slider-driven playback bypasses it entirely. Qt coalesces timer events while the handler runs, so this degrades to slower-than-requested playback rather than a queue explosion.
- **Depends on:** Nothing; independent of IMA-190.
