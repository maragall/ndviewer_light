# TODOS

## Well label in exported/screenshot images
- **What:** Draw the current well id into ndv canvas exports/screenshots.
- **Why:** IMA-178 ships the label as a Qt overlay, which is not part of the vispy canvas — any image saved from the viewer loses the label, so shared screenshots aren't self-documenting.
- **Pros:** Screenshots and exports carry the well id; better for lab notebooks and bug reports.
- **Cons:** Requires drawing inside the ndv/vispy canvas, which IMA-178 deliberately forbade (decoupling from third-party internals); demand unproven.
- **Context:** Deferred from the IMA-178 eng review (2026-07-04). The overlay is a QLabel parented to `LightweightViewer`; the canvas knows nothing about it. Start point: check whether newer ndv versions expose a sanctioned annotation/overlay API before writing any vispy code.
- **Depends on / blocked by:** ndv exposing a stable annotation API (none in 0.4.1).

## GL-capable widget-test infrastructure
- **What:** CI setup (xvfb + EGL, or a macOS runner) that can construct a real `ndv.ArrayViewer` inside tests.
- **Why:** IMA-178's rebuild-survival tests use a stand-in QWidget swapped the same way `_set_ndv_data` swaps the ndv widget — real-ndv integration breakage (e.g. an ndv upgrade renaming `display_model.current_index` or `_lut_controllers`) only surfaces at runtime today.
- **Pros:** ndv API drift caught in CI instead of by users; the viewer now has two private-API dependencies on ndv, so drift risk is real.
- **Cons:** Headless GL for vispy is finicky; nontrivial CI work that may need a dedicated runner.
- **Context:** Deferred from the IMA-178 eng review (2026-07-04). The offscreen `QApplication` pytest fixture added by IMA-178 is the starting point; the missing piece is a GL context for vispy.
- **Depends on / blocked by:** CI runner choice (GitHub-hosted vs self-hosted with GPU/EGL).
