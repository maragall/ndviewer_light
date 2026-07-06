# TODOS

## Export video as .mp4 (from IMA-177 eng review, 2026-07-04)

- **What:** An "Export video as .mp4" action that renders a recorded time series with current LUTs/contrast into a shareable file.
- **Why:** Showing a recording to a PI or collaborator currently requires ndviewer installed; an mp4 goes in Slack/email.
- **Pros:** High stakeholder-visible value; independent of all playback machinery; codecs guarantee smooth external viewing.
- **Cons:** Codec dependency (licensing questions), bakes display settings in at export time, adds a real UI surface.
- **Context:** mp4 was evaluated and REJECTED as the *playback* mechanism for IMA-177 (see `~/.gstack/projects/maragall-ndviewer_light/julioamaragall-juliomaragall-ima-177-video-player-design-20260704-173342.md`, Approach C) because it duplicates storage and leaves in-viewer scrubbing slow. As an *export* feature none of those objections apply — don't re-litigate mp4-as-playback; build mp4-as-export if demand appears.
- **Depends on / blocked by:** Nothing hard; benefits from the video data contract (IMA-177) landing first so exports read one store sequentially.
