# Anti-spoofing pipeline (`index.html` / `app.js`)

Question this answers: **can we detect "is this a real person or a spoof
(photo/screen replay), using only client-side JS, no LLM/VLM"?**

This is the pure-JS alternative to the project's existing `tms-liveness.js`
(VLM-based liveness via local LM Studio). It does not replace that -- see
"How this relates to tms-liveness.js" below.

## Pipeline

Every video frame runs through, in order:

1. **Face detection** -- face-api.js `TinyFaceDetector`. No face, no further work.
2. **Texture check (MiniFAS)** -- crops the face (1.5x bbox expansion), classifies
   real vs. spoof from skin/screen texture alone. Fast (<50ms), but confirmed
   fooled by a high-quality phone screen held at normal distance (see Finding 1).
3. **Object check (COCO-SSD)** -- runs a general object detector on the full
   frame; if a `cell phone`/`laptop`/`tv`/`remote`/`book` bounding box overlaps
   the face, that's a spoof regardless of what MiniFAS says. Catches what
   texture analysis misses (Finding 1), but its own per-frame confidence is
   noisy (Finding 2) -- a "sticky" 4-second window smooths this out.
4. **Geometry gate** -- if the face fills too much of the frame (>62% height or
   >28% area, thresholds copied from the existing `tms-app.js` `FACE_FRAMING`
   gate), reject on geometry alone. This is what actually catches "phone held
   right up against the camera lens," which defeats both layers 2 and 3
   (Finding 3).
5. **Motion parallax (debug-only, not wired into the final verdict yet)** --
   MediaPipe Face Landmarker (478 points) + a homography fit between the
   current frame and a reference frame from ~700ms ago. A flat photo/screen's
   entire point set moves through one shared homography (near-zero
   reprojection error); a real 3D face's motion can't be exactly explained by
   a single homography (points closer to the camera move more). See "Why not
   just look at MediaPipe's z coordinate" below -- that's the one subtlety
   that matters here.

Final verdict = `textureVerdict.isReal AND NOT deviceSticky AND NOT tooClose`.
Layer 5 (parallax) is computed and displayed every frame but does **not**
currently affect this verdict -- see Finding 4 for why.

## Why not just look at MediaPipe's z coordinate?

MediaPipe's per-landmark z is **not measured depth** -- it's inferred by
fitting a canonical 3D face model onto the 2D image, so it happily reports a
plausible-looking z for a flat photo too. The only signal a flat surface
cannot fake is the actual 2D motion field across frames (parallax), which is
why layer 5 tracks motion and fits a homography instead of reading z directly.

## Findings (in the order they were discovered)

1. **MiniFAS alone misses close-up, high-quality phone-screen replay.**
   Confirmed: holding a phone-displayed photo at normal-ish distance, MiniFAS
   read it as `REAL` with high confidence (`real=8.32 spoof=-8.31`). Motivated
   adding layer 3 (COCO-SSD).
2. **COCO-SSD's per-frame confidence for "cell phone" is noisy.** Same
   physical phone scored 86% one frame, 9% the next, as the hand/angle shifted
   a few degrees. Fixed with a 4-second "sticky" window (`DEVICE_STICKY_MS` in
   `app.js`): once a device is seen overlapping the face, treat the session as
   suspicious for the next 4 seconds instead of trusting only the current frame.
3. **Holding the phone right up against the lens defeats both layer 2 and
   layer 3.** No visible phone silhouette (COCO-SSD can't recognize a
   "phone-shaped" object with no visible edges: max confidence seen was 10%),
   and MiniFAS still reads clean texture at that focus distance. This is a
   *reproducible* failure mode, not a one-off. Fixed by layer 4 (geometry
   gate) -- a face that close is itself the anomaly, independent of what
   either model concludes about the content.
4. **Motion parallax gave false positives on casual real-face motion at the
   default motion threshold (4px).** Two consecutive real-face tests were
   flagged `PLANAR/suspicious` (4/6 and 6/6 votes). Root cause: real parallax
   (near points moving more than far points) is a second-order effect of
   rotation angle -- small, casual motions barely reveal it even for a
   genuinely 3D face, so small motions look "flat" whether real or fake.
   Raised `MOTION_MIN_PX` from 4 to 18 (only trust the residual when there's
   decisively large motion). After the fix: a deliberate head turn produced
   `motion=44.4px rmse=5.47px NON-PLANAR votes=0/6` -- correctly identified
   as real, large margin over the 1.2px planar threshold. **This has only
   been validated with deliberate large head turns, not casual/small motion
   at the new threshold** -- that's the open question below.

## Open questions / not yet done

- Layer 5 (parallax) is not wired into the final verdict. It needs more
  testing across natural (not deliberately exaggerated) head motion before
  it can be trusted as a veto -- see Finding 4.
- No false-positive-rate testing on layer 4 (geometry gate) for people who
  naturally sit close to their webcam (e.g. laptop users, short focal
  distance). The 62%/28% thresholds are borrowed from `tms-app.js`, not
  re-validated here.
- `app.js` currently has several `TEMP` diagnostic blocks (a synchronous
  canvas self-check, `window.__debugOverlay`/`__debugFrameStats`) left over
  from debugging a "can't see the drawn overlay" issue that turned out to be
  a rendering-visibility problem, not a logic bug. Safe to remove once
  confirmed no longer needed.

## How this relates to `tms-liveness.js` (the VLM approach)

This pure-JS pipeline and the existing VLM-based `tms-liveness.js` are
**complementary, not competing**: this one is fast (real-time, every frame)
and needs no external service, but each layer has known blind spots (above).
The VLM sees full-frame *scene context* (a hand holding a phone, a screen
bezel, the room around the subject) that none of these four layers reason
about directly -- COCO-SSD gets partway there (layer 3) but only recognizes
objects it was trained on, and only when their silhouette is visible.
Reasonable production architecture: this pipeline as a fast first-pass
filter, VLM review for anything it doesn't confidently clear.
