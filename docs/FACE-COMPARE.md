# Face recognition comparison (`face-compare.html` / `face-compare.js`)

Question this answers: **is face-api.js's recognition model (dlib, what
Face-API-WASM's `FaceMatcher.js` currently uses) as accurate as a newer
alternative (OpenCV's SFace)?**

## Two pipelines, run side by side on the same 3 input images

- **A1** and **A2**: two different photos of the same person
- **B**: a photo of a different person

| | Detector | Landmarks | Recognizer | Distance metric |
|---|---|---|---|---|
| dlib (existing) | face-api.js `TinyFaceDetector` | face-api.js 68-point | face-api.js `faceRecognitionNet` (128-d) | Euclidean distance, threshold ~0.6 (lower = more similar) |
| YuNet + SFace | YuNet (own ONNX, run via onnxruntime-web) | YuNet's own 5-point output | SFace ONNX (run via onnxruntime-web) | Cosine similarity, threshold ~0.363 (higher = more similar) |

Both pipelines are fully independent end to end (own detector, own
landmarks, own recognizer) so the comparison isn't contaminated by sharing
a detector -- each engine is tested exactly as it would run in production.

## Scripted/batch testing (`window.FaceCompareTest`)

For testing more than 3 images at a time (or driving the tool from browser
automation instead of manually clicking Capture/Upload), `face-compare.js`
exposes a `window.FaceCompareTest` hook once the page has loaded:

```js
const { loadImageToCanvas, extractDlib, extractSFace, euclideanDistance, cosineSimilarity, modelsReady } = window.FaceCompareTest;
await modelsReady; // wait for face-api.js + SFace + YuNet to finish loading
const canvasA = await loadImageToCanvas("staff_photo/p001.jpg");
const canvasB = await loadImageToCanvas("staff_photo/p002.jpg");
const distDlib = euclideanDistance(await extractDlib(canvasA), await extractDlib(canvasB));
const simSFace = cosineSimilarity(await extractSFace(canvasA), await extractSFace(canvasB));
```

This is the fastest path to properly re-testing with verified ground truth
(see caveat below) across many photo pairs at once instead of manually
re-running the 3-slot UI for every pair.

Inputs can come from the webcam (`Capture`) or from local files (`Upload
file`) -- file upload doesn't require camera permission and works with an
existing photo library (e.g. a staff photo folder) with no code changes.

## Why YuNet, not face-api.js's landmarks, for SFace's alignment

SFace's `alignCrop` step expects 5 specific points (right eye, left eye, nose
tip, right mouth corner, left mouth corner) in a specific order. The first
version of this tool approximated those 5 points from face-api.js's 68
landmarks (averaging eye-region points, guessing at "nose tip" from the last
few nose points, etc.) -- this works but isn't SFace's actual designed
pipeline, so a poor approximation could unfairly drag SFace's numbers down.
YuNet's own ONNX output already includes exactly the 5 points SFace expects,
in the exact order it expects (confirmed from OpenCV's own C++ source, see
`docs/DECISIONS.md`), so the current version uses that instead -- this is
SFace running the way it was actually designed to run.

## How alignment/preprocessing works (see `yunet.js` and `face-compare.js`)

1. YuNet detects the face + 5 landmarks (own ONNX model, 640x640 fixed input,
   decode logic ported from OpenCV's C++ source -- see `docs/DECISIONS.md`).
2. Those 5 points are fit to a standard 112x112 ArcFace-style reference
   template via a 2D similarity transform (rotation + uniform scale +
   translation, solved by plain linear least squares -- 4 unknowns, no SVD
   needed).
3. The warped 112x112 crop is fed to SFace: raw 0-255 RGB pixel values (no
   `/255` normalization -- confirmed from OpenCV's own
   `blobFromImage(img, 1, Size(112,112), Scalar(0,0,0), true, false)` call,
   scale factor 1 means no normalization), NCHW.
4. `match()` in OpenCV normalizes each feature vector then takes the dot
   product -- equivalent to plain cosine similarity, which is what this tool
   computes directly.

## Test results so far (⚠️ see caveat below)

Three tests run (2 uploaded-photo pairs, 1 live webcam pair), same "B"
diff-person photo reused across two of them:

| Test | dlib same | dlib diff | dlib verdict | SFace same | SFace diff | SFace verdict |
|---|---|---|---|---|---|---|
| 1 | 0.560 | 0.666 | correct, but only 0.04-0.07 from its 0.6 threshold | 0.500 | 0.090 | correct, large margin |
| 2 | 0.382 | 0.577 | **wrong** (0.577 < 0.6 -> would call two different people the same) | 0.505 | 0.117 | correct, large margin |
| 3 | 0.348 | 0.610 | correct, only 0.01 from threshold | 0.687 | 0.138 | correct, large margin |

Pattern across all three: dlib repeatedly operates right at its decision
boundary (once crossing it into a wrong answer); SFace's scores stay far from
its own boundary every time. This was consistent across 3 independent tests
using different capture methods (uploaded files and live webcam), which is
why it was written up as a real trend rather than one unlucky sample --
**but see the caveat immediately below before treating this as settled.**

### ⚠️ Caveat that undermines the above table

The photos used for these 3 tests came from a folder the user called
`staff_photo/`, which on inspection is **not actual employee photos** -- it
appears to be photos of one public figure across different contexts (a
candid photo, a studio promotional shot, an awards-ceremony photo with
visible trophy engravings, a press-conference photo, and a video screenshot
with a text overlay/watermark). Because of this, **it was never independently
confirmed that "B" is actually a different person from A1/A2** -- the "same
person / different person" labels above were taken on trust from how the
user assigned photos to slots, not verified against a known ground truth.
If B turns out to be the same person as A1/A2 in a very different setting
(quite possible given how different candid/studio/press-event photos of the
same person can look), then **the "dlib misclassification" in Test 2 might
not be a misclassification at all** -- it could be dlib correctly recognizing
the same person, and SFace incorrectly calling them different.

**This needs to be resolved with verified ground truth before the table
above is used to justify any real decision** (e.g. swapping recognition
engines in Face-API-WASM). See `docs/DECISIONS.md` for the reasoning.

## Open questions / not yet done

- Ground truth verification (above) -- the single most important next step.
- Only tested with 3 people total, all photos of adult East Asian men. No
  testing across age/gender/ethnicity diversity, which matters a lot for
  face recognition fairness and accuracy in production.
- 1:N (many registered people, find the best match) hasn't been tested --
  only 1:1 comparison. The underlying math is the same (see note in commit
  history / conversation: `findBestMatch` is just 1:1 comparison run N times,
  keep the best), but N-way false-accept-rate behaves differently than 1:1
  and should be tested at realistic staff-roster scale before drawing
  conclusions about production accuracy.
