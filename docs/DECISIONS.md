# Technical decisions and dead ends

Things worth knowing before touching this code again, in roughly the order
they came up.

## OpenCV.js must run in a Web Worker, never the main thread

The `@techstark/opencv-js` build embeds its WASM binary inside the 13MB JS
file and compiles it **synchronously** on load. Loading it directly on the
main thread froze the entire page for 30+ seconds -- confirmed by the page
becoming unresponsive to `preview_screenshot`/`preview_eval` until the
server was restarted. This is not an OpenCV.js problem in general, it's this
specific prebuilt distribution (no separate `.wasm` file to stream-compile).

Fix: `opencv-worker.js` loads OpenCV.js via `importScripts()` inside a Web
Worker. The heavy synchronous compile happens off the main thread; the page
stays responsive. The worker "warms up" in parallel with other model loading
(`fitter.start()` is called before `await` on anything else in
`ParallaxDetector.init()`), and callers fall back to a hand-rolled
least-squares homography fit (`fitHomographyLstsq` in `parallax.js`) until
the worker reports ready or if it dies.

**Lesson: any large third-party WASM dependency should be assumed to block
the main thread until proven otherwise. Load first, verify the page stays
interactive, before wiring in real logic.**

## YuNet's exact output format had to be found from OpenCV's own C++ source

The 2023mar ONNX export of YuNet (`opencv_zoo`'s
`face_detection_yunet_2023mar.onnx`) has a **different output structure**
than older versions/reference implementations found via casual search (e.g.
`geaxgx/depthai_yunet`, which targets an older `.blob` export with combined
`loc`/`conf`/`iou` tensors across 4 anchor-based feature maps). The 2023mar
version instead exports 12 separate tensors (`cls_8/16/32`, `obj_8/16/32`,
`bbox_8/16/32`, `kps_8/16/32` -- 3 anchor-free scales), confirmed via
`onnx.load()` inspecting the actual graph inputs/outputs.

The precise decode formula (score = `sqrt(clamp(cls) * clamp(obj))`, bbox as
grid-relative offsets scaled by stride, 5 landmarks as grid-relative offsets)
was found in OpenCV's own C++ source:
`opencv/modules/objdetect/src/face_detect.cpp`, function
`FaceDetectorYNImpl::postProcess`. This is the authoritative source for this
exact model version -- don't reuse decode logic from a reference targeting a
different YuNet export without checking the output tensor names/shapes
match first.

**Lesson: "this model architecture is well-documented" doesn't mean *this
exact exported version* matches every reference implementation you find --
different export dates can have materially different output formats. Check
the actual ONNX graph I/O before trusting a decode implementation.**

Also confirmed from the same source: YuNet's `blobFromImage` call uses
`swapRB=false` (default) -- since OpenCV Mats are BGR internally, this means
YuNet expects BGR input; canvas `ImageData` is RGB, so `yunet.js` explicitly
swaps R and B when building the input tensor. SFace's own preprocessing call
uses `swapRB=true` explicitly, which cancels out OpenCV's internal BGR --
meaning SFace's canvas-to-tensor code does NOT swap channels. Easy to get
backwards; both are documented with their source line in the respective
`*.js` files.

## SFace preprocessing: raw 0-255 pixel values, not `/255` normalized

Confirmed from `face_recognize.cpp`:
`dnn::blobFromImage(aligned_img, /*scale*/1, Size(112,112), Scalar(0,0,0), /*swapRB*/true, /*crop*/false)`.
Scale factor `1` means no normalization at all -- this is unlike most other
models in this project (e.g. MiniFAS, which does divide by 255). Don't
assume a shared preprocessing convention across models; check each model's
actual expected input range.

## MediaPipe's landmark `z` is not measured depth

It's inferred by fitting a canonical 3D face model onto the 2D image, so it
reports plausible-looking z values even for a flat photo. The motion
parallax layer deliberately does NOT use z at all -- it only uses the 2D
(x,y) motion field across frames and checks whether that motion is
explainable by a single homography (flat surface) or not (real 3D
structure). This is a common mistake in liveness-detection tutorials found
online; don't repeat it.

## Debug visualization: make it impossible to miss on the first try

Three consecutive rounds were spent chasing "why can't I see the landmark
points / bounding box on screen" before it was resolved (root cause: an
initial version drew small, low-opacity points that were real but easy to
miss in a screenshot, compounded by giving `#video`/`#overlay` no explicit
`z-index` and relying on DOM order for stacking). Fixed by giving `#overlay`
an explicit higher `z-index` than `#video`, and using bold, fully-opaque,
appropriately-sized markers for anything meant to be visually verified.

**Lesson: debug visualizations should default to maximally obvious (bright,
opaque, large) on the first attempt, not "subtle so it doesn't distract" --
subtlety is an optimization for later, once you've confirmed the thing
actually renders at all.**

## `staff_photo/` test images are not verified ground truth

See `docs/FACE-COMPARE.md`'s caveat section. The folder used to test face
recognition turned out to be photos of a public figure across different
contexts, not confirmed-same/confirmed-different labeled pairs. Any
conclusion drawn from it about dlib vs SFace accuracy should be treated as
provisional until re-tested with verified ground truth.
