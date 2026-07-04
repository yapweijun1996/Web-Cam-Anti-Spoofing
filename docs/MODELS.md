# Models used

All models run locally via WASM (onnxruntime-web, MediaPipe Tasks Vision, or
TensorFlow.js). Nothing is called over the network at inference time.

| File | Used by | Source | License |
|---|---|---|---|
| `models/minifas_quantized.onnx` | anti-spoof texture check | [facenox/face-antispoof-onnx](https://github.com/facenox/face-antispoof-onnx) (MiniFASNetV2-SE, trained on CelebA-Spoof) | Apache-2.0 (see `models/MINIFAS_LICENSE.txt`, `models/ATTRIBUTION.md`) |
| `models/coco-ssd/*` | anti-spoof device detection | `@tensorflow-models/coco-ssd` (`lite_mobilenet_v2` base), official TF.js model | Apache-2.0 |
| `models/face-detector/tiny_face_detector_*` | face-api.js detector (both tools) | [vladmandic/face-api](https://github.com/vladmandic/face-api) fork | MIT |
| `models/face-recognition/face_landmark_68_model.bin` | dlib pipeline (face-compare) | vladmandic/face-api fork | MIT |
| `models/face-recognition/face_recognition_model.bin` | dlib pipeline (face-compare) | vladmandic/face-api fork | MIT |
| `models/face-recognition/yunet.onnx` | YuNet detector (face-compare) | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo), `face_detection_yunet_2023mar.onnx` | Apache-2.0 |
| `models/face-recognition/sface.onnx` | SFace recognizer (face-compare) | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo), `face_recognition_sface_2021dec.onnx` (MobileFaceNet, SFace loss) | Apache-2.0 |
| `models/face-landmarker/face_landmarker.task` | motion parallax layer (anti-spoof) | Google MediaPipe, official model bundle | Apache-2.0 |

## Runtime libraries (`lib/`)

| File | Purpose | Notes |
|---|---|---|
| `ort.min.js` + `ort-wasm-simd-threaded.*` | onnxruntime-web | Forced single-threaded (`numThreads=1`) -- multi-threaded needs COOP/COEP headers a plain static server doesn't send |
| `face-api.js` | face-api.js (vladmandic fork) | Bundles its own TF.js copy -- causes harmless "kernel already registered" console warnings when `tf.min.js` is also loaded (filtered in `index.html`'s inline script) |
| `tf.min.js` | TensorFlow.js core | For `coco-ssd` |
| `coco-ssd.min.js` | COCO-SSD wrapper | |
| `mediapipe/vision_bundle.mjs` + `mediapipe/wasm/*` | MediaPipe Tasks Vision | ESM-only package (no UMD build), loaded via `<script type="module">` |
| `opencv.js` (+ `opencv.wasm`) | OpenCV.js (`@techstark/opencv-js` build) | **Must be loaded inside a Web Worker, never on the main thread** -- see `docs/DECISIONS.md` |

## Provenance

None of the above came from arbitrary third-party mirrors -- everything is
sourced directly from the official project repos (opencv_zoo, MediaPipe,
TensorFlow Models, vladmandic/face-api) or their official npm/jsDelivr
distribution. Model choice was driven entirely by "does the official repo
ship a permissively-licensed, browser-runnable artifact," not by
convenience or availability elsewhere.
