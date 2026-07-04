(function () {
  "use strict";

  // ---- config (mirrors facenox/face-antispoof-onnx defaults) ----
  const MODEL_IMG_SIZE = 128;
  const BBOX_EXPANSION_FACTOR = 1.5;
  const DETECTOR_INPUT_SIZE = 320;
  const DETECTOR_SCORE_THRESHOLD = 0.5;

  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const octx = overlay.getContext("2d");

  // TEMP diagnostic: expose the exact objects this script draws with, so we
  // can check from the console whether they're really the same DOM node /
  // context the page currently shows (rules out a duplicate-element bug).
  window.__debugOverlay = overlay;
  window.__debugOctx = octx;
  window.__debugSameAsQueried = () => document.getElementById("overlay") === overlay;
  const startBtn = document.getElementById("startBtn");
  const resetStatsBtn = document.getElementById("resetStatsBtn");
  const statusEl = document.getElementById("status");
  const thresholdSlider = document.getElementById("thresholdSlider");
  const thresholdVal = document.getElementById("thresholdVal");
  const liveVerdictEl = document.getElementById("liveVerdict");
  const deviceHintEl = document.getElementById("deviceHint");
  const debugObjectsEl = document.getElementById("debugObjects");
  const parallaxDebugEl = document.getElementById("parallaxDebug");
  const realCountEl = document.getElementById("realCount");
  const spoofCountEl = document.getElementById("spoofCount");
  const spoofRateEl = document.getElementById("spoofRate");

  let session = null;
  let cocoModel = null;
  let running = false;
  let probThreshold = 0.5;
  let realCount = 0;
  let spoofCount = 0;

  // MiniFAS only looks at face-crop texture and can be fooled by a
  // high-quality phone/tablet screen showing a photo (confirmed by testing).
  // COCO-SSD adds a second, LLM-free signal: does the object detector see an
  // actual screen/device (cell phone, laptop, tv, book, remote) overlapping
  // the detected face? If so, reject regardless of what the texture model
  // says -- same "one clear spoof cue is enough" policy as the VLM prompt.
  const SPOOF_DEVICE_CLASSES = new Set(["cell phone", "laptop", "tv", "remote", "book"]);
  const DEVICE_SCORE_THRESHOLD = 0.4;

  // Per-frame device-detection confidence is noisy: the same physical phone
  // can score 86% in one frame and 9% the next as the hand/angle shifts by a
  // few degrees (confirmed by testing). Once a device is seen overlapping the
  // face, keep treating the session as suspicious for a few seconds instead
  // of trusting only the current frame -- a real physical object doesn't
  // vanish between frames, so one clean hit in the window is enough.
  const DEVICE_STICKY_MS = 4000;
  let lastDeviceSeenAt = 0;

  // Confirmed failure mode: holding a phone right up against the camera
  // pushes its bezel/edges out of frame, so COCO-SSD has no "phone-shaped"
  // silhouette to key off (max confidence seen: 10%) and MiniFAS still reads
  // clean texture at that distance/focus. Same gap the existing tms-app.js
  // FACE_FRAMING gate was built to close: a face filling most of the frame
  // is itself a suspicious framing (no real user sits that close to a kiosk
  // camera), so reject on geometry alone before either model even matters.
  const MAX_FACE_HEIGHT_RATIO = 0.62;
  const MAX_FACE_AREA_RATIO = 0.28;

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = MODEL_IMG_SIZE;
  cropCanvas.height = MODEL_IMG_SIZE;
  const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

  function logitThresholdFromProb(p) {
    const clamped = Math.min(1 - 1e-6, Math.max(1e-6, p));
    return Math.log(clamped / (1 - clamped));
  }

  thresholdSlider.addEventListener("input", () => {
    probThreshold = parseFloat(thresholdSlider.value);
    thresholdVal.textContent = probThreshold.toFixed(2);
  });

  resetStatsBtn.addEventListener("click", () => {
    realCount = 0;
    spoofCount = 0;
    updateStats();
  });

  function updateStats() {
    realCountEl.textContent = String(realCount);
    spoofCountEl.textContent = String(spoofCount);
    const total = realCount + spoofCount;
    spoofRateEl.textContent = total ? Math.round((spoofCount / total) * 100) + "%" : "0%";
  }

  async function loadModels() {
    statusEl.textContent = "Loading face detector...";
    await faceapi.nets.tinyFaceDetector.loadFromUri("models/face-detector");

    statusEl.textContent = "Loading MiniFAS ONNX model...";
    // Must be an absolute URL: onnxruntime-web dynamic-imports the .mjs glue
    // file using this prefix directly, and a plain relative string like
    // "lib/" is treated as a bare module specifier (needs an import map)
    // and fails to resolve.
    ort.env.wasm.wasmPaths = new URL("lib/", document.baseURI).toString();
    // Multi-threaded WASM needs SharedArrayBuffer, which needs COOP/COEP
    // response headers our plain static server doesn't send. Force the
    // single-threaded SIMD path instead (plenty fast for this 600KB model).
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create("models/minifas_quantized.onnx", {
      executionProviders: ["wasm"],
    });

    statusEl.textContent = "Loading COCO-SSD device detector...";
    cocoModel = await cocoSsd.load({
      base: "lite_mobilenet_v2",
      modelUrl: new URL("models/coco-ssd/model.json", document.baseURI).toString(),
    });

    statusEl.textContent = "Loading MediaPipe Face Landmarker (parallax layer)...";
    await window.ParallaxDetector.init();

    statusEl.textContent = "Models loaded.";
  }

  function rectsOverlap(a, b) {
    return a.x < b.x + b.width && a.x + a.width > b.x &&
           a.y < b.y + b.height && a.y + a.height > b.y;
  }

  // COCO-SSD gives [x, y, width, height] per prediction; face-api.js boxes
  // are already {x, y, width, height} -- normalize to the same shape.
  async function detectSpoofDevice(faceBox) {
    // maxNumBoxes/minScore default to library values (score 0.5, 20 boxes);
    // widen minScore here so the debug readout shows low-confidence misses too.
    const predictions = await cocoModel.detect(video, 20, 0.05);
    let result = { detected: false };
    for (const p of predictions) {
      if (SPOOF_DEVICE_CLASSES.has(p.class) && p.score >= DEVICE_SCORE_THRESHOLD) {
        const [x, y, w, h] = p.bbox;
        if (rectsOverlap(faceBox, { x, y, width: w, height: h })) {
          result = { detected: true, className: p.class, score: p.score, bbox: { x, y, width: w, height: h } };
          break;
        }
      }
    }
    result.allPredictions = predictions; // raw, for the debug readout
    return result;
  }

  // Square crop centered on the detection box, expanded by BBOX_EXPANSION_FACTOR.
  // Simplification vs. the original repo: instead of reflection-padding pixels
  // that fall outside the frame, we shift the crop window to stay inside the
  // frame bounds (only matters when a face is very close to the frame edge).
  function squareCropRect(box, frameW, frameH) {
    const maxDim = Math.max(box.width, box.height);
    let size = Math.round(maxDim * BBOX_EXPANSION_FACTOR);
    size = Math.min(size, Math.min(frameW, frameH));

    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;
    let x0 = Math.round(cx - size / 2);
    let y0 = Math.round(cy - size / 2);

    x0 = Math.max(0, Math.min(x0, frameW - size));
    y0 = Math.max(0, Math.min(y0, frameH - size));

    return { x: x0, y: y0, size };
  }

  function preprocessToTensor(rect) {
    cropCtx.drawImage(
      video,
      rect.x, rect.y, rect.size, rect.size,
      0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE
    );
    const { data } = cropCtx.getImageData(0, 0, MODEL_IMG_SIZE, MODEL_IMG_SIZE);

    // RGBA -> CHW RGB, normalized to [0,1]
    const chw = new Float32Array(3 * MODEL_IMG_SIZE * MODEL_IMG_SIZE);
    const plane = MODEL_IMG_SIZE * MODEL_IMG_SIZE;
    for (let i = 0; i < plane; i++) {
      const o = i * 4;
      chw[i] = data[o] / 255;                 // R
      chw[plane + i] = data[o + 1] / 255;      // G
      chw[2 * plane + i] = data[o + 2] / 255;  // B
    }
    return new ort.Tensor("float32", chw, [1, 3, MODEL_IMG_SIZE, MODEL_IMG_SIZE]);
  }

  async function classify(rect) {
    const tensor = preprocessToTensor(rect);
    const inputName = session.inputNames[0];
    const outputs = await session.run({ [inputName]: tensor });
    const logits = outputs[session.outputNames[0]].data; // [real_logit, spoof_logit]

    const realLogit = logits[0];
    const spoofLogit = logits[1];
    const diff = realLogit - spoofLogit;
    const logitThreshold = logitThresholdFromProb(probThreshold);
    const isReal = diff >= logitThreshold;

    return { isReal, confidence: Math.abs(diff), realLogit, spoofLogit };
  }

  // video uses object-fit:cover, so it's scaled UNIFORMLY (preserving aspect
  // ratio) then center-cropped -- not stretched independently per axis. Using
  // separate scaleX/scaleY (as if it were object-fit:fill) misplaces the box.
  function videoToOverlayTransform() {
    const scale = Math.max(
      overlay.width / video.videoWidth,
      overlay.height / video.videoHeight
    );
    const offsetX = (overlay.width - video.videoWidth * scale) / 2;
    const offsetY = (overlay.height - video.videoHeight * scale) / 2;
    return { scale, offsetX, offsetY };
  }

  function drawBox(box, verdict) {
    const { scale, offsetX, offsetY } = videoToOverlayTransform();
    const x = box.x * scale + offsetX;
    const y = box.y * scale + offsetY;
    const w = box.width * scale;
    const h = box.height * scale;

    octx.lineWidth = 3;
    octx.strokeStyle = verdict.isReal ? "#3fbf5f" : "#ff5050";
    octx.strokeRect(x, y, w, h);

    const label = `${verdict.isReal ? "REAL" : "SPOOF"} (${verdict.confidence.toFixed(2)})`;
    octx.font = "16px -apple-system, sans-serif";
    const textW = octx.measureText(label).width + 10;
    octx.fillStyle = verdict.isReal ? "#3fbf5f" : "#ff5050";
    octx.fillRect(x, Math.max(0, y - 24), textW, 22);
    octx.fillStyle = "#000";
    octx.fillText(label, x + 5, Math.max(16, y - 7));
  }

  function drawDeviceBox(bbox, className) {
    const { scale, offsetX, offsetY } = videoToOverlayTransform();
    const x = bbox.x * scale + offsetX;
    const y = bbox.y * scale + offsetY;
    const w = bbox.width * scale;
    const h = bbox.height * scale;

    octx.setLineDash([6, 4]);
    octx.lineWidth = 2;
    octx.strokeStyle = "#fbbf24";
    octx.strokeRect(x, y, w, h);
    octx.setLineDash([]);

    octx.font = "13px -apple-system, sans-serif";
    octx.fillStyle = "#fbbf24";
    octx.fillText(className, x + 4, y + h - 6);
  }

  function drawParallaxPoints(parallax) {
    if (!parallax || !parallax.points) return;
    const { scale, offsetX, offsetY } = videoToOverlayTransform();
    const color =
      parallax.status !== "evaluated" ? "#00e5ff" : // bright cyan, fully opaque -- was too-subtle light gray before
      parallax.isPlanar ? "#ff5050" : "#3fbf5f";
    const radius = Math.max(3, 3.5 * scale); // scale with video->screen zoom so points stay visible
    octx.fillStyle = color;
    octx.strokeStyle = "#000";
    octx.lineWidth = 1.5;
    for (const [x, y] of parallax.points) {
      const sx = x * scale + offsetX;
      const sy = y * scale + offsetY;
      octx.beginPath();
      octx.arc(sx, sy, radius, 0, Math.PI * 2);
      octx.fill();
      octx.stroke();
    }
  }

  function formatParallaxDebug(p) {
    if (!p) return "-";
    const minPx = window.ParallaxDetector.MOTION_MIN_PX;
    let line;
    if (p.status === "loading") line = "loading model...";
    else if (p.status === "no-face") line = "no-face (landmarker)";
    else if (p.status === "collecting") line = "collecting reference frame...";
    else if (p.status === "too-still") line = `too-still: motion=${p.motion.toFixed(1)}px (need >=${minPx}px)`;
    else if (p.status === "fit-failed") line = `fit-failed: motion=${p.motion.toFixed(1)}px (degenerate)`;
    else if (p.status === "evaluated") {
      line = `motion=${p.motion.toFixed(1)}px rmse=${p.rmse.toFixed(2)}px ` +
        `${p.isPlanar ? "PLANAR(flat?)" : "NON-PLANAR(3D?)"} votes=${p.voteText} ` +
        `${p.suspicious ? "[SUSPICIOUS]" : ""}\n` +
        `fit engine: ${p.usingOpenCv ? "OpenCV.js (RANSAC)" : "hand-rolled least-squares (fallback)"} ` +
        `inliers=${p.inliers}/${p.totalPoints}`;
    } else line = p.status;

    // Raw point sanity check: print landmark #1 (a point near the nose) in
    // both MediaPipe video-space and on-screen canvas-space coordinates, so
    // we can tell "points are computed but drawn somewhere off-screen" apart
    // from "points are genuinely not being produced".
    if (p.points && p.points.length) {
      const { scale, offsetX, offsetY } = videoToOverlayTransform();
      const [vx, vy] = p.points[1];
      const sx = vx * scale + offsetX;
      const sy = vy * scale + offsetY;
      line += `\npoint#1: video=(${vx.toFixed(0)},${vy.toFixed(0)}) screen=(${sx.toFixed(0)},${sy.toFixed(0)}) ` +
        `overlay=${overlay.width}x${overlay.height} n=${p.points.length}`;
    }
    return line;
  }

  function resizeOverlay() {
    overlay.width = overlay.clientWidth;
    overlay.height = overlay.clientHeight;
  }

  async function loop() {
    if (!running) return;
    octx.clearRect(0, 0, overlay.width, overlay.height);

    if (video.readyState === video.HAVE_ENOUGH_DATA) {
      const detection = await faceapi.detectSingleFace(
        video,
        new faceapi.TinyFaceDetectorOptions({
          inputSize: DETECTOR_INPUT_SIZE,
          scoreThreshold: DETECTOR_SCORE_THRESHOLD,
        })
      );

      if (detection) {
        const box = detection.box;
        const heightRatio = box.height / video.videoHeight;
        const areaRatio = (box.width * box.height) / (video.videoWidth * video.videoHeight);
        const tooClose = heightRatio > MAX_FACE_HEIGHT_RATIO || areaRatio > MAX_FACE_AREA_RATIO;

        const rect = squareCropRect(box, video.videoWidth, video.videoHeight);
        const [textureVerdict, deviceVerdict] = await Promise.all([
          classify(rect),
          detectSpoofDevice(box),
        ]);

        const now = performance.now();
        if (deviceVerdict.detected) lastDeviceSeenAt = now;
        const deviceSticky = now - lastDeviceSeenAt < DEVICE_STICKY_MS;

        // Three independent vetoes, any one is enough to reject:
        // 1. texture model reads the face crop as spoof
        // 2. a screen/device object was seen overlapping the face recently
        // 3. framing itself is suspicious (face fills too much of the frame,
        //    which is how a phone held against the lens defeats #1 and #2 --
        //    neither model has a clean face-plus-context view to judge)
        const isReal = textureVerdict.isReal && !deviceSticky && !tooClose;
        const finalVerdict = { isReal, confidence: textureVerdict.confidence };

        drawBox(box, finalVerdict);

        // TEMP: synchronous self-check, zero time gap possible -- reads the
        // canvas back immediately after drawBox() in the exact same tick.
        // If this ever logs 0, the draw call itself is not landing in the
        // backing store (not a timing/race issue with external inspection).
        {
          const { scale, offsetX, offsetY } = videoToOverlayTransform();
          const checkX = Math.round(box.x * scale + offsetX);
          const checkY = Math.round(box.y * scale + offsetY);
          const checkData = octx.getImageData(
            Math.max(0, checkX - 2), Math.max(0, checkY - 2), 5, 5
          ).data;
          let checkNonZero = 0;
          for (let i = 3; i < checkData.length; i += 4) if (checkData[i] > 0) checkNonZero++;
          console.log(`[selfcheck] drawBox at (${checkX},${checkY}) -> nonTransparentInSample=${checkNonZero}/25`);
        }

        if (deviceVerdict.detected) drawDeviceBox(deviceVerdict.bbox, deviceVerdict.className);
        liveVerdictEl.textContent =
          `${isReal ? "REAL" : "SPOOF"} | texture: real=${textureVerdict.realLogit.toFixed(2)} spoof=${textureVerdict.spoofLogit.toFixed(2)}`;
        liveVerdictEl.style.color = isReal ? "#3fbf5f" : "#ff5050";

        const stickySecsLeft = ((DEVICE_STICKY_MS - (now - lastDeviceSeenAt)) / 1000).toFixed(1);
        const hints = [];
        if (tooClose) hints.push(`⚠ face fills ${Math.round(heightRatio * 100)}% of frame height -- step back`);
        if (deviceVerdict.detected) hints.push(`⚠ device seen: ${deviceVerdict.className} (${deviceVerdict.score.toFixed(2)})`);
        else if (deviceSticky) hints.push(`⚠ device seen recently (holding for ${stickySecsLeft}s more)`);
        deviceHintEl.textContent = hints.join(" | ");

        const geomLine =
          `face box: ${Math.round(box.width)}x${Math.round(box.height)} ` +
          `in video ${video.videoWidth}x${video.videoHeight} ` +
          `-> heightRatio=${(heightRatio * 100).toFixed(0)}% areaRatio=${(areaRatio * 100).toFixed(0)}% ` +
          `${tooClose ? "(TOO CLOSE)" : "(ok)"}`;
        const objLines = deviceVerdict.allPredictions.length
          ? deviceVerdict.allPredictions.map((p) => `${p.class} ${(p.score * 100).toFixed(0)}%`).join("\n")
          : "(no objects detected at all)";
        debugObjectsEl.textContent = geomLine + "\n" + objLines;

        // Motion-parallax layer: debug-only for now, not wired into isReal
        // yet. Draws the 478 tracked points and prints the raw numbers so
        // the fit can be sanity-checked against real vs. flat-photo motion
        // before trusting it as a veto.
        const parallax = await window.ParallaxDetector.evaluate(video, now);
        drawParallaxPoints(parallax);
        parallaxDebugEl.textContent = formatParallaxDebug(parallax);

        window.__debugFrameStats.hit++;
        if (isReal) realCount++; else spoofCount++;
        updateStats();
      } else {
        window.__debugFrameStats.miss++;
        liveVerdictEl.textContent = "no face detected";
        liveVerdictEl.style.color = "#ccc";
        deviceHintEl.textContent = "";
      }
    }

    requestAnimationFrame(loop);
  }

  window.__debugFrameStats = { hit: 0, miss: 0 };

  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    try {
      await loadModels();

      statusEl.textContent = "Requesting camera access...";
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
      video.srcObject = stream;
      await new Promise((resolve) => (video.onloadedmetadata = resolve));
      video.play();
      resizeOverlay();
      window.addEventListener("resize", resizeOverlay);

      statusEl.textContent = "Running. Hold a phone photo / printed photo up to test spoof detection.";
      running = true;
      loop();
    } catch (err) {
      statusEl.textContent = "Error: " + err.message;
      startBtn.disabled = false;
    }
  });
})();
