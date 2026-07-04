// Motion-parallax liveness layer: MediaPipe Face Landmarker (478 points) +
// a from-scratch homography fit to tell "flat surface" motion apart from
// real 3D head motion. ES module because @mediapipe/tasks-vision only ships
// ESM/CJS builds -- exposes window.ParallaxDetector so the classic-script
// app.js can call into it without itself becoming a module.
//
// Why homography residual, not the landmark z coordinate: MediaPipe's z is
// NOT a measured depth -- it's inferred by fitting a canonical 3D face model
// onto the 2D image, so it happily reports plausible-looking z values for a
// flat photo too. The only signal that can't be faked by a flat surface is
// the actual 2D motion field across frames: a real (non-planar) face cannot
// be exactly explained by a single homography (points closer to the camera,
// e.g. the nose tip, move more than points farther away, e.g. the ears);
// a flat photo/screen's entire point set moves through one shared homography
// with near-zero residual, no matter how "3D" its per-point z values look.

import { FaceLandmarker, FilesetResolver } from "./lib/mediapipe/vision_bundle.mjs";

// ---- homography fit: OpenCV.js RANSAC, hosted in a Web Worker ----
//
// Why RANSAC over plain least-squares: MediaPipe landmarks occasionally
// glitch on individual points (glasses glare, hair occlusion, blinking) --
// a few bad correspondences can drag a plain least-squares fit's residual
// up even for genuinely flat motion, or down even for genuinely 3D motion.
// RANSAC repeatedly samples subsets, keeps the homography with the most
// inliers, and we compute rmse only over THAT inlier set.
//
// Why a Worker: loading this OpenCV.js build on the main thread compiles
// its embedded WASM synchronously and froze the page for 30s+ (confirmed).
// The worker eats that cost off-thread; until it reports ready we fall back
// to the hand-rolled least-squares fit below, so the layer never blocks.

class OpenCvWorkerFitter {
  constructor() {
    this.worker = null;
    this.ready = false;
    this.failed = false;
    this.nextId = 1;
    this.pending = new Map(); // id -> {resolve}
  }

  start() {
    try {
      this.worker = new Worker("./opencv-worker.js");
    } catch (e) {
      this.failed = true;
      return;
    }
    this.worker.onmessage = (e) => {
      const { id, ok, warmedUp, rmse, inliers, total } = e.data;
      if (warmedUp) { this.ready = true; return; }
      const entry = this.pending.get(id);
      if (!entry) return;
      this.pending.delete(id);
      entry.resolve(ok ? { rmse, inliers, total } : null);
    };
    this.worker.onerror = () => {
      this.failed = true;
      this.ready = false;
      for (const { resolve } of this.pending.values()) resolve(null);
      this.pending.clear();
    };
    // Kick off the heavy load immediately so it warms up in parallel with
    // the other models instead of stalling the first real fit request.
    this.worker.postMessage({ id: 0, warmup: true });
  }

  fit(src, dst) {
    if (!this.ready || this.failed) return null; // caller falls back
    const id = this.nextId++;
    return new Promise((resolve) => {
      this.pending.set(id, { resolve });
      this.worker.postMessage({ id, src, dst });
    });
  }
}

// ---- fallback: pure least-squares via normal equations, no external lib ----
// (used only if OpenCV.js failed to load, so the layer degrades gracefully)

function accumulate(ATA, ATb, row, b) {
  for (let i = 0; i < 8; i++) {
    ATb[i] += row[i] * b;
    for (let j = 0; j < 8; j++) ATA[i][j] += row[i] * row[j];
  }
}

function solve8x8(A, b) {
  const n = 8;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < 1e-9) return null; // degenerate (e.g. no motion at all)
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= factor * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}

function fitHomographyLstsq(src, dst) {
  const ATA = Array.from({ length: 8 }, () => new Float64Array(8));
  const ATb = new Float64Array(8);
  for (let i = 0; i < src.length; i++) {
    const [x, y] = src[i];
    const [X, Y] = dst[i];
    accumulate(ATA, ATb, [x, y, 1, 0, 0, 0, -x * X, -y * X], X);
    accumulate(ATA, ATb, [0, 0, 0, x, y, 1, -x * Y, -y * Y], Y);
  }
  const h = solve8x8(ATA, ATb);
  if (!h) return null;
  const H = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  let sumSq = 0;
  for (let i = 0; i < src.length; i++) {
    const [x, y] = src[i];
    const [X, Y] = dst[i];
    const w = H[6] * x + H[7] * y + H[8];
    const px = (H[0] * x + H[1] * y + H[2]) / w;
    const py = (H[3] * x + H[4] * y + H[5]) / w;
    sumSq += (px - X) ** 2 + (py - Y) ** 2;
  }
  return { rmse: Math.sqrt(sumSq / src.length), inliers: src.length, total: src.length };
}

function meanDisplacement(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]);
  return sum / a.length;
}

// ---- stateful detector ----

class ParallaxDetector {
  constructor() {
    this.landmarker = null;
    this.fitter = new OpenCvWorkerFitter(); // RANSAC in a worker; lstsq fallback until ready
    this.ready = false;
    this.history = []; // rolling buffer: {t, points:[[x,y],...]}
    this.HISTORY_MS = 700; // compare current frame to ~700ms ago, not frame-to-frame
    // Confirmed by testing: at MOTION_MIN_PX=4, casual real-face motion (just
    // sitting/talking) triggered false "PLANAR/suspicious" verdicts twice in
    // a row. Real parallax (near points moving more than far points) is a
    // second-order effect of rotation angle -- small motions barely reveal
    // it even for a genuinely 3D face, so small motions look "flat" whether
    // real or fake. Raising the bar to require decisively larger motion
    // before trusting the residual at all.
    this.MOTION_MIN_PX = 18;
    this.PLANAR_RMSE_PX = 1.2; // below this reprojection error, the motion looks flat
    this.VOTE_WINDOW = 6;
    this.votes = [];
  }

  async init() {
    // Start the worker FIRST so OpenCV's heavy WASM compile warms up in
    // parallel while MediaPipe loads; neither blocks the main thread.
    this.fitter.start();
    const fileset = await FilesetResolver.forVisionTasks("./lib/mediapipe/wasm");
    this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "./models/face-landmarker/face_landmarker.task", delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
    });
    this.ready = true;
  }

  async evaluate(video, timestampMs) {
    if (!this.ready) return { status: "loading" };
    const result = this.landmarker.detectForVideo(video, timestampMs);
    if (!result.faceLandmarks.length) return { status: "no-face" };

    const points = result.faceLandmarks[0].map((p) => [p.x * video.videoWidth, p.y * video.videoHeight]);
    this.history.push({ t: timestampMs, points });
    while (this.history.length > 1 && timestampMs - this.history[0].t > this.HISTORY_MS) this.history.shift();

    if (this.history.length < 2) return { status: "collecting", points };

    const ref = this.history[0];
    const motion = meanDisplacement(ref.points, points);
    if (motion < this.MOTION_MIN_PX) return { status: "too-still", motion, points };

    // Prefer the worker's RANSAC fit; fall back to the in-thread
    // least-squares fit while the worker is still warming up (or if it died).
    const workerFit = this.fitter.fit(ref.points, points);
    const usingOpenCv = workerFit !== null;
    const fit = usingOpenCv ? await workerFit : fitHomographyLstsq(ref.points, points);
    this.usingOpenCv = usingOpenCv;
    if (!fit) return { status: "fit-failed", motion, points };

    const isPlanar = fit.rmse < this.PLANAR_RMSE_PX;
    this.votes.push(isPlanar);
    if (this.votes.length > this.VOTE_WINDOW) this.votes.shift();
    const planarCount = this.votes.filter(Boolean).length;
    const suspicious = this.votes.length >= 3 && planarCount / this.votes.length > 0.6;

    return {
      status: "evaluated",
      motion,
      rmse: fit.rmse,
      inliers: fit.inliers,
      totalPoints: fit.total,
      usingOpenCv: this.usingOpenCv,
      isPlanar,
      suspicious,
      voteText: `${planarCount}/${this.votes.length}`,
      points,
    };
  }
}

window.ParallaxDetector = new ParallaxDetector();
