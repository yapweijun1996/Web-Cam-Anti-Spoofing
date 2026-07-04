// Web Worker host for OpenCV.js homography fitting.
//
// Why a worker at all: the @techstark/opencv-js build embeds its WASM binary
// inside the 13MB JS file and compiles it synchronously on load -- loading it
// on the main thread froze the entire page for 30s+ (confirmed: page became
// unresponsive to screenshots/eval until killed). In a worker, that same
// compile cost happens off-thread; the page stays interactive and the
// parallax layer just uses the hand-rolled least-squares fallback until the
// worker reports ready.
//
// Protocol: main thread posts {id, src, dst} (arrays of [x,y] pairs);
// worker replies {id, ok, rmse, inliers, total} or {id, ok:false, error}.
// A {id, warmup:true} message just forces the cv module to finish loading.

"use strict";

let cvReadyPromise = null;

function ensureCv() {
  if (cvReadyPromise) return cvReadyPromise;
  importScripts("lib/opencv.js"); // sync compile happens HERE, off the main thread
  const mod = self.cv;
  cvReadyPromise = Promise.resolve(
    mod instanceof Promise ? mod : mod && mod.Mat ? mod : new Promise((resolve) => {
      mod.onRuntimeInitialized = () => resolve(mod);
    })
  ).then((m) => {
    self.cv = m;
    return m;
  });
  return cvReadyPromise;
}

function fitHomographyRansac(cv, src, dst) {
  const n = src.length;
  const srcArr = new Float32Array(n * 2);
  const dstArr = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    srcArr[i * 2] = src[i][0]; srcArr[i * 2 + 1] = src[i][1];
    dstArr[i * 2] = dst[i][0]; dstArr[i * 2 + 1] = dst[i][1];
  }

  const srcMat = cv.matFromArray(n, 1, cv.CV_32FC2, srcArr);
  const dstMat = cv.matFromArray(n, 1, cv.CV_32FC2, dstArr);
  const mask = new cv.Mat();
  let H = null;
  try {
    H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 3, mask);
    if (!H || H.empty()) return null;

    const h = H.data64F; // row-major 3x3
    let sumSq = 0;
    let inliers = 0;
    for (let i = 0; i < n; i++) {
      if (mask.data[i] === 0) continue; // RANSAC-flagged outlier: excluded from residual
      const [x, y] = src[i];
      const [X, Y] = dst[i];
      const w = h[6] * x + h[7] * y + h[8];
      const px = (h[0] * x + h[1] * y + h[2]) / w;
      const py = (h[3] * x + h[4] * y + h[5]) / w;
      sumSq += (px - X) ** 2 + (py - Y) ** 2;
      inliers++;
    }
    if (inliers < 4) return null;
    return { rmse: Math.sqrt(sumSq / inliers), inliers, total: n };
  } finally {
    srcMat.delete(); dstMat.delete(); mask.delete();
    if (H) H.delete();
  }
}

self.onmessage = async (e) => {
  const { id, warmup, src, dst } = e.data;
  try {
    const cv = await ensureCv();
    if (warmup) {
      self.postMessage({ id, ok: true, warmedUp: true });
      return;
    }
    const fit = fitHomographyRansac(cv, src, dst);
    if (!fit) {
      self.postMessage({ id, ok: false, error: "degenerate fit / too few inliers" });
      return;
    }
    self.postMessage({ id, ok: true, rmse: fit.rmse, inliers: fit.inliers, total: fit.total });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err) });
  }
};
