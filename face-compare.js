(function () {
  "use strict";

  const video = document.getElementById("video");
  const startBtn = document.getElementById("startBtn");
  const compareBtn = document.getElementById("compareBtn");
  const statusEl = document.getElementById("status");
  const resultTable = document.getElementById("resultTable");
  const resultBody = document.getElementById("resultBody");
  const resultNote = document.getElementById("resultNote");

  const SLOTS = ["A1", "A2", "B"];
  const slotCanvases = Object.fromEntries(SLOTS.map((k) => [k, document.getElementById("slot" + k)]));
  const slotData = {};

  let sfaceSession = null;
  let yunetSession = null;

  // ---- SFace preprocessing, per OpenCV's own C++ source (modules/objdetect/src/face_recognize.cpp):
  //   dnn::blobFromImage(aligned_img, /*scale*/1, Size(112,112), Scalar(0,0,0), /*swapRB*/true, /*crop*/false)
  // scale=1 means NO /255 normalization (raw 0-255 values); swapRB=true because OpenCV Mats are
  // BGR internally -- our canvas ImageData is already RGB, so we do NOT swap channels here.
  const SFACE_SIZE = 112;

  // Standard 112x112 ArcFace-style 5-point reference template (widely published,
  // used by insightface/SFace-style aligners). Order: left_eye, right_eye, nose,
  // left_mouth, right_mouth -- "left/right" meaning position in the OUTPUT image,
  // not the subject's anatomical left/right.
  const REF_5PT = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
  ];

  // Fits a 2D similarity transform T(x,y) = (a*x - b*y + tx, b*x + a*y + ty)
  // mapping src[i] -> dst[i] by least squares (linear in a,b,tx,ty -- no SVD needed).
  function fitSimilarity(src, dst) {
    // Normal equations for 4 unknowns [a, b, tx, ty].
    const ATA = Array.from({ length: 4 }, () => new Float64Array(4));
    const ATb = new Float64Array(4);
    const addRow = (row, rhs) => {
      for (let i = 0; i < 4; i++) {
        ATb[i] += row[i] * rhs;
        for (let j = 0; j < 4; j++) ATA[i][j] += row[i] * row[j];
      }
    };
    for (let i = 0; i < src.length; i++) {
      const [x, y] = src[i];
      const [X, Y] = dst[i];
      addRow([x, -y, 1, 0], X);
      addRow([y, x, 0, 1], Y);
    }
    // Solve via Gaussian elimination.
    const n = 4;
    const M = ATA.map((row, i) => [...row, ATb[i]]);
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
      [M[col], M[pivot]] = [M[pivot], M[col]];
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col] / M[col][col];
        for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
      }
    }
    const [a, b, tx, ty] = M.map((row, i) => row[n] / row[i]);
    return { a, b, tx, ty };
  }

  // Warps `sourceCanvas` so the face's 5 landmark points (from YuNet, order:
  // right eye, left eye, nose tip, right mouth corner, left mouth corner --
  // which is already position-for-position the same order as REF_5PT) land
  // on REF_5PT, cropped to a 112x112 canvas -- this is what SFace expects.
  function alignFace(sourceCanvas, points5) {
    const { a, b, tx, ty } = fitSimilarity(points5, REF_5PT);
    const out = document.createElement("canvas");
    out.width = SFACE_SIZE;
    out.height = SFACE_SIZE;
    const ctx = out.getContext("2d");
    // Canvas 2D transform matrix is [a, b, c, d, e, f] for
    // x' = a*x + c*y + e ; y' = b*x + d*y + f -- matches our T exactly.
    ctx.setTransform(a, b, -b, a, tx, ty);
    ctx.drawImage(sourceCanvas, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    return out;
  }

  function canvasToSFaceTensor(canvas) {
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    const { data } = ctx.getImageData(0, 0, SFACE_SIZE, SFACE_SIZE);
    const plane = SFACE_SIZE * SFACE_SIZE;
    const chw = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) {
      const o = i * 4;
      chw[i] = data[o];               // R -- raw 0-255, no /255 (scalefactor=1 in OpenCV's blobFromImage)
      chw[plane + i] = data[o + 1];   // G
      chw[2 * plane + i] = data[o + 2]; // B
    }
    return new ort.Tensor("float32", chw, [1, 3, SFACE_SIZE, SFACE_SIZE]);
  }

  async function loadSFace() {
    ort.env.wasm.wasmPaths = new URL("lib/", document.baseURI).toString();
    ort.env.wasm.numThreads = 1;
    sfaceSession = await ort.InferenceSession.create("models/face-recognition/sface.onnx", {
      executionProviders: ["wasm"],
    });
    yunetSession = await ort.InferenceSession.create("models/face-recognition/yunet.onnx", {
      executionProviders: ["wasm"],
    });
  }

  async function loadFaceApiModels() {
    await faceapi.nets.tinyFaceDetector.loadFromUri("models/face-detector");
    await faceapi.nets.faceLandmark68Net.loadFromUri("models/face-recognition");
    await faceapi.nets.faceRecognitionNet.loadFromUri("models/face-recognition");
  }

  // Models don't need any user permission (unlike the camera), so load them
  // immediately in the background -- this lets file-upload testing (e.g. an
  // existing staff photo folder) work without ever touching the webcam.
  const modelsReady = (async () => {
    statusEl.textContent = "Loading face-api.js models...";
    await loadFaceApiModels();
    statusEl.textContent = "Loading SFace + YuNet ONNX models...";
    await loadSFace();
    statusEl.textContent = "Models ready. Upload files or click Start camera.";
    return true;
  })().catch((err) => {
    statusEl.textContent = "Model load error: " + err.message;
    return false;
  });

  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    try {
      statusEl.textContent = "Waiting for models...";
      await modelsReady;

      statusEl.textContent = "Requesting camera...";
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
      video.srcObject = stream;
      await new Promise((resolve) => (video.onloadedmetadata = resolve));
      video.play();

      SLOTS.forEach((k) => (document.getElementById("capture" + k).disabled = false));
      statusEl.textContent = "Ready. Capture A1, A2 (same person), and B (different person).";
    } catch (err) {
      statusEl.textContent = "Error: " + err.message;
      startBtn.disabled = false;
    }
  });

  function markSlotFilled(k) {
    slotCanvases[k].classList.add("filled");
    slotData[k] = true;
    statusEl.textContent = `Loaded ${k}.`;
    if (SLOTS.every((s) => slotData[s])) compareBtn.disabled = false;
  }

  // Draws `source` (an <img> or <video>) into the slot canvas using
  // object-fit:cover-style scaling (crop to fill, preserve aspect ratio)
  // so uploaded photos of any size/shape don't get squashed.
  function drawCover(ctx, source, sw, sh, dw, dh) {
    const scale = Math.max(dw / sw, dh / sh);
    const w = sw * scale, h = sh * scale;
    ctx.drawImage(source, (dw - w) / 2, (dh - h) / 2, w, h);
  }

  SLOTS.forEach((k) => {
    document.getElementById("capture" + k).addEventListener("click", () => {
      const canvas = slotCanvases[k];
      drawCover(canvas.getContext("2d"), video, video.videoWidth, video.videoHeight, canvas.width, canvas.height);
      markSlotFilled(k);
    });

    document.getElementById("uploadBtn" + k).addEventListener("click", () => {
      document.getElementById("upload" + k).click();
    });

    document.getElementById("upload" + k).addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const img = new Image();
      const url = URL.createObjectURL(file);
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = url;
      });
      const canvas = slotCanvases[k];
      drawCover(canvas.getContext("2d"), img, img.naturalWidth, img.naturalHeight, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      markSlotFilled(k);
    });
  });

  function cosineSimilarity(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }

  async function extractDlib(canvas) {
    const withDesc = await faceapi
      .detectSingleFace(canvas, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceDescriptor();
    return withDesc ? withDesc.descriptor : null;
  }

  // Full "official" pipeline: YuNet detects the face + its own native 5-point
  // landmarks (already in the same left/right-eye, nose, mouth-corner order
  // SFace's alignCrop expects), we align to REF_5PT, then run SFace.
  async function extractSFace(canvas) {
    const face = await window.YuNet.detectFaceYuNet(yunetSession, canvas);
    if (!face) return null;
    const aligned = alignFace(canvas, face.landmarks);
    const tensor = canvasToSFaceTensor(aligned);
    const inputName = sfaceSession.inputNames[0];
    const outputs = await sfaceSession.run({ [inputName]: tensor });
    return outputs[sfaceSession.outputNames[0]].data; // Float32Array(128)
  }

  compareBtn.addEventListener("click", async () => {
    compareBtn.disabled = true;

    statusEl.textContent = "Waiting for models...";
    if (!(await modelsReady)) { compareBtn.disabled = false; return; }

    statusEl.textContent = "Running face-api.js (dlib)...";
    const dlib = {};
    for (const k of SLOTS) dlib[k] = await extractDlib(slotCanvases[k]);

    statusEl.textContent = "Running YuNet + SFace (ONNX via onnxruntime-web)...";
    const sface = {};
    for (const k of SLOTS) sface[k] = await extractSFace(slotCanvases[k]);

    statusEl.textContent = "Done.";
    const rows = [];

    if (dlib.A1 && dlib.A2 && dlib.B) {
      const same = faceapi.euclideanDistance(dlib.A1, dlib.A2);
      const diff = faceapi.euclideanDistance(dlib.A1, dlib.B);
      const margin = diff - same;
      rows.push({
        engine: "dlib (face-api.js, 现有系统)",
        sameLabel: same.toFixed(3) + " (dist)",
        diffLabel: diff.toFixed(3) + " (dist)",
        margin: margin.toFixed(3),
        good: margin > 0 && same < 0.6 && diff > 0.6,
      });
    } else {
      rows.push({ engine: "dlib (face-api.js)", sameLabel: "-", diffLabel: "-", margin: "face not detected", good: false });
    }

    if (sface.A1 && sface.A2 && sface.B) {
      const same = cosineSimilarity(sface.A1, sface.A2);
      const diff = cosineSimilarity(sface.A1, sface.B);
      const margin = same - diff;
      rows.push({
        engine: "YuNet + SFace (ONNX, 官方对齐流程)",
        sameLabel: same.toFixed(3) + " (cos)",
        diffLabel: diff.toFixed(3) + " (cos)",
        margin: margin.toFixed(3),
        good: margin > 0 && same > 0.363 && diff < 0.363,
      });
    } else {
      rows.push({ engine: "YuNet + SFace (ONNX)", sameLabel: "-", diffLabel: "-", margin: "face not detected (YuNet)", good: false });
    }

    resultBody.innerHTML = rows
      .map(
        (r) =>
          `<tr><td>${r.engine}</td><td>${r.sameLabel}</td><td>${r.diffLabel}</td>` +
          `<td class="${r.good ? "good" : "bad"}">${r.margin}</td></tr>`
      )
      .join("");
    resultTable.style.display = "";
    resultNote.textContent =
      "same-person 应该表示「很像」，diff-person 应该表示「不像」；margin 越大说明这套引擎把两种情况分得越开、越可靠。" +
      "dlib 用欧式距离（越小越像，好的 margin 是正数且 same<0.6<diff），SFace 用余弦相似度（越大越像，好的 margin 是正数且 same>0.363>diff）。" +
      "两者数值尺度不同，不能直接比大小，只看各自 margin 是否为正、是否明显。";

    compareBtn.disabled = false;
  });

  // Test hook: lets an external script (e.g. driven via browser automation)
  // batch-test arbitrary image URLs without going through the file-picker UI.
  window.FaceCompareTest = {
    modelsReady,
    async loadImageToCanvas(url, width = 220, height = 164) {
      const img = new Image();
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = url;
      });
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      drawCover(canvas.getContext("2d"), img, img.naturalWidth, img.naturalHeight, width, height);
      return canvas;
    },
    extractDlib,
    extractSFace,
    euclideanDistance: (a, b) => faceapi.euclideanDistance(a, b),
    cosineSimilarity,
  };
})();
