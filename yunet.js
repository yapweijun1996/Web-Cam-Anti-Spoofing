// YuNet face detector (2023mar ONNX export) decode logic, ported line-for-line
// from OpenCV's own C++ implementation: opencv/modules/objdetect/src/face_detect.cpp
// (FaceDetectorYNImpl::postProcess). Runs via onnxruntime-web, no OpenCV needed.
//
// This exact ONNX export has a FIXED [1,3,640,640] input (verified via
// `onnx.load` on the model file) -- unlike the C++ API's flexible setInputSize,
// so we always resize/stretch the source image to 640x640 and scale detections
// back to the original coordinate space afterward.
"use strict";

const YUNET_SIZE = 640;
const STRIDES = [8, 16, 32];

// Builds the NCHW input tensor OpenCV's blobFromImage would produce for this
// model: blobFromImage(pad_image) with default args -> scalefactor=1 (no /255),
// no mean subtraction, swapRB=false. OpenCV Mats are BGR internally and that
// default is left alone, so the model expects BGR order; our canvas ImageData
// is RGB, so we swap R and B channels here to match.
function canvasToYuNetTensor(canvas) {
  const resized = document.createElement("canvas");
  resized.width = YUNET_SIZE;
  resized.height = YUNET_SIZE;
  const resizedCtx = resized.getContext("2d", { willReadFrequently: true });
  resizedCtx.drawImage(canvas, 0, 0, YUNET_SIZE, YUNET_SIZE);
  const { data } = resizedCtx.getImageData(0, 0, YUNET_SIZE, YUNET_SIZE);

  const plane = YUNET_SIZE * YUNET_SIZE;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const o = i * 4;
    chw[i] = data[o + 2];               // B (swap: canvas is RGB, model wants BGR)
    chw[plane + i] = data[o + 1];       // G
    chw[2 * plane + i] = data[o];       // R
  }
  return new ort.Tensor("float32", chw, [1, 3, YUNET_SIZE, YUNET_SIZE]);
}

// outputs: the object returned by session.run(), keyed by tensor name.
// Returns faces in 640x640 coordinate space: {x,y,width,height,landmarks:[[x,y]x5],score}
function decodeYuNet(outputs, scoreThreshold) {
  const faces = [];
  for (let i = 0; i < STRIDES.length; i++) {
    const stride = STRIDES[i];
    const cols = YUNET_SIZE / stride;
    const rows = YUNET_SIZE / stride;

    const cls = outputs[`cls_${stride}`].data;
    const obj = outputs[`obj_${stride}`].data;
    const bbox = outputs[`bbox_${stride}`].data;
    const kps = outputs[`kps_${stride}`].data;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const idx = r * cols + c;
        const clsScore = Math.min(1, Math.max(0, cls[idx]));
        const objScore = Math.min(1, Math.max(0, obj[idx]));
        const score = Math.sqrt(clsScore * objScore);
        if (score < scoreThreshold) continue;

        const cx = (c + bbox[idx * 4 + 0]) * stride;
        const cy = (r + bbox[idx * 4 + 1]) * stride;
        const w = Math.exp(bbox[idx * 4 + 2]) * stride;
        const h = Math.exp(bbox[idx * 4 + 3]) * stride;
        const x = cx - w / 2;
        const y = cy - h / 2;

        const landmarks = [];
        for (let n = 0; n < 5; n++) {
          landmarks.push([
            (kps[idx * 10 + 2 * n] + c) * stride,
            (kps[idx * 10 + 2 * n + 1] + r) * stride,
          ]);
        }
        faces.push({ x, y, width: w, height: h, landmarks, score });
      }
    }
  }
  return faces;
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width), y2 = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

// Standard greedy NMS: highest score first, suppress others above nmsThreshold.
function nms(faces, nmsThreshold, topK) {
  const sorted = [...faces].sort((a, b) => b.score - a.score).slice(0, topK);
  const keep = [];
  for (const face of sorted) {
    if (keep.every((k) => iou(k, face) <= nmsThreshold)) keep.push(face);
  }
  return keep;
}

// Full pipeline: canvas -> best single face {x,y,width,height,landmarks,score}
// in the ORIGINAL canvas's coordinate space (not 640x640), or null.
async function detectFaceYuNet(session, canvas, scoreThreshold = 0.6, nmsThreshold = 0.3, topK = 50) {
  const tensor = canvasToYuNetTensor(canvas);
  const outputs = await session.run({ [session.inputNames[0]]: tensor });
  const faces = decodeYuNet(outputs, scoreThreshold);
  const kept = nms(faces, nmsThreshold, topK);
  if (!kept.length) return null;

  const best = kept.reduce((a, b) => (b.score > a.score ? b : a));
  const scaleX = canvas.width / YUNET_SIZE;
  const scaleY = canvas.height / YUNET_SIZE;
  return {
    x: best.x * scaleX,
    y: best.y * scaleY,
    width: best.width * scaleX,
    height: best.height * scaleY,
    landmarks: best.landmarks.map(([x, y]) => [x * scaleX, y * scaleY]),
    score: best.score,
  };
}

window.YuNet = { detectFaceYuNet, YUNET_SIZE };
