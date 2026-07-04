# antispoof-demo

Two standalone, pure-client-side (no LLM, no cloud API) research demos built to
answer concrete questions for the [Face-API-WASM](https://github.com/yapweijun1996/Face-API-WASM)
project: *"can we build face anti-spoofing without an LLM?"* and *"is face-api.js's
recognition model (dlib) as good as newer alternatives (SFace)?"*

Everything runs 100% in-browser via WASM (onnxruntime-web, MediaPipe Tasks Vision,
OpenCV.js). No servers, no API keys, no data leaves the machine.

## Pages

| Page | What it tests |
|---|---|
| `index.html` | Anti-spoofing: is the person in front of the camera real? |
| `face-compare.html` | Face recognition: dlib (face-api.js) vs YuNet+SFace (OpenCV model, run via onnxruntime-web) |

## Running

```bash
python3 -m http.server 8934   # any static file server works
```
Open `http://localhost:8934/index.html` or `/face-compare.html`. Camera access
requires `localhost` or `https` (browser security requirement).

## Docs

- [`docs/ANTISPOOF.md`](docs/ANTISPOOF.md) — the 4-layer anti-spoofing pipeline, why each layer exists, what it caught, what it missed
- [`docs/FACE-COMPARE.md`](docs/FACE-COMPARE.md) — dlib vs YuNet+SFace comparison tool, how it works, test results so far
- [`docs/MODELS.md`](docs/MODELS.md) — every model file used, where it came from, license
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — technical decisions and dead ends worth remembering (OpenCV.js freezing the main thread, YuNet's exact decode format, etc.)

## Status

Both tools are working and have been manually verified against a real webcam.
Neither is wired into a production decision yet -- see each doc's "Open
questions" section before relying on either for something real.
