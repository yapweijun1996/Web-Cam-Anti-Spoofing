# Model attribution

`minifas_quantized.onnx` is copied from https://github.com/facenox/face-antispoof-onnx
(Apache-2.0, see MINIFAS_LICENSE.txt), a MiniFASNetV2-SE face anti-spoofing classifier
trained on CelebA-Spoof. That project itself builds on the MiniFAS architecture from
Minivision AI's Silent-Face-Anti-Spoofing (Apache-2.0).

- Input: 128x128 RGB face crop, CHW, float32 normalized to [0,1]
- Output: raw logits shape (1,2) -> [real_logit, spoof_logit]
- Decision: is_real = (real_logit - spoof_logit) >= logit_threshold
  where logit_threshold = ln(p/(1-p)) for a target probability threshold p (default p=0.5 -> threshold=0)
