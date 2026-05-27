# Supertonic-3 FP16 v2

Supertonic-3 TTS ONNX runtime package — FP16 precision, CPU-optimized.

## Contents

```
onnx/
  vocoder.onnx            50.8 MB  (FP16)
  vector_estimator.onnx  128.5 MB  (FP16)
  text_encoder.onnx       18.6 MB  (FP16)
  duration_predictor.onnx  3.7 MB  (FP32)
  tts.json                         (config)
  unicode_indexer.json             (text processor)
voice_styles/
  F1–F5.json, M1–M5.json          (10 voices)
```

**Total:** 201.6 MB (models only) / ~196 MB (full package)

## Usage

```python
from helper import SupertonicTTSV3

model = SupertonicTTSV3("path/to/supertonic-3-fp16-v2", use_gpu=False)

wav, duration = model("Hello world.", "en", "F1", total_step=8)
# wav: (1, samples) float32 numpy array at 44100 Hz
```

## Dependencies

- onnxruntime >= 1.20
- transformers (tokenizer only)
- numpy

## Precision

All models are FP16 except `duration_predictor` (FP32, too small to benefit).
INT8 static quantization was attempted for the vocoder and vector estimator but
failed quality validation — this package remains at the FP16 v2 baseline.
