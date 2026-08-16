# Supertonic-3 FP16 v2 Model Bundle

An FP16 ONNX model bundle for Supertonic-3 text-to-speech inference. The
artifacts are optimized for CPU-friendly ONNX Runtime use and include ten
precomputed voice styles.

This repository contains model files and configuration only. The Python
`SupertonicTTSV3` wrapper referenced in older examples is not included here;
use a compatible external inference wrapper or add one to this repository
before treating it as a standalone package.

## Contents

```text
onnx/
  vocoder.onnx
  vector_estimator.onnx
  text_encoder.onnx
  duration_predictor.onnx
  tts.json
  unicode_indexer.json
voice_styles/
  F1.json ... F5.json
  M1.json ... M5.json
```

The model files are approximately 201.6 MB in total. Audio output is configured
for 44.1 kHz in the published inference examples.

## Model details

- Vocoder, vector estimator, and text encoder: FP16
- Duration predictor: FP32
- Voice styles: five female and five male style embeddings
- Intended runtime: ONNX Runtime `>=1.20`
- Common wrapper dependencies: `numpy` and `transformers`

Static INT8 conversion was attempted for the vocoder and vector estimator but
did not pass quality validation, so this bundle remains the FP16 v2 baseline.

## Example wrapper shape

When using a compatible external wrapper, the expected call shape is:

```python
from helper import SupertonicTTSV3

model = SupertonicTTSV3("path/to/supertonic-3-fp16-v2", use_gpu=False)
wav, duration = model("Hello world.", "en", "F1", total_step=8)
```

The example assumes that `helper.py` is supplied by the wrapper project. It is
not an executable command from this repository as currently checked out.

## Artifact and license notes

Keep the ONNX weights and voice-style JSON files together; the configuration
references the matching model layout. Check the upstream model license and
redistribution terms before publishing a derived package or hosted endpoint.
