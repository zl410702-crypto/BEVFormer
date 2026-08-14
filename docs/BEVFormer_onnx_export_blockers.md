# BEVFormer ONNX Export Blockers

Target: static BEVFormer tiny Golden Sample, ONNX opset 13.

Command:

```bash
python tools/export_bevformer_onnx.py --token 3e8750f331d7499e9b5123e9eb70f2e2 --opset 13 --output artifacts/bevformer_tiny_opset13.onnx
```

Exporter started: **YES**

| Module | Source | PyTorch Op / Dependency | Export Error | In Compatibility Report | Status | Proposed Action |
| --- | --- | --- | --- | --- | --- | --- |
| — | — | — | No blockers observed | N/A | PASS | NATIVE |

## Interpretation

- Environment blockers prevent tracing and are not PyTorch operator exporter blockers.
- Operator blockers are added only after `torch.onnx.export` actually starts and raises an error.
- Entries marked `UNKNOWN/CHECK` must not be promoted without a real opset-13 export attempt.
