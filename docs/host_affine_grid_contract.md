# BEVFormer Host Affine Grid Runtime Contract

The deployment-only stateful TensorRT path uses one additional runtime input.
Official BEVFormer inference and model mathematics are unchanged.

| Field | Contract |
| --- | --- |
| Name | `affine_grid` |
| Shape | `[1, 2500, 2]` |
| Dtype | FP32 |
| Layout | `[B, HW, XY]`, contiguous; XY order is x then y |
| Static/dynamic | Static for BEVFormer-tiny (`B=1`, `H=W=50`) |
| Producer | Host ego-motion preparation using `prepare_host_affine_grid()` |
| Consumer | ONNX `Reshape_666`, then the existing nearest-sampling graph |
| Semantic | PyTorch affine-grid coordinates before pixel scaling and `Round` |
| Ownership | Host owns the contiguous buffer until async H2D copy completes; TensorRT owns only its device binding buffer |

The engine retains coordinate conversion, rounding, validity masking, clipping,
linear-index generation, `GatherElements`, and zero-padding semantics. A
linear-index-only host contract is intentionally not used because it does not
carry the parallel validity mask.
