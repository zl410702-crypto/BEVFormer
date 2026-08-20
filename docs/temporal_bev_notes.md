# BEVFormer tiny: three-frame temporal BEV reference

## Verified sequence

The experiment used one model instance and the normal `return_loss=False` path.
The samples are linked by reciprocal nuScenes `prev`/`next` fields, belong to
scene `fcbccedd61424f1b85dcbf8f897f9754`, and have increasing timestamps.

| Frame | Sample token | Timestamp | Translation delta (m) | Rotation delta (deg) |
|---|---|---:|---|---:|
| 0 | `3e8750f331d7499e9b5123e9eb70f2e2` | 1533151603547590 | `[0, 0, 0]` | 0 |
| 1 | `3950bd41f74548429c0f7700ff3d8269` | 1533151604048025 | `[3.7057065, -2.1037858, 0]` | -1.0353196 |
| 2 | `c5f58c19249d4137ae063b0e9ecd8b8e` | 1533151604547893 | `[3.6282059, -2.1720683, 0]` | -1.5740092 |

```text
Frame 0 images -> BEV_0 -> model.prev_frame_info['prev_bev']
                                      |
Frame 1 images + BEV_0 + ego delta <--+
                  -> temporal attention -> BEV_1 -> saved state
                                                   |
Frame 2 images + BEV_1 + ego delta <---------------+
                  -> temporal attention -> BEV_2
```

## Actual source logic

The owner of runtime state is
`projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`, class
`BEVFormer`, functions `__init__` and `forward_test`.
`prev_frame_info` contains `prev_bev`, `scene_token`, `prev_pos`, and
`prev_angle`. A new `scene_token` clears `prev_bev`. With
`video_test_mode=False` every call also clears it; tiny sets this option true.
Before inference, `forward_test` copies the absolute current `can_bus[:3]` and
`can_bus[-1]`. If history exists, it subtracts saved absolute position/yaw in
place; otherwise those delta fields become zero. After `simple_test`, it saves
the copied absolute pose and the returned `new_prev_bev`.

`simple_test_pts` obtains that tensor from `outs['bev_embed']`, so the stored
state is precisely the encoder final BEV, layout `[2500, 1, 256]`. The next
`forward_test` passes it through `simple_test` -> `simple_test_pts` ->
`BEVFormerHead.forward` -> `PerceptionTransformer.forward`.

In `projects/mmdet3d_plugin/bevformer/modules/transformer.py`, class
`PerceptionTransformer`, function `get_bev_features`, `can_bus[0:2]` supplies
translation delta, `can_bus[-2]` supplies ego heading (radians), and
`can_bus[-1]` supplies rotation delta (degrees). It computes translation length
and direction, then normalized BEV `shift_x/shift_y`. With
`rotate_prev_bev=True`, torchvision `rotate` rotates each `[256, 50, 50]`
history map around configured center `[100, 100]` before the encoder call.

`projects/mmdet3d_plugin/bevformer/modules/encoder.py`, class
`BEVFormerEncoder.forward`, converts aligned history to `[1,2500,256]`, stacks
it with current query and reshapes to TemporalSelfAttention value
`[2,2500,256]`. Its reference points likewise combine shifted history and
unshifted current coordinates into `[2,2500,1,2]`. All three encoder layers
receive the same two-entry value; each layer's current query changes.
Without history (Frame 0), encoder passes `None`; `TemporalSelfAttention.forward`
duplicates current query to effective value `[2,2500,256]`, preserving the same
operator shape.

## Tensor correspondence

`temporal/prev_bev_stored.npy` is cloned at entry to the next `forward_test`,
before rotation. Therefore Frame 0 final BEV equals Frame 1 stored history, and
Frame 1 final BEV equals Frame 2 stored history exactly: MAE, MaxAE, and RMSE
are 0; cosine similarity is 1. `prev_bev_aligned.npy` is captured at encoder
entry after rotate and is not expected to equal the stored tensor.

## Deployment state mapping

| PyTorch runtime state | TensorRT/C++ state |
|---|---|
| `prev_frame_info['prev_bev']` `[2500,1,256]` float32 | persistent previous encoder output, with documented layout |
| `scene_token` | stream/scene identifier used to reset history |
| `prev_pos` (three absolute values) | previous absolute translation |
| `prev_angle` (absolute degrees) | previous absolute yaw |

C++ must compute the same deltas, zero them on reset, reproduce shift and
nearest-neighbor rotation semantics, and update state only after successful
inference. Saving an aligned tensor instead of the raw final BEV would change
the next-frame behavior.

## Verified call chain

```text
tools/run_golden_sequence.py
 -> MMDataParallel.forward -> BEVFormer.forward
 -> BEVFormer.forward_test -> BEVFormer.simple_test
 -> BEVFormer.extract_feat -> ResNet.forward -> FPN.forward
 -> BEVFormer.simple_test_pts -> BEVFormerHead.forward
 -> PerceptionTransformer.forward -> get_bev_features
 -> BEVFormerEncoder.forward
 -> BEVFormerLayer.forward
 -> TemporalSelfAttention.forward
 -> SpatialCrossAttention.forward -> MSDeformableAttention3D.forward
 -> DetectionTransformerDecoder.forward
 -> BEVFormerHead.get_bboxes -> NMSFreeCoder.decode/decode_single
```

