# BEVFormer tiny Golden Sample forward 调用链

本文对应仓库当前代码、`projects/configs/bevformer/bevformer_tiny.py` 和 Golden Sample
`3e8750f331d7499e9b5123e9eb70f2e2`。Tensor shape 来自一次真实 CUDA
运行生成的 `golden_samples/3e8750f331d7499e9b5123e9eb70f2e2/forward_trace.json`，不是按论文配置推测。

## 完整调用链

```text
tools/run_golden_sample.py: main
└── MMDataParallel(model)(return_loss=False, rescale=True, **batch)
    └── BEVFormer.forward
        └── BEVFormer.forward_test
            ├── scene_token 检查，首帧将 prev_bev 置为 None
            ├── CAN bus 变为相对 ego motion；无 prev_bev 时 delta 清零
            └── BEVFormer.simple_test  image_feature 提取
                ├── BEVFormer.extract_feat
                │   └── BEVFormer.extract_img_feat
                │       ├── GridMask 栅格随机遮挡，训练时候生效
                │       ├── ResNet.forward（6 相机作为 backbone batch）backbone推理
                │       ├── FPN.forward feature融合
                │       └── 恢复为 [batch, camera, channel, height, width]
                └── BEVFormer.simple_test_pts 
                    ├── BEVFormerHead.forward
                    │   ├── LearnedPositionalEncoding: (x,y) -> LearnedPositionalEncoding -> 256维度的position embedding
                    │   └── PerceptionTransformer.forward 
                    │       ├── PerceptionTransformer.get_bev_features
                    │       │   ├── CAN bus shift / 可选 prev_bev rotation
                    │       │   ├── camera/level embedding 和 image feature flatten(摊平): 组织特征信息,camera/level的信息嵌入到feature，明确相机来源，FPN层的来源。
                    │       │   └── BEVFormerEncoder.forward（3 layers）推理
                    │       │       ├── get_reference_points(3d / 2d) 
                    │       │       ├── point_sampling(lidar2img)
                    │       │       └── BEVFormerLayer.forward × 3
                    │       │           ├── TemporalSelfAttention.forward
                    │       │           ├── LayerNorm
                    │       │           ├── SpatialCrossAttention.forward
                    │       │           │   └── MSDeformableAttention3D.forward
                    │       │           ├── LayerNorm
                    │       │           ├── FFN
                    │       │           └── LayerNorm
                    │       └── DetectionTransformerDecoder.forward（6 layers）
                    │           ├── MultiheadAttention（object-query self attention）
                    │           ├── CustomMSDeformableAttention（对 bev_embed cross attention）
                    │           ├── FFN / Norm
                    │           └── reg_branches[lid]（逐层 reference refinement）
                    ├── BEVFormerHead.cls_branches / reg_branches（6 层输出）
                    ├── BEVFormerHead.get_bboxes
                    │   └── NMSFreeCoder.decode
                    │       └── decode_single
                    │           ├── sigmoid(class logits)
                    │           ├── 900 × 10 scores 全局 top-k(300)
                    │           ├── denormalize_bbox
                    │           └── post_center_range filter（无传统 NMS）
                    └── bbox3d2result
                        ├── boxes_3d:  (300, 9)
                        ├── scores_3d: (300,)
                        └── labels_3d: (300,)
```

`BEVFormerHead.forward` 先完整执行 encoder 和 decoder，再由 head 的 6 组 cls/reg
branch 生成最终 logits/box code。Decoder 内部还调用同一组 `reg_branches` 来逐层更新
reference point；这不是重复 decode。

## Golden Sample Tensor Shape 总表

记 `B=1`、`N=6` cameras、`C=256`、`H_bev=W_bev=50`、`Q_bev=2500`、
`Q_obj=900`、`L_enc=3`、`L_dec=6`、`D=4` pillar reference points。

| Stage | Variable | 真实 shape | 含义 |
| --- | --- | --- | --- |
| Pipeline | `img` | `(6, 3, 480, 800)` | 未显式写出 B 维的六相机图像 |
| Detector | `img` at `simple_test` | `(1, 6, 3, 480, 800)` | 显式 B=1、N=6；test augmentation 外层 list 已去除 |
| Feature extraction | `img` after squeeze | `(6, 3, 480, 800)` | batch=1 分支原地 squeeze，六相机作为 backbone batch |
| Backbone | ResNet input | `(6, 3, 480, 800)` | 六相机合并为 backbone batch |
| Backbone | C5 output | `(6, 2048, 15, 25)` | R50 stage 4，stride 32 |
| FPN | output level 0 | `(6, 256, 15, 25)` | tiny 只使用一个 feature level |
| Image feature | `mlvl_feats[0]` | `(1, 6, 256, 15, 25)` | 恢复 B/N 两维 |
| BEV query | embedding weight | `(2500, 256)` | `50 × 50` learned queries |
| BEV query | encoder input | `(2500, 1, 256)` | sequence-first |
| BEV position | `bev_pos` before flatten | `(1, 256, 50, 50)` | learned positional encoding |
| BEV position | encoder `bev_pos` | `(2500, 1, 256)` | flatten 后 sequence-first |
| Image flatten | `key`, `value` | `(6, 375, 1, 256)` | `375=15×25`，camera-first |
| Image levels | `spatial_shapes` | `(1, 2)`, value `[[15,25]]` | 一个 image feature level |
| Image levels | `level_start_index` | `(1,)`, value `[0]` | flattened level 起点 |
| BEV 3D refs | `ref_3d` | `(1, 4, 2500, 3)` | 每个 pillar 4 个高度点 |
| BEV 2D refs | `ref_2d` | `(1, 2500, 1, 2)` | TSA 平面 reference |
| Temporal | hybrid `reference_points` | `(2, 2500, 1, 2)` | queue=2；首帧是 current/current |
| Temporal | `prev_bev` | `None` | scene 第一帧没有历史 BEV |
| Temporal | TSA query | `(1, 2500, 256)` | batch-first BEV query |
| Temporal | internal value | `(2, 2500, 256)` | 首帧由当前 query 复制两份 |
| Temporal | offsets linear input/output | `(1,2500,512)` → `(1,2500,128)` | query 拼接后预测 offsets |
| Temporal | `sampling_offsets` reshaped | `(2,2500,8,1,4,2)` | queue×B, query, heads, levels, points, xy |
| Temporal | weights linear input/output | `(1,2500,512)` → `(1,2500,64)` | raw attention weights |
| Temporal | `attention_weights` reshaped | `(2,2500,8,1,4)` | queue×B, query, heads, levels, points |
| Temporal | TSA output | `(1, 2500, 256)` | 两个 queue slot 求均值并 residual add |
| Projection | six `lidar2img` | `6 × (4,4)` | 来自 `img_metas[0]` |
| Projection | `reference_points_cam` | `(6, 1, 2500, 4, 2)` | camera, B, BEV query, height point, xy |
| Projection | `bev_mask` | `(6, 1, 2500, 4)` | 每相机/高度点是否在图像内且深度为正 |
| Spatial | `queries_rebatch` | `(1, 6, 604, 256)` | 本样本各 camera 有效 query 补齐到动态 max=604 |
| Spatial | deform query | `(6, 604, 256)` | B 与 camera 合并 |
| Spatial | deform key/value | `(6, 375, 256)` | 每相机 image features |
| Spatial | deform reference points | `(6, 604, 4, 2)` | 对应 4 个 pillar 高度点 |
| Spatial | offsets linear output | `(6, 604, 128)` | reshape 为 `(6,604,8,1,8,2)` |
| Spatial | weights linear output | `(6, 604, 64)` | reshape 为 `(6,604,8,1,8)` |
| Spatial | SCA output | `(1, 2500, 256)` | 按 camera scatter-add，再按有效 camera 数平均 |
| Encoder | encoder output | `(1, 2500, 256)` | batch-first encoder result |
| Head/Transformer | returned `bev_embed` | `(2500, 1, 256)` | decoder 前在 transformer 中转为 sequence-first |
| Object query | `object_query_embeds` | `(900, 512)` | 256 query content + 256 query position |
| Decoder | query / query_pos | `(900, 1, 256)` each | 900 detection queries |
| Decoder | initial references | `(1, 900, 3)` | normalized x/y/z reference |
| Decoder | `inter_states` | `(6, 900, 1, 256)` | 6 decoder layers |
| Decoder | `inter_references` | `(6, 1, 900, 3)` | 每层 refinement 后 reference |
| Head | `hs` after permute | `(6, 1, 900, 256)` | 供每层 cls/reg branch 使用 |
| Head | `outputs_classes` | `(6, 1, 900, 10)` | 10 nuScenes class logits |
| Head | `outputs_coords` | `(6, 1, 900, 10)` | normalized box code，含 sin/cos yaw 和 vx/vy |
| Coder input | last layer cls/bbox | `(900,10)`, `(900,10)` | decode 只使用 `[-1]` decoder layer |
| Output | `boxes_3d.tensor` | `(300, 9)` | x,y,z,w,l,h,yaw,vx,vy |
| Output | `scores_3d` | `(300,)` | top-k sigmoid scores |
| Output | `labels_3d` | `(300,)` | class id, int64 |

## 关键函数

### 1. Golden Sample inference 入口

File: `tools/run_golden_sample.py`

Class / function: `main()`，模型调用位于 `model(return_loss=False, rescale=True, **batch)`。

Input: 官方 dataset test pipeline 产生的 `DataContainer`；pipeline 中当前帧 image 为
`(6,3,480,800)`，metadata 含六组 `lidar2img`、CAN bus、scene token 等。

Calls: `build_dataset`、`dataset[index]`、`collate`、`MMDataParallel`、模型 `forward`。

Output: 单元素 result list，后续提取 `result['pts_bbox']`。

Purpose: 复用当前 tiny config、dataset pipeline 和 checkpoint，只选择一个 token。

### 2. `BEVFormer.forward()` / `forward_test()`

File: `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`

Class: `BEVFormer`

Input: `return_loss=False`，`img` 和 `img_metas` 具有 test augmentation 外层 list。

Calls: `forward` dispatch 到 `forward_test`；`forward_test` 做 scene/temporal state 和
ego-motion delta 处理，再调用 `simple_test(img_metas[0], img[0], prev_bev=...)`。

Output: 当前 sample 的 bbox result list；同时将新的 `(2500,1,256)` BEV 保存到
`self.prev_frame_info['prev_bev']`，供同一 model 实例的下一帧使用。

Purpose: 测试态 dispatch 和跨帧 BEV 状态管理。

### 3. `BEVFormer.simple_test()` / `simple_test_pts()`

File: `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`

Input: 当前增强的 `img=(1,6,3,480,800)`、一个 sample 的 metadata list、`prev_bev=None`。

Calls: `extract_feat`；之后 `pts_bbox_head` 和 `pts_bbox_head.get_bboxes`。

Output: 新 `bev_embed=(2500,1,256)` 与 `pts_bbox` dictionary。

Purpose: 单 augmentation BEVFormer inference 主流程。

### 4. `BEVFormer.extract_img_feat()`

File: `projects/mmdet3d_plugin/bevformer/detectors/bevformer.py`

Input: 实测调用前为 `(1,6,3,480,800)`。函数先记录 `B=img.size(0)=1`，再在
batch=1 分支原地 squeeze 为 `(6,3,480,800)`；六相机随后作为 backbone batch。

Calls: `GridMask`、`img_backbone`、`img_neck`。

Output: `[mlvl_feats[0]]=[(1,6,256,15,25)]`。

Purpose: 提取每相机图像特征并恢复 B/camera 维。

### 5. `ResNet.forward()` / `FPN.forward()`

File: 环境中的 `mmdet/models/backbones/resnet.py` 与 `mmdet/models/necks/fpn.py`。

Class: MMDetection `ResNet`、`FPN`。

Input/output: ResNet `(6,3,480,800) → (6,2048,15,25)`；FPN
`(6,2048,15,25) → (6,256,15,25)`。

Purpose: R50 C5 feature 与单层、256-channel neck projection。

### 6. `BEVFormerHead.forward()`

File: `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`

Class: `BEVFormerHead`

Input: `mlvl_feats[0]=(1,6,256,15,25)`，`prev_bev=None`。

Calls: 创建 `(2500,256)` BEV embedding、`(900,512)` object-query embedding 和
`bev_pos=(1,256,50,50)`；调用 `PerceptionTransformer.forward`；对 6 层 `hs` 分别调用
cls/reg branch。

Output: `bev_embed=(2500,1,256)`、`all_cls_scores=(6,1,900,10)`、
`all_bbox_preds=(6,1,900,10)`。

Purpose: 组织 BEV encoder、object decoder 和最终 prediction branches。

### 7. `PerceptionTransformer.get_bev_features()`

File: `projects/mmdet3d_plugin/bevformer/modules/transformer.py`

Class: `PerceptionTransformer`

Input: image level `(1,6,256,15,25)`、BEV queries `(2500,256)`、
`bev_pos=(1,256,50,50)`、`prev_bev=None`。

Calls: 从 CAN bus 计算 `shift=(1,2)`；若有历史则旋转 `prev_bev`；加入 CAN bus、
camera 和 level embedding；flatten image features；调用 encoder。

Output: encoder batch-first BEV `(1,2500,256)`。

Purpose: 完成 ego-motion 对齐准备、image feature flatten 和 BEV encoder 调用。

### 8. `BEVFormerEncoder.forward()`

File: `projects/mmdet3d_plugin/bevformer/modules/encoder.py`

Class: `BEVFormerEncoder`

Input: `bev_query=(2500,1,256)`，key/value `(6,375,1,256)`，`prev_bev=None`。

Calls: `get_reference_points`、`point_sampling`，构造 temporal hybrid references，顺序
执行 3 个 `BEVFormerLayer`。

Output: `(1,2500,256)`。

Purpose: 用 temporal attention 和六相机 spatial attention 迭代增强 BEV grid。

### 9. `BEVFormerEncoder.get_reference_points()` / `point_sampling()`

File: `projects/mmdet3d_plugin/bevformer/modules/encoder.py`

Input/output: 3D refs `(1,4,2500,3)`；2D refs `(1,2500,1,2)`；投影后
`reference_points_cam=(6,1,2500,4,2)`、`bev_mask=(6,1,2500,4)`。

Calls: `point_sampling` 从每个 `img_meta['lidar2img']` 组装 `(B,N,4,4)` matrix，
把 normalized BEV xyz 反归一化到 point-cloud range，补齐齐次坐标，矩阵投影并按
`img_shape` 归一化 xy。

Purpose: 建立 BEV pillar 到六相机 image plane 的几何对应和有效性 mask。

`lidar2img` 最初在 `CustomNuScenesDataset.get_data_info()` 中由相机 intrinsic 和
sensor-to-lidar extrinsic 合成；`CustomCollect3D` 将其放进 `img_metas`，随后从
detector → head → transformer → encoder 原样透传。

### 10. `BEVFormerLayer.forward()`

File: `projects/mmdet3d_plugin/bevformer/modules/encoder.py`

Class: `BEVFormerLayer`

Input/output: 每层均为 `(1,2500,256)`。

Calls: 真实 `operation_order` 是
`('self_attn','norm','cross_attn','norm','ffn','norm')`，对应
`TemporalSelfAttention`、LN、`SpatialCrossAttention`、LN、MMCV FFN、LN。

Purpose: 一层先做 temporal fusion，再从当前六相机补充空间信息，最后 FFN。

因此“BEV query → temporal → spatial → FFN”的概括与当前代码一致；但本 Golden
Sample 没有历史信息，第一步实际是 current/current temporal self-attention，而不是
历史增强。

### 11. `TemporalSelfAttention.forward()`

File: `projects/mmdet3d_plugin/bevformer/modules/temporal_self_attention.py`

Class: `TemporalSelfAttention`

Input: query `(1,2500,256)`；从 layer 传入的 key/value 都是 `None`；hybrid
reference `(2,2500,1,2)`。

Calls: `value is None` 分支复制 query 得到 `(2,2500,256)`；`value[:B]` 与 query
拼接为 `(1,2500,512)`，预测 offsets/weights，调用 CUDA multi-scale deformable
attention；两个 queue slot 输出求均值，projection 和 residual add。

Output: `(1,2500,256)`。

Purpose: 融合一个历史 BEV 与当前 BEV；无历史时以两个当前 BEV slot 保持相同接口。

### 12. `SpatialCrossAttention.forward()`

File: `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`

Class: `SpatialCrossAttention`

Input: query `(1,2500,256)`，key/value `(6,375,1,256)`，camera refs
`(6,1,2500,4,2)`，mask `(6,1,2500,4)`。

Calls: 用 mask 找出每相机至少一个高度点有效的 query；本样本补齐到 max length
604；把 B/camera 合并后调用 `MSDeformableAttention3D`；再 scatter-add 回 2500
BEV slots，并按每个 slot 的有效 camera 数平均。

Output: `(1,2500,256)`。

Purpose: 每个 BEV query 只与能看到它的相机交互并聚合六相机 evidence。

### 13. `MSDeformableAttention3D.forward()`

File: `projects/mmdet3d_plugin/bevformer/modules/spatial_cross_attention.py`

Class: `MSDeformableAttention3D`

Input: query `(6,604,256)`，value `(6,375,256)`，reference
`(6,604,4,2)`，`spatial_shapes=[[15,25]]`。

Calls: 8 heads、1 level、配置的 8 total sampling points；按 4 个高度 anchor 重排
offsets 后调用 CUDA deformable attention。

Output: `(6,604,256)`。

Purpose: 在各相机 feature map 上围绕投影 reference points 做稀疏采样。

### 14. `PerceptionTransformer.forward()`

File: `projects/mmdet3d_plugin/bevformer/modules/transformer.py`

Input: encoder BEV `(1,2500,256)` 和 object embedding `(900,512)`。

Calls: object embedding 拆为 `query_pos/query=(900,1,256)`；生成
`init_reference=(1,900,3)`；将 BEV 转成 `(2500,1,256)` 后调用 decoder。

Output: `bev_embed=(2500,1,256)`、`inter_states=(6,900,1,256)`、
`init_reference=(1,900,3)`、`inter_references=(6,1,900,3)`。

Purpose: 串联 BEV encoder 与 detection decoder。

### 15. `DetectionTransformerDecoder.forward()`

File: `projects/mmdet3d_plugin/bevformer/modules/decoder.py`

Class: `DetectionTransformerDecoder`

Input: 900 object queries `(900,1,256)`、BEV value `(2500,1,256)`。

Calls: 6 个配置为 `DetrTransformerDecoderLayer` 的层；每层顺序同样是
self-attention → norm → cross-attention → norm → FFN → norm。这里 self-attention
是 MMCV `MultiheadAttention`，cross-attention 是当前文件中的
`CustomMSDeformableAttention`，不是 encoder 的 `SpatialCrossAttention`。每层之后
`reg_branches[lid]` 更新 normalized xyz reference。

Output: states `(6,900,1,256)`、references `(6,1,900,3)`。

Purpose: 让 900 object queries 在 BEV feature 上解码并逐层 refine reference。

### 16. `BEVFormerHead.get_bboxes()` / `NMSFreeCoder.decode_single()`

Files:

- `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head.py`
- `projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py`

Input: decode 只选择最后一层 `(1,900,10)` class logits 和 `(1,900,10)` box code。

Calls: 对 logits sigmoid，将 `900×10=9000` scores flatten 后直接
`topk(self.max_num)`；`label=index%10`、`bbox_index=index//10`；box code 经
`denormalize_bbox` 从 10 维（含 yaw sin/cos）变为 9 维；再做 center-range mask。

Output: `boxes=(300,9)`、`scores=(300,)`、`labels=(300,)`，之后包装为
`LiDARInstance3DBoxes`。

Purpose: NMS-free top-k decode。这里没有传统 NMS；`max_num=300` 来自 tiny config
的 `bbox_coder`，实际在 `NMSFreeCoder.decode_single()` 的 `topk(max_num)` 生效。
本样本 300 个候选全部通过 `post_center_range`，所以最终恰好为 300；如果 range mask
淘汰候选，最终数量可以少于 300。

## 首帧 `prev_bev` 的实际行为

1. 新建模型时 `prev_frame_info['scene_token']=None`、`prev_bev=None`。
2. `forward_test` 看到 Golden Sample scene token 与缓存不相同，再次明确清空历史。
3. 因没有历史，CAN bus translation delta 和 yaw delta 被设为 0。
4. `PerceptionTransformer.get_bev_features` 不执行历史 BEV rotate；encoder 收到
   `prev_bev=None`。
5. Encoder 仍执行全部 3 个 TSA。TSA 的 `value is None` fallback 将当前 query
   复制为两个 queue slots，并非跳过 TSA。
6. inference 完成后，新 `(2500,1,256)` BEV 被缓存；如果继续用同一 model 实例
   推理同 scene 下一帧，它才会成为真实历史 `prev_bev`。

所以本 Golden Sample **没有使用历史帧 BEV**，但 **执行了 TemporalSelfAttention**。

## 复现 trace

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

python tools/trace_bevformer_forward.py \
  --token 3e8750f331d7499e9b5123e9eb70f2e2
```

该脚本只在当前 Python 进程的 model instance 上包装方法并记录 shape，不修改核心
源码，也不调用 Golden Sample reference 保存逻辑。输出为
`golden_samples/3e8750f331d7499e9b5123e9eb70f2e2/forward_trace.json`。
