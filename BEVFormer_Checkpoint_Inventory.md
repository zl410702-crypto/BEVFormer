# BEVFormer-tiny Checkpoint 完整清单

来源：只读加载 `/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth` 的 `state_dict`。`Kind` 依据旧模型 `named_parameters()` / `named_buffers()` 的实际分类核对；本 Checkpoint 中所有 Buffer 均为 BatchNorm running statistics / batch counter。

| # | Checkpoint key | Shape | Dtype | Kind |
|---:|---|---|---|---|
| 1 | `pts_bbox_head.code_weights` | `[10]` | `float32` | Parameter |
| 2 | `pts_bbox_head.positional_encoding.row_embed.weight` | `[50, 128]` | `float32` | Parameter |
| 3 | `pts_bbox_head.positional_encoding.col_embed.weight` | `[50, 128]` | `float32` | Parameter |
| 4 | `pts_bbox_head.transformer.level_embeds` | `[4, 256]` | `float32` | Parameter |
| 5 | `pts_bbox_head.transformer.cams_embeds` | `[6, 256]` | `float32` | Parameter |
| 6 | `pts_bbox_head.transformer.encoder.layers.0.attentions.0.sampling_offsets.weight` | `[128, 512]` | `float32` | Parameter |
| 7 | `pts_bbox_head.transformer.encoder.layers.0.attentions.0.sampling_offsets.bias` | `[128]` | `float32` | Parameter |
| 8 | `pts_bbox_head.transformer.encoder.layers.0.attentions.0.attention_weights.weight` | `[64, 512]` | `float32` | Parameter |
| 9 | `pts_bbox_head.transformer.encoder.layers.0.attentions.0.attention_weights.bias` | `[64]` | `float32` | Parameter |
| 10 | `pts_bbox_head.transformer.encoder.layers.0.attentions.0.value_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 11 | `pts_bbox_head.transformer.encoder.layers.0.attentions.0.value_proj.bias` | `[256]` | `float32` | Parameter |
| 12 | `pts_bbox_head.transformer.encoder.layers.0.attentions.0.output_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 13 | `pts_bbox_head.transformer.encoder.layers.0.attentions.0.output_proj.bias` | `[256]` | `float32` | Parameter |
| 14 | `pts_bbox_head.transformer.encoder.layers.0.attentions.1.deformable_attention.sampling_offsets.weight` | `[128, 256]` | `float32` | Parameter |
| 15 | `pts_bbox_head.transformer.encoder.layers.0.attentions.1.deformable_attention.sampling_offsets.bias` | `[128]` | `float32` | Parameter |
| 16 | `pts_bbox_head.transformer.encoder.layers.0.attentions.1.deformable_attention.attention_weights.weight` | `[64, 256]` | `float32` | Parameter |
| 17 | `pts_bbox_head.transformer.encoder.layers.0.attentions.1.deformable_attention.attention_weights.bias` | `[64]` | `float32` | Parameter |
| 18 | `pts_bbox_head.transformer.encoder.layers.0.attentions.1.deformable_attention.value_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 19 | `pts_bbox_head.transformer.encoder.layers.0.attentions.1.deformable_attention.value_proj.bias` | `[256]` | `float32` | Parameter |
| 20 | `pts_bbox_head.transformer.encoder.layers.0.attentions.1.output_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 21 | `pts_bbox_head.transformer.encoder.layers.0.attentions.1.output_proj.bias` | `[256]` | `float32` | Parameter |
| 22 | `pts_bbox_head.transformer.encoder.layers.0.ffns.0.layers.0.0.weight` | `[512, 256]` | `float32` | Parameter |
| 23 | `pts_bbox_head.transformer.encoder.layers.0.ffns.0.layers.0.0.bias` | `[512]` | `float32` | Parameter |
| 24 | `pts_bbox_head.transformer.encoder.layers.0.ffns.0.layers.1.weight` | `[256, 512]` | `float32` | Parameter |
| 25 | `pts_bbox_head.transformer.encoder.layers.0.ffns.0.layers.1.bias` | `[256]` | `float32` | Parameter |
| 26 | `pts_bbox_head.transformer.encoder.layers.0.norms.0.weight` | `[256]` | `float32` | Parameter |
| 27 | `pts_bbox_head.transformer.encoder.layers.0.norms.0.bias` | `[256]` | `float32` | Parameter |
| 28 | `pts_bbox_head.transformer.encoder.layers.0.norms.1.weight` | `[256]` | `float32` | Parameter |
| 29 | `pts_bbox_head.transformer.encoder.layers.0.norms.1.bias` | `[256]` | `float32` | Parameter |
| 30 | `pts_bbox_head.transformer.encoder.layers.0.norms.2.weight` | `[256]` | `float32` | Parameter |
| 31 | `pts_bbox_head.transformer.encoder.layers.0.norms.2.bias` | `[256]` | `float32` | Parameter |
| 32 | `pts_bbox_head.transformer.encoder.layers.1.attentions.0.sampling_offsets.weight` | `[128, 512]` | `float32` | Parameter |
| 33 | `pts_bbox_head.transformer.encoder.layers.1.attentions.0.sampling_offsets.bias` | `[128]` | `float32` | Parameter |
| 34 | `pts_bbox_head.transformer.encoder.layers.1.attentions.0.attention_weights.weight` | `[64, 512]` | `float32` | Parameter |
| 35 | `pts_bbox_head.transformer.encoder.layers.1.attentions.0.attention_weights.bias` | `[64]` | `float32` | Parameter |
| 36 | `pts_bbox_head.transformer.encoder.layers.1.attentions.0.value_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 37 | `pts_bbox_head.transformer.encoder.layers.1.attentions.0.value_proj.bias` | `[256]` | `float32` | Parameter |
| 38 | `pts_bbox_head.transformer.encoder.layers.1.attentions.0.output_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 39 | `pts_bbox_head.transformer.encoder.layers.1.attentions.0.output_proj.bias` | `[256]` | `float32` | Parameter |
| 40 | `pts_bbox_head.transformer.encoder.layers.1.attentions.1.deformable_attention.sampling_offsets.weight` | `[128, 256]` | `float32` | Parameter |
| 41 | `pts_bbox_head.transformer.encoder.layers.1.attentions.1.deformable_attention.sampling_offsets.bias` | `[128]` | `float32` | Parameter |
| 42 | `pts_bbox_head.transformer.encoder.layers.1.attentions.1.deformable_attention.attention_weights.weight` | `[64, 256]` | `float32` | Parameter |
| 43 | `pts_bbox_head.transformer.encoder.layers.1.attentions.1.deformable_attention.attention_weights.bias` | `[64]` | `float32` | Parameter |
| 44 | `pts_bbox_head.transformer.encoder.layers.1.attentions.1.deformable_attention.value_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 45 | `pts_bbox_head.transformer.encoder.layers.1.attentions.1.deformable_attention.value_proj.bias` | `[256]` | `float32` | Parameter |
| 46 | `pts_bbox_head.transformer.encoder.layers.1.attentions.1.output_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 47 | `pts_bbox_head.transformer.encoder.layers.1.attentions.1.output_proj.bias` | `[256]` | `float32` | Parameter |
| 48 | `pts_bbox_head.transformer.encoder.layers.1.ffns.0.layers.0.0.weight` | `[512, 256]` | `float32` | Parameter |
| 49 | `pts_bbox_head.transformer.encoder.layers.1.ffns.0.layers.0.0.bias` | `[512]` | `float32` | Parameter |
| 50 | `pts_bbox_head.transformer.encoder.layers.1.ffns.0.layers.1.weight` | `[256, 512]` | `float32` | Parameter |
| 51 | `pts_bbox_head.transformer.encoder.layers.1.ffns.0.layers.1.bias` | `[256]` | `float32` | Parameter |
| 52 | `pts_bbox_head.transformer.encoder.layers.1.norms.0.weight` | `[256]` | `float32` | Parameter |
| 53 | `pts_bbox_head.transformer.encoder.layers.1.norms.0.bias` | `[256]` | `float32` | Parameter |
| 54 | `pts_bbox_head.transformer.encoder.layers.1.norms.1.weight` | `[256]` | `float32` | Parameter |
| 55 | `pts_bbox_head.transformer.encoder.layers.1.norms.1.bias` | `[256]` | `float32` | Parameter |
| 56 | `pts_bbox_head.transformer.encoder.layers.1.norms.2.weight` | `[256]` | `float32` | Parameter |
| 57 | `pts_bbox_head.transformer.encoder.layers.1.norms.2.bias` | `[256]` | `float32` | Parameter |
| 58 | `pts_bbox_head.transformer.encoder.layers.2.attentions.0.sampling_offsets.weight` | `[128, 512]` | `float32` | Parameter |
| 59 | `pts_bbox_head.transformer.encoder.layers.2.attentions.0.sampling_offsets.bias` | `[128]` | `float32` | Parameter |
| 60 | `pts_bbox_head.transformer.encoder.layers.2.attentions.0.attention_weights.weight` | `[64, 512]` | `float32` | Parameter |
| 61 | `pts_bbox_head.transformer.encoder.layers.2.attentions.0.attention_weights.bias` | `[64]` | `float32` | Parameter |
| 62 | `pts_bbox_head.transformer.encoder.layers.2.attentions.0.value_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 63 | `pts_bbox_head.transformer.encoder.layers.2.attentions.0.value_proj.bias` | `[256]` | `float32` | Parameter |
| 64 | `pts_bbox_head.transformer.encoder.layers.2.attentions.0.output_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 65 | `pts_bbox_head.transformer.encoder.layers.2.attentions.0.output_proj.bias` | `[256]` | `float32` | Parameter |
| 66 | `pts_bbox_head.transformer.encoder.layers.2.attentions.1.deformable_attention.sampling_offsets.weight` | `[128, 256]` | `float32` | Parameter |
| 67 | `pts_bbox_head.transformer.encoder.layers.2.attentions.1.deformable_attention.sampling_offsets.bias` | `[128]` | `float32` | Parameter |
| 68 | `pts_bbox_head.transformer.encoder.layers.2.attentions.1.deformable_attention.attention_weights.weight` | `[64, 256]` | `float32` | Parameter |
| 69 | `pts_bbox_head.transformer.encoder.layers.2.attentions.1.deformable_attention.attention_weights.bias` | `[64]` | `float32` | Parameter |
| 70 | `pts_bbox_head.transformer.encoder.layers.2.attentions.1.deformable_attention.value_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 71 | `pts_bbox_head.transformer.encoder.layers.2.attentions.1.deformable_attention.value_proj.bias` | `[256]` | `float32` | Parameter |
| 72 | `pts_bbox_head.transformer.encoder.layers.2.attentions.1.output_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 73 | `pts_bbox_head.transformer.encoder.layers.2.attentions.1.output_proj.bias` | `[256]` | `float32` | Parameter |
| 74 | `pts_bbox_head.transformer.encoder.layers.2.ffns.0.layers.0.0.weight` | `[512, 256]` | `float32` | Parameter |
| 75 | `pts_bbox_head.transformer.encoder.layers.2.ffns.0.layers.0.0.bias` | `[512]` | `float32` | Parameter |
| 76 | `pts_bbox_head.transformer.encoder.layers.2.ffns.0.layers.1.weight` | `[256, 512]` | `float32` | Parameter |
| 77 | `pts_bbox_head.transformer.encoder.layers.2.ffns.0.layers.1.bias` | `[256]` | `float32` | Parameter |
| 78 | `pts_bbox_head.transformer.encoder.layers.2.norms.0.weight` | `[256]` | `float32` | Parameter |
| 79 | `pts_bbox_head.transformer.encoder.layers.2.norms.0.bias` | `[256]` | `float32` | Parameter |
| 80 | `pts_bbox_head.transformer.encoder.layers.2.norms.1.weight` | `[256]` | `float32` | Parameter |
| 81 | `pts_bbox_head.transformer.encoder.layers.2.norms.1.bias` | `[256]` | `float32` | Parameter |
| 82 | `pts_bbox_head.transformer.encoder.layers.2.norms.2.weight` | `[256]` | `float32` | Parameter |
| 83 | `pts_bbox_head.transformer.encoder.layers.2.norms.2.bias` | `[256]` | `float32` | Parameter |
| 84 | `pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.in_proj_weight` | `[768, 256]` | `float32` | Parameter |
| 85 | `pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.in_proj_bias` | `[768]` | `float32` | Parameter |
| 86 | `pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.out_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 87 | `pts_bbox_head.transformer.decoder.layers.0.attentions.0.attn.out_proj.bias` | `[256]` | `float32` | Parameter |
| 88 | `pts_bbox_head.transformer.decoder.layers.0.attentions.1.sampling_offsets.weight` | `[64, 256]` | `float32` | Parameter |
| 89 | `pts_bbox_head.transformer.decoder.layers.0.attentions.1.sampling_offsets.bias` | `[64]` | `float32` | Parameter |
| 90 | `pts_bbox_head.transformer.decoder.layers.0.attentions.1.attention_weights.weight` | `[32, 256]` | `float32` | Parameter |
| 91 | `pts_bbox_head.transformer.decoder.layers.0.attentions.1.attention_weights.bias` | `[32]` | `float32` | Parameter |
| 92 | `pts_bbox_head.transformer.decoder.layers.0.attentions.1.value_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 93 | `pts_bbox_head.transformer.decoder.layers.0.attentions.1.value_proj.bias` | `[256]` | `float32` | Parameter |
| 94 | `pts_bbox_head.transformer.decoder.layers.0.attentions.1.output_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 95 | `pts_bbox_head.transformer.decoder.layers.0.attentions.1.output_proj.bias` | `[256]` | `float32` | Parameter |
| 96 | `pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.0.0.weight` | `[512, 256]` | `float32` | Parameter |
| 97 | `pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.0.0.bias` | `[512]` | `float32` | Parameter |
| 98 | `pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.1.weight` | `[256, 512]` | `float32` | Parameter |
| 99 | `pts_bbox_head.transformer.decoder.layers.0.ffns.0.layers.1.bias` | `[256]` | `float32` | Parameter |
| 100 | `pts_bbox_head.transformer.decoder.layers.0.norms.0.weight` | `[256]` | `float32` | Parameter |
| 101 | `pts_bbox_head.transformer.decoder.layers.0.norms.0.bias` | `[256]` | `float32` | Parameter |
| 102 | `pts_bbox_head.transformer.decoder.layers.0.norms.1.weight` | `[256]` | `float32` | Parameter |
| 103 | `pts_bbox_head.transformer.decoder.layers.0.norms.1.bias` | `[256]` | `float32` | Parameter |
| 104 | `pts_bbox_head.transformer.decoder.layers.0.norms.2.weight` | `[256]` | `float32` | Parameter |
| 105 | `pts_bbox_head.transformer.decoder.layers.0.norms.2.bias` | `[256]` | `float32` | Parameter |
| 106 | `pts_bbox_head.transformer.decoder.layers.1.attentions.0.attn.in_proj_weight` | `[768, 256]` | `float32` | Parameter |
| 107 | `pts_bbox_head.transformer.decoder.layers.1.attentions.0.attn.in_proj_bias` | `[768]` | `float32` | Parameter |
| 108 | `pts_bbox_head.transformer.decoder.layers.1.attentions.0.attn.out_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 109 | `pts_bbox_head.transformer.decoder.layers.1.attentions.0.attn.out_proj.bias` | `[256]` | `float32` | Parameter |
| 110 | `pts_bbox_head.transformer.decoder.layers.1.attentions.1.sampling_offsets.weight` | `[64, 256]` | `float32` | Parameter |
| 111 | `pts_bbox_head.transformer.decoder.layers.1.attentions.1.sampling_offsets.bias` | `[64]` | `float32` | Parameter |
| 112 | `pts_bbox_head.transformer.decoder.layers.1.attentions.1.attention_weights.weight` | `[32, 256]` | `float32` | Parameter |
| 113 | `pts_bbox_head.transformer.decoder.layers.1.attentions.1.attention_weights.bias` | `[32]` | `float32` | Parameter |
| 114 | `pts_bbox_head.transformer.decoder.layers.1.attentions.1.value_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 115 | `pts_bbox_head.transformer.decoder.layers.1.attentions.1.value_proj.bias` | `[256]` | `float32` | Parameter |
| 116 | `pts_bbox_head.transformer.decoder.layers.1.attentions.1.output_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 117 | `pts_bbox_head.transformer.decoder.layers.1.attentions.1.output_proj.bias` | `[256]` | `float32` | Parameter |
| 118 | `pts_bbox_head.transformer.decoder.layers.1.ffns.0.layers.0.0.weight` | `[512, 256]` | `float32` | Parameter |
| 119 | `pts_bbox_head.transformer.decoder.layers.1.ffns.0.layers.0.0.bias` | `[512]` | `float32` | Parameter |
| 120 | `pts_bbox_head.transformer.decoder.layers.1.ffns.0.layers.1.weight` | `[256, 512]` | `float32` | Parameter |
| 121 | `pts_bbox_head.transformer.decoder.layers.1.ffns.0.layers.1.bias` | `[256]` | `float32` | Parameter |
| 122 | `pts_bbox_head.transformer.decoder.layers.1.norms.0.weight` | `[256]` | `float32` | Parameter |
| 123 | `pts_bbox_head.transformer.decoder.layers.1.norms.0.bias` | `[256]` | `float32` | Parameter |
| 124 | `pts_bbox_head.transformer.decoder.layers.1.norms.1.weight` | `[256]` | `float32` | Parameter |
| 125 | `pts_bbox_head.transformer.decoder.layers.1.norms.1.bias` | `[256]` | `float32` | Parameter |
| 126 | `pts_bbox_head.transformer.decoder.layers.1.norms.2.weight` | `[256]` | `float32` | Parameter |
| 127 | `pts_bbox_head.transformer.decoder.layers.1.norms.2.bias` | `[256]` | `float32` | Parameter |
| 128 | `pts_bbox_head.transformer.decoder.layers.2.attentions.0.attn.in_proj_weight` | `[768, 256]` | `float32` | Parameter |
| 129 | `pts_bbox_head.transformer.decoder.layers.2.attentions.0.attn.in_proj_bias` | `[768]` | `float32` | Parameter |
| 130 | `pts_bbox_head.transformer.decoder.layers.2.attentions.0.attn.out_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 131 | `pts_bbox_head.transformer.decoder.layers.2.attentions.0.attn.out_proj.bias` | `[256]` | `float32` | Parameter |
| 132 | `pts_bbox_head.transformer.decoder.layers.2.attentions.1.sampling_offsets.weight` | `[64, 256]` | `float32` | Parameter |
| 133 | `pts_bbox_head.transformer.decoder.layers.2.attentions.1.sampling_offsets.bias` | `[64]` | `float32` | Parameter |
| 134 | `pts_bbox_head.transformer.decoder.layers.2.attentions.1.attention_weights.weight` | `[32, 256]` | `float32` | Parameter |
| 135 | `pts_bbox_head.transformer.decoder.layers.2.attentions.1.attention_weights.bias` | `[32]` | `float32` | Parameter |
| 136 | `pts_bbox_head.transformer.decoder.layers.2.attentions.1.value_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 137 | `pts_bbox_head.transformer.decoder.layers.2.attentions.1.value_proj.bias` | `[256]` | `float32` | Parameter |
| 138 | `pts_bbox_head.transformer.decoder.layers.2.attentions.1.output_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 139 | `pts_bbox_head.transformer.decoder.layers.2.attentions.1.output_proj.bias` | `[256]` | `float32` | Parameter |
| 140 | `pts_bbox_head.transformer.decoder.layers.2.ffns.0.layers.0.0.weight` | `[512, 256]` | `float32` | Parameter |
| 141 | `pts_bbox_head.transformer.decoder.layers.2.ffns.0.layers.0.0.bias` | `[512]` | `float32` | Parameter |
| 142 | `pts_bbox_head.transformer.decoder.layers.2.ffns.0.layers.1.weight` | `[256, 512]` | `float32` | Parameter |
| 143 | `pts_bbox_head.transformer.decoder.layers.2.ffns.0.layers.1.bias` | `[256]` | `float32` | Parameter |
| 144 | `pts_bbox_head.transformer.decoder.layers.2.norms.0.weight` | `[256]` | `float32` | Parameter |
| 145 | `pts_bbox_head.transformer.decoder.layers.2.norms.0.bias` | `[256]` | `float32` | Parameter |
| 146 | `pts_bbox_head.transformer.decoder.layers.2.norms.1.weight` | `[256]` | `float32` | Parameter |
| 147 | `pts_bbox_head.transformer.decoder.layers.2.norms.1.bias` | `[256]` | `float32` | Parameter |
| 148 | `pts_bbox_head.transformer.decoder.layers.2.norms.2.weight` | `[256]` | `float32` | Parameter |
| 149 | `pts_bbox_head.transformer.decoder.layers.2.norms.2.bias` | `[256]` | `float32` | Parameter |
| 150 | `pts_bbox_head.transformer.decoder.layers.3.attentions.0.attn.in_proj_weight` | `[768, 256]` | `float32` | Parameter |
| 151 | `pts_bbox_head.transformer.decoder.layers.3.attentions.0.attn.in_proj_bias` | `[768]` | `float32` | Parameter |
| 152 | `pts_bbox_head.transformer.decoder.layers.3.attentions.0.attn.out_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 153 | `pts_bbox_head.transformer.decoder.layers.3.attentions.0.attn.out_proj.bias` | `[256]` | `float32` | Parameter |
| 154 | `pts_bbox_head.transformer.decoder.layers.3.attentions.1.sampling_offsets.weight` | `[64, 256]` | `float32` | Parameter |
| 155 | `pts_bbox_head.transformer.decoder.layers.3.attentions.1.sampling_offsets.bias` | `[64]` | `float32` | Parameter |
| 156 | `pts_bbox_head.transformer.decoder.layers.3.attentions.1.attention_weights.weight` | `[32, 256]` | `float32` | Parameter |
| 157 | `pts_bbox_head.transformer.decoder.layers.3.attentions.1.attention_weights.bias` | `[32]` | `float32` | Parameter |
| 158 | `pts_bbox_head.transformer.decoder.layers.3.attentions.1.value_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 159 | `pts_bbox_head.transformer.decoder.layers.3.attentions.1.value_proj.bias` | `[256]` | `float32` | Parameter |
| 160 | `pts_bbox_head.transformer.decoder.layers.3.attentions.1.output_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 161 | `pts_bbox_head.transformer.decoder.layers.3.attentions.1.output_proj.bias` | `[256]` | `float32` | Parameter |
| 162 | `pts_bbox_head.transformer.decoder.layers.3.ffns.0.layers.0.0.weight` | `[512, 256]` | `float32` | Parameter |
| 163 | `pts_bbox_head.transformer.decoder.layers.3.ffns.0.layers.0.0.bias` | `[512]` | `float32` | Parameter |
| 164 | `pts_bbox_head.transformer.decoder.layers.3.ffns.0.layers.1.weight` | `[256, 512]` | `float32` | Parameter |
| 165 | `pts_bbox_head.transformer.decoder.layers.3.ffns.0.layers.1.bias` | `[256]` | `float32` | Parameter |
| 166 | `pts_bbox_head.transformer.decoder.layers.3.norms.0.weight` | `[256]` | `float32` | Parameter |
| 167 | `pts_bbox_head.transformer.decoder.layers.3.norms.0.bias` | `[256]` | `float32` | Parameter |
| 168 | `pts_bbox_head.transformer.decoder.layers.3.norms.1.weight` | `[256]` | `float32` | Parameter |
| 169 | `pts_bbox_head.transformer.decoder.layers.3.norms.1.bias` | `[256]` | `float32` | Parameter |
| 170 | `pts_bbox_head.transformer.decoder.layers.3.norms.2.weight` | `[256]` | `float32` | Parameter |
| 171 | `pts_bbox_head.transformer.decoder.layers.3.norms.2.bias` | `[256]` | `float32` | Parameter |
| 172 | `pts_bbox_head.transformer.decoder.layers.4.attentions.0.attn.in_proj_weight` | `[768, 256]` | `float32` | Parameter |
| 173 | `pts_bbox_head.transformer.decoder.layers.4.attentions.0.attn.in_proj_bias` | `[768]` | `float32` | Parameter |
| 174 | `pts_bbox_head.transformer.decoder.layers.4.attentions.0.attn.out_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 175 | `pts_bbox_head.transformer.decoder.layers.4.attentions.0.attn.out_proj.bias` | `[256]` | `float32` | Parameter |
| 176 | `pts_bbox_head.transformer.decoder.layers.4.attentions.1.sampling_offsets.weight` | `[64, 256]` | `float32` | Parameter |
| 177 | `pts_bbox_head.transformer.decoder.layers.4.attentions.1.sampling_offsets.bias` | `[64]` | `float32` | Parameter |
| 178 | `pts_bbox_head.transformer.decoder.layers.4.attentions.1.attention_weights.weight` | `[32, 256]` | `float32` | Parameter |
| 179 | `pts_bbox_head.transformer.decoder.layers.4.attentions.1.attention_weights.bias` | `[32]` | `float32` | Parameter |
| 180 | `pts_bbox_head.transformer.decoder.layers.4.attentions.1.value_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 181 | `pts_bbox_head.transformer.decoder.layers.4.attentions.1.value_proj.bias` | `[256]` | `float32` | Parameter |
| 182 | `pts_bbox_head.transformer.decoder.layers.4.attentions.1.output_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 183 | `pts_bbox_head.transformer.decoder.layers.4.attentions.1.output_proj.bias` | `[256]` | `float32` | Parameter |
| 184 | `pts_bbox_head.transformer.decoder.layers.4.ffns.0.layers.0.0.weight` | `[512, 256]` | `float32` | Parameter |
| 185 | `pts_bbox_head.transformer.decoder.layers.4.ffns.0.layers.0.0.bias` | `[512]` | `float32` | Parameter |
| 186 | `pts_bbox_head.transformer.decoder.layers.4.ffns.0.layers.1.weight` | `[256, 512]` | `float32` | Parameter |
| 187 | `pts_bbox_head.transformer.decoder.layers.4.ffns.0.layers.1.bias` | `[256]` | `float32` | Parameter |
| 188 | `pts_bbox_head.transformer.decoder.layers.4.norms.0.weight` | `[256]` | `float32` | Parameter |
| 189 | `pts_bbox_head.transformer.decoder.layers.4.norms.0.bias` | `[256]` | `float32` | Parameter |
| 190 | `pts_bbox_head.transformer.decoder.layers.4.norms.1.weight` | `[256]` | `float32` | Parameter |
| 191 | `pts_bbox_head.transformer.decoder.layers.4.norms.1.bias` | `[256]` | `float32` | Parameter |
| 192 | `pts_bbox_head.transformer.decoder.layers.4.norms.2.weight` | `[256]` | `float32` | Parameter |
| 193 | `pts_bbox_head.transformer.decoder.layers.4.norms.2.bias` | `[256]` | `float32` | Parameter |
| 194 | `pts_bbox_head.transformer.decoder.layers.5.attentions.0.attn.in_proj_weight` | `[768, 256]` | `float32` | Parameter |
| 195 | `pts_bbox_head.transformer.decoder.layers.5.attentions.0.attn.in_proj_bias` | `[768]` | `float32` | Parameter |
| 196 | `pts_bbox_head.transformer.decoder.layers.5.attentions.0.attn.out_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 197 | `pts_bbox_head.transformer.decoder.layers.5.attentions.0.attn.out_proj.bias` | `[256]` | `float32` | Parameter |
| 198 | `pts_bbox_head.transformer.decoder.layers.5.attentions.1.sampling_offsets.weight` | `[64, 256]` | `float32` | Parameter |
| 199 | `pts_bbox_head.transformer.decoder.layers.5.attentions.1.sampling_offsets.bias` | `[64]` | `float32` | Parameter |
| 200 | `pts_bbox_head.transformer.decoder.layers.5.attentions.1.attention_weights.weight` | `[32, 256]` | `float32` | Parameter |
| 201 | `pts_bbox_head.transformer.decoder.layers.5.attentions.1.attention_weights.bias` | `[32]` | `float32` | Parameter |
| 202 | `pts_bbox_head.transformer.decoder.layers.5.attentions.1.value_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 203 | `pts_bbox_head.transformer.decoder.layers.5.attentions.1.value_proj.bias` | `[256]` | `float32` | Parameter |
| 204 | `pts_bbox_head.transformer.decoder.layers.5.attentions.1.output_proj.weight` | `[256, 256]` | `float32` | Parameter |
| 205 | `pts_bbox_head.transformer.decoder.layers.5.attentions.1.output_proj.bias` | `[256]` | `float32` | Parameter |
| 206 | `pts_bbox_head.transformer.decoder.layers.5.ffns.0.layers.0.0.weight` | `[512, 256]` | `float32` | Parameter |
| 207 | `pts_bbox_head.transformer.decoder.layers.5.ffns.0.layers.0.0.bias` | `[512]` | `float32` | Parameter |
| 208 | `pts_bbox_head.transformer.decoder.layers.5.ffns.0.layers.1.weight` | `[256, 512]` | `float32` | Parameter |
| 209 | `pts_bbox_head.transformer.decoder.layers.5.ffns.0.layers.1.bias` | `[256]` | `float32` | Parameter |
| 210 | `pts_bbox_head.transformer.decoder.layers.5.norms.0.weight` | `[256]` | `float32` | Parameter |
| 211 | `pts_bbox_head.transformer.decoder.layers.5.norms.0.bias` | `[256]` | `float32` | Parameter |
| 212 | `pts_bbox_head.transformer.decoder.layers.5.norms.1.weight` | `[256]` | `float32` | Parameter |
| 213 | `pts_bbox_head.transformer.decoder.layers.5.norms.1.bias` | `[256]` | `float32` | Parameter |
| 214 | `pts_bbox_head.transformer.decoder.layers.5.norms.2.weight` | `[256]` | `float32` | Parameter |
| 215 | `pts_bbox_head.transformer.decoder.layers.5.norms.2.bias` | `[256]` | `float32` | Parameter |
| 216 | `pts_bbox_head.transformer.reference_points.weight` | `[3, 256]` | `float32` | Parameter |
| 217 | `pts_bbox_head.transformer.reference_points.bias` | `[3]` | `float32` | Parameter |
| 218 | `pts_bbox_head.transformer.can_bus_mlp.0.weight` | `[128, 18]` | `float32` | Parameter |
| 219 | `pts_bbox_head.transformer.can_bus_mlp.0.bias` | `[128]` | `float32` | Parameter |
| 220 | `pts_bbox_head.transformer.can_bus_mlp.2.weight` | `[256, 128]` | `float32` | Parameter |
| 221 | `pts_bbox_head.transformer.can_bus_mlp.2.bias` | `[256]` | `float32` | Parameter |
| 222 | `pts_bbox_head.transformer.can_bus_mlp.norm.weight` | `[256]` | `float32` | Parameter |
| 223 | `pts_bbox_head.transformer.can_bus_mlp.norm.bias` | `[256]` | `float32` | Parameter |
| 224 | `pts_bbox_head.cls_branches.0.0.weight` | `[256, 256]` | `float32` | Parameter |
| 225 | `pts_bbox_head.cls_branches.0.0.bias` | `[256]` | `float32` | Parameter |
| 226 | `pts_bbox_head.cls_branches.0.1.weight` | `[256]` | `float32` | Parameter |
| 227 | `pts_bbox_head.cls_branches.0.1.bias` | `[256]` | `float32` | Parameter |
| 228 | `pts_bbox_head.cls_branches.0.3.weight` | `[256, 256]` | `float32` | Parameter |
| 229 | `pts_bbox_head.cls_branches.0.3.bias` | `[256]` | `float32` | Parameter |
| 230 | `pts_bbox_head.cls_branches.0.4.weight` | `[256]` | `float32` | Parameter |
| 231 | `pts_bbox_head.cls_branches.0.4.bias` | `[256]` | `float32` | Parameter |
| 232 | `pts_bbox_head.cls_branches.0.6.weight` | `[10, 256]` | `float32` | Parameter |
| 233 | `pts_bbox_head.cls_branches.0.6.bias` | `[10]` | `float32` | Parameter |
| 234 | `pts_bbox_head.cls_branches.1.0.weight` | `[256, 256]` | `float32` | Parameter |
| 235 | `pts_bbox_head.cls_branches.1.0.bias` | `[256]` | `float32` | Parameter |
| 236 | `pts_bbox_head.cls_branches.1.1.weight` | `[256]` | `float32` | Parameter |
| 237 | `pts_bbox_head.cls_branches.1.1.bias` | `[256]` | `float32` | Parameter |
| 238 | `pts_bbox_head.cls_branches.1.3.weight` | `[256, 256]` | `float32` | Parameter |
| 239 | `pts_bbox_head.cls_branches.1.3.bias` | `[256]` | `float32` | Parameter |
| 240 | `pts_bbox_head.cls_branches.1.4.weight` | `[256]` | `float32` | Parameter |
| 241 | `pts_bbox_head.cls_branches.1.4.bias` | `[256]` | `float32` | Parameter |
| 242 | `pts_bbox_head.cls_branches.1.6.weight` | `[10, 256]` | `float32` | Parameter |
| 243 | `pts_bbox_head.cls_branches.1.6.bias` | `[10]` | `float32` | Parameter |
| 244 | `pts_bbox_head.cls_branches.2.0.weight` | `[256, 256]` | `float32` | Parameter |
| 245 | `pts_bbox_head.cls_branches.2.0.bias` | `[256]` | `float32` | Parameter |
| 246 | `pts_bbox_head.cls_branches.2.1.weight` | `[256]` | `float32` | Parameter |
| 247 | `pts_bbox_head.cls_branches.2.1.bias` | `[256]` | `float32` | Parameter |
| 248 | `pts_bbox_head.cls_branches.2.3.weight` | `[256, 256]` | `float32` | Parameter |
| 249 | `pts_bbox_head.cls_branches.2.3.bias` | `[256]` | `float32` | Parameter |
| 250 | `pts_bbox_head.cls_branches.2.4.weight` | `[256]` | `float32` | Parameter |
| 251 | `pts_bbox_head.cls_branches.2.4.bias` | `[256]` | `float32` | Parameter |
| 252 | `pts_bbox_head.cls_branches.2.6.weight` | `[10, 256]` | `float32` | Parameter |
| 253 | `pts_bbox_head.cls_branches.2.6.bias` | `[10]` | `float32` | Parameter |
| 254 | `pts_bbox_head.cls_branches.3.0.weight` | `[256, 256]` | `float32` | Parameter |
| 255 | `pts_bbox_head.cls_branches.3.0.bias` | `[256]` | `float32` | Parameter |
| 256 | `pts_bbox_head.cls_branches.3.1.weight` | `[256]` | `float32` | Parameter |
| 257 | `pts_bbox_head.cls_branches.3.1.bias` | `[256]` | `float32` | Parameter |
| 258 | `pts_bbox_head.cls_branches.3.3.weight` | `[256, 256]` | `float32` | Parameter |
| 259 | `pts_bbox_head.cls_branches.3.3.bias` | `[256]` | `float32` | Parameter |
| 260 | `pts_bbox_head.cls_branches.3.4.weight` | `[256]` | `float32` | Parameter |
| 261 | `pts_bbox_head.cls_branches.3.4.bias` | `[256]` | `float32` | Parameter |
| 262 | `pts_bbox_head.cls_branches.3.6.weight` | `[10, 256]` | `float32` | Parameter |
| 263 | `pts_bbox_head.cls_branches.3.6.bias` | `[10]` | `float32` | Parameter |
| 264 | `pts_bbox_head.cls_branches.4.0.weight` | `[256, 256]` | `float32` | Parameter |
| 265 | `pts_bbox_head.cls_branches.4.0.bias` | `[256]` | `float32` | Parameter |
| 266 | `pts_bbox_head.cls_branches.4.1.weight` | `[256]` | `float32` | Parameter |
| 267 | `pts_bbox_head.cls_branches.4.1.bias` | `[256]` | `float32` | Parameter |
| 268 | `pts_bbox_head.cls_branches.4.3.weight` | `[256, 256]` | `float32` | Parameter |
| 269 | `pts_bbox_head.cls_branches.4.3.bias` | `[256]` | `float32` | Parameter |
| 270 | `pts_bbox_head.cls_branches.4.4.weight` | `[256]` | `float32` | Parameter |
| 271 | `pts_bbox_head.cls_branches.4.4.bias` | `[256]` | `float32` | Parameter |
| 272 | `pts_bbox_head.cls_branches.4.6.weight` | `[10, 256]` | `float32` | Parameter |
| 273 | `pts_bbox_head.cls_branches.4.6.bias` | `[10]` | `float32` | Parameter |
| 274 | `pts_bbox_head.cls_branches.5.0.weight` | `[256, 256]` | `float32` | Parameter |
| 275 | `pts_bbox_head.cls_branches.5.0.bias` | `[256]` | `float32` | Parameter |
| 276 | `pts_bbox_head.cls_branches.5.1.weight` | `[256]` | `float32` | Parameter |
| 277 | `pts_bbox_head.cls_branches.5.1.bias` | `[256]` | `float32` | Parameter |
| 278 | `pts_bbox_head.cls_branches.5.3.weight` | `[256, 256]` | `float32` | Parameter |
| 279 | `pts_bbox_head.cls_branches.5.3.bias` | `[256]` | `float32` | Parameter |
| 280 | `pts_bbox_head.cls_branches.5.4.weight` | `[256]` | `float32` | Parameter |
| 281 | `pts_bbox_head.cls_branches.5.4.bias` | `[256]` | `float32` | Parameter |
| 282 | `pts_bbox_head.cls_branches.5.6.weight` | `[10, 256]` | `float32` | Parameter |
| 283 | `pts_bbox_head.cls_branches.5.6.bias` | `[10]` | `float32` | Parameter |
| 284 | `pts_bbox_head.reg_branches.0.0.weight` | `[256, 256]` | `float32` | Parameter |
| 285 | `pts_bbox_head.reg_branches.0.0.bias` | `[256]` | `float32` | Parameter |
| 286 | `pts_bbox_head.reg_branches.0.2.weight` | `[256, 256]` | `float32` | Parameter |
| 287 | `pts_bbox_head.reg_branches.0.2.bias` | `[256]` | `float32` | Parameter |
| 288 | `pts_bbox_head.reg_branches.0.4.weight` | `[10, 256]` | `float32` | Parameter |
| 289 | `pts_bbox_head.reg_branches.0.4.bias` | `[10]` | `float32` | Parameter |
| 290 | `pts_bbox_head.reg_branches.1.0.weight` | `[256, 256]` | `float32` | Parameter |
| 291 | `pts_bbox_head.reg_branches.1.0.bias` | `[256]` | `float32` | Parameter |
| 292 | `pts_bbox_head.reg_branches.1.2.weight` | `[256, 256]` | `float32` | Parameter |
| 293 | `pts_bbox_head.reg_branches.1.2.bias` | `[256]` | `float32` | Parameter |
| 294 | `pts_bbox_head.reg_branches.1.4.weight` | `[10, 256]` | `float32` | Parameter |
| 295 | `pts_bbox_head.reg_branches.1.4.bias` | `[10]` | `float32` | Parameter |
| 296 | `pts_bbox_head.reg_branches.2.0.weight` | `[256, 256]` | `float32` | Parameter |
| 297 | `pts_bbox_head.reg_branches.2.0.bias` | `[256]` | `float32` | Parameter |
| 298 | `pts_bbox_head.reg_branches.2.2.weight` | `[256, 256]` | `float32` | Parameter |
| 299 | `pts_bbox_head.reg_branches.2.2.bias` | `[256]` | `float32` | Parameter |
| 300 | `pts_bbox_head.reg_branches.2.4.weight` | `[10, 256]` | `float32` | Parameter |
| 301 | `pts_bbox_head.reg_branches.2.4.bias` | `[10]` | `float32` | Parameter |
| 302 | `pts_bbox_head.reg_branches.3.0.weight` | `[256, 256]` | `float32` | Parameter |
| 303 | `pts_bbox_head.reg_branches.3.0.bias` | `[256]` | `float32` | Parameter |
| 304 | `pts_bbox_head.reg_branches.3.2.weight` | `[256, 256]` | `float32` | Parameter |
| 305 | `pts_bbox_head.reg_branches.3.2.bias` | `[256]` | `float32` | Parameter |
| 306 | `pts_bbox_head.reg_branches.3.4.weight` | `[10, 256]` | `float32` | Parameter |
| 307 | `pts_bbox_head.reg_branches.3.4.bias` | `[10]` | `float32` | Parameter |
| 308 | `pts_bbox_head.reg_branches.4.0.weight` | `[256, 256]` | `float32` | Parameter |
| 309 | `pts_bbox_head.reg_branches.4.0.bias` | `[256]` | `float32` | Parameter |
| 310 | `pts_bbox_head.reg_branches.4.2.weight` | `[256, 256]` | `float32` | Parameter |
| 311 | `pts_bbox_head.reg_branches.4.2.bias` | `[256]` | `float32` | Parameter |
| 312 | `pts_bbox_head.reg_branches.4.4.weight` | `[10, 256]` | `float32` | Parameter |
| 313 | `pts_bbox_head.reg_branches.4.4.bias` | `[10]` | `float32` | Parameter |
| 314 | `pts_bbox_head.reg_branches.5.0.weight` | `[256, 256]` | `float32` | Parameter |
| 315 | `pts_bbox_head.reg_branches.5.0.bias` | `[256]` | `float32` | Parameter |
| 316 | `pts_bbox_head.reg_branches.5.2.weight` | `[256, 256]` | `float32` | Parameter |
| 317 | `pts_bbox_head.reg_branches.5.2.bias` | `[256]` | `float32` | Parameter |
| 318 | `pts_bbox_head.reg_branches.5.4.weight` | `[10, 256]` | `float32` | Parameter |
| 319 | `pts_bbox_head.reg_branches.5.4.bias` | `[10]` | `float32` | Parameter |
| 320 | `pts_bbox_head.bev_embedding.weight` | `[2500, 256]` | `float32` | Parameter |
| 321 | `pts_bbox_head.query_embedding.weight` | `[900, 512]` | `float32` | Parameter |
| 322 | `img_backbone.conv1.weight` | `[64, 3, 7, 7]` | `float32` | Parameter |
| 323 | `img_backbone.bn1.weight` | `[64]` | `float32` | Parameter |
| 324 | `img_backbone.bn1.bias` | `[64]` | `float32` | Parameter |
| 325 | `img_backbone.bn1.running_mean` | `[64]` | `float32` | Buffer |
| 326 | `img_backbone.bn1.running_var` | `[64]` | `float32` | Buffer |
| 327 | `img_backbone.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 328 | `img_backbone.layer1.0.conv1.weight` | `[64, 64, 1, 1]` | `float32` | Parameter |
| 329 | `img_backbone.layer1.0.bn1.weight` | `[64]` | `float32` | Parameter |
| 330 | `img_backbone.layer1.0.bn1.bias` | `[64]` | `float32` | Parameter |
| 331 | `img_backbone.layer1.0.bn1.running_mean` | `[64]` | `float32` | Buffer |
| 332 | `img_backbone.layer1.0.bn1.running_var` | `[64]` | `float32` | Buffer |
| 333 | `img_backbone.layer1.0.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 334 | `img_backbone.layer1.0.conv2.weight` | `[64, 64, 3, 3]` | `float32` | Parameter |
| 335 | `img_backbone.layer1.0.bn2.weight` | `[64]` | `float32` | Parameter |
| 336 | `img_backbone.layer1.0.bn2.bias` | `[64]` | `float32` | Parameter |
| 337 | `img_backbone.layer1.0.bn2.running_mean` | `[64]` | `float32` | Buffer |
| 338 | `img_backbone.layer1.0.bn2.running_var` | `[64]` | `float32` | Buffer |
| 339 | `img_backbone.layer1.0.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 340 | `img_backbone.layer1.0.conv3.weight` | `[256, 64, 1, 1]` | `float32` | Parameter |
| 341 | `img_backbone.layer1.0.bn3.weight` | `[256]` | `float32` | Parameter |
| 342 | `img_backbone.layer1.0.bn3.bias` | `[256]` | `float32` | Parameter |
| 343 | `img_backbone.layer1.0.bn3.running_mean` | `[256]` | `float32` | Buffer |
| 344 | `img_backbone.layer1.0.bn3.running_var` | `[256]` | `float32` | Buffer |
| 345 | `img_backbone.layer1.0.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 346 | `img_backbone.layer1.0.downsample.0.weight` | `[256, 64, 1, 1]` | `float32` | Parameter |
| 347 | `img_backbone.layer1.0.downsample.1.weight` | `[256]` | `float32` | Parameter |
| 348 | `img_backbone.layer1.0.downsample.1.bias` | `[256]` | `float32` | Parameter |
| 349 | `img_backbone.layer1.0.downsample.1.running_mean` | `[256]` | `float32` | Buffer |
| 350 | `img_backbone.layer1.0.downsample.1.running_var` | `[256]` | `float32` | Buffer |
| 351 | `img_backbone.layer1.0.downsample.1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 352 | `img_backbone.layer1.1.conv1.weight` | `[64, 256, 1, 1]` | `float32` | Parameter |
| 353 | `img_backbone.layer1.1.bn1.weight` | `[64]` | `float32` | Parameter |
| 354 | `img_backbone.layer1.1.bn1.bias` | `[64]` | `float32` | Parameter |
| 355 | `img_backbone.layer1.1.bn1.running_mean` | `[64]` | `float32` | Buffer |
| 356 | `img_backbone.layer1.1.bn1.running_var` | `[64]` | `float32` | Buffer |
| 357 | `img_backbone.layer1.1.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 358 | `img_backbone.layer1.1.conv2.weight` | `[64, 64, 3, 3]` | `float32` | Parameter |
| 359 | `img_backbone.layer1.1.bn2.weight` | `[64]` | `float32` | Parameter |
| 360 | `img_backbone.layer1.1.bn2.bias` | `[64]` | `float32` | Parameter |
| 361 | `img_backbone.layer1.1.bn2.running_mean` | `[64]` | `float32` | Buffer |
| 362 | `img_backbone.layer1.1.bn2.running_var` | `[64]` | `float32` | Buffer |
| 363 | `img_backbone.layer1.1.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 364 | `img_backbone.layer1.1.conv3.weight` | `[256, 64, 1, 1]` | `float32` | Parameter |
| 365 | `img_backbone.layer1.1.bn3.weight` | `[256]` | `float32` | Parameter |
| 366 | `img_backbone.layer1.1.bn3.bias` | `[256]` | `float32` | Parameter |
| 367 | `img_backbone.layer1.1.bn3.running_mean` | `[256]` | `float32` | Buffer |
| 368 | `img_backbone.layer1.1.bn3.running_var` | `[256]` | `float32` | Buffer |
| 369 | `img_backbone.layer1.1.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 370 | `img_backbone.layer1.2.conv1.weight` | `[64, 256, 1, 1]` | `float32` | Parameter |
| 371 | `img_backbone.layer1.2.bn1.weight` | `[64]` | `float32` | Parameter |
| 372 | `img_backbone.layer1.2.bn1.bias` | `[64]` | `float32` | Parameter |
| 373 | `img_backbone.layer1.2.bn1.running_mean` | `[64]` | `float32` | Buffer |
| 374 | `img_backbone.layer1.2.bn1.running_var` | `[64]` | `float32` | Buffer |
| 375 | `img_backbone.layer1.2.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 376 | `img_backbone.layer1.2.conv2.weight` | `[64, 64, 3, 3]` | `float32` | Parameter |
| 377 | `img_backbone.layer1.2.bn2.weight` | `[64]` | `float32` | Parameter |
| 378 | `img_backbone.layer1.2.bn2.bias` | `[64]` | `float32` | Parameter |
| 379 | `img_backbone.layer1.2.bn2.running_mean` | `[64]` | `float32` | Buffer |
| 380 | `img_backbone.layer1.2.bn2.running_var` | `[64]` | `float32` | Buffer |
| 381 | `img_backbone.layer1.2.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 382 | `img_backbone.layer1.2.conv3.weight` | `[256, 64, 1, 1]` | `float32` | Parameter |
| 383 | `img_backbone.layer1.2.bn3.weight` | `[256]` | `float32` | Parameter |
| 384 | `img_backbone.layer1.2.bn3.bias` | `[256]` | `float32` | Parameter |
| 385 | `img_backbone.layer1.2.bn3.running_mean` | `[256]` | `float32` | Buffer |
| 386 | `img_backbone.layer1.2.bn3.running_var` | `[256]` | `float32` | Buffer |
| 387 | `img_backbone.layer1.2.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 388 | `img_backbone.layer2.0.conv1.weight` | `[128, 256, 1, 1]` | `float32` | Parameter |
| 389 | `img_backbone.layer2.0.bn1.weight` | `[128]` | `float32` | Parameter |
| 390 | `img_backbone.layer2.0.bn1.bias` | `[128]` | `float32` | Parameter |
| 391 | `img_backbone.layer2.0.bn1.running_mean` | `[128]` | `float32` | Buffer |
| 392 | `img_backbone.layer2.0.bn1.running_var` | `[128]` | `float32` | Buffer |
| 393 | `img_backbone.layer2.0.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 394 | `img_backbone.layer2.0.conv2.weight` | `[128, 128, 3, 3]` | `float32` | Parameter |
| 395 | `img_backbone.layer2.0.bn2.weight` | `[128]` | `float32` | Parameter |
| 396 | `img_backbone.layer2.0.bn2.bias` | `[128]` | `float32` | Parameter |
| 397 | `img_backbone.layer2.0.bn2.running_mean` | `[128]` | `float32` | Buffer |
| 398 | `img_backbone.layer2.0.bn2.running_var` | `[128]` | `float32` | Buffer |
| 399 | `img_backbone.layer2.0.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 400 | `img_backbone.layer2.0.conv3.weight` | `[512, 128, 1, 1]` | `float32` | Parameter |
| 401 | `img_backbone.layer2.0.bn3.weight` | `[512]` | `float32` | Parameter |
| 402 | `img_backbone.layer2.0.bn3.bias` | `[512]` | `float32` | Parameter |
| 403 | `img_backbone.layer2.0.bn3.running_mean` | `[512]` | `float32` | Buffer |
| 404 | `img_backbone.layer2.0.bn3.running_var` | `[512]` | `float32` | Buffer |
| 405 | `img_backbone.layer2.0.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 406 | `img_backbone.layer2.0.downsample.0.weight` | `[512, 256, 1, 1]` | `float32` | Parameter |
| 407 | `img_backbone.layer2.0.downsample.1.weight` | `[512]` | `float32` | Parameter |
| 408 | `img_backbone.layer2.0.downsample.1.bias` | `[512]` | `float32` | Parameter |
| 409 | `img_backbone.layer2.0.downsample.1.running_mean` | `[512]` | `float32` | Buffer |
| 410 | `img_backbone.layer2.0.downsample.1.running_var` | `[512]` | `float32` | Buffer |
| 411 | `img_backbone.layer2.0.downsample.1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 412 | `img_backbone.layer2.1.conv1.weight` | `[128, 512, 1, 1]` | `float32` | Parameter |
| 413 | `img_backbone.layer2.1.bn1.weight` | `[128]` | `float32` | Parameter |
| 414 | `img_backbone.layer2.1.bn1.bias` | `[128]` | `float32` | Parameter |
| 415 | `img_backbone.layer2.1.bn1.running_mean` | `[128]` | `float32` | Buffer |
| 416 | `img_backbone.layer2.1.bn1.running_var` | `[128]` | `float32` | Buffer |
| 417 | `img_backbone.layer2.1.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 418 | `img_backbone.layer2.1.conv2.weight` | `[128, 128, 3, 3]` | `float32` | Parameter |
| 419 | `img_backbone.layer2.1.bn2.weight` | `[128]` | `float32` | Parameter |
| 420 | `img_backbone.layer2.1.bn2.bias` | `[128]` | `float32` | Parameter |
| 421 | `img_backbone.layer2.1.bn2.running_mean` | `[128]` | `float32` | Buffer |
| 422 | `img_backbone.layer2.1.bn2.running_var` | `[128]` | `float32` | Buffer |
| 423 | `img_backbone.layer2.1.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 424 | `img_backbone.layer2.1.conv3.weight` | `[512, 128, 1, 1]` | `float32` | Parameter |
| 425 | `img_backbone.layer2.1.bn3.weight` | `[512]` | `float32` | Parameter |
| 426 | `img_backbone.layer2.1.bn3.bias` | `[512]` | `float32` | Parameter |
| 427 | `img_backbone.layer2.1.bn3.running_mean` | `[512]` | `float32` | Buffer |
| 428 | `img_backbone.layer2.1.bn3.running_var` | `[512]` | `float32` | Buffer |
| 429 | `img_backbone.layer2.1.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 430 | `img_backbone.layer2.2.conv1.weight` | `[128, 512, 1, 1]` | `float32` | Parameter |
| 431 | `img_backbone.layer2.2.bn1.weight` | `[128]` | `float32` | Parameter |
| 432 | `img_backbone.layer2.2.bn1.bias` | `[128]` | `float32` | Parameter |
| 433 | `img_backbone.layer2.2.bn1.running_mean` | `[128]` | `float32` | Buffer |
| 434 | `img_backbone.layer2.2.bn1.running_var` | `[128]` | `float32` | Buffer |
| 435 | `img_backbone.layer2.2.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 436 | `img_backbone.layer2.2.conv2.weight` | `[128, 128, 3, 3]` | `float32` | Parameter |
| 437 | `img_backbone.layer2.2.bn2.weight` | `[128]` | `float32` | Parameter |
| 438 | `img_backbone.layer2.2.bn2.bias` | `[128]` | `float32` | Parameter |
| 439 | `img_backbone.layer2.2.bn2.running_mean` | `[128]` | `float32` | Buffer |
| 440 | `img_backbone.layer2.2.bn2.running_var` | `[128]` | `float32` | Buffer |
| 441 | `img_backbone.layer2.2.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 442 | `img_backbone.layer2.2.conv3.weight` | `[512, 128, 1, 1]` | `float32` | Parameter |
| 443 | `img_backbone.layer2.2.bn3.weight` | `[512]` | `float32` | Parameter |
| 444 | `img_backbone.layer2.2.bn3.bias` | `[512]` | `float32` | Parameter |
| 445 | `img_backbone.layer2.2.bn3.running_mean` | `[512]` | `float32` | Buffer |
| 446 | `img_backbone.layer2.2.bn3.running_var` | `[512]` | `float32` | Buffer |
| 447 | `img_backbone.layer2.2.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 448 | `img_backbone.layer2.3.conv1.weight` | `[128, 512, 1, 1]` | `float32` | Parameter |
| 449 | `img_backbone.layer2.3.bn1.weight` | `[128]` | `float32` | Parameter |
| 450 | `img_backbone.layer2.3.bn1.bias` | `[128]` | `float32` | Parameter |
| 451 | `img_backbone.layer2.3.bn1.running_mean` | `[128]` | `float32` | Buffer |
| 452 | `img_backbone.layer2.3.bn1.running_var` | `[128]` | `float32` | Buffer |
| 453 | `img_backbone.layer2.3.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 454 | `img_backbone.layer2.3.conv2.weight` | `[128, 128, 3, 3]` | `float32` | Parameter |
| 455 | `img_backbone.layer2.3.bn2.weight` | `[128]` | `float32` | Parameter |
| 456 | `img_backbone.layer2.3.bn2.bias` | `[128]` | `float32` | Parameter |
| 457 | `img_backbone.layer2.3.bn2.running_mean` | `[128]` | `float32` | Buffer |
| 458 | `img_backbone.layer2.3.bn2.running_var` | `[128]` | `float32` | Buffer |
| 459 | `img_backbone.layer2.3.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 460 | `img_backbone.layer2.3.conv3.weight` | `[512, 128, 1, 1]` | `float32` | Parameter |
| 461 | `img_backbone.layer2.3.bn3.weight` | `[512]` | `float32` | Parameter |
| 462 | `img_backbone.layer2.3.bn3.bias` | `[512]` | `float32` | Parameter |
| 463 | `img_backbone.layer2.3.bn3.running_mean` | `[512]` | `float32` | Buffer |
| 464 | `img_backbone.layer2.3.bn3.running_var` | `[512]` | `float32` | Buffer |
| 465 | `img_backbone.layer2.3.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 466 | `img_backbone.layer3.0.conv1.weight` | `[256, 512, 1, 1]` | `float32` | Parameter |
| 467 | `img_backbone.layer3.0.bn1.weight` | `[256]` | `float32` | Parameter |
| 468 | `img_backbone.layer3.0.bn1.bias` | `[256]` | `float32` | Parameter |
| 469 | `img_backbone.layer3.0.bn1.running_mean` | `[256]` | `float32` | Buffer |
| 470 | `img_backbone.layer3.0.bn1.running_var` | `[256]` | `float32` | Buffer |
| 471 | `img_backbone.layer3.0.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 472 | `img_backbone.layer3.0.conv2.weight` | `[256, 256, 3, 3]` | `float32` | Parameter |
| 473 | `img_backbone.layer3.0.bn2.weight` | `[256]` | `float32` | Parameter |
| 474 | `img_backbone.layer3.0.bn2.bias` | `[256]` | `float32` | Parameter |
| 475 | `img_backbone.layer3.0.bn2.running_mean` | `[256]` | `float32` | Buffer |
| 476 | `img_backbone.layer3.0.bn2.running_var` | `[256]` | `float32` | Buffer |
| 477 | `img_backbone.layer3.0.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 478 | `img_backbone.layer3.0.conv3.weight` | `[1024, 256, 1, 1]` | `float32` | Parameter |
| 479 | `img_backbone.layer3.0.bn3.weight` | `[1024]` | `float32` | Parameter |
| 480 | `img_backbone.layer3.0.bn3.bias` | `[1024]` | `float32` | Parameter |
| 481 | `img_backbone.layer3.0.bn3.running_mean` | `[1024]` | `float32` | Buffer |
| 482 | `img_backbone.layer3.0.bn3.running_var` | `[1024]` | `float32` | Buffer |
| 483 | `img_backbone.layer3.0.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 484 | `img_backbone.layer3.0.downsample.0.weight` | `[1024, 512, 1, 1]` | `float32` | Parameter |
| 485 | `img_backbone.layer3.0.downsample.1.weight` | `[1024]` | `float32` | Parameter |
| 486 | `img_backbone.layer3.0.downsample.1.bias` | `[1024]` | `float32` | Parameter |
| 487 | `img_backbone.layer3.0.downsample.1.running_mean` | `[1024]` | `float32` | Buffer |
| 488 | `img_backbone.layer3.0.downsample.1.running_var` | `[1024]` | `float32` | Buffer |
| 489 | `img_backbone.layer3.0.downsample.1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 490 | `img_backbone.layer3.1.conv1.weight` | `[256, 1024, 1, 1]` | `float32` | Parameter |
| 491 | `img_backbone.layer3.1.bn1.weight` | `[256]` | `float32` | Parameter |
| 492 | `img_backbone.layer3.1.bn1.bias` | `[256]` | `float32` | Parameter |
| 493 | `img_backbone.layer3.1.bn1.running_mean` | `[256]` | `float32` | Buffer |
| 494 | `img_backbone.layer3.1.bn1.running_var` | `[256]` | `float32` | Buffer |
| 495 | `img_backbone.layer3.1.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 496 | `img_backbone.layer3.1.conv2.weight` | `[256, 256, 3, 3]` | `float32` | Parameter |
| 497 | `img_backbone.layer3.1.bn2.weight` | `[256]` | `float32` | Parameter |
| 498 | `img_backbone.layer3.1.bn2.bias` | `[256]` | `float32` | Parameter |
| 499 | `img_backbone.layer3.1.bn2.running_mean` | `[256]` | `float32` | Buffer |
| 500 | `img_backbone.layer3.1.bn2.running_var` | `[256]` | `float32` | Buffer |
| 501 | `img_backbone.layer3.1.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 502 | `img_backbone.layer3.1.conv3.weight` | `[1024, 256, 1, 1]` | `float32` | Parameter |
| 503 | `img_backbone.layer3.1.bn3.weight` | `[1024]` | `float32` | Parameter |
| 504 | `img_backbone.layer3.1.bn3.bias` | `[1024]` | `float32` | Parameter |
| 505 | `img_backbone.layer3.1.bn3.running_mean` | `[1024]` | `float32` | Buffer |
| 506 | `img_backbone.layer3.1.bn3.running_var` | `[1024]` | `float32` | Buffer |
| 507 | `img_backbone.layer3.1.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 508 | `img_backbone.layer3.2.conv1.weight` | `[256, 1024, 1, 1]` | `float32` | Parameter |
| 509 | `img_backbone.layer3.2.bn1.weight` | `[256]` | `float32` | Parameter |
| 510 | `img_backbone.layer3.2.bn1.bias` | `[256]` | `float32` | Parameter |
| 511 | `img_backbone.layer3.2.bn1.running_mean` | `[256]` | `float32` | Buffer |
| 512 | `img_backbone.layer3.2.bn1.running_var` | `[256]` | `float32` | Buffer |
| 513 | `img_backbone.layer3.2.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 514 | `img_backbone.layer3.2.conv2.weight` | `[256, 256, 3, 3]` | `float32` | Parameter |
| 515 | `img_backbone.layer3.2.bn2.weight` | `[256]` | `float32` | Parameter |
| 516 | `img_backbone.layer3.2.bn2.bias` | `[256]` | `float32` | Parameter |
| 517 | `img_backbone.layer3.2.bn2.running_mean` | `[256]` | `float32` | Buffer |
| 518 | `img_backbone.layer3.2.bn2.running_var` | `[256]` | `float32` | Buffer |
| 519 | `img_backbone.layer3.2.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 520 | `img_backbone.layer3.2.conv3.weight` | `[1024, 256, 1, 1]` | `float32` | Parameter |
| 521 | `img_backbone.layer3.2.bn3.weight` | `[1024]` | `float32` | Parameter |
| 522 | `img_backbone.layer3.2.bn3.bias` | `[1024]` | `float32` | Parameter |
| 523 | `img_backbone.layer3.2.bn3.running_mean` | `[1024]` | `float32` | Buffer |
| 524 | `img_backbone.layer3.2.bn3.running_var` | `[1024]` | `float32` | Buffer |
| 525 | `img_backbone.layer3.2.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 526 | `img_backbone.layer3.3.conv1.weight` | `[256, 1024, 1, 1]` | `float32` | Parameter |
| 527 | `img_backbone.layer3.3.bn1.weight` | `[256]` | `float32` | Parameter |
| 528 | `img_backbone.layer3.3.bn1.bias` | `[256]` | `float32` | Parameter |
| 529 | `img_backbone.layer3.3.bn1.running_mean` | `[256]` | `float32` | Buffer |
| 530 | `img_backbone.layer3.3.bn1.running_var` | `[256]` | `float32` | Buffer |
| 531 | `img_backbone.layer3.3.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 532 | `img_backbone.layer3.3.conv2.weight` | `[256, 256, 3, 3]` | `float32` | Parameter |
| 533 | `img_backbone.layer3.3.bn2.weight` | `[256]` | `float32` | Parameter |
| 534 | `img_backbone.layer3.3.bn2.bias` | `[256]` | `float32` | Parameter |
| 535 | `img_backbone.layer3.3.bn2.running_mean` | `[256]` | `float32` | Buffer |
| 536 | `img_backbone.layer3.3.bn2.running_var` | `[256]` | `float32` | Buffer |
| 537 | `img_backbone.layer3.3.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 538 | `img_backbone.layer3.3.conv3.weight` | `[1024, 256, 1, 1]` | `float32` | Parameter |
| 539 | `img_backbone.layer3.3.bn3.weight` | `[1024]` | `float32` | Parameter |
| 540 | `img_backbone.layer3.3.bn3.bias` | `[1024]` | `float32` | Parameter |
| 541 | `img_backbone.layer3.3.bn3.running_mean` | `[1024]` | `float32` | Buffer |
| 542 | `img_backbone.layer3.3.bn3.running_var` | `[1024]` | `float32` | Buffer |
| 543 | `img_backbone.layer3.3.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 544 | `img_backbone.layer3.4.conv1.weight` | `[256, 1024, 1, 1]` | `float32` | Parameter |
| 545 | `img_backbone.layer3.4.bn1.weight` | `[256]` | `float32` | Parameter |
| 546 | `img_backbone.layer3.4.bn1.bias` | `[256]` | `float32` | Parameter |
| 547 | `img_backbone.layer3.4.bn1.running_mean` | `[256]` | `float32` | Buffer |
| 548 | `img_backbone.layer3.4.bn1.running_var` | `[256]` | `float32` | Buffer |
| 549 | `img_backbone.layer3.4.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 550 | `img_backbone.layer3.4.conv2.weight` | `[256, 256, 3, 3]` | `float32` | Parameter |
| 551 | `img_backbone.layer3.4.bn2.weight` | `[256]` | `float32` | Parameter |
| 552 | `img_backbone.layer3.4.bn2.bias` | `[256]` | `float32` | Parameter |
| 553 | `img_backbone.layer3.4.bn2.running_mean` | `[256]` | `float32` | Buffer |
| 554 | `img_backbone.layer3.4.bn2.running_var` | `[256]` | `float32` | Buffer |
| 555 | `img_backbone.layer3.4.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 556 | `img_backbone.layer3.4.conv3.weight` | `[1024, 256, 1, 1]` | `float32` | Parameter |
| 557 | `img_backbone.layer3.4.bn3.weight` | `[1024]` | `float32` | Parameter |
| 558 | `img_backbone.layer3.4.bn3.bias` | `[1024]` | `float32` | Parameter |
| 559 | `img_backbone.layer3.4.bn3.running_mean` | `[1024]` | `float32` | Buffer |
| 560 | `img_backbone.layer3.4.bn3.running_var` | `[1024]` | `float32` | Buffer |
| 561 | `img_backbone.layer3.4.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 562 | `img_backbone.layer3.5.conv1.weight` | `[256, 1024, 1, 1]` | `float32` | Parameter |
| 563 | `img_backbone.layer3.5.bn1.weight` | `[256]` | `float32` | Parameter |
| 564 | `img_backbone.layer3.5.bn1.bias` | `[256]` | `float32` | Parameter |
| 565 | `img_backbone.layer3.5.bn1.running_mean` | `[256]` | `float32` | Buffer |
| 566 | `img_backbone.layer3.5.bn1.running_var` | `[256]` | `float32` | Buffer |
| 567 | `img_backbone.layer3.5.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 568 | `img_backbone.layer3.5.conv2.weight` | `[256, 256, 3, 3]` | `float32` | Parameter |
| 569 | `img_backbone.layer3.5.bn2.weight` | `[256]` | `float32` | Parameter |
| 570 | `img_backbone.layer3.5.bn2.bias` | `[256]` | `float32` | Parameter |
| 571 | `img_backbone.layer3.5.bn2.running_mean` | `[256]` | `float32` | Buffer |
| 572 | `img_backbone.layer3.5.bn2.running_var` | `[256]` | `float32` | Buffer |
| 573 | `img_backbone.layer3.5.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 574 | `img_backbone.layer3.5.conv3.weight` | `[1024, 256, 1, 1]` | `float32` | Parameter |
| 575 | `img_backbone.layer3.5.bn3.weight` | `[1024]` | `float32` | Parameter |
| 576 | `img_backbone.layer3.5.bn3.bias` | `[1024]` | `float32` | Parameter |
| 577 | `img_backbone.layer3.5.bn3.running_mean` | `[1024]` | `float32` | Buffer |
| 578 | `img_backbone.layer3.5.bn3.running_var` | `[1024]` | `float32` | Buffer |
| 579 | `img_backbone.layer3.5.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 580 | `img_backbone.layer4.0.conv1.weight` | `[512, 1024, 1, 1]` | `float32` | Parameter |
| 581 | `img_backbone.layer4.0.bn1.weight` | `[512]` | `float32` | Parameter |
| 582 | `img_backbone.layer4.0.bn1.bias` | `[512]` | `float32` | Parameter |
| 583 | `img_backbone.layer4.0.bn1.running_mean` | `[512]` | `float32` | Buffer |
| 584 | `img_backbone.layer4.0.bn1.running_var` | `[512]` | `float32` | Buffer |
| 585 | `img_backbone.layer4.0.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 586 | `img_backbone.layer4.0.conv2.weight` | `[512, 512, 3, 3]` | `float32` | Parameter |
| 587 | `img_backbone.layer4.0.bn2.weight` | `[512]` | `float32` | Parameter |
| 588 | `img_backbone.layer4.0.bn2.bias` | `[512]` | `float32` | Parameter |
| 589 | `img_backbone.layer4.0.bn2.running_mean` | `[512]` | `float32` | Buffer |
| 590 | `img_backbone.layer4.0.bn2.running_var` | `[512]` | `float32` | Buffer |
| 591 | `img_backbone.layer4.0.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 592 | `img_backbone.layer4.0.conv3.weight` | `[2048, 512, 1, 1]` | `float32` | Parameter |
| 593 | `img_backbone.layer4.0.bn3.weight` | `[2048]` | `float32` | Parameter |
| 594 | `img_backbone.layer4.0.bn3.bias` | `[2048]` | `float32` | Parameter |
| 595 | `img_backbone.layer4.0.bn3.running_mean` | `[2048]` | `float32` | Buffer |
| 596 | `img_backbone.layer4.0.bn3.running_var` | `[2048]` | `float32` | Buffer |
| 597 | `img_backbone.layer4.0.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 598 | `img_backbone.layer4.0.downsample.0.weight` | `[2048, 1024, 1, 1]` | `float32` | Parameter |
| 599 | `img_backbone.layer4.0.downsample.1.weight` | `[2048]` | `float32` | Parameter |
| 600 | `img_backbone.layer4.0.downsample.1.bias` | `[2048]` | `float32` | Parameter |
| 601 | `img_backbone.layer4.0.downsample.1.running_mean` | `[2048]` | `float32` | Buffer |
| 602 | `img_backbone.layer4.0.downsample.1.running_var` | `[2048]` | `float32` | Buffer |
| 603 | `img_backbone.layer4.0.downsample.1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 604 | `img_backbone.layer4.1.conv1.weight` | `[512, 2048, 1, 1]` | `float32` | Parameter |
| 605 | `img_backbone.layer4.1.bn1.weight` | `[512]` | `float32` | Parameter |
| 606 | `img_backbone.layer4.1.bn1.bias` | `[512]` | `float32` | Parameter |
| 607 | `img_backbone.layer4.1.bn1.running_mean` | `[512]` | `float32` | Buffer |
| 608 | `img_backbone.layer4.1.bn1.running_var` | `[512]` | `float32` | Buffer |
| 609 | `img_backbone.layer4.1.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 610 | `img_backbone.layer4.1.conv2.weight` | `[512, 512, 3, 3]` | `float32` | Parameter |
| 611 | `img_backbone.layer4.1.bn2.weight` | `[512]` | `float32` | Parameter |
| 612 | `img_backbone.layer4.1.bn2.bias` | `[512]` | `float32` | Parameter |
| 613 | `img_backbone.layer4.1.bn2.running_mean` | `[512]` | `float32` | Buffer |
| 614 | `img_backbone.layer4.1.bn2.running_var` | `[512]` | `float32` | Buffer |
| 615 | `img_backbone.layer4.1.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 616 | `img_backbone.layer4.1.conv3.weight` | `[2048, 512, 1, 1]` | `float32` | Parameter |
| 617 | `img_backbone.layer4.1.bn3.weight` | `[2048]` | `float32` | Parameter |
| 618 | `img_backbone.layer4.1.bn3.bias` | `[2048]` | `float32` | Parameter |
| 619 | `img_backbone.layer4.1.bn3.running_mean` | `[2048]` | `float32` | Buffer |
| 620 | `img_backbone.layer4.1.bn3.running_var` | `[2048]` | `float32` | Buffer |
| 621 | `img_backbone.layer4.1.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 622 | `img_backbone.layer4.2.conv1.weight` | `[512, 2048, 1, 1]` | `float32` | Parameter |
| 623 | `img_backbone.layer4.2.bn1.weight` | `[512]` | `float32` | Parameter |
| 624 | `img_backbone.layer4.2.bn1.bias` | `[512]` | `float32` | Parameter |
| 625 | `img_backbone.layer4.2.bn1.running_mean` | `[512]` | `float32` | Buffer |
| 626 | `img_backbone.layer4.2.bn1.running_var` | `[512]` | `float32` | Buffer |
| 627 | `img_backbone.layer4.2.bn1.num_batches_tracked` | `[]` | `int64` | Buffer |
| 628 | `img_backbone.layer4.2.conv2.weight` | `[512, 512, 3, 3]` | `float32` | Parameter |
| 629 | `img_backbone.layer4.2.bn2.weight` | `[512]` | `float32` | Parameter |
| 630 | `img_backbone.layer4.2.bn2.bias` | `[512]` | `float32` | Parameter |
| 631 | `img_backbone.layer4.2.bn2.running_mean` | `[512]` | `float32` | Buffer |
| 632 | `img_backbone.layer4.2.bn2.running_var` | `[512]` | `float32` | Buffer |
| 633 | `img_backbone.layer4.2.bn2.num_batches_tracked` | `[]` | `int64` | Buffer |
| 634 | `img_backbone.layer4.2.conv3.weight` | `[2048, 512, 1, 1]` | `float32` | Parameter |
| 635 | `img_backbone.layer4.2.bn3.weight` | `[2048]` | `float32` | Parameter |
| 636 | `img_backbone.layer4.2.bn3.bias` | `[2048]` | `float32` | Parameter |
| 637 | `img_backbone.layer4.2.bn3.running_mean` | `[2048]` | `float32` | Buffer |
| 638 | `img_backbone.layer4.2.bn3.running_var` | `[2048]` | `float32` | Buffer |
| 639 | `img_backbone.layer4.2.bn3.num_batches_tracked` | `[]` | `int64` | Buffer |
| 640 | `img_neck.lateral_convs.0.conv.weight` | `[256, 2048, 1, 1]` | `float32` | Parameter |
| 641 | `img_neck.lateral_convs.0.conv.bias` | `[256]` | `float32` | Parameter |
| 642 | `img_neck.fpn_convs.0.conv.weight` | `[256, 256, 3, 3]` | `float32` | Parameter |
| 643 | `img_neck.fpn_convs.0.conv.bias` | `[256]` | `float32` | Parameter |
