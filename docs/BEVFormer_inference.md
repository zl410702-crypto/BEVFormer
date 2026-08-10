# BEVFormer Quick Inference

## 1. Activate environment

```bash
conda activate bevformer
```

进入工程：

```bash
cd ~/workspace/bevformer/BEVFormer
```

---

## 2. Set CUDA environment

```bash
export CUDA_HOME=/usr/local/cuda-11.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

---

## 3. Prepare dataset

Dataset: 数据已经生成，不需要重复执行

```
/data/bevformer/nuscenes
/data/bevformer/can_bus
```

生成 nuScenes info:

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH

python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes \
  --version v1.0-mini \
  --canbus ./data
```

---

## 4. Run inference

Checkpoint:

```
/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth
```

Run:

```bash
./tools/dist_test.sh \
projects/configs/bevformer/bevformer_tiny.py \
/data/bevformer/checkpoints/bevformer_tiny_epoch_24.pth \
1
```

参数:

```
config:
projects/configs/bevformer/bevformer_tiny.py

checkpoint:
bevformer_tiny_epoch_24.pth

GPU:
1
```

---

## 5. Verify result

Expected:

```
mAP ≈ 0.24

NDS ≈ 0.30
```

Example:

```
mAP: 0.24398
NDS: 0.30134
```

表示 BEVFormer Python inference 正常。
