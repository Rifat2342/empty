# Task 2 PyTorch Baseline

This package provides a simple multi-modal baseline for Task 2 pose
regression using RGB-D video frames and radio E2 features. It aligns video,
radio, and annotations by timestamp and predicts the UE translation
`(x, y, z)` in the liteon frame at time `t`.

## Setup
Create a conda environment and install dependencies:

```bash
conda env create -f baselines/task2_pytorch/environment.yml
conda activate task2-baseline
pip install -e baselines/task2_pytorch
```

GPU setup (CUDA):

```bash
conda env create -f baselines/task2_pytorch/environment.gpu.yml
conda activate task2-baseline
pip install -e baselines/task2_pytorch
```

Training will automatically use CUDA if available, or you can force it with
`--device cuda`.

## Quick start
Train on exp6..exp7 and validate on exp8:

```bash
python -m task2_baseline.train \
  --dataset-root dataset \
  --train-scenarios exp6,exp7 \
  --val-scenarios exp8 \
  --video-mode rgbd \
  --image-size 128 \
  --epochs 20 \
  --batch-size 16
```

Leave-one-scenario-out preset (hold out exp8):

```bash
python -m task2_baseline.train \
  --dataset-root dataset \
  --preset loso \
  --holdout-scenario exp8 \
  --video-mode rgbd \
  --image-size 128 \
  --epochs 20 \
  --batch-size 16
```

Pretrained visual backbone (useful for small datasets):

```bash
python -m task2_baseline.train \
  --dataset-root dataset \
  --backbone resnet18 \
  --pretrained \
  --freeze-backbone \
  --video-mode rgbd \
  --epochs 20
```

Note: `--pretrained` downloads torchvision weights if not cached. If you are
offline, pre-download the weights once or copy them into your Torch cache.

Evaluate a saved checkpoint:

```bash
python -m task2_baseline.eval \
  --dataset-root dataset \
  --scenarios exp8 \
  --checkpoint runs/task2_baseline/best.pt
```

Write per-frame predictions:

```bash
python -m task2_baseline.predict \
  --dataset-root dataset \
  --scenarios exp8 \
  --checkpoint runs/task2_baseline/best.pt \
  --output runs/task2_baseline/exp8_predictions.csv
```

## Model overview
The Task 2 baseline regresses 3D UE position with a fused video + radio model:
- Visual branch: lightweight 3-layer CNN or ResNet18 backbone, with the first
  convolution adapted for RGB-D (4 ch), RGB (3 ch), or disparity-only (1 ch).
- Radio branch: a 2-layer MLP that embeds the E2 feature vector.
- Fusion: concatenate visual and radio features, then regress (x, y, z) with a
  small MLP head.
- Normalization: E2 features and target positions are standardized with train-set
  statistics (metrics are reported in millimeters after denormalization).
- Loss/metrics: Huber (default) or MSE loss; reports MAE and RMSE.

## Performance
Best validation metrics observed in the current tests: val loss 1.0093, MAE 554.85 mm, RMSE 777.68 mm.

## Notes
- `dataset/index.csv` is used to locate frames, radio, and annotations.
- Targets are normalized using training-set statistics and stored in the
  checkpoint. Metrics are reported in millimeters.
- `--video-mode none` enables a radio-only baseline.
