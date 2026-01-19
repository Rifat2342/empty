#!/bin/bash
#
#SBATCH --partition=gpu_min24gb   # Partition where the job will be run. Check with "$ sinfo".
#SBATCH --qos=gpu_min24gb         # QoS level. Must match the partition name. External users must add the suffix "_ext". Check with "$sacctmgr show qos".
#SBATCH --job-name=task2_baseline # Job name
#SBATCH --output=logs/slurm_%x.%j.out  # File containing STDOUT output
#SBATCH --error=logs/slurm_%x.%j.err   # File containing STDERR output. If ommited, use STDOUT.


set -euo pipefail

# Configuration (override with environment variables or edit here).
REPO_ROOT=${REPO_ROOT:-/nas-ctm01/homes/fmribeiro/icassp2026-converge-gc}
DATASET_ROOT=${DATASET_ROOT:-/nas-ctm01/datasets/public/ICASSP2026_CONVERGE_GC/dataset}
CONDA_ENV=${CONDA_ENV:-task2-baseline}
TRAIN_SCENARIOS=${TRAIN_SCENARIOS:-exp6,exp7}
VAL_SCENARIOS=${VAL_SCENARIOS:-exp8}
OUT_DIR=${OUT_DIR:-runs/task2_baseline}

VIDEO_MODE=${VIDEO_MODE:-rgbd}
IMAGE_SIZE=${IMAGE_SIZE:-128}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-16}
LR=${LR:-1e-3}
NUM_WORKERS=${NUM_WORKERS:-4}

LABEL_TOL=${LABEL_TOL:-0.2}
E2_TOL=${E2_TOL:-0.05}
LOSS=${LOSS:-huber}

BACKBONE=${BACKBONE:-resnet18}
PRETRAINED=${PRETRAINED:-1}
FREEZE_BACKBONE=${FREEZE_BACKBONE:-1}

PRESET=${PRESET:-loso}
HOLDOUT_SCENARIO=${HOLDOUT_SCENARIO:-exp8}
DEVICE=${DEVICE:-}

cd "$REPO_ROOT"

# Load conda (adjust to your installation if needed).
CONDA_ROOT=${CONDA_ROOT:-/usr/local/miniconda3}
if [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Could not find conda.sh; set CONDA_ROOT or update this script." >&2
  exit 1
fi

# Some conda activate hooks reference unset vars; disable nounset temporarily.
set +u
conda activate "$CONDA_ENV"
set -u

if ! python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("task2_baseline") else 1)
PY
then
  python -m pip install -e "$REPO_ROOT/baselines/task2_pytorch"
fi

ARGS=(
  --dataset-root "$DATASET_ROOT"
  --train-scenarios "$TRAIN_SCENARIOS"
  --val-scenarios "$VAL_SCENARIOS"
  --video-mode "$VIDEO_MODE"
  --image-size "$IMAGE_SIZE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --num-workers "$NUM_WORKERS"
  --label-tolerance "$LABEL_TOL"
  --e2-tolerance "$E2_TOL"
  --out-dir "$OUT_DIR"
  --loss "$LOSS"
  --backbone "$BACKBONE"
)

if [ "$PRETRAINED" = "1" ]; then
  ARGS+=(--pretrained)
fi

if [ "$FREEZE_BACKBONE" = "1" ]; then
  ARGS+=(--freeze-backbone)
fi

if [ "$PRESET" != "none" ]; then
  ARGS+=(--preset "$PRESET")
  if [ "$PRESET" = "loso" ] && [ -n "$HOLDOUT_SCENARIO" ]; then
    ARGS+=(--holdout-scenario "$HOLDOUT_SCENARIO")
  fi
fi

if [ -n "$DEVICE" ]; then
  ARGS+=(--device "$DEVICE")
fi

python -m task2_baseline.train "${ARGS[@]}"
