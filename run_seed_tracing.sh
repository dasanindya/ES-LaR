#!/bin/bash
#SBATCH --job-name=seedtrace_es
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --account=pi_dagarwal_umass_edu
#SBATCH --mail-user=nvunnava@umass.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#
# Usage:
#   sbatch --export=RUN_ID=1,ITERATIONS=20,POP_SIZE=10,SIGMA=0.001,ALPHA=0.0005 run_seed_tracing.sh
#
# RUN_ID encoding:
#   - RUN_ID  1-12  → K=5
#   - RUN_ID 13-24  → K=10
#   - RUN_ID 25-36  → K=20
#
# Within each K block (same as old mapping):
# 1: seed=17, thoughts=0
# 2: seed=17, thoughts=2
# 3: seed=17, thoughts=4
# 4: seed=17, thoughts=8
# 5: seed=33, thoughts=0
# 6: seed=33, thoughts=2
# 7: seed=33, thoughts=4
# 8: seed=33, thoughts=8
# 9: seed=51, thoughts=0
# 10: seed=51, thoughts=2
# 11: seed=51, thoughts=4
# 12: seed=51, thoughts=8
#
# 1: seed=17, thoughts=0
# 2: seed=17, thoughts=2
# 3: seed=17, thoughts=4
# 4: seed=17, thoughts=8
# 5: seed=33, thoughts=0
# 6: seed=33, thoughts=2
# 7: seed=33, thoughts=4
# 8: seed=33, thoughts=8
# 9: seed=51, thoughts=0
# 10: seed=51, thoughts=2
# 11: seed=51, thoughts=4
# 12: seed=51, thoughts=8

set -euo pipefail

RUN_ID=${RUN_ID:-${1:-1}}
if [[ ! "$RUN_ID" =~ ^[0-9]+$ || "$RUN_ID" -lt 1 || "$RUN_ID" -gt 36 ]]; then
  echo "Usage: sbatch --export=RUN_ID=1 run_seed_tracing.sh (1-36)"
  echo "RUN_ID  1-12: K=5  | 13-24: K=10 | 25-36: K=20"
  exit 1
fi

# Tunables (override via --export=... to sbatch)
ITERATIONS=${ITERATIONS:-200}
POP_SIZE=${POP_SIZE:-30}
SIGMA=${SIGMA:-0.001}
ALPHA=${ALPHA:-0.0005}

# K is determined by RUN_ID block unless explicitly set
if [[ "${K:-}" != "" ]]; then
  K="$K"
else
  if [[ "$RUN_ID" -le 12 ]]; then
    K=5
  elif [[ "$RUN_ID" -le 24 ]]; then
    K=10
  else
    K=20
  fi
fi
SIGMA=${SIGMA:-0.001}
ALPHA=${ALPHA:-0.0005}

# Map RUN_ID onto the 1-12 grid (seed/thoughts), repeated per K block
RUN_BASE=$(( (RUN_ID - 1) % 12 + 1 ))

case $RUN_BASE in
  1) SEED=17; THOUGHTS=0 ;;
  2) SEED=17; THOUGHTS=2 ;;
  3) SEED=17; THOUGHTS=4 ;;
  4) SEED=17; THOUGHTS=8 ;;
  5) SEED=33; THOUGHTS=0 ;;
  6) SEED=33; THOUGHTS=2 ;;
  7) SEED=33; THOUGHTS=4 ;;
  8) SEED=33; THOUGHTS=8 ;;
  9) SEED=51; THOUGHTS=0 ;;
  10) SEED=51; THOUGHTS=2 ;;
  11) SEED=51; THOUGHTS=4 ;;
  12) SEED=51; THOUGHTS=8 ;;
esac

echo "Running config $RUN_ID (base=$RUN_BASE): seed=$SEED, num_latent_thoughts=$THOUGHTS, iter=$ITERATIONS, pop=$POP_SIZE, k=$K"

umask 002
cd /work/pi_dagarwal_umass_edu/project_8/prathik/es-fine-tuning-paper
mkdir -p logs

module load conda/latest
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /work/pi_dagarwal_umass_edu/project_8/conda_env/es

nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'gpus:', torch.cuda.device_count())"

export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

export SCRATCH_WS="/scratch/workspace/nvunnava_umass_edu-simple"
export HF_CACHE="$SCRATCH_WS/hf_cache"
export HF_HOME="$HF_CACHE"
mkdir -p "$HF_CACHE"

export WANDB_RUN_NAME="${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
export WANDB_MODE=online
export WANDB_DIR="$SCRATCH_WS/wandb"
mkdir -p "$WANDB_DIR"

# Where the trainer writes seed traces + final model
export OUTPUT_DIR="$SCRATCH_WS/seedtrace_runs/seed${SEED}_thoughts${THOUGHTS}_k${K}_pop${POP_SIZE}_iter${ITERATIONS}_s${SIGMA}_a${ALPHA}"
mkdir -p "$OUTPUT_DIR"

accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --machine_rank 0 \
  countdown/es_fine-tuning_countdown_softmax_topk_seed_Tracing.py \
  --data_sample 100 \
  --model_name Qwen/Qwen2.5-3B \
  --gpu_threads 1 \
  --hf_cache_dir "$HF_HOME" \
  --eval_dataset_size 80 \
  --num_latent_thoughts "$THOUGHTS" \
  --initial_seed "$SEED" \
  --population_size "$POP_SIZE" \
  --num_iterations "$ITERATIONS" \
  --blend_top_k "$K" \
  --sigma "$SIGMA" \
  --alpha "$ALPHA" \
  --output_dir "$OUTPUT_DIR" \
  --wandb_project "es-countdown-qwen-softmax-topk-new" \
  --wandb_entity nvunnava-umass-amherst \
  --wandb_run_name "seed${SEED}_thoughts${THOUGHTS}_k${K}_pop${POP_SIZE}_iter${ITERATIONS}_s${SIGMA}_a${ALPHA}"

