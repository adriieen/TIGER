#!/bin/bash
#SBATCH --job-name=deltaf0-router
#SBATCH --account=pause
#SBATCH --partition=gpu
#SBATCH --qos=gpu_8h
#SBATCH --gpus=rtx3090:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.out
#SBATCH --error=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.err

set -e

cd /idiap/home/adubois/project_ma1/TIGER/mixture_of_experts
PYTHON="/idiap/temp/adubois/miniconda3/envs/tiger/bin/python"

RUN_NAME="175k_gaussian_target"
CKPT_DIR="checkpoints/${RUN_NAME}"
EVAL_DIR="eval_results/${RUN_NAME}"

echo "=========================================================="
echo "Training DeltaF0Router (${RUN_NAME}) on $(hostname) at $(date)"
echo "=========================================================="

$PYTHON train_router.py --epochs 200 --batch_size 64 --num_workers 4 --gaussian_target --save_dir "${CKPT_DIR}"

echo ""
echo "=========================================================="
echo "Evaluating DeltaF0Router (${RUN_NAME}) on Test Set (6.0s & 2.0s)"
echo "=========================================================="

$PYTHON eval_router.py \
  --ckpt_path "${CKPT_DIR}/best_router.pt" \
  --thresholds_path "${CKPT_DIR}/thresholds.json" \
  --out_dir "${EVAL_DIR}"

$PYTHON eval_router.py \
  --crop_sec 2.0 \
  --ckpt_path "${CKPT_DIR}/best_router.pt" \
  --thresholds_path "${CKPT_DIR}/thresholds.json" \
  --out_dir "${EVAL_DIR}"

echo "Done at $(date)"
