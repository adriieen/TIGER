#!/bin/bash
#SBATCH --job-name=tiger-all-deciles
#SBATCH --account=pause
#SBATCH --partition=gpu
#SBATCH --qos=gpu_48h
#SBATCH --gpus=rtx3090:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=47:00:00
#SBATCH --output=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.out
#SBATCH --error=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.err

unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS

cd /idiap/home/adubois/project_ma1/TIGER
mkdir -p logs

PYTHON="/idiap/temp/adubois/miniconda3/envs/tiger/bin/python"
DIR="/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned/training_per_feature/f0_mean/deciles"

echo "=========================================================="
echo "Starting Sequential Training of All 10 Deciles on $(hostname)"
echo "Start time: $(date)"
echo "=========================================================="

for d in {1..10}; do
    CONF="${DIR}/decile${d}_f0mean.yml"
    echo ""
    echo "----------------------------------------------------------"
    echo "[$(date)] Starting Training: Decile ${d}"
    echo "Config: $CONF"
    echo "----------------------------------------------------------"

    $PYTHON audio_train.py --conf_dir "$CONF"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Decile ${d} failed with exit code $EXIT_CODE"
    else
        echo "[SUCCESS] Finished Decile ${d} at $(date)"
    fi
done

echo ""
echo "=========================================================="
echo "All 10 deciles completed at $(date)"
echo "=========================================================="
