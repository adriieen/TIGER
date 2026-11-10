#!/bin/bash
#SBATCH --job-name=tiger-all-quintiles
#SBATCH --account=pause
#SBATCH --partition=gpu
#SBATCH --qos=gpu_8h
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.out
#SBATCH --error=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.err

unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS

cd /idiap/home/adubois/project_ma1/TIGER
mkdir -p logs

PYTHON="/idiap/temp/adubois/miniconda3/envs/tiger/bin/python"
DIR="/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned/training_per_feature/f0_mean/quintiles"

echo "=========================================================="
echo "Starting Sequential Training of All 5 quintiles on $(hostname)"
echo "Start time: $(date)"
echo "=========================================================="

for q in {1..5}; do
    CONF="${DIR}/quintile${q}_f0mean.yml"
    echo ""
    echo "----------------------------------------------------------"
    echo "[$(date)] Starting Training: Quintile ${q}"
    echo "Config: $CONF"
    echo "----------------------------------------------------------"

    $PYTHON audio_train.py --conf_dir "$CONF"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Quintile ${q} failed with exit code $EXIT_CODE"
    else
        echo "[SUCCESS] Finished Quintile ${q} at $(date)"
    fi
done

echo ""
echo "=========================================================="
echo "All 5 quintiles completed at $(date)"
echo "=========================================================="
