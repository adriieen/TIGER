#!/bin/bash
#SBATCH --job-name=tiger-all-alpha
#SBATCH --account=pause
#SBATCH --partition=gpu
#SBATCH --qos=gpu_48h
#SBATCH --gpus=rtx3090:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.out
#SBATCH --error=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.err

unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS
cd /idiap/home/adubois/project_ma1/TIGER
mkdir -p logs

PYTHON="/idiap/temp/adubois/miniconda3/envs/tiger/bin/python"
ALPHA_DIR="/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned/training_per_feature/alpha_ratio/deciles"
F0_DIR="/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned/training_per_feature/f0_mean/deciles"

echo "=========================================================="
echo "Job Started on $(hostname) at $(date)"
echo "=========================================================="

# 1. Relance prioritaire du Décile 4 de f0_mean qui a crashé
echo ""
echo "----------------------------------------------------------"
echo "[$(date)] Re-running failed f0_mean Decile 4..."
echo "----------------------------------------------------------"
$PYTHON audio_train.py --conf_dir "${F0_DIR}/decile4_f0mean.yml"
echo "[$(date)] Finished f0_mean Decile 4 with exit code $?"

# 2. Entraînement séquentiel des 10 déciles alpha_ratio
for d in {1..10}; do
    CONF="${ALPHA_DIR}/decile${d}_alpharatio.yml"
    echo ""
    echo "----------------------------------------------------------"
    echo "[$(date)] Starting Training: AlphaRatio Decile ${d}"
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
echo "All runs completed at $(date)"
echo "=========================================================="
