#!/bin/bash
#SBATCH --job-name=tiger-f0-10-100
#SBATCH --account=pause
#SBATCH --partition=gpu
#SBATCH --qos=gpu_48h
#SBATCH --gpus=rtx3090:6
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=15:00:00
#SBATCH --output=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.out
#SBATCH --error=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.err

# Nettoyer les variables d'environnement Slurm qui perturbent PyTorch DDP
unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS

cd /idiap/home/adubois/project_ma1/TIGER
mkdir -p logs

PYTHON="/idiap/temp/adubois/miniconda3/envs/tiger/bin/python"
CONFIG_DIR="/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned/training_per_feature/f0_mean"

echo "=========================================================="
echo "Starting multi-run sequence (top20 to top100) on $(hostname)"
echo "Start time: $(date)"
echo "=========================================================="

for x in 20 30 40 50 60 70 80 90 100; do
    CONF="${CONFIG_DIR}/top${x}_f0mean.yml"
    
    if [ ! -f "$CONF" ]; then
        echo "[WARNING] Config file $CONF not found, skipping..."
        continue
    fi

    echo ""
    echo "----------------------------------------------------------"
    echo "[$(date)] Starting Training: top${x}_f0mean"
    echo "Config: $CONF"
    echo "----------------------------------------------------------"

    $PYTHON audio_train.py --conf_dir "$CONF"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Training top${x}_f0mean failed with exit code $EXIT_CODE"
        # Optionnel: décommentez 'exit $EXIT_CODE' si vous voulez stopper la suite en cas d'erreur
    else
        echo "[SUCCESS] Finished Training top${x}_f0mean at $(date)"
    fi
done

echo ""
echo "=========================================================="
echo "All runs completed at $(date)"
echo "=========================================================="
