#!/bin/bash
#SBATCH --job-name=tiger-resume-deciles
#SBATCH --account=pause
#SBATCH --partition=gpu
#SBATCH --qos=gpu_48h
#SBATCH --gpus=rtx3090:6
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.out
#SBATCH --error=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.err

# Nettoyer les variables d'environnement Slurm perturbatrices pour PyTorch DDP
unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS

cd /idiap/home/adubois/project_ma1/TIGER
mkdir -p logs

PYTHON="/idiap/temp/adubois/miniconda3/envs/tiger/bin/python"
DIR="/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned/training_per_feature/f0_mean/deciles"

echo "=========================================================="
echo "Pursuing Training of All 10 Deciles (+300 epochs) on $(hostname)"
echo "Start time: $(date)"
echo "=========================================================="

for d in {1..5}; do
    CONF="${DIR}/decile${d}_f0mean.yml"
    
    echo ""
    echo "----------------------------------------------------------"
    echo "[$(date)] Resuming Training: Decile ${d}"
    echo "Config: $CONF"
    echo "----------------------------------------------------------"

    $PYTHON audio_train.py --conf_dir "$CONF"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Decile ${d} exited with code $EXIT_CODE"
    else
        echo "[SUCCESS] Finished Decile ${d} at $(date)"
    fi
done

echo ""
echo "=========================================================="
echo "All decile runs completed at $(date)"
echo "=========================================================="
