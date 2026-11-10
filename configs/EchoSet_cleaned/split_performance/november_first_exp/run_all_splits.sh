#!/bin/bash
#SBATCH --job-name=tiger-splits-nov
#SBATCH --account=pause
#SBATCH --partition=gpu
#SBATCH --qos=gpu_48h
#SBATCH --gpus=rtx3090:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=47:00:00
#SBATCH --output=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.out
#SBATCH --error=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.err

# Nettoyer les variables Slurm perturbatrices pour DDP
unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS

cd /idiap/home/adubois/project_ma1/TIGER
mkdir -p logs

PYTHON="/idiap/temp/adubois/miniconda3/envs/tiger/bin/python"
DIR="/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned/split_performance/november_first_exp"

# Liste ordonnée des 4 modèles à entraîner
CONFIGS=(
    "baseline.yml"
    "even_split.yml"
    "normal_split.yml"
    # "non_split.yml"
)

echo "=========================================================="
echo "Starting Sequential Training of 4 Split Experiments on $(hostname)"
echo "Start time: $(date)"
echo "=========================================================="

for conf_name in "${CONFIGS[@]}"; do
    CONF="${DIR}/${conf_name}"
    
    if [ ! -f "$CONF" ]; then
        echo "[WARNING] Config $CONF not found, skipping..."
        continue
    fi

    echo ""
    echo "----------------------------------------------------------"
    echo "[$(date)] Starting Training: ${conf_name}"
    echo "Config: $CONF"
    echo "----------------------------------------------------------"

    $PYTHON audio_train.py --conf_dir "$CONF"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "[ERROR] Run ${conf_name} failed with exit code $EXIT_CODE"
    else
        echo "[SUCCESS] Finished ${conf_name} at $(date)"
    fi
done

echo ""
echo "=========================================================="
echo "All 4 split experiments finished at $(date)"
echo "=========================================================="