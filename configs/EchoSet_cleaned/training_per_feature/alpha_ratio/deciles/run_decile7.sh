#!/bin/bash
#SBATCH --job-name=tiger-d7-alpha
#SBATCH --account=pause
#SBATCH --partition=gpu
#SBATCH --qos=gpu_8h
#SBATCH --gpus=rtx3090:8
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.out
#SBATCH --error=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.err

unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS
cd /idiap/home/adubois/project_ma1/TIGER
mkdir -p logs

PYTHON="/idiap/temp/adubois/miniconda3/envs/tiger/bin/python"
CONF="configs/EchoSet_cleaned/training_per_feature/alpha_ratio/deciles/decile7_alpharatio.yml"

echo "=========================================================="
echo "Starting Training Decile 7 AlphaRatio ([0.60 - 0.70]) on $(hostname)"
echo "Config: $CONF"
echo "=========================================================="

$PYTHON audio_train.py --conf_dir "$CONF"
