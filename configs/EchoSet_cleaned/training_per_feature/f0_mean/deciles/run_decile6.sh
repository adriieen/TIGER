#!/bin/bash
#SBATCH --job-name=tiger-d6-f0
#SBATCH --account=pause
#SBATCH --partition=gpu
#SBATCH --qos=gpu_8h
#SBATCH --gpus=rtx3090:6
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.out
#SBATCH --error=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.err

# Nettoyer d'éventuelles variables d'environnement Slurm perturbatrices pour DDP
unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS

cd /idiap/home/adubois/project_ma1/TIGER
mkdir -p logs

PYTHON="/idiap/temp/adubois/miniconda3/envs/tiger/bin/python"
CONF="/idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned/training_per_feature/f0_mean/deciles/decile6_f0mean.yml"

echo "=========================================================="
echo "Starting Training Decile 6 (Tranche [0.50 - 0.60]) on $(hostname)"
echo "Start time: $(date)"
echo "Config: $CONF"
echo "=========================================================="

$PYTHON audio_train.py --conf_dir "$CONF"

echo "Finished Decile 6 at $(date) with exit code $?"
