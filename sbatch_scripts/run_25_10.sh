#!/bin/bash
#SBATCH --job-name=tiger-top10
#SBATCH --account=pause
#SBATCH --partition=gpu
#SBATCH --qos=gpu_8h
#SBATCH --gpus=rtx3090:6
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.out
#SBATCH --error=/idiap/home/adubois/project_ma1/TIGER/logs/%x_%j.err

# Nettoyer d'éventuelles variables d'environnement Slurm qui perturbent PyTorch DDP
unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS

# Se placer dans le répertoire du projet
cd /idiap/home/adubois/project_ma1/TIGER

# Créer le dossier de logs si nécessaire
mkdir -p logs

# Exécution du training
echo "Starting training on $(hostname) at $(date)..."
/idiap/temp/adubois/miniconda3/envs/tiger/bin/python audio_train.py \
    --conf_dir /idiap/home/adubois/project_ma1/TIGER/configs/EchoSet_cleaned/training_per_feature/top10_f0mean.yml
echo "Training finished at $(date)"
