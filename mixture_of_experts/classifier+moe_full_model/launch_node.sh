#!/bin/bash
# launch_node.sh
# Manual Multi-Node DDP launcher (6 machines x 1 GPU RTX 4000 Ada 20GB)
#
# Usage on each machine (0 = master, 1..5 = workers):
#   MASTER_ADDR=<MASTER_IP> NODE_RANK=<0..5> ./launch_node.sh
# Optional overrides:
#   MASTER_PORT=29500 NNODES=6 NCCL_SOCKET_IFNAME=eth0 RUN_NAME=moe_tiny_6nodes ./launch_node.sh

set -e

: "${MASTER_ADDR:?Please set MASTER_ADDR=<IP_OF_NODE_0>}"
: "${NODE_RANK:?Please set NODE_RANK=<0..5>}"

MASTER_PORT="${MASTER_PORT:-29500}"
NNODES="${NNODES:-6}"
RUN_NAME="${RUN_NAME:-moe_tiny_6nodes}"

# Clean any residual Slurm environment variables
unset SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS

# Network & NCCL stability flags for asynchronous multi-node launch
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4
if [ -n "${NCCL_SOCKET_IFNAME}" ]; then
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}"
fi
# Uncomment to debug network handshake:
# export NCCL_DEBUG=INFO

WORK_DIR="/idiap/home/adubois/project_ma1/TIGER/mixture_of_experts/classifier+moe_full_model"
cd "${WORK_DIR}"

TORCHRUN="/idiap/temp/adubois/miniconda3/envs/tiger/bin/torchrun"
CKPT_DIR="${WORK_DIR}/checkpoints/${RUN_NAME}"

echo "=========================================================="
echo "Launching JointMoETIGER DDP on $(hostname)"
echo "NODE_RANK=${NODE_RANK}/${NNODES} | MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "RUN_NAME=${RUN_NAME}"
echo "=========================================================="

$TORCHRUN \
    --nnodes="${NNODES}" \
    --nproc_per_node=1 \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    train_ddp_multinode.py \
    --save_dir "${CKPT_DIR}" \
    --epochs 200 \
    --batch_size 2 \
    --segment 3.0 \
    --lr 1e-3 \
    --lambda_cls 2.0 \
    --temperature 1.0 \
    --num_workers 4 \
    --wandb_name "${RUN_NAME}"
