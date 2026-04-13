#!/bin/bash
#SBATCH -A project_462000131
#SBATCH -p dev-g
#SBATCH --time 2:00:00
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node 4
#SBATCH --nodes 1
#SBATCH --mem 240G


# --- 1. Environment Setup ---
# We use the PyTorch container provided by the LUMI AI Factory Services, which contains vLLM.
export CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-latest.sif
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Where to store the huge models. Point this to your project's scratch directory.
export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/hf-cache/

# Torch compilation currently fails in the container, so we disable it here.
export TORCH_COMPILE_DISABLE=1


# --- 2. Model & Socket Configuration ---
MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
SOCKET_FILE=$TMPDIR/vllm-$SLURM_JOB_ID.sock


# --- 3. Execution ---
singularity exec \
    --bind $TMPDIR \
    --env HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
    $CONTAINER_IMAGE \
    vllm bench throughput \
    --model $MODEL_NAME \
    --dataset-name sharegpt \
    --tensor-parallel-size 4 \
    --num-prompts 300
