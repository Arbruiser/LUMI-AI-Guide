#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p dev-g
#SBATCH --time 00:30:00
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node 2
#SBATCH --nodes 1
#SBATCH --mem 120G


# --- 1. Environment Setup ---
# We use the PyTorch container provided by the LUMI AI Factory Services, which contains vLLM.
export CONTAINER_IMAGE=/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260415_130625/lumi-multitorch-full-u24r70f21m50t210-20260415_130625.sif
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Where to store the huge models. Point this to your project's scratch directory.
export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/hf-cache/

# Torch compilation currently fails in the container, so we disable it here.
export TORCH_COMPILE_DISABLE=1

# Make sure vLLM only only sees the specific GPUs Slurm has allocacted for us
export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES

# --- 2. Model & Socket Configuration ---
MODEL_NAME="Qwen/Qwen3.6-35B-A3B"


# --- 3. Run offline benchmark ---
srun singularity exec \
    --bind $TMPDIR \
    $CONTAINER_IMAGE \
    vllm bench throughput \
    --model $MODEL_NAME \
    --tensor-parallel-size $SLURM_GPUS_ON_NODE \
    --dataset-name sharegpt \
    --num-prompts 2000 \
    --load-format runai_streamer
