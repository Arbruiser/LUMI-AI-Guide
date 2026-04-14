# LLM inference
This chapter describes how to perform Large Language Model (LLM) inference on LUMI using vLLM. [vLLM](https://docs.vllm.ai/en/latest/) is a popular and memory-efficient inference engine for hosting LLMs.  

In this chapter, we will submit a batch job that starts a vLLM server with Qwen3-Code-Next, and we will run three Python scripts for interacting with and using the model.

This chapter uses `lumi-multitorch` container which includes vLLM that is optimised for running on LUMI. Note, the vLLM version may not be the absolute latest release as it takes time for our team to optimise and test the container.

## Why vLLM?
vLLM is the recommended and most popular LLM engine choice primarily due to two innovations:
- **Paged Attention**: Efficiently manages KV cache memory, allowing for much larger batch sizes, which leads to higher throughput and longer context windows.
- **Continuous Batching**: Reduces latency by processing new requests as soon as old ones finish, rather than waiting for an entire batch to complete.

### Understanding throughput
**Throughput** is the rate at which the model can process and generate tokens. Performance is split into two distinct stages, each bound by different hardware limits: 
- **Prefill (Compute bound)**: The model processes the entire input prompt (and history) in parallel. Since the GPU handles all input tokens at once, the bottleneck is the hardware's raw mathematical throughput (FLOPs). Prefill throughput is how many **input** tokens the model can be **processing**.
- **Decode (Memory bandwidth bound)**: Tokens are generated one by one. For every single token produced, the GPU must reload all the model's weights from VRAM to GPU's compute cores. This makes performance dependent on Memory Bandwidth - how fast data can move - rather than how fast the GPU can calculate. Decode throughput is how many **output** tokens the model can be **generating**.

## Inference workflows:
There are two ways to interact with the models:
| Workflow                  | Description                                                          | VRAM & Loading Behavior                                                                                                          | Best for...                                                                 |
|---------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Server-Client Mode**    | Deploys vLLM as an OpenAI-compatible API server using a Unix socket. | Load Once, Run Many: Weights are loaded into VRAM when the server starts and stay there until the Slurm job ends.                | Interactive testing, troubleshooting prompts, or building a chat interface. |
| **Offline (Python) Mode** | Uses the LLM class directly within a Python script to process data.  | Load, Process, Exit: Weights are loaded into VRAM, all prompts are processed in a single high-speed burst, then VRAM is cleared. | High-throughput batch jobs, benchmarking, and processing static files.      |


> [!NOTE] 
> It is important to distinguish between two different types of data movement:
> - Initialization (Disk → VRAM): Model weights move from the parallel file system into the GPU memory. This happens once at startup and takes several minutes. The weights must fit entirely in VRAM (assuming no offloading) and they stay there as long as the model is running.
> - Inference (VRAM → GPU Cores): During the Decode stage, the GPU must reload the model weights from VRAM into the compute cores for every single token generated.

We provide three example Python scripts for running LLM inference on LUMI using the `lumi-multitorch` container.

## Start the vLLM server
The `run-vllm-lumi4.sh` script asks Slurm for resources and handles the environment setup and launches the model. Run:

``` bash
# Make sure you have edited the project ID in the script first!
sbatch run-vllm-lumi4.sh
```

### Configuration highlights
- **Storage Redirection:** LLM weights can exceed hundreds of gigabytes, far surpassing the 20GB limit of the default `home` directory. To handle this, the script sets the `HF_HOME` environment variable to your project’s `/scratch/` directory;
- **Private Communication:** Instead of hosting the server on a standard network port, the script creates a **Unix Domain Socket** (.sock file). There are two benefits of this approach:
    - **No Port Collisions:** It avoids the common "Address already in use" error that occurs if another user is using the same port on a shared node.
    - **Enhanced Security:** The socket acts as a private gateway, removing the need for an API key. Access is restricted to your user session on that specific node, preventing other LUMI users from using your model instance.

### The execution command
The core of the script is the `srun` command, which launches the `lumi-multitorch` container and initializes the server:
``` bash
srun singularity exec \
    --bind $TMPDIR \
    --env HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES \
    $CONTAINER_IMAGE \
    vllm serve $MODEL_NAME \
    --tensor-parallel-size $SLURM_GPUS_ON_NODE \
    --uds $SOCKET_FILE \
    --load-format runai_streamer
``` 
**Flags explained:**
- `vllm serve $MODEL_NAME` is the heart of the command that starts our vLLM server;
- `--tensor-parallel-size` tells vLLM how many GPUs to split the model across. We set this to $SLURM_GPUS_ON_NODE so it automatically matches our #SBATCH request.
- `--uds $SOCKET_FILE`: This enables the Unix Domain Socket we discussed earlier.
- `--load-format runai_streamer`: This is a specialised loader that speeds up the transfer of supported model weights from the parallel file system to the GPUs. It helps significantly reduce the loading times for supported models.

## Use the model
Interacting with a running vLLM server requires you to be on the same compute node where the server (and its socket file) exists. We do this by 'jumping into' the comput node's shell, a technique called **overlapping**.

### 1. Interactive Chat (Server-Client Mode)
Start a vLLM server and start a chat (with history) with the LLM. 

1.  **Start the vLLM server.** (Make sure to update your billing project first)
    ```bash
    sbatch run-vllm-lumi4.sh
    ```
2.  **Connect to the compute node's shell.** Find your job ID with `squeue --me`, then overlap into the allocated node as soon as the job is running:
    ```bash
    srun --overlap --jobid <slurm-job-id> --pty bash
    ```
3. **Wait a few minutes for the model to load.** Monitor progress with `tail -f slurm-<job-id>.out`.
    The model has been loaded when you see a line similar to:
   ```bash
   (APIServer pid=8379) INFO: Application startup complete.
   ```

5.  **Launch the chat script.**
    ```bash
    singularity run -B /pfs,/scratch,/projappl /appl/local/laifs/containers/lumi-multitorch-latest.sif \
    python chat_with_LLM.py "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    ```
    Type 'exit' to stop.

---

### 2. Batched Inference with the server
Send a large volume of prompts from `prompts.txt` to the vLLM server, 256 at a time. 

1.  **Start the server and connect to the node:** (follow steps 1-3 from the Chat mode above).
2.  **Run the batch script:**
    ```bash
    singularity run -B /pfs,/scratch,/projappl /appl/local/laifs/containers/lumi-multitorch-latest.sif \
    python offline_batched_inference_from_server.py "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    ```
    *The results will be saved to `results.json`.*

---

### 3. **Python** Batch Inference
Get resources with _salloc_ and run batched inference directly in Python. Use this method for high-throughput and simplicity. 

1.  **Request an interactive GPU allocation:**
    ```bash
    salloc -p dev-g --nodes=1 --gpus-per-node=8 --ntasks-per-node=1 --cpus-per-task=56 --time=2:00:00 --account=project_XXXXXXXXX
    ```
2.  **Enter the compute node:**
    ```bash
    srun --overlap --jobid <slurm-job-id> --pty bash
    ```
3.  **Set required environment variables:** (enter your project ID)
    ```bash
    export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
    export TORCH_COMPILE_DISABLE=1
    export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/$USER/hf-cache
    ```

4.  **Run the script**:
    ```bash
    singularity run -B /pfs,/scratch,/projappl /appl/local/laifs/containers/lumi-multitorch-latest.sif \
    python offline_batched_inference_from_Python.py "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    ```


--- 
### Explaining how context works
### Throughput with sequential requests vs batches
### Pictures? 
### Some KV cache and VRAM explanations? 
### Benchmarking? Online and offline?