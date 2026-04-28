# LLM inference
This chapter describes how to perform Large Language Model (LLM) inference on LUMI using vLLM. [vLLM](https://docs.vllm.ai/en/latest/) is a popular and memory-efficient inference engine for hosting LLMs.  

In this chapter, we will submit a batch job that starts a vLLM server with deepseek-ai/DeepSeek-V3.2, and we will run three Python scripts for interacting with and using the model.

This chapter uses `lumi-multitorch` container which includes vLLM that is optimised for running on LUMI. Note, the vLLM version may not be the absolute latest release as it takes time for our team to optimise and test the container.

## Why vLLM?
vLLM is the recommended and most popular LLM engine choice primarily due to two innovations:
- **Paged Attention**: Efficiently manages KV cache memory, allowing for much larger batch sizes, higher throughput and longer context windows.
- **Continuous Batching**: Reduces latency by processing new requests as soon as old ones finish, rather than waiting for an entire batch to complete.

## Inference workflows:
There are two ways to interact with the models:
| Workflow                  | Description                                                          | VRAM & Loading Behavior                                                                                                          | Best for...                                                                 |
|---------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Server-Client Mode**    | Deploys vLLM as an OpenAI-compatible API server using a Unix socket. | Load Once, Run Many: Weights are loaded into VRAM when the server starts and stay there until the Slurm job ends.                | Interactive testing, checking the model's 'vibe', troubleshooting, or building a chat interface. |
| **Offline (Python) Mode** | Uses the LLM class directly within a Python script to process data.  | Load, Process, Exit: Weights are loaded into VRAM, all prompts are processed in a single high-speed burst, then VRAM is cleared. | High-throughput batch jobs, benchmarking, and processing static files.      |

## The example scripts
In this chapter, we use three distinct Python scripts to demonstrate different ways of interacting with the model:
- `chat_with_LLM.py`: An interactive script that enables back-and-forth dialogue with the model, including chat history.
- `batched_inference_from_server.py`: send hundreds of prompts simultaneously to a running vLLM server for fast dataset processing or benchmarking.
- `batched_inference_from_Python.py`: start vLLM directly in Python to load the model for fast dataset processing or benchmarking.

## Workflow A: Server-Client Mode
Use this if you want to keep the model loaded and interact with it multiple times.

### Step 1: Start the vLLM server
The `run-vllm-lumi4.sh` script asks Slurm for resources, handles the environment setup and launches the model. Update your project ID and submit:

``` bash
sbatch run-vllm-lumi4.sh
```
> [!WARNING]
> **Lukas's feedback**: "it'd be good to briefly go through the things that the run-vllm-lumi4.sh does (loading modules, set required env vars (and why they are needed), the model and number of nodes it sets up). Not in great depth, just so that the user doesn't need to go check out the file immediately at that point. So basically expanding what you have in the configuration highlights a bit"

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
> [!WARNING]
> **Note to self**: I either need to provide more detailed explanations like this about the Python scripts, or I need to remove these detailed explanations

## Step 2: Interact with the server
Interacting with a running vLLM server requires you to be on the same compute node where the server (and its socket file) exists. We do this by 'jumping into' the compute node's shell, which is called **overlapping**.

1.  **Enter the compute node's shell:** 
    Find your job ID with `squeue --me`. As soon as your job status is `R` (Running), overlap into the allocated node as soon as the job is running:
    ```bash
    srun --overlap --jobid <slurm-job-id> --pty bash
    ```

2. **Monitor the startup:** 
    Loading models into VRAM takes time. Check the logs and wait for the "Application startup complete" message:
    ```bash
    `tail -f slurm-<job-id>.out`.
    ```

3.  **Launch a client script.**
    Now you can run either the interactive chat or the batched-API script:
    - **Option 1: Interactive chat**. Best for having a back-and-forth conversation, quickly checking the model's "vibe", and output format.
    ```bash
    singularity run -B /pfs,/scratch,/projappl /appl/local/laifs/containers/lumi-multitorch-latest.sif \
    python chat_with_LLM.py "deepseek-ai/DeepSeek-V3.2"
    ```
> [!TIP]
> Type 'exit' to stop.

    - **Option 2: Batched API Inference.** Best for sending a lot of prompts, receiving LLM responses, and tweaking the model to run the prompts again.
    ```bash
    singularity run -B /pfs,/scratch,/projappl /appl/local/laifs/containers/lumi-multitorch-latest.sif \
    python batched_inference_from_server.py "deepseek-ai/DeepSeek-V3.2"
    ```
    *The results will be saved to `results.json`.*

---

## Workflow B: Offline Python Mode
Get resources with _salloc_ and run batched inference directly in Python. Use this method for high-throughput batch processing where you don't need a persistent server. This approach is "one-and-done": it requests resources, processes your list, and exits.

1.  **Request an interactive GPU allocation:**
    ```bash
    salloc -p dev-g --nodes=1 --gpus-per-node=8 --ntasks-per-node=1 --cpus-per-task=56 --time=2:00:00 --account=project_XXXXXXXXX
    ```
2.  **Enter the compute node:**
    ```bash
    srun --overlap --jobid <slurm-job-id> --pty bash
    ```
3.  **Set required environment variables:**
    ```bash
    export HIP_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES
    export TORCH_COMPILE_DISABLE=1
    export HF_HOME=/scratch/$SLURM_JOB_ACCOUNT/$USER/hf-cache
    ```
4.  **Run the script**:
    ```bash
    singularity run -B /pfs,/scratch,/projappl /appl/local/laifs/containers/lumi-multitorch-latest.sif \
    python batched_inference_from_Python.py "deepseek-ai/DeepSeek-V3.2"
    ```


---

## Deep dive: how things work
### 1. Data movement on the GPU
It is important to distinguish between two different types of data movement:
- Initialization (Disk → VRAM): Model weights move from the parallel file system into the GPU memory. This happens once at startup and takes several minutes. The weights must fit entirely in VRAM (assuming no offloading) and they stay there as long as the model is running.
- Inference (VRAM → GPU Cores): During the Decode stage, the GPU must reload the model weights from VRAM into the compute cores for every single token generated.

### 2. How "Memory" (Context) works
If you look at `chat_with_LLM.py`, you will notice a list called `messages`. It is a common misconception that LLMs "remember" conversations naturally. In reality, LLMs are stateless, they treat every request as a brand new interaction, and it's the client's job to provide the 'memory'.
    **The cost of context**: As the conversation grows longer, the "Prefill" stage takes longer and consumes more VRAM (to store the KV Cache). While Paged Attention makes this memory usage much more efficient, it doesn't make it free.
    **Client-side management**: In our example, the memory is managed by the Python script. If you stop the script and restart it, the "memory" is cleared, even if the vLLM server is still running.

### 3. Understanding throughput
**Throughput** is the rate at which the model can process and generate tokens. Performance is split into two distinct stages, each bound by different hardware limits: 
- **Prefill (Compute bound)**: The model processes the entire input prompt (and history) in parallel. Since the GPU handles all input tokens at once, the bottleneck is the hardware's raw mathematical throughput (FLOPs). Prefill throughput is how many **input** tokens the model can be **processing**.
- **Decode (Memory bandwidth bound)**: Tokens are generated one by one. For every single token produced, the GPU must reload all the model's weights from VRAM to GPU's compute cores. This makes performance dependent on Memory Bandwidth - how fast data can move - rather than how fast the GPU can calculate. Decode throughput is how many **output** tokens the model can be **generating**.

### 4. Throughput: Sequential vs. Batched
Why do we use asyncio and semaphores in `batched_inference_with_server.py`?
- Sequential (Slow): If you send one prompt, wait for the answer, and then send the next, you don't utilise the full batching power of vLLM and your GPU sits almost idle.
- Batched (Fast): By sending 256 prompts at once, vLLM’s **Continuous Batching** kicks in. While one request is in the "Decode" stage (generating a token), another can be in the "Prefill" stage. This saturates the GPU's memory bandwidth and drastically increases the number of tokens generated per second.

> [!TIP]
> If you want to see this in action, try running Script 2 with a semaphore of 1 (sequential) vs 256 (batched). This will only send one prompt at a time to the model, wait for the model's complete response, and only then send the next prompt. 

### Other content ideas:
- Pictures with multimodal LLMs? 
- Some KV cache explanations? 
- vLLMs built-in benchmarking with explanation of what results mean?
