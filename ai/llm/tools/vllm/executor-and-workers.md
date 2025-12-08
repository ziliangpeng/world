# vLLM Model Executor & Workers Deep Dive

## Overview

The **Model Executor** is responsible for distributing scheduled batches to GPU workers and collecting their outputs. It sits between the Scheduler (which decides what to run) and the Workers (which actually execute on GPUs).

**Core Flow**:
```
Scheduler.schedule()
    → SchedulerOutput
    → ModelExecutor.execute_model()
    → Workers.execute_model()
    → GPUModelRunner.execute_model()
    → Actual GPU forward pass
    → ModelRunnerOutput
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ EngineCore (Main Process)                                  │
│                                                             │
│  ┌──────────────┐                                          │
│  │  Scheduler   │                                          │
│  └──────┬───────┘                                          │
│         │ SchedulerOutput                                  │
│         ▼                                                   │
│  ┌──────────────────────────────────┐                      │
│  │  MultiprocExecutor               │                      │
│  │  - Manages worker processes      │                      │
│  │  - Sends RPC via MessageQueue    │                      │
│  │  - Returns Future[Output]        │                      │
│  └──────────────┬───────────────────┘                      │
└─────────────────┼────────────────────────────────────────────┘
                  │ IPC (shared memory message queues)
                  ▼
┌─────────────────────────────────────────────────────────────┐
│ Worker Processes (1 per GPU)                               │
│                                                             │
│  ┌────────────────────────────────┐                        │
│  │ WorkerProc                     │                        │
│  │ - Busy loop reading commands   │                        │
│  │ - Calls Worker.execute_model() │                        │
│  └──────────┬─────────────────────┘                        │
│             ▼                                               │
│  ┌────────────────────────────────┐                        │
│  │ Worker (GPUWorker)             │                        │
│  │ - Manages device/model         │                        │
│  │ - Pipeline parallelism logic   │                        │
│  └──────────┬─────────────────────┘                        │
│             ▼                                               │
│  ┌────────────────────────────────┐                        │
│  │ GPUModelRunner                 │                        │
│  │ - Prepares inputs/metadata     │                        │
│  │ - Runs model forward pass      │                        │
│  │ - Samples tokens               │                        │
│  └──────────┬─────────────────────┘                        │
│             ▼                                               │
│  ┌────────────────────────────────┐                        │
│  │ Model (PyTorch nn.Module)      │                        │
│  │ - Actual Transformer layers    │                        │
│  │ - Attention, MLP, etc.         │                        │
│  └────────────────────────────────┘                        │
│             │                                               │
│             ▼                                               │
│         GPU Hardware                                        │
└─────────────────────────────────────────────────────────────┘
```

## Component 1: MultiprocExecutor

**Location**: `vllm/v1/executor/multiproc_executor.py`

**Responsibility**: Spawn and manage worker processes, send work to them via IPC.

### Key Features

#### 1. Worker Process Management

```python
def _init_executor(self):
    self.world_size = parallel_config.world_size
    tensor_parallel_size = parallel_config.tensor_parallel_size
    pp_parallel_size = parallel_config.pipeline_parallel_size

    # Create message queue for broadcasting scheduler outputs
    self.rpc_broadcast_mq = MessageQueue(self.world_size, self.world_size)

    # Spawn worker processes
    for rank in range(self.world_size):
        unready_workers.append(
            WorkerProc.make_worker_process(
                vllm_config=self.vllm_config,
                local_rank=rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                input_shm_handle=scheduler_output_handle,
            ))

    # Wait for all workers to be ready
    self.workers = WorkerProc.wait_for_ready(unready_workers)

    # Start monitoring thread
    self.start_worker_monitor()
```

**Important**:
- Creates **one worker process per GPU** (world_size = TP × PP × DP)
- Uses **shared memory message queues** for efficient IPC
- **Monitors worker liveness** - shuts down executor if any worker dies

#### 2. Execute Model (Non-Blocking Mode)

```python
def execute_model(self, scheduler_output) -> Union[ModelRunnerOutput, Future[ModelRunnerOutput]]:
    non_block = self.max_concurrent_batches > 1  # True for async scheduling

    if not self.has_connector:
        # Get output only from output_rank (TP rank 0, PP rank -1)
        (output, ) = self.collective_rpc(
            "execute_model",
            args=(scheduler_output, ),
            unique_reply_rank=self.output_rank,
            non_block=non_block,  # ← Returns Future immediately!
        )
        return output
```

**When `non_block=True` (async scheduling)**:
- Returns `Future[ModelRunnerOutput]` immediately
- GPU worker processes the batch asynchronously
- Main process can schedule the next batch while GPU is busy

#### 3. Collective RPC

```python
def collective_rpc(self, method, args, non_block=False, unique_reply_rank=None):
    # Broadcast command to all workers via shared memory queue
    self.rpc_broadcast_mq.enqueue(
        (method, args, kwargs, unique_reply_rank)
    )

    workers = (self.workers[unique_reply_rank],) if unique_reply_rank else self.workers
    responses = []

    for w in workers:
        if non_block:
            # Submit to thread pool - returns Future immediately!
            result = self.io_thread_pool.submit(
                get_response,  # Blocks in background thread
                w,
                dequeue_timeout
            )
        else:
            # Block immediately until worker responds
            result = get_response(w, dequeue_timeout)

        responses.append(result)

    return responses
```

**Key insight**:
- `non_block=True` uses `ThreadPoolExecutor.submit()` which returns a `Future`
- Background thread waits for worker response from message queue
- Main thread continues immediately (can schedule next batch!)

#### 4. Worker Monitoring

```python
def start_worker_monitor(self):
    def monitor_workers():
        sentinels = [h.proc.sentinel for h in workers]
        died = multiprocessing.connection.wait(sentinels)  # Blocks until a worker dies

        _self.is_failed = True
        logger.error("Worker proc %s died unexpectedly", proc_name)
        _self.shutdown()

        # Notify engine to shut down gracefully
        callback = _self.failure_callback
        if callback:
            callback()

    Thread(target=monitor_workers, daemon=True).start()
```

## Component 2: WorkerProc

**Location**: `vllm/v1/executor/multiproc_executor.py:351`

**Responsibility**: Wrapper that runs one Worker in a separate process.

### Initialization Flow

```python
def __init__(
    self,
    vllm_config,
    local_rank,
    rank,
    distributed_init_method,
    input_shm_handle,
):
    self.rank = rank

    # Initialize worker wrapper
    wrapper = WorkerWrapperBase(vllm_config=vllm_config, rpc_rank=rank)
    wrapper.init_worker(all_kwargs)
    self.worker = wrapper

    # Set process name for debugging
    pp_str = f"PP{rank // tp_size}" if pp_size > 1 else ""
    tp_str = f"TP{rank % tp_size}" if tp_size > 1 else ""
    process_name = f"VllmWorker {pp_str}_{tp_str}"
    set_process_title(process_name)

    # Initialize message queues
    self.rpc_broadcast_mq = MessageQueue.create_from_handle(input_shm_handle, rank)
    self.worker_response_mq = MessageQueue(1, 1)

    # Initialize device and load model weights
    self.worker.init_device()
    self.worker.load_model()
```

### Worker Busy Loop

```python
def worker_busy_loop(self):
    """Main busy loop for Multiprocessing Workers"""
    while True:
        # Block waiting for command from executor
        method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()

        try:
            if isinstance(method, str):
                func = getattr(self.worker, method)  # e.g., self.worker.execute_model
            elif isinstance(method, bytes):
                func = partial(cloudpickle.loads(method), self.worker)

            # Execute the method
            output = func(*args, **kwargs)
        except Exception as e:
            logger.exception("WorkerProc hit an exception.")
            if output_rank is None or self.rank == output_rank:
                self.worker_response_mq.enqueue(
                    (ResponseStatus.FAILURE, str(e)))
            continue

        # Send output back to executor
        if output_rank is None or self.rank == output_rank:
            self.worker_response_mq.enqueue(
                (ResponseStatus.SUCCESS, output))
```

**Key points**:
- Infinite loop: read command → execute → send response
- Only sends response if this worker is the `output_rank` (for efficiency)
- Runs in separate process with its own GPU context

### Process Spawning

```python
@staticmethod
def make_worker_process(vllm_config, local_rank, rank, ...):
    context = get_mp_context()

    # Create ready pipe for synchronization
    reader, writer = context.Pipe(duplex=False)

    # Create death pipe to detect parent exit
    death_reader, death_writer = context.Pipe(duplex=False)

    # Spawn process
    proc = context.Process(
        target=WorkerProc.worker_main,
        kwargs=process_kwargs,
        name=f"VllmWorker-{rank}",
        daemon=True
    )
    proc.start()

    return UnreadyWorkerProcHandle(proc, rank, reader, death_writer)
```

## Component 3: Worker (GPUWorker)

**Location**: `vllm/v1/worker/gpu_worker.py:43`

**Responsibility**: Manage device initialization, model loading, and coordinate execution.

### Key Methods

#### 1. Initialize Device

```python
def init_device(self):
    if self.vllm_config.device_config.device.type == "cuda":
        # Set CUDA device
        torch.cuda.set_device(self.device)

        # Initialize distributed environment
        init_distributed_environment(
            world_size=parallel_config.world_size,
            rank=self.rank,
            distributed_init_method=self.distributed_init_method,
        )

        # Initialize model parallel groups (TP, PP, DP)
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_model_parallel_size=parallel_config.pipeline_parallel_size,
        )
```

#### 2. Load Model

```python
def load_model(self):
    self.model_runner = GPUModelRunner(
        vllm_config=self.vllm_config,
        ...
    )
```

#### 3. Execute Model (with Pipeline Parallelism)

```python
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
) -> Optional[ModelRunnerOutput]:
    intermediate_tensors = None

    # Pipeline Parallelism: receive from previous stage
    if not get_pp_group().is_first_rank:
        intermediate_tensors = IntermediateTensors(
            get_pp_group().recv_tensor_dict(
                all_gather_group=get_tp_group()
            )
        )

    # Run model forward pass
    output = self.model_runner.execute_model(
        scheduler_output,
        intermediate_tensors
    )

    # Pipeline Parallelism: send to next stage
    if not get_pp_group().is_last_rank:
        assert isinstance(output, IntermediateTensors)
        get_pp_group().send_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group()
        )
        return None  # Mid-stage workers don't return output

    # Only last PP stage returns ModelRunnerOutput
    return output
```

**Pipeline Parallelism Flow**:
```
PP Stage 0 (Layers 0-10):
  ├─ Run forward pass on layers 0-10
  └─ Send activations to PP Stage 1

PP Stage 1 (Layers 11-20):
  ├─ Receive activations from PP Stage 0
  ├─ Run forward pass on layers 11-20
  └─ Send activations to PP Stage 2

PP Stage 2 (Layers 21-31):
  ├─ Receive activations from PP Stage 1
  ├─ Run forward pass on layers 21-31
  ├─ Compute logits
  └─ Return ModelRunnerOutput
```

## Component 4: GPUModelRunner

**Location**: `vllm/v1/worker/gpu_model_runner.py`

**Responsibility**: Prepare inputs, run forward pass, sample tokens.

### Execute Model Flow

```python
@torch.inference_mode()
def execute_model(
    self,
    scheduler_output: SchedulerOutput,
    intermediate_tensors: Optional[IntermediateTensors] = None,
) -> Union[ModelRunnerOutput, IntermediateTensors]:

    # 1. Update internal state
    self._update_states(scheduler_output)

    # 2. Early exit if no work
    if not scheduler_output.total_num_scheduled_tokens:
        return EMPTY_MODEL_RUNNER_OUTPUT

    # 3. Prepare inputs (attention metadata, logits indices, etc.)
    (attn_metadata, logits_indices, spec_decode_metadata,
     num_scheduled_tokens_np, ...) = self._prepare_inputs(scheduler_output)

    # 4. Determine batch size (with CUDA graph padding if applicable)
    num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
    if cudagraph_mode and num_scheduled_tokens <= max_cudagraph_size:
        # Pad to CUDA graph batch size
        num_input_tokens = self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)
    else:
        # Eager mode - no padding
        num_input_tokens = num_scheduled_tokens

    # 5. Run multimodal encoder if needed
    if self.supports_mm_inputs:
        self._execute_mm_encoder(scheduler_output)
        mm_embeds = self._gather_mm_embeddings(scheduler_output)

        # Get input embeddings (text + vision)
        inputs_embeds = self.model.get_input_embeddings(
            input_ids=self.input_ids[:num_scheduled_tokens],
            multimodal_embeddings=mm_embeds,
        )
        input_ids = None
    else:
        # Text-only models use token IDs
        input_ids = self.input_ids[:num_input_tokens]
        inputs_embeds = None

    positions = self.positions[:num_input_tokens]

    # 6. Run the model forward pass
    with set_forward_context(attn_metadata, ...):
        model_output = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    # 7. For mid-pipeline stages, return hidden states
    if not get_pp_group().is_last_rank:
        assert isinstance(model_output, IntermediateTensors)
        return model_output  # Send to next PP stage

    # 8. For last stage: compute logits and sample
    hidden_states = model_output

    # For pooling tasks (embeddings), return pooled output
    if self.input_batch.pooling_params:
        return self._pool(hidden_states, ...)

    # For generation tasks, sample tokens
    sample_hidden_states = hidden_states[logits_indices]
    logits = self.model.compute_logits(sample_hidden_states, None)

    # Apply grammar constraints if needed
    if grammar_bitmask is not None:
        self._apply_grammar_bitmask(logits, grammar_bitmask)

    # Sample tokens
    sampled_token_ids = self._sample(logits, sampling_metadata)

    # 9. Return ModelRunnerOutput
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
        sampled_token_ids=sampled_token_ids,
        logprobs=logprobs,
        pooler_output=pooler_output,
    )
```

### Key Responsibilities

#### 1. State Management

```python
def _update_states(self, scheduler_output):
    # Update cached request states
    for req_data in scheduler_output.scheduled_new_reqs:
        self.input_batch.add_request(req_data)

    # Remove finished requests
    for req_id in scheduler_output.finished_req_ids:
        self.input_batch.remove_request(req_id)
```

#### 2. Attention Metadata Preparation

Prepares metadata for the attention kernels (PagedAttention, FlashAttention, etc.):

```python
def _prepare_inputs(self, scheduler_output):
    # Build attention metadata
    attn_metadata = AttentionMetadataBuilder.build(
        num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
        num_reqs=len(scheduler_output.num_scheduled_tokens),
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        max_query_len=max_query_len,
        # ... more metadata
    )

    return (attn_metadata, logits_indices, ...)
```

#### 3. CUDA Graph Support

For maximum performance on small decode batches:

```python
if cudagraph_mode and num_scheduled_tokens <= max_cudagraph_size:
    # Pad batch to fixed size for CUDA graph
    num_input_tokens = pad_for_cudagraph(num_scheduled_tokens)

    # Use pre-captured CUDA graph
    cudagraph_runtime_mode = CudagraphDispatcher.dispatch(batch_descriptor)
```

**CUDA Graph Benefits**:
- Eliminates kernel launch overhead
- ~2-3x speedup for small decode batches (batch size < 32)
- Requires fixed batch sizes (hence padding)

#### 4. Sampling

```python
def _sample(self, logits, sampling_metadata):
    # Apply temperature, top-p, top-k, etc.
    logits = self.logits_processors.apply(logits, sampling_metadata)

    # Sample tokens
    sampled_token_ids = self.sampler.sample(logits, sampling_metadata)

    return sampled_token_ids
```

## Multi-GPU Communication Patterns

### 1. Tensor Parallelism (TP)

**Purpose**: Split model weights across GPUs horizontally (within a layer).

**Example**: Llama-70B on 4 GPUs with TP=4

```
Layer: Linear(8192, 32768)

GPU 0: Linear(8192, 8192)   ← 1/4 of output dimension
GPU 1: Linear(8192, 8192)   ← 1/4 of output dimension
GPU 2: Linear(8192, 8192)   ← 1/4 of output dimension
GPU 3: Linear(8192, 8192)   ← 1/4 of output dimension

All-Gather to combine outputs
```

**Code**:
```python
# In model forward pass
output = linear(x)  # Each GPU computes partial output
output = tensor_model_parallel_all_gather(output)  # Combine across TP group
```

**Communication**:
- **All-Reduce**: Sum outputs across TP group (for MLP layers)
- **All-Gather**: Concatenate outputs across TP group (for attention)

### 2. Pipeline Parallelism (PP)

**Purpose**: Split model layers across GPUs vertically.

**Example**: Llama-70B on 4 GPUs with PP=4

```
GPU 0: Layers 0-19
GPU 1: Layers 20-39
GPU 2: Layers 40-59
GPU 3: Layers 60-79 + LM Head
```

**Code** (already shown above in Worker.execute_model):
```python
# Receive from previous stage
if not is_first_rank:
    intermediate_tensors = recv_tensor_dict()

# Run forward pass
output = model(input_ids, intermediate_tensors)

# Send to next stage
if not is_last_rank:
    send_tensor_dict(output)
```

**Communication**:
- **Point-to-point send/recv**: Between adjacent PP stages
- **No all-to-all communication**: Each stage only talks to neighbors

### 3. Data Parallelism (DP)

**Purpose**: Replicate model across GPUs, split data batches.

vLLM v1 supports DP for higher throughput (not commonly used).

**Example**: 8 GPUs with DP=2, TP=4

```
DP Replica 0: TP Group (GPU 0, 1, 2, 3) → Processes batch[0:128]
DP Replica 1: TP Group (GPU 4, 5, 6, 7) → Processes batch[128:256]
```

## Message Queue IPC

**Why shared memory queues?**
- **Fast**: ~100x faster than `multiprocessing.Queue` (avoids pickle overhead)
- **Large messages**: Can send multi-GB SchedulerOutput without copying
- **Low latency**: ~100μs for small messages

**Implementation**: `vllm/distributed/device_communicators/shm_broadcast.py`

```python
class MessageQueue:
    def __init__(self, num_readers, num_writers, max_chunk_bytes=10MB):
        # Allocate shared memory buffer
        self.shm = shared_memory.SharedMemory(create=True, size=buffer_size)
        self.buffer = np.ndarray(shape, dtype, buffer=self.shm.buf)

        # Synchronization primitives
        self.reader_semaphore = multiprocessing.Semaphore(0)
        self.writer_semaphore = multiprocessing.Semaphore(max_chunk_bytes)

    def enqueue(self, obj):
        # Serialize with pickle
        data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

        # Write to shared memory
        self.writer_semaphore.acquire()  # Wait for space
        self.buffer[write_idx:write_idx+len(data)] = data
        self.reader_semaphore.release()  # Signal readers

    def dequeue(self):
        # Wait for data
        self.reader_semaphore.acquire()

        # Read from shared memory
        data = bytes(self.buffer[read_idx:read_idx+msg_len])
        obj = pickle.loads(data)

        self.writer_semaphore.release()  # Signal writer
        return obj
```

## Execution Timeline (Async Scheduling)

```
Main Process:
│
├─ [Call 1] step_with_batch_queue()
│   ├─ scheduler.schedule() → B1                      [2ms]
│   ├─ executor.execute_model(B1)                     [0.1ms]
│   │   ├─ rpc_broadcast_mq.enqueue("execute_model", B1)
│   │   └─ Return Future[B1]                          ← Returns immediately!
│   └─ batch_queue.put(Future[B1])
│   └─ return (None, True)
│
├─ [Call 2] step_with_batch_queue()
│   ├─ scheduler.schedule() → B2                      [2ms]
│   ├─ executor.execute_model(B2)                     [0.1ms]
│   │   ├─ rpc_broadcast_mq.enqueue("execute_model", B2)
│   │   └─ Return Future[B2]
│   └─ return (None, True)
│
├─ [Call 3] step_with_batch_queue()
│   ├─ batch_queue.full() → skip scheduling
│   ├─ batch_queue.get() → Future[B1]
│   ├─ future.result() → BLOCKS                       [50ms] ← Wait for GPU
│   └─ return (outputs_B1, False)

Worker Process 0:
│
├─ worker_busy_loop()
│   ├─ rpc_broadcast_mq.dequeue() → "execute_model", B1
│   ├─ worker.execute_model(B1)
│   │   ├─ model_runner.execute_model(B1)
│   │   │   ├─ Prepare inputs                          [1ms]
│   │   │   ├─ model.forward()                         [45ms] ← GPU busy!
│   │   │   └─ sample tokens                           [2ms]
│   │   └─ return ModelRunnerOutput
│   ├─ worker_response_mq.enqueue(SUCCESS, output)
│   │
│   ├─ rpc_broadcast_mq.dequeue() → "execute_model", B2
│   ├─ worker.execute_model(B2)                        [48ms]
│   └─ worker_response_mq.enqueue(SUCCESS, output)
```

**Overlapping**:
- While Worker runs B1 forward pass (45ms), Main Process schedules B2 (2ms)
- B2 is queued and ready when B1 finishes
- GPU idle time reduced from ~4ms to ~0.5ms!

## Summary

**Key Takeaways**:

1. **MultiprocExecutor**: Manages worker processes, non-blocking RPC via shared memory queues
2. **WorkerProc**: Wrapper running in separate process, busy loop reading commands
3. **Worker (GPUWorker)**: Device management, pipeline parallelism coordination
4. **GPUModelRunner**: Input preparation, forward pass, sampling
5. **Async Execution**: `Future` + thread pool + message queues = overlapped scheduling
6. **Multi-GPU**: TP (within layer), PP (across layers), DP (data replication)
7. **IPC**: Shared memory message queues for low-latency communication

**Performance Optimizations**:
- **CUDA Graphs**: 2-3x speedup for small batches
- **Async Scheduling**: <1% GPU idle time (vs 2-5% synchronous)
- **Shared Memory Queues**: 100x faster than pickle-based queues
- **Pipeline Parallelism**: Overlap compute across stages
