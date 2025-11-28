# Batching Strategies for LLM Inference

Batching multiple requests together dramatically improves GPU utilization and throughput. The challenge for LLMs is that requests have variable lengths and arrive asynchronously, making traditional static batching inefficient. This document covers the evolution from static to continuous batching and the techniques that enable high-throughput LLM serving.

---

## Why Batching Matters

### GPU Utilization Without Batching

Single request inference wastes most GPU compute:

```
Single request (batch=1):
- Model weights loaded: 140 GB (70B model)
- Compute per token: ~2N FLOPs = 140 GFLOPs
- A100 peak: 312 TFLOPs (fp16)
- Utilization: 140 GFLOPs / 312 TFLOPs = 0.04%
```

The GPU spends most time waiting for memory transfers, not computing.

### Batching Improves Efficiency

```
Batch=32:
- Model weights loaded once: 140 GB
- Compute per token: 32 × 140 GFLOPs = 4.5 TFLOPs
- Utilization: 4.5 TFLOPs / 312 TFLOPs = 1.4%
- Speedup: 32× more tokens at ~same latency

Batch=256:
- Compute per token: 256 × 140 GFLOPs = 36 TFLOPs
- Utilization: ~12%
- Eventually compute-bound, not memory-bound
```

**Key insight**: Larger batches amortize the cost of loading model weights.

---

## Historical Evolution

### Phase 1: Static Batching (Pre-2022)

**Simple approach**: Collect N requests, process together, return all.

```python
def static_batching(requests, batch_size=8):
    batches = [requests[i:i+batch_size] for i in range(0, len(requests), batch_size)]

    for batch in batches:
        # Pad all sequences to same length
        max_len = max(len(r.prompt) for r in batch)
        padded = [pad(r.prompt, max_len) for r in batch]

        # Generate until longest sequence done
        while not all_done(batch):
            next_tokens = model(padded)
            # Update all sequences
```

**Problems**:
1. **Head-of-line blocking**: Short sequences wait for long ones
2. **Padding waste**: Compute wasted on padding tokens
3. **Fixed batch size**: Can't adapt to load

### Phase 2: Dynamic Batching (2022)

**Improvement**: Batch requests by similar length.

```python
def dynamic_batching(queue, max_wait_time=50ms):
    """Batch requests with similar lengths."""
    batches = defaultdict(list)

    while True:
        request = queue.get(timeout=max_wait_time)
        if request:
            # Bucket by length
            bucket = round_up(len(request.prompt), 128)
            batches[bucket].append(request)

        # Process full buckets
        for bucket, batch in batches.items():
            if len(batch) >= batch_size or timeout_reached:
                process_batch(batch)
                batches[bucket] = []
```

**Better, but still suffers from**:
- Sequences finishing at different times
- Waiting for batch to fill

### Phase 3: Continuous Batching (2022-2023)

**[ORCA](https://www.usenix.org/conference/osdi22/presentation/yu)** (OSDI 2022)

Revolutionary approach: Batch at iteration level, not request level.

```
Static batching:
Request 1: ████████████████████████████████ (32 tokens)
Request 2: ████████████░░░░░░░░░░░░░░░░░░░░ (12 tokens + 20 padding)
Request 3: ████████████████░░░░░░░░░░░░░░░░ (16 tokens + 16 padding)
           ↑─────── All wait for longest ─────────↑

Continuous batching:
Request 1: ████████████████████████████████
Request 2: ████████████ ← Done! Slot freed
Request 3: ████████████████
Request 4:             ████████████████████ ← Joins when slot opens
```

**Key innovation**: Each iteration, any request can join or leave the batch.

### Phase 4: Modern Inference Systems (2023-Present)

**vLLM**, **SGLang**, **TensorRT-LLM** combine:
- Continuous batching
- PagedAttention (KV cache management)
- Speculative decoding
- Prefix caching

---

## Batching Strategies

### 1. Static Batching

Wait for fixed number of requests, process together:

```python
class StaticBatcher:
    def __init__(self, batch_size=8, max_wait=100):
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.queue = []

    def add_request(self, request):
        self.queue.append(request)
        if len(self.queue) >= self.batch_size:
            return self.process_batch()
        return None

    def process_batch(self):
        batch = self.queue[:self.batch_size]
        self.queue = self.queue[self.batch_size:]

        # Pad to same length
        max_len = max(len(r.tokens) for r in batch)
        padded = torch.stack([
            F.pad(r.tokens, (0, max_len - len(r.tokens)))
            for r in batch
        ])

        # Generate (all sequences run same number of steps)
        outputs = model.generate(padded, max_new_tokens=256)
        return outputs
```

**Use case**: Simple deployments, debugging.

### 2. Dynamic Batching (Length-Aware)

Group requests by similar length:

```python
class DynamicBatcher:
    def __init__(self, bucket_size=64):
        self.buckets = defaultdict(list)  # length_bucket → requests
        self.bucket_size = bucket_size

    def add_request(self, request):
        bucket = (len(request.tokens) // self.bucket_size + 1) * self.bucket_size
        self.buckets[bucket].append(request)

    def get_batch(self, max_batch=32):
        # Get requests from buckets with similar lengths
        for bucket in sorted(self.buckets.keys()):
            if len(self.buckets[bucket]) >= max_batch:
                batch = self.buckets[bucket][:max_batch]
                self.buckets[bucket] = self.buckets[bucket][max_batch:]
                return batch
        return None
```

**Use case**: Reduce padding waste.

### 3. Continuous Batching

Process at iteration level:

```python
class ContinuousBatcher:
    def __init__(self, max_batch=256):
        self.running = []      # Currently generating
        self.waiting = []      # Waiting to start
        self.max_batch = max_batch

    def add_request(self, request):
        self.waiting.append(request)

    def step(self):
        # Add new requests if room
        while len(self.running) < self.max_batch and self.waiting:
            request = self.waiting.pop(0)
            self.running.append(request)
            # Run prefill for new request (can be batched too)
            request.kv_cache = prefill(request.prompt)

        if not self.running:
            return []

        # Decode step for all running requests
        next_tokens = decode_step(self.running)

        # Update states, check completion
        completed = []
        still_running = []
        for i, request in enumerate(self.running):
            request.tokens.append(next_tokens[i])
            if request.is_done():
                completed.append(request)
            else:
                still_running.append(request)

        self.running = still_running
        return completed

    def run_forever(self):
        while True:
            completed = self.step()
            for request in completed:
                request.callback(request.tokens)
```

**Use case**: Production serving, high throughput.

### 4. Chunked Prefill

Split long prompts into chunks to avoid blocking:

```
Without chunked prefill:
Request 1 (prompt=4K): ████████████████ (long prefill blocks batch)
Request 2 (prompt=100): waiting...
Request 3 (prompt=200): waiting...

With chunked prefill:
Request 1 chunk 1: ████
Request 2 prefill: ██
Request 3 prefill: ███
Request 1 chunk 2: ████
...
```

```python
def chunked_prefill(prompt, chunk_size=512):
    """Process long prompts in chunks to avoid blocking."""
    kv_cache = None

    for i in range(0, len(prompt), chunk_size):
        chunk = prompt[i:i+chunk_size]
        kv_cache = process_chunk(chunk, kv_cache)

    return kv_cache
```

**Benefit**: Better latency for short requests when long ones arrive.

---

## Scheduling Strategies

### First-Come-First-Served (FCFS)

```python
def fcfs_scheduler(queue):
    return queue.popleft()
```

**Pros**: Simple, fair
**Cons**: Long requests block short ones

### Shortest-Job-First (SJF)

```python
def sjf_scheduler(queue):
    # Estimate job length from prompt + expected output
    return min(queue, key=lambda r: r.estimated_length)
```

**Pros**: Minimizes average latency
**Cons**: Long requests may starve

### Priority Queuing

```python
def priority_scheduler(queue):
    # Consider: priority, wait time, estimated length
    def score(request):
        wait_penalty = request.wait_time * 0.1
        priority_bonus = request.priority * 10
        return priority_bonus + wait_penalty - request.estimated_length

    return max(queue, key=score)
```

**Use case**: Differentiated service levels.

### Fair Scheduling

```python
class FairScheduler:
    def __init__(self):
        self.tokens_served = defaultdict(int)

    def schedule(self, queue):
        # Prioritize users/tenants who have received fewer tokens
        def fairness_score(request):
            return -self.tokens_served[request.user_id]

        request = max(queue, key=fairness_score)
        return request

    def record(self, request, tokens):
        self.tokens_served[request.user_id] += tokens
```

**Use case**: Multi-tenant deployments.

---

## Implementation

### vLLM Configuration

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    max_num_seqs=256,           # Maximum concurrent sequences
    max_num_batched_tokens=4096, # Tokens per iteration
    enable_chunked_prefill=True, # Enable chunked prefill
)

# Continuous batching happens automatically
outputs = llm.generate(prompts, SamplingParams(max_tokens=256))
```

### TensorRT-LLM Configuration

```python
import tensorrt_llm

# Build engine with batching config
config = tensorrt_llm.BuildConfig(
    max_batch_size=256,
    max_input_len=2048,
    max_output_len=2048,
    max_num_tokens=8192,  # Tokens per batch
)

# Inflight batching enabled by default
runner = tensorrt_llm.ModelRunner.from_dir(engine_dir)
```

### Custom Continuous Batching

```python
import asyncio

class InferenceServer:
    def __init__(self, model, max_batch=64):
        self.model = model
        self.max_batch = max_batch
        self.request_queue = asyncio.Queue()
        self.running_requests = {}

    async def add_request(self, request_id, prompt):
        future = asyncio.Future()
        await self.request_queue.put({
            "id": request_id,
            "prompt": prompt,
            "future": future
        })
        return await future

    async def run_engine(self):
        while True:
            # Collect requests for this iteration
            batch = []
            while len(batch) < self.max_batch:
                try:
                    request = self.request_queue.get_nowait()
                    batch.append(request)
                    self.running_requests[request["id"]] = request
                except asyncio.QueueEmpty:
                    break

            if not self.running_requests:
                await asyncio.sleep(0.001)
                continue

            # Run one decode step
            outputs = self.model.decode_step(list(self.running_requests.values()))

            # Process outputs
            for request_id, output in outputs.items():
                request = self.running_requests[request_id]
                request["tokens"].append(output)

                if output == EOS_TOKEN:
                    request["future"].set_result(request["tokens"])
                    del self.running_requests[request_id]
```

---

## Performance Considerations

### Throughput vs Latency

| Strategy | Throughput | Latency | Use Case |
|----------|------------|---------|----------|
| Static, small batch | Low | Low | Interactive |
| Static, large batch | Medium | High | Batch processing |
| Continuous | High | Medium | Production serving |
| Continuous + chunked | High | Lower | Mixed workloads |

### Metrics

```python
class BatchingMetrics:
    def __init__(self):
        self.requests_processed = 0
        self.tokens_generated = 0
        self.total_time = 0
        self.request_latencies = []
        self.batch_sizes = []

    def record(self, batch_size, tokens, latency):
        self.batch_sizes.append(batch_size)
        self.tokens_generated += tokens
        self.request_latencies.append(latency)

    def report(self):
        return {
            "throughput_tps": self.tokens_generated / self.total_time,
            "avg_batch_size": mean(self.batch_sizes),
            "p50_latency": percentile(self.request_latencies, 50),
            "p99_latency": percentile(self.request_latencies, 99),
        }
```

### Tuning Parameters

| Parameter | Effect on Throughput | Effect on Latency |
|-----------|---------------------|-------------------|
| Max batch size ↑ | ↑ Higher | ↑ Higher (queueing) |
| Max wait time ↑ | ↑ Slightly | ↑ Higher |
| Chunk size ↓ | ↔ Same | ↓ Lower for short requests |
| Memory limit ↓ | ↓ Lower | ↔ Same |

---

## Best Practices

### For Low Latency

```python
config = {
    "max_num_seqs": 32,          # Smaller batches
    "enable_chunked_prefill": True,
    "max_wait_time_ms": 10,      # Don't wait long
}
```

### For High Throughput

```python
config = {
    "max_num_seqs": 256,         # Larger batches
    "max_num_batched_tokens": 8192,
    "gpu_memory_utilization": 0.95,
}
```

### For Mixed Workloads

```python
config = {
    "max_num_seqs": 128,
    "enable_chunked_prefill": True,
    "scheduling_policy": "priority",  # If available
}
```

---

## Future Directions

### Near-term (2025)

1. **Smarter scheduling**: ML-based request scheduling
2. **Disaggregated serving**: Separate prefill and decode
3. **Multi-GPU batching**: Coordinate across GPUs
4. **Adaptive batching**: Auto-tune parameters

### Research Frontiers

1. **Predictive batching**: Anticipate request patterns
2. **SLO-aware scheduling**: Meet latency guarantees
3. **Cost-aware batching**: Optimize for cost vs performance
4. **Hybrid strategies**: Different strategies for different request types

---

## Sources

### Foundational Papers
- [ORCA: A Distributed Serving System for Transformer-Based LLMs](https://www.usenix.org/conference/osdi22/presentation/yu) - OSDI 2022
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM, 2023

### Inference Systems
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [SGLang](https://github.com/sgl-project/sglang)
- [Text Generation Inference](https://github.com/huggingface/text-generation-inference)

### Optimization Techniques
- [Splitwise: Efficient Generative LLM Inference](https://arxiv.org/abs/2311.18677) - 2023
- [Sarathi: Efficient LLM Inference by Piggybacking Decodes](https://arxiv.org/abs/2308.16369) - 2023
