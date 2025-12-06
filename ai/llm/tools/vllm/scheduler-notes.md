# vLLM Scheduler - Interactive Deep Dive Notes

## The Core Problem

You have a GPU with limited memory. Multiple requests arrive asking for text generation. The scheduler must decide:

1. Which requests to run right now?
2. How many tokens to process for each request?
3. What to do when you run out of memory?

## The Fundamental Insight: No Prefill/Decode Distinction

Most LLM serving systems think in two phases:

- **Prefill**: Process the entire prompt at once
- **Decode**: Generate tokens one at a time

**vLLM's scheduler doesn't think this way.** Instead, it just tracks:

```python
request.num_computed_tokens   # How many tokens we've already processed
request.num_tokens            # How many tokens total (prompt + generated so far)

# Goal each step: make num_computed_tokens catch up to num_tokens
```

This is more flexible - it naturally handles:

- Chunked prefills (process prompt in chunks)
- Prefix caching (skip already-computed tokens)
- Speculative decoding (verify draft tokens)

## What Makes Scheduling Non-Trivial?

The basic loop is greedy, but these are the hard problems:

### 1. Memory Management & Preemption

When you run out of GPU memory for KV cache:

```python
while True:
    new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)

    if new_blocks is not None:
        break  # Success!

    # OUT OF MEMORY! Now what?
    # Pick a victim to preempt
    victim = max(self.running, key=lambda r: (r.priority, r.arrival_time))

    # Free their KV cache
    kv_cache_manager.free(victim)

    # CRITICAL: Lose ALL progress!
    victim.num_computed_tokens = 0  # Start over from beginning
    victim.status = PREEMPTED

    # Put back in waiting queue
    self.waiting.prepend_request(victim)
```

**Non-trivial question**: Who do you preempt? The scheduler tries multiple times in a loop - if preempting one request isn't enough, it keeps preempting until it has enough memory (or gives up).

### 2. Prefix Caching - Shared KV Blocks

Example:

```
Request 1: "Translate to French: Hello world"
Request 2: "Translate to French: Goodbye"
           ^^^^^^^^^^^^^^^^^^^^
           Same 16 tokens!
```

The scheduler must:

1. **Detect** the common prefix (hash-based lookup in KV cache)
2. **Allocate** only blocks for the unique suffix
3. **Track** which blocks are shared (ref counting)
4. **Avoid freeing** shared blocks until all requests finish

Code (scheduler.py:446-474):

```python
# Check if we can reuse cached KV blocks
new_computed_blocks, num_local_cached = kv_cache_manager.get_computed_blocks(request)

# Remote cache too (KV Transfer)
if self.connector is not None:
    num_external_cached, load_async = connector.get_num_new_matched_tokens(...)

# Only schedule the non-cached tokens
num_new_tokens = request.num_tokens - (num_local_cached + num_external_cached)
```

### 3. Chunked Prefill - Balancing Prefill vs Decode

**Problem**: A 100K token prompt will block all decode requests (which generate 1 token/step).

**Solution**: Process the prompt in chunks.

```python
# Long prompt: 10,000 tokens
# Threshold: 2,000 tokens

# Step 1: schedule 2,000 tokens (0-2000)
# Step 2: schedule 2,000 tokens (2000-4000)  <- decode reqs can run between steps!
# Step 3: schedule 2,000 tokens (4000-6000)
# ...
```

**Non-trivial decision**: What threshold? Too small = poor GPU utilization. Too large = decode requests starved.

Code (scheduler.py:228-230):

```python
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold  # Cap it!
```

### 4. Token Budget Management

You can't schedule infinite tokens in one step (limited by model's `max_batch_tokens`).

```python
token_budget = self.max_num_scheduled_tokens  # e.g., 8192

# Schedule running requests first
for req in self.running:
    num_new_tokens = min(num_new_tokens, token_budget)
    token_budget -= num_new_tokens

# Then schedule waiting requests
while self.waiting and token_budget > 0:
    num_new_tokens = min(req.num_tokens - cached, token_budget)
    token_budget -= num_new_tokens
```

**Non-trivial**: Order matters! Running requests get priority (because they already have KV cache allocated - preempting them wastes that memory).

### 5. Speculative Decoding - Rollback Logic

Draft model proposes 5 tokens: `[A, B, C, D, E]`

Target model accepts first 3, rejects last 2.

**Scheduler must rollback**:

```python
scheduled_spec_token_ids = [A, B, C, D, E]  # 5 draft tokens
generated_token_ids = [A, B, C, X]  # Target accepted 3, generated new token X

num_accepted = len(generated_token_ids) - 1  # = 3
num_rejected = 5 - 3  # = 2

# ROLLBACK!
request.num_computed_tokens -= num_rejected  # Undo the 2 rejected tokens
```

This affects KV cache allocation in the next step.

## How to Decide `max_num_batched_tokens`

This is a critical config that affects throughput vs latency.

### 1. Hardware Constraint (Primary)

GPU memory limits how many tokens you can process in one forward pass:

```
GPU Memory = Model Weights + KV Cache + Activation Memory

Activation Memory ≈ num_batched_tokens × hidden_dim × num_layers × dtype_size
```

**Example** (Llama 70B on A100 80GB):

- Model weights (FP16): ~140GB → doesn't fit! Need tensor parallelism across 2-4 GPUs
- With 4×A100, assume 200GB free for KV + activations
- Hidden dim: 8192, Layers: 80, FP16: 2 bytes
- Activation per token: `8192 × 80 × 2 ≈ 1.3MB`
- Safe `max_num_batched_tokens`: ~16,384 tokens (before KV cache)

### 2. Throughput vs Latency Tradeoff

**Large `max_num_batched_tokens`** (e.g., 32768):

- ✅ Higher throughput (more requests batched together)
- ✅ Better GPU utilization
- ❌ Longer decode latency (decode reqs wait for large prefills)

**Small `max_num_batched_tokens`** (e.g., 2048):

- ✅ Lower decode latency (decode reqs get scheduled faster)
- ❌ Lower throughput (can't batch as many requests)

### 3. Workload Characteristics

**Long prompts** (e.g., RAG with 10K context):

- Need higher `max_num_batched_tokens` to avoid excessive chunking
- Example: `max_num_batched_tokens = 16384` allows 10K prompt + some decode reqs

**Short prompts** (e.g., chatbot with 100 token prompts):

- Lower `max_num_batched_tokens` is fine
- Example: `max_num_batched_tokens = 4096` fits ~40 requests generating 1 token each

### 4. Interaction with Chunked Prefill

```python
# If max_num_batched_tokens = 8192
# And long_prefill_token_threshold = 2048

# 10K prompt will be split:
# Step 1: 2048 tokens (chunked)
# Step 2: 2048 tokens (chunked)  <- decode reqs can interleave here!
# Step 3: 2048 tokens (chunked)
# Step 4: 2048 tokens (chunked)
# Step 5: 1904 tokens (final chunk)
```

**Lower `max_num_batched_tokens`** → more aggressive chunking → better decode latency but lower throughput.

### vLLM's Auto-Detection

If you don't set it explicitly:

```python
# Default:
max_num_batched_tokens = max(max_model_len, 2048)

# Where max_model_len is the model's context window (e.g., 4096, 8192, 32768)
```

**Why this default?**

- Ensures you can always process at least one full-context request
- 2048 minimum for reasonable batching

### Real-World Tuning

**Step 1**: Start with auto-detected value

**Step 2**: Monitor metrics:

```python
scheduler_stats = engine.get_stats()
print(f"Num waiting: {scheduler_stats.num_waiting_reqs}")
print(f"Num running: {scheduler_stats.num_running_reqs}")
print(f"KV cache usage: {scheduler_stats.kv_cache_usage}")
```

**Step 3**: Adjust based on symptoms:

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| High `num_waiting_reqs` | Not batching enough | ↑ `max_num_batched_tokens` |
| High decode latency (P99) | Large prefills blocking | ↓ `max_num_batched_tokens` or ↓ `long_prefill_token_threshold` |
| Low GPU utilization | Batches too small | ↑ `max_num_batched_tokens` |
| OOM errors | Activation memory too high | ↓ `max_num_batched_tokens` |

### Example: Character.AI Workload

- Mix of short chat messages (~100 tokens) and long character cards (~2K tokens)
- Optimized for low P50 latency (chat feels snappy)

**Good config**:

```python
max_num_batched_tokens = 8192  # Moderate batching
long_prefill_token_threshold = 2048  # Chunk long prompts aggressively
max_num_seqs = 256  # Allow many concurrent requests
```

This balances:

- Short chats get fast responses (not blocked by long prefills)
- Long character cards get chunked (2K tokens fit in one chunk)
- Enough batching for good throughput

**Bad config** (high throughput, terrible latency):

```python
max_num_batched_tokens = 32768  # Too large!
long_prefill_token_threshold = -1  # No chunking
```

Result: One 20K character card blocks all chat requests for entire prefill.

## What Counts Toward `num_batched_tokens`?

**Critical insight**: Only **new tokens being computed** count, not total tokens.

### Example: Prefix Cache Hit

```python
request.num_tokens = 2000  # Total tokens (prompt + generated so far)
request.num_computed_tokens = 1200  # Already cached (prefix cache hit!)

# Scheduler calculates:
num_new_tokens = request.num_tokens - request.num_computed_tokens
               = 2000 - 1200
               = 800  # Only these count toward batch!

# Add to batch
num_scheduled_tokens[req_id] = 800
token_budget -= 800  # Deduct from budget
```

### Why This Makes Sense

**GPU work is proportional to NEW tokens**, not total tokens:

```
Forward pass processes:
- KV cache for tokens 0-1199: Already computed, just read from cache
- NEW computation for tokens 1200-1999: 800 tokens of actual work
```

The attention computation is:

```
Q @ K^T where:
- Q: queries for NEW tokens (800 tokens)
- K: keys for ALL tokens (2000 tokens, but 1200 from cache)
```

So the **FLOPs and activation memory scale with the 800 new tokens**, not the full 2000.

### Code Evidence

**scheduler.py:634-635**:

```python
total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
```

And **scheduler.py:607-608** when adding a waiting request:

```python
num_scheduled_tokens[request.request_id] = num_new_tokens  # Not num_tokens!
token_budget -= num_new_tokens
```

### Real Example: Prefix Caching Amplifies Throughput

Say `max_num_batched_tokens = 8192`:

**Scenario 1: No prefix caching**

```
Request A: 2000 token prompt, 0 cached → 2000 new tokens
Request B: 2000 token prompt, 0 cached → 2000 new tokens
Request C: 2000 token prompt, 0 cached → 2000 new tokens
Request D: 2000 token prompt, 0 cached → 2000 new tokens
Total: 8000 tokens → ✅ Fits in batch! (4 requests)
```

**Scenario 2: With prefix caching**

```
Request A: 2000 tokens, 1200 cached → 800 new tokens
Request B: 2000 tokens, 1500 cached → 500 new tokens
Request C: 2000 tokens, 1800 cached → 200 new tokens
Request D: 2000 tokens, 1900 cached → 100 new tokens
Request E: 2000 tokens, 0 cached → 2000 new tokens
Request F: 2000 tokens, 500 cached → 1500 new tokens
Request G: 2000 tokens, 1000 cached → 1000 new tokens
Request H: 2000 tokens, 800 cached → 1200 new tokens
Request I: 2000 tokens, 1700 cached → 300 new tokens
Total: 800+500+200+100+2000+1500+1000+1200+300 = 7600 new tokens
      → ✅ 9 requests in one batch! (vs 4 without caching)
```

**Prefix caching can more than double throughput** by letting you batch more requests!

### Edge Case: Decode Phase

When generating tokens one-by-one:

```python
request.num_tokens = 2005  # 2000 prompt + 5 generated
request.num_computed_tokens = 2004  # Processed up to token 2004

# This step:
num_new_tokens = 2005 - 2004 = 1  # Generate 1 new token

# Batch of 100 decode requests:
total_batched_tokens = 100 × 1 = 100 tokens
```

Even though the KV cache has 200K+ tokens across all requests, you're only computing 100 new tokens!

### Chunked Prefill Example

```python
request.num_tokens = 10000  # Long prompt
request.num_computed_tokens = 0  # First time

# Without chunking:
num_new_tokens = 10000  # Exceeds budget of 8192! Can't schedule.

# With long_prefill_token_threshold = 2048:
num_new_tokens = min(10000, 2048) = 2048  # Chunk it!
# Only 2048 counts toward batch

# Next step:
request.num_computed_tokens = 2048  # Updated after step 1
num_new_tokens = min(10000 - 2048, 2048) = 2048  # Another chunk
```

## Compute Limit vs Memory Limit

**Key insight**: `max_num_batched_tokens` caps **compute (FLOPs)**, not memory.

### The Two Separate Limits

#### 1. Compute Limit: `max_num_batched_tokens`

Controls how much **attention computation** happens per forward pass:

```
Attention FLOPs ≈ num_new_tokens × total_kv_tokens × hidden_dim × num_layers

Where:
- num_new_tokens: tokens being computed THIS step
- total_kv_tokens: all tokens in KV cache (past + new)
```

**Why limit this?**

- Prevents one batch from taking too long (latency)
- Keeps forward pass time bounded (~100ms typical)

#### 2. Memory Limit: KV Cache Blocks

Controlled by **GPU profiling during initialization**, not a config!

**EngineCore.__init__() flow** (core.py:155-170):

```python
# Step 1: Load model weights
self.model_executor = executor_class(vllm_config)

# Step 2: Profile to determine KV cache size
num_gpu_blocks, num_cpu_blocks = self._initialize_kv_caches(vllm_config)
# This runs a dummy forward pass and measures leftover GPU memory!

# Step 3: Create scheduler with actual memory constraints
self.scheduler = Scheduler(
    vllm_config,
    kv_cache_config,  # Contains num_gpu_blocks
    ...
)
```

**Profiling code** (_initialize_kv_caches):

```python
# Measure free GPU memory after loading model
free_gpu_memory = torch.cuda.mem_get_info()[0]

# Each KV block stores block_size tokens (typically 16)
# Block memory = 2 (K + V) × num_layers × num_heads × head_dim × block_size × dtype
kv_block_size = calculate_kv_block_size(model_config)

# Allocate as many blocks as fit
num_gpu_blocks = int(free_gpu_memory * 0.9 / kv_block_size)  # 90% safety margin
```

**This is the REAL memory limit!**

### How They Work Together

```python
def schedule():
    token_budget = max_num_batched_tokens  # Compute limit

    for request in self.running:
        num_new_tokens = min(num_new_tokens, token_budget)

        # Try to allocate KV cache blocks (MEMORY limit)
        new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)

        if new_blocks is None:
            # OUT OF MEMORY! Preempt someone
            preempt_lowest_priority_request()
            # This frees KV cache blocks, not compute budget!
```

**Two independent gates**:

1. **Compute gate**: `token_budget > 0` → controls batch size
2. **Memory gate**: `kv_cache_manager.allocate_slots() != None` → controls total active requests

### Example Scenario

**Setup**:

- `max_num_batched_tokens = 8192` (compute limit)
- `num_gpu_blocks = 10000` (memory limit from profiling)
- `block_size = 16` tokens/block

**Step 1**: Schedule 4 new long prompts

```python
Request A: 2000 tokens, 0 cached → needs 125 blocks, schedules 2000 tokens
Request B: 2000 tokens, 0 cached → needs 125 blocks, schedules 2000 tokens
Request C: 2000 tokens, 0 cached → needs 125 blocks, schedules 2000 tokens
Request D: 2000 tokens, 0 cached → needs 125 blocks, schedules 2000 tokens

Compute: 8000 / 8192 tokens → ✅ Fits compute budget
Memory: 500 / 10000 blocks → ✅ Fits memory budget
```

**Step 2**: 100 steps later, these 4 are decoding (each generated 100 tokens)

```python
Request A: 2100 tokens total, needs 131 blocks, schedules 1 token
Request B: 2100 tokens total, needs 131 blocks, schedules 1 token
Request C: 2100 tokens total, needs 131 blocks, schedules 1 token
Request D: 2100 tokens total, needs 131 blocks, schedules 1 token

Compute: 4 / 8192 tokens → ✅ Tons of compute budget left!
Memory: 524 / 10000 blocks → ✅ Tons of memory left!

# Can schedule many more waiting requests!
```

**Step 3**: Try to schedule 50 more waiting requests (short prompts, 200 tokens each)

```python
Request E-Z (46 requests): 200 tokens, 0 cached
Each needs: 13 blocks, schedules 200 tokens

Compute: 4 (decode) + 46×200 (prefill) = 9204 tokens → ❌ EXCEEDS 8192!
Memory: 524 + 46×13 = 1122 blocks → ✅ Fits in memory

Result: Can only schedule 40 waiting requests (8000 tokens)
```

**Bottleneck**: Compute limit, not memory!

**Step 4**: Later, 1000 short decode requests running (each 200 tokens, generated 50 so far)

```python
1000 requests × 250 tokens / 16 tokens per block = 15,625 blocks
Memory: 15,625 / 10000 blocks → ❌ OUT OF MEMORY!

Compute: 1000 requests × 1 token = 1000 tokens → ✅ Fits compute budget

Result: PREEMPTION! Must evict ~300 requests to free 5,625 blocks
```

**Bottleneck**: Memory limit, not compute!

### Why Two Separate Limits?

**Compute limit** prevents:

- Long forward passes (bad latency)
- GPU underutilization from tiny batches

**Memory limit** prevents:

- OOM crashes
- Excessive preemption

They're **independent** because:

```
Compute ∝ num_new_tokens (this step only)
Memory ∝ num_total_tokens × num_requests (cumulative across all active requests)
```

A batch can be:

- **Compute-bound**: 10 long prefills (8000 new tokens) → low memory usage
- **Memory-bound**: 1000 decode requests (1000 new tokens) → high memory usage

## Key Takeaways

1. **Scheduler is greedy but non-trivial** - the hard parts are memory management, preemption, prefix caching, chunking, and budget management.

2. **No prefill/decode distinction** - just "catch up `num_computed_tokens` to `num_tokens`" which naturally handles chunking, caching, and speculation.

3. **`max_num_batched_tokens` is a compute limit** - controls FLOPs per forward pass, not memory usage.

4. **Memory limit comes from profiling** - `num_gpu_blocks` is auto-detected based on free GPU memory after loading model weights.

5. **Prefix caching amplifies throughput** - only new tokens count toward batch, so cache hits let you fit more requests.

6. **Two independent bottlenecks** - compute (token budget) and memory (KV cache blocks) are checked separately.

7. **Preemption is expensive** - loses ALL progress, request must restart from scratch.

8. **Chunked prefill trades throughput for latency** - allows decode requests to interleave with long prefills.
