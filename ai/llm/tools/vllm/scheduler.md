# vLLM Scheduler Deep Dive

## Overview

The Scheduler is the "brain" of vLLM - it decides which requests to execute and how to batch them for maximum GPU utilization while respecting memory constraints.

**Location**: `vllm/v1/core/sched/scheduler.py` (1618 lines)

**Core Responsibility**: Given available GPU memory and token budgets, determine which requests to run and how many tokens to process for each request in the current step.

## Key Insight: No Prefill/Decode Distinction

From scheduler.py:189-199:

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has `num_computed_tokens` and `num_tokens_with_spec`. At each step, the scheduler tries to assign tokens to the requests so that each request's `num_computed_tokens` can catch up its `num_tokens_with_spec`.

This design is general enough to cover:

- Chunked prefills (split large prompts across multiple steps)
- Prefix caching (skip already-computed tokens)
- Speculative decoding (verify draft tokens)
- Future optimizations (jump decoding)

## Architecture

```
Scheduler
├── Request Queues (state containers)
│   ├── self.requests: dict[str, Request]  # Master registry
│   ├── self.waiting: RequestQueue         # Priority queue or FCFS deque
│   └── self.running: list[Request]        # Currently executing
│
├── Resource Managers
│   ├── KVCacheManager    # Allocates GPU KV cache blocks
│   └── EncoderCacheManager  # Manages vision encoder outputs
│
├── Scheduling Constraints
│   ├── max_num_running_reqs       # Max concurrent requests
│   ├── max_num_scheduled_tokens   # Max tokens per step
│   └── max_model_len              # Model's max context length
│
└── Scheduling Policies
    ├── FCFS (First-Come-First-Served)
    └── PRIORITY (lowest priority value first)
```

## The Main Loop: schedule()

**Entry Point**: scheduler.py:189

Returns `SchedulerOutput` containing:

- Which requests to run
- How many tokens to process for each request
- Block allocations for KV cache
- Encoder inputs to process
- Spec decode tokens

### High-Level Algorithm

```python
def schedule() -> SchedulerOutput:
    # 1. Schedule RUNNING requests first (already have KV cache)
    for req in self.running:
        num_new_tokens = calculate_tokens_to_schedule(req)
        new_blocks = kv_cache_manager.allocate_slots(req, num_new_tokens)

        if new_blocks is None:
            # Out of memory! Preempt lowest-priority request
            preempt_request()
            retry_allocation()

    # 2. Schedule WAITING requests (new or resumed from preemption)
    while self.waiting and has_token_budget and len(running) < max_running_reqs:
        req = self.waiting.peek()

        # Check prefix cache (local + remote KV transfer)
        num_cached_tokens = get_cached_tokens(req)
        num_new_tokens = req.num_tokens - num_cached_tokens

        # Allocate KV cache blocks
        new_blocks = kv_cache_manager.allocate_slots(req, num_new_tokens)

        if new_blocks is None:
            break  # Out of memory

        self.running.append(req)
        req.status = RequestStatus.RUNNING

    return SchedulerOutput(...)
```

### Step 1: Schedule RUNNING Requests (lines 218-380)

**Priority**: Existing requests get scheduled first to avoid wasting their KV cache.

**Key Logic**:

```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]

    # Calculate how many tokens to schedule
    num_new_tokens = (
        request.num_tokens_with_spec           # Tokens needed (including spec tokens)
        + request.num_output_placeholders       # For async scheduling
        - request.num_computed_tokens           # Already processed
    )

    # Respect chunked prefill threshold
    if 0 < long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = long_prefill_token_threshold

    # Don't exceed token budget
    num_new_tokens = min(num_new_tokens, token_budget)

    # Don't exceed max model length
    max_total_tokens = min(
        request.num_prompt_tokens + request.max_tokens,
        self.max_model_len
    )
    num_new_tokens = min(num_new_tokens, max_total_tokens - 1 - request.num_computed_tokens)
```

**Preemption Logic** (lines 276-335):

When `allocate_slots()` returns `None` (out of memory):

```python
while True:
    new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)

    if new_blocks is not None:
        break  # Success!

    # Out of memory - must preempt
    if policy == SchedulingPolicy.PRIORITY:
        # Preempt HIGHEST priority value (lowest priority)
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
    else:
        # FCFS: preempt last request
        preempted_req = self.running.pop()

    kv_cache_manager.free(preempted_req)
    encoder_cache_manager.free(preempted_req)
    preempted_req.status = RequestStatus.PREEMPTED
    preempted_req.num_computed_tokens = 0  # Restart from scratch!
    preempted_req.num_preemptions += 1

    self.waiting.prepend_request(preempted_req)  # Put back in waiting queue

    if preempted_req == request:
        break  # Can't schedule this request even after preempting itself
```

**Important**: Preempted requests lose ALL progress (`num_computed_tokens = 0`) and must restart from the beginning.

### Step 2: Schedule WAITING Requests (lines 395-628)

**Constraints Checked**:

1. Token budget not exhausted
2. Haven't hit `max_num_running_reqs`
3. Request ready (not waiting for remote KV or FSM compilation)
4. LoRA constraint satisfied (max concurrent LoRAs)

**Prefix Caching** (lines 446-480):

```python
if request.num_computed_tokens == 0:
    # First time scheduling this request

    # Check LOCAL prefix cache
    new_computed_blocks, num_local_cached = kv_cache_manager.get_computed_blocks(request)

    # Check REMOTE prefix cache (KV Transfer)
    if self.connector is not None:
        num_external_cached, load_async = connector.get_num_new_matched_tokens(
            request, num_local_cached
        )

    # Total cached tokens
    num_computed_tokens = num_local_cached + num_external_cached
else:
    # Request was preempted and resumed, already has num_computed_tokens set
    num_computed_tokens = request.num_computed_tokens
```

**Chunked Prefill** (lines 491-511):

```python
num_new_tokens = request.num_tokens - num_computed_tokens

# Apply chunking threshold
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold

# Check if chunking is disabled (for pooling tasks)
if not enable_chunked_prefill and num_new_tokens > token_budget:
    # Can't fit entire prompt - skip this request
    self.waiting.pop_request()
    skipped_waiting_requests.prepend_request(request)
    continue
```

**Request Promotion to RUNNING** (lines 581-613):

```python
request = self.waiting.pop_request()

if load_kv_async:
    # KV Transfer: waiting for remote KV cache to load
    skipped_waiting_requests.prepend_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    continue

# Move to RUNNING
self.running.append(request)
req_to_new_blocks[request.request_id] = kv_cache_manager.get_blocks(request.request_id)
num_scheduled_tokens[request.request_id] = num_new_tokens
token_budget -= num_new_tokens
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_computed_tokens

# Track prefix cache hits
if request.num_cached_tokens < 0:
    request.num_cached_tokens = num_computed_tokens
```

## Request Queues

**Two Implementations** (request_queue.py):

### FCFSRequestQueue (FCFS Policy)

- Based on `collections.deque`
- O(1) append/pop
- Simple FIFO ordering

### PriorityRequestQueue (PRIORITY Policy)

- Based on `heapq` (min-heap)
- Ordering: `(priority, arrival_time, request)`
- Lower priority values scheduled first
- Tie-breaking by arrival time (FCFS within same priority)

**Key Operations**:

```python
# Add request
self.waiting.add_request(request)

# Peek next request (without removing)
request = self.waiting.peek_request()

# Pop next request
request = self.waiting.pop_request()

# Prepend (for preempted requests)
self.waiting.prepend_request(request)

# Remove multiple requests
self.waiting.remove_requests(finished_requests)
```

## Resource Management

### KVCacheManager

**Responsibility**: Allocate and free KV cache blocks (GPU memory).

**Key Methods**:

```python
# Allocate blocks for request
blocks = kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_computed_tokens=0,
    computed_blocks=None,
    num_lookahead_tokens=0,  # For speculative decoding
)

# Free blocks when request finishes
kv_cache_manager.free(request)

# Get existing blocks for request
blocks = kv_cache_manager.get_blocks(request_id)

# Check prefix cache
computed_blocks, num_cached_tokens = kv_cache_manager.get_computed_blocks(request)

# Cache blocks for prefix sharing
kv_cache_manager.cache_blocks(request, num_computed_tokens)
```

**Memory Tracking**:

- Tracks `usage` (percentage of GPU KV cache blocks used)
- Returns `None` from `allocate_slots()` when out of memory

### EncoderCacheManager

**Responsibility**: Manage vision encoder outputs (for multimodal models).

**Why Separate Cache?**:

- Vision encoders process images/video once, cache the embeddings
- Different memory budget from KV cache
- Shared across requests with same image

**Key Methods**:

```python
# Check if encoder output is cached
is_cached = encoder_cache_manager.check_and_update_cache(request, encoder_input_idx)

# Allocate cache slot
encoder_cache_manager.allocate(request, encoder_input_idx)

# Free encoder cache for finished request
encoder_cache_manager.free(request)
```

## Scheduling Output

**SchedulerOutput** (output.py:150-194):

```python
@dataclass
class SchedulerOutput:
    # New requests (first time scheduled)
    scheduled_new_reqs: list[NewRequestData]

    # Running/resumed requests (already cached in workers)
    scheduled_cached_reqs: CachedRequestData

    # req_id -> num_tokens to process this step
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int

    # Speculative decoding
    scheduled_spec_decode_tokens: dict[str, list[int]]

    # Multimodal
    scheduled_encoder_inputs: dict[str, list[int]]

    # Prefix caching
    num_common_prefix_blocks: list[int]

    # Cleanup
    finished_req_ids: set[str]
    free_encoder_mm_hashes: list[str]

    # Grammar-constrained generation
    pending_structured_output_tokens: bool = False

    # KV Transfer metadata
    kv_connector_metadata: KVConnectorMetadata | None = None
    ec_connector_metadata: ECConnectorMetadata | None = None
```

**NewRequestData** (output.py:36-63):

Used for requests scheduled for the first time:

```python
@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec]  # Images/video/audio
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    block_ids: tuple[list[int], ...]  # KV cache block IDs
    num_computed_tokens: int  # Prefix cached tokens
    lora_request: LoRARequest | None
    prompt_embeds: torch.Tensor | None = None
```

**CachedRequestData** (output.py:102-145):

Used for running/resumed requests (workers already have full context):

```python
@dataclass
class CachedRequestData:
    req_ids: list[str]
    resumed_req_ids: set[str]  # Which ones were preempted and resumed
    new_token_ids: list[list[int]]  # For pipeline parallelism
    all_token_ids: dict[str, list[int]]  # For first-time scheduled in batch
    new_block_ids: list[tuple[list[int], ...] | None]
    num_computed_tokens: list[int]
    num_output_tokens: list[int]
```

## Update from Output: update_from_output()

**Entry Point**: scheduler.py:964

**Called After**: ModelExecutor returns sampled tokens.

**Responsibilities**:

1. Update request state with new tokens
2. Check stopping conditions
3. Free finished requests
4. Create `EngineCoreOutput` for each request

### Flow

```python
def update_from_output(
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:

    sampled_token_ids = model_runner_output.sampled_token_ids

    stopped_running_reqs = set()

    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
        request = self.requests.get(req_id)
        req_index = model_runner_output.req_id_to_index[req_id]
        generated_token_ids = sampled_token_ids[req_index].tolist()

        # Handle speculative decoding rejections
        if scheduled_spec_token_ids:
            num_accepted = len(generated_token_ids) - 1
            num_rejected = num_draft_tokens - num_accepted
            request.num_computed_tokens -= num_rejected  # Roll back!

        # Update request with new tokens
        new_token_ids, stopped = self._update_request_with_output(
            request, generated_token_ids
        )

        if stopped:
            kv_transfer_params = self._free_request(request)
            stopped_running_reqs.add(request)

        # Create output
        outputs[request.client_index].append(
            EngineCoreOutput(
                request_id=req_id,
                new_token_ids=new_token_ids,
                finish_reason=request.get_finished_reason(),
                # ... other fields
            )
        )

    # Remove stopped requests from running queue
    if stopped_running_reqs:
        self.running = remove_all(self.running, stopped_running_reqs)

    return engine_core_outputs
```

### Stopping Logic: check_stop()

**Location**: utils.py:42-72

```python
def check_stop(request: Request, max_model_len: int, pooler_output=None) -> bool:
    # 1. Pooling models (embeddings)
    if request.pooling_params:
        if pooler_output is not None:
            request.status = RequestStatus.FINISHED_STOPPED
            return True
        return False

    sampling_params = request.sampling_params

    # 2. Min tokens constraint
    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    # 3. EOS token
    last_token_id = request.output_token_ids[-1]
    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    # 4. Custom stop token IDs
    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = RequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True

    # 5. Length limit
    if (request.num_tokens >= max_model_len or
        request.num_output_tokens >= request.max_tokens):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True

    return False
```

**Important**: This only checks token-level stopping. String-level stopping (stop strings like "###") happens in `OutputProcessor` (API layer) after detokenization.

## Advanced Features

### 1. Chunked Prefill

**Config**: `scheduler_config.long_prefill_token_threshold`

**Purpose**: Split large prompts across multiple steps to avoid blocking decode requests.

**Example**:

- Prompt: 10,000 tokens
- Threshold: 2,000 tokens
- Steps:
  - Step 1: Process tokens 0-2000
  - Step 2: Process tokens 2000-4000
  - Step 3: Process tokens 4000-6000
  - Step 4: Process tokens 6000-8000
  - Step 5: Process tokens 8000-10000

**Code** (scheduler.py:228-229, 496-498):

```python
# For running requests
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold

# For waiting requests
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold
```

### 2. Prefix Caching

**Purpose**: Share KV cache blocks across requests with common prefixes.

**Example**:

```
Request 1: "Translate to French: Hello world"
Request 2: "Translate to French: How are you"
                          ^^^^^^^
                     Shared prefix (16 tokens)
```

**Flow**:

```python
# Get cached blocks for new request
new_computed_blocks, num_cached_tokens = kv_cache_manager.get_computed_blocks(request)

# Schedule only the non-cached portion
num_new_tokens = request.num_tokens - num_cached_tokens

# After execution, cache the blocks for future sharing
kv_cache_manager.cache_blocks(request, num_computed_tokens)
```

**Benefits**:

- Skip recomputing shared prefixes
- Lower latency for requests with common prompts
- Higher throughput (more requests fit in memory)

### 3. Speculative Decoding

**Purpose**: Use a small draft model to predict multiple tokens, verify with target model.

**Flow**:

1. Draft model generates N spec tokens
2. Scheduler schedules `num_tokens + N` for verification
3. Target model accepts/rejects each spec token
4. Scheduler rolls back `num_computed_tokens` for rejected tokens

**Code** (scheduler.py:1018-1040):

```python
scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
if scheduled_spec_token_ids:
    num_draft_tokens = len(scheduled_spec_token_ids)
    num_accepted = len(generated_token_ids) - 1
    num_rejected = num_draft_tokens - num_accepted

    # Roll back rejected tokens
    if request.num_computed_tokens > 0:
        request.num_computed_tokens -= num_rejected
    if request.num_output_placeholders > 0:
        request.num_output_placeholders -= num_rejected
```

### 4. Pipeline Parallelism

**Purpose**: Overlap scheduling next batch while executing current batch.

**Key Difference**: Scheduler must send `new_token_ids` in `CachedRequestData` because first-stage and last-stage workers can't communicate directly.

**Code** (scheduler.py:769-778):

```python
if self.use_pp:
    # Send sampled tokens back to first-stage worker
    token_ids = req.all_token_ids[
        req.num_computed_tokens : req.num_computed_tokens + num_tokens
    ]
    new_token_ids.append(token_ids)
```

### 5. KV Transfer (Disaggregated Prefill/Decode)

**Purpose**: Prefill on one cluster, transfer KV cache, decode on another cluster.

**States**:

- `RequestStatus.WAITING` - Normal waiting
- `RequestStatus.WAITING_FOR_REMOTE_KVS` - Async KV transfer in progress

**Async Loading Flow** (scheduler.py:582-587):

```python
if load_kv_async:
    # Allocate memory, start async transfer
    skipped_waiting_requests.prepend_request(request)
    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    continue  # Skip this request, revisit next step
```

**Completion Check** (scheduler.py:1411-1455):

```python
def _update_waiting_for_remote_kv(request: Request) -> bool:
    if request.request_id not in self.finished_recving_kv_req_ids:
        return False  # Still waiting

    # Transfer complete! Cache the blocks
    block_ids = kv_cache_manager.get_block_ids(request.request_id)
    num_computed_tokens = len(block_ids) * block_size
    kv_cache_manager.cache_blocks(request, num_computed_tokens)

    request.num_computed_tokens = num_computed_tokens
    request.status = RequestStatus.WAITING  # Ready to schedule!
    return True
```

### 6. Multimodal (Vision Encoder)

**Purpose**: Process images/video with vision encoder, cache outputs.

**Scheduling** (scheduler.py:803-937):

```python
def _try_schedule_encoder_inputs(request, num_computed_tokens, num_new_tokens):
    encoder_inputs_to_schedule = []

    for i, mm_feature in enumerate(request.mm_features):
        start_pos = mm_feature.mm_position.offset
        num_encoder_tokens = mm_feature.mm_position.length

        # Check if encoder output needed for this step
        if start_pos >= num_computed_tokens + num_new_tokens:
            break  # Not needed yet

        if start_pos + num_encoder_tokens <= num_computed_tokens:
            continue  # Already computed

        # Check encoder cache
        if encoder_cache_manager.check_and_update_cache(request, i):
            continue  # Already cached

        # Check remote encoder cache (EC Transfer)
        if ec_connector is not None and remote_cache_has_item[i]:
            external_load_encoder_input.append(i)
            continue

        # Check encoder budget
        if not encoder_cache_manager.can_allocate(request, i, encoder_budget):
            # Out of budget - schedule only decoder tokens before encoder input
            num_new_tokens = start_pos - num_computed_tokens
            break

        encoder_inputs_to_schedule.append(i)
        encoder_budget -= num_encoder_tokens

    return encoder_inputs_to_schedule, num_new_tokens, encoder_budget
```

### 7. Grammar-Constrained Generation (Structured Output)

**Purpose**: Constrain outputs to follow JSON schema, regex, etc.

**Bitmask**: For each vocab token, 1 = allowed, 0 = forbidden.

**Flow** (scheduler.py:939-962):

```python
def get_grammar_bitmask(scheduler_output: SchedulerOutput) -> GrammarOutput | None:
    # Collect requests using structured output
    structured_output_request_ids = [
        req_id
        for req_id in scheduler_output.num_scheduled_tokens
        if (req := self.requests.get(req_id)) and req.use_structured_output
    ]

    if not structured_output_request_ids:
        return None

    # Generate bitmask (vocab_size bits per request)
    bitmask = structured_output_manager.grammar_bitmask(
        self.requests,
        structured_output_request_ids,
        scheduled_spec_decode_tokens,
    )

    return GrammarOutput(structured_output_request_ids, bitmask)
```

**Model Runner**: Applies bitmask before sampling (sets logits to -inf for forbidden tokens).

## Request Lifecycle

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. New Request Added                                                 │
│    add_request(request)                                              │
│    └─> self.waiting.add_request(request)                            │
│    └─> self.requests[req_id] = request                              │
│    └─> request.status = WAITING                                     │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. First Schedule (WAITING → RUNNING)                               │
│    schedule()                                                        │
│    ├─> Check prefix cache (local + remote)                          │
│    ├─> Allocate KV cache blocks                                     │
│    ├─> self.running.append(request)                                 │
│    ├─> request.status = RUNNING                                     │
│    └─> request.num_computed_tokens = num_cached_tokens              │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 3. Continuous Execution (RUNNING → RUNNING)                         │
│    schedule() repeatedly                                             │
│    ├─> Calculate num_new_tokens                                     │
│    ├─> Allocate additional KV blocks if needed                      │
│    └─> request.num_computed_tokens += num_new_tokens                │
└──────────────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
┌───────────────────┐ ┌────────────────┐ ┌──────────────────┐
│ 4a. Preemption    │ │ 4b. Finish     │ │ 4c. Abort        │
│     (OOM)         │ │     (EOS)      │ │     (Client)     │
├───────────────────┤ ├────────────────┤ ├──────────────────┤
│ RUNNING →         │ │ RUNNING →      │ │ ANY →            │
│ PREEMPTED         │ │ FINISHED_*     │ │ FINISHED_ABORTED │
│                   │ │                │ │                  │
│ Free KV cache     │ │ Free KV cache  │ │ Free KV cache    │
│ num_computed = 0  │ │ Delete from    │ │ Delete from      │
│ Prepend to        │ │ self.requests  │ │ self.requests    │
│ self.waiting      │ │                │ │                  │
└───────────────────┘ └────────────────┘ └──────────────────┘
        │
        └─────────────────┐
                          ▼
            ┌──────────────────────────────┐
            │ 5. Resume from Preemption    │
            │    schedule()                │
            │    └─> Allocate KV cache     │
            │    └─> PREEMPTED → RUNNING   │
            │    └─> num_computed = 0      │
            │        (restart from scratch)│
            └──────────────────────────────┘
```

## Performance Optimizations

### 1. Fast Path for Single Item Removal

**Location**: utils.py:10-39

```python
def remove_all(lst: list, items_to_remove: set) -> list:
    if len(items_to_remove) == 1:
        # Fast path: O(n) single remove vs O(n) list comprehension
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst

    # Multiple items: use list comprehension
    return [item for item in lst if item not in items_to_remove]
```

**Why**: Most steps finish 0-1 requests, so optimizing the common case matters.

### 2. Cached Request Data

**Purpose**: Workers cache full request context on first schedule. Subsequent schedules only send diff (new tokens, new blocks).

**Benefits**:

- Lower serialization cost
- Lower network transfer (for distributed workers)
- Lower deserialization cost

### 3. Early Exit When Idle

**Location**: scheduler.py:189 (in EngineCore.step())

```python
if not self.scheduler.has_requests():
    return {}, False  # Skip scheduling and execution
```

**Benefits**: Avoid unnecessary work when no requests pending.

### 4. Batch KV Cache Allocation

All allocations happen in single `schedule()` call, enabling better memory planning.

## Key Metrics

**SchedulerStats** (scheduler.py:1320-1338):

```python
@dataclass
class SchedulerStats:
    num_running_reqs: int
    num_waiting_reqs: int
    kv_cache_usage: float  # Percentage of GPU KV cache used
    prefix_cache_stats: PrefixCacheStats  # Hit rate, tokens saved
    connector_prefix_cache_stats: PrefixCacheStats  # Remote cache hits
    spec_decoding_stats: SpecDecodingStats  # Acceptance rate
    kv_connector_stats: dict  # KV transfer metrics
```

**PrefixCacheStats**:

- `num_requests` - Total requests
- `num_hits` - Requests with prefix cache hits
- `num_tokens` - Total tokens requested
- `num_hit_tokens` - Tokens served from cache

**SpecDecodingStats**:

- `num_draft_tokens` - Speculative tokens proposed
- `num_accepted_tokens` - Tokens accepted by target model

## Common Pitfalls

### 1. Preemption Loses All Progress

```python
preempted_req.num_computed_tokens = 0  # Restart from scratch!
```

**Why**: KV cache blocks are freed on preemption. Can't resume from middle.

**Implication**: High preemption rate = wasted compute. Size your GPU appropriately.

### 2. Chunked Prefill Can Delay Decodes

Long prefills scheduled in chunks block decode requests from starting.

**Mitigation**: Tune `long_prefill_token_threshold` based on workload.

### 3. Priority Inversion

With PRIORITY policy, lower-priority requests can get preempted repeatedly by higher-priority requests.

**Code** (scheduler.py:290-294):

```python
# Preempt HIGHEST priority value (lowest priority request)
preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
```

**Mitigation**: Use FCFS for fair scheduling.

### 4. Prefix Cache Fragmentation

Slight prompt variations prevent cache hits:

```python
"Translate to French: Hello"
"Translate to French:  Hello"  # Extra space!
                      ^
              Different tokens, no cache hit
```

**Mitigation**: Normalize prompts at API layer.

## Summary

The Scheduler is a sophisticated resource manager that:

1. **Decides what to run**: Picks requests from waiting queue, schedules running requests
2. **Manages memory**: Allocates KV cache blocks, preempts when out of memory
3. **Optimizes throughput**: Batches requests, chunks large prefills, leverages prefix caching
4. **Handles advanced features**: Speculative decoding, multimodal, KV transfer, structured output

**Key Takeaways**:

- No prefill/decode distinction - just "catch up `num_computed_tokens` to `num_tokens_with_spec`"
- Preemption is expensive (loses all progress)
- Prefix caching is powerful (skip recomputing common prefixes)
- Chunked prefill balances latency (prefill) vs throughput (decode)
- Resource constraints (memory, token budget) drive all scheduling decisions
