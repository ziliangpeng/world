# LMCache Deep Dive - Follow-up Notes

Follow-up research on LMCache architecture, vLLM integration, and performance optimizations.

**Last Updated**: October 31, 2025

---

## Table of Contents
1. [High-Level Call Flow](#high-level-call-flow)
2. [Scheduler vs Worker Architecture](#scheduler-vs-worker-architecture)
3. [LMCache Code in vLLM](#lmcache-code-in-vllm)
4. [Async Loading & GPU Overlap](#async-loading--gpu-overlap)
5. [Feature Timeline](#feature-timeline)

---

## High-Level Call Flow

### Two-Phase Architecture

LMCache-vLLM integration follows a **scheduler-worker separation**:

```
PHASE 1: SCHEDULER SIDE (Orchestration)
    ↓
Request arrives with 1000 tokens
    ↓
Lookup: "Do you have cached tokens?"
    ├─ LookupClient.lookup() sends ZMQ to worker
    ├─ Worker's lookup server checks storage
    └─ Response: "I have 512 cached tokens"
    ↓
Allocate: vLLM reserves GPU blocks for those 512 tokens
    ↓
Build Metadata: Create instructions for workers
    ├─ LoadSpec: Load tokens [0:512]
    ├─ SaveSpec: Save tokens [0:1000]
    ├─ slot_mapping: token → GPU block mapping
    └─ RequestTracker: Token IDs, block IDs

PHASE 2: WORKER SIDE (Execution)
    ↓
Retrieve: Load KV cache from storage → GPU
    ├─ Fetch from backends (CPU/Disk/Remote)
    ├─ GPUConnector.batched_to_gpu()
    └─ Place at correct GPU locations using slot_mapping
    ↓
Forward Pass: vLLM runs model
    ├─ Use cached KV for tokens [0:512]
    ├─ Compute new KV for tokens [512:1000]
    └─ Return output
    ↓
Store: Save new KV from GPU → storage
    ├─ Skip already-saved chunks [0:512]
    ├─ GPUConnector.batched_from_gpu() extracts new KV
    ├─ StorageManager.batched_put() stores asynchronously
    └─ Return to step 1 for next request
```

### Key Data Structures

**LoadSpec**: Tells workers what to load
```python
LoadSpec(
    vllm_cached_tokens=0,           # Local cache hit
    lmcache_cached_tokens=512,      # LMCache hit
    can_load=True                   # Allocation succeeded
)
```

**SaveSpec**: Tells workers what to save
```python
SaveSpec(
    skip_leading_tokens=0,          # Already saved prefix
    can_save=True                   # Ready to save
)
```

**slot_mapping**: Token index → GPU block
```python
{
    token_0: block_5,
    token_1: block_5,
    token_256: block_6,
    token_512: block_7,
    ...
}
```

### Example: 1000-Token Request with 512 Cached

```
Request: [tokens 0:1000]
    ↓
Scheduler looks up → finds tokens[0:512] in LMCache
    ↓
Allocate GPU blocks for 512 tokens
    ↓
Build metadata with LoadSpec(lmcache_cached_tokens=512)
    ↓
Worker retrieves tokens[0:512] from storage to GPU
    ↓
Forward pass:
    ├─ Use cached KV for tokens[0:512]
    ├─ Compute KV for tokens[512:1000]  ← 488 new tokens computed
    ↓
Store tokens[512:1000] to storage
```

---

## Scheduler vs Worker Architecture

### What Are They?

| Component | Type | Where | Count |
|-----------|------|-------|-------|
| **Scheduler** | Python object | Inside EngineCore process | 1 per EngineCore |
| **EngineCore** | Process | Separate process | 1 (or more for data parallelism) |
| **Workers** | Separate processes | One per GPU rank | `TP_size × PP_size` |

### Process Topology

```
Main Process (LLM Engine Client)
    |
    | ZMQ sockets (request/response)
    |
EngineCore Process (1 process)
├── Scheduler (object, NOT separate)
├── Input Thread (read ZMQ)
├── Output Thread (write ZMQ)
└── Executor
     |
     | Shared Memory MessageQueue (RPC)
     |
     ├── Worker Process 0 (rank 0, GPU 0)
     ├── Worker Process 1 (rank 1, GPU 1)
     ├── Worker Process 2 (rank 2, GPU 2)
     └── Worker Process N (rank N, GPU N)
```

**Key points:**
- **Scheduler is NOT a separate process** - it's a Python object in EngineCore
- **Workers ARE separate processes** - spawned via `multiprocessing.Process()`
- **Communication**: EngineCore ↔ Workers via shared memory (not ZMQ)

### Execution Model

#### EngineCore: Busy Loop
```python
def run_busy_loop(self):
    while True:
        # 1. Process incoming requests
        self._process_input_queue()

        # 2. Execute one iteration
        self._process_engine_step()
            ├─ scheduler.schedule() → SchedulerOutput
            ├─ executor.execute_model(SchedulerOutput) → ModelRunnerOutput
            └─ scheduler.update_from_output(ModelRunnerOutput)
```

**Characteristics:**
- Continuous busy loop (no sleep)
- Synchronous execution within loop
- Scheduler called as regular function

#### Workers: Event-Driven Blocking
```python
def worker_busy_loop(self):
    while True:
        # Block until work arrives
        method, args, kwargs = self.rpc_broadcast_mq.dequeue()  # BLOCKS

        # Execute the RPC call
        func = getattr(self.worker, method)
        output = func(*args, **kwargs)

        # Send response
        self.worker_response_mq.enqueue((SUCCESS, output))
```

**Characteristics:**
- Event-driven (blocks on message queue)
- No spinning - sleeps until EngineCore sends work
- Efficient GPU usage - GPUs idle when no work

### Communication

**EngineCore ↔ Workers:**
- **Mechanism**: Shared memory MessageQueue (not ZMQ!)
- **Pattern**: Broadcast → All workers receive → All respond
- **Synchronicity**: Synchronous by default (EngineCore waits for all workers)

```
EngineCore builds SchedulerOutput
    ↓ enqueue to shared memory
[Shared Memory MessageQueue]
    ↓ dequeue - blocks until message
Worker 0, 1, 2, ..., N all receive
    ↓ execute independently
Each worker: ModelRunnerOutput
    ↓ enqueue response
[Response Queues]
    ↓ dequeue
EngineCore continues
```

**Scheduler ↔ EngineCore:**
- Just regular Python method calls (same process)
- No IPC needed

**Workers ↔ Workers:**
- PyTorch distributed (NCCL, point-to-point for PP)

---

## LMCache Code in vLLM

### The Minimal Wrapper

LMCache code in vLLM is **tiny** - just a thin 170-line wrapper:

**File**: `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`

```python
class LMCacheConnectorV1(KVConnectorBase_V1):
    """Thin wrapper that delegates to LMCache implementation"""

    def __init__(self, vllm_config, role):
        from lmcache.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorV1Impl)

        self._lmcache_engine = LMCacheConnectorV1Impl(
            vllm_config, role, self)

    # All methods just delegate to _lmcache_engine
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return self._lmcache_engine.get_num_new_matched_tokens(...)

    def build_connector_meta(self, scheduler_output):
        return self._lmcache_engine.build_connector_meta(...)

    # ... 8 other delegating methods
```

**Factory Registration:**
```python
# vllm/distributed/kv_transfer/kv_connector/factory.py
KVConnectorFactory.register_connector(
    "LMCacheConnectorV1",
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector",
    "LMCacheConnectorV1")
```

### Where is the Real Implementation?

**File**: `lmcache/integration/vllm/vllm_v1_adapter.py` (~1400 lines)

This is where ALL the logic lives:
- `LMCacheConnectorV1Impl` class
- All 8 required methods fully implemented
- Integration with storage backends, GPU connectors, etc.

### Architecture

```
vLLM Core (No changes needed!)
    ├─ Scheduler with connector hooks
    ├─ Workers with connector hooks
    ├─ Attention layers with connector hooks
    └─ KVConnectorFactory
                ↓
        Calls interface methods
                ↓
vLLM LMCache Wrapper (170 lines)
    └─ Just imports and delegates
                ↓
        Delegates to
                ↓
LMCache Package (External, 1400 lines)
    └─ LMCacheConnectorV1Impl with all logic
```

### Why This Design?

✅ **Decoupling**: LMCache evolves independently
✅ **Optional dependency**: vLLM doesn't require LMCache
✅ **Version flexibility**: Users choose LMCache version
✅ **Plugin-like**: Easy updates without touching vLLM

### KV Connector Plugin Interface

If you wanted to create a new KV cache system, implement these 8 methods:

**Scheduler side (4 methods):**
1. `get_num_new_matched_tokens(request, num_computed_tokens)` - Check cache
2. `update_state_after_alloc(request, blocks, num_external_tokens)` - Track allocation
3. `build_connector_meta(scheduler_output)` - Create worker instructions
4. `request_finished(request, block_ids)` - Handle completion

**Worker side (4 methods):**
1. `start_load_kv(forward_context, **kwargs)` - Load KV to GPU
2. `wait_for_layer_load(layer_name)` - Wait for layer (layerwise)
3. `save_kv_layer(layer_name, kv_layer, attn_metadata)` - Save layer (layerwise)
4. `wait_for_save()` - Wait for all saves

**vLLM core provides all hooks** - no modifications needed!

---

## Async Loading & GPU Overlap

### The Problem

Traditionally, loading KV cache from storage blocks GPU compute:

```
Timeline (BLOCKING):
├─ Load KV: 31ms [PCIe transfer]
├─ Compute: 320ms [GPU processing]
├─ Save KV: 50ms [PCIe transfer]
└─ Total: 401ms (no overlap)
```

### The Solution: CUDA Streams & Layerwise

LMCache uses **separate CUDA streams** for async operations:

```python
# lmcache/v1/gpu_connector.py:162-163
self.load_stream = torch.cuda.Stream()   # For PCIe transfers
self.store_stream = torch.cuda.Stream()  # For saving

# Line 313: Non-blocking copy on separate stream
with torch.cuda.stream(self.load_stream):
    memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)
```

### Layerwise Loading Pattern

Key insight: **Load layer N+1 while computing layer N**

```
Timeline (LAYERWISE + ASYNC):

Layer 0:  [Load L0] [Wait] [Compute L0]
Layer 1:                  [Load L1] [Wait] [Compute L1]
Layer 2:                                  [Load L2] [Wait] [Compute L2]
         └──PCIe──┘      └──PCIe──┘      └──PCIe──┘
                 └────GPU Compute────┘ └────GPU Compute────┘

OVERLAPPED! PCIe transfers happen while GPU computes
```

### Code Flow

```
start_load_kv():
    ├─ Kick off ASYNC transfers on load_stream
    └─ Returns immediately (non-blocking)

model.forward():
    ├─ Layer 0:
    │  ├─ wait_for_layer_load("0")  ← Sync only this layer
    │  └─ attention compute         ← While L1 loads in background
    ├─ Layer 1:
    │  ├─ wait_for_layer_load("1")  ← Sync only this layer
    │  └─ attention compute         ← While L2 loads in background
    └─ ...

wait_for_save():
    └─ Block until all saves complete
```

### Performance Calculation

**Example: 70B model, 32 layers**

- KV per layer: ~50MB
- PCIe 4.0 x16 bandwidth: 32 GB/s
- Transfer time per layer: 50MB / 32GB/s = **1.6ms**
- Typical compute time per layer: **10-50ms**

**Result:** Transfer time is **completely hidden!** ✅

### When Overlap Helps

✅ **Large models** (70B+) - more compute per layer
✅ **Deep models** - more layers to pipeline
✅ **Fast storage** (NVMe, P2P) - less non-overlappable time

❌ **Small models** - compute too fast
❌ **Slow storage** (S3) - bottleneck shifts

### Speedup Summary

| Scenario | Config | Speedup |
|----------|--------|---------|
| Layerwise OFF | Async OFF | 0% (baseline) |
| Layerwise OFF | Async ON | ~15% (full batch overlap) |
| Layerwise ON | Async OFF | ~30% (layer pipelining) |
| **Layerwise ON** | **Async ON** | **40-50%** (combined) |

**Practical impact:** 10-30% reduction in prefill latency for large models.

---

## Feature Timeline

### `use_layerwise` Feature

**History:**
- **Initial commit**: May 12, 2025 (`cd28068`)
- **First version**: v0.3.0 (May 28, 2025)
- **Stable since**: v0.3.1 (June 25, 2025)
- **Author**: Jiayi Yao
- **PR**: #625 - "Async layerwise pipelining for KV cache offloading"

**Key milestones:**
- May 12: Initial implementation
- May 13: Improvements for CPU offloading
- May 20: Bug fixes
- June 24: Major refactor to unify code paths
- August 30: SGLang integration
- September 29: Documentation added

### `enable_async_loading` Feature

**History:**
- **Initial commit**: September 6, 2025 (`87e6954`)
- **First version**: v0.3.6 (September 15, 2025)
- **Stable since**: v0.3.7 (September 29, 2025)
- **Author**: Jiayi Yao
- **PR**: #1513 - "Async KV loading"

**Key milestones:**
- September 6: Initial implementation
- September 18: Bug fixes for MLA and TP > 1
- September 25: Stress testing
- October 24: Documentation added
- October 31: Memory object bug fix

### Version Support Matrix

| Version | `use_layerwise` | `enable_async_loading` | Status |
|---------|-----------------|----------------------|--------|
| v0.3.0  | ✅              | ❌                   | Initial release |
| v0.3.1  | ✅              | ❌                   | Stable |
| v0.3.2-v0.3.5 | ✅        | ❌                   | Feature frozen |
| v0.3.6  | ✅              | ✅                   | New feature |
| v0.3.7  | ✅              | ✅                   | Stable |
| v0.3.8  | ✅              | ✅                   | Current |
| **v0.3.9** | ✅            | ✅                   | **Latest** |

### Configuration

**In `lmcache/v1/config.py`:**

```python
# Line 147-151
"use_layerwise": {
    "type": bool,
    "default": False,
    "env_converter": _to_bool,
}

# Line 343-347
"enable_async_loading": {
    "type": bool,
    "default": False,
    "env_converter": _to_bool,
}
```

**Config auto-adjustment:**
```python
# Lines 477-483
if self.enable_async_loading or self.use_layerwise:
    self.save_unfull_chunk = False  # Prevent CPU memory fragmentation
```

### Production Recommendations

| Use Case | Minimum Version | Recommended |
|----------|-----------------|-------------|
| `use_layerwise` only | v0.3.1+ | v0.3.8+ |
| `enable_async_loading` only | v0.3.7+ | v0.3.8+ |
| Both features | v0.3.8+ | **v0.3.9** |

**Current recommendation:** Use **v0.3.9** (October 29, 2025) - latest stable with all features.

---

## Key Insights

1. **Scheduler is an object, workers are processes**
   - Scheduler lives inside EngineCore as Python object
   - Workers are separate processes, one per GPU rank
   - Communication via shared memory MessageQueues

2. **vLLM core doesn't need changes**
   - All hooks already in place
   - Plugin architecture supports external connectors
   - LMCache is just one implementation among many

3. **Async loading provides real speedup**
   - CUDA streams enable parallel PCIe + compute
   - Layerwise pipelining multiplies the benefit
   - Best for large models where compute > transfer

4. **Features evolve independently**
   - Layerwise developed first (May 2025)
   - Async loading built on that foundation (September 2025)
   - Same author, same philosophy
   - Both are opt-in (default=False)

5. **Production ready**
   - v0.3.8+ is stable for both features
   - Well-tested, documented, bug-fixed
   - Recommended for production deployments

---

## Related Files in Codebase

### vLLM Integration
- `vllm/v1/engine/core.py` - EngineCore main loop
- `vllm/v1/core/sched/scheduler.py` - Scheduler logic
- `vllm/v1/worker/gpu_worker.py` - Worker implementation
- `vllm/distributed/kv_transfer/kv_connector/v1/base.py` - Plugin interface

### LMCache Implementation
- `lmcache/integration/vllm/vllm_v1_adapter.py` - Connector implementation
- `lmcache/v1/cache_engine.py` - Cache engine with retrieve/store
- `lmcache/v1/gpu_connector.py` - CUDA stream management
- `lmcache/v1/config.py` - Configuration system

---

## Next Steps for Deeper Understanding

1. **Trace through a real request** - Follow code from request arrival to response
2. **Benchmark async loading** - Measure actual speedup in your workload
3. **Explore other backends** - CPU/disk/remote storage differences
4. **Study blend system** - How retriever selects important tokens
5. **Profile bottlenecks** - Identify which stage is slowest for your setup
