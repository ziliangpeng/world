# LMCache Deep Dive: Architecture and Design

## Overall Architecture

### System Overview

LMCache is a **multi-tier hierarchical KV cache management system** that sits between LLM serving engines (vLLM/SGLang) and storage backends. It reduces TTFT (Time To First Token) by 3-10x through intelligent caching, async I/O, and tight integration with serving engines.

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM/SGLang Integration                   │
│                  (Connector/Adapter Layer)                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                    LMCache Engine (v1)                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Cache Engine - Orchestrator                           │ │
│  │  - Request lifecycle management                         │ │
│  │  - Token Database (chunking/hashing)                    │ │
│  │  - GPU Connector (format conversion)                    │ │
│  └────────────────────────────────────────────────────────┘ │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  Storage Manager                             │
│  - Multi-backend coordination                                │
│  - Async task scheduling                                     │
│  - Weighted semaphore for concurrency control                │
└──────────────────┬──────────────────────────────────────────┘
                   │
       ┌───────────┼───────────┬──────────────────┐
       ▼           ▼           ▼                  ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐
│ LocalCPU    │ │ LocalDisk   │ │ Remote       │ │ P2P/PD       │
│ Backend     │ │ Backend     │ │ Backend      │ │ Backend      │
│             │ │             │ │              │ │              │
│ - Hot cache │ │ - GDS/Weka  │ │ - Redis      │ │ - Disagg     │
│ - LRU evict │ │ - Local SSD │ │ - Mooncake   │ │ - NIXL       │
│ - Pinned    │ │             │ │ - InfiniStore│ │              │
└─────────────┘ └─────────────┘ └──────────────┘ └──────────────┘
       │               │               │                  │
       └───────────────┴───────────────┴──────────────────┘
                       │
                       ▼
           ┌─────────────────────────┐
           │  Memory Allocator       │
           │  - NUMA-aware           │
           │  - Pinned memory        │
           │  - Ref counting         │
           └─────────────────────────┘
```

### Architectural Layers

#### Integration Layer (`lmcache/integration/`)
- **Purpose**: Bridges serving engines to LMCache
- **Key Components**:
  - `vllm_v1_adapter.py` - Implements vLLM's `KVConnectorBase_V1` interface
  - Request tracking per inference request
  - Scheduler/Worker role separation
- **Responsibilities**:
  - Translate vLLM scheduler decisions → LMCache operations
  - Manage GPU memory slot_mapping
  - Handle disaggregated prefill coordination

#### Engine Layer (`lmcache/v1/cache_engine.py`)
- **Purpose**: Core orchestrator
- **Key Components**:
  - `LMCacheEngine` - Main engine class
  - `TokenDatabase` - Token chunking and hashing
  - `GPUConnector` - GPU ↔ CPU data movement
- **Responsibilities**:
  - Convert tokens → cache keys (chunks of 256 tokens)
  - Coordinate store/retrieve operations
  - Manage async loading pipeline

#### Storage Layer (`lmcache/v1/storage_backend/`)
- **Purpose**: Multi-tier cache hierarchy
- **Key Components**:
  - `StorageManager` - Routes to appropriate backend
  - `LocalCPUBackend` - Hot in-memory cache (LRU eviction)
  - `LocalDiskBackend` - Persistent local storage
  - `RemoteBackend` - Network-attached cache
  - `P2PBackend` - Direct GPU-to-GPU transfer
- **Responsibilities**:
  - Check which tier has cached data
  - Async prefetch from slower tiers
  - Eviction when memory pressure

#### Memory Layer (`lmcache/v1/memory_management.py`)
- **Purpose**: CPU memory management
- **Key Components**:
  - `MemoryObj` - Abstraction for CPU/GPU memory
  - `MixedMemoryAllocator` - NUMA-aware allocator
  - Reference counting for safe reuse
- **Responsibilities**:
  - Allocate pinned CPU memory
  - Track memory usage and pressure
  - Prevent fragmentation

### Thread Model

- **Main thread**: Inference engine (vLLM/SGLang)
- **Storage Manager thread**: AsyncIO event loop for backend operations
- **Job Executor threads**: Disk I/O workers
- **Controller thread**: ZMQ message loop (if enabled)
- **Lookup Server thread**: Async lookup processing

---

## How LMCache Interfaces with vLLM

### Key Files
- `/Users/victor.peng/code/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py` (lines 1-1409)
- `/Users/victor.peng/code/LMCache/lmcache/integration/vllm/lmcache_connector_v1.py` (lines 1-180)
- `/Users/victor.peng/code/LMCache/lmcache/v1/gpu_connector.py` (lines 1-1495)

### Architecture Overview

**Entry Point**: vLLM calls LMCache through the `KVConnectorBase_V1` interface, which is implemented by `LMCacheConnectorV1Dynamic`.

**Key Components:**

1. **LMCacheConnectorV1Dynamic** (`lmcache_connector_v1.py`):
   - Thin wrapper that delegates to `LMCacheConnectorV1Impl`
   - Implements vLLM's `KVConnectorBase_V1` interface
   - Has two roles: SCHEDULER (for lookup/planning) and WORKER (for actual KV transfer)

2. **LMCacheConnectorV1Impl** (`vllm_v1_adapter.py`):
   - Main adapter class that bridges vLLM and LMCache
   - Maintains request tracking via `RequestTracker` objects (lines 113-240)
   - Manages lookup and prefetch via `LookupClient` and `LookupServer` (lines 598-626)

### Data Flow

#### Scheduler Side (lines 1125-1386)
1. **`get_num_new_matched_tokens()`** (lines 1126-1211): vLLM asks "how many tokens can we load from cache?"
   - Extracts token IDs from request
   - Applies multimodal hashes if present (lines 1152-1158)
   - Calls `lookup_client.lookup()` to check cache
   - Returns number of external hit tokens

2. **`update_state_after_alloc()`** (lines 1213-1276): After vLLM allocates blocks
   - Clears local lookup status
   - Tracks disaggregated prefill specs if present
   - Sets `can_load=True` in LoadSpec if allocation successful

3. **`build_connector_meta()`** (lines 1278-1386): Builds metadata for current step
   - Creates `LMCacheConnectorMetadata` containing list of `ReqMeta` objects
   - Each `ReqMeta` contains: token_ids, slot_mapping, load_spec, save_spec, request_configs
   - Handles both new and cached requests

#### Worker Side (lines 777-1119)

4. **`start_load_kv()`** (lines 778-889): Initiate KV loading
   - Initializes KV cache pointers from forward context
   - For each request with valid load_spec:
     - Creates token mask (marks vLLM-cached tokens as False)
     - Calls `lmcache_engine.retrieve()` or `retrieve_layer()` (layerwise mode)
     - GPU connector handles copying to vLLM's paged memory

5. **`wait_for_layer_load()`** (lines 891-912): Synchronization point
   - Used in layerwise pipelining
   - Waits for specific layer to be loaded before attention computation

6. **`save_kv_layer()` / `wait_for_save()`** (lines 915-1114): Save KV to cache
   - Extracts KV from vLLM's paged memory via slot_mapping
   - Applies chunking (only saves complete chunks)
   - Calls `lmcache_engine.store()` or `store_layer()`

### GPU Connectors (`gpu_connector.py`)

**Interface**: `GPUConnectorInterface` defines:
- `to_gpu()`: Load memory object into GPU
- `from_gpu()`: Save GPU KV to memory object
- `batched_to_gpu()` / `batched_from_gpu()`: Batch operations
- `get_shape()`: Calculate tensor shape for given token count

**Implementations:**

1. **VLLMPagedMemGPUConnectorV2** (lines 111-326):
   - For standard vLLM paged attention
   - Uses `lmc_ops.multi_layer_kv_transfer()` CUDA kernel
   - Supports MLA (Multi-head Latent Attention) format
   - Memory format: `KV_2LTD` or `KV_MLA_FMT`

2. **VLLMPagedMemLayerwiseGPUConnector** (lines 703-1012):
   - For layerwise caching without blending
   - Generator-based: yields control between layers
   - Memory format: `KV_T2D`

3. **VLLMBufferLayerwiseGPUConnector** (lines 328-701):
   - For layerwise caching WITH blending
   - Maintains GPU buffer mapping per layer
   - Handles RoPE (Rotary Position Embedding) re-application
   - Ping-pong buffering for efficiency
   - Memory format: `KV_2TD`

### Data Format Exchange

- vLLM uses "paged memory" with slot_mapping to locate KV
- LMCache uses contiguous tensors in various formats (KV_2LTD, KV_T2D, KV_2TD)
- GPU connectors translate between these via CUDA kernels in `lmc_ops`

---

## How Engine Interfaces with Storage Backend

### Key Files
- `/Users/victor.peng/code/LMCache/lmcache/v1/cache_engine.py` (lines 1-1426)
- `/Users/victor.peng/code/LMCache/lmcache/v1/storage_backend/storage_manager.py` (lines 1-765)
- `/Users/victor.peng/code/LMCache/lmcache/v1/storage_backend/abstract_backend.py` (lines 1-388)

### Architecture

**LMCacheEngine** is the main orchestrator that:
1. Manages token database (chunking/hashing)
2. Allocates CPU memory via storage_manager
3. Coordinates GPU↔CPU↔Storage transfers
4. Handles lookup, retrieve, store operations

**StorageManager** acts as the routing layer that:
1. Maintains ordered dict of storage backends
2. Routes requests to appropriate backend(s)
3. Manages memory allocation/deallocation
4. Handles cross-backend data movement

### Request Flow

#### Store Operation (`cache_engine.py` lines 181-306)
```
1. store() called with tokens/hashes + mask + kwargs (contains kvcaches, slot_mapping)
2. Process tokens through token_database.process_tokens()
   - Splits into chunks based on chunk_size
   - Generates CacheEngineKey for each chunk
3. For each chunk:
   - storage_manager.allocate() → gets MemoryObj
   - gpu_connector.batched_from_gpu() → GPU to MemoryObj
   - storage_manager.batched_put() → MemoryObj to backends
4. Backend-specific handling happens asynchronously
```

#### Retrieve Operation (`cache_engine.py` lines 424-526)
```
1. retrieve() called with tokens + mask + kwargs
2. _process_tokens_internal():
   - Process tokens → CacheEngineKeys
   - storage_manager.contains() checks which backend has the key
   - Builds block_mapping: {backend_name → [(key, start, end), ...]}
3. For each backend with hits:
   - storage_manager.batched_get() → MemoryObjs
   - Appends to reordered_chunks list
4. gpu_connector.batched_to_gpu() → MemoryObjs to GPU
5. Returns ret_mask indicating which tokens were retrieved
```

### StorageBackendInterface (`abstract_backend.py`)

Core methods:
- `contains(key, pin)`: Check if key exists, optionally pin it
- `batched_submit_put_task(keys, objs)`: Async put operation
- `get_blocking(key)`: Synchronous get
- `batched_get_non_blocking(lookup_id, keys)`: Async get (used in prefetch)
- `batched_async_contains(lookup_id, keys, pin)`: Async contains for lookup
- `pin(key)` / `unpin(key)`: Prevent/allow eviction
- `remove(key, force)`: Delete from backend

**AllocatorBackendInterface** extends this with:
- `allocate(shape, dtype, fmt, eviction, busy_loop)`: Get memory from pool
- `batched_allocate(...)`: Batch allocation
- `get_memory_allocator()`: Return underlying allocator

### Storage Manager Routing Logic (`storage_manager.py`)

#### Initialization (lines 180-234)
- Creates ordered dict of backends via `CreateStorageBackends()`
- Order matters: earlier backends checked first
- Typical order: LocalCPU → LocalDisk → Remote → P2P

#### Put Operation (lines 316-359)
```python
def batched_put(keys, memory_objs, transfer_spec, location):
    obj_dict = {}  # Maps backend class → (keys, objs)

    # First entry: allocator backend's objects
    obj_dict[allocator_backend_class] = (keys, memory_objs)

    for backend in storage_backends:
        if location specified and doesn't match: skip

        allocator = backend.get_allocator_backend()
        if allocator not in obj_dict:
            # Allocate + copy to this backend's memory
            new_keys, new_objs = allocate_and_copy_objects(...)
            obj_dict[allocator] = (new_keys, new_objs)

        backend.batched_submit_put_task(keys, objs)

    # Ref count down for all objects
```

#### Get Operation (lines 406-421)
```python
def batched_get(keys, location):
    for backend in storage_backends:
        if location specified and doesn't match: skip

        memory_objs = backend.batched_get_blocking(keys)
        if memory_objs: return memory_objs

    return None
```

#### Contains Operation (lines 595-627)
- Iterates through backends in order
- Returns first backend name that has the key
- Optionally pins the key to prevent eviction

### Async Loading Support (`storage_manager.py` lines 493-593)

For low-latency scenarios, LMCache supports async lookup+prefetch:
1. `async_lookup_and_prefetch()`: Called by scheduler
2. For each backend:
   - `batched_async_contains()` → returns num_hit_chunks
   - Launches `batched_get_non_blocking()` task
3. Tasks registered in EventManager
4. When all tasks complete, notifies async_lookup_server
5. Scheduler gets response via callback

---

## Deep Dive into Each Backend

### Local CPU Backend (`local_cpu_backend.py`)

**Purpose**: Fast in-memory cache using system RAM

**Key Components:**
- `hot_cache`: OrderedDict mapping CacheEngineKey → MemoryObj
- `memory_allocator`: MixedMemoryAllocator or PagedCpuGpuMemoryAllocator
- `cache_policy`: Eviction policy (LRU, FIFO, etc.)

**Implementation Details** (lines 36-657):

#### Memory Allocation (lines 365-453)
```python
def allocate(shape, dtype, fmt, eviction, busy_loop):
    # Try direct allocation
    memory_obj = memory_allocator.allocate(shape, dtype, fmt)
    if memory_obj or not eviction: return memory_obj

    # Eviction loop
    while True:
        if use_hot:
            evict_keys = cache_policy.get_evict_candidates(hot_cache, num=1)
            if evict_keys:
                batched_remove(evict_keys)
                continue

        if not busy_loop: break  # Give up
        time.sleep(0.1)  # Wait for other requests

        memory_obj = memory_allocator.allocate(...)
        if memory_obj: break
```

#### Contains (lines 101-109)
- Checks hot_cache dict
- If pin=True: pins memory object and adds to keys_in_request list
- Used for vLLM lookup

#### Put (lines 124-164)
- Synchronous operation
- Ref count up, insert into hot_cache
- Update cache_policy
- Send KVAdmitMsg to controller (if enabled)

#### Get (lines 166-192)
- Returns MemoryObj from hot_cache
- Ref count up to prevent premature eviction
- Async version for prefetch pipeline

**NUMA Support**: Detects NUMA topology and pins memory to specific nodes for better performance

### Local Disk Backend

**Purpose**: Overflow cache using local SSD/HDD

**Key Features**:
- Uses filesystem or dedicated block device
- Serialization/deserialization via serde layer
- Async I/O via job executor
- Eviction based on disk space limits

### Remote Backend (`remote_backend.py`)

**Purpose**: Network-attached cache (Redis, S3, etc.)

**Connectors**:
- `redis_connector.py`: Redis-based distributed cache
- `s3_connector.py`: S3/object storage backend
- `lm_connector.py`: Custom LMCache remote protocol
- `fs_connector.py`: Network filesystem (NFS, etc.)

**Common Pattern**:
1. Serialize MemoryObj to bytes
2. Send over network with metadata
3. Receiver deserializes back to MemoryObj
4. Use async I/O to avoid blocking

### P2P Backend (`p2p_backend.py`)

**Purpose**: Direct GPU-to-GPU transfer between nodes for disaggregated prefill

**Architecture:**
- Uses ZMQ for control plane (lookup requests)
- Uses transfer channels (NCCL, GPUDirect, etc.) for data plane
- Two-phase protocol: lookup then transfer

#### Lookup Flow (lines 171-212)
```python
async def batched_async_contains(lookup_id, keys, pin):
    # Tier 1: local lookup cache (TODO)
    # Tier 2: controller lookup
    msg = BatchedP2PLookupMsg(hashes=hashes)
    ret_msg = await lmcache_worker.async_put_and_wait_msg(msg)

    # Extract peer info: (worker_id, location, num_hits, peer_init_url)
    if num_hit_chunks > 0:
        await _ensure_peer_connection(peer_init_url)
        lookup_id_to_peer_mapping[lookup_id] = (peer_init_url, peer_lookup_url, location)

    return num_hit_chunks
```

#### Transfer Flow (lines 231-275)
```python
async def _handle_peer_requests():
    while running:
        msg = await async_peer_socket.recv()  # ZMQ REP socket

        if BatchedLookupAndGetMsg:
            # Peer wants to read from us
            num_hit_chunks = await local_cpu_backend.batched_async_contains(...)
            mem_objs = await local_cpu_backend.batched_get_non_blocking(...)
            await transfer_channel.async_batched_write(mem_objs, transfer_spec)
            return BatchedLookupAndGetRetMsg(num_hit_chunks)

        elif BatchedLookupAndPutMsg:
            # Peer wants to write to us (receiver side)
            allocate local memory
            await transfer_channel.async_batched_read(...)
            local_cpu_backend.batched_submit_put_task(...)
```

**Transfer Channels**:
- NCCL: GPU-to-GPU via NCCL primitives
- GPUDirect RDMA: Direct GPU memory access over InfiniBand
- TCP: Fallback for CPU buffers

### GDS Backend

**Purpose**: GPU Direct Storage - direct GPU to NVMe transfers bypassing CPU

**Key Feature**: Uses cuFile API for zero-copy I/O

### Hybrid Backend

**Purpose**: Combines multiple backends with tiering logic

**Strategy**: Hot data in CPU, warm in disk, cold in remote

---

## How Blend Works

### Key Files
- `/Users/victor.peng/code/LMCache/lmcache/blend/interfaces.py` (lines 1-146)
- `/Users/victor.peng/code/LMCache/lmcache/blend/retriever.py` (lines 1-250)
- `/Users/victor.peng/code/LMCache/lmcache/v1/compute/blend/blender.py` (lines 1-168)
- `/Users/victor.peng/code/LMCache/lmcache/v1/compute/blend/utils.py` (lines 1-64)

### Blend Concept

**Problem**: In RAG scenarios, you have:
- Long context (retrieved documents): K_long, V_long
- Short query: Q_short

Traditional approach computes attention over full K_long × Q_short which is expensive.

**Blend Solution**:
1. Cache K_long, V_long from document processing
2. For query, compute fresh Q_short, K_short, V_short
3. Identify which Q tokens actually need full context (importance-based selection)
4. Blend: Keep only important Q tokens, concatenate with cached K_long, V_long
5. Compute attention only on selected tokens

### Architecture

**Two Main Interfaces:**

1. **BlendRetriever** (`interfaces.py` lines 74-99):
   ```python
   class BlendRetriever:
       def new_request(full_prompts, indices) -> BlendRetrieverTask:
           """Launch retrieval of cached KV for document segments"""

   class BlendRetrieverTask:
       def result(layer_id) -> BlendRetrieverResult:
           """Get K, V, valid_mask, original_positions for one layer"""
   ```

2. **BlendExecutor** (`interfaces.py` lines 101-145):
   ```python
   class BlendExecutor:
       def blend(layer_id, retrieved_k, retrieved_v, valid_mask,
                 original_positions, fresh_q, fresh_k, fresh_v,
                 positions, query_start_loc, token_dim) -> BlendOutput:
           """Blend cached and fresh KV, select important Q tokens"""
   ```

### SPTBlendRetriever Implementation (`retriever.py`)

**Purpose**: Retrieves cached KV using Special Token (SPT) as delimiter

**Key Insight**: Documents end with special token, so:
- Input: [doc1_tokens, SPT, doc2_tokens, SPT, query_tokens]
- Split on SPT → [doc1, doc2, query]
- Retrieve KV for [doc1, doc2] independently

**Implementation** (lines 180-249):
```python
class SPTBlendRetriever:
    def new_request(full_prompts, indices):
        # Split on indices (SPT positions)
        splitted_tokens = []
        for prompt in full_prompts:
            splitted_tokens.extend(torch.tensor_split(prompt, indices))

        # Launch parallel retrieval
        tasks = [executor.submit(cache_engine.retrieve, tokens)
                 for tokens in splitted_tokens]

        return SPTBlendRetrieverTask(token_segments, tasks, fmt)
```

**SPTBlendRetrieverTask** (lines 22-177):
- Waits for all retrieval tasks
- Concatenates K, V tensors across segments
- Creates valid_mask: 1 where KV was retrieved, 0 otherwise
- Returns per-layer results

### LMCBlender Implementation (`blender.py`)

#### Initialization (lines 24-57)
```python
def __init__(cache_engine, gpu_connector, vllm_model, config):
    self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)
    self.common_metadata = LMCBlendCommonMetadata(
        check_layers=config.blend_check_layers,  # Which layers to check for importance
        recomp_ratios=config.blend_recompute_ratios,  # % of Q tokens to keep
        thresholds=config.blend_thresholds,  # Similarity thresholds
    )
```

#### Blending Process (lines 59-119)
```python
def process_qkv(q, k, v, residual, layer_id, attn_output, attn_metadata):
    # Get cached K, V for this layer
    old_k, old_v = gpu_connector.get_kv(layer_id)

    # Apply RoPE to fresh Q, K
    q, k = attn_layer.rotary_emb(positions, q, k)

    # If this is a check layer: select important tokens
    if layer_id in check_layers:
        # Compute difference: ||K_fresh - K_cached||^2 per token
        diff_k = torch.sum((k - old_k) ** 2, dim=[1])

        # Select top-k tokens by difference (high difference = important)
        topk_num = int(len(diff_k) * recomp_ratio)
        top_indices = torch.topk(diff_k, k=topk_num).indices
        top_indices = torch.sort(top_indices)[0]

        # Keep only selected tokens
        q, k, v = q[top_indices], k[top_indices], v[top_indices]
        residual = residual[top_indices]

        # Update metadata
        self.metadata.imp_indices = top_indices
        self.metadata.positions = positions[top_indices]
        attn_metadata.update_from_top_indices(top_indices)

    # Update cached K, V with fresh values for selected tokens
    if imp_indices is not None:
        old_k[imp_indices] = k
        old_v[imp_indices] = v
        return q, old_k, old_v, ...  # Use blended K, V
    else:
        return q, k, v, ...  # First layer: use fresh K, V
```

#### Layerwise Orchestration (lines 123-168)
```python
def blend_layer(tokens, mask, **kwargs):
    layerwise_model_executor = layerwise_model.compute_layer(tokens)
    layerwise_retriever = cache_engine.retrieve_layer(tokens, mask, **kwargs)

    next(layerwise_retriever)  # Init retriever
    yield  # First yield

    for i in range(num_layers):
        next(layerwise_retriever)  # Load layer i KV into gpu_connector buffer
        next(layerwise_model_executor)  # Compute layer i with blending
        yield  # Yield between layers

    next(layerwise_retriever)  # Finalize retrieval
    metadata.clean()
    yield  # Final yield
```

### V1 Blend System (`v1/compute/blend/`)

**LMCBlenderBuilder** (`utils.py`):
- Singleton builder for blender instances
- Creates `LMCBlender` with cache_engine, gpu_connector, vllm_model

**Integration Points:**
1. GPU Connector provides `get_kv(layer_id)` to access cached KV
2. Cache Engine provides `retrieve_layer()` for layerwise retrieval
3. vLLM model provides `compute_layer()` for layerwise forward
4. Attention metadata tracks token selection across layers

### Configuration Options

- `enable_blending`: Enable blend mode
- `blend_check_layers`: List of layer IDs to check for importance (e.g., [0, 4, 8])
- `blend_recompute_ratios`: Percentage of Q tokens to keep per check layer
- `blend_thresholds`: Similarity thresholds for token selection
- `use_layerwise`: Must be True for blending to work

### Blend Data Flow

```
1. User Request: [doc1, SPT, doc2, SPT, query]
2. SPTBlendRetriever: Split and retrieve cached KV for [doc1, doc2]
3. vLLM Forward: Compute fresh Q, K, V for full input
4. Layer 0 (check layer):
   - Compute diff = ||K_fresh - K_cached||²
   - Select top-k important Q tokens
   - Update cached K, V at selected positions
5. Layers 1-N:
   - Use selected Q tokens only
   - Use blended K, V (mostly cached + updates)
6. Final Attention:
   - Short Q (selected tokens only)
   - Long K, V (full context with updates)
   - Dramatically reduced FLOPs
```

### Benefits

- **Speedup**: 3-10x faster for RAG workloads with long documents
- **Accuracy**: Maintains near-perfect accuracy by selecting important tokens
- **Memory**: Reuses cached KV, only stores fresh KV for selected tokens

---

## Additional Architecture Details

### Key Abstractions and Interfaces

#### Storage Layer
- **StorageBackendInterface** - Base storage interface with contains/get/put
- **AllocatorBackendInterface** - Memory allocation from backends

#### Compute Layer
- **GPUConnectorInterface** - GPU ↔ CPU data movement
- Engine-specific implementations for vLLM and SGLang

#### Memory Layer
- **MemoryObj** - Abstraction for CPU/GPU memory with ref counting
- **MemoryAllocatorInterface** - Memory allocation and pooling
- **MemoryFormat** enum - Different KV tensor layouts

#### Index Layer
- **TokenDatabase** - Token → Cache key mapping
- ChunkedTokenDatabase - Fixed-size chunking
- SegmentTokenDatabase - Separator-based segmentation

### Design Patterns

1. **Factory Pattern** - Dynamic backend creation via `CreateStorageBackends()`
2. **Singleton Pattern** - One engine instance per worker via `LMCacheEngineBuilder`
3. **Adapter Pattern** - Integration with different serving engines
4. **Strategy Pattern** - Pluggable cache policies (LRU, FIFO, etc.)
5. **Observer Pattern** - Event-driven async operations
6. **Decorator Pattern** - Audit wrapper for debugging

### Configuration and Initialization

#### Config Hierarchy
1. From defaults
2. From file (YAML)
3. From environment variables
4. From dictionary (programmatic)
5. Environment variables overlay

#### Key Config Options
- Storage: `local_cpu`, `local_disk`, `remote_url`
- Features: `use_layerwise`, `enable_blending`, `enable_p2p`
- Performance: `chunk_size` (256 default), `cache_policy`
- Controller: `enable_controller`, controller ZMQ endpoints

### Memory Formats

Different GPU memory layouts for different use cases:
- `KV_2LTD`: `[2, layers, tokens, hidden]` - standard vLLM
- `KV_T2D`: `[tokens, 2, hidden]` - layerwise
- `KV_2TD`: `[2, tokens, hidden]` - blending
- `KV_MLA_FMT`: Multi-head latent attention format
- `BINARY`: Compressed format for remote storage

### Legacy vs V1 Architecture

| Feature | Legacy (`lmcache/`) | V1 (`lmcache/v1/`) |
|---------|---------------------|-------------------|
| **I/O Model** | Synchronous | Async (event loop) |
| **Memory Management** | Basic pool | Ref counting + pinning + NUMA |
| **Backends** | Local, Remote | + P2P, GDS, Plugins |
| **Disaggregation** | ❌ | ✅ P2P + PD backends |
| **Controller** | ❌ | ✅ ZMQ-based control plane |
| **Blending** | Basic | Advanced with layerwise |
| **Prefetch** | ❌ | ✅ Background loading |
| **Status** | Backward compat | Production |

**Why V1?** Production needs required async I/O, better memory management, disaggregation support, and centralized control.

---

## Summary

LMCache achieves **3-10x TTFT reduction** through:

1. **Hierarchical caching**: Hot data in CPU RAM, warm on disk, cold in remote
2. **Async operations**: Non-blocking I/O allows prefetch while computing
3. **Intelligent chunking**: 256-token chunks with prefix hashing for reuse
4. **Disaggregation**: P2P backend enables prefill/decode separation
5. **Blending**: RAG optimization via importance-based token selection
6. **Tight integration**: Deep hooks into vLLM scheduler and memory system

The V1 architecture is production-ready with enterprise features like centralized control, plugin system, and extensive observability.
