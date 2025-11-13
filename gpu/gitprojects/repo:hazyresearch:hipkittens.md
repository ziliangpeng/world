# Research Report: HipKittens - High-Performance AI Kernels for AMD GPUs

## 1. Introduction

HipKittens is an open-source project from HazyResearch at Stanford University that provides a set of minimal, opinionated C++ primitives for writing high-performance AI kernels specifically for AMD GPUs. The project's primary motivation is to foster a "multi-silicon" future for AI, where software is not locked into a single hardware vendor's ecosystem. It aims to abstract hardware-specific details while exposing the necessary primitives for achieving near-metal performance. HipKittens builds upon the concepts of a previous project, ThunderKittens, and adapts them to the architecture of AMD's CDNA3 and CDNA4 GPUs.

### 1.1 What HipKittens Is (and Isn't)

HipKittens is **more than just a wrapper** around HIP/ROCm APIs - it's a high-performance kernel library that:

*   **Wraps low-level APIs**: Provides cleaner abstractions over raw HIP calls
*   **Provides opinionated primitives**: Pre-configured tensor core operations and optimized memory layouts
*   **Embeds hand-optimized code**: Direct assembly instructions for critical performance paths
*   **Codifies proven patterns**: Implements scheduling strategies like "8-wave ping pong" and "4-wave interleave" that aren't provided by HIP itself

The relationship can be visualized as:
```
Your Kernel Code
       ↓
   HipKittens (tile primitives, optimized patterns)
       ↓
     HIP API (low-level GPU programming interface)
       ↓
    ROCm Stack (AMD's GPU software platform)
       ↓
   AMD GPU Hardware
```

**Important**: HipKittens is AMD-specific and does not provide cross-vendor portability. Its sibling project, ThunderKittens, serves the same role for NVIDIA/CUDA GPUs. The two projects share conceptual patterns and design philosophy but require separate implementations.

## 2. Core Primitives and Concepts

HipKittens is designed from the "hardware up," meaning its design principles are dictated by the underlying silicon. The core of the library revolves around a few key concepts:

*   **Tile Primitives:** The fundamental unit of computation in HipKittens is the "tile," which is sized according to the tensor core units of the AMD GPUs. Memory operations on tiles are designed to be coalesced and bank-conflict-free, and they eagerly utilize tensor core layouts. The project emphasizes minimizing the cost of address computation.

*   **Python-Inspired Functions:** To provide a user-friendly interface, HipKittens wraps low-level HIP (Heterogeneous-compute Interface for Portability) and assembly code in lightweight, Python-inspired functions. These functions operate on tiles, enabling developers to express complex computations in a more intuitive way.

*   **Asynchronous Memory Operations:** To hide memory latency, HipKittens heavily utilizes asynchronous loads and stores. This is achieved by using direct buffer loads to shared memory, allowing the GPU to overlap memory transfers with computation.

*   **Scheduling and Overlapping:** The project identifies and implements two core patterns for overlapping computation and memory operations: "8-wave ping pong" and "4-wave interleave." These scheduling patterns are reused across different kernels to maximize GPU utilization.

## 3. Architecture and Implementation

The HipKittens repository is structured to separate the core library, kernel implementations, tests, and analysis tools.

*   **`include/`:** This directory contains the C++ header files that define the core primitives of the HipKittens library. These primitives provide the building blocks for creating custom kernels.

*   **`kernels/`:** This directory houses the implementations of several key AI kernels, demonstrating the use of the HipKittens primitives.

*   **`tests/`:** The project includes a suite of unit tests to ensure the correctness of the library functions.

*   **`analysis/`:** To validate the performance of the kernels, the repository provides scripts for benchmarking and plotting results.

*   **`training/`:** This directory contains scripts for training well-known models like BERT and Llama using the kernels implemented with HipKittens.

The project is intended to be used within a specific Docker environment based on ROCm (Radeon Open Compute platform), ensuring that all dependencies are correctly managed.

## 4. Key Kernels and Applications

HipKittens provides implementations for a variety of essential AI kernels, including:

*   **GEMM (General Matrix Multiplication):** A highly optimized BF16 GEMM kernel is provided, which is a cornerstone of many deep learning models.

*   **Attention:** The repository includes kernels for various attention mechanisms, such as Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and both causal and non-causal variants. Both forward and backward passes are implemented.

*   **Memory-Bound Kernels:** Kernels for memory-bound operations like Rotary Positional Embedding (RoPE) and Layer Normalization are also included.

These kernels are not just proof-of-concepts; they are integrated into training scripts for models like BERT and Llama, demonstrating their practical utility. The project also provides comparisons against other implementations, such as AITensor and PyTorch.

## 5. Benchmarking and Analysis

A significant part of the HipKittens project is its emphasis on performance analysis. The `analysis/` directory contains scripts to benchmark the provided kernels across a range of dimensions and settings. This allows researchers and developers to reproduce the performance results presented in the project's associated paper and to evaluate the effectiveness of the HipKittens primitives.

## 6. Scheduling Patterns and Latency Hiding

One of HipKittens' most significant contributions is the discovery and codification of scheduling patterns specifically optimized for AMD CDNA3 and CDNA4 GPUs. These patterns—"8-wave ping-pong" and "4-wave interleave"—represent a fundamental rethinking of how to achieve peak performance on AMD hardware, diverging from NVIDIA-centric optimization strategies.

### 6.1 The Scheduling Challenge on AMD GPUs

GPU kernel performance is fundamentally limited by the ability to hide memory latency behind computation. Understanding the latency characteristics is crucial:

*   **HBM (High Bandwidth Memory) loads**: ~300+ cycles
*   **Matrix Multiply-Accumulate (MMA) operations**: ~8 wavefront cycles to complete
*   **LDS (Local Data Store / Shared Memory) loads**: ~10 cycles

The challenge is to keep tensor cores busy while waiting for data. On NVIDIA GPUs, the dominant approach is **warp specialization**:

*   **Producer warps**: Dedicated to fetching data from HBM to shared memory
*   **Consumer warps**: Dedicated to performing computation (MMA operations)
*   **Key advantage on NVIDIA**: Dynamic register allocation means producer warps don't waste resources

However, on AMD CDNA3/4 GPUs with **static register allocation**, this approach fails:

*   Producer waves must allocate the same number of registers as consumer waves
*   Producer waves hold registers without contributing to computation
*   This limits the output tile size and reduces occupancy
*   **Result**: Wave specialization achieves only ~80% of peak BF16 GEMM performance on MI355X

HipKittens discovered that AMD GPUs require fundamentally different scheduling patterns to achieve peak performance.

### 6.2 8-Wave Ping-Pong Pattern

The **8-wave ping-pong** pattern is HipKittens' primary scheduling strategy, achieving performance that matches or exceeds AMD's hand-written assembly kernels.

#### How It Works

The pattern assigns **2 waves per SIMD unit**:

1. **Wave A**: Executes a cluster of memory instructions (loading from HBM to shared memory)
2. **Wave B**: Executes a cluster of compute instructions (MMA operations on register tiles)
3. At the end of each cluster, the waves swap roles

The name "8-wave" comes from the fact that MMA operations take approximately **8 wavefront cycles** to complete, which aligns perfectly with the time needed to fetch the next data tile from HBM.

#### Double-Buffered Structure

The implementation uses **ping-pong buffers** in shared memory:

```cpp
// Allocate two sets of shared memory buffers
ST_A (&As)[2][2] = al.allocate<ST_A, 2, 2>();  // [tic/toc][tile_idx]
ST_B (&Bs)[2][2] = al.allocate<ST_B, 2, 2>();

int tic = 0, toc = 1;  // Ping-pong indices
```

#### Step-by-Step Execution Flow

1. **Prefetch Phase**: Load `buffers[tic]` from HBM to shared memory
   ```cpp
   G::load(As[tic][0], g.a, {batch, row, col, k_iter});
   G::load(Bs[tic][0], g.b, {batch, row, col, k_iter});
   ```

2. **Wait for Shared Memory**: Ensure data is ready in LDS
   ```cpp
   asm volatile("s_waitcnt vmcnt(4)");  // Wait for HBM load
   __builtin_amdgcn_s_barrier();        // Synchronize workgroup
   ```

3. **Start Prefetch for Next Iteration**: Begin loading `buffers[toc]`
   ```cpp
   G::load(As[toc][0], g.a, {batch, row, col, k_iter + 1});
   ```

4. **Load to Registers**: Transfer from shared memory to register tiles
   ```cpp
   auto st_subtile = subtile_inplace<REG_M, REG_K>(As[tic][0], {warp_row, 0});
   load(A_tile, st_subtile);
   ```

5. **Compute**: Perform MMA while HBM load for next iteration is in flight
   ```cpp
   asm volatile("s_waitcnt lgkmcnt(0)");  // Wait for LDS load
   __builtin_amdgcn_s_setprio(1);         // Set HIGH priority for compute
   mma_ABt(C_accum, A_tile, B_tile, C_accum);
   __builtin_amdgcn_s_setprio(0);         // Reset to NORMAL priority
   ```

6. **Swap Buffers**: Exchange `tic` and `toc` for next iteration
   ```cpp
   tic ^= 1;
   toc ^= 1;
   ```

#### Code Example from Actual GEMM Kernel

From `kernels/gemm/bf16fp32/mi350x/256_256_64_32_with16x32.cpp`:

```cpp
#pragma unroll
for (int tile = 0; tile < num_tiles - 2; tile += 2) {
    // Load from shared to registers (using tic buffer)
    auto st_subtile_b = subtile_inplace<HALF_REG_BLOCK_N, K_STEP>(Bs[0][0], {warp_col, 0});
    load(B_tile_0, st_subtile_b);

    auto st_subtile_a = subtile_inplace<HALF_REG_BLOCK_M, K_STEP>(As[0][0], {warp_row, 0});
    load(A_tile, st_subtile_a);

    // Prefetch next iteration to toc buffer
    G::load(As[1][1], g.a, {0, 0, row*2 + 1, tile + 1}, ...);
    asm volatile("s_waitcnt lgkmcnt(8)");
    __builtin_amdgcn_s_barrier();

    // Compute on current data (tic buffer)
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_setprio(1);
    mma_ABt(C_accum[0][0], A_tile, B_tile_0, C_accum[0][0]);
    __builtin_amdgcn_s_setprio(0);
    __builtin_amdgcn_s_barrier();

    // Continue with next sub-tiles while prefetch continues...
}
```

#### Performance Characteristics

*   **BF16 GEMM**: Matches AMD's hand-optimized assembly kernels
*   **FP8 GEMM**: Competitive with vendor implementations
*   **Attention Forward**: State-of-the-art performance
*   **Code size**: Compact—large tile primitives lead to readable kernels

#### When to Use

The 8-wave ping-pong pattern is the **default choice** for most kernels:
*   Standard GEMM operations
*   Attention forward passes
*   Any kernel where code maintainability matters
*   When 95%+ peak performance is sufficient

### 6.3 4-Wave Interleave Pattern

The **4-wave interleave** pattern provides an alternative for kernels where 8-wave ping-pong leaves performance on the table, at the cost of significantly more complex code.

#### How It Works

The pattern assigns **1 wave per SIMD unit**, with threads in this wave finely switching between issuing memory and compute operations:

*   Instead of clustering operations, instructions are interleaved at a much finer granularity
*   Each iteration processes 4 sub-iterations (one per wave in a wave-group)
*   Memory operations for iteration N+1 are hidden between compute operations for iteration N

#### Execution Pattern

From `kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu`:

```cpp
__device__ inline static void do_interleaved_cluster(...) {
    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 0, 0);  // MMA for wave 0, sub-tile 0
    __builtin_amdgcn_sched_barrier(0);

    // Compute addresses while waiting
    precomputed_addresses addresses = precompute_addresses(dst_gl, src_gl, idx);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 1, 0);  // MMA for wave 0, sub-tile 1
    __builtin_amdgcn_sched_barrier(0);

    // Pre-compute shared memory offsets
    uint32_t swizzled_offsets[2];
    prefill_swizzled_offsets<2>(dst, src, swizzled_offsets);

    // Start HBM and LDS loads
    load_one<0>(dst_gl, src_gl, addresses);          // HBM → shared
    load_one<0, 0, 0>(dst, src, swizzled_offsets);   // Shared → registers

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 2, 0);  // MMA for wave 0, sub-tile 2
    __builtin_amdgcn_sched_barrier(0);

    load_one<0, 0, 1>(dst, src, swizzled_offsets);   // Another LDS load

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 3, 0);  // MMA for wave 0, sub-tile 3
    __builtin_amdgcn_sched_barrier(0);

    // Continue interleaving for remaining waves...
    load_one<1>(dst_gl, src_gl, addresses);
    load_one<1, 0, 0>(dst, src, swizzled_offsets);

    __builtin_amdgcn_sched_barrier(0);
    mma_ABt_one(c, a, b, c, 1, 0, 0);  // Wave 1 begins
    mma_ABt_one(c, a, b, c, 1, 1, 0);
    __builtin_amdgcn_sched_barrier(0);
    // ... pattern continues
}
```

#### Key Characteristics

The interleaving keeps all execution units busy:

1. **Cycle 0-1**: MMA_0 executes (waves 0-1 of warp-group)
2. **Cycle 2**: Address computation for next load
3. **Cycle 3-4**: MMA_1 executes (waves 2-3)
4. **Cycle 5**: HBM load initiated (asynchronous)
5. **Cycle 6-7**: MMA_2 executes
6. **Cycle 8**: Shared memory load
7. **Cycle 9-10**: MMA_3 executes
8. **Cycle 11+**: HBM load completes, cycle repeats

#### Trade-offs

**Advantages**:
*   **Higher performance on complex kernels**: GQA non-causal attention backward, fused operations
*   **Better functional unit utilization**: Keeps MMA, VALU, and memory units all busy
*   **Hides more latency**: Finer-grained overlapping finds more opportunities

**Disadvantages**:
*   **Code expansion**: 3-5x more code than 8-wave ping-pong
*   **Harder to write and maintain**: Manual instruction scheduling is error-prone
*   **Requires small tile primitives**: Must break operations into fine-grained pieces

#### When to Use

Use 4-wave interleave when:
*   8-wave ping-pong doesn't achieve target performance
*   Kernel complexity justifies the development cost
*   Squeezing the last 5-10% performance is critical
*   Example: GQA causal backward attention

### 6.4 Implementation: Manual Scheduling, Not Automated

A critical understanding: HipKittens **does not include a software scheduler**. The scheduling patterns are **manually coded** by expert kernel developers, with hints provided to AMD's hardware scheduler.

#### AMD Intrinsics for Scheduling Control

**Scheduling barriers** (provide hints to hardware scheduler):
```cpp
__builtin_amdgcn_sched_barrier(0);  // General scheduling barrier
```

**Grouped scheduling barriers** (schedule specific instruction types together):
```cpp
// From schedule_utils.cpp
constexpr int VALU = 1 << 1;      // Vector ALU operations
constexpr int SALU = 1 << 2;      // Scalar ALU operations
constexpr int MFMA = 1 << 3;      // Matrix FMA operations
constexpr int VMEM_READ = 1 << 5; // Vector memory reads
constexpr int DS_READ = 1 << 8;   // LDS reads
constexpr int DS_WRITE = 1 << 9;  // LDS writes

#define schedule_group_barrier(mask, size, sync_id) \
    __builtin_amdgcn_sched_group_barrier(mask, size, sync_id)
```

**Priority control** (favor compute over memory during critical sections):
```cpp
__builtin_amdgcn_s_setprio(1);  // HIGH priority - favor these instructions
mma_ABt(C_accum, A, B, C);      // Matrix multiply
__builtin_amdgcn_s_setprio(0);  // NORMAL priority - reset
```

**Memory operation counters** (precise control over when data is ready):
```cpp
// vmcnt: Vector memory operations count (HBM loads/stores)
asm volatile("s_waitcnt vmcnt(4)");  // Wait until ≤4 HBM ops remain

// lgkmcnt: LDS/GDS/scalar memory operations count
asm volatile("s_waitcnt lgkmcnt(0)"); // Wait for all LDS ops to complete
```

#### Helper Functions for Barrier Placement

From `kernels/attn/gqa/kernel_d64.cpp`:

```cpp
// Emit pairs of MFMA + VALU scheduling barriers
template<int Pairs, int VALU_CNT, int Group>
__device__ __forceinline__ void sched_barrier_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);           // 1 MFMA instruction
    SCHED_BARRIER(VALU_MASK, VALU_CNT, Group);    // VALU_CNT vector ALU ops
    if constexpr (Pairs > 1)
        sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
}

// Usage in kernel:
sched_barrier_pairs<10, 5, 1>();  // Schedule 10 pairs of (1 MFMA + 5 VALU)
```

#### Schedule Utility Functions

From `kernels/gemm/bf16fp32/mi350x/micros/hint_based/schedule_utils.cpp`:

```cpp
// Calculate how many MFMA instructions in a compute cluster
template<int REG_BLOCK_M, int REG_BLOCK_N, int DOT_SLICE>
constexpr int cluster_mfma_count() {
    constexpr int D_HEIGHT = REG_BLOCK_M / 32;  // Number of 32x32 tiles in M
    constexpr int D_WIDTH = REG_BLOCK_N / 32;   // Number of 32x32 tiles in N
    constexpr int A_WIDTH = DOT_SLICE / 16;     // Number of K-dimension tiles
    return D_HEIGHT * D_WIDTH * A_WIDTH;
}

// Calculate how many LDS read operations needed
template<int RT_HEIGHT, int RT_WIDTH>
constexpr int compute_ds_reads() {
    constexpr int k_iterations = 2;
    return k_iterations * RT_WIDTH * RT_HEIGHT;
}
```

These utilities help developers place the correct number of barriers for the hardware scheduler.

#### What the Hardware Scheduler Actually Does

AMD's GPU hardware scheduler makes real-time decisions about:
*   Which instructions to issue each cycle
*   Which wavefronts to execute when multiple are ready
*   How to handle cache misses and memory stalls
*   Resource allocation across competing work

HipKittens' manual patterns **guide** the scheduler by:
*   Organizing work into efficient clusters
*   Indicating priority for compute-heavy sections
*   Ensuring data dependencies are satisfied
*   Pre-computing addresses to reduce scheduler overhead

### 6.5 Industry Context vs HipKittens Innovation

HipKittens builds upon established GPU optimization concepts while introducing AMD-specific innovations.

#### Pre-Existing Industry Techniques

**Double Buffering / Ping-Pong Buffers** (NVIDIA CUTLASS):
*   Allocate two buffers in shared memory
*   While one buffer provides data for computation, the other is filled from HBM
*   Swap buffers when computation completes
*   General concept used across GPU vendors

**Multi-Stage Software Pipelining** (NVIDIA Hopper):
*   CUTLASS uses 3-7 stage pipelines on modern architectures
*   More stages = more opportunities for scheduler to hide latency
*   "Asynchronous pipeline" with TMA (Tensor Memory Accelerator) on H100
*   Automatically managed by NVIDIA's compiler in many cases

**Warp Specialization** (NVIDIA Hopper, Blackwell):
*   **Producer warps**: Dedicated to data movement (HBM → shared memory)
*   **Consumer warps**: Dedicated to computation (MMA operations)
*   Warps run concurrently, maximizing tensor core utilization
*   Leverages NVIDIA's **dynamic register allocation**

**Instruction Interleaving** (General GPU optimization):
*   Hardware schedulers interleave instructions from different threads/warps
*   Hides latency by switching to ready work when stalls occur
*   Manual interleaving can outperform hardware heuristics for specialized kernels

#### HipKittens Innovations

| Technique | Industry Standard | HipKittens Innovation | Target Architecture |
|-----------|------------------|----------------------|---------------------|
| Double buffering | ✅ CUTLASS, common practice | Extended to AMD-specific patterns | Both NVIDIA and AMD |
| Multi-stage pipelining | ✅ CUTLASS (3-7 stages) | - | NVIDIA-optimized |
| Warp specialization | ✅ NVIDIA Hopper/Blackwell | ❌ Underperforms on AMD (80% peak) | NVIDIA-optimized |
| **8-wave ping-pong** | ❌ New named pattern | ✅ AMD CDNA3/4-specific | **AMD CDNA3/4** |
| **4-wave interleave** | ❌ New named pattern | ✅ Fine-grained AMD optimization | **AMD CDNA3/4** |
| Manual scheduling hints | Partial (inline assembly) | ✅ Systematic use of AMD intrinsics | AMD-specific |

#### Key Discoveries

1. **Wave specialization fails on AMD**: On MI355X, achieves only 80% of peak BF16 GEMM performance
2. **Static register allocation is the culprit**: Producer waves hold registers without computing
3. **8-wave ping-pong achieves parity**: Matches AMD's hand-written assembly kernels
4. **4-wave interleave for complex kernels**: Provides additional 5-10% for GQA backward
5. **Named, reusable patterns**: Unlike ad-hoc optimizations, HipKittens codifies best practices

#### Why This Matters

Before HipKittens:
*   AMD kernel developers either used vendor libraries (limited flexibility) or wrote assembly (extremely difficult)
*   NVIDIA-style optimizations (like wave specialization) were blindly applied to AMD, yielding poor results
*   No systematic understanding of optimal scheduling patterns for AMD CDNA architectures

After HipKittens:
*   Clear, named patterns that researchers can understand and apply
*   Performance competitive with AMD's assembly without the complexity
*   Educational resource explaining **why** AMD requires different approaches
*   Reproducible patterns that work across different AMD CDNA-based GPUs (MI300, MI355, etc.)

### 6.6 AMD vs NVIDIA Architectural Differences

Understanding why HipKittens' patterns differ from NVIDIA best practices requires understanding the architectural divergence between vendors.

#### NVIDIA Hopper Architecture (H100)

**Register Allocation**:
*   **Dynamic**: Register allocator can assign different amounts to different warps
*   Producer warps can use minimal registers
*   Consumer warps can use maximum registers for large accumulator tiles
*   Result: Wave specialization is highly efficient

**Memory Hierarchy**:
*   Tensor Memory Accelerator (TMA): Hardware-accelerated async copy
*   L2 cache: 50-60 MB, shared across all SMs
*   Shared memory: Up to 228 KB per SM

**Scheduling Features**:
*   Cluster Launch Control (CLC): Pipeline depth of 3 for overlapping operations
*   Thread Block Clusters: Coordinate work across multiple SMs
*   Distributed shared memory: Access neighbor SM's shared memory

**Optimal Pattern**: Warp specialization with TMA-based async pipelining

#### AMD CDNA3/4 Architecture (MI300, MI355)

**Register Allocation**:
*   **Static**: All wavefronts in a workgroup receive the same register allocation
*   Producer waves consume the same registers as consumer waves
*   Cannot optimize register usage per wave role
*   Result: Wave specialization wastes resources

**Memory Hierarchy**:
*   Infinity Fabric: High-bandwidth interconnect between chiplets
*   L2 cache: Distributed across chiplets
*   LDS (Local Data Store): Shared memory, 64 KB per compute unit
*   No hardware-accelerated async copy (must use software prefetch)

**Scheduling Features**:
*   8 SIMD units per compute unit
*   4 wavefronts can execute per SIMD (32 total wavefronts per CU)
*   Static scheduling of wavefronts to SIMD units
*   Manual `sched_barrier` hints influence instruction ordering

**Optimal Pattern**: 8-wave ping-pong or 4-wave interleave

#### Side-by-Side Comparison

| Feature | NVIDIA Hopper (H100) | AMD CDNA3/4 (MI355) |
|---------|---------------------|---------------------|
| Register allocation | Dynamic (per-warp) | Static (per-workgroup) |
| Async memory copy | Hardware (TMA) | Software (manual prefetch) |
| Warp/Wave specialization | ✅ Highly effective | ❌ Wastes registers (80% peak) |
| Best scheduling pattern | Producer/consumer warps | 8-wave ping-pong |
| Tensor core format | WMMA / TMA | MFMA |
| Shared memory swizzling | Hardware-assisted | Manual XOR patterns |
| Optimal tile size (specialization) | Large (256x128 or bigger) | Medium (128x128, 256x256) |
| Optimal tile size (ping-pong) | N/A | Large (256x256 with 8-wave) |

#### Practical Implications for Kernel Developers

**On NVIDIA GPUs (with CUTLASS or ThunderKittens)**:
```cpp
// Warp specialization works great
if (warp_id < NUM_PRODUCER_WARPS) {
    // Producer: fetch data with minimal register usage
    load_tile_async(shared_mem, global_mem);  // TMA
} else {
    // Consumer: compute with large accumulator tiles
    mma(accum_256x128, A, B);  // Can use many registers
}
```

**On AMD GPUs (with HipKittens)**:
```cpp
// 8-wave ping-pong: all waves do both roles
ST_A (&As)[2] = al.allocate<ST_A, 2>();  // Ping-pong buffers

for (int k = 0; k < K_iters; k++) {
    // All waves: prefetch next tile
    G::load(As[toc], g.a, {batch, row, col, k+1});

    // All waves: compute on current tile
    load(A_reg, As[tic]);
    mma_ABt(C_accum, A_reg, B_reg, C_accum);

    tic ^= 1; toc ^= 1;  // Swap
}
```

The static register allocation on AMD means **all waves must do both memory and compute work** to utilize resources efficiently.

### 6.7 Performance Impact and Pattern Selection

#### Performance Measurements

From the HipKittens paper and benchmarks:

**Wave Specialization on AMD MI355X**:
*   BF16 GEMM: ~80% of peak theoretical performance
*   Reason: Register waste from producer waves limits tile size

**8-Wave Ping-Pong on AMD MI355X**:
*   BF16 GEMM: ~95-98% of peak (matches AMD assembly)
*   FP8 GEMM: Competitive with vendor kernels
*   Attention Forward (MHA, GQA): State-of-the-art performance

**4-Wave Interleave on AMD MI355X**:
*   GQA Causal Backward: +5-10% over 8-wave ping-pong
*   Complex fused kernels: Additional gains when fine-grained hiding matters
*   Trade-off: 3-5x code expansion

#### Decision Matrix: Which Pattern to Use

| Scenario | Recommended Pattern | Reasoning |
|----------|-------------------|-----------|
| Standard GEMM (BF16, FP8, FP16) | **8-wave ping-pong** | Achieves 95%+ peak, compact code |
| Attention forward (MHA, GQA, MQA) | **8-wave ping-pong** | Sufficient for SoTA performance |
| Attention backward (simple) | **8-wave ping-pong** | Start here, profile first |
| GQA causal backward | **4-wave interleave** | Demonstrated 5-10% improvement |
| Complex fused kernels | **4-wave interleave** | When every cycle counts |
| Memory-bound kernels (LayerNorm, RoPE) | **8-wave ping-pong** | Simpler code, compute isn't bottleneck |
| Prototyping / research | **8-wave ping-pong** | Faster development, good enough performance |
| Production-critical kernels | Profile both | Measure actual workload |

#### Code Size Impact

**8-Wave Ping-Pong Example** (GEMM kernel):
```cpp
// ~150 lines for mainloop
for (int tile = 0; tile < num_tiles; tile++) {
    load(A_tile, As[tic]);
    load(B_tile, Bs[tic]);
    G::load(As[toc], ...);  // Prefetch
    mma_ABt(C, A_tile, B_tile, C);
    tic ^= 1; toc ^= 1;
}
```

**4-Wave Interleave Example** (FP8 GEMM):
```cpp
// ~600 lines for mainloop - 4x expansion
for (int cluster = 0; cluster < num_clusters; cluster++) {
    // Wave 0, subtile 0
    sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 0, 0);
    sched_barrier(0);
    precompute_addresses(...);

    // Wave 0, subtile 1
    sched_barrier(0);
    mma_ABt_one(c, a, b, c, 0, 1, 0);
    sched_barrier(0);
    load_one<0>(...)

    // ... 40+ more lines per cluster
}
```

The 4-wave pattern requires manually scheduling every instruction, leading to significant code expansion.

#### When Performance Justifies Complexity

Use **4-wave interleave** when:
*   Profiling shows 8-wave leaving >5% performance on table
*   Kernel is production-critical and worth engineering investment
*   Development time budget allows for 3-5x longer implementation
*   Kernel will be reused extensively (e.g., in training frameworks)

Stick with **8-wave ping-pong** when:
*   95% of peak is sufficient for your application
*   Code maintainability and readability matter
*   Rapid prototyping is important
*   Teaching/research contexts where understanding matters

### 6.8 Reference and Further Reading

**Primary Publication**:
*   **Title**: "HipKittens: Fast and Furious AMD Kernels"
*   **Authors**: William Hu, Drew Wadsworth, Stanley Winata, Daniel Fu, Ryan Swann, Muhammad Osama, Sean Siddens, Christopher Ré, Simran Arora (Stanford HazyResearch)
*   **Date**: November 2025
*   **arXiv**: [https://arxiv.org/html/2511.08083v1](https://arxiv.org/html/2511.08083v1)

**Code Repository**:
*   **GitHub**: [https://github.com/HazyResearch/HipKittens](https://github.com/HazyResearch/HipKittens)
*   Includes full kernel implementations demonstrating both patterns
*   Analysis scripts to reproduce paper benchmarks
*   Training examples with BERT and Llama models

**Related Work**:
*   **ThunderKittens**: NVIDIA/CUDA predecessor project from same team
*   **CUTLASS**: NVIDIA's official kernel library (for comparison)
*   **Composable Kernel**: AMD's official library (different approach)

**Key Sections in Codebase**:
*   `kernels/gemm/bf16fp32/mi350x/` - 8-wave ping-pong GEMM examples
*   `kernels/gemm/fp8fp32/FP8_4wave/` - 4-wave interleave FP8 GEMM
*   `kernels/attn/gqa/` - Attention kernels with sched_barrier examples
*   `kernels/gemm/bf16fp32/mi350x/micros/hint_based/schedule_utils.cpp` - Scheduling utilities

## 7. Value Proposition and Ecosystem Position

### 7.1 Key Differentiators

HipKittens occupies a unique position in the AMD GPU programming ecosystem:

*   **Anti-Vendor-Lock-In Mission**: Explicitly designed to foster a "multi-silicon future" - an academic/open-source alternative not beholden to AMD's priorities or release cycles
*   **Minimal & Opinionated vs Comprehensive**: Unlike AMD's Composable Kernel (CK) which tries to cover everything, HipKittens deliberately provides minimal primitives with strong opinions on "the right way," trading generality for simplicity and teachability
*   **Educational & Research-First**: Explicitly names and teaches optimization patterns, designed to be understood and modified by researchers, includes full training examples (BERT, Llama) proving real-world utility
*   **Performance Sweet Spot**: Bridges the gap between "easy to use" (PyTorch) and "maximum performance" (hand-written assembly)

### 7.2 The AMD GPU Programming Spectrum

```
Low-level                                              High-level
(Hard, Fast)                                           (Easy, Slower)

Raw HIP/Assembly → HipKittens → Composable Kernel → PyTorch/JAX
```

HipKittens fills the critical gap for researchers and developers who need:
*   Competitive AMD GPU performance (not just NVIDIA fallback)
*   Ability to experiment with novel kernels without vendor bureaucracy
*   Understanding of *why* kernels are fast (not just black-box usage)
*   Freedom from CUDA ecosystem lock-in

### 7.3 Comparison to Alternatives

*   **vs Raw HIP**: Much easier to use while maintaining near-metal performance
*   **vs Composable Kernel (AMD official)**: Simpler, more accessible, better for learning and research
*   **vs Triton/JAX**: Lower-level control, AMD-specific optimizations, but requires separate implementation per vendor
*   **vs ThunderKittens**: Same team, same philosophy, but for NVIDIA/CUDA - patterns transfer but code doesn't

## 8. Conclusion

HipKittens is a valuable contribution to the field of high-performance computing for AI. By providing a set of well-designed primitives for AMD GPUs, it helps to bridge the gap between hardware-specific programming and high-level deep learning frameworks. The project's focus on performance, its comprehensive set of example kernels, and its open-source nature make it a promising tool for researchers and practitioners who want to unlock the full potential of AMD's AI hardware. It represents a significant step towards a more diverse and competitive hardware ecosystem for artificial intelligence.
