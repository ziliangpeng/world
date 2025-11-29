# GPU Kernels for AI/ML: A Comprehensive Guide

**Last Updated**: November 2025

## Table of Contents

1. [Introduction & Motivation](#introduction--motivation)
2. [GPU Architecture Fundamentals](#gpu-architecture-fundamentals)
3. [History: The Evolution of Kernel Development](#history-the-evolution-of-kernel-development)
4. [CUDA Programming](#cuda-programming)
5. [Triton](#triton)
6. [CUTLASS](#cutlass)
7. [Modern Frameworks](#modern-frameworks)
8. [Optimization Fundamentals](#optimization-fundamentals)
9. [Advanced Patterns](#advanced-patterns)
10. [Real-World Examples](#real-world-examples)
11. [Current Best Practices (2024-2025)](#current-best-practices-2024-2025)
12. [Sources](#sources)

---

## Introduction & Motivation

### Why Write Custom GPU Kernels?

Modern deep learning frameworks like PyTorch and TensorFlow provide thousands of pre-optimized operations. So why would you ever need to write a custom GPU kernel?

**The Performance Gap**: While framework operations are highly optimized, they're designed for general use cases. For specialized workloads, custom kernels can deliver:

- **2-10x speedups** for common operations through fusion and memory optimization
- **10-100x speedups** for specialized operations not in standard libraries
- **Breakthrough capabilities** like FlashAttention enabling 10x longer contexts

**When Custom Kernels Matter**:

1. **Novel Operations**: If you're implementing new research (e.g., attention variants), no library has your operation
2. **Fusion Opportunities**: Combining multiple operations eliminates memory roundtrips
3. **Memory Bottlenecks**: Standard operations may not optimally use the memory hierarchy
4. **Hardware-Specific Optimization**: Exploiting latest GPU features (Tensor Cores, asynchronous operations)

### The Cost of NOT Optimizing

Consider standard attention computation in transformers:

| Approach | Memory Reads/Writes | Speed (sequences/sec) | Context Length |
|----------|---------------------|----------------------|----------------|
| **Naive PyTorch** | O(N²) full materialization | 100 | 2K tokens |
| **Fused Kernel** | O(N²) but reduced passes | 300 | 4K tokens |
| **FlashAttention** | O(N) through tiling | 1,000 | 32K tokens |

The difference isn't marginal—it's the difference between research being possible or impossible.

### Three Eras of Kernel Writing

The difficulty and accessibility of writing GPU kernels has dramatically changed:

**Era 1: CUDA Expert Era (2007-2020)**
- Requires deep GPU architecture knowledge
- Manual memory management, thread scheduling
- Weeks to write a single optimized kernel
- Limited to CUDA specialists

**Era 2: Framework Abstraction Era (2015-2020)**
- Most researchers use pre-built kernels from cuDNN
- Custom kernels rare, framework-specific
- Deep learning accelerates without kernel expertise

**Era 3: Democratization Era (2021-present)**
- Triton enables Python-like kernel writing
- Auto-tuning handles optimization details
- Researchers write production-quality kernels in days
- FlashAttention, ThunderKittens make advanced patterns accessible

This guide covers all three eras, from CUDA fundamentals to modern high-level approaches.

---

## GPU Architecture Fundamentals

To write efficient GPU kernels, you need to understand how GPUs actually execute code. This section explains the key architectural concepts that matter for kernel development.

### The Parallel Processing Model

**CPUs vs GPUs: Different Design Philosophy**

| Aspect | CPU | GPU |
|--------|-----|-----|
| **Cores** | 8-64 powerful cores | 1,000s of simple cores |
| **Design Goal** | Minimize latency per thread | Maximize throughput across threads |
| **Cache** | Large (MB per core) | Small (KB per core) |
| **Best For** | Complex logic, branching | Massive parallel computation |

GPUs achieve their performance not by executing individual operations faster, but by executing thousands of operations simultaneously.

### Streaming Multiprocessors (SMs)

The fundamental compute unit of a GPU is the **Streaming Multiprocessor (SM)**:

- A modern GPU contains **50-150 SMs** (depending on model)
- Each SM can execute **hundreds of threads concurrently**
- SMs have their own **register files** and **shared memory**
- SMs operate **independently** (different SMs can run different code)

**Example: NVIDIA H100 GPU**
- 132 SMs
- Each SM: 128 CUDA cores + 4 Tensor Cores
- Total: 16,896 CUDA cores + 528 Tensor Cores

### CUDA Cores and Warps

**CUDA Core**: A simple processor that executes one floating-point or integer operation per clock cycle.

**The Warp: The True Execution Unit**

The most critical concept in GPU programming is the **warp**:

- **Warp size**: 32 threads (fixed since CUDA 1.0, hasn't changed)
- **Execution model**: SIMD (Single Instruction, Multiple Data)
- All 32 threads in a warp execute the **same instruction** on different data
- If threads diverge (branches), the warp serializes execution (slow!)

**Why This Matters for Kernels**:
```cuda
// BAD: Warp divergence
if (threadIdx.x % 2 == 0) {
    expensiveOperation();  // Half the warp executes this
} else {
    cheapOperation();      // Other half waits, then executes this
}

// GOOD: All threads take same path
int x = (threadIdx.x % 2 == 0) ? expensiveOperation() : cheapOperation();
```

### Tensor Cores

Starting with Volta architecture (2018), NVIDIA GPUs include **Tensor Cores**—specialized hardware for matrix operations:

**Evolution Across Generations**:

| GPU Architecture | Tensor Core Gen | Matrix Size | Precisions Supported | Performance (FP16) |
|------------------|-----------------|-------------|----------------------|-------------------|
| **Volta** (V100) | 1st | 4×4×4 | FP16 | 125 TFLOPS |
| **Turing** (T4) | 2nd | 8×8×4 | FP16, INT8, INT4 | 65 TFLOPS |
| **Ampere** (A100) | 3rd | 8×4×8 | FP16, BF16, TF32, FP64, INT8 | 312 TFLOPS |
| **Hopper** (H100) | 4th | Variable | FP8, FP16, BF16, TF32, FP64, INT8 | 989 TFLOPS (FP8) |

**Key Insight**: Tensor Cores are 10-20x faster than CUDA cores for matrix multiplication. Any kernel doing heavy linear algebra should use them.

### Memory Hierarchy: The Performance Bottleneck

GPU performance is often limited not by computation but by **memory bandwidth**. Understanding the memory hierarchy is crucial:

```
┌──────────────────────────────────────────────────────────┐
│ Registers (per thread)                                   │
│ - Fastest: 1 cycle latency                              │
│ - Capacity: ~64 KB per SM (split among threads)         │
│ - Private to each thread                                │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ Shared Memory / L1 Cache (per SM)                       │
│ - Very Fast: ~20-30 cycle latency                       │
│ - Capacity: 64-228 KB per SM (architecture-dependent)   │
│ - Shared across all threads in a block                  │
│ - Explicitly managed by programmer                      │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ L2 Cache (per GPU)                                       │
│ - Moderate: ~200 cycle latency                          │
│ - Capacity: 40-80 MB (shared across all SMs)           │
│ - Hardware-managed                                       │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│ Global Memory (HBM - High Bandwidth Memory)              │
│ - Slow: ~400-800 cycle latency                          │
│ - Capacity: 40-80 GB (H100/A100)                        │
│ - Bandwidth: 2-3 TB/s (bandwidth, not latency!)         │
│ - Visible to all threads across all SMs                 │
└──────────────────────────────────────────────────────────┘
```

**The Golden Rule of GPU Programming**: Minimize global memory accesses. A single trip to global memory costs as much as 400-800 arithmetic operations.

**Latency vs Bandwidth Example**:

```
# H100 GPU Specs:
- Peak FP16 Compute: 989 TFLOPS
- Global Memory Bandwidth: 3.35 TB/s

# For a simple elementwise operation: C = A + B
- Each element: 2 reads (A, B) + 1 write (C) = 3 × 2 bytes = 6 bytes
- Memory time: 6 bytes ÷ 3.35 TB/s = 1.79 ns
- Compute time: 1 FP16 add ÷ 989 TFLOPS = 0.001 ns

# Result: 99.9% of time is spent waiting for memory!
```

This is why optimization techniques focus heavily on memory management.

### Thread Hierarchy

CUDA organizes threads in a three-level hierarchy:

```
Grid (entire kernel launch)
├── Block 0 (runs on one SM)
│   ├── Warp 0 (32 threads)
│   ├── Warp 1 (32 threads)
│   └── Warp N...
├── Block 1 (runs on one SM)
│   └── Warps...
└── Block N...
```

**Key Properties**:
- **Threads within a block** can cooperate via shared memory and synchronize
- **Threads in different blocks** cannot synchronize (blocks may run in any order)
- **One block = one SM** (blocks don't migrate between SMs)
- **Multiple blocks can run on one SM** concurrently (if resources permit)

**Sizing Considerations**:
- Blocks should have **multiples of 32 threads** (warp size)
- Typical sizes: 128, 256, 512 threads per block
- Too small: underutilizes SM
- Too large: reduces occupancy (fewer blocks can fit)

### Occupancy: Keeping the GPU Busy

**Occupancy** = (Active Warps per SM) / (Maximum Warps per SM)

Higher occupancy allows the GPU to hide memory latency by switching between warps.

**Occupancy Limiters**:

1. **Registers**: Each SM has limited registers (65,536 on A100)
   - If your kernel uses 64 registers/thread × 1024 threads = 65,536 registers
   - Only 1 block can run → low occupancy

2. **Shared Memory**: Each SM has limited shared memory (64-228 KB)
   - If you allocate 48 KB per block, max 3-4 blocks can coexist

3. **Thread Blocks**: Hardware limit on concurrent blocks per SM (16-32 depending on architecture)

**Trade-off**: Sometimes lower occupancy but better per-thread performance is faster than high occupancy with poor per-thread performance.

### Why This Matters for Kernel Development

Every optimization technique in later sections traces back to these fundamentals:

- **Memory coalescing** → Exploit memory hierarchy
- **Kernel fusion** → Reduce global memory roundtrips
- **Tiling** → Reuse data in shared memory
- **Warp-level programming** → Minimize divergence
- **Tensor Core utilization** → 10x faster matrix operations

Understanding these concepts is the foundation for everything that follows.

---

## History: The Evolution of Kernel Development

The history of GPU kernel development is a story of democratization—from esoteric expertise available to a handful of specialists to accessible tools used by thousands of researchers. This evolution happened in four distinct phases.

### Phase 1: The CUDA Era (2007-2015)

**2007: CUDA Launches, GPU Computing is Born**

Before CUDA, programming GPUs meant writing in graphics APIs (OpenGL/DirectX) and framing every computation as a rendering problem. [NVIDIA introduced CUDA in 2007](https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/), fundamentally changing scientific computing.

**The Origin Story**:

The development of CUDA actually began in 2004 when Ian Buck, a Stanford PhD student, created **Brook**—a programming language for general-purpose GPU computing. NVIDIA hired Buck in 2004 and paired him with John Nickolls, NVIDIA's director of architecture. Together, they transformed Brook into CUDA.

**What CUDA Enabled**:

- **C/C++ programming model** instead of graphics shaders
- Explicit control over **memory hierarchy** (global, shared, registers)
- **Thread management** via blocks and grids
- **CUDA SDK** released February 15, 2007 for Windows and Linux

**The Challenge**: Writing CUDA kernels in this era required deep expertise:

```cuda
// 2007-era CUDA matrix multiplication (simplified)
__global__ void matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];  // Terrible memory access!
    }
    C[row * N + col] = sum;
}
```

This naive implementation is **30-50x slower** than optimal. Achieving good performance required:
- Manual shared memory tiling
- Coalesced memory access patterns
- Register blocking
- Occupancy tuning

**The Library Response** (2009-2014):

Rather than expecting every researcher to become a kernel expert, NVIDIA created highly optimized libraries:

| Library | Year | Purpose |
|---------|------|---------|
| **cuBLAS** | 2008 | Basic Linear Algebra (GEMM, etc.) |
| **cuFFT** | 2008 | Fast Fourier Transforms |
| **cuRAND** | 2010 | Random number generation |
| **cuDNN** | 2014 | Deep learning primitives |

**cuDNN (2014)**: The Game Changer for Deep Learning

[Released in September 2014](https://arxiv.org/pdf/1410.0759), cuDNN provided optimized implementations of:
- Convolutions (forward, backward)
- Activation functions (ReLU, sigmoid, tanh)
- Pooling operations
- Normalization (batch norm, layer norm)

**Impact**: Researchers could train neural networks without writing a single CUDA kernel. PyTorch and TensorFlow rely heavily on cuDNN to this day.

### Phase 2: Framework Abstraction Era (2015-2020)

**The Deep Learning Explosion**

By 2015, deep learning was transitioning from research curiosity to industry-changing technology. Frameworks emerged to make it accessible:

| Framework | Release | Kernel Strategy |
|-----------|---------|-----------------|
| **TensorFlow** | 2015 | Eigen (CPU) + cuDNN (GPU) |
| **PyTorch** | 2016 | ATen (CPU) + cuDNN (GPU) |
| **MXNet** | 2015 | cuDNN (GPU) |
| **Caffe** | 2014 | cuDNN (GPU) |

**The Abstraction Philosophy**:

Frameworks provided high-level APIs while delegating to cuDNN under the hood:

```python
# PyTorch (user perspective)
output = torch.nn.functional.conv2d(input, weight, bias)

# Under the hood:
# 1. PyTorch checks input shapes and types
# 2. Dispatches to cuDNN's cudnnConvolutionForward()
# 3. cuDNN selects optimal algorithm for this specific configuration
# 4. Highly optimized CUDA kernel executes
```

**Who Wrote Custom Kernels?**

In this era, custom kernel development was limited to:

1. **Framework developers**: Building the core operations
2. **NVIDIA engineers**: Optimizing cuDNN for each new GPU architecture
3. **Specialized researchers**: Implementing novel operations not in cuDNN

For 95% of deep learning researchers, kernel development was irrelevant—they used what frameworks provided.

**The Hidden Cost**:

While this abstraction enabled rapid progress, it had drawbacks:

- **Limited composability**: Hard to fuse operations across framework boundaries
- **Suboptimal memory usage**: Each operation reads from global memory → computes → writes back
- **Innovation bottleneck**: New operations required framework adoption (slow process)

**Example: The Attention Problem**

Standard transformer attention in 2017-2020:

```python
# Standard PyTorch attention (2020)
Q, K, V = ...  # [batch, heads, seq_len, head_dim]
scores = Q @ K.transpose(-2, -1)  # GPU kernel 1: matmul
scores = scores / math.sqrt(d_k)  # GPU kernel 2: elementwise
attn = scores.softmax(dim=-1)     # GPU kernel 3: softmax
output = attn @ V                  # GPU kernel 4: matmul
```

Each line is a separate kernel launch:
1. Compute scores → write to memory
2. Read scores → scale → write to memory
3. Read scores → softmax → write attention weights to memory
4. Read attention → matmul → write output

For a 2K token sequence, the intermediate `scores` and `attn` matrices are both (seq_len × seq_len) = 4M elements. With FP16, that's **8 MB of memory traffic per attention head**. With 32 heads, **256 MB just to shuttle intermediate results**.

This would become the catalyst for Phase 3.

### Phase 3: The Triton Revolution (2021-2023)

**2021: OpenAI Releases Triton**

On July 28, 2021, [OpenAI open-sourced Triton](https://openai.com/index/triton/)—a Python-like language for writing GPU kernels. The impact was immediate.

**The Core Innovation: Block-Level Programming**

Traditional CUDA requires thinking about individual threads:

```cuda
// CUDA: Thread-level thinking
__global__ void add(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // Which element am I?
    if (idx < N) C[idx] = A[idx] + B[idx];
}
```

Triton lets you think about blocks of data:

```python
# Triton: Block-level thinking
@triton.jit
def add_kernel(A, B, C, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    a = tl.load(A + offs, mask=mask)
    b = tl.load(B + offs, mask=mask)
    c = a + b
    tl.store(C + offs, c, mask=mask)
```

**Key Abstraction**: You specify operations on blocks (e.g., 1024 elements), and Triton's compiler:
- Figures out thread-level details
- Optimizes memory access patterns
- Handles shared memory automatically
- Tunes for your specific GPU

**The Democratization Thesis**:

Triton's announcement claimed: *"Researchers with no CUDA experience can write highly efficient GPU code—most of the time on par with what an expert would produce."*

This seemed too good to be true. It wasn't.

**Real-World Adoption**:

Within 18 months, Triton became the de facto standard for custom ML kernels:

| Project | Year | Impact |
|---------|------|--------|
| **Flash Attention** | 2022 | Reference implementation includes Triton version |
| **PyTorch 2.0** | 2023 | `torch.compile` backend uses Triton for fused kernels |
| **xFormers** (Meta) | 2022 | Memory-efficient attention kernels in Triton |
| **vLLM** | 2023 | High-performance LLM inference with Triton kernels |

**Why Triton Succeeded**:

1. **Python ecosystem**: Integrates seamlessly with PyTorch/JAX
2. **Auto-tuning**: Automatically searches for optimal configurations
3. **JIT compilation**: Compiles kernels on-demand for specific shapes
4. **Readable**: Code looks like NumPy, not assembly

**Limitations**:

- **Performance ceiling**: Expert hand-written CUDA can still be 10-30% faster for specific cases
- **Hardware specific**: Harder to write portable code (AMD, Intel)
- **Debugging**: Error messages can be cryptic

### Phase 4: The Modern Era (2023-2025)

**Specialized Frameworks and Hardware Co-Design**

While Triton democratized kernel writing, the bleeding edge moved to **specialized frameworks** that combine:
- Novel algorithmic insights
- Extreme hardware optimization
- Domain-specific languages (DSLs)

#### FlashAttention: The Attention Revolution

**FlashAttention v1** ([May 2022](https://arxiv.org/abs/2205.14135)) by Tri Dao et al. proved that algorithmic + kernel co-design could deliver orders-of-magnitude improvements:

**The IO-Aware Insight**:

Standard attention is memory-bound:
- Compute: O(N² × d) FLOPs
- Memory: O(N²) reads/writes for attention matrix

FlashAttention's innovation:
- **Tiling**: Break computation into blocks that fit in shared memory
- **Recomputation**: Recompute attention scores on-the-fly instead of storing
- **Fused kernel**: All operations in a single kernel

**Results**:
- **2-4x faster** than standard attention
- **10-20x memory reduction** (enables longer contexts)
- **Exact** (not an approximation)

**Evolution**:

| Version | Release | Key Innovation | Performance (H100) |
|---------|---------|----------------|-------------------|
| **v1** | May 2022 | IO-aware tiling | ~35% FLOPS utilization |
| **v2** | July 2023 | Better parallelism across warps | ~50% FLOPS utilization |
| **v3** | July 2024 | Asynchronous execution (Hopper) + FP8 | **75% FLOPS utilization** (1.2 PFLOPS) |

[FlashAttention-3](https://arxiv.org/abs/2407.08608) achieves **2x speedup over FlashAttention-2** on H100 by exploiting:
- Tensor Memory Accelerator (TMA) for asynchronous loads
- Warp specialization (some warps load, others compute)
- Low-precision FP8 with block quantization

#### ThunderKittens: Warp-Level Abstractions

[Released in 2024](https://arxiv.org/abs/2410.20399) by Stanford's Hazy Research lab, **ThunderKittens** provides a different abstraction:

**Philosophy**: Instead of hiding hardware details (like Triton), expose them in a structured way.

**Three-Level Hierarchy**:

1. **Warp-level**: 16×16 matrix tiles as basic data structures
2. **Block-level**: Templates for overlapping async operations
3. **Grid-level**: Hide launch overhead

**Code Example** (conceptual):

```cpp
// ThunderKittens
using namespace kittens;

__global__ void matmul_kernel(float *A, float *B, float *C) {
    st<16, 16> tile_A, tile_B, tile_C;  // 16×16 tiles in shared memory

    load(tile_A, A);   // Asynchronous load
    load(tile_B, B);

    mma(tile_C, tile_A, tile_B);  // Tensor Core multiply-accumulate

    store(C, tile_C);
}
```

**Key Features**:
- **Tensor Core first**: Designed around 16×16 tiles (tensor core native size)
- **Bank conflict free**: Shared memory layout automatically optimized
- **Async operations**: Easy to overlap compute and memory

**Performance**: Achieves **90%+ of cuBLAS performance** with significantly less code than raw CUDA.

#### CuTe (CUTLASS 3.0): Template Metaprogramming

NVIDIA's [CUTLASS 3.0](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design) (2022) introduced **CuTe**—a DSL embedded in C++ templates:

**Philosophy**: Use C++ templates to describe multi-dimensional layouts and let the compiler optimize.

**Power**:
- Describe complex memory layouts declaratively
- Compose transformations (transpose, partition, etc.)
- Compiler generates optimal code

**Trade-off**: Extremely powerful but steep learning curve (template metaprogramming).

### Summary: Four Phases

| Phase | Years | Kernel Writers | Tools | Speedup Potential |
|-------|-------|---------------|-------|------------------|
| **CUDA Era** | 2007-2015 | CUDA experts (~1,000s) | Raw CUDA, cuBLAS | 10-100x vs CPU |
| **Framework Era** | 2015-2020 | Framework devs (~100s) | cuDNN, TensorRT | 2-5x via libraries |
| **Triton Era** | 2021-2023 | ML researchers (~10,000s) | Triton, PyTorch 2.0 | 2-5x via fusion |
| **Modern Era** | 2023-2025 | Specialized researchers | FlashAttention, ThunderKittens | 2-10x via co-design |

**The Key Insight**: Each phase didn't replace the previous one—they stack. Modern AI systems use:
- cuDLAS/cuDNN for standard operations (Phase 1-2)
- Triton for custom fused kernels (Phase 3)
- FlashAttention/ThunderKittens for cutting-edge operations (Phase 4)
- Hand-optimized CUDA for the absolute critical path (Phase 1, never went away)

---

## CUDA Programming

CUDA remains the foundation of GPU programming. Even if you use higher-level tools like Triton, understanding CUDA concepts is essential for debugging, performance tuning, and pushing limits.

### CUDA Programming Model

**Kernel Definition**:

A CUDA kernel is a C++ function executed by thousands of threads in parallel:

```cuda
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

**Key Elements**:

- `__global__`: Function runs on GPU, callable from CPU
- `blockIdx.x`: Which block this thread belongs to (grid-level)
- `threadIdx.x`: Thread's position within its block (block-level)
- `blockDim.x`: How many threads per block

**Kernel Launch**:

```cpp
// Allocate GPU memory
float *d_A, *d_B, *d_C;
cudaMalloc(&d_A, N * sizeof(float));
cudaMalloc(&d_B, N * sizeof(float));
cudaMalloc(&d_C, N * sizeof(float));

// Copy data to GPU
cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

// Launch kernel: (blocks, threads_per_block)
int threadsPerBlock = 256;
int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

// Copy result back
cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
```

### Thread Hierarchy Deep Dive

**3D Grid and Block Dimensions**:

CUDA supports 3D grids and 3D blocks for indexing convenience:

```cuda
// 2D matrix addition
__global__ void matrixAdd(float *A, float *B, float *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

// Launch with 2D grid
dim3 threadsPerBlock(16, 16);  // 256 threads per block
dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
matrixAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);
```

### Memory Management

**Memory Types in CUDA**:

| Memory | Scope | Lifetime | Speed | Usage |
|--------|-------|----------|-------|-------|
| **Register** | Thread | Thread | Fastest (1 cycle) | Automatic variables |
| **Local** | Thread | Thread | Slow (global memory)* | Arrays, large structs |
| **Shared** | Block | Block | Fast (~20 cycles) | `__shared__` variables |
| **Global** | Grid | Application | Slow (~400 cycles) | `cudaMalloc` |
| **Constant** | Grid | Application | Fast (cached) | `__constant__` (read-only) |
| **Texture** | Grid | Application | Fast (cached) | Texture memory (specialized) |

*Local memory sounds fast but is actually stored in global memory—just private to each thread.

**Shared Memory Example**:

```cuda
__global__ void matmulShared(float *A, float *B, float *C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  // Ensure all threads loaded their data

        // Compute partial sum using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();  // Ensure all threads finished before loading next tile
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}
```

**Why This is Faster**:

- **Without shared memory**: Each thread reads N elements from global memory (slow)
- **With shared memory**: Threads cooperate to load tiles once, reuse from shared memory (fast)
- **Speedup**: 5-10x for matrix multiplication

### Synchronization

**Within a Block**:

```cuda
__syncthreads();  // All threads in block reach this point before any proceed
```

**Caution**: Never put `__syncthreads()` inside a conditional that's not uniform across the block:

```cuda
// BAD: Deadlock if some threads skip this
if (threadIdx.x < 16) {
    __syncthreads();  // Only half the threads reach here!
}

// GOOD: All threads execute
__syncthreads();
if (threadIdx.x < 16) {
    // Do work
}
```

**Across Blocks**:

You **cannot** synchronize threads in different blocks within a kernel. If you need this:
- Launch kernel → wait for completion → launch next kernel
- Use **Cooperative Groups** (advanced, limited support)

### Memory Access Patterns

**Coalesced vs Uncoalesced Access**:

```cuda
// COALESCED: Threads 0-31 access consecutive addresses
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float value = data[idx];  // Good!

// UNCOALESCED: Threads access strided addresses
int idx = threadIdx.x * 1024;
float value = data[idx];  // Bad! 32x slower memory bandwidth
```

**Rule of Thumb**: Within a warp (32 consecutive threads), if they access consecutive 4-byte words, the hardware combines them into a single memory transaction. Otherwise, it issues 32 separate transactions.

**Strided Access Example**:

```cuda
// Matrix stored in row-major order
// Reading a column is uncoalesced!
for (int i = 0; i < rows; i++) {
    float val = matrix[i * cols + col];  // Each iteration: different row
}

// Solution: Transpose the matrix, or use shared memory to reorganize
```

### Warp-Level Primitives

**Warp Shuffle Operations** (CUDA 9+):

Threads within a warp can directly exchange register values without shared memory:

```cuda
__global__ void warpSum(int *data, int *result) {
    int value = data[threadIdx.x];

    // Sum across warp using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    if (threadIdx.x == 0) {
        result[0] = value;
    }
}
```

This is **much faster** than using shared memory + atomics.

### Tensor Core Programming (WMMA)

**Warp Matrix Multiply-Accumulate API**:

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_gemm(half *A, half *B, float *C, int M, int N, int K) {
    // Declare fragments (16×16 tiles stored in registers)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);

    // Load matrices into fragments
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);

    // Perform D = A * B + C using Tensor Cores
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store result
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}
```

**Performance**: This single `mma_sync` call performs 16×16×16 = 4,096 FP16 multiply-adds in a few cycles—**10-20x faster** than CUDA cores.

### When to Use CUDA

**Use CUDA When**:

1. **Maximum performance is critical**: You need every last percent of GPU utilization
2. **Fine-grained control**: Specific memory layouts, instruction scheduling
3. **Hardware-specific features**: Latest GPU features not yet in higher-level tools
4. **Debugging**: Understanding exactly what the GPU is doing

**Avoid CUDA When**:

1. **Rapid prototyping**: Triton will get you 80-90% of the way with 10% of the effort
2. **Portability matters**: CUDA is NVIDIA-only
3. **You're not a GPU expert**: Easy to write slow CUDA code

**Reality Check**: Even in 2025, critical libraries like cuBLAS, cuDNN, and TensorRT are written in hand-optimized CUDA. For pushing absolute limits, CUDA is unavoidable.

---

## Triton

[Triton](https://github.com/triton-lang/triton), released by OpenAI in 2021, represents the most successful attempt to democratize GPU programming. It enables researchers to write high-performance kernels without CUDA expertise.

### Core Philosophy

**The Triton Thesis**: Most GPU performance comes from a few key decisions:
1. What data to load into shared memory (tiling)
2. How to partition work across threads (parallelization)
3. Memory access patterns (coalescing)

If a compiler can make these decisions automatically, programmers can focus on **what** to compute, not **how** to schedule it.

**Block-Level Abstraction**:

Instead of programming individual threads (CUDA), Triton lets you program **blocks** of data:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID: which block am I?
    pid = tl.program_id(axis=0)

    # This block processes elements [pid * BLOCK_SIZE : (pid+1) * BLOCK_SIZE]
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Load data (masked to handle edge cases)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    output = x + y

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

# Launch kernel
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output
```

**What Triton Handles Automatically**:
- Thread indexing within blocks
- Memory coalescing
- Register allocation
- Instruction scheduling

### Python Integration

Triton integrates seamlessly with PyTorch:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def fused_relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    output = tl.where(x > 0, x, 0.0)  # ReLU
    tl.store(output_ptr + offsets, output, mask=mask)

# PyTorch wrapper
def fused_relu(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

# Use it like any PyTorch function
x = torch.randn(1000000, device='cuda')
y = fused_relu(x)
```

### Auto-Tuning

One of Triton's killer features: automatic performance tuning.

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],  # Autotune based on these parameters
)
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # ... kernel implementation ...
```

**How Auto-Tuning Works**:
1. First call: Triton benchmarks all configurations
2. Picks fastest for this (M, N, K) combination
3. Caches result
4. Future calls with same shape use cached best config

This means you get expert-level performance tuning without manual work.

### Matrix Multiplication Example

Here's a simplified Triton matmul (educational, not production):

```python
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for this block's output
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers to blocks of A and B
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    # Accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)

        # Matrix multiply
        accumulator += tl.dot(a, b)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Store result
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator)
```

**Key Points**:
- `tl.dot()` compiles to Tensor Core instructions (fast!)
- Automatic shared memory management
- Block sizes are compile-time constants (`tl.constexpr`)

### Fused Softmax Example

One of Triton's most common use cases: fusing operations to reduce memory traffic.

```python
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one row
    row_idx = tl.program_id(0)

    # Pointers to this row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # Load row (with masking for variable length)
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # Softmax computation (fused!)
    row_minus_max = row - tl.max(row, axis=0)  # Numerical stability
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # Store result
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)
```

**Why This is Fast**:
- Standard PyTorch: 3 separate kernels (max, subtract+exp, sum+divide)
- Triton version: 1 kernel, data stays in registers/shared memory
- Result: **2-3x faster**

### Compilation Model

**JIT (Just-In-Time) Compilation**:

Triton compiles kernels on first use:

```
Python source
    ↓
Triton IR (Intermediate Representation)
    ↓
LLVM IR
    ↓
PTX (NVIDIA assembly)
    ↓
SASS (actual GPU instructions)
```

**Caching**: Compiled kernels are cached, so recompilation only happens when:
- Code changes
- Input shapes change (if not using constexpr)
- Tuning configs change

### Performance vs CUDA

**Benchmark: Matrix Multiplication (A100 GPU)**

| Implementation | Performance (TFLOPS) | Lines of Code |
|----------------|---------------------|---------------|
| **Naive PyTorch** | 15 TFLOPS | 1 line |
| **cuBLAS** | 310 TFLOPS | 1 line (PyTorch uses this) |
| **Hand-optimized CUDA** | 300 TFLOPS | ~500 lines |
| **Triton (auto-tuned)** | 280 TFLOPS | ~100 lines |

**Takeaway**: Triton gets you **90% of cuBLAS performance** with far less complexity than CUDA.

### Limitations

**When Triton Falls Short**:

1. **Irregular computation**: CUDA gives more control for non-uniform workloads
2. **Absolute peak performance**: Hand-tuned CUDA can still be 10-30% faster
3. **Hardware portability**: Triton is NVIDIA-focused (AMD support is experimental)
4. **Complex synchronization**: Cross-block coordination harder than in CUDA
5. **Debugging**: Error messages can be cryptic; no step-through debugger

### Real-World Adoption

**PyTorch 2.0 Integration**:

`torch.compile` uses Triton as a backend:

```python
@torch.compile
def fused_ops(x, w, b):
    return torch.relu(torch.matmul(x, w) + b)

# PyTorch automatically:
# 1. Traces the computation graph
# 2. Identifies fusion opportunities
# 3. Generates Triton kernels
# 4. Compiles and caches them
```

This gives **1.5-2x speedups** on many models with zero code changes.

**OpenAI GPT Training**:

OpenAI uses custom Triton kernels for:
- Flash Attention implementation
- Fused Adam optimizer
- Custom activation functions
- Gradient accumulation patterns

**vLLM (LLM Inference)**:

[vLLM](https://github.com/vllm-project/vllm) uses Triton for:
- PagedAttention (memory-efficient KV cache)
- Fused RoPE (rotary position embeddings)
- Quantization kernels (INT8, FP8)

### Learning Resources

**Official Resources**:
- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub](https://github.com/triton-lang/triton)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)

**Community Tutorials**:
- [CUDA Mode Lecture Series](https://github.com/cuda-mode/lectures) (covers Triton extensively)
- [OpenAI Blog Post](https://openai.com/index/triton/)

---

## CUTLASS

[CUTLASS](https://github.com/NVIDIA/cutlass) (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's library of highly optimized GEMM (General Matrix Multiply) kernels. While Triton aims for simplicity, CUTLASS targets absolute peak performance through C++ template metaprogramming.

### What is CUTLASS?

**Definition**: A collection of **CUDA C++ templates** for implementing high-performance matrix operations at all levels and scales.

**Philosophy**: Decompose GEMM into fundamental components (data movement, math operations, threadblock scheduling) and expose them as composable templates.

**Not a Library in the Traditional Sense**:
- cuBLAS: Call `cublasSgemm()`, get optimized kernel (black box)
- CUTLASS: Include templates, customize components, compiler generates your kernel

### Why CUTLASS Exists

**The Customization Problem**:

Standard cuBLAS is optimized for common cases:
- C = αAB + βC (standard GEMM)
- Common matrix sizes
- Standard data types (FP32, FP16)

But modern AI needs:
- Fused operations: C = GELU(αAB + βC + bias)
- Custom data types: FP8, INT4, custom formats
- Non-standard sizes: 127 × 4096 (oddly shaped)
- Mixed precision: FP16 input, FP32 accumulation, INT8 output

CUTLASS lets you build these custom kernels while maintaining cuBLAS-level performance.

### Core Concepts

**Template-Based Design**:

```cpp
#include "cutlass/gemm/device/gemm.h"

// Define GEMM kernel via templates
using Gemm = cutlass::gemm::device::Gemm<
    float,                                      // ElementA
    cutlass::layout::RowMajor,                  // LayoutA
    float,                                      // ElementB
    cutlass::layout::ColumnMajor,               // LayoutB
    float,                                      // ElementC
    cutlass::layout::RowMajor,                  // LayoutC
    float,                                      // ElementAccumulator
    cutlass::arch::OpClassTensorOp,             // Use Tensor Cores
    cutlass::arch::Sm80                         // Target Ampere architecture
>;

// Use it
Gemm gemm_op;
gemm_op(problem_size, alpha, A, B, beta, C, C);
```

**Composable Hierarchies**:

CUTLASS organizes GEMM into levels:

```
Grid Level (entire kernel)
    ↓
Threadblock Level (tile of output)
    ↓
Warp Level (sub-tile per warp)
    ↓
Thread Level (individual elements)
```

Each level has customizable components:
- **Tile sizes**: How much data each level processes
- **Data movement**: How data moves between memory levels
- **Math operations**: What computation happens (standard, Tensor Core, custom)

### CuTe: The Layout Language

**CUTLASS 3.0** introduced **CuTe**—a domain-specific language for describing memory layouts.

**The Layout Problem**:

GPUs have complex memory layouts:
- Row-major vs column-major matrices
- Strided access (every Nth element)
- Swizzled layouts (optimized for banks)
- Tensor Core native layouts (16×16 tiles)

**CuTe Solution**: Describe layouts declaratively, compiler optimizes.

**Example**:

```cpp
#include <cute/tensor.hpp>
using namespace cute;

// Describe a 2D layout
auto shape = make_shape(128, 64);    // 128 rows × 64 columns
auto stride = make_stride(64, 1);    // Row-major (stride 64 between rows)

auto layout = make_layout(shape, stride);

// Partition for different threads
auto tiled_layout = make_layout(
    make_shape(make_shape(4, 32), make_shape(2, 32)),  // (Tile, Thread) dimensions
    make_stride(make_stride(64, 1), make_stride(2048, 64))
);
```

**Power**: Once you describe the layout, CuTe generates optimal memory access patterns.

### Tensor Core Integration

CUTLASS was designed from the ground up to exploit Tensor Cores:

**WMMA Abstraction** (older):

```cpp
using MmaTensorOp = cutlass::gemm::warp::MmaTensorOp<
    cutlass::gemm::GemmShape<16, 16, 16>,   // Warp-level tile shape
    cutlass::half_t,                         // ElementA
    cutlass::layout::RowMajor,               // LayoutA
    cutlass::half_t,                         // ElementB
    cutlass::layout::ColumnMajor,            // LayoutB
    float,                                   // ElementC
    cutlass::layout::RowMajor                // LayoutC
>;
```

**CUTLASS 3.0 Tensor Core API**:

```cpp
#include <cute/atom/mma_atom.hpp>

// Ampere Tensor Core (FP16)
using MMA_Atom = cute::MMA_Atom<
    cute::SM80_16x8x16_F16F16F16F16_TN  // Ampere FP16 Tensor Core
>;

// Hopper Tensor Core (FP8)
using MMA_Atom_FP8 = cute::MMA_Atom<
    cute::SM90_64x64x16_F8F8F16_SS      // Hopper FP8 Tensor Core
>;
```

**Result**: CUTLASS kernels achieve **95%+ of theoretical Tensor Core peak**.

### Mixed-Precision Support

CUTLASS excels at mixed-precision GEMM:

```cpp
// INT8 input, INT32 accumulation, INT8 output
using Gemm = cutlass::gemm::device::Gemm<
    int8_t,                              // ElementA
    cutlass::layout::RowMajor,
    int8_t,                              // ElementB
    cutlass::layout::ColumnMajor,
    int8_t,                              // ElementC (output)
    cutlass::layout::RowMajor,
    int32_t,                             // ElementAccumulator (internal precision)
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80
>;
```

**Use Case**: Quantized neural network inference (INT8 weights, INT8 activations).

### Fusion via Epilogues

CUTLASS supports **fused epilogues**—operations applied to GEMM output before storing:

```cpp
// GEMM + Bias + ReLU fused
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    float,      // ElementC
    128 / cutlass::sizeof_bits<float>::value,  // Elements per vector load
    float,      // ElementAccumulator
    float       // ElementCompute
>;

using Gemm = cutlass::gemm::device::Gemm<
    // ... matrix types ...
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,  // Threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,    // Warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,     // Instruction shape
    EpilogueOp                               // ← Fused epilogue
>;
```

**Common Fused Operations**:
- Bias addition
- Activation functions (ReLU, GELU)
- Scaling and shifting
- Quantization

**Performance Gain**: Fusing saves a separate kernel launch and memory roundtrip—**1.5-2x faster** than separate operations.

### Performance

**Benchmark: GEMM on A100 GPU (FP16 Tensor Cores)**

| Implementation | Performance (TFLOPS) | Efficiency |
|----------------|---------------------|-----------|
| **cuBLAS** | 312 TFLOPS | 100% (baseline) |
| **CUTLASS (default config)** | 305 TFLOPS | 98% |
| **CUTLASS (tuned)** | 310 TFLOPS | 99.4% |
| **Triton (auto-tuned)** | 280 TFLOPS | 90% |
| **Hand-written CUDA** | 250 TFLOPS | 80% (unless you're an expert) |

**Key Insight**: CUTLASS achieves cuBLAS-level performance because NVIDIA engineers optimize CUTLASS, and cuBLAS itself uses CUTLASS components internally.

### When to Use CUTLASS

**Use CUTLASS When**:

1. **You need custom GEMM variants**: Fused ops, custom data types, irregular sizes
2. **Maximum performance is required**: 95%+ of Tensor Core peak
3. **You're willing to learn C++ templates**: Steep learning curve
4. **Targeting multiple GPU architectures**: CUTLASS handles Volta/Ampere/Hopper differences

**Avoid CUTLASS When**:

1. **Standard operations suffice**: Use cuBLAS directly
2. **Rapid prototyping**: Triton is faster to write
3. **You're not comfortable with C++ metaprogramming**: Template errors are brutal

### CUTLASS vs Triton

| Aspect | CUTLASS | Triton |
|--------|---------|--------|
| **Language** | C++ templates | Python-like DSL |
| **Performance** | 95-100% of cuBLAS | 80-95% of cuBLAS |
| **Learning Curve** | Steep | Gentle |
| **Flexibility** | Maximum | High |
| **Development Time** | Weeks | Days |
| **Use Case** | Production kernels | Research/prototyping |

**Complementary, Not Competing**: Use Triton for experimentation, CUTLASS for production deployment when you need that last 5-10% performance.

### Learning Resources

**Official**:
- [CUTLASS GitHub](https://github.com/NVIDIA/cutlass)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs)
- [NVIDIA Blog: CUTLASS 3.x](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design)

**Tutorials**:
- [CUTLASS Python Interface](https://github.com/NVIDIA/cutlass/tree/main/examples/python)
- [GTC Talks on CUTLASS](https://www.nvidia.com/en-us/on-demand/search/?facet.mimetype[]=event%20session&layout=list&page=1&q=cutlass&sort=relevance)

---

## Modern Frameworks

While CUDA, Triton, and CUTLASS provide the building blocks, specialized frameworks combine algorithmic insights with kernel expertise to achieve breakthrough performance on specific operations. This section covers the most influential modern frameworks.

### FlashAttention: The Attention Revolution

#### The Attention Bottleneck

Standard self-attention in transformers:

```python
# Q, K, V: [batch, heads, seq_len, head_dim]
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)  # [batch, heads, seq_len, seq_len]
attn = softmax(scores, dim=-1)                # [batch, heads, seq_len, seq_len]
output = attn @ V                              # [batch, heads, seq_len, head_dim]
```

**The Problem**:
- Intermediate `scores` and `attn` matrices: **O(N²)** memory
- For N=2048 (GPT-2), FP16, 12 heads: **96 MB** per sample
- For N=8192 (GPT-4), FP16, 32 heads: **4.2 GB** per sample

**Consequence**: Context length limited by memory, not compute.

#### FlashAttention v1: IO-Aware Algorithm

[FlashAttention](https://arxiv.org/abs/2205.14135) (May 2022) by Tri Dao, Daniel Fu, Stefano Ermon, Atri Rudra, and Christopher Ré introduced a revolutionary insight:

**Key Idea**: Don't materialize the N² attention matrix. Instead:
1. **Tile** Q, K, V into blocks that fit in SRAM (shared memory)
2. **Compute** attention incrementally, one block at a time
3. **Fuse** all operations (matmul, softmax, matmul) in a single kernel
4. **Recompute** during backward pass instead of storing intermediate values

**Algorithm (Simplified)**:

```
For each block of queries Q_i:
    Load Q_i into SRAM
    Initialize output O_i = 0

    For each block of keys K_j:
        Load K_j, V_j into SRAM
        Compute S_ij = Q_i @ K_j^T  (in SRAM)
        Compute P_ij = softmax(S_ij)  (in SRAM, online softmax)
        Update O_i += P_ij @ V_j

    Store O_i to HBM
```

**Memory Complexity**: O(N) instead of O(N²)

**Speed**: 2-4x faster (less memory traffic)

**Performance (v1 on A100)**:
- Standard PyTorch: ~15% of GPU utilization
- FlashAttention v1: ~35% of GPU utilization
- Speedup: **2-3x faster**
- Memory: **10-20x reduction**

#### FlashAttention v2: Better Parallelism

[FlashAttention-2](https://arxiv.org/abs/2307.08691) (July 2023) improved parallelism:

**Key Innovations**:

1. **Reduced non-matmul FLOPs**: Optimized softmax computation
2. **Parallelized across sequence length**: Even for a single head, split work across threadblocks
3. **Better work partitioning**: Distribute work between warps within a threadblock

**Result**:
- A100: **50-73% of theoretical max FLOPs**
- Training speed: **225 TFLOPs/s per A100** (72% model FLOPs utilization)
- Speedup vs v1: **~2x**

**Efficiency Comparison**:

| Operation | A100 FP16 Efficiency |
|-----------|---------------------|
| **GEMM (cuBLAS)** | 95-100% |
| **Standard Attention** | 15-20% |
| **FlashAttention v1** | 30-40% |
| **FlashAttention v2** | 50-73% |

#### FlashAttention v3: Hopper Optimization

[FlashAttention-3](https://arxiv.org/abs/2407.08608) (July 2024) targets Hopper (H100) GPUs:

**New Hardware Features Exploited**:

1. **TMA (Tensor Memory Accelerator)**: Asynchronous global memory → shared memory transfers
2. **Warp Specialization**: Some warps load data, others compute
3. **FP8 Tensor Cores**: Lower precision, higher throughput
4. **WGMMA (Warp Group Matrix Multiply-Accumulate)**: Collaborative operations across warp groups

**Techniques**:

```
Warp Specialization:
├── Producer Warps (25% of warps)
│   └── Asynchronously load Q, K, V via TMA
└── Consumer Warps (75% of warps)
    └── Compute attention using WGMMA while data is loading
```

**Overlap**: Computation and data movement happen simultaneously → hide latency.

**FP8 Quantization**:
- Block-wise quantization (quantize each tile independently)
- Incoherent processing (different quantization scales per block)

**Results (H100)**:
- FP16: **740 TFLOPs/s** (75% utilization)
- FP8: **1.2 PFLOPs/s** (>80% utilization)
- Speedup vs FA2: **1.5-2.0x**

#### FlashAttention Impact

**Adoption**:
- PyTorch: `torch.nn.functional.scaled_dot_product_attention()` uses FlashAttention backend
- HuggingFace Transformers: Optional FlashAttention integration
- Triton: Reference implementation available

**Enabled Capabilities**:
- GPT-4: 32K context → 128K context (via efficient attention)
- Claude: 100K token context windows
- Research: Sequence lengths up to 1M tokens

**Ecosystem**:
- Flash-Decoding: Optimized attention for inference (different bottlenecks)
- PagedAttention (vLLM): Memory-efficient KV cache management
- Multi-Query/Grouped-Query Attention variants

### ThunderKittens: Warp-Level Productivity

[ThunderKittens](https://arxiv.org/abs/2410.20399) (Stanford Hazy Research, 2024) takes a different approach: expose hardware details in a structured way.

**Philosophy**: CUDA is too low-level, Triton hides too much. ThunderKittens finds the middle ground.

#### Three-Level Abstraction

**1. Warp-Level (Primary Abstraction)**:

```cpp
#include <kittens.cuh>
using namespace kittens;

__global__ void gemm_kernel(float *A, float *B, float *C) {
    // Register tiles: 16×16 matrices in warp registers
    rt<16, 16, float> tile_A, tile_B, tile_C;

    // Shared memory tiles
    st<16, 16, float> smem_A, smem_B;

    // Load from global → shared (async)
    load_async(smem_A, A);
    load_async(smem_B, B);

    // Wait for loads
    await(smem_A, smem_B);

    // Shared → register
    load(tile_A, smem_A);
    load(tile_B, smem_B);

    // Matrix multiply (Tensor Cores)
    mma(tile_C, tile_A, tile_B);

    // Store result
    store(C, tile_C);
}
```

**Key Points**:
- `rt<16, 16>`: Register tile (lives in warp's registers)
- `st<16, 16>`: Shared memory tile
- 16×16 chosen because it's Tensor Core native size
- Operations are **warp-collective** (all 32 threads cooperate)

**2. Block-Level (Templates)**:

```cpp
// Template for pipelined operations
template<int STAGES>
__global__ void pipelined_gemm(float *A, float *B, float *C) {
    // Multi-stage pipeline (overlap compute and load)
    pipeline<STAGES> pipe;

    for (int k = 0; k < K; k += TILE_K) {
        pipe.producer([&] {
            load_async(smem_A[pipe.stage], A + k);
            load_async(smem_B[pipe.stage], B + k);
        });

        pipe.consumer([&] {
            mma(tile_C, smem_A[pipe.stage], smem_B[pipe.stage]);
        });
    }
}
```

**3. Grid-Level (Launch Helpers)**:

Simplify kernel launches and hide setup overhead.

#### Shared Memory Management

ThunderKittens automatically handles:
- **Bank conflicts**: Layout chosen to avoid conflicts
- **Swizzling**: Optimized patterns for Tensor Core data
- **Alignment**: Proper alignment for coalesced access

**Example**:

```cpp
// Automatic bank conflict avoidance
st<64, 64, float> smem;  // ThunderKittens picks optimal layout

// Manual CUDA equivalent: ~50 lines of swizzling code
```

#### Asynchronous Operations

Built-in support for async patterns:

```cpp
// Load asynchronously while computing
load_async(smem_next, ptr_next);
mma(output, smem_current, weights);  // Compute overlaps with load
await(smem_next);  // Wait only when needed
```

**Hardware**: Uses `cp.async` (Ampere+) and `TMA` (Hopper) instructions automatically.

#### Performance

**Benchmark: GEMM on H100**

| Implementation | TFLOPs | % of cuBLAS | Lines of Code |
|----------------|--------|-------------|---------------|
| **cuBLAS** | 985 | 100% | 1 (API call) |
| **CUTLASS** | 975 | 99% | ~200 |
| **ThunderKittens** | 950 | 96% | ~50 |
| **Triton** | 880 | 89% | ~100 |
| **Naive CUDA** | 300 | 30% | ~30 |

**Trade-off**: ThunderKittens is slightly slower than CUTLASS but requires 75% less code.

#### Use Cases

**When ThunderKittens Shines**:
1. **Exploration**: Faster than CUTLASS, more control than Triton
2. **Tensor Core operations**: Designed around 16×16 tiles
3. **Async patterns**: Built-in pipelining support
4. **Educational**: Great for learning how GPUs work

**Projects Using ThunderKittens**:
- Based (linear attention research)
- Custom attention variants
- Fused multi-head attention

#### Learning ThunderKittens

**Resources**:
- [GitHub](https://github.com/HazyResearch/ThunderKittens)
- [Paper](https://arxiv.org/abs/2410.20399)
- [Examples](https://github.com/HazyResearch/ThunderKittens/tree/main/examples)

### Other Notable Frameworks

#### xFormers (Meta)

Memory-efficient attention variants:

```python
from xformers.ops import memory_efficient_attention

# Automatically selects best kernel (FlashAttention, etc.)
output = memory_efficient_attention(Q, K, V)
```

Features:
- Multiple attention backends (FlashAttention, Memory-Efficient Attention)
- Automatic kernel selection
- Sparse attention patterns

#### vLLM: Production Inference

[vLLM](https://github.com/vllm-project/vllm) optimizes LLM inference:

**Key Innovation**: PagedAttention
- KV cache stored in non-contiguous blocks
- Reduces memory fragmentation
- Enables larger batch sizes

**Kernels**:
- Custom Triton kernels for attention
- Fused RoPE
- Quantization (INT8, FP8, AWQ)

**Performance**: 10-20x higher throughput than naive PyTorch inference.

#### TensorRT-LLM (NVIDIA)

NVIDIA's production inference framework:
- Fused kernels via CUTLASS
- INT8/FP8 quantization
- Multi-GPU tensor parallelism
- Custom optimizations per GPU architecture

**Performance**: State-of-the-art inference speed, especially on NVIDIA hardware.

---

## Optimization Fundamentals

Understanding optimization techniques is crucial whether you're writing CUDA, Triton, or debugging performance. This section covers the core principles that apply across all frameworks.

### Memory Coalescing

**The Problem**: Global memory access is expensive (~400-800 cycles latency).

**The Solution**: When threads in a warp access consecutive memory addresses, the hardware combines them into a single transaction.

#### Coalesced Access Pattern

```cuda
// GOOD: Coalesced access
__global__ void coalesced(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx];  // Thread 0→data[0], Thread 1→data[1], etc.
}
```

**Result**: 32 threads access 32 consecutive floats (128 bytes) → **1 memory transaction** (128-byte cache line).

#### Uncoalesced Access Pattern

```cuda
// BAD: Uncoalesced (strided) access
__global__ void uncoalesced(float *data) {
    int idx = threadIdx.x * 32;  // Thread 0→data[0], Thread 1→data[32], etc.
    float value = data[idx];
}
```

**Result**: 32 threads access 32 non-consecutive addresses → **32 separate transactions**.

**Performance Impact**: **10-32x slower** memory bandwidth.

#### Real-World Example: Matrix Transpose

```cuda
// Naive transpose: Uncoalesced writes
__global__ void transpose_naive(float *input, float *output, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        // Read is coalesced, but write is strided!
        output[col * N + row] = input[row * N + col];
    }
}

// Optimized: Use shared memory to reorganize
__global__ void transpose_optimized(float *input, float *output, int N) {
    __shared__ float tile[32][32 + 1];  // +1 to avoid bank conflicts

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;

    // Coalesced read
    if (row < N && col < N)
        tile[threadIdx.y][threadIdx.x] = input[row * N + col];

    __syncthreads();

    // Swap indices for transposed write
    row = blockIdx.x * 32 + threadIdx.y;
    col = blockIdx.y * 32 + threadIdx.x;

    // Coalesced write
    if (row < N && col < N)
        output[row * N + col] = tile[threadIdx.x][threadIdx.y];
}
```

**Speedup**: Optimized version is **5-10x faster**.

**Key Insight**: Shared memory acts as a "reorder buffer" to convert uncoalesced patterns into coalesced ones.

### Shared Memory Optimization

Shared memory is **100x faster** than global memory but requires careful management.

#### Bank Conflicts

Shared memory is divided into **32 banks**. If multiple threads in a warp access the same bank (but different addresses), access is serialized.

```cuda
__shared__ float data[1024];

// BAD: Bank conflicts
int idx = threadIdx.x * 2;  // Thread 0→bank 0, Thread 1→bank 2, etc. (OK)
float val = data[idx];      // But what if threads access same bank?

// Example of conflict:
// Thread 0 → data[0]  (bank 0)
// Thread 16 → data[16] (bank 16)
// Thread 0 → data[32] (bank 0)  ← Same bank as data[0]!
```

**Solution 1: Padding**

```cuda
__shared__ float data[32][32 + 1];  // +1 avoids conflicts
```

The extra column shifts each row into a different bank alignment.

**Solution 2: Swizzling**

Complex bit permutations to spread accesses (CUTLASS and ThunderKittens do this automatically).

#### Shared Memory Capacity

Each SM has limited shared memory:
- **Volta/Turing**: 64 KB per SM
- **Ampere**: 164 KB per SM (configurable)
- **Hopper**: 228 KB per SM

**Trade-off**: More shared memory per block → fewer concurrent blocks → lower occupancy.

**Example**:

```cuda
// Each block uses 48 KB of shared memory
__shared__ float tile[64][64];  // 64*64*4 = 16 KB
__shared__ float buffer[8192];  // 32 KB
// Total: 48 KB

// Ampere SM: 164 KB total
// 164 / 48 = 3 blocks can fit
// If blocks have 256 threads: 3 * 256 = 768 concurrent threads
```

**Optimization**: Balance shared memory usage vs occupancy.

### Register Pressure

Registers are the **fastest** memory (1-cycle access) but severely limited.

**Limits (per SM)**:
- Ampere: 65,536 registers
- Hopper: 65,536 registers

**Per Thread**:
- If kernel uses 64 registers/thread × 1024 threads = 65,536 → only 1 block can run
- If kernel uses 32 registers/thread × 1024 threads = 32,768 → 2 blocks can run

**Problem**: High register usage → low occupancy → can't hide latency.

**Solutions**:

1. **Reduce register usage**:
   ```cuda
   // BAD: Excessive temporaries
   float temp1 = a + b;
   float temp2 = c + d;
   float temp3 = temp1 * temp2;
   float result = temp3 / e;

   // GOOD: Reuse variables
   float temp = (a + b) * (c + d) / e;
   ```

2. **Compiler directives**:
   ```cuda
   __global__ void __launch_bounds__(256, 4)  // Max 256 threads, min 4 blocks/SM
   kernel(...) {
       // ...
   }
   ```

3. **Register spilling** (last resort):
   Compiler stores some registers to "local memory" (actually global memory) → slow.

### Occupancy Optimization

**Occupancy**: Fraction of maximum concurrent threads actually running.

**Formula**:
```
Occupancy = Active Warps per SM / Maximum Warps per SM
```

**Maximum Warps per SM**:
- Ampere: 64 warps (2048 threads)
- Hopper: 64 warps (2048 threads)

#### Occupancy Limiters

1. **Thread Block Size**:
   - Too small (64 threads): Underutilizes SM
   - Too large (1024 threads): May hit resource limits

2. **Registers per Thread**:
   - 32 regs/thread: Can fit 64 warps (2048 threads)
   - 64 regs/thread: Can fit 32 warps (1024 threads)
   - 128 regs/thread: Can fit 16 warps (512 threads)

3. **Shared Memory per Block**:
   - 16 KB/block: 10 blocks on Ampere (164 KB total)
   - 48 KB/block: 3 blocks
   - 80 KB/block: 2 blocks

#### Measuring Occupancy

**CUDA Occupancy Calculator**:

```cuda
int blockSize;
int minGridSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);

// Launch with suggested blockSize
kernel<<<gridSize, blockSize>>>(...);
```

**NVIDIA Nsight Compute**:

```bash
ncu --metrics smsp__warps_active.avg.pct_of_peak kernel_name
# Reports actual occupancy during execution
```

#### The Occupancy Fallacy

**Common Misconception**: Higher occupancy = faster kernel.

**Reality**: Not always! High occupancy is useful for **latency hiding** (memory-bound kernels). For **compute-bound** kernels, lower occupancy with better per-thread performance can be faster.

**Example**:
- Kernel A: 100% occupancy, 50% memory bandwidth utilization
- Kernel B: 50% occupancy, 90% memory bandwidth utilization
- **Kernel B is faster** (better resource utilization despite lower occupancy)

**Rule**: Aim for **sufficient** occupancy to hide latency, not maximum occupancy.

### Arithmetic Intensity

**Arithmetic Intensity**: FLOPs performed per byte transferred from global memory.

```
Arithmetic Intensity = FLOPs / Bytes Transferred
```

**Roofline Model**:

```
Peak Performance = min(Compute Bound, Memory Bound)
                 = min(Peak FLOPs, Arithmetic Intensity × Memory Bandwidth)
```

**Example (A100)**:
- Peak FP16 Compute: 312 TFLOPS
- Memory Bandwidth: 2 TB/s
- **Breakeven Arithmetic Intensity**: 312 TFLOPS / 2 TB/s = 156 FLOPs/byte

**Implications**:

| Operation | Arithmetic Intensity | Bottleneck |
|-----------|---------------------|------------|
| **Elementwise add** | 0.125 FLOPs/byte | Memory-bound |
| **Matrix multiply (large)** | 64+ FLOPs/byte | Compute-bound (Tensor Cores) |
| **Attention (naive)** | ~0.5 FLOPs/byte | Memory-bound |
| **FlashAttention** | ~10 FLOPs/byte | Better balanced |

**Optimization Strategy**:
1. **Memory-bound**: Reduce memory traffic (kernel fusion, tiling)
2. **Compute-bound**: Use Tensor Cores, better algorithms

### Kernel Fusion

**Concept**: Combine multiple operations into a single kernel to reduce memory roundtrips.

#### Unfused Operations

```python
# Three separate kernels
x = input + bias       # Kernel 1: Read input, write x
y = gelu(x)            # Kernel 2: Read x, write y
output = dropout(y, p) # Kernel 3: Read y, write output

# Total memory traffic: 6 reads + 6 writes = 12 × array size
```

#### Fused Kernel

```python
@triton.jit
def fused_bias_gelu_dropout(input_ptr, bias_ptr, output_ptr, N, p, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < N

    # Load once
    x = tl.load(input_ptr + idx, mask=mask)
    bias = tl.load(bias_ptr + idx, mask=mask)

    # Compute everything
    x = x + bias
    y = gelu(x)
    output = dropout(y, p)

    # Store once
    tl.store(output_ptr + idx, output, mask=mask)

# Total memory traffic: 2 reads + 1 write = 3 × array size
# Speedup: 4x less memory traffic
```

**Common Fusion Patterns**:
- Bias + Activation (e.g., + ReLU)
- LayerNorm + Linear
- Attention + Dropout
- Optimizer updates (AdamW is 10+ operations fused)

### Warp-Level Programming

Modern GPUs provide **warp-level primitives** for efficient cooperation within a warp.

#### Warp Shuffle

Exchange data between threads in a warp without shared memory:

```cuda
__device__ float warp_reduce_sum(float value) {
    // Threads exchange values via shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;  // Thread 0 has the sum
}

// Example: Sum reduction across warp
__global__ void reduce_kernel(float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (idx < N) ? input[idx] : 0.0f;

    // Reduce within warp
    value = warp_reduce_sum(value);

    // First thread in each warp writes result
    if (threadIdx.x % 32 == 0) {
        atomicAdd(output, value);
    }
}
```

**Advantage**: No shared memory, no synchronization, **much faster**.

#### Warp-Level Matrix Operations

**Cooperative Groups** (CUDA 11+):

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void warp_gemm(float *A, float *B, float *C) {
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());

    // Warp-collective matrix multiply
    // (Using WMMA or manual cooperation)
}
```

### Data Reuse via Tiling

**Problem**: Global memory bandwidth is limited.

**Solution**: Load data once into shared memory, reuse many times.

#### Matrix Multiplication Tiling

```
# Naive: Each thread loads N elements from A and N from B
# Memory per output element: 2N loads

# Tiled: Load tiles into shared memory, all threads reuse
# Memory per output element: 2N / TILE_SIZE loads
```

**Example**:
- Matrix size: 4096 × 4096
- Tile size: 64 × 64
- **Naive**: 4096 loads per output element
- **Tiled**: 4096 / 64 = 64 loads per output element
- **Speedup**: 64x reduction in global memory traffic

**Implementation Pattern**:

```cuda
for (int tile = 0; tile < K; tile += TILE_SIZE) {
    // Load tile into shared memory (coalesced)
    load_tile(smem_A, A + tile);
    load_tile(smem_B, B + tile);
    __syncthreads();

    // Compute using shared memory (fast)
    compute(output, smem_A, smem_B);
    __syncthreads();
}
```

### Asynchronous Operations

Modern GPUs can overlap computation and memory transfers.

#### Async Copy (Ampere+)

```cuda
// Async copy from global to shared memory
__pipeline_memcpy_async(smem_ptr, gmem_ptr, size);
__pipeline_commit();

// Do computation while copy happens
compute_something_else();

// Wait for copy to complete
__pipeline_wait_prior(0);
__syncthreads();
```

**Benefit**: Hide memory latency with computation.

#### Double Buffering

```cuda
__shared__ float buffer[2][TILE_SIZE];
int current = 0, next = 1;

// Load first tile
load_async(buffer[current], ptr);

for (int i = 0; i < N; i++) {
    // Load next tile
    load_async(buffer[next], ptr + i);

    // Compute current tile (overlaps with load)
    compute(buffer[current]);

    // Swap buffers
    current = 1 - current;
    next = 1 - next;
}
```

**Speedup**: 30-50% by overlapping loads and computation.

---

## Advanced Patterns

Beyond the fundamentals, advanced patterns unlock the highest performance for complex operations.

### Multi-Stage Pipelines

**Concept**: Break kernel into producer/consumer stages that run concurrently.

#### Software Pipelining

```cuda
// 3-stage pipeline: Load → Compute → Store
__global__ void pipelined_kernel(float *input, float *output, int N) {
    __shared__ float buffer[3][TILE_SIZE];

    // Prime the pipeline
    load_async(buffer[0], input);
    commit();

    load_async(buffer[1], input + TILE_SIZE);
    compute(buffer[0]);
    commit();

    // Main loop: All stages active
    for (int i = 2; i < N / TILE_SIZE; i++) {
        load_async(buffer[i % 3], input + i * TILE_SIZE);
        compute(buffer[(i-1) % 3]);
        store(output + (i-2) * TILE_SIZE, buffer[(i-2) % 3]);
        commit();
    }

    // Drain the pipeline
    compute(buffer[(N-1) % 3]);
    store(output + (N-2) * TILE_SIZE, buffer[(N-2) % 3]);
    store(output + (N-1) * TILE_SIZE, buffer[(N-1) % 3]);
}
```

**Key**: Load, compute, and store happen **simultaneously** for different tiles.

#### Hopper Warp Specialization

FlashAttention-3 uses this pattern:

```cuda
// Simplified concept
__global__ void warp_specialized_kernel(...) {
    int warp_id = threadIdx.x / 32;

    if (warp_id < 2) {
        // Producer warps: Load data via TMA
        while (more_data) {
            tma_load_async(smem, gmem);
            notify_consumers();
        }
    } else {
        // Consumer warps: Compute using WGMMA
        while (more_work) {
            wait_for_data();
            wgmma_async(output, smem_A, smem_B);
        }
    }
}
```

**Benefit**: Producers and consumers run **truly concurrently**—100% GPU utilization.

### Tensor Core Advanced Patterns

#### Mixed-Precision Accumulation

```cuda
// FP16 inputs, FP32 accumulation (standard)
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;  // ← FP32

wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

**Why**: FP16 multiply is fast, but accumulation in FP16 loses precision. FP32 accumulation costs minimal extra time.

#### Sparse Tensor Cores (Ampere+)

Ampere introduced **2:4 structured sparsity**: For every 4 values, 2 must be zero.

```cuda
// Sparse matrix multiply (2:4 sparsity)
// 50% of weights are zero → 2x faster
wmma::mma_sync_sparse(c_frag, a_frag, b_frag, c_frag);
```

**Use Case**: Pruned neural networks (50% weights zeroed, minimal accuracy loss).

**Speedup**: **2x** for same accuracy (if model supports sparsity).

### Flash Decoding

Attention during **inference** has different bottlenecks than training.

**Problem**:
- Training: Large batch sizes, many queries (parallelism across batch/queries)
- Inference: Small batch (often 1), single new query, very long KV cache

**Bottleneck**: Can't parallelize across sequence length for a single query.

**Flash Decoding Solution** (2023):

1. **Partition KV cache** across threadblocks
2. Each block computes partial attention for its KV segment
3. **Combine partials** using scaled softmax reduction

**Algorithm**:

```
For query q:
    Partition KV cache into blocks: KV_1, KV_2, ..., KV_N

    In parallel:
        Block 1: local_attn_1 = attention(q, KV_1)
        Block 2: local_attn_2 = attention(q, KV_2)
        ...

    Combine: global_attn = combine(local_attn_1, local_attn_2, ...)
```

**Speedup**: Up to **8x faster** for long-context inference.

### Page Attention (vLLM)

**Problem**: KV cache for each sequence must be contiguous in memory → fragmentation → wasted memory.

**Solution**: Store KV cache in **non-contiguous blocks** (like virtual memory paging).

**Mechanism**:

```
Traditional KV Cache:
    Sequence 1: [──────────────────] (contiguous, 10,000 tokens allocated)
    Sequence 2: [──────────────────] (contiguous, 10,000 tokens allocated)
    # Wastes memory if sequences are shorter

PagedAttention:
    Sequence 1: [Block 1]─[Block 5]─[Block 8]  (non-contiguous)
    Sequence 2: [Block 2]─[Block 3]─[Block 7]
    # Blocks allocated on-demand
```

**Kernel Implementation**:
- Attention kernel takes block table (maps logical position → physical block)
- Reads KV values from non-contiguous locations
- Slight overhead (~5%) but enables **10x higher throughput** via better memory utilization

### Quantization Kernels

Lower precision = faster compute + less memory.

#### INT8 Quantization

**Per-Tensor Quantization**:

```cuda
// Quantize FP16 → INT8
__global__ void quantize(half *input, int8_t *output, float scale, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float value = __half2float(input[idx]);
        output[idx] = (int8_t)(value * scale);  // Scale and round
    }
}

// INT8 GEMM (using Tensor Cores)
// Compute in INT32, dequantize to FP16
```

**Speedup**: **2x faster** than FP16 (on GPUs with INT8 Tensor Cores).

#### FP8 Quantization (Hopper)

```cuda
// Block-wise quantization (FlashAttention-3)
__global__ void fp8_quantize_blockwise(half *input, fp8 *output, float *scales) {
    __shared__ float block_max;

    // Find max in block
    float local_max = find_max(input);
    block_max = warp_reduce_max(local_max);

    // Compute scale
    float scale = 448.0f / block_max;  // FP8 max value
    scales[blockIdx.x] = scale;

    // Quantize
    output[idx] = (fp8)(input[idx] * scale);
}
```

**Hopper FP8 Tensor Cores**: **2x faster** than FP16, **4x faster** than FP32.

---

## Real-World Examples

This section walks through concrete examples, showing how theory translates to practice.

### Example 1: Optimized Matrix Multiplication

**Evolution from Naive to Optimized CUDA**:

#### Naive Implementation

```cuda
__global__ void matmul_naive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

**Performance**: ~15 GFLOPS (0.5% of A100 peak)

**Problems**:
- Uncoalesced memory access for B (column-major reads)
- No data reuse (each element loaded N times)
- No Tensor Core usage

#### Tiled with Shared Memory

```cuda
#define TILE_SIZE 32

__global__ void matmul_tiled(float *A, float *B, float *C, int N) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles (coalesced)
        if (row < N && t * TILE_SIZE + threadIdx.x < N)
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < N)
            tile_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}
```

**Performance**: ~500 GFLOPS (16% of peak)

**Improvements**:
- ✅ Coalesced memory access
- ✅ Data reuse via shared memory (TILE_SIZE × reuse)
- ❌ Still no Tensor Cores

#### Tensor Core Version (WMMA)

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void matmul_wmma(half *A, half *B, float *C, int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < K; k += 16) {
        int aRow = warpM * 16;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * 16;

        // Load 16×16 tiles
        wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
        wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

        // Matrix multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store result
    int cRow = warpM * 16;
    int cCol = warpN * 16;
    wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
}
```

**Performance**: ~5,000 GFLOPS (16% of peak, but FP16!)

**Improvements**:
- ✅ Tensor Cores (16×16×16 = 4,096 ops per `mma_sync`)

#### Production-Quality (CUTLASS-style)

For production, add:
- Multi-stage pipelining (async loads)
- Register blocking (higher arithmetic intensity)
- Warp tile scheduling
- Epilogue fusion (+ bias, activation)

**Result**: **300,000 GFLOPS** (95% of A100 FP16 Tensor Core peak)

**Full Code**: See [CUTLASS GEMM examples](https://github.com/NVIDIA/cutlass/tree/main/examples)

### Example 2: Fused Layer Normalization

Layer norm is critical in transformers but memory-bound.

#### Unfused PyTorch

```python
# Three separate kernels
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)
output = (x - mean) / torch.sqrt(var + eps)
```

**Memory Traffic**: 5 passes over data (mean, variance, subtract, divide, sqrt)

#### Fused Triton Kernel

```python
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    X, Y, Weight, Bias,
    stride, N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # Load input row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X + cols, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / N

    # Compute variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N

    # Normalize
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd

    # Apply weight and bias
    weight = tl.load(Weight + cols, mask=mask)
    bias = tl.load(Bias + cols, mask=mask)
    output = x_norm * weight + bias

    # Store
    tl.store(Y + cols, output, mask=mask)
```

**Memory Traffic**: 2 passes (read input + weight/bias, write output)

**Speedup**: **2-3x faster** than unfused PyTorch

**Real Usage**: PyTorch's `F.layer_norm()` uses fused kernels internally (via Apex or native).

### Example 3: Custom Activation Function

**Task**: Implement GeLU (Gaussian Error Linear Unit) efficiently.

**Math**: `GeLU(x) = x * Φ(x)` where Φ is cumulative distribution function.

**Approximation**: `GeLU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

#### Triton Implementation

```python
@triton.jit
def gelu_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X + offs, mask=mask)

    # GeLU approximation
    c1 = 0.5
    c2 = 0.7978845608  # sqrt(2/pi)
    c3 = 0.044715

    x3 = x * x * x
    tanh_arg = c2 * (x + c3 * x3)
    tanh_out = tl.libdevice.tanh(tanh_arg)
    gelu = c1 * x * (1.0 + tanh_out)

    tl.store(Y + offs, gelu, mask=mask)

def gelu(x: torch.Tensor):
    output = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, output, N, BLOCK_SIZE=1024)
    return output
```

**Performance**: **Match or beat PyTorch's built-in GeLU** (which is also fused).

---

## Current Best Practices (2024-2025)

### When to Use What

**Decision Tree**:

```
Need custom GPU kernel?
│
├─ Standard operation (GEMM, conv, attention)?
│  └─ Use cuBLAS / cuDNN / FlashAttention
│
├─ Simple fusion (bias + activation)?
│  └─ Use Triton (fastest development)
│
├─ Complex fusion or novel operation?
│  ├─ Prototype in Triton
│  └─ Optimize in CUTLASS (if need 95%+ efficiency)
│
├─ Warp-level patterns (tiling, async)?
│  └─ Use ThunderKittens
│
└─ Absolute maximum performance?
   └─ Hand-written CUDA + Tensor Cores
```

### Framework Recommendations by Use Case

| Use Case | First Choice | Second Choice | Production |
|----------|-------------|---------------|------------|
| **Research prototype** | Triton | ThunderKittens | Triton + tuning |
| **Simple fusion** | Triton | PyTorch JIT | Triton |
| **Novel attention variant** | FlashAttention API | Triton | CUTLASS |
| **Quantization** | Triton | CUTLASS | CUTLASS |
| **Custom GEMM** | CUTLASS | Triton | CUTLASS |
| **Inference optimization** | TensorRT-LLM | vLLM | TensorRT-LLM |

### Performance Optimization Checklist

**Before Optimizing**:
- [ ] Profile with `nsys` (NVIDIA Nsight Systems) to find bottlenecks
- [ ] Measure baseline: `ncu` (NVIDIA Nsight Compute) for kernel metrics
- [ ] Set target: What % of peak performance is theoretically possible?

**Memory Optimization**:
- [ ] Ensure coalesced global memory access
- [ ] Use shared memory for data reuse
- [ ] Avoid bank conflicts in shared memory
- [ ] Minimize global memory roundtrips (fusion)
- [ ] Use async copy (`cp.async`, TMA) to overlap loads and compute

**Compute Optimization**:
- [ ] Use Tensor Cores for matrix operations (FP16/FP8)
- [ ] Minimize warp divergence (uniform control flow)
- [ ] Reduce register pressure (check occupancy)
- [ ] Fuse operations to increase arithmetic intensity

**Profiling Metrics** (via `ncu`):

| Metric | Good Target | Meaning |
|--------|-------------|---------|
| **SM Efficiency** | >80% | SMs are busy |
| **Occupancy** | >50% | Enough warps to hide latency |
| **Memory Throughput** | >70% of peak | Using available bandwidth |
| **Tensor Core Utilization** | >80% | For GEMM-heavy kernels |

### Common Pitfalls

**1. Over-Optimizing Memory-Bound Kernels**:
   - Elementwise operations are memory-bound
   - Compute optimizations won't help if memory is the bottleneck
   - **Solution**: Fuse operations to reduce memory traffic

**2. Ignoring Tensor Cores**:
   - Modern GPUs are 10-20x faster with Tensor Cores
   - **Solution**: Use FP16/FP8, align to 16×16 tiles

**3. Too Many Small Kernels**:
   - Kernel launch overhead: ~5-10 μs per kernel
   - **Solution**: Fuse kernels

**4. Not Profiling**:
   - Guessing where the bottleneck is → wasted effort
   - **Solution**: Always profile first

**5. Premature Optimization**:
   - Writing CUDA when Triton would suffice
   - **Solution**: Prototype in high-level tools, optimize only if needed

### Debugging Tips

**Triton Debugging**:

```python
# Print intermediate values
@triton.jit
def debug_kernel(...):
    value = tl.load(...)
    tl.device_print("value:", value)  # Prints from GPU!
```

**CUDA Debugging**:

```bash
# Compute sanitizer (detects out-of-bounds, race conditions)
compute-sanitizer --tool memcheck ./program

# cuda-gdb (GPU debugger)
cuda-gdb ./program
(cuda-gdb) break kernel_name
(cuda-gdb) run
```

**Performance Debugging**:

```bash
# Profile entire application
nsys profile --stats=true ./program

# Detailed kernel metrics
ncu --set full --target-processes all ./program

# FlashAttention-specific profiling
ncu --metrics smsp__sass_thread_inst_executed_op_ffma_pred_on.sum ./fa_benchmark
```

### Future Trends (2025+)

**Hardware Trends**:
- **FP4/INT4**: Ultra-low precision for inference
- **Sparse Tensor Cores**: Beyond 2:4 sparsity
- **Larger shared memory**: 512 KB+ per SM
- **More asynchrony**: Better overlap of compute and memory

**Software Trends**:
- **Auto-tuning frameworks**: AI-driven kernel optimization
- **Cross-platform DSLs**: Portable Triton (NVIDIA, AMD, Intel)
- **Compiler fusion**: PyTorch 3.0 more aggressive auto-fusion
- **Specialized libraries**: FlashAttention for vision, sparse transformers, etc.

**Research Directions**:
- **Kernel learning**: Neural networks generating kernels
- **Mixed sparsity**: Combining structured and unstructured sparsity
- **Ultra-long context**: Kernels for 1M+ token sequences
- **Multi-modal kernels**: Fused text+image+audio processing

---

## Sources

### CUDA and GPU Architecture

- [CUDA Refresher: Reviewing the Origins of GPU Computing | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/)
- [Inside the Programming Evolution of GPU Computing | NVIDIA Technical Blog](https://developer.nvidia.com/blog/inside-the-programming-evolution-of-gpu-computing/)
- [CUDA C++ Programming Guide — CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- [Understanding NVIDIA CUDA Cores: A Comprehensive Guide](https://www.wevolver.com/article/understanding-nvidia-cuda-cores-a-comprehensive-guide)
- [How the hell are GPUs so fast? | Towards Data Science](https://towardsdatascience.com/how-the-hell-are-gpus-so-fast-a-e770d74a0bf/)

### Triton

- [Introducing Triton: Open-source GPU programming for neural networks | OpenAI](https://openai.com/index/triton/)
- [GitHub - triton-lang/triton: Development repository for the Triton language and compiler](https://github.com/triton-lang/triton)
- [Simplifying CUDA kernels with Triton: A Pythonic Approach to GPU Programming | by Arun Jith A | Medium](https://arunjitha.medium.com/simplifying-cuda-kernels-with-triton-a-pythonic-approach-to-gpu-programming-79bb7121e974)
- [Democratizing AI Accelerators and GPU Kernel Programming using Triton - Red Hat Emerging Technologies](https://next.redhat.com/2024/11/07/democratizing-ai-accelerators-and-gpu-kernel-programming-using-triton/)

### FlashAttention

- [[2205.14135] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [[2307.08691] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [[2407.08608] FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608)
- [GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention](https://github.com/Dao-AILab/flash-attention)
- [The Evolution of Flash Attention: Revolutionizing Transformer Efficiency | by Saiii | Medium](https://medium.com/@sailakkshmiallada/the-evolution-of-flash-attention-revolutionizing-transformer-efficiency-8a039918d507)

### ThunderKittens

- [GitHub - HazyResearch/ThunderKittens: Tile primitives for speedy kernels](https://github.com/HazyResearch/ThunderKittens)
- [[2410.20399] ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/abs/2410.20399)
- [ThunderKittens: Simple, Fast, and Adorable AI Kernels](https://arxiv.org/html/2410.20399v1)

### CUTLASS

- [GitHub - NVIDIA/cutlass: CUDA Templates and Python DSLs for High-Performance Linear Algebra](https://github.com/NVIDIA/cutlass)
- [CUTLASS: Fast Linear Algebra in CUDA C++ | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design | NVIDIA Technical Blog](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design)

### cuDNN

- [CUDA Deep Neural Network (cuDNN) | NVIDIA Developer](https://developer.nvidia.com/cudnn)
- [[1410.0759] cuDNN: Efficient Primitives for Deep Learning](https://ar5iv.labs.arxiv.org/html/1410.0759)
- [What is cuDNN?](https://blog.roboflow.com/what-is-cudnn/)

### Optimization Techniques

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Using Shared Memory in CUDA C/C++ | NVIDIA Technical Blog](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [How to Improve CUDA Kernel Performance with Shared Memory Register Spilling | NVIDIA Technical Blog](https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/)
- [In CUDA, what is memory coalescing, and how is it achieved? - Stack Overflow](https://stackoverflow.com/questions/5041328/in-cuda-what-is-memory-coalescing-and-how-is-it-achieved)
- [Faster Models with Graph Fusion: How Deep Learning Frameworks Optimize Your Computation | Practical ML](https://arikpoz.github.io/posts/2025-05-07-faster-models-with-graph-fusion-how-deep-learning-frameworks-optimize-your-computation/)

### Tensor Cores

- [Programming Tensor Cores in CUDA 9 | NVIDIA Technical Blog](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [NVIDIA Tensor Core Evolution: From Volta To Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
- [NVIDIA Hopper Architecture In-Depth | NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [Dissecting Tensor Cores via Microbenchmarks](https://arxiv.org/pdf/2206.02874)

---

**Document Statistics**:
- Lines: ~2,100
- Sections: 11 major sections
- Topics covered: CUDA history, GPU architecture, Triton, CUTLASS, FlashAttention, ThunderKittens, optimization techniques, real-world examples
- Code examples: 40+ snippets across CUDA, Triton, Python

**Last Updated**: November 2025
