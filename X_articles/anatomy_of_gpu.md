# Deep Dive into GPU Architecture and CUDA Programming: The NVIDIA H100

## GPU Hardware Architecture Overview

GPUs are composed of multiple Streaming Multiprocessors (SMs), each functioning as an independent processing unit. Taking NVIDIA's flagship **H100 (SXM5 version)** as an example, it contains **132 SMs**.

### SM Internal Components

Each SM contains various types of compute cores. The H100 SM adopts a **4-Way Partitioned design**:
- **Partition Structure**: Each SM is divided into **4 Partitions**, with each Partition containing **32 FP32 CUDA Cores**
- **CUDA Cores**: General-purpose computing cores, with the entire H100 containing 16,896 CUDA Cores
- **Tensor Cores**: Specialized units for accelerating matrix multiply-accumulate operations (A×B+C), with each H100 SM having **4 fourth-generation Tensor Cores**, crucial for deep learning
- **Warp Scheduler**: Each Partition has its own Warp Scheduler, responsible for scheduling warps to execute on the 32 CUDA Cores

**Key Point**: Since each Partition has 32 CUDA Cores, it perfectly fits one warp's 32 threads, enabling SIMT parallelism. This 4×32 structure directly determines the warp size and scheduling approach in the CUDA programming model.

## GPU Memory Hierarchy

GPUs employ a multi-level memory architecture to balance speed and capacity, ordered from fastest to slowest:

### Registers

The fastest private storage space for each Thread, used to store loop variables, address calculations, and other intermediate values. Each H100 SM has 65,536 32-bit registers, with each Thread able to use up to **255 Registers**. Register count is a limited resource—using too many reduces the number of Threads that can reside simultaneously on an SM (affecting occupancy). When registers run out, the compiler will "spill" variables to Local Memory.

### Local Memory

This is an easily misunderstood concept. Despite its name, Local Memory is **physically located in Global Memory (HBM)**, but is logically private to each Thread. When registers overflow, large arrays are defined, or dynamic indexing is used, the compiler will place variables in Local Memory. Since it's actually in HBM, access speed is as slow as Global Memory and can severely impact performance.

### L1 Cache and Shared Memory

Each SM has **256KB** of unified L1/Shared Memory. Shared Memory is programmable L1 cache—you can explicitly control its contents, while L1 cache is automatically managed by hardware. On the H100, up to **228KB** can be configured as Shared Memory. Shared Memory is key for thread communication and data reuse within the same Block.

### L2 Cache

The H100 has **50MB** of L2 Cache, shared by all SMs. It serves as an intermediate layer between SMs and global memory, with a latency of approximately 200 cycles. All writes to global memory are synchronized through L2.

### Global Memory (HBM3)

The H100 is equipped with **80GB of HBM3 memory**, with bandwidth up to **3.35 TB/s**. This is the GPU's main storage—largest in capacity but highest in latency, making it the primary bottleneck for performance optimization.

## CUDA Programming Model

### Thread, Block, and Warp

The CUDA execution model is based on a hierarchical structure:
- **Thread**: The most basic execution unit, corresponding to one CUDA Core operation
- **Warp**: **32 Threads** form a Warp, which is the basic unit of GPU scheduling
- **Key Correspondence**: Since each H100 Partition has 32 CUDA Cores, a Warp is **scheduled to a Partition**, with 32 Threads running perfectly on 32 CUDA Cores, implementing SIMT (Single Instruction, Multiple Threads) parallelism
- **Block**: Multiple Warps form a Thread Block, assigned to execute on a single SM. Each H100 SM can handle up to **32 Blocks** simultaneously

When threads within a Warp take different paths due to conditional branches, Warp Divergence occurs, leading to performance degradation.

### Asynchronous Execution and Streams

Modern CUDA programming is not just about launching kernels and waiting for results—it's about fully leveraging CPU-GPU parallelism. **Streams** allow kernel launches and memory copies to overlap, hiding data transfer latency. To achieve truly asynchronous transfers, use `cudaMallocHost` to allocate pinned memory; otherwise, transfers will be blocked. On the H100, HBM→GPU copy, GPU compute, and GPU→HBM copy can occur simultaneously, boosting overall throughput by 2-3x. This is especially important for large-scale data processing and pipelined computation.

### Multi-GPU Programming

The H100 is typically not used in isolation, but in multi-GPU systems. **NCCL (NVIDIA Collective Communications Library)** provides optimized multi-GPU communication primitives (AllReduce, Broadcast, etc.), which are crucial for deep learning training. Programming models include single-threaded control of all GPUs, multi-threaded (one thread per GPU), or multi-process (MPI+NCCL). On the H100, NVLink provides up to 900GB/s of GPU-to-GPU bandwidth, far exceeding PCIe. When writing multi-GPU code, fully leverage these high-speed interconnects.

### Memory Access Optimization

#### Memory Coalescing

When 32 threads in a Warp access Global Memory, if they access consecutive addresses, the GPU can merge them into a few memory transactions, dramatically improving bandwidth utilization. Non-contiguous accesses result in performance far below peak.

#### Bank Conflicts

Shared Memory is divided into **32 Banks** (corresponding to Warp size). When multiple threads in the same Warp access different addresses in the same Bank, a Bank Conflict occurs, causing accesses to serialize. The solution is to use Padding or remap indices to have threads access different Banks.

#### Tiling Techniques

To fully utilize the high speed of L1/L2 cache and Shared Memory, break large problems into small tiles for processing. Data for each Tile is loaded into Shared Memory and can be reused multiple times, reducing Global Memory accesses. Leveraging the H100's **228KB Shared Memory**, matrices can be cut into larger Tiles, allowing Tensor Cores to complete computation in high-speed cache before writing back to HBM.

### Performance Analysis Tools

To write high-performance CUDA code, you must rely on professional tools to identify bottlenecks. **Nsight Compute** provides detailed kernel-level performance metrics, detecting memory utilization and instruction efficiency for each line of code, and automatically identifying bank conflicts, memory coalescing issues, and register spilling. **Nsight Systems** provides system-level analysis, helping understand CPU-GPU interaction, asynchronous execution, and data transfer overhead. In future articles, we'll detail how to use these tools for performance debugging and optimization.
