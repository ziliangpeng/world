# GPU Connectivity Technologies - Comprehensive Overview

## Layered Taxonomy

### **Layer 1: Physical/Link Layer** (Die-to-die, GPU-to-GPU on same board/node)

#### Proprietary High-Speed Interconnects:
- **NVLink** (NVIDIA) - Dedicated GPU-to-GPU links
  - NVLink 1.0 (2016, P100): 160 GB/s (4 links @ 20 GB/s bidirectional)
  - NVLink 2.0 (2017, V100): 300 GB/s (6 links @ 25 GB/s bidirectional)
  - NVLink 3.0 (2020, A100): 600 GB/s (12 links @ 50 GB/s bidirectional)
  - NVLink 4.0 (2022, H100): 900 GB/s (18 links @ 50 GB/s bidirectional)
  - NVLink 5.0 (2024, B100): 1.8 TB/s (18 links @ 100 GB/s bidirectional)
- **XGMI/Infinity Fabric** (AMD) - AMD's GPU interconnect
  - MI200/MI250 (XGMI-2): 50 GB/s unidirectional per link (100 GB/s bidirectional).
    - MI250 intra-package: 400 GB/s bidirectional between GCDs (4 links).
    - MI250 inter-package: up to 8 external links on an MI250X.
  - MI300 (XGMI-3): 64 GB/s unidirectional per link (~48 GB/s effective). An MI300X has 8 links, totaling 512 GB/s.
- **Xe Link** (Intel) - Intel GPU interconnect for data center GPUs
  - Ponte Vecchio: ~26.5 GB/s per link unidirectional (53 Gbps, 90 Gb/s SerDes)
  - 16 Xe Links total per 2-stack GPU (8 links per stack)
  - Multi-GPU configs: 6 links (159 GB/s) or 2 links (53 GB/s) between GPUs
- **NVSwitch** (NVIDIA) - Switch fabric connecting multiple NVLink GPUs
  - NVSwitch 1.0 (2018, V100): Connects up to 16 GPUs in a node.
  - NVSwitch 2.0 (2020, A100): Connects up to 8 GPUs in a node.
  - NVSwitch 3.0 (2022, H100): Connects up to 8 GPUs in a node.
  - NVSwitch 4.0 (2024, B200): Scales to 72 GPUs in a single rack (GB200 NVL72).
  - Enables all-to-all GPU communication at full NVLink speed
  - **Note on Topology**: In smaller systems, NVLink connects GPUs directly. In larger systems, GPUs connect via NVLink to the NVSwitch, which creates a switched fabric for all-to-all communication.
- **NeuronLink** (AWS) - Proprietary interconnect for AWS Trainium/Inferentia chips.
- **ICI (Inter-Chip Interconnect)** (Google) - Proprietary physical link for Google TPUs, also used to build the Layer 3 network fabric in TPU Pods.
- **IPU-Link** (Graphcore) - Proprietary high-speed link for connecting IPU (Intelligence Processing Unit) chips.
- **SwarmX** (Cerebras) - Fabric for connecting multiple CS-2 Wafer-Scale Engine systems together.
- **On-Chip Network** (Tenstorrent) - Wormhole processors include high-speed Ethernet ports directly on-chip, allowing them to be connected in a grid without external switches.
- **Deterministic Interconnect** (Groq) - A custom, high-bandwidth link for Groq's LPU chips where data movement is statically scheduled by the compiler, eliminating the need for a dynamic network fabric.
- **Dojo Training Tile** (Tesla) - A 2D mesh of custom D1 chips with extremely high-bandwidth, low-latency interconnects between adjacent chips on the tile.
- **RDU Interconnect** (SambaNova) - A custom, high-speed interconnect for linking multiple RDU (Reconfigurable Dataflow Unit) chips.
- **MTIA Interconnect** (Meta) - A custom physical interconnect for Meta's line of MTIA (Meta Training and Inference Accelerator) chips.

#### Standard Interconnects (Primarily CPU-GPU):
- **PCIe** (PCI Express) - Universal standard, connects GPU to CPU and other PCIe devices
  - PCIe 1.0 (2003): ~4 GB/s per x16 slot
  - PCIe 2.0 (2007): ~8 GB/s per x16 slot
  - PCIe 3.0 (2010): ~16 GB/s per x16 slot
  - PCIe 4.0 (2017): ~32 GB/s per x16 slot
  - PCIe 5.0 (2019): ~64 GB/s per x16 slot
  - PCIe 6.0 (2022): ~128 GB/s per x16 slot
  - PCIe 7.0 (Spec target 2025): ~256 GB/s per x16 slot (in development)


**A Note on Aggregate vs. Per-Link Bandwidth:** A GPU uses only **one** PCIe x16 slot for CPU communication (~64 GB/s with PCIe 5.0). While this may be faster than a single proprietary link (e.g., one NVLink 4.0 link is 50 GB/s), a high-end GPU has many such links. For example, an NVIDIA H100 has **18** NVLink 4.0 links, providing a total of **900 GB/s** for GPU-to-GPU communication. It is this massive difference in *aggregate* bandwidth that makes proprietary interconnects crucial for multi-GPU performance.

---

### **Layer 2: Intra-Node Communication** (Within a single server)

**Note on Layers:** Layer 1 is the **connectivity hardware** (e.g., NVLink, PCIe). Layer 2 consists of the **drivers, specialized orchestration hardware (e.g., DMA engines for GPUDirect), and software libraries** that manage data transfers and collective operations across the Layer 1 hardware.

#### Public Standards:
- **CXL** (Compute Express Link) - Emerging cache-coherent interconnect (announced 2019).
  - Built on PCIe physical layer
  - Enables CPU-GPU memory coherency
  - **Performance & Adoption:**
    - **Bandwidth:** Determined by the underlying PCIe physical layer (e.g., ~64 GB/s on PCIe 5.0 x16).
    - **Latency:** While higher than local DRAM (~200-300ns vs. <100ns), it is orders of magnitude faster than NVMe storage.
    - **Outlook:** This performance trade-off is enabling large-scale memory expansion, and CXL is now seen as the essential successor to PCIe for memory-centric data center workloads.

#### Hardware/Driver Level:
- **NVIDIA CUDA Platform Features**
  - **GPUDirect:** Main goal is to bypass the CPU/system memory. This reduces latency, increases bandwidth, and frees up resources.
    - **P2P:** Direct memory transfers between GPUs (via NVLink or PCIe).
    - **Storage:** Direct data transfers between storage (NVMe) and a GPU (via PCIe).
    - **RDMA:** Direct data transfers between a GPU and a network card (via PCIe).
  - **UVM (Unified Virtual Memory):** Provides a unified address space and automatic page migration between CPU and GPU.
- **AMD ROCm Platform Features**
  - **Shared Virtual Memory:** Provides a unified address space for CPU/GPU, analogous to UVM.
  - **P2P Transfers:** Enables direct data transfers between AMD GPUs over Infinity Fabric or PCIe.
- **Intel oneAPI Features**
  - **Shared Virtual Memory (SVM):** Provides a unified address space for CPU/GPU within the oneAPI model.
- **AWS Neuron Platform Features**
  - **Neuron Driver:** Manages the Trainium/Inferentia accelerators.
  - **P2P Transfers:** Enables direct data transfers between accelerators over NeuronLink.

#### Software/Library Level:

##### Collective Communication Libraries:
These libraries focus specifically on multi-accelerator collective operations (AllReduce, Broadcast, etc.) and work on top of existing compute kernels.

**Proprietary:**
- **NCCL** (NVIDIA Collective Communications Library)
  - **Topology Awareness:** Automatically detects hardware topology (NVLink, PCIe, NUMA) to optimize communication paths.
  - **Collective Operations:** Implements standard collective primitives used in distributed training:
    - **AllReduce:** Reduces data across all GPUs and distributes result to all (most common for gradient synchronization).
    - **Broadcast:** Sends data from one GPU to all others.
    - **Reduce:** Reduces data from all GPUs to a single destination GPU.
    - **AllGather:** Gathers data from all GPUs and distributes complete result to all.
    - **ReduceScatter:** Reduces data across all GPUs and scatters results (each GPU gets a portion).
    - **AllToAll:** Each rank sends different data to every other rank (k ranks exchange k×N values).
    - **Gather:** Gathers data from all GPUs to root rank only (unlike AllGather which distributes to all).
    - **Scatter:** Root rank distributes different chunks to each GPU.
  - **Algorithm Selection:** Dynamically chooses the most efficient collective algorithm based on the detected topology and message size.
    - **Ring Algorithm:** Linear latency, 100% bandwidth utilization. Optimal for large messages and ring topologies.
    - **Tree Algorithm:** Logarithmic latency (up to 180x improvement at scale), 95% bandwidth using dual binary trees. Better for small/medium messages and high GPU counts.
    - **CollNet/Direct:** Leverages in-network reduction capabilities when available (e.g., InfiniBand switches with SHARP).
  - **Synchronization & Ordering:** Manages low-level synchronization and ordering of operations across GPUs.
  - **Data Type Optimization:** Provides optimized kernels for various data types (FP32, FP16, BF16).
  - **Point-to-Point Primitives:** Offers basic Send/Receive operations for building custom communication patterns.
- **Neuron CCL** (AWS's library for Trainium/Inferentia)
  - **Hardware Architecture:** Dedicated collective compute engines (CC-Cores) run in parallel to NeuronCores for compute-communication overlap.
    - 6 CC-Cores per chip for collective operations
    - Trainium: 4 NeuronLink-v2 connections (768 GB/s aggregate)
    - Inferentia2: 2 NeuronLink-v2 connections (192 GB/s aggregate)
  - **Collective Operations:** Implements standard collectives for distributed training:
    - **AllReduce:** Primary operation for gradient synchronization
    - **AllGather:** Gathers activations along dimensions (sequence parallelism)
    - **ReduceScatter:** Replaces AllReduce in tensor-parallel blocks
  - **Topology Support:** 2D torus topology within single instance (trn1.32xlarge supports 2, 8, or 32 ranks)
  - **Multi-Node:** Integrates with EFA (Elastic Fabric Adapter) for inter-instance communication (800 Gbps on Trn1)
  - **Data Path:** Direct transfers between Neuron devices and EFA, bypassing host CPU for low latency

**Open-Source:**
- **RCCL** (ROCm version for AMD)
  - **Goal:** Functional and API-compatible drop-in replacement for NCCL, optimized for AMD's Infinity Fabric.
- **oneCCL** (Intel's collective communications)
  - **Goal:** Functional and API-compatible drop-in replacement for NCCL, optimized for Intel's Xe Link.
- **Gloo** (Facebook/Meta)
  - An open-source, hardware-agnostic library. It provides a portable option for any hardware but is not optimized for specific accelerator interconnects like NCCL or other vendor-specific libraries.

##### Custom ASIC Programming Environments:
Unlike collective communication libraries, these are full-stack programming environments that compile high-level models down to custom hardware architectures with fundamentally different compute and communication paradigms.

- **Cerebras SDK** (Cerebras)
  - **Architecture Target:** Wafer-Scale Engine (WSE) - a single silicon wafer with 850,000+ cores.
  - **Programming Model:** Cerebras Software Language (CSL) - a C-like language built around a dataflow programming model.
  - **Communication Primitives:** Includes `<collectives_2d>` library with MPI-like collective operations (broadcast, scatter, gather, reduce) across rows/columns of processing elements.
  - **Abstraction Level:** Low-level spatial programming - developers explicitly map computation to the 2D PE array.
  - **Routing:** All routing configuration is handled automatically behind the scenes.
  - **Compilation:** Takes CSL code or high-level frameworks and compiles to WSE-specific execution.

- **Groq SDK** (Groq)
  - **Architecture Target:** Language Processing Unit (LPU) / Tensor Streaming Processor (TSP) - deterministic, software-defined architecture.
  - **Programming Model:** Producer-consumer dataflow model with static scheduling.
  - **Key Feature:** Deterministic execution - data flow is statically scheduled at compile time and executes identically every time.
  - **Architecture:** Functionally sliced microarchitecture with interleaved memory and compute units forming a programmable assembly line.
  - **Compiler:** Model-independent compiler that routes computation through the hardware deterministically.
  - **Power Efficiency:** Intelligent ability to turn off idle components and route around only necessary computation.

- **SambaFlow** (SambaNova)
  - **Architecture Target:** Reconfigurable Dataflow Unit (RDU) - a tiled architecture of reconfigurable functional units.
  - **Programming Model:** Dataflow architecture where communications are programmed and optimized for how data should transit computations.
  - **Workflow:**
    - Accepts PyTorch/TensorFlow models
    - Analyzes to extract dataflow graph
    - Compiles to PEF (Program Execution File) that defines the dataflow graph for RDU hardware
    - Runtime loads code/data onto RDUs and manages execution
  - **Scaling:** Seamlessly scales from one to multiple RDUs using the same programming model.
  - **Abstraction:** High-level (works with standard ML frameworks) but maps to spatial dataflow execution.

- **tt-metal** (Tenstorrent)
  - **Architecture Target:** Tensix cores - grid of specialized compute nodes, each with 5 RISC-V CPUs, matrix/vector engines, and 1.5MB SRAM.
  - **Programming Model:** Low-level heterogeneous programming - direct access to RISC-V processors, NoC (Network-on-Chip), and compute engines.
  - **Kernel Structure:** Typically requires three kernels per Tensix core: reader (data input), compute (calculations), writer (data output).
  - **Collective Operations:** Developing CCL with support for all-gather (line/ring topologies) and reduce-scatter (ring topology).
  - **Higher-Level Abstraction:** TTNN provides mesh tensors that wrap multi-chip partial tensors as single logical tensors for easier multi-chip programming.
  - **On-Chip Networking:** High-speed Ethernet ports directly on-chip enable grid connections without external switches.

---

### **Layer 3: Network Layer** (Multi-node, across servers)

#### Network Fabric:
- **InfiniBand** - High-performance RDMA network
  - Current links scale from 200 Gb/s (HDR) to 800 Gb/s (XDR) depending on generation and lane count
  - Generational timeline:
    - SDR (Single Data Rate, 2001): 2.5 Gb/s per lane (~10 Gb/s on x4 links)
    - DDR (Double Data Rate, 2005): 5 Gb/s per lane (~20 Gb/s on x4 links)
    - QDR (Quad Data Rate, 2008): 10 Gb/s per lane (~40 Gb/s on x4 links)
    - FDR (Fourteen Data Rate, 2011): 14 Gb/s per lane (~56 Gb/s on x4 links)
    - EDR (Enhanced Data Rate, 2014): 25 Gb/s per lane (~100 Gb/s on x4 links)
    - HDR (High Data Rate, 2017): 50 Gb/s per lane (~200 Gb/s on x4 links)
    - NDR (Next Data Rate, 2021): 100 Gb/s per lane (~400 Gb/s on x4 links, 800 Gb/s dual-port)
    - XDR (Extreme Data Rate, 2023): 200 Gb/s per lane (~800 Gb/s on x4 links)
  - **IBGDA** (InfiniBand GPU-Async) - GPU-initiated network operations
  - Connects host channel adapters (HCAs) in each server to InfiniBand switches arranged in multi-tier topologies (fat tree, dragonfly, etc.) for cluster-wide reachability
  - Uses queue pairs (send/receive and completion queues) exposed via the verbs API; NIC-resident DMA engines move data without CPU involvement
  - Provides RDMA read/write/atomic and send/receive operations with credit-based, lossless flow control and adaptive routing
  - In-network capabilities (e.g., SHARP reductions) can accelerate collectives when leveraged by libraries such as NCCL or UCX
- **RoCE** (RDMA over Converged Ethernet) - RDMA on Ethernet
  - 100 GbE, 200 GbE, 400 GbE variants
- **Ethernet** - Standard networking (100GbE, 200GbE, 400GbE)

#### Cloud-Specific Network Technologies:
- **GPUDirect-TCPX** (Google Cloud) - Custom RDMA networking stack for A3 VMs
  - RDMA semantics over TCP/IP transport
  - Direct GPU memory to network transfers (bypasses CPU)
  - Works with gVNIC (Google Virtual NIC)
  - Optimized for H100 GPUs on A3 instances
  - **TCPXO** variant on A3 Mega VMs for enhanced performance
  - Open-source: https://github.com/google/nccl-plugin-gpudirecttcpx
- **EFA (Elastic Fabric Adapter)** (AWS) - Custom network interface for AWS
  - High-performance inter-instance communication
  - OS-bypass hardware interface
  - Lower latency than traditional TCP transport

#### Network-GPU Integration:
- **GPUDirect RDMA** - Direct network-to-GPU transfers (bypasses CPU and system memory)
- **GPUDirect Async** - Async transfers between GPU and network

---

### **Layer 4: Communication Abstraction Layer** (Protocol/transport abstraction)

#### Low-Level Communication Frameworks:
- **UCX** (Unified Communication X) - General-purpose communication framework
  - Supports InfiniBand, RoCE, shared memory, GPUDirect
  - Used by many higher-level libraries
  - HPC and training focused

#### Specialized Inference Libraries:
- **NIXL** (NVIDIA Inference Xfer Library) - **Inference-optimized communication**
  - Part of NVIDIA Dynamo framework (announced GTC 2025)
  - Unified API across NVLink, InfiniBand, RoCE, Ethernet
  - Memory/storage abstraction (HBM, DRAM, SSD, networked storage)
  - GPU-initiated transfers via IBGDA
  - Nonblocking, noncontiguous transfers optimized for inference patterns
  - Abstracts GPUDirect Storage, UCX, S3 with common API
  - Open-source: https://github.com/ai-dynamo/nixl

#### Training-Focused Collective Libraries:
- **NCCL/RCCL** - Collective operations (AllReduce, etc.) optimized for training
- **MPI** (Message Passing Interface) - Standard parallel computing API
  - Can leverage GPUDirect and UCX underneath

---

### **Layer 5: Framework Integration** (Application-facing APIs)

#### Training Frameworks:
- **PyTorch Distributed** - Uses NCCL/Gloo under the hood
- **TensorFlow Distribution Strategy** - Multi-GPU/multi-node abstractions
- **Horovod** - Distributed training framework
- **DeepSpeed** - Microsoft's distributed training library
- **Megatron-LM** - Large-scale transformer training

#### Inference Frameworks:
- **NVIDIA Dynamo** - Low-latency distributed inference (uses NIXL)
- **Ray Serve** - Distributed inference serving
- **vLLM** - Inference engine (can integrate with various backends)
- **TensorRT-LLM** - NVIDIA's LLM inference engine

---

### **Layer 6: Orchestration/Scheduling** (Cluster management)

- **Kubernetes device plugins** - GPU topology awareness
- **SLURM** - HPC job scheduler with GPU topology support
- **Ray** - Distributed computing framework
- **Kubernetes topology manager** - NUMA/device topology hints

---

## Key Technology Relationships

```
Application Layer (PyTorch/vLLM)
         |
         v
Communication Layer (NCCL / NIXL / UCX)
         |
         v
GPU-Network Integration (GPUDirect RDMA / IBGDA)
         |
         v
Network Fabric (InfiniBand / RoCE)
         |
         v
Physical Network (NIC Hardware)

         [Parallel Path]
         |
         v
GPU Interconnect (NVLink / XGMI)
         |
         v
GPU-to-GPU Direct Communication
```

---

## Technology Comparison

### Bandwidth Comparison: PCIe vs GPU Interconnects

| Technology | Bandwidth | Notes |
|------------|-----------|-------|
| **PCIe 4.0 x16** | ~32 GB/s | General-purpose interconnect |
| **PCIe 5.0 x16** | ~64 GB/s | 2x PCIe 4.0 |
| **PCIe 6.0 x16** | ~128 GB/s | 4x PCIe 4.0 (emerging) |
| **NVLink 1.0** (P100) | 160 GB/s | **5x faster than PCIe 4.0** |
| **XGMI-2** (MI200) | 200 GB/s | **6x faster than PCIe 4.0** |
| **NVLink 2.0** (V100) | 300 GB/s | **9x faster than PCIe 4.0** |
| **XGMI-2** (MI250) | 400 GB/s | **12x faster than PCIe 4.0** |
| **NVLink 3.0** (A100) | 600 GB/s | **19x faster than PCIe 4.0** |
| **NVLink 4.0** (H100) | 900 GB/s | **28x faster than PCIe 4.0** |
| **NVLink 5.0** (GB200) | 1,800 GB/s | **56x faster than PCIe 4.0** |

**Key Insight:** GPU-to-GPU interconnects (NVLink/XGMI) are purpose-built for high-bandwidth parallel data transfer and are **5-56x faster** than PCIe. PCIe is a general-purpose interconnect and becomes a major bottleneck for GPU-to-GPU communication.

### NIXL vs UCX vs NCCL

| Feature | NIXL | UCX | NCCL |
|---------|------|-----|------|
| **Primary Use Case** | Inference point-to-point | General-purpose | Training collectives |
| **Optimization** | Low-latency, noncontiguous | Flexible transport | AllReduce, broadcast |
| **Memory Abstraction** | Unified (HBM/DRAM/SSD/S3) | Transport-focused | GPU memory |
| **API Style** | Point-to-point transfers | Message passing | Collective operations |
| **GPU Initiation** | Yes (IBGDA) | Partial | No |
| **Storage Integration** | Native (GPUDirect Storage, S3) | Limited | No |

### Cloud Network Technologies

| Technology | Provider | Transport | Use Case | Hardware |
|------------|----------|-----------|----------|----------|
| **InfiniBand** | Standard | Native RDMA | On-prem/HPC | Specialized NICs |
| **RoCE** | Standard | RDMA over Ethernet | Data center | RDMA-capable NICs |
| **TCPX** | Google Cloud | RDMA over TCP/IP | A3 VMs (H100) | gVNIC |
| **EFA** | AWS | Custom OS-bypass | EC2 instances | Custom adapter |

---

## Notes

- **NIXL** is positioned as a modern inference-specific abstraction that sits above network and hardware layers but below application frameworks
- **NVLink** and **XGMI** are competing proprietary technologies for high-bandwidth GPU-to-GPU communication
- **GPUDirect** is a suite of technologies that enable direct data paths, bypassing CPU overhead
- **IBGDA** enables GPU-initiated network operations without CPU involvement
- **CXL** is an emerging standard that may unify some of the proprietary interconnect space
- **TCPX** is Google's cloud-native answer to RDMA without specialized hardware - achieves high performance using standard TCP/IP with custom optimizations
- Cloud providers (Google, AWS) have developed custom networking solutions (TCPX, EFA) optimized for their infrastructure

---

*Last updated: 2025-10-09*
