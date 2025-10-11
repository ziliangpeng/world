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

##### Proprietary Libraries:
- **NCCL** (NVIDIA Collective Communications Library)
- **Neuron CCL** (AWS's library for Trainium/Inferentia)
- Custom ASIC vendors provide their own software stacks to manage their unique hardware. These include the **Cerebras SDK**, **Groq SDK**, **SambaFlow** (SambaNova), and **tt-metal** (Tenstorrent).

##### Open-Source Libraries:
- **RCCL** (ROCm version for AMD)
- **oneCCL** (Intel's collective communications)
- **Gloo** (Facebook's collective communications)

---

### **Layer 3: Network Layer** (Multi-node, across servers)

#### Network Fabric:
- **InfiniBand** - High-performance RDMA network
  - HDR: 200 Gb/s
  - NDR: 400 Gb/s
  - **IBGDA** (InfiniBand GPU-Async) - GPU-initiated network operations
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
