# GPU Connectivity Technologies - Comprehensive Overview

## Layered Taxonomy

### **Layer 1: Physical/Link Layer** (Die-to-die, GPU-to-GPU on same board/node)

#### Proprietary High-Speed Interconnects:
- **NVLink** (NVIDIA) - Dedicated GPU-to-GPU links
  - NVLink 1.0: 160 GB/s per GPU (P100, 4 links @ 20 GB/s bidirectional)
  - NVLink 2.0: 300 GB/s per GPU (V100, 6 links @ 25 GB/s bidirectional)
  - NVLink 3.0: 600 GB/s per GPU (A100, 12 links @ 50 GB/s bidirectional)
  - NVLink 4.0: 900 GB/s per GPU (H100, 18 links @ 50 GB/s bidirectional)
  - NVLink 5.0: 1.8 TB/s per GPU (B100/B200/GB200, 18 links @ 100 GB/s bidirectional, 224G SerDes)
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
  - NVSwitch 1.0: 7.2 Tb/s switch throughput (DGX-2, V100)
  - NVSwitch 2.0: Supports 12 NVLinks @ 50 GB/s (DGX A100)
  - NVSwitch 3.0: 12.8 Tb/s switch throughput, 18 NVLinks support (DGX H100)
  - NVSwitch 4.0: 1.8 TB/s GPU-to-GPU bandwidth, up to 576 GPUs (B100/B200, 1 PB/s total)
  - Enables all-to-all GPU communication at full NVLink speed
  - **Note on Topology**: In smaller systems, NVLink connects GPUs directly. In larger systems, GPUs connect via NVLink to the NVSwitch, which creates a switched fabric for all-to-all communication.

#### Standard Interconnects:
- **PCIe** (PCI Express) - Universal standard, connects GPU to CPU and other PCIe devices
  - PCIe 4.0: ~32 GB/s per x16 slot
  - PCIe 5.0: ~64 GB/s per x16 slot
  - PCIe 6.0: ~128 GB/s per x16 slot (emerging)
- **CXL** (Compute Express Link) - Emerging cache-coherent interconnect
  - Built on PCIe physical layer
  - Enables CPU-GPU memory coherency

---

### **Layer 2: Intra-Node Communication** (Within a single server)

#### Hardware/Driver Level:
- **GPUDirect P2P** - Direct GPU-to-GPU memory transfers over PCIe (bypassing CPU)
- **GPUDirect Storage** - Direct GPU-to-storage transfers
- **UVM (Unified Virtual Memory)** - Automatic page migration between CPU/GPU

#### Software/Library Level:
- **NCCL** (NVIDIA Collective Communications Library)
- **RCCL** (ROCm version for AMD)
- **Gloo** (Facebook's collective communications)
- **oneCCL** (Intel's collective communications)

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
