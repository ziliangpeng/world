# GPU Connectivity Technologies - Comprehensive Overview

## Layered Taxonomy

### **Layer 1: Physical/Link Layer** (Die-to-die, GPU-to-GPU on same board/node)

#### Proprietary High-Speed Interconnects:
- **NVLink** (NVIDIA) - Dedicated GPU-to-GPU links
  - NVLink 2.0: 300 GB/s per GPU (V100)
  - NVLink 3.0: 600 GB/s per GPU (A100)
  - NVLink 4.0: 900 GB/s per GPU (H100)
  - NVLink 5.0: 1.8 TB/s per GPU (B100/GB200)
- **XGMI/Infinity Fabric** (AMD) - AMD's GPU interconnect
  - Used in MI100, MI200, MI300 series
  - Up to 200-400 GB/s depending on generation
- **Xe Link** (Intel) - Intel GPU interconnect for data center GPUs
  - Used in Ponte Vecchio and newer Xe GPUs
- **NVSwitch** (NVIDIA) - Switch fabric connecting multiple NVLink GPUs
  - Enables all-to-all GPU communication at full NVLink speed

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

### NIXL vs UCX vs NCCL

| Feature | NIXL | UCX | NCCL |
|---------|------|-----|------|
| **Primary Use Case** | Inference point-to-point | General-purpose | Training collectives |
| **Optimization** | Low-latency, noncontiguous | Flexible transport | AllReduce, broadcast |
| **Memory Abstraction** | Unified (HBM/DRAM/SSD/S3) | Transport-focused | GPU memory |
| **API Style** | Point-to-point transfers | Message passing | Collective operations |
| **GPU Initiation** | Yes (IBGDA) | Partial | No |
| **Storage Integration** | Native (GPUDirect Storage, S3) | Limited | No |

---

## Notes

- **NIXL** is positioned as a modern inference-specific abstraction that sits above network and hardware layers but below application frameworks
- **NVLink** and **XGMI** are competing proprietary technologies for high-bandwidth GPU-to-GPU communication
- **GPUDirect** is a suite of technologies that enable direct data paths, bypassing CPU overhead
- **IBGDA** enables GPU-initiated network operations without CPU involvement
- **CXL** is an emerging standard that may unify some of the proprietary interconnect space

---

*Last updated: 2025-10-09*
