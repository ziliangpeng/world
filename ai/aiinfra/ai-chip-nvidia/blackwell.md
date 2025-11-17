# NVIDIA Blackwell Architecture - Deep Dive

> Research notes on NVIDIA's Blackwell architecture GPUs
> Last Updated: November 2025

---

## Overview

Blackwell is NVIDIA's GPU architecture that succeeds the Hopper architecture. Announced at GTC in March 2024, Blackwell is designed for the new era of generative AI and trillion-parameter models. It features a multi-chip module (MCM) design, a first for NVIDIA's GPUs.

**Process Technology**: Custom TSMC 4NP (an enhanced 4N process)
**Transistor Count**: 208 billion (via two 104-billion-transistor dies connected with a 10 TB/s link)
**Key Innovation**: Second-generation Transformer Engine with FP4/FP6 precision, Multi-chip module design
**NVLink**: 5th Generation with 1.8 TB/s per GPU

---

## Data Center GPU Variants (B100, B200)

The Blackwell data center GPUs are based on a dual-die, multi-chip module (MCM) design. They are offered in two main configurations, B100 and B200, which share the same memory subsystem but differ in performance and power consumption.

### B100 SXM - Infrastructure-Compatible Upgrade

The B100 is designed to be compatible with existing HGX H100 server infrastructure, offering a significant performance upgrade within the same power envelope.

**Specifications:**
- **Memory**: 192GB HBM3e
- **Memory Bandwidth**: 8 TB/s
- **TDP**: 700W
- **Form Factor**: SXM module
- **NVLink**: 1.8 TB/s (5th Generation)

**Performance:**
- FP4 (AI): 7 PFLOPS
- FP8: 3.5 PFLOPS

### B200 SXM - Maximum Performance

The B200 is the highest-performance Blackwell GPU, offering maximum compute capabilities for the most demanding AI training and inference workloads.

**Specifications:**
- **Memory**: 192GB HBM3e
- **Memory Bandwidth**: 8 TB/s
- **TDP**: 1000W
- **Form Factor**: SXM module
- **NVLink**: 1.8 TB/s (5th Generation)

**Performance:**
- FP4 (AI): 20 PFLOPS
- FP8: 9 PFLOPS (in a DGX B200 system)

---

## Grace Blackwell Superchip Family (GB200)

The Grace Blackwell Superchip is the successor to the Grace Hopper Superchip, integrating the Grace CPU with the new Blackwell GPU architecture. These superchips are the building blocks for NVIDIA's largest-scale systems.

### GB200 Superchip

This is the fundamental building block, combining a CPU and two GPUs into a single, integrated package.

**Architecture:**
- **1 × Grace CPU** (72-core ARM Neoverse V2)
- **2 × Blackwell GPUs**
- Connected via **NVLink-C2C**: 900 GB/s bidirectional CPU-GPU bandwidth

**Specifications:**
- **GPU Memory**: 384GB HBM3e (192GB per GPU)
- **CPU Memory**: Up to 480GB LPDDR5X
- **TDP**: Configurable up to 2700W

### GB200 NVL72 - Rack-Scale System

The GB200 NVL72 is a complete, liquid-cooled, rack-scale system that functions as a single, massive GPU. It is designed for exascale computing and training/inference of the largest trillion-parameter models.

**Architecture:**
- **36 × GB200 Superchips**, resulting in:
    - **36 × Grace CPUs** (2,592 total cores)
    - **72 × Blackwell GPUs**
- **5th-Generation NVLink Switch System:** Connects all 72 GPUs in a non-blocking fat-tree topology, allowing them to function as one. Total GPU communication bandwidth is 130 TB/s.

**System Specifications:**
- **Total GPU Memory**: 13.4 TB HBM3e
- **Total Fast Memory**: Up to 30.4 TB
- **System Power**: ~120 kW
- **Cooling**: Direct-to-chip liquid cooling

**System Performance:**
- **LLM Inference:** Up to 30x faster than a DGX H100 system.
- **LLM Training:** Up to 4x faster than a DGX H100 system.
- **Total FP4 Compute:** 1,440 PFLOPS (1.44 ExaFLOPS)

---

## DGX Systems

NVIDIA offers fully integrated, pre-configured DGX systems for both large-scale data center deployments and local development.

### DGX B200

The DGX B200 is the fifth-generation DGX system, designed for the most demanding AI training and inference workloads. It is the direct successor to the DGX H100/H200 systems.

**Configuration:**
- **GPUs**: 8 × B200 SXM GPUs
- **Total GPU Memory**: 1.4TB HBM3e
- **CPUs**: 2 × Intel Xeon Platinum 8570 Processors
- **System Memory**: 4TB DDR5
- **GPU Interconnect**: 1.8 TB/s NVLink per GPU (5th Generation)
- **Networking**: 8x 400Gb/s InfiniBand/Ethernet (via ConnectX-7 and BlueField-3)
- **System Power**: 14.3 kW

**Performance:**
- **FP8 Training**: 72 PFLOPS
- **FP4 Inference**: 144 PFLOPS

### DGX Spark

The DGX Spark is a compact, portable AI supercomputer designed as a workstation for developers and researchers to run AI projects locally.

**Configuration:**
- **Core**: 1 × GB10 Grace Blackwell Superchip (integrating a Blackwell GPU with a 20-core Arm CPU)
- **Total Unified Memory**: 128GB
- **Performance**: Up to 1 petaFLOP (FP4)
- **Connectivity**: Includes NVIDIA ConnectX networking, with the ability to link two DGX Spark systems together.

---

## Consumer GPU Variants (RTX 50-Series)

While the B100/B200 and GB200 systems are designed for the data center, the Blackwell architecture is also expected to power the next generation of consumer GeForce GPUs, rumored to be the RTX 50-Series. Information in this section is based on industry reports and rumors, as these products have not been officially announced.

The consumer Blackwell architecture is expected to introduce GDDR7 memory, DisplayPort 2.1, and a new version of DLSS (4.0).

### RTX 5090 (Rumored)

The flagship consumer card is expected to be the RTX 5090.

**Rumored Specifications:**
- **GPU Core**: GB202
- **Memory**: 32GB GDDR7
- **Memory Bus**: 512-bit
- **CUDA Cores**: ~21,760
- **TGP**: ~600W

### RTX 5080 (Rumored)

**Rumored Specifications:**
- **GPU Core**: GB203
- **Memory**: 16GB GDDR7
- **Memory Bus**: 256-bit
- **CUDA Cores**: ~10,752
- **TGP**: ~400W

---

## Key Innovations

The Blackwell architecture introduces several foundational technologies designed to power the next era of generative AI.

### 2nd-Generation Transformer Engine
This engine uses new micro-tensor scaling support and dynamic range management algorithms integrated with NVIDIA's TensorRT-LLM and NeMo frameworks. It is specifically designed to accelerate LLM inference and training by enabling new, lower-precision data formats.

### FP4 and FP6 Precision
Blackwell's 5th-generation Tensor Cores add support for new ultra-low precision formats: FP4 (4-bit floating point) and FP6 (6-bit floating point). This allows for a doubling of performance and model size per GPU compared to the FP8 precision of the Hopper architecture, while maintaining high accuracy for LLMs.

### 5th-Generation NVLink
To accelerate performance for trillion-parameter models, the latest generation of NVLink provides 1.8 TB/s of bidirectional bandwidth per GPU. This, combined with the NVLink Switch System, allows up to 576 GPUs to be seamlessly connected for massive-scale workloads.

### Multi-Chip Module (MCM) Design
For the first time, NVIDIA is using an MCM design for its GPUs. The Blackwell die is technically two smaller, reticle-limited dies fused together into a single, unified GPU. They are connected by a 10 TB/s NV-High Bandwidth Interface (NV-HBI), ensuring they function as one cohesive chip. This approach overcomes the physical manufacturing limits of single-die designs.

---

## Market Context

**Launch**: Announced March 18, 2024, at NVIDIA's GTC conference.

**Target Applications**:
The Blackwell architecture is designed for the era of generative AI and trillion-parameter models. Its primary applications include:
- **AI and Data Centers:** Large-scale AI training and real-time LLM inference, data processing, and HPC workloads.
- **Scientific Computing:** Engineering simulation, computer-aided drug design, and quantum computing.
- **Consumer and Workstation:** Dedicated dies will power the next generation of GeForce gaming and professional visualization GPUs.

**Competitive Landscape:**
NVIDIA continues to hold a dominant market share in AI accelerators. While competitors like AMD (Instinct series) and Google (TPUs) offer competitive hardware, NVIDIA's primary advantage remains its mature and comprehensive CUDA software ecosystem. The Blackwell platform aims to extend this lead by offering significant performance and efficiency gains for the most demanding AI workloads.
