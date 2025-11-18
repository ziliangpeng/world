# NVIDIA Hopper Architecture - Deep Dive

> Research notes on NVIDIA's Hopper architecture GPUs
> Last Updated: January 2025

---

## Overview

Hopper is NVIDIA's GPU architecture launched in 2022, featuring 4th-generation Tensor Cores with FP8 support and a Transformer Engine optimized for large language models. Total of 14 distinct products across different market segments.

**Process Technology**: TSMC 4N (custom 5nm), 80 billion transistors
**Key Innovation**: Fourth-generation Tensor Cores with FP8 precision, Transformer Engine for LLMs
**NVLink**: Version 4.0 with 900 GB/s per GPU

---

## H100 Variants (Announced March 2022, Shipping October 2022)

The H100 GPU was officially announced on [March 22, 2022](https://www.nvidia.com/en-us/about-nvidia/press-releases/2022/nvidia-hopper-architecture-the-new-engine-for-the-worlds-ai-infrastructures/). Initial shipments began in [October 2022](https://www.crn.com/news/components-peripherals/nvidia-h100-gpu-shipping-now-partners-see-big-ai-opportunities/).

### H100 SXM5 (80GB) - Flagship Training GPU

**Specifications:**

- **Memory**: 80GB HBM3, 3.35 TB/s bandwidth
- **TDP**: 700W (configurable)
- **Form Factor**: SXM5 module
- **CUDA Cores**: 16,896
- **Tensor Cores**: 528 (4th generation)
- **NVLink**: 900 GB/s (NVLink 4.0)
- **Boost Clock**: 1,830 MHz

**Performance:**

- FP32: ~67 TFLOPS
- FP16/BF16: 1,979 TFLOPS
- FP8 (with Transformer Engine): ~4,000 TFLOPS
- Sparsity: Up to 2x throughput for sparse models

**Pricing (2025):**

- Purchase: $25,000-$40,000 per GPU
- Manufacturing cost: ~$3,320
- 8-GPU systems: $300,000-$500,000 (including infrastructure)
- Cloud rental: $2.10-$6.00/hour

  - Budget providers (GMI Cloud, Jarvislabs): $2.10-$3.00/hour
  - Hyperscale (AWS, Azure, GCP): ~$5.00/hour on-demand

**Target Market:**

- Large language model training (GPT-class, 70B+ parameters)
- High-throughput inference for production LLMs
- Large-scale scientific computing and HPC
- Multi-modal AI training

**Key Features:**

- Requires liquid or specialized air cooling
- Maximum performance variant in H100 family
- FP8 Transformer Engine optimized for transformer models (GPT, BERT, LLaMA)
- 900 GB/s NVLink enables efficient multi-GPU scaling for 100B+ parameter models
- First-to-market advantage during 2023 LLM explosion

---

### H100 PCIe (80GB) - Standard Server Integration

**Specifications:**

- **Memory**: 80GB HBM2e, 2.0 TB/s bandwidth
- **TDP**: 350W
- **Form Factor**: Dual-slot PCIe Gen5 x16
- **CUDA Cores**: 14,592
- **Tensor Cores**: 456 (4th generation)
- **Interconnect**: PCIe Gen5 at 128 GB/s, Limited NVLink connectivity
- **Boost Clock**: 1,620 MHz

**Performance:**

- Lower than SXM5 variant due to reduced core count and memory bandwidth
- FP8: Lower TFLOPS than SXM5

**Pricing (2025):**

- Purchase: $20,000-$30,000

**Target Market:**

- Enterprises needing flexible server integration
- Mid-size AI teams
- Cloud providers
- Universities
- 7B-70B parameter model training

**Key Features:**

- First GPU with PCIe Gen5 support
- Easier integration into existing datacenter infrastructure
- Lower power requirements (fits standard datacenter power/cooling)
- No specialized cooling required

---

### H100 NVL (94GB per GPU, 188GB total) - LLM Inference Specialist

**Specifications:**

- **Memory**: 94GB HBM3 per GPU (188GB total for dual-GPU assembly), 3.9 TB/s per GPU
- **TDP**: 700-800W total (350-400W per GPU)
- **Form Factor**: Dual-GPU PCIe card with three NVLink 4 bridges
- **Interconnect**: 600 GB/s between GPUs via NVLink, PCIe Gen5 x16 to host
- **Cooling**: Passive cooling

**Performance:**

- Up to 12X performance improvement for GPT-175B over DGX A100

**Target Market:**

- Large language model (LLM) deployment and inference
- Production serving of 100B+ parameter models

**Key Features:**

- Dual-GPU configuration bridged together as single unit
- Optimized specifically for LLM inference workloads
- Higher memory capacity per GPU than standard H100 (94GB vs 80GB)

---

### H100 CNX (80GB) - Converged Accelerator

**Specifications:**

- **Memory**: 80GB HBM3 (H100 specifications)
- **TDP**: Similar to H100 base variant
- **Form Factor**: Converged platform
- **Interconnect**: Integrated ConnectX-7 SmartNIC with 400 Gb/s networking

**Target Market:**

- I/O-intensive applications
- Multinode AI training in enterprise datacenters
- 5G edge processing
- Applications requiring high-speed networking

**Key Features:**

- Combines H100 GPU with ConnectX-7 SmartNIC in single platform
- Reduces complexity and cost for networked AI workloads
- 400 Gb/s networking integrated on-chip

---

## H200 Variants (Announced November 2023, Shipping Q2 2024)

The H200 GPU was officially announced on [November 13, 2023](https://nvidianews.nvidia.com/news/nvidia-supercharges-generative-ai-with-h200-gpu-with-worlds-fastest-memory/). While initial anticipated shipping was Q2 2024, large-scale deliveries are now expected after Q3 2024. The first DGX H200 system was personally delivered to OpenAI in [late March 2024](https://www.tomshardware.com/pc-components/gpus/jensen-huang-delivers-first-nvidia-dgx-h200-to-openai-and-sam-altman/). Cloud provider CoreWeave began offering H200 GPUs in [August 2024](https://www.prnewswire.com/news-releases/coreweave-is-first-cloud-provider-to-offer-nvidias-h200-tensor-core-gpus-302220333.html/).

### H200 SXM5 (141GB) - Memory-Upgraded Flagship
A memory-focused refresh of the H100, designed for larger models and higher-throughput inference.

**Key Improvements Over H100:**

The H200 uses the same Hopper GH100 die but upgrades the memory subsystem:

| Feature | H100 | H200 | Improvement |
|---------|------|------|-------------|
| **Memory Capacity** | 80GB HBM3 | 141GB HBM3e | +76% |
| **Memory Bandwidth** | 3.35 TB/s | 4.8 TB/s | +43% |
| **TDP** | 700W | 700W | Same power envelope |

**Specifications:**

- **Memory**: 141GB HBM3e, 4.8 TB/s bandwidth
- **TDP**: 700W (configurable)
- **Form Factor**: SXM5 module
- **NVLink**: 900 GB/s (NVLink 4.0)
- **Cooling**: Air and liquid cooling options

**Performance:**

- FP8: 3,958 TFLOPS
- FP16/BFLOAT16: 1,979 TFLOPS
- TF32: 989 TFLOPS
- FP32: 67 TFLOPS
- FP64: 34 TFLOPS

**Performance Gains:**

- LLM Inference: Up to 2x faster than H100
- MLPerf Llama2-70B Benchmark:
  - H200: 31,712 tokens/second
  - H100: 21,806 tokens/second
  - **45% improvement** in real-world workloads

**Why Memory Matters:**

- Larger batch sizes: 141GB enables more concurrent requests for inference
- Bigger models: Fits 200B+ parameter models without model parallelism
- Longer context windows: Essential for RAG systems and document processing
- Same infrastructure: No power/cooling upgrades needed vs H100

**Pricing (2025):**

- Purchase: $30,000-$40,000 (15-20% premium over H100)
- Cloud rental: $2.50-$10.60/hour
  - Budget: Jarvislabs ($3.80/hour), GMI Cloud ($2.50/hour)
  - Hyperscale: AWS/Azure (~$10.60/hour)
  - Spot/preemptible: Google Cloud ($3.72/hour)

**Target Market:**

- Very large language models (200B+ parameters)
- High-throughput production inference
- Long-context applications (32K+ tokens)
- Retrieval-augmented generation (RAG) systems
- Multi-modal models requiring extensive memory

**Key Features:**

- First GPU with HBM3e memory
- 1.4X more memory and 2.4X more bandwidth than A100
- Up to 18% higher performance vs NVL

---

### H200 NVL (141GB) - Air-Cooled Enterprise Inference

**Specifications:**

- **Memory**: 141GB HBM3e, 4.8 TB/s bandwidth
- **TDP**: 600W (configurable)
- **Form Factor**: Dual-slot PCIe Gen5 x16
- **Interconnect**: NVLink bridges (2-way or 4-way) at 900 GB/s, PCIe Gen5
- **Cooling**: Air-cooled only

**Performance:**

- Up to 1.7X faster LLM inference vs H100 NVL when connected in 4-way configuration

**Target Market:**

- Enterprise AI inference
- Air-cooled deployments
- Flexible configurations requiring scalable inference

**Key Features:**

- Air-cooled only (no liquid cooling required)
- Flexible 2-way or 4-way configurations
- Same memory advantages as H200 SXM5 (141GB HBM3e)

---

## China-Specific Variants

### H800 - Export-Compliant (Pre-October 2023)

The H800 was a modified version of the H100, introduced by NVIDIA in late 2022 specifically to comply with then-current U.S. export regulations for the Chinese market. It served as a temporary solution until it was also banned by updated, stricter rules in October 2023.

**Export Regulation & Key Changes:**
The primary limitation imposed by the export rules was on the maximum chip-to-chip interconnect speed, designed to hinder the ability to build massive, interconnected GPU clusters for training large-scale AI models.

- **Performance Reduction:** The H800's key change was a significant reduction in its NVLink bandwidth to **400 GB/s**, down from the H100's 900 GB/s.
- **Core Architecture:** Beyond the interconnect speed, the H800 was largely identical to its H100 PCIe counterpart, retaining the same 80GB memory, CUDA core counts, and Tensor core capabilities.

**Performance Impact:**
- **Multi-GPU Training:** The reduced bandwidth directly slowed down large-scale training tasks that require frequent, high-speed communication across multiple GPUs.
- **Single-GPU Performance:** For workloads that run on a single chip (like inference or smaller model training), the H800's performance was very close to the H100, delivering up to 95% of the performance in many cases.

**Market Reception & Sales:**
As the most powerful compliant GPU available in China at the time, the H800 was highly sought after. Major Chinese technology and cloud companies (such as Alibaba, Baidu, and Tencent) were significant buyers, placing large orders to build out their AI infrastructure before the rules were tightened. While exact sales figures are not public, the high demand from these tech giants underscores its critical role in the Chinese market during that period.

---

### H20 - Current China Market GPU

The H20 is NVIDIA's most advanced chip designed for the Chinese market under the strict U.S. export regulations updated in October 2023. It became available in mid-2024 and represents a different compliance strategy compared to its predecessor, the H800.

**Export Regulation & Design Strategy:**
The October 2023 rules targeted overall "Total Processing Performance" and "Performance Density," forcing a more drastic redesign. Instead of only limiting interconnect speed, NVIDIA's strategy for the H20 was to:
1.  **Significantly Reduce Compute:** The chip's raw AI compute power (TFLOPS) was cut to approximately 15% of a full H100 to stay under the regulatory threshold.
2.  **Maintain Key Strengths:** Crucially, NVIDIA kept the full **900 GB/s NVLink** interconnect and equipped the H20 with a large and fast **96GB HBM3e memory** subsystem.

**Performance Profile:**
This design created a unique performance profile. The low TFLOPS make the H20 inefficient for training large models from scratch. However, its large memory and high bandwidth can make it faster than even an H100 for specific memory-bound tasks, particularly inference on already-trained large language models.

**Market Reception & Competition:**
The H20 received a mixed reception. Its ability to run large models and its integration with the mature CUDA software ecosystem made it a necessary option for some Chinese companies. However, its high price point for limited training performance led to reports of lukewarm demand.

Unlike the H800, which entered a market with no alternatives, the H20 faced strong competition from domestic chips like **Huawei's Ascend 910B**, which had become a viable alternative for many Chinese firms seeking to reduce reliance on U.S. technology.

---

## Grace Hopper Superchip Family

### GH200 Grace Hopper Superchip - CPU+GPU Hybrid

The Grace Hopper Superchip is fundamentally different from a standalone GPU. **Named GH200, this Superchip product** integrates a high-performance Grace CPU and a Hopper GPU into a single package. Every computer system requires a CPU (Central Processing Unit) to act as the "brain" or "general manager," running the operating system and orchestrating tasks. The GPU is a specialized "worker" for parallel calculations. The innovation of the Superchip is to integrate a high-performance Grace CPU and a Hopper GPU (which can be **either an H100 or an H200 variant**) into a single package, connecting them with an ultra-fast NVLink-C2C interconnect. This eliminates the traditional communication bottleneck of a PCIe bus, allowing the CPU and GPU to work together much more efficiently on data-intensive tasks. While the Superchip acts as the core processing engine, a complete server still requires standard components like storage, system memory, and power supplies.

**Architecture:**

Each superchip combines:
- **1 × Grace CPU** (72-core ARM Neoverse V2)
- **1 × Hopper GPU** (H100 or H200 variant)
- Connected via **NVLink-C2C**: 900 GB/s bidirectional CPU-GPU bandwidth

**Specifications:**

- **GPU Memory**: 96GB HBM3 or 141GB HBM3e (H200 variant)
- **CPU**: 72-core ARM Neoverse V2
- **CPU Memory**: 120GB, 240GB, or 480GB LPDDR5X with ECC
- **Total Memory**: Up to 624GB combined fast-access memory (141GB HBM3e + 480GB LPDDR5X)
- **TDP**: 450W to 1000W (configurable)
- **CPU-GPU Interconnect**: 900 GB/s NVLink-C2C (chip-to-chip), 7X faster than PCIe Gen5

**Target Market:**

- AI/HPC hybrid workloads requiring tight CPU-GPU integration
- Applications with large CPU working sets
- Scientific computing
- Graph analytics

**Key Features:**

- Coherent CPU+GPU memory model (GPU can access CPU memory)
- ARM-based architecture (not x86)
- Single-package integration
- 7X faster CPU-GPU communication vs PCIe Gen5
- Eliminates CPU-GPU bottleneck for hybrid workloads

---

### GH200 NVL2 - Dual Superchip

**Architecture:**

- 2× GH200 Superchips connected via NVLink

**Specifications:**

- **GPU Memory**: 288GB total high-bandwidth memory (2 × 141GB or 2 × 96GB)
- **CPU Memory**: Up to 960GB LPDDR5X combined (2 × 480GB)
- **Total Memory**: 1.2TB fast memory
- **Total Bandwidth**: 10 TB/s memory bandwidth

**Target Market:**

- Extremely large AI models
- Models requiring >600GB memory
- Multi-modal foundation models

**Key Features:**

- Fully connects two GH200 Superchips with NVLink
- Unified memory space across 2 CPUs and 2 GPUs
- 1.2TB total fast-access memory

---

### DGX GH200 - Supercomputer Scale

**Architecture:**

- 32 × GH200 Superchips

  - 32 × Grace CPUs (72-core ARM each)
  - 32 × H100 GPUs
- NVLink Switch System providing non-blocking fat-tree topology

**Specifications:**

- **Total GPU Memory**: 144TB shared memory accessible to all GPUs
- **System Bandwidth**: 128 TB/s bisection bandwidth
- **Compute**: 1 ExaFLOP FP8 with sparsity
- **Supported Precisions**: FP64, FP32, FP16, BF16, FP8, FP4

**Target Market:**

- Supercomputing centers
- National AI infrastructure
- Massive AI model training (trillion+ parameters)
- Research requiring unprecedented memory capacity

**Key Features:**

- First 100TB+ GPU memory system
- Every GPU can access all other GPU/CPU memory at 900 GB/s
- Non-blocking NVLink fabric
- Enables trillion-parameter models without external storage
- 32 × 72-core CPUs = 2,304 total CPU cores

---

## DGX H100 / H200 System Configurations

NVIDIA offers the DGX system as a complete, pre-configured AI development platform. It is available with either H100 or H200 GPUs, with the primary difference being the total GPU memory.

**Base Configuration:**

- **GPU Interconnect**: 900 GB/s NVLink per GPU
- **Networking**: 8× NVIDIA ConnectX-7 400 Gb/s NICs
- **System Power**: ~10.4kW total
- **Form Factor**: Rack-mounted system

| System | GPUs | Total GPU Memory |
| :--- | :--- | :--- |
| **DGX H100** | 8 × H100 SXM5 | 640GB HBM3 |
| **DGX H200** | 8 × H200 SXM5 | 1.1TB HBM3e |

**Target Market:**

- Enterprise AI infrastructure
- Unified AI development platform
- Organizations needing turnkey AI systems

**Key Features:**

- Complete system (not just GPUs)
- Pre-configured and optimized
- Full NVLink connectivity between all 8 GPUs
- High-speed networking for multi-node clusters

---

## Summary Statistics

**Total Hopper-based Products**: 14 distinct variants/configurations

**Product Categories:**

- H100 variants: 4 (SXM5, PCIe, NVL, CNX)
- H200 variants: 2 (SXM5, NVL)
- China-specific: 2 (H800, H20)
- Grace Hopper: 3 (GH200, GH200 NVL2, DGX GH200)
- System platforms: 1 (DGX H100)

**Variant Definitions:**

The various Hopper GPU products are primarily "variants" in that they are all based on the same core **GH100 silicon die**. Differences between these variants stem from their packaging, the number of active cores enabled (binning), and the additional components integrated onto the card, tailoring them for specific use cases and server environments.

- **SXM5:** A high-performance module designed for direct mounting onto a server baseboard (e.g., NVIDIA HGX). In an 8-GPU system, an NVLink Switch System allows all GPUs to communicate with each other at the full **900 GB/s** speed.
- **NVL (NVLink-connected):**
    Refers to a dual-GPU configuration on a single PCIe card, where two GPUs are connected to each other via high-speed NVLink bridges.

    - **On-card GPU-to-GPU speed:** 600 GB/s (H100 NVL) or 900 GB/s (H200 NVL).
    - **Off-card (to host/other cards) speed:** Limited by PCIe Gen5 x16 at 128 GB/s.
- **CNX (Converged Accelerator):** Refers to a single GPU combined with a high-speed ConnectX-7 SmartNIC on the same card. This **convergence** of compute and networking onto a single physical unit is designed to eliminate I/O bottlenecks in workloads requiring both massive computation and extremely fast data movement. The SmartNIC's **RDMA (Remote Direct Memory Access)** capabilities are crucial for enabling this high-speed, low-latency data transfer directly to the GPU.

**Memory Range:**

- Minimum: 80GB (H100 PCIe, H100 SXM5)
- Maximum single GPU: 141GB (H200)
- Maximum system: 144TB (DGX GH200 with 32 GPUs)

**TDP Range:**

- Minimum: 350W (H100 PCIe)
- Maximum single GPU: 700W (H100 SXM5, H200 SXM5)
- Maximum system: 10.4kW (DGX H100)

**Process Technology:**

- TSMC 4N (custom 5nm node)
- 80 billion transistors (H100/H200 die)

**Interconnect Evolution:**

- NVLink 4.0: 900 GB/s per GPU (H100, H200)
- NVLink-C2C: 900 GB/s CPU-GPU (Grace Hopper)
- PCIe Gen5: 128 GB/s (first GPU to support)

### Hopper GPU Comparison Table

| GPU Variant | Memory Size & Type | Memory Bandwidth | FP8 TFLOPS (AI) | Interconnect Bandwidth | Max TDP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H100 SXM5** | 80GB HBM3 | 3.35 TB/s | ~4,000 | 900 GB/s (NVLink 4.0) | 700W |
| **H100 PCIe** | 80GB HBM2e | 2.0 TB/s | Lower than SXM5 | 128 GB/s (PCIe Gen5) | 350W |
| **H100 NVL**¹ | 94GB HBM3 | 3.9 TB/s | Not specified | 600 GB/s (NVLink Bridge) | 350-400W |
| **H100 CNX** | 80GB HBM3 | Not specified | Not specified | 400 Gb/s (ConnectX-7) | ~350W |
| **H200 SXM5** | 141GB HBM3e | 4.8 TB/s | 3,958 | 900 GB/s (NVLink 4.0) | 700W |
| **H200 NVL**² | 141GB HBM3e | 4.8 TB/s | Not specified | 900 GB/s (NVLink Bridge) | 600W (Card) |
| **H800** | 80GB HBM | Not specified | ~70% of H100 | 300 GB/s (NVLink) | Not specified |
| **H20** | 96GB HBM3e | 4.0 TB/s | 296 | 900 GB/s (NVLink) | 400W |
| **GH200**³ | up to 141GB HBM3e | up to 4.8 TB/s | up to 3,958 | 900 GB/s (NVLink-C2C) | 1000W (Chip) |

<br>
¹ Specs are per GPU in a dual-GPU assembly.
<br>
² Specs are per GPU, but TDP is for the entire dual-slot card.
<br>
³ GPU specs depend on whether the H100 or H200 variant is used in the Superchip.

---

## Key Innovations

**Fourth-Generation Tensor Cores:**

- FP8 precision support (first GPU architecture with FP8)
- 2.5X AI performance improvement over previous generation
- Sparsity support for up to 2X additional throughput

**Transformer Engine:**

- Automatic precision switching for transformer models
- Optimized specifically for GPT, BERT, LLaMA architectures
- Dynamic precision selection (FP8 ↔ FP16/BF16)
- Purpose-built for LLM era

**NVLink 4.0:**

- 900 GB/s bidirectional bandwidth per GPU
- 3X improvement over NVLink 3.0 (300 GB/s)
- Enables efficient scaling to 100B+ parameter models

**HBM3/HBM3e Memory:**

- First GPU with HBM3 (H100)
- First GPU with HBM3e (H200)
- Up to 4.8 TB/s bandwidth (H200)
- Up to 141GB capacity (H200)

**Multi-Instance GPU (MIG):**

- Up to 7 GPU instances per physical GPU
- Hardware-level isolation
- Workload consolidation and QoS

**Confidential Computing:**

- Hardware-based TEE (Trusted Execution Environment)
- Memory encryption
- Secure key management

---

## Market Context

**Launch**: 2022 (H100), 2023 (H200)

**Timing**: Coincided with ChatGPT-driven AI explosion (November 2022)

**Dominance Factors:**

1. FP8 Transformer Engine optimized specifically for transformer models
2. 900 GB/s NVLink enables efficient multi-GPU scaling
3. First-to-market advantage during 2023 LLM boom
4. CUDA ecosystem maturity vs competitors
5. Memory capacity and bandwidth leadership

**Competition:**

- AMD MI300X
- Google TPU v5
- AWS Trainium
- Microsoft Maia
- Specialized accelerators (Groq, Cerebras)

**Export Restrictions:**

- H800: Initial China-compliant variant (banned October 2023)
- H20: Current China-compliant variant (reduced compute, maintained memory)
- Restrictions target AI compute performance (TFLOPS)

---

## Notes

- L20 and L2 use Ada Lovelace architecture, NOT Hopper
- H100 remains dominant for training in 2025
- H200 becoming standard for large-scale inference
- Grace Hopper targets hybrid CPU-GPU workloads
- China variants show shift from compute to memory bottleneck in LLMs
