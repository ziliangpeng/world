# Cerebras - Technology Stack

**Company:** Cerebras Systems, Inc.
**Founded:** 2015 (incorporated 2016)
**Focus:** Wafer-scale AI chips for training and inference
**Headquarters:** Sunnyvale, California

---

## Non-AI Tech Stack

Cerebras operates **data center deployments** including the **Condor Galaxy 3 supercomputer** (64 CS-3 systems delivering 8 exaflops, operational Q2 2024). The **CS-3 system** fits in **15U rack space** and connects via **12x 100 Gigabit Ethernet links** that translate standard TCP-IP traffic into Cerebras protocol at full line rate. The software infrastructure includes the **Cerebras SDK** with a **cycle-accurate simulator** for debugging without hardware access, and a **host runtime** responsible for launching programs on the wafer and moving data between CS-3 and host CPU servers (4x improvement in host-to-device transfer speeds). External **MemoryX units** scale from **24TB to 1,200TB** using flash, DRAM, and custom software stacks to pipeline load/store requests with minimal latency. The **SwarmX interconnect** scales up to **2048 CS-3 systems** via tree topology for modular cluster expansion. Founded by **Andrew Feldman** (former SeaMicro CEO), **Gary Lauterbach**, **Michael James**, **Sean Lie**, and **Jean-Philippe Fricker**, Cerebras raised **$1.1B Series G at $8.1B valuation** in September 2025 and filed (later withdrew) Nasdaq IPO in 2024.

**Salary Ranges**: Software Engineer $193K-$295K (median $285K total comp) | Hardware Engineer $188K median | Solution Architect up to $601K | Intern $35.85/hour

---

## AI/ML Tech Stack

### WSE-3 - The Largest Chip Ever Built

**What's unique**: Cerebras designed the **WSE-3 (Wafer-Scale Engine)**, the world's largest AI chip at **46,255 mm²** — **57x larger than the largest GPU**. The chip contains **4 trillion transistors** (19x more than NVIDIA B200) and **900,000 AI-optimized cores** (28x more compute than B200), all fabricated on a **5nm process**. Unlike traditional chips that use multiple smaller dies, Cerebras builds the entire processor on a **single silicon wafer**, eliminating die-to-die communication bottlenecks. The WSE-3 integrates **44 GB of on-chip SRAM** with **21 PByte/s memory bandwidth** and **214 Pbit/s fabric bandwidth**, delivering **125 petaflops of AI performance**. This radical architecture provides **2x tokens/second** compared to the previous WSE-2 generation **with no increase in power or cost**. A cluster of 2048 CS-3 systems delivers **256 exaflops** and **can train Llama2-70B from scratch in less than a day** — approximately one month on Meta's GPU cluster.

### Fine-Grained Dataflow Core Architecture

**What makes it different**: Each of the 900,000 cores implements a **fully programmable processor** optimized for neural network sparsity with **48KB local SRAM** (38,000 square microns per core). The cores feature **8-wide FP16 SIMD units** (2x increase over CS-2), operate at **1.1 GHz**, and consume only **30mW peak power** each. The memory design uses **eight single-ported, 32-bit wide SRAM banks** providing "two full 64-bit reads and one full 64-bit write per cycle," achieving approximately **200 times more memory bandwidth than GPUs when normalized to equivalent die area**. The architecture supports **eight simultaneous tensor operations** ("micro-threads") with cycle-by-cycle context switching, enabling hardware state machines to manage 4D tensor descriptors. **Native unstructured sparsity acceleration** filters zero values at the sender, eliminating wasted computation — critical for efficient inference.

### 2D Mesh Fabric - Wafer-Scale Interconnect

Cerebras's **2D mesh fabric** scales across the entire wafer with **5-port fabric routers** enabling **single-cycle latency** between neighboring cores. The network supports **24 independent static routing "colors"** for non-blocking multi-route communication with **native broadcast and multicast** capabilities within each router. Die-to-die connections span less than one millimeter using high-level metal layers, delivering **seven times more bandwidth than GPU die-to-die bandwidth at only 5 watts**. This interconnect architecture eliminates the traditional GPU limitation where memory bandwidth cannot scale proportionally with compute — Cerebras maintains full bandwidth even as core count reaches 900,000.

### MemoryX - Weight Streaming Architecture

**What sets Cerebras apart**: The **MemoryX** external memory system fundamentally changes how AI models scale. Traditional systems store entire model weights on-chip or in GPU HBM, limiting model size to available memory. Cerebras **streams weights layer-by-layer from MemoryX** to the WSE-3, immediately discarding weights after triggering AXPY operations. This architecture eliminates on-chip storage constraints, enabling **arbitrarily large models**. Configurations range from **24TB and 36TB** (enterprise) to **120TB and 1,200TB** (hyperscale). A **single CS-3 with 1,200TB MemoryX** can store models with **up to 24 trillion parameters** — matching the memory capacity of a **10,000-node GPU cluster** while maintaining single-device simplicity. MemoryX uses flash and DRAM with a custom software stack to pipeline load/store requests with minimal latency.

### SwarmX - Near-Linear Scale-Out

Cerebras's **SwarmX interconnect** enables data-parallel training across **up to 2048 CS-3 systems** (10x improvement over CS-2's 192-system limit) while maintaining **near-linear scalability** that GPUs struggle to achieve. The tree topology broadcasts weights from MemoryX to the entire cluster and reduces (sums) gradients in the opposite direction back to MemoryX. Unlike GPU clusters requiring complex model partitioning (tensor parallelism, pipeline parallelism, sequence parallelism), Cerebras clusters **program like a single chip** using only data parallelism. The decoupled compute and memory architecture enables flexible scaling — add more CS-3s for faster training, or add more MemoryX for larger models. This "single logical device abstraction" simplifies distributed computing, eliminating the coordination overhead that plagues multi-GPU training.

### Cerebras Software Language (CSL) - Dataflow Programming

The **Cerebras SDK** enables developers to write low-level kernels using **CSL (Cerebras Software Language)**, a C-like language built around a **dataflow programming model**. CSL is statically typed with syntax similar to C/C++, supporting signed and unsigned integers (16 and 32-bit), half-precision and single-precision floating point, and booleans. In the dataflow model, **computation is triggered by the arrival of data** on a processing element (PE) from the fabric that knits cores together, rather than by explicit instruction scheduling. The SDK includes a **cycle-accurate simulator** and debugging tools that allow developers to debug code without hardware access. The **cslc compiler** statically schedules execution, and the **host runtime** manages launching programs on the wafer and moving data between CS-3 and host servers. Researchers at TotalEnergies, KAUST, ANL, PSC, and EPCC use the Cerebras SDK for computational science applications including seismic processing and Monte Carlo particle transport.

### Real-World Performance & Enterprise Adoption

Cerebras achieves production performance that redefines AI training economics: a **2048 CS-3 cluster trains Llama2-70B from scratch in less than a day** versus approximately one month on Meta's GPU cluster, representing **30x faster training**. The **Condor Galaxy 3 supercomputer** (64 CS-3 systems) delivers **8 exaflops** of compute. Single CS-3 systems deliver **2x tokens/second** compared to CS-2 when running Llama 2, Falcon 40B, and MPT-30B models in real-world testing. The architecture's unstructured sparsity support (via Qualcomm partnership) enables efficient inference optimization. Unlike GPU clusters that require months of engineering to partition large models across thousands of devices, Cerebras systems scale from single CS-3 to 2048-system clusters **without changing the program** — the weight streaming and SwarmX architecture abstract away distributed computing complexity.

**Salary Ranges**: Software Engineer median $285K total comp (L2 $193K, L12 $295K) | Hardware Engineer $188K | Solution Architect up to $601K | Intern $35.85/hour

---

## Sources

**Cerebras Technical Blogs**:
- [Cerebras Architecture Deep Dive: HW/SW Co-Design for Deep Learning](https://www.cerebras.ai/blog/cerebras-architecture-deep-dive-first-look-inside-the-hw-sw-co-design-for-deep-learning)
- [Announcing the Cerebras Architecture for Extreme-Scale AI](https://www.cerebras.ai/blog/announcing-the-cerebras-architecture-for-extreme-scale-ai)
- [Cerebras CS-3: The World's Fastest and Most Scalable AI Accelerator](https://www.cerebras.ai/blog/cerebras-cs3)
- [Cerebras CS-3 vs. Nvidia B200: 2024 AI Accelerators Compared](https://www.cerebras.ai/blog/cerebras-cs-3-vs-nvidia-b200-2024-ai-accelerators-compared)
- [Supercharge Your HPC Research with the Cerebras SDK](https://www.cerebras.ai/blog/supercharge-your-hpc-research-with-the-cerebras-sdk)

**Cerebras SDK & CSL**:
- [Cerebras SDK Documentation](https://sdk.cerebras.net/)
- [CSL Language Guide](https://sdk.cerebras.net/csl/language_index)
- [What's New in R0.6 of the Cerebras SDK](https://www.cerebras.ai/blog/whats-new-in-r0.6-of-the-cerebras-sdk)
- [GitHub: CSL Examples](https://github.com/Cerebras/csl-examples)

**System Architecture**:
- [WSE-3 Chip Specifications](https://www.cerebras.ai/chip)
- [CS-3 System](https://www.cerebras.ai/system)
- [Cerebras Wafer-Scale Cluster Documentation](https://docs.cerebras.net/en/latest/wsc/Concepts/how-cerebras-works.html)

**Company Information**:
- [Cerebras Company Overview](https://www.cerebras.ai/company)
- [Cerebras Wikipedia](https://en.wikipedia.org/wiki/Cerebras)
- [Andrew Feldman CEO Profile](https://www.clay.com/dossier/cerebras-systems-ceo)

**Job Postings & Compensation**:
- [Cerebras Careers - TheLadders](https://www.theladders.com/company/cerebrassystems-jobs)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/cerebras-systems/salaries)
- [Glassdoor Salaries](https://www.glassdoor.com/Salary/Cerebras-CA-Salaries-E1821335.htm)

---

*Last updated: November 30, 2025*
