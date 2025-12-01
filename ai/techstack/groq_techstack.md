# Groq - Technology Stack

**Company:** Groq, Inc.
**Founded:** 2016
**Focus:** Custom AI inference chips (Language Processing Unit)
**Headquarters:** Mountain View, California

---

## Non-AI Tech Stack

Groq operates a **geo-agnostic remote-first engineering culture** with primary hardware labs in **San Jose, California**. The deployment infrastructure includes **GroqCloud** (global public cloud platform across four regions) and **GroqRack** (on-premises deployment for regulated industries and air-gapped environments). The platform supports **SOC 2, GDPR, and HIPAA compliance** with optional private tenancy for sensitive workloads. Groq's software-first architecture centers on a proprietary **compiler technology** that statically schedules execution down to individual clock cycles, choreographing all hardware operations. The infrastructure uses **air-cooled design** requiring minimal data center overhead compared to liquid-cooled GPU systems. Groq was founded by **Jonathan Ross**, former Google engineer who designed the **Tensor Processing Unit (TPU)** at Google X. The leadership team includes executives from Google, Netflix, Palo Alto Networks, and Pivotal Software. The company emphasizes real-world project portfolios over interview-style challenges, seeking engineers who demonstrate ownership and rapid learning ability.

**Salary Ranges**: Software Engineer median $361,650 total comp (L4 $200K, L5 $362K, L6 $334K) | Hardware Engineer $253K | Solution Architect $356K | Base salary range $189K-$332K

---

## AI/ML Tech Stack

### LPU (Language Processing Unit) - Purpose-Built Inference ASIC

**What's unique**: Groq designed the **first chip purpose-built exclusively for AI inference** rather than adapting general-purpose processors. The **LPU** (originally called **Tensor Streaming Processor**) features a **single-core, deterministic architecture** with **230 MB of on-chip SRAM** delivering **80 TB/s memory bandwidth** — **10x faster than GPU HBM's ~8 TB/s**. This SRAM serves as **primary weight storage, not cache**, fundamentally eliminating memory hierarchy bottlenecks. The chip achieves **750 TOPS at INT8** and **188 TeraFLOPS at FP16** with **320x320 fused dot product matrix multiplication** and **5,120 Vector ALUs**. Unlike GPUs designed for independent parallel operations (graphics), the LPU specifically targets **sequential linear algebra operations** fundamental to transformer models, achieving **up to 10x better energy efficiency** than GPUs for AI inference.

### Software-First Architecture - Compiler Choreographs Hardware

**What makes it different**: Groq's architecture inverts traditional chip design by giving the **compiler complete control over execution**. The compiler **statically schedules every operation down to individual clock cycles**, including inter-chip communication patterns, before runtime. This eliminates all sources of non-determinism found in CPUs/GPUs: **no cache misses, no branch prediction, no out-of-order execution, no speculative execution, no dynamic scheduling overhead**. The model-independent compiler enables developers to deploy any model without writing custom kernels, unlike GPUs requiring per-model optimization. By shifting control to software, Groq avoids "dark silicon" waste — transistors and memory bandwidth otherwise consumed by control circuitry are redirected toward computational performance. This **software-defined hardware** approach provides sustainable competitive advantage beyond what process technology scaling alone can achieve.

### TruePoint Numerics - Variable Precision Without Accuracy Loss

Groq employs **TruePoint numerics**, a precision-reduction strategy that maintains accuracy by storing **100 bits of intermediate accumulation** for lossless computations while selectively quantizing outputs based on downstream error sensitivity. The compiler applies varying precision levels: **FP32 for attention logits** (high sensitivity), **Block Floating Point for Mixture-of-Experts weights**, and **FP8 for error-tolerant activation layers**. This heterogeneous precision approach yields **2-4x speedup over BF16 with no appreciable accuracy loss**, enabling faster inference without the quality degradation typical of uniform quantization strategies.

### Deterministic Compute & Tensor Streaming Architecture

**What sets Groq apart**: The LPU features a **functionally sliced microarchitecture** where memory units are interleaved with vector and matrix computation units arranged in vertical slices by type: **MXM** (matrix operations), **SXM** (shift/rotate), **MEM** (memory), **VXM** (vector arithmetic), and **ICU** (instruction control). Each slice contains **20 tiles processing 16-element vectors**, yielding **320 total SIMD lanes**. Data flows as numbered streams (0-31) between slices via programmable "conveyor belts" — a streaming architecture that eliminates synchronization overhead. The compiler performs **two-dimensional scheduling of instructions and data in both time and space**, exposing temporal information to enable deterministic execution. This design removes contention for critical resources (data bandwidth and compute), ensuring **consistent performance without variability**.

### RealScale Multi-Chip Fabric - Hundreds of LPUs as Single Core

Groq's **RealScale technology** interconnects hundreds of LPU chips to function as **one shared resource fabric** without traditional bottlenecks. Individual LPUs connect via **plesiosynchronous, chip-to-chip protocol** with direct connectivity, eliminating complex switches and routers. The compiler predicts data arrival precisely down to clock cycles, enabling **software-scheduled networking** across the multi-chip fabric. This architecture supports both **tensor parallelism** (distributing single operations across processors to reduce latency) and **pipeline parallelism** (distributing layers across chips) with guaranteed synchronization. The result: **near-linear scalability** that GPUs struggle to match, maintaining consistent output speeds even for very large models like **Llama 4 Maverick (400B parameter MoE)**, which Groq deployed on release day.

### Industry-Leading Inference Performance

Groq achieves benchmark-leading speeds on production models: **Mixtral 8x7B at 480 tokens/second**, **Llama 2 70B at 300 tokens/second**, and successful deployment of **Llama 4 Maverick (400B MoE)** immediately upon release. The LPU's deterministic architecture enables **real-time AI applications at scale** with predictable latency — critical for production deployments where variability causes SLA violations. The on-chip SRAM and streaming architecture eliminate the "waiting for compute or memory resources" bottlenecks inherent in GPU hub-and-spoke models, providing consistent performance regardless of model size.

### Deployment Options & Enterprise Readiness

Groq offers dual deployment paths: **GroqCloud** provides global public cloud access across four regions with auto-scaling capabilities, while **GroqRack** enables on-premises deployment for regulated industries or air-gapped environments. The platform supports seamless transitions between cloud and on-prem deployments. Air-cooled LPU design reduces data center infrastructure requirements compared to liquid-cooled GPU systems. Enterprise features include **SOC 2, GDPR, and HIPAA compliance** with optional private tenancy for sensitive workloads.

**Salary Ranges**: Software Engineer median $361,650 total comp | Hardware Engineer $253K | Software Engineering Manager $275K | Solution Architect $356K | Sales $512K | Base $189K-$332K

---

## Sources

**Groq Technical Blogs**:
- [What is a Language Processing Unit?](https://groq.com/groq-tensor-streaming-processor-architecture-is-radically-different/)
- [Inside the LPU: Deconstructing Groq's Speed](https://groq.com/blog/inside-the-lpu-deconstructing-groq-speed)
- [Why AI Requires a New Chip Architecture](https://groq.com/blog/why-ai-requires-a-new-chip-architecture)
- [From Speed to Scale: How Groq Is Optimized for MoE & Other Large Models](https://groq.com/blog/from-speed-to-scale-how-groq-is-optimized-for-moe-other-large-models)
- [World, Meet Groq](https://groq.com/blog/world-meet-groq-2)

**External Technical Analysis**:
- [The Architecture of Groq's LPU - Abhinav Upadhyay](https://blog.codingconfessions.com/p/groq-lpu-design)

**Company Information**:
- [About Groq](https://groq.com/about-groq)
- [Groq Wikipedia](https://en.wikipedia.org/wiki/Groq)

**Job Postings & Compensation**:
- [Groq Careers](https://groq.com/careers)
- [Groq Jobs - LinkedIn](https://www.linkedin.com/company/groq/jobs)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/groq/salaries) - Software Engineer median $361,650 total comp

---

*Last updated: November 30, 2025*
