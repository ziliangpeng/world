# Together AI - Technology Stack

**Company:** Together AI
**Founded:** 2022
**Focus:** AI Acceleration Cloud for open-source and enterprise AI
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

Together AI operates on **AWS infrastructure** (EC2, S3, Kinesis, Lake Formation, IAM, TimeStream) with a polyglot backend using **Golang, Rust, Python, C++, TypeScript, Java, and Haskell**. The platform leverages **Kubernetes** for orchestration with **Terraform** for infrastructure-as-code, **ArgoCD** and **Skaffold** for continuous deployment, and **GitHub Actions** for CI/CD. Data infrastructure includes **ClickHouse**, **Apache Kafka**, and **Redpanda** for streaming. Monitoring runs on **Prometheus** and **Grafana**. The commerce platform integrates **Stripe** for payments. Network architecture uses advanced protocols including **RoCEv2**, **Ethernet fabrics**, **Infiniband**, **EVPN/VXLAN**, **SR-MPLS/SRv6**, **BGP**, and **OSPF** for high-performance GPU interconnects. The platform supports infrastructure-as-code tools **Skypilot** and **Terraform** for user deployments. Together AI operates data centers in **25+ cities** across North America with **200 MW of secured power capacity** and access to **100,000+ GPUs** throughout 2025, including facilities in **Maryland** (NVIDIA B200) and **Memphis, Tennessee** (GB200/GB300).

**Salary Ranges**: Software Engineer $160K-$375K (median $375K total comp, up to $547K) | Infrastructure Engineer $160K-$230K base + equity | Senior roles $200K+ base

---

## AI/ML Tech Stack

### Together Inference Engine - Proprietary Performance Leader

**What's unique**: Together AI built a **proprietary Inference Engine** that delivers **4x faster inference than vLLM**, the industry-standard open-source framework. The engine integrates **advanced quantization techniques** and custom kernel optimizations to achieve **2-3x faster inference than today's hyperscaler solutions** (AWS, GCP, Azure). The platform supports **200+ open-source models** across all modalities (chat, image, audio, vision, code, embeddings) with **99.9% uptime SLA** and **North American data centers**. Together AI serves **450,000+ AI developers** and processes massive inference workloads using **SGLang** and **vLLM** frameworks with proprietary optimizations layered on top.

### Together Kernel Collection (TKC) - FlashAttention-3 Integration

**What makes it different**: Together AI's Chief Scientist **Tri Dao** is the creator of **FlashAttention**, the breakthrough attention mechanism that revolutionized transformer efficiency. **FlashAttention-3** achieves **1.5-2x speedup over FlashAttention-2** and reaches **740 TFLOPs/s (75% of theoretical maximum on H100 GPUs)**, optimized specifically for Hopper architecture. The **Together Kernel Collection (TKC)** packages these innovations to provide **up to 10% faster training** and **75% faster inference** compared to standard implementations. These kernels are deeply integrated into Together's Inference Engine, giving customers access to cutting-edge optimizations without custom engineering. FlashAttention-3 uses **asynchrony and low-precision computation** to maximize GPU utilization while maintaining accuracy.

### ATLAS - Runtime-Learning Accelerators

Together AI pioneered **ATLAS (AdapTive-LeArning Speculator System)**, described as "a new paradigm in LLM inference via runtime-learning accelerators." ATLAS delivers **up to 4x faster LLM inference** through adaptive speculation techniques that learn optimal decoding strategies at runtime rather than relying on static heuristics. This represents a fundamental shift from traditional speculative decoding approaches, enabling the system to continuously improve inference performance based on observed request patterns.

### Research Innovations - Open-Source Contributions

**What sets Together AI apart**: The research lab operates at the intersection of AI and systems research, pioneering breakthrough methods that are released as **open-source contributions** to accelerate the entire industry. Key innovations include: **(1) Mixture of Agents (MoA)** - leverages collective strengths of multiple LLMs to achieve state-of-the-art results; **(2) Medusa** - speculative decoding for faster generation; **(3) Sequoia** - optimized tree-based speculation; **(4) Hyena** and **(5) Mamba** - efficient alternatives to attention mechanisms; **(6) CocktailSGD** - optimized distributed training. This research-first culture differentiates Together AI from pure infrastructure providers, as each innovation directly translates to customer performance gains.

### Massive NVIDIA GPU Infrastructure

Together AI operates one of the world's largest optimized GPU clusters, co-built with **Hypertec Cloud** featuring **36,000 NVIDIA GB200 NVL72 GPUs** (deployment started Q1 2025). The **GB200 NVL72** connects **72 Blackwell GPUs and 36 Grace CPUs** into one liquid-cooled, memory-coherent rack, delivering **30x faster real-time inference for trillion-parameter models** and **4x accelerated training** compared to previous architectures. The infrastructure spans multiple North American data centers with immediate access to thousands of **H100, H200, B200, and GB300 GPUs**. Together AI partnered with **Pegatron** and **5C** to deploy **liquid-cooled racks** (Maryland facility) and **air-cooled systems** for distributed workloads. The company secured **200 MW of power capacity** and **100,000+ GPUs** for 2025, enabling massive-scale training and inference for open-source and enterprise customers.

### Platform Services - End-to-End AI Lifecycle

Together AI's **AI Acceleration Cloud** spans the entire AI lifecycle: **(1) Inference** - production-grade API with 4x faster performance; **(2) Fine-tuning** - supports larger models and longer contexts with enhanced integrations; **(3) Training** - distributed training on frontier hardware; **(4) Agentic workflows** - built-in code interpretation for AI agents; **(5) Synthetic data generation** - high-quality training data at scale; **(6) Batch inference API** - 3000x rate limit increase for large-scale processing. The platform integrates with **NVIDIA DGX Cloud**, **MongoDB Atlas** for RAG applications, and supports **infrastructure-as-code** deployment via Skypilot and Terraform. Together AI collaborated with **Meta** to optimize **Llama 3.1** models, achieving accelerated performance at full accuracy.

**Salary Ranges**: Software Engineer $160K-$375K (median $375K) | ML Engineer $160K-$230K base + equity | Backend Engineer Inference Platform (Rust/Go/Python/CUDA) | Research Scientist | Senior/Staff roles $200K-$300K+ base

---

## Sources

**Together AI Technical Content**:
- [ATLAS: Runtime-Learning Accelerators](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas)
- [Tri Dao and FlashAttention-2](https://www.together.ai/blog/tri-dao-flash-attention)
- [FlashAttention-3 Technical Paper](https://tridao.me/blog/2024/flash3/)
- [Together Inference Engine](https://www.together.ai/inference)
- [Together Products Overview](https://www.together.ai/products)

**Infrastructure & GPU Clusters**:
- [NVIDIA GB200 Cluster with 36K Blackwell GPUs](https://www.together.ai/blog/nvidia-gb200-together-gpu-cluster-36k)
- [NVIDIA GB200 NVL72 Clusters](https://www.together.ai/nvidia-gb200-nvl72)
- [GPU Clusters: GB200, B200, H200, H100](https://www.together.ai/gpu-clusters)
- [Instant GPU Clusters](https://www.together.ai/instant-gpu-clusters)

**Research & Innovations**:
- [Mixture of Agents (MoA) Documentation](https://docs.together.ai/docs/mixture-of-agents)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention-3 GitHub](https://github.com/togethercomputer/flash-attention-3)

**Partnerships**:
- [Meta Llama 3.1 Partnership](https://www.together.ai/blog/meta-llama-3-1)
- [MongoDB Atlas RAG Integration](https://www.mongodb.com/blog/post/together-ai-advancing-frontier-open-source-embeddings-inference-atlas)
- [$305M Series B Funding Announcement](https://www.together.ai/blog/together-ai-announcing-305m-series-b)

**AI/ML Job Postings**:
- [Together AI Careers - Greenhouse](https://job-boards.greenhouse.io/togetherai)
- [Senior Backend Engineer, Inference Platform](https://job-boards.greenhouse.io/togetherai) - Rust, Go, Python, CUDA, SGLang, vLLM
- [ML Engineer](https://job-boards.greenhouse.io/togetherai) - Python, Go, Rust, C/C++, vLLM, SGLang, TensorRT
- [Infrastructure Engineer, Data Platform](https://job-boards.greenhouse.io/togetherai) - AWS, Terraform, $160K-$230K base
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/together-ai/salaries)

---

*Last updated: November 30, 2025*
