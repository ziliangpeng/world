# Baseten - Technology Stack

**Company:** Baseten
**Founded:** 2019
**Focus:** ML model deployment and inference platform
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

Baseten operates a **multi-cloud infrastructure** managing **thousands of GPUs** distributed across **10+ cloud providers** in dozens of regions globally, partnering with **9+ CSPs** through their proprietary **Multi-Cloud Capacity Management (MCM)** system. Primary infrastructure runs on **AWS EC2 P4d instances** with **NVIDIA A100 Tensor Core GPUs** (8 GPUs per instance, 400Gbps networking, NVLink/NVSwitch/NCCL/GPUDirect RDMA). Supporting hardware includes **NVIDIA A10G, T4, V100, H100** GPUs. The platform uses **NVIDIA Triton Inference Server** for model serving, **vLLM** and **TensorRT/TensorRT-LLM** for inference optimization, and integrates **PyTorch**, **TensorFlow**, and other frameworks. **Truss**, Baseten's open-source packaging library (GitHub: basetenlabs/truss), provides framework-agnostic model deployment without Docker, bridging development to production. The deployment model offers **serverless Model APIs** (one-click pre-optimized models) and **Dedicated Deployments** (custom hardware selection). Infrastructure automation handles **request routing** (KV cache-aware, geo-proximity, LoRA cache routing), **SLA-aware autoscaling** (spins up/down replicas in seconds), and **optimized cold starts** (5-10 seconds vs 5-minute baseline, **30-60X speedup**).

**Salary Ranges**: Software Engineer $230K median total comp (up to $342K) | ML Engineer L4 $237K ($160K base + $77K stock) | Forward Deployed Engineer $131K-$214K

---

## AI/ML Tech Stack

### The Baseten Inference Stack - Two-Layer Architecture

**What's unique**: Baseten built a proprietary **Baseten Inference Stack** with two tightly integrated layers optimized specifically for production inference (not training). **(1) Inference Runtime** - focuses on *how models run*, including runtime optimizations (TensorRT-LLM, vLLM, custom kernels), speculation engines for faster decoding, and framework-specific tuning. **(2) Inference-Optimized Infrastructure** - focuses on *connecting users to models reliably*, including intelligent request routing, SLA-aware autoscaling, multi-cloud capacity management, and cold start optimization. This separation enables Baseten to optimize each layer independently while maintaining tight integration - runtime improvements benefit from infrastructure scalability, while infrastructure innovations (like KV cache-aware routing) leverage runtime-level visibility. The stack delivers **40% lower latency on SDXL**, **sub-200ms time to first token on Mixtral 8x7B**, and **3x higher 7B LLM throughput on H100 GPUs**.

### Truss - Open-Source Model Packaging

**What makes it different**: **Truss** is Baseten's open-source framework for packaging any model (PyTorch, TensorFlow, TensorRT, Triton) for deployment anywhere - localhost or production - without Docker complexity. The development-to-production bridge makes serving a model locally as straightforward as deploying to production, enabling rapid dev/test loops. Truss integrates the **TensorRT-LLM Engine Builder** for automatic LLM optimization, achieving **2X inference performance improvements** (e.g., boosting a customer's LLM deployment on AWS). The framework handles dependency management, environment configuration, and deployment packaging, abstracting MLOps complexity from data scientists. Truss serves as both Baseten's commercial foundation and a standalone open-source tool (10K+ GitHub stars), democratizing production model serving beyond Baseten's platform.

### Multi-Cloud Capacity Management (MCM)

**What's unique**: Baseten's **MCM system** manages compute across **10+ cloud service providers** from a single control plane, operating **9+ partnerships** with cloud vendors globally. This enables: (1) **Geographic distribution** - route requests to GPUs in proximate regions for lower latency; (2) **Capacity arbitrage** - dynamically allocate workloads to providers with available capacity; (3) **Cost optimization** - leverage competitive pricing across vendors; (4) **Resilience** - failover across clouds if one provider experiences outages. MCM handles heterogeneous networking (NVIDIA Infiniband to custom solutions) and unifies deployments across Baseten Cloud, self-hosted, or hybrid configurations. Users can **deploy once, run anywhere** - Baseten handles orchestration across AWS, GCP, Azure, and specialized GPU clouds.

### Intelligent Request Routing & Autoscaling

**LLM-aware routing** represents Baseten's competitive advantage: **KV cache-aware routing** directs incoming requests to model replicas with **previously stored context** (cached key-value pairs from prior conversation turns), dramatically reducing redundant computation for multi-turn interactions. The system also routes to **warm LoRA caches** for fine-tuned model variants and **geographically proximate GPUs** for latency optimization. **SLA-aware autoscaling** analyzes incoming traffic patterns and spins up/down replicas **in seconds** (vs minutes industry-standard) to maintain SLAs while optimizing cost. **Optimized cold starts** load model weights and provision GPUs in **5-10 seconds** (30-60X faster than the 5-minute baseline), enabled by weight pre-loading, optimized container initialization, and GPU provisioning automation.

### Model Library & Serverless Inference

Baseten's **Model Library** provides **one-click deployment** of pre-optimized popular models (Llama, Whisper, DeepSeek, Qwen, Stable Diffusion) via **Model APIs**. Each Model API ships with **Baseten Inference Stack optimizations** applied, so users get ultra-fast performance out of the box without tuning. The **serverless Model APIs** handle autoscaling, routing, and infrastructure automatically - users only pay for compute consumed. For production workloads, users can transition from serverless Model APIs to **Dedicated Deployments** in **two clicks**, selecting custom hardware (A100, H100, A10G) while retaining the inference stack optimizations. This tiered deployment model supports both rapid experimentation (serverless) and production scale (dedicated).

### NVIDIA & AWS Partnership Integration

Baseten's infrastructure deeply integrates **NVIDIA software stack**: **TensorRT-LLM** for LLM inference acceleration, **Triton Inference Server** for multi-framework serving, and **NVIDIA GPUs** (A100, H100, A10G, T4, V100) for compute. The **AWS partnership** provides access to **EC2 P4d instances** (8x A100 per instance, 400Gbps networking) and **AWS UltraClusters** (4,000+ GPUs on-demand). Baseten's engineering team achieved **2X faster inference with NVIDIA Dynamo** and productized these optimizations through Truss. The combination enables Baseten to deliver cutting-edge performance (sub-200ms TTFT on Mixtral) while maintaining operational simplicity for users.

**Salary Ranges**: Software Engineer median $230K total comp | ML Engineer L4 $237K ($160K base + $77K stock) | Senior/Lead roles up to $342K total comp

---

## Sources

**Baseten Infrastructure & Architecture**:
- [The Baseten Inference Stack](https://www.baseten.co/resources/guide/the-baseten-inference-stack/)
- [How Baseten works](https://docs.baseten.co/concepts/howbasetenworks)
- [Cloud-native infrastructure](https://www.baseten.co/platform/cloud-native-infrastructure/)
- [Multi-cloud capacity management (MCM)](https://www.baseten.co/blog/how-baseten-multi-cloud-capacity-management-mcm-powers-cloud-self-hosted-and-hybr/)
- [Baseten Hybrid: control in your cloud and ours](https://www.baseten.co/blog/baseten-hybrid-control-and-flexibility-in-your-cloud-and-ours/)

**Truss Open Source**:
- [GitHub: basetenlabs/truss](https://github.com/basetenlabs/truss)
- [Why we built and open-sourced Truss](https://www.baseten.co/blog/why-we-open-sourced-truss/)
- [Automatic LLM optimization with TensorRT-LLM Engine Builder](https://www.baseten.co/blog/automatic-llm-optimization-with-tensorrt-llm-engine-builder/)

**Performance & Optimization**:
- [High performance ML inference with NVIDIA TensorRT](https://www.baseten.co/blog/high-performance-ml-inference-with-nvidia-tensorrt/)
- [How Baseten achieved 2x faster inference with NVIDIA Dynamo](https://www.baseten.co/blog/how-baseten-achieved-2x-faster-inference-with-nvidia-dynamo/)
- [How Baseten achieves 225% better cost-performance - Google Cloud Blog](https://cloud.google.com/blog/products/ai-machine-learning/how-baseten-achieves-better-cost-performance-for-ai-inference)

**Partnerships**:
- [Baseten's AI Inference Infrastructure - NVIDIA Case Study](https://www.nvidia.com/en-us/customer-stories/baseten/)
- [Baseten Delivers AI Inference with AWS and NVIDIA](https://aws.amazon.com/partners/success/baseten-nvidia/)

**Model Library & Deployment**:
- [Production-First Model APIs](https://www.baseten.co/products/model-apis/)
- [Dedicated Deployments](https://www.baseten.co/products/dedicated-deployments/)
- [Deploy from model library](https://docs.baseten.co/deploy/library)

**Jobs & Compensation**:
- [Baseten Careers](https://www.baseten.co/careers/)
- [Baseten Jobs - Ashby](https://jobs.ashbyhq.com/baseten)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/baseten/salaries)

---

*Last updated: November 30, 2025*
