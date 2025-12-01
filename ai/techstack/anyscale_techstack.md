# Anyscale - Technology Stack

**Company:** Anyscale, Inc.
**Founded:** 2019
**Focus:** Ray-based distributed computing platform for AI workloads
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

Anyscale operates from **San Francisco** with infrastructure on **AWS and Google Cloud Platform**, plus **Microsoft Azure** (private preview November 2025, GA 2026). Founded in **2019** by **Robert Nishihara** (CEO), **Ion Stoica** (Executive Chairman), **Philipp Moritz** (CTO), and **Michael I. Jordan** (University of California, Berkeley professor), the company raised **$281 million total funding** at a **$1+ billion valuation** (unicorn status) from investors including **Andreessen Horowitz, NEA, Addition, Intel Capital, Ant Financial, and Amplify Partners**. The founding team created **Ray** at **UC Berkeley RISELab** (successor to **AMPLab** that created **Apache Spark and Databricks**, which Stoica co-founded). In **October 2025**, **Ray joined the PyTorch Foundation** as a foundation-hosted project, forming a **unified open source AI compute stack** with **PyTorch** (model development), **vLLM** (inference), and **Ray** (distributed execution). Ray has **27 million monthly downloads** and **39,000 GitHub stars**, with downloads growing **nearly 10x over the past year**. The platform supports **multi-cloud deployment** via **VM or Kubernetes clusters** with **fault-tolerant deployments, zero-downtime upgrades, and automatic rollback**. Infrastructure includes **managed Prometheus/Grafana monitoring**, **persistent logging**, and **spot instance management** with on-demand fallback. The platform provides a **cloud-based IDE** accessible via **VSCode, Jupyter, and Cursor** with **interactive development console** and **profiling tools** for distributed systems. Backend uses **Python-native Ray framework** with **automatic dependency propagation** across Ray nodes. The **Microsoft Azure partnership** delivers **AI-native computing on Azure** through **Multi-Resource Cloud** extending Anyscale Cloud to hook up **multiple cloud resources** across different regions, providers, or stacks (VM/K8s). Customers include **Uber** (Michelangelo ML platform), **Instacart** (ML training), **Shopify** (Merlin ML platform), **OpenAI**, **Netflix**, **Lyft**, **Cruise**, **ByteDance**, and others scaling ML workloads with Ray.

**Salary Ranges**: Software Engineer $157K-$525K (median $350K total comp) | Infrastructure Engineer $202K-$237K base | Indeed reports $197K (11 salaries) | Equity vests over 4 years (25% year 1, then 2.08% monthly)

---

## AI/ML Tech Stack

### Ray Framework - Unified Distributed Computing for AI Workloads at Scale

**What's unique**: **Ray** is a **unified framework for scaling AI and Python applications** with **27 million monthly downloads** and **39,000 GitHub stars**, providing a **compute layer for parallel processing** so developers don't need distributed systems expertise. Ray consists of a **core distributed runtime** and a **set of AI Libraries** including **RLlib** (reinforcement learning), **Ray Tune** (hyperparameter tuning), **Ray Serve** (model serving), **Ray Train** (distributed training), and **Ray Data** (data processing), enabling developers to **scale from laptops to tens of thousands of nodes** without code changes. Unlike fragmented tools requiring separate systems for data prep (Spark), training (Horovod), tuning (Optuna), and serving (KServe), Ray provides a **single unified framework** where the same code runs locally and at scale. The architecture implements **dynamic task graphs** where Ray automatically schedules tasks across distributed workers, handling **fault tolerance, load balancing, and resource allocation** transparently. Ray's **actor model** enables stateful computation — long-running processes that maintain state across invocations — critical for ML applications like parameter servers, reinforcement learning environments, and streaming pipelines. The framework supports **heterogeneous clusters** mixing CPUs, GPUs, and TPUs with **resource-aware scheduling** ensuring tasks land on appropriate hardware. Ray's **autoscaling** dynamically provisions and deprovisions nodes based on workload demands, reducing costs during idle periods. In **October 2025**, Ray joined the **PyTorch Foundation** alongside PyTorch and vLLM, creating a **unified open source AI compute stack** where PyTorch handles model development, vLLM optimizes inference, and Ray orchestrates distributed execution across any accelerator at any scale.

### vLLM Integration - 23x Inference Throughput with Continuous Batching

**What makes it different**: Anyscale combines **Ray Serve for orchestration with vLLM for high-performance inference**, achieving **up to 23x LLM inference throughput** while reducing **p50 latency** compared to Hugging Face Transformers and **2.5–3x gains over existing continuous batching engines** like TGI (Text Generation Inference). The integration implements **continuous batching** where new requests join in-flight batches without waiting for previous requests to complete, more than doubling performance compared to naive batching. Anyscale's **proprietary vLLM optimizations** tune engine performance to **reduce batch and online inference costs by up to 20%** through memory management, prefill scheduling, and kernel fusion. The system uses **PagedAttention** allocating GPU memory in fixed-size blocks to eliminate fragmentation, enabling **near-zero waste** from dynamic sequence lengths. **Automatic Prefix Caching (APC)** stores **KV cache blocks** from previous requests and reuses them when new requests share string prefixes, dramatically accelerating **multi-turn conversations and RAG applications** where system prompts remain constant. Anyscale rapidly provisions **production-ready HTTPS endpoints** or **fault-tolerant batch inference jobs** with **autoscaling based on queue depth** — adding GPUs when request queues grow and releasing them when idle. The platform supports **quantization methods** (FP8, INT8, INT4) balancing throughput and quality, plus **speculative decoding** where a small draft model generates tokens verified by the full model for **2-3x speedup** on certain workloads. The integration handles **mixed-precision inference** automatically selecting optimal data types per operation, and supports **multi-GPU tensor parallelism** for models exceeding single-GPU memory.

### Anyscale Endpoints - 10x Cost Reduction for Open-Source LLMs

**What sets Anyscale apart**: **Anyscale Endpoints** is a **serverless LLM serving platform** offering **10x more cost-effective** serving for the **most popular open-source LLMs** (Llama, Mistral, Mixtral, Qwen) compared to commercial API providers while maintaining **OpenAI API compatibility** for seamless migration with minimal code changes. The platform uses a **serverless approach** mitigating infrastructure complexities — users call APIs without managing clusters, with Anyscale handling **autoscaling, instance selection, spot instance orchestration, and fault tolerance** behind the scenes. The architecture separates **control plane from data plane**, deploying Ray clusters in **private subnets** accessed over **private IP addresses** for additional security. Endpoints support **advanced autoscaling** reacting to request volume within seconds, **smart instance selection** choosing optimal GPU types per model size and latency requirements, and **intelligent spot instance support** using **60-80% cheaper spot instances** with automatic fallback to on-demand when spot capacity is unavailable. Users can deploy Endpoints within their **existing AWS or GCP cloud accounts** for data privacy and compliance, or use Anyscale's fully-managed infrastructure. The platform supports **user-customized Docker images** enabling proprietary models, custom inference engines (vLLM, TRT, TGI), and specialized dependencies. The **10x cost reduction** comes from **efficient resource utilization** (spot instances, autoscaling to zero), **optimized inference engines** (vLLM with continuous batching), and **multi-tenant infrastructure** amortizing fixed costs across users.

### RLlib - Scalable Reinforcement Learning with Built-In Autoscaling

**What's unique**: **RLlib** is Ray's **scalable distributed reinforcement learning library** providing **production-level, highly distributed RL workloads** with **unified and simple APIs** for diverse applications including **games, autoscaling, finance, robotics, recommendations, and supply chain**. RLlib directly leverages **Ray's autoscaling capabilities** — as RL training demands more compute (collecting environment samples, training policy networks, computing value functions), Ray dynamically provisions workers and releases them when training completes. The architecture separates **sample collection** (rollout workers interacting with environments) from **policy training** (gradient computation on GPUs), enabling **heterogeneous clusters** where CPUs generate experiences and GPUs train models efficiently. RLlib supports **all major RL algorithms** including **DQN, PPO, SAC, DDPG, A3C, APEX, IMPALA, and R2D2**, with implementations validated against published results. The library provides **multi-agent RL** where multiple policies train simultaneously in shared environments, critical for **game playing, autonomous driving simulations, and multi-robot coordination**. RLlib integrates with **popular RL environments** (OpenAI Gym, Atari, MuJoCo, Unity ML-Agents) and enables **custom environments** through simple APIs. The framework handles **distributed experience replay buffers** across cluster nodes, **priority sampling** for important transitions, and **policy serving** deploying trained agents as Ray Serve endpoints for real-time inference. RLlib enables **massively parallel RL** — training policies on **millions of environment samples per second** by distributing rollout workers across thousands of CPUs.

### Anyscale Runtime - Production-Grade Ray with Higher Resilience

**What makes it different**: **Anyscale Runtime** is a **production-grade, Ray-compatible runtime** delivering **higher resilience, faster performance, and lower compute cost** vs. open-source Ray **without requiring code changes**. The runtime implements **proprietary optimizations** not available in open-source Ray, including **faster scheduling algorithms** reducing task launch overhead, **improved fault tolerance** with faster recovery from node failures, and **memory optimizations** reducing per-task memory footprint. Anyscale Runtime features **head node recovery** — if the Ray cluster's head node fails, the runtime automatically recovers cluster state from checkpoints, preventing full cluster restarts that lose in-flight work. **Multi-AZ support** deploys worker nodes across **multiple availability zones** ensuring training and inference workloads survive zone failures. **Zero-downtime upgrades** enable Ray version updates and dependency changes **without stopping running workloads** — new tasks use updated environments while in-flight tasks complete on old environments. The runtime provides **cost governance tools** including **budgets and quotas** preventing runaway spending from misconfigured autoscaling. **GPU utilization efficiency features** pack multiple small tasks onto single GPUs when possible and preempt low-priority tasks when high-priority workloads arrive. Anyscale Runtime is **API-compatible** with open-source Ray, meaning code written for Ray runs unchanged on Anyscale Runtime while gaining performance and reliability improvements.

### Customer Success at Scale - Uber 50% Savings, Instacart 12x Speedup

**What sets Anyscale apart**: Anyscale and Ray power ML platforms at **Uber, Instacart, Shopify, OpenAI, Netflix, Lyft, Cruise, and ByteDance**, demonstrating production validation across diverse industries. **Uber's Michelangelo ML platform** evolved from monolithic Spark-based architecture to **Ray-based unified and scalable ML compute**, achieving **50% savings on ML compute** by implementing large-scale deep learning on Ray heterogeneous clusters. Uber's **Autotune service** using **Ray Tune** experienced **up to 4x speedup on hyperparameter tuning jobs** compared to previous systems. **Instacart reduced training times by 12x** compared to **AWS Batch and Celery**, from days down to hours, by leveraging Ray's distributed training. Ray and Anyscale speed up machine learning training workloads **10x compared with tools like Celery, AWS Batch, SageMaker, Vertex AI, and Dask** through efficient distributed execution and resource management. **Shopify's ML platform team** chose to build **"The Magic of Merlin"** on top of **Kubernetes and Ray**, providing unified ML infrastructure for product recommendations, fraud detection, and demand forecasting. The **"Training 1 Million ML Models in Record Time"** use case demonstrates Ray's ability to **parallelize model training** across massive model counts — training separate models for each store, product, or customer segment — scenarios impossible with traditional single-model training pipelines. These deployments validate that Ray handles **production-scale workloads** with billions of predictions daily, petabytes of training data, and thousands of concurrent training jobs.

### PyTorch Foundation Integration - Unified Open Source AI Compute Stack

**What's unique**: In **October 2025**, the **PyTorch Foundation** announced Ray as its **newest foundation-hosted project**, creating a **unified open source AI compute stack** combining **PyTorch** (model development), **vLLM** (inference), and **Ray** (distributed execution). This integration positions Ray as the **standard distributed runtime for PyTorch-based AI applications**, providing **seamless interoperability** where PyTorch models train on Ray clusters and deploy via vLLM on Ray Serve endpoints without switching frameworks. The stack enables **end-to-end workflows** — data preprocessing (Ray Data), distributed training (Ray Train + PyTorch), hyperparameter tuning (Ray Tune), and production serving (Ray Serve + vLLM) — all using **consistent APIs** and **shared resource management**. The **Linux Foundation** governance ensures Ray remains **vendor-neutral** and **community-driven**, preventing lock-in to Anyscale's commercial platform while enabling broader ecosystem adoption. The integration accelerates **enterprise adoption** — organizations already standardized on PyTorch can add distributed capabilities by adopting Ray without introducing new ML frameworks. The combined stack addresses the **full AI lifecycle** from research (PyTorch) to production (Ray + vLLM), eliminating the **research-to-production gap** where different frameworks are used for experimentation and deployment. The partnership demonstrates Ray's **strategic importance** in the AI ecosystem, positioning it alongside PyTorch as foundational infrastructure.

---

## Sources

**Anyscale Official**:
- [Anyscale Homepage](https://www.anyscale.com/)
- [Anyscale Careers](https://www.anyscale.com/careers)
- [Anyscale User Stories](https://www.anyscale.com/user-stories)
- [Anyscale LLM Suite](https://www.anyscale.com/product/platform/llm-suite)
- [Anyscale LLM Online Inference](https://www.anyscale.com/use-case/llm-online-inference)

**Ray Framework**:
- [Ray Homepage](https://www.ray.io/)
- [Ray Documentation](https://docs.ray.io/)
- [Ray GitHub Repository](https://github.com/ray-project/ray)
- [Ray Overview](https://docs.ray.io/en/latest/ray-overview/index.html)
- [RLlib: Scalable Reinforcement Learning](https://docs.ray.io/en/latest/rllib/index.html)

**PyTorch Foundation Integration**:
- [Ray Joins PyTorch Foundation](https://www.anyscale.com/blog/ray-by-anyscale-joins-pytorch-foundation)
- [PyTorch Foundation Welcomes Ray](https://www.linuxfoundation.org/press/pytorch-foundation-welcomes-ray-to-deliver-a-unified-open-source-ai-compute-stack)
- [PyTorch Foundation Ray Integration - HPCwire](https://www.hpcwire.com/off-the-wire/pytorch-foundation-welcomes-ray-to-deliver-a-unified-open-source-ai-compute-stack/)

**Technical Deep Dives**:
- [Continuous Batching: 23x LLM Inference Throughput](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [Anyscale Endpoints Launch](https://www.anyscale.com/blog/anyscale-endpoints-fast-and-scalable-llm-apis)
- [Anyscale Endpoints 10x Cost Reduction](https://www.anyscale.com/press/anyscale-launches-new-service-anyscale-endpoints-10x-more-cost-effective-for-most-popular-open-source-llms)
- [End-to-End LLM Workflows Guide](https://www.anyscale.com/blog/end-to-end-llm-workflows-guide)
- [Building RAG-Based LLM Applications](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)
- [Cloud Infrastructure for LLM and GenAI](https://www.anyscale.com/blog/cloud-infrastructure-for-llm-and-generative-ai-applications)

**Customer Success Stories**:
- [Training 1 Million ML Models in Record Time](https://www.anyscale.com/blog/training-one-million-machine-learning-models-in-record-time-with-ray)
- [Ray Summit 2022 Stories - ML Platforms](https://www.anyscale.com/blog/ray-summit-2022-stories-ml-platforms)
- [RLlib: Scalable RL Training in the Cloud](https://www.anyscale.com/events/2021/12/01/rllib-scalable-rl-training-and-serving-in-the-cloud-with-ray)
- [Deploying RL in Production with Ray and SageMaker](https://aws.amazon.com/blogs/machine-learning/deploying-reinforcement-learning-in-production-using-ray-and-amazon-sagemaker/)

**Platform & Partnerships**:
- [Ray Summit 2025 Anyscale Product Updates](https://www.anyscale.com/blog/ray-summit-2025-anyscale-product-updates)
- [Anyscale Collaborates with Microsoft on Azure](https://www.prnewswire.com/news-releases/anyscale-collaborates-with-microsoft-to-deliver-ai-native-computing-on-azure-302603470.html)
- [LLMs and Agentic AI on Anyscale Docs](https://docs.anyscale.com/llm)
- [Tune Parameters for LLMs Docs](https://docs.anyscale.com/llm/serving/parameter-tuning)
- [Optimize Performance for Ray Serve LLM](https://docs.anyscale.com/llm/serving/performance-optimization)

**Company & Funding**:
- [Anyscale Company Profile - Tracxn](https://tracxn.com/d/companies/anyscale/__9qatL-iNLAEZkPcRa1pWyWpgRnkA0yFrWG6KKNOk-9o)
- [Anyscale Launches with $20.6M - TechCrunch](https://techcrunch.com/2019/12/17/anyscale-ray-project-distributed-computing-a16z/)
- [Anyscale Launch Press Release](https://www.anyscale.com/press/founders-of-open-source-project-ray-launch-anyscale-with-usd-20-6m-in-funding-to-democratize-distributed-programmingfounders-of-open-source-project-ray-launch-anyscale-with-usd-20-6m-in-funding-to-democratize-distributed-programming)
- [Anyscale Series B: $40M Led by NEA](https://www.anyscale.com/press/anyscale-announces-usd40m-in-series-b-funding-led-by-nea)
- [Anyscale Nabs $100M - Datanami](https://www.datanami.com/2021/12/08/anyscale-nabs-100m-unleashes-parallel-serverless-computing-in-the-cloud/)
- [Anyscale Raises $99M - The Software Report](https://www.thesoftwarereport.com/anyscale-raises-99m-announces-debut-of-ray-2-0/)
- [Anyscale Company Profile - YourStory](https://yourstory.com/companies/anyscale)
- [Anyscale Crunchbase](https://www.crunchbase.com/organization/anyscale)

**Job Postings & Compensation**:
- [Anyscale Software Engineer Salaries - Levels.fyi](https://www.levels.fyi/companies/anyscale/salaries/software-engineer)
- [Anyscale Salaries - Levels.fyi](https://www.levels.fyi/companies/anyscale/salaries)
- [Anyscale Salaries - Glassdoor](https://www.glassdoor.com/Salary/Anyscale-CA-Salaries-E3377996.htm)
- [Anyscale Careers - The Ladders](https://www.theladders.com/company/anyscale-jobs)

---

*Last updated: November 30, 2025*
