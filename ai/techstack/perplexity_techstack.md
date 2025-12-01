# Perplexity AI - Technology Stack

**Company:** Perplexity AI
**Founded:** 2022
**Focus:** AI-powered conversational search engine
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

Perplexity runs entirely on **AWS**, using "pretty much every technology available" including **Amazon EC2** (compute), **Amazon S3** (storage), DNS, web servers, and GPU clusters. The search infrastructure is built on **Vespa.ai**, a distributed search engine handling **200+ billion unique URLs** with **400+ petabytes in hot storage** and processing **tens of thousands of index update requests per second**. Vespa's unique index technology enables real-time mutations of index structures, critical for keeping search results current. The platform serves **200 million daily queries** (400M+ monthly) with a small engineering team of **~38 people** focused on differentiating technology rather than commodity infrastructure. Orchestration uses **Kubernetes** to manage the fleet of **NVIDIA H100 GPU** pods. Storage infrastructure includes multi-tier systems for crawling, indexing, and serving. Backend, frontend, and search components run on Amazon EC2, with monitoring likely through AWS CloudWatch.

**Salary Ranges**: Software Engineer $121K-$450K (median $450K) | Backend SWE $180K-$230K | Staff SWE ~$160K-$200K (Glassdoor)

---

## AI/ML Tech Stack

### ROSE Inference Engine - Custom-Built Architecture

**What's unique**: Perplexity built **ROSE**, a proprietary inference engine designed for flexibility (rapidly testing new models) and extreme optimization. ROSE is primarily written in **Python** leveraging **PyTorch** for model definitions, but performance-critical components (serving logic, batch scheduling algorithms) are being **migrated to Rust** for C++-comparable performance with memory safety guarantees. The tech stack includes **Python, Rust, C++, PyTorch, Triton, CUDA, Kubernetes**. ROSE's core LLM engine loads model weights, generates decoded tokens, and supports advanced decoding strategies including **speculative decoding** and **MTP (Multi-Token Prediction)** decoders for improved latency. The engine bundles both **Triton and CUDA kernels** that efficiently implement and fuse layer computations. This custom architecture enables hyper-optimization unavailable in off-the-shelf inference servers.

### Sonar Models - Proprietary Fine-Tuned LLMs

Perplexity's proprietary **Sonar models** are based on **Llama 3.3 70B** with extensive fine-tuning to boost fact accuracy and response clarity. The models are purpose-built for **Retrieval-Augmented Generation (RAG)** and source transparency. Engineering focus areas include: (1) RAG orchestration logic, (2) fine-tuning proprietary Sonar models, and (3) hyper-optimization of the ROSE inference engine. The hybrid model architecture combines proprietary and open-source LLMs (Llama 3.1 70B base with custom Perplexity fine-tuning), integrated with real-time web search and multi-document synthesis. This differentiates Perplexity from pure API wrappers, as Sonar models are optimized specifically for conversational search with citation.

### RAG-First Pipeline Architecture

**What's unique**: A meticulously implemented **RAG pipeline executes for nearly every query**, ensuring answers are relevant and factually grounded in current information. Perplexity built its retrieval layer on **Vespa.ai**, described as "the only production-proven platform capable of powering real-time, large-scale RAG." The system uses **hybrid retrieval** (combining multiple retrieval strategies) and **intelligent context curation** to feed AI models optimal information. By outsourcing distributed real-time search to Vespa, Perplexity's small team focuses on unique differentiators rather than rebuilding search infrastructure. The RAG architecture processes 200M daily queries with real-time index updates, a scale challenge most RAG systems don't face.

### GPU Infrastructure & Optimization

Perplexity parallelizes models like **Llama 8B, 70B, and 405B** across multiple GPUs using **tensor parallelism**. The team collaborates with **NVIDIA Triton** engineering to deploy **disaggregated serving** that separates **prefill and decode inference phases onto separate GPUs**, optimizing resource utilization and latency. Training infrastructure uses **Amazon SageMaker HyperPod** with **Amazon EC2 P4de instances** (highest performance for ML training), achieving **40% reduction in model training time**. The platform runs on pods of **NVIDIA H100 GPUs** within AWS. Perplexity serves **400+ million search queries monthly** using the NVIDIA inference stack. Open-source contributions include **high-performance Mixture-of-Experts (MoE) communication kernels** in the **pplx-garden** GitHub repository, challenging NVIDIA's dominance with trillion-parameter model support on AWS.

### Small Team, Big Impact

With only **~38 engineers**, Perplexity focuses exclusively on three differentiating technologies: RAG orchestration, Sonar fine-tuning, and ROSE optimization. This lean approach is enabled by strategic infrastructure choices (Vespa for search, AWS for cloud, NVIDIA for GPUs) that offload commodity components, allowing the team to punch above its weight in the competitive AI search market.

**Salary Ranges**: ML Engineer $200K-$250K | AI Inference Engineer (specialized ROSE/GPU optimization) | Data Scientist up to $791K total comp | Software Engineer median $450K total comp

---

## Sources

**Engineering & Architecture**:
- [How Perplexity Built an AI Google - ByteByteGo](https://blog.bytebytego.com/p/how-perplexity-built-an-ai-google)
- [Perplexity builds AI Search at scale on Vespa.ai](https://blog.vespa.ai/perplexity-builds-ai-search-at-scale-on-vespa-ai/)
- [How Perplexity uses Vespa.ai to power answers](https://vespa.ai/perplexity/)
- [Architecting an AI-First Search API - Perplexity Research](https://research.perplexity.ai/articles/architecting-and-evaluating-an-ai-first-search-api)

**AWS Case Studies**:
- [Perplexity Accelerates Training 40% with SageMaker HyperPod](https://aws.amazon.com/solutions/case-studies/perplexity-case-study/)
- [Perplexity Uses Claude 3 in Amazon Bedrock](https://aws.amazon.com/solutions/case-studies/perplexity-bedrock-case-study/)
- [Perplexity at AWS re:Invent 2023](https://aws.amazon.com/solutions/case-studies/perplexity-keynote-aws-reinvent-2023/)

**NVIDIA & Inference**:
- [Perplexity Serves 400M Queries Using NVIDIA Inference Stack](https://developer.nvidia.com/blog/spotlight-perplexity-ai-serves-400-million-search-queries-a-month-using-nvidia-inference-stack/)
- [Perplexity Releases Open-Source AI for Trillion-Parameter Models](https://winbuzzer.com/2025/11/04/perplexity-releases-open-source-ai-system-to-run-trillion-parameter-models-on-aws-challenging-nvidias-dominance-xcxwbn/)

**AI/ML Job Postings**:
- [AI Inference Engineer](https://job-boards.greenhouse.io/perplexityai/jobs/4403747007) - ROSE optimization specialist
- [Machine Learning Engineer](https://www.perplexity.ai/hub/careers) - $200K-$250K
- [Backend Software Engineer](https://www.perplexity.ai/hub/careers) - $180K-$230K
- [AI Systems Engineer](https://careers.mavenventures.com/companies/perplexity-2/jobs/50956991-ai-systems-engineer)
- [Perplexity Careers - All Positions](https://www.perplexity.ai/hub/careers)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/perplexity-ai/salaries)

---

*Last updated: November 30, 2025*
