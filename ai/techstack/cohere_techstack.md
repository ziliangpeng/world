# Cohere - Technology Stack

**Company:** Cohere, Inc.
**Founded:** 2019
**Focus:** Enterprise AI platform with multilingual LLMs and retrieval systems
**Headquarters:** Toronto, Ontario, Canada (additional offices: San Francisco, London, New York)

---

## Non-AI Tech Stack

Cohere operates from **Toronto, Ontario** with infrastructure on **Google Cloud Platform** using **TPU v4 Pods** for training and serving. Founded in **2019** by **Aidan Gomez** (co-author of "Attention Is All You Need" transformer paper at age 20), **Nick Frosst**, and **Ivan Zhang** (all from Google Brain and University of Toronto), the company raised **$1.54 billion total funding** at a **$7 billion valuation** (September 2025) from investors including **Inovia Capital, Oracle, Nvidia, Cisco, AMD, Fujitsu, Salesforce, Tiger Global, and Radical Ventures**. The company reached **$100M+ annualized revenue** by May 2025 with **769 employees** and **$85M revenue in 2024**, with **85% of revenue from private deployments** serving finance, healthcare, and government sectors. Training infrastructure uses **JAX framework** with **pjit (partitioned just-in-time compilation)** on **TPU v4 Pods** achieving **1.7x speedup** over TPU v3. The platform supports **multi-cloud deployment** via **AWS (SageMaker, Bedrock), Microsoft Azure, Google Cloud, and Oracle Cloud Infrastructure**, with **SDKs for Python, TypeScript, Java, and Go**. The API architecture provides three primary endpoints: **Chat** (text generation), **Embed** (vector embeddings), **Rerank** (semantic ranking), available through both **v1 and v2 API versions**. Private deployment options include **Docker and Kubernetes** for **VPC** or **on-premises environments** with full network isolation. The company partners with **Dell, SAP, Oracle, AMD, and Nvidia** for enterprise and sovereign AI deployments. Cohere serves **technology, financial services, healthcare, manufacturing, energy, and public sector** clients including Oracle, Fujitsu, and Notion.

**Salary Ranges**: Software Engineer $133K-$894K (25th-90th percentile, median $235K) | Software Engineering Manager $333K | Technical Program Manager $83K-$116K | Intern $59/hr

---

## AI/ML Tech Stack

### Aya Model - 101-Language Open-Source Multilingual LLM

**What's unique**: Cohere for AI (nonprofit research subsidiary) released **Aya**, a **massively multilingual generative language model supporting 101 languages** including **more than 50 previously underserved languages** (Somali, Uzbek, and others), trained on the **Aya Collection** — the **most extensive assembly of multilingual instruction fine-tuning datasets** featuring **513 million prompts and completions across 114 languages**. Unlike proprietary multilingual models, Aya is **fully open-sourced under Apache 2.0 license** on Hugging Face (CohereForAI/aya-101), enabling commercial use without restrictions. The model uses a **sequence-to-sequence architecture based on mT5**, achieving **over 75% human evaluation scores** and **80-90% simulated win rates** compared to competitors like **mT0 and Bloomz**, which it surpasses **"by a wide margin"** on benchmark tests. Aya was developed through an **open science initiative involving over 3,000 independent researchers across 119 countries**, demonstrating collaborative AI research at unprecedented scale. The training datasets include **xP3x, Aya Dataset, Aya Collection, DataProvenance collection, and ShareGPT-Command**, providing diverse multilingual instruction-following examples. This approach democratizes access to advanced multilingual AI for low-resource languages traditionally neglected by major model providers, enabling applications in regions where English-centric models fail.

### Command A Series - Sovereign AI and Private Enterprise Deployments

**What makes it different**: Cohere developed the **Command A series** (111B parameters, 256k context length) optimized for **sovereign AI deployments** where governments and enterprises require **data sovereignty, security, and compliance** with national regulations. The models include **Command A** (general-purpose reasoning and RAG), **Command A Reasoning** (complex problem-solving and agentic workflows), **Command A Translate** (23 languages, 16k context), and **Command A Vision** (multimodal text+image). **85% of Cohere's revenue comes from private deployments** rather than public API usage, contrasting with OpenAI/Anthropic's cloud-first approach. Private deployment options span **three architectures**: **(1) Virtual Private Cloud (VPC)** on any cloud provider (AWS, Azure, GCP, OCI) with full control over data storage and processing; **(2) On-premises** deployment with customer-owned GPUs and servers for complete network isolation; **(3) Hybrid** configurations combining cloud infrastructure with private network security. Command A Reasoning runs efficiently on **one or two A100/H100 GPUs**, making sovereign-cloud scenarios feasible without massive compute infrastructure. The **AMD partnership** (September 2025) provides **AMD Instinct™ GPU-powered infrastructure** as an alternative to Nvidia, offering customers **more choices to meet performance and TCO goals** while supporting sovereign AI initiatives with domestic hardware. This deployment flexibility enables sectors with stringent data privacy requirements (finance, healthcare, defense, government) to leverage advanced LLMs without compromising security.

### Embed v4 - Multimodal Matryoshka Embeddings for Flexible Retrieval

**What sets Cohere apart**: Cohere's **Embed v4** implements **Matryoshka Representation Learning**, producing embeddings that can be **truncated to smaller dimensions (256, 512, 1024, 1536) with minimal fidelity loss**, enabling **flexible accuracy-cost tradeoffs** for different applications. Unlike fixed-size embeddings (OpenAI's 1536-dimensional embeddings cannot be truncated), Matryoshka embeddings allow developers to **choose dimension size based on use case** — high-accuracy retrieval uses 1536 dimensions while real-time applications use 256 dimensions for faster processing. Embed v4 is **multimodal**, embedding both **text and images** for unified semantic search across content types within the same vector space, enabling queries like **"find images similar to this text description"** or **"find documents related to this photo"**. The model supports **over 100 languages** with **128k context length**, processing documents up to **novel-length** in a single embedding. The architecture includes **Embed v3** (1024 dimensions, trained on nearly 1 billion English training pairs) and **Embed Multilingual 3** for cross-lingual applications. Embed models generate vector representations for **semantic search, classification, clustering, and RAG applications**, with deployment options in **VPC or on-premises environments** for data privacy. The **multimodal capability** enables enterprise knowledge bases combining PDFs, images, videos, and text to be searched through a single unified interface.

### Rerank API - 100+ Language Semantic Ranking at Scale

**What's unique**: Cohere's **Rerank API** indexes documents from **most to least semantically relevant to a query**, serving as a **second-stage reranking system** that dramatically improves retrieval accuracy beyond first-stage keyword or semantic search. The system **combines query and document tokens** to compute relevance scores with **context length of 4096 tokens** (rerank-v3.5 and rerank-v3.0 models), processing both inputs jointly rather than encoding them separately. For multi-chunk documents, Rerank **combines query with each chunk** and assigns the **highest chunk score as the document's final relevance score**, handling long-form content exceeding model context limits. The API supports **100+ languages** and **multi-aspect, semi-structured data** including emails, invoices, JSON documents, code, and tables, using specialized training to understand structured formats. Unlike naive embedding similarity (which misses nuanced relevance), Rerank uses **cross-attention** between query and document tokens, capturing complex semantic relationships. The architecture enables companies to **retain existing keyword-based or semantic search systems** for first-stage retrieval (candidate generation from millions of documents) and integrate Rerank for **second-stage reranking** (refining top 100-1000 candidates) with **high accuracy and minimal latency**. Cohere recommends **against sending more than 1,000 documents** in a single request for optimal performance. Rerank can be deployed in **VPC or on-premises environments**, critical for enterprise search over proprietary data.

### North Platform - Agentic AI for Enterprise Workplace Productivity

**What makes it different**: Cohere launched **North**, an **enterprise-ready AI platform designed for modern workplace productivity**, providing **agentic AI capabilities** that autonomously complete multi-step workflows rather than single-query interactions. North combines **Command A models** (reasoning and tool use), **Compass** (intelligent business data discovery), **Embed** (semantic search), and **Rerank** (relevance optimization) into a unified platform powering **internal search and complex workflows** designed for private deployment. The system enables AI agents to **autonomously plan, execute, and verify multi-step tasks** — for example, analyzing financial reports, extracting key metrics, generating summaries, and updating databases — rather than requiring manual orchestration. The **Dell partnership** (May 2025) combines **North's agentic AI technology with Dell's security-hardened, scalable infrastructure**, setting **"a new standard"** for enterprise AI deployment. North emphasizes **private deployment behind corporate firewalls** rather than cloud APIs, addressing enterprise concerns about **data leakage, model customization, and regulatory compliance**. The platform supports **fine-tuning on proprietary data** to create organization-specific AI systems that understand internal terminology, workflows, and business logic. Unlike general-purpose assistants (ChatGPT, Claude), North targets **knowledge work automation** across **technology, financial services, healthcare, manufacturing, energy, and public sector**, where tasks involve multi-system integration and domain expertise.

### TPU v4 Training Infrastructure - 1.7x Speedup with JAX Framework

**What's unique**: Cohere trains language models from scratch on **Google Cloud TPU v4 Pods** using the **JAX framework** with **pjit (partitioned just-in-time compilation)**, achieving **1.7x total speedup** over TPU v3 through both hardware improvements and software optimization. The **FAX framework** relies heavily on **pjit feature of JAX**, which **abstracts the relationship between device and workload**, allowing Cohere engineers to optimize efficiency and performance without manually managing parallelization strategies. Cohere's implementation takes advantage of TPU v4 Pods to perform **tensor parallelism**, which is **more efficient than the earlier pipeline parallelism** implementation on TPU v3. The superior performance of v4 chips enables Cohere to **iterate on ideas and validate them 1.7x faster in computation**, accelerating research cycles and enabling rapid model experimentation. The multi-year partnership with Google Cloud ensures training is powered by **90% carbon-free energy**, meeting sustainability standards while scaling to billions of parameters. The technical approach is detailed in the paper **"Scalable Training of Language Models using JAX pjit and TPUv4"**. This infrastructure supports training **Command series models (111B parameters)** and **Aya multilingual models** across TPU pods with thousands of chips, demonstrating JAX's capability for **large-scale distributed training**. Unlike companies locked to Nvidia's CUDA ecosystem, Cohere's JAX-based stack provides **hardware flexibility** and **vendor independence**, enabling future adoption of new accelerators (TPU v5, AMD Instinct) without framework rewrites.

### Private Deployment Architecture - 85% Revenue from Secure Enterprise AI

**What sets Cohere apart**: Cohere generates **85% of revenue from private deployments** rather than public API usage, fundamentally differentiating its business model from OpenAI/Anthropic's cloud-first approach. The architecture supports **three deployment pathways**: **(1) SaaS API** via Cohere's cloud for rapid prototyping; **(2) Virtual Private Cloud (VPC)** on any cloud provider (AWS, Azure, GCP, OCI) where **data never leaves the customer environment** and models can be **fully network-isolated**; **(3) On-premises** deployment with customer-owned hardware providing **full control over both data and AI systems** to insulate environments from external threats. This flexibility addresses **stringent data privacy rules** in finance, healthcare, and government sectors where cloud APIs are prohibited by regulation. The platform enables **fine-tuning on proprietary data** to create domain-specific models — for example, medical LLMs trained on internal clinical notes or legal assistants trained on case law databases — without sending sensitive data to third parties. The **multi-cloud support** prevents vendor lock-in, allowing customers to **migrate between cloud providers** or maintain **multi-region deployments** for disaster recovery. Private deployments include **Docker and Kubernetes configurations** with **access controls, encryption, and audit logging** meeting industry certifications (SOC 2, HIPAA, FedRAMP). The strategy directly targets the **"private AI"** market segment where enterprises demand **security, customization, and data sovereignty** over convenience of cloud APIs, capturing revenue streams inaccessible to cloud-only providers.

---

## Sources

**Cohere Official**:
- [Cohere Homepage](https://cohere.com/)
- [Cohere Documentation](https://docs.cohere.com)
- [Cohere Careers](https://cohere.com/careers)
- [Deployment Options](https://cohere.com/deployment-options)
- [Private Deployments](https://cohere.com/private-deployments)
- [Security](https://cohere.com/security)

**Products & Platforms**:
- [Cohere Rerank](https://cohere.com/rerank)
- [Aya Model Overview](https://cohere.com/research/aya)
- [Aya Project Page](https://sites.google.com/cohere.com/aya-en/home)
- [Models Overview - Cohere Docs](https://docs.cohere.com/v2/docs/models)
- [Command R Documentation](https://docs.cohere.com/docs/command-r)
- [Command R+ Documentation](https://docs.cohere.com/docs/command-r-plus)
- [Embed Models Documentation](https://docs.cohere.com/v2/docs/cohere-embed)
- [Rerank Overview](https://docs.cohere.com/docs/rerank-overview)

**Technical Infrastructure**:
- [Accelerating Language Model Training with Cohere and Google Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/accelerating-language-model-training-with-cohere-and-google-cloud-tpus)
- [Building Production AI on Cloud TPUs with JAX](https://developers.googleblog.com/building-production-ai-on-google-cloud-tpus-with-jax/)
- [Cohere Multilingual Model Launch](https://cohere.com/blog/multilingual)
- [Cross-Lingual Classification in NLP](https://cohere.com/blog/cross-lingual-classification)

**Open Source & Research**:
- [Aya Model - Hugging Face](https://huggingface.co/CohereForAI/aya-101)
- [Aya arXiv Paper](https://arxiv.org/abs/2402.07827)
- [Command R+ - Hugging Face](https://huggingface.co/CohereLabs/c4ai-command-r-plus)
- [Cohere for AI Open Source LLM Launch](https://venturebeat.com/ai/cohere-for-ai-launches-open-source-llm-for-101-languages)

**Partnerships & Enterprise**:
- [AMD and Cohere Partnership](https://ir.amd.com/news-events/press-releases/detail/1259/amd-and-cohere-expand-global-ai-collaboration-to-power-enterprise-and-sovereign-deployments-with-amd-ai-infrastructure)
- [Dell and Cohere Partnership](https://www.dell.com/en-us/blog/smart-simple-secure-enterprise-ai-with-dell-cohere/)
- [MongoDB Atlas and Cohere Command R+](https://www.mongodb.com/company/blog/technical/build-scalable-rag-mongodb-atlas-cohere-command-r-plus)

**Company & Funding**:
- [Cohere - Wikipedia](https://en.wikipedia.org/wiki/Cohere)
- [Aidan Gomez - Wikipedia](https://en.wikipedia.org/wiki/Aidan_Gomez)
- [Cohere Raises $550M at $5.5B Valuation](https://www.aibase.com/news/10490)
- [Cohere CEO on CNBC](https://www.cnbc.com/2024/07/06/cohere-ceo-and-ex-google-researcher-aidan-gomez-on-how-ai-makes-money.html)
- [Cohere Company Profile - Tracxn](https://tracxn.com/d/companies/cohere/__o4xfwmr3XwgsGEyH41XvwBm6Xd-SjsMlSld3d4ci6G0)
- [Cohere Revenue Analysis - Latka](https://getlatka.com/companies/cohere.com)

**Job Postings & Compensation**:
- [Cohere Salaries - Levels.fyi](https://www.levels.fyi/companies/cohere/salaries)
- [Cohere Compensation - 6figr](https://6figr.com/us/salary/cohere)
- [Cohere Salaries - Glassdoor](https://www.glassdoor.com/Salary/Cohere-Salaries-E6413613.htm)

---

*Last updated: November 30, 2025*
