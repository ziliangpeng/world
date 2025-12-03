# Arcee AI - Technology Stack

**Company:** Arcee AI, Inc.
**Founded:** 2023
**Focus:** Small language models (SLMs) and domain adaptation platform
**Headquarters:** Remote (US-based)

---

## Non-AI Tech Stack

Arcee AI operates as a **remote-first company** with **1-10 employees** across the United States. Founded in **2023** by **Mark McQuade** (CEO, early commercial hire at Hugging Face), **Jacob Solawetz** (CTO, YC company Roboflow), and **Brian Benedict** (CRO, early commercial hire at Hugging Face), the company raised **$29.5 million total funding** over **2 rounds**: **$5.5M seed** (January 2024) and **$24M Series A** (July 2024, Emergence Capital lead, Long Journey Ventures, Flybridge). The company reported **$2 million in revenue** with potential to **turn profitable by early 2025**. Infrastructure partners include **AWS** (Trainium for training, marketplace for deployment), **Microsoft Azure**, **NVIDIA** (GPU infrastructure), **Intel**, and **ARM** (edge deployment). The platform deploys within **customer virtual private clouds (VPC)** for data sovereignty and compliance in regulated industries (legal, healthcare, insurance, financial services). Models are available through **Hugging Face** (featured model partner), **OpenRouter** (API access), and **AWS Marketplace**. The company releases **open-weight models** under **Apache 2.0 license**, emphasizing transparency and community accessibility. Backend infrastructure supports **continual learning** with **"deploy once, improve continuously"** online reinforcement learning. Development uses **Python-based tooling** with **PyTorch** and **Hugging Face Transformers** for model training and inference. The platform provides **SDKs and APIs** for model customization, fine-tuning, and deployment. Arcee serves **highly regulated industries** (legal, healthcare, insurance, financial services) requiring **secure, compliant AI** within organizational boundaries.

**Salary Ranges**: Compensation not publicly disclosed; competitive salaries, equity, and benefits based on location, role, level, and experience

---

## AI/ML Tech Stack

### Small Language Models (SLMs) - 72B Parameters or Less for Domain-Specific Excellence

**What's unique**: Arcee AI defines **small language models (SLMs)** as anything with **72B parameters or less**, challenging the industry trend toward ever-larger models by demonstrating that **SLMs can outperform large language models (LLMs) when trained on domain-specific tasks**. The company has seen **great success with models as small as 7 billion parameters** through targeted domain adaptation, continual pre-training, and model merging techniques. This approach addresses the **cost, latency, and deployment challenges** of 100B+ parameter models — SLMs run on **smaller hardware configurations** (single GPU vs. multi-GPU clusters), respond with **lower latency** (milliseconds vs. seconds), and cost **significantly less to operate** (75% reduction in total deployment costs vs. closed-source LLMs). Arcee's SLMs achieve **frontier-class performance** on domain-specific tasks (medical diagnosis, legal document drafting, financial analysis) without the **computational overhead** of general-purpose mega-models. The platform enables enterprises to **train custom SLMs** on proprietary data within their VPC, creating **specialized models** that understand industry terminology, regulatory requirements, and organizational workflows better than generic LLMs. This **"small to win big"** strategy positions SLMs as the **practical choice for enterprise AI** where **cost efficiency, data privacy, and deployment flexibility** matter more than generic benchmark performance.

### SuperNova-Medius - 14B Model Rivaling 70B Through Cross-Architecture Distillation

**What makes it different**: **SuperNova-Medius** is a **14B parameter model** built on the **Qwen2.5-14B-Instruct architecture** that achieves performance **rivaling 70B models** through **cross-architecture distillation** combining knowledge from **Qwen2.5-72B-Instruct** and **Llama-3.1-405B-Instruct**. The development process uses **multi-architecture offline logit distillation** with **cross-architecture vocabulary alignment**, implementing a **multi-teacher approach** where the 14B student model learns from two larger teacher models simultaneously. The technical pipeline includes: **(1) Offline logit storage** — distilling **Llama 3.1 405B logits** and storing the **top-K logits per token** to capture probability mass while managing storage; **(2) Vocabulary alignment** — using **mergekit-tokensurgeon** to create a **Qwen2.5-14B version using Llama 3.1 405B vocabulary**, enabling Llama logits to train the Qwen-based model; **(3) Dual distillation** — training the adapted Qwen model with 405B logits while separately distilling **Qwen2-72B into 14B**. SuperNova-Medius **outperforms Qwen2.5-14B and SuperNova-Lite** across multiple benchmarks, excelling in **instruction-following (IFEval)** and **complex reasoning tasks (BBH)**. The model demonstrates that **knowledge distillation from multiple teacher architectures** produces **more capable students** than single-teacher approaches, enabling **compact deployment** (customer support, content creation, technical assistance) on **smaller hardware** while maintaining **high-quality generative AI** performance. Available under **Apache 2.0 license** on Hugging Face and AWS Marketplace.

### MergeKit - Open-Source Model Merging Toolkit Powering Thousands of Models

**What sets Arcee apart**: **MergeKit** is Arcee's **open-source library** for **model merging** with an **efficient and extensible framework** suitable for **any hardware**, having **facilitated the merging of thousands of models**. Model merging involves the **fusion of two or more LLMs into a singular, cohesive model**, creating **sophisticated models at a fraction of the cost** without heavy training and GPU resources. The toolkit implements multiple merging methods including **TIES** (addressing parameter interference), **DARE** (dropping redundant parameters), **SLERP** (spherical interpolation), and **task arithmetic** (adding/subtracting task vectors). MergeKit was recently **relicensed to LGPL v3**, expanding enterprise adoption — companies like **IBM use MergeKit** for model development and checkpoint evaluation. The toolkit enables **domain-adapted model merging** with general open-source models, affording AI engineers a **cost-effective method to extend general intelligence** to organizational needs. A key innovation: merging **continual pre-training (CPT) models** with **original instruct models** using TIES **significantly recovers lost general capabilities**, mitigating **catastrophic forgetting** observed during CPT. This approach preserves the **strengths of each model** — domain expertise from CPT and general reasoning from instruct models — producing **balanced, capable systems**. MergeKit democratizes **advanced model composition** techniques previously requiring deep ML expertise, enabling practitioners to **experiment with model combinations** through simple configuration files.

### Continual Pre-Training (CPT) with Model Merging - Domain Adaptation Without Catastrophic Forgetting

**What's unique**: Arcee pioneered **continual pre-training (CPT) combined with model merging** for **domain adaptation** that achieves **strong domain performance without catastrophic forgetting**. CPT extends training of **base models** (Llama-2-base, Mistral-7B-base) using **domain-specific datasets** (medical literature, legal case law, financial reports), fine-tuning models to **the nuances of specialized fields**. However, traditional CPT suffers from **catastrophic forgetting** — post-pretraining results in **deterioration of original general abilities**, hindering fine-tuned performance across various tasks. Arcee's breakthrough: **merging CPT models with original instruct models** using **TIES method significantly recovers lost general capabilities**, enabling models that are **both domain-expert and generally intelligent**. The technique is **incredibly compute-efficient** — instead of CPT over an entire model, you **train only a much smaller model** (adapters, LoRA), then **merge with a much larger model**, making it an **elegant approach** requiring minimal GPU resources. Arcee's research assessed effectiveness in **medical, legal, patent, and financial domains**, demonstrating that **merged models outperform both pure CPT and pure instruct models** on domain-specific benchmarks while maintaining general reasoning. This methodology enables **cost-effective domain adaptation** — enterprises train small domain modules on proprietary data, then merge with powerful open-source base models, achieving **specialized performance without training massive models from scratch**.

### EvolKit - Automatic Instruction Complexity Enhancement for Fine-Tuning

**What makes it different**: **EvolKit** is Arcee's **open-source framework** designed to **automatically enhance the complexity of instructions** used for fine-tuning large language models, generating **higher-quality training datasets** without manual curation. The system takes **simple instructions** and evolves them into **more complex, nuanced examples** that challenge models during fine-tuning, improving reasoning capabilities and instruction-following. EvolKit implements **evolutionary algorithms** that iteratively increase instruction difficulty through operations like **constraint addition** (adding requirements to prompts), **reasoning deepening** (requiring multi-step logic), **concretization** (adding specific details), and **task composition** (combining multiple subtasks). The framework automatically generates **diverse training datasets** by transforming seed instructions into variations spanning difficulty levels, enabling **curriculum learning** where models progressively tackle harder examples. This approach addresses the **data quality bottleneck** in instruction-tuning — manually crafting thousands of high-quality instruction-response pairs is expensive and time-consuming, while EvolKit **automates complexity enhancement** at scale. Arcee uses EvolKit in their **SuperNova model training pipeline**, combining evolved instructions with continual pre-training and model merging to produce models that excel at **complex reasoning and instruction adherence**. The toolkit demonstrates that **automatic data augmentation** through instruction evolution produces **better fine-tuned models** than training on static, simple datasets.

### Arcee Orchestra - No-Code Agentic AI Platform with Intelligent Model Routing

**What's unique**: **Arcee Orchestra** (launched 2025) is a **no-code platform** for building **custom AI workflows powered by state-of-the-art SLMs**, providing an **end-to-end agentic AI platform** that combines SLMs with **intelligent model routing and orchestration** — enabling users to **use the right model for the right tasks**. The platform includes **Arcee Conductor**, an **intelligent model routing system** that evaluates prompts and **sends them to the optimal SLM or LLM based on domain or task complexity**. Simple queries route to small, fast models while complex reasoning tasks route to more capable models, optimizing the **cost-performance tradeoff** across workflows. Orchestra enables **agentic AI workflows** where AI systems autonomously complete multi-step tasks — analyzing documents, extracting insights, generating reports, and taking actions — rather than single-query interactions. The **no-code interface** empowers business users to build AI agents without engineering expertise, selecting from **pre-trained domain-specific SLMs** (medical, legal, financial) or customizing models on proprietary data. The platform implements **continuous learning** where **deployed models improve automatically** through online reinforcement learning from user interactions, embodying Arcee's **"deploy once, improve continuously"** philosophy. Orchestra represents Arcee's evolution from **SLM model provider** to **agentic AI platform**, addressing enterprise demand for **integrated workflows** rather than standalone models.

### Customer Success at Scale - 96% Cost Reduction, 83% Performance Boost

**What sets Arcee apart**: Arcee AI's customer deployments demonstrate **dramatic cost and performance improvements** across regulated industries: A **financial services customer** saw a **23% boost in benchmarks and a 96% reduction in costs**, while an **insurance customer boosted performance by 83% and cut costs by 89%**. Arcee AI SLMs help customers **save up to 75% on total deployment costs** compared to traditional closed-source LLMs through smaller model sizes, efficient architectures, and VPC deployment avoiding API costs. The platform serves **legal, healthcare, insurance, and financial services** with industry-specific applications: **(1) Healthcare** — automating transcription of patient encounters, filling electronic health records (EHRs), processing insurance claims, reducing administrative burden; **(2) Legal** — automating drafting of contracts, wills, and pleadings, generating first-draft documents adhering to legal standards; **(3) Finance** — automating extraction, analysis, and summarization from complex documents (contracts, reports, regulatory filings), reducing manual labor and errors. Arcee's **SEC LLM** (specialized financial model) demonstrates domain expertise by understanding regulatory language, financial terminology, and document structures specific to securities filings. The **strong evaluations in medical, legal, and financial verticals** validate Arcee's thesis that **domain-adapted SLMs outperform general-purpose LLMs** on specialized tasks while costing dramatically less to operate.

### Open-Source Strategy - Trinity Models and Community-Driven Innovation

**What makes it different**: Arcee positions itself as **"A US-based Open Intelligence Lab"** with an **open-weight release strategy** providing **fully transparent architectures** under **Apache 2.0 license**. The **Trinity model series** delivered **three releases within six months**, with **Trinity Mini (26B)** as the latest offering available **free on OpenRouter** for limited periods. Trinity Mini operates as a **Mixture of Experts (MoE) model**, designed for **compact deployment while maintaining reasoning capabilities**. The open-source strategy includes **MergeKit** (recently relicensed to **LGPL v3** for broader enterprise adoption) and **EvolKit** (instruction evolution framework) on GitHub, building a **community ecosystem** around Arcee's techniques. This approach contrasts with competitors hoarding proprietary methods; Arcee believes **open models and tools accelerate industry progress** while positioning the company as a **trusted partner** for enterprises requiring transparent, auditable AI. The **featured partnership with Hugging Face** provides distribution reaching millions of developers, while **OpenRouter integration** enables API access for developers preferring managed inference. Arcee's strategy: **democratize SLM techniques** through open-source tools while monetizing **enterprise platform, training services, and custom model development** — a model proven by Hugging Face, where open-source builds community trust that converts to enterprise sales.

---

## Sources

**Arcee AI Official**:
- [Arcee AI Homepage](https://www.arcee.ai/)
- [Arcee AI Careers](https://www.arcee.ai/careers)
- [About Small Language Models (SLMs)](https://www.arcee.ai/about-slms)
- [Small Language Models Overview](https://www.arcee.ai/small-language-models)
- [Open Source Models and Toolkits](https://www.arcee.ai/open-source)
- [MergeKit Product Page](https://www.arcee.ai/product/mergekit)

**Company & Funding**:
- [Arcee AI Company Profile - Tracxn](https://tracxn.com/d/companies/arcee.ai/__v7VPBw8cFrHefsYBwK9bugpAeMXm3bbe1gw_ItIV92Q)
- [Arcee AI Crunchbase](https://www.crunchbase.com/organization/arcee-ai)
- [Arcee AI - CB Insights](https://www.cbinsights.com/company/arceeai)
- [Arcee AI - PitchBook](https://pitchbook.com/profiles/company/537629-05)
- [Small Language Models Rising as Arcee AI Lands $24M Series A - VentureBeat](https://venturebeat.com/ai/small-language-models-rising-as-arcee-ai-lands-24m-series-a)
- [Arcee AI Secures $24M Series A - Refresh Miami](https://refreshmiami.com/news/arcee-ai-secures-24m-series-a-to-transform-the-landscape-of-small-language-models/)
- [Arcee Is Going Small To Win Big - Emergence Capital](https://www.emcap.com/thoughts/arcee-is-going-small-to-win-big-in-the-long-run)

**Technical Blog Posts**:
- [Arcee AI: From SLM Pioneer to Agentic AI Workflows](https://www.arcee.ai/blog/arcee-ai-from-small-language-model-pioneer-to-pioneering-slm-powered-agentic-ai-workflows)
- [Why Agentic AI Tools Need SLMs](https://www.arcee.ai/blog/why-agentic-ai-tools-and-ai-agent-platforms-need-small-language-models-slms)
- [What is an SLM (Small Language Model)?](https://www.arcee.ai/blog/what-is-an-slm)
- [Top 5 Industries Ripe for SLM Adoption](https://www.arcee.ai/blog/top-five-industries-ripe-for-slm-adoption)
- [How Model Merging Fits Into Arcee's SLM System](https://www.arcee.ai/blog/how-model-merging-fits-into-slm-system)

**SuperNova-Medius**:
- [Introducing SuperNova-Medius - Arcee AI Blog](https://blog.arcee.ai/introducing-arcee-supernova-medius-a-14b-model-that-rivals-a-70b-2/)
- [SuperNova-Medius on Hugging Face](https://huggingface.co/arcee-ai/SuperNova-Medius)
- [SuperNova-Medius on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-7vvrzhgvorleu)
- [Arcee AI Unveils SuperNova - VentureBeat](https://venturebeat.com/ai/arcee-ai-unveils-supernova-a-customizable-instruction-adherent-model-for-enterprises)

**Continual Pre-Training & Model Merging**:
- [Domain Adaptation through CPT and Model Merging - Case Study](https://www.arcee.ai/blog/case-study-innovating-domain-adaptation-through-continual-pre-training-and-model-merging)
- [Domain Adaptation of Llama3-70B - arXiv Paper](https://arxiv.org/abs/2406.14971)
- [Continual Pre-Training Glossary](https://www.arcee.ai/glossaries/continuous-pre-training)
- [Revolutionizing LLM Training with Arcee and AWS Trainium](https://aws.amazon.com/blogs/machine-learning/revolutionizing-large-language-model-training-with-arcee-and-aws-trainium/)

**MergeKit & EvolKit**:
- [Arcee and MergeKit Unite - Blog](https://blog.arcee.ai/arcee-and-mergekit-unite/)
- [Arcee/MergeKit Launch Model Merging Hackathon](https://www.arcee.ai/blog/arcee-mergekit-launch-model-merging-hackathon)
- [Arcee's MergeKit: A Toolkit for Merging LLMs - arXiv](https://arxiv.org/html/2403.13257v3)
- [EvolKit on GitHub](https://github.com/arcee-ai/EvolKit)
- [Arcee AI GitHub Organization](https://github.com/arcee-ai)

**Customer Success & Use Cases**:
- [Arcee AI Case Study - AWS](https://aws.amazon.com/solutions/case-studies/arcee-ai-case-study/)
- [Introducing the Ultimate SEC LLM](https://www.arcee.ai/blog/introducing-the-ultimate-sec-data-chat-agent-revolutionizing-financial-insights)
- [How Do I Prep My Data to Train an LLM?](https://www.arcee.ai/blog/how-do-i-prep-my-data-to-train-an-llm-2)
- [Arcee is a Secure, Enterprise-Focused Platform - TechCrunch](https://techcrunch.com/2024/01/24/arcee-is-a-secure-enterprise-focused-platform-for-building-genai/)
- [AiHot100 #31 Arcee AI - Dynamic Business](https://dynamicbusiness.com/ai-tools/aihot100-31-arcee-ai-enterprise-ai-simplified.html)

**Job Postings & Compensation**:
- [Arcee AI Software Engineer Salaries - Levels.fyi](https://www.levels.fyi/companies/arcee-ai/salaries/software-engineer)
- [Arcee Jobs - Wellfound](https://wellfound.com/company/arcee/jobs)

---

*Last updated: November 30, 2025*
