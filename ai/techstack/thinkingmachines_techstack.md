# Thinking Machines Lab - Technology Stack

**Company:** Thinking Machines Lab Inc.
**Founded:** February 2025
**Focus:** Customizable multimodal AI systems for human-AI collaboration
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

Thinking Machines Lab operates from **San Francisco** as a **public benefit corporation** founded in **February 2025** by **Mira Murati** (former OpenAI Chief Technology Officer, 2022-2024). The company closed a **$2 billion seed round** in **July 2025** at a **$12 billion valuation** (one of the largest seed rounds in Silicon Valley history), led by **Andreessen Horowitz** with participation from **Nvidia, AMD, Cisco, Accel, ServiceNow, and Jane Street**. The founding team includes **Barret Zoph** (CTO, former OpenAI VP of Research/Post-Training), **John Schulman** (Chief Scientist, OpenAI co-founder), and **Lilian Weng** (former OpenAI VP). The company hired **~30 researchers and engineers** from **OpenAI, Meta AI, and Mistral**, with **two-thirds comprising former OpenAI employees** (~20 total OpenAI alumni). Advisors include **Bob McGrew** (former OpenAI Chief Research Officer) and **Alec Radford** (former OpenAI lead researcher). The team previously created **widely-adopted AI products** including **ChatGPT and Character.ai**, plus **open-source projects** like **PyTorch, OpenAI Gym, Fairseq, and Segment Anything**. Infrastructure uses **distributed GPU clusters** for training, with **Python-based SDK** (Tinker) handling scheduling, resource allocation, and recovery. The company emphasizes **publishing technical blog posts, papers, and code regularly** to advance broader understanding. Thinking Machines Lab reached **$12B valuation** less than a year old **before revealing products**, demonstrating massive investor appetite for AI labs led by proven talent.

**Salary Ranges**: Software Engineer (Product/Data Infrastructure) $300K-$350K | Technical Staff reported up to $500K (highest in AI industry, exceeding OpenAI/Anthropic's $190K-$400K range)

---

## AI/ML Tech Stack

### Tinker - Distributed LLM Fine-Tuning API

**What's unique**: Thinking Machines Lab launched **Tinker** (October 2025), a **flexible API for fine-tuning language models** that enables researchers and developers to **write training loops in Python on their laptops while Thinking Machines runs them on distributed GPUs**. Unlike traditional fine-tuning services that abstract away control, Tinker provides **low-level training API primitives** — **forward_backward** (gradient computation), **optim_step** (optimizer stepping), **save_state** (checkpointing), and **sample** (evaluation/inference) — giving users **direct control over custom training loops** while offloading the "heavy lifting of distributed compute and infrastructure management." The workflow: users create a **LoRA (Low-Rank Adaptation) training client** specifying base model (e.g., meta-llama/Llama-3.2-1B) and rank parameter, then orchestrate training through API calls controlling gradient updates, optimization steps, state saving, and sampling **inside custom Python loops**. Tinker handles **scheduling, tuning, resource management, and infrastructure reliability** behind the scenes — if a GPU crashes, the system automatically recovers without user intervention. This architecture differs fundamentally from high-level fine-tuning services (OpenAI, Anthropic) that require pre-defined hyperparameters and training recipes, versus Tinker's **"write code, we'll run it"** philosophy enabling novel training algorithms and research experimentation.

### Low-Level Training API - Python-Based Control Without Infrastructure Burden

**What makes it different**: Tinker implements a **Python-based training SDK** where users submit jobs through the API for fine-tuning **open-weight models**, which Thinking Machines **runs on internal GPU clusters** with full transparency into training loop implementation. The core primitives (**forward_backward, optim_step, save_state, sample**) expose **model internals** typically hidden in managed services, enabling researchers to implement **custom optimization algorithms, novel regularization techniques, and experimental training procedures** impossible with closed APIs. Users maintain **algorithmic control** (gradient accumulation strategies, learning rate schedules, mixed precision settings) while infrastructure code handles **distributed training orchestration, fault tolerance, and GPU resource allocation**. The architecture is **model-agnostic**, supporting both **traditional transformer architectures and Mixture-of-Experts (MoE) models** ranging from **compact models like Llama-3.2-1B to large MoEs like Qwen3-235B-A22B-Instruct**. Thinking Machines provides **tinker-cookbook** on GitHub (realistic fine-tuning examples) and **official documentation** (tinker-docs.thinkingmachines.ai) demonstrating best practices. This approach democratizes access to **distributed training infrastructure** previously requiring dedicated ML engineering teams to build and maintain, lowering barriers for academic researchers and startups.

### LoRA Fine-Tuning Infrastructure - Efficient Adaptation at Scale

**What sets Thinking Machines apart**: Tinker implements **Low-Rank Adaptation (LoRA) fine-tuning** rather than full fine-tuning, enabling **parameter-efficient adaptation** of large language models by training only **low-rank matrices** injected into transformer layers while keeping base model weights frozen. LoRA reduces **trainable parameters by 10,000x** (e.g., fine-tuning a 70B parameter model requires updating only ~7M parameters) while maintaining performance comparable to full fine-tuning. This efficiency enables **faster iteration cycles** (hours instead of days), **lower compute costs** (training on fewer GPUs), and **easier deployment** (storing only small adapter weights rather than full model copies). Thinking Machines' infrastructure **orchestrates distributed LoRA training** across **powerful GPU clusters** with automatic scaling — users specify desired throughput, and the system allocates appropriate resources. The platform handles **efficient gradient aggregation** across distributed workers, **checkpoint sharding** for large models, and **mixed precision training** (fp16/bf16) for memory efficiency. Unlike services requiring users to manage infrastructure (AWS SageMaker, GCP Vertex AI), Tinker provides **fully managed distributed training** while preserving algorithmic flexibility through low-level API access.

### Deterministic Output Achievement - 100% Consistent LLM Reasoning

**What's unique**: Thinking Machines Lab announced a **breakthrough achieving fully deterministic output in large language model reasoning processes**, solving the **model output uncertainty problem** that plagues production AI systems. Traditional LLMs produce **non-deterministic outputs** due to sampling randomness (temperature, top-p), floating-point precision variations across hardware, and non-deterministic GPU operations (parallel reduction order, tensor core accumulation). This non-determinism creates **reproducibility challenges** for debugging, compliance requirements (financial services, healthcare), and A/B testing (comparing model versions). Thinking Machines' solution enables **100% consistent output** for identical inputs, critical for applications requiring **auditability and explainability**. The technical approach likely involves **controlled random number generation** (fixing seeds across distributed workers), **deterministic GPU kernels** (forcing sequential operation ordering), and **precision management** (consistent rounding across hardware). This capability differentiates Thinking Machines from competitors where identical prompts produce varied outputs, complicating production deployment and regulatory compliance. The achievement demonstrates deep systems expertise in **distributed training infrastructure** and **numerical stability** — skills directly transferable from the team's OpenAI experience building ChatGPT's training infrastructure.

### Multimodal AI Focus - Human-AI Collaboration Over Autonomy

**What makes it different**: Thinking Machines Lab emphasizes **"multimodal systems that work with people collaboratively"** rather than pursuing **fully autonomous AI**, representing a philosophical departure from industry trends toward autonomous agents. The company views **multimodality as critical to enabling more natural and efficient communication**, preserving more information, better capturing intent, and supporting deeper integration into real-world environments. Multimodal capabilities span **text, images, audio, and video** processing with unified representations enabling cross-modal reasoning — answering questions about images, generating images from text descriptions, transcribing and summarizing video content. The founding team's experience building **ChatGPT's multimodal features** (GPT-4 Vision, DALL-E integration, voice mode) directly informs Thinking Machines' approach. The **human-AI collaboration focus** prioritizes **building more flexible, adaptable, and personalized AI systems** that augment human capabilities rather than replace them, targeting workflows where AI assists experts (scientists, programmers, designers) rather than operating independently. This approach aligns with the company's **public benefit corporation** structure, emphasizing **democratizing access to AI knowledge and tools** and making systems **more widely understood and customizable** versus proprietary black-box models.

### Frontier Model Capabilities - Science & Programming Domains

Thinking Machines Lab focuses on **building models at the frontier of capabilities in domains like science and programming**, targeting applications requiring **deep reasoning, domain expertise, and complex problem-solving**. The **science focus** likely encompasses **mathematical reasoning** (theorem proving, symbolic mathematics), **scientific computing** (simulating physical systems, analyzing experimental data), and **research assistance** (literature review, hypothesis generation). The **programming focus** extends beyond code generation to **program synthesis** (generating code from specifications), **formal verification** (proving code correctness), and **debugging assistance** (identifying and fixing bugs). The team's background includes creators of **PyTorch** (industry-standard deep learning framework), **OpenAI Gym** (reinforcement learning benchmark), and **Fairseq** (sequence modeling toolkit), demonstrating expertise in building **infrastructure for AI research**. Frontier capabilities require **scaling model size** (100B+ parameters), **high-quality training data** (curated scientific papers, code repositories), and **specialized training techniques** (reinforcement learning from human feedback, chain-of-thought prompting). The emphasis on **customizability** suggests Thinking Machines will enable users to **fine-tune frontier models** on domain-specific data (proprietary scientific datasets, internal codebases) through Tinker, unlocking performance impossible with generic pre-trained models.

---

## Sources

**Thinking Machines Lab Official**:
- [Thinking Machines Lab Homepage](https://thinkingmachines.ai/)
- [Announcing Tinker - Official Blog](https://thinkingmachines.ai/blog/announcing-tinker/)
- [Tinker Product Page](https://thinkingmachines.ai/tinker/)
- [Tinker Documentation](https://tinker-docs.thinkingmachines.ai/)

**GitHub & Open Source**:
- [Tinker Cookbook - GitHub](https://github.com/thinking-machines-lab/tinker-cookbook)

**Funding & Company**:
- [Thinking Machines Lab - Wikipedia](https://en.wikipedia.org/wiki/Thinking_Machines_Lab)
- [Mira Murati's Thinking Machines Lab $12B Valuation - TechCrunch](https://techcrunch.com/2025/07/15/mira-muratis-thinking-machines-lab-is-worth-12b-in-seed-round/)
- [Thinking Machines Lab Raises $2B - TechCrunch](https://techcrunch.com/2025/06/20/mira-muratis-thinking-machines-lab-closes-on-2b-at-10b-valuation/)
- [Thinking Machines Lab Profile - Tracxn](https://tracxn.com/d/companies/thinking-machines-lab/__uSPGa2dnvHKfeiBjkNrQCkPFX-QcGogbaGhPqmvYk8k)

**Tinker Technical Analysis**:
- [Tinker: Distributed LLM Fine-Tuning - VentureBeat](https://venturebeat.com/ai/thinking-machines-first-official-product-is-here-meet-tinker-an-api-for)
- [Tinker API Launch - MarkTechPost](https://www.marktechpost.com/2025/10/02/thinking-machines-launches-tinker-a-low-level-training-api-that-abstracts-distributed-llm-fine-tuning-without-hiding-the-knobs/)
- [Inside Tinker - Superintelligence News](https://superintelligencenews.com/research/tinker-api-thinking-machines-fine-tuning-open-models/)
- [Thinking Machines Releases Tinker API - InfoQ](https://www.infoq.com/news/2025/10/thinking-machines-tinker/)

**Team & Leadership**:
- [Mira Murati - Wikipedia](https://en.wikipedia.org/wiki/Mira_Murati)
- [Meet Mira Murati - Fortune](https://fortune.com/2025/10/03/mira-murati-career-ai-thinking-machines-goldman-sachs-tesla-leap-openai/)
- [Former OpenAI CTO Launches Startup - Daily Star](https://www.thedailystar.net/tech-startup/news/former-openai-cto-launches-new-startup-hires-several-openai-employees-3828161)
- [Meta Researcher Joins Thinking Machines Lab - Yahoo Finance](https://finance.yahoo.com/news/top-meta-ai-researcher-joined-230058151.html)

**Job Postings & Compensation**:
- [Thinking Machines Lab Careers](https://job-boards.greenhouse.io/thinkingmachines)
- [Software Engineer, Product Job](https://job-boards.greenhouse.io/thinkingmachines/jobs/4867473008)
- [Software Engineer, Data Infrastructure Job](https://job-boards.greenhouse.io/thinkingmachines/jobs/4879755008)
- [AI Startup Dangling $500K Salaries - KRON4](https://www.kron4.com/news/technology-ai/sf-ai-startup-reportedly-dangling-500k-salaries/)
- [Thinking Machines Lab - Built In SF](https://www.builtinsf.com/company/thinking-machines-lab)

---

*Last updated: November 30, 2025*
