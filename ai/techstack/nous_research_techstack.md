# Nous Research - Technology Stack

**Company:** Nous Research Inc.
**Founded:** 2022 (as volunteer collective), incorporated 2023
**Focus:** Open-source AI models and decentralized training infrastructure
**Headquarters:** United States (distributed team)

---

## Non-AI Tech Stack

Nous Research was founded in **2022 as a volunteer collective** of AI enthusiasts who met on **Discord, GitHub, and Twitter**, before incorporating in **2023**. The founding team includes **Jeffrey Quesnelle** (CEO), **Karan Malhotra** (Head of Behavior), **Teknium** (Head of Post Training, pseudonymous), and **Shivani Mitra**. The team started by fine-tuning existing models like **Meta's Llama and Mistral** to create their own versions, releasing the popular **Hermes series**. The company raised **$70M+ total funding**: **~$20M seed** (Distributed Global, North Island Ventures, Delphi Digital) and **$50M Series A** (April 2025, led by **Paradigm** at **$1B token valuation**). The team grew from a volunteer collective to **~20+ employees**. The Discord community has **75,000+ members**. Infrastructure uses **Solana blockchain** for the Psyche decentralized training network. Core models are trained on cloud GPU infrastructure including **192 NVIDIA B200 GPUs** for Hermes 4. All models, datasets, and training code are **fully open-source** under permissive licenses, published on **Hugging Face** and **GitHub**. The company positions itself as a leader in the **American open-source AI movement**, contrasting with closed-model labs like OpenAI and Anthropic.

**Salary Ranges**: Limited public data; small team structure with crypto/token-based compensation model

---

## AI/ML Tech Stack

### Hermes Model Family - User-Aligned Open-Source LLMs Without Censorship

**What's unique**: The **Hermes series** (Hermes 3, Hermes 4) are **full-parameter fine-tuned models** that prioritize **user alignment over corporate alignment** — meaning the model follows user instructions without the heavy-handed refusals common in ChatGPT/Claude. **Hermes 3** was the **first fine-tuned Llama 3.1 405B** to be released publicly. **Hermes 4** (August 2025) achieves **frontier-level performance through pure post-training techniques** — no architecture changes, just better fine-tuning. Key characteristics: **long responses, low hallucination rate, absence of censorship mechanisms, strong agentic capabilities**. The models support **hybrid reasoning** — toggling between standard responses and explicit reasoning using `<think>...</think>` tags. Hermes models consistently rank highly on **Hugging Face's Open LLM Leaderboard** and have been downloaded millions of times. This "uncensored but capable" positioning attracts developers who want powerful models without corporate guardrails.

### DisTrO - 1000x Communication Compression for Distributed Training

**What makes it different**: **DisTrO (Distributed Training Over-The-Internet)** reduces inter-GPU communication requirements by **1,000x to 10,000x** without degrading convergence. Traditional distributed training requires synchronizing full gradients across all GPUs with extremely high bandwidth connections. DisTrO uses **Discrete Cosine Transform (DCT)** — the same principle behind JPEG image compression — to compress gradients before transmission. In testing, a Llama 2 training run reduced communication from **74.4 GB to 86.8 MB** (857x compression). This enables model training on **ordinary internet connections** (as low as 100 Mbps download, 10 Mbps upload) while matching AdamW+All-Reduce convergence rates. DisTrO is open-sourced on GitHub. This innovation is foundational to Nous's vision of **globally distributed training** using volunteer compute rather than centralized datacenters.

### Psyche Network - Blockchain-Coordinated Decentralized Training on Solana

**What sets Nous apart**: **Psyche** is an open infrastructure for decentralized AI training that pools **idle GPUs from around the globe** — connecting consumer 4090s, datacenter A100s, and H100s into a unified training cluster. **Solana blockchain** serves as the coordination layer: smart contracts store training metadata, participant lists, and random task assignments, ensuring **transparency, tamper-proofing, and censorship-resistance**. In December 2024, Psyche successfully trained a **15 billion parameter model across 11,000 steps** on globally distributed hardware. The system is **fault-tolerant** — nodes can join/leave during training without disrupting the process. Solana's **high throughput and low transaction costs** enable the microtransactions required for incentivizing distributed compute contributors. This represents a fundamentally different approach than xAI's Colossus or OpenAI's Azure clusters — betting on **distributed commodity hardware** rather than massive centralized infrastructure.

### Forge Reasoning API - Multi-Model Collaboration with Code Interpreter

**What's unique**: **Forge** supercharges any model with **advanced reasoning capabilities and code interpreter access**. The system integrates multiple techniques: **Mixture of Agents (MoA)** where multiple models respond, confer, and synthesize consensus answers; **Chain of Code** for step-by-step reasoning with executable verification; and **Monte Carlo Tree Search** for exploring solution spaces. Forge allows **mixing models** — Hermes, Claude, Gemini, and OpenAI models can collaborate on difficult tasks. Users select **reasoning tiers** appropriate to problem complexity. The underlying Hermes models provide agentic features: **XML tags for structured output, scratchpads for intermediate processing, internal monologues for transparent decision-making, Mermaid diagrams for visualization**. Forge augments Hermes 70B to be **competitive with much larger frontier models** on reasoning benchmarks. Available via API and web interface (forge.nousresearch.com).

### Synthetic Data + Rejection Sampling - 1000+ Verifiers for Quality Control

**What makes it different**: Hermes models are trained on **primarily synthetically generated data** with rigorous quality control. The methodology uses **rejection sampling with 1,000+ task-specific verifiers** — each verifier evaluates specific aspects of model outputs (factual accuracy, instruction following, code correctness, reasoning validity). **Hermes 4** was post-trained on **~60 billion tokens** with emphasis on verified reasoning data. The training pipeline includes: (1) **synthetic data generation** for diverse reasoning scenarios and edge cases, (2) **rejection sampling** to filter high-quality responses, (3) **advanced data packing and flex attention** for efficient training. This approach produces models with **low hallucination rates** and **strong instruction-following** — the quality control happens at data generation time rather than relying on RLHF post-hoc. Earlier Hermes models were trained on **GPT-4 synthetic outputs** (300K+ instructions for Hermes-13b).

### Hermes 4 Training - 192 B200 GPUs, 60B Tokens, 71K GPU-Hours

**What sets Nous apart**: **Hermes 4** training infrastructure demonstrates Nous's ability to operate at frontier scale despite small team size. Training used **192 NVIDIA B200 GPUs** (latest Blackwell generation, ~180GB HBM3e each, 20 petaFLOPS per GPU). The **405B model required 71,616 GPU-hours**. Training config: **16,384 token context length**, **global batch size 384**, cosine learning rate schedule with **300 warmup steps and 9,000 total steps**. Parallelism strategy combines **Data Parallelism, Tensor Parallelism, and Fully Sharded Data Parallelism (FSDP)**. The system achieves **>99.9% batch efficiency** through efficient fillers handling highly heterogeneous sample lengths, plus flexible attention and complex loss masks. Hermes 4 is available in **14B, 70B, and 405B** parameter sizes, all open-weight on Hugging Face.

### Community-Driven Development - Discord to $1B Valuation

**What's unique**: Nous grew from a **Discord volunteer collective** to a **$1B valued company** without traditional VC-backed startup trajectory. The 75K+ member Discord community provides: (1) **rapid feedback on model releases**, (2) **distributed dataset curation** from community contributors, (3) **crowdsourced evaluation** of model capabilities. Key figures like **Teknium** (pseudonymous) maintain credibility through technical contributions rather than corporate credentials. This community-first approach enables **faster iteration** than traditional AI labs — the community can identify issues and suggest improvements within hours of release. The model of "open-source everything, build community, then raise at scale" provides a template for other open-source AI projects.

---

## Sources

**Nous Research Official**:

- [Nous Research Homepage](https://nousresearch.com/)
- [Hermes 3 Announcement](https://nousresearch.com/hermes3/)
- [Hermes 4 Homepage](https://hermes4.nousresearch.com/)
- [Psyche Network Architecture](https://nousresearch.com/nous-psyche/)
- [Forge Reasoning API](https://forge.nousresearch.com/)
- [Forge API Announcement](https://nousresearch.com/introducing-the-forge-reasoning-api-beta-and-nous-chat-an-evolution-in-llm-inference/)
- [Nous Research Blog](https://nousresearch.com/blog/)
- [Nous Research Discord](https://discord.com/invite/nousresearch)

**GitHub & Hugging Face**:

- [DisTrO GitHub](https://github.com/NousResearch/DisTrO)
- [DisTrO Preliminary Report](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf)
- [Atropos RL Framework](https://github.com/NousResearch/Atropos)
- [Teknium GitHub](https://github.com/teknium1)
- [Hermes 4 405B - Hugging Face](https://huggingface.co/NousResearch/Hermes-4-405B)
- [Hermes 4 70B - Hugging Face](https://huggingface.co/NousResearch/Hermes-4-70B)
- [Hermes 4 14B - Hugging Face](https://huggingface.co/NousResearch/Hermes-4-14B)
- [Hermes 3 Llama 3.1 8B - Hugging Face](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B)
- [Nous-Hermes-13b - Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-13b)
- [Nous-Hermes-Llama2-13b - Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-Llama2-13b)

**Funding & Company**:

- [Paradigm $50M Series A - Fortune](https://fortune.com/crypto/2025/04/25/paradigm-nous-research-crypto-ai-venture-capital-deepseek-openai-blockchain/)
- [$50M Raise - SiliconANGLE](https://siliconangle.com/2025/04/25/nous-research-raises-50m-decentralized-ai-training-led-paradigm/)
- [Nous $65M Funding - The AI Insider](https://theaiinsider.tech/2025/04/30/nous-research-lands-65m-to-champion-open-source-approach-to-ai-development/)
- [Paradigm $50M - The Block](https://www.theblock.co/post/352000/paradigm-leads-50-million-usd-round-decentralized-ai-project-nous-research)
- [Nous Research Crunchbase](https://www.crunchbase.com/organization/nous-research)
- [Nous Research PitchBook](https://pitchbook.com/profiles/company/572004-37)
- [Nous Research - RootData](https://www.rootdata.com/Projects/detail/Nous%20Research?k=MTczMzQ%3D)
- [Nous Research Team - CryptoRank](https://cryptorank.io/price/nous-research/team)

**Technical & Research**:

- [DisTrO Announcement - VentureBeat](https://venturebeat.com/ai/this-could-change-everything-nous-research-unveils-new-tool-to-train-powerful-ai-models-with-10000x-efficiency)
- [Distributed Training Over Internet - VentureBeat](https://venturebeat.com/ai/nous-research-is-training-an-ai-model-using-machines-distributed-across-the-internet)
- [DisTrO - Hacker News](https://news.ycombinator.com/item?id=41371083)
- [DisTrO - Simon Willison](https://simonwillison.net/2024/Aug/27/distro/)
- [Hermes 3 Open Source - MarkTechPost](https://www.marktechpost.com/2024/08/17/nous-research-open-sources-hermes-3-a-series-of-instruct-and-tool-use-model-with-strong-reasoning-and-creative-abilities/)
- [Hermes 4 Release - MarkTechPost](https://www.marktechpost.com/2025/08/27/nous-research-team-releases-hermes-4-a-family-of-open-weight-ai-models-with-hybrid-reasoning/)
- [Hermes 4 Outperforms ChatGPT - VentureBeat](https://venturebeat.com/ai/nous-research-drops-hermes-4-ai-models-that-outperform-chatgpt-without-content-restrictions)
- [Hermes 3 Existential Crises - VentureBeat](https://venturebeat.com/ai/meet-hermes-3-the-powerful-new-open-source-ai-model-that-has-existential-crises)
- [Unveiling Hermes 3 - Lambda Labs](https://lambda.ai/blog/unveiling-hermes-3-the-first-fine-tuned-llama-3.1-405b-model-is-on-lambdas-cloud)
- [Forge API - MarkTechPost](https://www.marktechpost.com/2024/11/13/nous-research-introduces-two-new-projects-the-forge-reasoning-api-beta-and-nous-chat/)

**Psyche Network**:

- [Nous Secures $50M for Decentralized AI - Cointelegraph](https://cointelegraph.com/news/nous-research-raises-50m-paradigm-decentralized-ai-solana)
- [Psyche Testnet Launch - PANews](https://www.panewslab.com/en/articles/rsw08x0c)
- [Psyche Launches - AI Base](https://www.aibase.com/news/18116)
- [Nous and Psyche Revolution - OAK Research](https://oakresearch.io/en/analyses/innovations/nous-research-psyche-open-source-decentralized-ai-revolution)
- [What is Nous Research - Bitget](https://web3.bitget.com/en/academy/what-is-nous-research-nous-paradigm-backed-project-redefining-open-source-ai-with-50-million)

**Interviews & Podcasts**:

- [Karan Malhotra on Data Synthesis - Practical AI #255](https://changelog.com/practicalai/255)
- [Ethical User-Aligned AI - TWiT](https://twit.tv/posts/tech/building-ethical-user-aligned-ai-what-nous-research-doing-differently)

---

*Last updated: December 6, 2025*
