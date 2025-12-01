# Anthropic - Technology Stack

**Company:** Anthropic PBC (Public Benefit Corporation)
**Founded:** 2021
**Focus:** AI safety and research, Claude AI models
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

Anthropic operates a **multi-cloud infrastructure** with **AWS as primary training partner** ($8B investment) and **Google Cloud** for additional capacity ($2B investment). The company's deployment stack includes **Amazon Bedrock**, **Google Vertex AI**, and direct API access. Claude Code, Anthropic's developer tool, is built with **TypeScript**, **React**, **Ink** (React for CLIs), **Yoga** (cross-platform layout engine), and **Bun** runtime - notably, **90% of Claude Code's codebase was written by Claude itself**. The **Claude Agent SDK** uses the same infrastructure powering Claude Code, providing agent orchestration, memory/context management, tool usage, and permission management. Anthropic's engineering philosophy eliminates the traditional research/engineering divide - all technical hires share the title **"Member of Technical Staff"** reflecting the dissolved boundary between ML research and engineering in the era of foundation models. The organization structure emphasizes both scaling (training larger models) and safety (Constitutional AI, interpretability research).

**Salary Ranges**: Member of Technical Staff (unified research/eng role) - Median $545K total comp | Software Engineer $550K-$700K (Senior→Lead) | Research Engineer Tokens $315K-$340K | Research Scientist Interpretability $315K-$560K

---

## AI/ML Tech Stack

### Constitutional AI - Unique Safety-First Training

**What's unique**: Anthropic pioneered **Constitutional AI (CAI)**, a training methodology where models learn harmlessness from AI feedback rather than exclusively human feedback, guided by a written "constitution" of principles. The two-phase process: **(1) Supervised Learning Phase** - model generates outputs, critiques them against constitutional principles, revises responses, then trains on these self-improved outputs; **(2) Reinforcement Learning Phase** - a paired AI model provides feedback to further reduce harmful outputs, replacing human labelers with AI feedback (RLAIF vs RLHF). This approach overcomes RLHF limitations: human labelers struggle with consistency on subjective harm judgments, can't scale to billions of training examples, and introduce geographical/cultural biases. CAI enables transparent principles (the constitution is public), scalable oversight, and principled rather than preference-based safety. Claude's constitution includes principles like "Choose the response that is least intended to build a relationship with the user" and "Choose the response that sounds most similar to what a peaceful, ethical, and dispassionate person would say."

### Project Rainier - Massive Multi-Cloud Training Infrastructure

Anthropic's training infrastructure centers on **Project Rainier**, AWS's largest AI cluster featuring **500,000+ AWS Trainium 2 chips** across multiple US data centers, scaling to **1 million chips** by year-end, delivering **hundreds of exaflops** of compute. This is **over 5x larger** than Anthropic's previous training cluster. The diversified compute strategy spans three platforms: **AWS Trainium/Inferentia** (primary, up to 50% cost savings vs EC2), **Google Cloud TPUs** (>1GW TPU buildout, including TPUv7 "Ironwood"), and **NVIDIA GPUs** (supplementary capacity). This multi-cloud approach provides redundancy, geographic distribution, and leverage across hardware vendors - a unique strategy among AI labs heavily investing in vendor lock-in. **Claude 3.5 Haiku** achieved **60% faster inference** on Trainium2 via latency-optimized mode. Training stack: **PyTorch**, **JAX**, and **Triton** frameworks across all platforms.

### Interpretability Research - "Opening the Black Box"

**What makes Anthropic different**: Dedicated interpretability team with researchers who pioneered mechanistic interpretability and authored foundational scaling laws papers. In May 2024, Anthropic published **"Scaling Monosemanticity"**, applying dictionary learning (sparse autoencoders) to **Claude 3 Sonnet** - the **first detailed look inside a production-grade LLM**. They identified **tens of millions of "features"** (neuron combinations representing semantic concepts) and used **scaling laws to guide sparse autoencoder training**. Key finding: understanding AI safety improves with model scale using interpretable features. Research also revealed that **repeating just 0.1% of training data 100 times** degrades an 800M parameter model to perform like a 2x smaller model - critical insights for data quality. The [Transformer Circuits](https://transformer-circuits.pub/) research thread represents systematic efforts to decompose language models into understandable components, advancing the field toward "AI safety through understanding."

### Multi-Agent Research Systems

Anthropic's advanced capabilities include **multi-agent systems** where multiple Claude instances collaborate. The Research feature uses a **lead agent (Claude Opus 4)** that plans research and spawns **parallel subagents (Claude Sonnet 4)** that search simultaneously. This architecture achieved **90.2% better performance** than single-agent Claude Opus 4 alone, demonstrating emergence of coordinated intelligence. The system autonomously uses tools in loops, with the lead agent orchestrating parallel execution - a architecture pattern Anthropic productized through the Claude Agent SDK.

### Model Family Architecture

The **Claude 3/3.5/4 families** span capability tiers: **Opus** (most capable, complex reasoning), **Sonnet** (balanced intelligence and speed), **Haiku** (fastest, most compact). Training methodology combines unsupervised learning, Constitutional AI, and traditional RLHF. Models feature **extended context windows** (up to 200K+ tokens), **vision capabilities** (analyzing images), and **tool use** (function calling, computer use). The architecture leverages transformers with proprietary modifications, trained on multi-platform infrastructure (AWS Trainium, Google TPUs, NVIDIA GPUs). Recent releases show Anthropic's aggressive scaling: **Sonnet 4.5** surpasses previous Opus performance at lower cost, demonstrating better scaling efficiency.

### Scaling Laws & Responsible Deployment

Anthropic's research contributions include foundational work on **scaling laws** predicting model performance from compute, dataset size, and parameters. The company implements a **Responsible Scaling Policy (RSP)** defining AI Safety Levels (ASL-1 through ASL-4), with Claude 3 classified as ASL-2 (current models). The policy triggers safety evaluations at capability thresholds (e.g., models that could enhance CBRN risks or autonomously self-replicate), mandating additional safeguards before further scaling. This systematic approach to deployment differentiates Anthropic from labs prioritizing speed-to-market over measured release.

**Salary Ranges**: Research roles $315K-$560K | Member of Technical Staff median $545K | Engineering $550K-$700K | Stock grants vest over 4 years (25% annually)

---

## Sources

**Anthropic Engineering & Research**:
- [How Anthropic teams use Claude Code](https://www.anthropic.com/news/how-anthropic-teams-use-claude-code)
- [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [A postmortem of three recent issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues)
- [The engineering challenges of scaling interpretability](https://www.anthropic.com/research/engineering-challenges-interpretability)

**Constitutional AI & Safety**:
- [Claude's Constitution](https://www.anthropic.com/news/claudes-constitution)
- [Constitutional AI: Harmlessness from AI Feedback (PDF)](https://www-cdn.anthropic.com/7512771452629584566b6303311496c262da1006/Anthropic_ConstitutionalAI_v2.pdf)
- [Anthropic's Responsible Scaling Policy](https://www.anthropic.com/news/anthropics-responsible-scaling-policy)

**Interpretability Research**:
- [Scaling Monosemanticity: Claude 3 Sonnet Features](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model)
- [Scaling Laws and Interpretability of Learning from Repeated Data](https://www.anthropic.com/research/scaling-laws-and-interpretability-of-learning-from-repeated-data)

**Infrastructure Partnerships**:
- [Powering AI development with AWS](https://www.anthropic.com/news/anthropic-amazon-trainium)
- [Expanding our use of Google Cloud TPUs](https://www.anthropic.com/news/expanding-our-use-of-google-cloud-tpus-and-services)
- [AWS and Anthropic complete Project Rainier](https://www.datacenterknowledge.com/supercomputers/project-rainer-aws-anthropic-complete-massive-ai-supercomputing-cluster)
- [Amazon invests additional $4B in Anthropic](https://www.aboutamazon.com/news/aws/amazon-invests-additional-4-billion-anthropic-ai)

**Model Releases**:
- [Introducing Claude Sonnet 4.5](https://www.anthropic.com/news/claude-sonnet-4-5)
- [Introducing the Claude 3 Family](https://www.anthropic.com/news/claude-3-family)
- [Claude 3 Model Card (PDF)](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)

**Jobs & Compensation**:
- [Anthropic Careers](https://www.anthropic.com/careers)
- [Research Scientist, Interpretability](https://job-boards.greenhouse.io/anthropic/jobs/4020159008) - $315K-$560K
- [Member of Technical Staff roles](https://www.anthropic.com/jobs)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/anthropic/salaries)
- [Anthropic Salary Overview - NAHC](https://www.nahc.io/blog/anthropic-salary-overview-how-much-do-employees-get-paid)

---

*Last updated: November 30, 2025*
