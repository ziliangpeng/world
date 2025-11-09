# DeepSeek (深度求索) - Deep Dive

## 3. DeepSeek

### 🏢 Company Overview

**DeepSeek (深度求索)** has emerged as one of the most significant and disruptive players in the global LLM landscape despite being founded in mid-2023. The company is known for radical cost efficiency, powerful open-source models, and technical innovations that fundamentally challenged Western assumptions about frontier AI training costs and efficiency. DeepSeek-V3 (671B parameters with 37B active) and DeepSeek-R1 (reasoning model) have attracted unprecedented international attention for achieving frontier performance at less than 3% of typical Western training budgets. The company's emergence has triggered geopolitical and industry-wide reassessment of AI development economics.

### 📚 Founding Story and History

#### Founder Background

**Liang Wenfeng (梁文锋) - Founder & CEO**
- **Birth**: 1985 in Mililing village, Wuchuan city, Guangdong Province (humble rural beginnings)
- **Family**: Parents were both primary school teachers
- **Education**:
  - Zhejiang University, BS in Electronic Information Engineering (2007)
  - Zhejiang University, MS in Information and Communication Engineering (2010)
  - Master's dissertation: "Study on object tracking algorithm based on low-cost PTZ camera"
- **Early career**: Relocated to Chengdu after graduation, experimented with AI in various industries with multiple failed ventures
- **Breakthrough discovery**: Found success applying AI to quantitative finance
- **High-Flyer founding path**:
  - 2008: Formed team during financial crisis to accumulate financial market data
  - 2015: Co-founded Hangzhou Huanfang Technology Co Ltd.
  - February 2016: Co-founded Ningbo High-Flyer Quantitative Investment Management (梁文锋与两位工程系同学联合创办)
  - Built High-Flyer into major hedge fund with 100+ billion yuan AUM (≈$14B) by 2021
  - Became billionaire through High-Flyer's success
- **Transition to AI**: May 2023 - announced High-Flyer would pivot to AGI development, launching DeepSeek

#### Strategic Funding Model

Unlike typical venture-backed AI startups, DeepSeek's unique funding structure enabled technical breakthroughs:

**Funding Structure:**
- **Sole investor**: High-Flyer hedge fund (梁文锋's company)
- **Initial investment**: $50 million (modest vs. Western competitors)
- **Ownership**: 100% owned and funded by High-Flyer, no external VC pressure
- **Patient capital**: Hedge fund's long-term investment horizon, not beholden to quarterly metrics
- **Key advantage**: Pre-emptive GPU acquisition (10,000 Nvidia A100s in 2022-2023 before US restrictions)
- **Revenue reinvestment**: Model deployment revenues reinvested into R&D

This structure enabled DeepSeek to pursue high-risk, long-term research without external pressure to monetize quickly—contributing directly to breakthrough efficiency innovations.

#### Timeline and Milestones

**2023 - Quiet Launch:**
- **May 2023**: High-Flyer announces AGI pursuit, DeepSeek officially established
- **July 2023**: DeepSeek officially founded as AI research subsidiary
- **~100 engineers**: Team scaled to ~100 within founding period (vs. 40 for Moonshot)
- **Nov 2, 2023**: DeepSeek Coder released (establishing technical credibility)
- **Nov 29, 2023**: DeepSeek-LLM series launched (7B, 13B, 67B, 97B variants)
- **Strategy**: Deliberate low-profile approach, minimal marketing, focus on technical excellence

**2024 - Technical Innovations:**
- **Early 2024**: Continued research into MLA, MoE efficiency improvements
- **Throughout 2024**: Various model releases demonstrating steady technical progress
- **Infrastructure achievement**: Showed ability to train competitive models with modest compute (2,048 H800 GPUs)
- **Market positioning**: Built reputation for cost-transparency and open-source commitment

**2024-2025 - Global Impact:**
- **December 26, 2024**: **DeepSeek-V3 released** - 671B parameters, 37B active
  - Training cost: $5.58 million (vs. OpenAI GPT-4o estimated $100M+)
  - 2,048 H800s × 57 days = 2.78M GPU hours
  - Performance: Competitive with GPT-4, Claude 3.5 Sonnet, o1
  - Open-source under MIT license
  - Immediate global impact: Sparked reassessment of AI economics

- **January 2025**: **DeepSeek-R1 released** - Reasoning model competing with o1
  - Training cost: $294,000 for post-training (frontier reasoning at minimal cost)
  - Outperforms o1-mini on many benchmarks
  - Open-source weights released

- **January 2025**: **DeepSeek Chatbot launched**
  - Over 10 million downloads within days
  - Became top-1 app in Apple App Store (China)
  - Demonstrated consumer-facing viability alongside research focus

- **January 2025**: DeepSeek received personal endorsement from China's Premier Li Qiang (significant political validation)

DeepSeek's transformation from unknown startup (mid-2023) to global AI disruptor (18 months) represents unprecedented impact velocity, fundamentally reshaping assumptions about AI development economics and feasibility.

#### Core Team Philosophy

**Hiring and Culture:**
- **Team size**: ~150 people (deliberately lean, high-density talent strategy)
- **Hiring philosophy**: Preference for new graduates and PhD students early in career
- **Unconventional approach**: Hired several doctoral students who hadn't yet graduated
- **Notable team members**:
  - Guo Daya: 2023 PhD graduate, Sun Yat-sen University
  - Zhu Qihao: Fresh PhD graduate, Peking University
  - Dai Damai: Fresh PhD graduate, Peking University
- **Organizational culture**:
  - Deliberately flat hierarchy, downplays job ranks
  - Research topic determination and resource mobilization democratized
  - Emphasis on "talent density" over "personnel scale" (质量优先)
  - All-elite team composition

**Leadership Philosophy:**
- Liang's quant trading background translated to AI: optimization mindset, cost-consciousness, efficiency obsession
- Philosophy: "Frontier models don't require $100M budgets—they require superior algorithms and architecture"
- Radical transparency: Publishing training costs and infrastructure details (unusual in AI industry)

### 💰 Funding and Investment

DeepSeek represents an unusual funding model:
- **Sole investor**: High-Flyer hedge fund, wholly owned and funded by the parent company
- **$50 million initial investment**: Significant but modest compared to Western AI companies
- **No external venture funding**: Deliberate choice to avoid external investor pressure
- **Self-funded operations**: Reinvestment of model deployment revenues into R&D

This funding structure enabled DeepSeek to pursue long-term, high-risk research without external pressure to monetize quickly, contributing to technical breakthroughs.

### 🎯 Strategic Positioning

DeepSeek positions as **"Efficient AI for Everyone"** with distinctive strategic elements:

1. **Cost Transparency**: Publicly sharing training costs ($5.58M for V3, $294K for R1 post-training)
2. **Open-Source Commitment**: Releasing models under permissive licenses
3. **Radical Efficiency**: Demonstrating that frontier models don't require $100M+ training budgets
4. **Technical Excellence**: Focus on novel architectures and training methodologies
5. **Speed-to-Market**: Rapid iteration and release cycles
6. **Independent Path**: Refusing to follow Silicon Valley playbooks

DeepSeek's positioning challenges fundamental assumptions about AI development, suggesting it's about technical innovation rather than brute-force compute scale.

### 🔧 Technical Innovations and Architecture

DeepSeek's efficiency breakthrough stems from revolutionary hardware-software co-design and novel architecture innovations:

**Multi-Head Latent Attention (MLA) - KV Cache Revolution**
- Compresses key-value pairs into low-rank latent representations
- **KV cache reduction**: 93.3% smaller compared to standard attention
- Low-rank compression allows different information to be used differently by heads
- Maintains full performance while dramatically reducing memory footprint
- Critical for 128K+ context windows and efficient inference
- Enables near-linear scaling of inference with context length

**DeepSeekMoE Architecture - Expert Specialization**
- **257 total experts**: 256 routed + 1 shared expert per layer
- **8 activated experts per token** (~9.4% activation rate for 671B model)
- Fine-grained routing: dynamic token routing to specialized experts
- **Auxiliary-loss-free load balancing**: expert-specific bias terms instead of traditional auxiliary losses
- Minimizes performance degradation from load balancing constraints
- Enables more granular expert specialization than competitor MoE designs

**FP8 Mixed-Precision Training - Hardware Revolution**
- **First large model to train with FP8** precision end-to-end
- Weights, activations, and gradients stored in FP8; accumulation in FP32
- **50% memory reduction** compared to FP16 training
- **Custom FP8-GEMM-with-rescaling kernel** for NVIDIA H-series GPUs
- Blockwise weight scaling (128×128 blocks) + tilewise activation scaling (1×128 tiles)
- **Online quantization**: computes scaling factors in real-time rather than delayed approach
- Leverages NVIDIA WGMMA instructions for maximum throughput (~2 petaFLOPS on H100)
- 4x bandwidth reduction through pre-transmission FP8 quantization

**Hardware-Software Co-Design**
- Optimized for NVIDIA H800 GPU cluster (2048 GPUs for V3 training)
- Custom kernels that leverage native FP8 support in Hopper architecture
- Infrastructure optimizations: communication overlap, gradient accumulation patterns
- Cross-node optimization eliminated traditional multi-node training bottlenecks
- Near-full computation-communication overlap through pipelining

**Multi-Token Prediction Training**
- Predicts up to next 2 tokens during training (85-90% acceptance rate for second token)
- Improves model performance on downstream benchmarks
- Reduces training compute per token without sacrificing quality

**Group Relative Policy Optimization (GRPO) - DeepSeek RL Innovation**
- **Invented by DeepSeek** and first published in DeepSeekMath (Feb 2024, arxiv:2402.03300)
- Originally designed for mathematical reasoning, extended to general reasoning in R1
- **Eliminates critic model**: compares response groups relative to group average rather than against learned value function
- **40-60% memory reduction** vs standard PPO by removing separate value function network
- Generates multiple responses per prompt, uses mean reward as baseline for advantage calculation
- Reduces policy update variance through group-based comparisons vs individual comparisons
- KL divergence constraint prevents drastic policy shifts during training
- Enables emergent reasoning patterns: self-reflection, verification, dynamic strategy adaptation
- No labeled reasoning trajectories required—verification via programmable reward functions
- Foundation for DeepSeek-R1 reasoning breakthrough ($294K post-training cost, Nature publication)

**Comparative Positioning**

DeepSeek's efficiency approach differs fundamentally from competitors:

**Architectural Advantages**:
- MLA vs Standard Attention: 93% KV cache reduction with no performance loss
- MoE vs Dense: 9.4% activation rate vs 100% for dense models (11x compute savings)
- GRPO vs PPO: 40-60% less memory, simpler training pipeline

**Hardware Optimization Depth**:
- Most competitors use FP16 training; DeepSeek uses native FP8
- Custom kernel development targeting specific GPU architectures
- Infrastructure co-design from GPU-level upward

**Training Philosophy**:
- **Efficiency-first**: Every component optimized for cost, not just performance
- **Hardware-aware**: Designs leverage native capabilities vs workarounds
- **End-to-end**: Optimization spans architecture → training → quantization → hardware

**Production Impact**:
- V3 training: $5.58M (70x cheaper than industry standard)
- R1 post-training: $294K (99.8% cheaper than GPT-4 reasoning models)
- Inference at scale: 93% KV cache reduction + FP8 storage = massive cost savings

### 👥 Team Background

**Leadership & Founder**

**Liang Wenfeng (梁文锋)** - Founder and CEO
- Born 1985 in Wuchuan, Guangdong; from family of educators
- Bachelor's in Electronic Information Engineering from Zhejiang University (2007)
- Master's in Information and Communication Engineering from Zhejiang University (2010)
- Co-founded quantitative hedge fund High-Flyer (2015) with AI-powered trading strategies
- Launched DeepSeek in July 2023 as High-Flyer's AI division
- Personally holds 84% stake in DeepSeek through shell corporations
- Philosophy: Patient capital and technical independence enable long-term AI research
- Links: [Wikipedia](https://en.wikipedia.org/wiki/Liang_Wenfeng)

**Core Research & Technical Leadership**

**Zhihong Shao (邵智宏)** - Core Researcher, LLM Reasoning Lead
- CS PhD from Tsinghua University; BS from Beihang University
- Named MIT Tech Review's 35 Innovators Under 35
- Key contributor to DeepSeek-R1, DeepSeek-Prover, DeepSeek-Coder-V2, DeepSeekMath
- Co-authored ICLR 2024 paper on ToRA (Tool-Integrated Reasoning Agent)
- Received Lenovo Scholarship from Tsinghua University (2023)
- Mentored by Nan Duan and Weizhu Chen at MSRA
- Links: [Personal Website](https://zhihongshao.github.io/), [LinkedIn](https://www.linkedin.com/in/zhihong-shao-18a1981b4/), [Google Scholar](https://scholar.google.com/citations?user=PZy4HEIAAAAJ&hl=zh-CN), [GitHub](https://github.com/ZhihongShao), [Twitter](https://x.com/zhs05232838)

**Zhibin Gou (苟志斌)** - Core Researcher, Reasoning & RL Specialist
- MS from Tsinghua University's SIGS, advised by Prof. Yujiu Yang
- BS from Beijing University of Posts and Telecommunications
- Research Intern at Microsoft Research Asia NLC Group (Jan 2023 - May 2024)
- Mentored by Yeyun Gong and Weizhu Chen at MSRA
- Co-authored ICLR 2024 paper on ToRA and CRITIC (200+ citations)
- Core contributor to DeepSeek-R1
- Links: [Personal Website](https://zubingou.github.io/), [Google Scholar](https://scholar.google.com/citations?user=jTMOma8AAAAJ&hl=en), [GitHub](https://github.com/ZubinGou), [Twitter](https://x.com/zebgou)

**Gao Huazuo (高华左)** - Senior Researcher, MLA Architecture Lead
- BS Physics from Peking University (2017)
- Key innovator in Multi-Head Latent Attention (MLA) architecture
- 12+ research papers with 886+ citations
- Major contributor to DeepSeek-R1, DeepSeek-VL2
- Focus on efficiency innovations and attention mechanisms
- Links: [ResearchGate](https://www.researchgate.net/scientific-contributions/Huazuo-Gao-2278776359), [OpenReview](https://openreview.net/profile?id=~Huazuo_Gao1)

**Junxiao Song** - Principal Researcher, Reinforcement Learning Lead
- PhD in Electronic and Computer Engineering from Hong Kong University of Science and Technology (2015)
- BS in Automation from Zhejiang University
- Proposed GRPO (Group Relative Policy Optimization) algorithm used across DeepSeek series
- Designed novel RL pipelines for DeepSeek-R1 eliminating need for supervised fine-tuning
- Links: [Personal Website](https://junxiaosong.github.io/), [GitHub](https://github.com/junxiaosong), [Google Scholar](https://scholar.google.com/citations?user=J95hmyQAAAAJ&hl=en)

**Luo Fuli (罗福利)** - Senior Researcher, NLP & MLA Expert (Age 29)
- Master's from Institute of Computational Linguistics, Peking University
- Previously at Alibaba DAMO Academy ML Lab (developed multilingual VECO, AliceMind)
- Joined DeepSeek in 2022; pivotal role in DeepSeek-V2 development
- Expertise in Multi-Level Attention (MLA) and multilingual NLP
- Published at ACL 2019 and other top NLP conferences
- Nickname: "AI Prodigy" among Chinese tech community
- Links: [ResearchGate](https://www.researchgate.net/scientific-contributions/Fuli-Luo-2142783103)

**Daya Guo (郭大雅)** - Senior Researcher, Code Intelligence & Reasoning
- BS from Sun Yat-sen University
- PhD from Sun Yat-sen University (2018-2023) through joint program with MSRA
- Supervised by Ming Zhou; mentored by Nan Duan and Duyu Tang at MSRA
- Held positions at Rensselaer Polytechnic Institute and Microsoft Research Asia
- Core author on DeepSeekMath, Coder-V2, V2, and R1 papers
- Research focus: NLP, code intelligence, reasoning ability of LLMs
- Links: [Personal Website](https://guoday.github.io/), [Google Scholar](https://scholar.google.com/citations?user=gCG4cPYAAAAJ&hl=en), [Twitter](https://x.com/guodaya)

**Zeng Wangding** - Technical Researcher, MLA Architecture
- Master's candidate at Beijing University of Posts and Telecommunications AI Institute (2021+)
- Key innovator in Multi-Head Latent Attention (MLA) mechanism alongside Gao Huazuo
- Focus on KV cache compression and inference efficiency
- Expertise in hardware-aligned attention mechanisms

**Pan Zizheng (潘子正)** - Senior Researcher, Multimodal Systems Lead
- BS from Harbin Institute of Technology
- MS from University of Adelaide (Australia)
- PhD candidate in Computer Science from Monash University (2021-2024)
- Interned at NVIDIA (Summer 2023); turned down full-time offer to join DeepSeek
- "Key role" in DeepSeek-VL2, DeepSeek-V3, DeepSeek-R1 projects
- Represents growing trend of Chinese AI talent rejecting Silicon Valley for opportunities in China
- Links: [Twitter](https://x.com/zizhpan)

**Supporting Research Infrastructure**

**Zhenda Xie** - Researcher, Model Architecture
- Research Intern at Microsoft Research Asia (2019-2023) under Dr. Han Hu's mentorship
- Core contributor to architectural innovations
- Links: [Personal Website](https://zdaxie.github.io/)

**Team Characteristics**

- **Talent Strategy**: Preference for young academics and recent graduates from top Chinese universities over established senior researchers
- **Academic Excellence**: Strong representation from Tsinghua University, Peking University, and Beihang University
- **International Experience**: Many team members studied or worked internationally (HKUST, University of Adelaide, Monash, MSRA)
- **Microsoft Research Asia Connection**: Strong pipeline of researchers trained at MSRA's Natural Language Computing Group
- **Lean Structure**: ~200 employees total as of January 2025 with ~50-100 core researchers despite frontier-class model development
- **Age Profile**: Mix of young prodigies (age 20s-early 30s) and experienced leaders, with "young geniuses" philosophy
- **No Competition Culture**: Emphasized collaboration over internal competition, unlike many tech companies
- **Patent Holders**: Multiple team members hold 20+ patents each
- **Publication Record**: Prolific academic contributors with 100+ research papers collective

**Organizational Philosophy**

DeepSeek's team culture prioritizes:
- **Technical Autonomy**: Freedom to pursue long-term research without external investor pressure
- **Financial Stability**: Patient capital from High-Flyer hedge fund enables risk-taking
- **Collaborative Environment**: Emphasis on team collaboration rather than individual recognition or competition
- **Merit-Based Hiring**: Ability and potential valued over credentials and experience
- **Open-Source Contribution**: Team members actively contribute to community infrastructure and tools
- **Rapid Iteration**: Ability to move quickly from research concept to production deployment

**Notable Achievements Across Team**

- Reduced training cost from $100M+ industry standard to $5.58M for frontier models (70x efficiency improvement)
- Achieved o1-class reasoning performance with $294K post-training investment (99.8% cost reduction vs industry estimates)
- 88.9% accuracy on formal theorem proving (DeepSeek-Prover-V2)
- State-of-the-art performance on MATH, GSM8K, and coding benchmarks
- Published in top venues: Nature (DeepSeek-R1), ICLR 2024 (ToRA), multiple arxiv technical reports

### 🚀 Model Lineage and Release Timeline

| Release Date | Model | Parameters | Key Features | Open Weights | Links |
|---|---|---|---|---|---|
| Nov 2, 2023 | DeepSeek Coder | 1.3B, 5.7B, 6.7B, 33B | Code generation specialist | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-Coder), [arXiv:2401.14196](https://arxiv.org/abs/2401.14196) |
| Nov 29, 2023 | DeepSeek-LLM | 7B, 67B | Foundation models | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-LLM), [arXiv:2401.02954](https://arxiv.org/abs/2401.02954) |
| Jan 2024 | DeepSeek-MoE | 16B-A2.8B | MoE architecture (Base and Chat variants) | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-MoE), [HF](https://huggingface.co/deepseek-ai/deepseek-moe-16b-base), [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) |
| Feb 2024 | DeepSeek-Math | 7B | Mathematical reasoning (Base, Instruct, RL) | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-Math), [HF](https://huggingface.co/deepseek-ai/deepseek-math-7b-base), [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| Mar 11, 2024 | DeepSeek-VL | 1.3B, 7B | Vision-language understanding (Base & Chat) | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-VL), [HF](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat), [arXiv:2403.05525](https://arxiv.org/abs/2403.05525) |
| May 6, 2024 | DeepSeek-Prover | 7B, 671B | Formal theorem proving in Lean 4 | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5), [arXiv:2405.14333](https://arxiv.org/abs/2405.14333) |
| May 6, 2024 | DeepSeek-V2 | 236B-A21B | MoE, multi-head latent attention (MLA) | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-V2), [HF](https://huggingface.co/deepseek-ai/DeepSeek-V2), [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| May 16, 2024 | DeepSeek-V2-Lite | 15.7B-A2.4B | Lightweight MoE, deployable on single 40G GPU | ✅ | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite), [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) |
| Jun 2024 | DeepSeek-Coder-V2 | 16B-A2.4B, 236B-A21B | Code intelligence, beats GPT-4 Turbo | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-Coder-V2), [HF](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct), [arXiv:2406.11931](https://arxiv.org/abs/2406.11931) |
| Aug 2024 | DeepSeek-Prover-V1.5 | 7B, 671B | Enhanced theorem proving with RMaxTS | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5), [arXiv:2408.08152](https://arxiv.org/abs/2408.08152) |
| Sep 2024 | DeepSeek-V2.5 | 236B-A21B | Fusion of V2-Chat and Coder-V2-Instruct | ✅ | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V2.5) |
| Oct 13, 2024 | Janus | 1.3B | Unified multimodal understanding & generation | ✅ | [GitHub](https://github.com/deepseek-ai/Janus), [HF](https://huggingface.co/deepseek-ai/Janus-1.3B), [arXiv:2410.13848](https://arxiv.org/abs/2410.13848) |
| Nov 13, 2024 | JanusFlow | 1.3B | Unified MLLM with rectified flow for generation | ✅ | [GitHub](https://github.com/deepseek-ai/Janus), [HF](https://huggingface.co/deepseek-ai/JanusFlow-1.3B), [arXiv:2411.07975](https://arxiv.org/abs/2411.07975) |
| Nov 20, 2024 | DeepSeek-R1-Lite-Preview | - | First reasoning model, o1-preview competitor | ✅ | - |
| Dec 13, 2024 | DeepSeek-VL2 | 1.0B-A, 2.8B-A, 4.5B-A | Advanced MoE vision-language (Tiny, Small, Full) | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-VL2), [HF](https://huggingface.co/deepseek-ai/deepseek-vl2), [arXiv:2412.10302](https://arxiv.org/abs/2412.10302) |
| Dec 10, 2024 | DeepSeek-V2.5-1210 | 236B-A21B | Updated V2.5 with improved math/coding | ✅ | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V2.5-1210) |
| Dec 2024 | DeepSeek-V3 | 671B-A37B | MoE, cost-efficient, frontier performance | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-V3), [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3), [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| Jan 20, 2025 | DeepSeek-R1 | 671B-A37B | Reasoning model, o1 competitor | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1), [HF](https://huggingface.co/deepseek-ai/DeepSeek-R1), [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| Jan 20, 2025 | DeepSeek-R1-Zero | 671B-A37B | Pure RL reasoning without distillation | ✅ | [HF](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero), [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| Jan 20, 2025 | DeepSeek-R1-Distill | Qwen: 1.5B, 7B, 14B, 32B; Llama: 8B, 70B | Distilled from R1 to Qwen2.5 and Llama3 | ✅ | [HF-Qwen](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B), [HF-Llama](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B), [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| Jan 20, 2025 | Janus-Pro | 1B, 7B | Advanced unified understanding & generation | ✅ | [GitHub](https://github.com/deepseek-ai/Janus), [HF](https://huggingface.co/deepseek-ai/Janus-Pro-7B), [arXiv:2501.17811](https://arxiv.org/abs/2501.17811) |
| Apr 2025 | DeepSeek-Prover-V2 | 7B, 671B | State-of-the-art formal theorem proving (88.9% MiniF2F) | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-Prover-V2), [arXiv:2504.21801](https://arxiv.org/abs/2504.21801) |
| Mar 24, 2025 | DeepSeek-V3-0324 | 671B-A37B | Enhanced reasoning and coding, MIT license | ✅ | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) |
| Aug 2025 | DeepSeek-V3.1 | 671B-A37B | Hybrid unified thinking mode, 128K context, AIME 93.1%, improved agent capabilities | ✅ | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) |
| Sep 21, 2025 | DeepSeek-V3.1-Terminus | 671B-A37B | Production-ready V3.1 polish, agent stability, language consistency fix, SWE Verified 68.4%, BrowseComp 38.5% | ✅ | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus) |
| Sep 29, 2025 | DeepSeek-V3.2-Exp | 671B-A37B | DeepSeek Sparse Attention (DSA), 50% cost reduction, parity with Terminus, experimental research step | ✅ | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) |
| Oct 20, 2025 | DeepSeek-OCR | - | Vision-text compression for long contexts (97% OCR at 10x) | ✅ | [GitHub](https://github.com/deepseek-ai/DeepSeek-OCR), [HF](https://huggingface.co/deepseek-ai/DeepSeek-OCR), [arXiv:2510.18234](https://arxiv.org/abs/2510.18234) |

### 📄 Research Papers & Techniques (Without Separate Models)

DeepSeek has published additional research papers on techniques and methodologies that enhance their models:

| Publication Date | Paper Title | arXiv | Key Contribution |
|---|---|---|---|
| Jul 2024 | Expert-Specialized Fine-Tuning (ESFT) for Sparse Architectural LLMs | [arXiv:2407.01906](https://arxiv.org/abs/2407.01906) | ESFT technique for optimizing MoE model fine-tuning |
| Feb 2025 | Native Sparse Attention: Hardware-Aligned Sparse Attention | [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) | Dynamic hierarchical sparse attention for long-context efficiency |
| Mar 2025 | A Review of DeepSeek Models' Key Innovative Techniques | [arXiv:2503.11486](https://arxiv.org/abs/2503.11486) | Comprehensive review of DeepSeek's core innovations |
| Apr 2025 | Inference-Time Scaling for Generalist Reward Modeling | [arXiv:2504.02495](https://arxiv.org/abs/2504.02495) | Self-Principled Critique Tuning (SPCT) for scalable reward modeling |

### 📊 Performance and Reception

**Benchmark Performance:**
- DeepSeek-V3: Outperforms Llama 3.1-405B and GPT-4o on many benchmarks
- DeepSeek-R1: Competitive with OpenAI o1 on reasoning benchmarks
- Exceptional performance on coding and mathematics
- Consistently among top performers on major benchmarks

**Market Reception and Global Impact:**
- **Immediate viral success**: V3 release in December 2024 caused international attention
- **US Government Response**: Sparked discussions about AI competitiveness and compute restrictions
- **OpenAI Response**: OpenAI released o1 and made strategic statements about o1 performance
- **Investment Interest**: Major interest from international investors and partnerships
- **Developer Adoption**: Massive download of DeepSeek app (10M+ in first days)
- **Enterprise Integration**: Rapid adoption by Chinese tech companies integrating V3 and R1

**Public Perception:**
- Seen as challenging myth that frontier AI requires $100M+ training budgets
- Positive reception for open-source approach and cost transparency
- Respected for technical excellence and innovation
- Some skepticism about claimed training costs (e.g., analysts questioning R1 cost estimates)

### ⭐ Notable Achievements and Stories

1. **Cost Revolution**: Trained V3 for $5.58M vs $100M+ for comparable Western models, fundamentally challenging assumptions about AI scaling
2. **R1 Breakthrough**: Demonstrated reasoning can be achieved through RL with minimal post-training cost ($294K)
3. **Open-Source Dominance**: V3 became most-used open model globally within weeks of release
4. **International Impact**: Triggered policy discussions in US about AI competitiveness and export controls
5. **Speed-to-Frontier**: From founding (Jul 2023) to frontier-class model (Dec 2024) in 17 months
6. **Independent Success**: Proved that AI breakthrough doesn't require Silicon Valley ecosystem or massive venture funding

### 🌐 Open Source Week and Community Contributions

Beyond model releases, DeepSeek has demonstrated deep commitment to building an open, collaborative AI ecosystem through community-driven initiatives and infrastructure tools.

**DeepSeek Open Source Week (February 24-28, 2025)**

During a dedicated week in late February 2025, DeepSeek released five major open-source tools and infrastructure projects, emphasizing the company philosophy: *"There are no ivory towers–just pure garage energy and community-driven innovation."*

**Major Tool Releases:**

1. **FlashMLA** - Hardware-optimized Multi-Head Latent Attention implementation
   - Achieved 5,000+ GitHub stars within 6 hours of release
   - Enables efficient inference for MLA-based models
   - Production-tested in DeepSeek's own deployments
   - Links: [GitHub](https://github.com/deepseek-ai/FlashMLA)

2. **DeepEP** - Expert Parallelism optimization framework
   - Optimized strategies for distributed MoE training
   - Reduces communication overhead in expert-sharded architectures
   - Battle-tested at scale
   - Links: [GitHub](https://github.com/deepseek-ai/DeepEP), [Website](https://www.deepep.org/en/)

3. **DeepGEMM** - GEMM (General Matrix Multiplication) library
   - Mathematical computation acceleration for tensor operations
   - Hardware-aligned implementations for efficiency
   - Critical component of DeepSeek's training infrastructure
   - Links: [GitHub](https://github.com/deepseek-ai/DeepGEMM)

4. **Optimized Parallelism Strategies** - Training efficiency tools (Two Components)
   - **DualPipe**: Bidirectional pipeline parallelism for computation-communication overlap
     - Links: [GitHub](https://github.com/deepseek-ai/DualPipe)
   - **EPLB**: Expert Parallelism Load Balancer for optimal expert placement
     - Links: [GitHub](https://github.com/deepseek-ai/EPLB), [Website](https://www.deepep.org/en/eplb)

5. **Fire-Flyer File System (3FS)** - Distributed parallel file system
   - Designed for AI workload I/O patterns
   - Optimized for large model training and inference serving
   - Production-proven infrastructure
   - Links: [GitHub](https://github.com/deepseek-ai/3FS)

**Community Ecosystem:**

- **700+ community models** based on DeepSeek-V3 and R1 on Hugging Face
- **5M+ downloads** of community-contributed models and derivatives
- **MIT licensing** across all open-source projects, enabling unrestricted use and modification
- **Developer support infrastructure**: Enhanced platform at platform.deepseek.com with SDKs, documentation, and API guides
- **Community channels**: Discord server with direct company team support and technical discussions

**Beyond Models:**

DeepSeek's community contributions extend beyond model releases:

- **Infrastructure tooling**: Complete stack of training and inference optimization tools
- **Real-world applications**: Practical utilities built on DeepSeek models (e.g., water management and environmental monitoring applications)
- **Educational initiatives**: Workshops, tutorials, and research collaborations with academic institutions
- **Industry engagement**: Hackathons, developer competitions, and partnerships with enterprises
- **Research sharing**: Comprehensive technical documentation and research papers enabling others to build upon DeepSeek's innovations

This approach to open-source contribution reflects DeepSeek's strategic positioning: rather than viewing models as proprietary assets, the company sees frontier AI as fundamentally a collaborative endeavor, with open infrastructure enabling a healthier, more competitive global AI ecosystem.
