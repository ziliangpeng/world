# DeepSeek - Deep Dive

## 3. DeepSeek

### 🏢 Company Overview

**DeepSeek** has emerged as one of the most significant players in the LLM landscape despite being founded in 2023. The company is known for radical cost efficiency, powerful open-source models, and technical innovations that challenged assumptions about training costs and efficiency. DeepSeek-V3 (671B parameters) and DeepSeek-R1 (reasoning model) have attracted significant international attention for achieving frontier performance at a fraction of the computational cost of Western models.

### 📚 Founding Story and History

DeepSeek was founded in **July 2023** by **Liang Wenfeng**, the co-founder of High-Flyer, a Chinese hedge fund. The unusual pedigree of a startup funded by a hedge fund (rather than VCs) meant DeepSeek had patient capital and independence from typical startup pressures. Liang Wenfeng, having built successful trading systems and quantitative algorithms at High-Flyer, brought a deep technical and computational mindset to AI development.

**Key Timeline:**
- **Jul 2023**: DeepSeek founded
- **Nov 2, 2023**: DeepSeek Coder released, establishing the company's technical credibility
- **Nov 29, 2023**: DeepSeek-LLM series launched
- **Dec 2024**: DeepSeek-V3 released, immediately causing international attention for cost-efficiency claims
- **Jan 2025**: DeepSeek-R1 launched, competing directly with OpenAI's o1
- **Jan 2025**: DeepSeek chatbot launched with over 10M downloads within days

DeepSeek's rapid ascent from unknown startup to industry-shaping player occurred in under 18 months, driven by technically superior models and radical cost transparency.

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

DeepSeek introduced several technical innovations that contributed to efficiency breakthroughs:

**Multi-Head Latent Attention (MLA):**
- Reduces memory footprint of attention operations while maintaining performance
- Key contributor to efficiency gains in V3
- Reduces KV cache size requirements

**Efficient Mixture of Experts:**
- Fine-grained MoE architecture in V3 (671B total, 37B activated)
- Dynamic routing of tokens to specialized experts
- Significantly reduces computational cost per inference

**Partial 8-bit Native Training:**
- Uses 8-bit precision for parts of training without full precision requirements
- Effectively doubles model size that fits in memory
- Co-designed with hardware and frameworks

**Cross-Node Optimization:**
- Eliminated communication bottlenecks in multi-node MoE training
- Achieved near-full computation-communication overlap
- Hardware-software co-design for efficiency

**Reinforcement Learning for Reasoning:**
- R1 uses RL to achieve reasoning capabilities with low post-training cost
- Trained on $294K (post V3-Base training)
- Competes with o1 despite massive cost difference

### 👥 Team Background

DeepSeek's leadership and team include:
- **Liang Wenfeng**: Founder and CEO, background in quantitative trading and algorithms
- Core team trained at High-Flyer
- Hiring from top AI labs (Tsinghua, Peking University, etc.)
- Relatively small team (50-100 core researchers) compared to Western AI companies
- Talent attracted by technical autonomy and financial stability

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
| Aug 2025 | DeepSeek-V3.1 | 671B-A37B | Improvements over V3 | ✅ | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3.1) |
| Sep 22, 2025 | DeepSeek-V3.1-Terminus | 671B-A37B | Updated V3.1 version | ✅ | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus) |
| Sep 29, 2025 | DeepSeek-V3.2-Exp | 671B-A37B | Experimental version | ✅ | [HF](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) |
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
