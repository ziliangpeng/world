# Prime Intellect INTELLECT-3

INTELLECT-3 is the first 106 billion parameter Mixture-of-Experts model trained through large-scale asynchronous reinforcement learning. Released in November 2025, it demonstrates that RL training can scale to 100B+ parameter models on 512 GPUs with efficient async orchestration, achieving state-of-the-art performance across math, code, and reasoning benchmarks.

**Official Resources**:
- 📄 [Technical Report](https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf) - Complete training methodology and results
- 🤗 [Model on HuggingFace](https://huggingface.co/PrimeIntellect/INTELLECT-3) - Weights and documentation
- 🔧 [Prime Intellect RL Stack](../tools/prime-intellect-rl-stack.md) - Infrastructure documentation

---

## Overview

| Aspect | Details |
|--------|---------|
| **Organization** | Prime Intellect |
| **Release Date** | November 26, 2025 |
| **Parameters** | 106 billion total, 12 billion active (A12B) |
| **Base Model** | GLM-4.5-Air-Base (Zhipu AI) |
| **Training Type** | Supervised Fine-Tuning + Large-scale Reinforcement Learning |
| **Infrastructure** | 512 NVIDIA H200 GPUs across 64 nodes |
| **Training Duration** | ~2 months |
| **Framework** | prime-rl (async RL at scale) |

**Key Innovation**: First demonstration of large-scale async RL training for 100B+ MoE models, introducing Prime Sandboxes for millisecond-latency code execution and the prime-rl framework for disaggregated training at scale.

---

## Model Specifications

### Architecture

INTELLECT-3 is built on GLM-4.5-Air-Base with Mixture-of-Experts architecture:

```
Model: INTELLECT-3
├── Total Parameters: 106 billion
├── Active Parameters: 12 billion (A12B)
├── Architecture: Mixture-of-Experts (MoE)
├── Base: GLM-4.5-Air-Base
├── Post-training: SFT → RL
└── Formats: BF16, FP8 (quantized)
```

### Base Model: GLM-4.5-Air

- **Organization**: Zhipu AI
- **Design**: Streamlined 106B total, 12B active parameters
- **MoE Structure**: Efficient sparse activation
- **Original Purpose**: Agentic, Reasoning, and Coding (ARC) tasks

### Training Stages

**Stage 1: Supervised Fine-Tuning (SFT)**
- Initial instruction alignment
- Diverse task coverage across math, code, science
- Foundation for RL phase

**Stage 2: Reinforcement Learning (RL)**
- Large-scale RL using prime-rl framework
- 500+ training environments from Environments Hub
- Verifiable rewards (math: symbolic, code: test execution)

### Training Environments (500+ Tasks)

| Domain | Task Types | Verification |
|--------|------------|--------------|
| **Mathematics** | Competition problems, theorem proving | Symbolic answer matching |
| **Code** | Software engineering, debugging | Test case execution |
| **Science** | Physics, chemistry, biology problems | Domain-specific verifiers |
| **Logic** | Reasoning chains, formal logic | Logical consistency |
| **Deep Research** | Multi-step analysis, literature synthesis | Quality scoring |
| **Agentic** | Long-horizon tasks, tool use | Goal completion |

### Deployment Options

| Format | Hardware | Use Case |
|--------|----------|----------|
| **BF16** | 2x H200 GPUs (tensor parallelism) | Maximum accuracy |
| **FP8** | 1x H200 GPU | Efficient deployment |

Both formats compatible with vLLM inference engine.

---

## Training Infrastructure

INTELLECT-3 was trained using Prime Intellect's RL infrastructure stack:

- **[prime-rl](../tools/prime-intellect-rl-stack.md#prime-rl-framework)**: Async RL training framework (512 GPUs, disaggregated trainer/inference/orchestrator)
- **[Prime Sandboxes](../tools/prime-intellect-rl-stack.md#prime-sandboxes)**: High-performance code execution (millisecond latency, thousands concurrent)
- **[Verifiers Library](../tools/prime-intellect-rl-stack.md#verifiers-library)**: Modular environment toolkit
- **[Environments Hub](../tools/prime-intellect-rl-stack.md#environments-hub)**: 500+ training tasks across math, code, science, logic, research, agentic

For detailed technical documentation of these infrastructure components, see **[Prime Intellect RL Stack](../tools/prime-intellect-rl-stack.md)**.

**INTELLECT-3 Training Configuration**:
- **GPUs**: 512 NVIDIA H200 across 64 nodes
- **Stages**: Two-stage post-training (SFT → RL)
- **Duration**: ~2 months total
- **Environments**: 500+ tasks from Environments Hub
- **Framework**: prime-rl (async orchestration)
- **Execution**: Prime Sandboxes (millisecond-latency code verification)

---

## Training Methodology

### Two-Stage Post-Training

INTELLECT-3 training followed a structured approach:

```
Stage 1: Supervised Fine-Tuning (SFT)
├── Goal: Instruction alignment
├── Data: Diverse task demonstrations
├── Duration: ~2 weeks
└── Outcome: Base instruction-following capability

        ↓

Stage 2: Reinforcement Learning (RL)
├── Goal: Optimize for verifiable outcomes
├── Environments: 500+ tasks from Hub
├── Framework: prime-rl (async orchestration)
├── Infrastructure: 512 H200 GPUs, 64 nodes
├── Duration: ~6 weeks
└── Outcome: INTELLECT-3 (state-of-the-art reasoning)
```

### Environment Diversity

Training spanned 6 major domains:

**1. Mathematics (150+ environments)**
- Competition problems (AMC, AIME, IMO)
- Theorem proving
- Algebra, calculus, number theory
- Verification: Symbolic answer matching

**2. Code (200+ environments)**
- Algorithmic programming (LeetCode, Codeforces)
- Software engineering (SWE-bench)
- Code synthesis (HumanEval, APPS)
- Verification: Test case execution in Prime Sandboxes

**3. Science (50+ environments)**
- Physics problem solving
- Chemistry calculations
- Biology reasoning
- Verification: Domain-specific verifiers

**4. Logic (40+ environments)**
- Formal reasoning chains
- Logical puzzles
- Deductive inference
- Verification: Logical consistency checks

**5. Deep Research (30+ environments)**
- Multi-step analysis
- Literature synthesis
- Hypothesis generation
- Verification: Quality scoring models

**6. Agentic Tasks (30+ environments)**
- Tool use (calculators, APIs, databases)
- Multi-step planning
- Web navigation
- Verification: Goal completion

### Async-Only Methodology

**Off-policy RL advantages**:

| Challenge | On-Policy (e.g., PPO) | Off-Policy (PRIME-RL) |
|-----------|----------------------|----------------------|
| **Rollout freshness** | Must use latest policy | Stale rollouts OK |
| **Sync barriers** | Training waits for rollouts | Training continues async |
| **Long-horizon tasks** | Timeouts block batch | Old rollouts still useful |
| **GPU efficiency** | Idle time during inference | Trainer always busy |

**Key insight**: Math and coding have verifiable outcomes, making off-policy stable even with delayed rollouts.

### Infrastructure Details

**Cluster Configuration**:
- **GPUs**: 512 NVIDIA H200
- **Nodes**: 64 interconnected
- **GPUs per node**: 8
- **Networking**: High-bandwidth interconnect
- **Duration**: ~2 months (SFT + RL)

**Parallelism Strategy**:
- **Trainer**: FSDP2 across multiple nodes
- **Inference**: vLLM distributed across separate nodes
- **Orchestrator**: Lightweight CPU coordination
- **Sandboxes**: Hundreds per node, thousands concurrent

---

## Performance

### Benchmark Results

INTELLECT-3 achieves state-of-the-art performance for its size:

| Benchmark | INTELLECT-3 | GLM-4.5 (355B) | DeepSeek v3.2 (355B) | Difference |
|-----------|-------------|----------------|---------------------|------------|
| **MATH-500** | **98.1%** | 97.0% | - | **+1.1 pp vs GLM** |
| **AIME24** | **90.8%** | 91.0% | 88.1% | **+2.7 pp vs DeepSeek** |
| **AIME25** | **88.0%** | - | 84.7% | **+3.3 pp vs DeepSeek** |
| **GPQA** | 74.4% | - | **81.4%** | -7.0 pp vs DeepSeek |
| **MMLU-Pro** | 81.9% | - | **84.6%** | -2.7 pp vs DeepSeek |

### Key Strengths

**1. Mathematical Reasoning**
- MATH-500: 98.1% (near-perfect on competition math)
- AIME24: 90.8% (American Invitational Mathematics Examination)
- AIME25: 88.0% (latest AIME competition)

**Context**: AIME is one of the hardest high school math competitions. 90.8% and 88.0% scores indicate advanced mathematical reasoning capability.

**2. Efficiency**
- Outperforms DeepSeek v3.2 (355B params) on math/reasoning
- INTELLECT-3 has only **106B total, 12B active** parameters
- **~3x smaller** yet competitive or better on reasoning tasks

**3. Deployment**
- FP8 quantization enables **single H200 GPU** deployment
- No accuracy loss for most applications
- Cost-effective inference

### Performance Analysis

**Why strong on math/reasoning?**
- Large-scale RL on 500+ diverse environments
- Verifiable rewards enable stable off-policy learning
- Prime Sandboxes allow massive concurrent verification
- Async training maximizes GPU utilization

**Trade-offs**:
- Slightly lower on general knowledge (GPQA, MMLU-Pro) vs. larger models
- Optimized for tasks with verifiable outcomes (math, code)
- MoE architecture: 12B active params at inference (efficiency)

---

## Technical Innovations Summary

INTELLECT-3 introduced four major infrastructure innovations (see [Prime Intellect RL Stack](../tools/prime-intellect-rl-stack.md) for full technical details):

| Innovation | Problem Solved | Impact |
|------------|----------------|--------|
| **[prime-rl](../tools/prime-intellect-rl-stack.md#prime-rl-framework)** | RL doesn't scale beyond small clusters | Async RL to 512+ GPUs, 1000+ capable |
| **[Prime Sandboxes](../tools/prime-intellect-rl-stack.md#prime-sandboxes)** | Code execution bottlenecks training | Millisecond latency, thousands concurrent |
| **[Verifiers Library](../tools/prime-intellect-rl-stack.md#verifiers-library)** | Building RL environments is hard | Modular toolkit, easy environment creation |
| **[Environments Hub](../tools/prime-intellect-rl-stack.md#environments-hub)** | No shared RL task repository | 500+ tasks, community contributions |

**Quick Summary**:
- **prime-rl**: Disaggregated trainer/inference/orchestrator, async-only methodology, 1000+ GPU scale
- **Prime Sandboxes**: Direct Rust-to-pod execution, sub-10s provisioning, millisecond latency
- **Verifiers**: Modular components, CPU development + GPU training, Prime Sandboxes backend
- **Environments Hub**: 500+ community tasks, versioned, HuggingFace integration

See **[Prime Intellect RL Stack documentation](../tools/prime-intellect-rl-stack.md)** for architecture diagrams, code examples, and integration guides.

---

## Significance

### Scaling RL to 100B+ MoE Models

**Achievement**: First large-scale async RL training of 106B MoE model on 512 GPUs.

**Prior work**:
- OpenAI (GPT-4, o1): RL training methods proprietary
- DeepSeek v3: Some RL, but methodology unclear
- Most open models: Supervised fine-tuning only, no large-scale RL

**INTELLECT-3**: Demonstrates that RL can scale to frontier model sizes with open-source infrastructure.

### Async Orchestration Enables Long-Horizon Tasks

**Challenge**: Agentic tasks can take minutes to complete (web navigation, multi-step research).

**Solution**: Async-only methodology means slow rollouts don't block training.

**Impact**: Enables RL on tasks previously impractical (timeouts would stall synchronous training).

### Open-Source Ecosystem

**Released under MIT/Apache 2.0**:
- INTELLECT-3 model weights (BF16, FP8)
- prime-rl training framework
- Verifiers library
- Environments Hub (500+ tasks)
- Technical report

**Philosophy**: Democratize large-scale RL training, not just inference.

### Evolution of INTELLECT Series

| Model | Focus | Innovation | Scale |
|-------|-------|------------|-------|
| **INTELLECT-1** (Nov 2024) | Decentralized pre-training | DiLoCo (distributed local SGD) | 10B, 112 H100s, 3 continents |
| **INTELLECT-2** (May 2025) | Decentralized RL | TOPLOC, SHARDCAST (untrusted workers) | 32B, distributed |
| **INTELLECT-3** (Nov 2025) | **Centralized large-scale RL** | **prime-rl, Prime Sandboxes** | **106B, 512 H200s** |

**Trajectory**: From distributed pre-training → distributed RL → large-scale centralized RL infrastructure.

**Key shift**: INTELLECT-3 focuses on **scaling RL infrastructure** (async, high-performance sandboxes) rather than decentralization, achieving frontier-level performance.

### Democratizing Large-Scale RL Training

**Barriers removed** (see [Prime Intellect RL Stack](../tools/prime-intellect-rl-stack.md)):
- ✅ Open-source RL framework ([prime-rl](../tools/prime-intellect-rl-stack.md#prime-rl-framework))
- ✅ High-performance execution layer ([Prime Sandboxes](../tools/prime-intellect-rl-stack.md#prime-sandboxes))
- ✅ Environment toolkit ([Verifiers Library](../tools/prime-intellect-rl-stack.md#verifiers-library))
- ✅ Shared task repository ([Environments Hub](../tools/prime-intellect-rl-stack.md#environments-hub))
- ✅ Complete training recipe (technical report)

**Future**: Any organization with GPU access can replicate large-scale RL training using the open-source Prime Intellect RL Stack, not just frontier labs.

---

## Future Roadmap

Prime Intellect has announced plans for the next models in the INTELLECT series:

### INTELLECT-4 (Planned)
**Focus**: o1/r1-style reasoning capabilities
- Deep chain-of-thought reasoning
- Test-time compute scaling
- Inspired by OpenAI o1 and DeepSeek R1 approaches
- Advanced reasoning verification

### INTELLECT-5 (Planned)
**Focus**: Large-scale pre-training
- Return to decentralized pre-training at scale
- Building on INTELLECT-1's DiLoCo foundations
- Frontier-scale model trained via distributed collaboration
- Demonstrating open pre-training at 100B+ parameters

---

## Sources

### Documentation
- **[Prime Intellect RL Stack](../tools/prime-intellect-rl-stack.md)** - Complete infrastructure documentation (prime-rl, Prime Sandboxes, Verifiers, Environments Hub)

### Papers & Reports
- [INTELLECT-3 Technical Report](https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf) - Complete training methodology
- [INTELLECT-1 Technical Report](https://arxiv.org/abs/2412.01152) - Decentralized pre-training
- [INTELLECT-2 Technical Report](https://arxiv.org/abs/2505.07291) - Decentralized RL
- [GLM-4.5 Paper](https://arxiv.org/abs/2508.06471) - Base model architecture

### Blog Posts
- [INTELLECT-3: A 100B+ MoE trained with large-scale RL](https://www.primeintellect.ai/blog/intellect-3) - Official release announcement
- [Environments Hub: A Community Hub To Scale RL To Open AGI](https://www.primeintellect.ai/blog/environments) - Environment ecosystem
- [INTELLECT-2 Release](https://www.primeintellect.ai/blog/intellect-2-release) - Decentralized RL predecessor
- [INTELLECT-1 Launch](https://www.primeintellect.ai/blog/intellect-1) - Decentralized pre-training predecessor

### Model & Code
- [INTELLECT-3 on HuggingFace](https://huggingface.co/PrimeIntellect/INTELLECT-3) - Model weights, documentation
- [prime-rl Framework](https://github.com/PrimeIntellect-ai/prime-rl) - Async RL training at scale
- [Verifiers Library](https://github.com/PrimeIntellect-ai/verifiers) - Environment toolkit
- [Prime SDK](https://github.com/PrimeIntellect-ai/prime) - Official CLI and Python SDK

### Infrastructure & Deployment
- [INTELLECT-3 on OpenRouter](https://openrouter.ai/prime-intellect/intellect-3/api) - API access
- [INTELLECT-3 on Vercel AI Gateway](https://vercel.com/changelog/intellect-3-model-from-prime-intellect-ai-available-on-the-vercel-ai-gateway) - Deployment option
- [Bitget News Coverage](https://www.bitget.com/news/detail/12560605085055) - Launch announcement
