# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

## Overview

**DeepSeek-R1** is a breakthrough open-source reasoning model released on **January 20, 2025** under the **MIT license**. It rivals OpenAI's o1 in performance while being trained at a fraction of the cost (~$5.58M vs $6B+). The key innovation is demonstrating that advanced reasoning capabilities can emerge purely from reinforcement learning without supervised fine-tuning.

### Model Information

| **Attribute** | **Details** |
|---------------|-------------|
| **Developer** | DeepSeek AI |
| **Release Date** | January 20, 2025 |
| **Model Type** | Reasoning-Oriented Large Language Model |
| **Base Architecture** | DeepSeek-V3 (671B parameters, MoE) |
| **Total Parameters** | 671B |
| **Activated Parameters** | 37B per token (~5% activation) |
| **Context Length** | 128K tokens |
| **License** | MIT (fully permissive, allows commercial use and distillation) |
| **Primary Sources** | [ArXiv 2501.12948](https://arxiv.org/abs/2501.12948), [GitHub](https://github.com/deepseek-ai/DeepSeek-R1), [Nature Publication](https://www.nature.com/articles/s41586-025-09422-z) |

### Notable Achievements

1. **Matches OpenAI o1**: 79.8% on AIME 2024, 97.3% on MATH-500
2. **Pure RL Reasoning Emergence**: First to validate reasoning from RL alone
3. **1,000× Cheaper Training**: $5.58M vs $6B+ for o1
4. **90-95% Cheaper Inference**: $0.55/M input tokens vs $15/M for o1
5. **Fully Open Source**: MIT license with model weights, code, and distilled variants

---

## Architecture Specifications

### Base Model: DeepSeek-V3

| **Parameter** | **Value** |
|---------------|-----------|
| **Foundation** | DeepSeek-V3-Base |
| **Total Parameters** | 671B (Mixture-of-Experts) |
| **Activated Parameters** | 37B per token (~5.5% activation rate) |
| **Context Length** | 128K tokens |
| **Precision** | BFloat16 |

### Multi-head Latent Attention (MLA)

| **Feature** | **Specification** |
|-------------|-------------------|
| **Mechanism** | Low-rank joint compression for attention keys and values |
| **Benefit** | Reduces Key-Value cache for efficient inference |
| **Impact** | Enables economical training at 671B scale |

### Mixture-of-Experts (DeepSeekMoE)

| **Component** | **Specification** |
|---------------|------------------|
| **Total Experts** | 257 (256 routed + 1 shared) |
| **Active Experts per Token** | 9 (8 routed + 1 shared) |
| **Activation Rate** | 5.5% of total parameters |
| **Load Balancing** | Auxiliary-loss-free strategy |
| **Training Objective** | Multi-token prediction |

**Pre-training Foundation**:
- 14.8 trillion diverse, high-quality tokens
- State-of-the-art base model capabilities
- Foundation for reasoning emergence

---

## Training Methodology

### Two Parallel Approaches

#### 1. DeepSeek-R1-Zero (Pure Reinforcement Learning)

**Revolutionary Approach**: Large-scale RL without ANY supervised fine-tuning

**Base**: DeepSeek-V3-Base + GRPO (Group Relative Policy Optimization)

**Key Finding**: Reasoning capabilities emerge naturally from RL alone

**Training Configuration**:
- **Iterations**: Tens of thousands (~8,000 gradient steps)
- **GPU Hours**: 2.788M H800 GPU hours
- **Total Cost**: ~$5.58 million
- **Training Duration**: Approximately 1.5-2 months

**Emergent Behaviors** (discovered without explicit training):
1. **Chain-of-Thought Reasoning**: Hundreds to thousands of tokens
2. **Self-Verification**: Checks own outputs for consistency and correctness
3. **Reflection**: Identifies and corrects errors in reasoning process
4. **Error Correction**: Refines outputs iteratively
5. **"Aha Moments"**: Steps back, spots mistakes, corrects itself mid-reasoning
6. **Dynamic Strategy Adaptation**: Explores alternative approaches during problem-solving

**Challenges**:
- Poor readability (verbose, unstructured)
- Language mixing (Chinese/English)
- Endless repetition in some cases
- Lack of markdown formatting
- Outputs unsuitable for direct user consumption

**Performance**:
- **AIME 2024**: 15.6% (base) → 71.0% (RL trained) → 86.7% (majority voting)
- Demonstrates that reasoning emerges purely from RL incentives

#### 2. DeepSeek-R1 (Multi-Stage Training)

Addresses R1-Zero's limitations through structured training:

**Stage 1: Cold Start (Initial SFT)**

**Purpose**: Establish readable reasoning format

**Data**: Small dataset of long Chain-of-Thought examples
- Size: ~10K-15K examples
- Quality: High-quality, reader-friendly reasoning traces
- Format: Includes summary at end of responses
- Filtering: Removes non-reader-friendly outputs

**Training**:
- Fine-tunes DeepSeek-V3-Base
- Prevents early training instability
- Establishes proper formatting conventions
- Short training phase (relative to RL)

**Stage 2: Reasoning-Oriented RL**

**Purpose**: Develop advanced reasoning capabilities

**Framework**: GRPO (Group Relative Policy Optimization)

**Focus**: Reasoning-intensive tasks
- Mathematics (MATH, AIME)
- Coding (Codeforces, SWE-bench)
- Scientific reasoning (GPQA)

**Training Configuration**:
- Iterations: ~4,000-8,000 RL iterations
- Batch size (B): 1,024
- Group size (G): 64
- Average sequence length (L): ~4,000 tokens
- Total gradient steps: ~8,000

**Reward System**:
1. **Accuracy Rewards**: Correctness of final answer
2. **Format Rewards**: Proper structuring of `<think>` and answer tags
3. **Language Consistency Rewards**: Ensures reasoning in same language as query

**Stage 3: Rejection Sampling + Second SFT**

**Purpose**: Combine RL insights with supervised learning

**Data Creation**:
- Rejection sampling from Stage 2 RL checkpoint
- 600,000 reasoning data samples (R1 generated)
- 200,000 non-reasoning SFT data samples (from DeepSeek-V3)

**Domains**:
- Writing
- Factual QA
- Self-cognition
- General helpfulness

**Training**:
- Standard supervised fine-tuning
- Combines best of RL exploration with supervised stability
- Enhances readability and formatting

**Stage 4: Comprehensive Alignment RL**

**Purpose**: Final polishing for human preferences

**Focus**:
- Helpfulness
- Harmlessness
- Safety
- User preference alignment

**Training**:
- Secondary RL phase
- Human preference data (RLHF)
- Model-as-a-judge rewards
- Final calibration

---

## Reinforcement Learning: GRPO Algorithm

### Group Relative Policy Optimization (GRPO)

**Key Innovation**: Eliminates critic model (typically same size as policy model)

**Traditional PPO** (used in ChatGPT):
- Requires separate critic model (value network)
- Critic model same size as policy model
- Doubles memory and compute requirements
- More complex to train

**GRPO Approach**:
- Estimates baseline from group scores instead of learned value function
- Compares outputs to past attempts (relative grading vs binary right/wrong)
- Derives relative rewards from multiple sampled responses
- Enables efficient and stable policy updates

**Benefits**:
- **~50% compute reduction** vs PPO
- **Up to 18× more cost-efficient** in certain scenarios
- More stable training (no critic model instability)
- Simpler implementation

**How GRPO Works**:

1. **Sample Multiple Outputs**: Generate G=64 responses for each problem
2. **Evaluate All Outputs**: Calculate rewards for all responses
3. **Compute Group Statistics**: Mean and standard deviation of rewards
4. **Relative Advantages**: Advantage = (reward - group_mean) / group_std
5. **Policy Update**: Update based on relative performance within group

**Mathematical Formula**:
```
Advantage(i) = (Reward(i) - mean(Rewards)) / std(Rewards)
Policy_Loss = -mean(Advantage(i) * log_prob(action_i))
```

**Training Parameters**:
- **Batch size (B)**: 1,024 problems
- **Group size (G)**: 64 responses per problem
- **Total samples per step**: 65,536 (B × G)
- **Sequence length (L)**: ~4,000 tokens average
- **Total gradient steps**: ~8,000

---

## Reward System Design

### DeepSeek-R1-Zero (Rule-Based)

**Two Components**:

1. **Accuracy Rewards**:
   - Evaluates correctness of final answer
   - Binary correct/incorrect for math problems
   - Execution success for code
   - No neural reward model

2. **Format Rewards**:
   - Enforces proper structuring of think/answer tags
   - Ensures stable inference
   - Prevents format violations

**No Neural Reward Model**: Used rule-based rewards instead of trained neural network

### DeepSeek-R1 (Three Components)

**Enhanced Reward System**:

1. **Accuracy Rewards** (inherited from R1-Zero):
   - Rule-based evaluation for clear right/wrong answers
   - Math: correct answer vs ground truth
   - Code: execution success and test case passing

2. **Format Rewards** (inherited from R1-Zero):
   - Correct use of `<think>` tags
   - Proper answer formatting
   - Ensures stable inference

3. **Language Consistency Rewards** (NEW):
   - **Calculation**: Proportion of target language words in CoT
   - **Bonus**: If answer language 100% matches question language
   - **Purpose**: Prevents language mixing (Chinese/English)
   - **Trade-off**: Slight performance degradation (~1-2%) but better human preferences

**Model-as-a-Judge**:
- Uses DeepSeek-V3 or other LLMs to evaluate subjective quality
- Helpful for open-ended tasks (writing, QA)
- Complements rule-based rewards

---

## Reasoning Capabilities

### Chain-of-Thought (CoT) Implementation

**Generation**:
- Enforced with `<think>\n` tag at response start
- Maximum generation length: 32,768 tokens
- Extends thinking for difficult problems (hundreds to thousands of tokens)
- Leverages extended test-time computation

**Structure**:
```
<think>
[Reasoning process here - can be very long]
- Problem analysis
- Strategy exploration
- Calculations and derivations
- Self-verification
- Error correction
</think>

[Final answer here]
```

### Emergent Reasoning Behaviors

**1. Self-Verification**
- Checks own outputs for consistency and correctness
- Emerges naturally from RL, not explicitly trained
- Re-evaluates steps if early approach seems likely to fail
- Example: "Let me verify this calculation: ..."

**2. Reflection & Error Correction**
- Identifies and corrects errors in reasoning process
- Refines outputs iteratively
- Shows "aha moments" - steps back, spots mistakes, corrects itself
- Reduces hallucinations through self-checking
- Example: "Wait, I made an error in step 3. Let me recalculate..."

**3. Dynamic Strategy Adaptation**
- Explores alternative approaches during problem-solving
- Adjusts reasoning depth based on problem complexity
- Develops more sophisticated behaviors with increased test-time computation
- Example: "This approach isn't working. Let me try a different method..."

**4. Extended Test-Time Computation**
- Longer reasoning for harder problems
- Can generate thousands of tokens of reasoning
- Trades compute at inference time for better accuracy
- Similar to AlphaGo's tree search at test time

### Reasoning Quality

**Characteristics**:
- **Depth**: Can reason through hundreds of steps
- **Breadth**: Explores multiple solution paths
- **Accuracy**: High correctness rate on math and coding
- **Transparency**: All reasoning is visible in `<think>` tags
- **Self-Correcting**: Catches and fixes own mistakes

---

## Benchmark Performance

### Mathematics Benchmarks

| **Benchmark** | **DeepSeek-R1** | **OpenAI o1-1217** | **Difference** |
|---------------|-----------------|-------------------|----------------|
| **AIME 2024 (Pass@1)** | **79.8%** | 79.2% | +0.6% |
| **AIME 2024 (Maj Vote)** | **86.7%** | ~86% (o1-0912) | Matches |
| **MATH-500 (Pass@1)** | **97.3%** | 96.4% | +0.9% |
| **MATH-500 (Maj Vote)** | **97.9%** | — | — |

**R1-Zero Progression on AIME 2024**:
- DeepSeek-V3-Base: 15.6%
- R1-Zero (RL trained): 71.0% (+55.4%)
- R1-Zero (majority voting): 86.7% (+71.1%)

**Key Insight**: Pure RL alone improved AIME 2024 score by 55.4 percentage points.

### Coding Benchmarks

| **Benchmark** | **DeepSeek-R1** | **OpenAI o1-1217** | **Difference** |
|---------------|-----------------|-------------------|----------------|
| **Codeforces** | 96.3% (2029 rating) | 96.6% | -0.3% (nearly identical) |
| **SWE-bench Verified** | **49.2%** | 48.9% | +0.3% |
| **LiveCodeBench** | **50.4%** | — | — |

**Codeforces Rating**: 2029 (equivalent to Expert level, top 7% of competitive programmers)

### General Knowledge & Reasoning

| **Benchmark** | **DeepSeek-R1** | **OpenAI o1-1217** | **Gap** |
|---------------|-----------------|-------------------|---------|
| **MMLU** | 90.8% | **91.8%** | -1.0% |
| **MMLU-Pro** | 84.0% | — | — |
| **GPQA Diamond** | 71.5% | **75.7%** | -4.2% |
| **BBH** | 89.3% | — | — |

**Analysis**:
- **Excels**: Mathematics (AIME, MATH), Software Engineering (SWE-bench)
- **Competitive**: Coding contests (Codeforces nearly identical to o1)
- **Slightly Behind**: General knowledge (MMLU -1.0%), Factual reasoning (GPQA -4.2%)

### Summary by Category

**R1 > o1**:
- MATH-500: +0.9%
- AIME 2024: +0.6%
- SWE-bench Verified: +0.3%

**R1 ≈ o1**:
- Codeforces: -0.3% (essentially tied)
- MMLU: -1.0% (minor gap)

**R1 < o1**:
- GPQA Diamond: -4.2% (notable gap in factual scientific reasoning)

---

## Model Variants

### Full Model

**DeepSeek-R1** (671B total / 37B activated)
- **Performance**: Comparable to OpenAI o1-1217
- **HuggingFace**: `deepseek-ai/DeepSeek-R1`
- **Use Case**: Maximum reasoning performance

### Distilled Models (Knowledge Distillation from R1)

**Distillation Process**:
- Fine-tuned with 800K samples curated from DeepSeek-R1
- Reasoning patterns transferred to smaller models
- Achieves better performance than RL on small models directly

#### Qwen-Based (derived from Qwen2.5)

| **Model** | **Params** | **AIME 2024** | **MATH-500** | **vs Baseline** |
|-----------|------------|---------------|--------------|-----------------|
| **R1-Distill-Qwen-1.5B** | 1.5B | — | — | Smallest variant |
| **R1-Distill-Qwen-7B** | 7B | 55.5% | — | Surpasses QwQ-32B-Preview |
| **R1-Distill-Qwen-14B** | 14B | — | — | — |
| **R1-Distill-Qwen-32B** | 32B | **72.6%** | **94.3%** | **Outperforms o1-mini** |

**HuggingFace**:
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

#### Llama-Based (derived from Llama 3.1/3.3)

| **Model** | **Base** | **Params** | **AIME 2024** | **MATH-500** |
|-----------|----------|------------|---------------|--------------|
| **R1-Distill-Llama-8B** | Llama3.1-8B-Base | 8B | — | — |
| **R1-Distill-Llama-70B** | Llama3.3-70B-Instruct | 70B | 70.0% | 94.5% |

**HuggingFace**:
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`

**Key Finding**:
- **R1-Distill-Qwen-32B outperforms OpenAI o1-mini** on AIME 2024 (72.6% vs ~60%)
- Distillation transfers reasoning patterns effectively
- Smaller models gain significant reasoning boost

---

## Training Infrastructure & Cost

### Training Costs

| **Model** | **Training Cost** | **Cost per Token** |
|-----------|------------------|--------------------|
| **DeepSeek-R1** | ~$5.58M | ~$0.377/B tokens |
| **OpenAI o1** | ~$6B+ (estimated) | — |
| **Cost Difference** | **>1,000× cheaper** | — |

**Breakdown**:
- GPU hours: 2.788 million H800 GPU hours
- Hardware: NVIDIA H800 (restricted H100 variant, 80GB HBM3)
- Training duration: ~1.5-2 months
- Pre-training tokens: 14.8 trillion (DeepSeek-V3 base)

### Compute Efficiency Comparison

| **Organization** | **Similar-Scale Model** | **GPU Hours** | **Efficiency** |
|------------------|------------------------|---------------|----------------|
| **DeepSeek** | R1 (671B) | 2.78M | Baseline |
| **Meta** | Similar scale | 30.8M | **11× less efficient** |

**Key Insight**: DeepSeek achieves 11× better compute efficiency than Meta for similar-scale models.

### Inference Costs (API Pricing)

| **Provider** | **Input (per 1M tokens)** | **Output (per 1M tokens)** | **Savings** |
|--------------|---------------------------|---------------------------|-------------|
| **DeepSeek-R1** | $0.55 | $2.19 | Baseline |
| **OpenAI o1** | $15 | $60 | **90-95% cheaper** |
| **OpenAI o1-mini** | $3 | $12 | **82-82% cheaper** |

**Example Calculation** (1M input, 1M output):
- **DeepSeek-R1**: $0.55 + $2.19 = **$2.74**
- **OpenAI o1**: $15 + $60 = **$75**
- **Savings**: **96.3%**

---

## API & Usage

### Availability

**Released**: January 20, 2025
**License**: MIT (permissive, allows commercial use, distillation, modification)

**Access Points**:
- **DeepSeek Official API** (OpenAI-compatible)
- **Together.ai** (serverless + dedicated GPUs)
- **DeepInfra** (OpenAI-compatible endpoints)
- **NVIDIA NIM** (trial service)
- **AWS Bedrock** (cross-region inference)
- **Fireworks AI** (speed-optimized)
- **Hugging Face** (open source weights)

### Usage Instructions

#### Temperature Settings

**Recommended**: 0.6 (range: 0.5-0.7)
- **Top-p**: 0.95
- **Default**: 1.0 (but not optimal)
- **Rationale**: Balances creativity and coherence in reasoning

**Not Recommended**:
- Temperature < 0.3: Too deterministic, less exploration
- Temperature > 0.8: Too random, incoherent reasoning

#### System Prompts

**CRITICAL**: **DO NOT USE** system prompts
- Provide all instructions directly in user query
- Model not trained with system prompts
- Using system prompts degrades performance

#### Prompting Best Practices

**Treat as "Senior Problem-Solver"**:
- Provide high-level objectives
- Let model determine methodology
- Describe problem, task, and output format in detail

**AVOID Few-Shot Examples**:
- Degrades performance significantly
- Model works best with zero-shot prompting
- Exception: Domain-specific notation conventions

**For Math Problems**:
- Add directive: "Please reason step by step, and put your final answer within \\boxed{}"
- Ensures proper formatting and clear final answer

**Example Good Prompt**:
```
Solve the following math problem. Show your reasoning step by step,
and put your final answer within \boxed{}.

Problem: Find all positive integer solutions to the equation x^2 + y^2 = 2025.
```

**Example Bad Prompt** (don't use):
```
System: You are a helpful math tutor.

User: Here are some examples:
1. Problem: 1+1 = ? Answer: 2
2. Problem: 2+2 = ? Answer: 4

Now solve: x^2 + y^2 = 2025
```

#### Output Format

**Standard Response Structure**:
```
<think>
[Long reasoning process here]
- Problem analysis
- Strategy exploration
- Calculations
- Self-verification
</think>

[Final answer here]
```

**Token Limits**:
- **Maximum reasoning tokens**: 32,768
- **Recommended max_tokens**: ≤8,192 for optimal quality
- Longer reasoning may degrade readability

### Deployment

**Compatible Frameworks**:
- **vLLM**: Recommended for production
- **SGLang**: Alternative high-performance serving
- **TensorRT-LLM**: NVIDIA optimized inference

**Recommended Configuration**:
- Tensor parallelism for 671B model
- BF16 precision
- FlashAttention-2 enabled

**Hardware Requirements**:
- **Minimum**: 8× A100 80GB or 8× H100 80GB
- **Recommended**: 8× H100 80GB for best performance

---

## Innovations & Breakthroughs

### 1. Pure RL Reasoning Emergence

**First Open Research** validating that advanced reasoning capabilities can be incentivized purely through RL without supervised fine-tuning.

**Significance**:
- Eliminates need for expensive human-labeled reasoning trajectories
- Shows reasoning is incentivizable through task performance alone
- Opens new research directions for reasoning AI

**Evidence**:
- R1-Zero: 15.6% → 71.0% on AIME 2024 (pure RL)
- Emergent behaviors: self-verification, reflection, error correction
- No explicit CoT training data needed

### 2. GRPO Algorithm Efficiency

**Group Relative Policy Optimization**:
- Eliminates critic model (cuts compute requirements ~50%)
- Group-based relative rewards instead of absolute scoring
- More stable and efficient than PPO

**Impact**:
- Enables efficient large-scale RL at 671B parameters
- Reduces training cost by ~50% vs traditional RLHF
- Simpler implementation and more stable training

### 3. Unprecedented Cost Efficiency

**Training**:
- **1,000× cheaper** than OpenAI o1 (~$5.58M vs $6B+)
- **11× more compute-efficient** than Meta's similar-scale models

**Inference**:
- **90-95% cheaper** than o1 ($0.55/M vs $15/M input tokens)
- Enables widespread deployment and research

### 4. Open Source Impact

**Full Transparency**:
- Model weights and code released under MIT license
- Enables research community to study reasoning emergence
- Allows commercial use and distillation
- Six distilled models (1.5B to 70B) outperform closed models

**Democratization**:
- Makes cutting-edge reasoning AI accessible
- Reduces barriers to entry for smaller organizations
- Accelerates reasoning AI development across industry

### 5. Emergent Reasoning Behaviors

**Without Explicit Training**, R1-Zero developed:
- **Extended chain-of-thought** (thousands of tokens)
- **Self-verification** (checking own outputs)
- **Error detection and correction** (spotting mistakes)
- **Reflection and strategy pivoting** ("aha moments")
- **Language consistency** (though imperfect initially)

**Significance**: Shows that reasoning emerges from task incentives, not just imitation learning.

---

## Disclosed vs Not Disclosed Information

### ✅ Fully Disclosed

**Architecture**:
- Base model (DeepSeek-V3)
- Total parameters (671B) and activated (37B)
- Context length (128K)
- Architecture details (MLA + DeepSeekMoE)
- MoE configuration (256 routed + 1 shared, 8 activated)

**Training**:
- Training approach (GRPO, multi-stage)
- SFT data size (600K reasoning + 200K non-reasoning)
- Cold-start data size (~10K-15K examples)
- RL iterations (~8,000 gradient steps)
- GPU hours (2.788M H800)
- Training cost (~$5.58M)
- Distillation dataset size (800K samples)

**Performance**:
- Comprehensive benchmark scores
- Detailed comparisons with OpenAI o1
- R1-Zero progression data
- Distilled model performance

**Open Source**:
- All model weights (full + 6 distilled variants)
- Full training code
- Evaluation scripts
- Inference code

**Reward System**:
- Accuracy rewards (rule-based)
- Format rewards (rule-based)
- Language consistency rewards (algorithm disclosed)
- No neural reward model used

### ⚠️ Partially Disclosed

**Training Data**:
- **Disclosed**: SFT and reasoning data sizes
- **Disclosed**: Pre-training token count (14.8T)
- **Not Disclosed**: Exact composition of pre-training data
- **Not Disclosed**: Specific data sources and curation methods

### ❌ Not Disclosed

**RLHF Dataset Size** (Most Significant Omission):
- Human preference (RLHF) dataset size undisclosed
- Likely 3-5M samples (speculation based on similar models)
- Stage 4 alignment RL data composition not detailed

**Hyperparameter Details**:
- Learning rates for each stage
- Optimizer specifics beyond AdamW
- Gradient clipping values
- Warmup schedules
- Learning rate decay schedules

**Infrastructure Details**:
- Specific hardware configuration
- Cluster setup and topology
- Network infrastructure
- Parallelism strategies beyond high-level description

**Data Collection Methodology**:
- How cold-start data was curated
- Quality filtering criteria
- Human annotation process for RLHF
- Data mixture ratios for pre-training

**Detailed Ablations**:
- Impact of each training stage individually
- Alternative reward designs tested
- Hyperparameter sensitivity analysis
- Scaling laws for reasoning emergence

---

## Comparison: DeepSeek-R1 vs OpenAI o1

| **Aspect** | **DeepSeek-R1** | **OpenAI o1** | **Advantage** |
|------------|-----------------|---------------|---------------|
| **Training Cost** | ~$5.58M | ~$6B+ | **DeepSeek 1,000× cheaper** |
| **API Cost (Input)** | $0.55/1M tokens | $15/1M tokens | **DeepSeek 95% cheaper** |
| **API Cost (Output)** | $2.19/1M tokens | $60/1M tokens | **DeepSeek 96% cheaper** |
| **Open Source** | Yes (MIT) | No | **DeepSeek** |
| **Training Approach** | Pure RL (R1-Zero), Multi-stage (R1) | Undisclosed | **DeepSeek (disclosed)** |
| **Model Weights** | Public (HuggingFace) | Private | **DeepSeek** |
| **AIME 2024** | 79.8% | 79.2% | **DeepSeek +0.6%** |
| **MATH-500** | 97.3% | 96.4% | **DeepSeek +0.9%** |
| **SWE-bench Verified** | 49.2% | 48.9% | **DeepSeek +0.3%** |
| **Codeforces** | 96.3% | 96.6% | **o1 +0.3% (tie)** |
| **MMLU** | 90.8% | 91.8% | **o1 +1.0%** |
| **GPQA Diamond** | 71.5% | 75.7% | **o1 +4.2%** |
| **Commercial Use** | Yes (MIT) | Yes (paid API) | **DeepSeek (self-host)** |
| **Distilled Models** | 6 variants (1.5B-70B) | o1-mini | **DeepSeek (more options)** |
| **Context Length** | 128K | 128K | Tie |
| **Max Reasoning Tokens** | 32,768 | 25,000-100,000 (varies) | Comparable |

**Verdict**:
- **Performance**: DeepSeek-R1 matches or slightly exceeds o1 on most reasoning benchmarks
- **Cost**: DeepSeek-R1 dramatically cheaper (1,000× training, 95% inference)
- **Transparency**: DeepSeek-R1 fully open source with disclosed methods
- **Accessibility**: DeepSeek-R1 available for self-hosting and distillation
- **Trade-offs**: o1 slightly better on general knowledge (MMLU) and factual reasoning (GPQA)

---

## Limitations & Challenges

### DeepSeek-R1-Zero (Pure RL Variant)

1. **Poor Readability**: Verbose, unstructured reasoning
2. **Language Mixing**: Chinese/English intermixed inconsistently
3. **Endless Repetition**: Sometimes gets stuck in loops
4. **Lack of Formatting**: No markdown, poor structure
5. **User-Unfriendly**: Outputs unsuitable for direct consumption

**Status**: Research artifact demonstrating pure RL emergence, not production-ready

### DeepSeek-R1 (Production Variant)

**Addressed from R1-Zero**:
- ✅ Readable formatting with markdown
- ✅ Language consistency (English or Chinese, not mixed)
- ✅ Proper structure with summaries
- ✅ User-friendly outputs

**Remaining Limitations**:

1. **GPQA Performance Gap**: 71.5% vs o1's 75.7% (-4.2%)
   - Weaker on graduate-level science questions
   - Factual knowledge gap vs reasoning strength

2. **MMLU Slight Gap**: 90.8% vs o1's 91.8% (-1.0%)
   - General knowledge slightly behind
   - Not critical for reasoning tasks

3. **Very Long Reasoning**: Can generate thousands of tokens
   - Sometimes verbose or redundant
   - Increases inference cost and latency
   - Requires patience for complex problems

4. **Specific Prompting Requirements**:
   - Cannot use system prompts (architectural limitation)
   - Few-shot examples degrade performance
   - Requires specific prompting style for optimal results

5. **Language Consistency Trade-off**:
   - Slight performance degradation (~1-2%) from language rewards
   - Necessary for user experience but impacts benchmark scores

### General Limitations

1. **Undisclosed RLHF Data**: Stage 4 alignment data size and composition not disclosed
2. **Hardware Requirements**: 671B model requires 8× A100 80GB minimum
3. **Recent Release**: Limited real-world deployment experience (Jan 2025)
4. **Inference Latency**: Long reasoning increases response time
5. **Not Multimodal**: Text-only, no vision or audio capabilities

---

## Key Quotes from Papers

> "DeepSeek-R1-Zero is trained via large-scale reinforcement learning (RL) without supervised fine-tuning (SFT) as a preliminary step, allowing the model to explore chain-of-thought (CoT) for solving complex problems."

> "On AIME 2024, the pass@1 score increases from 15.6% to 71.0%, and with majority voting, the score further improves to 86.7%, matching the performance of OpenAI-o1-0912."

> "DeepSeek-R1 achieves performance comparable to OpenAI-o1-1217 on reasoning tasks."

> "This is the first open research to validate that reasoning capabilities of LLMs can be incentivized purely through RL, without the need for SFT."

> "The model autonomously develops self-verification (where it checks its own outputs for consistency and correctness), reflection (identifying and correcting errors in its reasoning process) and error correction (to refine its outputs iteratively)."

> "During RL training, R1-Zero demonstrates remarkable emergent behaviors such as self-verification, reflection, and various reasoning patterns, all discovered without explicit supervision."

---

## Sources and References

### Primary Sources
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) - ArXiv 2501.12948
- [DeepSeek-R1 arXiv HTML](https://arxiv.org/html/2501.12948v1) - Formatted version
- [DeepSeek-R1 arXiv PDF](https://arxiv.org/pdf/2501.12948) - Full technical details
- [DeepSeek-R1 GitHub](https://github.com/deepseek-ai/DeepSeek-R1) - Official repository
- [DeepSeek-R1 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-R1) - Model card and weights
- [DeepSeek-R1 Nature Publication](https://www.nature.com/articles/s41586-025-09422-z) - Peer-reviewed publication
- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437) - Base model architecture

### Analysis & Comparisons
- [DeepSeek R1 vs OpenAI o1 - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2025/01/deepseek-r1-vs-openai-o1/)
- [DeepSeek R1 vs o1 - TextCortex](https://textcortex.com/post/deepseek-r1-vs-o1)
- [Vellum Analysis: OpenAI o1 vs DeepSeek R1](https://www.vellum.ai/blog/analysis-openai-o1-vs-deepseek-r1)
- [KNIME: OpenAI o1 vs DeepSeek-R1](https://www.knime.com/blog/openai-o1-vs-deepseek-r1)
- [VentureBeat: 95% Cost Reduction](https://venturebeat.com/ai/open-source-deepseek-r1-uses-pure-reinforcement-learning-to-match-openai-o1-at-95-less-cost)

### Technical Deep Dives
- [HuggingFace: From Zero to Reasoning Hero](https://huggingface.co/blog/NormalUhr/deepseek-r1-explained)
- [Understanding GRPO Math - Medium](https://medium.com/yugen-ai-technology-blog/understanding-the-math-behind-grpo-deepseek-r1-zero-9fb15e103a0a)
- [HuggingFace LLM Course: Understanding R1](https://huggingface.co/learn/llm-course/en/chapter12/3)
- [DeepSeek R1 Training Process - Vellum](https://www.vellum.ai/blog/the-training-of-deepseek-r1-and-ways-to-use-it)
- [Fireworks AI: DeepSeek-R1 Overview](https://fireworks.ai/blog/deepseek-r1-deepdive)

### API & Usage
- [DeepSeek API Documentation](https://api-docs.deepseek.com/)
- [Together.ai: DeepSeek R1 Quickstart](https://docs.together.ai/docs/deepseek-r1)
- [DeepSeek Temperature Settings](https://api-docs.deepseek.com/quick_start/parameter_settings)

### Cost & Training Data Analysis
- [R&D World: $0.55M Token Model](https://www.rdworldonline.com/this-week-in-ai-research-a-0-55-m-token-model-rivals-openais-60-flagship/)
- [Alexandr Wang on X: Training Data Size](https://x.com/alexandr_wang/status/1884440764677251515)
- [Epoch AI: What Went Into Training R1](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1)

---

## Conclusion

**DeepSeek-R1** represents a watershed moment in AI reasoning research. By demonstrating that advanced reasoning capabilities can emerge purely from reinforcement learning without supervised fine-tuning, it challenges conventional wisdom about how LLMs learn to reason.

**Key Achievements**:
1. **Matches OpenAI o1 Performance**: 79.8% AIME 2024, 97.3% MATH-500
2. **1,000× Cheaper Training**: $5.58M vs $6B+ for o1
3. **90-95% Cheaper Inference**: $0.55/M vs $15/M input tokens
4. **Fully Open Source**: MIT license with all weights and code
5. **Emergent Reasoning**: Self-verification, reflection, error correction from RL alone

**Impact**:
- **Democratizes reasoning AI**: Makes cutting-edge capabilities accessible
- **Reduces barriers to entry**: Smaller organizations can compete
- **Accelerates research**: Community can study reasoning emergence
- **Enables commercial use**: Permissive MIT license
- **Provides distilled variants**: 6 models from 1.5B to 70B parameters

**Scientific Contribution**:
- **First open research** validating pure RL reasoning emergence
- **GRPO algorithm**: 50% more efficient than PPO
- **Emergent behaviors**: Reasoning patterns not explicitly trained
- **Scaling insights**: Shows reasoning is incentivizable through task performance

The dramatic cost reduction (training and inference) while matching or exceeding o1's performance, combined with full open-source release under MIT license, democratizes access to reasoning AI and enables unprecedented research into how reasoning emerges in language models. This release may accelerate reasoning AI development across the industry while significantly reducing barriers to entry.

**Key Takeaway**: DeepSeek-R1 proves that advanced reasoning capabilities are achievable through efficient RL training at a fraction of the cost of closed models, and that these capabilities can be democratically shared through open-source releases without compromising performance.
