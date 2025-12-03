# OLMo 3: The Fully Open Language Model Family with Reasoning Capabilities

**Release Date:** November 20, 2025
**Developer:** Allen Institute for AI (AI2)
**Model Sizes:** 7B, 32B parameters
**Variants:** Base, Instruct, Think (9 total models)
**Context Window:** 65,536 tokens (65K)
**License:** Apache 2.0
**Model Flow:** Fully open (data, code, weights, checkpoints, logs)

---

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Model Variants](#model-variants)
4. [Training Data and Dolma 3](#training-data-and-dolma-3)
5. [Training Infrastructure](#training-infrastructure)
6. [Post-Training with Dolci](#post-training-with-dolci)
7. [Reasoning Models: OLMo 3-Think](#reasoning-models-olmo-3-think)
8. [olmOCR: PDF Processing Pipeline](#olmocr-pdf-processing-pipeline)
9. [Performance Benchmarks](#performance-benchmarks)
10. [The "Model Flow" Philosophy](#the-model-flow-philosophy)
11. [Comparison with Competing Models](#comparison-with-competing-models)
12. [Training Efficiency](#training-efficiency)
13. [Deployment and Inference](#deployment-and-inference)
14. [Key Innovations and Contributions](#key-innovations-and-contributions)
15. [Use Cases and Applications](#use-cases-and-applications)
16. [Licensing and Access](#licensing-and-access)
17. [Limitations and Considerations](#limitations-and-considerations)
18. [Technical Specifications](#technical-specifications)
19. [Resources and Links](#resources-and-links)

---

## Overview

OLMo 3, released on November 20, 2025, represents the Allen Institute for AI's (AI2) most advanced generation of fully open language models. Building on the foundation of OLMo 1.0 (February 2024) and OLMo 2 (November 2024), OLMo 3 introduces three critical innovations:

1. **First Fully Open Reasoning Models**: OLMo 3-Think variants are the first fully open models to generate explicit, step-by-step reasoning chains with complete traceability back to training data
2. **Extended Context Window**: 65,536-token context (16x larger than OLMo 2), enabling processing of book chapters, long reports, and multi-document analysis
3. **Complete "Model Flow"**: Unlike "open-weight" models that only release final weights, OLMo 3 exposes the entire development pipeline from raw data to deployment-ready variants

### Key Achievements

- **Best Fully Open Base Model**: OLMo 3-Base 32B outperforms all other fully open base models (where training data, code, and weights are all public) including Stanford's Marin 32B
- **Strongest Fully Open Reasoning Model**: OLMo 3-Think 32B narrows the gap to best open-weight models (Qwen 3-32B-Thinking series) while training on 6x fewer tokens
- **2.5x Training Efficiency**: More efficient to train than Meta's Llama 3.1 (based on GPU-hours per token)
- **Competitive Performance**: Ties or surpasses Qwen 2.5, Gemma 3, and Llama 3.1 in evaluations at similar scale

### Positioning in the AI Landscape

OLMo 3 occupies a unique position as the most transparent AI model release to date:

- **vs. Llama 3.1**: OLMo 3-Instruct should be a clear upgrade on Llama 3.1 8B, representing the best 7B scale model from a Western or American company
- **vs. Qwen 3**: Matches or comes within 1-2 points on major benchmarks while training on 6x fewer tokens
- **vs. DeepSeek R1**: Competitive on math and reasoning while maintaining full transparency
- **vs. Gemma 3**: Superior function calling performance and instruction-following in a fully open 7B model

The key differentiator is that OLMo 3 exposes the entire "model flow"—including the data, training recipes, and intermediate checkpoints—while competitors only release weights. This makes OLMo 3 the first frontier model where you can:

- Trace intermediate reasoning steps back to specific training data
- Understand what data shaped model capabilities
- Reproduce results from scratch
- Customize models at any training stage
- Study how reasoning emerges during training

### Release Highlights

On November 20, 2025, AI2 announced OLMo 3 with the following release components:

**9 Model Variants:**
- OLMo 3-Base (7B, 32B)
- OLMo 3-Instruct (7B, 32B)
- OLMo 3-Think (7B, 32B)
- Plus intermediate training variants (SFT, DPO, RL-Zero)

**3 New Datasets:**
- Dolma 3: ~9.3T token corpus for pretraining
- Dolci: Post-training data suite for reasoning, tool use, and instruction following
- olmOCR-processed scientific PDFs

**Complete Infrastructure:**
- All training code and recipes
- 500+ intermediate checkpoints
- Training logs and ablation studies
- olmOCR toolkit for PDF processing
- Deployment examples and optimizations

### What "Fully Open" Means

OLMo 3 represents the first truly open frontier model since GPT-2. The distinction matters:

**Open-Weight Models (Llama, Qwen, Gemma):**
- ✅ Final model weights
- ❌ Training data
- ❌ Training code
- ❌ Intermediate checkpoints
- ❌ Complete training logs
- ❌ Data curation tools

**Fully Open Models (OLMo 3):**
- ✅ Final model weights
- ✅ Complete training data (Dolma 3, Dolci)
- ✅ Training code and recipes
- ✅ 500+ intermediate checkpoints
- ✅ Complete training logs
- ✅ Data curation tools (olmOCR)
- ✅ Evaluation code and frameworks
- ✅ Post-training datasets (SFT, DPO, RLVR)
- ✅ Ablation studies and analysis

This level of transparency enables unprecedented research opportunities:
- Study how reasoning capabilities emerge during training
- Fork models at any checkpoint for custom development
- Trace model behaviors back to specific data sources
- Reproduce and verify all results
- Build custom training pipelines using proven components

---

## Model Architecture

OLMo 3 employs a decoder-only transformer architecture with several key innovations over OLMo 2, particularly in attention mechanisms and context handling.

### Core Architecture Specifications

```yaml
Architecture: Decoder-only Transformer

OLMo 3 7B:
  Parameters: 7 billion (7B)
  Layers: 32
  Hidden Dimension: 4,096
  Intermediate Size: 11,008
  Attention Heads: 32
  Attention Type: Multi-Head Attention (MHA)
  Context Window: 65,536 tokens
  Vocabulary Size: 50,304
  Sliding Window: 4,096 tokens (3 of 4 layers)

OLMo 3 32B:
  Parameters: 32 billion (32B)
  Layers: 32 (presumed based on documentation)
  Hidden Dimension: TBD
  Intermediate Size: TBD
  Attention Heads: TBD
  Attention Type: Grouped Query Attention (GQA)
  Context Window: 65,536 tokens
  Vocabulary Size: 50,304
  Sliding Window: 4,096 tokens (3 of 4 layers)

Positional Embeddings:
  Type: Rotary Position Embeddings (RoPE)
  RoPE Theta: 10,000.0
  Max Position Embeddings: 2,048 (default)

Activation Function: SiLU (Swish)
Normalization: Post-norm (stabilizes training)
```

### Architectural Innovations

#### 1. Sliding Window Attention

OLMo 3 implements a hybrid attention pattern that balances efficiency and performance:

**Attention Pattern:**
- **Layers 1, 2, 3**: Sliding window attention (4,096-token window)
- **Layer 4**: Full attention (entire 65K context)
- **Layers 5, 6, 7**: Sliding window attention
- **Layer 8**: Full attention
- **Pattern repeats**: 3 sliding window + 1 full attention

**Benefits:**
- **Reduced KV Cache**: Sliding window attention dramatically reduces memory requirements
- **Long Context Support**: Full attention every 4th layer maintains long-range dependencies
- **Efficiency**: Similar to Gemma 3's approach, optimizes inference speed
- **Quality**: Maintains performance while improving efficiency

```
Visual representation:
Layer 32: [Full Attention]     ← Accesses all 65K tokens
Layer 31: [Sliding Window]     ← 4K window
Layer 30: [Sliding Window]     ← 4K window
Layer 29: [Sliding Window]     ← 4K window
Layer 28: [Full Attention]     ← Accesses all 65K tokens
...
Layer 4:  [Full Attention]     ← Accesses all 65K tokens
Layer 3:  [Sliding Window]     ← 4K window
Layer 2:  [Sliding Window]     ← 4K window
Layer 1:  [Sliding Window]     ← 4K window
```

#### 2. Grouped Query Attention (GQA)

The 32B model uses GQA to balance performance and efficiency:

**GQA Configuration:**
- **Multi-Head Attention (MHA)**: When `num_key_value_heads == num_attention_heads`
- **Multi-Query Attention (MQA)**: When `num_key_value_heads == 1`
- **Grouped Query Attention (GQA)**: Otherwise

**Advantages:**
- Reduces KV cache size compared to MHA
- Better quality than MQA
- Optimal for 32B-scale models
- Enables efficient long-context inference

#### 3. Post-Norm Architecture

Unlike many modern transformers that use pre-norm (LayerNorm before attention/FFN), OLMo 3 employs post-norm:

**Post-Norm Benefits:**
- Improved training stability (discovered in OLMo 2 research)
- Better gradient flow
- Consistent with findings from OLMo 2 ablation studies
- Enables more reliable long training runs

```python
# Post-norm architecture (OLMo 3)
x = x + Attention(x)
x = LayerNorm(x)
x = x + FFN(x)
x = LayerNorm(x)

# vs. Pre-norm (traditional)
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

#### 4. Rotary Position Embeddings (RoPE)

OLMo 3 uses RoPE with a theta value of 10,000.0:

**RoPE Properties:**
- Encodes relative positions in attention mechanism
- Enables length extrapolation beyond training context
- Supports various scaling strategies: 'default', 'linear', 'dynamic', 'yarn', 'longrope', 'llama3'
- Theta = 10,000.0 balances near and far positional encoding

**Extended Context Support:**
The model can handle up to 65,536 tokens through:
1. Staged training with increasing context lengths
2. RoPE scaling techniques
3. Long-context specific training data (Dolma 3 Longmino Mix)

### Architecture Changes from OLMo 2

The overall architecture doesn't differ dramatically from OLMo 2, but key improvements include:

1. **Doubled Context Window**: Pre-training and mid-training context increased from 4,096 to 8,192 tokens
2. **Sliding Window Attention**: Implemented on 3 of 4 layers (new in OLMo 3)
3. **Extended Final Context**: Long-context extension stage expands to 65K tokens
4. **Post-Norm Stabilization**: Continues the post-norm approach validated in OLMo 2
5. **Optimized Vocabulary**: 50,304 tokens (optimized for GPU efficiency)

### Model Configuration

The models can be configured through Hugging Face's `transformers` library:

```python
from transformers import OLMo3Config, OLMo3Model

# Default 7B configuration
config = OLMo3Config(
    vocab_size=50304,
    hidden_size=4096,
    intermediate_size=11008,
    num_hidden_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,  # MHA for 7B
    rope_theta=10000.0,
    max_position_embeddings=2048,
    sliding_window=4096,
    attention_bias=False,
    hidden_act="silu",
)

model = OLMo3Model(config)
```

### Memory and Compute Requirements

The architectural choices directly impact deployment:

**7B Model:**
- **BF16**: ~14 GB VRAM (weights only)
- **FP16**: ~14 GB VRAM
- **INT8**: ~7 GB VRAM
- **INT4/GGUF**: ~4-5 GB VRAM

**32B Model:**
- **BF16**: ~64 GB VRAM (weights only)
- **FP16**: ~64 GB VRAM
- **INT8**: ~32 GB VRAM
- **INT4/GGUF**: ~16-20 GB VRAM

**KV Cache (65K context):**
- **7B MHA**: ~8-10 GB additional
- **32B GQA**: ~4-6 GB additional (reduced by GQA)
- **With Sliding Window**: Significantly reduced (only 4K tokens cached for 3/4 layers)

### Design Philosophy

The architecture reflects AI2's research-driven approach:

1. **Evidence-Based**: Every choice backed by ablation studies from OLMo 1.0 and OLMo 2
2. **Efficiency-Focused**: Sliding window attention and GQA balance quality and speed
3. **Context-Optimized**: Staged training and hybrid attention enable 65K context
4. **Reproducible**: Standard architecture makes research and reproduction easier
5. **Practical**: Optimized for real-world deployment on consumer and enterprise GPUs

---

## Model Variants

OLMo 3 is released as a family of 9 model variants across two sizes (7B and 32B) and three training pipelines (Base, Instruct, Think), plus intermediate training checkpoints.

### Complete Model Family

```
OLMo 3 Family Tree:
├── OLMo 3-Base 7B
│   ├── OLMo 3-Instruct 7B (SFT → DPO → RLVR)
│   │   ├── OLMo 3-7B-Instruct-SFT
│   │   ├── OLMo 3-7B-Instruct-DPO
│   │   └── OLMo 3-7B-Instruct (final)
│   └── OLMo 3-Think 7B (SFT → DPO → RLVR)
│       ├── OLMo 3-7B-Think-SFT
│       ├── OLMo 3-7B-Think-DPO
│       ├── OLMo 3-7B-RL-Zero-Code
│       └── OLMo 3-Think 7B (final)
└── OLMo 3-Base 32B
    ├── OLMo 3-Instruct 32B (SFT → DPO → RLVR)
    │   └── OLMo 3-Instruct 32B (final)
    └── OLMo 3-Think 32B (SFT → DPO → RLVR)
        ├── OLMo 3-32B-Think-SFT
        └── OLMo 3-Think 32B (final)
```

### Base Models

#### OLMo 3-Base 7B

**Purpose**: Foundation model for customization and fine-tuning

**Training Pipeline:**
1. **Pretraining**: Dolma 3 Mix (5.9T tokens)
   - 1,024 H100 GPUs
   - ~7,700 tokens/device/second

2. **Mid-training**: Dolma 3 Dolmino Mix (100B tokens)
   - 128 H100 GPUs
   - Focus: math, code, instruction-following, reading comprehension, thinking

3. **Long Context Extension**: Dolma 3 Longmino Mix (50B tokens)
   - 256 H100 GPUs
   - Focus: scientific PDFs, long documents, multi-chapter content

**Model Card**: `allenai/Olmo-3-1025-7B`
**HuggingFace**: [https://huggingface.co/allenai/Olmo-3-1025-7B](https://huggingface.co/allenai/Olmo-3-1025-7B)

**Key Benchmarks:**
- **GSM8K**: 75.5 (math word problems)
- **MATH**: 40.0 (competition mathematics)
- **HumanEval**: 49.1 (code generation)
- **Context**: 65,536 tokens

**Use Cases:**
- Research and experimentation
- Custom fine-tuning for domain-specific tasks
- Studying training dynamics via intermediate checkpoints
- Educational purposes and learning
- Building derivative models

#### OLMo 3-Base 32B

**Purpose**: Strongest fully open base model for advanced customization

**Training Pipeline:**
Same staged approach as 7B but optimized for larger scale:
1. **Pretraining**: Dolma 3 Mix (5.9T tokens)
2. **Mid-training**: Dolma 3 Dolmino Mix (100B tokens)
3. **Long Context Extension**: Dolma 3 Longmino Mix (100B tokens, 2x the 7B allocation)

**Model Card**: `allenai/Olmo-3-1125-32B`
**HuggingFace**: [https://huggingface.co/allenai/Olmo-3-1125-32B](https://huggingface.co/allenai/Olmo-3-1125-32B)

**Key Benchmarks:**
- **GSM8K**: 80.5 (outperforms Marin 32B: 69.1)
- **MATH**: 43.4
- **HumanEval**: 66.5 (outperforms Marin 32B: 52.3)
- **Context**: 65,536 tokens

**Performance vs. Fully Open Competitors:**
- **vs. Marin 32B**: +11.4 points GSM8K, +14.2 points HumanEval
- **vs. Apertus 70B**: Competitive despite being smaller
- **vs. Open-weight models**: Stays competitive with Qwen 2.5 and Gemma 3

**Use Cases:**
- Enterprise fine-tuning
- Research requiring full transparency
- Building specialized reasoning systems
- Domain adaptation for law, medicine, science
- Serving as base for further instruct/RLHF training

### Instruct Models

#### OLMo 3-Instruct 7B

**Purpose**: General-purpose chat and instruction-following model

**Post-Training Pipeline:**
```
OLMo 3-Base 7B
    ↓
Supervised Fine-Tuning (SFT)
    → Dolci-Instruct-SFT-7B dataset
    → Focus: conversation, tool use, instruction-following
    ↓
Direct Preference Optimization (DPO)
    → Dolci-Instruct-DPO-7B dataset
    → Preference learning without reward model
    ↓
Reinforcement Learning from Verifiable Rewards (RLVR)
    → Dolci-Instruct-RL-7B dataset
    → Focus: general chat, instruction-following
    ↓
OLMo 3-Instruct 7B (final)
```

**Model Card**: `allenai/Olmo-3-7B-Instruct`
**HuggingFace**: [https://huggingface.co/allenai/Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct)

**Key Capabilities:**
- **Chat**: Multi-turn conversations with context retention
- **Tool Use**: Function calling and API integration
- **Instruction Following**: High fidelity to user instructions
- **Long Context**: Document QA, summarization up to 65K tokens

**Benchmark Performance:**
- **IFEval**: Strong instruction-following (within 1-2 points of Qwen 3 8B)
- **PopQA**: Long-context question answering
- **General Knowledge**: Competitive with Llama 3.1 8B
- **Function Calling**: Best-in-class for fully open 7B models

**Comparison:**
- **vs. Llama 3.1 8B**: "Clear upgrade" according to AI2, better instruction-following
- **vs. Qwen 2.5 7B**: Ties or surpasses on evaluations
- **vs. Gemma 3 7B**: Superior function calling performance

**Use Cases:**
- Production chatbots and virtual assistants
- Customer service automation
- Document analysis and summarization
- Code assistance and generation
- Tool-augmented AI applications
- Educational tutoring systems

**Deployment Example:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "allenai/Olmo-3-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {"role": "user", "content": "Explain quantum entanglement in simple terms."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

#### OLMo 3-Instruct 32B

**Purpose**: Advanced instruction-following for complex tasks

**Post-Training Pipeline:**
Same three-stage approach as 7B Instruct (SFT → DPO → RLVR) but with 32B-specific datasets.

**Model Card**: `allenai/Olmo-3-32B-Instruct` (presumed naming)

**Advantages over 7B:**
- Better reasoning on complex instructions
- Superior long-context understanding
- More nuanced responses
- Better factual accuracy
- Enhanced coding abilities

**Use Cases:**
- Enterprise AI assistants
- Complex document analysis
- Advanced code generation
- Medical/legal document processing
- Research assistance
- Multi-document reasoning

### Reasoning Models: Think Variants

#### OLMo 3-Think 7B

**Purpose**: Transparent reasoning for math, code, and analytical tasks

**Unique Feature**: First fully open 7B model to generate explicit step-by-step reasoning chains with complete traceability back to training data.

**Post-Training Pipeline:**
```
OLMo 3-Base 7B
    ↓
Supervised Fine-Tuning (SFT)
    → Dolci-Think-SFT-7B dataset
    → Math, code, chat, general knowledge with reasoning traces
    ↓
Direct Preference Optimization (DPO)
    → Dolci-Think-DPO-7B dataset
    → Preference for quality reasoning chains
    ↓
Reinforcement Learning from Verifiable Rewards (RLVR)
    → Dolci-Think-RL-7B dataset
    → Math, code, instruction-following, general chat
    → Verifiable rewards (code execution, math verification)
    ↓
OLMo 3-Think 7B (final)
```

**Model Card**: `allenai/Olmo-3-7B-Think`
**HuggingFace**: [https://huggingface.co/allenai/Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think)

**Special Variant:**
- **OLMo 3-7B-RL-Zero-Code**: Reasoning model specifically optimized for code generation

**Key Benchmarks:**
- **MATH**: Matches Qwen 3 8B performance
- **AIME 2024**: Within a few points of Qwen 3 8B
- **AIME 2025**: Competitive with 8B reasoning models
- **HumanEvalPlus**: Leads all comparison models in code generation
- **MBPP**: Strong coding performance
- **LiveCodeBench**: Demonstrates particular strength in code-intensive reasoning
- **BigBenchHard**: Competitive multi-step reasoning

**Reasoning Transparency:**
Unlike black-box models, OLMo 3-Think exposes:
- Complete reasoning steps in natural language
- Intermediate calculations and logic
- Self-correction and verification steps
- Traceability back to specific training examples

**Example Output:**
```
User: Solve: If 3x + 5 = 20, what is x?

OLMo 3-Think 7B:
Let me solve this step by step.

Step 1: Start with the equation
3x + 5 = 20

Step 2: Isolate the term with x by subtracting 5 from both sides
3x + 5 - 5 = 20 - 5
3x = 15

Step 3: Solve for x by dividing both sides by 3
3x / 3 = 15 / 3
x = 5

Step 4: Verify the solution
3(5) + 5 = 15 + 5 = 20 ✓

Therefore, x = 5.
```

**Use Cases:**
- Mathematical problem solving
- Code generation and debugging
- Analytical reasoning tasks
- Educational applications (showing work)
- Research requiring explainable AI
- Competitive programming preparation

#### OLMo 3-Think 32B

**Purpose**: Strongest fully open reasoning model

**Unique Achievement**: First-ever fully open 32B thinking model, narrowing the gap to best open-weight models while training on 6x fewer tokens.

**Post-Training Pipeline:**
Same three-stage approach as 7B Think (SFT → DPO → RLVR) with 32B-scale datasets.

**Model Card**: `allenai/Olmo-3-32B-Think`
**HuggingFace**: [https://huggingface.co/allenai/Olmo-3-32B-Think](https://huggingface.co/allenai/Olmo-3-32B-Think)

**Key Benchmarks:**
- **MATH**: Wins or within ~2 points of best open-weight model
- **OMEGA**: Ties Qwen 3 VL 32B Thinking for top score
- **BigBenchHard**: Wins or within ~2 points of best
- **HumanEvalPlus**: Wins or within ~2 points of best
- **PopQA**: Effective long-context reasoning
- **IFEval**: Wins or within ~2 points of best
- **vs. Gemma 3 27B Instruct**: Clearly ahead
- **vs. DeepSeek R1 Distill 32B**: Competitive on math and reasoning

**Training Efficiency:**
- Trained on 6x fewer tokens than Qwen 3-32B-Thinking series
- Achieves within 1-2 points overall performance despite less training
- Demonstrates superior token efficiency in reasoning tasks

**Reasoning Capabilities:**
- Multi-step logical reasoning
- Complex mathematical proofs
- Advanced code generation and debugging
- Long-chain analytical thinking
- Self-verification and error correction

**Transparency Features:**
- Trace reasoning steps to Dolci-Think training data
- Understand what examples shaped reasoning patterns
- Reproduce reasoning behaviors
- Study emergence of reasoning during training stages
- Fork and customize reasoning approaches

**Use Cases:**
- Research mathematics and theorem proving
- Complex code generation (algorithms, system design)
- Scientific reasoning and hypothesis generation
- Educational content generation with explanations
- Legal/medical reasoning requiring explainability
- Competitive programming and olympiad preparation

**Comparison with Proprietary Reasoning Models:**
While OLMo 3-Think 32B doesn't match the absolute performance of proprietary reasoning models like GPT-4 or Claude Opus, it offers unique advantages:
- Complete transparency (trace all reasoning to data)
- Customizability (fork and fine-tune reasoning approach)
- Privacy (on-premise deployment)
- Cost (no API fees for self-hosted deployment)
- Research (study how reasoning emerges)

### Intermediate Training Checkpoints

AI2 releases intermediate checkpoints for all variant training:

**SFT Checkpoints:**
- `Olmo-3-7B-Instruct-SFT`
- `Olmo-3-7B-Think-SFT`
- `Olmo-3-32B-Think-SFT`

**DPO Checkpoints:**
- `Olmo-3-7B-Instruct-DPO`
- `Olmo-3-7B-Think-DPO`

**Specialized:**
- `Olmo-3-7B-RL-Zero-Code`: Code-focused reasoning

These intermediate checkpoints enable:
- Research on post-training dynamics
- Custom training pipelines (e.g., SFT only)
- A/B testing different training stages
- Building on partial training for specific use cases

### Variant Selection Guide

**Choose OLMo 3-Base when:**
- Fine-tuning for domain-specific tasks
- Conducting research requiring full transparency
- Building custom training pipelines
- Educational purposes and learning
- Need reproducibility and full control

**Choose OLMo 3-Instruct when:**
- Production chatbots and assistants
- General instruction-following tasks
- Tool use and function calling
- Multi-turn conversations
- Document analysis and summarization

**Choose OLMo 3-Think when:**
- Mathematical reasoning required
- Code generation and debugging
- Analytical and logical tasks
- Educational applications (show work)
- Research requiring explainable reasoning
- Competitive programming

**Choose 7B when:**
- Resource-constrained deployment
- Faster inference needed
- Consumer GPU deployment (RTX 4090, etc.)
- Cost-sensitive applications
- Lower latency critical

**Choose 32B when:**
- Maximum performance required
- Complex reasoning tasks
- Enterprise deployment with ample resources
- Best-in-class fully open model needed
- Advanced research applications

---

## Training Data and Dolma 3

At the core of OLMo 3's training pipeline is **Dolma 3**, a meticulously curated ~9.3 trillion token corpus that represents the largest and highest-quality fully open language model dataset to date.

### Dolma 3 Overview

Dolma 3 consists of three distinct datasets, each serving a specific stage of the training pipeline:

```yaml
Dolma 3 Composition:
  Total Corpus: ~9.3 trillion tokens

  1. Dolma 3 Mix (Pretraining):
     Tokens: 5.9 trillion (5.9T)
     Purpose: Broad knowledge and language understanding

  2. Dolma 3 Dolmino Mix (Mid-training):
     Tokens: 100 billion (100B)
     Source Pool: ~2.2T token pool sampled to 100B
     Purpose: Math, code, reasoning, instruction-following

  3. Dolma 3 Longmino Mix (Long-context):
     Tokens: 50B (7B model), 100B (32B model)
     Source Pool: 639B token pool
     Purpose: Long documents, scientific PDFs, multi-chapter content
```

### Dolma 3 Mix: Pretraining Dataset

**Size**: 5.9 trillion tokens
**Purpose**: Foundation pretraining for broad knowledge and language capabilities

**Data Sources:**

1. **Common Crawl (Web Content)**: Majority of dataset
   - Multi-snapshot crawls from diverse time periods
   - Extensive deduplication and quality filtering
   - Language filtering (primarily English, some multilingual)

2. **Scientific PDFs**: Processed with olmOCR
   - arXiv papers (physics, CS, math)
   - PubMed Central articles (biomedical)
   - Academic preprints and publications
   - Research reports and whitepapers

3. **Code Repositories**:
   - GitHub public repositories
   - Multiple programming languages
   - Documentation and comments
   - Higher proportion than earlier Dolma releases

4. **Mathematical Data**:
   - Math problems and solutions
   - Competition mathematics
   - Textbook problems
   - Proof datasets

5. **Encyclopedic Text**:
   - Wikipedia (multiple snapshots)
   - Educational resources
   - Reference materials

**Data Curation Process:**

```
Raw Sources (~20T+ tokens)
    ↓
Deduplication (exact and fuzzy)
    ↓
Quality Filtering
    - Perplexity filtering
    - Length filtering
    - Language identification
    ↓
N-gram Decontamination
    - Remove 8-word sequences matching benchmarks
    - GSM8K, MMLU, and other eval sets
    - Conservative contamination removal
    ↓
Data Mixing
    - Proportional sampling from sources
    - Higher code and math proportion vs. Dolma 2
    ↓
Dolma 3 Mix: 5.9T tokens
```

**Key Characteristics:**

- **Higher Quality**: More aggressive filtering than Dolma 1.0 and 2.0
- **More Code**: Increased coding data for better programming performance
- **More Math**: Enhanced mathematical reasoning capabilities
- **Extensive Decontamination**: Conservative removal of benchmark contamination
- **Scientific Focus**: Heavy emphasis on scientific PDFs via olmOCR
- **Diverse Time Coverage**: Multi-snapshot approach prevents temporal bias

**Quality Metrics:**

The Dolma 3 Mix represents a significant quality improvement:
- Perplexity-based filtering removes low-quality text
- Fuzzy deduplication (MinHash) removes near-duplicates
- Conservative contamination removal ensures benchmark integrity
- Balanced domain coverage prevents overfitting

### Dolma 3 Dolmino Mix: Mid-Training Dataset

**Size**: 100 billion tokens (sampled from ~2.2T pool)
**Purpose**: Targeted capability enhancement in critical domains

**Mid-Training Focus Areas:**

1. **Mathematics**:
   - Competition math problems (AMC, AIME, IMO)
   - Textbook problems (algebra, calculus, geometry)
   - Mathematical proofs
   - Problem-solving strategies

2. **Code**:
   - High-quality GitHub repositories
   - LeetCode and competitive programming
   - Algorithm implementations
   - Software engineering best practices

3. **Instruction-Following**:
   - Instruction-response pairs
   - Task completion examples
   - Multi-step instructions
   - Tool use demonstrations

4. **Reading Comprehension**:
   - Long-form QA pairs
   - Document understanding tasks
   - Summarization examples
   - Information extraction

5. **Reasoning Traces**:
   - Chain-of-thought examples
   - Step-by-step solutions
   - Logical reasoning demonstrations
   - Verification and self-correction examples

**Mid-Training Strategy:**

The mid-training stage bridges pretraining and post-training:

```
Pretraining (Broad knowledge)
    ↓
Mid-Training (Targeted skills)
    - 128 H100 GPUs
    - 100B tokens from 2.2T pool
    - Focus: math, code, reasoning
    ↓
Long Context Extension
    ↓
Post-Training (Alignment)
```

**Impact on Performance:**

Mid-training with Dolmino Mix produces measurable improvements:
- **GSM8K**: +10-15 points over base pretraining alone
- **HumanEval**: +8-12 points in code generation
- **MATH**: +5-8 points in competition math
- **Instruction-following**: Enables better post-training outcomes

**Data Composition:**

```yaml
Dolmino Mix Proportions (approximate):
  Code: 35%
  Mathematics: 25%
  Instruction-Following: 20%
  Reading Comprehension: 15%
  Reasoning Traces: 5%
```

### Dolma 3 Longmino Mix: Long-Context Dataset

**Size**:
- 7B models: 50 billion tokens
- 32B models: 100 billion tokens

**Source Pool**: 639 billion token pool

**Purpose**: Enable effective long-context understanding up to 65K tokens

**Data Sources:**

1. **Scientific PDFs** (olmOCR-processed):
   - Multi-page research papers
   - Technical reports
   - Academic dissertations
   - Long-form scientific articles

2. **Long Documents**:
   - Books and book chapters
   - Technical documentation
   - Legal documents
   - Reports and whitepapers

3. **Multi-Chapter Content**:
   - Educational materials
   - Textbooks
   - Reference guides

4. **Mid-Training Data** (mixed in):
   - Maintains skills learned in Dolmino stage
   - Prevents catastrophic forgetting

**Long-Context Training Approach:**

```
Stage 3: Long Context Extension
    ↓
Infrastructure:
    - 256 H100 GPUs
    - Extended training sequences
    ↓
Context Window Progression:
    - Start: 8,192 tokens (mid-training length)
    - Progress: Gradually increase
    - Final: 65,536 tokens
    ↓
Data Strategy:
    - Long documents from 639B pool
    - Mixed with Dolmino data
    - Maintains performance on short tasks
    ↓
Result: 65K token context window
```

**Benefits of Staged Long-Context Training:**

1. **Gradual Scaling**: Prevents instability from sudden context expansion
2. **Skill Retention**: Mixing in Dolmino data maintains math/code performance
3. **Quality Focus**: 639B pool → 50-100B ensures only highest-quality long docs
4. **Practical Coverage**: 65K tokens handles most real-world long-context needs

**Long-Context Capabilities:**

The Longmino-trained models can effectively:
- Process research papers (typical 8-12K tokens)
- Analyze book chapters (10-20K tokens)
- Understand technical documentation (15-30K tokens)
- Multi-document reasoning (combining several documents)
- Long conversation history (extended dialogues)

**Benchmark Performance:**

- **PopQA**: Handles ~65K token contexts effectively
- **Extended comprehension tasks**: Demonstrates information retention
- **Multi-document QA**: Answers questions spanning multiple sources
- **Needle-in-haystack**: Retrieves information from arbitrary context positions

### Data Quality and Decontamination

OLMo 3's training data undergoes rigorous quality control:

#### Decontamination Process

**N-gram Decontamination:**
- Scans all training data for 8-word sequences
- Matches against benchmark questions (GSM8K, MMLU, etc.)
- Deletes matching sequences to prevent benchmark leakage
- Conservative approach: better to remove borderline cases

**Why 8-word sequences?**
- Balance between false positives and false negatives
- Captures meaningful phrases without over-deletion
- Industry standard (used by Llama, Qwen, etc.)

**Benchmarks Decontaminated:**
- GSM8K (math word problems)
- MMLU (general knowledge)
- HumanEval (code generation)
- MATH (competition math)
- AIME (advanced math)
- Other standard evaluation sets

**Transparency:**
Unlike closed models, OLMo 3's decontamination is:
- Fully documented
- Reproducible (decontamination code released)
- Verifiable (can check training data directly)
- Conservative (errs on side of removing data)

#### Quality Filtering

**Perplexity Filtering:**
- Uses high-quality reference model
- Removes text with high perplexity (likely low-quality)
- Threshold tuned to balance quality and quantity

**Length Filtering:**
- Removes very short documents (< minimum threshold)
- Removes very long documents (> maximum threshold)
- Ensures reasonable document length distribution

**Language Filtering:**
- Primarily English for OLMo 3
- Language identification using fastText
- Removes non-English content (with some exceptions)

**Heuristic Filters:**
- Removes documents with excessive punctuation
- Filters out repetitive text
- Removes documents with low word diversity
- Eliminates adult content and toxic language

### Dolma Dataset Release

All three Dolma 3 datasets are fully released:

**HuggingFace Datasets:**
- `allenai/dolma3_mix-6T-1025`: Dolma 3 Mix (pretraining)
- `allenai/dolma3-dolmino-100B`: Dolmino Mix (mid-training)
- `allenai/dolma3-longmino-50B`: Longmino Mix (long-context, 7B)
- `allenai/dolma3-longmino-100B`: Longmino Mix (long-context, 32B)

**GitHub Repository:**
- `allenai/dolma3`: Tools and documentation
- Data curation pipeline
- Decontamination scripts
- Quality filtering code

**Dataset Characteristics:**
```python
from datasets import load_dataset

# Load Dolma 3 Mix
dataset = load_dataset("allenai/dolma3_mix-6T-1025", streaming=True)

# Each example contains:
# {
#   "text": "...",           # Document text
#   "source": "...",         # Data source (web, code, arxiv, etc.)
#   "metadata": {...}        # Additional metadata
# }
```

**Usage Rights:**
- **License**: ODC-BY 1.0 (Open Data Commons Attribution License)
- **Commercial Use**: Allowed
- **Redistribution**: Allowed with attribution
- **Modification**: Allowed

### Training Data Philosophy

AI2's approach to training data reflects their commitment to open science:

1. **Complete Release**: All training data publicly available
2. **Documentation**: Detailed description of curation process
3. **Reproducibility**: Tools provided to recreate datasets
4. **Transparency**: Data sources and mixing proportions disclosed
5. **Quality Focus**: Extensive filtering and decontamination
6. **Continuous Improvement**: Dolma 1 → 2 → 3 shows learning and refinement

This stands in stark contrast to closed models:
- **GPT-4**: Training data undisclosed
- **Claude**: Training data undisclosed
- **Gemini**: Training data undisclosed
- **Llama 3.1**: Training data undisclosed (open weights, closed data)

OLMo 3's data transparency enables:
- Research on data-model relationships
- Understanding of model biases and limitations
- Reproduction of training runs
- Custom dataset curation using proven pipelines
- Trust through verifiability

---

## Training Infrastructure

OLMo 3's training infrastructure represents a carefully optimized setup designed to maximize efficiency while maintaining full reproducibility. All training was conducted on NVIDIA H100 GPUs using AI2's research computing infrastructure.

### Hardware Configuration

**GPU Platform**: NVIDIA H100 (80GB HBM3)

**Three-Stage Infrastructure:**

```yaml
Stage 1: Pretraining (Dolma 3 Mix)
  GPUs: 1,024 H100
  Duration: Weeks (exact duration TBD)
  Throughput: ~7,700 tokens/device/second
  Total Tokens: 5.9 trillion

Stage 2: Mid-Training (Dolmino Mix)
  GPUs: 128 H100
  Duration: Days
  Throughput: Optimized for quality over speed
  Total Tokens: 100 billion

Stage 3: Long-Context Extension (Longmino Mix)
  GPUs: 256 H100
  Duration: Days
  Throughput: Reduced due to longer sequences
  Total Tokens: 50B (7B) / 100B (32B)
```

### Training Efficiency

#### Throughput Optimization

**Stage 1 (Pretraining):**
- **7,700 tokens/device/second** on H100
- 1,024 GPUs → ~7.88M tokens/second aggregate
- Efficient batch packing and sequence management
- Optimized data loading pipeline

**Efficiency Factors:**
- Flash Attention 2 for efficient attention computation
- Gradient checkpointing to reduce memory usage
- Mixed precision training (BF16)
- Optimized CUDA kernels
- Pipeline parallelism for large models

#### Parallelization Strategy

**7B Model:**
```
Data Parallelism: Primary strategy
  - Distribute batches across GPUs
  - Synchronize gradients
  - Efficient for 7B scale

Tensor Parallelism: Minimal
  - Model fits on single GPU
  - Only used for very long contexts
```

**32B Model:**
```
Hybrid Parallelism:
  Data Parallelism: Across nodes
  Tensor Parallelism: Within nodes (model sharding)
  Pipeline Parallelism: For long-context stages

Sharding Strategy:
  - FSDP (Fully Sharded Data Parallel)
  - ZeRO-3 optimization
  - Activation checkpointing
```

### Training Hyperparameters

**Optimizer**: AdamW

```yaml
Pretraining Hyperparameters:
  Optimizer: AdamW
  Learning Rate: Peak value TBD
  LR Schedule: Cosine decay with warmup
  Warmup Steps: ~2-5% of total steps
  Weight Decay: 0.1 (typical)
  Gradient Clipping: 1.0 (typical)

  Batch Size:
    7B: TBD (likely 2-4M tokens)
    32B: TBD (likely 4-8M tokens)

  Context Length Progression:
    Stage 1: 2,048 → 8,192 tokens
    Stage 2: 8,192 tokens
    Stage 3: 8,192 → 65,536 tokens

  Precision: BF16 (bfloat16)
```

**Mid-Training Hyperparameters:**
```yaml
Mid-Training:
  Learning Rate: Reduced from pretraining peak
  Batch Size: Smaller (focused on quality)
  Context Length: 8,192 tokens
  Epochs: Multiple passes over Dolmino Mix

Long-Context Extension:
  Learning Rate: Further reduced
  Batch Size: Reduced (long sequences)
  Context Length: Progressive 8K → 16K → 32K → 65K
  Sequence Packing: Disabled for long docs
```

### Training Pipeline

#### Stage 1: Pretraining

```
Infrastructure Setup:
  ├── 1,024 H100 GPUs (128 nodes × 8 GPUs)
  ├── High-bandwidth interconnect (NVLink/InfiniBand)
  └── Distributed storage for Dolma 3 Mix

Training Process:
  ├── Initialize model (random or from checkpoint)
  ├── Load Dolma 3 Mix (5.9T tokens)
  ├── Context: 2K → 8K token progression
  ├── Checkpoints: Every 1,000 steps
  └── Duration: Several weeks

Optimization:
  ├── Flash Attention 2
  ├── Gradient checkpointing
  ├── Mixed precision (BF16)
  ├── Data parallelism (primary)
  └── Sequence packing for efficiency
```

**Checkpoint Strategy:**
- Save every 1,000 steps (minimum)
- 500+ checkpoints total
- Includes optimizer states
- Enables resume from any point
- Released publicly for research

#### Stage 2: Mid-Training

```
Infrastructure Setup:
  ├── 128 H100 GPUs (16 nodes × 8 GPUs)
  ├── Reduced scale (smaller dataset)
  └── Load Dolmino Mix (100B tokens)

Training Process:
  ├── Initialize from best Stage 1 checkpoint
  ├── Load Dolmino Mix (math, code, reasoning)
  ├── Context: Fixed 8,192 tokens
  ├── Multiple epochs over data
  ├── Careful monitoring of metrics
  └── Duration: Days

Focus:
  ├── GSM8K, MATH improvements
  ├── HumanEval, MBPP code metrics
  ├── Instruction-following capability
  └── Reading comprehension
```

**Mid-Training Benefits:**
- +10-15 points GSM8K over base pretraining
- +8-12 points HumanEval
- Enables better post-training outcomes
- Maintains general knowledge while specializing

#### Stage 3: Long-Context Extension

```
Infrastructure Setup:
  ├── 256 H100 GPUs (32 nodes × 8 GPUs)
  ├── Optimized for long sequences
  └── Load Longmino Mix (50-100B tokens)

Training Process:
  ├── Initialize from Stage 2 checkpoint
  ├── Load Longmino Mix + Dolmino Mix
  ├── Context: Progressive extension
  │   ├── Start: 8,192 tokens
  │   ├── Middle: 16,384 → 32,768 tokens
  │   └── Final: 65,536 tokens
  ├── RoPE scaling techniques
  └── Duration: Days

Challenges:
  ├── Memory: Long sequences = large KV cache
  ├── Compute: O(n²) attention cost
  ├── Stability: Preventing performance degradation
  └── Quality: Maintaining short-context performance

Solutions:
  ├── Sliding window attention (3/4 layers)
  ├── Gradient checkpointing
  ├── Mixed data (long + Dolmino)
  └── Progressive length increase
```

### Training Costs

#### Estimated Compute Costs

**OLMo 3 7B:**
```
Stage 1 (Pretraining):
  GPUs: 1,024 H100
  Tokens: 5.9T
  Throughput: 7,700 tokens/device/sec
  Duration: ~21 days (estimated)
  GPU-hours: ~516,000 H100-hours

Stage 2 (Mid-training):
  GPUs: 128 H100
  Tokens: 100B
  Duration: ~2-3 days
  GPU-hours: ~7,200 H100-hours

Stage 3 (Long-context):
  GPUs: 256 H100
  Tokens: 50B
  Duration: ~1-2 days
  GPU-hours: ~9,600 H100-hours

Total: ~533,000 H100-hours
```

**OLMo 3 32B:**
```
Similar staged approach but:
  - Higher per-GPU memory usage
  - More tensor/pipeline parallelism
  - Stage 3: 100B tokens vs 50B
  - Estimated ~2-3x total compute of 7B
```

**Cost Comparison:**

At cloud pricing (~$2.50/H100-hour):
- **OLMo 3 7B**: ~$1.33M
- **OLMo 3 32B**: ~$3-4M (estimated)

**vs. Competitors:**
- **Llama 3.1 8B**: Significantly more (Meta doesn't disclose, likely $5-10M+)
- **OLMo 3 Efficiency**: 2.5x more efficient than Llama 3.1 (GPU-hours per token)

### Training Stability and Monitoring

**Monitoring Systems:**
- TensorBoard logging (loss curves, metrics)
- Wandb integration (experiment tracking)
- Custom evaluation harness (periodic benchmark evals)
- Gradient norm monitoring (detect instabilities)
- Learning rate scheduling visualization

**Stability Measures:**
- Gradient clipping prevents exploding gradients
- Post-norm architecture (proven in OLMo 2)
- Warmup phase prevents early instability
- Checkpoint every 1,000 steps (fast recovery)
- Automated alerts for anomalies

**Quality Checkpoints:**
During training, periodic evaluations on:
- GSM8K (math reasoning)
- HumanEval (code generation)
- MMLU (general knowledge)
- HellaSwag (commonsense reasoning)
- Custom internal benchmarks

### Reproducibility Features

**Complete Training Release:**
1. **Training Code**: Full PyTorch codebase
2. **Training Recipes**: Exact hyperparameters and schedules
3. **Data Pipelines**: Data loading and preprocessing code
4. **Checkpoints**: 500+ intermediate checkpoints
5. **Logs**: Complete training logs (loss curves, metrics)
6. **Optimizer States**: Full training state for exact resumption

**GitHub Repository:**
- `allenai/OLMo`: Main training repository
- Detailed README with setup instructions
- Environment specifications (PyTorch version, dependencies)
- Training scripts for each stage
- Evaluation harness

**Reproducing OLMo 3 Training:**

```bash
# Clone repository
git clone https://github.com/allenai/OLMo.git
cd OLMo

# Install dependencies
pip install -r requirements.txt

# Download Dolma 3 Mix
python scripts/download_dolma3.py

# Stage 1: Pretraining
torchrun --nproc_per_node=8 --nnodes=128 \
  scripts/train.py \
  --config configs/olmo3-7b-pretrain.yaml \
  --data dolma3_mix \
  --gpus 1024

# Stage 2: Mid-training
torchrun --nproc_per_node=8 --nnodes=16 \
  scripts/train.py \
  --config configs/olmo3-7b-midtrain.yaml \
  --data dolma3_dolmino \
  --init-from checkpoints/stage1-best.pt \
  --gpus 128

# Stage 3: Long-context
torchrun --nproc_per_node=8 --nnodes=32 \
  scripts/train.py \
  --config configs/olmo3-7b-longcontext.yaml \
  --data dolma3_longmino \
  --init-from checkpoints/stage2-best.pt \
  --gpus 256
```

### Infrastructure Philosophy

AI2's training infrastructure reflects their open science values:

1. **Transparency**: Complete disclosure of setup and hyperparameters
2. **Efficiency**: Optimized for cost-effectiveness (2.5x vs Llama 3.1)
3. **Reproducibility**: Everything needed to reproduce training released
4. **Academic Focus**: Designed for research, not just model production
5. **Accessibility**: Demonstrates what's possible with academic-scale compute

**Key Insight**: OLMo 3 proves that state-of-the-art models don't require hyperscale infrastructure. With 1,024 H100s and careful data curation, academic labs can produce frontier models.

---

## Post-Training with Dolci

After the three-stage pretraining pipeline (Dolma 3 Mix → Dolmino → Longmino), OLMo 3 undergoes sophisticated post-training using the **Dolci** data suite to produce Instruct and Think variants.

### Dolci: Post-Training Data Suite

**Dolci** (name TBD - specific meaning unclear from sources) is a new post-training data suite tailored specifically for:
- Reasoning capabilities
- Tool use and function calling
- Instruction following
- Conversational quality

Dolci provides separate data mixes for each stage of post-training: **SFT**, **DPO**, and **RLVR**.

### Three-Stage Post-Training Pipeline

OLMo 3's post-training follows a modern three-stage approach that avoids the complexity of traditional RLHF:

```
OLMo 3-Base
    ↓
[Stage 1] Supervised Fine-Tuning (SFT)
    ↓
[Stage 2] Direct Preference Optimization (DPO)
    ↓
[Stage 3] Reinforcement Learning from Verifiable Rewards (RLVR)
    ↓
OLMo 3-Instruct / OLMo 3-Think
```

**Key Advantage**: No separate reward model training required (unlike traditional RLHF/PPO)

### Stage 1: Supervised Fine-Tuning (SFT)

**Purpose**: Teach the model desired output formats and behaviors through supervised learning

**Dataset**: Dolci-SFT (separate mixes for Instruct and Think)

#### Dolci-Instruct-SFT

**Focus Areas:**
- High-quality conversational data
- Tool use demonstrations
- Function calling examples
- Instruction-response pairs
- Multi-turn dialogues

**Data Characteristics:**
- Curated from state-of-the-art SFT datasets
- Human-written and high-quality synthetic data
- Emphasis on helpful, harmless, honest (HHH) responses
- Diverse task coverage (summarization, QA, generation, etc.)

**Example SFT Data:**
```json
{
  "messages": [
    {"role": "user", "content": "Write a function to find prime numbers up to n."},
    {"role": "assistant", "content": "Here's a Python function to find prime numbers up to n:\n\n```python\ndef find_primes(n):\n    \"\"\"Return list of prime numbers up to n using Sieve of Eratosthenes.\"\"\"\n    if n < 2:\n        return []\n    \n    # Initialize sieve\n    is_prime = [True] * (n + 1)\n    is_prime[0] = is_prime[1] = False\n    \n    # Sieve of Eratosthenes\n    for i in range(2, int(n**0.5) + 1):\n        if is_prime[i]:\n            for j in range(i*i, n+1, i):\n                is_prime[j] = False\n    \n    return [i for i in range(n+1) if is_prime[i]]\n```\n\nThis implementation uses the efficient Sieve of Eratosthenes algorithm with O(n log log n) time complexity."}
  ]
}
```

#### Dolci-Think-SFT

**Focus Areas:**
- Math problems with step-by-step solutions
- Code generation with explanations
- Logical reasoning with explicit steps
- General knowledge with structured responses
- Chain-of-thought demonstrations

**Data Characteristics:**
- Emphasizes reasoning transparency
- Step-by-step solution formats
- Self-correction and verification examples
- Mathematical and logical rigor

**Example Think SFT Data:**
```json
{
  "messages": [
    {"role": "user", "content": "If f(x) = 2x + 3 and g(x) = x², what is g(f(2))?"},
    {"role": "assistant", "content": "Let me solve this step by step.\n\nStep 1: First, find f(2)\nf(x) = 2x + 3\nf(2) = 2(2) + 3\nf(2) = 4 + 3\nf(2) = 7\n\nStep 2: Now find g(f(2)) = g(7)\ng(x) = x²\ng(7) = 7²\ng(7) = 49\n\nTherefore, g(f(2)) = 49.\n\nVerification: f(2) = 7, and 7² = 49 ✓"}
  ]
}
```

**SFT Training Process:**

```yaml
SFT Training:
  Base Model: OLMo 3-Base (7B or 32B)
  Dataset: Dolci-{Instruct|Think}-SFT-{7B|32B}
  Epochs: 2-3 (typical for SFT)
  Learning Rate: Reduced from pretraining (e.g., 1e-5)
  Batch Size: Moderate
  Context Length: 8,192 tokens (sufficient for most SFT)
  Loss: Cross-entropy (standard supervised learning)

  Output: OLMo 3-{Instruct|Think}-SFT
```

**SFT Improvements:**
- Base model → SFT: Dramatic improvement in instruction-following
- Learns to format responses appropriately
- Adopts helpful and safe behavior patterns
- Enables multi-turn conversation
- Establishes reasoning format (for Think variants)

### Stage 2: Direct Preference Optimization (DPO)

**Purpose**: Align model outputs with human preferences without training a separate reward model

**Dataset**: Dolci-DPO (high-quality contrastive preference data)

#### DPO vs. Traditional RLHF

**Traditional RLHF (e.g., PPO):**
```
1. Train a reward model on preference data
2. Use reward model to score generations
3. Train policy with RL (PPO algorithm)
4. Complex, unstable, requires large compute
```

**DPO (Direct Preference Optimization):**
```
1. Directly optimize policy from preference data
2. No separate reward model needed
3. Stable supervised learning
4. Simpler, faster, more efficient
```

**Mathematical Formulation:**

DPO optimizes the following objective:

```
L_DPO(π_θ) = -E_{(x,y_w,y_l) ~ D} [log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))]

Where:
- π_θ: Policy model being trained
- π_ref: Reference policy (SFT model)
- y_w: Preferred response ("winner")
- y_l: Dispreferred response ("loser")
- β: Temperature parameter
- σ: Sigmoid function
```

Intuition: Increase probability of preferred responses, decrease probability of dispreferred responses, relative to the reference policy.

#### Dolci-DPO Datasets

**Dolci-Instruct-DPO:**
- Preference pairs for helpfulness
- Safety-aligned response pairs
- Tool-use quality comparisons
- Conversation quality preferences

**Dolci-Think-DPO:**
- Reasoning quality comparisons
- Correct vs. incorrect solution paths
- Clarity and completeness preferences
- Verification and self-correction examples

**Example DPO Data:**
```json
{
  "prompt": "Explain photosynthesis in simple terms.",
  "chosen": "Photosynthesis is the process plants use to make food from sunlight. Here's how it works:\n\n1. Plants absorb sunlight using chlorophyll (green pigment)\n2. They take in CO₂ from air and water from soil\n3. Using light energy, they convert these into glucose (sugar)\n4. Oxygen is released as a byproduct\n\nSimple equation: CO₂ + H₂O + light → glucose + O₂\n\nThis process is essential for life on Earth as it produces oxygen and food for the food chain.",
  "rejected": "Photosynthesis is a biochemical process. It's complicated. Plants do it with light. They make sugar or something."
}
```

**DPO Training Process:**

```yaml
DPO Training:
  Base Model: OLMo 3-{Instruct|Think}-SFT
  Dataset: Dolci-{Instruct|Think}-DPO-{7B|32B}
  Epochs: 1-2
  Learning Rate: Low (e.g., 5e-7)
  Batch Size: Smaller (preference pairs are expensive)
  β (DPO temperature): 0.1-0.5 (typical range)
  Reference Policy: Frozen SFT model

  Output: OLMo 3-{Instruct|Think}-DPO
```

**DPO Improvements:**
- SFT → DPO: Improved response quality and safety
- Better alignment with human preferences
- Reduced harmful or unhelpful outputs
- Enhanced reasoning quality (Think variants)
- More natural conversation flow

### Stage 3: Reinforcement Learning from Verifiable Rewards (RLVR)

**Purpose**: Further refine model using RL with verifiable, objective rewards (unlike human preferences)

**Dataset**: Dolci-RL (hard, diverse prompts for RL training)

#### What is RLVR?

RLVR is a form of reinforcement learning that uses **verifiable rewards**—objective, programmatically checkable outcomes rather than learned reward models or human judgments.

**Verifiable Rewards Examples:**

1. **Math Problems:**
   - Reward: Does the final answer match the ground truth?
   - Verification: Exact numeric/symbolic matching

2. **Code Generation:**
   - Reward: Does the code pass all test cases?
   - Verification: Execute code, check outputs

3. **Instruction-Following:**
   - Reward: Does output satisfy constraints?
   - Verification: Programmatic constraint checking

4. **Factual QA:**
   - Reward: Is the answer factually correct?
   - Verification: Knowledge base lookup

**RLVR vs. RLHF:**

| Aspect | RLVR | RLHF |
|--------|------|------|
| Reward Source | Verifiable outcomes | Human preferences |
| Reward Model | Not needed | Trained from human feedback |
| Objectivity | Objective (pass/fail) | Subjective (preferences) |
| Scalability | Highly scalable | Limited by human labeling |
| Domains | Math, code, logic | General conversation |

#### Dolci-RL Datasets

**Dolci-Instruct-RL:**
- General chat queries
- Instruction-following tasks with checkable constraints
- Tool-use scenarios with verifiable outcomes
- Mixed-domain challenges

**Dolci-Think-RL:**
- Math problems (GSM8K, MATH, competition problems)
- Code challenges (HumanEval, MBPP, LeetCode)
- Logical reasoning with verifiable conclusions
- Multi-step reasoning tasks

**Example RLVR Task:**
```python
{
  "prompt": "Write a function that returns the nth Fibonacci number using memoization.",
  "test_cases": [
    {"input": 0, "expected": 0},
    {"input": 1, "expected": 1},
    {"input": 10, "expected": 55},
    {"input": 20, "expected": 6765}
  ],
  "reward_function": lambda code, tests: sum(execute(code, t["input"]) == t["expected"] for t in tests) / len(tests)
}
```

**RLVR Training Process:**

```yaml
RLVR Training:
  Base Model: OLMo 3-{Instruct|Think}-DPO
  Dataset: Dolci-{Instruct|Think}-RL-{7B|32B}
  Algorithm: PPO or similar policy gradient method
  Reward: Verifiable outcomes (code execution, math verification)
  Epochs: Multiple passes with generation + reward + update cycles
  Learning Rate: Very low (e.g., 1e-6)
  KL Penalty: Prevents deviation from DPO policy

  Training Loop:
    1. Generate multiple responses for each prompt
    2. Execute/verify each response (reward)
    3. Compute policy gradient
    4. Update model with PPO
    5. Repeat

  Output: OLMo 3-{Instruct|Think} (final)
```

**RLVR Improvements:**
- DPO → RLVR: Improved task success rate
- Math: Higher accuracy on GSM8K, MATH
- Code: More solutions pass test cases
- Logic: Better constraint satisfaction
- Robustness: Handles diverse, hard prompts

### Specialized Variant: OLMo 3-7B-RL-Zero-Code

One notable specialized model is **OLMo 3-7B-RL-Zero-Code**, a reasoning model specifically optimized for code generation.

**Training:**
- Starts from OLMo 3-Base 7B
- Think-style SFT with heavy code emphasis
- RLVR focused exclusively on code tasks
- Reward: Test case pass rate

**Benchmarks:**
- **HumanEvalPlus**: Leads all comparison models
- **MBPP**: Strong performance
- **LiveCodeBench**: Demonstrates code reasoning strength

**Use Cases:**
- Code generation and completion
- Algorithm implementation
- Code debugging and correction
- Competitive programming

### Post-Training Datasets Release

All Dolci datasets are released for research and reproduction:

**HuggingFace Datasets:**
- `allenai/dolci-instruct-sft-7b`
- `allenai/dolci-instruct-dpo-7b`
- `allenai/dolci-instruct-rl-7b`
- `allenai/dolci-think-sft-7b`
- `allenai/dolci-think-dpo-7b`
- `allenai/dolci-think-rl-7b`
- (32B variants similarly)

**Dataset Contents:**
```python
# SFT Dataset
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}

# DPO Dataset
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "..."
}

# RL Dataset
{
  "prompt": "...",
  "test_cases": [...],
  "reward_function": "..."
}
```

### Post-Training Infrastructure

**Compute Requirements:**

```yaml
SFT Stage:
  GPUs: 64-128 H100
  Duration: 1-2 days
  Cost: Relatively low

DPO Stage:
  GPUs: 64-128 H100
  Duration: ~1 day
  Cost: Low (single epoch typical)

RLVR Stage:
  GPUs: 128-256 H100
  Duration: 2-4 days
  Cost: Highest (multiple generation-update cycles)

Total Post-Training: ~$100-300K per model variant
```

**vs. Pretraining:**
- Pretraining: $1-4M
- Post-training: $100-300K
- Post-training is ~5-10% of total cost

### Post-Training Philosophy

AI2's post-training approach reflects several key principles:

1. **Simplicity**: Avoid complex RLHF in favor of DPO
2. **Verifiability**: Use RLVR where possible (objective rewards)
3. **Transparency**: Release all post-training data and code
4. **Modularity**: Clean SFT → DPO → RLVR pipeline
5. **Reproducibility**: Researchers can reproduce or fork at any stage

**Key Innovation**: The combination of DPO (for preferences) and RLVR (for verifiable tasks) balances alignment with performance without the complexity and instability of traditional RLHF.

---

## Reasoning Models: OLMo 3-Think

The **OLMo 3-Think** variants represent a groundbreaking achievement: the first fully open language models to generate explicit, step-by-step reasoning chains with complete traceability back to training data.

### What Makes OLMo 3-Think Special

**First Fully Open Thinking Models:**
- OLMo 3-Think 32B: First-ever fully open 32B-scale thinking model
- OLMo 3-Think 7B: First fully open 7B model with chain-of-thought reasoning
- Complete "glass box" transparency: trace reasoning to training data

**vs. Proprietary Reasoning Models:**
- **GPT-4, Claude Opus, Gemini**: Reasoning process hidden
- **DeepSeek R1, Qwen QwQ**: Open weights, but training data/process hidden
- **OLMo 3-Think**: Everything open—data, code, intermediate checkpoints

**Key Capability**: Long chain-of-thought thinking that improves reasoning tasks like math and coding while maintaining full transparency.

### Chain-of-Thought (CoT) Reasoning

#### What is Chain-of-Thought?

Chain-of-Thought prompting encourages models to "think out loud," breaking down complex problems into explicit reasoning steps.

**Example Without CoT:**
```
User: If John has 3 apples and gives 2 to Mary, then buys 5 more, how many does he have?

Model: 6 apples.
```

**Example With CoT (OLMo 3-Think):**
```
User: If John has 3 apples and gives 2 to Mary, then buys 5 more, how many does he have?

Model: Let me work through this step by step.

Step 1: John starts with 3 apples
Initial: 3 apples

Step 2: John gives 2 apples to Mary
After giving: 3 - 2 = 1 apple

Step 3: John buys 5 more apples
Final: 1 + 5 = 6 apples

Therefore, John has 6 apples.

Verification: 3 (start) - 2 (gave) + 5 (bought) = 6 ✓
```

#### Why CoT Improves Reasoning

**Benefits of Explicit Reasoning:**

1. **Correctness**: Breaking down complex problems reduces errors
2. **Transparency**: Users can verify each step
3. **Debugging**: Identify exactly where reasoning fails
4. **Learning**: Models learn to self-correct during reasoning
5. **Verifiability**: Each step can be checked independently

**Research Evidence:**
- CoT prompting improves performance on GSM8K by 10-30%
- Larger models benefit more from CoT
- Training with CoT examples embeds reasoning capability
- Verifiable rewards during RLVR reinforce correct reasoning

### Training OLMo 3-Think

#### Data Pipeline: Reasoning-Focused

**Dolma 3 + Dolmino Mix:**
- Base pretraining includes math and code-heavy data
- Mid-training emphasizes reasoning-oriented tasks
- Reasoning traces (CoT examples) included in Dolmino

**Dolci-Think-SFT:**
- Math problems with detailed solution steps
- Code generation with explanatory comments
- Logical reasoning with explicit inference chains
- General knowledge with structured responses

**Example SFT Data:**
```json
{
  "messages": [
    {"role": "user", "content": "Prove that the sum of two even numbers is even."},
    {"role": "assistant", "content": "I'll prove this using the definition of even numbers.\n\nDefinition: An integer n is even if n = 2k for some integer k.\n\nProof:\n1. Let a and b be two even numbers.\n2. By definition, a = 2m for some integer m, and b = 2n for some integer n.\n3. Consider the sum: a + b = 2m + 2n\n4. Factor out 2: a + b = 2(m + n)\n5. Since m and n are integers, (m + n) is also an integer.\n6. Let k = m + n. Then a + b = 2k where k is an integer.\n7. By definition, a + b is even. ∎\n\nTherefore, the sum of two even numbers is even."}
  ]
}
```

**Dolci-Think-DPO:**
- Preference for clear, step-by-step reasoning
- Correct solution paths vs. incorrect ones
- Complete reasoning vs. jumping to conclusions
- Self-correction vs. persisting with errors

**Example DPO Data:**
```json
{
  "prompt": "What is 15% of 80?",
  "chosen": "Let me calculate 15% of 80 step by step.\n\nMethod 1 (Convert to decimal):\n15% = 15/100 = 0.15\n0.15 × 80 = 12\n\nMethod 2 (Fraction):\n15% = 15/100 = 3/20\n(3/20) × 80 = 3 × 4 = 12\n\nTherefore, 15% of 80 is 12.",
  "rejected": "15% of 80 is 12."
}
```

**Dolci-Think-RL:**
- Math problems with verifiable answers
- Code challenges with test cases
- Logic puzzles with checkable solutions
- Multi-step reasoning with intermediate verification

**Reward Function Example:**
```python
def math_reward(model_output, ground_truth):
    """
    Reward based on:
    1. Correct final answer (0.5 weight)
    2. Valid reasoning steps (0.3 weight)
    3. Self-verification (0.2 weight)
    """
    final_answer_correct = extract_answer(model_output) == ground_truth
    has_explicit_steps = count_reasoning_steps(model_output) >= 3
    has_verification = "verification" in model_output.lower() or "check" in model_output.lower()

    reward = 0.5 * final_answer_correct + 0.3 * has_explicit_steps + 0.2 * has_verification
    return reward
```

### OLMo 3-Think Performance

#### Benchmark Results

**OLMo 3-Think 7B:**

```yaml
Mathematics:
  MATH: Matches Qwen 3 8B
  AIME 2024: Within a few points of Qwen 3 8B
  AIME 2025: Within a few points of Qwen 3 8B
  GSM8K: Competitive with 8B reasoning models

Coding:
  HumanEvalPlus: Leads all comparison models
  MBPP: Strong performance
  LiveCodeBench: Particular strength in code-intensive reasoning

Multi-Step Reasoning:
  BigBenchHard: Competitive with Qwen 3 8B reasoning
  OMEGA: Strong multi-step reasoning performance
```

**OLMo 3-Think 32B:**

```yaml
Mathematics:
  MATH: Wins or within ~2 points of best open-weight model
  OMEGA: Ties Qwen 3 VL 32B Thinking for top score
  AIME: Competitive with top reasoning models

Coding:
  HumanEvalPlus: Wins or within ~2 points of best
  MBPP: Top-tier performance

Multi-Step Reasoning:
  BigBenchHard: Wins or within ~2 points of best
  IFEval: Wins or within ~2 points of best
  PopQA: Effective long-context reasoning

vs. Competitors:
  Qwen 3 VL 32B Thinking: Ties on OMEGA
  Gemma 3 27B Instruct: Clearly ahead
  DeepSeek R1 Distill 32B: Competitive on math/reasoning

Training Efficiency:
  6x fewer tokens than Qwen 3-32B-Thinking series
  Within 1-2 points overall despite less training
```

#### Detailed Benchmark Comparison

| Model | MATH | AIME 2025 | HumanEval+ | OMEGA | BigBenchHard |
|-------|------|-----------|------------|-------|--------------|
| **OLMo 3-Think 7B** | ~Match Qwen 3 8B | ~Competitive | **Leads** | Strong | Competitive |
| Qwen 3 8B Reasoning | Baseline | Baseline | Lower | Baseline | Baseline |
| Llama 3.1 8B | Lower | Lower | Lower | Lower | Lower |
| Gemma 3 7B | Lower | Lower | Lower | Lower | Lower |
| **OLMo 3-Think 32B** | ~Best | Competitive | ~Best | **Ties 1st** | ~Best |
| Qwen 3 VL 32B Thinking | ~Best | Competitive | ~Best | **Ties 1st** | ~Best |
| Gemma 3 27B Instruct | Lower | Lower | Lower | Lower | Lower |
| DeepSeek R1 Distill 32B | Competitive | Competitive | Competitive | Competitive | Competitive |

**"~Best"**: Within ~2 points of best open-weight model
**"Competitive"**: Within a few points
**"Leads"**: Best performance among comparisons

### Reasoning Transparency Features

#### Tracing Reasoning to Training Data

**Unique OLMo 3-Think Capability:**

Because all training data is released, researchers can:

1. **Identify Source Examples**: Find training examples that taught specific reasoning patterns
2. **Understand Emergence**: Study how reasoning capabilities emerged during training
3. **Debug Failures**: Trace incorrect reasoning back to data issues
4. **Improve Future Models**: Understand what data produces better reasoning

**Example Research Workflow:**
```python
# 1. Model generates reasoning chain
output = model.generate("Prove √2 is irrational")

# 2. Extract reasoning steps
steps = extract_reasoning_steps(output)

# 3. Search training data for similar reasoning patterns
similar_examples = search_dolci_think_sft(steps)

# 4. Analyze: What data taught this reasoning approach?
analyze_provenance(similar_examples)

# 5. Study: How did this capability emerge during training?
study_checkpoints(capability="irrational_number_proofs", checkpoints=500)
```

This level of transparency is **impossible** with closed models or even open-weight models (where training data is hidden).

#### Glass-Box AI

OLMo 3-Think represents "glass-box AI" vs. "black-box AI":

**Black-Box AI (GPT-4, Claude, Gemini):**
- Input → [???] → Output
- No visibility into reasoning process
- Can't trace behaviors to causes
- Can't verify training approach

**Open-Weight AI (Llama 3, Qwen 3):**
- Input → [Weights available] → Output
- Can inspect model internals
- Can't trace to training data
- Can't reproduce training

**Glass-Box AI (OLMo 3-Think):**
- Input → [Fully traceable] → Output
- Complete reasoning transparency
- Trace behaviors to training data
- Reproduce entire training pipeline
- Study emergence of capabilities

### Reasoning Use Cases

**Mathematical Problem Solving:**
- Homework assistance with step-by-step solutions
- Competition math preparation (AMC, AIME, IMO)
- Mathematical proofs and derivations
- Equation solving with verification

**Code Generation and Debugging:**
- Algorithm implementation with explanations
- Code debugging with reasoning about errors
- Competitive programming solutions
- Software design with architectural reasoning

**Analytical Reasoning:**
- Logical puzzles and games
- Scientific hypothesis generation
- Legal reasoning and case analysis
- Medical diagnosis support (with appropriate disclaimers)

**Educational Applications:**
- Tutoring systems that show work
- Generating worked examples for students
- Explaining complex concepts step-by-step
- Assessment with partial credit for reasoning

**Research Applications:**
- Studying AI reasoning capabilities
- Developing better reasoning training methods
- Understanding emergence of reasoning
- Building interpretable AI systems

### Limitations of OLMo 3-Think

**Not a Reasoning Specialist:**
While OLMo 3-Think has strong reasoning capabilities, it's not optimized exclusively for reasoning like some models:
- **DeepSeek R1**: Dedicated reasoning model, stronger on pure math/logic
- **OpenAI o1/o3**: Specialized reasoning models with extended thinking
- **OLMo 3-Think**: Balanced model with reasoning capabilities plus general knowledge

**Trade-offs:**
- OLMo 3-Think maintains general capabilities (chat, knowledge, etc.)
- Specialized reasoning models might outperform on narrow reasoning tasks
- But OLMo 3-Think offers full transparency (unique advantage)

**Performance Gaps:**
- Doesn't match proprietary reasoning models (GPT-4, Claude Opus) on hardest tasks
- Within 1-2 points of best open-weight models, not ahead
- Trained on 6x fewer tokens, so efficiency is high but not peak performance

**Context Length:**
- 65K tokens is good, but some reasoning tasks benefit from even longer context
- Extended reasoning (like o1) not the primary design goal

### Future of OLMo Reasoning Models

The OLMo 3-Think release establishes a foundation for:
- Future versions with extended reasoning (longer CoT)
- Specialized reasoning variants (math-only, code-only)
- Multimodal reasoning (images + reasoning)
- Larger models (70B+, 400B+ thinking models)

The key differentiator will remain: **full transparency** enabling research impossible with closed or open-weight-only models.

---

## olmOCR: PDF Processing Pipeline

A critical component of OLMo 3's training infrastructure is **olmOCR**, an open-source toolkit for converting PDFs into clean, structured text suitable for language model training. This innovation enables OLMo 3 to train on trillions of tokens from scientific PDFs that were previously inaccessible or poorly processed.

### The PDF Processing Challenge

**Why PDFs are Hard:**

Traditional OCR and PDF extraction tools struggle with:
1. **Complex Layouts**: Multi-column formats, sidebars, figures
2. **Mathematical Notation**: Equations, symbols, Greek letters
3. **Tables**: Structured data in various formats
4. **Reading Order**: Determining correct text flow
5. **Quality**: Scanned PDFs, handwritten text, poor scans
6. **Structured Content**: Preserving sections, lists, hierarchies

**Previous Approaches:**

- **Grobid**: Rule-based PDF parser for scientific papers
  - Good for basic extraction
  - Struggles with complex layouts
  - Limited handling of equations and tables

- **PyPDF2, pdfminer**: Basic text extraction
  - Loses structure and formatting
  - Poor handling of multi-column layouts
  - No understanding of semantics

- **Commercial OCR**: Tesseract, Adobe
  - Better quality but still rule-based
  - Not optimized for LLM training data
  - Proprietary in many cases

**Impact on LLM Training:**

Poor PDF processing means:
- Loss of valuable scientific knowledge
- Corrupted text in training data
- Inability to learn from equations and tables
- Reduced pretraining dataset quality

olmOCR solves these problems using a vision-language model approach.

### olmOCR Architecture

**Core Innovation**: Use a fine-tuned vision-language model (VLM) to "read" PDFs as images and extract structured text.

```yaml
olmOCR Pipeline:
  Input: PDF file

  Step 1: PDF to Images
    - Convert each PDF page to image (PNG/JPG)
    - Maintain resolution for quality

  Step 2: Vision-Language Model Processing
    - VLM: Fine-tuned 7B model on PDF-to-text task
    - Training Data: 260,000 pages from 100,000+ PDFs
    - Diverse Properties: Graphics, handwriting, poor scans

  Step 3: Structured Text Extraction
    - Natural reading order
    - Preserved structure (sections, lists, tables)
    - Equations in LaTeX or text form
    - Tables in markdown or structured format

  Output: Clean, linearized plain text in natural reading order
```

**Vision-Language Model:**
- **Size**: 7B parameters
- **Training**: Fine-tuned on olmOCR-mix-0225
- **Dataset**: 260,000 pages from 100,000+ crawled PDFs
- **Diversity**: Graphics, handwritten text, poor quality scans, equations, tables

### olmOCR Capabilities

**Content Types Handled:**

1. **Sections and Headings**:
   - Identifies hierarchical structure
   - Preserves section numbering
   - Maintains document organization

2. **Tables**:
   - Converts to markdown or structured format
   - Preserves row/column relationships
   - Handles merged cells and complex layouts

3. **Lists**:
   - Numbered lists, bullet points
   - Nested lists
   - Preserves list structure

4. **Equations**:
   - Extracts mathematical notation
   - LaTeX representation where appropriate
   - Inline and display equations

5. **Graphics and Figures**:
   - Can describe figures (if VLM has that capability)
   - At minimum, identifies figure captions
   - Maintains reading flow around figures

6. **Multi-Column Layouts**:
   - Determines correct reading order
   - Handles complex academic paper layouts
   - Preserves logical text flow

7. **Handwritten Text**:
   - OCR of handwritten annotations
   - Lecture notes and manuscripts
   - Quality depends on handwriting clarity

8. **Poor Quality Scans**:
   - Handles degraded images
   - Low-resolution scans
   - Artifacts and noise

### olmOCR Performance

**Benchmark: peS2o Dataset**

peS2o is a widely-used corpus of 7.9M scientific PDFs used in LLM pretraining. It was originally processed using Grobid + rule-based extraction.

**Experiment**: Replace Grobid-extracted text with olmOCR-extracted text.

**Result**: Training on olmOCR-peS2o produces a **+1.3 percentage point average improvement** on widely-reported LLM benchmark tasks.

| Dataset Version | MMLU | HellaSwag | PIQA | WinoGrande | Average |
|-----------------|------|-----------|------|------------|---------|
| Original peS2o (Grobid) | Baseline | Baseline | Baseline | Baseline | Baseline |
| olmOCR-peS2o | +1.5 | +1.3 | +1.1 | +1.2 | **+1.3** |

**Interpretation**:
- Higher quality text extraction → better LLM pretraining
- olmOCR captures information Grobid missed
- Structured content (tables, equations) improves reasoning
- +1.3 points is significant for pretraining improvements

### olmOCR in Dolma 3

**Scientific PDFs in OLMo 3 Training:**

```yaml
Dolma 3 Mix (Pretraining):
  Scientific PDFs (olmOCR-processed):
    - arXiv papers (physics, CS, math)
    - PubMed Central (biomedical)
    - Academic preprints and publications
    - Research reports and whitepapers

  Estimated Proportion: 10-20% of 5.9T tokens
  Tokens from PDFs: ~600B-1.2T tokens

Dolma 3 Longmino Mix (Long-context):
  Scientific PDFs:
    - Multi-page research papers
    - Technical reports
    - Dissertations

  Proportion: Majority of long-context data
  Tokens: ~25-50B from PDFs
```

**Impact on OLMo 3:**
- **Scientific Knowledge**: Access to cutting-edge research
- **Mathematical Capability**: Learns from equations in context
- **Technical Depth**: Exposure to detailed technical content
- **Long-Form Reasoning**: Multi-page papers teach coherent long-form thinking

### olmOCR Toolkit

**Open-Source Release:**

**GitHub**: `allenai/olmocr`
**Components**:
1. Fine-tuned 7B VLM for PDF-to-text
2. Training code and recipes
3. Efficient inference pipeline
4. Benchmark suite
5. Training dataset (olmOCR-mix-0225)

**Usage Example:**

```bash
# Install olmOCR
pip install olmocr

# Convert a single PDF to markdown
python -m olmocr.pipeline ./workspace --markdown --pdfs document.pdf

# Batch process PDFs
python -m olmocr.pipeline ./workspace --markdown --pdfs *.pdf

# Output: Clean markdown files with structured content
```

**Python API:**

```python
from olmocr import PDFProcessor

# Initialize processor
processor = PDFProcessor(model="allenai/olmocr-7b")

# Process PDF
result = processor.process_pdf("research_paper.pdf")

# Extract structured text
text = result.get_text()
sections = result.get_sections()
tables = result.get_tables()
equations = result.get_equations()

# Export to markdown
markdown = result.to_markdown()
with open("output.md", "w") as f:
    f.write(markdown)
```

### olmOCR Scalability

**Large-Scale Batch Processing:**

olmOCR is optimized for processing millions of PDFs efficiently:

**Cost Efficiency:**
- **1 million PDF pages**: $176 USD
- **Processing**: Flexibly scales to different hardware setups
- **GPU Options**: Can run on consumer GPUs (RTX 3090, 4090) or datacenter GPUs (A100, H100)

**Scaling Strategy:**

```yaml
Small Scale (< 10K PDFs):
  Hardware: Single RTX 4090 or A100
  Throughput: 100-500 pages/hour
  Cost: ~$0.10-0.50 per 1000 pages

Medium Scale (10K-100K PDFs):
  Hardware: 4-8 A100/H100 GPUs
  Throughput: 1000-5000 pages/hour
  Parallelization: Multi-GPU processing
  Cost: ~$0.05-0.20 per 1000 pages

Large Scale (1M+ PDFs, OLMo 3 scale):
  Hardware: 64-256 GPUs
  Throughput: 50,000+ pages/hour
  Infrastructure: Distributed processing cluster
  Cost: ~$0.02-0.10 per 1000 pages (with optimization)

Example: 7.9M peS2o PDFs
  Pages: ~60M pages (estimated)
  Cost: ~$10,500 at $0.176 per 1000 pages
  Time: ~1200 hours on 50K pages/hour → parallelized to days
```

### olmOCR Training Dataset

**olmOCR-mix-0225:**

```yaml
Dataset: olmOCR-mix-0225
Release Date: February 2025 (0225 = Feb 2025)
Size: 260,000 pages from 100,000+ PDFs
Purpose: Training the 7B VLM for PDF processing

Data Characteristics:
  - Diverse PDF types (scientific, technical, reports)
  - Graphics and figures
  - Handwritten text and annotations
  - Poor quality scans
  - Complex layouts (multi-column, tables, equations)

Ground Truth:
  - Human-labeled structured text
  - Correct reading order
  - Proper structure preservation

Training:
  - Fine-tuned from base VLM
  - Supervised learning (PDF image → structured text)
  - Multiple epochs for quality
```

**Dataset Release:**
- **HuggingFace**: `allenai/olmocr-mix-0225`
- **License**: Open for research and commercial use
- **Format**: PDF images + ground-truth structured text

### olmOCR Research Applications

**Enabled Research:**

1. **LLM Pretraining Quality**:
   - Study impact of PDF processing quality on downstream performance
   - Compare olmOCR vs. traditional methods
   - Optimize PDF data in pretraining mix

2. **Scientific Knowledge Extraction**:
   - Build knowledge bases from research papers
   - Extract datasets from publications
   - Create structured scientific corpora

3. **Document Understanding**:
   - Train document understanding models
   - Layout analysis research
   - Information extraction from complex documents

4. **Multimodal Learning**:
   - Learn from text + layout + figures
   - Visual-textual reasoning
   - Document-level multimodal models

### olmOCR Limitations

**Current Limitations:**

1. **Compute Cost**: Requires GPU inference (VLM-based)
2. **Speed**: Slower than rule-based methods (but higher quality)
3. **Errors**: Not perfect, especially on very poor quality scans
4. **Languages**: Primarily trained on English scientific PDFs
5. **Specialized Notation**: Some domain-specific symbols may be missed

**Future Improvements:**
- Faster inference (distillation, quantization)
- Multilingual support
- Specialized variants (math-heavy, code-heavy)
- Smaller models for edge deployment

### olmOCR Impact on OLMo 3

**Quantified Improvements:**

```yaml
Without olmOCR:
  - Scientific PDFs: Poorly processed or excluded
  - Training Tokens: Missing ~600B-1T high-quality tokens
  - Scientific Capability: Reduced exposure to research
  - Mathematical Reasoning: Fewer equation examples in context

With olmOCR:
  - Scientific PDFs: High-quality extraction
  - Training Tokens: +600B-1T tokens from scientific content
  - Scientific Capability: Strong performance on scientific QA
  - Mathematical Reasoning: +1.3 points on average benchmarks

Overall Impact:
  - OLMo 3 benefits from comprehensive scientific knowledge
  - Better performance on STEM tasks
  - Long-context understanding of technical documents
  - Reproducible pipeline for future models
```

**Open Science Impact:**

olmOCR's release enables the entire research community to:
- Access high-quality scientific text from PDFs
- Improve their own LLM pretraining datasets
- Build on AI2's proven methodology
- Advance document understanding research

This exemplifies AI2's commitment to **open infrastructure**—not just releasing models, but the tools to build them.

---

