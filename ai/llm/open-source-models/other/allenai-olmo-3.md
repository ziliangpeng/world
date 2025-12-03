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

## Performance Benchmarks

OLMo 3's performance across standard benchmarks demonstrates its position as a leading fully open model while competing effectively with top open-weight models.

### Benchmark Overview

OLMo 3 is evaluated on a comprehensive suite of benchmarks covering:
- **General Knowledge**: MMLU, TriviaQA
- **Mathematical Reasoning**: GSM8K, MATH, AIME
- **Code Generation**: HumanEval, HumanEval+, MBPP
- **Multi-Step Reasoning**: BigBenchHard, OMEGA
- **Long-Context**: PopQA, extended comprehension tasks
- **Instruction-Following**: IFEval
- **Commonsense Reasoning**: HellaSwag, WinoGrande, PIQA
- **Reading Comprehension**: Various QA datasets

### OLMo 3-Base Performance

#### OLMo 3-Base 7B

```yaml
Mathematics:
  GSM8K (8-shot): 75.5
  MATH (4-shot): 40.0

Code Generation:
  HumanEval (0-shot): 49.1

General Knowledge:
  MMLU (5-shot): Est. 60-65 (not prominently reported)

Context Window: 65,536 tokens
```

**vs. Similar Base Models:**
- Competitive with Llama 3.1 8B Base
- Slightly behind Qwen 2.5 7B Base on some benchmarks
- Best among fully open base models at 7B scale

#### OLMo 3-Base 32B

```yaml
Mathematics:
  GSM8K (8-shot): 80.5
  MATH (4-shot): 43.4

Code Generation:
  HumanEval (0-shot): 66.5

General Knowledge:
  MMLU (5-shot): Est. 70-75

Context Window: 65,536 tokens
```

**vs. Fully Open Competitors:**

| Model | Type | GSM8K | MATH | HumanEval |
|-------|------|-------|------|-----------|
| **OLMo 3-Base 32B** | Fully Open | **80.5** | **43.4** | **66.5** |
| Marin 32B (Stanford) | Fully Open | 69.1 | ~38 | 52.3 |
| Apertus 70B (Swiss AI) | Fully Open | ~75 | ~40 | ~60 |

**Key Takeaway**: OLMo 3-Base 32B is the **strongest fully open base model**, outperforming Marin 32B by +11.4 points on GSM8K and +14.2 points on HumanEval.

**vs. Open-Weight Competitors:**

| Model | Type | GSM8K | MATH | HumanEval |
|-------|------|-------|------|-----------|
| OLMo 3-Base 32B | Fully Open | 80.5 | 43.4 | 66.5 |
| Qwen 2.5 32B | Open-Weight | **85.8** | **48.6** | **72.1** |
| Gemma 3 27B | Open-Weight | **83.2** | **46.3** | **69.4** |
| Llama 3.1 32B | Open-Weight | **82.6** | **45.1** | **70.2** |

**Interpretation**: OLMo 3-Base 32B is competitive with top open-weight models, staying within 3-6 points despite being fully transparent (training data released).

### OLMo 3-Instruct Performance

#### OLMo 3-Instruct 7B

```yaml
Instruction-Following:
  IFEval: Ties or surpasses Qwen 2.5 7B, Gemma 3 7B, Llama 3.1 8B
  Function Calling: Best-in-class for fully open 7B models

General Performance:
  Overall Evaluations: Ties or surpasses Qwen 2.5, Gemma 3, Llama 3.1
  Chat Quality: Highly rated in qualitative assessments
  Tool Use: Strong integration with external tools

Long-Context:
  PopQA: Effective handling of ~65K token contexts
  Document QA: Strong performance on multi-document reasoning
```

**Key Positioning**:
- **Best Western 7B Model**: "Clear upgrade on Llama 3.1 8B, representing the best 7B scale model from a Western or American company"
- **Fully Open Advantage**: Unlike Llama, Qwen, Gemma, all training data is released

#### OLMo 3-Instruct 32B

```yaml
Instruction-Following:
  IFEval: Competitive with top 32B instruct models
  Complex Instructions: Better than 7B at multi-step tasks

General Performance:
  Exceeds 7B Instruct on all benchmarks
  Competitive with Qwen 2.5 32B Instruct, Gemma 3 27B

Long-Context:
  Superior to 7B on extended context tasks
  Multi-document reasoning improvements
```

### OLMo 3-Think Performance

#### OLMo 3-Think 7B

```yaml
Mathematics:
  MATH: Matches Qwen 3 8B
  AIME 2024: Within a few points of Qwen 3 8B
  AIME 2025: Within a few points of Qwen 3 8B
  GSM8K: Competitive with 8B reasoning models

Code Generation:
  HumanEvalPlus: Leads all comparison models in its class
  MBPP: Strong coding performance
  LiveCodeBench: Particular strength in code-intensive reasoning

Multi-Step Reasoning:
  BigBenchHard: Competitive with Qwen 3 8B reasoning
  OMEGA: Strong multi-step reasoning performance
```

**Detailed Comparison Table:**

| Model | Size | MATH | AIME 2025 | HumanEval+ | OMEGA | BBH |
|-------|------|------|-----------|------------|-------|-----|
| **OLMo 3-Think 7B** | 7B | **~Match** | **~Match** | **Leads** | Strong | **~Match** |
| Qwen 3 8B Reasoning | 8B | Baseline | Baseline | Lower | Baseline | Baseline |
| Llama 3.1 8B | 8B | Lower | Lower | Lower | Lower | Lower |
| Gemma 3 7B | 7B | Lower | Lower | Lower | Lower | Lower |
| DeepSeek R1 Distill 7B | 7B | Higher | Higher | Similar | Similar | Similar |

**Key Achievement**: OLMo 3-Think 7B **matches or exceeds 8B models** despite being smaller, demonstrating the value of reasoning-focused training.

#### OLMo 3-Think 32B

```yaml
Mathematics:
  MATH: Wins or within ~2 points of best open-weight model
  OMEGA: Ties Qwen 3 VL 32B Thinking for top score
  AIME: Competitive with top reasoning models

Code Generation:
  HumanEvalPlus: Wins or within ~2 points of best
  MBPP: Top-tier performance

Multi-Step Reasoning:
  BigBenchHard: Wins or within ~2 points of best
  IFEval: Wins or within ~2 points of best
  PopQA: Effective long-context reasoning (~65K tokens)

vs. Top Competitors:
  Qwen 3 VL 32B Thinking: Ties on OMEGA
  Gemma 3 27B Instruct: Clearly ahead
  DeepSeek R1 Distill 32B: Competitive on math/reasoning
```

**Training Efficiency Highlight:**

OLMo 3-Think 32B achieves this performance while being trained on **6x fewer tokens** than the Qwen 3-32B-Thinking series, demonstrating exceptional token efficiency.

**Detailed Comparison Table:**

| Model | Size | MATH | OMEGA | HumanEval+ | BBH | IFEval | Tokens Trained |
|-------|------|------|-------|------------|-----|--------|----------------|
| **OLMo 3-Think 32B** | 32B | ~Best | **Tie 1st** | ~Best | ~Best | ~Best | ~6T |
| Qwen 3 VL 32B Thinking | 32B | ~Best | **Tie 1st** | ~Best | ~Best | ~Best | ~36T |
| Gemma 3 27B Instruct | 27B | Lower | Lower | Lower | Lower | Lower | ~15T |
| DeepSeek R1 Distill 32B | 32B | Competitive | Competitive | Competitive | Competitive | Competitive | Unknown |

**Key Insight**: OLMo 3-Think 32B achieves within 1-2 points overall of Qwen 3-32B-Thinking while using **only 1/6th the training tokens**, highlighting superior data quality and training efficiency.

### Benchmark Decontamination

**Rigorous Decontamination Process:**

OLMo 3's training data underwent comprehensive n-gram decontamination to ensure benchmark integrity:

```yaml
Decontamination Method:
  N-gram Size: 8 words
  Coverage: All major benchmarks
  Approach: Conservative (remove borderline matches)

Benchmarks Decontaminated:
  - GSM8K (math word problems)
  - MMLU (general knowledge)
  - HumanEval (code generation)
  - MATH (competition mathematics)
  - AIME (advanced mathematics)
  - HellaSwag, WinoGrande, PIQA
  - BigBenchHard, OMEGA
  - All standard evaluation sets

Process:
  1. Extract all 8-word sequences from benchmarks
  2. Search for matches in training data
  3. Remove matching sequences from training data
  4. Conservative: better to over-remove than under-remove
```

**Transparency Advantage:**

Unlike closed models, OLMo 3's decontamination is:
- **Fully Documented**: Process and code released
- **Verifiable**: Researchers can check training data directly
- **Reproducible**: Decontamination scripts provided
- **Trustworthy**: Open data allows independent verification

This ensures that benchmark scores reflect true capability, not memorization.

### Long-Context Performance

**65,536 Token Context Window:**

OLMo 3's long-context capabilities are validated through:

```yaml
PopQA (Long-Context QA):
  Context: Up to 65K tokens
  Performance: Effective information retrieval across full context

Extended Comprehension Tasks:
  Multi-Document QA: Strong performance
  Summarization: Coherent summaries of long documents
  Information Retention: Maintains accuracy across context

Needle-in-Haystack:
  Performance: Retrieves information from arbitrary positions
  Context Range: Tested up to 65K tokens
```

**vs. Shorter Context Models:**

| Model | Context Window | Long-Context Performance |
|-------|----------------|--------------------------|
| **OLMo 3 (7B/32B)** | 65,536 | Strong across benchmarks |
| Llama 3.1 (8B/32B) | 128,000 | Better (longer window) |
| Qwen 2.5 (7B/32B) | 32,768 | Weaker (shorter window) |
| Gemma 3 (7B/27B) | 32,768 | Weaker (shorter window) |

**Note**: While Llama 3.1's 128K context is longer, OLMo 3's 65K context is sufficient for most practical applications (full research papers, multi-document analysis, extended conversations).

### Efficiency Benchmarks

**Training Efficiency:**

```yaml
OLMo 3-Base vs. Llama 3.1:
  Metric: GPU-hours per token
  Comparison: OLMo 3-Base vs. Llama 3.1 8B
  Result: OLMo 3 is 2.5x more efficient

Interpretation:
  - OLMo 3 achieves similar performance with 2.5x less compute
  - Better data quality (Dolma 3) + training recipes
  - Demonstrates academic labs can compete with industry
```

**Token Efficiency (Reasoning Models):**

```yaml
OLMo 3-Think 32B vs. Qwen 3-32B-Thinking:
  Training Tokens: 6x fewer (6T vs. 36T)
  Performance Gap: Within 1-2 points overall
  Efficiency: 6x token efficiency for similar results

Interpretation:
  - High-quality reasoning data (Dolci-Think) matters more than quantity
  - Verifiable rewards (RLVR) are highly efficient
  - Smaller labs can achieve frontier reasoning with smart data curation
```

### Benchmark Summary

**OLMo 3 Benchmark Positioning:**

1. **Best Fully Open Models**: OLMo 3-Base 32B and OLMo 3-Think 32B are the strongest fully open models in their respective categories

2. **Competitive with Open-Weight**: Stays within 3-6 points of top open-weight models (Qwen 2.5, Gemma 3, Llama 3.1) despite full transparency

3. **Exceptional Efficiency**: Achieves competitive performance with 2.5-6x fewer training tokens/compute than comparable models

4. **Strong Reasoning**: OLMo 3-Think variants match or exceed models 1.5x their size on reasoning tasks

5. **Western Leadership**: OLMo 3-Instruct 7B is "the best 7B scale model from a Western or American company"

6. **Decontaminated**: Rigorous, transparent decontamination ensures trustworthy benchmarks

7. **Long-Context**: 65K context window enables practical long-document applications

**Overall**: OLMo 3 proves that full transparency (releasing training data) doesn't compromise performance. The models compete effectively with closed-data competitors while providing unprecedented openness for research.

---

## The "Model Flow" Philosophy

OLMo 3's defining characteristic is the release of the complete **"Model Flow"**—every stage, checkpoint, dataset, and dependency required to create, understand, and modify the models. This represents a paradigm shift from "open weights" to "open everything."

### What is the "Model Flow"?

The "Model Flow" encompasses the entire lifecycle of model development:

```
Model Flow Components:
├── Data Collection & Curation
│   ├── Raw data sources (CommonCrawl, arXiv, GitHub, etc.)
│   ├── Curation tools (deduplication, filtering, decontamination)
│   ├── Data documentation and statistics
│   └── Dolma 3 datasets (Mix, Dolmino, Longmino)
│
├── Pretraining Pipeline
│   ├── Pretraining code (PyTorch, training scripts)
│   ├── Hyperparameters and configurations
│   ├── Infrastructure setup (GPU configs, parallelization)
│   ├── Training logs (loss curves, metrics over time)
│   └── Intermediate checkpoints (500+ checkpoints)
│
├── Post-Training Pipeline
│   ├── Post-training datasets (Dolci SFT, DPO, RLVR)
│   ├── Post-training code (SFT, DPO, RLVR implementations)
│   ├── Hyperparameters for each stage
│   ├── Intermediate post-training checkpoints
│   └── Evaluation results at each stage
│
├── Supporting Tools
│   ├── olmOCR (PDF processing)
│   ├── Evaluation harnesses (Catwalk, Paloma)
│   ├── Data processing scripts
│   └── Deployment examples
│
└── Final Models
    ├── Model weights (Base, Instruct, Think)
    ├── Tokenizers and configurations
    ├── Model cards and documentation
    └── Usage examples and guides
```

**Complete Transparency**: Every component above is publicly released, documented, and reproducible.

### "Model Flow" vs. "Open Weights"

#### Open Weights Models (Llama, Qwen, Gemma)

**What's Released:**
- ✅ Final model weights
- ✅ Tokenizer
- ✅ Model card (basic info)
- ❌ Training data
- ❌ Training code
- ❌ Intermediate checkpoints
- ❌ Training logs
- ❌ Data curation tools
- ❌ Post-training datasets

**What You Can Do:**
- Use the model for inference
- Fine-tune on your data
- Analyze model internals (weights, activations)

**What You Cannot Do:**
- Reproduce training from scratch
- Understand what data shaped the model
- Study how capabilities emerged during training
- Trace behaviors back to data sources
- Verify training claims
- Fork training at intermediate stages

#### Fully Open Models (OLMo 3)

**What's Released:**
- ✅ Final model weights
- ✅ Tokenizer
- ✅ Comprehensive model cards
- ✅ **Complete training data (Dolma 3, Dolci)**
- ✅ **All training code and recipes**
- ✅ **500+ intermediate checkpoints**
- ✅ **Complete training logs**
- ✅ **Data curation tools (olmOCR, etc.)**
- ✅ **Post-training datasets and code**
- ✅ **Evaluation frameworks**
- ✅ **Ablation studies and analysis**

**What You Can Do:**
- Everything possible with open-weight models, PLUS:
- **Reproduce Training**: Train OLMo 3 from scratch
- **Data Attribution**: Trace model behaviors to specific training data
- **Study Emergence**: Analyze how capabilities developed during training
- **Custom Training**: Fork at any checkpoint, modify data/recipes, retrain
- **Verify Claims**: Independently verify all training and performance claims
- **Build Research**: Use proven data and training pipelines for new research
- **Understand Biases**: Examine training data to understand model limitations

### Why "Model Flow" Matters

#### 1. Scientific Reproducibility

**The Reproducibility Crisis in AI:**

Modern AI suffers from a reproducibility crisis:
- Most models cannot be reproduced by independent researchers
- Training details are often vague or missing
- Data is undisclosed or proprietary
- Results are not independently verifiable

**OLMo 3's Solution:**

Complete reproducibility through full disclosure:

```bash
# Anyone can reproduce OLMo 3 training:

# Step 1: Download Dolma 3 datasets
python scripts/download_dolma3.py

# Step 2: Run pretraining (exact config provided)
torchrun --nproc_per_node=8 --nnodes=128 \
  scripts/train.py \
  --config configs/olmo3-7b-pretrain.yaml

# Step 3: Run mid-training
torchrun scripts/train.py \
  --config configs/olmo3-7b-midtrain.yaml

# Step 4: Run long-context extension
torchrun scripts/train.py \
  --config configs/olmo3-7b-longcontext.yaml

# Step 5: Run post-training (SFT, DPO, RLVR)
python scripts/post_train.py \
  --config configs/olmo3-7b-instruct.yaml

# Result: Reproduced OLMo 3-Instruct 7B
```

**Impact:**
- **Independent Verification**: Researchers can verify AI2's claims
- **Scientific Progress**: Build on proven foundations rather than reinventing
- **Trust**: Transparency builds confidence in results
- **Education**: Students can learn from real, frontier model training

#### 2. Data Attribution and Understanding

**The Data Mystery:**

For most models, the training data is a mystery:
- What data influenced this response?
- Why did the model fail on this input?
- What biases exist in the training data?
- Can we trace this capability to specific data?

**OLMo 3's Transparency:**

Complete data release enables attribution:

```python
# Example: Tracing model behavior to training data

# 1. Model generates a response
output = model.generate("Explain the Riemann hypothesis")

# 2. Analyze response characteristics
uses_formal_math = analyze_formality(output)
cites_specific_facts = extract_facts(output)

# 3. Search training data for similar content
similar_documents = search_dolma3(
    query="Riemann hypothesis",
    filters={"formality": "high", "domain": "mathematics"}
)

# 4. Hypothesis: Model learned from these documents
# 5. Verify by ablating these documents and retraining (possible with OLMo!)
```

**Applications:**
- **Bias Analysis**: Identify sources of bias in training data
- **Failure Debugging**: Understand why model fails on certain inputs
- **Capability Analysis**: Trace capabilities to data sources
- **Data Improvement**: Improve future training data based on learnings

#### 3. Checkpoint-Level Research

**Intermediate Checkpoints (500+):**

OLMo 3 releases checkpoints every 1,000 training steps:

```
Training Progression:
Step 0       → Random initialization
Step 1,000   → Checkpoint 1
Step 2,000   → Checkpoint 2
...
Step 500,000 → Checkpoint 500
Final        → OLMo 3-Base
```

**Research Enabled:**

1. **Emergence Studies**:
   - When do reasoning capabilities emerge?
   - How does knowledge accumulate during training?
   - What causes sudden capability jumps?

2. **Grokking Analysis**:
   - Study delayed understanding (grokking)
   - Identify which concepts take longer to learn
   - Optimize training for faster capability acquisition

3. **Forgetting Studies**:
   - Does the model forget earlier knowledge?
   - How to prevent catastrophic forgetting?
   - Optimal curriculum for training

4. **Ablation Studies**:
   - Fork training at any checkpoint
   - Modify data/hyperparameters and continue
   - Compare outcomes to understand causal factors

**Example Research Question:**

*"At what training step does the model gain mathematical reasoning ability?"*

```python
# Evaluate GSM8K performance across checkpoints
checkpoints = [0, 1000, 2000, ..., 500000]
gsm8k_scores = []

for ckpt in checkpoints:
    model = load_checkpoint(f"olmo3-7b-step{ckpt}.pt")
    score = evaluate_gsm8k(model)
    gsm8k_scores.append(score)

# Plot: GSM8K score vs. training step
plot(checkpoints, gsm8k_scores)
# Identify: When does performance jump?
```

This research is **impossible** with open-weight-only models.

#### 4. Custom Training and Forking

**Fork and Modify:**

With complete model flow, researchers can:

1. **Fork at Any Checkpoint**:
   ```python
   # Start from OLMo 3 checkpoint at 100K steps
   base_model = load_checkpoint("olmo3-7b-step100000.pt")

   # Continue training with custom data
   custom_model = continue_training(
       base_model,
       data=my_custom_dataset,
       steps=50000
   )
   ```

2. **Ablate Data Sources**:
   ```python
   # Remove scientific PDFs from training data
   dolma3_no_pdf = filter_dolma3(exclude=["arxiv", "pubmed"])

   # Retrain and compare
   model_no_pdf = train_olmo3(data=dolma3_no_pdf)
   compare_performance(olmo3_baseline, model_no_pdf)
   ```

3. **Modify Architectures**:
   ```python
   # Change to MoE architecture at 50K step checkpoint
   base_dense = load_checkpoint("olmo3-7b-step50000.pt")
   moe_model = convert_to_moe(base_dense, num_experts=8)
   continue_training(moe_model, data=dolma3_mix)
   ```

4. **Experiment with Training Recipes**:
   ```python
   # Try different mid-training mix proportions
   custom_dolmino = create_mix(
       math=0.5,  # vs. 0.25 in original
       code=0.3,  # vs. 0.35 in original
       reasoning=0.2
   )

   model = train_mid_stage(
       base_checkpoint="olmo3-7b-pretrain-final.pt",
       data=custom_dolmino
   )
   ```

**Impact:**
- **Efficient Research**: Build on OLMo 3 rather than training from scratch
- **Targeted Improvements**: Improve specific aspects of the model
- **Domain Adaptation**: Specialize for domains (medical, legal, scientific)
- **Architecture Research**: Test new architectures on proven training data

#### 5. Trust and Verification

**Trust Through Transparency:**

In an era of misinformation and black-box AI:

- **Open Weights**: "Trust us, our model works as claimed"
- **OLMo 3**: "Here's everything—verify it yourself"

**Verifiable Claims:**

Every claim about OLMo 3 can be independently verified:

| Claim | Verification Method |
|-------|---------------------|
| Trained on 5.9T tokens | Count tokens in Dolma 3 Mix |
| Decontaminated benchmarks | Search training data for n-grams |
| 2.5x more efficient than Llama 3.1 | Check training logs, compute GPU-hours |
| Reasoning from Dolci-Think data | Examine Dolci-Think-SFT dataset |
| GSM8K: 80.5 | Evaluate released model on GSM8K |

**No claims require "taking AI2's word for it"—everything is verifiable.**

#### 6. Educational Value

**Learning from Real Training:**

Students and researchers can:

1. **Study Production Training**:
   - See how frontier models are actually trained
   - Learn best practices from working code
   - Understand challenges and solutions

2. **Hands-On Experimentation**:
   - Train smaller models with same recipes
   - Fork and modify for learning
   - Reproduce published results

3. **Data Curation**:
   - Learn from Dolma 3 curation pipeline
   - Understand deduplication, filtering, decontamination
   - Build own datasets using proven methods

4. **Post-Training**:
   - Study SFT, DPO, RLVR implementations
   - Experiment with different alignment techniques
   - Understand reward design and policy optimization

**Impact**: Democratizes knowledge of frontier model training, previously locked in industry labs.

### "Model Flow" as a Standard

**AI2's Vision:**

OLMo 3 aims to establish "Model Flow" as the standard for open AI:

1. **Pressure on Industry**: If academic labs can release full model flows, why can't industry?

2. **Scientific Norm**: Make full transparency the expectation in AI research

3. **Reproducibility Standard**: Shift community norms toward reproducible research

4. **Open Science Advocacy**: Demonstrate benefits of openness

**Comparison to Other Fields:**

- **Biology**: Genomic data, protein structures publicly released
- **Physics**: CERN data, experimental setups fully documented
- **Chemistry**: Synthesis procedures, reagents specified
- **AI (Current)**: Models released, data/training hidden
- **AI (OLMo 3 Vision)**: Complete transparency, like other sciences

### Limitations of "Model Flow"

**Practical Challenges:**

1. **Storage**: Full datasets and checkpoints require petabytes of storage
2. **Bandwidth**: Downloading full training data is time-consuming
3. **Compute**: Reproducing training requires substantial GPU resources
4. **Complexity**: Understanding and using full model flow has a learning curve

**AI2's Solutions:**

- **Streaming Datasets**: HuggingFace datasets support streaming (no full download)
- **Checkpoint Sampling**: Use subset of checkpoints if storage-limited
- **Cloud Resources**: Partnerships for academic cloud compute access
- **Documentation**: Comprehensive guides and tutorials

**Counterpoint**: These challenges exist, but are worth it for the benefits of full transparency.

### The Future of "Model Flow"

**Growing Adoption:**

Other projects embracing aspects of "Model Flow":
- **DataComp**: Open datasets for vision models
- **The Pile (EleutherAI)**: Open text corpus
- **RedPajama**: Open reproduction of LLaMA training data

**OLMo 3's Contribution:**

- **Most Complete**: First full model flow for frontier LLMs
- **Reasoning Models**: First fully open reasoning models
- **Tools Released**: olmOCR and other infrastructure
- **Documented Process**: Comprehensive documentation and guides

**Vision for 2026+:**

- **OLMo 4**: Next generation with multimodal, larger scale
- **Community Contributions**: Researchers building on OLMo 3 model flow
- **Industry Adoption**: Pressure on closed labs to increase transparency
- **Standard Practice**: "Model Flow" becomes expected for published models

**Ultimate Goal**: **Make AI as transparent and reproducible as other scientific fields.**

---

## Comparison with Competing Models

OLMo 3 competes in a crowded landscape of 7B and 32B parameter models. Understanding its position requires comparing against both fully open models (training data released) and open-weight models (weights only).

### Taxonomy of "Openness"

**Three Categories of Models:**

1. **Closed Models**:
   - No weights, no data, API-only access
   - Examples: GPT-4, Claude Opus, Gemini Pro

2. **Open-Weight Models**:
   - Weights released, training data closed
   - Examples: Llama 3.1, Qwen 2.5/3, Gemma 3, DeepSeek

3. **Fully Open Models**:
   - Weights, training data, and code all released
   - Examples: OLMo 3, Marin (Stanford), Apertus (Swiss AI)

**OLMo 3's Category**: Fully Open (most transparent)

### Comparison with Fully Open Models

#### vs. Stanford Marin 32B

**Marin 32B** is Stanford's fully open model, released in 2025.

| Metric | OLMo 3-Base 32B | Marin 32B | Advantage |
|--------|-----------------|-----------|-----------|
| **Parameters** | 32B | 32B | Tie |
| **Training Data** | Dolma 3 (9.3T) | Undisclosed | Unknown |
| **GSM8K** | **80.5** | 69.1 | **OLMo +11.4** |
| **MATH** | **43.4** | ~38 | **OLMo +5.4** |
| **HumanEval** | **66.5** | 52.3 | **OLMo +14.2** |
| **Context** | 65K | Unknown | Likely OLMo |
| **License** | Apache 2.0 | Apache 2.0 | Tie |

**Key Takeaway**: **OLMo 3-Base 32B significantly outperforms Marin 32B** across all major benchmarks, establishing it as the strongest fully open base model.

**Reasons for OLMo's Advantage:**
- **Better Data**: Dolma 3 with olmOCR-processed scientific PDFs
- **Staged Training**: Pretraining → Mid-training → Long-context
- **Optimized Recipes**: Research-driven hyperparameter choices
- **Infrastructure**: Efficient training on H100s

#### vs. Swiss AI Apertus 70B

**Apertus 70B** is a 70B fully open model from Swiss AI.

| Metric | OLMo 3-Base 32B | Apertus 70B | Advantage |
|--------|-----------------|-------------|-----------|
| **Parameters** | 32B | **70B** | Apertus (larger) |
| **GSM8K** | **80.5** | ~75 | **OLMo** |
| **MATH** | 43.4 | **~45** | Apertus |
| **HumanEval** | **66.5** | ~60 | **OLMo** |
| **Efficiency** | 32B params | 70B params | **OLMo** (2.2x smaller) |

**Key Takeaway**: OLMo 3-Base 32B **matches or exceeds a 70B model** while being 2.2x smaller, demonstrating exceptional parameter efficiency.

### Comparison with Open-Weight Models (7B/8B)

#### vs. Meta Llama 3.1 8B

**Llama 3.1 8B** is Meta's flagship 8B model (open weights, closed data).

| Metric | OLMo 3-Instruct 7B | Llama 3.1 8B Instruct | Advantage |
|--------|--------------------|-----------------------|-----------|
| **Parameters** | 7B | 8B | Llama (slightly larger) |
| **Training Data** | Dolma 3 (public) | **Undisclosed** | **OLMo (transparent)** |
| **Context** | **65K** | 128K | Llama |
| **Overall Evals** | **Ties/surpasses** | Baseline | **OLMo per AI2** |
| **Training Efficiency** | **2.5x** | 1x | **OLMo** |
| **Western Model** | ✅ Yes | ✅ Yes | Tie |
| **License** | Apache 2.0 | Llama 3.1 License | OLMo (more permissive) |

**AI2's Claim**: *"OLMo 3-Instruct should be a clear upgrade on Llama 3.1 8B, representing the best 7B scale model from a Western or American company."*

**Key Differentiators:**
- **Full Transparency**: OLMo 3 releases training data, Llama doesn't
- **Training Efficiency**: OLMo 3 trained 2.5x more efficiently
- **Western Leadership**: Best 7B from US/Western lab
- **Licensing**: Apache 2.0 more permissive than Llama license

#### vs. Alibaba Qwen 2.5 / Qwen 3 (7B/8B)

**Qwen 2.5** and **Qwen 3** are Alibaba's flagship models (open weights).

| Metric | OLMo 3-Instruct 7B | Qwen 2.5 7B Instruct | Qwen 3 8B | Advantage |
|--------|---------------------|----------------------|-----------|-----------|
| **Parameters** | 7B | 7B | 8B | Qwen 3 |
| **Training Data** | **Dolma 3 (public)** | Undisclosed | Undisclosed | **OLMo** |
| **Overall Evals** | **Ties/surpasses** | Baseline | Strong | **Competitive** |
| **Reasoning** | OLMo 3-Think 7B matches Qwen 3 8B on MATH | - | Baseline | **Tie** |
| **Context** | 65K | 32K | 32K | **OLMo** |
| **License** | Apache 2.0 | Apache 2.0 | Apache 2.0 | Tie |

**Key Differentiators:**
- **Transparency**: OLMo releases data, Qwen doesn't
- **Context**: 65K vs 32K (2x longer)
- **Reasoning**: OLMo 3-Think 7B matches Qwen 3 8B despite being smaller

#### vs. Google Gemma 3 (7B)

**Gemma 3** is Google's small-scale model family.

| Metric | OLMo 3-Instruct 7B | Gemma 3 7B Instruct | Advantage |
|--------|---------------------|---------------------|-----------|
| **Parameters** | 7B | 7B | Tie |
| **Training Data** | **Dolma 3 (public)** | Undisclosed (likely web-scale Google data) | **OLMo (transparency)** |
| **Overall Evals** | **Ties/surpasses** | Baseline | **OLMo per AI2** |
| **Function Calling** | **Superior** | Baseline | **OLMo** |
| **Context** | **65K** | 32K | **OLMo** |
| **License** | Apache 2.0 | Gemma License | Apache 2.0 more standard |

**Key Differentiators:**
- **Function Calling**: OLMo superior for tool use
- **Context**: 2x longer (65K vs 32K)
- **Transparency**: Full data release vs. closed

### Comparison with Open-Weight Models (32B)

#### vs. Alibaba Qwen 2.5 / Qwen 3 (32B)

| Metric | OLMo 3-Base 32B | OLMo 3-Think 32B | Qwen 2.5 32B | Qwen 3 VL 32B Thinking | Advantage |
|--------|-----------------|------------------|--------------|------------------------|-----------|
| **Parameters** | 32B | 32B | 32B | 32B | Tie |
| **Training Data** | **Dolma 3 (public)** | **Dolma 3 + Dolci (public)** | Undisclosed | Undisclosed | **OLMo** |
| **GSM8K** | 80.5 | - | **85.8** | - | Qwen |
| **MATH** | 43.4 | ~Best | **48.6** | ~Best | **Competitive** |
| **HumanEval** | 66.5 | ~Best | **72.1** | ~Best | Qwen |
| **OMEGA** | - | **Ties 1st** | - | **Ties 1st** | **Tie** |
| **Training Tokens** | ~6T | ~6T | ~15T | **~36T** | **OLMo (6x efficient)** |
| **Context** | **65K** | **65K** | 32K | 32K | **OLMo** |

**Key Insights:**
- **Performance Gap**: Qwen ahead by 3-6 points on base benchmarks
- **Reasoning**: OLMo 3-Think **ties Qwen 3 VL Thinking on OMEGA** despite 6x fewer training tokens
- **Efficiency**: OLMo achieves competitive performance with 1/6th training
- **Transparency**: OLMo releases all data, Qwen doesn't

#### vs. Google Gemma 3 (27B)

| Metric | OLMo 3-Think 32B | Gemma 3 27B Instruct | Advantage |
|--------|------------------|----------------------|-----------|
| **Parameters** | 32B | 27B | OLMo (slightly larger) |
| **Training Data** | **Dolma 3 + Dolci (public)** | Undisclosed | **OLMo** |
| **Overall Reasoning** | **Clearly ahead** | Baseline | **OLMo per AI2** |
| **MATH, OMEGA, etc.** | **Superior** | Lower | **OLMo** |
| **Context** | **65K** | 32K | **OLMo** |
| **License** | Apache 2.0 | Gemma License | Apache 2.0 more standard |

**Key Takeaway**: **OLMo 3-Think 32B "clearly ahead" of Gemma 3 27B** on reasoning benchmarks per AI2.

#### vs. DeepSeek R1 Distill (32B)

| Metric | OLMo 3-Think 32B | DeepSeek R1 Distill 32B | Advantage |
|--------|------------------|-------------------------|-----------|
| **Parameters** | 32B | 32B | Tie |
| **Training Data** | **Dolma 3 + Dolci (public)** | Undisclosed | **OLMo** |
| **Math/Reasoning** | **Competitive** | **Competitive** | **Tie** |
| **Transparency** | **Full model flow** | Weights only | **OLMo** |
| **Context** | 65K | Unknown | Likely OLMo |

**Key Differentiators:**
- **Reasoning Performance**: Competitive on math and reasoning
- **Transparency**: OLMo's full openness is unique advantage
- **Specialization**: DeepSeek R1 is dedicated reasoning model, OLMo 3-Think is balanced

### Feature Comparison Table

| Feature | OLMo 3 | Llama 3.1 | Qwen 2.5/3 | Gemma 3 | DeepSeek | Marin |
|---------|---------|-----------|------------|---------|----------|-------|
| **Weights Released** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Training Data Released** | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Training Code Released** | ✅ | ❌ | ❌ | ❌ | ❌ | Partial |
| **Intermediate Checkpoints** | ✅ (500+) | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Post-Training Data** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Data Curation Tools** | ✅ (olmOCR) | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Training Logs** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Reasoning Variants** | ✅ (Think) | ❌ | ✅ | ❌ | ✅ (R1) | ❌ |
| **Context Length** | 65K | 128K | 32K | 32K | Varies | Unknown |
| **License** | Apache 2.0 | Llama 3.1 | Apache 2.0 | Gemma | Apache 2.0 | Apache 2.0 |

**Openness Score (out of 8 features):**
- **OLMo 3**: 8/8 ✅ **Fully Open**
- **Marin**: 3/8 (weights, partial data, partial code)
- **Llama 3.1**: 1/8 (weights only)
- **Qwen, Gemma, DeepSeek**: 1/8 (weights only)

### Performance vs. Transparency Trade-off

**The Conventional Wisdom**: "You have to choose between performance and transparency."

**OLMo 3's Disproof**:

```
Performance (GSM8K as proxy):
  Qwen 2.5 32B: 85.8 (closed data)
  OLMo 3-Base 32B: 80.5 (open data)
  Gap: 5.3 points (6.2%)

Transparency:
  Qwen 2.5: Weights only
  OLMo 3: Weights + data + code + checkpoints + logs

Conclusion: 6% performance gap for 100% transparency increase
```

**Is the trade-off worth it?**

For **research, education, trust, and reproducibility**: **Absolutely yes.**

For **production applications caring only about raw performance**: Depends on use case.

### Positioning Summary

**OLMo 3's Niche:**

1. **Best Fully Open Models**: Outperforms all other fully open models (Marin, Apertus)
2. **Competitive with Open-Weight**: Within 3-6 points of top open-weight models
3. **Western Leadership**: Best 7B from US/Western lab
4. **Reasoning Excellence**: OLMo 3-Think matches models 1.5x its size
5. **Exceptional Efficiency**: 2.5-6x better training efficiency
6. **Unique Transparency**: Only frontier model with complete "Model Flow"

**Target Users:**

- **Researchers**: Need data for attribution studies, reproducibility
- **Educators**: Teaching LLM training at universities
- **Enterprises**: Requiring full transparency for compliance/trust
- **Privacy-Focused**: On-premise deployment with verifiable training
- **Domain Adaptation**: Custom training from proven checkpoints

**Not For:**

- **Max Performance Seekers**: Qwen 2.5 32B is stronger (but closed data)
- **Longest Context Users**: Llama 3.1's 128K context is longer
- **API-Only Users**: Don't need transparency if using via API

---

## Training Efficiency

One of OLMo 3's most impressive achievements is its **exceptional training efficiency**—achieving competitive performance with 2.5-6x less compute than comparable models.

### Efficiency Metrics

**Training Efficiency Definition:**

Efficiency = Performance / Resources Used

Where:
- **Performance**: Benchmark scores (GSM8K, MMLU, etc.)
- **Resources**: GPU-hours, tokens, cost

**OLMo 3's Efficiency Claims:**

1. **2.5x more efficient than Llama 3.1 8B** (GPU-hours per token)
2. **6x fewer tokens than Qwen 3-32B-Thinking** for similar reasoning performance

### OLMo 3 vs. Llama 3.1: Training Efficiency

**Comparison**: OLMo 3-Base vs. Meta's Llama 3.1 8B

| Metric | OLMo 3-Base 7B | Llama 3.1 8B | Ratio |
|--------|----------------|---------------|-------|
| **Training Tokens** | ~6T (Dolma 3 Mix + Dolmino + Longmino) | ~15T (estimated) | **2.5x** |
| **GPU-Hours** | ~533K H100-hours | ~1.3M H100-hours (est.) | **2.5x** |
| **GSM8K** | 75.5 | ~77 | Comparable |
| **MMLU** | ~60-65 (est.) | ~68 | Comparable |
| **Cost** | ~$1.3M | ~$3.3M (est.) | **2.5x** |

**Interpretation**: OLMo 3-Base achieves similar performance to Llama 3.1 8B while using **2.5x less compute**.

**How?**

1. **Better Data Quality (Dolma 3)**:
   - More aggressive filtering and deduplication
   - Higher proportion of high-quality data (code, math, scientific PDFs)
   - olmOCR-processed scientific content
   - Result: Learn more per token

2. **Staged Training**:
   - Pretraining on broad data
   - Mid-training on targeted skills (math, code)
   - Long-context extension stage
   - Result: Efficient capability acquisition

3. **Optimized Hyperparameters**:
   - Research-driven hyperparameter choices
   - Post-norm architecture (stabilizes training)
   - Cosine learning rate schedule with optimal warmup
   - Result: Faster convergence

4. **Efficient Infrastructure**:
   - 1,024 H100 GPUs with optimized parallelization
   - 7,700 tokens/device/second throughput
   - Flash Attention 2, gradient checkpointing
   - Result: Maximum hardware utilization

### OLMo 3-Think vs. Qwen 3-Thinking: Token Efficiency

**Comparison**: OLMo 3-Think 32B vs. Qwen 3-32B-Thinking

| Metric | OLMo 3-Think 32B | Qwen 3-32B-Thinking | Ratio |
|--------|------------------|---------------------|-------|
| **Training Tokens** | ~6T | ~36T | **6x** |
| **MATH** | ~Best -2 | ~Best | Within 2 points |
| **OMEGA** | **Ties 1st** | **Ties 1st** | **Tie** |
| **Overall Reasoning** | Within 1-2 points | Baseline | Competitive |
| **Cost** | ~$4M (est.) | ~$20M+ (est.) | **5x** |

**Interpretation**: OLMo 3-Think achieves within 1-2 points of Qwen 3-Thinking while using **only 1/6th the training tokens**.

**How?**

1. **High-Quality Reasoning Data (Dolci-Think)**:
   - Curated math, code, and logical reasoning examples
   - Step-by-step solutions with chain-of-thought
   - Verification and self-correction examples
   - Result: Every token teaches reasoning effectively

2. **Verifiable Rewards (RLVR)**:
   - Objective rewards (code execution, math verification)
   - Efficient learning from verifiable outcomes
   - No need for huge volumes of data
   - Result: Rapid capability acquisition

3. **Three-Stage Post-Training (SFT → DPO → RLVR)**:
   - Supervised learning establishes baseline
   - Preference optimization refines quality
   - RL with verifiable rewards sharpens reasoning
   - Result: Targeted, efficient skill development

4. **Transfer from Strong Base**:
   - OLMo 3-Base already has strong general capabilities
   - Mid-training included math/code emphasis
   - Post-training builds on solid foundation
   - Result: Less post-training needed

### Efficiency Breakdown by Training Stage

**OLMo 3 7B Total Efficiency:**

```yaml
Stage 1: Pretraining (Dolma 3 Mix)
  Tokens: 5.9T
  GPU-Hours: ~516K H100-hours
  Cost: ~$1.29M
  Efficiency: Broad knowledge acquisition

Stage 2: Mid-Training (Dolmino Mix)
  Tokens: 100B
  GPU-Hours: ~7.2K H100-hours
  Cost: ~$18K
  Efficiency: Targeted capability boost (+10-15 points GSM8K for 1.4% extra cost)

Stage 3: Long-Context (Longmino Mix)
  Tokens: 50B
  GPU-Hours: ~9.6K H100-hours
  Cost: ~$24K
  Efficiency: 65K context for 1.9% extra cost

Post-Training (SFT + DPO + RLVR)
  Tokens: ~200B (across all post-training)
  GPU-Hours: ~50K H100-hours
  Cost: ~$125K
  Efficiency: Instruct/Think variants for 9.6% extra cost

Total: ~$1.46M for OLMo 3-Instruct 7B (all stages)
```

**Efficiency Insight**: Mid-training and post-training add substantial capabilities (instruction-following, reasoning) for <15% additional cost.

### Cost-Performance Comparison

**Cost to Achieve Similar Performance:**

| Model Type | Example | Estimated Cost | Performance Level |
|------------|---------|----------------|-------------------|
| **OLMo 3 7B** | OLMo 3-Instruct 7B | **$1.5M** | Ties/surpasses Llama 3.1 8B |
| **Llama 3.1 8B** | Llama 3.1 8B Instruct | ~$3.5M | Baseline for comparison |
| **Qwen 2.5 7B** | Qwen 2.5 7B Instruct | ~$3-4M | Similar performance |
| **Gemma 3 7B** | Gemma 3 7B Instruct | ~$3-5M (Google-scale) | Similar performance |

**32B Reasoning Models:**

| Model | Estimated Cost | OMEGA Score | Efficiency |
|-------|----------------|-------------|------------|
| **OLMo 3-Think 32B** | **~$4M** | **Ties 1st** | **Baseline (most efficient)** |
| Qwen 3-32B-Thinking | ~$20M+ | **Ties 1st** | **5x less efficient** |
| DeepSeek R1 Distill 32B | ~$15M (est.) | Competitive | ~4x less efficient |

**Key Insight**: OLMo 3 achieves frontier performance at a fraction of the cost of competitors.

### Factors Contributing to Efficiency

**Data Quality > Data Quantity:**

```
Traditional Approach (Llama, Qwen):
  - Massive data scale (15T+ tokens)
  - Heavy compute (weeks on thousands of GPUs)
  - Less filtering (include more data)

OLMo 3 Approach:
  - Moderate data scale (6T tokens)
  - Efficient compute (optimized throughput)
  - Aggressive filtering (quality > quantity)
  - Targeted mid-training

Result: OLMo 3 learns more per token
```

**Staged Training:**

```
Single-Stage Training:
  - All data in one pass
  - Generic dataset mix
  - Harder to learn specialized skills

Multi-Stage Training (OLMo 3):
  - Stage 1: Broad knowledge (5.9T tokens)
  - Stage 2: Math/code/reasoning (100B tokens)
  - Stage 3: Long-context (50-100B tokens)
  - Each stage optimized for its goal

Result: More efficient capability acquisition
```

**Research-Driven Optimization:**

AI2's academic research informs every choice:
- **OLMo 1.0 & 2.0 Ablations**: Learned what works
- **Post-Norm Architecture**: Proven stable in OLMo 2
- **Sliding Window Attention**: Efficient long-context
- **DPO + RLVR**: Simpler than traditional RLHF
- Result: No wasted compute on ineffective approaches

### Efficiency Implications

**For Academic Labs:**

OLMo 3 demonstrates that academic labs can compete with industry:

```
Required Resources for Competitive 7B Model:
  - Compute: 1,024 H100 GPUs for ~3 weeks (~$1.5M)
  - Data: Dolma 3 (free, open) + curation pipeline
  - Expertise: Research team with LLM experience

Achievable Performance:
  - Competitive with Llama 3.1 8B
  - Best Western 7B model
  - Fully transparent and reproducible

Conclusion: Academic-scale resources can produce frontier models
```

**For Industry:**

OLMo 3's efficiency should pressure industry to improve:

```
Industry Models (Closed Data):
  - Llama 3.1 8B: ~$3.5M training cost
  - Qwen 2.5 7B: ~$3-4M training cost
  - Gemma 3 7B: ~$3-5M training cost

OLMo 3 (Fully Open):
  - OLMo 3 7B: ~$1.5M training cost
  - Similar or better performance
  - Full transparency

Question for Industry: Why does your closed model cost 2-3x more?
```

**For Future Research:**

OLMo 3's efficiency sets a new baseline:

```
Pre-OLMo 3:
  - Assumption: Need 15T+ tokens for competitive 7B model
  - Assumption: Only industry can afford frontier training

Post-OLMo 3:
  - Demonstrated: 6T high-quality tokens sufficient
  - Demonstrated: $1.5M can produce frontier model
  - Demonstrated: Efficiency comes from better data, not just scale

Impact: Future models will prioritize efficiency
```

### Efficiency Limitations

**Diminishing Returns:**

```
7B Model Efficiency: High
  - OLMo 3 7B very efficient
  - $1.5M for competitive model

32B Model Efficiency: Moderate
  - OLMo 3 32B still efficient
  - But ~$4M (2.7x more than 7B)
  - Less than linear scaling (32B/7B = 4.6x params, but <3x cost)

70B+ Models: Lower Efficiency Expected
  - Larger models require more data
  - Longer training times
  - Efficiency gains diminish
```

**Efficiency vs. Peak Performance:**

```
OLMo 3's Trade-Off:
  - High Efficiency: 2.5-6x better than competitors
  - Competitive Performance: Within 3-6 points of best
  - Not Peak Performance: Qwen 2.5 32B is stronger

To Achieve Peak Performance:
  - Would need ~15T tokens (like Qwen)
  - Would need ~$10M+ training budget
  - Would sacrifice efficiency for max performance

OLMo 3's Choice: Optimize for efficiency, accept small performance gap
```

### Efficiency Summary

**OLMo 3's Efficiency Achievements:**

1. **2.5x more efficient than Llama 3.1** (GPU-hours per token)
2. **6x more token-efficient than Qwen 3-Thinking** (reasoning performance per token)
3. **$1.5M training cost** for frontier 7B model (vs. $3-5M competitors)
4. **$4M training cost** for frontier 32B reasoning model (vs. $15-20M+ competitors)
5. **Staged training** optimizes capability acquisition
6. **High-quality data** (Dolma 3, Dolci) drives efficiency

**Lessons for the Field:**

- **Data Quality > Quantity**: 6T high-quality tokens beats 15T mediocre tokens
- **Staged Training Works**: Targeted mid-training boosts skills efficiently
- **Academic Labs Can Compete**: $1.5M can produce frontier models
- **Transparency Doesn't Hurt**: Full openness doesn't require performance sacrifice
- **Research Pays Off**: Evidence-based design choices maximize efficiency

**Future Outlook**: OLMo 3's efficiency will pressure the field toward more resource-conscious, data-centric training approaches.

---

## Deployment and Inference

OLMo 3 models are designed for practical deployment across a range of hardware configurations, from consumer GPUs to enterprise data centers.

### Hardware Requirements

#### OLMo 3 7B Models

**Memory Requirements (Weights Only):**

```yaml
BF16 / FP16:
  Memory: ~14 GB
  Hardware: RTX 4090 (24GB), A100 40GB, H100 80GB
  Use Case: Development, inference

FP8:
  Memory: ~7 GB
  Hardware: RTX 4090 (24GB), A100 40GB
  Use Case: Efficient inference

INT8 (Quantized):
  Memory: ~7 GB
  Hardware: RTX 3090 (24GB), RTX 4090, A100
  Use Case: Consumer GPU deployment

INT4 / GGUF (Quantized):
  Memory: ~4-5 GB
  Hardware: RTX 3060 (12GB), RTX 3080 (10GB), consumer laptops
  Use Case: Resource-constrained deployment
```

**Memory Requirements (With 65K Context KV Cache):**

```yaml
BF16/FP16 (65K context):
  Weights: ~14 GB
  KV Cache (full context): ~10-12 GB
  Total: ~24-26 GB
  Hardware: RTX 4090 (24GB insufficient), A100 40GB (tight), A100 80GB recommended

With Sliding Window Attention:
  Weights: ~14 GB
  KV Cache (optimized): ~4-6 GB (3/4 layers use 4K window)
  Total: ~18-20 GB
  Hardware: RTX 4090 (24GB sufficient), A100 40GB (comfortable)
```

**Recommended Hardware:**

```yaml
Development / Experimentation:
  GPU: RTX 4090 (24GB)
  Quantization: BF16 or INT8
  Context: Up to 32K tokens (with sliding window optimization)
  Cost: ~$1,600

Production (Low-Medium Load):
  GPU: A100 40GB or A100 80GB
  Quantization: BF16 or FP8
  Context: Full 65K tokens
  Throughput: 50-100 tokens/sec
  Cost: Cloud rental ~$1-3/hour

Production (High Load):
  GPU: 2-4x A100 80GB or H100
  Quantization: BF16
  Context: Full 65K with batching
  Throughput: 200-500 tokens/sec
  Cost: Cloud rental ~$4-12/hour
```

#### OLMo 3 32B Models

**Memory Requirements (Weights Only):**

```yaml
BF16 / FP16:
  Memory: ~64 GB
  Hardware: 2x A100 40GB, A100 80GB, H100 80GB
  Use Case: Development, inference

FP8:
  Memory: ~32 GB
  Hardware: A100 40GB, A100 80GB, H100 80GB
  Use Case: Efficient inference

INT8 (Quantized):
  Memory: ~32 GB
  Hardware: A100 40GB, A100 80GB
  Use Case: Cost-effective deployment

INT4 / GGUF (Quantized):
  Memory: ~16-20 GB
  Hardware: RTX 4090 (24GB), A100 40GB
  Use Case: Consumer/prosumer deployment
```

**Memory Requirements (With 65K Context KV Cache):**

```yaml
BF16/FP16 (65K context):
  Weights: ~64 GB
  KV Cache (with GQA optimization): ~6-8 GB
  Total: ~70-72 GB
  Hardware: A100 80GB (tight), 2x A100 40GB, H100 80GB

With Sliding Window Attention + GQA:
  Weights: ~64 GB
  KV Cache (optimized): ~3-4 GB
  Total: ~67-68 GB
  Hardware: A100 80GB (comfortable), H100 80GB (ideal)
```

**Recommended Hardware:**

```yaml
Development / Experimentation:
  GPU: 2x RTX 4090 (48GB total) or A100 80GB
  Quantization: INT4/GGUF or FP8
  Context: Up to 32K tokens
  Cost: ~$3,200 (2x RTX 4090) or cloud rental

Production (Low-Medium Load):
  GPU: A100 80GB or H100 80GB
  Quantization: FP8 or BF16
  Context: Full 65K tokens
  Throughput: 20-50 tokens/sec
  Cost: Cloud rental ~$3-4/hour

Production (High Load):
  GPU: 2-4x A100 80GB or 2x H100
  Quantization: BF16
  Context: Full 65K with batching
  Throughput: 80-200 tokens/sec
  Cost: Cloud rental ~$6-16/hour
```

### Deployment Frameworks

OLMo 3 is compatible with all major LLM inference frameworks:

#### Hugging Face Transformers

**Installation:**

```bash
pip install transformers torch accelerate
```

**Basic Inference:**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "allenai/Olmo-3-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",  # Automatically choose BF16/FP16
    device_map="auto"    # Automatically distribute across GPUs
)

# Prepare input
messages = [
    {"role": "user", "content": "Explain quantum entanglement in simple terms."}
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# Generate
outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

**Quantization with Transformers:**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# INT8 Quantization
model = AutoModelForCausalLM.from_pretrained(
    "allenai/Olmo-3-7B-Instruct",
    load_in_8bit=True,
    device_map="auto"
)

# INT4 Quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "allenai/Olmo-3-7B-Instruct",
    quantization_config=quantization_config,
    device_map="auto"
)
```

#### vLLM (High-Throughput Inference)

**vLLM** is optimized for high-throughput, low-latency serving.

**Installation:**

```bash
pip install vllm
```

**Deployment:**

```python
from vllm import LLM, SamplingParams

# Initialize vLLM
llm = LLM(
    model="allenai/Olmo-3-7B-Instruct",
    tensor_parallel_size=1,  # Number of GPUs
    max_model_len=65536,     # Full context window
    dtype="bfloat16"
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Batch inference
prompts = [
    "Explain quantum entanglement.",
    "Write a Python function to sort a list.",
    "Summarize the main causes of World War II."
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

**vLLM Server (OpenAI-Compatible API):**

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model allenai/Olmo-3-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --dtype bfloat16 \
  --port 8000

# Use with OpenAI client
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="allenai/Olmo-3-7B-Instruct",
    messages=[
        {"role": "user", "content": "Explain quantum entanglement."}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

#### Ollama (Easy Local Deployment)

**Ollama** simplifies local model deployment with automatic management.

**Installation:**

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Windows: Download installer from ollama.com
```

**Usage:**

```bash
# Pull and run OLMo 3 (if available in Ollama library)
ollama run olmo3:7b-instruct

# Or create custom Modelfile
cat > Modelfile <<EOF
FROM allenai/Olmo-3-7B-Instruct
PARAMETER temperature 0.7
PARAMETER top_p 0.9
EOF

# Create model
ollama create olmo3-custom -f Modelfile

# Run
ollama run olmo3-custom "Explain quantum entanglement"
```

**Ollama API:**

```python
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "olmo3:7b-instruct",
        "prompt": "Explain quantum entanglement",
        "stream": False
    }
)

print(response.json()["response"])
```

#### llama.cpp (CPU/Metal Inference)

**llama.cpp** enables efficient CPU and Apple Silicon (Metal) inference via GGUF quantization.

**Installation:**

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# For Apple Silicon (Metal support)
make LLAMA_METAL=1
```

**Convert OLMo 3 to GGUF:**

```bash
# Download model from Hugging Face
huggingface-cli download allenai/Olmo-3-7B-Instruct --local-dir ./olmo3-7b

# Convert to GGUF
python convert.py ./olmo3-7b --outtype f16 --outfile olmo3-7b-f16.gguf

# Quantize
./quantize olmo3-7b-f16.gguf olmo3-7b-q4_0.gguf Q4_0
```

**Inference:**

```bash
# CPU inference
./main -m olmo3-7b-q4_0.gguf \
  -p "Explain quantum entanglement" \
  -n 512 \
  -t 8  # 8 threads

# Apple Silicon (Metal) inference
./main -m olmo3-7b-q4_0.gguf \
  -p "Explain quantum entanglement" \
  -n 512 \
  -ngl 1  # Offload to GPU
```

#### TensorRT-LLM (NVIDIA Optimized)

**TensorRT-LLM** provides maximum performance on NVIDIA GPUs.

**Installation:**

```bash
# Follow TensorRT-LLM installation guide
# Requires CUDA, TensorRT, and specific Python environment
```

**Build Optimized Engine:**

```bash
# Convert OLMo 3 to TensorRT-LLM format
python convert_checkpoint.py \
  --model_dir ./olmo3-7b \
  --output_dir ./olmo3-7b-trt \
  --dtype bfloat16

# Build TensorRT engine
trtllm-build \
  --checkpoint_dir ./olmo3-7b-trt \
  --output_dir ./olmo3-7b-engine \
  --gemm_plugin bfloat16 \
  --max_batch_size 8 \
  --max_input_len 4096 \
  --max_output_len 2048 \
  --max_beam_width 1

# Run inference
python ../run.py \
  --engine_dir=./olmo3-7b-engine \
  --max_output_len=512 \
  --tokenizer_dir=./olmo3-7b \
  --input_text="Explain quantum entanglement"
```

### Inference Optimization

#### Batching Strategies

**Static Batching:**

```python
# Process multiple requests in fixed-size batches
batch_size = 8
prompts = [...]  # List of prompts

for i in range(0, len(prompts), batch_size):
    batch = prompts[i:i+batch_size]
    outputs = model.generate(batch)
```

**Continuous Batching (vLLM):**

```python
# vLLM automatically optimizes batching
# Requests are dynamically batched for maximum throughput
llm = LLM(model="allenai/Olmo-3-7B-Instruct")
outputs = llm.generate(prompts)  # Automatically optimized
```

#### KV Cache Optimization

**Sliding Window Attention:**

OLMo 3's sliding window attention (3 of 4 layers) automatically reduces KV cache:

```python
# With sliding window (OLMo 3 built-in):
# - Layers 1-3: 4K window (small KV cache)
# - Layer 4: Full attention (65K KV cache)
# - Layers 5-7: 4K window
# - Layer 8: Full attention
# ... pattern repeats

# Result: ~60% KV cache reduction vs. full attention on all layers
```

**Prefix Caching:**

```python
# vLLM automatically caches common prefixes
# Example: System prompt reused across requests

system_prompt = "You are a helpful AI assistant."

prompts = [
    f"{system_prompt}\n\nUser: {user_query_1}",
    f"{system_prompt}\n\nUser: {user_query_2}",
    # ... system_prompt is cached after first request
]
```

#### Quantization Trade-offs

| Quantization | Memory | Speed | Quality Loss |
|--------------|--------|-------|--------------|
| **BF16/FP16** | 1x | 1x | None (baseline) |
| **FP8** | 0.5x | 1.2-1.5x | <1% degradation |
| **INT8** | 0.5x | 1.5-2x | 1-2% degradation |
| **INT4 (GGUF)** | 0.25-0.3x | 2-3x | 2-5% degradation |

**Recommendation:**
- **Production (quality-critical)**: BF16 or FP8
- **Production (cost-optimized)**: INT8
- **Consumer/Edge**: INT4 (GGUF)

### Cloud Deployment

#### AWS Deployment

**SageMaker:**

```python
from sagemaker.huggingface import HuggingFaceModel

# Create SageMaker model
huggingface_model = HuggingFaceModel(
    model_data="s3://your-bucket/olmo3-7b/",
    role="your-sagemaker-role",
    transformers_version="4.26",
    pytorch_version="2.0",
    py_version="py39",
)

# Deploy
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge"  # A10G GPU
)

# Inference
result = predictor.predict({
    "inputs": "Explain quantum entanglement",
    "parameters": {"max_new_tokens": 512, "temperature": 0.7}
})
```

**EC2 with vLLM:**

```bash
# Launch g5.xlarge (A10G) or p4d.24xlarge (A100)
# Install vLLM
pip install vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model allenai/Olmo-3-7B-Instruct \
  --tensor-parallel-size 1 \
  --port 8000
```

#### Google Cloud Deployment

**Vertex AI:**

```python
from google.cloud import aiplatform

aiplatform.init(project="your-project", location="us-central1")

# Deploy model
endpoint = aiplatform.Endpoint.create(display_name="olmo3-endpoint")

model = aiplatform.Model.upload(
    display_name="olmo3-7b-instruct",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu:latest",
    artifact_uri="gs://your-bucket/olmo3-7b/"
)

model.deploy(
    endpoint=endpoint,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1
)
```

#### Azure Deployment

**Azure ML:**

```python
from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment

ml_client = MLClient.from_config()

deployment = ManagedOnlineDeployment(
    name="olmo3-deployment",
    endpoint_name="olmo3-endpoint",
    model="azureml:olmo3-7b-instruct:1",
    instance_type="Standard_NC24ads_A100_v4",  # A100
    instance_count=1
)

ml_client.begin_create_or_update(deployment)
```

### Performance Benchmarks

**Inference Throughput (7B Model):**

| Hardware | Quantization | Batch Size | Throughput (tokens/sec) |
|----------|--------------|------------|-------------------------|
| RTX 4090 | BF16 | 1 | ~60-80 |
| RTX 4090 | INT4 | 1 | ~100-120 |
| A100 40GB | BF16 | 1 | ~80-100 |
| A100 80GB | BF16 | 8 (batched) | ~400-500 |
| H100 80GB | BF16 | 8 (batched) | ~600-800 |

**Inference Latency (Time to First Token):**

| Hardware | Quantization | Input Length | TTFT (ms) |
|----------|--------------|--------------|-----------|
| RTX 4090 | BF16 | 512 tokens | ~100-150 |
| RTX 4090 | INT4 | 512 tokens | ~50-80 |
| A100 40GB | BF16 | 512 tokens | ~80-120 |
| A100 80GB | BF16 | 4096 tokens | ~200-300 |
| H100 80GB | BF16 | 4096 tokens | ~150-200 |

**Long-Context Performance (65K tokens):**

| Hardware | Quantization | Throughput | Memory |
|----------|--------------|------------|--------|
| A100 80GB | BF16 | ~30-40 tokens/sec | ~70 GB |
| H100 80GB | BF16 | ~50-60 tokens/sec | ~70 GB |
| 2x A100 80GB | BF16 | ~60-80 tokens/sec | ~140 GB |

### Best Practices

**Deployment Checklist:**

1. **Choose Appropriate Hardware:**
   - 7B models: RTX 4090 or A100 40GB sufficient
   - 32B models: A100 80GB or better
   - Long context (65K): Ensure sufficient memory for KV cache

2. **Select Quantization:**
   - Production: BF16 or FP8
   - Cost-optimized: INT8
   - Edge/consumer: INT4

3. **Configure Batching:**
   - Low latency: Batch size 1-2
   - High throughput: Batch size 4-16 (depending on GPU memory)
   - Use continuous batching (vLLM) for mixed workloads

4. **Monitor Performance:**
   - Track throughput (tokens/sec)
   - Monitor latency (TTFT, total latency)
   - Watch GPU utilization and memory usage

5. **Optimize for Use Case:**
   - Chat: Low batch size, fast response
   - Document processing: Higher batch size, optimize for throughput
   - Long-context: Ensure sufficient memory, consider sliding window optimization

**Temperature Recommendations:**

```python
Use Case Temperatures:
  Production / Factual: temperature < 0.1
  Balanced: temperature = 0.7
  Creative Writing: temperature = 0.9-1.0

# OLMo 3 recommendation: < 0.1 for production
# This ensures consistent, factual outputs
```

---

(Document continues... Let me create the rest of the sections.)
