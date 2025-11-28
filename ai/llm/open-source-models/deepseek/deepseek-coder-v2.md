# DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence

## Overview

**DeepSeek-Coder-V2** is an open-source Mixture-of-Experts (MoE) code language model that achieves performance comparable to GPT-4 Turbo in code-specific tasks. Released in June 2024, it represents the first open-source model to surpass GPT-4 Turbo on coding and math benchmarks while supporting an unprecedented 338 programming languages with 128K context length.

### Model Information

| **Attribute** | **Details** |
|---------------|-------------|
| **Developer** | DeepSeek AI |
| **Release Date** | June 17, 2024 |
| **Model Type** | Mixture-of-Experts (MoE) Transformer |
| **Base Architecture** | DeepSeek-V2 (MLA + DeepSeekMoE) |
| **Variants** | 236B (21B active), 16B Lite (2.4B active) |
| **Context Length** | 128K tokens |
| **Programming Languages** | 338 (expanded from 86) |
| **Training Data** | 10.2T tokens total (6T additional for code) |
| **License** | MIT (code), DeepSeek Model License (model, commercial use supported) |
| **Primary Sources** | [ArXiv 2406.11931](https://arxiv.org/abs/2406.11931), [GitHub](https://github.com/deepseek-ai/DeepSeek-Coder-V2), [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct) |

### Notable Achievements

1. **First Open-Source to Beat GPT-4 Turbo**: 90.2% on HumanEval, 76.2% on MBPP+
2. **338 Programming Language Support**: Expanded from 86 languages
3. **128K Context Window**: Extended from 16K in original DeepSeek-Coder
4. **Superior Math Performance**: 75.7% on MATH, 94.9% on GSM8K
5. **Cost-Effective Architecture**: Only 21B activated parameters from 236B total

---

## Architecture Specifications

### 1. DeepSeek-Coder-V2 236B Model (Full)

Based on exact configuration from HuggingFace config.json:

| **Parameter** | **Value** |
|---------------|-----------|
| **Total Parameters** | 236B |
| **Activated Parameters per Token** | 21B (~8.9% activation rate) |
| **Transformer Layers** | 60 |
| **Hidden Dimension** | 5,120 |
| **Vocabulary Size** | 102,400 (Byte-level BPE) |
| **Context Length** | 128K tokens (163,840 max position embeddings) |
| **Precision** | BFloat16 |

#### **Attention Configuration (Multi-head Latent Attention - MLA)**

| **Parameter** | **Value** |
|---------------|-----------|
| **Attention Type** | Multi-head Latent Attention (MLA) |
| **Attention Heads** | 128 |
| **Key-Value Heads** | 128 |
| **Per-Head Dimension** | 128 |
| **QK NoP Head Dimension** | 128 |
| **QK RoPE Head Dimension** | 64 |
| **V Head Dimension** | 128 |
| **Total Query Dimension** | 16,384 (128 × 128) |
| **KV Compression Dimension** | 512 |
| **Query LoRA Rank** | 1,536 |
| **KV LoRA Rank** | 512 |
| **Compression Ratio** | 32:1 (16,384 / 512) |
| **KV Cache Reduction** | 93.3% |

**MLA Mechanism**:
1. **Compress**: c^KV = W^DKV x  (d_c = 512 dims)
2. **Cache**: [c^KV_1, ..., c^KV_n]  (only 512 dims per token)
3. **Decompress**: K = W^UK C^KV, V = W^UV C^KV  (16,384 dims)

**Decoupled RoPE**:
- K_h = [K^0_h; K^R_h]  (Non-RoPE (cached) + RoPE (fresh))
- Q_h = [Q^0_h; Q^R_h]  (Enables caching with position encoding)

#### **MoE Configuration per Layer**

| **Component** | **Specification** |
|---------------|------------------|
| **MoE Layers** | 58 out of 60 layers |
| **Shared Experts** | 2 (always activated) |
| **Routed Experts** | 160 per layer |
| **Activated Routed Experts** | 6 per token |
| **Expert FFN Intermediate** | 1,536 dims |
| **MoE Layer Frequency** | 1 (every layer after first 2) |
| **TopK Method** | "group_limited_greedy" |
| **TopK Group** | 3 |
| **Routed Scaling Factor** | 16.0 |
| **Total Activated Experts** | 8 (2 shared + 6 routed) |
| **Intermediate Size** | 12,288 (for non-MoE layers) |

#### **Position Embeddings**

| **Parameter** | **Value** |
|---------------|-----------|
| **Method** | Decoupled RoPE (Rotary Position Embeddings) |
| **Max Position Embeddings** | 163,840 |
| **RoPE Theta** | 10,000 |
| **RoPE Scaling Type** | YaRN |
| **YaRN Factor** | 40 |
| **YaRN Beta Fast** | 32 |
| **Original Max Position** | 4,096 |

#### **Training Configuration**

| **Parameter** | **Value** |
|---------------|-----------|
| **Activation Function** | SiLU |
| **Attention Dropout** | 0.0 |
| **RMS Norm Epsilon** | 1e-06 |
| **Initializer Range** | 0.02 |
| **Aux Loss Alpha** | 0.001 |
| **BOS Token ID** | 100,000 |
| **EOS Token ID** | 100,001 |
| **Tie Word Embeddings** | False |

### 2. DeepSeek-Coder-V2 Lite (16B Model)

| **Parameter** | **Value** |
|---------------|-----------|
| **Total Parameters** | 16B |
| **Activated Parameters per Token** | 2.4B (~15% activation rate) |
| **Transformer Layers** | 27 |
| **Hidden Dimension** | 2,048 |
| **Vocabulary Size** | 102,400 |
| **Context Length** | 128K tokens |

#### **Attention Configuration (Lite)**

| **Parameter** | **Value** |
|---------------|-----------|
| **Attention Heads** | 16 |
| **Key-Value Heads** | 16 |
| **QK NoP Head Dimension** | 128 |
| **QK RoPE Head Dimension** | 64 |
| **V Head Dimension** | 128 |

#### **MoE Configuration (Lite)**

| **Component** | **Specification** |
|---------------|------------------|
| **Routed Experts** | 64 per layer |
| **Shared Experts** | 2 (always activated) |
| **Activated Routed Experts** | 6 per token |
| **MoE Intermediate Size** | 1,408 |
| **MoE Layer Frequency** | 1 |
| **Intermediate Size** | 10,944 (for non-MoE layers) |

---

## Training Methodology

### 1. Training Data Composition (6T Additional Tokens)

**Total Training Exposure**: 10.2 trillion tokens
- 4.2T tokens from DeepSeek-V2 base model
- 6T tokens additional for code specialization

**6T Token Breakdown**:
- **60% Source Code** (~3.6T tokens, 1,170B after filtering)
  - 821B code tokens across 338 programming languages
  - 185B code-related text (markdown, issues, documentation)
- **10% Math Corpus** (~600B tokens, 221B after filtering)
  - Approximately doubles previous DeepSeekMath 120B corpus
- **30% Natural Language** (~1.8T tokens)
  - From DeepSeek-V2 dataset for general capabilities

### 2. Data Sources and Collection

**Code Corpus**:
- **Primary Source**: GitHub (pre-November 2023)
- **Secondary Source**: CommonCrawl with fastText-based domain classification
- **Languages**: 338 programming languages (expanded from 86)

**Filtering Rules**:
- Average line length ≤ 100 characters
- Maximum line length ≤ 1,000 characters
- ≥ 25% alphabetic characters
- **HTML**: ≥ 20% visible text content
- **JSON/YAML**: 50-5,000 characters
- Near-deduplication applied

**Result**: 821B code tokens across 338 languages + 185B code-related text

**Math Corpus**:
- 221B tokens (approximately 2× previous DeepSeekMath corpus)
- High-quality mathematical problems and solutions

### 3. Training Stages

#### **Stage 1: Pre-training from DeepSeek-V2 Checkpoint**

**Starting Point**: Intermediate checkpoint of DeepSeek-V2
- Inherits 4.2T tokens of general knowledge
- Already has MLA + DeepSeekMoE architecture

**Additional Pre-training**:
- 6T tokens code-focused data
- Total exposure: 10.2T tokens

**Training Objectives**:
- **236B Model**: Next-Token-Prediction (NTP) only
- **16B Lite Model**: NTP + Fill-In-Middle (FIM)

**FIM Configuration (16B Lite only)**:
- **FIM Rate**: 0.5 (50% of training samples)
- **Mode**: PSM (Prefix-Suffix-Middle)
- **Format**: `<｜fim_begin｜> prefix <｜fim_hole｜> suffix <｜fim_end｜> middle <|eos_token|>`
- **Application Level**: Document-level during pre-packing

#### **Stage 2: Supervised Fine-Tuning (SFT)**

**Dataset Composition**:
- 20K code instructions
- 30K math instructions
- General instruction data
- **Total**: ~300M tokens
- Batch size: 1M tokens

**Training Duration**:
- 1B tokens total
- Focus on instruction following and code generation

#### **Stage 3: Reinforcement Learning (GRPO)**

**Algorithm**: Group Relative Policy Optimization (GRPO)

**Key Innovation**: Eliminates critic model
- Uses empirical mean reward as baseline
- Computes advantage relative to group average
- **~50% compute reduction** vs PPO (used in ChatGPT)
- **Up to 18× more cost-efficient** than PPO in certain scenarios

**Reward Modeling**:
- Trained on compiler feedback
- Superior to raw 0-1 compiler signals
- Improves code correctness and execution success

**Benefits**:
- No separate value network needed
- Memory and compute savings
- Better sample efficiency

### 4. Training Hyperparameters

| **Parameter** | **Value** |
|---------------|-----------|
| **Optimizer** | AdamW |
| **β₁** | 0.9 |
| **β₂** | 0.95 |
| **Weight Decay** | 0.1 |
| **Learning Rate Schedule** | Cosine decay with warmup |
| **Warmup Steps** | 2,000 |
| **Final LR** | 10% of initial value |
| **SFT Batch Size** | 1M tokens |
| **SFT Total Tokens** | 1B tokens |

**Context Extension (YaRN)**:
- **Long Context Stage 1**:
  - Sequence length: 32K
  - Batch size: 1,152
  - Steps: 1,000
- **Long Context Stage 2**:
  - Sequence length: 128K
  - Batch size: 288
  - Steps: 1,000

### 5. Training Infrastructure

**Hardware** (inherited from DeepSeek-V2):
- **GPU Type**: NVIDIA H800 (restricted H100 variant)
- **GPU Specifications**:
  - Compute: 989 TFLOPS (BF16), 1,979 TFLOPS (FP8)
  - Memory: 80 GB HBM3
  - Bandwidth: 400 GB/s (vs 900 GB/s for H100)
- **Configuration**: 8 GPUs per node
- **Intra-Node**: NVLink and NVSwitch
- **Inter-Node**: InfiniBand

**Training Cost** (NOT DISCLOSED):
- Specific GPU hours not published
- Training budget not disclosed
- Estimated based on DeepSeek-V2 methodology
- Likely similar or incremental to V2 base model

---

## Programming Language Support

### 338 Languages Overview

**Expansion**: From 86 languages (DeepSeek-Coder V1) to 338 languages

**Coverage**: 821B code tokens across 338 programming languages

**Note**: Complete list of all 338 languages referenced in Appendix A of the paper but not fully enumerated in public documentation.

**Known Major Languages** (based on benchmark testing):
- Python (primary focus)
- Java
- JavaScript
- TypeScript
- C++
- C
- Go
- Rust
- Ruby
- PHP
- And 328+ others

**Achievement**: Multi-language capabilities with seamless switching between languages in same context window.

---

## Benchmark Performance

### 1. Code Generation Benchmarks

#### **HumanEval & MBPP (EvalPlus)**

| **Model** | **HumanEval** | **HumanEval+** | **MBPP** | **MBPP+** | **Average** |
|-----------|---------------|----------------|----------|-----------|-------------|
| **DeepSeek-Coder-V2-Instruct (236B)** | **90.2%** | **83.7%** | **77.5%** | **76.2%** | **81.9%** |
| GPT-4o-0513 | 91.0% | - | - | 73.5% | - |
| GPT-4-Turbo-0409 | 88.2% | - | - | 72.2% | - |
| Claude-3-Opus | 84.2% | - | - | 72.0% | - |
| Gemini-1.5-Pro | 84.1% | - | - | 69.8% | - |
| **DeepSeek-Coder-V2-Lite-Instruct (16B)** | 81.1% | 64.6% | 67.3% | 45.5% | **65.6%** |
| DeepSeek-Coder-33B-Instruct | 78.6% | 72.6% | 70.0% | 51.3% | 61.9% |

**Key Achievement**: 236B model achieves **90.2% on HumanEval**, establishing new state-of-the-art for open-source models and matching/exceeding GPT-4 Turbo.

#### **LiveCodeBench** (Recent Code Problems)

| **Model** | **Score** | **Period** |
|-----------|-----------|------------|
| **DeepSeek-Coder-V2-Instruct** | **43.4%** | Dec 2023 - June 2024 |
| GPT-4-Turbo-0409 | 45.7% | - |
| GPT-4o | 43.4% | - |

**Note**: On par with GPT-4o, second only to GPT-4-Turbo among all models.

#### **Fill-in-Middle (FIM) - Lite Model**

| **Language** | **Score** |
|--------------|-----------|
| **Python** | 80.0% |
| **Java** | 89.1% |
| **JavaScript** | 87.2% |
| **Mean** | **86.4%** |

**Note**: Despite only 2.4B active parameters, Lite achieves code completion in Python comparable to DeepSeek-Coder-Base 33B and in Java comparable to 7B model.

### 2. Mathematical Reasoning Benchmarks

| **Benchmark** | **DeepSeek-Coder-V2** | **GPT-4o** | **GPT-4-Turbo** |
|---------------|----------------------|------------|-----------------|
| **MATH** | **75.7%** | 76.6% | 72.2% |
| **GSM8K** | **94.9%** | 93.9% | 92.8% |
| **AIME 2024** (greedy) | 4/30 | - | - |
| **AIME 2024** (maj@64) | **5/30** | - | - |
| **Math Odyssey** | **53.7%** | 53.7% | - |

**Key Achievement**: Nearly matches GPT-4o on MATH (75.7% vs 76.6%), exceeds GPT-4-Turbo significantly.

### 3. General Language Benchmarks

| **Benchmark** | **Score** | **Description** |
|---------------|-----------|-----------------|
| **MMLU** | 79.2% | Multi-task Language Understanding (57 tasks) |
| **Arena-Hard** | 65.0 | Challenging user queries |
| **MT-Bench** | 8.77 | Multi-turn conversations |
| **AlignBench** | 7.84 | Chinese instruction following |

**Note**: Maintains strong general language performance despite code specialization.

### 4. Comparison with Other Code Models

#### **vs StarCoder2**

| **Aspect** | **DeepSeek-Coder-V2** | **StarCoder2-15B** |
|------------|----------------------|-------------------|
| **HumanEval** | 90.2% (236B) / 81.1% (Lite) | ~75% (15B) |
| **Language Focus** | 338 languages | Low-resource languages (Julia, Lua, Perl) |
| **Strengths** | Python completion, general coding | Math and code reasoning, low-resource langs |

**Verdict**: DeepSeek-Coder-V2 stronger on mainstream languages and overall performance.

#### **vs CodeLlama**

| **Aspect** | **DeepSeek-Coder-V2** | **CodeLlama-34B** |
|------------|----------------------|-------------------|
| **HumanEval** | 90.2% (236B) / 81.1% (Lite) | ~73% |
| **Architecture** | MoE (21B active from 236B) | Dense 34B |
| **Context** | 128K | 16K |
| **Performance** | Superior | Good but outdated |

**Verdict**: DeepSeek-Coder-V2 significantly outperforms CodeLlama on all metrics.

#### **vs Qwen2.5-Coder-7B**

| **Model** | **HumanEval** | **Note** |
|-----------|---------------|----------|
| **Qwen2.5-Coder-7B** | 88.4% | Smaller but highly optimized |
| **DeepSeek-Coder-V2-Lite-16B** | 81.1% | Larger but less specialized |
| **DeepSeek-Coder-V2-236B** | 90.2% | Beats all smaller models |

**Note**: Qwen2.5-Coder-7B achieves impressive performance for its size, but 236B DeepSeek-Coder-V2 remains superior.

### 5. Overall Performance Summary

**Coding Performance Ranking** (Average across benchmarks):
1. **DeepSeek-Coder-V2-Instruct (236B)**: 75.3% average
2. GPT-4o: 76.4% average
3. GPT-4-Turbo: ~73% average
4. Claude-3-Opus: ~72% average

**Key Takeaway**: First open-source model to achieve parity with or exceed GPT-4 Turbo on coding tasks.

---

## Innovations and Key Features

### 1. Code-Specific Optimizations

**Built on DeepSeek-V2 Base**:
- Inherits MLA (Multi-head Latent Attention) for efficient inference
- Inherits DeepSeekMoE for cost-effective training
- Adds 6T tokens of code-focused training

**Specialization Approach**:
- Continued pre-training from intermediate V2 checkpoint
- Maintains general capabilities while enhancing code
- No architectural changes from V2 base

### 2. Multi-head Latent Attention (MLA) Benefits for Code

**Memory Efficiency**:
- 93.3% KV cache reduction
- Enables longer code contexts (128K tokens)
- Critical for large codebases and multi-file reasoning

**Inference Speed**:
- 5.76× throughput vs dense models
- Faster code completion and generation
- Real-time coding assistance feasible

### 3. DeepSeekMoE for Code Specialization

**Expert Routing for Programming Languages**:
- 160 routed experts can specialize in different languages
- Fine-grained segmentation (1,536 dims per expert)
- 6 experts activated per token → can mix language expertise

**Benefits for Code**:
- Different experts for syntax, semantics, algorithms
- Multi-language code mixing handled naturally
- Shared experts capture universal programming concepts

### 4. Fill-in-Middle (FIM) Training (Lite Model Only)

**Training Approach**:
- 50% of 16B Lite training uses FIM
- PSM mode: Prefix-Suffix-Middle ordering
- Document-level application

**Use Cases**:
- Code completion in IDE
- Function body generation given signature
- Inserting code between existing lines

**Performance**:
- 86.4% mean FIM accuracy
- 89.1% on Java (highest)
- Competitive with much larger models

### 5. Group Relative Policy Optimization (GRPO)

**Replaces PPO in Alignment**:
- No critic model needed
- Uses group mean reward as baseline
- Computes advantage relative to peers

**Benefits**:
- ~50% compute reduction vs PPO
- Up to 18× more cost-efficient
- Better sample efficiency
- Simpler training pipeline

**Code-Specific Application**:
- Reward from compiler feedback
- Improves code correctness
- Reduces syntax and runtime errors

### 6. Context Extension to 128K

**From 16K (original DeepSeek-Coder) to 128K**:
- YaRN RoPE extension (s=40, α=1, β=32)
- Two-stage training (32K → 128K)
- Stable long-context retrieval

**Benefits for Code**:
- Process entire large files (e.g., >10K line modules)
- Multi-file reasoning without truncation
- Repository-level code understanding
- Extensive documentation ingestion

---

## Disclosed vs Not Disclosed Information

### ✅ Fully Disclosed

**Architecture**:
- Complete model specifications (layers, heads, dimensions)
- Exact MoE configuration (160 experts, 6 activated, 2 shared)
- MLA configuration (512 compression dim, 64 RoPE dim)
- All hyperparameters from config.json

**Training Data**:
- 10.2T total tokens (4.2T + 6T breakdown)
- 60% code / 10% math / 30% natural language
- 338 programming languages supported
- Data sources (GitHub, CommonCrawl)
- Filtering rules and criteria

**Training Methodology**:
- Three-stage pipeline (pretrain, SFT, RL)
- GRPO algorithm details
- Optimizer settings (AdamW, β values)
- Learning rate schedule
- Batch sizes and token counts

**Benchmarks**:
- Complete benchmark results on all major tests
- Comparison with GPT-4, Claude, Gemini
- Both base and instruct model scores
- Lite model performance

### ⚠️ Partially Disclosed

**Programming Languages**:
- **Disclosed**: 338 total languages
- **Not Disclosed**: Complete list of all 338 languages
- **Note**: Appendix A referenced but not fully public

**Training Infrastructure**:
- **Disclosed**: Hardware specs (H800 GPUs, 80GB memory)
- **Not Disclosed**: Exact number of GPUs used
- **Not Disclosed**: Total training time
- **Not Disclosed**: GPU hours or compute budget

**Model Variants**:
- **Disclosed**: 236B and 16B Lite models exist
- **Partially Disclosed**: Configuration differences fully detailed
- **Note**: No intermediate sizes released

### ❌ Not Disclosed

**Training Cost**:
- Total GPU hours (unlike DeepSeek-V3 which disclosed 2.788M H800 hours)
- Training budget in USD
- Compute cost breakdown by stage

**Detailed Training Curves**:
- Loss curves during training
- Validation perplexity over time
- Convergence behavior

**Intermediate Checkpoints**:
- Which specific DeepSeek-V2 checkpoint used
- How far into V2 training (e.g., after how many tokens)

**Data Mixture Details**:
- Exact token counts per programming language
- Language sampling strategy
- Data reweighting or curriculum

**Internal Design Decisions**:
- Why 236B chosen (vs other sizes)
- Ablation studies on code-specific modifications
- Expert specialization analysis (which experts learn what)

**Commercial Deployment Details**:
- Production serving infrastructure
- API pricing strategy
- Usage statistics

---

## Hardware Requirements

### Inference Requirements

**Full Precision (BF16) - 236B Model**:
- **VRAM**: 80GB × 8 GPUs = 640 GB total
- **GPUs**: A100 80GB, H100 80GB, or H800 80GB
- **Interconnect**: NVLink/NVSwitch (intra-node)
- **Use Case**: Full model with maximum quality

**Quantized Inference - 236B Model**:
- **8-bit**: ~320 GB (4× A100 80GB)
- **4-bit**: ~160 GB (2× A100 80GB)
- **Use Case**: Reduced memory with acceptable quality loss

**Lite Model (16B) - BF16**:
- **VRAM**: Minimum 40GB recommended (single GPU)
- **GPUs**: A100 40GB or better
- **Note**: Can run on consumer-grade high-end GPUs

**CPU Inference (Lite)**:
- **RAM**: ~16GB minimum
- **Performance**: Significantly slower than GPU
- **Use Case**: Development, testing, low-volume usage

### Training Requirements (Estimated)

**Fine-Tuning (LoRA or QLoRA)**:
- **236B Model**: 4-8× A100 80GB
- **Lite Model**: 2-4× A100 40GB or A100 80GB

**Full Fine-Tuning**:
- **236B Model**: Requires multi-node cluster (similar to V2 base)
- **Not Recommended**: Use LoRA instead

---

## Model Variants

### 1. DeepSeek-Coder-V2-Base (236B)

- **Parameters**: 236B total, 21B active
- **Training**: Pre-training only (no instruction tuning)
- **Use Case**: Further fine-tuning, research
- **HuggingFace**: `deepseek-ai/DeepSeek-Coder-V2-Base`

### 2. DeepSeek-Coder-V2-Instruct (236B)

- **Parameters**: 236B total, 21B active
- **Training**: Pre-training + SFT + GRPO RL
- **Use Case**: Production code generation, chat-based coding
- **HuggingFace**: `deepseek-ai/DeepSeek-Coder-V2-Instruct`
- **API**: Available via platform.deepseek.com
- **Chat Interface**: coder.deepseek.com

### 3. DeepSeek-Coder-V2-Lite-Base (16B)

- **Parameters**: 16B total, 2.4B active
- **Training**: Pre-training with FIM (50% rate)
- **Special Feature**: Fill-in-Middle capable
- **Use Case**: IDE code completion, smaller deployments
- **HuggingFace**: `deepseek-ai/DeepSeek-Coder-V2-Lite-Base`

### 4. DeepSeek-Coder-V2-Lite-Instruct (16B)

- **Parameters**: 16B total, 2.4B active
- **Training**: Pre-training + SFT + GRPO RL
- **Use Case**: Lightweight coding assistant
- **Performance**: 81.1% HumanEval (competitive with 33B models)
- **HuggingFace**: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`

### 5. DeepSeek-Coder-V2-Instruct-0724

- **Update Date**: July 24, 2024
- **Changes**: Improved instruction following
- **HuggingFace**: `deepseek-ai/DeepSeek-Coder-V2-Instruct-0724`

---

## Usage and Deployment

### Inference with HuggingFace Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model (236B Instruct)
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatic multi-GPU placement
    trust_remote_code=True
)

# Code completion
messages = [
    {"role": "user", "content": "Write a Python function to compute Fibonacci numbers"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

outputs = model.generate(
    inputs,
    max_new_tokens=512,
    temperature=0.0,  # Greedy for code
    do_sample=False
)

code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(code)
```

### Inference with vLLM (Recommended)

```bash
# Install vLLM with DeepSeek support
pip install vllm

# Serve model
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-Coder-V2-Instruct \
    --tensor-parallel-size 8 \
    --dtype bfloat16 \
    --max-model-len 8192
```

**Note**: Requires merging vLLM PR #4650 for full DeepSeek-V2 support.

### API Usage

```python
import openai

# Configure DeepSeek API
openai.api_key = "YOUR_DEEPSEEK_API_KEY"
openai.api_base = "https://api.deepseek.com"

response = openai.ChatCompletion.create(
    model="deepseek-coder",
    messages=[
        {"role": "user", "content": "Implement quicksort in Rust"}
    ],
    temperature=0.0
)

print(response.choices[0].message.content)
```

### Fill-in-Middle (FIM) with Lite Model

```python
# FIM format: <｜fim_begin｜> prefix <｜fim_hole｜> suffix <｜fim_end｜>
prefix = "def fibonacci(n):\n    if n <= 1:\n        return n\n"
suffix = "\n    return fibonacci(n-1) + fibonacci(n-2)"

fim_prompt = f"<｜fim_begin｜>{prefix}<｜fim_hole｜>{suffix}<｜fim_end｜>"

inputs = tokenizer(fim_prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)

middle = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(middle)  # Should generate the else branch
```

---

## Comparison with DeepSeek-V2 Base

| **Aspect** | **DeepSeek-V2** | **DeepSeek-Coder-V2** |
|------------|----------------|----------------------|
| **Training Data** | 8.1T tokens (general) | 10.2T tokens (4.2T + 6T code) |
| **Programming Languages** | General code support | 338 languages (specialized) |
| **Code Benchmarks** | HumanEval: 48.8% | HumanEval: 90.2% (+41.4%) |
| **Math Benchmarks** | MATH: 43.6% | MATH: 75.7% (+32.1%) |
| **Context Length** | 128K | 128K (same) |
| **Architecture** | MLA + DeepSeekMoE | MLA + DeepSeekMoE (inherited) |
| **Specialization** | General-purpose | Code + Math specialized |
| **Use Case** | General language tasks | Coding, math, and reasoning |

**Key Insight**: DeepSeek-Coder-V2 is a specialized variant, not a new architecture. It takes V2's efficient architecture and enhances it with massive code training.

---

## Strengths and Weaknesses

### Strengths

1. **Open-Source Leadership**: First open-source to match/exceed GPT-4 Turbo on code
2. **338 Language Support**: Unprecedented breadth of programming language coverage
3. **Cost-Effective Architecture**: Only 21B activated from 236B total (efficient inference)
4. **Long Context**: 128K tokens enables repository-level understanding
5. **Strong Math Reasoning**: 75.7% MATH, 94.9% GSM8K (near GPT-4o levels)
6. **Commercial License**: Permissive licensing for commercial use
7. **Efficient Inference**: 93.3% KV cache reduction via MLA
8. **Lite Variant**: 16B Lite model with 81.1% HumanEval at 2.4B active params
9. **Fill-in-Middle**: Lite model supports IDE-style code completion
10. **Well-Documented**: Detailed paper, config files, and code examples

### Weaknesses

1. **Large Memory Footprint**: 236B model requires 8× 80GB GPUs for BF16
2. **No Intermediate Sizes**: Only 236B and 16B variants (no 70B, 30B, etc.)
3. **Language List Not Public**: Complete 338 language list not enumerated
4. **Training Cost Undisclosed**: GPU hours and budget not published
5. **Complex Architecture**: MoE + MLA more complex than dense models
6. **vLLM Integration**: Requires specific PR merge for optimal serving
7. **FIM Only on Lite**: 236B model lacks Fill-in-Middle training
8. **Slightly Behind GPT-4o**: 75.7% vs 76.6% on MATH, though close
9. **No Function Calling**: Lacks tool use / function calling capabilities (unlike GPT-4)
10. **Chinese Bias**: Trained on bilingual data, may favor Chinese in some contexts

---

## Sources and References

### Official Papers
- [DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence](https://arxiv.org/abs/2406.11931) - ArXiv 2406.11931
- [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434) - ArXiv 2405.04434

### Official Repositories
- [GitHub - deepseek-ai/DeepSeek-Coder-V2](https://github.com/deepseek-ai/DeepSeek-Coder-V2)
- [GitHub - deepseek-ai/DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2)

### Model Cards
- [HuggingFace - DeepSeek-Coder-V2-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct)
- [HuggingFace - DeepSeek-Coder-V2-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Base)
- [HuggingFace - DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct)
- [HuggingFace - DeepSeek-Coder-V2-Lite-Base](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Base)
- [HuggingFace - DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)

### Technical Analyses
- [DeepSeek MoE and V2 - Chipstrat](https://www.chipstrat.com/p/deepseek-moe-and-v2)
- [Understanding DeepSeek Part I: DeepSeekMoE](https://www.chrishayduk.com/p/understanding-deepseek-part-i-deepseekmoe)
- [Understanding Multi-Head Latent Attention](https://planetbanatt.net/articles/mla.html)
- [DeepSeek-V3 Explained: Multi-head Latent Attention](https://medium.com/data-science/deepseek-v3-explained-1-multi-head-latent-attention-ed6bee2a67c4)
- [A Visual Walkthrough of DeepSeek's Multi-Head Latent Attention](https://towardsai.net/p/artificial-intelligence/a-visual-walkthrough-of-deepseeks-multi-head-latent-attention-mla-%EF%B8%8F)

### GRPO (Reinforcement Learning)
- [The Math Behind DeepSeek: A Deep Dive into GRPO](https://medium.com/@sahin.samia/the-math-behind-deepseek-a-deep-dive-into-group-relative-policy-optimization-grpo-8a75007491ba)
- [Group Relative Policy Optimization: DeepSeek's RL Cheat-Code](https://medium.com/better-ml/group-relative-policy-optimization-grpo-the-deep-seek-cheat-code-5c13a2c86317)
- [What is GRPO? The RL Algorithm Used to Train DeepSeek](https://medium.com/data-science-in-your-pocket/what-is-grpo-the-rl-algorithm-used-to-train-deepseek-12acc19798d3)
- [Why GRPO is Important and How it Works](https://ghost.oxen.ai/why-grpo-is-important-and-how-it-works/)

### Comparisons
- [DeepSeek-Coder-V2: Open-Source Model Beats GPT-4 and Claude Opus](https://the-decoder.com/deepseek-coder-v2-open-source-model-beats-gpt-4-and-claude-opus/)
- [Compare Code Llama vs DeepSeek Coder vs StarCoder](https://www.byteplus.com/en/topic/384880)
- [DeepSeek Coder vs. CodeLlama: Which Open-Source Model Actually Helps You Ship?](https://medium.com/@muhsinunc/deepseek-coder-vs-codellama-which-open-source-model-actually-helps-you-ship-e0ec5bd89a78)
- [Codestral 22B, Qwen 2.5 Coder 32B, and DeepSeek V2 Coder: Which AI Coder Should You Choose?](https://deepgram.com/learn/best-local-coding-llm)

### Documentation
- [DeepSeek-Coder-V2 Tutorial: Examples, Installation, Benchmarks](https://www.datacamp.com/tutorial/deepseek-coder-v2)
- [FIM Completion (Beta) | DeepSeek API Docs](https://api-docs.deepseek.com/guides/fim_completion)
- [DeepSeek Coder V2 | Open Laboratory](https://openlaboratory.ai/models/deepseek-coder-v2)

---

## Conclusion

**DeepSeek-Coder-V2** represents a landmark achievement in open-source code generation models, being the first to match or exceed GPT-4 Turbo on coding benchmarks. By leveraging the efficient MLA + DeepSeekMoE architecture from DeepSeek-V2 and specializing it with 6T additional tokens of code and math data, the model achieves:

- **90.2% on HumanEval** (state-of-the-art for open-source)
- **338 programming language support** (unprecedented breadth)
- **128K context length** (repository-level understanding)
- **75.7% on MATH** (near GPT-4o performance)
- **Cost-effective inference** (21B activated from 236B total)

The model's open-source nature, permissive commercial licensing, and availability in both full (236B) and lite (16B) variants make it accessible for a wide range of applications—from large-scale code generation services to lightweight IDE integration.

**Key Takeaway**: DeepSeek-Coder-V2 proves that open-source models can achieve frontier-level coding performance through architectural efficiency (MLA, MoE) and focused specialization (6T code tokens), democratizing access to state-of-the-art code intelligence.
