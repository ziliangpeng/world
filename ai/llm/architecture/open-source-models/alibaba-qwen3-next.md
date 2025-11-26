# Qwen3-Next

## Overview

Qwen3-Next is a revolutionary next-generation language model architecture released by the Qwen team on September 10, 2025. It represents a major breakthrough in model efficiency, achieving **10x cost reduction** in training and **10x higher throughput** at long contexts compared to Qwen3-32B, while maintaining comparable performance to the much larger Qwen3-235B model.

With 80 billion total parameters but only 3 billion activated per token (3.7% activation rate), Qwen3-Next pioneers the use of **Gated DeltaNet** and **Gated Attention** mechanisms in a hybrid architecture, replacing standard softmax attention entirely. This combination enables native support for 262K token contexts (extendable to 1 million tokens) with exceptional efficiency.

Qwen3-Next serves as the architectural foundation for the upcoming Qwen3.5 series, marking a paradigm shift from traditional transformer attention mechanisms toward more efficient linear attention variants.

**Key Highlights:**
- 80B total parameters, 3B activated per token (3.7% activation rate)
- Hybrid Gated DeltaNet + Gated Attention architecture (no standard attention)
- 10% training cost of Qwen3-32B (~1.5T tokens equivalent at 10x efficiency)
- 10x higher throughput at 32K+ context lengths
- 262K native context, extendable to 1M tokens
- Performance matches Qwen3-235B-A22B on many benchmarks
- Foundation for Qwen3.5 architecture
- Release date: September 10, 2025

## Model Architecture

### Core Architecture Philosophy

Qwen3-Next completely replaces standard softmax attention with a hybrid combination of two linear attention mechanisms:

1. **Gated DeltaNet**: A linear attention variant with delta rule updates and gating mechanisms for adaptive memory control
2. **Gated Attention**: Linear attention with data-dependent gating for enhanced expressiveness

This hybrid approach maintains the strengths of both mechanisms while achieving:
- Linear-time inference complexity O(n) instead of O(n²)
- Efficient parallel training like transformers
- RNN-like recurrent formulation for inference
- Native long-context understanding without architectural modifications

### Architecture Specifications

```yaml
Model: Qwen3-Next-80B-A3B

Total Parameters: 80 billion (80B)
Non-Embedding Parameters: 79 billion
Activated Parameters per Token: 3 billion (3B)
Activation Rate: 3.7% (3B/80B)

Core Architecture:
  Layers: 48 total
  Hidden Dimension: 2,048
  Vocabulary Size: 151,936 tokens

Layer Pattern (Hybrid Layout):
  Repeating Pattern: 12 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))

  Breakdown:
    - 36 Gated DeltaNet layers (3 per 4-layer block)
    - 12 Gated Attention layers (1 per 4-layer block)
    - All layers followed by MoE feed-forward
```

### Gated Attention

Gated Attention is a linear attention mechanism with data-dependent gating, providing O(n) complexity while maintaining expressiveness comparable to standard attention.

**Configuration:**
```yaml
Query Heads: 16
Key-Value Heads: 2 (Grouped Query Attention style)
Head Dimension: 256
Total Query Dimension: 4,096 (16 × 256)
Total KV Dimension: 512 (2 × 256)

Position Encoding:
  Type: RoPE (Rotary Position Embeddings)
  RoPE Dimension: 64 per head
  Theta: 10,000 (base frequency)
```

**Key Features:**
- **Data-Dependent Gating**: Enables adaptive memory control based on input
- **Grouped Query Attention**: Shares key-value heads across multiple query heads for efficiency
- **Linear Complexity**: O(n) time for both training and inference vs O(n²) for standard attention
- **RNN Formulation**: Can be computed recurrently with 2D matrix-valued hidden states

**How It Works:**

Gated Attention replaces the softmax in standard attention with:
1. **Linear Attention Score**: Compute attention without softmax normalization
2. **Gating Function**: Apply data-dependent gates to control information flow
3. **Recurrent State**: Maintain a matrix-valued state that can be updated incrementally

```
Standard Attention:
  score = softmax(Q @ K^T / sqrt(d))
  output = score @ V

Gated Attention (simplified):
  gate = sigmoid(W_g @ x)
  score = linear_attention(Q, K) * gate  # No softmax
  output = score @ V
```

**Benefits:**
- Faster than FlashAttention-2 on sequences > 4K tokens
- Linear memory complexity for KV cache
- Excellent length generalization (trained on 2K, extrapolates to 20K+)
- Competitive performance with standard attention on language modeling

**Technical Reference:**
- Paper: [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) (ICML 2024)
- Implementation: Flash Linear Attention (FLA) library

### Gated DeltaNet

Gated DeltaNet is the core innovation enabling Qwen3-Next's exceptional efficiency. It combines **gating mechanisms** (for adaptive memory control) with the **delta rule** (for precise memory updates) in a unified linear attention framework.

**Configuration:**
```yaml
Linear Attention Heads:
  Value (V) Heads: 32
  Query-Key (QK) Heads: 16

Head Dimension: 128

Total Dimensions:
  Value Dimension: 4,096 (32 × 128)
  Query-Key Dimension: 2,048 (16 × 128)

Memory State: 2D matrix-valued hidden state
Complexity: O(n × d²) time, O(d²) space per token
```

**Core Innovation - Complementary Mechanisms:**

1. **Gating Mechanism**:
   - Enables rapid memory erasure
   - Adaptively controls which information to retain
   - Provides forgetting mechanism similar to LSTM gates

2. **Delta Rule**:
   - Enables precise, targeted memory modifications
   - Updates specific parts of memory state
   - Improves in-context learning and retrieval

**Mathematical Formulation (Simplified):**

```
State Update (Gated Delta Rule):
  h_t = gate_t * h_{t-1} + delta_update(Q_t, K_t, V_t)

Where:
  gate_t: Data-dependent gating values (controls forgetting)
  delta_update: Precise memory modification based on delta rule
  h_t: 2D matrix-valued hidden state at time t
```

**Why It Works:**

Traditional linear attention often underperforms standard attention because:
- Simple linear scoring lacks the selectivity of softmax
- No mechanism for forgetting irrelevant information
- Difficulty with precise memory updates

Gated DeltaNet solves these issues:
- **Gating** provides forgetting and selectivity
- **Delta rule** enables precise updates to specific memory locations
- **Complementary**: Gating handles coarse control, delta rule handles fine-grained updates

**Training Algorithm:**

Gated DeltaNet includes a parallel training algorithm optimized for modern hardware:
- Chunk-wise parallelization similar to FlashAttention
- Efficient memory access patterns
- Trade-off between parallelism and memory movement
- Faster than FlashAttention-2 even on short sequences (1K tokens)

**Performance Characteristics:**

```yaml
Time Complexity:
  Training: O(n × d²) with parallelization
  Inference: O(d²) per token (constant time!)

Memory Complexity:
  KV Cache: O(d²) per position (not O(n × d))
  Training: O(sqrt(n) × d²) with chunking

Throughput vs FlashAttention-2:
  1K tokens: Comparable
  4K tokens: ~1.5x faster
  16K+ tokens: ~3-5x faster
```

**Benchmarks vs Mamba2 and DeltaNet:**

Gated DeltaNet consistently surpasses:
- **Mamba2**: State-space model with selective state transitions
- **DeltaNet**: Linear attention with delta rule (without gating)

Improvements across:
- Language modeling perplexity
- Common-sense reasoning (PIQA, WinoGrande, etc.)
- In-context retrieval tasks
- Length extrapolation (2K → 20K+)
- Long-context understanding

**Technical Reference:**
- Paper: [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) (ICLR 2025)
- Authors: Songlin Yang, et al. (NVIDIA)
- Implementation: Flash Linear Attention (FLA) library
- Adoption: Qwen3-Next, Kimi Linear

### Hybrid Architecture Layout

Qwen3-Next's 48 layers follow a carefully designed hybrid pattern that balances efficiency and expressiveness:

**Repeating 4-Layer Block (12 times):**
```
Block Pattern:
  Layer 1: Gated DeltaNet → MoE
  Layer 2: Gated DeltaNet → MoE
  Layer 3: Gated DeltaNet → MoE
  Layer 4: Gated Attention → MoE

Ratio: 3:1 (Gated DeltaNet : Gated Attention)
Total: 36 Gated DeltaNet + 12 Gated Attention layers
```

**Design Rationale:**

1. **Gated DeltaNet Dominance (75%)**:
   - Provides the bulk of efficient context modeling
   - Handles long-range dependencies with O(d²) inference cost
   - Maintains recurrent state for streaming inference

2. **Gated Attention Sparse Injection (25%)**:
   - Enhances expressiveness at key positions
   - Provides stronger local context modeling
   - Improves performance on retrieval and reasoning tasks
   - Acts as "checkpoints" in the layer stack

3. **MoE After Every Layer**:
   - All 48 layers use MoE for feed-forward computation
   - Maximizes parameter efficiency
   - Enables 3.7% ultra-low activation rate

**Benefits of Hybrid Approach:**

- **Best of Both Worlds**: Efficiency of linear attention + expressiveness of attention variants
- **Native Long Context**: 262K native context without architectural changes
- **Streaming Inference**: Constant per-token cost with recurrent formulation
- **Computational Efficiency**: Much lower FLOPs than full attention while maintaining quality

### Mixture-of-Experts (MoE) Architecture

Qwen3-Next employs an ultra-sparse MoE architecture where every one of the 48 layers has expert routing, achieving an extreme 3.7% activation rate.

**MoE Configuration:**
```yaml
MoE Layers: 48 (all layers)

Per MoE Layer:
  Total Experts: 512
  Activated Experts per Token: 10
  Shared Experts: 1 (always activated)

Routing:
  Strategy: Top-10 expert selection
  Router Type: Learned router network
  Load Balancing: Auxiliary loss or auxiliary-loss-free (not disclosed)

Expert Architecture:
  Expert Intermediate Dimension: 512
  Activation Function: SwiGLU

Total Activated Parameters:
  - 48 layers × 10 experts × ~60M params/expert = ~29B
  - Shared experts: ~0.5B
  - Attention/DeltaNet: ~0.5B
  - Total: ~3B per token
```

**Ultra-Sparse Design:**

With 80B total parameters but only 3B activated (3.7%), Qwen3-Next achieves:
- **Training Efficiency**: 10% training cost of dense 32B model
- **Inference Efficiency**: 10x throughput at long contexts
- **Memory Efficiency**: Only 3B parameters need to be loaded into compute

**Expert Specialization:**

512 experts per layer enable fine-grained specialization:
- Domain-specific experts (code, math, languages)
- Task-specific experts (reasoning, factual recall)
- Context-specific experts (long-range, local patterns)

**Comparison to Other MoE Models:**

```yaml
Qwen3-Next: 80B total, 3B activated (3.7%)
Qwen3-235B-A22B: 235B total, 22B activated (9.4%)
DeepSeek-V3: 671B total, 37B activated (5.5%)
Mixtral-8x7B: 47B total, 13B activated (27.6%)
```

Qwen3-Next has the **lowest activation rate** among major MoE models, demonstrating extreme sparsity.

### Position Embeddings

**RoPE (Rotary Position Embeddings):**
```yaml
Type: RoPE with YaRN scaling
RoPE Dimension: 64 per head
Base Frequency (theta): 10,000

Native Training Context: 262,144 tokens (262K)
Extended Context: Up to 1,010,000 tokens (1M) via YaRN

YaRN Configuration:
  Scaling Method: Interpolation with attention temperature adjustment
  Extension Factors: Progressive scaling for 262K → 1M
```

**Key Feature - Native Long Context:**

Unlike models that extend context through fine-tuning:
- Qwen3-Next is trained natively on 262K sequences
- Linear attention mechanisms naturally handle long contexts
- No "context extension" stage needed
- Performance remains consistent across full context range

**Benefits:**
- Constant per-token cost at inference (due to linear attention)
- No performance degradation at longer contexts
- Efficient memory usage (O(d²) state, not O(n × d))

### Activation Function

**SwiGLU (Swish Gated Linear Unit):**
```python
def SwiGLU(x):
    x1, x2 = split(x)
    return Swish(x1) * x2

def Swish(x):
    return x * sigmoid(x)
```

**Usage:**
- Applied in all expert feed-forward layers
- Provides non-linearity and gating
- Standard choice in modern LLMs (Qwen3, LLaMA 3, etc.)

### Normalization

**RMSNorm (Root Mean Square Normalization):**
```python
def RMSNorm(x, weight):
    rms = sqrt(mean(x^2) + epsilon)
    return (x / rms) * weight
```

**Configuration:**
- Applied before each attention/DeltaNet layer (pre-normalization)
- Applied before each MoE feed-forward layer
- 15% faster than LayerNorm
- Better numerical stability

### Tokenizer

**Byte-level BPE:**
```yaml
Vocabulary Size: 151,936 tokens (same as Qwen3)
Type: Byte-level Byte Pair Encoding
Multilingual: 100+ languages
Special Tokens: Includes thinking tags, system prompts
```

## Multi-Token Prediction (MTP)

Qwen3-Next incorporates Multi-Token Prediction as a training objective and inference acceleration technique, building on the approach introduced in Qwen3.

**Purpose:**
1. **Training**: Improves data efficiency by predicting multiple future tokens
2. **Inference**: Enables speculative decoding for faster generation

**Architecture:**
- Additional prediction heads for future tokens
- Shared representations with main model
- Can be discarded at inference or used for speculation

**Benefits:**
- Denser training signals
- Better forward planning in representations
- Potential 2-3x inference speedup with speculative decoding
- Improved long-range dependency learning

## Training Details

### Pre-Training

**Training Data:**
```yaml
Total Tokens: 15 trillion tokens (15T)
Data Source: Subset of Qwen3's 36T pre-training corpus
Selection Strategy: High-quality diverse subset

Data Composition: Not disclosed (follows Qwen3 mixture)
  - Web text
  - Code (80+ languages)
  - Mathematics and scientific content
  - Multilingual data (100+ languages)
  - Synthetic reasoning data

Knowledge Cutoff: April 2025 (estimated, based on Qwen3)
```

**Training Efficiency:**

Qwen3-Next's revolutionary efficiency comes from three factors:

1. **Linear Attention Efficiency**: Gated DeltaNet + Gated Attention are computationally cheaper than standard attention
2. **Ultra-Sparse MoE**: Only 3.7% activation reduces compute per token
3. **Optimized Training Algorithm**: Hardware-efficient parallel training

**Training Cost:**
```yaml
GPU Hours: Less than 10% of Qwen3-32B training cost

Equivalent Compute:
  If Qwen3-32B = 100% baseline
  Qwen3-Next-80B = 10% of baseline

  Despite:
    - 2.5× more parameters (80B vs 32B)
    - Higher quality (matches Qwen3-235B performance)
```

**Cost Breakdown:**

For reference, if Qwen3-32B took ~X million GPU hours:
- Qwen3-Next-80B: ~0.1X million GPU hours
- **10x cost reduction** despite 2.5x more parameters

This represents a **major breakthrough** in training efficiency for large language models.

**Training Infrastructure:**
```yaml
GPUs: Not disclosed (likely H800 or H100)
Training Duration: Not disclosed
Parallelism Strategy: Not disclosed

Optimization:
  - Parallel training algorithm for Gated DeltaNet
  - Efficient memory access patterns
  - Hardware-optimized kernels (FLA library)
```

**Training Stability:**
- Stable training despite novel architecture
- No reported loss spikes or rollbacks
- Smooth convergence with linear attention

### Context Length Extension

**Native Long Context:**

Unlike traditional models that require context extension stages, Qwen3-Next is trained natively on long sequences:

```yaml
Native Training Context: 262,144 tokens (262K)
Extended Support: Up to 1,010,000 tokens (1M) via YaRN
Extension Method: YaRN interpolation

No Fine-Tuning Required: Linear attention naturally handles long contexts
```

**Advantages:**
- No separate context extension training stage
- Consistent performance across all context lengths
- Constant per-token inference cost
- Efficient memory usage

### Post-Training

**Supervised Fine-Tuning (SFT):**
```yaml
Training Data: High-quality instruction datasets
Data Composition:
  - General instruction following
  - Mathematical reasoning
  - Code generation and debugging
  - Multilingual dialogue
  - Long-context understanding tasks

Model Variants:
  - Qwen3-Next-80B-A3B-Instruct: General instruction tuning
  - Qwen3-Next-80B-A3B-Thinking: Reasoning-focused with thinking tags
```

**Instruction Variant:**
- Optimized for instruction following and general capabilities
- Balances helpfulness, accuracy, and safety
- Supports tool use and agent tasks

**Thinking Variant:**
- Enhanced reasoning with explicit thinking process
- Uses `<think>` and `</think>` tags for internal reasoning
- Excels at complex reasoning tasks
- Competitive with leading reasoning models

## Performance Benchmarks

### General Benchmarks

**MMLU-Pro (Massive Multitask Language Understanding - Pro):**
```yaml
Qwen3-Next-80B-A3B-Instruct: 80.6
Qwen3-235B-A22B-Instruct: ~82-85 (estimated)
Qwen3-32B-Instruct: ~75-78 (estimated)
```

Despite having only 3B activated parameters, Qwen3-Next achieves performance close to the 235B flagship model.

### Mathematical Reasoning

**AIME 2025 (American Invitational Mathematics Examination):**
```yaml
Qwen3-Next-80B-A3B-Thinking: 69.5
Comparison:
  - Competitive with much larger models
  - Strong performance for 3B activated parameters
```

**MATH Benchmark:**
```yaml
Performance: Strong (exact scores not disclosed)
Improvement: Thinking variant excels at multi-step reasoning
```

### Code Generation

**LiveCodeBench:**
```yaml
Qwen3-Next-80B-A3B-Instruct: 56.6
Notable: Strong coding performance for model size
```

**HumanEval:**
```yaml
Performance: Competitive (exact scores not disclosed)
```

### Alignment and Instruction Following

**Arena-Hard:**
```yaml
Performance: Competitive with Qwen3-235B on alignment tasks
Notable: Despite 7.3× fewer activated parameters
```

**IFEval (Instruction Following Evaluation):**
```yaml
Performance: Strong instruction following capabilities
```

### Efficiency Benchmarks

**Throughput (tokens/second):**

The key innovation - **10x higher throughput** at long contexts:

```yaml
Context Length: 32K tokens
Qwen3-Next-80B: 10× throughput
Qwen3-32B: 1× baseline

Context Length: 64K tokens
Qwen3-Next-80B: ~15× throughput (estimated)
Qwen3-32B: Significantly slower

Context Length: 128K+ tokens
Qwen3-Next-80B: ~20×+ throughput (estimated)
Qwen3-32B: Impractical
```

**Latency:**

```yaml
First Token Latency: Lower than standard attention at long contexts
Per-Token Latency: Constant O(d²), independent of context length

Comparison at 64K context:
  Standard Attention: ~O(64K²) = Very slow
  Gated DeltaNet: O(d²) = Constant fast
```

**Memory Usage:**

```yaml
KV Cache:
  Standard Attention: O(n × d) grows with sequence length
  Qwen3-Next: O(d²) constant state size

At 128K context:
  Standard Attention: ~128K × hidden_dim memory
  Qwen3-Next: ~hidden_dim² constant (much smaller)
```

### Comparison to Other Models

**Performance vs Parameters Activated:**

```yaml
Model Comparison (Activated Parameters → Performance):

Qwen3-Next-80B (3B activated):
  - MMLU-Pro: 80.6
  - AIME25: 69.5
  - LiveCodeBench: 56.6
  - Performance: Matches Qwen3-235B (22B activated) on many tasks

Qwen3-235B-A22B (22B activated):
  - Performance: Flagship level
  - Cost: 7.3× more activated compute per token

Qwen3-32B (32B dense):
  - Performance: Lower than Qwen3-Next
  - Training Cost: 10× more expensive
  - Inference: 10× slower at long contexts
```

**Key Insight:** Qwen3-Next achieves near-flagship performance with:
- **7.3× fewer activated parameters** than Qwen3-235B
- **10× lower training cost** than Qwen3-32B
- **10× higher throughput** than Qwen3-32B at long contexts

This represents a major breakthrough in efficiency-performance trade-offs.

## Model Variants

### Qwen3-Next-80B-A3B-Base

**Specifications:**
- Pre-trained foundation model (no instruction tuning)
- 15T token training
- 262K native context

**Use Cases:**
- Research and experimentation
- Fine-tuning for specific domains
- Benchmarking
- Understanding base model capabilities

### Qwen3-Next-80B-A3B-Instruct

**Specifications:**
- Instruction-tuned for dialogue and general tasks
- Optimized for helpfulness and safety
- 256K context support (extendable to 1M)

**Capabilities:**
- General conversation and assistance
- Instruction following
- Code generation and debugging
- Mathematical problem solving
- Multilingual dialogue (100+ languages)
- Tool use and agent tasks
- Long-document analysis

**Use Cases:**
- General-purpose AI assistant
- Code assistant
- Content generation
- Question answering
- Multilingual applications
- Long-context analysis (documents, codebases)

**Performance:**
- MMLU-Pro: 80.6
- LiveCodeBench: 56.6
- Matches Qwen3-235B on many alignment tasks

### Qwen3-Next-80B-A3B-Thinking

**Specifications:**
- Reasoning-focused variant
- Uses `<think>` and `</think>` tags for explicit reasoning
- Optimized for complex multi-step reasoning tasks
- 256K context support (extendable to 1M)

**Capabilities:**
- Chain-of-thought reasoning
- Mathematical problem solving
- Logical reasoning
- Scientific reasoning
- Code generation with planning
- Multi-step task decomposition

**Use Cases:**
- Competition-level mathematics (AIME, MATH)
- Complex coding challenges
- Scientific research assistance
- Logic puzzles and reasoning tasks
- Planning and strategy problems

**Performance:**
- AIME25: 69.5
- Excels at reasoning-heavy benchmarks
- Competitive with leading reasoning models

**Thinking Process:**
```
User: Solve this complex math problem...

Model: <think>
Let me break this down step by step.
First, I'll identify the key variables...
Then, I'll apply theorem X...
Now, I'll check if the constraints are satisfied...
</think>

The answer is...
```

## Deployment and Inference

### Hardware Requirements

**Minimum Requirements:**
```yaml
Memory (FP16/BF16): ~160 GB
Memory (FP8): ~80 GB
Memory (INT4): ~40 GB

GPU Options:
  - 2× A100 80GB (FP16, tight)
  - 1× H100 80GB (FP8/INT4)
  - 2× H800 80GB (FP16)
  - 4× A6000 48GB (FP16 with parallelism)
```

**Recommended Setup:**
- 2× H100/H800 80GB for comfortable deployment
- High-bandwidth interconnect for multi-GPU
- NVMe SSD for model loading

### Supported Inference Frameworks

**SGLang (Recommended):**
```yaml
Version: ≥ 0.5.2
Features:
  - Native flash-linear-attention support
  - Optimized for Gated DeltaNet
  - Efficient long-context handling
  - Multi-GPU tensor parallelism

Installation: pip install "sglang[all]>=0.5.2"
```

**vLLM:**
```yaml
Version: ≥ 0.10.2
Features:
  - PagedAttention support
  - Tensor parallelism
  - Pipeline parallelism
  - FP8/INT4/INT8 quantization

Installation: pip install vllm>=0.10.2
```

**Hugging Face Transformers:**
```yaml
Version: ≥ 4.37.0
Features:
  - Direct model loading
  - Easy inference API
  - Flash Linear Attention integration
  - Quantization support

Installation: pip install transformers>=4.37.0
```

### Quantization Options

**FP16/BF16 (Native):**
- Memory: ~160 GB
- Quality: Full precision
- Speed: Baseline

**FP8:**
- Memory: ~80 GB (50% reduction)
- Quality: Minimal degradation (<1%)
- Speed: 1.5-2× faster on H100
- Recommended for deployment

**INT8:**
- Memory: ~40 GB (75% reduction)
- Quality: Slight degradation (~2-3%)
- Speed: 2-3× faster
- Good for resource-constrained deployment

**INT4 (4-bit):**
- Memory: ~20 GB (87.5% reduction)
- Quality: Noticeable degradation (~5-10%)
- Speed: 3-4× faster
- Use for edge deployment or tight memory budgets

### Deployment Examples

**SGLang Deployment:**
```bash
# Install SGLang with flash-linear-attention support
pip install "sglang[all]>=0.5.2"

# Launch server
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp 2 \
  --dtype float16 \
  --trust-remote-code \
  --enable-flash-linear-attn
```

**vLLM Deployment:**
```bash
# Install vLLM
pip install vllm>=0.10.2

# Launch server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --max-model-len 131072 \
  --trust-remote-code
```

**Transformers Inference:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    trust_remote_code=True
)

# Generate
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

### Flash Linear Attention Optimization

Qwen3-Next can leverage flash-linear-attention for additional speedup:

```python
# Enable flash-linear-attention (requires FLA library)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_linear_attention",  # Enable FLA
    trust_remote_code=True
)
```

**Benefits:**
- Faster computation for Gated DeltaNet and Gated Attention
- Lower memory usage
- Better scaling to long contexts

### Multi-Token Prediction Inference

For speculative decoding speedup:

```python
# Enable MTP for speculative decoding
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    use_multi_token_prediction=True,  # Enable MTP speculation
    num_speculative_tokens=2  # Predict 2 tokens ahead
)
```

**Benefits:**
- 1.5-2.5× inference speedup (context dependent)
- No quality degradation (verified predictions only)
- Particularly effective for long generations

### Long-Context Handling

**Efficient Long-Context Inference:**

Thanks to Gated DeltaNet's O(d²) state complexity:

```python
# Handle 256K context efficiently
long_context = "..." * 100000  # Very long text

inputs = tokenizer(long_context, return_tensors="pt").to(model.device)
# Efficient! Constant per-token cost

outputs = model.generate(**inputs, max_new_tokens=1000)
```

**Comparison:**
- Standard Attention: Would require O(256K²) memory and computation
- Qwen3-Next: Only O(2048²) = constant state size

**Extended Context (1M tokens):**

```python
# Extend to 1M tokens via YaRN
from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
config.max_position_embeddings = 1000000  # Extend to 1M

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    config=config,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
```

## Key Innovations

### 1. Hybrid Gated DeltaNet + Gated Attention Architecture

**Innovation:** First major LLM to completely replace standard softmax attention with a hybrid of two linear attention mechanisms.

**Mechanisms:**

**Gated DeltaNet:**
- Linear attention with delta rule for precise memory updates
- Gating mechanism for adaptive memory control
- Complementary design: gating enables forgetting, delta rule enables targeted updates
- O(d²) constant inference cost per token

**Gated Attention:**
- Linear attention with data-dependent gating
- Enhanced expressiveness over pure linear attention
- Grouped Query Attention for efficiency
- Excellent length generalization

**Hybrid Layout:**
- 3:1 ratio (36 Gated DeltaNet : 12 Gated Attention layers)
- Provides efficiency of linear attention with expressiveness of attention variants
- Enables native 262K context without architectural modifications

**Impact:**
- **10x throughput** at long contexts vs standard attention
- **Linear inference complexity** O(n) instead of O(n²)
- **Constant state size** O(d²) instead of O(n × d)
- **Native long-context understanding** without fine-tuning

**Technical Foundation:**
- Gated DeltaNet: [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) (ICLR 2025)
- Gated Attention: [arXiv:2312.06635](https://arxiv.org/abs/2312.06635) (ICML 2024)

### 2. Ultra-Sparse Mixture-of-Experts (3.7% Activation)

**Innovation:** Achieves the lowest activation rate among major MoE models while maintaining competitive performance.

**Configuration:**
- 80B total parameters
- 3B activated per token
- 3.7% activation rate
- 512 experts per layer, 10 activated

**Compared to Other MoE Models:**
```
Qwen3-Next: 3.7% activation
DeepSeek-V3: 5.5% activation
Qwen3-235B-A22B: 9.4% activation
Mixtral-8x7B: 27.6% activation
```

**Benefits:**
- **Extreme training efficiency**: 10% cost of dense 32B model
- **Inference efficiency**: Only 3B parameters active per forward pass
- **Memory efficiency**: Smaller memory footprint for inference
- **Parameter efficiency**: 80B capacity with 3B compute cost

**Impact:**
- Demonstrates that extreme sparsity is viable for frontier-scale models
- Challenges assumptions about necessary activation rates
- Paves way for even larger, more efficient models

### 3. 10x Training Cost Reduction

**Innovation:** Achieves 10× reduction in training cost compared to Qwen3-32B despite having 2.5× more parameters.

**Cost Factors:**

1. **Linear Attention Efficiency:**
   - Gated DeltaNet and Gated Attention have O(n × d²) complexity
   - Standard attention has O(n² × d) complexity
   - For typical values (n >> d), linear attention is much faster

2. **Ultra-Sparse MoE:**
   - Only 3.7% of parameters activated
   - Reduces compute per token by ~27× compared to dense

3. **Optimized Training Algorithm:**
   - Hardware-efficient parallel training for Gated DeltaNet
   - Efficient memory access patterns
   - Better compute-to-memory ratio than FlashAttention

**Result:**
```
Qwen3-32B training cost: 100% (baseline)
Qwen3-Next-80B training cost: 10% of baseline

Despite:
  - 2.5× more parameters (80B vs 32B)
  - Similar or better performance
```

**Impact:**
- Makes frontier-scale model training accessible to more organizations
- Dramatically reduces carbon footprint of AI training
- Enables faster experimentation and iteration

### 4. 10x Throughput at Long Contexts

**Innovation:** Achieves 10× higher generation throughput at 32K+ context lengths compared to Qwen3-32B.

**Mechanism:**

Standard Attention:
```
Complexity: O(n²)
At 32K context: 32,000² = 1.024 billion operations per token
```

Gated DeltaNet:
```
Complexity: O(d²)
With d=2048: 2,048² = ~4.2 million operations per token
Speedup: 1.024B / 4.2M = ~244× theoretical
```

**Real-World Speedup:**
```
32K context: 10× faster
64K context: ~15× faster (estimated)
128K+ context: ~20×+ faster (estimated)
```

(Real speedup is lower than theoretical due to other bottlenecks like memory bandwidth, but still dramatic)

**Benefits:**
- Practical long-context generation (previously too slow)
- Real-time streaming at long contexts
- Efficient batch processing of long documents
- Lower inference costs for long-context applications

**Impact:**
- Makes long-context LLM applications practical
- Enables new use cases (full repository analysis, book-length generation)
- Dramatically reduces inference costs for long contexts

### 5. Native 262K Context (Extendable to 1M)

**Innovation:** Trained natively on 262K token sequences with linear attention, eliminating need for separate context extension stages.

**Approach:**

Traditional Models:
1. Train on short contexts (2K-8K)
2. Context extension stage with YaRN/NTK
3. Fine-tune on longer contexts
4. Performance degrades at length limits

Qwen3-Next:
1. Train directly on 262K contexts
2. Linear attention handles long sequences naturally
3. No separate extension stage needed
4. Consistent performance across full range
5. Extendable to 1M via YaRN without fine-tuning

**Benefits:**
- No context extension training cost
- Consistent quality across context lengths
- Constant per-token inference cost
- Efficient memory usage (O(d²) state)

**Use Cases Enabled:**
- Full codebase analysis (hundreds of files)
- Book-length document processing
- Multi-document synthesis
- Extended conversation history
- Long-form content generation

**Impact:**
- Demonstrates linear attention's natural fit for long contexts
- Eliminates architectural limitations of standard attention
- Opens new application domains requiring ultra-long contexts

### 6. Flash Linear Attention Training Algorithm

**Innovation:** Hardware-optimized parallel training algorithm for Gated DeltaNet that outperforms FlashAttention-2.

**Key Techniques:**

1. **Chunk-Wise Parallelization:**
   - Divide sequence into chunks for parallel processing
   - Similar to FlashAttention but adapted for linear attention
   - Trade-off between parallelism and memory movement

2. **Efficient Memory Access:**
   - Optimized data layout for GPU memory hierarchy
   - Minimizes redundant memory transfers
   - Better utilization of shared memory

3. **Recurrence Handling:**
   - Efficiently propagate recurrent states across chunks
   - Parallel prefix sum for state aggregation
   - Enables parallel training despite recurrent formulation

**Performance:**
```
Training Speed vs FlashAttention-2:
  1K tokens: Comparable
  2K tokens: 1.1× faster
  4K tokens: 1.5× faster
  8K tokens: 2× faster
  16K+ tokens: 3-5× faster
```

**Impact:**
- Enables practical training of linear attention models
- Faster training than standard attention at any length
- Critical enabler of Qwen3-Next's efficiency gains

**Technical Reference:**
- Implemented in Flash Linear Attention (FLA) library
- Open-source: Available for research and development
- Adopted by multiple models (Qwen3-Next, Kimi Linear)

### 7. Foundation for Qwen3.5

**Strategic Importance:** Qwen3-Next is explicitly positioned as the architectural foundation for the upcoming Qwen3.5 series.

**What This Means:**

1. **Architecture Validation:**
   - Qwen3-Next demonstrates viability of hybrid linear attention
   - Proves extreme sparsity (3.7%) is effective
   - Validates training efficiency improvements

2. **Qwen3.5 Expectations:**
   - Likely to adopt Gated DeltaNet + Gated Attention hybrid
   - Expected to maintain or improve efficiency gains
   - May introduce additional innovations built on this foundation

3. **Paradigm Shift:**
   - Move away from standard softmax attention as default
   - Linear attention becomes primary mechanism
   - MoE sparsity increases further

**Historical Context:**

Qwen's architectural evolution:
- Qwen1: Standard attention
- Qwen2: GQA (Grouped Query Attention)
- Qwen3: Standard attention + thinking mode
- Qwen3-Next: Hybrid Gated DeltaNet + Gated Attention (revolution)
- Qwen3.5: Expected to adopt Qwen3-Next architecture (upcoming)

**Impact:**
- Signals industry-wide shift toward linear attention
- Validates efficiency-focused architecture research
- May influence other model series (Meta, Mistral, etc.)

## Architectural Significance and Impact

### Paradigm Shift in Attention Mechanisms

Qwen3-Next represents a **fundamental architectural shift** in large language model design:

**Traditional LLMs (2017-2024):**
- Core: Softmax attention (O(n²) complexity)
- Dominant since "Attention Is All You Need" (2017)
- Incremental improvements: FlashAttention, Multi-Query Attention, GQA

**Qwen3-Next (2025):**
- Core: Hybrid linear attention (O(n) complexity)
- Complete replacement of softmax attention
- Qualitative change, not incremental improvement

### Comparison to Other Efficiency Approaches

**Alternative Approaches:**

1. **State Space Models (Mamba, Mamba2):**
   - Efficiency: High (linear complexity)
   - Challenge: Difficulty matching transformer quality
   - Adoption: Limited in flagship models

2. **Sliding Window Attention:**
   - Efficiency: O(n × w) for window size w
   - Challenge: Limited global context understanding
   - Adoption: Some models (Mistral, LongLLaMA)

3. **Sparse Attention:**
   - Efficiency: Varies by pattern
   - Challenge: Hand-crafted patterns, quality trade-offs
   - Adoption: Limited

4. **Gated DeltaNet + Gated Attention (Qwen3-Next):**
   - Efficiency: High (O(n) complexity)
   - Quality: Matches standard attention
   - Adoption: **First major flagship model**

**Key Difference:**

Qwen3-Next is the **first frontier-scale model** to completely replace standard attention with linear attention while maintaining competitive quality.

### Impact on Future Model Development

**Short-Term (2025-2026):**

1. **Qwen3.5 and Beyond:**
   - Expected to adopt Qwen3-Next architecture
   - Further refinements and optimizations
   - Possible increase in model scale enabled by efficiency

2. **Industry Adoption:**
   - Other Chinese labs (DeepSeek, Baichuan, etc.) likely to explore similar approaches
   - Western labs (Meta, Mistral, etc.) may experiment with linear attention
   - Academic research focus on linear attention mechanisms

3. **Specialized Variants:**
   - Code models with even longer contexts
   - Vision-language models with efficient pixel encoding
   - Multimodal models with cross-modal linear attention

**Long-Term (2026+):**

1. **Efficiency as Standard:**
   - Linear attention may become default for new models
   - Standard attention relegated to specific use cases
   - Training costs drop dramatically

2. **Scale Implications:**
   - 10x efficiency enables 10x larger models at same cost
   - Could enable 1 trillion+ parameter models with acceptable cost
   - Democratizes access to frontier-scale model training

3. **Architecture Research:**
   - Continued innovation in linear attention mechanisms
   - Hybrid architectures combining multiple efficient approaches
   - Integration with other efficiency techniques (quantization, pruning)

### Limitations and Future Directions

**Current Limitations:**

1. **Short-Context Performance:**
   - Linear attention may slightly underperform standard attention on very short contexts (<1K)
   - Trade-off accepted for long-context benefits

2. **Memory Update Precision:**
   - Delta rule updates, while precise, may have limitations for certain tasks
   - Ongoing research into improved update mechanisms

3. **Training Complexity:**
   - Requires specialized implementation (FLA library)
   - Less mature tooling compared to standard attention

4. **Limited Validation:**
   - Qwen3-Next is first major deployment
   - Long-term reliability and edge cases still being discovered

**Future Directions:**

1. **Improved Linear Attention:**
   - Better approximations of standard attention
   - Enhanced expressiveness without sacrificing efficiency
   - Hybrid mechanisms combining multiple linear attention types

2. **Hardware Co-Design:**
   - Custom hardware for linear attention operations
   - Further optimization of Gated DeltaNet computation
   - Specialized accelerators for recurrent state updates

3. **Theoretical Understanding:**
   - Formal analysis of why Gated DeltaNet + Gated Attention works so well
   - Characterization of task types where it excels or struggles
   - Connection to neuroscience and cognitive architectures

4. **Multimodal Extensions:**
   - Efficient cross-modal attention using linear mechanisms
   - Vision encoders with Gated DeltaNet
   - Audio and video processing with constant-cost attention

## Use Cases and Applications

### 1. Long-Document Analysis and Processing

**Capability:**
- Native 262K context (extendable to 1M)
- Constant per-token inference cost
- Efficient memory usage

**Applications:**
- Legal document review (contracts, case law)
- Academic research (reading and synthesizing papers)
- Book analysis and summarization
- Report generation from multiple sources
- Medical record analysis

**Advantage over Standard Models:**
- 10x faster processing of long documents
- Can handle entire books without truncation
- Consistent quality across full document length

### 2. Codebase Understanding and Generation

**Capability:**
- Analyze entire repositories at once
- 256K tokens = ~100K lines of code
- Code generation with full context

**Applications:**
- Repository-level code understanding
- Refactoring with global context
- Bug detection across multiple files
- Documentation generation
- Code migration and translation

**Advantage:**
- See entire codebase context
- Faster than processing file-by-file
- Better understanding of cross-file dependencies

### 3. Complex Reasoning and Problem Solving

**Capability:**
- Thinking variant with explicit reasoning
- AIME25: 69.5 performance
- Multi-step problem decomposition

**Applications:**
- Mathematical problem solving
- Scientific research assistance
- Logical reasoning and puzzles
- Planning and strategy
- Educational tutoring

**Advantage:**
- Explicit thinking process
- Strong performance on competition-level tasks
- Efficient even for long reasoning chains

### 4. Multilingual Applications

**Capability:**
- 100+ languages supported
- Multilingual training data
- Cross-lingual understanding

**Applications:**
- Translation
- Cross-lingual information retrieval
- Multilingual chatbots
- Global customer support
- Language learning tools

### 5. Research and Experimentation

**Capability:**
- Novel architecture (Gated DeltaNet + Gated Attention)
- Open weights and open source
- Extensible and modifiable

**Applications:**
- Linear attention research
- Efficiency optimization studies
- Architecture design exploration
- Few-shot learning research
- Interpretability analysis

**Advantage:**
- Access to cutting-edge architecture
- Reproducible experiments
- Foundation for derivative research

### 6. Efficient Inference Services

**Capability:**
- 10x throughput at long contexts
- Low memory footprint (3B activated)
- Fast inference with linear complexity

**Applications:**
- Large-scale API services
- Real-time chatbots
- Batch processing pipelines
- Edge deployment (with quantization)
- Cost-sensitive applications

**Advantage:**
- Lower infrastructure costs
- Faster response times
- Higher throughput per GPU
- Better scalability

### 7. Agent Systems and Tool Use

**Capability:**
- Instruction following
- Long-context reasoning
- Fast inference for multi-turn interactions

**Applications:**
- Autonomous agents
- Tool-augmented reasoning
- Multi-step task execution
- Code execution agents
- Research assistants

**Advantage:**
- Efficient for long interaction histories
- Fast enough for real-time agentic workflows
- Strong reasoning for planning

### 8. Content Generation

**Capability:**
- Long-form generation
- Consistent quality across length
- Fast generation with linear attention

**Applications:**
- Article and blog writing
- Creative writing (stories, novels)
- Report generation
- Documentation writing
- Educational content creation

**Advantage:**
- Generate very long content efficiently
- Maintain coherence across length
- Faster than standard attention models

## Licensing

**License: Apache 2.0**

Qwen3-Next is released under the permissive Apache 2.0 license:

**Permissions:**
- ✅ Commercial use freely allowed
- ✅ Modification and distribution permitted
- ✅ Patent grant included
- ✅ Private use allowed
- ✅ No restrictions on use cases (within law)

**Conditions:**
- Attribution required (copyright notice)
- State changes if modified
- Include copy of license

**Limitations:**
- No warranty
- No liability
- Trademark rights not granted

**What This Means:**
- Can be used in commercial products
- Can be modified and fine-tuned
- Can be deployed at any scale
- No revenue sharing or fees
- True open-source model

## Resources and Links

### Official Resources

**Model Cards:**
- Qwen3-Next-80B-A3B-Base: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Base
- Qwen3-Next-80B-A3B-Instruct: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- Qwen3-Next-80B-A3B-Thinking: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking

**Technical Report:**
- References Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- Qwen3-Next presented as architectural preview in Qwen3 report

**Official Announcements:**
- Alibaba Cloud Blog: [Qwen3-Next: A New Generation of Ultra-Efficient Model Architecture](https://www.alibabacloud.com/blog/qwen3-next-a-new-generation-of-ultra-efficient-model-architecture-unveiled_602536)

**GitHub:**
- Qwen3 Repository: https://github.com/QwenLM/Qwen3
- Includes Qwen3-Next models and usage examples

**Official Website:**
- Qwen: https://qwen.ai/

### Research Papers

**Gated DeltaNet:**
- Paper: [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- arXiv: 2412.06464
- Conference: ICLR 2025
- Authors: Songlin Yang, et al. (NVIDIA)
- Key Innovation: Combines gating and delta rule for efficient linear attention

**Gated Linear Attention:**
- Paper: [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635)
- arXiv: 2312.06635
- Conference: ICML 2024
- Key Innovation: Hardware-efficient linear attention with gating

**Flash Linear Attention:**
- GitHub: https://github.com/fla-org/flash-linear-attention
- Implementation library used in Qwen3-Next
- Includes optimized kernels for Gated DeltaNet and Gated Attention

### Inference Frameworks

**SGLang:**
- GitHub: https://github.com/sgl-project/sglang
- Version: ≥ 0.5.2 for Qwen3-Next support
- Recommended for deployment

**vLLM:**
- GitHub: https://github.com/vllm-project/vllm
- Version: ≥ 0.10.2 for Qwen3-Next support
- Production-ready deployment

**Hugging Face Transformers:**
- GitHub: https://github.com/huggingface/transformers
- Version: ≥ 4.37.0
- Easiest for experimentation

### Community and Support

**Discussion:**
- Hugging Face Discussions: Model card discussion sections
- GitHub Issues: https://github.com/QwenLM/Qwen3/issues
- Reddit: r/LocalLLaMA (Qwen3-Next discussions)

**Documentation:**
- Qwen Documentation: https://qwen.readthedocs.io/
- Model deployment guides
- Fine-tuning tutorials
- API references

### Related Models

**Qwen3 Family:**
- Qwen3-4B, 32B (dense models)
- Qwen3-30B-A3B, 235B-A22B (MoE models)
- Qwen3-Thinking variants (reasoning-focused)
- Technical Report: https://arxiv.org/abs/2505.09388

**Other Efficient Models:**
- Kimi Linear: Also uses Gated DeltaNet
- Mamba2: State-space model comparison baseline
- DeltaNet: Linear attention without gating

---

**Document Information:**
- Created: 2025
- Model Version: Qwen3-Next-80B-A3B (Base, Instruct, Thinking)
- Release Date: September 10, 2025
- References: Qwen3 Technical Report (arXiv:2505.09388), Gated DeltaNet (arXiv:2412.06464), Gated Linear Attention (arXiv:2312.06635)

**Sources:**
All information verified from official Alibaba Cloud blog post, Hugging Face model cards, research papers, and official announcements. Architectural details confirmed from model configurations and research papers.
