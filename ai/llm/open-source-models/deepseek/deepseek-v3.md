# DeepSeek-V3

## Overview

DeepSeek-V3 is a Mixture-of-Experts (MoE) language model developed by DeepSeek AI and released on December 25, 2024. With 671 billion total parameters and 37 billion activated per token (~5.5% activation rate), it represents a major milestone in cost-efficient large-scale model training. The model was trained on 14.8 trillion high-quality tokens for only $5.57 million USD (2.788M H800 GPU hours), making it the most cost-effective training of any 671B parameter model to date.

DeepSeek-V3 achieves performance comparable to leading closed-source models like GPT-4o and Claude 3.5 Sonnet while being fully open-source. It introduces several groundbreaking innovations including auxiliary-loss-free load balancing, multi-token prediction training, FP8 mixed precision training at massive scale, and the DualPipe bidirectional pipeline parallelism algorithm.

**Key Highlights:**
- 671B total parameters, 37B activated per token
- Trained on 14.8 trillion tokens for $5.57M USD
- 128K token context window
- Performance comparable to GPT-4o and Claude 3.5 Sonnet
- No loss spikes or rollbacks during entire training
- FP8 native training with BF16 conversion available
- MIT license (DeepSeek-V3-0324 and later versions)

## Model Architecture

### Core Architecture Specifications

DeepSeek-V3 uses a Transformer-based architecture with 61 layers:
- **3 dense layers**: Standard Transformer layers
- **58 MoE layers**: Mixture-of-Experts layers with 256 routed experts each

**Architecture Parameters:**
```yaml
Total Parameters: 671 billion (671B)
Non-embedding Parameters: 670B
Activated Parameters per Token: 37 billion (37B)
Activation Rate: 5.5% (37B/671B)

Layers: 61 total
  - Dense Layers: 3
  - MoE Layers: 58

Hidden Dimension: 7,168
Intermediate Size: 18,432
Attention Heads: 128
Head Dimension: 128 per head
Vocabulary Size: 128,000 tokens

Context Window:
  - Training: 4,096 tokens (native)
  - Extended: 32,768 tokens (Stage 1)
  - Final: 128,000 tokens (Stage 2 with YaRN)
  - API: 64,000 tokens (commercial)
```

### Multi-Head Latent Attention (MLA)

DeepSeek-V3 uses an enhanced version of Multi-Head Latent Attention (MLA), first introduced in DeepSeek-V2. MLA achieves massive KV cache reduction through low-rank joint compression of attention keys and values while maintaining performance comparable to standard Multi-Head Attention (MHA).

**MLA Configuration:**
```yaml
Attention Heads: 128
Head Dimension: 128
Total Attention Dimension: 16,384 (128 × 128)

Compression:
  KV Latent Dimension (dc): 512
  Query Hidden Space (dh): 1,536
  Decoupled RoPE Dimension per Head (dR_h): 64

KV Cache Size per Token:
  DeepSeek-V3: 70.272 KB
  Qwen-2.5 72B: 327.68 KB
  LLaMA-3.1 405B: 512 KB

Cache Reduction:
  vs Qwen-2.5: 4.66× smaller
  vs LLaMA-3.1: 7.28× smaller
  vs Standard MHA: 93.3% reduction
```

**How MLA Works:**

1. **Low-Rank Compression**: Instead of storing full key and value vectors, MLA compresses them into lower-dimensional latent vectors
2. **Down-Projection**: Keys and values are projected down to latent dimension dc = 512
3. **Up-Projection**: During attention computation, latent vectors are projected back up to attention space
4. **Decoupled RoPE**: Separate RoPE components (qR and kR) with dimension dR_h per head preserve positional information without disrupting compression

**Architecture Details:**
```
Input: x (hidden state)

Compute Compressed Latent:
  c_t = W^(D) x_t  (down-projection to dimension dc)

Compute Query:
  q_t = W^(Q) c_t  (up-project to query space)
  qR_t = W^(QR) x_t  (compute RoPE query component)
  q_t = [q_t; qR_t]  (concatenate)

Compute Key-Value:
  k_t = W^(K) c_t  (up-project to key space)
  kR_t = W^(KR) x_t  (compute RoPE key component, shared across heads)
  k_t = [k_t; kR_t]  (concatenate)
  v_t = W^(V) c_t  (up-project to value space)

Cache: Only cache c_t (512-dim) instead of full k_t, v_t
```

**Benefits:**
- **Memory Efficiency**: 93.3% KV cache reduction enables much longer context at same memory
- **Inference Speed**: Smaller cache means faster memory access and higher throughput
- **Performance Preservation**: Maintains quality comparable to standard MHA
- **Positional Encoding**: Decoupled RoPE preserves positional sensitivity with compression

### Mixture-of-Experts (MoE) Architecture

DeepSeek-V3 employs the DeepSeekMoE architecture with auxiliary-loss-free load balancing, a pioneering innovation that achieves expert load balance without auxiliary loss functions.

**MoE Configuration:**
```yaml
MoE Layers: 58 out of 61 total layers

Per MoE Layer:
  Shared Experts: 1
  Routed Experts: 256
  Activated Experts: 8 per token
  Expert Hidden Dimension: 2,048

Routing:
  Strategy: Top-8 routing with node-limited selection
  Node Limit: Maximum 4 nodes per token
  Router Type: Sigmoid affinity scores
  Load Balancing: Auxiliary-loss-free with dynamic bias

Total Activated Parameters:
  - Dense layers: ~1B
  - MoE layers: ~36B (58 layers × 8 experts × ~80M per expert)
  - Total: ~37B per token
```

**Auxiliary-Loss-Free Load Balancing:**

Traditional MoE models use auxiliary loss functions to encourage balanced expert utilization, but this can hurt performance. DeepSeek-V3 pioneers a novel approach:

**Method:**
1. **Dynamic Bias Terms**: Each expert has a bias term b_i added to its affinity score
2. **Load Monitoring**: Expert load is monitored at each training step
3. **Bias Adjustment**:
   - If expert i is **overloaded**: decrease bias by γ (b_i ← b_i - γ)
   - If expert i is **underloaded**: increase bias by γ (b_i ← b_i + γ)
4. **Dynamic Equilibrium**: Bias adjustments push the system toward balanced expert utilization

**Complementary Loss:**
A small sequence-wise balance loss with coefficient α = 0.0001 provides additional stability:
```
L_balance = α × sequence_wise_balance_loss
```

**Benefits:**
- **No Token Dropping**: All tokens are processed (unlike capacity-based routing)
- **Better Performance**: No performance degradation from auxiliary loss
- **Simpler Training**: Eliminates auxiliary loss hyperparameter tuning
- **Stable Load Balance**: Achieves balance through dynamic adjustment

**Node-Limited Routing:**

To reduce cross-node communication overhead in distributed training:
- Each token activates experts from at most 4 different nodes
- Reduces all-to-all communication latency
- Maintains expert diversity while improving training efficiency

### Position Embeddings

**RoPE (Rotary Position Embeddings):**
- Base frequency (theta): 10,000
- Native training context: 4,096 tokens
- Extended context: 32K → 128K tokens using YaRN scaling

**Decoupled RoPE in MLA:**
- Separate RoPE components: qR (per-head) and kR (shared across heads)
- Dimension per head (dR_h): 64
- Applied separately from attention computation
- Preserves positional sensitivity without disrupting low-rank KV compression

**YaRN Scaling Configuration:**
```yaml
scaling_factor: Applied when seq_len > original_max_position_embeddings
original_max_position_embeddings: 4096
beta_fast: 32
beta_slow: 1
mscale: 1.0
mscale_all_dim: 0.0
```

### Activation Function

**SwiGLU (Swish Gated Linear Unit):**
```python
def SwiGLU(x):
    x1, x2 = split(x)  # Split into two halves
    return Swish(x1) * x2

def Swish(x):
    return x * sigmoid(x)
```

**Configuration:**
- Intermediate dimension: 18,432
- GLU dimension: ~2/3 of hidden dimension
- Applied in both dense and expert feed-forward layers

**Benefits:**
- Better gradient flow than ReLU
- Smoother activation than GELU
- Gating mechanism provides selective activation

### Normalization

**RMSNorm (Root Mean Square Normalization):**
```python
def RMSNorm(x, weight):
    rms = sqrt(mean(x^2) + epsilon)
    return (x / rms) * weight
```

**Configuration:**
- Applied before attention and feed-forward layers (pre-normalization)
- Precision: BF16 or FP32 for numerical stability
- Epsilon: Small constant (typically 1e-6) for numerical stability

**Additional Normalization:**
- RMSNorm layers after compressed latent vectors in MLA
- Scaling factors at width bottlenecks for stable training

**Benefits vs LayerNorm:**
- 15% faster computation (no mean calculation)
- Better numerical stability
- Simpler implementation

### Tokenizer

**Byte-level BPE (Byte Pair Encoding):**
```yaml
Vocabulary Size: 128,000 tokens (128K)
Type: Byte-level BPE
Character Set: Unicode (multilingual)

Optimizations:
  - Modified pretokenizer for improved multilingual compression
  - New tokens for punctuation + line break combinations
  - Improved code tokenization efficiency
```

**Improvements over DeepSeek-V2:**
- Better multilingual compression (especially for non-English languages)
- More efficient code tokenization
- Handles punctuation and formatting more effectively

## Multi-Token Prediction (MTP)

DeepSeek-V3 incorporates Multi-Token Prediction (MTP) as a training objective to improve data efficiency and model performance. Unlike parallel multi-token prediction approaches, DeepSeek-V3 uses **sequential prediction** that maintains the complete causal chain at each prediction depth.

### MTP Architecture

**Configuration:**
```yaml
Additional MTP Parameters: 14 billion (14B)
Total Model Size: 685B (671B main + 14B MTP)

MTP Modules per Layer:
  - D transformer blocks (one per prediction depth)
  - Shared embedding layer
  - Shared output heads
  - Cross-entropy loss per depth

Default Prediction Depth (D): 2 tokens ahead
```

**How MTP Works:**

1. **Primary Prediction**: Model predicts next token (t+1) as usual
2. **Secondary Prediction**: Additional modules predict token (t+2) based on predicted (t+1)
3. **Sequential Chain**: Each depth predicts based on previous prediction, maintaining causality
4. **Joint Training**: Loss is averaged across prediction depths and weighted by λ

**Loss Function:**
```python
# Primary token prediction
loss_1 = CrossEntropy(logits_t1, target_t1)

# Secondary token prediction (based on predicted t+1)
loss_2 = CrossEntropy(logits_t2, target_t2)

# Combined loss
total_loss = (loss_1 + λ * loss_2) / (1 + λ)
```

**Training Schedule:**
- Early training: λ = 0.3 (30% weight on MTP)
- Later training: λ = 0.1 (10% weight on MTP)
- Gradually reduces MTP weight as model matures

### Benefits of MTP

**1. Improved Data Efficiency:**
- Densifies training signals by predicting multiple tokens per forward pass
- Effectively trains on more future context per training step
- Better utilization of training compute

**2. Better Representations:**
- Forces model to "pre-plan" representations for better future token prediction
- Encourages learning more generalizable features
- Improves understanding of long-range dependencies

**3. Speculative Decoding:**
- MTP modules can be used for speculative decoding during inference
- Predict multiple tokens ahead, verify correctness, accept if valid
- Potential inference speedup without quality loss

**4. Performance Gains:**
- Measurable improvements on benchmarks (exact gains not disclosed)
- Particularly helpful for reasoning and code generation tasks
- Better understanding of context and dependencies

### Sequential vs Parallel MTP

**DeepSeek-V3 Sequential Approach:**
- Predict t+2 based on predicted t+1
- Maintains complete causal chain
- More realistic prediction scenario (mirrors actual generation)

**Alternative Parallel Approach:**
- Predict t+1 and t+2 independently from same context
- Simpler but less realistic
- Doesn't capture dependencies between predictions

### Inference Usage

**Training Time:**
- MTP modules active, contribute to loss
- 685B total parameters (671B + 14B MTP)

**Inference Time (Standard):**
- MTP modules can be discarded
- Main 671B model works independently
- Reduces memory footprint

**Inference Time (Speculative Decoding):**
- Keep MTP modules for speculative decoding
- Use secondary predictions as candidates
- Verify and accept when correct, rollback when wrong

## Training Details

### Pre-Training

**Training Data:**
```yaml
Total Tokens: 14.8 trillion tokens
Diversity: High-quality diverse tokens across domains

Data Composition:
  - Web text: Large-scale crawled web data
  - PDFs: Extracted via Qwen2.5-VL for better quality
  - Code: 80+ programming languages
  - STEM: Scientific, technical, engineering, mathematics
  - Multilingual: 119+ languages
  - Synthetic: Generated reasoning and dialogue data

Data Quality:
  - Extensive filtering and deduplication
  - Quality scoring and selection
  - Contamination detection and removal

Knowledge Cutoff: July 2024
```

**Note:** Exact data composition percentages are NOT disclosed in the technical report.

**Training Infrastructure:**
```yaml
GPUs: 2,048 NVIDIA H800 GPUs
Training Time: ~2 months

GPU Hours:
  - Pre-training: 2.664M H800 GPU hours
  - Post-training: 0.124M H800 GPU hours
  - Total: 2.788M H800 GPU hours

Training Cost: $5.576 million USD (at $2/GPU-hour)

Efficiency: 180K H800 hours per trillion tokens
  - Training speed: 3.7 days per trillion tokens on 2,048-GPU cluster
```

**Comparison:**
- GPT-4 training cost (estimated): ~$100 million (2023)
- LLaMA-3.1 405B training: Significantly more expensive
- DeepSeek-V3: **Most cost-efficient 671B parameter model training**

**Parallelism Strategy:**
```yaml
Pipeline Parallelism (PP): 16-way
Expert Parallelism (EP): 64-way
Data Parallelism (DP): ZeRO-1 optimizer
Total Configuration: 16-PP × 64-EP × ZeRO-1-DP

Node Configuration:
  - NVLink: Within-node GPU interconnect
  - InfiniBand: Cross-node communication
  - SM Allocation: 20 out of 132 SMs for communication overlap
```

**Training Schedule:**

**Phase 1: Warmup (2,200 steps)**
```yaml
Initial Learning Rate: 0
Peak Learning Rate: 2.2 × 10⁻⁴
Warmup Strategy: Linear increase
```

**Phase 2: Constant Learning Rate (until 10T tokens)**
```yaml
Learning Rate: 2.2 × 10⁻⁴ (constant)
Duration: ~10 trillion tokens
```

**Phase 3: Cosine Decay (4.3T tokens)**
```yaml
Starting LR: 2.2 × 10⁻⁴
Ending LR: 2.2 × 10⁻⁵
Decay Strategy: Cosine schedule
Token Range: 10T → 14.3T tokens
```

**Phase 4: Final Constant (remaining tokens)**
```yaml
Learning Rate: 7.3 × 10⁻⁶
Duration: ~500B tokens (to 14.8T)
```

**Batch Size Schedule:**
```yaml
Initial Batch Size: 3,072 tokens per step
Final Batch Size: 15,360 tokens per step
Strategy: Gradual increase during training
```

**Training Stability:**
- **Zero loss spikes**: No irrecoverable loss spikes during entire training
- **Zero rollbacks**: No need to rollback to previous checkpoints
- **Smooth training**: Extremely stable training dynamics

This is remarkable for a 671B parameter model and demonstrates the effectiveness of DeepSeek-V3's architectural innovations and training strategies.

### FP8 Mixed Precision Training

DeepSeek-V3 is the **first validation of FP8 training at extremely large scale** (671B parameters). FP8 training reduces memory footprint and increases training speed while maintaining model quality.

**FP8 Configuration:**

**High-Precision Components (BF16/FP32):**
```yaml
Components:
  - Embedding layers
  - Output layers (LM head)
  - Normalization operators (RMSNorm)
  - MoE gating modules (router)
  - Attention operations (QKV computation)
  - Gradient accumulation
  - Optimizer states (momentum tracking)
```

**Low-Precision Components (FP8):**
```yaml
Activations:
  - Format: E4M3 (4-bit exponent, 3-bit mantissa)
  - Quantization: Tile-wise (1 × 128 tiles)
  - Scope: Before MoE up-projections

Weights:
  - Format: E4M3
  - Quantization: Block-wise (128 × 128 blocks)
  - Scope: Linear layers in FFN and experts
```

**Key Challenge:**

H800 GPUs have limited FP8 GEMM accumulation precision (~14 bits). DeepSeek-V3 addresses this:

1. **Selective Quantization**: Only quantize activations before MoE up-projections
2. **High-Precision Paths**: Keep forward/backward combine components in BF16
3. **Cached Recomputation**: Cache SwiGLU inputs in FP8, recompute in BF16 during backward pass
4. **Optimizer in BF16**: Maintain momentum and parameter updates in higher precision

**Training Quality:**
```yaml
Relative Loss Error: <0.25% (consistently)
Acceptable Range: Well within training randomness
Convergence: Equivalent to BF16 training
```

**Benefits:**
- **Memory Reduction**: ~50% memory savings for activations and weights
- **Training Speed**: Faster GEMM operations with FP8
- **Cost Efficiency**: Contributes to $5.57M training cost
- **Quality Preservation**: <0.25% relative loss error

**Model Distribution:**
- Models are released in **FP8 format** (native training format)
- Conversion scripts provided for BF16 inference
- FP8 inference supported by SGLang, vLLM, LMDeploy

### DualPipe Training Algorithm

DualPipe is DeepSeek-V3's novel **bidirectional pipeline parallelism algorithm** that overlaps computation and communication to maximize training efficiency.

**Problem Addressed:**

In cross-node expert parallelism, computation-to-communication ratio is approximately 1:1:
- Computation time: Running MLP/expert operations
- Communication time: All-to-all dispatch and combine operations

Traditional pipeline parallelism has significant bubble time where GPUs wait idle.

**DualPipe Solution:**

**Key Idea**: Overlap computation and communication within forward/backward chunk pairs through bidirectional scheduling.

**Architecture:**
```yaml
Pipeline Parallelism: 16-way (16 pipeline stages)
Pipeline Stages: Each stage handles multiple layers

Stage Assignment:
  - Shallowest layers: Placed on PP rank 0
  - Deepest layers: Placed on PP rank 0 (same rank!)
  - Middle layers: Distributed across other PP ranks
  - Benefit: Parameter sharing between first and last layers
```

**Computation Phases per Chunk:**
1. **Attention**: Self-attention computation
2. **All-to-All Dispatch**: Send tokens to expert nodes
3. **MLP/Experts**: Expert computation
4. **All-to-All Combine**: Gather results from expert nodes

**Bidirectional Scheduling:**
```
Forward Pass:
  Rank 0: [Attn][Dispatch][MLP][Combine] → Rank 1
  Rank 1: [Attn][Dispatch][MLP][Combine] → Rank 2
  ...
  Rank 15: [Attn][Dispatch][MLP][Combine] → Rank 0

Backward Pass (reverse direction):
  Rank 0: [Combine][MLP][Dispatch][Attn] → Rank 15
  Rank 15: [Combine][MLP][Dispatch][Attn] → Rank 14
  ...

Overlap:
  - While Rank 0 does forward computation, Rank 15 does backward
  - Communication and computation overlap within chunks
  - Minimizes pipeline bubbles
```

**SM Allocation for Overlap:**
```yaml
Total SMs per H800: 132
Dedicated to Communication: 20 SMs
Dedicated to Computation: 112 SMs

Strategy:
  - Communication tasks run on dedicated SMs
  - Computation tasks run on remaining SMs
  - Enables true simultaneous computation-communication overlap
```

**Pipeline Bubble Reduction:**
```
Traditional Pipeline: (PP - 1) × (Forward + Backward + Weight_Update)
DualPipe: (PP/2 - 1) × (Forward + Backward - 3 × Weight_Update)

With PP=16:
  Traditional: 15 × (F + B + W)
  DualPipe: 7 × (F + B - 3W)

Reduction: ~53% fewer bubble cycles
```

**Benefits:**
- **Higher GPU Utilization**: Less idle time waiting for communication
- **Faster Training**: Reduced pipeline bubbles mean faster iteration
- **Better Scaling**: Maintains efficiency at large scale
- **Cost Savings**: Contributes to overall training cost efficiency

### Long Context Extension

DeepSeek-V3 was trained on 4K token sequences but extended to 128K tokens using a two-stage approach with YaRN (Yet another RoPE extensioN method).

**Stage 1: 4K → 32K Extension**
```yaml
Target Context: 32,768 tokens
Training Steps: 1,000 steps
Batch Size: 1,920 tokens per step
Total Tokens: ~1.92 million tokens
Method: YaRN scaling of RoPE frequencies
```

**Stage 2: 32K → 128K Extension**
```yaml
Target Context: 128,000 tokens
Training Steps: 1,000 steps
Batch Size: 480 tokens per step
Total Tokens: ~480K tokens
Method: Further YaRN scaling
```

**YaRN Configuration:**
```yaml
scaling_factor: Dynamic based on sequence length
beta_fast: 32
beta_slow: 1
mscale: 1.0
mscale_all_dim: 0.0
```

**Validation:**
- **NIAH (Needle In A Haystack)**: Perfect retrieval across all context lengths up to 128K
- **Performance**: Maintains quality across entire 128K context window
- **API Limit**: Commercial API limited to 64K tokens (input + output combined)

**Benefits:**
- Efficient long-document understanding
- Code repository analysis
- Multi-document reasoning
- Long conversation history

### Post-Training

DeepSeek-V3 undergoes extensive post-training to align with human preferences and unlock its full potential.

**Supervised Fine-Tuning (SFT):**
```yaml
Training Data: 1.5 million instruction instances
Epochs: 2
Learning Rate Schedule: 5 × 10⁻⁶ → 1 × 10⁻⁶

Data Composition:
  - General instruction following
  - Mathematical reasoning
  - Code generation and debugging
  - Multilingual dialogue
  - R1-distilled reasoning data with reflection patterns
  - Long-form reasoning chains

Strategy:
  - Balance accuracy and generation length
  - Incorporate DeepSeek-R1 reasoning capabilities
  - Maintain general capabilities while adding instruction following
```

**Reinforcement Learning (RL):**
```yaml
Algorithm: GRPO (Group Relative Policy Optimization)

Reward Models:
  - Rule-based rewards: Correct answer, format compliance
  - Model-based rewards: Quality, helpfulness, harmlessness
  - Combined approach for robust training

Training:
  - GPU Hours: ~0.1M H800 GPU hours (small fraction of pre-training)
  - Optimization: Balance multiple reward signals
  - Stability: Careful hyperparameter tuning to avoid collapse
```

**DeepSeek-R1 Integration:**

DeepSeek-V3-0324 (March 2025) incorporates reasoning capabilities distilled from DeepSeek-R1:
- **Improved reasoning**: Better step-by-step thinking
- **Better math performance**: Improved on AIME, MATH benchmarks
- **Enhanced code**: Better code generation and debugging
- **Reflection patterns**: Self-correction and verification

**Post-Training Philosophy:**
- Unlock model potential without over-constraining
- Balance accuracy with appropriate generation length
- Preserve general capabilities while adding specific skills
- Maintain robustness and safety

## Performance Benchmarks

### DeepSeek-V3-Base

**General Knowledge:**
```yaml
MMLU (5-shot): 87.1
  vs LLaMA-3.1 405B: 84.4
  vs Qwen2.5 72B: 85.9
```

**Mathematical Reasoning:**
```yaml
MATH (4-shot): 61.6
  vs LLaMA-3.1 405B: 49.0
  vs Qwen2.5 72B: 59.7

GSM8K (8-shot): 89.3
  vs LLaMA-3.1 405B: 89.0
  vs Qwen2.5 72B: 88.3
```

**Code Generation:**
```yaml
HumanEval (pass@1): 65.2
  vs LLaMA-3.1 405B: 54.9
  vs Qwen2.5 72B: 52.4

LiveCodeBench (pass@1): 19.4
  vs LLaMA-3.1 405B: 15.5
  vs Qwen2.5 72B: 18.3
```

**Long Context:**
```yaml
NIAH (Needle In A Haystack): Perfect retrieval up to 128K tokens
Context Window: 128,000 tokens with consistent performance
```

### DeepSeek-V3 (Chat)

**General Benchmarks:**
```yaml
MMLU: 88.5
  vs GPT-4o: 87.2
  vs Claude-3.5-Sonnet: 88.3

Arena-Hard: 85.5
  vs GPT-4o: 79.3
  vs Claude-3.5-Sonnet: 85.2
```

**Mathematical Reasoning:**
```yaml
Math-500: 90.2
  vs GPT-4o: 74.6
  vs Claude-3.5-Sonnet: 78.3

AIME 2024: ~30-40% (estimated, base model)
  Note: DeepSeek-R1 achieves much higher with reasoning
```

**Code Generation:**
```yaml
LiveCodeBench: Strong performance
HumanEval: Competitive with GPT-4o and Claude-3.5

Codeforces Percentile: 51.6
  Strongest among evaluated models
```

**Instruction Following:**
```yaml
IFEval: High performance
Comparable to Claude-3.5-Sonnet and GPT-4o
```

### DeepSeek-V3-0324 (March 2025)

**Improvements over Original V3:**
- Better reasoning performance (R1 distillation)
- Improved math and coding benchmarks
- Enhanced instruction following
- Better generation length balance

**Key Advances:**
- Outperforms GPT-4.5 on math and coding benchmarks (claimed)
- Maintains general capabilities while adding reasoning
- Full MIT license (major licensing improvement)

### Efficiency Metrics

**Training Efficiency:**
```yaml
Cost per Trillion Tokens: 180K H800 GPU hours
Training Speed: 3.7 days per trillion tokens (2,048 GPUs)
Total Cost: $5.576 million USD for 14.8T tokens
Cost per Effective Parameter: $8.31 per billion parameters (37B activated)
```

**Inference Efficiency:**
```yaml
Compute per Token: 250 GFLOPS
  vs LLaMA-3.1 405B: 2,448 GFLOPS (9.8× more)
  vs Qwen2.5 72B: ~750 GFLOPS (3× more)

Generation Throughput: Up to 5.76× higher than comparable models
Memory Footprint:
  - FP8: ~700 GB
  - BF16: ~1.4 TB
  - INT8: ~350 GB (quantized)
```

**KV Cache Efficiency:**
```yaml
Cache per Token: 70.272 KB
  vs Qwen-2.5 72B: 327.68 KB (4.66× reduction)
  vs LLaMA-3.1 405B: 512 KB (7.28× reduction)
```

## Model Variants

### DeepSeek-V3-Base (December 25, 2024)

**Specifications:**
- 671B total parameters, 37B activated
- Pre-trained foundation model (no instruction tuning)
- 128K context window
- Trained on 14.8T tokens

**Use Cases:**
- Research and experimentation
- Fine-tuning for specific domains
- Benchmarking and evaluation
- Foundation for derivative models

**Licensing:**
- Code: MIT License
- Model: Custom DeepSeek Model License (commercial use allowed)

### DeepSeek-V3 (Chat) (December 25, 2024)

**Specifications:**
- 671B total parameters, 37B activated
- Instruction-tuned for dialogue
- 128K context window (64K API)
- SFT + RL post-training

**Use Cases:**
- General conversation and assistance
- Question answering
- Code generation and debugging
- Mathematical reasoning
- Multilingual dialogue

**Licensing:**
- Code: MIT License
- Model: Custom DeepSeek Model License

### DeepSeek-V3-0324 (March 24, 2025)

**Specifications:**
- 671B → 685B total parameters (includes MTP modules)
- 37B activated per token
- 128K context window
- Improved post-training with R1 distillation

**Key Improvements:**
- Better reasoning capabilities (R1 RL techniques)
- Improved math and coding performance
- Enhanced instruction following
- Better generation length balance
- Outperforms GPT-4.5 on math/coding (claimed)

**Major Change:**
- **Full MIT License** for both code and model
- No usage restrictions
- No fees or profit sharing
- Commercial use freely allowed

**Use Cases:**
- Production deployment without licensing concerns
- Commercial applications
- Research requiring permissive licensing
- Derivative model development

### DeepSeek-V3.1 (August 2025)

**Specifications:**
- 840B total parameters (estimated)
- Hybrid reasoning model
- Thinking and non-thinking modes in single model
- MIT License

**Improvements:**
- Smarter tool calling
- Higher thinking efficiency
- Better reasoning performance
- More efficient inference

### DeepSeek-V3.1-Terminus (September 22, 2025)

**Specifications:**
- 671B total, 37B activated (same structure as V3)
- Enhanced agent capabilities
- FP8 microscaling for efficient inference
- MIT License

**Key Features:**
- "Finale to V3 era"
- Language consistency improvements
- Better agent interactions
- Optimized for production deployment

### DeepSeek-V3.2-Exp (September 29, 2025)

**Specifications:**
- 671B-685B total parameters
- Experimental with DeepSeek Sparse Attention (DSA)
- MIT License

**Innovations:**
- Fine-grained sparse attention for efficiency
- API prices dropped 50%+
- Performance on par with V3.1-Terminus
- Focus on inference cost reduction

## Deployment and Inference

### Hardware Requirements

**Minimum Requirements:**
```yaml
Operating System: Linux
Python Version: 3.10+
GPU Memory: ~700 GB for FP8, ~1.4 TB for BF16

Recommended Setup:
  - Multi-node deployment (model too large for single node)
  - 4 nodes with 8 GPUs each (typical configuration)
  - High-bandwidth interconnect (InfiniBand or equivalent)
  - NVLink for within-node communication
```

**Single-Node Limitations:**
- DeepSeek-V3 is too large for single-node deployment
- Requires tensor parallelism across multiple nodes
- Minimum 2 nodes recommended, 4 nodes optimal

### Supported Inference Frameworks

**SGLang (Recommended):**
```yaml
Features:
  - MLA optimizations
  - FP8 and BF16 support
  - AMD and NVIDIA GPU support
  - Multi-node deployment
  - Highest throughput

Installation: pip install sglang
Documentation: https://github.com/sgl-project/sglang
```

**vLLM (v0.6.6+):**
```yaml
Features:
  - Pipeline parallelism
  - Tensor parallelism
  - FP8 and BF16 support
  - Paged attention
  - Both AMD and NVIDIA GPUs

Installation: pip install vllm>=0.6.6
Documentation: https://github.com/vllm-project/vllm
```

**LMDeploy:**
```yaml
Features:
  - Offline and online serving
  - PyTorch integration
  - Efficient FP8 and BF16 inference
  - Production-ready

Installation: pip install lmdeploy
Documentation: https://github.com/InternLM/lmdeploy
```

**TensorRT-LLM:**
```yaml
Features:
  - BF16 support
  - INT4/INT8 quantization
  - FP8 support (in development)
  - NVIDIA GPU optimized
  - Production deployment

Documentation: https://github.com/NVIDIA/TensorRT-LLM
```

**LightLLM:**
```yaml
Features:
  - Single and multi-node deployment
  - FP8 and BF16 support
  - Lightweight implementation

Documentation: https://github.com/ModelTC/lightllm
```

**Huawei Ascend NPU (MindIE):**
```yaml
Features:
  - INT8 and BF16 support
  - Huawei NPU optimization
  - Production deployment on Ascend hardware
```

### Quantization Options

**FP8 (Native):**
- Model distributed in FP8 format
- ~700 GB memory footprint
- Highest inference speed
- Minimal quality loss (<0.25%)
- Supported by SGLang, vLLM, LMDeploy

**BF16 (Converted):**
- Conversion from FP8 using provided scripts
- ~1.4 TB memory footprint
- Slightly higher quality than FP8
- Slower inference than FP8
- Supported by all frameworks

**INT8 (Quantized):**
- ~350 GB memory footprint
- Moderate speed
- Some quality degradation
- Supported by TensorRT-LLM, Huawei MindIE

**INT4 (Quantized):**
- ~175 GB memory footprint
- Fastest inference
- Noticeable quality degradation
- Supported by TensorRT-LLM
- Use for resource-constrained scenarios

### Conversion and Setup

**FP8 to BF16 Conversion:**
```bash
# Clone repository
git clone https://github.com/deepseek-ai/DeepSeek-V3
cd DeepSeek-V3

# Download model
huggingface-cli download deepseek-ai/DeepSeek-V3 \
  --local-dir /path/to/model

# Convert to inference format
python convert.py \
  --hf-ckpt-path /path/to/DeepSeek-V3 \
  --save-path /path/to/output \
  --n-experts 256 \
  --model-parallel 16
```

**Multi-Node Inference:**
```bash
# Run across 2 nodes with 8 GPUs each
torchrun \
  --nnodes 2 \
  --nproc-per-node 8 \
  --master-addr <master_node_ip> \
  --master-port <port> \
  generate.py \
    --ckpt-path /path/to/model \
    --config configs/config_671B.json
```

**SGLang Deployment:**
```bash
# Single command deployment
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 16 \
  --trust-remote-code
```

**vLLM Deployment:**
```bash
# Deploy with tensor parallelism
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 16 \
  --dtype float16 \
  --max-model-len 65536
```

### Inference Performance

**Throughput (2000-token sequences):**
```yaml
Single Node (8× H800):
  - Input: 52.3K tokens/second
  - Output: 22.3K tokens/second

Four-Node Setup:
  - Input: ~200K+ tokens/second
  - Output: ~90K+ tokens/second
```

**Latency:**
```yaml
First Token Latency: ~100-200ms (depending on context length)
Per-Token Latency: ~40-50ms per token
```

**Memory Usage:**
```yaml
FP8 Weights: ~700 GB
KV Cache (128K context): ~70 KB per token × batch size × sequence length
Total Memory (64K context, batch 8): ~700 GB + 35 GB = ~735 GB
```

### API Access

**Official API:**
```yaml
Chat Interface: https://chat.deepseek.com
API Platform: https://platform.deepseek.com
API Type: OpenAI-compatible
Context Limit: 64,000 tokens (input + output)

Pricing:
  - Input: Competitive with GPT-4o
  - Output: Competitive with GPT-4o
  - V3.2-Exp: 50%+ price reduction
```

**Local Deployment:**
- Full control over model
- No API rate limits
- Data privacy
- Requires significant hardware investment

## Key Innovations

### 1. Auxiliary-Loss-Free Load Balancing

**Problem**: Traditional MoE models use auxiliary loss to encourage balanced expert utilization, but this can hurt performance.

**Innovation**: Dynamic bias adjustment achieves balance without auxiliary loss:
- Each expert has a bias term b_i added to routing score
- Bias decreased by γ if expert is overloaded
- Bias increased by γ if expert is underloaded
- System reaches dynamic equilibrium naturally

**Benefits**:
- No performance degradation from auxiliary loss
- Simpler training (fewer hyperparameters)
- All tokens processed (no token dropping)
- Better expert specialization

**Impact**: Pioneer work enabling auxiliary-loss-free MoE training at scale.

### 2. Multi-Token Prediction (MTP)

**Problem**: Standard next-token prediction may not fully utilize training data or encourage forward planning.

**Innovation**: Sequential multi-token prediction with separate modules:
- Predict t+1 and t+2 tokens in sequence
- 14B additional parameters for MTP modules
- Cross-entropy loss per depth, averaged and weighted
- Can be used for speculative decoding at inference

**Benefits**:
- Densifies training signals for better data efficiency
- Encourages pre-planning representations
- Improves long-range dependency learning
- Potential inference speedup via speculative decoding

**Impact**: First large-scale validation of sequential MTP at 671B scale.

### 3. FP8 Mixed Precision Training

**Problem**: Training 671B models in BF16 requires massive memory and compute.

**Innovation**: First validation of FP8 training at extreme scale:
- Selective quantization (activations before MoE up-projections)
- High-precision paths for critical operations
- Tile-wise and block-wise quantization strategies
- <0.25% relative loss error vs BF16

**Benefits**:
- ~50% memory savings
- Faster training (FP8 GEMM operations)
- Maintains training quality
- Enables cost-efficient training

**Impact**: Proves FP8 training is viable at 671B scale, paving way for future models.

### 4. DualPipe Bidirectional Pipeline Parallelism

**Problem**: Traditional pipeline parallelism has significant bubble time where GPUs idle waiting for communication.

**Innovation**: Bidirectional scheduling with computation-communication overlap:
- 16-way pipeline with bidirectional scheduling
- Overlap computation and communication phases
- Dedicated SMs (20 out of 132) for communication
- ~53% bubble reduction vs traditional pipeline

**Benefits**:
- Higher GPU utilization (less idle time)
- Faster training iterations
- Better scaling at large cluster sizes
- Cost savings from improved efficiency

**Impact**: Novel training algorithm enabling efficient cross-node expert parallelism.

### 5. Enhanced Multi-Head Latent Attention (MLA)

**Problem**: Standard attention has large KV cache that limits batch size and context length.

**Innovation**: Low-rank compression with decoupled RoPE:
- Compress KV to 512-dim latent vectors (93.3% reduction)
- Separate RoPE components preserve positional encoding
- Up-project only when needed for attention computation
- Maintains performance comparable to standard MHA

**Benefits**:
- 4.66× smaller cache vs Qwen-2.5 72B
- 7.28× smaller cache vs LLaMA-3.1 405B
- Enables longer context and larger batch sizes
- Faster inference with smaller memory footprint

**Impact**: Validates MLA at extreme scale (671B), proves viability for future models.

### 6. Cost-Efficient Training

**Achievement**: $5.57 million training cost for 671B parameters
- 14.8T tokens on 2,048 H800 GPUs
- 180K GPU hours per trillion tokens
- Most cost-efficient 671B model training to date

**Factors**:
- FP8 mixed precision training
- DualPipe pipeline parallelism
- Efficient MoE with load balancing
- No rollbacks or loss spikes
- Optimized training pipeline

**Impact**: Demonstrates that frontier-scale models can be trained affordably, democratizing access to large-scale AI.

### 7. No Token Dropping

**Problem**: Some MoE models drop tokens when expert capacity is exceeded, losing information.

**Innovation**: All tokens guaranteed to be processed:
- Auxiliary-loss-free load balancing ensures capacity
- No hard capacity limits on experts
- All information preserved during training and inference

**Benefits**:
- No information loss
- More stable training
- Better quality
- Simpler implementation

**Impact**: Proves token dropping is unnecessary with proper load balancing.

### 8. Training Stability

**Achievement**: Zero loss spikes, zero rollbacks during entire 671B training
- Trained on 14.8T tokens without irrecoverable issues
- Smooth training dynamics throughout
- No need to rollback to previous checkpoints

**Factors**:
- Careful architecture design (RMSNorm, decoupled RoPE)
- FP8 training framework with selective precision
- Auxiliary-loss-free load balancing
- Conservative learning rate schedule
- Gradient clipping and regularization

**Impact**: Demonstrates that extreme-scale training can be reliable and stable.

## Use Cases and Applications

### 1. General Language Understanding and Generation
- Conversational AI and chatbots
- Content creation and writing assistance
- Summarization and paraphrasing
- Translation (119+ languages)
- Question answering

### 2. Code Generation and Software Engineering
- Code completion and generation
- Bug detection and debugging
- Code review and refactoring
- Documentation generation
- Algorithm implementation
- Support for 80+ programming languages

### 3. Mathematical Reasoning
- Problem solving (GSM8K, MATH)
- Competition-level mathematics (AIME)
- Step-by-step reasoning
- Formula derivation
- Mathematical proof assistance

### 4. Long-Context Understanding
- Long document analysis (up to 128K tokens)
- Multi-document synthesis
- Code repository understanding
- Book and research paper analysis
- Extended conversation history

### 5. Research and Experimentation
- Foundation for domain-specific fine-tuning
- Benchmarking and evaluation
- Architecture research (MLA, MoE, FP8)
- Training efficiency studies
- Inference optimization research

### 6. Education and Tutoring
- Step-by-step explanations
- Concept teaching across domains
- Practice problem generation
- Homework assistance
- Multilingual education support

### 7. Enterprise and Business
- Customer service automation
- Data analysis and reporting
- Business intelligence
- Process automation
- Knowledge management
- Technical documentation

### 8. Specialized Applications
- Scientific research assistance
- Legal document analysis
- Medical information (not diagnostic)
- Financial analysis
- Creative writing and storytelling

## Licensing

### Original DeepSeek-V3 and DeepSeek-V3-Base

**Code License:**
- MIT License (permissive open-source)

**Model License:**
- Custom DeepSeek Model License
- Commercial use: Allowed
- Derivative development: Allowed (fine-tuning, quantization, distillation)
- Royalty-free and irrevocable copyright license
- Use-based restrictions apply
- **Does NOT meet open source license definition** (due to restrictions)

**Restrictions:**
- Cannot use for illegal purposes
- Cannot use to harm individuals or society
- Cannot violate third-party rights
- Subject to applicable laws and regulations

### DeepSeek-V3-0324 and Later (March 2025+)

**Full MIT License for Both Code and Model:**
- **No usage restrictions** (beyond standard MIT terms)
- **No fees or profit sharing**
- **Commercial use freely allowed**
- **Can be used for any lawful purpose**
- **True open source** (meets open source definition)

**MIT License Terms:**
- Permission to use, copy, modify, merge, publish, distribute, sublicense, and sell
- Attribution required (copyright notice and permission notice)
- No warranty
- No liability

**Major Improvement:**
This licensing change makes DeepSeek-V3-0324 and later versions truly open-source and removes barriers to adoption, especially for commercial applications.

## Limitations and Disclosure

### What Is Disclosed

**✓ Architecture Details (~90% disclosed):**
- Total parameters: 671B
- Activated parameters: 37B
- Layer count: 61 (3 dense + 58 MoE)
- Hidden dimension: 7,168
- Attention heads: 128
- MLA configuration (latent dims, compression ratios)
- MoE configuration (256 experts, Top-8 routing)
- Tokenizer: 128K vocabulary
- Context: 4K → 32K → 128K extension

**✓ Training High-Level Details (~70% disclosed):**
- Training data: 14.8 trillion tokens
- GPU infrastructure: 2,048 H800 GPUs
- GPU hours: 2.788M H800 GPU hours
- Training cost: $5.57M USD
- Learning rate schedule: Initial, constant, decay, final
- Batch size: 3,072 → 15,360 tokens
- Training stability: No loss spikes or rollbacks

**✓ Innovations (Fully Disclosed):**
- Auxiliary-loss-free load balancing (detailed)
- Multi-Token Prediction architecture (detailed)
- FP8 mixed precision training (detailed)
- DualPipe algorithm (detailed)
- Enhanced MLA (detailed)

**✓ Performance Benchmarks (Partially Disclosed):**
- MMLU, MATH, GSM8K, HumanEval (scores provided)
- Arena-Hard, Codeforces (scores provided)
- Many other benchmarks (scores provided)

### What Is NOT Disclosed

**✗ Training Hyperparameters (NOT disclosed):**
- Exact optimizer configuration
- Exact batch size schedule details
- Weight decay value
- Gradient clipping thresholds
- Warmup schedule details
- Dropout rates (if any)

**✗ Data Composition (NOT disclosed):**
- Exact percentage of web/code/STEM/multilingual data
- Data sources and datasets used
- Data filtering and quality criteria details
- Deduplication strategy specifics
- Data preprocessing pipeline details

**✗ Some Architecture Details (NOT disclosed):**
- Exact intermediate dimensions in some layers
- Specific initialization strategies
- Some normalization epsilon values
- Exact expert routing implementation details

**✗ Post-Training Details (Partially NOT disclosed):**
- RL reward model architectures
- Exact SFT data composition
- RL hyperparameters (PPO/GRPO specifics)
- Distillation from DeepSeek-R1 details

**✗ Some Benchmark Details (NOT disclosed):**
- Exact prompts used for evaluation
- Some few-shot learning configurations
- Full breakdown of all benchmark subscores

### Model Limitations

**1. Hallucination:**
- May generate plausible-sounding but incorrect information
- Should not be used for critical decisions without verification
- Particularly in specialized domains (medical, legal)

**2. Knowledge Cutoff:**
- Training data cutoff: July 2024
- No real-time information or recent events
- May have outdated information on rapidly evolving topics

**3. Reasoning Limitations:**
- Base model has limited step-by-step reasoning
- R1 integration improves reasoning but has limitations
- May struggle with very complex multi-step reasoning
- Can make arithmetic errors without tool use

**4. Bias and Fairness:**
- Trained on internet data containing biases
- May exhibit gender, racial, or cultural biases
- Should not be used for high-stakes decisions (hiring, lending, etc.)
- Bias mitigation applied but not perfect

**5. Context Length:**
- 128K context window (model capability)
- 64K context window (commercial API)
- Performance may degrade at extreme context lengths
- Memory requirements scale with context length

**6. Multimodal Limitations:**
- DeepSeek-V3 is text-only (no vision or audio)
- Specialized variants (Janus, multimodal) have different capabilities
- Cannot process images, videos, or audio natively

**7. Resource Requirements:**
- Very large model requiring multi-node deployment
- High memory requirements (~700 GB FP8, ~1.4 TB BF16)
- Significant computational resources for inference
- Not suitable for edge devices or single GPUs

**8. Language Imbalance:**
- Stronger performance in Chinese and English
- Some languages may have weaker performance
- Translation quality varies by language pair

### Disclosure Level Assessment

**Overall Disclosure: ~75-80%**

Compared to other models:
- **More disclosed than**: GPT-4, Claude 3.5 (closed-source)
- **Similar to**: LLaMA 3.1, Qwen2.5, Mistral (open-source)
- **Less disclosed than**: Fully reproducible academic models

DeepSeek-V3 provides substantial technical details enabling:
- Understanding of architecture and innovations
- Reproduction of key techniques (MLA, MoE, FP8, DualPipe)
- Fine-tuning and adaptation
- Research and experimentation

However, full reproduction from scratch would require:
- Figuring out undisclosed hyperparameters through experimentation
- Assembling training data from public sources
- Significant compute resources ($5.57M+)

## Resources and Links

### Official Resources

**Technical Report:**
- arXiv: https://arxiv.org/abs/2412.19437
- arXiv PDF: https://arxiv.org/pdf/2412.19437
- arXiv HTML: https://arxiv.org/html/2412.19437
- Version 1: December 27, 2024
- Version 2: February 18, 2025

**GitHub Repository:**
- DeepSeek-V3: https://github.com/deepseek-ai/DeepSeek-V3
- DualPipe: https://github.com/deepseek-ai/DualPipe
- License: MIT (code), Custom/MIT (model)

**Hugging Face:**
- DeepSeek-V3: https://huggingface.co/deepseek-ai/DeepSeek-V3
- DeepSeek-V3-Base: https://huggingface.co/deepseek-ai/DeepSeek-V3-Base
- DeepSeek-V3.1: https://huggingface.co/deepseek-ai/DeepSeek-V3.1
- DeepSeek-V3.1-Terminus: https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus
- DeepSeek-V3.2-Exp: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Paper Page: https://huggingface.co/papers/2412.19437

**Official Announcements:**
- DeepSeek V3 Release: https://api-docs.deepseek.com/news/news1226

**API Access:**
- Chat Interface: https://chat.deepseek.com
- API Platform: https://platform.deepseek.com
- API Docs: https://api-docs.deepseek.com

### Inference Frameworks

**SGLang:**
- GitHub: https://github.com/sgl-project/sglang
- Recommended for DeepSeek-V3 (MLA optimizations)

**vLLM:**
- GitHub: https://github.com/vllm-project/vllm
- Version: 0.6.6+ required

**LMDeploy:**
- GitHub: https://github.com/InternLM/lmdeploy

**TensorRT-LLM:**
- GitHub: https://github.com/NVIDIA/TensorRT-LLM

**LightLLM:**
- GitHub: https://github.com/ModelTC/lightllm

### Company Information

**DeepSeek AI:**
- Founded: July 17, 2023
- Location: Hangzhou, Zhejiang, China
- Founder/CEO: Liang Wenfeng
- Parent: High-Flyer (hedge fund)
- Team: ~200 employees
- Funding: Self-funded (no VC)
- Wikipedia: https://en.wikipedia.org/wiki/DeepSeek

### Related Models

**DeepSeek Model Family:**
- DeepSeek-V2: Predecessor with 236B parameters
- DeepSeek-R1: Reasoning model (January 2025)
- DeepSeek-Coder: Code specialist variants
- DeepSeek-Math: Math specialist
- Janus-Pro: Multimodal understanding and generation

### Research Papers and Analyses

**Technical Analyses:**
- "DeepSeek V3 Explained: Multi-Head Latent Attention" - Towards Data Science
- "The Inner Workings of DeepSeek-V3" - Chris McCormick
- "DeepSeek Model Architecture Deep Dive" - Fireworks AI

**Comparisons:**
- DeepSeek V3 vs GPT-4o vs LLaMA 3.3
- Comparing Advanced AI Models (DeepSeek, Qwen, LLaMA, Claude, GPT-4o)

### Community and Support

**Discussion:**
- Hugging Face Discussions: Model card discussion sections
- GitHub Issues: Repository issue trackers
- Reddit: r/LocalLLaMA, r/MachineLearning
- Twitter/X: @deepseek_ai

**Support:**
- API Documentation: https://api-docs.deepseek.com
- GitHub Issues: https://github.com/deepseek-ai/DeepSeek-V3/issues

---

**Document Information:**
- Created: 2025
- Model Version: DeepSeek-V3 and variants
- Technical Report: arXiv:2412.19437
- Model Release: December 25, 2024

**Sources:**
All information verified from official technical report, GitHub repository, Hugging Face model cards, and official announcements. Disclosure level assessment based on technical report completeness.
