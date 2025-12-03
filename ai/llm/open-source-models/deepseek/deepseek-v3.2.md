# DeepSeek-V3.2

**Release Date:** December 1, 2025
**Developer:** DeepSeek AI (Hangzhou, China)
**Model Size:** 671 billion total parameters (37 billion active)
**Architecture:** Mixture-of-Experts (MoE) with DeepSeek Sparse Attention (DSA)
**Context Window:** 128,000 tokens
**License:** MIT License
**Model Variants:** V3.2 (general), V3.2-Speciale (reasoning)

## Overview

DeepSeek-V3.2 represents a breakthrough in efficient AI reasoning, released on December 1, 2025, as DeepSeek AI's flagship model incorporating revolutionary **DeepSeek Sparse Attention (DSA)** technology. Building upon the foundation of DeepSeek-V3.1-Terminus (671B parameters, 37B active), V3.2 introduces fine-grained sparse attention that achieves **2-3× faster inference** on long contexts and **30-40% lower memory usage** while maintaining virtually identical output quality to dense attention models.

The release includes two specialized variants optimized for different use cases. **DeepSeek-V3.2** (standard) provides balanced reasoning performance comparable to OpenAI's GPT-5, with integrated tool-use capabilities that combine thinking and action execution—a first for DeepSeek models. **DeepSeek-V3.2-Speciale** (reasoning-focused) delivers maximal reasoning capabilities, achieving performance on par with Google's Gemini 3.0 Pro and **gold-medal performance** across four elite international competitions: the **2025 International Mathematical Olympiad (IMO)**, **International Olympiad in Informatics (IOI)**, **ICPC World Finals**, and **China Mathematical Olympiad (CMO)**.

On mathematics benchmarks, V3.2-Speciale achieves **96.0% on AIME 2025** (surpassing GPT-5's 94.6% and Gemini 3 Pro's 95.0%), and an extraordinary **99.2% on HMMT February 2025**—the highest score among all reasoning models. In competitive programming, it secured **2nd place at ICPC World Finals 2025** and **10th place (gold medal) at IOI 2025**. On coding benchmarks like Terminal Bench 2.0, V3.2 scores **46.4%** compared to GPT-5's 35.2% and Gemini 3 Pro's 54.2%.

The architectural innovation of **DeepSeek Sparse Attention (DSA)** transforms long-context efficiency by implementing a two-stage pipeline: a lightweight "**lightning indexer**" that rapidly identifies the most relevant tokens from the entire context window, followed by fine-grained token selection that narrows down to the critical subset. This reduces attention complexity from O(L²) to O(Lk), where k=2048 is the fixed number of selected tokens, delivering dramatic improvements in throughput and memory efficiency without quality degradation.

V3.2 introduces **massive agentic task synthesis** with training data covering **1,800+ environments** and **85,000+ complex instructions**, enabling robust tool-use capabilities. The model is the first in the DeepSeek family to integrate **thinking directly into tool execution**, supporting both reasoning-enhanced tool-use and direct tool invocation modes. This makes V3.2 particularly effective for autonomous agents, multi-step workflows, and complex interactive environments.

Released under the permissive **MIT License**, DeepSeek-V3.2 continues DeepSeek's commitment to open-source AI, providing full model weights, inference code, and high-performance kernels (DeepGEMM, FlashMLA) for community deployment. The models are available on Hugging Face, with V3.2 accessible via web, app, and API, while V3.2-Speciale is temporarily available via API until December 15, 2025.

**Official Resources:**
- [DeepSeek-V3.2 Release Announcement](https://api-docs.deepseek.com/news/news251201) (DeepSeek API Docs, December 1, 2025)
- [DeepSeek-V3.2 Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) (Hugging Face)
- [DeepSeek-V3.2-Speciale Model Card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale) (Hugging Face)
- [DeepSeek-V3.2-Exp GitHub Repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) (Technical implementation)
- [DeepSeek-V3.2-Exp Initial Announcement](https://api-docs.deepseek.com/news/news250929) (September 29, 2025)

---

## Model Architecture

DeepSeek-V3.2 inherits the core architecture of DeepSeek-V3 with revolutionary sparse attention enhancements, combining massive scale (671B parameters) with computational efficiency (37B active parameters per token).

### Model Lineup

DeepSeek-V3.2 is available in multiple variants optimized for different use cases:

| Variant | Parameters | Precision | Primary Use Case | Tool-Use Support | Availability |
|---------|------------|-----------|------------------|------------------|--------------|
| **DeepSeek-V3.2** | 671B total, 37B active | BF16/FP8 | General reasoning, agents, production | Yes (thinking + non-thinking) | App, Web, API |
| **DeepSeek-V3.2-Speciale** | 671B total, 37B active | BF16/FP8 | Deep reasoning, mathematics, competitive programming | No | API-only (temp) |
| **DeepSeek-V3.2-Exp** | 671B total, 37B active | BF16/FP8 | Experimental, research | Yes | GitHub, HuggingFace |
| **DeepSeek-V3.2-Exp-Base** | 671B total, 37B active | BF16 | Fine-tuning foundation | N/A | HuggingFace |

**Note:** V3.2-Speciale temporary API endpoint expires December 15, 2025, after which it may be integrated into main V3.2 or released as a permanent variant.

### Core Architecture Specifications

```yaml
Model Overview:
  Total Parameters: 671 billion (671B)
  Non-Embedding Parameters: 670B
  Active Parameters per Token: 37 billion (37B)
  Activation Rate: 5.5% (37B / 671B)
  Architecture Type: Sparse Mixture-of-Experts (MoE) with DSA

Base Architecture (Inherited from V3.1-Terminus):
  Layers: 61 total transformer layers
    - Dense Layers: 3 (standard transformer blocks)
    - MoE Layers: 58 (Mixture-of-Experts blocks)

  Hidden Dimension: 7,168
  Intermediate Size (FFN): 18,432
  Vocabulary Size: 128,000 tokens

Context Window:
  Training: 4,096 tokens (native)
  Extended: 32,768 tokens (Stage 1)
  Maximum: 128,000 tokens (Stage 2 with YaRN)
  API: 64,000 tokens (commercial offering)
  DSA Optimized: Full 128K with 2-3× speedup

MoE Configuration:
  Shared Experts: 1 per MoE layer
  Routed Experts: 256 per MoE layer
  Activated Experts: 8 per token (top-8 routing)
  Expert Hidden Dimension: 2,048

  Routing Strategy:
    - Node-Limited Routing: Max 4 nodes per token
    - Load Balancing: Auxiliary-loss-free with dynamic bias
    - Expert Selection: Sigmoid affinity scores

Attention Mechanism (MLA + DSA):
  Type: Multi-Head Latent Attention (MLA) with DeepSeek Sparse Attention
  Attention Heads: 128
  Head Dimension: 128 per head
  Total Attention Dimension: 16,384 (128 × 128)

  MLA Compression:
    KV Latent Dimension (d_c): 512
    Query Hidden Space (d_h): 1,536
    Decoupled RoPE Dimension per Head (d_R_h): 64

  DSA Sparse Attention:
    Top-K Selection: 2,048 tokens per query
    Lightning Indexer: Fast preliminary scoring (FP8, few heads)
    Fine-Grained Selection: Attention over top-k subset only
    Complexity Reduction: O(L²) → O(Lk), where k=2048

Position Embeddings:
  Type: Rotary Position Embeddings (RoPE)
  Base Frequency (Theta): 10,000
  YaRN Scaling: Applied for context extension beyond 4K

  YaRN Configuration:
    Original Max Position: 4,096
    Beta Fast: 32
    Beta Slow: 1
    MScale: 1.0
    MScale All Dim: 0.0

Activation Function:
  Type: SwiGLU (Swish-Gated Linear Unit)
  Formula: SwiGLU(x) = Swish(x_1) ⊗ x_2, where Swish(x) = x · sigmoid(x)
  Application: Feed-forward networks in each expert

Normalization:
  Type: RMSNorm (Root Mean Square Layer Normalization)
  Formula: RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ
  Application: Pre-normalization before attention and FFN blocks

Precision Support:
  Training: BF16 (bfloat16) for main weights
  Computation: FP8 for matrix multiplications (V3 innovation)
  Inference Options:
    - BF16: Full precision (670B × 2 bytes = ~1.3TB)
    - FP8: Mixed precision (670B × 1 byte = ~670GB)
    - KV Cache: FP8 quantized (656 bytes per token)

Vision Capabilities:
  Status: Not included in V3.2 (language-only)
  Note: See DeepSeek-VL2 for multimodal variants
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  DeepSeek-V3.2 Architecture                      │
│              671B Total | 37B Active Parameters                  │
│                    with DSA Sparse Attention                     │
└─────────────────────────────────────────────────────────────────┘

INPUT PROCESSING
─────────────────
Text Tokens (up to 128K)
     ↓
[Embedding Layer]
128K vocab → 7,168 dim
     ↓

TRANSFORMER LAYERS (61 layers total)
────────────────────────────────────

DENSE LAYERS (×3 layers)
  ┌──────────────────────────────────────┐
  │         Standard Dense Layer          │
  └──────────────────────────────────────┘
            ↓
      [RMSNorm]
            ↓
  ┌──────────────────────────────────────┐
  │    Multi-Head Latent Attention       │
  │         + DSA Sparse Attention        │
  │                                       │
  │    [MLA KV Compression: d_c=512]     │
  │              ↓                        │
  │    ┌──────────────────────────┐      │
  │    │   Lightning Indexer       │      │
  │    │   (Fast FP8 scoring)      │      │
  │    │   Scores all L tokens     │      │
  │    └──────────────────────────┘      │
  │              ↓                        │
  │    [Select Top-K=2048 tokens]        │
  │              ↓                        │
  │    ┌──────────────────────────┐      │
  │    │  Fine-Grained Attention   │      │
  │    │  (Only on selected k)     │      │
  │    │  Complexity: O(Lk) not O(L²)│   │
  │    └──────────────────────────┘      │
  └──────────────────────────────────────┘
            ↓
   [Residual Connection]
            ↓
      [RMSNorm]
            ↓
  ┌──────────────────────────────────────┐
  │     Dense Feed-Forward Network        │
  │     (Standard FFN, no experts)        │
  └──────────────────────────────────────┘
            ↓
   [Residual Connection]

MOE LAYERS (×58 layers)
  ┌──────────────────────────────────────┐
  │       Mixture-of-Experts Layer        │
  └──────────────────────────────────────┘
            ↓
      [RMSNorm]
            ↓
  ┌──────────────────────────────────────┐
  │    Multi-Head Latent Attention       │
  │         + DSA Sparse Attention        │
  │     (Same as dense layers above)     │
  └──────────────────────────────────────┘
            ↓
   [Residual Connection]
            ↓
      [RMSNorm]
            ↓
  ┌──────────────────────────────────────┐
  │   Mixture-of-Experts Feed-Forward    │
  │                                       │
  │         [Router Network]              │
  │         (Sigmoid affinity)            │
  │               ↓                       │
  │    Top-8 Expert Selection             │
  │    (From 256 experts)                 │
  │               ↓                       │
  │  ┌────────────────────────────────┐  │
  │  │  Expert 1  │ ... │  Expert 256│  │
  │  │  (2048 dim │     │  (2048 dim │  │
  │  │   SwiGLU)  │     │   SwiGLU)  │  │
  │  └────────────────────────────────┘  │
  │               ↓                       │
  │    [Weighted Combination]             │
  │    (Activate 8/256 = 3.1% experts)    │
  │    (Total: ~37B active params)        │
  └──────────────────────────────────────┘
            ↓
   [Residual Connection]
            ↓
       (Next Layer)

OUTPUT GENERATION
─────────────────
      [RMSNorm]
            ↓
      [LM Head]
            ↓
[Output Logits: 128,000 vocab]
            ↓
  Next Token Prediction
```

### Design Philosophy

```yaml
DeepSeek-V3.2 Design Principles:

Efficiency Through Sparsity:
  MoE Sparsity (Activated Experts):
    - 671B total capacity, 37B active compute
    - Each token activates 8 of 256 experts
    - 16.5× more capacity than compute cost
    - Enables frontier performance at manageable cost

  DSA Sparsity (Attention Tokens):
    - Attention over 2048 of potentially 128K tokens
    - Lightning indexer identifies relevant context
    - 64× reduction in attention computation (128K → 2K)
    - 2-3× speedup + 30-40% memory savings

  Combined Efficiency:
    - Both dimensions leverage conditional computation
    - Only compute what matters per token
    - Massive scale without proportional cost

Attention Innovation (DSA):
  Problem Solved:
    - Standard attention: O(L²) complexity
    - Prohibitive for long contexts (128K tokens)
    - Previous solutions (sliding window, etc.) lose global context

  DSA Solution:
    - Two-stage sparse attention pipeline
    - Fast indexing preserves global context awareness
    - Fine-grained selection maintains quality
    - Negligible quality loss vs dense attention

  Technical Breakthrough:
    - First fine-grained sparse attention at scale
    - Trained with KL-divergence alignment
    - Production-ready (not just research demo)
    - Open-source kernels (DeepGEMM, FlashMLA)

Multi-Head Latent Attention (MLA):
  KV Cache Compression:
    - 512-dim latent vs 16,384-dim full KV
    - 93.3% cache reduction
    - DeepSeek-V3.2: 656 bytes per token (FP8)
    - Standard MHA: ~8-10KB per token
    - Enables long context at reasonable memory

  Benefits for V3.2:
    - DSA operates on compressed KV cache
    - Lightning indexer uses cached keys
    - Combined: sparse + compressed = extreme efficiency
    - 128K context feasible on single-node inference

Auxiliary-Loss-Free Load Balancing:
  Innovation:
    - No auxiliary loss for expert balancing
    - Dynamic bias adjustment per expert
    - Load monitoring → bias modification
    - Natural equilibrium emerges

  Advantages:
    - No performance degradation from aux loss
    - Simpler hyperparameter tuning
    - All tokens processed (no dropping)
    - Stable training at massive scale

Training Efficiency:
  Pre-Training Approach:
    - 2-stage: Dense warm-up + Sparse training
    - 2.1B tokens dense (indexer learns)
    - 943.7B tokens sparse (main training)
    - Total: ~945.8B tokens for DSA training

  Cost Efficiency:
    - V3 trained for $5.57M (2.788M H800 hours)
    - V3.2 incremental training cost minimal
    - 50% faster training vs V3.1
    - Maintains training stability (no spikes)

FP8 Mixed Precision:
  Computation Efficiency:
    - All matrix multiplications in FP8
    - 2× speedup vs BF16 on H800/H100
    - Maintained throughout training
    - No quality degradation

  Storage:
    - Weights stored in BF16/FP32
    - Cast to FP8 for computation
    - KV cache quantized to FP8
    - Optimal quality/performance balance

Agentic Training Innovation:
  Massive Synthetic Dataset:
    - 1,800+ environments simulated
    - 85,000+ complex instructions
    - Reasoning-enhanced tool-use examples
    - Scalable synthesis pipeline

  Thinking with Tools:
    - First DeepSeek model integrating reasoning + tools
    - Two modes: thinking + non-thinking
    - Improved compliance in complex workflows
    - Better generalization to new environments
```

---

## DeepSeek Sparse Attention (DSA) Deep Dive

DeepSeek Sparse Attention represents the flagship innovation of V3.2, achieving fine-grained sparse attention for the first time in a production model.

### DSA Fundamentals

**The Long-Context Attention Problem:**

```yaml
Standard Attention Complexity:
  For sequence length L and hidden dimension d:
    Compute Cost: O(L² × d)
    Memory Cost: O(L²)

  Example (L=128K, d=7168):
    Attention Matrix: 128K × 128K = 16.4B elements
    Compute: 16.4B × 7,168 = ~117 trillion operations
    Memory: 16.4B × 4 bytes (FP32) = 64 GB per layer

  Result: Prohibitively expensive for long contexts

Previous Sparse Attention Approaches:
  1. Sliding Window:
     - Attend only to nearby tokens
     - Loses global context
     - Quality degradation

  2. Strided / Block Sparse:
     - Fixed patterns (every k-th token, blocks)
     - Inflexible, may miss important tokens
     - Moderate quality loss

  3. Learned Sparse (NSA, etc.):
     - Coarse-grained selection
     - Limited to small k values
     - Not production-ready

DSA Innovation:
  Fine-Grained Sparse Attention:
    - Selects specific relevant tokens per query
    - Dynamic selection (content-dependent)
    - Large k value (2,048) maintains quality
    - Production-ready at scale
```

### Two-Stage DSA Architecture

**Stage 1: Lightning Indexer**

```yaml
Purpose: Fast preliminary scoring of all tokens

Design:
  - Ultra-lightweight component
  - Operates in FP8 precision
  - Few attention heads (optimized for throughput)
  - Cached key values specifically for indexing

Process:
  Input: Query token q_i, All preceding keys K_cached

  Step 1 - Compute Index Logits:
    index_logits = LightningIndexer(q_i, K_cached)
    # Fast scoring: FP8, minimal heads
    # Shape: [L] - one score per preceding token

  Step 2 - Top-K Selection:
    top_k_indices = TopK(index_logits, k=2048)
    # Select 2,048 most relevant tokens

  Output: Indices of top-2048 tokens

Computational Cost:
  Complexity: O(L²) technically, but:
    - FP8 computation (2× faster)
    - Few heads (vs 128 full heads)
    - Highly optimized kernels (DeepGEMM)

  Net Effect: ~5-10% of full attention cost
  Result: Overhead is negligible

Implementation:
  - Dedicated KV cache for indexer (separate from main)
  - Cached key scale factors (FP8 quantization)
  - Paged memory management
  - Custom CUDA kernels (DeepGEMM library)

Training:
  Objective: Mimic dense attention distribution
  Loss: KL-divergence alignment

  KL_loss = KL(Dense_Attention_Dist || Lightning_Indexer_Scores)

  Process:
    1. Dense Warm-Up (2.1B tokens):
       - Main model frozen
       - Indexer learns to predict dense attention

    2. Sparse Training (943.7B tokens):
       - Main model trains normally
       - Indexer gradient separate from language loss
       - Indexer adapts to evolving attention patterns
```

**Stage 2: Fine-Grained Attention**

```yaml
Purpose: High-quality attention over selected tokens

Design:
  - Standard multi-head latent attention (MLA)
  - Operates on compressed KV cache (d_c=512)
  - Full precision (BF16 computation)
  - All 128 attention heads

Process:
  Input: Query q_i, Top-K indices from Stage 1

  Step 1 - Gather Selected KV:
    K_selected = K_cache[top_k_indices]  # Shape: [2048, 512]
    V_selected = V_cache[top_k_indices]  # Shape: [2048, 512]

  Step 2 - Standard Attention (Over Selected):
    # Perform full MLA attention, but only on 2048 tokens
    Q = UpProject(q_compressed)           # → [128, 128]
    K = UpProject(K_selected)             # → [2048, 128, 128]
    V = UpProject(V_selected)             # → [2048, 128, 128]

    scores = Q @ K^T / sqrt(d_head)       # [128, 2048]
    attn_weights = Softmax(scores)        # [128, 2048]
    output = attn_weights @ V             # [128, 128]

  Step 3 - Aggregate:
    final_output = CombineHeads(output)   # → [7168]

Computational Cost:
  Complexity: O(L × k), where k=2048

  For L=128K:
    Dense: O(128K × 128K) = O(16.4B)
    DSA: O(128K × 2K) = O(256M)
    Reduction: 64× fewer operations

  Wall-Clock Speedup: 2-3× (accounting for indexer overhead)

Quality Preservation:
  - k=2048 is large enough to capture relevant context
  - Dynamic selection per query (content-aware)
  - KL-divergence training ensures distribution matching
  - Benchmark results: identical to V3.1-Terminus
```

### DSA Performance Analysis

**Efficiency Gains:**

```yaml
Inference Speed:
  Benchmark: Long-context tasks (32K-128K tokens)

  Results:
    - 2-3× faster inference vs V3.1-Terminus
    - Scales better as context length increases
    - Maintains quality (MMLU, CodeForces, AIME identical)

  Example (128K context):
    V3.1: 120 tokens/second
    V3.2: 300 tokens/second
    Speedup: 2.5×

Memory Usage:
  Reduction: 30-40% lower memory consumption

  Sources of Savings:
    1. Smaller attention workspace:
       - Dense: L × L attention matrix
       - DSA: L × k attention matrix
       - Savings: (L - k) / L = (128K - 2K) / 128K = 98.4%

    2. Efficient indexer cache:
       - Separate, smaller cache for indexing
       - FP8 precision (vs BF16 main cache)

    3. Combined with MLA:
       - KV cache already compressed to 512 dim
       - DSA operates on compressed cache
       - Double efficiency multiplicative benefit

Training Efficiency:
  Pre-Training DSA:
    - 50% faster than dense baseline
    - Stage 1 (Dense Warm-Up): 2.1B tokens
    - Stage 2 (Sparse Training): 943.7B tokens
    - Total: 945.8B tokens

  Compared to Full Dense Training:
    - Dense would require ~1,900B tokens equivalent
    - DSA achieves same with 945.8B
    - 2× training efficiency

Cost Efficiency:
  API Pricing Impact:
    - 50% lower inference cost per token
    - Faster response (2-3× speedup)
    - Same quality as dense models

  Self-Hosting Benefits:
    - Fewer GPUs required (lower memory)
    - Higher throughput per GPU
    - Better ROI for long-context applications
```

**Benchmark Quality Validation:**

```yaml
Comparison: V3.1-Terminus vs V3.2-Exp (DSA)

General Knowledge:
  MMLU-Pro:
    - V3.1-Terminus: 85.0%
    - V3.2-Exp: 85.0%
    - Difference: 0.0% (identical)

Competitive Programming:
  Codeforces Rating:
    - V3.1-Terminus: 2046
    - V3.2-Exp: 2121
    - Improvement: +75 points (+3.7%)

Mathematics:
  AIME 2025:
    - V3.1-Terminus: 88.4%
    - V3.2-Exp: 89.3%
    - Improvement: +0.9% (within noise margin)

Factual Accuracy:
  SimpleQA:
    - V3.1-Terminus: 96.8%
    - V3.2-Exp: 97.1%
    - Improvement: +0.3%

Conclusion:
  DSA maintains performance parity with dense attention
  while delivering 2-3× speedup and 30-40% memory reduction.
  Some benchmarks show slight improvements, likely due to
  better generalization from sparse attention training.
```

### DSA Implementation Details

**High-Performance Kernels:**

```yaml
DeepGEMM (Indexer Logit Kernels):
  Purpose: Ultra-fast preliminary scoring
  Features:
    - FP8 matrix multiplication
    - Paged memory support
    - Fused operations (score + topk)
    - Optimized for H800/H100 GPUs

  Performance:
    - ~10× faster than naive PyTorch
    - Minimal overhead vs full attention
    - Production-ready

  Repository: Closed source (proprietary)
  Note: Optimized for DeepSeek infrastructure

FlashMLA (Sparse Attention Kernels):
  Purpose: Efficient sparse attention computation
  Features:
    - Fused attention over selected tokens
    - KV cache management
    - Low-rank MLA projections
    - Memory-efficient implementation

  Performance:
    - 2-3× faster than standard attention
    - Handles variable-length k
    - Production-ready

  Repository: https://github.com/deepseek-ai/FlashMLA
  License: MIT (open source)

TileLang (Research Kernels):
  Purpose: Readable research implementation
  Features:
    - High-level kernel language
    - Research-friendly design
    - Reference implementations

  Use Case: Understanding DSA internals
  Repository: Mentioned in V3.2-Exp repo

Integration:
  vLLM:
    - Native DSA support (v0.6.0+)
    - Automatic kernel selection
    - Transparent to users

  SGLang:
    - Day-0 DSA support
    - Docker deployment
    - Production-ready

  Hugging Face Transformers:
    - Custom modeling code required
    - Reference in DeepSeek repo
    - Community implementations
```

**Memory Layout:**

```yaml
KV Cache Structure (Per Token):
  Total Size: 656 bytes

  Components:
    1. Quantized NoPE (No Position Encoding):
       - 512 × float8_e4m3 values
       - Size: 512 bytes
       - Content: Compressed KV latent vectors

    2. Scale Factors:
       - 4 × float32 values
       - Size: 16 bytes (4 × 4 bytes)
       - Purpose: FP8 dequantization

    3. Metadata:
       - Indexer-specific cached keys
       - Paged memory pointers
       - Total overhead: ~128 bytes

Lightning Indexer Cache (Separate):
  - Dedicated key cache for indexing
  - FP8 precision
  - Size: ~256 bytes per token
  - Optimized for fast scoring

Total Memory (128K context):
  Main KV Cache: 128K × 656 bytes = 84 MB per layer
  Indexer Cache: 128K × 256 bytes = 33 MB per layer
  Total per layer: 117 MB
  Total for 61 layers: 7.1 GB

  Compare Dense (128K context):
    Standard MHA: 128K × 8KB per token = 1024 MB per layer
    Total for 61 layers: 62.5 GB

  DSA Savings: 8.8× smaller KV cache
```

**Training Methodology:**

```yaml
Two-Stage Training Process:

Stage 1: Dense Warm-Up
  Duration: 2.1 billion tokens
  Purpose: Train lightning indexer

  Process:
    1. Main model frozen (weights from V3.1-Terminus)
    2. Indexer learns to predict dense attention distribution
    3. Objective: Minimize KL-divergence
       KL(Dense_Attn || Indexer_Scores)
    4. No language modeling loss for indexer

  Why Needed:
    - Indexer must learn relevant token patterns
    - Cold start: random indexer would select irrelevant tokens
    - Main model provides supervision signal

Stage 2: Sparse Training
  Duration: 943.7 billion tokens
  Purpose: Train main model with sparse attention

  Process:
    1. Both main model and indexer active
    2. Main model trains with standard language modeling loss
    3. Indexer receives separate gradient:
       - Continues KL-divergence alignment
       - Adapts to evolving attention patterns
    4. Gradients separated: indexer doesn't affect LM loss

  Benefits:
    - Main model learns with sparse attention from start
    - Indexer co-evolves with model
    - No forgetting of dense attention patterns
    - Maintains quality while gaining efficiency

Training Configuration:
  Total Tokens: 945.8B (2.1B + 943.7B)
  Compared to V3.1-Terminus: Similar scale

  Top-K Selection: k=2048 throughout training
  Precision: FP8 for computation, BF16 for storage

  Infrastructure:
    - 2,048 × H800 GPUs (same as V3)
    - 256 server nodes (8 GPUs each)
    - InfiniBand interconnect

  Stability:
    - No loss spikes
    - No rollbacks required
    - Maintained throughout sparse training

Hyperparameters:
  KL Divergence Weight: Not disclosed
  Indexer Learning Rate: Separate from main model
  Indexer Heads: Few (exact number not disclosed)
  Balance Loss Coefficient: α = 0.0001 (inherited from V3)
```

---

## Model Variants

DeepSeek-V3.2 is available in three primary variants, each optimized for different use cases.

### DeepSeek-V3.2 (Standard)

```yaml
Overview:
  Release Date: December 1, 2025
  Target Use Case: General reasoning, agents, production deployment
  Performance Level: GPT-5 comparable

Key Features:
  1. Balanced Reasoning:
     - Inference speed vs reasoning quality tradeoff
     - Shorter generation trajectories than Speciale
     - Suitable for interactive applications

  2. Tool-Use Integration:
     - First DeepSeek model with thinking + tools
     - Two modes:
       a) Thinking Mode + Tools: Reasons before tool use
       b) Direct Tool-Use: Immediate tool invocation
     - OpenAI-compatible function calling format

  3. Agentic Training:
     - Trained on 1,800+ environments
     - 85,000+ complex instructions
     - Improved compliance in interactive workflows
     - Better generalization to new tools

Availability:
  - Web: chat.deepseek.com
  - App: iOS and Android
  - API: api.deepseek.com
  - Pricing: Standard rates (same as V3.1)

Technical Specifications:
  Parameters: 671B total, 37B active
  Precision: BF16 weights, FP8 computation
  Context: 128,000 tokens (64K via API)
  DSA: Enabled (k=2048)
  Tool Support: Yes

Benchmark Performance:
  AIME 2025: ~90% (estimated, not officially disclosed for base V3.2)
  MMLU-Pro: 85.0%
  Codeforces: ~2100
  LiveCodeBench: 83.3%
  SWE-Multilingual: 70.2%
  Terminal Bench 2.0: 46.4%

Ideal For:
  - Production chatbots and assistants
  - Agentic workflows and automation
  - Code generation and debugging
  - General-purpose reasoning
  - Interactive applications
  - Function calling / tool-use scenarios

Not Ideal For:
  - Competition-level mathematics (use Speciale)
  - Maximum reasoning quality (use Speciale)
  - Latency-critical applications (consider smaller models)
```

### DeepSeek-V3.2-Speciale (Reasoning)

```yaml
Overview:
  Release Date: December 1, 2025 (temporary API, expires Dec 15)
  Target Use Case: Deep reasoning, mathematics, competitive programming
  Performance Level: Gemini 3.0 Pro comparable

Key Features:
  1. Maximum Reasoning:
     - Longer generation trajectories
     - More thorough thinking process
     - Gold-medal competitive performance

  2. Reinforcement Learning Training:
     - Scaled post-training compute
     - Robust RL protocols
     - Optimized for correctness over speed

  3. Reasoning-Exclusive:
     - No tool-calling support
     - Pure reasoning focus
     - Designed for complex problem-solving

Availability:
  - API: api.deepseek.com (temporary endpoint)
  - Expiration: December 15, 2025
  - Future: May integrate into main V3.2 or separate release
  - No web/app access (API-only)

Technical Specifications:
  Parameters: 671B total, 37B active
  Precision: BF16 weights, FP8 computation
  Context: 128,000 tokens
  DSA: Enabled (k=2048)
  Tool Support: No (reasoning-only)

Benchmark Performance (Gold-Medal Results):
  IMO 2025: Gold medal (exact percentage not disclosed)
  IOI 2025: Gold medal, 10th place
  ICPC World Finals 2025: 2nd place
  CMO 2025: Gold medal

  AIME 2025: 96.0% (vs GPT-5: 94.6%, Gemini 3 Pro: 95.0%)
  HMMT February 2025: 99.2% (highest among all reasoning models)
  GPQA Diamond: Not disclosed (expected high)
  MATH: Not disclosed

Coding Performance:
  LiveCodeBench: Expected higher than V3.2 standard
  SWE Verified: Expected competitive with Gemini 3 Pro
  Terminal Bench 2.0: Expected higher than V3.2 standard

Competition Evaluation Methodology:
  Max Generation Length: 128,000 tokens
  No Internet Access: Offline evaluation
  No External Tools: Pure reasoning

  IOI Specific:
    - 500 candidate solutions per problem
    - Filter: Keep valid solutions only
    - Selection: Top 50 by longest thinking traces
    - Submission: Best from top 50

Ideal For:
  - Competition mathematics (IMO, AIME, CMO)
  - Competitive programming (IOI, ICPC)
  - Research problems requiring deep reasoning
  - Complex logical puzzles
  - Graduate-level science (GPQA)
  - Theorem proving

Not Ideal For:
  - Agentic workflows (no tool support)
  - Real-time applications (slower inference)
  - General chat (overkill)
  - Production chatbots (use standard V3.2)

Future:
  After December 15, 2025:
    Option 1: Merged into standard V3.2 as "reasoning mode"
    Option 2: Separate permanent API endpoint
    Option 3: Open-source release on Hugging Face
```

### DeepSeek-V3.2-Exp (Experimental)

```yaml
Overview:
  Release Date: September 29, 2025
  Target Use Case: Research, experimentation, DSA evaluation
  Status: Experimental (basis for V3.2 and Speciale)

Key Features:
  1. First DSA Implementation:
     - Initial release of sparse attention
     - Validated efficiency claims
     - Community testing ground

  2. Open Source:
     - Full model weights on Hugging Face
     - Inference code on GitHub
     - High-performance kernels (FlashMLA)

  3. Research Transparency:
     - Detailed architecture documentation
     - Training methodology disclosed
     - Ablation studies included

Availability:
  - Hugging Face: deepseek-ai/DeepSeek-V3.2-Exp
  - GitHub: deepseek-ai/DeepSeek-V3.2-Exp
  - License: MIT
  - Self-hosting: Full support

Technical Specifications:
  Parameters: 671B total, 37B active
  Precision: BF16 weights, FP8 computation
  Context: 128,000 tokens
  DSA: Enabled (k=2048)
  Tool Support: Basic (not as advanced as V3.2)

Benchmark Performance:
  MMLU-Pro: 85.0%
  Codeforces: 2121 (vs V3.1: 2046)
  AIME 2025: 89.3% (vs V3.1: 88.4%)
  SimpleQA: 97.1% (vs V3.1: 96.8%)

Differences from V3.2:
  - Experimental status (not production-ready)
  - Basic tool-use (not integrated thinking + tools)
  - Community-driven deployment
  - No official API support

Ideal For:
  - Research on sparse attention
  - Self-hosted deployments
  - Custom modifications and fine-tuning
  - Understanding DSA internals
  - Reproducing results

Base Variant:
  DeepSeek-V3.2-Exp-Base:
    - Pre-trained foundation (no instruction tuning)
    - For fine-tuning and research
    - Available on Hugging Face
    - Same architecture as Exp
```

---

## Reasoning Capabilities

DeepSeek-V3.2-Speciale achieves state-of-the-art reasoning performance through reinforcement learning and scaled post-training compute.

### Reinforcement Learning Training

```yaml
RL Training Methodology:

Overview:
  Approach: Scaled post-training with robust RL protocols
  Base Model: DeepSeek-V3.2-Exp (671B, 37B active)
  Goal: Optimize for reasoning correctness

RL Framework (Inferred from Speciale Behavior):

  Reward Model:
    Type: Outcome-based rewards
    Sources:
      - Math: Symbolic verification (answer correct/incorrect)
      - Code: Execution tests (passes/fails)
      - Competitive Programming: Judge verdict (AC/WA/TLE)

    Advantage: Objective, verifiable rewards
    No human feedback: Pure correctness optimization

  Policy Optimization:
    Algorithm: Likely PPO or similar (not disclosed)
    Objective: Maximize correct solutions
    Constraint: Maintain general capabilities

  Training Data:
    Competition Problems:
      - IMO-style mathematics
      - IOI algorithmic challenges
      - ICPC programming contests
      - CMO advanced mathematics

    Synthetic Data:
      - Generated reasoning traces
      - Solution verification
      - Multi-step problem breakdowns

    Scale: Not disclosed, but substantial (months of training)

Multi-Step Reasoning Training:

  Approach:
    1. Generate long-form solutions
    2. Verify correctness at end
    3. Assign reward based on outcome
    4. Update policy to favor successful reasoning patterns

  Thinking Trace Selection:
    - For IOI: 500 candidates per problem
    - Selection: Longest valid thinking traces
    - Rationale: Longer reasoning correlates with thoroughness

  Benefit: Model learns to show work, not just output answers

Comparison with V3.2 Standard:

  V3.2 Standard:
    - Supervised fine-tuning for instruction following
    - Balanced reasoning vs inference speed
    - Agentic training (tools + environments)

  V3.2-Speciale:
    - Heavy RL for reasoning optimization
    - Maximal reasoning quality prioritized
    - No tool-use (reasoning-exclusive focus)

  Trade-off:
    - Speciale: Better at complex problems, slower
    - Standard: Faster, good enough for most tasks
```

### Gold Medal Achievements

**International Mathematical Olympiad (IMO) 2025:**

```yaml
Competition Overview:
  Level: High school mathematics olympiad (world's most prestigious)
  Format: 6 problems, 2 days (4.5 hours per day)
  Difficulty: Extremely high (top students worldwide)

  Problem Types:
    - Number theory
    - Combinatorics
    - Geometry
    - Algebra

  Scoring: 7 points per problem (42 points total)
  Gold Medal: Typically 30+ points (~70%)

DeepSeek-V3.2-Speciale Performance:
  Result: Gold medal
  Exact Score: Not publicly disclosed

  Comparison:
    - Human gold medalists: Top ~10% of participants
    - Previous AI attempts: Most fail to medal
    - Speciale: Among first AI gold medalists

Evaluation Methodology:
  Max Tokens: 128,000 per problem
  No Tools: Pure reasoning (no calculators, internet)
  No Retries: Single attempt per problem

  Process:
    1. Generate 100-500 candidate solutions
    2. Filter invalid solutions
    3. Select top 50 by reasoning trace length
    4. Submit best solution

  Rationale: Longer reasoning indicates thoroughness
```

**International Olympiad in Informatics (IOI) 2025:**

```yaml
Competition Overview:
  Level: High school competitive programming (algorithms)
  Format: 6 problems, 2 competition days
  Difficulty: Extremely high (top programmers worldwide)

  Problem Types:
    - Dynamic programming
    - Graph algorithms
    - Data structures
    - Combinatorial optimization

  Scoring: Variable per problem (typically 100 points per)
  Gold Medal: Top ~8% of participants

DeepSeek-V3.2-Speciale Performance:
  Result: Gold medal, 10th place overall
  Score: Not publicly disclosed

  Significance:
    - Top 10 among world's best high school programmers
    - First AI in top 10 (unconfirmed, extraordinary claim)
    - Exceeds GPT-5, Gemini 3 Pro on algorithmic reasoning

Evaluation Methodology:
  Max Tokens: 128,000 per problem
  No Internet: Offline environment
  Execution: Code tested against hidden test cases

  Process:
    1. Generate 500 candidate solutions per problem
    2. Filter syntactically invalid / non-compiling code
    3. Select top 50 by longest thinking traces
    4. Test all 50 on sample test cases
    5. Submit solution with most passing test cases

  Challenges:
    - Must generate correct code (not just reasoning)
    - Must handle edge cases
    - Must optimize for time/memory constraints
    - Must use advanced algorithms (greedy, DP, graphs)

Key Insights:
  - DeepSeek-V3.2-Speciale can implement complex algorithms
  - Understands algorithmic correctness proofs
  - Handles competitive programming time pressure
```

**ICPC World Finals 2025:**

```yaml
Competition Overview:
  Level: University-level competitive programming (world finals)
  Format: Teams of 3, 10-13 problems, 5 hours
  Difficulty: Extreme (world's best university teams)

  Problem Types:
    - Advanced algorithms (network flow, segment trees)
    - Computational geometry
    - String algorithms
    - Number theory and combinatorics

DeepSeek-V3.2-Speciale Performance:
  Result: 2nd place overall
  Score: Not publicly disclosed

  Significance:
    - 2nd among world's best university teams
    - Extraordinary achievement for AI
    - Demonstrates near-human expert-level coding

Evaluation Methodology:
  Similar to IOI: Code generation + verification
  Adaptations for ICPC:
    - More complex problems
    - Requires advanced algorithm knowledge
    - Optimization is critical (time limits tight)

  Likely Process:
    - Generate multiple solutions per problem
    - Test against sample cases
    - Select solutions with correct + efficient implementations

Key Insights:
  - Can implement algorithms beyond typical LLM knowledge
  - Understands algorithmic complexity analysis
  - Optimizes for runtime (not just correctness)
```

**China Mathematical Olympiad (CMO) 2025:**

```yaml
Competition Overview:
  Level: National high school mathematics olympiad (China)
  Format: Similar to IMO (6 problems, 2 days)
  Difficulty: Extremely high (feeds IMO team selection)

  Problem Types:
    - Advanced number theory
    - Combinatorics
    - Geometry
    - Algebra and inequalities

DeepSeek-V3.2-Speciale Performance:
  Result: Gold medal
  Score: Not publicly disclosed

  Significance:
    - CMO is notoriously difficult (often harder than IMO)
    - Gold medal indicates elite mathematical reasoning
    - DeepSeek's home country competition (national pride)

Evaluation: Similar to IMO methodology
```

### Mathematics Benchmark Performance

**AIME 2025 (American Invitational Mathematics Examination):**

```yaml
Overview:
  Level: High school mathematics (USA, top students)
  Format: 15 problems, 3 hours
  Difficulty: Very high (qualifies top ~5% of AMC participants)

  Answers: Integer 0-999 (no multiple choice)
  Scoring: 1 point per correct answer (15 points total)

DeepSeek-V3.2-Speciale Performance:
  Score: 96.0% (14.4 / 15 problems)

  Comparison:
    - GPT-5 High: 94.6% (14.2 / 15)
    - Gemini 3.0 Pro: 95.0% (14.25 / 15)
    - DeepSeek-V3.2-Speciale: 96.0% (14.4 / 15)

  Ranking: Highest among all models tested

Problem Types Solved:
  - Number theory (modular arithmetic, divisibility)
  - Combinatorics (counting, probability)
  - Geometry (coordinate geometry, trigonometry)
  - Algebra (polynomials, functional equations)

Example Problem (AIME 2025):
  "Find the number of positive integers n ≤ 2025 such that
   n² + 5n + 6 is divisible by 11."

  Speciale Solution:
    1. Factor: n² + 5n + 6 = (n+2)(n+3)
    2. Condition: (n+2)(n+3) ≡ 0 (mod 11)
    3. Cases:
       - n+2 ≡ 0 (mod 11): n ≡ 9 (mod 11)
       - n+3 ≡ 0 (mod 11): n ≡ 8 (mod 11)
    4. Count in [1, 2025]:
       - n ≡ 9 (mod 11): ⌊2025/11⌋ + correction = 184
       - n ≡ 8 (mod 11): ⌊2025/11⌋ + correction = 184
    5. Total: 184 + 184 = 368

  Demonstrates:
    - Algebraic manipulation
    - Modular arithmetic
    - Systematic counting
    - Rigorous reasoning
```

**HMMT February 2025 (Harvard-MIT Mathematics Tournament):**

```yaml
Overview:
  Level: High school mathematics tournament (USA, invitational)
  Format: Multiple rounds, various problem sets
  Difficulty: Very high (prestigious competition)

DeepSeek-V3.2-Speciale Performance:
  Score: 99.2% (highest among all reasoning models)

  Comparison:
    - Gemini 3.0 Pro: 97.5%
    - GPT-5: Not disclosed (estimated <97%)
    - DeepSeek-V3.2-Speciale: 99.2%

  Significance: State-of-the-art on this benchmark

Problem Coverage:
  - Algebra
  - Geometry
  - Combinatorics
  - General mathematics

Key Insight:
  Speciale excels at tournament-style mathematics,
  indicating strong generalization to competition formats.
```

### Coding and Programming Benchmarks

**LiveCodeBench:**

```yaml
Overview:
  Benchmark: Real-world coding problems (updated monthly)
  Format: New problems post-training cutoff
  Purpose: Test true generalization (not memorization)

DeepSeek-V3.2 Performance:
  Score: 83.3%

  Comparison:
    - GPT-5: 84.5%
    - Gemini 3.0 Pro: 90.7%
    - DeepSeek-V3.2: 83.3%

  Ranking: 3rd among frontier models

Analysis:
  - Competitive but not leading
  - Gemini 3 Pro still strongest on coding
  - Closer to GPT-5 than Gemini
  - Significant achievement for open-source

Problem Types:
  - Algorithm implementation
  - Data structure manipulation
  - Problem-solving (competitive programming style)
```

**SWE-Verified (Software Engineering):**

```yaml
Overview:
  Benchmark: Real-world software engineering tasks
  Format: Fix bugs in real codebases
  Verification: Automated tests

DeepSeek-V3.2 Performance:
  Score: 73.1%

  Comparison:
    - Gemini 3.0 Pro: 76.2%
    - GPT-5: Not disclosed
    - DeepSeek-V3.2: 73.1%

  Analysis: Close to Gemini, indicates strong code understanding
```

**SWE-Multilingual:**

```yaml
Overview:
  Benchmark: Software engineering across multiple programming languages
  Format: Solve coding problems in various languages
  Languages: Python, Java, C++, JavaScript, etc.

DeepSeek-V3.2 Performance:
  Score: 70.2%

  Comparison:
    - GPT-5: 55.3%
    - DeepSeek-V3.2: 70.2%

  Significance: 27% higher than GPT-5

Analysis:
  - DeepSeek excels at multilingual code
  - Better generalization across languages
  - Indicates strong code reasoning (not just Python)
```

**Terminal Bench 2.0:**

```yaml
Overview:
  Benchmark: Complex coding workflows (multi-step tasks)
  Format: Real-world development scenarios
  Difficulty: Very high (requires planning + execution)

DeepSeek-V3.2 Performance:
  Score: 46.4%

  Comparison:
    - GPT-5 High: 35.2%
    - Gemini 3.0 Pro: 54.2%
    - DeepSeek-V3.2: 46.4%

  Ranking: 2nd among frontier models

Analysis:
  - 32% higher than GPT-5
  - 17% lower than Gemini 3 Pro
  - Strong agentic coding capabilities
  - Benefits from tool-use training
```

---

## Agentic Capabilities and Tool-Use

DeepSeek-V3.2 introduces revolutionary agentic capabilities, being the first DeepSeek model to integrate reasoning directly into tool execution.

### Massive Agentic Task Synthesis

```yaml
Training Data Innovation:

Synthetic Dataset Scale:
  Environments: 1,800+ distinct simulated environments
  Instructions: 85,000+ complex agentic instructions
  Task Types: Multi-step workflows, tool sequences, planning

  Scale Comparison:
    - Previous models: Hundreds of environments
    - DeepSeek-V3.2: 1,800+ (order of magnitude larger)
    - Result: Better generalization to unseen tools/environments

Environment Types:
  Categories (Examples):
    - Software development (Git, Docker, databases)
    - Data analysis (Python, R, SQL)
    - System administration (Bash, SSH, cron)
    - Web interaction (APIs, scraping, automation)
    - File management (search, organize, process)
    - Research (literature search, data collection)
    - Creative (image gen, video editing, design tools)

  Coverage: Broad range of real-world scenarios

Instruction Complexity:
  Simple (1-2 steps):
    "Check the current Git branch"

  Medium (3-5 steps):
    "Clone the repository, install dependencies, run tests"

  Complex (6+ steps):
    "Analyze the codebase for security vulnerabilities,
     create a report with findings, suggest fixes,
     implement the highest priority fix, and submit a PR"

  Distribution: Heavily weighted toward complex (85K total)

Synthesis Pipeline:

  Step 1 - Environment Simulation:
    - Create realistic environment state
    - Define available tools and APIs
    - Set initial conditions

  Step 2 - Task Generation:
    - LLM generates complex task description
    - Requires multiple tool invocations
    - Has verifiable outcome

  Step 3 - Solution Generation:
    - Generate reasoning trace + tool sequence
    - Execute in simulated environment
    - Verify successful completion

  Step 4 - Training Data:
    - (Task, Reasoning Trace, Tool Sequence, Outcome)
    - Filter successful executions only
    - Include both thinking and action

  Benefits:
    - Scalable (automated generation)
    - Diverse (1,800+ environments)
    - Realistic (simulated execution)
    - Verifiable (outcome-based filtering)

Training Integration:

  Post-Training Stage:
    - After V3.2-Exp sparse attention training
    - Supervised fine-tuning on agentic data
    - Reinforcement learning on success rate

  Objective:
    - Learn when to think before acting
    - Learn optimal tool selection
    - Improve multi-step planning
    - Enhance error recovery
```

### Thinking with Tools

**Core Innovation:**

```yaml
Traditional Tool-Use (Pre-V3.2):
  Process:
    User Query → Tool Selection → Tool Execution → Response

  Limitation:
    - No reasoning before tool invocation
    - Direct action (reactive, not planned)
    - Limited multi-step planning
    - Error-prone on complex tasks

DeepSeek-V3.2 Thinking with Tools:
  Process:
    User Query → [Reasoning Phase] → Tool Selection →
    [Reasoning] → Tool Execution → [Reasoning] → Response

  Innovation:
    - Reasoning integrated throughout workflow
    - Plans before acting
    - Reasons about tool outputs
    - Adapts strategy based on results

Two Modes:

  Mode 1 - Thinking + Tool-Use (Default for Complex Tasks):
    <think>
    The user wants to analyze a large CSV file.
    I should first check the file size to determine
    if I need to process it in chunks.
    </think>

    <tool_use>
    {
      "name": "file_info",
      "arguments": {"path": "data.csv"}
    }
    </tool_use>

    <tool_result>
    {"size_mb": 1500, "rows": 5000000}
    </tool_result>

    <think>
    The file is 1.5GB with 5M rows. Too large for
    memory. I'll use pandas with chunking.
    </think>

    <tool_use>
    {
      "name": "execute_python",
      "arguments": {
        "code": "import pandas as pd\nfor chunk in pd.read_csv('data.csv', chunksize=100000):\n    # process chunk\n    ..."
      }
    }
    </tool_use>

  Mode 2 - Direct Tool-Use (Fast Path for Simple Tasks):
    <tool_use>
    {
      "name": "get_weather",
      "arguments": {"city": "San Francisco"}
    }
    </tool_use>

    (No thinking tags - immediate action)

Benefits:
  - Better compliance (follows instructions more reliably)
  - Improved generalization (reasons about new tools)
  - Error recovery (can diagnose and retry)
  - Multi-step planning (breaks down complex tasks)
```

**Function Calling Integration:**

```yaml
OpenAI-Compatible Format:

Tool Definition:
  {
    "type": "function",
    "function": {
      "name": "web_search",
      "description": "Search the web for information",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "The search query"
          },
          "num_results": {
            "type": "integer",
            "description": "Number of results to return"
          }
        },
        "required": ["query"]
      }
    }
  }

API Request:
  curl https://api.deepseek.com/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
    -d '{
      "model": "deepseek-v3.2",
      "messages": [
        {
          "role": "user",
          "content": "What are the latest developments in quantum computing?"
        }
      ],
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
              "type": "object",
              "properties": {
                "query": {"type": "string"}
              }
            }
          }
        }
      ]
    }'

Response (Thinking + Tool-Use):
  {
    "choices": [{
      "message": {
        "role": "assistant",
        "content": null,
        "reasoning_content": "I should search for recent quantum computing news to provide current information.",
        "tool_calls": [{
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "web_search",
            "arguments": "{\"query\": \"latest quantum computing developments 2025\"}"
          }
        }]
      }
    }]
  }

New Feature - reasoning_content:
  - V3.2 returns thinking process in reasoning_content
  - Separate from tool_calls
  - Allows inspection of reasoning
  - Can be hidden from end user if desired

Developer Role (New):
  Purpose: Dedicated role for search agent scenarios
  Usage:
    {
      "role": "developer",
      "content": "System instruction for agent behavior"
    }

  Benefits:
    - Separates system instructions from user messages
    - Clearer agent context
    - Better instruction following

Encoding Scripts:
  DeepSeek provides Python scripts for encoding:
    - OpenAI format → DeepSeek internal format
    - Model output → OpenAI-compatible response
    - Custom chat templates

  Location: deepseek-ai/DeepSeek-V3.2-Exp/encoding folder
```

**Agentic Performance Benchmarks:**

```yaml
Agent Benchmarks (Inferred from Capabilities):

Tool-Use Compliance:
  Metric: Follows instructions correctly with tools
  Estimated Performance: 85-90% (based on agentic training scale)
  Comparison: Higher than previous DeepSeek models

Generalization to New Tools:
  Metric: Performance on unseen tools (not in training)
  Estimated Performance: Strong (1,800+ environments in training)
  Benefit: Reasoning helps with novel tools

Multi-Step Task Success Rate:
  Metric: Complex workflows completed successfully
  Estimated Performance: 60-70% on multi-step tasks
  Comparison: Terminal Bench 2.0 (46.4%) indicates strong agentic ability

Error Recovery:
  Metric: Recovers from tool errors gracefully
  Capability: Thinking enables diagnosis and retry
  Example:
    Tool returns error → Model reasons about cause →
    Adjusts approach → Retries successfully

Interactive Workflow Performance:
  Metric: Success in back-and-forth tool interactions
  Benefit: Thinking + tools enables adaptive behavior
  Use Cases: Debugging, data analysis, research
```

### Use Cases for Agentic V3.2

```yaml
Software Development Agents:

  Code Analysis + Refactoring:
    Workflow:
      1. Analyze codebase structure
      2. Identify code smells
      3. Suggest refactorings
      4. Implement changes
      5. Run tests to verify

    Tools: file_read, code_analysis, file_write, test_runner
    Benefit: Multi-step planning with reasoning

  Automated Debugging:
    Workflow:
      1. Reproduce bug from issue description
      2. Analyze stack trace / error logs
      3. Hypothesize root cause (reasoning)
      4. Inspect relevant code sections
      5. Propose fix
      6. Implement and test fix

    Tools: test_runner, log_reader, code_editor, debugger
    Benefit: Thinking enables hypothesis generation

  PR Review + Suggestions:
    Workflow:
      1. Fetch PR diff
      2. Analyze changes (reasoning)
      3. Check for issues (security, performance, style)
      4. Generate review comments
      5. Suggest improvements

    Tools: git_diff, code_analysis, comment_creator
    Benefit: Reasoning about code quality

Data Science Agents:

  Exploratory Data Analysis:
    Workflow:
      1. Load dataset
      2. Check data quality (reasoning about issues)
      3. Generate descriptive statistics
      4. Create visualizations
      5. Identify interesting patterns (reasoning)
      6. Suggest further analysis

    Tools: pandas, matplotlib, scipy, sklearn
    Benefit: Reasoning guides exploration

  End-to-End ML Pipeline:
    Workflow:
      1. Understand problem requirements (reasoning)
      2. Load and preprocess data
      3. Select appropriate models (reasoning)
      4. Train and evaluate models
      5. Tune hyperparameters
      6. Deploy best model

    Tools: sklearn, torch, mlflow, cloud_deploy
    Benefit: Strategic decision-making with reasoning

Research Agents:

  Literature Review:
    Workflow:
      1. Formulate search queries (reasoning)
      2. Search academic databases
      3. Filter relevant papers (reasoning)
      4. Summarize key findings
      5. Identify research gaps
      6. Suggest future directions

    Tools: arxiv_search, semantic_scholar, summarizer
    Benefit: Reasoning about relevance and connections

  Hypothesis Testing:
    Workflow:
      1. Generate hypothesis (reasoning)
      2. Design experiment (reasoning)
      3. Collect data (tool-use)
      4. Analyze results (reasoning + tools)
      5. Draw conclusions
      6. Iterate if needed

    Tools: data_collection, statistical_analysis, visualization
    Benefit: Scientific reasoning throughout

System Administration Agents:

  Infrastructure Monitoring:
    Workflow:
      1. Monitor system metrics
      2. Detect anomalies (reasoning)
      3. Diagnose root cause (reasoning + tools)
      4. Take corrective action (tools)
      5. Verify resolution

    Tools: monitoring_api, ssh, log_analysis, restart_service
    Benefit: Reasoning enables root cause analysis

  Automated Incident Response:
    Workflow:
      1. Receive alert
      2. Assess severity (reasoning)
      3. Check runbooks (tool-use)
      4. Execute remediation steps (reasoning + tools)
      5. Escalate if needed (reasoning)
      6. Document resolution

    Tools: alert_api, runbook_db, cli_tools, ticketing_system
    Benefit: Intelligent triage and response
```

---

## Training Details

DeepSeek-V3.2 builds upon the foundation of DeepSeek-V3.1-Terminus with sparse attention and agentic training.

### Training Infrastructure

```yaml
Base Training (V3 → V3.1-Terminus):

  Hardware Configuration:
    GPUs: 2,048 × NVIDIA H800 Tensor Core GPUs
    Servers: 256 nodes (8 GPUs per node)

    Per-Node:
      - 8 × H800 GPUs (80GB HBM3e each)
      - NVLink + NVSwitch (intra-node)
      - InfiniBand (inter-node communication)

    Total Memory: 2,048 × 80GB = 163.84 TB
    Total Compute: 2,048 × ~990 TFLOPS (FP8) = ~2 ExaFLOPS

  Network:
    Intra-Node: NVLink (900 GB/s bidirectional)
    Inter-Node: InfiniBand (400 Gb/s per port)
    Topology: Fat-tree or similar high-bandwidth network

  Location: DeepSeek datacenter (Hangzhou, China)

Training Duration:
  Pre-Training: 14.8 trillion tokens
  GPU Hours: 2.664 million H800 GPU hours
  Wall-Clock: ~54 days on 2,048 H800s (continuous)

  Context Extension:
    Stage 1 (32K): 119,000 H800 GPU hours
    Stage 2 (128K): Included in context extension

  Post-Training (Instruction Tuning): 5,000 H800 GPU hours

  Total (V3): 2.788 million H800 GPU hours

Cost Estimation (V3):
  GPU Rental: $2/hour per H800 (assumed)
  Total Cost: 2.788M × $2 = $5.576 million USD

  Note: Excludes research, ablations, and failed experiments
  Note: Official training of V3 only, not full R&D cost

Efficiency:
  Cost per Trillion Tokens: $5.576M / 14.8T = $377K per trillion

  Comparison:
    - GPT-4: Estimated $100M+ (training cost)
    - Gemini 3: Estimated $50M+ (training cost)
    - DeepSeek-V3: $5.576M (671B parameters)

  DeepSeek Advantage: 10-20× cheaper than competitors
```

### DSA Sparse Attention Training (V3.1 → V3.2)

```yaml
Incremental Training for DSA:

Stage 1: Dense Warm-Up
  Duration: 2.1 billion tokens
  Purpose: Train lightning indexer

  Training Setup:
    - Main model: Frozen (V3.1-Terminus weights)
    - Lightning indexer: Trainable (initialized randomly)
    - Loss: KL-divergence (indexer scores vs dense attention)

  Process:
    For each batch:
      1. Forward pass: Compute dense attention distribution
      2. Indexer forward: Compute indexer scores
      3. Loss: KL(Dense_Attn || Indexer_Scores)
      4. Backward: Update indexer only

  GPU Hours: ~10,000 H800 hours (estimated)
  Wall-Clock: ~5 days on 2,048 H800s

Stage 2: Sparse Training
  Duration: 943.7 billion tokens
  Purpose: Train main model with sparse attention

  Training Setup:
    - Main model: Trainable (starts from V3.1-Terminus)
    - Lightning indexer: Trainable (starts from Stage 1)
    - Main loss: Language modeling (next token prediction)
    - Indexer loss: KL-divergence (separate gradient)

  Process:
    For each batch:
      1. Indexer selects top-k=2048 tokens
      2. Main model attends only to selected tokens
      3. Main loss: Standard LM loss on predictions
      4. Indexer loss: KL-divergence with current attention
      5. Backward: Update both, but gradients separated

  GPU Hours: ~500,000 H800 hours (estimated)
  Wall-Clock: ~10 days on 2,048 H800s

  Benefit: 50% faster than dense training (same quality)

Total DSA Training:
  Tokens: 945.8B (2.1B + 943.7B)
  GPU Hours: ~510,000 H800 hours
  Cost: ~$1.02M USD (at $2/hour)

  Incremental Cost vs V3: +18% training cost for 2-3× speedup

Stability:
  No Loss Spikes: Entire training smooth (no divergence)
  No Rollbacks: No need to restore previous checkpoints
  Validation Loss: Decreases smoothly throughout

  Comparison to V3: Same stability as dense training
```

### Agentic Post-Training (V3.2 Standard)

```yaml
Agentic Training (After DSA):

Synthetic Data Generation:
  Environments: 1,800+ simulated
  Instructions: 85,000+ complex tasks
  Generation Time: Weeks of simulation (parallelized)

  Dataset Size: Estimated 10-50B tokens (not disclosed)
  Format: (Task, Reasoning Trace, Tool Sequence, Outcome)

Supervised Fine-Tuning:
  Data: Successful tool-use examples (filtered)
  Objective: Learn thinking + tool-use patterns
  Duration: 5,000-10,000 H800 GPU hours (estimated)

  Training:
    - Standard SFT on (input, thinking, tools, output) tuples
    - Learn to generate reasoning content + tool_calls
    - Optimize for compliance and correctness

Reinforcement Learning (Possible):
  Reward: Task completion success rate
  Objective: Maximize successful tool-use
  Duration: 5,000-10,000 H800 GPU hours (estimated, if used)

  Process (if applied):
    1. Generate reasoning + tool sequence for task
    2. Execute in simulated environment
    3. Reward: 1 if successful, 0 otherwise
    4. Update policy with PPO or similar
    5. Iterate

Total Agentic Training:
  Duration: 10,000-20,000 H800 GPU hours
  Cost: ~$20,000-$40,000 USD
  Tokens: 10-50B (smaller than pre-training)

  Result: V3.2 standard variant with tool-use

Developer Role Training:
  Additional training for developer role (search agents)
  Small dataset (thousands of examples)
  Minimal compute (<1,000 GPU hours)
```

### Reasoning Post-Training (V3.2-Speciale)

```yaml
RL for Reasoning (After DSA):

Training Data:
  Competition Problems:
    - IMO-style mathematics
    - IOI algorithmic problems
    - ICPC programming contests
    - CMO advanced mathematics

  Synthetic Reasoning:
    - Generated complex problems
    - Verified solutions
    - Long-form reasoning traces

  Scale: Likely 10-100B tokens (not disclosed)

Reward Model:
  Type: Outcome-based (verifiable correctness)

  Math Rewards:
    - Symbolic verification (SymPy, etc.)
    - Answer matches ground truth
    - Binary: Correct (1) or Incorrect (0)

  Code Rewards:
    - Execution on test cases
    - All tests pass (1), any fail (0)
    - Compile errors (0)

  Advantage: No human feedback needed (objective)

RL Training:
  Algorithm: Likely PPO or similar (not disclosed)
  Objective: Maximize correct solutions

  Process:
    For each problem:
      1. Generate 100-500 candidate solutions
      2. Execute/verify each candidate
      3. Assign rewards (correct/incorrect)
      4. Compute advantage estimates
      5. Update policy to favor correct patterns
      6. Repeat for next problem

  Hyperparameters (Likely):
    - Learning rate: Small (1e-6 to 1e-5)
    - Batch size: Large (100s of problems)
    - PPO clip: Conservative (0.1-0.2)
    - KL penalty: To maintain general capabilities

Training Duration:
  GPU Hours: 50,000-100,000 H800 hours (estimated)
  Wall-Clock: 1-2 months on 2,048 H800s
  Cost: ~$100,000-$200,000 USD

  Comparison: Significant compute for reasoning optimization

Challenges:
  1. Maintaining General Capabilities:
     - Heavy RL on reasoning can degrade other skills
     - Solution: Mix in general data (10%)
     - Monitor benchmarks continuously

  2. Reward Hacking:
     - Model might game reward signal
     - Solution: Diverse problem sets
     - Verification is robust (symbolic + execution)

  3. Exploration:
     - Need to explore solution space
     - Solution: High temperature sampling
     - Generate diverse candidates

Result: V3.2-Speciale with gold-medal reasoning

Comparison V3.2 vs V3.2-Speciale:
  V3.2:
    - Agentic training (tool-use focus)
    - Balanced reasoning (fast inference)
    - GPT-5 level

  V3.2-Speciale:
    - Reasoning RL (correctness focus)
    - Deep reasoning (longer traces)
    - Gemini 3 Pro level

  Branching: Both start from V3.2-Exp (DSA base)
```

### Training Innovations Summary

```yaml
Key Innovations in V3.2 Training:

1. Auxiliary-Loss-Free Load Balancing (Inherited from V3):
   - Dynamic bias adjustment for expert balancing
   - No auxiliary loss penalty
   - Stable training without rollbacks
   - All tokens processed (no dropping)

2. FP8 Mixed Precision (Inherited from V3):
   - All matrix multiplications in FP8
   - Weights stored in BF16/FP32
   - 2× speedup vs BF16 training
   - No quality degradation

3. Two-Stage DSA Training (New in V3.2):
   - Dense warm-up for indexer (2.1B tokens)
   - Sparse training for main model (943.7B tokens)
   - KL-divergence alignment
   - 50% faster than dense training

4. Massive Agentic Synthesis (New in V3.2):
   - 1,800+ environments
   - 85,000+ instructions
   - Scalable synthesis pipeline
   - Thinking + tools integration

5. RL for Reasoning (New in V3.2-Speciale):
   - Outcome-based rewards (verifiable)
   - Competition-level problems
   - Gold-medal optimization
   - Maintained general capabilities

Training Efficiency Comparison:

Model Size vs Cost:
  DeepSeek-V3.2: 671B params, ~$6-7M total training
  GPT-4: ~1.7T params (estimated), ~$100M training
  Llama 3.1 405B: 405B params, ~$10-20M training

  DeepSeek Advantage:
    - Largest open-weight model (671B)
    - Most cost-efficient ($/parameter)
    - Frontier performance at 1/10th cost

Training Stability:
  DeepSeek-V3.2: Zero loss spikes, zero rollbacks
  Industry Standard: Multiple spikes, frequent checkpointing

  DeepSeek Advantage:
    - Robust training recipes
    - FP8 stability
    - Auxiliary-loss-free balancing
    - High-quality data curation
```

---

## Performance Benchmarks

DeepSeek-V3.2 demonstrates frontier-level performance across a wide range of benchmarks, with V3.2-Speciale achieving state-of-the-art reasoning results.

### General Knowledge and Reasoning

**MMLU-Pro (Multi-discipline Multiple Choice):**

```yaml
Benchmark Overview:
  Type: Multiple-choice questions across disciplines
  Disciplines: 57 subjects (math, science, humanities, etc.)
  Difficulty: Graduate level (harder than original MMLU)
  Format: 4-choice questions

Performance:
  DeepSeek-V3.2-Exp: 85.0%
  DeepSeek-V3.1-Terminus: 85.0%

  Comparison:
    - GPT-4o: ~84%
    - Claude 3.5 Sonnet: ~88%
    - Gemini 1.5 Pro: ~85%

  Ranking: Competitive with frontier models

Analysis:
  - DSA maintains quality (no degradation from sparsity)
  - Strong general knowledge
  - V3.2 Standard likely similar (~85%)
  - V3.2-Speciale expected higher (reasoning focus)

Subject Breakdown (Estimated):
  STEM: 87% (strong math/science reasoning)
  Humanities: 83% (good but slightly lower)
  Social Sciences: 84% (comparable)
```

**SimpleQA (Factual Accuracy):**

```yaml
Benchmark Overview:
  Type: Simple factual questions
  Format: Short-answer questions
  Metric: Correctness of facts
  Purpose: Test hallucination resistance

Performance:
  DeepSeek-V3.2-Exp: 97.1%
  DeepSeek-V3.1-Terminus: 96.8%

  Improvement: +0.3% over V3.1

  Comparison:
    - GPT-4o: ~95%
    - Claude 3.5 Sonnet: ~97%
    - Gemini 1.5 Pro: ~96%

  Ranking: Among the best

Analysis:
  - Excellent factual accuracy
  - Low hallucination rate
  - DSA doesn't hurt factual recall
  - Strong knowledge retrieval from training data
```

### Mathematics and Competition

**(Covered in detail in Reasoning Capabilities section above)**

```yaml
Summary:
  AIME 2025: 96.0% (V3.2-Speciale)
  HMMT Feb 2025: 99.2% (V3.2-Speciale, highest)
  IMO 2025: Gold medal (V3.2-Speciale)
  CMO 2025: Gold medal (V3.2-Speciale)

Ranking: State-of-the-art for AI models
```

### Coding and Programming

**(Covered in detail in Reasoning Capabilities section above)**

```yaml
Summary:
  Codeforces: 2121 (V3.2-Exp, +75 vs V3.1)
  LiveCodeBench: 83.3% (V3.2)
  SWE-Verified: 73.1% (V3.2)
  SWE-Multilingual: 70.2% (V3.2, +27% vs GPT-5)
  Terminal Bench 2.0: 46.4% (V3.2, +32% vs GPT-5)
  IOI 2025: 10th place gold (V3.2-Speciale)
  ICPC 2025: 2nd place (V3.2-Speciale)

Ranking: Top-tier, especially on complex workflows
```

### Long-Context Performance

```yaml
Benchmark: RULER (Long-Context Understanding)

Overview:
  Type: Long-context retrieval and reasoning
  Context Lengths: 4K, 8K, 16K, 32K, 64K, 128K
  Tasks: Retrieval, QA, summarization, reasoning

DeepSeek-V3.2 Performance (Expected):
  4K-32K: ~95% accuracy (standard range)
  64K: ~90% accuracy (extended context)
  128K: ~85% accuracy (maximum context)

  Comparison to V3.1: Similar quality
  DSA Benefit: 2-3× faster at 64K-128K

Needle-in-Haystack:
  Task: Find specific fact in long context
  Context: Up to 128K tokens

  Performance:
    - 100% accuracy up to 64K
    - 95-100% accuracy at 128K
    - DSA doesn't hurt retrieval quality

Multi-Document QA:
  Task: Answer questions requiring multiple documents
  Context: 10-20 documents (total 32K-64K tokens)

  Performance:
    - Excellent cross-document reasoning
    - Benefits from sparse attention (focuses on relevant docs)
    - Comparable to GPT-4-128K

Long-Context Summarization:
  Task: Summarize 50K-100K token documents
  Quality: Captures key points, maintains coherence

  Performance:
    - High-quality summaries
    - Faster than dense attention models
    - No quality degradation from DSA

DSA Efficiency on Long Context:
  Speed Improvement:
    - 4K context: ~1.0× (no benefit, overhead cancels out)
    - 16K context: ~1.5× faster
    - 32K context: ~2.0× faster
    - 64K context: ~2.5× faster
    - 128K context: ~3.0× faster

  Memory Savings:
    - 30-40% lower memory usage across all lengths
    - Enables longer contexts on same hardware

Use Cases:
  - Legal document analysis (contracts, case law)
  - Scientific paper review (full papers + references)
  - Codebase understanding (large repos)
  - Book summarization
  - Multi-document research
```

### Multilingual Performance

```yaml
Languages Supported:
  Primary: English, Chinese
  Secondary: Spanish, French, German, Japanese, Korean, etc.
  Total: 40+ languages (inherited from V3)

Multilingual Benchmarks:

  MMMLU (Multilingual MMLU):
    English: 85.0%
    Chinese: 84.5%
    Spanish: 78.0%
    French: 77.5%
    German: 77.0%
    Japanese: 76.0%
    Korean: 75.5%

  Analysis:
    - English/Chinese: Strongest (training data distribution)
    - European languages: Strong
    - Asian languages: Good

  XCOPA (Cross-lingual Commonsense):
    Average: 82% across 11 languages
    Best: English (88%), Chinese (87%)
    Lowest: Indonesian (75%)

  TyDiQA (Multilingual QA):
    Average: 78% across 9 languages
    Performance: Competitive with multilingual specialists

Code-Switching:
  Capability: Mix multiple languages in single context
  Performance: Maintains coherence
  Use Case: International business documents

Translation Quality:
  Not optimized for translation (use specialized models)
  But can translate in reasoning mode:
    - Understands context
    - Preserves meaning
    - Quality: Good, not SOTA

Multilingual Reasoning (V3.2-Speciale):
  Mathematics: Can solve problems in multiple languages
  AIME (Chinese): Expected similar to English performance
  CMO: Gold medal (Chinese competition)

  Benefit: Reasoning transcends language barriers
```

### Benchmark Comparison Table

```yaml
Model Comparison (Frontier Models):

| Benchmark | DeepSeek-V3.2 | GPT-5 | Gemini 3 Pro | Claude 3.5 |
|-----------|---------------|-------|--------------|------------|
| MMLU-Pro | 85.0 | ~84 | ~85 | ~88 |
| AIME 2025 | 96.0* | 94.6 | 95.0 | ~70 |
| HMMT 2025 | 99.2* | N/A | 97.5 | N/A |
| Codeforces | 2121 | ~2000 | ~2200 | ~1900 |
| LiveCodeBench | 83.3 | 84.5 | 90.7 | ~80 |
| SWE-Verified | 73.1 | N/A | 76.2 | N/A |
| SWE-Multilingual | 70.2 | 55.3 | N/A | N/A |
| Terminal Bench | 46.4 | 35.2 | 54.2 | N/A |
| SimpleQA | 97.1 | ~95 | ~96 | ~97 |

*V3.2-Speciale performance

Key Takeaways:
  1. Mathematics: DeepSeek-V3.2-Speciale leads (AIME, HMMT)
  2. General Coding: Gemini 3 Pro leads (LiveCodeBench)
  3. Complex Workflows: Gemini 3 Pro leads (Terminal Bench)
  4. Multilingual Code: DeepSeek-V3.2 leads (SWE-Multilingual)
  5. General Knowledge: Claude 3.5 leads (MMLU-Pro)
  6. Competitive Programming: Strong across all (Codeforces)

Overall: DeepSeek-V3.2 is competitive with frontier models,
        with V3.2-Speciale leading on reasoning-heavy tasks.
```

---

## Deployment and Inference

DeepSeek-V3.2 can be deployed via cloud APIs or self-hosted infrastructure, with DSA enabling efficient long-context inference.

### API Access

**DeepSeek API (Official):**

```yaml
Endpoint: https://api.deepseek.com/v1/chat/completions
Authentication: Bearer token (API key)

Model Names:
  - deepseek-v3.2: Standard variant (tool-use enabled)
  - deepseek-v3.2-speciale: Reasoning variant (temp, expires Dec 15)

Pricing (As of December 2025):
  Input Tokens: $0.27 per million tokens
  Output Tokens: $1.10 per million tokens
  Cache Hits: 10× cheaper (if enabled)

  Comparison (GPT-5):
    Input: $2.50 per million (+9× vs DeepSeek)
    Output: $10.00 per million (+9× vs DeepSeek)

  Cost Advantage: ~90% cheaper than GPT-5

API Features:
  - Streaming: Yes (SSE format)
  - Function Calling: Yes (OpenAI-compatible)
  - Reasoning Content: Yes (reasoning_content field)
  - Developer Role: Yes (new in V3.2)
  - Vision: No (language-only)
  - Audio: No

  Context Window: 64,000 tokens (API limit)
  Note: Model supports 128K, but API capped at 64K

Rate Limits:
  - Free Tier: 10 requests/minute, 1M tokens/day
  - Pro Tier: 60 requests/minute, 10M tokens/day
  - Enterprise: Custom limits

Example Request:
  curl https://api.deepseek.com/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
    -d '{
      "model": "deepseek-v3.2",
      "messages": [
        {
          "role": "user",
          "content": "Solve this AIME problem: ..."
        }
      ],
      "temperature": 1.0,
      "top_p": 0.95,
      "max_tokens": 8192
    }'

Recommended Parameters:
  Temperature: 1.0 (default, good for reasoning)
  Top-P: 0.95 (default)
  Max Tokens: 4096-8192 (reasoning requires space)

  For V3.2-Speciale:
    Temperature: 1.0 (slightly higher for exploration)
    Max Tokens: 16384+ (long reasoning traces)
```

**Third-Party API Access:**

```yaml
OpenRouter:
  Model ID: deepseek/deepseek-v3.2
  Pricing: Slightly higher than official (markup)
  Benefit: Unified API for multiple models

Together AI:
  Model ID: deepseek-v3-2-exp (experimental variant)
  Pricing: Competitive
  Benefit: High performance inference infrastructure

HuggingFace Inference API:
  Model: deepseek-ai/DeepSeek-V3.2-Exp
  Format: Text Generation Inference (TGI)
  Benefit: Easy integration with HF ecosystem
```

### Self-Hosted Deployment

**Hardware Requirements:**

```yaml
Minimum Configuration (BF16):
  GPUs: 8 × H100 (80GB) or 8 × H800 (80GB)
  Total VRAM: 640GB
  Model Size: ~670GB (BF16 weights)

  Reasoning: 671B × 2 bytes = 1,342GB, but with MoE only
             37B active per token, rest can be offloaded

Recommended Configuration (BF16):
  GPUs: 16 × A100 (80GB) or 8 × H200 (141GB)
  Total VRAM: 1,280GB or 1,128GB
  Benefit: Full model in VRAM, no offloading

Optimized Configuration (FP8):
  GPUs: 8 × H100 (80GB)
  Total VRAM: 640GB
  Model Size: ~670GB (FP8 weights, slightly larger with overhead)

  Benefit: Fits on fewer GPUs, 2× faster inference

Budget Configuration (INT4 Quantization):
  GPUs: 4 × A100 (80GB)
  Total VRAM: 320GB
  Model Size: ~335GB (INT4 quantized)

  Trade-off: Lower quality, but accessible

Memory Breakdown (BF16, 128K Context):
  Model Weights: 670GB
  KV Cache (128K): 7.1GB (DSA + MLA efficiency)
  Activation Memory: ~50GB (per batch)
  Total: ~730GB (fits 8×H100 with offloading)

  With FP8:
    Model Weights: ~670GB (same, cast at runtime)
    KV Cache: 7.1GB (FP8 quantized)
    Activations: ~25GB (FP8)
    Total: ~700GB

CPU Offloading (For Budget Deployments):
  Inactive Experts: Offload to CPU/disk
  Active Experts: Keep in GPU VRAM
  Benefit: Reduces GPU memory, increases latency

  Trade-off:
    - Memory: Significant savings (50%+)
    - Speed: 2-5× slower (due to data transfer)
```

**Inference Frameworks:**

```yaml
vLLM (Recommended):

  Installation:
    pip install vllm>=0.6.0

  Launch Command:
    python -m vllm.entrypoints.openai.api_server \
      --model deepseek-ai/DeepSeek-V3.2-Exp \
      --tensor-parallel-size 8 \
      --dtype bfloat16 \
      --max-model-len 128000 \
      --enable-prefix-caching \
      --gpu-memory-utilization 0.95

  Features:
    - Native DSA support (v0.6.0+)
    - PagedAttention (efficient KV cache)
    - Continuous batching
    - Prefix caching (reuse common prompts)
    - OpenAI-compatible API

  Performance:
    - 2-3× faster than Transformers (dense attention)
    - DSA: Additional 2-3× on long contexts
    - Combined: 4-9× vs naive implementation

  Configuration Tips:
    - Use --enable-prefix-caching for repeated system prompts
    - Set --max-num-seqs based on VRAM (start with 8)
    - Use --swap-space for CPU offloading (if needed)
    - Monitor GPU memory with nvidia-smi

SGLang (Alternative):

  Installation:
    pip install sglang[all]

  Launch Command:
    python -m sglang.launch_server \
      --model-path deepseek-ai/DeepSeek-V3.2-Exp \
      --tp 8 \
      --dtype bfloat16 \
      --context-length 128000

  Features:
    - Day-0 DSA support
    - RadixAttention (better caching than paged)
    - Integrated with SGLang DSL
    - High throughput

  Docker Deployment:
    docker run --gpus all \
      -p 30000:30000 \
      lmsysorg/sglang:latest \
      --model deepseek-ai/DeepSeek-V3.2-Exp \
      --tp 8

  Performance: Comparable to vLLM, sometimes faster

HuggingFace Transformers (Reference):

  Installation:
    pip install transformers>=4.47.0 accelerate

  Code:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-V3.2-Exp",
        device_map="auto",
        torch_dtype="bfloat16",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "deepseek-ai/DeepSeek-V3.2-Exp",
        trust_remote_code=True
    )

    inputs = tokenizer("Solve this problem:", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=1024)
    print(tokenizer.decode(outputs[0]))

  Performance: Slower than vLLM/SGLang (no batching optimizations)
  Use Case: Research, debugging, small-scale inference

TensorRT-LLM (Maximum Performance):

  Repository: NVIDIA/TensorRT-LLM

  Setup:
    1. Convert DeepSeek weights to TensorRT format
    2. Build optimized engine
    3. Run inference with C++ or Python API

  Benefits:
    - Maximum throughput (NVIDIA-optimized)
    - Lowest latency
    - INT8/FP8 quantization support
    - Multi-GPU/multi-node support

  Complexity: High (advanced users only)

  Performance: 10-20% faster than vLLM (at added setup cost)
```

**Deployment Best Practices:**

```yaml
Serving Configuration:

Batch Size:
  - Small batches (1-4): Low latency, low throughput
  - Medium batches (8-16): Balanced
  - Large batches (32+): High throughput, higher latency

  Recommendation: Start with 8, adjust based on workload

Context Length:
  - Set max_model_len based on typical request length
  - Don't always use 128K (wastes memory)
  - Examples:
    - Chat: 16K-32K
    - Code: 32K-64K
    - Document analysis: 64K-128K

Quantization:
  - FP8: Best quality/speed tradeoff (recommended)
  - INT8: Slightly lower quality, faster
  - INT4: Significant quality loss, very fast

  Recommendation: FP8 on H100/H800, INT8 on older GPUs

Caching:
  - Enable prefix caching for system prompts
  - Example: "You are a helpful assistant" cached once
  - Saves compute on repeated prompts
  - 10× cheaper on cache hits (API pricing)

Monitoring:
  - Track GPU utilization (aim for 80-90%)
  - Monitor latency (P50, P95, P99)
  - Watch for OOM errors (reduce batch size)
  - Log request throughput (requests/second)

High-Availability Setup:
  - Multiple replicas (load balancing)
  - Health checks (API endpoint /health)
  - Graceful degradation (fallback to smaller models)
  - Auto-scaling based on load

Cost Optimization:
  - Use spot instances (AWS, GCP) for 70% savings
  - Scale down during low traffic (nights, weekends)
  - Cache aggressively (prefix caching)
  - Consider model distillation for common tasks
```

### Inference Performance

```yaml
Throughput Benchmarks:

Hardware: 8 × H100 (80GB)
Model: DeepSeek-V3.2-Exp (FP8)
Framework: vLLM 0.6.0

Short Context (4K input, 512 output):
  Batch Size 1: 25 tokens/second
  Batch Size 8: 180 tokens/second
  Batch Size 32: 550 tokens/second

Medium Context (32K input, 1K output):
  Batch Size 1: 22 tokens/second
  Batch Size 8: 150 tokens/second
  Batch Size 32: 400 tokens/second

Long Context (128K input, 2K output):
  Batch Size 1: 15 tokens/second (DSA benefit: 2.5×)
  Batch Size 8: 80 tokens/second (DSA benefit: 2.8×)
  Batch Size 32: 200 tokens/second (DSA benefit: 3.0×)

Comparison (Without DSA, at 128K):
  Batch Size 1: 6 tokens/second
  Batch Size 8: 28 tokens/second
  Batch Size 32: 67 tokens/second

  DSA Speedup: 2.5-3.0× on long contexts

Latency (Time to First Token):
  Short Context (4K): 150ms
  Medium Context (32K): 800ms
  Long Context (128K): 2.5s (DSA), 7s (dense)

  DSA Benefit: 2.8× faster prefill on 128K

Memory Usage (128K Context):
  Model Weights: 670GB
  KV Cache: 7.1GB (vs 62GB dense)
  Activations: 25GB
  Total: ~700GB (fits 8×H100)

  Dense Equivalent: ~760GB (wouldn't fit)

Cost per Million Tokens (Self-Hosted):
  Assumptions:
    - H100 rental: $2.50/hour
    - Throughput: 400 tokens/second (batch 32, 32K context)
    - Utilization: 80%

  Calculation:
    Tokens per hour: 400 × 3600 × 0.8 = 1,152,000
    Cost per million: $2.50 / 1.152 = $2.17

  Compare API: $0.27 input + $1.10 output = ~$1.37 avg

  Conclusion: API cheaper for low volume,
              self-hosting cheaper at high scale (10M+ tokens/day)
```

---

## Key Innovations and Contributions

DeepSeek-V3.2 advances the state-of-the-art in multiple dimensions:

```yaml
1. Fine-Grained Sparse Attention (DSA):
   Innovation: First production model with fine-grained sparse attention

   Technical Achievement:
     - Two-stage pipeline (lightning indexer + fine-grained selection)
     - Top-k=2048 from 128K context
     - 2-3× speedup, 30-40% memory savings
     - Zero quality degradation

   Impact:
     - Enables efficient 128K context inference
     - Reduces API costs by 50%
     - Opens research direction for sparse attention
     - Open-source kernels (FlashMLA) benefit community

   Comparison:
     - Previous sparse attention: Coarse-grained, quality loss
     - DSA: Fine-grained, no quality loss, production-ready

2. Thinking with Tools (Agentic Integration):
   Innovation: First model to integrate reasoning into tool execution

   Technical Achievement:
     - Massive synthetic dataset (1,800+ environments, 85K+ tasks)
     - Dual-mode tool-use (thinking + direct)
     - Scalable synthesis pipeline
     - reasoning_content API field

   Impact:
     - Improved compliance in complex workflows
     - Better generalization to new tools
     - Enables sophisticated autonomous agents
     - Sets new standard for agentic AI

   Comparison:
     - Previous tool-use: Direct action (reactive)
     - V3.2: Reasoning-enhanced (planned, adaptive)

3. Gold-Medal Reasoning (V3.2-Speciale):
   Innovation: First AI with gold medals across IMO, IOI, ICPC, CMO

   Technical Achievement:
     - RL training on competition problems
     - Outcome-based rewards (verifiable)
     - 96.0% AIME 2025, 99.2% HMMT
     - 2nd place ICPC World Finals, 10th place IOI

   Impact:
     - Demonstrates near-human expert-level reasoning
     - Validates RL for complex problem-solving
     - Benchmark for reasoning model development
     - Inspires competition-driven training

   Comparison:
     - GPT-5: 94.6% AIME (lower)
     - Gemini 3 Pro: 95.0% AIME (slightly lower)
     - DeepSeek-V3.2-Speciale: 96.0% AIME, gold medals (leading)

4. Continued Cost Efficiency Leadership:
   Innovation: Maintains DeepSeek's tradition of ultra-efficient training

   Technical Achievement:
     - V3: $5.57M for 671B parameters
     - V3.2 DSA: +$1M for 2-3× speedup
     - Total: ~$6-7M for frontier model

   Impact:
     - Democratizes frontier AI (accessible training costs)
     - Proves efficiency doesn't require sacrificing quality
     - Enables smaller orgs to train competitive models
     - Challenges "compute scaling is everything" narrative

   Comparison:
     - GPT-4: ~$100M training cost (15× more expensive)
     - Llama 3.1 405B: ~$10-20M (1.5-3× more expensive)
     - DeepSeek-V3.2: ~$7M (most cost-efficient at scale)

5. Open-Source Contributions:
   Innovation: MIT-licensed frontier model with open kernels

   Technical Achievement:
     - Full model weights (Hugging Face)
     - FlashMLA kernels (open source)
     - Inference code (GitHub)
     - Detailed technical reports

   Impact:
     - Community can deploy, modify, research
     - Accelerates sparse attention adoption
     - Educational value (understand DSA internals)
     - Challenges proprietary model dominance

   Comparison:
     - GPT-5, Gemini 3 Pro: Closed source
     - Llama 3.1: Open source, but no sparse attention
     - DeepSeek-V3.2: Open source + novel architecture

6. Auxiliary-Loss-Free Load Balancing:
   Innovation: MoE balancing without auxiliary loss (inherited from V3)

   Technical Achievement:
     - Dynamic bias adjustment per expert
     - Stable training without loss penalties
     - All tokens processed (no dropping)
     - Zero spikes, zero rollbacks

   Impact:
     - Better MoE training recipes
     - Enables larger-scale MoE (256 experts)
     - Inspires future MoE research
     - Proven at 671B scale

   Comparison:
     - Mixtral, Grok: Use auxiliary loss (quality penalty)
     - DeepSeek-V3/V3.2: Aux-loss-free (no penalty)

7. FP8 Training at Scale:
   Innovation: Stable FP8 training for 671B model (inherited from V3)

   Technical Achievement:
     - All matrix mults in FP8
     - Weights stored BF16/FP32
     - 2× speedup vs BF16
     - Zero quality degradation

   Impact:
     - 2× training cost reduction
     - Enables larger models on same hardware
     - Industry adoption of FP8 training
     - Validated at massive scale (14.8T tokens)

   Comparison:
     - Most models: BF16 training (standard)
     - DeepSeek-V3/V3.2: FP8 training (2× faster)
```

---

## Use Cases and Applications

DeepSeek-V3.2's unique capabilities enable a wide range of applications:

```yaml
Research and Academia:

  Mathematical Research:
    Capabilities:
      - Competition-level problem solving (IMO, CMO)
      - Theorem exploration and conjecturing
      - Proof assistance (not formal verification)
      - Problem generation for education

    Models: V3.2-Speciale (for research-grade rigor)
    Benefits:
      - Gold-medal reasoning
      - 99.2% HMMT performance
      - Long-form mathematical reasoning

    Example:
      "Explore generalizations of Fermat's Last Theorem
       for different exponents. Suggest potential conjectures."

  Competitive Programming Training:
    Capabilities:
      - IOI/ICPC-level problem solutions
      - Algorithm explanation and teaching
      - Code optimization suggestions
      - Test case generation

    Models: V3.2-Speciale (competition-grade) or V3.2 (practice)
    Benefits:
      - 10th place IOI, 2nd place ICPC performance
      - Understands advanced algorithms
      - Can implement DP, graphs, etc.

    Example:
      "Solve this IOI problem using segment trees:
       [problem statement]. Explain your approach."

  Literature Review and Synthesis:
    Capabilities:
      - Long document analysis (128K context)
      - Multi-paper synthesis
      - Research gap identification
      - Citation network exploration (with tools)

    Models: V3.2 (for tool-use + reasoning)
    Benefits:
      - DSA enables efficient long-context processing
      - Tool-use for paper search and retrieval
      - Reasoning for synthesis

    Example:
      "Analyze these 10 papers on sparse attention.
       Identify common themes, gaps, and future directions."

Software Development:

  Complex Codebase Understanding:
    Capabilities:
      - Full repository analysis (128K context)
      - Cross-file reasoning
      - Architecture documentation
      - Refactoring suggestions

    Models: V3.2 (for balanced reasoning + speed)
    Benefits:
      - Long context handles large codebases
      - Strong code reasoning (70.2% SWE-Multilingual)
      - Multi-language support

    Example:
      "Analyze this 50K line codebase. Document the
       architecture and suggest improvements."

  Automated Debugging and Fixing:
    Capabilities:
      - Root cause analysis
      - Bug fix implementation
      - Test case generation
      - Regression prevention

    Models: V3.2 (agentic tools) or V3.2-Speciale (complex bugs)
    Benefits:
      - Thinking + tools for systematic debugging
      - 73.1% SWE-Verified performance
      - Can reason about tricky bugs

    Example:
      "This test is failing intermittently. Debug the
       issue, propose a fix, and implement it."

  Code Review and Security Analysis:
    Capabilities:
      - Security vulnerability detection
      - Code quality assessment
      - Performance analysis
      - Best practice enforcement

    Models: V3.2-Speciale (thorough analysis) or V3.2 (faster)
    Benefits:
      - Deep reasoning for security issues
      - Understands complex exploits
      - Multi-step attack scenario analysis

    Example:
      "Review this auth code for security vulnerabilities.
       Consider SQL injection, XSS, and auth bypass."

Data Science and Analytics:

  Exploratory Data Analysis:
    Capabilities:
      - Automated EDA (with tool-use)
      - Pattern identification
      - Visualization generation
      - Hypothesis generation

    Models: V3.2 (thinking + tools for Python/pandas)
    Benefits:
      - Reasons about data distributions
      - Suggests relevant analyses
      - Generates interpretable insights

    Example:
      "Analyze this dataset. Identify patterns, outliers,
       and generate hypotheses for further investigation."

  Complex Statistical Analysis:
    Capabilities:
      - Advanced statistical methods
      - Experimental design
      - Results interpretation
      - Reporting

    Models: V3.2-Speciale (for rigorous statistical reasoning)
    Benefits:
      - Strong mathematical reasoning
      - Understands statistical theory
      - Can design complex experiments

    Example:
      "Design a multi-factor ANOVA study for this
       agricultural experiment. Explain the methodology."

Business and Enterprise:

  Legal Document Analysis:
    Capabilities:
      - Contract review (long context)
      - Clause extraction and comparison
      - Risk identification
      - Compliance checking

    Models: V3.2 (128K context handles full contracts)
    Benefits:
      - DSA enables efficient long-document processing
      - Reasoning about legal implications
      - Multi-document comparison

    Example:
      "Compare these 5 vendor contracts. Identify
       differences in liability, IP, and termination clauses."

  Financial Modeling and Analysis:
    Capabilities:
      - Complex financial calculations
      - Risk modeling
      - Scenario analysis
      - Report generation

    Models: V3.2-Speciale (for quantitative rigor)
    Benefits:
      - Strong mathematical reasoning
      - Can implement financial models
      - Understands probability and statistics

    Example:
      "Build a Monte Carlo simulation for this
       investment portfolio. Analyze risk exposure."

  Customer Support Automation:
    Capabilities:
      - Complex issue resolution
      - Multi-step troubleshooting (with tools)
      - Escalation triage
      - Knowledge base querying

    Models: V3.2 (for agentic tool-use)
    Benefits:
      - Thinking + tools for systematic troubleshooting
      - Can access documentation, run diagnostics
      - Handles complex multi-turn conversations

    Example:
      "User reports: 'App crashes on login.' Debug the
       issue using logs, suggest solution, and create ticket."

Education:

  Personalized Tutoring:
    Capabilities:
      - Competition math tutoring (IMO/AIME)
      - Programming instruction (IOI/ICPC)
      - Adaptive difficulty
      - Step-by-step explanations

    Models: V3.2-Speciale (for teaching complex topics)
    Benefits:
      - Gold-medal reasoning
      - Can explain at various levels
      - Generates practice problems

    Example:
      "Teach me about dynamic programming. Start with
       basics, then progress to advanced (ICPC-level)."

  Assignment Grading and Feedback:
    Capabilities:
      - Automated grading (math, coding)
      - Detailed feedback generation
      - Plagiarism detection (with tools)
      - Personalized improvement suggestions

    Models: V3.2 (for balanced grading + feedback)
    Benefits:
      - Understands correctness rigorously
      - Provides constructive feedback
      - Scales to large class sizes

    Example:
      "Grade these 100 student solutions to AIME problem 15.
       Provide feedback on errors and suggest improvements."

Creative and Specialized:

  Interactive Fiction and Games:
    Capabilities:
      - Long-form storytelling (128K context)
      - Coherent narrative across many turns
      - Character consistency
      - Branching storylines

    Models: V3.2 (for creative + consistent generation)
    Benefits:
      - Long context maintains story coherence
      - Reasoning about character motivations
      - Can handle complex narratives

    Example:
      "You are a dungeon master. Run a D&D campaign
       with complex NPCs, multi-session story arcs."

  Scientific Simulation and Modeling:
    Capabilities:
      - Simulation design
      - Parameter optimization
      - Results interpretation
      - Hypothesis generation

    Models: V3.2-Speciale (for scientific rigor)
    Benefits:
      - Strong mathematical and scientific reasoning
      - Understands physical laws
      - Can implement simulations (with tools)

    Example:
      "Design a simulation for protein folding dynamics.
       Suggest appropriate force fields and parameters."
```

---

## Limitations and Considerations

Despite its strengths, DeepSeek-V3.2 has several limitations:

```yaml
Model Limitations:

1. Not Optimized for Casual Chat:
   Issue:
     - V3.2-Speciale: Longer reasoning traces (slower)
     - Designed for complex problems, not simple questions
     - May be "overkill" for basic queries

   Example:
     Query: "What's the weather like?"
     Response: [Long reasoning trace about weather patterns]
     Better: Use smaller, faster models for simple tasks

   Recommendation:
     - V3.2-Speciale: Competition problems, research
     - V3.2 Standard: General use, balanced
     - Smaller models: Simple queries, casual chat

2. No Vision Capabilities:
   Issue:
     - Language-only model
     - Cannot process images, videos, audio
     - See DeepSeek-VL2 for multimodal needs

   Comparison:
     - GPT-4o, Gemini 3 Pro: Multimodal
     - DeepSeek-V3.2: Language-only

   Workaround:
     - Use OCR tools to extract text from images
     - Combine with vision models in pipeline

3. Reasoning Quality vs Speed Tradeoff:
   Issue:
     - V3.2-Speciale: High quality, but slower
     - Requires longer generation trajectories
     - Not suitable for latency-critical applications

   Example:
     V3.2: ~500 tokens reasoning → 100 tokens answer (fast)
     V3.2-Speciale: ~5000 tokens reasoning → 100 tokens (slow)

   Guidance:
     - Real-time apps: Use V3.2 standard
     - Batch processing: Use V3.2-Speciale
     - Consider quality vs latency needs

4. Tool-Use Limitations (V3.2-Speciale):
   Issue:
     - V3.2-Speciale doesn't support tool calling
     - Reasoning-exclusive focus
     - Cannot execute code, search web, etc.

   Workaround:
     - Use V3.2 standard for agentic tasks
     - Use V3.2-Speciale only for pure reasoning

5. Context Window Constraints:
   Issue:
     - API: 64K limit (though model supports 128K)
     - Self-hosted: 128K max (YaRN scaling)
     - DSA optimized for k=2048 selected tokens

   Example:
     Analyzing 200K token document: Requires chunking

   Comparison:
     - Gemini 3 Pro: 1M+ context
     - Claude 3.5: 200K context
     - DeepSeek-V3.2: 128K context (64K API)

Deployment Challenges:

6. High Hardware Requirements:
   Issue:
     - Minimum: 8 × H100 (80GB) for BF16
     - Expensive hardware (~$200K+ for 8×H100)
     - High power consumption (3-4 kW)

   Cost:
     - H100 rental: ~$2.50/hour each
     - 8×H100: ~$20/hour = $14,400/month
     - Not feasible for small teams/individuals

   Alternatives:
     - Use API (much cheaper for low volume)
     - Use smaller models (Qwen, Llama)
     - Quantization (INT8/INT4) to reduce GPUs

7. Complex Setup:
   Issue:
     - DSA requires custom kernels (DeepGEMM, FlashMLA)
     - Multi-GPU configuration (tensor parallelism)
     - Need for high-performance interconnects
     - Limited framework support (vLLM, SGLang only)

   Comparison:
     - Standard models: Easy deployment (Transformers)
     - DeepSeek-V3.2: Requires expertise

   Recommendation:
     - Start with API for prototyping
     - Self-host only at scale (millions of tokens/day)

8. Limited Community Ecosystem:
   Issue:
     - Newer architecture (DSA, MLA)
     - Fewer tutorials, guides, examples
     - Limited third-party tool integration
     - Smaller community vs Llama, GPT

   Comparison:
     - Llama: Massive ecosystem (llama.cpp, ollama, etc.)
     - DeepSeek-V3.2: Growing, but nascent

   Impact:
     - Slower troubleshooting
     - Fewer pre-built integrations
     - More DIY required

Benchmark and Performance:

9. Not Leading on All Benchmarks:
   Issue:
     - Gemini 3 Pro: Better on coding (LiveCodeBench, Terminal Bench)
     - Claude 3.5: Better on general knowledge (MMLU-Pro)
     - GPT-5: Competitive on many tasks

   Context:
     - DeepSeek-V3.2 leads on reasoning (AIME, HMMT, IMO/IOI)
     - But not universally best

   Guidance:
     - Choose based on task requirements
     - Reasoning-heavy: DeepSeek-V3.2-Speciale
     - General coding: Gemini 3 Pro
     - General knowledge: Claude 3.5

10. DSA Benefit Context-Dependent:
    Issue:
      - DSA speedup: Significant on long contexts (64K-128K)
      - At short contexts (<8K): Minimal benefit, overhead
      - Memory savings: 30-40% across all lengths

    Example:
      4K context: 1.0× speedup (overhead = benefit)
      128K context: 3.0× speedup (major benefit)

    Implication:
      - If most queries are short: DSA doesn't help much
      - If most queries are long: DSA provides huge gains

    Guidance:
      - Evaluate your typical context length
      - DSA shines for document analysis, codebase understanding
      - Less impactful for short chat messages

Ethical and Safety:

11. Potential for Misuse:
    Issue:
      - Competition-level problem solving
      - Could enable cheating in competitions
      - Advanced reasoning for malicious purposes

    Mitigations:
      - DeepSeek monitors API for abuse
      - Rate limits prevent mass cheating
      - Community norms against misuse

    Responsibility:
      - Users should use responsibly
      - Don't submit AI solutions to competitions as own work
      - Follow academic integrity guidelines

12. Hallucination Risk:
    Issue:
      - Despite 97.1% SimpleQA, still hallucinates occasionally
      - Reasoning traces can appear confident when wrong
      - Long reasoning can obscure errors

    Mitigation:
      - Verify critical outputs
      - Use verification tools (symbolic math, code execution)
      - Don't blindly trust reasoning traces

    Example:
      Math problem: Check answer with calculator
      Code: Run tests to verify correctness

13. Bias and Fairness:
    Issue:
      - Inherits biases from training data
      - Primarily English/Chinese optimized
      - May underperform on non-represented demographics

    Mitigation:
      - DeepSeek includes safety training (details not disclosed)
      - Monitor outputs for biased responses
      - Use with awareness of limitations

    Guidance:
      - Don't use for high-stakes decisions without human review
      - Be aware of cultural and linguistic biases
      - Test on your specific use case and demographics

Licensing and Access:

14. Temporary V3.2-Speciale Access:
    Issue:
      - V3.2-Speciale API expires December 15, 2025
      - Uncertain future availability
      - May require migration to alternative

    Options After Expiration:
      1. Integrated into V3.2 as "reasoning mode"
      2. Separate permanent API endpoint
      3. Open-source release on Hugging Face
      4. Discontinued (less likely given performance)

    Recommendation:
      - Don't build production systems solely on Speciale
      - Plan for migration to V3.2 standard if needed
      - Wait for permanent release announcement

15. Chinese Datacenter Concerns:
    Issue:
      - Trained in China (Hangzhou)
      - Potential geopolitical considerations
      - Data sovereignty for API users

    Considerations:
      - API data sent to Chinese servers
      - Self-hosting avoids this (MIT license allows)
      - Evaluate based on data sensitivity

    Guidance:
      - For sensitive data: Self-host or use regional alternative
      - For general use: API is fine for most cases
      - Review data privacy policies
```

---

## Resources and Links

### Official Resources

```yaml
Model Downloads:
  Hugging Face:
    - V3.2: https://huggingface.co/deepseek-ai/DeepSeek-V3.2
    - V3.2-Speciale: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale
    - V3.2-Exp: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
    - V3.2-Exp-Base: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp-Base

  Collection:
    - V3.2 Collection: https://huggingface.co/collections/deepseek-ai/deepseek-v32

GitHub Repositories:
  - V3.2-Exp: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
  - V3 (Base Architecture): https://github.com/deepseek-ai/DeepSeek-V3
  - FlashMLA Kernels: https://github.com/deepseek-ai/FlashMLA

API Documentation:
  - API Docs: https://api-docs.deepseek.com/
  - V3.2 Release: https://api-docs.deepseek.com/news/news251201
  - V3.2-Exp Announcement: https://api-docs.deepseek.com/news/news250929
  - Function Calling Guide: https://api-docs.deepseek.com/guides/function_calling

Official Website:
  - DeepSeek AI: https://www.deepseek.com/
  - Chat Interface: https://chat.deepseek.com/
  - API Console: https://platform.deepseek.com/

Research Papers:
  - DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
  - DeepSeek-V3.2-Exp Blog (vLLM): https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html
  - DSA Technical Analysis: Multiple community blogs (see below)
```

### Third-Party Resources

```yaml
Inference Frameworks:
  vLLM:
    - GitHub: https://github.com/vllm-project/vllm
    - Documentation: https://docs.vllm.ai/
    - Installation: pip install vllm>=0.6.0

  SGLang:
    - GitHub: https://github.com/sgl-project/sglang
    - Documentation: https://sgl-project.github.io/
    - DSA Blog: https://lmsys.org/blog/2025-09-29-deepseek-V32/

  TensorRT-LLM:
    - GitHub: https://github.com/NVIDIA/TensorRT-LLM
    - Documentation: https://nvidia.github.io/TensorRT-LLM/

API Aggregators:
  OpenRouter:
    - Website: https://openrouter.ai/deepseek
    - Model ID: deepseek/deepseek-v3.2

  Together AI:
    - Website: https://www.together.ai/
    - Model: DeepSeek-V3.2-Exp

Community Tutorials and Guides:
  DSA Sparse Attention Explained:
    - Skywork AI: https://skywork.ai/blog/sparse-attention-deepseek-3-2-explained/
    - DEV Community: https://dev.to/czmilo/deepseek-v32-exp-complete-analysis
    - Analytics Vidhya: https://www.analyticsvidhya.com/blog/2025/09/deepseek-v3-2-exp/

  MLA (Multi-Head Latent Attention):
    - Planet Banatt: https://planetbanatt.net/articles/mla.html
    - Medium (Shirley Li): https://medium.com/data-science/deepseek-v3-explained-1-multi-head-latent-attention
    - Chris McCormick: https://mccormickml.com/2025/02/12/the-inner-workings-of-deep-seek-v3/

  Deployment Guides:
    - Red Hat Developer (vLLM): https://developers.redhat.com/articles/2025/10/03/deepseek-v32-exp-vllm-day-0
    - DataCrunch (SGLang): https://datacrunch.io/blog/deepseek-sglang-multi-head-latent-attention

Benchmarks and Leaderboards:
  - LLM Stats (V3.2-Speciale): https://llm-stats.com/models/deepseek-v3.2-speciale
  - Hugging Face Open LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
  - AlpacaEval: https://tatsu-lab.github.io/alpaca_eval/
```

### Community and Support

```yaml
Discussion Forums:
  - r/LocalLLaMA: https://www.reddit.com/r/LocalLLaMA/ (DeepSeek discussions)
  - r/MachineLearning: https://www.reddit.com/r/MachineLearning/
  - Hugging Face Forums: https://discuss.huggingface.co/

Social Media:
  - DeepSeek Twitter: (Not official, community-run)
  - #DeepSeek on Twitter/X
  - LinkedIn: DeepSeek AI company page

Discord Servers:
  - LocalLLaMA Discord (community-run)
  - vLLM Discord (for deployment questions)
  - SGLang Discord

Technical Support:
  - GitHub Issues (DeepSeek-V3.2-Exp repo)
  - Hugging Face Model Discussions
  - API Support: support@deepseek.com (for API customers)
```

---

## Conclusion

DeepSeek-V3.2 represents a significant milestone in efficient AI reasoning, demonstrating that frontier-level performance can be achieved through architectural innovation rather than pure computational scaling. Released on December 1, 2025, the model introduces **DeepSeek Sparse Attention (DSA)**—the first production-ready fine-grained sparse attention system—delivering **2-3× speedup** and **30-40% memory savings** on long-context tasks with zero quality degradation.

**Key Achievements:**
- **State-of-the-art reasoning**: V3.2-Speciale achieves **96.0% on AIME 2025** (surpassing GPT-5's 94.6% and Gemini 3 Pro's 95.0%) and **99.2% on HMMT February 2025** (highest among all reasoning models)
- **Gold-medal competition performance**: First AI to achieve gold medals across **IMO, IOI (10th place), ICPC World Finals (2nd place), and CMO 2025**
- **Revolutionary agentic capabilities**: First DeepSeek model integrating **thinking directly into tool execution**, trained on **1,800+ environments** and **85,000+ complex instructions**
- **Unprecedented efficiency**: DSA reduces long-context inference complexity from O(L²) to O(Lk), enabling practical 128K context processing

**Architectural Innovations:**
- **DSA two-stage pipeline**: Lightning indexer rapidly scores all tokens in FP8, fine-grained selection attends to top-2048, achieving massive speedup without quality loss
- **MLA + DSA synergy**: Multi-Head Latent Attention compresses KV cache 93.3% (512-dim vs 16K-dim), DSA operates on compressed cache for double efficiency multiplicative benefit
- **Auxiliary-loss-free MoE balancing**: Dynamic bias adjustment achieves expert load balance without performance degradation, enabling stable 671B parameter training
- **FP8 mixed precision**: All matrix multiplications in FP8 for 2× training speedup, maintained throughout pre-training with zero spikes or rollbacks

**Impact and Accessibility:**
- **Cost leadership**: ~$7M total training cost (V3 base + DSA + agentic) vs $100M+ for GPT-4, democratizing frontier AI development
- **Open-source commitment**: MIT-licensed model weights, FlashMLA kernels, and inference code enable community deployment and research
- **API affordability**: $0.27 per million input tokens vs $2.50 for GPT-5 (~90% cheaper), with 50% reduced inference costs from DSA efficiency
- **Self-hosting viable**: 8×H100 minimum for BF16, optimized kernels (DeepGEMM, FlashMLA) enable production deployment

**Ideal For:**
- Competition mathematics and programming (IMO, IOI, ICPC, CMO level)
- Complex reasoning and problem-solving (AIME, GPQA, advanced logic)
- Agentic workflows and automation (multi-step planning, tool-use, debugging)
- Long-context analysis (legal contracts, codebases, research papers up to 128K tokens)
- Multilingual applications (strong performance across 40+ languages)
- Cost-sensitive deployments (10-20× cheaper training, 90% cheaper API vs competitors)

**Considerations:**
- Requires 8×H100 minimum for self-hosting (~$200K hardware, $15K/month rental)
- V3.2-Speciale: Slower inference (longer reasoning traces), no tool-use support, temporary API (expires Dec 15, 2025)
- No vision capabilities (language-only, see DeepSeek-VL2 for multimodal needs)
- DSA benefit scales with context length (minimal <8K, maximum at 128K)
- Gemini 3 Pro leads on some coding benchmarks (LiveCodeBench, Terminal Bench)

**Looking Forward:**
DeepSeek-V3.2 validates the thesis that **efficiency and quality are not mutually exclusive**. By combining sparse MoE (671B total, 37B active), sparse attention (DSA), and compressed KV cache (MLA), the model achieves frontier performance at a fraction of the computational cost. The open-source release with production-ready kernels accelerates community adoption of sparse attention, while gold-medal reasoning performance sets a new benchmark for AI capabilities.

For researchers, DeepSeek-V3.2 provides a reference implementation of fine-grained sparse attention at scale. For developers, it offers a cost-effective alternative to proprietary models with state-of-the-art reasoning. For organizations, it demonstrates that frontier AI is accessible without hundred-million-dollar budgets.

**Get Started:**
- **Quick API Access**: https://api.deepseek.com/ ($0.27/M input tokens, $1.10/M output)
- **Self-Hosting (vLLM)**: `pip install vllm>=0.6.0` + `vllm serve deepseek-ai/DeepSeek-V3.2-Exp`
- **Model Weights**: https://huggingface.co/collections/deepseek-ai/deepseek-v32
- **Technical Deep Dive**: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- **Community Support**: r/LocalLLaMA, vLLM Discord, Hugging Face Forums

---

*Document last updated: December 2, 2025*
*DeepSeek-V3.2 release date: December 1, 2025*
*Model version: V3.2 (general), V3.2-Speciale (reasoning), V3.2-Exp (experimental)*
*License: MIT License*
