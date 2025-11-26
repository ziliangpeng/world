# Zamba: Zyphra's 7B Mamba-Attention Hybrid Model

## 1. Overview

Zamba is a 7-billion parameter language model developed by Zyphra, released in May 2024, that pioneered a novel hybrid architecture combining State Space Model (SSM) Mamba blocks with shared transformer self-attention layers. The model demonstrates that a carefully designed hybrid approach can achieve competitive performance with pure transformer models while maintaining significantly better inference efficiency and memory usage.

Zyphra, headquartered in San Francisco, focuses on next-generation neural network architectures, long-term memory, and on-device AI deployment. The company's strategic vision emphasizes bringing powerful language models to consumer devices through improved architectural efficiency rather than simply scaling up parameters.

**Key Metrics:**
- **Parameters:** 7 billion (dense model, no mixture-of-experts)
- **Release Date:** May 2024 (Zamba-7B-v1)
- **Follow-up Release:** October 2024 (Zamba2 series with improved architecture)
- **License:** Apache 2.0 (fully open-source)
- **Training Scale:** Zamba-7B trained on 1 trillion tokens in 30 days using 128 H100 GPUs with a team of 7 researchers
- **Company Vision:** On-device personalization through efficient smaller models

The original Zamba was trained with remarkable efficiency, achieving competitive results with less than half the training data required by comparable models like LLaMA-2 7B and OLMo-7B, demonstrating Zyphra's focus on parameter efficiency and training compute optimization.

## 2. Shared Attention Innovation

### What is Shared Attention?

Traditional transformer models apply full self-attention at every layer, which creates quadratic complexity in sequence length and requires maintaining a Key-Value (KV) cache proportional to the context length times the number of layers. This makes transformers memory-intensive and slow for long sequences.

Shared attention is a novel mechanism where instead of having unique attention parameters at each layer, a single set of attention parameters is reused across multiple invocations in the network. In Zamba-7B, one attention block with shared weights is applied 13 times throughout the 80-layer network (every 6 Mamba blocks), creating a parameter and memory-efficient design that still captures the benefits of attention-based retrieval and in-context learning.

### How Shared Attention Works

**Core Principle:** The attention block shares both its query-key-value projection matrices and the attention output projection across multiple invocations, reducing parameter overhead while allowing the network to perform attention operations at strategic depths.

**Design Pattern:**
1. Input embeddings and residual stream are concatenated before feeding into the shared attention block
2. The shared attention module performs standard multi-head self-attention operations
3. Attention output is projected back to match the residual stream dimensionality
4. This shared block is applied every N (typically 6) Mamba blocks throughout the network

**Why Concatenation Matters:** Concatenating the original input embeddings with the residual stream before attention improves information flow and feature diversity, helping maintain signal quality across the deep network despite parameter sharing.

### Advantages of Shared Attention

1. **Dramatic KV Cache Reduction:** Since only the shared attention invocations require KV caches (not every layer), Zamba achieves up to 6x KV cache reduction compared to pure transformers
2. **Minimal Parameter Overhead:** A single 7B model gains attention capacity at roughly 2-3% additional parameter cost
3. **Maintained In-Context Learning:** Pure SSMs struggle with few-shot learning and context switching; shared attention restores this capability
4. **Fixed Memory Footprint:** SSM components have constant-size hidden states, and shared attention reduces attention memory requirements

### Comparison to Other Attention Approaches

- **Per-Layer Attention (Transformers):** Each layer has unique attention parameters; expensive in memory and parameters
- **Sparse Attention:** Reduces attention scope but still expensive; requires specialized kernels
- **Shared Attention (Zamba):** Reuses identical parameters; minimal overhead while preserving full attention connectivity

The shared attention approach is conceptualized as "one attention layer is all you need," capturing the essential insight that strategic, sparsely-applied attention can provide sufficient long-range dependency modeling without the cost of universal attention.

## 3. Mamba + Shared Attention Architecture

### Architectural Composition

Zamba-7B employs a specific layer arrangement optimized through experimentation:

**Base Architecture:**
- **Total Layers:** 80 layers
- **Hidden Dimension:** 3,712
- **Attention Spacing:** Shared attention block applied every 6 Mamba blocks (creating 13 total attention applications)
- **Layer Pattern:** 6 consecutive Mamba blocks → Shared Attention + MLP → Residual Connection → Repeat

### Detailed Layer Arrangement

```
Layer 1-6:    [Mamba, Mamba, Mamba, Mamba, Mamba, Mamba]
Layer 7-13:   [SharedAttn+MLP] ← attention invocation 1
Layer 14-19:  [Mamba, Mamba, Mamba, Mamba, Mamba, Mamba]
Layer 20-26:  [SharedAttn+MLP] ← attention invocation 2
...
(Pattern repeats 13 times with shared weights)
```

### Mamba Block Details

**Mamba (Selective State Space Model):**
- Mamba is an efficient sequence model that replaces attention with a learnable state matrix that processes sequences recurrently
- Uses selective scanning to adaptively focus on relevant context based on input
- Achieves O(N) complexity in sequence length with competitive performance on language tasks
- Maintains constant-size hidden state during inference, eliminating KV cache requirements

**Why Mamba in Zamba:**
- Linear scaling with sequence length (constant memory)
- Efficient inference kernels with high GPU utilization
- Faster token generation, especially for long sequences
- Complementary to attention's strengths in retrieval and pattern matching

### Zamba2 Architectural Improvements

Zamba2 series (released October 2024) enhanced the hybrid design with:

1. **Mamba2 Backbone:** Replaced Mamba1 with Mamba2 blocks
   - ~4x throughput improvement over Mamba1
   - Better parameter-to-performance scaling
   - Larger state size with improved expressivity

2. **Dual Shared Attention Blocks:** Instead of a single shared attention
   - Two alternating shared attention blocks (A and B) with distinct parameters
   - ABAB interleaving pattern: Mamba block → AttentionA → Mamba block → AttentionB → Repeat
   - Allows different depth-specific specialization

3. **LoRA-Enhanced Shared Blocks:**
   - Low-Rank Adapter (LoRA) projections applied to each shared MLP and attention block
   - Adds minimal parameters (few percent) while allowing depth-specific specialization
   - Each invocation of the shared block at different depths can specialize through its LoRA matrices

4. **Rotary Position Embeddings:** Added RoPE to shared attention blocks for better position awareness

### Information Flow Diagram

```
Input Sequence
    ↓
Embedding Layer
    ↓
6x Mamba Blocks (linear complexity, constant cache)
    ↓
[Input Embeddings + Residual] → Shared Attention A → MLP
    ↓
6x Mamba Blocks
    ↓
[Input Embeddings + Residual] → Shared Attention B → MLP
    ↓
(Pattern repeats with BOTH attention blocks in alternating pattern)
    ↓
Output Layer Norm + LM Head
```

## 4. Model Variants

### Zamba-7B-v1 (Original)
- **Parameters:** 7.0B
- **Training Tokens:** 1.0T (Phase 1) + 50B (Phase 2 annealing) = ~1T total
- **Attention Strategy:** Single shared attention block
- **SSM Type:** Mamba1
- **Release:** May 2024
- **Key Feature:** Demonstrated feasibility of shared attention approach
- **Status:** Superseded by Zamba2 but remains available

### Zamba2-1.2B
- **Parameters:** 1.2B
- **Training Tokens:** 3.0T
- **Attention Strategy:** Dual alternating shared attention (ABAB)
- **SSM Type:** Mamba2
- **Release:** October 2024
- **Key Feature:** Smallest Zamba2 model, excellent for edge deployment
- **Use Case:** On-device inference with minimal memory footprint

### Zamba2-2.7B
- **Parameters:** 2.7B
- **Training Tokens:** 3.0T
- **Attention Strategy:** Dual alternating shared attention (ABAB)
- **SSM Type:** Mamba2
- **Release:** October 2024
- **Key Feature:** Sweet spot for efficiency/capability tradeoff
- **Use Case:** Mobile, embedded, and edge computing; constrained environments
- **MMLU Score:** 55.97 (5-shot)

### Zamba2-7B / Zamba2-7.4B
- **Parameters:** 7.4B (called 7B colloquially)
- **Training Tokens:** 2.0T (due to compute constraints; intended 3.0T)
- **Attention Strategy:** Dual alternating shared attention (ABAB)
- **SSM Type:** Mamba2
- **Release:** October 2024
- **Key Feature:** State-of-the-art 7B performance
- **Use Case:** General purpose, server, and device deployment
- **MMLU Score:** 67.2 (5-shot)

### Instruction-Tuned Variants

All models have instruction-tuned variants available:
- Zamba-7B-v1 (base pretrain and instruct versions)
- Zamba2-1.2B-Instruct
- Zamba2-2.7B-Instruct / Zamba2-2.7B-Instruct-v2
- Zamba2-7B-Instruct

**Instruction Tuning Details:**
- Supervised Fine-Tuning (SFT) on 4M instruction-following samples
- Multiple Direct Preference Optimization (DPO) iterations using 200k+ preference pairs
- Context extended to 17,000 tokens through NTK-aware scaling

## 5. Architecture Details

### Mamba Block Implementation

**Selective State Space Model Operation:**

Each Mamba block processes the input sequence through a parameterized continuous function approximated as a discrete recurrence:

```
h[t] = A * h[t-1] + B * x[t]
y[t] = C * h[t]
```

Where:
- `h[t]` is the hidden state (fixed dimension, independent of sequence length)
- `x[t]` is the input at position t
- `A`, `B`, `C` are learnable matrices
- **Selectivity:** Inputs `A`, `B`, `C` depend on input `x[t]`, allowing adaptive focus

**Key Properties:**
- Computation happens in parallel during training (efficient)
- Inference is sequential but constant-memory (no KV cache)
- Forward pass: O(BLN) where B=batch, L=length, N=state_size
- Inference: O(1) memory in sequence length (constant hidden state)

### Shared Attention Block Implementation

**Multi-Head Self-Attention with Shared Parameters:**

```
Query = W_q * x
Key = W_k * x
Value = W_v * x
Attention = softmax(Query @ Key^T / sqrt(d_k)) @ Value
Output = W_o * Attention
```

All weight matrices (W_q, W_k, W_v, W_o) are **shared across all 13 invocations** in Zamba-7B (or dual-shared in Zamba2).

**Attention Modification for Zamba2:**
```
x_residual = concatenate(original_embeddings, residual_stream)
attn_output = MultiHeadAttention(x_residual, params_shared)
lora_output = attn_output @ W_lora_a @ W_lora_b  // Low-rank specialization
output = layer_norm(residual_stream + lora_output)
```

### Gating and Normalization

- **Pre-norm Architecture:** Layer normalization applied before each block
- **Residual Connections:** Each block output added to input via residual connection
- **Gate Matrices:** Mamba blocks use gating mechanisms to control information flow
- **No Bias Terms:** Modern efficient transformers often omit biases for speed

### Position Embeddings

- **Zamba-7B:** Rotary Position Embeddings (RoPE) applied in Mamba blocks
- **Zamba2:** Rotary Position Embeddings added to shared attention blocks (improvement)
- **RoPE Benefit:** Maintains relative position information; scales to longer contexts
- **Context Extension:** NTK-aware scaling allows 17k context in instruction-tuned variants

## 6. Training

### Zamba-7B-v1 Training

**Phase 1: Pretraining**
- **Duration:** ~3.5 weeks (approximately 22-25 days of compute)
- **Token Count:** 950 billion tokens
- **Hardware:** 128 H100 GPUs
- **Team Size:** 7 researchers
- **Data Composition:**
  - 62% RefinedWeb (filtered web text)
  - 13% C4 (diverse web corpus)
  - 15% The Pile (curated datasets)
  - 10% Other sources
- **Learning Rate Schedule:** Cosine decay from 1.5×10⁻⁴ to 7.5×10⁻⁵
- **Batch Size:** Optimized for H100 hardware
- **Loss Monitoring:** Standard next-token prediction loss

**Phase 2: Annealing**
- **Duration:** ~1 week
- **Total Tokens:** 50 billion high-quality tokens
- **Data Mix:**
  - 60% original Phase 1 data (for stability)
  - 40% new high-quality instruction and synthetic data
- **Learning Rate Schedule:** Exponential decay with rapid annealing
- **Purpose:** Fine-tune model for better downstream task performance without catastrophic forgetting

**Total Training Time:** 30 days from start to finish
**Total Compute Budget:** ~$200,000 (extremely efficient compared to LLaMA, Mistral, Phi)

### Zamba2 Training

**Improved Training Dataset: Zyda-2**

Zyphra developed Zyda-2, a proprietary pretraining dataset addressing limitations in public web data:

- **Size:** 5 trillion tokens total (3T used for Zamba2 series)
- **Composition:**
  - FineWeb-Edu (heavily filtered for educational quality)
  - DCLM (Diverse Corpus of Large Language Models)
  - Aggressive deduplication and quality filtering
- **Benefit:** Specialized for factual knowledge and reasoning (MMLU/ARC improvement)

**Phase 1 & 2 Pretraining:**

| Model | Phase 1 Tokens | Training Time | Hardware |
|-------|---|---|---|
| Zamba2-1.2B | 3.0T | ~20 days | 16x H100 nodes |
| Zamba2-2.7B | 3.0T | ~40 days | 16x H100 nodes |
| Zamba2-7.4B | 2.0T | ~60 days | 16x H100 nodes |

- **Phase 2 Annealing:** 100B tokens (50B each epoch) of high-quality instruction data
- **Infrastructure:** 16 nodes, 8 H100 per node (128 H100 total equivalent)
- **Framework:** Megatron-LM-based distributed training

**Post-Training Pipeline:**

1. **Supervised Fine-Tuning:** 4M instruction samples
2. **Direct Preference Optimization:** 200k+ preference pairs across multiple iterations
3. **Context Extension:** NTK-aware scaling to 17k tokens

### Training Strategy Highlights

- **Efficiency Focus:** Zamba2-7B trained on 2-3T tokens vs Llama3-8B's 15T tokens
- **Data Quality Over Quantity:** Aggressive filtering of training data
- **Two-Phase Approach:** Pretraining + high-quality annealing improves convergence
- **Parameter Sharing:** Reduces training compute for shared attention components
- **Mamba Efficiency:** SSM forward pass faster than attention during training

## 7. Performance Benchmarks

### Zamba-7B-v1 Results

**General Language Understanding:**

| Benchmark | Zamba-7B | LLaMA-2 7B | OLMo-7B | Mistral-7B |
|-----------|----------|-----------|---------|-----------|
| PIQA | ~79% | ~77% | ~76% | ~80% |
| HellaSwag | ~75% | ~73% | ~70% | ~76% |
| Winogrande | ~65% | ~62% | ~61% | ~67% |
| ARC-Easy | ~83% | ~80% | ~78% | ~85% |
| BoolQ | ~80% | ~78% | ~76% | ~82% |

**Strength Areas:**
- Reasoning benchmarks (HellaSwag, Winogrande) benefit from shared attention
- General language understanding competitive despite fewer training tokens
- Long-context tasks show Mamba's efficiency advantage

**Notable Achievement:** Approaches Mistral-7B and Gemma-7B performance despite:
- Training on significantly fewer tokens (1T vs 4-5T+)
- Using open web data only
- Smaller training team and budget

### Zamba2 Benchmarks

**MMLU (5-shot) Results:**

| Model | MMLU Score | Training Tokens | Notes |
|-------|-----------|-----------------|-------|
| Zamba2-1.2B | 43.1% | 3.0T | Outperforms comparable SSM models |
| Zamba2-2.7B | 55.97% | 3.0T | State-of-art 2.7B, competitive with Llama3.2 |
| Zamba2-7B | 67.2% | 2.0T | Competitive with Llama3-8B (65.17%) |
| Llama3.2-8B | ~64% | ~15T | Comparable 8B baseline |
| Gemma-7B | ~63% | ~6T | Comparable 7B baseline |
| Mistral-7B | ~62.3% | ~?T | Popular 7B baseline |

**Key Insight:** Zamba2-7B achieves 67.2% MMLU with 2T tokens, while Llama3-8B needs 15T tokens to reach 65.17%. This represents exceptional training efficiency.

**Comprehensive Benchmark Coverage:**

Zamba2 series evaluated across multiple domains:

- **Common Sense Reasoning:** PIQA, Winogrande, ARC-Easy, ARC-Challenge
- **Knowledge Recall:** MMLU, OpenBookQA
- **Reading Comprehension:** BoolQ, Lambada
- **Aggregate Metrics:** Arithmetic mean across 7-8 benchmarks

**Zamba2-2.7B Performance:**
- Consistently outperforms models of similar size
- Competitive with models 2-3x larger in some benchmarks
- Excellent scaling with training tokens

### Inference Efficiency Benchmarks

**Zamba2-7B vs Llama3-8B:**

| Metric | Zamba2-7B | Llama3-8B | Improvement |
|--------|-----------|-----------|-------------|
| Time-to-First-Token | -25% to -50% | baseline | **25-50% faster** |
| Tokens-Per-Second | +20% | baseline | **20% better throughput** |
| Peak KV Cache Memory | ~6x smaller | baseline | **6x reduction** |
| Memory per 1M tokens | Significantly lower | baseline | Better long-context |

**Context Length Performance:**

- **Needle-in-Haystack Testing:** Zamba2-7B-Instruct maintains high accuracy up to 16k context
- **Performance Degradation:** Sharp drop-off around 18k tokens (similar to other 7B models)
- **Extended Context:** NTK scaling enables stable inference at 17k tokens

### Inference Speed Characteristics

**Why Zamba is Fast:**

1. **No Quadratic Attention:** Mamba blocks process in O(N) without attention QK multiplication
2. **Minimal KV Cache:** Only 13 attention invocations vs 80 in pure transformer (93% reduction)
3. **Efficient Kernels:** Mamba2 blocks have ~4x throughput of transformer blocks
4. **Constant Memory:** SSM state size fixed regardless of sequence length

**Practical Speeds:**

- **Batch Size 1 (latency-bound):** Zamba significantly faster due to less memory bandwidth
- **Batch Size > 1 (compute-bound):** Still faster due to Mamba2 throughput advantage
- **Long Sequences:** Increasing advantage as context length grows (linear vs quadratic memory)

## 8. Memory Efficiency

### KV Cache Reduction

**Traditional Transformers:**
- Each layer maintains Key-Value cache: cache_size = batch × seq_len × num_heads × head_dim
- For 80-layer model: cache = batch × seq_len × heads × 96 bytes × 80
- With 4M context: KV cache = 4M × 80 × ~100MB = **tens of GB**

**Zamba Approach:**
- Mamba blocks: NO KV cache (constant-size hidden state)
- Shared Attention: Only 13 invocations (not 80)
- Effective reduction: 13/80 = ~16% of transformer KV cache
- With 4M context: Zamba KV cache = 4M × 13 × ~100MB = **~5GB** (vs 50GB+ for transformer)

**Practical Impact:**
- Enable longer context on consumer hardware
- Reduce server memory requirements for batch processing
- Enable multi-user serving on shared hardware
- Better mobile/edge device support

### State Cache vs KV Cache

**Mamba Hidden State:**
- Fixed size: hidden_dim × batch_size (e.g., 3712 × 1 = ~15KB per position-independent)
- Independent of sequence length
- Only one copy needed during generation

**Attention KV Cache:**
- Grows with: num_heads × head_dim × batch_size × sequence_length
- Zamba: 13 invocations × per-layer cache
- Pure transformer: 80 invocations × per-layer cache

**Memory Comparison at 128K Context:**

```
Llama3-8B (80 transformer layers):
KV Cache = 128K × 80 × 32 heads × 128 bytes × batch_size
         ≈ 40GB per batch (single item)

Zamba2-7B (78 Mamba + 2 Attention layers):
KV Cache = 128K × 2 × 32 heads × 128 bytes × batch_size
         ≈ 1GB per batch (single item)
         + Mamba constant state (~100KB)
```

### Parameter Memory

**Model Weights:**
- Zamba-7B: 7B params × 2 bytes (fp16) = **14GB**
- Zamba2-7B: 7.4B params × 2 bytes (fp16) = **~15GB**

**Comparison:**
- Llama3-8B: 8B × 2 = **16GB**
- Mistral-7B: 7B × 2 = **14GB**

Zamba parameter count competitive with peer models; advantage comes from KV cache reduction.

### Energy Efficiency

**Why Zamba Uses Less Energy:**
1. Fewer memory access patterns (smaller activations)
2. More efficient kernels (Mamba2 > transformer blocks)
3. Fewer total FLOPs due to sparse attention applications
4. Better cache locality (constant state vs growing KV)

**On-Device Benefits:**
- Extended battery life for mobile inference
- Lower thermal footprint (less power consumption)
- Practical for continuous AI services on phones/tablets

## 9. Shared Attention Deep Dive

### Why Share Attention Instead of Distribute It?

**Hypothesis:** Instead of having 80 layers each with their own attention (expensive), apply shared attention strategically at key depths.

**Rationale:**

1. **Empirical Finding:** Attention patterns often show consistency across depths
   - Similar tokens attend to similar context across layers
   - Not every layer needs unique attention parameters
   - Significant information captured by shared pattern

2. **Parameter Efficiency:**
   - Shared attention at 1/6th spacing: saves ~93% of attention parameters
   - Trade: Shared parameters vs distributed parameters
   - Reality: Shared parameters actually more beneficial than expected

3. **Information Flow:**
   - Early layers: Attend to surface patterns (shared suffices)
   - Middle layers: Attend to semantic patterns (shared suffices)
   - Late layers: Attend to task-specific patterns (alternating dual attention helps in Zamba2)

### Comparison: Per-Layer vs Shared vs Sparse

**Per-Layer Attention (Pure Transformers):**
```
Pros:
- Maximum expressivity and flexibility
- Can learn unique patterns at each depth
- Industry standard, well-studied

Cons:
- Expensive parameters (80× attention blocks)
- Expensive KV cache (quadratic in context)
- Slow inference due to attention QK multiplication
```

**Shared Attention (Zamba Strategy):**
```
Pros:
- Minimal parameter overhead (~2-3%)
- Massive KV cache reduction (6-13×)
- Comparable performance to per-layer
- Simpler architecture, faster implementation

Cons:
- Less flexibility than per-layer
- Assumes similar patterns across depths (mostly true)
- Novel approach, less established
```

**Sparse Attention (Alternative Approach):**
```
Pros:
- Reduces attention computation
- Can maintain unique parameters per layer
- Studied in literature

Cons:
- Still requires per-layer unique parameters
- Specialized kernels needed
- Communication overhead for sparsity
- Slower in practice on dense hardware
```

### Depth-Specific Specialization (Zamba2 Innovation)

**Problem with Pure Sharing:** All layers use identical attention weights, potentially limiting expressivity at different network depths.

**Zamba2 Solution: LoRA Enhancement**

```
Output = W_lora_a @ W_lora_b @ shared_attention_output

Where:
- W_lora_a: (hidden_dim × rank)
- W_lora_b: (rank × hidden_dim)
- rank: typically 8-16 (minimal overhead)
- Shared block: applies at all depths
- LoRA matrices: different per depth
```

**Benefits:**
- Each invocation of shared block can specialize through its own LoRA matrices
- Minimal parameter addition (few percent)
- Allows network to learn depth-appropriate transformations
- Maintains computational efficiency of shared structure

**Empirical Result:** Adding LoRA projections improved Zamba2 performance measurably while keeping parameter overhead minimal.

### Input Embedding Concatenation Strategy

**Zamba Modification:**
```
Input: residual_stream
Modified Input: [residual_stream, original_embeddings]
```

**Why This Helps:**
1. **Information Preservation:** Original embeddings carry word-level information unmodified by layers
2. **Path Length Reduction:** Shared attention receives richer input representation
3. **Gradient Flow:** Additional paths for backpropagation through shared component
4. **Positional Information:** Original embeddings maintain stronger positional signal

**Empirical Finding:** Simple concatenation improved performance, suggesting that shared attention benefits from diverse input features.

### Frequency of Application

**Design Choice: Every 6 Mamba Blocks**

Why not more? Why not less?

**If Every 3 Blocks (More Frequent):**
- Pro: More attention applications help performance
- Con: More KV cache entries, memory increases
- Trade: Lost some memory efficiency advantage

**If Every 12 Blocks (Less Frequent):**
- Pro: Minimal KV cache requirements
- Con: Performance degradation without sufficient attention
- Trade: In-context learning and retrieval suffer

**Every 6 Blocks (Chosen):**
- Empirically optimal balance
- ~13-14 total attention invocations in 80-layer model
- Maintains ~6× KV cache reduction
- Sufficient attention for in-context learning

**Zamba2 Pattern (ABAB with Dual Blocks):**
- Two alternating attention blocks with different LoRA specializations
- More frequent attention applications (better performance)
- Still maintains 6× KV cache reduction via SSM backbone

## 10. Comparison with Other Hybrids

### Zamba vs Jamba (AI21 Labs)

**Jamba Architecture:**
- 52B total parameters with Mixture-of-Experts (MoE)
- Only 12B active parameters during inference
- Interleaved Transformer and Mamba layers at 1:7 ratio
- 4 dedicated attention layers throughout the model
- 256K context length
- Released: March 2024

**Zamba Architecture:**
- 7B dense parameters (no MoE)
- All parameters active during inference
- Mamba backbone with strategically placed shared attention
- 1-2 shared attention blocks applied multiple times
- Standard context (16k for extended versions)
- Released: May 2024

**Comparison Table:**

| Aspect | Jamba | Zamba |
|--------|-------|-------|
| **Total Parameters** | 52B | 7B |
| **Active Parameters** | 12B | 7B |
| **Architecture** | Interleaved blocks | Shared blocks |
| **Attention Strategy** | 4 unique layers in 1:7 Mamba ratio | 1-2 shared blocks every 6 Mamba |
| **MoE** | Yes (major design) | No (dense model) |
| **KV Cache Reduction** | Moderate (4 layers) | High (6-13×) |
| **Model Complexity** | High (routing, load balancing) | Lower (shared parameters) |
| **Training Efficiency** | High (MoE capacity) | Very High (small team, quick) |
| **Inference Memory** | Lower than transformer (but not minimal) | Minimal (6× reduction) |
| **Max Context** | 256K tokens | 16K-17K tokens |
| **Use Case** | Large-scale, high-compute | Efficient, on-device |

**Key Philosophical Difference:**
- **Jamba:** Uses MoE to scale capacity while keeping active parameters reasonable (12B acts like larger model)
- **Zamba:** Uses parameter sharing to minimize overhead while keeping full model active

### Zamba vs Hymba (NVIDIA)

**Hymba Overview:**
- Mamba2 backbone with distributed attention (similar concept to Zamba)
- Attention layers interspersed differently throughout architecture
- Focus on long-context performance
- Released: Around same timeframe as Zamba2

**Similarities:**
- Both hybrid Mamba + attention approaches
- Both focus on efficiency improvements
- Both use SSM for backbone efficiency

**Differences:**
- Hymba may have more attention layers (more compute, better quality)
- Zamba emphasizes minimal attention (maximum efficiency)
- Different architectural placement strategies
- Zamba has more mature ecosystem and higher-quality training data

### Zamba vs Granite (IBM)

**Granite Overview:**
- Dense transformer models with various sizes
- Focuses on enterprise use cases
- Strong on instruction-following and safety
- No hybrid architecture; pure transformer

**Comparison:**
- **Granite:** Traditional transformer, strong but computationally expensive
- **Zamba:** Hybrid SSM, efficient but newer architecture
- **Granite Use Case:** Enterprise, safety-critical, established approach
- **Zamba Use Case:** Edge, efficiency-critical, innovative approach

### Zamba vs Mamba (Pure SSM)

**Pure Mamba (No Attention):**
- Excellent efficiency and speed
- Struggles with in-context learning (few-shot capability)
- Struggles with algorithmic tasks
- No proven ability to retrieve specific context

**Zamba (Mamba + Shared Attention):**
- Maintains SSM efficiency advantage
- Adds attention capability for retrieval and context learning
- Bridges gap between SSM efficiency and transformer capability
- Better empirical performance on diverse benchmarks

**Trade-off Summary:**
```
Pure Mamba:    [Efficiency] ████████████░░░░░░ [Capability]
Zamba:         [Efficiency] ██████░░░░░░░░░░░░ [Capability]
Pure Transform:[Efficiency] ░░░░░░░░░░░░░░░░░░ [Capability]
```

## 11. Use Cases

### When to Choose Zamba

**1. On-Device Inference**
- **Why:** 6× KV cache reduction, smaller memory footprint
- **Example:** Running LLM on iPhone, iPad, or edge device
- **Target:** Mobile AI applications, personal AI assistants
- **Benefit:** Extended battery life, faster responses, privacy-preserving

**2. Long-Context Applications**
- **Why:** Constant memory complexity of SSM backbone
- **Example:** Analyzing long documents, multi-turn conversations, code repositories
- **Target:** Document analysis, context-aware systems
- **Benefit:** No memory explosion at 16K+ context

**3. Cost-Optimized Inference (Server)**
- **Why:** Faster token generation, lower GPU memory, smaller batches possible
- **Example:** Running 1000+ concurrent inference requests
- **Target:** Public APIs, multi-tenant serving
- **Benefit:** Higher throughput per GPU, lower infrastructure costs

**4. Real-Time Latency-Sensitive Applications**
- **Why:** Faster time-to-first-token (25-50% improvement)
- **Example:** Interactive chatbots, real-time translation, live transcription
- **Target:** Consumer-facing interactive AI
- **Benefit:** Perceived responsiveness, better UX

**5. Education and Research**
- **Why:** Efficient training (1-2T tokens), reproducible, transparent
- **Example:** Studying hybrid architectures, running experiments
- **Target:** ML researchers, educators
- **Benefit:** Reproducible results, accessible research, educational value

**6. Fine-Tuning and Adaptation**
- **Why:** Shared attention structure allows efficient adaptation
- **Example:** Domain-specific fine-tuning, few-shot learning
- **Target:** Enterprise customization
- **Benefit:** Faster adaptation, lower fine-tuning costs

### When to Choose Alternative Models

**Choose Mistral-7B if:**
- You prioritize raw performance on benchmarks
- You need established, battle-tested architecture
- Hardware already optimized for transformer inference
- You need maximum model flexibility

**Choose Llama3-8B if:**
- You need strong general-purpose performance
- You prioritize community and ecosystem support
- You're willing to accept higher memory usage
- You need long-context out-of-the-box (attention handles this better)

**Choose Jamba if:**
- You need a larger effective model (12B active parameters)
- You're training at large scale with significant compute
- You need very long context (256K tokens)
- You want MoE's routing benefits for diverse tasks

**Choose Pure Mamba if:**
- Efficiency is absolutely paramount
- You're only doing language generation (no retrieval-based tasks)
- You need maximum inference speed
- You can tolerate slight performance degradation

**Choose Granite if:**
- You need enterprise-grade safety and governance
- You require provenance and training transparency
- You prioritize established vendor support
- You're building regulated systems

## 12. Implementation

### Frameworks and Libraries

**Official Support:**

1. **PyTorch (Official)**
   - GitHub: `Zyphra/Zamba2` (Zamba2 implementation)
   - Direct PyTorch code for inference and training
   - Supports both base models and fine-tuned variants
   - Status: Primary implementation, actively maintained

2. **Hugging Face Transformers**
   - Zamba models integrated into HF ecosystem
   - Access via `transformers` library version 4.36+
   - Model cards available for all variants
   - Example:
   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM

   model = AutoModelForCausalLM.from_pretrained("Zyphra/Zamba2-7B")
   tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-7B")
   ```

**Third-Party Support:**

1. **vLLM** (OpenAI-compatible inference)
   - High-throughput serving framework
   - Efficient batching and attention computation
   - Supports quantization and serving APIs

2. **Ollama** (Local deployment)
   - Single command download and run
   - No GPU/CUDA setup required
   - Perfect for local experimentation
   - Example: `ollama pull zamba2`

3. **LM Studio** (GUI-based local inference)
   - Graphical interface for Zamba models
   - No command-line needed
   - Cross-platform (Windows, Mac, Linux)

4. **LLaMA.cpp** (CPU inference)
   - Quantized model support
   - Can run on CPU (slow but possible)
   - Good for testing without GPU

### Model Availability

**Primary Repositories:**

1. **Hugging Face Hub**
   - Zyphra organization official models
   - Base models (pretrained only)
   - Instruct models (instruction-tuned)
   - Quantized versions from community

2. **GitHub**
   - `Zyphra/Zamba2` - Official implementation
   - Training code, inference scripts
   - Integration examples

3. **Model Zoo:**
   - Zamba-7B-v1
   - Zamba-7B-v1-phase1 (checkpoint during training)
   - Zamba2-1.2B / Zamba2-1.2B-Instruct
   - Zamba2-2.7B / Zamba2-2.7B-Instruct / Zamba2-2.7B-Instruct-v2
   - Zamba2-7B / Zamba2-7B-Instruct

### Hardware Requirements

**Inference (FP16):**

| Model | GPU VRAM | CPU RAM | Notes |
|-------|----------|---------|-------|
| Zamba2-1.2B | 3GB | 8GB | Mobile/edge capable |
| Zamba2-2.7B | 6GB | 16GB | Laptop/workstation |
| Zamba2-7B | 15GB | 32GB | Server/high-end GPU |

**Quantized Versions (GGUF/GPTQ):**

| Model | Quantization | VRAM | Use Case |
|-------|---|---|---|
| Zamba2-7B | 4-bit | 4-5GB | Consumer GPU, gaming card |
| Zamba2-7B | 8-bit | 8-10GB | Laptop with discrete GPU |

**Training (Zamba2-7B from scratch):**
- 16× H100 GPUs (128GB VRAM each) or equivalent
- 60 days for 2T tokens at this scale
- Megatron-LM or vLLM training framework
- Not practical for individual researchers

**Fine-Tuning:**
- Zamba2-7B: 2× A100 GPUs (80GB) with LoRA
- Zamba2-2.7B: 1× A100 GPU with LoRA
- Using parameter-efficient methods (LoRA, QLoRA)

### Quantization Support

**Available Formats:**

1. **GGUF** (CPU and mobile friendly)
   - Created by llama.cpp community
   - Various quantization levels (Q2, Q4, Q5, Q8)
   - Easy to run locally

2. **GPTQ** (GPU optimized)
   - Efficient GPU inference
   - 4-bit quantization typical
   - Less memory, slightly faster

3. **AWQ** (Alternative quantization)
   - Activation-aware quantization
   - Balanced accuracy and speed

4. **Native FP16/BF16**
   - Full precision, best quality
   - Requires full VRAM

### Inference Code Examples

**Basic Hugging Face Usage:**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Zyphra/Zamba2-7B",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Zyphra/Zamba2-7B")

# Prepare input
messages = [
    {"role": "user", "content": "Hello, how are you?"}
]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

# Generate
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

**Using vLLM for Efficient Serving:**
```python
from vllm import LLM, SamplingParams

# Initialize LLM
llm = LLM(
    model="Zyphra/Zamba2-7B",
    max_model_len=4096,  # Adjust based on GPU memory
    gpu_memory_utilization=0.9
)

# Set sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256
)

# Run inference
prompt = "What is the capital of France?"
result = llm.generate(prompt, sampling_params)
print(result[0].outputs[0].text)
```

## 13. Licensing

### Zamba License Structure

**Base Models (Zamba-7B-v1, Zamba2 series):**
- **License:** Apache 2.0
- **Type:** Permissive open-source
- **Permissions:**
  - Commercial use (free)
  - Modification
  - Distribution
  - Private use
  - Sublicensing
- **Requirements:**
  - Include license and copyright notice
  - State significant changes
  - Include source code changes

**Instruction-Tuned Variants:**
- **License:** Apache 2.0 (same as base)
- **Applies To:** All instruct model variants
- **Example:** Zamba2-7B-Instruct is fully Apache 2.0

### Derivative Works

**Training Data:**
- Base models trained on open web data
- No special licensing restrictions on generated outputs
- Models can be used freely for downstream tasks

**Fine-Tuning:**
- You can fine-tune Zamba models commercially
- Derivative models inherit Apache 2.0 requirements
- Share improvements back to community (optional but appreciated)

### Commercial Usage

**Allowed Without Restriction:**
- API services offering Zamba completions
- Embedded LLMs in applications
- Model distillation and compression
- Serving Zamba in commercial products
- Merging with other models

**Best Practices:**
- Include license in repository/product documentation
- Mention original authors and Zyphra
- Document any modifications or fine-tuning

### Comparison with Other Models

| Model | License | Commercial | Modification | Distribution |
|-------|---------|-----------|-------------|--------------|
| Zamba2-7B | Apache 2.0 | ✓ Free | ✓ | ✓ |
| Llama3-8B | Llama 2 Community | ✓ Free* | ✓ | ✓ |
| Mistral-7B | Apache 2.0 | ✓ Free | ✓ | ✓ |
| Gemma-7B | Gemma | ✓ Free* | ✓ | ✓ |

*Free for commercial use with specific guidelines

## 14. Zyphra's Approach and Research Focus

### Company Vision

Zyphra's mission is to build "your personal AI" through efficient, on-device language models. Founded on the principle that smaller, well-trained models are better than massive ones when compute and memory are constrained.

**Key Insight:** "Seven billion is actually the perfect scale to be able to run it locally on almost all devices you have" — Beren Millidge, Zyphra Co-founder

### Strategic Research Focus

**1. Efficient Architectures**
- State Space Models (SSMs) as transformers alternative
- Hybrid approaches combining SSM + attention
- Parameter sharing and depth-wise specialization
- Focus on inference efficiency over training scale

**2. Data Quality Over Quantity**
- Developed Zyda dataset (5T tokens) with aggressive filtering
- FineWeb-Edu and DCLM components
- Better performance per token than raw web data
- Educational quality focus for factual knowledge

**3. On-Device Deployment**
- Memory-efficient architectures
- Low latency inference
- Energy-conscious design
- Privacy-preserving (no cloud needed)

**4. Practical ML Engineering**
- Published Zyphra Training Cookbook (open-source guide)
- Emphasis on reproducibility and accessibility
- Small teams achieving large results (7 people, 30 days for Zamba-7B)
- Transparent about training, data, methodology

### Research Output

**Published Work:**
1. Zamba paper (arXiv 2405.16712) - Initial shared attention architecture
2. Zamba2 technical report (arXiv 2411.15242) - Improvements and scaling
3. Zyda-2 dataset paper - Data quality and composition
4. Zyphra Training Cookbook - Practical how-to guide

**Open Source Contributions:**
- Full model weights (base and instruct variants)
- Training code and scripts
- Zyda-2 dataset (5T tokens)
- Reference implementations

### Funding and Partnership

**Series A Funding (October 2025):**
- Investors: IBM, AMD
- Strategic focus: Multimodal agent systems
- Extended research: Long-term memory and RL

**Why IBM/AMD Investment:**
- Interest in efficient enterprise models
- AMD GPU optimization (alternative to NVIDIA)
- Potential integration into enterprise platforms

### Competitive Positioning

**Vs Large AI Labs:**
- Zyphra: Small team, efficient models, transparent research
- OpenAI/Google: Massive scale, closed models, resource-intensive
- Meta: Large research team, open models, diverse architectures

**Zyphra's Advantages:**
- Innovative hybrid architecture
- Exceptional training efficiency
- Practical focus on deployment
- Transparent research and methodology
- Permissive licensing

**Zyphra's Challenges:**
- Limited compute resources vs AI labs
- Smaller team and research budget
- Newer company (less established)
- Less funding than competitors

### Philosophy and Values

1. **Openness:** All models, weights, data released open-source
2. **Efficiency:** Maximum capability with minimum resources
3. **Accessibility:** Enable AI for all (developers, researchers, individuals)
4. **Reproducibility:** Transparent training, published papers, open code
5. **Pragmatism:** Focus on what works, not academic prestige

## 15. Future Directions

### Planned Research

**1. Larger Zamba Models**
- Zamba3 series potentially including 13B, 30B+ variants
- Extended Zamba2 success to larger scales
- Better performance/efficiency trade-offs

**2. Multimodal Extensions**
- Vision + Language Zamba variants
- Image understanding with efficient architecture
- Planned as part of Series A research

**3. Improved Attention Mechanisms**
- Refinements to shared attention design
- Potentially 3+ shared blocks in future
- Optimal frequency of attention application

**4. Long-Context Specialization**
- Extending Zamba2 to 32K, 64K, 100K+ tokens
- Optimizations for long-document processing
- Improved needle-in-haystack performance

### Data and Training Improvements

**1. Zyda Dataset Evolution**
- Zyda-3 or larger variants
- More aggressive filtering for specific domains
- Synthetic data generation for improved reasoning

**2. Post-Training Optimization**
- Advanced preference optimization (DPO, IPO, etc.)
- Reasoning-focused annealing
- Safety and alignment improvements

**3. Distillation Research**
- Creating smaller Zamba variants through distillation
- Knowledge transfer from Zamba2-7B to 1.2B
- Sub-billion parameter efficient models

### Hardware and Deployment

**1. Custom Hardware Optimization**
- Potential AMD optimization (given partnership)
- CPU optimizations for edge deployment
- Mobile-specific variants

**2. Inference Framework Development**
- Optimized CUDA kernels for Zamba2
- CPU/NPU support
- Specialized serving frameworks

**3. Edge AI Integration**
- Partnership opportunities for on-device deployment
- Integration with mobile OS (iOS, Android)
- Embedded system support

### Longer-Term Vision

**1. Personalized AI Agents**
- Zamba as backbone for personal AI assistants
- Long-term memory integration
- Continual learning capabilities

**2. Decentralized AI**
- Run models locally to preserve privacy
- Federated learning opportunities
- User control over data

**3. Efficiency As Standard**
- Shift industry paradigm toward efficient models
- Challenge "bigger is better" mentality
- Prove small models can be highly capable

## 16. Comparison Tables

### Architecture Comparison

| Feature | Zamba-7B-v1 | Zamba2-7B | Llama3-8B | Mistral-7B |
|---------|-----------|----------|-----------|-----------|
| **Total Parameters** | 7.0B | 7.4B | 8.0B | 7.3B |
| **Architecture Type** | Hybrid SSM-Attn | Hybrid SSM-Attn | Pure Transformer | Pure Transformer |
| **Backbone** | Mamba1 | Mamba2 | Attention | Attention |
| **Attention Blocks** | 1 shared | 2 alternating | 80 unique | 80 unique |
| **KV Cache (16K ctx)** | ~2.5GB | ~2.5GB | ~16GB | ~15GB |
| **Hidden Dimension** | 3,712 | 3,712 | 4,096 | 4,096 |
| **Training Tokens** | 1.0T | 2.0T | 15.0T | ~4T |

### Performance Comparison

| Benchmark | Zamba2-7B | Zamba-7B-v1 | Llama3-8B | Mistral-7B | Gemma-7B |
|-----------|----------|-----------|-----------|-----------|----------|
| **MMLU (5-shot)** | 67.2% | 57.72% | 65.17% | 62.30% | 63.0% |
| **ARC-Challenge** | High | Good | High | High | Good |
| **HellaSwag** | ~78% | ~75% | ~78% | ~77% | ~76% |
| **Winogrande** | ~65% | ~63% | ~66% | ~67% | ~65% |
| **PIQA** | ~80% | ~79% | ~81% | ~81% | ~80% |

### Inference Efficiency Comparison

| Metric | Zamba2-7B | Llama3-8B | Mistral-7B | Zamba2-2.7B |
|--------|----------|-----------|-----------|-----------|
| **Time-to-First-Token** | 25-50% faster | baseline | Similar | Fastest |
| **Tokens-Per-Second** | +20% | baseline | +10-15% | +15% |
| **KV Cache (16K)** | ~2.5GB | ~16GB | ~15GB | ~1GB |
| **Memory per 1M Tokens** | ~5GB | ~30GB | ~28GB | ~2GB |
| **Peak Speedup** | 30-50% | baseline | Moderate | 30-50% |

### Model Variants Overview

| Model Name | Parameters | Training Data | Focus | MMLU Score |
|-----------|-----------|---|---|---|
| Zamba-7B-v1 | 7.0B | 1T tokens | Proof-of-concept | 57.72% |
| Zamba2-1.2B | 1.2B | 3T tokens | Mobile/Edge | 43.1% |
| Zamba2-2.7B | 2.7B | 3T tokens | Balanced | 55.97% |
| Zamba2-7B | 7.4B | 2T tokens | State-of-art | 67.2% |
| Llama3-8B | 8.0B | 15T tokens | Industry standard | 65.17% |

### Training Efficiency Comparison

| Model | Parameters | Training Tokens | Training Time | Team Size | Cost Estimate |
|-------|-----------|---|---|---|---|
| Zamba-7B-v1 | 7B | 1T | 30 days | 7 people | ~$200K |
| Zamba2-7B | 7.4B | 2T | 60 days | ~10 people | ~$400K |
| Llama3-8B | 8B | 15T | ~400 days | Large team | ~$5M+ |
| Mistral-7B | 7.3B | ~4T | ~100 days | Small team | ~$1M |

## 17. Sources

### Official Zyphra Resources

- [Zyphra Official Website](https://www.zyphra.com/)
- [Zamba Announcement](https://www.zyphra.com/post/zamba)
- [Zamba2 Announcement](https://www.zyphra.com/post/zamba2-7b)
- [Zyphra GitHub Repository](https://github.com/Zyphra)
- [Zamba2 PyTorch Implementation](https://github.com/Zyphra/Zamba2)
- [Zyphra Training Cookbook](https://github.com/Zyphra/zcookbook)

### Academic Papers

- [Zamba: A Compact 7B SSM Hybrid Model (arXiv 2405.16712)](https://arxiv.org/abs/2405.16712)
- [The Zamba2 Suite: Technical Report (arXiv 2411.15242)](https://arxiv.org/abs/2411.15242)

### Hugging Face Models and Documentation

- [Zyphra Organization on Hugging Face](https://huggingface.co/Zyphra)
- [Zamba-7B-v1 Model Card](https://huggingface.co/Zyphra/Zamba-7B-v1)
- [Zamba2-7B Model Card](https://huggingface.co/Zyphra/Zamba2-7B)
- [Zamba2-2.7B Model Card](https://huggingface.co/Zyphra/Zamba2-2.7B)
- [Zamba2-1.2B Model Card](https://huggingface.co/Zyphra/Zamba2-1.2B)

### News and Coverage

- [Zamba Announcement at VentureBeat](https://venturebeat.com/ai/zyphra-releases-zamba-an-ssm-hybrid-foundation-model-to-bring-ai-to-more-devices)
- [Zamba on Hacker News](https://news.ycombinator.com/search?q=zamba)
- [Zamba2-7B on Hacker News](https://news.ycombinator.com/item?id=41842975)
- [Zyphra Series A Funding](https://tracxn.com/d/companies/zyphra/)

### Technical Articles and Analysis

- [Medium: Zamba Architecture Overview](https://medium.com/@bnjmn_marie/zamba-a-new-llm-architecture-with-state-space-model-layers-sharing-self-attention-ab8e24bc47da)
- [Medium: Zamba2-1.2B Analysis](https://medium.com/@bnjmn_marie/zamba2-1-2b-a-smaller-hybrid-ssm-transformer-391a4b1587e0)
- [Medium: Zamba2-7B by Zyphra Review](https://vivekupadhyay1.medium.com/zamba2-7b-by-zyphra-1a1bedd30ca8)
- [The Sequence: Understanding Zamba](https://thesequence.substack.com/p/edge-439-ssms-with-attention-understanding)
- [Mamba and Transformer Hybrids Overview](https://n1o.github.io/posts/ssm-transformer-hybrids-guide/)

### Related Research

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.08782)
- [Jamba: Hybrid Transformer-Mamba (AI21 Labs)](https://arxiv.org/abs/2403.19887)
- [Zyda-2 Dataset Research](https://dnyuz.com/2024/10/17/zyphras-new-zyda-2-dataset-lets-enterprises-train-small-llms-with-high-accuracy/)

### Implementation Resources

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [vLLM Inference Framework](https://github.com/lm-sys/vLLM)
- [Ollama Local LLM Running](https://ollama.ai/)
- [LM Studio GUI Interface](https://lmstudio.ai/)

---

**Document Version:** 1.0
**Last Updated:** November 2024
**Word Count:** 980 lines
**Author Context:** Comprehensive technical documentation created for reference and educational purposes
