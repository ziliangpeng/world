# Attention Mechanisms

Attention mechanisms are the core component of transformer-based LLMs. The evolution from Multi-Head Attention to more efficient variants has been crucial for scaling models and enabling long context windows.

## Multi-Head Attention (MHA) - Original (2017)

### From "Attention Is All You Need"

**Structure**:
```python
# For each attention head:
Q = W_q @ X  # Query projection
K = W_k @ X  # Key projection
V = W_v @ X  # Value projection

# Attention computation
scores = (Q @ K.T) / sqrt(d_k)
attention = softmax(scores)
output = attention @ V

# Concatenate all heads and project
multi_head_output = concat(head_1, ..., head_n) @ W_o
```

**Parameters**:
- Separate Q, K, V projections for each head
- Each head: `3 × d_model × d_head` parameters
- Total: `n_heads × 3 × d_model × d_head` parameters

**Memory (KV Cache)**:
```
KV cache size = 2 × batch × seq_len × n_heads × head_dim
```

### Advantages
1. **Parallel attention heads**: Each head can learn different patterns
2. **Rich representations**: Multiple attention patterns simultaneously
3. **Proven architecture**: Well-understood, reliable

### Disadvantages
1. **High memory bandwidth**: Large KV cache
2. **Inference cost**: Memory-bound at inference
3. **Scaling challenges**: Cache grows with sequence length and heads

### Used In
- GPT-3 (175B)
- Original BERT
- Early transformers
- Gemma 1 7B
- Many foundational models

---

## Multi-Query Attention (MQA) - 2019

### Key Innovation: Shared Keys and Values

**Structure**:
```python
# All query heads share single K, V
Q = [W_q1 @ X, W_q2 @ X, ..., W_qn @ X]  # n different query projections
K = W_k @ X  # Single shared key projection
V = W_v @ X  # Single shared value projection

# Each head computes attention with shared K, V
for each head_i:
    scores = (Q_i @ K.T) / sqrt(d_k)
    attention = softmax(scores)
    output_i = attention @ V
```

**Memory (KV Cache)**:
```
KV cache size = 2 × batch × seq_len × 1 × head_dim
```

**Reduction**: `n_heads × smaller` than MHA

### Advantages
1. **Dramatic KV cache reduction**: Only one K, V pair
2. **Faster inference**: Less memory bandwidth
3. **Same query diversity**: Still multiple query heads

### Disadvantages
1. **Quality degradation**: Especially at larger scales
2. **Less expressive**: All queries share same K, V
3. **Training-inference gap**: Optimization challenges

### Used In
- Falcon 40B, 180B
- Gemma 1 2B
- Some Google models
- Generally superseded by GQA

---

## Grouped-Query Attention (GQA) - 2023

### Sweet Spot: Between MHA and MQA

**Paper**: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (May 2023)

**Structure**:
```python
# Group query heads, share K,V within groups
n_groups = n_heads // group_size

# For each group:
Q_group = [W_q1 @ X, ..., W_q_group_size @ X]  # Multiple queries per group
K_group = W_k_group @ X  # Shared within group
V_group = W_v_group @ X  # Shared within group

# Example: 32 heads, 4 KV heads = 8 queries per KV pair
```

**Memory (KV Cache)**:
```
KV cache size = 2 × batch × seq_len × n_kv_heads × head_dim

n_kv_heads = n_heads / group_size
```

**Example (Llama 3 8B)**:
- 32 query heads
- 8 KV heads
- 4 queries share each K, V pair
- 4x reduction vs MHA
- Much better quality than MQA

### Advantages
1. **Near-MHA quality**: Minimal degradation
2. **Near-MQA speed**: Significant speedup over MHA
3. **Flexible trade-off**: Can tune group size
4. **Rapid adoption**: Now industry standard

### Disadvantages
1. **More complex**: Than pure MHA or MQA
2. **Tuning required**: Optimal group size varies
3. **Still has cache**: Though much smaller than MHA

### GQA Configurations in Practice

| Model | Total Heads | KV Heads | Ratio | KV Cache Reduction |
|-------|-------------|----------|-------|-------------------|
| Llama 3 8B | 32 | 8 | 4:1 | 4x vs MHA |
| Llama 3 70B | 64 | 8 | 8:1 | 8x vs MHA |
| Qwen 2.5 7B | 28 | 4 | 7:1 | 7x vs MHA |
| Phi-3-small | 32 | 8 | 4:1 | 4x vs MHA |
| Gemma 2 (all) | Varies | Varies | 4-8:1 | 4-8x vs MHA |

### Adoption Timeline
- **May 2023**: GQA paper published
- **July 2023**: Llama 2 70B uses GQA
- **2024**: Became standard for new models
  - Llama 3 (all sizes)
  - Qwen series
  - Gemma 2
  - Phi-3
  - Most new open models

### Why GQA Won

1. **Best trade-off**: ~90-95% MHA quality, ~70-80% MQA speed
2. **Easy adoption**: Can convert MHA checkpoints
3. **Flexible**: Tune group size for model size
4. **Validated quickly**: Multiple successful deployments

---

## Multi-head Latent Attention (MLA) - 2024

### DeepSeek's Innovation

**Paper**: DeepSeek-V2 (May 2024)

**Key Idea**: Compress K, V into low-rank latent representation

**Structure**:
```python
# Compress to latent space
kv_latent = W_down @ X  # [batch, seq, latent_dim]

# Expand to multi-head K, V
K = W_k_up @ kv_latent  # [batch, seq, n_heads * head_dim]
V = W_v_up @ kv_latent  # [batch, seq, n_heads * head_dim]

# Standard multi-head attention
Q = W_q @ X  # Full query projection
attention = MultiHeadAttention(Q, K, V)
```

**Memory (KV Cache)**:
```
KV cache size = 2 × batch × seq_len × latent_dim

latent_dim << n_heads × head_dim
```

**Typical Reduction**: 5-10x smaller than MHA

### Advantages
1. **Better than MHA**: Superior quality in practice
2. **Smaller cache**: Comparable to or better than GQA
3. **Learned compression**: Optimized latent representation
4. **Cross-head information**: Latent space enables sharing

### Disadvantages
1. **More complex**: Additional projections
2. **Newer**: Less battle-tested than GQA
3. **Training cost**: Slightly higher than GQA

### Used In
- DeepSeek-V2 (236B)
- DeepSeek-V3 (671B)
- Validated at massive scale

### Why MLA is Superior

**Low-Rank Compression Benefits**:
1. **Forced generalization**: Can't memorize, must compress
2. **Information sharing**: Latent space shared across heads
3. **Regularization**: Bottleneck prevents overfitting
4. **Efficiency**: Smaller cache, faster inference

**Empirical Results**:
- Lower perplexity than MHA on same model
- Better downstream task performance
- More efficient at all sequence lengths

### Future Potential

MLA could become next standard:
- Better quality than GQA
- Similar or better efficiency
- Validated at 671B scale
- Simple to implement

---

## FlashAttention - Memory Efficiency (2022-2024)

### Not a New Attention Mechanism

FlashAttention is an **implementation optimization**, not a new architecture.

### FlashAttention 1 (2022)

**Key Innovation**: IO-aware algorithm with tiling

**Standard Attention**:
```
1. Compute Q @ K.T → HBM (high bandwidth memory)
2. Load from HBM, apply softmax → HBM
3. Load from HBM, multiply by V → HBM
Multiple slow HBM accesses
```

**FlashAttention**:
```
1. Tile Q, K, V
2. Compute attention in SRAM (on-chip)
3. Avoid HBM round trips
4. Fuse operations
```

**Benefits**:
- Linear memory: O(n) vs O(n²)
- 10X memory savings at 2K length
- 20X memory savings at 4K length
- Faster training and inference

### FlashAttention-2 (2024)

**Improvements**:
- Better work partitioning
- Reduced non-matmul FLOPs
- Parallelized across thread blocks
- **2x speedup** over FlashAttention-1
- 50-73% of theoretical max FLOPs (A100)

### FlashAttention-3 (2024)

**Optimized for Hopper GPUs (H100)**:
- 1.5-2.0x faster than FlashAttention-2 (FP16)
- Up to 740 TFLOPS (75% H100 utilization)
- **FP8 support**: Close to 1.2 PFLOPS
- 2.6x lower error than baseline FP8

**Impact**:
- Enabled context expansion: 2-4K → 128K-1M+ tokens
- Made long-context models practical
- Essential for modern LLMs

### FlashAttention + ALiBi (2024)

**ALiBi in FlashAttention v2.4**:
- 4-5x speedup for ALiBi-based models
- Makes ALiBi practical at scale
- Combines best of both approaches

### Used Everywhere

FlashAttention is nearly universal:
- Llama series
- Qwen series
- Most modern LLMs
- Training and inference

---

## Attention Mechanism Evolution

### Timeline

```
2017: Multi-Head Attention (MHA)
      ↓
2019: Multi-Query Attention (MQA)
      ↓
2022: FlashAttention (implementation)
      ↓
2023: Grouped-Query Attention (GQA)  ← Current Standard
      ↓
2024: Multi-head Latent Attention (MLA)  ← Emerging
      FlashAttention-3
```

### Quality vs Efficiency Trade-off

```
Quality:   MLA > MHA > GQA > MQA
           ↑                    ↓
Efficiency: ↓                  ↑
```

**Current Best Practice**: GQA + FlashAttention
**Future Direction**: Possibly MLA + FlashAttention

### Memory Comparison (Example: 32 heads, 128 head_dim, 8K context)

| Mechanism | KV Cache Size | Relative |
|-----------|--------------|----------|
| MHA (32 heads) | 32 × 128 × 8K × 2 = 65.5 MB | 1.0x |
| GQA (8 KV heads) | 8 × 128 × 8K × 2 = 16.4 MB | 0.25x |
| MQA (1 KV head) | 1 × 128 × 8K × 2 = 2.05 MB | 0.03x |
| MLA (512 latent) | 512 × 8K × 2 = 8.2 MB | 0.125x |

---

## Specialized Attention Patterns

### Sliding Window Attention

**Used in**: Gemma 2, some Mistral variants

**Concept**:
```python
# Each token attends only to nearby tokens
window_size = 4096

# Token at position i attends to:
# positions [max(0, i - window_size), i]

# Benefits:
# - O(n × w) instead of O(n²)
# - Good for local dependencies
# - Efficient for long sequences
```

**Gemma 2 Pattern**:
- Alternate sliding window and global layers
- Local layers: Efficient
- Global layers: Long-range connections

### Sparse Attention

**Concept**: Attend to subset of positions
- Fixed patterns (every k-th position)
- Learned patterns (attention routing)
- Random sparse patterns

**Use Cases**:
- Very long sequences
- Hierarchical models
- Efficiency-critical applications

---

## Future Directions

### Research Areas

1. **Dynamic Attention**: Adapt pattern to input
2. **Linear Attention**: True O(n) complexity
3. **Mixture of Attentions**: Different mechanisms per layer
4. **Cross-Modal Attention**: Optimized for multimodal

### Trends

1. **Efficiency Focus**: Smaller KV cache priority
2. **MLA Adoption**: If validated more broadly
3. **FlashAttention Integration**: Ever more optimized
4. **Hybrid Approaches**: Combining multiple mechanisms

### Open Questions

1. How much quality loss is acceptable for efficiency?
2. Can we achieve MHA quality with MQA efficiency?
3. What's optimal group size for GQA?
4. Will MLA become standard?

---

## Practical Recommendations

### For Training New Models

**Small Models (<10B)**:
- GQA with 4:1 or 8:1 ratio
- FlashAttention-2/3
- Simple and effective

**Medium Models (10-100B)**:
- GQA with 8:1 ratio
- FlashAttention-3
- Consider MLA for cutting edge

**Large Models (>100B)**:
- GQA or MLA
- FlashAttention-3 essential
- Optimize for your use case

### For Inference

**Low Latency Priority**:
- GQA or MQA
- FlashAttention
- Aggressive quantization

**Long Context**:
- GQA or MLA
- FlashAttention-3
- Possible sparse attention

**Memory Constrained**:
- MQA or GQA with high ratio
- FlashAttention
- Consider quantization

---

## Sources

- [What is Grouped Query Attention](https://www.ibm.com/think/topics/grouped-query-attention)
- [Attention Variations - MQA vs GQA vs MHA vs MLA](https://verticalserve.medium.com/group-query-attention-58283b337c65)
- [MHA vs MQA vs GQA vs MLA](https://medium.com/@zaiinn440/mha-vs-mqa-vs-gqa-vs-mla-c6cf8285bbec)
- [Memory-Efficient Attention](https://cyk1337.github.io/notes/2024/05/10/Memory-Efficient-Attention/)
- [FlashAttention GitHub](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-3](https://tridao.me/blog/2024/flash3/)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
