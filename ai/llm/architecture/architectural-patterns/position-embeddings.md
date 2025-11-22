# Position Embeddings

Position embeddings enable transformers to understand token order. The evolution from absolute embeddings to RoPE and ALiBi has been crucial for context length scaling.

## Why Position Embeddings Matter

Transformers are **permutation-invariant** without position information:
- Self-attention treats input as a set, not a sequence
- "The cat sat" = "sat cat The" without position encoding
- Position embeddings inject sequence order

## Absolute / Learned Positional Embeddings

### Original Transformer Approach

**Implementation**:
```python
# Learned embedding matrix
pos_embeddings = nn.Embedding(max_seq_len, d_model)

# Add to input
x = token_embeddings + pos_embeddings[positions]
```

**Characteristics**:
- Fixed maximum length (e.g., 512, 2048 tokens)
- Trainable parameters for each position
- Simple to implement

### Advantages
1. **Simple**: Easy to understand and implement
2. **Learned**: Can adapt to data
3. **Proven**: Works well for fixed lengths

### Disadvantages
1. **Fixed length**: Can't extrapolate beyond training length
2. **Memory**: Requires parameters for each position
3. **Poor generalization**: Struggles with unseen lengths

### Used In
- Original BERT (512 tokens)
- Early GPT models
- Mostly deprecated in modern LLMs

---

## Sinusoidal Positional Encoding

### Original "Attention Is All You Need" (2017)

**Formula**:
```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

# Alternating sine and cosine at different frequencies
```

**Properties**:
- Deterministic (not learned)
- Continuous function of position
- Each dimension = different frequency

### Fourier Series Interpretation

Represents position as Fourier series:
- Low dimensions: High frequency (local patterns)
- High dimensions: Low frequency (global position)

### Advantages
1. **No parameters**: Deterministic, no training needed
2. **Infinite length**: Can generate for any position
3. **Relative position**: sin(a+b) can be expressed using sin(a), sin(b)

### Disadvantages
1. **Limited extrapolation**: Performance degrades beyond training length
2. **Fixed pattern**: Can't adapt to data
3. **Superseded**: Better methods available

### Used In
- Original Transformer (2017)
- Some early models
- Rarely used in modern LLMs

---

## RoPE (Rotary Position Embeddings) - 2021

### Key Innovation: Rotation in Complex Space

**Paper**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)

**Core Idea**: Rotate query and key vectors by position-dependent angle

**Mathematical Formulation**:
```python
# For each attention head dimension pair (2D subspace)
def rotate(x, position, theta):
    # Convert to complex number
    x_complex = x[::2] + 1j * x[1::2]

    # Compute rotation angle
    freqs = 1.0 / (theta ** (torch.arange(0, d, 2) / d))
    angles = position * freqs

    # Rotate
    rotation = torch.exp(1j * angles)
    x_rotated = x_complex * rotation

    # Convert back
    return torch.view_as_real(x_rotated).flatten()

# Apply to queries and keys
Q_rotated = rotate(Q, position_q, theta=10000)
K_rotated = rotate(K, position_k, theta=10000)

# Attention with rotated Q, K
attention = softmax(Q_rotated @ K_rotated.T / sqrt(d))
```

**Key Property**: Relative position encoding
```python
Q_m @ K_n.T = (rotate(Q, m) @ rotate(K, n).T)
           = Q @ rotate_matrix(n - m) @ K.T
           # Depends only on (n - m), the relative position!
```

### Advantages
1. **Relative position**: Naturally encodes relative distances
2. **Better extrapolation**: Works beyond training length (with scaling)
3. **No extra parameters**: Applied as rotation, no learned embeddings
4. **Efficient**: Simple rotation operation
5. **Flexible**: Can scale for longer contexts

### RoPE Scaling for Long Context

**Challenge**: Extrapolate from 2K training to 128K inference

**Solutions**:

1. **Linear Interpolation**:
   ```python
   # Compress position before rotation
   scaled_pos = position * (training_length / target_length)
   ```

2. **NTK-Aware Scaling**:
   ```python
   # Adjust base frequency
   theta_scaled = theta * (target_length / training_length) ** (d / (d-2))
   ```

3. **YaRN (Yet another RoPE extensioN)**:
   - Different scaling for different frequency bands
   - Better preservation of local and global information

### Used In (Widely Adopted)
- **Llama series** (2, 3, 3.1, 3.2, 3.3)
- Qwen series
- Yi models
- Falcon
- OpenELM
- StableLM (partial RoPE)
- **Most modern decoder-only LLMs**

### Why RoPE Won

1. **Relative position**: Better inductive bias than absolute
2. **Extrapolation**: Can extend to longer contexts
3. **Efficiency**: No extra parameters, simple computation
4. **Flexibility**: Scales with interpolation/extrapolation
5. **Validated**: Proven in largest models (Llama 3.1 405B with 128K context)

---

## ALiBi (Attention with Linear Biases) - 2022

### Key Innovation: Bias Instead of Embedding

**Paper**: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (2022)

**Core Idea**: Add distance-based bias to attention scores

**Implementation**:
```python
# Standard attention
scores = (Q @ K.T) / sqrt(d_k)

# ALiBi: Add linear bias based on distance
# For each head, different slope m
distance = positions_q - positions_k  # Distance matrix
bias = -m * abs(distance)  # Linear penalty for distance

# ALiBi attention
scores_alibi = scores + bias
attention = softmax(scores_alibi)
```

**Head-specific slopes**:
```python
# Different heads have different slopes
# Geometric sequence: m = 2^(-8/n * i) for head i

# Example (8 heads):
m = [2^-1, 2^-2, 2^-3, 2^-4, 2^-5, 2^-6, 2^-7, 2^-8]
  = [0.5, 0.25, 0.125, ...]
```

### Extraordinary Extrapolation

**Key Result**: Train on 512 tokens → Test on 3072+ tokens

- Minimal performance degradation
- Better than any other position encoding
- Enables extreme length extrapolation

### Advantages
1. **No positional embeddings**: Removes embedding layer entirely
2. **Excellent extrapolation**: Train short, test long
3. **Fewest parameters**: Just attention bias
4. **Fast computation**: Simple addition
5. **Memory efficient**: No embedding storage

### Disadvantages
1. **Less adoption**: RoPE more popular
2. **Fixed penalty**: Linear might not always be optimal
3. **Head-specific tuning**: Slope selection matters

### ALiBi + FlashAttention (2024)

**Major Update**: ALiBi support in FlashAttention v2.4

- **4-5x speedup** for ALiBi models
- Makes ALiBi practical at scale
- Combines extrapolation benefits with efficiency

### Used In
- **MPT** (7B, 30B)
- **BLOOM** (176B)
- Replit Code
- First architecture enabling 100K+ context

### Why ALiBi Hasn't Dominated

Despite superior extrapolation:
1. **RoPE momentum**: Llama success drove RoPE adoption
2. **Ecosystem**: More tooling for RoPE
3. **Familiarity**: Teams know RoPE better
4. **Both work**: RoPE with scaling also works well

### Future Potential

ALiBi could see renaissance:
- FlashAttention 2.4 makes it faster
- Extreme extrapolation valuable
- Simplicity appealing
- Could combine with other techniques

---

## Comparison Summary

| Feature | Absolute | Sinusoidal | RoPE | ALiBi |
|---------|----------|------------|------|-------|
| **Parameters** | O(L × d) | 0 | 0 | 0 |
| **Extrapolation** | Poor | Limited | Good (with scaling) | Excellent |
| **Efficiency** | Medium | High | High | Highest |
| **Adoption (2024)** | Low | Low | Very High | Medium |
| **Typical Length** | <2K | <8K | Up to 1M+ | Up to 100K+ |

### Extrapolation Performance

```
Training Length = 2K, Test Length = 8K

Absolute: ████░░░░░░ (40% performance)
Sinusoidal: █████░░░░░ (50% performance)
RoPE: ████████░░ (80% performance)
RoPE + Scaling: █████████░ (90% performance)
ALiBi: ██████████ (95%+ performance)
```

---

## Special Cases and Variations

### Partial RoPE (StableLM)

**Innovation**: Apply RoPE to only first 25% of dimensions

```python
# Split dimensions
rope_dims = d_head // 4
normal_dims = 3 * d_head // 4

# Apply RoPE to first 25%
x[:rope_dims] = apply_rope(x[:rope_dims], position)
# Keep rest unchanged
x[rope_dims:] = x[rope_dims:]
```

**Benefits**:
- Some position information
- Some position-independent features
- Potentially better generalization

### Learned Relative Position Bias

**Concept**: Learn bias for relative distances

```python
# Learn bias for each relative distance
bias_table = nn.Embedding(2 * max_distance + 1, num_heads)

# Apply based on relative position
relative_pos = positions_q - positions_k
bias = bias_table[relative_pos + max_distance]
```

**Used in**: Some research models, less common in production

### Conditional Position Encodings

**Idea**: Position encoding depends on content

- Research area
- Not widely adopted yet
- Potential for adaptive position information

---

## Evolution Timeline

```
2017: Sinusoidal (Original Transformer)
      ↓
2018: Learned Absolute (BERT, GPT-2)
      ↓
2021: RoPE (RoFormer)  ← Rotation-based
      ↓
2022: ALiBi (BLOOM, MPT)  ← Bias-based
      ↓
2023-2024: RoPE Dominance
      - Llama 2, 3 adoption
      - Scaling techniques mature
      ↓
2024: ALiBi + FlashAttention
      - Renewed interest
      - Performance improvements
```

---

## Modern Best Practices

### For New Models

**Standard Choice**: RoPE
- Proven at scale (Llama 3.1 405B, 128K context)
- Good extrapolation with scaling
- Excellent ecosystem support
- Well-understood

**Alternative**: ALiBi
- If extreme extrapolation needed
- Simpler implementation
- Fewer parameters
- With FlashAttention 2.4+

### For Long Context (>100K tokens)

**RoPE + Scaling**:
- Linear interpolation for moderate extension (2-4x)
- NTK-aware or YaRN for larger extension
- Validated to 1M+ tokens

**ALiBi**:
- Natural long context support
- Less tuning needed
- Excellent extrapolation

### For Short Context (<8K tokens)

Either RoPE or ALiBi works well:
- RoPE: More standard
- ALiBi: Slightly simpler

---

## Context Length Achievements

### With RoPE

| Model | Training Context | Inference Context | Technique |
|-------|-----------------|-------------------|-----------|
| Llama 2 | 4K | 4K | Base RoPE |
| Llama 3 | 8K | 8K | Base RoPE |
| Llama 3.1 | 128K | 128K | RoPE + Scaling |
| Qwen 2.5 | 128K | 128K-1M | RoPE + Scaling |

### With ALiBi

| Model | Training Context | Inference Context | Extrapolation |
|-------|-----------------|-------------------|---------------|
| MPT-7B | 2K | 8K+ | 4x+ |
| BLOOM | 2K | 3K+ | 1.5x+ |
| Custom ALiBi | 512 | 3072+ | 6x+ |

---

## Implementation Considerations

### RoPE Implementation

**Efficient**:
```python
# Precompute frequencies
freqs = 1.0 / (10000 ** (torch.arange(0, d, 2) / d))

# For each forward pass
cos = torch.cos(position * freqs)
sin = torch.sin(position * freqs)

# Fast rotation with cached sin/cos
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

q_embed = q * cos + rotate_half(q) * sin
k_embed = k * cos + rotate_half(k) * sin
```

### ALiBi Implementation

**Simple**:
```python
# Precompute slopes
n_heads = 8
slopes = torch.tensor([2 ** (-8/n_heads * i) for i in range(1, n_heads+1)])

# For each forward pass
distance = positions[:, None] - positions[None, :]  # [seq, seq]
bias = -slopes[:, None, None] * distance.abs()  # [heads, seq, seq]

# Add to attention scores
attention_scores = attention_scores + bias
```

---

## Future Directions

### Research Areas

1. **Adaptive Position**: Content-dependent position encoding
2. **Hybrid Approaches**: Combine RoPE and ALiBi
3. **Learned Scaling**: Automatically learn extrapolation
4. **Multi-scale Position**: Different granularities

### Trends

1. **Longer contexts**: Push toward million+ tokens
2. **Better extrapolation**: Train shorter, deploy longer
3. **Efficiency**: Combine with FlashAttention
4. **Standardization**: RoPE becoming universal

### Open Questions

1. Is linear bias (ALiBi) optimal, or can we do better?
2. How to optimally scale RoPE for very long contexts?
3. Can we eliminate position encoding entirely?
4. What's the theoretical limit of extrapolation?

---

## Practical Recommendations

### For Most Use Cases
**Use RoPE**: Industry standard, well-supported, proven

### For Extreme Extrapolation
**Consider ALiBi**: Best extrapolation, especially with FlashAttention 2.4

### For Research
**Try Both**: Compare on your specific task and data

### For Production
**RoPE + Scaling**: Safest bet, most tooling support

---

## Sources

- [Inside RoPE](https://learnopencv.com/rope-position-embeddings/)
- [All About Modern Positional Encodings](https://newsletter.theaiedge.io/p/all-about-the-modern-positional-encodings)
- [Complete Summary of Position Embeddings](https://azizbelaweid.substack.com/p/complete-summary-of-absolute-relative)
- [ALiBi FlashAttention](https://pli.princeton.edu/blog/2024/alibi-flashattention-speeding-alibi-3-5x-hardware-efficient-implementation)
- RoFormer paper (2021)
- ALiBi paper (2022)
