# Activation Functions and Normalization

Activation functions and normalization techniques are critical for training stability and model performance in LLMs.

## Activation Functions

### GELU (Gaussian Error Linear Unit)

**Formula**:
```python
GELU(x) = x * Φ(x)
# Where Φ(x) is the CDF of standard normal distribution

# Approximation:
GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
```

**Properties**:
- Smooth, non-monotonic
- Stochastic regularization interpretation
- Used in original GPT, BERT

**Used In**:
- GPT-2, GPT-3
- BERT
- Many earlier transformers
- Phi series (some variants)
- MPT, Falcon

### SwiGLU (Swish Gated Linear Unit)

**Formula**:
```python
SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)

# Where Swish(x) = x * σ(x) = x * sigmoid(x)
```

**In FFN Context**:
```python
# Standard FFN
FFN(x) = activation(x @ W1) @ W2

# SwiGLU FFN (3 projections)
FFN_SwiGLU(x) = (Swish(x @ W_gate) * (x @ W_up)) @ W_down
```

**PaLM Optimization**:
- Use 2/3 of standard FFN dimension to maintain parameter parity
- E.g., if standard FFN has 4d intermediate, SwiGLU uses 8d/3 ≈ 2.67d

**Advantages**:
1. **Better performance**: Outperforms ReLU, GELU empirically
2. **Gating mechanism**: Selective information flow
3. **Smooth**: Non-linear, differentiable
4. **Proven**: Works at massive scale

**Challenges**:
- FP8 training instabilities at scale (Intel research, Sept 2024)
- Slightly more compute than GELU
- Requires 3 weight matrices vs 2

**Used In** (Now Standard):
- Llama series (2, 3, 3.1)
- Mistral/Mixtral
- Qwen
- PaLM 2
- Most modern LLMs post-2023

### Why SwiGLU Won

Despite being more complex:
1. **Empirical superiority**: Consistently better results
2. **GLU family success**: Gated mechanisms work well
3. **Scale validation**: Proven in 100B+ models
4. **Google PaLM**: Strong endorsement from research

### Other Activation Functions

**ReLU (x) = max(0, x)**:
- Original transformer FFNs
- Rarely used in LLMs now

**Swish (x) = x * sigmoid(x)**:
- Smooth alternative to ReLU
- Component of SwiGLU

**xIELU (2024 Research)**:
- Recent research shows superiority over SwiGLU and ReLU²
- Not yet widely adopted

---

## Normalization Techniques

### LayerNorm (Original Transformer)

**Formula**:
```python
# For each sample and layer
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)

# Normalize
x_norm = (x - mean) / (std + ε)

# Scale and shift (learnable)
output = γ * x_norm + β
```

**Properties**:
- Normalizes across features (not batch)
- Learned scale (γ) and bias (β) parameters
- Centers data (subtracts mean)

**Computational Cost**:
- Mean calculation
- Variance calculation
- Division

**Used In**:
- Original Transformer (2017)
- GPT-2, GPT-3
- BERT
- StableLM (with learned bias)
- Many models still use it

### RMSNorm (Root Mean Square Normalization)

**Formula**:
```python
# Simpler: no mean subtraction
rms = sqrt(mean(x²) + ε)

# Normalize
x_norm = x / rms

# Scale (learnable, no bias)
output = γ * x_norm
```

**Key Difference**: No mean centering
- LayerNorm: `(x - mean) / std`
- RMSNorm: `x / rms`

**Advantages**:
1. **Simpler**: No mean calculation
2. **Faster**: ~7-64% faster than LayerNorm
3. **Fewer parameters**: No bias term
4. **Better for distributed**: More stable in distributed training
5. **Scale-invariant**: Normalizes magnitude only

**Mathematical Insight**:
- When input mean ≈ 0, LayerNorm ≈ RMSNorm
- Many intermediate activations have mean ≈ 0
- RMSNorm: "good enough" approximation that's cheaper

**Used In** (Now Standard):
- Llama series (all versions)
- Mistral/Mixtral
- Qwen
- Gemma (without bias)
- Most modern decoder-only LLMs

### Pre-Normalization vs Post-Normalization

**Post-Norm** (Original Transformer):
```python
# Attention
x = x + Attention(LayerNorm(x))  # Norm after residual
x = x + FFN(LayerNorm(x))
```

**Pre-Norm** (Modern Standard):
```python
# Attention
x = x + Attention(LayerNorm(x))  # Norm before operation
# FFN
x = x + FFN(LayerNorm(x))
```

**Pre-Norm Advantages**:
1. **Training stability**: Gradients flow better
2. **No warmup needed**: Can use higher learning rates from start
3. **Deeper networks**: Enables 100+ layer models
4. **Standard now**: GPT-style and LLaMA-style both use pre-norm

### Advanced Normalization

**QK-Norm (Qwen 3)**:
```python
# Normalize queries and keys before attention
Q_norm = normalize(Q, dim=-1)  # Per-head normalization
K_norm = normalize(K, dim=-1)
attention = softmax(Q_norm @ K_norm.T / sqrt(d))
```

**Benefits**:
- Prevents attention score explosion
- More stable training at scale
- Enables higher learning rates

**LayerNorm Without Biases (Gemma 2, StableLM 2)**:
```python
# Remove bias parameter entirely
x_norm = γ * (x / rms)  # Only scale, no shift
```

**Benefits**:
- Fewer parameters
- Cleaner architecture
- No quality degradation

**Logit Soft-Capping (Gemma 2)**:
```python
# Prevent extreme logit values
cap = 30.0
logits = cap * tanh(model_output / cap)
```

**Benefits**:
- Training stability
- Prevents numerical issues
- Smoother optimization

---

## Evolution and Adoption

### Timeline

```
2017: LayerNorm + ReLU (Original Transformer)
      ↓
2018-2019: LayerNorm + GELU (GPT-2, BERT)
      ↓
2020: GLU variants research
      ↓
2021: SwiGLU + RMSNorm (PaLM, LLaMA research)
      ↓
2023-2024: SwiGLU + RMSNorm Standard
      - Llama 2
      - Mistral
      - Qwen
      - Most new models
```

### Current Standard Stack (2024)

**Decoder-Only Transformer Layer**:
```python
# Pre-normalization
def transformer_layer(x):
    # Attention block
    x = x + Attention(RMSNorm(x))

    # FFN block with SwiGLU
    x_norm = RMSNorm(x)
    x = x + SwiGLU_FFN(x_norm)

    return x
```

**Components**:
1. **RMSNorm**: Pre-normalization before each sublayer
2. **SwiGLU**: Activation in FFN
3. **Residual Connections**: Skip connections around each block

---

## Comparison Summary

### Activation Functions

| Function | Complexity | Performance | Adoption (2024) |
|----------|-----------|-------------|-----------------|
| ReLU | Low | Baseline | Low |
| GELU | Medium | Good | Medium |
| SwiGLU | High | Best | Very High |

### Normalization

| Technique | Speed | Stability | Adoption (2024) |
|-----------|-------|-----------|-----------------|
| LayerNorm | 1.0x | Good | Medium |
| RMSNorm | 1.07-1.64x | Better | Very High |

---

## Why RMSNorm + SwiGLU Won

### RMSNorm Advantages

1. **Efficiency**: 7-64% faster than LayerNorm
2. **Simplicity**: Easier to implement and optimize
3. **Stability**: Better for deep networks and distributed training
4. **Validation**: Proven in Llama, Mistral, Qwen

### SwiGLU Advantages

1. **Performance**: Best empirical results
2. **Gating**: Selective information flow
3. **Scale**: Works well at 100B+ parameters
4. **Validation**: Google PaLM, Meta Llama endorsement

### Combined Benefits

Modern stack (RMSNorm + SwiGLU + Pre-Norm):
- Fast training and inference
- Stable at scale (100+ layers, 400B+ parameters)
- Best performance on benchmarks
- Industry standard

---

## Practical Recommendations

### For New Models

**Standard Choice**:
```python
# Use this stack
- RMSNorm (pre-normalization)
- SwiGLU activation
- No bias in normalization
```

**Proven, fast, effective**

### For Specialized Needs

**Maximum Stability**:
- Add QK-Norm (Qwen 3 style)
- Logit soft-capping (Gemma 2 style)
- Conservative learning rate

**Maximum Efficiency**:
- RMSNorm without bias
- Optimize FFN dimensions for SwiGLU
- FlashAttention integration

### For Research

**Try Innovations**:
- xIELU activation (new research)
- Adaptive normalization
- Learned normalization parameters

---

## Implementation Details

### RMSNorm Implementation

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * x / rms
```

### SwiGLU FFN Implementation

```python
class SwiGLU_FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # Three projections for SwiGLU
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # Swish gate
        gate = F.silu(self.w_gate(x))  # SiLU = Swish
        # Up projection
        up = self.w_up(x)
        # Combine and project down
        return self.w_down(gate * up)
```

---

## Future Directions

### Research Areas

1. **New Activations**: Beyond SwiGLU
2. **Adaptive Normalization**: Content-dependent
3. **Learned Techniques**: Meta-learning normalization
4. **Efficiency**: Even faster normalization

### Trends

1. **Simplification**: Remove unnecessary components (biases)
2. **Stability**: More techniques like QK-Norm
3. **Performance**: Continue empirical search
4. **Understanding**: Why does SwiGLU work so well?

### Open Questions

1. Why is SwiGLU empirically superior?
2. Can we do better than RMSNorm?
3. Optimal FFN dimension for SwiGLU?
4. Are there even better activations undiscovered?

---

## Sources

- [Exploring SwiGLU](https://medium.com/@s_boudefel/exploring-swiglu-the-activation-function-powering-modern-llms-9697f88221e7)
- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
- [SwiGLU: The FFN Upgrade](https://dev.to/mshojaei77/swiglu-the-ffn-upgrade-i-use-to-get-free-performance-33jc)
- [Re-Introducing LayerNorm](https://arxiv.org/abs/2409.12951)
- [Mastering LLama - RMSNorm](https://medium.com/@hugmanskj/mastering-llama-rmsnorm-ae5c5d504e9a)
- [Normalization Techniques in Transformer-Based LLMs](https://sushant-kumar.com/blog/normalization-in-transformer-based-llms)
- GLU Variants Improve Transformer (2020)
- PaLM paper (2022)
