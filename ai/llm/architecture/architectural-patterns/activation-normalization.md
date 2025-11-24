# Activation Functions and Normalization

Activation functions and normalization techniques are critical for training stability and model performance in LLMs.

## Historical Context: The Evolution of Activation Functions

### Original Transformer (2017): ReLU Era

The original "Attention Is All You Need" paper (Vaswani et al., 2017) used **ReLU (Rectified Linear Unit)** in the feed-forward networks.

**FFN Formula**:
```python
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```

**Architecture Details**:
- Input/output dimension: d_model = 512
- Inner layer dimension: d_ff = 2048 (4x expansion)
- Two linear transformations with ReLU activation in between
- Applied position-wise (same network at each position)

**Why ReLU?**
- **Standard choice in 2017**: De facto activation for deep learning
- **Computationally efficient**: Just `max(0, x)`
- **Proven track record**: Worked well in CNNs and other architectures
- **No justification provided**: Paper did not conduct ablation studies on activation functions

**The Transition**: ReLU (2017) → GELU (2018) → SwiGLU (2023)

---

## Activation Functions by Model (2017-2025)

| Model/Series | Activation | Year | Organization | Notes |
|--------------|-----------|------|--------------|-------|
| **Original Transformer** | ReLU | 2017 | Google | "Attention Is All You Need" |
| **GPT-1** | GELU | 2018 | OpenAI | First major model to use GELU in transformers |
| **BERT** | GELU | 2018 | Google | Followed GPT-1's innovation |
| **GPT-2** | GELU-New | 2019 | OpenAI | Variant of GELU |
| **GPT-3** | GELU-New | 2020 | OpenAI | Continued GELU tradition |
| **GPT-4** | GELU (likely) | 2023 | OpenAI | Architecture not disclosed |
| **PaLM** | SwiGLU | 2022 | Google | First major model with SwiGLU |
| **PaLM 2** | SwiGLU | 2023 | Google | Validated at scale |
| **Llama 1** | SwiGLU | 2023 | Meta | Popularized in open-source |
| **Llama 2** | SwiGLU | 2023 | Meta | Industry standard emerges |
| **Llama 3/3.1** | SwiGLU | 2024 | Meta | Continued standard |
| **Llama 4** | SwiGLU | 2025 | Meta | With MoE architecture |
| **Mistral 7B** | SwiGLU | 2023 | Mistral AI | All variants |
| **Mixtral 8x7B/8x22B** | SwiGLU | 2024 | Mistral AI | MoE with SwiGLU |
| **DeepSeek V2** | SwiGLU | 2024 | DeepSeek | MoE architecture |
| **DeepSeek V3** | SwiGLU | 2024 | DeepSeek | 671B params |
| **Qwen2/2.5** | SwiGLU | 2024 | Alibaba | All sizes |
| **Gemma** | GeGLU | 2024 | Google | GELU variant with gating |
| **Yi** | SwiGLU | 2023 | 01.AI | Based on Llama architecture |
| **BLOOM** | GELU | 2022 | BigScience | 176B multilingual |
| **Falcon** | GELU | 2023 | TII | Uses standard GELU |
| **Claude** | Unknown | 2023+ | Anthropic | Architecture not disclosed |
| **Gemini** | Unknown | 2023+ | Google | Architecture not disclosed |

---

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
- **Original transformer** (2017) used ReLU in FFNs
- Standard choice in deep learning at the time
- Rarely used in modern LLMs now
- **Recent research (2024)**: "ReLU Strikes Back" shows ReLU can match GELU/SwiGLU with proper tuning and slightly longer training, while providing faster inference

**Swish (x) = x * sigmoid(x)**:
- Smooth alternative to ReLU
- Component of SwiGLU
- Discovered via automatic search (Google, 2017)

**xIELU (2024 Research)**:
- Recent research shows superiority over SwiGLU and ReLU²
- Not yet widely adopted

**Obsolete in Modern LLMs**:
- **Sigmoid/Tanh**: Vanishing gradient problems, only used in gating mechanisms
- **LeakyReLU, PReLU, ELU**: Superseded by GELU/SwiGLU, no clear advantages
- **Mish**: Computational cost without clear benefits

---

## The Industry Transition Story

### Phase 1: ReLU Era (2017)

**Original Transformer** (Vaswani et al., 2017):
- Used standard ReLU activation
- No exploration of alternatives
- Simple, fast, proven choice

**Limitation**: ReLU's hard cutoff at zero (non-differentiable at x=0) and sparse activations weren't optimal for transformers.

### Phase 2: GELU Revolution (2018-2022)

**GPT-1 Pioneers GELU** (OpenAI, 2018):
- First major transformer to use GELU
- Showed smooth, non-monotonic activation improved performance
- BERT immediately followed this innovation

**Why GELU Won**:
1. **Smooth and differentiable**: Better gradient flow than ReLU
2. **Probabilistic interpretation**: Weights inputs by percentile
3. **Empirically better**: Improved performance across tasks
4. **Non-zero gradient at x=0**: Allows learning in that region

**Adoption**: Became standard for transformers (2018-2022)
- GPT-2, GPT-3 (GELU-New variant)
- BERT and derivatives
- Most early transformer models

### Phase 3: SwiGLU Era (2022-Present)

**Introduction** (Noam Shazeer, 2020):
- Paper: "GLU Variants Improve Transformer"
- Tested GEGLU, ReGLU, SwiGLU
- SwiGLU and GEGLU produced best results

**First Major Adopter: PaLM** (Google, April 2022):
- 540B parameter model with SwiGLU
- Validated at extreme scale
- Set precedent for industry

**Popularizer: Llama** (Meta, February 2023):
- First major open-source model with SwiGLU
- Made architecture accessible to researchers
- Spawned ecosystem of derivatives (Vicuna, Alpaca, etc.)

**Rapid Adoption** (2023-2024):
- Mistral, Mixtral
- DeepSeek V2, V3
- Qwen 2, 2.5
- Nearly all new major open models

**Why SwiGLU Won**:
1. **Empirical superiority**: Consistently better results at scale
2. **Gating mechanism**: Explicit control over information flow
3. **Training stability**: Critical for very deep networks (100+ layers)
4. **Better gradient flow**: Prevents vanishing/exploding gradients
5. **Scale validation**: Proven in 100B+ parameter models

### Timeline Summary

```
2017: ReLU (Original Transformer)
      ↓
2018: GELU (GPT-1 innovation, BERT adoption)
      ↓
2019-2020: GELU standard (GPT-2, GPT-3)
      ↓
2020: SwiGLU introduced (research)
      ↓
2022: PaLM validates SwiGLU at scale
      ↓
2023: Llama popularizes SwiGLU in open-source
      ↓
2024-2025: SwiGLU becomes industry standard
```

---

## Current Industry Consensus (2024-2025)

### Open-Source Models: SwiGLU Dominant

**Nearly universal adoption**:
- All Llama derivatives (Llama 2, 3, 4)
- Mistral/Mixtral series
- DeepSeek series
- Qwen series
- Yi models
- Most new models default to SwiGLU

**Why**: Llama's success and influence, proven performance, community standardization

### Proprietary Models: Split

**OpenAI (GPT series)**:
- Likely continues GELU tradition
- GPT-1, 2, 3 all used GELU
- GPT-4 architecture undisclosed but probably GELU
- Reason: "If it ain't broke, don't fix it" + proprietary optimizations

**Anthropic (Claude)**:
- Architecture completely undisclosed
- May use novel activation or variant
- Research-driven, likely experimenting

**Google**:
- PaLM/PaLM 2: SwiGLU
- Gemma: GeGLU (GELU variant with gating)
- Gemini: Unknown
- Different teams, different choices

### The Consensus

**For new large-scale LLMs (>10B params)**:
- ✅ **Use SwiGLU** - Industry standard
- Proven, fast, effective
- Best empirical results

**For smaller models or efficiency-focused**:
- ⚠️ **GELU still viable** - Mature implementations, lower complexity

**Key Insight**: The industry has converged on just **two main choices** (GELU or SwiGLU), with SwiGLU dominant for modern models.

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

---

## The Normalization Transition Story

### Phase 1: BatchNorm Era (Pre-2017)

**BatchNorm Dominance** (2015-2016):
- Standard normalization for CNNs and deep learning
- Normalizes across the batch dimension
- Highly successful in computer vision

**Why BatchNorm Failed for Transformers**:
1. **Training-Inference Discrepancy (TID)**: Uses batch statistics during training but running averages during inference
2. **Large Statistical Fluctuations**: NLP data exhibits much larger fluctuations than computer vision
3. **Variable Sequence Lengths**: Different batches have different normalization constants
4. **Poor Generalization**: Trains fast but validation performance degrades

**Result**: BatchNorm was abandoned for sequence models and transformers.

---

### Phase 2: LayerNorm Era (2016-2022)

**LayerNorm Introduction** (2016):
- **Paper**: "Layer Normalization" (Ba, Kiros, Hinton)
- **Original purpose**: Designed for RNNs
- **Key innovation**: Normalizes across features for each sample independently
- **Solves**: Batch dependency problem

**Original Transformer** (2017):
- Used **Post-Norm** (LayerNorm after residual): `LayerNorm(x + Sublayer(x))`
- Combined with ReLU activation
- Standard for early transformers

**Post-Norm Era Models** (2017-2018):
- Original Transformer (2017)
- BERT (2018)
- GPT-1 (2018)

**Limitation**: Post-Norm requires learning rate warmup and struggles with very deep networks (>10 layers).

---

### Phase 3: Pre-Norm Revolution (2019-2020)

**GPT-2 Changes the Game** (2019):
- **Major shift**: Moved LayerNorm to INPUT of each sub-block
- **Pre-Norm**: `x + Sublayer(LayerNorm(x))`
- Similar to pre-activation ResNet

**Why Pre-Norm Won**:
1. **Training stability**: Better gradient flow for deep networks
2. **No warmup needed**: Can use higher learning rates from start
3. **Faster convergence**: Trains faster than Post-Norm
4. **Enables depth**: Critical for 100+ layer models
5. **Better gradients**: Well-behaved at initialization

**Adoption**:
- **2017-2018**: Post-Norm standard (Original Transformer, BERT, GPT-1)
- **2019**: GPT-2 introduces Pre-Norm
- **2020+**: Pre-Norm becomes universal (GPT-3, all modern models)

**Trade-off**: Post-Norm performs better in shallow transformers (≤6 layers), but Pre-Norm essential for deep models.

---

### Phase 4: RMSNorm Era (2019-Present)

**RMSNorm Introduction** (2019):
- **Paper**: "Root Mean Square Layer Normalization" (Zhang & Sennrich)
- **Published**: NeurIPS 2019
- **Hypothesis**: Re-scaling (not re-centering) is what matters
- **Innovation**: Remove mean subtraction, normalize only by RMS

**Early Adopters** (2019-2021):

**T5** (Google, 2019):
- Early adopter alongside paper publication
- Achieved 7-9% training speedup
- Demonstrated viability at scale

**Gopher** (DeepMind, 2021):
- 280B parameters, 80 layers
- Used RMSNorm throughout
- Validated extreme-scale effectiveness
- State-of-the-art on 81% of 152 tasks

**Chinchilla** (DeepMind, 2022):
- Gopher family, continued RMSNorm usage
- Further validation at scale

**Industry Validation** (2022):

**PaLM** (Google, April 2022):
- 540B parameters
- Combined RMSNorm + SwiGLU + RoPE
- Set modern architecture precedent

**Popularization** (2023):

**Llama 1** (Meta, February 2023):
- **Confirmed RMSNorm**: Pre-normalization with RMSNorm
- First major open-source model with modern stack
- Made architecture accessible to researchers
- **Turning point**: RMSNorm becomes standard after Llama

**Rapid Adoption** (2023-2024):
- Llama 2, 3, 3.1, 4 (Meta)
- Mistral 7B, Mixtral series (Mistral AI)
- DeepSeek V2, V3 (DeepSeek)
- Qwen 2, 2.5 (Alibaba)
- Gemma (Google)
- Yi (01.AI)
- Nearly all new open models

**Why RMSNorm Won**:
1. **Computational efficiency**: 7-64% faster than LayerNorm
2. **Simplicity**: No mean calculation needed
3. **Fewer parameters**: No bias term
4. **Better stability**: More stable for very deep networks (80+ layers)
5. **Distributed training**: Better for large-scale training
6. **Empirical success**: Proven at 100B+ parameter scale
7. **Mathematical insight**: Works because intermediate activations have mean ≈ 0

---

### Timeline Summary

```
Pre-2017: BatchNorm (CNNs, fails for sequences)
    ↓
2016: LayerNorm paper (designed for RNNs)
    ↓
2017: Original Transformer (Post-LN + LayerNorm + ReLU)
    ↓
2018: BERT, GPT-1 (Post-LN + LayerNorm + GELU)
    ↓
2019: GPT-2 introduces Pre-LN (major stability breakthrough)
      RMSNorm paper + T5 early adoption
    ↓
2020: GPT-3 (Pre-LN + LayerNorm + GELU)
    ↓
2021: Gopher validates RMSNorm at 280B scale
    ↓
2022: PaLM (Pre-LN + RMSNorm + SwiGLU)
      Modern stack emerges
    ↓
2023: Llama 1 (Feb) - RMSNorm becomes standard
      Llama 2, Mistral follow
    ↓
2024-2025: RMSNorm universal in new models
           Llama 3/4, Mixtral, DeepSeek, Qwen
```

---

## Normalization by Model (2017-2025)

### Post-Norm + LayerNorm Era (2017-2018)

| Model | Year | Organization | Normalization | Activation |
|-------|------|--------------|---------------|------------|
| **Original Transformer** | 2017 | Google | Post-LN + LayerNorm | ReLU |
| **BERT** | 2018 | Google | Post-LN + LayerNorm | GELU |
| **GPT-1** | 2018 | OpenAI | Post-LN + LayerNorm | GELU |

### Pre-Norm + LayerNorm Era (2019-2022)

| Model | Year | Organization | Normalization | Activation |
|-------|------|--------------|---------------|------------|
| **GPT-2** | 2019 | OpenAI | Pre-LN + LayerNorm | GELU |
| **GPT-3** | 2020 | OpenAI | Pre-LN + LayerNorm | GELU |
| **GPT-4** | 2023 | OpenAI | Pre-LN + LayerNorm (likely) | GELU (likely) |
| **BLOOM** | 2022 | BigScience | Pre-LN + LayerNorm | GELU |
| **Falcon** | 2023 | TII | Pre-LN + LayerNorm | GELU |
| **StableLM** | 2023 | Stability AI | Pre-LN + LayerNorm | GELU |

### Pre-Norm + RMSNorm Era (2019-Present)

| Model | Year | Organization | Normalization | Activation |
|-------|------|--------------|---------------|------------|
| **T5** | 2019 | Google | Pre-LN + RMSNorm | Early adopter |
| **Gopher** | 2021 | DeepMind | Pre-LN + RMSNorm | 280B, 80 layers |
| **Chinchilla** | 2022 | DeepMind | Pre-LN + RMSNorm | Gopher family |
| **PaLM** | 2022 | Google | Pre-LN + RMSNorm | SwiGLU, 540B |
| **Llama 1** | 2023 | Meta | Pre-LN + RMSNorm | SwiGLU - Popularizer |
| **Llama 2** | 2023 | Meta | Pre-LN + RMSNorm | SwiGLU |
| **Llama 3/3.1** | 2024 | Meta | Pre-LN + RMSNorm | SwiGLU |
| **Llama 4** | 2025 | Meta | Pre-LN + RMSNorm | SwiGLU + MoE |
| **Mistral 7B** | 2023 | Mistral AI | Pre-LN + RMSNorm | SwiGLU |
| **Mixtral 8x7B/8x22B** | 2024 | Mistral AI | Pre-LN + RMSNorm | SwiGLU + MoE |
| **DeepSeek V2** | 2024 | DeepSeek | Pre-LN + RMSNorm | SwiGLU + MoE |
| **DeepSeek V3** | 2024 | DeepSeek | Pre-LN + RMSNorm | SwiGLU, 671B |
| **Qwen2/2.5** | 2024 | Alibaba | Pre-LN + RMSNorm | SwiGLU + QK-Norm |
| **Gemma** | 2024 | Google | Pre-LN + RMSNorm | GeGLU, no bias |
| **Yi** | 2023 | 01.AI | Pre-LN + RMSNorm | SwiGLU |

---

## Current Normalization Consensus (2024-2025)

### Open-Source Models: RMSNorm Dominant

**Nearly universal adoption**:
- All Llama derivatives (Llama 2, 3, 4)
- Mistral/Mixtral series
- DeepSeek series
- Qwen series
- Gemma (Google)
- Yi models
- All new major open models

**Why**:
- Llama's influence in open-source community
- Proven performance at scale
- Computational efficiency
- Community standardization

### Proprietary Models: Mixed

**OpenAI (GPT series)**:
- Likely continues LayerNorm tradition
- GPT-2, 3 all used LayerNorm
- GPT-4 architecture undisclosed but probably LayerNorm
- Reason: Established optimization, "if it ain't broke"

**Older Models Still Using LayerNorm**:
- BLOOM (2022)
- Falcon (2023)
- StableLM (some versions)
- Various pre-2023 models

**Unknown**:
- Claude (Anthropic) - architecture undisclosed
- Gemini (Google) - architecture undisclosed

### Advanced Variants (2024+)

**QK-Norm** (Qwen 3):
- Normalize queries and keys before attention
- Prevents attention score explosion
- Enables higher learning rates

**LayerNorm without Bias** (Gemma 2):
- Remove bias parameter entirely
- Fewer parameters, cleaner architecture
- No quality degradation

**Logit Soft-Capping** (Gemma 2):
- Prevent extreme logit values
- Training stability
- Smoother optimization

### The Consensus

**For new large-scale LLMs**:
- ✅ **Use RMSNorm + Pre-Norm** - Industry standard
- Proven, fast, stable
- Best for 10B+ parameter models

**For legacy/smaller models**:
- ⚠️ **LayerNorm still viable** - Mature, well-understood
- Acceptable for models <10B or where optimization is critical

**Key Insight**: Just like activation functions, normalization has converged on a clear winner (RMSNorm) for modern models, driven by Llama's 2023 success.

---

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

### Timeline: Complete Evolution

```
2017: LayerNorm + ReLU (Original Transformer)
      ↓
2018: LayerNorm + GELU (GPT-1 pioneers, BERT follows)
      ↓
2019-2020: LayerNorm + GELU becomes standard (GPT-2, GPT-3)
      ↓
2020: GLU variants research (SwiGLU introduced)
      ↓
2021-2022: SwiGLU + RMSNorm research
      ↓
2022: PaLM validates SwiGLU + RMSNorm at 540B scale
      ↓
2023: Llama popularizes SwiGLU + RMSNorm in open-source
      - Llama 1, 2
      - Mistral
      ↓
2024-2025: SwiGLU + RMSNorm Industry Standard
      - Llama 3, 4
      - Mixtral
      - DeepSeek V2, V3
      - Qwen 2, 2.5
      - Most new models
```

**Key Transition Points**:
- **2017**: ReLU era (original transformer)
- **2018**: GELU revolution (GPT-1 innovation)
- **2022**: SwiGLU validation (PaLM at scale)
- **2023**: SwiGLU popularization (Llama open-source)
- **2024**: SwiGLU dominance (industry standard)

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

### Original Papers
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - Original Transformer with ReLU
- [GELU: Gaussian Error Linear Units (2016)](https://arxiv.org/abs/1606.08415) - GELU introduction
- [GLU Variants Improve Transformer (2020)](https://arxiv.org/abs/2002.05202) - SwiGLU introduction
- [PaLM: Scaling Language Modeling with Pathways (2022)](https://arxiv.org/abs/2204.02311) - First major SwiGLU adopter

### Activation Functions
- [Exploring SwiGLU](https://medium.com/@s_boudefel/exploring-swiglu-the-activation-function-powering-modern-llms-9697f88221e7)
- [SwiGLU: The FFN Upgrade](https://dev.to/mshojaei77/swiglu-the-ffn-upgrade-i-use-to-get-free-performance-33jc)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/01/attention.html) - Original Transformer implementation
- [ReLU Strikes Back (2024)](https://arxiv.org/abs/2310.04564) - Recent ReLU research

### Normalization
- [Layer Normalization (2016)](https://arxiv.org/abs/1607.06450) - Original LayerNorm paper (Ba, Kiros, Hinton)
- [Root Mean Square Layer Normalization (2019)](https://arxiv.org/abs/1910.07467) - RMSNorm paper (Zhang & Sennrich)
- [On Layer Normalization in the Transformer Architecture (2020)](https://arxiv.org/abs/2002.04745) - Post-norm vs Pre-norm analysis
- [Understanding the Failure of Batch Normalization for Transformers in NLP](https://arxiv.org/abs/2210.05153) - Why BatchNorm failed
- [Re-Introducing LayerNorm](https://arxiv.org/abs/2409.12951)
- [Mastering LLama - RMSNorm](https://medium.com/@hugmanskj/mastering-llama-rmsnorm-ae5c5d504e9a)
- [Normalization Techniques in Transformer-Based LLMs](https://sushant-kumar.com/blog/normalization-in-transformer-based-llms)
- [RMSNorm: Fueling the Next Generation of LLMs](https://medium.com/foundation-models-deep-dive/rmsnorm-fueling-the-next-generation-of-llms-f61cd8c0c09f)
- [Why the Original Transformer Figure Shows Pre-LN](https://magazine.sebastianraschka.com/p/why-the-original-transformer-figure)

### Architecture Comparisons
- [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)
- [All the Activation Functions](https://dublog.net/blog/all-the-activations/)
- [Activation Functions - AussieAI Book](https://www.aussieai.com/book/ch21-activation-functions)
