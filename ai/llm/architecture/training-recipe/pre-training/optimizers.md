# Optimizers for LLM Training

The choice of optimizer is one of the most settled aspects of LLM training. While data, architecture, and alignment techniques continue to evolve rapidly, optimization has converged on a clear standard: **AdamW with cosine learning rate decay and linear warmup**. This document traces how we got here and explores emerging alternatives.

---

## Historical Evolution

### The Optimization Timeline

| Year | Optimizer | Key Innovation | Limitation |
|------|-----------|----------------|------------|
| 1951 | SGD | Stochastic gradient descent | Slow convergence, sensitive to LR |
| 1986 | Momentum | Accelerated gradients | Fixed momentum coefficient |
| 2011 | AdaGrad | Per-parameter adaptive LR | LR decays too aggressively |
| 2012 | RMSprop | Exponential moving average of squared gradients | No momentum |
| 2014 | Adam | Combined momentum + adaptive LR | Weight decay implementation issue |
| 2017 | AdamW | Decoupled weight decay | Memory-intensive |
| 2023+ | Lion, Sophia | Efficiency improvements | Less proven at scale |

### Phase 1: Classical Optimizers (1951-2010)

**Stochastic Gradient Descent (SGD)**

The foundation of all neural network training:
```
θ = θ - η · ∇L(θ)
```
Where η is the learning rate and ∇L is the gradient of the loss.

**Problem**: High variance in gradient estimates, slow convergence, very sensitive to learning rate choice.

**SGD with Momentum** (1986)

Adds velocity term to smooth updates:
```
v = γ·v + η·∇L(θ)
θ = θ - v
```
Where γ is momentum coefficient (typically 0.9).

**Improvement**: Faster convergence, dampens oscillations. Used by Krizhevsky et al. for AlexNet (2012).

### Phase 2: Adaptive Learning Rates (2011-2014)

**AdaGrad** (2011) - Duchi et al.

Per-parameter learning rates based on historical gradients:
```
G = G + (∇L)²           # Accumulate squared gradients
θ = θ - (η/√(G+ε)) · ∇L  # Scale LR by accumulated magnitude
```

**Problem**: Learning rate decays to zero—good for sparse data, bad for deep learning.

**RMSprop** (2012) - Hinton (unpublished)

Exponential moving average instead of sum:
```
E[g²] = ρ·E[g²] + (1-ρ)·(∇L)²
θ = θ - (η/√(E[g²]+ε)) · ∇L
```
Where ρ ≈ 0.9 controls decay rate.

**Improvement**: Learning rate doesn't decay to zero, adapts throughout training.

### Phase 3: Adam Era (2014-2017)

**[Adam](https://arxiv.org/abs/1412.6980)** (2014) - Kingma & Ba (ICLR 2015)

Combined momentum with adaptive learning rates:
```python
# First moment (momentum-like)
m = β₁·m + (1-β₁)·∇L

# Second moment (RMSprop-like)
v = β₂·v + (1-β₂)·(∇L)²

# Bias correction (critical for early steps)
m̂ = m / (1-β₁ᵗ)
v̂ = v / (1-β₂ᵗ)

# Update
θ = θ - η · m̂ / (√v̂ + ε)
```

Default hyperparameters: β₁=0.9, β₂=0.999, ε=1e-8

**Why Adam Dominated**:
1. Works well out-of-the-box with default hyperparameters
2. Handles sparse gradients effectively
3. Robust to learning rate choice
4. Fast convergence in early training

**The Weight Decay Problem**:

Adam implemented L2 regularization incorrectly—applying decay to the scaled gradient rather than the weights directly:
```python
# Incorrect (original Adam with L2)
θ = θ - η · (m̂ / (√v̂ + ε) + λ·θ)  # λ·θ is scaled by adaptive LR

# Correct (decoupled)
θ = θ - η · m̂ / (√v̂ + ε) - η·λ·θ   # λ·θ applied directly
```

### Phase 4: AdamW Standard (2017-Present)

**[AdamW](https://arxiv.org/abs/1711.05101)** (2017) - Loshchilov & Hutter

Decoupled weight decay from gradient updates:
```python
# Gradient moments (same as Adam)
m = β₁·m + (1-β₁)·∇L
v = β₂·v + (1-β₂)·(∇L)²
m̂ = m / (1-β₁ᵗ)
v̂ = v / (1-β₂ᵗ)

# Decoupled update
θ = θ - η · m̂ / (√v̂ + ε)  # Adam update
θ = θ - η · λ · θ          # Weight decay (separate)
```

**Why AdamW Became Standard**:
1. Weight decay works correctly (regularization independent of LR)
2. More stable training dynamics at scale
3. Simpler hyperparameter tuning (λ less coupled to η)
4. Adopted by all major LLMs: GPT-3, LLaMA, Mistral, Claude

---

## The LLM Training Standard

### Default Configuration

The following configuration is used by virtually all modern LLMs:

```python
optimizer = AdamW(
    lr=3e-4,              # Peak learning rate (varies: 1e-4 to 6e-4)
    betas=(0.9, 0.95),    # Note: β₂=0.95, not 0.999
    eps=1e-8,
    weight_decay=0.1      # Higher than typical (0.01-0.1)
)
```

**Key Hyperparameters**:

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Peak LR | 3e-4 | 1e-4 to 6e-4 | Lower for larger models |
| β₁ | 0.9 | Fixed | Momentum coefficient |
| β₂ | 0.95 | 0.95-0.999 | Lower than Adam default |
| ε | 1e-8 | 1e-8 | Rarely changed |
| Weight decay | 0.1 | 0.01-0.1 | Higher for larger models |
| Gradient clipping | 1.0 | 0.5-1.0 | Max gradient norm |

### Why β₂=0.95?

LLM training uses β₂=0.95 rather than Adam's default 0.999:
- **Faster adaptation**: Second moment updates more quickly
- **Better for non-stationary**: Language data distributions shift during training
- **Empirically validated**: GPT-3, LLaMA papers both use 0.95

---

## Learning Rate Schedules

The learning rate schedule is more impactful than optimizer choice. Modern LLM training universally uses:

### Warmup + Cosine Decay

```
LR
 ↑
 |      ╱‾‾‾‾‾‾‾‾‾‾‾‾╲
 |     /              ╲
 |    /                ╲
 |   /                  ╲
 |  /                    ╲
 | /                      ╲___
 └─────────────────────────────→ Steps
   warmup    training      final
```

**Three Phases**:

1. **Linear Warmup** (~1-5% of training)
   - LR increases linearly from 0 to peak
   - Stabilizes early training before gradients are accurate
   - Prevents large, destabilizing updates

2. **Cosine Decay** (main training)
   - LR follows cosine curve from peak to minimum
   - Gradual reduction allows fine-tuning later in training
   - Formula: `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t/T))`

3. **Final LR** (typically 10% of peak)
   - Training continues at reduced rate
   - Some implementations hold constant, others continue decay

### Implementation

```python
def cosine_schedule_with_warmup(step, total_steps, warmup_steps, lr_max, lr_min):
    if step < warmup_steps:
        # Linear warmup
        return lr_max * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * progress))
```

### Typical Values

| Model Scale | Peak LR | Warmup Steps | Final LR |
|-------------|---------|--------------|----------|
| 1B params | 3e-4 | 2000 | 3e-5 (10%) |
| 7B params | 3e-4 | 2000 | 3e-5 (10%) |
| 70B params | 1.5e-4 | 2000 | 1.5e-5 (10%) |
| 175B params | 6e-5 | 375 | 6e-6 (10%) |

### Why Warmup Matters

Without warmup:
1. **Adam's bias correction** is unstable in early steps
2. **Random initialization** means early gradients are unreliable
3. **Large updates** can push model into bad loss regions

Warmup allows:
1. Bias-corrected moments to stabilize
2. Gradients to become meaningful
3. Model to settle before aggressive optimization

### Alternative: Warmup-Stable-Decay (WSD)

Recent research proposes a three-phase schedule:
```
LR
 ↑
 |      ╱‾‾‾‾‾‾‾‾‾‾‾‾‾‾╲
 |     /                 ╲
 |    /                   ╲
 └───/─────────────────────╲──→ Steps
   warmup     stable      decay
```

**Advantage**: Decouples training duration from schedule design—useful for continual learning.

---

## Memory Considerations

### AdamW Memory Cost

AdamW requires storing:
- Model parameters: P
- Gradients: P
- First moments (m): P
- Second moments (v): P

**Total**: ~4× model size in optimizer state

For a 7B parameter model in fp32:
```
Model: 7B × 4 bytes = 28 GB
Gradients: 28 GB
First moment: 28 GB
Second moment: 28 GB
Total: 112 GB (just for optimizer!)
```

This is why:
- Mixed precision training uses fp16/bf16 for forward/backward
- Optimizer states often kept in fp32 for stability
- Memory optimization techniques (ZeRO, FSDP) shard optimizer state

### Low-Memory Alternatives

**8-bit Adam** (bitsandbytes)
- Stores optimizer states in int8
- ~4× memory reduction with minimal quality loss
- Used for fine-tuning when memory constrained

**Adafactor** (2018)
- Factorizes second moment matrix
- Memory: O(rows + cols) instead of O(rows × cols)
- Used by T5, some smaller models

---

## Emerging Optimizers

### Lion (2023) - Google Brain

EvoLved Sign Momentum optimizer:
```python
# Much simpler than Adam
update = sign(β₁·m + (1-β₁)·∇L)
m = β₂·m + (1-β₂)·∇L
θ = θ - η·update - η·λ·θ
```

**Benefits**:
- 2× memory efficient (only stores m, not v)
- Often faster convergence
- Simpler compute

**Limitations**:
- Requires different hyperparameters than AdamW
- Less proven at frontier scale

### Sophia (2023) - Stanford

Second-order optimizer using Hessian diagonal:
```
θ = θ - η · clip(m̂ / max(ĥ, γ), ρ)
```
Where ĥ estimates diagonal Hessian.

**Benefits**:
- 2× fewer steps to same loss
- Better validation performance in some settings

**Limitations**:
- Hessian estimation adds overhead
- Complex implementation

### Schedule-Free AdamW (2024) - Meta

Eliminates learning rate schedule entirely:
```python
# Momentum-based averaging replaces scheduling
y = (1-β)·θ + β·z
# Dynamic adjustment instead of cosine decay
```

**Benefits**:
- No schedule hyperparameters
- Won MLCommons AlgoPerf Challenge
- Adapts dynamically

**Status**: Promising but not yet standard in production.

### Adam-mini (2024)

Reduces memory by 50%:
- Only stores second moments for subset of parameters
- Carefully selected based on gradient structure

**Benefits**:
- 50% less memory than AdamW
- 49.6% higher throughput for billion-parameter LLMs
- Maintains quality

---

## What Major LLMs Use

| Model | Optimizer | Peak LR | β₂ | Weight Decay | Warmup |
|-------|-----------|---------|-----|--------------|--------|
| GPT-3 175B | Adam | 6e-5 | 0.95 | 0.1 | 375 steps |
| LLaMA 1 | AdamW | 3e-4 (7B) / 1.5e-4 (65B) | 0.95 | 0.1 | 2000 steps |
| LLaMA 2 | AdamW | 3e-4 (7B) / 1.5e-4 (70B) | 0.95 | 0.1 | 2000 steps |
| Mistral 7B | AdamW (likely) | ~2e-4 | 0.95 | 0.1 | ~2000 |
| Falcon 40B | AdamW | 2e-4 | 0.95 | 0.1 | - |
| Qwen | AdamW | - | 0.95 | 0.1 | - |

**Pattern**: Everyone uses AdamW with β₂=0.95, weight_decay=0.1, cosine schedule.

---

## Practical Recommendations

### Recommended Configurations by Model Size

| Model Size | Peak LR | Min LR | β₁ | β₂ | Weight Decay | Warmup Steps | Grad Clip |
|------------|---------|--------|-----|-----|--------------|--------------|-----------|
| 125M | 6e-4 | 6e-5 | 0.9 | 0.95 | 0.1 | 2000 | 1.0 |
| 350M | 3e-4 | 3e-5 | 0.9 | 0.95 | 0.1 | 2000 | 1.0 |
| 1.3B | 2e-4 | 2e-5 | 0.9 | 0.95 | 0.1 | 2000 | 1.0 |
| 7B | 3e-4 | 3e-5 | 0.9 | 0.95 | 0.1 | 2000 | 1.0 |
| 13B | 3e-4 | 3e-5 | 0.9 | 0.95 | 0.1 | 2000 | 1.0 |
| 30B | 2e-4 | 2e-5 | 0.9 | 0.95 | 0.1 | 2000 | 1.0 |
| 65-70B | 1.5e-4 | 1.5e-5 | 0.9 | 0.95 | 0.1 | 2000 | 1.0 |
| 175B+ | 6e-5 | 6e-6 | 0.9 | 0.95 | 0.1 | 375 | 1.0 |

**Key patterns**:
- Peak LR decreases with model size (larger models need smaller LRs)
- β₂=0.95 is universal for LLM pre-training (not Adam's default 0.999)
- Weight decay=0.1 is standard across all sizes
- Min LR is typically 10% of peak LR
- Warmup steps ~2000 for most sizes, shorter for very large models

### Complete Pre-training Configuration

```python
# Standard configuration
optimizer = AdamW(
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps,
    eta_min=lr * 0.1  # 10% of peak
)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### For Fine-tuning

```python
# Lower learning rate, less weight decay
optimizer = AdamW(
    lr=2e-5,           # 10-100× smaller than pretraining
    betas=(0.9, 0.999),  # Default β₂ often fine
    weight_decay=0.01,   # Less regularization
)

# Shorter warmup, linear or cosine decay
warmup_steps = int(0.06 * total_steps)  # 6% warmup
```

### Debugging Tips

1. **Loss exploding**: Lower learning rate, increase warmup, check gradient clipping
2. **Loss plateauing**: Increase learning rate, check batch size
3. **Loss spiking**: Enable gradient clipping, check for data issues
4. **Memory issues**: Use 8-bit Adam, Adafactor, or shard optimizer state

---

## Future Directions

### Near-term (2025)

1. **Schedule-free methods**: Eliminating LR schedule hyperparameters
2. **Memory efficiency**: Adam-mini and variants becoming standard
3. **Second-order methods**: Sophia-like approaches for faster convergence

### Open Questions

1. **Optimal β₂ for different scales**: Why does 0.95 work better than 0.999?
2. **Warmup theory**: Mathematical understanding of why warmup helps
3. **Schedule design**: Principled approach to choosing schedules
4. **Optimizer-architecture co-design**: Different optimizers for different layer types?

---

## Sources

### Foundational Papers
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (ICLR 2015)
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) (ICLR 2019)
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) (ICLR 2017)

### Emerging Optimizers
- [Symbolic Discovery of Optimization Algorithms (Lion)](https://arxiv.org/abs/2302.06675)
- [Sophia: A Scalable Stochastic Second-order Optimizer](https://arxiv.org/abs/2305.14342)
- [The Road Less Scheduled (Schedule-Free)](https://arxiv.org/abs/2405.15682)
- [Adam-mini: Use Fewer Learning Rates To Gain More](https://arxiv.org/abs/2406.16793)

### LLM Training Details
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165) - Training configuration
- [LLaMA Paper](https://arxiv.org/abs/2302.13971) - AdamW settings
- [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288) - Training details

### Guides and Tutorials
- [AdamW: The Gold Standard Optimizer for Training LLMs](https://www.metriccoders.com/post/adamw-the-gold-standard-optimizer-for-training-llms)
- [Tips for LLM Pretraining](https://magazine.sebastianraschka.com/p/tips-for-llm-pretraining-and-evaluating-rms) - Sebastian Raschka
- [Dive into Deep Learning: LR Scheduling](https://d2l.ai/chapter_optimization/lr-scheduler.html)
