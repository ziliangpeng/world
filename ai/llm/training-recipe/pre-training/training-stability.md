# Training Stability for LLMs

Training instability is one of the most challenging aspects of large-scale LLM training. A single divergence event can waste millions of dollars in compute. This document covers the causes, detection, and prevention of training instabilities—knowledge accumulated from training thousands of models across the industry.

---

## The Stakes

| Model | Training Cost | Impact of Instability |
|-------|---------------|----------------------|
| 7B (8 GPUs, 1T tokens) | ~$100K | Days lost per restart |
| 70B (256 GPUs, 2T tokens) | ~$5M | Weeks lost, major budget impact |
| 175B (1000+ GPUs, 300B tokens) | ~$10M | Catastrophic if unrecovered |

**The problem**: At scale, small numerical issues compound. Gradient explosion, loss spikes, and divergence can occur suddenly after weeks of stable training. Prevention is cheaper than recovery.

---

## Types of Instability

### 1. Loss Spikes

Sudden, temporary increases in loss that the model (sometimes) recovers from:

```
Loss
  |
  |     ╱╲
  |    /  \
  |   /    ╲___
  |  /          ╲____
  | /                 ╲_____
  └─────────────────────────────→ Steps
```

**Characteristics**:
- Sharp increase (2-10× normal loss)
- May recover in 100-1000 steps
- More common later in training
- Often correlate with specific data batches

**Causes**:
- Data quality issues (corrupted samples, extreme values)
- Learning rate too high for current loss landscape
- Numerical precision issues (fp16 overflow/underflow)
- Gradient accumulation bugs

### 2. Gradient Explosion

Gradients grow unboundedly, causing weights to diverge:

```
Gradient Norm
  |
  |                    ↗ (explodes)
  |                   /
  |                  /
  |        ________/
  |_______/
  └─────────────────────────────→ Steps
```

**Characteristics**:
- Gradient norm increases exponentially
- Weights become NaN/Inf
- Training unrecoverable without rollback
- Often sudden onset

**Causes**:
- Learning rate too high
- Poor initialization
- Deep networks without proper normalization
- Attention logits exceeding fp16 range

### 3. Slow Divergence

Gradual degradation that's easy to miss:

```
Loss (zoomed in)
  |
  |    ___________
  |   /           \__
  |  /               \___
  | /                    \____
  |/                          \_____
  └─────────────────────────────────→ Steps
      (looks flat, but slowly rising)
```

**Characteristics**:
- Loss plateaus, then slowly increases
- Model quality degrades imperceptibly
- Often only detected in evaluation metrics
- Difficult to pinpoint onset

**Causes**:
- Optimizer instability (β₂ too high)
- Data distribution shift
- Accumulation of numerical errors

### 4. Training Collapse

Complete failure—model outputs become degenerate:

**Characteristics**:
- Model outputs constant values
- Attention becomes uniform
- Representations collapse to similar values
- Unrecoverable

**Causes**:
- Embedding initialization issues
- Extreme weight decay
- Architecture bugs
- NaN propagation

---

## Root Causes

### Numerical Precision

**fp16 (float16)**:
- Range: ~6×10⁻⁵ to ~6×10⁴
- Precision: 11 bits mantissa
- Problem: Attention logits often exceed safe range

```python
# Attention scores can overflow
# If q·k ≈ 100 (not unusual), exp(100) >> max_fp16

# Softmax in fp16 before scaling:
# logits = Q @ K.T  # Can be large
# exp(logits)       # Overflow!
```

**bf16 (bfloat16)**:
- Range: Same as fp32 (~1×10⁻³⁸ to ~3×10³⁸)
- Precision: 7 bits mantissa (less than fp16)
- Trade-off: Safer range, less precision

**Mixed Precision**:
```python
# Typical mixed precision strategy
# Forward/backward: bf16 (fast, wide range)
# Weights: fp32 master copy
# Optimizer: fp32 states
# Loss scaling: Dynamic for fp16

# PyTorch AMP
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
    loss = criterion(output, target)
```

### Attention Stability

The attention mechanism is numerically sensitive:

```python
# Potential issues in attention

# 1. Large dot products
# With d_head=128, random q,k: q·k ~ O(√d_head) ≈ 11
# After training, can be much larger

# 2. Softmax saturation
# softmax([100, 0, 0]) ≈ [1, 0, 0] - gradients vanish

# 3. fp16 overflow
# exp(89) ≈ 4×10³⁸ (fp32 max)
# exp(11) ≈ 6×10⁴ (fp16 max)
```

**Solutions**:
```python
# Solution 1: Scale attention by 1/√d
attention = (Q @ K.T) / sqrt(d_head)  # Standard, keeps variance ~1

# Solution 2: QK-Norm (Qwen, Gemma 2)
q = q / q.norm(dim=-1, keepdim=True)
k = k / k.norm(dim=-1, keepdim=True)
attention = (Q @ K.T) * learned_scale  # Bounded by 1

# Solution 3: Attention logit capping (PaLM, Gemini)
attention = (Q @ K.T) / sqrt(d_head)
attention = attention.clamp(-50, 50)  # Hard cap
```

### Gradient Issues

**Gradient explosion**:
```python
# Detection
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
if grad_norm > threshold:
    logger.warning(f"Large gradient: {grad_norm}")

# Prevention: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Gradient vanishing**:
- Less common in Transformers (residual connections help)
- Can occur in very deep networks without proper normalization
- Pre-norm architecture helps (normalize before rather than after)

### Weight and Activation Magnitudes

```python
# Monitor weight norms
def log_weight_norms(model):
    for name, param in model.named_parameters():
        norm = param.norm().item()
        if norm > 1000 or norm < 0.001:
            logger.warning(f"Extreme weight norm: {name} = {norm}")

# Monitor activation magnitudes
def activation_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            max_val = output.abs().max().item()
            if max_val > 1000:
                logger.warning(f"Large activation in {name}: {max_val}")
    return hook
```

---

## Prevention Strategies

### 1. Architecture Choices

**Pre-norm vs Post-norm**:
```python
# Post-norm (original Transformer, less stable)
x = x + attention(layernorm(x))  # Normalize after

# Pre-norm (GPT-2+, more stable)
x = x + attention(layernorm(x))  # Normalize before

# Pre-norm stabilizes by ensuring normalized inputs to attention/FFN
```

**RMSNorm** (vs LayerNorm):
```python
# LayerNorm: Normalize and shift
# More parameters, potentially less stable

# RMSNorm: Normalize only, no mean subtraction
# Simpler, more stable, standard since LLaMA
class RMSNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
```

**QK-Norm** (Qwen 2+, Gemma 2):
```python
# Normalize Q and K separately before attention
# Bounds attention logits regardless of training dynamics
q = self.q_norm(q)
k = self.k_norm(k)
attn = (q @ k.T) * self.scale  # scale is learned, ~1/√d initially
```

**Attention Logit Capping** (PaLM, Gemini):
```python
# Hard cap on attention logits
# Prevents exp() overflow even without QK-norm
attn_logits = torch.clamp(attn_logits, -50.0, 50.0)
```

### 2. Initialization

**Scaled initialization**:
```python
# Standard transformer init scales with depth
# For model with L layers:

# Residual projections (output of attention, FFN)
std = 0.02 / math.sqrt(2 * L)  # Scale down by √(2L)
nn.init.normal_(layer.weight, mean=0, std=std)

# Other weights
std = 0.02
nn.init.normal_(layer.weight, mean=0, std=std)
```

**μP (Maximal Update Parameterization)**:
```python
# Width-independent initialization and LR
# Allows hyperparameters to transfer across model sizes

# Key insight: Scale learning rates by 1/width
# Initialize certain weights by 1/√width
# Enables predictable scaling
```

### 3. Gradient Clipping

**Global norm clipping** (standard):
```python
# Clip total gradient norm across all parameters
# Most common: max_norm = 1.0

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Per-layer clipping**:
```python
# Sometimes helps for very deep networks
for param in model.parameters():
    if param.grad is not None:
        torch.nn.utils.clip_grad_norm_([param], max_norm=1.0)
```

**Adaptive clipping** (experimental):
```python
# Adjust clip threshold based on recent gradient history
# Used in some specialized settings
```

### 4. Learning Rate Management

**Warmup is crucial**:
```python
# Linear warmup
def lr_warmup(step, warmup_steps, peak_lr):
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    return peak_lr

# Typical warmup: 2000 steps for pretraining
# Longer warmup (5-10% of training) for larger models
```

**Learning rate decay**:
```python
# Cosine decay prevents late-training instability
def cosine_decay(step, total_steps, peak_lr, min_lr):
    if step < warmup_steps:
        return warmup(step)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### 5. Mixed Precision Best Practices

**bf16 preferred over fp16**:
```python
# bf16: Wider range, less precision
# Better for training stability
# Standard since LLaMA

with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
```

**fp32 for sensitive operations**:
```python
# Keep certain operations in fp32
# - Softmax (especially attention)
# - Layer normalization
# - Loss computation

class StableAttention(nn.Module):
    def forward(self, q, k, v):
        # Compute in bf16
        attn = (q @ k.T) / math.sqrt(self.head_dim)

        # Softmax in fp32 for stability
        attn = attn.float().softmax(dim=-1).to(q.dtype)

        return attn @ v
```

---

## Monitoring and Detection

### Key Metrics to Track

```python
class TrainingMonitor:
    def __init__(self):
        self.loss_history = []
        self.grad_norm_history = []

    def log_step(self, loss, model):
        # Loss tracking
        self.loss_history.append(loss)

        # Gradient norm
        grad_norm = self._compute_grad_norm(model)
        self.grad_norm_history.append(grad_norm)

        # Detect anomalies
        self._check_loss_spike(loss)
        self._check_grad_explosion(grad_norm)
        self._check_weight_norms(model)

    def _check_loss_spike(self, loss, threshold=3.0):
        if len(self.loss_history) > 100:
            recent_mean = np.mean(self.loss_history[-100:-1])
            if loss > recent_mean * threshold:
                alert(f"Loss spike: {loss} vs mean {recent_mean}")

    def _check_grad_explosion(self, grad_norm, threshold=100.0):
        if grad_norm > threshold:
            alert(f"Gradient explosion: {grad_norm}")

    def _check_weight_norms(self, model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                alert(f"NaN in weights: {name}")
            if torch.isinf(param).any():
                alert(f"Inf in weights: {name}")
```

### Recommended Monitoring Dashboard

| Metric | Frequency | Alert Threshold |
|--------|-----------|-----------------|
| Training loss | Every step | >3× recent mean |
| Validation loss | Every 1K steps | Increasing for 3 checks |
| Gradient norm | Every step | >100 |
| Weight norms | Every 100 steps | >1000 or <0.001 |
| Activation max | Every 100 steps | >1000 |
| Learning rate | Every step | Sanity check |
| NaN count | Every step | >0 |

---

## Recovery Procedures

### 1. Automatic Recovery from Checkpoints

```python
class CheckpointManager:
    def __init__(self, save_dir, keep_n=5):
        self.save_dir = save_dir
        self.keep_n = keep_n
        self.checkpoints = []

    def save(self, step, model, optimizer, scheduler):
        path = f"{self.save_dir}/ckpt_{step}.pt"
        torch.save({
            'step': step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, path)

        self.checkpoints.append(path)
        # Keep only N most recent
        while len(self.checkpoints) > self.keep_n:
            oldest = self.checkpoints.pop(0)
            os.remove(oldest)

    def recover_from_spike(self, model, optimizer, scheduler):
        """Rollback to previous checkpoint after instability."""
        if len(self.checkpoints) < 2:
            raise RuntimeError("Not enough checkpoints for recovery")

        # Load second-to-last checkpoint
        checkpoint = torch.load(self.checkpoints[-2])
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        return checkpoint['step']
```

### 2. Recovery Strategies

**Option 1: Simple rollback**
```python
# Rollback to last good checkpoint, continue training
# Works for transient spikes

step = checkpoint_manager.recover_from_spike(model, optimizer, scheduler)
# Continue training from step
```

**Option 2: Rollback + reduce LR**
```python
# Rollback and reduce learning rate
# For persistent instability

step = checkpoint_manager.recover_from_spike(model, optimizer, scheduler)
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.5
# Continue with lower LR
```

**Option 3: Skip bad batch**
```python
# Skip the problematic batch, continue
# Works when specific data causes spikes

def safe_training_step(batch, model, optimizer):
    try:
        loss = model(batch)
        if torch.isnan(loss) or loss > loss_threshold:
            logger.warning(f"Skipping bad batch, loss={loss}")
            return None
        loss.backward()
        optimizer.step()
        return loss.item()
    except RuntimeError as e:
        logger.error(f"Training step failed: {e}")
        return None
```

**Option 4: Data investigation**
```python
# Log problematic batches for investigation
def investigate_spike(batch, step):
    # Save batch data
    torch.save(batch, f"debug/spike_batch_{step}.pt")

    # Log batch statistics
    for key, tensor in batch.items():
        logger.info(f"{key}: min={tensor.min()}, max={tensor.max()}, "
                   f"mean={tensor.mean()}, std={tensor.std()}")
```

### 3. Checkpointing Strategy

| Training Phase | Checkpoint Frequency | Rationale |
|---------------|---------------------|-----------|
| Warmup | Every 100 steps | High instability risk |
| Early training | Every 500 steps | Moderate risk |
| Mid training | Every 1000 steps | Lower risk |
| Late training | Every 500 steps | Instability can return |

**Storage strategy**:
- Keep 5 most recent checkpoints (rolling)
- Keep milestone checkpoints (every 10K steps)
- Keep checkpoints with best validation loss

---

## Case Studies

### GPT-3 (OpenAI)

From the paper:
> "During training we observed occasional instabilities... loss spikes occurred roughly 1-2 times per 1000 steps."

**Their approach**:
- Lowered learning rate from spike checkpoint
- Skipped ~100-500 data batches ahead
- Restarted training

### PaLM (Google)

From the paper:
> "We found that the model could be trained stably... attention logits soft capping was necessary."

**Key technique**: Attention logit capping at 50

### BLOOM (BigScience)

From the training chronicles:
> "Loss spikes occurred 4 times during 176B training..."

**Their solutions**:
- Embedding layernorm
- Gradient checkpointing
- Careful mixed-precision configuration

### LLaMA (Meta)

From the paper:
> "No training instabilities reported"

**Their recipe**:
- RMSNorm (not LayerNorm)
- Pre-norm architecture
- bf16 training
- Standard gradient clipping (1.0)

---

## Best Practices Summary

### Before Training

1. **Architecture**: Use pre-norm, RMSNorm, consider QK-norm for large models
2. **Initialization**: Scale by depth, use μP for hyperparameter transfer
3. **Mixed precision**: Prefer bf16 over fp16, fp32 for softmax
4. **Warmup**: At least 2000 steps, longer for larger models

### During Training

1. **Monitor**: Loss, grad norm, weight norms, activation magnitudes
2. **Clip gradients**: max_norm=1.0 is standard
3. **Checkpoint frequently**: Every 500-1000 steps
4. **Alert on anomalies**: Automated detection for spikes

### After Instability

1. **Don't panic**: Some spikes are recoverable
2. **Rollback**: To checkpoint before spike
3. **Reduce LR**: If instability persists
4. **Investigate data**: Bad batches can cause issues

---

## Sources

### Foundational Research
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) - Pre-norm analysis
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) - RMSNorm paper
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466) - μP

### Training Reports
- [GPT-3: Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - Training details
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) - Attention capping
- [BLOOM: A 176B-Parameter Open-Access Multilingual LLM](https://arxiv.org/abs/2211.05100) - Training chronicles

### Practical Guides
- [Training Stability in Deep Learning](https://www.microsoft.com/en-us/research/publication/training-stability-in-deep-learning/) - Microsoft Research
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) - NVIDIA
- [Tips for LLM Pretraining](https://magazine.sebastianraschka.com/p/tips-for-llm-pretraining-and-evaluating-rms) - Sebastian Raschka
