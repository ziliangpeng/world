# Continued Pre-training (Domain Adaptation)

Continued pre-training bridges the gap between generic pre-training and task-specific fine-tuning. It adapts a pre-trained model to a new domain by training on domain-specific data while preserving general capabilities.

---

## When to Use Continued Pre-training

### The Spectrum of Adaptation

```
Pre-training ──────► Continued Pre-training ──────► Fine-tuning
(from scratch)       (domain adaptation)            (task-specific)

Data: Web crawl      Data: Domain corpus           Data: Task examples
Tokens: 1-15T        Tokens: 50B-500B              Tokens: 1M-100M
LR: 3e-4             LR: 1e-5 to 5e-5              LR: 1e-6 to 2e-5
Objective: LM        Objective: LM                 Objective: Task loss
```

### Decision Framework

| Situation | Recommended Approach |
|-----------|---------------------|
| Model lacks domain knowledge entirely | Continued pre-training |
| Model knows domain but wrong style | Fine-tuning |
| Need to add new language | Continued pre-training |
| Domain terminology is specialized | Continued pre-training |
| Only have task examples (no raw text) | Fine-tuning |
| Have >10B domain tokens | Continued pre-training |
| Have <1B domain tokens | Fine-tuning (or SFT) |

### Use Cases

**1. Domain Expertise**
- Medical: PubMed, clinical notes → BioMedLLM, MedPaLM
- Legal: Case law, contracts → LegalBERT successors
- Finance: SEC filings, news → BloombergGPT
- Code: Repositories, documentation → CodeLlama

**2. Language Adaptation**
- English model → Japanese: Train on Japanese web data
- Multilingual → Low-resource language: Focus on target language

**3. Knowledge Freshness**
- Update model on recent events/data
- Add knowledge post-cutoff date

**4. Internal Knowledge**
- Company documents, wikis, codebases
- Proprietary data not in pre-training

---

## The Forgetting Problem

### Catastrophic Forgetting

Training on domain-specific data causes the model to forget general knowledge:

```
Before continued pre-training:
- General knowledge: 85%
- Domain knowledge: 20%

Naive continued pre-training (domain-only):
- General knowledge: 45% ↓ (forgotten!)
- Domain knowledge: 75% ↑

Proper continued pre-training (with replay):
- General knowledge: 80% (mostly preserved)
- Domain knowledge: 70%
```

### Why It Happens

1. **Distribution shift**: Domain data has different statistics than pre-training data
2. **Parameter overwriting**: Weights optimized for domain lose general patterns
3. **No rehearsal**: Model doesn't "practice" general knowledge

### Mitigation Strategies

| Strategy | Description | Effectiveness |
|----------|-------------|---------------|
| Data replay | Mix domain data with general data | High |
| Lower learning rate | Smaller updates preserve weights | Medium |
| Regularization | Constrain weight changes | Medium |
| LoRA/Adapters | Train only new parameters | High (but ceiling) |
| Elastic Weight Consolidation | Penalize changes to important weights | Medium |

---

## Data Mixing (Replay)

### The Core Technique

Mix domain-specific data with samples from pre-training distribution:

```python
def create_continued_pretraining_mix(domain_data, replay_data, domain_ratio=0.7):
    """
    Mix domain data with replay data to prevent forgetting.

    Args:
        domain_data: Your domain-specific corpus
        replay_data: Sample from original pre-training distribution
        domain_ratio: Fraction of domain data (0.5-0.9)

    Returns:
        Mixed dataset
    """
    total_tokens = len(domain_data)
    replay_tokens = int(total_tokens * (1 - domain_ratio) / domain_ratio)

    mixed = interleave(
        domain_data,
        sample(replay_data, replay_tokens)
    )
    return mixed
```

### Recommended Mix Ratios

| Scenario | Domain Ratio | Replay Ratio | Notes |
|----------|-------------|--------------|-------|
| Heavy domain shift | 50% | 50% | Medical, legal |
| Moderate shift | 70% | 30% | Code, scientific |
| Light shift | 90% | 10% | Style/format only |
| No general use needed | 100% | 0% | Domain-only deployment |

### What to Use for Replay

**Best**: Original pre-training data (if available)
- Exact distribution match
- Preserves learned patterns

**Good**: High-quality general corpus
- Wikipedia, books, curated web
- Similar to pre-training distribution

**Acceptable**: Synthetic replay
- Generate with frozen model
- Captures model's own distribution

```python
# Synthetic replay generation
def generate_synthetic_replay(model, num_samples, max_length=512):
    """Generate samples from frozen model as replay data."""
    model.eval()
    replay_samples = []

    # Use diverse prompts
    prompts = load_diverse_prompts()  # News, wiki, code, etc.

    for prompt in random.sample(prompts, num_samples):
        with torch.no_grad():
            sample = model.generate(prompt, max_length=max_length)
            replay_samples.append(sample)

    return replay_samples
```

### Dynamic Mixing

Adjust ratio during training:

```python
def dynamic_mix_schedule(step, total_steps):
    """
    Start with more replay, gradually increase domain ratio.

    Early: 50% domain, 50% replay (stabilize)
    Late: 80% domain, 20% replay (specialize)
    """
    progress = step / total_steps

    if progress < 0.1:
        return 0.5  # 50% domain
    elif progress < 0.5:
        return 0.5 + 0.3 * (progress - 0.1) / 0.4  # Ramp to 80%
    else:
        return 0.8  # 80% domain
```

---

## Learning Rate Strategies

### The Key Insight

Continued pre-training requires **much lower learning rates** than initial pre-training:
- Pre-training peak LR: 3e-4 (for 7B model)
- Continued pre-training: 1e-5 to 5e-5 (10-30× lower)

### Schedule Options

**Option 1: Re-warmup + Cosine Decay**

```python
def continued_pretraining_schedule(step, total_steps, peak_lr=3e-5):
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    if step < warmup_steps:
        # Linear warmup from ~0 to peak
        return peak_lr * step / warmup_steps
    else:
        # Cosine decay to 10% of peak
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        min_lr = peak_lr * 0.1
        return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

**Option 2: Continue from Pre-training LR**

If you have access to the pre-training LR at end of training:

```python
def resume_schedule(step, total_steps, pretrain_final_lr=3e-5):
    """Continue from where pre-training left off."""
    # Start at pre-training's final LR
    start_lr = pretrain_final_lr
    min_lr = start_lr * 0.1

    progress = step / total_steps
    return min_lr + 0.5 * (start_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

**Option 3: Constant Low LR**

Simple and effective for shorter runs:

```python
def constant_lr_schedule(step, lr=2e-5):
    """Constant low learning rate."""
    warmup_steps = 1000
    if step < warmup_steps:
        return lr * step / warmup_steps
    return lr
```

### Learning Rate Guidelines

| Base Model Size | Recommended Peak LR | Warmup Steps |
|-----------------|---------------------|--------------|
| 1-3B | 5e-5 | 1000 |
| 7B | 3e-5 | 1000 |
| 13B | 2e-5 | 1000 |
| 30B+ | 1e-5 | 2000 |
| 70B+ | 5e-6 | 2000 |

---

## Training Configuration

### Full Configuration Example

```python
# Continued pre-training configuration for 7B model
config = {
    # Data
    "domain_data": "medical_corpus_100B_tokens",
    "replay_data": "redpajama_sample_30B_tokens",
    "domain_ratio": 0.7,

    # Model
    "base_model": "llama-2-7b",
    "precision": "bf16",

    # Optimization
    "optimizer": "AdamW",
    "lr": 3e-5,
    "min_lr": 3e-6,
    "betas": (0.9, 0.95),
    "weight_decay": 0.1,
    "grad_clip": 1.0,

    # Schedule
    "warmup_steps": 1000,
    "total_steps": 50000,  # ~100B tokens with batch size 2M
    "lr_schedule": "cosine",

    # Training
    "batch_size": 2_000_000,  # tokens per batch
    "sequence_length": 4096,
    "gradient_accumulation": 32,
}
```

### Parameter-Efficient Alternative (LoRA)

When compute-constrained or forgetting is severe:

```python
# LoRA configuration for continued pre-training
lora_config = {
    "r": 64,                    # Rank (higher than fine-tuning)
    "lora_alpha": 128,          # Scaling factor
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
}

# Can use higher LR with LoRA
lora_lr = 2e-4  # 10× higher than full continued pre-training
```

**LoRA trade-offs for continued pre-training**:
- ✓ Much less forgetting (base weights frozen)
- ✓ Lower compute cost
- ✗ Capacity ceiling (can't fully absorb domain)
- ✗ Inference overhead (unless merged)

---

## Evaluation During Training

### What to Monitor

**1. Domain Performance** (primary goal)
```python
def eval_domain(model, domain_eval_set):
    """Measure improvement on domain-specific data."""
    perplexity = compute_perplexity(model, domain_eval_set)
    domain_accuracy = run_domain_benchmark(model)  # e.g., MedQA for medical
    return {"domain_ppl": perplexity, "domain_acc": domain_accuracy}
```

**2. General Performance** (forgetting check)
```python
def eval_general(model, general_eval_sets):
    """Measure retention of general capabilities."""
    return {
        "mmlu": evaluate_mmlu(model),
        "hellaswag": evaluate_hellaswag(model),
        "general_ppl": compute_perplexity(model, general_eval_set),
    }
```

**3. Combined Score**
```python
def combined_score(domain_score, general_score, domain_weight=0.6):
    """Weighted combination of domain and general performance."""
    return domain_weight * domain_score + (1 - domain_weight) * general_score
```

### Evaluation Schedule

```python
eval_schedule = {
    "domain_eval": "every_1000_steps",    # Frequent domain checks
    "general_eval": "every_5000_steps",    # Less frequent general checks
    "full_benchmark": "end_of_training",   # Comprehensive final eval
}
```

### When to Stop

```python
def should_stop_training(metrics_history):
    """
    Stop when:
    1. Domain performance plateaus
    2. General performance drops significantly
    """
    recent = metrics_history[-5:]

    # Domain plateau check
    domain_improvement = recent[-1]["domain_ppl"] - recent[0]["domain_ppl"]
    if domain_improvement > -0.01:  # Less than 1% improvement
        return True, "domain_plateau"

    # Forgetting check
    general_drop = metrics_history[0]["mmlu"] - recent[-1]["mmlu"]
    if general_drop > 0.05:  # More than 5% drop
        return True, "excessive_forgetting"

    return False, None
```

---

## Case Studies

### CodeLlama: Code Domain Adaptation

```
Base: LLaMA 2 (7B, 13B, 34B)
Domain data: 500B code tokens
Result: 70B → significantly better coding with preserved general ability

Key decisions:
- Used infilling objective (not just next-token)
- Long context training (16K → 100K)
- Code-specific tokenizer improvements
```

### BioMedLLM: Medical Adaptation

```
Base: PubMedBERT-style or LLaMA
Domain data: PubMed abstracts + clinical text
Result: SOTA on MedQA, BioASQ with reasonable general performance

Key decisions:
- Heavy replay (50% general) due to distribution shift
- Domain vocabulary expansion
- Careful filtering of PHI (Protected Health Information)
```

### Bloomberg GPT: Finance Adaptation

```
Base: Trained from scratch with mixed data
Domain data: 363B finance tokens (51% of training)
General data: 345B general tokens (49%)
Result: Best-in-class finance, competitive general

Key insight: Mixed from the start, not continued pre-training
- Avoided forgetting entirely
- But: Required training from scratch
```

---

## Common Pitfalls

### 1. Learning Rate Too High

**Symptom**: Loss spikes, degraded general performance
**Fix**: Start with 1/10 of pre-training LR, adjust up if stable

### 2. No Replay Data

**Symptom**: Massive forgetting on MMLU, general benchmarks
**Fix**: Add 20-50% replay data, even synthetic

### 3. Domain Data Too Small

**Symptom**: Overfitting to domain, poor generalization
**Fix**: If <10B tokens, consider fine-tuning instead

### 4. Training Too Long

**Symptom**: Diminishing returns, increasing forgetting
**Fix**: Monitor eval metrics, stop when plateauing

### 5. Wrong Tokenizer

**Symptom**: Poor handling of domain terminology
**Fix**: Consider vocabulary extension (advanced technique)

```python
# Vocabulary extension (use with caution)
def extend_tokenizer(tokenizer, new_tokens, model):
    """Add domain-specific tokens to vocabulary."""
    num_added = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new embeddings (average of subword pieces)
    with torch.no_grad():
        for token in new_tokens:
            subwords = tokenizer.tokenize(token)
            subword_ids = tokenizer.convert_tokens_to_ids(subwords)
            avg_embedding = model.embeddings.weight[subword_ids].mean(dim=0)
            new_id = tokenizer.convert_tokens_to_ids(token)
            model.embeddings.weight[new_id] = avg_embedding
```

---

## Practical Checklist

### Before Training

- [ ] Sufficient domain data (>10B tokens recommended)
- [ ] Replay data prepared (20-50% of total)
- [ ] Evaluation sets ready (domain + general)
- [ ] Learning rate selected (much lower than pre-training)
- [ ] Forgetting budget defined (acceptable general capability drop)

### During Training

- [ ] Monitor domain perplexity (should decrease)
- [ ] Monitor general benchmarks (should stay stable)
- [ ] Check for loss spikes (may need lower LR)
- [ ] Validate data mixing is working

### After Training

- [ ] Full benchmark evaluation
- [ ] Compare to base model on domain tasks
- [ ] Verify acceptable general capability retention
- [ ] Test on held-out domain examples

---

## Summary

Continued pre-training is the right choice when:
- You have substantial domain data (>10B tokens)
- The model lacks foundational domain knowledge
- Fine-tuning alone underperforms

Key principles:
1. **Lower learning rate**: 10-30× less than pre-training
2. **Replay data**: 20-50% general data to prevent forgetting
3. **Monitor both**: Domain improvement AND general retention
4. **Stop early**: Diminishing returns set in; don't overtrain

The goal is not maximum domain performance—it's the best domain performance you can achieve while maintaining acceptable general capabilities.
