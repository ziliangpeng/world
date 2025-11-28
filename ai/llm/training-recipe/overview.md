# LLM Training Recipe: Overview

This document provides a map of the complete LLM training pipeline—from raw data to deployed model. Each component documented in this folder represents a critical stage, and understanding how they connect is essential for building production-quality language models.

---

## The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PRE-TRAINING                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │     Data     │───▶│   Training   │───▶│  Distributed │───▶│   Training   │  │
│  │ Preparation  │    │   (Optim +   │    │   Training   │    │  Stability   │  │
│  │              │    │ Scaling Laws)│    │              │    │              │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │                   │           │
│         ▼                   ▼                   ▼                   ▼           │
│  [data-preparation]   [optimizers]      [distributed-      [training-          │
│                       [scaling-laws]     training]          stability]         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │   Base Model    │
                               │  (Pre-trained)  │
                               └────────┬────────┘
                                        │
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              POST-TRAINING                                       │
│                                        │                                         │
│                     ┌──────────────────┼──────────────────┐                     │
│                     ▼                  ▼                  ▼                     │
│              ┌────────────┐     ┌────────────┐     ┌────────────┐              │
│              │    SFT     │     │  Synthetic │     │Distillation│              │
│              │            │◀────│    Data    │────▶│            │              │
│              └─────┬──────┘     └────────────┘     └────────────┘              │
│                    │            [synthetic-data]   [distillation]               │
│                    ▼                                                            │
│  [supervised-fine-tuning]                                                       │
│                    │                                                            │
│         ┌─────────┴─────────┐                                                   │
│         ▼                   ▼                                                   │
│  ┌────────────┐      ┌────────────┐                                            │
│  │    RLHF    │  OR  │    DPO     │                                            │
│  │            │      │ (& variants)│                                            │
│  └─────┬──────┘      └─────┬──────┘                                            │
│        │                   │                                                    │
│  [rlhf]             [direct-preference]                                         │
│        │                   │                                                    │
│        └─────────┬─────────┘                                                    │
│                  ▼                                                              │
│           ┌────────────┐                                                        │
│           │   Safety   │                                                        │
│           │ Alignment  │                                                        │
│           └─────┬──────┘                                                        │
│                 │                                                               │
│           [safety-alignment]                                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │ Aligned Model   │
                               │   (Deployed)    │
                               └────────┬────────┘
                                        │
                                        ▼
                               ┌─────────────────┐
                               │   Evaluation    │
                               │  [benchmarks]   │
                               └─────────────────┘
```

---

## Stage 1: Pre-Training

Pre-training creates the foundation—a model that understands language and has broad knowledge. This is the most expensive stage, often costing millions of dollars.

### Components

| Component | Purpose | Key Decisions |
|-----------|---------|---------------|
| [Data Preparation](pre-training/data-preparation.md) | Create training corpus | Filtering, deduplication, mixing ratios |
| [Optimizers](pre-training/optimizers.md) | Update model weights | AdamW, learning rate schedule |
| [Scaling Laws](pre-training/scaling-laws.md) | Allocate compute | Model size vs data size |
| [Distributed Training](pre-training/distributed-training.md) | Scale to many GPUs | Parallelism strategy |
| [Training Stability](pre-training/training-stability.md) | Prevent failures | Architecture choices, monitoring |
| [Continued Pre-training](pre-training/continued-pretraining.md) | Domain adaptation | Replay data, learning rate, forgetting |

### How They Connect

```
1. SCALING LAWS determine how much data and how large a model
   │
   ├──▶ Data budget: 20× parameters (Chinchilla) or more (inference-optimal)
   │
   └──▶ Model size: Given compute budget, optimal N and D

2. DATA PREPARATION creates the training corpus
   │
   ├──▶ Filter and deduplicate raw data (web crawls, books, code)
   │
   ├──▶ Mix domains according to target distribution
   │
   └──▶ Tokenize and shard for distributed loading

3. DISTRIBUTED TRAINING determines how to use hardware
   │
   ├──▶ Choose parallelism: DP, TP, PP based on model/cluster size
   │
   ├──▶ Configure ZeRO stage for memory efficiency
   │
   └──▶ Set batch size and gradient accumulation

4. OPTIMIZERS configure the learning process
   │
   ├──▶ AdamW with β₂=0.95, weight_decay=0.1
   │
   ├──▶ Cosine LR schedule with warmup
   │
   └──▶ Gradient clipping (max_norm=1.0)

5. TRAINING STABILITY prevents expensive failures
   │
   ├──▶ Architecture: Pre-norm, RMSNorm, QK-norm
   │
   ├──▶ Precision: bf16 with fp32 optimizer states
   │
   └──▶ Monitoring: Loss, grad norm, weight norms
```

### Typical Configuration

```python
# Pre-training configuration example
config = {
    # From scaling laws
    "model_params": 7e9,
    "training_tokens": 1e12,  # ~140 tokens/param

    # From data preparation
    "data_mix": {
        "web": 0.67,
        "code": 0.08,
        "books": 0.05,
        "scientific": 0.05,
        "wikipedia": 0.05,
    },

    # From distributed training
    "tensor_parallel": 8,
    "pipeline_parallel": 1,
    "data_parallel": 8,
    "zero_stage": 1,

    # From optimizers
    "optimizer": "AdamW",
    "lr": 3e-4,
    "betas": (0.9, 0.95),
    "weight_decay": 0.1,
    "warmup_steps": 2000,
    "lr_schedule": "cosine",

    # From training stability
    "precision": "bf16",
    "grad_clip": 1.0,
    "checkpoint_interval": 1000,
}
```

---

## Stage 2: Post-Training

Post-training transforms the base model into an assistant that follows instructions, aligns with human preferences, and avoids harmful outputs. This stage is cheaper but equally critical for usability.

### Components

| Component | Purpose | Key Decisions |
|-----------|---------|---------------|
| [Supervised Fine-Tuning](post-training/supervised-fine-tuning.md) | Learn to follow instructions | Data format, full FT vs LoRA |
| [RLHF](post-training/rlhf.md) | Optimize for human preference | Reward model quality, PPO tuning |
| [Direct Preference](post-training/direct-preference.md) | RL-free preference learning | DPO vs IPO vs KTO |
| [Test-Time Compute](post-training/test-time-compute.md) | Train reasoning models | RL for reasoning, GRPO, PRMs |
| [Safety Alignment](post-training/safety-alignment.md) | Avoid harmful outputs | Constitutional AI, red teaming |
| [Synthetic Data](post-training/synthetic-data.md) | Generate training data | Teacher quality, diversity |
| [Distillation](post-training/distillation.md) | Transfer to smaller models | Teacher-student gap |

### How They Connect

```
BASE MODEL
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ SYNTHETIC DATA generates training examples for all stages     │
│                                                               │
│  • Instruction data for SFT (Self-Instruct, Evol-Instruct)   │
│  • Preference pairs for RLHF/DPO                              │
│  • Constitutional critiques for safety                        │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ SFT teaches the model to follow instructions                  │
│                                                               │
│  Input: Base model + demonstration data                       │
│  Output: Model that produces assistant-like responses         │
│  Duration: ~1-3 epochs on 10K-100K examples                   │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ PREFERENCE OPTIMIZATION aligns with human values              │
│                                                               │
│  Option A: RLHF                                               │
│    1. Train reward model on preference data                   │
│    2. Optimize policy with PPO against reward model           │
│    Pros: Gold standard, well-understood                       │
│    Cons: Complex, expensive, 4 models needed                  │
│                                                               │
│  Option B: DPO (and variants)                                 │
│    1. Directly optimize on preference pairs                   │
│    Pros: Simple, stable, 2 models needed                      │
│    Cons: Offline only, may not match RLHF quality            │
│                                                               │
│  Most teams: Start with DPO, add RLHF if needed              │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ SAFETY ALIGNMENT ensures harmlessness                         │
│                                                               │
│  • Constitutional AI: Self-critique against principles        │
│  • Red teaming: Find and fix vulnerabilities                  │
│  • RLAIF: Scale safety feedback with AI                       │
│  • Can be integrated with RLHF/DPO or as separate stage      │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────────────────────┐
│ DISTILLATION (optional) creates smaller deployable models     │
│                                                               │
│  • Teacher: Aligned large model                               │
│  • Student: Smaller model (7B from 70B)                       │
│  • Method: Train on teacher outputs (black-box distillation)  │
│  • Quality: ~80-90% of teacher at ~10% cost                   │
└───────────────────────────────────────────────────────────────┘
    │
    ▼
ALIGNED MODEL
```

### Typical Post-Training Pipeline

```python
# Post-training pipeline example

# Stage 1: SFT
sft_config = {
    "base_model": "llama-2-7b",
    "data": "alpaca_52k + custom_demonstrations",
    "method": "full_finetune",  # or "lora"
    "lr": 2e-5,
    "epochs": 3,
}
sft_model = sft_train(base_model, sft_config)

# Stage 2: Preference Optimization (choose one)

# Option A: DPO (simpler)
dpo_config = {
    "model": sft_model,
    "preference_data": "ultrafeedback_60k",
    "beta": 0.1,
    "lr": 5e-7,
}
aligned_model = dpo_train(sft_model, dpo_config)

# Option B: RLHF (more complex)
rm_config = {
    "base_model": sft_model,
    "preference_data": "comparison_data_50k",
}
reward_model = train_reward_model(rm_config)

ppo_config = {
    "policy": sft_model,
    "reward_model": reward_model,
    "kl_coef": 0.1,
}
aligned_model = ppo_train(ppo_config)

# Stage 3: Safety (can be integrated or separate)
safety_config = {
    "model": aligned_model,
    "constitution": "anthropic_principles",
    "red_team_data": "harmful_prompts_10k",
}
safe_model = safety_train(aligned_model, safety_config)
```

---

## Stage 3: Evaluation

Evaluation happens throughout but is critical before deployment. Use [benchmarks](benchmarks.md) to measure capabilities and safety.

### Evaluation Points

```
Pre-training:
├── During: Validation loss, perplexity
└── After: MMLU, HellaSwag (knowledge/reasoning check)

Post-SFT:
├── MT-Bench (instruction following)
└── AlpacaEval (comparison to baseline)

Post-Alignment:
├── Chatbot Arena (human preference)
├── TruthfulQA (honesty)
└── HarmBench (safety)

Before Deployment:
├── Task-specific evals (your use case)
├── Red team evaluation
└── Bias and fairness audits
```

---

## Common Pipelines

### GPT-4 Style (Full Pipeline)

```
Pre-training (large scale) → SFT → RLHF → Safety RLHF
```

### LLaMA/Mistral Style (Open Source)

```
Pre-training (efficient) → SFT → DPO
```

### Phi Style (Synthetic Focus)

```
Synthetic pre-training data → Smaller model → SFT → DPO
```

### Distillation Pipeline

```
Large aligned model (teacher) → Generate responses → Train small model → Optional DPO
```

---

## Key Trade-offs

### Pre-training

| Decision | Trade-off |
|----------|-----------|
| Model size vs data | Larger model = better quality, more compute |
| Data quality vs quantity | High quality > large quantity |
| Training time | Longer training = better, diminishing returns |

### Post-training

| Decision | Trade-off |
|----------|-----------|
| SFT data size | More diverse > larger, quality matters most |
| RLHF vs DPO | RLHF = slight quality edge, DPO = simpler |
| Safety strictness | More safety = more over-refusal |

### Deployment

| Decision | Trade-off |
|----------|-----------|
| Model size | Larger = better, more expensive |
| Quantization | Lower bits = faster/cheaper, quality loss |
| Distillation | Smaller = faster, capability ceiling |

---

## Quick Reference: Which Doc to Read

| Question | Document |
|----------|----------|
| How much data do I need? | [scaling-laws.md](pre-training/scaling-laws.md) |
| How do I prepare training data? | [data-preparation.md](pre-training/data-preparation.md) |
| What optimizer settings? | [optimizers.md](pre-training/optimizers.md) |
| How to train on multiple GPUs? | [distributed-training.md](pre-training/distributed-training.md) |
| Why is my training unstable? | [training-stability.md](pre-training/training-stability.md) |
| How to adapt model to my domain? | [continued-pretraining.md](pre-training/continued-pretraining.md) |
| How to make model follow instructions? | [supervised-fine-tuning.md](post-training/supervised-fine-tuning.md) |
| Should I use RLHF or DPO? | [rlhf.md](post-training/rlhf.md), [direct-preference.md](post-training/direct-preference.md) |
| How to train reasoning models (o1-style)? | [test-time-compute.md](post-training/test-time-compute.md) |
| How to make model safe? | [safety-alignment.md](post-training/safety-alignment.md) |
| Can I generate my own data? | [synthetic-data.md](post-training/synthetic-data.md) |
| How to make a smaller model? | [distillation.md](post-training/distillation.md) |
| How do I evaluate my model? | [benchmarks.md](benchmarks.md) |

---

## Further Reading

For production deployment patterns:
- [LLMOps: Production Patterns](llmops/production-patterns.md) - Evals, RAG, guardrails, caching, feedback loops

For inference optimization after training, see:
- [Quantization](../inference-optimization/quantization.md)
- [Speculative Decoding](../inference-optimization/speculative-decoding.md)
- [KV Cache Optimization](../inference-optimization/kv-cache-optimization.md)
- [Batching Strategies](../inference-optimization/batching-strategies.md)

For external learning resources (blogs, courses, papers):
- [Curated Resources](resources.md)
