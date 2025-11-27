# Direct Preference Optimization

Direct Preference Optimization (DPO) and its variants represent a paradigm shift in alignment: achieving RLHF-quality results without the complexity of reinforcement learning. Instead of training a reward model and running PPO, these methods directly optimize the language model on preference data. This simplification has made preference learning accessible and practical.

---

## The Problem with RLHF

RLHF is effective but complex:

| Challenge | Impact |
|-----------|--------|
| Four models | Policy, Reference, Reward, Value function |
| Training instability | PPO sensitive to hyperparameters |
| Reward hacking | Model exploits reward model artifacts |
| Computational cost | 4× resources vs SFT |
| Implementation complexity | Multiple interacting components |

**The question**: Can we get RLHF benefits without RL?

---

## Historical Evolution

### Phase 1: The DPO Breakthrough (2023)

**[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)** (Stanford, May 2023)

The key insight: The optimal RLHF policy has a closed-form solution. Instead of training a reward model then running PPO, we can directly optimize the policy.

**Mathematical derivation**:

Starting from RLHF objective:
```
max_π E[r(x,y)] - β × KL[π(y|x) || π_ref(y|x)]
```

The optimal policy is:
```
π*(y|x) = (1/Z(x)) × π_ref(y|x) × exp(r(x,y)/β)
```

Rearranging for the reward:
```
r(x,y) = β × log(π*(y|x) / π_ref(y|x)) + β × log(Z(x))
```

Substituting into Bradley-Terry preference model:
```
p(y_w > y_l | x) = σ(r(x,y_w) - r(x,y_l))
                 = σ(β × log(π(y_w|x)/π_ref(y_w|x)) - β × log(π(y_l|x)/π_ref(y_l|x)))
```

**DPO Loss**:
```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    Direct Preference Optimization loss.

    L_DPO = -log σ(β × (log π(y_w|x) - log π_ref(y_w|x))
                  - β × (log π(y_l|x) - log π_ref(y_l|x)))
    """
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps

    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -F.logsigmoid(logits).mean()

    return loss
```

**Result**: DPO matched or exceeded PPO on summarization and dialogue tasks.

### Phase 2: Improvements and Variants (2023-2024)

**[IPO (Identity Preference Optimization)](https://arxiv.org/abs/2310.12036)** (October 2023)

Fixed DPO's tendency to overfit by regularizing:

```python
def ipo_loss(chosen_logratios, rejected_logratios, beta=0.1):
    """IPO adds regularization to prevent overfitting."""
    logits = chosen_logratios - rejected_logratios

    # IPO loss: (logits - 1/(2β))²
    loss = (logits - 1 / (2 * beta)) ** 2
    return loss.mean()
```

**Key insight**: DPO can push the chosen/rejected gap too wide, IPO bounds it.

**[cDPO (Conservative DPO)](https://ericmitchell.ai/cdpo.pdf)** (2023)

Added label smoothing for noisy preferences:

```python
def cdpo_loss(logits, label_smoothing=0.1):
    """Conservative DPO with label smoothing."""
    # Soft targets instead of hard 0/1
    target_pos = 1 - label_smoothing
    target_neg = label_smoothing

    loss = -target_pos * F.logsigmoid(logits) - target_neg * F.logsigmoid(-logits)
    return loss.mean()
```

### Phase 3: Beyond Pairwise Comparisons (2024)

**[KTO (Kahneman-Tversky Optimization)](https://arxiv.org/abs/2402.01306)** (February 2024)

Doesn't require pairwise comparisons—just "good" or "bad" labels:

```python
def kto_loss(policy_logps, ref_logps, is_chosen, beta=0.1):
    """
    KTO: Binary preference optimization.
    Inspired by prospect theory (asymmetric loss/gain).
    """
    logratios = policy_logps - ref_logps

    # Separate handling for chosen vs rejected
    chosen_mask = is_chosen
    rejected_mask = ~is_chosen

    # KTO uses asymmetric weighting
    chosen_losses = 1 - F.sigmoid(beta * logratios[chosen_mask])
    rejected_losses = F.sigmoid(beta * logratios[rejected_mask])

    # Reference point: average logratio
    ref_point = logratios.mean().detach()

    loss = chosen_losses.mean() + rejected_losses.mean()
    return loss
```

**Advantage**: Can use existing thumbs-up/down data, doesn't need paired comparisons.

**[ORPO (Odds Ratio Preference Optimization)](https://arxiv.org/abs/2403.07691)** (March 2024)

Combined SFT and preference optimization in one stage:

```python
def orpo_loss(input_ids, chosen_ids, rejected_ids, policy, beta=0.1):
    """
    ORPO: SFT + preference in single loss.
    No reference model needed!
    """
    # SFT loss on chosen response
    sft_loss = cross_entropy(policy(input_ids), chosen_ids)

    # Odds ratio penalty
    chosen_logps = policy.log_prob(chosen_ids)
    rejected_logps = policy.log_prob(rejected_ids)

    log_odds = (chosen_logps - rejected_logps) - (
        torch.log(1 - torch.exp(chosen_logps)) -
        torch.log(1 - torch.exp(rejected_logps))
    )

    odds_ratio_loss = -F.logsigmoid(log_odds).mean()

    return sft_loss + beta * odds_ratio_loss
```

**Advantage**: Single training stage, no reference model.

**[SimPO (Simple Preference Optimization)](https://arxiv.org/abs/2405.14734)** (May 2024)

Simplified DPO by removing reference model:

```python
def simpo_loss(chosen_logps, rejected_logps, beta=2.0, gamma=0.5):
    """
    SimPO: DPO without reference model.
    Uses length-normalized log probabilities.
    """
    # Length normalization
    chosen_logps_norm = chosen_logps / chosen_length
    rejected_logps_norm = rejected_logps / rejected_length

    # Target margin
    logits = beta * (chosen_logps_norm - rejected_logps_norm) - gamma

    return -F.logsigmoid(logits).mean()
```

**Advantage**: No reference model needed, simpler implementation.

### Phase 4: Online and Iterative Methods (2024)

**Online DPO variants**:

Instead of fixed preference data, generate and label on-the-fly:

```python
def online_dpo_iteration(policy, reward_model, prompts):
    # Generate pairs from current policy
    responses_a = policy.generate(prompts)
    responses_b = policy.generate(prompts)

    # Label with reward model (or AI judge)
    chosen, rejected = label_preferences(responses_a, responses_b, reward_model)

    # DPO update
    loss = dpo_loss(policy, chosen, rejected)
    return loss
```

**[Self-Play Preference Optimization (SPPO)](https://arxiv.org/abs/2405.00675)**

Iterative self-improvement using DPO.

---

## Comparison of Methods

| Method | Needs Reference | Needs Pairs | Stages | Key Advantage |
|--------|-----------------|-------------|--------|---------------|
| **DPO** | Yes | Yes | 2 (SFT → DPO) | Simplest RL-free |
| **IPO** | Yes | Yes | 2 | Prevents overfitting |
| **KTO** | Yes | No | 2 | Binary labels only |
| **ORPO** | No | Yes | 1 | Single-stage |
| **SimPO** | No | Yes | 2 | Simpler than DPO |
| **PPO** | Yes | Yes | 3 (SFT → RM → PPO) | Gold standard |

### When to Use What

| Scenario | Recommended Method |
|----------|-------------------|
| Standard alignment, pairwise data | DPO |
| Noisy preference labels | cDPO or IPO |
| Only thumbs up/down data | KTO |
| Want single training stage | ORPO |
| Maximum simplicity | SimPO |
| Best possible quality | PPO (if resources allow) |

---

## Implementation

### Basic DPO Training

```python
from trl import DPOTrainer, DPOConfig

# Load models
model = AutoModelForCausalLM.from_pretrained("sft_model")
ref_model = AutoModelForCausalLM.from_pretrained("sft_model")

# Config
config = DPOConfig(
    beta=0.1,                    # KL penalty strength
    learning_rate=5e-7,          # Very low LR
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    max_length=1024,
    max_prompt_length=512,
    num_train_epochs=1,
)

# Trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Dataset Format

```python
# DPO dataset format
preference_example = {
    "prompt": "Explain quantum computing",
    "chosen": "Quantum computing uses quantum bits (qubits)...",
    "rejected": "Quantum computing is very complex and hard to explain..."
}

# KTO dataset format (no pairs needed)
kto_example = {
    "prompt": "Explain quantum computing",
    "response": "Quantum computing uses quantum bits (qubits)...",
    "label": True  # or False for rejected
}
```

### Hyperparameters

| Parameter | DPO | IPO | KTO | ORPO |
|-----------|-----|-----|-----|------|
| β (beta) | 0.1-0.5 | 0.1 | 0.1-0.5 | 0.1 |
| Learning rate | 1e-7 to 5e-6 | Same | Same | 5e-6 to 1e-5 |
| Epochs | 1-3 | 1-3 | 1-3 | 1-3 |
| Label smoothing | 0 | 0 | N/A | 0 |

**Critical**: Learning rate must be much lower than SFT (10-100×).

---

## DPO vs RLHF

### Advantages of DPO

1. **Simplicity**: No reward model, no RL algorithm
2. **Stability**: Supervised learning, predictable
3. **Efficiency**: 2× fewer models, faster training
4. **Accessibility**: Easy to implement and debug

### Limitations of DPO

1. **Offline only**: Uses fixed preference data
2. **No exploration**: Can't discover novel behaviors
3. **Distribution mismatch**: Preferences from SFT model, applied to DPO model
4. **Theoretical gap**: May not match true preference distribution

### Empirical Comparison

| Benchmark | DPO | PPO | Notes |
|-----------|-----|-----|-------|
| MT-Bench | 7.2 | 7.3 | Nearly equivalent |
| AlpacaEval | 85% | 88% | PPO slightly better |
| Training time | 1× | 3-4× | DPO much faster |
| Compute cost | 1× | 4× | DPO much cheaper |

**Verdict**: DPO achieves ~95% of PPO quality at ~25% of the cost.

---

## Best Practices

### Data Quality

1. **Clear preferences**: Ambiguous pairs hurt training
2. **Diverse prompts**: Cover many capabilities
3. **Quality chosen**: Best responses, not just "better"
4. **Meaningful rejected**: Should be plausibly wrong

```python
def filter_preference_pair(chosen, rejected, prompt):
    # Both should be reasonable length
    if len(chosen) < 50 or len(rejected) < 50:
        return False

    # Should be meaningfully different
    if similarity(chosen, rejected) > 0.9:
        return False

    # Rejected shouldn't be obviously wrong
    if is_gibberish(rejected):
        return False

    return True
```

### Training Tips

1. **Start from good SFT**: DPO refines, doesn't create capability
2. **Low learning rate**: 1e-7 to 5e-6, err on lower side
3. **Watch for collapse**: If responses become repetitive, lower LR or β
4. **Short training**: Often 1 epoch is enough
5. **Validate often**: Track preference accuracy, generation quality

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Mode collapse | Repetitive outputs | Lower LR, lower β |
| No improvement | Metrics flat | Check data quality, increase β |
| Quality drop | Worse than SFT | Too many epochs, LR too high |
| Overfitting | Training acc high, eval low | Add regularization, fewer epochs |

---

## Advanced Topics

### Multi-Turn DPO

Extending to conversations:

```python
def multi_turn_dpo_loss(conversation_chosen, conversation_rejected):
    """DPO loss summed over assistant turns."""
    total_loss = 0
    for turn_chosen, turn_rejected in zip(
        get_assistant_turns(conversation_chosen),
        get_assistant_turns(conversation_rejected)
    ):
        total_loss += single_turn_dpo_loss(turn_chosen, turn_rejected)
    return total_loss
```

### Iterative DPO

Multiple rounds of DPO:

```python
def iterative_dpo(base_model, prompts, judge, n_iterations=3):
    model = base_model
    for i in range(n_iterations):
        # Generate responses with current model
        responses_a = model.generate(prompts)
        responses_b = model.generate(prompts)

        # Get preferences from judge
        preferences = judge.compare(prompts, responses_a, responses_b)

        # DPO training
        model = dpo_train(model, preferences)

    return model
```

### Combining with RLHF

DPO as warm-start for PPO:

```python
# Stage 1: SFT
sft_model = sft_train(base_model, demonstrations)

# Stage 2: DPO (cheap improvement)
dpo_model = dpo_train(sft_model, preferences)

# Stage 3: PPO (fine polish, if needed)
final_model = ppo_train(dpo_model, reward_model)
```

---

## Future Directions

### Near-term (2025)

1. **Online DPO**: Learning from live user feedback
2. **Multi-objective DPO**: Balancing helpfulness, safety, honesty
3. **Efficient variants**: Lower memory, faster training
4. **Better data**: Synthetic preference generation

### Research Frontiers

1. **Theoretical understanding**: When does DPO equal RLHF?
2. **Distribution correction**: Handling offline data better
3. **Curriculum learning**: Progressive preference difficulty
4. **Length and verbosity**: Preventing reward hacking

---

## Sources

### Foundational Papers
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) - Stanford, 2023
- [IPO: A General Class of Preference Optimization](https://arxiv.org/abs/2310.12036) - 2023
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) - 2024

### Method Variants
- [ORPO: Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691) - 2024
- [SimPO: Simple Preference Optimization](https://arxiv.org/abs/2405.14734) - 2024
- [Self-Play Preference Optimization](https://arxiv.org/abs/2405.00675) - 2024

### Analysis
- [A General Theoretical Paradigm for Preference-Based RL](https://arxiv.org/abs/2310.12036) - 2023
- [Is DPO Superior to PPO for LLM Alignment?](https://arxiv.org/abs/2404.10719) - 2024

### Implementations
- [TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [Axolotl DPO](https://github.com/OpenAccess-AI-Collective/axolotl)
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
