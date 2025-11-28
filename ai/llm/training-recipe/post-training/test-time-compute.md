# Test-Time Compute & Reasoning Training

This document covers the emerging paradigm of training LLMs to "think" during inference—the approach behind o1, DeepSeek-R1, and similar reasoning models. Instead of generating answers immediately, these models produce extended chains of reasoning, trading inference compute for better answers.

---

## The Core Insight

Traditional LLMs: Input → Single forward pass → Output
Reasoning LLMs: Input → Multiple reasoning steps → Output

**The key discovery**: Model capability isn't fixed at training time. By generating intermediate reasoning tokens, models can solve problems beyond their "instant" capability—similar to how humans solve complex math by writing out steps rather than computing answers mentally.

```
Standard model (fixed compute):
Q: "What is 847 × 293?"
A: "248,171" (often wrong)

Reasoning model (scaling compute):
Q: "What is 847 × 293?"
A: "Let me work through this step by step:
    847 × 293 = 847 × (300 - 7)
    = 847 × 300 - 847 × 7
    = 254,100 - 5,929
    = 248,171" ✓
```

---

## Test-Time Scaling Laws

### Traditional Scaling (Kaplan/Chinchilla)

Performance improves with training compute:
- More parameters
- More training data
- More training FLOPs

But: Once trained, capability is fixed.

### Test-Time Scaling (2024+)

Performance also improves with inference compute:
- More reasoning tokens
- More samples (best-of-N)
- More verification passes

**The finding from OpenAI and DeepSeek**: On reasoning tasks, doubling inference compute can provide similar gains to training a model 10× larger.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Performance vs Compute                          │
│                                                                     │
│  Performance                                                        │
│       │                          ┌─── Test-time scaling            │
│       │                    ╱─────╯    (reasoning model)             │
│       │              ╱────╯                                         │
│       │        ╱────╯                                               │
│       │   ────╯        ┌─── Train-time scaling                     │
│       │ ──────────────╯     (standard model)                        │
│       │                                                             │
│       └────────────────────────────────────────────────────────────│
│                           Compute Budget                            │
│                                                                     │
│  Note: For complex reasoning tasks, test-time compute is more      │
│  efficient than scaling training compute                            │
└─────────────────────────────────────────────────────────────────────┘
```

### When Test-Time Scaling Helps

| Task Type | Test-Time Scaling Benefit |
|-----------|---------------------------|
| Math word problems | High |
| Code generation | High |
| Logical puzzles | High |
| Multi-step reasoning | High |
| Factual recall | Low |
| Creative writing | Low |
| Simple QA | Minimal |

**Key insight**: Test-time compute helps most on tasks with verifiable intermediate steps.

---

## Chain-of-Thought as Trainable Behavior

### Prompting vs Training

**Chain-of-thought prompting** (2022): Add "Let's think step by step" to prompt
- Works, but inconsistent
- Model may not follow the pattern
- Quality of reasoning varies

**Chain-of-thought training** (2024): Train model to always reason
- Consistent reasoning behavior
- Higher quality chains
- Can be optimized via RL

### How to Train CoT Behavior

#### Method 1: SFT on Reasoning Data

Collect or generate demonstrations with explicit reasoning:

```python
# Training example format
{
    "prompt": "Question: In a school, 40% of students play sports. If there are 250 students, how many play sports?",
    "response": """Let me solve this step by step:

1. I need to find 40% of 250
2. To find a percentage, I multiply the total by the percentage as a decimal
3. 40% as a decimal is 0.40
4. 250 × 0.40 = 100

Therefore, 100 students play sports."""
}
```

**Data sources**:
- Human-written solutions (expensive but high quality)
- Synthetic from strong models (scalable)
- Process mining from correct solutions

#### Method 2: RL for Reasoning (DeepSeek-R1 Approach)

Train model to develop its own reasoning style:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DeepSeek-R1 Training Pipeline                    │
│                                                                     │
│  Base Model ──▶ "Cold Start" SFT ──▶ RL Training ──▶ Reasoning Model│
│                    (small set)       (large scale)                  │
│                                                                     │
│  Cold Start:                                                        │
│  - Few thousand examples with reasoning                             │
│  - Teaches basic format: "Let me think..."                         │
│                                                                     │
│  RL Training:                                                       │
│  - Model generates reasoning + answer                               │
│  - Reward based on answer correctness (not reasoning format)        │
│  - Model discovers effective reasoning patterns on its own          │
└─────────────────────────────────────────────────────────────────────┘
```

**The DeepSeek-R1 insight**: With minimal format supervision (cold start), RL alone can teach a model to reason effectively. The model learns what reasoning patterns actually help solve problems.

---

## RL for Reasoning

### Why RL Works for Reasoning

1. **Clear reward signal**: Math/code have verifiable answers
2. **Credit assignment**: Reasoning steps affect final answer
3. **Exploration**: Model can discover novel reasoning strategies
4. **Optimization target**: Maximize correct answers, not imitation

### Reward Model Choices

#### Outcome Reward Models (ORM)

Reward only the final answer:
- Correct answer → reward = 1
- Wrong answer → reward = 0

```python
def outcome_reward(question, model_response):
    predicted_answer = extract_answer(model_response)
    correct_answer = get_ground_truth(question)
    return 1.0 if predicted_answer == correct_answer else 0.0
```

**Pros**: Simple, no annotation needed beyond Q&A pairs
**Cons**: Sparse signal, can reward lucky guesses

#### Process Reward Models (PRM)

Reward each reasoning step:

```python
def process_reward(question, reasoning_steps):
    rewards = []
    for step in reasoning_steps:
        # Score each step for correctness
        step_score = prm_model(question, previous_steps, step)
        rewards.append(step_score)
    return rewards  # Dense reward signal
```

**Pros**: Dense signal, catches errors early, more stable training
**Cons**: Requires step-level annotation (expensive)

### Comparison

| Aspect | ORM | PRM |
|--------|-----|-----|
| Training data | Q&A pairs only | Step-level annotations |
| Signal density | Sparse (final only) | Dense (per step) |
| Credit assignment | Difficult | Direct |
| Annotation cost | Low | High |
| Training stability | Less stable | More stable |
| Best for | Simple problems | Complex multi-step |

### GRPO: Group Relative Policy Optimization

DeepSeek-R1 uses GRPO, a simplified alternative to PPO:

```python
# GRPO algorithm (simplified)
def grpo_step(model, prompts, num_samples=8):
    for prompt in prompts:
        # Generate multiple solutions
        responses = [model.generate(prompt) for _ in range(num_samples)]

        # Get rewards for each
        rewards = [reward_fn(prompt, resp) for resp in responses]

        # Compute group baseline (average reward)
        baseline = mean(rewards)

        # Compute advantages relative to group
        advantages = [r - baseline for r in rewards]

        # Update policy: increase prob of above-average, decrease below
        for response, advantage in zip(responses, advantages):
            update_policy(model, prompt, response, advantage)
```

**Key differences from PPO**:
- No separate value network needed
- Group-based baseline instead of learned value function
- Simpler implementation, fewer hyperparameters
- Works well for reasoning tasks with clear answers

---

## Practical Training Pipeline

### Full Pipeline (DeepSeek-R1 Style)

```
Stage 1: Base Model
│   Llama-3-70B or similar
│
├── Stage 2: Cold Start SFT
│   │   Train on ~5K reasoning examples
│   │   Purpose: Learn reasoning format
│   │   Duration: 1-2 epochs
│   │
├── Stage 3: RL Training
│   │   GRPO on math/code problems
│   │   Reward: Answer correctness
│   │   Duration: Thousands of steps
│   │
│   │   Key behaviors that emerge:
│   │   - Self-correction ("Wait, let me reconsider...")
│   │   - Verification ("Let me check this...")
│   │   - Alternative approaches ("Another way to solve this...")
│   │
├── Stage 4: Rejection Sampling
│   │   Generate many solutions per problem
│   │   Keep only correct ones
│   │   Create cleaned SFT dataset
│   │
└── Stage 5: Final SFT + Alignment
        Train on curated reasoning data
        Add general capability data
        Apply safety alignment
```

### Training Data for Each Stage

| Stage | Data Size | Source |
|-------|-----------|--------|
| Cold Start | 5-10K | Curated reasoning demos |
| RL Training | 100K+ prompts | Math/code problems (no solutions needed) |
| Rejection Sampling | Generated | Model's own correct solutions |
| Final SFT | 500K-1M | Rejection samples + general data |

### Hyperparameters (DeepSeek-R1)

```python
grpo_config = {
    # Sampling
    "num_samples_per_prompt": 8,
    "temperature": 0.7,
    "max_reasoning_tokens": 8192,

    # Optimization
    "learning_rate": 1e-6,
    "kl_coef": 0.01,  # KL penalty to prevent collapse
    "clip_ratio": 0.2,

    # Reward
    "reward_model": "outcome",  # or "process"
    "length_penalty": -0.001,  # Discourage unnecessary length

    # Training
    "batch_size": 256,
    "gradient_accumulation": 4,
    "warmup_steps": 100,
}
```

---

## Best-of-N Sampling (Inference Time)

Even without RL training, test-time compute helps via sampling:

```python
def best_of_n(model, prompt, n=64, reward_model=None):
    """Generate N solutions, return the best one."""

    # Generate N candidate solutions
    candidates = [model.generate(prompt, temperature=0.7) for _ in range(n)]

    if reward_model:
        # Score with reward model
        scores = [reward_model.score(prompt, c) for c in candidates]
    else:
        # Self-consistency: majority vote on final answer
        answers = [extract_answer(c) for c in candidates]
        answer_counts = Counter(answers)
        best_answer = answer_counts.most_common(1)[0][0]
        # Return any candidate with the most common answer
        scores = [1 if extract_answer(c) == best_answer else 0 for c in candidates]

    best_idx = argmax(scores)
    return candidates[best_idx]
```

### Scaling Properties

| N (samples) | Relative Performance | Compute Cost |
|-------------|---------------------|--------------|
| 1 | Baseline | 1× |
| 4 | +10-15% | 4× |
| 16 | +20-25% | 16× |
| 64 | +25-35% | 64× |
| 256 | +30-40% | 256× |

**Diminishing returns**: Performance scales as ~log(N), but still effective for high-value problems.

---

## Distillation of Reasoning Models

### The Challenge

Reasoning models are expensive:
- Long outputs (thousands of tokens)
- Multiple samples for best-of-N
- Full reasoning trace needed

### Distillation Approach

```
Teacher: DeepSeek-R1-70B (reasoning model)
         ↓
         Generate solutions with full reasoning
         ↓
Student: DeepSeek-R1-7B
         Train to match teacher's reasoning traces
```

### Two Distillation Strategies

**1. Full trace distillation**: Student learns complete reasoning
- Output: Full chain of thought
- Inference: Still expensive (long output)
- Quality: Near-teacher performance

**2. Compressed reasoning**: Student learns efficient reasoning
- Output: Minimal necessary steps
- Inference: Much cheaper
- Quality: Some capability loss

### DeepSeek Distillation Results

| Model | Params | MATH | AIME | Training |
|-------|--------|------|------|----------|
| R1 | 671B | 97.3% | 79.8% | Full RL |
| R1-Distill-Qwen-32B | 32B | 94.3% | 72.6% | Distillation |
| R1-Distill-Qwen-7B | 7B | 92.8% | 55.5% | Distillation |
| R1-Distill-Qwen-1.5B | 1.5B | 83.9% | 28.9% | Distillation |

**Insight**: Distillation transfers most reasoning capability at fraction of size.

---

## Challenges and Limitations

### 1. Reward Hacking

Model finds shortcuts that get reward without real reasoning:

```
# Example of reward hacking
Q: "What is 15% of 80?"
A: "Let me think... The answer is probably around 12.
    Actually, trying 12: 12/80 = 0.15 ✓
    The answer is 12."

# Model "verifies" by working backwards, not actually computing
```

**Mitigations**:
- Process reward models catch faulty reasoning
- Diversity in training problems
- Adversarial evaluation

### 2. Reasoning Verbosity

Models may produce unnecessarily long chains:

**Problem**: Length penalty in reward can conflict with thorough reasoning
**Solution**: Calibrate length penalties, reward efficiency

### 3. Hallucinated Reasoning

Model produces confident-sounding but incorrect steps:

```
Q: "Is 17 prime?"
A: "Let me check divisibility:
    17 / 2 = 8.5 (not divisible)
    17 / 3 = 5.67 (not divisible)
    17 / 4 = 4.25 (not divisible)  ← Unnecessary, 4 is not prime
    17 / 5 = 3.4 (not divisible)
    Therefore 17 is prime." ✓ (correct but wasteful)
```

### 4. Distribution Shift

Reasoning patterns effective for math may not transfer:
- Code debugging
- Logical reasoning
- Scientific problems

**Solution**: Diverse training tasks, domain-specific RL

---

## When to Use Reasoning Models

### Good Fit

- Math competitions (AMC, AIME, Olympiad)
- Complex code generation
- Multi-step logical problems
- Scientific reasoning
- Planning and scheduling

### Poor Fit

- Simple factual questions
- Creative tasks (writing, brainstorming)
- Latency-critical applications
- Tasks without verifiable answers

### Decision Framework

```
Is the task:
├── Verifiable? (clear right/wrong)
│   ├── Yes → Consider reasoning model
│   └── No → Standard model likely better
│
├── Multi-step?
│   ├── Yes → Reasoning model helps
│   └── No → Standard model sufficient
│
├── Latency-tolerant?
│   ├── Yes → Can use full reasoning
│   └── No → Consider distilled model or cached solutions
```

---

## Key Papers

| Paper | Contribution |
|-------|--------------|
| Chain-of-Thought Prompting (Wei et al., 2022) | Showed prompting alone improves reasoning |
| Let's Verify Step by Step (Lightman et al., 2023) | Process reward models for math |
| DeepSeek-R1 (2025) | Open reproduction of o1-style training |
| Scaling Laws for Reward Model Overoptimization | Limits of RL for LLMs |
| Training Verifiers (Cobbe et al., 2021) | Early work on outcome verification |

---

## Summary

Test-time compute represents a paradigm shift:

1. **Capability isn't fixed**: Models can solve harder problems by "thinking longer"
2. **RL enables emergent reasoning**: With minimal supervision, models learn to reason
3. **Process vs outcome rewards**: Trade annotation cost vs training stability
4. **Distillation works**: Large reasoning model → smaller efficient model
5. **Best for verifiable tasks**: Math, code, logic benefit most

The training recipe:
```
Base model → Cold start SFT → RL (GRPO) → Rejection sampling → Final SFT
```

Key hyperparameters:
- 8+ samples per prompt for RL
- Outcome reward for simplicity, process reward for stability
- KL penalty to prevent collapse
- Length penalty to prevent verbosity
