# Reinforcement Learning from Human Feedback (RLHF)

RLHF is the technique that transformed language models from capable but unreliable text completers into helpful, harmless assistants. By learning from human preferences rather than just demonstrations, RLHF aligns model behavior with human values in ways that supervised learning cannot. This document traces the evolution of RLHF and explains its technical foundations.

---

## Why RLHF?

Supervised Fine-Tuning (SFT) teaches models to imitate demonstrations, but has fundamental limitations:

| Problem | SFT Limitation | RLHF Solution |
|---------|---------------|---------------|
| **Imitation only** | Can only match demonstrated behavior | Learns to optimize beyond demonstrations |
| **Average behavior** | Learns to produce "average" of training data | Learns to maximize preference |
| **Hard to specify** | Some behaviors hard to demonstrate | Easier to judge than demonstrate |
| **Scaling labels** | Need demonstrations for every behavior | Preferences more efficient to collect |

**The key insight**: It's often easier to evaluate responses than to create perfect responses. RLHF exploits this by learning from evaluations.

---

## Historical Evolution

### Phase 1: RL from Preferences (2017-2019)

**[Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)** (OpenAI/DeepMind, 2017)

Applied preference learning to Atari games and robotics:
- Human compares pairs of trajectories
- Train reward model from comparisons
- Use reward model to train policy via RL

**Key insight**: ~1000 preference labels sufficient for complex behaviors.

### Phase 2: Language Model Applications (2019-2021)

**[Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)** (OpenAI, 2019)

First application to text generation:
- Stylistic continuation (sentiment, topic)
- Summarization quality
- Simple reward model + policy gradient

**[Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)** (OpenAI, 2020)

Applied RLHF to summarization at scale:
- 64K human preference labels
- Reward model predicting human preferences
- PPO to optimize against reward model
- Models preferred over supervised baseline 70% of time

### Phase 3: InstructGPT and ChatGPT (2022)

**[InstructGPT](https://arxiv.org/abs/2203.02155)** (OpenAI, March 2022)

The paper that defined modern RLHF:

```
Pipeline:
1. SFT: Fine-tune on demonstrations
2. RM: Train reward model on preferences
3. PPO: Optimize policy with reward model
```

**Scale**:
- 40 human labelers
- 13K preference comparisons for RM
- PPO training with reward model

**Results**: InstructGPT-1.3B preferred over GPT-3-175B

**ChatGPT** (November 2022)

Applied InstructGPT recipe to GPT-3.5, creating the conversational AI breakthrough that reached 100M users in 2 months.

### Phase 4: Open RLHF (2023)

**[Open LLaMA RLHF](https://github.com/OpenLMLab/MOSS-RLHF)** and others

Open-source implementations enabling community research:
- TRL (HuggingFace)
- DeepSpeed-Chat (Microsoft)
- Colossal-AI RLHF

**[Anthropic's Constitutional AI](https://arxiv.org/abs/2212.08073)** (December 2022)

Self-improvement through AI feedback:
- Generate responses
- AI critiques and revises responses
- Train on AI-preferred revisions

Led to RLAIF (RL from AI Feedback) research direction.

### Phase 5: Alternatives and Simplifications (2023-2024)

Researchers explored simpler methods:
- **DPO**: Direct preference optimization without RL
- **ORPO**: Combined SFT and preference in one stage
- **KTO**: Binary "good/bad" instead of comparisons

> **See also**: [Direct Preference Optimization](direct-preference.md) for RL-free alternatives.

---

## The RLHF Pipeline

### Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RLHF Pipeline                                  │
└─────────────────────────────────────────────────────────────────────────┘

Phase 1: SFT                    Phase 2: Reward Model            Phase 3: PPO
┌──────────────┐                ┌──────────────────────┐        ┌──────────────────┐
│              │                │                      │        │                  │
│  Base Model  │                │  Prompt + Response A │        │    SFT Model     │
│      ↓       │                │  Prompt + Response B │        │        ↓         │
│  SFT Model   │                │         ↓            │        │  Generate        │
│  (demo data) │                │  Human Preference    │        │  Response        │
│              │                │         ↓            │        │        ↓         │
└──────────────┘                │  Reward Model        │        │  Score with RM   │
                                │                      │        │        ↓         │
                                └──────────────────────┘        │  PPO Update      │
                                                                │  (+ KL penalty)  │
                                                                └──────────────────┘
```

### Step 1: Supervised Fine-Tuning (SFT)

Train base model on high-quality demonstrations:
```python
# Start from pretrained model
model = AutoModelForCausalLM.from_pretrained("base_model")

# Fine-tune on demonstrations
trainer = SFTTrainer(model=model, dataset=demonstrations)
sft_model = trainer.train()
```

**Purpose**: Create a model that understands the desired format and can generate reasonable responses.

### Step 2: Reward Modeling

Train a model to predict human preferences:

**Data collection**:
```python
# For each prompt, collect multiple responses
prompt = "Explain quantum computing simply"
response_A = model.generate(prompt, do_sample=True)
response_B = model.generate(prompt, do_sample=True)

# Human labels which is better
preference = human_eval(prompt, response_A, response_B)
# Returns: "A", "B", or "tie"
```

**Reward model architecture**:
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask)
        # Use last token's hidden state
        last_hidden = outputs.last_hidden_state[:, -1, :]
        reward = self.reward_head(last_hidden)
        return reward
```

**Bradley-Terry loss** (pairwise preferences):
```python
def reward_model_loss(chosen_rewards, rejected_rewards):
    """
    Bradley-Terry model: P(A > B) = sigmoid(r_A - r_B)
    Loss: -log(sigmoid(r_chosen - r_rejected))
    """
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

**Training**:
```python
for batch in preference_dataset:
    chosen_reward = reward_model(batch["chosen_ids"])
    rejected_reward = reward_model(batch["rejected_ids"])

    loss = reward_model_loss(chosen_reward, rejected_reward)
    loss.backward()
    optimizer.step()
```

### Step 3: PPO Training

Optimize the policy (language model) to maximize reward while staying close to SFT model:

**Objective**:
```
maximize E[r(x, y)] - β × KL[π(y|x) || π_ref(y|x)]

Where:
- r(x, y): reward model score
- π: current policy
- π_ref: reference policy (SFT model)
- β: KL penalty coefficient
```

**The KL penalty is crucial**: Without it, the model "hacks" the reward model—finding high-reward but degenerate outputs.

**PPO algorithm for LLMs**:
```python
class PPOTrainer:
    def __init__(self, policy_model, ref_model, reward_model, kl_coef=0.1):
        self.policy = policy_model
        self.ref = ref_model  # Frozen SFT model
        self.rm = reward_model
        self.kl_coef = kl_coef

    def compute_rewards(self, prompts, responses):
        """Compute rewards with KL penalty."""
        # Reward model score
        rm_rewards = self.rm(prompts, responses)

        # KL divergence penalty
        policy_logprobs = self.policy.log_prob(responses, prompts)
        ref_logprobs = self.ref.log_prob(responses, prompts)
        kl = policy_logprobs - ref_logprobs

        # Combined reward
        rewards = rm_rewards - self.kl_coef * kl
        return rewards, kl

    def ppo_step(self, batch):
        """Single PPO update step."""
        prompts = batch["prompts"]

        # Generate responses with current policy
        responses = self.policy.generate(prompts)

        # Compute rewards
        rewards, kl = self.compute_rewards(prompts, responses)

        # Compute advantages (GAE or simple)
        advantages = compute_advantages(rewards)

        # PPO clipped objective
        old_logprobs = batch["old_logprobs"]
        new_logprobs = self.policy.log_prob(responses, prompts)

        ratio = (new_logprobs - old_logprobs).exp()
        clipped_ratio = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)

        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()

        # Value function loss (optional)
        value_loss = F.mse_loss(self.policy.value(prompts), rewards)

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        return loss, {"reward": rewards.mean(), "kl": kl.mean()}
```

---

## Technical Challenges

### Reward Hacking

The model finds ways to get high rewards without being genuinely helpful:

| Hack | Example | Solution |
|------|---------|----------|
| Length gaming | Longer responses get higher rewards | Normalize by length |
| Format gaming | Always use bullet points | Diverse training data |
| Sycophancy | Always agree with user | Adversarial prompts |
| Repetition | Repeat high-reward phrases | N-gram penalties |

### KL Divergence Management

Too high KL → Model drifts too far, degrades
Too low KL → Model doesn't improve

```python
# Adaptive KL controller
class AdaptiveKLController:
    def __init__(self, target_kl=6.0):
        self.target = target_kl
        self.kl_coef = 0.1

    def update(self, current_kl):
        """Adjust KL coefficient to hit target."""
        if current_kl < self.target / 1.5:
            self.kl_coef *= 0.9  # Decrease penalty, allow more exploration
        elif current_kl > self.target * 1.5:
            self.kl_coef *= 1.1  # Increase penalty, stay closer to ref
```

### Reward Model Quality

The reward model is the bottleneck—poor RM = poor alignment:

**Evaluation**:
```python
def evaluate_reward_model(rm, test_preferences):
    """Accuracy on held-out preference data."""
    correct = 0
    for chosen, rejected in test_preferences:
        if rm(chosen) > rm(rejected):
            correct += 1
    return correct / len(test_preferences)

# Good RM: >70% accuracy
# Great RM: >80% accuracy
```

**Common issues**:
- **Distribution shift**: RM trained on SFT outputs, evaluated on PPO outputs
- **Overconfidence**: RM gives extreme scores, limiting gradients
- **Noise**: Human preferences are inconsistent

### Training Stability

PPO can be unstable for language models:

```python
# Stability techniques
stability_config = {
    "clip_eps": 0.2,           # Clip ratio for stable updates
    "value_clip": 0.2,         # Also clip value function
    "grad_clip": 1.0,          # Gradient clipping
    "entropy_coef": 0.01,      # Entropy bonus for exploration
    "gae_lambda": 0.95,        # GAE for variance reduction
    "mini_batch_size": 64,     # Smaller batches for stability
    "ppo_epochs": 4,           # Multiple PPO epochs per batch
}
```

---

## Implementation Options

### TRL (Transformers Reinforcement Learning)

HuggingFace's library:

```python
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# Model with value head
model = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("sft_model")

config = PPOConfig(
    learning_rate=1e-5,
    batch_size=64,
    mini_batch_size=16,
    ppo_epochs=4,
    kl_penalty="kl",
    target_kl=6.0,
)

ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# Training loop
for batch in dataloader:
    query_tensors = batch["input_ids"]

    # Generate responses
    response_tensors = ppo_trainer.generate(query_tensors)

    # Get rewards from reward model
    rewards = [reward_model(q, r) for q, r in zip(query_tensors, response_tensors)]

    # PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

### DeepSpeed-Chat

Microsoft's efficient implementation:

```python
# Three-stage pipeline
# Stage 1: SFT
deepspeed_sft(model, sft_dataset)

# Stage 2: Reward modeling
deepspeed_reward(model, preference_dataset)

# Stage 3: PPO
deepspeed_ppo(actor, critic, reward_model, ref_model, prompts)
```

### Resource Requirements

| Model Size | SFT | RM Training | PPO Training |
|------------|-----|-------------|--------------|
| 7B | 1× A100 | 1× A100 | 4× A100 |
| 13B | 2× A100 | 2× A100 | 8× A100 |
| 70B | 8× A100 | 8× A100 | 32× A100 |

PPO requires ~4× more GPUs than SFT due to:
- Actor model
- Reference model
- Critic model
- Reward model

---

## Alternatives to PPO

### REINFORCE

Simpler than PPO but higher variance:

```python
def reinforce_loss(responses, rewards, logprobs):
    """Basic policy gradient."""
    baseline = rewards.mean()
    advantages = rewards - baseline
    loss = -(logprobs * advantages).mean()
    return loss
```

### Reward-Ranked Fine-Tuning (RAFT)

Sample multiple responses, train on best:

```python
def raft_step(prompt, model, reward_model, n_samples=4):
    # Generate multiple responses
    responses = [model.generate(prompt) for _ in range(n_samples)]

    # Score with reward model
    scores = [reward_model(prompt, r) for r in responses]

    # Fine-tune on best response
    best_response = responses[scores.index(max(scores))]
    return sft_loss(prompt, best_response)
```

### ReST (Reinforced Self-Training)

Iterative improvement with self-generated data:

```python
def rest_iteration(model, prompts, reward_model, threshold):
    # Generate responses
    responses = model.generate(prompts)

    # Filter by reward threshold
    good_pairs = [
        (p, r) for p, r in zip(prompts, responses)
        if reward_model(p, r) > threshold
    ]

    # Fine-tune on filtered data
    sft_train(model, good_pairs)
```

---

## Best Practices

### Data Collection

1. **Diversity**: Varied prompts covering different capabilities
2. **Quality**: Experienced labelers with clear guidelines
3. **Consistency**: Inter-annotator agreement checks
4. **Balance**: Equal representation of capabilities

### Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| KL coefficient | 0.01-0.1 | Start low, increase if drifting |
| Learning rate | 1e-6 to 1e-5 | Very low vs SFT |
| PPO epochs | 2-4 | Per batch of generations |
| Batch size | 64-512 | Larger helps stability |
| Target KL | 6-10 | Stop if exceeded |

### Monitoring

```python
metrics_to_track = [
    "reward/mean",          # Should increase
    "reward/std",           # Should remain stable
    "kl/mean",              # Should stay near target
    "policy/entropy",       # Should decrease slowly
    "policy/ratio",         # Should stay near 1.0
    "value/explained_var",  # Should be high
]
```

---

## Future Directions

### Near-term (2025)

1. **Reward model improvements**: Better generalization, less hackable
2. **Sample efficiency**: Fewer human labels needed
3. **Online RLHF**: Continuous learning from user feedback

### Research Frontiers

1. **Constitutional AI**: Self-improvement with AI feedback
2. **Process supervision**: Reward steps, not just outcomes
3. **Multi-objective RLHF**: Balance multiple rewards
4. **Scalable oversight**: Human supervision for superhuman models

---

## Sources

### Foundational Papers
- [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) - OpenAI/DeepMind, 2017
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) - OpenAI, 2020
- [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) - OpenAI, 2022

### RL Algorithms
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) - OpenAI, 2017
- [Secrets of RLHF in Large Language Models Part I](https://arxiv.org/abs/2307.04964) - 2023
- [Secrets of RLHF in Large Language Models Part II](https://arxiv.org/abs/2401.06080) - 2024

### Implementations
- [TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat)
- [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF)

### Analysis and Best Practices
- [RLHF: Theory and Practice](https://arxiv.org/abs/2403.04642) - Survey, 2024
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) - DeepMind, 2022
