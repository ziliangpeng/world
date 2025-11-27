# Knowledge Distillation for LLMs

Knowledge distillation transfers capabilities from large, expensive "teacher" models to smaller, efficient "student" models. For LLMs, this enables deploying powerful capabilities on constrained hardware—running GPT-4-level reasoning on a smartphone. This document covers the techniques that make distillation work for language models.

---

## Why Distillation?

### The Deployment Problem

| Model | Parameters | Quality | Inference Cost |
|-------|------------|---------|----------------|
| GPT-4 | ~1.8T | Excellent | $$$$ |
| LLaMA-70B | 70B | Very Good | $$$ |
| **Distilled 7B** | 7B | Good | $ |

**Goal**: Achieve 80-90% of teacher quality at 10% of the cost.

### Use Cases

1. **Edge deployment**: Run on mobile devices
2. **Cost reduction**: Lower API/compute costs
3. **Latency improvement**: Faster inference
4. **Capability transfer**: Give small models specific skills
5. **Specialization**: Task-specific expert models

---

## Historical Evolution

### Phase 1: Classical Distillation (2015-2019)

**[Distilling Knowledge in Neural Networks](https://arxiv.org/abs/1503.02531)** (Hinton et al., 2015)

The foundational paper. Key insight: The teacher's "soft labels" (probability distributions) contain more information than hard labels.

```python
# Hard label: [0, 1, 0] (correct answer only)
# Soft label: [0.05, 0.90, 0.05] (includes "dark knowledge")
```

**Temperature-scaled softmax**:
```python
def soft_labels(logits, temperature=2.0):
    """Higher temperature → softer distribution → more knowledge transfer."""
    return F.softmax(logits / temperature, dim=-1)

def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    # Soft loss: match teacher's distribution
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchwise_mean'
    ) * (temperature ** 2)

    # Hard loss: match correct labels
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss
```

**DistilBERT** (2019): Applied to BERT, achieved 97% performance with 60% parameters.

### Phase 2: Sequence-Level Distillation (2020-2022)

**[Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947)** (2016)

For generation tasks, match sequences not just tokens:

```python
def sequence_distillation(prompts, teacher, student):
    # Generate sequences from teacher
    teacher_sequences = teacher.generate(prompts, max_length=512)

    # Train student to reproduce teacher sequences
    student_loss = F.cross_entropy(
        student(teacher_sequences[:, :-1]),
        teacher_sequences[:, 1:]
    )
    return student_loss
```

This became the default approach for LLMs—train student on teacher-generated text.

### Phase 3: LLM-Specific Distillation (2023)

**[Orca](https://arxiv.org/abs/2306.02707)** (June 2023) - "Explanation Tuning"

Distill not just answers but reasoning:

```
Teacher (GPT-4) generates:
"Let me solve this step by step:
1. First, I identify...
2. Then, I calculate...
3. Therefore, the answer is..."

Student learns to reproduce the full reasoning trace.
```

**Key insight**: Reasoning process is more valuable than just the answer.

**[Zephyr](https://arxiv.org/abs/2310.16944)** (October 2023)

Combined distillation with preference learning:
1. Generate responses with teacher (GPT-4)
2. SFT student on teacher responses
3. DPO with teacher-generated preferences

**Result**: Zephyr-7B outperformed 70B models.

### Phase 4: Modern Distillation (2024)

**Mixture of Experts Distillation**:
- Distill large MoE to dense student
- Or distill to smaller MoE

**Multi-Teacher Distillation**:
```python
def multi_teacher_distillation(student, teachers, weights):
    """Combine knowledge from multiple teachers."""
    total_loss = 0
    for teacher, weight in zip(teachers, weights):
        teacher_logits = teacher(input)
        total_loss += weight * kl_div(student(input), teacher_logits)
    return total_loss
```

**On-Policy Distillation**:
```python
def on_policy_distillation(student, teacher, prompts):
    """Student generates, teacher evaluates."""
    # Student generates
    student_responses = student.generate(prompts)

    # Teacher scores
    teacher_scores = teacher.score(student_responses)

    # Filter best responses
    good_responses = filter_by_score(student_responses, teacher_scores)

    # Train on good responses
    return sft_loss(student, good_responses)
```

---

## Distillation Methods

### 1. Black-Box Distillation

Only access to teacher outputs, not internal states:

**Sequence-level** (most common for LLMs):
```python
def blackbox_distillation(teacher_api, student, prompts):
    """Train student on teacher-generated responses."""
    dataset = []
    for prompt in prompts:
        response = teacher_api.generate(prompt)  # API call
        dataset.append({"prompt": prompt, "response": response})

    # Standard SFT on teacher outputs
    return sft_train(student, dataset)
```

**Advantages**:
- Works with closed-source teachers (GPT-4, Claude)
- Simple to implement
- No need for teacher internals

**Disadvantages**:
- No logit-level knowledge
- Limited by generation quality
- API costs

### 2. White-Box Distillation

Access to teacher's logits and/or hidden states:

**Logit matching**:
```python
def whitebox_distillation(teacher, student, data, temperature=2.0):
    """Match student logits to teacher logits."""
    for batch in data:
        with torch.no_grad():
            teacher_logits = teacher(batch)

        student_logits = student(batch)

        # KL divergence on softened distributions
        loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction='batchwise_mean'
        ) * (temperature ** 2)

        loss.backward()
        optimizer.step()
```

**Feature matching**:
```python
def feature_distillation(teacher, student, data, layer_mapping):
    """Match intermediate representations."""
    for batch in data:
        teacher_features = teacher.get_hidden_states(batch, layers=layer_mapping.keys())
        student_features = student.get_hidden_states(batch, layers=layer_mapping.values())

        loss = 0
        for t_layer, s_layer in layer_mapping.items():
            # Project if dimensions differ
            projected = projection[s_layer](student_features[s_layer])
            loss += F.mse_loss(projected, teacher_features[t_layer])

        loss.backward()
        optimizer.step()
```

**Advantages**:
- More information transfer
- Better quality
- Can match reasoning process

**Disadvantages**:
- Requires open teacher model
- More complex implementation
- Higher memory requirements

### 3. Self-Distillation

Model distills from itself:

```python
def self_distillation(model, data, n_iterations=3):
    """Iteratively improve through self-distillation."""
    for i in range(n_iterations):
        # Generate responses with current model
        responses = model.generate(data["prompts"])

        # Score responses (self or external judge)
        scores = model.score(responses)

        # Keep top responses
        top_responses = select_top_k(responses, scores, k=0.5)

        # Fine-tune on top responses
        model = sft_train(model, top_responses)

    return model
```

### 4. Task-Specific Distillation

Distill specific capabilities:

```python
def task_distillation(teacher, student, task_data):
    """Distill specific task capability."""
    # Task-focused prompts
    task_responses = []
    for prompt, expected_format in task_data:
        response = teacher.generate(prompt, format=expected_format)
        task_responses.append((prompt, response))

    # Fine-tune on task
    return sft_train(student, task_responses, epochs=3)
```

---

## Architecture Considerations

### Size Ratios

| Teacher Size | Student Size | Expected Performance |
|--------------|--------------|---------------------|
| 70B | 7B (10%) | 75-85% of teacher |
| 70B | 13B (18%) | 80-90% of teacher |
| 70B | 1B (1.4%) | 60-70% of teacher |

**Rule of thumb**: Students can retain ~80-90% of teacher capability at ~10-20% of parameters.

### Architecture Matching

**Same architecture family works best**:
```
Teacher: LLaMA-70B → Student: LLaMA-7B ✓
Teacher: LLaMA-70B → Student: Mistral-7B (okay)
Teacher: LLaMA-70B → Student: RWKV-7B (harder)
```

**Dimension adaptation**:
```python
class DimensionAdapter(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        self.down = nn.Linear(teacher_dim, student_dim)

    def adapt(self, teacher_features):
        return self.down(teacher_features)
```

### Layer Mapping

For feature distillation, map layers:
```python
# Uniform mapping
def uniform_layer_mapping(teacher_layers, student_layers):
    ratio = teacher_layers // student_layers
    return {i * ratio: i for i in range(student_layers)}

# Example: 64-layer teacher → 32-layer student
# Maps teacher layers 0, 2, 4, ... to student layers 0, 1, 2, ...
```

---

## Training Strategies

### Curriculum Distillation

Start easy, increase difficulty:
```python
def curriculum_distillation(teacher, student, data, stages=3):
    # Sort by difficulty
    sorted_data = sort_by_difficulty(data)

    for stage in range(stages):
        # Use increasingly difficult data
        start_idx = int(len(data) * stage / stages)
        end_idx = int(len(data) * (stage + 1) / stages)
        stage_data = sorted_data[start_idx:end_idx]

        student = distill(teacher, student, stage_data)

    return student
```

### Progressive Distillation

Incrementally transfer knowledge:
```python
def progressive_distillation(teacher, student_init, data, steps=5):
    student = student_init

    for step in range(steps):
        # Interpolate target
        alpha = step / steps
        target_logits = (1 - alpha) * student(data) + alpha * teacher(data)

        # Train student toward interpolated target
        loss = kl_div(student(data), target_logits)
        # ...
```

### Multi-Task Distillation

Transfer multiple capabilities:
```python
def multitask_distillation(teacher, student, task_datasets, task_weights):
    """Distill multiple capabilities simultaneously."""
    total_loss = 0
    for task, (data, weight) in enumerate(zip(task_datasets, task_weights)):
        teacher_outputs = teacher.generate(data["prompts"])
        task_loss = sft_loss(student, data["prompts"], teacher_outputs)
        total_loss += weight * task_loss

    return total_loss
```

---

## Quality Considerations

### What Transfers Well

| Capability | Transfer Quality | Notes |
|------------|-----------------|-------|
| Formatting | Excellent | Easy to learn from examples |
| Factual recall | Good | Limited by student capacity |
| Reasoning | Moderate | Requires chain-of-thought |
| Creativity | Moderate | Diversity may decrease |
| Long context | Poor | Architectural limitation |

### What Transfers Poorly

1. **Emergent abilities**: May require teacher scale to emerge
2. **Deep reasoning**: Complex multi-step reasoning hard to compress
3. **World knowledge**: Student has limited memory
4. **Robustness**: Student may be more brittle

### Evaluating Distillation

```python
def evaluate_distillation(teacher, student, test_set):
    results = {
        "teacher_performance": evaluate(teacher, test_set),
        "student_performance": evaluate(student, test_set),
        "retention_rate": None
    }

    results["retention_rate"] = (
        results["student_performance"] /
        results["teacher_performance"]
    )

    return results

# Target: retention_rate > 0.80
```

---

## Notable Distillation Projects

| Student | Teacher | Method | Result |
|---------|---------|--------|--------|
| DistilBERT | BERT | Logit + feature | 97% perf, 60% size |
| Alpaca | text-davinci-003 | Sequence | Competitive 7B |
| Vicuna | GPT-3.5/4 (ShareGPT) | Sequence | ~90% ChatGPT |
| Orca | GPT-4 | Explanation | Near-GPT-4 reasoning |
| Zephyr | GPT-4 | Sequence + DPO | SOTA 7B |
| OpenHermes | GPT-4 | Sequence | Strong 7B |
| Phi-3 | GPT-4 (indirect) | Synthetic data | Competitive small |

---

## Best Practices

### Data

1. **Diverse prompts**: Cover many capabilities
2. **High-quality responses**: Filter teacher outputs
3. **Reasoning traces**: Include step-by-step when possible
4. **Balance**: Even coverage across tasks

### Training

1. **Lower learning rate**: 1e-5 to 5e-6 (lower than SFT)
2. **Temperature**: 2-4 for logit distillation
3. **Mixed loss**: Combine soft and hard targets
4. **Validation**: Monitor retention on benchmarks

### Architecture

1. **Match families**: Same architecture when possible
2. **Proper initialization**: From pretrained weights
3. **Layer mapping**: Uniform or learned mapping
4. **Adapter projections**: When dimensions differ

---

## Future Directions

### Near-term (2025)

1. **Automated distillation**: End-to-end pipelines
2. **Capability-specific**: Distill targeted skills
3. **Multi-modal**: Image-text capability transfer
4. **MoE distillation**: Dense from sparse

### Research Frontiers

1. **Efficiency**: Better quality per parameter
2. **Selective transfer**: Choose what to distill
3. **Verification**: Ensure capability transfer
4. **Composition**: Combine multiple distilled models

---

## Sources

### Foundational Papers
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - Hinton et al., 2015
- [DistilBERT](https://arxiv.org/abs/1910.01108) - HuggingFace, 2019
- [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947) - 2016

### LLM Distillation
- [Orca: Progressive Learning from Complex Explanation Traces](https://arxiv.org/abs/2306.02707) - 2023
- [Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944) - 2023
- [The False Promise of Imitating Proprietary LLMs](https://arxiv.org/abs/2305.15717) - 2023

### Practical Guides
- [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525) - 2020
- [Distilling Step-by-Step!](https://arxiv.org/abs/2305.02301) - 2023
