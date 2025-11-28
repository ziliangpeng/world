# Curriculum Learning and Data Mixing for LLMs

Curriculum learning—training models in stages with carefully designed data mixes—can significantly improve final model quality, training efficiency, and domain performance. This document covers practical strategies for designing training curricula and mixing data sources effectively.

---

## Why Curriculum Learning Matters

### The Motivation

**Human learning analogy**: We don't teach calculus before arithmetic. Similarly, LLMs may benefit from structured learning progressions.

**Benefits**:
1. **Faster convergence**: Easier examples early → faster initial learning
2. **Better generalization**: Diverse data exposure → robust representations
3. **Domain specialization**: Targeted data mixing → specialized capabilities
4. **Data efficiency**: Strategic sampling → better use of limited data

### Evidence from Research

**Key findings**:
- Llama 3: Multi-stage training with progressive context extension (8K → 128K)
- Qwen 2.5: Curriculum across domains (general → code → math)
- DeepSeek-V3: Long-context curriculum (4K → 32K → 128K)
- Phi-3: Quality-over-quantity with curated curriculum

---

## Curriculum Strategies

### 1. Difficulty Progression

**Concept**: Train on easier examples first, progressively increase difficulty.

```python
# Example: Token-level difficulty progression
def difficulty_score(document):
    """
    Estimate document difficulty
    - Lower perplexity (by small reference model) = easier
    - Simpler vocabulary = easier
    - Shorter sentences = easier
    """
    scores = {
        'perplexity': compute_perplexity(reference_model, document),
        'vocab_rarity': compute_vocab_rarity(document),
        'avg_sentence_length': compute_avg_sentence_length(document),
    }

    # Weighted combination
    difficulty = (
        scores['perplexity'] * 0.5 +
        scores['vocab_rarity'] * 0.3 +
        scores['avg_sentence_length'] * 0.2
    )

    return difficulty

# Create curriculum stages
stages = [
    {
        'name': 'Stage 1: Easy',
        'steps': 10000,
        'data': filter_documents(corpus, difficulty_range=(0, 0.3)),
    },
    {
        'name': 'Stage 2: Medium',
        'steps': 20000,
        'data': filter_documents(corpus, difficulty_range=(0.3, 0.7)),
    },
    {
        'name': 'Stage 3: Hard',
        'steps': 30000,
        'data': filter_documents(corpus, difficulty_range=(0.7, 1.0)),
    },
    {
        'name': 'Stage 4: Mixed',
        'steps': 40000,
        'data': corpus,  # All difficulties
    },
]
```

**Results from research**:
- Faster initial convergence (10-20% fewer steps to target perplexity)
- Better final performance on complex tasks
- More stable training (fewer divergences)

### 2. Domain Progression

**Concept**: Progress from general knowledge to specialized domains.

```python
# Example: General → Specialized curriculum
curriculum = [
    {
        'stage': 'Foundation (General Knowledge)',
        'steps': 50000,
        'data_mix': {
            'web_text': 0.50,
            'books': 0.30,
            'wikipedia': 0.20,
        },
        'objective': 'Broad language understanding',
    },
    {
        'stage': 'Code Specialization',
        'steps': 20000,
        'data_mix': {
            'web_text': 0.20,
            'code': 0.60,
            'code_docs': 0.20,
        },
        'objective': 'Programming capabilities',
    },
    {
        'stage': 'Math Specialization',
        'steps': 15000,
        'data_mix': {
            'web_text': 0.10,
            'code': 0.20,
            'math': 0.50,
            'math_reasoning': 0.20,
        },
        'objective': 'Mathematical reasoning',
    },
    {
        'stage': 'Integration',
        'steps': 15000,
        'data_mix': {
            'web_text': 0.30,
            'code': 0.25,
            'math': 0.20,
            'books': 0.15,
            'wikipedia': 0.10,
        },
        'objective': 'Unified capabilities',
    },
]
```

**Rationale**:
- General data builds foundational language understanding
- Specialized domains added progressively
- Final integration stage ensures no catastrophic forgetting

### 3. Length Progression

**Concept**: Start with short sequences, progressively increase length.

```python
# Progressive sequence length curriculum
length_curriculum = [
    {'stage': 1, 'seq_len': 512,  'steps': 10000, 'tokens_per_step': 512 * batch_size},
    {'stage': 2, 'seq_len': 1024, 'steps': 15000, 'tokens_per_step': 1024 * batch_size},
    {'stage': 3, 'seq_len': 2048, 'steps': 25000, 'tokens_per_step': 2048 * batch_size},
    {'stage': 4, 'seq_len': 4096, 'steps': 50000, 'tokens_per_step': 4096 * batch_size},
]

# Benefits:
# - Faster early training (shorter sequences = less compute)
# - Gradual learning of long-range dependencies
# - Smoother transition to final context length
```

**Used by**: Most modern LLMs (Llama, Qwen, Mistral)

**Trade-offs**:
- ✅ Faster initial training (2-3x speedup in early stages)
- ✅ Better long-range dependency learning
- ❌ More complex training pipeline
- ❌ Need to manage sequence length transitions

### 4. Quality Filtering Stages

**Concept**: Progress from broad data to high-quality filtered data.

```python
# Quality-based curriculum (Phi-3 approach)
quality_curriculum = [
    {
        'stage': 'Broad Coverage',
        'steps': 30000,
        'data': web_data,  # Large, diverse, medium quality
        'filters': ['basic_dedup', 'language_filter'],
    },
    {
        'stage': 'High Quality',
        'steps': 40000,
        'data': filtered_web_data,  # Smaller, higher quality
        'filters': ['basic_dedup', 'language_filter', 'quality_filter', 'educational_filter'],
    },
    {
        'stage': 'Synthetic & Curated',
        'steps': 30000,
        'data': curated_data,  # Textbooks, synthetic reasoning, high-quality sources
        'filters': ['all_filters'],
    },
]
```

**Microsoft Phi approach**:
- Stage 1: Filtered web data (1T tokens)
- Stage 2: "Textbook quality" synthetic data (100B tokens)
- Stage 3: Reasoning-focused synthetic data (50B tokens)

Result: Small models (1.5B-14B) with strong reasoning despite size.

---

## Data Mixing Strategies

### 1. Static Data Mixing

**Concept**: Fixed proportions throughout training.

```python
# Example: Llama 3 data mix (approximate)
data_mix = {
    'web_text': 0.50,       # General web (CommonCrawl, etc.)
    'code': 0.20,           # GitHub, StackOverflow
    'wikipedia': 0.10,      # Factual knowledge
    'books': 0.10,          # Long-form text
    'academic': 0.05,       # arXiv, papers
    'other': 0.05,          # Misc sources
}

def create_dataloader(data_sources, mix_proportions, batch_size):
    """
    Sample from multiple sources according to fixed proportions
    """
    # Create weighted sampler
    samplers = []
    for source, proportion in mix_proportions.items():
        sampler = WeightedRandomSampler(
            data_sources[source],
            num_samples=int(total_samples * proportion)
        )
        samplers.append(sampler)

    # Interleave samplers
    dataloader = InterleavedDataLoader(samplers, batch_size=batch_size)
    return dataloader
```

**Pros**:
- Simple to implement
- Reproducible
- Easy to analyze

**Cons**:
- May not be optimal throughout training
- Doesn't adapt to learning progress

### 2. Dynamic Data Mixing

**Concept**: Adjust proportions based on training progress.

```python
# Example: Adaptive data mixing
def adaptive_data_mix(current_step, total_steps, val_losses_by_domain):
    """
    Adjust data mix based on validation losses

    Strategy: Upweight domains where model is struggling
    """

    # Base mix
    base_mix = {
        'web': 0.40,
        'code': 0.20,
        'math': 0.20,
        'books': 0.20,
    }

    # Compute relative losses (higher = more struggle)
    total_loss = sum(val_losses_by_domain.values())
    loss_weights = {
        domain: loss / total_loss
        for domain, loss in val_losses_by_domain.items()
    }

    # Adaptive mix: upweight struggling domains
    alpha = min(current_step / total_steps, 1.0)  # 0 → 1 over training
    adaptive_mix = {}

    for domain in base_mix:
        # Interpolate: base_mix → loss-weighted mix
        adaptive_mix[domain] = (
            (1 - alpha) * base_mix[domain] +
            alpha * loss_weights[domain]
        )

    # Normalize
    total = sum(adaptive_mix.values())
    adaptive_mix = {k: v / total for k, v in adaptive_mix.items()}

    return adaptive_mix
```

**Benefits**:
- Focuses on weak areas
- Adapts to training dynamics
- Can improve sample efficiency

**Challenges**:
- More complex implementation
- Harder to reproduce
- Risk of instability

### 3. Temperature Sampling

**Concept**: Sample domains according to temperature-adjusted probabilities.

```python
def temperature_sampled_mix(domain_sizes, temperature=0.7):
    """
    DoReMi (Domain Reweighting with Minimax Optimization) approach

    Lower temperature: More uniform sampling across domains
    Higher temperature: More proportional to domain size
    """

    # Compute base probabilities (proportional to size)
    total_size = sum(domain_sizes.values())
    base_probs = {
        domain: size / total_size
        for domain, size in domain_sizes.items()
    }

    # Apply temperature
    temp_probs = {
        domain: prob ** (1 / temperature)
        for domain, prob in base_probs.items()
    }

    # Normalize
    total = sum(temp_probs.values())
    final_mix = {k: v / total for k, v in temp_probs.items()}

    return final_mix

# Example
domain_sizes = {
    'web': 1_000_000_000,   # 1B documents
    'code': 100_000_000,    # 100M documents
    'math': 10_000_000,     # 10M documents
    'books': 50_000_000,    # 50M documents
}

# Temperature = 1.0: Proportional sampling
mix_t1 = temperature_sampled_mix(domain_sizes, temperature=1.0)
# Result: web=86%, code=9%, math=1%, books=4%

# Temperature = 0.5: More balanced
mix_t05 = temperature_sampled_mix(domain_sizes, temperature=0.5)
# Result: web=70%, code=15%, math=5%, books=10%

# Temperature = 0.3: Nearly uniform
mix_t03 = temperature_sampled_mix(domain_sizes, temperature=0.3)
# Result: web=40%, code=25%, math=15%, books=20%
```

**Used by**: DoReMi paper, some frontier models

**Benefit**: Prevents over-representation of large domains, ensures rare domain coverage.

### 4. Curriculum with Replay

**Concept**: Revisit earlier stage data periodically to prevent forgetting.

```python
# Progressive curriculum with replay
curriculum_with_replay = [
    {
        'stage': 1,
        'steps': 10000,
        'main_data': {'general': 1.0},
        'replay_data': {},  # No replay yet
    },
    {
        'stage': 2,
        'steps': 15000,
        'main_data': {'code': 0.8},
        'replay_data': {'general': 0.2},  # Replay 20% from stage 1
    },
    {
        'stage': 3,
        'steps': 20000,
        'main_data': {'math': 0.7},
        'replay_data': {'general': 0.15, 'code': 0.15},  # Replay from stages 1 & 2
    },
    {
        'stage': 4,
        'steps': 25000,
        'main_data': {'general': 0.33, 'code': 0.33, 'math': 0.34},
        'replay_data': {},  # Final integration
    },
]
```

**Purpose**: Mitigate catastrophic forgetting when shifting domains.

---

## Practical Curriculum Recipes

### Recipe 1: General-Purpose LLM (Llama-style)

```python
# Llama 3 approximate curriculum
llama_curriculum = [
    # Stage 1: Core pre-training (majority of compute)
    {
        'name': 'Core Pre-training',
        'steps': 900_000,  # ~90% of training
        'seq_len': 8192,
        'data_mix': {
            'web_filtered': 0.50,
            'code': 0.20,
            'wikipedia': 0.10,
            'books': 0.10,
            'academic': 0.05,
            'other': 0.05,
        },
        'learning_rate': 3e-4,
    },

    # Stage 2: Long-context extension
    {
        'name': 'Context Extension',
        'steps': 50_000,  # ~5% of training
        'seq_len': 131_072,  # 128K context
        'data_mix': {
            'long_documents': 0.40,  # Books, papers
            'code_repos': 0.30,      # Full repositories
            'web_filtered': 0.30,
        },
        'learning_rate': 1e-4,  # Lower LR
        'rope_scaling': {'type': 'yarn', 'factor': 16.0},
    },

    # Stage 3: Annealing
    {
        'name': 'Annealing',
        'steps': 50_000,  # ~5% of training
        'seq_len': 8192,
        'data_mix': {  # Back to standard mix
            'web_filtered': 0.50,
            'code': 0.20,
            'wikipedia': 0.10,
            'books': 0.10,
            'academic': 0.05,
            'other': 0.05,
        },
        'learning_rate_schedule': 'cosine_decay',  # Decay to 0
    },
]
```

### Recipe 2: Code-Specialized Model (CodeLlama-style)

```python
# Code-specialized curriculum
code_curriculum = [
    # Stage 1: General foundation
    {
        'name': 'General Foundation',
        'steps': 500_000,
        'data_mix': {
            'general_web': 0.60,
            'code': 0.30,
            'wikipedia': 0.10,
        },
    },

    # Stage 2: Code focus
    {
        'name': 'Code Specialization',
        'steps': 300_000,
        'data_mix': {
            'code': 0.70,
            'code_docs': 0.20,
            'general_web': 0.10,
        },
    },

    # Stage 3: Long-context code
    {
        'name': 'Repository-Level Understanding',
        'steps': 100_000,
        'seq_len': 100_000,  # Very long context
        'data_mix': {
            'full_repos': 0.80,
            'code': 0.20,
        },
    },

    # Stage 4: Instruction tuning (code-specific)
    {
        'name': 'Code Instruction Tuning',
        'steps': 100_000,
        'data_mix': {
            'code_instructions': 1.0,  # HumanEval-style, LeetCode, etc.
        },
    },
]
```

### Recipe 3: Math-Focused Model (Qwen-style)

```python
# Math-focused curriculum
math_curriculum = [
    # Stage 1: General pre-training
    {
        'name': 'General Pre-training',
        'steps': 600_000,
        'data_mix': {
            'general_web': 0.50,
            'code': 0.20,
            'books': 0.15,
            'math': 0.10,
            'science': 0.05,
        },
    },

    # Stage 2: Math-heavy continuation
    {
        'name': 'Math Specialization',
        'steps': 200_000,
        'data_mix': {
            'math_problems': 0.40,
            'math_textbooks': 0.30,
            'code': 0.15,  # Algorithmic reasoning
            'general_web': 0.15,
        },
    },

    # Stage 3: Reasoning curriculum
    {
        'name': 'Reasoning Enhancement',
        'steps': 100_000,
        'data_mix': {
            'chain_of_thought': 0.40,  # Synthetic reasoning chains
            'competition_math': 0.30,   # AMC, AIME, IMO problems
            'code_reasoning': 0.20,
            'general_reasoning': 0.10,
        },
    },

    # Stage 4: Integration
    {
        'name': 'Integration',
        'steps': 100_000,
        'data_mix': {
            'general_web': 0.30,
            'math': 0.25,
            'code': 0.20,
            'reasoning': 0.15,
            'books': 0.10,
        },
    },
]
```

### Recipe 4: Small High-Quality Model (Phi-style)

```python
# Quality-over-quantity curriculum (Phi approach)
phi_curriculum = [
    # Stage 1: Filtered web data
    {
        'name': 'High-Quality Web',
        'steps': 100_000,
        'tokens': 1_000_000_000_000,  # 1T tokens
        'data_mix': {
            'filtered_web': 1.0,  # Aggressive quality filtering
        },
        'filters': [
            'educational_content',
            'low_toxicity',
            'high_coherence',
            'factual_content',
        ],
    },

    # Stage 2: Textbook-quality synthetic
    {
        'name': 'Textbook Quality',
        'steps': 50_000,
        'tokens': 100_000_000_000,  # 100B tokens
        'data_mix': {
            'synthetic_textbooks': 0.60,  # GPT-4 generated
            'real_textbooks': 0.40,
        },
    },

    # Stage 3: Reasoning-focused
    {
        'name': 'Reasoning',
        'steps': 25_000,
        'tokens': 50_000_000_000,  # 50B tokens
        'data_mix': {
            'synthetic_reasoning': 0.70,  # Chain-of-thought examples
            'competition_problems': 0.30,
        },
    },
]
```

---

## Implementation Details

### 1. Transitioning Between Stages

```python
def transition_to_new_stage(model, optimizer, scheduler, new_stage_config):
    """
    Smoothly transition between curriculum stages
    """

    # Option 1: Hard transition (abrupt change)
    # - Simply switch data mix
    # - Keep optimizer state
    dataloader = create_dataloader(new_stage_config['data_mix'])

    # Option 2: Gradual transition (recommended)
    # - Interpolate between old and new mix over N steps
    for step in range(transition_steps):
        alpha = step / transition_steps
        interpolated_mix = interpolate_mixes(
            old_mix, new_stage_config['data_mix'], alpha
        )
        batch = sample_batch(interpolated_mix)
        train_step(batch)

    # Option 3: Optimizer state reset (for major shifts)
    # - Reset optimizer for fresh start in new domain
    if new_stage_config.get('reset_optimizer', False):
        optimizer = create_optimizer(model, new_stage_config['learning_rate'])
        scheduler = create_scheduler(optimizer, new_stage_config)

    return dataloader, optimizer, scheduler
```

### 2. Monitoring Curriculum Health

```python
def monitor_curriculum_progress(model, val_sets_by_domain):
    """
    Track model performance across domains during curriculum
    """

    metrics = {}

    for domain, val_set in val_sets_by_domain.items():
        # Compute domain-specific metrics
        perplexity = evaluate_perplexity(model, val_set)
        metrics[f'{domain}_perplexity'] = perplexity

    # Check for catastrophic forgetting
    for domain in prev_metrics:
        current = metrics[f'{domain}_perplexity']
        previous = prev_metrics[f'{domain}_perplexity']

        if current > previous * 1.3:  # 30% degradation
            alert(f"Catastrophic forgetting in {domain}: "
                  f"{previous:.2f} → {current:.2f}")

    return metrics
```

### 3. Data Mixing Implementation

```python
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler

class MixedDataLoader:
    """
    DataLoader that samples from multiple datasets according to mix proportions
    """

    def __init__(self, datasets, mix_proportions, batch_size):
        self.datasets = datasets
        self.mix_proportions = mix_proportions
        self.batch_size = batch_size

        # Create per-dataset samplers
        self.samplers = {}
        for name, dataset in datasets.items():
            num_samples = int(len(dataset) * mix_proportions[name])
            self.samplers[name] = RandomSampler(
                dataset,
                replacement=True,
                num_samples=num_samples
            )

    def __iter__(self):
        # Create iterators for each dataset
        iters = {
            name: iter(DataLoader(
                self.datasets[name],
                sampler=self.samplers[name],
                batch_size=self.batch_size // len(self.datasets)
            ))
            for name in self.datasets
        }

        # Interleave batches
        while True:
            batch = {}
            for name, it in iters.items():
                try:
                    batch[name] = next(it)
                except StopIteration:
                    # Re-create iterator when exhausted
                    iters[name] = iter(DataLoader(
                        self.datasets[name],
                        sampler=self.samplers[name],
                        batch_size=self.batch_size // len(self.datasets)
                    ))
                    batch[name] = next(iters[name])

            # Concatenate batches from different domains
            yield concatenate_batches(batch.values())
```

---

## Best Practices

### ✅ Do This

1. **Start simple, add complexity**
   - Begin with static mixing
   - Add curriculum if baseline shows domain imbalance

2. **Monitor per-domain metrics**
   - Track validation loss for each domain
   - Detect catastrophic forgetting early

3. **Use progressive sequence length**
   - Start short (512-1024 tokens)
   - Increase gradually
   - Saves compute in early training

4. **Include replay/integration stage**
   - Final stage with balanced mix
   - Prevents forgetting from specialization

5. **Log curriculum transitions**
   - Track which data mix at each step
   - Essential for reproduction and debugging

6. **Validate curriculum design**
   - Run small-scale experiments first
   - Test transitions between stages

### ❌ Avoid This

1. **Don't forget earlier domains**
   - Include replay or final integration stage
   - Monitor per-domain validation loss

2. **Don't make abrupt transitions**
   - Gradual interpolation better than hard switches
   - Prevents training instability

3. **Don't over-complicate**
   - Static mixing works well for most cases
   - Curriculum adds complexity and debugging burden

4. **Don't ignore data quality**
   - Curriculum doesn't fix bad data
   - Quality filtering should come first

5. **Don't use curriculum blindly**
   - Requires careful monitoring and tuning
   - Not always better than static mixing

---

## Case Studies

### Case Study 1: Llama 3 Context Extension

**Approach**: Two-stage curriculum for context extension

**Recipe**:
1. Pre-training at 8K context (90% of compute)
2. Continued pre-training at 128K context (5% of compute)
3. Annealing at 8K context (5% of compute)

**Results**:
- Successfully extended 8K → 128K
- Maintained 8K performance
- ~50x cheaper than full 128K pre-training

**Lesson**: Context extension via curriculum is cost-effective.

### Case Study 2: Phi-3 Quality Curriculum

**Approach**: Quality-focused curriculum (textbook quality → reasoning)

**Recipe**:
1. Filtered web (1T tokens)
2. Synthetic textbook (100B tokens)
3. Reasoning data (50B tokens)

**Results**:
- 3.8B model competitive with 7-13B models
- Strong reasoning despite small size
- Validates quality-over-quantity

**Lesson**: Curriculum quality matters more than quantity for small models.

### Case Study 3: Failed Math Curriculum

**Problem**: Specialized on math too early, forgot general capabilities.

**Approach attempted**:
1. General pre-training (50K steps)
2. Math-only (100K steps) ← Problem here
3. No integration stage ← Problem here

**Results**:
- Excellent math performance
- Terrible general conversation
- Catastrophic forgetting

**Fix**:
- Reduced math-only stage (50K steps)
- Added replay (20% general data)
- Added final integration stage

**Lesson**: Always include replay or integration to prevent forgetting.

---

## Resources

### Papers

**Curriculum Learning**:
- [Curriculum Learning](https://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf) - Original concept
- [On the Power of Curriculum Learning](https://arxiv.org/abs/1904.03626) - Analysis

**Data Mixing**:
- [DoReMi: Domain Reweighting with Minimax Optimization](https://arxiv.org/abs/2305.10429) - Temperature sampling
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) - Data mixing strategies

**Model-Specific**:
- [Llama 3 Model Card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md) - Training curriculum
- [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) - Phi-1 approach
- [Qwen 2.5 Technical Report](https://arxiv.org/abs/2407.10671) - Multi-stage curriculum

### Tools

- [datasets](https://github.com/huggingface/datasets) - HuggingFace dataset mixing
- [DataTrove](https://github.com/huggingface/datatrove) - Data processing pipelines
- [Dolma](https://github.com/allenai/dolma) - Data curation toolkit

---

**Related Documentation**:
- [Data Preparation](data-preparation.md) - Data collection and preprocessing
- [Context Extension](context-extension.md) - Length curriculum details
- [Training Stability](training-stability.md) - Handling curriculum transitions
- [Scaling Laws](scaling-laws.md) - Compute-optimal allocation
