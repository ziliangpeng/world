# Scaling Laws for Language Models

Scaling laws are the empirical relationships that govern how language model performance improves with compute, data, and parameters. Understanding these laws transformed LLM development from art to engineering—enabling researchers to predict model capabilities before training and optimize resource allocation for maximum performance.

---

## Why Scaling Laws Matter

Before scaling laws, model development was trial-and-error: train a model, evaluate, adjust, repeat. This was prohibitively expensive at scale—a failed 175B parameter run could cost millions of dollars.

Scaling laws enable:

| Capability | Practical Impact |
|------------|------------------|
| **Performance prediction** | Know loss before training completes |
| **Compute allocation** | Optimal split between parameters and data |
| **Budget planning** | Predict cost to achieve target capability |
| **Architecture comparison** | Fair evaluation across scales |
| **Training efficiency** | Avoid over/under-training |

**The core insight**: Model loss follows predictable power laws. Given any two of {compute, parameters, data}, you can predict optimal third variable and expected loss.

---

## Historical Evolution

### Phase 1: Initial Discovery (2017-2020)

**Early Observations**

Researchers noticed logarithmic improvements with scale, but lacked rigorous quantification. The [Transformer paper](https://arxiv.org/abs/1706.03762) (2017) and subsequent work showed bigger models performed better, but the relationship wasn't formalized.

**[Kaplan Scaling Laws](https://arxiv.org/abs/2001.08361)** (January 2020) - OpenAI

The seminal paper that established the field. Key findings:

**Loss follows power laws**:
```
L(N) = (N_c / N)^α_N     # Loss vs parameters
L(D) = (D_c / D)^α_D     # Loss vs data
L(C) = (C_c / C)^α_C     # Loss vs compute
```

Where:
- L = cross-entropy loss
- N = number of parameters
- D = dataset size (tokens)
- C = compute (FLOPs)
- α = scaling exponent
- Subscript c = critical constants

**Discovered exponents** (Kaplan):
| Relationship | Exponent | Interpretation |
|--------------|----------|----------------|
| Loss vs Parameters | α_N ≈ 0.076 | 10× params → ~16% loss reduction |
| Loss vs Data | α_D ≈ 0.095 | 10× data → ~19% loss reduction |
| Loss vs Compute | α_C ≈ 0.050 | 10× compute → ~11% loss reduction |

**Critical implication**: Parameters scale faster than data for fixed compute. Kaplan recommended:
```
N ∝ C^0.73    # Parameters should scale with compute^0.73
D ∝ C^0.27    # Data should scale with compute^0.27
```

This suggested **prioritizing model size over data size**—train larger models on relatively less data.

**GPT-3 followed Kaplan**: 175B parameters trained on only 300B tokens (~1.7 tokens/parameter).

### Phase 2: The Chinchilla Revolution (2022)

**[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)** - Hoffmann et al., DeepMind

The Chinchilla paper fundamentally revised Kaplan's recommendations:

**Key finding**: Kaplan models were severely **undertrained on data**.

**Revised exponents**:
| Relationship | Kaplan | Chinchilla |
|--------------|--------|------------|
| N ∝ C^x | x = 0.73 | x = 0.50 |
| D ∝ C^x | x = 0.27 | x = 0.50 |

**New recommendation**: Parameters and data should scale **equally** with compute.

```
Optimal: N ≈ D/20    # ~20 tokens per parameter

For compute budget C:
N_opt ∝ C^0.5
D_opt ∝ C^0.5
```

**The Chinchilla Rule**: Train on ~20 tokens per parameter for compute-optimal performance.

| Model | Parameters | Training Tokens | Tokens/Param | Status |
|-------|------------|-----------------|--------------|--------|
| GPT-3 | 175B | 300B | 1.7 | Severely undertrained |
| Gopher | 280B | 300B | 1.1 | Severely undertrained |
| Chinchilla | 70B | 1.4T | 20 | Compute-optimal |
| PaLM | 540B | 780B | 1.4 | Undertrained |

**Chinchilla's proof**: A 70B model trained compute-optimally outperformed the 280B Gopher on all benchmarks—same compute, 4× fewer parameters, 4× more data.

### Phase 3: Beyond Chinchilla (2023-Present)

**LLaMA's Deviation** (February 2023)

Meta deliberately violated Chinchilla by training longer:

| Model | Params | Tokens | Tokens/Param | Chinchilla Optimal |
|-------|--------|--------|--------------|-------------------|
| LLaMA-7B | 7B | 1T | 143 | 140B |
| LLaMA-13B | 13B | 1T | 77 | 260B |
| LLaMA-65B | 65B | 1.4T | 22 | 1.3T |

**Rationale**: Chinchilla optimizes for training compute, but **inference compute** also matters. A smaller model trained longer:
- Costs more to train (violates Chinchilla)
- Costs less to deploy (wins on total cost of ownership)
- Enables broader accessibility

**LLaMA-2 continued this**: 70B model on 2T tokens (29 tokens/param).

**[Chinchilla Scaling: A Replication Attempt](https://arxiv.org/abs/2404.10102)** (2024)

Recent work questions exact Chinchilla ratios:
- Optimal ratio may be 40-70 tokens/param (higher than 20)
- Depends on data quality and domain
- Inference costs shift optimal point

**Current Industry Practice**

| Model | Params | Tokens | Tokens/Param | Strategy |
|-------|--------|--------|--------------|----------|
| LLaMA 3 8B | 8B | 15T+ | ~1900 | Extreme overtrain |
| Mistral 7B | 7B | ~8T? | ~1100 | Heavy overtrain |
| Qwen2.5 7B | 7B | 18T | ~2500 | Extreme overtrain |
| Gemma 2 2B | 2B | 2T | 1000 | Heavy overtrain |

**Modern consensus**: Chinchilla is compute-optimal for training, but inference-optimal requires overtraining by 10-100×.

---

## The Mathematics

### Power Law Formulation

The general scaling law:
```
L(N, D) = E + A/N^α + B/D^β
```

Where:
- **E**: Irreducible entropy (perfect model on infinite data)
- **A/N^α**: Loss from finite parameters
- **B/D^β**: Loss from finite data
- α, β: Scaling exponents

**Fitted values** (Chinchilla):
```
E ≈ 1.69
A ≈ 406.4
B ≈ 410.7
α ≈ 0.34
β ≈ 0.28
```

### Compute-Optimal Frontier

Given compute budget C ≈ 6ND (for Transformer FLOPs):

```python
def chinchilla_optimal(C):
    """Compute-optimal N and D for budget C."""
    # From Chinchilla Approach 3
    a = 0.5  # N scales as C^a
    b = 0.5  # D scales as C^b

    # Fitted constants
    N_opt = 0.7 * (C ** a)
    D_opt = 0.7 * (C ** b)

    return N_opt, D_opt
```

### Predicting Loss

```python
def predicted_loss(N, D):
    """Predict loss for given N parameters and D tokens."""
    E = 1.69
    A, B = 406.4, 410.7
    alpha, beta = 0.34, 0.28

    L = E + A / (N ** alpha) + B / (D ** beta)
    return L

# Example: Predict LLaMA-7B loss
loss = predicted_loss(N=7e9, D=1e12)
print(f"Predicted loss: {loss:.3f}")  # ~1.85 nats
```

### Compute Estimation

For a dense Transformer with N parameters trained on D tokens:

```
Forward pass: ~2N FLOPs per token
Backward pass: ~4N FLOPs per token
Total: ~6N FLOPs per token
Training compute: C ≈ 6ND FLOPs
```

**Example calculations**:

| Model | N | D | C (FLOPs) | C (GPU-hours, A100) |
|-------|---|---|-----------|---------------------|
| GPT-3 | 175B | 300B | 3.15×10²³ | ~3,640,000 |
| LLaMA-65B | 65B | 1.4T | 5.46×10²³ | ~6,300,000 |
| Chinchilla-70B | 70B | 1.4T | 5.88×10²³ | ~6,800,000 |
| LLaMA 3-70B | 70B | 15T | 6.3×10²⁴ | ~73,000,000 |

---

## Scaling Law Variants

### Downstream Task Scaling

Loss scaling doesn't directly predict task performance. Research has explored:

**[Emergent Abilities](https://arxiv.org/abs/2206.07682)** (2022):

Some capabilities appear suddenly at scale:
- Multi-step arithmetic: emerges ~100B parameters
- Word unscrambling: emerges ~10B parameters
- Chain-of-thought: emerges ~60B parameters

**Critique**: Later work suggested "emergence" may be measurement artifact—continuous improvement, sharp thresholds in metrics.

**[Scaling Laws for Transfer](https://arxiv.org/abs/2102.01293)** (2021):

Pre-training loss predicts fine-tuning performance, but with task-specific scaling:
```
Performance = f(L_pretrain) ≈ a - b × L_pretrain
```

### Architecture-Specific Scaling

Different architectures have different scaling exponents:

| Architecture | α_N (params) | Notes |
|--------------|--------------|-------|
| Transformer | 0.076 (Kaplan) / 0.34 (Chinchilla) | Standard |
| Mixture of Experts | ~0.06 | Better scaling per FLOP |
| State Space Models | Similar to Transformer | Limited data |
| Linear Attention | Slightly worse | Ongoing research |

**MoE Advantage**: Sparse models achieve better loss per FLOP but require more memory.

### Inference Scaling Laws

**[LLaMA Inference Scaling](https://arxiv.org/abs/2402.16363)** (2024):

Total cost = Training cost + Inference cost × Expected queries
```
C_total = C_train + C_infer × Q
```

For high-query applications (ChatGPT-like), smaller overtrained models are optimal:
- 7B model costs ~100× less per query than 70B
- Overtraining 7B by 10× costs ~10% of training 70B
- Break-even after ~10M queries

### Multimodal Scaling

Vision-language models follow modified laws:
- Image tokens scale differently than text
- Cross-modal attention has unique scaling
- [LLaVA scaling](https://arxiv.org/abs/2310.03744) shows vision encoders matter less at scale

---

## Practical Applications

### Recommended Configurations by Budget

**GPU-hours to compute budget conversion** (A100-80GB):
- 1 A100 ≈ 312 TFLOP/s (bf16)
- 1 GPU-hour ≈ 1.1 × 10¹⁸ FLOPs (at ~50% utilization)

| Budget (A100-hours) | Chinchilla-Optimal | Inference-Optimal | Use Case |
|---------------------|-------------------|-------------------|----------|
| 1K | 300M params, 6B tokens | 125M params, 50B tokens | Experimentation |
| 10K | 1B params, 20B tokens | 350M params, 150B tokens | Small production |
| 100K | 3B params, 60B tokens | 1.3B params, 500B tokens | Medium production |
| 1M | 10B params, 200B tokens | 7B params, 1T tokens | LLaMA-7B class |
| 10M | 30B params, 600B tokens | 13B params, 3T tokens | LLaMA-13B class |
| 100M | 100B params, 2T tokens | 70B params, 15T tokens | Frontier class |

### Quick Reference: "Given X, Do Y"

**I have $10K and access to cloud GPUs. What can I train?**
- Budget: ~1,000 A100-hours at ~$3/hr spot pricing
- Recommendation: 350M-1B model on 20-50B tokens
- Quality: Useful for specific tasks, not general-purpose

**I have $100K. What's achievable?**
- Budget: ~10,000-30,000 A100-hours
- Recommendation: 3-7B model on 100-300B tokens
- Quality: Competitive with open-source 7B models

**I have $1M. What should I build?**
- Budget: ~100,000-300,000 A100-hours
- Recommendation: 7-13B model on 1-2T tokens (inference-optimal) OR 30B on 600B (Chinchilla)
- Quality: State-of-the-art for size class

**I have $10M+. Frontier territory?**
- Budget: 1M+ A100-hours
- Recommendation: 70B+ model on 10T+ tokens
- Quality: Frontier-competitive

### Budget Allocation

Given a training budget, how to allocate?

**Training-focused** (Chinchilla-optimal):
```python
def training_optimal_allocation(budget_flops):
    # C = 6ND, N = D/20 → D = sqrt(C*20/6), N = D/20
    D = math.sqrt(budget_flops * 20 / 6)
    N = D / 20
    return N, D
```

**Inference-focused** (LLaMA-style):
```python
def inference_optimal_allocation(budget_flops, expected_queries):
    # Favor smaller N, larger D
    # Heuristic: 100-500 tokens per parameter
    # Adjust based on expected inference volume
    target_ratio = 100 + 400 * min(1, expected_queries / 1e9)
    N = budget_flops / (6 * target_ratio * target_ratio)
    D = N * target_ratio
    return N, D
```

### Practical Budget Calculator

```python
def estimate_training(params_billions, tokens_trillions,
                      gpu_type="A100", utilization=0.5):
    """
    Estimate training requirements.

    Args:
        params_billions: Model parameters in billions
        tokens_trillions: Training tokens in trillions
        gpu_type: "A100" (312 TFLOP/s) or "H100" (990 TFLOP/s)
        utilization: Expected MFU (model FLOP utilization)

    Returns:
        dict with GPU-hours, cost estimates, wall-clock time
    """
    # Compute FLOPs
    N = params_billions * 1e9
    D = tokens_trillions * 1e12
    compute_flops = 6 * N * D

    # GPU throughput
    throughput = {
        "A100": 312e12,  # TFLOP/s
        "H100": 990e12,
    }[gpu_type] * utilization

    # Calculate time
    gpu_seconds = compute_flops / throughput
    gpu_hours = gpu_seconds / 3600

    # Cost estimates (spot pricing)
    hourly_cost = {"A100": 2.0, "H100": 4.0}[gpu_type]

    return {
        "compute_flops": f"{compute_flops:.2e}",
        "gpu_hours": f"{gpu_hours:,.0f}",
        "cost_estimate": f"${gpu_hours * hourly_cost:,.0f}",
        "wall_clock_days_64_gpus": f"{gpu_hours / (64 * 24):,.1f}",
        "wall_clock_days_256_gpus": f"{gpu_hours / (256 * 24):,.1f}",
    }

# Examples
print(estimate_training(7, 1))    # 7B, 1T tokens
# {'compute_flops': '4.20e+22', 'gpu_hours': '74,786',
#  'cost_estimate': '$149,573', 'wall_clock_days_64_gpus': '48.7',
#  'wall_clock_days_256_gpus': '12.2'}

print(estimate_training(70, 15))  # 70B, 15T tokens
# {'compute_flops': '6.30e+24', 'gpu_hours': '11,217,949',
#  'cost_estimate': '$22,435,897', 'wall_clock_days_64_gpus': '7,303.9',
#  'wall_clock_days_256_gpus': '1,826.0'}
```

### When to Stop Training

**Loss-based stopping**:
```python
def should_stop(current_loss, predicted_loss, tokens_seen, budget):
    # Stop when within 1% of predicted optimal
    if current_loss < predicted_loss * 1.01:
        return True
    # Stop when budget exhausted
    if tokens_seen >= budget:
        return True
    return False
```

**Convergence detection**:
- Loss plateau for >1% of training
- Gradient norm stabilization
- Validation loss divergence from training loss

### Model Selection

Given deployment constraints:

| Constraint | Recommendation |
|------------|----------------|
| Memory limited | Smaller N, more D (overtrain) |
| Latency critical | Smallest N meeting quality bar |
| Quality critical | Largest N within budget |
| Cost critical | Compute-optimal N/D balance |

---

## Limitations and Caveats

### What Scaling Laws Don't Capture

1. **Data quality**: Laws assume fixed quality; improvements can beat predictions
2. **Architecture innovations**: GQA, MoE change the curves
3. **Task-specific performance**: Loss ≠ benchmark scores
4. **Emergent behaviors**: Discontinuous capabilities
5. **Safety/alignment**: More scale can increase harm

### Known Issues

**Extrapolation risk**: Laws fit on smaller models, extrapolating to 10× larger is uncertain.

**Data constraints**: At scale, unique high-quality data becomes limiting factor—can't just scale D indefinitely.

**Diminishing returns**: Exponents mean each 10× improvement costs more. Eventually impractical.

### Active Research

1. **Optimal data mixtures**: How does domain weighting interact with scaling?
2. **Architecture-aware laws**: Unified theory across Transformers, SSMs, MoE
3. **Capability-specific scaling**: Different tasks scale differently
4. **Post-training scaling**: How does RLHF/DPO effort scale?

---

## Industry Impact

### How Labs Use Scaling Laws

**OpenAI**: Pioneered with GPT-3, now uses for capability forecasting
**DeepMind**: Chinchilla changed industry practice, continued with Gemini
**Meta**: LLaMA explicitly optimized for inference, not training
**Anthropic**: Uses scaling laws for safety research and capability prediction

### The "Scaling Hypothesis"

The belief that continued scaling will yield AGI-like capabilities:
- **Proponents**: Sufficient scale may be sufficient for general intelligence
- **Critics**: Architecture/data innovations equally important; ceiling exists

Scaling laws neither prove nor disprove this—they show reliable improvement within tested regime.

---

## Future Directions

### Near-term (2025)

1. **Data wall**: Unique internet data approaching exhaustion; synthetic data scaling laws needed
2. **Inference optimization**: Distillation scaling laws for deployment
3. **Efficient architectures**: Scaling laws for MoE, SSM hybrids
4. **Multimodal integration**: Unified scaling across modalities

### Research Frontiers

1. **Post-training scaling**: How do RLHF compute and human feedback scale?
2. **Agent capabilities**: Scaling laws for tool use, planning
3. **Safety scaling**: Do risks scale with capabilities?
4. **Mechanistic understanding**: Why do power laws emerge?

### Open Questions

1. **Is there a ceiling?** Will scaling laws continue to 10¹⁸ FLOPs?
2. **Data efficiency**: Can we beat the scaling laws with better algorithms?
3. **Compute-capability mapping**: When does X capability emerge?
4. **Economic limits**: When is further scaling not cost-effective?

---

## Sources

### Foundational Papers
- [Scaling Laws for Neural Language Models (Kaplan)](https://arxiv.org/abs/2001.08361) - OpenAI, 2020
- [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) - DeepMind, 2022
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - Meta, 2023

### Extensions and Analysis
- [Scaling Laws for Transfer](https://arxiv.org/abs/2102.01293) - OpenAI, 2021
- [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) - Google, 2022
- [Chinchilla Scaling: A Replication Attempt](https://arxiv.org/abs/2404.10102) - Epoch AI, 2024

### Inference-Aware Scaling
- [Beyond Chinchilla-Optimal: Accounting for Inference](https://arxiv.org/abs/2401.00448) - 2024
- [LLaMA Inference Scaling](https://arxiv.org/abs/2402.16363) - 2024

### Guides and Explanations
- [Scaling Laws for Deep Learning](https://epochai.org/blog/scaling-laws-literature-review) - Epoch AI
- [The Scaling Hypothesis](https://gwern.net/scaling-hypothesis) - Gwern
- [AI and Compute](https://openai.com/research/ai-and-compute) - OpenAI
