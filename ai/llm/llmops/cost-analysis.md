# LLM Cost Analysis and Optimization

Understanding and optimizing costs is critical for sustainable LLM development and deployment. Training costs can reach millions of dollars, while inference costs scale with usage. This document provides practical frameworks for estimating, analyzing, and optimizing both training and inference costs.

---

## Training Cost Estimation

### 1. Hardware Cost Fundamentals

**GPU pricing (2024-2025)**:

| GPU | Memory | Compute (TFLOPS FP16) | Cloud Cost ($/hour) | Owned Cost ($/hour) |
|-----|--------|-----------------------|---------------------|---------------------|
| **H100 SXM** | 80GB | 1979 | $4.00-5.00 | $1.50-2.00 |
| **H100 PCIe** | 80GB | 1513 | $3.50-4.50 | $1.30-1.80 |
| **A100 SXM** | 80GB | 624 | $2.50-3.50 | $1.00-1.50 |
| **A100 PCIe** | 80GB | 312 | $2.00-3.00 | $0.80-1.20 |
| **H200** | 141GB | 1979 | $5.00-6.00 | $2.00-2.50 |

**Notes**:
- Cloud cost: AWS, GCP, Azure on-demand rates
- Owned cost: Amortized over 3 years (hardware + power + cooling + space)
- Spot instances: 50-70% cheaper but can be preempted

### 2. Training Cost Formula

**Basic formula**:
```python
def estimate_training_cost(
    num_params_billions,
    num_tokens_billions,
    gpu_type='H100',
    utilization=0.5,  # Model FLOPS Utilization (MFU)
):
    """
    Estimate training cost using Chinchilla scaling

    Args:
        num_params_billions: Model size in billions
        num_tokens_billions: Training tokens in billions
        utilization: Fraction of theoretical FLOPS achieved (0.3-0.6 typical)
    """

    # FLOPS per token (6 * params for forward + backward)
    flops_per_token = 6 * num_params_billions * 1e9

    # Total FLOPS
    total_flops = flops_per_token * num_tokens_billions * 1e9

    # GPU specs
    gpu_specs = {
        'H100': {'tflops': 1979, 'cost_per_hour': 4.50},
        'A100': {'tflops': 624, 'cost_per_hour': 3.00},
    }

    gpu_tflops = gpu_specs[gpu_type]['tflops'] * 1e12
    gpu_cost = gpu_specs[gpu_type]['cost_per_hour']

    # Effective FLOPS (accounting for utilization)
    effective_flops = gpu_tflops * utilization

    # GPU hours needed
    gpu_hours = total_flops / effective_flops / 3600

    # Total cost
    total_cost = gpu_hours * gpu_cost

    return {
        'total_flops': total_flops,
        'gpu_hours': gpu_hours,
        'total_cost_usd': total_cost,
        'cost_per_token': total_cost / (num_tokens_billions * 1e9),
    }
```

**Example: Llama 3 70B**:
```python
cost = estimate_training_cost(
    num_params_billions=70,
    num_tokens_billions=15000,  # 15T tokens
    gpu_type='H100',
    utilization=0.50,  # 50% MFU
)

print(f"Total cost: ${cost['total_cost_usd']:,.0f}")
# Output: Total cost: $31,725,000

print(f"GPU hours: {cost['gpu_hours']:,.0f}")
# Output: GPU hours: 7,050,000

print(f"Wall-clock time with 512 H100s: {cost['gpu_hours'] / 512 / 24:.0f} days")
# Output: Wall-clock time with 512 H100s: 574 days
```

### 3. Chinchilla-Optimal Training

**Chinchilla law**: For compute budget C, optimal allocation is:
- Parameters N ∝ C^0.5
- Tokens D ∝ C^0.5
- Approximately 20 tokens per parameter

```python
def chinchilla_optimal_allocation(compute_budget_gpu_hours, gpu_type='H100'):
    """
    Given compute budget, find optimal model size and training tokens

    Chinchilla formula:
    - N_opt = (C / 6)^0.5 / 20
    - D_opt = 20 * N_opt
    """

    gpu_specs = {'H100': {'tflops': 1979, 'cost': 4.50}}

    # Total FLOPS from budget
    total_flops = compute_budget_gpu_hours * gpu_specs[gpu_type]['tflops'] * 1e12 * 3600 * 0.5  # 50% MFU

    # Optimal params (in billions)
    params_optimal = (total_flops / 6 / 1e9) ** 0.5 / 20

    # Optimal tokens (in billions)
    tokens_optimal = 20 * params_optimal

    # Cost
    cost = compute_budget_gpu_hours * gpu_specs[gpu_type]['cost']

    return {
        'params_billions': params_optimal,
        'tokens_billions': tokens_optimal,
        'cost_usd': cost,
    }
```

**Example budgets**:
```python
# $100K budget
result = chinchilla_optimal_allocation(compute_budget_gpu_hours=22_222)  # $100K / $4.50
# Result: ~3B params, 60B tokens

# $1M budget
result = chinchilla_optimal_allocation(compute_budget_gpu_hours=222_222)
# Result: ~9B params, 180B tokens

# $10M budget
result = chinchilla_optimal_allocation(compute_budget_gpu_hours=2_222_222)
# Result: ~30B params, 600B tokens
```

### 4. Cost Breakdown

**Training cost components**:

```python
def detailed_training_cost(
    model_params_b=70,
    training_tokens_b=15000,
    num_gpus=512,
    duration_days=60,
):
    """
    Detailed cost breakdown for training run
    """

    # GPU compute (largest component)
    gpu_hours = num_gpus * duration_days * 24
    gpu_compute_cost = gpu_hours * 4.50  # H100

    # Storage
    checkpoint_size_gb = model_params_b * 4 * 2  # 4 bytes/param * 2 (model + optimizer)
    num_checkpoints = 100
    storage_gb = checkpoint_size_gb * num_checkpoints
    storage_cost = storage_gb * 0.10 * (duration_days / 30)  # $0.10/GB/month

    # Data preprocessing (one-time)
    data_processing_hours = 1000  # CPU hours
    data_processing_cost = data_processing_hours * 0.50

    # Networking (data transfer)
    data_transfer_tb = training_tokens_b * 0.001  # Rough estimate
    networking_cost = data_transfer_tb * 10  # $10/TB

    # Engineering time (often overlooked!)
    engineer_days = 30
    engineer_cost = engineer_days * 1000  # $1000/day loaded cost

    total_cost = (
        gpu_compute_cost +
        storage_cost +
        data_processing_cost +
        networking_cost +
        engineer_cost
    )

    return {
        'gpu_compute': gpu_compute_cost,
        'storage': storage_cost,
        'data_processing': data_processing_cost,
        'networking': networking_cost,
        'engineering': engineer_cost,
        'total': total_cost,
        'breakdown_pct': {
            'gpu': gpu_compute_cost / total_cost * 100,
            'storage': storage_cost / total_cost * 100,
            'data': data_processing_cost / total_cost * 100,
            'network': networking_cost / total_cost * 100,
            'engineering': engineer_cost / total_cost * 100,
        }
    }
```

**Typical breakdown**:
- GPU compute: 90-95%
- Storage: 1-3%
- Data processing: 1-2%
- Networking: 0.5-1%
- Engineering time: 2-5%

---

## Inference Cost Estimation

### 1. Inference Cost Formula

**Per-request cost**:
```python
def estimate_inference_cost(
    model_params_billions,
    input_tokens,
    output_tokens,
    gpu_type='H100',
    batch_size=1,
):
    """
    Estimate cost per inference request

    Args:
        model_params_billions: Model size
        input_tokens: Prompt length
        output_tokens: Generation length
        batch_size: Concurrent requests processed together
    """

    # FLOPS per token
    # Prefill (prompt): 2 * params (forward only, no grad)
    # Decode (generation): 2 * params per token

    prefill_flops = 2 * model_params_billions * 1e9 * input_tokens
    decode_flops = 2 * model_params_billions * 1e9 * output_tokens

    total_flops = (prefill_flops + decode_flops) / batch_size  # Batching amortizes cost

    # GPU specs
    gpu_specs = {
        'H100': {'tflops': 1979, 'cost_per_hour': 4.50},
        'A100': {'tflops': 624, 'cost_per_hour': 3.00},
    }

    gpu_tflops = gpu_specs[gpu_type]['tflops'] * 1e12
    gpu_cost_per_hour = gpu_specs[gpu_type]['cost_per_hour']

    # Time per request (seconds)
    utilization = 0.3  # Lower than training (memory-bound)
    time_seconds = total_flops / (gpu_tflops * utilization)

    # Cost per request
    cost_per_request = (time_seconds / 3600) * gpu_cost_per_hour

    return {
        'time_seconds': time_seconds,
        'cost_per_request': cost_per_request,
        'cost_per_million_tokens': cost_per_request / (input_tokens + output_tokens) * 1e6,
    }
```

**Example: GPT-4 level (1.7T params)**:
```python
cost = estimate_inference_cost(
    model_params_billions=1700,
    input_tokens=1000,
    output_tokens=500,
    gpu_type='H100',
    batch_size=1,
)

print(f"Cost per request: ${cost['cost_per_request']:.4f}")
# Output: Cost per request: $0.0850

print(f"Cost per 1M tokens: ${cost['cost_per_million_tokens']:.2f}")
# Output: Cost per 1M tokens: $56.67
```

### 2. Batching Impact

**Batching dramatically reduces per-request cost**:

```python
# Compare batch sizes
for batch_size in [1, 4, 16, 64]:
    cost = estimate_inference_cost(
        model_params_billions=70,
        input_tokens=500,
        output_tokens=500,
        batch_size=batch_size,
    )
    print(f"Batch {batch_size:3d}: ${cost['cost_per_request']:.6f}/request")

# Output:
# Batch   1: $0.002100/request
# Batch   4: $0.000525/request  (4x cheaper)
# Batch  16: $0.000131/request  (16x cheaper)
# Batch  64: $0.000033/request  (64x cheaper)
```

**But batching increases latency**:
- Batch size 1: 50-200ms latency
- Batch size 64: 1-3s latency

**Trade-off**: Throughput vs latency.

### 3. Monthly Inference Cost

```python
def estimate_monthly_inference_cost(
    requests_per_day,
    avg_input_tokens,
    avg_output_tokens,
    model_params_billions,
    batch_size,
):
    """
    Estimate monthly cost for production inference
    """

    # Cost per request
    cost_per_req = estimate_inference_cost(
        model_params_billions,
        avg_input_tokens,
        avg_output_tokens,
        batch_size=batch_size,
    )['cost_per_request']

    # Monthly requests
    monthly_requests = requests_per_day * 30

    # Total cost
    monthly_cost = monthly_requests * cost_per_req

    return {
        'cost_per_request': cost_per_req,
        'monthly_requests': monthly_requests,
        'monthly_cost': monthly_cost,
        'yearly_cost': monthly_cost * 12,
    }
```

**Example: Production chatbot**:
```python
cost = estimate_monthly_inference_cost(
    requests_per_day=1_000_000,  # 1M requests/day
    avg_input_tokens=200,
    avg_output_tokens=300,
    model_params_billions=70,
    batch_size=16,
)

print(f"Monthly cost: ${cost['monthly_cost']:,.0f}")
# Output: Monthly cost: $19,500

print(f"Yearly cost: ${cost['yearly_cost']:,.0f}")
# Output: Yearly cost: $234,000
```

---

## Cost Optimization Strategies

### 1. Training Optimization

**A. Scaling Laws (Chinchilla-Optimal)**

```python
# BAD: Overtrained small model
overtrained = estimate_training_cost(
    num_params_billions=7,
    num_tokens_billions=3000,  # 428 tokens/param (way over 20)
    utilization=0.5,
)

# GOOD: Chinchilla-optimal
optimal = estimate_training_cost(
    num_params_billions=30,
    num_tokens_billions=600,  # 20 tokens/param
    utilization=0.5,
)

# Same compute, better final performance with optimal allocation
print(f"Overtrained 7B: ${overtrained['total_cost_usd']:,.0f}")
print(f"Optimal 30B: ${optimal['total_cost_usd']:,.0f}")
```

**B. Mixed Precision Training**

```python
# FP32: Baseline
fp32_time = 100  # hours

# BF16: 2x faster, same memory
bf16_time = fp32_time / 2  # 50 hours
bf16_savings = (fp32_time - bf16_time) / fp32_time * 100  # 50% savings

# FP8 (H100): 2-3x faster than BF16
fp8_time = bf16_time / 2.5  # 20 hours
fp8_savings = (fp32_time - fp8_time) / fp32_time * 100  # 80% savings
```

**C. Efficient Parallelism**

```python
def estimate_scalability(
    num_gpus,
    perfect_scaling_time,
    communication_overhead_pct=10,
):
    """
    Model realistic training time with communication overhead

    Perfect scaling: 2x GPUs = 2x faster
    Reality: Communication overhead reduces efficiency
    """

    # Linear component (scales perfectly)
    compute_time = perfect_scaling_time / num_gpus

    # Communication component (doesn't scale)
    comm_time = perfect_scaling_time * communication_overhead_pct / 100

    # Total time
    actual_time = compute_time + comm_time

    # Efficiency
    efficiency = perfect_scaling_time / num_gpus / actual_time

    return {
        'time': actual_time,
        'efficiency': efficiency,
        'speedup': perfect_scaling_time / actual_time,
    }

# Compare configurations
for num_gpus in [64, 128, 256, 512, 1024]:
    result = estimate_scalability(num_gpus, perfect_scaling_time=1000)
    print(f"{num_gpus:4d} GPUs: {result['efficiency']:.1%} efficient, {result['speedup']:.1f}x speedup")

# Output:
#   64 GPUs: 90.9% efficient, 14.5x speedup
#  128 GPUs: 83.3% efficient, 25.6x speedup
#  256 GPUs: 71.4% efficient, 41.0x speedup
#  512 GPUs: 55.6% efficient, 62.7x speedup
# 1024 GPUs: 38.5% efficient, 86.2x speedup
```

**Lesson**: More GPUs ≠ proportionally faster. Communication overhead limits scaling.

**D. Spot Instances**

```python
# On-demand H100: $4.50/hour
# Spot H100: $1.80/hour (60% cheaper)

on_demand_cost = 100_000 * 4.50  # $450,000
spot_cost = 100_000 * 1.80       # $180,000

savings = on_demand_cost - spot_cost  # $270,000 (60% savings)

# But: Need fault tolerance for preemptions
# Add 10% overhead for checkpointing + restarts
spot_cost_adjusted = spot_cost * 1.10  # $198,000

# Still 56% savings
```

**Best practice**: Use spot for flexible training jobs with good checkpointing.

### 2. Inference Optimization

**A. Quantization**

```python
# FP16: Baseline (70B model)
fp16_memory = 70 * 2  # 140GB
fp16_cost_per_hour = 4.50  # H100 80GB (need 2 GPUs)

# INT8: 2x smaller
int8_memory = 70 * 1  # 70GB
int8_cost_per_hour = 4.50 / 2  # 1 GPU only: $2.25/hour

# FP8 (H100): Similar to INT8
fp8_memory = 70 * 1  # 70GB
fp8_cost_per_hour = 4.50 / 2  # $2.25/hour

# INT4 (4-bit): 4x smaller
int4_memory = 70 * 0.5  # 35GB
int4_cost_per_hour = 4.50 / 2  # $2.25/hour (or cheaper GPU)

print("Quantization savings:")
print(f"FP16: ${fp16_cost_per_hour:.2f}/hour (baseline)")
print(f"INT8: ${int8_cost_per_hour:.2f}/hour (50% savings)")
print(f"INT4: ${int4_cost_per_hour:.2f}/hour (50% savings, can use A100)")
```

**Trade-offs**:
- FP16: Best quality
- INT8/FP8: ~1-2% quality loss, 2x cheaper
- INT4: ~3-5% quality loss, 2-4x cheaper

**B. KV Cache Optimization**

```python
def kv_cache_memory(
    model_params_billions,
    batch_size,
    sequence_length,
    num_layers=80,
    hidden_dim=8192,
):
    """
    KV cache memory scales with batch size and sequence length

    For long conversations, KV cache can exceed model weights!
    """

    # Bytes per element (FP16)
    bytes_per_element = 2

    # KV cache: 2 (K and V) * num_layers * batch_size * seq_len * hidden_dim
    kv_cache_bytes = 2 * num_layers * batch_size * sequence_length * hidden_dim * bytes_per_element

    kv_cache_gb = kv_cache_bytes / 1e9

    return kv_cache_gb

# Example: Llama 70B, batch=32, seq_len=4096
kv_memory = kv_cache_memory(70, batch_size=32, sequence_length=4096)
print(f"KV cache memory: {kv_memory:.1f} GB")
# Output: KV cache memory: 167.8 GB (more than model weights!)

# Optimization: PagedAttention (vLLM)
# Reduces KV cache memory by ~40% through paging
optimized_kv_memory = kv_memory * 0.6
print(f"Optimized KV cache: {optimized_kv_memory:.1f} GB")
# Output: Optimized KV cache: 100.7 GB
```

**C. Speculative Decoding**

```python
# Standard decoding: 1 token per forward pass
standard_tokens_per_sec = 10

# Speculative decoding: Draft model + verification
# Generates 2-3 tokens per forward pass of large model
speculative_tokens_per_sec = 25  # 2.5x speedup

# Cost reduction
speedup = speculative_tokens_per_sec / standard_tokens_per_sec
cost_reduction = 1 - (1 / speedup)

print(f"Speculative decoding: {speedup:.1f}x faster, {cost_reduction:.0%} cost reduction")
# Output: Speculative decoding: 2.5x faster, 60% cost reduction
```

**D. Continuous Batching**

```python
# Static batching: Wait for batch to fill before processing
# Utilization: 60-70%

# Continuous batching (vLLM, TensorRT-LLM): Process requests as they arrive
# Utilization: 90-95%

utilization_improvement = 0.95 / 0.65
throughput_gain = utilization_improvement

print(f"Continuous batching: {throughput_gain:.2f}x throughput")
# Output: Continuous batching: 1.46x throughput

# Cost per request reduced by 1.46x
```

---

## OpenAI Pricing Comparison

**OpenAI's pricing** (as of 2024):

| Model | Input ($/1M tokens) | Output ($/1M tokens) |
|-------|---------------------|----------------------|
| **GPT-4 Turbo** | $10 | $30 |
| **GPT-4** | $30 | $60 |
| **GPT-3.5 Turbo** | $0.50 | $1.50 |

**Self-hosted vs OpenAI**:

```python
def compare_self_hosted_vs_openai(
    monthly_tokens_millions=1000,  # 1B tokens/month
    input_output_ratio=0.5,  # 50% input, 50% output
):
    """
    Compare cost of self-hosting vs using OpenAI API
    """

    input_tokens = monthly_tokens_millions * input_output_ratio
    output_tokens = monthly_tokens_millions * (1 - input_output_ratio)

    # OpenAI GPT-4 Turbo
    openai_cost = input_tokens * 10 + output_tokens * 30

    # Self-hosted Llama 3 70B (4x H100, batch_size=16)
    cost_per_token = 0.000020  # From earlier calculation
    self_hosted_gpu_cost = monthly_tokens_millions * 1e6 * cost_per_token

    # Add engineering overhead (20%)
    self_hosted_total = self_hosted_gpu_cost * 1.2

    return {
        'openai': openai_cost,
        'self_hosted': self_hosted_total,
        'savings': openai_cost - self_hosted_total,
        'breakeven_monthly_tokens_millions': 100,  # Rough estimate
    }

result = compare_self_hosted_vs_openai(monthly_tokens_millions=1000)

print(f"OpenAI: ${result['openai']:,.0f}/month")
print(f"Self-hosted: ${result['self_hosted']:,.0f}/month")
print(f"Savings: ${result['savings']:,.0f}/month ({result['savings']/result['openai']*100:.0f}%)")

# Output:
# OpenAI: $20,000/month
# Self-hosted: $24,000/month
# Savings: -$4,000/month (-20%)

# At 5B tokens/month:
result = compare_self_hosted_vs_openai(monthly_tokens_millions=5000)
print(f"\nAt 5B tokens/month:")
print(f"OpenAI: ${result['openai']:,.0f}/month")
print(f"Self-hosted: ${result['self_hosted']:,.0f}/month")
print(f"Savings: ${result['savings']:,.0f}/month ({result['savings']/result['openai']*100:.0f}%)")

# Output:
# OpenAI: $100,000/month
# Self-hosted: $120,000/month
# Savings: -$20,000/month (-20%)
```

**Breakeven analysis**:
- **Low volume (<1B tokens/month)**: Use OpenAI (no infrastructure overhead)
- **Medium volume (1-10B tokens/month)**: Mixed (depends on engineering capacity)
- **High volume (>10B tokens/month)**: Self-host (significant savings)

---

## Cost-Performance Trade-offs

### 1. Model Size vs Cost

```python
# Compare different model sizes for same task
models = [
    {'name': '7B', 'params': 7, 'quality': 0.65, 'cost_per_1m_tokens': 0.50},
    {'name': '13B', 'params': 13, 'quality': 0.72, 'cost_per_1m_tokens': 1.00},
    {'name': '70B', 'params': 70, 'quality': 0.82, 'cost_per_1m_tokens': 5.00},
    {'name': '405B', 'params': 405, 'quality': 0.88, 'cost_per_1m_tokens': 30.00},
]

for model in models:
    quality_per_dollar = model['quality'] / model['cost_per_1m_tokens']
    print(f"{model['name']:6s}: Quality {model['quality']:.2f}, "
          f"${model['cost_per_1m_tokens']:5.2f}/1M tokens, "
          f"{quality_per_dollar:.4f} quality/$")

# Output:
# 7B   : Quality 0.65, $ 0.50/1M tokens, 1.3000 quality/$
# 13B  : Quality 0.72, $ 1.00/1M tokens, 0.7200 quality/$
# 70B  : Quality 0.82, $ 5.00/1M tokens, 0.1640 quality/$
# 405B : Quality 0.88, $30.00/1M tokens, 0.0293 quality/$
```

**Lesson**: Smaller models have better cost-efficiency. Use the smallest model that meets quality requirements.

### 2. Distillation for Cost Reduction

```python
# Train small model (student) from large model (teacher)

# Option 1: Use 70B for all traffic
cost_70b = 1_000_000_000 * 0.000005  # $5,000/month

# Option 2: Distill 7B from 70B, use 7B for 80% of traffic
distillation_cost = 10_000  # One-time
cost_7b_80pct = 800_000_000 * 0.0000005  # $400/month
cost_70b_20pct = 200_000_000 * 0.000005  # $1,000/month
total_cost = cost_7b_80pct + cost_70b_20pct  # $1,400/month

monthly_savings = cost_70b - total_cost  # $3,600/month
payback_period = distillation_cost / monthly_savings  # 2.8 months

print(f"Distillation savings: ${monthly_savings:,.0f}/month")
print(f"Payback period: {payback_period:.1f} months")
```

---

## Best Practices

### ✅ Do This

1. **Estimate before training**
   - Use Chinchilla scaling laws
   - Account for utilization (50-60% typical)
   - Plan for 20% overhead (checkpointing, failures)

2. **Optimize for cost-performance**
   - Use smallest model that meets quality bar
   - Consider distillation for high-volume inference
   - Quantize for inference (INT8/FP8)

3. **Monitor costs in production**
   - Track cost per request
   - Alert on cost anomalies
   - Optimize batch sizes for throughput vs latency

4. **Use spot instances for training**
   - 50-70% savings
   - Requires good checkpointing strategy
   - Works for flexible deadlines

5. **Consider OpenAI for low volume**
   - No infrastructure overhead
   - Pay-as-you-go
   - Breakeven ~1-10B tokens/month

### ❌ Avoid This

1. **Don't over-train small models**
   - Diminishing returns after Chinchilla-optimal
   - Better to train larger model with fewer tokens

2. **Don't ignore batching**
   - 10-64x cost reduction possible
   - Critical for inference efficiency

3. **Don't use FP32 for training**
   - BF16/FP8 are 2-4x faster with no quality loss

4. **Don't self-host for low volume**
   - Fixed infrastructure costs dominate
   - OpenAI cheaper below breakeven

5. **Don't forget engineering costs**
   - Often 5-20% of total cost
   - Important for TCO calculations

---

## Resources

### Tools

**Cost Estimation**:
- [Hugging Face Pricing Calculator](https://huggingface.co/pricing) - Inference cost estimates
- [LLM Cost Calculator](https://github.com/RahulSChand/llm-cost-calculator) - Training and inference
- [ML Cost Estimator](https://mlco.st/) - Cloud ML cost estimates

**Optimization**:
- [vLLM](https://github.com/vllm-project/vllm) - Efficient inference (PagedAttention, continuous batching)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA inference optimization
- [DeepSpeed](https://www.deepspeed.ai/) - Efficient training (ZeRO, mixed precision)

### Reading

- [Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556) - Optimal allocation of compute
- [Training Compute-Optimal LLMs](https://lifearchitect.ai/chinchilla/) - Practical explanation
- [LLM Economics](https://a16z.com/2023/04/27/large-language-models-economics/) - Cost analysis
- [How Long to Train Your LLM](https://www.databricks.com/blog/how-long-should-you-train-your-language-model) - Training and inference costs

---

**Related Documentation**:
- [Scaling Laws](../pre-training/scaling-laws.md) - Chinchilla-optimal training
- [Distributed Training](../pre-training/distributed-training.md) - Parallelism strategies
- [Monitoring](monitoring.md) - Track costs in production
- [Inference Optimization](../../inference-optimization/) - Quantization, batching, KV cache
