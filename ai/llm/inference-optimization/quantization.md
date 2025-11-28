# Quantization for LLM Inference

Quantization reduces model precision—from 16-bit floating point to 8-bit, 4-bit, or even lower—dramatically reducing memory requirements and often improving inference speed. A 70B model that requires 140GB in fp16 can run in 35GB with 4-bit quantization, enabling deployment on consumer hardware.

---

## Why Quantization?

### The Memory Problem

| Model | fp16 Size | 4-bit Size | Hardware Required |
|-------|-----------|------------|-------------------|
| LLaMA-7B | 14 GB | 3.5 GB | 1× RTX 3080 |
| LLaMA-13B | 26 GB | 6.5 GB | 1× RTX 4090 |
| LLaMA-70B | 140 GB | 35 GB | 1× A100-80GB |
| LLaMA-405B | 810 GB | 203 GB | 3× A100-80GB |

**Key insight**: Memory bandwidth is the bottleneck for LLM inference. Smaller weights = faster memory transfers = faster inference.

### Speed Improvements

| Precision | Memory Bandwidth | Typical Speedup |
|-----------|------------------|-----------------|
| fp16 | 2 bytes/param | 1× (baseline) |
| int8 | 1 byte/param | 1.5-2× |
| int4 | 0.5 byte/param | 2-3× |
| int2 | 0.25 byte/param | 3-4× (experimental) |

---

## Precision Formats

### Floating Point Formats

**fp32 (float32)**:
- 1 sign + 8 exponent + 23 mantissa bits
- Range: ~1.2×10⁻³⁸ to ~3.4×10³⁸
- Rarely used for inference (too slow)

**fp16 (float16)**:
- 1 sign + 5 exponent + 10 mantissa bits
- Range: ~6×10⁻⁵ to ~6×10⁴
- Standard for training and inference

**bf16 (bfloat16)**:
- 1 sign + 8 exponent + 7 mantissa bits
- Same range as fp32, less precision
- Preferred for training stability

**fp8 (E4M3/E5M2)**:
- 1 sign + 4/5 exponent + 3/2 mantissa bits
- Emerging standard for inference (H100, MI300)
- 2× throughput vs fp16 with minimal quality loss

### Integer Formats

**int8**:
- 8-bit signed integer (-128 to 127)
- 4× smaller than fp32, 2× smaller than fp16
- Well-supported on most hardware

**int4**:
- 4-bit signed integer (-8 to 7)
- Requires packing (2 values per byte)
- Sweet spot for LLM inference

**int2/int3**:
- Extreme compression
- Significant quality loss
- Research frontier

---

## Historical Evolution

### Phase 1: Post-Training Quantization (2020-2022)

**LLM.int8()** (bitsandbytes, 2022)

First practical LLM quantization:
- Mixed int8/fp16 based on outlier detection
- Emergent outlier features remain in fp16
- ~0% quality loss for most models

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,  # LLM.int8()
)
```

### Phase 2: Weight-Only Quantization (2022-2023)

**[GPTQ](https://arxiv.org/abs/2210.17323)** (October 2022)

Accurate post-training weight quantization:
- Layer-wise quantization with error correction
- Uses calibration data (~128 samples)
- 4-bit with ~1% quality loss

**Algorithm**:
```
For each layer:
    1. Compute Hessian of layer output
    2. Quantize weights column by column
    3. Update remaining weights to compensate for error
```

**[AWQ](https://arxiv.org/abs/2306.00978)** (June 2023)

Activation-Aware Weight Quantization:
- Identifies "salient" weights via activation magnitudes
- Scales salient weights before quantization
- Often better quality than GPTQ at same bits

```python
# AWQ detects important weights
importance = activation_magnitude × weight_magnitude
# Scale important weights up, quantize, scale back down
```

### Phase 3: Format Standardization (2023-2024)

**GGUF/GGML** (llama.cpp)

Portable quantization format:
- Multiple quantization levels (Q2_K through Q8_0)
- CPU-optimized inference
- Cross-platform compatibility

```
GGUF Quantization Levels:
Q2_K: 2-bit with k-quants (aggressive)
Q3_K_S/M/L: 3-bit variants
Q4_K_S/M: 4-bit variants (popular)
Q5_K_S/M: 5-bit variants
Q6_K: 6-bit (high quality)
Q8_0: 8-bit (near-lossless)
```

**FP8** (H100/MI300 native)

Hardware-native 8-bit floating point:
- E4M3 for weights, E5M2 for activations
- Supported in NVIDIA H100, AMD MI300
- Near-fp16 quality with 2× throughput

### Phase 4: Advanced Techniques (2024)

**[SqueezeLLM](https://arxiv.org/abs/2306.07629)**: Non-uniform quantization
**[QuIP](https://arxiv.org/abs/2307.13304)**: Incoherence processing for extreme quantization
**[AQLM](https://arxiv.org/abs/2401.06118)**: Additive quantization (2-bit competitive)
**[SpQR](https://arxiv.org/abs/2306.03078)**: Sparse-Quantized representation

---

## Quantization Methods

### 1. Post-Training Quantization (PTQ)

Quantize pre-trained weights without retraining:

**Round-to-Nearest (RTN)**:
```python
def quantize_rtn(weights, bits=4):
    """Simple round-to-nearest quantization."""
    scale = weights.abs().max() / (2 ** (bits - 1) - 1)
    quantized = torch.round(weights / scale)
    quantized = quantized.clamp(-(2**(bits-1)), 2**(bits-1) - 1)
    return quantized, scale

def dequantize(quantized, scale):
    return quantized * scale
```

**GPTQ** (Optimal Brain Quantization):
```python
def gptq_quantize(W, H, bits=4):
    """
    GPTQ: Quantize with Hessian-based error compensation.
    W: weight matrix
    H: Hessian (from calibration data)
    """
    n_cols = W.shape[1]
    Q = torch.zeros_like(W)

    for i in range(n_cols):
        # Quantize column i
        w = W[:, i]
        q = quantize_rtn(w, bits)
        Q[:, i] = q

        # Compensate error in remaining columns
        error = (w - dequantize(q)) / H[i, i]
        W[:, i+1:] -= error.unsqueeze(1) @ H[i, i+1:].unsqueeze(0)

    return Q
```

**AWQ** (Activation-Aware):
```python
def awq_quantize(W, activations, bits=4):
    """
    AWQ: Scale salient weights before quantization.
    """
    # Compute importance from activations
    importance = activations.abs().mean(dim=0)

    # Find optimal scale per channel
    scale = compute_optimal_scale(W, importance, bits)

    # Scale weights, quantize, scale back
    W_scaled = W * scale
    Q = quantize_rtn(W_scaled, bits)

    return Q, scale
```

### 2. Weight-Only vs Weight-Activation

**Weight-only** (most common for LLMs):
- Only weights quantized
- Activations remain in fp16
- Easier, often sufficient

**Weight-activation** (W8A8, W4A8):
- Both weights and activations quantized
- Higher compression, more speedup
- Requires activation calibration

```python
# W4A16 (weight-only)
output = dequantize(Q_weights) @ fp16_activations

# W4A8 (both)
output = int8_activations @ int4_weights  # Special kernel
```

### 3. Quantization-Aware Training (QAT)

Train model knowing it will be quantized:

```python
class QuantizedLinear(nn.Module):
    def forward(self, x):
        # Forward pass uses quantized weights
        q_weights = fake_quantize(self.weight)
        return F.linear(x, q_weights, self.bias)

def fake_quantize(x, bits=4):
    """Quantize and dequantize for gradient flow."""
    scale = x.abs().max() / (2 ** (bits - 1) - 1)
    q = torch.round(x / scale)
    q = q.clamp(-(2**(bits-1)), 2**(bits-1) - 1)
    # Straight-through estimator: gradient flows as if no quantization
    return (q * scale).detach() + x - x.detach()
```

**Trade-offs**:
| Method | Quality | Cost | Use Case |
|--------|---------|------|----------|
| PTQ (RTN) | Low | Free | Quick experiments |
| PTQ (GPTQ/AWQ) | High | Minutes | Production |
| QAT | Highest | Days | Maximum quality |

---

## Practical Implementation

### Using Transformers + bitsandbytes

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# 8-bit quantization
model_8bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
)

# 4-bit quantization (NF4)
model_4bit = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4-bit
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)
```

### Using AutoGPTQ

```python
from auto_gptq import AutoGPTQForCausalLM

# Load pre-quantized model
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7B-GPTQ",
    device="cuda:0",
)

# Or quantize yourself
from transformers import AutoTokenizer
from auto_gptq import BaseQuantizeConfig

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoGPTQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Calibration data
calibration_data = [tokenizer(text, return_tensors="pt") for text in examples]

# Quantize
quantize_config = BaseQuantizeConfig(bits=4, group_size=128)
model.quantize(calibration_data)
model.save_quantized("Llama-2-7b-4bit-gptq")
```

### Using llama.cpp (GGUF)

```bash
# Convert model to GGUF
python convert.py models/llama-2-7b --outtype f16 --outfile llama-2-7b.gguf

# Quantize to Q4_K_M
./quantize llama-2-7b.gguf llama-2-7b-Q4_K_M.gguf Q4_K_M

# Run inference
./main -m llama-2-7b-Q4_K_M.gguf -p "Hello, world"
```

### Using vLLM with Quantization

```python
from vllm import LLM

# FP8 quantization (H100)
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="fp8",
)

# AWQ quantized model
llm = LLM(
    model="TheBloke/Llama-2-7B-AWQ",
    quantization="awq",
)
```

---

## Quality vs Compression Trade-offs

### Perplexity Impact

| Model | fp16 PPL | 8-bit PPL | 4-bit PPL | 3-bit PPL |
|-------|----------|-----------|-----------|-----------|
| LLaMA-7B | 5.68 | 5.70 | 5.79 | 6.15 |
| LLaMA-13B | 5.09 | 5.11 | 5.18 | 5.45 |
| LLaMA-70B | 3.32 | 3.33 | 3.38 | 3.52 |

**Pattern**: Larger models tolerate quantization better.

### Task Performance

| Quantization | MMLU | GSM8K | HumanEval | Notes |
|--------------|------|-------|-----------|-------|
| fp16 | 64.0% | 56.8% | 29.9% | Baseline |
| 8-bit | 63.8% | 56.5% | 29.5% | Negligible loss |
| 4-bit GPTQ | 63.2% | 55.2% | 28.4% | ~1-2% loss |
| 4-bit AWQ | 63.5% | 55.8% | 28.9% | Slightly better |
| 3-bit | 60.1% | 48.5% | 24.1% | Noticeable loss |

### When to Use What

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| Production, quality-critical | 8-bit or fp16 | Minimal quality loss |
| Consumer GPU deployment | 4-bit (Q4_K_M) | Good balance |
| Mobile/edge | 4-bit or 3-bit | Size constraints |
| Experimentation | 4-bit GGUF | Fast iteration |
| Maximum compression | AQLM 2-bit | Research/specialized |

---

## Hardware Considerations

### GPU Support

| Format | NVIDIA | AMD | Intel | Apple |
|--------|--------|-----|-------|-------|
| fp16 | All | All | Arc | M-series |
| int8 | All | MI-series | Arc | M-series |
| int4 | SM80+ | MI-series | Limited | M-series |
| fp8 | H100+ | MI300+ | No | No |

### Memory Bandwidth vs Compute

LLM inference is memory-bound:
```
Computation: O(batch × seq × d²) FLOPs
Memory: O(params) bytes loaded

For small batches: Memory dominates
For large batches: Compute can dominate
```

**Quantization helps most** when memory-bound (small batches, long contexts).

### Kernel Availability

| Format | cuBLAS | CUTLASS | Triton | llama.cpp |
|--------|--------|---------|--------|-----------|
| fp16 | ✅ | ✅ | ✅ | ✅ |
| int8 | ✅ | ✅ | ✅ | ✅ |
| int4 | ❌ | ✅ | ✅ | ✅ |
| fp8 | ✅ (H100) | ✅ | ✅ | ❌ |

---

## Best Practices

### Choosing Quantization

1. **Start with 4-bit AWQ or GPTQ** for most use cases
2. **Use 8-bit** if quality-critical
3. **Use GGUF Q4_K_M** for CPU or consumer GPUs
4. **Use FP8** on H100/MI300 for best throughput

### Calibration

1. **Use representative data**: Match your deployment distribution
2. **Enough samples**: 128-512 usually sufficient
3. **Sequence length**: Match your expected usage
4. **Validate**: Always check quality after quantization

### Deployment

1. **Pre-quantize**: Don't quantize at load time in production
2. **Match kernels**: Use optimized kernels for your format
3. **Batch appropriately**: Larger batches may not benefit
4. **Monitor quality**: Track metrics in production

---

## Future Directions

### Near-term (2025)

1. **FP8 adoption**: As H100/MI300 deploy widely
2. **Better 2-bit**: AQLM and successors
3. **Dynamic quantization**: Adjust precision per layer/token
4. **Hardware co-design**: Quantization-aware accelerators

### Research Frontiers

1. **1-bit models**: BitNet and extreme compression
2. **Mixed-precision**: Different bits for different components
3. **Learned quantization**: End-to-end optimization
4. **Sparsity + quantization**: Combined compression

---

## Sources

### Foundational Papers
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [LLM.int8(): 8-bit Matrix Multiplication](https://arxiv.org/abs/2208.07339)

### Advanced Methods
- [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629)
- [QuIP: 2-Bit Quantization via Incoherence](https://arxiv.org/abs/2307.13304)
- [AQLM: Additive Quantization](https://arxiv.org/abs/2401.06118)
- [SpQR: Sparse-Quantized Representation](https://arxiv.org/abs/2306.03078)

### Tools and Implementations
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [vLLM Quantization](https://docs.vllm.ai/en/latest/quantization/fp8.html)
