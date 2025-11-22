# Qwen Series (Alibaba Cloud)

Qwen (通义千问) from Alibaba Cloud represents a leading series of multilingual LLMs with both dense and MoE variants, known for strong performance across languages and modalities.

## Qwen 2.5 (December 2024)

### Model Variants
- **0.5B**: 0.5 billion parameters
- **1.5B**: 1.5 billion parameters
- **3B**: 3 billion parameters
- **7B**: 7 billion parameters
- **14B**: 14 billion parameters
- **32B**: 32 billion parameters
- **72B**: 72 billion parameters

### Architecture

**Type**: Dense decoder-only transformer

**Key Components**:
- **Attention**: Grouped Query Attention (GQA)
- **Activation**: SwiGLU
- **Position Encoding**: RoPE (Rotary Position Embeddings)
- **Normalization**: RMSNorm with pre-normalization
- **Optimization**: FlashAttention 2 for efficient attention computation

### Training Details
- **Tokens**: 18 trillion tokens
- **Context Window**: 128K tokens (some versions handle up to 1M tokens)
- **Languages**: Advanced multilingual capabilities (29+ languages)
- **Modalities**: Text, vision (Qwen2.5-VL variants)

### Specifications (7B Example)
- **Layers**: 28
- **Hidden Dimension**: 3,584
- **Attention Heads**: 28
- **KV Heads**: 4 (GQA with 7 queries per key/value)
- **FFN Dimension**: 18,944
- **Vocabulary**: ~152K tokens

### Special Capabilities
- Code generation and understanding
- Mathematical reasoning
- Long-context understanding (128K-1M tokens)
- Instruction following
- Multimodal understanding (vision variants)

## Qwen 3 (2025)

### Model Variants
Dense and MoE variants across multiple sizes

### Architecture Updates

**Dense Architecture Improvements**:
- **Removed**: QKV-bias (present in Qwen2)
- **Added**: QK-Norm for stable training
- **Enhanced**: Three-stage training process
- **Vocabulary**: Similar to Qwen 2.5 (~64K-152K)

**QK-Norm Innovation**:
- Per-head normalization of queries and keys
- Stabilizes training at scale
- Reduces training instabilities
- Allows for more aggressive learning rates

### Training Details
- **Tokens**: 36 trillion tokens (2x Qwen 2.5)
- **Languages**: 119 languages
- **Context**: Started at 4,096 → expanded to 32,768 tokens

### Three-Stage Training Process

1. **Stage 1: General Knowledge**
   - Broad coverage across domains
   - Multilingual data
   - Code and text

2. **Stage 2: STEM & Coding Specialization**
   - Enhanced mathematical reasoning
   - Advanced code understanding
   - Scientific domains

3. **Stage 3: Long Context Training**
   - Context window expansion: 4K → 32K
   - Long-range dependency learning
   - Document-level understanding

## Qwen 3 MoE (Mixture of Experts)

### Architecture Type
**Fine-Grained Sparse MoE**

### Specifications
- **Total Experts**: 128 experts
- **Activated per Token**: 8 experts
- **Expert Granularity**: Fine-grained segmentation (inspired by DeepSeekMoE)

### MoE Design Philosophy

**Fine-Grained Segmentation**:
- 128 smaller experts instead of 8-16 large experts
- Better specialization potential
- More flexible routing
- Improved load balancing

**Activation Pattern**:
- 8 of 128 experts = ~6.25% activation rate
- Lower than Mixtral (25%) but more experts
- Allows finer-grained specialization

### Benefits of 128-Expert Design

1. **Specialization**: Each expert can focus on narrower domain
2. **Flexibility**: More routing combinations (C(128,8) possibilities)
3. **Load Balancing**: Easier to distribute load across many experts
4. **Capacity**: High total capacity with controlled active compute

## Common Architectural Foundation

### Transformer Stack

```
Input → Token Embedding + RoPE
  ↓
[Repeated N times]:
  RMSNorm
  → Grouped-Query Attention
    - Q: Multiple query heads
    - K, V: Shared across query groups
    - RoPE applied
    - QK-Norm (Qwen 3)
  → Residual Connection
  → RMSNorm
  → SwiGLU FFN (or MoE layer)
  → Residual Connection
  ↓
Final RMSNorm → Output Projection
```

### GQA Configuration

Qwen uses aggressive GQA configurations for efficiency:
- **Example (7B)**: 28 query heads, 4 KV heads
- **Ratio**: 7 queries share each K/V
- **Memory Savings**: ~7x reduction in KV cache size vs MHA
- **Quality**: Minimal degradation vs full MHA

## Architectural Innovations

### 1. QK-Norm (Qwen 3)

Traditional attention:
```python
attention = softmax(Q @ K.T / sqrt(d))
```

With QK-Norm:
```python
Q_norm = normalize(Q, dim=-1)  # Per-head normalization
K_norm = normalize(K, dim=-1)
attention = softmax(Q_norm @ K_norm.T / sqrt(d))
```

**Benefits**:
- Prevents attention score explosion
- Stabilizes training with large models
- Enables higher learning rates
- Reduces gradient issues

### 2. Aggressive GQA

Qwen pushes GQA further than most models:
- Higher query-to-KV ratios
- Significant memory savings
- Maintains quality through careful tuning

### 3. Massive Vocabulary

- Up to 152K tokens (vs 32K-64K in many models)
- Better multilingual coverage
- More efficient encoding for many languages
- Fewer tokens per sequence on average

### 4. Long Context Scaling

Multiple strategies for context scaling:
- RoPE interpolation
- Progressive training (4K → 32K → 128K)
- FlashAttention 2 for efficiency
- Some variants support 1M tokens

## Evolution Summary

| Version | Max Tokens Trained | Context Window | Key Innovation |
|---------|-------------------|----------------|----------------|
| Qwen 1.x | ~3T | 8K-32K | Multilingual foundation |
| Qwen 2.5 | 18T | 128K-1M | FlashAttention, expanded context |
| Qwen 3 | 36T | 32K+ | QK-Norm, 3-stage training, MoE with 128 experts |

## Performance Characteristics

### Strengths
1. **Multilingual**: Industry-leading support for 119 languages
2. **Code**: Strong performance on coding benchmarks
3. **Math**: Excellent mathematical reasoning
4. **Long Context**: Effective use of 128K+ token windows
5. **Efficiency**: GQA and FlashAttention enable fast inference

### Benchmarks
- Competitive with or exceeding Llama 3, Mistral on many tasks
- Strong showing on multilingual evaluations
- Excellent code generation (HumanEval, MBPP)
- High scores on GSM8K, MATH for mathematical reasoning

## Deployment Considerations

### Dense Models
- Straightforward deployment
- Excellent for general-purpose use
- Various sizes for different compute budgets

### MoE Models
- Higher throughput than dense models of similar quality
- Requires MoE-aware inference frameworks
- More memory for storing all experts
- Lower active compute per token

### Context Length
- 128K context requires significant memory
- FlashAttention essential for practical use
- Consider model size vs available VRAM

## Impact on the Field

### Technical Contributions
1. **QK-Norm**: Novel normalization technique for attention stability
2. **Fine-grained MoE**: 128-expert architecture
3. **Multilingual Excellence**: Demonstrated how to scale across languages
4. **Long Context**: Practical 128K-1M token models

### Open Source Impact
- Apache 2.0 / custom open licenses
- Enabled multilingual applications
- Strong alternative to Western-focused models
- Popular for commercial deployments in Asia

## Sources

- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923)
- [Qwen 2.5 7B - Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B)
- [Qwen 3 Benchmarks and Specifications](https://dev.to/best_codes/qwen-3-benchmarks-comparisons-model-specifications-and-more-4hoa/comments)
