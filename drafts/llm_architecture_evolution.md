# LLM Architecture Evolution: GPT-2 to Modern SOTA

## GPT-2 (2019) - The Foundation
**Core Architecture:**
- Decoder-only Transformer (Transformer-D)
- Multi-head self-attention + Feed-forward networks (FFN)
- Layer normalization (pre-norm)
- Learned positional embeddings (absolute)
- Largest model: 1.5B parameters, 48 layers

**Key Characteristics:**
- Autoregressive language modeling objective
- BPE tokenization
- Context length: 1024 tokens
- Training: 40GB of web text

## Major Architectural Innovations (2019-2025)

### 1. Scaling & Model Size
**GPT-3 (2020):** 175B parameters
- Same architecture as GPT-2, just scaled up
- Demonstrated few-shot learning capabilities
- Context: 2048 tokens

**PaLM (2022):** 540B parameters
- Introduced SwiGLU activation (instead of GeLU/ReLU)
- Parallel attention + FFN blocks (Parallel Transformer)
- RMSNorm instead of LayerNorm

**GPT-4 (2023):** Rumored 1.7T parameters (MoE)
**Llama 3.1 (2024):** 405B dense model
**Qwen 3 (2025):** 235B parameters

### 2. Positional Encoding Evolution
**ALiBi (2022):** Attention with Linear Biases
- No learned positional embeddings
- Better length extrapolation

**RoPE (2021):** Rotary Position Embedding
- Used in Llama, Qwen, Mistral
- Better for long context
- Enables position interpolation

**YaRN/ABF (2023+):** RoPE extensions
- Extends context to 128K+ tokens
- Temperature scaling for different frequency bands

### 3. Attention Mechanism Improvements
**Multi-Query Attention (MQA):**
- Single key-value head, multiple query heads
- Faster inference, less memory

**Grouped-Query Attention (GQA):**
- Used in Llama 2/3, Qwen 2/3
- Balance between MHA and MQA
- Groups of queries share KV heads

**Sliding Window Attention:**
- Mistral (2023): 4096 token sliding window
- Enables longer effective context

**Flash Attention (2022-2024):**
- Not architecture change, but critical optimization
- 2-4x faster training/inference
- Enables longer contexts

### 4. Mixture of Experts (MoE)
**Switch Transformer (2021):** First large-scale MoE
**Mixtral 8x7B (2023):**
- 8 experts, top-2 routing
- 47B total, 13B active per token
- Better quality/cost ratio

**DeepSeek-V2/V3 (2024):**
- Fine-grained MoE (256 experts)
- Multi-head latent attention (MLA)
- Drastically reduced KV cache

**Qwen 2.5 MoE:** 14B/57B variants
- 64 experts, top-8 routing

### 5. Normalization & Activation
**RMSNorm (2019):** Simpler, faster than LayerNorm
- Used in Llama, Qwen, Mistral

**Pre-LayerNorm → Pre-RMSNorm:**
- More stable training

**Activation Functions:**
- GeLU → SwiGLU/GeGLU (better performance)
- GLU variants now standard in FFN

### 6. Context Length Extensions
**2019:** 1K-2K tokens (GPT-2)
**2023:** 32K tokens (GPT-4, Claude 2)
**2024:** 128K-200K tokens (GPT-4 Turbo, Claude 3)
**2025:** 1M+ tokens (Gemini 1.5, Claude 3.5)

**Techniques:**
- RoPE interpolation
- Attention sinks
- Sparse attention patterns
- Long context fine-tuning

### 7. Training Improvements
**FP16 → BF16 → FP8:**
- Lower precision for efficiency
- FP8 training becoming standard

**Parallelism Strategies:**
- Data parallelism → Tensor parallelism → Pipeline parallelism → 3D parallelism
- ZeRO optimizer states sharding

**Curriculum Learning:**
- Start with shorter contexts, gradually increase
- Used in long-context models

### 8. Multi-Modal Extensions (2023-2025)
**Vision Encoders Integration:**
- GPT-4V, Claude 3, Gemini
- Cross-attention or adapter layers
- Vision tokens merged into text stream

**Vision-Language Architectures:**
- CLIP-style encoders
- Q-Former adapters
- Direct pixel-to-text models

## Modern SOTA Architecture (2025)

**Typical Components:**
- Decoder-only Transformer
- RoPE or ALiBi positional encoding
- GQA or MLA attention
- SwiGLU activation in FFN
- RMSNorm
- BF16/FP8 precision
- 32K-128K+ context length
- Optional: MoE for efficiency

**Example: Qwen 3 (235B)**
- 96 layers
- 128 attention heads
- GQA (16 KV heads)
- Hidden size: 18,432
- FFN: 49,152
- RoPE with 128K context
- SwiGLU activation
- RMSNorm

**Example: DeepSeek-V3 (685B total, 37B active)**
- Multi-head latent attention (MLA)
- 256 experts, top-1 routing
- Auxiliary loss for load balancing
- Drastically reduced KV cache size

## Key Trends

1. **Scale + Efficiency:** Larger models with smarter routing (MoE)
2. **Long Context:** From 1K to 1M+ tokens
3. **Attention Optimization:** MQA/GQA/MLA for faster inference
4. **Training Stability:** Better norms, activations, precision
5. **Multi-Modal:** Text+Vision+Audio in single model
6. **Inference Optimization:** KV cache reduction, speculative decoding

## What Changed vs GPT-2

| Aspect | GPT-2 (2019) | Modern SOTA (2025) |
|--------|--------------|-------------------|
| Size | 1.5B | 200B-600B (dense/MoE) |
| Context | 1K tokens | 128K-1M+ tokens |
| Position | Learned absolute | RoPE/ALiBi |
| Attention | MHA | GQA/MLA |
| Activation | GeLU | SwiGLU |
| Norm | LayerNorm | RMSNorm |
| Precision | FP32/FP16 | BF16/FP8 |
| Architecture | Dense | Dense or MoE |

## References
- GPT-2 (Radford et al., 2019)
- GPT-3 (Brown et al., 2020)
- PaLM (Chowdhery et al., 2022)
- Llama (Touvron et al., 2023)
- Mistral (Jiang et al., 2023)
- DeepSeek-V2/V3 (2024)
- Qwen Technical Reports (2024-2025)
