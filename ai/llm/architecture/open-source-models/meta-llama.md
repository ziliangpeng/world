# Meta Llama Series

The Llama series from Meta represents one of the most influential open-source LLM families, setting standards for decoder-only transformer architectures.

## Llama 2 (July 2023)

### Model Variants
- **7B**: 7 billion parameters
- **13B**: 13 billion parameters
- **70B**: 70 billion parameters

### Architecture

**Base Design**: Decoder-only transformer with optimizations

**Key Components**:
- **Normalization**: RMSNorm pre-normalization
- **Activation**: SwiGLU activation function
- **Position Encoding**: RoPE (Rotary Position Embeddings)
- **Attention**:
  - 7B and 13B: Multi-Head Attention (MHA)
  - 70B: Grouped-Query Attention (GQA)

### Training Details
- **Tokens**: 2 trillion tokens
- **Context Window**: 4,096 tokens
- **Vocabulary**: 32K tokens (SentencePiece tokenizer)

### Significance
- First major open-source model to rival proprietary models
- Introduced optimizations that became standard (RMSNorm, SwiGLU, RoPE)
- 70B variant pioneered GQA in production LLMs

## Llama 3 (April 2024)

### Model Variants
- **8B**: 8 billion parameters
- **70B**: 70 billion parameters

### Architecture Updates

**Enhanced Design**: Optimized transformer decoder

**Key Improvements**:
- **Attention**: GQA extended to ALL model sizes (including 8B)
- **Tokenizer**: Upgraded to TikToken with ~128K vocabulary (4x expansion)
- **Context**: 8K tokens

### Training Details
- **Tokens**: 15T+ tokens from publicly available sources
- **Dataset**: Multilingual, code-heavy, high-quality curation

### Innovations
- Extended GQA to smaller models (8B), validating efficiency gains
- Massive vocabulary expansion for better multilingual support
- Superior performance per parameter vs Llama 2

## Llama 3.1 (July 2024)

### Model Variants
- **8B**: 8 billion parameters
- **70B**: 70 billion parameters
- **405B**: 405 billion parameters (flagship)

### Architecture Specifications

**405B Details**:
- **Layers**: 126 transformer layers
- **Hidden Dimension**: 16,384
- **Attention Heads**: 128 heads
- **GQA Configuration**: Grouped query attention across all heads
- **FFN Dimension**: ~53,248 (using SwiGLU)

**Key Features**:
- RoPE with scaling for long context
- RMSNorm pre-normalization
- SwiGLU activation
- Same tokenizer as Llama 3 (~128K vocab)

### Training Details
- **Tokens**: 15T+ tokens
- **Context Window**: **128K tokens** (16x expansion from Llama 3)
- **Multilingual**: Enhanced support for multiple languages

### Significance
- First open model to compete with GPT-4 class models
- Massive context window expansion (8K → 128K)
- Demonstrated scaling laws continue to work at 400B+ parameters

## Llama 3.2 (September 2024)

### Model Variants
- **1B**: 1 billion parameters
- **3B**: 3 billion parameters
- **Vision-capable variants**: Multimodal models

### Architecture
- Same foundation as Llama 3/3.1 (GQA, RoPE, SwiGLU, RMSNorm)
- Optimized for edge deployment and on-device inference
- Maintains architectural consistency with larger siblings

### Focus
- Smaller models for resource-constrained environments
- Vision capabilities for multimodal tasks
- Efficiency without sacrificing too much quality

## Llama 3.3 (Late 2024)

### Updates
- Latest iteration maintaining architectural consistency
- Further optimizations for efficiency and performance
- Continued refinement of training data and processes

## Common Architectural Foundation

### Decoder-Only Transformer Stack

```
Input → Embedding
  ↓
[Repeated 32-126x depending on model size]:
  RMSNorm (pre-normalization)
  → Grouped-Query Attention (with RoPE)
  → Residual Connection
  → RMSNorm
  → SwiGLU FFN
  → Residual Connection
  ↓
Final RMSNorm → Output Projection
```

### Key Design Choices

1. **RMSNorm over LayerNorm**: Simpler, faster, better for distributed training
2. **SwiGLU over GELU**: Better performance, standard in modern LLMs
3. **RoPE over absolute**: Better extrapolation, efficient parameters
4. **GQA over MHA**: Near-MHA quality with significantly better efficiency
5. **Pre-normalization**: Stabilizes training in deep networks

### Evolution Summary

| Version | Sizes | Context | Vocab | Key Innovation |
|---------|-------|---------|-------|----------------|
| Llama 2 | 7B, 13B, 70B | 4K | 32K | GQA in 70B, SwiGLU, RoPE |
| Llama 3 | 8B, 70B | 8K | 128K | GQA for all sizes, TikToken |
| Llama 3.1 | 8B, 70B, 405B | 128K | 128K | Massive context, 405B flagship |
| Llama 3.2 | 1B, 3B, Vision | varies | 128K | Edge optimization, multimodal |
| Llama 3.3 | TBD | TBD | 128K | Continued refinement |

## Impact on the Field

The Llama series has been transformative for open-source AI:

1. **Democratization**: Made state-of-the-art models accessible to researchers and developers
2. **Architectural Standards**: RMSNorm + SwiGLU + RoPE + GQA became the standard stack
3. **Fine-tuning Ecosystem**: Enabled countless specialized models (Code Llama, Vicuna, Alpaca, etc.)
4. **Research Acceleration**: Open weights allowed rapid experimentation with RLHF, quantization, etc.
5. **Commercial Viability**: Proved open models can compete with proprietary alternatives

## Sources

- [Llama 3.2 1B - Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [The Evolution of Llama: From Llama 1 to Llama 3.1](https://towardsdatascience.com/the-evolution-of-llama-from-llama-1-to-llama-3-1-13c4ebe96258/)
- [Llama 3.1 - 405B, 70B & 8B](https://huggingface.co/blog/llama31)
- [Introducing Llama 3.1: Key points of paper](https://medium.com/@vkmauryavk/introducing-llama-3-1-key-points-of-paper-165c29d9c7fd)
- [Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)
- [Llama 3.1 8B - Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B)
- [Llama 2 13B - Hugging Face](https://huggingface.co/meta-llama/Llama-2-13b)
- [Understanding LLaMA-2 Architecture](https://medium.com/towards-generative-ai/understanding-llama-2-architecture-its-ginormous-impact-on-genai-e278cb81bd5c)
