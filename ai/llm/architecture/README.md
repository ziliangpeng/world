# LLM Architecture Research

Comprehensive documentation of Large Language Model architectures, focusing on major open source models and significant proprietary models.

## Overview

This repository contains detailed research on:
- Architecture specifications (parameters, layers, attention mechanisms)
- Training details (dataset size, compute requirements)
- Key innovations and optimizations
- Architectural patterns and trends

## Directory Structure

### [Open Source Models](../OPEN_SOURCE_MODELS.md)

Comprehensive list of 40+ open source LLM families including:
- Major foundation models (Llama, Mistral, Qwen, DeepSeek, Gemma, Phi)
- Chinese models (Baichuan, InternLM, ChatGLM, Hunyuan)
- Code specialists (StarCoder, CodeGen)
- Enterprise/RAG models (Cohere, DBRX, Palmyra)
- Alternative architectures (RWKV, Jamba, Mamba-based)
- Multimodal models (LLaVA, CogVLM, Fuyu)
- Research models (OLMo, Pythia)
- Efficient small models (TinyLlama, MiniCPM)

See [OPEN_SOURCE_MODELS.md](../OPEN_SOURCE_MODELS.md) for complete list with links to detailed documentation.

### Proprietary Models

Private models with known architectural details:

#### [OpenAI GPT Series](../proprietary-models/openai-gpt.md)
- GPT-4 (rumored ~1.76T total, MoE)
- GPT-4 Turbo (128K context)
- GPT-4o (multimodal, 128K context)
- GPT-4o mini
- GPT-5 (reported 400K context)

#### [Anthropic Claude](../proprietary-models/anthropic-claude.md)
- Claude 3 Family (Haiku, Sonnet, Opus)
- Claude 3.5 Sonnet
- Claude 3.5 Haiku
- Claude Sonnet 4.5

#### [Google Gemini](../proprietary-models/google-gemini.md)
- Gemini 1.5 Pro (MoE, 1M-2M context)
- Gemini 2.0 Flash (MoE, 1M context)

#### [Other Proprietary](../proprietary-models/other-proprietary.md)
- Google PaLM 2 (540B, deprecated)
- xAI Grok-1 (314B MoE, open-sourced)
- xAI Grok 1.5
- xAI Grok 3 (314B MoE)

### Architectural Patterns

Common patterns and innovations across models:

- [Attention Mechanisms](architectural-patterns/attention-mechanisms.md) - MHA, MQA, GQA, MLA, FlashAttention
- [Position Embeddings](architectural-patterns/position-embeddings.md) - RoPE, ALiBi, Sinusoidal
- [Activation & Normalization](architectural-patterns/activation-normalization.md) - SwiGLU, GELU, RMSNorm, LayerNorm
- [Mixture of Experts](architectural-patterns/mixture-of-experts.md) - MoE architectures and implementations
- [Tokenizers](architectural-patterns/tokenizers.md) - BPE, SentencePiece, tiktoken
- [Context Windows](architectural-patterns/context-windows.md) - Evolution and scaling techniques

## Key Trends (2024-2025)

### Architectural Evolution
- **Decoder-only dominance**: Nearly all modern LLMs use decoder-only transformers
- **Attention**: Shift from MHA → GQA/MLA for efficiency
- **Position**: RoPE and ALiBi replacing absolute encodings
- **Activation**: SwiGLU becoming standard over GELU
- **Normalization**: RMSNorm replacing LayerNorm in many models

### Training Scale
- Trillion-token training is now standard (15T-36T tokens)
- Compute-optimal scaling: data size ≈ model size
- Multilingual and multimodal becoming default
- Synthetic data for specialized domains

### Context Windows
- Rapid expansion: 2K → 128K → 1M+ tokens
- FlashAttention enabling efficient long-context processing
- Some models reaching 10M+ token windows (Llama 4)

### Mixture of Experts
- MoE enabling massive scale with sparse activation
- Models like DeepSeek-V3: 671B total, 37B activated
- Fine-grained expert segmentation for better performance

## Standard Modern LLM Stack (2024)

1. **Architecture**: Decoder-only transformer
2. **Attention**: GQA or MLA
3. **Position**: RoPE or ALiBi
4. **Activation**: SwiGLU
5. **Normalization**: RMSNorm with pre-normalization
6. **Efficiency**: FlashAttention-2/3
7. **Tokenizer**: BPE-based (expanding vocabularies: 100K-256K)
8. **Scaling**: MoE for largest models

## Research Methodology

Information gathered from:
- Official model cards and technical papers
- Model creator blog posts and announcements
- Hugging Face documentation
- Academic publications
- Web search for latest developments (2024-2025)

All sources are cited in individual documentation files.

---

Last Updated: 2025-11-21
