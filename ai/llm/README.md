# Large Language Models (LLM)

Comprehensive documentation of Large Language Model architectures, training methodologies, and applications.

## Quick Navigation

### 📚 Model Documentation
- **[Open Source Models Index](OPEN_SOURCE_MODELS.md)** - Catalog of 40+ open source LLM families
  - [Open Source Models Directory](open-source-models/) - Detailed docs for Llama, Mistral, Qwen, DeepSeek, Gemma, Phi, and more
  - [Proprietary Models](proprietary-models/) - OpenAI GPT, Anthropic Claude, Google Gemini, and others

### 🏗️ Architecture
- **[Architecture Documentation](architecture/README.md)** - Comprehensive architectural research
  - [Architectural Patterns](architecture/) - Attention mechanisms, position embeddings, MoE, tokenizers, etc.
- **[Important LLM Papers](important-llm-papers.md)** - Foundational research and breakthrough techniques

### 🎓 Training & Operations
- **[Training Recipe](training-recipe/overview.md)** - Complete LLM training pipeline
  - [Pre-training](training-recipe/pre-training/) - Data preparation, optimizers, scaling laws, distributed training
  - [Post-training](training-recipe/post-training/) - SFT, RLHF, DPO, test-time compute, distillation
  - [Benchmarks](training-recipe/benchmarks.md) - Evaluation datasets and metrics
- **[LLMOps](llmops/)** - Production patterns, monitoring, cost analysis, checkpointing, evaluation

### ⚡ Inference & Applications
- **[Inference Optimization](inference-optimization/)** - Quantization, KV cache, speculative decoding, batching
- **[Applications](applications/)** - Code generation, agentic systems, education, conversational AI, and more
- **[Tools](tools/)** - Infrastructure and frameworks (Prime Intellect RL stack, etc.)

### 🔬 Research Notes
- [LMCache Deep Dive](lmcache_deep_dive.md) - Detailed analysis of LMCache architecture
- [LMCache Random Notes](lmcache_random_notes_1.md) - Additional LMCache observations

## Overview

This repository contains:
- **Architecture specifications**: Parameters, layers, attention mechanisms, architectural innovations
- **Training details**: Dataset sizes, compute requirements, training recipes
- **Key innovations**: Breakthrough techniques and optimizations
- **Architectural patterns**: Common patterns across models
- **Production guidance**: LLMOps, inference optimization, deployment

## Research Methodology

Information gathered from:
- Official model cards and technical papers
- Model creator blog posts and announcements
- Hugging Face documentation
- Academic publications
- Web search for latest developments (2024-2025)

All sources are cited in individual documentation files.

---

*Last Updated: 2025-11-27*
