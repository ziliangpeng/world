# xAI Grok-1: The 314B Parameter MoE Model Released via BitTorrent

## Overview

Grok-1 is xAI's 314 billion parameter Mixture of Experts (MoE) language model, [released open-source on March 17, 2024](https://x.ai/news/grok-os), under Apache 2.0. Notable for its unconventional BitTorrent distribution—a first for a model of this scale.

**Performance**: GPT-3.5 class—outperforms GPT-3.5 on most benchmarks (73% vs ~70% MMLU, 63% vs 57% HumanEval) but trails GPT-4 significantly. Math reasoning is the weakest area.

### Key Highlights

- **314B total parameters** with 8 experts, 2 active per token (~86B active, 25% activation)
- **Released via BitTorrent** as 318GB torrent file
- **Apache 2.0 license** - unrestricted commercial use
- **JAX + Rust + Kubernetes** training stack (unusual choice vs PyTorch)
- **131K token vocabulary** (4x larger than Llama/Mixtral's 32K)
- **October 2023 pretraining cutoff** - base model checkpoint only

> **See also**: [xAI Overview](xai-overview.md) for company background and infrastructure details.

---

## Model Specifications

| Specification | Value |
|--------------|-------|
| **Total Parameters** | 314 billion |
| **Active Parameters** | ~86B (25% activation) |
| **Architecture** | Sparse MoE, Decoder-only Transformer |
| **Experts** | 8 total, 2 active per token |
| **Layers** | 64 |
| **Attention** | GQA: 48 query heads, 8 KV heads (6:1 ratio) |
| **Embedding Size** | 6,144 |
| **FFN Hidden** | ~24,576 (estimated 4x embed) |
| **Context Window** | 8,192 tokens |
| **Vocabulary** | 131,072 tokens (SentencePiece) |
| **Position Encoding** | RoPE |
| **Activation** | SwiGLU (likely) |
| **Normalization** | RMSNorm (likely) |
| **Model Size** | ~318GB (torrent) / ~300GB (weights) |

### Tokenizer

**131,072 tokens** - unusually large vocabulary (2^17):

- Llama 2 / Mixtral: 32,000 tokens
- Grok-1: 131,072 tokens (4x larger)

Special tokens include `<|separator|>`, `<|mask|>`, and 20 control tokens (`<|control0|>` through `<|control19|>`).

---

## Architecture

### MoE Configuration

```
Total Parameters:     314B
Number of Experts:    8
Active per Token:     2
Activation Rate:      25% (2/8)
Active Parameters:    ~86B per forward pass
```

Top-2 routing with learned gating network. Load balancing via noise injection and auxiliary loss.

### Key Architectural Choices

| Component | Grok-1 Choice | Benefit |
|-----------|---------------|---------|
| **Attention** | GQA (48Q/8KV) | 83% KV cache reduction |
| **Position** | RoPE | Relative encoding, context extension |
| **Activation** | SwiGLU | Better gradient flow |
| **Norm** | RMSNorm | 50% less compute than LayerNorm |
| **MoE** | 8x2 sparse | 314B capacity, 86B compute |

---

## Training

### Infrastructure

Trained on [~16,000 Nvidia GPUs via Oracle Cloud Infrastructure](https://www.datacenterdynamics.com/en/news/xai-to-use-oracle-cloud-infrastructure-to-train-and-run-inferencing-for-grok/)—xAI was one of Oracle's largest cloud customers. The split came later when [Oracle couldn't match xAI's timeline demands](https://aibusiness.com/verticals/musk-xai-ditches-oracle-cloud-to-build-massive-gpu-cluster-for-grok-3) for Grok-2/3, leading to the [Colossus supercomputer](https://www.nextplatform.com/2024/07/30/so-who-is-building-that-100000-gpu-cluster-for-xai/) (100K+ GPUs in Memphis).

### Training Stack

xAI built a custom distributed training framework:

**JAX** (Primary ML Framework):

- Automatic differentiation, JIT compilation
- Hardware agnostic (TPU/GPU)

**Rust** (Infrastructure):
> "Rust provides confidence that any code modification or refactor is likely to produce working programs that will run for months with minimal supervision."

**Kubernetes** (Orchestration):

- Container orchestration, fault tolerance, scalability

### Training Data

- **Internet data**: Web crawl, code repositories, technical docs
- **AI Tutor data**: Human-curated and reviewed content
- **Cutoff**: October 2023 (Q3 2023)
- **Token count**: Not disclosed

### What Was Released vs. Not Released

| Released | Not Released |
|----------|--------------|
| ✅ Model weights (314B) | ❌ Training data |
| ✅ Inference code (JAX) | ❌ Training code |
| ✅ Architecture details | ❌ Fine-tuning data |
| ✅ Tokenizer | ❌ RLHF details |

This is an "open-weight" release, not fully "open-source" (training details remain proprietary).

---

## Performance

| Benchmark | Grok-1 | GPT-3.5 | GPT-4 | Llama 2 70B |
|-----------|--------|---------|-------|-------------|
| **MMLU** (0-shot) | 73% | ~70% | 86.4% | ~69% |
| **HumanEval** (0-shot) | 63.2% | 57.1% | ~85% | ~55% |
| **GSM8K** (8-shot) | > GPT-3.5 | baseline | 92% | ~50% |
| **MATH** (4-shot) | Weak | baseline | ~52% | - |
| **Hungarian Math** | 59% (C) | - | 68% (B) | - |

**Key insight**: Grok-1 [outperforms GPT-3.5 across benchmarks](https://vectorinstitute.ai/benchmarking-xais-grok-1/), but trails GPT-4 significantly. Math reasoning is the weakest area.

> "Grok-1 is only surpassed by models that were trained with a significantly larger amount of training data and compute resources like GPT-4." — [xAI](https://x.ai/news/grok-os)

**Context**: This is a base model. Fine-tuned Grok versions (used in X Premium+) perform better.

---

## BitTorrent Release

### The Story

**March 11, 2024**: Elon Musk announced Grok would be open-sourced

**March 17, 2024**: Released via BitTorrent under Apache 2.0

The announcement included a jab at OpenAI:

- **Grok**: Posted "░W░E░I░G░H░T░S░I░N░B░I░O░"
- **ChatGPT**: Replied "stole my whole joke"
- **Musk**: "Tell us more about the 'open' part of OpenAI…"

### Distribution

**Magnet link**:
```
magnet:?xt=urn:btih:5f96d43576e3d386c9ba65b883210a393b68210e
```

- **Total size**: 318.24 GB
- **Files**: 773 files in JAX checkpoint format
- **Trackers**: [Academic Torrents](https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e), public trackers

### Why BitTorrent?

1. **Cost**: Distributing 300GB to 10K users via CDN ≈ $240K. BitTorrent: $0 after seeding.
2. **Scalability**: P2P scales naturally with demand
3. **Censorship resistance**: No single point of failure
4. **Precedent**: Meta also distributed Llama via torrent

### Alternative Access

- **GitHub**: [xai-org/grok-1](https://github.com/xai-org/grok-1)
- **HuggingFace**: [xai-org/grok-1](https://huggingface.co/xai-org/grok-1) (official), [hpcai-tech/grok-1](https://huggingface.co/hpcai-tech/grok-1) (PyTorch conversion)
- **Quantized**: GGUF versions available for llama.cpp

**Practical barrier**: Even INT4 quantized requires 2x H100 GPUs (~$50K+), making local use impractical for most.

---

## Sources

### Official
- [Open Release of Grok-1 | xAI](https://x.ai/news/grok-os)
- [GitHub - xai-org/grok-1](https://github.com/xai-org/grok-1)
- [xai-org/grok-1 · Hugging Face](https://huggingface.co/xai-org/grok-1)

### Technical Analysis
- [Grok-1 code and model weights release - Simon Willison](https://simonwillison.net/2024/Mar/17/grok-1/)
- [Benchmarking xAI's Grok-1 - Vector Institute](https://vectorinstitute.ai/benchmarking-xais-grok-1/)
- [hpcai-tech/grok-1 · Hugging Face](https://huggingface.co/hpcai-tech/grok-1) (PyTorch conversion)

### Distribution
- [grok-1 - Academic Torrents](https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e)
