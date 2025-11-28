# Mistral 7B

## Overview

**Release Date**: September 27, 2023 | **Organization**: Mistral AI | **License**: Apache 2.0

Mistral 7B is [Mistral AI's](mistral-overview.md) debut model—built in just 3 months by a startup that didn't exist 6 months prior. It proved that architectural efficiency (not just scale) could deliver breakthrough performance.

**Performance**: GPT-3.5 class—[outperforms Llama 2 13B on all benchmarks](https://arxiv.org/abs/2310.06825) and matches Llama 1 34B on reasoning tasks, despite being 2-5x smaller (60.1% MMLU, 52.2% GSM8K).

**Significance**: First model to prove 7B could compete with 13B through architectural innovation. Sparked shift from "bigger is better" to "efficiency matters." European AI sovereignty milestone.

**Technical Innovations**: [Sliding Window Attention (SWA)](../../architecture/attention-mechanisms.md) for efficient long-context, [Grouped Query Attention (GQA)](../../architecture/attention-mechanisms.md) for faster inference. Together, 2x faster inference with 50% less cache memory.

### The BitTorrent Release

On September 26, 2023, [Mistral AI posted only a magnet link](https://mistral.ai/news/announcing-mistral-7b) on Twitter—no press release, no gated access. The model was distributed peer-to-peer, essentially uncensorable once released. This "guerrilla" distribution became legendary in AI circles, contrasting sharply with corporate AI launches.

> **See also**: [Mistral AI Overview](mistral-overview.md) for company background and founder details.

---

## Model Specifications

| Specification | Value |
|--------------|-------|
| **Parameters** | 7.3B |
| **Layers** | 32 |
| **Hidden Dimension** | 4,096 |
| **FFN Dimension** | 14,336 |
| **Attention Heads** | 32 (query), 8 (KV) |
| **Head Dimension** | 128 |
| **Context Window** | 8,192 tokens |
| **Sliding Window** | 4,096 tokens |
| **Vocabulary** | 32,000 tokens (32,768 in v0.3) |
| **Position Encoding** | [RoPE](../../architecture/position-embeddings.md) (θ=10,000; 1M in v0.2 Instruct) |
| **Normalization** | [RMSNorm](../../architecture/normalization.md) |
| **Activation** | [SwiGLU](../../architecture/activations.md) |
| **Data Type** | bfloat16 |
| **License** | Apache 2.0 |

---

## Model Variants

| Variant | Release | Key Changes |
|---------|---------|-------------|
| **v0.1 Base** | Sep 2023 | Foundation model |
| **v0.1 Instruct** | Sep 2023 | Fine-tuned on conversation data, `[INST]`/`[/INST]` format |
| **v0.2 Instruct** | Late 2023 | RoPE theta increased to 1M for better long-context |
| **v0.3 Instruct** | May 2024 | Extended vocabulary (32,768), function calling support |

All variants available on [HuggingFace](https://huggingface.co/mistralai).

---

## Architecture

### Key Innovations

**Sliding Window Attention (SWA)**: Each layer attends only to the previous 4,096 tokens (the "window"). Through recursive propagation across 32 layers, the effective attention span reaches ~131K tokens (4,096 × 32). A rolling buffer cache keeps memory fixed at 4,096 tokens regardless of sequence length—[8x memory reduction](https://arxiv.org/abs/2310.06825) for 32K sequences.

**Grouped Query Attention (GQA)**: 32 query heads share 8 KV heads (4 queries per KV head). This reduces KV cache by 4x compared to standard MHA while maintaining quality—enabling 2x faster inference and higher batch sizes.

**Combined Effect**: SWA + GQA together deliver 2x faster inference and 50% less cache memory compared to Llama 2 architecture.

### Comparison to Llama 2

| Component | Llama 2 | Mistral 7B |
|-----------|---------|------------|
| Attention | MHA | GQA (8 KV heads) |
| Context | Standard (4K) | SWA (4K window, ~131K effective) |
| Inference | Baseline | 2x faster |

---

## Training

Mistral AI did not disclose full training details. Known information:

| Aspect | Details |
|--------|---------|
| **Duration** | ~3 months (May-September 2023) |
| **Infrastructure** | [CoreWeave Cloud](https://www.coreweave.com/) (H100 GPUs), Leonardo EuroHPC |
| **Data** | Not disclosed; estimated up to 8T tokens |
| **Optimizer** | Likely AdamW with cosine schedule (standard practice) |

The founders' experience from LLaMA enabled a compressed timeline—what typically takes 12-18 months was achieved in 90 days.

---

## Performance

### Benchmark Results

[From the official paper](https://arxiv.org/abs/2310.06825):

| Benchmark | Mistral 7B | Llama 2 7B | Llama 2 13B | Category |
|-----------|------------|------------|-------------|----------|
| **MMLU** (5-shot) | **60.1%** | 44.4% | 55.6% | Knowledge |
| **HellaSwag** (0-shot) | **81.3%** | 77.1% | 80.7% | Commonsense |
| **GSM8K** (8-shot) | **52.2%** | 16.0% | 34.3% | Math |
| **MATH** (4-shot) | **13.1%** | 3.9% | 6.0% | Math |
| **HumanEval** (0-shot) | **30.5%** | 11.6% | 18.9% | Code |
| **ARC-Challenge** (0-shot) | **55.5%** | 43.2% | 48.8% | Reasoning |

**Key Result**: Beats Llama 2 13B on 10/12 benchmarks despite being nearly half the size. 3-4x better on math (GSM8K, MATH).

### Instruct Model

**MT-Bench**: Mistral 7B Instruct scored 6.84, surpassing Llama 2 13B Chat (6.65).

**Human Evaluation**: Preferred over Llama 2 13B outputs 54.8% of the time in Chatbot Arena.

---

## Legacy

Mistral 7B proved architectural efficiency could substitute for scale:

- **Before**: 7B competes with 7B, 13B is 2x better
- **After**: Well-designed 7B matches 13B performance

This insight influenced every model released afterward. SWA and GQA became standard techniques. Apache 2.0 licensing set the expectation for "truly open" models.

The model launched Mistral AI to €11B valuation within 18 months, demonstrating open-source and commercial success aren't mutually exclusive.

---

## Sources

### Official
- [Announcing Mistral 7B | Mistral AI](https://mistral.ai/news/announcing-mistral-7b)
- [Mistral 7B Paper (arXiv)](https://arxiv.org/abs/2310.06825)

### HuggingFace
- [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
- [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
