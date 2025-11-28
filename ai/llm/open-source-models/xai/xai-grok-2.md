# xAI Grok-2: Smaller Model, More Active Parameters

## Overview

Grok-2 is xAI's second major model release, [announced in beta August 2024](https://x.ai/news/grok-2). Weights were [later released on HuggingFace](https://huggingface.co/xai-org/grok-2) under a custom community license (not Apache 2.0).

**Performance**: Benchmarks not officially published by xAI. Positioned between Grok-1.5 and Grok-3 in capability.

**Notable**: Grok-2 is *smaller* than Grok-1 (270B vs 314B total) but has *more active parameters* (115B vs 86B)—a more efficient MoE configuration.

### Key Highlights

- **~270B total parameters**, ~115B active per token ([community analysis](https://huggingface.co/xai-org/grok-2/discussions/24))
- **MoE architecture**: 8 experts, 2 active (~43% activation vs Grok-1's 25%)
- **Trained on ~20,000 H100 GPUs** ([Elon Musk, April 2024](https://www.datacenterdynamics.com/en/news/elon-musk-xais-grok-2-requires-20000-nvidia-h100-gpus-grok-3-may-need-100000/))
- **Custom license**: Grok 2 Community License—free for research, restrictions on training other models
- **tiktoken tokenizer** (same format as OpenAI models)

> **See also**: [xAI Overview](xai-overview.md) for company background and infrastructure details.

---

## Model Specifications

| Specification | Value | Source |
|--------------|-------|--------|
| **Total Parameters** | ~270B (269.5B) | [Community analysis](https://huggingface.co/xai-org/grok-2/discussions/24) |
| **Active Parameters** | ~115B per token | Community analysis |
| **Architecture** | MoE: 8 experts, 2 active | Community analysis |
| **Activation Rate** | ~43% (vs Grok-1's 25%) | Derived |
| **Model Size** | ~500 GB | [HuggingFace](https://huggingface.co/xai-org/grok-2) |
| **Inference** | TP=8, requires 8 GPUs (>40GB each) | HuggingFace |
| **Quantization** | FP8 | HuggingFace |
| **Tokenizer** | tiktoken format | HuggingFace |
| **Release** | August 2024 (beta) | xAI |
| **Weights Release** | 2025 | HuggingFace |
| **License** | Grok 2 Community License | HuggingFace |

**Note**: xAI did not publish official architecture specs. Parameters derived from [community analysis of released weights](https://huggingface.co/xai-org/grok-2/discussions/24).

---

## Architecture Changes from Grok-1

| Aspect | Grok-1 | Grok-2 | Change |
|--------|--------|--------|--------|
| **Total Params** | 314B | ~270B | -14% smaller |
| **Active Params** | ~86B | ~115B | +34% more active |
| **Activation Rate** | 25% | ~43% | Higher utilization |
| **Experts** | 8, 2 active | 8, 2 active | Same |

The shift to fewer total but more active parameters suggests xAI optimized for inference efficiency—more compute per token while reducing memory footprint.

---

## Training

Trained on [~20,000 Nvidia H100 GPUs](https://www.datacenterdynamics.com/en/news/elon-musk-xais-grok-2-requires-20000-nvidia-h100-gpus-grok-3-may-need-100000/), likely using Oracle Cloud (before xAI built Colossus).

| Model | Training GPUs | Infrastructure |
|-------|---------------|----------------|
| Grok-1 | ~16,000 | Oracle Cloud |
| Grok-2 | ~20,000 H100 | Oracle Cloud (likely) |
| Grok-3 | 100,000-200,000 H100 | Colossus (Memphis) |

---

## License: Not Open Source

Unlike Grok-1's Apache 2.0 license, Grok-2 uses a **custom community license** with restrictions:

| Aspect | Grok-1 | Grok-2 |
|--------|--------|--------|
| **License** | Apache 2.0 | Grok 2 Community License |
| **Commercial Use** | Unrestricted | Allowed with guidelines |
| **Train Other Models** | Allowed | **Prohibited** |
| **Attribution** | Not required | "Powered by xAI" required |
| **Revocable** | No | Yes |

This is "open weights" not "open source"—you can use the model but cannot use it to train competing models.

---

## Inference Requirements

From [HuggingFace](https://huggingface.co/xai-org/grok-2):

- **Tensor Parallelism**: TP=8 (requires 8 GPUs)
- **GPU Memory**: >40GB per GPU (8x A100-80GB or 8x H100 recommended)
- **Model Size**: ~500 GB download (42 files)
- **Inference Engine**: SGLang v0.5.1+
- **Quantization**: FP8

---

## Availability

- **X Premium+**: Available via X platform
- **API**: [xAI API](https://docs.x.ai/docs/models)
- **Weights**: [HuggingFace](https://huggingface.co/xai-org/grok-2) (Grok 2 Community License)

---

## Sources

### Official
- [Grok-2 Beta Release | xAI](https://x.ai/news/grok-2)
- [xai-org/grok-2 · HuggingFace](https://huggingface.co/xai-org/grok-2)
- [xAI API Documentation](https://docs.x.ai/docs/models)

### Technical Analysis
- [Parameter Scale of Grok-2: 270B Total, 115B Activated](https://huggingface.co/xai-org/grok-2/discussions/24)
- [What do we know about the architecture?](https://huggingface.co/xai-org/grok-2/discussions/6)
- [Grok 2 requires 20,000 H100 GPUs - DCD](https://www.datacenterdynamics.com/en/news/elon-musk-xais-grok-2-requires-20000-nvidia-h100-gpus-grok-3-may-need-100000/)
