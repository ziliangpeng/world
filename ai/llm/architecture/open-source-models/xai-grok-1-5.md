# Grok-1.5 and Grok-1.5V: Extended Context and Multimodal Models

## Overview

Grok-1.5 is xAI's second-generation language model, [announced March 29, 2024](https://x.ai/news/grok-1.5)—just 12 days after Grok-1 was open-sourced. Key upgrade: 16x context window expansion (8K → 128K tokens) and significant reasoning improvements.

**Performance**: Between GPT-3.5 and GPT-4—major improvement over Grok-1, competitive on math/coding but trails GPT-4 on general knowledge (81% vs 86% MMLU).

[Grok-1.5V (Vision)](https://x.ai/news/grok-1.5v) followed on April 12, 2024, adding multimodal capabilities—xAI's first model to process images.

### Key Highlights

- **128K token context** (16x over Grok-1's 8K)—matches GPT-4 Turbo
- **90% on GSM8K** (up from ~63%)—dramatic math reasoning improvement
- **100% Needle-in-Haystack** at 128K tokens—perfect long-context retrieval
- **Vision capabilities** in 1.5V variant
- **Same 314B MoE architecture** as Grok-1—improvements came from training, not architecture changes
- **Closed source**—available only via X Premium+ (unlike Grok-1's Apache 2.0 release)

> **See also**: [xAI Overview](xai-overview.md) for company background and infrastructure details.

---

## Model Specifications

| Specification | Grok-1.5 | Grok-1.5V |
|--------------|----------|-----------|
| **Parameters** | 314B total | 314B+ (vision encoder added) |
| **Active Parameters** | ~78B (25% activation) | ~78B+ |
| **Architecture** | MoE: 8 experts, 2 active | MoE + Vision Encoder |
| **Context Window** | 128,000 tokens | 128,000 tokens |
| **Layers** | 64 | 64 |
| **Attention** | GQA: 48Q/8KV heads | GQA: 48Q/8KV heads |
| **Position Encoding** | RoPE (extended) | RoPE (extended) |
| **Multimodal** | Text only | Text + Images |
| **Release** | March 29, 2024 | April 12, 2024 |
| **Availability** | X Premium+ | X Premium+ |
| **License** | Closed | Closed |

### Context Extension

Extended from 8K to 128K using RoPE scaling techniques (likely [YaRN](https://blog.eleuther.ai/yarn/) or similar). The 100% Needle-in-a-Haystack score confirms effective long-context handling without degradation.

---

## Key Improvements Over Grok-1

| Capability | Grok-1 | Grok-1.5 | Change |
|------------|--------|----------|--------|
| **Context Window** | 8,192 | 128,000 | **16x** |
| **GSM8K (math)** | ~63% | 90% | +27 pp |
| **MATH** | ~23.9% | 50.6% | +26.7 pp |
| **HumanEval (code)** | ~63.2% | 74.1% | +10.9 pp |
| **MMLU** | ~73% | 81.3% | +8.3 pp |
| **Needle-in-Haystack** | N/A | 100% | Perfect |

**What changed**: Same architecture, better training data. The dramatic math improvements suggest [significantly enhanced mathematical reasoning data](https://synthedia.substack.com/p/grok-15-closes-gap-with-openai-google) and longer training sequences.

---

## Performance Benchmarks

### Grok-1.5

| Benchmark | Grok-1.5 | GPT-4 | Claude 3 Opus | Gemini 1.5 Pro |
|-----------|----------|-------|---------------|----------------|
| **MMLU** | 81.3% | 86.4% | 86.8% | 81.9% |
| **GSM8K** | 90% | 92% | 95% | 91.7% |
| **MATH** | 50.6% | 52.9% | 60.1% | 58.5% |
| **HumanEval** | 74.1% | 67% | 84.9% | 71.9% |

Grok-1.5 [closed the gap significantly](https://www.maginative.com/article/x-ai-announces-grok-1-5/) with GPT-4 and Claude 3, particularly on math reasoning.

### Grok-1.5V (Vision)

| Benchmark | Grok-1.5V | GPT-4V | Claude 3 Opus | Gemini Pro 1.5 |
|-----------|-----------|--------|---------------|----------------|
| **MMMU** | 53.6% | 56.8% | 59.4% | 58.5% |
| **MathVista** | 52.8% | 49.9% | 50.5% | 52.1% |
| **AI2D** | 88.3% | 78.2% | 70.6% | 80.3% |
| **TextVQA** | 78.1% | 78.0% | - | 73.5% |
| **ChartQA** | 76.1% | 78.5% | 80.8% | 81.3% |
| **DocVQA** | 85.6% | 88.4% | 89.3% | 86.5% |
| **RealWorldQA** | 68.7% | 61.4% | - | - |

---

## Grok-1.5V: Vision Capabilities

### RealWorldQA Benchmark

xAI [introduced RealWorldQA](https://the-decoder.com/xai-introduces-grok-1-5-vision-multimodal-ai-model-and-a-physical-world-benchmark/) alongside Grok-1.5V—a benchmark testing real-world spatial understanding using driving scenario images.

**Grok-1.5V scored 68.7%**, outperforming GPT-4V (61.4%) on this physical world reasoning task.

### Vision Architecture

- **Base**: Grok-1.5 language model (314B MoE)
- **Vision Encoder**: Dedicated ViT for image processing
- **Fusion**: Multimodal integration layer
- **Input Types**: Documents, diagrams, charts, screenshots, photographs

### Demonstrated Capabilities

From the [official announcement](https://x.ai/news/grok-1.5v):

- **Code from drawings**: Hand-drawn UI mockups → functional code
- **Document processing**: PDF extraction, OCR, table parsing
- **Chart interpretation**: Data visualization understanding
- **Real-world reasoning**: Physical inspection tasks (e.g., identifying structural issues)

---

## Development Speed

xAI's rapid iteration from Grok-1 to Grok-1.5 (~4 months) reflects:

1. **Same architecture**: No fundamental redesign—focused on training improvements
2. **Context extension**: RoPE scaling is a known technique, not novel research
3. **Competitive pressure**: GPT-4 Turbo (128K), Claude 3 (200K), Gemini 1.5 (1M) all had longer context
4. **Small team velocity**: xAI's lean structure enables fast iteration

---

## Availability

- **X Premium+**: $22/month (as of late 2024)
- **API**: Available via [xAI API](https://docs.x.ai/docs/models)
- **Open Source**: No—unlike Grok-1, these remain closed

---

## Sources

### Official
- [Announcing Grok-1.5 | xAI](https://x.ai/news/grok-1.5)
- [Grok-1.5 Vision Preview | xAI](https://x.ai/news/grok-1.5v)
- [xAI API Documentation](https://docs.x.ai/docs/models)

### Analysis
- [Grok-1.5 Closes Gap with OpenAI, Google, and Anthropic](https://synthedia.substack.com/p/grok-15-closes-gap-with-openai-google)
- [xAI introduces Grok-1.5 Vision and RealWorldQA benchmark](https://the-decoder.com/xai-introduces-grok-1-5-vision-multimodal-ai-model-and-a-physical-world-benchmark/)
- [X.ai Announces Grok-1.5 with Improved Reasoning](https://www.maginative.com/article/x-ai-announces-grok-1-5/)
- [Extending the RoPE | EleutherAI](https://blog.eleuther.ai/yarn/)
