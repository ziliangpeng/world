# Other Notable Open Source Models

This document covers additional significant open-source LLMs that have contributed important innovations or served specific niches in the ecosystem.

## Yi Series (01.ai)

### Yi 1.5 34B (May 2024)

**Model Specifications**:
- **Parameters**: 34 billion
- **Training**: 3.1 trillion tokens (bilingual English/Chinese)
- **Vocabulary**: 64,000 tokens (SentencePiece BPE)
- **Context**: Standard transformer context

**Architecture**:
- Modified decoder-only transformer
- Similar to Llama but independently developed
- Uses `LlamaForCausalLM` architecture class
- **Important**: NOT a derivative of Llama weights

**Components**:
- Grouped-Query Attention
- RoPE position embeddings
- SwiGLU activation
- RMSNorm pre-normalization

**Performance**:
- MMLU: 76.3
- BBH: 54.3
- C-Eval: 81.4 (strong Chinese performance)

**Significance**:
- Strong bilingual (English/Chinese) capabilities
- Independent development validates Llama-style architecture
- Excellent Chinese language performance

---

## Falcon Series (TII - Technology Innovation Institute)

### Falcon 40B

**Model Specifications**:
- **Parameters**: 40 billion
- **Layers**: 60
- **Hidden Dimension**: 8,192
- **Training**: 1 trillion tokens from RefinedWeb dataset

**Architecture**:
- Adapted from GPT-3 with key modifications
- **Attention**: Multi-Query Attention (MQA)
- **Position**: Rotary Position Embeddings (RoPE)
- **Optimization**: FlashAttention
- **Novel**: Parallel attention and MLP with two-layer normalization

**Parallel Architecture**:
```python
# Standard transformer:
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))

# Falcon parallel:
attn_out = Attention(LayerNorm(x))
mlp_out = MLP(LayerNorm(x))
x = x + attn_out + mlp_out
```

**Training Infrastructure**:
- 384 A100 40GB GPUs
- 3D parallelism: TP=8, PP=4, DP=12
  - Tensor Parallelism (TP): Split layers
  - Pipeline Parallelism (PP): Split stages
  - Data Parallelism (DP): Replicate model

### Falcon 180B

**Model Specifications**:
- **Parameters**: 180 billion
- **Layers**: 80
- **Hidden Dimension**: 14,848
- **Vocabulary**: 65,024 tokens
- **Context**: 2,048 tokens
- **Training**: 3,500B tokens (3.5T)

**Data Sources**:
- RefinedWeb (high-quality web data)
- Curated corpora (code, conversations, technical)

**Architecture**:
- Multi-Query Attention (MQA)
- RoPE position embeddings
- Parallel attention/MLP from GPT-3 with modifications

**Training Infrastructure**:
- Up to 4,096 A100 40GB GPUs
- 3D parallelism: TP=8, PP=8, DP=64
- ~7 million GPU hours
- One of the largest open-source training efforts at the time

**Performance**:
- State-of-the-art open model when released
- Competitive with PaLM 2, GPT-3.5
- Strong coding and reasoning

**Significance**:
- Demonstrated large-scale open training
- RefinedWeb dataset proved importance of data quality
- MQA validation at scale

### Falcon 3 (Latest)

- Continues Falcon lineage
- Focus on making advanced AI accessible
- Builds on Falcon 180B/40B foundations

---

## BLOOM (BigScience)

**Model Specifications**:
- **Parameters**: 176 billion
- **Architecture**: GPT-3 based decoder-only transformer with modifications
- **Training**: 117 days
- **Languages**: 46 natural languages + 13 programming languages

**Key Features**:
- Different positional embeddings than standard GPT-3
- ALiBi (Attention with Linear Biases) for position encoding
- Truly multilingual from the ground up

**Significance**:
- Community-built (HuggingFace + BigScience consortium)
- Proved community can build large models
- Strong multilingual capabilities
- Open research artifact

**Impact**:
- Democratized LLM research
- Enabled multilingual AI research
- Blueprint for community model development

---

## EleutherAI Models

### GPT-NeoX-20B

**Model Specifications**:
- **Parameters**: 20 billion
- **Layers**: 44
- **Hidden Dimension**: 6,144
- **Attention Heads**: 64

**Architecture**:
- Autoregressive transformer (GPT-3 based)
- **Training Data**: The Pile dataset
- **Position**: No learned positional embeddings

**Key Differences from GPT-3**:
1. FFN initialization: `(1/L√d)` scheme from GPT-J
2. Other layers: Small init `(2/d+4d)`
3. No positional embedding learning

**Significance**:
- Fully open-source: weights, training code, evaluation code
- Enabled research into training dynamics
- The Pile dataset became widely used

### GPT-J (6B)

**Model Specifications**:
- **Parameters**: 6 billion
- Open-source alternative to GPT-3

**Innovations**:
- Novel initialization schemes
- Inspired later models
- Proved viability of open alternatives

**Impact**:
- One of the first high-quality open alternatives
- Widely used for research and applications
- Initialization techniques adopted by others

---

## StableLM (Stability AI)

### Stable LM 2 1.6B (February 2024)

**Model Specifications**:
- **Parameters**: 1.6 billion
- **Training**: 2 trillion tokens across 7 languages
- **Tokenizer**: Arcade100k (BPE extended from tiktoken.cl100k_base)

**Architecture**:
- Decoder-only transformer (LLaMA-like)
- **Position**: RoPE applied to first 25% of head embedding dimensions
- **Normalization**: LayerNorm with learned bias (vs RMSNorm)

**Unique Features**:
1. Removed bias terms from FFN and attention (except QKV projections)
2. Digits split into individual tokens
3. Partial RoPE application (only 25% of dimensions)

**Training Infrastructure**:
- 512 NVIDIA A100 40GB GPUs

### Stable LM 2 12B (October 2024)

**Model Specifications**:
- **Parameters**: 12 billion
- **Training**: 2 trillion tokens across 7 languages
- **Tokenizer**: Arcade100k

**Architecture Enhancements**:
- RoPE applied to first 25% of head dimensions
- **Parallel attention and feed-forward** with single LayerNorm
- LayerNorm without biases
- Per-head QK normalization
- All bias terms removed from FFN and GQA layers

**Training Infrastructure**:
- 384 NVIDIA H100 GPUs

**Significance**:
- Efficient small-to-medium models
- Multilingual from the start
- Aggressive architectural simplification (bias removal)

---

## MPT (MosaicML)

### MPT-7B

**Model Specifications**:
- **Architecture**: Modified GPT-style decoder-only transformer
- **Context**: 8K tokens (extendable via finetuning)
- **Training**: 1T tokens, 440 A100-40GB GPUs, ~9.5 days

**Key Innovations**:

#### 1. ALiBi (Attention with Linear Biases)
- Replaces positional embeddings entirely
- Linear bias added to attention scores
- Excellent context length extrapolation
- Lower memory footprint

#### 2. FlashAttention
- Memory-efficient attention implementation
- Faster training and inference

#### 3. QK LayerNorm
- Normalization applied to queries and keys
- Improved training stability

**Training Details**:
- Batch size: 1760
- Sequence length: 2048
- ~9.5 days on 440 A100s

### MPT-30B

**Model Specifications**:
- **Context**: 8K tokens
- Same architectural features as MPT-7B
- ALiBi + FlashAttention combination

**2024 Update**:
- ALiBi now in FlashAttention v2.4
- 4-5x speedup for ALiBi-based models
- Makes ALiBi more practical

**Significance**:
- Validated ALiBi for production use
- Showed importance of efficient attention
- Context extrapolation capabilities

---

## Apple OpenELM (April 2024)

### Model Variants
- **270M**: 270 million parameters
- **450M**: 450 million parameters
- **1.1B**: 1.1 billion parameters
- **3B**: 3 billion parameters

### Unique Architecture

**Layer-Wise Scaling**:
Each transformer layer has different configuration:
```
Layer 1 (near input): Smaller latent dimensions
Layer 2: Slightly larger
...
Layer N (near output): Widest dimensions
```

**Concept**:
- Early layers: Simple pattern recognition
- Later layers: Complex reasoning
- Gradual scaling matches computational needs

**Components**:
- Grouped-Query Attention
- SwiGLU feed-forward networks
- RoPE position embeddings

### Training Details
- **Tokens**: ~1.8 trillion
- **Data**: RefinedWeb, PILE, RedPajama, Dolma v1.6
- Both pretrained and instruction-tuned versions

### Significance
- Novel layer-wise scaling approach
- Efficient small models
- Apple's first open LLM
- Validated gradual scaling hypothesis

---

## Architectural Patterns Across Models

### Common Foundations

Most modern open-source models share:
1. **Decoder-only transformer** base
2. **RoPE or ALiBi** for position encoding
3. **GQA or MQA** for efficiency
4. **SwiGLU** activation (or GELU)
5. **RMSNorm or LayerNorm** for normalization
6. **FlashAttention** for efficiency

### Divergent Choices

**Attention**:
- MQA: Falcon, Gemma 2B
- GQA: Most modern models
- MHA: Older or smaller models

**Position**:
- RoPE: Llama-style models, Yi, StableLM
- ALiBi: MPT, BLOOM
- Standard learned: Older models

**Normalization**:
- RMSNorm: Llama-style, Qwen, most new models
- LayerNorm: GPT-style, StableLM, some others

### Innovation Areas

**Falcon**: Parallel attention/MLP, RefinedWeb data
**BLOOM**: Multilingual at scale, community development
**MPT**: ALiBi for context extrapolation
**Yi**: Bilingual excellence
**StableLM**: Bias removal, partial RoPE
**OpenELM**: Layer-wise scaling

## Comparative Summary

| Model | Size | Key Innovation | Use Case |
|-------|------|----------------|----------|
| Yi 34B | 34B | Bilingual EN/CN | Chinese + English |
| Falcon 180B | 180B | MQA at scale, RefinedWeb | Large-scale open model |
| BLOOM 176B | 176B | 46 languages, community | Multilingual research |
| GPT-NeoX 20B | 20B | Open training code | Research transparency |
| StableLM 12B | 12B | Bias removal, partial RoPE | Efficient deployment |
| MPT 7B | 7B | ALiBi, long context | Context extrapolation |
| OpenELM 3B | 3B | Layer-wise scaling | Efficient small model |

## Historical Context

These models represent different eras and approaches:

**Early Era** (2021-2022):
- GPT-J, GPT-NeoX: Proving open models viable
- BLOOM: Community collaboration

**Scaling Era** (2022-2023):
- Falcon 180B: Large-scale open training
- Yi: Bilingual excellence

**Efficiency Era** (2024):
- StableLM: Architectural simplification
- OpenELM: Novel scaling approaches

## Sources

- [Yi 34B - Hugging Face](https://huggingface.co/01-ai/Yi-34B)
- [Yi: Open Foundation Models](https://arxiv.org/abs/2403.04652)
- [Yi 1.5 34B Introduction](https://www.marktechpost.com/2024/05/18/01-ai-introduces-yi-1-5-34b-model-an-upgraded-version-of-yi-with-a-high-quality-corpus-of-500b-tokens-and-fine-tuned-on-3m-diverse-fine-tuning-samples/)
- [Falcon 180B - Hugging Face](https://huggingface.co/tiiuae/falcon-180B)
- [Falcon 180B and 40B Comparison](https://meetcody.ai/blog/falcon-180b-40b-difference-usecase-performance-architecture-open-source/)
- [Spread Your Wings: Falcon 180B](https://huggingface.co/blog/falcon-180b)
- [Introduction to Falcon 40B](https://www.datacamp.com/tutorial/introduction-to-falcon-40b)
- [BLOOM Open-Source Alternative](https://the-decoder.com/bloom-is-a-real-open-source-alternative-to-gpt-3/)
- [GPT-NeoX-20B](https://arxiv.org/abs/2204.06745)
- [GPT-NeoX GitHub](https://github.com/EleutherAI/gpt-neox)
- [StableLM GitHub](https://github.com/Stability-AI/StableLM)
- [Stable LM 2 1.6B - Hugging Face](https://huggingface.co/stabilityai/stablelm-2-1_6b)
- [Stability AI Releases Stable LM 2](https://www.infoq.com/news/2024/01/stabie-lm-2/)
- [Introducing Stable LM 2 12B](https://stability.ai/news/introducing-stable-lm-2-12b)
- [Stable LM 2 Technical Report](https://arxiv.org/html/2402.17834v1)
- [MPT-7B - Hugging Face](https://huggingface.co/mosaicml/mpt-7b)
- [Introducing MPT-7B](https://www.databricks.com/blog/mpt-7b)
- [ALiBi FlashAttention](https://pli.princeton.edu/blog/2024/alibi-flashattention-speeding-alibi-3-5x-hardware-efficient-implementation)
- [Apple Open-Sources OpenELM](https://www.infoq.com/news/2024/05/apple-llm-openelm/)
- [OpenELM - Apple Research](https://machinelearning.apple.com/research/openelm)
- [OpenELM - Hugging Face](https://huggingface.co/apple/OpenELM)
