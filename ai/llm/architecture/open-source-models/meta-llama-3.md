# Llama 3 Family (3, 3.1, 3.2, 3.3)

The Llama 3 family represents Meta's 2024 push for state-of-the-art open models, spanning from tiny edge devices (1B) to GPT-4 class models (405B), with multimodal capabilities.

---

## Llama 3 (April 2024)

**Release Date**: April 18, 2024

The foundation of the Llama 3 family - a complete architectural overhaul with massive improvements.

### Model Variants

- **8B**: 8 billion parameters
- **70B**: 70 billion parameters

**Both available as Base and Instruct**

### Architecture Updates

**Enhanced Design**: Optimized transformer decoder

**Key Improvements**:
- **Attention**: GQA extended to ALL model sizes (including 8B) - previously only 70B had GQA
- **Tokenizer**: Upgraded to TikToken with **~128K vocabulary** (4x expansion from 32K)
- **Context**: 8K tokens (2x Llama 2's 4K)

### Training Details

- **Tokens**: **15T+ tokens** from publicly available sources (7.5x Llama 2's 2T)
- **Dataset**: Multilingual, code-heavy, high-quality curation
- **Quality Focus**: Extensive filtering and curation

### Innovations

- **GQA Everywhere**: Extended to smaller models (8B), validating efficiency gains at all scales
- **Massive Vocabulary Expansion**: Better multilingual support, reduced token count for same text
- **Superior Performance**: Best performance per parameter vs Llama 2

### Links

- **Paper**: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- **Blog**: [Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)
- **Hugging Face**:
  - Base: [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B), [Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
  - Instruct: [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)

---

## Llama 3.1 (July 2024)

**Release Date**: July 23, 2024

The flagship release - massive context window and first open 400B+ model.

### Model Variants

- **8B**: 8 billion parameters
- **70B**: 70 billion parameters
- **405B**: 405 billion parameters (flagship)

**All available as Base and Instruct**

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

- **Tokens**: 15T+ tokens (same data as Llama 3)
- **Context Window**: **128K tokens** (16x expansion from Llama 3's 8K!)
- **Multilingual**: Enhanced support for multiple languages

### Major Innovation: Context Scaling

**The Big Change**: 8K → 128K context
- Can now process entire books in one go
- Full codebase analysis
- Long conversation history
- Multi-document analysis

**How**: RoPE scaling techniques to extend beyond training length

### Significance

- **First open model to compete with GPT-4** class models
- **Massive context window expansion** enables new use cases
- **Demonstrated scaling laws continue to work** at 400B+ parameters
- **Proved open can match closed** on frontier capabilities

### Links

- **Paper**: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- **Blog**: [Introducing Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/)
- **Hugging Face**:
  - Base: [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B), [Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)
  - Instruct: [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)

---

## Llama 3.2 (September 2024)

**Release Date**: September 25, 2024

Diversification into edge devices and multimodal capabilities.

### Model Variants

**Text-Only Models**:
- **1B**: 1 billion parameters
- **3B**: 3 billion parameters

**Vision Models**:
- **11B Vision**: 11 billion parameters (built on Llama 3.1 8B)
- **90B Vision**: 90 billion parameters (built on Llama 3.1 70B)

### Architecture

**Text-Only Models** (1B, 3B):
- Same foundation as Llama 3/3.1 (GQA, RoPE, SwiGLU, RMSNorm)
- Optimized for edge deployment and on-device inference
- Maintains architectural consistency with larger siblings
- **Use Case**: Mobile phones, IoT devices, edge computing

**Vision Models - Multimodal Architecture**:

**Two-Stage Vision Processing**:
1. **Stage 1 - Feature Extraction**:
   - 32-layer transformer processing patched image inputs
   - Outputs 1280-dimensional features
   - Preserves intermediate representations

2. **Stage 2 - Global Encoding**:
   - 8-layer global encoder with gated attention
   - Concatenates intermediate features with final output
   - Creates rich multi-level visual representation

**Cross-Attention Integration**:
- Language component: 40-layer decoder-only transformer (4096 hidden size)
- Cross-attention layers integrated every 5th layer
- Separately trained adapter weights connect vision and language
- Adapter trained on **6 billion image-text pairs**
- Vision encoder updated, language model frozen (preserves text capabilities)

### Training

**Vision Model Pretraining**:
- 6B image-text pairs
- Adapter-based approach
- Maintains Llama 3.1 text capabilities
- Drop-in replacement for corresponding text models

### Capabilities

**Text-Only** (1B, 3B):
- Edge device deployment
- On-device inference
- Resource-constrained environments
- Real-time applications

**Vision Models** (11B, 90B):
- Visual recognition and reasoning
- Image captioning
- Visual question answering
- Document understanding with charts/graphs
- Maintains all text-only capabilities

### Links

- **Paper**: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- **Blog**: [Llama 3.2: Revolutionizing edge AI and vision](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- **Hugging Face**: [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B), [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B), [Llama-3.2-11B-Vision](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision), [Llama-3.2-90B-Vision](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision)

---

## Llama 3.3 (December 2024)

**Release Date**: Late 2024

Incremental improvements and optimizations.

### Model Variant

- **70B**: 70 billion parameters (Instruct only)

### Updates

- Latest iteration maintaining architectural consistency
- Further optimizations for efficiency and performance
- Continued refinement of training data and processes
- **Claimed**: 70B performance matching 3.1's 405B with fewer compute requirements

### Links

- **Blog**: [The future of AI: Built with Llama](https://ai.meta.com/blog/future-of-ai-built-with-llama/)
- **Hugging Face**: [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)

---

## Llama 3 Family Evolution Summary

| Version | Release | Sizes | Context | Key Innovation |
|---------|---------|-------|---------|----------------|
| **3** | Apr 2024 | 8B, 70B | 8K | GQA everywhere, 128K vocab, 15T tokens |
| **3.1** | Jul 2024 | 8B, 70B, **405B** | **128K** | Long context, GPT-4 class |
| **3.2** | Sep 2024 | **1B, 3B**, 11B Vision, 90B Vision | Varies | **Edge + Multimodal** |
| **3.3** | Dec 2024 | 70B | 128K | Efficiency improvements |

## Common Architecture Foundation

All Llama 3 family models share:
- **Decoder-only transformer**
- **GQA** (Grouped-Query Attention)
- **RoPE** (Rotary Position Embeddings)
- **SwiGLU** activation
- **RMSNorm** pre-normalization
- **TikToken** tokenizer with ~128K vocabulary

## Impact

The Llama 3 family demonstrated that:
1. **Open models can match GPT-4** (405B)
2. **Small models can be powerful** (1B-3B for edge)
3. **Multimodal is achievable** (Vision models)
4. **Context can scale dramatically** (8K → 128K)
5. **One architecture scales everywhere** (1B → 405B)

This family solidified Meta's position as the leader in open-source AI.
