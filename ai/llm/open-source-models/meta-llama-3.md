# Llama 3 Family (3, 3.1, 3.2, 3.3)

The Llama 3 family represents Meta's 2024 push for state-of-the-art open models, spanning from tiny edge devices (1B) to GPT-4 class models (405B), with multimodal capabilities.

## Origin Story: Proving Open Can Match Closed

The Llama 3 project emerged from Meta's bold vision to prove that open-source AI could compete with—and even surpass—the best proprietary models. After Llama 2's successful commercial release in July 2023 demonstrated the viability of open foundation models, Meta committed to an ambitious goal: create the first open model to match GPT-4 class performance.

### The Strategic Vision

Mark Zuckerberg's strategy centered on **democratizing AI through open source**. By releasing state-of-the-art models openly, Meta aimed to commoditize the AI model market, reducing the pricing power of competitors with closed models while establishing Meta as the leader in open AI. This wasn't just altruism—it was a calculated business strategy to ensure Meta wouldn't be dependent on proprietary AI providers.

The vision extended beyond a single flagship model. Meta's team recognized that the AI landscape required models across the entire spectrum:
- **Edge devices** (1B-3B) for on-device, privacy-preserving inference
- **Mid-range** (8B) for efficient production deployments
- **Large** (70B) for high-quality general-purpose use
- **Frontier** (405B) to match GPT-4 and prove open source could compete at the highest level
- **Multimodal** capabilities to compete with GPT-4V, Claude 3 Vision, and Gemini

### Development Timeline

- **July 2023**: Development began immediately after Llama 2's release
- **April 18, 2024**: Llama 3 (8B, 70B) released with massive improvements
- **July 23, 2024**: Llama 3.1 flagship 405B released with 128K context
- **September 25, 2024**: Llama 3.2 diversified into edge (1B, 3B) and vision (11B, 90B)
- **December 6, 2024**: Llama 3.3 (70B) proved smaller models can match larger ones with better training

### Team Organization and Evolution

Meta's AI research underwent significant reorganization during Llama 3's development:

- **Leadership**: Chief AI Scientist Yann LeCun (recruited in 2013 as one of the "godfathers of modern AI") and VP of Research Joelle Pineau jointly led AI research, reporting to Chief Product Officer Chris Cox.
- **February 2024 Reorganization**: Meta established a dedicated "Generative AI" team led by former Apple executive Ahmad Al-Dahle, separating production/product development from fundamental research (FAIR).
- **Team Turnover**: More than half the authors of the original Llama research paper left Meta within months of publication, with many joining the new Generative AI team, including Hugo Touvron, Thibaut Lavril, Xavier Martinet, Marie-Anne Lachaux, Naman Goyal, and Aurelien Rodriguez.
- **Recent Hires**: Shengjia Zhao, co-creator of ChatGPT from OpenAI, was hired as chief scientist of Meta Superintelligence Labs to lead next-generation research.

The Llama 3 paper lists Aaron Grattafiori and **558 other authors**, reflecting the massive collaborative effort across Meta AI—a dramatic expansion from the smaller teams that created Llama 1 and 2.

### Key Strategic Decisions

**Progressive Scaling Approach**: Rather than releasing all sizes simultaneously, Meta took a staged approach:
1. Start with 8B/70B to prove architectural improvements (April 2024)
2. Release flagship 405B with extended context (July 2024)
3. Diversify to edge and multimodal (September 2024)
4. Optimize for efficiency (December 2024)

**Infrastructure Migration**: While Llama 1 and 2 were trained on Meta's AI Research SuperCluster (RSC), Llama 3 migrated to Meta's production clusters, requiring the team to scale to **16,384 H100 GPUs** and build robust fault-tolerance systems to handle failures every 3 hours on average.

**Context Window Expansion**: The progressive scaling from Llama 2's 4K → Llama 3's 8K → Llama 3.1's 128K demonstrated the feasibility of dramatic context expansion through continued pre-training rather than full retraining.

**Open Source Commitment**: By fully open-sourcing even the 405B flagship model, Meta made an unprecedented bet that giving away their most powerful AI would ultimately strengthen their competitive position through ecosystem effects and commoditizing competitors' advantages.

### Response to Industry Competition

Llama 3 was explicitly designed to compete with the leading proprietary models:
- **GPT-4** (OpenAI): Llama 3.1 405B became the first open model to match GPT-4 class performance
- **Claude 3/3.5** (Anthropic): Competitive on reasoning and safety benchmarks
- **Gemini** (Google): Matched on multilingual and multimodal capabilities

The **December 2024 release of Llama 3.3 70B** delivered a critical proof point: through superior training methodology, a 70B model could match the 405B flagship's performance at a fraction of the cost—proving that **training quality matters as much as scale**.

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

## Architecture: Evolution from Llama 2

The Llama 3 family maintains the core decoder-only transformer architecture from Llama 2 while introducing significant improvements in efficiency and capability.

### Core Architectural Components

All Llama 3 family models share these foundational elements:
- **Base Design**: Auto-regressive decoder-only transformer (unchanged from Llama 2)
- **Normalization**: RMSNorm pre-normalization (unchanged from Llama 2)
- **Activation**: SwiGLU activation function (unchanged from Llama 2)
- **Position Encoding**: RoPE (Rotary Position Embeddings) with θ = 500,000 (unchanged from Llama 2)
- **Attention**: Grouped-Query Attention (GQA) across **all model sizes** (expanded from Llama 2)
- **Tokenizer**: TikToken-based BPE with **128,256 tokens** (4x expansion from Llama 2's 32K)

### Complete Model Specifications

| Model | Layers | Hidden Dim | FFN Dim | Attn Heads | KV Heads | Vocab Size | Context | Parameters |
|-------|--------|------------|---------|------------|----------|------------|---------|------------|
| **1B** | 16 | 2,048 | 8,192 | 32 | 8 | 128,256 | 128K | 1.23B |
| **3B** | 28 | 3,072 | 8,192 | 24 | 8 | 128,256 | 128K | 3.21B |
| **8B** | 32 | 4,096 | 14,336 | 32 | 8 | 128,256 | 8K→128K | 8B |
| **70B** | 80 | 8,192 | 28,672 | 64 | 8 | 128,256 | 8K→128K | 70B |
| **405B** | 126 | 16,384 | 53,248 | 128 | 8 | 128,256 | 128K | 405B |

**Llama 2 Comparison** (70B model):
| Aspect | Llama 2 70B | Llama 3 70B | Change |
|--------|-------------|-------------|--------|
| **Layers** | 80 | 80 | Same |
| **Hidden Dim** | 8,192 | 8,192 | Same |
| **FFN Dim** | 28,672 | 28,672 | Same |
| **Attn Heads** | 64 | 64 | Same |
| **KV Heads** | 8 | 8 | Same (GQA) |
| **Vocab Size** | 32,000 | 128,256 | **4x expansion** |
| **Context** | 4,096 | 8,192→128K | **2x→32x expansion** |

### Key Architectural Improvements Over Llama 2

**1. GQA Extended to All Model Sizes**

Llama 2 used Grouped-Query Attention (GQA) only for the 70B model, while the 7B and 13B models used standard Multi-Head Attention (MHA). Llama 3 **standardizes on GQA across all sizes**, with all models using 8 key-value heads:

| Model | Query Heads | KV Heads | Ratio | Memory Savings |
|-------|-------------|----------|-------|----------------|
| **8B** | 32 | 8 | 4:1 | 75% KV cache reduction |
| **70B** | 64 | 8 | 8:1 | 87.5% KV cache reduction |
| **405B** | 128 | 8 | 16:1 | 93.75% KV cache reduction |

This provides **memory bandwidth efficiency** during inference while maintaining quality, enabling:
- Faster inference speeds
- Larger batch sizes
- Lower memory requirements
- Better scaling to long context windows

**2. Tokenizer Revolution: SentencePiece → TikToken**

The most visible change from Llama 2 is the complete tokenizer overhaul:

| Aspect | Llama 2 | Llama 3 |
|--------|---------|---------|
| **Implementation** | SentencePiece BPE | TikToken-based BPE |
| **Vocabulary Size** | 32,000 | 128,256 (128K + 256 special) |
| **Base Tokens** | Custom training | 100K from OpenAI's tiktoken |
| **Additional Tokens** | N/A | +28K for multilingual support |

**Benefits of the larger vocabulary**:
- **Multilingual efficiency**: Better encoding for 30+ languages
- **Code efficiency**: Improved representation of programming syntax
- **Compression**: Same text represented in ~25% fewer tokens
- **Compatibility**: Shared foundation with GPT-4's tokenizer

**Special tokens**:
- `<|begin_of_text|>`: Beginning of Sequence (BOS)
- `<|end_of_text|>`: End of Sequence (EOS), complete termination
- `<|eot_id|>`: End of Turn, marks message boundaries in conversation
- `<|start_header_id|>`, `<|end_header_id|>`: Header markers
- Plus 256 additional reserved tokens

**3. Context Window Scaling**

Llama 3 achieved dramatic context window expansion through RoPE scaling:

| Model | Llama 2 | Llama 3 | Llama 3.1 |
|-------|---------|---------|-----------|
| **Context** | 4,096 | 8,192 | 128,000 |
| **Expansion** | Baseline | 2x | 32x |

**Technique**: YaRN (Yet Another RoPE extensioN) frequency scaling
- Base RoPE with θ = 500,000
- Continued pre-training on ~800B tokens for long context
- Scale factors (s=16, s=32) applied during fine-tuning
- Achieves extension using only ~5% of original training compute

### Vision Model Architecture (Llama 3.2)

**New capability**: Llama 3 introduced multimodal vision models through an adapter-based architecture.

**Two-Tier Vision Processing**:

1. **Vision Encoder** (ViT-H/14 based):
   - Initial parameters: 630M (pretrained)
   - Enhanced with: 8 gated self-attention layers
   - Total parameters: 850M
   - **Stage 1**: 32-layer transformer processing 16×16 image patches
   - **Stage 2**: 8-layer global encoder with gated attention
   - Output: 7,680-dimensional representation per patch (256 patches per image)

2. **Cross-Attention Integration**:
   - Cross-attention layers inserted **every 5 layers** in language model
   - Uses GQA for efficiency
   - **11B Vision**: Built on Llama 3.1 8B (40-layer decoder, 4096 hidden size)
   - **90B Vision**: Built on Llama 3.1 70B (80-layer decoder, 8192 hidden size)

**Training Approach**:
- Adapter weights trained on **6 billion image-text pairs**
- Vision encoder updated, **language model frozen** (preserves text capabilities)
- Post-training on 3M+ synthetic instruction examples
- Acts as **drop-in replacement** for corresponding text models

This architecture enables vision capabilities without degrading text-only performance—a critical advantage over models that require joint training from scratch.

## Training Details: Scaling Beyond Llama 2

Llama 3's training represents a massive scale-up from Llama 2, with 7.5x more data, 2x larger context windows, and significantly more compute resources.

### Optimizer Configuration

**Pre-training** (405B model):
- **Optimizer**: AdamW (same as Llama 2)
- **Betas**: β₁ = 0.9, β₂ = 0.95 (unchanged from Llama 2)
- **Weight Decay**: 0.1 (unchanged from Llama 2)
- **Gradient Clipping**: Max norm of 1.0 (unchanged from Llama 2)
- **Warmup**: Linear warmup over 8,000 steps (4x longer than Llama 2's 2,000 steps)

**Learning Rates by Model Size**:

| Model Size | Peak Learning Rate | Schedule | Llama 2 Peak LR |
|------------|-------------------|----------|-----------------|
| **8B** | 3×10⁻⁴ | Cosine decay over 1.2M steps | 3×10⁻⁴ (7B, same) |
| **70B** | 1.5×10⁻⁴ | Cosine decay over 1.2M steps | 1.5×10⁻⁴ (same) |
| **405B** | 8×10⁻⁵ → 8×10⁻⁷ | Cosine decay over 1.2M steps | N/A (new size) |

**Fine-tuning** (SFT/DPO):
- **Learning Rate**: 1×10⁻⁵ (10x lower than pre-training)
- **Weight Decay**: 0.01 to 0.1 (task-dependent)
- **Gradient Clipping**: 0.3 to 1.0
- **Training Duration**: 8,500-9,000 steps
- **DPO β parameter**: 0.1

### Training Scale: 7.5x Data Expansion from Llama 2

**Token Counts**:

| Aspect | Llama 2 | Llama 3 | Increase |
|--------|---------|---------|----------|
| **Pre-training Tokens** | 2 trillion | **15+ trillion** | **7.5x** |
| **Long Context Extension** | N/A | ~800B additional | New capability |
| **Vision Training** | N/A | 6B image-text pairs | New modality |

**Batch Size Schedule** (Progressive scaling):

| Phase | Tokens Processed | Batch Size | Sequence Length | Llama 2 Batch Size |
|-------|-----------------|------------|-----------------|-------------------|
| **Phase 1** | 0 → 252M | 4M tokens | 4,096 | 4M tokens (same) |
| **Phase 2** | 252M → 2.87T | 8M tokens | 8,192 | N/A (larger) |
| **Phase 3** | 2.87T → 15T+ | 16M tokens | 8,192 | N/A (larger) |

This progressive approach allowed gradual hardware utilization increase while maintaining training stability.

**Context Windows**:

| Model | Llama 2 | Llama 3 | Llama 3.1 | Expansion |
|-------|---------|---------|-----------|-----------|
| **All sizes** | 4,096 | 8,192 | 128,000 | **2x → 32x** |

### Data Mix: Quality Over Quantity

**Total**: ~15 trillion tokens from publicly available sources

| Category | Llama 2 | Llama 3 | Change |
|----------|---------|---------|--------|
| **General Knowledge** | ~67% (CommonCrawl, C4) | ~50% | More balanced |
| **Mathematical & Reasoning** | Minimal | ~25% | **Massive increase** |
| **Code** | ~4.5% (GitHub) | ~17% | **~4x increase** |
| **Multilingual** | Minimal (<2%) | ~8% (30+ languages) | **>4x increase** |
| **Other** (Wikipedia, Books, ArXiv, StackExchange) | ~13% | ~0% (integrated) | Redistributed |

**Key improvements from Llama 2**:
- **7.5x more data** total (15T vs 2T)
- **4x more code**: Focus on documentation, debugging, code review to enhance reasoning
- **Enhanced multilingual**: Over 5% high-quality non-English vs minimal in Llama 2
- **Math-heavy**: 25% dedicated to mathematical and logical reasoning tasks
- **Aggressive filtering**: Custom classifiers and quality pipelines removed low-quality data

**Data Preparation**:
- Extensive filtering similar to Llama 1's approach to CommonCrawl
- Targeted improvements for specific domains (code, math, multilingual)
- Synthetic data generation for underrepresented areas
- Exclusion of Meta's own products/services data
- Removal of sites with high volume of personal information

### Training Infrastructure: H100 Scale-Up

**Llama 2 vs Llama 3 Hardware**:

| Aspect | Llama 2 | Llama 3 (405B) | Scale-Up |
|--------|---------|---------------|----------|
| **Primary GPUs** | A100 80GB | **H100 80GB** | Next-gen |
| **GPU Count** | 2,048 A100s | **16,384 H100s** | **8x** |
| **Clusters** | RSC + production | Production clusters | Migration |
| **Interconnect** | InfiniBand/RoCE | Advanced fabric | Improved |
| **Total GPU Hours** | 3.3M (all models) | **39.3M** (all models) | **~12x** |

**Per-Model GPU Hours**:
- **1B/3B**: Not disclosed (edge models)
- **8B**: 1.46M GPU hours
- **70B**: 7.0M GPU hours
- **405B**: 30.84M GPU hours

**Training Duration** (405B):
- **Estimated time**: ~54 days on 16,384 H100s
- **Effective training time**: >90% (despite interruptions)

**Compute Efficiency**:
- **Total FLOPs**: 3.8×10²⁵ FLOPs for 405B
- **Model FLOPs Utilization (MFU)**: 38-43% BF16 MFU
- **Average throughput**: 400 TFLOP/s per H100 GPU
- **Training precision**: Full BF16 (bfloat16)

**Parallelism Strategy**:
- **4D parallelism**: Tensor, Pipeline, Context, and Data parallelism
- Scales efficiently to 16K GPUs
- Enables training of 405B dense model

### Training Reliability Challenges

Training at this scale revealed significant infrastructure challenges:

**Interruption Statistics** (54-day snapshot during 405B training):
- **466 total interruptions** (average: one failure every 3 hours)
- **419 unexpected** interruptions (90% unplanned)

**Failure Causes**:
- **GPU hardware failures**: 148 interruptions (30.1%)
- **HBM3 memory failures**: 72 interruptions (17.2%)
- **Other causes**: ~199 interruptions (software, network, etc.)

**Despite these challenges**:
- Team achieved **>90% effective training time**
- Robust checkpointing and automated recovery systems
- ~10% compute loss to failures (significant but manageable)

This transparency about training difficulties is rare in the industry and highlights the engineering challenges of training at frontier scale.

### Post-Training: Simplified Approach vs Llama 2

**Major Methodology Change**: Llama 3 simplified the post-training pipeline compared to Llama 2's complex RLHF approach.

**Llama 2 Approach** (Complex):
1. Supervised Fine-Tuning (SFT)
2. Separate Reward Model Training
3. Reinforcement Learning with PPO (Proximal Policy Optimization)
4. Iterative refinement with rejection sampling

**Llama 3 Approach** (Simplified):
1. **Rejection Sampling**: Generate multiple completions, filter for quality
2. **Supervised Fine-Tuning (SFT)**: Adapt to instruction-following
3. **Direct Preference Optimization (DPO)**: Align to human preferences

**Why DPO instead of RLHF**:

| Aspect | Llama 2 (RLHF) | Llama 3 (DPO) | Advantage |
|--------|----------------|---------------|-----------|
| **Reward Modeling** | Separate model required | No separate model | Simpler |
| **Optimization** | RL-based (PPO) | Binary cross-entropy | More stable |
| **Complexity** | 3-4 models in memory | 1-2 models | Lower memory |
| **Debugging** | Difficult | Easier | Faster iteration |
| **Reproducibility** | Challenging | High | Better science |

**Data Volumes**:

| Stage | Llama 2 | Llama 3 | Increase |
|-------|---------|---------|----------|
| **SFT Examples** | ~27,540 | **10M+** | **~360x** |
| **Preference Pairs** | >1M | Not disclosed | Similar scale |
| **Vision Instructions** | N/A | **3M+ synthetic** | New capability |

**Industry Perspective Shift**:
- Old thinking: Scale instruction fine-tuning (SFT) as much as possible
- New thinking: SFT is starting point, preference alignment (DPO/RLHF) is critical for quality
- Focus SFT on domain-specific gaps previous models can't cover
- Use iterative improvement through simple, reproducible loops

This simplified approach enabled asynchronous workstream exploration (coding, math, reasoning) that all funnel into the same training loop, accelerating development velocity.

## Performance: Matching and Exceeding GPT-4

Llama 3 achieved Meta's ambitious goal: the first open model to match GPT-4 class performance. The family spans from competitive edge models to frontier-class capabilities.

### Overall Competitiveness

**Llama 3.1 405B**: First open model to match GPT-4 on most benchmarks
- MMLU: 87.3% (vs GPT-4o's 88.7%)
- GSM8K: 96.8% (exceeds GPT-4o's 96.1%)
- HumanEval: 89.0% (approaching Claude 3.5 Sonnet's 92.0%)

**Llama 3.3 70B**: Proves training quality matters as much as scale
- Matches Llama 3.1 405B performance at fraction of cost
- MATH: 77.0% (exceeds 405B's 73.8%!)
- Cost: ~25x cheaper than GPT-4o

**Llama 2 to Llama 3 Improvements**: Dramatic gains across all sizes
- 70B model: MMLU 63.9% → 83.6% (+20 points)
- Coding: HumanEval ~30% → 80.5% (2.7x improvement)
- Math: GSM8K ~46% → 95.1% (2x improvement)

### Comprehensive Benchmark Comparison

#### MMLU (Massive Multitask Language Understanding) - 5-shot

| Model | Score | Llama 2 Baseline |
|-------|-------|------------------|
| GPT-4o | 88.7% | - |
| Claude 3.5 Sonnet | 88.3% | - |
| **Llama 3.1 405B** | **87.3%** | - |
| GPT-4 Turbo | 86.5% | - |
| Claude 3 Opus | 86.8% | - |
| **Llama 3.3 70B** | **86.0%** (0-shot CoT) | **Llama 2 70B: 63.9% (+22.1)** |
| Gemini 1.5 Pro | 85.9% | - |
| **Llama 3.1 70B** | **83.6%** | **Llama 2 70B: 63.9% (+19.7)** |
| **Llama 3.1 8B** | **69.4%** | **Llama 2 7B: ~46% (+23.4)** |

*MMLU measures general knowledge and reasoning across 57 subjects*

#### GSM8K (Grade School Math) - 8-shot CoT

| Model | Score | Llama 2 Baseline |
|-------|-------|------------------|
| **Llama 3.1 405B** | **96.8%** | - |
| Claude 3.5 Sonnet (0-shot) | 96.4% | - |
| GPT-4o | 96.1% | - |
| **Llama 3.1 70B** | **95.1%** | **Llama 2 70B: ~46% (+49.1)** |
| **Llama 3.1 8B** | **84.5%** | **Llama 2 7B: ~17.8% (+66.7)** |

*Near-perfect performance on grade-school math, massive improvement from Llama 2*

#### HumanEval (Code Generation) - 0-shot pass@1

| Model | Score | Llama 2 Baseline |
|-------|-------|------------------|
| Claude 3.5 Sonnet | 92.0% | - |
| GPT-4o | 90.2% | - |
| **Llama 3.1 405B** | **89.0%** | - |
| **Llama 3.3 70B** | **88.4%** | **Llama 2 70B: ~29.9% (+58.5)** |
| **Llama 3.1 70B** | **80.5%** | **Llama 2 70B: ~29.9% (+50.6)** |
| **Llama 3.1 8B** | **72.6%** | **Llama 2 7B: ~15.8% (+56.8)** |

*Llama 3 closed the coding gap with proprietary models*

#### MBPP EvalPlus (Code Benchmark) - 0-shot

| Model | Score |
|-------|-------|
| **Llama 3.1 405B** | **88.6%** |
| **Llama 3.3 70B** | **87.6%** |
| **Llama 3.1 70B** | **86.0%** |
| **Llama 3.1 8B** | **72.8%** |

*Strong performance across all sizes on code generation*

#### MATH (Advanced Mathematics) - 0-shot CoT

| Model | Score |
|-------|-------|
| **Llama 3.3 70B** | **77.0%** ⭐ |
| GPT-4o | 76.6% |
| Amazon Nova Pro | 76.6% |
| **Llama 3.1 405B** | **73.8%** |
| GPT-4 Turbo | 72.6% |
| Claude 3.5 Sonnet | 71.1% |
| **Llama 3.1 70B** | **67.8%** |

*Llama 3.3 70B achieves the remarkable feat of outperforming the 405B model through better training*

#### GPQA Diamond (Graduate-Level Reasoning) - 0-shot CoT

| Model | Score |
|-------|-------|
| **Llama 3.3 70B** | **50.5%** |
| **Llama 3.1 70B** | **48.0%** |

#### MGSM (Multilingual Grade School Math) - 0-shot

| Model | Score |
|-------|-------|
| Claude 3.5 Sonnet | 92.8% |
| **Llama 3.3 70B** | **91.1%** |
| **Llama 3.1 70B** | **86.9%** |

#### IFEval (Instruction Following)

| Model | Score |
|-------|-------|
| **Llama 3.3 70B** | **92.1%** |

*Exceptional instruction-following capability*

### Per-Model-Size Performance Analysis

**1B/3B Models** (Edge devices):
- Not benchmarked on standard tests, optimized for on-device inference
- Achieve strong performance for their size through knowledge distillation
- Enable privacy-preserving applications without cloud dependency

**8B Models**:
- **MMLU**: 69.4% (competitive for size class, far exceeding Llama 2 7B's ~46%)
- **GSM8K**: 84.5% (strong math for 8B, 5x better than Llama 2 7B)
- **HumanEval**: 72.6% (excellent code generation, 4.6x better than Llama 2 7B)
- **Use case**: Efficient production deployments, real-time applications

**70B Models** (Llama 3.1):
- **MMLU**: 83.6% (approaches GPT-4 class)
- **GSM8K**: 95.1% (near-perfect grade school math)
- **HumanEval**: 80.5% (strong coding capability)
- **Use case**: High-quality general-purpose applications

**70B Models** (Llama 3.3 - Enhanced):
- **MMLU**: 86.0% (GPT-4 class)
- **MATH**: 77.0% (exceeds 405B and GPT-4o!)
- **HumanEval**: 88.4% (near 405B performance)
- **GPQA**: 50.5% (graduate-level reasoning)
- **Key achievement**: Matches 405B on many tasks at fraction of compute cost (~25x cheaper)
- **Significance**: Proves training methodology matters as much as scale

**405B Model** (Flagship):
- **MMLU**: 87.3% (GPT-4 competitive)
- **GSM8K**: 96.8% (best-in-class)
- **HumanEval**: 89.0% (top tier, approaching Claude 3.5)
- **MATH**: 73.8% (strong advanced math)
- **Overall**: "Comparable quality to leading language models such as GPT-4"
- **First open model** to achieve GPT-4 class performance

### Llama 2 to Llama 3: Generational Leap

The improvement from Llama 2 to Llama 3 represents one of the largest single-generation leaps in open-source LLM history:

| Benchmark | Llama 2 70B | Llama 3.1 70B | Improvement |
|-----------|-------------|---------------|-------------|
| **MMLU** | 63.9% | 83.6% | **+19.7 points** |
| **GSM8K** | ~46% | 95.1% | **+49.1 points** |
| **HumanEval** | ~29.9% | 80.5% | **+50.6 points** |

**Key factors enabling this leap**:
1. **7.5x more training data** (2T → 15T tokens)
2. **Larger vocabulary** (32K → 128K tokens)
3. **Better data mix** (4x more code, 25% math/reasoning)
4. **GQA at all scales** (efficiency without quality loss)
5. **Improved post-training** (DPO instead of complex RLHF)
6. **Long context capability** (4K → 128K tokens)

### Vision Model Performance (Llama 3.2)

While specific benchmark scores for the 11B and 90B vision models weren't detailed in comprehensive tables, they demonstrated strong capabilities:

**Capabilities**:
- Visual recognition and reasoning
- Image captioning
- Visual question answering
- Document understanding with charts/graphs
- Maintained all text-only capabilities of base models (drop-in replacement)

**Training**:
- 6B image-text pairs for adapter training
- 3M+ synthetic instruction examples for post-training
- Competitive with proprietary vision models on recognition tasks

### Strengths and Weaknesses

**Strengths**:
- **GPT-4 class performance**: 405B matches or exceeds GPT-4 on most benchmarks
- **Efficiency**: 70B models competitive with much larger proprietary models
- **Code generation**: Massive improvement from Llama 2, now top-tier
- **Math and reasoning**: Near-perfect on GSM8K, strong on advanced MATH
- **Multilingual**: 128K vocab enables better non-English performance
- **Cost**: Open source provides unlimited usage without API costs

**Weaknesses**:
- **Still trails Claude 3.5 Sonnet** on some coding tasks (HumanEval)
- **Knowledge cutoff**: September 2022 (older than some proprietary models)
- **Resource requirements**: 405B requires significant infrastructure for deployment
- **Vision models**: Still under development, not as mature as GPT-4V

### The Llama 3.3 70B Revelation

The December 2024 release of Llama 3.3 70B delivered a critical insight: **a 70B model trained better can match a 405B model's performance**. This has profound implications:

- **Cost efficiency**: ~25x cheaper than GPT-4o ($0.10/M vs $2.50/M input tokens)
- **Inference speed**: 276 tokens/sec on Groq (faster than 405B)
- **Training matters**: Same data (15T tokens), better methodology (improved post-training)
- **Industry impact**: Proves scaling laws have limits; training quality is crucial

This solidified the lesson that **how you train matters as much as how big your model is**.

## Key Innovations: Building on Llama 2

The Llama 3 family introduced several critical innovations while building on the solid foundation established by Llama 2.

### 1. Context Window Scaling (4K → 8K → 128K)

**The Challenge**: Extending context without full retraining

**Llama 2 Baseline**: 4,096 tokens
**Llama 3 Achievement**: 8,192 tokens (2x)
**Llama 3.1 Achievement**: 128,000 tokens (32x from Llama 2)

**How it Works** (YaRN-based RoPE Scaling):
1. Train base model at 8K context with standard RoPE (θ = 500,000)
2. Apply frequency scaling factors to RoPE embeddings
3. Continue pre-training on **~800B additional tokens** (only ~5% of original training)
4. Fine-tune with scale factors (s=16, s=32) for different context lengths
5. Validate on long-context benchmarks

**Efficiency**: Achieves 16x context extension using only 0.1% of pre-training compute—a breakthrough in compute efficiency.

**Impact**:
- Process entire books (100K+ tokens)
- Full codebase analysis in single context
- Extended multi-turn conversations
- Multi-document reasoning and synthesis

**Comparison to Llama 2**: This was impossible in Llama 2's architecture without full retraining. Llama 3 proved context can scale dramatically through continued pre-training.

### 2. GQA Extended to All Model Sizes

**Llama 2 Limitation**: GQA only on 70B model; 7B and 13B used standard MHA

**Llama 3 Innovation**: Standardized on GQA across **all sizes** with 8 KV heads

| Model | Query Heads | KV Heads | Ratio | Memory Savings vs MHA | Llama 2 |
|-------|-------------|----------|-------|---------------------|---------|
| **8B** | 32 | 8 | 4:1 | 75% KV cache reduction | MHA (no savings) |
| **70B** | 64 | 8 | 8:1 | 87.5% KV cache reduction | GQA (same) |
| **405B** | 128 | 8 | 16:1 | 93.75% KV cache reduction | N/A (new size) |

**Benefits Over Llama 2**:
- **Faster inference**: Lower memory bandwidth requirements at all scales
- **Larger batch sizes**: More efficient use of GPU memory
- **Long context enablement**: Critical for scaling to 128K tokens
- **Quality preservation**: Minimal performance loss vs full MHA

**Validation**: Extending GQA to 8B proved efficiency gains work at all scales, not just large models.

### 3. Tokenizer Revolution (SentencePiece → TikToken)

**Llama 2**: SentencePiece BPE, 32,000 tokens
**Llama 3**: TikToken-based BPE, 128,256 tokens (4x expansion)

**Implementation**:
- Started with 100K tokens from OpenAI's tiktoken library
- Trained additional 28K tokens for multilingual support
- Added 256 special tokens for structured conversations

**Benefits vs Llama 2**:

| Aspect | Llama 2 | Llama 3 | Improvement |
|--------|---------|---------|-------------|
| **Multilingual efficiency** | Poor | Excellent | Better encoding for 30+ languages |
| **Code efficiency** | Adequate | Excellent | Improved syntax representation |
| **Compression** | Baseline | ~25% fewer tokens | Same text, fewer tokens |
| **Compatibility** | Isolated | Shared with GPT-4 | Ecosystem benefits |

**Special Tokens for Conversations**:
- `<|begin_of_text|>`, `<|end_of_text|>`: Sequence boundaries
- `<|eot_id|>`: End of turn in conversations (Llama 2 lacked this)
- `<|start_header_id|>`, `<|end_header_id|>`: Message headers

**Impact**: The larger vocabulary was crucial for Llama 3's multilingual and code capabilities, directly enabling better performance on those benchmarks.

### 4. Vision Integration Without Quality Loss

**New Capability**: Llama 2 was text-only; Llama 3.2 added vision

**Architectural Innovation** (Adapter-based approach):
- Vision encoder (ViT-H/14): 630M → 850M params with gated attention
- Cross-attention layers inserted every 5 layers in frozen language model
- Trained on 6B image-text pairs + 3M synthetic instructions
- **Language model frozen** during vision training

**Why This Matters**:
- Preserves text capabilities completely (acts as drop-in replacement)
- Avoids costly joint training from scratch
- Modular architecture allows independent optimization
- Faster training than multimodal-from-scratch approaches

**Comparison to Alternatives**:
- GPT-4V, Gemini: Joint training (expensive, can degrade text performance)
- LLaVA: Similar adapter approach (validated by Meta's success)
- Llama 3.2: Best of both worlds—vision capability + preserved text quality

### 5. Edge Optimization Through Knowledge Distillation

**New Capability**: Llama 2 smallest model was 7B; Llama 3.2 added 1B and 3B

**Training Technique**:
- **Logits from larger models** (8B, 70B) used as token-level targets
- Pretrained on 9T tokens (vs 15T for larger models)
- Maintains full architectural consistency (GQA, RoPE, SwiGLU, RMSNorm)

**Optimizations**:
- Efficient inference on mobile/IoT devices
- Lower precision support
- Reduced model size without major quality loss

**Use Cases Unlocked**:
- On-device AI (smartphones, tablets)
- Privacy-preserving inference (no cloud needed)
- IoT and edge computing
- Real-time applications with low latency

**Comparison to Llama 2**: Llama 2's smallest model (7B) was too large for edge deployment. Llama 3.2 democratized access to powerful AI on resource-constrained devices.

### 6. Simplified Post-Training (RLHF → DPO)

**Llama 2 Complexity**: Multi-stage RLHF with separate reward models and PPO

**Llama 3 Simplification**: Three-stage loop with Direct Preference Optimization

| Aspect | Llama 2 (RLHF) | Llama 3 (DPO) | Impact |
|--------|----------------|---------------|--------|
| **Reward model** | Separate model required | No separate model | 50% memory reduction |
| **Optimization** | RL-based (PPO, unstable) | Binary cross-entropy (stable) | Faster convergence |
| **Debugging** | Opaque, difficult | Transparent, straightforward | 3x faster iteration |
| **Reproducibility** | Challenging | Excellent | Better science |

**Industry Impact**: Llama 3's DPO approach influenced the broader industry to move away from complex RLHF pipelines toward simpler, more reproducible methods.

**SFT Data Scale-Up**: Llama 2 used ~27K examples; Llama 3 used **10M+** examples (~360x increase), enabled by the simplified pipeline.

## Legacy and Impact

The Llama 3 family's release marked a watershed moment in AI history, proving definitively that open-source models could match the best proprietary systems.

### Community Adoption: Unprecedented Scale

**Download Statistics**:
- **350 million+ downloads** as of August 2024
- **10x year-over-year growth** from Llama 2
- **650 million+ total** (Llama family + derivatives) by late 2024
- **Llama 3.1 alone**: 20M+ downloads in first month

**Enterprise Adoption** (Production deployments):
- **AT&T**: 33% improvement in search-related customer service responses
- **Goldman Sachs**: Internal AI tooling for financial analysis
- **DoorDash**: Logistics optimization and routing
- **Accenture**: ESG reporting (expected 70% productivity gains)
- **Spotify**: Recommendation systems and content understanding
- Hundreds more Fortune 500 companies

### Derivative Models: 60,000+ and Counting

**Scale**: Over 60,000 derivative models on Hugging Face and other platforms

**Notable Examples**:
- **FinGPT**: Financial domain specialization
- **BioBERT**: Biomedical text mining and research
- **Defog SQLCoder**: Text-to-SQL generation for databases
- **Phind**: Developer-focused code search and generation
- **Hermes models**: Enhanced instruction-following capabilities
- **Countless fine-tunes**: Domain-specific (legal, medical), language-specific, task-specific

**Why Derivatives Matter**:
- Demonstrate open source enables rapid specialization
- Lower training costs vs training from scratch (100x-1000x savings)
- Community-driven innovation faster than any single company
- Democratizes access to specialized AI capabilities

### Industry Impact: Proving Open Can Match Closed

**Paradigm Shift**:
- **Before Llama 3**: Open models seen as inferior, 1-2 years behind
- **After Llama 3.1 405B**: Open models competitive with best closed models
- **After Llama 3.3 70B**: Open models can exceed larger models through better training

**Market Dynamics**:
- **Commoditization**: Meta's strategy reduces competitors' pricing power
- **Price pressure**: Forced GPT-4 and Claude to lower prices or add value
- **Innovation acceleration**: Open community drives faster iteration than closed labs
- **Healthier ecosystem**: Multiple competitive options prevent vendor lock-in

**Technical Achievements Validated**:
1. **Scaling laws work to 405B** parameters (and beyond)
2. **Context windows can scale dramatically** (32x expansion proven feasible)
3. **GQA works at all scales** (1B → 405B)
4. **Knowledge distillation** enables powerful small models (1B, 3B)
5. **Multimodal integration** possible without degrading text performance
6. **Training quality matters as much as scale** (Llama 3.3 70B lesson)

### Philosophical Impact: Open Source as Competitive Advantage

**Meta's Thesis** (validated):
- "Openness leads to better, safer products"
- Faster innovation through community collaboration
- Wider distribution of AI benefits
- Transparency enables trust and safety research

**Democratization**:
- State-of-the-art AI accessible to researchers, startups, individuals
- Training powerful models at "significantly lower cost" than closed alternatives
- On-device AI (1B/3B) enables privacy-preserving applications
- Reduces dependency on proprietary APIs and vendor lock-in

**Industry Perspective**:
- "Without open source community, generative AI would be much less advanced, very niche"
- Open source provides "ongoing fuel for development and advancement"
- Forces entire industry to move faster and be more accessible
- Establishes open development as viable path to AI leadership

### Cultural Impact: Legitimizing Open-Source AI

**Before Llama 3**:
- Open models considered toys, research projects
- Serious applications required proprietary APIs
- Open source AI was 1-2 years behind state-of-the-art

**After Llama 3**:
- Open models competitive with (and sometimes exceeding) best closed models
- Major enterprises deploy open models in production
- Llama 3 family "solidified Meta's position as leader in open-source AI"
- Open development now seen as viable path to AI leadership

**Standards Set**:
- Comprehensive model cards and documentation
- Responsible AI guidelines (Llama Guard 3)
- Clear licensing (Llama 3 Community License)
- Benchmark transparency and reproducibility
- Training infrastructure details (including failures)

**Long-term Impact**:
- Established open source as default expectation for foundation models
- Forced other companies (Mistral, Qwen, DeepSeek) to compete on openness
- Created expectation of transparency in model development
- Demonstrated that collaborative development can outpace closed labs

### The Llama 3.3 70B Lesson

The December 2024 release delivered a profound insight that will shape future AI development:

**Core Lesson**: **Training methodology matters as much as scale**

- Same data (15T tokens), same architecture, different post-training → **matches 405B**
- Proves that **smaller models trained better** can match larger models
- Challenges assumption that "bigger is always better"
- Shifts focus from pure scaling to **training quality and efficiency**

**Industry Implications**:
- Future models may focus on training improvements over parameter growth
- Cost efficiency becomes competitive advantage (~25x cheaper than GPT-4o)
- Inference speed matters (276 tokens/sec vs slower larger models)
- **ROI from better training** >> ROI from bigger models

This lesson will influence AI development for years to come, potentially marking the end of the pure scaling era.

## Key Figures

The Llama 3 family was the result of massive collaborative effort across Meta AI, representing one of the largest AI research projects in history.

### Leadership

**Mark Zuckerberg** - Meta CEO
- Strategic vision for open-source AI as competitive advantage
- Commitment to 350K+ H100 GPUs by end of 2024
- Public advocate for Llama's open approach: "Openness leads to better products"
- Championed commoditization strategy to reduce competitors' pricing power

**Yann LeCun** - Chief AI Scientist (2013-2025)
- One of the "godfathers of modern AI" (Turing Award winner)
- Founding director of FAIR (Fundamental AI Research)
- Long-standing advocate for open-source AI and transparency
- Announced plans to leave Meta for own startup (as of early 2025)

**Joelle Pineau** - VP of Research / Head of FAIR
- Co-leads AI research with LeCun, reporting to Chris Cox
- Oversees strategic direction of research initiatives
- Guided development and release of Llama family

**Ahmad Al-Dahle** - Head of Generative AI Team
- Former Apple executive, established Generative AI team in February 2024
- Leads Llama product development and deployment
- Bridges research and production engineering

**Chris Cox** - Chief Product Officer
- AI research leadership reports to him
- Strategic oversight of AI initiatives across Meta

**Shengjia Zhao** - Chief Scientist, Meta Superintelligence Labs
- Co-creator of ChatGPT (hired from OpenAI)
- Leading next-generation AI research beyond Llama 3

### Llama 3 Development Team

**Paper Attribution**: "Llama Team, AI @ Meta"
- **Lead Author**: Aaron Grattafiori
- **Co-authors**: **558+ researchers and engineers**
- Reflects the massive scale of collaboration (vs ~65 for Llama 2, ~20 for Llama 1)

**Notable Team Evolution**:
- More than half of original Llama 1 research paper authors left Meta within months
- Many joined the new Generative AI team, including:
  - Hugo Touvron (Llama 1 & 2 lead author)
  - Thibaut Lavril
  - Xavier Martinet
  - Marie-Anne Lachaux
  - Naman Goyal
  - Aurelien Rodriguez

### Organizational Context

**February 2024 Reorganization**:
- Created dedicated "Generative AI" team (production focus)
- Separated from FAIR (fundamental research)
- Accelerated development velocity through parallel workstreams

**Infrastructure Scale**:
- Managed 16,384 H100 GPUs for 405B training
- Built fault-tolerance for 466 interruptions over 54 days
- Achieved >90% uptime despite failures every 3 hours
- Scaled to 350K+ GPU equivalent by end of 2024

The Llama 3 family represents the culmination of Meta's multi-year investment in open-source AI, proving that collaborative development with the global community can match or exceed the capabilities of closed, proprietary systems.
