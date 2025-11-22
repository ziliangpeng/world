# Meta Llama Series

The Llama series from Meta represents one of the most influential open-source LLM families, setting standards for decoder-only transformer architectures.

## Llama 1 (February 2023)

### Model Variants
- **7B**: 7 billion parameters
- **13B**: 13 billion parameters
- **33B**: 33 billion parameters
- **65B**: 65 billion parameters

### Architecture

**Base Design**: Auto-regressive decoder-only transformer

**Key Components**:
- **Normalization**: RMSNorm pre-normalization (instead of post-normalization)
- **Activation**: SwiGLU activation function (from PaLM)
- **Position Encoding**: Rotary Embeddings (RoPE, not absolute positional embeddings)
- **Attention**: Multi-Head Attention (MHA)
- **FFN Dimension**: 2/3 × 4d instead of 4d (as in PaLM)

### Training Details
- **Tokens**:
  - 65B & 33B: 1.4 trillion tokens
  - 7B: 1 trillion tokens
- **Context Window**: 2,048 tokens
- **Vocabulary**: 32K tokens (SentencePiece tokenizer)
- **Training Data**: Publicly available datasets only
  - English CommonCrawl, C4
  - GitHub, Wikipedia
  - Gutenberg and Books3
  - ArXiv, Stack Exchange

### Performance
- LLaMA-13B outperformed GPT-3 175B on most benchmarks
- LLaMA-65B competitive with Chinchilla-70B and PaLM-540B

### Significance
- First major open-source model from Meta
- Proved open models could compete with proprietary ones
- Established architectural patterns (RMSNorm, SwiGLU, RoPE)
- Sparked explosion of derivative models

### Links
- **Paper**: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- **Meta AI Research**: [LLaMA Publication](https://ai.meta.com/research/publications/llama-open-and-efficient-foundation-language-models/)
- **Hugging Face**: Not officially released by Meta (research-only release requiring application)
  - Community conversions available: [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf), [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)
  - Note: Community versions are format conversions (PyTorch .pth → HuggingFace Transformers format) of the same official weights, not different models. Early conversions like decapoda-research may be outdated.

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

### Links
- **Paper**: [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- **Blog**: [Meta and Microsoft Introduce the Next Generation of Llama](https://ai.meta.com/blog/llama-2/)
- **Hugging Face**: [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b), [Llama-2-13b](https://huggingface.co/meta-llama/Llama-2-13b), [Llama-2-70b](https://huggingface.co/meta-llama/Llama-2-70b)

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

### Links
- **Paper**: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- **Blog**: [Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/)
- **Hugging Face**: [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B), [Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)

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

### Links
- **Paper**: [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)
- **Blog**: [Introducing Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/)
- **Hugging Face**: [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)

## Llama 3.2 (September 2024)

### Model Variants

**Text-Only Models**:
- **1B**: 1 billion parameters
- **3B**: 3 billion parameters

**Vision Models**:
- **11B Vision**: 11 billion parameters (built on Llama 3.1 8B)
- **90B Vision**: 90 billion parameters (built on Llama 3.1 70B)

### Architecture

**Text-Only Models**:
- Same foundation as Llama 3/3.1 (GQA, RoPE, SwiGLU, RMSNorm)
- Optimized for edge deployment and on-device inference
- Maintains architectural consistency with larger siblings

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

## Llama 3.3 (Late 2024)

### Updates
- Latest iteration maintaining architectural consistency
- Further optimizations for efficiency and performance
- Continued refinement of training data and processes

### Links
- **Blog**: [The future of AI: Built with Llama](https://ai.meta.com/blog/future-of-ai-built-with-llama/)
- **Hugging Face**: [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)

## Llama 4 (April 2025)

### Model Variants

**Scout**:
- 17B active parameters (109B total parameters)
- 16 experts (MoE architecture)
- 10 million token context window

**Maverick**:
- 17B active parameters (400B total parameters)
- 128 experts (MoE architecture)
- 1 million token context window
- Natively multimodal (text, images, video)

**Behemoth** (Announced, not yet released):
- 288B active parameters (~2T total parameters)
- 16 experts
- Still in training

### Architecture

**First Llama with MoE**: Llama 4 is Meta's first model family built using Mixture-of-Experts (MoE) architecture

**Natively Multimodal**:
- Analyzes and understands text, images, and video
- Built multimodal from the ground up (not adapted like Llama 3.2 Vision)

**Unprecedented Context**:
- Scout: 10 million token context window
- Largest context window in the Llama family

### Training Details

- **Tokens**: 30+ trillion tokens (2x Llama 3's training data)
- **Compute**: Trained on cluster with 100,000+ H100 GPUs
- **Multimodal Training**: Native multimodal training from scratch

### Innovations

- First open-weight natively multimodal models with MoE
- Massive context length (10M tokens)
- Advanced reasoning and speech capabilities
- Doubled training data compared to Llama 3

### Links
- **Blog**: [The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- **Announcement**: [Meta Launches Llama 4 Models](https://www.socialmediatoday.com/news/meta-releases-llama-4-ai-models/744560/)

---

## Code Llama (August 2023)

### Model Sizes and Variants

**Base Models**: 7B, 13B, 34B, 70B parameters

**Three Variants** (all available in all sizes):
1. **Code Llama (Base)**: Foundation for general code tasks
2. **Code Llama - Python**: Python-specialized versions
3. **Code Llama - Instruct**: Instruction-following for code tasks

### Architecture Modifications

**Base**: Built on Llama 2, initialized with pretrained weights

**Extended Context**:
- Context expanded from 4,096 to **100,000 tokens**
- Modified RoPE parameters for long sequences
- Trained on 16K token sequences
- Strong extrapolation up to 100K tokens

**Fill-in-the-Middle (FIM)**:
- Supported: 7B, 13B, 70B (base and instruct)
- NOT supported: 34B models, Python variants
- Enables code completion and insertion
- Uses causal infilling alongside autoregressive prediction

### Training Approach

**Multi-Stage Specialization**:
1. Initialize with Llama 2 pretrained weights (already saw 80B code tokens)
2. Train on 500B tokens of code (1T for 70B model)
3. Long context fine-tuning (separate stage)
4. Instruction fine-tuning (for Instruct variants)

**Training Data** (500B tokens, 1T for 70B):
- **85%**: Open-source GitHub code
- **8%**: Natural language about code
- **7%**: General natural language

**Multitask Objective**: 7B, 13B, 70B use both autoregressive and causal infilling

### Key Innovations
- First major open-source code model from Meta
- Successfully extended context to 100K tokens with RoPE modifications
- Multi-stage specialization: pretrain → code → long context → instruct
- Fill-in-the-middle for real-time editor integration
- Cost-efficient long-context fine-tuning as separate stage

### Use Cases
- Code completion in editors
- Code generation from natural language
- Docstring generation
- Code explanation and debugging
- Code translation between languages

### Links
- **Paper**: [Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950)
- **Blog**: [Introducing Code Llama](https://ai.meta.com/blog/code-llama-large-language-model-coding/)
- **Hugging Face**: [CodeLlama-7b-hf](https://huggingface.co/meta-llama/CodeLlama-7b-hf), [CodeLlama-13b-hf](https://huggingface.co/meta-llama/CodeLlama-13b-hf), [CodeLlama-34b-hf](https://huggingface.co/meta-llama/CodeLlama-34b-hf), [CodeLlama-70b-hf](https://huggingface.co/meta-llama/CodeLlama-70b-hf)

---

## Llama Guard (Safety & Moderation)

### Purpose
LLM-based input-output safeguard for human-AI conversations, providing content moderation and safety classification.

### Model Evolution

**Llama Guard 1** (December 2023):
- **7B parameters** (based on Llama 2-7B)
- Initial safety classification model

**Llama Guard 2**:
- Expanded to 11 safety categories
- Improved taxonomy

**Llama Guard 3** (Current):
- **8B parameters** (based on Llama 3.1-8B)
- Supports 8 languages
- MLCommons-aligned taxonomy (13 categories)
- INT8 quantized version available

**Llama Guard 3-1B**:
- **1B parameters** for resource-constrained environments
- **1B-INT4**: 440MB compressed (7x smaller)
- F1 score: 0.904 (outperforms uncompressed 1B)

**Llama Guard 4**:
- **12B parameters** (based on Llama 4)
- Multimodal input/output moderation
- Supports multiple images in prompts
- 14 hazard categories (MLCommons taxonomy) + code interpreter abuse

### Safety Taxonomy

**Llama Guard 3** (13 Categories - MLCommons aligned):
- Violent Crimes
- Non-Violent Crimes
- Sex-Related Crimes
- Child Sexual Exploitation
- Defamation
- Specialized Advice
- Privacy violations
- Intellectual Property
- Indiscriminate Weapons
- Hate
- Suicide & Self-Harm
- Sexual Content
- Elections (manipulation)

### Capabilities
- Classifies prompts and responses as safe/unsafe
- Lists violated categories when unsafe
- Supports both input (prompt) and output (response) classification
- Flexible, customizable taxonomy
- Multilingual support (8 languages in Guard 3)
- Optimized for search and code interpreter tool calls

### Links
- **Paper (Guard 1)**: [Llama Guard: LLM-based Input-Output Safeguard](https://arxiv.org/abs/2312.06674)
- **Paper (Guard 3-1B)**: [Llama Guard 3-1B-INT4: Compact and Efficient Safeguard](https://arxiv.org/abs/2411.17713)
- **Hugging Face**: [LlamaGuard-7b](https://huggingface.co/meta-llama/LlamaGuard-7b), [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B), [Llama-Guard-3-1B](https://huggingface.co/meta-llama/Llama-Guard-3-1B), [Llama-Guard-4-12B](https://huggingface.co/meta-llama/Llama-Guard-4-12B)
- **Documentation**: [Llama Guard 4 Model Card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/)

---

## Purple Llama (Trust & Safety Tools)

### What It Is

**NOT a Model**: Purple Llama is an umbrella project for open trust and safety tools and evaluations for responsible generative AI development.

**Name Origin**: "Purple" from cybersecurity (Red team + Blue team = Purple team) + "Llama"

### Main Components

**1. CyberSecEval** - Cybersecurity Benchmarks

**Purpose**: Comprehensive evaluation of LLM cybersecurity risks

**Evaluation Domains**:
- Propensity to generate insecure code
- Compliance when asked to assist in cyberattacks

**Evolution**:
- CyberSecEval 1 (December 2023)
- CyberSecEval 2, 3 (2024)
- CyberSecEval 4 (Latest): Adds CyberSOCEval + AutoPatchBench

**Tools**:
- **Insecure Code Detector (ICD)**: 189 static analysis rules, 50 insecure practices
- **MITRE Tests**: MITRE ATT&CK framework compliance evaluation

**2. Llama Guard**:
- Input/output safeguards (see Llama Guard section above)
- Pretrained safety classification

**3. Prompt Guard**:
- Protection against malicious prompts
- Prevents prompt injection attacks
- Application security
- **Prompt Guard 2**: 86M parameters (high performance) and 22M parameters (low latency)

**4. Code Shield**:
- Inference-time filtering of insecure code
- Prevents code interpreter abuse
- Secure command execution
- Supports 7 programming languages
- Average latency: 200ms

**5. LlamaFirewall**:
- Integration suite for Llama Protections
- Combines Llama Guard, Prompt Guard, and Code Shield
- Unified protection layer

### Licensing and Collaboration

**License**: Permissively licensed for research and commercial use

**Partners**: AI Alliance, AMD, AWS, Google Cloud, Hugging Face, IBM, Intel, Microsoft, NVIDIA, MLCommons, and many others

**Repository**: https://github.com/meta-llama/PurpleLlama

### Relationship to Llama Models

Provides tools to assess and improve security of Llama and other LLMs:
- Evaluate cybersecurity risks
- Prevent unsafe outputs
- Filter insecure code
- Protect against attacks

### Links
- **Repository**: [PurpleLlama GitHub](https://github.com/meta-llama/PurpleLlama)
- **Blog**: [Purple Llama announcement](https://ai.meta.com/blog/purple-llama-open-trust-safety-generative-ai/)

---

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
