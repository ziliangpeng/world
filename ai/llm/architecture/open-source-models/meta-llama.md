# Meta Llama Series

The Llama series from Meta represents one of the most influential open-source LLM families, setting standards for decoder-only transformer architectures and democratizing access to state-of-the-art AI.

## Origin Story: How Llama Came to Be

### The Catalyst: ChatGPT Disruption (November 2022)

When OpenAI launched ChatGPT in late November 2022, Meta was caught off guard. The company was perceived as having fallen far behind OpenAI, Anthropic, and Google in generative AI. This became the catalyst for what would become Meta's most important AI project.

### The Response: Efficiency Over Scale (2022-2023)

Meta's **FAIR** (Fundamental AI Research) team, led by **Joelle Pineau** (VP of AI Research, head of FAIR), developed a different approach from competitors:

**The Philosophy:**
- While competitors focused on massive models (GPT-3: 175B, PaLM: 540B), LLaMA focused on **parameter efficiency**
- Train models of various sizes (7B-65B) on **more data** (1.4T tokens)
- Use **only publicly available data** for reproducibility
- Make it accessible to researchers with **different hardware capabilities**
- **Result**: LLaMA-13B matched GPT-3 175B on many benchmarks

**Initial Release (February 24, 2023):**
- Paper: "LLaMA: Open and Efficient Foundation Language Models" by Hugo Touvron + 13 authors
- Access: Restricted to approved researchers via application (case-by-case basis)
- Goal: Research utility, not commercial release
- Inference code: Released as open-source (GPLv3)

### The Plot Twist: The 4chan Leak (March 3, 2023)

Just days after the restricted release, someone posted a **BitTorrent link** to the entire LLaMA package on **4chan**. This was historic: the first time a major tech firm's proprietary AI model leaked publicly.

**Meta's Response:**
- Filed takedown requests (but torrents spread anyway)
- Stood by their democratization approach
- Didn't punish the research community

**Impact:** The leak inadvertently achieved Meta's goal of democratization, giving developers and researchers unprecedented access to cutting-edge AI. What was meant to be a controlled research release became an open-source movement by accident.

### The Open-Source Pivot: Llama 2 (July 18, 2023)

Rather than fight the democratization, Meta embraced it fully.

**The Decision:**
- **Yann LeCun** (Chief AI Scientist) and **Joelle Pineau** pushed for wide open-source release
- **Mark Zuckerberg** made the final call to release Llama 2 as fully open-source
- Partnership with **Microsoft** for distribution

**Llama 2 Changes:**
- Truly open-source with commercial license
- No application needed
- Available to everyone
- Turned potential PR disaster into strategic advantage

### The Transformation

Since then, Meta has fully rebuilt around Llama:
- From afterthought to core company strategy
- Rapid iteration: Llama 3, 3.1, 3.2, 3.3, and Llama 4
- Zuckerberg committed to making Meta the leader in open-source AI
- Became the most influential open-source LLM family

### Key Figures

**Yann LeCun** - Meta's Chief AI Scientist
- Played "very indirect" role in Llama 1
- Pushed for Llama 2 open-source release

**Joelle Pineau** - VP of AI Research, Head of FAIR
- Led development of original Llama and Llama 2
- Drove the efficiency-focused approach

**Mark Zuckerberg** - Meta CEO
- Made final decision to go fully open-source with Llama 2
- Committed Meta's strategic direction to open AI

---

## The Llama Family: Overview

### Main Language Models

**[Llama 1](meta-llama-1.md)** (February 2023) - The Original
- 7B, 13B, 33B, 65B parameters
- Research-only, then leaked
- Proved efficiency matters more than size
- [Read more →](meta-llama-1.md)

**[Llama 2](meta-llama-2.md)** (July 2023) - The Open Release
- 7B, 13B, 70B parameters
- First truly open-source with commercial license
- Introduced Chat variants (SFT + RLHF)
- 70B pioneered GQA in production
- [Read more →](meta-llama-2.md)

**[Llama 3 Family](meta-llama-3.md)** (2024) - State of the Art
- **Llama 3** (April): 8B, 70B - GQA everywhere, 128K vocab, 15T tokens
- **Llama 3.1** (July): + 405B flagship, 128K context
- **Llama 3.2** (September): 1B, 3B edge + 11B, 90B Vision
- **Llama 3.3** (December): 70B efficiency improvements
- [Read more →](meta-llama-3.md)

**[Llama 4](meta-llama-4.md)** (April 2025) - Multimodal MoE
- Scout (17B active, 109B total, 10M context)
- Maverick (17B active, 400B total, 1M context)
- Behemoth (288B active, ~2T total, announced)
- First Llama with MoE architecture
- Natively multimodal (text, images, video)
- [Read more →](meta-llama-4.md)

### Specialized Models

**[Code Llama](meta-llama-code.md)** (August 2023)
- 7B, 13B, 34B, 70B in Base/Python/Instruct variants
- 100K token context for entire repositories
- Fill-in-the-Middle for IDE integration
- [Read more →](meta-llama-code.md)

**[Safety & Security Tools](meta-llama-misc.md)**
- **Llama Guard** (1B-12B): Content moderation, multimodal
- **Prompt Guard** (22M-86M): Injection attack prevention
- **Code Shield**: Insecure code filtering
- **Purple Llama**: Cybersecurity evaluation tools
- **LlamaFirewall**: Unified protection layer
- [Read more →](meta-llama-misc.md)

---

## Model Variants Explained

### Base vs Chat/Instruct Models

**Base (Pretrained) Models:**
- Trained on massive unlabeled text datasets during pretraining
- Predict next tokens, complete text
- NOT optimized for conversation or instruction-following
- Foundation for further fine-tuning
- Examples: `Llama-2-7b`, `Meta-Llama-3-8B`, `Llama-3.1-70B`

**Chat/Instruct (Fine-tuned) Models:**
- Fine-tuned from base models using advanced techniques:
  - **SFT (Supervised Fine-Tuning)**: Trained on instruction-following datasets
  - **RLHF (Reinforcement Learning from Human Feedback)**: Aligned with human preferences
- Optimized for dialogue and instruction-following
- Better at following instructions, safer, more helpful responses
- Examples: `Llama-2-7b-chat`, `Meta-Llama-3-8B-Instruct`, `Llama-3.1-70B-Instruct`

**Relationship:**
- Chat/Instruct models are **NOT different models** - they are fine-tuned versions of Base models
- Base → (+SFT + RLHF) → Chat/Instruct
- Same architecture, different training objectives

### HuggingFace Naming Convention: "-hf" Suffix

**Without "-hf"** (e.g., `Llama-2-7b-chat`):
- Meta's original PyTorch checkpoint format (.pth files)
- Requires conversion to use with HuggingFace Transformers

**With "-hf"** (e.g., `Llama-2-7b-chat-hf`):
- HuggingFace Transformers-compatible format
- Uses `pytorch_model.bin` or `model.safetensors`
- Works natively with `transformers` library
- **Same weights** as non-hf version, just different file format

**Example Variants for Llama 2 13B:**
- `Llama-2-13b` - Base model (Meta format)
- `Llama-2-13b-hf` - Base model (HuggingFace format)
- `Llama-2-13b-chat` - Chat model (Meta format)
- `Llama-2-13b-chat-hf` - Chat model (HuggingFace format)

### Training Pipeline

```
Pretraining (Base Model)
  ↓
Supervised Fine-Tuning (SFT)
  ↓
Reinforcement Learning from Human Feedback (RLHF)
  ↓
Chat/Instruct Model
```

**SFT:** Trained on 25M+ instruction-following examples (Llama 3.1)
**RLHF:** Uses PPO (Proximal Policy Optimization) with reward model based on human preferences

---

## Evolution Timeline

### Chronological Release History

```
2023:
├─ Feb 24: Llama 1 (7B-65B) - Research-only
├─ Mar 3:  4chan Leak - Accidental democratization
├─ Jul 18: Llama 2 (7B-70B) - Fully open-source
└─ Aug 24: Code Llama (7B-70B) - Code specialist

2024:
├─ Apr 18: Llama 3 (8B, 70B) - GQA everywhere, 128K vocab
├─ Jul 23: Llama 3.1 (+ 405B, 128K context) - GPT-4 class
├─ Sep 25: Llama 3.2 (1B-3B edge, 11B-90B Vision) - Multimodal
└─ Dec:    Llama 3.3 (70B) - Efficiency improvements

2025:
└─ Apr 5:  Llama 4 (Scout, Maverick, Behemoth) - MoE + Multimodal
```

### Evolution Summary Table

| Version | Release | Sizes | Context | Vocab | Key Innovation |
|---------|---------|-------|---------|-------|----------------|
| **Llama 1** | Feb 2023 | 7-65B | 2K | 32K | Efficiency > Size |
| **Llama 2** | Jul 2023 | 7-70B | 4K | 32K | Open + GQA (70B) |
| **Llama 3** | Apr 2024 | 8-70B | 8K | 128K | GQA everywhere, TikToken |
| **Llama 3.1** | Jul 2024 | 8-405B | 128K | 128K | Long context, 405B |
| **Llama 3.2** | Sep 2024 | 1B-90B | Varies | 128K | Edge + Vision |
| **Llama 3.3** | Dec 2024 | 70B | 128K | 128K | Efficiency |
| **Llama 4** | Apr 2025 | 17B-288B active | 1M-10M | 128K | MoE + Native multimodal |

### Key Milestones

- **2K → 10M tokens**: Context expanded 5000x over 2 years
- **7B → 405B**: Dense models scaled 60x
- **400B total (17B active)**: First MoE Llama (Llama 4)
- **32K → 128K vocab**: 4x vocabulary expansion (Llama 3)
- **Text-only → Multimodal**: Native vision + text (Llama 4)

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

### Architectural Evolution

- **Llama 1**: MHA, RMSNorm, SwiGLU, RoPE (foundation)
- **Llama 2**: + GQA for 70B (efficiency)
- **Llama 3**: GQA for all sizes, TikToken tokenizer (universality)
- **Llama 3.1**: RoPE scaling for long context (128K)
- **Llama 3.2**: Vision adapters, edge optimization (specialization)
- **Llama 4**: MoE, native multimodal (next generation)

---

## Impact on the Field

The Llama series has been transformative for open-source AI:

### 1. Democratization
- Made state-of-the-art models accessible to researchers and developers
- No expensive API costs
- Run locally on consumer hardware (smaller models)
- Enabled experimentation without corporate gatekeeping

### 2. Architectural Standards
**RMSNorm + SwiGLU + RoPE + GQA** became the standard stack:
- Nearly all modern open models use this combination
- Validated by extensive real-world use
- Efficient and effective at all scales

### 3. Fine-tuning Ecosystem
Enabled countless specialized models:
- **Instruction-tuned**: Alpaca, Vicuna, Orca
- **Domain-specific**: Medical, legal, finance
- **Language-specific**: Non-English variants
- **Quantized**: GGUF, GPTQ for consumer hardware

### 4. Research Acceleration
Open weights allowed rapid experimentation:
- RLHF techniques
- Quantization methods (4-bit, 8-bit)
- Efficient fine-tuning (LoRA, QLoRA)
- Context extension techniques
- Architecture ablations

### 5. Commercial Viability
**Proved open models can compete with proprietary alternatives**:
- Llama 2 → GPT-3.5 level
- Llama 3.1 405B → GPT-4 level
- Llama 4 → GPT-4 + multimodal

Businesses can build on open foundations without vendor lock-in.

### 6. Transparency and Safety
- Open weights enable security research
- Community can audit for biases and vulnerabilities
- Purple Llama provides safety tools
- Responsible AI development in the open

---

## The Llama Philosophy

### Core Principles

1. **Efficiency Over Brute Force**: Smaller well-trained models beat larger poorly-trained ones
2. **Openness Drives Innovation**: Open weights accelerate research and development
3. **Democratization**: AI should be accessible to all, not just tech giants
4. **Community Collaboration**: Open development benefits from diverse perspectives
5. **Safety and Responsibility**: Provide tools (Llama Guard, Purple Llama) alongside models

### Why Meta Chose Open

- **Research Acceleration**: Community finds improvements faster
- **Ecosystem Growth**: More developers = more innovation
- **Trust**: Open models can be audited and verified
- **Competition**: Open models push proprietary competitors to improve
- **Strategic**: Established Meta as AI leader through openness

---

## Looking Forward

The Llama series continues to push boundaries:

- **Llama 4**: First open MoE + native multimodal with 10M context
- **Behemoth** (upcoming): 288B active (~2T total) parameters
- **Future**: Continued scaling, efficiency improvements, new modalities

Meta has committed to maintaining Llama as the flagship open-source AI model family, competing with and often surpassing the best proprietary models.

---

## Resources

### Official Links
- **Meta AI Blog**: [ai.meta.com/blog](https://ai.meta.com/blog/)
- **Llama Models Page**: [llama.com](https://www.llama.com/)
- **Hugging Face**: [huggingface.co/meta-llama](https://huggingface.co/meta-llama)
- **Purple Llama GitHub**: [github.com/meta-llama/PurpleLlama](https://github.com/meta-llama/PurpleLlama)

### Key Papers
- [LLaMA (Llama 1)](https://arxiv.org/abs/2302.13971)
- [Llama 2](https://arxiv.org/abs/2307.09288)
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (covers 3, 3.1, 3.2)
- [Code Llama](https://arxiv.org/abs/2308.12950)
- [Llama Guard](https://arxiv.org/abs/2312.06674)

### Detailed Documentation
- [Llama 1 →](meta-llama-1.md)
- [Llama 2 →](meta-llama-2.md)
- [Llama 3 Family →](meta-llama-3.md)
- [Llama 4 →](meta-llama-4.md)
- [Code Llama →](meta-llama-code.md)
- [Safety & Security Tools →](meta-llama-misc.md)
