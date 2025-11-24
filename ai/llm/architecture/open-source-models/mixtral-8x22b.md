# Mixtral 8x22B

**Release Date**: April 17, 2024

## Links

- **Official Announcement**: [Cheaper, Better, Faster, Stronger](https://mistral.ai/news/mixtral-8x22b) (Mistral AI Blog)
- **Paper**: No dedicated paper. Architecture builds on [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (arXiv:2401.04088)
- **Hugging Face**:
  - [mistralai/Mixtral-8x22B-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1) (Base Model)
  - [mistralai/Mixtral-8x22B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) (Instruct Model)
- **Documentation**: [docs.mistral.ai - Mixtral 8x22B](https://docs.mistral.ai/models/mixtral-8x22b-0-1-0-3)

## Origin Story: Scaling the MoE Vision

Just three months after releasing Mixtral 8x7B, Mistral AI dropped their next bombshell: **Mixtral 8x22B**, a massive scaling of their Sparse Mixture-of-Experts architecture. Released on April 17, 2024, this model represented a **3x increase in active parameters** (from 13B to 39B) and a **2x expansion of context window** (from 32K to 64K tokens), while maintaining the core efficiency advantage that made the original Mixtral a sensation.

### The Unconventional Release (Again)

True to form, Mistral AI continued their now-signature release strategy: **a raw torrent link dropped on X/Twitter**. On April 10, 2024—during Google Cloud Next conference—Mistral tweeted a magnet link to a 281GB file, with no fanfare, no blog post, no demo. The formal announcement came a week later on April 17, but by then the community had already been experimenting with the weights.

This "tweet-a-torrent" approach, which started with the original Mixtral, had become Mistral's "standard operating procedure," a deliberate rejection of traditional AI lab marketing in favor of raw access and community-driven discovery.

### The Competitive Context: April 2024

Mixtral 8x22B launched into an increasingly competitive open-source landscape:

- **Llama 2 70B** was the reigning champion of open models, but Mixtral 8x7B had already proven competitive
- **GPT-4 Turbo with vision** and **Gemini 1.5 Pro** had raised the bar for proprietary models
- The open-source community was hungry for a model that could narrow the gap with frontier proprietary models

Mixtral 8x22B's goal was clear: **push open-source performance to new heights** while maintaining the efficiency advantage of sparse activation. With 141B total parameters but only 39B active at any time, it aimed to outperform dense 70B models while running faster and cheaper.

### Apache 2.0: Doubling Down on Open

Like its predecessor, Mixtral 8x22B was released under the **Apache 2.0 license**—the most permissive open-source license available. This meant unrestricted commercial use, no registration requirements, and complete freedom to modify and distribute the model. In an era where many "open" models came with restrictive licenses, Mistral AI continued to be a true believer in radical openness.

## Model Variants

- **Mixtral-8x22B-v0.1**: Base pretrained generative Sparse Mixture of Experts model
- **Mixtral-8x22B-Instruct-v0.1**: Instruction-tuned variant optimized for conversational use and tool calling

## Architecture

### Core Design: Sparse Mixture of Experts (SMoE)

Mixtral 8x22B uses the same fundamental architecture as Mixtral 8x7B: a **decoder-only transformer** where each layer contains **8 expert networks**, but only **2 experts are activated per token**. This sparse activation is what allows a 141B parameter model to run with the speed and memory footprint of a ~39B dense model.

The key innovation is **scale**: everything is bigger than 8x7B while maintaining the same MoE efficiency principles.

### Model Specifications

| Parameter | Mixtral 8x22B | Mixtral 8x7B | Change |
|-----------|---------------|--------------|--------|
| **Total Parameters** | 141B | 46.7B | **3.0x** |
| **Active Parameters** | 39B | 12.9B | **3.0x** |
| **Layers** | 56 | 32 | **1.75x** |
| **Attention Heads** | 48 | 32 | **1.5x** |
| **Key-Value Heads** | 8 | 8 | Same (GQA) |
| **Hidden Dimensions** | 6,144 | 4,096 | **1.5x** |
| **FFN Intermediate Size** | 16,384 | 14,336 | **1.14x** |
| **Context Window** | 65,536 | 32,768 | **2.0x** |
| **Vocabulary Size** | 32,000 | 32,000 | Same |
| **Number of Experts** | 8 | 8 | Same |
| **Active Experts** | 2 | 2 | Same |

### Architectural Components

- **Base Design**: Decoder-only transformer with Sparse Mixture of Experts
- **Normalization**: RMSNorm (ε = 1e-05)
- **Activation**: SiLU (in expert FFNs)
- **Position Encoding**: RoPE (Rotary Position Embeddings) with θ = 1,000,000
- **Attention**: Grouped Query Attention (GQA) - 48 query heads, 8 KV heads
- **Attention Dropout**: 0.0 (no dropout)
- **Expert Routing**: Top-2 gating with auxiliary load balancing loss (coefficient: 0.001)
- **Precision**: bfloat16 (BF16)
- **Tokenizer**: SentencePiece (32K vocabulary)

### What Changed from 8x7B?

While the core MoE architecture remains identical, Mixtral 8x22B scales up in three critical dimensions:

1. **Depth**: 56 layers vs 32 layers (75% more layers for deeper reasoning)
2. **Width**: 6,144 hidden dimensions vs 4,096 (50% wider for richer representations)
3. **Context**: 64K tokens vs 32K tokens (2x context for longer documents)

The result is a model with **3x the active parameters** and **2x the context window**, but still using the same sparse activation principle: only 2 out of 8 experts are active per token, keeping inference efficient.

### The 64K Context Window

The expansion to 64K tokens was a major upgrade, enabling:
- Full book chapters or technical papers in a single context
- Complex multi-document reasoning
- Long conversation histories without truncation
- Large codebases in context

This was achieved by setting the **RoPE theta to 1,000,000** (vs typical values of 10,000), which extends the effective range of rotary position embeddings to handle longer sequences.

### Native Function Calling

Unlike Mixtral 8x7B, the 8x22B Instruct variant includes **native function calling support** with special tokens:
- `[AVAILABLE_TOOLS]`: Defines tools available to the model
- `[TOOL_CALLS]`: Model's decision to invoke a tool
- `[TOOL_RESULTS]`: Results returned from tool execution

This makes Mixtral 8x22B Instruct compatible with constrained output modes and structured tool use, positioning it for agentic applications.

## Training Details

Unlike the original Llama or GPT models, Mistral AI does not publish detailed training information for their models. Mixtral 8x22B was released without a dedicated research paper, documented only through a blog post and model cards on HuggingFace.

### What the Documentation Discloses

**Model Type**: Pretrained generative Sparse Mixture of Experts

**Fine-Tuning (Instruct Variant)**:
- **Likely Method**: Direct Preference Optimization (DPO)
  - The original Mixtral 8x7B used DPO for instruction tuning
  - Mixtral 8x22B Instruct likely follows the same methodology
  - DPO is simpler than RLHF, conducting preference learning in a single supervised step

**Architecture**: Fully specified (see Architecture section above)

**Inference Details**: Context window, tokenization, and model configuration are all documented

### What Is NOT Disclosed

Mistral AI has not publicly released the following training details:

**Optimizer Configuration:**
- Optimizer type (AdamW, Adam, Lion, etc.)
- Learning rate and schedule (warmup steps, decay strategy)
- Beta parameters (β₁, β₂)
- Epsilon value
- Weight decay coefficient
- Gradient clipping threshold

**Training Hyperparameters:**
- Batch size (tokens per batch)
- Sequence length during training
- Total training tokens
- Number of training steps
- Gradient accumulation steps
- Mixed precision strategy

**Training Data:**
- Data sources (web, books, code, scientific papers, etc.)
- Data mix ratios (% CommonCrawl, % GitHub, etc.)
- Multilingual distribution (% English vs other languages)
- Data quality filters and preprocessing
- Deduplication strategy

**Training Infrastructure:**
- GPU type (A100, H100, etc.)
- Number of GPUs
- Training duration (days/weeks)
- Total compute (FLOPs or GPU-hours)
- Training throughput (tokens/sec/GPU)
- Distributed training strategy (data parallel, pipeline parallel, tensor parallel)

**MoE-Specific Training:**
- Load balancing loss coefficient (config shows 0.001, but training dynamics not explained)
- Expert capacity factors
- Router z-loss or other routing stabilization techniques
- How expert specialization emerged during training

**Environmental Impact:**
- Carbon emissions estimate
- Energy consumption

### Why the Lack of Disclosure?

This pattern—**open weights, closed training recipe**—is now standard practice for commercial AI companies (OpenAI, Anthropic, Google, Mistral AI). The rationale typically includes:

1. **Competitive Advantage**: Training methodology represents proprietary know-how
2. **Cost of Reproduction**: Even with full details, reproducing training runs costs millions
3. **Community Acceptance**: The open-source community values access to weights over training transparency

Importantly, the **Apache 2.0 license** means researchers and developers get full access to the trained model without barriers, even if the training process remains opaque. For most use cases—fine-tuning, deployment, research—the weights are what matter, not the training recipe.

## Performance

Mixtral 8x22B set new benchmarks for open-source models at its release in April 2024, delivering performance competitive with models 2-3x its active parameter count.

### Base Model Benchmarks

| Benchmark | Mixtral 8x22B | Mixtral 8x7B | Llama 2 70B | GPT-4 | Metric |
|-----------|---------------|--------------|-------------|-------|--------|
| **MMLU** (5-shot) | 77.8 | 70.6 | 69.9 | 86.6 | Accuracy (%) |
| **HellaSwag** (10-shot) | 88.9 | 86.7 | ~85 | ~95 | Accuracy (%) |
| **ARC Challenge** | 70.5 | 67.0 | ~65 | ~96 | Accuracy (%) |
| **GSM8K** | 76.5 | 74.4 | 69.6 | ~92 | Accuracy (%) |
| **HumanEval** | Leading | 40.2 | 29.3 | ~86 | pass@1 (%) |
| **MBPP** | Leading | 60.7 | 49.8 | ~85 | pass@1 (%) |

*Note: "Leading" indicates best-in-class for open models at release; exact scores not disclosed in all sources.*

### Instruct Model Performance

The instruction-tuned variant shows significant improvements on reasoning-heavy tasks:

| Benchmark | Mixtral 8x22B Instruct | Metric |
|-----------|------------------------|--------|
| **GSM8K** (maj@8) | 90.8% | Math reasoning with majority voting |
| **MATH** (maj@4) | 44.6% | Advanced mathematics |
| **MMLU** | 75.6% | General knowledge (slight drop from base) |
| **HumanEval** | Best among open models | Code generation pass@1 |
| **MBPP** | Best among open models | Code generation pass@1 |

The **maj@8** and **maj@4** notation refers to **majority voting**: the model generates multiple solutions (8 for GSM8K, 4 for MATH), and the most common answer is selected. This technique significantly boosts accuracy on problems with definitive correct answers.

### Key Strengths

**Mathematics**: The 90.8% GSM8K score (with majority voting) places Mixtral 8x22B Instruct among the strongest open models for grade-school math. The 44.6% MATH score shows competitive performance on competition-level mathematics.

**Coding**: Both HumanEval and MBPP scores are described as "leading among open models," indicating best-in-class code generation for open-source LLMs at the time of release.

**Multilingual**: Mixtral 8x22B **strongly outperforms Llama 2 70B** on non-English benchmarks:
- HellaSwag (French, German, Spanish, Italian)
- ARC Challenge (French, German, Spanish, Italian)
- MMLU (French, German, Spanish, Italian)

The model is fluent in **English, French, Italian, German, and Spanish**, reflecting Mistral AI's European roots and multilingual training data.

### Comparative Analysis

**vs Mixtral 8x7B**: Significant improvements across all benchmarks (typically +5-10 percentage points), validating the scaling strategy.

**vs Llama 2 70B**: Mixtral 8x22B outperforms on most tasks while being **6x faster** at inference (39B active vs 70B always active). Particularly strong advantages in code and math.

**vs GPT-4**: GPT-4 maintains a substantial lead (~10-30 points on most benchmarks), but Mixtral 8x22B narrows the gap compared to earlier open models. The cost-performance ratio favors Mixtral for many applications.

**vs Llama 3.1 70B** (released July 2024, after Mixtral 8x22B):
- Llama 3.1 70B: 80.5 MMLU vs Mixtral 8x22B: 77.8
- Llama 3.1 70B: 95.1 GSM8K vs Mixtral 8x22B Instruct: 88.2
- Llama 3.1 eventually surpassed Mixtral 8x22B, but Mixtral held the open-source crown for ~3 months

### Efficiency Metrics

**Inference Speed**: Faster than any dense 70B model despite having 141B total parameters. The sparse activation of only 39B parameters per token delivers:
- Lower latency per token
- Lower memory bandwidth requirements
- Better throughput on GPU clusters

**Resource Requirements** (vs Mixtral 8x7B):
- **2.1x slower** inference
- **3.3x more RAM** required
- Still significantly faster and cheaper than dense models of comparable quality (e.g., Llama 2 70B)

## Impact and Legacy

### Reception: "Most Powerful Open-Source Model"

At release, Mixtral 8x22B was widely hailed as **"one of the most powerful open-source AI models yet"**. The combination of strong performance, Apache 2.0 licensing, and the now-iconic torrent release created immediate excitement in the AI community.

Key reactions:
- Recognition as a "key milestone for open-source generative AI"
- Praise for the cost-performance ratio enabling broader access
- Appreciation for unrestricted licensing vs competitors with restrictive terms
- Validation of the MoE architecture as a viable path for open models

### Adoption and Derivatives

The open-source community quickly embraced Mixtral 8x22B:

**Direct Derivatives** (as of documentation):
- **7 fine-tuned variants** for specialized domains
- **2 LoRA adapters** for efficient task-specific tuning
- **3 model merges** combining Mixtral with other models
- **6+ quantized versions** (4-bit, 8-bit) for consumer hardware
- **Powers 100+ Spaces** on HuggingFace (applications, demos, experiments)

### Use Cases and Applications

Mistral AI and the community identified several high-value applications:

**Enterprise**:
- Customer service and support chatbots
- Content generation and summarization
- Code generation and assistance

**Research**:
- Drug discovery (processing scientific literature)
- Climate modeling (analyzing large datasets)
- Scientific research (long document analysis with 64K context)

**Platform Integration**:
- Released on "la Plateforme" (Mistral's API platform)
- Available on HuggingFace Inference API
- Supported by Together AI
- Integrated into NVIDIA NIM (optimized inference)
- Available through various cloud providers

### The Open-Source Throne (April-July 2024)

For approximately **three months**, Mixtral 8x22B held the title of best open-source LLM:

- **April 2024**: Release, immediately surpasses Llama 2 70B
- **May-June 2024**: Dominates open-source benchmarks
- **July 2024**: Llama 3.1 70B release eventually takes the crown

This period demonstrated that a well-executed MoE architecture could compete with—and temporarily beat—the best dense open models, while maintaining superior efficiency.

### Continued Relevance

Even after Llama 3.1's release, Mixtral 8x22B remains relevant for:

**Efficiency-Sensitive Applications**: The 6x inference speedup vs dense 70B models matters for high-throughput serving.

**Multilingual Use Cases**: Strong non-English performance makes it a top choice for European languages.

**Function Calling**: Native tool use support makes it well-suited for agentic applications.

**Cost-Conscious Deployments**: Lower inference costs enable broader accessibility.

### The Mistral AI Legacy

Mixtral 8x22B cemented several key aspects of Mistral AI's identity:

1. **MoE as a Signature Architecture**: Validated sparse activation as Mistral's competitive differentiator
2. **Unconventional Releases**: The "tweet-a-torrent" approach became a beloved quirk
3. **True Open Source**: Apache 2.0 licensing without compromise
4. **European AI Leadership**: Demonstrated that European AI labs could compete with US giants

By April 2024, Mistral AI—founded just 10 months earlier—had released two of the most influential open-source models of the era, fundamentally reshaping the landscape of accessible AI.

## Sources

- Mistral AI Official Blog: [Cheaper, Better, Faster, Stronger](https://mistral.ai/news/mixtral-8x22b)
- HuggingFace Model Cards: [Base](https://huggingface.co/mistralai/Mixtral-8x22B-v0.1), [Instruct](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1)
- Mistral AI Documentation: [docs.mistral.ai](https://docs.mistral.ai/models/mixtral-8x22b-0-1-0-3)
- Model Configuration: [config.json](https://huggingface.co/mistral-community/Mixtral-8x22B-v0.1/blob/main/config.json)
- Mixtral of Experts Paper: [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
- Technical Analysis: [Prompt Engineering Guide](https://www.promptingguide.ai/models/mixtral-8x22b)
- Benchmark Analysis: [Towards Data Science](https://towardsdatascience.com/mistral-vs-mixtral-comparing-the-7b-8x7b-and-8x22b-large-language-models-58ab5b2cc8ee/)
- News Coverage: [SiliconANGLE](https://siliconangle.com/2024/04/10/mistralai-debuts-mixtral-8x22b-one-powerful-open-source-ai-models-yet/), [VentureBeat](https://venturebeat.com/ai/mistral-ai-drops-new-mixture-of-experts-model-with-a-torrent-link/), [AI News](https://www.artificialintelligence-news.com/2024/04/18/mixtral-8x22b-sets-new-benchmark-open-models/)
