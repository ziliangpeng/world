# Llama 2

**Release Date**: July 18, 2023

## Links

- **Paper**: [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- **Blog**: [Meta and Microsoft Introduce the Next Generation of Llama](https://ai.meta.com/blog/llama-2/)
- **Hugging Face**:
  - Base: [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf), [Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf), [Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)
  - Chat: [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)

## Origin Story: The Open-Source Pivot

The release of Llama 2 was a direct response to the "unintended" but massively successful democratization of Llama 1 after its weights were leaked. While the Llama 1 leak sparked safety concerns, it also triggered a Cambrian explosion of innovation as developers and researchers worldwide began building on the powerful base model.

Recognizing the immense value of this open ecosystem, Meta made a pivotal strategic decision. Instead of doubling down on restricted access, they chose to embrace the open-source movement fully. In a landmark partnership with **Microsoft**, Llama 2 was released under a **commercial-friendly license**, making it the first state-of-the-art foundation model that was both free to access and viable for business use. This move turned the potential PR challenge of the Llama 1 leak into a massive strategic win, cementing Meta as a leader in open-source AI and setting a new standard for the industry.

## Model Variants

- **7B**: 7 billion parameters
- **13B**: 13 billion parameters
- **70B**: 70 billion parameters

**All sizes available in two versions**:
- **Base**: Pretrained foundation models
- **Chat**: Fine-tuned with SFT + RLHF for conversation

*Note: The Llama 2 paper also details a 34B model which was part of the research and training, but it was not included in the official public release.*

## Architecture: Refinements Over Llama 1

The Llama 2 models maintain the decoder-only transformer architecture established with Llama 1 while introducing key optimizations, particularly Grouped-Query Attention for the largest variant.

### Core Architectural Components:
*   **Base Design**: Auto-regressive decoder-only transformer (unchanged from Llama 1)
*   **Normalization**: RMSNorm pre-normalization (unchanged from Llama 1)
*   **Activation**: SwiGLU activation function (unchanged from Llama 1)
*   **Position Encoding**: RoPE (Rotary Position Embeddings) (unchanged from Llama 1)
*   **Attention**: Multi-Head Attention (MHA) for 7B and 13B models (same as Llama 1). The 70B model utilizes **Grouped-Query Attention (GQA)** for improved inference efficiency (new optimization).
*   **Vocabulary**: 32K tokens using SentencePiece tokenizer (unchanged from Llama 1)

### Model Specifications:

| Parameters | Dimension (`dim`) | # Layers (`n_layers`) | # Heads (`n_heads`) | # KV Heads (`n_kv_heads`) | Context |
| :--------- | :---------------- | :-------------------- | :------------------ | :------------------------ | :------ |
| **7B**     | 4096              | 32                    | 32                  | 32 (MHA)                  | 4,096   |
| **13B**    | 5120              | 40                    | 40                  | 40 (MHA)                  | 4,096   |
| **70B**    | 8192              | 80                    | 64                  | 8 (GQA)                   | 4,096   |

**Llama 1 Comparison** (closest size match - 13B vs 65B):

| Aspect | Llama 1 13B | Llama 2 13B | Llama 1 65B | Llama 2 70B | Change |
|--------|-------------|-------------|-------------|-------------|--------|
| **Dimension** | 5120 | 5120 | 8192 | 8192 | Same |
| **Layers** | 40 | 40 | 80 | 80 | Same |
| **Heads** | 40 | 40 | 64 | 64 | Same |
| **KV Heads** | 40 (MHA) | 40 (MHA) | 64 (MHA) | **8 (GQA)** | **GQA for 70B** |
| **Vocabulary** | 32,000 | 32,000 | 32,000 | 32,000 | Same |
| **Context Window** | 2,048 | **4,096** | 2,048 | **4,096** | **2x increase** |

### Key Architectural Improvements Over Llama 1

**1. Grouped-Query Attention (GQA) for 70B Model**

The most significant architectural change from Llama 1 is the introduction of GQA for the 70B model:

| Model | Llama 1 | Llama 2 | Change |
|-------|---------|---------|--------|
| **7B/13B** | MHA (all heads independent) | MHA (same) | No change |
| **65B/70B** | MHA (64 query heads, 64 KV heads) | **GQA (64 query heads, 8 KV heads)** | **8:1 sharing ratio** |

**Benefits of GQA**:
- **87.5% reduction in KV cache size** (64 → 8 KV heads)
- **Faster inference** due to lower memory bandwidth requirements
- **Larger batch sizes** possible with same memory
- **Quality preservation**: Minimal performance loss vs full MHA

This innovation paved the way for GQA's broader adoption in Llama 3 across all model sizes.

**2. Context Window Expansion (2K → 4K)**

All Llama 2 models feature a **2x larger context window** compared to Llama 1:
- **Llama 1**: 2,048 tokens
- **Llama 2**: 4,096 tokens

This enables:
- Longer conversations and documents
- Better few-shot learning (more examples in context)
- Improved reasoning over longer passages

**3. No New Model Size (33B Dropped)**

Llama 1 included a 33B variant, but Llama 2 focused on 7B, 13B, and 70B:
- Likely due to 70B with GQA being more efficient than 33B with MHA
- Simplified the model lineup for users

*Note: The paper mentions a 34B model was trained and studied, but not publicly released.*

### Architectural Consistency with Llama 1

Llama 2 deliberately maintained **architectural compatibility** with Llama 1 for several reasons:
- **Proven foundation**: Llama 1's architecture choices (RMSNorm, SwiGLU, RoPE) were already state-of-the-art
- **Community ecosystem**: Maintaining compatibility with existing tools and fine-tunes
- **Controlled experimentation**: Isolating improvements to training data, scale, and GQA

The conservative architectural approach allowed Meta to focus on **data quality and alignment** rather than risky architectural changes.

## Training Details: Scaling Up from Llama 1

Llama 2 significantly expanded training scale compared to Llama 1, with 40% more data, 2x larger context windows, and more sophisticated post-training.

### Optimizer Configuration

**Pre-training** (unchanged from Llama 1):
*   **Optimizer:** AdamW with **β₁=0.9, β₂=0.95, and weight decay of 0.1** (same as Llama 1)
*   **Learning Rate Schedule:** Cosine decay with 2,000-step warmup (same as Llama 1)
*   **Peak Learning Rates:**
    *   **7B & 13B models:** 3.0 × 10⁻⁴ (same as Llama 1's 7B & 13B)
    *   **34B & 70B models:** 1.5 × 10⁻⁴ (same as Llama 1's 33B & 65B)
*   **Gradient Clipping:** 1.0 (same as Llama 1)

**Fine-tuning** (new in Llama 2 - not in base Llama 1):
*   **Supervised Fine-Tuning (SFT):** Cosine learning rate schedule, initial LR of 2 × 10⁻⁵, weight decay 0.1, batch size 64, sequence length 4096
*   **Reward Models (RLHF):** AdamW with constant LR of 1 × 10⁻⁶, weight decay 0.1, gradient clipping 1.0
*   **Chat Fine-tuning (RLHF):** Max LR of 5 × 10⁻⁶ (70B) or 1 × 10⁻⁵ (others), cosine decay to 10% of max

**Comparison**: Llama 1 had **no official fine-tuning**; it was released as a pure pre-trained base model. Llama 2 introduced comprehensive post-training.

### Training Scale and Data Expansion

**Token Counts:**

| Aspect | Llama 1 | Llama 2 | Increase |
|--------|---------|---------|----------|
| **Pre-training Tokens (7B)** | 1 trillion | **2 trillion** | **2x** |
| **Pre-training Tokens (13B, 65B/70B)** | 1.4 trillion | **2 trillion** | **1.43x** |
| **Context Window** | 2,048 | **4,096** | **2x** |
| **Vocabulary** | 32,000 | 32,000 | Same |

**Key Changes**:
- **40% more data** overall compared to Llama 1's largest training run
- **Uniform training**: All Llama 2 models saw 2T tokens (vs Llama 1's variable amounts)
- **Longer contexts**: 4,096 tokens enables better long-form understanding

### Data Preparation and Tokenization

**Data Mix Evolution** (compared to Llama 1):

While Llama 2 doesn't disclose exact percentages like Llama 1 did, key differences:

| Aspect | Llama 1 | Llama 2 |
|--------|---------|---------|
| **Data Mix Disclosed** | ✅ Yes (67% CommonCrawl, 15% C4, etc.) | ❌ No (proprietary curation) |
| **Data Sources** | Fully public datasets | New publicly available sources |
| **Meta Products Data** | Not explicitly excluded | **Explicitly excluded** |
| **Personal Info Filtering** | Basic filtering | **Enhanced removal** from high-PI sites |
| **Tokenization** | SentencePiece BPE, 32K vocab | SentencePiece BPE, 32K vocab (same) |
| **Number Handling** | Split into individual digits | Likely similar (not specified) |

**Fine-tuning Data** (new capability in Llama 2):

| Stage | Llama 1 | Llama 2 |
|-------|---------|---------|
| **SFT Examples** | N/A (no fine-tuning) | **~27,540** high-quality examples |
| **RLHF Annotations** | N/A (no fine-tuning) | **>1 million** human preference pairs |

The introduction of official fine-tuning data was Llama 2's biggest training innovation, enabling chat capabilities out-of-the-box.

### Training Infrastructure Scale-Up

**Llama 1 vs Llama 2 Hardware:**

| Aspect | Llama 1 | Llama 2 | Change |
|--------|---------|---------|--------|
| **Primary GPUs** | A100 80GB | A100 80GB | Same |
| **GPU Count** | 2,048 A100s | Not disclosed (similar scale) | Similar |
| **Clusters** | RSC (single cluster) | **RSC + Production clusters** | Multi-cluster |
| **Total GPU Hours** | Not disclosed | **3.3 million** | - |
| **Training Duration (65B/70B)** | ~21 days | Likely similar | Similar |
| **Throughput (per GPU)** | ~380 tokens/sec | Likely similar | Similar |

**Key Infrastructure Improvements**:
- **Multi-cluster training**: Llama 2 utilized both RSC and production clusters
  - **RSC**: Quantum InfiniBand interconnects, 400W GPU power cap
  - **Production**: RoCE (RDMA over Ethernet), 350W GPU power cap
- **Fine-tuning infrastructure**: Added support for A100 and **H100 GPUs** for post-training
- **Flexible configurations**: 1-8 GPU setups for different fine-tuning tasks

**Compute Efficiency**:
- Llama 1 pioneered efficient training techniques (activation checkpointing, overlapped communication)
- Llama 2 built on this foundation, achieving similar throughput despite 40% more data

### The "Code for Reasoning" Hypothesis: Continued Validation

While Llama 1 validated that code-heavy training improves reasoning:
- Llama 2 **likely continued** this practice (data mix not disclosed)
- The strong performance on math and reasoning benchmarks suggests code remained a significant portion
- For 33B and 65B models, Llama 1 increased code proportion specifically for reasoning gains
- Llama 2's 70B model's strong reasoning performance suggests similar data composition

This principle became standard practice across the industry, influencing Llama 3 (17% code) and other models.

## Fine-Tuning Process (Chat Variants): A Major Innovation Over Llama 1

Llama 2 introduced official fine-tuned chat models for the first time, built through a rigorous multi-stage process involving Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) with human annotations.

**Llama 1 Context**: Llama 1 was released as a **base model only** with no official fine-tuning or alignment. The community filled this gap by creating instruction-tuned variants like **Stanford's Alpaca** (fine-tuned on 52K synthetic examples from GPT-3.5) and **Vicuna** (fine-tuned on user-shared ChatGPT conversations), which demonstrated the base model's potential but lacked official support and safety guardrails.

### 1. Supervised Fine-Tuning (SFT)

*   **Objective:** To initialize the model's ability to follow instructions and generate helpful responses.
*   **Data:** Approximately **27,540 meticulously annotated human-written prompts and high-quality responses** were used. This dataset emphasized data quality over quantity.
*   **Methodology:** The pre-trained Llama 2 models were fine-tuned on this instruction-following dataset to adapt them for conversational use.

### 2. Reinforcement Learning from Human Feedback (RLHF)

RLHF was applied in an iterative process to further align the model with human preferences for helpfulness and safety.

*   **Human Preference Data:** Over **1 million human annotations** were collected, where human annotators ranked different model responses based on helpfulness and safety criteria. This massive dataset was critical for training the reward models.
*   **Reward Models:** Two separate reward models were trained:
    *   **Helpfulness Reward Model:** Trained on Meta Helpfulness data, combined with Meta Safety and open-source data.
    *   **Safety Reward Model:** Trained predominantly on Meta Safety and Anthropic Harmless data, with a smaller proportion of helpfulness data (90/10 mix). This separation allowed for distinct optimization of helpfulness and safety.
*   **Iterative Refinement (PPO & Rejection Sampling):**
    *   **PPO (Proximal Policy Optimization):** This reinforcement learning algorithm was used to fine-tune the chat models directly using the feedback from the reward models. The models were iteratively refined to maximize the reward scores.
    *   **Rejection Sampling:** For a given prompt, the model generates multiple responses, which are then scored by the reward models. The highest-scoring response is selected. This allows for a more efficient use of the expensive human preference data.

**Result:** Through this rigorous SFT and multi-stage RLHF process, Llama 2-Chat models achieved competitiveness with proprietary models like ChatGPT for many conversational use cases, while demonstrating improved safety characteristics.

### Llama 1 vs Llama 2: Post-Training Comparison

| Aspect | Llama 1 | Llama 2 | Impact |
|--------|---------|---------|--------|
| **Official Fine-Tuning** | ❌ None | ✅ SFT + RLHF | Chat capability out-of-the-box |
| **Instruction Following** | None (base only) | 27,540 SFT examples | Professional-quality responses |
| **Human Alignment** | None | >1M preference pairs | Safety and helpfulness |
| **Reward Models** | None | 2 separate (helpfulness + safety) | Balanced optimization |
| **Chat Variants** | Community-created (Alpaca, Vicuna) | **Official Meta release** | Supported and safe |
| **Safety Guardrails** | None | Llama Guard + RLHF alignment | Production-ready safety |
| **Conversational Quality** | Poor (base model completions) | **Competitive with ChatGPT** | Major leap |

**The Innovation**:
- **Llama 1's approach**: Release base model, let community handle fine-tuning
  - Led to Alpaca (Stanford), Vicuna, and dozens of derivatives
  - Demonstrated potential but lacked official support
  - No safety alignment or quality control

- **Llama 2's approach**: Official end-to-end solution
  - Professional SFT with carefully curated data
  - Dual reward models for helpfulness and safety
  - Iterative RLHF with PPO and rejection sampling
  - Result: **First open model competitive with ChatGPT**

**Industry Impact**:
- Llama 1 proved base models could be fine-tuned by community
- Llama 2 proved open-source could match proprietary chat quality
- This validated that **open models + proper alignment = production-ready AI**
- Paved the way for Llama 3's simplified DPO approach (moving beyond complex RLHF)

## Performance: Significant Gains Over Llama 1

Llama 2 demonstrated major improvements over Llama 1, closing the gap with GPT-3.5 and even surpassing it in chat applications through superior alignment.

### Overall Competitiveness

**vs GPT-3.5**:
*   **Llama 2-70B Base** performs nearly on par with **GPT-3.5** on traditional benchmarks (MMLU, GSM8k)
*   **Llama 2-Chat 70B** competitive with or superior to **ChatGPT (GPT-3.5)** in helpfulness and safety
*   36% win rate against ChatGPT, beating ChatGPT-03 by over 4 points in helpfulness

**vs Llama 1**:
*   **Llama 2-13B** significantly outperforms **Llama 1-13B** across all benchmarks
*   **Llama 2-70B** shows substantial gains over **Llama 1-65B** despite similar size
*   Introduction of Chat variants provided instruction-following capability lacking in Llama 1

### Key Benchmark Results with Llama 1 Comparison

#### MMLU (Massive Multitask Language Understanding)

| Model | Score | Improvement from Llama 1 |
|-------|-------|--------------------------|
| **Llama 2-70B (Base)** | **~63.9%** | Llama 1-65B: ~63.4% (+0.5) |
| GPT-3.5 | ~69.2% | - |
| **Llama 2-13B (Base)** | **~54.8%** | Llama 1-13B: ~46.9% **(+7.9)** |
| **Llama 2-7B (Base)** | **~45.3%** | Llama 1-7B: ~35.1% **(+10.2)** |

*Llama 2 shows significant improvements at smaller sizes, validating better training data and longer context*

#### GSM8k (Grade School Math)

| Model | Score | Improvement from Llama 1 |
|-------|-------|--------------------------|
| GPT-3.5 | ~57.0% | - |
| **Llama 2-70B (Base)** | **~46.0%** | Llama 1-65B: ~50.9% **(-4.9)** |
| **Llama 2-13B (Base)** | **~28.7%** | Llama 1-13B: ~17.8% **(+10.9)** |

*Mixed results: 70B slightly regressed, but 13B showed massive improvement*

#### HumanEval (Code Generation, 0-shot pass@1)

| Model | Score | Improvement from Llama 1 |
|-------|-------|--------------------------|
| GPT-3.5 | ~48.1% | - |
| **Llama 2-70B (Base)** | **~29.9%** | Llama 1-65B: ~23.7% **(+6.2)** |
| **Llama 2-13B (Base)** | **~18.3%** | Llama 1-13B: ~15.8% **(+2.5)** |

*Coding improved but still trails GPT-3.5 significantly; gap would be closed in Llama 3*

#### Chat Model Performance (New in Llama 2)

| Benchmark | Llama 2-70B-Chat | ChatGPT (GPT-3.5) | Llama 1 Baseline |
|-----------|------------------|-------------------|------------------|
| **Helpfulness** | High | Competitive | N/A (no chat variant) |
| **Safety** | High | Competitive | N/A (no chat variant) |
| **Factual Accuracy** | Near GPT-4 | Better than GPT-3.5-turbo | N/A (no chat variant) |

*Llama 1 had no official chat variant; community created Alpaca, Vicuna, etc. as alternatives*

### Llama 1 to Llama 2: Improvement Analysis

**Where Llama 2 Excelled**:
1. **Smaller models improved dramatically**: 7B and 13B variants showed 8-10 point MMLU gains
2. **Chat capabilities**: Official alignment made Llama 2-Chat competitive with ChatGPT
3. **Coding gains**: HumanEval improved 6+ points for 70B model
4. **Knowledge recency**: September 2022 cutoff vs Llama 1's earlier cutoff

**Where Improvements Were Mixed**:
1. **Math reasoning (70B)**: GSM8k slightly regressed (-4.9 points), possibly due to different data mix
2. **Coding still lagged**: Despite improvements, HumanEval remained far behind GPT-3.5

**Key Factors Enabling Improvements**:
1. **40% more training data** (2T vs 1.4T tokens for large models)
2. **2x context window** (4K vs 2K tokens)
3. **Better data curation** and filtering
4. **GQA for 70B** enabled more efficient inference
5. **RLHF alignment** for chat variants

### Detailed Comparison Table (Base Models Only)

| Benchmark | Llama 1-13B | Llama 2-13B | Δ | Llama 1-65B | Llama 2-70B | Δ |
|-----------|-------------|-------------|---|-------------|-------------|---|
| **MMLU** | 46.9% | 54.8% | **+7.9** | 63.4% | 63.9% | +0.5 |
| **GSM8k** | 17.8% | 28.7% | **+10.9** | 50.9% | 46.0% | -4.9 |
| **HumanEval** | 15.8% | 18.3% | **+2.5** | 23.7% | 29.9% | **+6.2** |
| **HellaSwag** | ~79.2% | ~82.0% | **+2.8** | ~84.2% | ~85.3% | +1.1 |

### Strengths and Weaknesses

**Strengths** (vs Llama 1):
*   **General knowledge**: MMLU improvements, especially for smaller models
*   **Helpfulness**: Official chat variants match ChatGPT quality
*   **Safety**: Strong safety alignment through RLHF (vs Llama 1's lack of guardrails)
*   **Factual accuracy**: Near GPT-4 level on summarization tasks
*   **Knowledge cutoff**: September 2022 (more recent than GPT-3.5's June 2021)
*   **Context window**: 4K tokens enables longer conversations (vs Llama 1's 2K)

**Weaknesses** (vs GPT-3.5 and Llama 1 gaps):
*   **Coding**: Still significantly behind GPT-3.5 on HumanEval (would improve in Llama 3)
*   **Math (70B)**: Slight regression on GSM8k compared to Llama 1-65B
*   **Specialized tasks**: Less effective where GPT-3.5 might have specific fine-tuning

**The Path Forward**: These benchmarks revealed that while Llama 2 made significant progress, there was still room for improvement in coding and math reasoning—challenges that would be addressed in Llama 3 through dramatically increased code and math data (17% code, 25% math/reasoning).

## Legacy and Impact

The release of Llama 2 was a pivotal moment for the open-source AI movement, proving that open models could be directly competitive with and commercially viable against closed, proprietary systems.

*   **Established a New Open-Source Standard:** With its commercial-friendly license, high performance rivaling models like GPT-3.5, and official safety-aligned chat variants, Llama 2 became the go-to foundation for countless projects and businesses. It proved that open-source AI was ready for production.
*   **Pioneered GQA at Scale:** The 70B model's use of Grouped-Query Attention was the first major production deployment of the technique, demonstrating its efficiency benefits and paving the way for its adoption in later models.
*   **Accelerated the Fine-Tuning Ecosystem:** The fully accessible and high-quality base models triggered an explosion of innovation. The community created thousands of specialized fine-tunes for specific domains (e.g., medical, legal, coding) and developed new quantization techniques (GGUF, GPTQ) to run models on consumer hardware.
*   **Democratized Access to AI:** By removing the application process required for Llama 1 and partnering with Microsoft for broad cloud availability on Azure, Llama 2 made state-of-the-art AI truly accessible to individual developers, researchers, and businesses of all sizes.

## Key Figures

The development and open-sourcing of Llama 2 was a collaborative effort involving a large team of researchers and engineers, guided by Meta's AI leadership.

*   **Hugo Touvron (Lead Author):** As the first author on the Llama 2 paper (and Llama 1), Touvron was a primary force in the hands-on research and development of the model.
*   **Louis Martin, Kevin Stone, and the Llama Team:** Among over 65 contributors listed on the paper, Martin and Stone are noted as primary authors, representing the extensive team effort behind Llama 2's development.
*   **Joelle Pineau (Head of FAIR):** As the Managing Director of Meta AI's Fundamental AI Research (FAIR) group, Pineau played a pivotal leadership role, guiding the strategic development and release of the Llama models.
*   **Yann LeCun (Chief AI Scientist):** While stating his direct involvement in Llama 2's development was "very indirect," LeCun was a prominent advocate for the full open-source release of Llama 2, championing Meta's commitment to open AI.