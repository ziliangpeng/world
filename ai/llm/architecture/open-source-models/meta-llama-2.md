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

## Architecture

The Llama 2 models maintain the decoder-only transformer architecture, building upon the foundation established with Llama 1 while introducing key optimizations, especially for larger variants.

### Core Architectural Components:
*   **Base Design**: Auto-regressive decoder-only transformer
*   **Normalization**: RMSNorm pre-normalization
*   **Activation**: SwiGLU activation function
*   **Position Encoding**: RoPE (Rotary Position Embeddings)
*   **Attention**: Multi-Head Attention (MHA) for 7B and 13B models. The 70B model utilizes Grouped-Query Attention (GQA) for improved inference efficiency.

### Model Specifications:

| Parameters | Dimension (`dim`) | # Layers (`n_layers`) | # Heads (`n_heads`) | # KV Heads (`n_kv_heads`) |
| :--------- | :---------------- | :-------------------- | :------------------ | :------------------------ |
| **7B**     | 4096              | 32                    | 32                  | 8                         |
| **13B**    | 5120              | 40                    | 40                  | 8                         |
| **70B**    | 8192              | 80                    | 64                  | 8                         |

*Note: Grouped-Query Attention (GQA) is an optimization where multiple query heads share the same key and value heads, reducing the computational cost and memory footprint during inference, especially beneficial for larger models like the 70B variant.*

## Training Details

Llama 2 was trained on an even larger dataset than its predecessor, with a sophisticated multi-stage training and fine-tuning process.

### Optimizer Configuration

*   **Pre-training:** The AdamW optimizer was used with **β₁=0.9, β₂=0.95, and a weight decay of 0.1**. A cosine learning rate schedule was employed with a 2,000-step warm-up, and the final learning rate decayed to 10% of the peak. The peak learning rates were:
    *   **7B & 13B models:** 3.0 × 10⁻⁴
    *   **34B & 70B models:** 1.5 × 10⁻⁴
*   **Supervised Fine-Tuning (SFT):** Also utilized a cosine learning rate schedule, starting with an initial learning rate of 2 × 10⁻⁵. This phase included a weight decay of 0.1, a batch size of 64, and a sequence length of 4096 tokens.
*   **Reward Models (for RLHF):** Trained with the AdamW optimizer, using a constant learning rate of 1 × 10⁻⁶, a weight decay of 0.1, and gradient clipping at 1.0.
*   **Llama 2-Chat Fine-tuning:** For the final RLHF stage, the maximum learning rate was set to 5 × 10⁻⁶ for the 70B parameter model and 1 × 10⁻⁵ for other models, decaying on a cosine schedule to 10% of the maximum.

### Training Scale and Data

*   **Tokens Trained:** The models were pre-trained on a massive **2 trillion tokens** of new and publicly available online data, a 40% increase over the Llama 1 dataset.
*   **Context Window:** The context window was doubled to **4,096 tokens**, allowing the models to process and generate longer sequences.
*   **Vocabulary:** The vocabulary remained at **32K tokens** using the SentencePiece tokenizer, consistent with Llama 1.

### Data Preparation and Tokenization

*   **Data Mix and Filtering:** Unlike Llama 1, the paper does not disclose the exact pre-training data mix. However, it emphasizes that the data is from new, publicly available sources and that significant effort was made to **exclude data from Meta's own products or services** and to remove data from sites known to contain a high volume of personal information.
*   **Fine-tuning Data:**
    *   **SFT:** Approximately **27,540 meticulously annotated, high-quality instruction-tuning instances** were used for Supervised Fine-Tuning.
    *   **RLHF:** Over **1 million human annotations** were collected to create the preference dataset for training the reward models for Reinforcement Learning from Human Feedback.

### Training Infrastructure

Llama 2's pre-training was a colossal undertaking, leveraging Meta's advanced GPU clusters.

*   **GPUs:** Pre-training primarily utilized **NVIDIA A100 GPUs (80GB)**.
*   **Clusters:** The effort was spread across Meta's **Research Super Cluster (RSC)** and internal production clusters.
    *   **RSC:** Featured NVIDIA Quantum InfiniBand interconnects, with GPUs operating under a 400W power consumption cap.
    *   **Production Clusters:** Employed RoCE (RDMA over Converged Ethernet) solutions, with GPUs operating under a 350W power consumption cap.
*   **Total Compute:** The entire pre-training amounted to an astounding **3.3 million GPU hours** on A100-80GB GPUs.
*   **Fine-tuning Hardware:** Fine-tuning also used A100 and newer H100 GPUs (80GB), often in configurations of up to 8 GPUs per node (640GB total memory), with smaller models being fine-tuned on single GPUs.

## Fine-Tuning Process (Chat Variants)

Llama 2 introduced official fine-tuned chat models, built through a rigorous multi-stage process involving Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) with human annotations.

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

## Performance

Llama 2 demonstrated significant advancements over its predecessor, achieving competitive performance against leading proprietary models like GPT-3.5 and even approaching ChatGPT's quality in chat applications.

### Overall Competitiveness

*   **Llama 2-70B Base model** performs nearly on par with **GPT-3.5** on many traditional benchmarks like MMLU and GSM8k.
*   **Llama 2-Chat models** (especially the 70B variant) have been shown through human evaluations to be competitive with or superior to **ChatGPT (GPT-3.5)** in terms of helpfulness and safety, achieving a 36% win rate and beating ChatGPT-03 by over 4 points in helpfulness assessments.

### Key Benchmark Results

| Benchmark Category | Llama 2-70B (Base) | GPT-3.5 | Llama 2-70B (Chat) | ChatGPT | Notes |
| :----------------- | :----------------- | :------ | :----------------- | :------ | :---- |
| **MMLU** (Average) | ~63.9%             | ~69.2%  | -                  | -       | Base model close to GPT-3.5 |
| **GSM8k** (Math)   | ~46.0%             | ~57.0%  | -                  | -       | Base model performs well, still behind GPT-3.5 |
| **HumanEval** (Code, pass@1) | ~29.9%             | ~48.1%  | -                  | -       | Llama 2 underperforms GPT-3.5 in coding. |
| **Helpfulness** (Human Eval) | -                  | -       | High               | Competitive | Llama 2-Chat often preferred over ChatGPT. |
| **Safety** (Human Eval) | -                  | -       | High               | Competitive | Llama 2-Chat designed with strong safety alignment. |
| **Factual Accuracy** (Summarization) | -                  | -       | Near GPT-4         | Better than GPT-3.5-turbo | Llama 2-70B Chat model. |

*Note: Specific numerical scores for GPT-3.5 and ChatGPT are indicative averages from various reports and may vary based on exact model versions and evaluation methodologies. Llama 2's knowledge cutoff was September 2022, providing it with more recent information than GPT-3.5's June 2021 cutoff.*

### Strengths and Weaknesses

*   **Strengths:** Llama 2 excels in general helpfulness, safety, and factual accuracy, often matching or surpassing GPT-3.5. It also benefits from faster response times and a more recent knowledge cutoff.
*   **Weaknesses:** It tends to underperform GPT-3.5 in coding benchmarks (HumanEval) and can be less effective in highly specialized tasks where GPT-3.5 might be fine-tuned.

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