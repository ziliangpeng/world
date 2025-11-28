# Llama 1 (LLaMA)

## Overview

**Release Date**: February 24, 2023 | **Organization**: Meta AI | **License**: Research-only (later leaked)

LLaMA was Meta's proof that smaller models trained on more data could outperform larger models—validating DeepMind's Chinchilla scaling laws in practice. The project started with just 5 researchers at Meta AI Paris.

**Performance**: GPT-3 class—the 13B model [outperforms GPT-3 (175B)](https://arxiv.org/abs/2302.13971) on most benchmarks despite being 10x smaller (46.9% vs 43.9% MMLU). The 65B model is competitive with Chinchilla-70B but doesn't reach GPT-3.5 tier (63.4% MMLU).

**Significance**: First major open model to challenge "bigger is better" orthodoxy. Sparked open-source LLM explosion when weights leaked to 4chan one week after release. Models like Alpaca, Vicuna, and most subsequent open-source work trace back to LLaMA.

**Technical Innovations**: Pragmatic synthesis of cutting-edge techniques (RMSNorm, SwiGLU, RoPE) that became the template for all modern LLMs. Proved code training improves reasoning.

## Origin Story: An Underdog Mission

The Llama 1 project began in August 2022 with a core team of only about **five researchers** at Meta AI. Their mission was to challenge the prevailing "bigger is always better" philosophy by proving that smaller, more efficiently trained models could outperform larger competitors.

Their core strategy was to bet on the "[Chinchilla scaling laws](https://arxiv.org/abs/2203.15556)"—DeepMind's theory that most large models were "starved" of data. The Llama team's goal was to prove that a model trained on massive data (1+ trillion tokens) could match larger models with fewer parameters (7B to 65B). This focus on **data volume and quality over sheer model size** became the project's guiding principle.

## Model Variants

- **7B**: 7 billion parameters
- **13B**: 13 billion parameters
- **33B**: 33 billion parameters
- **65B**: 65 billion parameters

*Note: Llama 1 was released as a base model only. No official chat or instruct-tuned variants were provided by Meta.*

## Architecture

### Core Architectural Components

- **Base Design**: Auto-regressive decoder-only transformer
- **Normalization**: [RMSNorm](../../architecture/normalization.md) pre-normalization
- **Activation**: [SwiGLU](../../architecture/activations.md)
- **Position Encoding**: [RoPE](../../architecture/position-embeddings.md)
- **Attention**: Multi-Head Attention (MHA)
- **FFN Dimension**: 2/3 × 4d (as in PaLM)
- **Vocabulary**: 32K tokens (SentencePiece tokenizer)

### Model Specifications

| Parameters | Dimension (`dim`) | # Layers (`n_layers`) | # Heads (`n_heads`) | FFN Dim (`hidden_dim`) |
|------------|-------------------|-----------------------|---------------------|------------------------|
| **7B**     | 4096              | 32                    | 32                  | 11008                  |
| **13B**    | 5120              | 40                    | 40                  | 13824                  |
| **33B**    | 6656              | 60                    | 52                  | 17920                  |
| **65B**    | 8192              | 80                    | 64                  | 22016                  |

### Architectural Philosophy: Pragmatic Innovation

The architectural genius of Llama 1 was not the invention of entirely new components, but the pragmatic and effective **synthesis of existing, cutting-edge techniques**. The team surveyed the landscape of recent LLM research, identified the most promising individual improvements from different papers, and integrated them into a single, cohesive, and highly optimized model.

This approach involved a few key layers of innovation:

1.  **Validation at Scale:** It's one thing for a technique like RoPE to show promise in an academic paper. It's another to prove it works reliably in a trillion-token training run. The Llama team's rigorous, large-scale validation of this specific "recipe" was a major engineering contribution that de-risked these choices for the entire community.

2.  **Proving the "Smaller is Better" Thesis:** The architecture was the vehicle for proving the theory that smaller models trained on far more data could outperform larger models. Llama 1 was the first major open model to deliver on this, with the 13B model famously beating the 175B GPT-3 on most benchmarks.

3.  **Creating a High-Quality Open Baseline:** Before Llama, open-source models were not considered serious competitors to proprietary ones. By releasing an exceptionally good model, Meta provided a high-quality baseline that unlocked a cambrian explosion of research and fine-tuning in the open-source community.

The decision to use this specific combination was the result of extensive ablation studies on smaller models, where different components were tested and measured for their impact on performance, stability, and training speed. The final choices represent a carefully optimized balance of performance gains versus computational cost and implementation risk.


### Hyperparameter Choices and Scaling

The specific values for dimension, layers, and heads are not arbitrary. They are the result of a complex optimization process guided by three main factors:

1.  **[Chinchilla Scaling Laws](https://arxiv.org/abs/2203.15556):** This DeepMind research argued that for optimal performance, model size and training data size should be scaled proportionally. Llama was the first major project to prove this theory in practice, intentionally training smaller models on far more data than previous models like GPT-3.

2.  **Hardware Efficiency:** The specific numbers (e.g., a `dim` of 4096, a `head_dim` of 128) are chosen to maximize training throughput on the underlying hardware (NVIDIA A100 GPUs). Dimensions are typically set to multiples of 64 or 128 to align with the architecture of GPU cores and memory, making matrix calculations significantly faster.

3.  **Iterative Experimentation:** While guided by scaling laws, the final ratios of depth vs. width were likely refined through extensive, small-scale experiments. Teams typically run hundreds of tests on smaller models to find a "sweet spot" before committing to a multi-million dollar training run.

These choices are highly sensitive and interdependent. Changing one value requires re-balancing the others to stay within a specific parameter budget, all while trying to maximize performance and efficiency.


## Training Details
Llama 1 was released as a purely pre-trained base model, with no official post-training or fine-tuning (e.g., for instruction-following or alignment) applied by Meta.

### Optimizer Configuration
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95)
- **Learning Rate Schedule**: Cosine decay, with a 2,000-step warmup.
  - **Peak LR (7B, 13B)**: 3.0 × 10⁻⁴
  - **Peak LR (33B, 65B)**: 1.5 × 10⁻⁴
- **Batch Size**: 4M tokens
- **Weight Decay**: 0.1
- **Gradient Clipping**: 1.0

These optimizer settings are not novel; they represent the established best practices for training large Transformers in 2022-2023. The choices of AdamW, a cosine learning rate schedule, and gradient clipping were standard. The specific beta values (`β₂=0.95`) and learning rates were the result of meticulous hyperparameter tuning, following the precedent set by other large-scale models like GPT-3 rather than inventing new techniques. The innovation was in the flawless execution and precise tuning of these known methods at a massive scale.

### Training Scale and Data
- **Tokens Trained**:
  - **7B**: 1 trillion
  - **13B, 33B, 65B**: 1.4 trillion
- **Context Window**: 2,048 tokens
- **Training Data Mix**: A blend of public datasets:
  - **CommonCrawl (filtered)**: 67%
  - **C4**: 15%
  - **GitHub**: 4.5%
  - **Wikipedia**: 4.5%
  - **Books (Gutenberg & Books3)**: 4.5%
  - **ArXiv**: 2.5%
  - **Stack Exchange**: 2%

The most significant innovation in Llama 1's training was the **massive volume of data**, specifically the 1 to 1.4 trillion tokens used. This was a direct, successful application of the Chinchilla scaling laws, proving that training smaller models on significantly more data yielded superior performance compared to larger models trained on less data, a paradigm shift for the industry. In contrast, the 2,048-token context window was a standard industry practice at the time. The data mix, while using common sources, represented innovation not in novel data sources, but in the meticulous curation, rigorous filtering, quality control, and strategic blending of these diverse, publicly available sources (from code to scientific papers) ...to create a high-quality foundation for the model.

### Data Preparation and Tokenization

The quality of the training data was a primary focus, involving several key preparation steps:

*   **Aggressive Filtering:** A custom classifier was trained to filter the vast CommonCrawl dataset, removing low-quality pages and keeping text that resembled high-quality reference material like Wikipedia.
*   **Strategic Blending:** The data mix was intentionally skewed towards high-quality sources. For the larger 33B and 65B models, the proportion of code from GitHub was increased to specifically enhance reasoning and problem-solving capabilities.
*   **Tokenization Details:
    *   **Tokenizer:** A SentencePiece tokenizer using the Byte-Pair Encoding (BPE) algorithm was used, with a vocabulary size of 32K tokens.
    *   **Number Handling:** A crucial detail is that all numbers were split into individual digits. This forces the model to learn the mathematical properties of numbers rather than memorizing them as distinct entities, which aids in numerical reasoning.

#### The "Code for Reasoning" Hypothesis

While the Llama 1 team was not the first to discover the link between code-training and reasoning ability—predecessors like OpenAI's Codex and Google's PaLM had already indicated this—their work served as a massive, large-scale validation of the concept. By explicitly showing that increasing the proportion of code data directly improved performance on math and reasoning benchmarks, they helped solidify this into a core principle for the entire AI community.

    This finding has had a lasting impact, influencing nearly all subsequent foundation models (including the Mistral series and later Llama versions). Including a significant portion of high-quality code in the pre-training data is now considered standard practice for building capable, general-purpose models.

### Training Infrastructure

Training Llama 1 models required a substantial compute infrastructure and sophisticated optimization to achieve the reported performance and scale.

*   **Hardware:** The models were trained on a cluster of **2,048 NVIDIA A100 GPUs**, each equipped with **80GB of High Bandwidth Memory (HBM)**. This massive parallel processing capability was crucial. The efficient communication between these GPUs, likely facilitated by NVLink interconnects, was vital for distributed training.
*   **Performance:** The largest LLaMA-65B model was trained over **21 days** on 1.4 trillion tokens, achieving a high throughput of approximately **380 tokens/second/GPU**.
*   **Software Optimizations:** Achieving this efficiency demanded custom software engineering beyond standard frameworks:
    *   **Efficient Causal Multi-head Attention:** An optimized implementation, available in the `xformers` library, was used. This technique reduced memory usage and runtime by not storing attention weights and avoiding unnecessary computations for masked key/query scores.
    *   **Selective Activation Checkpointing:** Instead of relying on PyTorch's default autograd, a manual backward pass was implemented. This allowed for strategic saving of computationally expensive activations, optimizing the trade-off between memory and re-computation during the backward pass.
    *   **Overlapping Computation and Communication:** The training pipeline was designed to overlap communication between GPUs (e.g., `all_reduce` operations for gradient synchronization) with computation, effectively hiding network latency and keeping GPUs busy.

*   **Carbon Footprint:** In a notable move towards transparency, the paper estimated the total carbon emissions for the project to be **~2,200 tCO2eq** (tons of CO2 equivalent) before any carbon offsets. This highlights the significant environmental cost of large-scale AI training.

## Performance

Llama 1 demonstrated groundbreaking performance, proving that smaller, more efficiently trained models could outperform larger competitors. The headline result is that **LLaMA-13B outperforms the much larger GPT-3 (175B) on most benchmarks**.

### Overall Competitiveness

*   **LLaMA-13B vs. GPT-3 (175B):** The 13B model, despite being over 10x smaller, achieved superior results on most benchmarks, including reasoning, reading comprehension, and world knowledge (MMLU). This validated the "Chinchilla" scaling hypothesis that more data was more important than more parameters, beyond a certain point.
*   **LLaMA-65B vs. State-of-the-Art:** The largest model was shown to be competitive with the leading models of the era, such as Chinchilla-70B and Google's PaLM-540B, particularly in zero-shot and few-shot settings.

### Key Benchmark Results

The paper highlights several key areas where Llama 1 excelled.

| Benchmark (Setting) | LLaMA-13B | GPT-3 (175B) | LLaMA-65B | PaLM-540B |
| :--- | :---: | :---: | :---: | :---: |
| **MMLU** (5-shot, Avg) | 46.9 | 43.9 | 63.4 | 69.3 |
| **GSM8k** (Math, few-shot) | 17.8 | - | 50.9 | 56.5 |
| **HumanEval** (Code, 0-shot pass@1) | 15.8 | - | 23.7 | 26.2 |
| **HellaSwag** (Reasoning, 0-shot) | 79.2 | 78.9 | 84.2 | 83.4 |

*Scores are accuracy percentages. Higher is better.*

### Strengths and Weaknesses

*   **Strengths:** Llama 1's primary strength was its unprecedented efficiency. It proved that a smaller model could achieve state-of-the-art performance through massive data training and a well-chosen architecture. It excelled at common sense reasoning and general knowledge tasks.
*   **Weaknesses:** As a pure pre-trained base model, it lacked instruction-following capabilities and safety guardrails out-of-the-box. While the 65B model was competitive, it still did not consistently surpass the very largest proprietary models like PaLM-540B on all tasks.

## The Leak and Its Aftermath

### The Controlled Release and 4chan Leak
Meta's initial plan was a traditional, controlled release to the research community under a non-commercial license. However, just **one week** after the announcement in February 2023, the entire plan was turned upside down. A user on the anonymous forum **4chan** posted a BitTorrent link containing the full weights for all the Llama 1 models. For the first time, a state-of-the-art foundation model from a major AI lab was fully in the public domain.

### The Dual Impact
The leak's impact was immediate and twofold:
1.  **The Alarm:** It sparked intense debate about the risks of AI proliferation, with many warning of potential misuse for spam and misinformation. Meta began filing takedown requests for the leaked copies.
2.  **The Cambrian Explosion:** Simultaneously, the open-source community ignited. Because Llama 1 was so small and efficient, developers worldwide immediately began experimenting, creating iconic instruction-tuned models like **Stanford's Alpaca** and **Vicuna** that demonstrated the base model's incredible potential.

### The Unintended Legacy
The leak, while unintentional, became a massive, real-world experiment that proved giving the community access to powerful base models leads to an explosion of innovation. This widespread, positive impact almost certainly influenced Meta's future strategy, leading them to fully and officially open-source Llama 2 just five months later.

## Legacy and Impact

Llama 1's release was a watershed moment for open-source AI. It fundamentally shifted the landscape by proving that smaller, efficiently trained models could outperform larger, proprietary counterparts like GPT-3. This established a new paradigm where training quality and data curation were understood to be as important as sheer model size.

**Key Impacts**:

- **Architectural Standards**: It popularized a set of architectural choices—**RMSNorm**, **SwiGLU**, and **RoPE**—that became foundational for many subsequent open-source models.
- **Democratization via "The Leak"**: Though initially released under a restrictive, research-only license, the model's weights were leaked online in March 2023. This unintended distribution massively accelerated the democratization of advanced AI, putting powerful foundation models into the hands of the global open-source community.
- **An Explosion of Fine-Tuning**: As a base model with no official chat variant, Llama 1 became the foundation for a vibrant ecosystem of community-driven chat and instruction-tuned models. Notable examples include:
  - **Alpaca (Stanford)**: An early and influential instruction-following model.
  - **Vicuna**: A high-quality chat model that demonstrated the power of community fine-tuning.

The success of Llama 1 and the community's enthusiastic adoption directly set the stage for Meta's subsequent release of Llama 2 as a fully open, commercially viable model, cementing the role of open-source in the future of AI development.

## Key Figures

The Llama 1 project was the result of work from a dedicated research team and the strategic guidance of Meta's AI leadership.

*   **Hugo Touvron (Lead Author):** As the first author on the Llama 1 paper, Touvron played a primary role in the hands-on research, experimentation, and development of the model.
*   **Guillaume Lample (Lead Author):** The last author on the paper, signifying his role as the senior researcher who supervised the project. Lample was a key leader of the Llama 1 effort and has been a public voice explaining its development. He has since co-founded the influential AI company Mistral AI.
*   **Joelle Pineau (Head of FAIR):** As VP of AI Research, Pineau directly led the development and release of the Llama 1 project. She was instrumental in driving the lab's strategic focus on model efficiency and open science.
*   **Yann LeCun (Chief AI Scientist):** While stating his own role in Llama 1 was "very indirect," LeCun's long-standing and vocal advocacy for open-source AI was crucial in shaping the environment at Meta that allowed the project to flourish. He was a primary proponent for the full open-sourcing of its successor, Llama 2.

## Sources

### Official
- [LLaMA: Open and Efficient Foundation Language Models (Paper)](https://arxiv.org/abs/2302.13971)
- [Meta AI Research Publication](https://ai.meta.com/research/publications/llama-open-and-efficient-foundation-language-models/)

### HuggingFace
- Community conversions: [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf), [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)
- Note: Not officially released by Meta (research-only). Community versions are format conversions of the same weights.

### Related
- [Chinchilla Scaling Laws (DeepMind)](https://arxiv.org/abs/2203.15556)
