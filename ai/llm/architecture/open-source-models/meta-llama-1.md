# Llama 1 (LLaMA)

**Release Date**: February 24, 2023

## Links

- **Paper**: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- **Meta AI Research**: [LLaMA Publication](https://ai.meta.com/research/publications/llama-open-and-efficient-foundation-language-models/)
- **Hugging Face**: Not officially released by Meta (research-only release requiring application)
  - Community conversions available: [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf), [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)
  - Note: Community versions are format conversions (PyTorch .pth → HuggingFace Transformers format) of the same official weights, not different models. Early conversions like decapoda-research may be outdated.

The original LLaMA (Large Language Model Meta AI) - Meta's first open foundation model that proved smaller, well-trained models could outperform much larger proprietary models.

## Architecture

**Common Components**:

- **Base Design**: Auto-regressive decoder-only transformer
- **Normalization**: RMSNorm pre-normalization
- **Activation**: SwiGLU
- **Position Encoding**: Rotary Embeddings (RoPE)
- **Attention**: Multi-Head Attention (MHA)
- **FFN Dimension**: 2/3 × 4d (as in PaLM)
- **Vocabulary**: 32K tokens (SentencePiece tokenizer)

**Model Specifications**:

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

1.  **Chinchilla Scaling Laws:** The Llama paper explicitly followed the guidance from DeepMind's [Chinchilla paper (2022)](https://arxiv.org/abs/2203.15556). This research argued that for optimal performance, model size and training data size should be scaled proportionally. Llama was the first major project to prove this theory in practice, intentionally training smaller models on far more data than previous models like GPT-3.

2.  **Hardware Efficiency:** The specific numbers (e.g., a `dim` of 4096, a `head_dim` of 128) are chosen to maximize training throughput on the underlying hardware (NVIDIA A100 GPUs). Dimensions are typically set to multiples of 64 or 128 to align with the architecture of GPU cores and memory, making matrix calculations significantly faster.

3.  **Iterative Experimentation:** While guided by scaling laws, the final ratios of depth vs. width were likely refined through extensive, small-scale experiments. Teams typically run hundreds of tests on smaller models to find a "sweet spot" before committing to a multi-million dollar training run.

These choices are highly sensitive and interdependent. Changing one value requires re-balancing the others to stay within a specific parameter budget, all while trying to maximize performance and efficiency.


## Training Details

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

## Performance

- **LLaMA-13B outperformed GPT-3 175B** on most benchmarks
- **LLaMA-65B competitive** with Chinchilla-70B and PaLM-540B

This was groundbreaking: a 13B model beating a 175B model showed that training quality and data matter more than sheer size.

## Legacy and Impact

Llama 1's release was a watershed moment for open-source AI. It fundamentally shifted the landscape by proving that smaller, efficiently trained models could outperform larger, proprietary counterparts like GPT-3. This established a new paradigm where training quality and data curation were understood to be as important as sheer model size.

**Key Impacts**:
- **Architectural Standards**: It popularized a set of architectural choices—**RMSNorm**, **SwiGLU**, and **RoPE**—that became foundational for many subsequent open-source models.
- **Democratization via "The Leak"**: Though initially released under a restrictive, research-only license, the model's weights were leaked online in March 2023. This unintended distribution massively accelerated the democratization of advanced AI, putting powerful foundation models into the hands of the global open-source community.
- **An Explosion of Fine-Tuning**: As a base model with no official chat variant, Llama 1 became the foundation for a vibrant ecosystem of community-driven chat and instruction-tuned models. Notable examples include:
  - **Alpaca (Stanford)**: An early and influential instruction-following model.
  - **Vicuna**: A high-quality chat model that demonstrated the power of community fine-tuning.

The success of Llama 1 and the community's enthusiastic adoption directly set the stage for Meta's subsequent release of Llama 2 as a fully open, commercially viable model, cementing the role of open-source in the future of AI development.
