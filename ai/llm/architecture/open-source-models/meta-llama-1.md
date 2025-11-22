# Llama 1 (LLaMA)

**Release Date**: February 24, 2023

The original LLaMA (Large Language Model Meta AI) - Meta's first open foundation model that proved smaller, well-trained models could outperform much larger proprietary models.

## Model Variants

- **7B**: 7 billion parameters
- **13B**: 13 billion parameters
- **33B**: 33 billion parameters
- **65B**: 65 billion parameters

## Architecture

**Base Design**: Auto-regressive decoder-only transformer

**Key Components**:
- **Normalization**: RMSNorm pre-normalization (instead of post-normalization)
- **Activation**: SwiGLU activation function (from PaLM)
- **Position Encoding**: Rotary Embeddings (RoPE, not absolute positional embeddings)
- **Attention**: Multi-Head Attention (MHA)
- **FFN Dimension**: 2/3 × 4d instead of 4d (as in PaLM)

## Training Details

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

## Performance

- **LLaMA-13B outperformed GPT-3 175B** on most benchmarks
- **LLaMA-65B competitive** with Chinchilla-70B and PaLM-540B

This was groundbreaking: a 13B model beating a 175B model showed that training quality and data matter more than sheer size.

## Significance

- **First major open-source model from Meta**
- **Proved open models could compete with proprietary ones**
- **Established architectural patterns**: RMSNorm, SwiGLU, RoPE became standard
- **Sparked explosion of derivative models**: Alpaca, Vicuna, and countless fine-tunes

## Access and Distribution

- **Initial Release**: Research-only, application required
- **Access**: Case-by-case basis to academic researchers, government, civil society, academia, and industry research labs
- **Inference Code**: Released as open-source (GPLv3 license)
- **The Leak**: March 3, 2023 - Posted on 4chan via BitTorrent, democratizing access

## Variants

**No Official Chat/Instruct Versions**: Llama 1 was base-only. Community created fine-tuned versions:
- **Alpaca** (Stanford) - Instruction-tuned using Self-Instruct method
- **Vicuna** - Community chat model
- Many others

Meta didn't release official chat variants until Llama 2.

## Links

- **Paper**: [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- **Meta AI Research**: [LLaMA Publication](https://ai.meta.com/research/publications/llama-open-and-efficient-foundation-language-models/)
- **Hugging Face**: Not officially released by Meta (research-only release requiring application)
  - Community conversions available: [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf), [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)
  - Note: Community versions are format conversions (PyTorch .pth → HuggingFace Transformers format) of the same official weights, not different models. Early conversions like decapoda-research may be outdated.

## Legacy

Llama 1 changed everything. It showed the world that:
1. Open-source could compete with closed models
2. Smaller, well-trained models beat larger, poorly-trained ones
3. The AI community would embrace and build on open foundations

The leak, while not Meta's intention, accelerated the democratization of AI and set the stage for Llama 2's fully open release.
