# Llama 2

**Release Date**: July 18, 2023

Meta's first truly open-source LLM with commercial license, released in partnership with Microsoft. Llama 2 marked the shift from research-only to fully accessible AI.

## Model Variants

- **7B**: 7 billion parameters
- **13B**: 13 billion parameters
- **70B**: 70 billion parameters

**All sizes available in two versions**:
- **Base**: Pretrained foundation models
- **Chat**: Fine-tuned with SFT + RLHF for conversation

## Architecture

**Base Design**: Decoder-only transformer with optimizations

**Key Components**:
- **Normalization**: RMSNorm pre-normalization
- **Activation**: SwiGLU activation function
- **Position Encoding**: RoPE (Rotary Position Embeddings)
- **Attention**:
  - 7B and 13B: Multi-Head Attention (MHA)
  - **70B: Grouped-Query Attention (GQA)** - First major production use!

## Training Details

- **Tokens**: 2 trillion tokens (2x Llama 1)
- **Context Window**: 4,096 tokens (2x Llama 1's 2K)
- **Vocabulary**: 32K tokens (SentencePiece tokenizer, same as Llama 1)

## Fine-Tuning (Chat Variants)

**Supervised Fine-Tuning (SFT)**:
- Trained on instruction-following datasets
- Optimized for dialogue

**Reinforcement Learning from Human Feedback (RLHF)**:
- Reward model trained on human preferences
- PPO (Proximal Policy Optimization) for alignment
- Focus on helpfulness and safety

**Result**: Llama 2-Chat models competitive with ChatGPT for many use cases

## Significance

- **First major open-source model to rival proprietary models** with commercial license
- **Introduced optimizations that became standard**: RMSNorm, SwiGLU, RoPE
- **70B variant pioneered GQA in production LLMs** - proved efficiency gains at scale
- **Democratization**: No application needed, anyone can use
- **Partnership with Microsoft**: Integrated into Azure, widespread cloud availability

## Key Innovations

1. **GQA in 70B**: Reduced memory and compute while maintaining quality
2. **Commercial License**: Unlike Llama 1, fully open for business use
3. **Chat Variants**: First official Meta fine-tuned conversational models
4. **RLHF at Scale**: Demonstrated effective alignment techniques

## Performance

- Llama 2-70B competitive with GPT-3.5 on many benchmarks
- Llama 2-Chat variants approached ChatGPT quality
- Open community fine-tuned countless specialized versions

## Links

- **Paper**: [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
- **Blog**: [Meta and Microsoft Introduce the Next Generation of Llama](https://ai.meta.com/blog/llama-2/)
- **Hugging Face**:
  - Base: [Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf), [Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf), [Llama-2-70b-hf](https://huggingface.co/meta-llama/Llama-2-70b-hf)
  - Chat: [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), [Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf), [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)

## Community Impact

Llama 2's open release triggered an explosion of derivative works:
- Specialized fine-tunes for medical, legal, coding domains
- Quantized versions for consumer hardware (GGUF, GPTQ)
- Integration into countless products and services
- Academic research on alignment, safety, and capabilities

## Legacy

Llama 2 proved that open-source AI could be:
1. **Commercially viable**: Businesses could build on it
2. **Competitive**: Matched proprietary models
3. **Responsible**: Included safety measures (RLHF alignment)
4. **Accessible**: No barriers to entry

This set the stage for the explosive growth of open AI in 2024.
