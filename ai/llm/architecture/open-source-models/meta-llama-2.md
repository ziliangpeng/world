# Llama 2

**Release Date**: July 18, 2023

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

## Legacy and Impact

The release of Llama 2 was a pivotal moment for the open-source AI movement, proving that open models could be directly competitive with and commercially viable against closed, proprietary systems.

*   **Established a New Open-Source Standard:** With its commercial-friendly license, high performance rivaling models like GPT-3.5, and official safety-aligned chat variants, Llama 2 became the go-to foundation for countless projects and businesses. It proved that open-source AI was ready for production.
*   **Pioneered GQA at Scale:** The 70B model's use of Grouped-Query Attention was the first major production deployment of the technique, demonstrating its efficiency benefits and paving the way for its adoption in later models.
*   **Accelerated the Fine-Tuning Ecosystem:** The fully accessible and high-quality base models triggered an explosion of innovation. The community created thousands of specialized fine-tunes for specific domains (e.g., medical, legal, coding) and developed new quantization techniques (GGUF, GPTQ) to run models on consumer hardware.
*   **Democratized Access to AI:** By removing the application process required for Llama 1 and partnering with Microsoft for broad cloud availability on Azure, Llama 2 made state-of-the-art AI truly accessible to individual developers, researchers, and businesses of all sizes.
