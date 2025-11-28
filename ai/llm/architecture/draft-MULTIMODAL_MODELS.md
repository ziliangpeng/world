# Multimodal Models

A comprehensive overview of multimodal language models that can process and understand multiple modalities such as vision, audio, and text.

## Educational Resources

- [Multi-Modal Pretraining Notes](https://github.com/hu-po/docs/tree/main/2024.03.17.multi_modal_pretraining) - Deep dive into Apple MM1 and multimodal pretraining methods
- [Building Multimodal Models](https://github.com/hu-po/docs/tree/main/2024.05.19.building_multimodal_models) - Architectural approaches and design patterns

## Timeline: All Multimodal Models

| Date | Model | Organization | Type | Status | Notes | Significance |
|------|-------|--------------|------|--------|-------|--------------|
| Jan 2022 | 📜 [BLIP](https://arxiv.org/abs/2201.12086) | Salesforce | Vision-Language | Open | Vision-language pre-training with bootstrapping | 🔥 Foundation - bootstrapping methodology |
| Jan 2023 | 📜 [BLIP-2](https://arxiv.org/abs/2301.12597) | Salesforce | Vision-Language | Open | Efficient with frozen image encoders and LLMs | 🔥 Q-Former architecture, frozen encoders |
| Apr 2023 | 📜 [LLaVA](https://arxiv.org/abs/2304.08485) | Microsoft/Wisconsin | Vision-Language | Open | 7B, 13B, 34B - Visual instruction tuning | 🔥 Visual instruction tuning paradigm |
| Apr 2023 | 📜 [MiniGPT-4](https://arxiv.org/abs/2304.10592) | KAUST | Vision-Language | Open | GPT-4-like vision with minimal training | 🥱 Adapter approach |
| Apr 2023 | 📜 [LLaMA-Adapter V2](https://arxiv.org/abs/2304.15010) | Shanghai AI Lab | Vision-Language | Open | Parameter-efficient multimodal fine-tuning | 🥱 PEFT variation |
| May 2023 | 📜 [InstructBLIP](https://arxiv.org/abs/2305.06500) | Salesforce | Vision-Language | Open | Instruction-tuning for vision-language models | 🥱 Instruction tuning BLIP-2 |
| Aug 2023 | 📜 [IDEFICS](https://arxiv.org/abs/2306.16527) | Hugging Face | Vision-Language | Open | 9B, 80B - Flamingo-like open vision-language model | 🥱 Flamingo replication |
| Oct 2023 | 📜 [Qwen-VL](https://arxiv.org/abs/2308.12966) | Alibaba | Vision-Language | Open | Vision-language with grounding capabilities | 🔥 Grounding/localization capabilities |
| Oct 2023 | [Fuyu](https://www.adept.ai/blog/fuyu-8b) | Adept | Vision-Language | Open | 8B - Multimodal for digital agents | 🔥 Optimized for UI/agent tasks |
| Nov 2023 | 📜 [Qwen-Audio](https://arxiv.org/abs/2311.07919) | Alibaba | Audio-Language | Open | Universal audio understanding, 30+ tasks | 🔥 First major open audio LLM |
| Nov 2023 | 📜 [CogVLM](https://arxiv.org/abs/2311.03079) | Tsinghua | Vision-Language | Open | 8B-17B - Vision understanding | 🥱 Standard VLM |
| Nov 2023 | 📜 [Mirasol3B](https://arxiv.org/abs/2311.05698) | Google | Vision-Audio-Language | Proprietary | 3B - Time-aligned and contextual modalities (paper only) | 🔥 Novel temporal alignment architecture |
| Mar 2024 | 📜 [DeepSeek-VL](https://arxiv.org/abs/2403.05525) | DeepSeek | Vision-Language | Open | 1.3B, 7B - Vision-language model | 🥱 Standard VLM |
| Mar 2024 | 📜 [MM1](https://arxiv.org/abs/2403.09611) | Apple | Vision-Language | Proprietary | Up to 30B - Dense and MoE variants (paper only) | 🔥 Best pretraining methodology documentation |
| May 2024 | 📜 [PaliGemma](https://arxiv.org/abs/2407.07726) | Google | Vision-Language | Open | Vision-language model based on Gemma | 🥱 Gemma + vision |
| May 2024 | 📜 [Chameleon](https://arxiv.org/abs/2405.09818) | Meta | Vision-Language | Open | 7B, 30B - Mixed-modal early-fusion foundation models | 🔥 Revolutionary early-fusion architecture |
| May 2024 | 📜 [Phi-3-vision](https://arxiv.org/abs/2404.14219) | Microsoft | Vision-Language | Open | 4.2B - Multimodal, 128K context | 🥱 Phi + vision |
| May 2024 | 📜 [CogVLM2](https://arxiv.org/abs/2408.16500) | Tsinghua | Vision-Language | Open | 8B-17B - Enhanced vision understanding | 🥱 CogVLM iteration |
| Jul 2024 | 📜 [Qwen2-Audio](https://arxiv.org/abs/2407.10759) | Alibaba | Audio-Language | Open | Voice chat + audio analysis modes | 🥱 Qwen-Audio v2 |
| Aug 2024 | [Phi-3.5-vision-instruct](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/discover-the-new-multi-lingual-high-quality-phi-3-5-slms/ba-p/4225280) | Microsoft | Vision-Language | Open | 4.2B - Multimodal, 128K context | 🥱 Phi-3-vision iteration |
| Sep 2024 | 📜 [Llama 3.2 Vision](https://arxiv.org/abs/2407.21783) | Meta | Vision-Language | Open | 11B, 90B - First Llama multimodal models | 🔥 Meta's entry, flagship quality |
| Sep 2024 | 📜 [Qwen2-VL](https://arxiv.org/abs/2409.12191) | Alibaba | Vision-Language | Open | 2B, 7B - Any resolution vision | 🔥 Dynamic resolution/adaptive tokens |
| Sep 2024 | 📜 [Pixtral 12B](https://arxiv.org/abs/2410.07073) | Mistral AI | Vision-Language | Open | 12B + 400M vision encoder | 🥱 Mistral + vision |
| Nov 2024 | 📜 [JanusFlow](https://arxiv.org/abs/2411.07975) | DeepSeek | Vision Generation | Open | 1.3B - Unified image understanding + generation | 🔥 Unified understanding + generation |
| Nov 2024 | [Pixtral Large](https://mistral.ai/news/pixtral-large) | Mistral AI | Vision-Language | Open | 123B + 1B vision encoder | 🥱 Pixtral scaled up |
| Dec 2024 | 📜 [DeepSeek-VL2](https://arxiv.org/abs/2412.10302) | DeepSeek | Vision-Language | Open | 3.37B-27.5B MoE - Advanced multimodal | 🥱 DeepSeek-VL iteration |
| Jan 2025 | Janus-Pro | DeepSeek | Vision | Open | 1.5B, 7B - Understanding + generation | 🥱 JanusFlow iteration |
| Jan 2025 | Qwen2.5-VL | Alibaba | Vision-Language | Open | 3B-72B - Multi-resolution vision | 🥱 Qwen2-VL iteration |
| Feb 2025 | 📜 [Phi-4-multimodal-instruct](https://arxiv.org/abs/2503.01743) | Microsoft | Vision-Audio-Language | Open | 5.6B - Speech + vision + text, 20+ languages | 🔥 True multimodal (speech+vision) in SLM |
| Mar 2025 | 📜 [Gemma 3](https://arxiv.org/abs/2503.19786) | Google | Vision-Language | Open | 270M-27B - Multimodal, 140+ languages | 🥱 Gemma + multimodal |
| Mar 2025 | Qwen2.5-Omni | Alibaba | Omnimodal | Open | 7B - Text/image/video/audio | 🔥 Omnimodal integration |
| Apr 2025 | 📜 [Llama 4](https://arxiv.org/abs/2510.12178) | Meta | Vision-Language | Open | Scout 17B, Maverick, Behemoth 288B - Multimodal | 🔥 10M context, MoE multimodal |
| Sep 2025 | Qwen3-Omni | Alibaba | Omnimodal | Open | Real-time multimodal with speech generation | 🔥 Real-time speech generation |
| Oct 2025 | 📜 [DeepSeek-OCR](https://arxiv.org/abs/2510.18234) | DeepSeek | Vision-OCR | Open | 3B MoE (570M active) - Optical context compression | 🔥 Novel OCR context compression

## Proprietary/API-Only Models

| Date | Model | Organization | Type | Notes | Significance |
|------|-------|--------------|------|-------|--------------|
| Feb 2024 | Gemini 1.5 | Google | Vision-Language-Audio | 1M+ token context, API-only | 🔥 Extreme long-context frontier |
| May 2024 | GPT-4o | OpenAI | Vision-Audio-Language | Omnimodal, API-only | 🔥 Native omnimodal, real-time |

## Major Model Families

### Meta Llama Vision Models

**Llama 3.2 Vision** (September 2024)
- 11B and 90B variants
- First multimodal Llama models
- Vision + language understanding
- [Llama Documentation](../open-source-models/meta-llama.md)

**Llama 4** (April 2025)
- Scout 17B, Maverick, Behemoth 288B
- MoE architecture with multimodal capabilities
- 10M token context window
- [Llama Documentation](../open-source-models/meta-llama.md)

### Alibaba Qwen Multimodal Series

**Vision Models:**
- **Qwen-VL** (Oct 2023) - Vision-language with grounding
- **Qwen2-VL** (Sep 2024) - 2B, 7B with any resolution
- **Qwen2.5-VL** (Jan 2025) - 3B-72B multi-resolution

**Audio Models:**
- **Qwen-Audio** (Nov 2023) - Universal audio, 30+ tasks
- **Qwen2-Audio** (Jul 2024) - Voice chat + analysis

**Omnimodal:**
- **Qwen2.5-Omni** (Mar 2025) - 7B text/image/video/audio
- **Qwen3-Omni** (Sep 2025) - Real-time with speech generation

[Qwen Documentation](../open-source-models/qwen.md)

### DeepSeek Vision & Generation

**Vision-Language:**
- **DeepSeek-VL** (Mar 2024) - 1.3B, 7B
- **DeepSeek-VL2** (Dec 2024) - 3.37B-27.5B MoE

**Image Generation + Understanding:**
- **JanusFlow** (Nov 2024) - 1.3B unified model
- **Janus-Pro** (Jan 2025) - 1.5B, 7B enhanced

**OCR Specialized:**
- **DeepSeek-OCR** (Oct 2025) - 3B MoE for optical context

[DeepSeek Documentation](../open-source-models/deepseek.md)

### Google Gemma Vision

- **PaliGemma** (May 2024) - Vision-language model
- **Gemma 3** (Mar 2025) - 270M-27B multimodal, 140+ languages

[Gemma Documentation](../open-source-models/google-gemma.md)

### Microsoft Phi Vision

- **Phi-3-vision** (May 2024) - 4.2B, 128K context
- **Phi-3.5-vision-instruct** (Aug 2024) - 4.2B enhanced
- **Phi-4-multimodal-instruct** (Feb 2025) - 5.6B speech+vision+text

[Phi Documentation](../open-source-models/microsoft-phi.md)

### Mistral Pixtral Series

- **Pixtral 12B** (Sep 2024) - 12B + 400M vision encoder
- **Pixtral Large** (Nov 2024) - 123B + 1B vision encoder

[Mistral Documentation](../open-source-models/mistral-mixtral.md)

### Salesforce BLIP Family

**BLIP** (January 2022)
- Vision-language pre-training with bootstrapping
- Foundation for many subsequent models

**BLIP-2** (January 2023)
- Efficient with frozen image encoders and LLMs
- Q-Former architecture for cross-modal alignment

**InstructBLIP** (May 2023)
- Instruction-tuning for vision-language tasks
- Better instruction following

### Other Notable Vision-Language Models

**LLaVA** (April 2023)
- Microsoft/Wisconsin collaboration
- 7B, 13B, 34B variants
- Visual instruction tuning methodology
- 📜 [Paper](https://arxiv.org/abs/2304.08485)

**MiniGPT-4** (April 2023)
- KAUST research
- GPT-4-like vision with minimal training
- 📜 [Paper](https://arxiv.org/abs/2304.10592)

**LLaMA-Adapter V2** (April 2023)
- Shanghai AI Lab
- Parameter-efficient multimodal fine-tuning
- 📜 [Paper](https://arxiv.org/abs/2304.15010)

**IDEFICS** (August 2023)
- Hugging Face
- 9B, 80B variants
- Open Flamingo-like architecture
- 📜 [Paper](https://arxiv.org/abs/2306.16527)

**Chameleon** (May 2024)
- Meta AI
- 7B, 30B variants
- Mixed-modal early-fusion foundation models
- Research license, weights available
- 📜 [Paper](https://arxiv.org/abs/2405.09818)

**CogVLM Series** (Tsinghua)
- **CogVLM** (Nov 2023) - 8B-17B vision understanding
- **CogVLM2** (May 2024) - Enhanced version
- 📜 [CogVLM Paper](https://arxiv.org/abs/2311.03079)
- 📜 [CogVLM2 Paper](https://arxiv.org/abs/2408.16500)

**Fuyu** (October 2023)
- Adept AI
- 8B parameters
- Optimized for digital agents
- [Blog Post](https://www.adept.ai/blog/fuyu-8b)

## Research-Only Models (Proprietary)

### Apple MM1 (March 2024)

**Methods, Analysis & Insights from Multimodal LLM Pre-training**

- Up to 30B parameters
- Dense and MoE variants
- Comprehensive pretraining methodology
- **Paper only** - weights not released
- 📜 [Paper](https://arxiv.org/abs/2403.09611)
- [Apple ML Research](https://machinelearning.apple.com/research/mm1-methods-analysis-insights)
- [Educational Notes](https://github.com/hu-po/docs/tree/main/2024.03.17.multi_modal_pretraining)

**Key Components:**
- DFN5B-CLIP-ViT-H-14 vision encoder
- Novel pretraining strategies
- Extensive architecture analysis

### Google Mirasol3B (November 2023)

**A Multimodal Autoregressive Model for Time-Aligned and Contextual Modalities**

- 3B parameters
- Time-aligned multimodal architecture
- Vision + audio + language
- **Paper only** - no weights, code, or API
- 📜 [Paper](https://arxiv.org/abs/2311.05698)
- [Google DeepMind Blog](https://blog.research.google/2023/11/scaling-multimodal-understanding-to.html)
- [Educational Notes](https://github.com/hu-po/docs/tree/main/2024.05.19.building_multimodal_models)

**Unofficial Implementation:**
- [lucidrains/mirasol-pytorch](https://github.com/lucidrains/mirasol-pytorch) - Community PyTorch implementation

## Architectural Patterns

### Vision Integration Approaches

1. **Cross-Attention** (BLIP-2, Flamingo-style)
   - Frozen vision encoder + frozen LLM
   - Learnable Q-Former for cross-modal alignment
   - Parameter efficient

2. **Early Fusion** (Chameleon)
   - Unified token space for all modalities
   - Joint training from scratch
   - Better cross-modal reasoning

3. **Adapter-Based** (LLaMA-Adapter V2)
   - Parameter-efficient fine-tuning
   - Add small adapter layers
   - Preserve base model performance

4. **Vision Encoder + Projection** (LLaVA)
   - CLIP vision encoder
   - Simple linear projection
   - Fine-tune language model

### Audio Processing

1. **Spectrogram Representations** (Qwen-Audio, Mirasol3B)
   - Convert audio to visual spectrograms
   - Process with vision-like architectures
   - Time-frequency domain features

2. **Dedicated Audio Encoders** (Phi-4-multimodal)
   - Specialized audio feature extraction
   - Speech and audio understanding
   - Multi-language support

### Context Management

- **Average Pooling** - Compress vision tokens
- **Q-Former** - Cross-attention based compression (BLIP-2)
- **Adaptive Resolution** - Dynamic token allocation (Qwen2-VL)
- **Long Context** - Extended context windows (Gemini 1.5: 1M+ tokens)

## Key Innovations

### Instruction Tuning for Vision (LLaVA, InstructBLIP)
- Visual instruction-response pairs
- Better instruction following
- Improved zero-shot generalization

### Grounding Capabilities (Qwen-VL)
- Locate objects in images
- Bounding box generation
- Spatial reasoning

### Multi-Resolution Support (Qwen2-VL)
- Any resolution input
- Dynamic token allocation
- Better efficiency

### Unified Generation + Understanding (JanusFlow, Janus-Pro)
- Single model for both tasks
- Image understanding and generation
- Shared representation learning

### Time-Aligned Processing (Mirasol3B)
- Synchronize vision, audio, language
- Temporal alignment
- Contextual modality understanding

## Resources & Learning

### Educational Materials
- [Multi-Modal Pretraining - Apple MM1 Deep Dive](https://github.com/hu-po/docs/tree/main/2024.03.17.multi_modal_pretraining)
- [Building Multimodal Models - Architecture Guide](https://github.com/hu-po/docs/tree/main/2024.05.19.building_multimodal_models)

### Major Conferences & Workshops
- **ICLR** - International Conference on Learning Representations
- **NeurIPS** - Neural Information Processing Systems
- **CVPR** - Computer Vision and Pattern Recognition
- **ACL** - Association for Computational Linguistics

### Key Research Labs
- **Meta FAIR** - Fundamental AI Research
- **Google DeepMind** - Gemini, Mirasol
- **Microsoft Research** - LLaVA, Phi
- **Alibaba DAMO Academy** - Qwen series
- **DeepSeek** - DeepSeek-VL, Janus
- **Salesforce Research** - BLIP family
- **Tsinghua University** - CogVLM

## Status Legend

- **Open** - Model weights publicly available (may have specific licenses)
- **Proprietary** - Weights not available; paper-only or API-only access
- 📜 - arXiv paper available

## See Also

- [Open Source Models](OPEN_SOURCE_MODELS.md) - Main index of all open-source LLMs
- [Important LLM Papers](important-llm-papers.md) - Foundational research papers
- [Architectural Patterns](architectural-patterns.md) - Common LLM architecture components
