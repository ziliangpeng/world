# Open Source Models

Major open source LLM families with full architecture documentation.

## Industry Timeline: All Models

| Date | Model | Notes |
|------|-------|-------|
| Feb 2023 | [Llama 1](https://arxiv.org/abs/2302.13971) (Meta) | 7B, 13B, 33B, 65B - First major open LLM |
| Jul 2023 | [Llama 2](https://arxiv.org/abs/2307.09288) (Meta) | 7B, 13B, 70B - Fully open-source with commercial license |
| Aug 2023 | [Code Llama](https://arxiv.org/abs/2308.12950) (Meta) | 7B, 13B, 34B - Code specialized |
| Sep 2023 | [Mistral 7B](https://arxiv.org/abs/2310.06825) (Mistral AI) | Dense 7B, Sliding Window Attention |
| Dec 2023 | [Mixtral 8x7B](https://arxiv.org/abs/2401.04088) (Mistral AI) | MoE, 46.7B total, 12.9B active - First open MoE |
| Dec 2023 | [Llama Guard](https://arxiv.org/abs/2312.06674) (Meta) | 7B safety model |
| Jan 2024 | Code Llama 70B (Meta) | 70B variant added |
| Feb 2024 | Mistral Small (Mistral AI) | Dense 22B, low latency |
| Apr 2024 | [Llama 3](https://arxiv.org/abs/2407.21783) (Meta) | 8B, 70B |
| Apr 2024 | [Llama Guard 2](https://arxiv.org/abs/2407.21783) (Meta) | 8B safety model |
| Apr 2024 | Mixtral 8x22B (Mistral AI) | MoE, 141B total, 39B active, 64K context |
| May 2024 | Codestral (Mistral AI) | Dense 22B, code specialized, 80+ languages |
| Jul 2024 | [Llama 3.1](https://arxiv.org/abs/2407.21783) (Meta) | 8B, 70B, 405B - First 400B+ open model |
| Jul 2024 | [Llama Guard 3](https://arxiv.org/abs/2407.21783) (Meta) | 1B, 12B safety models |
| Jul 2024 | Mistral NeMo (Mistral AI) | Dense 12B, NVIDIA collab, Tekken tokenizer, 128K context |
| Jul 2024 | Mistral Large 2 (Mistral AI) | Dense 123B, flagship model, 128K context |
| Jul 2024 | Codestral Mamba (Mistral AI) | SSM 7B, State Space Model, 256K context |
| Jul 2024 | Mathstral 7B (Mistral AI) | Dense 7B, math/STEM specialized, 32K context |
| Sep 2024 | [Llama 3.2](https://arxiv.org/abs/2407.21783) (Meta) | 1B, 3B, 11B Vision, 90B Vision - First Llama multimodal |
| Sep 2024 | [Pixtral 12B](https://arxiv.org/abs/2410.07073) (Mistral AI) | Multimodal, 12B + 400M vision encoder |
| Oct 2024 | Ministral 3B (Mistral AI) | Dense 3B, edge/on-device, 128K context |
| Oct 2024 | Ministral 8B (Mistral AI) | Dense 8B, edge/on-device, 128K context |
| Nov 2024 | Prompt Guard 2 (Meta) | 86M, 22M - Injection attack prevention |
| Nov 2024 | Mistral Large 24.11 (Mistral AI) | Dense 123B, improved long context, 131K context |
| Nov 2024 | Pixtral Large (Mistral AI) | Multimodal, 123B + 1B vision encoder, frontier multimodal |
| Nov 2024 | [INTELLECT-1](https://arxiv.org/abs/2412.01152) (Prime Intellect) | 10B - First decentralized training across 3 continents |
| Dec 2024 | [Llama 3.3](https://arxiv.org/abs/2407.21783) (Meta) | 70B |
| Apr 2025 | [Llama 4](https://arxiv.org/abs/2510.12178) (Meta) | Scout 17B, Maverick, Behemoth 288B - MoE, multimodal, 10M context |
| May 2025 | [INTELLECT-2](https://arxiv.org/abs/2505.07291) (Prime Intellect) | 32B - First decentralized RL training |

---

## [Meta Llama Series](open-source-models/meta-llama.md)

| Date | Model | Notes |
|------|-------|-------|
| Feb 2023 | Llama 1 | 7B, 13B, 33B, 65B |
| Jul 2023 | Llama 2 | 7B, 13B, 70B |
| Aug 2023 | Code Llama | 7B, 13B, 34B - Base/Python/Instruct |
| Dec 2023 | Llama Guard | 7B safety model |
| Jan 2024 | Code Llama 70B | 70B variant added |
| Apr 2024 | Llama 3 | 8B, 70B |
| Apr 2024 | Llama Guard 2 | 8B safety model |
| Jul 2024 | Llama 3.1 | 8B, 70B, 405B |
| Jul 2024 | Llama Guard 3 | 1B, 12B safety models |
| Sep 2024 | Llama 3.2 | 1B, 3B, 11B Vision, 90B Vision |
| Nov 2024 | Prompt Guard 2 | 86M, 22M - Injection attack prevention |
| Dec 2024 | Llama 3.3 | 70B |
| Apr 2025 | Llama 4 | Scout 17B, Maverick, Behemoth 288B - MoE, multimodal, 10M context |

## [Mistral/Mixtral](open-source-models/mistral-mixtral.md)

| Date | Model | Notes |
|------|-------|-------|
| Sep 2023 | Mistral 7B | Dense 7B, Sliding Window Attention |
| Dec 2023 | Mixtral 8x7B | MoE, 46.7B total, 12.9B active |
| Feb 2024 | Mistral Small | Dense 22B, low latency |
| Apr 2024 | Mixtral 8x22B | MoE, 141B total, 39B active, 64K context |
| May 2024 | Codestral | Dense 22B, code specialized, 80+ languages |
| Jul 2024 | Mistral NeMo | Dense 12B, NVIDIA collab, Tekken tokenizer, 128K context |
| Jul 2024 | Mistral Large 2 | Dense 123B, flagship model, 128K context |
| Jul 2024 | Codestral Mamba | SSM 7B, State Space Model, 256K context |
| Jul 2024 | Mathstral 7B | Dense 7B, math/STEM specialized, 32K context |
| Sep 2024 | Pixtral 12B | Multimodal, 12B + 400M vision encoder |
| Oct 2024 | Ministral 3B | Dense 3B, edge/on-device, 128K context |
| Oct 2024 | Ministral 8B | Dense 8B, edge/on-device, 128K context |
| Nov 2024 | Mistral Large 24.11 | Dense 123B, improved long context, 131K context |
| Nov 2024 | Pixtral Large | Multimodal, 123B + 1B vision encoder, frontier multimodal |

## [Qwen Series](open-source-models/qwen.md)
- Qwen 2.5 (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B)
- Qwen 3 (Dense and MoE variants with 128 experts)

## [DeepSeek](open-source-models/deepseek.md)
- DeepSeek-V2 (236B total, 21B active)
- DeepSeek-V3 (671B total, 37B active)

## [Google Gemma](open-source-models/google-gemma.md)
- Gemma 1 (2B, 7B)
- Gemma 2 (2B, 9B, 27B)

## [Microsoft Phi](open-source-models/microsoft-phi.md)
- Phi-3 Family (3.8B mini, 7B small, 14B medium, 4.2B vision)
- Phi-4 (14B)

## [Other Notable Models](open-source-models/other-models.md)
- GPT-Neo (2.7B) - EleutherAI - First free GPT-3 alternative (March 2021)
- GPT-J (6B) - EleutherAI - Largest public GPT-3 style model at release (June 2021)
- GPT-NeoX (20B) - EleutherAI - Scaled up to 20B (February 2022)
- OPT (125M-175B) - Meta AI - Democratizing LLM access (May 2022)
- BLOOM (176B) - BigScience/HuggingFace - Multilingual collaborative effort (July 2022)
- Yi 1.5 (34B) - 01.ai
- Falcon (40B, 180B) - TII Abu Dhabi
- StableLM (1.6B, 12B) - Stability AI
- MPT (7B, 30B) - MosaicML/Databricks
- Apple OpenELM (270M, 450M, 1.1B, 3B) - Apple
- Baichuan (7B, 13B) - Baichuan Inc
- InternLM (7B, 20B) - Shanghai AI Lab
- ChatGLM (6B, 130B) - Tsinghua/Zhipu AI
- Cohere Command R/R+ (104B) - Cohere
- Databricks DBRX (132B MoE) - Databricks
- OLMo (1B-32B) - Allen Institute for AI
- Pythia Suite (70M-12B) - EleutherAI
- StarCoder/StarCoder2 (3B, 7B, 15B) - BigCode/HuggingFace
- CodeGen (350M-16B) - Salesforce
- TinyLlama (1.1B) - Community/Zhang et al
- MiniCPM (1.2B, 2.4B, 8B) - OpenBMB/Tsinghua
- RWKV (0.1B-14B) - Bo Peng et al - RNN-Transformer hybrid
- Jamba (12B-398B MoE) - AI21 Labs - SSM-Transformer hybrid, 256K context
- LLaVA (7B, 13B, 34B) - Microsoft/Wisconsin - Vision-language model
- CogVLM/CogVLM2 (8B-17B) - Tsinghua - Vision understanding
- Fuyu (8B) - Adept - Multimodal for digital agents
- Aya 101 (13B) - Cohere for AI - 101 languages
- Hunyuan-Large (389B MoE) - Tencent - 256K context, Chinese-English
- Falcon Mamba (7B) - TII - Pure SSM architecture
- Palmyra (128M-20B) - Writer - Enterprise, 1M context
- BioGPT (349M) - Microsoft - Biomedical domain
- INTELLECT-1 (10B) - Prime Intellect - First decentralized training across continents (Nov 2024)
- INTELLECT-2 (32B) - Prime Intellect - First decentralized RL training (May 2025)
