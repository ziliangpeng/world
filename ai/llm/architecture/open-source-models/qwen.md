# Qwen Series (Alibaba Cloud)

Qwen (通义千问, "Tongyi Qianwen") from Alibaba Cloud represents a leading series of multilingual LLMs spanning text, vision, audio, and multimodal capabilities. The series has evolved from the original Qwen 1.0 in 2023 to the advanced Qwen3 family in 2025, featuring both dense and MoE (Mixture of Experts) variants with exceptional multilingual support across 119 languages.

## Timeline Overview

| Date | Model | Key Features |
|------|-------|-------------|
| Aug 2023 | Qwen 1.0 | 1.8B-72B, first series |
| Oct 2023 | Qwen-VL | Vision-language with grounding |
| Nov 2023 | Qwen-Audio | Universal audio understanding |
| Feb 2024 | Qwen 1.5 | 0.5B-110B, first MoE variant |
| Jun 2024 | Qwen2 | 0.5B-72B + 57B MoE |
| Jul 2024 | Qwen2-Audio | Enhanced audio models |
| Sep 2024 | Qwen2-VL | Any-resolution vision |
| Sep 2024 | Qwen2.5 | 0.5B-72B, 18T tokens |
| Sep 2024 | Qwen2.5-Coder | Code specialist |
| Sep 2024 | Qwen2.5-Math | Math specialist |
| Nov 2024 | QwQ-32B | Reasoning model |
| Jan 2025 | Qwen2.5-VL | Multi-resolution vision |
| Mar 2025 | Qwen2.5-Omni | Omni-modal 7B |
| Apr 2025 | Qwen3 | 0.6B-235B, 36T tokens |
| Jul 2025 | Qwen3-Coder | 480B MoE for coding |
| Sep 2025 | Qwen3-Omni | Real-time multimodal |
| Sep 2025 | Qwen3-Next | 80B MoE (3B active) |

---

## Qwen 1.0 Series (August-November 2023)

### Overview
The inaugural Qwen series, establishing Alibaba's presence in the open-source LLM landscape.

**Paper**: [Qwen Technical Report](https://arxiv.org/abs/2309.16609) (September 2023)

### Model Variants
- **Qwen-1.8B** & **Qwen-1.8B-Chat** (November 30, 2023)
- **Qwen-7B** & **Qwen-7B-Chat** (August 3, 2023)
- **Qwen-14B** & **Qwen-14B-Chat** (September 25, 2023)
- **Qwen-72B** & **Qwen-72B-Chat** (November 30, 2023)

### Training Details
- **Tokens**: 3 trillion tokens (72B variant)
- **Context Window**: 2,048 tokens (base), extended to 8K-32K
- **Languages**: Multilingual with strong Chinese and English support
- **Quantization**: Int4 and Int8 variants available

### Key Features
- Pre-normalization with RMSNorm
- SwiGLU activation functions
- RoPE (Rotary Position Embeddings)
- Strong multilingual capabilities
- Efficient quantized versions

---

## Qwen-VL (Vision-Language) - October 2023

**Paper**: [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)

### Model Variants
- **Qwen-VL**: Base vision-language model
- **Qwen-VL-Chat**: Instruction-tuned multimodal assistant

### Architecture
- Vision encoder for image understanding
- Text decoder based on Qwen-7B
- Cross-modal attention mechanisms

### Capabilities
- **Grounding**: Object localization with bounding boxes
- **Text Reading**: OCR and text understanding in images
- **Multi-Image**: Process multiple images in conversation
- **Creative**: Image-based creative content generation
- **Multi-Round QA**: Sustained visual conversation

### Training Approach
- Image-caption-box tuples for grounding alignment
- Instruction tuning for conversational abilities
- Multilingual image-text datasets

---

## Qwen-Audio - November 2023

**Paper**: [Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models](https://arxiv.org/abs/2311.07919)

### Overview
Universal audio understanding model handling diverse audio types and 30+ tasks.

### Model Variants
- **Qwen-Audio**: Base audio-language model
- **Qwen-Audio-Chat**: Conversational audio assistant

### Audio Capabilities
- **Human Speech**: Speech recognition, translation, analysis
- **Natural Sounds**: Environmental sound understanding
- **Music**: Music analysis and understanding
- **Songs**: Vocal and lyrical analysis

### Architecture
- Audio encoder (multi-resolution)
- Cross-modal fusion with Qwen LLM base
- Universal audio representation learning

### Task Coverage (30+ tasks)
- Automatic Speech Recognition (ASR)
- Speech-to-Text Translation
- Audio Captioning
- Sound Event Detection
- Music Analysis
- Emotion Recognition from Audio

---

## Qwen 1.5 Series (February 2024)

### Model Variants

**Dense Models**:
- **0.5B, 1.8B, 4B, 7B, 14B, 32B, 72B, 110B**

**MoE Variant**:
- **Qwen1.5-MoE-A2.7B**: First Qwen MoE model
  - Total parameters: ~14.3B
  - Activated parameters: ~2.7B per token
  - Efficient inference with MoE architecture

### Key Improvements
- Extended model size range (0.5B to 110B)
- Introduction of Mixture of Experts
- Improved instruction following
- Better multilingual performance
- Enhanced context understanding

### Context Window
- Standard: 32K tokens
- Some variants support up to 128K

### Release Notes
- Announced via [blog post](https://qwenlm.github.io/blog/qwen1.5/)
- No dedicated arXiv paper (referenced in Qwen2 technical report)

---

## Qwen 2.0 Series (June 2024)

**Paper**: [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) (July 15, 2024)

### Model Variants

**Dense Models**:
- **0.5B, 1.5B, 7B, 57B, 72B**

**MoE Model**:
- **Qwen2-57B-A14B**
  - Total parameters: 57B
  - Activated parameters: 14B per token
  - Experts: 128 (evolved from Qwen1.5-MoE architecture)
  - Context: 65,536 tokens (64K)

### Architecture

**Type**: Decoder-only transformer

**Key Components**:
- **Attention**: Grouped Query Attention (GQA)
- **Activation**: SwiGLU
- **Position Encoding**: RoPE (Rotary Position Embeddings)
- **Normalization**: RMSNorm with pre-normalization
- **QKV Bias**: Attention QKV bias included
- **Optimization**: FlashAttention 2

### MoE Architecture (Qwen2-57B-A14B)
- **128 experts** total
- **Top-K routing**: Activates subset of experts per token
- **Fine-grained**: Smaller experts for better specialization
- **Load balancing**: Auxiliary loss for expert utilization

### Training Details
- Extensive multilingual training data
- Code and text mixed corpus
- Context length: Up to 65,536 tokens
- Instruction tuning variants available

### Specialized Variants

**Qwen2-Math**:
- Model sizes: 1.5B, 7B, 72B
- Math and STEM specialized through targeted pretraining
- Strong performance on GSM8K, MATH benchmarks

---

## Qwen2-Audio (July 2024)

**Paper**: [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759)

### Overview
Second-generation audio-language model with enhanced capabilities.

### Interaction Modes

1. **Voice Chat Mode**
   - Natural spoken conversation
   - Real-time audio understanding
   - Multi-turn dialogue

2. **Audio Analysis Mode**
   - Detailed audio content analysis
   - Sound event detection
   - Audio quality assessment

### Performance
- Outperformed Gemini-1.5-pro on AIR-Bench
- State-of-the-art on multiple audio understanding tasks
- Improved multilingual audio capabilities

### Architecture Improvements
- Enhanced audio encoder
- Better cross-modal alignment
- Improved temporal modeling

---

## Qwen2-VL (Vision-Language) - September 2024

**Paper**: [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191)

### Model Variants
- **Qwen2-VL-2B**: 2 billion parameters
- **Qwen2-VL-7B**: 7 billion parameters

### Key Innovation: Any Resolution
- **Dynamic Resolution**: Processes images at their native resolution
- **No Fixed Input Size**: Adapts to image dimensions
- **Better Detail**: Preserves fine-grained visual information

### Architecture
- Advanced vision encoder
- Cross-attention fusion with Qwen2 language model
- Multi-scale visual feature extraction

### Capabilities
- High-resolution image understanding
- Document analysis and OCR
- Chart and diagram interpretation
- Multi-image reasoning
- Video understanding (frame-by-frame)

### Performance
- State-of-the-art on visual QA benchmarks
- Excellent document understanding
- Strong performance on chart/diagram tasks

---

## Qwen 2.5 Series (September 2024)

**Paper**: [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) (December 20, 2024)

### Model Variants
- **0.5B**: 0.5 billion parameters
- **1.5B**: 1.5 billion parameters
- **3B**: 3 billion parameters
- **7B**: 7 billion parameters
- **14B**: 14 billion parameters
- **32B**: 32 billion parameters
- **72B**: 72 billion parameters

### Architecture

**Type**: Dense decoder-only transformer

**Key Components**:
- **Attention**: Grouped Query Attention (GQA)
- **Activation**: SwiGLU
- **Position Encoding**: RoPE (Rotary Position Embeddings)
- **Normalization**: RMSNorm with pre-normalization
- **Optimization**: FlashAttention 2 for efficient attention computation

### Training Details
- **Tokens**: 18 trillion tokens
- **Context Window**: 128K tokens (some versions handle up to 1M tokens)
- **Languages**: Advanced multilingual capabilities (29+ languages)
- **Modalities**: Text, vision (Qwen2.5-VL variants)

### Specifications (7B Example)
- **Layers**: 28
- **Hidden Dimension**: 3,584
- **Attention Heads**: 28
- **KV Heads**: 4 (GQA with 7 queries per key/value)
- **FFN Dimension**: 18,944
- **Vocabulary**: ~152K tokens

### Special Capabilities
- Code generation and understanding
- Mathematical reasoning
- Long-context understanding (128K-1M tokens)
- Instruction following
- Multimodal understanding (vision variants)

---

## Qwen2.5-Coder (September 2024)

**Paper**: [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186)

### Model Variants
- **0.5B, 1.5B, 3B, 7B, 14B, 32B**

### Training Details
- **Code Data**: 5.5+ trillion tokens of code
- **Languages**: 92 programming languages
- **Context Window**: 128K tokens
- **License**: Apache 2.0

### Capabilities
- Code completion and generation
- Code understanding and explanation
- Bug fixing and debugging
- Code translation between languages
- Test generation
- Documentation generation

### Architecture
- Based on Qwen2.5 architecture
- Specialized code tokenization
- Code-aware attention patterns

### Performance
- State-of-the-art on HumanEval, MBPP
- Excellent multi-language support
- Strong performance on real-world code tasks

### Variants
- **Base**: Pure code completion
- **Instruct**: Instruction-following for coding tasks

---

## Qwen2.5-Math (September 2024)

**Paper**: [Qwen2.5-Math Technical Report: Toward Mathematical Expert Model via Self-Improvement](https://arxiv.org/abs/2409.12122)

### Model Variants
- **1.5B, 7B, 72B**

### Key Innovation: Self-Improvement
- Iterative training process
- Self-generated reasoning chains
- Curriculum learning for math

### Additional Components
- **Qwen2.5-Math-RM**: Reward model for mathematical reasoning
- Used for reinforcement learning from feedback

### Capabilities
- Advanced mathematical reasoning
- Step-by-step problem solving
- Theorem proving assistance
- Mathematical proof verification

### Performance
- State-of-the-art on GSM8K, MATH
- Strong competition-level math performance
- Excellent reasoning chain quality

### Training Approach
1. Base mathematical pretraining
2. Self-improvement through generated solutions
3. Reward model-guided refinement
4. Instruction tuning for usability

---

## QwQ-32B-Preview (November 2024)

### Overview
Qwen's reasoning-focused model, similar to OpenAI's o1 series.

**Blog Post**: [QwQ: Reflect Deeply on the Boundaries of the Unknown](https://qwenlm.github.io/blog/qwq-32b-preview/)

### Specifications
- **Parameters**: 32 billion
- **Context Window**: 32,768 tokens (32K)
- **License**: Apache 2.0

### Key Features

**Thinking Mode**:
- Extended reasoning chains
- Self-reflection during problem-solving
- Step-by-step logical reasoning
- Verification of intermediate steps

**Capabilities**:
- Advanced mathematical problem solving
- Complex coding challenges
- Logical reasoning tasks
- Multi-step problem decomposition

### Performance
- Excels at math and coding benchmarks
- Comparable to reasoning-specialized models
- Strong performance on competition-level problems

### Architecture
- Based on Qwen architecture
- Enhanced with reasoning-specific training
- Reinforcement learning for reasoning quality

### Status
- **Preview Release**: Experimental model
- Demonstrates reasoning capabilities
- Foundation for future reasoning-focused Qwen models

---

## Qwen2.5-VL (Vision-Language) - January 2025

### Model Variants
- **3B, 7B, 32B, 72B**

### Key Features
- Multi-resolution vision understanding
- Enhanced image and video comprehension
- Better document and chart analysis

### Improvements over Qwen2-VL
- Larger model sizes (up to 72B)
- Better visual reasoning
- Improved multi-image understanding
- Enhanced video temporal modeling

### Architecture
- Advanced vision encoder with multiple scales
- Cross-modal fusion with Qwen2.5 base
- Efficient attention for high-resolution inputs

---

## Qwen2.5-Omni (March 2025)

### Overview
First Qwen omni-modal model combining text, vision, audio, and video.

### Specifications
- **Model Size**: 7 billion parameters
- **License**: Apache 2.0

### Input Modalities
- **Text**: Natural language
- **Images**: Single or multiple images
- **Videos**: Video understanding
- **Audio**: Speech and sounds

### Output Modalities
- **Text**: Generated text responses
- **Audio**: Synthesized speech output

### Capabilities
- Unified multimodal understanding
- Cross-modal reasoning
- Multimodal conversation
- Real-time interaction (with audio I/O)

### Architecture
- Multi-modal encoders (vision, audio)
- Unified representation space
- Cross-attention fusion
- Dual output heads (text, audio)

---

## Qwen 3 Series (April 2025)

**Paper**: [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (May 14, 2025)

### Model Variants

**Dense Models**:
- **0.6B, 1.7B, 4B, 8B, 14B, 32B**

**MoE Models**:
- **Qwen3-30B**: 30B total, ~3B activated
- **Qwen3-235B**: 235B total, ~22B activated

### Major Architectural Innovation: QK-Norm

**Traditional Attention**:
```python
attention = softmax(Q @ K.T / sqrt(d))
```

**With QK-Norm**:
```python
Q_norm = normalize(Q, dim=-1)  # Per-head normalization
K_norm = normalize(K, dim=-1)
attention = softmax(Q_norm @ K_norm.T / sqrt(d))
```

**Benefits**:
- Prevents attention score explosion at scale
- Stabilizes training of large models
- Enables higher learning rates
- Reduces gradient instabilities
- Critical for training 235B MoE model

### Architecture Changes from Qwen2

**Removed**:
- QKV bias in attention (present in Qwen2)

**Added**:
- **QK-Norm**: Per-head normalization of queries and keys
- More aggressive GQA configurations
- Enhanced position encodings

**Retained**:
- Grouped Query Attention (GQA)
- SwiGLU activation
- RoPE position embeddings
- RMSNorm with pre-normalization
- FlashAttention 2

### Training Details
- **Tokens**: 36 trillion tokens (2x Qwen 2.5)
- **Languages**: 119 languages and dialects
- **Context**: 4,096 → extended to 32,768 tokens
- **License**: Apache 2.0

### Three-Stage Training Process

**Stage 1: General Knowledge**
- Broad coverage across domains
- Multilingual data (119 languages)
- Code and natural language mixed
- Foundation building

**Stage 2: STEM & Coding Specialization**
- Enhanced mathematical reasoning
- Advanced code understanding
- Scientific domain knowledge
- Technical documentation

**Stage 3: Long Context Training**
- Context window expansion: 4K → 32K
- Long-range dependency learning
- Document-level understanding
- Multi-document reasoning

### Dual-Mode Operation

**Non-Thinking Mode**:
- Fast, direct responses
- Standard inference
- Lower latency

**Thinking Mode**:
- Extended reasoning chains
- Step-by-step problem solving
- Self-reflection
- Higher quality for complex tasks

### Qwen3 MoE Architecture

**Type**: Fine-Grained Sparse MoE

**Qwen3-30B**:
- Total parameters: ~30B
- Active parameters: ~3B per token
- Expert count: Not specified in public docs

**Qwen3-235B**:
- Total parameters: ~235B
- Active parameters: ~22B per token
- Expert count: Not specified in public docs
- Largest open-source Qwen MoE model

**Note**: Earlier documentation referenced "128 experts" but official Qwen3 technical report doesn't specify exact expert counts for the 30B and 235B variants. The fine-grained MoE design allows flexible expert configurations.

**MoE Design Philosophy**:
- Fine-grained expert segmentation
- Dynamic routing based on input
- Load balancing across experts
- Efficient activation patterns

### GQA Configuration

**Aggressive GQA** for memory efficiency:
- **Example (7B-class)**: 28 query heads, 4 KV heads
- **Ratio**: 7 queries share each K/V pair
- **Memory Savings**: ~7x reduction in KV cache vs MHA
- **Quality**: Maintained through careful tuning

### Massive Vocabulary
- **Up to 152K tokens** (vs 32K-64K in most models)
- Better multilingual coverage (119 languages)
- More efficient encoding across languages
- Fewer tokens per sequence on average
- Critical for multilingual scaling

### Performance Characteristics

**Strengths**:
1. **Multilingual**: Industry-leading 119 languages
2. **Code**: State-of-the-art coding benchmarks
3. **Math**: Excellent mathematical reasoning
4. **Long Context**: Effective 32K+ token windows
5. **Efficiency**: GQA and FlashAttention enable fast inference

**Benchmarks**:
- Competitive with or exceeding Llama 3, GPT-4 class models
- Strong multilingual evaluation scores
- Excellent HumanEval, MBPP (code)
- High GSM8K, MATH scores (reasoning)

---

## Qwen3-Coder (July 2025)

### Specifications
- **Architecture**: MoE (Mixture of Experts)
- **Total Parameters**: 480 billion
- **Active Parameters**: 35 billion per token
- **License**: Apache 2.0

### Key Features

**Agentic Coding**:
- Tool use integration
- Multi-step coding workflows
- Environment interaction
- Self-debugging capabilities

**Capabilities**:
- Advanced code generation
- Complex system design
- Multi-file project understanding
- Automated testing and debugging
- Code review and optimization

### Architecture
- Based on Qwen3 MoE framework
- Specialized code experts
- Enhanced tool-use capabilities
- Long-context support for large codebases

---

## Qwen3-Omni (September 2025)

### Overview
Next-generation omni-modal model with real-time capabilities.

### Specifications
- **License**: Apache 2.0
- Based on Qwen3 architecture

### Input Modalities
- **Text**: Natural language
- **Images**: Visual understanding
- **Audio**: Speech and sound
- **Video**: Video comprehension

### Output Modalities
- **Text**: Generated responses
- **Audio**: Natural speech synthesis in real-time

### Key Innovation: Real-Time Speech
- Low-latency audio generation
- Natural prosody and intonation
- Multi-speaker capabilities
- Streaming audio output

### Capabilities
- Real-time multimodal conversation
- Cross-modal reasoning
- Unified representation learning
- Simultaneous multi-modal processing

---

## Qwen3-Max (September 2025)

### Overview
Flagship proprietary model from Alibaba Cloud.

**Release**: September 5, 2025

### Deployment
- **Availability**: Alibaba Cloud API only
- **Not Open Source**: Proprietary model
- Commercial offering

### Characteristics
- Largest and most capable Qwen model
- Enhanced performance across all tasks
- Advanced reasoning capabilities
- Multimodal support

### Use Cases
- Enterprise applications
- Complex reasoning tasks
- Production deployments
- Mission-critical applications

---

## Qwen3-Next (September 2025)

### Specifications
- **Total Parameters**: 80 billion
- **Active Parameters**: 3 billion per token
- **Architecture**: MoE (Mixture of Experts)
- **License**: Apache 2.0

### Characteristics
- Extremely efficient MoE design
- High parameter count with low activation
- Excellent quality-to-compute ratio

### Architecture
- Fine-grained expert specialization
- Efficient routing mechanisms
- Based on Qwen3 foundation

**Release**: September 10, 2025

---

## Qwen3-VL (Vision-Language) - 2025

### Overview
Most powerful vision-language model in the Qwen series.

### Key Capabilities

**Enhanced Spatial Perception**:
- 3D spatial understanding
- Depth reasoning
- Object relationships
- Scene composition

**Video Understanding**:
- Temporal dynamics
- Action recognition
- Event understanding
- Long-form video analysis

**Visual Agent Capabilities**:
- PC/Desktop GUI operation
- Mobile interface interaction
- Web navigation
- Visual task automation

**3D Grounding**:
- 3D object localization
- Spatial relationship reasoning
- Scene reconstruction understanding

### Architecture
- Advanced vision encoder with 3D awareness
- Temporal modeling for videos
- Agent-oriented visual understanding
- Enhanced cross-modal fusion

### Use Cases
- Visual agents and automation
- Complex visual reasoning
- 3D scene understanding
- Interactive GUI systems

---

## Proprietary API Models

Alibaba Cloud offers several proprietary Qwen models via API.

### Qwen2.5-Turbo

**Architecture**: MoE (Mixture of Experts)

**Pricing**:
- Input: $0.0004 per 1K tokens
- Output: $0.0012 per 1K tokens

**Characteristics**:
- Cost-effective
- Fast inference
- Good quality-to-price ratio

### Qwen2.5-Plus (Qwen-Plus)

**Characteristics**:
- Competitive with GPT-4o
- Balanced performance
- Production-ready

**Pricing**:
- Input: $0.0030 per 1K tokens
- Output: $0.0090 per 1K tokens

### Qwen2.5-Max (Qwen-Max)

**Architecture**: Large-scale MoE

**Training**:
- 20+ trillion tokens pretrained
- Extensive instruction tuning

**Pricing**:
- Input: $0.0100 per 1K tokens
- Output: $0.0300 per 1K tokens

**Characteristics**:
- Highest quality Qwen API model
- Advanced reasoning
- Multimodal capabilities

---

## Common Architectural Foundation

### Transformer Stack

```
Input → Token Embedding + RoPE
  ↓
[Repeated N times]:
  RMSNorm
  → Grouped-Query Attention
    - Q: Multiple query heads
    - K, V: Shared across query groups
    - RoPE applied to Q and K
    - QK-Norm (Qwen 3+)
  → Residual Connection
  → RMSNorm
  → SwiGLU FFN (or MoE layer)
  → Residual Connection
  ↓
Final RMSNorm → Output Projection
```

### Key Architectural Components

**Attention Mechanism**:
- **Grouped Query Attention (GQA)**: Memory-efficient attention
- **QK-Norm (Qwen3+)**: Per-head normalization for stability
- **RoPE**: Rotary position embeddings for relative positions
- **FlashAttention 2**: Optimized attention computation

**Feed-Forward Network**:
- **SwiGLU Activation**: Gated activation function
- **Expansion Ratio**: Typically 2.5-3x hidden dimension
- **MoE Variants**: Sparse expert routing in MoE models

**Normalization**:
- **RMSNorm**: Root Mean Square normalization
- **Pre-normalization**: Applied before attention and FFN

### Long Context Scaling Strategies

**RoPE Interpolation**:
- Dynamic position encoding extension
- Maintains relative position information

**Progressive Training**:
- Start with shorter contexts (4K)
- Gradually extend to 32K, 128K, or 1M tokens
- Curriculum learning approach

**FlashAttention 2**:
- Memory-efficient attention
- Enables practical long-context inference
- Critical for 128K+ contexts

---

## Evolution Summary

| Version | Tokens Trained | Context Window | Key Innovation |
|---------|----------------|----------------|----------------|
| Qwen 1.0 | ~3T | 2K-32K | Multilingual foundation |
| Qwen 1.5 | ~5T | 32K-128K | First MoE variant (MoE-A2.7B) |
| Qwen 2.0 | ~7T | 64K | 128-expert MoE architecture |
| Qwen 2.5 | 18T | 128K-1M | FlashAttention 2, massive scale |
| Qwen 3 | 36T | 32K+ | QK-Norm, 3-stage training, 119 languages |

---

## Specialized Capabilities Across Series

### Code Generation
- **Qwen2.5-Coder**: 92 languages, 128K context
- **Qwen3-Coder**: 480B MoE, agentic capabilities

### Mathematics
- **Qwen2-Math**: Targeted math training
- **Qwen2.5-Math**: Self-improvement approach with reward model

### Reasoning
- **QwQ-32B**: o1-style extended reasoning
- **Qwen3 Thinking Mode**: Dual-mode operation

### Vision
- **Qwen-VL**: Grounding and text reading
- **Qwen2-VL**: Any-resolution processing
- **Qwen2.5-VL**: Multi-resolution (up to 72B)
- **Qwen3-VL**: 3D grounding, visual agents

### Audio
- **Qwen-Audio**: 30+ audio tasks
- **Qwen2-Audio**: Voice chat + analysis modes

### Multimodal
- **Qwen2.5-Omni**: 7B omni-modal
- **Qwen3-Omni**: Real-time speech generation

---

## Deployment Considerations

### Dense Models
- **Straightforward deployment**: Standard inference frameworks
- **Various sizes**: 0.5B to 72B for different compute budgets
- **Excellent for general-purpose use**: Balanced performance

### MoE Models
- **Higher throughput**: More parameters with lower active compute
- **Memory requirements**: All experts must be loaded
- **Framework support**: Requires MoE-aware inference (vLLM, TGI, etc.)
- **Cost-effective**: Better quality per active parameter

### Context Length Trade-offs
- **128K-1M contexts**: Significant memory requirements
- **FlashAttention essential**: Practical inference impossible without it
- **KV cache scaling**: GQA provides major savings
- **Progressive context**: Can start small and extend as needed

### Quantization
- **Int8/Int4 support**: Most models support quantization
- **GPTQ/AWQ**: Weight-only quantization
- **Quality preservation**: Minimal degradation with careful quantization

---

## Impact on the Field

### Technical Contributions

1. **QK-Norm**: Novel normalization for training stability at scale
2. **Fine-Grained MoE**: 128-expert architecture with efficient routing
3. **Multilingual Excellence**: 119 languages with strong performance
4. **Long Context**: Practical 128K-1M token models
5. **Multimodal Integration**: Unified omni-modal architectures

### Open Source Impact
- **Apache 2.0 / Custom Open Licenses**: Permissive licensing
- **Multilingual Accessibility**: Enabled applications worldwide
- **Strong Alternative**: Competitive with Western-focused models
- **Commercial Adoption**: Popular for production deployments in Asia

### Research Innovations
- Self-improvement for mathematical reasoning
- Any-resolution vision processing
- Real-time multimodal generation
- Visual agent capabilities
- Decentralized training demonstrations

---

## Sources

### Primary Papers
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609) - Qwen 1.0
- [Qwen-VL](https://arxiv.org/abs/2308.12966) - Vision-Language
- [Qwen-Audio](https://arxiv.org/abs/2311.07919) - Audio Understanding
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) - Qwen 2.0
- [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759) - Enhanced Audio
- [Qwen2-VL](https://arxiv.org/abs/2409.12191) - Any Resolution Vision
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) - Qwen 2.5
- [Qwen2.5-Coder Technical Report](https://arxiv.org/abs/2409.12186) - Code Specialist
- [Qwen2.5-Math Technical Report](https://arxiv.org/abs/2409.12122) - Math via Self-Improvement
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) - Qwen 3

### Official Resources
- [Qwen GitHub - QwenLM/Qwen](https://github.com/QwenLM/Qwen)
- [Qwen HuggingFace](https://huggingface.co/Qwen)
- [Qwen Blog](https://qwenlm.github.io/blog/)
- [Alibaba Cloud Qwen Documentation](https://www.alibabacloud.com/help/en/model-studio/what-is-qwen-llm)

### Specific Releases
- [Introducing Qwen1.5](https://qwenlm.github.io/blog/qwen1.5/)
- [Qwen2.5: A Party of Foundation Models!](https://qwenlm.github.io/blog/qwen2.5/)
- [Qwen2.5-Coder: Code More, Learn More!](https://qwenlm.github.io/blog/qwen2.5-coder/)
- [QwQ: Reflect Deeply on the Boundaries of the Unknown](https://qwenlm.github.io/blog/qwq-32b-preview/)
- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [Qwen2.5-Max: Exploring the Intelligence of Large-scale MoE Model](https://qwenlm.github.io/blog/qwen2.5-max/)

### Model Repositories
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni)
- [Qwen3-Coder GitHub](https://github.com/QwenLM/Qwen3-Coder)
- [QwQ GitHub](https://github.com/QwenLM/QwQ)

### Additional References
- [Qwen - Wikipedia](https://en.wikipedia.org/wiki/Qwen)
- [Qwen/QwQ-32B-Preview on HuggingFace](https://huggingface.co/Qwen/QwQ-32B-Preview)
