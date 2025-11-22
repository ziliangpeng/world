# Llama 4

**Release Date**: April 5, 2025

Meta's first natively multimodal model family with Mixture-of-Experts architecture and unprecedented 10M token context window.

## Model Variants

###

 Scout
- **17B active parameters** (109B total parameters)
- **16 experts** (MoE architecture)
- **10 million token context window**
- Natively multimodal (text, images, video)

### Maverick
- **17B active parameters** (400B total parameters)
- **128 experts** (MoE architecture)
- **1 million token context window**
- Natively multimodal (text, images, video)

### Behemoth (Announced, Not Yet Released)
- **288B active parameters** (~2T total parameters)
- **16 experts**
- Still in training

## Major Architectural Changes

### First Llama with Mixture-of-Experts (MoE)

**What Changed**: Llama 1-3 were dense models. Llama 4 uses sparse MoE.

**How MoE Works**:
- Total parameters ≠ Active parameters
- Each token routed to subset of experts
- Massive capacity with reasonable compute
- **Scout**: 109B total, only 17B active per token
- **Maverick**: 400B total, only 17B active per token

**Benefits**:
- Scale capacity without proportional compute increase
- Specialization: Different experts for different domains
- Efficiency: Similar compute to dense 17B, capacity of 100B+

### Natively Multimodal

**Built from Scratch**: Unlike Llama 3.2 Vision (adapter approach), Llama 4 is natively multimodal.

**Capabilities**:
- Analyzes and understands **text, images, and video**
- Joint training on multimodal data from the start
- Unified representation space

**Difference from 3.2 Vision**:
- Llama 3.2: Text model + vision adapter (vision added later)
- Llama 4: Multimodal from the ground up

### Unprecedented Context Window

**Scout: 10 Million Tokens**
- Largest context window in Llama family history
- Can process:
  - Massive codebases
  - Multiple books simultaneously
  - Years of conversation history
  - Comprehensive documentation sets

**Maverick: 1 Million Tokens**
- Still massive compared to earlier models
- 8x Llama 3.1's 128K

**Comparison**:
- Llama 1: 2K
- Llama 2: 4K
- Llama 3: 8K
- Llama 3.1: 128K
- Llama 4 Scout: **10,000K** (10M)

## Training Details

- **Tokens**: **30+ trillion tokens** (2x Llama 3's 15T)
- **Compute**: Trained on cluster with **100,000+ H100 GPUs**
- **Multimodal Training**: Native multimodal training from scratch
- **Data**: Text, images, video combined

### Training Scale

The scale is unprecedented:
- 100K+ H100 GPUs (massive infrastructure)
- 30T tokens (double previous generation)
- Multimodal from start (not adapter approach)

## Key Innovations

1. **First Open-Weight Natively Multimodal MoE Models**
   - Combines MoE + multimodal in open model
   - Previously only proprietary models (GPT-4, Gemini)

2. **Massive Context Length (10M tokens)**
   - 78x larger than Llama 3.1
   - Enables entirely new use cases

3. **Advanced Reasoning and Speech Capabilities**
   - Enhanced reasoning over previous versions
   - Speech processing capabilities

4. **Doubled Training Data**
   - 30T tokens vs 15T in Llama 3
   - Higher quality multimodal data

## Use Cases

**Scout (10M context)**:
- Entire large codebase analysis
- Multi-book comparative analysis
- Comprehensive research across many papers
- Years of chat history

**Maverick (1M context, 128 experts)**:
- High-capacity multimodal tasks
- Complex video understanding
- Document analysis with images

**Both**:
- Visual question answering
- Video summarization and analysis
- Multimodal reasoning
- Code generation with visual context

## Significance

Llama 4 represents Meta's push into:
1. **Frontier open models** - Competing with GPT-4, Gemini 1.5
2. **Multimodal by default** - Not text-first with vision added
3. **Extreme context** - 10M tokens unprecedented in open models
4. **MoE architecture** - Efficient scaling

## Comparison to Llama 3

| Feature | Llama 3.1 | Llama 4 |
|---------|-----------|---------|
| **Architecture** | Dense | **MoE (sparse)** |
| **Largest Model** | 405B dense | 400B total (17B active) |
| **Context** | 128K | **10M (Scout)** |
| **Modality** | Text-only | **Natively multimodal** |
| **Training Tokens** | 15T | **30T** |
| **Training Compute** | Unknown | **100K+ H100s** |

## Links

- **Blog**: [The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- **Announcement**: [Meta Launches Llama 4 Models](https://www.socialmediatoday.com/news/meta-releases-llama-4-ai-models/744560/)

## Future: Behemoth

When released, Behemoth will be:
- **288B active parameters** (~2T total with MoE)
- Likely similar multimodal capabilities
- Potentially even longer context

This would make it one of the largest open models ever released.

## Impact

Llama 4 demonstrates:
1. **Open can match proprietary** on frontier capabilities (MoE, multimodal)
2. **Context can scale to extremes** (10M tokens)
3. **MoE is the future** for efficient scaling
4. **Multimodal is standard** moving forward

Meta continues to push the boundaries of what's possible with open AI.
