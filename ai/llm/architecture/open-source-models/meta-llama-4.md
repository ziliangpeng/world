# Llama 4

**Release Date**: April 5, 2025

Meta's first natively multimodal model family with Mixture-of-Experts architecture and unprecedented 10M token context window.

## Origin Story: A Complete Architectural Reimagining

Llama 4 represents Meta's most ambitious architectural departure in the Llama family history, moving from dense transformers to Mixture-of-Experts while introducing native multimodality and extreme context lengths. This wasn't an incremental improvement—it was a fundamental redesign.

### The Strategic Pivot

After Llama 3's success in matching GPT-4 with dense models, Meta faced a critical decision: continue scaling dense models to ever-larger sizes, or adopt the sparse MoE architecture that had enabled competitors like GPT-4 and Gemini 1.5 to achieve superior efficiency.

**The MoE Decision**:
- **Dense scaling limits**: Training a 1T+ parameter dense model would be prohibitively expensive
- **MoE efficiency**: 400B total parameters with only 17B active delivers 400B capacity at 17B cost
- **Competitive necessity**: GPT-4, Gemini 1.5, DeepSeek V3 all use MoE
- **Future-proofing**: MoE enables continued scaling without proportional compute growth

**The Multimodal Shift**:
- Llama 3.2's adapter approach (vision added later) had fundamental limitations
- Native multimodality from token 0 enables deeper cross-modal understanding
- Unified representation space across text, images, and video
- Competitive response to GPT-4o, Gemini 1.5, Claude 3.5 multimodal capabilities

### Development Timeline and Challenges

- **Development Start**: After Llama 3.1 release (July 2024), "complete re-design" initiated
- **Release Date**: April 5, 2025 (Saturday release—unusual timing that sparked skepticism)
- **Initial Models**: Scout and Maverick released immediately
- **Behemoth**: Originally planned for April 2025 (LlamaCon), pushed to June, still in training as of late 2025

**Development Challenges**:
- **Behemoth delays**: Serious engineering/research concerns about meeting claimed capabilities
- **Real-world performance**: Public Maverick version received mixed feedback; advertised LMArena ranking was from unreleased chat-optimized experimental version
- **Context window issues**: Significant degradation even at 120k tokens (15.6% accuracy vs advertised 10M support)
- **Talent exodus**: 11 of the original 14 PhD researchers who created Llama 1 have left Meta by early 2025

### Team Organization and Leadership Changes

**May 2025 Restructuring**:
Meta split its AI division into two units:
- **AI Products Team**: Led by Connor Hayes, focuses on product integration
- **AGI Foundations Unit**: Co-led by Ahmad Al-Dahle and Amir Frenkel, handles foundational research

**February 2025 Leadership Appointments**:
- **Loredana Crisan**: Lead PM for AI products
- **Amir Frenkel**: Engineering head (former VP of Mixed Reality)

**Current Leadership**:
- **Ahmad Al-Dahle**: VP, Head of GenAI at Meta, Head of Llama Team
- Reports to Chief Product Officer Chris Cox
- Leads AGI Foundations Unit alongside Frenkel

**Team Challenges**:
- Tighter performance standards implemented
- Significant talent loss from original Llama team
- Organizational restructuring mid-development
- Pressure to compete with OpenAI (named top competitor in internal documents)

### Strategic Objectives

**Competing at the Frontier**:
1. **Match GPT-4o**: Multimodal, efficient, strong reasoning
2. **Match Gemini 1.5**: 1M+ context, multimodal, MoE architecture
3. **Match Claude 3.5**: Strong reasoning and coding
4. **Beat DeepSeek V3**: More efficient MoE with competitive reasoning/coding

**Open-Source Leadership**:
- Maintain Llama's 9% enterprise AI market share
- First open-weight natively multimodal MoE models
- Democratize frontier capabilities previously only in proprietary models
- Prove open-source can match closed models on cutting-edge features

**Future-Proofing AI Development**:
- Belief that future AI agents will be conversational, not text-based
- Speech capabilities (Omni models) to compete with GPT-4o Voice Mode, Gemini Live
- Extreme context (10M tokens) enables entirely new use cases
- MoE as template for all future Llama models

### The Gamble and Early Reception

**The Saturday Launch**:
- Unusual April 5, 2025 (Saturday) release timing raised eyebrows
- Sparked immediate skepticism about model readiness
- Some viewed it as rushed to meet competitive pressure

**Mixed Initial Response**:
- **Positive**: Impressive benchmarks (Maverick beats GPT-4o on broad benchmarks per Meta)
- **Negative**: Public Maverick version didn't match advertised performance
- **Controversial**: LMArena ranking claim based on unreleased experimental version
- **Technical issues**: Context window degradation at much shorter lengths than advertised

**The Behemoth Situation**:
- Announced as flagship 288B active parameter model
- Originally planned for April 2025 launch
- Pushed to June, then delayed further
- Still in training with concerns about capability targets
- Creates uncertainty about Meta's flagship model delivery

Despite early challenges, Llama 4 represents Meta's bold bet that the future of open AI requires MoE, native multimodality, and extreme context—even if the execution has been imperfect.

## Model Variants

### Scout (17Bx16E)
- **17B active parameters** (109B total parameters)
- **16 experts** (MoE architecture)
- **10 million token context window**
- Natively multimodal (text, images, video)
- **Training**: ~40 trillion tokens

### Maverick (17Bx128E)
- **17B active parameters** (400B total parameters)
- **128 experts** (MoE architecture)
- **1 million token context window**
- Natively multimodal (text, images, video)
- **Training**: ~22 trillion tokens

### Behemoth (17Bx...E - Still Training)
- **288B active parameters** (~2T total parameters)
- **16 experts** (corrected from earlier reports)
- Still in training on 32,000 H100 GPUs
- Processes 30+ trillion tokens

*Both Scout and Maverick available as base and instruct variants*

## Architecture: From Dense to Sparse MoE

Llama 4 represents a radical architectural shift from the dense transformer approach of Llama 1-3 to a sparse Mixture-of-Experts design with native multimodality.

### Core Architectural Components

**Shared with Llama 3**:
- **Base Design**: Auto-regressive decoder-only transformer (maintained)
- **Normalization**: RMSNorm pre-normalization (unchanged)
- **Activation**: SwiGLU activation function (unchanged in dense layers)
- **Position Encoding**: RoPE (Rotary Position Embeddings) with iRoPE extensions (evolved)
- **Attention**: Grouped-Query Attention (GQA) with 8 KV heads (maintained)

**New in Llama 4**:
- **Architecture Type**: Mixture-of-Experts (MoE) - sparse activation
- **Multimodal**: Natively multimodal from token 0 (not adapter-based)
- **Tokenizer**: Expanded to 202,048 tokens (1.6x from Llama 3's 128K)
- **Context**: Up to 10M tokens via iRoPE (78x from Llama 3.1's 128K)

### Complete Model Specifications

#### Llama 4 Scout (17Bx16E)

| Component | Specification | Llama 3 70B Comparison |
|-----------|---------------|------------------------|
| **Hidden Size** | 8,192 | 8,192 (same) |
| **Layers** | 80 | 80 (same) |
| **Attention Heads** | 64 | 64 (same) |
| **KV Heads** | 8 (GQA) | 8 (GQA, same) |
| **Head Dimension** | 128 | 128 (same) |
| **FFN Size (Dense)** | 8,192 | 28,672 (**Scout smaller**) |
| **FFN Size (MoE Experts)** | 16,384 | N/A (dense model) |
| **Vocabulary** | 202,048 | 128,256 **(1.6x larger)** |
| **Context Window** | **10,000,000** | 128,000 **(78x larger)** |
| **Max Position Embeddings** | 131,072 (extended to 10M) | 128,000 |
| **Total Parameters** | 109B | 70B **(1.6x larger)** |
| **Active Parameters** | **17B** | 70B **(4.1x smaller active)** |
| **Experts** | 16 | N/A (dense) |
| **Experts per Token** | 1 | N/A (all params active) |

#### Llama 4 Maverick (17Bx128E)

| Component | Specification | Llama 3.1 405B Comparison |
|-----------|---------------|---------------------------|
| **Hidden Size** | 8,192 (estimated) | 16,384 **(half)** |
| **Layers** | Alternating dense/MoE | 126 |
| **Attention Heads** | 64 (estimated) | 128 |
| **KV Heads** | 8 | 8 (same) |
| **FFN Size (Dense)** | 8,192 | 53,248 |
| **FFN Size (MoE Experts)** | 16,384 | N/A (dense) |
| **Vocabulary** | 202,048 | 128,256 **(1.6x larger)** |
| **Context Window** | **1,000,000** | 128,000 **(7.8x larger)** |
| **Total Parameters** | 400B | 405B (**similar total**) |
| **Active Parameters** | **17B** | 405B **(23.8x smaller active)** |
| **Experts** | 128 | N/A (dense) |
| **Experts per Token** | 1 + shared | N/A (all params active) |

### MoE Architecture Deep Dive

**Interleaved Design**:
- Dense transformer layers alternate with MoE layers
- Pattern: Layer 0 (Dense) → Layer 1 (MoE) → Layer 2 (Dense) → Layer 3 (MoE)
- Balances comprehensive understanding (dense) with efficiency (sparse)

**MoE Layer Structure** (per layer):
- **Maverick**: 1 shared expert + 128 routed experts = 129 experts per MoE layer
- **Scout**: 1 shared expert + 16 routed experts = 17 experts per MoE layer
- **Expert selection**: Top-1 routing (each token sent to 1 routed expert + shared expert)

**Router Mechanism**:
1. Input token with hidden state vector `x`
2. Learned projection produces logit for each expert
3. Softmax converts logits to probability distribution
4. Top-1 expert selected based on highest probability
5. Token processed by shared expert AND selected routed expert

**Load Balancing**:
- Challenge: Routing collapse (repeatedly selecting same few experts)
- Solution: Load-balancing loss during training
- Rewards equal probability assignment and uniform token routing
- Prevents expert underutilization

**Efficiency vs Dense**:

| Aspect | Dense 400B (Llama 3.1) | MoE 400B (Maverick) | Advantage |
|--------|------------------------|---------------------|-----------|
| **Training FLOPs** | All 405B params every token | Only 17B active per token | **23.8x fewer** |
| **Inference FLOPs** | All 405B params every token | Only 17B active per token | **23.8x fewer** |
| **Model Capacity** | 405B | 400B total | Similar |
| **Memory Footprint** | 405B stored | 400B stored | Similar |
| **Throughput** | Slower (more compute) | **Faster** (less compute) | **23.8x speedup** |

### Multimodal Architecture: Native vs Adapter

**Llama 3.2 Vision (Adapter Approach)**:
1. Pre-train text-only model
2. Freeze text model weights
3. Train vision encoder separately
4. Add cross-attention adapter layers
5. Limited cross-modal understanding

**Llama 4 (Native Multimodal)**:
1. **Early fusion**: Text, images, video as single token sequence from token 0
2. Joint pre-training across all modalities
3. Unified latent space for text/image/video
4. Native processing without translation layers

**Vision Encoder**:
- Based on MetaCLIP architecture
- Trained separately in conjunction with frozen Llama model
- Better adaptation of encoder output to LLM expectations
- Seamlessly embeds visual inputs alongside text tokens

**Integration**:
- Vision tokens integrated directly into transformer backbone
- Joint attention mechanism across text and visual tokens
- No separate encoders/decoders needed
- Unified representation enables cross-modal reasoning

### iRoPE: Enabling 10M Context Window

**Traditional RoPE Limitations**:
- Llama 3.1: 128K context with RoPE scaling
- Beyond ~200K: Significant quality degradation
- Memory/compute constraints at extreme lengths

**iRoPE Components** (Interleaved Rotary Position Embeddings):

1. **Interleaved Attention Layers**:
   - Alternating attention types for local and global dependencies
   - **NoPE layers**: No positional encoding (every 4th layer)
   - **RoPE layers**: Standard rotary embeddings (3 out of 4 layers)

2. **Chunked Attention**:
   - Chunk size: 8,192 tokens
   - Local attention computed within chunks
   - Used in RoPE layers
   - Reduces memory footprint

3. **Inference-Time Temperature Scaling**:
   - Scales attention scores by temperature parameter
   - Controls attention distribution sharpness
   - Enables focus on relevant context parts in very long sequences

4. **Hybrid Attention Mechanism**:
   - Global attention without positional encoding (NoPE layers)
   - Local attention in chunks (RoPE layers)
   - Balances comprehensive understanding with memory efficiency

**Training for Long Context**:
- Mid-training phase with specialized long-context datasets
- Continued training to unlock 10M context
- Enhanced model quality during extension

**Practical Performance** (Fiction.LiveBench):
- **Advertised**: 10M tokens
- **Reality at 120k**: 15.6% accuracy (severe degradation)
- **Comparison**: Gemini 2.5 Pro at 120k: 90.6%
- **Issue**: Significant gap between claimed and actual long-context performance

### Tokenizer Evolution

**Llama 3 → Llama 4 Comparison**:

| Aspect | Llama 3 | Llama 4 | Change |
|--------|---------|---------|--------|
| **Implementation** | TikToken-based BPE | TikToken-based BPE | Same |
| **Vocabulary Size** | 128,256 | **202,048** | **1.6x larger** |
| **Base Tokens** | 128,000 | 200,000 | +72,000 |
| **Special Tokens** | 256 | **2,048** | **8x more** |
| **Pattern** | O200K_PATTERN (estimated) | O200K_PATTERN regex | Similar |

**Special Tokens** (new in Llama 4):
- `<|header_start|>`, `<|header_end|>`: Message headers
- `<|eom|>`: End of message
- `<|eot|>`: End of turn
- `<|step|>`: Step marker (possibly for reasoning chains)
- Plus 2,048 reserved special tokens

**Benefits of Larger Vocabulary**:
- Better compression for same text (fewer tokens needed)
- Improved multilingual support (200 languages trained)
- Enhanced code representation
- More specialized tokens for structured outputs

### Key Architectural Differences from Llama 3

| Feature | Llama 3 | Llama 4 | Impact |
|---------|---------|---------|--------|
| **Architecture** | Dense | **MoE (sparse)** | 23x efficiency gain |
| **Largest Model** | 405B dense | 400B total, 17B active | Similar capacity, much faster |
| **Multimodal** | Adapter (3.2) | **Native** | Better cross-modal understanding |
| **Context** | 128K | **10M (Scout)** | 78x larger (with caveats) |
| **Vocabulary** | 128K | **202K** | 1.6x expansion |
| **Training Tokens** | 15T | **30T+** | 2x more data |
| **Vision Integration** | Cross-attention adapter | **Early fusion** | Unified from start |

The shift to MoE represents Llama's most significant architectural evolution, trading dense computation for sparse efficiency while maintaining model capacity.

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
