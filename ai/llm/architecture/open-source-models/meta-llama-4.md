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

## Training Details: Doubling Down on Scale

Llama 4's training represents a massive scale-up from Llama 3, with 2x the data, new multimodal training methodology, and revolutionary post-training approaches.

### Training Scale: 2x Data Expansion from Llama 3

**Token Counts**:

| Model | Training Tokens | Llama 3 Baseline | Increase |
|-------|----------------|------------------|----------|
| **Scout** | ~40 trillion | 15T (Llama 3) | **2.7x** |
| **Maverick** | ~22 trillion | 15T (Llama 3) | **1.5x** |
| **Behemoth** | 30+ trillion | 15T (Llama 3) | **2x+** |

**Key Changes**:
- **Doubled training data** overall compared to Llama 3
- **Multimodal from token 0**: Text, images, video jointly trained
- **200 languages**: 100+ languages with >1B tokens each
- **10x more multilingual**: Compared to Llama 3's multilingual data

**Context Windows**:

| Model | Training Context | Extended Context | Llama 3.1 Baseline |
|-------|------------------|------------------|-------------------|
| **Scout** | 131,072 (131K) | 10,000,000 (10M) | 128K (**78x larger**) |
| **Maverick** | Unknown | 1,000,000 (1M) | 128K (**7.8x larger**) |

### Data Mix: Multimodal by Default

**Modalities** (exact ratios not publicly disclosed):
- **Text**: Primary training data across 200 languages
- **Images**: Jointly trained with text from token 0
- **Video**: Native video understanding capability

**Data Sources**:
- Mix of publicly available data
- Licensed data
- **Meta's products and services**:
  - Publicly shared Instagram posts
  - Publicly shared Facebook posts
  - User interactions with Meta AI
- **Knowledge cutoff**: August 2024

**Comparison to Llama 3**:

| Aspect | Llama 3 | Llama 4 | Change |
|--------|---------|---------|--------|
| **Text Data** | 15T tokens | 30T+ tokens | **2x** |
| **Image Data** | Adapter training only (3.2) | **Native from start** | Revolutionary |
| **Video Data** | None | **Native** | New capability |
| **Languages** | ~100 | **200** | **2x** |
| **Multilingual Tokens** | Baseline | **10x more** | Massive increase |

### Infrastructure: Unprecedented GPU Scale

**Llama 3 vs Llama 4 Hardware**:

| Aspect | Llama 3 (405B) | Llama 4 (Scout/Maverick) | Llama 4 (Behemoth) | Scale-Up |
|--------|---------------|--------------------------|-------------------|----------|
| **Primary GPUs** | H100 80GB | H100 80GB | H100 80GB | Same gen |
| **GPU Count** | 16,384 H100s | **100,000+ H100s** | **32,000 H100s** | **6x-20x** |
| **Total GPU Hours** | 39.3M | Unknown | Unknown | - |
| **Training Precision** | BF16 | Likely BF16/FP8 | **FP8** | More efficient |
| **TFLOPs/GPU** | ~400 | Unknown | **390** (FP8) | Similar |

**Infrastructure Achievements**:
- **100,000+ H100 GPUs**: Largest training cluster in Llama history
- **FP8 precision** on Behemoth: 390 TFLOPs/GPU efficiency
- **7.38 million GPU hours**: Scout + Maverick combined
- **Massive parallelism**: Required advanced distributed training techniques

**Environmental Impact**:
- **Greenhouse gas emissions** (Scout + Maverick):
  - Location-based: 1,999 tons CO2eq
  - Market-based: 0 tons CO2eq (renewable energy purchases)

### Optimizer & Training Configuration

**Optimizer**: Likely AdamW (not explicitly disclosed, following Llama 3 precedent)

**Novel Training Techniques**:

1. **MetaP** (New):
   - Optimizes per-layer learning rates
   - Optimizes initialization scales
   - Enables more reliable training at extreme scale
   - Critical for MoE stability

2. **FP8 Precision Training**:
   - Achieved efficient training with FP8
   - 390 TFLOPs/GPU on Behemoth
   - Significant efficiency gains over BF16

3. **Long Context Extension**:
   - Mid-training phase with specialized datasets
   - Unlocked 10M context for Scout, 1M for Maverick
   - Enhanced model quality during extension
   - Continued training beyond base pre-training

### Multimodal Training Methodology

**Pre-training Approach**:
- **Early fusion**: Text, image, video as unified token sequence
- **Joint pre-training** with large unlabeled multimodal data
- **Vision encoder** (MetaCLIP-based) trained with frozen Llama model
- **Unified representation space** from token 0

**Comparison to Llama 3.2**:

| Aspect | Llama 3.2 Vision | Llama 4 | Advantage |
|--------|-----------------|---------|-----------|
| **Approach** | Adapter (vision added later) | **Native (from token 0)** | Better integration |
| **Text Model** | Pre-trained separately | **Joint training** | Unified understanding |
| **Cross-modal** | Limited (adapter layers) | **Deep (attention across all)** | Better reasoning |
| **Training Data** | 6B image-text pairs | **Multimodal from start** | More comprehensive |

### Post-Training: Revolutionary Approach

Meta revamped the entire post-training pipeline for Llama 4, achieving **10x efficiency improvement** over Llama 3's approach.

**Three-Stage Pipeline**:

**Stage 1: Lightweight Supervised Fine-Tuning (SFT)**:
- Used **Llama models as judges** to filter low-complexity prompts
- Removed >50% of data tagged as "easy"
- Fine-tuned only on high-difficulty tasks
- Highly pruned, curated dataset
- Initial instruction-following stage

**Stage 2: Intensive Online Reinforcement Learning (RL)**:
- Focus on hard prompts (pass@k analysis for coding, math, reasoning)
- **Continuous online learning cycle**:
  1. Model trains on hard prompts
  2. Generates new data
  3. Filters for medium-to-hard difficulty
  4. Creates dynamic learning curriculum
- **Adaptive, curriculum-based RL**
- Maintains proficiency across reasoning, coding, dialogue
- **~10x efficiency improvement** over Llama 3 (for Behemoth)
- Required revamping underlying RL infrastructure for 2T parameter model

**Stage 3: Lightweight Direct Preference Optimization (DPO)**:
- Applied to handle corner cases
- Focused on response quality
- Balance between intelligence and conversational abilities
- Addresses multimodal balance challenges

**Comparison to Llama 3**:

| Aspect | Llama 3 | Llama 4 | Impact |
|--------|---------|---------|--------|
| **SFT Approach** | Heavy SFT (10M+ examples) | **Lightweight (pruned difficult only)** | More efficient |
| **Primary Training** | Multiple rounds SFT + DPO | **Intensive online RL** | Better performance |
| **Curriculum** | Static datasets | **Dynamic, adaptive** | Continuous improvement |
| **Efficiency** | Baseline | **10x better** | Massive speedup |
| **Focus** | Broad coverage | **Hard prompt specialization** | Targeted improvement |

### Safety & Alignment

**Training-Time Safety**:

1. **GOAT** (Generative Offensive Agent Tester):
   - Used throughout training
   - Highlights LLM susceptibilities
   - Improves model safety proactively

2. **Safety Fine-Tuning Objectives**:
   - Provide readily available safe model
   - Reduce deployment workload
   - Resource for research community on robustness

**Evaluation & Red Teaming**:

1. **CBRN Risk Assessment** (Chemical, Biological, Radiological, Nuclear):
   - Expert-designed evaluations
   - Assesses capability increase for malicious actors
   - Targets proliferation of weapons

2. **Child Safety Risk Assessment**:
   - Expert team evaluation
   - Informs additional fine-tuning
   - In-depth red teaming exercises

3. **Additional Red Teaming**:
   - Content policy violations
   - Multi-modal safety concerns
   - Political/social topic handling

**Safety Tools Integration**:
- **Llama Guard**: Input/output safety classifier
- **Prompt Guard**: Jailbreak detection
- **Code Shield**: Code safety validation
- **CyberSecEval**: Cybersecurity evaluation

**Safety Results vs Llama 3**:

| Metric | Llama 3.3 | Llama 4 | Improvement |
|--------|-----------|---------|-------------|
| **Political/social refusal rate** | 7% | **<2%** | **71% reduction** |
| **False refusals** | Higher | **Lower** | Better usability |
| **Conversationality** | Good | **Better** | Improved tone |
| **System prompt steerability** | Good | **Enhanced** | More controllable |

### Training Innovations Summary

**Key Advancements Over Llama 3**:
1. **2x training data** with native multimodality
2. **100K+ GPU cluster** (6x larger than Llama 3)
3. **FP8 precision training** for efficiency
4. **MetaP** for per-layer optimization
5. **10x more efficient post-training** via online RL
6. **Dynamic curriculum learning** vs static datasets
7. **10M context extension** via specialized training
8. **Better safety** with lower false refusal rates

## Performance: Competitive with GPT-4o and Gemini

Llama 4 achieves competitive performance with leading proprietary models while being the first open-weight natively multimodal MoE family.

### Overall Competitiveness

**Llama 4 Maverick**: Matches or exceeds GPT-4o on broad benchmarks
- Crossed LMArena 1400 rating (beating GPT-4o)
- MMLU Pro: 80.5% (strong general knowledge)
- HumanEval: 82.4% (strong coding)
- **Caveat**: Public version received mixed feedback; advertised performance from unreleased experimental version

**Llama 4 Scout**: Strong performance for 17B active parameters
- MMLU Pro: 74.3% (competitive for size)
- HumanEval: 74.1% (excellent coding for 17B active)
- 10M context window (with practical limitations)

**Llama 4 Behemoth**: Still in training, early results impressive
- MATH-500: 95.0% (near-perfect math)
- MMLU Pro: 82.2% (exceeds Maverick)
- Targets frontier-level performance

### Pre-Trained Models Performance

#### Llama 3 vs Llama 4 Comparison

| Benchmark | Llama 3.1 70B | Llama 3.1 405B | Llama 4 Scout | Llama 4 Maverick | Scout vs 70B | Maverick vs 405B |
|-----------|---------------|----------------|---------------|------------------|--------------|------------------|
| **MMLU** | 79.3% | 85.2% | 79.6% | 85.5% | **+0.3** | **+0.3** |
| **MATH** | 41.6% | 53.5% | 50.3% | 61.2% | **+8.7** | **+7.7** |
| **MBPP** | 66.4% | 74.4% | 67.8% | 77.6% | **+1.4** | **+3.2** |

*Llama 4 shows improvements across the board despite having far fewer active parameters*

### Instruction-Tuned Models Performance

#### Core Benchmarks with Llama 3 Comparisons

| Benchmark | Llama 3.3 70B | Llama 3.1 405B | Llama 4 Scout | Llama 4 Maverick | Scout Δ | Maverick Δ |
|-----------|---------------|----------------|---------------|------------------|---------|------------|
| **MMLU Pro** | 68.9% | 73.4% | **74.3%** | **80.5%** | **+5.4** | **+7.1** |
| **GPQA Diamond** | 50.5% | 49.0% | **57.2%** | **69.8%** | **+6.7** | **+20.8** |
| **HumanEval** | ~60% | ~65% | **74.1%** | **82.4%** | **+14.1** | **+17.4** |
| **LiveCodeBench** | 33.3% | 27.7% | 32.8% | **43.4%** | -0.5 | **+15.7** |

*Massive improvements in coding (HumanEval) and reasoning (GPQA) over Llama 3*

#### Multimodal Benchmarks (New Capability)

| Benchmark | Llama 3.2 11B Vision | Llama 3.2 90B Vision | Llama 4 Scout | Llama 4 Maverick |
|-----------|---------------------|---------------------|---------------|------------------|
| **ChartQA** | — | — | 88.8% | 90.0% |
| **DocVQA** | — | — | 94.4% | 94.4% |

*Llama 4's native multimodal approach delivers strong vision performance*

### Llama 4 Behemoth Performance (Preliminary)

| Benchmark | Behemoth Score | Llama 3.1 405B | Improvement |
|-----------|---------------|----------------|-------------|
| **MATH-500** | **95.0%** | ~53.5% | **+41.5** |
| **MMLU Pro** | **82.2%** | 73.4% | **+8.8** |
| **LiveCodeBench** | **49.4%** | 27.7% | **+21.7** |
| **University Math** | **78.0%** | Unknown | - |

*Behemoth shows potential to significantly exceed Llama 3.1 405B across the board*

### Comparison to Leading Proprietary Models

#### vs GPT-4o

| Benchmark | GPT-4o | Llama 4 Scout | Llama 4 Maverick | Maverick vs GPT-4o |
|-----------|--------|---------------|------------------|-------------------|
| **MMLU** | 88.70% | 79.6% | 85.5% | -3.2 |
| **HumanEval** | 90.20% | 74.1% | 82.4% | -7.8 |
| **General Benchmarks** | Baseline | Competitive | **Beats per Meta** | +? |
| **LMArena** | <1400 | Unknown | **>1400** | **Better** |

*Maverick competitive with GPT-4o on many benchmarks, exceeds on some*

#### vs Gemini

| Model | Context | Multimodal | Performance vs Llama 4 |
|-------|---------|------------|------------------------|
| **Gemini 2.0 Flash** | 1M | Yes | **Maverick beats on broad benchmarks** |
| **Gemini 2.5 Pro** | 1M | Yes | Outperforms Scout/Maverick in raw scores |
| **Gemini 2.5 Pro** | 1M | Yes | Better long-context performance (90.6% at 120k vs 15.6% Scout) |

*Llama 4 competitive but Gemini 2.5 Pro leads on some metrics*

#### vs Claude

| Model | HumanEval | Context | Notes |
|-------|-----------|---------|-------|
| **Claude 3.5 Sonnet** | 92.00% | 200K | Beats Maverick on coding |
| **Claude 3.7 Sonnet** | Unknown | 200K | Behemoth outperforms on STEM |
| **Llama 4 Maverick** | 82.4% | 1M | Strong but trails Claude coding |
| **Llama 4 Scout** | 74.1% | **10M** | Extreme context advantage |

*Claude still leads on coding, but Llama 4 has massive context advantage*

#### vs DeepSeek V3

| Aspect | DeepSeek V3 | Llama 4 Maverick | Advantage |
|--------|-------------|------------------|-----------|
| **Active Parameters** | >17B | 17B | Llama 4 more efficient |
| **Reasoning/Coding** | Strong | **Comparable** | Similar performance |
| **Architecture** | MoE | MoE | Both sparse |
| **Open Source** | Yes | Yes | Both available |

*Maverick achieves comparable results with fewer active parameters*

### Strengths and Weaknesses

**Strengths**:
- **MoE efficiency**: 17B active delivering 400B capacity performance
- **Multimodal**: Native text/image/video understanding
- **Math reasoning**: Massive improvements (MATH: +7.7 to +8.7 over Llama 3)
- **Coding**: Major gains (HumanEval: +14.1 to +17.4 over Llama 3)
- **Context window**: 10M tokens (Scout) enables new use cases
- **Safety**: Better usability (<2% political refusal vs 7% Llama 3.3)

**Weaknesses**:
- **Context degradation**: Advertised 10M, but 15.6% accuracy at 120k (vs Gemini 90.6%)
- **Mixed reception**: Public Maverick version underperformed advertised benchmarks
- **Trails Claude**: Still behind Claude 3.5 on coding (92% vs 82.4%)
- **Trails GPT-4o**: Slightly behind on some benchmarks (MMLU, HumanEval)
- **Behemoth delays**: Flagship model still not released, creating uncertainty

### Per-Model-Size Analysis

**Scout (17B active, 109B total)**:
- **Best for**: Extreme context applications (10M tokens)
- **Performance**: Competitive with Llama 3.1 70B despite 4.1x fewer active params
- **Trade-off**: Context window impressive but practical performance degrades
- **Use case**: Large codebase analysis, multi-book processing

**Maverick (17B active, 400B total)**:
- **Best for**: High-capacity multimodal tasks, general-purpose deployment
- **Performance**: Matches/exceeds Llama 3.1 405B with 23.8x fewer active params
- **Trade-off**: Public version vs experimental version performance gap
- **Use case**: Production deployments requiring GPT-4 class performance

**Behemoth (288B active, ~2T total)**:
- **Best for**: Frontier-level reasoning, math, coding
- **Performance**: Early results show massive improvements (MATH-500: 95%)
- **Trade-off**: Still in training, delayed release
- **Use case**: When absolute best performance needed

### The Llama 3 → Llama 4 Performance Leap

| Category | Llama 3.1 405B | Llama 4 Maverick (17B active) | Change |
|----------|---------------|------------------------------|--------|
| **MMLU Pro** | 73.4% | 80.5% | **+7.1** |
| **GPQA** | 49.0% | 69.8% | **+20.8** |
| **HumanEval** | ~65% | 82.4% | **+17.4** |
| **Context** | 128K | 1M | **7.8x** |
| **Active Params** | 405B | **17B** | **23.8x fewer** |

**Key Insight**: Llama 4 Maverick delivers better performance than Llama 3.1 405B with 23.8x fewer active parameters—validating the MoE architecture's efficiency.

## Key Innovations: Pushing the Frontier of Open AI

Llama 4 introduces several breakthrough innovations that mark a fundamental departure from the Llama 3 architecture and methodology, bringing open-source models to the frontier of AI capabilities.

### 1. First Open-Weight Natively Multimodal MoE Models

**What's New**: Llama 4 combines two frontier techniques—Mixture-of-Experts (MoE) and native multimodality—in an open-weight model for the first time.

**Llama 3 Baseline**:
- **Architecture**: Dense transformers (all parameters active every token)
- **Multimodal**: Adapter-based approach (Llama 3.2 Vision)
  - Text model trained separately
  - Vision encoder added later via cross-attention adapters
  - Limited cross-modal understanding

**Llama 4 Advancement**:
- **MoE Architecture**: Sparse activation (only 17B of 400B params active per token)
  - **Efficiency**: 23.8x fewer FLOPs than Llama 3.1 405B dense model
  - **Capacity**: Maintains 400B parameter capacity while computing with 17B
  - **Specialization**: 128 experts (Maverick) can specialize in different domains
- **Native Multimodality**: Early fusion from token 0
  - Text, images, video jointly trained from the start
  - Unified latent space across all modalities
  - Deep cross-modal attention throughout model
  - No adapter layers needed

**Why This Matters**:
- **Democratization**: Previously, only proprietary models (GPT-4, Gemini, Claude 3.5) combined these techniques
- **Open research**: Enables community to study MoE + multimodal architectures
- **Efficiency**: Makes frontier-level capabilities accessible at lower compute cost
- **Future-proofing**: MoE + multimodal is the new standard for state-of-the-art models

**Technical Implementation**:
```
Llama 3.2 Vision:
1. Pre-train text model (dense) → 2. Freeze weights → 3. Train vision encoder separately →
4. Add cross-attention adapters → 5. Fine-tune adapters only

Llama 4:
1. Joint pre-training (text + images + video as unified token sequence from token 0) →
2. Native attention across all modalities → 3. No separate encoders/adapters needed
```

### 2. Extreme Context Windows via iRoPE

**What's New**: Scout achieves 10 million token context—78x larger than Llama 3.1—through novel iRoPE (Interleaved Rotary Position Embeddings) architecture.

**Llama 3 Baseline**:
- **Llama 3**: 8K context window (standard RoPE)
- **Llama 3.1**: 128K context window (RoPE scaling)
  - Near limits of traditional RoPE scaling
  - Quality degradation beyond ~200K tokens

**Llama 4 Advancement**:

| Model | Context Window | Llama 3.1 Baseline | Increase |
|-------|---------------|-------------------|----------|
| **Scout** | **10,000,000** | 128,000 | **78x** |
| **Maverick** | **1,000,000** | 128,000 | **7.8x** |

**Technical Innovation - iRoPE Components**:

1. **Interleaved Attention Layers**:
   - **NoPE layers** (every 4th layer): No positional encoding, global attention
   - **RoPE layers** (3 out of 4 layers): Standard rotary embeddings with chunking
   - Enables model to balance local and global dependencies

2. **Chunked Attention** (8,192 token chunks):
   - Local attention computed within chunks in RoPE layers
   - Reduces memory footprint from O(n²) to O(n×chunk_size)
   - Maintains quality while enabling extreme lengths

3. **Inference-Time Temperature Scaling**:
   - Scales attention scores to focus on relevant context parts
   - Critical for finding relevant information in 10M token sequences

4. **Specialized Long-Context Training**:
   - Mid-training phase with long-context datasets
   - Continued training to unlock extreme context
   - Quality-enhancing during extension (not just extrapolation)

**Why This Matters**:
- **New use cases**: Entire large codebases, multiple books, years of conversation history
- **Context=Memory**: Models can maintain context across previously impossible scales
- **Competitive**: Matches Gemini 1.5 (1M-2M context) in scale

**Practical Limitations**:
- **Advertised**: 10M tokens (Scout)
- **Reality**: Significant degradation at 120K tokens (15.6% accuracy on Fiction.LiveBench)
- **Comparison**: Gemini 2.5 Pro at 120K: 90.6% accuracy
- **Gap**: Significant difference between claimed and actual long-context performance

### 3. Revolutionary Post-Training: 10x Efficiency Improvement

**What's New**: Complete overhaul of post-training pipeline achieving 10x efficiency over Llama 3 through online RL and dynamic curriculum learning.

**Llama 3 Approach**:
- **Heavy SFT**: 10M+ supervised fine-tuning examples
- **Multiple rounds**: Iterative SFT + DPO (Direct Preference Optimization)
- **Static datasets**: Fixed training data throughout
- **Broad coverage**: All difficulty levels included
- **Efficiency**: Baseline

**Llama 4 Approach**:

**Stage 1 - Lightweight SFT**:
- **Llama-as-Judge**: Used Llama models to filter training data
- **Pruning**: Removed >50% of data tagged as "easy" or "low complexity"
- **Focus**: Only high-difficulty tasks for initial instruction-following
- **Result**: Highly curated, pruned dataset

**Stage 2 - Intensive Online RL** (Primary Innovation):
- **Hard prompt selection**: Used pass@k analysis for coding, math, reasoning
- **Continuous learning cycle**:
  1. Model trains on hard prompts
  2. Generates new data from interactions
  3. Filters for medium-to-hard difficulty
  4. Creates dynamic, adaptive curriculum
  5. Repeat continuously
- **Adaptive curriculum**: Difficulty increases as model improves
- **Multi-domain balance**: Maintains proficiency across reasoning, coding, dialogue
- **Efficiency**: **~10x improvement** over Llama 3 (for Behemoth)

**Stage 3 - Lightweight DPO**:
- Applied to corner cases only
- Balances intelligence and conversational abilities
- Handles multimodal balance challenges

**Comparison Table**:

| Aspect | Llama 3 | Llama 4 | Impact |
|--------|---------|---------|--------|
| **SFT Data Volume** | 10M+ examples | **Pruned (50%+ removed)** | More efficient |
| **Data Selection** | All difficulty levels | **Hard prompts only** | Targeted learning |
| **Primary Training** | Multiple SFT rounds | **Intensive online RL** | Better performance |
| **Curriculum** | Static datasets | **Dynamic, adaptive** | Continuous improvement |
| **Learning Loop** | Offline (fixed data) | **Online (self-generated)** | Self-improving |
| **Efficiency** | Baseline | **10x faster** | Massive speedup |
| **Infrastructure** | Standard RL | **Revamped for 2T params** | Scaled to Behemoth |

**Why This Matters**:
- **Cost reduction**: 10x efficiency = 10x lower post-training cost
- **Better results**: Dynamic curriculum targets model weaknesses
- **Scalability**: Online RL enables continuous improvement
- **Community impact**: More efficient fine-tuning methods for open models

### 4. MetaP Optimizer: Per-Layer Learning Rate Optimization

**What's New**: Novel optimizer that adjusts learning rates and initialization scales per layer, enabling stable training at extreme MoE scale.

**Llama 3 Baseline**:
- **Optimizer**: AdamW with global learning rate
- **Learning rate**: Single peak LR for entire model
- **Initialization**: Standard scaling across all layers
- **Challenge**: Works well for dense models up to 405B

**Llama 4 Advancement**:
- **MetaP**: Optimizes per-layer learning rates individually
- **Per-layer initialization**: Optimizes initialization scales per layer
- **MoE stability**: Critical for training 400B MoE with 128 experts
- **Scale enabler**: Required for Behemoth's 2T parameter MoE training

**Why This Matters**:
- **MoE challenges**: Different expert layers need different learning rates
- **Training stability**: Prevents divergence in massive MoE models
- **Efficiency**: Faster convergence with per-layer optimization
- **Future scaling**: Enables even larger MoE models beyond Behemoth

### 5. FP8 Precision Training: Efficiency at Scale

**What's New**: Successfully trained models using FP8 (8-bit floating point) precision, significantly reducing compute and memory requirements.

**Llama 3 Baseline**:
- **Precision**: BF16 (16-bit brain floating point)
- **Memory**: 2 bytes per parameter
- **Compute**: Standard FLOPs for 16-bit operations
- **405B model**: ~810GB memory for parameters alone

**Llama 4 Advancement**:
- **Precision**: FP8 (8-bit floating point)
- **Memory**: 1 byte per parameter (50% reduction)
- **Compute**: 390 TFLOPs/GPU on H100 (vs ~300 for BF16)
- **Throughput**: Significant speedup in training

**Efficiency Gains**:

| Aspect | BF16 (Llama 3) | FP8 (Llama 4) | Improvement |
|--------|----------------|---------------|-------------|
| **Memory/param** | 2 bytes | 1 byte | **50% reduction** |
| **TFLOPs/GPU** | ~300 | **390** | **30% increase** |
| **Model size** | 400B × 2 = 800GB | 400B × 1 = 400GB | **50% smaller** |
| **Training speed** | Baseline | **Faster** | Throughput gain |

**Why This Matters**:
- **Cost reduction**: Lower memory = more model fits per GPU = lower cost
- **Speed**: Higher TFLOPs = faster training iterations
- **Scalability**: Enables training even larger models (Behemoth's 2T params)
- **Inference**: FP8 models can run inference with half the memory

### 6. 100,000+ GPU Training Infrastructure

**What's New**: Largest training cluster in Llama history—100,000+ H100 GPUs—requiring advanced distributed training techniques.

**Llama 3 Baseline**:
- **GPU count**: 16,384 H100 GPUs
- **Scale**: Already massive by industry standards
- **405B training**: Required 39.3M GPU hours

**Llama 4 Advancement**:
- **Scout/Maverick**: **100,000+ H100 GPUs**
- **Behemoth**: 32,000 H100 GPUs (dedicated cluster)
- **Scale**: **6x-20x larger** than Llama 3
- **Total GPU hours**: 7.38M (Scout + Maverick combined)

**Infrastructure Challenges Solved**:
1. **Communication overhead**: 100K GPU cluster requires extremely fast interconnects
2. **Fault tolerance**: At this scale, hardware failures are constant
3. **Load balancing**: Ensuring all 100K GPUs stay busy
4. **Synchronization**: Gradient synchronization across 100K devices
5. **Memory management**: Coordinating distributed MoE expert placement

**Why This Matters**:
- **Frontier capabilities**: Only achievable with this scale of compute
- **Open AI competitiveness**: Matches proprietary model training scales
- **Future-proofing**: Infrastructure ready for even larger future models
- **Environmental**: Despite scale, market-based emissions: 0 tons CO2eq (renewable energy)

### Llama 3 → Llama 4 Innovation Summary

| Innovation | Llama 3 | Llama 4 | Impact on Community |
|------------|---------|---------|---------------------|
| **Architecture** | Dense | **MoE + Multimodal** | First open MoE+multimodal |
| **Context** | 128K (RoPE scaling) | **10M (iRoPE)** | New use cases unlocked |
| **Post-training** | Static SFT/DPO | **Online RL (10x efficient)** | Cheaper fine-tuning |
| **Optimizer** | AdamW (global LR) | **MetaP (per-layer)** | Enables MoE stability |
| **Precision** | BF16 | **FP8** | 50% memory reduction |
| **Infrastructure** | 16K GPUs | **100K+ GPUs** | Frontier-scale training |
| **Training tokens** | 15T | **30T+** | 2x more data |
| **Modalities** | Text (+ adapter vision) | **Native text/image/video** | True multimodal from start |

**The Bottom Line**: Llama 4 represents the most significant architectural and methodological leap in the Llama family, bringing together innovations that make frontier capabilities—previously exclusive to proprietary models—available to the open-source community. While execution has faced challenges (context degradation, Behemoth delays), the technical innovations push the boundaries of what's possible in open AI.

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
