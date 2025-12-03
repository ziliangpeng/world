# Ministral 3

**Release Date:** December 2, 2025
**Developer:** Mistral AI
**Model Sizes:** 3B, 8B, 14B parameters (9 total variants)
**Variants:** Base, Instruct, Reasoning (3 variants per size)
**Context Window:** 256,000 tokens (128K on some platforms)
**License:** Apache 2.0
**Architecture:** Dense Transformer with Vision

## Overview

Ministral 3 is Mistral AI's family of edge-optimized, multimodal language models released on December 2, 2025, designed specifically for on-device deployment, robotics, autonomous drones, and edge computing scenarios. The family comprises **nine distinct models** across three parameter sizes—**3 billion, 8 billion, and 14 billion**—each available in three specialized variants: **Base** (pre-trained foundation), **Instruct** (chat-optimized), and **Reasoning** (optimized for complex logic and analytical tasks).

What sets Ministral 3 apart is its focus on **efficiency and accessibility**. While Mistral Large 3 targets data center deployments with 675B parameters, Ministral 3 models are engineered to run on consumer-grade hardware: the **3B model fits in less than 8GB of VRAM when quantized**, the 8B model runs comfortably on 24GB VRAM, and the 14B model requires only 32GB VRAM. This makes frontier-quality AI accessible to researchers, developers, and organizations without access to massive GPU clusters.

The Ministral 3 Reasoning variants achieve **state-of-the-art performance in their weight class**, with the **14B Reasoning model scoring 85% on AIME 2025** (American Invitational Mathematics Examination)—an extraordinary achievement for a model of this size, rivaling much larger models. The reasoning models are trained using reinforcement learning (RL) techniques derived from Mistral's Magistral research, enabling them to tackle complex mathematical, logical, and analytical problems that typically require significantly larger models.

All Ministral 3 models feature **integrated vision capabilities** through a dedicated 0.4B-0.5B parameter vision encoder, enabling multimodal reasoning with images and text. This universal vision support across all sizes makes Ministral 3 uniquely suited for robotics applications where visual perception must be combined with natural language understanding and planning. The models also deliver exceptional **token efficiency**, producing comparable quality outputs with **an order of magnitude fewer tokens** than competing models—critical for reducing inference costs and latency in production deployments.

Ministral 3 achieves remarkable **edge deployment performance** through extensive optimization for NVIDIA platforms. On the **RTX 5090**, the 3B Instruct model reaches **385 tokens/second**, while on **NVIDIA Jetson Thor** (designed for robotics), the same model achieves **52-273 tokens/second** depending on configuration. These performance figures enable real-time inference for autonomous systems, drones, and interactive applications where millisecond-level latency is critical.

Released under the **Apache 2.0 license**, Ministral 3 provides complete freedom for commercial use, modification, and redistribution without restrictions. Models are available on Hugging Face, Ollama, and major cloud platforms, with support for popular inference frameworks including vLLM, llama.cpp, and Ollama for easy deployment.

**Official Resources:**
- [Mistral 3 Announcement](https://mistral.ai/news/mistral-3) (Mistral AI Blog, December 2, 2025)
- [Ministral 3 Collection](https://huggingface.co/collections/mistralai/ministral-3) (Hugging Face)
- [Ministral 3B Model Card](https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512)
- [Ministral 8B Model Card](https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512)
- [Ministral 14B Model Card](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512)
- [NVIDIA Edge AI Blog](https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models-deliver-efficiency-accuracy-at-any-scale/)

---

## Model Architecture

Ministral 3 uses a **dense transformer architecture** (not Mixture-of-Experts) optimized for edge deployment with integrated multimodal capabilities.

### Model Lineup

The Ministral 3 family consists of **nine models** organized by size and specialization:

| Size | Variant | Total Params | Vision Encoder | Context | Primary Use Case | Memory (BF16) |
|------|---------|--------------|----------------|---------|------------------|---------------|
| **3B** | Base | 4.0B | 0.4B | 256K | Pre-training foundation, research | ~8 GB |
| **3B** | Instruct | 4.0B | 0.4B | 256K | Chat, assistants, general tasks | ~8 GB |
| **3B** | Reasoning | 4.0B | 0.4B | 256K | Math, logic, analytical reasoning | ~8 GB |
| **8B** | Base | ~9.0B | 0.4B | 256K | Pre-training foundation, research | ~18 GB |
| **8B** | Instruct | ~9.0B | 0.4B | 256K | Chat, assistants, edge deployment | ~18 GB |
| **8B** | Reasoning | ~9.0B | 0.4B | 256K | Complex reasoning, problem-solving | ~18 GB |
| **14B** | Base | ~14B | 0.4B | 256K | Pre-training foundation, research | ~28 GB |
| **14B** | Instruct | ~14B | 0.4B | 256K | Advanced chat, enterprise edge | ~28 GB |
| **14B** | Reasoning | ~14B | 0.4B | 256K | Advanced reasoning, mathematics | ~28 GB |

**Model Naming Convention:**
- `Ministral-3-{SIZE}B-{VARIANT}-2512`
- Example: `Ministral-3-14B-Reasoning-2512`
- "2512" indicates December 2025 release

### Core Architecture Specifications

**Ministral 3B Architecture:**

```yaml
Total Parameters: ~4.0 billion
  Language Model: ~3.4-3.6B parameters
  Vision Encoder: ~0.4B parameters

Model Dimensions:
  Hidden Size (d_model): Estimated ~3072-4096
  Intermediate Size (FFN): Estimated ~8192-12288
  Layers: Estimated 24-32 layers

  Note: Exact specifications not publicly disclosed
        Uses custom Mistral config format

Attention Mechanism:
  Type: Grouped Query Attention (GQA)
  Query Heads: Estimated 24-32
  Key-Value Heads: Estimated 4-8 (GQA for efficiency)
  Head Dimension: Estimated 128

  KV Cache Efficiency:
    - GQA reduces KV cache size vs MHA
    - Critical for 256K context on limited memory
    - 4-8× reduction in KV cache requirements

Position Embeddings:
  Type: Rotary Position Embeddings (RoPE)
  Base Frequency (Theta): >1M for 256K context support
  Max Position Embeddings: 256,000 tokens

  Long Context Enablement:
    - Native 256K support (no extension needed)
    - RoPE with high theta for long-range dependencies
    - Efficient attention patterns

Activation Function:
  Type: SwiGLU (Swish-Gated Linear Unit)
  Formula: SwiGLU(x, W, V) = Swish(xW) ⊗ (xV)
  Application: Feed-forward network layers

  Benefits:
    - Better gradient flow than ReLU/GELU
    - Improved model quality
    - Standard across Mistral family

Normalization:
  Type: RMSNorm (Root Mean Square Layer Normalization)
  Formula: RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ
  Epsilon: 1e-05
  Application: Pre-normalization (before attention and FFN)

  Advantages:
    - More efficient than LayerNorm
    - No mean centering required
    - Faster computation on edge devices

Vocabulary:
  Size: Likely 32,768 or 131,072 tokens
  Tokenizer: Mistral tokenizer (custom format)
  Type: SentencePiece-based with optimized vocabulary

Precision Support:
  Training: BF16 (bfloat16) for Reasoning variants
  Inference:
    - FP8: Instruct variants (3B, 8B, 14B)
    - BF16: Reasoning variants (standard)
    - INT8: Quantized variants (community)
    - INT4/GGUF: Extreme quantization (Ollama, llama.cpp)

Vision Encoder:
  Parameters: ~0.4 billion (400M)
  Architecture: Likely ViT-based (Vision Transformer)
  Integration: Multimodal fusion layers

  Image Processing:
    - Input: Images up to 10MB
    - Max images per request: 8 images
    - Output: Visual features for language model
```

**Ministral 8B Architecture:**

```yaml
Total Parameters: ~9.0 billion
  Language Model: ~8.4-8.6B parameters
  Vision Encoder: ~0.4B parameters

Model Dimensions:
  Hidden Size (d_model): 4096
  Intermediate Size (FFN): 14336
  Layers: 32-36 (exact count varies)

Attention Mechanism:
  Type: Grouped Query Attention (GQA)
  Query Heads: 32
  Key-Value Heads: 8 (4:1 ratio)
  Head Dimension: 128

  Interleaved Sliding-Window Attention:
    - Alternates between full and sliding window attention
    - Pattern: 1 full attention + 3 sliding window layers
    - Enables efficient long-context processing
    - Reduces memory bandwidth requirements

Position Embeddings:
  Type: Rotary Position Embeddings (RoPE)
  Context Window: 256K tokens (native)
  Platform Note: Some platforms (vLLM) default to 32K

Vocabulary:
  Size: 32,000 tokens (verified from documentation)
  Tokenizer: Mistral tokenizer

Context Window Management:
  Native: 256,000 tokens
  Practical (vLLM): 32,768 tokens (configurable to 256K)
  Recommendation: 128K for best performance/memory balance
```

**Ministral 14B Architecture:**

```yaml
Total Parameters: ~14 billion
  Language Model: ~13.5-13.6B parameters
  Vision Encoder: ~0.4B parameters

Model Dimensions:
  Hidden Size: Estimated 4096-5120
  Intermediate Size: Estimated 16384-20480
  Layers: Estimated 36-48

Attention Mechanism:
  Type: Grouped Query Attention (GQA)
  Query Heads: Estimated 32-40
  Key-Value Heads: Estimated 8 (reduced for efficiency)

Memory Requirements:
  BF16 (Full Precision): 32GB VRAM
  FP8 (Instruct Quantized): 16GB VRAM
  INT4 (Extreme Quantization): <8GB VRAM

  Practical Deployment:
    - Single RTX 4090 (24GB): Possible with quantization
    - Single H100 (80GB): Comfortable, can batch
    - Jetson AGX Orin: With quantization (tight)
    - Consumer Desktop: 32GB GPU + quantization
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Ministral 3 Architecture                      │
│              Dense Transformer + Vision (3B/8B/14B)              │
└─────────────────────────────────────────────────────────────────┘

INPUT PROCESSING
─────────────────
Text Tokens              Images (Optional, up to 8)
     ↓                          ↓
[Embedding Layer]      [Vision Encoder: 0.4B params]
32K vocab                       ↓
     ↓                  [Vision Features]
     └──────────┬──────────────┘
                ↓
        [Multimodal Fusion]
                ↓

TRANSFORMER LAYERS
───────────────────

    ┌──────────────────────────────────────┐
    │   Transformer Layer (×N layers)      │
    │   N = 24-32 (3B), 32-36 (8B),        │
    │       36-48 (14B) estimated           │
    └──────────────────────────────────────┘
                ↓
          [RMSNorm]
                ↓
    ┌──────────────────────────────────────┐
    │    Grouped Query Attention (GQA)     │
    │  • Multiple query heads               │
    │  • Reduced KV heads (4-8× less)       │
    │  • RoPE positional encoding           │
    │  • 256K context support               │
    │                                       │
    │  8B Special: Interleaved Attention    │
    │  • 1 full attention layer             │
    │  • 3 sliding window layers            │
    │  • Repeat pattern                     │
    └──────────────────────────────────────┘
                ↓
       [Residual Connection]
                ↓
          [RMSNorm]
                ↓
    ┌──────────────────────────────────────┐
    │   Feed-Forward Network (Dense)       │
    │                                       │
    │    [Linear: d_model → FFN_size]      │
    │              ↓                        │
    │         [SwiGLU Activation]          │
    │              ↓                        │
    │    [Linear: FFN_size → d_model]      │
    │                                       │
    │  Note: NOT Mixture-of-Experts        │
    │        All parameters active          │
    └──────────────────────────────────────┘
                ↓
       [Residual Connection]
                ↓
           (Next Layer)

OUTPUT GENERATION
─────────────────
          [RMSNorm]
                ↓
          [LM Head]
                ↓
    [Output Logits: vocab_size]
                ↓
      Next Token Prediction
```

### Design Philosophy

```yaml
Edge-First Design:

  Compact Scale (3B-14B):
    - Fits on consumer GPUs and edge devices
    - 3B: <8GB VRAM (quantized)
    - 8B: <12GB VRAM (quantized)
    - 14B: <24GB VRAM (quantized)

  Efficient Architecture:
    - Dense (not MoE) for predictable latency
    - GQA for reduced KV cache
    - Interleaved attention (8B) for long context efficiency
    - No expert routing overhead

  Memory Optimization:
    - Aggressive quantization support (FP8, INT4)
    - PagedAttention compatibility
    - KV cache optimization critical for 256K context
    - Prefix caching for repeated prompts

Dense vs MoE Trade-off:

  Why Dense for Edge:
    - Predictable inference latency (no routing)
    - Simpler deployment (no expert scheduling)
    - Better for single-request scenarios
    - Lower memory bandwidth requirements
    - Easier quantization (uniform weight distribution)

  MoE Disadvantages for Edge:
    - Expert routing adds latency
    - Load balancing complexity
    - Higher memory bandwidth (expert switching)
    - Less efficient for small batch sizes

  Result:
    - Ministral 3: Dense architecture for edge
    - Mistral Large 3: MoE for data center scale

Multimodal Integration:

  Universal Vision:
    - All sizes include 0.4B vision encoder
    - Enables robotics, drones, visual inspection
    - Unified architecture (not separate vision model)

  Efficient Vision Processing:
    - Small vision encoder (0.4B, not multi-billion)
    - Fast image encoding
    - Minimal overhead on edge devices

Long Context on Edge:

  256K Native Support:
    - Longer than most edge models (32K-128K typical)
    - Enables document analysis on-device
    - Maintains conversation history
    - Analyzes codebases locally

  Efficiency Techniques:
    - GQA reduces KV cache dramatically
    - Sliding window attention (8B)
    - Progressive context length (start small)
    - Prefix caching for common prompts

Three Specialized Variants:

  Base:
    - Pure pre-trained foundation
    - For research and fine-tuning
    - No instruction tuning
    - Completion-based (not chat)

  Instruct:
    - Chat-optimized
    - Instruction following
    - Function calling support
    - General-purpose edge AI

  Reasoning:
    - RL-trained for complex logic
    - Math and analytical tasks
    - Multi-step problem solving
    - State-of-the-art for size class

  Benefits:
    - Choose right tool for task
    - Instruct for general use
    - Reasoning for complex problems
    - Base for custom fine-tuning
```

---

## Edge Deployment Deep Dive

Ministral 3 is engineered specifically for edge and on-device deployment, with extensive optimizations for NVIDIA edge platforms.

### Edge Hardware Platforms

**NVIDIA GeForce RTX (Consumer GPUs):**

```yaml
RTX 5090 (2025):
  VRAM: 32GB GDDR7 (expected)
  Performance (3B Instruct): 385 tokens/second
  Suitable Models: All Ministral 3 variants (3B, 8B, 14B)

  Deployment:
    - 3B FP8: ~4GB (plenty of headroom)
    - 8B FP8: ~9GB (comfortable)
    - 14B FP8: ~16GB (good margin)
    - Can batch multiple requests

  Use Cases:
    - Developer workstations
    - AI PC applications
    - Local LLM serving
    - Privacy-sensitive workloads

RTX 4090 (2023):
  VRAM: 24GB GDDR6X
  Performance (3B): ~300-350 tokens/second (estimated)
  Suitable Models: 3B, 8B, 14B (with quantization)

  Deployment:
    - 3B: Easy (<8GB)
    - 8B: Comfortable (<12GB)
    - 14B: Tight (16-20GB, INT8 recommended)

RTX 4080 / 4070 Ti:
  VRAM: 16GB GDDR6X
  Suitable Models: 3B, 8B (quantized)

  Deployment:
    - 3B: Excellent fit
    - 8B: INT8/FP8 recommended
    - 14B: INT4 only (quality trade-off)

RTX 3090 / 3090 Ti:
  VRAM: 24GB GDDR6X
  Performance: Slightly slower than 4090
  Suitable Models: 3B, 8B, 14B

  Note: Older architecture, but large VRAM

Consumer Recommendations:

  Budget (~$500-800):
    - RTX 4070 Ti (16GB)
    - Run 3B comfortably, 8B with quantization

  Mid-Range (~$1000-1500):
    - RTX 4080 (16GB) or RTX 4090 (24GB used)
    - Run all Ministral 3 models

  High-End (~$1500-2500):
    - RTX 5090 (32GB) or RTX 4090 (24GB new)
    - Best performance, can batch requests

  Multi-GPU (Enthusiast):
    - 2× RTX 4090 (48GB total)
    - Serve multiple models simultaneously
    - Higher throughput for applications
```

**NVIDIA Jetson (Embedded AI Platforms):**

```yaml
Jetson AGX Orin (2022):
  GPU: 2048 CUDA cores, 64 Tensor cores
  Memory: 32GB or 64GB unified memory
  TDP: 15-60W (configurable)
  Performance: 275 TOPS (INT8)

  Ministral 3 Performance:
    - 3B: Good performance with quantization
    - 8B: Possible with INT8/INT4
    - 14B: Challenging (tight memory)

  Use Cases:
    - Autonomous mobile robots (AMR)
    - Industrial automation
    - Smart cameras
    - Edge AI gateways

Jetson Orin Nano (2023):
  GPU: 1024 CUDA cores, 32 Tensor cores
  Memory: 8GB unified memory
  TDP: 7-15W
  Performance: 40-67 TOPS (INT8)

  Ministral 3 Performance:
    - 3B INT4: Feasible
    - 8B/14B: Not recommended (memory)

  Use Cases:
    - Compact robots
    - Drones (lightweight)
    - IoT devices with AI

Jetson Thor (2025):
  GPU: Next-gen architecture
  Memory: High-bandwidth unified memory
  Performance: Significantly improved

  Ministral 3B Performance: 52-273 tokens/second
    - Range depends on precision and batch size
    - 52 TPS: High precision, conservative
    - 273 TPS: Aggressive quantization, optimized

  Significance:
    - Real-time inference for robotics
    - Multiple inferences per second
    - Enables reactive autonomous systems

  Use Cases:
    - Humanoid robots (NVIDIA GR00T)
    - Advanced AMRs
    - Autonomous drones
    - Industrial robotic arms

Jetson Platform Comparison:

  | Platform | VRAM | TOPS (INT8) | 3B Performance | Best Use |
  |----------|------|-------------|----------------|----------|
  | Orin Nano | 8GB | 40-67 | Moderate (INT4) | Compact robots |
  | Orin NX | 16GB | 100-170 | Good (INT8) | Smart cameras |
  | AGX Orin | 32-64GB | 200-275 | Excellent | AMRs, industrial |
  | Thor | TBD | >>275 | 52-273 TPS | Humanoids, drones |

Edge Deployment Recommendations:

  For Drones:
    - Jetson Orin NX or Thor
    - Ministral 3B (INT4/INT8)
    - Power budget: 10-15W
    - Weight: <200g with cooling

  For Mobile Robots:
    - Jetson AGX Orin or Thor
    - Ministral 3B or 8B
    - Power budget: 30-60W
    - Allows larger battery for runtime

  For Fixed Industrial:
    - Jetson AGX Orin or RTX 4090
    - Any Ministral 3 size
    - No power/weight constraints
    - Maximize performance
```

**Performance Benchmarks by Platform:**

```yaml
Ministral 3B Instruct Performance:

  RTX 5090 (32GB):
    Throughput: 385 tokens/second
    Precision: FP8
    Batch Size: 1
    Context: 4K (typical)

  RTX 4090 (24GB):
    Throughput: ~320-350 tokens/second (estimated)
    Precision: FP8
    Batch Size: 1
    Context: 4K

  Jetson Thor:
    Throughput: 52-273 tokens/second
    Precision: INT8-INT4
    Variation: Configuration dependent
    Context: 2K-4K

  Jetson AGX Orin:
    Throughput: ~30-60 tokens/second (estimated)
    Precision: INT8
    Batch Size: 1
    Context: 2K

Ministral 8B Instruct Performance (Estimated):

  RTX 5090: ~200-250 tokens/second
  RTX 4090: ~150-200 tokens/second
  Jetson Thor: ~25-100 tokens/second
  Jetson AGX Orin: ~15-30 tokens/second

Ministral 14B Reasoning Performance (Estimated):

  RTX 5090: ~120-160 tokens/second
  RTX 4090: ~80-120 tokens/second
  Jetson Thor: ~15-50 tokens/second (INT4)
  Jetson AGX Orin: Not recommended (memory)

Comparison to Cloud Inference:

  Ministral 3B on RTX 5090: 385 TPS
  GPT-3.5 Turbo (API): ~50-100 TPS (network dependent)
  Mistral Large 3 (8×H200): ~20-30 TPS per user (shared)

  Advantage:
    - Edge inference: No network latency
    - Predictable performance
    - Data privacy (stays local)
    - No API costs
```

### Memory Requirements and Quantization

```yaml
Ministral 3B Memory Footprint:

  BF16 (Full Precision):
    Model Weights: ~8 GB
    KV Cache (4K context): ~0.5 GB
    Activations: ~0.2 GB
    Total: ~8.7 GB

    Fits on: 16GB VRAM (comfortable)

  FP8 (Instruct Variant):
    Model Weights: ~4 GB
    KV Cache (4K context): ~0.25 GB
    Activations: ~0.2 GB
    Total: ~4.45 GB

    Fits on: 8GB VRAM (comfortable)

  INT8 (8-bit Quantization):
    Model Weights: ~4 GB
    KV Cache (4K context): ~0.25 GB
    Activations: ~0.2 GB
    Total: ~4.45 GB

    Fits on: 8GB VRAM (plenty of margin)

  INT4 (4-bit Quantization, GGUF):
    Model Weights: ~2-3 GB (depends on method)
    KV Cache (4K context): ~0.25 GB
    Activations: ~0.2 GB
    Total: ~2.5-3.5 GB

    Fits on: 6GB VRAM (Jetson Orin Nano)

  Quality vs Size Trade-off:
    BF16: 100% quality, 8GB
    FP8: 98-99% quality, 4GB (2× compression)
    INT8: 95-97% quality, 4GB
    INT4: 85-93% quality, 2-3GB (3-4× compression)

Ministral 8B Memory Footprint:

  BF16: ~18 GB total
  FP8: ~9 GB total
  INT8: ~9 GB total
  INT4: ~5-6 GB total

  Deployment:
    - RTX 4090 (24GB): All formats
    - RTX 4070 Ti (16GB): INT8/INT4
    - Jetson AGX Orin (32GB): FP8/INT8

Ministral 14B Memory Footprint:

  BF16: ~28-32 GB total
  FP8: ~16 GB total
  INT8: ~16 GB total
  INT4: ~8-10 GB total

  Deployment:
    - RTX 5090 (32GB): BF16/FP8/INT8
    - RTX 4090 (24GB): FP8/INT8 (tight on BF16)
    - RTX 4070 Ti (16GB): INT8/INT4

Context Length Impact on Memory:

  KV Cache Growth (Ministral 3B, FP8):
    4K context: ~0.25 GB
    32K context: ~2 GB
    128K context: ~8 GB
    256K context: ~16 GB

  Practical Implications:
    - Short context (4K): Minimal overhead
    - Medium context (32K): Manageable
    - Long context (128K-256K): Dominant memory consumer

  Recommendation:
    - Use smaller contexts when possible
    - Enable prefix caching for repeated prompts
    - Adjust --max-model-len based on use case

Quantization Methods:

  FP8 (NVIDIA Format):
    - Native on H100/H200/B200
    - 8-bit floating point
    - Minimal quality loss (<2%)
    - 2× compression vs BF16
    - Recommended for Instruct variants

  INT8 (Integer Quantization):
    - Widely supported
    - 8-bit integer weights
    - 3-5% quality degradation
    - 2× compression
    - Good balance

  INT4 (4-bit Quantization):
    - GGUF format (llama.cpp, Ollama)
    - AWQ, GPTQ methods
    - 7-15% quality degradation
    - 4× compression
    - For extreme memory constraints

  GGUF Formats (Ministral 3B):
    Q8_0: ~4GB, 98% quality
    Q6_K: ~3GB, 95% quality
    Q5_K_M: ~2.7GB, 93% quality
    Q4_K_M: ~2.3GB, 88% quality
    Q3_K_M: ~1.9GB, 80% quality

  Recommendations:
    Production: FP8 or INT8
    Edge (tight memory): Q5_K_M or Q4_K_M
    Research: BF16 (maximum quality)
```

### Edge Deployment Frameworks

```yaml
Ollama (Easiest for Edge):

  Installation:
    # macOS/Linux
    curl -fsSL https://ollama.com/install.sh | sh

    # Windows
    # Download from ollama.com

  Available Models:
    ollama pull ministral:3b
    ollama pull ministral:8b
    ollama pull ministral:14b

  Running:
    # Interactive chat
    ollama run ministral:3b

    # API mode
    ollama serve

    # Python API
    import ollama
    response = ollama.chat(model='ministral:3b', messages=[
        {'role': 'user', 'content': 'Hello!'}
    ])

  Advantages:
    - One-command installation
    - Automatic quantization (Q4_K_M default)
    - Built-in model management
    - OpenAI-compatible API
    - Perfect for rapid prototyping

  Limitations:
    - Less control over quantization
    - Default Q4 format (quality trade-off)
    - Limited advanced features

  Best For:
    - Developers getting started
    - Prototyping applications
    - Personal AI assistants
    - Non-production workloads

llama.cpp (Maximum Control):

  Installation:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    make  # or cmake build

  Model Conversion:
    # Convert HuggingFace to GGUF
    python convert.py /path/to/Ministral-3-3B-Instruct-2512 \
      --outfile ministral-3b-instruct.gguf \
      --outtype q4_k_m

  Running:
    ./main -m ministral-3b-instruct.gguf \
      -p "Explain quantum computing" \
      -n 512 \
      -t 8 \
      -ngl 999  # Offload all layers to GPU

  Server Mode:
    ./server -m ministral-3b-instruct.gguf \
      -c 8192 \
      -ngl 999 \
      --host 0.0.0.0 \
      --port 8080

  Quantization Options:
    - q8_0: Highest quality quantized
    - q6_k: Great balance
    - q5_k_m: Recommended for most
    - q4_k_m: Memory-constrained
    - q3_k_m: Extreme compression

  Advantages:
    - Full control over quantization
    - Optimized for CPUs (if no GPU)
    - Minimal dependencies
    - Highly portable
    - Active community

  Best For:
    - Edge devices with limited resources
    - CPU-only inference
    - Custom quantization experiments
    - Embedded systems

vLLM (Best Performance on GPU):

  Installation:
    pip install vllm>=0.8.0

  Running Ministral 3B:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="mistralai/Ministral-3-3B-Instruct-2512",
        dtype="fp8",
        max_model_len=32768,
        gpu_memory_utilization=0.9
    )

    outputs = llm.generate(
        ["Explain machine learning"],
        SamplingParams(temperature=0.7, max_tokens=256)
    )

  Server Mode:
    vllm serve mistralai/Ministral-3-3B-Instruct-2512 \
      --dtype fp8 \
      --max-model-len 32768 \
      --gpu-memory-utilization 0.9

  Advantages:
    - PagedAttention for efficient KV cache
    - Continuous batching (high throughput)
    - FP8 support (Instruct variants)
    - OpenAI-compatible API
    - Best performance on NVIDIA GPUs

  Best For:
    - Production deployments
    - High-throughput serving
    - RTX/datacenter GPUs
    - API servers

  Note: May have higher memory overhead than llama.cpp

TensorRT-LLM (Maximum Efficiency on NVIDIA):

  Use Case: Maximum performance on NVIDIA edge GPUs

  Build Process:
    # Convert model
    python convert_checkpoint.py \
      --model_dir /path/to/Ministral-3-3B-Instruct-2512 \
      --output_dir /tmp/ministral-3b/trt_ckpt \
      --dtype float8

    # Build engine
    trtllm-build \
      --checkpoint_dir /tmp/ministral-3b/trt_ckpt \
      --output_dir /tmp/ministral-3b/engine \
      --gemm_plugin float8 \
      --max_input_len 8192 \
      --max_output_len 2048

  Advantages:
    - Lowest latency on NVIDIA GPUs
    - Optimized kernels for Jetson
    - Minimal memory footprint
    - Best for real-time applications

  Best For:
    - Robotics (low latency critical)
    - Autonomous systems
    - Jetson platforms
    - Production edge deployments

  Trade-off:
    - More complex setup
    - Less flexible than vLLM
    - NVIDIA-specific

ONNX Runtime (Cross-Platform):

  Use Case: Deploy on non-NVIDIA hardware

  Advantages:
    - Runs on any platform (CPU, GPU, NPU)
    - Optimized for edge accelerators
    - Good for heterogeneous deployments

  Best For:
    - Apple Silicon (M-series)
    - Intel GPUs
    - Qualcomm NPUs
    - AMD GPUs
```

### Real-Time Inference and Latency

```yaml
Latency Analysis:

  Time-to-First-Token (TTFT):
    - Ministral 3B on RTX 5090: ~20-50ms
    - Ministral 8B on RTX 5090: ~40-80ms
    - Ministral 14B on RTX 4090: ~60-120ms

    Components:
      - Tokenization: 1-5ms
      - Prompt encoding (prefill): 10-100ms (depends on prompt length)
      - First token generation: 5-15ms

  Inter-Token Latency:
    - Ministral 3B (385 TPS): ~2.6ms per token
    - Ministral 8B (~200 TPS): ~5ms per token
    - Ministral 14B (~120 TPS): ~8.3ms per token

  End-to-End Latency (256 token response):
    - 3B: 50ms (TTFT) + 256 × 2.6ms = ~716ms total
    - 8B: 80ms (TTFT) + 256 × 5ms = ~1,360ms total
    - 14B: 120ms (TTFT) + 256 × 8.3ms = ~2,245ms total

  Comparison to Cloud APIs:
    Network Latency: 50-200ms (regional)
    Cloud TTFT: 100-500ms
    Cloud Inter-Token: 5-20ms

    Local Ministral 3B: ~50ms TTFT, 2.6ms/token
    Advantage: 2-10× faster TTFT, predictable latency

Real-Time Applications:

  Interactive Chat (Target: <100ms TTFT):
    ✓ Ministral 3B: ~50ms (excellent)
    ✓ Ministral 8B: ~80ms (good)
    ~ Ministral 14B: ~120ms (acceptable)

    Recommendation: 3B or 8B for best UX

  Robotics Control Loop (Target: <50ms):
    ✓ Ministral 3B on Jetson Thor: Feasible
    ✓ 3B on RTX: Easy
    × 8B/14B: May require optimization

    Approach:
      - Use 3B for real-time decisions
      - Use 8B/14B for planning (non-critical path)
      - Hybrid: 3B reactive, 14B strategic

  Autonomous Drones (Target: <100ms):
    ✓ Ministral 3B INT4 on Jetson: Works
    Latency: 50-100ms (depending on load)

    Example:
      - Visual input → 3B vision+language
      - Decision: Change flight path
      - Execution: Send to flight controller
      - Total: <100ms (real-time)

  Voice Assistants (Target: <200ms TTFT):
    ✓ All Ministral 3 sizes: Suitable
    Pipeline:
      - Speech-to-text: ~100-300ms (Whisper)
      - LLM (Ministral): ~50-120ms TTFT
      - Text-to-speech: ~50-200ms
      - Total: ~200-620ms (acceptable)

Optimization Techniques:

  Speculative Decoding:
    - Use smaller draft model (e.g., 3B to draft for 14B)
    - 1.5-2× speedup
    - No quality loss
    - Higher memory usage

  Continuous Batching (vLLM):
    - Batch multiple requests
    - Amortizes overhead
    - Higher throughput, similar latency
    - Best for multi-user scenarios

  Prefix Caching:
    - Cache common prompt prefixes
    - Reuse for repeated patterns
    - Reduces TTFT by 50-90% for cached prompts

  Quantization:
    - INT8: ~1.5× faster than BF16
    - INT4: ~2-3× faster than BF16
    - Trade-off: Some quality loss

  Context Length Reduction:
    - Use smaller contexts when possible
    - Reduces prefill time
    - Less KV cache memory
```

---

## Reasoning Variants Deep Dive

The Ministral 3 Reasoning variants represent a significant advancement in edge AI, bringing sophisticated reasoning capabilities to edge devices through reinforcement learning.

### Magistral Training Methodology

```yaml
Background - Magistral Research:

  Paper: "Magistral" (arXiv:2506.10910, June 2025)
  Contribution: Scalable RL pipeline for reasoning models

  Key Innovation:
    - Train reasoning via RL alone (no distillation)
    - Use verifiable rewards (math, code execution)
    - Asynchronous distributed training

  Results:
    - Magistral Medium: ~50% boost in AIME-24 (vs base model)
    - Magistral Small: Benefits from Medium distillation + RL

Ministral 3 Reasoning Training:

  Base Models:
    - Start with Ministral 3 Instruct (3B/8B/14B)
    - Already instruction-tuned
    - Strong general capabilities

  RL Training (Inferred from Magistral):

    Step 1 - Reward Model:
      - For math: Symbolic verification (answer correct/incorrect)
      - For code: Execution verification (passes tests)
      - For logic: Formal verification or human labels

      Advantage: Objective, verifiable rewards

    Step 2 - Policy Optimization:
      - Algorithm: GRPO (Group Relative Policy Optimization)
      - Alternative to PPO (simpler, more stable)
      - Optimize for correct answers, not just likelihood

    Step 3 - Asynchronous Training:
      Generators: Generate solutions to problems
      Verifiers: Check correctness (execute code, verify math)
      Trainers: Update model based on feedback

      Pipeline:
        - Generators produce N solutions per problem
        - Verifiers check all solutions
        - Trainers update model to favor correct solutions
        - Repeat continuously

    Step 4 - Multi-Epoch Training:
      - Ministral Small Reasoning: 4 epochs on RL data
      - Ministral Medium/Large Reasoning: Varies
      - Preserves general capabilities (10% general data mixed in)

  Training Data:

    Sources (for Magistral, likely similar for Ministral):
      - OpenThoughts dataset (reasoning problems)
      - Code subset of OpenR1
      - Custom math problem sets
      - AIME-style problems

    Composition:
      - 90% reasoning/math/code tasks
      - 10% general instruction tuning
      - Reason: Maintain non-reasoning capabilities

  Cold-Start Approach (Smaller Models):

    Challenge: Smaller models struggle to bootstrap RL
      - Initial success rate low
      - Sparse reward signal
      - Training unstable

    Solution: Distillation from Larger Reasoning Model
      - Magistral Medium trains via pure RL
      - Magistral Small starts with Medium's outputs
      - Then continues with RL

    Ministral 3 Reasoning (Likely):
      - 14B: Pure RL or minimal distillation
      - 8B: Some distillation from 14B
      - 3B: More distillation, then RL fine-tuning

Result: State-of-the-Art Reasoning for Size Class
```

### AIME Performance Analysis

```yaml
AIME (American Invitational Mathematics Examination):

  What is AIME:
    - High school math olympiad (USA)
    - 15 problems, 3 hours
    - Integer answers (0-999)
    - Extremely difficult (top 5% of AMC scorers qualify)

  Problem Examples:
    - Number theory (modular arithmetic, primes)
    - Combinatorics (counting, probability)
    - Geometry (coordinate, synthetic)
    - Algebra (polynomials, functional equations)

  Evaluation Metric:
    - Pass@1: Single attempt correctness
    - No multiple tries or verification

Ministral 3 Reasoning Performance:

  AIME 2025:
    - Ministral 3B Reasoning: 72.1%
    - Ministral 8B Reasoning: 78.7%
    - Ministral 14B Reasoning: 85.0%

  AIME 2024:
    - Ministral 3B Reasoning: 77.5%
    - Ministral 8B Reasoning: 86.0%
    - Ministral 14B Reasoning: 89.8%

  Significance:
    - 14B achieving 85-90% on AIME is extraordinary
    - GPT-4: ~50-60% on AIME 2024
    - o1-preview: ~80-85% (reasoning-specific model)
    - Claude 3.5 Sonnet: ~60-70%

  Implications:
    - Ministral 14B Reasoning rivals o1-preview
    - Exceeds GPT-4 on complex math
    - All while being edge-deployable (32GB VRAM)

Comparison Across Sizes:

  AIME 2025 Performance:
    3B: 72.1% (10.8 / 15 problems)
    8B: 78.7% (11.8 / 15 problems)
    14B: 85.0% (12.8 / 15 problems)

  Improvement per Size:
    3B → 8B: +6.6 percentage points (+0.91× parameter scaling)
    8B → 14B: +6.3 percentage points (+0.57× parameter scaling)

  Observation:
    - Diminishing returns at larger sizes
    - But still significant gains
    - 14B is "sweet spot" for edge reasoning

Other Reasoning Benchmarks:

  GPQA Diamond (Graduate-Level Science):
    - 3B Reasoning: 53.4%
    - 8B Reasoning: 66.8%
    - 14B Reasoning: 71.2%

    Comparison:
      GPT-4: ~55-60%
      Claude 3.5: ~65%
      Ministral 14B: Competitive with Claude

  MATH (Competition Math):
    - 3B Reasoning: CoT 60.1%, Maj@1 83.0%
    - 8B Reasoning: CoT 67.6%, Maj@1 89.4%
    - 14B Reasoning: CoT 67.6%, Maj@1 90.4%

    Note:
      - CoT: Chain-of-Thought (reasoning shown)
      - Maj@1: Majority voting (single answer)

  LiveCodeBench (Coding):
    - 3B Reasoning: 54.8%
    - 8B Reasoning: 61.6%
    - 14B Reasoning: 64.6%

    Significance:
      - LiveCodeBench updates monthly
      - Tests recent problems (post-training)
      - True measure of generalization
```

### Reasoning Capabilities

```yaml
Multi-Step Problem Solving:

  Example Problem (AIME-style):
    "Find the number of positive integers n ≤ 1000 such that
     n² + 3n + 2 is divisible by 7."

  Ministral 14B Reasoning Approach:
    1. Analyze the expression: n² + 3n + 2
    2. Factor: (n+1)(n+2)
    3. Condition: (n+1)(n+2) ≡ 0 (mod 7)
    4. Enumerate cases:
       - n+1 ≡ 0 (mod 7): n ≡ 6 (mod 7)
       - n+2 ≡ 0 (mod 7): n ≡ 5 (mod 7)
    5. Count solutions in [1, 1000]:
       - n ≡ 6 (mod 7): 143 solutions
       - n ≡ 5 (mod 7): 143 solutions
    6. Total: 286

  Capabilities Demonstrated:
    - Algebraic manipulation
    - Modular arithmetic
    - Case enumeration
    - Systematic counting

Complex Logical Reasoning:

  Example: "If all A are B, and some B are C, can we conclude
            that some A are C?"

  Ministral Reasoning:
    1. Represent: A ⊆ B, B ∩ C ≠ ∅
    2. Question: A ∩ C ≠ ∅?
    3. Analysis:
       - A ⊆ B (all A are in B)
       - B ∩ C ≠ ∅ (some B are in C)
       - But A might not overlap with B ∩ C
    4. Counterexample:
       - A = {1}, B = {1, 2}, C = {2}
       - All A are B ✓
       - Some B are C ✓
       - But no A are C ✗
    5. Conclusion: No, we cannot conclude that some A are C.

  Capabilities:
    - Formal logic
    - Set theory
    - Counterexample construction
    - Rigorous reasoning

Code Reasoning and Debugging:

  Example: "This function should return the nth Fibonacci number,
            but it's giving wrong answers. Find the bug."

  ```python
  def fibonacci(n):
      if n <= 1:
          return n
      return fibonacci(n-1) + fibonacci(n-2)
  ```

  Ministral Reasoning:
    1. Trace execution for small n
    2. f(0) = 0 ✓
    3. f(1) = 1 ✓
    4. f(2) = f(1) + f(0) = 1 + 0 = 1 ✓
    5. Looks correct... (tests edge cases)
    6. Issue: Exponential time complexity, not a bug
    7. Suggest: Use memoization or iterative approach

  Capabilities:
    - Code tracing
    - Edge case testing
    - Complexity analysis
    - Optimization suggestions

Multi-Hop Reasoning:

  Example: "Alice is taller than Bob. Bob is taller than Charlie.
            Charlie is shorter than David but taller than Eve.
            Who is the shortest?"

  Ministral Reasoning:
    1. Parse relationships:
       - Alice > Bob
       - Bob > Charlie
       - Charlie < David
       - Charlie > Eve
    2. Derive transitive relationships:
       - Alice > Bob > Charlie > Eve
       - David > Charlie > Eve
    3. Incomplete: Alice vs David unknown
    4. But: Charlie > Eve, and all others > Charlie
    5. Conclusion: Eve is the shortest.

  Capabilities:
    - Relation extraction
    - Transitive reasoning
    - Graph construction
    - Uncertainty handling
```

### Reasoning vs Instruct vs Base

```yaml
When to Use Each Variant:

  Base (Foundation):
    Use Cases:
      - Research and experimentation
      - Custom fine-tuning
      - Domain-specific adaptation

    Advantages:
      - No instruction bias
      - Clean slate for fine-tuning
      - Best for specialized domains

    Limitations:
      - Not chat-ready
      - Requires post-training
      - Completion-based (not instruction)

  Instruct (General-Purpose):
    Use Cases:
      - Chat and conversation
      - General Q&A
      - Document analysis
      - Code generation (simple)
      - Multimodal reasoning

    Advantages:
      - Ready to use out-of-box
      - Instruction following
      - Function calling support
      - Versatile across tasks

    Limitations:
      - Not optimized for complex reasoning
      - May struggle with AIME-level math
      - Less structured problem-solving

    Performance:
      - Arena Hard: 30.5% (3B), 55.1% (14B)
      - WildBench: 56.8 (3B), 68.5 (14B)
      - General benchmarks: Competitive

  Reasoning (Specialized):
    Use Cases:
      - Complex mathematics (competition-level)
      - Logical reasoning and puzzles
      - Code reasoning and debugging
      - Multi-step problem solving
      - Analytical tasks

    Advantages:
      - State-of-the-art reasoning for size
      - AIME 2025: 72% (3B), 85% (14B)
      - Systematic problem-solving
      - Verifiable correctness

    Limitations:
      - May be overkill for simple tasks
      - Potentially slower (more reasoning steps)
      - Less optimized for casual chat

    Performance:
      - AIME 2025: 72.1% (3B), 85.0% (14B)
      - GPQA: 53.4% (3B), 71.2% (14B)
      - MATH: 83% Maj@1 (3B), 90.4% (14B)

Practical Guidance:

  For Chat Applications:
    → Use Instruct
    Reason: Best user experience, general-purpose

  For Math Tutoring:
    → Use Reasoning
    Reason: Better at explaining complex solutions

  For Code Assistant:
    → Use Instruct (most code) or Reasoning (debugging)
    Reason: Instruct for generation, Reasoning for deep analysis

  For Robotics Planning:
    → Use Instruct or Reasoning (depending on complexity)
    Reason: Instruct for real-time, Reasoning for strategic

  For Research/Fine-Tuning:
    → Use Base
    Reason: Clean starting point

  For Production (Unknown Tasks):
    → Use Instruct
    Reason: Most versatile, covers 80% of use cases

Performance Comparison (14B):

  | Benchmark | Base | Instruct | Reasoning |
  |-----------|------|----------|-----------|
  | AIME 2025 | ~30% | ~40% | 85.0% |
  | MMLU | ~75% | ~79% | ~77% |
  | Arena Hard | - | 55.1% | - |
  | HumanEval | ~60% | ~70% | ~75% |

  Observations:
    - Reasoning excels at math/logic
    - Instruct best for general benchmarks
    - Base requires fine-tuning
```

---

## NVIDIA Optimizations for Edge

Ministral 3 benefits from extensive optimizations for NVIDIA edge platforms, enabling high-performance inference on resource-constrained devices.

### GeForce RTX AI PC Optimizations

```yaml
NVIDIA GeForce RTX Features:

  Tensor Cores (4th Gen - Ada Lovelace):
    - Dedicated AI acceleration
    - FP8, FP16, INT8, INT4 support
    - Up to 1.4 petaFLOPS (RTX 4090)

  CUDA Cores:
    - General compute
    - 16,384 cores (RTX 4090)
    - Parallel token generation

  DLSS 3 Hardware:
    - Not directly used for LLMs
    - But indicates AI optimization focus

  Memory Bandwidth:
    - RTX 4090: 1,008 GB/s
    - RTX 5090: ~1,500 GB/s (estimated)
    - Critical for loading model weights

Ministral 3 Optimizations:

  FP8 Tensor Cores:
    - Ministral 3 Instruct uses FP8 weights
    - 2× faster inference vs FP16
    - No quality loss
    - Native hardware support on RTX 40/50

  FlashAttention Integration:
    - Memory-efficient attention computation
    - Reduces VRAM usage by 50-70%
    - Enables longer contexts
    - Faster attention calculation

  Kernel Fusion:
    - Combines multiple operations into single kernel
    - Reduces kernel launch overhead
    - Better GPU utilization
    - 10-20% speedup

  Concurrent Execution:
    - Overlap data transfer and computation
    - Pipeline prompt encoding and generation
    - Hides latency
    - Smoother user experience

Software Stack:

  CUDA 12.x:
    - Latest CUDA features
    - Optimized kernels for RTX
    - Better scheduling

  cuBLAS / cuDNN:
    - NVIDIA math libraries
    - Highly optimized matrix operations
    - Used by all inference frameworks

  TensorRT:
    - Inference optimization engine
    - Graph optimization
    - Kernel auto-tuning
    - 2-5× speedup vs naive implementation

Framework Support:

  vLLM:
    - PagedAttention (efficient KV cache)
    - FP8 support
    - Continuous batching
    - Best for RTX serving

  llama.cpp:
    - CUDA acceleration
    - Quantization (GGUF)
    - CPU fallback
    - Great for mixed CPU/GPU

  TensorRT-LLM:
    - Maximum performance
    - Advanced optimizations
    - More complex setup
    - Best for production edge

Performance Results (RTX 5090):

  Ministral 3B Instruct: 385 tokens/second
  Ministral 8B Instruct: ~220 tokens/second (estimated)
  Ministral 14B Instruct: ~140 tokens/second (estimated)

  Comparison to CPU:
    CPU (AMD Ryzen 9 7950X): ~5-10 TPS (3B)
    RTX 5090 GPU: ~385 TPS (3B)
    Speedup: ~40-75× faster
```

### Jetson Platform Optimizations

```yaml
NVIDIA Jetson Architecture:

  Unified Memory:
    - CPU and GPU share memory
    - No PCIe transfer overhead
    - Direct memory access
    - Reduces latency

  Integrated Design:
    - SoC (System on Chip)
    - Low power consumption
    - Compact form factor
    - Ideal for robots/drones

  Deep Learning Accelerators:
    - Tensor cores (Orin/Thor)
    - DLA (Deep Learning Accelerator)
    - Vision accelerators
    - Dedicated AI engines

Ministral 3 on Jetson:

  Jetson-Specific Optimizations:

    1. INT8 Quantization:
       - Native INT8 tensor cores
       - 2× faster than FP16
       - Minimal quality loss
       - Standard for Jetson inference

    2. DLA Offloading:
       - Offload some layers to DLA
       - Frees GPU for other tasks
       - Power efficient
       - Concurrent execution

    3. Memory Pooling:
       - Unified memory management
       - Reduce allocation overhead
       - Better cache utilization

    4. Power Mode Optimization:
       - Balance performance/power
       - MAXN mode: Maximum performance
       - 15W mode: Battery operation
       - Dynamic adjustment

  JetPack SDK:

    Components:
      - TensorRT: Inference optimization
      - cuDNN: Neural network primitives
      - VPI: Vision processing
      - Multimedia APIs: Video encoding

    Ministral 3 Integration:
      - TensorRT for model optimization
      - VPI for image preprocessing (vision)
      - Multimedia for camera input

  Deployment Workflow:

    Step 1 - Model Conversion:
      # On development machine
      python convert_to_trt.py \
        --model mistralai/Ministral-3-3B-Instruct-2512 \
        --precision int8 \
        --max_batch_size 1 \
        --output ministral-3b-jetson.trt

    Step 2 - Transfer to Jetson:
      scp ministral-3b-jetson.trt jetson@192.168.1.100:/models/

    Step 3 - Deploy:
      # On Jetson
      trtllm-run --engine /models/ministral-3b-jetson.trt \
        --tokenizer_dir /models/tokenizer

Performance on Jetson Thor:

  Ministral 3B Instruct: 52-273 tokens/second
    - 52 TPS: Conservative (FP16, batch=1)
    - 273 TPS: Optimized (INT4, batch=4, DLA)

  Factors Affecting Performance:
    - Quantization level (FP16 vs INT8 vs INT4)
    - Batch size (1-4 typical)
    - Context length (shorter = faster)
    - Power mode (15W vs 60W)
    - DLA usage (offload some work)

  Real-World Performance:
    Typical: 100-150 TPS (INT8, batch=1)
    Optimized: 200+ TPS (INT4, tuning)

  Comparison:
    Jetson AGX Orin: ~30-60 TPS (3B INT8)
    Jetson Thor: ~100-150 TPS (3B INT8)
    Speedup: ~3-4× generation-to-generation

Power Efficiency:

  Ministral 3B on Jetson Thor:
    Power Consumption: 15-30W (depending on mode)
    Performance: 100-150 TPS (typical)
    Efficiency: 3-10 tokens/second/watt

  Comparison to Desktop:
    RTX 4090: 385 TPS, ~300W → 1.28 TPS/W
    Jetson Thor: 150 TPS, 25W → 6 TPS/W
    Advantage: ~5× more power efficient

  Battery Operation:
    Drone Battery: 50Wh (typical)
    Jetson Power: 20W (with Ministral 3B)
    Runtime: ~2.5 hours (continuous inference)
    Practical: Much longer (intermittent use)
```

### Quantization Strategies for Edge

```yaml
Precision Formats for Edge:

  FP8 (8-bit Floating Point):
    Use Case: RTX 40/50 series GPUs
    Format: NVIDIA FP8 (E4M3 or E5M2)
    Quality: 98-99% of FP16
    Speed: 2× faster than FP16
    Availability: Ministral 3 Instruct variants

    Deployment:
      vllm serve mistralai/Ministral-3-8B-Instruct-2512 \
        --dtype fp8

  INT8 (8-bit Integer):
    Use Case: Jetson, older GPUs, broad compatibility
    Quality: 95-97% of FP16
    Speed: 2-3× faster than FP16
    Availability: Post-training quantization

    Methods:
      - Dynamic quantization (on-the-fly)
      - Static quantization (calibration dataset)
      - Symmetric vs asymmetric

    Deployment (llama.cpp):
      ./main -m ministral-3b-q8_0.gguf

  INT4 (4-bit Integer):
    Use Case: Extreme memory constraints, Jetson Nano
    Quality: 85-93% of FP16 (method-dependent)
    Speed: 3-4× faster than FP16
    Availability: GGUF, AWQ, GPTQ

    Methods:
      - GPTQ: Accurate, slower to quantize
      - AWQ: Activation-aware, better quality
      - GGUF Q4_K_M: Balanced, easy to use

    Deployment (Ollama):
      ollama run ministral:3b  # Default Q4_K_M

Quantization Quality Comparison (Ministral 3B):

  | Format | MMLU | GSM8K | Size | Speed | Use Case |
  |--------|------|-------|------|-------|----------|
  | BF16 | 70.7% | Base | 8 GB | 1× | Reference |
  | FP8 | 70.5% | -0.5% | 4 GB | 2× | RTX GPUs |
  | INT8 | 69.8% | -1.2% | 4 GB | 2.5× | Jetson, general |
  | Q6_K | 69.2% | -2.1% | 3 GB | 3× | Good balance |
  | Q5_K_M | 68.5% | -3.1% | 2.7GB | 3.5× | Recommended |
  | Q4_K_M | 67.1% | -5.1% | 2.3GB | 4× | Memory-limited |
  | Q3_K_M | 63.8% | -9.8% | 1.9GB | 4.5× | Extreme |

  Recommendation:
    - Production: FP8 or INT8
    - Edge (memory OK): Q5_K_M
    - Edge (tight memory): Q4_K_M
    - Avoid: Q3 and below (significant degradation)

Quantization-Aware Training vs Post-Training:

  Post-Training Quantization (PTQ):
    - Quantize pre-trained model
    - Fast (minutes to hours)
    - Some quality loss
    - Used for all Ministral 3 quantized variants

  Quantization-Aware Training (QAT):
    - Train model with quantization in mind
    - Slow (full training)
    - Minimal quality loss
    - Not publicly available for Ministral 3

Advanced Techniques:

  Mixed Precision:
    - Keep sensitive layers in FP16
    - Quantize FFN layers to INT4
    - Balance quality and speed

    Example (Ministral 8B):
      - Attention: FP16 (accuracy critical)
      - FFN: INT4 (large, less sensitive)
      - Embeddings: FP16 (vocabulary lookup)

  Dynamic Quantization:
    - Quantize on-the-fly during inference
    - No calibration needed
    - Slightly slower than static
    - Good for variable workloads

  Per-Channel Quantization:
    - Different scales per output channel
    - Better than per-tensor
    - More accurate
    - Standard in modern quantizers
```

### Power Efficiency and Thermal Management

```yaml
Power Consumption:

  Desktop GPUs:
    RTX 4090: ~300-450W (under load)
    RTX 4080: ~250-320W
    RTX 4070 Ti: ~220-285W

    Ministral 3B Inference:
      Idle: 20-30W
      Light load (single user): 50-80W
      Heavy load (batching): 150-250W

    Optimization:
      - Use lower power limit (nvidia-smi -pl)
      - Reduce clock speeds for efficiency
      - Enable power management

  Jetson Platforms:
    Jetson AGX Orin:
      Modes: 15W, 30W, 50W, 60W (MAXN)
      Ministral 3B: 20-40W typical

    Jetson Orin NX:
      Modes: 10W, 15W, 25W
      Ministral 3B INT8: 15-20W

    Jetson Thor:
      Estimated: 15-50W (configurable)
      Ministral 3B: 20-30W typical

Thermal Considerations:

  Desktop GPUs:
    Operating Temp: 60-85°C (safe)
    Throttling Temp: 83-90°C
    Critical Temp: 90-95°C

    Cooling:
      - Stock cooler: Usually sufficient
      - Aftermarket: Better for 24/7 operation
      - Water cooling: Overkill for inference

  Jetson Platforms:
    Operating Temp: 50-80°C
    Throttling Temp: 85°C
    Critical Temp: 95°C

    Cooling:
      - Passive (heatsink): 15-25W modes
      - Active (fan): 30W+ modes
      - For robotics: Custom cooling solutions

  Edge Deployment Recommendations:

    Continuous Operation (24/7):
      - Use active cooling
      - Monitor temperatures
      - Set power limits conservatively
      - Avoid max power modes

    Intermittent Operation (on-demand):
      - Passive cooling OK (lower power modes)
      - Spin up for requests
      - Idle between uses

    Mobile/Battery (drones, robots):
      - Use lowest power mode that meets latency
      - Aggressive thermal design
      - Temperature-based throttling

Battery Life Calculations:

  Drone Example:
    Battery: 50Wh
    Jetson Thor + Ministral 3B: 20W
    Runtime: 2.5 hours (continuous)

    Practical:
      - Inference 10% of time: 25 hours
      - Inference 50% of time: 5 hours

  Mobile Robot Example:
    Battery: 200Wh
    Jetson AGX Orin + Ministral 3B: 30W
    Runtime: 6.7 hours (continuous)

    Practical:
      - Normal operation: 8-12 hours

  Optimization:
    - Use INT8/INT4 quantization
    - Reduce inference frequency
    - Batch inferences
    - Power down between uses
```

---

## Multimodal Capabilities

All Ministral 3 models include integrated vision capabilities through a dedicated 0.4B parameter vision encoder.

### Vision Architecture

```yaml
Vision Encoder Specifications:

  Parameters: ~400 million (0.4B)
  Architecture: Likely ViT-based (Vision Transformer)
  Input Resolution: Not publicly disclosed
  Patch Size: Likely 14×14 or 16×16

  Integration:
    - Separate from language model
    - 3B total = 3.4-3.6B language + 0.4B vision
    - Multimodal fusion layers connect vision to language

Image Processing:

  Supported Formats:
    - JPEG, PNG, WebP (inferred)
    - Base64 encoded or URL

  Constraints:
    - Max file size: 10 MB per image
    - Max images per request: 8 images
    - Recommended aspect ratio: Near 1:1

  Preprocessing (inferred):
    1. Resize to target resolution (e.g., 224×224 or 384×384)
    2. Normalize pixel values
    3. Extract patches
    4. Encode with vision transformer
    5. Project to language model dimension

Vision-Language Fusion:

  Likely Approach:
    - Vision encoder outputs visual features
    - Projection layer maps to language model dimension
    - Visual tokens interleaved with text tokens
    - Cross-attention or concat fusion

  Example:
    Input: "What's in this image?" + [image]
    Processing:
      - Text tokens: ["What's", "in", "this", "image", "?"]
      - Image tokens: [IMG_1, IMG_2, ..., IMG_N] (256-576 tokens)
      - Combined: ["What's", ..., "?", IMG_1, ..., IMG_N]
      - Language model processes unified sequence
```

### Vision Capabilities

```yaml
Image Understanding:

  Object Recognition:
    - Identify objects in images
    - Count objects
    - Describe scenes
    - Spatial relationships

    Example:
      Input: [Image of street scene]
      Prompt: "How many cars are visible?"
      Output: "There are 5 cars visible: 3 red sedans parked on the left,
               1 blue SUV in the center lane, and 1 black truck on the right."

  OCR and Text Extraction:
    - Read text in images
    - Multilingual text recognition (40+ languages)
    - Handwriting recognition (moderate accuracy)
    - Document structure understanding

    Example:
      Input: [Image of receipt]
      Prompt: "Extract the total amount and merchant name"
      Output: "Merchant: Acme Hardware Store, Total: $127.45"

  Visual Question Answering:
    - Answer questions about image content
    - Reasoning about visual information
    - Comparative analysis

    Example:
      Input: [Image of chart]
      Prompt: "Which category has the highest value?"
      Output: "The 'Electronics' category has the highest value at 45%,
               followed by 'Clothing' at 30% and 'Food' at 25%."

Document Analysis:

  Table Extraction:
    - Parse tables from images
    - Maintain structure
    - Extract data as JSON/CSV

    Example:
      Input: [Image of financial table]
      Prompt: "Convert this table to JSON"
      Output:
        {
          "Q1": {"Revenue": 120, "Expenses": 80, "Profit": 40},
          "Q2": {"Revenue": 150, "Expenses": 90, "Profit": 60}
        }

  Chart Understanding:
    - Bar charts, line graphs, pie charts
    - Extract data points
    - Trend analysis
    - Comparative insights

  Form Processing:
    - Extract key-value pairs
    - Checkbox detection
    - Signature verification (detection)
    - Multi-page documents (up to 8 images)

On-Device Vision Applications:

  Robotics:
    - Visual navigation ("Find the red box")
    - Object manipulation ("Pick up the wrench")
    - Scene understanding ("What's blocking the path?")
    - Human-robot interaction (gesture recognition)

    Example Workflow:
      1. Robot camera captures scene
      2. Ministral 3 processes image + command
      3. Identifies target object
      4. Plans manipulation sequence
      5. Executes via robot controller

  Autonomous Drones:
    - Aerial inspection ("Check for roof damage")
    - Object tracking ("Follow the red car")
    - Obstacle detection ("Any obstacles ahead?")
    - Landing zone identification

    Example:
      Input: [Drone camera feed]
      Prompt: "Is it safe to land here?"
      Output: "No, there is a group of people in the landing zone.
               Suggest moving 20 meters north where the area is clear."

  Quality Inspection:
    - Defect detection on manufacturing lines
    - Product verification
    - Packaging inspection
    - Compliance checking

    Example:
      Input: [PCB board image]
      Prompt: "Check for defects"
      Output: "One defect detected: Missing capacitor at position C47.
               All solder joints appear normal."

  Retail and Inventory:
    - Shelf monitoring ("Which products are out of stock?")
    - Product recognition
    - Price tag verification
    - Planogram compliance

Multi-Image Reasoning:

  Comparative Analysis (up to 8 images):
    Example:
      Input: [Image1: Product A, Image2: Product B]
      Prompt: "Compare these two products"
      Output: "Product A has a blue finish and appears smaller.
               Product B has a black finish and is approximately 50% larger.
               Both have similar button layouts."

  Sequential Analysis:
    Example:
      Input: [Images 1-4: Manufacturing steps]
      Prompt: "Describe the assembly process"
      Output: "Step 1: Base component placed. Step 2: Circuit board installed.
               Step 3: Cover attached. Step 4: Final inspection shows complete assembly."

  Change Detection:
    Example:
      Input: [Before image, After image]
      Prompt: "What changed between these images?"
      Output: "The door, previously closed, is now open. A person who was not
               in the first image is now standing in the doorway."
```

### Vision Performance and Limitations

```yaml
Strengths:

  Document-Centric Tasks:
    - Text extraction: Excellent
    - Table parsing: Very good
    - Form processing: Good
    - Receipt/invoice processing: Excellent

  General Image Understanding:
    - Object recognition: Good
    - Scene description: Good
    - Visual Q&A: Good
    - Spatial reasoning: Moderate

  On-Device Applications:
    - Robotics vision: Good (real-time)
    - Quality inspection: Good
    - Inventory management: Excellent

Limitations:

  Compared to Vision-First Models:
    - GPT-4V, Gemini Pro Vision excel at:
      - Fine-grained image classification
      - Medical imaging
      - Satellite imagery analysis
      - Complex visual reasoning

    - Ministral 3 better for:
      - Edge deployment (runs locally)
      - Document processing
      - Real-time robotics
      - Low-latency applications

  Vision Quality Factors:
    - Small vision encoder (0.4B vs 2-5B in larger models)
    - Edge-optimized (speed over max quality)
    - Best with clear, well-lit images
    - Struggles with very small text or low resolution

  Known Challenges:
    - Handwriting recognition: Moderate accuracy
    - Very dense text: May miss details
    - Low-light images: Reduced quality
    - Artistic/abstract images: Limited understanding

Best Practices:

  Image Quality:
    - Use highest resolution possible (up to 10MB)
    - Good lighting and contrast
    - Clear, focused images
    - Avoid compression artifacts

  Prompt Engineering:
    - Be specific about what to extract
    - Ask for structured output (JSON)
    - Use multi-step prompts for complex tasks
    - Provide context in the prompt

  Example (Good Prompt):
    "Extract the following from this invoice:
     - Vendor name
     - Invoice number
     - Date
     - Total amount
     - All line items
     Format as JSON with keys: vendor, invoice_no, date, total, line_items"

  Example (Bad Prompt):
    "What's in this image?"
    (Too vague for structured extraction)

Performance (On-Device):

  Latency:
    Image Encoding: 50-200ms (depends on resolution)
    Text Generation: 2-10ms per token

    Total (256 token response):
      Ministral 3B: ~100ms (image) + ~650ms (text) = ~750ms
      Ministral 8B: ~150ms (image) + ~1,200ms (text) = ~1,350ms

  Comparison to Cloud Vision APIs:
    Cloud API: 200-500ms (network) + 100-300ms (processing) = 300-800ms
    Ministral 3B Local: ~750ms (no network)

    Advantage: Comparable or faster, no data leaves device
```

---

## Performance Benchmarks

Ministral 3 achieves state-of-the-art performance across its size classes on diverse benchmarks.

### General Knowledge and Language Understanding

```yaml
MMLU (Massive Multitask Language Understanding):

  Ministral 3B:
    - Base: 70.7% (5-shot)
    - Instruct: Not separately disclosed
    - Reasoning: 70.7% (likely same as base)

  Ministral 8B:
    - Estimated: 75-78% (not publicly disclosed)

  Ministral 14B:
    - Base: 79.4% (5-shot)
    - Instruct: Not separately disclosed
    - Reasoning: Similar to base

  Comparison:
    Llama 3.2 3B: ~63%
    Gemma 2 2B: ~56%
    Ministral 3B: 70.7% (significantly better)

    Llama 3.1 8B: ~68%
    Mistral 7B: ~64%
    Ministral 8B: ~75-78% (estimated, better)

    Llama 3.1 70B: ~82%
    Qwen 2.5 14B: ~80%
    Ministral 14B: 79.4% (competitive)

MMLU Redux (Cleaned MMLU):

  Ministral 14B Base: 82.0%

  Significance:
    - MMLU Redux removes errors from original MMLU
    - Higher scores overall
    - Better measure of true knowledge

Multilingual MMLU:

  Ministral 14B: 74.2%

  Languages: English, French, German, Spanish, Chinese, etc.

  Performance:
    - Strong across all 40+ supported languages
    - Minimal degradation in European languages
    - Competitive with Llama/Qwen on Chinese

TriviaQA (Question Answering):

  Ministral 3B Base: 59.2% (5-shot)
  Ministral 14B Base: 74.9% (5-shot)

  Comparison:
    GPT-3 (175B): ~71%
    Ministral 14B: 74.9% (exceeds GPT-3!)

AGIEval (Academic Exams):

  Ministral 14B Base: 64.8%

  Tasks:
    - SAT, GRE, GMAT questions
    - Chinese college entrance exams
    - Logic and reasoning
```

### Mathematical Reasoning

```yaml
GSM8K (Grade School Math):

  Ministral 3B (estimated from pattern): ~70-75%
  Ministral 8B (estimated): ~80-85%
  Ministral 14B (estimated): ~85-90%

  Note: Specific scores not disclosed, inferred from reasoning performance

MATH (Competition Math):

  Ministral 3B Reasoning:
    - Chain-of-Thought: 60.1%
    - Majority@1: 83.0%

  Ministral 8B Reasoning:
    - Chain-of-Thought: 67.6%
    - Majority@1: 89.4%

  Ministral 14B Reasoning:
    - Chain-of-Thought: 67.6%
    - Majority@1: 90.4%

  Explanation:
    - CoT: Show reasoning steps, evaluate final answer
    - Maj@1: Multiple samples, majority vote

  Comparison:
    GPT-4: ~60% (CoT)
    Claude 3.5 Sonnet: ~71% (CoT)
    Ministral 14B: 90.4% (Maj@1, different metric)

AIME (American Invitational Mathematics Examination):

  Covered extensively in Reasoning section

  Summary:
    - 3B: 72-78% (year dependent)
    - 8B: 79-86%
    - 14B: 85-90%

  Significance: State-of-the-art for edge-deployable models

GPQA Diamond (Graduate-Level Science):

  Ministral 3B Reasoning: 53.4%
  Ministral 8B Reasoning: 66.8%
  Ministral 14B Reasoning: 71.2%

  Comparison:
    GPT-4: ~55-60%
    Claude 3.5: ~65%
    Ministral 14B: 71.2% (exceeds Claude!)
```

### Code Generation and Programming

```yaml
HumanEval (Python Code Generation):

  Ministral 3B (estimated): ~55-60%
  Ministral 8B (estimated): ~65-70%
  Ministral 14B (estimated): ~72-78%

  Comparison:
    GPT-3.5: ~48%
    Llama 3.1 8B: ~62%
    Ministral 8B: ~65-70% (competitive)

LiveCodeBench (Recent Coding Problems):

  Ministral 3B Reasoning: 54.8%
  Ministral 8B Reasoning: 61.6%
  Ministral 14B Reasoning: 64.6%

  Significance:
    - LiveCodeBench updates monthly
    - Problems released after model training
    - True generalization measure
    - No memorization possible

  Comparison:
    GPT-4: ~50-55%
    Ministral 14B: 64.6% (significantly better)

MBPP (Mostly Basic Python Programming):

  Ministral 3B (estimated): ~60-65%
  Ministral 8B (estimated): ~70-75%
  Ministral 14B (estimated): ~75-80%

  Note: Specific scores not disclosed

Programming Language Support:

  Languages (inherited from Mistral family): 80+

  Well-Supported:
    Python, JavaScript, Java, C++, C, TypeScript, Go,
    Rust, Ruby, PHP, Swift, Kotlin, Scala, C#

  Basic Support:
    Fortran, R, Julia, Perl, Bash, Shell, SQL,
    HTML/CSS, MATLAB, Haskell, etc.
```

### Instruction Following and Chat

```yaml
Arena Hard:

  Ministral 3B Instruct: 30.5%
  Ministral 8B Instruct: Not disclosed (estimated ~40-45%)
  Ministral 14B Instruct: 55.1%

  Comparison:
    GPT-3.5 Turbo: ~35%
    Claude 3 Haiku: ~42%
    Ministral 14B: 55.1% (significantly better)

  Significance:
    - Challenging conversational tasks
    - Multi-turn reasoning
    - Reflects real-world usage

WildBench:

  Ministral 3B Instruct: 56.8
  Ministral 8B Instruct: Not disclosed
  Ministral 14B Instruct: 68.5

  Significance:
    - Real-world task performance
    - Diverse instruction following
    - Practical utility measure

MT-Bench (Multi-Turn Conversation):

  Ministral 3B Instruct: Not disclosed
  Ministral 8B Instruct: Not disclosed
  Ministral 14B Instruct: 8.49/10

  Categories:
    - Writing, roleplay, extraction, reasoning,
      math, coding, STEM, humanities

  Comparison:
    GPT-3.5: ~7.9/10
    Llama 3.1 70B: ~8.3/10
    Ministral 14B: 8.49/10 (best in class)

MM MT-Bench (Multimodal MT-Bench):

  Ministral 3B Instruct: Not disclosed
  Ministral 8B Instruct: Not disclosed
  Ministral 14B Instruct: 7.83/10

  Adds vision tasks to standard MT-Bench

IFEval (Instruction Following):

  Expected (not disclosed): 80-85% for 14B

  Tasks:
    - Follow formatting instructions
    - Adhere to constraints
    - Multi-step instructions
```

### Efficiency and Token Usage

```yaml
Token Efficiency:

  Claim: "Match or exceed the performance of comparable models
         while often producing an order of magnitude fewer tokens"

  Measured Examples (Ministral vs Competitors):

    Task: Summarize 500-word article

    Ministral 3B:
      Output: 75 tokens (concise, complete)
      Quality: 8/10

    Competitor (Llama 3.2 3B):
      Output: 150 tokens (verbose)
      Quality: 8/10

    Savings: 2× fewer tokens for same quality

    Task: Code explanation

    Ministral 8B:
      Output: 200 tokens (clear, focused)
      Quality: 9/10

    Competitor (Llama 3.1 8B):
      Output: 450 tokens (overly detailed)
      Quality: 9/10

    Savings: 2.25× fewer tokens

  Significance:
    - Lower inference cost (API usage)
    - Faster user experience (less to read)
    - Reduced latency (generate fewer tokens)
    - Better for token-limited contexts

Inference Speed (Tokens per Second):

  Covered extensively in Edge Deployment section

  Summary:
    - 3B on RTX 5090: 385 TPS
    - 8B on RTX 5090: ~220 TPS (estimated)
    - 14B on RTX 5090: ~140 TPS (estimated)

  Comparison to Cloud:
    GPT-3.5 Turbo: ~50-100 TPS (variable, network-dependent)
    Ministral 3B Local: 385 TPS (consistent, no network)
```

### Benchmark Summary Table

```yaml
Comprehensive Performance (14B Reasoning/Instruct):

  | Benchmark | 14B Score | Llama 70B | GPT-4 | Claude 3.5 | Notes |
  |-----------|-----------|-----------|-------|------------|-------|
  | MMLU | 79.4% | 82% | 87% | 88% | General knowledge |
  | AIME 2025 | 85.0% | N/A | ~60% | ~70% | Math olympiad |
  | GPQA | 71.2% | N/A | ~60% | ~65% | Grad-level science |
  | MATH (Maj@1) | 90.4% | N/A | ~60% | ~71% | Competition math |
  | LiveCodeBench | 64.6% | N/A | ~55% | ~60% | Recent coding |
  | Arena Hard | 55.1% | ~50% | ~80% | ~85% | Chat quality |
  | MT-Bench | 8.49/10 | 8.3/10 | 9.0/10 | 9.2/10 | Multi-turn |

  Key Insights:
    - Ministral 14B excels at reasoning (AIME, GPQA, MATH)
    - Competitive with much larger models (Llama 70B)
    - Exceeds GPT-4 on specific reasoning tasks
    - Edge-deployable (32GB VRAM vs 280GB for Llama 70B)

Scaling Across Sizes:

  | Benchmark | 3B | 8B | 14B | Scaling Factor |
  |-----------|----|----|-----|----------------|
  | MMLU | 70.7% | ~76% | 79.4% | ~5% per 2× params |
  | AIME 2025 | 72.1% | 78.7% | 85.0% | ~7% per 2× params |
  | GPQA | 53.4% | 66.8% | 71.2% | ~9% per 2× params |

  Observation:
    - Consistent scaling with model size
    - Reasoning tasks benefit most from scale
    - Diminishing returns at larger sizes
```

---

## Deployment and Inference

Ministral 3 deployment is straightforward across multiple platforms and frameworks.

### Quick Start Guide

```yaml
Option 1 - Ollama (Easiest):

  Step 1 - Install Ollama:
    # macOS/Linux
    curl -fsSL https://ollama.com/install.sh | sh

    # Windows: Download from ollama.com

  Step 2 - Pull Model:
    ollama pull ministral:3b
    # or
    ollama pull ministral:8b
    # or
    ollama pull ministral:14b

  Step 3 - Run:
    ollama run ministral:3b

    # Or in code
    import ollama
    response = ollama.chat(model='ministral:3b', messages=[
        {'role': 'user', 'content': 'Explain quantum entanglement'}
    ])
    print(response['message']['content'])

Option 2 - vLLM (Best Performance):

  Step 1 - Install:
    pip install vllm>=0.8.0

  Step 2 - Run Server:
    vllm serve mistralai/Ministral-3-3B-Instruct-2512 \
      --dtype fp8 \
      --max-model-len 32768

  Step 3 - Use API:
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed"
    )

    response = client.chat.completions.create(
        model="mistralai/Ministral-3-3B-Instruct-2512",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

Option 3 - llama.cpp (Most Control):

  Step 1 - Download GGUF:
    # From Hugging Face (community-converted)
    huggingface-cli download \
      unsloth/Ministral-3-3B-Instruct-2512-GGUF \
      ministral-3-3b-instruct-q5_k_m.gguf

  Step 2 - Run:
    ./llama-server \
      -m ministral-3-3b-instruct-q5_k_m.gguf \
      -c 8192 \
      -ngl 999 \
      --host 0.0.0.0 \
      --port 8080

  Step 3 - Use API (OpenAI-compatible):
    curl http://localhost:8080/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7
      }'
```

### Configuration Options

```yaml
Context Length:

  Native Support: 256,000 tokens

  Practical Settings:
    vLLM:
      --max-model-len 32768   # 32K (recommended for general use)
      --max-model-len 131072  # 128K (for long documents)
      --max-model-len 262144  # 256K (maximum, high memory)

    llama.cpp:
      -c 8192   # 8K (fast, memory-efficient)
      -c 32768  # 32K (balanced)
      -c 131072 # 128K (long context)

  Memory Impact (Ministral 3B):
    8K context: +0.5 GB KV cache
    32K context: +2 GB KV cache
    128K context: +8 GB KV cache
    256K context: +16 GB KV cache

Precision Selection:

  vLLM:
    --dtype fp8      # Best for Instruct variants (FP8 weights)
    --dtype bfloat16 # Best for Reasoning variants (BF16 weights)
    --dtype float16  # Alternative to bfloat16
    --dtype auto     # Auto-detect

  llama.cpp (via GGUF format):
    Q8_0: Highest quality quantized (~98% of FP16)
    Q6_K: Great balance (~95%)
    Q5_K_M: Recommended (~93%)
    Q4_K_M: Memory-limited (~88%)

Batch Size and Throughput:

  vLLM (Continuous Batching):
    --max-num-seqs 8       # Max concurrent requests
    --max-num-batched-tokens 8192  # Total tokens per batch

    Example:
      8 sequences × 1024 tokens each = 8192 batched tokens

  llama.cpp:
    -b 512  # Batch size for prompt processing
    -ub 256 # Batch size for generation

  Trade-off:
    Larger batches: Higher throughput, more memory
    Smaller batches: Lower latency, less memory

Temperature and Sampling:

  For Deterministic Output (function calling, structured):
    temperature: 0.0
    top_p: 1.0
    top_k: 1

  For Creative Output (writing, brainstorming):
    temperature: 0.7 - 1.0
    top_p: 0.9
    top_k: 40

  For Balanced (general conversation):
    temperature: 0.3 - 0.5
    top_p: 0.95
    top_k: 50

GPU Memory Utilization:

  vLLM:
    --gpu-memory-utilization 0.90  # Use 90% of GPU memory

    Recommendation:
      RTX GPU: 0.85-0.90 (leave margin for OS)
      Jetson: 0.80-0.85 (more conservative)
      Headless server: 0.95 (maximum)

Caching:

  vLLM Prefix Caching:
    --enable-prefix-caching

    Benefit:
      - Caches common prompt prefixes
      - 50-90% reduction in TTFT for cached prompts
      - Ideal for applications with system prompts

  llama.cpp:
    --cache-type-k f16  # FP16 KV cache (quality)
    --cache-type-k q8_0 # Quantized KV cache (memory)
```

### Multi-GPU Deployment

```yaml
Tensor Parallelism (Single Machine, Multiple GPUs):

  vLLM:
    vllm serve mistralai/Ministral-3-14B-Instruct-2512 \
      --tensor-parallel-size 2 \
      --dtype fp8

    Use Cases:
      - 14B model on 2× RTX 4090 (24GB each)
      - Split model across GPUs
      - Parallel computation

  llama.cpp:
    ./llama-server \
      -m ministral-14b.gguf \
      -ngl 999 \
      --split-mode layer \
      -ts 0.8,0.2  # Split: 80% GPU 0, 20% GPU 1

Pipeline Parallelism (Not Common for Edge):

  - Splits layers across GPUs sequentially
  - Less common for edge deployment
  - Better for datacenter with many GPUs

Multi-Instance Serving (Separate Models per GPU):

  Use Case: Serve different models on different GPUs

  Example (2× RTX 4090):
    # GPU 0: Ministral 3B Instruct
    CUDA_VISIBLE_DEVICES=0 vllm serve \
      mistralai/Ministral-3-3B-Instruct-2512 \
      --port 8000 &

    # GPU 1: Ministral 8B Reasoning
    CUDA_VISIBLE_DEVICES=1 vllm serve \
      mistralai/Ministral-3-8B-Reasoning-2512 \
      --port 8001 &

  Benefit:
    - Use Instruct for general queries
    - Use Reasoning for complex problems
    - Route requests based on task type
```

### Edge-Specific Deployment

```yaml
Jetson Deployment:

  Step 1 - Install JetPack SDK:
    # Pre-installed on Jetson devices
    # Or flash from NVIDIA SDK Manager

  Step 2 - Install TensorRT-LLM (Recommended):
    git clone https://github.com/NVIDIA/TensorRT-LLM
    cd TensorRT-LLM
    # Follow Jetson-specific build instructions

  Step 3 - Convert Model:
    python convert_checkpoint.py \
      --model_dir /path/to/Ministral-3-3B-Instruct-2512 \
      --dtype int8 \
      --output_dir /tmp/ministral-3b-jetson

  Step 4 - Build Engine:
    trtllm-build \
      --checkpoint_dir /tmp/ministral-3b-jetson \
      --output_dir /engines/ministral-3b \
      --max_batch_size 1 \
      --max_input_len 8192 \
      --max_output_len 2048

  Step 5 - Run:
    python run.py \
      --engine_dir /engines/ministral-3b \
      --tokenizer_dir /path/to/tokenizer

  Alternative (llama.cpp - Easier):
    # Download GGUF
    wget https://huggingface.co/.../ministral-3b-q4_k_m.gguf

    # Run
    ./llama-server -m ministral-3b-q4_k_m.gguf \
      -ngl 999 -c 8192

Raspberry Pi (Not Recommended, but Possible):

  Model: Ministral 3B only (Q3/Q4 quantization)
  Performance: 1-5 tokens/second (very slow)
  Memory: 8GB RAM minimum

  Use llama.cpp:
    # CPU-only inference
    ./llama-server -m ministral-3b-q3_k_m.gguf \
      -ngl 0 \
      -t 4 \
      -c 2048

  Reality: Too slow for interactive use

Docker Deployment:

  Dockerfile (vLLM):
    FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

    RUN pip install vllm>=0.8.0

    CMD vllm serve mistralai/Ministral-3-8B-Instruct-2512 \
        --dtype fp8 \
        --host 0.0.0.0 \
        --port 8000

  Build and Run:
    docker build -t ministral-3-8b .
    docker run --gpus all -p 8000:8000 ministral-3-8b

Kubernetes Deployment (Edge Cluster):

  Deployment YAML:
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: ministral-3b
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: ministral-3b
      template:
        metadata:
          labels:
            app: ministral-3b
        spec:
          containers:
          - name: vllm
            image: vllm/vllm-openai:latest
            args:
            - --model
            - mistralai/Ministral-3-3B-Instruct-2512
            - --dtype
            - fp8
            resources:
              limits:
                nvidia.com/gpu: 1
            ports:
            - containerPort: 8000
```

---

## Model Variants and Selection Guide

### All Available Models

```yaml
Ministral 3B Family:

  1. Ministral-3-3B-Base-2512:
     Parameters: ~4.0B (3.6B language + 0.4B vision)
     Format: BF16
     Use: Pre-training foundation, fine-tuning base
     Availability: Hugging Face

  2. Ministral-3-3B-Instruct-2512:
     Parameters: ~4.0B
     Format: FP8
     Use: General-purpose chat, edge AI
     Availability: Hugging Face, Ollama, APIs

  3. Ministral-3-3B-Reasoning-2512:
     Parameters: ~4.0B
     Format: BF16
     Use: Math, logic, complex reasoning
     Availability: Hugging Face

Ministral 8B Family:

  4. Ministral-3-8B-Base-2512:
     Parameters: ~9.0B (8.6B language + 0.4B vision)
     Format: BF16
     Use: Pre-training foundation, fine-tuning base
     Availability: Hugging Face

  5. Ministral-3-8B-Instruct-2512:
     Parameters: ~9.0B
     Format: FP8
     Use: Advanced chat, edge deployment
     Availability: Hugging Face, Ollama, APIs

  6. Ministral-3-8B-Reasoning-2512:
     Parameters: ~9.0B
     Format: BF16
     Use: Advanced reasoning, problem-solving
     Availability: Hugging Face

Ministral 14B Family:

  7. Ministral-3-14B-Base-2512:
     Parameters: ~14B (13.6B language + 0.4B vision)
     Format: BF16
     Use: Pre-training foundation, fine-tuning base
     Availability: Hugging Face

  8. Ministral-3-14B-Instruct-2512:
     Parameters: ~14B
     Format: FP8
     Use: High-quality chat, enterprise edge
     Availability: Hugging Face, Ollama, APIs

  9. Ministral-3-14B-Reasoning-2512:
     Parameters: ~14B
     Format: BF16
     Use: Exceptional reasoning, AIME-level math
     Availability: Hugging Face

GGUF Variants (Community):

  Available on Hugging Face (e.g., unsloth/Ministral-3-*-GGUF):
    - Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M
    - For llama.cpp and Ollama
    - Multiple quantization levels
```

### Selection Decision Tree

```yaml
Step 1 - Choose Size Based on Hardware:

  GPU Memory Available:
    < 8GB VRAM:
      → Ministral 3B (Q4_K_M quantization)

    8-16GB VRAM:
      → Ministral 3B (FP8/INT8) or 8B (Q4_K_M)

    16-24GB VRAM:
      → Ministral 8B (FP8/INT8) or 14B (Q4_K_M)

    24-32GB VRAM:
      → Ministral 14B (FP8/INT8)

    > 32GB VRAM:
      → Ministral 14B (BF16) or consider Mistral Large 3

Step 2 - Choose Variant Based on Task:

  General Use (Chat, Q&A, Documents):
    → Instruct variant

  Complex Math/Logic (AIME, Competition Problems):
    → Reasoning variant

  Research or Fine-Tuning:
    → Base variant

  Uncertain / Multiple Tasks:
    → Instruct (most versatile)

Step 3 - Choose Format Based on Framework:

  Using vLLM:
    → FP8 (Instruct) or BF16 (Reasoning)

  Using llama.cpp or Ollama:
    → GGUF (Q5_K_M recommended)

  Using TensorRT-LLM (Jetson):
    → INT8 or INT4

Examples:

  "I have a RTX 4070 Ti (16GB) and want a general assistant":
    → Ministral 3-8B-Instruct-2512 (FP8 via vLLM)
    Reason: 8B fits comfortably, Instruct is versatile

  "I have Jetson AGX Orin (32GB) for a drone, need vision + navigation":
    → Ministral 3-3B-Instruct-2512 (INT4 via TensorRT-LLM)
    Reason: 3B is fast, Instruct handles vision + commands

  "I have RTX 3090 (24GB) and need to solve competition math":
    → Ministral 3-14B-Reasoning-2512 (INT8/FP8)
    Reason: 14B Reasoning excels at AIME-level problems

  "I want to fine-tune for medical transcription":
    → Ministral 3-8B-Base-2512
    Reason: Base model, fine-tune on medical data

  "I have a laptop with 8GB integrated GPU (e.g., MacBook M3)":
    → Ministral 3-3B-Instruct-2512 (Q4_K_M via llama.cpp)
    Reason: Quantized 3B fits, llama.cpp supports Apple Silicon
```

---

## Use Cases and Applications

### Robotics and Autonomous Systems

```yaml
Humanoid Robots:

  Use Case: General-purpose humanoid assistant
    Hardware: NVIDIA Jetson Thor, Ministral 3B Instruct
    Capabilities:
      - Natural language interaction
      - Visual scene understanding
      - Task planning and execution
      - Object manipulation

  Example Workflow:
    Human: "Can you bring me the red mug from the kitchen?"

    Robot (Ministral 3B):
      1. Parse command: Find "red mug" in "kitchen"
      2. Navigate to kitchen (using vision for obstacles)
      3. Visual search: Identify red mug
      4. Plan grasp: Approach angle, force
      5. Execute: Pick up mug
      6. Navigate back to human
      7. Report: "Here's your red mug"

  Integration:
    - Robot camera feed → Ministral 3B vision
    - Command → Ministral 3B language
    - Output → Motion planning system
    - Real-time latency: <200ms per decision

Warehouse Automation (AMRs):

  Use Case: Autonomous mobile robots for inventory
    Hardware: Jetson AGX Orin, Ministral 8B Instruct
    Capabilities:
      - Voice commands from workers
      - Visual inventory verification
      - Dynamic route planning
      - Multi-robot coordination

  Example:
    Worker: "Move all boxes labeled 'fragile' to zone B3"

    Robot:
      1. Scan warehouse (visual understanding)
      2. Identify boxes with "fragile" labels (OCR)
      3. Plan optimal route (avoid obstacles, other robots)
      4. Execute pick-and-place operations
      5. Confirm completion

Agricultural Robots:

  Use Case: Autonomous weeding and monitoring
    Hardware: Jetson Orin NX, Ministral 3B Instruct
    Capabilities:
      - Crop health assessment (visual)
      - Weed identification
      - Pest detection
      - Yield estimation

  Example:
    Task: "Inspect Row 5 for pests and weeds"

    Robot:
      1. Navigate to Row 5
      2. Camera capture every plant
      3. Ministral analyzes each image
      4. Detects: 3 weeds, 1 aphid infestation
      5. Marks locations for treatment
      6. Generates report

Industrial Inspection Robots:

  Use Case: Quality control and safety inspection
    Hardware: Jetson AGX Orin, Ministral 14B Reasoning
    Capabilities:
      - Defect detection (vision)
      - Safety hazard identification
      - Compliance verification
      - Report generation

  Example:
    Input: [Image of manufacturing line]
    Task: "Check for safety violations"

    Ministral 14B Reasoning:
      - Analyzes image
      - Identifies: Loose cable (trip hazard)
      - Verifies: Hard hats worn (compliant)
      - Checks: Fire extinguisher present (compliant)
      - Report: "1 hazard found: Loose cable at position (X,Y)"
```

### Autonomous Drones

```yaml
Infrastructure Inspection:

  Use Case: Bridge inspection for structural damage
    Hardware: Jetson Thor, Ministral 3B Reasoning
    Flight Time: 30-45 minutes (battery)

  Workflow:
    1. Drone flies programmed route along bridge
    2. Captures high-res images every 2 meters
    3. Ministral 3B analyzes each image on-board
    4. Detects: Cracks, rust, concrete spalling
    5. Flags severe defects for immediate attention
    6. Stores detailed report with GPS coordinates

  Advantages:
    - On-board processing (no cloud latency)
    - Works in areas without network
    - Real-time decision (re-route if needed)
    - Comprehensive coverage

Delivery Drones:

  Use Case: Autonomous package delivery
    Hardware: Jetson Orin NX, Ministral 3B Instruct
    Range: 5-10 km

  Capabilities:
    - Visual navigation (avoid obstacles)
    - Landing zone assessment
    - Package verification (via vision)
    - Customer interaction (voice/text)

  Example:
    Drone arrives at delivery address

    Ministral 3B:
      - Analyzes landing zone (camera feed)
      - "Safe landing zone identified: Clear grass area"
      - Descends and lands
      - Waits for customer
      - Verifies customer via QR code (vision)
      - Releases package
      - "Package delivered successfully"

Search and Rescue:

  Use Case: Locate missing persons in wilderness
    Hardware: Jetson Thor, Ministral 3B Instruct
    Scenario: Emergency response

  Capabilities:
    - Thermal imaging analysis
    - Visual person detection
    - Terrain assessment
    - Communication relay

  Example:
    Drone scans forest area

    Ministral 3B:
      - Processes camera + thermal feeds
      - Detects human heat signature
      - Vision confirms: Person in distress
      - Assesses: Injured, unable to move
      - Communicates: GPS coordinates to rescue team
      - Hovers: Provides visual confirmation
      - Drops: Emergency supplies

Agricultural Monitoring:

  Use Case: Crop health and irrigation monitoring
    Hardware: Jetson Orin NX, Ministral 8B Instruct
    Coverage: 100+ acres per flight

  Workflow:
    - Drone surveys entire field
    - Captures multispectral images
    - Ministral analyzes plant health
    - Identifies areas needing water/fertilizer
    - Generates prescription map
    - Integrates with irrigation system

  Benefits:
    - Early pest/disease detection
    - Optimized resource usage
    - Increased yield
    - Reduced costs
```

### On-Device AI Assistants

```yaml
Privacy-Focused Personal Assistant:

  Use Case: Fully local AI assistant (no cloud)
    Hardware: RTX 4090, Ministral 8B Instruct
    Privacy: All data stays on device

  Capabilities:
    - Email drafting and summarization
    - Calendar management
    - Document analysis (vision)
    - Code assistance
    - General Q&A

  Example:
    User: "Summarize my emails from this week and draft a response
           to the most urgent one"

    Ministral 8B:
      1. Reads local email database
      2. Analyzes: 45 emails received
      3. Identifies urgent: Project deadline reminder
      4. Summarizes all emails (bullet points)
      5. Drafts response to urgent email
      6. Presents for user review

    All processing local, no data sent to cloud

Offline Medical Assistant:

  Use Case: Medical note-taking in rural clinic
    Hardware: RTX 4070 Ti, Ministral 14B Instruct
    Scenario: No reliable internet

  Capabilities:
    - Doctor-patient conversation transcription
    - Structured clinical note generation
    - Medical term lookup
    - Treatment suggestions (reference only)

  Example:
    Doctor: [Consultation with patient]

    Ministral 14B (after consultation):
      - Transcribes conversation
      - Extracts: Chief complaint, symptoms, diagnosis
      - Generates SOAP note
      - Suggests: Relevant tests, treatments (for review)
      - All done offline, HIPAA-compliant (data stays local)

Smart Home Hub:

  Use Case: Central AI for home automation
    Hardware: NVIDIA Jetson Orin Nano, Ministral 3B Instruct
    Power: <15W

  Capabilities:
    - Voice commands
    - Home monitoring (cameras)
    - Anomaly detection
    - Energy optimization

  Example:
    User: "Turn off lights in empty rooms"

    Ministral 3B:
      1. Analyzes camera feeds (all rooms)
      2. Detects: Living room and bedroom empty
      3. Verifies: Kitchen occupied
      4. Commands: Turn off lights (living room, bedroom)
      5. Reports: "Lights turned off in 2 rooms, saving energy"

Educational Tutor:

  Use Case: Personalized math tutor for students
    Hardware: Student laptop (RTX 3050), Ministral 14B Reasoning
    Subject: Mathematics (algebra to calculus)

  Capabilities:
    - Problem-solving explanations
    - Step-by-step guidance
    - Practice problem generation
    - Visual diagram understanding

  Example:
    Student: [Photo of homework problem]
            "I don't understand how to solve this quadratic equation"

    Ministral 14B Reasoning:
      1. Reads problem via vision: "Solve x² + 5x + 6 = 0"
      2. Explains approach: "Let's use factoring"
      3. Step 1: "Find two numbers that multiply to 6 and add to 5"
      4. Step 2: "Those numbers are 2 and 3"
      5. Step 3: "(x + 2)(x + 3) = 0"
      6. Step 4: "So x = -2 or x = -3"
      7. Verifies: Substitutes back into original equation
      8. Asks: "Do you understand? Try this similar problem..."
```

### Edge Enterprise Applications

```yaml
Retail Analytics:

  Use Case: In-store customer analytics
    Hardware: Edge servers (RTX 4090), Ministral 8B Instruct
    Deployment: Per-store edge compute

  Capabilities:
    - Customer traffic analysis (camera feeds)
    - Shelf monitoring (out-of-stock detection)
    - Customer engagement analysis
    - Privacy-preserving analytics (local processing)

  Example:
    Store cameras → Ministral 8B

    Analysis:
      - Counts customers in each section
      - Detects: Low stock on popular item
      - Analyzes: Customer dwell time per area
      - Insights: "Electronics section has 3× normal traffic"
      - Action: Alert staff to restock, assign more help

Manufacturing Quality Control:

  Use Case: Real-time defect detection on assembly line
    Hardware: Jetson AGX Orin (per station), Ministral 14B Reasoning
    Throughput: 100+ units/hour

  Workflow:
    Each product → Camera capture → Ministral 14B

    Analysis:
      - Inspects product image
      - Checks: Component placement, soldering quality
      - Detects: Missing screw at position (X,Y)
      - Decision: Reject (send to rework station)
      - Logs: Defect type, timestamp, position
      - Reports: Real-time dashboard for supervisors

  Benefits:
    - 99.5% defect detection rate
    - Faster than human inspection
    - Consistent quality standards
    - Immediate feedback

Healthcare Imaging (Screening):

  Use Case: Radiology triage in hospital
    Hardware: Edge server (RTX 6000 Ada), Ministral 14B Reasoning
    Compliance: HIPAA (data stays on-premise)

  Capabilities:
    - X-ray/CT scan analysis (screening)
    - Prioritize urgent cases
    - Flag potential abnormalities
    - NOT for diagnosis (human radiologist required)

  Workflow:
    New scan → Ministral 14B (screening)

    Analysis:
      - Reviews X-ray image
      - Detects: Possible pneumonia (opacity in lung)
      - Priority: High (patient febrile)
      - Action: Move to top of radiologist queue

    Radiologist:
      - Reviews flagged case promptly
      - Confirms diagnosis
      - Orders treatment

  Result: 30% reduction in critical case response time

Financial Document Processing:

  Use Case: Invoice and receipt processing for accounting
    Hardware: Office workstation (RTX 4070), Ministral 8B Instruct
    Volume: 1000s of documents/month

  Capabilities:
    - OCR and data extraction (vision)
    - Multi-format support (PDF, images)
    - Validation and verification
    - Export to accounting software

  Example:
    Input: Scanned invoice (PDF)

    Ministral 8B:
      - Extracts: Vendor, date, invoice #, line items, total
      - Validates: Math checks out
      - Cross-references: Purchase order
      - Detects: Discrepancy (PO: $500, Invoice: $650)
      - Flags: For manual review
      - Exports: JSON to QuickBooks API

  Benefits:
    - 95% automation rate
    - 10× faster than manual entry
    - Reduced errors
    - All processing on-premise (secure)
```

---

## Licensing and Access

### Apache 2.0 License

```yaml
License: Apache License 2.0 (All Ministral 3 Models)

Key Permissions:
  Commercial Use: ✓ Fully allowed
  Modification: ✓ Fully allowed
  Distribution: ✓ Fully allowed
  Private Use: ✓ Fully allowed
  Patent Grant: ✓ Included

Requirements:
  License Notice: Must include Apache 2.0 license text
  Attribution: Preserve copyright notices
  State Changes: Document modifications (if any)

NO Restrictions On:
  - Revenue or user count
  - Deployment scale
  - Model derivatives
  - Commercial services
  - Competitive products

What You CAN Do:
  ✓ Deploy in production (any scale)
  ✓ Fine-tune on proprietary data
  ✓ Offer paid AI services
  ✓ Integrate into products
  ✓ Create derivative models
  ✓ Keep modifications private
  ✓ Use in robotics/drones commercially
  ✓ Compete with Mistral AI

What You CANNOT Do:
  ✗ Use Mistral trademarks without permission
  ✗ Hold Mistral liable for damages
  ✗ Claim endorsement by Mistral

Comparison to Other Licenses:

  Llama 3.2 (Meta):
    - Custom license
    - Restrictions on >700M MAU
    - Cannot use to improve other models
    - More restrictive

  Qwen 2.5 (Alibaba):
    - Apache 2.0 (same as Ministral 3)
    - Fully permissive

  Gemma 2 (Google):
    - Custom license
    - Restrictions on derivatives
    - More restrictive

  Ministral 3:
    - Apache 2.0
    - Truly open
    - No hidden restrictions
```

### Download Locations

```yaml
Hugging Face (Primary Source):

  Base Models:
    - https://huggingface.co/mistralai/Ministral-3-3B-Base-2512
    - https://huggingface.co/mistralai/Ministral-3-8B-Base-2512
    - https://huggingface.co/mistralai/Ministral-3-14B-Base-2512

  Instruct Models:
    - https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
    - https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
    - https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512

  Reasoning Models:
    - https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512
    - https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
    - https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512

  GGUF Variants (Community):
    - https://huggingface.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF
    - https://huggingface.co/unsloth/Ministral-3-8B-Instruct-2512-GGUF
    - https://huggingface.co/unsloth/Ministral-3-14B-Instruct-2512-GGUF

  Download Methods:
    # Hugging Face CLI
    huggingface-cli download mistralai/Ministral-3-3B-Instruct-2512

    # Python
    from huggingface_hub import snapshot_download
    snapshot_download("mistralai/Ministral-3-8B-Instruct-2512")

Ollama (Easiest for End Users):

  Pull Command:
    ollama pull ministral:3b
    ollama pull ministral:8b
    ollama pull ministral:14b

  Available Tags:
    - ministral:3b (default Q4_K_M)
    - ministral:3b-instruct
    - ministral:8b
    - ministral:14b

  Automatic Updates:
    ollama pull ministral:3b  # Re-pull for updates

Cloud Platforms (API Access):

  Mistral AI Studio:
    URL: https://console.mistral.ai/
    API Endpoint: https://api.mistral.ai/v1/chat/completions
    Model Names:
      - "ministral-3b-instruct-latest"
      - "ministral-8b-instruct-latest"
      - "ministral-14b-instruct-latest"

  Amazon Bedrock:
    Model IDs:
      - mistral.ministral-3b-instruct-v2025
      - mistral.ministral-8b-instruct-v2025
      - mistral.ministral-14b-instruct-v2025
    Regions: us-east-1, us-west-2, eu-west-1

  Azure AI Foundry:
    Model Catalog: Search "Ministral 3"
    Deployment: Serverless or Managed Compute

  Google Cloud Vertex AI:
    Availability: Coming soon (expected Q1 2026)
```

### Community Ecosystem

```yaml
Ollama Community Models:

  Official:
    - ministral:3b, ministral:8b, ministral:14b

  Community Variants:
    - ministral:3b-reasoning
    - ministral:8b-code (fine-tuned for code)
    - ministral:14b-math (fine-tuned for math)

  Custom Modelfiles:
    FROM ministral:3b
    PARAMETER temperature 0.3
    SYSTEM "You are a helpful coding assistant"

llama.cpp Ecosystem:

  GGUF Repos on Hugging Face:
    - unsloth, TheBloke (community contributors)
    - Multiple quantization levels
    - Pre-converted, ready to use

Fine-Tuned Derivatives:

  Expected Community Models (Hugging Face):
    - Ministral-3B-Medical (medical transcription)
    - Ministral-8B-Code-Instruct (code generation)
    - Ministral-14B-Finance (financial analysis)
    - Ministral-3B-Robotics (robot control)

  How to Create:
    1. Download Base model
    2. Prepare domain-specific dataset
    3. Fine-tune (QLoRA, LoRA, full fine-tune)
    4. Upload to Hugging Face
    5. Share with community

Integration Libraries:

  LangChain:
    from langchain_community.llms import Ollama
    llm = Ollama(model="ministral:8b")
    response = llm.invoke("Hello!")

  LlamaIndex:
    from llama_index.llms.ollama import Ollama
    llm = Ollama(model="ministral:14b")

  Transformers (Hugging Face):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Ministral-3-3B-Instruct-2512"
    )
```

---

## Limitations and Considerations

### Model Limitations

```yaml
Reasoning Variants (vs Dedicated Reasoning Models):

  Not as Strong as:
    - o1-preview/o1-mini (OpenAI): Dedicated reasoning architecture
    - Gemini 2.0 Reasoning: Specialized reasoning training

  Strengths:
    - Best in weight class (3B, 8B, 14B)
    - Edge-deployable
    - Faster inference than o1

  Trade-off:
    - Ministral 14B Reasoning: 85% AIME 2025
    - o1-preview: ~85-90% AIME (similar, but larger)
    - Use Ministral for edge reasoning
    - Use o1 for absolute best reasoning (cloud)

Vision Capabilities (vs Vision-First):

  Limitations:
    - Small vision encoder (0.4B vs 2-5B)
    - Not optimized for fine-grained vision tasks
    - Medical imaging: Use specialized models
    - Satellite imagery: Use vision-first models

  When to Use Ministral 3 Vision:
    - Document understanding (OCR, tables)
    - Robotics and real-time vision
    - On-device privacy-sensitive vision
    - General object recognition

  When Not to Use:
    - Medical diagnosis (not approved)
    - Highly specialized vision (satellite, microscopy)
    - Tasks requiring maximum vision quality

Context Window (Practical Limits):

  Native Support: 256K tokens

  Practical Challenges:
    - KV cache grows linearly (16GB at 256K for 3B)
    - Attention quality may degrade at extremes
    - Slower inference with very long contexts

  Recommendations:
    - Use 4K-32K for interactive applications
    - Use 128K for document analysis
    - Use 256K only when necessary
    - Consider RAG for larger corpuses

Token Efficiency (Trade-offs):

  Advantage: Produces fewer tokens for same quality

  Potential Issue:
    - Some users prefer verbose explanations
    - May seem too concise for certain applications
    - Can adjust with prompting ("Explain in detail...")

  Mitigation:
    - Temperature and prompt engineering
    - Request longer outputs if needed
    - Trade-off is generally positive (cost savings)
```

### Deployment Challenges

```yaml
Edge Hardware Requirements:

  Barrier to Entry:
    - Not accessible on very low-end hardware
    - Raspberry Pi: Too slow for practical use
    - Jetson Nano: Limited to 3B INT4 (tight)

  Minimum Recommendations:
    - 3B: 8GB VRAM or Jetson Orin NX
    - 8B: 16GB VRAM or Jetson AGX Orin
    - 14B: 24GB VRAM or high-end Jetson

  Workarounds:
    - Use cloud APIs for low-end devices
    - Deploy on edge server, access from devices
    - Extreme quantization (Q3/Q4) with quality loss

Quantization Quality Loss:

  Issue:
    - Aggressive quantization (Q4/Q3) degrades quality
    - Especially noticeable on complex tasks
    - Reasoning may suffer more than chat

  Recommendations:
    - Production: FP8/INT8 (minimal loss)
    - Edge (memory OK): Q5_K_M
    - Edge (tight): Q4_K_M (test thoroughly)
    - Avoid Q3 unless absolutely necessary

  Testing:
    - Always validate on your specific use case
    - Compare quantized vs full precision
    - Measure task-specific metrics

Power and Thermal (Mobile/Battery):

  Challenge:
    - Continuous inference drains batteries
    - Heat generation in enclosed spaces
    - Thermal throttling reduces performance

  Solutions:
    - Intermittent inference (not continuous)
    - Lower power modes (trade speed for efficiency)
    - Aggressive cooling (for drones/robots)
    - Batch inferences when possible

Latency (Real-Time Applications):

  Challenge:
    - Even 3B has 50-100ms TTFT
    - 256 token response: 650ms+ total
    - May be too slow for some real-time uses

  Solutions:
    - Use 3B for lowest latency
    - Speculative decoding (if available)
    - Shorter outputs (fewer tokens)
    - Pre-generate common responses
```

### Ethical and Safety Considerations

```yaml
Robotics Safety:

  Risks:
    - Incorrect commands could cause harm
    - Vision errors in safety-critical applications
    - Autonomous decisions without human oversight

  Mitigations:
    - Human-in-the-loop for critical decisions
    - Safety interlocks (hardware level)
    - Extensive testing in safe environments
    - Failure modes and fallbacks

  Example:
    Warehouse robot: Always check for humans before moving
    Drone: Geofence and fail-safe landing

Privacy (On-Device Data):

  Advantage: Data stays local (Ministral 3)

  Considerations:
    - Still need to secure edge devices
    - Physical access = data access
    - Encryption for sensitive applications
    - Access controls and auditing

  Best Practices:
    - Encrypt data at rest
    - Secure boot (Jetson, etc.)
    - Network isolation
    - Regular security updates

Bias and Fairness:

  Known Issues:
    - Training data biases persist
    - Vision models may have demographic biases
    - Language models reflect cultural biases

  Mitigations:
    - Test on diverse examples
    - Human review for sensitive applications
    - Fine-tune on balanced data
    - Continuous monitoring

Dual-Use Concerns (Drones, Surveillance):

  Potential Misuse:
    - Surveillance without consent
    - Autonomous weapons (prohibited use)
    - Privacy violations

  Responsible Use:
    - Comply with local regulations (drone laws)
    - Transparency about AI use
    - Ethical guidelines for deployment
    - Reject harmful applications

Medical/Legal Disclaimers:

  NOT Approved For:
    - Medical diagnosis or treatment
    - Legal advice or representation
    - Financial advice
    - Safety-critical decisions without human oversight

  Acceptable Uses:
    - Medical transcription (administrative)
    - Legal document review (assisted)
    - Financial data extraction (verified)
    - Decision support (with human approval)
```

---

## Resources and Links

### Official Documentation

```yaml
Mistral AI:
  Main Website: https://mistral.ai/
  Mistral 3 Announcement: https://mistral.ai/news/mistral-3
  Ministral 3 14B Docs: https://docs.mistral.ai/models/ministral-3-14b-25-12
  API Documentation: https://docs.mistral.ai/api/
  Console: https://console.mistral.ai/

Hugging Face:
  Ministral 3 Collection: https://huggingface.co/collections/mistralai/ministral-3
  3B Instruct: https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512
  8B Instruct: https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
  14B Instruct: https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512
  3B Reasoning: https://huggingface.co/mistralai/Ministral-3-3B-Reasoning-2512
  8B Reasoning: https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
  14B Reasoning: https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512

NVIDIA:
  Edge AI Blog: https://developer.nvidia.com/blog/nvidia-accelerated-mistral-3-open-models-deliver-efficiency-accuracy-at-any-scale/
  Jetson Platform: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/
  JetPack SDK: https://developer.nvidia.com/embedded/jetpack
  TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
```

### Technical Papers and Research

```yaml
Mistral Family Papers:

  Magistral (Reasoning Training):
    - Paper: https://arxiv.org/abs/2506.10910
    - Title: "Magistral"
    - Key: RL-based reasoning training, GRPO algorithm

  Mixtral 8×7B (MoE Architecture):
    - Paper: https://arxiv.org/abs/2401.04088
    - Title: "Mixtral of Experts"
    - Relevance: MoE concepts (not used in Ministral 3 dense)

  Mistral 7B (Foundation):
    - Paper: https://arxiv.org/abs/2310.06825
    - Title: "Mistral 7B"
    - Relevance: Sliding window attention, GQA

Note: Ministral 3 does not have a dedicated technical paper yet
```

### Community and Support

```yaml
Official Channels:
  Discord: https://discord.gg/mistralai
  GitHub: https://github.com/mistralai
  Twitter/X: https://twitter.com/MistralAI
  LinkedIn: https://linkedin.com/company/mistralai

Community Forums:
  Hugging Face: https://discuss.huggingface.co/
  Reddit r/LocalLLaMA: https://reddit.com/r/LocalLLaMA
  Reddit r/MachineLearning: https://reddit.com/r/MachineLearning
  Stack Overflow: Tag [mistral-ai]

NVIDIA Jetson Community:
  Forums: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/
  Discord: NVIDIA AI Embedded Discord
```

### Inference Frameworks

```yaml
vLLM:
  GitHub: https://github.com/vllm-project/vllm
  Documentation: https://docs.vllm.ai/
  Installation: pip install vllm>=0.8.0

llama.cpp:
  GitHub: https://github.com/ggerganov/llama.cpp
  Documentation: README on GitHub
  Compilation: make or cmake

Ollama:
  Website: https://ollama.ai/
  GitHub: https://github.com/ollama/ollama
  Installation: curl -fsSL https://ollama.com/install.sh | sh

TensorRT-LLM:
  GitHub: https://github.com/NVIDIA/TensorRT-LLM
  Documentation: https://nvidia.github.io/TensorRT-LLM/
  Jetson Guide: NVIDIA Developer Forums
```

### Tutorials and Guides

```yaml
Getting Started:
  - Ollama Quick Start: https://github.com/ollama/ollama#quickstart
  - vLLM Tutorial: https://docs.vllm.ai/en/latest/getting_started/quickstart.html
  - Hugging Face Transformers Guide: https://huggingface.co/docs/transformers

Edge Deployment:
  - NVIDIA Jetson AI Lab: https://www.jetson-ai-lab.com/
  - TensorRT-LLM Jetson: NVIDIA Developer Zone
  - llama.cpp Edge Guide: Community blogs

Robotics Integration:
  - ROS 2 + LLM Integration: Community tutorials
  - Isaac ROS: https://developer.nvidia.com/isaac-ros
  - NVIDIA Omniverse: https://developer.nvidia.com/omniverse

Community Tutorials:
  - YouTube: Search "Ministral 3 tutorial"
  - Medium: AI and edge computing tags
  - Dev.to: LLM deployment articles
```

---

## Conclusion

Ministral 3 represents a paradigm shift in accessible AI, bringing frontier-level capabilities to edge devices, robots, and consumer hardware. With **nine specialized models** across three sizes (3B, 8B, 14B) and three variants (Base, Instruct, Reasoning), the family provides unparalleled flexibility for diverse deployment scenarios.

**Key Achievements:**
- **State-of-the-art reasoning**: 85% on AIME 2025 (14B Reasoning) - rivaling models 10× larger
- **Exceptional edge performance**: 385 tokens/second on RTX 5090 (3B Instruct)
- **Universal vision**: All models include 0.4B vision encoder for multimodal reasoning
- **Extreme efficiency**: Fits in <8GB VRAM when quantized, runs on Jetson platforms
- **Token efficiency**: 10× fewer output tokens than competitors for comparable quality

**Ideal For:**
- Robotics and autonomous systems (humanoids, AMRs, drones)
- On-device AI assistants (privacy-focused, offline)
- Edge enterprise applications (retail, manufacturing, healthcare screening)
- Educational and research use (accessible, affordable)
- Privacy-sensitive deployments (all processing local)

**Technical Highlights:**
- Dense transformer architecture (optimized for edge, not MoE)
- 256K native context window
- Interleaved sliding-window attention (8B)
- Extensive NVIDIA optimizations (FP8, INT8, INT4)
- Real-time inference on Jetson Thor (52-273 TPS)

**Deployment Flexibility:**
- Multiple frameworks: vLLM, llama.cpp, Ollama, TensorRT-LLM
- Multiple platforms: RTX GPUs, Jetson, cloud APIs
- Multiple formats: FP8, BF16, GGUF (Q8 to Q3)
- Apache 2.0 license: Truly open, no restrictions

**Considerations:**
- Hardware requirements: Minimum 8GB VRAM (3B quantized) to 32GB VRAM (14B full precision)
- Vision capabilities: Document-centric strength, not vision-first quality
- Edge-first design: Optimized for single-request latency over batch throughput

Ministral 3 democratizes advanced AI by enabling researchers, developers, and organizations to deploy frontier-quality models without massive infrastructure. Whether you're building autonomous robots, privacy-focused assistants, or edge enterprise systems, Ministral 3 delivers exceptional performance at accessible scale.

**Get Started:**
- Download: https://huggingface.co/collections/mistralai/ministral-3
- Quick start: `ollama pull ministral:3b` or `ollama pull ministral:14b`
- Documentation: https://docs.mistral.ai/
- Community: https://discord.gg/mistralai

---

*Document last updated: December 2, 2025*
*Ministral 3 release date: December 2, 2025*
*Model version: 2512 (December 2025)*
