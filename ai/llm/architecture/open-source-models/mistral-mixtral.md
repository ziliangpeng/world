# Mistral and Mixtral Series

Mistral AI's models are known for exceptional performance-to-size ratios and pioneering work in Mixture of Experts (MoE) architectures.

## Mistral 7B (September 2023)

### Architecture
- **Type**: Dense decoder-only transformer
- **Parameters**: 7 billion
- **Context**: 8K tokens (extended to 32K in some variants)
- **Release Date**: September 2023
- **License**: Apache 2.0

### Key Features
- Grouped-Query Attention (GQA)
- Sliding Window Attention (SWA) with 4096 token window for efficient long-range dependencies
- RoPE position embeddings
- Byte-fallback BPE tokenizer with 32K vocabulary

### Significance
- Outperformed Llama 2 13B while being nearly half the size
- Demonstrated efficiency gains from architectural optimizations
- Set new standards for small model performance
- First model to popularize Sliding Window Attention at scale

## Mixtral 8x7B (December 2023)

### Architecture Type
**Sparse Mixture-of-Experts (SMoE)**

### Model Specifications
- **Total Parameters**: 46.7 billion
- **Active Parameters per Token**: 12.9 billion
- **Number of Experts**: 8
- **Experts Activated per Token**: 2
- **Context Window**: 32K tokens
- **Release Date**: December 2023
- **License**: Apache 2.0

### How It Works

Each layer contains 8 expert feed-forward networks. A router network selects the top-2 experts for each token:

```
Input Token
  ↓
Self-Attention Layer
  ↓
Router Network (learns which experts to use)
  ↓
Top-2 Expert Selection
  ↓
Expert 1 (weighted) + Expert 2 (weighted)
  ↓
Combined Output
```

### Training Details
- Multilingual: English, French, German, Spanish, Italian
- Same context window technology as Mistral 7B
- Apache 2.0 license for full commercial use

### Performance
- **Active Compute**: Only 12.9B parameters active per token
- **Quality**: Matches or exceeds 70B dense models
- **Efficiency**: ~5x more efficient than comparable dense models
- **Speed**: Faster inference than Llama 2 70B

## Mistral Small (22B) (February 2024)

### Architecture
- **Type**: Dense decoder-only transformer
- **Parameters**: 22 billion
- **Context**: 32K tokens
- **Release Date**: February 2024
- **License**: Apache 2.0 (later versions)

### Key Features
- Optimized for low-latency workloads
- Strong performance on coding tasks
- Multilingual capabilities
- Cost-effective alternative to larger models

### Performance
- Outperformed Mixtral 8x7B on many benchmarks
- Faster inference than MoE models due to dense architecture
- Better suited for edge deployment and resource-constrained environments

## Codestral (22B) (May 2024)

### Architecture
- **Type**: Dense decoder-only transformer optimized for code
- **Parameters**: 22 billion
- **Context**: 32K tokens
- **Release Date**: May 2024
- **License**: Mistral AI Non-Production License (later Apache 2.0 for Codestral Mamba)

### Training
- Trained on 80+ programming languages
- Specialized training on code completion and fill-in-the-middle tasks
- Enhanced code understanding and generation capabilities

### Key Features
- Fill-in-the-middle (FIM) capabilities for IDE integration
- Strong performance on HumanEval, MBPP, and MultiPL-E benchmarks
- Optimized for code completion, explanation, and refactoring
- Support for long-range code context (32K tokens)

### Performance
- Outperformed GPT-3.5 on code generation tasks
- Competitive with specialized code models like CodeLlama 34B
- Particularly strong on Python, JavaScript, and TypeScript

## Mistral NeMo (12B) (July 2024)

### Architecture
- **Type**: Dense decoder-only transformer
- **Parameters**: 12 billion
- **Context**: 128K tokens
- **Release Date**: July 2024 (collaboration with NVIDIA)
- **License**: Apache 2.0

### Key Innovations
- **Tekken Tokenizer**: New tokenizer with improved efficiency
  - More efficient compression for code and multilingual text
  - Better handling of special characters and technical content
  - Reduces token count by ~30% compared to standard tokenizers

### Features
- Extended 128K context window
- Drop-in replacement for Mistral 7B in many applications
- Optimized for NVIDIA GPUs
- Strong multilingual capabilities (English, French, German, Spanish, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, Hindi)

### Performance
- Better than Mistral 7B despite being only ~70% larger
- Competitive with Gemma 2 9B and Llama 3 8B
- Excellent long-context reasoning with 128K window

## Mistral Large 2 (123B) (July 2024)

### Architecture
- **Type**: Dense decoder-only transformer
- **Parameters**: 123 billion
- **Context**: 128K tokens
- **Release Date**: July 2024
- **License**: Mistral Research License (non-commercial) / Commercial license available

### Model Specifications
- **Layers**: 88
- **Attention**: Grouped-Query Attention (GQA)
- **Vocabulary**: 32K tokens
- **Hidden Dimension**: 12,288
- **FFN Dimension**: 28,672

### Key Features
- Flagship Mistral model with best overall performance
- Native function calling and tool use
- Strong multilingual support (dozens of languages)
- Advanced reasoning and mathematics capabilities
- Precise instruction following

### Performance
- Competitive with GPT-4, Claude 3 Opus, and Llama 3.1 405B
- Top-tier performance on MMLU, GSM8K, and HumanEval
- Excellent long-context understanding up to 128K tokens
- Strong performance across math, coding, and reasoning benchmarks

### Training Details
- Trained on diverse multilingual corpus
- Advanced safety training and alignment
- Optimized for both general and specialized tasks

## Codestral Mamba (7B) (July 2024)

### Architecture Type
**State Space Model (SSM) - Mamba Architecture**

### Model Specifications
- **Parameters**: 7 billion
- **Context**: 256K tokens (effective infinite context via SSM)
- **Release Date**: July 2024
- **License**: Apache 2.0

### Architectural Innovation
Unlike transformer-based models, Codestral Mamba uses the Mamba architecture:

```
Traditional Transformer:
- O(n²) complexity for attention
- Fixed context window
- Memory scales quadratically

Mamba SSM:
- O(n) linear complexity
- Theoretically infinite context
- Memory scales linearly
```

### Key Features
- **Linear Complexity**: Constant-time inference regardless of context length
- **Code Specialization**: Optimized for code completion and generation
- **Fill-in-the-middle**: Strong FIM capabilities for IDE integration
- **Extreme Efficiency**: Much faster than transformer models of similar quality

### Performance
- Competitive with CodeLlama 7B on code tasks
- Superior speed on long-context code understanding
- Excellent for real-time code completion in IDEs
- Handles entire codebases in context (256K tokens)

### Significance
- First major code-specialized SSM model
- Demonstrates SSM viability for specialized domains
- Alternative to transformer architecture for sequential tasks

## Mathstral 7B (July 2024)

### Architecture
- **Type**: Dense decoder-only transformer specialized for mathematics
- **Parameters**: 7 billion
- **Context**: 32K tokens
- **Release Date**: July 16, 2024
- **License**: Apache 2.0

### Training
- Based on Mistral 7B architecture
- Specialized training on STEM subjects and mathematical reasoning
- Named as tribute to Archimedes (2311th anniversary)
- Developed in collaboration with Project Numina

### Key Features
- Specialized for mathematical reasoning and scientific discovery
- Strong performance on mathematical benchmarks
- Optimized for scientific and technical domains

### Performance
- 56.6% on MATH benchmark
- 63.47% on MMLU
- 68.37% on MATH with majority voting
- 74.59% with strong reward model among 64 candidates

### Use Cases
- Mathematical problem solving
- Scientific research and discovery
- Educational applications in STEM
- Technical documentation and analysis

## Pixtral 12B (September 2024)

### Architecture
- **Type**: Multimodal decoder-only transformer
- **Parameters**: 12 billion (language) + 400M vision encoder
- **Context**: 128K tokens (text), variable image inputs
- **Release Date**: September 2024
- **License**: Apache 2.0

### Multimodal Architecture
- **Language Model**: Based on Mistral NeMo 12B
- **Vision Encoder**: 400M parameter vision transformer
- **Image Resolution**: Native support for high-resolution images
- **Image Tokens**: Variable, up to 1024 tokens per image

### Key Features
- Can process multiple images in a single prompt
- Arbitrary image sizes and aspect ratios
- Strong visual reasoning and understanding
- Code generation from UI screenshots
- Document understanding and OCR
- Chart and graph analysis

### Performance
- Competitive with GPT-4V on visual understanding tasks
- Strong performance on document QA and chart reasoning
- Excellent UI-to-code generation capabilities
- Good balance between speed and quality for multimodal tasks

### Use Cases
- Visual question answering
- Document analysis and extraction
- UI/UX to code conversion
- Scientific figure interpretation
- Multimodal chatbots

## Ministral 3B (October 2024)

### Architecture
- **Type**: Dense decoder-only transformer for edge computing
- **Parameters**: 3 billion
- **Context**: 128K tokens (32K on vLLM)
- **Release Date**: October 16, 2024
- **License**: Commercial (Mistral Commercial License)

### Key Features
- Part of "les Ministraux" family for on-device computing
- Designed for smartphones, tablets, and IoT devices
- Supports function calling and tool use
- Efficient inference for resource-constrained environments

### Performance
- Outperforms Mistral 7B on most benchmarks despite smaller size
- MMLU score: 60.9% (vs Gemma 2 2B: 52.4%, Llama 3.2 3B: 56.2%)
- Strong performance on reasoning and knowledge tasks

### Pricing
- $0.04 per million tokens (highly cost-effective)

### Use Cases
- Mobile and edge applications
- On-device AI assistants
- IoT and embedded systems
- Privacy-focused local inference

## Ministral 8B (October 2024)

### Architecture
- **Type**: Dense decoder-only transformer for edge computing
- **Parameters**: 8 billion
- **Context**: 128K tokens (32K on vLLM)
- **Release Date**: October 16, 2024
- **License**: Commercial (Mistral Commercial License)

### Key Features
- Part of "les Ministraux" family
- Interleaved sliding-window attention pattern for faster inference
- Memory-efficient inference mechanism
- Strong performance-to-size ratio

### Performance
- MMLU score: 65.0% (vs Llama 3.1 8B: 64.7%)
- Outperforms comparable 8B models on most benchmarks
- Excellent balance of quality and efficiency

### Pricing
- $0.10 per million tokens

### Use Cases
- Edge computing applications
- Local AI deployments
- Privacy-sensitive applications
- Cost-optimized cloud deployments

## Mistral Large 24.11 (November 2024)

### Architecture
- **Type**: Dense decoder-only transformer (updated flagship)
- **Parameters**: 123 billion
- **Context**: 131K tokens
- **Release Date**: November 19, 2024
- **License**: Mistral Commercial License

### Key Improvements over Mistral Large 2
- Enhanced long context understanding
- Improved function calling capabilities
- Better system prompt handling
- Superior performance on coding and reasoning tasks

### Performance
- State-of-the-art reasoning and knowledge capabilities
- Competitive with GPT-4, Claude 3.5 Sonnet
- Strong multilingual support
- Excellent at following complex instructions

### Pricing
- $2.00 per million input tokens
- $6.00 per million output tokens

### Availability
- Azure AI Studio
- Google Vertex AI
- Mistral API (la Plateforme)
- Various cloud providers

## Pixtral Large (November 2024)

### Architecture Type
**Frontier Multimodal Model**

### Model Specifications
- **Language Parameters**: 123 billion
- **Vision Encoder**: 1 billion parameters
- **Context**: 128K tokens
- **Image Capacity**: Minimum 30 high-resolution images
- **Release Date**: November 18, 2024
- **License**: Mistral Research License (MRL)

### Multimodal Architecture
- Built on top of Mistral Large 2
- 123B parameter multimodal decoder
- 1B parameter vision encoder
- Native support for interleaved text and images

### Key Features
- Frontier-level image understanding
- Document, chart, and natural image comprehension
- Maintains text-only performance of Mistral Large 2
- High-resolution image processing
- Multiple image handling in single context

### Performance
- **MathVista**: 69.4% (best among all models tested)
- Outperforms Claude 3.5 Sonnet, Gemini 1.5 Pro, GPT-4o on MM-MT-Bench
- State-of-the-art visual reasoning
- Excellent document understanding and OCR

### Use Cases
- Advanced document analysis
- Scientific figure interpretation
- Visual question answering at scale
- Multi-image reasoning tasks
- Chart and graph analysis
- UI/UX understanding and code generation

### Availability
- API: `pixtral-large-latest`
- Self-deployment: HuggingFace (Mistral Large 24.11)
- Integrated into le Chat platform

## Mixtral 8x22B (April 2024)

### Architecture Type
**Sparse Mixture-of-Experts (SMoE) - Scaled Up**

### Model Specifications
- **Total Parameters**: 141 billion
- **Active Parameters per Token**: ~39 billion
- **Number of Experts**: 8
- **Experts Activated per Token**: 2
- **Context Window**: 64K tokens
- **Release Date**: April 2024
- **License**: Apache 2.0

### Key Improvements
- **Doubled context**: 32K → 64K tokens
- **Native function calling**: Built-in tool use capabilities
- **Better reasoning**: Enhanced mathematical and coding abilities
- **Multilingual**: Expanded language coverage

### Performance Characteristics
- Comparable to GPT-4 on many benchmarks
- Significantly faster than dense models of similar quality
- Excellent cost-performance ratio

### Use Cases
- Complex reasoning tasks
- Code generation and debugging
- Long document analysis (64K context)
- Function calling and agent workflows

## Sparse MoE Architecture Benefits

### Computational Efficiency
1. **Sparse Activation**: Only 25% of weights active (2 of 8 experts)
2. **Faster Inference**: Lower compute per token than dense models
3. **Better Scaling**: Can increase capacity without proportional compute increase

### Training Advantages
1. **Specialization**: Different experts learn different patterns
2. **Capacity**: More total parameters without proportional costs
3. **Flexibility**: Can balance generalization vs specialization

### Challenges Addressed
1. **Load Balancing**: Router learns to distribute tokens evenly across experts
2. **Expert Collapse**: Auxiliary losses prevent all tokens going to same expert
3. **Communication Overhead**: Efficient expert routing minimizes latency

## Architectural Details

### Common Components (Both Models)

**Attention Mechanism**:
- Grouped-Query Attention (GQA)
- Sliding Window Attention for local context
- Full attention periodically for global context

**Position Encoding**:
- RoPE (Rotary Position Embeddings)
- Scaled for extended context windows

**Normalization**:
- RMSNorm with pre-normalization

**Activation**:
- SwiGLU in expert feed-forward networks

**Tokenizer**:
- SentencePiece BPE
- ~32K vocabulary

### MoE Layer Structure

```python
# Conceptual structure
class MixtralMoELayer:
    def __init__(self):
        self.experts = [FFN() for _ in range(8)]  # 8 expert networks
        self.router = RouterNetwork()  # Learns expert selection

    def forward(self, x):
        # Router outputs probabilities for each expert
        expert_probs = self.router(x)

        # Select top-2 experts per token
        top2_indices, top2_weights = top_k(expert_probs, k=2)

        # Compute expert outputs
        expert_outputs = [
            self.experts[idx](x) * weight
            for idx, weight in zip(top2_indices, top2_weights)
        ]

        # Combine weighted expert outputs
        return sum(expert_outputs)
```

## Mistral Model Family Overview

### Timeline and Evolution

**2023:**
- **September**: Mistral 7B - First model, established core architecture
- **December**: Mixtral 8x7B - First open-source production MoE

**2024:**
- **February**: Mistral Small 22B - Mid-size dense model
- **April**: Mixtral 8x22B - Scaled up MoE with function calling
- **May**: Codestral 22B - Code-specialized model
- **July**: Mistral NeMo 12B (w/ NVIDIA, new Tekken tokenizer)
- **July**: Mistral Large 2 123B - Flagship model
- **July**: Codestral Mamba 7B - First SSM architecture
- **July**: Mathstral 7B - Math/STEM specialized model
- **September**: Pixtral 12B - First multimodal model
- **October**: Ministral 3B/8B - Edge computing models
- **November**: Mistral Large 24.11 - Updated flagship
- **November**: Pixtral Large - Frontier multimodal (123B + 1B vision)

## Comparison: Mistral Model Lineup

| Model | Type | Params | Active | Context | Release | Key Feature |
|-------|------|--------|--------|---------|---------|-------------|
| Mistral 7B | Dense | 7B | 7B | 32K | Sep 2023 | Sliding Window Attention |
| Mixtral 8x7B | MoE | 46.7B | 12.9B | 32K | Dec 2023 | First open MoE |
| Mistral Small | Dense | 22B | 22B | 32K | Feb 2024 | Low latency |
| Mixtral 8x22B | MoE | 141B | 39B | 64K | Apr 2024 | Function calling |
| Codestral | Dense | 22B | 22B | 32K | May 2024 | Code specialized |
| Mistral NeMo | Dense | 12B | 12B | 128K | Jul 2024 | Tekken tokenizer |
| Mistral Large 2 | Dense | 123B | 123B | 128K | Jul 2024 | Flagship model |
| Codestral Mamba | SSM | 7B | 7B | 256K | Jul 2024 | Linear complexity |
| Mathstral | Dense | 7B | 7B | 32K | Jul 2024 | Math/STEM specialized |
| Pixtral 12B | Multimodal | 12B+400M | 12B+400M | 128K | Sep 2024 | Vision + language |
| Ministral 3B | Dense | 3B | 3B | 128K | Oct 2024 | Edge/on-device |
| Ministral 8B | Dense | 8B | 8B | 128K | Oct 2024 | Edge/on-device |
| Mistral Large 24.11 | Dense | 123B | 123B | 131K | Nov 2024 | Updated flagship |
| Pixtral Large | Multimodal | 123B+1B | 123B+1B | 128K | Nov 2024 | Frontier multimodal |

**Key Insights:**
- **MoE models** achieve similar quality to much larger dense models while using only ~25% compute per token
- **Context evolution**: 8K → 32K → 64K → 128K → 256K tokens
- **Architectural diversity**: Dense transformers, MoE, SSM, and multimodal
- **Specialization trend**: General → Code → Math → Vision → Edge computing
- **Scale range**: From 3B edge models to 141B MoE systems

## Impact and Applications

### Research Impact
- Validated sparse MoE for production LLMs
- Demonstrated expert specialization in practice
- Showed path to scaling beyond dense model limits

### Production Use Cases
- **High-throughput applications**: Where inference speed matters
- **Long-context tasks**: 32K-64K token windows
- **Multi-domain applications**: Experts can specialize by domain
- **Cost-sensitive deployments**: Better performance per compute dollar

### Open Source Contribution
- Apache 2.0 license enables commercial use
- Weights and inference code fully available
- Enabled research into MoE architectures
- Inspired other MoE models (Qwen MoE, DeepSeek MoE)

## Future Directions

Mixtral demonstrated that:
1. Sparse MoE can match or exceed dense model quality
2. Expert routing can be learned effectively
3. Context windows can scale with MoE architectures
4. Open-source MoE is viable for production

This paved the way for even larger MoE models like DeepSeek-V3 (671B total, 37B active).

## Sources

### Official Mistral AI Announcements
- [Mistral 7B Release](https://mistral.ai/news/announcing-mistral-7b)
- [Mixtral of Experts](https://mistral.ai/news/mixtral-of-experts)
- [Cheaper, Better, Faster, Stronger - Mixtral 8x22B](https://mistral.ai/news/mixtral-8x22b)
- [Codestral: Hello, World!](https://mistral.ai/news/codestral)
- [Mistral NeMo](https://mistral.ai/news/mistral-nemo)
- [Mistral Large 2](https://mistral.ai/news/mistral-large-2407)
- [Codestral Mamba](https://mistral.ai/news/codestral-mamba)
- [Mathstral 7B](https://mistral.ai/news/mathstral)
- [Pixtral 12B](https://mistral.ai/news/pixtral-12b)
- [Ministral 3B and 8B](https://mistral.ai/news/ministraux)
- [Pixtral Large](https://mistral.ai/news/pixtral-large)

### Research Papers
- [Mistral 7B - arXiv](https://arxiv.org/abs/2310.06825)
- [Mixtral of Experts - arXiv](https://arxiv.org/abs/2401.04088)
- [Mamba: Linear-Time Sequence Modeling - arXiv](https://arxiv.org/abs/2312.00752)

### Technical Resources
- [Mistral AI Documentation](https://docs.mistral.ai/)
- [HuggingFace Mistral Models](https://huggingface.co/mistralai)
- [Mistral AI GitHub](https://github.com/mistralai)

### Analysis and Comparisons
- [Mistral vs Mixtral Comparison](https://towardsdatascience.com/mistral-vs-mixtral-comparing-the-7b-8x7b-and-8x22b-large-language-models-58ab5b2cc8ee/)
- [Mixtral 8x22B Release Coverage](https://siliconangle.com/2024/04/10/mistralai-debuts-mixtral-8x22b-one-powerful-open-source-ai-models-yet/)
- [Mistral Large 2 Analysis](https://techcrunch.com/2024/07/24/mistral-releases-mistral-large-2-its-latest-flagship-ai-model)
- [Mathstral 7B Analysis - MarkTechPost](https://www.marktechpost.com/2024/07/16/mistral-ai-unveils-mathstral-7b-and-math-fine-tuning-base-achieving-56-6-on-math-and-63-47-on-mmlu-restructuring-mathematical-discovery/)
- [Ministral 3B and 8B Release - SiliconANGLE](https://siliconangle.com/2024/10/16/mistral-introduces-ministral-3b-8b-device-ai-computing-models/)
- [Mistral Large 24.11 on Vertex AI - Google Cloud Blog](https://cloud.google.com/blog/products/ai-machine-learning/announcing-new-mistral-large-model-on-vertex-ai)
- [Pixtral Large Release - MarkTechPost](https://www.marktechpost.com/2024/11/18/mistral-ai-releases-pixtral-large-a-124b-open-weights-multimodal-model-built-on-top-of-mistral-large-2/)
