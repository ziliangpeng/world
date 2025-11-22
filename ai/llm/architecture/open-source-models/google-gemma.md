# Google Gemma Series

Gemma is Google's series of open-weight models, built on research and technology from the Gemini models, designed for responsible AI development.

## Gemma 1 (February 2024)

### Model Variants
- **2B**: 2 billion parameters
- **7B**: 7 billion parameters

### Architecture

**Type**: Decoder-only transformer

**Key Differences by Size**:
- **2B**: Multi-Query Attention (MQA)
- **7B**: Multi-Head Attention (MHA)

**Common Components**:
- RoPE position embeddings
- GeGLU activation functions
- RMSNorm normalization
- SentencePiece tokenizer

### Training
- Trained on filtered, diverse datasets
- Strong safety filters and responsible AI focus
- Base and instruction-tuned variants

## Gemma 2 (June 2024)

### Model Variants
- **2B**: 2 billion parameters
- **9B**: 9 billion parameters
- **27B**: 27 billion parameters

### Architecture Innovations

#### 1. Grouped-Query Attention (GQA)

All Gemma 2 models use GQA:
- Better efficiency than MHA
- Better quality than MQA (used in Gemma 1 2B)
- Balances memory usage and performance

#### 2. Alternating Attention Mechanisms

**Sliding Window + Global Attention**:
```
Layer 1: Local sliding window attention (e.g., 4096 tokens)
Layer 2: Global full attention
Layer 3: Local sliding window attention
Layer 4: Global full attention
...
```

**Benefits**:
- Local layers: Efficient, capture nearby dependencies
- Global layers: Long-range dependencies
- Combined: Best of both worlds
- Memory efficient while maintaining quality

#### 3. Logit Soft-Capping

**Innovation for Training Stability**:
```python
# Traditional logits
logits = model_output

# With soft-capping
cap = 30.0  # Example threshold
logits = cap * tanh(model_output / cap)
```

**Benefits**:
- Prevents extreme logit values
- Stabilizes training
- Reduces numerical issues
- Smoother optimization landscape

#### 4. Architectural Improvements

**Network Depth**:
- Gemma 2 models are deeper than Gemma 1 equivalents
- More layers for same parameter count
- Better capacity utilization

**Normalization**:
- LayerNorm without biases (cleaner, simpler)
- Per-head QK normalization
- Better training stability

**Embedding**:
- All bias terms removed from FFN and GQA layers
- Reduced parameter count for same capacity

### Training Details

**27B Model**:
- Trained on 13 trillion tokens
- Largest Gemma 2 variant

**9B Model**:
- Trained on 8 trillion tokens
- Knowledge distillation from 27B

**2B Model**:
- Distilled from 27B and 9B models
- Efficient for edge deployment

### Tokenizer Evolution

**Massive Vocabulary Expansion**:
- Gemma 1: Standard vocabulary (~32-64K range)
- **Gemma 2: 256,128 tokens**

**Exponential Expansion Strategy**:
- Much larger than typical LLMs
- Better multilingual coverage
- More efficient tokenization
- Fewer tokens per sequence

This is one of the largest vocabularies in any LLM:
- GPT-4: ~100K
- Llama 3: ~128K
- Qwen: ~152K
- **Gemma 2: ~256K**

### Knowledge Distillation

**Training Strategy**:
1. Train largest model (27B) from scratch
2. Distill to 9B model
3. Distill to 2B model

**Benefits**:
- Smaller models learn from larger models
- Better performance than training from scratch
- Efficient use of compute
- Consistent behavior across sizes

## Architectural Stack

### Gemma 2 Layer Structure

```
Input → Token Embedding
  ↓
[Repeated N times with alternating pattern]:
  RMSNorm (no bias)
  → Layer i (odd): Sliding Window Attention
    or
    Layer i (even): Global Attention
    - Grouped-Query Attention (GQA)
    - RoPE position encoding
    - Per-head QK normalization
  → Residual Connection
  → RMSNorm (no bias)
  → FFN (GeGLU activation, no bias)
  → Residual Connection
  ↓
Final RMSNorm → Output (with logit soft-capping)
```

### Sliding Window Attention

**How It Works**:
```python
# Each token attends to a window of nearby tokens
window_size = 4096  # Example

# Token at position i attends to:
# positions [i - window_size, i]

# Benefits:
# - O(n * w) instead of O(n^2) where w = window_size
# - Efficient for long sequences
# - Good for local dependencies
```

**Alternation**:
- Sliding window layers: Local context
- Global layers: Cross-document connections
- Together: Efficient + effective

## Comparison: Gemma 1 vs Gemma 2

| Feature | Gemma 1 2B | Gemma 1 7B | Gemma 2 2B | Gemma 2 9B | Gemma 2 27B |
|---------|-----------|-----------|-----------|-----------|------------|
| Attention | MQA | MHA | GQA | GQA | GQA |
| Layers | Fewer | Fewer | More | More | More |
| Vocab | ~32K | ~32K | 256K | 256K | 256K |
| Training | From scratch | From scratch | Distilled | Distilled | 13T tokens |
| Sliding Window | No | No | Yes | Yes | Yes |
| Logit Capping | No | No | Yes | Yes | Yes |
| QK Norm | No | No | Yes | Yes | Yes |

## Technical Innovations

### 1. Vocabulary Scaling

Gemma 2's 256K vocabulary is groundbreaking:
- **Multilingual efficiency**: Better coverage of non-English languages
- **Compression**: Fewer tokens needed per text
- **Rare words**: Better handling of technical terms, names
- **Challenge**: Larger embedding matrices

### 2. Attention Pattern Alternation

Novel approach to balancing efficiency and quality:
- Not purely local (like some models)
- Not purely global (like standard transformers)
- Hybrid: Best of both

### 3. Per-Head QK Normalization

```python
# Applied to each attention head independently
Q_normalized = normalize(Q, dim=-1)
K_normalized = normalize(K, dim=-1)

# Benefits:
# - Prevents attention score outliers
# - More stable gradients
# - Better multi-head collaboration
```

### 4. Bias Removal

Systematic removal of bias terms:
- FFN layers: No biases
- GQA layers: No biases
- Only LayerNorm parameters

**Benefits**:
- Fewer parameters
- Cleaner architecture
- Easier to analyze
- No degradation in quality

## Training Data

**Web Text**:
- Filtered for quality
- Safety considerations
- Diverse domains

**Code**:
- Programming languages
- GitHub and other sources
- Code understanding and generation

**Mathematics**:
- Mathematical reasoning
- Problem-solving datasets
- STEM content

**Multilingual**:
- Multiple languages
- Balanced coverage
- Cultural diversity

## Safety and Responsibility

Gemma series emphasizes responsible AI:
- Filtered training data
- Safety red-teaming
- Bias evaluations
- Clear usage guidelines
- Responsible AI toolkit

## Performance

### Benchmarks (Gemma 2)

**27B**:
- Competitive with Llama 3 70B on many tasks
- Excellent performance-to-size ratio

**9B**:
- Outperforms Gemma 1 7B significantly
- Competitive with larger models

**2B**:
- Best-in-class for its size
- Excellent for edge deployment

### Use Cases

**27B/9B**:
- General-purpose applications
- Research and development
- Production deployments

**2B**:
- On-device inference
- Resource-constrained environments
- Mobile and edge devices

## Open Weights vs Open Source

**Important Distinction**:
- Gemma is "open weights" not fully "open source"
- Weights are available for download
- Custom license (not Apache/MIT)
- Some restrictions on usage
- Code and tools are open source

**License Highlights**:
- Free for research and commercial use
- Some limitations on redistribution
- Responsible AI requirements
- Different from unrestricted open source

## Impact on the Field

### Technical Contributions

1. **Sliding Window + Global**: Novel attention pattern
2. **Logit Soft-Capping**: Training stability technique
3. **256K Vocabulary**: Pushed boundaries of tokenizer scale
4. **Distillation Strategy**: Effective smaller model training

### Practical Impact

1. **Accessibility**: High-quality models for researchers
2. **Efficiency**: Strong performance at small sizes
3. **Responsible AI**: Focus on safety and ethics
4. **Google Research**: Democratized Google's technology

## Future Directions

Gemma demonstrates trends toward:
1. **Hybrid attention**: Not purely dense or sparse
2. **Larger vocabularies**: Better multilingual coverage
3. **Distillation**: Efficient small model training
4. **Stability techniques**: Logit capping, QK norm
5. **Responsible AI**: Safety as a core design principle

## Gemma 3 (March 2025)

Gemma 3 models are multimodal, accepting both text and image input, and support over 140 languages.

### Model Variants
- **270M**: 270 million parameters
- **1B**: 1 billion parameters
- **4B**: 4 billion parameters
- **12B**: 12 billion parameters
- **27B**: 27 billion parameters

Models are available as base, instruction-tuned, and quantized aware trained (QAT) versions.



## Specialized Gemma Variants



### CodeGemma

Lightweight models for programming tasks (code completion, chat).

- **2B**

- **7B**



### PaliGemma & PaliGemma 2

Vision-language models for visual and image data processing.



### ShieldGemma & ShieldGemma 2

Safety models to assess and filter harmful content. ShieldGemma 2 (4B) is based on Gemma 3.



### MedGemma

Tailored for medical applications, including image analysis. Based on Gemma 3.

- **4B**

- **27B**



### Other Variants

- **DolphinGemma**: For studying dolphin communication.

- **DataGemma**: Fine-tuned on Google's Data Commons.

- **RecurrentGemma**: Uses a recurrent neural network architecture.

- **T5Gemma**: Text-focused variants with different parameter counts (e.g., 18B, 11B, 4B).



## Sources

- [Google launches Gemma 2](https://blog.google/technology/developers/google-gemma-2/)

- [Gemma explained: Overview](https://developers.googleblog.com/gemma-explained-overview-gemma-model-family-architectures/)

- [Gemma explained: What's new in Gemma 2](https://developers.googleblog.com/en/gemma-explained-new-in-gemma-2/)

- [Gemma 2 27B - Hugging Face](https://huggingface.co/google/gemma-2-27b)
