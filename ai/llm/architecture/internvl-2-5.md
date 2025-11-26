# InternVL 2.5: Open-Source Vision-Language Model

## Overview

InternVL 2.5 is an advanced open-source multimodal large language model (MLLM) series developed by OpenGVLab and Shanghai AI Lab. Released in December 2024, it represents a significant milestone in open-source AI research, becoming the first open-source MLLM to achieve over 70% accuracy on the MMMU (Multimodal Multitask Understanding) benchmark—a challenging dataset of college-level multimodal questions.

The InternVL family positions itself as "a pioneering open-source alternative to GPT-4o," bridging the gap between open-source models and proprietary commercial systems like OpenAI's GPT-4o and Google's Gemini.

### OpenGVLab and Shanghai AI Lab

OpenGVLab is a research group from Shanghai AI Lab focused on "General Vision" (GV) research, emphasizing vision-centric AI that can be easily adapted to new vision-based tasks. The organization maintains a commitment to open-source development, providing comprehensive models, code, and documentation to the broader AI community.

Shanghai AI Lab, a top-tier AI research institution in China, provides the foundational support and resources for InternVL's development. This institutional backing ensures sustained research and development of cutting-edge multimodal AI technologies.

## Evolution of InternVL

### InternVL 1.x Series (2023-2024)

InternVL 1.0 pioneered the vision foundation model aligned with large language models, establishing the core "ViT-MLP-LLM" architecture that would persist through subsequent versions. The early versions demonstrated competitive performance on standard benchmarks but had limitations in high-resolution processing and language-specific capabilities.

### InternVL 1.1 (January 2024)

Enhanced Chinese language support and OCR capabilities through increased resolution to 448×448 pixels. The team collected high-quality bilingual datasets covering common scenes and document images with English-Chinese question-answer pairs, significantly improving performance in OCR and Chinese-related understanding tasks.

### InternVL 1.5 (April 2024)

Introduced robust bilingual capabilities with dynamic high-resolution processing, supporting up to 4K resolution through tiling strategies. InternVL 1.5 surpassed proprietary models like GPT-4V in Chinese-related tasks, establishing a new benchmark for open-source bilingual vision-language models.

Notably, InternVL 1.5 demonstrated that with a more powerful vision encoder (InternViT-6B instead of CLIP-ViT), the resulting models could outperform larger language model-based approaches, validating the importance of vision encoder quality.

### InternVL 2.0 (July 2024)

A major upgrade featuring:
- Expanded model family from 1B to 108B parameters
- Extended 8k context window for handling long documents
- Support for multiple images, long texts, medical data, and videos
- Dynamic resolution strategy with up to 40 tiles (4K resolution)
- Progressive alignment training strategy
- Achieved 62.0% on MMMU, matching GPT-4o performance

InternVL 2.0 established dominance in open-source MMOLMs, consistently outperforming competitors on document understanding, chart comprehension, OCR, and mathematical reasoning tasks.

### InternVL 2.5 (December 2024)

Maintains the proven architecture while introducing significant enhancements:
- First open-source MLLM to exceed 70% on MMMU benchmark
- Improved training strategies with better data quality filtering
- Enhanced test-time scaling through Chain-of-Thought (CoT) reasoning
- Progressive scaling that reduces training data dependency
- Support for the newest InternLM 2.5 and Qwen 2.5 language models

## InternVL 2.5 Improvements Over 2.0

InternVL 2.5 builds upon the solid foundation of 2.0 while introducing three key enhancement dimensions:

### 1. Progressive Scaling Strategy

Rather than simply increasing model size, InternVL 2.5 implements intelligent staged training:

- **Stage 1 (MLP Warmup)**: Only the MLP projector is trained while the vision encoder and language model remain frozen. A dynamic high-resolution training strategy is applied, focusing on robust cross-modal alignment and preparing the model for stable multimodal training.

- **Stage 1.5 (ViT Incremental Learning - Optional)**: The vision encoder undergoes incremental training alongside the MLP projector using targeted data. This phase enhances the encoder's ability to handle rare and specialized domains such as multilingual OCR, mathematical charts, and complex diagrams.

- **Stage 2 (LLM Training)**: The full model trains with the language model unfrozen, enabling deeper multimodal understanding while leveraging the optimized cross-modal alignment from earlier stages.

This staged approach dramatically reduces computational requirements: InternVL 2.5-78B uses only 120 billion training tokens compared to Qwen2-VL's 1.4 trillion tokens—less than one-tenth of the data.

### 2. Improved Training Techniques

InternVL 2.5 incorporates sophisticated training innovations:

- **Random JPEG Compression**: Simulates real-world image degradation, improving robustness to compressed or low-quality images commonly encountered in production systems.

- **Loss Reweighting**: Carefully balances gradient biases to prevent the model from favoring either short or long responses, ensuring balanced generation quality across diverse output lengths.

- **Data Quality Pipeline**: Combines LLM-based quality scoring and rule-based filtering to eliminate problematic samples. Specifically addresses "repetitive generation" issues, a common pathology in multimodal models where the system produces repetitive or hallucinated text.

### 3. Data Quality and Size Optimization

Doubles the dataset size compared to 2.0 but applies strict filtering:

- Excludes anomalous samples with repetitive patterns or hallucinations
- Prioritizes high-quality instruction-tuning data
- Maintains balanced coverage across diverse domains and languages
- Emphasizes quality over pure quantity, enabling efficient training with fewer tokens

The research demonstrates that data quality impacts reasoning tasks far more than dataset size alone, supporting the philosophy that strategic data curation beats brute-force scaling.

### 4. Test-Time Scaling with Chain-of-Thought

Achieves 3.7 percentage point improvement on challenging tasks like MMMU through:

- Chain-of-Thought (CoT) reasoning where the model explicitly shows reasoning steps before final answers
- Majority voting across multiple model generations
- Effective handling of complex multi-disciplinary problems requiring step-by-step reasoning

This approach validates that open-source models can benefit substantially from inference-time techniques, enabling smaller models to approach the performance of larger systems.

## Model Family

InternVL 2.5 spans a wide range of model sizes, from edge-friendly compact variants to powerful flagship models:

### Compact Models (1B-4B)

**InternVL 2.5-1B**
- Parameters: 1 billion
- Vision Encoder: InternViT-300M-448px-V2_5
- Language Model: InternLM 2.5-1B
- Use Case: Edge devices, low-latency inference, resource-constrained environments
- Typical GPU Memory: 4-6GB

**InternVL 2.5-2B**
- Parameters: 2 billion
- Vision Encoder: InternViT-300M-448px-V2_5
- Language Model: InternLM 2.5-1.8B
- Use Case: Mobile applications, edge AI, lightweight servers
- Typical GPU Memory: 6-8GB

**InternVL 2.5-4B**
- Parameters: 4 billion
- Vision Encoder: InternViT-300M-448px-V2_5
- Language Model: InternLM 2.5-4B
- Use Case: Small workstations, intermediate deployment
- Typical GPU Memory: 8-12GB

### Mid-Range Models (8B)

**InternVL 2.5-8B**
- Parameters: 8 billion
- Vision Encoder: InternViT-300M-448px-V2_5
- Language Model: InternLM 2.5-7B
- Use Case: Balanced performance-efficiency trade-off, fine-tuning
- Typical GPU Memory: 16-20GB
- Performance: Excellent for general-purpose vision-language tasks

### Large Models (26B)

**InternVL 2.5-26B**
- Parameters: 26 billion
- Vision Encoder: InternViT-6B-448px-V2_5
- Language Model: InternLM 2.5-20B
- Use Case: Production systems requiring higher accuracy
- Typical GPU Memory: 48-64GB
- Capabilities: Superior document understanding, complex reasoning

### Extra-Large Models (38B, 76B, 78B)

**InternVL 2.5-38B**
- Parameters: 38 billion
- Vision Encoder: InternViT-6B-448px-V2_5
- Language Model: Qwen 2.5-32B
- Use Case: High-performance production deployments
- Typical GPU Memory: 80-100GB

**InternVL 2.5-76B** (Experimental)
- Parameters: 76 billion
- Vision Encoder: InternViT-6B-448px-V2_5
- Language Model: Qwen 2.5-72B
- Status: Research variant

**InternVL 2.5-78B** (Flagship)
- Parameters: 78 billion
- Vision Encoder: InternViT-6B-448px-V2_5
- Language Model: Qwen 2.5-72B (with 6B projector)
- Use Case: Maximum performance scenarios
- Typical GPU Memory: 160-200GB (H100/A100 clusters)
- Performance: First open-source MLLM to exceed 70% on MMMU
- Benchmark Leadership: Competitive with GPT-4o across most benchmarks

### MPO Variants

InternVL 2.5-MPO models are fine-tuned with Mixed Preference Optimization, achieving average improvements of 2 points across all scales on the OpenCompass leaderboard:

- InternVL 2.5-1B-MPO
- InternVL 2.5-2B-MPO
- InternVL 2.5-4B-MPO
- InternVL 2.5-26B-MPO
- InternVL 2.5-38B-MPO

### Quantized Variants

AWQ (Active Weight Quantization) 4-bit quantized variants provide 2.4x faster inference than FP16:

- InternVL 2.5-38B-AWQ
- InternVL 2.5-78B-AWQ (when available)

## Architecture

InternVL 2.5 maintains the proven "ViT-MLP-LLM" paradigm established in earlier versions, validating that architectural stability combined with training innovation delivers superior results.

### Core Components

The architecture consists of three main components working in concert:

1. **Vision Encoder (InternViT)**: Processes images and extracts visual features
2. **Projection Module (MLP)**: Aligns visual features with language model input space
3. **Language Model (InternLM 2.5 or Qwen 2.5)**: Performs reasoning and generates responses

### ViT-MLP-LLM Paradigm

The ViT-MLP-LLM design represents a clean separation of concerns:

- **ViT (Vision Transformer)**: Leverages pre-trained vision foundation models to capture comprehensive visual information
- **MLP (Multi-Layer Perceptron)**: A projection network that transforms visual tokens to be compatible with language model embeddings
- **LLM (Large Language Model)**: Processes both visual and textual information to perform reasoning and generation

This modular design offers significant advantages:

- **Component Reusability**: Pre-trained components from different organizations can be composed together
- **Independent Scaling**: Vision encoders and language models can be scaled independently based on task requirements
- **Efficient Transfer Learning**: A trained vision encoder can be reused across different language models without retraining
- **Flexible Composition**: Different vision encoders can be paired with different language models to create variants

### Model Configuration Examples

**Smaller Models**
```
InternVL 2.5-1B:
- Vision: InternViT-300M + MLP Projector
- Language: InternLM-2.5-1B
```

**Mid-Range Models**
```
InternVL 2.5-8B:
- Vision: InternViT-300M + MLP Projector
- Language: InternLM-2.5-7B
```

**Large Models**
```
InternVL 2.5-26B:
- Vision: InternViT-6B + MLP Projector
- Language: InternLM-2.5-20B
```

**Flagship Models**
```
InternVL 2.5-78B:
- Vision: InternViT-6B + MLP Projector
- Language: Qwen-2.5-72B
```

### Token Efficiency Optimization

A critical architectural innovation is the **pixel unshuffle operation**, which reduces visual token count to one-quarter of the original:

- Standard 448×448 image would normally produce 197 visual tokens (196 tokens + 1 class token)
- After pixel unshuffle: reduced to ~49-64 visual tokens
- Benefit: Maintains visual information density while dramatically reducing computational load
- Result: Enables efficient processing of high-resolution images without proportional increase in compute

This technique allows InternVL 2.5 to maintain strong performance on visual reasoning tasks while using significantly fewer tokens than competitors.

## Vision Encoder: InternViT-6B

The InternViT vision encoder is a cornerstone of InternVL's success, representing years of research into scaling vision transformers effectively.

### Design Philosophy

InternViT-6B is a custom-designed vanilla Vision Transformer (ViT) built to achieve optimal trade-offs between:
- High visual quality perception
- Computational efficiency
- Training stability
- Practical deployability

### Architecture Details

InternViT-6B implements careful hyperparameter optimization:

- **Base Configuration**: Originally 48 transformer blocks with 5.9B parameters
- **Optimized Configuration**: Reduced to 45 blocks (last 3 blocks removed) = 5.5-6B effective parameters
- **Block Depth Options**: Explored 32, 48, 64, and 80-block variants
- **Head Dimension**: Tuned between 64 and 128 for stability
- **MLP Ratio**: Varied between 4 and 8 for efficiency

The optimization found that using the output after the 42nd block (rather than the final block) produced optimal MLLM performance, suggesting that earlier transformer layers capture more useful visual semantics for multimodal tasks.

### Resolution Evolution

**InternViT-6B-224px (Early)**
- Fixed 224×224 pixel resolution
- Limited capability for text and fine details

**InternViT-6B-448px (Current)**
- Increased to 448×448 pixels
- 4x increase in resolution captures finer details
- Enables strong OCR and document understanding
- Supports dynamic resolution tiling strategy

### Training Approach

InternViT-6B undergoes progressive training:

1. **Stage 1 - Vision-Language Contrastive Training**:
   - Large-scale noisy image-text pairs from LAION-en, LAION-multi, LAION-COCO, COYO, Wukong
   - Aligns with a smaller LLM (e.g., LLaMA-7B)
   - Learns robust visual representations through contrastive learning

2. **Stage 1.5 - Incremental Learning (Optional)**:
   - Fine-tunes specifically on rare domains: multilingual OCR, mathematical charts, scientific diagrams
   - Enhances encoder's ability to perceive specialized visual patterns
   - Reusable across downstream models without retraining

3. **Stage 2 - High-Resolution Tuning**:
   - Both vision encoder and MLP are trained together on high-resolution data
   - Incorporates OCR-specific datasets to boost document and text understanding
   - Applies dynamic resolution strategies for variable input sizes

### Key Capabilities

- **Pixel-Level Perception**: Superior ability to capture fine-grained visual details compared to smaller encoders
- **Multilingual OCR**: Excellent performance on text extraction from documents in multiple languages
- **Chart and Diagram Understanding**: Strong capability to interpret scientific charts, graphs, and technical diagrams
- **Long-Form Reasoning**: Maintains visual context needed for multi-step visual reasoning
- **Rare Domain Adaptation**: Through incremental learning, can be specialized for specific visual domains

### Comparison with Alternatives

| Aspect | InternViT-6B | CLIP-ViT | DinO-v2 |
|--------|-------------|---------|---------|
| Parameters | 6B | 300M | ~1B |
| Specialized for MLLM | Yes | No | No |
| OCR Capability | Excellent | Limited | Good |
| Chart Understanding | Excellent | Limited | Good |
| Training Efficiency | High (reusable) | Lower | Lower |

The 6B size represents a significant investment in vision understanding compared to the 300M vision encoders used by LLaVA and early competitors, directly contributing to InternVL's performance superiority.

### Scaling Benefits

Research shows that large vision encoders dramatically reduce language model scaling requirements:

- **InternVL 2.5-78B** with 6B vision encoder achieves better performance than **Qwen2-VL-72B** with 600M vision encoder while using only 1/10 the training tokens
- The 10x difference (6B vs 600M) translates to roughly 10x better token efficiency
- Demonstrates that scaling the vision component is more efficient than scaling the language model for vision-language tasks

## Language Model Backbone: InternLM 2.5

InternVL 2.5 integrates the latest InternLM 2.5 and Qwen 2.5 language models, with InternLM 2.5 providing the backbone for smaller variants.

### Why InternLM 2.5?

**Preservation of Language Capabilities**
- During InternVL 2.0's training, the model experienced degradation in pure language understanding tasks
- InternVL 2.5 addresses this through high-quality open-source data and aggressive low-quality data filtering
- InternLM 2.5's strong pre-training ensures that multimodal training doesn't compromise base language abilities

**Efficient Training**
- InternLM 2.5-7B and InternLM 2.5-20B are optimized for multimodal alignment
- Further pre-trained with general domain data and domain-enhanced corpus
- State-of-the-art performance in language evaluation tasks
- Proven stability in downstream multimodal applications

**Open-Source Philosophy**
- Both InternLM 2.5 and Qwen 2.5 are fully open-source
- Enables full reproducibility and community research
- No proprietary restrictions on model usage or fine-tuning
- Active community support and development

**Strategic Synergy**
- InternLM and InternVL developed by related organizations (both from China's top AI institutions)
- Deep optimization between vision encoder and language model
- Shared understanding of training methodologies and best practices

### InternLM 2.5 Architecture

**Base Model Variants**
- InternLM 2.5-7B: Balance of performance and efficiency
- InternLM 2.5-20B: High-performance base for larger InternVL models
- InternLM 2.5-1B, 4B: Lightweight variants for edge applications

**Key Features**
- 8k context window enables processing of long documents
- Support for multiple documents and images in a single session
- Optimized for instruction following and reasoning
- Strong performance in both English and Chinese
- Reduced hallucination through improved training data curation

### Integration with Qwen 2.5

For larger models, InternVL 2.5 integrates Qwen 2.5:

- **Qwen 2.5-32B**: Used in InternVL 2.5-38B variant
- **Qwen 2.5-72B**: Powers the flagship InternVL 2.5-78B
- Provides extreme reasoning capability
- Superior performance on complex mathematical and logical reasoning
- Strong multilingual capabilities including Chinese, English, and others

The flexibility to combine InternViT-6B with either InternLM or Qwen models demonstrates the composability of the architecture.

## Dynamic Resolution Strategy

InternVL 2.5's dynamic resolution strategy enables efficient handling of images with vastly different aspect ratios and resolutions, a critical requirement for real-world applications.

### Problem Statement

Real-world images vary dramatically:
- Smartphone photos: square or landscape (16:9, 4:3)
- Documents: portrait (8.5:11 or A4)
- Screenshots: variable aspects
- Web images: mixed ratios

Using fixed resolution (e.g., always 448×448) either:
- Loses information (downsampling portrait images)
- Introduces distortion (stretching images to fit)
- Wastes computation (padding with zeros)

### Solution: Tile-Based Dynamic Resolution

InternVL 2.5 divides variable-resolution images into tiles of 448×448 pixels:

**Training Strategy**
- Images divided into 1 to 12 tiles based on aspect ratio and resolution
- Each tile processed independently through the vision encoder
- Visual tokens from all tiles combined for language model processing
- MLP projects all tokens into language model's input space

**Example**
```
Wide landscape image (2000x800):
- Divided into tiles (448x448 each)
- 5-6 tiles horizontal, 2 tiles vertical
- Total: 10-12 tiles processed
- Information preserved, no distortion

Portrait document (800x2000):
- 2 tiles horizontal, 5-6 tiles vertical
- Total: 10-12 tiles processed
- Maintains readability and structure
```

### Inference Scaling

During inference, the strategy can be scaled beyond training:

- **Training**: 1-12 tiles (up to ~4K resolution equivalent)
- **Inference**: Can scale to 40 tiles or beyond
- **Result**: Supports zero-shot scaling to high resolutions
- **Use Cases**: Large documents, high-resolution charts, detailed photos

### Token Efficiency with Pixel Unshuffle

The dynamic resolution strategy becomes practical through pixel unshuffle:

- Standard tiling would require 12 tiles × 196 tokens/tile = ~2,400 tokens per image
- With pixel unshuffle: 12 tiles × 49 tokens/tile = ~600 tokens per image
- 4x reduction in visual tokens while maintaining information density
- Enables processing of high-resolution images in reasonable compute

### Practical Benefits

1. **No Information Loss**: Preserves all visual information regardless of input aspect ratio
2. **Consistent Processing**: Each tile sees uniform 448×448 resolution
3. **Scalable**: Can handle resolutions from small thumbnails to large documents
4. **Efficient**: Pixel unshuffle keeps token count manageable
5. **Flexible**: Inference can use more tiles than training for extreme detail

## Training Pipeline

InternVL 2.5's training pipeline reflects years of refinement in vision-language model development.

### Stage 1: MLP Warmup

**Objective**: Establish robust cross-modal alignment between vision and language modalities

**Configuration**
- Vision Encoder: Frozen (no gradient updates)
- Language Model: Frozen (no gradient updates)
- MLP Projector: Trained (random initialization)

**Training Data**
- Diverse multimodal instruction-tuning data
- Balanced across different vision-language tasks
- Mix of general and domain-specific examples
- Size: ~300K-500K examples

**Training Strategy**
- Dynamic high-resolution training (variable tiling)
- Causal language modeling loss
- Gradient accumulation for stability
- Learning rate schedule: warmup → linear decay

**Duration**: 5,000-10,000 training steps

**Outcome**
- MLP learns to map visual tokens to language model's semantic space
- Prevents catastrophic forgetting of either modality
- Establishes foundation for deeper training

### Stage 1.5: ViT Incremental Learning (Optional)

**Objective**: Enhance vision encoder's ability to handle specialized visual domains

**When Used**
- Required for models with 6B vision encoder (InternViT-6B)
- Optional for 300M vision encoder variants
- Applied when domain-specific improvements are needed

**Specialization Areas**
- **Multilingual OCR**: Text in diverse languages, scripts, and orientations
- **Mathematical Charts**: Scientific plots, equations, technical diagrams
- **Document Understanding**: Layout analysis, table detection, form fields
- **Rare Visual Patterns**: Domain-specific imagery with limited training data

**Configuration**
- Vision Encoder: Trained (full fine-tuning)
- Language Model: Frozen (no gradient updates)
- MLP Projector: Trained (fine-tuning from Stage 1)

**Training Data**
- Specialized datasets targeting identified weak domains
- High-quality examples with detailed annotations
- Size: ~100K-200K domain-specific examples

**Key Insight**
- Once trained, the enhanced vision encoder can be reused across different language models
- Eliminates need to retrain the vision encoder for each downstream model
- Enables efficient scaling to large language models

**Duration**: 2,000-5,000 training steps

### Stage 2: Full Model Training

**Objective**: Optimize all components for downstream multimodal tasks

**Configuration**
- Vision Encoder: Frozen or fine-tuned (depends on variant)
- Language Model: Trained (full fine-tuning)
- MLP Projector: Trained (fine-tuning)

**Training Data**
- Large-scale multimodal instruction-tuning corpus
- Rigorous quality filtering (removes repetitive, hallucinated, or anomalous samples)
- Mix of multiple data sources with careful weighting
- Focus on high-quality responses over quantity

**Training Techniques**
1. **Loss Reweighting**: Balances gradients to prevent model from biasing toward short or long responses
2. **JPEG Compression**: Random compression simulates real-world image degradation
3. **Data Packing**: Efficient GPU utilization through careful batch composition
4. **Dynamic Resolution**: Variable tile counts based on image characteristics

**Data Sources**
- Visual instruction-tuning data (e.g., LLaVA, ShareGPT-4V)
- OCR datasets (LLaVA-ZH, DVQA, SynthDoG)
- Chart understanding (ChartQA, AI2D)
- Document understanding (DocVQA, GeoQA+)
- General VQA and visual reasoning
- Custom high-quality instruction data

**Duration**: 20,000-50,000 training steps

**Convergence**
- Model trained until validation metrics plateau
- Early stopping based on benchmark performance
- Final checkpoint represents optimal performance

### Key Training Parameters

**Optimization**
- Optimizer: AdamW with weight decay
- Learning Rate: Varies by stage (higher for MLP, lower for full training)
- Batch Size: 512-2048 depending on available GPU memory
- Mixed Precision: BF16 training (F32 for stability-critical ops)

**Hardware**
- Compute: Multiple A100/H100 GPUs
- Distributed Training: PyTorch Distributed Data Parallel
- Training Duration: 5-14 days depending on model size and data scale

**Monitoring**
- Validation on multiple benchmarks (MMBench, MMMU, DocVQA, etc.)
- Loss tracking at frequent intervals
- Gradient statistics for anomaly detection
- Inference evaluation every 5,000 steps

## Performance Benchmarks

InternVL 2.5 demonstrates competitive performance across a diverse range of vision-language understanding benchmarks, achieving a critical milestone with over 70% accuracy on MMMU.

### MMMU: The 70% Breakthrough

**Benchmark Details**
- Dataset: 11,500 multimodal questions from college exams and textbooks
- Disciplines: 6 core areas (Art & Design, Business, Science, Health & Medicine, Humanities & Social Science, Tech & Engineering)
- Subjects: 30 across the 6 disciplines
- Subfields: 183 total
- Question Types: 30 different image types (charts, diagrams, tables, music sheets, chemical structures, etc.)
- Difficulty: College/university level
- Significance: Benchmark for complex multimodal reasoning

**InternVL 2.5 Results**
- InternVL 2.5-78B: 70.1% accuracy (with Chain-of-Thought)
- InternVL 2.5-26B: ~67-68% accuracy
- Improvement over 2.0: +7.4 percentage points vs InternVL 2-Llama3-76B (62.7%)
- First open-source MLLM to exceed 70%
- Approaches proprietary model performance

**Chain-of-Thought Impact**
- Direct response: 66.4% (InternVL 2.5-78B)
- With CoT reasoning: 70.1% (same model)
- Improvement: 3.7 percentage points
- Demonstrates test-time scaling effectiveness

### General Vision-Language Benchmarks

**MMBench** (Comprehensive VQA)
- InternVL 2.5-78B: ~83-84%
- Covers diverse visual understanding tasks
- State-of-the-art among open-source models

**WildVision** (User Experience Assessment)
- InternVL 2.5-78B: 71.4%
- GPT-4o: 80.6%
- Gap indicates room for improvement in real-world user satisfaction

**MMIU** (General Understanding)
- InternVL 2.5-78B: 55.8%
- GPT-4o: 55.7%
- Near-parity with leading commercial model

### Document and Text Understanding

**DocVQA** (Document Question Answering)
- InternVL 2.5-26B: ~88-90%
- InternVL 2.5-78B: ~91-92%
- State-of-the-art performance

**ChartQA** (Chart Interpretation)
- InternVL 2.5-26B: ~87-88%
- InternVL 2.5-78B: ~89-90%
- Excellent understanding of data visualization

**InfographicVQA** (Infographic Comprehension)
- Competitive with or exceeding GPT-4o on many examples
- Strong layout understanding and information extraction

**OCRBench** (Scene Text Recognition)
- InternVL 2.5: Competitive performance
- Multilingual text recognition capability
- Handles diverse fonts, orientations, and degradation

### Mathematical and Scientific Reasoning

**MathVista** (Mathematical Reasoning)
- InternVL 2.0: 66.3% (surpasses commercial models)
- InternVL 2.5: Expected ~67-68%
- Strong geometry and diagram interpretation

**Science Benchmarks**
- Excellent performance on scientific diagrams and equations
- Strong in biology, chemistry, physics domains

### Hallucination and Safety

**HallusionBench** (Hallucination Detection)
- InternVL 2.5-78B: 57.4%
- Qwen2-VL-72B: 58.1%
- GPT-4o: 55.0%
- Shows competitive resistance to hallucination

### Multilingual Capabilities

**Chinese Language Understanding**
- Outperforms GPT-4V on Chinese-specific tasks
- Excellent performance on Chinese OCR and scene understanding
- Bilingual instruction-following

**COCO Caption** (Image Captioning)
- InternVL 2.5: Competitive captioning quality
- Both English and Chinese caption generation

### Multi-Image Understanding

**MMStar** (Multi-Image Benchmark)
- Strong performance on multi-image reasoning
- Handles spatial relationships between multiple images
- Supports image grounding tasks

### Comparison with GPT-4o

| Benchmark | InternVL 2.5-78B | GPT-4o | Gap |
|-----------|-----------------|--------|-----|
| MMMU | 70.1% | ~71% | Near parity |
| MMBench | ~83-84% | ~86% | -2-3 pts |
| MMIU | 55.8% | 55.7% | Ahead |
| MMT-Bench | 70.8% | 65.4% | Ahead |
| HallusionBench | 57.4% | 55.0% | Ahead |
| WildVision | 71.4% | 80.6% | -9 pts |
| ChartQA | ~89-90% | ~88-89% | Competitive |
| DocVQA | ~91-92% | ~92-94% | Slightly behind |

**Summary**: InternVL 2.5-78B achieves near-parity with GPT-4o on reasoning-heavy tasks (MMMU, MMT-Bench) and outperforms on some benchmarks, while remaining slightly behind on user experience (WildVision) and complex document understanding. The gap continues to narrow with each version.

### Performance Scaling with Model Size

**Trend Analysis**
- Performance improvements don't scale linearly with parameter count
- Smaller models (1B-8B) show strong scaling efficiency
- Large models show diminishing returns but achieve higher absolute performance
- Data quality becomes more important at larger scales

**Recommendations by Use Case**
- **Edge/Mobile (1B-4B)**: ~40-50% of flagship performance, suitable for simple tasks
- **General Purpose (8B-26B)**: ~70-85% of flagship performance, balanced deployment
- **High-Accuracy Production (38B-78B)**: ~90-100% performance, maximum capability

## Chinese and English Bilingual Capabilities

InternVL 2.5 maintains and extends the bilingual excellence established in earlier versions, providing first-class support for both English and Chinese vision-language understanding.

### Chinese Language Support

**Comprehensive Coverage**
- Mandarin Chinese instruction-following
- Simplified and traditional characters
- Handwritten text recognition
- Vertical and horizontal text orientations

**High-Quality Bilingual Dataset**
- Carefully curated English-Chinese question-answer pairs
- Covers common scenes, document images, and specialized domains
- Balanced representation preventing English-biased training

**Superior Performance**
- Often outperforms GPT-4V on Chinese-specific tasks
- Excellent Chinese scene text understanding
- Strong performance on Chinese OCR benchmarks
- Cultural knowledge including Chinese traditions and references

### English Language Support

**Extensive Capability**
- Fluent English instruction-following
- Nuanced understanding of idiomatic English
- Strong English scientific and technical understanding
- Cultural knowledge including English-speaking contexts

**Diverse Data Sources**
- Standard English vision-language datasets
- Academic papers and technical documentation
- Web-sourced high-quality examples
- Professional domain examples

### Bilingual Benchmarks

**Language Preservation**
- Both English and Chinese instruction-following preserved during MLLM training
- No degradation in language capabilities from 1.5 to 2.5 versions
- Strong pure language understanding without visual context

**Practical Examples**
1. **OCR Task**: "Extract all text from this document" → Handles mixed English-Chinese text seamlessly
2. **Document Analysis**: "Summarize this report" → Works across languages in same document
3. **Chart Interpretation**: "What does this chart show?" → Understands labels, legends, and axis text in either language
4. **Math Problems**: "Solve this equation" → Works whether problem is in English or Chinese

### Data Composition

The high-quality bilingual dataset includes:

1. **Scene Understanding**
   - Street scenes with mixed-language signs
   - Indoor scenes with English and Chinese text
   - Natural environments

2. **Document Images**
   - English academic papers
   - Chinese news articles and reports
   - Bilingual technical documentation
   - Forms and official documents

3. **OCR-Specific Training**
   - Natural scene text in both languages
   - Document OCR with mixed languages
   - Handwritten Chinese character recognition
   - Mathematical notation and symbols

4. **Specialized Domains**
   - Medical documents in English and Chinese
   - Legal documents
   - Technical specifications
   - Business reports

### Practical Applications

**Document Processing**
- Automatic translation of extracted text
- Cross-language document understanding
- Bilingual form filling and data extraction

**International E-Commerce**
- Product descriptions in multiple languages
- Instruction manual understanding
- Customer service applications

**Healthcare**
- Medical record analysis (English and Chinese)
- Prescription and lab result interpretation
- Multilingual patient communication

**Education**
- Textbook and exam questions in both languages
- Bilingual student support
- Cross-language knowledge transfer

## Multi-Image Support

InternVL 2.5's ability to process and reason about multiple images simultaneously enables sophisticated applications impossible with single-image models.

### Architecture Support

**Multiple Image Handling**
- Process 2-8 images in a single inference call
- Maintain spatial relationships between images
- Track object correspondences across images
- Reason about temporal sequences

**Context Window**
- 8K context window supports long multimodal context
- Sufficient for several high-resolution images plus detailed text
- Enables complex multi-image narratives

### Use Cases

**Comparison Tasks**
```
User: "Compare these two charts. Which shows higher growth?"
System: Analyzes both images, extracts data, performs comparison
```

**Sequential Understanding**
```
User: "Walk me through this process step by step"
System: Examines images showing different steps in sequence
```

**Relationship Inference**
```
User: "What is the relationship between these photos?"
System: Identifies shared objects, settings, or narrative connections
```

**Document Summarization**
```
User: "Summarize this entire report"
System: Processes multiple pages simultaneously, synthesizes information
```

### Technical Implementation

**Token Efficiency**
- Multiple images processed through independent vision encoder passes
- Pixel unshuffle applied to each image independently
- All visual tokens concatenated before language model
- Context window adequately sized for multi-image processing

**Attention Mechanisms**
- Language model attends to visual tokens from all images
- Can establish cross-image references
- Maintains spatial and semantic relationships

**Practical Limitations**
- Processing time scales with image count
- Very high-resolution images may require reducing tile count per image
- Trade-off between image quality and context window usage

## Comparison with GPT-4o

InternVL 2.5 represents the first time an open-source MLLM approaches parity with OpenAI's GPT-4o across most benchmarks, marking a significant inflection point in open-source AI.

### Strengths of InternVL 2.5 Relative to GPT-4o

**Reasoning Performance**
- **MMMU (70.1% vs ~71%)**: Near-parity on college-level multimodal reasoning
- **MMT-Bench (70.8% vs 65.4%)**: Outperforms on complex reasoning
- **MMIU (55.8% vs 55.7%)**: Virtually identical on general understanding
- **HallusionBench (57.4% vs 55.0%)**: Better resistance to hallucination

**Document Understanding**
- **DocVQA**: Competitive on document question-answering
- **ChartQA**: Strong chart interpretation, often exceeds GPT-4o
- **OCR Tasks**: Excellent multilingual text extraction
- **Form Understanding**: Strong form field detection and extraction

**Efficiency**
- Open-source: Can be self-hosted without API costs
- Reproducible: Full model transparency
- Customizable: Fine-tuning enabled for domain specialization
- Lower latency: No API call round-trips

**Bilingual Excellence**
- Often superior to GPT-4o on Chinese-specific tasks
- Bilingual instruction-following without handicapping either language
- Cultural knowledge for both English and Chinese contexts

### Weaknesses Relative to GPT-4o

**User Experience**
- **WildVision (71.4% vs 80.6%)**: Noticeable gap in user-perceived quality
- Real-world application performance may vary
- GPT-4o demonstrates broader capability in diverse scenarios

**Complex Multi-Image Reasoning**
- **BLINK, MuirBench**: ~5-point gap vs GPT-4o
- GPT-4o shows stronger inter-image relationship understanding
- Would benefit from more multi-image training data

**Edge Cases and Robustness**
- GPT-4o shows greater robustness to adversarial or unusual inputs
- Better handling of ambiguous or misleading prompts
- More consistent across diverse use cases

**Video Understanding**
- GPT-4o (newer versions) superior video processing
- InternVL 2.5 focuses primarily on still images and multi-image

### Technical Differences

| Aspect | InternVL 2.5-78B | GPT-4o |
|--------|-----------------|--------|
| Architecture | ViT-MLP-LLM | Proprietary |
| Vision Encoder | InternViT-6B | Proprietary |
| Language Model | Qwen 2.5-72B | Proprietary |
| Open Source | Yes | No |
| Model Size | 78B | Unknown (~175B estimated) |
| Training Data | ~120B tokens | ~1.6T+ tokens |
| Quantization | Full precision available | Not available |
| Fine-tuning | Supported | API-only |
| Deployment | Self-hosted or API | API-only |

### Strategic Positioning

**When to Choose InternVL 2.5-78B**
- Cost-sensitive applications (self-hosted saves API calls)
- Need for model customization or fine-tuning
- Privacy-critical applications (no external API calls)
- Research and reproducibility requirements
- Multilingual Chinese-English applications
- Document and OCR-heavy workloads

**When to Choose GPT-4o**
- Maximum accuracy required (slight edge on some benchmarks)
- Broad user-perceived quality matters (WildVision gap)
- Multi-image reasoning is critical (BLINK, MuirBench)
- Video understanding required
- Minimal setup complexity desired (fully hosted service)
- Latest proprietary techniques

### Convergence Trajectory

The performance gap between open-source and proprietary models is narrowing:

- **2023**: Open-source was 15-20 points behind on most benchmarks
- **2024 (early)**: Gap narrowed to 8-12 points
- **2024 (late, InternVL 2.5)**: Gap now 0-5 points on reasoning tasks
- **2025+**: Expectation of further convergence with InternVL 3.0

This trajectory reflects the accelerating pace of open-source AI development and the competitive pressure from models like LLaVA, Qwen-VL, and others.

## Comparison with LLaVA

InternVL 2.5 builds upon many concepts established by LLaVA but introduces significant improvements that explain its superior performance.

### Architectural Differences

**Vision Encoder**
- **LLaVA**: CLIP-ViT-L/14 with 304M parameters
- **InternVL 2.5**: InternViT-6B or InternViT-300M
- **Impact**: InternVL's 6B encoder captures 20x more visual detail
- This single change drives much of the performance improvement

**Projection Layer**
- **LLaVA**: Simple linear projection + additional layers
- **InternVL 2.5**: Carefully designed MLP with pixel unshuffle
- **Impact**: More sophisticated alignment reduces information loss

**Language Model**
- **LLaVA**: Llama 2-7B, 13B, or larger
- **InternVL 2.5**: InternLM 2.5 or Qwen 2.5
- **Impact**: State-of-the-art language models improve reasoning

### Data and Training Differences

**Training Approach**
- **LLaVA**: Single-stage fine-tuning on ~600K examples
- **InternVL 2.5**: Three-stage progressive training
- **Data Size**: LLaVA uses smaller dataset; InternVL 2.5 uses larger, higher-quality corpus
- **Quality Focus**: InternVL 2.5's strict filtering removes hallucinations

**Instruction Tuning**
- **LLaVA**: Leverages GPT-4-generated annotations
- **InternVL 2.5**: Combines multiple high-quality datasets with human-curated filtering
- **Multilingual**: InternVL 2.5 includes Chinese; LLaVA focuses on English

### Performance Comparison

| Benchmark | InternVL 2.5-26B | LLaVA-1.5-7B | InternVL 2.5-78B | LLaVA-NeXT-34B |
|-----------|-----------------|--------------|-----------------|----------------|
| MMBench | ~82-83% | ~66% | ~83-84% | ~71% |
| MMMU | ~67-68% | ~29% | 70.1% | ~44% |
| DocVQA | ~88-90% | ~60% | ~91-92% | ~71% |
| ChartQA | ~87-88% | ~57% | ~89-90% | ~67% |
| MMStar | Strong | Limited | Excellent | Good |

**Summary**: Even smaller InternVL 2.5 models outperform larger LLaVA variants, with the gap widening significantly on reasoning-heavy benchmarks like MMMU.

### Design Philosophy Differences

**LLaVA Approach**
- Pragmatic simplicity: minimal changes to established components
- Reuse CLIP encoders despite suboptimal fit for MMOLMs
- Efficient training with smaller datasets
- Single-stage training pipeline

**InternVL Approach**
- Custom vision encoder designed specifically for MMOLMs
- Specialized training methodology with multiple stages
- Data quality over quantity
- Continuous improvement through iterative research

### When to Use Each

**Use LLaVA When**
- Minimal compute is available for inference
- Fine-tuning on consumer GPUs
- Established codebase compatibility needed
- Research into efficient multimodal architectures

**Use InternVL 2.5 When**
- Maximum accuracy is the priority
- Access to adequate compute resources
- Chinese language support needed
- Document and OCR tasks are important
- Willing to invest in custom implementation

### Learning from Each Other

Modern MLLM development benefits from both approaches:
- **From LLaVA**: Simplicity, training efficiency, accessibility
- **From InternVL**: Vision encoder importance, progressive training, data quality focus

The field continues to evolve, with both models pushing boundaries and establishing new baselines.

## Use Cases and Applications

InternVL 2.5's versatile capabilities enable a wide range of practical applications:

### Document Understanding and Processing

**Automated Document Analysis**
- Extract structured information from forms, applications, and questionnaires
- Classify documents by type (invoices, contracts, resumes, etc.)
- Detect missing or incomplete sections
- Flag documents requiring human review
- Extract key information (dates, amounts, signatories)

**Receipt and Invoice Processing**
- Automatic receipt scanning and expense classification
- Invoice line-item extraction
- Vendor identification
- Date and amount detection
- Integration with accounting systems

**Medical and Healthcare**
- Lab result interpretation and normalization
- Medical form filling and data extraction
- Prescription analysis
- Patient history documents
- Insurance claim processing

**Legal Document Analysis**
- Contract clause identification and extraction
- Risk assessment of legal documents
- Deadline and obligation detection
- Defined term extraction
- Inconsistency detection across documents

### Optical Character Recognition (OCR)

**Multilingual Text Extraction**
- Extract text from images containing multiple languages
- Preserve layout information for reconstructed documents
- Handle diverse fonts, sizes, and orientations
- Recognize handwritten text (limited capability)

**Scene Text Understanding**
- Street sign recognition and understanding
- Product label interpretation
- Nameplate and placard reading
- Sign language recognition (text-related)

**Document Digitization**
- Convert paper documents to searchable digital format
- Preserve formatting and structure
- Enable full-text search on scanned documents
- Archive historical documents

### Data Visualization and Chart Understanding

**Automated Chart Analysis**
- Extract numerical data from charts and graphs
- Identify trends and patterns
- Compare data across multiple visualizations
- Generate textual descriptions of data
- Detect anomalies and outliers

**Scientific Data Interpretation**
- Analyze scientific plots and graphs
- Extract experimental results
- Interpret technical diagrams
- Understand molecular structures
- Read spectrograms and waveforms

**Business Intelligence**
- Dashboard and report understanding
- KPI extraction from visualizations
- Comparative analysis of metrics
- Trend identification
- Forecasting support

### General Visual Question Answering (VQA)

**Image Captioning and Description**
- Generate natural language descriptions of images
- Support both English and Chinese captions
- Adapt description level to context
- Identify and describe objects, scenes, and actions

**Visual Search and Retrieval**
- Use images as queries to find similar content
- Describe what you're looking for in images
- Find specific objects, scenes, or concepts
- Cross-modal search (text-to-image, image-to-image)

**Content Moderation**
- Identify inappropriate or sensitive content
- Context-aware classification
- Explain why content was flagged
- Assist human reviewers

### Accessibility and Assistive Technology

**Image Description for Blind/Low Vision Users**
- Detailed scene descriptions
- Object identification and localization
- Action and activity understanding
- Emotional content analysis
- Cultural context explanation

**Real-Time Assistance**
- Navigation support through visual understanding
- Menu and sign reading
- Object identification for daily tasks
- Document comprehension assistance

### E-Commerce and Product Analysis

**Product Image Understanding**
- Automatic product categorization
- Feature detection (color, size, material)
- Damage or defect identification
- Similar product recommendation
- Product description generation

**Quality Control**
- Manufacturing defect detection
- Packaging integrity verification
- Labeling accuracy checks
- Product specification compliance

**Customer Service**
- Answer questions about product images
- Provide recommendations based on images
- Process customer-uploaded photos
- Troubleshoot based on images

### Educational Applications

**Homework Help and Tutoring**
- Solve math problems from photos
- Explain solution steps
- Identify knowledge gaps
- Suggest similar practice problems

**Exam Preparation**
- Practice on multimodal questions similar to MMMU
- Understand complex diagrams and charts
- Learn from visual explanations
- Build visual literacy

**Research and Learning**
- Understand scientific papers with diagrams
- Learn from infographics and visual content
- Extract and synthesize information from multiple images
- Generate summaries of visual content

### Specialized Domain Applications

**Autonomous Driving** (related research)
- Scene understanding and interpretation
- Obstacle and hazard detection
- Road sign and signal recognition
- Traffic situation analysis

**Medical Imaging** (research direction)
- X-ray and CT scan interpretation (limited by training data)
- Pathology slide analysis
- Radiology report generation
- Abnormality detection (with proper training)

**Geological and Environmental** (research direction)
- Satellite image interpretation
- Land use classification
- Disaster impact assessment
- Environmental monitoring

## Implementation and Deployment

InternVL 2.5 provides multiple pathways for integration into applications, from simple API-based access to complete self-hosted solutions.

### HuggingFace Integration

**Direct Model Access**
All InternVL 2.5 models available on Hugging Face Model Hub:
- OpenGVLab/InternVL2_5-1B
- OpenGVLab/InternVL2_5-2B
- OpenGVLab/InternVL2_5-4B
- OpenGVLab/InternVL2_5-8B
- OpenGVLab/InternVL2_5-26B
- OpenGVLab/InternVL2_5-38B
- OpenGVLab/InternVL2_5-78B
- MPO variants (-MPO suffix)
- AWQ quantized variants (-AWQ suffix)

**Using Transformers Library**
```python
from transformers import AutoTokenizer, AutoModel
import torch

model_id = "OpenGVLab/InternVL2_5-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
```

**Requirements**
- transformers >= 4.37.2
- BF16 mixed precision recommended
- CUDA >= 11.8 for GPU inference

### Fine-Tuning Frameworks

**SWIFT Framework**
- Official fine-tuning support
- Efficient parameter-efficient methods (LoRA, QLoRA)
- Distributed training capabilities
- Custom dataset support

**XTurner**
- Alternative training framework
- Supports instruction tuning
- Efficient memory usage
- Custom task adaptation

**LLaMA-Factory**
- General-purpose LLM/MLLM training
- Supports InternVL models
- Advanced training strategies
- Web UI for configuration

### Inference Optimization

**LMDeploy**
- Specialized for VLM inference
- AWQ 4-bit quantization (2.4x speedup over FP16)
- Efficient batch inference
- API server deployment

```bash
lmdeploy serve api_server \
  OpenGVLab/InternVL2_5-8B \
  --server-port 23333 \
  --model-format awq  # if using quantized variant
```

**vLLM Integration**
- High-throughput inference
- Batch processing
- KV cache optimization
- OpenAI-compatible API

**TGI (Text Generation Inference)**
- Docker-based deployment
- Load balancing support
- Automatic quantization
- Streaming response support

### Production Deployment

**Cloud Platforms**
- **HuggingFace Inference API**: Simple hosted endpoint
- **AWS SageMaker**: Managed inference endpoints
- **Azure ML**: Enterprise deployment
- **Google Cloud Vertex AI**: Scalable endpoints

**On-Premise Solutions**
- Docker containerization
- Kubernetes orchestration
- Load balancing
- Monitoring and alerting

**API Server Example**
Using LMDeploy with OpenAI-compatible interface:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:23333/v1",
    api_key="fake-key"
)

response = client.chat.completions.create(
    model="internvl",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "image.jpg"}}
            ]
        }
    ]
)
```

### Quantization Options

**AWQ Quantization (Recommended)**
- 4-bit weight quantization
- 2.4x inference speedup
- Minimal accuracy loss
- Lower VRAM requirements
- Available for multiple model sizes

**GPTQ Quantization**
- Alternative 4-bit method
- Good accuracy preservation
- Community-supported variants
- Slightly slower than AWQ

**BNB Quantization**
- WARNING: BNB 4-bit causes serious issues with InternViT-6B
- Results in nonsensical outputs and image incomprehension
- Should be avoided
- Use only if no alternatives available

### Hardware Requirements

**Minimum Configuration**
- InternVL 2.5-1B: 4GB VRAM (8GB recommended)
- InternVL 2.5-8B: 16GB VRAM
- InternVL 2.5-26B: 48GB VRAM
- InternVL 2.5-78B: 160-200GB VRAM

**Recommended Platforms**
- NVIDIA A100 (40GB): Supports models up to 26B efficiently
- NVIDIA H100 (80GB): Supports up to 78B models
- NVIDIA RTX 4090 (24GB): Good for models up to 4B
- Multiple GPUs: Distributed inference for large models

**CPU Inference**
- Possible but slow (10-20x slower than GPU)
- Suitable only for very small models (1B-2B)
- Long latency (several seconds per query)

## Licensing and Availability

InternVL 2.5 maintains commitment to open-source accessibility with clear licensing:

### Model Licensing

**Apache 2.0 License**
- Most InternVL 2.5 models released under Apache 2.0
- Permissive open-source license
- Commercial use permitted
- Modification and distribution allowed
- Minimal attribution requirements

**License Variants**
- Base models: Apache 2.0
- Some variants: May have additional conditions based on language model component
- Qwen 2.5-based models: Subject to Qwen licensing
- InternLM 2.5-based models: Subject to InternLM licensing

### Code Availability

**GitHub Repository**
- OpenGVLab/InternVL: Complete source code
- CVPR 2024 Oral paper behind research publication
- Active development and maintenance
- Community contributions welcomed

**Documentation**
- Comprehensive readthedocs documentation
- Usage examples and tutorials
- Training scripts and configurations
- Benchmark evaluation code

### Component Licensing

**Vision Encoder (InternViT-6B)**
- Independently licensed
- Can be used separately from InternVL
- Published as InternViT-6B standalone

**Language Models**
- **InternLM 2.5**: Commercially usable
- **Qwen 2.5**: Subject to Qwen licensing terms
- Both are open-source
- Derivative work licenses apply

### Academic vs. Commercial Use

**Academic Research**
- Full freedom under Apache 2.0
- Publication of research findings encouraged
- Citations to InternVL papers appreciated
- Pre-prints and papers available on arXiv

**Commercial Applications**
- Permitted under Apache 2.0 license
- No licensing fees
- No usage restrictions
- Reselling as-is not typical
- Selling fine-tuned versions possible

**Enterprise Deployment**
- No special licensing needed
- Standard Apache 2.0 terms apply
- Companies may implement internal fine-tuning
- Commercial support available through related organizations

## Limitations and Challenges

Despite strong performance, InternVL 2.5 has identifiable limitations important to understand for appropriate deployment:

### Reasoning and Understanding

**Chain-of-Thought Limitations**
- While CoT improves MMMU performance by 3.7%, most open-source MLLMs underperform with CoT relative to other model families
- Potential for circular reasoning or error amplification
- Longer reasoning chains don't always improve accuracy
- Domain-specific reasoning may require fine-tuning

**Complex Multi-Step Reasoning**
- Struggles with very long reasoning chains (5+ steps)
- May lose track of intermediate results
- Accumulates errors in sequential reasoning
- Performs better on single-step reasoning

### Knowledge and Factual Accuracy

**Hallucination Tendency**
- Model may generate plausible-sounding but incorrect information
- Particularly in domains outside training data
- Can confidently state false facts
- HallusionBench performance (57.4%) shows room for improvement

**Knowledge Cutoff**
- Training data frozen at specific point in time
- No knowledge of very recent events
- Cannot access real-time information
- Lacks awareness of developments after training cutoff

**Domain Expertise Limitations**
- Strong on general domains covered in training
- Weaker on specialized technical domains
- Medical interpretation limited without fine-tuning
- Legal analysis requires domain-specific fine-tuning for reliability

### Multi-Image Reasoning

**Cross-Image Relationships**
- Gap of ~5 points vs. GPT-4o on BLINK and MuirBench
- Struggles to establish complex relationships between multiple images
- Limited understanding of temporal sequences
- Spatial relationships sometimes misunderstood

**Context Window Constraints**
- 8K context window limits number of high-resolution images
- Very high-resolution images consume more tokens
- Trade-off between image quality and count
- Larger images receive more visual tokens

### Quantization Issues

**BNB 4-bit Quantization**
- WARNING: Causes serious degradation with InternViT-6B
- Results in nonsensical outputs
- Model cannot understand images properly
- Should be strictly avoided
- Use AWQ instead for 4-bit quantization

**Other Quantization Methods**
- GPTQ acceptable with minor accuracy loss
- INT8 quantization maintains good accuracy
- Lower quantization levels (2-bit) not recommended
- Quantization errors accumulate in vision encoder

### Safety and Ethical Concerns

**Unexpected Outputs**
- Model may generate biased, discriminatory, or harmful content
- Probabilistic generation can produce adversarial outputs
- No guaranteed safety despite safety training
- Requires content filtering in production

**Bias in Training Data**
- Reflects biases present in training datasets
- May perpetuate stereotypes
- Different language models bring different biases
- Fine-tuning data composition affects output bias

**Misuse Potential**
- Can potentially be misused for disinformation
- Generated images descriptions could be manipulated
- Unauthorized surveillance applications possible
- Proper ethical guidelines recommended

### Computational Requirements

**Large Model Sizes**
- 78B model requires GPUs with 160-200GB VRAM
- High electricity consumption during training and inference
- Environmental impact of large-scale deployment
- Cost barrier for resource-limited organizations

**Training Resource Intensity**
- Fine-tuning still requires substantial compute
- Multiple A100/H100 GPUs needed for reasonable speed
- Training time measured in days even on high-end hardware
- Inference latency still measured in seconds for large models

**Scalability Challenges**
- Inference latency increases with batch size
- Distributed inference across multiple GPUs complex
- Real-time applications challenging for largest models
- Smaller models needed for latency-critical applications

### Language and Cultural Biases

**English-Centric Training**
- Despite bilingual training, may favor English in multilingual contexts
- English examples often dominate benchmarks
- Western cultural references more prominent
- Non-English-speaking context understanding limited

**Underrepresented Languages**
- Chinese language excellent; other Asian languages limited
- African and low-resource languages underrepresented
- Right-to-left languages (Arabic, Hebrew) have limited support
- Language-specific idioms and cultural nuances sometimes missed

### Technical Limitations

**Image Encoding Artifacts**
- JPEG compression in training introduces artifacts
- Very high-quality uncompressed images processed differently
- Extreme aspect ratios may cause issues
- Rotated or mirrored images require careful handling

**Fine Details vs. Global Understanding**
- Strong on global scene understanding
- Weaker on extremely fine details (single pixels)
- Millimeter-level precision not achievable
- Microscopy-level detail insufficient

**Video and Temporal Understanding**
- Limited to still image processing
- Cannot understand motion or temporal sequences
- Video frame understanding requires processing each frame separately
- Temporal relationships between frames not explicitly modeled

## Future Directions

The InternVL family continues to evolve with planned improvements and new research directions:

### InternVL 2.5-MPO Enhancement

**Mixed Preference Optimization (MPO)**
Released December 20, 2024, this variant applies advanced preference optimization:

- **Algorithm**: MPO enables models to learn three dimensions simultaneously:
  1. Relative preference between response pairs
  2. Absolute quality of individual responses
  3. Process for generating preferred responses

- **Performance Improvement**: Average +2 points across all model scales on OpenCompass leaderboard

- **Available Variants**:
  - InternVL 2.5-1B-MPO
  - InternVL 2.5-2B-MPO
  - InternVL 2.5-4B-MPO
  - InternVL 2.5-26B-MPO
  - InternVL 2.5-38B-MPO

- **Impact**: Shows that post-training optimization remains a frontier for MLLM improvement

### InternVL 3.0 (April 2025)

Announced and partially released, InternVL 3.0 represents the next major generation:

**Key Design Innovations**

1. **Variable Visual Position Encoding**
   - More flexible visual token positioning
   - Better handling of arbitrary image layouts
   - Improved support for complex multi-image arrangements

2. **Native Multimodal Pre-Training**
   - Joint vision-language pre-training from the start
   - Not simply combining pre-trained components
   - Deeper multimodal alignment

3. **Continued Mixed Preference Optimization**
   - Full integration of MPO into training pipeline
   - Extended beyond post-training to main training
   - Stronger reasoning capabilities

4. **Multimodal Test-Time Scaling**
   - Advanced inference-time optimization
   - Majority voting across diverse generations
   - Enhanced reasoning through sampling

**Expected Performance**
- InternVL3-78B achieves state-of-the-art performance
- Superior perception and reasoning compared to 2.5
- Extended capabilities: tool usage, GUI agents, 3D vision

**Expanded Capabilities**
- **Tool Usage**: Integration with external tools and APIs
- **GUI Agents**: Understanding and interacting with graphical interfaces
- **Industrial Image Analysis**: Manufacturing and quality control applications
- **3D Vision Perception**: Understanding 3D scenes and objects
- Extended multimodal capabilities beyond standard vision-language

### Research Directions Being Explored

**Long-Context Understanding**
- Extend context window beyond 8K tokens
- Better handling of very long documents
- Video understanding as extended image sequences
- Persistent memory across multiple interactions

**Domain-Specific Fine-Tuning**
- Pre-built variants for medical imaging
- Legal document specialist models
- Scientific paper understanding models
- Industry-specific applications

**Efficiency Improvements**
- Distilled smaller models (Sub-1B parameters)
- More aggressive quantization without accuracy loss
- Dynamic inference (vary computation by input complexity)
- Speculative decoding for faster generation

**Safety and Alignment**
- Improved instruction following
- Better refusal of harmful requests
- Reduced hallucination rate
- Constitutional AI approaches for alignment

**Multilingual Excellence**
- Better support for languages beyond Chinese
- African language support
- Low-resource language adaptation
- Code-mixing (mixed language) support

**Video and Temporal Understanding**
- Frame-level video understanding
- Temporal relationship modeling
- Activity recognition from video
- Instructional video comprehension

## Sources and Further Reading

### Official Documentation and Blogs

- [InternVL 2.5 Blog Post](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/)
- [InternVL 2.5 Documentation](https://internvl.readthedocs.io/en/latest/internvl2.5/introduction.html)
- [InternVL 2.5-MPO Blog Post](https://internvl.github.io/blog/2024-12-20-InternVL-2.5-MPO/)
- [InternVL 3.0 Blog Post](https://internvl.github.io/blog/2025-04-11-InternVL-3.0/)

### Research Papers

- [Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling (arXiv:2412.05271)](https://arxiv.org/abs/2412.05271)
- [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks (CVPR 2024 Oral)](https://arxiv.org/abs/2312.14238)
- [InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)

### Code and Models

- [OpenGVLab/InternVL GitHub Repository](https://github.com/OpenGVLab/InternVL)
- [InternVL Models on Hugging Face](https://huggingface.co/OpenGVLab)
- [InternVL 2.5-78B on Hugging Face](https://huggingface.co/OpenGVLab/InternVL2_5-78B)
- [InternViT-6B-448px-V2.5 on Hugging Face](https://huggingface.co/OpenGVLab/InternViT-6B-448px-V2_5)

### Related Resources

- [MMMU Benchmark](https://mmmu-benchmark.github.io/)
- [OpenGVLab on GitHub](https://github.com/OpenGVLab)
- [OpenGVLab on Hugging Face](https://huggingface.co/OpenGVLab)
- [LMDeploy Toolkit](https://github.com/InternLM/lmdeploy)
- [InternLM Project](https://github.com/InternLM/InternLM)

---

**Document Summary**

InternVL 2.5 represents a watershed moment for open-source multimodal AI, achieving over 70% accuracy on the challenging MMMU benchmark and approaching parity with commercial models like GPT-4o across most evaluation metrics. Through careful attention to vision encoder scaling (InternViT-6B), progressive training strategies, and data quality over quantity, the model family demonstrates that open-source development can compete with the most advanced proprietary systems.

The architecture's modularity, the comprehensive model family from 1B to 78B parameters, bilingual English-Chinese excellence, and extensive real-world applicability across document understanding, OCR, chart interpretation, and visual reasoning make InternVL 2.5 a powerful tool for both researchers and practitioners. With the promising InternVL 3.0 on the horizon, the trajectory of open-source vision-language models continues accelerating, suggesting that the gap between open-source and proprietary models will continue to narrow.

For organizations seeking advanced multimodal AI capabilities with the benefits of open-source transparency, reproducibility, and customizability, InternVL 2.5 provides a compelling alternative to proprietary solutions.
