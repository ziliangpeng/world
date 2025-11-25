# Qwen2.5: A Party of Foundation Models with 18 Trillion Tokens

## Overview

**Qwen2.5** represents a monumental scale-up in the Qwen series, training on **18 trillion tokens** (2.57× more than Qwen2's 7T) to create what Alibaba Cloud calls "A Party of Foundation Models" - a comprehensive suite spanning seven parameter sizes with specialized variants for coding, mathematics, vision, and multimodal understanding. Released in September 2024, Qwen2.5-72B-Instruct achieves performance **comparable to Llama-3-405B** while using only **one-fifth the parameters**, demonstrating remarkable efficiency from the massive data scale-up.

The series encompasses **seven base sizes** (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B), multiple **specialized variants** (Coder, Math, VL, Omni), and proprietary **MoE models** (Turbo, Plus, Max). All models maintain Qwen2's proven architecture (GQA, SwiGLU, RMSNorm) while achieving substantial improvements through enhanced data quality, increased scale, and refined training methodology. The flagship achieves **86.1 MMLU**, **83.1 MATH**, and **88.2 MBPP**, with specialized variants pushing even higher.

### Quick Facts

- **Release Date**: September 19, 2024
- **Developer**: Qwen Team, Alibaba Cloud
- **Training Data**: **18 trillion tokens** (2.57× Qwen2)
- **Model Sizes**: 7 sizes (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B)
- **License**: Apache 2.0 (most models; exceptions: 3B, 72B)
- **Context Length**: 128K (131,072) standard, 1M extended
- **Languages**: 29+ languages
- **arXiv Paper**: [2412.15115](https://arxiv.org/abs/2412.15115)

### Model Variants

| Model | Total Params | Context | Key Achievements |
|-------|--------------|---------|------------------|
| **Qwen2.5-0.5B** | 0.5B | 32K | Edge devices, mobile |
| **Qwen2.5-1.5B** | 1.5B | 32K | Portable deployment |
| **Qwen2.5-3B** | 3B | 32K | Mobile applications |
| **Qwen2.5-7B** | 7B | 128K | MMLU 74.2, HumanEval 84.8 |
| **Qwen2.5-14B** | 14B | 128K | MMLU 79.7, BBH 78.2 |
| **Qwen2.5-32B** | 32B | 128K | MATH 57.7, MBPP 84.5 |
| **Qwen2.5-72B** | 72B | 128K | MMLU 86.1, MATH 83.1, Arena-Hard 81.2 |

**Extended Context Variants**:
- Qwen2.5-7B-Instruct-1M
- Qwen2.5-14B-Instruct-1M

---

## Key Innovations

### 1. Massive 18 Trillion Token Scale-Up

**Revolutionary Data Scale**: 2.57× increase from Qwen2

| Model Series | Training Tokens | Scaling Factor |
|--------------|-----------------|----------------|
| Qwen1.5 | ~3 trillion | Baseline |
| Qwen2 | 7 trillion | 2.3× vs. Qwen1.5 |
| **Qwen2.5** | **18 trillion** | **2.57× vs. Qwen2** |
| | | **6× vs. Qwen1.5** |

**Impact**: The most significant improvement in Qwen2.5, providing substantially stronger foundation for:
- Common sense understanding
- Expert knowledge across domains
- Reasoning capabilities
- Multilingual proficiency
- Code understanding
- Mathematical reasoning

**Qwen2.5-Coder**: Additional **5.5 trillion tokens** of code-related data on top of base 18T

**Qwen2.5-Max**: **20+ trillion tokens** (proprietary flagship)

### 2. "Party of Foundation Models" Philosophy

**Comprehensive Model Family**: Seven parameter sizes + specialized variants

#### Scale Diversity (7 Base Sizes)

**New Sizes Added** (vs. Qwen2):
- **3B**: Mobile applications, cost-effective deployment
- **14B**: Sweet spot between 7B and 32B
- **32B**: Bridge to flagship performance

**Complete Range**:
```
Edge Devices: 0.5B, 1.5B, 3B
Balanced: 7B, 14B
Large-Scale: 32B, 72B
```

#### Domain Specialization

**Qwen2.5-Coder**:
- 6 sizes: 0.5B, 1.5B, 3B, 7B, 14B, 32B
- +5.5T code tokens
- 40+ programming languages
- State-of-the-art code generation

**Qwen2.5-Math**:
- 3 sizes: 1.5B, 7B, 72B + reward model
- Self-improvement methodology
- Bilingual (Chinese/English)
- MATH: 79.7-87.8

**Multimodal Extensions**:
- **Qwen2.5-VL**: Vision-language understanding
- **Qwen2.5-Omni**: End-to-end multimodal (text + audio + video + speech)
- **QwQ**: Advanced reasoning model

#### Deployment Flexibility

**Mobile (0.5B-3B)**:
- Smartphones
- Earphones
- Edge devices
- IoT applications

**Cloud (7B-32B)**:
- Cost-effective serving
- Batch processing
- API services

**Flagship (72B)**:
- Maximum performance
- Research applications
- Competitive with 405B models

#### MoE Architecture Variants (Proprietary)

**Qwen2.5-Turbo**:
- MoE architecture
- Competitive with GPT-4o-mini
- Cost-effective alternative

**Qwen2.5-Plus**:
- MoE architecture
- Competitive with GPT-4o
- Superior cost-effectiveness

**Qwen2.5-Max**:
- 20+ trillion tokens
- Largest MoE variant
- Flagship proprietary model

### 3. Enhanced Data Quality and Diversity

**Quality-First Approach**: Learning from Qwen2's experiments

**Rigorous Preprocessing Pipeline**:
1. **Multi-Stage Filtering**: Eliminate irrelevant, low-quality, duplicated content
2. **Model-Based Quality Assessment**: Qwen2-Instruct models used as quality filters
3. **Multi-Dimensional Analysis**: Comprehensive quality evaluation
4. **Domain-Specific Curation**: Specialized data for code and mathematics

**Data Sources**:
- Common Crawl
- Wikipedia
- BooksCorpus
- Scientific papers
- Specialized domain content (code, mathematics)
- Synthetic data (for specialized variants)

**Quality > Quantity**: Emphasis on data quality over raw volume

### 4. Staged Pre-Training with Progressive Complexity

**Training Stages**:

**Stage 1: Initial Pre-training**
- **Context Length**: 4,096 tokens
- **Focus**: Core language understanding
- **Data**: General domain mixture

**Stage 2: Context Extension**
- **Transition**: 4,096 → 32,768 tokens
- **Focus**: Long-range dependencies
- **Method**: Gradual extension

**Stage 3: Data Mixture Transitions**
- **Strategy**: Staged transitions among different data mixtures
- **Benefit**: Curriculum learning from simple to complex

**Domain-Specific Progressive Training**:
- **Qwen2.5-Coder**: Per-file and per-repository training
- **Qwen2.5-Math**: Progressive mathematical complexity
- **Base Models**: Gradual domain integration

### 5. Substantially Improved Instruction Following

**Post-Training Scale-Up**: Over **1 million high-quality supervised fine-tuning samples**

**Multi-Stage Reinforcement Learning**:
- Iterative RL implementation
- Enhanced human preference alignment
- Long-text generation capabilities (2K → 8K tokens)
- Better tool use and structured I/O

**Usability Improvements**:

| Capability | Qwen2 | Qwen2.5 | Improvement |
|------------|-------|---------|-------------|
| **Generation Length** | 2K tokens | 8K tokens | 4× longer |
| **Structured Output** | Basic | Enhanced | JSON, tables |
| **Tool Use** | Limited | Easier | Simplified API |
| **System Prompts** | Sensitive | Resilient | Diverse support |
| **Instruction Following** | Good | Excellent | Higher quality |

---

## Architecture Details

### Core Architecture (Maintained from Qwen2)

**Fundamental Design**: No significant architectural changes to base language model

**Key Components**:
- **Type**: Decoder-only Transformer
- **Position Embeddings**: Rotary Position Embeddings (RoPE)
- **RoPE Theta**: 10,000 (standard), extensible to 1,000,000 (long context)
- **Activation Function**: SwiGLU (Swish-Gated Linear Unit)
- **Normalization**: RMSNorm (Root Mean Square Layer Normalization)
- **Attention**: Grouped Query Attention (GQA) with QKV bias

**Philosophy**: Improvements through data quality, quantity, and training methodology rather than architectural changes

### Model-Specific Configurations

#### Qwen2.5-0.5B
- **Layers**: 24
- **Attention Heads**: 14 query, 2 key-value (GQA 7:1)
- **Vocabulary**: 151,936
- **Context**: 32,768 tokens input, 8,192 generation
- **Target**: Edge devices, mobile deployment

#### Qwen2.5-1.5B
- **Layers**: 28
- **Attention Heads**: 12 query, 2 key-value (GQA 6:1)
- **Vocabulary**: 151,936
- **Context**: 32,768 tokens input, 8,192 generation
- **Target**: Portable devices, cost-effective cloud

#### Qwen2.5-3B (New Size)
- **Attention Heads**: 16 query, 2 key-value (GQA 8:1)
- **Vocabulary**: 151,936
- **Context**: 32,768 tokens input, 8,192 generation
- **Target**: Mobile applications, balanced performance

#### Qwen2.5-7B
- **Layers**: 28
- **Attention Heads**: 28 query, 4 key-value (GQA 7:1)
- **Vocabulary**: 151,936
- **Context**: 131,072 tokens input, 8,192 generation
- **Achievement**: MMLU 74.2, HumanEval 84.8

#### Qwen2.5-14B (New Size)
- **Layers**: 48
- **Hidden Size**: 5,120
- **Intermediate Size**: 13,824
- **Attention Heads**: 40 query, 8 key-value (GQA 5:1)
- **Head Dimension**: 128
- **Vocabulary**: 151,936
- **Total Parameters**: 14.7B
- **Non-Embedding Parameters**: 13.1B
- **Context**: 131,072 tokens input, 8,192 generation
- **Achievement**: MMLU 79.7, BBH 78.2

#### Qwen2.5-32B (New Size)
- **Layers**: 64
- **Hidden Size**: 5,120
- **Intermediate Size**: 27,648
- **Attention Heads**: 40 query, 8 key-value (GQA 5:1)
- **Head Dimension**: 128
- **Vocabulary**: 151,936
- **Context**: 131,072 tokens input, 8,192 generation
- **Achievement**: MATH 57.7 (base), MBPP 84.5

#### Qwen2.5-72B (Flagship)
- **Layers**: 80
- **Attention Heads**: 64 query, 8 key-value (GQA 8:1)
- **Vocabulary**: 151,936
- **Context**: 131,072 tokens input, 8,192 generation
- **Achievement**: MMLU 86.1, MATH 83.1, Arena-Hard 81.2

### Context Length Capabilities

**Standard Models**:
- **Large (7B-72B)**: 131,072 tokens input, 8,192 generation
- **Small (0.5B-3B)**: 32,768 tokens input, 8,192 generation

**Extended Context Variants**:
- **Qwen2.5-7B-Instruct-1M**: Up to 1 million tokens
- **Qwen2.5-14B-Instruct-1M**: Up to 1 million tokens
- **Method**: Enhanced YARN + Dual Chunk Attention from Qwen2

---

## Training Details

### Pre-Training Data Scale

**Massive Expansion**:
- **Base Models**: 18 trillion tokens
- **Qwen2.5-Coder**: 18T base + 5.5T code = 23.5T total
- **Qwen2.5-Max**: 20+ trillion tokens

**Comparison Timeline**:
```
Qwen1.5: 3T tokens
   ↓  2.3× scale-up
Qwen2: 7T tokens
   ↓  2.57× scale-up
Qwen2.5: 18T tokens
   ↓  cumulative 6× vs Qwen1.5
```

### Data Composition

**Primary Sources**:
- **Common Crawl**: Web-scale text data
- **Wikipedia**: Encyclopedic knowledge
- **BooksCorpus**: Long-form text
- **Scientific Papers**: Domain expertise
- **Code Repositories**: Programming knowledge
- **Mathematics Content**: Problem-solving data
- **Synthetic Data**: High-quality generated examples

**Domain Distribution** (estimated):
- General text: ~60%
- Code: ~25%
- Mathematics: ~10%
- Other specialized: ~5%

**Multilingual Content**: Over 29 languages with focus on Asian languages

### Training Methodology

#### Staged Pre-Training

**Phase 1: Short Context (4,096 tokens)**
- Initial training on diverse corpus
- Core language understanding
- Foundation building

**Phase 2: Context Extension (4,096 → 32,768 tokens)**
- Gradual extension of context window
- Long-range dependency learning
- Memory and coherence training

**Phase 3: Data Mixture Transitions**
- Transitions among different data mixtures
- Curriculum learning approach
- Progressive complexity increase

#### Domain-Specific Training

**Qwen2.5-Coder**:
- Per-file basis: Individual file understanding
- Per-repository basis: Cross-file dependencies
- Maximum 32,768 token processing
- Supports 40+ programming languages

**Qwen2.5-Math**:
- Self-improvement pipeline
- Reward model-guided training
- Synthetic data generation via Qwen2-Math
- Iterative quality enhancement

### Training Infrastructure

**Hardware**: NVIDIA A100/H100 GPUs (estimated, not disclosed)

**Precision**: BFloat16

**Training Duration**: Not publicly disclosed

**Estimated Compute**: Massive scale based on 18T token corpus

---

## Post-Training and Alignment

### Supervised Fine-Tuning (SFT)

**Scale**: Over **1 million high-quality supervised fine-tuning samples**

**Coverage Areas**:
- **Instruction Following**: Enhanced command understanding
- **Long Text Generation**: Extended from 2K to 8K tokens
- **Structured Data**: Tables, JSON, XML understanding and generation
- **Coding**: Multiple programming languages
- **Mathematics**: Problem-solving and reasoning
- **Roleplay**: Character consistency and engagement
- **Multilingual**: 29+ languages
- **Safety**: Harmlessness and alignment

**Training Configuration**:
- Full context length: 32,768 tokens (small) or 131,072 tokens (large)
- Multiple epochs for instruction optimization
- Gradual increase in task complexity

### Multi-Stage Reinforcement Learning

**Iterative RL Process**:
1. **Generation**: Model generates multiple responses
2. **Evaluation**: Human or reward model ranking
3. **Optimization**: Policy updated based on preferences
4. **Iteration**: Repeated cycles for refinement

**Key Improvements**:
- **Human Preference Alignment**: Better matches user expectations
- **Long-Text Generation**: Coherent 8K+ token outputs
- **Tool Use**: Simplified and more effective API calling
- **Structured I/O**: Enhanced JSON, table generation

### Usability Enhancements Over Qwen2

**Generation Length**:
- **Qwen2**: 2,000 tokens typical max
- **Qwen2.5**: 8,192 tokens generation limit
- **Impact**: Better for articles, reports, documentation

**Structured Data Handling**:
- **Improved JSON Generation**: Valid syntax, better formatting
- **Table Understanding**: Parse and generate markdown/HTML tables
- **Schema Adherence**: Follow specified data structures

**Tool Use**:
- **Easier Integration**: Simplified function calling
- **Better Understanding**: Clearer parameter interpretation
- **Error Handling**: More graceful failure modes

**System Prompt Resilience**:
- **Diverse Prompts**: Works with varied instruction styles
- **Role-Play**: Better character consistency
- **Conditional Settings**: Handles complex constraints

---

## Performance Benchmarks

### Qwen2.5-72B-Instruct (Flagship)

#### Language Understanding

| Benchmark | Qwen2.5-72B | Qwen2-72B | Improvement |
|-----------|-------------|-----------|-------------|
| **MMLU** | **86.1** | 84.2 | +1.9 ✓ |
| **MMLU-redux** | **86.8** | N/A | New |
| **Arena-Hard** | **81.2** | 48.1 | +33.1 ✓✓ |

**Analysis**: Substantial improvements in both raw knowledge (MMLU) and conversational ability (Arena-Hard).

#### Coding Performance

| Benchmark | Qwen2.5-72B-Instruct | Notes |
|-----------|---------------------|-------|
| **HumanEval** | N/A | See Coder variants |
| **MBPP** | **88.2** | Strong code generation |
| **LiveCodeBench** | **55.5** | Real-world coding |
| **MultiPL-E** | **75.1** | Multilingual code |

#### Mathematical Reasoning

| Benchmark | Qwen2.5-72B-Instruct | Qwen2-72B | Improvement |
|-----------|---------------------|-----------|-------------|
| **MATH** | **83.1** | ~50 | +33 ✓✓ |

**Analysis**: Dramatic improvement in mathematical reasoning, likely due to increased math content in 18T training data.

### Comparison with Competitors

#### vs. Llama-3-405B (5× Larger)

| Benchmark | Qwen2.5-72B-Instruct | Llama-3-405B-Instruct | Verdict |
|-----------|---------------------|----------------------|---------|
| **General Performance** | **Comparable** | Baseline | Qwen2.5 competitive |
| **Parameters** | 72B | **405B** | Qwen2.5 5× smaller |
| **Efficiency** | **Higher** | Lower | Qwen2.5 wins |

**Key Insight**: Qwen2.5-72B achieves results comparable to Llama-3-405B while using only one-fifth the parameters, demonstrating exceptional efficiency from the 18T token scale-up.

#### vs. GPT-4o

| Domain | Qwen2.5-72B | GPT-4o | Verdict |
|--------|-------------|--------|---------|
| **Language Understanding** | Strong | **Superior** | GPT-4o leads |
| **Coding** | Competitive | **Better** | GPT-4o leads |
| **Mathematics** | Strong | Comparable | Close |
| **Cost** | **Open-source** | Expensive | Qwen2.5 wins |

**Qwen2.5-Plus (MoE)**: Competitive with GPT-4o at superior cost-effectiveness

### Performance by Model Size

#### Qwen2.5-7B

| Benchmark | Qwen2.5-7B-Instruct | Notes |
|-----------|-------------------|-------|
| **MMLU** | **74.2** | Strong for 7B |
| **HumanEval** | **84.8** | Excellent coding |
| **MATH** | **75.5** (instruct) | Good reasoning |

**Analysis**: Exceptional performance for 7B size class, competitive with much larger models.

#### Qwen2.5-14B

| Benchmark | Qwen2.5-14B | Notes |
|-----------|-------------|-------|
| **MMLU** | **79.7** | Sweet spot size |
| **BBH** | **78.2** | Strong reasoning |

#### Qwen2.5-32B

| Benchmark | Qwen2.5-32B | Notes |
|-----------|-------------|-------|
| **MATH** | **57.7** (base) | Strong base model |
| **MBPP** | **84.5** | Excellent coding |

### Specialized Variant Performance

#### Qwen2.5-Coder

**HumanEval (Pass@1)**:
- **Qwen2.5-Coder-32B-Instruct**: **92.7%**
- **Qwen2.5-Coder-7B-Instruct**: **88.4%**
- Competitive with GPT-4o

**LiveCodeBench**:
- **Qwen2.5-Coder-7B-Instruct**: **37.6**
- Best among open-source models

**McEval (40+ Languages)**:
- **Qwen2.5-Coder**: **65.9**
- Broad language support

#### Qwen2.5-Math

**MATH Benchmark (TIR - Tool-Integrated Reasoning)**:
- **Qwen2.5-Math-72B-Instruct**: **87.8**
- **Qwen2.5-Math-7B-Instruct**: **85.3**
- **Qwen2.5-Math-1.5B-Instruct**: **79.7**

**With Reward Model (RM@8 TIR)**:
- **Qwen2.5-Math-72B-Instruct**: **92.9**
- Surpasses GPT-4o and Gemini Math-Specialized 1.5 Pro

---

## Improvements Over Qwen2

### Data Scale and Quality

**Massive Scale-Up**:
- **Qwen2**: 7 trillion tokens
- **Qwen2.5**: **18 trillion tokens** (2.57× increase)
- **Impact**: Substantially stronger foundation across all capabilities

**Enhanced Quality**:
- Rigorous multi-stage preprocessing
- Model-based quality filtering (using Qwen2-Instruct)
- Multi-dimensional quality analysis
- Synthetic high-quality data generation

### Capability Improvements

**Knowledge Enhancement**:
- **MMLU**: 84.2 → 86.1 (+1.9)
- **General Knowledge**: Significantly broader and deeper
- **Expert Domains**: Better coverage of specialized fields

**Coding Excellence**:
- Dedicated Qwen2.5-Coder variants
- **HumanEval**: Substantial improvements across sizes
- **MBPP**: 88.2 (72B-Instruct)
- **LiveCodeBench**: 55.5 (best among open-source)

**Mathematical Prowess**:
- Dedicated Qwen2.5-Math variants
- **MATH**: ~50 → 83.1 (+33 points)
- **Self-improvement methodology**: Iterative quality enhancement
- **Reward model**: 92.9 with RM@8 TIR

**Usability Leap**:
- **Generation Length**: 2K → 8K tokens (4× increase)
- **Structured Output**: Enhanced JSON, table generation
- **Tool Use**: Simplified and more effective
- **System Prompt Resilience**: Better handling of diverse prompts

### Architectural Consistency

**No Major Changes**: Maintained proven Qwen2 architecture
- Grouped Query Attention (GQA)
- SwiGLU activation
- RMSNorm normalization
- RoPE position embeddings

**Philosophy**: Improvements through data and training, not architectural complexity

### Model Family Expansion

**New Sizes Added**:
- **3B**: Mobile applications sweet spot
- **14B**: Bridge between 7B and 32B
- **32B**: Large-scale performance at reasonable cost

**Complete Range**: Seven sizes from 0.5B to 72B

**Specialized Variants**:
- Qwen2.5-Coder (6 sizes)
- Qwen2.5-Math (3 sizes + reward model)
- Qwen2.5-VL (vision-language)
- Qwen2.5-Omni (multimodal)
- QwQ (reasoning)

### Context and Generation

**Extended Context**:
- Maintained 128K standard
- Added 1M token variants (7B, 14B)
- Improved long-text coherence

**Generation Quality**:
- Longer outputs (8K tokens)
- Better structure and formatting
- Enhanced factual accuracy
- Improved stylistic control

---

## Multilingual Capabilities

### Language Support

**Total**: Over **29 languages**

**Primary Languages**:
1. English (primary)
2. Chinese (primary)
3. Spanish
4. French
5. Portuguese
6. German
7. Italian
8. Russian
9. Japanese
10. Korean
11. Vietnamese
12. Thai
13. Arabic

**Additional Languages** (16+):
Croatian, Czech, Danish, Dutch, Estonian, Finnish, Greek, Hungarian, Indonesian, Khmer, Latvian, Lithuanian, Norwegian, Polish, Swedish, Bengali, Hindi, and more.

### Multilingual Training Enhancements

**Focus Areas**:
- **Asian Languages**: Particular emphasis and quality improvement
- **European Languages**: Broad coverage with high quality
- **Code-Switching**: Better handling of multilingual contexts
- **Cultural Context**: Improved understanding of cultural nuances

**Training Strategy**:
- 18T token corpus includes diverse multilingual content
- Quality filtering applied across all languages
- Balanced representation while maintaining quality

---

## Technical Resources and Integration

### Official Resources

#### Papers
- **Primary**: [Qwen2.5 Technical Report (arXiv:2412.15115)](https://arxiv.org/abs/2412.15115)
  - Published: December 19, 2024 (v1), January 3, 2025 (v2)

- **Specialized**:
  - [Qwen2.5-Coder (arXiv:2409.12186)](https://arxiv.org/abs/2409.12186)
  - [Qwen2.5-Math (arXiv:2409.12122)](https://arxiv.org/abs/2409.12122)
  - [Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/pdf/2502.13923)

#### Official Blog Posts
- [Qwen2.5: A Party of Foundation Models!](https://qwenlm.github.io/blog/qwen2.5/)
- [Qwen2.5-LLM: Extending the boundary of LLMs](https://qwenlm.github.io/blog/qwen2.5-llm/)
- [Qwen2.5-Coder Series](https://qwenlm.github.io/blog/qwen2.5-coder-family/)
- [Qwen2.5-Max](https://qwenlm.github.io/blog/qwen2.5-max/)

#### GitHub Repositories
- **Main Repository**: [github.com/QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5)
- **Legacy Repository**: [github.com/QwenLM/Qwen](https://github.com/QwenLM/Qwen) (no longer maintained)

#### Model Cards (Hugging Face)
- **Organization**: [huggingface.co/Qwen](https://huggingface.co/Qwen)
- **Collection**: [Qwen2.5 Collection](https://huggingface.co/collections/Qwen/qwen25)
- **Individual Models**:
  - [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
  - [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
  - [Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
  - [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
  - [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
  - And more...

### Framework Integration

#### Basic Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Prepare messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a Python function for binary search."}
]

# Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Generate response
inputs = tokenizer([text], return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

#### Long Context Generation (8K Tokens)

```python
# Configure for long-form generation
sampling_params = {
    "max_new_tokens": 8192,  # Extended generation limit
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "length_penalty": 1.0
}

# Generate long-form content
long_prompt = "Write a comprehensive technical article about quantum computing..."
inputs = tokenizer([long_prompt], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, **sampling_params)
```

#### Structured Output (JSON)

```python
# Generate structured JSON output
messages = [
    {"role": "system", "content": "Generate valid JSON matching the specified schema."},
    {"role": "user", "content": """Generate a JSON object for a person with fields:
    name (string), age (number), hobbies (array of strings), address (object with street, city, country)"""}
]

# Apply chat template
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate with constrained decoding (requires additional tools like guidance or outlines)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
json_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Licensing

**Most Models**: Apache 2.0 (commercial use allowed)

**Exceptions**:
- **Qwen2.5-3B**: Different license (check model card)
- **Qwen2.5-72B**: Different license (check model card)

**Specialized Variants**: Generally Apache 2.0 (verify individual model cards)

---

## Summary of Technical Contributions

### 1. Massive 18 Trillion Token Scale-Up

**Innovation**: 2.57× increase from Qwen2 (7T → 18T tokens)

**Impact**:
- Substantially stronger foundation for all capabilities
- Better common sense and expert knowledge
- Improved reasoning across domains
- Enhanced multilingual proficiency
- Superior code and mathematical understanding
- Established new benchmark for open-source training scale

### 2. "Party of Foundation Models" Ecosystem

**Innovation**: Comprehensive model family with seven sizes and multiple specialized variants

**Impact**:
- Deployment flexibility from edge (0.5B) to cloud (72B)
- Domain specialization (Coder, Math, VL, Omni)
- Cost-effectiveness across use cases
- Unified architecture for consistency
- Democratized access to powerful models

### 3. Enhanced Data Quality and Curation

**Innovation**: Rigorous multi-stage preprocessing with model-based quality filtering

**Impact**:
- Higher quality training corpus
- Better domain coverage and balance
- Improved factual accuracy
- Reduced low-quality and duplicated content
- Foundation for specialized variant quality

### 4. Substantial Usability Improvements

**Innovation**: 4× longer generation (2K → 8K tokens), better structured output, easier tool use

**Impact**:
- Practical long-form content generation
- Reliable JSON and table generation
- Simplified API integration
- Better handling of diverse prompts
- Enhanced instruction following

### 5. Specialized Variant Excellence

**Innovation**: Qwen2.5-Coder and Qwen2.5-Math achieve state-of-the-art in their domains

**Impact**:
- **Coder**: Competitive with GPT-4o (HumanEval 92.7% for 32B)
- **Math**: Surpasses GPT-4o (MATH 92.9 with RM@8 TIR)
- Best open-source performance on domain benchmarks
- Self-improvement methodology for mathematics
- 40+ programming language support

### 6. Exceptional Parameter Efficiency

**Innovation**: Qwen2.5-72B comparable to Llama-3-405B (5× larger)

**Impact**:
- Demonstrates power of data scale and quality
- Reduces computational requirements
- Lower deployment costs
- Faster inference
- Environmental benefits

### 7. Architectural Consistency and Stability

**Innovation**: Maintained proven Qwen2 architecture while achieving substantial improvements

**Impact**:
- Proven reliability and stability
- Smooth migration path from Qwen2
- Focus on data and training methodology
- Avoided unnecessary complexity
- Foundation for continued innovation

---

## Conclusion

Qwen2.5 represents a monumental achievement in open-source language models through its **18 trillion token scale-up** (2.57× Qwen2) and comprehensive "Party of Foundation Models" approach spanning seven parameter sizes with multiple specialized variants. Released in September 2024, the flagship **Qwen2.5-72B-Instruct achieves performance comparable to Llama-3-405B** while using only **one-fifth the parameters**, demonstrating exceptional efficiency from massive data scale and quality.

Key achievements include:

- **Massive scale**: 18T tokens (2.57× Qwen2), 20T+ for Qwen2.5-Max
- **Comprehensive family**: 7 sizes (0.5B-72B) + specialized variants
- **Language understanding**: MMLU 86.1 (+1.9 vs Qwen2)
- **Mathematical reasoning**: MATH 83.1 (+33 vs Qwen2), 92.9 with RM@8
- **Coding excellence**: MBPP 88.2, HumanEval 92.7% (Coder-32B)
- **Usability leap**: 8K generation (+4× vs Qwen2), enhanced structured output
- **Specialized leadership**: Coder and Math achieve state-of-the-art in their domains
- **Parameter efficiency**: 72B comparable to 405B models

The **Apache 2.0 license** (for most models) democratizes access to advanced language AI across the full spectrum of deployment scenarios, from edge devices to cloud infrastructure. Qwen2.5's focus on data quality and scale over architectural complexity establishes a new paradigm for model development, demonstrating that massive high-quality training data can achieve substantial improvements while maintaining architectural stability.

As the foundation for specialized variants (Coder, Math, VL, Omni) and successor models (Qwen3), Qwen2.5 exemplifies the "Party of Foundation Models" philosophy: providing a comprehensive, unified ecosystem of models tailored to diverse use cases while maintaining consistent quality and capabilities across the entire family.

---

## References and Citations

### Primary Sources

1. **Qwen2.5 Technical Report**
   Qwen Team. (2024). Qwen2.5 Technical Report. *arXiv preprint arXiv:2412.15115*.
   [https://arxiv.org/abs/2412.15115](https://arxiv.org/abs/2412.15115)

2. **Qwen2.5-Coder Technical Report**
   Hui, B., et al. (2024). Qwen2.5-Coder Technical Report. *arXiv preprint arXiv:2409.12186*.
   [https://arxiv.org/abs/2409.12186](https://arxiv.org/abs/2409.12186)

3. **Qwen2.5-Math Technical Report**
   Yang, A., et al. (2024). Qwen2.5-Math: Toward Mathematical Expert Model via Self-Improvement. *arXiv preprint arXiv:2409.12122*.
   [https://arxiv.org/abs/2409.12122](https://arxiv.org/abs/2409.12122)

### Official Resources

4. **Qwen2.5: A Party of Foundation Models!**
   [https://qwenlm.github.io/blog/qwen2.5/](https://qwenlm.github.io/blog/qwen2.5/)

5. **Qwen2.5-LLM: Extending the boundary of LLMs**
   [https://qwenlm.github.io/blog/qwen2.5-llm/](https://qwenlm.github.io/blog/qwen2.5-llm/)

6. **Qwen2.5-Coder Series Blog**
   [https://qwenlm.github.io/blog/qwen2.5-coder-family/](https://qwenlm.github.io/blog/qwen2.5-coder-family/)

7. **Qwen2.5-Max Blog**
   [https://qwenlm.github.io/blog/qwen2.5-max/](https://qwenlm.github.io/blog/qwen2.5-max/)

### GitHub and Model Cards

8. **Qwen2.5 GitHub Repository**
   [https://github.com/QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5)

9. **Qwen Hugging Face Organization**
   [https://huggingface.co/Qwen](https://huggingface.co/Qwen)

10. **Qwen2.5 Model Collection**
    [https://huggingface.co/collections/Qwen/qwen25](https://huggingface.co/collections/Qwen/qwen25)

11. **Qwen2.5-72B-Instruct Model Card**
    [https://huggingface.co/Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)

---

**Document Version**: 1.0
**Last Updated**: 2025-01-24
**Model Versions Covered**: Qwen2.5 (0.5B-72B), Qwen2.5-Coder, Qwen2.5-Math
**License**: Apache 2.0 (most models; exceptions: 3B, 72B - check model cards)
