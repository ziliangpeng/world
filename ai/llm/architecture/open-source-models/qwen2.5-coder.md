# Qwen2.5-Coder: State-of-the-Art Open-Source Coding LLM

## Overview

**Qwen2.5-Coder** is a code-specialized large language model series developed by the Qwen team at Alibaba Cloud, released in September 2024 (expanded in November 2024). The series includes six model sizes (0.5B, 1.5B, 3B, 7B, 14B, and 32B parameters) trained on 5.5 trillion tokens of diverse programming data. The flagship **Qwen2.5-Coder-32B-Instruct** achieves state-of-the-art performance among open-source models and **matches GPT-4o's coding capabilities** while remaining fully open-source under Apache 2.0 license.

**Key Innovation**: Massive 5.5T token code-specialized training with balanced data mixing (70% code, 20% text-code grounding, 10% math) combined with synthetic data generation and executor-based validation, achieving GPT-4o-level performance on code repair (Aider: 73.7%) and multi-language coding tasks.

## Release Information

- **Initial Release**: September 18, 2024 (1.5B, 7B models; 32B announced)
- **Expanded Release**: November 12, 2024 (full family: 0.5B, 1.5B, 3B, 7B, 14B, 32B)
- **Technical Paper**: arXiv:2409.12186 (v1: Sept 18, 2024; v3: Nov 12, 2024)
- **Organization**: Qwen Team, Alibaba Cloud
- **License**:
  - Apache 2.0: 0.5B, 1.5B, 7B, 14B, 32B (fully permissive for commercial use)
  - Qwen-Research: 3B only (different terms)

### Model Variants

Each model size comes in two variants:
- **Base models**: For continued pretraining or custom fine-tuning
- **Instruct models**: Instruction-tuned for conversational coding tasks

## Architecture

### Core Components

Qwen2.5-Coder inherits the Qwen2.5 transformer architecture:

```
Transformer Architecture:
├── Position Encoding: RoPE (Rotary Position Embedding)
├── Activation: SwiGLU
├── Normalization: RMSNorm
├── Attention: GQA (Grouped Query Attention) with QKV bias
└── Context Length: 131,072 tokens (132K) with YARN
```

### Model Specifications

#### Qwen2.5-Coder-7B
```yaml
Parameters: 7.6B total
Attention Heads:
  Query (Q): 28 heads
  Key-Value (KV): 4 heads
  GQA Ratio: 7:1
Context Length: 131,072 tokens
```

#### Qwen2.5-Coder-32B (Flagship Model)
```yaml
Total Parameters: 32.5B
Non-Embedding Parameters: 31.0B
Layers: 64
Attention Heads:
  Query (Q): 40 heads
  Key-Value (KV): 8 heads
  GQA Ratio: 5:1
Context Length: 131,072 tokens (132K)
```

### Tokenizer

```yaml
Vocabulary Size: 151,646 tokens (inherited from Qwen2.5)
Type: BPE-based

Special Tokens for Code:
  Sequence Control:
    - <|endoftext|>: End of sequence marker

  Fill-in-the-Middle (FIM):
    - <|fim_prefix|>: Code before the gap
    - <|fim_middle|>: Code to be generated (the gap)
    - <|fim_suffix|>: Code after the gap
    - <|fim_pad|>: FIM padding

  Repository Context:
    - <|repo_name|>: Repository identifier
    - <|file_sep|>: File separator for multi-file context

  ChatML Format:
    - <|im_start|>: Chat message start
    - <|im_end|>: Chat message end
```

### Context Window Scaling

```
Training Progression:
  Initial: 8,192 tokens
      ↓
  Extended: 32,768 tokens (during repo-level pretraining)
      ↓
  Maximum: 131,072 tokens (132K) with YARN

RoPE Configuration:
  Base Frequency: 10,000 → 1,000,000
  YARN Scaling Factor: 4.0
  Original Max Position: 32,768
```

**Note**: To enable full 128K context, add rope_scaling configuration to config.json. vLLM only supports static YARN (constant scaling factor regardless of input length), which may impact performance on shorter texts.

## Training Methodology

### Pre-training Data Composition

**Total Training Tokens**: 5.5 Trillion

#### Data Type Breakdown (5 Categories)

```
Data Mix Distribution:
├── Source Code Data (~70%)
│   ├── Public GitHub repositories (created before Feb 2024)
│   ├── 92 programming languages
│   ├── Raw code files
│   ├── Pull Requests
│   ├── Commits
│   ├── Jupyter Notebooks
│   └── Kaggle datasets
│
├── Text-Code Grounding Data (~20%)
│   ├── 4-stage filtering applied
│   └── Quality improvement: 41.6% → 46.8% on HumanEval/MBPP
│
├── Synthetic Data
│   ├── Generated using CodeQwen1.5 (predecessor)
│   ├── Executor-based validation
│   └── Ensures code executability
│
├── Math Data (~10%)
│   ├── From Qwen2.5-Math corpus
│   └── Enhances mathematical reasoning
│
└── General Text Data
    └── Maintains versatile language understanding
```

### Data Quality Measures

1. **Weak Classifiers**: For quality assessment
2. **10-gram Overlap Decontamination**: Prevent data leakage
3. **Executor Validation**: Ensure synthetic code executes correctly
4. **Rule-based Cleaning**: Meticulous cleaning throughout
5. **4-Stage Text-Code Filtering**: Quality improvement from 41.6% → 46.8%

### Training Objectives

- **Next Token Prediction**: Standard autoregressive training
- **Fill-in-the-Middle (FIM)**: Code completion within existing code

### Context Length Extension Process

```
Stage 1: Initial Training
  - Context: 8,192 tokens
  - Training: General code patterns

Stage 2: Repository-Level Pretraining
  - Context: 32,768 tokens
  - Training: Multi-file understanding
  - RoPE adjustment: Base 10K → 1M

Stage 3: YARN Application
  - Context: 131,072 tokens (132K)
  - Mechanism: Length extrapolation
  - Scaling: Factor 4.0
```

### Data Mixing Strategy

**Design Philosophy**: Balance specialization with versatility

```
Balanced Modality Mixing:
  Code: 70% (specialization)
  Text-Code: 20% (grounding)
  Math: 10% (reasoning)

Goal: Enhance coding performance while retaining general capabilities
```

## Post-Training

### Available Post-Training Methods

**Base Models Support**:
- Supervised Fine-Tuning (SFT)
- Reinforcement Learning from Human Feedback (RLHF)
- Continued pretraining
- Fill-in-the-Middle tasks

### Instruct Models Post-Training

Based on broader Qwen2.5 series approach:

**Data Volume**: ~1 million examples

```
Post-Training Pipeline:
├── Stage 1: SFT (Supervised Fine-Tuning)
│   ├── Phase 1: Short instructions only (up to 32K tokens)
│   └── Phase 2: Mixed short (32K) + long (262K tokens)
│
├── Stage 2: DPO (Direct Preference Optimization)
│   └── Learn from preference pairs
│
└── Stage 3: GRPO (Group Relative Policy Optimization)
    └── Group-based RL refinement
```

**Official Recommendation**: Use pre-trained instruct versions rather than applying custom post-training to base models.

## Key Capabilities

### 1. Multi-Language Programming Support

**Total Languages**: 92 programming languages

**Examples Mentioned**:
- Python, Java, C++, JavaScript
- Ruby, Rust, Haskell, Racket
- And 84 more languages

**Note**: Complete list of all 92 languages not publicly disclosed.

### 2. Fill-in-the-Middle (FIM)

**Purpose**: Predict missing parts of code blocks

**Usage Format**:
```
<|fim_prefix|>{code_before}<|fim_suffix|>{code_after}<|fim_middle|>{model_generates_here}
```

**Applications**:
- IDE code completion
- Inline code suggestions
- Smart refactoring assistance

### 3. Repository-Level Code Understanding

**Special Tokens**: `<|repo_name|>` and `<|file_sep|>`

**Format**:
```
<|repo_name|>{repository_name}
<|file_sep|>{file_path_1}
{file_content_1}
<|file_sep|>{file_path_2}
{file_content_2}
...
```

**Capabilities**:
- Multi-file context understanding
- Cross-file dependency resolution
- Project-wide refactoring
- Inter-file relationship reasoning

### 4. Chat Template (Instruct Models)

**Format**: ChatML

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud.<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>
```

## Benchmark Performance

### Code Generation

#### HumanEval (Python Code Generation)

```
Model                          Pass@1
────────────────────────────────────────────
Qwen2.5-Coder-7B-Instruct      88.4%
Qwen2.5-Coder-32B-Instruct     92.7%

Comparisons:
  CodeStral-22B                ~81%
  DeepSeek-Coder-33B-Instruct  ~79%
  GPT-4o                       ~90%
```

#### LiveCodeBench (2024.07-2024.11)

**Purpose**: Test on latest questions to avoid training data leakage

```
Qwen2.5-Coder-7B-Instruct: 37.6% (Pass@1)
```

### Code Completion (SOTA on 5 Benchmarks)

**Qwen2.5-Coder-32B** achieved state-of-the-art on:

1. **Humaneval-Infilling**: Fill-in-the-middle code completion
2. **CrossCodeEval**: Cross-file code understanding
3. **CrossCodeLongEval**: Long-context code completion
4. **RepoEval**: Repository-level evaluation
5. **SAFIM**: Structured code infilling

### Code Repair

#### Aider Benchmark (Code Editing in Conversational Context)

```
Model                          PASS@1
────────────────────────────────────────────
Qwen2.5-Coder-7B-Instruct      51.9%
  ├── vs CodeStral-22B         Higher
  └── vs DS-Coder-33B-Inst     Higher

Qwen2.5-Coder-32B-Instruct     73.7%
  └── Comparable to GPT-4o     ✓
```

#### MdEval (Multi-Language Code Repair)

**Dataset**: 18 languages, 1,200 samples

```
Qwen2.5-Coder-32B-Instruct: 75.2%
  ├── Rank: #1 among all open-source models
  └── vs GPT-4o: Higher performance
```

### Multi-Language Coding

#### McEval (40+ Programming Languages)

```
Qwen2.5-Coder-32B-Instruct: 65.9%

Strong performance noted in:
  - Haskell
  - Racket
  - Other less common languages
```

#### MultiPL-E (8 Programming Languages)

```
Qwen2.5-Coder-7B-Instruct:
  - Surpasses CodeStral-22B
  - Surpasses DS-Coder-33B-Instruct

Qwen2.5-Coder-14B-Instruct:
  - Strong performance across all 8 languages

Qwen2.5-72B-Instruct (general model): 75.1%
```

### Extended Evaluation

#### EvalPlus (Extended HumanEval + MBPP)

```
Qwen2.5-Coder-7B-Instruct:
  - Outperforms comparable-size models
  - Outperforms larger 20B+ parameter models

Qwen2.5-Coder-32B-Instruct:
  - Highest performance on EvalPlus
  - Outperforms DS-Coder-V2-Instruct
```

#### BigCodeBench

**Result**: Qwen2.5-Coder-32B-Instruct achieved best performance among open-source models

**Note**: Specific numerical scores not disclosed in technical report.

### Mathematical Reasoning

#### MATH Benchmark

```
Qwen2.5-Coder-32B: 57.2%

Note: Integration of Qwen2.5-Math pre-training corpus
      enhances mathematical reasoning capabilities
```

## Comparison: Qwen2.5-Coder vs GPT-4o

### Performance Parity

| Metric | Qwen2.5-Coder-32B | GPT-4o | Winner |
|--------|-------------------|---------|---------|
| Aider (Code Repair) | 73.7% | ~73-75% | Tied |
| MdEval (Multi-Lang) | 75.2% | Lower | Qwen2.5-Coder |
| HumanEval | 92.7% | ~90% | Qwen2.5-Coder |
| General Coding | Competitive | Competitive | Tied |

### Key Distinctions

```
Qwen2.5-Coder-32B-Instruct:
  ✓ Open-source (Apache 2.0)
  ✓ Free to use
  ✓ Self-hostable
  ✓ Full model access
  ✓ No API costs
  ✓ No usage restrictions

GPT-4o:
  ✗ Proprietary
  ✗ API costs required
  ✗ Usage limits
  ✗ No model access
```

## Quantization Support

### Available Formats

Qwen2.5-Coder provides official quantized models in three formats:

#### 1. GGUF (for llama.cpp)

**Quantization Levels**:
- q2_K, q3_K_M, q4_0, q4_K_M
- q5_0, q5_K_M, q6_K, q8_0

**Platform**: Hugging Face (available for all sizes)

#### 2. AWQ (Activation-aware Weight Quantization)

**Available**: All model sizes
**Support**: Hugging Face transformers

#### 3. GPTQ

**Available**: All model sizes
**Support**: Hugging Face transformers

### Benefits

```
Quantization Impact:
├── Memory: 2-8× reduction
├── Speed: Minimal degradation
├── Accuracy: 95-99% retention (depending on level)
└── Accessibility: Enables consumer hardware deployment
```

## Hardware Requirements

### GPU Requirements (Full Precision/BF16)

#### Qwen2.5-Coder-32B
```
Recommended: 24GB+ VRAM
  - NVIDIA RTX 3090
  - NVIDIA RTX 4090
  - Future: RTX 5090 (32GB VRAM)

Quantized (Q4KM): 24GB VRAM minimum
  - RTX 3090, RTX 4090
```

#### Qwen2.5-Coder-7B / 14B
```
Comfortable Execution: 24GB VRAM

GPTQ Quantized (7B): 6GB VRAM minimum
  - GTX 1660
  - RTX 2060
  - RTX 3050/3060
  - AMD 5700 XT
```

### Inference Speed (Tokens per Second)

#### Qwen2.5-Coder-32B
```
Hardware                Speed (tok/s)
──────────────────────────────────────
RTX 3090 (Q4KM)        37-40 tok/s
Mobile RTX 5090 (24GB) ~25 tok/s
```

#### Qwen2.5-Coder-7B
```
Hardware                        Speed (tok/s)
────────────────────────────────────────────
GPU (various)                   20-30+ tok/s
CPU (Ryzen 5 5600X, 4-bit)      ~9 tok/s
CPU (32GB+ RAM, full precision) 1-3 tok/s
```

**Performance Notes**:
- GPU deployment strongly recommended for production
- Quantization enables consumer hardware deployment
- Context length affects inference speed

## Disclosed Limitations

### 1. Context Window Issues

```
Challenges:
├── Breakdown at Limits: Output becomes nonsense near 132K limit
├── Tool Limitations: Some tools cap at 33K instead of 128K
└── Management Required: Careful input size monitoring necessary
```

### 2. Dataset Coverage

```
Limitations:
├── Training Cutoff: GitHub repos only through February 2024
├── Specialization Gaps: May not cover highly niche domains
└── 5.5T Tokens: Cannot cover all possible scenarios
```

### 3. Semantic Reasoning

```
Areas for Improvement:
├── Program Semantics: Reasoning about code meaning needs work
├── Verification: Outputs not always verifiably correct
└── Creativity: Exploratory coding in novel tasks can be limited
```

### 4. Usage Constraints

```
Constraints:
├── Base Models: Not for direct conversation without post-training
├── Context (Default): 32,768 may limit very large codebases
└── YARN in vLLM: Static scaling may impact shorter text performance
```

## Key Innovations

### 1. Massive Code-Specialized Training

```
Scale Comparison:
  CodeQwen1.5: Unknown token count
        ↓
  Qwen2.5-Coder: 5.5 trillion tokens
        ↓
  Result: Significantly improved generation, reasoning, fixing
```

### 2. Balanced Data Mixing

**Innovation**: Maintain general capabilities while specializing

```
Mix Strategy:
  70% Code → Specialization
  20% Text-Code → Grounding
  10% Math → Reasoning
  ═══════
  Result: Specialized yet versatile
```

### 3. Synthetic Data with Executor Validation

**Process**:
```
CodeQwen1.5 → Generate synthetic code
      ↓
Executor → Validate executability
      ↓
Training → Only validated code included
      ↓
Result → Mitigates hallucination risks
```

### 4. 4-Stage Text-Code Filtering

**Impact**: Quality improvement from 41.6% → 46.8% on HumanEval/MBPP

### 5. Repository-Level Understanding

**Special Tokens**: `<|repo_name|>`, `<|file_sep|>`

**Enables**:
- Multi-file context
- Cross-file dependencies
- Project-wide understanding

### 6. GPT-4o Performance Parity (Open-Source)

**Achievement**: First open-source model to match GPT-4o on coding tasks

```
Comparison:
  Performance: Comparable to GPT-4o
  License: Apache 2.0 (fully open)
  Cost: Free
  Access: Full model weights

Impact: Democratizes GPT-4o-level coding capabilities
```

## Comparison: Qwen2.5-Coder vs CodeQwen1.5

| Aspect | CodeQwen1.5 | Qwen2.5-Coder | Improvement |
|--------|-------------|---------------|-------------|
| Training Tokens | Unknown | 5.5 Trillion | Massive increase |
| Model Sizes | Fewer variants | 6 sizes (0.5B-32B) | More options |
| Languages | Limited | 92 languages | Much wider |
| Context Window | Smaller | 128K tokens | Extended |
| Performance | Baseline | GPT-4o parity | Breakthrough |
| Synthetic Data | Basic | Executor-validated | More reliable |

## Comparison: Qwen2.5-Coder vs Qwen2.5 (General)

| Feature | Qwen2.5 (General) | Qwen2.5-Coder | Difference |
|---------|-------------------|---------------|------------|
| Training Focus | General purpose | Code-specialized | 70% code data |
| FIM Support | No | Yes | Code completion |
| Repo Tokens | No | Yes | Multi-file context |
| HumanEval (7B) | ~82% | 88.4% | +6.4% improvement |
| Aider (32B) | Lower | 73.7% | Code repair focus |
| Math (32B) | 83.1% | 57.2% | General model better |

**Key Insight**: Qwen2.5-Coder trades some general capabilities for dramatically improved coding performance.

## Use Cases

### 1. Code Generation
```python
# Example: Generate Python function
User: "Write a function to find the longest palindrome in a string"
Model: Generates optimized O(n^2) or Manacher's O(n) algorithm
```

### 2. Code Completion (FIM)
```python
# Example: IDE integration
def calculate_fibonacci(n):
    <|fim_middle|>  # Model completes: if n <= 1: return n
                    #                   return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
```

### 3. Code Repair
```python
# Example: Debug existing code
User: "This function has a bug when n=0"
Model: Identifies edge case, suggests fix with explanation
```

### 4. Repository Understanding
```
# Example: Multi-file refactoring
User: "Rename DatabaseConnection class across all files"
Model: Identifies all usages, suggests changes in 15 files
```

### 5. Multi-Language Translation
```
# Example: Port algorithm
User: "Convert this Python sorting algorithm to Rust"
Model: Translates with idiomatic Rust patterns
```

### 6. Documentation Generation
```
# Example: Add docstrings
User: "Add comprehensive docstrings to this module"
Model: Generates NumPy/Google-style documentation
```

## Deployment Options

### 1. Local Inference (llama.cpp)
```bash
# Download GGUF model
wget https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF

# Run with llama.cpp
./main -m qwen2.5-coder-7b-instruct-q4_k_m.gguf \
       -n 512 \
       -p "Write a Python function to sort a list"
```

### 2. Hugging Face Transformers
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

prompt = "Write a function to calculate factorial"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 3. vLLM (Production Serving)
```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-Coder-32B-Instruct")
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)

outputs = llm.generate(
    ["Write a REST API endpoint in FastAPI"],
    sampling_params
)
```

### 4. Ollama (Easy Local Setup)
```bash
# Pull model
ollama pull qwen2.5-coder:7b

# Run interactively
ollama run qwen2.5-coder:7b
>>> Write a function to merge two sorted arrays
```

### 5. API Deployment
```python
# Using Hugging Face Inference API
import requests

API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-7B-Instruct"
headers = {"Authorization": "Bearer YOUR_TOKEN"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

output = query({
    "inputs": "Write a binary search function in C++",
})
```

## Information Disclosure Status

### DISCLOSED ✓

- Model sizes and parameter counts (6 sizes: 0.5B-32B)
- Training token count (5.5 trillion)
- Data type categories (5 types)
- Data mixing ratios (70% code, 20% text, 10% math)
- Architecture components (RoPE, SwiGLU, RMSNorm, GQA)
- Benchmark results on major benchmarks (HumanEval, Aider, etc.)
- Context window specifications (132K with YARN)
- Tokenizer vocabulary size (151,646)
- Special tokens and their purposes
- Quantization availability (GGUF, AWQ, GPTQ)
- Licensing terms (Apache 2.0 for most sizes)
- Post-training methodology overview (SFT, DPO, GRPO)
- Limitations and weaknesses

### NOT DISCLOSED / Partially Disclosed ✗

- **Complete list of 92 programming languages**: Only examples provided (Python, Java, C++, Ruby, Rust, Haskell, Racket, JavaScript)
- **Exact SFT/RLHF dataset composition**: "~1 million examples" mentioned but no details
- **Specific SWE-Bench scores**: Not provided for Qwen2.5-Coder (only for Qwen3-Coder)
- **BigCodeBench exact scores**: Mentioned as SOTA but numbers not detailed
- **Training infrastructure**: GPU count, training time, compute budget
- **Detailed hyperparameters**: Learning rates, batch sizes, optimizer settings
- **Full benchmark tables**: Some referenced (Tables 6, 17) but not in web sources
- **Exact synthetic data prompts**: Methodology described but prompts not shared
- **Data cleaning rules**: "Meticulous" cleaning described but specific criteria not detailed
- **Reward model details**: For RLHF/GRPO stages not disclosed
- **MBPP exact scores**: Mentioned but specific numbers not provided

## Evolution: CodeQwen → Qwen2.5-Coder → Qwen3-Coder

```
Timeline:
  CodeQwen1.5 (2024 early)
      ↓
      • Basic code generation
      • Limited language support
      • Smaller scale
      ↓
  Qwen2.5-Coder (Sept 2024)
      ↓
      • 5.5T tokens training
      • 92 languages
      • GPT-4o parity
      • 128K context
      • Apache 2.0 license
      ↓
  Qwen3-Coder (2024 late/2025)
      ↓
      • Agent RL training
      • 20,000 parallel environments
      • SWE-Bench: 67-81.6%
      • Agentic coding workflows
      • Repository-scale context (256K native, 1M with YARN)
```

## Community Reception and Adoption

### Strengths Highlighted by Community

1. **Performance**: Matches GPT-4o while being open-source
2. **Accessibility**: Apache 2.0 license enables commercial use
3. **Versatility**: Strong across multiple programming languages
4. **Efficiency**: Smaller models (7B, 14B) punch above their weight
5. **Quantization**: Official GGUF/AWQ/GPTQ support for consumer hardware

### Common Use Cases

1. **IDE Integration**: Code completion and suggestions
2. **Code Review**: Automated review and bug detection
3. **Documentation**: Auto-generating docstrings and comments
4. **Translation**: Porting code between languages
5. **Learning**: Educational tool for programming

### Deployment Trends

- **Local deployment**: Popular for privacy-sensitive code
- **vLLM serving**: Production API deployments
- **Ollama**: Easy experimentation and prototyping
- **Fine-tuning**: Companies fine-tuning on proprietary codebases

## Future Directions

### Potential Improvements (Disclosed Limitations)

1. **Semantic Reasoning**: Deeper understanding of program semantics
2. **Verification**: More verifiably correct outputs
3. **Creativity**: More inventive approaches in creative coding tasks
4. **Context Management**: Better handling of context window limits
5. **Domain Coverage**: Expanding to more specialized programming domains

### Successor: Qwen3-Coder

Already released (late 2024/early 2025) with:
- **Agent RL**: Long-horizon reinforcement learning
- **20K Parallel Environments**: Massive-scale agentic training
- **SWE-Bench**: 67-81.6% (benchmark for real-world coding tasks)
- **Repository-Scale**: 256K native, 1M with YARN

## References and Resources

### Official Papers
- [Qwen2.5-Coder Technical Report (arXiv:2409.12186)](https://arxiv.org/abs/2409.12186)
- [Qwen2.5-Coder Technical Report PDF](https://arxiv.org/pdf/2409.12186)
- [Qwen2.5-Coder Technical Report HTML v3](https://arxiv.org/html/2409.12186v3)
- [Qwen2.5 Technical Report (arXiv:2412.15115)](https://arxiv.org/abs/2412.15115)

### Official Blogs
- [Qwen2.5-Coder: Code More, Learn More!](https://qwenlm.github.io/blog/qwen2.5-coder/)
- [Qwen2.5-Coder Series: Powerful, Diverse, Practical](https://qwenlm.github.io/blog/qwen2.5-coder-family/)
- [Qwen2.5: A Party of Foundation Models!](https://qwenlm.github.io/blog/qwen2.5/)

### Model Cards (Hugging Face)
- [Qwen/Qwen2.5-Coder-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
- [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct)
- [Qwen/Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
- [Qwen/Qwen2.5-Coder-32B (Base)](https://huggingface.co/Qwen/Qwen2.5-Coder-32B)
- [Qwen/Qwen2.5-Coder-7B (Base)](https://huggingface.co/Qwen/Qwen2.5-Coder-7B)

### GitHub Repositories
- [QwenLM/Qwen3-Coder](https://github.com/QwenLM/Qwen3-Coder) (includes Qwen2.5-Coder examples)

### Documentation
- [Qwen Key Concepts](https://qwen.readthedocs.io/en/latest/getting_started/concepts.html)
- [Qwen GPTQ Documentation](https://qwen.readthedocs.io/en/latest/quantization/gptq.html)

### Community Resources
- [Ollama Qwen2.5-Coder](https://ollama.com/library/qwen2.5-coder)
- [Hardware Corner - Qwen LLM Database](https://www.hardware-corner.net/llm-database/Qwen/)

## Summary

Qwen2.5-Coder represents a **breakthrough in open-source code generation**, achieving performance parity with proprietary GPT-4o while remaining fully accessible under Apache 2.0 license. With 5.5 trillion tokens of code-specialized training across 92 programming languages, the model excels at code generation (HumanEval: 92.7% for 32B), code repair (Aider: 73.7%, matching GPT-4o), and multi-language coding tasks. The series spans six model sizes (0.5B to 32B), enabling deployment from edge devices to production servers, with official quantization support (GGUF, AWQ, GPTQ) for consumer hardware accessibility. Key innovations include balanced data mixing (70% code, 20% text-code grounding, 10% math), executor-validated synthetic data generation, repository-level understanding with special tokens, and 128K context support via YARN. The flagship 32B model achieves state-of-the-art performance among open-source models on five code completion benchmarks and ranks first on MdEval multi-language code repair, while smaller 7B and 14B variants outperform larger proprietary models on many tasks. Qwen2.5-Coder democratizes GPT-4o-level coding capabilities, enabling privacy-preserving local deployment and cost-free commercial use, marking a significant milestone in the democratization of AI-assisted programming.
