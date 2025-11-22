# Code Llama

**Release Date**: August 24, 2023

Specialized version of Llama 2 for code generation, with 100K token context and Fill-in-the-Middle capability.

## Model Sizes and Variants

**Base Models**: 7B, 13B, 34B, 70B parameters

**Three Variants** (all available in all sizes):
1. **Code Llama (Base)**: Foundation for general code tasks
2. **Code Llama - Python**: Python-specialized versions
3. **Code Llama - Instruct**: Instruction-following for code tasks

**Total**: 12 models (4 sizes × 3 variants)

## Architecture Modifications

**Base**: Built on Llama 2, initialized with pretrained weights

### Extended Context: 4K → 100K Tokens

**Major Innovation**: Context expanded from 4,096 to **100,000 tokens**

**How**:
- Modified RoPE parameters for long sequences
- Trained on 16K token sequences
- Strong extrapolation up to 100K tokens

**Why This Matters**:
- Can process entire repositories
- Understand large codebases in context
- Handle long code files without truncation

### Fill-in-the-Middle (FIM)

**Supported Models**:
- ✅ 7B, 13B, 70B (base and instruct)
- ❌ NOT supported: 34B models, Python variants

**What FIM Does**:
- Enables code completion and insertion (not just generation)
- Can fill gaps in existing code
- Ideal for IDE integration
- Uses causal infilling alongside autoregressive prediction

**Example Use**:
```python
def calculate_sum(a, b):
    # <FILL> - Model fills in the implementation
    return result
```

## Training Approach

### Multi-Stage Specialization

**Stage 1: Start with Llama 2**
- Initialize with Llama 2 pretrained weights
- Already saw 80B code tokens during Llama 2 training

**Stage 2: Code Specialization**
- Train on 500B tokens of code (1T for 70B model)
- Focus on programming languages and code patterns

**Stage 3: Long Context Fine-Tuning**
- Separate stage to extend context to 100K
- Cost-efficient approach (extend context after main training)

**Stage 4: Instruction Fine-Tuning**
- For Instruct variants only
- Teach model to follow coding instructions

### Training Data Composition

**500B tokens total** (1T for 70B):
- **85%**: Open-source GitHub code
- **8%**: Natural language about code (documentation, discussions)
- **7%**: General natural language

**Programming Languages**:
- Python, C++, Java, JavaScript, TypeScript
- PHP, C#, Bash, and more

### Multitask Objective

**Models 7B, 13B, 70B** use both:
- Autoregressive prediction (standard next-token prediction)
- Causal infilling (Fill-in-the-Middle)

**Model 34B**: Autoregressive only (no FIM)

## Key Innovations

1. **First Major Open-Source Code Model from Meta**
   - Competitive with GitHub Copilot's underlying models

2. **Successfully Extended Context to 100K Tokens**
   - With RoPE modifications
   - Separate fine-tuning stage (cost-efficient)

3. **Multi-Stage Specialization Pipeline**
   - Pretrain → Code → Long Context → Instruct
   - Each stage optimized separately

4. **Fill-in-the-Middle for Real-Time Integration**
   - Enables IDE autocomplete
   - Better than pure generation models

5. **Cost-Efficient Long-Context Training**
   - Extend context as separate stage
   - Don't need to train from scratch with long context

## Variant Differences

### Code Llama (Base)
- General code generation
- Multiple programming languages
- Foundation for other variants

### Code Llama - Python
- Specialized for Python
- 100B additional Python-specific tokens
- Better Python completion and generation
- **No FIM support** (trade-off for specialization)

### Code Llama - Instruct
- Instruction-following for coding tasks
- Natural language → Code
- Explain code, write tests, debug
- **Has FIM support** (except 34B)

## Use Cases

**Code Completion** (FIM models):
- Real-time autocomplete in IDEs
- Fill in function bodies
- Complete partially written code

**Code Generation** (All models):
- Natural language → Code
- "Write a function to sort a list"
- Generate boilerplate

**Documentation**:
- Docstring generation
- Code comments
- README files

**Code Understanding**:
- Explain existing code
- Debugging assistance
- Code review

**Code Translation**:
- Convert between programming languages
- Refactor code
- Modernize legacy code

## Performance

- Competitive with GitHub Copilot (Codex-based)
- Code Llama-34B outperformed GPT-3.5 on HumanEval
- Code Llama-Python-70B: Best open Python model at release
- Strong on long-context code understanding

## Links

- **Paper**: [Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950)
- **Blog**: [Introducing Code Llama](https://ai.meta.com/blog/code-llama-large-language-model-coding/)
- **Hugging Face**:
  - Base: [CodeLlama-7b-hf](https://huggingface.co/meta-llama/CodeLlama-7b-hf), [CodeLlama-13b-hf](https://huggingface.co/meta-llama/CodeLlama-13b-hf), [CodeLlama-34b-hf](https://huggingface.co/meta-llama/CodeLlama-34b-hf), [CodeLlama-70b-hf](https://huggingface.co/meta-llama/CodeLlama-70b-hf)
  - Python: [CodeLlama-7b-Python-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Python-hf), [CodeLlama-13b-Python-hf](https://huggingface.co/meta-llama/CodeLlama-13b-Python-hf), etc.
  - Instruct: [CodeLlama-7b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-7b-Instruct-hf), [CodeLlama-13b-Instruct-hf](https://huggingface.co/meta-llama/CodeLlama-13b-Instruct-hf), etc.

## Integration and Ecosystem

**IDE Extensions**:
- VS Code extensions using Code Llama
- JetBrains IDE plugins
- Vim/Emacs integrations

**Inference Frameworks**:
- llama.cpp (CPU inference)
- vLLM (fast GPU inference with FIM)
- Text Generation Inference

**Quantization**:
- GGUF formats for consumer hardware
- INT8/INT4 quantization
- Run 7B/13B on laptops

## Comparison to Alternatives

| Model | Size | Context | FIM | Open |
|-------|------|---------|-----|------|
| **Code Llama** | 7-70B | **100K** | ✅ | ✅ |
| GitHub Copilot | ~12B | Unknown | ✅ | ❌ |
| StarCoder | 15B | 8K | ✅ | ✅ |
| Codex (GPT-3.5 based) | ~175B | ~4K | ❌ | ❌ |

Code Llama's 100K context was unique at release.

## Legacy and Impact

Code Llama democratized AI-powered coding:
1. **Open-Source Alternative** to GitHub Copilot
2. **100K Context** enabled whole-repository understanding
3. **Multiple Sizes** for different hardware
4. **FIM** made IDE integration practical
5. **Specialized Variants** (Python, Instruct) showed path to domain-specific models

It proved open-source code models could compete with proprietary alternatives, paving the way for StarCoder2, DeepSeek Coder, and others.

## Limitations

- **No 34B FIM**: 34B models don't support Fill-in-the-Middle
- **No Python FIM**: Python variants trade FIM for specialization
- **Based on Llama 2**: Uses older architecture (MHA for 7B/13B, not GQA)
- **SentencePiece Tokenizer**: Older tokenizer, not TikToken like Llama 3

## Future

Code Llama was based on Llama 2 (2023). Meta hasn't yet released a Llama 3-based code model, but the community has fine-tuned Llama 3 for code. A potential "Code Llama 3" would likely have:
- GQA across all sizes
- TikToken tokenizer (128K vocab)
- Better base capabilities
- Potentially longer context
