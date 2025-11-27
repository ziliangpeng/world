# LLM Benchmarks and Evaluation

Benchmarks are how the field measures progress—but they're imperfect proxies for what we actually care about. This document covers the evolution of LLM evaluation, what major benchmarks measure, their limitations, and how to interpret results critically.

---

## The Evaluation Challenge

### What We Want to Measure

| Goal | Challenge |
|------|-----------|
| General intelligence | Hard to define, context-dependent |
| Helpfulness | Subjective, task-dependent |
| Truthfulness | Requires ground truth |
| Safety | Adversarial evaluation needed |
| Reasoning | Many types, hard to isolate |

### What Benchmarks Actually Measure

- Performance on specific, static test sets
- Often narrow slices of capability
- Subject to gaming and overfitting
- May not reflect real-world usage

---

## Historical Evolution

### Phase 1: NLP Task Benchmarks (2018-2020)

**GLUE / SuperGLUE**

Standard NLP evaluation:
- Sentiment analysis (SST-2)
- Natural language inference (MNLI, RTE)
- Sentence similarity (STS-B)
- Question answering (BoolQ)

**Limitation**: Saturated quickly, models exceeded human performance.

### Phase 2: Knowledge and Reasoning (2020-2022)

**[MMLU](https://arxiv.org/abs/2009.03300)** (Massive Multitask Language Understanding, 2020)

57 subjects from STEM to humanities:
- Elementary math to professional law
- Multiple choice format
- 14,042 questions total

**Significance**: Became the standard knowledge benchmark.

**[GSM8K](https://arxiv.org/abs/2110.14168)** (Grade School Math, 2021)

8,500 math word problems requiring multi-step reasoning:
```
Problem: A store sells apples for $2 each and oranges for $3 each.
If Sarah buys 4 apples and 3 oranges, how much does she spend?
Answer: 4 × $2 + 3 × $3 = $8 + $9 = $17
```

**[HumanEval](https://arxiv.org/abs/2107.03374)** (2021)

164 Python programming problems with test cases:
```python
def solve_task(prompt):
    """Write a function that returns the sum of all even numbers in a list."""
    # Model generates solution
    # Evaluated by running test cases
```

### Phase 3: Comprehensive Evaluation (2022-2023)

**[BIG-bench](https://arxiv.org/abs/2206.04615)** (2022)

204 diverse tasks from 450+ authors:
- JSON parsing, emoji understanding, causal reasoning
- Many "canary" tasks to detect memorization
- Designed to be hard to saturate

**[HELM](https://crfm.stanford.edu/helm/)** (Stanford, 2022)

Holistic Evaluation of Language Models:
- 42 scenarios across 7 metrics
- Standardized prompting and evaluation
- Emphasis on calibration, robustness, fairness

### Phase 4: Real-World Evaluation (2023-2024)

**[Chatbot Arena](https://chat.lmsys.org/)** (LMSYS, 2023)

Human preference evaluation at scale:
- Users compare anonymous model outputs
- Elo rating system
- 1M+ human votes

**MT-Bench** (2023)

Multi-turn conversation evaluation:
- 80 carefully curated questions
- Tests instruction following, reasoning, coding
- GPT-4 as automated judge

**[AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/)** (2023)

Automated evaluation using GPT-4:
- Win rate vs reference (text-davinci-003)
- Fast, reproducible
- Correlates with human preference

---

## Major Benchmarks

### Knowledge Benchmarks

| Benchmark | Questions | Format | What It Tests |
|-----------|-----------|--------|---------------|
| **MMLU** | 14,042 | 4-choice MC | Broad knowledge (57 subjects) |
| **ARC** | 7,787 | MC | Science reasoning (grade school) |
| **TriviaQA** | 95K | Open QA | Factual knowledge |
| **NaturalQuestions** | 307K | Open QA | Wikipedia knowledge |

**MMLU Details**:
```
Categories:
- STEM: physics, chemistry, math, CS, engineering
- Humanities: history, philosophy, law
- Social Sciences: economics, psychology, sociology
- Other: professional exams (medical, legal, accounting)

Evaluation: 5-shot prompting, exact match
```

### Reasoning Benchmarks

| Benchmark | Questions | Format | What It Tests |
|-----------|-----------|--------|---------------|
| **GSM8K** | 8,500 | Math word problems | Arithmetic reasoning |
| **MATH** | 12,500 | Competition math | Advanced math |
| **HellaSwag** | 70K | Sentence completion | Commonsense |
| **WinoGrande** | 44K | Pronoun resolution | Commonsense |
| **BBH** | 6,511 | Diverse tasks | Chain-of-thought |

**GSM8K Evaluation**:
```
Standard: Extract final number, exact match
CoT: Allow step-by-step reasoning before answer
```

### Code Benchmarks

| Benchmark | Problems | Languages | Evaluation |
|-----------|----------|-----------|------------|
| **HumanEval** | 164 | Python | Functional correctness |
| **MBPP** | 974 | Python | Functional correctness |
| **MultiPL-E** | 164× | Multiple | HumanEval translated |
| **DS-1000** | 1,000 | Python | Data science tasks |

**HumanEval Metrics**:
```
pass@k: Probability of at least one correct solution in k attempts

pass@1: Single attempt accuracy
pass@10: Best of 10 attempts
pass@100: Best of 100 attempts (measures capability ceiling)
```

### Instruction Following

| Benchmark | Questions | Format | Evaluation |
|-----------|-----------|--------|------------|
| **MT-Bench** | 80 | Multi-turn | GPT-4 judge (1-10) |
| **AlpacaEval** | 805 | Single-turn | GPT-4 win rate |
| **IFEval** | 541 | Verifiable | Rule-based check |

**MT-Bench Categories**:
- Writing, roleplay
- Reasoning, math
- Coding
- Extraction
- STEM, humanities

### Safety Benchmarks

| Benchmark | Focus | Evaluation |
|-----------|-------|------------|
| **ToxiGen** | Toxicity generation | Toxicity classifier |
| **TruthfulQA** | Truthfulness | MC + open generation |
| **BBQ** | Bias in QA | Accuracy + bias score |
| **HarmBench** | Harmful behaviors | Attack success rate |

---

## Benchmark Limitations

### 1. Contamination

Training data may contain benchmark questions:

```python
# Example: GSM8K question found in Common Crawl
# Model memorizes answer, doesn't learn to reason
```

**Detection methods**:
- n-gram overlap analysis
- Performance on variants
- Canary strings

**Mitigation**:
- Use held-out test sets
- Evaluate on new benchmarks
- Test generalization to variations

### 2. Gaming and Overfitting

| Gaming Strategy | Example |
|-----------------|---------|
| Train on test set | Include benchmark in training data |
| Prompt optimization | Engineer prompts for specific benchmark |
| Format overfitting | Optimize for MC even if poor at generation |
| Selective reporting | Report best number, hide failures |

### 3. Narrow Scope

Benchmarks test specific capabilities:
```
MMLU: Factual recall ✓, Reasoning ✓, Creativity ✗, Safety ✗
HumanEval: Code correctness ✓, Code quality ✗, Debugging ✗
MT-Bench: Helpfulness ✓, Long-term coherence ✗
```

### 4. Format Sensitivity

Small changes dramatically affect scores:
```
5-shot vs 0-shot: ±10% on MMLU
Chain-of-thought vs direct: ±20% on GSM8K
Prompt wording variations: ±5% on most benchmarks
```

### 5. Ceiling Effects

Top models cluster near 100%:
```
GPT-4: 86.4% MMLU
Claude 3 Opus: 86.8% MMLU
Gemini Ultra: 90.0% MMLU

Difference: <4% for very different capabilities
```

---

## How to Interpret Results

### Red Flags

| Signal | Concern |
|--------|---------|
| Much higher than similar-size models | Possible contamination |
| Great on benchmarks, poor in practice | Overfitting |
| Only reports best numbers | Selective reporting |
| No evaluation on new benchmarks | May not generalize |

### Healthy Skepticism

```python
def evaluate_benchmark_claim(claim, context):
    checks = [
        "Was evaluation done by independent party?",
        "Are prompts and settings disclosed?",
        "Are results consistent with similar models?",
        "Does model perform well on related benchmarks?",
        "How does it perform on real user tasks?"
    ]
    # If multiple checks fail, be skeptical
```

### What Actually Matters

| Benchmark Score | Real-World Relevance |
|-----------------|---------------------|
| MMLU | General knowledge tasks |
| GSM8K | Basic math applications |
| HumanEval | Simple coding tasks |
| MT-Bench | Conversation quality |
| Chatbot Arena | User preference |

**Best signal**: Chatbot Arena (real human preference on diverse tasks)

---

## Evaluation Best Practices

### For Model Developers

1. **Multiple benchmarks**: Don't optimize for single metric
2. **Ablation studies**: Show what contributes to performance
3. **Contamination checks**: Verify training data doesn't overlap
4. **Real-world evaluation**: Test on actual use cases
5. **Error analysis**: Understand failure modes

### For Model Users

1. **Test on your tasks**: Benchmarks may not reflect your needs
2. **Compare fairly**: Same prompting, same conditions
3. **Consider task distribution**: Your tasks may differ from benchmarks
4. **Human evaluation**: For subjective quality
5. **Cost-performance tradeoff**: Include inference cost

### Evaluation Code

```python
import lm_eval

# Standardized evaluation with lm-eval-harness
results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained=meta-llama/Llama-2-7b-hf",
    tasks=["mmlu", "gsm8k", "hellaswag"],
    num_fewshot=5,
    batch_size="auto"
)

# Parse results
for task, metrics in results["results"].items():
    print(f"{task}: {metrics['acc']:.2%}")
```

---

## Modern Leaderboards

### Open LLM Leaderboard (HuggingFace)

Standardized evaluation on 6 benchmarks:
- ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K

**Pros**: Reproducible, open
**Cons**: Can be gamed, narrow scope

### Chatbot Arena (LMSYS)

Human preference at scale:
- Anonymous pairwise comparison
- Elo rating system
- ~1M+ votes

**Pros**: Real human preference, hard to game
**Cons**: Biased toward chat, doesn't test all capabilities

### HELM (Stanford)

Holistic evaluation across scenarios:
- Multiple metrics per scenario
- Standardized conditions
- Transparency reports

**Pros**: Comprehensive, standardized
**Cons**: Resource-intensive, may lag new models

---

## Future of Evaluation

### Near-term (2025)

1. **Dynamic benchmarks**: Regularly updated to prevent contamination
2. **Agent benchmarks**: Evaluate tool use, planning, execution
3. **Multi-modal**: Joint text-image-video evaluation
4. **Safety focus**: More comprehensive harm evaluation

### Research Frontiers

1. **Capability elicitation**: How to properly prompt for max capability?
2. **Robust evaluation**: Resist gaming and overfitting
3. **Emergent capability detection**: What new abilities appear at scale?
4. **Real-world proxies**: Better correlation with actual usefulness

### Open Questions

1. **What to measure**: What capabilities actually matter?
2. **How to measure**: Automated vs human evaluation?
3. **Gaming resistance**: Can we make ungameable benchmarks?
4. **Generalization**: Do benchmark gains transfer to real tasks?

---

## Sources

### Benchmark Papers
- [MMLU: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- [GSM8K: Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- [HumanEval: Evaluating Large Language Models on Code](https://arxiv.org/abs/2107.03374)
- [BIG-bench: Beyond the Imitation Game](https://arxiv.org/abs/2206.04615)

### Evaluation Frameworks
- [HELM: Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
- [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Chatbot Arena](https://arxiv.org/abs/2403.04132)

### Critique and Analysis
- [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109)
- [Challenges and Applications of LLM Evaluation](https://arxiv.org/abs/2405.00267)
- [Data Contamination in LLMs](https://arxiv.org/abs/2311.04850)

### Leaderboards
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Chatbot Arena Leaderboard](https://chat.lmsys.org/)
- [HELM Leaderboard](https://crfm.stanford.edu/helm/latest/)
