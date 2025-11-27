# Supervised Fine-Tuning (SFT)

Supervised Fine-Tuning transforms a base language model into an assistant that follows instructions and engages in conversations. While pre-training teaches the model *language*, SFT teaches it *behavior*—how to respond helpfully, follow formats, and interact naturally. This document covers the evolution of SFT techniques and the practical details of building instruction-following models.

---

## Why SFT Matters

A base model (pretrained on next-token prediction) has remarkable capabilities buried within it, but doesn't know how to express them as an assistant:

| Query | Base Model Response | SFT Model Response |
|-------|--------------------|--------------------|
| "What is 2+2?" | "2+2 = 4. And 3+3 = 6. And..." | "2+2 equals 4." |
| "Write a haiku about AI" | "Here is another haiku..." (continuation) | "Silicon minds think / Algorithms dream of us / Code becomes aware" |
| "Help me debug this code" | Random code continuation | Analyzes code, identifies bug, suggests fix |

**The transformation**: SFT unlocks latent capabilities by teaching the model the *format* of helpful responses.

---

## Historical Evolution

### Phase 1: Task-Specific Fine-Tuning (2018-2020)

**BERT-style fine-tuning**: Add task-specific head, fine-tune on labeled data.

```
[CLS] The movie was great [SEP] → [Sentiment: Positive]
```

**Limitations**:
- One model per task
- Requires task-specific architecture changes
- Doesn't leverage language generation capabilities

### Phase 2: Prompt-Based Fine-Tuning (2020-2021)

**GPT-3 few-shot learning** showed models could adapt via prompts without fine-tuning:

```
Translate English to French:
English: Hello
French: Bonjour
English: Goodbye
French:
```

**Limitations**:
- Uses context window for examples (expensive)
- Inconsistent performance
- No persistent improvement

### Phase 3: Instruction Tuning (2021-2022)

**[FLAN](https://arxiv.org/abs/2109.01652)** (Google, September 2021)

First systematic instruction tuning:
- 62 NLP datasets converted to instruction format
- Single model handles all tasks
- Better zero-shot generalization than few-shot GPT-3

```
# FLAN instruction format
Input: Translate "Hello" to French
Output: Bonjour

Input: Is this review positive or negative? "Great movie!"
Output: Positive
```

**[T0](https://arxiv.org/abs/2110.08207)** (BigScience, October 2021)

Prompt-tuning across tasks with natural language prompts.

**[InstructGPT](https://arxiv.org/abs/2203.02155)** (OpenAI, March 2022)

Combined SFT with RLHF:
1. Collect demonstration data from human labelers
2. Fine-tune GPT-3 on demonstrations (SFT)
3. Further align with RLHF

This became the standard recipe: **SFT → RLHF**.

### Phase 4: Open Instruction Tuning (2023)

**[Self-Instruct](https://arxiv.org/abs/2212.10560)** (December 2022)

Generated instruction data using GPT-3 itself:
1. Start with 175 seed tasks
2. Generate new instructions via LLM
3. Generate outputs via LLM
4. Filter for quality

Enabled instruction tuning without expensive human annotation.

**[Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)** (March 2023)

Applied Self-Instruct to generate 52K examples, fine-tuned LLaMA-7B:
- Cost: ~$500 for data generation
- Result: Competitive with davinci-003
- Impact: Democratized instruction tuning

**[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)** (March 2023)

Fine-tuned LLaMA on 70K ShareGPT conversations:
- Used real user-ChatGPT conversations
- Achieved ~90% of ChatGPT quality
- Introduced ShareGPT format

### Phase 5: Scaled Instruction Data (2023-2024)

**[FLAN-T5, FLAN-PaLM](https://arxiv.org/abs/2210.11416)** (October 2022)

Scaled instruction tuning to 1,800+ tasks, showed continued improvement.

**[Orca](https://arxiv.org/abs/2306.02707)** (June 2023)

Used GPT-4 as teacher with detailed explanations:
- Rich reasoning traces, not just answers
- "Explanation tuning"
- Smaller models learned reasoning patterns

**[Zephyr](https://arxiv.org/abs/2310.16944)** (October 2023)

Combined UltraChat (synthetic conversations) + UltraFeedback for SFT + DPO:
- Mistral-7B base → Zephyr-7B-beta
- Outperformed 70B models on MT-Bench
- Demonstrated small model potential with good data

### Phase 6: Modern SFT Pipelines (2024)

Current state-of-the-art:

1. **Base model**: Strong pretrained model (LLaMA 3, Mistral, etc.)
2. **SFT data**: Mix of:
   - High-quality human demonstrations
   - Synthetic data from stronger models
   - Curated multi-turn conversations
3. **SFT training**: Full fine-tuning or LoRA
4. **Alignment**: DPO or RLHF for final polish

---

## SFT Data Formats

### Single-Turn Instructions

**Alpaca format**:
```json
{
  "instruction": "Explain photosynthesis",
  "input": "",
  "output": "Photosynthesis is the process by which plants..."
}
```

**With input context**:
```json
{
  "instruction": "Summarize the following text",
  "input": "The quick brown fox jumps over the lazy dog...",
  "output": "A fox jumped over a dog."
}
```

### Multi-Turn Conversations

**ShareGPT format**:
```json
{
  "conversations": [
    {"from": "human", "value": "What's the capital of France?"},
    {"from": "gpt", "value": "The capital of France is Paris."},
    {"from": "human", "value": "What's its population?"},
    {"from": "gpt", "value": "Paris has approximately 2.1 million residents..."}
  ]
}
```

**OpenAI messages format**:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

### Chat Templates

Models use special tokens to delimit turns. Common formats:

**ChatML** (OpenAI-style):
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi! How can I help?<|im_end|>
```

**Llama 2 Chat**:
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

Hello! [/INST] Hi! How can I help? </s><s>[INST] Thanks! [/INST]
```

**Llama 3 Chat**:
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi! How can I help?<|eot_id|>
```

**Mistral Instruct**:
```
<s>[INST] Hello! [/INST] Hi! How can I help?</s>[INST] Thanks! [/INST]
```

**Key principle**: The model only predicts assistant tokens; user/system tokens are masked from loss.

---

## Training Methodology

### Data Preparation

```python
def prepare_sft_example(example, tokenizer):
    """Convert conversation to training format."""

    # Apply chat template
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

    # Tokenize
    tokens = tokenizer(text, return_tensors="pt")

    # Create labels: -100 for non-assistant tokens (masked from loss)
    labels = tokens["input_ids"].clone()

    # Mask everything except assistant responses
    assistant_mask = create_assistant_mask(tokens, tokenizer)
    labels[~assistant_mask] = -100

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": labels
    }
```

### Full Fine-Tuning

Update all model parameters:

```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

training_args = TrainingArguments(
    output_dir="./llama-sft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
    learning_rate=2e-5,             # Lower than pretraining
    num_train_epochs=3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

### Parameter-Efficient Fine-Tuning (PEFT)

**[LoRA](https://arxiv.org/abs/2106.09685)** (Low-Rank Adaptation):

Instead of updating W, learn low-rank decomposition ΔW = BA:

```
Original: Y = XW
LoRA:     Y = XW + X(BA)   where B ∈ R^{d×r}, A ∈ R^{r×k}, r << d,k
```

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                    # Rank of decomposition
    lora_alpha=32,           # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

**[QLoRA](https://arxiv.org/abs/2305.14314)** (Quantized LoRA):

Combine 4-bit quantization with LoRA for memory efficiency:

```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
)

# Now add LoRA adapters
model = get_peft_model(model, lora_config)
```

**Comparison**:

| Method | Memory (7B) | Parameters Updated | Quality |
|--------|-------------|-------------------|---------|
| Full FT | ~60 GB | 100% | Best |
| LoRA r=16 | ~30 GB | ~0.06% | ~98% of full |
| QLoRA r=16 | ~10 GB | ~0.06% | ~95% of full |

### Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Learning rate | 1e-5 to 2e-5 | 10-100× lower than pretraining |
| Batch size | 32-128 | Effective, via accumulation |
| Epochs | 1-3 | Often 1 is enough |
| Warmup | 3-10% of steps | Less than pretraining |
| Weight decay | 0 or 0.01 | Less important than pretraining |
| LR schedule | Cosine | Standard |

### Training Tips

1. **Don't overtrain**: SFT data is small; overfitting is easy
2. **Monitor validation loss**: Stop when it increases
3. **Data quality > quantity**: 10K high-quality > 100K noisy
4. **Include diverse tasks**: Format variety prevents overfitting
5. **Preserve base capabilities**: Some general text helps

---

## SFT Data Sources

### Public Datasets

| Dataset | Size | Type | Quality |
|---------|------|------|---------|
| [Alpaca](https://github.com/tatsu-lab/stanford_alpaca) | 52K | Single-turn | Medium |
| [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) | 70K | Multi-turn | High |
| [FLAN Collection](https://github.com/google-research/FLAN) | 1.8M | Multi-task | Medium |
| [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1) | 161K | Multi-turn | High |
| [UltraChat](https://github.com/thunlp/UltraChat) | 1.5M | Multi-turn | Medium-High |
| [Capybara](https://huggingface.co/datasets/LDJnr/Capybara) | 16K | Multi-turn | High |

### Synthetic Data Generation

**From stronger model**:
```python
def generate_sft_data(prompts, teacher_model):
    """Generate SFT data using a teacher model."""
    examples = []
    for prompt in prompts:
        response = teacher_model.generate(prompt, temperature=0.7)
        examples.append({
            "instruction": prompt,
            "output": response
        })
    return examples
```

**Self-Instruct pipeline**:
1. Seed instructions (175 hand-written)
2. Generate new instructions
3. Classify task type
4. Generate instances (input-output pairs)
5. Filter for quality

**Evol-Instruct** (WizardLM):
Evolve simple instructions into complex ones:
- Add constraints
- Increase reasoning depth
- Combine multiple requirements

---

## Quality Considerations

### Data Curation

```python
def filter_sft_example(example):
    """Filter low-quality examples."""

    # Length filters
    if len(example["output"]) < 10:
        return False  # Too short
    if len(example["output"]) > 8000:
        return False  # Too long

    # Quality signals
    if example["output"].count("I cannot") > 0:
        return False  # Refusal
    if example["output"].count("I apologize") > 2:
        return False  # Over-apologetic

    # Deduplication
    if is_near_duplicate(example, seen_examples):
        return False

    return True
```

### Balancing

```python
# Balance task types
task_distribution = {
    "coding": 0.25,
    "writing": 0.20,
    "analysis": 0.20,
    "math": 0.15,
    "conversation": 0.10,
    "factual": 0.10,
}

# Balance difficulty
difficulty_distribution = {
    "easy": 0.30,
    "medium": 0.50,
    "hard": 0.20,
}
```

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Overfitting | Repeats training responses | More data, less training |
| Format collapse | Only outputs one format | Diverse task types |
| Capability loss | Worse at general tasks | Include general examples |
| Over-refusal | Refuses benign requests | Balance safety data |
| Verbosity | Unnecessarily long responses | Include concise examples |

---

## Evaluation

### Automatic Metrics

**Loss-based**:
```python
# Validation loss on held-out SFT data
val_loss = trainer.evaluate()["eval_loss"]
```

**Benchmark-based**:
- MT-Bench (multi-turn capability)
- AlpacaEval (instruction following)
- MMLU (knowledge retention)

### Human Evaluation

**Pairwise comparison**:
- Show two responses, ask which is better
- Calculate win rate vs baseline

**Likert rating**:
- Rate helpfulness 1-5
- Rate harmlessness 1-5
- Rate honesty 1-5

---

## Modern Best Practices

### Data Recipe

1. **Core**: 10-50K high-quality demonstrations
2. **Diversity**: Multiple task types, formats, lengths
3. **Quality**: Strong teacher models (GPT-4, Claude)
4. **Filtering**: Remove low-quality, deduplicate

### Training Recipe

1. **Method**: Full FT for max quality, LoRA for efficiency
2. **Duration**: 1-3 epochs, watch validation loss
3. **Learning rate**: 1e-5 to 5e-5
4. **Batch size**: Larger is better (32-128)

### Post-SFT

SFT alone is usually insufficient for production:
- Follow with DPO for preference alignment
- Or RLHF for reward optimization
- Add safety fine-tuning for deployment

---

## Sources

### Foundational Papers
- [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) - Google, 2021
- [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) - OpenAI, 2022
- [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) - 2022

### Parameter-Efficient Methods
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) - Microsoft, 2021
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) - UW, 2023

### Notable SFT Models
- [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [Vicuna: An Open-Source Chatbot](https://lmsys.org/blog/2023-03-30-vicuna/)
- [Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944)

### Guides
- [HuggingFace SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [Axolotl Fine-tuning Framework](https://github.com/OpenAccess-AI-Collective/axolotl)
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory)
