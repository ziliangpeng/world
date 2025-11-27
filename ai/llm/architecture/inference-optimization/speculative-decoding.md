# Speculative Decoding

Speculative decoding accelerates LLM inference by using a small, fast "draft" model to propose multiple tokens that the large "target" model then verifies in parallel. This exploits the fact that verification is much cheaper than generation, achieving 2-3× speedups without changing output quality.

---

## The Autoregressive Bottleneck

### Why LLM Inference is Slow

Standard autoregressive decoding generates one token at a time:

```
Prompt: "The capital of France is"
Step 1: Generate "Paris"    → Load 70B params, compute
Step 2: Generate "."        → Load 70B params again, compute
Step 3: Generate "\n"       → Load 70B params again, compute
...
```

**The problem**: Each token requires loading all model weights from memory. For a 70B model, that's 140GB of memory transfer per token.

```
Time per token ≈ Model size / Memory bandwidth

70B fp16 model:
140 GB / 2 TB/s (A100) ≈ 70ms per token
```

### Memory-Bound vs Compute-Bound

| Scenario | Bottleneck | Time per Token |
|----------|------------|----------------|
| Single token (batch=1) | Memory bandwidth | 50-100ms |
| Many tokens (batch=64) | Compute | 10-20ms |

**Key insight**: With batch size 1, we only use ~5% of GPU compute. The rest waits for memory.

Speculative decoding exploits this idle compute.

---

## How Speculative Decoding Works

### The Core Idea

1. **Draft**: Small model quickly proposes K candidate tokens
2. **Verify**: Large model checks all K tokens in parallel
3. **Accept**: Keep tokens where large model agrees
4. **Reject**: Resample from large model where it disagrees

```
Draft model (7B): "Paris is the capital"    [fast: 4 tokens]
Target model (70B): Verify all 4 in parallel [one forward pass]
Result: Accept "Paris is the" (3 tokens)    [reject "capital"]
Continue from "the"...
```

### Why This Works

**Verification is cheap**: Checking K tokens takes almost the same time as generating 1 token (memory-bound, not compute-bound).

**Agreement is common**: Draft and target often agree, especially for common patterns.

```
Cost comparison:
Standard: K tokens × (load 70B) = K × 70ms = 280ms for 4 tokens
Speculative: 4 tokens × (load 7B) + 1 × (load 70B)
           = 4×10ms + 70ms = 110ms for ~3 accepted tokens
Speedup: ~2.5×
```

---

## Algorithms

### Basic Speculative Decoding

```python
def speculative_decode(prompt, draft_model, target_model, K=4):
    """
    Basic speculative decoding algorithm.
    K: number of tokens to speculate
    """
    tokens = prompt

    while not done:
        # Draft: generate K tokens with small model
        draft_tokens = []
        draft_probs = []
        for _ in range(K):
            logits = draft_model(tokens + draft_tokens)
            prob = softmax(logits[-1])
            token = sample(prob)
            draft_tokens.append(token)
            draft_probs.append(prob[token])

        # Verify: run target model on all drafts in parallel
        all_tokens = tokens + draft_tokens
        target_logits = target_model(all_tokens)  # Single forward pass

        # Accept/reject each draft token
        n_accepted = 0
        for i in range(K):
            target_prob = softmax(target_logits[len(tokens) + i])
            draft_prob = draft_probs[i]

            # Rejection sampling
            r = random.random()
            if r < min(1, target_prob[draft_tokens[i]] / draft_prob):
                # Accept
                tokens.append(draft_tokens[i])
                n_accepted += 1
            else:
                # Reject: sample from adjusted distribution
                adjusted = relu(target_prob - draft_prob * r)
                adjusted = adjusted / adjusted.sum()
                token = sample(adjusted)
                tokens.append(token)
                break  # Stop accepting after first rejection

        # If all accepted, sample one more from target
        if n_accepted == K:
            token = sample(softmax(target_logits[-1]))
            tokens.append(token)

    return tokens
```

### Parallel Verification

The key efficiency: verify all K tokens with one forward pass:

```python
def parallel_verify(tokens, draft_tokens, target_model):
    """
    Verify K draft tokens with single forward pass.
    """
    # Concatenate prompt + draft tokens
    full_sequence = tokens + draft_tokens

    # Single forward pass through target
    # Shape: [1, seq_len, vocab_size]
    all_logits = target_model(full_sequence)

    # Extract logits for each position we need to verify
    # Position i predicts token i+1
    verify_positions = range(len(tokens) - 1, len(tokens) - 1 + len(draft_tokens))
    draft_logits = all_logits[verify_positions]

    return draft_logits
```

### Tree-Based Speculation

Instead of linear speculation, build a tree of possibilities:

```
              "The"
             /    \
         "cat"   "dog"
         /  \      |
      "is" "was"  "is"
```

**Advantage**: Higher acceptance rate (explores multiple branches)
**Disadvantage**: More complex verification

---

## Draft Model Strategies

### 1. Separate Draft Model

Use a smaller model from same family:

| Target | Draft | Speedup |
|--------|-------|---------|
| LLaMA-70B | LLaMA-7B | 2-2.5× |
| Mistral-7B | Mistral-2B | 1.5-2× |
| GPT-4 | GPT-3.5 | 2-3× |

**Requirements**:
- Same tokenizer
- Similar training distribution
- Smaller but competent

### 2. Self-Speculative Decoding

Use the same model with early exit:

```python
def self_speculative(model, prompt, exit_layer=6):
    """
    Use early layers as draft model.
    """
    tokens = prompt

    while not done:
        # Draft: use first N layers
        draft_tokens = []
        for _ in range(K):
            hidden = model.embed(tokens + draft_tokens)
            for layer in model.layers[:exit_layer]:
                hidden = layer(hidden)
            logits = model.lm_head(hidden)
            token = sample(softmax(logits[-1]))
            draft_tokens.append(token)

        # Verify: use full model
        full_logits = model(tokens + draft_tokens)
        # ... accept/reject as before
```

**Advantage**: No separate model needed
**Disadvantage**: Lower draft quality

### 3. Medusa Heads

Train additional prediction heads on the target model:

```python
class MedusaModel(nn.Module):
    def __init__(self, base_model, n_heads=4):
        self.base = base_model
        # Additional heads predict future tokens
        self.medusa_heads = nn.ModuleList([
            nn.Linear(hidden_size, vocab_size)
            for _ in range(n_heads)
        ])

    def forward(self, x):
        hidden = self.base.get_hidden(x)

        # Base prediction (next token)
        base_logits = self.base.lm_head(hidden)

        # Medusa predictions (tokens 2, 3, 4, 5)
        medusa_logits = [head(hidden) for head in self.medusa_heads]

        return base_logits, medusa_logits
```

**Advantage**: No separate model, lightweight
**Disadvantage**: Requires fine-tuning heads

### 4. Lookahead Decoding

Use n-gram cache from previous generations:

```python
class LookaheadCache:
    def __init__(self):
        self.ngram_cache = {}  # Maps context → likely continuations

    def lookup(self, context, n=4):
        """Find cached continuations for context."""
        key = tuple(context[-n:])
        return self.ngram_cache.get(key, [])

    def update(self, context, continuation):
        """Add new n-gram to cache."""
        key = tuple(context[-4:])
        self.ngram_cache[key] = continuation
```

**Advantage**: Zero overhead, improves over time
**Disadvantage**: Limited to seen patterns

---

## Implementation

### vLLM Speculative Decoding

```python
from vllm import LLM, SamplingParams

# With separate draft model
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="meta-llama/Llama-2-7b-hf",
    num_speculative_tokens=5,
)

# With ngram-based speculation
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="[ngram]",  # Use ngram lookahead
    ngram_prompt_lookup_max=4,
)

output = llm.generate("Explain quantum computing", SamplingParams())
```

### HuggingFace Assisted Generation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load models
target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-hf")
draft = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")

# Generate with speculation
inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = target.generate(
    **inputs,
    assistant_model=draft,
    max_new_tokens=50,
)
```

### Medusa Implementation

```python
# Using Medusa-trained model
from medusa import MedusaModel

model = MedusaModel.from_pretrained("medusa-vicuna-7b-v1.3")

# Generate with tree-based speculation
output = model.generate(
    prompt,
    temperature=0.0,  # Greedy for max speedup
    max_steps=512,
)
```

---

## Performance Analysis

### Speedup Factors

| Factor | Impact on Speedup |
|--------|-------------------|
| Draft quality | Higher acceptance → more speedup |
| K (speculation length) | More tokens → higher potential gain |
| Target model size | Larger → more memory-bound → more speedup |
| Batch size | Larger → less memory-bound → less speedup |
| Task type | Predictable text → more speedup |

### Typical Results

| Target | Draft | Task | Speedup |
|--------|-------|------|---------|
| LLaMA-70B | LLaMA-7B | General | 1.8-2.5× |
| LLaMA-70B | Medusa | General | 2.0-2.5× |
| Vicuna-33B | Vicuna-7B | Chat | 1.5-2.0× |
| CodeLlama-34B | CodeLlama-7B | Code | 2.0-3.0× |

### When It Helps Most

1. **Large models**: More memory-bound, more room for improvement
2. **Predictable outputs**: Code, structured text, common phrases
3. **Small batch sizes**: Batch=1 is optimal
4. **Greedy decoding**: Temperature=0 has highest acceptance

### When It Helps Less

1. **Small models**: Already fast, less memory-bound
2. **Creative tasks**: High temperature → low acceptance
3. **Large batches**: Already compute-bound
4. **Very long contexts**: KV cache dominates

---

## Trade-offs

### Advantages

1. **Lossless**: Output distribution unchanged
2. **Significant speedup**: 2-3× common
3. **Works with any target**: No model changes needed
4. **Composable**: Combines with quantization, batching

### Disadvantages

1. **Extra model**: Draft model uses memory
2. **Complexity**: More inference infrastructure
3. **Variable speedup**: Depends on draft quality
4. **Overhead**: If acceptance low, can be slower

### Memory Overhead

```
Standard:     70B target = 140GB
Speculative:  70B target + 7B draft = 140GB + 14GB = 154GB
Overhead:     ~10%
```

For Medusa heads: ~2-5% overhead (just extra linear layers).

---

## Best Practices

### Choosing Draft Model

1. **Same tokenizer**: Must match exactly
2. **Same family**: Similar distributions help
3. **10× smaller**: Good balance of speed and quality
4. **Fine-tune if needed**: On similar data as target

### Tuning K (Speculation Length)

```python
# Empirical tuning
for K in [2, 4, 6, 8]:
    speedup, acceptance = benchmark(target, draft, K)
    print(f"K={K}: speedup={speedup:.2f}×, acceptance={acceptance:.2%}")

# Typical sweet spot: K=4-6
```

### Task-Specific Optimization

| Task | Recommendation |
|------|----------------|
| Code generation | K=6-8 (highly predictable) |
| Creative writing | K=2-3 (low acceptance expected) |
| Translation | K=4-5 (structured output) |
| Chat | K=4 (mixed predictability) |

---

## Future Directions

### Near-term (2025)

1. **Better draft models**: Specialized for speculation
2. **Adaptive K**: Adjust speculation length dynamically
3. **Hardware support**: Custom kernels for verification
4. **Integration**: Standard in all inference frameworks

### Research Frontiers

1. **Learned rejection**: Train to predict acceptance
2. **Multi-draft**: Multiple drafts in parallel
3. **Speculative prefill**: Apply to prompt processing
4. **Hybrid strategies**: Combine lookahead, Medusa, draft

---

## Sources

### Foundational Papers
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) - Google, 2022
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) - DeepMind, 2023

### Variants
- [Medusa: Simple Framework for Accelerating LLM Generation](https://arxiv.org/abs/2401.10774) - 2024
- [Lookahead Decoding](https://arxiv.org/abs/2402.02057) - 2024
- [EAGLE: Speculative Sampling with Draft Model Cooperation](https://arxiv.org/abs/2401.15077) - 2024

### Implementations
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/models/spec_decode.html)
- [HuggingFace Assisted Generation](https://huggingface.co/docs/transformers/generation_strategies#assisted-generation)
- [Medusa GitHub](https://github.com/FasterDecoding/Medusa)
