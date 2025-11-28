# Context Length Extension for LLMs

Extending the context window of pre-trained LLMs enables handling longer documents, conversations, and reasoning chains. This document covers practical techniques for context extension, from simple inference-time scaling to full retraining approaches.

---

## Why Context Extension Matters

### The Context Bottleneck

**Original context limits** (common pre-training lengths):
- GPT-3: 2,048 tokens (~1,500 words)
- Llama 1: 2,048 tokens
- Llama 2: 4,096 tokens
- Llama 3: 8,192 → 128K (extended)

**Real-world needs**:
- Long document analysis: 10K-100K tokens
- Entire codebases: 50K-500K tokens
- Multi-turn conversations: 5K-20K tokens
- RAG with many retrieved chunks: 10K-50K tokens

### Cost of Limited Context

| Problem | Impact |
|---------|--------|
| **Truncation** | Lost information, incomplete reasoning |
| **Chunking overhead** | Multiple inference calls, expensive |
| **Loss of coherence** | Model "forgets" earlier content |
| **Poor RAG** | Cannot fit all relevant context |

---

## Context Extension Approaches

### Overview of Techniques

| Approach | Training Required | Cost | Context Gain | Quality |
|----------|------------------|------|--------------|---------|
| **RoPE Linear Scaling** | No | Free | 2-4x | Moderate |
| **RoPE NTK Scaling** | No | Free | 2-8x | Good |
| **YaRN** | No | Free | 2-16x | Better |
| **Fine-tuning with PI** | Yes (cheap) | Low | 4-32x | Good |
| **Continued pre-training** | Yes (expensive) | High | 16-128x | Best |
| **Sparse attention** | Architecture change | Varies | Unlimited | Varies |

---

## Position Encoding Scaling (Inference-Time)

### 1. RoPE Linear Scaling

**Concept**: Scale position indices down to fit within original training range.

```python
# Original RoPE (trained on 2048 tokens)
def rotary_position_embedding(seq_len, dim, base=10000):
    position = torch.arange(seq_len)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    emb = position.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([emb.sin(), emb.cos()], dim=-1)

# Linear scaling (extend to 8192 tokens)
def rotary_position_embedding_scaled(seq_len, dim, base=10000, scale=4.0):
    # Scale positions down: 0, 1, 2, ... 8191 → 0, 0.25, 0.5, ... 2047.75
    position = torch.arange(seq_len) / scale
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    emb = position.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([emb.sin(), emb.cos()], dim=-1)
```

**Example: Extending 2K → 8K**
```python
# model trained with 2K context
model = load_model('llama-2k')

# Extend to 8K with linear scaling
model.config.max_position_embeddings = 8192
model.config.rope_scaling = {
    "type": "linear",
    "factor": 4.0  # 8K / 2K = 4
}

# Now can handle 8K tokens (with quality degradation)
output = model.generate(input_ids, max_length=8192)
```

**Limitations**:
- Quality degrades beyond 2-4x extension
- High-frequency components become too compressed
- Attention patterns become less precise

### 2. RoPE NTK-Aware Scaling

**Concept**: Adjust the RoPE base frequency instead of scaling positions.

**Neural Tangent Kernel (NTK) insight**: Changing the base preserves low-frequency position information better than linear scaling.

```python
def rotary_position_embedding_ntk(seq_len, dim, original_base=10000,
                                   original_max_len=2048, new_max_len=8192):
    # Calculate new base using NTK scaling
    scale = new_max_len / original_max_len  # 4.0
    new_base = original_base * (scale ** (dim / (dim - 2)))

    position = torch.arange(seq_len)
    freqs = 1.0 / (new_base ** (torch.arange(0, dim, 2).float() / dim))
    emb = position.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([emb.sin(), emb.cos()], dim=-1)
```

**Example: NTK scaling 2K → 16K**
```python
model.config.rope_scaling = {
    "type": "ntk",
    "factor": 8.0,  # 16K / 2K = 8
    "original_max_position_embeddings": 2048
}
```

**Benefits over linear**:
- Better preserves position information
- Can extend 4-8x with minimal degradation
- No training required

**Used by**: Qwen, Code Llama extended versions

### 3. YaRN (Yet another RoPE extensioN method)

**Concept**: Hybrid approach - scale high frequencies linearly, interpolate low frequencies, and add attention temperature.

**Components**:
1. **Frequency-aware scaling**: Different scaling for different frequency bands
2. **Attention temperature**: Softens attention to reduce pressure on OOD positions
3. **Ramp function**: Smooth transition between scaling strategies

```python
def yarn_scaled_rope(seq_len, dim, original_max_len=2048, new_max_len=32768,
                     base=10000, beta_fast=32, beta_slow=1, alpha=1):
    """
    YaRN: Better context extension with frequency-aware scaling

    Args:
        beta_fast: Cutoff for high-frequency (scaled linearly)
        beta_slow: Cutoff for low-frequency (interpolated)
        alpha: Temperature scaling factor
    """
    scale = new_max_len / original_max_len

    position = torch.arange(seq_len)
    dim_range = torch.arange(0, dim, 2).float()
    freqs = 1.0 / (base ** (dim_range / dim))

    # Frequency-dependent scaling
    wavelength = 2 * math.pi / freqs
    is_high_freq = wavelength < beta_fast
    is_low_freq = wavelength > beta_slow

    # High-freq: linear scaling, Low-freq: interpolation, Mid: ramp
    scaling_factor = torch.where(
        is_high_freq,
        torch.ones_like(freqs),  # No scaling
        torch.where(
            is_low_freq,
            torch.ones_like(freqs) / scale,  # Interpolation
            # Ramp function for middle frequencies
            (1 - (wavelength - beta_fast) / (beta_slow - beta_fast)) * (1 - 1/scale) + 1/scale
        )
    )

    # Apply scaling
    scaled_position = position.unsqueeze(1) * scaling_factor.unsqueeze(0)
    emb = scaled_position * freqs.unsqueeze(0)

    return torch.cat([emb.sin(), emb.cos()], dim=-1), alpha  # alpha for attention temp
```

**Example: YaRN 8K → 128K**
```python
model.config.rope_scaling = {
    "type": "yarn",
    "factor": 16.0,  # 128K / 8K = 16
    "original_max_position_embeddings": 8192,
    "beta_fast": 32,
    "beta_slow": 1,
    "attention_factor": 1.0,  # Temperature scaling
}
```

**Benefits**:
- Can extend 16-32x with good quality
- Best zero-shot (no training) approach
- Used by: Llama 3 128K extension

**Paper**: [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071)

---

## Fine-Tuning Approaches

### 1. Position Interpolation (PI)

**Concept**: Fine-tune model with interpolated positions on long-context data.

**Training recipe**:
```python
# Step 1: Configure position interpolation
model.config.rope_scaling = {
    "type": "linear",
    "factor": 4.0  # 2K → 8K
}

# Step 2: Fine-tune on long documents (short training)
train_config = {
    "sequence_length": 8192,  # Target context length
    "batch_size": 8,  # Smaller due to memory
    "learning_rate": 2e-5,  # Lower than pre-training
    "steps": 1000,  # Only 1000 steps needed!
    "warmup_steps": 100,
}

# Step 3: Use long-context data
# Mix of lengths: 25% at 8K, 50% at 4K, 25% at 2K
dataset = create_length_distributed_dataset([2048, 4096, 8192])

# Fine-tune
trainer.train(model, dataset, train_config)
```

**Cost comparison**:

| Training Type | Steps | GPU Hours | Cost |
|--------------|-------|-----------|------|
| **Pre-training** (from scratch) | 100K-1M | 10K-100K | $500K-5M |
| **Position Interpolation** | 1K-10K | 100-1K | $5K-50K |

**Results** (Meta's Llama 2 extension):
- Llama 2 7B: 4K → 32K in 1000 steps
- Perplexity at 32K: comparable to 4K
- Cost: ~$10K (vs millions for pre-training)

**Paper**: [Extending Context Window via Position Interpolation](https://arxiv.org/abs/2306.15595)

### 2. Two-Stage Training (Short → Long)

**Concept**: Pre-train with short context, then continue pre-training with progressively longer context.

**Stage 1: Initial pre-training (short context)**
```python
# Train with 2K context (standard)
pretrain_config = {
    "sequence_length": 2048,
    "batch_size": 256,
    "steps": 100000,
    "learning_rate": 3e-4,
}
```

**Stage 2: Context extension (long context)**
```python
# Continue training with longer context
extend_config = {
    "sequence_length": 8192,  # 4x longer
    "batch_size": 64,  # Reduced due to memory
    "steps": 5000,  # ~5% of original training
    "learning_rate": 1e-4,  # Lower LR
    "rope_scaling": {"type": "linear", "factor": 4.0},
}
```

**Progressive extension** (recommended):
```python
# Gradually increase context
stages = [
    {"seq_len": 2048, "steps": 100000},  # Stage 1: Standard pre-training
    {"seq_len": 4096, "steps": 2000},    # Stage 2: 2x extension
    {"seq_len": 8192, "steps": 2000},    # Stage 3: 4x extension
    {"seq_len": 16384, "steps": 1000},   # Stage 4: 8x extension
]

for stage in stages:
    model.config.max_position_embeddings = stage["seq_len"]
    # Update RoPE scaling
    model.config.rope_scaling = {
        "type": "linear",
        "factor": stage["seq_len"] / 2048
    }
    train(model, dataset, steps=stage["steps"])
```

**Used by**:
- Llama 3: 8K → 128K
- Qwen 2: 4K → 128K
- Mistral: 8K → 32K

### 3. Long-Context Instruction Fine-Tuning

**After extending base model**, fine-tune on long-context tasks:

```python
# Tasks requiring long context
long_context_tasks = [
    "summarization",  # Summarize 10K-word documents
    "qa",  # Question answering over long documents
    "multi_turn",  # Long conversations (50+ turns)
    "code_completion",  # Complete functions with large context
]

# Instruction dataset with long contexts
dataset = [
    {
        "instruction": "Summarize this research paper",
        "input": paper_text,  # 8K tokens
        "output": summary,  # 500 tokens
    },
    # ... more examples
]

# Fine-tune
sft_config = {
    "sequence_length": 16384,
    "batch_size": 4,
    "learning_rate": 1e-5,
    "steps": 5000,
}
```

---

## Continued Pre-Training (Full Extension)

### When to Use

**Use continued pre-training when**:
- Need maximum quality at long context
- Extending >16x (e.g., 8K → 128K)
- Have compute budget for substantial training

**Continued pre-training recipe**:

```python
# Example: Llama 3 8K → 128K extension
config = {
    # Context
    "original_context": 8192,
    "target_context": 131072,  # 128K
    "rope_scaling": {
        "type": "yarn",  # or "linear" + fine-tuning
        "factor": 16.0,
    },

    # Training
    "tokens": 50_000_000_000,  # 50B tokens (~0.5% of original pre-training)
    "batch_size": 64,
    "sequence_length": 131072,
    "learning_rate": 1e-4,  # Lower than initial pre-training
    "steps": ~380_000,  # 50B tokens / (64 batch * 128K seq_len)

    # Data
    "data_mix": {
        "long_documents": 0.4,  # Books, papers (>32K tokens)
        "code_repositories": 0.3,  # Full repos (>16K tokens)
        "standard_web": 0.3,  # Regular pre-training data
    },

    # Hardware
    "gpus": 256,  # H100s
    "duration": "~1 week",
}
```

**Cost estimate** (Llama 3 70B, 8K→128K):
- Tokens: 50B
- GPU hours: ~100K H100 hours
- Cost: ~$300K-500K

**vs Full pre-training** (Llama 3 70B from scratch):
- Tokens: 15T (15,000B)
- GPU hours: ~6M H100 hours
- Cost: ~$20M-30M

**Savings**: ~50-100x cheaper than full re-training

---

## Data Preparation for Context Extension

### 1. Long Document Collection

**Sources**:
```python
long_context_sources = {
    "books": {
        "avg_length": 50_000,  # tokens
        "sources": ["Project Gutenberg", "Books3"],
    },
    "papers": {
        "avg_length": 15_000,
        "sources": ["arXiv", "PubMed"],
    },
    "code": {
        "avg_length": 20_000,
        "sources": ["GitHub full repos", "The Stack"],
    },
    "conversations": {
        "avg_length": 10_000,
        "sources": ["Long Reddit threads", "StackExchange"],
    },
}
```

### 2. Length Distribution

**Progressive length curriculum**:
```python
# Don't train only on max length - use distribution
length_distribution = {
    "2-4K": 0.30,   # Short documents (original range)
    "4-8K": 0.25,   # Medium
    "8-16K": 0.20,  # Long
    "16-32K": 0.15, # Very long
    "32-64K": 0.08, # Extremely long
    "64K+": 0.02,   # Maximum length examples
}

def sample_document(dataset, target_length_range):
    """Sample document matching target length"""
    min_len, max_len = target_length_range
    candidates = [d for d in dataset if min_len <= len(d) <= max_len]
    return random.choice(candidates)
```

**Why diverse lengths?**
- Model doesn't forget short-context performance
- Computational efficiency (shorter = faster)
- Matches real-world distribution

### 3. Packing Long Documents

```python
def pack_documents(documents, target_seq_len=8192):
    """
    Pack multiple documents into training sequences

    Strategy:
    - For short docs: concatenate with separator
    - For long docs: use full document
    """
    packed_sequences = []
    current_sequence = []
    current_length = 0

    for doc in documents:
        doc_length = len(doc)

        if doc_length >= target_seq_len:
            # Long document: use as-is (potentially truncate to target_seq_len)
            packed_sequences.append(doc[:target_seq_len])
        else:
            # Short document: pack with others
            if current_length + doc_length + 1 > target_seq_len:
                # Current sequence full, start new one
                packed_sequences.append(current_sequence)
                current_sequence = [doc]
                current_length = doc_length
            else:
                # Add to current sequence
                current_sequence.append(doc)
                current_length += doc_length + 1  # +1 for separator

    return packed_sequences
```

---

## Evaluation of Extended Context

### 1. Perplexity Across Positions

```python
def evaluate_perplexity_by_position(model, long_documents):
    """
    Test if model maintains quality across full context window
    """
    results = []

    for doc in long_documents:
        # Compute per-token perplexity
        perplexities = []
        for i in range(len(doc)):
            context = doc[:i+1]
            ppl = compute_perplexity(model, context)
            perplexities.append(ppl)

        # Perplexity should stay flat across positions
        results.append({
            'doc_length': len(doc),
            'avg_ppl': np.mean(perplexities),
            'ppl_at_2k': perplexities[2000],
            'ppl_at_8k': perplexities[8000] if len(doc) > 8000 else None,
            'ppl_at_16k': perplexities[16000] if len(doc) > 16000 else None,
        })

    return results
```

**Good result**: Perplexity stays roughly constant across positions
**Bad result**: Perplexity increases significantly at longer positions

### 2. Needle-in-Haystack Test

**Test**: Can model retrieve specific information from arbitrary position in long context?

```python
def needle_in_haystack_test(model, context_length=32000):
    """
    Insert 'needle' (specific fact) at different positions in long 'haystack' (filler text)
    Query model to retrieve the needle
    """
    results = []

    # Needle: specific retrievable fact
    needle = "The secret code is: X7J9K2"

    # Haystack: long filler text
    haystack = generate_filler_text(context_length)

    # Test insertion at different depths
    for depth_pct in [0, 10, 25, 50, 75, 90, 100]:
        # Insert needle at depth% through context
        insert_pos = int(context_length * depth_pct / 100)
        context = haystack[:insert_pos] + needle + haystack[insert_pos:]

        # Query model
        query = "What is the secret code?"
        response = model.generate(context + query)

        # Check if model retrieved needle
        success = "X7J9K2" in response
        results.append({
            'depth_pct': depth_pct,
            'success': success
        })

    return results
```

**Ideal result**: 100% success at all depths
**Practical target**: >95% success at all depths

**Paper**: [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)

### 3. Long-Context Benchmarks

| Benchmark | Context Length | Task Type |
|-----------|----------------|-----------|
| **RULER** | Up to 128K | Retrieval, reasoning, aggregation |
| **LongBench** | 4K-32K | Summarization, QA, code |
| **InfiniteBench** | 100K-1M | Extremely long contexts |
| **ZeroSCROLLS** | 10K-100K | Summarization, QA |

**Example evaluation**:
```bash
# Evaluate on RULER
python eval_ruler.py \
    --model llama-3-8k-extended \
    --context_lengths 4000,8000,16000,32000,64000,128000
```

---

## Architectural Alternatives

### 1. Sparse Attention Patterns

**Problem**: Standard attention is O(n²) in sequence length.

**Solutions**:

**Sliding window attention**:
```python
# Mistral/Llama 3 approach
# Attend only to last 4K tokens (sliding window)
# But use RoPE scaling for positions beyond 4K
config = {
    "sliding_window": 4096,  # Only attend to last 4K
    "max_position_embeddings": 128000,  # But track positions up to 128K
}
```

**Sparse patterns**:
- **Longformer**: Local + global attention
- **BigBird**: Random + window + global
- **Flash-Decoding++**: Optimized sparse patterns

### 2. Retrieval-Augmented Context

**Alternative**: Don't extend context, retrieve relevant chunks.

```python
# Instead of 100K context:
full_document = load_document()  # 100K tokens

# Retrieve top-K relevant chunks
query_embedding = embed(user_query)
chunk_scores = score_chunks(full_document, query_embedding)
relevant_chunks = top_k_chunks(chunk_scores, k=10)

# Fit in 8K context window
context = concatenate(relevant_chunks)  # 5K tokens
response = model.generate(context + user_query)
```

**Trade-offs**:
- ✅ Fits in existing context window
- ✅ Cheaper inference
- ❌ May miss relevant info (retrieval failures)
- ❌ Cannot handle tasks requiring full document reasoning

---

## Practical Recommendations

### Quick Extension (No Training)

**For 2-4x extension**:
1. Use NTK-aware RoPE scaling
2. Test on your domain
3. If quality acceptable, deploy

```python
model.config.rope_scaling = {
    "type": "ntk",
    "factor": 4.0,
}
```

**Cost**: $0
**Time**: Immediate
**Quality**: Good for 2-4x

### Medium Extension (Light Fine-Tuning)

**For 4-8x extension**:
1. Start with YaRN scaling
2. Fine-tune 1K-5K steps on long-context data
3. Evaluate on target benchmarks

```python
# 1. YaRN
model.config.rope_scaling = {"type": "yarn", "factor": 8.0}

# 2. Fine-tune
train(model, long_context_data, steps=2000, lr=1e-5)
```

**Cost**: $5K-20K
**Time**: 1-3 days
**Quality**: Good for 8-16x

### Full Extension (Continued Pre-Training)

**For >16x extension or maximum quality**:
1. Use YaRN or learned scaling
2. Continued pre-training with 50B-500B tokens
3. Long-context instruction tuning
4. Comprehensive evaluation

```python
# 1. Configure scaling
model.config.rope_scaling = {"type": "yarn", "factor": 16.0}

# 2. Continued pre-training
pretrain(model, long_data, tokens=50_000_000_000)

# 3. Instruction tuning
sft(model, instruction_data, steps=5000)
```

**Cost**: $100K-1M
**Time**: 1-4 weeks
**Quality**: Best for 32-128x

---

## Common Pitfalls

### ❌ Mistakes to Avoid

1. **Training only on max-length examples**
   - Model forgets short-context performance
   - Use length distribution

2. **Not evaluating across positions**
   - Quality may degrade at certain positions
   - Use needle-in-haystack and per-position perplexity

3. **Excessive extension without training**
   - Linear scaling >4x has poor quality
   - Use fine-tuning or YaRN for larger extensions

4. **Ignoring memory constraints**
   - 128K context with full attention = OOM
   - Use gradient checkpointing, sparse attention, or optimized kernels

5. **Wrong RoPE scaling type**
   - Linear scaling for small extensions (2-4x)
   - NTK/YaRN for larger (4-16x)
   - Training for extreme (16x+)

---

## Case Studies

### Case Study 1: Llama 3 8K → 128K

**Approach**: Continued pre-training + YaRN

**Recipe**:
1. YaRN scaling configured (16x factor)
2. Continued pre-training on 50B tokens
3. Long-context instruction tuning
4. Total cost: ~$500K (vs $20M for full re-training)

**Results**:
- RULER 128K: 85% accuracy
- Needle-in-haystack: >95% at all positions
- Perplexity: Stable across 128K window

**Lesson**: Continued pre-training works for extreme extensions with <3% of original training cost.

### Case Study 2: Code Llama 16K → 100K

**Approach**: Position interpolation + repository-level training

**Recipe**:
1. Linear position interpolation (6.25x)
2. Fine-tuning on full code repositories (5K steps)
3. Evaluation on long-file completion

**Results**:
- Can complete functions with 50K tokens of context
- Repository-level understanding
- Cost: ~$15K

**Lesson**: Domain-specific extension (code) can be done cheaply with targeted fine-tuning.

### Case Study 3: Failed Extension (Linear 32x)

**Approach**: Naive linear scaling from 4K to 128K without training

**Recipe**:
1. Applied linear RoPE scaling (factor=32)
2. No fine-tuning
3. Deployed directly

**Results**:
- Perplexity at 64K: 3x higher than 4K
- Needle-in-haystack: 30% success rate beyond 16K
- Model "lost" information at long positions

**Lesson**: Aggressive scaling (>8x) without training fails. Always evaluate before deploying.

---

## Resources

### Papers
- [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071) - Best inference-time scaling
- [Extending Context is Hard But Not Impossible](https://arxiv.org/abs/2401.01325) - Analysis of extension methods
- [Position Interpolation for Llama](https://arxiv.org/abs/2306.15595) - Meta's PI approach
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) - Context utilization analysis
- [Effective Long Context Scaling](https://arxiv.org/abs/2309.16039) - Qwen's approach

### Benchmarks
- [RULER](https://github.com/hsiehjackson/RULER) - Comprehensive long-context evaluation
- [LongBench](https://github.com/THUDM/LongBench) - Multi-task long-context benchmark
- [InfiniteBench](https://github.com/OpenBMB/InfiniteBench) - 100K-1M context evaluation

### Tools
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Efficient attention for long context
- [vLLM](https://github.com/vllm-project/vllm) - Efficient serving with long context support
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/llama#transformers.LlamaConfig.rope_scaling) - RoPE scaling configs

---

**Related Documentation**:
- [Context Windows](../../architecture/context-windows.md) - Architecture fundamentals
- [Position Embeddings](../../architecture/position-embeddings.md) - RoPE and alternatives
- [Training Stability](training-stability.md) - Handling training issues during extension
- [Evaluation](../llmops/evaluation.md) - How to evaluate extended models
