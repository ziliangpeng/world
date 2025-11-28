# KV Cache Optimization

The Key-Value (KV) cache stores attention states from previous tokens, avoiding redundant computation during autoregressive generation. For long sequences and large models, this cache becomes the dominant memory consumer and a key bottleneck. This document covers techniques that have revolutionized KV cache management, from PagedAttention to advanced compression methods.

---

## Why KV Cache Matters

### The Memory Problem

During generation, each token needs to attend to all previous tokens. Without caching:
```
Token 1: Compute K, V for all previous tokens (0 tokens)
Token 2: Compute K, V for all previous tokens (1 token)
Token 3: Compute K, V for all previous tokens (2 tokens)
...
Token N: Compute K, V for all previous tokens (N-1 tokens)

Total: O(N²) recomputation
```

With KV cache:
```
Token 1: Store K₁, V₁
Token 2: Store K₂, V₂; Attend to K₁₂, V₁₂
Token 3: Store K₃, V₃; Attend to K₁₂₃, V₁₂₃
...
Only O(N) computation total
```

### Cache Size Calculation

```
KV cache size = 2 × batch × layers × seq_len × heads × head_dim × bytes

For LLaMA-70B with 4K context:
= 2 × 1 × 80 × 4096 × 64 × 128 × 2 (fp16)
= 10.7 GB per sequence
```

| Model | 4K Context | 32K Context | 128K Context |
|-------|-----------|-------------|--------------|
| LLaMA-7B | 1 GB | 8 GB | 32 GB |
| LLaMA-70B | 10 GB | 80 GB | 320 GB |
| Mixtral-8×7B | 2 GB | 16 GB | 64 GB |

**The challenge**: KV cache can exceed model weights for long contexts.

---

## Historical Evolution

### Phase 1: Static Allocation (Pre-2023)

Early inference systems pre-allocated fixed KV cache:

```python
# Static allocation (wasteful)
kv_cache = torch.zeros(batch, max_seq_len, layers, 2, heads, head_dim)
# Pre-allocates for max length even if most sequences are short
```

**Problems**:
- Memory fragmentation
- Can't pack variable-length sequences
- Limited batch size due to worst-case allocation

### Phase 2: PagedAttention (2023)

**[vLLM](https://arxiv.org/abs/2309.06180)** (September 2023)

Revolutionary memory management inspired by OS virtual memory:

```
Instead of: [Seq1: K₁K₂K₃K₄K₅K₆K₇K₈] (contiguous)

Use pages:  [Page1: K₁K₂K₃K₄] [Page2: K₅K₆K₇K₈]
            Pages can be anywhere in memory
            Only allocate pages as needed
```

**Benefits**:
- Near-zero memory waste
- Efficient memory sharing (prefix caching)
- Flexible batch scheduling

### Phase 3: Compression and Efficiency (2024)

Multiple approaches to reduce KV cache size:
- **Quantization**: Store KV in int8/int4
- **Compression**: Learned compression of KV states
- **Eviction**: Drop less important KV pairs
- **Sharing**: Cross-layer KV sharing

---

## PagedAttention

### Core Concept

```
Traditional:
┌─────────────────────────────────────────┐
│ Sequence 1 KV Cache (contiguous block)  │
└─────────────────────────────────────────┘
┌─────────────────────────┐
│ Sequence 2 (shorter)    │ WASTED SPACE
└─────────────────────────┘

PagedAttention:
┌────┐ ┌────┐ ┌────┐ ┌────┐
│ S1 │ │ S1 │ │ S2 │ │ S1 │ Pages allocated as needed
└────┘ └────┘ └────┘ └────┘ Any sequence can use any page
```

### Implementation

```python
class PagedKVCache:
    def __init__(self, page_size=16, num_pages=1024):
        self.page_size = page_size
        # Physical pages stored in GPU memory
        self.pages = torch.zeros(num_pages, page_size, 2, heads, head_dim)
        self.free_pages = set(range(num_pages))
        # Mapping from (seq_id, logical_page) → physical_page
        self.page_table = {}

    def allocate_page(self, seq_id, logical_idx):
        if not self.free_pages:
            raise MemoryError("Out of KV cache pages")
        physical_page = self.free_pages.pop()
        self.page_table[(seq_id, logical_idx)] = physical_page
        return physical_page

    def get_kv(self, seq_id, position):
        logical_page = position // self.page_size
        offset = position % self.page_size
        physical_page = self.page_table[(seq_id, logical_page)]
        return self.pages[physical_page, offset]

    def free_sequence(self, seq_id):
        # Return all pages to free pool
        for key, physical_page in list(self.page_table.items()):
            if key[0] == seq_id:
                self.free_pages.add(physical_page)
                del self.page_table[key]
```

### Prefix Caching

Share KV cache across requests with common prefixes:

```
Request 1: "You are a helpful assistant. User: What is 2+2?"
Request 2: "You are a helpful assistant. User: What is the capital of France?"

Shared prefix: "You are a helpful assistant. User: "

┌────────────────────────┐
│ Shared Prefix KV Cache │ ← Both requests reference same pages
└────────────────────────┘
    ↓           ↓
┌───────┐   ┌───────┐
│ Req 1 │   │ Req 2 │  ← Only unique portions allocated
└───────┘   └───────┘
```

**vLLM automatic prefix caching**:
```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True,  # Enable automatic prefix sharing
)
```

---

## KV Cache Compression

### Quantization

Store KV cache in lower precision:

```python
def quantize_kv_cache(keys, values, bits=8):
    """Quantize KV cache to int8."""
    # Compute scale per head
    k_scale = keys.abs().amax(dim=-1, keepdim=True) / 127
    v_scale = values.abs().amax(dim=-1, keepdim=True) / 127

    k_quant = (keys / k_scale).round().clamp(-128, 127).to(torch.int8)
    v_quant = (values / v_scale).round().clamp(-128, 127).to(torch.int8)

    return k_quant, v_quant, k_scale, v_scale

def dequantize_kv_cache(k_quant, v_quant, k_scale, v_scale):
    """Dequantize for attention computation."""
    keys = k_quant.float() * k_scale
    values = v_quant.float() * v_scale
    return keys, values
```

**Memory savings**: 2× for int8, 4× for int4

### KV Cache Eviction

Drop less important KV pairs:

**[StreamingLLM](https://arxiv.org/abs/2309.17453)** (2023):
Keep attention sinks (first few tokens) + sliding window:

```python
def streaming_kv_cache(kv_cache, window_size=1024, sink_size=4):
    """
    Keep first `sink_size` tokens + last `window_size` tokens.
    """
    total_len = kv_cache.shape[1]
    if total_len <= sink_size + window_size:
        return kv_cache

    # Keep sinks + recent window
    keep_indices = list(range(sink_size)) + list(range(total_len - window_size, total_len))
    return kv_cache[:, keep_indices]
```

**[H2O (Heavy-Hitter Oracle)](https://arxiv.org/abs/2306.14048)**:
Keep tokens with highest cumulative attention:

```python
def h2o_eviction(kv_cache, attention_scores, budget):
    """Keep tokens with highest cumulative attention."""
    # Sum attention across all queries
    importance = attention_scores.sum(dim=-2)  # [batch, heads, seq_len]

    # Keep top-k important tokens
    _, keep_indices = importance.topk(budget, dim=-1)
    return kv_cache.gather(dim=1, index=keep_indices)
```

### Learned Compression

Train models to compress KV states:

```python
class KVCompressor(nn.Module):
    def __init__(self, head_dim, compressed_dim):
        self.compress = nn.Linear(head_dim, compressed_dim)
        self.decompress = nn.Linear(compressed_dim, head_dim)

    def forward(self, kv):
        # Compress for storage
        compressed = self.compress(kv)
        return compressed

    def restore(self, compressed):
        # Decompress for attention
        return self.decompress(compressed)
```

---

## Architecture-Level Optimizations

### Grouped Query Attention (GQA)

Share KV heads across query heads:

```
MHA: 32 Q heads, 32 K heads, 32 V heads → Full KV cache
GQA: 32 Q heads, 8 K heads, 8 V heads   → 4× smaller KV cache
MQA: 32 Q heads, 1 K head, 1 V head     → 32× smaller KV cache
```

**KV cache savings**:
| Attention | Q Heads | KV Heads | KV Cache Size |
|-----------|---------|----------|---------------|
| MHA | 32 | 32 | 1× (baseline) |
| GQA-8 | 32 | 8 | 0.25× |
| GQA-4 | 32 | 4 | 0.125× |
| MQA | 32 | 1 | 0.03× |

> **See also**: [Attention Mechanisms](../architecture/attention-mechanisms.md) for GQA details.

### Multi-Query Attention (MQA)

Extreme KV sharing—single K, V for all queries:

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)  # Full Q
        self.k_proj = nn.Linear(hidden_dim, self.head_dim)  # Single K
        self.v_proj = nn.Linear(hidden_dim, self.head_dim)  # Single V

    def forward(self, x, kv_cache=None):
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, 1, self.head_dim)
        v = self.v_proj(x).view(B, T, 1, self.head_dim)

        # K, V broadcast to all heads
        k = k.expand(-1, -1, self.num_heads, -1)
        v = v.expand(-1, -1, self.num_heads, -1)

        return attention(q, k, v)
```

### Cross-Layer KV Sharing

Share KV cache across layers:

**[YOCO](https://arxiv.org/abs/2405.05254)** (2024):
- Self-attention in early layers
- Cross-attention to early layers' KV in later layers
- Significant KV cache reduction

---

## Implementation

### vLLM KV Cache Management

```python
from vllm import LLM, SamplingParams

# Automatic PagedAttention
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    gpu_memory_utilization=0.9,  # Use 90% of GPU for KV cache
    max_model_len=32768,         # Maximum context length
)

# With KV cache quantization
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    kv_cache_dtype="fp8",  # Quantize KV cache
)
```

### SGLang RadixAttention

Extends prefix caching with radix tree structure:

```python
from sglang import RuntimeEndpoint

# RadixAttention automatically shares prefixes
runtime = RuntimeEndpoint(
    model="meta-llama/Llama-2-70b-hf",
)

# Multi-turn conversation efficiently reuses context
response1 = runtime.generate("Context: ... Question 1")
response2 = runtime.generate("Context: ... Question 2")  # Reuses "Context: ..." cache
```

### FlashAttention Integration

FlashAttention optimizes attention computation with KV cache:

```python
from flash_attn import flash_attn_with_kvcache

# Efficient attention with existing KV cache
output = flash_attn_with_kvcache(
    q,           # New query
    k_cache,     # Existing K cache
    v_cache,     # Existing V cache
    k_new,       # New K to append
    v_new,       # New V to append
)
```

---

## Memory Management Strategies

### GPU Memory Budget

```python
def plan_kv_cache(model_params, gpu_memory, model_memory):
    """Plan KV cache allocation."""
    available = gpu_memory - model_memory

    # KV cache per token
    kv_per_token = 2 * model_params["layers"] * model_params["heads"] * \
                   model_params["head_dim"] * 2  # 2 bytes for fp16

    max_tokens = available / kv_per_token

    # Distribute across batch and sequence length
    # e.g., 10 sequences × 10K tokens or 100 sequences × 1K tokens
    return max_tokens
```

### Dynamic Allocation

```python
class DynamicKVCache:
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.current_usage = 0
        self.sequences = {}

    def can_allocate(self, seq_id, tokens):
        required = self.calculate_memory(tokens)
        return self.current_usage + required <= self.max_memory

    def allocate(self, seq_id, tokens):
        if not self.can_allocate(seq_id, tokens):
            # Preemption: evict lower-priority sequence
            self.preempt_lowest_priority()

        memory = self.calculate_memory(tokens)
        self.sequences[seq_id] = {"tokens": tokens, "memory": memory}
        self.current_usage += memory

    def preempt_lowest_priority(self):
        # Swap to CPU or drop
        lowest = min(self.sequences.keys(), key=lambda s: self.priority(s))
        self.swap_to_cpu(lowest)
```

---

## Best Practices

### Model Selection

| Use Case | Recommended Architecture |
|----------|-------------------------|
| Long context, memory-limited | GQA with 4-8 KV heads |
| Extreme context (>100K) | MQA or hybrid SSM |
| Quality-critical | Full MHA with compression |
| High throughput | GQA + PagedAttention |

### Configuration

```python
# High-throughput serving
vllm_config = {
    "gpu_memory_utilization": 0.95,
    "max_num_seqs": 256,
    "enable_prefix_caching": True,
    "kv_cache_dtype": "auto",  # Let vLLM choose
}

# Long-context serving
vllm_config = {
    "gpu_memory_utilization": 0.90,
    "max_model_len": 128000,
    "max_num_seqs": 16,  # Fewer sequences for longer context
}
```

### Monitoring

Key metrics to track:
- KV cache utilization (%)
- Cache hit rate (for prefix caching)
- Preemption rate
- Memory fragmentation

---

## Future Directions

### Near-term (2025)

1. **FP8 KV cache**: Standard on H100/MI300
2. **Better compression**: Learned compression becoming practical
3. **Hardware support**: Dedicated KV cache memory
4. **Longer contexts**: 1M+ token efficient serving

### Research Frontiers

1. **Semantic compression**: Compress based on importance
2. **Hierarchical caching**: Multi-level cache (GPU → CPU → disk)
3. **Predictive caching**: Anticipate future KV needs
4. **Hybrid architectures**: Combine SSM (no cache) with attention

---

## Sources

### Foundational Papers
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM, 2023
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) - 2023

### KV Cache Optimization
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) - StreamingLLM, 2023
- [H2O: Heavy-Hitter Oracle for Efficient KV Cache](https://arxiv.org/abs/2306.14048) - 2023
- [GQA: Training Generalized Multi-Query Transformer](https://arxiv.org/abs/2305.13245) - Google, 2023

### Implementations
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
