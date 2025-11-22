# Context Windows

Context window refers to the maximum number of tokens a model can process in a single forward pass. The evolution from 2K to multi-million tokens represents one of the most dramatic improvements in LLM capabilities.

## Why Context Windows Matter

**Context = Memory**:
- All information model can reference
- Previous conversation
- Documents to analyze
- Code to understand

**Larger Context Enables**:
- Entire books in one go
- Full codebases
- Long conversations
- Multi-document analysis

---

## Historical Evolution (2018-2025)

### The Journey from 2K to 10M+ Tokens

```
2018: GPT-2
  - 1,024 tokens
  - ~750 words

2019: GPT-3
  - 2,048 tokens
  - ~1,500 words

2020: Llama 1
  - 2,048 tokens
  - Standard for early LLMs

2022: Llama 2
  - 4,096 tokens
  - ~3,000 words
  - Still relatively short

2023: Context Explosion Begins
  - MPT-7B StoryWriter: 65K tokens
  - Claude 2: 100K tokens
  - GPT-4 Turbo: 128K tokens

2024: Million Token Era
  - Llama 3.1: 128K tokens
  - Qwen 2.5: 128K-1M tokens
  - Claude 3: 200K tokens (1M available)
  - Gemini 1.5 Pro: 1M-2M tokens
  - DeepSeek-R1: 128K+

2025: Ten Million Tokens
  - GPT-5 (reported): 400K input, 128K output
  - Llama 4 (rumored): 10M+ tokens
```

**Growth**: 1K → 10M = **10,000x increase** in ~7 years!

---

## Technical Challenges

### The Quadratic Attention Problem

**Standard Self-Attention**: O(n²) complexity
```
For sequence length n:
- Memory: n² attention matrix
- Compute: n² dot products

Examples:
1K tokens: 1M operations
4K tokens: 16M operations (16x more)
128K tokens: 16B operations (16,000x more!)
1M tokens: 1T operations (1,000,000x more!)
```

**Why This is Hard**:
1. **Memory**: Can't fit attention matrix in GPU memory
2. **Compute**: Quadratic growth unsustainable
3. **Bandwidth**: Memory bandwidth bottleneck

### Solutions

**1. Efficient Attention (FlashAttention)**:
- Tiling and fusion
- O(n) memory instead of O(n²)
- 10-20x memory savings
- Enables 128K+ contexts

**2. Position Encoding That Scales**:
- RoPE with interpolation/extrapolation
- ALiBi (linear biases)
- Can extend beyond training length

**3. Sparse Attention**:
- Don't attend to all positions
- Sliding windows + occasional global
- Reduces from O(n²) to O(n×w)

**4. Optimized Implementations**:
- FlashAttention-2: 2x faster
- FlashAttention-3: Another 1.5-2x faster (H100)
- Custom kernels for specific patterns

---

## Context Length by Model (2024-2025)

### Open Source Models

| Model | Context Length | Position Encoding | Notes |
|-------|---------------|-------------------|-------|
| Llama 2 | 4K | RoPE | Base length |
| Llama 3 | 8K | RoPE | 2x increase |
| Llama 3.1 | 128K | RoPE + scaling | 16x increase! |
| Mistral 7B | 8K | RoPE | Standard |
| Mixtral 8x7B | 32K | RoPE | Longer context |
| Mixtral 8x22B | 64K | RoPE | 2x Mixtral 8x7B |
| Qwen 2.5 | 128K-1M | RoPE + scaling | Variable by size |
| Qwen 3 | 32K+ | RoPE + QK-Norm | Improved training |
| DeepSeek-V3 | 128K+ | RoPE + MLA | Efficient long context |
| MPT-7B | 8K (65K+ possible) | ALiBi | Excellent extrapolation |

### Proprietary Models

| Model | Context Length | Notes |
|-------|---------------|-------|
| GPT-4 (initial) | 8K-32K | Two variants |
| GPT-4 Turbo | 128K | 8x increase |
| GPT-4o | 128K | Maintained long context |
| GPT-5 (reported) | 400K input, 128K output | Massive input context |
| Claude 3 Opus | 200K (1M available) | Production: 200K |
| Claude 3.5 Sonnet | 200K | Maintained |
| Gemini 1.5 Pro | 1M (2M waitlist) | Largest production |
| Gemini 2.0 Flash | 1M | Maintained + faster |

**Leaders**:
- Gemini 1.5/2.0: 1-2M tokens (production)
- GPT-5: 400K tokens (reported)
- Claude 3: 200K-1M tokens

---

## Enabling Technologies

### FlashAttention Evolution

**FlashAttention 1 (2022)**:
- First efficient long-context attention
- O(n) memory vs O(n²)
- 10-20x memory savings
- Enabled 8K-32K contexts

**FlashAttention 2 (2024)**:
- 2x faster than FA1
- Better parallelization
- 50-73% of theoretical max FLOPs
- Enabled 128K contexts

**FlashAttention 3 (2024)**:
- Optimized for H100
- 1.5-2x faster than FA2
- FP8 support (2.6x lower error)
- Up to 740 TFLOPS (75% H100 utilization)
- **Enabled 1M+ token contexts**

**Impact**: Without FlashAttention, million-token contexts would be impractical.

### RoPE Scaling Techniques

**Challenge**: Train on 4K, deploy on 128K

**Linear Interpolation**:
```python
# Compress positions before rotation
scaled_pos = position * (train_length / target_length)

# Example: Train 4K, deploy 128K
scaled_pos = position * (4096 / 131072)  # Compress 32x
```

**Pros**: Simple, works reasonably well
**Cons**: Some quality degradation

**NTK-Aware Scaling**:
```python
# Adjust base frequency θ
theta_new = theta_old * (target_len / train_len) ** (d / (d - 2))
```

**Pros**: Better preservation of local patterns
**Cons**: More complex

**YaRN (Yet another RoPE extensioN)**:
- Different scaling for different frequency bands
- Low frequencies: Global position
- High frequencies: Local patterns
- Best quality for extreme scaling

**LongRoPE (2024 Research)**:
- Can extend to 2M+ tokens
- Refined scaling strategies
- State-of-the-art extrapolation

### ALiBi Advantage

**Attention with Linear Biases**:
- No position embeddings at all
- Add distance-based bias to attention
- **Extraordinary extrapolation**: Train 512 → Test 3072+ tokens

**Why ALiBi is Great for Long Context**:
1. **Natural extrapolation**: No special scaling needed
2. **Simple**: Just add bias
3. **Proven**: BLOOM trained on 2K, works on much longer

**With FlashAttention 2.4**:
- 4-5x speedup for ALiBi
- Makes ALiBi practical at scale
- Could see renaissance for long-context models

---

## Context Length Scaling Strategies

### Progressive Training

**Qwen 3 Approach**:
```
Stage 1: Train on 4K tokens (general knowledge)
Stage 2: Train on 4K tokens (STEM/coding)
Stage 3: Extend to 32K tokens (long context)
```

**Benefits**:
- Cheaper initial training (4K quadratic << 32K quadratic)
- Build capabilities before extending
- Focused long-context training

### Continued Pretraining

**Llama 3 → 3.1**:
```
Llama 3: Trained on 8K
Llama 3.1: Continued training on 128K
```

**Approach**:
- Start with strong 8K model
- Continue training with longer sequences
- RoPE scaling to enable extension

### Context Window Expansion in Practice

**Typical Recipe**:
1. Train base model on standard context (2K-8K)
2. Apply position encoding scaling (RoPE interpolation)
3. Continued training on longer sequences
4. Gradually increase sequence length
5. Validate on long-context benchmarks

---

## Long Context Use Cases

### What Can You Do With Long Context?

**200K tokens ≈ 150,000 words ≈ 300 pages**

**4K tokens (3,000 words)**:
- ✅ Short conversation
- ✅ Single article
- ❌ Long document
- ❌ Codebase

**32K tokens (24,000 words)**:
- ✅ Multiple articles
- ✅ Medium conversation
- ✅ Small code files
- ❌ Entire book
- ❌ Large codebase

**128K tokens (96,000 words)**:
- ✅ Entire novella
- ✅ Large conversation
- ✅ Medium codebase
- ✅ Multiple documents
- ❌ Very large codebases
- ❌ Multiple books

**1M tokens (750,000 words)**:
- ✅ Multiple books
- ✅ Entire large codebase
- ✅ Full chat history
- ✅ Comprehensive research
- ✅ Video (hours of transcripts)

**10M tokens (7.5M words)**:
- ✅ Massive repositories
- ✅ Entire documentation sets
- ✅ Years of conversation
- ✅ Comprehensive knowledge bases

---

## Practical Considerations

### Memory Requirements

**KV Cache Dominates Memory**:
```
For model with GQA (8 KV heads, 128 head_dim):
Context = 128K tokens
Batch = 1
Precision = FP16 (2 bytes)

KV cache = 2 (K,V) × 128K × 8 × 128 × 2 bytes
         = 524 MB per layer

For 80 layers: 41.9 GB just for KV cache!
```

**Solutions**:
1. **GQA/MQA**: Reduce KV heads (4-8x savings)
2. **MLA**: Compress to latent space (5-10x savings)
3. **Quantization**: INT8/FP8 (2x savings)
4. **Paged Attention**: Efficient memory management

### Inference Cost

**Longer Context = Higher Cost**:
- Quadratic attention (even with FlashAttention)
- Larger KV cache
- More memory bandwidth

**Example Pricing** (Typical):
```
Input: $3 per 1M tokens (short context)
Input: $6 per 1M tokens (128K context)
Input: $12 per 1M tokens (1M context)

Cost scales with context capability
```

### Quality vs Length Trade-off

**Observations**:
1. **Needle in haystack**: Models struggle with perfect recall
2. **Middle curse**: Harder to recall from middle of context
3. **Degradation**: Some quality loss at extreme lengths

**Best Practices**:
- Put important information at start and end
- Use retrieval for very long documents
- Test on your specific use case

---

## Benchmarks and Evaluation

### Long Context Benchmarks

**Needle in a Haystack**:
- Hide "needle" (specific fact) in "haystack" (long context)
- Ask model to retrieve it
- Test recall at different positions and depths

**LongBench**:
- Real-world long-context tasks
- Multi-document QA
- Code repository understanding
- Long conversation

**RULER**:
- Variable-length evaluation
- Different task types
- Comprehensive long-context testing

### Model Performance

**Generally**:
- Claude 3: Excellent long-context recall
- Gemini 1.5/2.0: Strong on very long contexts
- GPT-4 Turbo: Good, some degradation at extremes
- Llama 3.1: Competitive at 128K
- Qwen 2.5: Strong long-context performance

**Trend**: Getting better, but not perfect

---

## Future Directions

### Research Areas

**1. Infinite Context**:
- Can we handle unlimited length?
- Memory-augmented models
- Hierarchical processing

**2. Better Recall**:
- Solve "middle curse"
- Perfect retrieval from millions of tokens
- Attention improvements

**3. Efficiency**:
- Linear attention (O(n) not O(n²))
- Constant-time attention?
- Smarter sparse patterns

**4. Quality**:
- No degradation at long lengths
- Better utilization of full context

### Emerging Techniques

**RingAttention**:
- Distributed attention across devices
- Theoretically infinite context
- Communication overhead challenges

**StreamingLLM**:
- Keep first K and last N tokens
- Drop middle for efficiency
- Good for very long generations

**Retrieval-Augmented**:
- Don't put everything in context
- Retrieve relevant chunks
- Hybrid approach

---

## Practical Recommendations

### Choosing Context Length

**For Applications**:

**Chatbots**: 8K-32K sufficient for most
**Document Analysis**: 32K-128K for comprehensive
**Code Understanding**: 128K+ for large repos
**Research**: 1M+ for comprehensive analysis

**Cost Consideration**:
- Longer context = higher cost
- Use minimum needed
- Retrieval for very long documents

### Optimization Strategies

**1. Context Packing**:
- Fit multiple examples in one context
- Amortize fixed costs
- Better throughput

**2. Caching**:
- Cache common prefixes (system prompts)
- Reuse KV cache
- Significant savings

**3. Chunking**:
- Split very long documents
- Process in pieces
- Combine results

**4. Hybrid Approach**:
- Retrieval + long context
- Best of both worlds
- Cost-effective

---

## The Race to Longer Context

### Why Everyone Wants Long Context

**1. User Demand**:
- Process entire documents
- Long conversations
- Complex analysis

**2. Competitive Advantage**:
- Differentiation
- New use cases
- Marketing

**3. Technical Achievement**:
- Demonstrates capability
- Research advancement
- Pushes boundaries

### Current Leaders

**Production**:
- Gemini 1.5/2.0: 1M-2M tokens (available now)
- Claude 3: 200K standard, 1M available
- GPT-4 Turbo/4o: 128K tokens

**Reported/Upcoming**:
- GPT-5: 400K tokens
- Llama 4: Possibly 10M+ tokens

### Diminishing Returns?

**Question**: Do we need 10M tokens?

**Arguments For**:
- Entire codebases
- Comprehensive analysis
- No chunking needed

**Arguments Against**:
- Retrieval often better
- Quality degradation
- Cost prohibitive
- Most use cases don't need it

**Reality**: Probably peaks around 1M-10M for practical use

---

## Conclusion

Context windows have evolved dramatically:
- **2018**: 1K tokens
- **2025**: 1M-10M tokens
- **Growth**: 1000-10000x in ~7 years

**Key Enablers**:
- FlashAttention and variants
- RoPE/ALiBi position encodings
- Better hardware (H100, etc.)
- Algorithmic improvements

**Future**: Likely plateau around 1M-10M tokens for most use cases, with continued efficiency improvements.

---

## Sources

- [Context Window Evolution](https://research.trychroma.com/context-rot)
- [Towards Infinite LLM Context Windows](https://towardsdatascience.com/towards-infinite-llm-context-windows-e099225abaaf/)
- [LLMs with Largest Context Windows](https://codingscape.com/blog/llms-with-largest-context-windows)
- [LongRoPE: Extending LLM Context Beyond 2M Tokens](https://arxiv.org/abs/2402.13753)
- [Scaling to Millions of Tokens](https://developer.nvidia.com/blog/scaling-to-millions-of-tokens-with-efficient-long-context-llm-training/)
- FlashAttention papers (2022, 2023, 2024)
- Model documentation and papers
