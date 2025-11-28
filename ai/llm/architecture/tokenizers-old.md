# Tokenizers

Tokenization is the process of converting text into numerical tokens that models can process. The choice of tokenizer significantly impacts model efficiency, multilingual support, and performance.

## Why Tokenization Matters

**Text → Tokens → Model → Tokens → Text**

Tokenization affects:
1. **Vocabulary size**: Model embedding size
2. **Sequence length**: Longer sequences = more compute
3. **Multilingual support**: How well non-English works
4. **Efficiency**: Tokens per text unit
5. **Model performance**: Subtle but real impacts

---

## Byte-Pair Encoding (BPE)

### Foundation for Most Tokenizers

**Core Algorithm**:
1. Start with character-level vocabulary
2. Find most frequent adjacent pair
3. Merge pair into new token
4. Repeat until target vocabulary size

**Example**:
```
Text: "lower lower lower lowest"
Initial: ['l', 'o', 'w', 'e', 'r', ...]

Iteration 1:
Most frequent pair: ('e', 'r') → 'er'
Result: ['l', 'o', 'w', 'er', ...]

Iteration 2:
Most frequent pair: ('l', 'o') → 'lo'
Result: ['lo', 'w', 'er', ...]

Continue merging...
Final: ['low', 'er', 'low', 'est']
```

**Properties**:
- Data-driven vocabulary
- Balances word-level and character-level
- Can handle unseen words (fall back to characters)
- Deterministic encoding

### Advantages

1. **Handles unknown words**: Can tokenize anything
2. **Efficient**: Good compression of common text
3. **Language-agnostic**: Works for any language
4. **Data-driven**: Learns from corpus

### Disadvantages

1. **Context-independent**: "new" in "news" vs "new car" same tokenization
2. **Greedy**: Locally optimal merges might not be globally optimal
3. **Corpus-dependent**: Different corpora → different vocabularies

---

## SentencePiece

### Used By: Llama 1/2, Mistral, Yi, Qwen (early), Most Open-Source Models

**Key Innovation**: Treats text as raw stream, includes whitespace

**Differences from word-based BPE**:
- Works on Unicode code points
- Treats spaces as special character (▁)
- Language-independent from the start

**Example**:
```
Text: "Hello world"
SentencePiece: ['▁Hello', '▁world']  # ▁ represents space

Text: "你好世界" (Chinese)
SentencePiece: ['▁你好', '▁世界']  # Works same way
```

**Features**:
- No preprocessing required
- Truly language-agnostic
- Reversible encoding (preserves spaces)
- Subword regularization (for training)

### Advantages

1. **Language-independent**: No language-specific preprocessing or assumptions about text structure
2. **Handles any text**: Unicode support
3. **Reversible**: Can exactly recover original text (lossless encoding)
4. **Training features**: Subword regularization for robustness
5. **Scalable**: Can scale from 32K to 256K vocabulary (proven by Gemma 2)
6. **Mature and stable**: Well-understood, battle-tested in production
7. **Backward compatibility**: Existing models maintain tokenizer for consistency

### Implementation

```python
import sentencepiece as spm

# Train tokenizer
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    character_coverage=0.9995,
    model_type='bpe'
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')

tokens = sp.encode_as_pieces("Hello world")
# ['▁Hello', '▁world']
```

---

## tiktoken

### Used By: GPT-4, Llama 3, GPT-3.5

**Key Innovation**: UTF-8 byte-level BPE

**Approach**:
- Works on UTF-8 bytes, not code points
- 256 bytes as base vocabulary
- Builds up from there with BPE

**Differences from SentencePiece**:
```
SentencePiece: Works on code points (characters)
tiktoken: Works on UTF-8 bytes

Example: "你好" (Chinese)
SentencePiece: Treats as 2 code points
tiktoken: Treats as UTF-8 bytes (6 bytes)
```

**Advantages**:
1. **Can encode anything**: Any byte sequence is valid
2. **Fast**: Optimized Rust implementation
3. **No special cases**: Uniform handling
4. **Open source**: Available for use

### tiktoken Encodings

**cl100k_base** (GPT-4, GPT-3.5-turbo):
- ~100,000 tokens
- Used in GPT-4
- Better multilingual support than earlier versions

**o200k_base** (Newer models):
- ~200,000 tokens
- Even better efficiency

### Vocabulary Evolution in OpenAI Models

| Model | Encoding | Vocab Size |
|-------|----------|-----------|
| GPT-2 | GPT-2 BPE | ~50K |
| GPT-3 | GPT-2 BPE | ~50K |
| GPT-3.5/4 | cl100k_base | ~100K |
| Newer | o200k_base | ~200K |

**Trend**: Larger vocabularies over time

### Implementation

```python
import tiktoken

# GPT-4 tokenizer
enc = tiktoken.encoding_for_model("gpt-4")

# Encode
tokens = enc.encode("Hello, world!")
# [9906, 11, 1917, 0]

# Decode
text = enc.decode(tokens)
# "Hello, world!"

# Count tokens
num_tokens = len(enc.encode("Some text"))
```

---

## The Tokenization Transition Story

### Phase 1: Early BPE Era (2017-2020)

**Original Transformer** (2017):
- Used both **BPE** and **WordPiece** depending on task
- English-German: BPE (~37K vocab)
- English-French: WordPiece (32K vocab)
- Reflected experimental nature of early work

**GPT-1** (2018):
- First to standardize **Byte Pair Encoding (BPE)**
- Vocabulary: ~40K tokens (estimated)
- Established BPE as standard for GPT series

**BERT** (2018):
- Used **WordPiece** tokenizer
- Vocabulary: **30,522 tokens** (often rounded to "30K")
- Became standard for encoder models
- Special tokens: [PAD], [CLS], [SEP], [MASK]

**GPT-2** (2019):
- **Byte-level BPE** (innovation)
- Vocabulary: **50,257 tokens** (256 base bytes + merges)
- Handles any Unicode without unknown tokens
- Set new standard for decoder-only models

**GPT-3** (2020):
- Same tokenizer as GPT-2
- Vocabulary: **50,257 tokens**
- No tokenization innovation - focused on scaling

**T5** (2019):
- **SentencePiece** tokenizer
- Vocabulary: **32,000 tokens**
- First major model to adopt SentencePiece
- Beginning of SentencePiece's rise

**Why This Era Mattered**:
- Established BPE as the foundational algorithm
- WordPiece vs BPE competition
- 30-50K became the standard vocabulary range
- Byte-level approach solved unknown token problem

---

### Phase 2: SentencePiece Dominance (2020-2023)

**Why SentencePiece Won**:
1. **Language independence**: Treats text as raw stream, no whitespace assumptions
2. **Lossless tokenization**: Preserves spaces perfectly
3. **No preprocessing**: Simplified pipeline
4. **Critical for multilingual**: Chinese, Japanese, Thai don't use spaces
5. **Simpler**: Fewer language-specific rules

**BLOOM** (2022):
- Tokenizer: **SentencePiece BPE**
- Vocabulary: **250,880 tokens** (massive!)
- 46 languages supported
- Showed large vocabs necessary for true multilingual support

**OPT** (Meta, 2022):
- Tokenizer: GPT-2 BPE
- Vocabulary: 50,272 tokens

**Falcon** (2023):
- Tokenizer: Custom BPE
- Vocabulary: **65,024 tokens**
- Trend toward larger vocabularies

**Llama 1** (February 2023) - **The Standardizer**:
- Tokenizer: **SentencePiece BPE**
- Vocabulary: **32,000 tokens**
- **Established the "32K standard"** for open-source LLMs

**Llama 2** (July 2023):
- Tokenizer: **SentencePiece BPE**
- Vocabulary: **32,000 tokens**
- Maintained the 32K standard

**Mistral 7B & Mixtral** (2023):
- Tokenizer: **SentencePiece BPE** (v1 tokenizer)
- Vocabulary: **32,000 tokens**
- Later: Mistral v3 expanded to 32,768 tokens

**Why 32K Became Standard**:
1. **Historical precedent**: BERT (30K), T5 (32K) set the pattern
2. **GPU efficiency**: 32,768 (2^15) is GPU-friendly
3. **Good balance**: Coverage without excessive embedding costs
4. **Computational constraints**: 32K × 4096 dims = ~130M embedding parameters
5. **Convention over optimization**: Research now shows 32K was too small

**Limitation**: English-centric - poor multilingual efficiency (3-6x worse for Chinese)

---

### Phase 3: tiktoken Revolution (2023)

**tiktoken Introduction** (2023):
- **Developer**: OpenAI
- **Innovation**: Fast BPE optimized for GPT models
- **Key difference**: Skips merge rules when token already in vocab
- "Violates" standard BPE but improves efficiency

**GPT-3.5 & GPT-4** (2023):
- Tokenizer: **tiktoken** with **cl100k_base** encoding
- Vocabulary: **~100,000 tokens**
- **Breakthrough**: First major jump beyond 32-64K range
- Significantly better compression than 50K vocab

**Why the Switch from SentencePiece to tiktoken**:
1. **Larger vocabularies** (100K vs 32K) for better compression
2. **Optimized for speed**: "Fast BPE tokenizer"
3. **Better multilingual efficiency** with larger vocab
4. **OpenAI control**: Vertical integration of pipeline
5. **Prepares for scaling**: Enables even larger vocabularies

**Llama 3** (April 2024) - **The Game Changer**:
- Tokenizer: **tiktoken-based BPE**
- Vocabulary: **128,000 tokens** (specifically 128,256)
- **Revolutionary shift**: Meta abandoned SentencePiece for tiktoken
- **4x vocabulary expansion** from Llama 2
- Trade-off: Model grew from 7B → 8B parameters (larger embeddings)

**Impact of Llama 3's Switch**:
- Validated tiktoken for open-source models
- Made 128K the new standard for frontier models
- Proved multilingual efficiency gains worth embedding cost
- Most new models followed Llama 3's lead

**Why Llama 3 Switched**:
1. **Multilingual push**: 4x better compression for non-English
2. **Code efficiency**: Better tokenization of programming languages
3. **Following GPT-4**: Competitive pressure
4. **Research validation**: 32K proven suboptimal
5. **Future-proofing**: Enables better scaling

---

### Phase 4: Ultra-Large Vocabulary Era (2024-2025)

**The 100-128K Standard** (2024):

**Qwen 2 / 2.5**:
- Tokenizer: **tiktoken BPE**
- Vocabulary: **151,646-151,936 tokens**
- Optimized for Chinese and multilingual

**DeepSeek V2**:
- Tokenizer: **Byte-level BPE**
- Vocabulary: **100,000 tokens**

**DeepSeek V3** (2024):
- Tokenizer: **Byte-level BPE**
- Vocabulary: **~128,000 tokens** (129,280 precisely)
- Follows Llama 3's lead

**Phi-3** (Microsoft):
- Mini: 32,064 tokens
- Small: **100,000 tokens** with tiktoken

**Phi-4** (Microsoft, 2024):
- 14B: **100,352 tokens**
- Mini: **200,064 tokens**

**The 200-256K Frontier**:

**GPT-4o** (2024):
- Tokenizer: **tiktoken** with **o200k_base**
- Vocabulary: **~200,000 tokens**
- 29% better compression than cl100k_base
- 4x improvement for Indic languages (Malayalam, Telugu, Kannada)

**Gemma 2 / Gemini** (2024) - **The Record Holder**:
- Tokenizer: **SentencePiece** (largest implementation)
- Vocabulary: **256,000 tokens**
- Still using SentencePiece, proving it can scale
- Optimized for heavily multilingual use

**Current State** (2025):
- **RoPE**: ~80% of new models use 100K+ vocabularies
- **100-128K**: New standard (Llama 3/4, DeepSeek, Qwen)
- **200-256K**: Specialized heavily multilingual (GPT-4o, Gemma)
- **32K**: Legacy (older models, backward compatibility)

---

### Timeline Summary

```
2017: Mixed (BPE vs WordPiece experimentation)
    ↓
2018: GPT-1 (BPE ~40K), BERT (WordPiece 30K)
    ↓
2019: GPT-2 (Byte-level BPE 50K), T5 (SentencePiece 32K)
    ↓
2020: GPT-3 (50K), SentencePiece emerging
    ↓
2022: BLOOM (SentencePiece 250K - outlier)
      OPT (50K), Falcon (65K)
    ↓
2023: Llama 1/2 (SentencePiece 32K) - "32K standard"
      Mistral (SentencePiece 32K)
      GPT-4 (tiktoken 100K) - BREAKTHROUGH
    ↓
2024: Llama 3 (tiktoken 128K) - SHIFT TO TIKTOKEN
      Qwen 2.5 (tiktoken 152K)
      DeepSeek V2/V3 (100-128K)
      Phi-3/4 (100-200K)
      GPT-4o (tiktoken 200K)
      Gemma 2 (SentencePiece 256K)
    ↓
2025: 100-128K standard
      Llama 4 (tiktoken 128K)
      Continued vocabulary growth
```

---

## Tokenizers by Model (2017-2025)

### Early BPE Era (2017-2020)

| Model | Year | Tokenizer Type | Vocabulary Size | Notes |
|-------|------|---------------|----------------|-------|
| **Original Transformer** | 2017 | BPE / WordPiece | 32-37K | Mixed approach |
| **GPT-1** | 2018 | BPE | ~40K | First BPE standard |
| **BERT** | 2018 | WordPiece | 30,522 | Encoder standard |
| **GPT-2** | 2019 | Byte-level BPE | 50,257 | Innovation |
| **T5** | 2019 | SentencePiece | 32,000 | First major SentencePiece |
| **GPT-3** | 2020 | Byte-level BPE | 50,257 | Same as GPT-2 |

### SentencePiece Dominance Era (2020-2023)

| Model | Year | Tokenizer Type | Vocabulary Size | Notes |
|-------|------|---------------|----------------|-------|
| **GPT-NeoX-20B** | 2022 | BPE (GPT-2-like) | 50,257 | Similar to GPT-2 |
| **OPT** | 2022 | GPT-2 BPE | 50,272 | Slight expansion |
| **BLOOM** | 2022 | SentencePiece BPE | 250,880 | Massive multilingual |
| **Falcon** | 2023 | Custom BPE | 65,024 | Larger vocab trend |
| **Llama 1** | 2023 | SentencePiece BPE | 32,000 | **32K standard** |
| **Llama 2** | 2023 | SentencePiece BPE | 32,000 | Maintained standard |
| **Mistral 7B** | 2023 | SentencePiece BPE | 32,000 | v1 tokenizer |
| **Mixtral 8x7B** | 2023 | SentencePiece BPE | 32,000 | v1 tokenizer |
| **Yi** | 2023 | SentencePiece BPE | 64,000 | Larger for multilingual |

### tiktoken Revolution Era (2023-2024)

| Model | Year | Tokenizer Type | Vocabulary Size | Notes |
|-------|------|---------------|----------------|-------|
| **GPT-3.5/4** | 2023 | tiktoken (cl100k_base) | ~100,000 | **Breakthrough** |
| **Llama 3** | 2024 | tiktoken BPE | 128,256 | **Meta switches to tiktoken** |
| **Llama 3.1/3.2/3.3** | 2024 | tiktoken BPE | 128,256 | Maintained |
| **Qwen 2** | 2024 | tiktoken BPE | 151,646 | Optimized for Chinese |
| **DeepSeek V2** | 2024 | Byte-level BPE | 100,000 | MoE architecture |
| **Phi-3 Small** | 2024 | tiktoken | 100,000 | Microsoft |

### Ultra-Large Vocabulary Era (2024-2025)

| Model | Year | Tokenizer Type | Vocabulary Size | Notes |
|-------|------|---------------|----------------|-------|
| **Qwen 2.5** | 2024 | tiktoken BPE | 151,936 | Enhanced multilingual |
| **DeepSeek V3** | 2024 | Byte-level BPE | ~129,280 | 671B params |
| **Phi-4 (14B)** | 2024 | tiktoken | 100,352 | Microsoft |
| **Phi-4 Mini** | 2024 | tiktoken | 200,064 | Large vocab |
| **GPT-4o** | 2024 | tiktoken (o200k_base) | ~200,000 | 29% better compression |
| **Gemma 2** | 2024 | SentencePiece | 256,000 | **Largest!** |
| **Llama 4** | 2025 | tiktoken BPE | 128,256 | Maintained Llama 3 approach |
| **Mistral v3** | 2024 | SentencePiece | 32,768 | Slight expansion |

### Proprietary Models (Details Known)

| Model | Year | Tokenizer Type | Vocabulary Size | Notes |
|-------|------|---------------|----------------|-------|
| **Claude 3** | 2024 | Custom BPE | ~65,000 | 70% overlap with GPT-4 |
| **Gemini** | 2024 | SentencePiece | 256,000 | Same as Gemma 2 |

---

## Current Tokenization Consensus (2024-2025)

### tiktoken-Based with 100K+ Vocab: New Standard

**Adoption**: ~70% of new frontier models

**Major Models**:
- Llama 3/3.1/3.2/3.3/4 (128K)
- Qwen 2/2.5 (152K)
- DeepSeek V2/V3 (100-128K)
- Phi-3/4 (100-200K)
- GPT-4, GPT-4o (100-200K)

**Why tiktoken Won**:
1. **The "Llama 3 effect"**: Meta's switch from SentencePiece validated tiktoken for open-source
2. **GPT-4's proven success**: Demonstrated reliability at massive scale
3. **Network effects**: Once leaders adopted it, followers converged quickly

### SentencePiece: Still Viable

**Adoption**: ~20% of models, including some largest

**Major Models**:
- Mistral/Mixtral (32K, legacy)
- Gemma 2 / Gemini (256K - largest implementation!)
- Some multilingual models

**Why SentencePiece Still Thrives**:
- **Ecosystem inertia**: Billions invested in models that can't switch tokenizers
- **Research continuity**: Enables fair comparison across model generations
- **Conservative choice**: Proven stability for production deployments

### The Vocabulary Size Consensus

**By Use Case**:
- **English-focused**: 32-64K (legacy, sufficient)
- **General purpose**: 100-128K (current standard)
- **Multilingual-focused**: 150-200K (Qwen, GPT-4o)
- **Heavily multilingual**: 250-256K (BLOOM, Gemma 2)

**Trade-offs Understood**:
- **Cost**: 128K vocab adds ~500M parameters vs 32K
- **Benefit**: 20-40% compression, 3-6x better for non-English
- **Break-even**: Efficiency gains outweigh embedding costs at scale

**Key Insight**: Like activation functions (SwiGLU), normalization (RMSNorm), and position encoding (RoPE), tokenization has converged on a clear trend (tiktoken + 100K+ vocab), driven by multilingual requirements and the "Llama 3 effect."

---

## Why Transitions Happened

### Early BPE → SentencePiece (2019-2020)

**Drivers**:
1. **Multilingual necessity**: WordPiece assumes spaces, fails for CJK languages
2. **Preprocessing complexity**: SentencePiece eliminates language-specific rules
3. **Lossless requirement**: Perfect reversibility needed
4. **T5's success**: Google's validation mattered
5. **Simplicity**: One tool for all languages

**Impact**: SentencePiece became standard for open-source multilingual models (2020-2023)

### SentencePiece 32K → tiktoken 100K+ (2023-2024)

**Drivers**:
1. **Research proved 32K too small**: Llama 2-70B should have had 216K (7x larger)
2. **Multilingual inefficiency**: 3-6x more tokens for Chinese with 32K
3. **GPT-4's success**: Proved 100K viable and beneficial
4. **Competitive pressure**: Open models had to match GPT-4's efficiency
5. **Hardware improvements**: Modern GPUs handle larger embedding matrices
6. **Fairness concerns**: Under-tokenized languages perform worse and cost more

**The Llama 3 Effect**:
- Meta's switch from SentencePiece (32K) to tiktoken (128K) validated the transition
- Most subsequent models followed: Qwen, DeepSeek, Phi
- 128K became the new "standard" like 32K was before

**Quantified Benefits**:
- **English**: 10-15% compression improvement
- **Chinese**: 40% length reduction, 3-6x efficiency gain
- **Indic languages**: 3-4x improvement (GPT-4o)
- **Code**: 20-30% better tokenization
- **Overall**: 20-40% fewer tokens for same text

### 100K → 200-256K (2024)

**Drivers**:
1. **Heavily multilingual models**: BLOOM, Gemma, GPT-4o
2. **Diminishing returns understood**: Beyond 256K not worth it yet
3. **Specialized use cases**: Global products need equal language support
4. **Research shows**: Optimal vocab scales with model size
5. **Competition**: Push for "best multilingual" claims

**Open Question**: Will 256K become standard, or is 100-128K the sweet spot?

---

## Multilingual Efficiency: The Driving Force

### The Problem with Small Vocabularies

**Example: Chinese Text with Different Vocabularies**

**32K vocabulary** (Llama 2):
```
"你好世界" (Hello World) → 8-12 tokens
Character-level fallback, very inefficient
```

**128K vocabulary** (Llama 3):
```
"你好世界" → 2-4 tokens
Proper subword representation, 3-6x better
```

**Quantified Impact**:
- **Chinese**: 3-6x more efficient with 128K vocab
- **Korean, Arabic**: 2-3x improvement
- **Indic languages**: 3-4x improvement (GPT-4o's 200K vocab)
- **European languages**: 30% improvement
- **Code**: 20-30% better

### Why This Matters

**User Impact**:
1. **Cost**: Token-based pricing means non-English users pay 3-6x more with small vocabs
2. **Context**: More efficient tokenization = more text fits in context window
3. **Performance**: Models perform better when not wasting capacity on sub-tokens
4. **Fairness**: Under-tokenized languages get worse service

**Model Impact**:
1. **Inference speed**: Fewer tokens = faster processing
2. **Memory**: Shorter sequences fit in memory
3. **Quality**: Model doesn't waste capacity learning to compose sub-tokens

**Scaling Laws**: Research shows optimal vocabulary size scales with:
- Model size (larger models deserve larger vocabs)
- Training compute (more training justifies more embedding parameters)
- Language coverage (multilingual requires larger vocabs)

---

## Vocabulary Size Evolution

### Historical Trend

```
2018-2020: ~30-50K tokens
  - GPT-2: 50K
  - BERT: 30K
  - Early models

2021-2022: ~32-64K tokens
  - Llama 1/2: 32K
  - Many open models: 32-64K

2023: ~100-128K tokens
  - GPT-4: ~100K (cl100k)
  - Llama 3: ~128K (tiktoken)
  - Phi-3-small: 100K

2024: ~150-256K tokens
  - Qwen 2.5: ~152K
  - Gemma 2: 256,128 tokens (largest!)

Trend: Exponential expansion
```

## Special Tokenizer Features

### Arcade100k (StableLM)

**Based on**: tiktoken cl100k_base (GPT-4's tokenizer)
**Extension**: Modified for StableLM specific needs

**Special feature**: Digits split into individual tokens
```
"12345" → ['1', '2', '3', '4', '5']
```

**Rationale**: Better mathematical reasoning

### Llama 3 Tokenizer

**Base**: tiktoken
**Vocabulary**: ~128,000 tokens

**Improvements over Llama 2**:
- 4x larger vocabulary (32K → 128K)
- Better multilingual tokenization
- More efficient encoding
- Same byte-level approach as GPT-4

**Impact**: Significant improvement in non-English performance

---

## Tokenization Impact on Model Performance

### Sequence Length

**More tokens = More compute**:
```
Text: "Hello, how are you?"

Tokenizer A: ['Hello', ',', 'how', 'are', 'you', '?'] = 6 tokens
Tokenizer B: ['Hello,', 'how', 'are', 'you?'] = 4 tokens

4 tokens = ~33% less compute than 6 tokens!
```

**Real Impact**:
- Longer sequences = quadratic attention cost O(n²)
- Fewer tokens = faster inference and training
- Critical for long-context models

### Out-of-Vocabulary Handling

**Good tokenizer**: Graceful degradation to subwords/characters
**Poor tokenizer**: Many UNK tokens, information loss

---

## Tokenizer Design Trade-offs

### Vocabulary Size

**Small (30-50K)**:
- ✅ Smaller embedding matrix (fewer parameters)
- ✅ Faster output projection
- ✅ Lower training cost (fewer tokens to learn)
- ❌ More tokens per text (less efficient encoding)
- ❌ Poor multilingual support
- ❌ Worse compression for rare words

**Large (100-256K)**:
- ✅ Fewer tokens per text (better compression)
- ✅ Better multilingual efficiency (3-6x for non-English)
- ✅ Handles rare words and technical terms as single tokens
- ✅ Better handling of code, math, specialized domains
- ❌ Larger embedding matrix and output layer (more parameters)
- ❌ Slower output projection
- ❌ Higher training cost
- ❌ Diminishing returns beyond certain point

### Granularity

**Word-level**:
- Fast encoding
- Compact for common text
- Huge vocabulary needed
- Can't handle unseen words

**Character-level**:
- Tiny vocabulary
- Handles anything
- Very long sequences
- Model must learn composition

**Subword (BPE)**:
- ✅ Balanced approach
- ✅ Handles rare words
- ✅ Efficient encoding
- ✅ **Industry standard**

---

## Practical Considerations

### Training a Tokenizer

**Key Decisions**:

1. **Vocabulary Size**:
   - Small model (<10B): 32-64K
   - Medium (10-100B): 64-128K
   - Large (>100B): 128-256K

2. **Training Data**:
   - Must match model training data
   - Multilingual: Include all languages
   - Balanced representation

3. **Special Tokens**:
   - `<s>`, `</s>`: Start/end of sequence
   - `<pad>`: Padding
   - `<unk>`: Unknown (though rare with BPE)
   - Task-specific tokens

### Tokenizer Compatibility

**Important**: Can't change tokenizer after training
- Tokenizer vocabulary baked into model
- Embedding matrix size fixed
- Changing = retraining from scratch

**Fine-tuning**: Must use same tokenizer as base model

---

## Future Directions

### Research Areas

1. **Byte-level models**: Skip tokenization entirely (ByT5, etc.)
2. **Learned tokenization**: Train tokenizer end-to-end with model
3. **Context-sensitive**: Different tokenization based on context
4. **Multimodal tokenizers**: Unified for text, image, audio

### Trends

1. **Larger vocabularies**: 256K+ tokens
2. **Better multilingual**: Equal treatment of all languages
3. **Efficiency focus**: Minimize tokens per text
4. **Standardization**: tiktoken, SentencePiece dominate

### Open Questions

1. Optimal vocabulary size for given model size?
2. Can we do better than BPE?
3. How to handle code vs natural language optimally?
4. Should digits/numbers be special-cased?

---

## Recommendations

### For Training New Models

**General Purpose**:
- Use tiktoken (like Llama 3)
- 100-128K vocabulary
- Balanced multilingual training data

**Multilingual Focus**:
- Larger vocabulary (128-256K)
- Careful language balancing
- Test on all target languages

**Code-Heavy**:
- Consider special handling for syntax
- Larger vocabulary for identifiers
- Test on code benchmarks

### For Applications

**Be Aware of Tokenizer**:
- Counts tokens, not words
- Different models = different token counts
- Affects API pricing

**Testing**:
```python
# Always test your text with actual tokenizer
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
print(len(enc.encode("Your text here")))
```

---

## Sources

### Foundational Papers
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762) - Original Transformer
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - WordPiece 30K
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) - 50K BPE

### Tokenizer Libraries
- [SentencePiece - Google](https://github.com/google/sentencepiece)
- [tiktoken - OpenAI](https://github.com/openai/tiktoken)

### Technical Deep Dives
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Let's Build the GPT Tokenizer](https://www.fast.ai/posts/2025-10-16-karpathy-tokenizers.html)
- [Comparing GPT Tokenizers](https://medium.com/@sweety.tripathi13/comparing-gpt-tokenizers-968b60f5a72b)
- [Understanding GPT tokenizers](https://simonwillison.net/2023/Jun/8/gpt-tokenizers/)
- [Llama 3 Tokenization](https://adalkiran.github.io/llama-nuts-and-bolts/12-TOKENIZATION/)
- [tiktoken vs SentencePiece Discussion](https://discuss.huggingface.co/t/what-is-the-difference-between-tiktoken-and-sentencepice-implements-about-bpe/86079)
- [Demystifying Tokenization](https://medium.com/@vanshcodeworks/demystifying-tokenization-the-hidden-language-of-ai-models-from-openais-tiktoken-to-google-s-8ed8bf2132b4)

### Vocabulary Size Research
- [Balancing Vocabulary Size in Modern LLMs](https://www.rohan-paul.com/p/tutorial-balancing-vocabulary-size)
- [Scaling Laws with Vocabulary](https://arxiv.org/abs/2407.13623) - Optimal vocab sizes
- [Large Vocabulary Size Improves Large Language Models](https://arxiv.org/abs/2406.16508)
- [Why LLM Vocabulary Size Matters](https://shekhargulati.com/2024/12/11/why-llm-vocabulary-size-matters/)

### Multilingual Efficiency
- [Chinese Tokenization Efficiency](https://direct.mit.edu/coli/article/51/3/785/128327/Tokenization-Changes-Meaning-in-Large-Language)
- [Accelerating Multilingual Language Model](https://arxiv.org/abs/2401.10660)
- [GPT-4o Tokenizer Analysis](https://leehanchung.github.io/blogs/2024/05/15/gpt-4o-tokenizer/)
- [GPT-4o vs GPT-4 Tokenization](https://llm-calculator.com/blog/gpt-4o-vs-gpt-4-tokenization/)

### BPE vs WordPiece vs SentencePiece
- [WordPiece vs SentencePiece](https://medium.com/@lmpo/from-text-to-tokens-understanding-bpe-wordpiece-and-sentencepiece-in-nlp-1367d9d610af)
- [BPE vs WordPiece vs SentencePiece Guide](https://medium.com/@dhiyaadli/bpe-vs-wordpiece-vs-sentencepiece-a-beginner-friendly-guide-to-subword-tokenization-8047b39d82e0)
- [Exploring BERT's Vocabulary](http://juditacs.github.io/2019/02/19/bert-tokenization-stats.html)
- [BERT WordPiece Tokenizer](https://stackoverflow.com/questions/73232413/why-was-berts-default-vocabulary-size-set-to-30522)

### Model-Specific Documentation
- [GPT-2 Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/gpt2)
- [T5 Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/t5)
- [Llama Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/llama)
- [Llama 2 Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/llama2)
- [Mistral Tokenization Guide](https://docs.mistral.ai/guides/tokenization/)
- [Qwen Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/qwen2)
- [Gemma 2 Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/gemma2)
- [Gemma Explained - Google Developers](https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Phi-3 Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/phi3)
- [Phi-4 Model Page - Hugging Face](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [BLOOM Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/bloom)
- [Papers Explained: BLOOM](https://medium.com/dair-ai/papers-explained-52-bloom-9654c56cd2)
- [GPT-NeoX Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/gpt_neox)
- [OPT Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/opt)
- [Falcon Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/falcon)

### Proprietary Model Analysis
- [The Worst (But Only) Claude 3 Tokenizer](https://javirando.com/blog/2024/claude-tokenizer/)
- [Dissecting Gemini's Tokenizer](https://dejan.ai/blog/gemini-toknizer/)

### Practical Guides
- [LLM Tokenization](https://hundredblocks.github.io/transcription_demo/)
- [Transformers 101: Tokens, Attention, and Beyond](https://medium.com/@mayanksultania/transformers-101-tokens-attention-and-beyond-b080a900ca6c)
