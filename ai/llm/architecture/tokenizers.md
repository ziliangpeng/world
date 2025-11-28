# Tokenizers

Tokenization is the process of converting text into numerical tokens that language models can process. The choice of tokenizer profoundly impacts model efficiency, multilingual capabilities, and overall performance.

---

# Part I: Foundation (Why & What)

## 1. Introduction & Motivation

### What is Tokenization?

**Text → Tokens → Model → Tokens → Text**

Every language model must bridge the gap between human text and numerical computation. Tokenization is that bridge—breaking text into discrete units (tokens) that can be embedded as vectors and processed by neural networks.

### Why Tokenization Matters

The tokenizer is not just a pre-processing step—it's a fundamental architectural choice that affects:

1. **Vocabulary size**: Determines embedding matrix size (# params)
2. **Sequence length**: Fewer tokens = faster inference, less memory
3. **Multilingual support**: How efficiently non-English languages are encoded
4. **Model performance**: Subtle but measurable impacts on downstream tasks
5. **Training efficiency**: Shorter sequences reduce compute costs

### The Core Tension

Tokenization involves fundamental trade-offs:

**Small Vocabulary (30-50K)**:
- ✅ Fewer parameters (smaller embedding matrix)
- ✅ Faster output projection
- ❌ More tokens per text (longer sequences)
- ❌ Poor multilingual support

**Large Vocabulary (100-256K)**:
- ✅ Fewer tokens per text (better compression)
- ✅ Better multilingual efficiency
- ❌ More parameters (larger embedding + output layer)
- ❌ Higher training cost

The industry has evolved from 30K → 50K → 100K → 256K vocabularies as models scale and multilingual AI becomes essential.

---

## 2. How Tokenization Works: Byte-Pair Encoding (BPE)

### Foundation for Most Tokenizers

Nearly all modern tokenizers—SentencePiece, tiktoken, GPT's tokenizer—are built on **Byte-Pair Encoding (BPE)**, a data-driven algorithm that learns subword vocabularies from training data.

### The BPE Algorithm

**Core Algorithm**:
1. Start with character-level vocabulary (or bytes)
2. Find most frequent adjacent pair in corpus
3. Merge that pair into a new token
4. Repeat until target vocabulary size reached

**Example**:
```
Text: "lower lower lower lowest"
Initial vocabulary: ['l', 'o', 'w', 'e', 'r', ...]

Iteration 1:
Most frequent pair: ('e', 'r') → merge into 'er'
Result: ['l', 'o', 'w', 'er', ...]

Iteration 2:
Most frequent pair: ('l', 'o') → merge into 'lo'
Result: ['lo', 'w', 'er', ...]

Iteration 3:
Most frequent pair: ('lo', 'w') → merge into 'low'
Result: ['low', 'er', ...]

Continue merging...
Final vocabulary includes: ['low', 'er', 'est', ...]
Final tokenization: "lower" → ['low', 'er']
```

### Properties of BPE

**Advantages**:
1. **Handles unknown words**: Can tokenize anything by falling back to characters/bytes
2. **Efficient**: Good compression of common text patterns
3. **Language-agnostic**: Works for any language (no hand-crafted rules)
4. **Data-driven**: Learns vocabulary from actual corpus

**Disadvantages**:
1. **Context-independent**: "new" in "news" vs "new car" tokenized identically
2. **Greedy**: Locally optimal merges may not be globally optimal
3. **Corpus-dependent**: Different training data → different vocabularies

---

### BPE vs WordPiece vs Unigram

While BPE dominates modern LLMs (2023-2025), three main subword tokenization algorithms exist:

#### Byte-Pair Encoding (BPE)

**Algorithm**: Frequency-based iterative merging of character/byte pairs

**Used by**: GPT series, tiktoken, SentencePiece (BPE mode), Llama 3+, most modern models

**Why it won**:
- Truly language-agnostic (works for all scripts including CJK)
- No preprocessing required
- Proven at massive scale (GPT-4o with 200K vocab)
- Handles any UTF-8 text without unknown tokens (byte-level variant)

#### WordPiece

**Algorithm**: Greedy likelihood-based tokenization with word-level assumptions

**Used by**: BERT (2018), Original Transformer English-French (2017)

**Why it lost**:
- **Assumes word boundaries** (requires space-separated preprocessing)
- **Fails for CJK languages** (Chinese, Japanese, Korean don't use spaces)
- Cannot handle agglutinative languages well
- Lost competition as multilingual requirements grew (2019-2020)

**Historical note**: WordPiece vs BPE competition in 2017-2020. WordPiece dominated encoders (BERT), BPE dominated decoders (GPT). By 2023, BPE won completely.

#### Unigram Language Model

**Algorithm**: Probabilistic language model approach (trains vocabulary distribution)

**Used by**: SentencePiece (Unigram mode option)

**Why rarely deployed**:
- More complex than BPE
- No clear advantages in practice
- Theoretically available in SentencePiece but virtually no production models use it
- All major SentencePiece deployments (Llama 1/2, Mistral v1) chose BPE mode

**Current Reality (2023-2025)**: ~95% of modern LLMs use BPE variants. WordPiece is historical, Unigram is theoretical.

---

### One Algorithm, Many Tokenizers

**Critical Distinction**: BPE is an **algorithm** (the iterative merging process), but each model organization trains its **own tokenizer** using that algorithm.

#### How This Works

When a team builds a new LLM, they:

1. **Take the BPE algorithm** (the same merging process everyone uses)
2. **Run it on their own training corpus** (different data from other models)
3. **Get a unique vocabulary** with different token boundaries

**Result**: GPT-2's tokenizer splits text differently than RoBERTa's, even though both use BPE, because they learned from different data.

#### Early BPE Era (2017-2020): Same Algorithm, Different Implementations

During the 2017-2020 period, every major model trained its own BPE tokenizer on different data:

| Model | Year | Training Corpus | Vocabulary | Notes |
|-------|------|-----------------|------------|-------|
| **GPT-2** | 2019 | **WebText** (40GB from Reddit links) | 50,257 | Introduced byte-level BPE |
| **RoBERTa** | 2019 | Different corpus than GPT-2 | 50,265 | Also byte-level BPE |
| **BART** | 2019 | Similar to GPT-2 approach | 50,265 | Facebook AI |
| **XLM** | 2019 | Multilingual corpus | Varies | Language-pair specific |
| **GPT-3** | 2020 | Larger than WebText | 50,257 | Reused GPT-2 tokenizer |

**Key Insight**: Each organization ran BPE on **their own data**, producing **incompatible tokenizers**. You couldn't use GPT-2's tokenizer with RoBERTa's weights, even though both are "BPE tokenizers."

#### Why Different Data Matters

The training corpus determines which merges happen first:

**Example - Code-heavy corpus**:
- Frequently sees `def`, `class`, `return`
- BPE learns these as single tokens early
- Better compression for code

**Example - Multilingual corpus**:
- Frequently sees characters from many scripts
- BPE learns multilingual patterns
- Better compression across languages

**Example - GPT-2's WebText** (Reddit-curated internet text):
- English-heavy with informal language
- Learned common internet slang as tokens
- Worse at non-English and formal text

#### The Consolidation Era (2023-2025)

Modern era shows industry consolidation around **two standard implementations**:

1. **tiktoken** (OpenAI): ~70% of new models (GPT-4, Llama 3, Qwen 2.5)
2. **SentencePiece BPE** (Google): ~25% of models (Llama 1/2, Gemma 2, Mistral)

But even within tiktoken, models **still train unique tokenizers**:
- Llama 3: 128,000 tokens (multilingual focus)
- GPT-4o: 200,064 tokens (maximum efficiency)
- Qwen 2.5: 152,064 tokens (multilingual + code)

**Same technology, different vocabularies** - because each team trains on different data.

---

# Part II: Tokenizer Technologies

## 3. The Three Approaches

Modern LLMs use three main tokenizer implementations: **SentencePiece** (language-independent BPE), **tiktoken** (optimized byte-level BPE), and specialized variants. Understanding these technologies is essential for interpreting model architectures.

### 3.1 SentencePiece

**Used by**: Llama 1/2, Mistral/Mixtral, Yi, Gemma 2, early Qwen models

**Key Innovation**: Treats text as a raw stream, with whitespace encoded as a special character (`▁`).

#### How It Works

SentencePiece differs from traditional word-based BPE:
- Works directly on Unicode code points (no pre-tokenization)
- Treats spaces as special character (`▁` = U+2581)
- Truly language-independent from the start
- No assumptions about word boundaries

**Example**:
```python
Text: "Hello world"
SentencePiece: ['▁Hello', '▁world']  # ▁ represents space

Text: "你好世界" (Chinese: "Hello world")
SentencePiece: ['▁你好', '▁世界']  # Works identically
```

#### Features

- **No preprocessing required**: Raw text → tokens directly
- **Truly language-agnostic**: Unicode support, no language-specific rules
- **Reversible encoding**: Can exactly recover original text (lossless)
- **Subword regularization**: Optional stochastic sampling for training robustness

#### Advantages

1. **Language-independent**: No language-specific preprocessing or assumptions about text structure
2. **Handles any text**: Full Unicode support
3. **Reversible**: Can exactly recover original text (lossless encoding)
4. **Training features**: Subword regularization for robustness
5. **Scalable**: Can scale from 32K to 256K vocabulary (proven by Gemma 2)
6. **Mature and stable**: Well-understood, battle-tested in production
7. **Backward compatibility**: Existing models maintain tokenizer for consistency

#### Implementation

```python
import sentencepiece as spm

# Train tokenizer
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    character_coverage=0.9995,  # Ensure rare chars covered
    model_type='bpe'            # or 'unigram'
)

# Load and use
sp = spm.SentencePieceProcessor(model_file='tokenizer.model')
tokens = sp.encode('Hello world', out_type=str)
# ['▁Hello', '▁world']

text = sp.decode(tokens)
# 'Hello world'
```

---

### 3.2 tiktoken

**Used by**: GPT-4, GPT-4o, Llama 3, Qwen 2.5, DeepSeek V2/V3, Phi-3/4

**Key Innovation**: Optimized byte-level BPE implementation designed for efficiency and large vocabularies.

#### How It Works

tiktoken operates at the **UTF-8 byte level** rather than Unicode code points:
- Encodes text as UTF-8 bytes first
- Applies BPE at byte level
- No special handling of spaces (everything is bytes)
- Extremely fast Rust implementation

**Example**:
```python
Text: "Hello world"
UTF-8 bytes: [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]
tiktoken (conceptual): ['Hello', ' world']  # After BPE merges

# In practice, tokens are opaque - might be bytes, subwords, or words
```

#### Features

- **UTF-8 byte-level**: Can encode any valid UTF-8 without unknown tokens
- **Deterministic**: No special cases, consistent encoding
- **Fast**: Optimized Rust core (~5-10x faster than pure Python)
- **Large vocabulary support**: Efficiently handles 100K-200K token vocabularies

#### Advantages

1. **Universal encoding**: Handles any UTF-8 text (including emojis, rare scripts)
2. **No special cases**: Clean, uniform byte-level representation
3. **Optimized for speed**: Critical for high-throughput inference
4. **Better compression**: Typically 20-40% fewer tokens than 32K SentencePiece
5. **Multilingual efficiency**: 3-6x better for non-English (with large vocab)

#### Available Encodings

| Encoding | Vocabulary | Used By |
|----------|-----------|---------|
| **r50k_base** | 50,257 | GPT-3, early GPT-3.5 |
| **p50k_base** | 50,281 | Codex, code-davinci-002 |
| **cl100k_base** | 100,277 | GPT-4, GPT-3.5-turbo |
| **o200k_base** | 200,019 | GPT-4o |

#### Implementation

```python
import tiktoken

# Load encoding
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

# Encode
tokens = enc.encode("Hello world")
# [9906, 1917]  # Actual token IDs (opaque)

# Decode
text = enc.decode(tokens)
# 'Hello world'

# Count tokens (useful for API limits)
num_tokens = len(enc.encode("Your text here"))
```

---

### 3.3 Special Features & Variants

Beyond the two main approaches, several models use specialized tokenizer features or custom implementations.

#### Arcade100k (StableLM)

**Based on**: tiktoken `cl100k_base` (GPT-4's tokenizer)

**Special feature**: Digits split into individual tokens
```python
"12345" → ['1', '2', '3', '4', '5']
```

**Rationale**: Forces model to learn mathematical properties of numbers rather than memorizing them as fixed entities. Improves numerical reasoning.

---

#### Llama 3 Tokenizer

**Based on**: tiktoken byte-level BPE

**Vocabulary**: 128,256 tokens (128K, rounded to power-of-2 + special tokens)

**Special optimizations**:
- Heavily optimized for multilingual text (especially non-Latin scripts)
- Increased code token coverage
- Better handling of structured data (JSON, XML)

**Why Meta switched** (from SentencePiece in Llama 2):
- 3x better compression for non-English
- Llama 3's multilingual focus required larger, more efficient vocabulary
- Inference speed gains from shorter sequences

---

#### Qwen Tokenizers (Evolution)

Qwen models demonstrate progressive tokenizer evolution across generations:

| Version | Tokenizer | Vocabulary | Key Improvements |
|---------|-----------|-----------|------------------|
| **Qwen 1.0** | SentencePiece BPE | 151,646 tokens | Initial multilingual focus |
| **Qwen 1.5** | SentencePiece BPE | 151,646 tokens | Same as 1.0 |
| **Qwen 2.0** | Custom tiktoken BPE | 151,936 tokens | Improved multilingual compression |
| **Qwen 2.5** | Custom tiktoken BPE | 152,064 tokens | Enhanced code optimization |

**Why the progression**:
- Qwen 1.0/1.5: Started with large SentencePiece vocab for Chinese optimization
- Qwen 2.0: Switched to tiktoken-based BPE for better compression (+290 tokens)
- Qwen 2.5: Further refined vocabulary for code (+128 tokens)

**Performance impact**:
- 30% better compression than 32K SentencePiece tokenizers
- Especially efficient for Chinese (3-4x better than English-focused tokenizers)
- Code tokenization improved 20-25% from Qwen 2.0 → 2.5

**Significance**: Demonstrates that vocabulary evolution continues even after reaching 150K+ scale—refinement matters as much as size.

---

#### Gemma 2: Largest SentencePiece

**Vocabulary**: 256,000 tokens (largest known SentencePiece deployment)

**Significance**: Proves SentencePiece can scale to ultra-large vocabularies while maintaining stability. Shows the technology isn't limited to 32-64K range.

---

#### Mistral Tekken

**Used by**: Mistral NeMo (July 2024), Mistral-Small-24B-Instruct-2501 (January 2025)

**Based on**: tiktoken byte-level BPE

**Vocabulary**: 131,072 tokens

**Key improvements**:
- Trained on 100+ languages with extensive multilingual coverage
- 30% more efficient for source code compression
- Outperforms Llama 3 tokenizer for ~85% of languages tested
- Significantly better multilingual coverage than previous Mistral v1/v3 SentencePiece tokenizers (32K-32,768 tokens)

**Why Mistral switched** (from SentencePiece):
- Better compression efficiency across languages and code
- Improved multilingual support critical for global deployment
- Competitive with frontier models (Llama 3, GPT-4)
- tiktoken's proven scalability and performance at 100K+ vocabulary sizes

---

#### StarCoder Tokenizer (Code-Optimized)

**Used by**: IBM Granite 3.0, IBM Granite 3.1

**Based on**: Byte Pair Encoding (BPE)

**Vocabulary**: ~49,000 tokens (49,152 precisely)

**Specialization**: Code-focused tokenization optimized for programming languages

**Key Features**:
- Trained on **116 programming languages** + 12 natural languages
- Optimized token boundaries for code syntax (functions, variables, operators)
- Better handling of camelCase, snake_case, and language-specific patterns
- Efficient tokenization of common code patterns and keywords

**Why code-specific tokenizers matter**:
- General-purpose tokenizers treat code as text, splitting identifiers inefficiently
- Code has distinct patterns: `functionName`, `snake_case`, operators (`&&`, `||`)
- StarCoder learned these patterns from massive code corpus
- Results in 20-30% better compression for code vs general tokenizers

**Performance**: IBM Granite 3.x models using StarCoder tokenizer show strong coding benchmarks (HumanEval, MBPP) partly due to efficient code tokenization.

---

#### Modified GPT-NeoX (Security-Enhanced)

**Used by**: AllenAI OLMo 0424 and earlier versions (pre-OLMo 2)

**Based on**: GPT-NeoX-20B tokenizer

**Vocabulary**: ~50,280 tokens

**Unique Feature**: **PII (Personally Identifiable Information) masking**

**How it works**:
- Tokenizer includes PII detection during preprocessing
- Masks email addresses, phone numbers, SSNs, and other sensitive data
- First tokenizer with built-in privacy preservation at tokenization stage
- Part of OLMo's broader data safety pipeline

**Why it matters**:
- Traditional tokenizers preserve PII in training data verbatim
- PII leakage is a major concern for LLM deployment
- OLMo pioneered security-focused tokenization
- Demonstrates tokenizers can do more than just text→tokens conversion

**Evolution**: OLMo 2 (2024) switched to standard tiktoken cl100k_base, dropping the custom PII-masked tokenizer. The PII protection moved to data preprocessing pipeline instead.

---

#### Cohere Custom BPE (Enterprise)

**Used by**: Cohere Command R, Cohere Command R+

**Based on**: GPT-2 style Byte Pair Encoding (BPE)

**Vocabulary**: 256,000 tokens (tied with Gemma 2 for largest)

**Specialization**: Enterprise and multilingual optimization

**Key Features**:
- Optimized for **23 languages** with focus on business/enterprise text
- Massive vocabulary enables efficient tokenization across diverse content
- Designed for RAG (Retrieval-Augmented Generation) and document processing
- Better handling of business terminology, proper nouns, technical jargon

**Why 256K vocabulary**:
- Enterprise documents contain diverse terminology
- Better compression for multilingual business content
- Reduces context window pressure when processing long documents
- Trade-off: Larger embedding layer accepted for efficiency gains

**Performance**: Command R+ demonstrates strong multilingual capabilities and document understanding, partly enabled by the large, diverse tokenizer vocabulary.

---

#### Phi-4 tiktoken Migration

**Used by**: Phi-4 (14B and Mini variants)

**Migration**: Switched from custom Phi tokenizer to **tiktoken** (OpenAI standard)

**Vocabulary**:
- Phi-4 14B: 100,352 tokens
- Phi-4 Mini: 200,064 tokens (second-largest tiktoken deployment after GPT-4o)

**Why Microsoft switched**:
- **Multilingual requirements**: Phi series expanding beyond English
- **Industry standardization**: tiktoken becoming de facto standard
- **Proven scalability**: OpenAI demonstrated 200K vocab viability
- **Competitive pressure**: Match Llama 3 (128K) and GPT-4o (200K) efficiency

**Significance of 200K vocab**:
- Phi-4 Mini's 200K vocabulary is the second-largest tiktoken deployment
- Only GPT-4o (~200K) matches this scale
- Demonstrates Microsoft betting on ultra-large vocabularies for small models
- Trade-off: Larger embedding overhead (~600M params) for better compression

---

#### Alternative Architecture Tokenizers

Beyond mainstream transformers, alternative architectures use custom tokenizers:

**RWKV (Finch Series)**:
- Architecture: RNN-like (not transformer)
- Tokenizer: Custom implementation (details not publicly documented)
- Vocabulary: Unknown
- Note: Alternative architectures may have different tokenization requirements

**xLSTM**:
- Architecture: Extended LSTM (NeuralCompany)
- Tokenizer: Custom
- Vocabulary: 32,000 tokens
- Note: Uses standard vocab size despite non-transformer architecture

**Why mention these**: While transformer-based models dominate (GPT, Llama, etc.), alternative architectures remind us that tokenization choices depend on model architecture, not just data.

---

# Part III: Evolution & Adoption

## 4. The Tokenization Journey (2017-2025)

The history of tokenization mirrors the broader evolution of language models—from experimental diversity to standardization, driven by multilingual requirements and competitive pressure.

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
- **Trend**: ~80% of new models use 100K+ vocabularies
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

## 5. Tokenizers by Model (2017-2025)

### Early BPE Era (2017-2020)

| Model | Year | Tokenizer Type | Vocabulary Size | Notes |
|-------|------|---------------|----------------|-------|
| **Original Transformer** | 2017 | BPE / WordPiece | 32-37K | Mixed approach |
| **GPT-1** | 2018 | BPE | ~40K | First BPE standard |
| **BERT** | 2018 | **WordPiece (non-BPE)** | 30,522 | Encoder standard, assumes spaces |
| **GPT-2** | 2019 | Byte-level BPE | 50,257 | Innovation |
| **T5** | 2019 | SentencePiece BPE | 32,000 | First major SentencePiece |
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
| **Mistral NeMo** | 2024 | tiktoken (Tekken) | 131,072 | Mistral switches to tiktoken |
| **Mistral-Small-24B-Instruct-2501** | 2025 | tiktoken (Tekken) | 131,072 | Continued Tekken adoption |
| **IBM Granite 3.0** | 2024 | BPE (StarCoder) | 49,152 | Code-optimized (116 langs) |
| **IBM Granite 3.1** | 2024 | BPE (StarCoder) | 49,152 | Same as 3.0 |
| **Cohere Command R** | 2024 | Custom BPE | 256,000 | Enterprise multilingual |
| **Cohere Command R+** | 2024 | Custom BPE | 256,000 | Tied for largest |
| **OLMo 0424** | 2024 | Modified GPT-NeoX | ~50,280 | PII masking feature |
| **OLMo 2** | 2024 | tiktoken (cl100k) | ~100,000 | Switched from GPT-NeoX |
| **Phi-4 (14B)** | 2024 | tiktoken | 100,352 | Switched from custom |
| **Phi-4 Mini** | 2024 | tiktoken | 200,064 | 2nd largest tiktoken |
| **RWKV Finch** | 2024 | Custom | Unknown | Alternative architecture |
| **xLSTM** | 2024 | Custom | 32,000 | LSTM-based architecture |

### Proprietary Models (Details Known)

| Model | Year | Tokenizer Type | Vocabulary Size | Notes |
|-------|------|---------------|----------------|-------|
| **Claude 3** | 2024 | Custom BPE | ~65,000 | 70% overlap with GPT-4 |
| **Gemini** | 2024 | SentencePiece | 256,000 | Same as Gemma 2 |
| **Cohere Command R** | 2024 | Custom BPE | 256,000 | Enterprise multilingual optimization |
| **Cohere Command R+** | 2024 | Custom BPE | 256,000 | Tied for largest tokenizer |
| **xAI Grok-2** | 2024 | tiktoken | ~100,000 (estimated) | OpenAI standard format |

---

## 6. Current Consensus (2024-2025)

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

# Part IV: The Driving Forces

## 7. Why Transitions Happened

### 7.1 Multilingual Efficiency: The Driving Force

#### The Problem with Small Vocabularies

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

#### Why This Matters

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

#### Historical Transitions Driven by Multilingual Needs

**Early BPE → SentencePiece (2019-2020)**:

**Drivers**:
1. **Multilingual necessity**: WordPiece assumes spaces, fails for CJK languages
2. **Preprocessing complexity**: SentencePiece eliminates language-specific rules
3. **Lossless requirement**: Perfect reversibility needed
4. **T5's success**: Google's validation mattered
5. **Simplicity**: One tool for all languages

**Impact**: SentencePiece became standard for open-source multilingual models (2020-2023)

**SentencePiece 32K → tiktoken 100K+ (2023-2024)**:

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

**100K → 200-256K (2024)**:

**Drivers**:
1. **Heavily multilingual models**: BLOOM, Gemma, GPT-4o
2. **Diminishing returns understood**: Beyond 256K not worth it yet
3. **Specialized use cases**: Global products need equal language support
4. **Research shows**: Optimal vocab scales with model size
5. **Competition**: Push for "best multilingual" claims

**Open Question**: Will 256K become standard, or is 100-128K the sweet spot?

---

### 7.2 Vocabulary Size Evolution

#### Historical Trend

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

#### The Economics of Larger Vocabularies

**Costs**:
- **Embedding matrix**: 128K vocab × 4096 dims = ~524M parameters (vs 131M for 32K)
- **Output layer**: Same size increase (another ~524M parameters)
- **Total overhead**: ~400M extra parameters for 128K vs 32K vocab
- **Training**: More tokens to learn, more data needed

**Benefits**:
- **Compression**: 20-40% fewer tokens per text
- **Inference speed**: Shorter sequences process faster
- **Multilingual**: 3-6x better for non-English
- **Quality**: Model capacity not wasted on sub-token composition

**Break-even Point**: For models >7B parameters, the ~400M parameter overhead is <6% of total model size. The efficiency gains (faster inference, better quality, lower API costs) outweigh the embedding cost.

---

### 7.3 Technical Performance Impact

#### Sequence Length Reduction

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

#### Out-of-Vocabulary Handling

**Good tokenizer**: Graceful degradation to subwords/characters
**Poor tokenizer**: Many UNK tokens, information loss

Modern byte-level BPE (tiktoken, GPT-2) eliminates UNK tokens entirely by falling back to raw bytes.

---

# Part V: Design & Practice

## 8. Tokenizer Design Trade-offs

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

## 9. Practical Guidance

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

## 10. Future Directions

### Non-BPE Approaches: Why BPE Remains Dominant

Despite research into alternatives, **BPE variants (tiktoken, SentencePiece BPE) dominate 95%+ of modern LLMs**. Here's why:

**Byte-level / Character-level models** (ByT5):
- ✅ No tokenizer needed, simpler pipeline
- ❌ Extremely long sequences (10x longer than BPE)
- ❌ Prohibitive inference cost at scale
- **Status**: Research curiosity, not production-viable

**WordPiece** (BERT era):
- ❌ Assumes spaces, fails for CJK languages
- ❌ Lost the 2017-2020 competition to BPE
- **Status**: Historical only, not used in modern models

**Unigram Language Model**:
- ✅ Theoretically interesting (probabilistic approach)
- ❌ No proven advantages over BPE in practice
- ❌ More complex to implement and tune
- **Status**: Available in SentencePiece but virtually unused

**Why BPE won and will continue to dominate**:
1. Language-agnostic (works for all scripts including CJK)
2. Proven at massive scale (GPT-4o with 200K vocab)
3. Network effects: Industry standardized on tiktoken/SentencePiece BPE
4. No compelling alternative has emerged despite years of research

### Research Areas

1. **Byte-level models**: Skip tokenization entirely (ByT5, etc.) - **Research only, not production**
2. **Learned tokenization**: Train tokenizer end-to-end with model
3. **Context-sensitive**: Different tokenization based on context
4. **Multimodal tokenizers**: Unified for text, image, audio
5. **Specialized tokenizers**: Code (StarCoder), security (PII masking), domain-specific

### Production Trends (2024-2025)

1. **Larger vocabularies**: 100-200K is new standard, 256K for heavy multilingual
2. **tiktoken dominance**: ~70% of new models, driven by "Llama 3 effect"
3. **Vocabulary refinement**: Qwen shows evolution continues even at 150K+ scale
4. **Specialization**: Code-focused (StarCoder), enterprise (Cohere 256K)
5. **Standardization**: Industry converging on tiktoken BPE as de facto standard

### Open Questions

1. Optimal vocabulary size for given model size? (Research suggests: scales with model size and training compute)
2. Can we do better than BPE? (No compelling alternative yet, despite research)
3. How to handle code vs natural language optimally? (StarCoder shows specialized tokenizers work)
4. Should digits/numbers be special-cased? (Arcade100k experiments with this)
5. Will 256K+ vocabularies become standard? (Trade-off: embedding cost vs compression gains)

---

## 11. Sources

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
- [Mistral NeMo Announcement (Tekken)](https://mistral.ai/news/mistral-nemo)
- [Mistral AI & NVIDIA Release Mistral NeMo with Tekken Tokenizer](https://www.marktechpost.com/2024/07/18/mistral-ai-and-nvidia-collaborate-to-release-mistral-nemo-a-12b-open-llm-featuring-128k-context-window-multilingual-capabilities-and-tekken-tokenizer/)
- [Mistral-Small-24B-Instruct-2501 - HuggingFace](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)
- [Qwen Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/qwen2)
- [Gemma 2 Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/gemma2)
- [Gemma Explained - Google Developers](https://developers.googleblog.com/en/gemma-explained-overview-gemma-model-family-architectures/)
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [Phi-3 Documentation - Hugging Face](https://huggingface.co/docs/transformers/model_doc/phi3)
- [Phi-4 Model Page - Hugging Face](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [IBM Granite 3.0/3.1 - HuggingFace](https://huggingface.co/collections/ibm-granite/granite-30-language-models-6751dbbf2f3389bec5c6f02e)
- [AllenAI OLMo - HuggingFace](https://huggingface.co/collections/allenai/olmo-suite-65aeaae8fe5b6b2122b46778)
- [Cohere Command R - HuggingFace](https://huggingface.co/CohereForAI/c4ai-command-r-v01)
- [Cohere Command R+ - HuggingFace](https://huggingface.co/CohereForAI/c4ai-command-r-plus)
- [xAI Grok-2 - HuggingFace](https://huggingface.co/xai-org/grok-2)
- [RWKV Documentation - HuggingFace](https://huggingface.co/docs/transformers/model_doc/rwkv)
- [xLSTM Paper - arXiv](https://arxiv.org/abs/2405.04517)
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
