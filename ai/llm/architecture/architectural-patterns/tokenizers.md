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

### Vocabulary Sizes in Practice

| Model | Vocabulary | Tokenizer |
|-------|-----------|-----------|
| Llama 1 | 32K | SentencePiece |
| Llama 2 | 32K | SentencePiece |
| Mistral 7B | ~32K | SentencePiece |
| Yi 34B | 64K | SentencePiece BPE |
| Qwen 2 | 64K | SentencePiece |

**Typical Range**: 32K - 64K tokens

### Advantages

1. **Language-independent**: No language-specific preprocessing
2. **Handles any text**: Unicode support
3. **Reversible**: Can exactly recover original text
4. **Training features**: Subword regularization for robustness

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

### Why Larger Vocabularies?

**Advantages**:
1. **Multilingual efficiency**: Better coverage of non-English
2. **Fewer tokens per text**: More efficient encoding
3. **Better compression**: Rare words as single tokens
4. **Technical terms**: Better handling of code, math, etc.

**Disadvantages**:
1. **Larger embedding matrix**: More parameters
2. **Output layer**: Larger final projection
3. **Training cost**: More tokens to learn
4. **Diminishing returns**: At some point, not worth it

### Vocabulary Size by Model

| Model | Vocabulary | Tokenizer Type |
|-------|-----------|---------------|
| GPT-2/3 | 50K | BPE |
| Llama 1/2 | 32K | SentencePiece |
| Mistral | 32K | SentencePiece |
| Yi | 64K | SentencePiece |
| GPT-4 | 100K | tiktoken |
| Llama 3 | 128K | tiktoken |
| Qwen 2.5 | 152K | Custom |
| **Gemma 2** | **256K** | **Custom (largest!)** |

---

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

### Multilingual Performance

**Example**: Chinese text

**English-focused tokenizer** (32K vocab):
```
"你好世界" → 8-12 tokens (character-level fallback)
```

**Multilingual tokenizer** (128K vocab):
```
"你好世界" → 2-4 tokens (proper subwords)
```

**Impact**:
- 3-6x more efficient for non-English
- Better long-context support
- Improved performance on multilingual tasks

### Out-of-Vocabulary Handling

**Good tokenizer**: Graceful degradation to subwords/characters
**Poor tokenizer**: Many UNK tokens, information loss

---

## Tokenizer Design Trade-offs

### Vocabulary Size

**Small (30-50K)**:
- ✅ Smaller embedding matrix
- ✅ Faster output projection
- ❌ More tokens per text
- ❌ Poor multilingual support

**Large (100-256K)**:
- ✅ Fewer tokens per text
- ✅ Better multilingual
- ✅ Handles rare words
- ❌ Larger parameters
- ❌ Slower output projection

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

- [LLM Tokenization](https://hundredblocks.github.io/transcription_demo/)
- [Balancing Vocabulary Size in Modern LLMs](https://www.rohan-paul.com/p/tutorial-balancing-vocabulary-size)
- [SentencePiece](https://github.com/google/sentencepiece)
- [tiktoken](https://github.com/openai/tiktoken)
- BPE paper (1994, compression) and adaptations for NLP
- Various model papers and documentation
