# Data Preparation for LLM Pre-training

Data is the most important factor in LLM quality—more than architecture, scale, or training methodology. As the field has matured, data preparation has evolved from "scrape the web" to sophisticated pipelines involving deduplication, quality filtering, content classification, and carefully tuned mixing ratios. This document traces that evolution and explains the techniques that define modern data preparation.

---

## The Data Quality Thesis

### Why Data Matters More Than Compute

The [Chinchilla paper](https://arxiv.org/abs/2203.15556) (2022) established that models are typically undertrained on data. But subsequent research revealed something deeper: **data quality dominates quantity**.

| Finding | Source | Implication |
|---------|--------|-------------|
| 1T high-quality tokens > 2T low-quality | [Phi-1](https://arxiv.org/abs/2306.11644) | Quality filtering crucial |
| Data mixtures affect downstream capabilities | [DoReMi](https://arxiv.org/abs/2305.10429) | Domain weighting matters |
| Deduplication improves generalization | [Deduplicating Training Data](https://arxiv.org/abs/2107.06499) | Repetition hurts |
| Synthetic data can match web data | [Phi-3](https://arxiv.org/abs/2404.14219) | Data generation viable |

**Modern consensus**: A well-curated 1T token dataset outperforms a poorly-filtered 10T dataset. Data preparation isn't preprocessing—it's a core part of model development.

---

## Historical Evolution

### Phase 1: Web Scraping Era (2018-2020)

**GPT-2** (2019) introduced WebText—40GB of text from Reddit links with >3 karma. This was revolutionary: curating via social signals rather than manual selection.

```
WebText Pipeline (GPT-2):
Reddit links → Filter: >3 karma → Dragnet/Newspaper extraction → Deduplicate → 40GB text
```

**Limitations**:
- Small scale (40GB)
- Single source (Reddit)
- Limited quality filtering

**Common Crawl** became the default large-scale source. ~300TB compressed per month, but extremely noisy—needed heavy filtering.

### Phase 2: Standardized Datasets (2020-2022)

**[The Pile](https://arxiv.org/abs/2101.00027)** (January 2021) - EleutherAI

First attempt at principled dataset curation:
- 825GB from 22 diverse sources
- Academic papers (arXiv, PubMed), code (GitHub), books, Wikipedia
- Per-source sampling weights

**Key innovation**: Explicit diversity over pure scale.

| Source | Size | Rationale |
|--------|------|-----------|
| Pile-CC | 227GB | Filtered Common Crawl |
| PubMed Central | 90GB | Scientific text |
| Books3 | 101GB | Long-form literature |
| GitHub | 95GB | Code quality |
| Wikipedia | 17GB | Factual grounding |

**[C4](https://arxiv.org/abs/1910.10683)** (T5 Paper, 2019)

"Colossal Clean Crawled Corpus"—filtered Common Crawl:
- English language detection
- Sentence completeness heuristics
- "Bad word" filtering
- Deduplication by 3-sentence spans

C4 became the default baseline, though later analysis revealed significant issues (profanity filters removing legitimate content, geographic bias).

**[ROOTS](https://arxiv.org/abs/2303.03915)** (BLOOM, 2022)

1.6TB across 46 languages with:
- Per-language quality filtering
- PII removal
- Governance framework

First major attempt at multilingual curation and ethical data documentation.

### Phase 3: Quality-First Pipelines (2022-2023)

**[RefinedWeb](https://arxiv.org/abs/2306.01116)** (Falcon, 2023)

Demonstrated that properly-filtered Common Crawl alone could match curated datasets:

```
RefinedWeb Pipeline:
Common Crawl → URL filtering → Text extraction → Language ID →
Quality filtering → Deduplication → 600B high-quality tokens
```

Key insight: Filtering quality > source diversity for web data.

**LLaMA Data** (2023)

Meta's 1.4T token dataset mixed:
| Source | Proportion | Notes |
|--------|------------|-------|
| Common Crawl | 67% | CCNet pipeline |
| C4 | 15% | Additional filtered web |
| Wikipedia | 4.5% | All languages |
| GitHub | 4.5% | Code |
| Books | 4.5% | Books3, Gutenberg |
| ArXiv | 2.5% | Scientific papers |
| StackExchange | 2% | QA pairs |

This mixture became influential—most subsequent models use similar proportions.

### Phase 4: Synthetic and Curated (2024-Present)

**[Phi Series](https://arxiv.org/abs/2306.11644)** - Microsoft

Proved that synthetic data could outperform web scraping:
- Phi-1: 7B code tokens from "textbook-quality" synthetic data
- Phi-2: 250B tokens mixing synthetic + filtered web
- Phi-3: Heavily synthetic curriculum

**[FineWeb](https://arxiv.org/abs/2406.17557)** (HuggingFace, 2024)

15T tokens from Common Crawl with:
- State-of-the-art quality filtering
- Line-level deduplication
- Educational content boosting (FineWeb-Edu)

Became new standard for open pre-training data.

**[DCLM](https://arxiv.org/abs/2406.11794)** (DataComp-LM, 2024)

Systematic benchmark for data curation:
- Standardized filtering evaluations
- Reproducible baselines
- Community competition for best filtering

---

## Core Techniques

### 1. Text Extraction

Raw HTML → clean text is non-trivial. Key tools:

| Tool | Approach | Trade-off |
|------|----------|-----------|
| **trafilatura** | Heuristic extraction | Fast, sometimes misses content |
| **resiliparse** | C++ extraction | Very fast, used in RefinedWeb |
| **jusText** | Paragraph classification | Accurate, slower |
| **boilerpy3** | Boilerplate removal | Good for articles |

**Common issues**:
- Navigation/boilerplate text
- Cookie banners and popups
- Comment sections
- Multi-page articles

Modern pipelines use ensemble approaches—extract with multiple tools, compare, take consensus.

### 2. Language Identification

Critical for multilingual models and English-only filtering:

| Tool | Languages | Notes |
|------|-----------|-------|
| **fastText lid** | 176 languages | Most common, fast |
| **CLD3** | 107 languages | Chrome's detector |
| **langdetect** | 55 languages | Python wrapper |
| **OpenLID** | 200+ | Recent, most accurate |

**Challenge**: Short texts, code-mixed content, romanized languages.

**Best practice**: Threshold at confidence >0.65, discard ambiguous.

### 3. Quality Filtering

#### Heuristic Filters

Rule-based removal of low-quality content:

```python
# Common heuristics (from C4, RefinedWeb)
def heuristic_filter(doc):
    # Length
    if len(doc.words) < 50 or len(doc.words) > 100000:
        return False

    # Sentence completeness
    if not doc.text.endswith(('.', '?', '!', '"')):
        return False

    # Word length
    avg_word_len = sum(len(w) for w in doc.words) / len(doc.words)
    if avg_word_len < 3 or avg_word_len > 10:
        return False

    # Symbol ratio
    if doc.symbol_ratio > 0.1:  # Too many symbols
        return False

    # Repetition
    if doc.line_repetition_ratio > 0.3:
        return False
    if doc.paragraph_repetition_ratio > 0.3:
        return False

    # "Bad" patterns
    if doc.contains_lorem_ipsum:
        return False
    if doc.javascript_ratio > 0.05:
        return False

    return True
```

#### Model-Based Quality Scoring

Train a classifier to predict "quality":

**Perplexity Filtering** (GPT-3, CCNet):
- Train small LM on high-quality data (Wikipedia)
- Score documents by perplexity
- Keep low-perplexity (fluent) documents

```python
# CCNet-style perplexity filtering
def perplexity_filter(doc, threshold=230):
    ppl = wiki_lm.perplexity(doc)
    return ppl < threshold
```

**Classifier Filtering** (Phi, Llama 3):
- Train binary classifier: high-quality vs low-quality
- Use Wikipedia/textbooks as positive, random web as negative
- Score and threshold

**LLM-as-Judge** (FineWeb-Edu, Phi-3):
- Use LLM to rate educational value
- Train small classifier on LLM judgments
- Scale to full corpus

```
# FineWeb-Edu approach
1. Sample 500K documents from FineWeb
2. Prompt Llama-3-70B-Instruct to rate educational value (0-5)
3. Train classifier on (document, score) pairs
4. Apply classifier to full 15T tokens
5. Filter to educational score > 3
```

### 4. Deduplication

Repetition in training data hurts generalization and memorization. Multiple approaches:

#### Exact Deduplication

Remove identical documents:

```python
# Hash-based exact dedup
seen = set()
for doc in corpus:
    hash = md5(doc.text.encode())
    if hash in seen:
        continue
    seen.add(hash)
    yield doc
```

**Problem**: Misses near-duplicates (same content, different formatting).

#### Near-Duplicate Detection

**MinHash + LSH** (Most common):
- Convert documents to n-gram sets
- Compute MinHash signatures (compact representation)
- Use Locality-Sensitive Hashing to find similar pairs efficiently
- O(n) instead of O(n²) comparison

```python
# MinHash signature
def minhash_signature(doc, num_hashes=128):
    ngrams = set(ngram(doc.text, n=5))  # 5-grams
    signature = []
    for i in range(num_hashes):
        min_hash = min(hash_func(i, ng) for ng in ngrams)
        signature.append(min_hash)
    return signature

# LSH for finding candidates
def lsh_dedup(signatures, bands=20, rows=6):
    # 128 hashes / 20 bands = ~6 rows per band
    # Probability of collision for similarity s: 1 - (1 - s^6)^20
    buckets = defaultdict(set)
    for doc_id, sig in signatures:
        for band_idx in range(bands):
            band = tuple(sig[band_idx*rows:(band_idx+1)*rows])
            buckets[band].add(doc_id)
    # Return candidate pairs from same bucket
```

**Suffix Array Deduplication** (The Pile):
- Build suffix array of concatenated corpus
- Find repeated substrings above threshold length
- Remove documents containing long exact repeats

**SimHash** (Google):
- Represent documents as vectors
- Hash to compact fingerprint
- Compare fingerprints via Hamming distance

#### Deduplication Levels

| Level | What's Removed | Trade-off |
|-------|----------------|-----------|
| Document | Identical/near-identical docs | Preserves some redundancy |
| Paragraph | Repeated paragraphs across docs | More aggressive |
| Line | Boilerplate lines (headers, footers) | Very aggressive |
| Substring | N-gram level | Risk of removing legitimate repetition |

**RefinedWeb finding**: Document + URL deduplication sufficient; finer-grained can harm quality.

### 5. Content Classification

Categorize documents for mixture control:

**Domain Classification**:
- Train classifier on manually-labeled samples
- Categories: news, blogs, forums, e-commerce, academic, etc.
- Use for domain-weighted sampling

**Topic Classification**:
- Identify subject matter (science, politics, sports, etc.)
- Balance topic representation in final dataset

**Toxicity/Safety Filtering**:
- Perspective API for toxicity scoring
- Custom classifiers for specific concerns
- Threshold-based removal

```python
# Toxicity filtering example
def safety_filter(doc, threshold=0.8):
    toxicity = perspective_api.score(doc.text)
    if toxicity > threshold:
        return False

    # PII detection
    if contains_pii(doc.text):
        return False

    # Adult content
    if nsfw_classifier.score(doc.text) > 0.9:
        return False

    return True
```

---

## Data Mixing

### The Mixture Problem

Given filtered data from multiple domains, how to combine?

**Simple approach**: Proportional to source size (natural distribution)

**Problem**: Web dominates (>90%), underrepresenting important domains (code, science, books).

### Domain Weighting Strategies

#### Manual Curation (LLaMA approach)

Expert-defined weights based on downstream goals:

| Domain | Natural % | LLaMA % | Upsampling |
|--------|-----------|---------|------------|
| Common Crawl | 90%+ | 67% | 0.7x |
| Code | <1% | 4.5% | ~5x |
| Books | <1% | 4.5% | ~5x |
| Wikipedia | <1% | 4.5% | ~5x |
| Scientific | <1% | 2.5% | ~3x |

Intuition: Upsample high-quality, knowledge-dense sources.

#### Learned Weights (DoReMi)

**[DoReMi](https://arxiv.org/abs/2305.10429)** - Domain Reweighting with Minimax Optimization:

1. Train small proxy model on uniform mixture
2. Compute per-domain excess loss vs reference
3. Upweight domains with high excess loss
4. Iterate until convergence

Result: Automatically discovers that rare domains (code, math) should be upsampled.

#### Temperature Scaling

Sample from each domain with temperature-adjusted probability:

```python
# Temperature sampling
def sample_domain(domain_sizes, temperature=1.0):
    # p_i = (size_i)^(1/T) / sum((size_j)^(1/T))
    scaled = {d: s ** (1/temperature) for d, s in domain_sizes.items()}
    total = sum(scaled.values())
    probs = {d: s/total for d, s in scaled.items()}
    return random.choices(list(probs.keys()), weights=probs.values())[0]

# T=1.0: proportional to size (web dominates)
# T=0.5: more uniform (upweight rare domains)
# T=0.0: equal weighting (ignore size)
```

**Llama 3** uses T~0.7 for code/math domains.

### Epoch and Repetition

**Key finding**: Repeating high-quality data helps, but with diminishing returns.

| Epochs | Effect | Risk |
|--------|--------|------|
| 1 | Baseline | May underfit |
| 2-4 | Often beneficial | Acceptable |
| 5+ | Diminishing returns | Memorization risk |
| 10+ | Quality degradation | Overfitting |

**Best practice**:
- Core data (books, Wikipedia, code): 2-4 epochs acceptable
- Web data: 1 epoch preferred
- Synthetic data: Varies by quality

---

## Pipeline Architecture

### Modern Data Pipeline

```
                                    ┌─────────────────┐
                                    │  Common Crawl   │
                                    │  (raw WARC)     │
                                    └────────┬────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │    Text Extraction          │
                              │    (trafilatura/resiliparse)│
                              └──────────────┬──────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │    Language Identification  │
                              │    (fastText lid.176)       │
                              └──────────────┬──────────────┘
                                             │
               ┌─────────────────────────────┼─────────────────────────────┐
               │                             │                             │
    ┌──────────▼──────────┐    ┌─────────────▼─────────────┐    ┌─────────▼─────────┐
    │   URL Filtering     │    │   Heuristic Filters       │    │   PII Removal     │
    │   (blocklists)      │    │   (length, quality, etc.) │    │                   │
    └──────────┬──────────┘    └─────────────┬─────────────┘    └─────────┬─────────┘
               │                             │                             │
               └─────────────────────────────┼─────────────────────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │    Deduplication            │
                              │    (MinHash + URL)          │
                              └──────────────┬──────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │    Quality Scoring          │
                              │    (classifier/perplexity)  │
                              └──────────────┬──────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │    Domain Classification    │
                              │    + Mixing                 │
                              └──────────────┬──────────────┘
                                             │
                              ┌──────────────▼──────────────┐
                              │    Tokenization             │
                              │    + Packing                │
                              └──────────────┬──────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │  Final Dataset  │
                                    │  (sharded)      │
                                    └─────────────────┘
```

### Processing at Scale

For trillion-token datasets:

**Parallelization**:
- Shard by URL/domain
- Process shards independently
- Merge for deduplication

**Storage**:
- Parquet format for columnar efficiency
- Compressed JSON for flexibility
- Custom binary for maximum throughput

**Tooling**:
| Tool | Use |
|------|-----|
| datatrove | HuggingFace's processing library |
| dolma | AI2's data toolkit |
| CCNet | Facebook's CC pipeline |
| RedPajama | Together's reproduction |

---

## What Major LLMs Use

| Model | Dataset | Size | Key Characteristics |
|-------|---------|------|---------------------|
| GPT-3 | WebText + Books + Wikipedia | 300B tokens | Reddit-filtered web |
| LLaMA | Custom mix | 1.4T | 67% CC, code/books upsampled |
| LLaMA 2 | Extended | 2T | More code, longer training |
| LLaMA 3 | Curated | 15T+ | Extensive quality filtering |
| Falcon | RefinedWeb | 600B | Web-only, heavy dedup |
| Mistral | Undisclosed | ~8T? | Likely LLaMA-style |
| Phi-3 | Synthetic + web | 4.8T | "Textbook quality" focus |
| Qwen | Custom | ~3T | Chinese + English |
| DeepSeek | Custom | ~2T | Specialized for Chinese |

**Pattern**: All modern LLMs use:
1. Filtered Common Crawl as base
2. Upsampled code, books, scientific text
3. Heavy deduplication
4. Quality scoring (model-based)

---

## Common Pitfalls

### 1. Over-Aggressive Filtering

Removing too much can hurt diversity:
- "Bad word" lists removing medical/legal content
- Perplexity filtering removing creative writing
- Sentence heuristics removing dialogue

**Solution**: Validate filters on downstream tasks.

### 2. Benchmark Contamination

Training data may contain benchmark test sets:
- GSM8K problems on web forums
- MMLU questions in study guides
- HumanEval solutions on GitHub

**Solution**:
- Aggressive deduplication against benchmarks
- n-gram filtering for evaluation data
- [Documented contamination checks](https://arxiv.org/abs/2311.04850)

### 3. Temporal Cutoff Issues

Web data has implicit temporal distribution:
- Information may be outdated
- Events after cutoff unknown
- Temporal reasoning affected

**Solution**: Include time metadata, consider freshness in sampling.

### 4. Geographic and Cultural Bias

Common Crawl overrepresents:
- English (>50% of web)
- US/Western content
- Certain demographics

**Solution**: Intentional multilingual/multicultural data inclusion.

---

## Future Directions

### Near-term (2025)

1. **Synthetic data scaling**: More sophisticated generation pipelines
2. **Continual data curation**: Dynamic datasets that evolve
3. **Specialized mixtures**: Task-specific data optimization
4. **Better contamination detection**: Systematic benchmark isolation

### Research Frontiers

1. **Data attribution**: Understanding which training data caused outputs
2. **Optimal mixture theory**: Principled approach to domain weighting
3. **Data efficiency**: Extracting more value per token
4. **Multimodal data curation**: Images, video, audio alongside text

### Open Questions

1. **Scaling synthetic data**: When does generation quality plateau?
2. **Data diversity vs quality**: How to optimally balance?
3. **Curriculum learning**: Should data order matter?
4. **Cross-lingual transfer**: Optimal multilingual mixtures?

---

## Sources

### Foundational Papers
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) - WebText description
- [The Pile: An 800GB Dataset of Diverse Text](https://arxiv.org/abs/2101.00027) - EleutherAI
- [Exploring the Limits of Transfer Learning (T5/C4)](https://arxiv.org/abs/1910.10683) - Google

### Data Curation Research
- [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) - Google
- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) - TII
- [DoReMi: Optimizing Data Mixtures Speeds Up LM Pretraining](https://arxiv.org/abs/2305.10429) - Google

### Modern Datasets
- [FineWeb: Decanting the Web for the Finest Text Data](https://arxiv.org/abs/2406.17557) - HuggingFace
- [DataComp-LM: In Search of the Next Generation of Training Sets](https://arxiv.org/abs/2406.11794) - DCLM

### Quality Filtering
- [Textbooks Are All You Need (Phi)](https://arxiv.org/abs/2306.11644) - Microsoft
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) - Microsoft
- [Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets](https://arxiv.org/abs/2103.12028)

### Tooling and Implementation
- [datatrove](https://github.com/huggingface/datatrove) - HuggingFace data processing
- [dolma](https://github.com/allenai/dolma) - AI2 data toolkit
- [CCNet](https://github.com/facebookresearch/cc_net) - Facebook pipeline
