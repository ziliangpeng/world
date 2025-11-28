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

| Source         | Proportion | Notes                    |
|----------------|------------|--------------------------|
| Common Crawl   | 67%        | CCNet pipeline           |
| C4             | 15%        | Additional filtered web  |
| Wikipedia      | 4.5%       | All languages            |
| GitHub         | 4.5%       | Code                     |
| Books          | 4.5%       | Books3, Gutenberg        |
| ArXiv          | 2.5%       | Scientific papers        |
| StackExchange  | 2%         | QA pairs                 |

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

### Foundation Model Data Mixtures

| Model | Tokens | Web | Code | Math | Multilingual | Notes |
|-------|--------|-----|------|------|--------------|-------|
| **GPT-3** (2020) | 300B | ~60% | ~8% | Minimal | Minimal | Baseline |
| **LLaMA 1** (2023) | 1.4T | ~82% | 4.5% | 2.5% | Minimal | CC + C4 |
| **LLaMA 2** (2023) | 2T | ~82% | 4.5% | 2.5% | ~2% | Extended |
| **LLaMA 3** (2024) | 15T | ~50% | ~17% | ~25% | ~8% (30+ langs) | 4x code, 10x math |
| **Qwen2.5** (2024) | 18T | ~60% | ~25% | ~10% | 29+ langs | High code emphasis |
| **Phi-4** (2024) | 10T | 30% | 20% | Included | 40 langs | 40% synthetic |
| **DeepSeek-V3** (2024) | 14.8T | Not disclosed | Not disclosed | Not disclosed | 119+ langs | Sources known |
| **Qwen3** (2025) | 36T | Multi-stage | Multi-stage | Multi-stage | 119 langs | 3-stage training |
| **Mistral/Mixtral** | ~8T | Not disclosed | Not disclosed | Not disclosed | Not disclosed | Proprietary |
| **Gemma 2/3** | Not disclosed | Not disclosed | Not disclosed | Not disclosed | 140+ langs | Proprietary |

**Key patterns**: Modern models (2024+) use ~50-60% web, 17-25% code, 10-25% math vs early models' 82% web, 4.5% code. Synthetic data emerging (Phi-4: 40%).

### Domain Specialist Models

| Model | Base | Additional Training | Performance |
|-------|------|---------------------|-------------|
| **Qwen2.5-Coder** | Qwen2.5 (18T) | +5.5T code tokens | HumanEval: 92.1 (32B) |
| **Qwen2.5-Math** | Qwen2.5 (18T) | Self-improvement RL | MATH: 83.1 vs base 50 |
| **Code Llama** | LLaMA 2 (2T) | +500B code tokens | HumanEval: 53.7 (34B) |
| **DeepSeek-Coder-V2** | DeepSeek-V2 | +6T code tokens | HumanEval: 90.2 (236B) |
| **DeepSeek-Math** | DeepSeek-V2 | Math-focused training | MATH: 78.5 (7B) |

---

## Multi-Stage Pre-training

Modern LLMs don't train in a single phase—they use 3-4 distinct stages with different data mixes, learning rates, and objectives. This staged approach has become standard for state-of-the-art models.

### Why Multi-Stage?

Single-phase training has limitations:
- **Fixed context length** from start wastes compute on short sequences
- **Static data mix** can't adapt to model's evolving needs
- **No curriculum** means no progression from easy to hard
- **Quality dilution** from mixing high and low quality data equally

Multi-stage training addresses these by adapting the training recipe as the model develops.

### The Four Stages

Based on analysis of Llama 3, Qwen 2, Gemma 2, and Apple AFM technical reports:

```
Stage 1: Core Pre-training
├── Tokens: 7-15T
├── Context: 4K-8K
├── Data: Web-heavy (67%+), code/math baseline
├── LR: Standard warmup + cosine decay
└── Goal: General language understanding

        ↓

Stage 2: Continued Pre-training
├── Tokens: 500B-1T additional
├── Context: Same as Stage 1
├── Data: Upweight math, code, reasoning (2-3x)
├── LR: Reduced (often 10-30% of peak)
└── Goal: Capability enhancement in key domains

        ↓

Stage 3: Context Lengthening
├── Tokens: 100-500B
├── Context: Extend to 32K, 128K, or longer
├── Data: Long documents, synthetic long-context
├── LR: Further reduced
└── Goal: Long-range dependency learning

        ↓

Stage 4: Annealing
├── Tokens: 50-100B
├── Context: Full length
├── Data: Highest quality only (benchmark-like)
├── LR: Linear decay to near-zero
└── Goal: Final quality refinement
```

### Stage-Specific Data Strategies

#### Stage 1: Core Pre-training

Standard mixture similar to original LLaMA:

| Domain | Proportion | Notes |
|--------|------------|-------|
| Web (filtered) | 65-75% | Quality-scored Common Crawl |
| Code | 5-10% | GitHub, StackOverflow |
| Books/Literature | 5-8% | Long-form, narrative |
| Scientific | 3-5% | arXiv, PubMed |
| Wikipedia | 3-5% | Factual grounding |
| Math | 2-3% | Textbooks, problems |

#### Stage 2: Continued Pre-training (Domain Enhancement)

Shift toward capability-building domains:

| Domain | Change from Stage 1 | Rationale |
|--------|---------------------|-----------|
| Math | 2-3x increase | Reasoning foundation |
| Code | 1.5-2x increase | Structured thinking |
| Scientific | 1.5x increase | Technical knowledge |
| Web | Decrease to 40-50% | Make room for quality |
| Synthetic | Add 10-20% | Targeted skill data |

**Llama 3 approach**:
- Upsampled mathematical data
- Added synthetic reasoning examples
- Increased code proportion

**Qwen 2 approach**:
- Heavy math/code emphasis
- Multilingual balancing (Chinese + English)

#### Stage 3: Context Lengthening

Data requirements change significantly:

```python
# Context extension data mix
context_extension_mix = {
    "long_documents": 0.40,    # Books, papers, legal docs (>8K tokens)
    "synthetic_long": 0.30,    # Generated long-context examples
    "short_replay": 0.20,      # Prevent forgetting short-context ability
    "qa_long_context": 0.10,   # Long-context QA pairs
}
```

**Key techniques**:
- **Synthetic long-context**: Generate examples requiring long-range retrieval
- **Position interpolation**: Extend RoPE frequencies gradually
- **Replay buffer**: Include short sequences to prevent degradation

#### Stage 4: Annealing

Final refinement on highest-quality data:

| Data Type | Purpose |
|-----------|---------|
| Curated Q&A | Instruction-following baseline |
| Benchmark-adjacent | (Carefully decontaminated) Similar format/difficulty |
| Expert-written | Human-authored high-quality text |
| Filtered top-1% | Highest quality-score web content |

**Critical**: Avoid actual benchmark contamination while training on similar-quality data.

### Learning Rate Schedules Across Stages

```
           │
    LR     │    ╱╲
           │   ╱  ╲
           │  ╱    ╲________
           │ ╱              ╲____
           │╱                    ╲___
           └─────────────────────────────────
              Stage 1    Stage 2   Stage 3  Stage 4

Stage 1: Warmup → Peak → Cosine decay to ~10% of peak
Stage 2: Resume at ~10-30% of peak → Slow decay
Stage 3: ~5-10% of original peak → Slow decay
Stage 4: Linear decay to near-zero
```

**Llama 3.1 example**:
- Stage 1: Peak LR 1.5e-4, decay to 1.5e-5
- Stage 2: Start at 1.5e-5, decay to 1.5e-6
- Context extension: Start at 1e-5
- Annealing: Linear to 0

### Practical Considerations

**When to transition stages**:
- Loss plateaus suggest current data mix exhausted
- Capability evals show diminishing returns
- Validation perplexity on held-out set

**Data preparation for multi-stage**:
```python
# Prepare stage-specific datasets
datasets = {
    "stage1": {
        "shards": ["web_filtered/", "code/", "books/", ...],
        "weights": {"web": 0.70, "code": 0.08, ...},
    },
    "stage2": {
        "shards": ["math_enhanced/", "code/", "synthetic_reasoning/", ...],
        "weights": {"math": 0.20, "code": 0.15, ...},  # Reweighted
    },
    "stage3": {
        "shards": ["long_documents/", "synthetic_long/", "replay/", ...],
        "sequence_length": 32768,  # Extended
    },
    "stage4": {
        "shards": ["curated_high_quality/"],
        "epochs": 1,  # No repetition
    },
}
```

**Checkpoint strategy**:
- Save at end of each stage
- Enable restart from any stage if issues found
- Track per-stage metrics separately

### Evidence from Technical Reports

| Model | Stages | Key Innovation |
|-------|--------|----------------|
| Llama 3 | 4 | 15T tokens, extensive annealing |
| Qwen 2 | 3+ | Dynamic mixture optimization |
| Gemma 2 | 3 | Knowledge distillation in later stages |
| Apple AFM | 4 | Rejection sampling for quality |

> **Llama 3 Technical Report**: "We found that annealing on small amounts of high-quality code and math data during pre-training improved performance on key benchmarks."

> **Qwen 2 Technical Report**: "We employ a multi-stage pre-training approach with curriculum learning, progressively increasing data complexity."

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

## Synthetic Contamination in the LLM Era

As LLMs proliferate, the web is increasingly filled with AI-generated content. This creates a fundamental challenge for future model training: **model collapse** from recursive training on synthetic data.

### The Model Collapse Problem

**Definition**: Model collapse occurs when generative models are repeatedly trained on data produced by previous generations of models, leading to irreversible performance degradation.

**Key Research**: [Shumailov et al. (Nature, 2024)](https://www.nature.com/articles/s41586-024-07566-y) demonstrated that:
- Indiscriminate use of model-generated content causes defects in resulting models
- **Tails of original distributions disappear** (rare events vanish first)
- Outputs drift toward bland central tendencies with weird outliers
- Affects LLMs, VAEs, and Gaussian mixture models

**The Mechanism**:
```python
# Model collapse cycle
Generation 1: Model trained on human data → outputs H1
Generation 2: Model trained on H1 + synthetic S1 → outputs H2 (degraded)
Generation 3: Model trained on H2 + synthetic S2 → outputs H3 (further degraded)
...
Generation N: Model produces nonsensical outputs, diversity lost
```

**Earlier work**: ["The Curse of Recursion: Training on Generated Data Makes Models Forget"](https://arxiv.org/abs/2305.17493) (arXiv 2023) showed models progressively "forget" when trained recursively on their own outputs.

### Web Contamination Statistics

The problem is accelerating rapidly:

| Timeframe | Metric | Value | Source |
|-----------|--------|-------|--------|
| **April 2025** | New webpages with AI text | **74.2%** | 900K pages surveyed |
| **May 2024 → July 2025** | AI-written pages in Google top-20 | 11.11% → 19.56% | +0.6 pp/month |
| **August 2025** | AI-generated sources in Google AI Overviews | **10.4%** | Citations analysis |

**Implication**: The "uncontaminated pre-2022 data" held by established players (OpenAI, Google, Meta, Anthropic) becomes increasingly valuable. Future entrants face a fundamentally harder training data problem.

### Detection and Mitigation Strategies

#### 1. Watermarking

**[SynthID-Text](https://ai.google.dev/responsible/docs/safeguards/synthid)** (Google, 2024):
- Production-ready text watermarking scheme
- Preserves text quality while enabling high detection accuracy
- **Detection without model access**: Algorithmically detectable without loading LLM
- Probabilistic detection: outputs "watermarked", "not watermarked", or "uncertain"

**[Nature publication](https://www.nature.com/articles/s41586-024-08025-4)** (2024): "Scalable watermarking for identifying large language model outputs"

**Challenges**:
- Not yet universally adopted
- Some watermarks removable via paraphrasing
- Balancing detectability vs. text quality

#### 2. Temporal Cutoffs

**Strategy**: Use only pre-LLM-era data (pre-2022) for critical training phases.

**Examples**:
- Common Crawl snapshots from 2019-2021
- GitHub commits before GPT-3 release
- arXiv papers submitted before 2022

**Limitation**: Misses recent developments and creates knowledge staleness.

#### 3. Source Filtering

**High-trust sources** (lower synthetic probability):
- Academic publishers with peer review
- Licensed datasets (books, news archives)
- Verified human-authored content (pre-2022 timestamps)

**Low-trust sources** (higher synthetic probability):
- Generic web scrapes (2023+)
- Content farms and SEO sites
- User-generated content platforms

#### 4. Synthetic Detection Models

**Approaches**:
- N-gram frequency analysis (synthetic text has different patterns)
- Perplexity scoring (LLM-generated text often has suspiciously low perplexity)
- Cross-table transfer detection (generalize across different datasets)

**Limitation**: [Detection research](https://link.springer.com/chapter/10.1007/978-3-031-91398-3_7) shows cross-table transfer (deployment on unseen tables) remains challenging.

### Benchmark Contamination

A related but distinct problem: evaluation data leaking into training sets.

#### Recent Studies

**[LessLeak-Bench](https://arxiv.org/html/2502.06215v1)** (February 2025):
- Analyzed 83 software engineering benchmarks
- Average leakage: **4.8% (Python), 2.8% (Java), 0.7% (C/C++)**
- **Impact**: StarCoder-7B achieved **4.9x higher** Pass@1 on leaked vs. non-leaked samples

**["Leak, Cheat, Repeat"](https://aclanthology.org/2024.eacl-long.5/)** (EACL 2024):
- GPT-3.5 and GPT-4 exposed to **~4.7M samples from 263 benchmarks** in first year
- Analysis of 255 papers documenting widespread contamination

**[Benchmark Data Contamination Survey](https://arxiv.org/html/2406.04244v1)** (2024):
- Comprehensive survey of contamination detection methods
- Defines data leakage as "unintentional inclusion of evaluation data during model construction"

#### Mitigation Approaches

**1. Dynamic Benchmarks**:
- **[AntiLeakBench](https://arxiv.org/abs/2412.13670)** (December 2024): Automatically constructs benchmarks with explicitly new knowledge
- Fully automated workflow to update benchmarks without human labor

**2. Detection Pipelines**:
- Perplexity-based detection (leaked data has lower perplexity)
- N-gram overlap analysis
- "Benchmark Transparency Card" to document benchmark usage

**3. Temporal Isolation**:
- Create benchmarks from post-training-cutoff data
- Version-controlled benchmark releases

### Industry Approaches

Different organizations handle synthetic contamination differently:

| Organization | Strategy | Transparency |
|-------------|----------|--------------|
| **Meta (Llama)** | Pre-2023 data focus, heavy filtering | High (publishes data mixture) |
| **Microsoft (Phi)** | **40% synthetic** but controlled generation | High (discloses synthetic sources) |
| **Alibaba (Qwen)** | Multi-stage with temporal cutoffs | Medium (stages known, details partial) |
| **Google (Gemma)** | Quality scoring, undisclosed filtering | Low (methods proprietary) |
| **DeepSeek** | Sources known, filtering methods secret | Medium (sources disclosed) |

**Microsoft's Phi-4 approach** (unique):
- Embraces synthetic data but **controls the generation pipeline**
- 40% synthetic (~400B tokens across 50+ dataset types)
- Avoids collapse by using fresh base models for generation, not prior Phi outputs

### Recent Research and Surveys

#### Major 2024-2025 Papers

**Data Contamination**:
- [**"A Survey on Data Contamination for Large Language Models"**](https://arxiv.org/html/2502.14425v2) (February 2025) - Comprehensive survey
- [**"Benchmark Data Contamination of Large Language Models: A Survey"**](https://arxiv.org/html/2406.04244v1) (June 2024) - Evaluation contamination focus

**Training Data Preparation**:
- [**"A Survey on Efficient Large Language Model Training: From Data-centric Perspectives"**](https://arxiv.org/html/2510.25817v1) (October 2024):
  - Data selection methodologies
  - Quality enhancement techniques
  - Synthetic data generation
  - Self-evolving data ecosystems

- [**"Large Language Models: A Survey"**](https://arxiv.org/abs/2402.06196) (Updated March 2025):
  - Popular datasets for LLM training/fine-tuning/evaluation

**Specific Datasets**:
- [**"Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training"**](https://arxiv.org/html/2506.01732v1) (2025):
  - C4C, Open License Corpus (228B tokens)
  - KL3M (1.2T tokens)

**Curated Lists**:
- [**"LLM Research Papers: The 2025 List"**](https://magazine.sebastianraschka.com/p/llm-research-papers-2025-list-one) - 200+ papers, topic-organized
- [**"Selecting and Preparing Training Data for LLMs (2024–2025)"**](https://www.rohan-paul.com/p/selecting-and-preparing-training) - Practitioner guide

### Key Takeaways

1. **Model collapse is real**: Recursive training on synthetic data degrades models irreversibly
2. **Web is 74%+ synthetic**: As of April 2025, most new web content contains AI-generated text
3. **Pre-2022 data is precious**: Uncontaminated human-generated data becomes increasingly scarce
4. **Controlled synthetic works**: Phi-4's 40% synthetic approach avoids collapse via controlled generation
5. **Benchmark leakage is pervasive**: Evaluation contamination inflates reported performance
6. **Detection is hard**: Cross-dataset generalization of synthetic detection remains challenging

### Recommendations for Practitioners

**For training data preparation**:
1. **Temporal filtering**: Prefer pre-2022 data for foundation training
2. **Source verification**: Prioritize high-trust sources (academic, licensed, verified)
3. **Watermark detection**: Run available detectors (SynthID, etc.) on crawled data
4. **Deduplication**: Aggressively deduplicate to avoid synthetic content amplification

**For synthetic data use**:
1. **Control generation**: Use fresh, diverse base models (not your own prior outputs)
2. **Mix with real data**: Never train purely on synthetic (Phi-4: 40% synthetic max)
3. **Quality verification**: Verify synthetic data meets quality thresholds
4. **Document thoroughly**: Disclose synthetic proportions for transparency

**For evaluation**:
1. **Temporal isolation**: Use post-cutoff data for benchmarks
2. **Check leakage**: Run contamination detection before reporting scores
3. **Dynamic benchmarks**: Prefer auto-updating benchmarks when available
4. **Report honestly**: Disclose any known contamination

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
