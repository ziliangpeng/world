# Data Mixing for LLM Pre-training

Once you have clean, filtered data from your [data preparation pipeline](data-preparation.md), the next critical challenge is combining these sources optimally. Data mixing—deciding what proportion of code vs web vs books vs math to include, and how those proportions evolve across training stages—profoundly impacts model capabilities and efficiency.

Modern LLMs don't train on a single static mixture. They use sophisticated multi-stage curricula where different data compositions serve different learning objectives. This document explores the strategies, evolution, and best practices for data mixing in frontier LLM development.

---

# Part I: Foundation

## 1. Introduction & The Mixture Problem

### Why Mixing Matters

**The Challenge**: You have multiple clean data sources—web pages, code repositories, books, scientific papers, Q&A forums. How do you combine them?

**Naive approach**: Sample proportionally to size
```
Common Crawl: 90% (because it's huge)
Code: <1% (much smaller)
Books: <1%
Wikipedia: <1%
```

**Problem**: This under-represents high-value, knowledge-dense sources. Models trained this way are web-heavy, with poor code and reasoning capabilities.

**Modern approach**: Strategic upsampling based on value
```
LLaMA 3 example:
Web: 50% (downsampled from 90%+)
Code: 17% (upsampled ~20x)
Math: 25% (upsampled ~30x)
Wikipedia: 8%
```

**Result**: Better code generation, mathematical reasoning, and knowledge retention—without proportionally more total data.

### The Core Tension

**Quality vs Coverage**:
- Upsample too much: Models memorize limited high-quality data
- Upsample too little: Models dominated by noisy web text

**Domain Specialization vs Generalization**:
- Too much code: Poor general language understanding
- Too little code: Weak programming capabilities

**Cost vs Capability**:
- More training epochs on quality data: Higher cost but better performance
- Single epoch on massive noisy data: Cheaper but lower ceiling

The art of data mixing is navigating these trade-offs for your specific model goals and budget.

---

## 2. The Evolution of Data Mixtures (2018-2025)

### Phase 1: Web-Heavy Era (2018-2020)

**GPT-2** (2019) - WebText:
- 40GB curated from Reddit links
- Single source, minimal mixing decisions
- Proportional sampling (whatever Reddit pointed to)

**GPT-3** (2020):
- First large-scale mixture: 300B tokens
- Common Crawl: ~60% (410B tokens, filtered, weighted 0.6)
- WebText2: ~19% (19B tokens, weighted 0.92)
- Books: ~16% (12B + 55B tokens)
- Wikipedia: ~3% (3B tokens, weighted 3.0)
- Code: Minimal presence

**Key insight**: GPT-3 already used non-proportional weighting (Wikipedia weighted 3.0x means sampled 3x more than its size suggests). But still web-dominated.

**Limitations**:
- Code capabilities weak
- Mathematical reasoning limited
- Heavily English-centric

### Phase 2: Manual Curation Era (2020-2023)

**The Pile** (2021) - 825GB, 22 sources:
- Explicit diversity over scale
- Manual per-source weights
- Introduced systematic code inclusion (GitHub: 95GB)

**LLaMA 1** (February 2023) - **Established the "standard mixture"**:

| Source         | Proportion | Tokens | Rationale                    |
|----------------|------------|--------|------------------------------|
| Common Crawl   | 67%        | 938B   | General knowledge            |
| C4             | 15%        | 210B   | Filtered web                 |
| GitHub         | 4.5%       | 63B    | Code capability              |
| Wikipedia      | 4.5%       | 63B    | Factual grounding            |
| Books          | 4.5%       | 63B    | Long-form reasoning          |
| ArXiv          | 2.5%       | 35B    | Scientific knowledge         |
| StackExchange  | 2%         | 28B    | Q&A reasoning                |

**Why this mattered**:
- **Systematic upsampling**: Code/books/Wikipedia are <1% of web naturally, upsampled to ~11% combined
- **5x upsampling**: Code, books, Wikipedia each ~5x overrepresented vs natural distribution
- **Reproducible baseline**: Open-source community standardized on this

**LLaMA 2** (July 2023):
- 2T tokens (40% more data)
- **Same mixture proportions** as LLaMA 1
- Validated that the LLaMA 1 mixture was robust

**Consensus** (2022-2023):
- 67-82% web (filtered)
- 4-5% code
- 4-5% books
- 2-5% scientific
- 2-5% Wikipedia

### Phase 3: Balanced Mixture Era (2023-2024)

**Shift**: Recognition that 67-82% web is still too high for frontier capabilities

**LLaMA 3** (April 2024) - **The Paradigm Shift**:

| Source | LLaMA 2 | LLaMA 3 | Change |
|--------|---------|---------|--------|
| Web    | ~82%    | ~50%    | **-40% relative** |
| Code   | 4.5%    | ~17%    | **+280% (4x increase)** |
| Math   | 2.5%    | ~25%    | **+900% (10x increase)** |
| Multilingual | ~2% | ~8% | **+300%** |

**Why LLaMA 3 changed the game**:
1. **Code explosion**: 4x increase in code proportion
2. **Math emphasis**: 10x increase in mathematical content
3. **Multilingual**: 4x increase for global capability
4. **Web reduction**: Freed capacity by cutting web to 50%

**DoReMi** (May 2023) - **Learned Optimization**:
- Automated mixture discovery through minimax optimization
- Train proxy model on uniform mixture
- Compute per-domain excess loss
- Upweight domains with high excess loss
- Iterate until convergence

**Key finding**: DoReMi automatically discovered that code and math should be heavily upsampled—validating the LLaMA 3 approach

### Phase 4: Synthetic-First Era (2024-Present)

**Phi-1** (June 2023) - **The Textbook Hypothesis**:
- 7B code tokens of "textbook quality" **synthetic** data
- Generated by GPT-3.5 with careful prompting
- Outperformed much larger models trained on web-scraped code
- **Thesis**: Quality >> Quantity for small models

**Phi-4** (December 2024) - **Synthetic at Scale**:

| Source Type | Proportion | Tokens | Notes |
|-------------|------------|--------|-------|
| **Synthetic** | **40%** | **~400B** | **50+ dataset types** |
| Web         | 30%      | ~300B   | Heavily filtered |
| Code        | 20%      | ~200B   | Real + synthetic |
| Academic    | 10%      | ~100B   | Papers, textbooks |

**Why Phi-4 is revolutionary**:
- **Synthetic-first**: 40% synthetic is primary source, not supplement
- **Controlled generation**: Fresh base models generate data, avoiding model collapse
- **Small model efficiency**: 14B model competitive with 70B+ dense models
- **Cost efficiency**: Synthetic data generation cheaper than massive web scraping at this quality level

**Strategy** (critical for avoiding model collapse):
1. Use **fresh, diverse base models** for generation (not prior Phi outputs)
2. Carefully prompt for specific skills (reasoning, coding patterns, math)
3. Quality-filter generated data (reject low-quality outputs)
4. Mix with real data (never train purely on synthetic)

**Microsoft's thesis**: For small models (<30B), synthetic "textbook quality" data yields better per-token learning than large-scale web scraping

### Phase 5: Multi-Stage Sophistication (2024-Present)

**Modern paradigm**: Data mixture evolves across training stages

**Qwen3** (2025) - **3-Stage Curriculum**:

```
Stage 1: General Pre-training (7-10T tokens)
├── Web: 60%
├── Code: 15%
├── Math: 10%
├── Books: 10%
└── Scientific: 5%

        ↓

Stage 2: Capability Enhancement (1-2T tokens)
├── Math: 30% (3x increase)
├── Code: 25% (1.7x increase)
├── Scientific: 20% (4x increase)
├── Synthetic reasoning: 15%
└── Web: 10% (6x decrease)

        ↓

Stage 3: Annealing & Refinement (100-500B tokens)
├── Curated Q&A: 40%
├── Benchmark-adjacent: 30%
├── Expert-written: 20%
└── Top-1% web: 10%
```

**Why multi-stage matters**:
- **Stage 1**: Broad knowledge acquisition (web-heavy acceptable)
- **Stage 2**: Targeted capability building (code/math intensive)
- **Stage 3**: Quality refinement (only highest-quality data)

**LLaMA 3.1** approach:
- Annealing on small amounts of high-quality code/math during pre-training
- Improved benchmark performance without massive compute increase

### Evolution Summary

The field progressed from:
- **2018-2020**: "Scrape the web" (GPT-3: 60% web)
- **2020-2023**: "Manually upsample quality sources" (LLaMA 1/2: 82% web, 4.5% code)
- **2023-2024**: "Balance web with capabilities" (LLaMA 3: 50% web, 17% code, 25% math)
- **2024**: "Synthetic can dominate" (Phi-4: 40% synthetic)
- **2024-2025**: "Evolve mixture across stages" (Qwen3: 3-stage curriculum)

**Key insight**: Mixture sophistication enables efficiency. Phi-4's 14B model with strategic mixture competes with 70B models trained on standard mixtures.

---

# Part II: Mixing Strategies

## 3. Domain Weighting Approaches

### Manual Curation (LLaMA Approach)

**Philosophy**: Expert-defined weights based on downstream goals

**LLaMA 1/2 rationale**:

| Domain | Natural % | LLaMA % | Upsampling | Justification |
|--------|-----------|---------|------------|---------------|
| Common Crawl | 90%+ | 67% | 0.7x | **Downsample**: Reduce noise |
| Code | <1% | 4.5% | ~5x | **Upsample**: Critical for programming |
| Books | <1% | 4.5% | ~5x | **Upsample**: Long-form reasoning |
| Wikipedia | <1% | 4.5% | ~5x | **Upsample**: Factual grounding |
| Scientific | <1% | 2.5% | ~3x | **Upsample**: Technical knowledge |

**Intuition**: Upsample high-quality, knowledge-dense sources. Downsample noisy but abundant web text.

**LLaMA 2 → LLaMA 3 evolution**:

| Capability Goal | LLaMA 2 Mix | LLaMA 3 Mix | Reasoning |
|-----------------|-------------|-------------|-----------|
| **Programming** | 4.5% code | ~17% code | **4x increase**: Frontier code capabilities required |
| **Reasoning** | 2.5% math | ~25% math | **10x increase**: Math critical for reasoning foundation |
| **Multilingual** | ~2% | ~8% | **4x increase**: Global AI requires equal language support |
| **General knowledge** | ~82% web | ~50% web | **Reduced**: Make room for capabilities, larger vocab helps |

**How to determine weights**:
1. **Identify target capabilities**: What should the model excel at?
2. **Evaluate baselines**: Which domains correlate with those capabilities?
3. **Constrain by budget**: More upsampling = more epochs = higher cost
4. **Iterate on small proxy models**: Test mixture, measure downstream metrics
5. **Commit to final mixture**: Train full model

### Synthetic-First Approach (Phi Series)

**Key insight**: For small models, quality >> quantity. Synthetic "textbook quality" data provides better per-token learning.

**Phi-1 Strategy** (7B code model):
- **Data**: 7B tokens, 100% synthetic code
- **Generation**: GPT-3.5 prompted to write "textbook-quality" code examples
- **Filtering**: Reject outputs that don't meet quality bar
- **Result**: Outperformed models 10x larger trained on GitHub scrapes

**Phi-4 Strategy** (14B general model):

| Source | Proportion | Tokens | Generation Strategy |
|--------|------------|--------|---------------------|
| Synthetic | 40% | 400B | **50+ dataset types** |
| Web | 30% | 300B | Heavily filtered (FineWeb-Edu style) |
| Code | 20% | 200B | Mix of real (GitHub) + synthetic |
| Academic | 10% | 100B | ArXiv, textbooks, PubMed |

**Synthetic generation categories**:
1. **Reasoning chains**: Step-by-step problem solving
2. **Code patterns**: Common algorithms with explanations
3. **Q&A pairs**: High-quality questions + comprehensive answers
4. **Domain-specific**: Math problems, science explanations, etc.

**Critical**: Avoiding model collapse
- **Use fresh base models**: Generate from GPT-4, Claude, Gemini—not prior Phi versions
- **Mix with real data**: Never train purely on synthetic (40% max)
- **Quality filter**: Reject low-quality synthetic outputs
- **Diverse prompting**: 50+ prompt templates to avoid repetitive patterns

**When synthetic-first works**:
- **Small models** (<30B): Limited capacity benefits from curated high-quality data
- **Specific capabilities**: Targeted synthetic data (e.g., chain-of-thought reasoning)
- **Cost constraints**: Generating 400B quality tokens cheaper than scraping/filtering 10T tokens
- **Controlled environment**: You have strong base models available for generation

**When to avoid**:
- **Large models** (>100B): Can learn from noisy data, need scale
- **General knowledge**: Synthetic can't replace comprehensive web coverage
- **Unknown domains**: Hard to synthetically generate what you don't understand

**Results**: Phi-4 (14B) performs competitively with 70B dense models on reasoning benchmarks, demonstrating the power of mixture quality for small models.

### Learned Weights (DoReMi)

**DoReMi: Domain Reweighting with Minimax Optimization**

**Problem**: Manual weights require expert intuition and many experiments. Can we learn optimal weights automatically?

**Algorithm**:
1. **Train proxy model** on uniform mixture (equal weight per domain)
2. **Compute excess loss**: For each domain, measure how much worse model performs vs reference distribution
3. **Upweight struggling domains**: Domains with high excess loss get higher sampling weight
4. **Retrain with new weights**: Update mixture, train new proxy
5. **Iterate**: Repeat until convergence

**Mathematical formulation**:
```
Domain weight update:
w_i^(t+1) = w_i^(t) * exp(α * excess_loss_i)

Where:
- w_i: weight for domain i
- excess_loss_i: loss on domain i - loss on reference
- α: learning rate for weight updates
```

**Key findings** (from original paper):
- **Automatically discovered**: Code and math should be heavily upsampled
- **Validated intuition**: DoReMi's learned weights closely match LLaMA 3's manual weights
- **Generalization**: Improves downstream task performance, not just perplexity

**Practical considerations**:
- **Requires proxy training**: Need compute budget for multiple small model runs
- **Domain definition**: Results sensitive to how you split data into domains
- **Reference distribution**: Choice of reference (uniform? proportional?) affects outcome
- **Diminishing returns**: After initial optimization, manual tweaking often needed

**When to use DoReMi**:
- You have compute budget for proxy models
- Many domains (10+) make manual tuning intractable
- Unclear which domains matter for your use case
- Want data-driven justification for mixture choices

### Temperature Scaling

**Concept**: Sample from each domain with temperature-adjusted probability

**Formula**:
```python
# Natural sampling (proportional to size)
p_i = size_i / sum(size_j for all j)

# Temperature-scaled sampling
p_i = (size_i)^(1/T) / sum((size_j)^(1/T) for all j)
```

**Temperature effects**:
- **T = 1.0**: Proportional to size (web dominates)
- **T = 0.5**: More uniform (upweight rare domains)
- **T = 0.3**: Highly uniform (near-equal sampling)
- **T = 0.0**: Fully uniform (ignore size)
- **T = 2.0**: Even more size-dependent (extreme web bias)

**Example** (3 domains):
```
Domain sizes:
- Web: 1000B tokens
- Code: 10B tokens
- Math: 5B tokens

T=1.0 (proportional):
- Web: 98.5%
- Code: 1.0%
- Math: 0.5%

T=0.5:
- Web: 76.5%
- Code: 13.7%
- Math: 9.8%

T=0.0 (uniform):
- Web: 33.3%
- Code: 33.3%
- Math: 33.3%
```

**LLaMA 3 usage**: Reportedly uses T ≈ 0.7 for code and math domains
- Upsamples code/math significantly
- Not as extreme as pure uniform
- Balances representation without ignoring size entirely

**Advantages**:
- Simple conceptual model (one hyperparameter)
- Smooth interpolation between proportional and uniform
- Easy to implement and understand

**Disadvantages**:
- All domains treated identically (no per-domain control)
- Finding optimal T requires experimentation
- May not match optimal mixture (DoReMi-style learned weights more flexible)

**Best practice**: Use temperature scaling for initial exploration, then fine-tune with manual weights for production

---

## 4. Epoch and Repetition Strategies

### Effects by Epoch Count

**The Trade-off**: Repeating data allows more learning, but risks memorization and overfitting

| Epochs | Effect on Model | Risk Level | When to Use |
|--------|-----------------|------------|-------------|
| **1** | Baseline learning | Low (underfit risk) | Massive datasets (10T+ tokens) |
| **2-4** | Often beneficial | Acceptable | Standard training (1-3T tokens per source) |
| **5-9** | Diminishing returns | Moderate (memorization) | High-value small datasets |
| **10+** | Quality degradation | High (overfitting) | Rarely justified |

**Key finding** (LLaMA 3 Technical Report):
> "We found that repeating high-quality data (code, math) for 2-4 epochs improved performance, while web data was best used for a single epoch."

### Per-Domain Strategies

**Domain-specific epoch budgets**:

```
Web data (noisy, abundant):
├── 1 epoch preferred
├── Rationale: Plenty of data, repeating doesn't help
└── Risk: Memorizing web noise

Code (high-quality, moderate size):
├── 2-4 epochs acceptable
├── Rationale: Limited quality code, worth repeating
└── Risk: Memorizing specific implementations

Books (high-quality, limited):
├── 3-5 epochs acceptable
├── Rationale: Valuable long-form reasoning patterns
└── Risk: Memorizing entire books

Wikipedia (high-quality, limited):
├── 2-3 epochs acceptable
├── Rationale: Factual grounding, worth reinforcing
└── Risk: Memorizing facts (actually desired here)

Math (high-quality, can be generated):
├── 2-5 epochs, or supplement with synthetic
├── Rationale: Reasoning patterns benefit from repetition
└── Risk: Limited if synthetic mixed in

Synthetic data:
├── Varies by quality
├── High-quality synthetic: 2-3 epochs safe
└── Low-quality synthetic: Avoid repetition
```

**Example mixture with epochs** (hypothetical 1.4T token training):

| Source | Tokens Available | Epochs | Tokens Seen | Proportion |
|--------|------------------|--------|-------------|------------|
| Web | 1000B | 1 | 1000B | 71% |
| Code | 100B | 3 | 300B | 21% |
| Books | 50B | 1.5 | 75B | 5% |
| Wikipedia | 20B | 1.5 | 30B | 2% |
| Math | 10B | 3 | 30B | 2% |

### Diminishing Returns Analysis

**Research finding**: Marginal benefit of each additional epoch decreases

```
First epoch: 100% learning efficiency
Second epoch: ~70% efficiency
Third epoch: ~40% efficiency
Fourth epoch: ~20% efficiency
Fifth+ epoch: ~10% efficiency, increasing overfitting risk
```

**Implication**: Better to get more diverse data than repeat excessively

**Cost-benefit**:
- **Cheap data** (web scraping): Prefer fresh data over repetition
- **Expensive data** (human-written, curated): Repetition more justifiable
- **Synthetic data**: Can generate fresh data instead of repeating

**Modern best practice** (2024-2025):
1. **Web**: 1 epoch maximum
2. **Code/books/Wikipedia**: 2-3 epochs standard
3. **Math/reasoning**: 2-4 epochs, supplement with synthetic
4. **Annealing stage**: Fresh high-quality data only (no repetition)

---

# Part III: Multi-Stage Pre-training

## 5. Why Multi-Stage Training

### Limitations of Single-Phase Training

**Traditional approach**: Train on fixed mixture from start to finish

**Problems**:
1. **Fixed context length**: Training on max context (128K) from start wastes compute on short sequences
2. **Static data mix**: Can't adapt to model's evolving needs (early training needs breadth, late training needs depth)
3. **No curriculum**: All data treated equally, no progression from easy to hard
4. **Quality dilution**: High and low quality data mixed equally throughout

### The Four-Stage Paradigm

Modern frontier models use 3-4 distinct training stages with different data mixtures, learning rates, and objectives:

```
Stage 1: Core Pre-training
├── Goal: General language understanding
├── Tokens: 7-15T (majority of training)
├── Context: 4K-8K
├── Data: Web-heavy (60-70%), baseline code/math
└── LR: Standard warmup + cosine decay

        ↓

Stage 2: Continued Pre-training (Capability Enhancement)
├── Goal: Targeted skill building
├── Tokens: 500B-1T additional
├── Context: Same as Stage 1
├── Data: Upweight math (2-3x), code (1.5-2x), reasoning
└── LR: Reduced (10-30% of peak)

        ↓

Stage 3: Context Lengthening
├── Goal: Long-range dependency learning
├── Tokens: 100-500B
├── Context: Extend to 32K, 128K, or longer
├── Data: Long documents, synthetic long-context
└── LR: Further reduced

        ↓

Stage 4: Annealing
├── Goal: Final quality refinement
├── Tokens: 50-100B
├── Context: Full length
├── Data: Highest quality only (benchmark-like)
└── LR: Linear decay to near-zero
```

### Evidence from Technical Reports

**LLaMA 3 Technical Report**:
> "We found that annealing on small amounts of high-quality code and math data during pre-training improved performance on key benchmarks."

**Qwen 2 Technical Report**:
> "We employ a multi-stage pre-training approach with curriculum learning, progressively increasing data complexity."

**Gemma 2 & Apple AFM**: Both use variants of multi-stage training with knowledge distillation and quality progression

---

## 6. Stage-Specific Data Strategies

### Stage 1: Core Pre-training

**Goal**: Broad knowledge acquisition across all domains

**Standard mixture** (LLaMA baseline):

| Domain | Proportion | Notes |
|--------|------------|-------|
| Web (filtered) | 65-75% | Quality-scored Common Crawl |
| Code | 5-10% | GitHub, StackOverflow |
| Books/Literature | 5-8% | Long-form, narrative |
| Scientific | 3-5% | arXiv, PubMed |
| Wikipedia | 3-5% | Factual grounding |
| Math | 2-3% | Textbooks, problems |

**Why web-heavy is acceptable here**:
- Model learning basic language patterns
- Need broad coverage of concepts
- Web provides diversity

**Context length**: 4-8K tokens (efficient for most web content)

**Tokens**: 7-15T (bulk of training budget)

### Stage 2: Continued Pre-training (Domain Enhancement)

**Goal**: Build targeted capabilities (coding, reasoning, multilingual)

**Shift toward capability-building domains**:

| Domain | Change from Stage 1 | Rationale |
|--------|---------------------|-----------| | Math | **2-3x increase** (5-9%) | Reasoning foundation |
| Code | **1.5-2x increase** (10-15%) | Structured thinking |
| Scientific | **1.5x increase** (5-8%) | Technical knowledge |
| Synthetic reasoning | **Add 10-20%** | Targeted skill data |
| Web | **Decrease to 40-50%** | Make room for quality |

**LLaMA 3 approach**:
- Upsampled mathematical data
- Added synthetic reasoning examples
- Increased code proportion
- Result: Major improvements on MATH, GSM8K, HumanEval benchmarks

**Qwen 2 approach**:
- Heavy math/code emphasis
- Multilingual balancing (Chinese + English)
- Curriculum: easier problems → harder problems within math/code domains

**Why this stage matters**:
- Model already has general language ability from Stage 1
- Now optimize for high-value capabilities
- Cheaper than training on this mixture from scratch

**Tokens**: 500B-1T additional

### Stage 3: Context Lengthening

**Goal**: Extend context window from 8K → 32K → 128K+ without catastrophic forgetting

**Data requirements change significantly**:

```python
context_extension_mix = {
    "long_documents": 0.40,    # Books, papers, legal docs (>8K tokens)
    "synthetic_long": 0.30,    # Generated long-context examples
    "short_replay": 0.20,      # Prevent forgetting short-context ability
    "qa_long_context": 0.10,   # Long-context QA pairs
}
```

**Long document sources**:
- Full books (novels, textbooks)
- ArXiv papers (full papers, not abstracts)
- Legal documents (contracts, case law)
- GitHub repositories (full repo context)

**Synthetic long-context generation**:
```
Task: Generate examples requiring long-range retrieval
Method: Use GPT-4 to create:
1. Document with information spread across 32K+ tokens
2. Questions requiring synthesis from multiple distant sections
3. Verification that answers require full context
```

**Replay buffer** (critical):
- Include 20% short sequences from Stage 1
- Prevents degradation of short-context performance
- Model learns "context length doesn't matter" rather than "always expect long context"

**Position interpolation**: Extend RoPE frequencies gradually
- Start at 8K → 16K → 32K → 64K → 128K
- Gradual extension more stable than jumping to 128K

**Tokens**: 100-500B

**Why this is a separate stage**:
- Can't train on 128K context from start (4x memory cost)
- Model needs strong short-context foundation first
- Extension via fine-tuning more efficient

### Stage 4: Annealing

**Goal**: Final quality refinement on highest-quality data

**Data composition**:

| Data Type | Proportion | Purpose |
|-----------|------------|---------|
| Curated Q&A | 40% | Instruction-following baseline |
| Benchmark-adjacent | 30% | Similar difficulty/format (decontaminated) |
| Expert-written | 20% | Human-authored high-quality text |
| Top-1% web | 10% | Highest quality-score web content |

**Critical**: Avoid actual benchmark contamination
- Use problems *similar* to GSM8K, not actual GSM8K test problems
- Math problems at benchmark difficulty level
- Code problems in benchmark style
- Decontaminate aggressively (n-gram filtering against test sets)

**Why annealing works**:
- Model already has broad knowledge and capabilities
- Small amounts of very high-quality data refine edges
- Learning rate near-zero means fine-grained adjustments

**Tokens**: 50-100B

**LLaMA 3.1 example**:
- Annealed on high-quality code and math
- Improved MATH benchmark: 50 → 55
- Improved HumanEval: 75 → 80
- Total cost: <5% of total training budget

---

## 7. Learning Rate Schedules Across Stages

### Per-Stage LR Progression

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

### LLaMA 3.1 Example

**Stage 1**: Core pre-training
- Warmup: 2000 steps, linear 0 → peak
- Peak LR: 1.5e-4
- Decay: Cosine to 1.5e-5 (10% of peak)
- Tokens: ~14T

**Stage 2**: Continued pre-training
- Start: 1.5e-5 (where Stage 1 ended)
- Decay: Slow cosine to 1.5e-6
- Tokens: ~500B

**Stage 3**: Context extension
- Start: 1e-5 (reset slightly higher)
- Decay: Cosine to 5e-7
- Tokens: ~200B

**Stage 4**: Annealing
- Start: 5e-6
- Decay: Linear to near-zero
- Tokens: ~50B

### Why LR Must Decrease Across Stages

**Intuition**: Later stages are refinement, not major learning

**Early training** (Stage 1):
- Model learning basic patterns
- Large updates acceptable
- High LR (1e-4)

**Late training** (Stage 4):
- Model already capable
- Small adjustments only
- Low LR (1e-6 or less)

**Empirical finding**: Restarting at high LR in later stages causes instability and performance degradation

---

## 8. Practical Guidance

### When to Transition Between Stages

**Stage 1 → Stage 2 transition indicators**:
- Loss plateaus on validation set
- Capability evals show basic competence but ceiling effects
- Typically after 7-15T tokens
- Validation perplexity stops improving meaningfully

**Stage 2 → Stage 3 transition**:
- Target capabilities (code, math) reach acceptable level
- Typically after 500B-1T additional tokens
- Downstream benchmarks plateau

**Stage 3 → Stage 4 transition**:
- Context length extended to target (e.g., 128K)
- Long-context benchmarks (RULER, InfiniteBench) show competence
- Typically after 100-500B long-context tokens

### Data Preparation for Multi-Stage

**Organize data by stage**:

```python
datasets = {
    "stage1": {
        "shards": ["web_filtered/", "code/", "books/", ...],
        "weights": {"web": 0.70, "code": 0.08, ...},
        "epochs": {"web": 1, "code": 2, "books": 2, ...},
    },
    "stage2": {
        "shards": ["math_enhanced/", "code/", "synthetic_reasoning/", ...],
        "weights": {"math": 0.20, "code": 0.15, ...},  # Reweighted
        "epochs": {"math": 3, "code": 3, ...},
    },
    "stage3": {
        "shards": ["long_documents/", "synthetic_long/", "replay/", ...],
        "sequence_length": 32768,  # Extended
        "weights": {"long": 0.40, "synthetic": 0.30, "replay": 0.20, ...},
    },
    "stage4": {
        "shards": ["curated_high_quality/"],
        "epochs": 1,  # No repetition
        "weights": {"qa": 0.40, "benchmark_adjacent": 0.30, ...},
    },
}
```

**Key principles**:
- Pre-filter and organize data before training starts
- Shard data to enable efficient sampling
- Document mixture decisions for reproducibility

### Checkpoint Strategy

**Save at end of each stage**:
- Enables restart from any stage if issues found
- Can experiment with different Stage 2/3/4 strategies from same Stage 1 checkpoint
- Reduces risk of catastrophic training failures

**Track per-stage metrics separately**:
- Stage 1: General perplexity, broad capability evals
- Stage 2: Domain-specific benchmarks (HumanEval, MATH, MMLU)
- Stage 3: Long-context benchmarks (RULER, InfiniteBench)
- Stage 4: Final benchmark suite

**Storage considerations**:
- Full checkpoints at stage boundaries (100-500GB each)
- Lightweight checkpoints within stages for safety
- Total storage: ~1-2TB for 4-stage training

### Mixture Validation

**Before full training**:
1. **Train proxy models**: 1-7B models on proposed mixture
2. **Measure downstream metrics**: Evaluate on target benchmarks
3. **Iterate quickly**: Adjust mixture based on proxy results
4. **Commit to final mixture**: Scale to full model

**During training**:
- **Monitor domain-specific validation loss**: Each domain should show progress
- **Check capability benchmarks**: Code, math, reasoning metrics
- **Watch for overfitting**: Validation loss diverging from training loss

**After training**:
- **Comprehensive eval suite**: MMLU, GSM8K, HumanEval, etc.
- **Ablation studies** (if budget allows): Train with different mixtures to validate choices
- **Document findings**: What worked, what didn't, what would you change

---

# Part IV: Real-World Mixtures

## 9. What Major LLMs Use

### Foundation Model Data Mixtures

**Evolution of Mixtures (2020-2025)**:

| Model | Year | Total Tokens | Web | Code | Math | Multilingual | Key Innovation |
|-------|------|--------------|-----|------|------|--------------|----------------|
| **GPT-3** | 2020 | 300B | ~60% | ~8% | Minimal | Minimal | First large-scale mixture |
| **LLaMA 1** | 2023 | 1.4T | ~82% | 4.5% | 2.5% | Minimal | "Standard mixture" |
| **LLaMA 2** | 2023 | 2T | ~82% | 4.5% | 2.5% | ~2% | Extended LLaMA 1 |
| **LLaMA 3** | 2024 | 15T | ~50% | ~17% | ~25% | ~8% (30+ langs) | **4x code, 10x math** |
| **Qwen2.5** | 2024 | 18T | ~60% | ~25% | ~10% | 29+ langs | High code emphasis |
| **Phi-4** | 2024 | 10T | 30% | 20% | Included | 40 langs | **40% synthetic** |
| **DeepSeek-V3** | 2024 | 14.8T | Not disclosed | Not disclosed | Not disclosed | 119+ langs | Sources known |
| **Qwen3** | 2025 | 36T | Multi-stage | Multi-stage | Multi-stage | 119 langs | **3-stage curriculum** |
| **Mistral/Mixtral** | 2023-24 | ~8T | Not disclosed | Not disclosed | Not disclosed | Not disclosed | Proprietary |
| **Gemma 2/3** | 2024-25 | Not disclosed | Not disclosed | Not disclosed | Not disclosed | 140+ langs | Proprietary |

**Key patterns**:
- **Web dominance declining**: 82% (2023) → 50% (2024)
- **Code explosion**: 4.5% (2023) → 17-25% (2024)
- **Math emergence**: 2.5% (2023) → 10-25% (2024)
- **Synthetic adoption**: 0% (2023) → 40% (Phi-4, 2024)
- **Multi-stage becoming standard**: Qwen3, LLaMA 3.1 use stage-specific mixtures

### Domain Specialist Models

**Continued pre-training on specialized data**:

| Model | Base | Additional Training | Domain | Performance |
|-------|------|---------------------|--------|-------------|
| **Qwen2.5-Coder** | Qwen2.5 (18T) | +5.5T code tokens | Programming | HumanEval: 92.1 (32B) |
| **Qwen2.5-Math** | Qwen2.5 (18T) | Self-improvement RL | Mathematics | MATH: 83.1 vs base 50 |
| **Code Llama** | LLaMA 2 (2T) | +500B code tokens | Programming | HumanEval: 53.7 (34B) |
| **DeepSeek-Coder-V2** | DeepSeek-V2 | +6T code tokens | Programming | HumanEval: 90.2 (236B) |
| **DeepSeek-Math** | DeepSeek-V2 | Math-focused training | Mathematics | MATH: 78.5 (7B) |

**Strategy**: Train general model first, then continue pre-training on domain-specific data

**Why this works**:
- General model provides language understanding
- Continued training specializes without catastrophic forgetting
- More efficient than training specialist from scratch

---

## 10. Common Pitfalls

### 1. Poor Domain Weighting

**Problem**: Suboptimal domain proportions hurt downstream performance

**Example pitfalls**:
- **Too much web**: Model generates verbose, low-quality text
- **Too little code**: Poor programming capabilities despite upsampling
- **Math/code imbalance**: Strong coding but weak math (or vice versa)

**Solution**:
- Train proxy models (1-7B) with different mixtures
- Measure downstream task performance
- Iterate quickly before committing to full-scale training

### 2. Suboptimal Epoch Strategies

**Problem**: Repeating wrong data or repeating too much

**Pitfalls**:
- **Repeating web 3+ epochs**: Memorizing noisy patterns, overfitting
- **Single epoch on high-quality code**: Underutilizing valuable data
- **Excessive repetition** (10+ epochs): Model degradation

**Solution**:
- Web: 1 epoch max
- Code/books/Wikipedia: 2-3 epochs
- Math/reasoning: 2-4 epochs, supplement with synthetic

### 3. Multi-Stage Transition Timing

**Problem**: Transitioning too early or too late between stages

**Too early** (Stage 1 → Stage 2):
- Model lacks foundation for specialized learning
- Wasted compute on capability training before ready

**Too late** (Stage 1 → Stage 2):
- Diminishing returns on general pre-training
- Could have specialized earlier

**Solution**:
- Monitor validation loss plateaus
- Track capability benchmarks (HumanEval, MATH)
- Transition when baseline competence established but before plateau

### 4. Ignoring Multi-Stage Benefits

**Problem**: Training on single static mixture for entire run

**Consequence**:
- Missing 10-15% performance gains from annealing
- Inefficient use of compute (same mixture early and late)

**Solution**:
- Plan multi-stage from the start
- Reserve 10-20% of training budget for Stages 2-4
- Even simple 2-stage (main + annealing) provides major benefits

### 5. Benchmark Contamination in Mixtures

**Problem**: Including actual test data in training mixture

**Sources of contamination**:
- GSM8K problems on web forums
- HumanEval solutions on GitHub
- MMLU questions in study guides

**Solution**:
- Aggressive n-gram filtering against benchmark test sets
- Use benchmark-*adjacent* data (similar difficulty/format) in annealing stage
- Document decontamination procedures

---

## 11. Future Directions

### Near-term (2025-2026)

**1. Dynamic Mixture Optimization**:
- Real-time adjustment of domain weights during training
- Model self-evaluates capability gaps, requests more of specific data
- Adaptive mixtures based on loss signals

**2. More Sophisticated Synthetic Generation**:
- Multi-model ensembles for generation (GPT-4 + Claude + Gemini)
- Adversarial filtering (discriminator rejects low-quality synthetic)
- Recursive improvement (models generate training data for next generation)

**3. Automated Curriculum Learning**:
- Easy → hard progression within domains
- Model readiness signals trigger domain transitions
- Reinforcement learning for mixture optimization

### Research Frontiers

**1. Optimal Mixture Theory**:
- Principled approach to domain weighting
- Scaling laws for mixture composition
- Theoretical bounds on mixture benefit

**2. Continual Learning Mixtures**:
- Adding new domains without catastrophic forgetting
- Incremental data incorporation
- Lifelong learning from streaming data

**3. Task-Specific Mixtures**:
- Predict optimal mixture for target task
- Meta-learning for mixture optimization
- Transfer learning across mixture strategies

**4. Multimodal Data Mixing**:
- Optimal ratios of text, images, video, audio
- Cross-modal transfer effects
- Unified representation learning

### Open Questions

1. **Synthetic ceiling**: What percentage synthetic is optimal? Is 40% (Phi-4) near the limit?
2. **Domain granularity**: Should we split "code" into Python, JavaScript, C++, etc.?
3. **Mixture personalization**: Different mixtures for different model sizes/use cases?
4. **Stage count**: Is 4 stages optimal, or will we move to 6-8 micro-stages?
5. **Real-time adaptation**: Can mixtures adjust during training based on model signals?

---

## 12. Sources

### Foundational Papers

- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) - First large-scale mixture
- [The Pile: An 800GB Dataset of Diverse Text](https://arxiv.org/abs/2101.00027) - Systematic mixture design
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) - "Standard mixture"
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) - Validated mixture

### Mixing Strategy Research

- [DoReMi: Optimizing Data Mixtures Speeds Up LM Pretraining](https://arxiv.org/abs/2305.10429) - Learned mixture optimization
- [Textbooks Are All You Need (Phi-1)](https://arxiv.org/abs/2306.11644) - Synthetic-first approach
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) - Synthetic data at scale
- [Scaling Laws with Vocabulary](https://arxiv.org/abs/2407.13623) - Optimal mixture composition

### Multi-Stage Training

- [LLaMA 3 Model Card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md) - Multi-stage approach
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) - Curriculum learning
- [Gemma 2 Technical Report](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf) - Multi-stage distillation

### Modern Datasets

- [FineWeb: Decanting the Web for the Finest Text Data](https://arxiv.org/abs/2406.17557) - Quality filtering
- [DataComp-LM: In Search of the Next Generation of Training Sets](https://arxiv.org/abs/2406.11794) - Mixture benchmarking
- [RedPajama: An Open Dataset for Training Large Language Models](https://github.com/togethercomputer/RedPajama-Data) - Open reproduction

### Model-Specific Documentation

- [Qwen2.5 Blog Post](https://qwenlm.github.io/blog/qwen2.5/) - 18T token mixture details
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) - 14.8T tokens, 119 languages
- [Phi-4 Technical Report](https://arxiv.org/abs/2412.08905) - 40% synthetic mixture
- [Mistral AI Blog](https://mistral.ai/news/) - Mixture hints (proprietary)
