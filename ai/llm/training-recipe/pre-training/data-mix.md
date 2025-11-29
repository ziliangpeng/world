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

## 10. Model-Type-Specific Mixing Strategies

The data mixtures discussed so far focus primarily on general-purpose foundation models. However, different model specializations require fundamentally different data compositions. A code-specialized model uses 40-60% code (vs 4-5% in general models), math models rely heavily on synthetic reasoning chains, and multimodal models operate on entirely different data paradigms (image-text pairs, video).

This section explores the evolution and best practices for data mixing across five model types, each with distinct requirements and optimization strategies.

### 10.1 Text-Only Foundation Models (Baseline)

**Purpose**: General-purpose language understanding across diverse domains

This is the baseline covered extensively in earlier sections, provided here for comparison with specialized models.

**Standard Modern Mixture** (LLaMA 3 as reference):

| Domain | Proportion | Rationale |
|--------|------------|-----------|
| Web (filtered) | ~50% | General knowledge, linguistic diversity |
| Code | ~17% | Structured reasoning, programming capability |
| Math | ~25% | Mathematical reasoning foundation |
| Multilingual | ~8% | Global language coverage (30+ languages) |

**Key characteristics**:
- **Balanced across capabilities**: No single domain dominates excessively
- **Web reduction**: Down from 82% (LLaMA 2) to 50% (LLaMA 3)
- **Capability emphasis**: Code + math = 42% of mixture (vs 7% in LLaMA 2)
- **Multi-stage training**: Different mixtures across training phases (see Section 6)

**Evolution covered in Section 2**:
- Phase 1-2: Web-heavy (60-82% web)
- Phase 3: Balanced (50% web, capability-focused)
- Phase 4-5: Synthetic integration, multi-stage sophistication

**Reference performance** (LLaMA 3 70B):
- MMLU: 79.5
- HumanEval: 81.7
- GSM8K: 93.0
- MATH: 50.4

This baseline establishes the comparison point for specialized models below.

### 10.2 Code-Specialized Models

**Challenge**: General models (4-5% code) have limited programming capabilities. How much code is optimal, and how should it be mixed?

**Evolution Across Three Phases**:

#### Phase 1: Code-Augmented General Models (2021-2022)

**Codex** (OpenAI, 2021):
- **Approach**: GPT-3 (175B) + fine-tuning on GitHub code
- **Strategy**: Post-training on code, not pre-training mixture
- **Result**: HumanEval ~47% (12B variant)
- **Limitation**: General model foundation, code added later

**Key insight**: Code as fine-tuning works, but limited by general-model base

#### Phase 2: Code-Heavy Pre-training (2023)

**Code Llama** (Meta, August 2023):
- **Base**: LLaMA 2 (2T tokens, ~4.5% code)
- **Additional training**: +500B code tokens via continued pre-training
- **Final code proportion**: ~40% (500B code / 1.25T effective tokens)
- **Languages**: 20+ programming languages
- **Performance**: HumanEval 53.7 (34B model)

**StarCoder** (BigCode, May 2023):
- **Approach**: Code-first from scratch (not continued training)
- **Data**: 1T tokens, 80+ programming languages
- **Code proportion**: ~80% code, 20% natural language
- **Performance**: HumanEval 33.6 (15B model)
- **Trade-off**: Excellent code, weaker general reasoning

**Key finding**: **Continued training >> code-first**
- Code Llama (40% code) outperforms StarCoder (80% code)
- Preserving general language understanding critical
- Diminishing returns beyond 50-60% code

#### Phase 3: Extreme Code Focus (2024-2025)

**DeepSeek-Coder-V2** (June 2024):
- **Base**: DeepSeek-V2 (general MoE model, 21T tokens)
- **Additional training**: +6T code tokens
- **Final code %**: ~50-60% (6T code / ~10T effective continued training)
- **Languages**: 300+ programming languages
- **Performance**: HumanEval 90.2 (236B MoE, 21B active)

**Qwen2.5-Coder** (November 2024):
- **Base**: Qwen2.5 (18T tokens, includes code)
- **Additional training**: +5.5T code tokens
- **Final code %**: ~60% during continued training phase
- **Languages**: 92 programming languages
- **Performance**: HumanEval 92.1 (32B model), **state-of-the-art**

**Key innovation**: Massive continued training (5-6T tokens) on code

**Data Mixture Comparison**:

| Model | Base Tokens | Additional Code | Final Code % | Strategy | HumanEval |
|-------|-------------|-----------------|--------------|----------|-----------|
| **Code Llama 7B** | LLaMA 2 (2T) | +500B code | ~40% | Continued pre-training | 33.5 |
| **Code Llama 34B** | LLaMA 2 (2T) | +500B code | ~40% | Continued pre-training | 53.7 |
| **StarCoder 15B** | From scratch | 1T code (80+ langs) | ~80% | Code-first | 33.6 |
| **DeepSeek-Coder-V2** | DeepSeek-V2 (21T) | +6T code | ~50-60% | Continued pre-training | 90.2 |
| **Qwen2.5-Coder 32B** | Qwen2.5 (18T) | +5.5T code | ~60% | Continued pre-training | 92.1 |

**Performance vs Code % Correlation**:

```
Code Proportion → HumanEval Pass@1 (approximate trends):

20-30% code:  ~50-60 HumanEval (general models with code upsampling)
40-50% code:  ~70-85 HumanEval (Code Llama tier)
50-60% code:  ~85-92 HumanEval (DeepSeek-Coder-V2, Qwen2.5-Coder)
80%+ code:    ~65-75 HumanEval (StarCoder - too specialized, weak reasoning)
```

**Diminishing returns pattern**:
- 0% → 20% code: Massive gains (+40-50 HumanEval)
- 20% → 40% code: Large gains (+20-30 HumanEval)
- 40% → 60% code: Moderate gains (+10-15 HumanEval)
- 60% → 80% code: Negative returns (hurts general reasoning)

**Key Insights**:

1. **Optimal code proportion**: 40-60% for specialist models
   - Below 40%: Underutilizing code's benefit
   - Above 60%: Hurting general language understanding
   - StarCoder's 80% is too extreme (weak natural language reasoning)

2. **Continued training dominates**:
   - Train general model first (broad knowledge)
   - Continue pre-training on code (specialization)
   - More efficient than code-first (DeepSeek-Coder-V2, Qwen2.5-Coder prove this)

3. **Language diversity matters**:
   - Early models: 20-40 languages
   - Modern: 80-300 languages
   - Breadth improves cross-language transfer, niche language support

4. **Synthetic code emergence**:
   - Early: Pure GitHub scraping
   - Modern: GitHub + synthetic code explanations
   - Synthetic helps: algorithm variations, educational code, edge cases
   - Proportion: ~10-20% synthetic within code mixture

5. **Massive scale unlocks specialist performance**:
   - Code Llama: +500B tokens → HumanEval 53.7
   - DeepSeek-Coder-V2: +6T tokens → HumanEval 90.2
   - Qwen2.5-Coder: +5.5T tokens → HumanEval 92.1
   - **10x more code data = +40-point HumanEval improvement**

**Practical recommendations**:
- **Small models** (<10B): 40-50% code (Code Llama approach)
- **Large models** (30B+): 50-60% code (Qwen2.5-Coder approach)
- **Always** start from strong general base, then continue pre-training
- Include synthetic code (10-20%) for educational quality
- Cover 50+ languages minimum, 100+ ideal for global use

### 10.3 Math-Specialized Models

**Challenge**: General models struggle with mathematical reasoning. LLaMA 3 70B achieves only ~50 on MATH benchmark despite 25% math in pre-training. How do specialized math models reach 78-83?

**Evolution Across Three Phases**:

#### Phase 1: Math-Augmented General Models (2022-2023)

**Minerva** (Google, June 2022):
- **Base**: PaLM (540B parameters)
- **Additional training**: 38.5B tokens of scientific/math data
- **Sources**: arXiv papers, math webpages, math textbooks
- **Strategy**: Continued pre-training on natural math content
- **Performance**: MATH 50.3, GSM8K 78.5
- **Limitation**: Relies on naturally occurring math, doesn't scale

**LLaMA 3** (April 2024):
- **Approach**: Heavy math upsampling in base pre-training (25% math)
- **Data**: Mix of textbooks, arXiv, math problems
- **Performance**: MATH ~50 (70B model)
- **Key insight**: Even 25% math in 15T tokens plateaus around MATH 50

**Challenge identified**: Natural math data is limited. Need synthetic generation to break through.

#### Phase 2: Synthetic Reasoning Chains (2023-2024)

**DeepSeek-Math** (February 2024):
- **Base**: DeepSeek-Coder (good reasoning foundation from code)
- **Additional training**: Math-focused continued pre-training
- **Key innovation**: **Synthetic chain-of-thought reasoning**
  - Generate step-by-step solutions to math problems
  - Use stronger models (GPT-4, Claude) to create training data
  - Focus on process, not just answers

**Math data composition**:

| Data Type | Proportion | Purpose |
|-----------|------------|---------|
| Synthetic reasoning chains | 40% | Step-by-step problem solving |
| Math competition problems | 30% | High-difficulty training |
| Natural math (arXiv, textbooks) | 20% | Conceptual foundations |
| Self-generated solutions | 10% | Model improving itself |

**Performance**: MATH 78.5 (7B model) - **28-point improvement over LLaMA 3 70B**

**Key techniques**:
1. **Chain-of-thought generation**: Models learn reasoning *process*, not memorize answers
2. **Difficulty progression**: Easy → medium → hard problems across training
3. **Self-consistency**: Generate multiple solutions, train on verified correct ones
4. **Process supervision**: Reward intermediate steps, not just final answers

#### Phase 3: Self-Improvement & RL (2024-2025)

**Qwen2.5-Math** (September 2024):
- **Base**: Qwen2.5 (18T tokens, includes math)
- **Stage 1**: Math-focused continued pre-training (similar to DeepSeek-Math)
- **Stage 2**: **Reinforcement learning from self-improvement**
  - Model generates solutions to math problems
  - Verifier checks correctness (outcome reward)
  - Process reward model scores intermediate steps
  - Iterate: model improves → generates better data → trains on it

**RL data loop**:
```
1. Base model solves 10K math problems
2. Verifier identifies correct solutions (40% correct)
3. Process reward model scores reasoning quality
4. Train on high-quality correct solutions
5. Repeat with improved model (now 60% correct)
6. Continue until convergence
```

**Performance**: MATH 83.1 (7B model), **85.9 (72B model)** - **state-of-the-art**

**Qwen2.5-Math-Instruct** (instruction-tuned variant):
- MATH 87.0 (7B model)
- Demonstrates post-training amplifies pre-training math ability

**Data Mixture Evolution**:

| Model | Base | Math Strategy | MATH Score | Key Innovation |
|-------|------|---------------|------------|----------------|
| **Minerva (540B)** | PaLM | +38.5B natural math | 50.3 | Natural math sources |
| **LLaMA 3 (70B)** | From scratch | 25% math in 15T tokens | ~50 | Heavy math upsampling |
| **DeepSeek-Math (7B)** | DeepSeek-Coder | Synthetic reasoning chains | 78.5 | Chain-of-thought generation |
| **Qwen2.5-Math (7B)** | Qwen2.5 | Self-improvement RL | 83.1 | Iterative self-generated data |
| **Qwen2.5-Math (72B)** | Qwen2.5 | Self-improvement RL | 85.9 | Scale + RL |

**Math Data Composition (Modern Approach)**:

```
Traditional (Minerva, LLaMA 3):
├── Math textbooks: 30%
├── arXiv papers: 25%
├── Math problem datasets: 25%
├── Math webpages: 15%
└── Synthetic: 5%

Modern (DeepSeek-Math, Qwen2.5-Math):
├── Synthetic reasoning chains: 40-50%
├── Self-generated solutions: 20-25%
├── Competition problems (IMO, AMC): 15-20%
├── Natural math (textbooks, arXiv): 10-15%
└── Verified step-by-step solutions: Throughout
```

**Key Insights**:

1. **Synthetic data is essential**:
   - Natural math data caps performance at MATH ~50
   - Synthetic reasoning chains unlock 78-85 range
   - Can't break through without generation

2. **Chain-of-thought >> answers**:
   - Training on "Answer: 42" teaches memorization
   - Training on "Let x = ..., then ... therefore 42" teaches reasoning
   - Process supervision crucial

3. **Self-improvement works**:
   - Model generates training data for next iteration
   - Verifier ensures quality (reject wrong solutions)
   - Iterative improvement: each cycle boosts capability

4. **Process rewards > outcome rewards**:
   - Outcome: "Did you get 42?" (binary)
   - Process: "Is each step logically sound?" (fine-grained)
   - Process rewards guide better reasoning patterns

5. **Verification is critical**:
   - Synthetic math can be wrong (hallucinated steps)
   - Use verifiers: symbolic checkers, stronger models, multiple solutions
   - Only train on verified correct reasoning chains

6. **Small models can excel with good data**:
   - DeepSeek-Math 7B (78.5) beats Minerva 540B (50.3)
   - **70x smaller, better performance** through data quality
   - Qwen2.5-Math 7B (83.1) matches much larger models

**Performance vs Data Quality**:

```
Natural math only:        MATH ~50 (ceiling)
+ Synthetic reasoning:    MATH ~78-80
+ Self-improvement:       MATH ~83-86
+ RL fine-tuning:         MATH ~87-90 (instruct models)
```

**Practical Recommendations**:

- **Don't rely on natural math alone**: Limited quantity, caps at MATH ~50
- **Generate synthetic reasoning chains**: Use GPT-4/Claude to create step-by-step solutions
- **Implement verification**: Symbolic checkers (for algebra), stronger model verification
- **Progressive difficulty**: Start easy (basic arithmetic) → hard (competition math)
- **Process supervision**: Reward intermediate reasoning, not just answers
- **Self-improvement loop**: Let model generate training data, verify, iterate
- **Mix synthetic with natural**: 40-50% synthetic, 10-20% natural for grounding

### 10.4 Reasoning Models (o1-Style)

**IMPORTANT CAVEAT**: OpenAI has released very limited technical details about o1, o1-pro, and o3. This section presents educated inferences based on:
- OpenAI's o1 system card and blog posts
- Published research on chain-of-thought and process supervision
- Observable behavior and benchmark performance
- Industry analysis and reverse-engineering attempts

**Treat mixture estimates as hypotheses, not confirmed facts.**

**The Reasoning Paradigm Shift**:

Traditional models output answers directly. Reasoning models (o1, o3) generate explicit "thinking" tokens before answering, showing step-by-step reasoning. This fundamental change affects data mixture requirements.

**Hypothesized Evolution**:

#### Phase 1: Chain-of-Thought Post-Training (2022-2023)

**GPT-3.5 + CoT Prompting**:
- Base model: GPT-3.5 (standard pre-training)
- Post-training: Instruction tuning with chain-of-thought examples
- Data: Human-written reasoning chains for math, logic problems
- Limitation: CoT as prompt pattern, not core capability

**Early research findings**:
- Few-shot CoT prompting improves reasoning significantly
- Model benefits from seeing step-by-step examples
- But requires prompting - not native reasoning

#### Phase 2: RL-Based Reasoning (2023-2024)

**Hypothesized GPT-4 → o1 transition approach**:

**Process reward models (PRM)**:
- Train reward model to score intermediate reasoning steps
- Not just "is answer correct?" but "is step 3 logically valid?"
- Requires dataset of solutions with step-level annotations

**Reinforcement learning setup**:
```
1. Base model generates solution with reasoning steps
2. PRM scores each step (0-1 quality score)
3. Outcome verifier checks final answer (correct/incorrect)
4. Combined reward = process_score × outcome_correctness
5. Train with PPO/similar to maximize combined reward
```

**Key insight**: Process supervision >> outcome supervision
- Outcome only: "Answer wrong, try again" (sparse signal)
- Process: "Step 3 is flawed because..." (dense signal)
- Research shows 2-3x improvement with process rewards

#### Phase 3: Reasoning-Focused Pre-training (2024-2025)

**o1 hypothesized approach**:

OpenAI likely uses **multi-stage training** with reasoning-heavy data mixture:

**Stage 1: Base pre-training** (similar to GPT-4):
- Standard mixture: web, code, books, etc.
- Foundation for general knowledge

**Stage 2: Reasoning-focused continued pre-training**:
- Hypothesized mixture shift toward reasoning-dense data
- Synthetic chain-of-thought examples
- Explicit "thinking" vs "answering" modes

**Stage 3: RL for reasoning refinement**:
- Process reward models
- Self-play improvement loops
- Verification-guided training

**Inferred Data Mixture (o1-style)**:

**SPECULATIVE** - based on performance characteristics:

| Data Type | Estimated % | Rationale |
|-----------|-------------|-----------|
| **Reasoning chains** | 30-40% | Core capability - step-by-step problem solving |
| **Math problems** | 20-30% | Structured reasoning foundation (o1 excels at MATH: ~94) |
| **Code + explanations** | 15-20% | Logical thinking patterns (HumanEval: ~92) |
| **Scientific reasoning** | 10-15% | Complex multi-step analysis (GPQA: ~78) |
| **General web/text** | 10-20% | Preserve general knowledge |

**Reasoning chain composition** (hypothesized):
```
Synthetic generated CoT: 60-70% (GPT-4, Claude generate examples)
Human-written reasoning: 20-30% (competition math, textbooks)
Self-generated (model's own): 10-20% (self-improvement)
```

**Key Characteristics of o1 Data (Inferred)**:

1. **Explicit thinking tokens**:
   - Models trained to separate "thinking" from "answer"
   - Special tokens marking reasoning vs output phases
   - Possibly: `<think>step 1...</think><answer>result</answer>`

2. **Process supervision throughout**:
   - Every reasoning chain annotated for step quality
   - Verification at each step, not just final answer
   - Possibly: Human contractors score step validity

3. **Multi-turn reasoning**:
   - Problems requiring iterative refinement
   - Self-correction examples
   - "Try approach A, fails, try approach B, succeeds"

4. **Verification-heavy**:
   - High ratio of verifier data (check if solution correct)
   - Self-consistency: generate 10 solutions, train on agreements
   - Formal verification where possible (math, code)

**Performance Characteristics (Observed)**:

| Benchmark | GPT-4 | o1-preview | o1 | o3 | Interpretation |
|-----------|-------|------------|----|----|----------------|
| **MATH** | ~50 | ~85 | ~94 | ~97 | Heavy math reasoning data |
| **GPQA** | ~50 | ~75 | ~78 | ~88 | Scientific reasoning emphasis |
| **Codeforces** | ~10th | ~89th | ~93rd | ~97th | Competition programming focus |
| **MMLU** | ~86 | ~88 | ~90 | ~91 | General knowledge preserved |

**Key observations**:
- Massive gains on reasoning tasks (MATH +44 points)
- Smaller gains on knowledge tasks (MMLU +4 points)
- Suggests data mixture heavily emphasizes reasoning over knowledge breadth

**Inferred Training Pipeline**:

```
Stage 1: Base Model (GPT-4-level)
├── Standard pre-training mixture
└── Broad knowledge, basic reasoning

        ↓

Stage 2: Reasoning-Focused Continued Pre-training
├── Mixture shift: 30-40% reasoning chains
├── Synthetic CoT generation
└── Explicit thinking/answering separation

        ↓

Stage 3: Process-Supervised RL
├── Process reward model training
├── RL with dense step-level feedback
├── Self-improvement iterations
└── Verification-guided refinement

        ↓

Result: o1 (reasoning specialist)
```

**Open Questions** (Awaiting Public Research):

1. **Exact mixture proportions**: How much reasoning data exactly?
2. **Synthetic vs natural ratio**: Percentage of generated vs human-written reasoning?
3. **Pre-training vs post-training split**: How much is pre-training mixture vs RL?
4. **Thinking token mechanism**: Are special tokens used? How are they implemented?
5. **Process reward model details**: How is step-level quality scored?
6. **Self-play dynamics**: Does o1 generate training data for improved versions?
7. **Verification systems**: What verifiers are used (symbolic, model-based)?

**Why Limited Public Information?**

OpenAI considers o1's training approach a significant competitive advantage:
- Process supervision techniques proprietary
- Data mixture optimizations confidential
- RL training pipelines trade secrets

**Lessons for Practitioners** (Based on Available Evidence):

1. **Chain-of-thought is essential**:
   - Don't train on answers alone
   - Include step-by-step reasoning in data
   - Explicit reasoning improves generalization

2. **Process supervision helps**:
   - Published research confirms process > outcome
   - Invest in step-level reward models
   - Annotate reasoning quality, not just correctness

3. **Synthetic reasoning data works**:
   - Use GPT-4/Claude to generate CoT examples
   - Verify generated reasoning with stronger models
   - Mix synthetic with human-written (60/40 split plausible)

4. **Separate thinking from answering**:
   - Train models to show reasoning explicitly
   - Possibly use special tokens or formatting
   - Makes reasoning inspectable and trainable

5. **Multi-stage approach**:
   - Base model first (general knowledge)
   - Reasoning-focused continued training
   - RL refinement for quality

**Comparison to Math Models**:

o1-style reasoning models overlap with but differ from math specialists:
- **Math models** (DeepSeek-Math, Qwen2.5-Math): Optimize for math domain specifically
- **Reasoning models** (o1): Optimize for general reasoning (math, code, science, logic)
- **Data difference**: o1 likely more diverse reasoning (15-20% science vs 0% in math models)

**Future Directions**:

As OpenAI releases more information (likely in future papers):
- Refined mixture estimates
- Confirmed process supervision techniques
- Public benchmarks for reasoning-focused training
- Open-source reasoning model attempts (DeepSeek-R1 rumored)

**Summary**:

While specifics remain confidential, evidence suggests o1 uses:
- **30-40% reasoning chain data** (vs 10-25% math in general models)
- **Process supervision** throughout training
- **Multi-stage approach**: base → reasoning-focused → RL refinement
- **Synthetic heavy**: 60-70% of reasoning data likely generated
- **Explicit thinking tokens**: Separation of reasoning and answering

These are educated hypotheses pending public research confirmation.

### 10.5 Multimodal Models (Vision-Language Models)

**Fundamental Paradigm Shift**: Multimodal models don't just use different text mixtures - they train on entirely different data types (images, video, audio). The "mixture" problem becomes multi-dimensional: how to balance modalities, not just text domains.

This subsection focuses on **vision-language models (VLMs)** - the most mature multimodal category.

**Evolution Across Three Phases**:

#### Phase 1: Image-Text Pair Training (2021-2023)

**CLIP** (OpenAI, January 2021):
- **Paradigm**: Contrastive learning on image-text pairs
- **Data**: 400M image-text pairs from the web
- **Training**: Separate image encoder (ViT) + text encoder (Transformer)
- **Loss**: Maximize similarity of matching pairs, minimize mismatched pairs
- **Limitation**: No text generation, only image-text matching

**DALL-E** (OpenAI, January 2021):
- **Paradigm**: Text → image generation
- **Data**: 250M image-text pairs
- **Approach**: Treat images as sequences of discrete tokens
- **Limitation**: Generation only, no vision understanding

**Flamingo** (DeepMind, April 2022):
- **Paradigm**: Frozen LLM + vision adapter
- **Data**: Interleaved image-text documents from web
- **Architecture**:
  - Pre-trained LLM (frozen): 70B Chinchilla
  - Vision encoder (frozen): CLIP
  - Cross-attention adapters (trainable): Connect vision to language
- **Key innovation**: Can process documents with images and text mixed
- **Limitation**: Expensive (requires massive LLM)

**Key insight**: Frozen LLM preserves language ability, adapter learns vision

#### Phase 2: Unified Multimodal Pre-training (2023-2024)

**LLaVA** (April 2023) - **Democratized VLMs**:
- **Paradigm**: Instruction tuning for vision understanding
- **Base components**:
  - LLM: LLaMA / Vicuna (7B-13B)
  - Vision encoder: CLIP ViT-L/14
  - Projection: Simple linear layer connecting vision to language
- **Data**: 665K vision-language instruction pairs (GPT-4 generated)
- **Training stages**:
  1. Pre-training: Align vision encoder to LLM (freeze both, train projection)
  2. Instruction tuning: Train LLM + projection on conversation data

**LLaVA 1.5** (October 2023):
- **Data**: 1.3M instruction pairs (expanded dataset)
- **Key improvement**: Academic task instructions + conversation data
- **Performance**: Competitive with GPT-4V on many tasks

**GPT-4V** (September 2023):
- **Details**: Proprietary (limited technical info)
- **Presumed approach**: Native multimodal training in GPT-4
- **Scale**: Massive (billions of image-text pairs likely)
- **Performance**: State-of-the-art vision understanding

**Gemini** (December 2023):
- **Paradigm**: Native multimodal from scratch
- **Key claim**: Not adapter-based, jointly trained
- **Data**: Text, images, video, audio (proportions undisclosed)
- **Scale**: Largest reported multimodal training (exact tokens unknown)

**Paradigm split emerging**:
- **Adapter approach** (LLaVA): Cheap, preserves LLM quality, slightly lower vision performance
- **Native multimodal** (Gemini, GPT-4V): Expensive, potentially better integration, state-of-the-art

#### Phase 3: Long-Context Multimodal & Video (2024-2025)

**Gemini 1.5** (February 2024):
- **Key innovation**: 1M token context with video understanding
- **Data mixture** (inferred):
  - Long video sequences
  - Interleaved image-text documents
  - Audio-visual pairs
  - Text (majority for language ability)
- **Capability**: Process 1 hour of video, 11 hours of audio in context

**GPT-4o** (May 2024):
- **Omni-modal**: Text + vision + audio, native integration
- **Real-time**: Audio input/output with visual context
- **Data** (speculative): Heavy emphasis on audio-visual-text triplets

**Qwen-VL** & **Qwen2-VL** (2023-2024):
- **Approach**: Adapter-based like LLaVA
- **Base**: Qwen LLM (strong multilingual)
- **Data**: 1.4B image-text pairs (multilingual)
- **Key strength**: Multilingual vision understanding

**Data Mixture Comparison**:

| Model | Architecture | Text-Only | Image-Text Pairs | Video | Training Approach |
|-------|--------------|-----------|------------------|-------|-------------------|
| **CLIP** | Dual encoder | - | 400M pairs | - | Contrastive learning |
| **Flamingo** | Frozen LLM + adapter | 70B (frozen Chinchilla) | Interleaved docs | - | Adapter training only |
| **LLaVA 1.5** | Frozen LLM + adapter | 13B (LLaMA/Vicuna) | 1.3M instruction pairs | - | 2-stage: align + instruct |
| **GPT-4V** | Native multimodal | Unknown (GPT-4 scale) | Billions (estimated) | Unknown | Proprietary |
| **Gemini 1.0** | Native multimodal | Unknown | Unknown | Unknown | Joint training |
| **Gemini 1.5** | Native multimodal | Unknown | Unknown | Long video | 1M context |
| **Qwen-VL** | Frozen LLM + adapter | Qwen baseline | 1.4B pairs | - | Multilingual focus |

**Mixture Ratios (Estimated for Native Multimodal)**:

**LLaVA approach** (adapter-based):
```
Stage 1: Vision-language alignment (595K image-caption pairs)
├── Text component: Pre-trained frozen LLM (100% preserved)
└── Vision alignment: Learn projection only

Stage 2: Instruction tuning
├── Image-text instructions: 665K pairs
├── Additional conversation data: ~200K
└── LLM fine-tuning: Adjust to visual inputs
```

**Hypothesized Gemini/GPT-4V approach** (native multimodal):
```
Pre-training mixture (speculative):
├── Text-only: 70-85% (preserve language ability)
├── Image-text pairs: 10-20% (vision grounding)
├── Video: 2-5% (temporal understanding)
├── Audio: 1-3% (speech/sound)
└── Interleaved multimodal documents: 5-10%

Instruction tuning:
├── Vision-language instructions: 1-3%
├── Multi-turn conversations: 1-2%
└── Task-specific data: <1%
```

**Critical insight**: Text dominates even in multimodal models (70-85%)
- Too much vision data hurts language understanding
- Language ability is foundation; vision is extension
- Models must excel at language first, vision second

**Key Insights**:

1. **Two architectural paradigms**:
   - **Adapter (LLaVA, Qwen-VL)**:
     - Pros: Cheap ($1K for LLaVA 1.5), preserves LLM quality, easy to upgrade LLM
     - Cons: Slightly lower vision performance, adapter can be bottleneck
   - **Native (Gemini, GPT-4V)**:
     - Pros: Better vision-language integration, state-of-the-art
     - Cons: Expensive (millions of dollars), must retrain for LLM improvements

2. **Image-text pair quantity matters**:
   - Small models (7-13B): 1-2B pairs sufficient (LLaVA, Qwen-VL)
   - Large models (100B+): Billions to tens of billions estimated
   - Diminishing returns after ~1-2B for small models

3. **Instruction tuning is critical**:
   - Raw image-caption pairs insufficient for conversation
   - Need vision-language instruction pairs (Q&A, reasoning)
   - GPT-4 can generate high-quality instruction data from captions

4. **Text preservation is paramount**:
   - Bad VLMs: Lose language ability while gaining vision
   - Good VLMs: Preserve language, add vision
   - Keep text-only data >> vision data (70-85% vs 10-20%)

5. **Video = sequential images + special considerations**:
   - Video tokens: ~100x more than images (1 second video = 30 frames)
   - Temporal modeling: Different attention patterns
   - Expensive: Hard to scale (Gemini 1.5 is exception)
   - Current focus: Short clips (<1 minute) in most models

6. **Modality balance**:
   ```
   Text-only: 70-85% (language foundation)
   Image-text: 10-20% (vision grounding)
   Video: 2-5% (temporal understanding)
   Audio: 1-3% (speech/sound)
   Interleaved: 5-10% (real-world documents)
   ```

**Data Sources for VLM Training**:

**Image-text pairs**:
- LAION-400M / LAION-5B: Web-scraped image-alt-text pairs
- COYO-700M: Curated Common Crawl image-text
- DataComp-1B: Filtered, high-quality pairs
- CC12M, CC3M: Smaller curated datasets
- Proprietary: Google, OpenAI scrape massive additional data

**Instruction data**:
- GPT-4 generated: Captions → questions/answers/reasoning
- Human-annotated: Visual reasoning, VQA datasets
- Academic: VQA, GQA, Visual Genome, etc.

**Video** (limited availability):
- WebVid-10M: Text-video pairs
- HowTo100M: Instructional videos
- Proprietary: YouTube (Google), internal data

**Quality vs Quantity**:
- **Adapter approach**: Quality matters more (665K high-quality instruction pairs > 100M raw captions)
- **Native approach**: Quantity matters more (billions of pairs needed for joint training)

**Practical Recommendations**:

**For small teams / budget-constrained**:
- Use adapter approach (LLaVA-style)
- Start with pre-trained LLM (LLaMA, Qwen, Mistral)
- Use pre-trained vision encoder (CLIP, SigLIP)
- Focus on high-quality instruction data (1-2M pairs)
- Cost: $1K-$10K GPU time

**For large organizations / frontier models**:
- Consider native multimodal (Gemini-style)
- Scrape/curate billions of image-text pairs
- Joint training from scratch or early pre-training integration
- Budget: Millions of dollars

**Emerging Trends**:
1. **Efficient vision encoders**: Reducing vision tokens (from 256 to 64 tokens per image)
2. **Video compression**: Temporal pooling to handle long videos
3. **Audio-visual**: Integrating sound (GPT-4o-style)
4. **High-resolution**: Supporting images up to 4K (Qwen2-VL)
5. **Multilingual vision**: Non-English image understanding (Qwen-VL)

**Open Research Questions**:

1. What's the optimal text/image ratio for different model sizes?
2. How much video data is beneficial without hurting efficiency?
3. Should vision be integrated early (Gemini) or late (LLaVA) in training?
4. How to efficiently scale to 10M+ image-text pairs?
5. Multi-task learning: Should vision models train on OCR, detection, segmentation simultaneously?

**Summary**:

Multimodal data mixing is fundamentally different:
- **Text remains dominant**: 70-85% even in vision-language models
- **Image-text pairs**: 10-20% for vision grounding (1-10B pairs depending on scale)
- **Video**: 2-5% when included (very expensive)
- **Adapter vs native**: Cheap and effective vs expensive and state-of-the-art
- **Instruction tuning critical**: Raw captions insufficient for conversational ability

The field is rapidly evolving - expect major changes in 2025-2026 as video understanding and audio integration mature.

---

## 11. Common Pitfalls

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

## 12. Future Directions

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

## 13. Sources

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

### Code-Specialized Models

- [Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950) - Continued pre-training approach
- [StarCoder: A State-of-the-Art LLM for Code](https://arxiv.org/abs/2305.06161) - Code-first training
- [DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models](https://arxiv.org/abs/2406.11931) - 6T code tokens
- [Qwen2.5-Coder Technical Report](https://qwenlm.github.io/blog/qwen2.5-coder/) - 5.5T code tokens, 92 languages

### Math-Specialized Models

- [Solving Quantitative Reasoning Problems with Language Models (Minerva)](https://arxiv.org/abs/2206.14858) - Natural math data approach
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300) - Synthetic reasoning chains
- [Qwen2.5-Math Technical Report](https://qwenlm.github.io/blog/qwen2.5-math/) - Self-improvement RL
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) - Process supervision for math

### Reasoning Models

- [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/) - Limited technical details
- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903) - CoT foundations
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) - Zero-shot CoT
- [STaR: Self-Taught Reasoner](https://arxiv.org/abs/2203.14465) - Self-improvement for reasoning

### Multimodal Models (VLM)

- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) - Image-text contrastive learning
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) - Frozen LLM + adapter
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) - Instruction tuning for vision
- [Improved Baselines with Visual Instruction Tuning (LLaVA 1.5)](https://arxiv.org/abs/2310.03744) - Expanded instruction data
- [Gemini Technical Report](https://arxiv.org/abs/2312.11805) - Native multimodal training
- [GPT-4V System Card](https://openai.com/index/gpt-4v-system-card/) - Vision capabilities (limited details)
- [Qwen-VL: A Versatile Vision-Language Model](https://arxiv.org/abs/2308.12966) - Multilingual vision understanding
