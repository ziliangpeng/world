# AI Labs Documentation TODO

## Overview
Organize and expand Chinese AI labs documentation by creating focused files for each lab's models, evolution, and analysis.

---

## Phase 1: Organize Existing Content

### Chinese AI Labs Comparison
- [ ] Extract comparison section from index into dedicated file: `ailabs-llm-chinese-comparison.md`
- [ ] Include: company overviews, funding, strategic positioning, team info, timeline
- [ ] Add comparison tables across all labs (models, params, context window, open source status)
- [ ] Add competitive landscape analysis
- [ ] Add market positioning matrix

### DeepSeek Lab Files
- [ ] Create `ailabs-deepseek-models.md` - consolidate model details from main file
  - ABAB series models (1.0, 1.5, 2.0, 2.5)
  - DeepSeek-R1 family (reasoning models)
  - Training specs, benchmarks, open source status
  - Paper links and technical resources

- [ ] Create `ailabs-deepseek-evolution.md` - split from existing evolution file
  - Timeline of releases and milestones
  - Technical breakthroughs (MLA, MoE scaling, RL)
  - Competitive positioning evolution
  - Pricing evolution and market strategy changes

- [ ] Create `ailabs-deepseek-analysis.md` - public analysis compilation
  - Blog posts and technical analyses (VentureBeat, MarkTechPost, etc.)
  - Research papers and preprints
  - Community discussions and benchmarks
  - Competitive comparisons

### MiniMax Lab Files
- [ ] Create `ailabs-minimax-models.md` - consolidate from main file
  - ABAB 6.5 series
  - MiniMax-Text-01 & MiniMax-VL-01
  - MiniMax-M1 & M1-80K
  - MiniMax-M2
  - Speech-02, Music-01, Hailuo video models
  - Complete technical specifications

- [ ] Create `ailabs-minimax-evolution.md` - model evolution & milestones
  - MoE journey (2023 research → ABAB 6.5 → M1)
  - Lightning Attention development
  - Multimodal expansion (speech, music, video)
  - Hailuo platform evolution

- [ ] Create `ailabs-minimax-analysis.md` - public analysis
  - Technical deep-dives (Hugging Face blogs, APIdog)
  - Research papers (arXiv links)
  - Community analysis and comparisons
  - Performance benchmarks and evaluations

### Moonshot Lab Files
- [ ] Create `ailabs-moonshot-models.md`
  - Kimi chat models (versions and variants)
  - Claude collaboration
  - Long context journey
  - Technical specifications

- [ ] Create `ailabs-moonshot-evolution.md`
  - Founding to long-context pioneer
  - Product development timeline
  - Competitive positioning changes

- [ ] Create `ailabs-moonshot-analysis.md`
  - Public analysis and reviews
  - Kimi vs Claude comparisons
  - Market reception

### Qwen/Alibaba Lab Files
- [ ] Create `ailabs-qwen-models.md`
  - Qwen family (1.5, 2, 2.5, 3, etc.)
  - Model evolution and improvements
  - Parameters, context, specifications

- [ ] Create `ailabs-qwen-evolution.md`
  - Alibaba's AI journey
  - Qwen positioning and strategy
  - Technical milestones

- [ ] Create `ailabs-qwen-analysis.md`
  - Public analysis
  - Integration into Alibaba products
  - Competitive analysis

### Zhipu/GLM Lab Files
- [ ] Create `ailabs-zhipu-models.md`
  - GLM family (1.4, 4, 4V, etc.)
  - ChatGLM evolution
  - Technical specifications

- [ ] Create `ailabs-zhipu-evolution.md`
  - Academic spinoff story
  - Tsinghua connection
  - Model evolution timeline

- [ ] Create `ailabs-zhipu-analysis.md`
  - Public analysis
  - Academic perspective
  - Community reception

---

## Phase 2: Enhanced Comparisons

### New Comparison Files
- [ ] Create `ailabs-llm-chinese-comparison-models.md`
  - Comprehensive model comparison table (all labs, all major models)
  - Context window comparison
  - Parameter efficiency comparison
  - Open source vs proprietary breakdown
  - Multimodal capabilities comparison

- [ ] Create `ailabs-llm-chinese-comparison-strategy.md`
  - Business model comparison
  - Funding and valuation timeline
  - Team composition and experience
  - Geographic positioning
  - International expansion strategy

- [ ] Create `ailabs-llm-chinese-comparison-technical.md`
  - Architecture innovations by lab
  - Training efficiency comparison
  - Inference cost analysis
  - Research publication volume
  - Patent landscape

---

## Phase 3: Special Topics

### Cross-Lab Analysis
- [ ] Create `ailabs-llm-chinese-mooe-efficiency.md`
  - MoE adoption across labs (DeepSeek, MiniMax, Qwen)
  - MoE vs dense comparison
  - Routing strategies
  - Expert specialization

- [ ] Create `ailabs-llm-chinese-long-context.md`
  - Long context comparison (MiniMax 4M, Moonshot 2M, others)
  - Technical approaches to extending context
  - Benchmark results on long-context tasks
  - Use cases and applications

- [ ] Create `ailabs-llm-chinese-multimodal.md`
  - Multimodal models by lab
  - Vision-language approaches
  - Audio and speech synthesis
  - Video generation capabilities
  - Future directions

- [ ] Create `ailabs-llm-chinese-open-source.md`
  - Open weight models comparison
  - Licensing comparison (MIT, Apache, etc.)
  - Community adoption metrics
  - Commercial use policies

---

## Phase 4: Index and Navigation

- [ ] Update main `ailabs-llm-chinese-index.md`
  - Add navigation guide to new files
  - Create clear links structure
  - Add executive summary for quick access

- [ ] Create `ailabs-llm-chinese-navigation.md`
  - Quick reference to find specific information
  - By lab index
  - By topic index
  - By date/timeline

---

## File Organization Structure (Target)

```
ai/
├── ailabs-llm-chinese-index.md (main overview)
├── ailabs-llm-chinese-navigation.md (quick links)
│
├── Comparisons/
│   ├── ailabs-llm-chinese-comparison.md (company comparison)
│   ├── ailabs-llm-chinese-comparison-models.md
│   ├── ailabs-llm-chinese-comparison-strategy.md
│   └── ailabs-llm-chinese-comparison-technical.md
│
├── DeepSeek/
│   ├── ailabs-deepseek.md (main file)
│   ├── ailabs-deepseek-models.md
│   ├── ailabs-deepseek-evolution.md
│   └── ailabs-deepseek-analysis.md
│
├── MiniMax/
│   ├── ailabs-minimax.md (main file)
│   ├── ailabs-minimax-models.md
│   ├── ailabs-minimax-evolution.md
│   └── ailabs-minimax-analysis.md
│
├── Moonshot/
│   ├── ailabs-moonshot.md
│   ├── ailabs-moonshot-models.md
│   ├── ailabs-moonshot-evolution.md
│   └── ailabs-moonshot-analysis.md
│
├── Qwen/
│   ├── ailabs-qwen.md
│   ├── ailabs-qwen-models.md
│   ├── ailabs-qwen-evolution.md
│   └── ailabs-qwen-analysis.md
│
├── Zhipu/
│   ├── ailabs-zhipu.md
│   ├── ailabs-zhipu-models.md
│   ├── ailabs-zhipu-evolution.md
│   └── ailabs-zhipu-analysis.md
│
└── Topics/
    ├── ailabs-llm-chinese-moe-efficiency.md
    ├── ailabs-llm-chinese-long-context.md
    ├── ailabs-llm-chinese-multimodal.md
    └── ailabs-llm-chinese-open-source.md
```

---

## Progress Tracking

### Completed (Current)
- ✅ MiniMax comprehensive documentation (company, team, models, technical details)
- ✅ DeepSeek main documentation
- ✅ Moonshot main documentation
- ✅ Qwen main documentation
- ✅ Zhipu main documentation
- ✅ Chinese AI Overview index

### In Progress
- 🔄 None currently

### Not Started
- ⬜ All phase 1, 2, 3, 4 items above

---

## Notes

- Each lab file should be self-contained but cross-reference other files
- Maintain consistent structure across lab documentation
- Include direct links to papers, GitHub repos, APIs, and resources
- Add timestamps for when information was last updated
- Include researcher bios and team information where available
- Highlight open source vs proprietary models clearly
- Add performance benchmark tables for easy comparison

---

## Research Tasks

- [ ] Research 'AI Scientist' - autonomous AI research systems

