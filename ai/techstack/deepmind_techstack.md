# Google DeepMind - Technology Stack

**Company:** Google DeepMind (Alphabet subsidiary)
**Founded:** 2010 (acquired by Google 2014, merged with Google Brain 2023)
**Focus:** Artificial general intelligence research and foundation models
**Headquarters:** London, UK (with research centers in US, Canada, France, Germany, Switzerland)

---

## Non-AI Tech Stack

Google DeepMind is a **British-American AI research laboratory** and **subsidiary of Alphabet Inc.**, formed in **April 2023** through the merger of **DeepMind** (founded 2010) and **Google Brain**. The company was founded by **Demis Hassabis** (CEO, 2024 Nobel Prize in Chemistry), **Shane Legg**, and **Mustafa Suleyman** in London. Google acquired DeepMind in **January 2014** for a reported **$500-650M**. The merged entity is led by **Demis Hassabis** who oversees approximately **5,600-6,600 employees** across 6 continents (up from 2,500 two years ago). Headquarters is at **Kings Cross, London**, with major research centers in the **United States, Canada, France, Germany, and Switzerland**. In **2024**, Hassabis and **John Jumper** (DeepMind Director) were awarded the **Nobel Prize in Chemistry** for AlphaFold's protein structure prediction breakthrough, sharing the prize with David Baker. The latest internal valuation is approximately **$6B** (July 2025), though as an Alphabet subsidiary, DeepMind operates under Google's corporate structure rather than as an independent funded entity. The organizational structure underwent a **2025 reorganization** where DeepMind research now flows into engineering earlier in the development cycle, with foundational models rising upward into Google product groups. DeepMind's infrastructure backbone uses custom **Tensor Processing Units (TPUs)** and a unified **hypercomputer architecture** combining TPUs, GPUs, and interconnects for massive model training and inference.

**Salary Ranges**: Research Engineer $168K-$250K total | Software Engineer $177K-$253K total | Research Scientist $203K-$262K (L3-L8 range $179K-$893K) | Research Intern $72/hr

---

## AI/ML Tech Stack

### TPU Ironwood (v7) - First Inference-Optimized TPU with 42.5 Exaflops per Superpod

**What's unique**: Google DeepMind unveiled **Ironwood** (TPU v7) in 2025 as their **first TPU explicitly designed for large-scale AI inference** rather than training. Each chip delivers **4,614 TFLOPs peak compute** (FP8 precision) with **192 GB HBM3e memory** and **7.2 TBps memory bandwidth**. The architecture uses a **dual-chiplet design** — each chiplet is a self-contained unit with one TensorCore, two SparseCores, and 96 GB HBM, departing from previous unified "MegaCore" designs. Ironwood scales to **9,216 chips per superpod** connected via **9.6 Tb/s Inter-Chip Interconnect (ICI)**, delivering **42.5 Exaflops of compute** with access to **1.77 Petabytes of shared HBM**. The enhanced **SparseCore** accelerates ultra-large embeddings for ranking and recommendation workloads. Ironwood achieves **2x energy efficiency** versus the previous Trillium TPU (v6) and **3,600x inference speed** versus the original 2018 Cloud TPU with **30x power efficiency improvement**. This purpose-built inference architecture enables Google to serve Gemini and other foundation models at scale while competitors face NVIDIA GPU supply constraints.

### Gemini 3 - Natively Multimodal MoE with 1M Token Context Window

**What makes it different**: **Gemini 3** is Google DeepMind's frontier multimodal model family featuring a **refined transformer decoder architecture** optimized for TPU v5p/v6/v7. The architecture incorporates **Mixture-of-Experts (MoE)** that selectively engages expert layers for efficient scaling. Key components include a **multimodal encoder** integrating visual, speech, and text data, and a **cross-modal attention network** linking modalities. Gemini supports **1 million token context windows** enabling analysis of full codebases, comprehensive document processing, and multi-document synthesis. The model family spans **Ultra, Pro, Flash, and Nano** tiers for different performance/cost tradeoffs. Training uses diverse pre-training data encompassing web documents, text, code, images, audio (speech and other), and video — with modalities **interleaved in any order** rather than fixed sequences. Post-training includes **instruction tuning, reinforcement learning, and human preference data**. **Gemini 2.5 Deep Think** was additionally trained on novel RL techniques leveraging multi-step reasoning, problem-solving, and theorem-proving data, achieving **gold-medal standard** at the International Mathematical Olympiad.

### AlphaFold 3 - Nobel Prize-Winning Protein Structure Prediction

**What sets DeepMind apart**: **AlphaFold** solved the 50-year-old protein folding problem, earning Hassabis and Jumper the **2024 Nobel Prize in Chemistry**. AlphaFold 2 (2020) achieved **sub-1 Angstrom median error** at CASP14 — 3x more accurate than the next best system and comparable to experimental methods. The architecture evolved from AlphaFold 1's separately trained modules to AlphaFold 2's **interconnected sub-networks forming a single end-to-end differentiable model**. Training used protein DNA sequences, known structures, and co-evolutionary information indicating which sequences evolve together. **AlphaFold 3** (2024) predicts structures of complexes with DNA, RNA, ligands, and ions, showing **minimum 50% improvement** for protein-molecule interactions versus existing methods. The **AlphaFold Protein Structure Database** provides predictions for over 200 million proteins. **AlphaProteo** extends this to **de novo protein design**, generating novel high-strength protein binders with **3-300x better binding affinities** than existing methods, trained on PDB data plus 100M+ AlphaFold predictions. This represents a fundamental scientific breakthrough with direct drug discovery applications.

### AlphaGeometry 2 & AlphaProof - IMO Silver-Medal Mathematics Reasoning

**What's unique**: **AlphaGeometry** is a **neuro-symbolic system** combining a neural language model with a symbolic deduction engine for geometry theorem proving. Given a problem and premises, the symbolic engine deduces new statements; if exhausted without solution, the language model adds a potentially useful construct opening new deduction paths. Training used **100 million synthetic theorems and proofs** generated by DeepMind rather than human examples. AlphaGeometry solved **25 of 30 Olympiad problems** within time limits (vs. 10 for previous SOTA and 25.9 for average human gold medalists). **AlphaGeometry 2** combined with **AlphaProof** solved **4 of 6 problems** from the 2024 IMO, achieving **silver-medal level** — the first AI system to do so. An advanced Gemini version with **Deep Think** subsequently achieved **gold-medal standard**. The code is open-sourced on GitHub. This demonstrates DeepMind's leadership in formal mathematical reasoning beyond pattern matching.

### Veo 3 - Unified Audio-Visual Generation with Native Sound Synthesis

**What makes it different**: **Veo 3** uses an advanced **3D latent diffusion architecture** treating video generation as a spatiotemporal problem (height, width, time) rather than 2D images extended with time. The architecture uses **3D Convolutional Layers within a U-Net** processing spatiotemporal data across channels, time, height, and width simultaneously. Built on a **Latent Diffusion Transformer**, it processes video and audio in compressed latent space rather than raw pixels/soundwaves. Text prompts are processed by a **UL2 Encoder**, images by an image encoder, combined into embedded prompts for the diffusion model. The breakthrough is **native, single-pass generation of synchronized audio** — dialogue, sound effects, and ambient noise — alongside visual output. Production capabilities include **1080p at 24fps** in landscape (16:9) and portrait (9:16), with internal research achieving **4K resolution** and durations beyond one minute. Veo builds on previous DeepMind advances: GQN, DVD-GAN, Imagen-Video, Phenaki, WALT, VideoPoet, and Lumiere.

### JAX Ecosystem - Composable Transformations for TPU-Native Training

**What sets DeepMind apart**: DeepMind standardized on **JAX** as their primary ML framework — a Python library for hardware accelerator-oriented array computation with composable function transformations. JAX provides **automatic differentiation** (grad, hessian, jacfwd, jacrev), **automatic vectorization** (vmap), and **data parallelism** (pmap) that elegantly distributes data across accelerators. DeepMind built an ecosystem of JAX libraries:

- **Haiku** — neural network library
- **Optax** — gradient processing and optimization
- **RLax** — reinforcement learning building blocks (TD-learning, policy gradients, actor-critics, PPO, exploration)
- **Jraph** — graph neural networks
- **Chex** — testing utilities

The composable transformation model enables training large-scale models with **data and model parallelism without custom distributed code**. JAX's design aligns naturally with TPU architecture, providing performance advantages over competitors using PyTorch on NVIDIA GPUs. **JAX-Privacy** enables differentially private training on foundation models at scale.

### WARP & SCoRe - Advanced RLHF for Model Alignment

**What's unique**: DeepMind developed **WARP (Weight Averaged Rewarded Policies)** to optimize the KL-reward Pareto front in RLHF alignment. WARP uses three types of model merging: **exponential moving average** of policy in KL regularization, **spherical interpolation** to merge fine-tuned policies, and **linear interpolation** between merged model and initialization to preserve pre-training features. **SCoRe** is a multi-turn online RL approach that improves LLM self-correction using entirely self-generated data. DeepMind's RLHF infrastructure uses **70B Chinchilla models** for both LM and reward models, with research extending to 280B parameter Gopher. Gemini 2.5 Deep Think used **novel RL techniques** leveraging multi-step reasoning and theorem-proving data. **Reinforcement Learning Fine-Tuning (RLFT)** uses self-generated Chain-of-Thought rationales as training signals, evaluating rewards of actions following specific reasoning steps. This advanced alignment research enables DeepMind to produce models that reason more reliably than pure supervised fine-tuning approaches.

### Hypercomputer Architecture - Co-Designed Hardware/Software Stack

**What makes it different**: DeepMind operates on Google's **hypercomputer architecture** — a unified system combining TPUs, GPUs, interconnects, and software designed together rather than assembled from commodity components. The approach uses **co-design** where chips, networks, cooling systems, and software are optimized together from the ground up through collaboration between hardware engineers, software developers, and data center teams. TPUs are trained on Google's proprietary silicon, avoiding NVIDIA supply constraints faced by competitors. The **layered organizational structure** has foundational models rising from DeepMind research into Google product groups earlier in the development cycle. Google possesses unique advantages: **AI infrastructure spanning TPUs, global data centers, product distribution (Search, Android), mature safety systems, and massive invocation gateways**. The enterprise engine includes **Vertex AI, feature stores, MLOps pipelines, model registry, monitoring, and governance frameworks** for production deployment.

---

## Sources

**Google DeepMind Official**:

- [Google DeepMind Homepage](https://deepmind.google/)
- [About Google DeepMind](https://deepmind.google/about/)
- [Gemini Models](https://deepmind.google/models/gemini/)
- [AlphaFold](https://deepmind.google/science/alphafold/)
- [AlphaFold Database](https://alphafold.ebi.ac.uk/)
- [AlphaProteo Announcement](https://deepmind.google/discover/blog/alphaproteo-generates-novel-proteins-for-biology-and-health-research/)
- [AlphaGeometry Blog](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/)
- [AlphaGeometry GitHub](https://github.com/google-deepmind/alphageometry)
- [AI Solves IMO at Silver Medal Level](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)
- [Gemini Deep Think Gold Medal IMO](https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/)
- [Veo Models](https://deepmind.google/models/veo/)
- [Nobel Prize Announcement](https://deepmind.google/discover/blog/demis-hassabis-john-jumper-awarded-nobel-prize-in-chemistry/)
- [Using JAX to Accelerate Research](https://deepmind.google/discover/blog/using-jax-to-accelerate-our-research/)
- [Careers](https://deepmind.google/careers/)

**TPU & Infrastructure**:

- [Ironwood TPU Announcement](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/)
- [3 Things to Know About Ironwood](https://blog.google/products/google-cloud/ironwood-google-tpu-things-to-know/)
- [Inside the Ironwood TPU Codesigned Stack](https://cloud.google.com/blog/products/compute/inside-the-ironwood-tpu-codesigned-ai-stack/)
- [TPU7x Documentation](https://docs.cloud.google.com/tpu/docs/tpu7x)
- [Ironwood at Hot Chips 2025 - ServeTheHome](https://www.servethehome.com/google-ironwood-tpu-swings-for-reasoning-model-leadership-at-hot-chips-2025/)
- [TPUv7 Analysis - SemiAnalysis](https://newsletter.semianalysis.com/p/tpuv7-google-takes-a-swing-at-the)

**Model Cards & Technical Papers**:

- [Gemini 2.5 Pro Model Card](https://modelcards.withgoogle.com/assets/documents/gemini-2.5-pro.pdf)
- [Gemini 2.5 Deep Think Model Card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-2-5-Deep-Think-Model-Card.pdf)
- [Gemini 3 Pro Model Card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf)
- [Gemini Technical Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf)
- [AlphaGeometry 2 Paper - arXiv](https://arxiv.org/html/2502.03544v1)

**Company & Leadership**:

- [Google DeepMind Wikipedia](https://en.wikipedia.org/wiki/Google_DeepMind)
- [Demis Hassabis Wikipedia](https://en.wikipedia.org/wiki/Demis_Hassabis)
- [Demis Hassabis - Britannica](https://www.britannica.com/biography/Demis-Hassabis)
- [Google DeepMind Crunchbase](https://www.crunchbase.com/organization/deepmind)
- [Google DeepMind PitchBook](https://pitchbook.com/profiles/company/60887-17)
- [Google DeepMind LinkedIn](https://www.linkedin.com/company/googledeepmind)

**News & Analysis**:

- [DeepMind Rose From Fringe Research to Engine Room](https://techtrendske.co.ke/2025/11/28/deepmind-google-machine-intelligence-integration/)
- [How Google Pulled Off Its AI Turnaround](https://www.bigtechnology.com/p/how-google-pulled-off-its-stunning)
- [Google CTO Reveals Gemini Strategy - DigiTimes](https://www.digitimes.com/news/a20251202PD207/google-gemini-deepmind-infrastructure-training.html)
- [AlphaFold Five Years of Impact - Fortune](https://fortune.com/2025/11/28/google-deepmind-alphafold-science-ai-killer-app/)
- [Hassabis and Jumper Win Nobel - Fortune](https://fortune.com/2024/10/09/google-deepmind-leaders-hassabis-and-jumper-win-nobel-prize-for-chemistry/)
- [Veo 3 Technical Analysis - Medium](https://medium.com/google-cloud/deconstructing-veo-3-a-technical-analysis-of-googles-unified-audio-visual-generation-model-6be023888489)

**RLHF & Training Techniques**:

- [WARP Method - MarkTechPost](https://www.marktechpost.com/2024/06/29/google-deepmind-introduces-warp-a-novel-reinforcement-learning-from-human-feedback-rlhf-method-to-align-llms-and-optimize-the-kl-reward-pareto-front-of-solutions/)
- [SCoRe Self-Correction via RL - arXiv](https://arxiv.org/pdf/2409.12917)
- [RLFT Knowing-Doing Gap - MarkTechPost](https://www.marktechpost.com/2025/05/18/llms-struggle-to-act-on-what-they-know-google-deepmind-researchers-use-reinforcement-learning-fine-tuning-to-bridge-the-knowing-doing-gap/)
- [Learning Through Human Feedback](https://deepmind.google/discover/blog/learning-through-human-feedback/)

**Compensation**:

- [Google DeepMind Salaries - Glassdoor](https://www.glassdoor.com/Salary/Google-DeepMind-Salaries-E1596815.htm)
- [Google Research Scientist Salaries - Levels.fyi](https://www.levels.fyi/companies/google/salaries/software-engineer/title/research-scientist)
- [DeepMind Research Intern - Levels.fyi](https://www.levels.fyi/internships/DeepMind/Research-Intern/)

---

*Last updated: December 6, 2025*
