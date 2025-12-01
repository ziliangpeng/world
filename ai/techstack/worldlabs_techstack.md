# World Labs - Technology Stack

**Company:** World Labs
**Founded:** 2024 (launched September 2024)
**Focus:** Spatial intelligence AI and Large World Models
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

World Labs operates from **San Francisco headquarters** with a team spanning **AI research, systems engineering, and product design** creating "tight feedback loops between cutting-edge research and products." The company released **Spark**, an **open-source MIT-licensed Gaussian Splatting renderer** that integrates with **Three.js** for browser-based real-time 3D rendering with advanced features including **stochastic rendering** (skipping explicit depth sorting), **Float32 sorting** for higher precision, **SOGS and SPZ v3 format support** for compressed splats, and **visual transition effects** (Magic, Spread, Unroll, Explosion, Flow, Twister, Rain). The web infrastructure supports **3D exports** in Gaussian splats, triangle meshes, collision meshes for physics, and video renders with pixel-accurate camera control. Founded by **Fei-Fei Li** (CEO, Stanford HAI co-director, ImageNet creator, former Google Cloud AI lead), **Justin Johnson** (computer vision researcher), **Christoph Lassner** (graphics specialist), and **Ben Mildenhall** (NeRF co-creator), World Labs raised **$230M+ total funding** at **$1B+ valuation** (September 2024) from **Andreessen Horowitz, NEA, Radical Ventures** (co-leads), with participation from **NVentures (NVIDIA), Adobe Ventures, AMD Ventures, Intel Capital, Samsung Venture Investment, Temasak, Databricks, Salesforce Ventures**, plus angel investors **Eric Schmidt** and **Geoffrey Hinton**. The company reached **unicorn status in four months**.

---

## AI/ML Tech Stack

### Marble - Multimodal World Model for 3D Generation

**What's unique**: World Labs built **Marble**, the first **generative multimodal world model** that creates **spatially cohesive, high-fidelity, persistent 3D worlds from a single image, video, text prompt, or coarse 3D layout**. Unlike traditional 3D generation systems that produce isolated objects or scenes, Marble generates **navigable, controllable, expansive environments** with consistent geometry and physics. The model demonstrates **multimodal flexibility** — accepting single images, multiple reference images from different angles for enhanced precision, descriptive text, or 3D structural guides created through Chisel (AI-native editing tool). Generated worlds are **bigger, more stylistically diverse, and have cleaner 3D geometry** compared to previous systems. Marble enables **iterative refinement** through world expansion (enlarging targeted regions) and composition (combining multiple generated worlds into larger spaces). The system employs **structure-style decoupling** where coarse 3D scenes determine spatial layout while text prompts control visual appearance, providing independent control over geometry and aesthetics. Outputs export in **multiple 3D representations**: Gaussian splats (highest fidelity), triangle meshes, collision meshes for physics simulations, and video renders with precise camera control — enabling integration with game engines, simulation platforms, and creative tools.

### RTFM - Real-Time Frame Model with Spatial Memory

**What makes it different**: World Labs developed **RTFM (Real-Time Frame Model)**, an **autoregressive diffusion transformer** that operates on frame sequences, trained end-to-end on **large-scale video data** to predict the next frame conditioned on previous frames. RTFM functions as a **learned neural renderer** — input images convert to neural network activations (KV cache) representing the world implicitly, then the network generates new views via attention mechanisms **without building explicit 3D geometry**. The innovation: RTFM runs **inference at interactive framerates on a single H100 GPU**, achieving real-time performance through a **spatial memory system** that assigns each frame a pose (position and orientation) in 3D space. This **"context juggling" approach** retrieves nearby frames spatially to form custom contexts for generation, enabling **unbounded persistence** without compute constraints scaling linearly with context length — traditional transformers require ever-growing contexts that become computationally prohibitive. RTFM learns **reflections, shadows, glossy surfaces, and lens flare** end-to-end from training data rather than using hand-engineered graphics algorithms, blurring the distinction between reconstruction and generation as a unified learning problem. The system maintains **persistent worlds indefinitely during interaction**, representing a paradigm shift: viewing **World Models as renderers learned end-to-end from data** rather than traditional graphics pipelines.

### Large World Models (LWMs) - Spatial Intelligence Foundation

**What sets World Labs apart**: World Labs pioneered **Large World Models (LWMs)**, a new class of foundation models that **understand and reason about the 3D world** from images and other modalities. Unlike Large Language Models (text) or Vision-Language Models (2D images with text), LWMs focus on **spatial intelligence** — the ability to perceive, generate, reason about, and interact with three-dimensional environments. The technical thesis: World models that unlock spatial understanding must also **generate simulated worlds of their own**, capable of spawning **endlessly varied and diverse simulated environments** that follow semantic or perceptual instructions while remaining **geometrically, physically, and dynamically consistent** — whether representing real or virtual spaces. Fei-Fei Li's founding vision positions spatial intelligence as **"AI's next frontier"** beyond text and 2D pixels, transforming **"seeing into doing, perceiving into reasoning, and imagining into creating."** LWMs will unlock **new forms of storytelling, creativity, design, simulation, and immersive experiences** across both virtual and physical worlds. This approach fundamentally differs from traditional computer vision (which analyzes 2D projections) or classical robotics (which requires explicit 3D modeling) — LWMs learn spatial structure implicitly through generative modeling on video and multi-view data.

### Spark - Open-Source Gaussian Splatting Renderer

**What's unique**: World Labs released **Spark**, a **high-performance, browser-based Gaussian Splatting renderer for Three.js** under **MIT license**, pushing the boundaries of real-time radiance fields in web browsers. Gaussian Splatting represents 3D scenes as collections of oriented 3D Gaussians with learned colors and opacities, enabling **photorealistic novel view synthesis** at interactive framerates without traditional mesh or voxel representations. Spark introduces **stochastic rendering** that skips explicit depth sorting entirely by using randomness to approximate correct ordering, **dramatically speeding up rendering with minimal fidelity decreases**. The library supports **Float32-based sorting** for more stable, higher precision rendering to address Z-fighting and inconsistencies. Recent versions (v0.1.5-v0.1.9) added **SOGS format support** (reduced file size without quality loss for faster loading) and **SPZ v3 compatibility** (compressed splat pipelines). Version 0.1.9 includes **visual transition and reveal effects** (Magic, Spread, Unroll, Explosion, Flow, Twister, Rain) for creative presentations. Spark is **actively developed by World Labs and the open-source community**, with the GaussianSplats3D community recommending it for advanced features and capabilities. The open-source release democratizes access to cutting-edge 3D rendering technology while enabling World Labs to integrate community contributions into their commercial products.

### Chisel - AI-Native 3D Editing Tool

World Labs introduced **Chisel**, an **AI-native experimental editing mode** for advanced users to create 3D worlds by **laying out coarse structure using 3D shapes** like boxes or planes, or **importing existing 3D assets** as structural guides. Chisel implements **structure-style decoupling** — the coarse 3D scene determines the world's spatial layout and geometry, while text prompts control overall visual style, lighting, and aesthetic properties. This architecture enables **independent control** over spatial structure and appearance, allowing users to define precise geometric constraints (room layouts, building shapes, terrain topology) while delegating surface details to the generative model. Chisel represents a **hybrid approach** between traditional 3D modeling (explicit geometric control) and purely generative AI (emergent structure from prompts). Users can **sketch rough 3D blockouts** that the model refines into photorealistic environments, or **import CAD models** that guide generation while adding realistic materials, lighting, and details. This tool addresses a key limitation of text-to-3D systems: difficulty specifying precise spatial relationships and constraints through language alone. Chisel's integration with Marble enables professional workflows combining manual design with AI generation.

### Team & Research Philosophy

World Labs brings together **"the most formidable slate of pixel talent ever assembled"** across computer vision, graphics, generative AI, and systems engineering. Founder **Fei-Fei Li** created **ImageNet** (large-scale visual database with 14M+ labeled images that catalyzed deep learning breakthroughs), co-authored the **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)** paper cited 25,000+ times, and served as **Chief Scientist of AI/ML at Google Cloud** (2017-2018) where she launched Google Cloud AI and ML products. Co-founder **Ben Mildenhall** co-created **NeRF (Neural Radiance Fields)**, the 2020 breakthrough in novel view synthesis that won **Best Paper at ECCV 2020** and catalyzed the field of neural rendering. **Justin Johnson** created **fast neural style transfer** and **contributed to Mask R-CNN**, foundational computer vision papers. **Christoph Lassner** specialized in human pose estimation and 3D reconstruction. The company operates with a **"shared curiosity, passion, and deep backgrounds in technology"** philosophy, emphasizing tight integration between research and product development rather than separation between research labs and engineering teams. Job openings span **Research Scientists (3D, Generative Modeling), Research Engineers (3D Reconstruction/Rendering, ML/Systems, Model Optimization), Infrastructure/Backend Engineers, and Senior Product Engineers** — all based in **San Francisco HQ** with on-site requirements emphasizing collaboration.

---

## Sources

**World Labs Official**:
- [World Labs Homepage](https://www.worldlabs.ai/)
- [About World Labs](https://www.worldlabs.ai/about)
- [Research & Insights Blog](https://www.worldlabs.ai/blog)

**Technical Blogs**:
- [Marble: A Multimodal World Model](https://www.worldlabs.ai/blog/marble-world-model)
- [RTFM: A Real-Time Frame Model](https://www.worldlabs.ai/blog/rtfm)
- [Generating Bigger and Better Worlds](https://www.worldlabs.ai/blog/bigger-better-worlds)

**Funding & Company**:
- [World Labs Comes Out of Stealth with $230M - TechCrunch](https://techcrunch.com/2024/09/13/fei-fei-lis-world-labs-comes-out-of-stealth-with-230m-in-funding/)
- [Introducing World Labs - Radical Ventures](https://radical.vc/introducing-world-labs/)
- [AI Startup World Labs Launches - Nasdaq](https://www.nasdaq.com/articles/ai-startup-world-labs-launches-230m-andreessen-horowitz-nvidias-venture-arm)
- [From Words to Worlds: Spatial Intelligence - a16z](https://www.a16z.news/p/from-words-to-worlds-spatial-intelligence)

**Founders & Team**:
- [Fei-Fei Li Wikipedia](https://en.wikipedia.org/wiki/Fei-Fei_Li)
- [Aidan Gomez - Wikipedia](https://en.wikipedia.org/wiki/Aidan_Gomez)
- [World Labs Founder Interview - Bloomberg](https://www.bloomberg.com/features/2025-fei-fei-li-weekend-interview/)

**Open Source & Technical**:
- [Spark Releases v0.1.9 - Radiance Fields](https://radiancefields.com/world-labs-releases-spark-v0-1-9)
- [Spark Releases v0.1.8 - Radiance Fields](https://radiancefields.com/spark-releases-v0-1-8-to-3dgs-three-js-renderer)
- [GitHub: GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D)

**Job Postings**:
- [World Labs Careers](https://jobs.ashbyhq.com/worldlabs)
- [World Labs Careers - Levels.fyi](https://www.levels.fyi/companies/world-labs)

---

*Last updated: November 30, 2025*
