# Runway - Technology Stack

**Company:** Runway AI, Inc.
**Founded:** 2018
**Focus:** Generative AI for video and image creation
**Headquarters:** New York City

---

## Non-AI Tech Stack

Runway operates offices in **New York, San Francisco, Seattle, London, and Tel Aviv** with **remote-first** operations across North America and Europe. The frontend infrastructure uses **TypeScript, React/Redux, WebGL2, and WebAssembly** to build interfaces for generative AI tools with **real-time compositing capabilities**. Backend **serverless APIs** written in **TypeScript/Node.js** deploy on **AWS Lambda** for real-time collaboration and media asset management. The ML infrastructure is built on **Ray and Anyscale**, integrated with **Kubernetes** cluster management via **Anyscale Operator for Kubernetes** and **Anyscale on GKE**. The platform uses **Grafana** for log integration and observability. **Ray Serve** offloads preprocessing to dedicated compute pools separate from GPU training resources, preventing GPU memory contention during model training while enabling rapid data iteration. Anyscale enables **sub-60-second cluster launches**, reliable **spot instance management**, and automatic failure recovery. Founded by **Cristóbal Valenzuela** (CEO), **Alejandro Matamala**, and **Anastasis Germanidis** (CTO) who met at NYU Tisch School of the Arts, Runway raised **$544M total** ($308M Series D April 2025 at $3B valuation from Fidelity/Baillie Gifford; $141M Series C extension June 2023 at $1.5B valuation from Google, NVIDIA, Salesforce). The company offers **unlimited PTO** (15 days minimum recommended), **paid sabbaticals** after 3/5/7 years, and **100% medical/dental/vision coverage** for US employees.

**Salary Ranges**: Software Engineer highest $332K total comp (median £252K UK, up to £300K UK)

---

## AI/ML Tech Stack

### Gen-3 Alpha - Large-Scale Multimodal Training Infrastructure

**What's unique**: Runway built **"a new infrastructure built for large-scale multimodal training"** to create Gen-3 Alpha, their most advanced video generation model (launched summer 2024). The model was **trained jointly on videos and images** using **highly descriptive, temporally dense captions** that enable precise control over temporal sequences and scene elements. Gen-3 Alpha represents **"a major improvement in fidelity, consistency, and motion"** over Gen-2, demonstrating progress toward building **General World Models**. The model **"excels at generating expressive human characters with a wide range of actions, gestures, and emotions,"** supporting diverse cinematic terminology and artistic styles. Core capabilities span **Text to Video, Image to Video, and Text to Image** generation, with tools including **Motion Brush, Advanced Camera Controls, and Director Mode**. The development involved collaborative efforts across **research scientists, engineers, and artists** working across disciplines. Gen-3 Alpha includes **"a new set of safeguards"** with an improved in-house visual moderation system and **C2PA provenance standards** compliance for content authenticity.

### Act-One - Facial Performance Capture Without Motion Capture Equipment

Runway released **Act-One** in October 2024, a **state-of-the-art tool for generating expressive character performances** inside Gen-3 Alpha. The innovation: **"a simple at-home camera setup captures an actor's driving performance to animate a generated character"** without requiring motion capture equipment or manual face rigging. Users record themselves on **consumer-grade cameras (even smartphones)**, capturing facial expressions to replicate on AI-generated video characters. **Eye-lines, micro-expressions, pacing, and delivery are all faithfully represented** in the final generated output. The system uses **"a completely different pipeline, driven directly and only by a performance of an actor"** with no extra equipment. This enables narrative content creation using **"nothing more than a consumer grade camera and one actor reading and performing different characters from a script."** Act-One includes **safeguards to detect and block attempts to generate content featuring public figures without authorization**, addressing deepfake concerns. Industry professionals called it a **"game changer"** for AI filmmaking and animation workflows.

### Ray & Anyscale Infrastructure - 13x Faster Model Loading

**What makes it different**: Runway selected **Ray** as their foundation for scaling compute after prototyping within one day demonstrated the framework's accessibility. An engineer noted: **"Using Ray was a really straightforward decision. It's hard to even compare Ray to anything."** While KubeRay initially worked, scaling to 4-5 concurrent researchers created severe observability problems nearly requiring a dedicated infrastructure engineer. Switching to **Anyscale** solved these limitations through **enterprise-level observability and governance**, **out-of-the-box autoscaling** (clusters launch in under 60 seconds), **reliable spot instance management**, and **seamless ephemeral cluster handling**. The quantifiable results: **13x faster model loading**, **85% reduction in pipeline development time** (week to one day), and **40-50 Runway engineers using Anyscale**. Runway uses **Ray Serve** to offload preprocessing tasks to dedicated compute pools separate from GPU training resources, preventing GPU memory contention while enabling rapid data iteration. Anyscale partnered with Runway to develop **Anyscale Operator for Kubernetes** and **Anyscale on GKE** integration, enabling secure enterprise-grade Ray deployment within Runway's existing Kubernetes infrastructure.

### Model Evolution - Gen-1, Gen-2, Gen-3, Gen-4

**Gen-1** (Feb 2023): Runway published **"Structure and Content-Guided Video Synthesis with Diffusion Models"** (arXiv 2302.03011, ICCV 2023), using **structure and content-guided diffusion** for video-to-video generation. Training on monocular depth estimates provided control over structure and content fidelity. The model was **trained jointly on images and videos**, exposing explicit control of temporal consistency through novel guidance methods. Gen-1 used **input video as conditioning** to determine output video structure, solving temporal consistency issues that cause flicker when generating videos frame-by-frame.

**Gen-2** (June 2023): Trained on an internal dataset of **240 million images and 6.4 million video clips**, Gen-2 **removed the need for structure conditioning** and tackled **text-guided video generation directly**. The model supported **text-to-video, image-to-video, and video-to-video** generation. **Notably, Gen-2 does not have a dedicated standalone research paper** published in traditional academic venues — Runway shifted focus to proprietary development.

**Gen-3 Alpha** (Summer 2024): Represents a **step toward General World Models** with major improvements in fidelity, consistency, and motion.

**Gen-4**: **"Excels in its ability to generate highly dynamic videos with realistic motion as well as subject, object and style consistency with superior prompt adherence and best in class world understanding."**

### Research Contributions - Co-Authors of Latent Diffusion

Runway is **co-author of the Latent Diffusion paper** (collaboration with LMU Munich) that **gave birth to Stable Diffusion**, one of the most influential generative AI breakthroughs. This contribution established Runway's research credibility in the generative AI community. The company's research focuses on **multimodal AI systems** for computer vision, generative audio, language models, reinforcement learning, and model scaling. Applied research teams pioneer new forms of creativity through **human-in-the-loop systems** for creative workflows, dataset engineering, and data acquisition optimization supporting large-scale AI model training.

### Engineering for Creative Workflows

Runway's technical organization emphasizes **creative, open-minded, caring, and ambitious people** dedicated to building **"impossible things"** and empowering **"everyone to create content without barriers to entry."** Frontend teams build interfaces with **real-time compositing capabilities** using WebGL2 and WebAssembly for performant browser-based video editing. Backend serverless APIs handle **real-time collaboration** across distributed teams. ML research teams focus on **computer vision, generative audio, language models, reinforcement learning**, and **scaling multimodal systems**. Dataset engineering represents a **strategic organizational priority** for acquiring and optimizing training data at scale. The company commits to diversity, welcoming talent **"regardless of race, gender identity or expression, sexual orientation, religion, origin, ability, age, veteran status."**

**Salary Ranges**: Software Engineer highest $332K total comp | UK median £252K (up to £300K)

---

## Sources

**Runway Research & Models**:
- [Introducing Gen-3 Alpha: A New Frontier for Video Generation](https://runwayml.com/research/introducing-gen-3-alpha)
- [Introducing Runway Gen-4](https://runwayml.com/research/introducing-runway-gen-4)
- [Gen-2: Generate Novel Videos with Text, Images or Video Clips](https://runwayml.com/research/gen-2)
- [Gen-1: The Next Step Forward for Generative AI](https://runwayml.com/research/gen-1)
- [Introducing Act-One](https://runwayml.com/research/introducing-act-one)
- [Scale, Speed and Stepping Stones: The Path to Gen-2](https://runwayml.com/research/scale-speed-and-stepping-stones-the-path-to-gen-2)
- [Runway Research Publications](https://runwayml.com/research/publications)

**Research Papers**:
- [Structure and Content-Guided Video Synthesis with Diffusion Models (arXiv 2302.03011)](https://arxiv.org/abs/2302.03011)
- [Structure and Content-Guided Video Synthesis - ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Esser_Structure_and_Content-Guided_Video_Synthesis_with_Diffusion_Models_ICCV_2023_paper.pdf)

**Infrastructure & Engineering**:
- [How Runway Powers AI Media Creation with Anyscale](https://www.anyscale.com/resources/case-study/runway)

**Company & Funding**:
- [Runway Careers](https://runwayml.com/careers)
- [Runway Wikipedia](https://en.wikipedia.org/wiki/Runway_(company))
- [Runway: 2025 CNBC Disruptor 50](https://www.cnbc.com/2025/06/10/runway-cnbc-disruptor-50.html)

**Job Postings & Compensation**:
- [Runway Jobs - Greenhouse](https://job-boards.greenhouse.io/runwayml)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/runway/salaries)

---

*Last updated: November 30, 2025*
