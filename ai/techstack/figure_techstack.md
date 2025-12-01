# Figure AI - Technology Stack

**Company:** Figure AI, Inc.
**Founded:** 2022
**Focus:** General-purpose humanoid robots
**Headquarters:** Sunnyvale, California

---

## Non-AI Tech Stack

Figure AI operates a hybrid cloud infrastructure using **Microsoft Azure** for large-scale AI model training via **ND H100 GPU clusters** and on-premise **BotQ manufacturing facility** capable of producing **12,000 humanoids annually**. The manufacturing software stack includes custom-built **MES (Manufacturing Execution System)**, **Dassault Systèmes Enovia** (PLM), **Oracle NetSuite** (ERP), and integrated **WMS (Warehouse Management System)** for tracking inventory, production flow, genealogy, and real-time IoT device monitoring. Figure designed the entire robot hardware from scratch: actuators, motors, sensors, battery pack, electronics, achieving **35 degrees of freedom** including human-like wrists, hands, and individual fingers. The **Figure 02** platform includes **6 RGB cameras**, dual low-power embedded GPUs per robot, and perception systems. Development tools leverage **NVIDIA Isaac Sim** (built on Omniverse) for synthetic data generation, enabling rapid iteration on perception, mobility, and manipulation models. The BotQ facility will use humanoid robots to assemble other humanoids (robots building robots) starting this year, with supply chain scalable to **100,000 robots or 3 million actuators** in four years. Figure requires 5 days/week in-office collaboration at their San Jose, CA headquarters.

**Salary Ranges**: Software Engineer roles ~$175K-$350K | General robotics AI specialists $150K+ | Experienced engineers in high-demand sectors earn considerably more

---

## AI/ML Tech Stack

### Helix VLA Model - Dual-System Architecture

**What's unique**: Figure developed **Helix**, a proprietary Vision-Language-Action (VLA) model with a novel **dual-system architecture** after ending their OpenAI partnership in 2025 (stating LLMs became "commoditized"). Helix consists of **System 2 (S2)**: a **7-billion parameter vision-language model** operating at **7-9 Hz** for high-level scene understanding, language comprehension, and behavioral planning; and **System 1 (S1)**: an **80-million parameter visuomotor policy** running at **200 Hz** for precise, reactive low-level control. This is the **first VLA to output high-rate continuous control of the entire humanoid upper body** (wrists, torso, head, individual fingers) and the **first VLA to operate simultaneously on two robots** solving shared long-horizon manipulation tasks with never-before-seen objects. The dual-system design mirrors human cognition: S2 thinks slowly for complex reasoning while S1 reacts instantly for physical execution.

### On-Robot GPU Deployment & Training

Each Figure robot contains **two low-power embedded GPUs**, with the inference pipeline split across dedicated hardware: S2 (high-level latent planning) runs on one GPU, S1 (low-level control) on the other, maintaining the critical **200Hz control loop** for smooth whole-body actions. During training, a **temporal offset** is added between S1 and S2 inputs, calibrated to match deployment inference latency, ensuring real-time control requirements are reflected in training. Figure trained Helix on **~500 hours of teleoperated robot behaviors**, then used **auto-labeling** to generate natural language instructions for each demonstration. Training infrastructure leverages **Microsoft Azure ND H100 GPU clusters**, processing the initial **20TB Helix training dataset**. Model parallel deployment enables efficient inference on resource-constrained embedded GPUs.

### Synthetic Data Pipeline with NVIDIA Isaac Sim

Figure accelerated development using **NVIDIA Isaac Sim** (Omniverse-based reference application) to design, train, and test AI models with **synthetic data**. The **Figure 02** perception AI models are trained using synthetic datasets generated in Isaac Sim, employing **domain randomization** techniques to create diverse scenarios (one demonstration generated **90,000+ images**). Isaac Sim facilitates three workflows: synthetic data generation for perception/mobility/manipulation, simulation-based validation, and digital twin testing. This approach enables rapid iteration without requiring massive real-world data collection. NVIDIA's technology stack for Figure includes: **NVIDIA DGX** (AI training), **NVIDIA Omniverse** (simulation), **NVIDIA Jetson** (on-robot compute), and **NVIDIA Isaac GR00T** foundation model support. Figure is an initial member of the **NVIDIA Humanoid Robot Developer Program**, gaining early access to cutting-edge tools.

### Real-World Deployment & Continuous Learning

Figure deployed **Figure 02** to **BMW Group's Spartanburg, South Carolina production line** for real-world data collection and use-case training, retiring the initial deployment after 11 months of work. The robots perform **high-precision pick-and-place tasks** required for smart manufacturing using perception models trained with Isaac Sim synthetic data. New hardware features (human-scale hands, 6 RGB cameras) enable complex manipulation. The continuous learning loop: synthetic pre-training → real-world deployment → teleoperated data collection → model refinement → redeployment represents Figure's unique approach to achieving generalist humanoid capabilities.

### Strategic Pivot from OpenAI

In February 2024, Figure secured **$675M funding** from Jeff Bezos, Microsoft, NVIDIA, Intel, Amazon, and OpenAI's startup division, announcing an OpenAI partnership for specialized robot AI models. However, Figure **ended the collaboration in 2025**, determining that large language models were "getting smarter yet more commoditized." This strategic decision led to building Helix in-house, differentiating Figure through proprietary VLA technology optimized for embodied control rather than relying on third-party foundation models. This pivot demonstrates Figure's commitment to owning their core AI stack.

### Manufacturing Automation Loop

Figure's unique "robots building robots" strategy closes the loop: Helix-powered humanoids will perform assembly and material handling in the BotQ facility, using the same AI that enables commercial deployments for internal manufacturing. This self-sustaining approach validates the technology's generalist capabilities while scaling production. The MES software integrates with IoT devices for real-time monitoring, ensuring quality control as humanoids handle repetitive assembly tasks.

**Salary Ranges**: AI Engineer Post-Training (Helix Team) $175K-$350K | ML/Robotics Engineers $150K-$200K+ | Senior roles with autonomous vehicle/advanced manufacturing experience earn considerably more

---

## Sources

**Figure AI Technical Content**:
- [Helix: A Vision-Language-Action Model for Generalist Humanoid Control](https://www.figure.ai/news/helix)
- [BotQ: A High-Volume Manufacturing Facility for Humanoid Robots](https://www.figure.ai/news/botq)
- [Master Plan - Figure AI](https://www.figure.ai/master-plan)

**NVIDIA Partnership & Technology**:
- [Figure Unveils Next-Gen Conversational Humanoid Robot - NVIDIA Blog](https://blogs.nvidia.com/blog/figure-humanoid-robot-autonomous/)
- [Build Synthetic Data Pipelines with NVIDIA Isaac Sim](https://developer.nvidia.com/blog/build-synthetic-data-pipelines-to-train-smarter-robots-with-nvidia-isaac-sim/)

**Analysis & Research**:
- [Figure Business Breakdown & Founding Story - Contrary Research](https://research.contrary.com/company/figure)
- [Inside Figure AI: How this startup is reshaping humanoid robots](https://roboticsandautomationnews.com/2025/05/06/spotlight-on-humanoids-a-deep-dive-into-figure-ai/90373/)
- [Figure's Helix: AI that Brings Human-Like Robots - Analytics Vidhya](https://www.analyticsvidhya.com/blog/2025/02/figures-helix/)

**Manufacturing & Deployment**:
- [Figure AI unveils BotQ facility - The Robot Report](https://www.therobotreport.com/figure-ai-unveils-botq-high-volume-humanoid-manufacturing-facility/)
- [Figure humanoid robots retire after 11 months at BMW](https://interestingengineering.com/ai-robotics/figure-humanoid-robots-retires-bmw)

**AI/ML Job Postings**:
- [AI Engineer, Post-Training - Helix Team](https://job-boards.greenhouse.io/figureai/jobs/4443109006) - $175K-$350K
- [Figure AI Careers](https://www.figure.ai/careers)
- [Figure AI Jobs - Built In](https://builtin.com/company/figureai/jobs)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/figure-ai)

---

*Last updated: November 30, 2025*
