# 1X Technologies - Technology Stack

**Company:** 1X Technologies (formerly Halodi Robotics)
**Founded:** 2014 (rebranded 2022)
**Focus:** Home humanoid robots with embodied AI
**Headquarters:** Palo Alto, California (manufacturing: Hayward, CA and Moss, Norway)

---

## Non-AI Tech Stack

1X Technologies is a **Norwegian-American robotics company** founded in 2014 by **Bernt Øivind Børnich** (CEO, Norwegian roboticist), rebranded from Halodi Robotics in March 2022. The company operates **manufacturing facilities in Hayward, California and Moss, Norway** with offices in Palo Alto and Oslo. The infrastructure supports **teleoperation systems** enabling data collectors to remotely control robots for training data generation, with neural networks trained on **thousands of hours of video and actuator data from company's own robots** rather than simulation. 1X produces two robot platforms: **EVE** (wheeled humanoid for industrial/institutional logistics, security, healthcare) and **NEO** (bipedal humanoid for home use, priced at **$20,000 or $499/month subscription**, delivery starting 2026). The **NEO Gamma** (unveiled February 2025) features **40 degrees of freedom**, **tendon drive actuation**, **4 microphones with beamforming**, **3-speaker system**, and **3D printed nylon exterior**, standing **175cm tall** and weighing **65kg**. The company raised **$126M total** (**$100M Series B** January 2024 at **$500M+ valuation** led by EQT Ventures with participation from Sandvik, Nistad Group, and Samsung; **$23.5M Series A2** March 2023 led by **OpenAI Startup Fund** with Tiger Global and Skagerak Capital). 1X achieved **10x hardware reliability improvement** and **10 dB noise reduction** in NEO Gamma compared to Beta 2.

---

## AI/ML Tech Stack

### Redwood - Vision-Language Transformer for Home Environments

**What's unique**: 1X developed **Redwood**, a **vision-language-action (VLA) transformer model** specifically trained for **home environment tasks** that enables NEO to perform **end-to-end household chores** from raw sensory input. Unlike general-purpose foundation models, Redwood is trained on data collected from **1X's own fleet of teleoperated robots** performing real-world tasks in diverse home settings. The model processes visual input and language instructions to directly output motor commands, eliminating the need for traditional robotics programming with inverse kinematics, motion planning, or manual control systems. Redwood represents 1X's approach to **embodied AI** — intelligence that emerges from physical interaction with the real world rather than purely digital training. The model enables NEO to understand natural language commands like "clean the kitchen" and autonomously execute the full sequence of actions required, adapting to variations in home layouts, object positions, and environmental conditions.

### 1XWM World Model - End-to-End Neural Networks Trained on Raw Sensor Data

**What makes it different**: 1X's **1XWM (1X World Model)** is an **end-to-end neural network** trained directly on **raw sensor data** (camera feeds, microphone arrays, actuator positions) from teleoperated robots, bypassing traditional robotics approaches entirely. The model learns to **predict the actions human teleoperators would take** given sensory inputs, effectively learning **human-like decision-making patterns** from demonstration data. This architecture fundamentally differs from classical robotics systems that rely on hand-engineered perception pipelines, symbolic planning, and control theory. Training data comes from **thousands of hours of video and actuator logs** captured as data collectors (employees and external contractors) remotely pilot EVE and NEO robots through real-world tasks. The World Model approach enables **generalization to novel situations** not explicitly programmed, as the neural network learns underlying patterns of how humans navigate physical spaces, manipulate objects, and respond to environmental changes. 1X's thesis: **"the best way to create intelligent robots is to collect data from real-world robot operation and train neural networks on that data"** — rejecting simulation-heavy approaches.

### Data Collectors Train Capabilities Themselves - First-Principles Data Generation

**What sets 1X apart**: 1X pioneered a **novel organizational structure** where **data collectors (teleoperators) directly train the AI capabilities** they later deploy. The company employs data collectors who remotely control EVE and NEO robots to perform tasks in real environments (warehouses, offices, homes), with every action captured as training data. This approach differs fundamentally from traditional robotics companies that separate data collection teams from AI research teams. According to 1X's engineering blog, **"data collectors themselves are training the capabilities they will later use"** — creating a tight feedback loop between data quality and model performance. The teleoperation interface provides **low-latency control** over robot movements while automatically logging synchronized sensor streams and actuator commands. This strategy addresses the **sim-to-real gap** (discrepancy between simulated and real-world performance) by training exclusively on real-world data, though at higher cost per training example. Job postings emphasize hiring **Data Operators** who "collect training data by teleoperating robots in different environments," making data generation a core operational function rather than a research activity.

### NEO Gamma Hardware - Tendon Drive Actuation for Safe Home Deployment

**What's unique**: The **NEO Gamma** robot uses **tendon drive actuation** rather than traditional rigid gear-based actuators, providing **inherent compliance and safety** critical for operation in human homes. Tendon drives work like biological muscles — motors pull cables routed through the robot's skeleton, allowing joints to **absorb impacts and adapt to contact forces** without rigid mechanical resistance. This design choice enables NEO to **safely interact with humans, furniture, and delicate objects** without complex force-sensing feedback loops. The robot's **40 degrees of freedom** (significantly higher than most humanoid prototypes) provide **human-like dexterity** for manipulation tasks including grasping irregularly shaped objects, opening doors, and operating household appliances. NEO Gamma features **soft safety covers** over mechanical components to prevent pinch points. The **audio system** (4 microphones with beamforming algorithms, 3-speaker array) enables **natural voice interaction** and spatial audio awareness for navigation. The **3D printed nylon exterior** reduces manufacturing costs while allowing rapid design iteration — 1X achieved **10x reliability improvement** (mean time between failures) and **10 dB noise reduction** compared to earlier Beta 2 prototypes. The tendon drive architecture pairs naturally with 1X's neural network approach, as compliance simplifies control compared to stiff actuators requiring precise torque regulation.

### NVIDIA GR00T N1 Collaboration - Foundation Model Integration

1X partnered with **NVIDIA** to integrate the **GR00T N1 foundation model** into NEO robots, leveraging NVIDIA's **GEAR (Generalist Embodied Agent Research)** team's work on **multimodal foundation models for robotics**. GR00T (Generalist Robot 00 Technology) is designed to enable robots to **learn from human demonstrations** and **generalize across tasks** using transformer architectures similar to large language models. The collaboration provides 1X access to NVIDIA's research in **vision-language-action models**, **sim-to-real transfer techniques**, and **large-scale robot learning infrastructure**. NVIDIA's Isaac Sim platform may support 1X's training pipelines, though 1X's emphasis on real-world data collection suggests GR00T primarily serves as a **foundation model** that 1X fine-tunes with proprietary teleoperation data. This partnership positions NEO to benefit from NVIDIA's broader robotics ecosystem while maintaining 1X's differentiated approach to embodied AI training. Job postings mention collaboration with NVIDIA's robotics research division, indicating ongoing technical integration beyond a simple vendor relationship.

### Real-World Deployment Timeline - Consumer Home Robots by 2026

1X commits to **delivering NEO to consumers starting in 2026** with a **$20,000 purchase price** or **$499/month subscription model**, representing the **first attempt to commercialize bipedal home humanoid robots at consumer price points**. The pricing strategy targets **affordability comparable to a used car** rather than industrial robotics ($100K+ for competitors like Boston Dynamics). The **subscription model** ($499/month = ~$6,000/year) enables access without upfront capital while allowing 1X to **continuously update AI capabilities** through over-the-air software updates — analogous to Tesla's approach with Autopilot. This business model requires **high reliability** (NEO must operate unsupervised in homes) and **continuous AI improvement** (justifying ongoing subscription fees). 1X's current fleet of EVE robots operates in **ADT security facilities and logistics environments**, providing production deployment experience before NEO's consumer launch. The company's **manufacturing facilities in Hayward and Moss** are scaling production capacity to meet 2026 delivery timelines, with job postings emphasizing hiring for **Manufacturing Engineers** and **Production Operations** roles alongside AI research positions.

---

## Sources

**1X Company & Products**:
- [1X Technologies Official Website](https://www.1x.tech/)
- [NEO: Your Humanoid Companion](https://www.1x.tech/discover/neo)
- [EVE: Safe and Intelligent Android](https://www.1x.tech/discover/eve)
- [1X Wikipedia](https://en.wikipedia.org/wiki/1X_Technologies)

**Funding & Partnerships**:
- [OpenAI's Startup Fund Invests in 1X](https://openai.com/index/1x/)
- [1X Raises $100M Series B Led by EQT Ventures](https://www.eqtgroup.com/news/eqt-ventures-leads-usd-100-million-series-b-in-1x/)
- [NVIDIA GR00T Announcement](https://blogs.nvidia.com/blog/gr00t-humanoid-robot-model/)

**Technical Blogs & Research**:
- [1X Blog: Building the Future of Work](https://www.1x.tech/discover/blog)
- [Introducing NEO Gamma](https://www.1x.tech/discover/neo-gamma)

**Job Postings**:
- [1X Careers](https://www.1x.tech/company/careers)
- [AI Jobs at 1X](https://jobs.ashbyhq.com/1x/collections/engineering---ai)
- [Hardware Engineering Jobs](https://jobs.ashbyhq.com/1x/collections/engineering---hardware)

---

*Last updated: November 30, 2025*
