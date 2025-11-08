# Robotics and Embodied AI Foundation Models Research

This document tracks comprehensive research on robotics foundation models, embodied AI systems, and Vision-Language-Action (VLA) models from companies and research labs worldwide.

**Last Updated:** 2025-11-07

**Related:** See `llm-overview.md` for LLM models, `non-llm-overview.md` for other modalities.

---

## Table of Contents

1. [Commercial Humanoid Robots](#commercial-humanoid-robots)
2. [Research Labs & Institutions](#research-labs--institutions)
3. [Quadruped Robots](#quadruped-robots)
4. [Foundation Model Frameworks](#foundation-model-frameworks)
5. [Open Source Projects](#open-source-projects)
6. [Autonomous Driving (Embodied AI)](#autonomous-driving-embodied-ai)
7. [Key Metrics & Comparisons](#key-metrics--comparisons)

---

## Commercial Humanoid Robots

### United States

#### Tesla (Optimus)
- **Country:** United States
- **Founded:** 2003 (company), Optimus project ~2021
- **Robot Platform:** Tesla Optimus (Gen 1, Gen 2)
- **Foundation Model:**
  - End-to-end neural network trained on video, maps, and kinematic data
  - Leverages FSD (Full Self-Driving) AI stack
  - Neural World Simulator: Trained on fleet data, generates high-fidelity synthetic video
  - Runs neural network directly on hardware for real-time decisions
  - 48 neural networks working in concert (from FSD architecture)
- **Key Capabilities:**
  - Vision-based system (no LiDAR)
  - Reinforcement learning for task improvement
  - AI-powered motion controllers for human-like movement
  - Transfer learning from vehicle fleet to robot
- **Training Data:** "Niagara Falls of data" from Tesla vehicle fleet (>1.5 petabytes)
- **Physical Platform:**
  - Human-like proportions and dexterity
  - Dexterous hands
  - Vision-based navigation
- **Production Plans:** 5,000 units targeted for 2025
- **Status:** Active development, factory deployment testing
- **Key Tech:** Shared AI infrastructure with FSD, world model simulation

#### Figure AI
- **Country:** United States
- **Founded:** 2022
- **Robot Platform:** Figure 01, Figure 02
- **Foundation Model:**
  - Vision-language model (VLM) for conversation and perception
  - Custom AI models trained in partnership with OpenAI
  - 6 RGB cameras paired with onboard VLM
  - End-to-end learning models integrating language, vision, and action
- **Physical Specs (Figure 02):**
  - Height: 168 cm
  - Weight: 70 kg
  - Payload: 20 kg (25 kg for hands)
  - Battery: 5 hours continuous operation
  - Walking speed: 1.2 m/s
  - Hands: 16 degrees of freedom
- **Key Capabilities:**
  - 3x computing power vs Figure 01
  - Full conversation through onboard microphones/speakers
  - Autonomous task execution
  - Natural language command understanding
- **Funding:** $675M Series B (Feb 2024), $2.6B valuation
  - Latest talks: $39.5B valuation (Feb 2025)
- **Investors:** Microsoft, OpenAI Startup Fund, NVIDIA, Amazon Industrial Innovation Fund, Jeff Bezos
- **Status:** Commercial deployment, BMW factory pilot
- **Goal:** General-purpose humanoid trained by one AI model for multiple use cases

#### Physical Intelligence (π)
- **Country:** United States
- **Founded:** 2024 (8 months to π0 model)
- **Foundation Model:** π0 (pi-zero)
  - **Architecture:** 3.3B parameters total
    - Base: PaliGemma 3B vision-language model
    - Additional: 300M parameters for robot control
  - Novel "flow matching" architecture
  - High-frequency control: up to 50 Hz
  - Vision-Language-Action (VLA) model
- **Training Data:**
  - Custom dataset: 7 different robots, 68 tasks
  - Open X-Embodiment dataset
  - 1-20 hours sufficient for fine-tuning new tasks
- **Key Capabilities:**
  - Cross-embodiment control
  - Dexterous manipulation: folding laundry, assembling boxes, bussing dishes
  - Unique tasks: folding laundry from hamper, cardboard box assembly
  - Low-level motor commands at 50 Hz
- **Status:** Open-sourced (openpi repository on GitHub)
- **Funding:** $400M (Nov 2024)
- **Notable:** First to successfully fold laundry and assemble boxes autonomously

#### Agility Robotics (Digit)
- **Country:** United States
- **Founded:** 2015 (spun out from Oregon State University)
- **Robot Platform:** Digit
- **Foundation Model:**
  - Whole-Body Control Foundation Model
  - Acts as "motor cortex" for robot
  - Small LSTM neural network: <1M parameters
  - Trained in NVIDIA Isaac Sim for decades of simulated time (3-4 days real time)
- **Training Data:** 2,000 hours of simulated motion
  - Diverse scenarios: uneven terrain, object manipulation, disturbances
- **Key Capabilities:**
  - Zero-shot sim-to-real transfer
  - Stable locomotion, manipulation, disturbance recovery
  - "Always on" safety layer
  - Reactive and intuitive control
- **Physical Platform:**
  - Bipedal humanoid
  - Designed for warehouse/logistics
- **Deployments:**
  - GXO: Georgia fulfillment center (Robots-as-a-Service)
  - Schaeffler: Cheraw, SC plant (washing machine housings)
- **Status:** Commercial production and deployment
- **Partnership:** NVIDIA (Isaac Sim training)

#### Apptronik (Apollo)
- **Country:** United States
- **Founded:** 2016 (from UT Austin Human Centered Robotics Lab)
- **Robot Platform:** Apollo
- **Foundation Model:**
  - Integrates with NVIDIA Project GR00T
  - VLM-based system for multimodal input (text, video, images)
  - Generates actions from language/visual prompts
- **Physical Specs:**
  - Height: 5'8" (173 cm)
  - Weight: 160 lbs (73 kg)
  - Payload: 55 lbs (25 kg)
- **Key Capabilities:**
  - Task learning from text, video, demonstrations
  - Coordination and dexterity
  - Autonomous operation (e.g., operating juicer)
- **Status:** Testing with Mercedes-Benz, commercial pilots
- **Launch Date:** August 2023
- **Design Focus:** Friendly interaction, mass manufacturability, safety

#### Sanctuary AI (Phoenix)
- **Country:** Canada
- **Founded:** Unknown
- **Robot Platform:** Phoenix (6th generation)
- **Foundation Model:** Carbon AI Control System
  - Simulates human brain subsystems (memory, sensory perception)
  - Symbolic and logical reasoning + modern LLMs
  - Domain-specific integrations
  - Multimodal understanding
- **Physical Specs:**
  - Height: 170 cm
  - Weight: 70 kg
  - Payload: 25 kg
  - Hands: 20 degrees of freedom (proprietary haptic technology)
- **Key Capabilities:**
  - Industry-leading hand dexterity
  - Human-like sense of touch
  - Task automation in <24 hours (down from weeks)
  - Hundreds of tasks across 12+ industries
  - Natural language taskability
- **Status:** Commercial deployment
- **Notable:** Phoenix mimics human hand dexterity with proprietary haptic sensors

#### Skild AI
- **Country:** United States (Carnegie Mellon spinout)
- **Founded:** 2023
- **Founders:** CMU Professors Deepak Pathak & Abhinav Gupta
- **Foundation Model:** Skild Brain
  - Scalable robotics foundation model
  - Adapts across hardware and tasks
  - Trained on 1,000x more data than competing models (claimed)
- **Platforms:**
  - Mobile manipulation platform
  - Quadruped platform (security/inspection)
- **Expertise:** 25+ years combined experience in:
  - Self-supervised robotics
  - Curiosity-driven agents
  - Adaptive robot learning
- **Training Approach:**
  - Reinforcement learning in simulation
  - Sim2Real techniques
- **Funding:** $300M (2024)
- **Status:** Active development

#### Mentee Robotics
- **Country:** Israel
- **Founded:** 2022
- **Founders:** Prof. Amnon Shashua (expert in AI, computer vision, NLP)
- **Robot Platform:** MenteeBot (V3.0)
- **Foundation Model:**
  - Enables "mentoring" via text instructions + live demonstrations
  - Transformer-based LLMs for task interpretation
  - Integrates: language, vision, speech, robot control
  - Two paradigms: "Learning from data" + "Learning from experience"
- **Training:** RL in simulation + Sim2Real transfer
- **Key Capabilities:**
  - Complete end-to-end: verbal command → task completion
  - Navigation, locomotion, scene understanding
  - Object detection/localization, grasping
  - Natural language understanding
  - Leader in locomotion + dexterity
  - 360-degree vision, marker-less 3D navigation
  - Human-like grip strength
- **Status:** Active development
- **Notable:** "AI-first" approach with strong simulation foundation

### China

#### 1X Technologies (formerly Halodi Robotics)
- **Country:** Norway → United States (OpenAI backed)
- **Founded:** 2014 (as Halodi Robotics, rebranded 2022)
- **Founder:** Bernt Øivind Børnich
- **Robot Platform:** EVE (commercial), NEO (consumer home robot)
- **Foundation Model:** Redwood AI
  - Vision-language transformer
  - Tailored for humanoid form factor
  - End-to-end chore execution
  - Trained on real-world data
- **World Model:** 1X World Model
  - Data-driven simulator
  - Grounded physics understanding
  - Predicts outcomes before real-world execution
- **Key Capabilities (NEO):**
  - Voice command understanding
  - Natural language interaction
  - Laundry, answering doors, navigation
  - Mobile manipulation tasks
  - World's first consumer-ready humanoid
- **Training:** Gets more robust/versatile as it performs more tasks across homes
- **Funding:** $23.5M Series A2 (March 2023, led by OpenAI Startup Fund)
- **Status:** NEO in consumer rollout, EVE in commercial settings
- **Notable:** First consumer-focused humanoid with home deployment

#### UBTech (Walker Series)
- **Country:** China
- **Founded:** Unknown
- **Robot Platform:** Walker S1, Walker S2, Tiangong Walker
- **Foundation Model:**
  - First major reasoning multimodal model for humanoid robots
  - Developed with DeepSeek-R1 deep reasoning technology
  - BrainNet framework: "super brain" + "intelligent sub-brain"
  - Trained on high-quality industrial dataset from real factories
- **Key Achievement:** World's first multi-humanoid robot coordination in factories
- **Key Capabilities:**
  - Human-like common-sense reasoning
  - Swarm intelligence coordination
  - Factory automation tasks
- **Deployments:**
  - BYD factories (world's largest EV manufacturer)
  - Audi China plant
  - Zeekr 5G smart factory
  - 500+ orders for Walker S1
- **Production Target:** 500-1,000 units in 2025 (60%+ Walker S2)
- **Status:** Mass production, commercial deployment
- **Notable:** First successful multi-robot coordination in production environments

#### AgiBot (Zhiyuan Robotics)
- **Country:** China (Shanghai)
- **Founded:** 2023
- **Founders:** Deng Taihua & Peng Zhihui (former Huawei engineers)
- **Robot Platform:** Yuanzheng series (A2, A2-Max, A2-W), Lingxi series (X1, X1-W)
- **Foundation Model:** Genie Operator-1 (GO-1)
  - Generalist embodied foundation model
  - Built on extensive dataset
  - General-purpose brain for humanoid robots
  - Fast learning and adaptation
- **Dataset:** AgiBot World (open-source, 2024)
  - 1M+ diverse training sets
  - 100 robots contributing data
  - Largest humanoid manipulation dataset (as of Dec 2024)
- **Physical Specs (Yuanzheng A2):**
  - Height: 1.75 m
  - Weight: 55 kg
  - Biped humanoid
- **Production:** 962 units manufactured as of Dec 15, 2024
- **Status:** Mass production begun Dec 2024
- **Goal:** Match Tesla Optimus output in 2025
- **Partnerships:** Tencent investor

#### Fourier Intelligence (GR-1)
- **Country:** China (Shanghai)
- **Founded:** 2015
- **Founder:** Alex Gu
- **Robot Platform:** GR-1
- **Foundation Model:** Hardware support for NVIDIA GR00T N1 development
- **Physical Specs:**
  - Height: 1.65 m
  - Weight: 55 kg
  - DOF: 40 degrees of freedom
  - Payload: Up to 50 kg (nearly its own weight)
- **Launch:** Unveiled at World AI Conference Shanghai (July 2023)
- **Production:** 100+ units delivered across various sectors
- **Status:** World's first mass-produced humanoid robot (claimed)
- **Training:** NVIDIA Isaac Gym
- **Notable:** Achieved bipedal walking in 2022, mass production in 2023

#### Unitree Robotics
- **Country:** China
- **Founded:** Unknown
- **CEO:** Wang Xingxing
- **Robot Platform:**
  - Humanoids: H1, G1 ($16K)
  - Quadrupeds: Go1, Go2, B2
- **Foundation Model Plans:**
  - Integrating general AI models with robotics
  - Visual perception, tactile sensing, decision-making, interaction systems
  - CEO predicts: At least one company will develop general-purpose robotic AI model by end of 2025
- **Key Capabilities (G1 - 2024):**
  - Superhuman flexibility
  - Martial arts maneuvers (aerial cartwheels, kip-ups)
  - Data capture + reinforcement learning
- **Technology Focus:**
  - Years of quadruped tech accumulation
  - First bipedal prototype in 6 months (2023)
- **Status:** Mass production (G1 launched Aug 2024)
- **Notable:** $16K price point for G1, lowest-cost humanoid

#### Ex-Robots (Dalian Tiasi Technology)
- **Country:** China (Dalian)
- **Founded:** By Li Boyang (PhD in AI from Waseda University 2010)
- **Specialization:** Entertainment and customer service humanoids
- **Foundation Model:**
  - Multi-modal model for environment recognition + response
  - Appropriate facial feedback based on input
- **Key Feature:** Lifelike facial movements and expressions
- **Production:**
  - 200+ robots operational (early 2024)
  - Target: 500+ units by end 2024
  - Dedicated production facility in Dalian
- **Production Time:** 2 weeks to 1 month per robot
- **Price:** 1.5M - 2M yuan (~$210K-$280K)
- **Partnerships:** Huawei, iFlyTek, China Mobile
  - Access to state-of-the-art foundation models and cloud AI
- **Status:** Active production
- **Notable:** Focus on ultra-realistic facial expressions

#### Xiaomi (CyberOne)
- **Country:** China
- **Founded:** 2010 (company), CyberOne unveiled 2022
- **CEO:** Lei Jun
- **Robot Platform:** CyberOne
- **Foundation Model:**
  - Mi-Sense depth vision module
  - AI interaction algorithm
  - Self-developed by Xiaomi Robotics Lab
- **Physical Specs:**
  - Height: 177 cm
  - Weight: 52 kg
  - DOF: 21 degrees of freedom
  - Display: Curved OLED for interactive information
- **Key Capabilities:**
  - 3D space perception
  - Individual/gesture/expression recognition
  - 45 classifications of human emotion
- **Future Plans:**
  - Phased implementation in Xiaomi manufacturing
  - Utilize intelligence in specific manufacturing scenarios
- **Investment:** Heavy R&D in software, hardware, algorithms
- **Status:** R&D phase, manufacturing integration planned

#### Huawei (Kuafu Robot)
- **Country:** China
- **Founded:** 1987 (company)
- **Robot Platform:** Kuafu-MY
- **Foundation Model:** Pangu Embodied AI Model 5.0
  - Understanding, NLP command recognition
  - Task planning, dual-arm collaboration
  - Autonomous execution
  - Complete series: NLP, CV, multimodal, prediction, scientific computing
  - Parameters: billions to trillions across different sizes
- **Infrastructure:**
  - CloudRobo Embodied AI Platform
  - Deploys algorithms/logic on cloud for lightweight robots
- **Innovation Center:**
  - Huawei (Shenzhen) Global Embodied AI Industry Innovation Centre (2024)
  - Integrates embodied AI capabilities across teams
  - Key foundational technologies: embodied AI models + computing power
- **Launch:** Kuafu-MY launched May 2024
- **Partnerships:** Leju Robot, Han's Robot (Shenzhen)
- **Investment:** $413M into robotics subsidiary
- **Platform:** OpenHarmony + Huawei Cloud
- **Goal:** "Humanoid Robot+" open ecosystem for universal embodied solution

---

## Research Labs & Institutions

### United States

#### Google DeepMind Robotics
- **Country:** United States / United Kingdom
- **Key Models:**
  - **RT-1 (Robotics Transformer 1):** Initial VLA model
  - **RT-2 (Robotics Transformer 2):**
    - First-of-its-kind Vision-Language-Action (VLA) model
    - Transformer-based, trained on text + images from web + robotic data
    - Directly outputs robotic actions
    - Actions represented as language tokens
    - Compatible with large pre-trained VLMs
  - **RT-2-X:**
    - Trained on RT-2 + web + robotics data
    - 3x more successful than RT-2 for emergent skills
  - **RT-X Models:** Cross-embodiment family
- **Key Capabilities:**
  - Improved generalization to novel objects
  - Interprets commands not in robot training data
  - Rudimentary reasoning in response to user commands
  - Semantic and visual understanding beyond robot data
- **Training Data:** Internet-scale vision-language datasets + robotic datasets
- **Status:** Active research, published models
- **Notable:** Pioneered VLA paradigm in robotics

#### Berkeley RAIL (Robot AI & Learning Lab)
- **Country:** United States (UC Berkeley)
- **Key Projects:**
  - **BridgeData V2:** Large-scale robot learning dataset
    - Object manipulation: pick-and-place, pushing, sweeping
    - Environment manipulation: doors, drawers
    - Complex tasks: stacking blocks, folding cloths, sweeping granular media
  - **CrossFormer:**
    - Transformer-based robot policy
    - 900K robot trajectories across 20 different embodiments
- **Hardware:** WidowX 250 6DOF robot arm
- **Data Collection:** VR controller teleoperation
- **Research Focus:**
  - Flexible and adaptable behavior through learning
  - Learning algorithms, robotics, computer vision
- **Status:** Active research, open datasets
- **Notable:** Major contributor to cross-embodiment learning

#### MIT CSAIL (Embodied Intelligence)
- **Country:** United States
- **Key Projects:**
  - **Foundation Model Supervision:** Leverage non-robot foundation models for scalable supervision
  - **KALM Framework:**
    - Uses pre-trained VLMs for task-relevant keypoints
    - Guides diffusion action model
    - Keypoint-conditioned policies generalize across poses, views, instances
    - 5-10 demonstrations sufficient
  - **SceneComplete:** Composes foundation models for 3D scene completion
  - **HiP (Hierarchical Planning):** Compositional foundation models for planning
  - **Neural Jacobian Fields (NJF):**
    - Robots learn body response to control solely through vision
    - No hand-designed models or complex sensors
    - Bodily self-awareness
- **Research Focus:**
  - Motion/task planning, ML, RL, computer vision
  - Intelligent behavior across wide problem domains
- **Status:** Active research
- **Challenge:** Robot datasets orders of magnitude smaller than vision/language datasets

#### Carnegie Mellon University (Robotics Institute)
- **Country:** United States
- **Key Projects:**
  - **Skild AI** (spinout, see above)
  - **CMU Vision-Language-Autonomy Challenge:**
    - Simulation-only in 2024
    - Real robot deployment starting 2025
    - Workshop at IROS 2024
- **CMLH-NVIDIA Joint Research Center:** Robotics, Autonomy & AI
- **Definition:** Embodied AI = ML + CV + robot learning + language → robots that perceive, act, collaborate
- **Status:** Active research, major spinouts

#### Stanford Robotics
- **Country:** United States
- **Key Projects:**
  - **Mobile ALOHA:**
    - Low-cost bimanual mobile manipulation
    - Whole-body teleoperation system
    - Cost: ~$32K (vs $200K for commercial bimanual robots)
    - Teleoperation: Human tethered by waist, drives system, operates arms with controllers
    - **Training:** Supervised behavior cloning
      - Co-training with static ALOHA datasets boosts performance
      - 50 demonstrations per task
      - Co-training increases success rates up to 90%
  - **Capabilities:**
    - Sautéing and serving shrimp
    - Opening two-door cabinets, storing pots
    - Calling and entering elevator
    - Rinsing pans with faucet
  - **OpenVLA:** 7B parameter open-source VLA model (June 2024)
    - Trained on Open X-Embodiment dataset
    - Outperforms RT-2-X (55B) by 16.5% with 7x fewer parameters
    - 29 tasks, multiple embodiments
- **Status:** Active research, open source
- **Notable:** Mobile ALOHA demonstrates low-cost path to bimanual manipulation

#### UC San Diego (Contextual Robotics Institute)
- **Country:** United States
- **Focus:**
  - Bridging Embodied AI and Applications (2025 Forum)
  - Cross-embodiment robot intelligence
  - Transfer learning to new hardware
- **Research Areas:**
  - Foundation models for industrial robots
  - Medical procedure training potential
  - Visuomotor skills development
  - Large-scale simulation environments
- **Goal:** Foundation models that adapt to worn motors, custom hardware, home-built robots
- **Status:** Active research

#### Toyota Research Institute (TRI)
- **Country:** United States / Japan
- **Foundation Model:** Large Behavior Models (LBMs)
  - Diffusion-based transformer architecture
  - Processes visual, proprioceptive, textual data
  - Real-time decision making
- **Training Data:**
  - ~1,700 hours of robot data
  - 1,800 real-world evaluation rollouts
  - 47,000+ simulation rollouts
- **2024 Breakthrough:**
  - Single LBM learns hundreds of manipulation tasks
  - 80% less data needed vs traditional approaches
  - Pretrained on thousands of hours of physical robot data
  - Learns new tasks 3-5x faster
- **Partnership:** Boston Dynamics (Atlas humanoid)
  - Announced Oct 2024
  - TRI's LBMs + Boston Dynamics' Atlas robot
  - Goal: Accelerate general-purpose humanoid development
- **LBM Architecture:**
  - 450M-parameter diffusion transformer
  - Processes: stereo camera images, proprioceptive data, language prompts
  - Outputs: continuous actions at 30 Hz
  - Controls all 50 DOF of Atlas
- **Training Pipeline:**
  - VR teleoperation for demonstrations
  - ML pipeline for policy training
  - High-fidelity simulator evaluation
  - Deploy on physical robot
  - Iterative loop for continuous improvement
- **Status:** Active research, commercial partnership
- **Notable:** "If you can demonstrate it, the robot can learn it"

### Europe

#### ETH Zurich Robotics
- **Country:** Switzerland
- **Key Labs:**
  - Autonomous Systems Lab (ASL)
  - Robotics and Perception Group
  - Institute of Robotics and Intelligent Systems (IRIS) - 7 labs
- **Research Focus:**
  - Autonomous navigation using only onboard cameras/computation
  - No external infrastructure (GPS, position tracking, off-board computing)
  - Control of automation systems with physical interactions
  - Novel control, communication, decision-making strategies
  - Flying robots, service robots, mobile robots
- **Status:** Active research
- **Notable:** Strong foundation in autonomous systems, perception

### Asia

#### Tencent (Robotics X Lab)
- **Country:** China
- **Founded:** 2018 (Robotics X Lab)
- **Foundation Model:** Tairos Platform
  - Modular, plug-and-play embodied AI platform
  - Acts as "AI brain" for humanoid robots
  - **SLAP3 Framework:**
    - **S**ensing models: Interpret environment + internal states
    - **L**earning models
    - **A**ction models: Convert sensory input to actions
    - **P**lanning models: Break complex tasks into steps
    - **P**erception models
    - **P**erception-action models
- **Infrastructure:**
  - Cloud: Simulation, training, data management
  - APIs/SDKs for developer access
- **Strategic Positioning:**
  - Neutral platform for startups lacking compute capacity
  - Bridges training/fine-tuning cost gaps
- **Partnerships:** AgiBot, KEENON, Unitree
- **Target Industries:** Manufacturing, logistics, services
- **Status:** Active development, industry partnerships
- **Investment:** Agibot investor

---

## Quadruped Robots

### Boston Dynamics (Spot, Atlas)
- **Country:** United States
- **Founded:** 1992 (spun out from MIT)
- **Robot Platforms:**
  - Spot (quadruped)
  - Atlas (humanoid, all-electric new version 2024)
- **Foundation Model:** Large Behavior Models (LBMs) - Partnership with TRI
  - 450M-parameter diffusion transformer
  - Processes: images, proprioceptive data, language prompts
  - 30 Hz action output
  - 50 DOF control (Atlas)
- **Training:**
  - Pretrained on thousands of hours of physical robot data
  - VR teleoperation for demonstrations
  - High-fidelity simulation before physical deployment
  - 3-5x less task-specific data needed
- **Key Capabilities:**
  - "If you can demonstrate it, the robot can learn it"
  - Learns rigid and deformable object manipulation
  - T-shirt folding, block stacking, complex assembly
- **Status:** Commercial (Spot), Research (Atlas LBM)
- **Partnership:** Toyota Research Institute (Oct 2024)
- **Notable:** Industry leader in dynamic mobility

### ANYbotics (ANYmal)
- **Country:** Switzerland (ETH Zurich spinout)
- **Founded:** 2016 (research started 2009)
- **Robot Platform:** ANYmal (ANYmal-B, ANYmal-C, Generation D)
- **Foundation Model Approach:**
  - Attention-based recurrent encoder
  - End-to-end training for perception fusion (exteroceptive + proprioceptive)
  - Neural network policy trained in simulation → transferred to robot
- **Training:**
  - Fast, automated, cost-effective data generation in sim
  - Locomotion skills: jumping, climbing, crouching, walking
  - Obstacle parkour navigation
  - Multi-robot learning: 16 different quadrupeds (2-200 kg)
- **Physical Platform:** Medium dog-sized quadruped
- **Key Capabilities:**
  - Autonomous operation in challenging environments
  - Robust perceptive locomotion
  - Integrated exteroceptive/proprioceptive perception
- **Community:** Hundreds of contributors (university labs + corporate innovation)
- **Status:** Commercial product + research platform
- **Notable:** Strong sim-to-real transfer, robust outdoor operation

### Unitree Robotics (Quadrupeds)
- **Country:** China
- **Quadruped Models:** Go1, Go2, B2
- **Approach:** Foundation for humanoid development
  - Years of quadruped tech → bipedal robots
  - First bipedal prototype in 6 months (2023)
- **Status:** Commercial quadruped sales, humanoid development

---

## Foundation Model Frameworks

### NVIDIA Isaac Platform
- **Country:** United States
- **Key Models:**
  - **Project GR00T:** General-purpose foundation model for humanoid robots (announced 2024)
  - **Isaac GR00T N1:** World's first open humanoid robot foundation model
  - **Isaac GR00T N1.5:** Improved architecture and data (latest)
- **Architecture:**
  - **Dual-System Design:**
    - **System 1:** Fast-thinking action model (reflexes/intuition)
    - **System 2:** Slow-thinking deliberate decision-making (VLM-powered)
  - Vision Language Model for reasoning
  - Multimodal input: language, images
  - Cross-embodiment model
- **Training:**
  - GPU-accelerated simulation
  - Imitation learning, RL, video data
  - Trained with heterogeneous data: real robot, human videos, synthetic
- **Platform:**
  - Isaac Lab: Open-source, modular robot learning framework
  - Isaac Sim: High-fidelity simulation environment
  - Omniverse: Digital twin platform
- **Partnerships:**
  - Figure AI, Apptronik, Agility Robotics, Boston Dynamics, Fourier Intelligence
- **Status:** Open source (GitHub: NVIDIA/Isaac-GR00T)
- **Model Sizes:** GR00T N1.5-3B on Hugging Face
- **Notable:** First open foundation model for humanoids

### Covariant (RFM-1)
- **Country:** United States
- **Founded:** By AI researchers focused on warehouse robotics
- **Foundation Model:** RFM-1 (Robotics Foundation Model)
  - **Architecture:** 8B parameter transformer
  - Multimodal any-to-any sequence model
  - Tokenizes all modalities into common space
  - Next-token prediction training
- **Training Data:**
  - 4 years of warehouse pick-and-place data
  - General internet data
  - Rich physical real-world interactions
- **Key Capabilities:**
  - Human-like reasoning ability
  - First successful commercial Generative AI for robots
  - Understands language + physical world
  - Natural language tasking (robots + humans communicate)
  - Physics world model: AI-generated video predictions
  - Simulates future scenarios, selects best action
- **Input/Output:** Any modality in, any modality out
- **Applications:** Warehouse pick-and-place operations
- **Status:** Commercial deployment
- **Notable:** First to give commercial robots deeper understanding via GenAI

### Open X-Embodiment
- **Type:** Collaborative dataset and models
- **Organizations:** 21 institutions worldwide
- **Dataset:**
  - 1M+ real robot trajectories
  - 22 robot embodiments (single arms, bimanual, quadrupeds)
  - 527 skills (160,266 tasks)
  - Standardized data formats
- **Models:** RT-X (trained on Open X-Embodiment data)
  - Positive transfer across platforms
  - Leverages experience from other robots
- **Goal:** Train generalist X-robot policies adaptable to new robots, tasks, environments
- **Published:** IEEE ICRA 2024 (May 13-17)
- **Repository:** Open source (Google DeepMind GitHub)
- **Status:** Active, growing collaboration
- **Notable:** "ImageNet of robot learning"

---

## Open Source Projects

### Hugging Face (LeRobot)
- **Country:** United States
- **Lead:** Remi Cadene (ex-Tesla)
- **Platform:** LeRobot
  - Models, datasets, tools for real-world robotics
  - PyTorch-based
  - Goal: Lower barrier to entry for robotics
- **Models:**
  - **π0 and π0-FAST:** From Physical Intelligence
  - **GR00T N1.5:** From NVIDIA
  - **SmolVLA:** 450M parameters, runs on consumer hardware
    - Cross-embodiment model
    - Pretrained on open-source community datasets
- **Infrastructure:**
  - Datasets v3.0 (scalable)
  - Plugin system for hardware integration
  - Hugging Face Hub: Most widely used for open robotics
- **Community:** 12,000+ GitHub stars in 12 months
- **Status:** Active, rapid growth
- **Notable:** Central hub for open robotics ecosystem

### Dyna Robotics
- **Country:** United States
- **Founded:** 2024
- **Foundation Model:** DYNA-1
  - Proprietary foundation model
  - >99% success rate in 24-hour non-stop operation
- **Funding:**
  - $23.5M seed (March 2024)
  - $120M Series A (2024)
- **Goal:** General-purpose robots in commercial environments
- **Status:** Active development, rapid scaling

### X Square Robot
- **Country:** China
- **Foundation Model:** Wall-OSS
  - Proprietary open-source foundation model
  - For robotic platforms
- **Robot Platform:** Quanta X2 (self-developed)
- **Funding:** $100M+ Series A+ (2024)
- **Status:** Active development

---

## Autonomous Driving (Embodied AI)

### Wayve
- **Country:** United Kingdom
- **Founded:** 2017
- **Focus:** Embodied AI for autonomous driving
- **Foundation Model:**
  - **LINGO-1:** Vision-language model for driving
    - Improves performance and interpretability of end-to-end AI models
    - Increases transparency in reasoning/decision-making
  - **LINGO-2:** First closed-loop vision-language-action driving model
    - Tested on public roads
    - Outputs: driving action + continuous driving commentary
    - Explains motion planning decisions
- **Training Data:**
  - Expert drivers commentating while driving around UK
  - Image, language, action data
  - Scalable and diverse dataset
- **Key Capabilities:**
  - "GPT for driving"
  - Foundation models empower any vehicle to see, think, drive
  - Natural language for learning + explainability
  - VLA models for driving
- **Approach:** Embodied AI Technology (AV 2.0)
- **Funding:** $1B+ Series C (led by SoftBank)
- **Product:** Wayve AI Driver
- **Status:** Active development, road testing
- **Notable:** Pioneer in vision-language-action for autonomous driving

---

## Key Metrics & Comparisons

### Foundation Model Parameter Counts

| Model | Parameters | Type | Organization |
|-------|-----------|------|--------------|
| RT-2-X | 55B | VLA | Google DeepMind |
| RFM-1 | 8B | Multimodal Transformer | Covariant |
| OpenVLA | 7B | VLA | Stanford |
| π0 (pi-zero) | 3.3B (3B + 0.3B) | VLA | Physical Intelligence |
| GR00T N1.5 | 3B | VLA | NVIDIA |
| DiVLA-2B | 2B | VLA | Research |
| MiniVLA | 1B | VLA | Stanford |
| Agility Digit | <1M | LSTM | Agility Robotics |
| SmolVLA | 450M | VLA | Hugging Face |
| OpenVLA variants | 0.5B, 1B, 7B | VLA | Stanford |

**Note:** Boston Dynamics/TRI LBM = 450M parameters (diffusion transformer)

### Training Data Scale

| Organization | Data Scale | Type |
|--------------|-----------|------|
| Tesla | >1.5 petabytes | Fleet video data |
| Covariant | 4 years | Warehouse operations |
| TRI | 1,700+ hours | Robot manipulation |
| Open X-Embodiment | 1M+ trajectories | 22 embodiments, 527 skills |
| AgiBot World | 1M+ training sets | 100 robots |
| Physical Intelligence | 7 robots, 68 tasks + Open X | Custom + public |
| Agility Robotics | 2,000 hours | Simulated motion |
| Skild AI | 1,000x competitors (claimed) | Not specified |

### Humanoid Robot Specifications

| Robot | Height | Weight | Payload | Battery | Speed | Hands DOF |
|-------|--------|--------|---------|---------|-------|-----------|
| Tesla Optimus | ~170 cm | ~70 kg | TBD | TBD | Walking | Dexterous |
| Figure 02 | 168 cm | 70 kg | 20 kg | 5 hours | 1.2 m/s | 16 |
| Sanctuary Phoenix | 170 cm | 70 kg | 25 kg | TBD | Walking | 20 |
| Apptronik Apollo | 173 cm | 73 kg | 25 kg | TBD | Walking | N/A |
| AgiBot Yuanzheng A2 | 175 cm | 55 kg | TBD | TBD | Walking | N/A |
| Fourier GR-1 | 165 cm | 55 kg | 50 kg | TBD | Walking | 40 DOF total |
| Unitree G1 | ~170 cm | ~55 kg | TBD | TBD | Fast | N/A |
| UBTech Walker S1 | ~170 cm | TBD | TBD | TBD | Walking | N/A |
| Xiaomi CyberOne | 177 cm | 52 kg | TBD | TBD | Walking | 21 DOF total |
| 1X NEO | ~170 cm | ~65 kg | TBD | TBD | Walking | N/A |

### Key Funding Rounds (2024-2025)

| Company | Amount | Valuation | Date | Lead Investors |
|---------|--------|-----------|------|----------------|
| Figure AI | $675M | $2.6B | Feb 2024 | Microsoft, OpenAI, NVIDIA, Amazon, Bezos |
| Figure AI (talks) | TBD | $39.5B | Feb 2025 | TBD |
| Physical Intelligence | $400M | TBD | Nov 2024 | TBD |
| Skild AI | $300M | TBD | 2024 | TBD |
| Dyna Robotics | $120M | TBD | 2024 | Series A |
| X Square Robot | $100M+ | TBD | 2024 | Series A+ |
| 1X Technologies | $23.5M | TBD | Mar 2023 | OpenAI Startup Fund |
| Wayve | $1B+ | TBD | 2024 | SoftBank |

### Commercial Deployments

| Company | Robot | Deployment | Application |
|---------|-------|------------|-------------|
| Agility Robotics | Digit | GXO Georgia | Fulfillment center (RaaS) |
| Agility Robotics | Digit | Schaeffler SC | Washing machine housings |
| UBTech | Walker S1 | BYD factories | EV manufacturing |
| UBTech | Walker S1 | Audi China | Automotive manufacturing |
| UBTech | Walker S1 | Zeekr | 5G smart factory |
| Figure AI | Figure 02 | BMW | Automotive pilot |
| Apptronik | Apollo | Mercedes-Benz | Automotive testing |
| Covariant | RFM-1 | Warehouses | Pick-and-place operations |
| 1X Technologies | EVE | Commercial | Logistics, security, medical |
| Boston Dynamics | Spot | Various | Inspection, security |

### Production Targets

| Company | Target | Timeline | Status |
|---------|--------|----------|--------|
| Tesla | 5,000 units | 2025 | Planned |
| AgiBot | Match Optimus output | 2025 | Goal |
| UBTech | 500-1,000 units | 2025 | Planned |
| AgiBot | 962 units | Dec 2024 | Achieved |
| Ex-Robots | 500+ units | End 2024 | Target |
| Fourier | 100+ units | 2023 | Achieved |

---

## VLA Model Performance

### Generalization Capabilities

- **OpenVLA:** Outperforms RT-2-X (55B) by 16.5% with 7x fewer parameters
- **RT-2-X:** 3x more successful than RT-2 for emergent skills
- **π0:** Successfully performs tasks no prior system achieved (laundry folding, box assembly)
- **TRI LBMs:** 80% less data needed vs traditional approaches, 3-5x faster learning
- **Physical Intelligence:** 1-20 hours sufficient for task fine-tuning

### Inference Speed vs Model Size

- **DiVLA-2B:** 82 Hz on single A6000 GPU
- **π0:** Up to 50 Hz control frequency
- **TRI LBM:** 30 Hz continuous action output
- **Trade-off:** Larger models (7B+) require more compute, smaller models (<2B) enable real-time on-device inference

### Zero-Shot Transfer

- **Agility Digit:** Zero-shot sim-to-real transfer
- **Boston Dynamics/TRI:** Sim-to-real with minimal fine-tuning
- **ANYbotics:** Effective sim-to-real for quadruped locomotion

---

## Architectural Approaches

### Dual-System Architecture (inspired by human cognition)

- **NVIDIA GR00T N1:** System 1 (fast/reflexive) + System 2 (slow/deliberate)
- **Sanctuary Carbon:** Simulates human brain subsystems

### End-to-End Learning

- **Tesla Optimus:** Single end-to-end neural network
- **Figure 02:** End-to-end learning for language, vision, action
- **Google RT-2:** End-to-end VLA model

### Foundation Model + Fine-Tuning

- **Physical Intelligence π0:** Large pre-trained model + task-specific fine-tuning (1-20 hours)
- **OpenVLA:** Pre-trained on Open X-Embodiment + fine-tuning
- **TRI LBMs:** Pre-trained on thousands of hours + task-specific data

### Modular/Compositional

- **MIT HiP:** Compositional foundation models for hierarchical planning
- **Tencent Tairos:** Modular platform with separate perception/planning/action models

### Sim-to-Real

- **Agility Robotics:** Decades of simulation → zero-shot real
- **Skild AI:** RL in simulation → Sim2Real transfer
- **TRI/Boston Dynamics:** High-fidelity simulation → physical deployment

---

## Key Trends & Insights

### 2024-2025 Breakthroughs

1. **Open Foundation Models:** NVIDIA GR00T N1/N1.5 first open humanoid foundation model
2. **VLA Proliferation:** OpenVLA, π0, SmolVLA democratizing robot intelligence
3. **Chinese Acceleration:** UBTech, AgiBot, Huawei, Tencent rapid progress
4. **Commercial Deployments:** Agility, UBTech, Covariant in production environments
5. **Massive Funding:** Figure AI $39.5B valuation talks, Physical Intelligence $400M, Skild $300M
6. **Multi-Robot Coordination:** UBTech achieves world's first swarm intelligence in factories
7. **Large Datasets:** AgiBot World 1M+ sets, Open X-Embodiment 1M+ trajectories
8. **Sim-to-Real Maturity:** Zero-shot transfer becoming standard (Agility, TRI)
9. **Language Integration:** VLA models making robots naturally taskable via language
10. **Convergence:** Similar architectures emerging (VLA, dual-system, diffusion transformers)

### Competitive Landscape

**US Leaders:**
- Foundation Models: Google DeepMind, NVIDIA, Physical Intelligence
- Humanoid Hardware: Tesla, Figure AI, Agility Robotics, Apptronik
- Research: Stanford, MIT, Berkeley, CMU

**Chinese Leaders:**
- Humanoid Hardware: UBTech, AgiBot, Unitree, Fourier Intelligence
- Foundation Models: Huawei (Pangu), Tencent (Tairos)
- Production Scale: UBTech (500+ orders), AgiBot (1000 units planned)

**European Leaders:**
- Quadrupeds: ANYbotics (Switzerland)
- Autonomous Systems: ETH Zurich
- Autonomous Driving: Wayve (UK)

**Cross-Border:**
- 1X Technologies (Norway → US, OpenAI backed)
- Sanctuary AI (Canada)
- Mentee Robotics (Israel)

### Technology Gaps & Challenges

1. **Data Scarcity:** Robot datasets still orders of magnitude smaller than vision/language
2. **Sim-to-Real Gap:** While improving, still requires significant data/tuning
3. **Real-Time Inference:** Balance between model size and control frequency
4. **Cost:** Humanoid robots still expensive ($16K-$200K+)
5. **Generalization:** Single model handling all tasks remains elusive
6. **Safety:** Robust safety guarantees in human environments
7. **Dexterity:** Fine manipulation still challenging
8. **Energy Efficiency:** Battery life limits operational time
9. **Standards:** Lack of standardized benchmarks/datasets (improving with Open X-Embodiment)
10. **Regulation:** Unclear regulatory landscape for humanoid deployment

### Future Predictions (2025-2026)

1. **General-Purpose AI Models:** Unitree CEO predicts at least one company achieves general robotic AI by end 2025
2. **Production Scale:** Tesla 5,000 units, UBTech 500-1,000 units, AgiBot matching Optimus
3. **Cost Reduction:** Humanoid costs likely to drop below $10K (Unitree G1 already at $16K)
4. **Deployment Expansion:** From factories to retail, hospitality, healthcare
5. **Foundation Model Convergence:** Likely consolidation around 2-7B parameter VLA models
6. **Open Source Growth:** More open-source models and datasets (driven by Hugging Face, NVIDIA)
7. **China-US Competition:** Intensifying competition in both hardware and software
8. **Multi-Modal Integration:** Better integration of vision, language, touch, proprioception
9. **Autonomous Driving Integration:** Wayve-style VLA models for vehicles
10. **Consumer Market:** 1X NEO-style home robots entering consumer market

---

## Additional Companies to Watch

### Not Yet Covered

- **iRobot:** Consumer robotics, potential foundation model work
- **Collaborative Robotics (Cobot companies):** Universal Robots, ABB, KUKA
- **Surgical Robotics:** Intuitive Surgical (da Vinci), potential for foundation models
- **Agricultural Robotics:** Various startups applying embodied AI
- **Warehouse Automation:** Amazon Robotics, Locus Robotics, others
- **Japanese Companies:** Honda (retired ASIMO, but ASIMO OS for EVs), Sony, SoftBank (Pepper)
- **South Korean Companies:** Samsung, Hyundai (Boston Dynamics acquisition), LG

### Emerging Startups

- **FieldAI:** One autonomy for all robots
- **Embodied Intelligence:** Learning from humans in VR
- **Walker Labs / Boston Engineering:** Robot AI development
- **Various Chinese Startups:** Rapid emergence of new humanoid companies

---

## Resources & References

### Key Papers

- Open X-Embodiment: Robotic Learning Datasets and RT-X Models (2024)
- RT-2: Vision-Language-Action Models (2023)
- GR00T N1: An Open Foundation Model for Generalist Humanoid Robots (2025)
- π0: A Vision-Language-Action Flow Model for General Robot Control (2024)
- Foundation models in robotics: Applications, challenges, and the future (2024)

### Key Conferences

- IEEE ICRA (International Conference on Robotics and Automation)
- IROS (IEEE/RSJ International Conference on Intelligent Robots and Systems)
- CoRL (Conference on Robot Learning)
- CVPR (Computer Vision and Pattern Recognition) - Embodied AI workshops
- NeurIPS - Robot learning workshops

### Key Datasets

- Open X-Embodiment (1M+ trajectories, 22 embodiments)
- AgiBot World (1M+ training sets, 100 robots)
- BridgeData V2 (UC Berkeley)
- RT-X datasets (Google DeepMind)

### Open Source Repositories

- NVIDIA Isaac-GR00T: github.com/NVIDIA/Isaac-GR00T
- Hugging Face LeRobot: github.com/huggingface/lerobot
- Open X-Embodiment: github.com/google-deepmind/open_x_embodiment
- Physical Intelligence OpenPI: github.com/physicalintelligence/openpi

---

**Last Updated:** 2025-11-07
**Compiled by:** Research on robotics and embodied AI foundation models
**Status:** Living document, subject to updates as field rapidly evolves
