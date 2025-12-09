# Bittensor Subnet Directory

## Overview

This document catalogs all Bittensor subnets, including their purpose, operators, and evaluation mechanisms. Bittensor had approximately **89 active subnets** as of early December 2025, though the network has since expanded to include subnet numbers up to approximately **SN114+**. The subnet landscape changes frequently as new subnets register and others are deregistered based on emissions voting.

This directory includes **placeholder entries for SN0-SN89+**, with detailed information for SN0-SN37 and partial information for select higher-numbered subnets. Subnets cover diverse AI tasks from text generation to compute marketplaces, DeFi predictions, data infrastructure, and specialized applications.

**Resources:**
- **Live Subnet Data:** [taostats.io/subnets](https://taostats.io/subnets)
- **Browse Subnets:** [tao.app](https://tao.app) (code repositories and details)
- **Analytics:** [taomarketcap.com](https://taomarketcap.com)
- **CLI Command:** `btcli subnets list`
- **Awesome Bittensor:** [learnbittensor/awesome-bittensor](https://github.com/learnbittensor/awesome-bittensor) (comprehensive resource list)

**Core Repositories:**
- **Bittensor SDK:** [`opentensor/bittensor`](https://github.com/opentensor/bittensor) - Internet-scale Neural Networks
- **Subtensor Blockchain:** [`opentensor/subtensor`](https://github.com/opentensor/subtensor) - Bittensor blockchain layer
- **Subnet Template:** [`opentensor/bittensor-subnet-template`](https://github.com/opentensor/bittensor-subnet-template) - Template for building subnets

**Major Subnet Operators:**
- **Opentensor Foundation:** [`opentensor`](https://github.com/opentensor) - Core Bittensor development
- **Macrocosmos:** [`macrocosm-os`](https://github.com/macrocosm-os) - SN1 Apex, Data Universe (SN13), Mainframe (SN25)
- **Nous Research:** [`NousResearch`](https://github.com/NousResearch) - SN6 fine-tuning
- **Neural Internet:** [`neuralinternet`](https://github.com/neuralinternet) - SN27 compute
- **WOMBO:** [`womboai`](https://github.com/womboai) - SN30 image generation
- **MyShell:** [`myshell-ai`](https://github.com/myshell-ai) - SN3 text-to-speech

**Note:** Subnet landscape changes frequently as new subnets register and others are deregistered. This document is maintained as a reference but may not reflect real-time changes.

---

## Subnet Categories

Bittensor subnets span **6 major categories:**

1. **Content Generation** - Text, images, audio, video
2. **Data Collection & Processing** - Scraping, storage, datasets
3. **LLM Ecosystem** - Training, fine-tuning, inference
4. **Decentralized Infrastructure** - Compute, storage, networking
5. **DeFi** - Financial predictions, trading strategies, liquidity
6. **Other Applications** - Protein folding, research, specialized tasks

---

## Core Governance Subnet

### SN0: Root Network
- **Name:** Root
- **Purpose:** Governance and TAO emissions distribution
- **Operator:** Opentensor Foundation
- **Function:** Controls how TAO emissions are allocated across all subnets
- **Evaluation:** Validators vote on subnet emission weights based on subnet utility and performance
- **Special Status:** Not a production subnet; governs the network itself

---

## Production Subnets (Confirmed Active)

### SN1: Text Prompting (Apex)
- **Name:** Apex (formerly Text Prompting)
- **Operator:** [Macrocosmos](https://www.macrocosmos.ai) / Opentensor Foundation
- **GitHub:** [`macrocosm-os/apex`](https://github.com/macrocosm-os/apex)
- **Previous Repo:** [`opentensor/prompting`](https://github.com/opentensor/prompting) (legacy)
- **Purpose:** Conversational AI and text generation (LLM inference)
- **Description:** "Most intelligent inference model on Bittensor" - first agent to achieve deep-researcher reasoning on protocol
- **Evaluation Method:**
  - **Phase 1:** RLHF + DPO scoring vs reference answers (60%/40% weighted)
  - **Phase 2:** GAN-style mechanism - miners act as both generators and discriminators
  - **Phase 3:** Organic query integration - real user questions mixed with synthetic prompts
  - Zero-sum game: Generators earn by fooling discriminators, discriminators earn by catching generators
- **Status:** Most mature subnet, flagship Bittensor application
- **See Also:** [Detailed Evaluation Deep Dive](subnet1-evaluation-deep-dive.md)

### SN2: Machine Translation / Omron
- **Name:** Omron (previously Machine Translation)
- **Operator:** Lucrosus Capital / Inference Labs
- **Purpose:** Intelligent capital networks (evolved from translation tasks)
- **Registered:** April 10, 2024
- **Evaluation Method:** TBD (research in progress)
- **Note:** Subnet purpose appears to have evolved over time

### SN3: Data Scraping / MyShell TTS
- **Name:** MyShell (previously Data Scraping)
- **Operator:** [MyShell.ai](https://myshell.ai) / GitPhantom
- **GitHub:** [`myshell-ai/MyShell-TTS-Subnet`](https://github.com/myshell-ai/MyShell-TTS-Subnet)
- **Purpose:** Text-to-Speech (TTS) generation
- **Description:** Innovative, open-source Text-to-Speech technology
- **Registered:** March 25, 2024
- **Evaluation Method:** TTS quality assessment (specifics TBD)
- **Note:** Evolved from data scraping to specialized TTS subnet

### SN4: Multi-Modality
- **Name:** Multi-Modality
- **Operator:** Carro
- **Purpose:** Multi-modal AI (text + images + other modalities combined)
- **Evaluation Method:** TBD (research in progress)

### SN5: Image Generation / OpenKaito
- **Name:** OpenKaito (previously Image Generation)
- **Operator:** CreativeBuilds / yz_h
- **Purpose:** Unknown (evolved from image generation)
- **Evaluation Method:** TBD (research in progress)
- **Note:** Subnet repurposed from original image focus

### SN6: Nous
- **Name:** Nous
- **Operator:** [Nous Research](https://nousresearch.com) / [@theemozilla](https://twitter.com/theemozilla)
- **GitHub:** [`NousResearch/finetuning-subnet`](https://github.com/NousResearch/finetuning-subnet)
- **Purpose:** Continuous fine-tuning of LLMs with incentivized synthetic data
- **Description:** Miners host fine-tuned models evaluated on latest Cortex.t subnet data
- **Evaluation Method:** Loss comparison - miners with lower loss on new data earn higher rewards
- **Related:** Works with Cortex.t subnet for synthetic data generation

### SN7: Subvortex
- **Name:** Subvortex
- **Operator:** ch3rnobog
- **Purpose:** Boosts Bittensor with decentralized subtensor nodes
- **Evaluation Method:** TBD (research in progress)

### SN8: Taoshi
- **Name:** Taoshi / Time Series Prediction
- **Operator:** Arrash / taoshidev
- **GitHub:** [`taoshidev/time-series-prediction-subnet`](https://github.com/taoshidev/time-series-prediction-subnet)
- **Purpose:** The Nexus of Decentralized Financial Forecasting
- **Evaluation Method:** TBD (research in progress)

### SN9: Pre Training
- **Name:** Pre Training
- **Operator:** Const / unconst
- **GitHub:** [`unconst/pretrain-subnet`](https://github.com/unconst/pretrain-subnet)
- **Purpose:** Facilitates pre-training of AI models on large-scale datasets
- **Evaluation Method:** TBD (research in progress)

### SN10: Apollo ZK / Map Reduce
- **Name:** Apollo ZK (previously Map Reduce)
- **Operator:** ChainDude / dream-well
- **GitHub:** [`dream-well/map-reduce-subnet`](https://github.com/dream-well/map-reduce-subnet)
- **Purpose:** Zero-knowledge proof computing with network scaling
- **Evaluation Method:** TBD (research in progress)

### SN11: Transcription
- **Name:** Transcription
- **Operator:** Cazure
- **Purpose:** Decentralized AI for audio-to-text transcription
- **Evaluation Method:** TBD (research in progress)

### SN12: Compute Horde / Horde
- **Name:** Compute Horde (Horde)
- **Operator:** rhef
- **Purpose:** Enables decentralized, parallel task processing
- **Description:** Dedicated to decentralized resource allocation for computing
- **Evaluation Method:** TBD (research in progress)

### SN13: Dataverse / Data Universe
- **Name:** Dataverse (Data Universe)
- **Operator:** Macrocosmos / RusticLuftig
- **GitHub:** [`RusticLuftig/data-universe`](https://github.com/RusticLuftig/data-universe)
- **Purpose:** Data infrastructure for Bittensor AI development
- **Description:** Critical pillar of AI for Bittensor
- **Evaluation Method:** TBD (research in progress)

### SN14: LLM Defender / VectorStore
- **Name:** LLM Defender (VectorStore)
- **Operator:** ceterum1
- **GitHub:** [`ceterum1/llm-defender-subnet`](https://github.com/ceterum1/llm-defender-subnet)
- **Purpose:** Enhancing LLM security through decentralized, multi-layered defense
- **Evaluation Method:** TBD (research in progress)

### SN15: Blockchain Insights
- **Name:** Blockchain Insights
- **Operator:** blockchain-insights
- **GitHub:** [`blockchain-insights/blockchain-data-subnet`](https://github.com/blockchain-insights/blockchain-data-subnet)
- **Purpose:** Graph-based blockchain data analytics and insight platform
- **Evaluation Method:** TBD (research in progress)

### SN16: BitAds / Audio Subnet
- **Name:** BitAds (previously AudioSubnet)
- **Operator:** Emperor Wang / UncleTensor
- **GitHub:** [`UncleTensor/AudioSubnet`](https://github.com/UncleTensor/AudioSubnet)
- **Purpose:** Decentralized advertising network (evolved from music generation)
- **Evaluation Method:** TBD (research in progress)

### SN17: PixML / Three Gen / Flavia
- **Name:** PixML (Flavia / Three Gen)
- **Operator:** CortexLM
- **GitHub:** [`CortexLM/flavia`](https://github.com/CortexLM/flavia)
- **Purpose:** Diffusion model fine-tuning via Vision Subnet
- **Description:** Focused on finetuning/pre-training Diffusion models on the Bittensor network
- **Evaluation Method:** TBD (research in progress)

### SN18: Cortex.T / Zeus
- **Name:** Cortex.T (Zeus)
- **Operator:** corcel-api
- **GitHub:** [`corcel-api/cortex.t`](https://github.com/corcel-api/cortex.t)
- **Purpose:** AI development and synthetic data generation platform
- **Evaluation Method:** TBD (research in progress)

### SN19: Nineteen / Vision
- **Name:** Nineteen (Vision)
- **Operator:** namoray
- **GitHub:** [`namoray/vision`](https://github.com/namoray/vision)
- **Purpose:** Decentralized AI model inference at scale / Multi-model inference platform
- **Focus:** Text and image generation
- **Models:** LLaMA 3, Stable Diffusion derivatives
- **Evaluation Method:** TBD (research in progress)
- **Description:** Leading inference subnet providing access to advanced open-source models

### SN20: BitAgent / Oracle
- **Name:** BitAgent (Oracle Subnet)
- **Operator:** Rizzo / oracle-subnet
- **GitHub:** [`oracle-subnet/oracle-subnet`](https://github.com/oracle-subnet/oracle-subnet)
- **Purpose:** Intelligent agent integration and automation
- **Evaluation Method:** TBD (research in progress)

### SN21: Filetao / Storage / Omega Any-to-Any
- **Name:** Filetao (Storage Subnet / Omega Any-to-Any)
- **Operator:** philanthrope / ifrit98
- **GitHub:** [`ifrit98/storage-subnet`](https://github.com/ifrit98/storage-subnet)
- **Purpose:** Decentralized storage with zero-knowledge proofs
- **Evaluation Method:** TBD (research in progress)

### SN22: Meta Search / Datura
- **Name:** Meta Search (Datura)
- **Operator:** floppyfish
- **Purpose:** Twitter data analysis platform
- **Evaluation Method:** TBD (research in progress)

### SN23: Niche Image
- **Name:** Niche Image
- **Purpose:** Decentralized image generation (distributed image generation using Bittensor)
- **Function:** Miners contribute compute to produce images across various models
- **Evaluation Method:** Image quality assessment (specifics TBD)

### SN24: Omega Labs
- **Name:** Omega Labs
- **Operator:** omegalabs
- **GitHub:** [`omegalabsinc/omegalabs-bittensor-subnet`](https://github.com/omegalabsinc/omegalabs-bittensor-subnet)
- **Purpose:** Decentralized multimodal dataset for AGI research
- **Description:** Data collection supporting SN21's Any-to-Any multimodal model development
- **Evaluation Method:** TBD (research in progress)

### SN25: Hivetrain / Mainframe
- **Name:** Hivetrain (Mainframe)
- **Operator:** bitcurrent / Macrocosmos
- **Purpose:** Distributed deep learning approach
- **Evaluation Method:** TBD (research in progress)

### SN26: Image Alchemy
- **Name:** Image Alchemy
- **Operator:** Emperor Wang
- **Purpose:** Decentralized image creation and synthesis
- **Evaluation Method:** TBD (research in progress)

### SN27: Compute Subnet
- **Name:** Compute
- **Operator:** [Neural Internet](https://neuralinternet.ai)
- **GitHub:** [`neuralinternet/compute-subnet`](https://github.com/neuralinternet/compute-subnet)
- **Alternative Link:** [`neuralinternet/SN27`](https://github.com/neuralinternet/SN27)
- **Purpose:** Decentralized GPU marketplace / AI compute network
- **Description:** "Powers a decentralized compute market, enabling miners to contribute GPU resources and earn rewards"
- **Function:** Miners rent out GPU compute power, users rent for AI workloads
- **Evaluation Method:** Proof of compute delivery, verification of work completion
- **Features:** Trustless GPU rental with built-in verification

### SN28: Foundry S&P 500 Oracle
- **Name:** Foundry S&P 500 Oracle
- **Operator:** theBom
- **Purpose:** Stock price prediction incentives
- **Description:** Encourages miners to frequently predict the S&P 500 price during trading hours
- **Evaluation Method:** TBD (research in progress)

### SN29: Fractal
- **Name:** Fractal
- **Operator:** theAdoringFan
- **Purpose:** Decentralized video generation inference
- **Evaluation Method:** TBD (research in progress)

### SN30: WomboAI
- **Name:** WomboAI
- **Operator:** [WOMBO](https://www.wombo.ai) / WOMBO Dream team
- **GitHub:** [`womboai/wombo-bittensor-subnet`](https://github.com/womboai/wombo-bittensor-subnet)
- **Organization:** [`womboai`](https://github.com/womboai)
- **Purpose:** Image generation and social sharing / Decentralized content creation engine
- **Description:** Powers WOMBO Dream and WOMBO Me apps with ~200K DAU, ~5M MAU
- **Function:** High-quality image generation through Bittensor, powering real-world applications
- **Applications:** [WOMBO Dream](https://dream.ai), [WOMBO Me](https://wombo.me)
- **Evaluation Method:** Image quality + social engagement metrics (TBD)
- **Additional Tools:** [`rusttensor`](https://github.com/womboai/rusttensor) - generalized Rust interface for subnets

### SN31: NAS Chain
- **Name:** NAS Chain
- **Purpose:** Neural architecture search distribution
- **Evaluation Method:** TBD (research in progress)

### SN32: Its-AI / It's AI
- **Name:** Its-AI (It's AI)
- **Purpose:** Solutions aimed at identifying LLM-generated content
- **Description:** Focuses on detecting AI-generated content amid the rapid growth of Large Language Models
- **Evaluation Method:** TBD (research in progress)

### SN33: ReadyAI / Conversation Genome Project
- **Name:** ReadyAI (Conversation Genome Project)
- **Operator:** 0xai / Afterparty AI
- **Purpose:** Contributing high-quality datasets to open source AI
- **Description:** Addresses scarcity of high-quality datasets, team from Afterparty AI (founded 2021)
- **Evaluation Method:** TBD (research in progress)

### SN34: BitMind
- **Name:** BitMind
- **Purpose:** AI image detection
- **Description:** Utilizes advanced generative and discriminative AI models to detect AI-generated images
- **Evaluation Method:** TBD (research in progress)

### SN35: LogicNet / Cartha
- **Name:** Cartha (previously LogicNet)
- **Operator:** AIT Protocol (original), repurposed in 2025
- **GitHub:** [`LogicNet-Subnet/LogicNet`](https://github.com/LogicNet-Subnet/LogicNet)
- **Purpose:** Decentralized liquidity provisioning (2025) / Mathematical AI (original)
- **Description:** Originally focused on advanced mathematical and logical AI, repurposed in 2025 as Cartha - a decentralized liquidity engine for the 0xMarkets DEX
- **Original Mission:** Fine-tuning competition for AI/ML models specializing in mathematics, computational reasoning, and data analysis
- **Current Function:** Miners rewarded for committing capital; protocols redirect liquidity to forex pairs and volatile markets in real-time
- **Evaluation Method:** Liquidity provision metrics (current) / Mathematical reasoning benchmarks (original)
- **Resources:** [SubnetAlpha - Cartha](https://subnetalpha.ai/subnet/cartha/)
- **Note:** Complete transformation from AI reasoning to DeFi liquidity subnet

### SN36: Web Agents / Pyramid Scheme
- **Name:** Web Agents (Pyramid Scheme)
- **Purpose:** Complex systems analysis
- **Description:** Takes on complexity science's core mysteries: uncovering how simple rules can create complex, seemingly random behavior
- **Evaluation Method:** TBD (research in progress)

### SN37: Finetuning
- **Name:** Finetuning
- **Operator:** Macrocosmos & Taoverse
- **Website:** [macrocosmos.ai/sn37](https://www.macrocosmos.ai/sn37)
- **Purpose:** Fine-tuned model production
- **Description:** Rewards miners for producing fine-tuned models according to competition parameters
- **Function:** Acts like continuous benchmark, rewarding miners for achieving best losses on randomly sampled competition data
- **Evaluation Method:** Loss-based scoring on competition datasets
- **Resources:** [SubnetAlpha - Finetuning](https://subnetalpha.ai/subnet/finetuning-by-macrocosmos-and-taoverse/)

### SN38: Placeholder
- **Name:** TBD (research in progress)
- **Purpose:** TBD (research in progress)
- **Evaluation Method:** TBD (research in progress)

### SN39: Placeholder
- **Name:** TBD (research in progress)
- **Purpose:** TBD (research in progress)
- **Evaluation Method:** TBD (research in progress)

### SN40: Placeholder
- **Name:** TBD (research in progress)
- **Purpose:** TBD (research in progress)
- **Evaluation Method:** TBD (research in progress)

### SN41: SportsTensor
- **Name:** SportsTensor
- **Purpose:** Sports analytics and predictions
- **Description:** Blends advanced technology with sports analytics, develops accurate decentralized sports prediction algorithms
- **Evaluation Method:** TBD (research in progress)

### SN42: Placeholder
- **Name:** TBD (research in progress)
- **Purpose:** TBD (research in progress)
- **Evaluation Method:** TBD (research in progress)

### SN43: Graphite
- **Name:** Graphite
- **Operator:** GraphiteAI
- **GitHub:** [`GraphiteAI/Graphite-Subnet`](https://github.com/GraphiteAI/Graphite-Subnet)
- **Purpose:** Graph optimization problems / Traveling Salesman Problem (TSP) solver
- **Description:** Specialized subnet designed to handle graphical problems efficiently, focusing on the Traveling Salesman Problem - finding shortest routes visiting cities and returning to start
- **Algorithms:** Miners use nearest neighbor heuristic, dynamic programming, A* search, and deep learning approaches
- **Function:** Decentralized network connecting miners and validators to solve complex graph optimization problems
- **Evaluation Method:** Performance-based rewards for high-performance TSP solvers that outperform existing algorithms
- **Significance:** Tackles fundamental computer science challenge with exponential complexity growth
- **Resources:** [Learn Bittensor - Graphite](https://learnbittensor.org/subnets/43)

### SN44: Score Vision / Score Predict
- **Name:** Score Vision (Score Predict)
- **Purpose:** Football match prediction
- **Description:** Incentivizes precise football (soccer) match predictions
- **Evaluation Method:** TBD (research in progress)

### SN45-SN46: Placeholder
- **Name:** TBD (research in progress)
- **Purpose:** TBD (research in progress)
- **Evaluation Method:** TBD (research in progress)

### SN47: Reboot
- **Name:** Reboot
- **Website:** [getreboot.org](https://getreboot.org/)
- **Purpose:** Decentralized robotics AI network
- **Description:** Provides scalable compute power and data for robotics development and AI training
- **Key Features:** Advanced pathfinding, scheduling, optimization algorithms for autonomous robotic decision-making; swarm robotics and collaborative problem-solving
- **Technology:** Distributed AI network enabling peer-to-peer competition, validated through Bittensor consensus
- **Software:** Open-source Python stack (3.11+) with optional Docker support
- **Function:** Miners provide robotics AI solutions, validators score outputs; leverages TAO staking and rewards
- **Evaluation Method:** Bittensor consensus validation of robotic solutions and autonomous agent performance
- **Community:** Open-source development by global AI and robotics researchers

### SN48-SN50: Placeholder
- **Name:** TBD (research in progress)
- **Purpose:** TBD (research in progress)
- **Evaluation Method:** TBD (research in progress)

### SN51: Celium / Lium
- **Name:** Lium (previously Celium)
- **Operator:** commune-ai
- **GitHub:** [`commune-ai/celium`](https://github.com/commune-ai/celium)
- **Purpose:** Decentralized GPU rental platform / compute marketplace
- **Description:** "The AWS/Azure of Bittensor" - connects miners with idle GPUs to users needing compute resources for AI training and scientific computing
- **Key Features:** Affordable GPU rentals, developer-friendly templates (PyTorch, TensorFlow), peer-to-peer network
- **Performance:** Onboarded ~500 Nvidia H100 GPUs in first month (late 2024), became top-emitting subnet (6-7% of TAO emissions by 2025)
- **Function:** Global GPU pool where miners contribute resources, users rent for ML and data analysis
- **Evaluation Method:** GPU quality and performance-based compensation for miners
- **Resources:** [SubnetAlpha - Lium](https://subnetalpha.ai/subnet/lium/)

### SN52-SN61: Placeholder
- **Name:** TBD (research in progress)
- **Purpose:** TBD (research in progress)
- **Evaluation Method:** TBD (research in progress)

### SN62: AgenTao / Ridges AI
- **Name:** Ridges AI (originally AgenTao)
- **Operator:** taoagents
- **GitHub:** [`taoagents/agentao`](https://github.com/taoagents/agentao)
- **Purpose:** Decentralized marketplace for autonomous software agents solving coding challenges
- **Description:** AI agents autonomously solve complex software engineering problems by deconstructing tasks into discrete units (fixing regressions, writing tests, resolving GitHub issues)
- **Core Technology:** Cerebro - learning-based system that classifies task difficulty, supervises solutions, and refines reward models
- **Architecture:** Validators propose and assess tasks; miners compete to deliver best solutions
- **Development Phases:**
  - Epoch 1: Synthetic dataset collection
  - Epoch 2: Real-world GitHub issues
  - Epoch 3: Containerized agent marketplaces
  - Epoch 4: Fully autonomous local development
- **Recent Innovation:** OpenMine - developers log in with Google and submit agents without crypto/mining complexity
- **Evaluation Method:** Cerebro-based task difficulty classification and solution quality scoring
- **Token:** SN62/TAO on Subnet Tokens exchange
- **Resources:** [SubnetAlpha - Ridges AI](https://subnetalpha.ai/subnet/ridgesai/)

### SN63: Placeholder
- **Name:** TBD (research in progress)
- **Purpose:** TBD (research in progress)
- **Evaluation Method:** TBD (research in progress)

### SN64: Chutes
- **Name:** Chutes
- **Operator:** Rayon Labs
- **GitHub:** [`minersunion/sn64-tools`](https://github.com/minersunion/sn64-tools) (community tools)
- **Purpose:** Serverless AI compute / instant AI model hosting
- **Description:** First $100M subnet on Bittensor (9 weeks after dTAO launch) - provides "instant on" AI model deployment in seconds
- **Key Features:** Deploy and run AI models (DeepSeek, Mistral, etc.) via platform or API without managing infrastructure; Docker-based applications
- **Cost Advantage:** 85% less expensive than AWS for AI model hosting
- **Performance:** #1 ranked subnet on Bittensor with $50M+ valuation, powers most DeepSeek free token outputs on OpenRouter
- **Developer:** One of three Rayon Labs subnets (alongside Gradients SN56, Nineteen SN19)
- **Function:** Serverless platform for deploying, running, and scaling any AI model - quick experimentation to production
- **Evaluation Method:** Model deployment performance, uptime, and compute delivery verification
- **Market Position:** Top-performing subnet token with strong monthly gains

### SN65-SN66: Placeholder
- **Name:** TBD (research in progress)
- **Purpose:** TBD (research in progress)
- **Evaluation Method:** TBD (research in progress)

### SN67: Tenex / Tenexium
- **Name:** Tenex (Tenexium / "10x")
- **Operator:** Tenexium
- **GitHub:** [`Tenexium/tenex-subnet`](https://github.com/Tenexium/tenex-subnet)
- **Twitter:** [@Tenex_SN67](https://x.com/Tenex_SN67)
- **Purpose:** Decentralized long-only spot margin protocol / DeFi leverage desk
- **Description:** First native DeFi layer on Bittensor enabling leveraged trading of subnet tokens (alpha tokens) using TAO collateral
- **Key Features:** Users borrow TAO against TAO collateral to amplify exposure on subnet tokens; long-only (no short selling) to avoid sell pressure
- **Dual Benefits:**
  - Traders: Establish leveraged long positions on subnet tokens
  - Liquidity providers: Earn sustainable yields from Bittensor emissions + protocol fees (trading, borrowing, liquidations)
- **Function:** Transforms TAO from staking token into productive financial instrument within Bittensor ecosystem
- **Innovation:** "Bittensor's first native leverage desk" - enables DeFi inside the network
- **Evaluation Method:** Protocol performance, liquidation health, liquidity provision metrics
- **Resources:** [SubnetAlpha - Tenex](https://subnetalpha.ai/subnet/tenex/), [Backprop Finance](https://backprop.finance/dtao/subnets/67-tenexium/liquidityProviders)

### SN68-SN89: Placeholder
- **Name:** TBD (research in progress)
- **Purpose:** TBD (research in progress)
- **Evaluation Method:** TBD (research in progress)
- **Note:** These subnet numbers exist but detailed information is not yet documented. As of December 2025, the network has expanded to include subnets up to approximately SN114, though many are in immunity period or have low emissions.

---

## Subnet Categories (High-Level)

### Content Generation Subnets
- SN1: Text generation (Apex)
- SN3: Text-to-Speech (MyShell)
- SN5: Image generation (evolved)
- SN19: Multi-modal inference (Nineteen)
- SN23: Niche image generation
- SN30: Image generation + social (WomboAI)

### Data & Infrastructure Subnets
- SN27: GPU compute marketplace
- Storage subnets (numbers TBD)
- Data collection subnets (numbers TBD)

### LLM Ecosystem Subnets
- SN1: Inference (Apex)
- SN6: Training/fine-tuning (Nous)
- SN19: Inference at scale (Nineteen)

### DeFi Subnets
- Crypto market analysis subnets (numbers TBD)
- DeFi strategy optimization subnets (numbers TBD)
- Liquidity engine subnets (numbers TBD)

### Specialized Subnets
- Protein folding subnets (numbers TBD)
- Financial predictions (numbers TBD)
- Research tasks (numbers TBD)

---

## Subnet Lifecycle

### Registration
- Anyone can register a new subnet
- Requires burning TAO tokens
- Limited to 32 subnet slots initially (may have expanded)
- Competition for slots based on utility and emissions

### Deregistration
- Subnets can be deregistered if they fail to maintain utility
- Emissions voting determines which subnets survive
- Inactive or low-quality subnets risk removal

### Evolution
- Subnets can change purpose over time
- Examples: SN2 (translation → capital networks), SN5 (image → OpenKaito)
- Operators adapt to market demand and technical feasibility

---

## How to Explore Subnets

### For Users
1. Visit https://taostats.io/subnets for live data
2. Browse subnet code at https://tao.app
3. Check analytics at https://taomarketcap.com
4. Read subnet documentation (varies by subnet)

### For Developers/Miners
1. Review subnet GitHub repositories
2. Study evaluation mechanisms
3. Assess mining profitability (emissions vs compute cost)
4. Join subnet Discord/Telegram communities

### For Validators
1. Evaluate subnet utility and quality
2. Vote on emissions distribution (via SN0 Root)
3. Run validation nodes for chosen subnets
4. Contribute to subnet governance

---

## Common Evaluation Patterns

While each subnet has unique evaluation criteria, common patterns include:

### 1. Reference Answer Comparison
- Validator generates ground truth
- Miners' outputs compared to reference
- Similarity scoring (literal + semantic)
- Used by: SN1 (in part), many text/data subnets

### 2. Adversarial / GAN-Style
- Miners compete against each other
- Dual roles: generator and discriminator
- Zero-sum game drives quality
- Used by: SN1 (Apex)

### 3. Proof of Work/Compute
- Miners prove they delivered compute
- Verification of actual work done
- Payment tied to verified delivery
- Used by: SN27 (Compute), storage subnets

### 4. Quality Scoring Models
- Pre-trained models evaluate outputs
- RLHF, DPO, or similar human preference models
- Multi-dimensional scoring (quality, relevance, safety)
- Used by: SN1, SN19, image generation subnets

### 5. User Feedback
- Real users rate outputs
- Organic demand signals quality
- Market-driven evaluation
- Used by: SN30 (WomboAI), other consumer-facing subnets

---

## Subnet Emissions Distribution

TAO emissions are distributed across subnets based on:

1. **Root Network Voting (SN0):** Validators vote on subnet utility
2. **Subnet Performance:** How well subnet delivers on its mission
3. **Demand Signals:** User adoption and organic usage
4. **Quality Metrics:** Validator assessments of miner outputs
5. **Network Contribution:** Overall value to Bittensor ecosystem

**Result:** Most valuable subnets receive largest emissions, incentivizing quality and utility.

---

## Current Documentation Status

This document now includes **placeholder entries for all subnet numbers** from SN0 through SN89+.

**Fully Documented Subnets (with detailed info):**
- SN0-SN37 (governance + first 37 production subnets with purposes, operators, evaluation methods)
- SN41 (SportsTensor), SN43 (Graphite), SN44 (Score Vision), SN47 (Reboot), SN51 (Lium), SN62 (Ridges AI), SN64 (Chutes), SN67 (Tenex) - all now have complete descriptions

**Placeholder Subnets (awaiting detailed research):**
- SN38-SN40, SN42, SN45-SN46, SN48-SN50, SN52-SN61, SN63, SN65-SN66, SN68-SN89+

**Notable Subnet Evolutions:**
- SN35: LogicNet (mathematical AI) → Cartha (DeFi liquidity) - complete repurposing in 2025
- SN51: Celium → Lium (GPU marketplace) - rebranded but same purpose
- SN62: AgenTao → Ridges AI (software agent marketplace) - rebranded but same purpose

## Incremental Updates Needed

To complete this document, the following information is needed:

- **Detailed descriptions** for placeholder subnets (SN38-SN89+)
- **Operator information** and team backgrounds for all subnets
- **Detailed evaluation mechanisms** for subnets currently marked "TBD"
- **Performance metrics** (emissions, miners, validators) from taostats.io
- **Use cases and applications** for each subnet
- **Integration examples** (how to use subnet outputs)
- **Additional GitHub repositories** and documentation links
- **Community resources** (Discord, Telegram, forums)

---

## Contributing to This Document

To add or update subnet information:

1. Research subnet via [taostats.io/subnets](https://taostats.io/subnets) or subnet GitHub repos
2. Document: Number, Name, Operator, Purpose, Evaluation Method
3. Verify information is current (subnet landscape changes frequently)
4. Replace placeholder entries with detailed information
5. Add sources and links (GitHub, websites, operator info)

**High Priority Updates Needed:**
- Fill in placeholder subnets (SN38-SN40, SN42, SN45-SN46, SN48-SN50, etc.)
- Add detailed evaluation methods for all subnets
- Document higher subnet numbers (SN90-SN114+ exist as of Dec 2025)
- Add performance metrics and emissions data
- Link to subnet Discord/Telegram communities

---

## Sources

- [Subnets - Taostats](https://taostats.io/subnets)
- [Explore Subnets - Bittensor](https://learnbittensor.org/subnets)
- [Subnets - TaoMarketCap](https://taomarketcap.com/)
- [Working with Subnets - Bittensor Docs](https://docs.learnbittensor.org/subnets/working-with-subnets)
- [TAO Community Hub - Twitter/X Subnet Lists](https://x.com/TAOCommunityHub)
- [In-depth analysis of Bittensor: 34 subnets - Bitget](https://www.bitget.com/news/detail/12560604026299)
- [Bittensor Subnets - Bittensor123](https://bittensor123.com/subnets/)
- [6 Top Subnets on Bittensor - Altcoin Buzz](https://www.altcoinbuzz.io/reviews/6-top-subnets-on-bittensor/)

**Document Last Updated:** December 8, 2025

**Note:** This is a living document. Subnet information changes frequently. For most current data, always check taostats.io/subnets or run `btcli subnets list`.
