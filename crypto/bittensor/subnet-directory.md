# Bittensor Subnet Directory

## Overview

This document catalogs all active Bittensor subnets, including their purpose, operators, and evaluation mechanisms. Bittensor currently has **89 subnets** (as of December 2025), covering diverse AI tasks from text generation to compute marketplaces.

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
- **Purpose:** TBD (research in progress)
- **Evaluation Method:** TBD (research in progress)

### SN19: Nineteen
- **Name:** Nineteen
- **Purpose:** Decentralized AI model inference at scale
- **Focus:** Text and image generation
- **Models:** LLaMA 3, Stable Diffusion derivatives
- **Evaluation Method:** TBD (research in progress)
- **Description:** Leading inference subnet providing access to advanced open-source models

### SN23: Niche Image
- **Name:** Niche Image
- **Purpose:** Decentralized image generation
- **Function:** Miners contribute compute to produce images across various models
- **Evaluation Method:** Image quality assessment (specifics TBD)

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

### SN30: WomboAI
- **Name:** WomboAI
- **Operator:** [WOMBO](https://www.wombo.ai) / WOMBO Dream team
- **GitHub:** [`womboai/wombo-bittensor-subnet`](https://github.com/womboai/wombo-bittensor-subnet)
- **Organization:** [`womboai`](https://github.com/womboai)
- **Purpose:** Image generation and social sharing
- **Description:** Powers WOMBO Dream and WOMBO Me apps with ~200K DAU, ~5M MAU
- **Function:** High-quality image generation through Bittensor, powering real-world applications
- **Applications:** [WOMBO Dream](https://dream.ai), [WOMBO Me](https://wombo.me)
- **Evaluation Method:** Image quality + social engagement metrics (TBD)
- **Additional Tools:** [`rusttensor`](https://github.com/womboai/rusttensor) - generalized Rust interface for subnets

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

## To Be Added (Incremental Updates)

This document will be incrementally updated with:

- **Detailed subnet descriptions** for all 89 active subnets
- **Operator information** and team backgrounds
- **Evaluation mechanisms** for each subnet
- **Performance metrics** (emissions, miners, validators)
- **Use cases and applications** for each subnet
- **Integration examples** (how to use subnet outputs)
- **GitHub repositories** and documentation links
- **Community resources** (Discord, Telegram, forums)

---

## Contributing to This Document

To add or update subnet information:

1. Research subnet via taostats.io or subnet GitHub
2. Document: Number, Name, Operator, Purpose, Evaluation Method
3. Verify information is current (subnet landscape changes)
4. Add to appropriate category
5. Include sources and links

**Priority subnets to document next:**
- SN8-SN18 (missing in current research)
- SN20-SN22, SN24-SN26, SN28-SN29 (gaps)
- SN31-SN89 (majority of subnets)
- DeFi subnets (specific numbers TBD)
- Storage subnets (specific numbers TBD)

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
