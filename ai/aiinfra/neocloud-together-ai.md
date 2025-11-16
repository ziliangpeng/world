# Together AI: The Hybrid Neocloud Redefining AI Infrastructure

## 🏢 Company Overview

**Valuation & Market Position**

Together AI stands out in the GPU cloud landscape with a **$3.3 billion valuation** as of February 2025, achieved through a $305 million Series B round led by General Catalyst and Prosperity7 Ventures. What makes Together AI truly unique among neoclouds is its **hybrid business model**: approximately 60-70% of revenue comes from traditional GPU infrastructure (like CoreWeave/Lambda), while 30-40% comes from platform services like managed inference APIs and fine-tuning (like OpenAI's API layer).

This dual identity positions Together AI as the **only major neocloud offering both bare-metal GPU clusters AND managed AI platform services** at scale.

**Funding History**
- **Total Funding**: $534M across 4 rounds
- **Series B (Feb 2025)**: $305M at $3.3B valuation (General Catalyst, Prosperity7 Ventures)
- **Series A2 (Mar 2024)**: $106M at $1.25B valuation (Salesforce Ventures)
- **Series A (Nov 2023)**: $102.5M (Kleiner Perkins, NVIDIA)
- **Key Investors**: NVIDIA, Salesforce Ventures, Kleiner Perkins, Emergence Capital, Lux Capital, Coatue, March Capital, DAMAC Capital, Greycroft

**Scale & Growth Metrics**

- **Revenue**: $130M (end of 2024) → **$300M annualized** (Sept 2025) - **130% YoY growth**
- **GPU Fleet**: 16,000-100,000+ GPUs (H100, H200, B200, GB200 NVL72)
- **Power Capacity**: 200 MW secured
- **Data Centers**: 25+ cities across North America, expanding to Europe (Sweden operational 2025)
- **Model Catalog**: 200+ open-source models available via API
- **Employees**: 197-253 employees (165% growth in recent year)
- **Gross Margins**: ~45%, expected to improve with GPU ownership transition

**Business Model Breakdown**

**Infrastructure (60-70% Revenue):**
- GPU cluster rentals (Instant, Reserved, Frontier AI Factory)
- Hourly/monthly billing for H100, H200, B200, GB200 NVL72 GPUs
- Bare-metal access with Kubernetes or Slurm orchestration
- Traditional neocloud IaaS model

**Platform Services (30-40% Revenue):**
- Serverless Inference API (200+ models, pay-per-token)
- Fine-tuning Platform (LoRA, full fine-tuning, DPO)
- Batch API (50% discount for non-real-time workloads)
- Code Interpreter and evaluation tools
- Managed Dedicated Endpoints

This hybrid model provides revenue stability (infrastructure base) while scaling with customer success (platform usage).

**Market Classification**

- **SemiAnalysis ClusterMAX Rating**: **Silver tier** (v2.0) - alongside AWS, Google, Lambda Labs
- **Note**: Downgraded from Gold tier (v1.0), behind CoreWeave (Platinum), Nebius/Oracle/Azure/Crusoe (Gold)
- **Industry Position**: Unique "infrastructure + platform" hybrid that defies pure categorization

**Customer Base**

- **AI-Native Startups**: Pika Labs (video generation), Cartesia (audio AI), Voyage AI (embeddings), NexusFlow (cybersecurity AI)
- **Fortune 100 Enterprises**: Multiple enterprise customers using VPC deployments for data sovereignty
- **Research Institutions**: Stanford CRFM, ETH Zurich collaborations
- **Developer Community**: Thousands of developers using serverless inference APIs

**Leadership Team**

- **Vipul Ved Prakash** - Co-Founder & CEO
  - Serial entrepreneur with proven exits
  - Founded Topsy (acquired by Apple for $200M+ in 2013)
  - Senior Director at Apple AI/ML (2013-2018)
  - Inventor of Vipul's Razor anti-spam system (MIT TR Top 100 Innovators 2003)

- **Ce Zhang** - Co-Founder & CTO
  - Former Associate Professor at ETH Zurich and University of Chicago
  - PhD from University of Wisconsin-Madison
  - Expertise in distributed systems and ML systems
  - Led DS3 Lab at ETH Zurich

- **Percy Liang** - Co-Founder
  - Associate Professor at Stanford Computer Science
  - Director of Stanford Center for Research on Foundation Models (CRFM)
  - Leading researcher in AI model architectures

- **Chris Re** - Co-Founder
  - Associate Professor at Stanford AI Lab
  - Co-founded Stanford CRFM with Percy Liang
  - Foundational systems research in ML infrastructure

- **Tri Dao** - Chief Scientist
  - Creator of FlashAttention (1, 2, 3)
  - Incoming Assistant Professor at Princeton (2025)
  - Recently completed PhD at Stanford (advised by Christopher Ré and Stefano Ermon)

**Organizational Culture**

- **Research-driven**: ~50% of staff are researchers
- **Academic ties**: Strong connections to Stanford AI Lab, ETH Zurich, University of Chicago
- **Global remote-first**: Teams across North America, Europe, Asia-Pacific, Latin America
- **Headquarters**: San Francisco, CA (584 Castro St #2050)

---

## 📜 Founding Story

**The Stanford Connection (2022)**

Together AI was founded in **June 2022** by four individuals with deep roots in AI research and industry:

- **Vipul Ved Prakash (CEO)**: Serial entrepreneur who founded Cloudmark (anti-spam), Topsy (social analytics acquired by Apple for $200M+), and served as senior director at Apple AI/ML
- **Ce Zhang (CTO)**: Academic systems researcher who led the DS3 Lab at ETH Zurich, with expertise in decentralized training and data-centric ML Ops
- **Percy Liang**: Stanford CS professor and director of the Stanford Center for Research on Foundation Models (CRFM)
- **Chris Re**: Stanford AI Lab professor who co-founded Stanford CRFM with Liang

The founding story began with conversations between Chris Re and Vipul Ved Prakash about reducing model training costs through more efficient systems. When Chris introduced Vipul to Ce Zhang, **"within the first five minutes of talking to Ce, Vipul decided they were starting the company."**

**The Original Vision: Decentralized AI Cloud**

In 2022, the founding team recognized that **foundation models represented a generational shift in technology** requiring open, accessible alternatives to closed systems dominated by big tech. Their initial vision was ambitious:

- **Goal**: Build "the first decentralized cloud dedicated to efficiently working with large foundation models"
- **Tentative Name**: "Together Decentralized Cloud"
- **Concept**: Pool hardware resources including GPUs from volunteers around the internet (similar to distributed computing models like SETI@home)
- **Core Belief**: "Generative models are a consequential technology for society and open and decentralized alternatives to closed systems are going to be critical to enable the best outcomes for AI and society"

The mission was clear: **democratize access to powerful AI** by making open-source models cost-effective and performant enough to compete with proprietary alternatives.

**The Pivot: From Decentralized Dream to Centralized Reality**

Within the first year, the team recognized that the decentralized approach faced fundamental challenges:

- **Performance**: Distributed GPU networks suffered from high latency and inconsistent availability
- **Reliability**: Volunteer-based infrastructure couldn't meet enterprise SLAs
- **Economics**: Centralized high-performance clusters proved more cost-effective than coordinating distributed nodes

**Evolution Timeline:**
- **2022**: Founded with decentralized vision, tentative name "Together Decentralized Cloud"
- **2023**: Shifted to centralized high-performance GPU infrastructure while maintaining open-source focus
  - Initially sourced GPU capacity from CoreWeave, Lambda Labs, and academic institutions
  - Launched RedPajama open dataset (1.2T tokens, reproducing LLaMA training data)
  - Found product-market fit with per-token API pricing as developer experience layer
- **2025**: Transitioned to owning data centers (Maryland operational July 2025, Memphis forthcoming)

Despite the architectural pivot, Together AI **maintained its core values**:
- **Transparency**: Open datasets (RedPajama), open research (FlashAttention), open pricing
- **Open-source commitment**: 200+ open models, no proprietary model lock-in
- **Accessibility**: Cost-effective alternatives to closed-source APIs

**Key Milestones**

- **June 2022**: Company founded
- **2023**: Launched RedPajama-V1 (1.2T tokens, open reproduction of LLaMA training data)
- **May 2023**: Early funding rounds began
- **November 2023**: $102.5M Series A led by Kleiner Perkins, with **NVIDIA as strategic investor**
- **March 2024**: $106M Series A2 led by Salesforce Ventures, $1.25B valuation
- **July 2024**: Released FlashAttention-3, Tri Dao joins as Chief Scientist
- **July 2025**: Maryland data center operational (transition to GPU ownership)
- **September 2025**: Launched Instant GPU Clusters, European expansion (Sweden)
- **February 2025**: $305M Series B, **$3.3B valuation** (163% increase in 11 months)
- **March 2025**: Announced 36,000+ GB200 NVL72 GPU deployment with Hypertec

**The Product-Market Fit Discovery**

Together AI's breakthrough came from recognizing that developers wanted **multiple consumption models**:

1. **Serverless APIs for prototyping**: Pay-per-token, no infrastructure management, rapid iteration
2. **Dedicated clusters for production**: Full control, custom models, enterprise SLAs
3. **Everything in between**: Managed endpoints, batch processing, fine-tuning services

By offering **both infrastructure and platform services**, Together AI captured demand across the entire AI lifecycle that pure neoclouds (CoreWeave, Lambda) and pure API providers (OpenAI, Anthropic) couldn't serve.

**Mission Evolution**

- **Original (2022)**: "Build the first decentralized cloud for large foundation models"
- **Current (2025)**: "Make high-performance AI infrastructure and platform services accessible to everyone through open-source models, cost-efficient compute, and leading-edge research"

The shift from "decentralized" to "high-performance centralized" reflects pragmatic adaptation, while the commitment to accessibility and open-source remains unchanged.

**The Research Heritage**

A defining characteristic of Together AI is its **research-first culture**:

- **~50% of staff are researchers** - far higher than typical cloud providers
- **FlashAttention series**: Tri Dao's groundbreaking attention mechanism optimizations (1.5-2x faster, 75% GPU utilization on H100)
- **RedPajama datasets**: 100T+ tokens of open training data, used by Snowflake Arctic, Salesforce XGen, AI2 OLMo, 500+ community models
- **Together Kernel Collection (TKC)**: Proprietary inference optimizations achieving 4x speedup vs vLLM

This research heritage distinguishes Together AI from infrastructure-focused neoclouds and positions them as **both a cloud provider and an AI research organization**.

---

## 🛠️ Product Lineup

Together AI's product portfolio spans two major categories: **Infrastructure Products** (traditional GPU rentals) and **Platform Services** (managed AI tools). This breadth is unique among neoclouds.

### A. INFRASTRUCTURE PRODUCTS (60-70% Revenue)

**1. Instant GPU Clusters**

On-demand H100 clusters without long-term commitments:

- **Configuration**: 8×H100 80GB SXM5 with NVLink per node
- **Orchestration**: Kubernetes or Slurm on Kubernetes
- **Networking**: InfiniBand 3.2 Tbps, NVLink inter-GPU
- **Pricing**: $2.82-$2.85/GPU/hour (on-demand hourly billing)
- **Features**: Free ingress/egress, spin up/down in minutes, no upfront commitments
- **Use Case**: Experimentation, short-term training, bursty workloads

**2. Reserved Clusters**

Long-term commitments for production workloads:

| GPU Type | Cluster Size | Commitment | Pricing (per GPU/hour) |
|----------|--------------|------------|------------------------|
| H100 80GB | 36-10,000+ GPUs | 1+ months | $1.76-$2.39 (frequency commitment) |
| H200 | Custom | 3+ months | Custom pricing |
| B200 | Custom | 3+ months | $4.00 (long-term) to $5.50 (on-demand) |
| GB200 NVL72 | 72-36,000+ GPUs | 6+ months | Custom enterprise pricing |

**Features:**
- CPU: 96-128 AMD/Intel cores per node
- Memory: 1TB DDR5 per node
- Storage: Configurable NVMe/SSD options
- Networking: 3.2 Tbps InfiniBand, NVLink
- Orchestration: Kubernetes or Slurm on Kubernetes
- Support: Dedicated customer success, SLAs

**3. Frontier AI Factory (GB200 NVL72 Clusters)**

Together AI's flagship infrastructure offering for frontier AI training:

- **Partnership**: Together AI + Hypertec Cloud co-build
- **Scale**: 36,000+ NVIDIA GB200 NVL72 GPUs (2025-2026 deployment)
- **Specification**: 18 nodes × 72 Blackwell GPUs per cluster
- **Memory**: 384 GB HBM3e per GPU
- **Target**: 100,000+ GPUs deployed in 2025
- **Timeline**: Q1 2025 initial deployment, scaling through 2026
- **Use Case**: Frontier model training (GPT-4 scale and beyond), multi-month campaigns

**Technical Infrastructure**

- **Data Center Locations**: 25+ cities, primarily North America + Europe (Sweden operational)
- **Owned Facilities**: Maryland (operational July 2025), Memphis (upcoming GB300 systems)
- **Power Capacity**: 200 MW secured, expanding for 100K+ GPU deployment
- **Strategy**: Transitioning from GPU reseller (CoreWeave/Lambda) to owner-operator for better margins

### B. PLATFORM SERVICES (30-40% Revenue)

**1. Serverless Inference API**

Pay-per-token access to 200+ open-source models without infrastructure management:

**Model Selection:**
- **Language Models**: DeepSeek-R1 (reasoning), Llama 3/3.1/3.2/3.3 (8B-405B), Qwen 2.5/3 (up to 235B), Mistral families
- **Image Generation**: FLUX.1 (schnell, dev, pro), Stable Diffusion variants
- **Code Models**: Specialized coding assistants
- **Audio & Embeddings**: Speech, audio generation, text embeddings

**Performance:**
- **Throughput**: 400+ tokens/second on Llama 3 8B
- **Latency**: Sub-100ms inference
- **Speed**: 4x faster than open-source vLLM, 1.3-2.5x faster than competitors (Amazon Bedrock, Azure AI, Fireworks, Octo AI)

**Pricing Examples:**
- **Llama 3.3 70B**: 11x lower cost than GPT-4o
- **DeepSeek-R1**: 9x lower cost than OpenAI o1
- **Llama 3 8B Lite**: $0.10/million tokens (6x cheaper than GPT-4o-mini)
- **Base pricing**: From $0.06/million tokens

**API Compatibility:**
- OpenAI-compatible endpoints (drop-in replacement)
- Python SDK with async support
- Streaming support for real-time responses
- Rate limiting and throttling controls

**2. Batch API (50% Discount)**

Asynchronous batch processing for non-real-time workloads:

- **Discount**: 50% cost reduction vs real-time inference
- **Capacity**: Up to 50,000 requests per batch (100MB file limit)
- **Completion Time**: 1-12 hours typical (up to 24 hours)
- **Features**: Separate rate limit pools, pay only for successful completions
- **Use Cases**: Evaluations, dataset classification, content generation, data transformations

**3. Fine-Tuning Platform**

Custom model training on Together AI infrastructure:

**LoRA Fine-Tuning:**
- Memory-efficient adapter-based training
- Lightweight checkpoint deployment
- Same base model pricing with multiple adapters
- Serverless Multi-LoRA: Deploy hundreds of adapters at base model cost

**Full Fine-Tuning:**
- Complete custom model training from base model
- Full weight updates and optimization
- Download checkpoints for deployment anywhere
- 100% model ownership

**Advanced Techniques:**
- **DPO (Direct Preference Optimization)**: Alignment without reward models
- **Weights & Biases Integration**: Experiment tracking and monitoring
- **Hyperparameter Tuning**: Automated optimization

**Supported Base Models:**
- Llama families (3, 3.1, 3.2, 3.3)
- Qwen families (2.5, 3)
- Mistral models
- Growing catalog of open-source models

**Pricing:**
- Based on model size and training tokens
- LoRA and full fine-tuning same price (pay for compute, not method)

**4. Managed Dedicated Endpoints**

Private model deployment with guaranteed capacity:

- **Billing**: Per-minute billing (no per-token costs)
- **Customization**: Configurable compute resources (GPU type, count)
- **Performance**: Guaranteed capacity and SLA
- **Use Case**: Production applications requiring consistent performance

**5. Code Execution & Evaluation Tools**

**Together Code Interpreter:**
- Sandboxed Python execution environment
- Library installation support
- File upload and data analysis capabilities
- Pricing: $0.03/session

**Scale & Performance:**
- 100+ concurrent sandboxes
- Thousands of evaluations per minute
- Use cases: RL post-training cycles, model evaluation at scale, agentic workflows

**6. Enterprise Platform Features**

**VPC Deployment:**
- Deploy in your own VPC on any cloud (AWS, Azure, GCP)
- Data sovereignty: All processing within customer-controlled environment
- VPC peering for secure connections
- BYOC-style architecture (Bring Your Own Cloud)
- Performance: 2-3x faster inference, up to 50% lower operational costs

**Security & Compliance:**
- **SOC 2 Type 2 certified**
- **HIPAA compliant** with BAAs available
- Network segmentation and continuous monitoring
- Automated threat detection
- Regular vulnerability assessments and penetration testing
- Encryption in transit and at rest
- Audit logging
- **CCPA compliant**: Does not sell personal information

**Enterprise Support:**
- 99.9% SLA uptime guarantee
- Dedicated customer success representative
- Scale/Enterprise plans with unlimited rate limits
- Priority support and custom integrations

### C. DEVELOPER TOOLS & ECOSYSTEM

**SDKs & APIs**

- **Python SDK v1**: OpenAI-compatible, async support, fine-tuning CLI
  - Installation: `pip install --upgrade together`
- **REST API**: Comprehensive documentation, streaming support
- **Drop-in OpenAI Replacement**: Change base URL to `https://api.together.xyz/v1`

**Integrations**

- **Observability**: Helicone, Arize tracing, Composio
- **Frameworks**: Vercel AI SDK, LangChain, LlamaIndex
- **MLOps**: Standard ML tooling compatibility
- **Vector Databases**: RAG application integrations

**Performance Monitoring**

- Token caching for cost optimization
- Real-time progress tracking for batch jobs
- Usage monitoring and optimization recommendations
- Rate limiting controls

### D. OPEN-SOURCE CONTRIBUTIONS

**RedPajama Initiative**

Together AI's most significant open-source contribution:

**RedPajama-V1** (2023):
- 1.2 trillion tokens
- Open reproduction of LLaMA training data
- 7 data slices: CommonCrawl, C4, GitHub, Books, ArXiv, Wikipedia, StackExchange

**RedPajama-V2** (2024):
- 100 billion+ documents
- 84 CommonCrawl snapshots
- 30 billion documents with quality signals
- 100T+ tokens across multiple domains

**Impact:**
- Used by Snowflake Arctic, Salesforce XGen, AI2 OLMo
- 500+ community models built on RedPajama datasets
- Standard training datasets for open-source AI community

**Together Kernel Collection (TKC)**

Proprietary inference optimizations:

- **FlashAttention-3**: 1.5-2x faster than FA-2, 740 TFLOPS on H100 (75% utilization)
- **FP8 Support**: Near 1.2 PFLOPS with 2.6x smaller error than baseline FP8
- **Training**: Up to 10% faster
- **Inference**: Up to 75% faster
- **Custom Speculators**: Based on RedPajama datasets
- **Quantization**: Market-leading INT4, FP8 implementations

---

## 💎 Value Proposition & Differentiation

### A. The Hybrid Model Advantage: Infrastructure + Platform

Together AI's **unique positioning as the only major neocloud offering both bare-metal GPU clusters AND managed platform services** provides several strategic advantages:

**Full-Stack AI Lifecycle Coverage:**

1. **Prototyping**: Serverless Inference API with pay-per-token pricing, no infrastructure management
2. **Experimentation**: Instant GPU Clusters for custom model training, hourly billing
3. **Fine-Tuning**: Managed fine-tuning platform with LoRA/full training options
4. **Production**: Reserved Clusters + Managed Dedicated Endpoints for scale
5. **Enterprise**: VPC deployment for data sovereignty and compliance

**Customer Benefits:**

- **Single Vendor Simplicity**: Unified billing, account management, support across entire AI workflow
- **Seamless Transitions**: Start with Serverless API, graduate to Dedicated Clusters as needs grow
- **Flexibility**: Choose infrastructure control (DIY) or platform convenience (managed) based on specific use case
- **Cost Optimization**: Pay-per-token for variable workloads, hourly for predictable workloads

**Revenue Stability:**

- **Infrastructure base (60-70%)**: Predictable committed revenue from long-term cluster reservations
- **Platform growth (30-40%)**: Usage-based scaling as customer workloads grow
- **Higher margins on platform**: Software layer reduces cost relative to raw infrastructure

### B. Cost Competitiveness: The 6-11x Advantage

**vs Closed-Source API Providers (OpenAI, Anthropic)**

Together AI's most compelling value proposition is **dramatic cost savings** for equivalent open-source models:

| Comparison | Together AI Cost | Competitor Cost | Savings |
|------------|------------------|-----------------|---------|
| Llama 3.3 70B vs GPT-4o | Together: $X | OpenAI: $Y | **11x cheaper** |
| DeepSeek-R1 vs OpenAI o1 | Together: $X | OpenAI: $Y | **9x cheaper** |
| Llama 3 8B Lite vs GPT-4o-mini | $0.10/M tokens | $0.60/M tokens | **6x cheaper** |

**Additional Cost Benefits:**
- **Batch API**: Additional 50% discount for non-real-time workloads
- **Model Ownership**: Fine-tune once, deploy forever without ongoing API costs
- **No Lock-In**: Download checkpoints, deploy on-premises or any cloud

**vs Hyperscalers (AWS, Azure, GCP)**

- **Claimed savings**: ~80% cost reduction (context-dependent, primarily for GPU workloads)
- **No data egress fees**: Free ingress/egress on GPU clusters (hyperscalers charge $0.08-$0.12/GB)
- **AI-optimized infrastructure**: Specialized for GPU workloads, no "flexibility premium"
- **Transparent pricing**: No hidden costs, complex commitment structures, or tiered discounts

**vs Other Neoclouds (CoreWeave, Lambda, Crusoe)**

Together AI's **infrastructure pricing is competitive** with other neoclouds:

| Provider | H100 Pricing (On-Demand) | H100 Pricing (Reserved) |
|----------|--------------------------|-------------------------|
| Together AI | $2.82-$2.85/GPU/hour | $1.76-$2.39/GPU/hour |
| Lambda Labs | $2.49/GPU/hour | N/A (limited availability) |
| Vultr | $2.99/GPU/hour | $2.30/GPU/hour |
| CoreWeave | $3.00-$4.24/GPU/hour | Custom enterprise pricing |

**Differentiation**: Platform layer (Serverless API, fine-tuning) provides value beyond raw infrastructure that pure neoclouds don't offer.

### C. Performance Leadership: 4x Faster Inference

**Together Inference Stack Benchmarks:**

- **4x faster than open-source vLLM** (industry standard)
- **1.3-2.5x faster than commercial competitors** (Amazon Bedrock, Azure AI, Fireworks, Octo AI)
- **400+ tokens/second** on Llama 3 8B
- **Sub-100ms latency** for inference requests
- **Dynamic optimization**: Balances tokens/sec vs overall throughput based on traffic patterns

**Technical Innovations Enabling Performance:**

**FlashAttention Architecture:**
- Linear memory complexity (O(N)) vs quadratic (O(N²)) standard attention
- Tiling and recomputation: Optimizes HBM ↔ SRAM data movement
- 1.5-2x faster than FlashAttention-2
- 740 TFLOPS on H100 (75% GPU utilization vs ~50% typical)

**FP8 Low-Precision Arithmetic:**
- Near 1.2 PFLOPS throughput with FP8
- 2.6x smaller quantization error than baseline FP8
- Maintains model quality while dramatically increasing throughput

**Advanced Inference Techniques:**
- Speculative decoding with custom speculators trained on RedPajama
- Dynamic batching for optimal GPU utilization
- Model quantization (INT4, FP8) without significant accuracy loss
- Asynchronous processing and pipelining

**Training Performance:**
- **10% faster training** with Together Kernel Collection
- **75% faster inference** vs baseline implementations
- **Custom kernel library** optimized for NVIDIA H100/H200/B200 architectures

### D. Open-Source Commitment: Transparency & Flexibility

**Philosophy: Open Alternatives to Closed Systems**

Together AI's founding mission—democratizing AI through open-source—remains central to its value proposition:

**RedPajama Datasets (100T+ Tokens):**
- Complete transparency in data curation
- Open reproduction of proprietary training datasets (LLaMA)
- Used by 500+ community models (Snowflake Arctic, Salesforce XGen, AI2 OLMo)
- Ongoing contributions (V1, V2, future V3 with multimodal data)

**Model Selection & Ownership:**
- **200+ open-source models** spanning language, image, audio, code
- **No proprietary lock-in**: Use any model or bring your own
- **Rapid updates**: New releases added quickly (DeepSeek-R1, Llama 3.3, etc.)
- **100% ownership**: Fine-tune and download checkpoints for deployment anywhere

**Research Contributions:**
- **FlashAttention series** (Tri Dao): Foundational optimization for efficient attention
- **Together Kernel Collection**: Performance optimizations shared with community
- **Academic partnerships**: Stanford CRFM, ETH Zurich collaborations
- **~50% researcher staff**: Publication-driven culture

**Community Benefits:**
- Multi-LoRA: Run hundreds of custom variants at base model cost
- Weights & Biases integration for experiment tracking
- Compatible with LangChain, LlamaIndex, standard ML frameworks
- Open documentation and examples

### E. Developer Experience: Simplicity Meets Power

**Ease of Use:**

- **OpenAI-compatible APIs**: Drop-in replacement, change base URL only
- **Single API for 200+ models**: Unified interface across language, image, audio models
- **Python SDK with async support**: Production-ready tooling
- **Comprehensive documentation**: Examples, tutorials, API reference
- **Fast onboarding**: NexusFlow running workloads in <90 minutes

**Flexible Deployment Models:**

1. **Serverless**: For prototyping, variable workloads, no infrastructure management
2. **Dedicated Endpoints**: For production reliability, guaranteed capacity
3. **Reserved Clusters**: For custom training, full infrastructure control
4. **VPC Deployment**: For enterprise security, data sovereignty
5. **Hybrid**: Mix and match based on specific workload requirements

**Performance Monitoring & Optimization:**

- Observability integrations (Helicone, Arize)
- Token caching for cost reduction
- Real-time usage monitoring
- Optimization recommendations
- Rate limiting controls

### F. Enterprise-Grade Capabilities

**Security & Compliance:**

- **SOC 2 Type 2**: Independently audited security controls
- **HIPAA**: Healthcare data protection with Business Associate Agreements
- **VPC Deployment**: Data never leaves customer's controlled environment
- **Encryption**: In transit (TLS) and at rest
- **Audit Logging**: Complete activity tracking
- **Threat Detection**: Automated monitoring and alerting
- **Network Segmentation**: Isolated workloads
- **CCPA Compliant**: Does not sell personal information

**Scale & Reliability:**

- **99.9% SLA**: Uptime guarantee for enterprise customers
- **200 MW power capacity**: Supporting 100K+ GPU deployment
- **25+ city global footprint**: Geographic distribution for low latency
- **Redundancy & Failover**: High availability architecture
- **Independent batch rate limits**: No contention with real-time traffic

**Enterprise Support:**

- Dedicated customer success representative
- Unlimited rate limits on Scale/Enterprise plans
- Custom SLAs and support agreements
- Private Slack/Discord channels
- Quarterly business reviews

### G. Customer Success Stories

**Pika Labs (Video Generation Unicorn)**

- **Challenge**: Scale text-to-video model from prototype to millions of videos/month
- **Solution**: Started with Together Inference API for prototyping, transitioned to GPU Clusters for custom model training
- **Results**: Scaled to millions of videos/month within 6 months, top users spending ~10 hours/day on platform
- **Quote**: "Together AI's flexibility let us start fast and scale seamlessly"

**NexusFlow (Cybersecurity AI)**

- **Challenge**: Reduce R&D cloud compute costs while maintaining performance
- **Solution**: Migrated to Together GPU Clusters for model development
- **Results**:
  - Onboarded and running workloads in **<90 minutes**
  - **40% reduction** in R&D cloud compute costs
  - Faster response times and lower latency
- **Quote**: "Together AI delivered immediate cost savings and better performance"

**Fortune 100 Enterprises**

- **Use Cases**: Training, fine-tuning, and inference at scale with governance and compliance requirements
- **Solution**: VPC deployments for data sovereignty
- **Benefits**:
  - Data processing within customer-controlled environment
  - 2-3x faster inference vs hyperscaler alternatives
  - Up to 50% lower operational costs
  - SOC 2 Type 2 and HIPAA compliance

**Voyage AI, Cartesia**

- Customers using infrastructure and platform services
- Specific metrics not publicly disclosed
- Representative of AI-native startups leveraging hybrid model

### H. Competitive Positioning Matrix

**vs Pure Neoclouds (CoreWeave, Lambda, Crusoe, Nebius)**

| Factor | Together AI | Pure Neoclouds |
|--------|-------------|----------------|
| GPU Infrastructure | ✓ (60-70% revenue) | ✓ (100% revenue) |
| Platform Services | ✓ (30-40% revenue) | ✗ |
| Pricing Flexibility | Hourly + Per-Token | Hourly only |
| ClusterMAX Rating | Silver | Platinum/Gold |
| Developer Experience | API-first | Infrastructure-first |
| Enterprise Focus | Growing | Established (CoreWeave) |

**Advantage**: Platform layer differentiation, flexible consumption models
**Disadvantage**: Lower ClusterMAX rating, less infrastructure specialization

**vs API Providers (OpenAI, Anthropic, Google)**

| Factor | Together AI | Closed-Source APIs |
|--------|-------------|-------------------|
| Cost | 6-11x cheaper | Premium pricing |
| Model Selection | 200+ open-source | Proprietary only |
| Customization | Full fine-tuning, ownership | Limited fine-tuning |
| Data Sovereignty | VPC deployment | Shared infrastructure |
| Transparency | Open datasets, research | Closed systems |

**Advantage**: Massive cost savings, model ownership, transparency
**Disadvantage**: No frontier proprietary models (GPT-4o, Claude Opus)

**vs Hyperscalers (AWS, Azure, GCP)**

| Factor | Together AI | Hyperscalers |
|--------|-------------|-------------|
| AI-Optimized | ✓ Purpose-built | Generic cloud |
| Cost | ~80% savings | Premium pricing |
| Complexity | Simple, focused | 200+ services |
| GPU Access | Fast (GB200 in 2025) | Slower (enterprise priority) |
| Ecosystem | AI-specific | Broad enterprise |

**Advantage**: Cost, AI specialization, simplicity, latest GPU access
**Disadvantage**: No broad service catalog (databases, networking, DevOps)

**Unique Value Proposition Summary**

Together AI occupies a **unique position in the market** as the only provider offering:

1. **Both infrastructure and platform** at scale (60-70% / 30-40% revenue split)
2. **Cost leadership** for open-source AI (6-11x vs OpenAI, ~80% vs hyperscalers)
3. **Performance leadership** for inference (4x faster than vLLM, 1.3-2.5x vs competitors)
4. **Open-source commitment** with research contributions (FlashAttention, RedPajama)
5. **Enterprise capabilities** with developer experience (VPC deployment, SOC 2, HIPAA, 99.9% SLA)

This hybrid model enables Together AI to **capture demand across the full AI lifecycle** that pure infrastructure plays (CoreWeave, Lambda) and pure API plays (OpenAI, Anthropic) cannot individually serve.

---

## 🗺️ Future Roadmap & Strategic Direction

### A. GPU Ownership Strategy: The Margin Expansion Play

**Historical Model: GPU Reseller (2022-2024)**

- Sourced GPU capacity from CoreWeave, Lambda Labs, academic institutions
- Acted as intermediary: Added platform services on top of third-party infrastructure
- **Gross margins**: ~45% (limited by reseller economics)

**Current Transition: Owner-Operator (2025-2027)**

- **Maryland data center**: Operational July 2025 (first owned facility)
- **Memphis facility**: Forthcoming with GB300 systems
- **Financial impact**: Higher upfront capex, but significantly better long-term margins
- **Operational benefits**:
  - Guaranteed capacity (no dependency on third-party availability)
  - Faster provisioning (direct control over hardware)
  - Better SLAs (eliminate reseller risk)
  - Margin expansion (eliminate reseller markup)

**Target State: Hybrid Owned + Partner (2026+)**

- Own strategic capacity in key markets (Maryland, Memphis, Europe)
- Partner with Hypertec for massive GB200 deployment (36K+ GPUs)
- Maintain flexibility: Owned for core, partners for scale and geographic expansion

### B. NVIDIA Partnership & Blackwell Deployment

**Strategic Alliance with NVIDIA**

- **NVIDIA as investor**: Series A (November 2023)
- **AI Infrastructure America**: Part of NVIDIA's initiative to scale US AI infrastructure
- **Early access**: First/early access to Blackwell GB200 and B200 GPUs
- **Technical collaboration**: Together Kernel Collection optimizations + NVIDIA CUTLASS/CuTe libraries

**GB200 NVL72 "Frontier AI Factory" Deployment**

**Partnership Structure:**
- Together AI + Hypertec Cloud co-build
- NVIDIA GPU supply and technical support

**Scale:**
- **36,000+ NVIDIA GB200 NVL72 GPUs** (announced March 2025)
- **18-node clusters** × 72 Blackwell GPUs per cluster = **2,000+ clusters**
- **384GB HBM3e** per GPU (vs H100's 80GB = 4.8x memory)

**Timeline:**
- Q1 2025: Initial deployment announced
- March 2025: First B200 cluster deployed
- 2025: Scaling to 36K+ GPUs
- 2026: Continued expansion, additional facilities

**Strategic Importance:**
- **Frontier AI training**: Enable GPT-4 scale and beyond model training
- **Competitive positioning**: Among first cloud providers with GB200 at scale (alongside CoreWeave, Nebius)
- **Customer attraction**: Enterprises building frontier models need latest hardware
- **Margin improvement**: Newest GPUs command premium pricing

**Hardware Roadmap Beyond Blackwell:**

- **Current generation**: H100, H200 (widely deployed)
- **2025**: B200, GB200 NVL72 (deploying)
- **2026+**: GB300 systems mentioned for Memphis facility
- **Future**: Early access to next-gen NVIDIA architectures via strategic partnership

### C. Infrastructure Expansion: Path to 100K+ GPUs

**Current State (2025):**
- **GPU fleet**: 16,000-100,000+ GPUs deployed (exact number varies by quarter)
- **Locations**: 25+ cities across North America
- **Power capacity**: 200 MW secured

**2025 Targets:**
- **100,000+ GPUs**: Together AI + Hypertec partnership
- **European expansion**: Sweden operational (2025), additional EU locations planned
- **New data centers**: Memphis facility coming online

**2026-2027 Projections:**
- **Geographic distribution**: Expand from 25+ cities to broader global footprint
- **Power capacity**: Expand beyond 200 MW to support larger fleet
- **New markets**: Potential Asia-Pacific expansion (not yet announced)
- **Strategy**: Follow customer demand, power availability, and data sovereignty requirements

**Capital Requirements & Financing:**

- **$305M Series B (Feb 2025)**: Provides runway for initial expansion
- **GPU capex**: $30K-$40K per H100/H200, higher for GB200 (~$50K-$70K estimated)
- **Data center buildout**: Significant additional capex for facilities, power, networking
- **Future fundraising**: Likely additional rounds or debt financing for sustained growth

### D. Platform Service Expansion: Growing the 30-40%

**Current Platform Revenue (30-40%):**
- Serverless Inference API (200+ models)
- Fine-tuning Platform (LoRA, full fine-tuning)
- Batch API, Code Interpreter
- Managed Dedicated Endpoints

**Inference Optimization Roadmap:**

**FlashAttention Evolution:**
- **FlashAttention-4**: Expected from Tri Dao's ongoing research
- **FlashAttention-5+**: Continued optimization for future GPU architectures (GB200, GB300)
- **Multi-modal FlashAttention**: Optimizations for vision, audio, multimodal models

**Expanded Model Catalog:**
- **Current**: 200+ models
- **Target**: Broader coverage across modalities (language, vision, audio, video, multimodal)
- **Rapid updates**: Continue adding latest open-source releases (Meta, Mistral, DeepSeek, Qwen, etc.)
- **Community models**: Support models built on RedPajama datasets

**Advanced Platform Features:**

**Agentic AI Support:**
- Code Interpreter expansion for more complex workflows
- Tool use and function calling optimizations
- Memory and state management for long-running agents
- Multi-agent orchestration

**Fine-Tuning Enhancements:**
- More base models supported (beyond Llama, Qwen, Mistral)
- Advanced techniques: RLHF, PPO, other alignment methods beyond DPO
- Synthetic data generation for training
- Automated hyperparameter tuning
- Dataset curation tools (building on RedPajama expertise)

**MLOps & Developer Tools:**
- Expanded observability integrations (DataDog, New Relic, etc.)
- Workflow orchestration (Airflow, Prefect, etc.)
- Model versioning and deployment management
- A/B testing infrastructure for models
- Cost optimization recommendations and automated scaling

**Vertical Solutions:**
- Industry-specific models and workflows (healthcare, finance, legal, cybersecurity)
- Compliance-focused offerings (HIPAA, SOC 2, FedRAMP for regulated industries)
- Managed services for common use cases (RAG, summarization, classification)

### E. Enterprise Growth: Moving Upmarket

**Current Customer Mix (2024-2025):**
- **Primary**: Individual developers and AI-native startups (Pika Labs, NexusFlow, Cartesia, Voyage AI)
- **Growing**: Fortune 100 enterprises using VPC deployments

**Enterprise Strategy (2025-2027):**

**VPC Deployment Expansion:**
- **Current**: Deploy in customer's VPC on AWS, Azure, GCP
- **Future**: Expanded BYOC (Bring Your Own Cloud) capabilities
- **On-premises**: Potential on-prem deployment option for highly regulated industries
- **Hybrid**: Seamless workload migration between Together Cloud and customer infrastructure

**Compliance & Certifications:**
- **Achieved**: SOC 2 Type 2, HIPAA
- **Potential**: FedRAMP (US government), ISO 27001 (international), PCI DSS (payments)
- **Regional**: EU AI Act compliance, GDPR enhancements

**Enterprise Platform Features:**
- **Scale/Enterprise tiers**: Unlimited rate limits, 99.9% SLA
- **Dedicated support**: Customer success representatives, private Slack/Discord channels
- **Custom integrations**: Enterprise SSO, audit logging, advanced security features
- **Quarterly business reviews**: Strategic account management

**Go-to-Market:**
- Build enterprise sales team (currently lean)
- Expand partnerships (AWS Marketplace, Azure Marketplace, GCP Marketplace)
- Industry-specific solutions (healthcare AI, financial AI, legal AI)

**Target Customer Mix (2027):**
- AI-native startups: 40-50% (maintain strong base)
- Mid-market: 20-30% (expand)
- Enterprise: 30-40% (significant growth)

### F. Open-Source & Research Leadership

**RedPajama Roadmap:**

**RedPajama-V3 (Expected 2025-2026):**
- Expanded domains beyond text (code, multimodal, specialized datasets)
- Larger scale (beyond 100T tokens)
- Improved quality signals and filtering
- Community collaboration (continue partnerships with Stanford CRFM, ETH Zurich, etc.)

**Research Agenda:**

**Inference Optimization:**
- FlashAttention-4, FlashAttention-5+ (Tri Dao's ongoing work)
- Quantization research (INT4, FP8, lower precision)
- Pruning and distillation for efficient deployment
- Speculative decoding improvements

**Training Efficiency:**
- Distributed training optimizations
- Gradient compression and communication efficiency
- Memory-efficient training techniques
- Multi-node scaling (10K-100K+ GPUs)

**Novel Architectures:**
- Monarch Mixer explorations
- State space models (Mamba variants)
- Hybrid architectures combining transformers and alternatives
- Multimodal architecture research

**Systems Research:**
- Scheduling and resource allocation
- Memory management optimizations
- Networking and communication (InfiniBand, Ethernet)
- Fault tolerance for long-running training campaigns

**Academic Partnerships:**

- **Stanford CRFM**: Ongoing collaboration (Percy Liang, Chris Re)
- **ETH Zurich**: Continued through Ce Zhang network
- **Princeton**: Tri Dao joining as Assistant Professor (potential collaboration)
- **Open publications**: Maintain research-driven culture with public contributions

### G. Market Positioning Evolution

**Infrastructure vs Platform Revenue Balance:**

- **Current (2025)**: 60-70% infrastructure, 30-40% platform
- **Trend**: Platform services growing faster than infrastructure (higher margins)
- **Strategic Goal**: Maintain both, gradually increase platform share to 40-50% by 2027
- **Rationale**:
  - Higher margins on platform (software layer)
  - Stronger customer lock-in (API usage vs raw infrastructure)
  - Differentiation vs pure neoclouds

**Competitive Positioning Strategy:**

**Against Pure Neoclouds (CoreWeave, Lambda, Crusoe, Nebius):**
- **Emphasize**: Platform differentiation (inference API, fine-tuning, developer tools)
- **Messaging**: "Full-stack AI platform, not just GPU rental"
- **Target**: Customers wanting both infrastructure flexibility and platform convenience

**Against API Providers (OpenAI, Anthropic, Google):**
- **Emphasize**: Cost savings (6-11x), open-source flexibility, VPC deployment, model ownership
- **Messaging**: "Open-source AI with enterprise capabilities at fraction of closed-source cost"
- **Target**: Cost-conscious enterprises, customers wanting data sovereignty

**Against Hyperscalers (AWS, Azure, GCP):**
- **Emphasize**: AI-optimized infrastructure, cost-efficiency (~80% savings), simplicity
- **Messaging**: "Purpose-built for AI, not general-purpose cloud with AI bolted on"
- **Target**: AI-first companies prioritizing performance and cost over broad service catalog

**Against Specialized AI Platforms (Baseten, Modal, Replicate):**
- **Emphasize**: Full-stack approach (infrastructure + platform), scale (100K+ GPUs), research leadership
- **Messaging**: "Complete AI lifecycle from training to production, not just inference"
- **Target**: Enterprises building frontier models, not just deploying existing models

**Geographic Expansion Strategy:**

- **North America (2025-2026)**: Consolidate leadership, expand data center footprint (Maryland, Memphis, additional cities)
- **Europe (2025-2027)**: Active expansion (Sweden operational, additional EU locations planned 2026)
- **Asia-Pacific (2027+)**: Potential future expansion following customer demand
- **Strategy**: Follow power availability, customer demand, and data sovereignty requirements

### H. Strategic Priorities & Execution Plan

**2025 Priorities:**

1. **Scale infrastructure**: Deploy 100,000+ GPUs (Together + Hypertec GB200 partnership)
2. **Transition to ownership**: Maryland and Memphis data centers operational, improve margins
3. **European expansion**: Sweden operational, additional EU locations
4. **Enterprise growth**: Expand VPC deployments, achieve additional compliance certifications
5. **Platform differentiation**: Maintain inference performance leadership (4x vLLM), expand model catalog

**2026 Priorities:**

1. **Margin expansion**: Realize benefits of GPU ownership, target 55-60%+ gross margins (up from 45%)
2. **Enterprise penetration**: Grow Fortune 100 customer base, build enterprise sales team
3. **Platform revenue growth**: Increase platform from 30-40% to 40-50% of total revenue
4. **RedPajama-V3**: Release next-generation datasets (multimodal, specialized domains)
5. **ClusterMAX upgrade**: Target Gold tier ranking (up from Silver)

**2027 Vision:**

- **Revenue**: $300M (2025) → $600M-$1B+ trajectory
- **GPU fleet**: 100K (2025) → 200K+ potential
- **Gross margins**: 45% (2024) → 60%+ with ownership and platform mix
- **Enterprise revenue**: Grow from <20% to 30-40% of total
- **Geographic footprint**: 25+ cities (2025) → 40+ cities (North America, Europe, potential APAC)
- **Market position**: Recognized leader in open-source AI infrastructure + platform

### I. Risks & Challenges

**Competitive Dynamics:**

- **CoreWeave (Platinum tier)** and **Nebius (Gold tier)** have stronger ClusterMAX infrastructure ratings
- **Hyperscalers** (AWS, Azure, GCP) adding AI-optimized infrastructure and reducing GPU pricing
- **API providers** (OpenAI, Anthropic) may reduce pricing as models commoditize
- **Mitigation**: Hybrid model (infrastructure + platform), open-source focus, cost leadership, research differentiation

**Technology Evolution:**

- **GPU architecture changes**: Post-Blackwell architectures may require significant re-optimization
- **Open-source model convergence**: If open models fully match closed models, differentiation weakens
- **Inference optimization commoditization**: If vLLM/competitors match Together's 4x advantage
- **Mitigation**: Research culture (Tri Dao, FlashAttention series), early NVIDIA access, continuous innovation

**Capital Requirements:**

- **GPU ownership**: Requires massive capex ($30K-$70K per GPU × 100K = $3B-$7B total)
- **Data center buildout**: Significant additional capex for facilities, power, networking
- **Competitor funding**: CoreWeave IPO potential, Nebius $700M NVIDIA funding, hyperscaler infinite capital
- **Mitigation**: $305M Series B provides runway, partnerships (Hypertec, power providers), potential future fundraising or IPO

**Execution Risks:**

- **Enterprise sales cycles**: Longer and more complex than startup motion, requires team buildout
- **Ownership transition**: Operational complexity of running data centers vs reselling
- **ClusterMAX rating**: Silver tier may hurt enterprise credibility vs Gold/Platinum competitors
- **Mitigation**: Proven leadership team (Vipul's Topsy exit, Stanford pedigree), focus on developer experience, gradual enterprise expansion

**Market Dynamics:**

- **Potential AI recession**: If AI hype dies down, GPU demand craters
- **Regulatory risks**: AI regulation, data privacy laws, export controls
- **Energy costs**: Power prices and availability constrain GPU deployment
- **Mitigation**: Dual customer base (startups + enterprise), platform services less capital-intensive, open-source focus reduces regulatory risk

### J. 2025-2027 Concrete Milestones

**Q2-Q3 2025:**
- Maryland data center fully operational
- European expansion (Sweden + 1-2 additional EU locations)
- GB200 initial deployments (first clusters)
- Enterprise customer wins announced (Fortune 100 logos)

**Q4 2025:**
- 100,000+ GPUs deployed (Together + Hypertec)
- Memphis facility operational with GB300 systems
- RedPajama-V3 announced/released
- Platform revenue reaches 40% of total

**2026:**
- Gross margins improve to 55-60%+ with GPU ownership
- ClusterMAX rating upgrade to Gold tier (target)
- 150K-200K GPUs deployed
- Enterprise revenue reaches 30% of total
- FlashAttention-4 released
- Additional compliance certifications (FedRAMP potential, ISO 27001)

**2027:**
- $600M-$1B revenue run rate
- 200K+ GPUs operational
- 40+ global locations
- Enterprise revenue 30-40% of total
- Recognized as leading open-source AI infrastructure platform
- Potential IPO consideration

### K. Long-Term Vision: The Open-Source AI Cloud

**Strategic North Star (2027-2030):**

Become the **default AI acceleration cloud for open-source AI**, combining:

1. **Best-in-class GPU infrastructure** (100K-200K+ GPUs, latest NVIDIA/AMD hardware)
2. **Leading platform services** (inference API, fine-tuning, MLOps tools)
3. **Cost-performance leadership** (6-11x vs closed-source, ~80% vs hyperscalers)
4. **Research-driven innovation** (FlashAttention series, RedPajama datasets, novel architectures)
5. **Enterprise-grade capabilities** (SOC 2, HIPAA, FedRAMP, 99.9% SLA, VPC deployment)

**Market Positioning Goal:**

- **Infrastructure**: Move from Silver to Gold tier in ClusterMAX, recognized as top 5 GPU cloud
- **Platform**: Leading open-source AI inference provider (top 3 by volume)
- **Hybrid**: Unique positioning combining both at scale
- **Brand**: Synonymous with "open-source AI at scale" (like Hugging Face for models, Together AI for infrastructure+platform)

**If Successful, Together AI Could:**

- **$10B+ valuation** by 2028-2030 (from $3.3B in 2025)
- **$1B-$2B+ revenue** by 2030 (from $300M in 2025)
- **Top 3 neocloud** by revenue and GPU count
- **IPO candidate** in 2027-2029 timeframe
- **Category leader** in "hybrid infrastructure + platform for open-source AI"

---

## Key Takeaways

**Together AI's Unique Position:**

Together AI represents the **only major neocloud combining infrastructure (60-70% revenue) and platform services (30-40% revenue)** at scale. This hybrid model enables them to capture demand across the full AI lifecycle—from prototyping with serverless APIs to production training on 100K+ GPU clusters—that pure infrastructure plays (CoreWeave, Lambda) and pure API plays (OpenAI, Anthropic) cannot individually serve.

**The Three Strategic Pillars:**

1. **Cost Leadership**: 6-11x cheaper than closed-source APIs, ~80% savings vs hyperscalers
2. **Performance Leadership**: 4x faster inference than vLLM, sub-100ms latency, 400+ tokens/sec
3. **Open-Source Commitment**: RedPajama datasets (100T+ tokens), FlashAttention innovations, 200+ models

**Competitive Advantages:**

- **Hybrid model flexibility**: Infrastructure + platform services under one roof
- **Research-driven innovation**: ~50% staff researchers, Tri Dao as Chief Scientist, Stanford/ETH partnerships
- **NVIDIA strategic partnership**: Early GB200 access, $305M Series B co-led with AMD Ventures
- **Enterprise capabilities**: SOC 2 Type 2, HIPAA, VPC deployment, 99.9% SLA
- **Developer experience**: OpenAI-compatible APIs, Python SDK, comprehensive documentation

**The Path Forward:**

Together AI's success hinges on executing three major transitions: (1) **GPU ownership** to improve margins from 45% to 60%+, (2) **enterprise growth** to balance startup customer base, and (3) **platform revenue expansion** from 30-40% to 40-50%+ of total revenue. With $305M Series B funding, 100K+ GPU deployment plans, and the NVIDIA GB200 partnership, Together AI is positioned to become a **$1B+ revenue, top 3 neocloud by 2027-2030**.

**The Hybrid Model Bet:**

If the hybrid infrastructure + platform model proves superior to pure infrastructure plays (CoreWeave's Platinum tier specialization) or pure API plays (OpenAI's proprietary model focus), Together AI could emerge as the **category-defining company for open-source AI at scale**—capturing the massive middle market between DIY infrastructure and closed-source APIs.
