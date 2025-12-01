# Scale AI - Technology Stack

**Company:** Scale AI, Inc.
**Founded:** 2016
**Focus:** Data labeling, RLHF, and GenAI platform for enterprise AI training
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

Scale AI operates from **San Francisco** with infrastructure certified **FedRAMP, ISO, and AICPA SOC compliant** for government and enterprise deployments. Founded in **2016** by **Alexandr Wang** (19, MIT dropout studying mathematics/CS) and **Lucy Guo** (fired 2018) through **Y Combinator**, the company raised **$1.6 billion total funding** reaching **$29 billion valuation** after **Meta acquired 49% stake for $14.8 billion** (June 2025). Wang became the **world's youngest self-made billionaire at age 24** (2021, net worth $3.6B at 28) and joined Meta as part of the deal, with **Jason Droege** (former Uber executive, chief strategy officer) replacing him as CEO. The company operates through a **workforce of 240,000 data labelers** across **Kenya, Philippines, and Venezuela** via subsidiary **Remotasks**, combining **human-in-the-loop annotation with ML-assisted tools** for computer vision and autonomous vehicles. Infrastructure supports **multi-cloud deployment** via **AWS, Azure, and GCP** with **VPC** options for private enterprise deployments. Backend services integrate with **all leading AI models** (Google, Meta, Cohere, Anthropic, OpenAI) supporting both **closed-source and open-source** foundation models. The platform uses **SDK support for Python, TypeScript, Java, and Go** with APIs for data annotation, model evaluation, and RLHF workflows. Customers include **OpenAI** (ChatGPT training partner), **Meta** (49% owner, Llama training), **Google, Microsoft, General Motors, Anthropic, Cohere, U.S. Army, U.S. Air Force, DOD CDAO**, and Fortune 500 enterprises. The company crossed **$100M+ annualized revenue** by May 2025, though OpenAI and Google **cut ties following the Meta deal**, with Microsoft and xAI exploring alternatives. Scale previously worked at Quora before founding Scale, and his parents were **Chinese immigrant physicists for the U.S. Air Force**.

**Salary Ranges**: Software Engineer $183K-$721K (L3-L6, median $275K) | Staff Software Engineer $160K-$192K base | Total comp 25th percentile $226K, 90th percentile $699K

---

## AI/ML Tech Stack

### Data Engine with RLHF - Powering ChatGPT, Claude, and Llama Training

**What's unique**: Scale's **Data Engine** provides **reinforcement learning from human feedback (RLHF)** infrastructure that powered the **initial creation of ChatGPT**, with Scale becoming OpenAI's **"preferred partner" to fine-tune GPT-3.5** (August 2023). The platform generates **quality RLHF data through customized training workflows** with skilled annotators in **linguistics, programming, and mathematics**, implementing the **three-step RLHF process**: **(1) Collect demonstration data** to fine-tune language models; **(2) Collect comparison data** where annotators rank model outputs to train a reward model rating response quality; **(3) Perform reinforcement learning** using the reward model to optimize the original language model. Scale's RLHF services drove **$18B in capital flowing into foundational model companies** (OpenAI, Anthropic, Cohere) in 2023, with models fine-tuned using human feedback **consistently outperforming those without across-the-board**. The **240,000-person workforce** provides domain expertise that pure ML-based annotation cannot replicate — linguists evaluate translation quality, mathematicians verify complex proofs, programmers assess code correctness. This **human-in-the-loop** approach creates training data with **nuanced preference judgments** (helpfulness, safety, truthfulness) rather than simple correctness labels. The 2023 revenue explosion demonstrated Scale's **central role in the generative AI supply chain** — nearly every major LLM uses Scale's RLHF services. The **fast feedback loop** enables model developers to iterate rapidly, collecting comparison data on new model versions and retraining reward models within days.

### Rapid Annotation Tool - Computer Vision for Autonomous Vehicles at Scale

**What makes it different**: **Scale Rapid** is Scale's annotation platform enabling **rapid experimentation** with **full-scene and partial-scene segmentation** for computer vision applications, serving as the **dominant vendor for autonomous vehicle companies** including **General Motors, Tesla, and others** requiring **massive labeled datasets**. The platform features **Autosegment**, where annotators **draw a box around an object and segmentation masks are automatically generated**, combining **AI-based techniques with human-in-the-loop** to deliver labeled data at **unprecedented quality, scalability, and efficiency**. For autonomous vehicles and intelligent robots where **nuanced scene understanding is mission-critical**, building models requires **massive amounts of highly accurate segmentation masks with pixel-level precision** across millions of images. Scale pioneered techniques where **ML models assist human annotators** — the system pre-labels objects using neural networks, then human experts **correct mistakes and refine boundaries**, with corrections feeding back to improve the ML models in a **continuous learning loop**. **Hundreds of thousands of images** have been annotated using Rapid, covering diverse conditions (weather, lighting, geography) essential for safe autonomous driving. The platform supports **3D bounding boxes, semantic segmentation, polyline annotation, and keypoint labeling**, enabling rich scene understanding beyond simple object detection. Scale's **workforce distribution across three continents** (Kenya, Philippines, Venezuela) provides **24/7 annotation cycles**, accelerating iteration velocity for customers. Unlike competitors offering annotation tools requiring customers to manage labelers, Scale provides **end-to-end managed services** where customers submit unlabeled data and receive precisely annotated datasets.

### Donovan Platform - AI Agents on Classified DOD Networks with $100M+ Contracts

**What sets Scale apart**: Scale deployed the **Donovan platform on classified networks** in partnership with the **U.S. Army's XVIII Airborne Corps**, securing **$100M+ in defense contracts** including a **$100M agreement with DOD CDAO** (Chief Digital and AI Office) and a **$99.5M U.S. Army contract** for AI research and development services. Donovan **deploys mission-tailored AI agents from concept to combat in record time**, available on **classified and air-gapped networks** to empower **intelligence analysts and mission operators** sifting through **massive amounts of unstructured data** to generate actionable insights. The platform integrates with **Scale GenAI Platform** to build and customize AI agents through a **no-code interface** by **accessing data sources, creating knowledge bases, leveraging state-of-the-art models, assigning agent instructions, and connecting agents to existing systems and tools**. Donovan applies **fine-tuned LLMs for specific industries** (defense, intelligence) to extract and process **tons of existing data** through cloud or API access, providing **relevant outputs and generating necessary reports** for tactical and strategic decision-making. The **Other Transaction Authority (OTA) procurement structure** allows Scale to **move at speed and partner with non-traditional tech companies**, streamlining processes so **any component across the entire DOD can access Scale's AI platform**. Donovan also powers Scale's **Thunderforge program** (awarded by Defense Innovation Unit), the **DOD's flagship program to leverage AI for military planning and operations**, working with Anduril, Microsoft, and others to develop and deploy AI agents. This defense focus positions Scale as the **leading AI contractor for national security**, contrasting with commercial-first competitors.

### SEAL Lab - First US AI Safety Institute Third-Party Evaluator

**What's unique**: Scale launched **SEAL (Safety, Evaluations and Alignment Lab)**, selecting the company as the **first third-party evaluator authorized to assess AI models on behalf of the U.S. AI Safety Institute (AISI)**, a role no other private company holds. SEAL is led by **Summer Yue** (former RLHF research lead for Bard at Google DeepMind) as **Director of Safety and Standards**, building **robust evaluation products** and tackling **challenging research problems in evaluation and red teaming**. The lab developed **novel evaluations jointly with AISI** assessing model performance in **math, reasoning, and AI coding**, plus safety benchmarks including the **Weapons of Mass Destruction Proxy (WMDP)** measuring **hazardous knowledge in biosecurity, chemical security, and cybersecurity**. Scale's **PropensityBench** reveals AI models **choose dangerous shortcuts 47% of the time under pressure**, testing **50 dangerous capabilities across 5,874 total tasks** in four high-risk domains. The **Scale Evaluation platform** enables frontier model developers to **understand, analyze, and iterate on models** through detailed breakdowns of LLM **performance and safety across multiple facets**. The platform was **selected by the White House to conduct public assessments** of models from leading AI developers using **1,000s of red teamers trained on advanced tactics** and in-house prompt engineers enabling **state-of-the-art red teaming at scale**. This evaluation infrastructure addresses the **AI Executive Order requirements** for federal agencies to **identify AI risks and measure AI in real-world context**, positioning Scale as the **de facto evaluation standard** for government AI deployments.

### GenAI Platform - Full-Stack Enterprise with RAG, Fine-Tuning, and Agentic Solutions

**What makes it different**: Scale's **GenAI Platform** is described as the **"only full-stack GenAI Platform for your Enterprise"**, integrating **foundation model access, fine-tuning/RLHF, enterprise data integration, model evaluation, and agentic solutions** in a unified system that enables AI teams to **build, evaluate, and control agentic solutions that reason over enterprise data, take action with tools, and continuously improve with human-agent interactions**. The platform uses **advanced Retrieval Augmented Generation (RAG) pipelines** to transform **proprietary data into high-quality training datasets and embeddings**, supporting **fine-tuning of LLMs with both proprietary and expert data** to create domain-specific models. Scale **securely customizes and deploys enterprise-grade GenAI applications in customer VPCs** (AWS, Azure, GCP), ensuring **data remains private and secure** within virtual private clouds rather than shared infrastructure. The system enables **agentic workflows** where AI systems autonomously complete multi-step tasks — analyzing documents, extracting insights, generating reports, and updating databases — rather than single-query interactions. The **no-code interface** allows non-technical users to build AI agents by connecting to **enterprise data sources** (databases, document repositories, APIs), creating **knowledge bases** with semantic search, assigning **agent instructions** defining behavior and goals, and integrating with **existing systems and tools** (Salesforce, SAP, ServiceNow). The platform supports **rigorous testing to maintain integrity** of AI applications, implementing **guardrails and monitoring** for production deployments. Scale's partnerships with **Dell, SAP, Oracle, AMD, Nvidia** extend global reach and platform integration, while supporting both **closed-source models** (GPT-4, Claude) and **open-source models** (Llama, Mistral) for vendor flexibility.

### 240K Human Workforce - Unmatched Scale in Human-in-the-Loop Annotation

**What sets Scale apart**: Scale operates a **workforce of 240,000 data labelers** across **Kenya, Philippines, and Venezuela** via subsidiary **Remotasks**, providing **human-in-the-loop (HITL) annotation** that combines **AI-assisted tools with human expertise** for tasks requiring **nuanced judgment, domain knowledge, and contextual understanding**. This workforce scale exceeds all competitors by orders of magnitude — while other annotation platforms offer **crowd-sourcing marketplaces**, Scale provides **managed services** with **quality control, annotator training, and domain specialization** (linguistics, programming, mathematics, computer vision). The HITL approach addresses limitations of **pure automated labeling** — ML models struggle with **edge cases, ambiguous scenarios, and tasks requiring common sense reasoning** that humans handle effortlessly. Scale's **three annotation modes** serve different use cases: **(1) Automated labeling** using custom ML models for high-volume, straightforward tasks; **(2) Human-only labeling** for complex, nuanced scenarios; **(3) HITL labeling** where ML pre-labels data and humans refine, combining **speed with accuracy**. The **geographic distribution** enables **24/7 annotation cycles** — when U.S. customers submit data at end-of-day, overnight shifts in Kenya and Philippines complete labeling for next-morning delivery. The workforce provides **linguistic diversity** for multilingual NLP tasks and **cultural context** for content moderation. Scale's infrastructure manages **task assignment, quality metrics, annotator performance tracking, and payment distribution** across the global workforce, solving operational complexity that prevents competitors from scaling. The **2023 revenue explosion** ($18B flowing to OpenAI, Anthropic, Cohere) demonstrates market validation that **high-quality human-labeled data remains irreplaceable** for training frontier models.

### Model Evaluation & Safety Infrastructure - Responsible AI for Public Sector

**What's unique**: Scale's **Evaluation platform** equips **public sector organizations** with tools to fulfill **Biden-Harris Administration AI Executive Order requirements**, helping federal agencies **identify AI risks and measure AI in real-world context** through comprehensive testing infrastructure. The platform enables **responsible AI deployment** by assessing models across **performance dimensions** (accuracy, reasoning, coding) and **safety dimensions** (bias, toxicity, robustness, security vulnerabilities). Scale provides **detailed breakdowns of LLMs across multiple facets**, allowing developers to understand **where models fail** (specific task types, input patterns, demographic groups) and **why they fail** (knowledge gaps, reasoning errors, safety misalignments). The **red teaming capabilities** leverage **1,000s of trained red teamers** using **advanced adversarial tactics** to probe models for vulnerabilities — jailbreaks, prompt injections, data extraction attacks, harmful content generation. Scale's **WMDP benchmark** measures **hazardous knowledge** in **biosecurity** (bioweapon design), **chemical security** (explosive synthesis), and **cybersecurity** (exploit development), critical for evaluating **dual-use risks** where models could assist malicious actors. The **PropensityBench** tests whether models **choose dangerous shortcuts under pressure** — for example, given a tight deadline to complete a coding task, does the model introduce security vulnerabilities for speed? The evaluation platform integrates with **SEAL Lab's research** on novel threat models and evaluation methodologies, ensuring the platform evolves with frontier AI capabilities. Scale's position as the **first US AISI third-party evaluator** means federal agencies **trust Scale's assessments** when procuring AI systems, creating a **moat** where competing on government contracts requires passing Scale's evaluations.

---

## Sources

**Scale AI Official**:
- [Scale AI Homepage](https://scale.com/)
- [Scale Careers](https://scale.com/careers)
- [Scale GenAI Platform](https://scale.com/genai-platform)
- [Scale Data Engine (Rapid)](https://scale.com/rapid)
- [Scale Donovan](https://scale.com/donovan)
- [Scale Defense Solutions](https://scale.com/defense)
- [Scale RLHF](https://scale.com/rlhf)
- [Scale Evaluation](https://scale.com/evaluation/model-developers)

**Technical Infrastructure**:
- [SEAL: Safety, Evaluations and Alignment Lab](https://scale.com/blog/safety-evaluations-alignment-lab)
- [Generative AI for the Enterprise Guide](https://scale.com/guides/generative-ai)
- [Test and Evaluation Vision](https://scale.com/guides/test-and-evaluation-vision)
- [Machine Learning-Assisted Image Semantic Segmentation](https://scale.com/blog/ml-image-segmentation)
- [Autonomous Driving Data Solutions](https://scale.com/automotive)
- [Why is ChatGPT so good? (RLHF)](https://scale.com/blog/chatgpt-reinforcement-learning)
- [GenAI Platform Launch](https://scale.com/blog/genai-platform)

**Government & Defense**:
- [Scale AI and DOD Expand Army R&D Partnership](https://scale.com/blog/scale-ai-dod-expand-army-rd-partnership)
- [Scale AI & CDAO Sign $100M AI Agreement](https://scale.com/blog/scale-ai-inks-100m-deal)
- [Scale AI Awarded DoD Data Curation Contract](https://scale.com/blog/scale-dod-contract-data-curation-joint-force)
- [Introducing Thunderforge: AI for American Defense](https://scale.com/blog/thunderforge-ai-for-american-defense)
- [Scale AI Provides AI Tools Under $100M Pentagon Agreement](https://www.govconwire.com/articles/scale-ai-dod-ota-agreement-donovan-gen-ai)
- [DoD Taps Scale AI for Top Secret Networks](https://www.theregister.com/2025/09/17/dod_scale_ai_deal/)

**Safety & Evaluation**:
- [Scale Partnering with US AISI](https://scale.com/blog/first-independent-model-evaluator-for-the-USAISI)
- [US AI Safety Institute Taps Scale AI](https://fedscoop.com/us-ai-safety-institute-taps-scale-ai-for-model-evaluation/)
- [Scale AI, USAI Safety Institute Partnership](https://executivebiz.com/2025/02/scale-ai-usai-safety-institute-partnership-advance-ai-model-evaluation/)
- [PropensityBench AI Safety Risks](https://www.how2shout.com/news/propensitybench-ai-models-safety-risks-under-pressure.html)
- [Responsible AI with Scale Evaluation for Public Sector](https://scale.com/blog/responsible-ai-scale-evaluation-for-public-sector)

**Partnerships & Enterprise**:
- [Partnering with Scale to Bring GenAI to Enterprises - Anthropic](https://www.anthropic.com/news/partnering-with-scale)
- [Meta's $14.3B Investment in Scale AI](https://goodai.substack.com/p/metas-143b-investment-in-scale-ai)
- [How Meta's Scale Deal Upended the AI Data Industry - TIME](https://time.com/7294699/meta-scale-ai-data-industry/)
- [Scale AI Revenue Analysis - Sacra](https://sacra.com/c/scale-ai/)

**Company & Funding**:
- [Scale AI - Wikipedia](https://en.wikipedia.org/wiki/Scale_AI)
- [Alexandr Wang - Wikipedia](https://en.wikipedia.org/wiki/Alexandr_Wang)
- [Scale AI Secures $1B at $14B Valuation - Fortune](https://fortune.com/2024/05/21/scale-ai-funding-valuation-ceo-alexandr-wang-profitability/)
- [Scale Business Breakdown - Contrary Research](https://research.contrary.com/company/scale)
- [8 Scale AI Statistics - TapTwice Digital](https://taptwicedigital.com/stats/scale-ai)
- [Scale AI Data is the Code - Generational](https://www.generational.pub/p/scale-ai)

**Alexandr Wang Profile**:
- [Meet Alexandr Wang, MIT Dropout to Billionaire - Entrepreneur](https://www.entrepreneur.com/business-news/who-is-alexandr-wang-the-founder-of-scale-ai-joining-meta/493281)
- [Alexandr Wang: Youngest Self-Made Billionaire - The Week](https://theweek.com/news/technology/961534/alexandr-wang-profile)
- [From MIT Dropout to AI Mogul - VnExpress](https://e.vnexpress.net/news/tech/tech-news/from-mit-dropout-to-ai-mogul-how-the-world-s-youngest-self-made-tech-billionaire-alexandr-wang-builds-data-empire-4873124.html)
- [Inside Alexandr Wang and Meta's $14B Bet - Fortune](https://fortune.com/2025/06/22/inside-rise-scale-alexandr-wang-meta-zuckerberg-14-billion-deal-acquihire-ai-supremacy-race/)

**Job Postings & Compensation**:
- [Scale AI Software Engineer Salaries - Levels.fyi](https://www.levels.fyi/companies/scale-ai/salaries/software-engineer)
- [Scale AI Salaries - Levels.fyi](https://www.levels.fyi/companies/scale-ai/salaries)
- [Scale AI Salaries - 6figr](https://6figr.com/us/salary/scale-ai)
- [Scale AI Salary - Blind](https://www.teamblind.com/company/Scale-AI/salaries/united-states)
- [Scale AI Careers - The Ladders](https://www.theladders.com/company/scaleai-jobs)

---

*Last updated: November 30, 2025*
