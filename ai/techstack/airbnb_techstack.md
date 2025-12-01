# Airbnb - Technology Stack

**Company:** Airbnb, Inc.
**Founded:** 2008
**Focus:** Online marketplace for lodging and vacation rentals
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

Airbnb operates from **San Francisco** with infrastructure entirely on **AWS** (EC2, RDS MySQL, S3) using a polyglot backend: **Ruby** (Rails monolith and core services), **Java** (scalable backend services), **Scala** (data pipelines, financial systems), **Kotlin** (monorepo services), **Python** (automation, data science), **Node.js** (APIs), and **Go** (infrastructure). The architecture evolved from a **single Rails monolith to 1000+ microservices** managed through **100+ Kubernetes clusters** (30+ cluster types) with **Spinnaker** (continuous delivery) and **OneTouch** (infrastructure as code). Frontend uses **React** (web) and **React Native** (mobile). The data platform processes **35+ billion events/day** through **Apache Kafka** (streaming), **Apache Spark** (batch/streaming processing), **Apache Airflow** (workflow orchestration, **created by Airbnb in 2014** by Maxime Beauchemin, Apache top-level project since 2019), **Apache Hive** (data warehouse on 11+ PB HDFS), and **Presto/Trino** (query engine). Visualization via **Apache Superset** (**created by Airbnb in 2015** at company hackathon by Maxime Beauchemin, donated to Apache Software Foundation). Both Airflow and Superset were **open-sourced from the beginning**, demonstrating Airbnb's commitment to community-driven innovation. The build system migrated from Gradle to **Bazel** over **4.5 years**, achieving **3-5x faster builds**. Analytics migrated from **Apache Druid** to **StarRocks** for the **Minerva platform** (30K+ metrics, 7K+ dimensions). Founded in 2008, Airbnb has become a global platform operating in 220+ countries with billions in annual revenue.

**Salary Ranges**: Software Engineer $187K-$924K (G7-G11, median $496K) | Senior SWE $190K-$250K | Staff SWE $250K-$350K

---

## AI/ML Tech Stack

### Zipline Feature Store - 60% Time Reduction from Months to Days

**What's unique**: Airbnb built **Zipline**, a **declarative feature engineering framework** that serves as a **data management platform specifically designed for ML use cases**, reducing the time ML practitioners spend collecting data and writing transformations from **months to approximately one day** (**60% time savings**). Before Zipline, data scientists manually wrote complex SQL queries, Spark jobs, and Python scripts to extract features from Airbnb's massive data warehouse, requiring deep knowledge of data schemas and infrastructure. Zipline provides **access to point-in-time correct features** for both **offline model training and online inference**, solving the critical problem of **train-serve skew** where models trained on historical data fail in production due to feature computation inconsistencies. The architecture uses a **Lambda approach** with **batch processing via Spark** and **streaming via Flink**, enabling both historical feature computation for training and low-latency serving for real-time inference. Zipline sits in front of Airbnb's data warehouse, allowing ML teams to **define features declaratively** and automatically handling the complex transformations, joins, and aggregations required. The framework integrates directly with **Bighead** (Airbnb's ML platform) and **ML Automator** (batch inference engine), creating a seamless pipeline from feature definition to model deployment. Zipline's **feature repository** provides **discoverability and reusability** — data scientists can search for existing features rather than duplicating work, accelerating model development across teams.

### Listing Embeddings - Real-Time Personalization in Search Ranking

**What makes it different**: Airbnb developed **listing and user embedding techniques** using **neural networks** to create **vector representations of homes** learned from **search sessions** that measure **similarities between listings**, enabling **real-time personalization** in search ranking and similar listing recommendations. The embeddings capture **guests' short-term and long-term interests** through session behavior — click sequences, bookings, and rejections — encoding complex preferences into dense vectors where **similar listings cluster together** in embedding space. This approach differs fundamentally from traditional collaborative filtering (which suffers from cold-start problems) and content-based recommendations (which miss behavioral signals). Airbnb's embedding model was specifically **tailored for the two-sided marketplace**, capturing both **guest preferences** (what travelers want) and **host preferences** (which guests hosts accept). The system uses **more than 100 signals** in ranking models, with embeddings serving as powerful features alongside explicit attributes (location, price, amenities). The **host preference modeling** personalizes search by predicting which guests a host will accept, showing **3.75% increase in booking conversion** in A/B testing. Airbnb published this work at **KDD 2018** ("Real-time Personalization using Embeddings for Search Ranking at Airbnb"), demonstrating how embeddings enable **99% of booking conversions** through Search Ranking and Similar Listings combined. The technical implementation uses **gradient boosted decision trees (GBDT)** and **neural network architectures** trained to assign **booked listings at high ranks**.

### Dynamic Pricing - Smart Pricing ML Algorithm for Revenue Optimization

**What sets Airbnb apart**: Airbnb's **Smart Pricing** tool uses **machine learning to automatically adjust nightly prices** based on **demand, availability, and market trends**, analyzing **massive amounts of data** including **local demand patterns** (events, holidays, seasonal trends), **historical booking data**, **current market conditions** (competitor pricing, occupancy rates), and **listing-specific characteristics**. The system performs **daily dynamic pricing** — offering new price tips based on changing market conditions rather than static recommendations. At **KDD 2018**, Airbnb presented their technical approach: **(1) Binary classification model** predicts booking probability for each listing-night; **(2) Regression model** predicts optimal price for each listing-night using a **customized loss function** to guide learning. This two-stage approach enables the model to balance **maximizing booking probability** with **maximizing revenue** — sometimes a lower price increases bookings enough to outperform higher prices. The algorithm considers **unusual, even surprising characteristics of listings** that traditional pricing models miss, such as unique architectural features, location micro-trends, and guest review sentiment. Smart Pricing operates at **massive scale** across millions of listings worldwide, requiring efficient distributed training on Spark and low-latency inference for real-time price updates. Hosts can set **minimum and maximum bounds**, allowing algorithmic suggestions while maintaining control. The system **continuously learns** from new bookings and market changes, adapting to shifts in travel patterns (pandemic impact, work-from-home trends, seasonal variations).

### Translation Engine - 60+ Languages Automatic Translation at Scale

**What's unique**: Airbnb deployed a **Translation Engine** in **November 2021** that provides **automatic translation** of reviews and descriptions in **more than 60 languages** through a **sophisticated machine-learning process**, enabling **seamless experiences** for homeowners and guests without requiring manual translation button clicks. The system was designed to handle **user-generated content (UGC) at scale** — millions of listings, reviews, and messages created daily in diverse languages and writing styles. Airbnb partnered with **Italian LSP Translated** using **ModernMT**, implementing a **human-machine symbiotic approach** where **every single correction from the localization team improves machine translation instantly**, creating a **continuous learning loop**. This hybrid model combines **neural machine translation** (fast, scalable) with **human expertise** (accuracy, cultural relevance), achieving quality superior to either approach alone. The Translation Engine uses **natural language processing (NLP)** for **real-time message translations** between hosts and guests who speak different languages, removing communication barriers that previously prevented bookings. Beyond translation, Airbnb leverages **generative AI and NLP** for **personalized user experiences** — conversational search powered by **natural language understanding** allows users to **"search as if talking to a friend"** with simple queries rather than structured filters. The 2025 AI developments include **ranking, recommendations, retrieval, and GenAI** integration across the platform, from **large-scale learning-to-rank systems** and **embedding-based search** to **contextual bandits**, **generative metadata**, and **agentic AI**.

### Bighead ML Platform - Framework-Agnostic End-to-End Infrastructure

**What makes it different**: Airbnb built **Bighead**, an **end-to-end machine learning platform** designed to **standardize and simplify the ML development workflow**, reducing model building time from **weeks/months to days/weeks** while enabling **more teams to utilize ML**. Bighead is **framework-agnostic**, supporting **TensorFlow, PyTorch, Keras, MXNet, Scikit-learn, XGBoost, H2O, and R**, recognizing that **each ML problem presents unique challenges** requiring different tools. The platform provides a **unified model building API**, **Redspot** (centrally-managed Jupyter notebook service for exploration), and modular components usable independently. A key innovation: **Dockerized models** where users provide a **Docker image** within which model code always runs, solving the **dependency management problem** (ML models have diverse, conflicting requirements). Bighead built a **lightweight API to interact with dockerized models**, ensuring **consistent environment across the stack** — development, training, and production use identical containerized environments, eliminating "works on my machine" issues. The platform emphasizes **consistent data transformation** where **model transformation code is the same in online and offline environments**, preventing train-serve skew. Bighead integrates with **Zipline** (feature store) for point-in-time correct features, **ML Automator** (batch inference) for periodic retraining, and Airbnb's **data infrastructure** (35B+ events/day via Kafka and Spark). Recent focus on **GenAI infrastructure** with tools for rapid AI application development, improved iteration speed, AI safety, and guardrails. Bighead demonstrates Airbnb's philosophy: **versatile, consistent, and scalable ML** accessible to engineers without deep ML expertise.

### ML Automator - Batch Inference Engine on Airflow and Spark

**What's unique**: Airbnb developed **ML Automator**, a **workflow engine** that runs behind the scenes to **automate common offline tasks** including **periodic model (re)training and evaluation**, **batch scoring**, **uploading scores**, and **creating dashboards and alerts based on scores**. ML Automator users **specify tasks declaratively**, and the system **generates Airflow DAGs** under the covers, specifying appropriate connections to **Bighead resources** (model registry, training infrastructure) and **Zipline data** (feature store). **Computation runs on Spark for scalability**, handling batch inference across millions of listings and users. This automation eliminates manual workflow management — before ML Automator, data scientists wrote custom Airflow DAGs, Spark jobs, and orchestration scripts for each model, duplicating boilerplate code and creating maintenance burden. ML Automator abstracts the complexity: define **what** to compute (model retraining schedule, inference targets), and the platform handles **how** (resource allocation, dependency management, failure recovery). The system automatically monitors **model performance** through integrated dashboards, alerting when metrics degrade below thresholds. ML Automator enables **continuous model improvement** — models retrain on fresh data (daily, weekly, monthly schedules), incorporating new user behavior and market trends without manual intervention. The integration with Airflow (created by Airbnb) leverages a battle-tested orchestration framework while abstracting away low-level details, demonstrating Airbnb's approach to **democratizing ML** through tooling that hides complexity behind simple APIs.

### Trust & Safety - Fraud Detection and Risk Models at Scale

Airbnb built a **machine learning system for risk** architected to handle both **fast, robust scoring** (real-time fraud detection during bookings) and **agile model-building pipelines** (rapid iteration on fraud signals). The Trust & Safety team protects the community from **online fraud** (monetary loss, compromised accounts, spam/scam messages, fake inventory) and **offline fraud** (theft, property damage, personal safety). The system uses **random forest classifiers** in many risk mitigation models, trained on **past examples of confirmed good and confirmed fraudulent behavior**. A separate **fraud prediction service** (built in Java) handles **deriving all features for a particular model**, **parallelizing database queries** necessary for feature generation to maintain low latency. **Financial fraud detection** focuses on **chargebacks** — transactions from stolen credit cards — using ML models trained on historical fraud patterns. **Fake listing detection** evaluates each listing against **hundreds of risk signals** including **host reputation**, **template messaging**, **duplicate photos**, and **other discrepancies**, using data learnings from **millions of listings**. **Reservation screening technology** identifies **potentially high-risk reservations** before they complete, preventing platform abuse. Airbnb implements **targeted friction** in the fraud detection workflow — legitimate users experience seamless booking while suspicious transactions face additional verification steps, balancing **security with user experience**. The risk models operate at **massive scale** (hundreds of millions of annual bookings), requiring efficient inference infrastructure and continuous model updates as fraudsters adapt tactics.

---

## Sources

**Airbnb Official**:
- [Airbnb Tech Blog - Medium](https://medium.com/airbnb-engineering)
- [Airbnb Engineering & Data Science](https://airbnb.io/)
- [Airbnb Open Source Projects](https://airbnb.io/projects/)
- [Airbnb Careers](https://careers.airbnb.com/)

**ML Platform & Infrastructure**:
- [Bighead: Airbnb's End-to-End ML Platform - O'Reilly Strata](https://conferences.oreilly.com/strata/strata-ny-2018/public/schedule/detail/69383.html)
- [Bighead: Airbnb's End-to-End ML Platform - DataCouncil](https://www.datacouncil.com/talks/bighead-airbnbs-end-to-end-machine-learning-platform)
- [Zipline Feature Store - O'Reilly Strata](https://conferences.oreilly.com/strata/strata-ny-2018/public/schedule/detail/68114.html)
- [Zipline - DataCouncil](https://www.datacouncil.ai/talks/zipline-airbnbs-declarative-feature-engineering-framework)

**Search & Personalization**:
- [ML-Powered Search Ranking - Airbnb Tech Blog](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789)
- [Listing Embeddings for Search Ranking - Airbnb Tech Blog](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e)
- [Real-time Personalization using Embeddings - KDD 2018](https://dl.acm.org/doi/10.1145/3219819.3219885)
- [Search Ranking & Personalization - RecSys 2017](https://dl.acm.org/doi/10.1145/3109859.3109920)
- [Host Preference Detection - Airbnb Tech Blog](https://medium.com/airbnb-engineering/how-airbnb-uses-machine-learning-to-detect-host-preferences-18ce07150fa3)

**Dynamic Pricing**:
- [Customized Regression Model for Dynamic Pricing - KDD 2018](https://www.kdd.org/kdd2018/accepted-papers/view/customized-regression-model-for-airbnb-dynamic-pricing)
- [How Airbnb Uses AI for Dynamic Pricing - Medium](https://subashpalvel.medium.com/how-airbnb-uses-ai-for-dynamic-pricing-a-deep-dive-into-smart-revenue-optimization-66b17c532080)
- [Secret of Airbnb's Pricing Algorithm - IEEE Spectrum](https://spectrum.ieee.org/the-secret-of-airbnbs-pricing-algorithm)

**Translation & NLP**:
- [AI Text Generation Models in Customer Support - Airbnb Tech Blog](https://medium.com/airbnb-engineering/how-ai-text-generation-models-are-reshaping-customer-support-at-airbnb-a851db0b4fa3)
- [Airbnb Translation Engine - Slator](https://slator.com/airbnb-translation-engine-applies-machine-translation-to-ugc/)
- [Airbnb Translation Engine - Machine Translation](https://www.machinetranslation.com/blog/airbnb-own-machine-translation-engine-draws-increasing-returns)
- [AI @ Airbnb Event](https://aiairbnb.splashthat.com/)

**Trust & Safety / Fraud Detection**:
- [Architecting ML System for Risk - Airbnb Tech Blog](http://nerds.airbnb.com/architecting-machine-learning-system-risk/)
- [Fighting Financial Fraud - Airbnb Tech Blog](https://medium.com/airbnb-engineering/fighting-financial-fraud-with-targeted-friction-82d950d8900e)
- [Fighting Financial Fraud - InfoQ](https://www.infoq.com/news/2018/03/financial-fraud-ml-airbnb/)
- [Detecting Good and Bad with AI - MIT Sloan](https://sloanreview.mit.edu/audio/detecting-the-good-and-the-bad-with-ai-airbnbs-naba-banerjee/)

**Open Source Contributions**:
- [From Airflow to Superset - Preset Blog](https://preset.io/blog/from-airflow-to-superset-how-one-data-engineers-mission-became-an-open/)
- [Apache Airflow - Wikipedia](https://en.wikipedia.org/wiki/Apache_Airflow)
- [Supercharging Apache Superset - Airbnb Tech Blog](https://medium.com/airbnb-engineering/supercharging-apache-superset-b1a2393278bd)
- [Maxime Beauchemin Profile - Datanami](https://www.datanami.com/2023/03/31/meet-maxime-beauchemin-a-2023-person-to-watch/)

**Infrastructure & Data**:
- [Data Infrastructure at Airbnb - Airbnb Tech Blog](https://medium.com/airbnb-engineering/data-infrastructure-at-airbnb-8adfb34f169c)
- [Dynamic Kubernetes Cluster Scaling - Airbnb Tech Blog](https://medium.com/airbnb-engineering/dynamic-kubernetes-cluster-scaling-at-airbnb-d79ae3afa132)
- [Rearchitecting Frontend - Airbnb Tech Blog](https://medium.com/airbnb-engineering/rearchitecting-airbnbs-frontend-5e213efc24d2)
- [MySQL in the Cloud - Airbnb Tech Blog](http://nerds.airbnb.com/mysql-in-the-cloud-at-airbnb/)
- [Continuous Delivery at Airbnb](https://airbnb.tech/infrastructure/continuous-delivery-at-airbnb/)

**Job Postings & Compensation**:
- [Airbnb Careers - All Positions](https://careers.airbnb.com/)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/airbnb/salaries)

---

*Last updated: November 30, 2025*
