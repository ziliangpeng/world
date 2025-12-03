# Databricks - Technology Stack

**Company:** Databricks, Inc.
**Founded:** 2013
**Focus:** Data and AI platform (Lakehouse architecture)
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

Databricks was founded by the team that **created Apache Spark** at UC Berkeley's AMPLab: **Matei Zaharia** (CTO, Spark creator), **Ali Ghodsi** (CEO), **Ion Stoica**, **Reynold Xin**, **Andy Konwinski**, **Patrick Wendell**, and **Arsalan Tavakoli-Shiraji**. The platform is built on the **Lakehouse architecture**, combining data lake economics with data warehouse performance through **Delta Lake** (open storage format with ACID transactions, time travel, schema enforcement) and **Unity Catalog** (unified governance for structured/unstructured data and AI models with fine-grained access control and lineage tracking). The **Photon Engine** is a C++ vectorized query engine that accelerates Delta Lake queries by **up to 20x** (2-4x for ETL, 20x for complex joins) through improved query optimization, caching layers, and predicate pushdown optimized for Parquet columnar storage. **MLflow** provides open-source ML lifecycle management with experiment tracking, model registry (Unity Catalog integration), and deployment to **Mosaic AI Model Serving**. The platform supports **Apache Spark, Delta Lake, Apache Iceberg, Hudi, and Parquet** formats. Databricks operates R&D centers in **San Francisco, Mountain View, Seattle, Bellevue, Amsterdam, Serbia, and Berlin**. The company raised **$130B valuation** (announced Nov 2025), backed by Qatar Investment Authority, Temasek, Andreessen Horowitz, DST Global, GIC, Insight Partners, Thrive Capital, Wellington Management, and others. Zaharia won the **2014 ACM Doctoral Dissertation Award** for Spark (cited 12,840+ times), and Forbes ranks him and Stoica as 3rd-richest Romanians at $2.7B net worth.

**Salary Ranges**: Software Engineer median $246K total comp (L3 $253K, L4 $408K, L5 $643K, L7 $1.25M, highest $1.74M)

---

## AI/ML Tech Stack

### DBRX - Fine-Grained Mixture-of-Experts LLM

**What's unique**: Databricks built **DBRX**, an open-source, commercially usable LLM with **132B total parameters** and **36B active parameters** using a **fine-grained mixture-of-experts (MoE) architecture**. Unlike Mixtral-8x7B and Grok-1 (8 experts choose 2), DBRX uses **16 experts and chooses 4**, providing **65x more possible combinations** that improve model quality. DBRX was **developed in just two months with a $10M budget**, pretrained on **12T tokens** of carefully curated text and code with a **32K token context length**. The model uses **rotary position encodings (RoPE), gated linear units (GLU), and grouped query attention (GQA)**. DBRX achieves **2x faster inference throughput** compared to LLaMA2-70B thanks to having about half as many active parameters. The model outperformed Mixtral, Claude 3, Llama 2, and Grok-1 on standard benchmarks, setting new benchmarks for open LLMs with efficiency and advanced capabilities in chat, coding, and math. DBRX is available on Hugging Face (databricks/dbrx-instruct, databricks/dbrx-base) and deployed via **Mosaic AI Model Serving** with accelerated inference.

### MosaicML Acquisition - $1.3B for Generative AI Leadership

Databricks **acquired MosaicML in July 2023 for $1.3 billion** (inclusive of retention packages), bringing the **entire 62-person team** including an industry-leading research team to Databricks. MosaicML, launched two years prior with **$64 million in funding**, built a platform for organizations to **train large language models and deploy generative AI tools** based on them. MosaicML is known for **state-of-the-art MPT (MosaicML Pretrained Transformer) large language models**. The acquisition's strategic rationale: **"make generative AI accessible for every organization, enabling them to build, own and secure generative AI models with their own data."** MosaicML's expertise in neural networks and LLM training infrastructure became the **Mosaic AI** product line within Databricks. The rapid $1.3B acquisition (announced June 26, completed July 19, 2023) positioned Databricks as an **OpenAI competitor** in the generative AI platform space, differentiating through data ownership and security rather than API-only access.

### Mosaic AI - Production-Quality ML & GenAI Platform

**Mosaic AI** provides end-to-end capabilities for **model training, fine-tuning, and deployment** integrated with Databricks Lakehouse. The platform includes **Mosaic AI Model Serving** for hosting ML models registered in Unity Catalog as REST endpoints, supporting both **CPU and GPU optimization**. Models deploy with **reproducible packaging** (code, dependencies, weights) and serve via REST APIs or high-throughput batch inference using `ai_query()`. **MLflow 3** introduces **Logged Models** (tracking model progress throughout lifecycle) and **Deployment Jobs** (managing evaluation, approval, deployment). The platform enables **tracing for agents**, **human-in-the-loop workflows**, and **dataset engineering** for LLM training at scale. Mosaic AI integrates with **Unity Catalog** for centralized AI model and artifact management with governance, lineage tracking, and fine-grained access control.

### Photon Engine - C++ Vectorized Query Acceleration

**What makes it different**: Databricks built **Photon**, a **C++ vectorized query engine** from scratch (fully compatible with Spark APIs) that provides **extremely fast query performance at low cost**. Photon leverages **modern CPU architecture with Delta Lake** to enhance Apache Spark 3.0 performance by **up to 20x**. The engine has three components: **(1) Improved query optimizer** extending Spark 3.0's cost-based optimizer and adaptive query execution with advanced statistics (18x speedup on star schema); **(2) Caching layer** between execution and cloud object storage; **(3) Native vectorized execution engine** written in C++. Photon was **built specifically for Delta Lake's Parquet format**, exploiting columnar storage properties through **predicate pushdown** (filtering data earlier in query execution) to minimize I/O. Performance improvements: **2x speedup per TPC-DS 1TB benchmark**, **3x-8x speedups on average** in customer workloads, **2x-4x for ETL and feature engineering**, and **up to 20x for certain complex joins**. Photon is optimized for Delta Lake and Parquet, achieving best performance with **dynamic partition pruning, Z-ordering, and optimized file sizes**.

### Delta Lake - ACID Transactions for Data Lakes

**Delta Lake** is Databricks' **open storage format** that brings **ACID transactions, time travel, and schema enforcement** to data lakes. The format provides **versioning and reliability** to raw data with managed tables always using Delta format. Key capabilities: **ACID transactions** prevent data corruption during concurrent writes; **time travel** enables querying previous versions; **schema enforcement and evolution** maintain data quality; **Z-ordering** co-locates related information for faster queries; **dynamic partition pruning** eliminates unnecessary data reads. Delta Lake integrates with **Photon Engine** for query acceleration and **Unity Catalog** for lineage tracking. The open format works across **Apache Spark, Delta Lake, Apache Iceberg, Hudi, and Parquet**, providing **data warehousing performance at data lake economics** — the core of Databricks' Lakehouse architecture.

### Unity Catalog - Unified Governance for Data & AI

**Unity Catalog** is Databricks' **unified, fine-grained governance solution** for all structured data, unstructured data, business metrics, and AI models across open formats (Delta Lake, Apache Iceberg, Hudi, Parquet). The catalog provides **centralized management** across clouds with **fine-grained access control** (table, column, row-level), **lineage tracking** as data transforms and refines, and **unified governance** to keep sensitive data private and secure. Unity Catalog integrates with **MLflow Model Registry** for centralized AI model management, **Mosaic AI Model Serving** for governed model deployment, and **Delta Lake** for data asset tracking. The system enables **multi-cloud governance** (AWS, Azure, GCP) from a single catalog, critical for enterprises managing AI models and datasets across distributed infrastructure.

### MLflow - Open-Source ML Lifecycle Management

Databricks created **MLflow**, the **open-source platform** for developing models and generative AI applications with components for **Tracking** (experiments, parameters, results), **Model Registry** (versioning, staging, production promotion), and **Deployment**. Databricks provides a **hosted MLflow tracking server** with no setup, storing experiment data in workspaces. Users log notebooks, training datasets, parameters, metrics, tags, and artifacts related to model training. **MLflow 3** introduces **Logged Models** (tracking model progress throughout lifecycle) and **Deployment Jobs** (managing evaluation, approval, deployment). **MLflow Model Registry integrates with Unity Catalog** for centralized governance. Models deploy via **Mosaic AI Model Serving** as REST endpoints or batch inference with `ai_query()`. MLflow supports **MLOps workflows** for production model lifecycle management with reproducible packaging formats.

### Apache Spark Origins - Created by Databricks Founders

**What sets Databricks apart**: The company was **founded by the team that created Apache Spark** at UC Berkeley in 2009. **Matei Zaharia** (Databricks CTO) started Spark during his Ph.D. at Berkeley's AMPLab as a **faster alternative to MapReduce**, and it became the **most active open-source project in Big Data**. Spark **transformed large-scale data processing** and has been **cited 12,840+ times**, earning Zaharia the **2014 ACM Doctoral Dissertation Award**. Databricks **continues to drive Apache Spark development**, maintaining direct influence over the project's roadmap. This unique position gives Databricks deep expertise in distributed computing, enabling innovations like **Photon Engine** (C++ rewrite optimized for modern hardware), **Delta Lake** (ACID transactions for Spark), and **MLflow** (ML lifecycle for Spark workloads). The founders also created **Delta Lake** and **DBRX**, demonstrating continued innovation beyond Spark.

**Salary Ranges**: Software Engineer median $246K total comp (L3 $253K, L4 $408K, L5 $643K, L7 $1.25M, highest $1.74M)

---

## Sources

**Databricks Products & Platform**:
- [Databricks Lakehouse Platform](https://www.databricks.com/)
- [Unity Catalog](https://www.databricks.com/product/unity-catalog)
- [Delta Lake on Databricks](https://www.databricks.com/product/delta-lake-on-databricks)
- [Databricks Photon Engine](https://www.databricks.com/product/photon)
- [Managed MLflow](https://www.databricks.com/product/managed-mlflow)
- [Mosaic AI](https://www.databricks.com/product/artificial-intelligence)

**DBRX Model & Research**:
- [Introducing DBRX: A New State-of-the-Art Open LLM](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)
- [Accelerated DBRX Inference on Mosaic AI Model Serving](https://www.databricks.com/blog/accelerated-dbrx-inference-mosaic-ai-model-serving)
- [databricks/dbrx-instruct - Hugging Face](https://huggingface.co/databricks/dbrx-instruct)
- [databricks/dbrx - GitHub](https://github.com/databricks/dbrx)
- [Databricks Mosaic Research](https://www.databricks.com/research/mosaic)

**MosaicML Acquisition**:
- [Databricks Signs Agreement to Acquire MosaicML](https://www.databricks.com/company/newsroom/press-releases/databricks-signs-definitive-agreement-acquire-mosaicml-leading-generative-ai-platform)
- [Databricks Completes Acquisition of MosaicML](https://www.databricks.com/company/newsroom/press-releases/databricks-completes-acquisition-mosaicml)
- [Databricks picks up MosaicML for $1.3B - TechCrunch](https://techcrunch.com/2023/06/26/databricks-picks-up-mosaicml-an-openai-competitor-for-1-3b/)

**Technical Documentation**:
- [Photon: A Fast Query Engine for Lakehouse Systems - SIGMOD 2022](https://people.eecs.berkeley.edu/~matei/papers/2022/sigmod_photon.pdf)
- [MLflow Documentation](https://docs.databricks.com/aws/en/mlflow/)
- [Unity Catalog Documentation](https://docs.databricks.com/aws/en/data-governance/unity-catalog/)
- [Delta Lake Documentation](https://docs.databricks.com/aws/en/lakehouse/)

**Company & Founders**:
- [Databricks Founders](https://www.databricks.com/company/founders)
- [Matei Zaharia - Wikipedia](https://en.wikipedia.org/wiki/Matei_Zaharia)
- [Databricks - Wikipedia](https://en.wikipedia.org/wiki/Databricks)
- [Story of Databricks](https://www.sunrisegeek.com/post/story-of-databricks)

**Job Postings & Compensation**:
- [Databricks Careers](https://www.databricks.com/company/careers)
- [Engineering at Databricks](https://www.databricks.com/company/careers/engineering-at-databricks)
- [Open Positions](https://www.databricks.com/company/careers/open-positions)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/databricks/salaries)

---

*Last updated: November 30, 2025*
