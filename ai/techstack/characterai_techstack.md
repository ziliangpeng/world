# Character.AI - Technology Stack

**Company:** Character Technologies, Inc. (Character.AI)
**Founded:** 2021 (Beta: September 2022)
**Focus:** Conversational AI platform and personalized AI characters
**Headquarters:** Menlo Park, California

---

## Non-AI Tech Stack

Character.AI runs entirely on **Google Cloud Platform** with a hybrid architecture: **Django** (Python monolith for core services) and **Golang** (chat microservice). The database strategy uses **AlloyDB for PostgreSQL** (frontend chat operations, 20x capacity increase, <100ms replication lag, 150% QPS increase) and **Cloud Spanner** (chat microservice, unlimited scale, terabytes/day). Supporting databases include **Cloud SQL** and **Memorystore (Redis)** for caching. Data infrastructure uses **Google Cloud Storage** (object storage), **Datastream** (CDC for real-time replication), and **BigQuery** (analytics). The platform serves **~20,000 queries per second** (20% of Google Search volume) at **<$0.01 per hour** of conversation cost. Infrastructure scaled from monolith through zero-downtime migration to handle exponential growth post-beta launch. Frontend uses **React/React Native** (inferred) with WebSocket for real-time chat.

**Salary Ranges**: Software Engineer $150K-$300K | Staff SWE $180K-$300K (median total comp $410K)

---

## AI/ML Tech Stack

Character.AI's ML infrastructure is built on pure **PyTorch** with the open-sourced **pipeling-sft** framework for fine-tuning MoE (Mixture-of-Experts) LLMs like DeepSeek V3. Training and inference run on **TPU v5p** (8,960 chips per pod, 2x higher FLOPS, 3x more HBM than v4) and **NVIDIA H100 GPUs** on A3 Mega VMs, leveraging Google Cloud's AI Hypercomputer architecture for workload-specific optimization. The platform achieved **33x cost reduction** since 2022 launch and runs **13.5x cheaper** than competitors using commercial APIs through proprietary Transformer optimizations, KV cache optimization, and inter-turn caching techniques. Founded by Noam Shazeer and Daniel De Freitas (creators of Transformer architecture and Google's LaMDA), the company emphasizes cost-efficient inference for consumer-scale AI. Infrastructure includes **Kubernetes** for ML workload orchestration, Google Compute Engine for distributed training, and comprehensive MLOps for model deployment.

**Salary Ranges**: ML Infrastructure Engineer $150K-$350K | Research Engineer (ML Systems, Post-Training, Safety) $225K-$400K | Software Engineer Safety $150K-$300K

---

## Sources

**Engineering Blogs**:
- [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai/)
- [Introducing Character](https://blog.character.ai/introducing-character/)
- [Character.AI Open Sources pipeling-sft](https://blog.character.ai/character-ai-open-sources-pipeling-sft-a-scalable-framework-for-fine-tuning-moe-llms-like-deepseek-v3/)

**Google Cloud Case Studies**:
- [Why Character.ai chose Spanner and AlloyDB for PostgreSQL](https://cloud.google.com/blog/products/databases/why-characterai-chose-spanner-and-alloydb-for-postgresql)

**AI/ML Job Postings**:
- [ML Infrastructure Engineer](https://jobs.ashbyhq.com/character/) - $150K-$350K + equity
- [Research Engineer, AI Safety & Alignment](https://jobs.ashbyhq.com/character/) - $225K-$400K + equity
- [Research Engineer, ML Systems](https://jobs.ashbyhq.com/character/) - $225K-$400K + equity
- [Research Engineer, Post-Training](https://jobs.ashbyhq.com/character/) - $225K-$400K + equity
- [Software Engineer, Safety](https://jobs.ashbyhq.com/character/) - $150K-$300K + equity
- [Character.AI Careers - Ashby](https://jobs.ashbyhq.com/character/)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/characterai/salaries)

---

*Last updated: November 30, 2025*
