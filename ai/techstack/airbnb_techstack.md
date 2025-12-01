# Airbnb - Technology Stack

**Company:** Airbnb, Inc.
**Founded:** 2008
**Focus:** Online marketplace for lodging and vacation rentals
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

Airbnb's infrastructure runs entirely on **AWS** (EC2, RDS MySQL, S3) with a polyglot backend using **Ruby** (Rails monolith and core services), **Java** (scalable backend services), **Scala** (data pipelines, financial systems), **Kotlin** (monorepo services), **Python** (automation, data science), **Node.js** (APIs), and **Go** (infrastructure). The architecture evolved from a single Rails monolith to **1000+ microservices** managed through **100+ Kubernetes clusters** (30+ cluster types) with **Spinnaker** (continuous delivery) and **OneTouch** (infrastructure as code). Frontend uses **React** (web) and **React Native** (mobile). The data platform processes **35+ billion events/day** through **Apache Kafka** (streaming), **Apache Spark** (batch/streaming processing), **Apache Airflow** (workflow orchestration, created by Airbnb), **Apache Hive** (data warehouse on 11+ PB HDFS), and **Presto/Trino** (query engine). Visualization via **Apache Superset** (created by Airbnb). Build system migrated from Gradle to **Bazel** over 4.5 years, achieving 3-5x faster builds.

**Salary Ranges**: Software Engineer $187K-$924K (G7-G11, median $496K) | Senior SWE $190K-$250K | Staff SWE $250K-$350K

---

## AI/ML Tech Stack

Airbnb's ML platform **Bighead** is framework-agnostic, supporting **TensorFlow**, **PyTorch**, **Keras**, **MXNet**, **Scikit-learn**, **XGBoost**, **H2O**, and **R**. Platform components include a unified model building API, **Redspot** (centrally-managed Jupyter notebook service), and **ML Automator** (batch inference engine built on Airflow and Spark). Recent focus on **GenAI infrastructure** with tools for rapid AI application development, improved iteration speed, AI safety, and guardrails. Data infrastructure provides 35B+ events/day for ML training through Kafka and Spark pipelines. Analytics migrated from **Apache Druid** to **StarRocks** for the Minerva platform (30K+ metrics, 7K+ dimensions). Production ML includes batch inference, real-time serving, model versioning, A/B testing frameworks, and comprehensive MLOps. ML use cases span search ranking, recommendations, pricing optimization, fraud detection, personalization, and AI-powered translation.

**Salary Ranges**: Senior ML Engineer $191K-$223K | Staff ML Engineer $204K-$255K | Senior Staff ML Engineer $244K-$305K | Data Scientist $241K-$827K (L3-L7, median $355K)

---

## Sources

**Engineering Blogs**:
- [The Airbnb Tech Blog - Medium](https://medium.com/airbnb-engineering)
- [Data Infrastructure at Airbnb](https://medium.com/airbnb-engineering/data-infrastructure-at-airbnb-8adfb34f169c)
- [Rearchitecting Airbnb's Frontend](https://medium.com/airbnb-engineering/rearchitecting-airbnbs-frontend-5e213efc24d2)
- [Dynamic Kubernetes Cluster Scaling at Airbnb](https://medium.com/airbnb-engineering/dynamic-kubernetes-cluster-scaling-at-airbnb-d79ae3afa132)
- [Continuous Delivery at Airbnb](https://airbnb.tech/infrastructure/continuous-delivery-at-airbnb/)
- [MySQL in the cloud at Airbnb](http://nerds.airbnb.com/mysql-in-the-cloud-at-airbnb/)

**ML Platform & Data**:
- [Bighead: Airbnb's ML platform - O'Reilly Strata](https://conferences.oreilly.com/strata/strata-ny-2018/public/schedule/detail/69383.html)
- [How Superset and Druid Power Real-Time Analytics at Airbnb](https://www.datacouncil.ai/talks/how-superset-and-druid-power-real-time-analytics-at-airbnb)

**AI/ML Job Postings**:
- [Senior ML Engineer](https://careers.airbnb.com/positions/) - $191K-$223K + equity
- [Staff ML Engineer, AI Translation](https://careers.airbnb.com/positions/6838683/) - $204K-$255K + equity
- [Senior Staff ML Engineer, Relevance & Personalization](https://careers.airbnb.com/positions/6237467/) - $244K-$305K + equity
- [Senior Staff ML Engineer, AI Safety & Guardrail](https://careers.airbnb.com/positions/7159569/) - $244K-$305K + equity
- [Senior SWE, ML Infrastructure](https://careers.airbnb.com/positions/7081518/)
- [Staff SWE, AI Enablement](https://careers.airbnb.com/positions/7081653/)
- [Airbnb Careers - All Positions](https://careers.airbnb.com/)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/airbnb/salaries)

---

*Last updated: November 30, 2025*
