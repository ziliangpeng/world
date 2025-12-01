# Nuro - Technology Stack

**Company:** Nuro, Inc.
**Founded:** 2016
**Focus:** Autonomous delivery vehicles
**Headquarters:** Mountain View, California

---

## Non-AI Tech Stack

Nuro's infrastructure runs entirely on **Google Cloud Platform** with a polyglot backend using **Python** (data processing, automation), **C++** (performance-critical systems, robotics), **Go** (infrastructure services), and **Rust** (data mining). The data platform processes hundreds of petabytes through **AlloyDB for PostgreSQL** (vector search with ScaNN indexing), **BigQuery** (analytics), **Cloud Spanner** (distributed data), and **Cloud Storage**. Infrastructure is orchestrated through **Kubernetes** with BATES, a custom distributed task orchestration system managing millions of daily tasks and GPU hours of simulation. The vehicle platform runs on **NVIDIA DRIVE AGX Thor** with **DriveOS** for safety-critical operations. Backend services use **ROS** (Robot Operating System) for sensor integration. Development infrastructure includes **Bazel** (build system), **Docker** (containerization), and automated CI/CD pipelines.

**Salary Ranges**: Software Engineer $167K-$251K | Tech Lead Manager $223K-$334K | Staff roles $200K-$300K+

---

## AI/ML Tech Stack

Nuro's ML infrastructure centers on the **Nuro ML Scheduler**, a custom distributed training orchestration system built on Kubeflow that supports **PyTorch** (primary framework), **JAX** (high-performance ML), **TensorFlow**, and **Horovod** (distributed training). Training runs on **NVIDIA V100/A100/H100 GPUs** and **Google Cloud TPUs** across hundreds of accelerators. The **FTL Model Compiler** optimizes inference through ONNX and TensorRT integration, achieving 27% latency reduction on perception models. ML stack supports data parallelism, model parallelism, mixed precision training (FP16/FP8), and quantization methods (AWQ, AQT, GPTQ). Reinforcement learning infrastructure enables **CIMRL** (Combining Imitation and Reinforcement Learning) for safe autonomous driving, training years of driving experience in hours through large-scale simulation. Research areas include vision transformers, foundation models, multi-sensor fusion (LiDAR/camera/radar), end-to-end driving, 3D vision (NeRF, Gaussian Splatting), and multi-agent trajectory prediction.

**Salary Ranges**: ML Research Scientist $180K-$280K | Senior/Lead Scientist $200K-$350K+ | ML Infrastructure Engineer $167K-$251K | Tech Lead Manager ML Training $223K-$334K

---

## Sources

**Engineering Blogs**:
- [Scaling ML Training at Nuro](https://www.nuro.ai/blog/scaling-ml-training-at-nuro)
- [FTL Model Compiler Framework](https://www.nuro.ai/blog/ftl-model-compiler-framework)
- [CIMRL: Combining Imitation and Reinforcement Learning](https://www.nuro.ai/blog/cimrl-combining-imitation-reinforcement-learning-for-safe-autonomous-driving)
- [Enabling Reinforcement Learning at Scale](https://www.nuro.ai/blog/enabling-reinforcement-learning-at-scale)
- [The Nuro Autonomy Stack](https://www.nuro.ai/blog/the-nuro-autonomy-stack)
- [Scaling Autonomy in the Cloud](https://www.nuro.ai/blog/scaling-autonomy-in-the-cloud)

**Google Cloud Case Studies**:
- [Nuro drives autonomous innovation with AlloyDB](https://cloud.google.com/blog/products/databases/nuro-drives-autonomous-innovation-with-alloydb-for-postgresql/)
- [Nuro builds the future of delivery with Google](https://cloud.google.com/blog/products/storage-data-transfer/nuro-builds-the-future-of-delivery-and-robotics-with-google)

**AI/ML Job Postings**:
- [Tech Lead Manager, ML Training Infrastructure](https://jobs.icehouseventures.co.nz/companies/nuro/jobs/59711721-tech-lead-manager-ml-training-infrastructure) - $223K-$334K
- [Machine Learning Research Scientist roles](https://boards.greenhouse.io/nuro) - Multiple positions
- [Nuro Careers - All Positions](https://www.nuro.ai/careers)
- [Early Career Opportunities](https://www.nuro.ai/early-career)

---

*Last updated: November 30, 2025*
