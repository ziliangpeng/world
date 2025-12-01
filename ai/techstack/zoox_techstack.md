# Zoox - Technology Stack

**Company:** Zoox, Inc. (Amazon subsidiary)
**Founded:** 2014 (Acquired by Amazon: 2020)
**Focus:** Autonomous robotaxi service
**Headquarters:** Foster City, California

---

## Non-AI Tech Stack

Zoox operates on a **hybrid cloud infrastructure**, combining an on-premise supercomputer cluster with **AWS** for high-performance computing. The infrastructure uses **Amazon EC2** with **Slurm** (open-source workload manager) for dynamic resource allocation, **Amazon EKS** (Elastic Kubernetes Service) for long-running services and batch jobs, and **Amazon S3** with **Amazon FSx for Lustre** for storage. Workflow orchestration runs on **Apache Airflow**. Each vehicle generates **up to 4TB of raw data per hour**, uploaded via **AWS Data Transfer Terminals** at secure locations with speeds up to **400 Gbps**. For advanced workloads, Zoox uses **EC2 Capacity Blocks** to reserve up to **2,000 GPUs in a single cluster** for large-scale simulations. The robotaxi platform runs on **multiple NVIDIA GPUs** for real-time sensor processing. Monitoring infrastructure uses **Amazon CloudWatch**. Development tools include standard version control, CI/CD pipelines, and containerization with Docker.

**Salary Ranges**: Software Engineer $149K-$391K (L1-L6, median $281K) | ML Infrastructure roles ~$180K-$332K

---

## AI/ML Tech Stack

### Core Architecture & Unique Approach

Zoox's AI stack follows a **three-stage pipeline**: **perception** (understanding current state), **prediction** (forecasting 8 seconds into the future), and **planning** (motion planning and decision-making). What makes this unique is the **8-second prediction horizon** – significantly longer than industry standard, enabling the robotaxi to predict trajectories of vehicles, people, and even animals far into the future. The system also features a **Collision Avoidance System (CAS)**, a parallel and complementary redundant safety layer running alongside the main AI autonomy stack, architecting safety redundancy at the software level.

### ML Frameworks & Training

The ML infrastructure is built on **PyTorch**, **TensorFlow**, and **JAX**, with fluency across all three frameworks required for cross-team collaboration (Perception, Prediction, Planning, Simulation, Systems Engineering). Training leverages the hybrid cloud model: on-premise supercomputer for baseline workloads plus AWS EC2 for burst capacity. The Prediction & Behavior ML team develops deep learning algorithms that learn behaviors from millions of road miles collected by the test fleet (Toyota Highlanders with identical sensor architecture to the robotaxi). Models are trained using **CUDA programming** and optimized for deployment with **NVIDIA TensorRT**.

### Inference Optimization & Deployment

**What's unique**: Zoox achieves **2-6x speedup in FP32 and 9-19x in INT8** compared to TensorFlow through TensorRT optimization. On-vehicle inference runs all perception, prediction, and planning neural networks on NVIDIA GPUs via TensorRT, while cloud-based **high-throughput batch inference** uses **Ray Serve on Amazon EKS**. Zoox developed custom tools for TensorRT deployment, validation, and maintenance, including conversion checkers and validation pipelines to ensure networks can be successfully optimized. This dual-deployment strategy (on-vehicle TensorRT + cloud Ray Serve) enables both real-time autonomous driving and large-scale batch processing for simulation and validation.

### Simulation & Validation Infrastructure

Zoox runs **state-of-the-art simulation using NVIDIA GPUs** with millions of GPU hours daily. **Adversarial simulations** – carefully crafted scenarios designed to test system limits and uncover edge cases – differentiate their approach from standard simulation testing. Generative AI creates diverse scenarios for comprehensive validation before real-world deployment. The simulation infrastructure processes the massive datasets (4TB/hour per vehicle) to continuously improve the AI stack through data-driven iteration.

### Sensor Fusion & Multi-Modal Learning

The robotaxi integrates **cameras, lidar, radar, long-wave infrared sensors, and microphones** – a more comprehensive sensor suite than most competitors. The onboard computing rapidly fuses multi-modal sensor data to provide coherent environmental understanding, then flows through perception → prediction → planning → control systems. Foundation models are being developed for unified multi-modal understanding across sensor types.

**Salary Ranges**: ML Engineer Collision Avoidance $173K-$246K | ML Engineer Sensor Fusion $180K-$289K | Prediction ML Engineer $221K-$319K | Learned Trajectory ML Engineer $230K-$332K | Senior/Staff roles $210K-$300K

---

## Sources

**Engineering & Technical**:
- [How the Zoox robotaxi predicts everything, everywhere, all at once - Amazon Science](https://www.amazon.science/latest-news/how-the-zoox-robotaxi-predicts-everything-everywhere-all-at-once)
- [Optimizing NVIDIA TensorRT Conversion for Real-time Inference - NVIDIA](https://developer.nvidia.com/blog/optimizing-nvidia-tensorrt-conversion-for-real-time-inference-on-autonomous-vehicles/)
- [The 'full-stack' behind autonomous driving - Zoox](https://zoox.com/autonomy/)
- [ML infrastructure at Zoox - AWS re:Invent](https://www.antstack.com/talks/reinvent24/ml-infrastructure-at-zoox-that-powers-autonomous-driving-of-robotaxis-amz201/)

**AWS Case Studies**:
- [Zoox Case Study - AWS](https://aws.amazon.com/solutions/case-studies/zoox/)
- [How AWS fuels Zoox's autonomous robotaxis - SiliconANGLE](https://siliconangle.com/2025/10/31/aws-fuels-zooxs-autonomous-robotaxis/)

**AI/ML Job Postings**:
- [ML Engineer - Sensor Fusion Detection](https://zoox.com/careers/81f7fbe7-4fd3-4814-b07d-2fd2177e32a4) - $180K-$289K
- [Prediction ML Engineer](https://zoox.com/careers/f61979b0-eb89-43ad-b11b-544b2d46baa8) - $221K-$319K
- [Learned Trajectory ML Engineer](https://www.zoox.com/careers/24acfe86-aa1b-4aa6-903b-99b4596e42cf) - $230K-$332K
- [ML Engineer - Collision Avoidance System](https://zoox.com/careers/aacead0e-8309-48ca-af2c-813d084f6233) - $173K-$246K
- [Senior/Staff SWE - Collision Avoidance](https://zoox.com/careers/9b898524-93e6-478a-ad0f-8a5a514ba502) - $210K-$300K
- [Senior/Staff ML Engineer - Foundation Models](https://zoox.com/careers/2eee4eb4-be0f-4317-972c-39c9a900ea63)
- [Principal SWE - ML Infrastructure](https://jobs.lever.co/zoox/d2d21053-e4b7-4b70-b409-db5763e14914)
- [Zoox Careers - All Positions](https://zoox.com/careers)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/zoox/salaries/software-engineer)

---

*Last updated: November 30, 2025*
