# Nuro AI - Technology Stack Analysis

**Company:** Nuro, Inc.
**Founded:** 2016
**Focus:** Autonomous delivery vehicles and self-driving technology
**Headquarters:** Mountain View, California, USA

---

## Overview

This document provides a comprehensive analysis of Nuro's technology stack, derived from job postings, engineering blogs, technical documentation, and public information. Nuro has built an AI-first autonomous driving system combining proprietary hardware, advanced machine learning infrastructure, and scalable cloud systems.

---

## 1. Programming Languages & Core Frameworks

### Primary Languages
- **Python** - Primary language for ML/AI development, data processing, and automation
- **C++** - Core language for performance-critical systems, robotics, and embedded software
- **Go** - Used for infrastructure services, routing systems, and backend services
- **Rust** - Data mining infrastructure, high-performance computing tasks

### Machine Learning Frameworks
- **PyTorch** - Primary deep learning framework
- **JAX** - Advanced ML framework for scalability and optimization
- **TensorFlow** - Supporting ML framework
- **Keras** - High-level ML API

### Robotics & Middleware
- **ROS (Robot Operating System)** - Robotics middleware for sensor integration and control systems
- Custom robotics frameworks for autonomous vehicle control

---

## 2. Cloud Infrastructure (Google Cloud Platform)

Nuro has built its entire infrastructure on Google Cloud Platform, leveraging various managed services and custom systems.

### Database Systems

#### AlloyDB for PostgreSQL
- **Primary transactional database** for core workloads
- Manages metadata for logs, trips, simulations, and real-time autonomy issues
- **AlloyDB AI** with ScaNN indexing for vector embeddings
- Achieves 20,000+ high-precision vector search results in seconds
- Zero-downtime migration capability

#### BigQuery
- Main backend for analytical metrics
- Enables precise evaluation and validation of the Nuro Driver
- Receives near real-time data from AlloyDB via Datastream

#### Cloud Spanner
- Stores log namespace metadata
- Distributed, globally consistent database

#### Supporting Databases
- **Firestore** - Document database for various applications
- **Memorystore (Redis)** - Caching layer for high-performance access
- **Redis** - Used in BATES system for task status monitoring

### Storage & Data Pipeline

#### Cloud Storage
- Primary storage for:
  - Autonomy logs from on-road operations
  - Simulation records
  - ML evaluation files
  - Sensor data (LiDAR, camera, radar telemetry)
- **Scale**: Hundreds of petabytes of data managed
- Global availability across multiple operating cities

#### Transfer Appliance
- Physical hardware device for depot-to-cloud data transfer
- Supports SFTP, SCP, and NFS mounting
- **Hourly automated transfers** from vehicle depots
- Encryption at-rest and in-flight
- Enables nightly builds with fresh data

#### Datastream
- Change data capture (CDC) system
- Near real-time replication from AlloyDB to BigQuery
- Enables live analytics on transactional data

### Compute & Orchestration

#### BATES (Batch Task Execution System)
- **Custom-built distributed task orchestration platform**
- Manages **millions of tasks daily**
- DAG-based workflow management
- Features:
  - Custom message queue with dynamic priority scheduling
  - Automatic task assignment based on job size, start time, and real-time factors
  - Auto-scaling strategies aware of job SLAs and cloud VM cost tiers
  - Horizontal scalability for compute optimization
- Architecture:
  - OLTP database + Redis for task status monitoring
  - Smart scheduling for millions of GPU hours of daily simulation
  - Optimizes job turnaround time based on SLA requirements

#### Cloud Infrastructure
- Provisions and maintains all cloud resources
- Generic compute platform for mission-critical workflows:
  - Data processing pipelines
  - Simulation environments
  - ML training and evaluation

---

## 3. Hardware & Compute Platform

### Vehicle Compute Platform

#### NVIDIA DRIVE AGX Thor
- **Primary compute SoC** for autonomous driving
- Automotive-grade performance and reliability
- Combines with Nuro's proprietary ECU modules

#### NVIDIA DriveOS
- Safety-certified operating system
- Runs on NVIDIA DRIVE hardware
- Provides real-time performance for autonomy stack

### Sensor Suite

#### Multi-Modal Sensor Architecture
- **LiDAR**
  - 360° coverage
  - Multiple units for redundancy
  - High-precision 3D environmental perception

- **Cameras**
  - High-definition multi-view cameras
  - Multiple camera positions for complete coverage
  - Supports 2D/3D perception tasks

- **Radar**
  - All-weather sensing capability
  - Velocity and range detection
  - Redundant perception modality

- **IMU (Inertial Measurement Unit)**
  - Tracks vehicle acceleration and orientation
  - Supports localization and state estimation

#### Sensor Fusion
- Multi-sensor approach for robust object detection
- ML-based fusion with geometric approaches
- Reliable performance in adverse conditions (night, rain)
- Redundancy for safety-critical operations

### Vehicle Hardware
- **Custom-built autonomous delivery vehicles** (R2 generation, R3 in development)
- Proprietary sensor integrations
- External airbags for pedestrian safety
- Redundant safety systems (braking, steering, compute)
- Electric powertrain, zero emissions

---

## 4. ML & AI Infrastructure

### Training Infrastructure

#### Nuro ML Scheduler
- **Custom-built distributed training orchestration system**
- Accepts training jobs from ML engineers and allocates compute resources
- Optimizes resource allocation across the cluster
- **Framework Support**: Horovod, PyTorch, JAX
- **Integration**: Built on Kubeflow open-source training operators
- **Abstraction Layer**: Shared core scheduling logic across frameworks and accelerator types
- Enables fast ML development velocity by avoiding custom recovery implementations per framework
- **Reference**: [Scaling ML Training at Nuro](https://www.nuro.ai/blog/scaling-ml-training-at-nuro)

#### GPU/TPU Accelerators
- **NVIDIA V100 GPUs** - Previous generation training
- **NVIDIA A100 GPUs** - High-performance training workloads
- **NVIDIA H100 GPUs** - Latest generation for cutting-edge training
- **Google Cloud TPUs** - Specialized AI training accelerators
- Large number of distributed training jobs run daily across diverse hardware
- Multi-accelerator support through unified scheduler

#### Distributed Training Systems
- Large-scale distributed model training across multiple nodes
- Support for multi-GPU, multi-node training configurations
- **Training Strategies Supported**:
  - Data parallelism
  - Model parallelism
  - Mixed precision training
- Training runtime performance optimization
- Profiling and optimization tools for ML frameworks
- Custom recovery mechanisms for fault tolerance

#### ML Frameworks & Tools
- **PyTorch** - Primary deep learning framework for research and production
- **JAX** - Advanced ML framework for scalability and optimization, high-performance computing
- **TensorFlow** - Supporting ML framework
- **Keras** - High-level ML API
- **Horovod** - Distributed training framework
- Scalability and optimization focus across all frameworks
- Experience with multithreading, x86 architecture
- GPU/CPU/FPGA compute modality support

#### FTL Model Compiler Framework
- **Nuro's proprietary ML model compiler** for inference optimization
- **Multi-framework support** via ONNX (Open Neural Network Exchange)
- **TensorRT integration** for GPU acceleration
- **Architecture**:
  - Orchestrator Segmenter: Identifies and compiles graph portions for TensorRT
  - Sub-compiler passes: Multiple optimization passes for binary generation
  - Segment Breaker: Isolates problematic subgraphs for specialized compilation
- **Multi-GPU inference** support with ~27% latency reduction on perception detectors
- Custom kernel injection (PyTorch GPU kernels into compiled graphs)
- Precision management (FP32 conversion for accuracy-sensitive operations)
- **Reference**: [FTL Model Compiler Framework](https://www.nuro.ai/blog/ftl-model-compiler-framework)

#### Reinforcement Learning Infrastructure
- **Safe RL at scale** - Years of driving experience trained in single hour
- **CIMRL (Combining Imitation and Reinforcement Learning)** for safe autonomous driving
- Distributed training systems for RL workloads
- **Simulation-based training** - Impossible to collect safety-critical data in real world
- Policy training completed in single work day
- **Integration**: Simulation stack for environment interaction
- **References**:
  - [CIMRL Paper](https://www.nuro.ai/blog/cimrl-combining-imitation-reinforcement-learning-for-safe-autonomous-driving)
  - [Enabling Reinforcement Learning at Scale](https://www.nuro.ai/blog/enabling-reinforcement-learning-at-scale)

### Data Platform

#### Comprehensive Data Management
- Manages data, labels, and metrics for the Nuro Driver
- **Scale**: Hundreds of petabytes
- Data types:
  - Raw sensor logs (LiDAR, camera, radar)
  - Annotated training data
  - Simulation outputs
  - Real-world driving scenarios

#### Data Annotation Systems
- Multi-domain annotation capabilities:
  - **2D/3D perception** (object detection, segmentation)
  - **Mapping** (HD map generation)
  - **Behavior trajectory** annotation
  - **Language/text** annotation for scene understanding
- Supports labeling infrastructure for ML training pipelines

### Simulation Platform

#### Large-Scale Simulation
- **Millions of hours** of simulation run daily
- Combines on-road data with synthetic augmentation
- Smart agents for realistic traffic simulation
- Advanced reconstruction techniques
- Fault injection and latency measurement
- Validates autonomy stack at scale before real-world deployment

#### Capabilities
- Scene-level simulation (millions of scenes in DAGs)
- Physics-based environment modeling
- Sensor simulation (LiDAR, camera, radar)
- Edge case generation and testing

---

## 5. Autonomy Stack Components

### AI-First Architecture
Nuro employs an **ML-first autonomy stack** with robust fallback systems, emphasizing safety and continuous improvement through data scaling.

### Perception
- **Multi-sensor object detection** (LiDAR, camera, radar fusion)
- ML models combined with geometric approaches
- Functionality across multiple sensor modalities
- Occlusion handling and visibility-based decision making
- Robust performance in adverse weather (rain, night, fog)

### Localization & Mapping
- HD map-based localization
- Sensor stack integration for precise positioning
- Route selection and navigation
- Real-time map updates

### Prediction & Planning
- **Multi-hypothesis prediction system**
- 10-second future trajectory rollout
- Edge case handling (e.g., children, unpredictable pedestrians)
- Speed modulation for safety
- Cautious progression with occlusion awareness

### Remote Operations & Telemetry
- **Remote diagnostics and log download**
- **Over-the-air (OTA) software updates**
- Fleet management and monitoring
- Real-time telemetry streaming
- Video streaming for remote assistance
- Teleoperation capabilities

---

## 6. Complete AI/ML Stack Overview

Nuro's AI stack is purpose-built for autonomous driving at scale, combining research, infrastructure, and production systems.

### ML Research & Development

#### Research Areas
- **Perception**: Vision transformers, foundation models, multi-sensor fusion
- **Prediction**: Multi-agent trajectory prediction, scene understanding
- **Planning**: Imitation learning, reinforcement learning, hybrid approaches ([CIMRL](https://www.nuro.ai/blog/cimrl-combining-imitation-reinforcement-learning-for-safe-autonomous-driving))
- **End-to-End Driving**: Unified models integrating perception → planning
- **3D Vision**: NeRF, Gaussian Splatting, 3D reconstruction

#### ML Frameworks Stack
- **Primary**: PyTorch - Research and production deep learning
- **High-Performance**: JAX - Advanced ML, numerical computing
- **Supporting**: TensorFlow, Keras
- **Distributed Training**: Horovod - Multi-GPU/multi-node training

### ML Infrastructure & Training

#### Training Orchestration
- **[Nuro ML Scheduler](https://www.nuro.ai/blog/scaling-ml-training-at-nuro)** - Custom resource management system
  - Built on Kubeflow training operators
  - Supports PyTorch, JAX, Horovod workloads
  - Optimizes GPU/TPU allocation across cluster

#### Compute Resources
- **GPUs**: NVIDIA V100, A100, H100 - Training & fine-tuning
- **TPUs**: Google Cloud TPUs - Large-scale training
- **Clusters**: Distributed training on hundreds of accelerators
- **Orchestration**: Kubernetes for job scheduling

#### Training Infrastructure
- **Data Parallelism**: Sharded training across multiple GPUs
- **Model Parallelism**: Large model distribution
- **Mixed Precision**: FP16/FP8 training for efficiency
- **Quantization**: [AWQ](https://arxiv.org/abs/2306.00978), [AQT](https://arxiv.org/abs/2106.07152), [GPTQ](https://arxiv.org/abs/2210.17323) methods
- **Fault Tolerance**: Automatic checkpointing and recovery

### Model Deployment & Inference

#### FTL Model Compiler
- **[FTL Framework](https://www.nuro.ai/blog/ftl-model-compiler-framework)** - Nuro's proprietary model compiler
- **ONNX Support**: Multi-framework model conversion
- **TensorRT Integration**: GPU inference acceleration
- **Multi-GPU Inference**: ~27% latency reduction on perception models
- **Custom Kernels**: PyTorch kernel injection for specialized ops

#### Inference Optimization
- **Precision Management**: FP32/FP16/INT8 conversion
- **Graph Optimization**: Subgraph compilation for performance
- **Edge Deployment**: On-vehicle inference optimization
- **Batching**: Dynamic batching for throughput

### Reinforcement Learning Pipeline

#### Safe RL Infrastructure
- **[Simulation-Based Training](https://www.nuro.ai/blog/enabling-reinforcement-learning-at-scale)** - Years of driving in hours
- **Distributed RL**: Scalable policy training across cluster
- **Sim-to-Real Transfer**: Domain adaptation techniques
- **Safety Constraints**: Recovery policies and safety guarantees

#### RL Approaches
- **Imitation Learning**: Learning from expert demonstrations
- **Reinforcement Learning**: Policy optimization in simulation
- **[CIMRL](https://www.nuro.ai/blog/cimrl-combining-imitation-reinforcement-learning-for-safe-autonomous-driving)**: Hybrid imitation + RL for safe driving
- **Multi-Task Learning**: Shared representations across tasks

### Data Pipeline

#### Data Management (Hundreds of Petabytes)
- **Storage**: Google Cloud Storage, HDFS-equivalent systems
- **Databases**: AlloyDB, BigQuery, Spanner
- **CDC**: Datastream for real-time replication
- **Versioning**: Dataset versioning and lineage tracking

#### Data Processing
- **ETL Pipelines**: Sensor data preprocessing and transformation
- **Annotation**: Multi-domain labeling (2D/3D perception, trajectories, mapping)
- **Quality Control**: Automated validation and quality checks
- **Feature Engineering**: Derived features for ML models

### Simulation Platform

#### Large-Scale Simulation
- **Scale**: Millions of GPU hours daily
- **Physics Engine**: High-fidelity vehicle and sensor simulation
- **Scenario Generation**: Diverse driving scenarios and edge cases
- **Smart Agents**: Realistic traffic participant behavior
- **Sensor Sim**: LiDAR, camera, radar simulation

### MLOps & Production

#### Experiment Tracking
- Model versioning and experiment management
- Hyperparameter optimization
- A/B testing framework for models

#### Monitoring & Observability
- Real-time model performance monitoring
- Data drift detection
- Model degradation alerts
- Production metrics dashboards

### AI Stack Team Roles & Compensation

#### ML Infrastructure Engineers
- **Senior/Staff Software Engineer, ML Infrastructure**
  - Salary: $167,200 - $250,800
  - [Apply](https://boards.greenhouse.io/nuro)

- **Tech Lead Manager, ML Training Infrastructure**
  - Salary: $222,775 - $333,925
  - Focus: Quantization, distributed training, team leadership
  - [Job Details](https://jobs.icehouseventures.co.nz/companies/nuro/jobs/59711721-tech-lead-manager-ml-training-infrastructure)

#### ML Research Scientists
- **Lead ML Research Scientist**
  - Salary: $250,000 - $350,000+
  - Focus: Research leadership, publications

- **Senior ML Research Scientist, End-to-End Driving**
  - Salary: $200,000 - $300,000
  - Focus: E2E learning, multi-task models

- **Senior ML Research Scientist, Perception Foundation Encoder**
  - Salary: $200,000 - $300,000
  - Focus: Vision transformers, foundation models

- **ML Research Scientist, Behavior Planning**
  - Salary: $180,000 - $280,000
  - Focus: RL, imitation learning, planning

#### New Grad / Early Career
- **Software Engineer, AI Platform - New Grad**
  - Salary: $145,000 - $170,000
  - [Apply](https://www.nuro.ai/early-career)

*All roles include performance bonuses, equity grants, and comprehensive benefits*

**Career Pages**:
- [Nuro Careers](https://www.nuro.ai/careers)
- [Greenhouse Job Board](https://boards.greenhouse.io/nuro)
- [Early Career Opportunities](https://www.nuro.ai/early-career)

---

## 7. Development Tools & Engineering Practices

### Engineering Productivity
- **Build systems and CI/CD pipelines**
- Automated testing frameworks
- Code review and collaboration tools
- Version control (Git-based workflows)

### Technical Infrastructure Group
Owns fundamental engineering services:
- Generic compute platform for mission-critical workflows
- Storage management service (hundreds of PB)
- Cloud infrastructure provisioning and maintenance
- Engineering productivity tools (build, CI/CD, testing)

### Software Development Requirements (from job postings)
- **Experience with**:
  - Large-scale distributed systems
  - Data storage and processing systems
  - Advanced algorithms in C++ and Python
  - Software performance tuning and optimization
  - Robotics software frameworks
  - x86 architecture, multithreading

---

## 7. Key Technical Innovations

### BATES - Custom Task Orchestration
Nuro developed a proprietary batch task execution system capable of:
- Managing millions of daily tasks
- Running millions of GPU hours of simulation
- Dynamic priority scheduling based on SLAs
- Cost-aware auto-scaling across cloud VM tiers

### AlloyDB AI Vector Search
- Migrated all vector embeddings to AlloyDB AI
- ScaNN indexing for high-performance similarity search
- Outperforms alternative indexing methods (IVF, HNSW)
- Substantial operational cost reduction
- Horizontal scalability for multiple embedding types

### Transfer Appliance with Online Transfer
- Hourly automated data sync from vehicle depots to cloud
- Reduces data migration time by more than 50%
- Enables nightly builds with fresh on-road data
- Critical for rapid iteration on autonomy software

### AI-First Autonomy with Fallback
- ML-first stack that improves with data scaling
- Robust fallback architecture for safety
- Continuous improvement through the "Data Loop"
- Efficiently utilizes collected on-road data

---

## 8. Inferred Technologies

Based on job requirements and industry standards, Nuro likely uses:

### Version Control & Collaboration
- **Git** for source control
- GitHub or similar platforms for code collaboration
- Code review tools

### Containerization & Orchestration
- **Docker** for containerization
- **Kubernetes** for orchestration (common with GCP)
- Container registries (Google Container Registry/Artifact Registry)

### Monitoring & Observability
- Cloud monitoring and logging (Google Cloud Operations)
- Custom telemetry and metrics systems
- Performance profiling tools

### Build & CI/CD
- **Bazel** (common in robotics/ML companies, Google heritage)
- Automated testing frameworks
- Continuous integration pipelines on cloud infrastructure

### Data Processing
- Apache Beam or Dataflow for data pipelines
- Batch and stream processing systems
- ETL pipelines for sensor data

---

## 9. AI Infrastructure & Research Roles (Detailed)

### ML Infrastructure Engineering Roles

#### Tech Lead Manager, ML Training Infrastructure
- **Location**: [Mountain View, CA](https://www.nuro.ai/careers)
- **Compensation**: $222,775 - $333,925 base + bonus + equity

**Key Responsibilities**:
- Drive adoption of quantized training methods ([AWQ](https://arxiv.org/abs/2306.00978), [AQT](https://arxiv.org/abs/2106.07152), [GPTQ](https://arxiv.org/abs/2210.17323), FP-8 training)
- Lead design and implementation of model training efficiency initiatives
- Implement low-bit quantization, pruning, and knowledge distillation
- Mentor and grow team of engineers and researchers
- Optimize ML infrastructure for autonomous driving workloads

**Required Experience**:
- 6+ years in ML infrastructure, distributed training, or ML systems engineering
- Hands-on experience with quantization methods (AWQ, AQT, FP-8)
- Deep understanding of model compression techniques
- Leadership experience managing technical teams

**Reference**: [Tech Lead Manager, ML Training Infrastructure @ Nuro](https://jobs.icehouseventures.co.nz/companies/nuro/jobs/59711721-tech-lead-manager-ml-training-infrastructure)

#### Senior/Staff Software Engineer, ML Infrastructure
- **Location**: Mountain View, CA
- **Focus**: Distributed Training and Optimization

**Key Responsibilities**:
- Build scalable systems with Autonomy and ML infrastructure teams
- Develop model serving platform for large-scale simulations
- Support reinforcement learning (RL) training infrastructure
- Optimize distributed training performance
- Design fault-tolerant training systems

**Required Skills**:
- Strong coding, software design, debugging in Python or C++
- Experience in large-scale distributed systems
- Data storage and processing systems expertise
- Advanced algorithms using C++ and Python
- Machine learning, multithreading, x86 architecture
- Software performance tuning and optimization

#### Software Engineer, ML Infrastructure - Optimization
- **Location**: Mountain View, CA
- **Focus**: Training Runtime Performance

**Key Responsibilities**:
- Optimize training runtime for faster model iteration
- Build infrastructure for model training at scale
- Profile and improve training job performance
- Integrate with Nuro ML Scheduler for resource allocation
- Support multiple ML frameworks (PyTorch, JAX, TensorFlow)

**Technical Stack**:
- PyTorch, JAX, TensorFlow
- Kubernetes for job orchestration
- NVIDIA GPUs (V100, A100, H100) and Google Cloud TPUs
- Distributed training frameworks (Horovod)

### ML Research Scientist Roles

#### Lead ML Research Scientist
- **Location**: Mountain View, CA
- **Senior Research Leadership Role**

**Key Responsibilities**:
- Lead research initiatives in autonomous driving
- Bring ML advancements to AV domain
- Work closely with autonomy teams on perception, prediction, planning
- Move towards end-to-end autonomous driving approaches
- Publish research at top-tier conferences (CVPR, NeurIPS, ICRA)

**Required Expertise**:
- PhD in Computer Science, Robotics, or related field (preferred)
- Strong publication record in ML/CV conferences
- Deep learning, computer vision, robotics
- Leadership in research direction and team mentoring

#### Senior ML Research Scientist, End-to-End Autonomous Driving
- **Location**: Mountain View, CA
- **Focus**: E2E learning for autonomous systems

**Key Responsibilities**:
- Research and develop end-to-end learning approaches
- Integrate perception, prediction, planning in unified models
- Scale learning systems with large datasets
- Collaborate with ML infrastructure on training systems
- Conduct experiments and publish findings

**Technical Focus**:
- End-to-end neural network architectures
- Multi-task learning for autonomous driving
- Imitation learning and reinforcement learning
- Large-scale model training
- Safety-aware learning methods

**Reference**: [CIMRL: Combining Imitation and Reinforcement Learning](https://www.nuro.ai/blog/cimrl-combining-imitation-reinforcement-learning-for-safe-autonomous-driving)

#### Senior ML Research Scientist, Perception Foundation Encoder
- **Location**: Mountain View, CA
- **Focus**: Vision foundation models for AV perception

**Key Responsibilities**:
- Develop foundation models for multi-sensor perception
- Research vision transformers and modern architectures
- Scale perception models with diverse driving data
- Integrate with 3D detection and tracking systems
- Optimize for on-vehicle inference

**Technical Stack**:
- Vision transformers, CNN architectures
- Multi-modal learning (camera, LiDAR fusion)
- Self-supervised and semi-supervised learning
- 3D computer vision, point cloud processing
- Model compression for deployment

### ML Research Scientist, Behavior Planning
- **Location**: Mountain View, CA
- **Focus**: Learning-based planning and decision making

**Key Responsibilities**:
- Develop ML models for behavior planning
- Research imitation learning from expert demonstrations
- Integrate reinforcement learning for edge cases
- Work on simulation-to-reality transfer
- Collaborate with perception and prediction teams

**Technical Approaches**:
- **Imitation Learning**: Learning from human driving data
- **Reinforcement Learning**: Safe RL for autonomous driving
- **Hybrid Methods**: [CIMRL approach](https://www.nuro.ai/blog/cimrl-combining-imitation-reinforcement-learning-for-safe-autonomous-driving)
- Behavior cloning with safety guarantees

### Job Requirements Summary Across Roles

#### Programming Languages
- **Primary**: Python (all ML roles), C++ (infrastructure & systems)
- **Supporting**: Go (infrastructure), Rust (data mining)

#### ML Frameworks & Libraries
- PyTorch - Primary framework for research and production
- JAX - High-performance ML, especially for research
- TensorFlow - Supporting framework
- Horovod - Distributed training
- ONNX - Model interoperability
- TensorRT - Inference optimization

#### Infrastructure & Tools
- Kubernetes - Container orchestration
- Kubeflow - ML on Kubernetes
- Google Cloud Platform services
- Docker - Containerization
- Git version control

#### Hardware Experience
- NVIDIA GPUs (V100, A100, H100)
- Google Cloud TPUs
- Multi-GPU and multi-node training
- Performance profiling and optimization

#### Domain Knowledge
- Autonomous driving systems
- Computer vision and perception
- 3D geometry and reconstruction
- Sensor fusion (LiDAR, camera, radar)
- Safety-critical systems
- Real-time inference constraints

### Compensation Ranges

| Role Level | Base Salary Range |
|------------|------------------|
| New Grad Software Engineer | $145,000 - $170,000 |
| Senior ML Infrastructure Engineer | $167,200 - $250,800 |
| Staff ML Engineer | $200,000 - $300,000+ |
| Tech Lead Manager, ML Training Infra | $222,775 - $333,925 |
| Senior/Lead Research Scientist | $200,000 - $350,000+ |

*Plus performance bonuses, equity grants, and comprehensive benefits*

---

## 10. Additional Technical Role Categories

### Data Platform Engineering
- **Languages**: Python, Scala, SQL
- **Technologies**: Spark, Airflow, BigQuery, Datastream
- **Focus**: ETL pipelines, data warehouse, real-time processing

### Autonomy Engineering
- **Languages**: C++, Python
- **Domains**: Perception, prediction, planning, controls
- **Skills**: Sensor fusion, path planning, decision making

### Systems & Infrastructure
- **Languages**: C++, Python, Go, Rust
- **Focus**: Cloud infrastructure, distributed systems, reliability
- **Tools**: Kubernetes, monitoring, observability

---

## 11. Summary

Nuro has built a comprehensive, scalable technology stack centered on:

1. **Complete AI/ML Stack** (Section 6) - Purpose-built for autonomous driving
   - Research: PyTorch/JAX, vision transformers, end-to-end learning
   - Infrastructure: Nuro ML Scheduler, V100/A100/H100 GPUs, TPUs
   - Production: FTL compiler, ONNX/TensorRT, multi-GPU inference
   - RL Pipeline: Safe RL at scale, CIMRL hybrid approach

2. **Google Cloud Platform** as the foundation for compute, storage, and data services
   - AlloyDB, BigQuery, Spanner for databases
   - Hundreds of petabytes managed via Cloud Storage
   - BATES custom orchestration system

3. **Programming Stack**
   - Python and C++ as primary development languages
   - Go for infrastructure services, Rust for high-performance tasks
   - Multiple ML frameworks: PyTorch, JAX, TensorFlow, Horovod

4. **Hardware Platform**
   - NVIDIA DRIVE AGX Thor for in-vehicle compute
   - Multi-sensor suite: LiDAR, cameras, radar, IMU
   - Custom autonomous delivery vehicles (R2, R3 in development)

5. **ML Infrastructure & Research**
   - Distributed training on hundreds of GPUs/TPUs
   - Advanced simulation running millions of GPU hours daily
   - Quantization (AWQ, AQT, GPTQ) for efficient deployment
   - Research in perception, prediction, planning, E2E driving

6. **Competitive Compensation**
   - New Grad: $145K-$170K
   - Senior ML Infrastructure: $167K-$251K
   - Tech Lead Manager: $223K-$334K
   - Senior/Lead Research Scientists: $200K-$350K+

The tech stack reflects Nuro's focus on autonomous delivery at scale, with emphasis on cost efficiency, rapid iteration, safety-critical systems, and cutting-edge AI research.

---

## Sources

### Company & Product Information
- [Nuro Careers](https://www.nuro.ai/careers)
- [Work at Nuro](https://www.nuro.ai/careersitem)
- [Nuro Driver™](https://www.nuro.ai/nuro-driver)

### Engineering & Technical Blogs
- [The Nuro Autonomy Stack](https://www.nuro.ai/blog/the-nuro-autonomy-stack)
- [Scaling Autonomy in the Cloud](https://www.nuro.ai/blog/scaling-autonomy-in-the-cloud)
- [Scaling ML Training at Nuro](https://www.nuro.ai/blog/scaling-ml-training-at-nuro) - ML Scheduler details
- [FTL Model Compiler Framework](https://www.nuro.ai/blog/ftl-model-compiler-framework) - Inference optimization
- [CIMRL: Combining Imitation and Reinforcement Learning](https://www.nuro.ai/blog/cimrl-combining-imitation-reinforcement-learning-for-safe-autonomous-driving) - Safe RL research
- [Enabling Reinforcement Learning at Scale](https://www.nuro.ai/blog/enabling-reinforcement-learning-at-scale) - RL infrastructure
- [Introducing the Nuro Driver's Next-Generation Sensor Architecture](https://www.nuro.ai/blog/introducing-the-nuro-drivers-next-generation-sensor-architecture)

### Google Cloud Case Studies
- [Nuro drives autonomous innovation with AlloyDB for PostgreSQL](https://cloud.google.com/blog/products/databases/nuro-drives-autonomous-innovation-with-alloydb-for-postgresql/)
- [Nuro builds the future of delivery and robotics with Google](https://cloud.google.com/blog/products/storage-data-transfer/nuro-builds-the-future-of-delivery-and-robotics-with-google)

### NVIDIA Partnership
- [Nuro to License Its Autonomous Driving System - NVIDIA Blog](https://blogs.nvidia.com/blog/nuro-driver/)
- [NVIDIA and Nuro Optimize Autonomous Driving with Real-Time 3D and AI](https://www.xrstager.com/en/nvidia-and-nuro-optimize-autonomous-driving-with-real-time-3d-and-ai)

### AI Infrastructure & ML Engineering Roles

**Job Boards**:
- [Jobs at Nuro - Greenhouse](https://boards.greenhouse.io/embed/job_board?for=nuro) - Main job board
- [Jobs at Nuro - Alternative Greenhouse Link](https://job-boards.greenhouse.io/embed/job_board?for=nuro)
- [Early Career Opportunities at Nuro](https://www.nuro.ai/early-career)

**Specific ML Infrastructure Roles**:
- [Tech Lead Manager, ML Training Infrastructure](https://jobs.icehouseventures.co.nz/companies/nuro/jobs/59711721-tech-lead-manager-ml-training-infrastructure) - $222K-$334K, Quantization expertise
- [Machine Learning Research Scientist, ML Planning](https://boards.greenhouse.io/embed/job_app?for=nuro&token=5108162) - Behavior planning role
- [Senior/Staff Software Engineer, ML Infrastructure](https://boards.greenhouse.io/nuro) - Multiple openings
- [Software Engineer, AI Platform - New Grad](https://www.builtinsf.com/job/software-engineer-ai-platform-new-grad/3654504) - $145K-$170K

**Job Aggregators**:
- [Nuro Jobs - LinkedIn](https://www.linkedin.com/jobs/nuro-jobs) - 52+ positions
- [Nuro Jobs - Built In San Francisco](https://www.builtinsf.com/company/nuro/jobs)
- [Nuro Jobs - Built In (National)](https://builtin.com/company/nuro/jobs)
- [Nuro Jobs - Wellfound](https://wellfound.com/company/nuro/jobs)
- [Nuro AI Jobs - ZipRecruiter](https://www.ziprecruiter.com/Jobs/Nuro-Ai)
- [Nuro Jobs - Indeed](https://www.indeed.com/q-Nuro-jobs.html)
- [Nuro Jobs - Jobright.ai](https://jobright.ai/) - AI-powered job matching

### Research Papers & Academic Resources
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [AQT: Accurate Quantized Training](https://arxiv.org/abs/2106.07152)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)

---

*Document created: November 30, 2025*
*Research methodology: Job posting analysis, engineering blog review, technical documentation, and public sources*
