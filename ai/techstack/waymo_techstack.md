# Waymo - Technology Stack

**Company:** Waymo LLC (Alphabet subsidiary)
**Founded:** 2009 (Google Self-Driving Car Project), spun out December 2016
**Focus:** Autonomous driving / Robotaxi service
**Headquarters:** Mountain View, California

---

## Non-AI Tech Stack

Waymo operates a **fleet of 2,500 robotaxis** across **five U.S. cities** (San Francisco, Los Angeles, Phoenix, Austin, Atlanta) with expansion to Miami, Dallas, Houston, San Antonio, and Orlando announced November 2025. The infrastructure is built on **Alphabet's technical infrastructure** with **Google Cloud** (likely **TPUs** for training, given Alphabet integration). Data pipelines process **20+ million autonomous miles** from real-world driving plus third-party datasets (**NHTSA Crash Data**, **Naturalistic Driving Study**). The **6th-generation Waymo Driver** hardware features **13 cameras, 4 lidar, 6 radar, and external audio receivers (EARs)** with overlapping fields of view providing **360-degree coverage up to 500 meters away**. Sensors operate in extreme heat, fog, rain, and hail with preventive maintenance measures for winter climates. ML frameworks include **TensorFlow** and **JAX** for distributed training, with the **Waymo Open Dataset** (TFRecord format) available for research. **Simulation City** (proprietary full-trip simulation system) and **Waymax** (JAX-based open-source simulator) enable rapid iteration without physical road testing. The company provides **250,000+ paid rides per week** (1 million+ miles monthly) and has accumulated **100 million fully autonomous miles** (96 million rider-only miles) as of July 2025.

**Salary Ranges**: Software Engineer median $375K total comp (L3 $232K, L4 $312K, L5 $429K, L6 $587K, L7 $986K) | Hardware Engineer L4 $200K, L6 $479K | Systems Engineer L4 $270K, L6 $483K

---

## AI/ML Tech Stack

### Waymo Foundation Model - Embodied AI for Autonomous Driving

**What's unique**: Waymo developed a proprietary **Foundation Model** that marries "AV-specific ML advances with general world knowledge of VLMs" to create purpose-built embodied AI for driving. According to Drago Anguelov (Head of Research), the model combines **autonomous vehicle expertise** accumulated over **16 years** (since 2009) with **generative AI capabilities** from Large Language Models and Vision-Language Models. The Foundation Model significantly enhances the Waymo Driver across multiple functions: **scene interpretation** (understanding complex road environments), **driving plan generation** (deciding optimal actions), and **trajectory prediction** (forecasting other agents' behavior). Unlike general-purpose VLMs, Waymo's model is specifically trained on driving contexts, leveraging the company's **100 million autonomous miles** and expertise in safety-critical decision-making. The model also powers **simulation realism** by modeling future world states and predicting vehicle behavior, accelerating development cycles.

### 6th-Generation Waymo Driver - Optimized Hardware & Sensor Fusion

The **6th-generation sensor suite** represents a strategic evolution: **13 cameras, 4 lidar, 6 radar, and an array of external audio receivers (EARs)**. Waymo achieved **significantly reduced cost** while delivering **even more resolution, range, and compute power** through strategic sensor placement and reduction in overall sensor count while maintaining **safety-critical redundancies**. The three complementary sensing modalities (camera, lidar, radar) provide **overlapping coverage** across the entire vehicle perimeter, enabling safe navigation in diverse conditions including **extreme heat, fog, rain, and hail**. Each sensor type compensates for others' limitations — cameras provide high-resolution color imagery, lidar measures precise distances via laser reflection, radar penetrates weather to detect objects, and EARs capture audio signals (sirens, horns). The system was designed for **modular configuration**, allowing Waymo to swap sensing components to match specific operating environments (e.g., enhanced sensor cleaning for colder climates). Performance metrics show the 6th-gen suite is **on track to begin operating without a human behind the wheel in about half the time** compared to previous generations.

### Fleet Learning - Collective Intelligence Across 100M+ Miles

**What makes it different**: Waymo's ML models **learn from each mile driven by the entire fleet** rather than individual vehicles learning in isolation. The company has accumulated **100 million fully autonomous miles** (96 million rider-only miles) across **dozens of operational cities**, creating the world's largest real-world autonomous driving dataset. ML Infrastructure teams develop innovative techniques to **automatically identify interesting interactions** from tens of millions of miles, focusing training on edge cases, safety-critical scenarios, and diverse road conditions. The Driver has a "unique ability to learn from road events across the entire fleet, advancing its capabilities at incredible speeds." Each of the **250,000+ weekly paid rides** (as of 2025) generates data that improves perception, prediction, and planning models while maintaining industry-leading safety standards. This collective learning approach enables rapid capability expansion — new cities leverage knowledge from existing deployments rather than starting from scratch.

### Simulation City - Synthetic Journey Generation at Scale

Waymo's **Simulation City** represents the most advanced autonomous driving simulation system, **automatically synthesizing entire journeys** to assess the Waymo Driver's performance from full-length trips (20-minute urban routes to 11-hour long-haul deliveries). Unlike isolated scenario testing, Simulation City generates **"a random outcome from a statistical distribution of the real world,"** evaluating how the Driver responds to varied human behaviors — both typical and edge cases. The system leverages **20+ million autonomous miles**, third-party datasets (NHTSA Crash Data, Naturalistic Driving Study Data), and **daily updates from production vehicle telemetry** across operational cities. **Sensor simulation** recreates realistic inputs including **"raindrops on our sensors,"** environmental effects like lighting changes, and solar glare during specific conditions. **Agent behavior modeling** uses ML to generate realistic trajectory distributions for traffic participants. The platform enables **parallel evaluation impossible with physical fleets**, removing constraints on vehicle count, simulation timing, and geographic coverage. Simulation City significantly speeds up data generation, producing **fully synthetic journeys** without requiring real-world test miles.

### ML Infrastructure - TensorFlow, JAX & Distributed Training

Waymo's **ML Infrastructure team** develops models in **Perception and Planning** that are core to autonomous driving software, creating libraries and tools that enhance **TensorFlow and JAX** to address **scalability, reliability, and performance challenges** at massive scale. The infrastructure includes components for **distributed training** (parallelizing across hundreds of accelerators), **job scheduling and resource management** (optimizing compute allocation), **data distribution** (feeding petabyte-scale datasets), and **model synchronization** (coordinating parameter updates). Waymo operates **"cutting-edge compute infrastructure and advanced closed-loop simulation systems"** that enable rapid model iteration. The team has invested in **off-board infrastructure** for effective large-scale model training, deployment, and evaluation of neural networks and the system as a whole. **Waymax**, Waymo's open-source JAX-based simulator, allows researchers to run simulations on hardware accelerators using scenarios from the **Waymo Open Motion Dataset**, democratizing autonomous driving research.

### Real-World Deployment at Commercial Scale

Waymo operates the **only fully autonomous, rider-only robotaxi service at scale** in the United States. The fleet breakdown: **~800-1,000 vehicles in San Francisco, 700 in Los Angeles, 500 in Phoenix, 200 in Austin, and 100 in Atlanta**, with **250,000+ paid rides per week** (over 1 million miles monthly). In 2024 alone, Waymo's all-electric autonomous fleet drove **over 25 million miles**. The company became the **first to offer public service without safety drivers** in October 2020. November 2025 expansion brings fully autonomous operations to **five additional cities**: Miami (launched November 18, 2025), Dallas, Houston, San Antonio, and Orlando. This deployment scale is unmatched — no other autonomous vehicle company operates rider-only commercial service across ten cities. Waymo serves **hundreds of thousands of weekly riders** across diverse environments (urban cores, suburbs, freeways, airports), generating continuous real-world validation data that feeds back into ML model improvement.

### Waymo Open Dataset - Research Democratization

Waymo released the **Waymo Open Dataset** (high-resolution sensor data from diverse conditions) for **non-commercial research use**, enabling the broader research community to advance autonomous driving. The dataset is integrated with **TensorFlow Datasets** and supports both **TensorFlow and JAX**, providing version-specific packages (waymo-open-dataset-tf-2-11-0, waymo-open-dataset-tf-2-12-0). Data format is **TFRecord** (TensorFlow's native format) with conversion tools for formats like KITTI. The dataset includes perception labels, motion prediction scenarios, and behavioral cloning data. **Waymax** (JAX-based simulation library) uses scenarios from the Waymo Open Motion Dataset, allowing researchers to evaluate autonomous driving algorithms on hardware accelerators without access to Waymo's internal infrastructure.

**Salary Ranges**: Software Engineer median $375K total comp (L3 $232K, L4 $312K, L5 $429K, L6 $587K, L7 $986K) | Hardware Engineer L4 $200K, L6 $479K

---

## Sources

**Waymo Technical Blogs**:
- [Meet the 6th-Generation Waymo Driver](https://waymo.com/blog/2024/08/meet-the-6th-generation-waymo-driver)
- [Behind the Innovation: AI & ML at Waymo](https://waymo.com/blog/2024/10/ai-and-ml-at-waymo)
- [Simulation City: Waymo's Most Advanced Simulation System](https://waymo.com/blog/2021/07/simulation-city)
- [The Waymo Driver's Rapid Learning Curve](https://waymo.com/blog/2023/08/the-waymo-drivers-rapid-learning-curve)
- [Cities, Freeways, Airports: How We've Built a Scalable Autonomous Driver](https://waymo.com/blog/2022/05/howwevebuiltascalableautonomousdriver)
- [Waypoint - Official Waymo Blog](https://waymo.com/blog)

**Company & History**:
- [Waymo About Page](https://waymo.com/about/)
- [Waymo Wikipedia](https://en.wikipedia.org/wiki/Waymo)
- [Waymo - A Google X Moonshot](https://x.company/projects/waymo/)
- [Waymo FAQ](https://waymo.com/faq/)

**ML Infrastructure & Open Source**:
- [Waymo Open Dataset](https://waymo.com/open/)
- [Waymax: Accelerated Simulator for Autonomous Driving Research](https://waymo.com/research/waymax/)
- [GitHub: Waymo Open Dataset](https://github.com/waymo-research/waymo-open-dataset)
- [Waymo Open Dataset - TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/waymo_open_dataset)

**Scale & Operations**:
- [Waymo Stats 2025: Fleet Size, Miles Driven, Coverage](https://www.thedriverlessdigest.com/p/waymo-stats-2025-funding-growth-coverage)
- [Waymo Reaches 100M Fully Autonomous Miles](https://www.therobotreport.com/waymo-reaches-100m-fully-autonomous-miles-across-all-deployments/)
- [Waymo Now Has 2,000 Vehicles in Their US Fleet](https://www.thedriverlessdigest.com/p/waymo-now-has-2000-vehicles-in-their)

**Job Postings & Compensation**:
- [Waymo Careers](https://waymo.com/careers/)
- [Open Career Opportunities - Google Careers](https://www.google.com/about/careers/applications/jobs/results/88815398314484422-open-career-opportunities/)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/waymo/salaries)

---

*Last updated: November 30, 2025*
