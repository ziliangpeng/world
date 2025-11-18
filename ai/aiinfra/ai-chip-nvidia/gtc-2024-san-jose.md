# GTC 2024 San Jose

> **NVIDIA GPU Technology Conference** | March 18-21, 2024

---

## Event Details

**Dates:** March 18-21, 2024

**Location:** San Jose McEnery Convention Center, San Jose, CA

**Significance:** The first in-person GTC since 2019, marking a major return for the AI community and the official unveiling of the Blackwell architecture.

---

## Keynote

**Date:** March 18, 2024

**Speaker:** Jensen Huang, NVIDIA CEO

**Watch Keynote:** [YouTube Link](https://www.youtube.com/watch?v=Y2F8yisiS6E)

---

## Major Announcements

> Announcements listed in keynote presentation order

| Announcement | Details | Timestamp |
| :--- | :--- | :--- |
| **Blackwell Architecture & GB200 Superchip** | Unveiling of the Blackwell GPU architecture, GB200 Grace Blackwell Superchip, and its core features. | [~0:26:07](https://www.youtube.com/watch?v=Y2F8yisiS6E&t=26m7s) |
| **Next-Gen Interconnects & Networking** | Introduction of 5th-Gen NVLink, NVLink Switch, and X800 series network switches. | [~0:34:38](https://www.youtube.com/watch?v=Y2F8yisiS6E&t=34m38s) |
| **Blackwell Reliability & Security (RAS Engine, Confidential Computing)** | Features like the RAS Engine, confidential computing, and decompression engine for robust and secure AI. | [~0:37:40](https://www.youtube.com/watch?v=Y2F8yisiS6E&t=37m40s) |
| **NVIDIA NIM (Inference Microservices) & AI Enterprise 5.0** | New software for deploying enterprise generative AI, including NIMs and AI Enterprise 5.0. | [~0:51:55](https://www.youtube.com/watch?v=Y2F8yisiS6E&t=51m55s) |
| **Cloud & Enterprise Partnerships** | Deepened partnerships with major cloud providers (AWS, Google Cloud, Oracle, Microsoft) and enterprises (Dell, SAP, ServiceNow, Cohesity, Snowflake, NetApp). | [~1:00:51](https://www.youtube.com/watch?v=Y2F8yisiS6E&t=1h0m51s) |
| **Project GR00T & Robotics** | A new foundation model for humanoid robots, updates to the Isaac robotics platform, and Jetson Thor. | [~1:06:00](https://www.youtube.com/watch?v=Y2F8yisiS6E&t=1h6m0s) |
| **Earth-2 & Weather Prediction (Corrdi)** | A digital twin of Earth for high-resolution weather prediction using generative AI (Corrdi model). | [~1:10:00](https://www.youtube.com/watch?v=Y2F8yisiS6E&t=1h10m0s) |
| **NVIDIA Healthcare & BioNemo** | Advancements in computational drug discovery using BioNemo microservices for virtual screening. | [~1:13:00](https://www.youtube.com/watch?v=Y2F8yisiS6E&t=1h13m0s) |
| **AI Co-pilots for Chip Design** | Using AI chatbots (NIMs) to assist in chip design, exemplified by the Compute Trace Library (CTL). | [~1:21:00](https://www.youtube.com/watch?v=Y2F8yisiS6E&t=1h21m0s) |
| **Vector Databases & Chat with Data** | Encoding proprietary information into vector databases for AI interaction and chat. | [~1:24:00](https://www.youtube.com/watch?v=Y2F8yisiS6E&t=1h24m0s) |
| **Omniverse Cloud APIs & Apple Vision Pro** | New Omniverse Cloud APIs for digital twin development and integration with Apple Vision Pro. | [~1:37:00](https://www.youtube.com/watch?v=Y2F8yisiS6E&t=1h37m0s) |
| **AI Chip Roadmap** | Future architectures including Blackwell Ultra, Rubin, and Feynman. | [~1:43:40](https://www.youtube.com/watch?v=Y2F8yisiS6E&t=1h43m40s) |

---

### Details on Major Announcements

#### Blackwell Architecture & GB200 Superchip
The keynote's centerpiece was the unveiling of the Blackwell GPU architecture, designed to power the next generation of AI. This included the introduction of the GB200 Grace Blackwell Superchip, which connects two B200 Tensor Core GPUs to a Grace CPU. Blackwell-based products were announced to be available later in 2024. Jensen Huang highlighted Blackwell's 208 billion transistors and 20 petaflops of performance, emphasizing its role in "AI factories."

#### Next-Generation Interconnects & Networking
To support the massive communication needs of trillion-parameter models, NVIDIA announced two key interconnect advancements:
- **5th-Generation NVLink:** Providing 1.8 TB/s of bidirectional throughput per GPU, with computation capabilities directly in the network.
- **NVLink Switch:** A dedicated chip with 50 billion transistors and four 1.8 TB/s NVLinks, enabling every GPU to communicate with every other GPU at full speed.
- **X800 Series Switches:** A new series of network switches, including the Quantum-X800 (InfiniBand) and Spectrum-X800 (Ethernet), capable of 800 Gb/s speeds, expected in 2025.

#### Blackwell Reliability & Security (RAS Engine, Confidential Computing)
Blackwell introduces several features for robust and secure AI deployments:
- **RAS Engine:** A reliability engine that performs 100% self-tests of every gate and memory bit, ensuring high utilization of supercomputers.
- **Confidential Computing:** New capabilities to encrypt data at rest, in transit, and during computation, ensuring secure processing of private data.
- **Decompression Engine:** A high-speed decompression engine capable of 800 GB/s, accelerating data movement and keeping Blackwell GPUs fed.

#### NVIDIA NIM (Inference Microservices) & AI Enterprise 5.0
NVIDIA introduced NVIDIA Inference Microservices (NIM) to simplify the deployment of enterprise generative AI. NIMs are pre-built, optimized containers for state-of-the-art AI models, packaged with all dependencies (CUDA, TensorRT-LLM, Triton Inference Server) and optimized for various GPU configurations. This was announced alongside new tools in NVIDIA AI Enterprise 5.0.

#### Cloud & Enterprise Partnerships
NVIDIA announced extensive partnerships for Blackwell and its AI ecosystem:
- **Cloud Providers:** AWS, Google Cloud,, Microsoft Azure, and Oracle Cloud Infrastructure are adopting the Blackwell platform and offering related services like DGX Cloud. AWS is building a 2.2 exaflops system.
- **Enterprises:** Dell is creating the "Dell AI Factory" built on NVIDIA infrastructure, and SAP will integrate NVIDIA RAG capabilities into its Joule copilot. ServiceNow is building virtual assistants. Cohesity is building its Gaia generative AI agent. Snowflake is building co-pilots for its data warehouse. NetApp is helping build chatbots and co-pilots.

#### Project GR00T & Robotics
NVIDIA signaled a major push into robotics and physical AI with the announcement of Project GR00T, a general-purpose foundation model for humanoid robots. This was accompanied by significant updates to the Isaac robotics platform and the Jetson Thor robotic computer.

#### Earth-2 & Weather Prediction (Corrdi)
NVIDIA introduced Earth-2, a digital twin of the Earth for high-resolution weather prediction. The Corrdi generative AI model, trained on high-resolution radar data, can super-resolve weather events from 25 km to 2 km resolution with 1,000x speed and 3,000x energy efficiency compared to conventional models.

#### NVIDIA Healthcare & BioNemo
Advancements in computational drug discovery were highlighted with NVIDIA BioNemo microservices. These NIMs enable generative screening paradigms for protein structure prediction, molecule generation, and docking, allowing for the rapid generation and screening of candidate molecules in minutes.

#### AI Co-pilots for Chip Design
NVIDIA showcased the use of AI chatbots (NIMs) as co-pilots for chip designers. An example was given of a Llama 2-based NIM that understood and generated code for NVIDIA's internal Compute Trace Library (CTL), significantly boosting designer productivity.

#### Vector Databases & Chat with Data
The keynote emphasized the ability to encode proprietary information (like PDFs or internal data) into vector databases. These "smart databases" allow users to "chat with data," enabling AI to understand the meaning of internal company information and generate relevant responses.

#### Omniverse Cloud APIs & Apple Vision Pro
New Omniverse Cloud APIs were introduced to better connect the platform for creating digital twins with other industry tools. A key announcement was the ability to stream Omniverse content to the Apple Vision Pro, merging high-fidelity digital twins with mixed reality.

#### AI Chip Roadmap
Jensen Huang outlined NVIDIA's aggressive future AI chip architecture roadmap:
- **Blackwell Ultra:** An enhanced version of Blackwell, expected later in 2025.
- **Vera Rubin:** The next major architecture, anticipated in late 2026, with Rubin Ultra following in 2027.
- **Feynman:** Teased for a 2028 release.

---

## Key Themes

### A Platform, Not Just a Chip
The GTC 2024 keynote made it clear that Blackwell is not just a new GPU, but an entire platform. The announcements of the Blackwell GPU, 5th-Gen NVLink, and X800 series switches were presented as a holistic system designed to power the next generation of data centers and AI factories at massive scale.

### From Training to Deployment
With the introduction of NVIDIA Inference Microservices (NIM), NVIDIA showed a significant focus on simplifying the deployment of AI. The message was about moving beyond just providing the tools for training models and also providing the software infrastructure to help enterprises easily run those models in production.

### Embodied AI: The Next Frontier
A major theme of the keynote was "embodied AI" or "physical AI." The unveiling of Project GR00T for humanoid robots and the significant updates to the Isaac robotics platform signaled a clear strategic direction: enabling AI to move from the digital world to interact with and automate the physical world.

### The Power of the Ecosystem
The keynote was a showcase of NVIDIA's vast and growing ecosystem. By announcing that every major cloud provider and enterprise hardware partner (like Dell and SAP) would be adopting the Blackwell platform, NVIDIA reinforced its position as the underlying standard for the AI industry.

---

## Sessions & Content

**Format:**
- The first in-person GTC since 2019, also available virtually.
- Featured hundreds of sessions, hands-on workshops, and networking opportunities with AI experts.

**Availability:** Sessions and training are available to explore online.

---

## Strategic Context

**Significance:**
- Marked the major return of the in-person GTC format.
- Served as the official launch platform for the Blackwell architecture, setting the stage for the next generation of AI infrastructure.
- Solidified NVIDIA's strategy of providing full-stack solutions, from silicon (Blackwell) to software (NIM) and robotics (GR00T).