# Modal - Technology Stack

**Company:** Modal Labs (Modal)
**Founded:** 2021
**Focus:** Serverless platform for AI/ML workloads
**Headquarters:** New York City (Soho), with offices in Stockholm and San Francisco

---

## Non-AI Tech Stack

Modal operates a **custom-built infrastructure** rejecting Docker and Kubernetes in favor of speed, built entirely in **Rust** and **Python**. The core runtime uses **gVisor** (Modal was the first to run gVisor with GPUs) with a proprietary **Rust FUSE filesystem** implementing lazy-loading content-addressed storage and tiered caching (memory page cache, local SSD, zonal cache servers, regional CDN, blob storage). HTTP infrastructure uses **Rust** with **Hyper HTTP library**, **Tokio async runtime**, and **tokio-tungstenite** for WebSockets, translating HTTP/WebSocket requests into serverless function calls via **ASGI events** encoded as **Protocol Buffers**. Ingress runs on a **TCP Network Load Balancer** fronting a **Kubernetes cluster** with **Caddy** (TLS termination) and **modal-http** (Rust), though serverless functions run on Modal's custom runtime, not K8s. **VolumeFS** provides global mutable storage via FUSE backed by **Google Spanner** (file tree metadata) and content-addressed CDN. Real-time collaboration uses **Redis Streams** with operational transformation. Modal Notebooks integrate **CodeMirror 6** (editor), **Ruff WebAssembly** (auto-formatting), **LSP** (completions/hover), and **Claude 4** for edit suggestions, with kernel execution via **ZeroMQ** (Jupyter protocol). Founded by **Erik Bernhardsson** (CEO, former Spotify/Better.com CTO, creator of Luigi) and **Akshat Bubna** (CTO), Modal raised **$111M total** ($87M Series B July 2025 at $1.1B valuation) from **Lux Capital, Redpoint, Amplify, Definition Capital, Creandum**, and Elad Gil.

**Salary Ranges**: Software Engineer L3 $194K | L4 $283K | L5 $401K total comp (base $157K-$220K + bonus $21K-$42K + equity $78K-$157K)

---

## AI/ML Tech Stack

### Custom Container Runtime - gVisor with Rust FUSE

**What's unique**: Modal deliberately rejected Docker and Kubernetes to prioritize **sub-second container launch times**. The founding team built a **custom Rust-based container runtime using gVisor** as the execution layer, becoming the **first company to run gVisor on machines with GPUs** and contributing to production-readiness of gVisor GPU support. Modal bypasses Docker entirely, using lightweight runtimes (runc/gVisor) that point to a root filesystem with JSON configuration. The **Rust FUSE filesystem** (initially prototyped in Python, rewritten in Rust for performance) implements **lazy-loading** that loads only a lightweight metadata index at startup rather than downloading entire container layers. Files load on-demand through a **content-addressed storage system** with **tiered caching** (memory, SSD, zonal cache, regional CDN, blob storage). This architecture exploits that only small image fractions are read and massive overlap exists between images, achieving "very high cache efficiency." Result: Modal can **build 100GB containers and boot 100 instances in seconds** — startup times reduced from ~60 seconds to seconds, enabling true elastic scaling.

### Python SDK - Decorator-Based Serverless Deployment

**What makes it different**: Modal exposes everything through a **Python SDK** where developers simply **decorate Python functions** to deploy to the cloud with GPUs. Developers declare **container images and hardware directly in Python** without writing YAML, Dockerfiles, or Kubernetes manifests. The SDK handles task scheduling onto workers, automatic scaling, and lifecycle management. Code launches in the cloud within **one second** and scales to **hundreds of GPU-backed workers within seconds**. This developer experience differentiates Modal from infrastructure-heavy platforms requiring deep DevOps expertise — data scientists and ML engineers ship production workloads without learning container orchestration. The SDK integrates with Modal Sandboxes (isolated processes supporting hundreds of CPUs, top-tier NVIDIA GPUs, gigabytes of disk) that automatically pause when idle and restart within seconds when users return.

### Elastic GPU Infrastructure - Scale to Zero

Modal provides access to **NVIDIA B200, H200, H100, A100 (40GB/80GB), L4, T4, and L40S GPUs** with **up to 8 GPUs per instance** (up to 1,536 GB GPU RAM). The platform scales to **thousands of GPUs across clouds** with **no quotas or reservations**, then **scales back to zero** when not in use via **entirely usage-based pricing** ($0.001097/sec H100, $0.000694/sec A100-80GB, $0.000583/sec A100-40GB). Developers switch GPU types with a simple `gpu="H100"` flag in Python code without worrying about hardware procurement. Modal shares compute resources across workloads using their custom scheduler, enabling cost-efficient autoscaling impossible with reserved GPU clusters. The infrastructure handles common AI workloads including inference for custom models, fine-tuning, training, and GPU-accelerated tasks.

### Fast HTTP/WebSocket Ingress - Rust Performance

Modal built **modal-http**, a Rust-based service that translates HTTP/WebSocket requests into serverless function calls, fundamentally differing from traditional reverse proxies. The implementation uses **Hyper HTTP library** and **Tokio async runtime** for concurrent request processing, streaming HTTP requests as **ASGI events** (binary Protocol Buffers) rather than raw HTTP/1.1. This enables **4 GiB request bodies** (versus AWS Lambda's 6 MB limit), **unlimited streaming response sizes**, and proper backpressure handling via TCP flow control. Switching from Python-based ingress to Rust **decreased 502 Bad Gateway errors by 99.7%** through clearer error handling and request lifetime tracking. The architecture buffers request bodies in 1 MiB chunks or 2-millisecond intervals, supporting production-scale AI inference workloads.

### Modal Notebooks - Cloud GPU Notebooks Booting in Seconds

Modal built **cloud GPU notebooks that boot in seconds** rather than minutes, combining the lazy-loading container filesystem with Jupyter protocol integration. Notebooks run kernels inside **Modal Sandboxes** with access to hundreds of CPUs and top-tier NVIDIA GPUs. A daemon called `modal-kernelshim` translates **ZeroMQ Jupyter protocol messages** into HTTP calls through Modal's control plane, enabling remote kernel access that appears instantaneous. The editor features **Language Server Protocol implementation** (completions, hover, semantic tokens), **Ruff WebAssembly** for auto-formatting, and **Claude 4 integration** for edit suggestions. **Real-time collaboration** uses operational transformation via **Redis Streams** with **CodeMirror 6** handling presence indicators and concurrent cursors across multiple users. **VolumeFS** provides global mutable storage via a FUSE filesystem backed by distributed infrastructure.

### Why Custom Infrastructure Matters

Erik Bernhardsson's founding thesis: **"make the feedback loops fast"** drives all architectural decisions. After seven years at Spotify and six years as CTO, he observed that data teams face unique challenges requiring **production data**, **frequent infrastructure changes**, and **hardware diversity** (GPUs, high-memory instances) — forcing data practices into backend-normative frameworks (Docker/K8s) kills productivity. Modal's custom file systems, container engines, schedulers, and image builders deliver **one-second launch times** and **scale to 100 containers in seconds** — performance impossible with traditional orchestration. Most VCs thought the team "crazy building custom file systems and container engines," but the approach succeeded: Modal reached **8-figure revenue in one year**, tripled headcount, and raised **$111M** at **$1.1B valuation**. The team includes creators of popular open-source projects (Seaborn, Luigi), academic researchers, and international olympiad medalists, operating from NYC, Stockholm, and San Francisco.

**Salary Ranges**: Software Engineer L3 $194K | L4 $283K | L5 $401K total comp | Base $157K-$220K + bonus $21K-$42K + equity $78K-$157K

---

## Sources

**Modal Technical Blogs**:
- [Lambda on Hard Mode: Inside Modal's Web Infrastructure](https://modal.com/blog/serverless-http)
- [Inside Modal Notebooks: How We Built a Cloud GPU Notebook That Boots in Seconds](https://modal.com/blog/notebooks-internals)
- [How Modal Speeds Up Container Launches in the Cloud](https://modal.com/blog/speeding-up-container-launches)
- [A10 vs. A100 vs. H100 - Which One Should You Choose?](https://modal.com/blog/gpu-types)
- [NVIDIA Showdown: A100s vs H100s vs H200s](https://modal.com/blog/h200-vs-h100-vs-a100)
- [Introducing: H100s on Modal](https://modal.com/blog/introducing-h100)

**Founding Story & Architecture**:
- [What I Have Been Working On: Modal - Erik Bernhardsson](https://erikbern.com/2022/12/07/what-ive-been-working-on-modal.html)
- [Modal: Our Investment in Erik and Akshat - Amplify Partners](https://www.amplifypartners.com/blog-posts/modal)
- [How Modal Built a Data Cloud from the Ground Up - Amplify Partners](https://www.amplifypartners.com/blog-posts/how-modal-built-a-data-cloud-from-the-ground-up)

**Company & Funding**:
- [Modal Company Overview](https://modal.com/company)
- [Announcing Our $87M Series B](https://modal.com/blog/announcing-our-series-b)
- [Modal Labs Raises $80M to Simplify Cloud AI Infrastructure - SiliconANGLE](https://siliconangle.com/2025/09/29/modal-labs-raises-80m-simplify-cloud-ai-infrastructure-programmable-building-blocks/)

**Documentation & Technical Resources**:
- [Modal Documentation](https://modal.com/docs)
- [GPU Acceleration Guide](https://modal.com/docs/guide/gpu)
- [modal.gpu API Reference](https://modal.com/docs/reference/modal.gpu)

**Job Postings & Compensation**:
- [Modal Careers - Amplify Partners](https://talent.amplifypartners.com/jobs/modal-labs)
- [Modal Careers - Redpoint Ventures](https://careers.redpoint.com/companies/modal-labs-2)
- [Salary Data - Levels.fyi](https://www.levels.fyi/companies/modal-labs/salaries/software-engineer)
- [Glassdoor Salaries](https://www.glassdoor.com/Salary/Modal-Salaries-E1953636.htm)

---

*Last updated: November 30, 2025*
