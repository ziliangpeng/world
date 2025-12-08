# Figma - Technology Stack

**Company:** Figma, Inc.
**Founded:** 2012
**Focus:** Collaborative design platform (UI/UX design, prototyping, whiteboarding)
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

Figma was founded in **2012** by **Dylan Field** (CEO) and **Evan Wallace** (former CTO, left 2021) while they were computer science students at **Brown University**. Field was inspired by Wallace's experiments with **WebGL** enabling high-quality graphics rendering in browsers. The founders spent **four years in development** before publicly launching in **2016**. The company has raised **$749M total funding** across 7 rounds, with the latest being a **$416M Series F** (May 2024) from Coatue, Sequoia, and a16z. In **September 2022**, Adobe announced a **$20B acquisition** of Figma — the largest software acquisition ever attempted at the time. However, in **December 2023**, the deal was terminated due to regulatory pushback from the **European Commission and UK CMA**, with Adobe paying Figma a **$1B reverse termination fee**. Figma **filed its S-1** on July 1, 2025 for NYSE listing under ticker 'FIG', with a **$14.6-16.4B target valuation**. As of October 2025, market cap reached **$24.4B**. The company achieved **$749M revenue in 2024** and **$228M Q1 2025 revenue** (+46% YoY) with **$44.9M net income**. Team size has grown to approximately **2,700+ employees** with offices globally. Over **80% of weekly active users are outside the US**, prompting infrastructure expansion beyond their primary Oregon datacenter. The backend stack includes **Ruby (Sinatra)**, **Node.js**, **Go**, **TypeScript**, **React**, **PostgreSQL**, and **AWS** (including EKS for Kubernetes). The company migrated to **Kubernetes** for a majority of core services in less than 12 months.

**Salary Ranges**: Software Engineer $215K-$780K total (L1-L5) | Median $483K | 90th percentile $674K | NYC area $225K-$504K

---

## AI/ML Tech Stack

### C++/WebAssembly Rendering Engine - Browser-Based "Gaming Engine" Architecture

**What's unique**: Figma's renderer is written in **C++ and compiled to WebAssembly (Wasm)** using **Emscripten**, achieving performance that often exceeds native desktop applications. The architecture is described as a **"browser inside a browser"** with its own DOM, compositor, and text layout engine. The renderer is a **highly-optimized tile-based engine** supporting masking, blurring, dithered gradients, blend modes, nested layer opacity — all GPU-rendered and fully anti-aliased. Switching to WebAssembly delivered **3x faster load times** regardless of document size. The same C++ codebase is **cross-compiled** to native x64/arm64 for server-side rendering, testing, and debugging. Figma has migrated from **WebGL to WebGPU** using Emscripten's built-in WebGPU bindings for improved performance. This approach — betting on browser capabilities when native apps dominated — was considered extremely ambitious in 2013 but enabled Figma's key differentiator: instant access without installation and seamless real-time collaboration.

### Multiplayer - Hybrid CRDT/OT with Last-Writer-Wins Conflict Resolution

**What makes it different**: Figma's **multiplayer service** synchronizes file state across clients via **WebSockets** with a pragmatic hybrid approach combining elements of **CRDTs (Conflict-free Replicated Data Types)** and **Operational Transformations (OT)**. Instead of implementing full OT (complex and designed for text editing), Figma uses a simpler model: changes are broadcast in real-time, and simultaneous edits to the **same property on the same object** use **last-writer-wins** resolution. The multiplayer service is **authoritative** — handling validation, ordering, and conflict resolution while holding file state **in-memory** for speed. This design deliberately trades off certain functionality (multi-user editable text strings are harder to support) for implementation simplicity and reliability. Figma doesn't use managed services for multiplayer because it's **core to the experience** and requires tight control, while using commodity services for other components. This architecture enables the signature Figma experience where multiple designers see each other's cursors and changes instantly.

### LiveGraph - Real-Time Data Fetching with PostgreSQL WAL Tailing

**What sets Figma apart**: **LiveGraph** is Figma's schema-based real-time database service providing GraphQL-like subscriptions that return JSON trees. The system tails the **PostgreSQL Write-Ahead Log (WAL)** replication stream to deliver live updates — when database rows change, connected clients receive updates automatically without engineers manually coding event streams. The architecture evolved significantly: the original design used one server with an in-memory query cache tailing a single Postgres instance. **LiveGraph 100x** (the redesigned version) is written in **Go** with three services: an **edge** handling client requests, expanding them into queries, and reconstructing results. Legacy TypeScript code remains but Go handles performance-intensive operations due to its multithreading capabilities. LiveGraph generates **TypeScript bindings** for the GraphQL API, enabling type-safe client integration. This infrastructure powers real-time collaboration across Figma's product suite beyond just the design canvas.

### Database Architecture - Vertical Partitioning on "One of the Beefiest Machines in AWS"

**What's unique**: Figma's infrastructure follows the **KISS (Keep It Simple, Stupid)** principle, originally running on a **single Amazon RDS database instance** — described as "one of the beefiest machines in AWS." By 2020, with database traffic growing **~3x annually** and hitting **65%+ CPU utilization** during peaks, they needed to scale. Rather than rewriting the system, Figma chose **vertical partitioning** — splitting tables by domain into separate databases while keeping each partition on powerful single instances. This allowed them to scale to **4M+ users without a major rewrite**. The team has a dedicated **"Creation Engine" pillar** — an infrastructure team focused on the UI engine rather than traditional web-scale concerns. A separate **"Native" pillar** handles iOS, Android, and Electron expertise, with engineers reporting into the pillar while working across product teams.

### Figma AI - Claude-Powered Design Generation with Model Context Protocol

**What makes it different**: **Figma Make** is Figma's generative AI tool that transforms static designs into interactive prototypes through **natural language prompts**, powered by **Anthropic's Claude 3.7** model. The integration uses Figma's **Model Context Protocol (MCP)** — a direct data pipeline feeding structured design information (component properties, variants, design tokens) to AI coding assistants like **GitHub Copilot or Cursor**. Figma has incorporated **OpenAI's gpt-image-1** model for generating and editing images via text prompts directly within the platform. **AI-enhanced semantic search** understands meaning and context behind queries, returning relevant components even when search terms don't exactly match names. **Visual Search** lets users find similar design elements. According to Figma's 2025 AI Report, **1 in 3 respondents are launching AI-powered products** (up 50% from prior year), and **80%+ of designers and developers** say learning to work with AI is essential to future success.

### Dev Mode - AI-Powered Design-to-Code with Code Connect

**What sets Figma apart**: **Dev Mode** evolved in 2025 from a simple inspection panel to an **AI-powered co-pilot for developers**. **Code Connect** links codebases with Figma design systems — when inspecting components with connected code, developers see **actual design system code from their libraries** instead of auto-generated snippets, driving design system adoption. The MCP integration enables AI assistants to access structured design data directly. Features include **annotations** for designers to markup specs and measurements, **statuses** to track when sections are ready for development with real-time notifications, and automatic updates when designs change so developers never build from outdated specs. While Figma generates code, it still requires manual review — it's a **handoff and collaboration tool** rather than production-ready code generator. This bridges the designer-developer gap that historically caused friction and rework.

### FigJam & Figma Slides - Product Suite Expansion Beyond Design

**What's unique**: Figma expanded beyond UI design with **FigJam** (collaborative whiteboard) and **Figma Slides** (presentations). FigJam provides online whiteboarding with audio chat, live chat, comments, and real-time collaboration — **included free with all seats on every plan**. Figma Slides combines design precision with FigJam's collaborative spirit, supporting **polls, voting, and interactive prototypes** for audience engagement. **Figma AI can generate slide decks from FigJam boards**, enabling workflow from brainstorm sticky notes to polished presentations. This suite expansion positions Figma to compete with Miro (whiteboarding), Google Slides/PowerPoint (presentations), while maintaining the real-time collaboration DNA across all products.

---

## Sources

**Figma Official**:

- [Figma Homepage](https://www.figma.com/)
- [Figma Infrastructure Blog](https://www.figma.com/blog/under-the-hood-of-figmas-infrastructure/)
- [How Figma's Multiplayer Technology Works](https://www.figma.com/blog/how-figmas-multiplayer-technology-works/)
- [Making Multiplayer More Reliable](https://www.figma.com/blog/making-multiplayer-more-reliable/)
- [LiveGraph: Real-Time Data Fetching](https://www.figma.com/blog/livegraph-real-time-data-fetching-at-figma/)
- [LiveGraph 100x: Real-Time Data at Scale](https://www.figma.com/blog/livegraph-real-time-data-at-scale/)
- [WebAssembly Cut Load Time by 3x](https://www.figma.com/blog/webassembly-cut-figmas-load-time-by-3x/)
- [Building a Professional Design Tool on the Web](https://www.figma.com/blog/building-a-professional-design-tool-on-the-web/)
- [Figma Rendering: Powered by WebGPU](https://www.figma.com/blog/figma-rendering-powered-by-webgpu/)
- [Keeping Figma Fast](https://www.figma.com/blog/keeping-figma-fast/)
- [Introducing Figma AI](https://www.figma.com/blog/introducing-figma-ai/)
- [Figma's 2025 AI Report](https://www.figma.com/reports/ai-2025/)
- [Dev Mode](https://www.figma.com/dev-mode/)
- [Guide to Dev Mode](https://help.figma.com/hc/en-us/articles/15023124644247-Guide-to-Dev-Mode)
- [Figma Make](https://www.figma.com/make/)
- [FigJam](https://www.figma.com/figjam/)
- [Figma Slides](https://www.figma.com/slides/)
- [Adobe Merger Abandonment](https://www.figma.com/blog/figma-adobe-abandon-proposed-merger/)
- [Engineering Blog](https://www.figma.com/blog/engineering/)
- [Infrastructure Blog Tag](https://www.figma.com/blog/infrastructure/)

**Company & Funding**:

- [Figma Wikipedia](https://en.wikipedia.org/wiki/Figma)
- [Dylan Field Wikipedia](https://en.wikipedia.org/wiki/Dylan_Field)
- [Figma Crunchbase](https://www.crunchbase.com/organization/figma)
- [Figma PitchBook](https://pitchbook.com/profiles/company/57686-86)
- [Figma Tracxn](https://tracxn.com/d/companies/figma/__ax4OtPOcmyjsFUT81_oGsjztypgqH8K3cQ0_zcUnNBo)
- [Figma Revenue & Metrics - GetLatka](https://getlatka.com/companies/figma)
- [Figma Revenue - Sacra](https://sacra.com/c/figma/)
- [Figma CNBC Disruptor 50 2025](https://www.cnbc.com/2025/06/10/figma-2025-cnbc-disruptor-50.html)
- [Figma S-1 Filing - ESO Fund](https://www.esofund.com/blog/figma-ipo)

**Acquisition & Regulatory**:

- [Adobe/Figma Deal Collapse - Axios](https://www.axios.com/2023/12/18/adobe-figma-deal-collapse)
- [Adobe/Figma Deal Called Off - CNBC](https://www.cnbc.com/2023/12/18/adobe-and-figma-call-off-20-billion-merger.html)
- [Adobe $20B Acquisition on UK Radar - TechCrunch](https://techcrunch.com/2023/05/03/adobes-20b-figma-acquisition-fall-on-uk-competition-radar/)
- [Adobe/Figma Antitrust Analysis - ABA](https://www.americanbar.org/groups/antitrust_law/resources/newsletters/adobe-figma-merger/)
- [Adobe Failed Acquisition Cost - Yahoo Finance](https://finance.yahoo.com/news/adobe-failed-acquisition-figma-cost-151035766.html)

**Technical Deep Dives**:

- [Inside Figma's Engineering Culture - Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/inside-figmas-engineering-culture)
- [Inside Figma's Multiplayer Infrastructure - Runtime News](https://www.runtime.news/inside-figmas-multiplayer-infrastructure/)
- [Figma's 100x Approach to Scaling - BetterStack](https://newsletter.betterstack.com/p/figmas-100x-approach-to-scaling-its)
- [Architecture Decision for 4M Users - Medium](https://medium.com/@thekareneme/the-architecture-decision-that-let-figma-scale-to-4m-users-without-rewriting-97d07c25eb9e)
- [Notes From Figma Engineering - Andrew Chan](https://andrewkchan.dev/posts/figma2.html)
- [Evan Wallace - Made by Evan](https://madebyevan.com/figma/)
- [Dylan Field Spotlight - Sequoia](https://sequoiacap.com/article/dylan-field-figma-spotlight/)

**Compensation**:

- [Figma Salaries - Levels.fyi](https://www.levels.fyi/companies/figma/salaries)
- [Figma Software Engineer Salary - Levels.fyi](https://www.levels.fyi/companies/figma/salaries/software-engineer)
- [Figma Salaries - Glassdoor](https://www.glassdoor.com/Salary/Figma-Salaries-E1537286.htm)
- [Figma Salaries - Blind](https://www.teamblind.com/company/Figma/salaries/united-states)

---

*Last updated: December 6, 2025*
