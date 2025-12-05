# Anysphere (Cursor) - Technology Stack

**Company:** Anysphere, Inc.
**Founded:** 2022
**Focus:** AI-powered code editor and development platform
**Headquarters:** San Francisco, California

---

## Non-AI Tech Stack

Anysphere operates from **San Francisco** with offices in **North Beach, San Francisco** and **Manhattan, New York**, maintaining an **in-person culture** with cozy offices replete with well-stocked libraries. Founded in **2022** by **MIT students Michael Truell, Sualeh Asif, Arvid Lunnemark, and Aman Sanger**, the company raised **$3.2+ billion total funding** across multiple rounds: **$8M seed** (October 2023, OpenAI Startup Fund lead, angels including former GitHub CEO Nat Friedman and Dropbox co-founder Arash Ferdowsi), **$60M+ Series A** (August 2024, Andreessen Horowitz and Thrive Capital, $400M valuation), **$105M Series B** (January 2025, Thrive Capital, $2.5B valuation), **$900M Series C** (June 2025, Thrive Capital lead, $9.9B valuation crossing $500M ARR), and **$2.3B Series D** (November 2025, Accel and Coatue lead with participation from Thrive, Andreessen Horowitz, NVIDIA, Google, and DST, $29.3B valuation crossing $1B ARR). The company achieved the **fastest SaaS growth from $1M to $100M ARR in 12 months** (January 2025), faster than Wiz (18 months), Deel (20 months), and Ramp (24 months), reaching this milestone with **no marketing** and a **team of fewer than 20 people** initially, growing to **300 total employees** as of November 2025 (some reports indicate ~150 employees as of August 2025). The platform is built as a **fork of Visual Studio Code**, maintaining **full compatibility with VS Code extensions and workspaces** while adding deep AI integration that extends beyond VS Code's extension API. Infrastructure supports **multi-channel deployment** across desktop (macOS, Windows, Linux) with **cloud-based AI agent execution** using **git worktrees and remote machines** for parallel agent isolation. The organization maintains a **very flat structure** with a **talent-dense team** including **International Mathematical Olympiad (IMO) gold medalists, former startup founders, and top open-source contributors**, and notably **prohibits AI tools during first-round coding interviews**, inviting finalists for **two-day on-site projects**. Cursor serves approximately **360,000 paying developers** at **$20-40/month** for an **average ACV of $276**, contrasting with enterprise-focused competitors charging $384K+ ACV. The company culture emphasizes **truthseeking, passionate and creative** individuals who enjoy **spirited debate, crazy ideas, and shipping code**.

**Salary Ranges**: Software Engineer salaries available on Levels.fyi (specific figures not publicly disclosed in recent data)

---

## AI/ML Tech Stack

### Composer - Proprietary MoE Coding Model at 250 Tokens/Sec, 4x Faster Than Frontier Systems

**What's unique**: Cursor released **Composer** in **November 2025** as part of **Cursor 2.0**, their **first proprietary in-house coding large language model (LLM)**, delivering **4x faster performance** than comparably intelligent frontier systems while maintaining **frontier-level coding intelligence**. Composer is a **mixture-of-experts (MoE) language model** specialized for **software engineering through reinforcement learning (RL)** in diverse development environments, generating at **250 tokens per second** — approximately **twice as fast as leading fast-inference models** and **four times faster than comparable frontier systems**. The model **completes most interactions in less than 30 seconds** while maintaining high reasoning ability across large and complex codebases, trained for **"agentic" workflows** where autonomous coding agents **plan, write, test, and review code collaboratively**. Composer has access to **simple tools** (reading and editing files) and **powerful ones** (terminal commands and codebase-wide semantic search), enabling end-to-end software development workflows. The **reinforcement learning training methodology** actively specializes the model for effective software engineering, **incentivizing efficient tool use** and **maximizing parallelism** whenever possible to reduce response latency. Cursor developed **Cursor Bench**, consisting of **real agent requests from engineers and researchers at Cursor** along with **hand-curated optimal solutions**, measuring not just correctness but also **adherence to codebase abstractions and software engineering practices**. Composer runs **exclusively in Cursor's cloud** with no open API outside Cursor 2.0, maintaining proprietary control over the model. This approach contrasts with competitors relying solely on third-party models (OpenAI, Anthropic); Cursor's vertical integration enables **optimization for interactive development speed**, critical for maintaining conversational flow during coding sessions.

### Multi-Agent Architecture - Up to 8 Parallel Agents Using Git Worktrees

**What makes it different**: **Cursor 2.0** introduced a **revolutionary multi-agent interface** (October 2025) allowing developers to **orchestrate up to eight AI agents working simultaneously** on different tasks, **reorganizing the IDE around agents rather than files**. The platform enables **parallel agent execution without file conflicts** using **git worktrees or remote machines**, where **each agent operates in its own isolated copy of the codebase**, preventing interference while enabling concurrent feature development. Users can **run agents in the foreground or background**, or **use parallel agents to create and review multiple plans simultaneously** — for instance, planning with one model and executing with another, or generating multiple implementation approaches for comparison. The **Background Agents feature** automates routine tasks (linting, testing, documentation creation) directly within the IDE, **reducing context switching by up to 40%** for standardized workflows. This architecture transforms development workflows from **sequential single-threaded coding** to **parallel multi-agent orchestration**, enabling developers to work on **frontend, backend, testing, and documentation concurrently** without manual context management. The technical implementation leverages **git worktrees** — lightweight filesystem mechanisms creating multiple working directories from a single repository — ensuring each agent sees a consistent codebase snapshot while changes remain isolated until merge. Cursor's multi-agent system contrasts with competitors' single-agent assistants (GitHub Copilot, Tabnine) that handle one task at a time; Cursor enables **simultaneous exploration of multiple implementation strategies**, accelerating development velocity through parallelism rather than just automation.

### Fastest SaaS to $100M ARR in 12 Months - 9,900% YoY Growth, 360K Paying Developers

**What sets Anysphere apart**: Cursor achieved **$100M ARR in January 2025**, just **12 months after launch**, making it the **fastest-growing SaaS company of all time** from $1M to $100M ARR — surpassing previous record-holder **Wiz** (18 months), **Deel** (20 months), and **Ramp** (24 months). The company reached this milestone with approximately **360,000 paying developers** at **$20-40/month**, achieving an **average contract value (ACV) of $276** — a dramatically different go-to-market strategy than enterprise SaaS competitors. This contrasts sharply with Wiz ($100M ARR with ~260 enterprise customers at $384K ACV), Deel (1,800 customers at $55K ACV), and Ramp (5,000 customers at $20K ACV), demonstrating that **developer-focused bottom-up adoption** can achieve comparable revenue velocity to enterprise top-down sales. Cursor's growth represents **9,900% year-over-year revenue growth**, achieved with **no marketing** and an **initial team of fewer than 20 people** — validating the power of **product-led growth** in developer tools. By **June 2025**, Cursor surpassed **$500M ARR**, and by **November 2025** crossed **$1 billion ARR**, demonstrating sustained hypergrowth. The company's **$20/month Pro plan** and **$40/month Business plan** pricing enables **individual developer adoption** without procurement friction, while the **freemium model** with generous free tier drives viral growth through word-of-mouth. Cursor's success demonstrates that **AI-native development tools** can achieve unprecedented adoption velocity when they deliver **measurable productivity gains** — users report **2-4x faster coding** through AI assistance, justifying subscription costs through time savings alone.

### Multi-LLM Orchestration - OpenAI, Anthropic, Gemini, xAI with Proprietary cursor-small Model

**What's unique**: Cursor implements a **multi-level AI model system** that **balances cost, performance, and latency** across different coding tasks, using a **proprietary cursor-small model for inline predictions**, **Claude 3.7 Sonnet for cross-file processing**, and **GPT-4 Turbo for background agent tasks**, enabling **real-time suggestions** while reserving powerful models for complex operations. Users can **freely choose between frontier models** from **OpenAI, Anthropic, Gemini, and xAI**, providing flexibility for different coding scenarios — for instance, using Claude for nuanced refactoring, GPT-4 for architectural planning, or Gemini for multimodal tasks involving screenshots or diagrams. The platform's **intelligent routing** ensures **inline autocomplete feels instant** (cursor-small generates predictions in milliseconds) while **cross-file edits leverage deeper reasoning** (Claude 3.7 Sonnet understands complex dependencies) and **background agents handle long-running tasks** (GPT-4 Turbo manages multi-step workflows like migrations or test generation). Cursor's **BERT-based codebase embedding algorithm** extracts the most important classes and methods while omitting implementation details, achieving **98% accuracy when answering queries on large codebases**, enabling agents to navigate projects with millions of lines of code efficiently. The **codebase embedding model gives agents deep understanding and recall**, functioning as a semantic search layer that surfaces relevant context for model queries. This multi-model architecture contrasts with single-model competitors (GitHub Copilot uses only OpenAI models); Cursor's **model-agnostic approach** prevents vendor lock-in and enables **continuous optimization** as new models release, automatically routing to the best-performing option for each task.

### VSCode Fork with Full Extension Compatibility - Seamless Migration, Deep AI Integration

**What makes it different**: Cursor is built as a **fork of Visual Studio Code**, maintaining **full compatibility with VS Code extensions, settings, and workspaces** while adding **deep AI integration** that extends beyond VS Code's extension API capabilities. Users can **import VS Code settings and extensions with one click**, making migration from VS Code to Cursor seamless — the platform uses the **same workspace format**, so existing project files work without modification. Cursor **regularly rebases onto the latest VS Code version** to stay current with features and fixes, though it often uses slightly older versions to ensure stability. The **forking decision** (rather than building a VS Code extension) enables Cursor to **modify core architecture** for system-level AI integration — VS Code's extension API was never designed for the kind of deep, context-aware AI assistance Cursor provides, requiring modifications to the editor's file handling, rendering pipeline, and process model. However, this architectural choice creates **potential fragmentation** — some VS Code extensions lag behind in Cursor's marketplace, and updates don't sync in real-time with the official VS Code marketplace. Cursor addresses this through **compatibility testing** and community support for popular extensions. The **VSCode foundation** provides Cursor with a **proven editor infrastructure** (syntax highlighting, debugging, Git integration, terminal, extension ecosystem) while the team focuses exclusively on **building the best AI-powered coding experience**. This strategy contrasts with competitors building editors from scratch (Replit, Zed) or operating purely as extensions (GitHub Copilot, Tabnine); Cursor's **fork approach** delivers **familiar UX** with **radical AI innovation**, minimizing adoption friction for the 15M+ developers already using VS Code.

### Context-Aware Coding - Rules System, Plan Mode, and Codebase Indexing at 98% Accuracy

**What sets Cursor apart**: Cursor implements a **Rules system** that **fixes memory issues** by allowing developers to **pin instructions to a project, user, or team**, stored as **Project rules** in `.cursor/rules/*.mdc`, **User rules** in Cursor Settings, **Team rules** in the team dashboard, or **Agent rules** in `AGENTS.md` at repo root. Rules enable **persistent context** across sessions — for instance, specifying coding standards (TypeScript strict mode, React functional components, TailwindCSS for styling), architectural patterns (service layer abstraction, repository pattern), or project-specific constraints (no external API calls in tests, use existing utility functions before creating new ones). The **Plan Mode** works in the background, where developers **create a plan with one model and execute it with another**, or **use parallel agents to review multiple plans simultaneously** before committing to implementation, enabling **deliberate architecture decisions** rather than immediate code generation. Cursor's **BERT-based codebase embedding** achieves **98% accuracy** answering queries on large codebases by extracting important classes and methods while omitting implementation details, enabling agents to **understand project structure** without overwhelming context windows. The platform provides **Slash Commands** for common workflows (`/edit`, `/chat`, `/fix`, `/test`), **Browser control** for debugging web applications, and **Hooks** for custom automation triggered by IDE events. The **Fusion Tab model** supports **multi-file processing and syntax highlighting** with **response times under 200ms**, enabling cross-file refactorings that maintain consistency across dozens of files simultaneously. This context management contrasts with competitors requiring **manual context selection** (Copilot's `#file` references) or lacking project-wide understanding (Tabnine's local-only models); Cursor's **automatic context retrieval** and **persistent rules** reduce cognitive load, allowing developers to focus on problem-solving rather than prompt engineering.

### Agent Development Innovation - Cursor Bench Evaluation, RL Training for Agentic Workflows

**What's unique**: Cursor pioneered **agent-first development workflows** where AI coding agents **plan, write, test, and review code collaboratively** with developers, moving beyond autocomplete to **autonomous multi-step software engineering**. The company developed **Cursor Bench**, an evaluation consisting of **real agent requests from Cursor engineers and researchers** with **hand-curated optimal solutions**, measuring not just **correctness** but also **adherence to existing abstractions and software engineering practices** — ensuring agents write code that **fits the project's patterns** rather than introducing novel approaches that conflict with existing conventions. Cursor's **reinforcement learning training** actively specializes models for **effective software engineering**, incentivizing **efficient tool use** (choosing grep over reading entire files, using semantic search before broad exploration) and **maximizing parallelism** (batching file edits, running tests concurrently) to reduce completion time. The **agent interface redesign** in Cursor 2.0 reorganizes the IDE around **agents as first-class entities** rather than files, with **dedicated agent panels** showing progress, **multi-agent orchestration views** displaying parallel work streams, and **plan review interfaces** for approving generated architectures before execution. Cursor's agents demonstrate **pragmatic intelligence** — understanding when to ask clarifying questions, when to propose multiple solutions for developer choice, and when to execute confidently without human intervention. The **agentic workflow training** ensures Composer makes **realistic decisions** about task decomposition (breaking large features into mergeable chunks), **error recovery** (recognizing test failures and iterating toward fixes), and **code review** (identifying edge cases and suggesting improvements). This agent-centric approach contrasts with competitors' **reactive autocomplete** (GitHub Copilot suggests next lines based on cursor position); Cursor enables **proactive feature implementation** where agents drive development forward with human oversight rather than requiring line-by-line guidance.

---

## Sources

**Anysphere/Cursor Official**:
- [Cursor Homepage](https://cursor.com/)
- [Cursor Features](https://cursor.com/features)
- [Cursor 2.0 Announcement](https://cursor.com/blog/2-0)
- [Composer: Building a Fast Frontier Model with RL](https://cursor.com/blog/composer)
- [Series D Announcement](https://cursor.com/blog/series-d)
- [Cursor Docs - VS Code Migration](https://cursor.com/docs/configuration/migrations/vscode)
- [Cursor Docs - Extensions](https://cursor.com/docs/configuration/extensions)
- [Cursor Docs - Parallel Agents](https://cursor.com/docs/configuration/worktrees)
- [Cursor Changelog - 2.0](https://cursor.com/changelog/2-0)

**Company & Funding**:
- [AI Startup Cursor Raises $2.3 Billion at $29.3 Billion Valuation - CNBC](https://www.cnbc.com/2025/11/13/cursor-ai-startup-funding-round-valuation.html)
- [Anysphere's Cursor Soars to $29B Valuation - TechFundingNews](https://techfundingnews.com/anysphere-soars-to-29-3b-valuation-with-2-3b-funding-redefining-the-future-of-coding/)
- [Cursor's Anysphere Nabs $9.9B Valuation, Soars Past $500M ARR - TechCrunch](https://techcrunch.com/2025/06/05/cursors-anysphere-nabs-9-9b-valuation-soars-past-500m-arr/)
- [Anysphere Raises $900M for AI-Powered Cursor Code Editor - SiliconANGLE](https://siliconangle.com/2025/06/05/anysphere-raises-900m-ai-powered-cursor-code-editor/)
- [AI-Powered Coding Tool Anysphere Raises $900M at $9.9B Valuation - Crunchbase News](https://news.crunchbase.com/ai/anysphere-cursor-venture-funding-thrive/)
- [Cursor in Talks to Raise at $10B Valuation - TechCrunch](https://techcrunch.com/2025/03/07/cursor-in-talks-to-raise-at-a-10b-valuation-as-ai-coding-sector-booms/)
- [Anysphere Raises $8M from OpenAI - TechCrunch](https://techcrunch.com/2023/10/11/anysphere-raises-8m-from-openai-to-build-an-ai-powered-ide/)
- [Exclusive: Anysphere Raised $60M+ Series A at $400M Valuation - TechCrunch](https://techcrunch.com/2024/08/09/anysphere-a-github-copilot-rival-has-raised-60m-series-a-at-400m-valuation-from-a16z-thrive-sources-say/)
- [The $29 Billion Editor: How Cursor Eclipsed Giants - WebProNews](https://www.webpronews.com/the-29-billion-editor-how-cursor-eclipsed-giants-to-rewrite-the-rules-of-software-engineering/)
- [Anysphere Wikipedia](https://en.wikipedia.org/wiki/Anysphere)
- [Anysphere Crunchbase](https://www.crunchbase.com/organization/anysphere)
- [Anysphere PitchBook Profile](https://pitchbook.com/profiles/company/519419-89)
- [Anysphere CB Insights](https://www.cbinsights.com/company/anysphere)

**Growth & Metrics**:
- [Cursor Went from 1-100M ARR in 12 Months: The Fastest SaaS - Medium](https://medium.com/strategy-decoded/cursor-went-from-1-100m-arr-in-12-months-the-fastest-saas-to-achieve-this-19d811c4f0bb)
- [Cursor at $100M ARR - Sacra](https://sacra.com/research/cursor-at-100m-arr/)
- [How Did Cursor Grow So Fast - $1M to $100M ARR in 24 Months - Product Market Fit](https://www.productmarketfit.tech/p/how-did-cursor-grow-so-fast-1m-to)
- [Cursor: $1M to $100M ARR in 12 Months - Synergy Startup](https://synergystartup.substack.com/p/cursor-1m-to-100m-arr-in-12-months)
- [Anysphere's AI-Code Editor Cursor Fastest to Reach $100M ARR - AIM](https://aimmediahouse.com/market-industry/anyspheres-ai-code-editor-cursor-fastest-to-reach-100m-arr-in-12-months)
- [Cursor Grew to $100M ARR in 12 Months - NextBigFuture](https://www.nextbigfuture.com/2025/02/cursor-grew-to-100m-in-annual-recurring-revenue-in-12-months.html)
- [How Cursor AI Hit $100M ARR: The Freemium-Fueled Rocket Ship - We Are Founders](https://www.wearefounders.uk/how-cursor-ai-hit-100m-arr-in-12-months-the-freemium-fueled-rocket-ship-taking-on-github-copilot/)
- [How Cursor Grew from $1M to $100M ARR in a Year - Frictionless Post](https://www.frictionlesspost.com/p/how-cursor-grew-from-1m-to-100m-arr)
- [How Cursor Grows - Product Growth by Aakash Gupta](https://www.news.aakashg.com/p/how-cursor-grows)
- [40+ Cursor Statistics (2025): Usage, Growth & Revenue - Shipper](https://shipper.now/cursor-stats/)

**Technology & Features**:
- [Cursor 2.0 Revolutionizes AI Coding with Multi-Agent Architecture - Artezio](https://www.artezio.com/pressroom/blog/revolutionizes-architecture-proprietary/)
- [Vibe Coding Platform Cursor Releases First In-House LLM Composer - VentureBeat](https://venturebeat.com/ai/vibe-coding-platform-cursor-releases-first-in-house-llm-composer-promising)
- [Cursor 2.0 Debuts with Proprietary Composer Model - CXO DigitalPulse](https://www.cxodigitalpulse.com/cursor-2-0-debuts-with-proprietary-composer-model-and-multi-agent-coding-capabilities/)
- [Cursor 2.0: New AI Model Explained - Codecademy](https://www.codecademy.com/article/cursor-2-0-new-ai-model-explained)
- [Composer: What Cursor's New Coding Model Means for LLMs - PromptLayer](https://blog.promptlayer.com/composer-what-cursors-new-coding-model-means-for-llms/)
- [Cursor 2.0 Launches with 4x Faster AI-Assisted Coding - The Outpost](https://theoutpost.ai/news-story/cursor-2-0-launches-with-composer-first-in-house-ai-coding-model-delivers-4x-speed-boost-21310/)
- [Composer: A Fast New AI Coding Model by Cursor - Medium](https://medium.com/@leucopsis/composer-a-fast-new-ai-coding-model-by-cursor-e1a023614c07)
- [Cursor 2.0 Lets Developers Run 8 AI Agents in Parallel - AIM](https://analyticsindiamag.com/ai-news-updates/cursor-2-0-lets-developers-run-8-ai-agents-in-parallel-adds-its-own-coding-model/)
- [Cursor 2.0 Multi-Agent Suite Explained - Skywork AI](https://skywork.ai/blog/vibecoding/cursor-2-0-multi-agent-suite/)
- [Background Agents in Cursor: Cloud-Powered Coding at Scale - Decoupled Logic](https://decoupledlogic.com/2025/05/29/background-agents-in-cursor-cloud-powered-coding-at-scale/)
- [Cursor 2.0 Introduces Parallel Agents and New Model - Techzine Global](https://www.techzine.eu/news/devops/135916/cursor-2-0-introduces-parallel-agents-and-new-model/)
- [Cursor 2.0: New Multi-Agent Interface Explained - Lilys AI](https://lilys.ai/notes/en/cursor-20-20251106/cursor-new-multi-agent-interface)
- [Cursor 2.0 and Composer: Multi-Agent Rethink - CometAPI](https://www.cometapi.com/cursor-2-0-what-changed-and-why-it-matters/)
- [Cursor AI Review (2025): Features, Workflow - Prismic](https://prismic.io/blog/cursor-ai)
- [Cursor AI Review: Revolutionary AI-Powered Code Editor - CrewStack](https://crewstack.net/tools/2025-11-11-cursor-ai-review-revolutionary-ai-powered-code-editor-for-2025/)
- [Cursor AI Update 2025: New Tab Model and Background Agent - AI Rockstars](https://ai-rockstars.com/cursor-ai-update-2025-new-tab-model-and-background-agent-change-development-work/)
- [Cursor AI in 2025: The Future of Coding - Medium](https://medium.com/@vikranthsalian/cursor-ai-in-2025-the-future-of-coding-with-ai-powered-assistance-ace7c411c8a1)

**Migration & Compatibility**:
- [Migrating from VS Code to Cursor - Complete Guide - Cursor History](https://cursorhistory.com/blog/vscode-migration)
- [Forked by Cursor: The Hidden Cost of VS Code Fragmentation - DEV Community](https://dev.to/pullflow/forked-by-cursor-the-hidden-cost-of-vs-code-fragmentation-4p1)
- [The VS Code Fork Dilemma: Innovation at the Cost of Fragmentation - Pullflow](https://pullflow.com/blog/cursor-vs-code-fragmentation/)
- [Extensions under Cursor IDE Not Up to Date - GitHub Issue](https://github.com/cursor/cursor/issues/1602)
- [GitHub - CodeCursor Extension](https://github.com/Helixform/CodeCursor)

**Company Analysis**:
- [Report: Anysphere Business Breakdown & Founding Story - Contrary Research](https://research.contrary.com/company/anysphere)
- [Founder Story: Michael Truell of Cursor AI - Frederick AI](https://www.frederick.ai/blog/michael-truell-cursor-ai)
- [United States AI Coding App Cursor Creator - Caproasia (Series D)](https://www.caproasia.com/2025/11/15/united-states-ai-coding-app-cursor-creator-anysphere-raised-2-3-billion-in-series-d-funding-at-29-3-billion-valuation-founded-in-2022-by-mit-graduates-sualeh-asif-arvid-lunnemark-aman-sanger-mi/)
- [United States AI Coding App Cursor Creator - Caproasia (Series C)](https://www.caproasia.com/2025/05/09/united-states-ai-coding-app-cursor-creator-anysphere-raised-900-million-at-9-billion-valuation-founded-in-2022-by-mit-graduates-sualeh-asif-arvid-lunnemark-aman-sanger-michael-truell-investors/)
- [Anysphere Raises $60M For AI-Powered Coding Tool - Maginative](https://www.maginative.com/article/anysphere-raises-60m-for-ai-powered-coding-tool-cursor/)

**Compensation**:
- [Cursor Software Engineer Salary - Levels.fyi](https://www.levels.fyi/companies/cursor/salaries/software-engineer)
- [Cursor Salaries - Levels.fyi](https://www.levels.fyi/companies/cursor/salaries)
- [Anysphere Engineer Job Listing - Glassdoor](https://www.glassdoor.com/job-listing/engineer-anysphere-JV_IC1147401_KO0,8_KE9,18.htm?jl=1009767668668)
- [Working at Cursor - Glassdoor](https://www.glassdoor.com/Overview/Working-at-Cursor-EI_IE1889800.11,17.htm)

---

*Last updated: December 5, 2025*
