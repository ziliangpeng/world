# Llama Ecosystem: Safety, Security, and Tools

Meta's Llama ecosystem extends beyond language models to include safety, security, and moderation tools.

---

## Llama Guard (Safety & Moderation)

**Purpose**: LLM-based input-output safeguard for human-AI conversations, providing content moderation and safety classification.

### Model Evolution

**Llama Guard 1** (December 2023):
- **7B parameters** (based on Llama 2-7B)
- Initial safety classification model
- First LLM-based content moderation from Meta

**Llama Guard 2**:
- Expanded to 11 safety categories
- Improved taxonomy
- Better multilingual support

**Llama Guard 3** (July 2024):
- **8B parameters** (based on Llama 3.1-8B)
- Supports **8 languages**
- MLCommons-aligned taxonomy (13 categories)
- INT8 quantized version available
- Optimized for search and code interpreter tool calls

**Llama Guard 3-1B**:
- **1B parameters** for resource-constrained environments
- **1B-INT4**: 440MB compressed (7x smaller than uncompressed)
- F1 score: 0.904 (outperforms uncompressed 1B)
- Perfect for edge deployment

**Llama Guard 4** (April 2025):
- **12B parameters** (based on Llama 4)
- **Multimodal input/output moderation** (text + images + video)
- Supports **multiple images** in prompts
- **14 hazard categories** (MLCommons taxonomy) + code interpreter abuse
- Most capable Llama Guard yet

### Safety Taxonomy

**Llama Guard 3/4** (13-14 Categories - MLCommons aligned):
1. **Violent Crimes** - Violence, harm, killing
2. **Non-Violent Crimes** - Theft, fraud, illegal activities
3. **Sex-Related Crimes** - Sexual assault, trafficking
4. **Child Sexual Exploitation** - CSAM, grooming
5. **Defamation** - Slander, libel, false accusations
6. **Specialized Advice** - Medical, legal, financial advice without qualification
7. **Privacy Violations** - Doxxing, personal information exposure
8. **Intellectual Property** - Copyright infringement, piracy
9. **Indiscriminate Weapons** - Bioweapons, chemical weapons, explosives
10. **Hate** - Hate speech, discrimination
11. **Suicide & Self-Harm** - Encouraging self-harm
12. **Sexual Content** - Explicit sexual content
13. **Elections** - Election manipulation, misinformation
14. **Code Interpreter Abuse** (Guard 4) - Malicious code execution

### How It Works

**Input Classification**:
```
User Prompt → Llama Guard → Safe/Unsafe + Categories
```

**Output Classification**:
```
LLM Response → Llama Guard → Safe/Unsafe + Categories
```

**Result Format**:
- **Safe**: No violations detected
- **Unsafe**: Lists violated categories (e.g., "Unsafe: 1, 10" for Violent Crimes + Hate)

### Capabilities

- Classifies prompts and responses as safe/unsafe
- Lists violated categories when unsafe
- Supports both input (prompt) and output (response) classification
- **Flexible, customizable taxonomy** - Can define custom categories
- Multilingual support (8 languages in Guard 3)
- **Multimodal** (Guard 4) - Text, images, video
- Optimized for search and code interpreter tool calls
- Extremely fast inference (important for production)

### Use Cases

**Content Moderation**:
- Filter user inputs before processing
- Check LLM outputs before showing to users
- Real-time chat moderation

**Application Safety**:
- Prevent jailbreaks
- Block unsafe requests
- Ensure compliance with policies

**Tool Call Safety**:
- Check code interpreter commands
- Validate search queries
- Protect against injection attacks

### Links

- **Paper (Guard 1)**: [Llama Guard: LLM-based Input-Output Safeguard](https://arxiv.org/abs/2312.06674)
- **Paper (Guard 3-1B)**: [Llama Guard 3-1B-INT4: Compact and Efficient Safeguard](https://arxiv.org/abs/2411.17713)
- **Hugging Face**:
  - [LlamaGuard-7b](https://huggingface.co/meta-llama/LlamaGuard-7b)
  - [Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B)
  - [Llama-Guard-3-1B](https://huggingface.co/meta-llama/Llama-Guard-3-1B)
  - [Llama-Guard-4-12B](https://huggingface.co/meta-llama/Llama-Guard-4-12B)
- **Documentation**: [Llama Guard 4 Model Card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/)

---

## Purple Llama (Trust & Safety Tools)

**What It Is**: Purple Llama is an umbrella project for open trust and safety tools and evaluations for responsible generative AI development.

**NOT a Model** - It's a collection of tools and benchmarks.

**Name Origin**: "Purple" from cybersecurity (Red team + Blue team = Purple team) + "Llama"

### Main Components

### 1. CyberSecEval - Cybersecurity Benchmarks

**Purpose**: Comprehensive evaluation of LLM cybersecurity risks

**Evaluation Domains**:
- Propensity to generate insecure code
- Compliance when asked to assist in cyberattacks
- Code vulnerability detection
- Secure coding practices

**Evolution**:
- **CyberSecEval 1** (December 2023) - Initial benchmark
- **CyberSecEval 2, 3** (2024) - Expanded coverage
- **CyberSecEval 4** (Latest) - Adds CyberSOCEval + AutoPatchBench

**Tools**:
- **Insecure Code Detector (ICD)**: 189 static analysis rules, 50 insecure practices
- **MITRE Tests**: MITRE ATT&CK framework compliance evaluation
- **CyberSOCEval**: Security operations center evaluation
- **AutoPatchBench**: Automatic vulnerability patching benchmark

### 2. Llama Guard

See dedicated Llama Guard section above.

### 3. Prompt Guard

**Purpose**: Protection against malicious prompts and injection attacks

**Capabilities**:
- Detect prompt injection attempts
- Identify jailbreak attempts
- Protect LLM applications from manipulation
- Application security layer

**Prompt Guard 2** (Latest):
- **86M parameters** - High performance version
- **22M parameters** - Low latency version
- Reduced false positives
- More robust detection
- Faster inference

**Use Cases**:
- Protect against indirect prompt injection
- Detect malicious instructions hidden in documents
- Validate user inputs before processing
- Secure RAG systems (Retrieval-Augmented Generation)

### 4. Code Shield

**Purpose**: Inference-time filtering of insecure code produced by LLMs

**Capabilities**:
- Detect insecure code patterns
- Prevent code interpreter abuse
- Secure command execution
- Filter dangerous code suggestions

**Technical Details**:
- Supports **7 programming languages**
- Average latency: **200ms** (fast enough for real-time)
- Static analysis-based detection
- Integration with LLM code generation

**What It Catches**:
- SQL injection vulnerabilities
- Command injection
- Path traversal
- Insecure deserialization
- And 50+ other insecure practices

**Use Cases**:
- Filter code from Code Llama
- Secure copilot applications
- Code review automation
- Prevent malicious code execution in sandboxes

### 5. LlamaFirewall

**Purpose**: Integration suite for Llama Protections

**What It Does**:
- Combines Llama Guard, Prompt Guard, and Code Shield
- Unified protection layer
- Easy integration into applications
- Centralized security management

**Architecture**:
```
User Input → Prompt Guard → LLM → Llama Guard → Code Shield → Output
```

**Benefits**:
- Defense in depth
- Layered security
- Comprehensive protection
- Easy deployment

### Licensing and Collaboration

**License**: Permissively licensed for research and commercial use

**Partners**:
- AI Alliance
- AMD, AWS, Google Cloud
- Hugging Face, IBM, Intel, Microsoft, NVIDIA
- MLCommons
- And many others

**Repository**: https://github.com/meta-llama/PurpleLlama

### Relationship to Llama Models

Purple Llama provides tools to assess and improve security of Llama and other LLMs:
- **Evaluate**: CyberSecEval benchmarks
- **Protect**: Llama Guard content moderation
- **Secure**: Prompt Guard and Code Shield
- **Integrate**: LlamaFirewall unified layer

**Works with**:
- All Llama models
- Other open-source LLMs
- Proprietary models (evaluation)

### Links

- **Repository**: [PurpleLlama GitHub](https://github.com/meta-llama/PurpleLlama)
- **Blog**: [Purple Llama announcement](https://ai.meta.com/blog/purple-llama-open-trust-safety-generative-ai/)
- **Documentation**: [Llama Protections](https://www.llama.com/llama-protections/)

---

## Llama Protections Ecosystem

The complete safety and security stack:

| Component | Type | Purpose |
|-----------|------|---------|
| **Llama Guard** | LLM (1B-12B) | Content moderation |
| **Prompt Guard** | Classifier (22M-86M) | Prompt injection detection |
| **Code Shield** | Static analyzer | Insecure code detection |
| **CyberSecEval** | Benchmark | Security evaluation |
| **LlamaFirewall** | Integration | Unified protection layer |

## Use Case Example: Secure AI Application

```
1. User sends prompt
   ↓
2. Prompt Guard checks for injection
   ↓
3. Llama Guard checks input safety
   ↓
4. LLM processes (e.g., Code Llama)
   ↓
5. Code Shield filters insecure code
   ↓
6. Llama Guard checks output safety
   ↓
7. Return safe, secure response
```

## Impact

Meta's Llama Protections demonstrate:
1. **Safety is a priority** in open AI
2. **Open tools** for responsible AI development
3. **Layered defense** approach
4. **Community collaboration** (MLCommons, industry partners)
5. **Production-ready** security

These tools enable developers to build safe, responsible AI applications with Llama models.
