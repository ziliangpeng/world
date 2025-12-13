# Bittensor Subnet 1 (Apex): Complete Overview

## Quick Reference

**Name:** Apex (formerly Text Prompting)
**Subnet ID:** SN1
**Operator:** Macrocosmos / Opentensor Foundation
**GitHub:** https://github.com/macrocosm-os/apex
**Launch Date:** January 22, 2024 (rebranded from Text Prompting)
**Current Version:** Apex 3.0 (August 2025)
**Status:** Most mature subnet, flagship evaluation platform

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Evolution Timeline](#evolution-timeline)
3. [Current Purpose (Apex 3.0)](#current-purpose-apex-30)
4. [API Access & Pricing](#api-access--pricing)
5. [How Evaluation Works](#how-evaluation-works)
6. [Factual Grounding Mechanism](#factual-grounding-mechanism)
7. [Integration with Other Subnets](#integration-with-other-subnets)
8. [Who Uses Apex](#who-uses-apex)
9. [Technical Architecture](#technical-architecture)
10. [Comparison to Alternatives](#comparison-to-alternatives)
11. [Resources](#resources)

---

## Executive Summary

Subnet 1 (Apex) is Bittensor's flagship subnet that has **evolved from an LLM inference platform into an AI quality evaluation and training data generation system**. As of Apex 3.0 (August 2025), it no longer provides direct inference to end users. Instead, it focuses on:

- **Game-theoretic quality evaluation** using adversarial mechanisms
- **Training data generation** (millions of tokens per day)
- **Quality judgment validation** for AI model alignment
- **Feeding data to other subnets** (particularly SN37 for finetuning)

**Key Insight:** Apex shifted from "Can we decentralize ChatGPT?" to "Can we decentralize the RLHF/alignment process that makes ChatGPT good?"

---

## Evolution Timeline

### **Pre-2021: Conceptual Origins**
- Bittensor founded on decentralizing AI training/inference

### **2021-2023: Single-Task Era**
- **November 2021:** Bittensor launches with "Satoshi" version
- **Function:** Simple text prompting inference
- Entire network focused on serving LLM inference

### **March 2023: Blockchain Migration**
- Migrated from Polkadot parachain to independent Layer 1

### **August 2023: DPO Integration**
- Implemented Direct Preference Optimization
- **4x performance boost**

### **October 2023: Revolution Upgrade**
- Subnets architecture officially launched
- SN1 became first production subnet

### **Phase 1: Reference Answer Approach (2023 - Early 2024)**

**Evaluation Method:**
```
Validator → Creates reference answer with GPT-4/Claude
          ↓
Miners → Generate responses
          ↓
Scoring → (60% RLHF + 40% DPO) × Diversity × Relevance × NSFW_filter
```

**Models Used:**
- RLHF: `OpenAssistant/reward-model-deberta-v3-large-v2`
- DPO scoring models
- External APIs for factual grounding

**Limitations:**
- Validators could overfit to specific styles
- Miners reverse-engineered preferences
- High computational burden on validators

### **January 22, 2024: Rebrand to "Apex"**

**Key Changes:**
- Name: Text Prompting → **Apex**
- New GitHub: `macrocosm-os/apex`
- Operator: Macrocosmos (partnership with OTF)
- Mission: "Internet-scale conversational intelligence"
- Achievement: First agent to reach "deep-researcher reasoning"

### **Phase 2: GAN-Style Mechanism (Mid-2024)**

**Revolutionary Innovation:** Miners evaluate each other

**Dual-Role System:**
- **Generator:** Miners create responses
- **Discriminator:** Same miners identify validator vs miner responses
- **Zero-sum game:** Rewards split between roles

**How It Works:**
1. Validator creates reference answer (GPT-4/Claude)
2. Miners submit their answers
3. All responses mixed anonymously
4. Miners guess which are validator-quality
5. Generators rewarded for fooling discriminators
6. Discriminators rewarded for catching generators

**Game Theory Benefits:**
- Can't collude (zero-sum)
- Best strategy: Actually improve quality
- Forces indistinguishability from top models
- Self-regulating quality control

### **Phase 3: Organic Query Integration (Late 2024)**

**Problem:** Miners overfitting to synthetic prompts

**Solution:** Mix real user queries with synthetic ones

**Benefits:**
- Unpredictable evaluation
- Real-world utility validation
- Broader topic coverage

### **March 2025: First GAN Experiments**
- Began testing pure GAN architectures
- "Encouraging results" reported
- Set stage for Apex 3.0

### **August 2025: Apex 3.0 Launch**

**Paradigm Shift:** "Incentivizing answers" → **"Incentivizing judgment"**

**Major Changes:**

1. **Specialized Focus:**
   - **Before:** SN1 did everything (inference, retrieval, evaluation)
   - **After:** Outsourced to specialized subnets
     - Inference → SN64 (Chutes)
     - Web retrieval → SN13 (Dataverse/Gravity)
     - SN1 focuses **exclusively on quality evaluation**

2. **Pure GAN Architecture:**
   - Eliminated hybrid scoring
   - 100% adversarial evaluation
   - Generators create under time constraints
   - Discriminators judge quality
   - Validators provide "oracle answers"

3. **New Economic Model:**
   - Stricter zero-sum settlement
   - Emphasis on **judgment validation**
   - Question: "Can you prove this answer achieves its goal?"

4. **Agentic Capabilities:**
   - LLMs equipped with tools and function calls
   - Miners must use tools/APIs effectively
   - Inter-subnet communication layer

5. **Data Pipeline:**
   - Generates **millions of tokens per day**
   - High-quality training datasets
   - Feeds into SN37 (Finetuning)

---

## Current Purpose (Apex 3.0)

### **Primary Function: AI Quality Evaluation**

Apex is now a **decentralized RLHF/DPO evaluation platform** that:

1. **Produces quality judgments** through adversarial evaluation
2. **Generates training data** for model alignment
3. **Validates judgment accuracy** using game theory
4. **Feeds downstream subnets** with high-quality datasets

### **NOT an Inference API**

**Important:** As of Apex 3.0, SN1 does NOT provide LLM inference to end users.

```
❌ Old (Pre-2025): User → SN1 API → Get text response
✅ New (2025): User → SN64 API → Get inference
              SN1 → Evaluation games → Training data → SN37
```

### **Value Proposition**

Apex provides the **decentralized alternative to OpenAI's alignment team:**

| OpenAI Alignment | Apex (Decentralized) |
|------------------|----------------------|
| Human labelers rate outputs | Miners rate each other's outputs |
| Centralized RLHF process | Distributed adversarial evaluation |
| Internal datasets | Open, subnet-generated datasets |
| Proprietary methods | Transparent game-theoretic mechanisms |

---

## API Access & Pricing

### **Macrocosmos Apex API**

**Base URL:** `https://sn1.api.macrocosmos.ai`

**Endpoints:**
- `POST /v1/chat/completions` - Chat completion endpoint
- `POST /web_retrieval` - Web retrieval endpoint

**Authentication:** API Key required

#### **Example Request: Chat Completions**

```bash
curl https://sn1.api.macrocosmos.ai/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "uids": [1, 2, 3],
    "messages": [
      {
        "content": "Tell me about neural networks",
        "role": "user"
      }
    ],
    "seed": 42,
    "model": "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
    "sampling_parameters": {
      "do_sample": true,
      "max_new_tokens": 512,
      "temperature": 0.7,
      "top_k": 50,
      "top_p": 0.95
    },
    "inference_mode": "Reasoning-Fast",
    "stream": true
  }'
```

#### **Example Response**

```json
{
  "id": "ca0b8681-7b78-4234-8868-71ad1ebfa9ed",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Neural networks are a type of machine learning model inspired by the human brain's structure. They consist of interconnected nodes arranged in layers, including input, hidden, and output layers."
      }
    }
  ]
}
```

#### **Python Example (OpenAI-Compatible)**

```python
import asyncio
import openai

API_KEY = "your-api-key"

async def main():
    client = openai.AsyncOpenAI(
        base_url="https://sn1.api.macrocosmos.ai/v1",
        api_key=API_KEY,
        timeout=openai.Timeout(120, connect=10, read=110),
    )

    result = await client.chat.completions.create(
        model="Default",
        messages=[{"role": "user", "content": "List 5 popular places in Hawaii"}],
        stream=True,
        extra_body={
            "seed": 42,
            "sampling_parameters": {
                "temperature": 0.7,
                "max_new_tokens": 512,
                "top_p": 0.95
            },
            "inference_mode": "Reasoning-Fast"
        }
    )

asyncio.run(main())
```

### **Pricing**

**Rate Limits:**
- Regular API keys: **100 requests/hour**
- Validator keys: **1,000 requests/hour**

**Cost:**
- Pricing listed on "Cravity Page" (not publicly accessible)
- Must sign up at https://app.macrocosmos.ai to see pricing
- Contact: support@macrocosmos.ai for enterprise pricing

**How to Access Pricing:**
1. Sign up at [Constellation](https://app.macrocosmos.ai/signup)
2. Check "TOP UP CREDITS" in Account Settings
3. Or contact support@macrocosmos.ai

**Note:** The Constellation platform primarily shows **Data Universe (SN13 Gravity)** pricing, not Apex inference pricing.

### **Alternative: Inference via Other Subnets**

Since Apex 3.0 outsources inference, consider:

**SN64 (Chutes) - Recommended for Inference:**
- **Base:** $3/month (300 requests/day)
- **Plus:** $10/month (2,000 requests/day)
- **Pro:** $20/month (5,000 requests/day)
- ~85% cheaper than AWS
- Direct platform: https://chutes.ai

**Corcel (Aggregator):**
- Free consumer tier
- Simple API access
- https://corcel.io

**OpenRouter:**
- Pay-per-token pricing
- Access to multiple Bittensor subnets
- https://openrouter.ai

---

## How Evaluation Works

### **Pure Adversarial Evaluation (Apex 3.0)**

#### **Step 1: Prompt Generation**

```python
# Validator generates structured prompt
prompt = validator.generate_prompt(
    task_type="summarization",
    context_source="wikipedia_api",
    topic="Bitcoin",
    difficulty="intermediate",
    length="3 paragraphs"
)
```

#### **Step 2: Reference Answer Creation**

```python
# Validator creates ground truth using unlimited resources
reference = validator.generate_reference(
    model="gpt-4-turbo",  # or Claude, Gemini, etc.
    prompt=prompt,
    context=wikipedia_bitcoin_article,
    max_tokens=500,
    temperature=0.3
)
```

#### **Step 3: Miner Query Broadcast (Generator Role)**

```python
# Send prompt to all active miners
active_miners = get_active_miners(subnet_id=1)

miner_responses = {}
for miner in active_miners:
    response = miner.generate_response(
        prompt=prompt,
        context=wikipedia_bitcoin_article,
        timeout=30_seconds
    )
    miner_responses[miner.uid] = response
```

#### **Step 4: Discriminator Round**

```python
# Mix validator reference with miner responses
all_responses = shuffle([
    {"source": "validator", "text": reference},
    {"source": "miner_42", "text": miner_responses[42]},
    {"source": "miner_17", "text": miner_responses[17]},
    # ... more responses
])

# Remove source labels
anonymized_responses = [r["text"] for r in all_responses]

# Each miner acts as discriminator
for discriminator_miner in active_miners:
    guesses = discriminator_miner.classify_responses(
        responses=anonymized_responses,
        task="identify_validator"
    )

    accuracy = calculate_accuracy(guesses, ground_truth_sources)
    discriminator_scores[discriminator_miner.uid] = accuracy
```

#### **Step 5: Reward Distribution**

```python
# Zero-sum reward system
total_pool = 1.0

# Discriminator rewards
for miner_uid, accuracy in discriminator_scores.items():
    discriminator_reward = (accuracy / len(active_miners)) * 0.5
    allocate_reward(miner_uid, discriminator_reward)

# Generator rewards (inverse of discriminator success)
for miner_uid in miner_responses.keys():
    times_fooled = count_discriminators_fooled(miner_uid)
    fool_rate = times_fooled / len(active_miners)
    generator_reward = fool_rate * 0.5
    allocate_reward(miner_uid, generator_reward)
```

### **Key Evaluation Principles**

**1. Adversarial Improvement**
- Generators try to fool discriminators
- Discriminators try to catch generators
- Arms race drives continuous improvement

**2. Zero-Sum Economics**
- Generators + Discriminators = 1.0 total reward
- One side winning means other side loses
- Prevents collusion

**3. Judgment Focus**
- Question shifted from "Is this correct?" to "Can you prove this achieves its goal?"
- Emphasis on validation over generation

**4. Multi-Dimensional Scoring**
- Generator performance
- Discriminator performance
- Quality judgments
- All contribute to final ranking

---

## Factual Grounding Mechanism

### **How External Data is Integrated**

#### **Current System: Validator-Controlled Context (Primary)**

External APIs (Wikipedia, Wolfram Alpha, news, etc.) are integrated through **pre-fetched context:**

```python
# Step 1: Validator fetches external data BEFORE querying miners
validator_process:
1. Generates prompt: "Summarize Bitcoin's history"
2. Fetches context from Wikipedia API → gets full Bitcoin article
3. Creates reference answer using GPT-4/Claude + context

# Step 2: Validator sends BOTH prompt + context to miners
for miner in active_miners:
    response = miner.generate_response(
        prompt="Summarize Bitcoin's history in 3 paragraphs",
        context=wikipedia_bitcoin_article,  # Pre-fetched by validator
        timeout=30_seconds
    )
```

**Key Point:** Miners receive context as input. They don't decide "I need to fetch Wikipedia now" - the validator already did that.

#### **Example: Live Data Query**

```
User/Validator Prompt: "What's the current price of gold?"

Validator Process:
1. Fetches live gold price from financial API
   → Result: $2,050 per ounce

2. Generates reference answer:
   "Gold is currently trading at $2,050 per ounce as of [timestamp]"

3. Includes context with prompt sent to miners

Miner Evaluation:
- Factual accuracy weighted heavily
- Correct price: High score
- Slightly outdated: Medium score
- Wrong/hallucinated: Low/zero score
```

#### **Multi-Turn Conversations: Context Accumulation**

```
Turn 1:
  Context: [Wikipedia: Bitcoin]
  Prompt: "What is Bitcoin?"
  Expected: Factually accurate intro

Turn 2:
  Context: [Wikipedia: Bitcoin] + [Live Bitcoin price API]
  Prompt: "How has Bitcoin's price changed?"
  Expected: Accurate historical movements

Turn 3:
  Context: [Bitcoin whitepaper PDF]
  Prompt: "What problem was Bitcoin designed to solve?"
  Expected: Reference to double-spending, decentralization
```

Each turn, the **validator decides** what context to fetch and include.

### **Emerging: Agentic Tool Use (Apex 3.0)**

**However,** newer developments introduce true agentic behavior:

> "Validators host an ongoing competitive arena—where **LLMs are equipped with tools and function calls**—and only the most capable earn TAO emissions."

> "Miners must become **adept at using tools and APIs** in order to fulfill validation tasks."

**System is Transitioning:**

| Aspect | Original Design | Apex 3.0 (2025) |
|--------|----------------|-----------------|
| Context source | Validator pre-fetches | Miner can fetch |
| Tool access | Passive (context provided) | Active (miner calls tools) |
| Decision-making | Validator decides what data | Miner decides when to use tools |
| Function calling | No | Yes |

### **Why This Hybrid Approach?**

**For evaluation consistency:**
- All miners see **same context** → fair comparison
- Prevents "miner A has better APIs than miner B" issues
- Validator controls "ground truth" data

**For agentic evolution:**
- Miners demonstrate **tool use skills**
- Rewards adaptability, not just generation
- Enables complex multi-step reasoning

### **Bottom Line**

**Current state:** Mostly **validator-controlled RAG** (Retrieval-Augmented Generation)
- Validator retrieves → includes as context → miners respond

**Future direction:** **Agentic tool use**
- Miners have access to tools (search, calculators, APIs)
- Miners decide when and how to use them
- Evaluated on both output quality AND tool use effectiveness

The "web_retrieval" endpoint suggests some agentic capability already exists where miners can perform their own searches.

---

## Integration with Other Subnets

### **The Bittensor AI Stack (Apex-Centric View)**

```
┌─────────────────────────────────────────┐
│         SN1 (Apex)                      │
│   Quality Evaluation & Training Data    │
│   - GAN-style adversarial evaluation    │
│   - Millions of tokens/day output       │
└──────────┬──────────────────────────────┘
           │
           │ Training Data
           ↓
┌─────────────────────────────────────────┐
│      SN37 (Finetuning)                  │
│   Uses Apex data to improve models      │
│   - Loss-based competition               │
│   - Continuous model improvement         │
└──────────┬──────────────────────────────┘
           │
           │ Improved Models
           ↓
┌─────────────────────────────────────────┐
│      SN64 (Chutes)                      │
│   Serves improved models to users       │
│   - Serverless AI inference              │
│   - 85% cheaper than AWS                 │
└─────────────────────────────────────────┘
           │
           │ Inference
           ↓
        End Users
```

### **SN13 (Dataverse/Gravity) - Web Retrieval**

**Function:** Apex outsources web search and data retrieval to SN13

**Flow:**
```
Apex needs context → Queries SN13 → Gets web data → Includes in evaluation
```

**Why Separate:**
- SN13 specializes in data scraping at scale
- More efficient than each validator scraping independently
- Shared infrastructure benefits all subnets

### **SN37 (Finetuning) - Model Improvement**

**Function:** Uses Apex's training data to finetune models

**Flow:**
```
Apex generates millions of tokens/day (high-quality Q&A pairs)
    ↓
SN37 miners use this data for model finetuning
    ↓
Improved models deployed back to inference subnets
```

**Feedback Loop:**
- Better Apex evaluation → Better training data
- Better training data → Better models
- Better models → Higher quality inference
- Cycle repeats

### **SN64 (Chutes) - Inference Delivery**

**Function:** Apex outsources actual LLM inference to SN64

**Why:**
- Chutes specializes in serverless compute
- More efficient GPU utilization
- Cheaper for end users (85% less than AWS)
- Apex focuses purely on evaluation

**Architecture:**
```
User query → SN64 (Chutes) → Get response
                ↓
         Meanwhile, separately:
         SN1 (Apex) → Evaluates quality → Generates training data
```

### **The Value Chain**

```
SN1 (Apex) → Quality judgments + training data
              ↓
SN37 (Finetuning) → Uses data to improve models
              ↓
SN64 (Chutes) → Serves improved models to users
              ↓
SN13 (Gravity) → Provides context/data for evaluation
              ↑
        Feeds back to SN1
```

Apex moved **upstream** in the value chain—from serving end users to serving other subnets.

---

## Who Uses Apex

### **Direct Users (Developers)**

#### **1. Researchers & Experimenters**

**Use Cases:**
- Academic research on decentralized AI
- AI safety research
- Studying game-theoretic evaluation mechanisms
- Novel incentive mechanism research

**Why Apex:**
- Unique GAN-style architecture
- Access to training data pipeline
- Open experimentation (run miner/validator)
- Novel token economics

#### **2. Training Data Consumers**

**Use Cases:**
- Fine-tuning custom models
- Building RLHF datasets
- Creating alignment training data
- Research on human preferences

**Why Apex:**
- Millions of tokens/day of high-quality data
- Adversarially validated quality
- Organic + synthetic query mix
- Open, decentralized source

#### **3. Subnet Operators (Indirect)**

**Use Cases:**
- SN37 (Finetuning) - Primary consumer
- Other LLM subnets needing training data
- Research subnets studying evaluation

**Why Apex:**
- Continuous data pipeline
- Quality-validated outputs
- Game-theoretically sound
- Open access

### **Indirect Users (via Integrated Services)**

#### **4. End Users of Downstream Services**

**Flow:**
```
User uses Chutes/Corcel/OpenRouter
    ↓
Gets inference from models
    ↓
Those models improved by Apex training data
    ↓
User benefits from better quality (indirectly via Apex)
```

**Examples:**
- ChatGPT-style apps using Bittensor models
- AI coding assistants
- Content generation tools
- Translation services

### **Who Should NOT Use Apex Directly**

**Wrong Use Cases:**

1. **Looking for LLM inference API**
   - Apex no longer provides this
   - Use: SN64 (Chutes), Corcel, or OpenRouter instead

2. **Need simple chat interface**
   - Apex is evaluation platform, not chat
   - Use: Chattensor or Chutes

3. **Enterprise requiring SLAs**
   - Apex is experimental/research-focused
   - Use: OpenAI/Anthropic for production

4. **Non-technical users**
   - Apex requires understanding of subnet mechanics
   - Use: Consumer apps built on Bittensor

---

## Technical Architecture

### **Subnet Configuration**

**Subnet ID:** 1
**Maximum Neurons:** 256
**Validator Slots:** 64 (reserved)
**Miner Slots:** 192 (remainder)
**Tempo:** ~360 blocks (~12 minutes)

### **Validator Requirements**

**Responsibilities:**
- Generate prompts
- Create reference answers (using GPT-4/Claude/Gemini)
- Query miners
- Run discriminator evaluation rounds
- Submit weights to Yuma Consensus

**Resources Needed:**
- Access to external LLM APIs (GPT-4, Claude, etc.)
- API keys for data sources (Wikipedia, Wolfram, etc.)
- Compute for running evaluation models
- TAO stake (minimum ~11,500 TAO for validator slot)

### **Miner Requirements**

**Responsibilities:**
- **Generator role:** Create responses to prompts
- **Discriminator role:** Classify responses as validator/miner quality
- Maintain high uptime
- Process requests quickly

**Resources Needed:**
- GPU for inference (RTX 4090, A100, etc.)
- LLM model (Llama 3, Mistral, custom fine-tuned)
- Optional: Tool-use capabilities for agentic tasks
- TAO for registration (~0.001 TAO)

### **Emission Distribution**

**Total to SN1:** Varies (Root Network voting determines)
- Historically: ~5-10% of total TAO emissions
- As of Dec 2025: Subject to dTAO (dynamic TAO) voting

**Split:**
- **41% to miners** (via Yuma Consensus)
- **18% to validators** (based on alignment with consensus)
- **Remaining** to subnet owner/protocol

### **Yuma Consensus Mechanism**

**Process:**
1. Each validator submits weights (scores) for all miners
2. Weights are stake-weighted (more stake = more influence)
3. Outlier weights clipped (prevents manipulation)
4. Consensus weights computed
5. Emissions distributed proportionally

**Formula (simplified):**
```python
consensus_weight[miner] = weighted_median(
    validator_weights[miner] for each validator,
    weighted_by=validator_stake
)

miner_reward = consensus_weight[miner] * total_emissions
```

---

## Comparison to Alternatives

### **Apex vs Centralized Evaluation (OpenAI/Anthropic)**

| Aspect | OpenAI RLHF | Apex (SN1) |
|--------|-------------|------------|
| **Evaluators** | Human labelers (paid contractors) | Miners (economically incentivized) |
| **Data Source** | Internal proprietary datasets | Open, subnet-generated data |
| **Method** | Human preference comparisons | Adversarial GAN-style evaluation |
| **Scale** | Limited by labeler availability | Unlimited (distributed miners) |
| **Cost** | High (labor costs) | Lower (automated evaluation) |
| **Transparency** | Closed, proprietary | Open, on-chain |
| **Control** | Centralized company | Decentralized consensus |
| **Quality Assurance** | Expert oversight | Game theory + consensus |

**OpenAI Advantages:**
- Consistent quality standards
- Expert human oversight
- Proven track record
- Higher initial quality baseline

**Apex Advantages:**
- Decentralized (no single point of failure)
- Economically incentivized improvement
- Open, transparent process
- Scales globally without labor constraints
- Censorship-resistant

### **Apex vs Traditional Benchmarks (MMLU, HumanEval)**

| Aspect | Static Benchmarks | Apex |
|--------|------------------|------|
| **Evaluation Type** | Fixed test sets | Continuous adversarial |
| **Gaming Resistance** | Low (can memorize) | High (arms race) |
| **Real-World Utility** | Indirect correlation | Organic query validation |
| **Update Frequency** | Rare (manual updates) | Continuous (every tempo) |
| **Generalization** | Test set overfitting risk | Broad, adversarial robustness |

### **Apex vs Other Bittensor Evaluation Subnets**

**SN6 (Nous):**
- Focus: Fine-tuning competition
- Method: Loss comparison on standard datasets
- Data: Cortex.t synthetic data
- Output: Fine-tuned models

**Apex (SN1):**
- Focus: Quality judgment validation
- Method: Adversarial GAN evaluation
- Data: Organic + synthetic queries
- Output: Training data + quality scores

**Complementary, not competitive**

---

## Use Cases & Applications

### **1. Research Applications**

**Academic Research:**
- Studying decentralized AI evaluation
- Game theory in AI systems
- Incentive mechanism design
- AI alignment research

**Corporate R&D:**
- Experimenting with novel evaluation methods
- Benchmarking internal models against decentralized alternatives
- Building hybrid centralized-decentralized systems

### **2. Training Data Generation**

**Model Finetuning:**
- RLHF dataset creation
- DPO dataset creation
- Instruction-following datasets
- Conversational AI training data

**Quality:** Millions of tokens/day, adversarially validated

### **3. Quality Assessment**

**Model Benchmarking:**
- Test custom models against Apex evaluation
- Compare to decentralized quality standards
- Validate alignment with human preferences

**Production Monitoring:**
- Continuous quality monitoring of deployed models
- Detect degradation over time
- A/B testing different model versions

### **4. Indirect Benefits (via Ecosystem)**

**Better Models Everywhere:**
- SN37 uses Apex data → better models
- SN64 serves those models → better inference
- Users get higher quality → ecosystem grows

**Network Effects:**
- More miners → more evaluation capacity
- More data → better finetuning
- Better models → more users
- More users → more value → more miners

---

## Technical Challenges & Solutions

### **Challenge 1: Discriminator Collusion**

**Problem:** Can miners collude to share signals about validator responses?

**Mitigations:**
1. Response anonymization (no metadata leaks)
2. Random sampling (different miners see different subsets)
3. Timing variance (queries at random intervals)
4. Cryptographic mixing (responses shuffled)
5. Stake slashing (detected collusion penalties)

### **Challenge 2: Reference Model Bias**

**Problem:** If all validators use GPT-4, miners just clone GPT-4 style.

**Mitigations:**
1. Diverse validator models (GPT-4, Claude, Gemini, Llama)
2. Organic queries (no predictable "reference style")
3. Discriminator task rewards indistinguishability, not style matching
4. Explicit diversity rewards
5. Periodic human evaluation

### **Challenge 3: Computational Cost**

**Problem:** Running both generator AND discriminator is expensive.

**Mitigations:**
1. Lightweight discriminators (smaller models)
2. Reward balancing (discriminator rewards proportional to cost)
3. Batch processing (amortize cost)
4. Model caching (reuse loaded models)
5. Specialized hardware (GPU optimization)

### **Challenge 4: Gaming the Discriminator**

**Problem:** Miners might recognize validator models via stylistic tells.

**Mitigations:**
1. Style randomization (validators vary writing styles)
2. Model rotation (change reference models periodically)
3. Semantic focus (score meaning, not style)
4. Adversarial prompts (designed to prevent style gaming)
5. Organic query mixing (unpredictable patterns)

---

## Future Directions

### **Potential Improvements**

**1. Multi-Modal Evaluation**
- Evaluate text + images + code simultaneously
- Cross-modal quality assessment
- Richer evaluation signals

**2. Hierarchical Evaluation**
- Specialist discriminators for different task types
- Meta-discriminators evaluate discriminator quality
- Multi-level quality control

**3. User Feedback Integration**
- Actual end-users rate outputs
- Organic feedback loop
- Align with real-world utility

**4. Automated Prompt Generation**
- AI-generated prompts (validated by humans)
- Infinite diverse prompts
- Reduce validator burden

**5. Cross-Subnet Quality Signals**
- Apex outputs feed into other subnets
- Quality propagates across network
- Emergent evaluation standards

### **Open Research Questions**

**1. Optimal reward balance?**
- Currently arbitrary (e.g., 50/50 generator/discriminator)
- Could dynamic adjustment improve outcomes?
- Market-driven equilibrium?

**2. Validator centralization?**
- High-quality validators may dominate consensus
- Small validators discouraged
- Need validator diversity for robustness

**3. Generalization to arbitrary AI tasks?**
- Works well for text, but:
  - Long-form reasoning?
  - Creative tasks (art, music)?
  - Agentic behavior?
  - Multi-modal outputs?

**4. Long-term equilibrium?**
- Will generators always fool discriminators?
- Will discriminators become perfect detectors?
- Stable oscillation or runaway dynamics?

**5. "Real" quality vs "gaming the metric"?**
- Goodhart's Law applies
- Are miners producing genuinely useful outputs?
- Or just optimizing for Yuma consensus?
- Human evaluation still gold standard?

---

## Getting Started

### **For Users (API Access)**

**If you want inference (NOT evaluation):**

1. **Use Chutes (SN64) instead:**
   - Visit: https://chutes.ai
   - Pricing: $3-20/month
   - 85% cheaper than AWS

2. **Or use Corcel:**
   - Visit: https://corcel.io
   - Free consumer tier
   - Simple API

3. **Or use OpenRouter:**
   - Visit: https://openrouter.ai
   - Pay-per-token
   - Access multiple Bittensor subnets

**If you want Apex API for research:**

1. Contact Macrocosmos: support@macrocosmos.ai
2. Sign up at: https://app.macrocosmos.ai/signup
3. Request access to evaluation API

### **For Miners**

**Requirements:**
- GPU (RTX 4090+ recommended)
- LLM model (Llama 3, Mistral, custom)
- Python environment
- ~0.001 TAO for registration

**Setup:**
1. Clone Apex repo: `git clone https://github.com/macrocosm-os/apex`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure miner settings
4. Register on subnet: `btcli subnet register --netuid 1`
5. Start mining: `python neurons/miner.py`

**Documentation:** https://docs.macrocosmos.ai/subnets/subnet-1-apex

### **For Validators**

**Requirements:**
- ~11,500+ TAO for validator slot
- Access to GPT-4/Claude/Gemini APIs
- High-uptime server
- Python environment

**Setup:**
1. Clone Apex repo
2. Configure validator settings
3. Set up API keys (OpenAI, Anthropic, etc.)
4. Register validator: `btcli subnet register --netuid 1`
5. Start validating: `python neurons/validator.py`

**Documentation:** https://docs.macrocosmos.ai/subnets/subnet-1-apex/subnet-1-validation

### **For Researchers**

**Access Training Data:**
- Contact Macrocosmos for dataset access
- Or run your own miner/validator to collect data
- Join Discord: https://discord.gg/bittensor (#apex-sn1)

**Run Experiments:**
- Fork Apex repo
- Modify evaluation mechanisms
- Test on testnet first
- Propose improvements via GitHub

---

## Resources

### **Official Documentation**

- **Apex Homepage:** https://www.macrocosmos.ai/sn1
- **GitHub:** https://github.com/macrocosm-os/apex
- **Macrocosmos Docs:** https://docs.macrocosmos.ai/subnets/subnet-1-apex
- **API Docs:** https://docs.macrocosmos.ai/developers/api-documentation/sn1-apex
- **Bittensor Docs:** https://docs.learnbittensor.org/subnets/walkthrough-prompting

### **Analytics & Metrics**

- **Taostats SN1:** https://taostats.io/subnets/1/chart
- **Taostats Metagraph:** https://taostats.io/subnets/1/metagraph
- **SubnetAlpha:** https://subnetalpha.ai/subnet/apex/
- **TaoMarketCap:** https://taomarketcap.com

### **Community**

- **Bittensor Discord:** https://discord.gg/bittensor (channel: #apex-sn1)
- **Macrocosmos Twitter:** https://x.com/macrocosmos_ai
- **Bittensor Twitter:** https://x.com/bittensor_
- **Reddit:** r/bittensor

### **Research & Analysis**

- **Apex 3.0 Announcement:** https://macrocosmosai.substack.com/p/apex-30-game-theoretic-ai-on-bittensor
- **Subnet 1 Deep Dive:** [subnet1-evaluation-deep-dive.md](./subnet1-evaluation-deep-dive.md)
- **Messari Research:** https://messari.io (search "Bittensor")
- **Bittensor Whitepaper:** https://bittensor.com/whitepaper

### **Development Resources**

- **Bittensor SDK:** https://github.com/opentensor/bittensor
- **Subnet Template:** https://github.com/opentensor/bittensor-subnet-template
- **Apex Repo:** https://github.com/macrocosm-os/apex
- **Python Package:** `pip install bittensor`

### **Related Subnets**

- **SN13 (Gravity):** https://datauniverse.macrocosmos.ai
- **SN37 (Finetuning):** https://www.macrocosmos.ai/sn37
- **SN64 (Chutes):** https://chutes.ai
- **Directory:** [subnet-directory.md](./subnet-directory.md)

---

## Frequently Asked Questions

### **Q: Can I use Apex for ChatGPT-style inference?**

**A:** As of Apex 3.0 (August 2025), **no**. Apex no longer provides direct inference. Use:
- SN64 (Chutes): https://chutes.ai
- Corcel: https://corcel.io
- OpenRouter: https://openrouter.ai

### **Q: What is Apex used for then?**

**A:** Apex is now an **AI quality evaluation and training data generation platform**. It:
- Validates AI quality through adversarial evaluation
- Produces training datasets for model alignment
- Feeds data to SN37 (Finetuning) for model improvement

### **Q: Is Apex free to use?**

**A:** Pricing is not publicly listed. Contact Macrocosmos (support@macrocosmos.ai) for:
- API access pricing
- Training data access
- Enterprise/research partnerships

### **Q: How is Apex different from OpenAI's RLHF?**

**A:**
- **OpenAI:** Human labelers rate outputs (centralized, proprietary)
- **Apex:** Miners rate each other's outputs via game theory (decentralized, open)

Both achieve similar goals (AI alignment) through different mechanisms.

### **Q: Can I mine on Apex with consumer hardware?**

**A:** Yes, but competitiveness depends on:
- GPU quality (RTX 4090 competitive, better with A100/H100)
- Model quality (fine-tuned models perform better)
- Uptime (24/7 recommended)

Entry barrier is lower than inference subnets.

### **Q: How much can I earn mining Apex?**

**A:** Varies widely:
- Top miners: $50K-200K/month (in TAO)
- Mid-tier: $5K-20K/month
- Bottom 50%: $100-1K/month

Depends on: performance, competition, TAO price, subnet emissions.

### **Q: Is Apex's evaluation better than traditional benchmarks?**

**A:** Different trade-offs:
- **Apex:** Continuous, adversarial, resistant to gaming
- **Benchmarks:** Standardized, reproducible, established

Apex is complementary, not replacement for MMLU/HumanEval.

### **Q: What's the relationship between SN1 and SN64?**

**A:**
- **SN1 (Apex):** Evaluation + training data generation
- **SN64 (Chutes):** Inference delivery to end users

Apex outsourced inference to Chutes in Apex 3.0 to specialize.

### **Q: How do I get Apex training data for my research?**

**A:** Options:
1. Contact Macrocosmos for dataset access
2. Run your own miner/validator to collect data
3. Join Bittensor Discord and ask in #apex-sn1
4. Check if data is published on Hugging Face

### **Q: Will Apex add inference back in the future?**

**A:** Unlikely. The strategic shift to evaluation specialization is intentional. For inference, use SN64 (Chutes) or other inference subnets.

---

## Conclusion

**Subnet 1 (Apex)** represents one of the most sophisticated experiments in decentralized AI quality evaluation. Its evolution from a simple inference API to a game-theoretic evaluation platform demonstrates the power of economic incentives to drive continuous improvement in AI systems.

**Key Takeaways:**

1. **Apex is NOT an inference API** (as of 2025) - use SN64 (Chutes) instead
2. **Apex IS an evaluation platform** - produces quality judgments and training data
3. **GAN-style mechanism** - miners compete as both generators and discriminators
4. **Zero-sum economics** - aligned incentives through game theory
5. **Feeds ecosystem** - training data improves models across all subnets
6. **Open and transparent** - decentralized alternative to closed RLHF processes

**The Bigger Picture:**

Apex proves that **decentralized AI quality evaluation is possible**. If it succeeds at scale, it demonstrates:
- Economic incentives can coordinate complex AI tasks
- Game theory can solve alignment problems
- Decentralized networks can match centralized quality
- Open, transparent processes can compete with proprietary systems

The question isn't "Will Apex replace OpenAI's alignment team?" but rather "Can Apex carve out a significant niche in the AI alignment ecosystem?" Given its progress and innovation, the answer appears to be **yes**.

---

**Document Version:** 1.0
**Last Updated:** December 13, 2025
**Contributors:** Research compiled from Bittensor documentation, Macrocosmos resources, and community analysis
**License:** MIT

**Related Documents:**
- [Subnet 1 Evaluation Deep Dive](./subnet1-evaluation-deep-dive.md)
- [Subnet Directory](./subnet-directory.md)
- [Bittensor History](./history.md)
