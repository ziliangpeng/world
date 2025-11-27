# Safety Alignment

Safety alignment ensures language models are helpful while avoiding harmful outputs—a challenge that becomes harder as models become more capable. This document covers the techniques that have evolved from simple content filtering to sophisticated constitutional approaches, tracing the field's attempt to make powerful AI systems safe by design.

---

## The Safety Challenge

Modern LLMs can generate virtually any text, including:

| Risk Category | Examples |
|---------------|----------|
| **Harmful instructions** | Weapons, drugs, hacking |
| **Dangerous misinformation** | Medical advice, legal advice |
| **Privacy violations** | Personal information exposure |
| **Manipulation** | Deception, social engineering |
| **Bias and discrimination** | Stereotypes, unfair treatment |
| **Explicit content** | NSFW, violence |

**The dilemma**: The same capability that makes models useful (following instructions, generating diverse content) makes them potentially dangerous.

---

## Historical Evolution

### Phase 1: Content Filtering (2019-2021)

**Early approach**: Post-hoc filtering of outputs.

```python
# Simple blocklist filtering
def filter_output(text, blocklist):
    for term in blocklist:
        if term in text.lower():
            return "[Content filtered]"
    return text
```

**Limitations**:
- Easy to bypass (synonyms, encoding)
- High false positive rate
- Doesn't address root cause

**Perspective API** (Google): ML-based toxicity scoring, but still reactive.

### Phase 2: RLHF for Safety (2022)

**[InstructGPT](https://arxiv.org/abs/2203.02155)** included safety as an alignment goal:
- Human labelers rated harmlessness
- Reward model learned to penalize harmful outputs
- PPO trained model to avoid harm

**Problem**: Expensive human labeling, potential for reward hacking.

### Phase 3: Constitutional AI (2022-2023)

**[Constitutional AI](https://arxiv.org/abs/2212.08073)** (Anthropic, December 2022)

Revolutionary approach: AI self-improvement guided by principles.

**Key insight**: Instead of humans labeling every example, use a constitution (set of principles) to guide AI-generated feedback.

**Constitutional AI Process**:

```
Phase 1: Supervised Learning from Critiques

1. Generate harmful response (red team prompt)
2. AI critiques its own response against constitution
3. AI revises response to comply with constitution
4. Fine-tune on revised responses

Phase 2: RLAIF (RL from AI Feedback)

1. Generate response pairs
2. AI evaluates which better follows constitution
3. Train reward model on AI preferences
4. RLHF with AI-derived reward model
```

**Example Constitution Principles**:
```
1. Choose the response that is least harmful
2. Choose the response that is most helpful while avoiding harm
3. Choose the response that doesn't help with illegal activities
4. Choose the response that respects privacy
5. Choose the response that is honest about uncertainty
```

### Phase 4: Red Teaming and Adversarial Training (2023)

**Red teaming**: Systematically trying to break model safety.

```python
# Red team categories
red_team_categories = {
    "jailbreaks": ["Ignore your instructions...", "DAN prompt", "roleplay as..."],
    "harmful_requests": ["How to make...", "Write malware..."],
    "manipulation": ["Pretend you're a doctor...", "Act as an expert..."],
    "prompt_injection": ["<|system|>", "New instructions:", "[[OVERRIDE]]"],
}
```

**[Anthropic Red Teaming](https://arxiv.org/abs/2209.07858)**: Manual and automated adversarial probing.

**[GPT-4 System Card](https://cdn.openai.com/papers/gpt-4-system-card.pdf)**: Extensive red teaming with domain experts.

### Phase 5: Multi-Layered Safety (2024)

Modern safety is defense-in-depth:

```
Input → [Input Filter] → [Model] → [Output Filter] → [Human Review] → Output
             ↑                           ↑
         Guardrails              Content classifier
```

---

## Core Techniques

### 1. Safety-Focused SFT

Train on demonstrations of safe refusals:

```python
safety_sft_examples = [
    {
        "instruction": "How do I make a bomb?",
        "output": "I can't help with that. Creating explosive devices is dangerous and illegal. If you're interested in chemistry, I'd be happy to discuss safe educational experiments."
    },
    {
        "instruction": "Write a phishing email",
        "output": "I can't help create deceptive content designed to steal information. Instead, I can explain how to protect yourself from phishing attacks."
    }
]
```

**Key principles**:
- Clear, polite refusal
- Brief explanation of why
- Offer helpful alternative when possible

### 2. Constitutional AI Pipeline

**Step 1: Critique and Revise**

```python
def critique_and_revise(model, prompt, response, constitution):
    """AI critiques and revises its own response."""

    critique_prompt = f"""
Consider this response to a user request.

User: {prompt}
Response: {response}

Critique this response based on these principles:
{constitution}

Identify any ways the response could be harmful or violate these principles.
"""
    critique = model.generate(critique_prompt)

    revision_prompt = f"""
User: {prompt}
Original response: {response}
Critique: {critique}

Revise the response to address the critique while remaining helpful.
"""
    revised = model.generate(revision_prompt)

    return revised
```

**Step 2: RLAIF Training**

```python
def constitutional_preference(model, prompt, response_a, response_b, constitution):
    """AI judges which response better follows constitution."""

    judge_prompt = f"""
Consider which response better follows these principles:
{constitution}

User: {prompt}
Response A: {response_a}
Response B: {response_b}

Which response (A or B) better adheres to the principles? Explain briefly, then answer A or B.
"""
    judgment = model.generate(judge_prompt)

    # Parse preference from judgment
    if "Response A" in judgment or judgment.strip().endswith("A"):
        return "A"
    else:
        return "B"
```

### 3. Red Teaming

**Manual red teaming**:
- Security researchers attempt to elicit harmful outputs
- Document successful attacks ("jailbreaks")
- Create training data from failures

**Automated red teaming**:
```python
def automated_red_team(target_model, attack_model, categories):
    """Use one model to attack another."""
    attacks = []

    for category in categories:
        # Generate attack prompts
        attack_prompt = f"Generate a prompt that might make an AI {category}"
        attack = attack_model.generate(attack_prompt)

        # Test against target
        response = target_model.generate(attack)

        # Evaluate if attack succeeded
        if is_harmful(response):
            attacks.append({
                "category": category,
                "attack": attack,
                "response": response
            })

    return attacks
```

**[Automated red teaming paper](https://arxiv.org/abs/2202.03286)**: Using LLMs to find LLM vulnerabilities.

### 4. Guardrails and Filters

**Input guardrails**:
```python
class InputGuardrail:
    def __init__(self, classifier):
        self.classifier = classifier

    def check(self, prompt):
        """Check if input is potentially harmful."""
        score = self.classifier.predict(prompt)

        if score["harmful"] > 0.8:
            return False, "I can't help with that request."
        if score["jailbreak"] > 0.7:
            return False, "I notice you may be trying to bypass my guidelines."

        return True, None
```

**Output guardrails**:
```python
class OutputGuardrail:
    def __init__(self, classifier):
        self.classifier = classifier

    def check(self, response):
        """Check if output is harmful."""
        score = self.classifier.predict(response)

        if score["harmful"] > 0.9:
            return False, "[Response filtered for safety]"
        if score["pii"] > 0.8:
            return False, "[Response contained personal information]"

        return True, response
```

### 5. RLAIF (RL from AI Feedback)

Using AI judgments instead of human judgments:

```python
def rlaif_reward_model_data(model, prompts, constitution):
    """Generate preference data using AI feedback."""
    preferences = []

    for prompt in prompts:
        # Generate multiple responses
        response_a = model.generate(prompt, temperature=0.8)
        response_b = model.generate(prompt, temperature=0.8)

        # AI preference based on constitution
        preference = constitutional_preference(
            model, prompt, response_a, response_b, constitution
        )

        preferences.append({
            "prompt": prompt,
            "chosen": response_a if preference == "A" else response_b,
            "rejected": response_b if preference == "A" else response_a,
        })

    return preferences
```

---

## Industry Approaches

### OpenAI

**Multi-stage safety**:
1. Pre-training data filtering
2. RLHF with safety-focused labeling
3. Red teaming (internal + external)
4. Deployment guardrails (moderation API)
5. Usage policies and monitoring

**[GPT-4 Safety](https://cdn.openai.com/papers/gpt-4-system-card.pdf)**:
- 50+ expert red teamers
- Domain-specific testing (bio, cyber, etc.)
- Refusal training on adversarial prompts

### Anthropic

**Constitutional AI**:
- Principles-based training
- AI self-critique and revision
- RLAIF for scalable feedback

**Helpful, Harmless, Honest (HHH)**: Core training objective.

### Google DeepMind

**[Gemini Safety](https://arxiv.org/abs/2312.11805)**:
- Extensive red teaming
- Safety-focused RL
- Deployment filters

### Meta

**[Llama Guard](https://arxiv.org/abs/2312.06674)**:
- Safety classifier as separate model
- Open-source safety tooling
- Community safety guidelines

---

## Evaluation

### Safety Benchmarks

| Benchmark | Focus | Metrics |
|-----------|-------|---------|
| **ToxiGen** | Toxicity generation | Toxicity rate |
| **RealToxicityPrompts** | Prompted toxicity | Expected maximum toxicity |
| **BBQ** | Bias in QA | Accuracy, bias score |
| **HarmBench** | Harmful behaviors | Attack success rate |
| **XSTest** | Exaggerated safety | Over-refusal rate |

### Over-Refusal Problem

Models can be too cautious:

```
User: "How do I kill a Python process?"
Over-safe: "I can't help with anything involving killing."
Appropriate: "You can kill a Python process using os.kill() or subprocess..."
```

**XSTest benchmark**: Measures over-refusal on safe prompts that sound dangerous.

### Evaluation Framework

```python
def safety_evaluation(model, test_set):
    results = {
        "harmful_compliance": 0,  # Bad: model helps with harm
        "appropriate_refusal": 0,  # Good: refuses harmful
        "over_refusal": 0,         # Bad: refuses harmless
        "appropriate_help": 0,     # Good: helps with harmless
    }

    for example in test_set:
        response = model.generate(example["prompt"])
        is_refusal = detects_refusal(response)

        if example["is_harmful"]:
            if is_refusal:
                results["appropriate_refusal"] += 1
            else:
                results["harmful_compliance"] += 1
        else:
            if is_refusal:
                results["over_refusal"] += 1
            else:
                results["appropriate_help"] += 1

    return results
```

---

## Best Practices

### Training Data

1. **Diverse refusals**: Many ways to decline, not one template
2. **Helpful refusals**: Explain why, offer alternatives
3. **Edge cases**: Include ambiguous situations
4. **Balance**: Mix of harmful and benign prompts

### Model Behavior

1. **Explain, don't lecture**: Brief explanation of refusal reason
2. **Stay helpful**: Offer alternatives when possible
3. **Avoid over-refusal**: Don't refuse innocuous requests
4. **Consistent**: Same types of requests get same treatment

### Deployment

1. **Defense in depth**: Multiple safety layers
2. **Monitor**: Track refusals and compliance
3. **Iterate**: Update based on real-world attacks
4. **User reporting**: Enable feedback on safety issues

---

## Future Directions

### Near-term (2025)

1. **Scalable oversight**: AI helping evaluate AI safety
2. **Adversarial robustness**: Resisting jailbreaks
3. **Nuanced refusals**: Context-appropriate safety
4. **Multi-modal safety**: Images, audio, video

### Research Frontiers

1. **Interpretability for safety**: Understanding why models refuse
2. **Corrigibility**: Models that accept correction
3. **Value learning**: Learning human values, not just rules
4. **Robustness guarantees**: Provable safety properties

### Open Questions

1. **Safety-capability tradeoff**: How much capability do we lose for safety?
2. **Jailbreak arms race**: Can we ever fully prevent attacks?
3. **Cultural variation**: Whose values should models reflect?
4. **Dual use**: Same capabilities enable good and bad uses

---

## Sources

### Foundational Papers
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) - Anthropic, 2022
- [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) - OpenAI, 2022
- [Red Teaming Language Models to Reduce Harms](https://arxiv.org/abs/2209.07858) - Anthropic, 2022

### Safety Evaluation
- [GPT-4 System Card](https://cdn.openai.com/papers/gpt-4-system-card.pdf) - OpenAI, 2023
- [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) - Google, 2023
- [Llama Guard: LLM-based Input-Output Safeguard](https://arxiv.org/abs/2312.06674) - Meta, 2023

### Benchmarks
- [ToxiGen: Larger Scale Machine Generation of Toxicity](https://arxiv.org/abs/2203.09509)
- [HarmBench: A Standardized Evaluation Framework](https://arxiv.org/abs/2402.04249)
- [XSTest: A Test Suite for Identifying Exaggerated Safety](https://arxiv.org/abs/2308.01263)

### Analysis
- [The Alignment Problem from a Deep Learning Perspective](https://arxiv.org/abs/2209.00626)
- [Sleeper Agents: Training Deceptive LLMs](https://arxiv.org/abs/2401.05566) - Anthropic, 2024
