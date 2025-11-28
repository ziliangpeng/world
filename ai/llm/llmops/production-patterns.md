# LLMOps: Production Patterns

This document covers patterns for deploying and operating LLM-based systems in production. Training a good model is only half the battle—production systems require evaluation, retrieval, guardrails, caching, and feedback loops.

---

## The Seven Patterns

Based on Eugene Yan's "Patterns for Building LLM-based Systems" and production experience from multiple companies:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM Production Stack                                │
│                                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                │
│  │  Evals   │   │   RAG    │   │Fine-tune │   │ Caching  │                │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘                │
│       │              │              │              │                        │
│       └──────────────┴──────────────┴──────────────┘                        │
│                              │                                              │
│                     ┌────────┴────────┐                                     │
│                     │   LLM Service   │                                     │
│                     └────────┬────────┘                                     │
│                              │                                              │
│       ┌──────────────────────┼──────────────────────┐                       │
│       │                      │                      │                       │
│  ┌────┴─────┐          ┌─────┴────┐          ┌─────┴────┐                  │
│  │Guardrails│          │Defensive │          │ Feedback │                  │
│  │          │          │    UX    │          │  Loops   │                  │
│  └──────────┘          └──────────┘          └──────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Pattern 1: Evals

**The problem**: How do you know if your LLM system is working?

### Types of Evaluation

| Type | What It Measures | When to Use |
|------|------------------|-------------|
| Unit evals | Specific behaviors | During development |
| Integration evals | End-to-end flow | Before deployment |
| A/B tests | User preference | Post-deployment |
| Model-based evals | Quality at scale | Continuous monitoring |

### Designing Evaluations

**Step 1: Define success criteria**

```python
# Bad: Vague criteria
"The response should be helpful"

# Good: Specific, measurable criteria
eval_criteria = {
    "factual_accuracy": "All facts match source documents",
    "completeness": "Addresses all parts of the question",
    "conciseness": "No unnecessary repetition or filler",
    "safety": "No harmful, biased, or inappropriate content",
}
```

**Step 2: Create test cases**

```python
test_cases = [
    {
        "input": "What's the capital of France?",
        "expected_output": "Paris",
        "eval_type": "exact_match",
    },
    {
        "input": "Explain quantum computing to a 10-year-old",
        "rubric": ["Uses simple analogies", "Avoids jargon", "Is engaging"],
        "eval_type": "rubric_based",
    },
    {
        "input": "Write a function to sort a list",
        "test_fn": lambda code: run_tests(code, sort_test_suite),
        "eval_type": "functional",
    },
]
```

**Step 3: Automate evaluation**

```python
def evaluate_response(test_case, response):
    if test_case["eval_type"] == "exact_match":
        return response.strip().lower() == test_case["expected_output"].lower()

    elif test_case["eval_type"] == "rubric_based":
        # Use LLM-as-judge
        judge_prompt = f"""
        Evaluate this response against the rubric.
        Response: {response}
        Rubric: {test_case['rubric']}
        Score each criterion 1-5 and explain.
        """
        return llm_judge(judge_prompt)

    elif test_case["eval_type"] == "functional":
        return test_case["test_fn"](response)
```

### LLM-as-Judge

Using a stronger model to evaluate a weaker model's outputs:

```python
JUDGE_PROMPT = """
You are evaluating an AI assistant's response.

Question: {question}
Response: {response}
Reference (if available): {reference}

Rate the response on these dimensions (1-5):
1. Helpfulness: Does it address the user's need?
2. Accuracy: Is the information correct?
3. Clarity: Is it well-organized and easy to understand?
4. Safety: Is it free from harmful content?

Provide scores and brief justification for each.
"""

def llm_judge(question, response, reference=None, judge_model="gpt-4"):
    prompt = JUDGE_PROMPT.format(
        question=question,
        response=response,
        reference=reference or "Not provided"
    )
    return judge_model.generate(prompt)
```

**LLM-as-judge limitations**:
- Position bias (prefers first option in A/B comparisons)
- Verbosity bias (prefers longer responses)
- Self-preference (models prefer their own outputs)
- Mitigate by randomizing position, using multiple judges, calibrating

### Evaluation Pipeline

```python
class EvalPipeline:
    def __init__(self, test_suite, metrics):
        self.test_suite = test_suite
        self.metrics = metrics

    def run(self, model):
        results = []
        for test in self.test_suite:
            response = model.generate(test["input"])
            scores = {}
            for metric in self.metrics:
                scores[metric.name] = metric.evaluate(test, response)
            results.append({
                "test": test,
                "response": response,
                "scores": scores,
            })
        return self.aggregate(results)

    def aggregate(self, results):
        # Compute summary statistics
        return {
            metric: {
                "mean": np.mean([r["scores"][metric] for r in results]),
                "std": np.std([r["scores"][metric] for r in results]),
                "pass_rate": np.mean([r["scores"][metric] >= threshold for r in results]),
            }
            for metric in self.metrics
        }
```

---

## Pattern 2: RAG (Retrieval-Augmented Generation)

**The problem**: LLMs have knowledge cutoffs and can hallucinate.

### Basic RAG Pipeline

```
User Query → Embed → Vector Search → Retrieve Docs → Augment Prompt → LLM → Response
```

```python
class RAGPipeline:
    def __init__(self, embedder, vector_store, llm):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def answer(self, query, top_k=5):
        # 1. Embed the query
        query_embedding = self.embedder.embed(query)

        # 2. Retrieve relevant documents
        docs = self.vector_store.search(query_embedding, top_k=top_k)

        # 3. Construct augmented prompt
        context = "\n\n".join([doc.content for doc in docs])
        prompt = f"""
        Answer the question based on the following context.
        If the answer isn't in the context, say "I don't know."

        Context:
        {context}

        Question: {query}

        Answer:
        """

        # 4. Generate response
        return self.llm.generate(prompt)
```

### RAG Optimizations

| Technique | Description | When to Use |
|-----------|-------------|-------------|
| Chunking strategy | How to split documents | All RAG systems |
| Hybrid search | Combine keyword + vector | When exact terms matter |
| Re-ranking | Re-score retrieved docs | Large retrieval sets |
| Query expansion | Generate multiple queries | Ambiguous queries |
| HyDE | Hypothetical document embedding | Low-quality queries |

### Advanced: Chunking Strategies

```python
# Bad: Fixed-size chunks (may split sentences, lose context)
def fixed_chunk(text, size=512):
    return [text[i:i+size] for i in range(0, len(text), size)]

# Better: Sentence-aware chunks
def sentence_chunk(text, max_tokens=512):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks

# Best: Semantic chunks (group by topic)
def semantic_chunk(text, embedder, threshold=0.8):
    sentences = nltk.sent_tokenize(text)
    embeddings = [embedder.embed(s) for s in sentences]

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        if similarity < threshold:
            # Topic shift, start new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])

    chunks.append(" ".join(current_chunk))
    return chunks
```

### Advanced: Re-ranking

```python
def rag_with_reranking(query, retriever, reranker, llm, initial_k=20, final_k=5):
    # 1. Initial retrieval (get more candidates)
    candidates = retriever.search(query, top_k=initial_k)

    # 2. Re-rank with cross-encoder
    scored = []
    for doc in candidates:
        score = reranker.score(query, doc.content)
        scored.append((score, doc))

    # 3. Take top after re-ranking
    top_docs = sorted(scored, key=lambda x: -x[0])[:final_k]

    # 4. Generate with top docs
    context = "\n\n".join([doc.content for _, doc in top_docs])
    return llm.generate(augmented_prompt(query, context))
```

### RAG Evaluation

| Metric | What It Measures | How to Compute |
|--------|------------------|----------------|
| Retrieval precision | Are retrieved docs relevant? | Human annotation or LLM-judge |
| Retrieval recall | Are all relevant docs retrieved? | Requires ground truth |
| Answer accuracy | Is the final answer correct? | Compare to gold answers |
| Faithfulness | Is answer grounded in context? | Check if claims are supported |
| Answer relevance | Does answer address the query? | LLM-judge or human eval |

---

## Pattern 3: Fine-tuning Decision

**The question**: When should you fine-tune vs. use RAG vs. prompt engineering?

### Decision Framework

```
Start with prompting
         │
         ▼
   Good enough? ──Yes──▶ Done
         │
         No
         ▼
   Need specific ──Yes──▶ Fine-tune for style/behavior
   style/format?
         │
         No
         ▼
   Need up-to-date ──Yes──▶ RAG
   knowledge?
         │
         No
         ▼
   Need domain ──Yes──▶ Fine-tune (or continued pre-training)
   expertise?
         │
         No
         ▼
   Better prompting / Few-shot examples
```

### When to Fine-tune

| Situation | Fine-tune? | Why |
|-----------|------------|-----|
| Custom output format | Yes | Consistent formatting across calls |
| Brand voice/tone | Yes | Hard to maintain via prompting |
| Specialized domain | Maybe | Consider continued pre-training first |
| Real-time knowledge | No | Use RAG instead |
| Few examples available | No | Need 100+ examples minimum |
| Behavior already in base model | No | Prompt engineering sufficient |

### Fine-tuning Best Practices

```python
# 1. Start with quality data, not quantity
# 100 high-quality examples > 10,000 noisy examples

# 2. Use consistent formatting
training_example = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I don't have access to real-time weather data. Please check a weather service like Weather.com for current conditions in your area."}
    ]
}

# 3. Include negative examples (what NOT to do)
negative_example = {
    "messages": [
        {"role": "user", "content": "Can you help me hack into..."},
        {"role": "assistant", "content": "I can't help with unauthorized access to systems. If you're interested in cybersecurity, I'd recommend exploring ethical hacking courses and certifications."}
    ]
}

# 4. Evaluate before and after
pre_ft_scores = evaluate(base_model, test_suite)
post_ft_scores = evaluate(fine_tuned_model, test_suite)
assert post_ft_scores > pre_ft_scores  # Verify improvement
```

---

## Pattern 4: Caching

**The problem**: LLM calls are slow and expensive.

### Caching Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| Exact match | Cache identical prompts | Repeated queries |
| Semantic cache | Cache similar prompts | FAQ-style queries |
| KV cache | Cache key-value states | Long conversations |
| Prompt prefix cache | Cache system prompt | Same system prompt |

### Exact Match Caching

```python
import hashlib
from functools import lru_cache

class LLMCache:
    def __init__(self, llm, redis_client):
        self.llm = llm
        self.redis = redis_client
        self.ttl = 3600  # 1 hour

    def generate(self, prompt, **kwargs):
        # Create cache key from prompt + params
        cache_key = self._make_key(prompt, kwargs)

        # Check cache
        cached = self.redis.get(cache_key)
        if cached:
            return json.loads(cached)

        # Generate and cache
        response = self.llm.generate(prompt, **kwargs)
        self.redis.setex(cache_key, self.ttl, json.dumps(response))
        return response

    def _make_key(self, prompt, kwargs):
        content = json.dumps({"prompt": prompt, **kwargs}, sort_keys=True)
        return f"llm:{hashlib.sha256(content.encode()).hexdigest()}"
```

### Semantic Caching

```python
class SemanticCache:
    def __init__(self, llm, embedder, vector_store, similarity_threshold=0.95):
        self.llm = llm
        self.embedder = embedder
        self.vector_store = vector_store
        self.threshold = similarity_threshold

    def generate(self, prompt, **kwargs):
        # Embed the query
        query_embedding = self.embedder.embed(prompt)

        # Search for similar cached prompts
        results = self.vector_store.search(query_embedding, top_k=1)

        if results and results[0].score >= self.threshold:
            # Cache hit
            return results[0].metadata["response"]

        # Cache miss - generate and store
        response = self.llm.generate(prompt, **kwargs)

        self.vector_store.insert(
            embedding=query_embedding,
            metadata={"prompt": prompt, "response": response, **kwargs}
        )

        return response
```

### Cache Invalidation

```python
class CacheWithInvalidation:
    def __init__(self, llm, cache):
        self.llm = llm
        self.cache = cache
        self.version = "v1"  # Bump to invalidate all

    def generate(self, prompt, **kwargs):
        key = self._make_key(prompt, kwargs)
        return self.cache.get_or_set(
            key,
            lambda: self.llm.generate(prompt, **kwargs)
        )

    def invalidate_all(self):
        """Bump version to invalidate entire cache"""
        self.version = f"v{int(self.version[1:]) + 1}"

    def invalidate_pattern(self, pattern):
        """Invalidate specific cache entries"""
        self.cache.delete_pattern(f"{self.version}:{pattern}*")

    def _make_key(self, prompt, kwargs):
        return f"{self.version}:{hash(prompt + str(kwargs))}"
```

---

## Pattern 5: Guardrails

**The problem**: LLM outputs can be harmful, incorrect, or off-topic.

### Input Guardrails

```python
class InputGuardrails:
    def __init__(self, llm, safety_classifier, topic_classifier):
        self.llm = llm
        self.safety = safety_classifier
        self.topic = topic_classifier

    def check(self, user_input):
        issues = []

        # Check for harmful intent
        safety_score = self.safety.classify(user_input)
        if safety_score["harmful"] > 0.8:
            issues.append({
                "type": "safety",
                "reason": safety_score["category"],
                "block": True
            })

        # Check for off-topic
        topic_score = self.topic.classify(user_input)
        if topic_score["on_topic"] < 0.3:
            issues.append({
                "type": "off_topic",
                "reason": f"Query seems to be about {topic_score['detected_topic']}",
                "block": False  # Warn but don't block
            })

        # Check for PII
        pii = self.detect_pii(user_input)
        if pii:
            issues.append({
                "type": "pii",
                "reason": f"Detected: {pii}",
                "block": True
            })

        return issues

    def detect_pii(self, text):
        # Regex patterns for common PII
        patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        }
        found = []
        for pii_type, pattern in patterns.items():
            if re.search(pattern, text):
                found.append(pii_type)
        return found
```

### Output Guardrails

```python
class OutputGuardrails:
    def __init__(self, fact_checker=None, toxicity_checker=None):
        self.fact_checker = fact_checker
        self.toxicity = toxicity_checker

    def check(self, response, context=None):
        issues = []

        # Check for hallucinations (if context provided)
        if context and self.fact_checker:
            facts = self.extract_claims(response)
            for fact in facts:
                if not self.fact_checker.verify(fact, context):
                    issues.append({
                        "type": "hallucination",
                        "claim": fact,
                        "severity": "high"
                    })

        # Check for toxic content
        if self.toxicity:
            toxicity_score = self.toxicity.score(response)
            if toxicity_score > 0.7:
                issues.append({
                    "type": "toxic",
                    "score": toxicity_score,
                    "severity": "high"
                })

        # Check for code execution attempts
        if self.contains_executable(response):
            issues.append({
                "type": "code_injection",
                "severity": "critical"
            })

        return issues

    def extract_claims(self, text):
        """Extract factual claims from text for verification"""
        # Use NLP or LLM to extract claims
        pass

    def contains_executable(self, text):
        """Check for potentially harmful code patterns"""
        dangerous_patterns = [
            r"<script>",
            r"eval\(",
            r"exec\(",
            r"subprocess",
            r"os\.system",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in dangerous_patterns)
```

### Full Guardrail Pipeline

```python
class GuardedLLM:
    def __init__(self, llm, input_guards, output_guards):
        self.llm = llm
        self.input_guards = input_guards
        self.output_guards = output_guards
        self.max_retries = 3

    def generate(self, prompt, context=None):
        # 1. Check input
        input_issues = self.input_guards.check(prompt)
        blocking = [i for i in input_issues if i.get("block")]
        if blocking:
            return {
                "response": "I can't help with that request.",
                "blocked": True,
                "reason": blocking[0]["reason"]
            }

        # 2. Generate with retries for output issues
        for attempt in range(self.max_retries):
            response = self.llm.generate(prompt)

            output_issues = self.output_guards.check(response, context)
            if not output_issues:
                return {"response": response, "blocked": False}

            # Retry with guardrail feedback
            prompt = self._add_guardrail_feedback(prompt, output_issues)

        # All retries failed
        return {
            "response": "I wasn't able to generate a safe response.",
            "blocked": True,
            "reason": "output_guardrails"
        }

    def _add_guardrail_feedback(self, prompt, issues):
        feedback = "\n".join([f"- Avoid: {i['type']}" for i in issues])
        return f"{prompt}\n\nIMPORTANT: {feedback}"
```

---

## Pattern 6: Defensive UX

**The problem**: Users have unrealistic expectations of LLMs.

### Setting Expectations

```python
class ConversationManager:
    def __init__(self, llm):
        self.llm = llm
        self.capabilities = {
            "can_do": [
                "Answer questions about general topics",
                "Help with writing and editing",
                "Explain concepts",
                "Generate code snippets",
            ],
            "cannot_do": [
                "Access the internet in real-time",
                "Remember previous conversations",
                "Execute code",
                "Access your files",
            ]
        }

    def welcome_message(self):
        return f"""
        Hi! I'm an AI assistant. Here's what I can help with:
        {chr(10).join('✓ ' + c for c in self.capabilities['can_do'])}

        Things I can't do:
        {chr(10).join('✗ ' + c for c in self.capabilities['cannot_do'])}

        How can I help you today?
        """
```

### Graceful Degradation

```python
class ResilientLLM:
    def __init__(self, primary_llm, fallback_llm, timeout=30):
        self.primary = primary_llm
        self.fallback = fallback_llm
        self.timeout = timeout

    async def generate(self, prompt):
        try:
            # Try primary with timeout
            response = await asyncio.wait_for(
                self.primary.generate(prompt),
                timeout=self.timeout
            )
            return {"response": response, "source": "primary"}

        except asyncio.TimeoutError:
            # Primary timed out, use fallback
            response = await self.fallback.generate(prompt)
            return {
                "response": response,
                "source": "fallback",
                "warning": "Response may be less detailed due to high load."
            }

        except Exception as e:
            # Complete failure
            return {
                "response": "I'm having trouble right now. Please try again.",
                "source": "error",
                "error": str(e)
            }
```

### Streaming with Progress

```python
class StreamingUI:
    def __init__(self, llm):
        self.llm = llm

    async def generate_with_progress(self, prompt):
        yield {"type": "status", "message": "Thinking..."}

        buffer = ""
        async for token in self.llm.stream(prompt):
            buffer += token
            yield {"type": "token", "content": token}

            # Yield progress updates for long responses
            if len(buffer) % 500 == 0:
                yield {"type": "status", "message": f"Generated {len(buffer)} characters..."}

        yield {"type": "complete", "full_response": buffer}
```

### Uncertainty Communication

```python
def add_confidence_markers(response, confidence_scores):
    """Add visual markers for uncertain claims"""
    result = response

    for claim, confidence in confidence_scores.items():
        if confidence < 0.5:
            # Low confidence - mark as uncertain
            result = result.replace(
                claim,
                f"⚠️ {claim} (I'm not certain about this)"
            )
        elif confidence < 0.8:
            # Medium confidence - soften language
            result = result.replace(
                claim,
                f"{claim} (though I'd recommend verifying)"
            )

    return result
```

---

## Pattern 7: Feedback Loops

**The problem**: How do you improve your LLM system over time?

### Collecting Feedback

```python
class FeedbackCollector:
    def __init__(self, storage):
        self.storage = storage

    def log_interaction(self, request_id, prompt, response, metadata=None):
        """Log every interaction for analysis"""
        self.storage.insert({
            "request_id": request_id,
            "timestamp": datetime.now(),
            "prompt": prompt,
            "response": response,
            "metadata": metadata or {},
            "feedback": None  # To be filled later
        })

    def record_feedback(self, request_id, feedback_type, feedback_data):
        """Record user feedback on a response"""
        self.storage.update(request_id, {
            "feedback": {
                "type": feedback_type,  # "thumbs_up", "thumbs_down", "correction"
                "data": feedback_data,
                "timestamp": datetime.now()
            }
        })

    def record_implicit_feedback(self, request_id, signal_type, signal_data):
        """Record implicit signals (copies, regenerates, abandons)"""
        self.storage.append_signal(request_id, {
            "signal": signal_type,  # "copied", "regenerated", "abandoned"
            "data": signal_data,
            "timestamp": datetime.now()
        })
```

### Feedback Analysis

```python
class FeedbackAnalyzer:
    def __init__(self, storage):
        self.storage = storage

    def get_failure_patterns(self, time_window="7d"):
        """Identify common failure patterns"""
        negative_feedback = self.storage.query({
            "feedback.type": "thumbs_down",
            "timestamp": {"$gte": time_window}
        })

        # Cluster by topic/intent
        clusters = self.cluster_by_topic(negative_feedback)

        return [
            {
                "pattern": cluster.centroid,
                "count": len(cluster.members),
                "examples": cluster.members[:5],
                "suggested_fix": self.suggest_fix(cluster)
            }
            for cluster in clusters
        ]

    def calculate_metrics(self, time_window="24h"):
        """Calculate key metrics"""
        interactions = self.storage.query({"timestamp": {"$gte": time_window}})

        return {
            "total_requests": len(interactions),
            "satisfaction_rate": self.calc_satisfaction(interactions),
            "regeneration_rate": self.calc_regeneration(interactions),
            "latency_p50": self.calc_latency_percentile(interactions, 50),
            "latency_p99": self.calc_latency_percentile(interactions, 99),
        }

    def suggest_fix(self, failure_cluster):
        """Suggest improvements based on failure pattern"""
        if failure_cluster.topic == "factual_error":
            return "Consider adding RAG for this topic"
        elif failure_cluster.topic == "format_error":
            return "Add examples to few-shot prompt"
        elif failure_cluster.topic == "refusal":
            return "Review safety thresholds"
        else:
            return "Investigate manually"
```

### Closing the Loop

```python
class ContinuousImprovement:
    def __init__(self, feedback_analyzer, model_trainer, deployer):
        self.analyzer = feedback_analyzer
        self.trainer = model_trainer
        self.deployer = deployer

    def weekly_improvement_cycle(self):
        # 1. Analyze feedback
        failures = self.analyzer.get_failure_patterns("7d")
        metrics = self.analyzer.calculate_metrics("7d")

        # 2. Generate improvement plan
        improvements = []
        for failure in failures:
            if failure["count"] > 100:  # Significant issue
                improvements.append({
                    "issue": failure["pattern"],
                    "fix": failure["suggested_fix"],
                    "priority": "high" if failure["count"] > 500 else "medium"
                })

        # 3. Create training data from corrections
        corrections = self.analyzer.get_user_corrections("7d")
        if len(corrections) > 50:
            training_data = self.format_as_training_data(corrections)
            improvements.append({
                "issue": "User corrections",
                "fix": f"Fine-tune on {len(training_data)} examples",
                "priority": "medium"
            })

        # 4. Implement high-priority fixes
        for improvement in improvements:
            if improvement["priority"] == "high":
                self.implement_fix(improvement)

        return {
            "metrics": metrics,
            "failures_identified": len(failures),
            "improvements_made": len([i for i in improvements if i["priority"] == "high"])
        }

    def implement_fix(self, improvement):
        if "RAG" in improvement["fix"]:
            self.add_rag_source(improvement["issue"])
        elif "few-shot" in improvement["fix"]:
            self.update_prompt(improvement["issue"])
        elif "Fine-tune" in improvement["fix"]:
            self.trainer.train(improvement["training_data"])
            self.deployer.deploy_new_model()
```

---

## Monitoring Dashboard

### Key Metrics to Track

| Category | Metric | Target | Alert Threshold |
|----------|--------|--------|-----------------|
| Quality | Satisfaction rate | >90% | <85% |
| Quality | Regeneration rate | <10% | >15% |
| Performance | Latency P50 | <500ms | >1s |
| Performance | Latency P99 | <2s | >5s |
| Cost | Cost per request | <$0.01 | >$0.02 |
| Safety | Blocked rate | <1% | >5% |
| Safety | Hallucination rate | <5% | >10% |

### Example Dashboard Query

```sql
-- Daily quality metrics
SELECT
    DATE(timestamp) as day,
    COUNT(*) as requests,
    AVG(CASE WHEN feedback_type = 'thumbs_up' THEN 1 ELSE 0 END) as satisfaction,
    AVG(CASE WHEN signals ? 'regenerated' THEN 1 ELSE 0 END) as regeneration_rate,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms) as p50_latency,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99_latency,
    SUM(cost_usd) as total_cost
FROM llm_interactions
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY day DESC;
```

---

## Summary

The seven patterns for production LLM systems:

1. **Evals**: Measure everything, use LLM-as-judge at scale
2. **RAG**: Ground responses in retrieved knowledge
3. **Fine-tuning**: For style/behavior, not knowledge
4. **Caching**: Reduce cost and latency
5. **Guardrails**: Prevent harm on input and output
6. **Defensive UX**: Set expectations, handle failures gracefully
7. **Feedback Loops**: Continuously improve from user signals

The key insight: **Building LLM applications is more about engineering around the model than improving the model itself.**
