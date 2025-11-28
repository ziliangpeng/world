# Prime Intellect RL Infrastructure Stack

Prime Intellect's open-source infrastructure for large-scale reinforcement learning training of LLMs. This stack powers INTELLECT-3 and enables any organization to train models using async RL at scale with verifiable rewards.

---

## Overview

The Prime Intellect RL Stack consists of four integrated components designed to solve the unique challenges of large-scale RL training:

| Component | Purpose | Key Innovation |
|-----------|---------|----------------|
| **[prime-rl](#prime-rl-framework)** | Async RL training framework | Disaggregated trainer/inference/orchestrator for 1000+ GPU scale |
| **[Prime Sandboxes](#prime-sandboxes)** | High-performance code execution | Millisecond latency, bypass Kubernetes control plane |
| **[Verifiers Library](#verifiers-library)** | RL environment toolkit | Modular components for building custom environments |
| **[Environments Hub](#environments-hub)** | Community task platform | 500+ tasks across math, code, science, logic, research, agentic |

### Use Cases

**Who should use this stack:**
- Research labs training models with reinforcement learning
- Organizations building coding or reasoning models
- Teams needing verifiable reward signals (math, code, science)
- Anyone wanting to scale RL beyond small clusters

**Real-world usage:**
- **INTELLECT-3**: 106B MoE model, 512 H200 GPUs, 500+ environments, 98.1% on MATH-500
- **INTELLECT-2**: 32B model, decentralized RL training
- Community projects leveraging Environments Hub

---

## PRIME-RL Framework

PRIME-RL is Prime Intellect's framework for asynchronous reinforcement learning at scale, designed to train models on 1000+ GPUs with efficient orchestration.

### Why Async RL at Scale?

Traditional RL training has bottlenecks:

| Challenge | Synchronous Approach | PRIME-RL (Async) |
|-----------|---------------------|------------------|
| **Inference bottleneck** | Training waits for rollouts | Disaggregated: training continues |
| **GPU utilization** | Idle during rollout generation | Parallel: trainer and inference separate |
| **Scaling** | Limited by slowest node | Async: nodes progress independently |
| **Long-horizon tasks** | Timeouts block training | Off-policy: stale rollouts OK |

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                     PRIME-RL Architecture                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                   FSDP2 Trainer Nodes                          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │  │
│  │  │ Trainer  │  │ Trainer  │  │ Trainer  │  │ Trainer  │      │  │
│  │  │    1     │  │    2     │  │    3     │  │    N     │      │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │  │
│  └───────┼─────────────┼─────────────┼─────────────┼─────────────┘  │
│          │             │             │             │                 │
│          └─────────────┴─────────────┴─────────────┘                 │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    Orchestrator (CPU)                          │  │
│  │                                                                │  │
│  │  • Collects rollouts from inference service                   │  │
│  │  • Assembles packed batches for training                      │  │
│  │  • Relays updated weights to inference                        │  │
│  │  • Schedules verifier execution                               │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │               vLLM Inference Service                           │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │  │
│  │  │Inference │  │Inference │  │Inference │  │Inference │      │  │
│  │  │  Node 1  │  │  Node 2  │  │  Node 3  │  │  Node M  │      │  │
│  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │  │
│  └───────┼─────────────┼─────────────┼─────────────┼─────────────┘  │
│          │             │             │             │                 │
│          └─────────────┴──────┬──────┴─────────────┘                 │
│                               │                                       │
│                               ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │         Verifiers Environments (Environments Hub)              │  │
│  │                                                                │  │
│  │  • Multi-turn rollout generation                              │  │
│  │  • Scoring via Prime Sandboxes                                │  │
│  │  • 500+ tasks (math, code, science, logic, research)          │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### Disaggregated Training Design

**Three core components**:

#### 1. FSDP2 Trainer
```python
# Conceptual trainer architecture
class FSDP2Trainer:
    """
    Fully Sharded Data Parallel v2 trainer.

    Handles model updates from batches of rollouts.
    """

    def __init__(self, model, optimizer):
        self.model = shard_model_fsdp2(model)  # Shard across trainer nodes
        self.optimizer = optimizer

    def train_step(self, rollout_batch):
        """Process batch of rollouts from orchestrator."""
        # Unpack batch (prompt, response, reward)
        loss = compute_rl_loss(self.model, rollout_batch)

        # Standard training update
        loss.backward()
        self.optimizer.step()

        # Send updated weights to orchestrator
        return get_model_weights(self.model)
```

#### 2. vLLM Inference Service
```python
# Conceptual inference service
class vLLMInference:
    """
    Disaggregated inference for rollout generation.

    Runs independently from training, avoiding bottlenecks.
    """

    def __init__(self, model_weights):
        self.engine = vLLM(model_weights)

    def generate_rollouts(self, prompts, environments):
        """Generate rollouts for batch of prompts."""
        rollouts = []

        for prompt, env in zip(prompts, environments):
            # Multi-turn interaction with environment
            response = self.engine.generate(prompt)
            reward = env.verify(response)  # Execute in Prime Sandbox

            rollouts.append({
                'prompt': prompt,
                'response': response,
                'reward': reward
            })

        return rollouts

    def update_weights(self, new_weights):
        """Async update from trainer."""
        self.engine.load_weights(new_weights)
```

#### 3. Orchestrator
```python
# Conceptual orchestrator
class Orchestrator:
    """
    Lightweight CPU process coordinating training loop.

    Handles data flow between trainer, inference, and environments.
    """

    def __init__(self, trainer, inference, environments):
        self.trainer = trainer
        self.inference = inference
        self.environments = environments
        self.rollout_buffer = []

    def training_loop(self):
        while True:
            # Async: Request rollouts from inference (non-blocking)
            prompts = sample_prompts(self.environments)
            self.inference.request_rollouts(prompts)

            # Collect completed rollouts (may be from earlier requests)
            new_rollouts = self.inference.get_completed_rollouts()
            self.rollout_buffer.extend(new_rollouts)

            # Train when buffer is full
            if len(self.rollout_buffer) >= BATCH_SIZE:
                batch = assemble_packed_batch(self.rollout_buffer)
                new_weights = self.trainer.train_step(batch)

                # Async: Send weights to inference (non-blocking)
                self.inference.update_weights(new_weights)

                self.rollout_buffer = []
```

### Async-Only Methodology

**Off-policy advantages**:
- Rollouts don't need to be from latest policy
- Stale rollouts still provide learning signal
- No synchronization barriers
- Long-horizon tasks don't block training

**Key insight**: For tasks like math and coding with verifiable outcomes, off-policy learning is stable even with delayed rollouts.

---

## Prime Sandboxes

Prime Sandboxes is Prime Intellect's high-performance execution layer for running untrusted code during RL training.

### The Execution Latency Problem

Traditional approaches to code execution in RL:

```
Problem:
┌─────────────────────────────────────────────────────────────┐
│  Traditional Container Orchestration (Kubernetes)           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Request → K8s API → Scheduler → Pod Creation → Execution  │
│             ▲                                               │
│             └── Latency: SECONDS (thousands of milliseconds)│
│                                                             │
│  For RL training:                                          │
│  • Thousands of concurrent code executions needed          │
│  • Each execution should be milliseconds                   │
│  • Orchestration overhead dominates                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Impact**: RL training became bottlenecked on waiting for code verification, not on model training.

### High-Performance Design

Prime Sandboxes bypasses the Kubernetes control plane with a direct execution path:

```
Solution:
┌─────────────────────────────────────────────────────────────┐
│  Prime Sandboxes Architecture                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Request → Rust Executor → Direct Pod Communication        │
│             ▲                                               │
│             └── Latency: MILLISECONDS                       │
│                                                             │
│  Design principles:                                         │
│  • Direct Rust-to-pod execution path                       │
│  • Bypass Kubernetes API server                            │
│  • Pre-warmed sandbox pools                                │
│  • Native async Rust runtime                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Performance Characteristics

| Metric | Traditional (K8s) | Prime Sandboxes |
|--------|------------------|-----------------|
| **Provisioning** | 10-60 seconds | **<10 seconds** |
| **Execution latency** | Seconds | **Milliseconds** |
| **Concurrent sandboxes/node** | 10-20 | **Hundreds** |
| **Total concurrent rollouts** | Hundreds | **Thousands** |

### Scale Capabilities

```python
# Conceptual Prime Sandboxes usage
class PrimeSandbox:
    """
    High-performance sandbox for code execution.

    Sub-second provisioning, millisecond execution.
    """

    def __init__(self):
        # Pre-warm sandbox pool
        self.sandbox_pool = initialize_sandbox_pool()

    def execute_code(self, code, test_cases, timeout_ms=1000):
        """
        Execute code with test cases.

        Returns results in milliseconds.
        """
        # Get sandbox from pool (no K8s overhead)
        sandbox = self.sandbox_pool.get()

        try:
            results = []
            for test in test_cases:
                # Direct execution (Rust → pod)
                start = time_milliseconds()
                output = sandbox.run(code, test.input, timeout=timeout_ms)
                latency = time_milliseconds() - start

                results.append({
                    'output': output,
                    'expected': test.expected,
                    'passed': output == test.expected,
                    'latency_ms': latency  # Typically <10ms
                })

            return results
        finally:
            # Return to pool (reuse)
            self.sandbox_pool.release(sandbox)
```

**Real-world impact**: INTELLECT-3 training generated and verified thousands of concurrent code solutions with minimal latency overhead, enabling massive-scale RL on coding tasks.

---

## Verifiers Library

**Purpose**: Modular toolkit for building RL environments and evaluations for LLMs.

**Features**:
- Abstraction for multi-turn rollout generation
- Modular verifier components
- CPU-based development/evaluation with API models
- GPU-based large-scale RL training with prime-rl
- Test execution via Prime Sandboxes

```python
# Conceptual verifiers library usage
from verifiers import Environment, MathVerifier, CodeVerifier

class CustomMathEnvironment(Environment):
    """
    Example math environment using verifiers.
    """

    def __init__(self, problems_dataset):
        self.problems = problems_dataset
        self.verifier = MathVerifier()  # Symbolic verification

    def sample_prompt(self):
        """Sample problem from dataset."""
        problem = random.choice(self.problems)
        return {
            'prompt': problem.question,
            'ground_truth': problem.answer
        }

    def verify(self, prompt, response):
        """
        Verify model response against ground truth.

        Returns reward (1.0 if correct, 0.0 otherwise).
        """
        predicted = self.verifier.extract_answer(response)
        correct = self.verifier.symbolic_match(
            predicted,
            prompt['ground_truth']
        )
        return 1.0 if correct else 0.0

class CustomCodeEnvironment(Environment):
    """
    Example code environment with test execution.
    """

    def __init__(self, coding_problems):
        self.problems = coding_problems
        self.sandbox = PrimeSandbox()  # High-performance execution

    def verify(self, prompt, response):
        """
        Execute code against test cases in sandbox.

        Returns reward (fraction of tests passed).
        """
        code = extract_code_from_response(response)
        results = self.sandbox.execute_code(
            code,
            prompt['test_cases'],
            timeout_ms=1000
        )

        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        return passed / total  # Reward: 0.0 to 1.0
```

---

## Environments Hub

**Platform**: Community hub for RL environments (https://huggingface.co/PrimeIntellect/environments)

**Features**:
- **500+ tasks** across math, code, science, logic, research, agentic domains
- Reproducible environment versioning
- Community contributions
- Direct integration with prime-rl

### Task Categories

| Category | Tasks | Example Environments |
|----------|-------|---------------------|
| **Math** | 150+ | Competition math, theorem proving, algebra |
| **Code** | 200+ | LeetCode, HumanEval, APPS, SWE-bench |
| **Science** | 50+ | Physics, chemistry, biology problems |
| **Logic** | 40+ | Formal reasoning, puzzle solving |
| **Research** | 30+ | Literature synthesis, hypothesis generation |
| **Agentic** | 30+ | Tool use, multi-step planning, web navigation |

### Integration with prime-rl

```python
from verifiers.hub import load_environment

# Load any environment from Hub
math_env = load_environment("prime-intellect/competition-math")
code_env = load_environment("prime-intellect/humaneval-plus")

# Plug into prime-rl training loop
trainer.add_environments([math_env, code_env])
```

---

## End-to-End Integration Example

Here's how all components work together in a complete RL training workflow:

```python
from prime_rl import FSDP2Trainer, vLLMInference, Orchestrator
from verifiers.hub import load_environment
from prime_sandboxes import PrimeSandbox

# 1. Setup infrastructure
trainer = FSDP2Trainer(
    model=load_model("intellect-3"),
    optimizer="adamw",
    num_gpus=512
)

inference = vLLMInference(
    model_weights=trainer.get_weights(),
    num_nodes=32
)

sandboxes = PrimeSandbox(
    num_pools=10,
    sandboxes_per_pool=50  # 500 concurrent
)

# 2. Load environments from Hub
environments = [
    load_environment("prime-intellect/competition-math"),
    load_environment("prime-intellect/humaneval-plus"),
    load_environment("prime-intellect/aime-2024"),
    load_environment("prime-intellect/swe-bench"),
    # ... 500+ total environments
]

# Connect sandboxes to code environments
for env in environments:
    if env.requires_execution:
        env.set_sandbox(sandboxes)

# 3. Launch orchestrator
orchestrator = Orchestrator(
    trainer=trainer,
    inference=inference,
    environments=environments,
    batch_size=1024,
    async_mode=True
)

# 4. Start training
orchestrator.train(
    total_steps=100000,
    checkpoint_every=1000,
    log_every=100
)
```

**What happens during training:**
1. **Orchestrator** samples prompts from Environments Hub
2. **vLLM Inference** generates responses (asynchronously)
3. **Prime Sandboxes** execute code and verify outputs (milliseconds)
4. **Verifiers** compute rewards based on task success
5. **FSDP2 Trainer** updates model weights
6. **Orchestrator** relays updated weights to inference (non-blocking)
7. Cycle continues with no synchronization barriers

**Result**: Continuous async training at 512+ GPU scale with thousands of concurrent rollouts.

---

## Installation & Setup

### Requirements
- NVIDIA GPUs (H100 or H200 recommended for large models)
- PyTorch 2.0+
- CUDA 12.1+
- Python 3.10+

### Installing prime-rl
```bash
# Clone framework
git clone https://github.com/PrimeIntellect-ai/prime-rl
cd prime-rl

# Install dependencies
pip install -e .
```

### Installing Verifiers Library
```bash
# Clone library
git clone https://github.com/PrimeIntellect-ai/verifiers
cd verifiers

# Install dependencies
pip install -e .
```

### Installing Prime SDK (includes CLI)
```bash
pip install prime-sdk
```

### Quick Start: Training on a Single Environment

```python
from prime_rl import SimpleSFTTrainer  # Simpler API for getting started
from verifiers.hub import load_environment

# Load environment
math_env = load_environment("prime-intellect/gsm8k")

# Setup trainer (single GPU for quick start)
trainer = SimpleSFTTrainer(
    model="meta-llama/Llama-3-8B",
    environment=math_env,
    num_gpus=1
)

# Train
trainer.train(steps=1000)
```

---

## Comparison to Other RL Frameworks

| Framework | Sync/Async | Disaggregated | Scale | Code Execution |
|-----------|------------|---------------|-------|----------------|
| **OpenAI RL** | Unknown | Unknown | Unknown | Proprietary |
| **Ray RLlib** | Sync | No | 10s of GPUs | External tools |
| **TRL (HuggingFace)** | Sync | No | Single node | Not built-in |
| **PRIME-RL** | **Async** | **Yes** | **1000+ GPUs** | **Prime Sandboxes (millisecond)** |

**Key advantages:**
- **Async-only**: No synchronization barriers, handles long-horizon tasks
- **Disaggregated**: Trainer and inference run independently
- **Scalable**: Designed for 1000+ GPUs
- **Integrated execution**: Prime Sandboxes provide millisecond-latency code verification
- **Community environments**: 500+ tasks ready to use

---

## Real-World Usage

### INTELLECT-3 Training
- **Model**: 106B MoE (12B active)
- **Infrastructure**: 512 NVIDIA H200 GPUs across 64 nodes
- **Environments**: 500+ tasks from Environments Hub
- **Duration**: ~2 months (SFT + RL)
- **Performance**: MATH-500 98.1%, AIME24 90.8%, AIME25 88.0%

**Technical details**:
- FSDP2 for model sharding across trainer nodes
- vLLM inference service with 32 nodes
- Prime Sandboxes: thousands of concurrent code executions
- Verifiers: math (symbolic), code (test execution), science (domain-specific)

### INTELLECT-2 Training (Predecessor)
- **Model**: 32B parameters
- **Infrastructure**: Globally distributed, permissionless
- **Innovation**: Extended decentralized training to RL (not just pre-training)

For full INTELLECT series documentation, see:
- [INTELLECT-1 docs](../open-source-models/prime-intellect-1.md): Decentralized pre-training
- [INTELLECT-2 docs](../open-source-models/prime-intellect-2.md): Decentralized RL
- [INTELLECT-3 docs](../open-source-models/prime-intellect-3.md): Large-scale centralized RL

---

## Sources

### GitHub Repositories
- [prime-rl Framework](https://github.com/PrimeIntellect-ai/prime-rl) - Async RL training at scale
- [Verifiers Library](https://github.com/PrimeIntellect-ai/verifiers) - Environment toolkit
- [Prime SDK](https://github.com/PrimeIntellect-ai/prime) - Official CLI and Python SDK

### Documentation & Papers
- [INTELLECT-3 Technical Report](https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf) - Complete methodology
- [Environments Hub Blog Post](https://www.primeintellect.ai/blog/environments) - Environment ecosystem overview
- [INTELLECT-3 Release Announcement](https://www.primeintellect.ai/blog/intellect-3) - Official launch post

### Models & Datasets
- [INTELLECT-3 on HuggingFace](https://huggingface.co/PrimeIntellect/INTELLECT-3) - Model weights
- [Environments Hub](https://huggingface.co/PrimeIntellect/environments) - 500+ training tasks

### Community & Support
- [Prime Intellect Website](https://www.primeintellect.ai) - Official site
- [Prime Intellect Discord](https://discord.gg/primeintellect) - Community support
- [GitHub Issues](https://github.com/PrimeIntellect-ai/prime-rl/issues) - Bug reports and feature requests
