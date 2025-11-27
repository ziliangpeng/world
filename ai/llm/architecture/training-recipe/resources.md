# LLM Training Resources

Curated external resources for deeper learning on LLM training topics.

---

## Comprehensive Guides

### Books & Courses

| Resource | Author | Description |
|----------|--------|-------------|
| [Build a Large Language Model from Scratch](https://github.com/rasbt/LLMs-from-scratch) | Sebastian Raschka | Complete PyTorch implementation from architecture to fine-tuning |
| [LLM Course](https://github.com/mlabonne/llm-course) | Maxime Labonne | Roadmaps and Colab notebooks covering full LLM lifecycle |
| [Coding LLMs from the Ground Up](https://magazine.sebastianraschka.com/p/coding-llms-from-the-ground-up) | Sebastian Raschka | 3-hour workshop companion |

### Blog Series

| Resource | Author | Description |
|----------|--------|-------------|
| [Lil'Log](https://lilianweng.github.io/) | Lilian Weng (OpenAI) | Deep technical posts on ML/AI fundamentals |
| [Sebastian Raschka's Blog](https://sebastianraschka.com/blog/) | Sebastian Raschka | LLM training, research summaries |
| [Eugene Yan](https://eugeneyan.com/writing/) | Eugene Yan (Amazon) | Applied ML, LLM systems |
| [Chip Huyen](https://huyenchip.com/blog/) | Chip Huyen | MLOps, production systems |

---

## Pre-Training

| Resource | Focus |
|----------|-------|
| [New LLM Pre-training and Post-training Paradigms](https://magazine.sebastianraschka.com/p/new-llm-pre-training-and-post-training) | Analysis of Qwen 2, Apple AFM, Gemma 2, Llama 3.1 training pipelines |
| [Everything about Distributed Training](https://sumanthrh.com/post/distributed-and-efficient-finetuning/) | FSDP vs DeepSpeed ZeRO, 3D parallelism, efficient fine-tuning |
| [FSDP vs DeepSpeed](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed) | HuggingFace official comparison guide |

---

## Post-Training

| Resource | Focus |
|----------|-------|
| [RLHF Overview](https://huyenchip.com/2023/04/11/llm-engineering.html#rlhf) | Chip Huyen's breakdown of RLHF phases and costs |
| [LLM Course - RLHF Section](https://github.com/mlabonne/llm-course#5-rlhf) | Curated RLHF and preference tuning resources |

---

## Production & Deployment

| Resource | Focus |
|----------|-------|
| [What We Learned from a Year of Building with LLMs](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/) | Production lessons on prompting, RAG, fine-tuning, evals (Eugene Yan et al.) |
| [Patterns for Building LLM-based Systems](https://eugeneyan.com/writing/llm-patterns/) | 7 key patterns: Evals, RAG, Fine-tuning, Caching, Guardrails, UX, Feedback |
| [Building LLM Applications for Production](https://huyenchip.com/2023/04/11/llm-engineering.html) | Cost analysis, RLHF phases, LLMOps considerations |

---

## Emerging Topics

| Resource | Focus |
|----------|-------|
| [Why We Think](https://lilianweng.github.io/posts/2025-05-01-thinking/) | Test-time compute, chain-of-thought, RL for reasoning (o1/DeepSeek-R1 style) |
| [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) | Agent architectures, planning, memory systems |
| [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/) | Comprehensive prompt engineering techniques |

---

## Technical Deep Dives

### Attention & Architecture
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [Attention Is All You Need (annotated)](https://nlp.seas.harvard.edu/annotated-transformer/) - Harvard NLP

### Optimization
- [AdamW and Super-Convergence](https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html) - fast.ai
- [Gradient Accumulation](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation) - HuggingFace

### Distributed Training
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) - PyTorch
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) - Microsoft

---

## Model-Specific Technical Reports

| Model | Report |
|-------|--------|
| Llama 3 | [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) |
| Qwen 2 | [arXiv:2407.10671](https://arxiv.org/abs/2407.10671) |
| Gemma 2 | [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) |
| Mistral | [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) |
| DeepSeek-R1 | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
