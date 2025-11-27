# Distributed Training for LLMs

Training a 70B parameter model requires ~140GB of memory just for weights in fp16—far exceeding any single GPU. Modern LLM training distributes computation across hundreds or thousands of GPUs using a combination of parallelism strategies. This document traces the evolution from simple data parallelism to the sophisticated "3D parallelism" that enables training at scale.

---

## The Memory Problem

Before understanding distributed training, we must understand what consumes GPU memory:

### Memory Breakdown

For a model with N parameters trained with mixed precision:

| Component | Memory (bytes) | 7B Model | 70B Model |
|-----------|----------------|----------|-----------|
| Model weights (fp16) | 2N | 14 GB | 140 GB |
| Gradients (fp16) | 2N | 14 GB | 140 GB |
| Optimizer states (fp32) | 8N | 56 GB | 560 GB |
| Activations | Variable | 10-100 GB | 100-1000 GB |
| **Total (training)** | **12N+** | **~100 GB** | **~1000 GB** |

**The problem**: A single A100 (80GB) cannot even hold the optimizer states of a 70B model.

**The solution**: Distribute memory and computation across multiple GPUs.

---

## Historical Evolution

### Phase 1: Data Parallelism (2012-2018)

**Data Parallelism (DP)** - The simplest strategy

Each GPU holds a full copy of the model and processes different data batches:

```
                    ┌─────────────────────────────────────┐
                    │         Parameter Server            │
                    │    (aggregates gradients)           │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
    ┌──────▼──────┐         ┌──────▼──────┐         ┌──────▼──────┐
    │   GPU 0     │         │   GPU 1     │         │   GPU 2     │
    │  Full Model │         │  Full Model │         │  Full Model │
    │  Batch 0    │         │  Batch 1    │         │  Batch 2    │
    └─────────────┘         └─────────────┘         └─────────────┘
```

**Algorithm**:
1. Each GPU computes forward/backward on its local batch
2. All gradients sent to parameter server
3. Server averages gradients, updates weights
4. Updated weights broadcast to all GPUs

**Problem**: Parameter server becomes bottleneck; communication overhead scales poorly.

### Phase 2: All-Reduce Data Parallelism (2018-2020)

**Distributed Data Parallel (DDP)** - PyTorch's solution

Eliminated parameter server using ring all-reduce:

```
Forward:  Each GPU computes independently
Backward: Overlap gradient computation with all-reduce
Update:   Each GPU updates its own copy
```

**Ring All-Reduce**:

```
GPU 0 ──grad₀──▶ GPU 1 ──grad₁──▶ GPU 2 ──grad₂──▶ GPU 3 ──grad₃──▶ GPU 0
           ◀────────────────────────────────────────────────────────
```

- Each GPU sends to next, receives from previous
- After N-1 steps, all GPUs have sum of all gradients
- Communication: O(2N) per GPU, independent of GPU count

**PyTorch DDP**:
```python
# Simple DDP setup
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank
)

# Training loop unchanged
for batch in dataloader:
    loss = model(batch)
    loss.backward()  # Gradient sync happens here automatically
    optimizer.step()
```

**Limitation**: Each GPU still needs full model copy—doesn't scale to large models.

### Phase 3: Memory-Efficient Parallelism (2019-2021)

**[ZeRO (Zero Redundancy Optimizer)](https://arxiv.org/abs/1910.02054)** - Microsoft DeepSpeed

The insight: In DDP, each GPU stores identical copies. ZeRO partitions this redundancy:

| ZeRO Stage | What's Partitioned | Memory Reduction |
|------------|-------------------|------------------|
| ZeRO-1 | Optimizer states | 4× |
| ZeRO-2 | + Gradients | 8× |
| ZeRO-3 | + Parameters | Linear with GPUs |

**ZeRO-1**: Partition optimizer states
```
Traditional DDP (8 GPUs):
Each GPU: 2N (weights) + 2N (grads) + 8N (opt) = 12N per GPU

ZeRO-1 (8 GPUs):
Each GPU: 2N (weights) + 2N (grads) + N (opt/8) = ~5N per GPU
```

**ZeRO-2**: Also partition gradients
```
Each GPU: 2N (weights) + N/8 (grads/8) + N (opt/8) = ~3N per GPU
```

**ZeRO-3**: Also partition weights
```
Each GPU: N/8 (weights) + N/8 (grads) + N (opt/8) = ~1.5N/8 per GPU
```

**Communication cost**: ZeRO-3 requires all-gather before forward/backward, adding 1.5× communication vs DDP.

**FSDP (Fully Sharded Data Parallel)** - PyTorch native

Facebook's implementation of ZeRO-3 concepts:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3 equivalent
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    ),
)
```

### Phase 4: Model Parallelism (2019-Present)

When a single layer is too large, partition the model itself:

**Tensor Parallelism (TP)** - Partition within layers

Split weight matrices across GPUs:

```
Standard Linear:  Y = XW          (W is [d_in × d_out])

Tensor Parallel:  Y = X[W₀|W₁]    (Each GPU has [d_in × d_out/2])
                  Y = concat(Y₀, Y₁)
```

For Transformer attention:
```
                    ┌──────────────────────────────────────┐
                    │           Input Sequence              │
                    └──────────────────┬───────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
             ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐
             │   Head 0-3   │    │  Head 4-7   │    │  Head 8-11  │
             │   (GPU 0)    │    │   (GPU 1)   │    │   (GPU 2)   │
             └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       │ All-Reduce
                    ┌──────────────────▼───────────────────┐
                    │           Output Sequence             │
                    └──────────────────────────────────────┘
```

**MLP tensor parallelism**:
```
Column parallel: Split FFN up-projection across GPUs
Row parallel:    Split FFN down-projection across GPUs
                 All-reduce after down-projection
```

**Trade-off**: Requires all-reduce after each layer—only efficient within a node (NVLink).

**Pipeline Parallelism (PP)** - Partition across layers

Assign different layers to different GPUs:

```
GPU 0: Layers 0-15   GPU 1: Layers 16-31   GPU 2: Layers 32-47   GPU 3: Layers 48-63
    │                    │                     │                     │
    └────────────────────▶────────────────────▶────────────────────▶
                    (activations flow between stages)
```

**Problem**: Naive PP has low utilization—GPU 0 idle while GPU 3 computes.

**GPipe**: Micro-batching to improve utilization
```
Batch = 4 micro-batches

Time →
GPU 0: [mb0] [mb1] [mb2] [mb3] ────── [mb0] [mb1] [mb2] [mb3]
GPU 1:       [mb0] [mb1] [mb2] [mb3] ────── [mb0] [mb1] [mb2] [mb3]
GPU 2:             [mb0] [mb1] [mb2] [mb3] ────── [mb0] [mb1] [mb2]
GPU 3:                   [mb0] [mb1] [mb2] [mb3] ────── [mb0] [mb1]
        ├──────── Forward ────────────────┤├───── Backward ─────┤
                                    "Pipeline bubble"
```

**1F1B (One Forward One Backward)**: Interleaved schedule reduces bubble
```
GPU 0: F0 F1 F2 F3 B0 B1 B2 B3
GPU 1:    F0 F1 F2 B0 B1 F3 B2 B3
GPU 2:       F0 F1 B0 F2 B1 B2 F3 B3
GPU 3:          F0 B0 F1 B1 F2 B2 F3 B3
```

### Phase 5: 3D Parallelism (2021-Present)

Modern large-scale training combines all three:

```
3D Parallelism: Data × Tensor × Pipeline

                    ┌─────────────────────────────────────────────┐
                    │              Data Parallel Replica 0         │
                    │  ┌─────────────────────────────────────────┐│
                    │  │         Pipeline Stage 0                ││
                    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐   ││
                    │  │  │TP GPU 0 │ │TP GPU 1 │ │TP GPU 2 │   ││
                    │  │  │Layer 0-7│ │Layer 0-7│ │Layer 0-7│   ││
                    │  │  └─────────┘ └─────────┘ └─────────┘   ││
                    │  └─────────────────────────────────────────┘│
                    │  ┌─────────────────────────────────────────┐│
                    │  │         Pipeline Stage 1                ││
                    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐   ││
                    │  │  │TP GPU 3 │ │TP GPU 4 │ │TP GPU 5 │   ││
                    │  │  │Layer 8-15│ │Layer 8-15│ │Layer 8-15│  ││
                    │  │  └─────────┘ └─────────┘ └─────────┘   ││
                    │  └─────────────────────────────────────────┘│
                    └─────────────────────────────────────────────┘

                    (Replicated for Data Parallel Replica 1, 2, ...)
```

**Typical configuration for 70B model on 128 GPUs**:
- TP=8 (within node, 8 GPUs per node)
- PP=2 (across 2 stages)
- DP=8 (8 data parallel replicas)
- Total: 8 × 2 × 8 = 128 GPUs

**Why this split?**
- TP within node: Requires high bandwidth (NVLink: 600+ GB/s)
- PP across nodes: Moderate bandwidth (InfiniBand: ~200 GB/s)
- DP across everything: Most communication-efficient

### Phase 6: Expert Parallelism for MoE (2022-Present)

[Mixture of Experts](../architectural-patterns/mixture-of-experts.md) adds a new dimension:

```
Expert Parallelism (EP): Distribute experts across GPUs

                    Token Routing
                         │
           ┌─────────────┼─────────────┐
           │             │             │
    ┌──────▼──────┐ ┌────▼────┐ ┌──────▼──────┐
    │   GPU 0     │ │  GPU 1  │ │   GPU 2     │
    │ Expert 0-3  │ │Expert 4-7│ │ Expert 8-11 │
    └─────────────┘ └─────────┘ └─────────────┘
                         │
                    All-to-All
                    (route tokens to experts)
```

**Communication pattern**: All-to-all (each GPU sends/receives from all others)

**Modern MoE training (Mixtral, DeepSeek-V3)**:
- TP for dense layers
- EP for expert layers
- PP across expert layers
- DP as outermost dimension

---

## Modern Frameworks

### DeepSpeed

Microsoft's framework, implements ZeRO and more:

```python
import deepspeed

# Config specifies parallelism
config = {
    "train_batch_size": 1024,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,  # ZeRO-3
        "offload_optimizer": {"device": "cpu"},  # Offload to CPU RAM
    }
}

model, optimizer, _, scheduler = deepspeed.initialize(
    model=model,
    config=config
)
```

**Key features**:
- ZeRO stages 1-3
- Offloading to CPU/NVMe
- 3D parallelism integration
- Sparse attention support

### Megatron-LM

NVIDIA's framework, pioneered tensor/pipeline parallelism:

```python
# Megatron parallel groups
mpu.initialize_model_parallel(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=2  # Interleaved PP
)

# Model automatically partitioned
model = GPTModel(
    config,
    num_layers=64,
    hidden_size=12288,
    num_attention_heads=96
)
```

**Key features**:
- Sequence parallelism (partition activations along sequence)
- Interleaved pipeline schedules
- Optimized CUDA kernels
- Context parallelism for long sequences

### Megatron-DeepSpeed

Combines both for maximum scalability:
- Megatron's tensor/pipeline parallelism
- DeepSpeed's ZeRO optimizer
- Used for GPT-NeoX-20B, BLOOM-176B

### PyTorch Native (2024+)

PyTorch now includes native distributed primitives:

```python
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module
)

# Native tensor parallelism
model = parallelize_module(
    model,
    device_mesh["tp"],
    {
        "attention.q_proj": ColwiseParallel(),
        "attention.k_proj": ColwiseParallel(),
        "attention.v_proj": ColwiseParallel(),
        "attention.o_proj": RowwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
    }
)
```

### Comparison

| Framework | Strengths | Best For |
|-----------|-----------|----------|
| **DeepSpeed** | ZeRO, offloading, ease of use | Training with memory constraints |
| **Megatron-LM** | Performance, TP/PP | Maximum throughput |
| **FSDP** | PyTorch native, simplicity | Research, moderate scale |
| **Megatron-DeepSpeed** | Combination | Largest models (100B+) |

---

## Communication Primitives

### Collective Operations

| Operation | What It Does | Use Case |
|-----------|--------------|----------|
| **All-Reduce** | Sum across all GPUs, result everywhere | Gradient synchronization |
| **All-Gather** | Gather from all, result everywhere | FSDP parameter gathering |
| **Reduce-Scatter** | Sum and scatter result | FSDP gradient reduction |
| **All-to-All** | Each sends different data to each | MoE expert routing |
| **Broadcast** | One sends to all | Weight initialization |

### Network Topology

**Within node** (8 GPUs):
- NVLink/NVSwitch: 600-900 GB/s bidirectional
- PCIe: ~32 GB/s per GPU (fallback)

**Across nodes**:
- InfiniBand HDR: 200 Gb/s (~25 GB/s)
- InfiniBand NDR: 400 Gb/s (~50 GB/s)
- RoCE: Similar to InfiniBand, cheaper

**Rule of thumb**: Keep high-bandwidth operations (TP) within node, lower-bandwidth (DP, PP) across nodes.

---

## Practical Configuration

### Memory Estimation

```python
def estimate_memory_per_gpu(
    n_params: int,
    n_gpus: int,
    tp_size: int,
    pp_size: int,
    zero_stage: int,
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    n_layers: int
) -> dict:
    """Estimate memory usage per GPU."""
    dp_size = n_gpus // (tp_size * pp_size)

    # Model memory (fp16)
    model_mem = 2 * n_params / tp_size / pp_size

    # Gradient memory
    if zero_stage >= 2:
        grad_mem = 2 * n_params / tp_size / pp_size / dp_size
    else:
        grad_mem = 2 * n_params / tp_size / pp_size

    # Optimizer memory (fp32 master weights + Adam states)
    if zero_stage >= 1:
        opt_mem = 12 * n_params / tp_size / pp_size / dp_size
    else:
        opt_mem = 12 * n_params / tp_size / pp_size

    # Activation memory (simplified)
    layers_per_stage = n_layers // pp_size
    act_mem = 2 * batch_size * seq_len * hidden_dim * layers_per_stage / tp_size

    return {
        "model_gb": model_mem / 1e9,
        "gradient_gb": grad_mem / 1e9,
        "optimizer_gb": opt_mem / 1e9,
        "activation_gb": act_mem / 1e9,
        "total_gb": (model_mem + grad_mem + opt_mem + act_mem) / 1e9
    }
```

### Common Configurations

| Model Size | GPUs | TP | PP | DP | ZeRO |
|------------|------|----|----|----|----|
| 7B | 8 | 1 | 1 | 8 | 2 |
| 13B | 8 | 2 | 1 | 4 | 2 |
| 70B | 64 | 8 | 2 | 4 | 1 |
| 175B | 256 | 8 | 4 | 8 | 1 |
| 540B | 1024 | 8 | 8 | 16 | 1 |

### Tuning Tips

1. **Start with DP only** (FSDP/ZeRO-3) for models up to ~30B
2. **Add TP=8** when single-node memory exhausted
3. **Add PP** for models exceeding single-node capacity
4. **Tune micro-batch size** to maximize GPU utilization
5. **Overlap communication** with computation where possible

---

## Training Efficiency Metrics

### MFU (Model FLOPs Utilization)

```
MFU = Achieved FLOPs / Peak FLOPs

Achieved FLOPs = 6 × N × D / Training_Time
Peak FLOPs = GPU_Count × GPU_Peak_FLOPs
```

**Typical MFU values**:
| Configuration | MFU |
|---------------|-----|
| Single GPU | 50-60% |
| 8 GPU (DP) | 45-55% |
| 64 GPU (TP+DP) | 40-50% |
| 256+ GPU (3D) | 30-45% |

### Communication Overhead

```
Communication_Overhead = 1 - (Compute_Time / Total_Time)
```

**Rule of thumb**: Communication should be <30% of total time.

---

## Future Directions

### Near-term (2025)

1. **Sequence parallelism**: Partitioning along sequence dimension for very long contexts
2. **Context parallelism**: Splitting attention computation for 1M+ tokens
3. **Async pipeline**: Reducing bubble overhead
4. **FP8 training**: Halving communication volume

### Research Frontiers

1. **Automatic parallelism**: Learning optimal parallelism strategies
2. **Heterogeneous clusters**: Mixed GPU types, memory tiers
3. **Cross-datacenter training**: Training across geographic regions
4. **Energy-aware scheduling**: Optimizing for power consumption

### Open Questions

1. **Optimal parallelism search**: How to find best configuration automatically?
2. **Fault tolerance**: Handling GPU/node failures gracefully
3. **Elastic training**: Adding/removing GPUs during training
4. **Communication compression**: Trading quality for bandwidth

---

## Sources

### Foundational Papers
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) - Microsoft, 2019
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053) - NVIDIA, 2019
- [GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965) - Google, 2018

### Framework Documentation
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)

### Training Reports
- [LLaMA Training Details](https://arxiv.org/abs/2302.13971) - Meta, 2023
- [BLOOM Training](https://arxiv.org/abs/2211.05100) - BigScience, 2022
- [PaLM Training](https://arxiv.org/abs/2204.02311) - Google, 2022

### Guides
- [Efficient Training on Multiple GPUs](https://huggingface.co/docs/transformers/perf_train_gpu_many) - HuggingFace
- [FSDP vs DeepSpeed](https://huggingface.co/docs/accelerate/concept_guides/fsdp_and_deepspeed) - HuggingFace
- [Model Parallelism Guide](https://huggingface.co/docs/transformers/parallelism) - HuggingFace
