# LLM Checkpoint Management

Effective checkpoint management is critical for large-scale LLM training. With multi-billion parameter models trained over weeks or months, checkpoint strategies determine your ability to recover from failures, iterate on experiments, and deploy models efficiently. This document covers practical checkpoint management from training through production.

---

## Why Checkpoint Management Matters

### The Cost of Poor Checkpointing

| Problem | Impact | Cost |
|---------|--------|------|
| **No checkpointing** | Lost training progress on failure | Days to weeks of compute wasted |
| **Checkpoint corruption** | Cannot resume training | Rollback to older checkpoint |
| **Too frequent checkpointing** | I/O bottleneck, slow training | 10-30% throughput degradation |
| **Too infrequent checkpointing** | Lost progress on failure | Hours of recomputation |
| **Poor organization** | Cannot find best checkpoint | Manual searching, deployment delays |

### Checkpointing at Scale

**Example: Llama 3 70B training**
- Checkpoint size: ~280GB (BF16, unsharded)
- Training duration: ~1.7M GPU hours
- Checkpointing every 1000 steps at 20min/checkpoint = 33 hours of I/O
- Without optimization: 2% of training time spent on I/O

**Optimization matters**: Reducing checkpoint time from 20min to 5min saves 25 hours.

---

## Checkpoint Strategy During Training

### 1. Checkpoint Frequency

**Three-tier checkpointing strategy:**

```python
# Training loop with tiered checkpointing
for step in range(max_steps):
    train_step()

    # Tier 1: Frequent lightweight checkpoints (recovery only)
    if step % 100 == 0:
        save_lightweight_checkpoint(step)  # Optimizer state only

    # Tier 2: Regular full checkpoints
    if step % 1000 == 0:
        save_full_checkpoint(step)

    # Tier 3: Evaluation checkpoints (keep forever)
    if step % 5000 == 0:
        checkpoint = save_full_checkpoint(step)
        eval_score = evaluate(checkpoint)
        if eval_score > best_score:
            mark_as_best(checkpoint)
```

**Frequency guidelines:**

| Checkpoint Type | Frequency | What to Save | Retention |
|----------------|-----------|--------------|-----------|
| **Recovery** | Every 100-500 steps | Model + optimizer state | Last 3-5 only |
| **Regular** | Every 1000-2000 steps | Full checkpoint | Last 10-20 |
| **Evaluation** | Every 5000-10000 steps | Full + metrics | All or top-K |
| **End-of-epoch** | Per epoch | Full + metrics | All |

### 2. What to Save

**Minimum checkpoint contents:**
```python
checkpoint = {
    # Model
    'model_state_dict': model.state_dict(),

    # Optimizer
    'optimizer_state_dict': optimizer.state_dict(),

    # Scheduler
    'scheduler_state_dict': scheduler.state_dict(),

    # Training state
    'step': current_step,
    'epoch': current_epoch,
    'rng_state': torch.get_rng_state(),

    # Metadata
    'config': model_config,
    'timestamp': datetime.now(),
}
```

**Extended checkpoint (for production):**
```python
checkpoint = {
    **base_checkpoint,

    # Metrics
    'train_loss': train_loss,
    'val_loss': val_loss,
    'perplexity': perplexity,
    'eval_metrics': {
        'mmlu': 0.70,
        'hellaswag': 0.82,
    },

    # Data state
    'dataset_position': dataloader.get_position(),
    'data_seed': data_seed,

    # Environment
    'git_commit': git_hash,
    'training_script': script_path,
}
```

### 3. Sharded vs Consolidated Checkpoints

**Sharded checkpoints** (distributed training):
```
checkpoint-1000/
├── model_0.pt  # GPU 0's shard
├── model_1.pt  # GPU 1's shard
├── model_2.pt  # GPU 2's shard
├── model_3.pt  # GPU 3's shard
└── metadata.json
```

**Consolidated checkpoint** (single file):
```
checkpoint-1000/
└── model.pt  # All parameters merged
```

| Aspect | Sharded | Consolidated |
|--------|---------|--------------|
| **Save speed** | Fast (parallel writes) | Slow (gather then write) |
| **Load speed** | Fast (parallel reads) | Slow (read then scatter) |
| **Disk usage** | Efficient | Same or more |
| **Portability** | Requires same # GPUs | Works anywhere |
| **Inference** | Needs conversion | Ready to use |

**Best practice**:
- Save sharded during training (fast)
- Consolidate periodically for evaluation/deployment

---

## Storage Formats

### 1. PyTorch Native (.pt, .pth)

```python
# Save
torch.save(checkpoint, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

**Pros:**
- Native PyTorch, fast
- Preserves all Python objects

**Cons:**
- Security risk (pickle-based, can execute code)
- Not cross-framework
- Can be large

### 2. SafeTensors (Recommended)

```python
from safetensors.torch import save_file, load_file

# Save
save_file(model.state_dict(), 'model.safetensors')

# Load
state_dict = load_file('model.safetensors')
model.load_state_dict(state_dict)
```

**Pros:**
- Safe (no code execution)
- Fast loading (memory-mapped)
- Cross-framework compatible
- HuggingFace standard

**Cons:**
- Only tensors (no optimizer state without separate file)

### 3. HuggingFace Format

```python
# Save
model.save_pretrained('checkpoint-1000/')
tokenizer.save_pretrained('checkpoint-1000/')

# Directory structure:
# checkpoint-1000/
# ├── config.json
# ├── model.safetensors (or pytorch_model.bin)
# └── tokenizer_config.json

# Load
model = AutoModelForCausalLM.from_pretrained('checkpoint-1000/')
```

**Pros:**
- Ecosystem compatibility
- Easy sharing/deployment
- Automatic sharding for large models

**Cons:**
- More files to manage
- Opinionated structure

### 4. DeepSpeed / FSDP Formats

**DeepSpeed:**
```python
# Automatically sharded
model_engine.save_checkpoint('checkpoints/', tag='step-1000')

# Directory: checkpoints/step-1000/
# ├── mp_rank_00_model_states.pt
# ├── mp_rank_01_model_states.pt
# └── zero_pp_rank_0_mp_rank_00_optim_states.pt
```

**FSDP:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Save sharded
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    state_dict = model.state_dict()
    if rank == 0:
        torch.save(state_dict, 'checkpoint.pt')
```

---

## Checkpoint Organization

### 1. Directory Structure

**Recommended structure:**
```
experiment-llama-3-70b/
├── checkpoints/
│   ├── step-1000/
│   │   ├── model.safetensors
│   │   ├── optimizer.pt
│   │   ├── scheduler.pt
│   │   └── metadata.json
│   ├── step-2000/
│   ├── step-3000/
│   └── best/  # Symlink to best checkpoint
├── eval_results/
│   ├── step-1000.json
│   ├── step-2000.json
│   └── step-3000.json
└── logs/
    ├── train.log
    └── tensorboard/
```

### 2. Naming Conventions

**Step-based naming:**
```
checkpoint-{step:08d}/  # checkpoint-00001000/
checkpoint-{step:08d}-{metric:.2f}/  # checkpoint-00001000-70.5/
```

**Time-based naming:**
```
checkpoint-{timestamp}/  # checkpoint-2024-03-15-14-30-00/
```

**Hybrid (recommended):**
```
checkpoint-step{step:08d}-{timestamp}/  # checkpoint-step00001000-2024-03-15/
```

### 3. Retention Policy

**Example policy:**
```python
def cleanup_checkpoints(checkpoint_dir, keep_last=5, keep_best=3):
    """
    Keep:
    - Last N checkpoints (for resumption)
    - Top K checkpoints by eval metric
    - All checkpoints divisible by 10000 (milestones)
    """
    checkpoints = sorted(list_checkpoints(checkpoint_dir))

    # Always keep
    keep = set()
    keep.update(checkpoints[-keep_last:])  # Last N
    keep.update(get_top_k_checkpoints(checkpoints, k=keep_best))  # Best K
    keep.update([c for c in checkpoints if c.step % 10000 == 0])  # Milestones

    # Delete others
    for checkpoint in checkpoints:
        if checkpoint not in keep:
            delete_checkpoint(checkpoint)
```

---

## Cloud Storage Strategies

### 1. Storage Tiers

| Tier | Use Case | Cost | Retrieval Time |
|------|----------|------|----------------|
| **Local SSD** | Active training | Highest | Instant |
| **Network storage** | Recent checkpoints | High | Seconds |
| **Object storage (hot)** | Last 20 checkpoints | Medium | Seconds to minutes |
| **Object storage (cold)** | Archive | Low | Hours |

**Example workflow:**
```bash
# During training: Save to local SSD
save_checkpoint('/local_ssd/checkpoints/step-1000')

# Background: Upload to cloud (async)
aws s3 sync /local_ssd/checkpoints/ s3://bucket/experiment/checkpoints/ &

# After training: Move old checkpoints to cold storage
aws s3 mv s3://bucket/experiment/checkpoints/step-1000/ \
    s3://bucket/archive/step-1000/ \
    --storage-class GLACIER
```

### 2. Async Checkpointing

**Non-blocking checkpointing:**
```python
import threading

def async_checkpoint(step, state_dict):
    """Save checkpoint in background thread"""
    def _save():
        torch.save(state_dict, f'checkpoint-{step}.pt')
        upload_to_cloud(f'checkpoint-{step}.pt')

    thread = threading.Thread(target=_save)
    thread.start()
    return thread

# Training loop
for step in range(max_steps):
    train_step()

    if step % 1000 == 0:
        # Non-blocking save
        state = get_checkpoint_state()  # Quick copy
        checkpoint_thread = async_checkpoint(step, state)

    # Continue training immediately
```

**Benefits:**
- Training doesn't wait for I/O
- Can overlap checkpoint with next iteration
- Critical for large models (100B+)

---

## Recovery & Fault Tolerance

### 1. Checkpoint Resumption

**Robust resumption script:**
```python
def resume_training(checkpoint_dir):
    """Resume from latest valid checkpoint"""
    # Find latest checkpoint
    checkpoints = sorted(list_checkpoints(checkpoint_dir))

    for checkpoint in reversed(checkpoints):
        try:
            # Verify checkpoint integrity
            if not verify_checkpoint(checkpoint):
                print(f"Checkpoint {checkpoint} corrupted, trying previous")
                continue

            # Load checkpoint
            state = load_checkpoint(checkpoint)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            scheduler.load_state_dict(state['scheduler_state_dict'])

            # Restore RNG state for reproducibility
            torch.set_rng_state(state['rng_state'])

            # Resume from next step
            start_step = state['step'] + 1
            print(f"Resumed from step {state['step']}")
            return start_step

        except Exception as e:
            print(f"Failed to load {checkpoint}: {e}")
            continue

    # No valid checkpoint found
    print("Starting from scratch")
    return 0
```

### 2. Checkpoint Verification

**Integrity checks:**
```python
def verify_checkpoint(checkpoint_path):
    """Verify checkpoint is valid before using"""
    checks = []

    # 1. File exists and size is reasonable
    if not os.path.exists(checkpoint_path):
        return False

    file_size = os.path.getsize(checkpoint_path)
    if file_size < 1024:  # Suspiciously small
        return False

    # 2. Can load without errors
    try:
        state = torch.load(checkpoint_path, map_location='cpu')
    except Exception:
        return False

    # 3. Has required keys
    required_keys = ['model_state_dict', 'optimizer_state_dict', 'step']
    if not all(k in state for k in required_keys):
        return False

    # 4. Parameter count matches expected
    param_count = sum(p.numel() for p in state['model_state_dict'].values())
    if not (expected_params * 0.95 < param_count < expected_params * 1.05):
        return False

    return True
```

### 3. Automatic Rollback

```python
# Training script with auto-rollback
try:
    train(resume_from_checkpoint='latest')
except RuntimeError as e:
    if 'CUDA out of memory' in str(e) or 'NaN' in str(e):
        # Load previous checkpoint and continue
        rollback_steps = 1000
        checkpoint = get_checkpoint(current_step - rollback_steps)
        train(resume_from_checkpoint=checkpoint)
```

---

## Format Conversion

### 1. DeepSpeed ↔ HuggingFace

**DeepSpeed to HuggingFace:**
```python
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

# Convert DeepSpeed ZeRO checkpoint to single file
convert_zero_checkpoint_to_fp32_state_dict(
    checkpoint_dir='checkpoints/step-1000',
    output_file='pytorch_model.bin'
)

# Load into HuggingFace
model = AutoModelForCausalLM.from_pretrained('.')
```

**HuggingFace to DeepSpeed:**
```python
# HuggingFace model automatically works with DeepSpeed
model = AutoModelForCausalLM.from_pretrained('checkpoint/')
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)
```

### 2. FSDP ↔ Standard PyTorch

**FSDP to PyTorch:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

# Consolidate FSDP checkpoint
save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
    consolidated_state = model.state_dict()

# Save as standard PyTorch
if rank == 0:
    torch.save(consolidated_state, 'model.pt')
```

**PyTorch to FSDP:**
```python
# Standard PyTorch checkpoint loads directly into FSDP model
state_dict = torch.load('model.pt')
with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
    model.load_state_dict(state_dict)
```

### 3. Quantization Conversion

**FP32/BF16 → INT8/FP8:**
```python
from transformers import AutoModelForCausalLM
import bitsandbytes as bnb

# Load full precision
model = AutoModelForCausalLM.from_pretrained(
    'checkpoint/',
    torch_dtype=torch.bfloat16
)

# Quantize and save
quantized_model = bnb.quantize_8bit(model)
quantized_model.save_pretrained('checkpoint-int8/')
```

---

## Versioning & Metadata

### 1. Git for Small Models

**Git LFS for models <10GB:**
```bash
# Track checkpoint files with Git LFS
git lfs track "*.safetensors"
git lfs track "*.bin"

# Commit checkpoint
git add checkpoints/step-1000/
git commit -m "Checkpoint at step 1000: val_loss=2.45, MMLU=68.5"
git tag step-1000
git push origin main --tags
```

### 2. Experiment Tracking Integration

**Weights & Biases:**
```python
import wandb

# Log checkpoint as artifact
wandb.init(project='llm-training')

artifact = wandb.Artifact(
    name=f'checkpoint-step-{step}',
    type='model',
    metadata={
        'step': step,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'mmlu': mmlu_score,
    }
)
artifact.add_file('checkpoint.pt')
wandb.log_artifact(artifact)
```

**MLflow:**
```python
import mlflow

with mlflow.start_run():
    # Log checkpoint
    mlflow.log_artifact('checkpoint-1000/', artifact_path='checkpoints')

    # Log metrics
    mlflow.log_metrics({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'step': step,
    })
```

### 3. Checkpoint Metadata File

**metadata.json:**
```json
{
  "checkpoint_name": "checkpoint-step-00001000",
  "step": 1000,
  "epoch": 0.5,
  "timestamp": "2024-03-15T14:30:00Z",

  "training_config": {
    "model": "llama-3-70b",
    "batch_size": 1024,
    "learning_rate": 3e-4,
    "sequence_length": 4096
  },

  "metrics": {
    "train_loss": 2.45,
    "val_loss": 2.52,
    "perplexity": 12.4,
    "mmlu": 68.5,
    "hellaswag": 79.2
  },

  "hardware": {
    "num_gpus": 512,
    "gpu_type": "H100",
    "hours_trained": 48.5
  },

  "git_info": {
    "commit": "a3f2c8d",
    "branch": "main",
    "dirty": false
  }
}
```

---

## Production Checkpoint Management

### 1. Model Registry

**Centralized model registry pattern:**
```
Model Registry
├── experiments/
│   ├── llama-3-70b-run-1/
│   │   ├── checkpoints/  # All training checkpoints
│   │   └── best/  # Best checkpoint
│   └── llama-3-70b-run-2/
├── staging/
│   └── llama-3-70b-candidate/  # Model being tested
└── production/
    ├── llama-3-70b-v1/  # Current production
    └── llama-3-70b-v2/  # Previous version (rollback)
```

**Promotion workflow:**
```python
def promote_to_production(checkpoint_path):
    """Promote checkpoint through stages"""
    # 1. Validate checkpoint
    assert verify_checkpoint(checkpoint_path)

    # 2. Run offline eval suite
    eval_results = run_eval_suite(checkpoint_path)
    assert eval_results['mmlu'] > production_threshold

    # 3. Convert to inference format
    inference_model = convert_for_inference(checkpoint_path)

    # 4. Deploy to staging
    deploy_to_staging(inference_model)

    # 5. Run online eval (shadow mode)
    shadow_metrics = run_shadow_eval(days=3)

    # 6. A/B test with 5% traffic
    ab_results = run_ab_test(traffic_pct=5, duration_hours=48)

    # 7. If passing, promote to production
    if ab_results['win_rate'] > 0.5:
        promote_to_prod(inference_model)
        archive_old_production()
```

### 2. Checkpoint Serving

**Hot-swapping models without downtime:**
```python
class CheckpointServer:
    def __init__(self):
        self.model = None
        self.lock = threading.Lock()

    def load_checkpoint(self, checkpoint_path):
        """Load new checkpoint with zero downtime"""
        # Load new model in background
        new_model = load_model(checkpoint_path)

        # Atomic swap
        with self.lock:
            old_model = self.model
            self.model = new_model

        # Cleanup old model
        del old_model
        torch.cuda.empty_cache()

    def inference(self, input_text):
        with self.lock:
            return self.model.generate(input_text)
```

### 3. Multi-Version Deployment

**Serving multiple checkpoint versions:**
```python
# Load multiple versions for A/B testing
models = {
    'v1': load_checkpoint('production/v1/'),
    'v2': load_checkpoint('production/v2/'),
    'v3': load_checkpoint('staging/v3/')
}

def route_request(user_id, request):
    """Route to different model versions"""
    hash_val = hash(user_id) % 100

    if hash_val < 5:  # 5% to v3 (staging)
        return models['v3'].generate(request)
    elif hash_val < 20:  # 15% to v2
        return models['v2'].generate(request)
    else:  # 80% to v1 (stable)
        return models['v1'].generate(request)
```

---

## Best Practices Summary

### ✅ Do This

1. **Use three-tier checkpointing**: Frequent recovery, regular full, milestone evaluation
2. **Save sharded during training**: Parallel I/O for speed
3. **Use SafeTensors for model weights**: Secure, fast, compatible
4. **Verify checkpoints before use**: Catch corruption early
5. **Async checkpointing for large models**: Don't block training
6. **Organize with clear naming**: step-{step:08d} or hybrid
7. **Implement retention policies**: Keep last N + best K + milestones
8. **Version with metadata**: Track config, metrics, git state
9. **Test resumption regularly**: Ensure recovery works
10. **Cloud storage with tiers**: Local SSD + object storage + archive

### ❌ Avoid This

1. **Don't checkpoint too frequently**: I/O bottleneck
2. **Don't keep all checkpoints**: Storage explosion
3. **Don't use pickle for production**: Security risk
4. **Don't skip verification**: Corrupted checkpoints waste time
5. **Don't forget optimizer state**: Can't resume training properly
6. **Don't mix formats**: Standardize on one format per project
7. **Don't checkpoint to same path**: Risk overwriting good checkpoint
8. **Don't forget RNG state**: Breaks reproducibility

---

## Checkpoint Workflow

```
Training Start
    ↓
Initialize or Resume
├─ Check for existing checkpoints
├─ Verify integrity
└─ Load latest valid checkpoint
    ↓
Training Loop
├─ Every 100 steps: Save recovery checkpoint (async)
├─ Every 1000 steps: Save full checkpoint (async)
└─ Every 5000 steps: Save + evaluate + promote if best
    ↓
Checkpoint Saved
├─ Verify integrity
├─ Upload to cloud (async)
├─ Update metadata
└─ Cleanup old checkpoints
    ↓
Training Complete
├─ Save final checkpoint
├─ Run full eval suite
├─ Convert to inference format
└─ Promote to model registry
    ↓
Production Deployment
├─ Deploy to staging
├─ Shadow evaluation
├─ A/B testing
└─ Promote to production
```

---

## Case Studies

### Case Study 1: Recovery from Week-Long Failure

**Problem**: Training job failed after 5 days due to node failure.

**Without good checkpointing**:
- Last checkpoint: 3 days ago
- Lost: 2 days of training (~$50,000 of compute)
- Action: Resume from 3-day-old checkpoint

**With good checkpointing**:
- Last checkpoint: 15 minutes ago
- Lost: 15 minutes (~$200 of compute)
- Action: Resume from latest checkpoint, minimal loss

**Lesson**: Frequent recovery checkpoints (every 100-500 steps) critical for large-scale training.

### Case Study 2: Checkpoint Corruption

**Problem**: Final checkpoint corrupted during save, cannot deploy model.

**Investigation**:
- Training completed, saved final checkpoint
- File size: 50% of expected (incomplete write)
- No verification step

**Solution**:
- Rolled back to checkpoint 1000 steps earlier
- Re-ran last 1000 steps
- Implemented checkpoint verification

**Lesson**: Always verify checkpoint integrity, keep multiple recent checkpoints.

### Case Study 3: Format Conversion Bottleneck

**Problem**: 2-day delay converting DeepSpeed checkpoint to HuggingFace for deployment.

**Investigation**:
- DeepSpeed ZeRO-3 checkpoint: 512 shards
- Manual conversion: 48 hours
- Blocking production deployment

**Solution**:
- Automated conversion pipeline during training
- Save both DeepSpeed (for training) and HuggingFace (for inference)
- Async conversion doesn't slow training

**Lesson**: Plan for inference format early, automate conversion.

---

## Tools & Infrastructure

### Checkpoint Management Tools

**Hugging Face Hub:**
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path='checkpoint-1000/',
    repo_id='org/model-name',
    revision='step-1000',
)
```

**AWS S3 + lifecycle policies:**
```bash
# S3 lifecycle policy for automatic archival
aws s3api put-bucket-lifecycle-configuration \
  --bucket training-checkpoints \
  --lifecycle-configuration file://lifecycle.json

# lifecycle.json: Move to Glacier after 30 days
{
  "Rules": [{
    "Id": "ArchiveOldCheckpoints",
    "Filter": {"Prefix": "checkpoints/"},
    "Status": "Enabled",
    "Transitions": [{
      "Days": 30,
      "StorageClass": "GLACIER"
    }]
  }]
}
```

**Checkpoint verification script:**
```python
#!/usr/bin/env python3
import os
import torch
import hashlib

def verify_all_checkpoints(checkpoint_dir):
    """Verify all checkpoints in directory"""
    checkpoints = sorted(os.listdir(checkpoint_dir))

    results = []
    for ckpt in checkpoints:
        path = os.path.join(checkpoint_dir, ckpt, 'model.pt')

        # File integrity
        if not os.path.exists(path):
            results.append((ckpt, 'MISSING'))
            continue

        # Loadable
        try:
            state = torch.load(path, map_location='cpu')
            param_count = sum(p.numel() for p in state['model_state_dict'].values())
            results.append((ckpt, 'OK', param_count))
        except Exception as e:
            results.append((ckpt, 'CORRUPTED', str(e)))

    return results
```

---

## Resources

### Documentation
- [PyTorch Distributed Checkpoint](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)
- [DeepSpeed Checkpointing](https://www.deepspeed.ai/tutorials/model-checkpointing/)
- [FSDP State Dict Documentation](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [SafeTensors Documentation](https://huggingface.co/docs/safetensors/index)

### Tools
- [SafeTensors](https://github.com/huggingface/safetensors) - Safe, fast tensor serialization
- [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/index) - Model hosting and versioning
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html) - Model lifecycle management
- [DVC](https://dvc.org/) - Data and model versioning

### Blog Posts
- [Efficient Checkpointing at Scale](https://engineering.fb.com/2021/07/15/open-source/fsdp/) - Meta FSDP
- [Checkpointing Best Practices](https://huggingface.co/docs/transformers/main_classes/trainer#checkpointing) - HuggingFace
- [Asynchronous Checkpointing](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) - Microsoft DeepSpeed

---

**Related Documentation**:
- [Distributed Training](../pre-training/distributed-training.md) - FSDP, DeepSpeed, parallelism strategies
- [Evaluation](evaluation.md) - When and how to evaluate checkpoints
- [Monitoring](monitoring.md) - Track checkpoint health and storage usage
