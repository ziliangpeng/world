# DeepSeek Series

DeepSeek models are known for groundbreaking efficiency innovations, particularly Multi-head Latent Attention (MLA) and highly optimized Mixture of Experts (MoE) architectures. Since November 2023, DeepSeek has released 40+ models across 8 major families.

## Model Families Overview

### V-Series (Foundation Models)
- DeepSeek-LLM V1 (Nov 2023) - 7B, 67B
- DeepSeek-V2 (May 2024) - 236B MoE
- DeepSeek-V2.5 (Sep 2024) - 236B MoE
- DeepSeek-V3 (Dec 2024) - 671B MoE
- DeepSeek-V3-0324 (Mar 2025) - 671B MoE, MIT licensed
- DeepSeek-V3.1 (Aug 2025) - 840B MoE
- DeepSeek-V3.1-Terminus (Sep 2025) - 671B MoE
- DeepSeek-V3.2-Exp (Sep 2025) - 671B MoE

### R-Series (Reasoning Models)
- DeepSeek-R1-Lite-Preview (Nov 2024) - Preview only
- DeepSeek-R1 (Jan 2025) - 671B MoE
- DeepSeek-R1 Distilled (Jan 2025) - 1.5B-70B dense
- DeepSeek-R1-0528 (May 2025) - 685B MoE

### Coder Series
- DeepSeek-Coder V1 (Nov 2023) - 1.3B, 5.7B, 6.7B, 33B
- DeepSeek-Coder-V1.5 (Early 2024) - 7B
- DeepSeek-Coder-V2 (Jun 2024) - 16B, 236B MoE

### Specialized Models
- DeepSeek-Math (Feb 2024) - 7B
- DeepSeek-VL (Mar 2024) - 1.3B, 7B
- DeepSeek-VL2 (Dec 2024) - 3.37B-27.5B MoE
- JanusFlow (Nov 2024) - 1.3B
- Janus-Pro (Jan 2025) - 1.5B, 7B
- DeepSeek-Prover V1 (May 2024)
- DeepSeek-Prover-V1.5 (Aug 2024)
- DeepSeek-Prover-V2 (Apr 2025) - 7B, 671B
- DeepSeek-OCR (Oct 2025) - 3B MoE

---

## DeepSeek-Coder V1 (November 2, 2023)

**First DeepSeek model ever released**

### Model Specifications
- **Parameters**: 1.3B, 5.7B, 6.7B, 33B (4 sizes)
- **Architecture**: Dense transformer, trained from scratch
- **Context**: 16K tokens
- **Training Data**: 2T tokens (87% code, 13% natural language)
- **arXiv**: [2401.14196](https://arxiv.org/abs/2401.14196)

### Key Features
- First model from DeepSeek
- Trained specifically for code generation
- Base and Instruct variants for each size
- Competitive with CodeLlama on HumanEval

---

## DeepSeek-LLM V1 (November 29, 2023)

### Model Specifications
- **Parameters**: 7B, 67B
- **Architecture**: Dense transformer (67B uses Grouped-Query Attention)
- **Context**: 4K tokens
- **Vocabulary**: 102,400 tokens
- **Training Data**: 2T tokens in English and Chinese
- **arXiv**: [2401.02954](https://arxiv.org/abs/2401.02954)

### Key Features
- First general-purpose LLM from DeepSeek
- Bilingual (English and Chinese)
- Base and Chat variants

---

## DeepSeek-Math (February 5, 2024)

### Model Specifications
- **Parameters**: 7B
- **Architecture**: Dense (continues from DeepSeek-Coder-V1.5-7B)
- **Training Data**: 120B math-related tokens from Common Crawl
- **arXiv**: [2402.03300](https://arxiv.org/abs/2402.03300)

### Performance
- 51.7% on MATH benchmark
- Variants: Base, Instruct, RL

---

## DeepSeek-VL (March 11, 2024)

### Model Specifications
- **Parameters**: 1.3B, 7B
- **Architecture**: Hybrid vision encoder + LLM
- **arXiv**: [2403.05525](https://arxiv.org/abs/2403.05525)

### Key Features
- First vision-language model from DeepSeek
- Processes 1024×1024 images efficiently
- Hybrid encoder for semantic + detail capture
- Base and Chat variants

---

## DeepSeek-V2 (May 6, 2024)

### Model Specifications
- **Total Parameters**: 236 billion
- **Activated per Token**: 21 billion (~8.9% activation)
- **Architecture**: MoE with Multi-head Latent Attention (MLA)
- **Context**: 128K tokens
- **arXiv**: [2405.04434](https://arxiv.org/abs/2405.04434)

### Key Innovations

#### 1. Multi-head Latent Attention (MLA)

Revolutionary attention mechanism that dramatically reduces KV cache size:

**Traditional Multi-Head Attention (MHA)**:
- Separate K, V for each head
- KV cache size: `batch × seq_len × n_heads × head_dim`
- Large memory footprint

**Multi-head Latent Attention (MLA)**:
- Compress K, V into low-rank latent representation
- KV cache size: `batch × seq_len × latent_dim`
- 5-10x smaller KV cache
- Better quality than MHA in practice

**How MLA Works**:
```python
# Conceptual structure
class MLA:
    def __init__(self, d_model, n_heads, latent_dim):
        # Project to shared latent space
        self.kv_proj_down = Linear(d_model, latent_dim)
        # Project from latent to per-head K, V
        self.k_proj_up = Linear(latent_dim, n_heads * head_dim)
        self.v_proj_up = Linear(latent_dim, n_heads * head_dim)

    def forward(self, x):
        # Compress to latent
        kv_latent = self.kv_proj_down(x)  # [batch, seq, latent_dim]

        # Expand to multi-head
        k = self.k_proj_up(kv_latent)
        v = self.v_proj_up(kv_latent)

        return multi_head_attention(q, k, v)
```

#### 2. DeepSeekMoE Architecture

**Fine-Grained Expert Segmentation**:
- Many small experts instead of few large experts
- More flexible routing and specialization

**Isolated Shared Experts**:
- Some experts always activated (shared knowledge)
- Other experts routed (specialized knowledge)
- Better balance between generalization and specialization

---

## DeepSeek-Prover V1 (May 23, 2024)

### Model Specifications
- **Architecture**: Theorem proving model for Lean 4
- **arXiv**: [2405.14333](https://arxiv.org/abs/2405.14333)

### Key Features
- First theorem proving model from DeepSeek
- Uses large-scale synthetic data
- Formal verification in Lean 4

---

## DeepSeek-Coder-V2 (June/July 2024)

### Model Specifications
- **Parameters**: 16B (2.4B active), 236B (21B active)
- **Architecture**: MoE based on DeepSeekMoE framework
- **Context**: 128K tokens

### Key Features
- Breaking barrier of closed-source code models
- Base and Instruct variants
- API model upgraded to DeepSeek-Coder-V2-0724 in July 2024

---

## DeepSeek-Prover-V1.5 (August 2024)

### Key Features
- Enhanced theorem proving
- RLPAF (Reinforcement Learning from Proof Assistant Feedback)
- Improved success rate on Lean 4 proofs

---

## DeepSeek-V2.5 (September 5, 2024)

### Model Specifications
- **Parameters**: 236B total, 21B active (same as V2)
- **Architecture**: MoE
- **Context**: 128K tokens

### Key Features
- Combines V2-Chat and Coder-V2-Instruct capabilities
- Unified general and coding abilities
- Revised in December 2024

---

## JanusFlow (November 13, 2024)

### Model Specifications
- **Parameters**: 1.3B
- **Architecture**: Unified multimodal (autoregressive + rectified flow)
- **arXiv**: [2411.07975](https://arxiv.org/abs/2411.07975)

### Key Features
- Unified image understanding and generation
- SigLIP-L vision encoder (384×384)
- SDXL-VAE for generation

---

## DeepSeek-R1-Lite-Preview (November 20, 2024)

### Model Specifications
- **Parameters**: Not disclosed (preview/API only)
- **Architecture**: Reasoning model with chain-of-thought

### Key Features
- First reasoning model preview
- Transparent thought process
- o1-preview-level performance on AIME & MATH
- Preview only, full model not released

---

## DeepSeek-VL2 (December 13, 2024)

### Model Specifications
- **Tiny**: 3.37B total (1.0B activated)
- **Small**: 16.1B total (2.8B activated)
- **Standard**: 27.5B total (4.5B activated)
- **Architecture**: MoE vision-language models
- **arXiv**: [2412.10302](https://arxiv.org/abs/2412.10302)

### Key Features
- Advanced multimodal understanding
- Superior VQA, OCR, document/table/chart understanding

---

## DeepSeek-V3 (December 25, 2024)

### Model Specifications
- **Total Parameters**: 671 billion
- **Activated per Token**: 37 billion (~5.5% activation)
- **Training Data**: 14.8 trillion tokens
- **Training Cost**: Only 2.788M H800 GPU hours (~$5.57M USD)
- **Context**: 128K tokens
- **arXiv**: [2412.19437](https://arxiv.org/abs/2412.19437)

### Architecture Improvements

#### Enhanced Multi-head Latent Attention (MLA)
- Further optimized low-rank compression
- Better latent dimension sizing
- Even lower KV cache overhead

#### Improved DeepSeekMoE

**Auxiliary-Loss-Free Load Balancing**:
- Achieves balance without auxiliary loss
- Simpler training, better performance

**No Token Dropping**:
- Guarantees all tokens are processed
- More stable training and inference

**Sigmoid Affinity Scores**:
- Router uses sigmoid instead of softmax
- Better numerical stability

### Training Efficiency
- **Record-breaking cost**: 671B parameters for only $5.57M USD
- Most cost-efficient training of any model this size
- Competitive with models 10x more expensive to train

### Performance
- Competitive with GPT-4 class models
- Excellent reasoning and coding abilities
- Strong multilingual performance

---

## DeepSeek-R1 (January 20, 2025)

### Model Specifications
- **Parameters**: 671B total, 37B active (based on V3)
- **Architecture**: MoE with large-scale RL (no SFT pretraining)
- **Context**: 128K tokens
- **arXiv**: [2501.12948](https://arxiv.org/abs/2501.12948)

### Key Features
- Reasoning model comparable to OpenAI o1
- Includes R1-Zero trained purely via RL
- Chain-of-thought reasoning
- Major breakthrough in open-source reasoning models

### DeepSeek-R1 Distilled Models (January 20, 2025)

**Qwen-based**: 1.5B, 7B, 14B, 32B (from Qwen 2.5 series)
**Llama-based**: 8B (from Llama3.1-8B), 70B (from Llama3.3-70B-Instruct)

**Performance**:
- 32B: 72.6% on AIME 2024, 94.3% on MATH-500
- 70B: 70.0% on AIME 2024, 94.5% on MATH-500
- Trained on 800K samples from DeepSeek-R1

---

## Janus-Pro (January 27, 2025)

### Model Specifications
- **Parameters**: 1.5B, 7B
- **Architecture**: Unified multimodal understanding and generation

### Key Features
- Based on DeepSeek-LLM-1.5b/7b-base
- SigLIP-L vision encoder (384×384)
- Outperforms DALL-E 3 and SD3 Medium on GenEval/DPG

---

## DeepSeek-V3-0324 (March 24, 2025)

### Model Specifications
- **Parameters**: 671B total, 37B active (same as V3)
- **Context**: 128K tokens
- **License**: **MIT** (changed from custom license)

### Key Features
- Significant improvements in reasoning, coding, math
- Low-key release with no official announcement
- Same architecture as V3 but better trained

---

## DeepSeek-Prover-V2 (April 30, 2025)

### Model Specifications
- **Parameters**: 7B, 671B
- **Architecture**: Theorem proving with recursive proof search

### Performance
- **State-of-the-art theorem proving**
- 88.9% pass ratio on MiniF2F-test
- 49/658 on PutnamBench
- Subgoal decomposition for complex proofs
- **License**: MIT

---

## DeepSeek-R1-0528 (May 28, 2025)

### Model Specifications
- **Parameters**: 685B, plus 8B distilled variant
- **Architecture**: MoE reasoning model
- **License**: MIT

### Key Features
- System prompt support
- JSON output and function calling
- 87.5% on AIME 2025 (up from 70%)
- Reduced hallucinations
- DeepSeek-R1-0528-Qwen3-8B distilled variant

---

## DeepSeek-V3.1 (August 2025)

### Model Specifications
- **Parameters**: 840B total (based on sources)
- **Architecture**: MoE hybrid reasoning model
- **Context**: 128K tokens
- **License**: MIT

### Key Features
- Supports both thinking and non-thinking modes in single model
- Smarter tool calling
- Higher thinking efficiency
- Hybrid reasoning capabilities

---

## DeepSeek-V3.1-Terminus (September 22, 2025)

### Model Specifications
- **Parameters**: 671B total, 37B active
- **Architecture**: MoE (same structure as V3)
- **Context**: 128K tokens (two-phase long-context training)
- **License**: MIT

### Key Features
- Update addressing language consistency
- Enhanced agent capabilities
- FP8 microscaling for efficient inference
- "Finale to V3 era"

---

## DeepSeek-V3.2-Exp (September 29, 2025)

### Model Specifications
- **Parameters**: 671B-685B total
- **Architecture**: MoE with DeepSeek Sparse Attention (DSA)
- **Context**: Extended (long-context optimization)
- **License**: MIT

### Key Features
- Experimental model
- Fine-grained sparse attention for training/inference efficiency
- API prices dropped 50%+
- Performance on par with V3.1-Terminus

---

## DeepSeek-OCR (October 2025)

### Model Specifications
- **Parameters**: 3B MoE (570M activated)
- **Architecture**: DeepEncoder + MoE decoder (DeepSeek3B-MoE-A570M)
- **arXiv**: [2510.18234](https://arxiv.org/abs/2510.18234)

### Key Features
- Contexts Optical Compression
- Compresses long contexts via optical 2D mapping
- Specialized for document understanding and OCR

---

## Key Architectural Innovations

### Multi-head Latent Attention (MLA)

**Memory Comparison**:

**Standard MHA**:
```
KV cache = 2 × batch × seq_len × n_heads × head_dim
Example: 67M elements
```

**GQA** (Grouped Query Attention):
```
KV cache = 2 × batch × seq_len × n_kv_heads × head_dim
Example: 8.4M elements (8x reduction)
```

**MLA**:
```
KV cache = 2 × batch × seq_len × latent_dim
Example: 8.4M elements but BETTER quality than GQA
```

### DeepSeekMoE Architecture

| Component | Traditional MoE | DeepSeekMoE |
|-----------|----------------|-------------|
| Expert Count | 8-16 | 64-256 |
| Expert Size | Large | Small |
| Shared Experts | No | Yes (isolated) |
| Aux Loss | Required | Not required (V3) |
| Token Dropping | Sometimes | Never |
| Routing | Softmax top-K | Sigmoid affinity |

### Architectural Stack

```
Input → Embedding
  ↓
[Repeated ~60-100x]:
  RMSNorm
  → Multi-head Latent Attention (MLA)
    - Down-project to latent (compression)
    - Up-project to multi-head K, V
    - Standard Q projection
    - Attention with compressed KV cache
  → Residual Connection
  → RMSNorm
  → DeepSeekMoE Layer
    - Router (sigmoid affinity)
    - Isolated shared experts (always active)
    - Routed experts (top-K selection)
    - No token dropping
    - No auxiliary loss (V3+)
  → Residual Connection
  ↓
Final RMSNorm → Output
```

---

## Timeline and Evolution

### 2023: Foundation
- Nov 2: DeepSeek-Coder V1 (first model)
- Nov 29: DeepSeek-LLM V1 (general-purpose)

### 2024: MoE Innovations
- Feb: DeepSeek-Math
- Mar: DeepSeek-VL (vision)
- May: **DeepSeek-V2** (MLA + MoE breakthrough)
- Jun: DeepSeek-Coder-V2
- Sep: DeepSeek-V2.5
- Dec: **DeepSeek-V3** (671B, $5.57M training cost)
- Dec: DeepSeek-VL2

### 2025: Reasoning and Refinement
- Jan: **DeepSeek-R1** (reasoning breakthrough)
- Mar: DeepSeek-V3-0324 (MIT license)
- Apr: DeepSeek-Prover-V2
- May: DeepSeek-R1-0528
- Aug: DeepSeek-V3.1
- Sep: DeepSeek-V3.1-Terminus, V3.2-Exp
- Oct: DeepSeek-OCR

---

## Impact on the Field

### Technical Innovations
1. **MLA**: New attention mechanism paradigm with 5-10x KV cache reduction
2. **Auxiliary-loss-free MoE**: Simpler, better training
3. **No token dropping**: Quality improvement
4. **Sigmoid routing**: Better than softmax for MoE
5. **Reasoning via RL**: Pure RL training (R1-Zero)

### Cost Efficiency
- Proved that smart architecture > brute force compute
- 671B model trained for ~$5.6M USD
- Democratizes large-scale model training
- Shows path to sustainable AI development

### Open Source Contribution
- Released weights and technical details for most models
- Enabled research into MLA and improved MoE
- Set new efficiency standards
- MIT licensing for recent models

---

## Sources

### Research Papers
- [DeepSeek-Coder](https://arxiv.org/abs/2401.14196)
- [DeepSeek-LLM](https://arxiv.org/abs/2401.02954)
- [DeepSeek-Math](https://arxiv.org/abs/2402.03300)
- [DeepSeek-VL](https://arxiv.org/abs/2403.05525)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek-Prover](https://arxiv.org/abs/2405.14333)
- [JanusFlow](https://arxiv.org/abs/2411.07975)
- [DeepSeek-VL2](https://arxiv.org/abs/2412.10302)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948)
- [DeepSeek-OCR](https://arxiv.org/abs/2510.18234)
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066)

### Official Resources
- [DeepSeek Official Website](https://www.deepseek.com/)
- [DeepSeek GitHub Organization](https://github.com/deepseek-ai)
- [DeepSeek API Documentation](https://api-docs.deepseek.com/)
- [HuggingFace DeepSeek](https://huggingface.co/deepseek-ai)

### Technical Analyses
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek MoE and V2 Analysis](https://www.chipstrat.com/p/deepseek-moe-and-v2)
- [DeepSeek-V3 Release Analysis](https://www.helicone.ai/blog/deepseek-v3)
- [Complete Guide to DeepSeek Models](https://www.bentoml.com/blog/the-complete-guide-to-deepseek-models-from-v3-to-r1-and-beyond)
