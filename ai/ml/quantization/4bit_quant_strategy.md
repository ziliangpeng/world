# 4‑Bit Quantization Strategy for vLLM (H100 and MI325X)

## Scope

- 4‑bit weight‑only quantization for LLM inference using vLLM.
- Two scenarios: H100‑only (NVIDIA) and cross‑vendor H100 + MI325X (AMD ROCm).

## Goals

- Maximize throughput and VRAM savings with minimal quality loss.
- Keep artifacts/runtimes simple, reproducible, and easy to operate.
- Provide a small experiment matrix and clear acceptance criteria.

## Methods

- NVFP4 (ModelOpt FP4, groupwise weight‑only)
  - Data: Optional (data‑free works; small calibration improves quality).
  - Runtime: vLLM on NVIDIA via `--quantization modelopt_fp4`.
  - Notes: Best fit for H100; artifact is NVIDIA‑only.
- GPTQ (least‑squares PTQ, weight‑only)
  - Data: Required (small calibration set of activations).
  - Runtime: vLLM via `--quantization gptq` on NVIDIA and AMD.
  - Notes: Strong accuracy at 4‑bit; portable.
- AWQ (activation‑aware PTQ, weight‑only)
  - Data: Required (light calibration).
  - Runtime: vLLM via `--quantization awq` on NVIDIA and AMD.
  - Notes: Competitive with GPTQ; simpler flow in some stacks.

## Decision Guide

- H100‑only
  - First choice: NVFP4 (highest H100 perf; simplest ops).
  - Alternative: GPTQ/AWQ if you want a single artifact later usable on AMD.
- H100 + MI325X
  - First choice: GPTQ or AWQ (portable across vendors).
  - Avoid NVFP4 for cross‑vendor unless maintaining a separate H100 artifact.

## Calibration Data

- NVFP4 (optional): 20k–200k in‑domain tokens can improve clipping/scales and inform exemptions.
- GPTQ/AWQ (needed): 128–512 sequences (mixed lengths up to target context), diverse and in‑domain.
- Collection tips: de‑dup, mix short/long prompts, include target tasks; avoid instruction‑only if serving general chat.

## Layer/Module Policy

- Default exemptions
  - Always consider exempting: `lm_head`.
  - Optionally exempt: `embed_tokens` (if quality is tight and VRAM allows).
  - Boundary protection: first and last transformer block (whole block or just `mlp.down_proj` and `attn.o_proj`).
- Typical sensitivity order
  - Most sensitive: `lm_head`, embeddings, boundary blocks, `mlp.down_proj`, `attn.o_proj`.
  - Usually quantizable: `mlp.up_proj`, `mlp.gate` (avoid blanket gate exemptions unless data suggests it).
- Strategy
  - Start minimal: exempt `lm_head` (+ embeddings if needed).
  - If quality misses: protect boundary blocks; next, target `down_proj`/`o_proj` in those blocks.
  - Use per‑layer sensitivity only if pushing for near‑parity.

## Group Size

- Start with G64; try G128 for better perf/memory if quality allows.
- Use G32 as a quality fallback; G16 is most robust but adds overhead (more scales, slower).
- Expectation: G64 is often the best Pareto; very large models or hard tasks may prefer G32.

## vLLM Configuration

- H100 + NVFP4
  - `--quantization modelopt_fp4 --dtype bf16 --kv-cache-dtype fp8`
  - Keep weights FP4 (ModelOpt), compute BF16, KV cache FP8.
- H100 + GPTQ/AWQ (portable)
  - `--quantization gptq|awq --dtype bf16 --kv-cache-dtype fp8`
- MI325X (ROCm)
  - `--quantization gptq|awq --dtype bf16`
  - KV cache: FP8 may be experimental; default to `--kv-cache-dtype bf16` unless your build supports FP8 KV reliably.

## Experiment Matrix (rank by quality, then tokens/s)

- NVFP4 (H100)
  - A1: G64, exempt `lm_head`
  - A2: A1 + exempt `embed_tokens`
  - A3: A2 + BF16 first+last block
  - A4: A2 + only `down_proj`/`o_proj` in first+last block BF16 (instead of whole blocks)
  - Optional: G128 variant of A2/A4 for throughput
- GPTQ (portable)
  - B1: GPTQ G128, exempt `lm_head`
  - B2: B1 + BF16 first+last block
  - B3: GPTQ G64, exempt `lm_head`
- AWQ (portable)
  - C1: AWQ G128, exempt `lm_head`
  - C2: C1 + BF16 first+last block

## Acceptance Criteria

- Quality: ≤ X% relative PPL increase vs BF16 on held‑out; task metrics within Y%. Example: PPL +5%, task −1–2%.
- Performance: ≥ Z% tokens/s gain vs BF16 at target batch/seq. Example: +60–100% for 4‑bit.
- Stability: No divergence/regressions on long‑context; no OOM at target concurrency.

## Evaluation Protocol

- Perplexity: 5k–50k tokens, mixed lengths up to your max context.
- Tasks: 3–5 lightweight benchmarks relevant to your use (e.g., MT‑bench‑style chat, domain QA).
- Long context: 16k–128k synthetic + real prompts; check drift with FP8 KV (H100).
- Report: PPL, task metrics (EM/F1/ROUGE/BLEU as relevant), latency p50/p95, tokens/s, VRAM.

## Production Guidance

- Version pins
  - vLLM: recent release with `modelopt_fp4` support for H100.
  - NVIDIA: CUDA + TransformerEngine aligned with FP4/FP8 kernels.
  - AMD: ROCm + vLLM build known‑good for GPTQ/AWQ.
- Repro artifacts
  - Save `hf_quant_config.json`, the quant recipe, code commits, vLLM + driver versions, and calibration manifest.
  - Validate checkpoint load on target GPUs before running large evals.

## Risks & Mitigations

- Quality dips (formatting, safety): add boundary protection or move G64→G32 (NVFP4) / increase GPTQ calibration size.
- Long‑context degradation: keep KV in BF16 for stress tests; re‑enable FP8 KV if acceptable.
- Over‑exemption bloat: prefer targeted `down_proj`/`o_proj` over blanket `mlp.gate`; monitor VRAM.
- ROCm regressions: use GPTQ/AWQ; avoid NVFP4 on AMD.

## Quick Commands

- H100 NVFP4

```bash
vllm serve <model_path_or_hub_id> \
  --quantization modelopt_fp4 \
  --dtype bf16 \
  --kv-cache-dtype fp8 \
  --max-model-len <ctx_len> \
  --max-num-seqs <concurrency>
```

- H100/MI325X GPTQ

```bash
vllm serve <model_path_or_hub_id> \
  --quantization gptq \
  --dtype bf16 \
  --max-model-len <ctx_len> \
  --max-num-seqs <concurrency>
```

- H100/MI325X AWQ

```bash
vllm serve <model_path_or_hub_id> \
  --quantization awq \
  --dtype bf16 \
  --max-model-len <ctx_len> \
  --max-num-seqs <concurrency>
```

## Recommended Starting Points

- H100‑only
  - NVFP4 G64, exempt `lm_head`; enable FP8 KV; add boundary protection only if needed.
- H100 + MI325X
  - GPTQ G128, exempt `lm_head`; FP8 KV on H100, BF16 KV on MI325X; add boundary protection as needed.

---

Appendix: Minimal Calibration Guidance

- GPTQ/AWQ: 128–512 sequences; mix short/long up to target context; ensure in‑domain coverage.
- NVFP4 (optional): 20k–200k tokens; improves clipping/scale choices and exemption policy for large models.

