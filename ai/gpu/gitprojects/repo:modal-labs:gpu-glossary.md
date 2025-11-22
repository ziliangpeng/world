# Modal GPU Glossary

**GitHub:** https://github.com/modal-labs/gpu-glossary
**Website:** https://modal.com/gpu-glossary

## Overview

A hypertext dictionary of 70+ GPU programming terms focused on NVIDIA GPUs. Published by Modal (Dec 2024) as "GPU documentation for humans."

## Content Categories

1. **Device Hardware** - SMs, CUDA cores, Tensor cores, memory hierarchy, caches
2. **Device Software** - CUDA, PTX, kernels, threads, warps, blocks
3. **Host Software** - nvcc, nvidia-smi, cuBLAS, cuDNN, TensorRT, Nsight
4. **Performance** - Occupancy, roofline model, arithmetic intensity, memory bandwidth

## Key Features

- Open source (CC BY 4.0 for content, MIT for code)
- Interactive hypertext format with cross-linked concepts
- Practical insights that cut through marketing specs
- Organized as markdown files in GitHub repo
- Retro terminal-style UI on website

## Notable Insights

- "CUDA Core" has no fixed definition across architectures
- High occupancy ≠ high performance (modern GEMM kernels run at single-digit occupancy)
- Warps are implementation details, not part of official CUDA programming model
- SM context switches are 1000x faster than CPU context switches
