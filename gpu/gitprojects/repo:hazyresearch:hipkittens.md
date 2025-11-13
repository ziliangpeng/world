# Research Report: HipKittens - High-Performance AI Kernels for AMD GPUs

## 1. Introduction

HipKittens is an open-source project from HazyResearch at Stanford University that provides a set of minimal, opinionated C++ primitives for writing high-performance AI kernels specifically for AMD GPUs. The project's primary motivation is to foster a "multi-silicon" future for AI, where software is not locked into a single hardware vendor's ecosystem. It aims to abstract hardware-specific details while exposing the necessary primitives for achieving near-metal performance. HipKittens builds upon the concepts of a previous project, ThunderKittens, and adapts them to the architecture of AMD's CDNA3 and CDNA4 GPUs.

### 1.1 What HipKittens Is (and Isn't)

HipKittens is **more than just a wrapper** around HIP/ROCm APIs - it's a high-performance kernel library that:

*   **Wraps low-level APIs**: Provides cleaner abstractions over raw HIP calls
*   **Provides opinionated primitives**: Pre-configured tensor core operations and optimized memory layouts
*   **Embeds hand-optimized code**: Direct assembly instructions for critical performance paths
*   **Codifies proven patterns**: Implements scheduling strategies like "8-wave ping pong" and "4-wave interleave" that aren't provided by HIP itself

The relationship can be visualized as:
```
Your Kernel Code
       ↓
   HipKittens (tile primitives, optimized patterns)
       ↓
     HIP API (low-level GPU programming interface)
       ↓
    ROCm Stack (AMD's GPU software platform)
       ↓
   AMD GPU Hardware
```

**Important**: HipKittens is AMD-specific and does not provide cross-vendor portability. Its sibling project, ThunderKittens, serves the same role for NVIDIA/CUDA GPUs. The two projects share conceptual patterns and design philosophy but require separate implementations.

## 2. Core Primitives and Concepts

HipKittens is designed from the "hardware up," meaning its design principles are dictated by the underlying silicon. The core of the library revolves around a few key concepts:

*   **Tile Primitives:** The fundamental unit of computation in HipKittens is the "tile," which is sized according to the tensor core units of the AMD GPUs. Memory operations on tiles are designed to be coalesced and bank-conflict-free, and they eagerly utilize tensor core layouts. The project emphasizes minimizing the cost of address computation.

*   **Python-Inspired Functions:** To provide a user-friendly interface, HipKittens wraps low-level HIP (Heterogeneous-compute Interface for Portability) and assembly code in lightweight, Python-inspired functions. These functions operate on tiles, enabling developers to express complex computations in a more intuitive way.

*   **Asynchronous Memory Operations:** To hide memory latency, HipKittens heavily utilizes asynchronous loads and stores. This is achieved by using direct buffer loads to shared memory, allowing the GPU to overlap memory transfers with computation.

*   **Scheduling and Overlapping:** The project identifies and implements two core patterns for overlapping computation and memory operations: "8-wave ping pong" and "4-wave interleave." These scheduling patterns are reused across different kernels to maximize GPU utilization.

## 3. Architecture and Implementation

The HipKittens repository is structured to separate the core library, kernel implementations, tests, and analysis tools.

*   **`include/`:** This directory contains the C++ header files that define the core primitives of the HipKittens library. These primitives provide the building blocks for creating custom kernels.

*   **`kernels/`:** This directory houses the implementations of several key AI kernels, demonstrating the use of the HipKittens primitives.

*   **`tests/`:** The project includes a suite of unit tests to ensure the correctness of the library functions.

*   **`analysis/`:** To validate the performance of the kernels, the repository provides scripts for benchmarking and plotting results.

*   **`training/`:** This directory contains scripts for training well-known models like BERT and Llama using the kernels implemented with HipKittens.

The project is intended to be used within a specific Docker environment based on ROCm (Radeon Open Compute platform), ensuring that all dependencies are correctly managed.

## 4. Key Kernels and Applications

HipKittens provides implementations for a variety of essential AI kernels, including:

*   **GEMM (General Matrix Multiplication):** A highly optimized BF16 GEMM kernel is provided, which is a cornerstone of many deep learning models.

*   **Attention:** The repository includes kernels for various attention mechanisms, such as Multi-Head Attention (MHA), Grouped-Query Attention (GQA), and both causal and non-causal variants. Both forward and backward passes are implemented.

*   **Memory-Bound Kernels:** Kernels for memory-bound operations like Rotary Positional Embedding (RoPE) and Layer Normalization are also included.

These kernels are not just proof-of-concepts; they are integrated into training scripts for models like BERT and Llama, demonstrating their practical utility. The project also provides comparisons against other implementations, such as AITensor and PyTorch.

## 5. Benchmarking and Analysis

A significant part of the HipKittens project is its emphasis on performance analysis. The `analysis/` directory contains scripts to benchmark the provided kernels across a range of dimensions and settings. This allows researchers and developers to reproduce the performance results presented in the project's associated paper and to evaluate the effectiveness of the HipKittens primitives.

## 6. Value Proposition and Ecosystem Position

### 6.1 Key Differentiators

HipKittens occupies a unique position in the AMD GPU programming ecosystem:

*   **Anti-Vendor-Lock-In Mission**: Explicitly designed to foster a "multi-silicon future" - an academic/open-source alternative not beholden to AMD's priorities or release cycles
*   **Minimal & Opinionated vs Comprehensive**: Unlike AMD's Composable Kernel (CK) which tries to cover everything, HipKittens deliberately provides minimal primitives with strong opinions on "the right way," trading generality for simplicity and teachability
*   **Educational & Research-First**: Explicitly names and teaches optimization patterns, designed to be understood and modified by researchers, includes full training examples (BERT, Llama) proving real-world utility
*   **Performance Sweet Spot**: Bridges the gap between "easy to use" (PyTorch) and "maximum performance" (hand-written assembly)

### 6.2 The AMD GPU Programming Spectrum

```
Low-level                                              High-level
(Hard, Fast)                                           (Easy, Slower)

Raw HIP/Assembly → HipKittens → Composable Kernel → PyTorch/JAX
```

HipKittens fills the critical gap for researchers and developers who need:
*   Competitive AMD GPU performance (not just NVIDIA fallback)
*   Ability to experiment with novel kernels without vendor bureaucracy
*   Understanding of *why* kernels are fast (not just black-box usage)
*   Freedom from CUDA ecosystem lock-in

### 6.3 Comparison to Alternatives

*   **vs Raw HIP**: Much easier to use while maintaining near-metal performance
*   **vs Composable Kernel (AMD official)**: Simpler, more accessible, better for learning and research
*   **vs Triton/JAX**: Lower-level control, AMD-specific optimizations, but requires separate implementation per vendor
*   **vs ThunderKittens**: Same team, same philosophy, but for NVIDIA/CUDA - patterns transfer but code doesn't

## 7. Conclusion

HipKittens is a valuable contribution to the field of high-performance computing for AI. By providing a set of well-designed primitives for AMD GPUs, it helps to bridge the gap between hardware-specific programming and high-level deep learning frameworks. The project's focus on performance, its comprehensive set of example kernels, and its open-source nature make it a promising tool for researchers and practitioners who want to unlock the full potential of AMD's AI hardware. It represents a significant step towards a more diverse and competitive hardware ecosystem for artificial intelligence.
