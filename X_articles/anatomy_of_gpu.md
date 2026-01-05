# 深入淺出 GPU 架構與 CUDA 編程：以 NVIDIA H100 為例

## GPU 硬件架構概覽

GPU由多個Streaming Multiprocessors (SM)組成，每個SM是獨立的處理單元。以NVIDIA的旗艦**H100 (SXM5版本)**為例，它擁有**132個SM**。

### SM內部組成

每個SM包含多種類型的計算核心。H100的SM採用**4-Way Partitioned設計**：
- **Partition結構**：每個SM分為**4個Partition**，每個Partition有**32個FP32 CUDA Cores**
- **CUDA Cores**：通用運算核心，整個H100共有16,896個CUDA Cores
- **Tensor Cores**：專門加速矩陣乘加運算(A×B+C)的專用單元，每個H100 SM有**4個第四代Tensor Cores**，對深度學習至關重要
- **Warp Scheduler**：每個Partition有自己的Warp Scheduler，負責調度warp到32個CUDA Cores上執行

**關鍵點**：由於每個Partition有32個CUDA Cores，啱啱好可以跑一個warp嘅32個thread，實現SIMT並行。呢個4x32嘅結構直接決定咗CUDA編程模型入面warp嘅大小同調度方式。

## GPU內存層次結構

GPU採用多層次內存架構來平衡速度和容量，從快到慢依次是：

### Registers

每個Thread最快的私有存儲空間，用於存放循環變量、地址計算等中間值。H100每個SM有65,536個32-bit寄存器，每個Thread最多可使用**255個Registers**。Register數量是有限資源，用太多會降低SM同時駐留的Thread數量（影響occupancy），當Register不夠用時，編譯器會將變量"spill"到Local Memory。

### Local Memory

這是一個容易誤解的概念。雖然名字叫"Local"，但它**物理上位於Global Memory (HBM)中**，只是邏輯上每個Thread私有。當Register溢出、定義大數組或使用動態索引時，編譯器會將變量放到Local Memory。由於它實際在HBM中，訪問速度和Global Memory一樣慢，會嚴重影響性能。

### L1 Cache與Shared Memory

每個SM擁有**256KB**的統一L1/Shared Memory。Shared Memory是可編程的L1 cache，你可以明確控制其內容，而L1 cache由硬件自動管理。在H100上，最多可配置**228KB**作為Shared Memory。Shared Memory是同一Block內線程通信和數據重用的關鍵。

### L2 Cache

H100擁有**50MB**的L2 Cache，所有SM共享。它作為SM和全局內存之間的中間層，延遲約200個週期。所有寫入全局內存的操作都會通過L2進行同步。

### Global Memory (HBM3)

H100配備**80GB的HBM3內存**，帶寬高達**3.35 TB/s**。這是GPU的主存儲，容量最大但延遲最高，是性能優化的主要瓶頸。

## CUDA編程模型

### Thread、Block與Warp

CUDA執行模型基於層次結構：
- **Thread**: 最基本的執行單元，對應一個CUDA Core的操作
- **Warp**: **32個Thread**組成一個Warp，是GPU調度的基本單位
- **關鍵對應**：由於H100每個Partition有32個CUDA Cores，一個Warp會被**schedule到一個Partition上**，32個Thread啱啱好跑在32個CUDA Cores上面，實現SIMT（Single Instruction, Multiple Threads）並行
- **Block**: 多個Warp組成Thread Block，分配到單個SM執行，H100每個SM最多同時處理**32個Blocks**

當Warp內線程因條件分支走不同路徑時，會發生Warp Divergence，導致性能下降。

### 異步執行與Streams

現代CUDA編程不僅是launch kernel等結果，更要充分利用CPU-GPU並行性。**Streams**允許將kernel launch同memory copy重疊，隱藏數據傳輸延遲。要實現真正異步傳輸，需用`cudaMallocHost`分配pinned memory，否則傳輸會被block。在H100上，可同時進行HBM→GPU copy、GPU compute、GPU→HBM copy，將整體吞吐量提升2-3倍。這對大規模數據處理和流水線式計算特別重要。

### 多GPU編程

H100通常不會單獨使用，而是在多GPU系統中。**NCCL (NVIDIA Collective Communications Library)** 提供優化的多GPU通信原語（AllReduce、Broadcast等），對深度學習訓練至關重要。編程模型可選single-threaded控制所有GPU、multi-threaded（每GPU一個thread）或multi-process（MPI+NCCL）。在H100上，NVLink提供高達900GB/s的GPU-to-GPU帶寬，遠超PCIe，寫multi-GPU code時要充分利用這些高速互連。

### 內存訪問優化

#### Memory Coalescing

當Warp中32個線程訪問Global Memory時，如果訪問連續地址，GPU可以合併為少數幾個內存事務，大幅提升帶寬利用率。不連續的訪問會導致性能遠低於峰值。

#### Bank Conflicts

Shared Memory被分為**32個Banks**（對應Warp大小）。當同一Warp中多個線程訪問同一Bank的不同地址時，會發生Bank Conflict，導致訪問序列化。解決方法是通過Padding或重新映射索引來讓線程訪問不同Banks。

#### Tiling技術

為充分利用L1/L2 cache和Shared Memory的高速度，將大問題分解為小塊(tiles)處理。每個Tile的數據載入Shared Memory後可被多次重用，減少Global Memory訪問次數。利用H100的**228KB Shared Memory**，可以將矩陣切成較大的Tiles，讓Tensor Cores在高速緩存中完成計算，再寫回HBM。

### 性能分析工具

要寫出高性能CUDA code，必須借助專業工具定位瓶頸。**Nsight Compute**提供詳細的kernel-level性能指標，可檢測每行code的memory utilization和指令效率，自動識別bank conflict、memory coalescing問題和register spilling。**Nsight Systems**則提供系統級分析，幫助理解CPU-GPU互動、異步執行和數據傳輸開銷。在後續文章中，我們將詳細介紹如何使用這些工具進行性能調試和優化。
