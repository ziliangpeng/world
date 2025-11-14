# Neocloud Infrastructure Legacy Burden  

## What Is Legacy Burden?  
Legacy burden refers to the technical debt and operational overhead that come from maintaining older data centres, networking stacks, orchestration layers, contracts and service portfolios that were not built for modern AI workloads. A cloud provider with legacy burden must support general purpose VMs and 100+ services, maintain backward compatibility and cross‑subsidise non‑AI products. This adds cost and slows the rollout of new GPU architectures.  

Nebius was spun out of Yandex in 2023 and built its AI cloud from a clean sheet. It does not have any legacy data centres or general‑purpose cloud products; its entire stack is purpose‑built for GPU workloads and InfiniBand networking, so it avoids legacy burden entirely.  

## Legacy Burden Across Neoclouds  

| Provider | Legacy burden | Notes |  
|---|---|---|  
| **Nebius** | **None** | Built post‑2022 for AI; no general cloud services; no old hardware or networks to maintain. |  
| **CoreWeave** | **High** | Started as an Ethereum mining company; repurposed mining data centres and networking; must service debt and legacy infrastructure. |  
| **Lambda** | **Moderate** | Evolved from on‑prem server sales and VM‑based research clusters; has to support earlier Lambda Stack deployments and VM layers. |  
| **Crusoe** | **Moderate** | Originated with flare‑gas compute and portable data centres; initial stack was not ML‑optimised. |  
| **Hyperscalers (AWS, Azure, GCP)** | **Very high** | Must support dozens of older services, maintain backward compatibility and subsidise non‑GPU offerings; carry huge architectural and organisational overhead. |  

## Why Lack of Legacy Burden Is a Moat  
- **Lower operating cost:** No need to maintain old data‑centre assets or pay for backward compatibility.  
- **Faster adoption of new GPU designs:** Clean architecture allows rapid deployment of new hardware like H200 and B200 without migration headaches.  
- **Pure focus on AI:** Resources and engineering are focused on GPU workloads, not diverted to other cloud products.  
- **Pricing flexibility:** With fewer overheads and no legacy cross‑subsidies, Nebius can sustain lower pricing while remaining profitable. 
