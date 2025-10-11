# AI Datacenter Networking

This document covers network topology, fabric architecture, and other networking aspects for AI datacenters. For connectivity technologies (RoCE, RDMA, NVLink, etc.), see [connectivity/overview.md](connectivity/overview.md).

---

## Network Topology

### Common Datacenter Topologies

#### Fat-Tree Topology
- **Structure**: Multi-tier Clos network with full bandwidth between any two nodes
- **Characteristics**:
  - Multiple layers: access (ToR), aggregation, and spine/core layers
  - Non-blocking: Equal bandwidth between any source-destination pair
  - Oversubscription ratio: Typically 1:1 at spine layer, may be 2:1 or 4:1 at aggregation
- **Use Case**: Traditional datacenter standard, scales to thousands of nodes
- **Pros**: Proven design, predictable performance, good for diverse workloads
- **Cons**: High switch count, complex cabling, high power consumption

#### Spine-Leaf (2-Tier Clos)
- **Structure**: Simplified fat-tree with only two layers
- **Characteristics**:
  - Every leaf switch connects to every spine switch
  - No leaf-to-leaf or spine-to-spine connections
  - 1-2 hop maximum latency between any two servers
  - Common oversubscription: 2:1 to 4:1
- **Use Case**: Modern cloud datacenters, AI training clusters
- **Pros**: Simple, scalable, low latency, easy to expand horizontally
- **Cons**: Limited by spine port density, doesn't scale beyond ~10k servers without hierarchy

#### Dragonfly Topology
- **Structure**: Hierarchical design with local groups connected via global links
- **Characteristics**:
  - Routers organized into groups, fully connected within each group
  - Groups connected by high-bandwidth global channels
  - Minimal path diameter (typically 2-3 hops)
  - Designed for high-radix routers
- **Use Case**: HPC supercomputers, large-scale research clusters
- **Pros**: Excellent scalability, lower cost per port at very large scale
- **Cons**: Complex routing, load balancing challenges, adaptive routing required

#### Torus/Mesh Topology
- **Structure**: Grid arrangement with wraparound connections (torus) or without (mesh)
- **Characteristics**:
  - Nodes connected in 2D/3D/nD grid pattern
  - Direct neighbor connections only
  - Path length scales with grid dimensions
- **Use Case**: Specialized HPC systems, Google TPU Pods
- **TPU Pod Example**:
  - 3D torus topology for v4/v5 TPU Pods
  - 2D torus for v3 TPU Pods
  - Optimized for AllReduce collective patterns
- **Pros**: Predictable, good for nearest-neighbor communication, simple wiring
- **Cons**: Higher diameter than fat-tree, not ideal for all-to-all traffic

#### Ring Topology
- **Structure**: Circular arrangement where each node connects to exactly two neighbors
- **Characteristics**:
  - Simple point-to-point connections
  - Data flows in one or both directions around the ring
  - Path length scales linearly with ring size
- **Use Case**: Single-node multi-GPU systems, NCCL ring algorithm
- **Example**: 8x GPU server with GPUs connected in a ring via NVLink
- **Pros**: Simple, minimal wiring, works well for small GPU counts
- **Cons**: Poor scaling beyond 8-16 GPUs, high latency for distant nodes

---

## Network Architecture Design

### Oversubscription

**Definition**: Ratio of total bandwidth available from access layer to bandwidth available to core/spine layer.

**Common Ratios**:
- **1:1** (Non-blocking): Full bandwidth available, expensive, ideal for AI training
- **2:1**: Common for latency-sensitive workloads, moderate cost
- **4:1**: Typical for general datacenter use, cost-effective
- **8:1 or higher**: Web servers, storage, less critical workloads

**AI Training Considerations**:
- Collective operations (AllReduce) generate significant east-west traffic
- Oversubscription creates bottlenecks during gradient synchronization
- Modern AI clusters trend toward 1:1 or 2:1 oversubscription
- Trade-off between cost and training performance

### East-West vs North-South Traffic

**North-South Traffic**: Between datacenter and external networks (internet, other DCs)
- Traditional datacenters: 70-80% of total traffic
- Examples: User requests, API calls, data ingestion

**East-West Traffic**: Between servers within the datacenter
- Modern datacenters: 70-80% of total traffic
- AI training clusters: 90%+ of traffic is east-west
- Examples: GPU-to-GPU communication, distributed training, microservices

**AI Training Traffic Patterns**:
- Dominated by collective operations (AllReduce, AllGather, ReduceScatter)
- Regular, predictable bursts synchronized across all GPUs
- Requires low latency and high bandwidth
- Benefits from optimized topology (e.g., rail-optimized networks)

### Rail-Optimized Networks

**Concept**: Partition network into independent "rails" to minimize contention.

**Design**:
- Each server has multiple NICs (e.g., 4x or 8x NICs)
- Each NIC connects to a separate network fabric (rail)
- Rails operate independently with no inter-rail communication
- Communication patterns pinned to specific rails

**Benefits**:
- Reduced congestion and head-of-line blocking
- Improved predictability and tail latency
- Better isolation between different traffic types
- Higher effective bandwidth utilization

**Example**:
- 8x H100 server with 8x 400GbE NICs
- Each NIC connects to separate spine-leaf fabric
- NCCL configured to use specific rails for specific GPU pairs
- Result: 3.2 Tbps total bisection bandwidth with minimal contention

**Use Case**: Large-scale AI training clusters (1000+ GPUs)

---

## Network Switch Architecture

### Merchant Silicon vs Custom ASICs

#### Merchant Silicon
**Examples**: Broadcom Tomahawk, Trident; Nvidia Spectrum

**Characteristics**:
- Off-the-shelf switching chips
- Multiple vendors use same silicon
- Standard features, proven designs
- Lower development cost

**Pros**:
- Lower cost, proven reliability
- Broad vendor ecosystem
- Regular updates and new generations
- Well-understood performance characteristics

**Cons**:
- Limited customization
- May not optimize for specific workloads
- Feature parity across vendors

#### Custom ASICs
**Examples**: Google Jupiter, AWS Nitro, Microsoft Catapult

**Characteristics**:
- Proprietary switching silicon designed in-house
- Optimized for specific datacenter needs
- Tight integration with infrastructure

**Pros**:
- Tailored performance characteristics
- Custom features (e.g., in-network compression, custom congestion control)
- Potential cost advantages at hyperscale
- Vertical integration benefits

**Cons**:
- High development cost (billions of dollars)
- Long development cycles (3-5 years)
- Requires massive scale to justify investment
- Limited to largest hyperscalers

### In-Network Computing

**Concept**: Offload computation from endpoints to network switches/NICs.

#### SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)
- **Vendor**: NVIDIA/Mellanox
- **Function**: In-network reduction for collective operations
- **How it works**:
  - InfiniBand switches perform reduction operations (sum, max, etc.)
  - Reduces network traffic and improves AllReduce performance
  - NCCL can leverage SHARP automatically when available
- **Performance**: Up to 5x improvement in AllReduce latency for large clusters

#### Programmable Switches
- **Examples**: Barefoot Tofino, Broadcom Tomahawk programmable variants
- **Language**: P4 (Programming Protocol-independent Packet Processors)
- **Use Cases**:
  - Custom congestion control
  - Load balancing algorithms
  - Telemetry and monitoring
  - In-network aggregation

---

## Latency Considerations

### Components of Network Latency

**Total Latency = Serialization + Propagation + Switching + Queueing + Processing**

1. **Serialization Latency**: Time to put bits on the wire
   - Depends on bandwidth and packet size
   - Example: 1500-byte packet on 100 Gbps = 0.12 μs

2. **Propagation Latency**: Speed of light in fiber/copper
   - ~5 ns/meter (speed of light in fiber ≈ 0.67c)
   - Example: 100 meters = 0.5 μs

3. **Switching Latency**: Time spent in switch
   - Cut-through: 30-300 ns (modern low-latency switches)
   - Store-and-forward: 1-10 μs
   - Port-to-port: InfiniBand ~100 ns, RoCE ~230 ns

4. **Queueing Latency**: Time waiting in switch buffers
   - Highly variable, depends on congestion
   - Can dominate total latency under load
   - Priority Flow Control (PFC) creates backpressure to minimize drops

5. **Processing Latency**: NIC and software stack
   - Hardware RDMA: 1-2 μs
   - Kernel TCP/IP: 20-50 μs

### Latency Budget Example (InfiniBand, 2-hop)

| Component | Latency |
|-----------|---------|
| NIC processing (sender) | 0.3 μs |
| Serialization (1 KB packet, 200 Gbps) | 0.04 μs |
| Propagation (50m cable) | 0.25 μs |
| Switch 1 (cut-through) | 0.1 μs |
| Propagation (50m cable) | 0.25 μs |
| Switch 2 (cut-through) | 0.1 μs |
| Propagation (50m cable) | 0.25 μs |
| NIC processing (receiver) | 0.3 μs |
| **Total** | **~1.6 μs** |

**Key Insight**: At datacenter scale, propagation delay dominates. This is why proximity (rack locality, pod locality) matters for latency-sensitive workloads.

---

## Congestion Control

### Priority Flow Control (PFC)

**Purpose**: Create lossless Ethernet for RoCE by preventing buffer overflows.

**Mechanism**:
- IEEE 802.1Qbb standard
- Per-priority PAUSE frames
- Backpressure from receiver to sender
- Prevents packet drops at the cost of increased latency

**Configuration**:
- Applied to specific traffic classes (typically DSCP/CoS marked)
- Must be configured on every switch in the path
- Requires careful tuning to avoid deadlocks and congestion spreading

**Challenges**:
- PFC storms: Congestion can spread backward through network
- Head-of-line blocking: One slow flow can block others
- Deadlocks: Circular dependencies in network paths

### Explicit Congestion Notification (ECN)

**Purpose**: Signal congestion before buffers fill, enabling proactive rate reduction.

**Mechanism**:
- IEEE 802.1Qau and RFC 3168
- Switches mark packets when queues reach threshold
- Receivers echo ECN bits back to senders
- Senders reduce transmission rate in response

**Benefits**:
- Prevents PFC activation under normal conditions
- Lower latency than PFC-based flow control
- Maintains high throughput while avoiding congestion

**Tuning Parameters**:
- ECN marking threshold (typically 50-200 KB buffer occupancy)
- PFC headroom (buffer reserved for in-flight packets)
- Priority queue configuration

**Best Practice**: Use ECN as primary congestion signal, PFC as safety net.

---

## Cabling and Physical Layer

### Cable Types

#### Copper (DAC - Direct Attach Copper)
- **Distance**: 1-7 meters
- **Use Case**: Intra-rack, ToR to servers
- **Pros**: Lower cost, lower power, lower latency
- **Cons**: Limited distance, heavier, less flexible

#### Active Optical Cable (AOC)
- **Distance**: 1-100 meters
- **Use Case**: Rack-to-rack within same row
- **Pros**: Lightweight, flexible, no transceivers needed
- **Cons**: Fixed length, cannot repair/replace optics

#### Optical Transceiver + Fiber
- **Distance**: 100 meters to 10+ km (depends on optic type)
- **Types**:
  - **SR (Short Range)**: Multi-mode fiber, 100-400m
  - **LR (Long Range)**: Single-mode fiber, 10km
  - **ER (Extended Range)**: Single-mode fiber, 40km
- **Use Case**: Inter-row, inter-building, long-haul
- **Pros**: Flexible, replaceable optics, long distance
- **Cons**: Higher cost, higher power

### Transceiver Technologies

#### QSFP (Quad Small Form-factor Pluggable)
- **Variants**:
  - **QSFP28**: 100 Gbps (4x 25 Gbps lanes)
  - **QSFP56**: 200 Gbps (4x 50 Gbps lanes)
  - **QSFP-DD**: 400 Gbps (8x 50 Gbps lanes, double-density)
  - **QSFP112**: 800 Gbps (8x 100 Gbps lanes)

#### OSFP (Octal Small Form-factor Pluggable)
- **Bandwidth**: 400-800 Gbps (8x 50-100 Gbps lanes)
- **Power**: Higher power budget than QSFP-DD
- **Use Case**: 800 GbE and future higher speeds

**Key Insight**: Transceiver and cable costs can be 30-50% of total network infrastructure cost.

---

## Network Telemetry and Monitoring

### Key Metrics for AI Workloads

#### Performance Metrics
- **Throughput**: Achieved bandwidth per link, per rail, per job
- **Latency**: P50, P99, P99.9 latency for different message sizes
- **Packet Loss**: Should be near-zero for RoCE/InfiniBand
- **ECN Marked Packets**: Indicator of congestion

#### Health Metrics
- **Link Errors**: CRC errors, symbol errors indicate physical layer issues
- **PFC Pause Frames**: High PFC rate indicates congestion or misconfig
- **Buffer Occupancy**: Queue depths at switches
- **Retransmissions**: RoCE/InfiniBand retransmit rate

#### Collective Performance
- **NCCL AllReduce Time**: Per-iteration collective operation latency
- **Algorithmic Bandwidth**: Effective bandwidth (accounts for message size overhead)
- **Bus Bandwidth**: Normalized bandwidth for collective operations

### Monitoring Tools

- **NVIDIA DCGM**: GPU telemetry including NVLink and network metrics
- **Prometheus + Grafana**: Time-series metrics and dashboards
- **NCCL Tests**: Benchmark collective operation performance
- **ib_write_bw / ib_send_bw**: InfiniBand bandwidth tests
- **perftest suite**: Comprehensive RDMA benchmarking
- **What About Network (WAN)**: Facebook's network monitoring tool (open source)

---

## TODO / Future Research

- Network performance comparison: Google TCPX vs AWS EFA vs Azure InfiniBand
- Detailed fabric design for 10k+ GPU clusters
- Storage network integration (GPUDirect Storage over network)
- Multi-tenancy and network isolation in shared AI clusters
- Network power consumption analysis
- Ultra Ethernet Consortium developments
- Optical circuit switching for AI workloads

---

*Last updated: 2025-10-11*
