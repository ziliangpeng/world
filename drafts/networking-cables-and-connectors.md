# Networking Cables and Connectors

## Overview
This document covers cable and connector technology used in modern networking, from high-performance datacenter interconnects to residential internet access infrastructure.

---

## Datacenter & High-Performance Networking

### InfiniBand
**Physical Media:**
- Copper: QSFP (Quad Small Form-factor Pluggable) cables for short distances (1-7m typically)
  - Direct Attach Copper (DAC) cables - passive or active
  - Lower cost, higher power consumption, distance limited
- Fiber: QSFP optical transceivers for long distances
  - Single-mode fiber (SMF): 10km-40km+
  - Multi-mode fiber (MMF): 100m-300m

**Connector Types:**
- QSFP28 (100 Gbps)
- QSFP56 (200 Gbps)
- QSFP-DD (400 Gbps, Double Density)
- OSFP (800 Gbps, Octal Small Form Factor Pluggable)

**Speed Evolution:**
- SDR: 8 Gbps (obsolete)
- DDR: 16 Gbps (obsolete)
- QDR: 32 Gbps
- FDR: 56 Gbps
- EDR: 100 Gbps
- HDR: 200 Gbps
- NDR: 400 Gbps
- XDR: 1000 Gbps (future)

### RoCE (RDMA over Converged Ethernet)
**Physical Media:**
- Uses standard Ethernet physical layer
- Same cable types as Ethernet but requires lossless Ethernet (DCB/PFC)
- Copper: Twinax DAC cables
- Fiber: SR/LR optical modules

**Connector Types:**
- SFP/SFP+: 1-10 Gbps
- SFP28: 25 Gbps
- QSFP28: 100 Gbps (4x25G)
- QSFP56: 200 Gbps (4x50G or 2x100G)
- QSFP-DD: 400 Gbps (8x50G)

**Advantages over InfiniBand:**
- Uses Ethernet infrastructure (switches, cables)
- Lower cost at scale
- Easier integration with existing networks

### Ethernet (Datacenter)
**Copper Cables:**
- Cat5e: Up to 1 Gbps (obsolete in datacenters)
- Cat6/6a: Up to 10 Gbps, 100m max
- Cat7: Up to 10 Gbps, better shielding
- Cat8: Up to 40 Gbps, 30m max (datacenter use)
- DAC (Direct Attach Copper): Cost-effective for ToR to server (1-7m)

**Fiber Cables:**
- Multi-mode (OM3/OM4/OM5): 100m-400m, datacenter standard
  - OM3: up to 100 Gbps at 100m
  - OM4: up to 100 Gbps at 150m
  - OM5: optimized for SWDM (Short Wavelength Division Multiplexing)
- Single-mode (OS1/OS2): 10km-80km+, long distance
  - OS2: lower attenuation, preferred for new deployments

**Optical Transceivers:**
- SR (Short Range): Multi-mode, 100-400m
- LR (Long Range): Single-mode, 10km
- ER (Extended Range): Single-mode, 40km
- ZR (Extended Extended Range): Single-mode, 80km+

**Speed Standards:**
- 1GbE: 1000BASE-T (Cat5e/6), 1000BASE-SX/LX (fiber)
- 10GbE: 10GBASE-T (Cat6a), 10GBASE-SR/LR (fiber)
- 25GbE: SFP28, popular for modern servers
- 40GbE: QSFP+
- 100GbE: QSFP28 (4x25G), 100GBASE-SR4/LR4
- 200GbE: QSFP56 (4x50G or 2x100G)
- 400GbE: QSFP-DD (8x50G), OSFP
- 800GbE: QSFP-DD800, OSFP (emerging)

---

## Residential & Enterprise Access

### DSL (Digital Subscriber Line)
**Physical Media:**
- Existing telephone copper wire (twisted pair)
- RJ-11 connector at customer premises
- Maximum distance: ~5km from DSLAM (degrades with distance)

**Technology Generations:**
- ADSL (Asymmetric DSL): 1-8 Mbps down, 64-1024 Kbps up
- ADSL2+: up to 24 Mbps down
- VDSL: 50-100 Mbps (shorter distances)
- VDSL2: up to 200 Mbps (very short distances, <500m)
- G.fast: up to 1 Gbps (<250m, rarely deployed)

**Limitations:**
- Distance sensitive (copper attenuation)
- Shared medium issues (crosstalk)
- Being phased out in favor of fiber

### Cable Internet (DOCSIS)
**Physical Media:**
- Coaxial cable (originally for cable TV)
- F-connector or BNC connector
- HFC (Hybrid Fiber-Coaxial): Fiber to neighborhood node, coax to home

**Technology Generations:**
- DOCSIS 1.0: 40 Mbps down
- DOCSIS 2.0: 40 Mbps down, 30 Mbps up
- DOCSIS 3.0: 1 Gbps down, 200 Mbps up (channel bonding)
- DOCSIS 3.1: 10 Gbps down, 1-2 Gbps up
- DOCSIS 4.0: 10 Gbps symmetrical (Full Duplex)

**Characteristics:**
- Shared bandwidth in neighborhood (contention)
- Asymmetric speeds (historically)
- Widely available in US and Europe

### Fiber to the Premises (FTTP/FTTH)
**Physical Media:**
- Single-mode optical fiber
- SC, LC, or FC connectors (SC/APC common in residential)
- ONT (Optical Network Terminal) at customer premises

**Technologies:**
- PON (Passive Optical Network) - most common for residential
  - GPON: 2.5 Gbps down, 1.25 Gbps up (shared among 32-128 users)
  - XG-PON: 10 Gbps down, 2.5 Gbps up
  - XGS-PON: 10 Gbps symmetrical
  - NG-PON2: 40-80 Gbps (wavelength division multiplexing)
- Active Ethernet: Dedicated fiber per customer (less common, higher cost)

**Advantages:**
- Future-proof (multi-gigabit capable)
- Low latency
- No electromagnetic interference
- Symmetric speeds possible

### Ethernet over Twisted Pair (Business/Enterprise)
**Physical Media:**
- Cat5e/Cat6/Cat6a copper cables
- RJ-45 connector (8P8C modular connector)
- Maximum distance: 100m per segment

**Common Deployments:**
- Office buildings: Cat6/Cat6a for 1-10 Gbps
- Metro Ethernet: Carrier-grade Ethernet for business connectivity
- DIA (Dedicated Internet Access): 100 Mbps to 10 Gbps

---

## Global Infrastructure Deployment

### United States
**Fiber Availability:**
- ~50% of households have access to fiber (as of 2024)
- Concentrated in urban and suburban areas
- Rural fiber deployment accelerating (government subsidies)
- Major providers: AT&T Fiber, Verizon Fios, Google Fiber, regional providers

**Cable (DOCSIS) Availability:**
- ~90% of households
- Dominant in suburban/urban markets
- Major providers: Comcast/Xfinity, Charter/Spectrum, Cox

**DSL Availability:**
- Legacy technology, being decommissioned
- Still used in rural areas without alternatives
- AT&T, CenturyLink phasing out in favor of fiber/5G

**Next-Gen:**
- 5G Fixed Wireless competing with fiber/cable in some markets
- Starlink satellite for very rural areas

### Europe
**Fiber Availability:**
- Highly variable by country
- Spain, Portugal, Sweden: >80% FTTH coverage
- UK, Germany, Italy: 20-40% (rapidly growing)
- Eastern Europe: varies widely, some cities well-covered

**Cable Availability:**
- Strong in UK (Virgin Media), Netherlands, Belgium
- Less prevalent than in US
- Many markets skipped cable, went directly to fiber

**DSL:**
- Still significant in Germany, UK (BT)
- VDSL2 vectoring extends life in some markets

### Asia-Pacific
**Fiber Availability:**
- South Korea: >95% FTTH, world leader
- Japan: >85% FTTH
- Singapore: Nearly 100% FTTH nationwide
- China: Extensive urban fiber, 500M+ FTTH subscribers
- India: Rapid fiber rollout in cities (Jio, Airtel)

**Characteristics:**
- Many markets leapfrogged DSL/cable to fiber
- Government-led infrastructure investments
- High-density urban areas favor fiber economics

### Other Regions
**Middle East:**
- UAE, Qatar: High fiber penetration in cities
- Israel: Strong fiber deployment

**Latin America:**
- Brazil, Chile: Growing fiber in urban areas
- Many areas still on DSL or cable

**Africa:**
- Urban fiber growing (South Africa, Kenya, Nigeria)
- Mobile/wireless often primary internet access
- Limited fixed-line infrastructure in rural areas

---

## Technology Comparison

### Speed Comparison (Typical Residential/Business)
| Technology | Typical Speed | Max Theoretical | Distance Limit |
|------------|---------------|-----------------|----------------|
| DSL (ADSL2+) | 10-24 Mbps | 24 Mbps | ~3-5 km |
| VDSL2 | 50-100 Mbps | 200 Mbps | ~1 km |
| Cable (DOCSIS 3.0) | 100-300 Mbps | 1 Gbps | N/A (HFC) |
| Cable (DOCSIS 3.1) | 1 Gbps | 10 Gbps | N/A (HFC) |
| Fiber (GPON) | 1 Gbps | 2.5 Gbps | ~20 km |
| Fiber (XGS-PON) | 1-10 Gbps | 10 Gbps | ~20 km |
| Ethernet (Cat6) | 1 Gbps | 10 Gbps | 100 m |

### Cost Comparison (Infrastructure)
- **Fiber**: High upfront cost, low maintenance, future-proof
- **Cable**: Moderate cost (if coax exists), upgradeable to DOCSIS 4.0
- **DSL**: Low cost (copper exists), limited performance, being retired
- **5G FWA**: Moderate cost, quick deployment, capacity constraints

---

## Future Trends

### Datacenter
- 800G and 1.6T Ethernet becoming standard for AI/ML clusters
- Co-packaged optics (CPO): Optical transceivers integrated with switch chips
- Linear Pluggable Optics (LPO): Lower power consumption
- Copper remaining relevant for short-distance (<5m) due to cost

### Residential
- 10 Gbps residential fiber (XGS-PON) becoming common in competitive markets
- DOCSIS 4.0 allowing cable to compete with fiber
- 5G/6G Fixed Wireless as alternative in some markets
- Wi-Fi 7 (802.11be) as limiting factor for multi-gig home networking

### Emerging Technologies
- Hollow-core fiber: Lower latency, higher speeds (research phase)
- Terahertz wireless: Short-range ultra-high-speed (>100 Gbps)
- Li-Fi: Optical wireless networking

---

## References & Standards Bodies
- IEEE 802.3: Ethernet standards
- InfiniBand Trade Association (IBTA)
- CableLabs: DOCSIS standards
- ITU-T: DSL and PON standards (G.987, G.989, G.9807)
- Fiber Broadband Association: FTTH deployment tracking
