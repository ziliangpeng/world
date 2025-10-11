# Google Network Infrastructure Stack (Outline)

## Purpose

Outline the major technologies in Google's data center networking stack, from physical fabrics to application-facing services.

## Layers to Detail

1. Physical fabric generations (Firehose, Jupiter, Andromeda, etc.)
2. NICs, offload engines, and CPU/GPU integration (gVNIC, Hypercompute)
3. Control planes (Andromeda SDN, Maglev load balancing)
4. Software/services interfacing with the network (B4 WAN, Cloud TPU/GPUDirect integrations)
5. Operational tooling, observability, and SLO management

## Open Questions

- What are the bandwidth/latency targets per generation?
- How do TCPX and gRPC variants leverage the fabric?
- What hardware/software co-designs enable GPU direct access?

## Next Steps

- Collect public papers/blog posts on each layer
- Build comparison tables vs. other hyperscalers (AWS, Azure, Meta)
- Identify areas where internal naming overlaps (e.g., Jupiter vs. Andromeda roles)
