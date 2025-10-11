# History of PC Expansion Buses

This document tracks the history of the primary expansion buses used in PCs to connect peripherals, leading up to the modern PCIe standard.

## Pre-PCIe Era: A Fragmented, Parallel World

Before PCIe unified everything, the landscape was a mix of different standards, all of which were **parallel** buses. Parallel buses send data across many wires simultaneously, a design that becomes difficult to synchronize at very high speeds.

### ISA (Industry Standard Architecture)

*   **Era:** 1980s - Mid 1990s
*   **Description:** The original 16-bit expansion bus from the early IBM PCs. It was the standard for many years but was very slow by modern standards.

### PCI (Peripheral Component Interconnect)

*   **Era:** Mid 1990s - Mid 2000s
*   **Description:** Introduced in 1992, PCI replaced the aging ISA bus. It was a 32-bit or 64-bit shared parallel bus.
*   **Limitation:** All devices on the bus (sound cards, network cards, etc.) had to share the total available bandwidth (typically 133 MB/s). This "party line" design became a significant bottleneck, especially for graphics cards.

### AGP (Accelerated Graphics Port)

*   **Era:** Late 1990s - Early 2000s
*   **Description:** Introduced in 1997 specifically to solve the PCI bottleneck for graphics. AGP was a dedicated, point-to-point port just for the graphics card.
*   **Limitation:** While much faster than PCI for graphics (up to 2.1 GB/s for AGP 8x), it was still a parallel bus and eventually hit its own speed limits.

## The Shift to Serial: PCIe

The physical and synchronization limits of these parallel buses led to the development of PCIe.

*   **PCIe (PCI Express)** introduced a **serial** architecture. Instead of a wide, slow bus, it uses one or more very high-speed "lanes." This design scales much better in frequency.
*   PCIe was so successful that it replaced both the AGP slot and the PCI slots, unifying all peripheral connections under a single, scalable standard.
