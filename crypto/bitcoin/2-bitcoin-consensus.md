# Bitcoin Consensus: How the Network Agrees on Truth

## Introduction

In `1-bitcoin-basics.md`, we covered the foundation: blockchain as an immutable ledger, mining as a lottery, and the UTXO model for tracking ownership. But we glossed over a critical question:

**How does a network of thousands of computers, operated by strangers who don't trust each other, agree on a single version of reality?**

This is the **consensus problem**, and it's one of the hardest challenges in distributed systems. Bitcoin's solution—combining economic incentives with cryptographic proof—is what made decentralized digital currency possible for the first time in history.

This file dives deep into the mechanics:
- How validation and mining are separate (and why that matters)
- The cryptographic details of Proof-of-Work
- How difficulty adjusts to maintain 10-minute blocks
- What happens when the network temporarily splits
- Why rewriting history requires overwhelming resources
- How the peer-to-peer network propagates information

Let's start by understanding a crucial separation of concerns.

---

## The Separation of Concerns: Validation vs Mining

One of the most important concepts in Bitcoin is that **validation** and **mining** are two separate processes.

### Validation: Everyone's Job

Every full node in the Bitcoin network validates every transaction and every block independently. They check:

**For transactions:**
- ✓ Does the digital signature match the claimed sender?
- ✓ Do the referenced input UTXOs exist?
- ✓ Have those UTXOs already been spent (double-spend check)?
- ✓ Do the outputs sum to less than or equal to the inputs?
- ✓ Are the scripts (spending conditions) satisfied?

**For blocks:**
- ✓ Does the block hash meet the difficulty target?
- ✓ Are all transactions in the block valid?
- ✓ Does the coinbase transaction (miner reward) follow the rules?
- ✓ Is the block size within limits?
- ✓ Does it correctly reference the previous block?

**If anything fails validation, the node rejects it.** Doesn't matter who sent it or how much work went into mining it—invalid is invalid.

### Mining: A Specialized Lottery

Mining is the process of **competing to write the next block**. Only miners do this. But here's the key:

**Miners can't force the network to accept invalid transactions.**

Even if a miner solves the puzzle (finds a valid nonce), if they include an invalid transaction, every other node will reject the block. The miner wasted electricity for nothing.

### Example: A Malicious Miner

Let's say Bob is a miner. He wants to create 1000 BTC out of thin air and give it to himself.

**Step 1:** Bob creates a transaction: `"Transfer 1000 BTC to Bob's address"`
- No input (no UTXOs being spent)
- Just output: 1000 BTC to Bob

**Step 2:** Bob includes this transaction in a block and starts mining.

**Step 3:** Bob gets lucky and finds a valid nonce! The block hash has enough leading zeros.

**Step 4:** Bob broadcasts the block to the network.

**Step 5:** Other nodes receive Bob's block and validate it:
```
Checking block 850,001 from Bob...
- Block hash: 0000000000000000000123abc... ✓ (valid nonce)
- Coinbase transaction: 3.125 BTC + fees ✓ (correct reward)
- Transaction 1: "Transfer 1000 BTC to Bob"
  - Inputs: (none)
  - Checking UTXO existence... ❌ INVALID (no input UTXO)

BLOCK REJECTED. Invalid transaction detected.
```

**Step 6:** Every node rejects Bob's block. Bob wasted his electricity. His block is ignored.

**The separation of validation and mining is what prevents miners from changing the rules.** Miners can only choose which valid transactions to include and in what order—they can't create money out of thin air or steal others' coins.

---

## Proof-of-Work: The Technical Details

Now let's dive into exactly how mining works at a technical level.

### The Cryptographic Hash Function: SHA-256

Bitcoin uses **SHA-256** (Secure Hash Algorithm, 256-bit) for hashing.

**Properties of SHA-256:**

1. **Deterministic:** Same input always produces same output
   ```
   SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
   (always this exact output)
   ```

2. **Fast to compute:** Takes microseconds on modern hardware

3. **One-way:** Can't reverse the hash to get the input
   ```
   Given: 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
   Cannot determine: "hello"
   Must try all possible inputs
   ```

4. **Avalanche effect:** Tiny input change = completely different output
   ```
   SHA-256("hello")  = 2cf24dba5fb0a30e26e83b2ac5b9e29e...
   SHA-256("hellp")  = 6f8c0b9f2d97f3ed0c4fa3a89a61e3c2...
   (changed one letter, entire hash changed)
   ```

5. **Fixed output size:** Always 256 bits (64 hexadecimal characters), regardless of input size

### What Miners Are Actually Computing

The block header contains several pieces of information:

```
Block Header (80 bytes):
┌─────────────────────────────────────┐
│ Version (4 bytes)                   │
│ Previous Block Hash (32 bytes)      │
│ Merkle Root (32 bytes)*             │
│ Timestamp (4 bytes)                 │
│ Difficulty Target (4 bytes)         │
│ Nonce (4 bytes)                     │
└─────────────────────────────────────┘

*Merkle root = hash of all transactions in the block
```

**The mining puzzle:**
```
Find a nonce such that:
SHA-256(SHA-256(block_header)) < target

Where:
- block_header includes the nonce
- target is a 256-bit number derived from difficulty
- The hash must be LESS than the target (numerically)
```

Bitcoin actually applies SHA-256 **twice** for additional security.

### The Target and Difficulty

The **difficulty** is a measure of how hard it is to find a valid block. The **target** is the actual number miners must beat.

**Relationship:**
```
target = target_max / difficulty

Where:
target_max = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
(the maximum possible target, from Bitcoin's genesis)
```

**Current Bitcoin difficulty (December 2024):**
```
Difficulty: 109,600,000,000,000 (109.6 trillion)

Target (in hex):
0000000000000000000407a00000000000000000000000000000000000000000

This means the hash must start with approximately 20 zero bits.
```

**What this means in practice:**

A valid block hash might look like:
```
00000000000000000001a3f5b2c8d4e9f7a1c3e5b7d9f1a3c5e7b9d1f3a5c7e9
^                 ^ (19 leading zeros in hex = ~76 leading zero bits)
```

An invalid hash (doesn't meet target):
```
0000000000000001a3f5b2c8d4e9f7a1c3e5b7d9f1a3c5e7b9d1f3a5c7e9b1d3f5
^               ^ (only 15 leading zeros)
```

**Probability of finding a valid hash:**
```
P = 1 / difficulty
P = 1 / 109,600,000,000,000
P ≈ 0.00000000000009% per attempt

This is why miners need to try quintillions of hashes.
```

### Hash Rate: The Network's Computing Power

**Hash rate** measures how many hashes the network computes per second.

**Current Bitcoin network hash rate (December 2024):**
```
~700 EH/s (exahashes per second)
= 700,000,000,000,000,000,000 hashes per second
= 700 quintillion hashes per second
```

**To put this in perspective:**
- All the world's supercomputers combined: ~1-2 EH/s
- Bitcoin network: ~700 EH/s
- **Bitcoin's network is 350-700x more powerful than all supercomputers combined** (for this specific computation)

**How long to find a block:**
```
Expected time = (difficulty × 2^32) / hash_rate

Where 2^32 = 4,294,967,296 (number of nonce values in 4 bytes)

With current numbers:
Expected time = (109.6T × 4.29B) / 700 EH/s
≈ 10 minutes (by design)
```

### The Evolution of Mining Hardware

**2009-2010: CPU Mining**
- Satoshi and early adopters mined with regular CPUs
- Hash rate: ~1-10 MH/s per CPU
- Difficulty: 1 (anyone could mine)

**2010-2013: GPU Mining**
- GPUs (graphics cards) are better at parallel computation
- Hash rate: ~100-500 MH/s per GPU
- Difficulty rose to millions

**2013-Present: ASIC Mining**
- ASICs (Application-Specific Integrated Circuits) are chips designed ONLY for SHA-256
- Cannot do anything else (can't run games, browse web, etc.)
- Modern ASIC (Antminer S21, 2024): ~335 TH/s (terahashes per second)
- Cost: ~$5,000
- Power consumption: ~3,500 watts

**Why ASICs won:**
- 100,000x more efficient than CPUs for SHA-256
- Economics: if you're spending on electricity, might as well maximize hashing

**The arms race:**
As more miners joined with better hardware, difficulty increased, which incentivized even better hardware. This is now a specialized industry dominated by large mining farms in regions with cheap electricity (China historically, now USA, Kazakhstan, Canada, etc.).

### Energy Consumption

**Bitcoin's annual energy consumption (2024):**
```
~150 TWh/year (terawatt-hours)
≈ 0.6% of global electricity consumption
≈ Similar to Argentina or the Netherlands
```

**Environmental considerations:**
- ~56% of Bitcoin mining uses renewable energy (2024 estimates)
- Miners seek cheapest electricity (often renewable: hydroelectric, geothermal)
- Debate: Is this waste or securing a $2T asset?

**For context:**
- Global banking system: ~260 TWh/year
- Gold mining: ~240 TWh/year
- YouTube: ~244 TWh/year

The energy debate is complex and politically charged. The technical reality is that Proof-of-Work intentionally uses energy as a security mechanism—it makes attacks expensive in the real world, not just in Bitcoin terms.

---

## Difficulty Adjustment: Maintaining 10-Minute Blocks

Bitcoin targets an average block time of 10 minutes. But as miners join (or leave) the network, the total hash rate changes. How does Bitcoin maintain the 10-minute target?

### The Adjustment Mechanism

**Every 2016 blocks** (approximately 2 weeks), Bitcoin recalculates the difficulty.

**Formula:**
```
new_difficulty = old_difficulty × (actual_time / target_time)

Where:
- target_time = 2016 blocks × 10 minutes = 20,160 minutes (2 weeks)
- actual_time = how long the last 2016 blocks actually took
```

**Limits:**
To prevent extreme swings, adjustment is capped at:
- Maximum increase: 4x per adjustment
- Maximum decrease: 0.25x per adjustment (can't drop more than 75%)

### Example Calculation

**Scenario:** Hash rate increased (miners joined), so blocks were found faster.

```
Old difficulty: 100,000,000,000,000
Target time:    20,160 minutes
Actual time:    15,120 minutes (blocks came 25% faster than expected)

new_difficulty = 100T × (15,120 / 20,160)
new_difficulty = 100T × 0.75
new_difficulty = 75T

Difficulty DECREASED by 25%
(Wait, decreased? Yes—because blocks were coming TOO FAST)
```

**Wait, that seems backwards?**

Actually, I made an error in the scenario. Let me correct:

**If blocks came FASTER (15,120 min instead of 20,160 min):**
This means we need to make it HARDER, so:
```
new_difficulty = old_difficulty × (20,160 / 15,120)
new_difficulty = old_difficulty × 1.33
Difficulty INCREASED by 33%
```

**The correct formula:**
```
new_difficulty = old_difficulty × (target_time / actual_time)

If actual_time < target_time → difficulty increases (make it harder)
If actual_time > target_time → difficulty decreases (make it easier)
```

### Historical Difficulty Growth

```
Year    Difficulty              Hash Rate         Event
2009    1                      ~5 MH/s           CPU mining
2010    ~1,000                 ~100 MH/s         GPU mining starts
2013    ~3.5M                  ~25 TH/s          ASIC mining begins
2017    ~923B                  ~7 EH/s           Bull run
2021    ~25T                   ~180 EH/s         ATH at $69k
2024    ~110T                  ~700 EH/s         ETF era
```

Difficulty has increased by a factor of **110 trillion** since Bitcoin's launch.

### What Would Happen Without Adjustment?

**Scenario:** Difficulty frozen at 2009 levels (difficulty = 1).

**If the current network (700 EH/s) mined at 2009 difficulty:**
```
Block time = (difficulty × 2^32) / hash_rate
Block time = (1 × 4.29B) / (700 × 10^18)
Block time ≈ 0.000006 seconds

That's 166,000 blocks PER SECOND.
The entire 21 million Bitcoin supply would be mined in ~20 seconds.
```

**Difficulty adjustment is what makes Bitcoin's monetary policy work.** Without it, the fixed supply and halving schedule would be meaningless.

---

## Chain Forks: When Two Miners Win Simultaneously

Despite the 10-minute average, mining is probabilistic. Sometimes two miners find valid blocks at nearly the same time. What happens?

### How a Fork Occurs

**Scenario:**
1. Miner Alice (in USA) finds Block 850,000 at 12:00:00.000
2. Miner Bob (in China) finds a different Block 850,000 at 12:00:00.123
3. Both blocks are valid (correct nonce, valid transactions)
4. Both miners broadcast their blocks

**Network splits:**
```
                    Block 849,999
                         ↓
           ┌─────────────┴─────────────┐
           ↓                           ↓
    Block 850,000a              Block 850,000b
    (Alice's block)             (Bob's block)
           ↓                           ↓
    50% of nodes                 50% of nodes
    see this first               see this first
```

**Why nodes see different blocks first:**
- Network latency: Alice's block reaches USA nodes first, Bob's reaches China nodes first
- It takes ~1-5 seconds for blocks to propagate globally
- Nodes accept the first valid block they see

### How the Fork Resolves

The rule is simple: **The longest chain wins.**

Miners building on Alice's block and miners building on Bob's block are now in a race to find Block 850,001.

```
Step 1: Fork occurs
    Block 849,999
         ↓
    ┌────┴────┐
    ↓         ↓
  850,000a  850,000b
  (50% mine) (50% mine)

Step 2: Someone finds Block 850,001 (let's say on Alice's chain)
    Block 849,999
         ↓
    ┌────┴────┐
    ↓         ↓
  850,000a  850,000b
    ↓
  850,001
  (Alice's chain is now longer)

Step 3: Network sees the longer chain and switches
    Block 849,999
         ↓
      850,000a ← Everyone builds on this chain now
         ↓
      850,001

    850,000b ← ORPHANED (abandoned)
```

**What happens to nodes that were on Bob's chain:**
1. They receive Block 850,001 (built on Alice's chain)
2. They see that chain A (849,999 → 850,000a → 850,001) is longer than chain B (849,999 → 850,000b)
3. They **reorganize** (reorg):
   - Discard Block 850,000b
   - Accept Block 850,000a and 850,001
   - Move transactions that were only in 850,000b back to the mempool

**Block 850,000b is called an "orphan" or "stale" block.** The miner (Bob) gets no reward—the block reward from an orphaned block is lost.

### What Happens to Transactions in Orphaned Blocks?

**Most transactions are fine:**

If a transaction was in **both** blocks (850,000a and 850,000b), it remains confirmed. Most transactions appear in both because:
- Both miners were drawing from the same mempool
- Popular transactions (high fees) get included by everyone

**Some transactions might be affected:**

**Case 1: Transaction only in the orphaned block**
```
Block 850,000a: [tx1, tx2, tx3, tx4, tx5]
Block 850,000b: [tx1, tx2, tx3, tx6, tx7] ← ORPHANED

tx6 and tx7 go back to mempool (unconfirmed)
They'll likely get mined in Block 850,001 or 850,002
```

**Case 2: Conflicting transactions (potential double-spend)**
```
Alice has 1 BTC UTXO.

Transaction A: Alice → Bob (1 BTC)    [in Block 850,000a]
Transaction B: Alice → Carol (1 BTC)  [in Block 850,000b, orphaned]

Block 850,000a wins → Transaction A is confirmed, Bob gets the BTC
Transaction B is now INVALID (double-spend) and rejected from mempool
Carol doesn't get the BTC
```

**This is why merchants wait for multiple confirmations.** With 1 confirmation, there's a small chance of a natural fork reversing the transaction.

### Probability of Extended Forks

How likely is a fork to last multiple blocks?

**Assumptions:**
- Network split 50/50
- Each side has equal chance of finding the next block

**Probabilities:**
```
Fork lasts 1 block:  50% chance (one side finds the next block)
Fork lasts 2 blocks: 25% chance (both sides find 1 block, still tied)
Fork lasts 3 blocks: 12.5% chance
Fork lasts 4 blocks: 6.25% chance
Fork lasts 6 blocks: 1.56% chance
Fork lasts 10 blocks: 0.098% chance
```

**In practice, most forks resolve within 1-2 blocks (10-20 minutes).**

### Real Example: The March 2013 Fork

On March 11, 2013, a fork occurred that lasted **6 hours** (~24 blocks). This was not a natural mining fork, but a software bug.

**What happened:**
- Bitcoin v0.7 had a database limitation (BerkeleyDB)
- Bitcoin v0.8 removed the limitation, allowing larger blocks
- A miner running v0.8 created a block that v0.7 nodes rejected
- Network split: ~60% on v0.8 (new version), ~40% on v0.7 (old version)

**Resolution:**
- Bitcoin core developers made a controversial decision: **Revert to v0.7 chain**
- Large mining pools voluntarily downgraded to v0.8
- The v0.8 chain (longer!) was abandoned
- **Why?** v0.7 had more nodes, broader support, and reverting caused less disruption

**Consequences:**
- Several transactions were reversed
- A few double-spends occurred (~$10,000 in losses)
- Proved that social consensus can override technical "longest chain" rule in emergencies

**This was a wake-up call:** Bitcoin governance is not purely algorithmic—human coordination matters in crises.

---

## Rewriting History: The 51% Attack

Now let's tackle the big question: Can someone rewrite Bitcoin's history?

**Short answer:** Yes, but it's absurdly expensive and likely to fail.

### The Attack Scenario

**Alice wants to double-spend:**

1. Alice has 10 BTC
2. She buys a car from Bob, sends 10 BTC (gets confirmed in Block 850,000)
3. Bob gives Alice the car (she drives away)
4. Alice secretly starts mining an alternate chain where her 10 BTC transaction never happened
5. If Alice can create a longer chain than the honest network, she can broadcast it and rewrite history
6. Her transaction to Bob disappears, Bob loses the car AND the Bitcoin

### Why You Need >50% Hash Power

Let's say Alice controls 40% of the network's hash power (honest network has 60%).

**The race:**
```
Honest chain (60% hash power):
Block 850,000 → 850,001 → 850,002 → 850,003 → ...
(Alice's payment to Bob confirmed here)

Alice's secret chain (40% hash power):
Block 849,999 → 850,000* → 850,001* → 850,002* → ...
(* = different blocks, no payment to Bob)
```

**Alice mines slower than the honest network:**
- Honest network: Mines 1 block every 10 minutes (on average)
- Alice: Mines 1 block every 15 minutes (on average, with 40% hash power)

**While Alice mines her secret chain:**
- She's 1 block behind initially (needs to catch up from 849,999)
- For every 2 blocks Alice mines, the honest network mines 3 blocks
- Alice falls **further behind**, not closer

**Probability Alice catches up from N blocks behind (with 40% hash power):**
```
N=1: ~13% chance
N=2: ~2% chance
N=6: ~0.0003% chance (essentially impossible)
```

**With 51% hash power:**
- Alice mines faster than the honest network
- For every 51 blocks Alice mines, honest network mines 49 blocks
- Alice slowly but surely catches up
- Given enough time, she **will** overtake (100% probability eventually)

**This is why 51% is the magic number.**

### The Real-World Cost (2024 Numbers)

Let's calculate what a 51% attack would actually cost.

**Step 1: Acquire the hash power**

```
Bitcoin network hash rate: ~700 EH/s
Need for 51% attack: ~357 EH/s

Modern ASIC (Antminer S21):
- Hash rate: 335 TH/s
- Cost: $5,000

Number of miners needed:
357 EH/s ÷ 335 TH/s = 1,065,000 miners

Hardware cost:
1,065,000 × $5,000 = $5.3 billion
```

**Step 2: Operational costs**

```
Power consumption per miner: 3,500 watts
Total power: 1,065,000 × 3,500W = 3.7 GW (gigawatts)

Electricity cost (industrial rate $0.05/kWh):
3.7 GW × 24 hours = 88.8 GWh per day
88.8 GWh × $0.05 = $4.44 million per day
= $185,000 per hour
```

**Step 3: Additional costs and barriers**

- **Supply constraint:** 1 million+ ASICs don't exist. Total global production is ~2-3 million units/year. It would take months to acquire this many.
- **Price impact:** Buying that many ASICs would drive prices up 2-5x.
- **Infrastructure:** Need industrial-scale power (3.7 GW is a nuclear power plant's output), cooling, facilities.
- **Detection:** Such a large ramp-up would be noticed. Hash rate would spike, community would be alerted.

**Step 4: The paradox**

Even if you successfully attacked:
- Bitcoin's price would crash (you just proved it's insecure)
- Your $5.3B in ASICs become worthless (only useful for mining Bitcoin)
- Any Bitcoin you "stole" is now worth far less
- You might be legally prosecuted
- The community could hard-fork to a new PoW algorithm, bricking your ASICs

**It's economically irrational.** You'd spend $5B+ to maybe steal a few million dollars worth of Bitcoin, while destroying the value of your own hardware and the stolen coins.

### Has a 51% Attack Ever Happened?

**On Bitcoin: No.** Bitcoin has never been successfully 51% attacked.

**On smaller cryptocurrencies: Yes.**

Examples:
- **Bitcoin Gold (2018):** Attacker rented hash power, stole ~$18M from exchanges
- **Ethereum Classic (2019, 2020):** Multiple attacks, millions stolen
- **Vertcoin (2018):** $100k stolen

**Why these but not Bitcoin?**
- Smaller hash rate = cheaper to attack
- Bitcoin Gold: ~100 TH/s network (tiny compared to Bitcoin's 700 EH/s)
- Cost to attack BTG: ~$5,000-10,000 (rental from NiceHash)
- Cost to attack Bitcoin: ~$5 billion (can't rent, must own hardware)

**The lesson:** Security scales with network size. Bitcoin's massive hash rate is its shield.

---

## Network Architecture

Bitcoin is a peer-to-peer network with no central servers. Let's break down how it actually works.

### Node Types

**1. Full Nodes**
- **What they do:** Store the entire blockchain, validate every transaction and block
- **Storage:** ~600 GB (as of 2024, grows ~50-100 GB/year)
- **Validation:** Check every rule, reject invalid data
- **Count:** ~15,000-50,000 globally (estimates vary)
- **Who runs them:** Enthusiasts, businesses, exchanges, miners

**Example:** Bitcoin Core (the reference implementation)

**2. Pruned Nodes**
- **What they do:** Validate everything but only keep recent blocks
- **Storage:** ~10 GB (configurable)
- **How:** After validating old blocks, delete them (keep only headers)
- **Trade-off:** Can't serve full blockchain to other nodes, but still fully validate

**Good for:** Running a full node on limited storage (e.g., Raspberry Pi)

**3. Mining Nodes**
- **What they do:** Full node + mining hardware (ASICs)
- **Role:** Validate transactions, assemble blocks, solve PoW puzzle
- **Count:** ~10-20 major mining pools (representing millions of individual ASICs)
- **Who:** Professional mining operations

**Note:** Most individual miners join pools—the pool runs the full node, individual miners just contribute hash power.

**4. SPV Clients (Simplified Payment Verification)**
- **What they do:** Store only block headers (~100 MB), trust that transactions in headers are valid
- **Validation:** Don't verify all transactions, only check if their own transactions are in blocks
- **Trust model:** Trust that if most miners accepted a block, it's probably valid
- **Count:** Millions (most mobile wallets: Electrum, BlueWallet, etc.)

**Trade-off:** Lightweight but less secure—must trust other nodes aren't lying.

### Node Type Comparison

| Feature | Full Node | Pruned Node | Mining Node | SPV Client |
|---------|-----------|-------------|-------------|------------|
| **Storage** | ~600 GB | ~10 GB | ~600 GB | ~100 MB |
| **Validates all transactions** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| **Validates all blocks** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No (only headers) |
| **Can serve full blockchain** | ✅ Yes | ❌ No | ✅ Yes | ❌ No |
| **Mines blocks** | ❌ No | ❌ No | ✅ Yes | ❌ No |
| **Trust requirements** | None | None | None | Some (trust miners) |
| **Suitable for** | Businesses, enthusiasts | Home users | Professional miners | Mobile wallets |

### How Nodes Discover Peers

Bitcoin nodes connect to each other in a peer-to-peer mesh. But how do new nodes find peers?

**Bootstrapping process:**

1. **DNS Seeds:** Hard-coded DNS addresses that return IP addresses of active nodes
   ```
   seed.bitcoin.sipa.be
   dnsseed.bluematt.me
   seed.bitcoinstats.com
   ```
   Query: "Give me some Bitcoin node IPs" → Get list of ~100 IPs

2. **Connect to peers:** Node attempts connections to 8-10 peers from the list

3. **Exchange peer lists:** Once connected, nodes share their known peers (`addr` messages)

4. **Store for future:** Node saves working peer IPs to local database for next restart

**Maintaining connections:**
- Nodes ping peers regularly (keep-alive)
- If a peer stops responding, replace it
- Try to maintain ~8 outbound connections, accept ~117 inbound connections (default limits)

### Gossip Protocol: How Information Spreads

When a transaction or block is created, it propagates through the network via **gossip protocol**.

**Example: Alice broadcasts a transaction**

```
Step 1: Alice's wallet sends to 8 connected peers
Alice → [Node 1, Node 2, Node 3, ..., Node 8]

Step 2: Each peer validates and forwards to their peers
Node 1 → [Node A, Node B, Node C, ...]
Node 2 → [Node D, Node E, Node F, ...]
...

Step 3: Nodes keep track of what they've seen
If Node A receives the same transaction from Node 1 and Node 2:
  - Node A only forwards it once (avoids spam)
  - Uses transaction ID (hash) to identify duplicates

Step 4: Within 5-15 seconds, most nodes have seen it
```

**Propagation times (estimates):**
- 50% of nodes: 1-3 seconds
- 95% of nodes: 5-10 seconds
- 99%+ of nodes: 10-20 seconds

**Block propagation is similar but faster (more priority):**
- Blocks use **compact block relay** (optimization introduced in 2016)
- Instead of sending all transactions, send:
  - Block header
  - Short IDs of transactions (6 bytes instead of ~250 bytes avg)
  - Nodes reconstruct the block using their mempool
- Result: Block propagation went from ~10 seconds to ~1-2 seconds

**Why speed matters:**
- Slow block propagation → more orphaned blocks
- Miners with better connectivity have advantage (centralization pressure)
- Optimizations like compact block relay level the playing field

### Network Resilience

**How many nodes can fail before Bitcoin breaks?**

Short answer: **Bitcoin is extremely resilient.**

- As long as 1 full node exists, the entire blockchain can be recovered
- As long as 1 miner exists, new blocks can be produced
- Network is global: nodes in 100+ countries
- No single point of failure

**Attacks on the network:**

1. **Sybil attack:** Attacker runs many nodes to surround a victim
   - Mitigation: Nodes limit connections, diversify peer geography
   - Victim can connect to known good nodes manually

2. **Eclipse attack:** Isolate a node from the honest network
   - Mitigation: Similar to Sybil, plus nodes remember long-lived peers

3. **DDoS attacks:** Overwhelm nodes with traffic
   - Happens occasionally to specific nodes/pools
   - Network routes around them (other nodes still function)

**The network has survived:**
- China's mining ban (2021) - 50% of hash rate went offline overnight, network kept running
- Multiple nation-state threats
- Countless DDoS attacks
- No central servers to shut down, no company to target

---

## The Byzantine Generals Problem (Historical Context)

Bitcoin's consensus mechanism solves a famous problem in computer science: the **Byzantine Generals Problem**.

### The Classic Problem

Imagine several Byzantine army generals surrounding a city. They need to coordinate: attack at dawn, or retreat?

**Challenges:**
1. Generals can only communicate via messengers
2. Some generals might be traitors (Byzantine fault) who:
   - Send different messages to different generals
   - Lie about what others told them
3. Messengers might be intercepted or delayed

**Question:** How can loyal generals reach consensus (agree on a plan) when some might be traitors?

**This maps to distributed systems:**
- Generals = Nodes in a network
- Traitors = Malicious/faulty nodes
- Messengers = Network communication
- Goal = Reach consensus on a shared state (the ledger)

### Why This Was Considered Impossible (Before Bitcoin)

Before Bitcoin, consensus in a truly open, permissionless network was thought to be unsolvable:

**Traditional consensus (Paxos, Raft, PBFT):**
- Requires a **known set of participants**
- Assumes **<33% are malicious**
- Doesn't work if anyone can join (Sybil attack: attacker creates millions of fake identities)

**Bitcoin's insight:**
- Replace "1 node = 1 vote" with **"1 hash = 1 vote"**
- Can't fake hash power (requires real-world resources)
- Economic cost to attack (not just computational)
- Probabilistic finality (not instant, but gets more certain over time)

### How Bitcoin Solves It

**Proof-of-Work as voting mechanism:**
1. Nodes propose blocks (their version of history)
2. "Voting" is done by miners expending energy (hash power)
3. The chain with the most cumulative work wins
4. Attacking requires outspending the honest network

**Economic game theory:**
- Honest behavior is more profitable than attacking
- Attacking destroys the value of your investment
- Decentralized: no single point of failure

**Why this works:**
- **Permissionless:** Anyone can join (download software, start mining)
- **Trustless:** Don't need to know or trust other participants
- **Censorship-resistant:** No central authority to shut down
- **Globally verifiable:** Anyone can check the rules are followed

This was a genuine breakthrough in distributed systems. Before Bitcoin (2008), no one had solved Byzantine consensus in a permissionless setting.

---

## Summary: Consensus Without Central Authority

Let's tie it all together.

Bitcoin achieves consensus through a combination of:

### 1. Separation of Powers
- **Everyone validates** (checks rules) → no single party can change rules
- **Miners compete** to write blocks → decentralizes block production
- **Economic incentives** align with honest behavior → attacking is more expensive than cooperating

### 2. Proof-of-Work
- Makes block production expensive (real-world cost)
- Rate-limits block creation (~10 minutes)
- Randomly selects block producers (proportional to hash power)
- Creates an audit trail (chain of work)

### 3. Difficulty Adjustment
- Maintains predictable block times regardless of hash rate
- Enables fixed supply (21 million BTC)
- Self-regulates without central authority

### 4. Longest Chain Rule
- Simple, objective rule for resolving conflicts
- Forks naturally resolve (usually within 1-2 blocks)
- Deeper blocks become exponentially harder to reverse

### 5. Peer-to-Peer Network
- No central servers (censorship-resistant)
- Gossip protocol spreads information fast
- Redundancy (thousands of nodes)
- Global distribution

### The Result

A system where:
- ✅ Anyone can join or leave freely
- ✅ No trusted third parties required
- ✅ Rules are enforced by math and economics, not authority
- ✅ History becomes immutable over time
- ✅ Attacking is more expensive than the potential gain
- ✅ Network self-regulates through emergent behavior

**This is Bitcoin's core innovation:** not blockchain (that's just a data structure), not digital signatures (existed before), but **using economic incentives and Proof-of-Work to achieve consensus in an open, trustless, permissionless network**.

In the next file (`3-bitcoin-transactions.md`), we'll dive into the details of how transactions are constructed, broadcast, and confirmed—including the practical aspects of fees, double-spend attacks, and why merchants wait for multiple confirmations.

---

*Understanding consensus is understanding Bitcoin's security model. Everything else (wallets, addresses, transactions) is built on top of this foundation.*
