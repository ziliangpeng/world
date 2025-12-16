# Bitcoin Transactions: From Creation to Confirmation

## Introduction

In the previous files, we've covered Bitcoin's foundation:
- `1-bitcoin-basics.md`: The double-spend problem, blockchain, mining, and UTXO model
- `2-bitcoin-consensus.md`: How the network reaches agreement through Proof-of-Work

Now let's dive into the **lifecycle of a transaction**—from the moment Alice decides to send Bitcoin to Bob, through broadcasting, mempool, mining, and finally deep confirmation.

We'll also tackle practical questions:
- Why do some transactions confirm instantly while others sit for hours?
- What are transaction fees and why do they vary wildly?
- When is a transaction truly "final"?
- How do double-spend attacks work in practice?
- Why do exchanges make you wait for confirmations?

By the end, you'll understand not just how transactions work theoretically, but how they behave in the real world.

---

## Creating a Transaction

Let's start at the beginning: Alice wants to send 0.5 BTC to Bob.

### Step 1: Coin Selection

Alice's wallet needs to select which UTXOs (coins) to spend. Remember from file 1: Bitcoin doesn't have account balances, only discrete UTXOs.

**Alice's wallet scans the blockchain and finds her UTXOs:**
```
UTXO₁: 0.3 BTC (from transaction abc123...)
UTXO₂: 0.8 BTC (from transaction def456...)
UTXO₃: 0.1 BTC (from transaction ghi789...)
UTXO₄: 1.2 BTC (from transaction jkl012...)

Total available: 2.4 BTC
```

**To send 0.5 BTC, Alice's wallet might choose:**

**Option A (minimize inputs, minimize fee):**
- Use UTXO₂ (0.8 BTC)
- Send 0.5 BTC to Bob
- Send 0.2999 BTC back to Alice (change)
- Fee: 0.0001 BTC

**Option B (use smallest UTXOs, consolidate):**
- Use UTXO₁ (0.3 BTC) + UTXO₂ (0.8 BTC)
- Send 0.5 BTC to Bob
- Send 0.5999 BTC back to Alice (change)
- Fee: 0.0001 BTC

Different wallets use different **coin selection algorithms**:
- **Minimize fee:** Use fewest inputs (each input adds to transaction size)
- **Privacy:** Avoid linking UTXOs from different sources
- **Consolidation:** Use many small UTXOs when fees are low (clean up wallet)

Most wallets default to minimizing fees (Option A).

### Step 2: Constructing the Transaction

The wallet creates a transaction data structure:

```
Transaction Structure:
├── Version: 2 (current Bitcoin protocol version)
│
├── Input Count: 1
├── Inputs:
│   └── Input 1:
│       ├── Previous TX Hash: def456... (points to UTXO₂)
│       ├── Output Index: 0 (which output of that TX)
│       ├── ScriptSig: (signature + public key - filled in step 3)
│       └── Sequence: 0xFFFFFFFF (enables RBF if lower)
│
├── Output Count: 2
├── Outputs:
│   ├── Output 1:
│   │   ├── Value: 50,000,000 satoshis (0.5 BTC)
│   │   └── ScriptPubKey: OP_DUP OP_HASH160 <Bob's address hash> OP_EQUALVERIFY OP_CHECKSIG
│   │       (spending condition: "must provide signature matching Bob's address")
│   └── Output 2 (change):
│       ├── Value: 29,990,000 satoshis (0.2999 BTC)
│       └── ScriptPubKey: OP_DUP OP_HASH160 <Alice's new address hash> OP_EQUALVERIFY OP_CHECKSIG
│
└── Locktime: 0 (transaction valid immediately)
```

**Key points:**

**Amounts in satoshis:**
Bitcoin uses the smallest unit, satoshis (1 BTC = 100,000,000 satoshis), for precision.

**ScriptPubKey (spending conditions):**
This is Bitcoin Script that defines who can spend this output. For a standard payment to an address, it requires a valid signature from that address's private key.

**Change address:**
Alice generates a new address for her change (0.2999 BTC). This is good for privacy—observers can't easily tell which output is the payment and which is change.

**Transaction fee:**
```
Fee = Total Inputs - Total Outputs
Fee = 0.8 BTC - (0.5 BTC + 0.2999 BTC)
Fee = 0.0001 BTC = 10,000 satoshis
```

The fee goes to the miner who includes this transaction in a block.

### Step 3: Signing the Transaction

Alice must prove she owns UTXO₂. She does this with a digital signature.

**Process:**
1. Wallet creates a hash of the transaction data (excluding signatures)
2. Wallet signs this hash using Alice's private key (for the address that owns UTXO₂)
3. Signature and public key are inserted into ScriptSig field

**Result:**
```
ScriptSig:
<signature> <public key>

Example (hex-encoded):
304402201a3b5c... (72-byte signature)
02c3a7e9f1d5b3... (33-byte compressed public key)
```

**Why this works:**
- Public key can verify the signature (cryptography)
- Public key hash matches the address that owns UTXO₂ (proves ownership)
- Signature is specific to this transaction (can't be reused for other transactions)

**Transaction is now complete and ready to broadcast.**

**Transaction size:**
```
Version: 4 bytes
Input count: 1 byte
Input data: ~180 bytes (hash + index + signature + pubkey + sequence)
Output count: 1 byte
Output data: ~68 bytes (value + script for 2 outputs)
Locktime: 4 bytes

Total: ~258 bytes
```

Transaction size matters because fees are calculated per byte.

---

## Broadcasting: From Wallet to Network

Alice's wallet now broadcasts the transaction to the Bitcoin network.

### The Propagation Process

```
Step 1: Alice's wallet connects to peers
┌─────────────┐
│ Alice Wallet│
└──────┬──────┘
       │ Connected to 8 peers
       ↓
┌──────────────────────────────────────────┐
│ Node₁  Node₂  Node₃  Node₄  Node₅ ...   │
└──────────────────────────────────────────┘

Step 2: Wallet sends transaction to all connected peers
Alice → Node₁, Node₂, Node₃, ..., Node₈

Step 3: Each node validates the transaction
Node₁ checks:
- ✓ Signature valid?
- ✓ UTXO₂ exists and unspent?
- ✓ Inputs ≥ Outputs?
- ✓ Scripts satisfy spending conditions?

If valid, node:
1. Adds transaction to its mempool
2. Forwards to its peers (except the one it received from)

Step 4: Gossip protocol spreads transaction
Node₁ → Node_A, Node_B, Node_C, ...
Node₂ → Node_D, Node_E, Node_F, ...
Node₃ → Node_G, Node_H, Node_I, ...
...

Step 5: Duplicate detection
If Node_A receives the same transaction from Node₁ and Node₂:
- Recognizes duplicate (by transaction ID/hash)
- Only forwards once
- Prevents network spam

Step 6: Network saturation
Within 5-15 seconds, 90%+ of nodes have seen the transaction
```

**Propagation times (real-world estimates):**
- 50% of nodes: 1-3 seconds
- 95% of nodes: 5-10 seconds
- 99%+ of nodes: 10-20 seconds

**What can go wrong:**
- **Invalid transaction:** Nodes reject and don't forward (stops propagation)
- **Network partition:** Some nodes temporarily isolated (rare, usually resolves quickly)
- **Low fee:** Transaction is valid but miners may ignore it (sits in mempool)

---

## The Mempool: Waiting Room for Transactions

Once nodes validate Alice's transaction, they store it in their **mempool** (memory pool).

### What is the Mempool?

The mempool is:
- **Each node's local waiting area** for unconfirmed transactions
- **Not global:** Each node maintains its own mempool
- **Temporary:** Transactions leave mempool when included in a block (or dropped)
- **Prioritized:** Transactions ordered by fee rate for miners

**Mempool is NOT consensus:**
- Nodes may have slightly different mempools (received transactions in different order)
- That's okay—only blocks are consensus (part of the blockchain)

### Mempool Ordering: Fee Priority

Miners want to maximize profit, so they prioritize high-fee transactions.

**Fee rate calculation:**
```
Fee rate = Transaction fee / Transaction size (in bytes)

Alice's transaction:
Fee: 10,000 satoshis
Size: 258 bytes
Fee rate: 10,000 / 258 ≈ 38.76 sat/byte
```

**Mempool visualization (simplified):**
```
┌─────────────────────────────────────────┐
│  HIGH FEE ZONE (>100 sat/byte)         │
│  Tx₁: 150 sat/byte                     │
│  Tx₂: 120 sat/byte                     │
│  Tx₃: 105 sat/byte                     │
├─────────────────────────────────────────┤
│  MEDIUM FEE ZONE (20-100 sat/byte)     │
│  Tx₄: 85 sat/byte                      │
│  Alice's tx: 38 sat/byte ← HERE        │
│  Tx₅: 25 sat/byte                      │
├─────────────────────────────────────────┤
│  LOW FEE ZONE (<20 sat/byte)           │
│  Tx₆: 10 sat/byte                      │
│  Tx₇: 5 sat/byte                       │
│  Tx₈: 1 sat/byte (may never confirm)   │
└─────────────────────────────────────────┘
```

Miners fill blocks starting from the top (highest fee rate) and work down until the block is full.

**Block space limits:**
- Legacy block: ~1 MB (~2,000-3,000 transactions)
- SegWit block: ~4 MB weight units (~4,000-7,000 transactions)

**When mempool is full:**
If more than ~7,000 transactions are waiting and blocks are full, nodes may:
- Drop the lowest-fee transactions (typically <1 sat/byte)
- Set a minimum relay fee (won't accept transactions below threshold)

### Replace-By-Fee (RBF)

What if Alice set the fee too low and her transaction is stuck?

**RBF allows replacing an unconfirmed transaction with a higher-fee version:**

**Original transaction:**
```
Transaction A:
- Alice → Bob: 0.5 BTC
- Fee: 10,000 sats (38 sat/byte)
- Sequence: 0xFFFFFFFD (signals RBF-enabled)
```

**If stuck, Alice creates Transaction B:**
```
Transaction B:
- Alice → Bob: 0.5 BTC
- Fee: 50,000 sats (194 sat/byte)
- Same inputs, outputs adjusted for higher fee
- MUST increase fee by enough to incentivize replacement
```

**Nodes receive Transaction B:**
- Recognize it conflicts with Transaction A (same input)
- Check if fee is higher
- If yes, replace Transaction A with B in mempool
- Transaction B now has priority

**Limitations:**
- Only works if original transaction signaled RBF (sequence < 0xFFFFFFFE)
- Original transaction must still be unconfirmed
- Replacement must pay higher absolute fee AND higher fee rate

**Historical note:** RBF was controversial (enables easier double-spend attempts) but is now widely accepted as a useful tool.

### Child-Pays-For-Parent (CPFP)

Alternative to RBF: if you're the **recipient**, you can speed up an incoming transaction.

**Scenario:**
- Bob is waiting for Alice's low-fee transaction
- Alice can't/won't use RBF
- Bob creates a new transaction spending the unconfirmed UTXO from Alice

**CPFP transaction:**
```
Bob's transaction:
- Input: Alice's UTXO (0.5 BTC, currently unconfirmed)
- Output: Bob's new address
- Fee: 100,000 sats (very high)

Miner sees:
- Alice's tx: 10,000 sats fee
- Bob's tx: 100,000 sats fee
- BUT Bob's tx can only be mined if Alice's tx is mined first

Combined fee rate: (10,000 + 100,000) / (Alice's size + Bob's size)
= 110,000 / 516 bytes ≈ 213 sat/byte (very high priority!)

Miner includes BOTH transactions to collect total fee.
```

**Why it works:** Miners can see transaction dependencies and consider packages (parent + child) as a unit.

---

## Transaction Lifecycle: Confirmations

Let's follow Alice's transaction from unconfirmed to deeply buried.

### Stage 1: 0 Confirmations (Unconfirmed)

**Status:**
- Transaction is in mempools
- Visible on block explorers as "unconfirmed"
- Bob's wallet shows "pending" incoming payment

**Security level: LOW**
- Transaction could be replaced (RBF)
- If a block is found without this transaction, it stays in mempool
- Vulnerable to double-spend attacks (covered later)

**Use cases:**
- Low-value transactions ($5-20)
- Trusted parties
- Situations where reversal risk is acceptable

**Typical duration:**
- High fee: Next block (~10 minutes)
- Medium fee: 1-3 blocks (~10-30 minutes)
- Low fee: Hours to days (or never)

### Stage 2: 1 Confirmation

**Event:** A miner includes Alice's transaction in Block 850,000.

```
Block 850,000 (just mined):
- Contains ~3,500 transactions
- Including: Alice → Bob (0.5 BTC)
- Block hash: 00000000000000000001a3f5b2c8...
```

**Status:**
- Transaction is now "on-chain"
- Embedded in the blockchain
- Bob's wallet shows "1 confirmation"

**Security level: MEDIUM-LOW**
- Much safer than 0-conf
- But block could be orphaned (see file 2 on forks)
- Small chance (~1-5%) of reversal due to natural fork

**Use cases:**
- Medium-value transactions ($100-500)
- Digital goods
- Fast-food restaurants

### Stage 3: 2-5 Confirmations

**Event:** More blocks are mined on top.

```
Block 850,000 (Alice's transaction)
     ↓
Block 850,001 (2 confirmations)
     ↓
Block 850,002 (3 confirmations)
     ↓
Block 850,003 (4 confirmations)
     ↓
Block 850,004 (5 confirmations)
```

**Security level: MEDIUM-HIGH**
- Each additional block makes reversal exponentially harder
- 3 confirmations: ~0.1% natural fork chance
- 5 confirmations: ~0.001% chance

**Use cases:**
- Higher-value transactions ($1,000-10,000)
- Some exchanges (minimum deposit requirement)

### Stage 4: 6+ Confirmations (Industry Standard)

**Event:** Alice's transaction is now buried under 6 blocks.

```
Block 850,000 (Alice's transaction here)
     ↓
Block 850,001
     ↓
Block 850,002
     ↓
Block 850,003
     ↓
Block 850,004
     ↓
Block 850,005
     ↓
Block 850,006 (6 confirmations) ← Standard threshold
```

**Security level: VERY HIGH**
- Reversing requires rewriting 6 blocks PLUS outpacing the network
- Only possible with 51% attack (see file 2)
- Natural fork lasting 6 blocks: ~0.0001% probability

**Use cases:**
- Large transactions (>$10,000)
- Exchange deposits (most require 3-6 confirmations)
- High-security applications

**Why 6 is the standard:**
It's mentioned in the original Bitcoin whitepaper as a reasonable threshold. We'll see the math behind this in the next section.

**Time to 6 confirmations:**
- Expected: ~60 minutes (6 × 10 minutes)
- Reality: 30-90 minutes (variance in block times)

---

## The Math: Reversal Probability

Why is 6 confirmations considered secure? Let's look at the probability of an attacker successfully reversing a transaction.

### The Reversal Attack Model

**Assumptions:**
- Attacker has q% of network hash power
- Honest network has p% (where p + q = 1)
- Transaction has z confirmations
- Attacker is mining a secret chain to double-spend

**Formula (derived in Bitcoin whitepaper):**
```
P(attacker succeeds) = 1 - Σ (from k=0 to z) [ λ^k * e^(-λ) / k! * (1 - (q/p)^(z-k)) ]

Where λ = z * (q/p)

Simplified approximation for q < p:
P ≈ (q/p)^z
```

**For most practical purposes, the simplified formula is sufficient.**

### Probability Table

| Confirmations | 10% Attacker | 20% Attacker | 30% Attacker | 40% Attacker | 49% Attacker |
|---------------|--------------|--------------|--------------|--------------|--------------|
| **1** | 11.1% | 25.0% | 42.9% | 66.7% | 96.1% |
| **2** | 1.2% | 6.3% | 18.4% | 44.4% | 92.3% |
| **3** | 0.14% | 1.6% | 7.9% | 29.6% | 88.6% |
| **4** | 0.015% | 0.39% | 3.4% | 19.8% | 85.1% |
| **5** | 0.0017% | 0.098% | 1.5% | 13.2% | 81.7% |
| **6** | 0.00019% | 0.024% | 0.64% | 8.8% | 78.5% |

**Key insights:**

**With 10% hash power:**
- 1 confirmation: 11% success rate
- 6 confirmations: 0.0002% success rate (essentially impossible)

**With 30% hash power:**
- 1 confirmation: 43% success rate (quite likely!)
- 6 confirmations: 0.64% success rate (still very low)

**With 49% hash power:**
- Even at 6 confirmations: 78.5% success rate (nearly guaranteed to succeed eventually)
- **This is why 51% is the critical threshold** (see file 2)

### Why 6 Confirmations?

From the table:
- Even a powerful attacker (30% hash power) has <1% success rate
- A realistic attacker (10-20%) has essentially zero chance
- Provides security margin against unknown attackers

**Trade-offs:**
- **Too few confirmations:** Security risk
- **Too many confirmations:** Poor user experience (long wait times)
- **6 confirmations:** Reasonable balance (~1 hour wait, very high security)

**When to require MORE:**
- Very large transactions (>$100,000): 12-20 confirmations
- Paranoid security: 100+ confirmations
- Smart contracts: Often coded to require 20+ confirmations

**When to accept LESS:**
- Small amounts + trusted context: 1-3 confirmations
- Digital goods (low reversal cost): 1 confirmation
- Physical goods (already shipped): 3 confirmations

---

## Double-Spend Attacks

Now let's look at how double-spend attacks actually work in practice.

### Attack Type 1: Race Attack (0-Confirmation)

The simplest and most common attack targets zero-confirmation transactions.

**Scenario: Alice wants to defraud a merchant**

**Step 1:** Alice creates two conflicting transactions:
```
Transaction A:
- Input: Alice's 1 BTC UTXO
- Output: 1 BTC to Merchant

Transaction B:
- Input: Alice's 1 BTC UTXO (same UTXO!)
- Output: 1 BTC to Alice's other address
```

**Step 2:** Alice broadcasts them strategically:
- Transaction A → Sent directly to Merchant's node
- Transaction B → Sent to well-connected mining pools

**Step 3:** Race begins:
```
Merchant's node sees Transaction A first:
- Validates it ✓
- Shows "payment received" to merchant
- Merchant gives Alice the goods

Mining pools see Transaction B first:
- Validates it ✓
- Adds to mempool with high priority (Alice paid higher fee)
```

**Step 4:** Miner includes Transaction B in a block:
```
Block 850,000 contains Transaction B (Alice → Alice)
Transaction A is now invalid (double-spend attempt)
Merchant's node rejects Transaction A when it sees the block
```

**Result:**
- Alice got the goods
- Merchant got nothing
- Alice still has her 1 BTC

**Success rate:**
Depends on:
- Network topology (how well-connected is merchant's node?)
- Fee differential (Transaction B had higher fee)
- Timing (how long did merchant wait?)

**Estimated success:** 10-50% if executed well against 0-conf.

**Merchant protection:**
- **Don't accept 0-conf for valuable items**
- Use well-connected nodes (harder to isolate)
- Wait 1+ confirmations
- For small amounts (<$20), accept the risk

### Attack Type 2: Finney Attack (0-Confirmation)

Named after Hal Finney, this is a pre-mining attack.

**Requirements:**
- Attacker must be a miner (or rent hash power)
- Requires timing and luck

**Scenario:**

**Step 1:** Alice (a miner) pre-mines a block **without broadcasting it**
```
Block 850,000 (Alice's secret block):
- Contains Transaction B: Alice → Alice (1 BTC)
- Valid nonce found
- NOT broadcast to network yet
```

**Step 2:** Alice makes a purchase from a merchant
- Merchant accepts 0-conf transactions
- Alice broadcasts Transaction A: Alice → Merchant (1 BTC)
- Uses the same UTXO as in her secret Transaction B

**Step 3:** Merchant sees Transaction A
- Validates ✓
- Shows "payment received"
- Gives Alice the goods

**Step 4:** Alice immediately broadcasts her pre-mined block
```
Network receives Block 850,000:
- Contains Transaction B (conflicts with Transaction A)
- Is a valid block (proper PoW)
- Miners build on Alice's block
- Transaction A becomes invalid
```

**Result:**
- Alice got the goods
- Alice keeps her 1 BTC
- Merchant gets nothing

**Success rate:**
Depends on:
- Alice's hash power (affects how often she can pre-mine blocks)
- Timing (must make purchase quickly after finding block)
- Network propagation (Alice's block must reach most nodes before Transaction A gets mined)

**Estimated success:** 1-5% per attempt (requires finding a block, then executing attack before someone else finds a block).

**Why it's hard:**
- Finding a block takes ~10 minutes on average for the ENTIRE network
- For a miner with 1% hash power, finding a block takes ~1,000 minutes (16 hours)
- Must find merchant, make purchase, and broadcast—all before next block
- Low probability, but higher than race attack if you CAN pre-mine

**Merchant protection:**
- Wait for 1 confirmation (eliminates this attack entirely)
- For 0-conf, only accept small amounts where attack cost > profit

### Attack Type 3: 51% Attack (Multi-Confirmation)

We covered this in detail in file 2, but let's see it from the transaction perspective.

**Scenario:** Alice buys a car for 10 BTC, waits for 6 confirmations, then attacks.

**Step 1:** Alice's transaction gets 6 confirmations
```
Block 850,000 (Alice → Dealer: 10 BTC)
     ↓
Block 850,001
     ↓
...
     ↓
Block 850,006 (6 confirmations)

Dealer considers transaction final, gives Alice the car.
```

**Step 2:** Alice secretly mines alternate chain
Alice controls 51% hash power and mines a secret chain starting from Block 849,999:
```
Public chain:
849,999 → 850,000 → 850,001 → ... → 850,006 (6 blocks)

Alice's secret chain:
849,999 → 850,000* → 850,001* → ... → 850,006* → 850,007* (7 blocks)
         (no Alice→Dealer transaction)
```

**Step 3:** Alice broadcasts her chain
Once Alice's chain is longer, she broadcasts it:
```
Network receives Alice's chain (7 blocks)
Compares to public chain (6 blocks)
Longest chain rule: Alice's chain wins
Network reorganizes to Alice's chain
```

**Result:**
- Alice's payment transaction disappears (never existed in Alice's chain)
- Dealer loses car and Bitcoin
- Alice has car AND 10 BTC

**Cost:** As calculated in file 2: $5.3 billion in hardware + $185k/hour in electricity.

**Why it's impractical:**
- Absurdly expensive
- Destroys Bitcoin's value (making your stolen Bitcoin worthless)
- Would be detected (massive hash rate increase)
- Community could hard-fork, bricking your hardware

**Merchant protection:**
- 6+ confirmations makes this the ONLY possible attack
- And even then, it requires nation-state level resources

---

## Fee Markets: Why Transactions Cost Money

Transaction fees are one of Bitcoin's most misunderstood aspects. Let's break down how they work.

### Why Fees Exist

**Block space is limited:**
```
Pre-SegWit (before 2017):
- Block size: 1 MB
- Typical transaction: ~250 bytes
- Max transactions per block: ~4,000

Post-SegWit (after 2017):
- Block weight: 4 MB (measured differently)
- Typical transaction: ~140 weight units (SegWit discount)
- Max transactions per block: ~7,000-10,000
```

**Blocks are mined every ~10 minutes:**
```
Maximum throughput:
7,000 transactions per block ÷ 10 minutes = 700 tx/min ≈ 11.7 tx/sec

Compare to:
- Visa: 24,000 tx/sec
- Ethereum: ~15 tx/sec
- Solana: 3,000+ tx/sec
```

**When demand > supply, prices rise (fees increase).**

### Fee Calculation

**Fees are calculated per byte (or weight unit for SegWit):**
```
Transaction fee = Size (bytes) × Fee rate (satoshis per byte)

Example:
- Transaction size: 250 bytes
- Fee rate: 50 sat/byte
- Total fee: 250 × 50 = 12,500 satoshis = 0.000125 BTC

At $95,000/BTC:
12,500 sats × ($95,000 / 100,000,000 sats) = $11.88
```

**Factors affecting transaction size:**
- Number of inputs (each input adds ~180 bytes for legacy, ~68 bytes for SegWit)
- Number of outputs (each output adds ~34 bytes)
- Type of address (SegWit is smaller than legacy)

**SegWit discount:**
SegWit (Segregated Witness) moves signature data to a separate "witness" field that's counted at a 75% discount.

```
Legacy transaction: 250 bytes = 250 weight units
SegWit transaction: 150 base bytes + 100 witness bytes = 175 weight units
                    (150 × 4) + (100 × 1) = 700 weight units
                    → Effective size: 175 bytes

Result: SegWit transactions are ~30-40% cheaper
```

### Historical Fee Data

Fees vary wildly based on network demand:

```
Year/Event           Avg Fee      Peak Fee    Context
2015-2016           $0.05        $0.50       Low adoption
Dec 2017            $25-55       $55         ICO mania, no SegWit adoption
2018-2019           $0.50-2      $5          Crypto winter
May 2021            $10-20       $62         Bull run peak
2022-2023           $1-3         $8          Bear market
Q1 2024             $3-8         $30         ETF launch, BRC-20 tokens
Q4 2024             $2-5         $15         Normal activity

(Fees in USD at time of transaction)
```

**What causes fee spikes:**
1. **Bull markets:** More speculation = more transactions
2. **New use cases:** Ordinals/BRC-20 tokens (2023-2024) clogged network
3. **Exchange movements:** Large exchanges moving funds (batch deposits/withdrawals)
4. **Panic/FOMO:** Everyone rushing to buy/sell at once

### Fee Estimation Strategies

How do wallets decide what fee to set?

**Method 1: Historical analysis**
- Look at recent blocks
- Calculate: What fee rate got confirmed in X blocks?
- Recommend based on user urgency:
  - Low priority: Confirmed within 6 blocks (1 hour) → 10 sat/byte
  - Medium: Within 3 blocks (30 min) → 25 sat/byte
  - High: Next block (10 min) → 50 sat/byte

**Method 2: Mempool analysis**
- Examine current mempool
- Estimate: How much fee needed to be in top 4,000 transactions?
- More accurate but requires full node

**Services that provide fee estimates:**
- mempool.space (shows real-time mempool and recommendations)
- Bitcoin Core's `estimatesmartfee` RPC
- Block explorers (blockchain.com, blockchair.com)

**User options:**
Most wallets offer:
- **Slow (cheap):** 6+ blocks, ~5-10 sat/byte
- **Medium:** 3-6 blocks, ~15-30 sat/byte
- **Fast (expensive):** 1-2 blocks, ~50-100 sat/byte
- **Custom:** Set your own fee

**Overpaying:**
Setting fee too high (500 sat/byte when 50 would work) wastes money—no speed benefit once you're in the top tier.

**Underpaying:**
Setting fee too low (1 sat/byte when mempool minimum is 10) means transaction may never confirm—sits in mempool forever or gets dropped.

---

## Transaction Malleability and SegWit

A brief but important topic: transaction malleability.

### What is Transaction Malleability?

**The problem (pre-SegWit):**
Transaction IDs (hashes) could be modified WITHOUT changing transaction validity.

**Example:**
```
Original transaction:
- ID: abc123...
- Alice → Bob: 1 BTC
- Signature: <sig_data>

Modified transaction:
- ID: def456... (DIFFERENT!)
- Alice → Bob: 1 BTC (same)
- Signature: <modified_sig_data> (still valid, but slightly different encoding)
```

The signature data could be altered (e.g., adding extra leading zeros in encoding) without invalidating it. This changed the transaction ID.

**Why it was a problem:**
- Can't reliably track transactions by ID before confirmation
- Breaks chains of unconfirmed transactions
- **Blocked Lightning Network development** (Lightning requires chaining unconfirmed transactions)

### How SegWit Fixed It

**Segregated Witness (SegWit, activated August 2017):**
- Moved signature data to a separate "witness" field
- Transaction ID is calculated ONLY from non-witness data
- Signature can no longer affect transaction ID

```
Post-SegWit transaction structure:
┌─────────────────────────────┐
│ Transaction Data            │ ← Used for transaction ID
│ - Inputs (without sigs)     │
│ - Outputs                   │
└─────────────────────────────┘
┌─────────────────────────────┐
│ Witness Data (segregated)   │ ← NOT used for transaction ID
│ - Signatures                │
│ - Public keys               │
└─────────────────────────────┘
```

**Benefits:**
1. **Fixed malleability:** Transaction ID can't be changed
2. **Enabled Lightning Network:** Unconfirmed transaction chains now safe
3. **Increased capacity:** Witness data gets 75% discount, effective block size increased
4. **Enabled future upgrades:** Taproot (2021) built on SegWit foundation

**Adoption:**
- August 2017: 10% of transactions
- 2024: ~70-80% of transactions use SegWit

Users benefit from lower fees, so SegWit adoption has grown organically.

---

## Real-World Usage Patterns

Let's step back and look at how Bitcoin is actually used in 2024-2025.

### The Reality: Most "Transactions" Are Off-Chain

**On-chain Bitcoin transactions per day (2024):**
- ~300,000-500,000 transactions/day
- ~10-15 transactions per second (on average)

**But "Bitcoin activity" is much larger:**

**Exchange internal transfers:**
- When you send Bitcoin from one Coinbase user to another: **No on-chain transaction**
- Coinbase just updates its internal database
- Millions of such "transactions" per day

**Lightning Network:**
- Thousands of off-chain payments
- Only 2 on-chain transactions per channel (open + close)

**Liquid Network (sidechain):**
- Institutional transfers
- Rare on-chain Bitcoin transactions

**Estimate:**
- **On-chain:** 500k tx/day
- **Off-chain (exchanges):** 5-10M tx/day
- **Lightning:** ~100k payments/day
- **Total "Bitcoin transactions":** 5-10M/day

**Most Bitcoin "movement" never touches the blockchain.**

### What On-Chain Transactions Actually Are

**Analysis of on-chain transactions (typical breakdown):**

```
Transaction Type              Percentage
─────────────────────────────────────────
Exchange deposits/withdrawals    ~40%
Large transfers (>$10k)          ~25%
Consolidations (batching UTXOs)  ~15%
Smart contract interactions       ~10%
(Ordinals, BRC-20)
Regular payments                  ~10%
```

**Key insight:** Bitcoin on-chain is increasingly a **settlement layer** for large/final transactions, not a payment network for daily purchases.

### Average Transaction Value

```
Year    Avg Transaction Value
2015    $500
2017    $3,000 (peak bull run)
2019    $2,000
2021    $15,000 (institutions)
2024    $45,000 (large settlements)
```

**Bitcoin transactions are getting LARGER and FEWER.**

This aligns with the "digital gold" narrative—Bitcoin is for storing/transferring value, not buying coffee.

### Confirmation Time Recommendations

**General guidelines:**

| Value | Use Case | Confirmations | Wait Time |
|-------|----------|---------------|-----------|
| <$20 | Coffee, small purchases | 0 (accept risk) | 0 min |
| $20-$100 | Fast food, groceries | 0-1 | 0-10 min |
| $100-$1,000 | Electronics, clothing | 1-3 | 10-30 min |
| $1,000-$10,000 | Laptop, used car | 3-6 | 30-60 min |
| $10,000-$100,000 | New car, down payment | 6-12 | 1-2 hours |
| $100,000+ | Real estate, large assets | 12-20 | 2-3 hours |
| Paranoid security | Any amount | 100+ | ~16 hours |

**Exchange deposits (industry standards):**
- Binance: 1 confirmation
- Coinbase: 3 confirmations
- Kraken: 6 confirmations
- Gemini: 6 confirmations

Exchanges vary based on risk tolerance and insurance policies.

---

## Summary: The Transaction Journey

Let's recap the full lifecycle of a Bitcoin transaction:

### Stage 1: Creation
- Wallet selects UTXOs (coin selection)
- Constructs transaction (inputs, outputs, fee)
- Signs with private key
- Total time: <1 second

### Stage 2: Broadcasting
- Sent to 8-10 peers
- Gossip protocol spreads across network
- Reaches 90%+ of nodes
- Total time: 5-15 seconds

### Stage 3: Mempool
- Validated by nodes
- Sits in mempool, prioritized by fee
- Miners select for next block
- Total time: Seconds to hours (depends on fee)

### Stage 4: Confirmation
- Included in a block (1 confirmation)
- Additional blocks mined on top (2, 3, 4, 5, 6+)
- Security increases exponentially with depth
- Total time: 10-60+ minutes (for 6 confirmations)

### Stage 5: Deep Burial
- Transaction becomes part of permanent history
- Reversal requires overwhelming resources
- Effectively irreversible
- Total time: 1+ hours

**Key Takeaways:**

1. **Fees matter:** Higher fee = faster confirmation
2. **Confirmations matter:** More confirmations = higher security
3. **0-conf is risky:** Only for small amounts or trusted parties
4. **6 confirmations is standard:** Industry best practice for security
5. **Transaction finality is probabilistic:** Not instant, but becomes increasingly certain over time

### What's Next?

In `4-bitcoin-limitations.md`, we'll explore why Bitcoin can't scale to be a global payment network, the inherent trade-offs in blockchain design (the trilemma), and why Bitcoin evolved from "peer-to-peer electronic cash" into "digital gold."

We'll also examine:
- Why 7 transactions per second isn't enough
- Why simply "making blocks bigger" doesn't work
- What happens when blocks are full
- Why Bitcoin Lightning Network was necessary (but has its own limitations)

---

*Transactions are Bitcoin's lifeblood—understanding their mechanics, costs, and security model is essential for using Bitcoin safely and effectively.*
