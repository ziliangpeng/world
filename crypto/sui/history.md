# SUI: Major History and Key Events

## Overview

Sui is a Layer 1 blockchain founded by Mysten Labs in 2021 and launched in May 2023. Built by former Meta (Facebook) engineers who worked on the failed Diem (Libra) project, Sui represents the rebirth of Facebook's blockchain ambitions. Using the Move programming language and an innovative object-centric architecture, Sui has rapidly emerged as one of the fastest-growing DeFi platforms, achieving $2B+ TVL in less than two years.

**Current Status (December 2025):**

- Market Cap: ~$6B (Ranked #13-20)
- Price: ~$2.00-3.00
- All-Time High: $5.34 (January 2025)
- TVL: $2.1B+ (Top 10 DeFi blockchain)
- Daily DEX Volume: $250M average
- Cumulative DEX Volume: $110B+
- Daily Active Addresses: 470,000+
- Consensus: Narwhal & Bullshark (DAG-based BFT)
- Block Finality: Sub-second

---

## The Meta/Facebook Origins (2017-2021)

### 2017-2019: Project Libra at Facebook
- **Facebook announced Libra** - ambitious stablecoin project to create global digital currency
- Led by David Marcus, head of Facebook Financial (F2)
- Assembled team of world-class cryptographers and engineers
- **Sam Blackshear created Move programming language** for Libra
- Move designed for secure asset management and formal verification

### 2020: Libra Rebrands to Diem
- **Regulatory pressure forced rebrand** from Libra to Diem
- Global regulators (Fed, ECB, G7) feared Facebook's scale could threaten financial stability
- Concerns about money laundering, terrorist financing, data privacy
- Project scaled back ambitions but continued development

### 2021: Diem Project Abandoned
- **Meta abandoned Diem under regulatory pressure**
- Sold Diem assets to Silvergate Bank for ~$200M
- Core technology (Move language, consensus mechanisms) not discarded
- **September 2021: Mysten Labs founded** by five former Meta/Diem engineers
- Team decided to build public blockchain using Diem's technology

**The Founding Team (All ex-Meta/Novi Research):**

- **Evan Cheng** - CEO, former Head of R&D at Novi, Technical Director at Meta, 10+ years at Apple
- **Sam Blackshear** - CTO, creator of Move programming language
- **Adeniyi Abiodun** - CPO, former Meta engineer
- **George Danezis** - Chief Scientist, cryptographer and researcher
- **Kostas Chalkias** - Chief Cryptographer, cryptography expert

**Significance:** Meta's failed Diem project became the foundation for two major blockchains - Sui and Aptos (founded by different Meta factions).

---

## Mysten Labs and Fundraising (2021-2022)

### December 2021: Series A Funding
- **$36M Series A led by a16z crypto**
- Validation from top-tier crypto VC
- Team began building Sui blockchain from scratch
- Focus on scalability, security, and developer experience

### September 2022: Series B Funding
- **$300M Series B led by FTX Ventures**
- Valuation: $2B+
- One of the largest infrastructure raises in crypto history
- **Third-largest infrastructure fundraise** behind zkSync ($458M) and Avalanche ($716M)
- Other investors: Jump Crypto, Binance Labs, Coinbase Ventures, Circle Ventures

**FTX Connection:**

- FTX Ventures led the round just 2 months before FTX collapse
- Unlike Solana, Sui was not deeply integrated with FTX
- Project survived FTX bankruptcy without major impact
- Demonstrated strong fundamentals beyond single backer

---

## Mainnet Launch and Airdrop Controversy (May 2023)

### May 3, 2023: Mainnet Launch
- **Sui mainnet went live publicly**
- Initial SUI price: $1.33
- Roughly 5% of total token supply in circulation at launch
- Total supply capped at 10 billion SUI

### The Airdrop Controversy
- **Sui announced NO AIRDROP** - major community backlash
- Decision was "publicly-stated, intentional"
- Airdrop hunters who participated in testnets felt betrayed
- Many had expected rewards for early participation

**Instead of Airdrop: Community Access Program**

- **Early token sale: 3 cents per SUI** on OKX, KuCoin, ByBit (U.S. users excluded)
- **Public token sale: 10 cents per SUI** (capped at 10,000 tokens per person)
- Token immediately traded at $1.33 (13x from early sale, 3.3x from public sale)
- Early buyers made massive profits, testers got nothing

**Community Reaction:**

- Critics blasted Sui for unfair tokenomics
- Compared unfavorably to other L1s with airdrops (Aptos, Arbitrum, Optimism)
- Some called it a "VC chain" favoring insiders
- Team defended decision, arguing airdrops create mercenary users

### Token Vesting and Lockups
- **One-year cliff for all initial investors** (May 2023 - May 2024)
- Investors blocked from transferring initial SUI stake to market
- Reduced immediate selling pressure
- Unlocks began May 2024

---

## Price Performance and Market Cycles (2023-2025)

### 2023: Launch and Bear Market
- **Launched at $1.33** (May 2023)
- Crypto market still in bear phase
- **Dropped to all-time low of $0.36** (late 2023)
- 73% decline from launch price
- "Dead on arrival" narratives from critics

### 2024: DeFi Boom and Recovery
- **TVL exploded 1,500%** in late 2023 / early 2024
- DeFi protocols (NAVI, Scallop, Cetus) gained traction
- Price recovered from $0.36 to $1.50+ range
- Network activity surged

### January 2025: All-Time High
- **Reached $5.34 ATH** (January 2025)
- 14.8x from all-time low of $0.36
- 4x from launch price of $1.33
- Driven by DeFi growth, institutional interest, broader crypto bull market

### Current (December 2025)
- Trading at $2.00-3.00 range
- Market cap ~$6B
- Established as top 15-20 cryptocurrency
- Volatility continues but fundamentals strengthening

---

## Technical Innovation: Object-Centric Architecture

### Object-Centric Model
Unlike traditional blockchains (Ethereum, Solana) that use account-based models, Sui uses an **object-centric data model**:

- Every piece of data exists as a distinct object with unique ID
- Assets (coins, NFTs, etc.) exist as independent objects, not account-bound entries
- Transactions take objects as input, produce new/modified objects as output
- Enables parallel execution of unrelated transactions

**Advantage:** Avoids bottlenecks seen in sequential blockchains like Ethereum.

### Parallel Execution
- **Dependencies explicitly encoded** in Move's strong ownership types
- Validators process transactions involving different objects in parallel
- Dramatically increases throughput
- Simple transactions (e.g., payments) bypass consensus entirely

**Performance:** Sub-second finality, high throughput (100,000+ TPS theoretical).

### Narwhal & Bullshark Consensus

**Narwhal (Mempool):**
- DAG-based (Directed Acyclic Graph) mempool
- Collects transaction batches from validators
- Ensures data availability for consensus

**Bullshark (Consensus):**
- Byzantine Fault Tolerant (BFT) protocol
- Orders data prepared by Narwhal
- Handles shared objects requiring consensus
- Optimized for high throughput in partially synchronous environments

**Innovation:** Decoupling transaction dissemination (Narwhal) from consensus (Bullshark) enables very high throughput.

### Move Programming Language
- **Created by Sam Blackshear at Meta for Diem**
- Rust-based language for secure smart contracts
- Strong ownership types prevent common bugs
- Formal verification capabilities
- Resource-oriented design (assets can't be copied or lost)

**Sui's Variant:** Modified version of Move optimized for object-centric model, differs from Aptos's implementation.

---

## The DeFi Explosion (2024-2025)

### TVL Growth Trajectory
- **May 2023 launch:** ~$25M TVL
- **Late 2023:** +1,500% TVL growth
- **Q1 2024:** $500M TVL milestone
- **Q2-Q3 2024:** $1B TVL (+42% growth)
- **Q4 2024:** $1.7B TVL (+67.9% QoQ, +698.5% YoY)
- **June 2025:** $2.1B+ TVL
- **Top 10 DeFi blockchain in less than one year**

### Trading Volume Explosion
- **Q3 2024:** $4.5B cumulative DEX volume
- **Q4 2024:** $44.3B cumulative DEX volume (+444.8% QoQ)
- **Average daily DEX volume:** $265.5M in Q4 2024 (+1,591.1% YoY)
- **Total cumulative DEX volume:** $110B+ by mid-2025

### Stablecoin Growth
- **January 2024:** $400M stablecoin market cap
- **May 2025:** $1.2B stablecoin market cap (+200%)
- **Monthly stablecoin transfer volume:** $70B+
- Circle's USDC officially integrated

---

## Major DeFi Protocols

### DeepBook (Native Liquidity Layer)
- Sui's native on-chain order book and liquidity protocol
- **Cumulative volume:** $500B+ by Q3 2024
- **DeepBook V3 launched October 2024:**
  - Dynamic trading fees
  - Improved gas efficiency
  - Flash loans
  - Shared liquidity across pools
  - DEEP token for governance

### NAVI Protocol (Lending & Liquid Staking)
- DeFi protocol combining DEX aggregation, liquid staking, and lending
- **TVL:** $714M
- **Users:** 800,000+
- Launched Leveraged Strategies for accessible leverage
- One of Sui's flagship DeFi apps

### Scallop (Lending Protocol)
- **Total lending/borrowing volume:** $100B+ (Q3 2024)
- Strategic investment from Sui Foundation
- Launched Scallop Lite and Isolated Asset Pools
- Focused on money markets and lending

### Cetus Protocol (DEX)
- Leading decentralized exchange on Sui
- **Q4 2024 TVL:** $213M
- **Average daily DEX volume:** $187.8M (Q4 2024)
- **Volume growth:** 522.34% (Q4 2024)
- Concentrated liquidity AMM

### Turbos Finance (DEX)
- **Q3 2024 avg daily volume:** $7.5M
- **Q4 2024 avg daily volume:** $18.6M (+148%)
- Known for supporting meme tokens
- Innovative trading mechanisms

### Other Key Projects:
- **Aftermath Finance** - DeFi protocol
- **Kriya DEX** - Decentralized exchange
- **Haedal Protocol** - Liquid staking
- **Suilend** - Lending protocol

---

## Sui vs Aptos: The Diem Heirs

Both Sui and Aptos are "Facebook's blockchain heirs" - built by different factions of the Meta Diem team using the Move language.

### Key Differences:

**Team Origins:**
- **Sui:** Novi Research team (Evan Cheng, Sam Blackshear - Move creator)
- **Aptos:** Diem core blockchain team (Mo Shaikh, Avery Ching)

**Architecture:**
- **Sui:** Object-centric model (unique innovation)
- **Aptos:** Account-centric model (similar to Ethereum/Solana)

**Move Language:**
- **Sui:** Modified Move optimized for objects
- **Aptos:** Closer to original Diem Move implementation

**Consensus:**
- **Sui:** Narwhal & Bullshark (DAG-based)
- **Aptos:** AptosBFT (BFT-based)

**Performance:**
- **Sui:** Sub-second finality, parallel execution
- **Aptos:** ~1 second finality, parallel execution

**Adoption (2025):**
- **Sui:** $2.1B TVL, 470K daily active users, leading in DeFi growth
- **Aptos:** ~$800M TVL, strong but trailing Sui

**Market Cap:**
- **Sui:** ~$6B (#13-20)
- **Aptos:** ~$5B (#15-25)

**Verdict:** Sui has pulled ahead in adoption metrics and DeFi ecosystem, while Aptos maintains strong technical foundation.

---

## Key Controversies and Challenges

### No Airdrop Decision
- Alienated early testnet participants
- "VC chain" criticism
- Community felt betrayed after months of testing
- Defenders argue airdrops attract mercenaries, not long-term users

### Tokenomics Criticism
- Heavily VC-backed with large allocations
- One-year cliff, but long-term unlocks create selling pressure
- Early sale prices (3 cents) vs public (10 cents) created wealth disparity
- Compared unfavorably to more "fair" launches

### Meta Association Concerns
- Some resist "Facebook's blockchain"
- Privacy concerns due to Meta lineage
- Centralization fears (though Mysten Labs is independent)
- Regulatory scrutiny due to Diem heritage

### Competition
- Aptos as direct competitor (same origins)
- Ethereum dominates smart contracts
- Solana leads in speed/memecoins
- Must differentiate in crowded L1 space

### Relatively New
- Launched May 2023, less than 2 years old
- Network stability unproven over long term
- Smaller developer community than Ethereum
- Ecosystem still maturing

---

## Sui's Evolution: Platform Phases

### 2017-2020: Meta's Diem Era
- Libra/Diem project at Facebook
- Move language created
- Regulatory pressure kills project

### 2021-2022: Mysten Labs Formation
- September 2021: Mysten Labs founded
- December 2021: $36M Series A
- September 2022: $300M Series B
- Building Sui blockchain

### May 2023: Controversial Launch
- Mainnet goes live
- No airdrop backlash
- Token sales at 3-10 cents
- Price launches at $1.33

### Late 2023: Bear Market Bottom
- Price crashes to $0.36
- TVL starts growing despite price
- DeFi protocols gaining traction

### 2024: DeFi Breakout Year
- TVL explodes from $25M to $1.7B
- Top 10 DeFi blockchain achieved
- Daily DEX volume surges
- Institutional integrations (Circle USDC)

### 2025: Maturation and Growth
- New ATH $5.34 (January)
- $2.1B+ TVL (June)
- $110B+ cumulative DEX volume
- 470K+ daily active users
- Established as major L1

---

## Sui by the Numbers (Historical)

| Quarter | Price Range | TVL | Notable Event |
|---------|-------------|-----|---------------|
| Q2 2023 | $1.33 → $0.50 | $25M | Mainnet launch, initial decline |
| Q3 2023 | $0.50 → $0.40 | $100M | Bear market bottom |
| Q4 2023 | $0.36 → $1.20 | $500M | DeFi growth begins, +1,500% TVL |
| Q1 2024 | $1.20 → $1.50 | $800M | Continued DeFi expansion |
| Q2-Q3 2024 | $1.00 → $2.00 | $1.0B | Top 10 DeFi blockchain |
| Q4 2024 | $2.00 → $4.00 | $1.7B | Trading volume explosion |
| Q1 2025 | $4.00 → $5.34 (ATH) | $1.5B | New all-time high |
| Q2 2025 | $3.00 → $2.00 | $2.1B+ | Consolidation, continued growth |

---

## The Future of Sui

### Bulls Argue:
- **Fastest-growing DeFi blockchain** - $2.1B TVL in under 2 years
- **Technical innovation** - Object-centric model enables true parallel execution
- **World-class team** - Ex-Meta engineers with proven track record
- **Move language advantage** - Superior security and formal verification
- **Institutional backing** - a16z, Circle, Jump, Coinbase, Binance
- **Network effects building** - 470K+ daily active users, strong developer growth
- **Early stage** - Massive room to grow vs Ethereum/Solana
- **$110B+ trading volume** proves product-market fit

### Bears Argue:
- **VC-heavy tokenomics** - Large insider allocations, ongoing unlocks
- **No airdrop** damaged community trust
- **Meta association** - Privacy and centralization concerns
- **Aptos competition** - Similar tech, same origins, fragments liquidity
- **Ethereum dominance** - Hard to dislodge entrenched leader
- **Relatively unproven** - Less than 2 years old, long-term stability unknown
- **Smaller ecosystem** - Fewer developers and projects than established L1s

### Key Developments to Watch:
- **Gaming and NFT expansion** - Leveraging low fees and high throughput
- **Mobile integration** - Sui designed for mobile-first applications
- **Enterprise adoption** - Meta connections could lead to corporate use cases
- **zkLogin and social login** - Simplified onboarding for mainstream users
- **Cross-chain bridges** - Integration with Ethereum, Solana ecosystems
- **Token unlocks** - How market handles ongoing vesting schedules

### Key Questions for Next 5 Years:
- Can Sui surpass $10B TVL and compete with Ethereum L2s?
- Will object-centric model prove superior to account-based designs?
- Can Sui differentiate from Aptos or will both struggle against Ethereum/Solana?
- Will Move language attract developers from Solidity/Rust?
- Can Sui reach $10+ price ($100B+ market cap) or fade into obscurity?
- Will Meta ever use Sui for payments/Web3 integration?

---

## Sources

- [Meet the Founders of Sui Blockchain - Backpack](https://learn.backpack.exchange/articles/who-founded-sui-blockchain)
- [Mysten Labs - CryptoSlate](https://cryptoslate.com/companies/mysten-labs/)
- [The Sui Story: How Evan Cheng Forged a Web3 Leader from Meta's Ruins - CryptoBlogs](https://www.cryptoblogs.io/the-sui-story-how-evan-cheng-forged-a-web3-leader-from-metas-ruins/)
- [Sui vs. Aptos: How Are These Blockchains Different? - Bankless](https://www.bankless.com/sui-vs-aptos)
- [Sui (SUI) and Aptos (APT): The Diem Legacy Transforming Web3 - Stakin](https://stakin.com/blog/sui-sui-and-aptos-apt-comparing-move-layer-1-heavyweights)
- [Sui Network to Issue Token Following Exchange Sale; Airdrop Hunters Dismayed - CoinDesk](https://www.coindesk.com/business/2023/04/26/sui-network-to-issue-token-following-exchange-sale-airdrop-hunters-dismayed)
- [Sui Mainnet Goes Live - CoinDesk](https://www.coindesk.com/business/2023/05/03/sui-mainnet-goes-live-token-trades-at-133)
- [Sui Q1 2025 DeFi Roundup - Sui Blog](https://blog.sui.io/q1-2025-defi-roundup/)
- [Sui's DeFi Ecosystem Thrives in Q4 2024 with Record Growth - Blockchain News](https://blockchain.news/news/suis-defi-ecosystem-thrives-q4-2024)
- [State of Sui Q3 2025 - Messari](https://messari.io/report/state-of-sui-q3-2025)
- [Sui's Booming DeFi Ecosystem Surpasses $2B TVL - Sui Blog](https://blog.sui.io/2-billion-tvl-milestone-defi/)
- [Sui Becomes a Top 10 DeFi Blockchain in Less Than a Year - CoinDesk](https://www.coindesk.com/tech/2024/01/31/sui-becomes-a-top-10-defi-blockchain-in-less-than-a-year)
- [What is Sui Network? How it works - Kraken](https://www.kraken.com/learn/what-is-sui-network-sui)
- [Consensus - Sui Documentation](https://docs.sui.io/concepts/sui-architecture/consensus)

**Data Last Updated:** December 8, 2025
