# How Betting Apps Do Market Making

## The Core Model: House vs Exchange

Betting platforms operate under two primary models:

**Traditional Sportsbooks (DraftKings, FanDuel, BetMGM)**

- Users bet **against the house**
- Bookmaker sets the odds and takes the opposite side of every bet
- Profit comes from the "vig" (vigorish) or "juice" built into odds
- House has exposure to outcomes but manages risk through balancing

**Betting Exchanges (Betfair, newer prediction markets)**

- Users bet **against each other** (peer-to-peer)
- Platform acts as pure middleman/facilitator
- Prices adjust dynamically based on supply and demand
- Platform profits from small commission/fees on winning bets
- No direct outcome exposure for the platform

This document focuses on traditional sportsbook market making, which is what most major betting apps (DraftKings, FanDuel, Caesars, BetMGM) use.

## The Overround: Built-In Profit Margin

The fundamental way bookmakers make money is through the **overround** (also called "vig", "juice", or "margin").

### How It Works

Instead of offering "fair" odds that reflect true probabilities, bookmakers adjust odds so that the sum of implied probabilities exceeds 100%.

**Example: Coin Flip**

*Fair Odds (no vig):*

- Heads: +100 (50% implied probability)
- Tails: +100 (50% implied probability)
- Total: 100%

*Bookmaker Odds (with 4.5% vig):*

- Heads: -110 (52.4% implied probability)
- Tails: -110 (52.4% implied probability)
- Total: 104.8%

The 4.8% difference is the bookmaker's edge. If betting is balanced equally on both sides, the bookmaker is guaranteed to profit 4.8% of total handle regardless of outcome.

### Typical Margins by Sport

- **Two-way markets** (NFL, NBA spreads): 4-5% margin
- **Three-way markets** (soccer: win/draw/win): 6-8% margin
- **Futures** (championship winner): 20-40% margin
- **Parlays**: Compounding margins (highly profitable for books)
- **Same-game parlays**: 15-30%+ margins (often mispriced)

## The Balancing Act: Risk Management

The ideal scenario for a bookmaker is a **balanced book** - equal money wagered on all possible outcomes.

### Why Balancing Matters

**Balanced Book Example:**

- $100,000 bet on Team A (-110)
- $100,000 bet on Team B (-110)
- Total handle: $200,000

If Team A wins:

- Pay out: $190,909 ($100k stake + $90,909 winnings)
- Collect from Team B bettors: $100,000
- Net profit: $9,091 (~4.5% of handle)

If Team B wins: Same math, same $9,091 profit.

**The bookmaker doesn't care who wins - they profit either way.**

### Unbalanced Book Risk

If $150k is on Team A and only $50k on Team B:

- If Team A wins: Pay out $286,364, collect $50k → **Loss of $86,364**
- If Team B wins: Pay out $95,455, collect $150k → **Profit of $54,545**

Bookmaker is now exposed to outcome risk and is "rooting" for Team B.

## How Bookmakers Balance Books

### 1. Dynamic Odds Adjustment

When too much money comes in on one side, bookmakers adjust odds to make the other side more attractive.

**Example:**

- Opening line: Patriots -3 (-110) / Dolphins +3 (-110)
- Heavy betting on Patriots
- Adjusted line: Patriots -3.5 (-115) / Dolphins +3.5 (-105)

The adjustment:

- Makes Patriots less attractive (worse price, bigger spread)
- Makes Dolphins more attractive (better price, smaller underdog)
- Incentivizes new bets to flow toward Dolphins side

### 2. Betting Limits

When sharp (professional) bettors identify value and place large bets on one side, bookmakers may:

- **Limit bet sizes** for sharp accounts
- **Reduce maximum stakes** on specific markets
- **Ban accounts** that consistently win (soft books)
- **Welcome sharp action** and use it to set better lines (sharp books like Pinnacle)

### 3. Layoff Bets

If a bookmaker has too much exposure on one outcome, they can place bets with other bookmakers to hedge risk.

Example:

- Local bookie has $500k on hometown team
- To reduce exposure, places $200k on opponent with another book
- Reduces potential loss if hometown team wins

## Sharp Books vs Soft Books

### Sharp Books (Market Makers)

**Examples:** Pinnacle, Circa Sports, CRIS

**Characteristics:**

- Set their own opening lines using sophisticated models
- Welcome professional/sharp bettors
- Use sharp action as "market research" to improve lines
- Quickly adjust odds based on sharp betting patterns
- Lower margins (2-3% typical)
- Higher betting limits
- Other books follow their lines

**Philosophy:** Sharp bettors are price discovery mechanisms. Their bets reveal new information, which helps set more accurate odds.

### Soft Books (Market Followers)

**Examples:** Most consumer betting apps (DraftKings, FanDuel, BetMGM)

**Characteristics:**

- Copy opening lines from sharp books
- Limit or ban winning players
- Target recreational bettors
- Higher margins (4.5-8%)
- Heavy marketing and promotions
- Focus on player acquisition, not price accuracy

**Philosophy:** Maximize revenue from recreational players who bet on favorites, popular teams, and emotional/fun bets rather than value.

## Modern Market Making: Technology & Algorithms

### Automated Odds Management

2025-era sportsbooks use sophisticated systems:

**Real-Time Data Ingestion:**

- Live betting action across all books
- Competitor odds (odds feeds from aggregators)
- Statistical models and probabilities
- News events (injuries, weather, lineup changes)
- Sharp book line movements

**Algorithmic Trading:**

- Automated odds adjustments every few seconds
- Machine learning models predict betting patterns
- Risk exposure monitored across all markets
- Dynamic vig adjustment based on confidence

**Platform Examples:**

- Kambi (powers DraftKings and others)
- Scientific Games OpenSports (powers FanDuel)
- OddsMatrix
- SBTech

### Risk Management Systems

Modern sportsbooks use centralized risk management platforms:

- **Real-time exposure monitoring** across all events
- **Automated alerts** when exposure exceeds thresholds
- **Player profiling** to identify sharp vs recreational bettors
- **Fraud detection** for arbitrage and coordinated betting
- **Liability management** for parlays and correlated bets

## Speculative Positioning: Beyond Balancing

Contrary to traditional belief, bookmakers don't always aim for perfect balance. Research shows they often take **speculative positions**.

### Why Take Risk?

**Information Advantage:** Bookmakers have more data and better models than most bettors. If they're confident their odds are accurate, they can profit by letting the book be imbalanced.

**Example:**

- Bookmaker sets Patriots -7 based on sophisticated model
- Public bets heavily on Patriots (recreational bias)
- Bookmaker doesn't move line because they believe Patriots will win by < 7
- If correct, they profit from both the vig AND the imbalanced bets

### The Two Profit Sources

1. **Riskless profit:** Vig/margin from balanced book
2. **Speculative profit:** Expected gains from "wrong odds" that attract one-sided action

Research suggests sharp bookmakers actively use both strategies to maximize expected profits.

## Same-Game Parlays: The Profit Goldmine

Same-game parlays (SGPs) have become hugely profitable for sportsbooks:

**Why SGPs Are Profitable:**

- **Correlation mispricing:** Hard to accurately price correlated outcomes
- **Opaque odds:** Bettors can't easily compare SGP odds across books
- **High margins:** 15-30%+ margins vs 4-5% on straight bets
- **Recreational appeal:** Fun, lottery-like bets attract casual players

**The Risk:**

When correlation models are wrong, sportsbooks can face massive payouts. Several US operators issued profit warnings in 2024-2025 after mispriced SGPs led to multi-million dollar losses.

## How Sportsbooks Actually Make Money

### Revenue Streams

1. **Hold %** (realized profit margin)
   - Industry average: 5-10% of handle
   - Lower on straight bets, higher on parlays/futures

2. **Volume**
   - More total betting handle = more absolute profit
   - Marketing and promotions to acquire customers

3. **Product Mix**
   - Push customers toward high-margin bets (parlays, SGPs, futures)
   - Limit low-margin bets (sharp two-way markets)

4. **Player Lifetime Value**
   - Recreational players who lose over time
   - Limit/ban winners, retain losers

### Example P&L

**Monthly Handle:** $100M

**Product Mix:**

- Straight bets: $70M @ 5% hold = $3.5M profit
- Parlays: $20M @ 15% hold = $3.0M profit
- Same-game parlays: $10M @ 25% hold = $2.5M profit

**Total Gross Gaming Revenue (GGR):** $9M (9% blended hold)

**Costs:**

- Taxes: ~20-50% depending on state (let's say 30% = $2.7M)
- Marketing: $2M
- Platform/tech: $1M
- Operations: $1M

**Net profit:** $2.3M/month (2.3% of handle)

## The Bottom Line

Betting apps make money through:

1. **The Vig:** Building a margin into every market ensures profit if balanced
2. **Balancing:** Adjusting odds dynamically to attract offsetting action
3. **Risk Management:** Limiting sharp players, managing exposure, hedging when needed
4. **Speculative Edge:** Taking intelligent risk when confident in their numbers
5. **Product Engineering:** Steering customers toward high-margin bets (parlays, SGPs)
6. **Volume:** Marketing heavily to acquire recreational players and maximize total handle

The key insight: **Bookmakers aren't trying to predict winners - they're managing a two-sided marketplace to extract the vig while controlling risk.**

---

## Sources

- [How Do Bookies Set Odds? - OddsMatrix](https://oddsmatrix.com/bookmaker-odds/)
- [The Role of Bookmakers - LSports](https://www.lsports.eu/blog/the-role-of-bookmakers-how-sports-betting-odds-are-set/)
- [How Bookies Stay Profitable - SportBex](https://sportbex.com/blog/bookies-set-sports-betting-odds-stay-profitable/)
- [Bookmaker Margins Explained - Online Betting UK](https://www.onlinebetting.org.uk/betting-guides/how-do-bookmakers-set-odds-and-make-money.html)
- [How Sportsbooks Set Odds - Bettors Insider](https://www.bettorsinsider.com/news/2024/03/15/how-do-sportsbooks-set-odds)
- [Sharp vs Soft Books - Outlier](https://help.outlier.bet/en/articles/9922960-how-sportsbooks-set-odds-soft-vs-sharp-books)
- [Vigorish - Wikipedia](https://en.wikipedia.org/wiki/Vigorish)
- [Sportsbook Risk Management - Kambi](https://www.kambi.com/news-insights/why-risk-management-has-never-been-more-important-for-sportsbook-operators/)
- [House Edge in Sports Betting - Altenar](https://altenar.com/blog/house-edge-in-sports-betting-how-the-best-sportsbooks-manage-their-edge/)
- [How Sportsbooks Make Money - Risk.inc](https://www.risk.inc/blog/how-sportsbooks-make-money---a-look-inside-the-online-betting-business)
- [DraftKings, FanDuel Enter Prediction Markets - Axios](https://www.axios.com/2025/11/13/draftkings-fanduel-prediction-markets)
- [Prediction Markets vs Sportsbooks - Front Office Sports](https://frontofficesports.com/u-s-sports-betting-in-flux-as-fanduel-draftkings-enter-prediction-markets/)
- [FanDuel vs DraftKings - CBS Sports](https://www.cbssports.com/betting/news/fanduel-vs-draftkings-which-is-the-bigger-sports-betting-giant/)
