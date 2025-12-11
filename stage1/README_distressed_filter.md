# Ultra-Distressed Filter: Technical Documentation

## Overview

The **ultra-distressed filter** detects anomalous prices in distressed corporate bond transaction data. These anomalies include ultra-low prices, upward spikes, persistent plateaus, and intraday inconsistencies that indicate data quality issues rather than genuine market conditions.

The algorithm employs four independent filters operating on daily aggregated price data, using lookback/lookforward windows, round-number heuristics, and ratio-based thresholds to identify error candidates.

**When This Filter Is Most Useful**: This filter primarily addresses data quality issues that arise when Stage 0 applies minimal or no volume filtering, or when filtering only on par volume. Such configurations retain micro trades at obscure round numbers that may represent placeholder values or data entry errors. This filter provides a cleanup mechanism for these edge cases.
These bond prices are often associated with debt issues trading in default. Thank you to Colin Philipps (FINRA) for some clarification on these extremely low/rounded prices in TRACE, "_... not the TRACE system adding anything to the tape but are transactions reported broker dealers. Likely they are cleaning up positions/books. TRACE only transmits what is submitted to the system._".

---

## Framework

### Problem Statement

Given a time-ordered sequence of daily aggregated prices $\{P_1, P_2, \ldots, P_n\}$ for a bond, identify observations $P_i$ that represent **candidate errors** characterized by one or more of the following:

1. **Ultra-low anomaly**: Price $P_i$ is significantly lower than surrounding prices (downward outlier)
2. **Upward spike**: Price $P_i$ is significantly higher than preceding prices and quickly recovers
3. **Persistent plateau**: Sequence of identical ultra-low or round prices lasting multiple days
4. **Intraday inconsistency**: Large discrepancies between different intraday price measures (first, last, min, max)

The algorithm must distinguish these errors from **genuine distressed pricing** (e.g., default events, bankruptcy) that persist consistently over time.

### Core Algorithm Components

#### 1. Round Number Detection

For all filters, compute a **round number mask** to identify prices at suspicious round levels:

$$
R_i = \begin{cases}
1 & \text{if } \exists r \in \mathcal{R} : |P_i - r| < \epsilon_{\text{round}} \\
0 & \text{otherwise}
\end{cases}
$$

where:
- $R_i$ = binary indicator (1 if price $P_i$ matches a suspicious round number, 0 otherwise)
- $r$ = an individual round number from the set $\mathcal{R}$
- $\mathcal{R} = \{0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00\}$ = suspicious round numbers (in % of par)
- $\epsilon_{\text{round}} = 0.0001$ = tolerance for round detection

**Important: Price Units.** All prices in this filter are expressed as **% of par**. With par = $1,000:

| Price (% of par) | Dollar Value |
|------------------|--------------|
| 0.001 | $0.01 |
| 0.01 | $0.10 |
| 0.05 | $0.50 |
| 0.10 | $1.00 |
| 0.25 | $2.50 |
| 0.50 | $5.00 |
| 0.75 | $7.50 |
| 1.00 | $10.00 |

**Note on Round Numbers**: We are in contact with FINRA/TRACE to understand these odd rounded numbers that often occur in recognizable sequences. According to Colin Philipps (FINRA), it could be associated with dealers "cleaning-up" the books.

---

## Filter Components

### Filter 1: Anomaly Detection (Downward Outliers)

**Purpose**: Detect isolated ultra-low prices that are outliers compared to surrounding normal prices.

#### Opening Conditions

A row $i$ becomes a **candidate for anomaly flagging** if:

$$
P_i < \tau_{\text{low}} \quad \text{OR} \quad R_i = 1
$$

where:
- $P_i$ = price at row $i$ (in % of par)
- $R_i$ = round number indicator (1 if $P_i$ matches a suspicious round number, 0 otherwise; see Section 1)
- $\tau_{\text{low}} = 0.10$ = ultra-low threshold (**0.10% of par = $1**)

**Interpretation**: A price is a candidate if it is below $1 (extremely low) OR matches a suspicious round number.

#### Surrounding Price Analysis

For each candidate at row $i$, collect **surrounding prices** from lookback and lookforward windows:

$$
\mathcal{S}_i = \{P_j : j \in [i-L_{\text{back}}, i-1] \cup [i+1, i+L_{\text{fwd}}], \, P_j > P_i, \, P_j \text{ valid}\}
$$

where:
- $L_{\text{back}} = 5$ = lookback window (days)
- $L_{\text{fwd}} = 5$ = lookforward window (days)
- Only prices **strictly greater than** $P_i$ are collected

Compute the **median of surrounding prices**:

$$
M_{\text{surr}} = \text{median}(\mathcal{S}_i)
$$

#### Anomaly Criterion

Flag row $i$ as anomalous if:

$$
\frac{M_{\text{surr}}}{P_i + \epsilon} \geq \rho_{\text{anomaly}}
$$

where:
- $\rho_{\text{anomaly}} = 3.0$ = minimum ratio for anomaly detection
- $\epsilon = 10^{-10}$ = numerical stability constant

**Interpretation**: The median of surrounding (higher) prices is at least 3× higher than the current price.

#### Numerical Example

| Day | Price (% par) | Price ($) | Candidate? | Surrounding (> $P_i$) | Median | Ratio | Flagged? |
|-----|---------------|-----------|------------|----------------------|--------|-------|----------|
| 1 | 45.0 | $450 | No | — | — | — | No |
| 2 | 44.5 | $445 | No | — | — | — | No |
| 3 | **0.05** | **$0.50** | Yes (< 0.10 AND round) | {45.0, 44.5, 45.2, 44.8} | 44.85 | 897× | **Yes** |
| 4 | 45.2 | $452 | No | — | — | — | No |
| 5 | 44.8 | $448 | No | — | — | — | No |

Day 3 is flagged because: (1) price 0.05 < τ_low = 0.10, (2) 0.05 is a round number, and (3) surrounding median / current price = 44.85 / 0.05 ≈ 897 ≥ 3.0.

#### Anomaly Classification

Flag types:
- **`ultra_low`** ($t = 1$): $P_i < \tau_{\text{low}}$ AND NOT $R_i$
- **`round_number`** ($t = 2$): $R_i$ AND NOT $(P_i < \tau_{\text{low}})$
- **`ultra_low_round`** ($t = 3$): $P_i < \tau_{\text{low}}$ AND $R_i$

---

### Filter 2: Spike Detection (Upward Outliers)

**Purpose**: Detect temporary upward price spikes that quickly revert to lower levels.

#### Opening Conditions

A row $i$ becomes a **candidate for spike flagging** if:

$$
P_i > \tau_{\text{high}} \quad \text{OR} \quad (R_i = 1 \text{ AND } P_i > 0.50)
$$

where:
- $P_i$ = price at row $i$ (in % of par)
- $R_i$ = round number indicator (1 if $P_i$ matches a suspicious round number, 0 otherwise; see Section 1)
- $\tau_{\text{high}} = 5.0$ = high spike threshold (**5% of par = $50**)
- Round number condition requires $P_i > 0.50$ (**0.50% of par = $5**) to avoid flagging tiny round numbers

**Interpretation**: A price is a candidate if it exceeds $50 OR is a suspicious round number above $5.

#### Pre-Spike Price Analysis

For each candidate at row $i$, collect **preceding prices** from lookback window:

$$
\mathcal{P}_{\text{pre}} = \{P_j : j \in [i-L_{\text{back}}, i-1], \, P_j < P_i, \, P_j \text{ valid}\}
$$

where only prices **strictly less than** $P_i$ are collected.

Compute the **median of pre-spike prices**:

$$
M_{\text{pre}} = \text{median}(\mathcal{P}_{\text{pre}})
$$

#### Spike Magnitude Check

Check if spike is sufficiently large:

$$
\frac{P_i}{M_{\text{pre}} + \epsilon} \geq \rho_{\text{spike}}
$$

where $\rho_{\text{spike}} = 3.0$ = minimum spike ratio.

#### Recovery Check

Define the **recovery threshold**:

$$
P_{\text{recovery}} = M_{\text{pre}} \cdot \rho_{\text{recovery}}
$$

where $\rho_{\text{recovery}} = 2.0$ = recovery ratio.

Search forward for **evidence of recovery**:

$$
\exists j \in [i+1, i+L_{\text{fwd}}] : P_j \leq P_{\text{recovery}} \text{ AND } P_j \text{ valid}
$$

**Decision**: Flag row $i$ as spike if **both** magnitude check AND recovery check pass.

#### Numerical Example

| Day | Price (% par) | Price ($) | Candidate? | Pre-prices (< $P_i$) | Median | Spike Ratio | Recovery? | Flagged? |
|-----|---------------|-----------|------------|---------------------|--------|-------------|-----------|----------|
| 1 | 12.5 | $125 | No | — | — | — | — | No |
| 2 | 11.8 | $118 | No | — | — | — | — | No |
| 3 | **100.0** | **$1,000** | Yes (round & > 0.50) | {12.5, 11.8} | 12.15 | 8.23× | Check day 4 | — |
| 4 | 12.2 | $122 | — | — | — | — | 12.2 ≤ 24.3 ✓ | — |
| 5 | 11.5 | $115 | No | — | — | — | — | No |

Day 3 is flagged because:
1. Candidate: 100.0 is a round number AND > 0.50 ✓
2. Pre-spike median = (11.8 + 12.5) / 2 = 12.15
3. Spike ratio = 100.0 / 12.15 = 8.23 ≥ 3.0 ✓
4. Recovery threshold = 12.15 × 2.0 = 24.3; Day 4 price 12.2 ≤ 24.3 ✓

#### Spike Classification

Flag types:
- **`high_spike`** ($s = 1$): $P_i > \tau_{\text{high}}$ AND NOT $R_i$
- **`round_spike`** ($s = 2$): $R_i$ AND NOT $(P_i > \tau_{\text{high}})$
- **`high_round_spike`** ($s = 3$): $P_i > \tau_{\text{high}}$ AND $R_i$

---

### Filter 3: Plateau Detection (Persistent Identical Prices)

**Purpose**: Detect sequences of consecutive days with identical ultra-low or round prices.

#### Opening Conditions

A row $i$ becomes the **start of a potential plateau** if:

$$
P_i < \tau_{\text{plateau}} \quad \text{OR} \quad R_i = 1
$$

where:
- $P_i$ = price at row $i$ (in % of par)
- $R_i$ = round number indicator (1 if $P_i$ matches a suspicious round number, 0 otherwise; see Section 1)
- $\tau_{\text{plateau}} = 0.15$ = plateau threshold (**0.15% of par = $1.50**)

**Interpretation**: Only ultra-low prices (< $1.50) or round numbers can start a plateau.

#### Plateau Identification

Starting from row $i$, find the **extent of exact price equality**:

$$
\ell_{\text{plateau}} = \max\{k : P_{i+j} = P_i \text{ for all } j \in [0, k-1]\}
$$

where $\ell_{\text{plateau}}$ is the length of consecutive **identical** prices (exact equality required).

#### Minimum Run Length

A plateau is considered for flagging only if:

$$
\ell_{\text{plateau}} \geq \ell_{\min}
$$

where $\ell_{\min} = 2$ (default minimum plateau days).

#### Plateau Suspicion Criteria

A plateau is flagged as **suspicious** if **any** of the following hold:

**Criterion 1: Pre-plateau displacement**

If $i > 0$ and $P_{i-1}$ is valid:

$$
\frac{P_{i-1}}{P_i + \epsilon} \geq \rho_{\text{plateau}}
$$

**Criterion 2: Post-plateau displacement**

Let $j = i + \ell_{\text{plateau}}$ (first row after plateau). If $j < n$ and $P_j$ is valid:

$$
\frac{P_j}{P_i + \epsilon} \geq \rho_{\text{plateau}}
$$

**Criterion 3: Round number plateau**

$$
R_i = 1
$$

where $\rho_{\text{plateau}} = 3.0$ = pre/post price ratio threshold.

**Decision**: If any criterion holds, flag **all rows** in $[i, i+\ell_{\text{plateau}}-1]$ and assign plateau ID.

#### Numerical Example

| Day | Price (% par) | Price ($) | Candidate? | Plateau? | Pre-ratio | Post-ratio | Round? | Flagged? |
|-----|---------------|-----------|------------|----------|-----------|------------|--------|----------|
| 1 | 55.0 | $550 | No | — | — | — | — | No |
| 2 | 54.8 | $548 | No | — | — | — | — | No |
| 3 | **0.01** | **$0.10** | Yes (< 0.15 & round) | Start | 5,480× | — | Yes | **Yes** |
| 4 | **0.01** | **$0.10** | — | Continue | — | — | — | **Yes** |
| 5 | **0.01** | **$0.10** | — | Continue | — | 5,550× | — | **Yes** |
| 6 | 55.5 | $555 | No | — | — | — | — | No |

Days 3-5 are flagged because:
1. Price 0.01 < τ_plateau = 0.15 AND is a round number ✓
2. Plateau length = 3 ≥ ℓ_min = 2 ✓
3. Suspicion criteria (all three met!):
   - Pre-plateau: 54.8 / 0.01 = 5,480 ≥ 3.0 ✓
   - Post-plateau: 55.5 / 0.01 = 5,550 ≥ 3.0 ✓
   - Round number: 0.01 ∈ R ✓

---

### Filter 4: Intraday Inconsistency Detection

**Purpose**: Detect days where intraday price measures (first, last, high, low) show implausibly large discrepancies.

#### Price Range Analysis

For each row $i$, collect available intraday prices:

$$
\mathcal{I}_i = \{P_i^{\text{first}}, P_i^{\text{last}}, P_i^{\text{high}}, P_i^{\text{low}}\}
$$

Compute **intraday range** and **mean**:

$$
R_{\mathrm{intraday}} = \max(\mathcal{I}_i) - \min(\mathcal{I}_i)
$$

$$
\bar{P}_{\mathrm{intraday}} = \frac{1}{|\mathcal{I}_i|} \sum_{p \in \mathcal{I}_i} p
$$

#### Inconsistency Criterion

Flag row $i$ as inconsistent if **both** conditions hold:

$$
\exists p \in \mathcal{I}_i : p < \tau_{\mathrm{intraday}}
$$

$$
\frac{R_{\mathrm{intraday}}}{\bar{P}_{\mathrm{intraday}}} > \gamma_{\mathrm{range}}
$$

where:
- $\tau_{\mathrm{intraday}} = 20.0$ = price threshold (**20% of par = $200**; only check days with at least one low price)
- $\gamma_{\mathrm{range}} = 0.75$ = maximum allowed range as fraction of mean (75%)

**Interpretation**: At least one intraday price is below $200, AND the intraday range exceeds 75% of the mean price.

#### Numerical Example

| Row | `prc_hi` (% par) | `prc_lo` (% par) | Min | Max | Mean | Range | Range/Mean | Flagged? |
|-----|------------------|------------------|-----|-----|------|-------|------------|----------|
| A | 88.5 | 87.2 | 87.2 | 88.5 | 87.85 | 1.3 | 1.5% | No (no price < 20) |
| B | 89.0 | **0.10** | 0.10 | 89.0 | 44.55 | 88.9 | **199%** | **Yes** |
| C | 15.0 | 14.5 | 14.5 | 15.0 | 14.75 | 0.5 | 3.4% | No (range/mean ≤ 0.75) |

Row B is flagged because:
1. At least one price < 20: prc_lo = 0.10 < 20 ✓
2. Range/Mean = 88.9 / 44.55 = 1.99 > 0.75 ✓

**Note**: Row B shows an erroneous low print ($1 vs $890) on an otherwise normally-priced day.

---

## Function Signature

```python
def ultra_distressed_filter(
    df: pd.DataFrame,
    *,
    id_col: str = "cusip_id",
    date_col: str = "trd_exctn_dt",
    price_col: str = "pr",
    enable_anomaly_filter: bool = True,
    ultra_low_threshold: float = 0.10,
    min_normal_price_ratio: float = 3.0,
    enable_spike_filter: bool = True,
    high_spike_threshold: float = 5.0,
    min_spike_ratio: float = 3.0,
    recovery_ratio: float = 2.0,
    enable_plateau_filter: bool = True,
    plateau_ultra_low_threshold: float = 0.15,
    min_plateau_days: int = 2,
    suspicious_round_numbers: List[float] = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00],
    round_tolerance: float = 0.0001,
    lookback: int = 5,
    lookforward: int = 5,
    pre_post_price_ratio: float = 3.0,
    enable_intraday_filter: bool = True,
    price_cols: list = ["prc_ew", "prc_vw", "prc_first", "prc_last"],
    intraday_range_threshold: float = 0.75,
    intraday_price_threshold: float = 20.0,
    verbose: bool = False,
    keep_flag_columns: bool = False,
) -> pd.DataFrame
```

---

## Parameters

### Input Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | (required) | Input panel with daily aggregated bond data |
| `id_col` | `str` | `"cusip_id"` | Column name for bond identifier |
| `date_col` | `str` | `"trd_exctn_dt"` | Column name for trade execution date |
| `price_col` | `str` | `"pr"` | Column name for primary price (in % of par) |

### Filter Toggle Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_anomaly_filter` | `bool` | `True` | Enable downward anomaly detection |
| `enable_spike_filter` | `bool` | `True` | Enable upward spike detection |
| `enable_plateau_filter` | `bool` | `True` | Enable plateau sequence detection |
| `enable_intraday_filter` | `bool` | `True` | Enable intraday inconsistency detection |

### Anomaly Detection Parameters (Filter 1)

| Parameter | Type | Default | Mathematical Notation | Description |
|-----------|------|---------|----------------------|-------------|
| `ultra_low_threshold` | `float` | `0.10` | $\tau_{\text{low}}$ | Price threshold for ultra-low detection (0.10% of par = $1) |
| `min_normal_price_ratio` | `float` | `3.0` | $\rho_{\text{anomaly}}$ | Minimum ratio of surrounding median to current price |
| `lookback` | `int` | `5` | $L_{\text{back}}$ | Number of days to look back for surrounding prices |
| `lookforward` | `int` | `5` | $L_{\text{fwd}}$ | Number of days to look forward for surrounding prices |

### Spike Detection Parameters (Filter 2)

| Parameter | Type | Default | Mathematical Notation | Description |
|-----------|------|---------|----------------------|-------------|
| `high_spike_threshold` | `float` | `5.0` | $\tau_{\text{high}}$ | Price threshold for high spike detection (5% of par = $50) |
| `min_spike_ratio` | `float` | `3.0` | $\rho_{\text{spike}}$ | Minimum ratio of spike price to pre-spike median |
| `recovery_ratio` | `float` | `2.0` | $\rho_{\text{recovery}}$ | Maximum recovery price as multiple of pre-spike median |

### Plateau Detection Parameters (Filter 3)

| Parameter | Type | Default | Mathematical Notation | Description |
|-----------|------|---------|----------------------|-------------|
| `plateau_ultra_low_threshold` | `float` | `0.15` | $\tau_{\text{plateau}}$ | Price threshold for plateau candidates (0.15% of par = $1.50) |
| `min_plateau_days` | `int` | `2` | $\ell_{\min}$ | Minimum consecutive days to qualify as plateau |
| `pre_post_price_ratio` | `float` | `3.0` | $\rho_{\text{plateau}}$ | Minimum ratio of adjacent prices to plateau price |

### Round Number Parameters (All Filters)

| Parameter | Type | Default | Mathematical Notation | Description |
|-----------|------|---------|----------------------|-------------|
| `suspicious_round_numbers` | `List[float]` | `[0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]` | $\mathcal{R}$ | List of suspicious round price levels (in % of par; see table above) |
| `round_tolerance` | `float` | `0.0001` | $\epsilon_{\text{round}}$ | Numerical tolerance for round number matching |

### Intraday Inconsistency Parameters (Filter 4)

| Parameter | Type | Default | Mathematical Notation | Description |
|-----------|------|---------|----------------------|-------------|
| `price_cols` | `list` | `["prc_ew", "prc_vw", "prc_first", "prc_last"]` | — | Intraday price columns to check for consistency |
| `intraday_range_threshold` | `float` | `0.75` | $\gamma_{\mathrm{range}}$ | Maximum intraday range as fraction of mean price (75%) |
| `intraday_price_threshold` | `float` | `20.0` | $\tau_{\mathrm{intraday}}$ | Only check inconsistency if any price below this (20% of par = $200) |

### Performance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `False` | Print detailed progress information |
| `keep_flag_columns` | `bool` | `False` | If True, retain individual filter flag columns; otherwise drop to save memory |

---

## Algorithm Logic (Step-by-Step)

### Step 1: Preprocessing
1. **Sort** DataFrame by `[id_col, date_col]`
2. **Round** all price columns to 4 decimal places
3. Initialize output arrays:
   - `flag_anomalous_price` (int8)
   - `anomaly_type` (object)
   - `flag_upward_spike` (int8)
   - `spike_type` (object)
   - `flag_plateau_sequence` (int8)
   - `plateau_id` (int32)

### Step 2: Per-CUSIP Processing

For each bond group (`id_col`), iterate through the following filters:

```
INITIALIZE:
  Extract prices array: P = df[price_col].values
  valid_mask = ~isnan(P)

  # Precompute round number mask
  is_round = compute_round_mask(P, round_numbers, round_tolerance, valid_mask)

  # For spike filter: round numbers > 0.50
  is_round_spike = is_round & (P > 0.50)

# FILTER 1: ANOMALY DETECTION
IF enable_anomaly_filter:
    is_ultra_low = (P < ultra_low_threshold)

    FOR i = 0 to n-1:
        IF NOT ((is_ultra_low[i] OR is_round[i]) AND valid_mask[i]):
            CONTINUE

        # Collect surrounding prices
        surr = []
        FOR j in [max(0, i-lookback), i-1]:
            IF valid_mask[j] AND P[j] > P[i]:
                surr.append(P[j])
        FOR j in [i+1, min(n, i+lookforward)]:
            IF valid_mask[j] AND P[j] > P[i]:
                surr.append(P[j])

        IF len(surr) == 0:
            CONTINUE

        median_surr = median(surr)
        price_ratio = median_surr / (P[i] + 1e-10)

        IF price_ratio >= min_normal_price_ratio:
            flag_anomalous_price[i] = 1
            IF is_ultra_low[i] AND is_round[i]:
                anomaly_type[i] = 'ultra_low_round'
            ELIF is_ultra_low[i]:
                anomaly_type[i] = 'ultra_low'
            ELSE:
                anomaly_type[i] = 'round_number'

# FILTER 2: SPIKE DETECTION
IF enable_spike_filter:
    is_high = (P > high_spike_threshold)

    FOR i = 0 to n-1:
        IF NOT ((is_high[i] OR is_round_spike[i]) AND valid_mask[i]):
            CONTINUE

        # Collect pre-spike prices
        pre_prices = []
        FOR j in [max(0, i-lookback), i-1]:
            IF valid_mask[j] AND P[j] < P[i]:
                pre_prices.append(P[j])

        IF len(pre_prices) == 0:
            CONTINUE

        median_pre = median(pre_prices)
        spike_ratio = P[i] / (median_pre + 1e-10)

        IF spike_ratio < min_spike_ratio:
            CONTINUE

        # Check recovery
        recovery_threshold = median_pre * recovery_ratio
        has_recovery = FALSE
        FOR j in [i+1, min(n, i+lookforward)]:
            IF valid_mask[j] AND P[j] <= recovery_threshold:
                has_recovery = TRUE
                BREAK

        IF NOT has_recovery:
            CONTINUE

        flag_upward_spike[i] = 1
        IF is_high[i] AND is_round_spike[i]:
            spike_type[i] = 'high_round_spike'
        ELIF is_high[i]:
            spike_type[i] = 'high_spike'
        ELSE:
            spike_type[i] = 'round_spike'

# FILTER 3: PLATEAU DETECTION
IF enable_plateau_filter:
    is_ultra_low_plateau = (P < plateau_ultra_low_threshold)
    plateau_count = 0
    i = 0

    WHILE i < n:
        IF NOT (valid_mask[i] AND (is_ultra_low_plateau[i] OR is_round[i])):
            i += 1
            CONTINUE

        # Find extent of plateau
        j = i + 1
        WHILE j < n AND P[j] == P[i]:
            j += 1

        plateau_length = j - i

        IF plateau_length >= min_plateau_days:
            # Check suspicion criteria
            is_suspicious = FALSE

            # Pre-plateau check
            IF i > 0 AND valid_mask[i-1]:
                IF P[i-1] / (P[i] + 1e-10) >= pre_post_price_ratio:
                    is_suspicious = TRUE

            # Post-plateau check
            IF j < n AND valid_mask[j]:
                IF P[j] / (P[i] + 1e-10) >= pre_post_price_ratio:
                    is_suspicious = TRUE

            # Round number check
            IF is_round[i]:
                is_suspicious = TRUE

            # Flag entire plateau
            IF is_suspicious:
                FOR k in [i, j-1]:
                    flag_plateau_sequence[k] = 1
                    plateau_id[k] = plateau_count
                plateau_count += 1

        i = j  # Jump to end of plateau

# FILTER 4: INTRADAY INCONSISTENCY
IF enable_intraday_filter:
    FOR i = 0 to n-1:
        # Extract intraday prices
        intraday_prices = [P_col[i] for P_col in price_cols if not isnan(P_col[i])]

        IF len(intraday_prices) < 2:
            CONTINUE

        # Check if any price is low
        IF NOT any(p < intraday_price_threshold for p in intraday_prices):
            CONTINUE

        price_range = max(intraday_prices) - min(intraday_prices)
        price_mean = mean(intraday_prices)

        IF price_mean > 0:
            range_pct = price_range / price_mean
            IF range_pct > intraday_range_threshold:
                flag_intraday_inconsistent[i] = 1
```

### Step 3: Combined Flag

Compute the unified flag (stored in the `flag_refined_any` column):

$$
\mathit{flagRefinedAny}_i = \begin{cases}
1 & \text{if any filter flagged row } i \\
0 & \text{otherwise}
\end{cases}
$$

### Step 4: Output

Return DataFrame with added column:
- `flag_refined_any` (int8): `1` if flagged by any filter, `0` otherwise

Note: Individual filter flags and metadata are dropped to conserve memory.

---

## Examples

### Example 1: Ultra-Low Anomaly Detection

**Input Data** (CUSIP = `XYZ1`):

| Row $i$ | Date | Price $P_i$ | Notes |
|---------|------|-------------|-------|
| 0 | 2024-01-10 | 45.2 | Normal distressed |
| 1 | 2024-01-11 | 44.8 | Normal distressed |
| 2 | 2024-01-12 | **0.05** | **Data entry error** |
| 3 | 2024-01-13 | 45.5 | Reverts to normal |
| 4 | 2024-01-14 | 44.3 | Normal distressed |

**Algorithm Execution (Filter 1)**:

1. **Row 2 (Candidate)**:
   - $P_2 = 0.05 < \tau_{\text{low}} = 0.10$ ✓
   - $|0.05 - 0.05| < 0.0001$ ✓ (is round number)
   - **Candidate opened**

2. **Surrounding Prices**:
   - Lookback: $\{P_0 = 45.2, P_1 = 44.8\}$ (both > 0.05)
   - Lookforward: $\{P_3 = 45.5, P_4 = 44.3\}$ (both > 0.05)
   - $\mathcal{S}_2 = \{45.2, 44.8, 45.5, 44.3\}$
   - $M_{\text{surr}} = \text{median}(\{44.3, 44.8, 45.2, 45.5\}) = \frac{44.8 + 45.2}{2} = 45.0$

3. **Anomaly Check**:
   - $\frac{M_{\text{surr}}}{P_2} = \frac{45.0}{0.05} = 900.0 \geq 3.0$ ✓
   - **Flag as anomaly**
   - Type: `ultra_low_round` (ultra-low AND round)

**Output**:

| Row | Price | `flag_refined_any` | Notes |
|-----|-------|--------------------|-------|
| 2 | 0.05 | **1** | Anomalous (ultra_low_round) |

---

### Example 2: Upward Spike Detection

**Input Data** (CUSIP = `XYZ2`):

| Row | Date | Price | Notes |
|-----|------|-------|-------|
| 0 | 2024-02-10 | 12.5 | Distressed level |
| 1 | 2024-02-11 | 11.8 | Distressed level |
| 2 | 2024-02-12 | **100.0** | **Erroneous par spike** |
| 3 | 2024-02-13 | 12.2 | Recovery |
| 4 | 2024-02-14 | 11.5 | Normal |

**Algorithm Execution (Filter 2)**:

1. **Row 2 (Candidate)**:
   - $|P_2 - 100.0| < 0.0001$ ✓ (round number)
   - $P_2 = 100.0 > 0.50$ ✓ (qualifies for spike round check)
   - **Candidate opened**

2. **Pre-Spike Prices**:
   - $\mathcal{P}_{\text{pre}} = \{P_0 = 12.5, P_1 = 11.8\}$ (both < 100.0)
   - $M_{\text{pre}} = \frac{11.8 + 12.5}{2} = 12.15$

3. **Spike Magnitude**:
   - $\frac{P_2}{M_{\text{pre}}} = \frac{100.0}{12.15} \approx 8.23 \geq 3.0$ ✓

4. **Recovery Check**:
   - $P_{\text{recovery}} = 12.15 \times 2.0 = 24.30$
   - Row 3: $P_3 = 12.2 \leq 24.30$ ✓
   - **Recovery found**

5. **Decision**: **Flag as spike** (type: `round_spike`)

**Output**:

| Row | Price | `flag_refined_any` |
|-----|-------|--------------------|
| 2 | 100.0 | **1** |

---

### Example 3: Plateau Detection

**Input Data** (CUSIP = `XYZ3`):

| Row | Date | Price | Notes |
|-----|------|-------|-------|
| 0 | 2024-03-10 | 55.2 | Normal |
| 1 | 2024-03-11 | 54.8 | Normal |
| 2 | 2024-03-12 | **0.01** | **Error plateau starts** |
| 3 | 2024-03-13 | **0.01** | **Continues** |
| 4 | 2024-03-14 | **0.01** | **Continues** |
| 5 | 2024-03-15 | 55.5 | Reverts |

**Algorithm Execution (Filter 3)**:

1. **Row 2 (Plateau Start)**:
   - $P_2 = 0.01 < \tau_{\text{plateau}} = 0.15$ ✓
   - $|P_2 - 0.01| < 0.0001$ ✓ (round number)

2. **Plateau Extent**:
   - $P_2 = P_3 = P_4 = 0.01$
   - $\ell_{\text{plateau}} = 3 \geq \ell_{\min} = 2$ ✓

3. **Suspicion Criteria**:
   - **Pre-plateau**: $\frac{P_1}{P_2} = \frac{54.8}{0.01} = 5480.0 \geq 3.0$ ✓
   - **Post-plateau**: $\frac{P_5}{P_2} = \frac{55.5}{0.01} = 5550.0 \geq 3.0$ ✓
   - **Round number**: $R_2 = 1$ ✓
   - **Decision**: Suspicious (multiple criteria met)

4. **Flagging**: Flag rows 2, 3, 4 with plateau ID = 0

**Output**:

| Row | Price | `flag_refined_any` | Notes |
|-----|-------|--------------------|-------|
| 2 | 0.01 | **1** | Plateau (ID=0) |
| 3 | 0.01 | **1** | Plateau (ID=0) |
| 4 | 0.01 | **1** | Plateau (ID=0) |

---

### Example 4: Genuine Distressed Trading (No Flag)

**Input Data** (CUSIP = `XYZ4`, genuine default scenario):

| Row | Date | Price (% par) | Price ($) | Notes |
|-----|------|---------------|-----------|-------|
| 0 | 2024-04-10 | 65.0 | $650 | Pre-default |
| 1 | 2024-04-11 | 8.5 | $85 | Default announced |
| 2 | 2024-04-12 | 7.8 | $78 | Continued distress |
| 3 | 2024-04-13 | 8.2 | $82 | Slight recovery |
| 4 | 2024-04-14 | 7.5 | $75 | Distressed level |

**Algorithm Execution**:

**Filter 1 (Anomaly)**: No candidates
- None of the prices {65.0, 8.5, 7.8, 8.2, 7.5} are < τ_low = 0.10
- None are round numbers from R = {0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00}
- **No candidates opened** → Filter 1 passes all rows

**Filter 2 (Spike)**: No candidates
- No price > τ_high = 5.0 AND no round numbers > 0.50
- (65.0 is not a round number)
- **No spike candidates**

**Filter 3 (Plateau)**: No candidates
- No price < τ_plateau = 0.15 AND no round numbers
- **No plateau candidates**

**Filter 4 (Intraday)**: Not applicable (would need intraday data)

**Output**: All rows have `flag_refined_any = 0` (genuine distressed trading preserved)

**Key Insight**: The filter correctly ignores genuine distressed bonds because:
1. Even $75-85 prices are well above the ultra-low thresholds ($1-$1.50)
2. Prices aren't suspicious round numbers
3. Prices aren't identical across days (no plateau)

---

### Example 5: Intraday Inconsistency

**Input Data** (CUSIP = `XYZ5`, single day with `prc_hi` and `prc_lo` columns):

| Row | Date | `prc_hi` (% par) | `prc_lo` (% par) | `prc_hi` ($) | `prc_lo` ($) | Notes |
|-----|------|------------------|------------------|--------------|--------------|-------|
| 0 | 2024-05-10 | 89.0 | **0.10** | $890 | **$1** | Erroneous low print |

**Algorithm Execution (Filter 4)**:

1. **Intraday Prices** (from available columns):
   - $\mathcal{I}_0 = \{89.0, 0.10\}$ (all non-NaN values from price_cols)

2. **Low Price Check**:
   - $\min(\mathcal{I}_0) = 0.10 < \tau_{\mathrm{intraday}} = 20.0$ ✓

3. **Range Analysis**:
   - $R_{\mathrm{intraday}} = 89.0 - 0.10 = 88.9$
   - $\bar{P}_{\mathrm{intraday}} = \frac{89.0 + 0.10}{2} = 44.55$
   - $\frac{R_{\mathrm{intraday}}}{\bar{P}_{\mathrm{intraday}}} = \frac{88.9}{44.55} \approx 2.0 > 0.75$ ✓

4. **Decision**: **Flag as intraday inconsistent**

**Output**:

| Row | `flag_refined_any` | Notes |
|-----|--------------------|-------|
| 0 | **1** | Intraday inconsistency |

**Note**: This example shows an erroneous low print of $1 on a day where the high was $890—an implausible 89× intraday swing indicating data error.

---

## Default Configuration

From `stage1/_stage1_settings.py`:

```python
ULTRA_DISTRESSED_CONFIG = {
    'price_col': 'pr',

    # Intraday inconsistency (Filter 4)
    'intraday_range_threshold': 0.75,  # % Within Day Move for Flag
    'intraday_price_threshold': 20,    # Kicks in below this % of par ($200)

    # Anomaly detection (Filter 1)
    'ultra_low_threshold': 0.10,       # 0.10% of par = $1
    'min_normal_price_ratio': 3.0,

    # Plateau detection (Filter 3)
    'plateau_ultra_low_threshold': 0.15,  # 0.15% of par = $1.50
    'min_plateau_days': 2,

    # Round numbers (all filters)
    'suspicious_round_numbers': [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00],

    # Spike detection (Filter 2)
    'price_cols': ['prc_hi', 'prc_lo'],
    'high_spike_threshold': 5.0,       # 5% of par = $50
    'min_spike_ratio': 3.0,
    'recovery_ratio': 2.0,

    'verbose': True,
}
```

**Note**: Parameters not in this config (e.g., `lookback`, `lookforward`, `pre_post_price_ratio`) use function defaults.

---

## Typical Usage in Pipeline

```python
from stage1.helper_functions import ultra_distressed_filter
from stage1._stage1_settings import ULTRA_DISTRESSED_CONFIG

# Load daily aggregated TRACE data
df_daily = pd.read_parquet("stage1_daily_aggregated.parquet")

# Apply ultra-distressed filter
df_flagged = ultra_distressed_filter(
    df_daily,
    id_col="cusip_id",
    date_col="trd_exctn_dt",
    price_col="prc_vw",
    **ULTRA_DISTRESSED_CONFIG
)

# Remove flagged rows
df_clean = df_flagged[df_flagged["flag_refined_any"] == 0].copy()

n_flagged = df_flagged["flag_refined_any"].sum()
n_total = len(df_flagged)
affected_cusips = df_flagged.loc[df_flagged["flag_refined_any"] == 1, "cusip_id"].nunique()

print(f"Flagged {n_flagged:,} / {n_total:,} observations ({100*n_flagged/n_total:.2f}%)")
print(f"Affected CUSIPs: {affected_cusips:,}")
# Output: Flagged 12,847 / 3,245,892 observations (0.40%)
# Output: Affected CUSIPs: 2,153
```

---

## Design Rationale

### Why Four Independent Filters?

Different error types require specialized detection logic:

1. **Anomaly Filter**: Detects isolated downward outliers (temporary data entry errors)
2. **Spike Filter**: Detects temporary upward outliers (erroneous par or high prints)
3. **Plateau Filter**: Detects persistent placeholder values (multi-day data gaps)
4. **Intraday Filter**: Detects within-day inconsistencies (aggregation errors)

Each filter can be disabled independently for sensitivity analysis.

### Why Round Number Heuristic?

Corporate bond prices rarely settle exactly at round levels like $0.01$ or $1.00$ due to:
- Bid-ask spreads
- Accrued interest calculations
- Market microstructure

Round numbers in distressed bonds are almost always:
- Placeholder values (e.g., $0.01$ as "effectively zero")
- Data entry errors (e.g., typing $1.00$ instead of actual price)
- Dealers "cleaning up" their books (per Colin Philipps, FINRA)

### Why Ratio-Based Thresholds?

Using **multiplicative ratios** (e.g., $3\times$) instead of absolute differences provides:
- **Scale invariance**: Works for both high-yield ($50-80$) and deeply distressed ($5-20$) bonds
- **Robustness**: Adapts to local price level without manual tuning

### Why Lookback/Lookforward Windows?

**Symmetric windows** ($L_{\text{back}} = L_{\text{fwd}} = 5$) provide:
- **Temporal context**: Compare price to both past and future
- **Genuine event detection**: Permanent changes (defaults) won't revert in lookforward
- **Reduced false positives**: Temporary errors isolated from surrounding normal prices

### Why Separate Intraday Filter?

Intraday inconsistencies can occur even when daily aggregates appear normal:
- Erroneous single trade at $0.10$ among normal trades at $85-90$
- Daily volume-weighted average may mask the error
- Filter 4 catches these within-day anomalies

---

## Performance Considerations

### Numba Compilation

Core detection functions (`_detect_anomalies_ultra`, `_detect_spikes_ultra`, `_detect_plateaus_ultra`) are compiled with Numba using:
- `nopython=True`: Pure NumPy operations (no Python overhead)
- `cache=True`: Compiled functions cached across runs
- `fastmath=True`: Aggressive floating-point optimizations
- `nogil=True`: Release Python GIL for potential parallelism

**Typical Performance**: Processes ~500,000 bond-days in ~15 seconds (single-threaded).

### Memory Optimization

At function exit, individual flag columns are **dropped** to conserve RAM:
- Only `flag_refined_any` is retained (1 byte per row)
- Metadata columns (`anomaly_type`, `spike_type`, `plateau_id`) discarded
- Reduces memory footprint by ~90% for large datasets

---

## Comparison with Other Filters

| Aspect | Bounce-Back Filter | Ultra-Distressed Filter |
|--------|-------------------|------------------------|
| **Data Level** | Intraday transactions | Daily aggregates |
| **Error Type** | Transient price spikes (both up/down) | Persistent anomalies, plateaus, inconsistencies |
| **Detection Method** | Reversion pattern (lookahead for bounce-back) | Ratio-based outlier detection + plateau sequences |
| **Action** | **Flag** transactions | **Flag** daily observations |
| **Sequence** | Applied in Stage 0 (intraday) | Applied in Stage 1 (daily) |
| **Primary Use Case** | Real-time transaction filtering | Post-aggregation quality control |

**Complementary Relationship**:
- Bounce-back filter removes intraday errors **before** aggregation
- Ultra-distressed filter catches errors that survive aggregation or appear **only** in daily data

---

## References

```bibtex
@unpublished{dickerson2025pitfalls,
  author = {Dickerson, Alexander and Robotti, Cesare and Rossetti, Giulio},
  title = {Common pitfalls in the evaluation of corporate bond strategies},
  year = {2025},
  note = {Working Paper}
}

@unpublished{dickerson2025constructing,
  author = {Dickerson, Alexander and Rossetti, Giulio},
  title = {Constructing TRACE Corporate Bond Datasets},
  year = {2025},
  note = {Working Paper}
}
```

---

## See Also

- `README_bounce_back_filter.md` — Documentation for the intraday bounce-back filter (Stage 0)
- `_stage1_settings.py` — Default configuration parameters
- `helper_functions.py:1003-1219` — Source code for `ultra_distressed_filter()`
- `step_05_apply_ultra_distressed_filter.py` — Pipeline implementation
