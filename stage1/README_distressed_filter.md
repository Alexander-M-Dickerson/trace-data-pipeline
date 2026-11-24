# Ultra-distressed filter: technical documentation

## Overview

The ultra-distressed filter detects anomalous prices in distressed corporate bond transaction data. The anomalies are ultra-low prices, upward spikes, persistent plateaus and intraday inconsistencies that come from data errors and do not reflect market conditions.

The algorithm runs four independent filters on daily aggregated prices. They use lookback and lookforward windows, round-number heuristics and ratio-based thresholds to find candidate errors.

The filter is most useful when Stage 0 applies little or no volume filtering, or filters only on par volume. Those configurations keep micro trades at obscure round numbers, which may be placeholder values or data entry errors, and this filter cleans up those cases.
These bond prices often occur on debt issues trading in default. Thank you to Colin Philipps (FINRA) for some clarification on these extremely low/rounded prices in TRACE, "_... not the TRACE system adding anything to the tape but are transactions reported [by] broker dealers. Likely they are cleaning up positions/books. TRACE only transmits what is submitted to the system._".

## Framework

### Problem statement

Given a time-ordered sequence of daily aggregated prices $\{P_1, P_2, \ldots, P_n\}$ for a bond, the task is to find the observations $P_i$ that are candidate errors. A candidate error is one or more of the following.

1. An ultra-low anomaly, where price $P_i$ is much lower than the surrounding prices (a downward outlier)
2. An upward spike, where price $P_i$ is much higher than the preceding prices and quickly recovers
3. A persistent plateau, a sequence of identical ultra-low or round prices lasting multiple days
4. An intraday inconsistency, an implausibly wide intraday high/low range (`prc_hi` vs `prc_lo`)

The algorithm must tell these errors apart from genuine distressed pricing (e.g., default events, bankruptcy), which persists over time.

### Core algorithm components

#### 1. Round number detection

For all filters, compute a round number mask that identifies prices at suspicious round levels.

$$
R_i = \begin{cases}
1 & \text{if } \exists r \in \mathcal{R} : |P_i - r| < \epsilon_{\text{round}} \\
0 & \text{otherwise}
\end{cases}
$$

- $R_i$ is a binary indicator (1 if price $P_i$ matches a suspicious round number, 0 otherwise)
- $r$ is an individual round number from the set $\mathcal{R}$
- $\mathcal{R} = \{0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00\}$ is the set of suspicious round numbers (in % of par).
  **0.75 is deliberately not in this set**, because it is a plausible distressed price and not a placeholder.
- $\epsilon_{\text{round}} = 0.0001$ is the tolerance for round detection

All prices in this filter are expressed in % of par. With par = $1,000, the round numbers have these dollar values.

| Price (% of par) | Dollar value |
|------------------|--------------|
| 0.001 | $0.01 |
| 0.01 | $0.10 |
| 0.05 | $0.50 |
| 0.10 | $1.00 |
| 0.25 | $2.50 |
| 0.50 | $5.00 |
| 1.00 | $10.00 |

We are in contact with FINRA/TRACE to understand these odd rounded numbers, which often occur in recognizable sequences. According to Colin Philipps (FINRA), it could be dealers "cleaning-up" the books.

## Filter components

### Filter 1: anomaly detection (downward outliers)

This filter detects isolated ultra-low prices that are outliers compared to the surrounding normal prices.

#### Opening conditions

A row $i$ becomes a candidate for anomaly flagging if

$$
P_i < \tau_{\text{low}} \quad \text{OR} \quad R_i = 1
$$

- $P_i$ is the price at row $i$ (in % of par)
- $R_i$ is the round number indicator (1 if $P_i$ matches a suspicious round number, 0 otherwise, see Section 1)
- $\tau_{\text{low}} = 0.10$ is the ultra-low threshold (0.10% of par = $1)

In words, a price is a candidate if it is below $1 (extremely low) or matches a suspicious round number.

#### Surrounding price analysis

For each candidate at row $i$, collect the surrounding prices from the lookback and lookforward windows.

$$
\mathcal{S}_i = \{P_j : j \in [i-L_{\text{back}}, i-1] \cup [i+1, i+L_{\text{fwd}}], \, P_j > P_i, \, P_j \text{ valid}\}
$$

- $L_{\text{back}} = 5$ is the lookback window, in observations (days on which the bond traded)
- $L_{\text{fwd}} = 5$ is the lookforward window, in observations
- Only prices strictly greater than $P_i$ are collected

Then compute the median of the surrounding prices.

$$
M_{\text{surr}} = \text{median}(\mathcal{S}_i)
$$

#### Anomaly criterion

Flag row $i$ as anomalous if

$$
\frac{M_{\text{surr}}}{P_i + \epsilon} \geq \rho_{\text{anomaly}}
$$

- $\rho_{\text{anomaly}} = 3.0$ is the minimum ratio for anomaly detection
- $\epsilon = 10^{-10}$ is the numerical stability constant

That is, the median of the surrounding (higher) prices is at least 3 times the current price.

#### Numerical example

| Day | Price (% par) | Price ($) | Candidate? | Surrounding (> $P_i$) | Median | Ratio | Flagged? |
|-----|---------------|-----------|------------|----------------------|--------|-------|----------|
| 1 | 45.0 | $450 | No | | | | No |
| 2 | 44.5 | $445 | No | | | | No |
| 3 | 0.05 | $0.50 | Yes (< 0.10 and round) | {45.0, 44.5, 45.2, 44.8} | 44.9 | 898× | Yes |
| 4 | 45.2 | $452 | No | | | | No |
| 5 | 44.8 | $448 | No | | | | No |

Day 3 is flagged because (1) price 0.05 < τ_low = 0.10, (2) 0.05 is a round number, and (3) surrounding median / current price = 44.9 / 0.05 = 898 ≥ 3.0.

#### Anomaly classification

The flag type depends on which conditions hold.
- `ultra_low` ($t = 1$) when $P_i < \tau_{\text{low}}$ and not $R_i$
- `round_number` ($t = 2$) when $R_i$ and not $(P_i < \tau_{\text{low}})$
- `ultra_low_round` ($t = 3$) when $P_i < \tau_{\text{low}}$ and $R_i$

### Filter 2: spike detection (upward outliers)

This filter detects temporary upward price spikes that quickly revert to lower levels.

#### Opening conditions

A row $i$ becomes a candidate for spike flagging if

$$
P_i > \tau_{\text{high}} \quad \text{OR} \quad (R_i = 1 \text{ AND } P_i > 0.50)
$$

- $P_i$ is the price at row $i$ (in % of par)
- $R_i$ is the round number indicator (1 if $P_i$ matches a suspicious round number, 0 otherwise, see Section 1)
- $\tau_{\text{high}} = 5.0$ is the high spike threshold (5% of par = $50)
- The round number condition requires $P_i > 0.50$ (0.50% of par = $5), so that tiny round numbers are not flagged

So a price is a candidate if it exceeds $50 or is a suspicious round number above $5.

#### Pre-spike price analysis

For each candidate at row $i$, collect the preceding prices from the lookback window.

$$
\mathcal{P}_{\text{pre}} = \{P_j : j \in [i-L_{\text{back}}, i-1], \, P_j < P_i, \, P_j \text{ valid}\}
$$

Only prices strictly less than $P_i$ are collected.

Then compute the median of the pre-spike prices.

$$
M_{\text{pre}} = \text{median}(\mathcal{P}_{\text{pre}})
$$

#### Spike magnitude check

Check that the spike is large enough.

$$
\frac{P_i}{M_{\text{pre}} + \epsilon} \geq \rho_{\text{spike}}
$$

Here $\rho_{\text{spike}} = 3.0$ is the minimum spike ratio.

#### Recovery check

Define the recovery threshold as

$$
P_{\text{recovery}} = M_{\text{pre}} \cdot \rho_{\text{recovery}}
$$

where $\rho_{\text{recovery}} = 2.0$ is the recovery ratio.

Then search forward for evidence of recovery.

$$
\exists j \in [i+1, i+L_{\text{fwd}}] : P_j \leq P_{\text{recovery}} \text{ AND } P_j \text{ valid}
$$

Flag row $i$ as a spike if both the magnitude check and the recovery check pass.

#### Numerical example

| Day | Price (% par) | Price ($) | Candidate? | Pre-prices (< $P_i$) | Median | Spike ratio | Recovery? | Flagged? |
|-----|---------------|-----------|------------|---------------------|--------|-------------|-----------|----------|
| 1 | 12.5 | $125 | Yes (> 5.0) | none | | | | No |
| 2 | 11.8 | $118 | Yes (> 5.0) | none | | | | No |
| 3 | 100.0 | $1,000 | Yes (> 5.0) | {12.5, 11.8} | 12.15 | 8.23× | Yes, day 4 (12.2 ≤ 24.3) | Yes |
| 4 | 12.2 | $122 | Yes (> 5.0) | {11.8} | 11.8 | 1.03× | | No |
| 5 | 11.5 | $115 | Yes (> 5.0) | none | | | | No |

Day 3 is flagged for these reasons.
1. It is a candidate, because 100.0 > τ_high = 5.0
2. The pre-spike median is (11.8 + 12.5) / 2 = 12.15
3. The spike ratio is 100.0 / 12.15 = 8.23 ≥ 3.0
4. The recovery threshold is 12.15 × 2.0 = 24.3, and the Day 4 price is 12.2 ≤ 24.3

Days 1, 2, 4 and 5 are candidates too, because each is above 5.0, and none is flagged. Days 1, 2
and 5 have no earlier price below their own, so there is nothing to compare with. Day 4's spike
ratio is 12.2 / 11.8 = 1.03, below 3.0.

#### Spike classification

The flag type depends on which conditions hold.
- `high_spike` ($s = 1$) when $P_i > \tau_{\text{high}}$ and not $R_i$
- `round_spike` ($s = 2$) when $R_i$ and not $(P_i > \tau_{\text{high}})$
- `high_round_spike` ($s = 3$) when $P_i > \tau_{\text{high}}$ and $R_i$

### Filter 3: plateau detection (persistent identical prices)

This filter detects sequences of consecutive days with identical ultra-low or round prices.

#### Opening conditions

A row $i$ becomes the start of a potential plateau if

$$
P_i < \tau_{\text{plateau}} \quad \text{OR} \quad R_i = 1
$$

- $P_i$ is the price at row $i$ (in % of par)
- $R_i$ is the round number indicator (1 if $P_i$ matches a suspicious round number, 0 otherwise, see Section 1)
- $\tau_{\text{plateau}} = 0.15$ is the plateau threshold (0.15% of par = $1.50)

Only ultra-low prices (< $1.50) or round numbers can start a plateau.

#### Plateau identification

Starting from row $i$, find how far the exact price equality extends.

$$
\ell_{\text{plateau}} = \max\{k : P_{i+j} = P_i \text{ for all } j \in [0, k-1]\}
$$

Here $\ell_{\text{plateau}}$ is the length of the run of consecutive identical prices (exact equality is required).

#### Minimum run length

A plateau is considered for flagging only if

$$
\ell_{\text{plateau}} \geq \ell_{\min}
$$

where $\ell_{\min} = 2$ (the default minimum plateau length, in observations).

#### Plateau suspicion criteria

A plateau is flagged as suspicious if any of the following criteria holds.

Criterion 1 is pre-plateau displacement. If $i > 0$ and $P_{i-1}$ is valid, it holds when

$$
\frac{P_{i-1}}{P_i + \epsilon} \geq \rho_{\text{plateau}}
$$

Criterion 2 is post-plateau displacement. Let $j = i + \ell_{\text{plateau}}$ (the first row after the plateau). If $j < n$ and $P_j$ is valid, it holds when

$$
\frac{P_j}{P_i + \epsilon} \geq \rho_{\text{plateau}}
$$

Criterion 3 is a round number plateau.

$$
R_i = 1
$$

Here $\rho_{\text{plateau}} = 3.0$ is the pre/post price ratio threshold.

If any criterion holds, flag all rows in $[i, i+\ell_{\text{plateau}}-1]$ and assign a plateau ID.

#### Numerical example

| Day | Price (% par) | Price ($) | Candidate? | Plateau? | Pre-ratio | Post-ratio | Round? | Flagged? |
|-----|---------------|-----------|------------|----------|-----------|------------|--------|----------|
| 1 | 55.0 | $550 | No | | | | | No |
| 2 | 54.8 | $548 | No | | | | | No |
| 3 | 0.01 | $0.10 | Yes (< 0.15 and round) | Start | 5,480× | | Yes | Yes |
| 4 | 0.01 | $0.10 | | Continue | | | | Yes |
| 5 | 0.01 | $0.10 | | Continue | | 5,550× | | Yes |
| 6 | 55.5 | $555 | No | | | | | No |

Days 3 to 5 are flagged for these reasons.
1. The price 0.01 is below τ_plateau = 0.15 and is a round number
2. The plateau length is 3 ≥ ℓ_min = 2
3. All three suspicion criteria are met
   - The pre-plateau ratio is 54.8 / 0.01 = 5,480 ≥ 3.0
   - The post-plateau ratio is 55.5 / 0.01 = 5,550 ≥ 3.0
   - The price is a round number, 0.01 ∈ R

### Filter 4: intraday inconsistency detection

This filter detects days where the intraday high/low range (`prc_hi`, `prc_lo`) is implausibly wide relative to their mean. It is a range test on two columns and does not compare price estimators.

#### Price range analysis

For each row $i$, collect the available intraday prices.

$$
\mathcal{I}_i = \{P_i^{\text{high}}, P_i^{\text{low}}\} \quad (\texttt{prc\_hi}, \texttt{prc\_lo})
$$

Then compute the intraday range and mean.

$$
R_{\mathrm{intraday}} = \max(\mathcal{I}_i) - \min(\mathcal{I}_i)
$$

$$
\bar{P}_{\mathrm{intraday}} = \frac{1}{|\mathcal{I}_i|} \sum_{p \in \mathcal{I}_i} p
$$

#### Inconsistency criterion

Flag row $i$ as inconsistent if both conditions hold.

$$
\exists p \in \mathcal{I}_i : p < \tau_{\mathrm{intraday}}
$$

$$
\frac{R_{\mathrm{intraday}}}{\bar{P}_{\mathrm{intraday}}} > \gamma_{\mathrm{range}}
$$

- $\tau_{\mathrm{intraday}} = 20.0$ is the price threshold (20% of par = $200). Only days with at least one low price are checked
- $\gamma_{\mathrm{range}} = 0.75$ is the maximum allowed range as a fraction of the mean (75%)

That is, at least one intraday price is below $200 and the intraday range exceeds 75% of the mean price.

#### Numerical example

| Row | `prc_hi` (% par) | `prc_lo` (% par) | Min | Max | Mean | Range | Range/Mean | Flagged? |
|-----|------------------|------------------|-----|-----|------|-------|------------|----------|
| A | 88.5 | 87.2 | 87.2 | 88.5 | 87.85 | 1.3 | 1.5% | No (no price < 20) |
| B | 89.0 | 0.10 | 0.10 | 89.0 | 44.55 | 88.9 | 199% | Yes |
| C | 15.0 | 14.5 | 14.5 | 15.0 | 14.75 | 0.5 | 3.4% | No (range/mean ≤ 0.75) |

Row B is flagged for these reasons.
1. At least one price is below 20, because prc_lo = 0.10 < 20
2. Range/Mean = 88.9 / 44.55 = 1.99 > 0.75

Row B shows an erroneous low print ($1 vs $890) on an otherwise normally priced day.

## Function signature

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
    suspicious_round_numbers: List[float] = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00],
    round_tolerance: float = 0.0001,
    lookback: int = 5,
    lookforward: int = 5,
    pre_post_price_ratio: float = 3.0,
    enable_intraday_filter: bool = True,
    price_cols: list = ["prc_hi", "prc_lo"],
    intraday_range_threshold: float = 0.75,
    intraday_price_threshold: float = 20.0,
    verbose: bool = False,
    keep_flag_columns: bool = False,
    n_jobs: int = -1,              # accepted, not used
) -> pd.DataFrame
```

## Parameters

### Input data parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | (required) | Input panel with daily aggregated bond data |
| `id_col` | `str` | `"cusip_id"` | Column name for bond identifier |
| `date_col` | `str` | `"trd_exctn_dt"` | Column name for trade execution date |
| `price_col` | `str` | `"pr"` | Column name for primary price (in % of par) |

### Filter toggle parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_anomaly_filter` | `bool` | `True` | Enable downward anomaly detection |
| `enable_spike_filter` | `bool` | `True` | Enable upward spike detection |
| `enable_plateau_filter` | `bool` | `True` | Enable plateau sequence detection |
| `enable_intraday_filter` | `bool` | `True` | Enable intraday inconsistency detection |

### Anomaly detection parameters (Filter 1)

| Parameter | Type | Default | Mathematical notation | Description |
|-----------|------|---------|----------------------|-------------|
| `ultra_low_threshold` | `float` | `0.10` | $\tau_{\text{low}}$ | Price threshold for ultra-low detection (0.10% of par = $1) |
| `min_normal_price_ratio` | `float` | `3.0` | $\rho_{\text{anomaly}}$ | Minimum ratio of surrounding median to current price |
| `lookback` | `int` | `5` | $L_{\text{back}}$ | Number of the bond's earlier observations to look back over |
| `lookforward` | `int` | `5` | $L_{\text{fwd}}$ | Number of the bond's later observations to look forward over |

### Spike detection parameters (Filter 2)

| Parameter | Type | Default | Mathematical notation | Description |
|-----------|------|---------|----------------------|-------------|
| `high_spike_threshold` | `float` | `5.0` | $\tau_{\text{high}}$ | Price threshold for high spike detection (5% of par = $50) |
| `min_spike_ratio` | `float` | `3.0` | $\rho_{\text{spike}}$ | Minimum ratio of spike price to pre-spike median |
| `recovery_ratio` | `float` | `2.0` | $\rho_{\text{recovery}}$ | Maximum recovery price as multiple of pre-spike median |

### Plateau detection parameters (Filter 3)

| Parameter | Type | Default | Mathematical notation | Description |
|-----------|------|---------|----------------------|-------------|
| `plateau_ultra_low_threshold` | `float` | `0.15` | $\tau_{\text{plateau}}$ | Price threshold for plateau candidates (0.15% of par = $1.50) |
| `min_plateau_days` | `int` | `2` | $\ell_{\min}$ | Minimum consecutive observations (days the bond traded) to qualify as plateau |
| `pre_post_price_ratio` | `float` | `3.0` | $\rho_{\text{plateau}}$ | Minimum ratio of adjacent prices to plateau price |

### Round number parameters (all filters)

| Parameter | Type | Default | Mathematical notation | Description |
|-----------|------|---------|----------------------|-------------|
| `suspicious_round_numbers` | `List[float]` | `[0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00]` | $\mathcal{R}$ | List of suspicious round price levels (in % of par, see the table above) |
| `round_tolerance` | `float` | `0.0001` | $\epsilon_{\text{round}}$ | Numerical tolerance for round number matching |

### Intraday inconsistency parameters (Filter 4)

| Parameter | Type | Default | Mathematical notation | Description |
|-----------|------|---------|----------------------|-------------|
| `price_cols` | `list` | `["prc_hi", "prc_lo"]` | | The two columns the range test reads. It is a high/low range test and does not compare estimators. `prc_vw` is not a Stage 1 column. |
| `intraday_range_threshold` | `float` | `0.75` | $\gamma_{\mathrm{range}}$ | Maximum intraday range as fraction of mean price (75%) |
| `intraday_price_threshold` | `float` | `20.0` | $\tau_{\mathrm{intraday}}$ | Only check inconsistency if any price below this (20% of par = $200) |

### Performance parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `False` | Print detailed progress information |
| `keep_flag_columns` | `bool` | `False` | If True, keep the individual filter flag columns. Otherwise drop them to save memory |

## Algorithm logic (step-by-step)

### Step 1: preprocessing
1. Sort the DataFrame by `[id_col, date_col]`
2. Round all price columns to 4 decimal places
3. Initialize the output arrays
   - `flag_anomalous_price` (int8)
   - `anomaly_type` (object)
   - `flag_upward_spike` (int8)
   - `spike_type` (object)
   - `flag_plateau_sequence` (int8)
   - `plateau_id` (int32)

   A seventh column, `flag_intraday_inconsistent` (int8), is added after the per-CUSIP pass by
   Filter 4. It is all 0 when that filter is off.

### Step 2: per-CUSIP processing

For each bond group (`id_col`), run the following filters.

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

### Step 3: combined flag

Compute the unified flag (stored in the `flag_refined_any` column).

$$
\mathit{flagRefinedAny}_i = \begin{cases}
1 & \text{if any filter flagged row } i \\
0 & \text{otherwise}
\end{cases}
$$

### Step 4: output

Return the DataFrame with the added column `flag_refined_any` (int8), which is `1` if the row was flagged by any filter and `0` otherwise.

The individual filter flags and metadata are dropped to conserve memory.

## Examples

### Example 1: ultra-low anomaly detection

The input data is for CUSIP `XYZ1`.

| Row $i$ | Date | Price $P_i$ | Notes |
|---------|------|-------------|-------|
| 0 | 2024-01-10 | 45.2 | Normal distressed |
| 1 | 2024-01-11 | 44.8 | Normal distressed |
| 2 | 2024-01-12 | 0.05 | Data entry error |
| 3 | 2024-01-13 | 45.5 | Reverts to normal |
| 4 | 2024-01-14 | 44.3 | Normal distressed |

Filter 1 runs in these steps.

1. Row 2 is a candidate.
   - $P_2 = 0.05 < \tau_{\text{low}} = 0.10$, so the price is ultra-low
   - $|0.05 - 0.05| < 0.0001$, so it is a round number
   - The candidate is opened

2. The surrounding prices are collected.
   - The lookback gives $\{P_0 = 45.2, P_1 = 44.8\}$ (both > 0.05)
   - The lookforward gives $\{P_3 = 45.5, P_4 = 44.3\}$ (both > 0.05)
   - $\mathcal{S}_2 = \{45.2, 44.8, 45.5, 44.3\}$
   - $M_{\text{surr}} = \text{median}(\{44.3, 44.8, 45.2, 45.5\}) = \frac{44.8 + 45.2}{2} = 45.0$

3. The anomaly check passes.
   - $\frac{M_{\text{surr}}}{P_2} = \frac{45.0}{0.05} = 900.0 \geq 3.0$
   - The row is flagged as an anomaly
   - The type is `ultra_low_round` (ultra-low and round)

The output is as follows.

| Row | Price | `flag_refined_any` | Notes |
|-----|-------|--------------------|-------|
| 2 | 0.05 | 1 | Anomalous (ultra_low_round) |

### Example 2: upward spike detection

The input data is for CUSIP `XYZ2`.

| Row | Date | Price | Notes |
|-----|------|-------|-------|
| 0 | 2024-02-10 | 12.5 | Distressed level |
| 1 | 2024-02-11 | 11.8 | Distressed level |
| 2 | 2024-02-12 | 100.0 | Erroneous par spike |
| 3 | 2024-02-13 | 12.2 | Recovery |
| 4 | 2024-02-14 | 11.5 | Normal |

Filter 2 runs in these steps.

1. Row 2 is a candidate.
   - $P_2 = 100.0 > \tau_{\text{high}} = 5.0$ (a high price)
   - 100.0 is not in the round-number set $\mathcal{R}$, so the round test plays no part here
   - The candidate is opened

2. The pre-spike prices are collected.
   - $\mathcal{P}_{\text{pre}} = \{P_0 = 12.5, P_1 = 11.8\}$ (both < 100.0)
   - $M_{\text{pre}} = \frac{11.8 + 12.5}{2} = 12.15$

3. The spike is large enough.
   - $\frac{P_2}{M_{\text{pre}}} = \frac{100.0}{12.15} \approx 8.23 \geq 3.0$

4. The recovery check passes.
   - $P_{\text{recovery}} = 12.15 \times 2.0 = 24.30$
   - At row 3, $P_3 = 12.2 \leq 24.30$
   - A recovery is found

5. The row is flagged as a spike (type `high_spike`).

Running `ultra_distressed_filter` on this input with the default configuration confirms it.
Row 2 is the only row flagged, with `spike_type = high_spike`.

The output is as follows.

| Row | Price | `flag_refined_any` |
|-----|-------|--------------------|
| 2 | 100.0 | 1 |

### Example 3: plateau detection

The input data is for CUSIP `XYZ3`.

| Row | Date | Price | Notes |
|-----|------|-------|-------|
| 0 | 2024-03-10 | 55.2 | Normal |
| 1 | 2024-03-11 | 54.8 | Normal |
| 2 | 2024-03-12 | 0.01 | Error plateau starts |
| 3 | 2024-03-13 | 0.01 | Continues |
| 4 | 2024-03-14 | 0.01 | Continues |
| 5 | 2024-03-15 | 55.5 | Reverts |

Filter 3 runs in these steps.

1. Row 2 is the start of a potential plateau.
   - $P_2 = 0.01 < \tau_{\text{plateau}} = 0.15$
   - $|P_2 - 0.01| < 0.0001$ (a round number)

2. The plateau extent is measured.
   - $P_2 = P_3 = P_4 = 0.01$
   - $\ell_{\text{plateau}} = 3 \geq \ell_{\min} = 2$

3. The suspicion criteria are checked.
   - Pre-plateau, $\frac{P_1}{P_2} = \frac{54.8}{0.01} = 5480.0 \geq 3.0$
   - Post-plateau, $\frac{P_5}{P_2} = \frac{55.5}{0.01} = 5550.0 \geq 3.0$
   - Round number, $R_2 = 1$
   - The plateau is suspicious (multiple criteria met)

4. Rows 2, 3, 4 are flagged with plateau ID = 0.

The output is as follows.

| Row | Price | `flag_refined_any` | Notes |
|-----|-------|--------------------|-------|
| 2 | 0.01 | 1 | Plateau (ID=0), also anomalous (`ultra_low_round`) |
| 3 | 0.01 | 1 | Plateau (ID=0), also anomalous (`ultra_low_round`) |
| 4 | 0.01 | 1 | Plateau (ID=0), also anomalous (`ultra_low_round`) |

Run on this input, the anomaly filter (Filter 1) flags the same three rows on its own, because
each 0.01 is ultra-low, round, and far below the prices around it. Either filter is enough to
remove them.

### Example 4: genuine distressed trading (no flag)

The input data is for CUSIP `XYZ4`, a genuine default scenario.

| Row | Date | Price (% par) | Price ($) | Notes |
|-----|------|---------------|-----------|-------|
| 0 | 2024-04-10 | 65.0 | $650 | Pre-default |
| 1 | 2024-04-11 | 8.5 | $85 | Default announced |
| 2 | 2024-04-12 | 7.8 | $78 | Continued distress |
| 3 | 2024-04-13 | 8.2 | $82 | Slight recovery |
| 4 | 2024-04-14 | 7.5 | $75 | Distressed level |

The filters give these results.

Filter 1 (Anomaly) has no candidates.
- None of the prices {65.0, 8.5, 7.8, 8.2, 7.5} are < τ_low = 0.10
- None are round numbers from R = {0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.00}
- No candidates are opened, so Filter 1 passes all rows

Filter 2 (Spike) has candidates but flags none.
- Every price is above τ_high = 5.0, so every row is a spike candidate
- Row 0 (65.0) has no earlier prices to compare against
- Rows 1, 2 and 4 have no earlier price below their own, so no pre-spike median exists
- Row 3 (8.2) has one lower earlier price, 7.8, and 8.2 / 7.8 ≈ 1.05 is below the 3.0 ratio
- No spike is flagged

Filter 3 (Plateau) has no candidates.
- No price is < τ_plateau = 0.15 and none is a round number
- There are no plateau candidates

Filter 4 (Intraday) does not apply, because it would need intraday data.

All rows have `flag_refined_any = 0` in the output, so the genuine distressed trading is preserved.
Running `ultra_distressed_filter` on this input with the default configuration confirms it.

The filter leaves genuine distressed bonds alone for these reasons.
1. Even $75-85 prices are well above the ultra-low thresholds ($1-$1.50)
2. The prices are not suspicious round numbers
3. The prices are not identical across days (no plateau)

### Example 5: intraday inconsistency

The input data is for CUSIP `XYZ5`, a single day with `prc_hi` and `prc_lo` columns.

| Row | Date | `prc_hi` (% par) | `prc_lo` (% par) | `prc_hi` ($) | `prc_lo` ($) | Notes |
|-----|------|------------------|------------------|--------------|--------------|-------|
| 0 | 2024-05-10 | 89.0 | 0.10 | $890 | $1 | Erroneous low print |

Filter 4 runs in these steps.

1. The intraday prices are read from the available columns.
   - $\mathcal{I}_0 = \{89.0, 0.10\}$ (all non-NaN values from price_cols)

2. The low price check passes.
   - $\min(\mathcal{I}_0) = 0.10 < \tau_{\mathrm{intraday}} = 20.0$

3. The range is analyzed.
   - $R_{\mathrm{intraday}} = 89.0 - 0.10 = 88.9$
   - $\bar{P}_{\mathrm{intraday}} = \frac{89.0 + 0.10}{2} = 44.55$
   - $\frac{R_{\mathrm{intraday}}}{\bar{P}_{\mathrm{intraday}}} = \frac{88.9}{44.55} \approx 2.0 > 0.75$

4. The row is flagged as intraday inconsistent.

The output is as follows.

| Row | `flag_refined_any` | Notes |
|-----|--------------------|-------|
| 0 | 1 | Intraday inconsistency |

This example shows an erroneous low print of $1 on a day where the high was $890, an implausible 890× intraday swing that indicates a data error. Running `ultra_distressed_filter` on this input confirms it, and the row is flagged `flag_intraday_inconsistent = 1`.

## Default configuration

The defaults come from `stage1/_stage1_settings.py`.

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

    # Intraday inconsistency (Filter 4)
    'price_cols': ['prc_hi', 'prc_lo'],

    # Spike detection (Filter 2)
    'high_spike_threshold': 5.0,       # 5% of par = $50
    'min_spike_ratio': 3.0,
    'recovery_ratio': 2.0,

    'verbose': True,

    # Chunking (read by stage1_pipeline.py, not a parameter of the function)
    'target_rows_per_chunk': 500000,
}
```

Parameters that are not in this config (e.g., `lookback`, `lookforward`, `pre_post_price_ratio`) use the function defaults.

## Typical usage in pipeline

**In the pipeline this is not optional.** `step10a` applies `flag_refined_any != 1`
as an unconditional row filter, so a flagged bond-day is deleted from the published
panel and is not left in it with a flag. Changing any parameter below moves the sample.

**Do not splat `ULTRA_DISTRESSED_CONFIG`.** It contains `price_col`, which would
collide with an explicit `price_col=`, and `target_rows_per_chunk`, which is not a
parameter of this function. Either one raises `TypeError`. `stage1_pipeline.py` passes the
keys individually for that reason, and the code below does the same.

```python
import helper_functions as hf
from _stage1_settings import ULTRA_DISTRESSED_CONFIG as cfg

# The daily panel as it exists mid-pipeline; `pr` is the volume-weighted clean price.
df_flagged = hf.ultra_distressed_filter(
    df_daily,
    price_col=cfg['price_col'],
    intraday_range_threshold=cfg['intraday_range_threshold'],
    intraday_price_threshold=cfg['intraday_price_threshold'],
    ultra_low_threshold=cfg['ultra_low_threshold'],
    min_normal_price_ratio=cfg['min_normal_price_ratio'],
    plateau_ultra_low_threshold=cfg['plateau_ultra_low_threshold'],
    min_plateau_days=cfg['min_plateau_days'],
    suspicious_round_numbers=cfg['suspicious_round_numbers'],
    price_cols=cfg['price_cols'],
    high_spike_threshold=cfg['high_spike_threshold'],
    min_spike_ratio=cfg['min_spike_ratio'],
    recovery_ratio=cfg['recovery_ratio'],
    verbose=cfg['verbose'],
)

# What the pipeline then does, unconditionally:
df_clean = df_flagged[df_flagged["flag_refined_any"] != 1].copy()

n_flagged = df_flagged["flag_refined_any"].sum()
n_total = len(df_flagged)
affected_cusips = df_flagged.loc[df_flagged["flag_refined_any"] == 1, "cusip_id"].nunique()

print(f"Flagged {n_flagged:,} / {n_total:,} observations ({100*n_flagged/n_total:.2f}%)")
print(f"Affected CUSIPs: {affected_cusips:,}")
# The 2026-09-21 run printed the equivalent of:
#   Flagged 10,182 observations (0.03%)
#   Affected CUSIPs: 1,045
```

## Design rationale

### Why four independent filters?

Each type of error needs its own detection logic.

1. The anomaly filter detects isolated downward outliers (temporary data entry errors)
2. The spike filter detects temporary upward outliers (erroneous par or high prints)
3. The plateau filter detects persistent placeholder values (multi-day data gaps)
4. The intraday filter detects within-day inconsistencies (aggregation errors)

Each filter can be disabled independently for sensitivity analysis.

### Why round number heuristic?

Corporate bond prices rarely settle exactly at round levels like $0.01$ or $1.00$, because of bid-ask spreads, accrued interest calculations and market microstructure.

Round numbers in distressed bonds are almost always one of these.
- Placeholder values (e.g., $0.01$ as "effectively zero")
- Data entry errors (e.g., typing $1.00$ instead of the actual price)
- Dealers "cleaning up" their books (per Colin Philipps, FINRA)

### Why ratio-based thresholds?

The thresholds use multiplicative ratios (e.g., $3\times$) instead of absolute differences. A ratio is scale invariant, so it works for both high-yield ($50-80$) and deeply distressed ($5-20$) bonds, and it adapts to the local price level without manual tuning.

### Why lookback/lookforward windows?

Symmetric windows ($L_{\text{back}} = L_{\text{fwd}} = 5$) compare each price to both its past and its future. A permanent change (a default) does not revert in the lookforward window, which is how a genuine event is recognized. A temporary error is isolated from the normal prices around it, which reduces false positives.

### Why separate intraday filter?

Intraday inconsistencies can occur even when the daily aggregates appear normal. A single erroneous trade at $0.10$ among normal trades at $85-90$ may be masked by the daily volume-weighted average. Filter 4 catches these within-day anomalies.

## Performance considerations

### Numba compilation

The core detection functions (`_detect_anomalies_ultra`, `_detect_spikes_ultra`, `_detect_plateaus_ultra`) are compiled with Numba when it is installed. `requirements.txt` installs it only below Python 3.14, and without it the same code runs as plain Python, more slowly. The compilation uses these options.
- `nopython=True` for pure NumPy operations (no Python overhead)
- `cache=True` so that compiled functions are cached across runs
- `fastmath=True` for aggressive floating-point optimizations
- `nogil=True` to release the Python GIL for potential parallelism

The whole Step 8 (the filter plus its chunking and bookkeeping) took about 10 minutes over the full sample on the 2026-09-10 run.

### Memory optimization

At function exit, the individual flag columns are dropped to conserve RAM.
- Only `flag_refined_any` is retained (1 byte per row)
- The metadata columns (`anomaly_type`, `spike_type`, `plateau_id`) are discarded
- This reduces the memory footprint by ~90% for large datasets

## Comparison with other filters

| Aspect | Bounce-back filter | Ultra-distressed filter |
|--------|-------------------|------------------------|
| Data level | Intraday transactions | Daily aggregates |
| Error type | Transient price spikes (both up/down) | Persistent anomalies, plateaus, inconsistencies |
| Detection method | Reversion pattern (lookahead for bounce-back) | Ratio-based outlier detection + plateau sequences |
| Action | Flag transactions, and Stage 0 drops the flagged ones | Delete flagged daily observations (in step 10a) |
| Sequence | Applied in Stage 0 (intraday) | Applied in Stage 1 (daily) |
| Primary use case | Transaction-level cleaning | Post-aggregation quality control |

The two filters complement each other.
- The bounce-back filter removes intraday errors before aggregation
- The ultra-distressed filter catches errors that survive aggregation or appear only in daily data

## References

```bibtex
@unpublished{dickerson2026replication,
  author = {Dickerson, Alexander and Robotti, Cesare and Rossetti, Giulio},
  title = {The Corporate Bond Factor Replication Crisis},
  year = {2026},
  note = {Working Paper. Earlier versions circulated as "Common pitfalls in the evaluation of corporate bond strategies"}
}

@unpublished{dickerson2025constructing,
  author = {Dickerson, Alexander and Rossetti, Giulio},
  title = {Constructing TRACE Corporate Bond Datasets},
  year = {2025},
  note = {Working Paper}
}
```

## See also

- `../stage0/README_bounce_back_filter.md` documents the intraday bounce-back filter (Stage 0)
- `_stage1_settings.py` holds the default configuration parameters
- `stage1/helper_functions.py`, `ultra_distressed_filter()` (tag `filter.ultra_distressed_flags`) is its source
- `stage1/stage1_pipeline.py::step8_ultra_distressed()` is where the pipeline calls it,
  and `step10a_build_filter_tables()` is where flagged rows are deleted
