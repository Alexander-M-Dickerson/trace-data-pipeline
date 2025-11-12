# Decimal Shift Corrector: Technical Documentation

## Overview

The **decimal shift corrector** is a statistical algorithm designed to detect and correct multiplicative price errors in corporate bond transaction data. These errors occur when prices are recorded with incorrect decimal placement (e.g., 10.5 recorded as 105.0 or 1050.0), typically due to data entry mistakes or system conversion errors.

The algorithm tests candidate multiplicative factors against a robust rolling anchor price, accepting corrections only when they demonstrably improve alignment with the bond's recent price history while passing strict acceptance gates.

---

## Mathematical Framework

### Problem Statement

Given a time series of prices $\{P_t\}$ for a bond, identify observations $P_i$ that are likely decimal-shifted from their true value $P_i^*$ by a multiplicative factor $f$:

$$
P_i = f \cdot P_i^* \quad \text{where } f \in \mathcal{F} = \{0.1, 0.01, 10.0, 100.0\}
$$

The algorithm must distinguish genuine price movements from data entry errors.

### Core Algorithm Components

#### 1. Anchor Price Construction

For each observation $i$, compute a **rolling unique-median anchor** $A_i$ using a centered window of width $2w + 1$:

$$
A_i = \text{median}\left(\text{unique}(\{P_{i-w}, \ldots, P_i, \ldots, P_{i+w}\})\right)
$$

**Fallback logic** (when centered window is unavailable):
- If $A_i$ is undefined, use **forward-looking median**: $A_i = \text{median}(\{P_{i+1}, \ldots, P_{i+w+1}\})$
- If still undefined, use **backward-looking median**: $A_i = \text{median}(\{P_{i-w}, \ldots, P_{i-1}\})$
- If still undefined (rare), use **global median**: $A_i = \text{median}(\{P_1, \ldots, P_n\})$

**Note**: The algorithm uses **unique values** in the rolling window to reduce bias from repeated transactions at the same price.

#### 2. Relative Error Metrics

Define the **raw relative error** (before correction):

$$
\epsilon_{\mathrm{raw}}(i) = \frac{|P_i - A_i|}{A_i}
$$

For each candidate factor $f \in \mathcal{F}$, compute the **corrected price** and its **relative error**:

$$
\tilde{P}_i(f) = f \cdot P_i
$$

$$
\epsilon_{\mathrm{corr}}(i, f) = \frac{|\tilde{P}_i(f) - A_i|}{A_i}
$$

#### 3. Acceptance Criteria

A correction with factor $f$ is accepted if **all five conditions** hold:

**Condition 1: Raw error is large (error present)**
$$
\epsilon_{\mathrm{raw}}(i) > \tau_{\mathrm{bad}} \quad \text{(default: } \tau_{\mathrm{bad}} = 0.05 = 5\%)
$$

**Condition 2a: Corrected relative error is small (primary gate)**
$$
\epsilon_{\mathrm{corr}}(i, f) \leq \tau_{\mathrm{pct}} \quad \text{(default: } \tau_{\mathrm{pct}} = 0.02 = 2\%)
$$

**Condition 2b: OR corrected absolute error is small (alternative gate)**
$$
|\tilde{P}_i(f) - A_i| \leq \tau_{\mathrm{abs}} \quad \text{(default: } \tau_{\mathrm{abs}} = 8.0 \text{ price points)}
$$

**Condition 2c: OR par-proximity rule (relaxed gate for near-par bonds)**

If both $|A_i - 100| \leq \delta_{\mathrm{par}}$ and $|\tilde{P}_i(f) - 100| \leq \delta_{\mathrm{par}}$:
$$
\text{Accept correction} \quad \text{(default: } \delta_{\mathrm{par}} = 15.0)
$$

**Condition 3: Corrected error is substantially better than raw error (improvement gate)**
$$
\epsilon_{\mathrm{corr}}(i, f) \leq \gamma \cdot \epsilon_{\mathrm{raw}}(i) \quad \text{(default: } \gamma = 0.2 = 20\%)
$$

**Condition 4: Corrected price is plausible (sanity check)**
$$
P_{\mathrm{low}} \leq \tilde{P}_i(f) \leq P_{\mathrm{high}} \quad \text{(default: } P_{\mathrm{low}} = 5.0, P_{\mathrm{high}} = 300.0)
$$

**Condition 5: Best factor among all candidates (optimality)**

Among all factors satisfying Conditions 1-4, choose the factor $f^*$ that minimizes $\epsilon_{\mathrm{corr}}(i, f)$:
$$
f^* = \arg\min_{f \in \mathcal{F}} \epsilon_{\mathrm{corr}}(i, f)
$$

---

## Function Signature

```python
def decimal_shift_corrector(
    df: pd.DataFrame,
    *,
    id_col: str = "cusip_id",
    date_col: str = "trd_exctn_dt",
    time_col: str | None = "trd_exctn_tm",
    price_col: str = "rptd_pr",
    factors: tuple = (0.1, 0.01, 10.0, 100.0),
    tol_pct_good: float = 0.02,
    tol_abs_good: float = 8.0,
    tol_pct_bad: float = 0.05,
    low_pr: float = 5.0,
    high_pr: float = 300.0,
    anchor: str = "rolling",
    window: int = 5,
    improvement_frac: float = 0.2,
    par_snap: bool = True,
    par_band: float = 15.0,
    output_type: str = "uncleaned"
)
```

---

## Parameters

### Input Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | (required) | Input panel with transaction-level data |
| `id_col` | `str` | `"cusip_id"` | Column name for bond identifier |
| `date_col` | `str` | `"trd_exctn_dt"` | Column name for trade execution date |
| `time_col` | `str \| None` | `"trd_exctn_tm"` | Column name for trade execution time (optional; used for sorting) |
| `price_col` | `str` | `"rptd_pr"` | Column name for reported price to evaluate |

### Algorithm Parameters

| Parameter | Type | Default | Mathematical Notation | Description |
|-----------|------|---------|----------------------|-------------|
| `factors` | `tuple[float]` | `(0.1, 0.01, 10.0, 100.0)` | $\mathcal{F}$ | Candidate multiplicative factors to test |
| `tol_pct_good` | `float` | `0.02` | $\tau_{\mathrm{pct}}$ | Relative error threshold for accepting correction (2% default) |
| `tol_abs_good` | `float` | `8.0` | $\tau_{\mathrm{abs}}$ | Absolute distance threshold in price points (alternative acceptance gate) |
| `tol_pct_bad` | `float` | `0.05` | $\tau_{\mathrm{bad}}$ | Minimum raw relative error to consider a candidate for correction (5% default) |
| `low_pr` | `float` | `5.0` | $P_{\mathrm{low}}$ | Lower bound for plausible corrected prices |
| `high_pr` | `float` | `300.0` | $P_{\mathrm{high}}$ | Upper bound for plausible corrected prices |
| `anchor` | `str` | `"rolling"` | — | Anchor type; currently only `"rolling"` is supported |
| `window` | `int` | `5` | $w$ | Half-window size for rolling anchor (effective window = $2w+1 = 11$ observations) |
| `improvement_frac` | `float` | `0.2` | $\gamma$ | Required proportional improvement vs raw error (20% means corrected error must be $\leq$ 20% of raw error) |
| `par_snap` | `bool` | `True` | — | Enable relaxed acceptance for observations near par ($P = 100$) |
| `par_band` | `float` | `15.0` | $\delta_{\mathrm{par}}$ | Par-proximity band; if both anchor and corrected price are within $\pm 15$ of par, accept |
| `output_type` | `str` | `"uncleaned"` | — | Output format: `"uncleaned"` (add diagnostic columns) or `"cleaned"` (apply corrections) |

---

## Algorithm Logic (Step-by-Step)

### Step 1: Data Preparation
1. Sort DataFrame by `[id_col, date_col, time_col]` (if `time_col` is present)
2. Reset index to ensure contiguous row numbering

### Step 2: Anchor Construction
For each bond (`id_col` group):

1. **Remove duplicate prices**: Drop rows with identical `(id_col, date_col, price_col)` combinations (keep first occurrence)
2. **Compute rolling medians**:
   - Centered median: `window = 2*w + 1, center=True, min_periods=w+1`
   - Forward median: `window = w + 1` on reversed series
   - Backward median: `window = w + 1` on original series
3. **Compose anchor**: Use centered median; if NaN, use forward; if still NaN, use backward; if still NaN, use global median
4. **Merge back**: Join anchor values to original DataFrame via `(id_col, date_col, price_col)`

### Step 3: Candidate Testing
For each row $i$ and each factor $f \in \mathcal{F}$:

1. Compute candidate price: $\tilde{P}_i(f) = P_i \times f$
2. Check plausibility: $P_{\mathrm{low}} \leq \tilde{P}_i(f) \leq P_{\mathrm{high}}$
3. Compute relative error: $\epsilon_{\mathrm{corr}}(i, f) = |\tilde{P}_i(f) - A_i| / A_i$
4. Track best factor: If $\epsilon_{\mathrm{corr}}(i, f) < \epsilon_{\mathrm{best}}$, update best factor

### Step 4: Acceptance Gates
For the best factor $f^*$ at row $i$:

```
raw_rel = |P_i - A_i| / A_i

# Gate 1: Raw error is large enough
IF raw_rel <= tol_pct_bad:
    REJECT (no error to fix)

# Gate 2a: Corrected relative error is small
best_rel = |P_i * f* - A_i| / A_i
accept_rel = (best_rel <= tol_pct_good)

# Gate 2b: Corrected absolute error is small
best_abs = |P_i * f* - A_i|
accept_abs = (best_abs <= tol_abs_good)

# Gate 2c: Par-proximity rule
near_par_anchor = |A_i - 100| <= par_band
near_par_best   = |P_i * f* - 100| <= par_band
accept_par = (near_par_anchor AND near_par_best)  IF par_snap ELSE False

# Gate 3: Improvement requirement
accept_improve = (best_rel <= improvement_frac * raw_rel)

# Final decision
IF (accept_rel OR accept_abs OR accept_par) AND accept_improve:
    ACCEPT correction with factor f*
ELSE:
    REJECT
```

### Step 5: Output Generation

**If `output_type = "uncleaned"`** (default for auditing):
- Return DataFrame with three added columns:
  - `dec_shift_flag` (int8): `1` if correction accepted, `0` otherwise
  - `dec_shift_factor` (float): Chosen factor $f^*$ (or `1.0` if no correction)
  - `suggested_price` (float): Corrected price $\tilde{P}_i(f^*)$ (or original $P_i$ if no correction)

**If `output_type = "cleaned"`** (apply corrections):
- Overwrite `price_col` with `suggested_price` where `dec_shift_flag == 1`
- Return tuple: `(cleaned_df, n_corrected, affected_cusips)`
  - `cleaned_df`: DataFrame with corrected prices
  - `n_corrected`: Count of corrected rows
  - `affected_cusips`: Sorted list of unique bond identifiers with at least one correction

---

## Examples

### Example 1: Basic Decimal Shift Detection

**Input Data** (CUSIP = `12345X678`, date = `2024-01-15`):

| Row | Time | Reported Price | Notes |
|-----|------|----------------|-------|
| 1 | 09:30:00 | 98.5 | Normal trade |
| 2 | 10:00:00 | 99.0 | Normal trade |
| 3 | 10:30:00 | 985.0 | **Error: 10x too high** |
| 4 | 11:00:00 | 98.8 | Reverts to anchor |
| 5 | 11:30:00 | 99.2 | Normal trade |

**Algorithm Execution**:

1. **Anchor Construction** (Row 3, $w = 5$):
   - Unique values in window: $\{98.5, 99.0, 985.0, 98.8, 99.2\}$
   - Centered median (unique): $A_3 = 99.0$

2. **Error Detection**:
   - Raw relative error: $\epsilon_{\mathrm{raw}}(3) = |985.0 - 99.0| / 99.0 = 8.949 = 894.9\%$ ✓ (exceeds 5% threshold)

3. **Candidate Testing**:

   | Factor $f$ | Candidate $\tilde{P}_3(f)$ | Plausible? | $\epsilon_{\mathrm{corr}}(3, f)$ |
   |-----------|---------------------------|-----------|-------------------------------|
   | 0.1 | 98.5 | ✓ | $\|98.5 - 99.0\| / 99.0 = 0.0051 = 0.51\%$ |
   | 0.01 | 9.85 | ✗ (below 5.0) | — |
   | 10.0 | 9850.0 | ✗ (above 300.0) | — |
   | 100.0 | 98500.0 | ✗ (above 300.0) | — |

   - **Best factor**: $f^* = 0.1$ with $\epsilon_{\mathrm{corr}} = 0.51\%$

4. **Acceptance Gates**:
   - Gate 1 (raw error large): $894.9\% > 5\%$ ✓
   - Gate 2a (corrected rel. error small): $0.51\% \leq 2\%$ ✓
   - Gate 3 (improvement): $0.51\% \leq 0.2 \times 894.9\% = 179\%$ ✓

   **Decision**: **ACCEPT** correction

**Output** (if `output_type = "cleaned"`):

| Row | Time | Reported Price → **Corrected Price** | dec_shift_flag |
|-----|------|-------------------------------------|----------------|
| 3 | 10:30:00 | ~~985.0~~ → **98.5** | 1 |

---

### Example 2: False Positive Prevention (Genuine Price Jump)

**Input Data** (CUSIP = `99999Z999`, credit downgrade event):

| Row | Time | Reported Price | Notes |
|-----|------|----------------|-------|
| 1 | 09:00:00 | 95.0 | Pre-downgrade |
| 2 | 09:30:00 | 94.5 | Pre-downgrade |
| 3 | 10:00:00 | 85.0 | **Genuine price drop (downgrade announced)** |
| 4 | 10:30:00 | 84.8 | Post-downgrade |
| 5 | 11:00:00 | 85.5 | Post-downgrade |

**Algorithm Execution** (Row 3):

1. **Anchor**: $A_3 = \text{median}(\{95.0, 94.5, 85.0, 84.8, 85.5\}) = 85.5$

2. **Raw Error**: $\epsilon_{\mathrm{raw}}(3) = |85.0 - 85.5| / 85.5 = 0.0058 = 0.58\%$

3. **Gate 1 Check**: $0.58\% \not> 5\%$ → **REJECT** (raw error too small; not a decimal shift)

**Decision**: **NO correction** (genuine price movement preserved)

---

### Example 3: Par-Proximity Rule

**Input Data** (CUSIP = `88888Y888`, recently issued bond trading near par):

| Row | Time | Reported Price | Notes |
|-----|------|----------------|-------|
| 1 | 09:00:00 | 99.8 | Near par |
| 2 | 09:30:00 | 100.0 | At par |
| 3 | 10:00:00 | 1000.0 | **Error: 10x too high** |
| 4 | 10:30:00 | 100.2 | Near par |

**Algorithm Execution** (Row 3):

1. **Anchor**: $A_3 = 100.0$

2. **Raw Error**: $\epsilon_{\mathrm{raw}}(3) = |1000.0 - 100.0| / 100.0 = 9.0 = 900\%$ ✓

3. **Best Factor**: $f^* = 0.1$ gives $\tilde{P}_3(0.1) = 100.0$

4. **Corrected Error**: $\epsilon_{\mathrm{corr}}(3, 0.1) = |100.0 - 100.0| / 100.0 = 0.0\%$

5. **Acceptance Gates**:
   - Gate 2a: $0.0\% \leq 2\%$ ✓
   - Gate 2c (par rule): $|A_3 - 100| = 0 \leq 15$ ✓ AND $|\tilde{P}_3 - 100| = 0 \leq 15$ ✓
   - Gate 3: $0.0\% \leq 0.2 \times 900\% = 180\%$ ✓

**Decision**: **ACCEPT** (par-proximity rule provides additional confidence)

---

### Example 4: Improvement Gate Rejection

**Input Data** (CUSIP = `77777W777`, noisy price series):

| Row | Time | Reported Price | Anchor | Notes |
|-----|------|----------------|--------|-------|
| 1 | 09:00:00 | 80.0 | — | |
| 2 | 09:30:00 | 120.0 | 100.0 | Volatile |
| 3 | 10:00:00 | 85.0 | 100.0 | Volatile |
| 4 | 10:30:00 | 115.0 | 100.0 | **Test this row** |

**Algorithm Execution** (Row 4):

1. **Raw Error**: $\epsilon_{\mathrm{raw}}(4) = |115.0 - 100.0| / 100.0 = 0.15 = 15\%$ ✓ (exceeds 5%)

2. **Best Factor**: $f^* = 0.1$ gives $\tilde{P}_4(0.1) = 11.5$
   - Plausibility check: $11.5 > 5.0$ ✓
   - $\epsilon_{\mathrm{corr}}(4, 0.1) = |11.5 - 100.0| / 100.0 = 0.885 = 88.5\%$

3. **Improvement Gate**: $88.5\% \not\leq 0.2 \times 15\% = 3\%$ → **FAIL**

**Decision**: **REJECT** (correction makes things worse, not better)

---

## Default Configuration

From `stage0/_trace_settings.py`:

```python
DS_PARAMS = {
    "factors": (0.1, 0.01, 10.0, 100.0),
    "tol_pct_good": 0.02,           # 2% relative error gate
    "tol_abs_good": 8.0,            # 8 price points absolute gate
    "tol_pct_bad": 0.05,            # 5% minimum raw error
    "low_pr": 5.0,                  # Minimum plausible price
    "high_pr": 300.0,               # Maximum plausible price
    "anchor": "rolling",
    "window": 5,                    # Effective window = 11 observations
    "improvement_frac": 0.2,        # 20% improvement requirement
    "par_snap": True,
    "par_band": 15.0,               # Par ± 15 price points
    "output_type": "cleaned",
}
```

---

## Typical Usage in Pipeline

```python
from create_daily_standard_trace import decimal_shift_corrector
from _trace_settings import DS_PARAMS

# Load raw TRACE data
df_raw = pd.read_parquet("trace_enhanced_20240115_raw.parquet")

# Apply decimal shift corrector (returns cleaned data + audit info)
df_clean, n_corrected, affected_cusips = decimal_shift_corrector(
    df_raw,
    id_col="cusip_id",
    date_col="trd_exctn_dt",
    time_col="trd_exctn_tm",
    price_col="rptd_pr",
    output_type="cleaned",
    **DS_PARAMS
)

print(f"Corrected {n_corrected:,} transactions across {len(affected_cusips):,} bonds")
# Output: Corrected 12,847 transactions across 1,203 bonds
```

---

## Design Rationale

### Why Rolling Unique-Median Anchor?

1. **Robustness to outliers**: Median is less sensitive than mean to extreme values
2. **Handles repeated prints**: Using unique values prevents bias from high-frequency trades at the same price
3. **Adaptive to local volatility**: Rolling window adapts to changing price levels (e.g., bond approaching maturity)

### Why Five Acceptance Gates?

1. **Gate 1** (raw error large): Prevents correcting already-good prices
2. **Gates 2a/2b** (corrected error small): Ensures correction brings price close to anchor
3. **Gate 2c** (par rule): Accommodates newly-issued bonds that trade tightly around par
4. **Gate 3** (improvement): Rejects corrections that don't substantially reduce error (prevents random noise corrections)
5. **Gate 4** (plausibility): Sanity check to avoid absurd corrected prices

### Why Not Just Flag All Large Jumps?

Large price changes can be **genuine** due to:
- Credit rating downgrades
- Earnings surprises
- Macroeconomic shocks
- Maturity approaching (bonds converge to par)

The algorithm distinguishes errors from genuine moves by requiring:
1. A multiplicative relationship with a standard factor (10x, 100x, etc.)
2. Alignment with nearby prices in the time series
3. Substantial improvement vs. doing nothing

---

## References

**Dickerson, A., Robotti, C., & Rossetti, G. (2024)**. "Common pitfalls in the evaluation of corporate bond strategies." Working Paper.

---

## See Also

- `README_bounce_back_filter.md` — Documentation for the bounce-back price error filter
- `_trace_settings.py` — Default configuration parameters
- `create_daily_enhanced_trace.py` — Full pipeline implementation
