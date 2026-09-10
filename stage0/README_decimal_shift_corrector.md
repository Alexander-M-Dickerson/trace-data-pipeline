# Decimal Shift Corrector: Technical Documentation

## Overview

The **decimal shift corrector** is an algorithm designed to detect and correct multiplicative price errors in corporate bond transaction data. These errors occur when prices are recorded with incorrect decimal placement (e.g., 10.5 recorded as 105.0 or 1050.0), typically due to data entry mistakes or system conversion errors.

The algorithm tests candidate multiplicative factors against a robust rolling anchor price, accepting corrections only when they demonstrably improve alignment with the bond's recent price history while passing strict acceptance gates.

---

## Framework

### Problem Statement

Given a time series of prices $\{P_t\}$ for a bond, identify observations $P_i$ that are likely decimal-shifted from their true value $P_i^*$ by a multiplicative factor $f$:

$$
P_i = f \cdot P_i^* \quad \text{where } f \in \mathcal{F} = \{0.1, 0.01, 10.0, 100.0\}
$$

The algorithm must distinguish genuine price movements from data entry errors.

### Core Algorithm Components

#### 1. Anchor Price Construction

Anchors are computed on a **de-duplicated** view of the panel: rows sharing the same
$(\texttt{id\_col}, \texttt{date\_col}, \texttt{price\_col})$ collapse to their first
occurrence, so a price printed ten times the same day contributes once. The medians
below are then plain medians over that de-duplicated series.

For each observation $i$, the primary anchor $A_i$ is a **centered rolling median** of
width $2w + 1$:

$$
A_i = \mathrm{median}\left(\{P_{i-w}, \ldots, P_i, \ldots, P_{i+w}\}\right)
$$

❗**This requires at least $w+1$ (default 6) observations inside the window.** With
fewer, the centered median is undefined and the fallbacks below take over — which is
what happens for any bond with a short de-duplicated history.

**Fallback logic** (when the centered window is unavailable):

- **Forward-looking median**: $A_i = \mathrm{median}(\{P_i, P_{i+1}, \ldots, P_{i+w}\})$
- then **backward-looking median**: $A_i = \mathrm{median}(\{P_{i-w}, \ldots, P_{i-1}, P_i\})$
- then, only if the price itself is missing, the **median of every remaining row in the
  frame** — across all bonds, not within the bond.

❗**Both fallbacks include $P_i$ itself.** If $P_i$ is the decimal-shifted print, it
contaminates its own anchor. On a short series this can pull the anchor far enough
toward the bad price that the correction is rejected: a two-point tail of
$\{1000.0, 100.2\}$ gives a forward anchor of $550.1$, against which neither the true
price nor the shifted one looks right. The centered anchor does not have this problem
in the same degree, because the bad print is one of eleven values rather than one of
two.

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

If both $|A_i - 100| \leq \delta_{\mathrm{par}}$ and the corrected price is within par band:

$$
|\tilde{P}_i(f) - 100| \leq \delta_{\mathrm{par}} \quad \text{(default: } \delta_{\mathrm{par}} = 15.0)
$$

Then accept correction.

**Condition 3: Corrected error is substantially better than raw error (improvement gate)**

$$
\epsilon_{\mathrm{corr}}(i, f) \leq \gamma \cdot \epsilon_{\mathrm{raw}}(i) \quad \text{(default: } \gamma = 0.2 = 20\%)
$$

**Condition 4: Corrected price is plausible (sanity check)**

$$
P_{\mathrm{low}} \leq \tilde{P}_i(f) \leq P_{\mathrm{high}} \quad \text{(default: } P_{\mathrm{low}} = 5.0, P_{\mathrm{high}} = 300.0)
$$

**Condition 5: Best factor among the plausible candidates (optimality)**

The search and the gates happen in that order, and only **Condition 4** filters the
candidate set. The winner is

$$
f^* = \arg\min_{\{f \,\in\, \mathcal{F} \;:\; P_{\mathrm{low}} \,\leq\, f P_i \,\leq\, P_{\mathrm{high}}\}}
      \epsilon_{\mathrm{corr}}(i, f)
$$

and Conditions 1, 2a/2b/2c and 3 are then applied to **$f^*$ alone**. A factor that
would have passed the tolerance gates is never reconsidered once a different factor
took the minimum. If no factor is plausible, the row is left untouched.

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
| `time_col` | `str \| None` | `"trd_exctn_tm"` | Accepted for signature compatibility with the other filters. **Never read** — the function does not sort. |
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
| `anchor` | `str` | `"rolling"` | — | `"rolling"` uses the centered/fallback median described above. **Any other value** selects a live alternative branch: a plain per-`(id_col, date_col)` median, i.e. one anchor per bond-day. |
| `window` | `int` | `5` | $w$ | Half-window size for rolling anchor (effective window = $2w+1 = 11$ observations) |
| `improvement_frac` | `float` | `0.2` | $\gamma$ | Required proportional improvement vs raw error (20% means corrected error must be $\leq$ 20% of raw error) |
| `par_snap` | `bool` | `True` | — | Enable relaxed acceptance for observations near par ($P = 100$) |
| `par_band` | `float` | `15.0` | $\delta_{\mathrm{par}}$ | Par-proximity band; if both anchor and corrected price are within $\pm 15$ of par, accept |
| `output_type` | `str` | `"uncleaned"` | — | Output format: `"uncleaned"` (add diagnostic columns) or `"cleaned"` (apply corrections) |

---

## Algorithm Logic (Step-by-Step)

### Step 1: Data Preparation

❗**The function does not sort and does not reset the index** — it works on the row order
it is given, and `time_col` is accepted but never read. In the pipeline the frame arrives
already ordered by `[cusip_id, trd_exctn_dt, trd_exctn_tm]`; if you call the function
directly on unordered rows, the rolling anchors are computed over that order and are
meaningless. Sort before calling.

### Step 2: Anchor Construction
For each bond (`id_col` group):

1. **Remove duplicate prices**: Drop rows with identical `(id_col, date_col, price_col)` combinations (keep first occurrence)
2. **Compute rolling medians** (per bond, on the de-duplicated series):
   - Centered median: `window = 2*w + 1, center=True, min_periods=w+1` — undefined below 6 observations
   - Forward median: `window = w + 1, min_periods=1` on the reversed series, i.e. $\{P_i, \ldots, P_{i+w}\}$
   - Backward median: `window = w + 1, min_periods=1` on the original series, i.e. $\{P_{i-w}, \ldots, P_i\}$
3. **Compose anchor**: centered; if NaN, forward; if still NaN, backward; if still NaN, the median of the whole de-duplicated frame (all bonds)
4. **Merge back**: Join anchor values to the original DataFrame via `(id_col, date_col, price_col)`, `validate="m:1"`. Rows that find no match (their price was not in the de-duplicated view) fall back to the plain `(id_col, date_col)` median.

### Step 3: Candidate Testing
For each row $i$ and each factor $f \in \mathcal{F}$:

1. Compute candidate price: $\tilde{P}_i(f) = P_i \times f$
2. Check plausibility: Candidate must satisfy $P_{\mathrm{low}} \leq \tilde{P}_i(f) \leq P_{\mathrm{high}}$
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

Every number below was produced by running `decimal_shift_corrector` on the stated
input, not derived by hand. The series are 13 prints long because the centered anchor
needs six observations; a four- or five-row toy never reaches it.

**Input Data** (CUSIP = `12345X678`, date = `2024-01-15`, one 10x print at row 7):

```
98.5  99.0  98.7  99.1  98.6  98.9  985.0  98.8  99.2  98.4  99.3  98.3  99.4
                                     ^^^^^ row 7
```

**Algorithm Execution** (row 7):

1. **Anchor**: the centered window is available, $A_7 = 98.90$.

2. **Raw relative error**: $\epsilon_{\mathrm{raw}} = |985.0 - 98.90| / 98.90 = 8.9596$
   (896.0%) — comfortably past the 5% floor.

3. **Candidate testing**:

   | Factor $f$ | Candidate $f \cdot P_7$ | Plausible? | $\epsilon_{\mathrm{corr}}$ |
   |-----------|------------------------|-----------|---------------------------|
   | 0.1 | 98.50 | ✓ | 0.0040 |
   | 0.01 | 9.85 | ✓ | 0.9004 |
   | 10.0 | 9,850 | ✗ (above 300.0) | — |
   | 100.0 | 98,500 | ✗ (above 300.0) | — |

   **Best factor**: $f^* = 0.1$, $\epsilon_{\mathrm{corr}} = 0.0040$.

4. **Acceptance gates**: Gate 1 ✓ ($8.9596 > 0.05$); Gate 2a ✓ ($0.0040 \leq 0.02$);
   Gate 3 ✓ ($0.0040 \leq 0.2 \times 8.9596 = 1.7919$).

   **Decision: ACCEPT** — `dec_shift_flag = 1`, `dec_shift_factor = 0.1`,
   `suggested_price = 98.50`.

---

### Example 2: False Positive Prevention (Genuine Price Jump)

**Input Data** (CUSIP = `99999Z999`, a credit downgrade between rows 6 and 7):

```
95.0  94.8  95.2  94.5  95.1  94.9  85.0  84.8  85.5  84.9  85.2  85.3  84.7
                                    ^^^^ row 7
```

**Algorithm Execution** (row 7):

1. **Anchor**: $A_7 = 85.50$. The centered window straddles the downgrade, and the median
   has already moved to the post-downgrade level — which is the point of a median anchor.

2. **Raw relative error**: $|85.0 - 85.50| / 85.50 = 0.0058$ (0.58%).

3. **Gate 1**: $0.0058 \not> 0.05$ → **REJECT**. The price is where its neighbours are;
   there is no error to fix.

**Decision: NO correction.** A 10-point genuine move is invisible to this filter, which
is the intended behaviour — it looks for *multiplicative* errors, not large moves.

---

### Example 3: Par-Proximity Rule

**Input Data** (CUSIP = `88888Y888`, recently issued, trading tightly around par):

```
99.8  100.0  99.9  100.1  99.7  100.2  1000.0  100.3  99.6  100.4  99.5  100.5  99.4
                                       ^^^^^^ row 7
```

**Algorithm Execution** (row 7):

1. **Anchor**: $A_7 = 100.10$.
2. **Raw relative error**: $|1000.0 - 100.10| / 100.10 = 8.9900$ (899.0%) ✓
3. **Best factor**: $f^* = 0.1$ → $100.00$, $\epsilon_{\mathrm{corr}} = 0.0010$.
   ($f = 0.01$ gives $10.00$, plausible but $\epsilon = 0.9001$; the 10x and 100x
   candidates exceed `high_pr`.)
4. **Gates**: 2a ✓ ($0.0010 \leq 0.02$); 2b ✓ ($|100.00 - 100.10| = 0.10 \leq 8.0$);
   2c ✓ (anchor and candidate both within $\pm15$ of par); Gate 3 ✓.

   **Decision: ACCEPT** → `suggested_price = 100.00`.

Note that here the par rule is not doing the work — 2a already passes on its own. The
next example is one where the par rule *is* the only tolerance gate that passes, and the
correction is still refused.

---

### Example 4: Improvement Gate Rejection

**Input Data** (CUSIP = `77777W777`, trading near 90, one print at 11.25):

```
90.0  89.5  90.5  89.8  90.2  90.6  11.25  90.1  89.9  90.3  89.7  90.4  89.6
                                    ^^^^^ row 7
```

**Algorithm Execution** (row 7):

1. **Anchor**: $A_7 = 90.10$.
2. **Raw relative error**: $|11.25 - 90.10| / 90.10 = 0.8751$ (87.5%) ✓ — a large error,
   and $11.25 \approx 112.5 / 10$ looks exactly like a 10x down-shift.
3. **Best factor**: $f^* = 10.0$ → $112.50$. It is the only plausible candidate
   ($1.125$, $0.1125$ and $1{,}125$ all fall outside $[5, 300]$).
   $\epsilon_{\mathrm{corr}} = |112.50 - 90.10| / 90.10 = 0.2486$.
4. **Gates**:
   - 2a ✗ — $0.2486 \not\leq 0.02$
   - 2b ✗ — $|112.50 - 90.10| = 22.40 \not\leq 8.0$
   - **2c ✓** — $|90.10 - 100| = 9.90 \leq 15$ and $|112.50 - 100| = 12.50 \leq 15$, so
     the par rule *would* accept
   - **Gate 3 ✗** — $0.2486 \not\leq 0.2 \times 0.8751 = 0.1750$

   **Decision: REJECT.** The par rule passed and the correction was still refused,
   because $112.50$ is 24.9% away from the anchor when the improvement gate demands
   17.5% or better. This is the case Gate 3 exists for: the par band is 30 points wide,
   so on a bond trading near par it will wave through corrections that are merely
   *plausible* rather than *right*.

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
# DS_PARAMS already carries output_type="cleaned". Passing it explicitly AND
# splatting the dict raises TypeError: got multiple values for 'output_type'.
df_clean, n_corrected, affected_cusips = decimal_shift_corrector(
    df_raw,
    id_col="cusip_id",
    date_col="trd_exctn_dt",
    time_col="trd_exctn_tm",
    price_col="rptd_pr",
    **DS_PARAMS
)

print(f"Corrected {n_corrected:,} transactions across {len(affected_cusips):,} bonds")
# Output: Corrected 12,847 transactions across 1,203 bonds
```

---

## Design Rationale

### Why Five Acceptance Gates?

1. **Gate 1** (raw error large): Prevents correcting already-good prices
2. **Gates 2a/2b** (corrected error small): Ensures correction brings price close to anchor
3. **Gate 2c** (par rule): Accommodates newly-issued bonds that trade tightly around par
4. **Gate 3** (improvement): Rejects corrections that don't substantially reduce error (prevents random noise corrections)
5. **Gate 4** (plausibility): Sanity check to avoid absurd corrected prices

### Why Not Just Flag All Large Jumps?

The algorithm distinguishes errors from genuine moves by requiring:
1. A multiplicative relationship with a standard factor (10x, 100x, etc.)
2. Alignment with nearby prices in the time series
3. Substantial improvement vs. doing nothing

---

## References

**Dickerson, A., Robotti, C., & Rossetti, G. (2025)**. "Common pitfalls in the evaluation of corporate bond strategies." Working Paper.

---

## See Also

- `README_bounce_back_filter.md` — Documentation for the bounce-back price error filter
- `_trace_settings.py` — Default configuration parameters
- `create_daily_enhanced_trace.py` — Full pipeline implementation
