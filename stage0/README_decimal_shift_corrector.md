# Decimal shift corrector: technical documentation

## Overview

The decimal shift corrector detects and corrects multiplicative price errors in corporate bond transaction data. These errors occur when a price is recorded with the decimal point in the wrong place (10.5 recorded as 105.0 or 1050.0, for example), typically because of a data entry mistake or a system conversion error.

The algorithm tests candidate multiplicative factors against a rolling median anchor price. It accepts a correction only when the corrected price is clearly closer to the bond's prices around that trade (by default a median of the prints just before and after it) and passes the acceptance gates.

## Framework

### Problem statement

Given a time series of prices $\{P_t\}$ for a bond, identify observations $P_i$ that are likely decimal-shifted from their true value $P_i^*$ by a multiplicative factor $f$.

$$
P_i = f \cdot P_i^* \quad \text{where } f \in \mathcal{F} = \{0.1, 0.01, 10.0, 100.0\}
$$

The algorithm must distinguish genuine price movements from data entry errors.

### Core algorithm components

#### 1. Anchor price construction

Anchors are computed on a de-duplicated view of the panel. Rows that share the same
$(\texttt{id\_col}, \texttt{date\_col}, \texttt{price\_col})$ collapse to their first
occurrence, so a price printed ten times the same day contributes once. The medians
below are then plain medians over that de-duplicated series.

For each observation $i$, the primary anchor $A_i$ is a centered rolling median of
width $2w + 1$.

$$
A_i = \mathrm{median}\left(\{P_{i-w}, \ldots, P_i, \ldots, P_{i+w}\}\right)
$$

The centered median requires at least $w+1$ (default 6) observations inside the window.
With fewer it is undefined and the fallbacks below take over, which is what happens for
any bond with a short de-duplicated history.

When the centered window is unavailable, the fallbacks are tried in this order.

- The forward-looking median, $A_i = \mathrm{median}(\{P_i, P_{i+1}, \ldots, P_{i+w}\})$.
- Then the backward-looking median, $A_i = \mathrm{median}(\{P_{i-w}, \ldots, P_{i-1}, P_i\})$.
- Then, only if the price itself is missing, the median of every remaining row in the
  frame, taken across all bonds and not within the bond.

Both fallbacks include $P_i$ itself. If $P_i$ is the decimal-shifted print, it
contaminates its own anchor. On a short series this can pull the anchor far enough
toward the bad price that the correction is rejected. A two-point tail of
$\{1000.0, 100.2\}$ gives a forward anchor of $550.1$, against which neither the true
price nor the shifted one looks right. The centered anchor has this problem to a
smaller degree, because there the bad print is one of eleven values and here it is one
of two.

#### 2. Relative error metrics

The raw relative error (before correction) is

$$
\epsilon_{\mathrm{raw}}(i) = \frac{|P_i - A_i|}{A_i}
$$

For each candidate factor $f \in \mathcal{F}$, compute the corrected price and its relative error.

$$
\tilde{P}_i(f) = f \cdot P_i
$$

$$
\epsilon_{\mathrm{corr}}(i, f) = \frac{|\tilde{P}_i(f) - A_i|}{A_i}
$$

#### 3. Acceptance criteria

A correction with factor $f$ is accepted if all five conditions hold. Condition 2 holds when any one
of 2a, 2b and 2c does.

Condition 1 is that the raw error is large, so an error is present.

$$
\epsilon_{\mathrm{raw}}(i) > \tau_{\mathrm{bad}} \quad \text{(default: } \tau_{\mathrm{bad}} = 0.05 = 5\%)
$$

Condition 2a is that the corrected relative error is small (the primary gate).

$$
\epsilon_{\mathrm{corr}}(i, f) \leq \tau_{\mathrm{pct}} \quad \text{(default: } \tau_{\mathrm{pct}} = 0.02 = 2\%)
$$

Or, as Condition 2b, the corrected absolute error is small (the alternative gate).

$$
|\tilde{P}_i(f) - A_i| \leq \tau_{\mathrm{abs}} \quad \text{(default: } \tau_{\mathrm{abs}} = 8.0 \text{ price points)}
$$

Or, as Condition 2c, the par-proximity rule holds (a relaxed gate for near-par bonds). If both $|A_i - 100| \leq \delta_{\mathrm{par}}$ and the corrected price is within the par band

$$
|\tilde{P}_i(f) - 100| \leq \delta_{\mathrm{par}} \quad \text{(default: } \delta_{\mathrm{par}} = 15.0)
$$

then accept the correction.

Condition 3 is that the corrected error is substantially better than the raw error (the improvement gate).

$$
\epsilon_{\mathrm{corr}}(i, f) \leq \gamma \cdot \epsilon_{\mathrm{raw}}(i) \quad \text{(default: } \gamma = 0.2 = 20\%)
$$

Condition 4 is that the corrected price is plausible (a sanity check).

$$
P_{\mathrm{low}} \leq \tilde{P}_i(f) \leq P_{\mathrm{high}} \quad \text{(default: } P_{\mathrm{low}} = 5.0, P_{\mathrm{high}} = 300.0)
$$

Condition 5 is that the factor is the best among the plausible candidates (optimality).

The search and the gates happen in that order, and only Condition 4 filters the
candidate set. The winner is

$$
f^* = \arg\min_{\{f \,\in\, \mathcal{F} \;:\; P_{\mathrm{low}} \,\leq\, f P_i \,\leq\, P_{\mathrm{high}}\}}
      \epsilon_{\mathrm{corr}}(i, f)
$$

and Conditions 1, 2a/2b/2c and 3 are then applied to $f^*$ alone. A factor that
would have passed the tolerance gates is never reconsidered once a different factor
took the minimum. If no factor is plausible, the row is left untouched.

## Function signature

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

## Parameters

### Input data parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | (required) | Input panel with transaction-level data |
| `id_col` | `str` | `"cusip_id"` | Column name for bond identifier |
| `date_col` | `str` | `"trd_exctn_dt"` | Column name for trade execution date |
| `time_col` | `str \| None` | `"trd_exctn_tm"` | Accepted for signature compatibility with the other filters. Never read, because the function does not sort. |
| `price_col` | `str` | `"rptd_pr"` | Column name for reported price to evaluate |

### Algorithm parameters

| Parameter | Type | Default | Mathematical notation | Description |
|-----------|------|---------|----------------------|-------------|
| `factors` | `tuple[float]` | `(0.1, 0.01, 10.0, 100.0)` | $\mathcal{F}$ | Candidate multiplicative factors to test |
| `tol_pct_good` | `float` | `0.02` | $\tau_{\mathrm{pct}}$ | Relative error threshold for accepting correction (2% default) |
| `tol_abs_good` | `float` | `8.0` | $\tau_{\mathrm{abs}}$ | Absolute distance threshold in price points (alternative acceptance gate) |
| `tol_pct_bad` | `float` | `0.05` | $\tau_{\mathrm{bad}}$ | Minimum raw relative error to consider a candidate for correction (5% default) |
| `low_pr` | `float` | `5.0` | $P_{\mathrm{low}}$ | Lower bound for plausible corrected prices |
| `high_pr` | `float` | `300.0` | $P_{\mathrm{high}}$ | Upper bound for plausible corrected prices |
| `anchor` | `str` | `"rolling"` | none | `"rolling"` uses the centered/fallback median described above. Any other value selects a live alternative branch, a plain per-`(id_col, date_col)` median, which is one anchor per bond-day. |
| `window` | `int` | `5` | $w$ | Half-window size for rolling anchor (effective window = $2w+1 = 11$ observations) |
| `improvement_frac` | `float` | `0.2` | $\gamma$ | Required proportional improvement vs raw error (20% means corrected error must be $\leq$ 20% of raw error) |
| `par_snap` | `bool` | `True` | none | Enable relaxed acceptance for observations near par ($P = 100$) |
| `par_band` | `float` | `15.0` | $\delta_{\mathrm{par}}$ | Par-proximity band. If both anchor and corrected price are within $\pm 15$ of par, accept |
| `output_type` | `str` | `"uncleaned"` | none | Output format, either `"uncleaned"` (add diagnostic columns) or `"cleaned"` (apply corrections) |

## Algorithm logic (step-by-step)

### Step 1: Data preparation

**The function does not sort** and does not reset the index. It works on the row order
it is given, and `time_col` is accepted but never read. In the pipeline the frame arrives
already ordered by `[cusip_id, trd_exctn_dt, trd_exctn_tm]`. If you call the function
directly on unordered rows, the rolling anchors are computed over that order and are
meaningless. Sort before calling.

### Step 2: Anchor construction

The anchor is built for each bond (`id_col` group).

1. Remove duplicate prices. Drop rows with identical `(id_col, date_col, price_col)` combinations and keep the first occurrence.
2. Compute the rolling medians, per bond, on the de-duplicated series.
   - The centered median uses `window = 2*w + 1, center=True, min_periods=w+1` and is undefined below 6 observations.
   - The forward median uses `window = w + 1, min_periods=1` on the reversed series, which is $\{P_i, \ldots, P_{i+w}\}$.
   - The backward median uses `window = w + 1, min_periods=1` on the original series, which is $\{P_{i-w}, \ldots, P_i\}$.
3. Compose the anchor. Take the centered median. If it is NaN, take the forward one. If that is still NaN, take the backward one. If that is still NaN, take the median of the whole de-duplicated frame (all bonds).
4. Merge back. Join the anchor values to the original DataFrame via `(id_col, date_col, price_col)`, `validate="m:1"`. Every row finds its match, because every `(id_col, date_col, price_col)` is in the de-duplicated view by construction. A row whose anchor is still missing after the join falls back to the plain `(id_col, date_col)` median.

### Step 3: Candidate testing

For each row $i$ and each factor $f \in \mathcal{F}$, do the following.

1. Compute the candidate price $\tilde{P}_i(f) = P_i \times f$.
2. Check plausibility. The candidate must satisfy $P_{\mathrm{low}} \leq \tilde{P}_i(f) \leq P_{\mathrm{high}}$.
3. Compute the relative error $\epsilon_{\mathrm{corr}}(i, f) = |\tilde{P}_i(f) - A_i| / A_i$.
4. Track the best factor. If $\epsilon_{\mathrm{corr}}(i, f) < \epsilon_{\mathrm{best}}$, update the best factor.

### Step 4: Acceptance gates

The gates are applied to the best factor $f^*$ at row $i$.

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

### Step 5: Output generation

With `output_type = "uncleaned"` (the default, for auditing), the function returns the DataFrame with seven added columns.

- `dec_shift_flag` (int8) is `1` if a correction was accepted and `0` otherwise.
- `dec_shift_factor` (float) is the chosen factor $f^*$ (or `1.0` if there is no correction).
- `suggested_price` (float) is the corrected price $\tilde{P}_i(f^*)$ (or the original $P_i$ if there is no correction).
- `anchor_price` (float) is the anchor the tests used.
- `anchor_med_center`, `anchor_med_fwd` and `anchor_med_back` (float) are the three medians it was composed from.

With `output_type = "cleaned"` (apply corrections), the function overwrites `price_col` with `suggested_price` where `dec_shift_flag == 1` and returns the tuple `(cleaned_df, n_corrected, affected_cusips)`.

- `cleaned_df` is the DataFrame with corrected prices.
- `n_corrected` is the count of corrected rows.
- `affected_cusips` is the sorted list of unique bond identifiers with at least one correction.

## Examples

### Example 1: Basic decimal shift detection

Every number below was produced by running `decimal_shift_corrector` on the stated
input, not derived by hand. The series are 13 prints long because the centered anchor
needs six observations, and a four- or five-row toy never reaches it.

The input is CUSIP `12345X678` on `2024-01-15`, with one 10x print at row 7.

```
98.5  99.0  98.7  99.1  98.6  98.9  985.0  98.8  99.2  98.4  99.3  98.3  99.4
                                     ^^^^^ row 7
```

At row 7 the algorithm runs as follows.

1. The centered window is available, so the anchor is $A_7 = 98.90$.

2. The raw relative error is $\epsilon_{\mathrm{raw}} = |985.0 - 98.90| / 98.90 = 8.9596$
   (896.0%), far past the 5% floor.

3. The candidates are tested.

   | Factor $f$ | Candidate $f \cdot P_7$ | Plausible? | $\epsilon_{\mathrm{corr}}$ |
   |-----------|------------------------|-----------|---------------------------|
   | 0.1 | 98.50 | yes | 0.0040 |
   | 0.01 | 9.85 | yes | 0.9004 |
   | 10.0 | 9,850 | no (above 300.0) | n/a |
   | 100.0 | 98,500 | no (above 300.0) | n/a |

   The best factor is $f^* = 0.1$, with $\epsilon_{\mathrm{corr}} = 0.0040$.

4. The acceptance gates pass. Gate 1 passes ($8.9596 > 0.05$), Gate 2a passes
   ($0.0040 \leq 0.02$) and Gate 3 passes ($0.0040 \leq 0.2 \times 8.9596 = 1.7919$).

   The decision is to accept, with `dec_shift_flag = 1`, `dec_shift_factor = 0.1` and
   `suggested_price = 98.50`.

### Example 2: False positive prevention (genuine price jump)

The input is CUSIP `99999Z999`, with a credit downgrade between rows 6 and 7.

```
95.0  94.8  95.2  94.5  95.1  94.9  85.0  84.8  85.5  84.9  85.2  85.3  84.7
                                    ^^^^ row 7
```

At row 7 the algorithm runs as follows.

1. The anchor is $A_7 = 85.50$. The centered window straddles the downgrade, and the median
   has already moved to the post-downgrade level, which is the point of a median anchor.

2. The raw relative error is $|85.0 - 85.50| / 85.50 = 0.0058$ (0.58%).

3. Gate 1 rejects, because $0.0058 \not> 0.05$. The price is where its neighbours are, so
   there is no error to fix.

The decision is no correction. A 10-point genuine move is invisible to this filter, which
is the intended behaviour. It looks for multiplicative errors, not for large moves.

### Example 3: Par-proximity rule

The input is CUSIP `88888Y888`, recently issued and trading tightly around par.

```
99.8  100.0  99.9  100.1  99.7  100.2  1000.0  100.3  99.6  100.4  99.5  100.5  99.4
                                       ^^^^^^ row 7
```

At row 7 the algorithm runs as follows.

1. The anchor is $A_7 = 100.10$.
2. The raw relative error is $|1000.0 - 100.10| / 100.10 = 8.9900$ (899.0%), which is large.
3. The best factor is $f^* = 0.1$, which gives $100.00$ and $\epsilon_{\mathrm{corr}} = 0.0010$.
   ($f = 0.01$ gives $10.00$, plausible but with $\epsilon = 0.9001$, and the 10x and 100x
   candidates exceed `high_pr`.)
4. The gates pass. 2a passes ($0.0010 \leq 0.02$), 2b passes
   ($|100.00 - 100.10| = 0.10 \leq 8.0$), 2c passes (anchor and candidate both within
   $\pm15$ of par) and Gate 3 passes.

   The decision is to accept, with `suggested_price = 100.00`.

Here the par rule does not decide the outcome, because 2a already passes on its own. In the
next example the par rule is the only tolerance gate that passes, and the correction is still
refused.

### Example 4: Improvement gate rejection

The input is CUSIP `77777W777`, trading near 90, with one print at 11.25.

```
90.0  89.5  90.5  89.8  90.2  90.6  11.25  90.1  89.9  90.3  89.7  90.4  89.6
                                    ^^^^^ row 7
```

At row 7 the algorithm runs as follows.

1. The anchor is $A_7 = 90.10$.
2. The raw relative error is $|11.25 - 90.10| / 90.10 = 0.8751$ (87.5%). That is a large
   error, and $11.25 \approx 112.5 / 10$ looks exactly like a 10x down-shift.
3. The best factor is $f^* = 10.0$, which gives $112.50$. It is the only plausible candidate
   ($1.125$, $0.1125$ and $1{,}125$ all fall outside $[5, 300]$).
   $\epsilon_{\mathrm{corr}} = |112.50 - 90.10| / 90.10 = 0.2486$.
4. The gates are checked in turn.
   - 2a fails, because $0.2486 \not\leq 0.02$.
   - 2b fails, because $|112.50 - 90.10| = 22.40 \not\leq 8.0$.
   - 2c passes, because $|90.10 - 100| = 9.90 \leq 15$ and $|112.50 - 100| = 12.50 \leq 15$, so
     the par rule would accept.
   - Gate 3 fails, because $0.2486 \not\leq 0.2 \times 0.8751 = 0.1750$.

   The decision is to reject. The par rule passed and the correction was still refused,
   because $112.50$ is 24.9% away from the anchor when the improvement gate demands
   17.5% or better. Gate 3 exists for this case. The par band is 30 points wide,
   so on a bond trading near par it lets through corrections that are plausible
   without being right.

## Default configuration

The defaults are in `stage0/_trace_settings.py`.

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

## Typical usage in pipeline

Stage 0 calls this inside each chunk and writes no trade-level file, so the input below is
a trade-level frame you hold yourself (the file name is illustrative).

```python
import pandas as pd
from create_daily_standard_trace import decimal_shift_corrector
from _trace_settings import DS_PARAMS

# Trade-level TRACE data (illustrative file name)
df_raw = pd.read_parquet("my_trades.parquet")

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
# On the full Enhanced sample (2026-09-10 run) this came to 2,223 transactions across
# 1,149 bonds, summed over the run's 485 chunks.
```

## Design rationale

### Why these acceptance conditions?

The numbers are those of *Acceptance criteria* above.

1. Condition 1 (raw error large) prevents correcting prices that are already good.
2. Conditions 2a and 2b (corrected error small) make sure the correction brings the price close to the anchor.
3. Condition 2c (par rule) accommodates newly issued bonds that trade tightly around par.
4. Condition 3 (improvement) rejects corrections that do not substantially reduce the error, which prevents corrections of random noise.
5. Condition 4 (plausibility) is a sanity check that avoids absurd corrected prices. It is applied in the candidate search (Step 3), before the gates of Step 4.

### Why not just flag all large jumps?

The algorithm distinguishes errors from genuine moves by requiring all of the following.

1. A multiplicative relationship with a standard factor (10x, 100x, etc.)
2. Alignment with nearby prices in the time series
3. Substantial improvement vs. doing nothing

## References

Dickerson, A., Robotti, C., & Rossetti, G. (2026). "The Corporate Bond Factor Replication Crisis." Working Paper. Earlier versions circulated as "Common pitfalls in the evaluation of corporate bond strategies."

## See also

- `README_bounce_back_filter.md` documents the bounce-back price error filter.
- `_trace_settings.py` holds the default configuration parameters.
- `create_daily_enhanced_trace.py` is the full pipeline implementation.
