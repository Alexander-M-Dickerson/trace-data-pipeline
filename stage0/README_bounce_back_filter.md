# Bounce-Back Filter: Technical Documentation

## Overview

The **bounce-back filter** (formally: `flag_price_change_errors`) is a sophisticated algorithm designed to detect transient price-entry errors in corporate bond transaction data. These errors manifest as **large price spikes that quickly revert** to a trailing baseline, indicating data entry mistakes rather than genuine market movements.

The algorithm uses backward-looking anchors, lookahead windows, and par-specific heuristics to identify erroneous transactions while preserving genuine volatility such as credit events, liquidity shocks, and market dislocations.

---

## Mathematical Framework

### Problem Statement

Given a time-ordered sequence of transaction prices $\{P_1, P_2, \ldots, P_n\}$ for a bond, identify observations $P_i$ that represent **transient errors** characterized by:

1. **Large jump**: $|\Delta P_i| = |P_i - P_{i-1}| \geq \tau$ (price change exceeds threshold $\tau$)
2. **Quick reversion**: Within a lookahead window of $L$ trades, price returns partially toward a backward-looking anchor
3. **Displacement from anchor**: Price $P_i$ is significantly displaced from the trailing baseline $B_i$

The algorithm must distinguish these errors from **genuine price moves** (e.g., credit downgrades) that persist over multiple trades.

### Core Algorithm Components

#### 1. Backward-Looking Anchor (Trailing Baseline)

For each observation $i$, compute a **strictly backward-looking anchor** $B_i$ using only past data:

$$
B_i = \text{median}\left(\text{unique}(\{P_{i-w}, \ldots, P_{i-1}\})\right)
$$

where:
- $w$ = window size (default: 5)
- **unique()** removes duplicate prices to reduce bias from repeated prints
- $B_i$ is **shifted by 1** to ensure no look-ahead bias (does not include $P_i$)

**Key Property**: $B_i$ is computed using only transactions **before** row $i$, making the filter suitable for real-time detection.

#### 2. Price Change Dynamics

Define the **one-step price change** (delta):

$$
\Delta P_i = P_i - P_{i-1}
$$

And the **displacement from baseline**:

$$
d_i = P_i - B_i
$$

#### 3. Candidate Opening Conditions

A row $i$ becomes a **candidate for flagging** if **any** of the following conditions hold:

**Condition 1: Large jump relative to previous trade**

$$
|\Delta P_i| \geq \tau - \delta_{\mathrm{slack}} \quad \text{(default: } \tau = 35.0, \delta_{\mathrm{slack}} = 1.0)
$$

**Condition 2: Large displacement from baseline**

$$
|d_i| = |P_i - B_i| \geq \tau - \delta_{\mathrm{slack}}
$$

**Condition 3: Par-spike heuristic (if enabled)**

If $|P_i - P_{\mathrm{par}}| \leq \epsilon_{\mathrm{par}}$ (price is at par, default $P_{\mathrm{par}} = 100.0$, $\epsilon_{\mathrm{par}} = 10^{-8}$):

$$
|P_i - B_i| \geq \alpha \cdot \tau \quad \text{(default: } \alpha = 0.25)
$$

---

#### 4. Bounce-Back Resolution (Lookahead Scan)

Once a candidate is opened at row $i$, scan forward up to $L$ rows (default $L = 5$) to find evidence of **reversion**:

**Path A: Opposite-signed large move**

Find the first row $j > i$ such that:

$$
\operatorname{sign}(\Delta P_j) = -\operatorname{sign}(\Delta P_i) \quad \text{AND} \quad |\Delta P_j| \geq \tau - \delta_{\mathrm{slack}}
$$

**Path B: Return to anchor**

Find the first row $k > i$ such that:

$$
|P_k - B_i| \leq \alpha \cdot \tau
$$

**Resolution**: If either Path A (row $j$) or Path B (row $k$) is found, the algorithm proceeds to flagging logic. Otherwise, the candidate is **rejected** (no bounce-back detected).

---

#### 5. Flagging Logic and Plateau Extension

Once a bounce-back is detected (resolved at row $j_{\mathrm{stop}}$):

**Step 1: Reassignment (Blame Attribution)**

Check if the **previous row** ($i-1$) is more displaced than the current row ($i$):

$$
|P_{i-1} - B_{i-1}| - |P_i - B_i| \geq \delta_{\mathrm{reassign}} \quad \text{(default: } \delta_{\mathrm{reassign}} = 5.0)
$$

AND

$$
|P_{i-1} - B_{i-1}| \geq \alpha \cdot \tau
$$

If both hold, reassign the flag to row $i-1$ (it was the true error).

**Step 2: Flag the Start**

Flag the identified error row (either $i$ or $i-1$).

**Step 3: Extend Plateau Flags**

Flag additional rows in the range $[i_{\mathrm{start}}+1, \min(j_{\mathrm{stop}}, i_{\mathrm{start}} + S)]$ if they remain **displaced from baseline**:

For each row $k$ in this range:

- **If par-spike**: Flag if $|P_k - P_{\mathrm{par}}| \leq \epsilon_{\mathrm{par}}$
- **If non-par**: Flag if $|P_k - B_{i_{\mathrm{start}}}| \geq \alpha \cdot \tau$; stop at first row that fails this test

Where $S$ = max span (default: 5).

---

#### 6. Par-Specific Heuristic (Persistent Par Blocks)

For bonds trading at par, apply special handling to avoid false positives from legitimate par prints:

**Persistent Par Block Detection**:

If a sequence of trades $\{P_i, P_{i+1}, \ldots, P_j\}$ all satisfy $|P_k - P_{\mathrm{par}}| \leq \epsilon_{\mathrm{par}}$:

- Compute run length: $\ell = j - i + 1$
- **Only flag** if $\ell \geq \ell_{\min}$ (default: $\ell_{\min} = 3$)

**Cooldown Period**:

After flagging a par block ending at row $j$, suppress further flags for the next $C$ rows (default: $C = 2$):

$$
\text{No flags issued for rows } j+1, j+2, \ldots, j+C
$$

This prevents cascading false positives in par-trading regions.

---

## Function Signature

```python
def flag_price_change_errors(
    df: pd.DataFrame,
    *,
    id_col: str = "cusip_id",
    date_col: str = "trd_exctn_dt",
    time_col: Optional[str] = "trd_exctn_tm",
    price_col: str = "rptd_pr",
    threshold_abs: float = 35.0,
    lookahead: int = 5,
    max_span: int = 5,
    window: int = 5,
    back_to_anchor_tol: float = 0.25,
    candidate_slack_abs: float = 1.0,
    reassignment_margin_abs: float = 5.0,
    use_unique_trailing_median: bool = True,
    par_spike_heuristic: bool = True,
    par_level: float = 100.0,
    par_equal_tol: float = 1e-8,
    par_min_run: int = 3,
    par_cooldown_after_flag: int = 2,
) -> pd.DataFrame
```

---

## Parameters

### Input Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | (required) | Input panel with intraday transaction data |
| `id_col` | `str` | `"cusip_id"` | Column name for bond identifier |
| `date_col` | `str` | `"trd_exctn_dt"` | Column name for trade execution date |
| `time_col` | `str \| None` | `"trd_exctn_tm"` | Column name for trade execution time (used for sorting; **recommended**) |
| `price_col` | `str` | `"rptd_pr"` | Column name for reported price |

### Core Detection Parameters

| Parameter | Type | Default | Mathematical Notation | Description |
|-----------|------|---------|----------------------|-------------|
| `threshold_abs` | `float` | `35.0` | $\tau$ | Minimum absolute price change (in price points) to open a candidate |
| `lookahead` | `int` | `5` | $L$ | Maximum number of rows ahead to search for bounce-back |
| `max_span` | `int` | `5` | $S$ | Maximum total path length from candidate to resolution (caps plateau extension) |
| `window` | `int` | `5` | $w$ | Backward window length for trailing median anchor |
| `back_to_anchor_tol` | `float` | `0.25` | $\alpha$ | Fraction of threshold; prices within $\alpha \cdot \tau$ of anchor are "returned" |
| `candidate_slack_abs` | `float` | `1.0` | $\delta_{\mathrm{slack}}$ | Slack around threshold when opening candidates (makes threshold effectively $\tau - 1.0$) |
| `reassignment_margin_abs` | `float` | `5.0` | $\delta_{\mathrm{reassign}}$ | Margin for blame reassignment to previous row (in price points) |

### Anchor Construction Parameters

| Parameter | Type | Default | Mathematical Notation | Description |
|-----------|------|---------|----------------------|-------------|
| `use_unique_trailing_median` | `bool` | `True` | — | If `True`, use unique prices in trailing window (reduces duplicate-print bias) |

### Par-Specific Parameters

| Parameter | Type | Default | Mathematical Notation | Description |
|-----------|------|---------|----------------------|-------------|
| `par_spike_heuristic` | `bool` | `True` | — | Enable special handling for prices at or near par |
| `par_level` | `float` | `100.0` | $P_{\mathrm{par}}$ | Numerical par level (typically 100.0 for bonds) |
| `par_equal_tol` | `float` | `1e-8` | $\epsilon_{\mathrm{par}}$ | Tolerance for treating a price as exactly at par |
| `par_min_run` | `int` | `3` | $\ell_{\min}$ | Minimum contiguous par-only run length to flag (prevents spurious flags) |
| `par_cooldown_after_flag` | `int` | `2` | $C$ | Number of rows to skip after flagging a par block |

---

## Algorithm Logic (Step-by-Step)

### Step 1: Preprocessing
1. **Sort** DataFrame by `[id_col, date_col, time_col]` (if `time_col` is present)
2. Compute **one-step price changes**: `delta_rptd_pr = df.groupby(id_col)[price_col].diff()`
3. Compute **backward-looking anchor**: `baseline_trailing` using trailing median (window = $w+1$, shifted by 1)

### Step 2: Main Scan (Per Bond)
For each bond group (`id_col`), iterate through rows $i = 0, 1, \ldots, n-1$:

```
INITIALIZE:
  filtered = zeros array of length n
  par\_cooldown\_until = -1  (local index)

FOR i = 0 to n-1:

  # Check cooldown (skip non-par flags if within cooldown)
  IF (i <= par\_cooldown\_until) AND (P[i] is not at par):
      CONTINUE to next i

  # Candidate opening conditions
  cond\_jump     = |ΔP[i]| >= τ - δ\_slack
  cond\_far\_prev = |P[i] - B[i]| >= τ - δ\_slack
  cond\_par      = (|P[i] - P\_par| <= ε\_par) AND (|P[i] - B[i]| >= α·τ)

  par\_only = cond\_par AND NOT cond\_jump

  IF (cond\_jump OR cond\_far\_prev OR cond\_par):

      # Lookahead scan for bounce-back
      j\_match  = NULL  (opposite big move)
      k\_return = NULL  (return to anchor)

      IF NOT par\_only:
          FOR j = i+1 to min(i+L, n-1):
              # Path A: Opposite-signed large move
              IF sign(ΔP[j]) == -sign(ΔP[i]) AND |ΔP[j]| >= τ - δ\_slack:
                  j\_match = j
                  BREAK

              # Path B: Return to anchor
              IF |P[j] - B[i]| <= α·τ:
                  k\_return = j
                  BREAK

      # Resolution check
      IF (j\_match OR k\_return):
          j\_stop = j\_match if j\_match else k\_return
          flag\_start = i

          # Blame reassignment
          IF i-1 >= 0:
              dev\_prev = |P[i-1] - B[i-1]|
              dev\_curr = |P[i] - B[i]|
              IF (dev\_prev - dev\_curr >= δ\_reassign) AND (dev\_prev >= α·τ):
                  flag\_start = i-1

          # Flag the start row
          IF (NOT par\_start) OR (P[flag\_start] is at par):
              filtered[flag\_start] = 1

          # Extend plateau flags
          span\_end = min(j\_stop, flag\_start + S)
          FOR k = flag\_start+1 to span\_end:
              IF par\_start:
                  IF P[k] is at par:
                      filtered[k] = 1
              ELSE:
                  IF |P[k] - B[flag\_start]| >= α·τ:
                      filtered[k] = 1
                  ELSE:
                      BREAK

          # Par cooldown
          IF par\_start:
              par\_cooldown\_until = max(par\_cooldown\_until, j\_stop + C)

          i = j\_stop + 1
          CONTINUE

      # Persistent par block (no quick-correction found)
      IF par\_start:
          run\_end = i
          WHILE (run\_end+1 < n) AND (P[run\_end+1] is at par):
              run\_end += 1
          run\_len = run\_end - i + 1

          IF run\_len >= ℓ\_min:
              FOR k = i to run\_end:
                  filtered[k] = 1
              par\_cooldown\_until = max(par\_cooldown\_until, run\_end + C)
              i = run\_end + 1
              CONTINUE

  i += 1  (advance to next row)
```

### Step 3: Output
- Add column `filtered_error` (int8): `1` if flagged, `0` otherwise
- Return the DataFrame with added diagnostic columns (typically `delta_rptd_pr`, `baseline_trailing`, `filtered_error`)

---

## Examples

### Example 1: Basic Bounce-Back (Opposite Big Move)

**Input Data** (CUSIP = `11111A111`, date = `2024-02-20`):

| Row $i$ | Time | Price $P_i$ | $\Delta P_i$ | Baseline $B_i$ | Notes |
|---------|------|-------------|--------------|----------------|-------|
| 0 | 09:00:00 | 92.0 | — | — | |
| 1 | 09:15:00 | 93.5 | +1.5 | 92.0 | |
| 2 | 09:30:00 | 94.0 | +0.5 | 92.75 | |
| 3 | 09:45:00 | **165.0** | **+71.0** | 93.2 | **Error spike** |
| 4 | 10:00:00 | 168.0 | +3.0 | 93.2 | Plateau |
| 5 | 10:15:00 | **92.5** | **-75.5** | 93.2 | **Bounce-back** |
| 6 | 10:30:00 | 93.8 | +1.3 | 93.2 | Normal |

**Algorithm Execution**:

1. **Row 3 (Candidate Opening)**:
   - $|\Delta P_3| = |165.0 - 94.0| = 71.0 \geq \tau - \delta_{\mathrm{slack}} = 35.0 - 1.0 = 34.0$ ✓
   - $|P_3 - B_3| = |165.0 - 93.2| = 71.8 \geq 34.0$ ✓
   - **Candidate opened** at row 3

2. **Lookahead Scan** ($i=3$, $L=5$):
   - Row 4: $\Delta P_4 = +3.0$ (same sign as $\Delta P_3$, not opposite) ✗
   - Row 5: $\Delta P_5 = -75.5$ (opposite sign) ✓ AND $|\Delta P_5| = 75.5 \geq 34.0$ ✓
   - **Path A resolved** at $j_{\mathrm{match}} = 5$

3. **Blame Reassignment**:
   - $|P_2 - B_2| = |94.0 - 92.75| = 1.25$
   - $|P_3 - B_3| = 71.8$
   - $1.25 - 71.8 = -70.55 \not\geq 5.0$ ✗
   - **No reassignment** (row 3 is the error)

4. **Flagging**:
   - Flag row 3: `filtered[3] = 1`
   - **Plateau extension** (rows 4 to $\min(5, 3+5) = 5$):
     - Row 4: $|P_4 - B_3| = |168.0 - 93.2| = 74.8 \geq \alpha \cdot \tau = 0.25 \times 35.0 = 8.75$ ✓ → Flag
     - Row 5: $|P_5 - B_3| = |92.5 - 93.2| = 0.7 \not\geq 8.75$ ✗ → Stop
   - Final flags: rows 3, 4

**Output**:

| Row | Price | `filtered_error` | Notes |
|-----|-------|------------------|-------|
| 3 | 165.0 | **1** | Spike flagged |
| 4 | 168.0 | **1** | Plateau flagged |
| 5 | 92.5 | 0 | Bounce-back preserved (not flagged) |

---

### Example 2: Genuine Downgrade (No Bounce-Back)

**Input Data** (CUSIP = `22222B222`, credit downgrade):

| Row | Time | Price | $\Delta P$ | Baseline | Notes |
|-----|------|-------|------------|----------|-------|
| 0 | 09:00:00 | 90.0 | — | — | Pre-downgrade |
| 1 | 09:30:00 | 89.5 | -0.5 | 90.0 | |
| 2 | 10:00:00 | **52.0** | **-37.5** | 89.75 | **Downgrade announced** |
| 3 | 10:30:00 | 51.5 | -0.5 | 77.0 | Post-downgrade |
| 4 | 11:00:00 | 52.5 | +1.0 | 64.3 | Stabilizing |
| 5 | 11:30:00 | 52.2 | -0.3 | 57.7 | New level |

**Algorithm Execution**:

1. **Row 2 (Candidate Opening)**:
   - $|\Delta P_2| = 37.5 \geq 34.0$ ✓
   - $|P_2 - B_2| = |52.0 - 89.75| = 37.75 \geq 34.0$ ✓
   - **Candidate opened**

2. **Lookahead Scan** (rows 3-6):
   - Row 3: $\Delta P_3 = -0.5$ (same sign, not opposite) ✗
   - Row 4: $\Delta P_4 = +1.0$ (opposite sign) ✓, but $|\Delta P_4| = 1.0 \not\geq 34.0$ ✗
   - Row 5: $\Delta P_5 = -0.3$ (same sign) ✗
   - **No opposite big move found** (Path A fails)
   - **Check Path B**: $|P_3 - B_2| = |51.5 - 89.75| = 38.25 \not\leq 8.75$ ✗
   - ... all subsequent rows remain far from $B_2 = 89.75$ ...
   - **No return to anchor** (Path B fails)

3. **Decision**: **NO FLAG** (no bounce-back detected; genuine downgrade)

**Output**: All rows have `filtered_error = 0`

---

### Example 3: Par-Spike Heuristic (Short Par Run)

**Input Data** (CUSIP = `33333C333`, occasional par prints):

| Row | Time | Price | Baseline | Notes |
|-----|------|-------|----------|-------|
| 0 | 09:00:00 | 98.5 | — | |
| 1 | 09:30:00 | 99.2 | 98.5 | |
| 2 | 10:00:00 | **100.0** | 98.85 | **Par print** (isolated) |
| 3 | 10:30:00 | 99.1 | 98.9 | Reverts |
| 4 | 11:00:00 | 98.8 | 99.1 | Normal |

**Algorithm Execution**:

1. **Row 2 (Par Candidate)**:
   - $|P_2 - P_{\mathrm{par}}| = |100.0 - 100.0| = 0 \leq 10^{-8}$ ✓ (price is at par)
   - $|P_2 - B_2| = |100.0 - 98.85| = 1.15$
   - $\alpha \cdot \tau = 0.25 \times 35.0 = 8.75$
   - $1.15 \not\geq 8.75$ ✗
   - **Par candidate condition fails** (displacement too small)

2. **Decision**: **NO FLAG** (isolated par print is plausible, not displaced enough)

**Output**: All rows have `filtered_error = 0`

---

### Example 4: Persistent Par Block (Flagged)

**Input Data** (CUSIP = `44444D444`, erroneous par plateau):

| Row | Time | Price | Baseline | Notes |
|-----|------|-------|----------|-------|
| 0 | 09:00:00 | 85.0 | — | Normal |
| 1 | 09:30:00 | 84.5 | 85.0 | Normal |
| 2 | 10:00:00 | **100.0** | 84.75 | **Error: par spike** |
| 3 | 10:15:00 | **100.0** | 85.0 | Continues |
| 4 | 10:30:00 | **100.0** | 89.8 | Continues |
| 5 | 10:45:00 | 84.8 | 92.3 | Returns |

**Algorithm Execution**:

1. **Row 2 (Par Candidate)**:
   - Price is at par ✓
   - $|P_2 - B_2| = |100.0 - 84.75| = 15.25 \geq 8.75$ ✓
   - **Par candidate opened**

2. **Lookahead Scan**:
   - No opposite big move (Path A fails)
   - Row 5 returns to baseline: $|P_5 - B_2| = |84.8 - 84.75| = 0.05 \leq 8.75$ ✓
   - **Path B resolved** at $j_{\mathrm{return}} = 5$

3. **Flagging** (par-spike mode):
   - Rows 2, 3, 4 all satisfy $|P_k - 100.0| \leq 10^{-8}$ ✓
   - Flag rows 2, 3, 4
   - Row 5 is not at par → not flagged

4. **Cooldown**: Suppress non-par flags until row $5 + 2 = 7$ (next 2 rows after bounce-back)

**Output**:

| Row | Price | `filtered_error` |
|-----|-------|------------------|
| 2 | 100.0 | **1** |
| 3 | 100.0 | **1** |
| 4 | 100.0 | **1** |
| 5 | 84.8 | 0 |

---

### Example 5: Blame Reassignment

**Input Data** (CUSIP = `55555E555`, error on previous row):

| Row | Time | Price | Baseline | Notes |
|-----|------|-------|----------|-------|
| 0 | 09:00:00 | 78.0 | — | |
| 1 | 09:30:00 | 80.0 | 78.0 | Normal |
| 2 | 10:00:00 | **185.0** | 79.0 | **True error here** |
| 3 | 10:30:00 | 180.0 | 79.0 | Derivative error |
| 4 | 11:00:00 | **79.5** | 79.0 | Bounce-back |

**Algorithm Execution**:

1. **Row 3 (Candidate Opening)**:
   - $|\Delta P_3| = |180.0 - 185.0| = 5.0 \not\geq 34.0$ ✗ (no big jump at row 3)
   - But $|P_3 - B_3| = |180.0 - 79.0| = 101.0 \geq 34.0$ ✓ (displaced from baseline)
   - **Candidate opened** at row 3

2. **Lookahead Scan**:
   - Row 4: $\Delta P_4 = 79.5 - 180.0 = -100.5$ (opposite sign) ✓ AND $|\Delta P_4| = 100.5 \geq 34.0$ ✓
   - **Path A resolved** at row 4

3. **Blame Reassignment** (check row 2):
   - $|P_2 - B_2| = |185.0 - 79.0| = 106.0$
   - $|P_3 - B_3| = 101.0$
   - $106.0 - 101.0 = 5.0 \geq 5.0$ ✓
   - $106.0 \geq \alpha \cdot \tau = 8.75$ ✓
   - **Reassign to row 2** (it is more displaced)

4. **Flagging**: Rows 2, 3 flagged (row 2 is the true error, row 3 is the plateau)

**Output**:

| Row | Price | `filtered_error` | Notes |
|-----|-------|------------------|-------|
| 2 | 185.0 | **1** | True error (reassigned) |
| 3 | 180.0 | **1** | Plateau flagged |
| 4 | 79.5 | 0 | Bounce-back preserved |

---

## Default Configuration

From `stage0/_trace_settings.py`:

```python
BB_PARAMS = {
    "threshold_abs": 35.0,                  # 35 price points absolute threshold
    "lookahead": 5,                         # Search up to 5 rows ahead
    "max_span": 5,                          # Cap plateau extension at 5 rows
    "window": 5,                            # Trailing window for anchor
    "back_to_anchor_tol": 0.25,             # 25% of threshold (8.75 price points)
    "candidate_slack_abs": 1.0,             # Effective threshold = 34.0
    "reassignment_margin_abs": 5.0,         # 5 price points for blame shift
    "use_unique_trailing_median": True,
    "par_spike_heuristic": True,
    "par_level": 100.0,
    "par_equal_tol": 1e-8,                  # Numerical precision for par
    "par_min_run": 3,                       # Require ≥3 consecutive par prints
    "par_cooldown_after_flag": 2,           # Skip 2 rows after par flag
}
```

---

## Typical Usage in Pipeline

```python
from create_daily_standard_trace import flag_price_change_errors
from _trace_settings import BB_PARAMS

# Load cleaned TRACE data (after decimal shift correction)
df_clean = pd.read_parquet("trace_enhanced_20240115_cleaned.parquet")

# Apply bounce-back filter
df_flagged = flag_price_change_errors(
    df_clean,
    id_col="cusip_id",
    date_col="trd_exctn_dt",
    time_col="trd_exctn_tm",
    price_col="rptd_pr",
    **BB_PARAMS
)

# Remove flagged rows
df_final = df_flagged[df_flagged["filtered_error"] == 0].copy()

n_flagged = df_flagged["filtered_error"].sum()
affected_cusips = df_flagged.loc[df_flagged["filtered_error"] == 1, "cusip_id"].nunique()

print(f"Flagged {n_flagged:,} transactions across {affected_cusips:,} bonds")
# Output: Flagged 8,423 transactions across 892 bonds
```

---

## Design Rationale

### Why Backward-Looking Anchor?

1. **No look-ahead bias**: Can be used in real-time trading systems
2. **Robust to future volatility**: Does not assume prices stabilize after error
3. **Realistic baseline**: Reflects what a trader would see at the moment of the transaction

### Why Two Resolution Paths?

1. **Path A (opposite big move)**: Catches errors followed by corrections in opposite direction
2. **Path B (return to anchor)**: Catches errors that revert without large opposite move

### Why Par-Specific Logic?

Newly-issued bonds often trade tightly around par ($P = 100.0$) with occasional prints at exactly par:
- These can appear as "spikes" relative to bid-ask spread ($\pm 0.5$)
- But they are **not errors** if isolated
- The algorithm requires **persistent par runs** ($\geq 3$ consecutive) + displacement from baseline to flag

### Why Cooldown Period?

After flagging a par block, the baseline may be shifted by the flagged rows. Cooldown prevents:
- Cascading false positives in subsequent par-trading regions
- Over-flagging when prices legitimately oscillate around par

---

## Comparison with Decimal Shift Corrector

| Aspect | Decimal Shift Corrector | Bounce-Back Filter |
|--------|------------------------|--------------------|
| **Error Type** | Multiplicative (10x, 100x) | Additive (transient spikes) |
| **Anchor** | Rolling unique-median (centered) | Trailing unique-median (backward-looking) |
| **Detection Method** | Test specific factors {0.1, 10, 100} | Detect large jumps + reversion pattern |
| **Action** | **Correct** prices | **Flag** (remove) transactions |
| **Sequence** | Applied first (Stage 2) | Applied after decimal correction (Stage 7) |

---

## References

**Dickerson, A., Robotti, C., & Rossetti, G. (2024)**. "Common pitfalls in the evaluation of corporate bond strategies." Working Paper.

---

## See Also

- `README_decimal_shift_corrector.md` — Documentation for the decimal shift corrector
- `_trace_settings.py` — Default configuration parameters
- `create_daily_enhanced_trace.py` — Full pipeline implementation
- `FAQ.md` — Common questions about TRACE data cleaning
