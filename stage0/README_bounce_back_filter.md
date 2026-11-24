# Bounce-back filter: technical documentation

## Overview

The bounce-back filter (the function `flag_price_change_errors`) detects transient price-entry errors in corporate bond transaction data. These errors show up as large price spikes that quickly revert to a trailing baseline, which points to a data entry mistake and not a genuine market movement.

The algorithm finds error candidates with backward-looking anchors, lookahead windows and par-specific heuristics.

## Framework

### Problem statement

Given a time-ordered sequence of transaction prices $\{P_1, P_2, \ldots, P_n\}$ for a bond, identify the observations $P_i$ that are candidate errors. A candidate error has these features.

1. A large jump, $|\Delta P_i| = |P_i - P_{i-1}| \geq \tau$ (the price change exceeds the threshold $\tau$).
2. A quick reversion. Within a lookahead window of $L$ trades, the price returns partially toward a backward-looking anchor.
3. A displacement from the anchor. The price $P_i$ is significantly displaced from the trailing baseline $B_i$.

The algorithm must distinguish these errors from genuine price moves (credit downgrades, for example) that persist over multiple trades.

### Core algorithm components

#### 1. Backward-looking anchor (trailing baseline)

For each observation $i$, compute an anchor $B_i$ that is strictly backward-looking and uses only past data.

$$
B_i = \text{median}\left(\text{unique}(\{P_{i-w-1}, \ldots, P_{i-1}\})\right)
$$

where

- $w$ is the window size (default 5). The rolling window itself is $w+1$ wide (six
  trailing prints at the default), because the code calls
  `rolling_unique_median(s, window=window+1)`. The support is
  $\{P_{i-w-1}, \ldots, P_{i-1}\}$.
- unique() removes duplicate prices, which reduces the bias from repeated prints.
- $B_i$ is shifted by 1, so it does not include $P_i$ and has no look-ahead bias.

$B_i$ is computed from transactions before row $i$ only. The decision to flag row $i$ still looks at up to $L$ later trades (step 4), so in real time a flag comes up to $L$ trades late.

#### 2. Price change dynamics

The one-step price change (delta) is

$$
\Delta P_i = P_i - P_{i-1}
$$

and the displacement from the baseline is

$$
d_i = P_i - B_i
$$

#### 3. Candidate opening conditions

A row $i$ becomes a candidate for flagging if any one of the following conditions holds.

Condition 1 is a large jump relative to the previous trade.

$$
|\Delta P_i| \geq \tau - \delta_{\mathrm{slack}} \quad \text{(default: } \tau = 35.0, \delta_{\mathrm{slack}} = 1.0)
$$

Condition 2 is a large displacement from the baseline.

$$
|d_i| = |P_i - B_i| \geq \tau - \delta_{\mathrm{slack}}
$$

Condition 3 is the par-spike heuristic, if it is enabled. It applies if $|P_i - P_{\mathrm{par}}| \leq \epsilon_{\mathrm{par}}$ (the price is at par, default $P_{\mathrm{par}} = 100.0$, $\epsilon_{\mathrm{par}} = 10^{-8}$), and it then requires

$$
|P_i - B_i| \geq \alpha \cdot \tau \quad \text{(default: } \alpha = 0.25)
$$

#### 4. Bounce-back resolution (lookahead scan)

Once a candidate is opened at row $i$, scan forward up to $L$ rows (default $L = 5$) for evidence of reversion along one of two paths.

Path A is an opposite-signed large move. Find the first row $j > i$ such that

$$
\mathrm{sign}(\Delta P_j) = -\mathrm{sign}(\Delta P_i) \quad \text{AND} \quad |\Delta P_j| \geq \tau - \delta_{\mathrm{slack}}
$$

Path B is a return to the anchor. Find the first row $k > i$ such that

$$
|P_k - B_i| \leq \alpha \cdot \tau
$$

If either Path A (row $j$) or Path B (row $k$) is found, the algorithm moves on to the flagging logic. Otherwise the candidate is rejected, because no bounce-back was detected.

#### 5. Flagging logic and plateau extension

Once a bounce-back is detected (resolved at row $j_{\mathrm{stop}}$), the flagging runs in steps.

Step 1 is reassignment (blame attribution). Check whether the previous row ($i-1$) is more displaced than the current row ($i$), that is, whether

$$
|P_{i-1} - B_{i-1}| - |P_i - B_i| \geq \delta_{\mathrm{reassign}} \quad \text{(default: } \delta_{\mathrm{reassign}} = 5.0)
$$

and

$$
|P_{i-1} - B_{i-1}| \geq \alpha \cdot \tau
$$

If both hold, reassign the flag to row $i-1$, which was the true error.

Step 2 flags the start. Flag the identified error row (either $i$ or $i-1$).

Step 3 extends the plateau flags. Flag additional rows in the range $[i_{\mathrm{start}}+1, \min(j_{\mathrm{stop}}, i_{\mathrm{start}} + S)]$ if they remain displaced from the baseline. For each row $k$ in this range

- in the par-spike case, flag it if $|P_k - P_{\mathrm{par}}| \leq \epsilon_{\mathrm{par}}$
- in the non-par case, flag it if $|P_k - B_{i_{\mathrm{start}}}| \geq \alpha \cdot \tau$, and stop at the first row that fails this test

Here $S$ is the max span (default 5).

#### 6. Par-specific heuristic (persistent par blocks)

Bonds trading at par get special handling, so that legitimate par prints do not become false positives.

A persistent par block is detected when a sequence of trades $\{P_i, P_{i+1}, \ldots, P_j\}$ all satisfy $|P_k - P_{\mathrm{par}}| \leq \epsilon_{\mathrm{par}}$. For such a sequence

- compute the run length $\ell = j - i + 1$
- flag it only if $\ell \geq \ell_{\min}$ (default $\ell_{\min} = 3$)

A cooldown period follows. After flagging a par block ending at row $j$, suppress further flags for the next $C$ rows (default $C = 2$).

$$
\text{No non-par flags issued for rows } j+1, j+2, \ldots, j+C
$$

This prevents cascading false positives in par-trading regions.

## Function signature

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

## Parameters

### Input data parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | `pd.DataFrame` | (required) | Input panel with intraday transaction data |
| `id_col` | `str` | `"cusip_id"` | Column name for bond identifier |
| `date_col` | `str` | `"trd_exctn_dt"` | Column name for trade execution date. Accepted and never read (the caller sorts, see Step 1) |
| `time_col` | `str \| None` | `"trd_exctn_tm"` | Column name for trade execution time. Accepted and never read (the caller sorts, see Step 1) |
| `price_col` | `str` | `"rptd_pr"` | Column name for reported price |

### Core detection parameters

| Parameter | Type | Default | Mathematical notation | Description |
|-----------|------|---------|----------------------|-------------|
| `threshold_abs` | `float` | `35.0` | $\tau$ | Minimum absolute price change (in price points) to open a candidate |
| `lookahead` | `int` | `5` | $L$ | Maximum number of rows ahead to search for bounce-back |
| `max_span` | `int` | `5` | $S$ | Maximum total path length from candidate to resolution (caps plateau extension) |
| `window` | `int` | `5` | $w$ | Backward window length for trailing median anchor |
| `back_to_anchor_tol` | `float` | `0.25` | $\alpha$ | Fraction of threshold. Prices within $\alpha \cdot \tau$ of the anchor count as "returned" |
| `candidate_slack_abs` | `float` | `1.0` | $\delta_{\mathrm{slack}}$ | Slack around threshold when opening candidates (makes threshold effectively $\tau - 1.0$) |
| `reassignment_margin_abs` | `float` | `5.0` | $\delta_{\mathrm{reassign}}$ | Margin for blame reassignment to previous row (in price points) |

### Anchor construction parameters

| Parameter | Type | Default | Mathematical notation | Description |
|-----------|------|---------|----------------------|-------------|
| `use_unique_trailing_median` | `bool` | `True` | none | If `True`, use unique prices in trailing window (reduces duplicate-print bias) |

### Par-specific parameters

| Parameter | Type | Default | Mathematical notation | Description |
|-----------|------|---------|----------------------|-------------|
| `par_spike_heuristic` | `bool` | `True` | none | Enable special handling for prices at or near par |
| `par_level` | `float` | `100.0` | $P_{\mathrm{par}}$ | Numerical par level (typically 100.0 for bonds) |
| `par_equal_tol` | `float` | `1e-8` | $\epsilon_{\mathrm{par}}$ | Tolerance for treating a price as exactly at par |
| `par_min_run` | `int` | `3` | $\ell_{\min}$ | Minimum contiguous par-only run length to flag (prevents spurious flags) |
| `par_cooldown_after_flag` | `int` | `2` | $C$ | Number of rows to skip after flagging a par block |

## Algorithm logic (step-by-step)

### Step 1: Preprocessing

**The function does not sort**, and it ignores `date_col` and `time_col` entirely.
Both appear in the signature but never in the body. The only columns it reads are
`id_col` and `price_col`. Row order is whatever the caller supplies, and the result
depends entirely on that order, so `clean_trace_data` sorts immediately before calling it
(the "Pre BB sort"). Enhanced is sorted by
`[cusip_id, trd_exctn_dt, trd_exctn_tm, trd_rpt_dt, trd_rpt_tm, msg_seq_nb]`, Standard and
144A by `[cusip_id, trd_exctn_dt, trd_exctn_tm, msg_seq_nb]`. An unsorted frame gives
wrong flags without an error.

1. Copy the input. There is no sort and no `reset_index`.
2. Compute the one-step price changes with `delta_rptd_pr = df.groupby(id_col)[price_col].diff()`.
3. Compute the backward-looking anchor `baseline_trailing` as a trailing median (window = $w+1$, shifted by 1).

### Step 2: Main scan (per bond)

For each bond group (`id_col`), iterate through rows $i = 0, 1, \ldots, n-1$.

```
INITIALIZE:
  filtered = zeros array of length n
  par_cooldown_until = -1  (local index)

FOR i = 0 to n-1:

  # Check cooldown (skip non-par flags if within cooldown)
  IF (i <= par_cooldown_until) AND (P[i] is not at par):
      CONTINUE to next i

  # Candidate opening conditions
  cond_jump     = |ΔP[i]| >= τ - δ_slack
  cond_far_prev = |P[i] - B[i]| >= τ - δ_slack
  cond_par      = (|P[i] - P_par| <= ε_par) AND (|P[i] - B[i]| >= α·τ)

  par_only = cond_par AND NOT cond_jump
  par_start = cond_par   (the candidate opened at par)

  IF (cond_jump OR cond_far_prev OR cond_par):

      # Lookahead scan for bounce-back
      j_match  = NULL  (opposite big move)
      k_return = NULL  (return to anchor)

      IF NOT par_only:
          FOR j = i+1 to min(i+L, n-1):
              # Path A: Opposite-signed large move
              IF sign(ΔP[j]) == -sign(ΔP[i]) AND |ΔP[j]| >= τ - δ_slack:
                  j_match = j
                  BREAK

              # Path B: Return to anchor
              IF |P[j] - B[i]| <= α·τ:
                  k_return = j
                  BREAK

      # Resolution check
      IF (j_match OR k_return):
          j_stop = j_match if j_match else k_return
          flag_start = i

          # Blame reassignment
          IF i-1 >= 0:
              dev_prev = |P[i-1] - B[i-1]|
              dev_curr = |P[i] - B[i]|
              IF (dev_prev - dev_curr >= δ_reassign) AND (dev_prev >= α·τ):
                  flag_start = i-1

          # Flag the start row
          IF (NOT par_start) OR (P[flag_start] is at par):
              filtered[flag_start] = 1

          # Extend plateau flags
          span_end = min(j_stop, flag_start + S)
          FOR k = flag_start+1 to span_end:
              IF par_start:
                  IF P[k] is at par:
                      filtered[k] = 1
              ELSE:
                  IF |P[k] - B[flag_start]| >= α·τ:
                      filtered[k] = 1
                  ELSE:
                      BREAK

          # Par cooldown
          IF par_start:
              par_cooldown_until = max(par_cooldown_until, j_stop + C)

          i = j_stop + 1
          CONTINUE

      # Persistent par block (no quick-correction found)
      IF par_start:
          run_end = i
          WHILE (run_end+1 < n) AND (P[run_end+1] is at par):
              run_end += 1
          run_len = run_end - i + 1

          IF run_len >= ℓ_min:
              FOR k = i to run_end:
                  filtered[k] = 1
              par_cooldown_until = max(par_cooldown_until, run_end + C)
              i = run_end + 1
              CONTINUE

  i += 1  (advance to next row)
```

### Step 3: Output

- Add the column `filtered_error` (int8), which is `1` if the row is flagged and `0` otherwise.
- Return the DataFrame with the added diagnostic columns (typically `delta_rptd_pr`, `baseline_trailing`, `filtered_error`).

## Examples

### Example 1: Basic bounce-back (opposite big move)

The input is CUSIP `11111A111` on `2024-02-20`.

| Row $i$ | Time | Price $P_i$ | $\Delta P_i$ | Baseline $B_i$ | Notes |
|---------|------|-------------|--------------|----------------|-------|
| 0 | 09:00:00 | 92.0 | n/a | n/a | |
| 1 | 09:15:00 | 93.5 | +1.5 | 92.0 | |
| 2 | 09:30:00 | 94.0 | +0.5 | 92.75 | |
| 3 | 09:45:00 | 165.0 | +71.0 | 93.5 | Error spike |
| 4 | 10:00:00 | 168.0 | +3.0 | 93.75 | Plateau |
| 5 | 10:15:00 | 92.5 | -75.5 | 94.0 | Bounce-back |
| 6 | 10:30:00 | 93.8 | +1.3 | 93.75 | Normal |

The algorithm runs as follows.

1. Row 3 is tested for candidate opening.
   - $|\Delta P_3| = |165.0 - 94.0| = 71.0 \geq \tau - \delta_{\mathrm{slack}} = 35.0 - 1.0 = 34.0$, which holds.
   - $|P_3 - B_3| = |165.0 - 93.5| = 71.5 \geq 34.0$, which holds.
   - A candidate is opened at row 3.

2. The lookahead scan starts from $i=3$ with $L=5$.
   - Row 4 has $\Delta P_4 = +3.0$ (same sign as $\Delta P_3$, not opposite), so it fails.
   - Row 5 has $\Delta P_5 = -75.5$ (opposite sign) and $|\Delta P_5| = 75.5 \geq 34.0$, so both tests hold.
   - Path A is resolved at $j_{\mathrm{match}} = 5$.

3. Blame reassignment is tested.
   - $|P_2 - B_2| = |94.0 - 92.75| = 1.25$
   - $|P_3 - B_3| = 71.5$
   - $1.25 - 71.5 = -70.25 \not\geq 5.0$, so the test fails.
   - There is no reassignment (row 3 is the error).

4. The rows are flagged.
   - Row 3 is flagged with `filtered[3] = 1`.
   - The plateau extension covers rows 4 to $\min(5, 3+5) = 5$.
     - Row 4 has $|P_4 - B_3| = |168.0 - 93.5| = 74.5 \geq \alpha \cdot \tau = 0.25 \times 35.0 = 8.75$, so it is flagged.
     - Row 5 has $|P_5 - B_3| = |92.5 - 93.5| = 1.0 \not\geq 8.75$, so the extension stops.
   - The final flags are rows 3 and 4.

The output is as follows.

| Row | Price | `filtered_error` | Notes |
|-----|-------|------------------|-------|
| 3 | 165.0 | 1 | Spike flagged |
| 4 | 168.0 | 1 | Plateau flagged |
| 5 | 92.5 | 0 | Bounce-back preserved (not flagged) |

### Example 2: Genuine downgrade (no bounce-back)

The input is CUSIP `22222B222`, with a credit downgrade.

| Row | Time | Price | $\Delta P$ | Baseline | Notes |
|-----|------|-------|------------|----------|-------|
| 0 | 09:00:00 | 90.0 | n/a | n/a | Pre-downgrade |
| 1 | 09:30:00 | 89.5 | -0.5 | 90.0 | |
| 2 | 10:00:00 | 52.0 | -37.5 | 89.75 | Downgrade announced |
| 3 | 10:30:00 | 51.5 | -0.5 | 89.50 | Post-downgrade |
| 4 | 11:00:00 | 52.5 | +1.0 | 70.75 | Stabilizing |
| 5 | 11:30:00 | 52.2 | -0.3 | 52.50 | New level |

The algorithm runs as follows.

1. Row 2 is tested for candidate opening.
   - $|\Delta P_2| = 37.5 \geq 34.0$, which holds.
   - $|P_2 - B_2| = |52.0 - 89.75| = 37.75 \geq 34.0$, which holds.
   - The candidate is opened.

2. The lookahead scan covers rows 3 to 5.
   - Row 3 has $\Delta P_3 = -0.5$ (same sign, not opposite), so it fails.
   - Row 4 has $\Delta P_4 = +1.0$ (opposite sign), but $|\Delta P_4| = 1.0 \not\geq 34.0$, so it fails.
   - Row 5 has $\Delta P_5 = -0.3$ (same sign), so it fails.
   - No opposite big move is found, so Path A fails.
   - Path B is checked next. $|P_3 - B_2| = |51.5 - 89.75| = 38.25 \not\leq 8.75$, so the test fails.
   - All subsequent rows remain far from $B_2 = 89.75$.
   - There is no return to the anchor, so Path B fails.

3. The decision is no flag. No bounce-back is detected, and the move is a genuine downgrade.

In the output, all rows have `filtered_error = 0`.

### Example 3: Par-spike heuristic (short par run)

The input is CUSIP `33333C333`, with occasional par prints.

| Row | Time | Price | Baseline | Notes |
|-----|------|-------|----------|-------|
| 0 | 09:00:00 | 98.5 | n/a | |
| 1 | 09:30:00 | 99.2 | 98.5 | |
| 2 | 10:00:00 | 100.0 | 98.85 | Par print (isolated) |
| 3 | 10:30:00 | 99.1 | 99.20 | Reverts |
| 4 | 11:00:00 | 98.8 | 99.15 | Normal |

The algorithm runs as follows.

1. Row 2 is tested as a par candidate.
   - $|P_2 - P_{\mathrm{par}}| = |100.0 - 100.0| = 0 \leq 10^{-8}$, which holds (the price is at par).
   - $|P_2 - B_2| = |100.0 - 98.85| = 1.15$
   - $\alpha \cdot \tau = 0.25 \times 35.0 = 8.75$
   - $1.15 \not\geq 8.75$, so the test fails.
   - The par candidate condition fails, because the displacement is too small.

2. The decision is no flag. An isolated par print is plausible, and this one is not displaced enough.

In the output, all rows have `filtered_error = 0`.

### Example 4: Persistent par block (flagged)

The input is CUSIP `44444D444`, with an erroneous par plateau.

| Row | Time | Price | Baseline | Notes |
|-----|------|-------|----------|-------|
| 0 | 09:00:00 | 85.0 | n/a | Normal |
| 1 | 09:30:00 | 84.5 | 85.00 | Normal |
| 2 | 10:00:00 | 100.0 | 84.75 | Error, a par spike |
| 3 | 10:15:00 | 100.0 | 85.00 | Continues |
| 4 | 10:30:00 | 100.0 | 85.00 | Continues |
| 5 | 10:45:00 | 84.8 | 85.00 | Returns |

The anchors were verified by running the function. They stay at 85.00 because the median is
taken over unique prices, and $\{85.0, 84.5, 100.0\}$ has median 85.0 however often a price
repeats.

The algorithm runs as follows.

1. Row 2 opens as a par-only candidate.
   - $|\Delta P_2| = |100.0 - 84.5| = 15.5$, which is below $\tau = 35$, so the
     jump condition does not fire.
   - The price is at par and displaced from the anchor, so the par condition does.
   - Hence `par_only = True`.

2. The lookahead is skipped. The scan is guarded by `if not par_only:`, so a par-only
   candidate never takes the quick-correction paths, and neither $j_{\mathrm{match}}$
   nor $k_{\mathrm{return}}$ is ever computed. Only a persistent par run can flag here.

3. The persistent par-run branch applies. Rows 2, 3 and 4 are all within `par_equal_tol`
   of 100.0, so the run length is 3, which meets `par_min_run = 3`, and rows 2, 3 and 4
   are flagged. Row 5 is not at par and is not flagged.

4. The cooldown suppresses non-par flags until `run_end + par_cooldown_after_flag`
   $= 4 + 2 = 6$.

The output is as follows.

| Row | Price | `filtered_error` |
|-----|-------|------------------|
| 2 | 100.0 | 1 |
| 3 | 100.0 | 1 |
| 4 | 100.0 | 1 |
| 5 | 84.8 | 0 |

### Example 5: Spike, plateau, bounce-back

The input is CUSIP `55555E555`, a two-print spike that snaps back.

| Row | Time | Price | Baseline | Notes |
|-----|------|-------|----------|-------|
| 0 | 09:00:00 | 78.0 | n/a | |
| 1 | 09:30:00 | 80.0 | 78.0 | Normal |
| 2 | 10:00:00 | 185.0 | 79.0 | True error here |
| 3 | 10:30:00 | 180.0 | 80.0 | Derivative error |
| 4 | 11:00:00 | 79.5 | 130.0 | Bounce-back |

The anchors were verified by running the function.

The algorithm runs as follows.

1. Row 2 opens the candidate. The scan reaches it first.
   - $|\Delta P_2| = |185.0 - 80.0| = 105.0 \geq \tau = 35$, which holds.
   - The candidate is opened at row 2, the row that has the bad print.

   Row 3 is never evaluated as a candidate. Its own $|\Delta P_3| = 5.0$ is far below
   $\tau$, and after row 2 resolves, the cursor jumps past it.

2. The lookahead runs from row 2. Row 4 gives $\Delta P_4 = -100.5$, opposite in sign and
   large, so Path A resolves at row 4.

3. Blame reassignment is tested and declined.
   - $\mathrm{dev}_{\mathrm{prev}} = |P_1 - B_1| = |80.0 - 78.0| = 2.0$
   - $\mathrm{dev}_{\mathrm{curr}} = |P_2 - B_2| = |185.0 - 79.0| = 106.0$
   - Reassignment needs $\mathrm{dev}_{\mathrm{prev}} - \mathrm{dev}_{\mathrm{curr}}
     \geq 5.0$. Here it is $-104.0$, so there is no reassignment.

   That is the right outcome, because row 2 is already the culprit. There is currently no
   worked example in which the reassignment branch fires.

4. The flags go to row 2 as the candidate start and to row 3 as the plateau that has not yet
   returned to the anchor. Row 4 is the bounce-back and is preserved.

The output is as follows.

| Row | Price | `filtered_error` | Notes |
|-----|-------|------------------|-------|
| 2 | 185.0 | 1 | True error (candidate start) |
| 3 | 180.0 | 1 | Plateau flagged |
| 4 | 79.5 | 0 | Bounce-back preserved |

## Default configuration

The defaults are in `stage0/_trace_settings.py`.

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

## Typical usage in pipeline

Stage 0 calls this inside each chunk and writes no trade-level file, so the input below is
a trade-level frame you hold yourself (the file name is illustrative).

```python
import pandas as pd
from create_daily_standard_trace import flag_price_change_errors
from _trace_settings import BB_PARAMS

# Trade-level TRACE data, after the decimal-shift correction (illustrative file name)
df_clean = pd.read_parquet("my_trades_after_decimal_shift.parquet")

# The function does not sort, and its result depends on row order
df_clean = df_clean.sort_values(
    ["cusip_id", "trd_exctn_dt", "trd_exctn_tm", "msg_seq_nb"]).reset_index(drop=True)

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
# On the full Enhanced sample (2026-09-10 run) this came to 8,877 transactions across
# 2,901 bonds, summed over the run's 485 chunks.
```

## Design rationale

### Why backward-looking anchor?

1. It uses no later prices. The flag itself still waits for up to $L$ later trades (step 4).
2. Future volatility does not affect it, because it does not assume that prices stabilize after an error.
3. It is a realistic baseline, which reflects what a trader would see at the moment of the transaction.

### Why two resolution paths?

1. Path A (opposite big move) catches errors followed by corrections in the opposite direction.
2. Path B (return to anchor) catches errors that revert without a large opposite move.

### Why par-specific logic?

Newly issued bonds often trade tightly around par ($P = 100.0$), with occasional prints at exactly par. These prints are not errors if isolated. To flag, the algorithm requires a persistent par run ($\geq 3$ consecutive) and displacement from the baseline.

### Why cooldown period?

After a par block is flagged, the baseline may be shifted by the flagged rows. The cooldown prevents cascading false positives in subsequent par-trading regions, and over-flagging when prices legitimately oscillate around par.

## Comparison with decimal shift corrector

| Aspect | Decimal shift corrector | Bounce-back filter |
|--------|------------------------|--------------------|
| Error type | Multiplicative (10x, 100x) | Additive (transient spikes) |
| Anchor | Median of the de-duplicated prices (centered) | Trailing unique-median (backward-looking) |
| Detection method | Test specific factors {0.1, 0.01, 10, 100} | Detect large jumps and a reversion pattern |
| Action | Correct prices | Flag (remove) transactions |
| Sequence | Applied first (Filter 2) | Applied after decimal correction (Filter 7) |

## References

Dickerson, A., Robotti, C., & Rossetti, G. (2026). "The Corporate Bond Factor Replication Crisis." Working Paper. Earlier versions circulated as "Common pitfalls in the evaluation of corporate bond strategies."

## See also

- `README_decimal_shift_corrector.md` documents the decimal shift corrector.
- `_trace_settings.py` holds the default configuration parameters.
- `create_daily_enhanced_trace.py` is the full pipeline implementation.
- [FAQ.md](../FAQ.md) answers common questions about TRACE data cleaning.
