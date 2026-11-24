# Default handling in bond return calculations

## Overview

This document describes how bond returns are calculated when a bond enters default or trades under default. The standard bond return formula is adjusted for defaulted bonds.

## Rating variables

We use the raw agency ratings from the stage 1 daily panel, `sp_rating` and `mdy_rating`, to identify default status. Step 1 calls them `sp_rat` and `mdy_rat`, the names used below. Neither is a column of the published panel, which has the composite ratings `spc_rat` and `mdc_rat`.

| Rating Agency | Variable | Default Level |
|--------------|----------|---------------|
| S&P | `sp_rat` | 22 |
| Moody's | `mdy_rat` | 21 |

A bond is in default if `sp_rat == 22` or `mdy_rat == 21`.

## Return formulas

### Standard return (non-default)

For bonds not in default, we use the standard total return formula

$$r_{i,t+1}^{\text{standard}} = \frac{P_{i,t+1}^{\text{end}} + AI_{i,t+1}^{\text{end}} + C_{i,t+1}}{P_{i,t}^{\text{end}} + AI_{i,t}^{\text{end}}} - 1$$

where $P$ is the clean price, $AI$ is accrued interest, and $C$ is the coupon payment.

In the data, this is `ret_type = 'standard'`.

### Case 1: Default event return

This case applies when a bond moves into default at time $t+1$.
- At $t$ the bond is not in default (`sp_rat != 22` and `mdy_rat != 21`).
- At $t+1$ the bond is in default (`sp_rat == 22` or `mdy_rat == 21`).

When a bond defaults, coupon payments cease. The return should reflect only the price change, so it compares the clean price at default with the dirty price before default. The formula is

$$r_{i,t+1}^{\text{default}} = \frac{P_{i,t+1}^{\text{end}}}{P_{i,t}^{\text{end}} + AI_{i,t}^{\text{end}}} - 1$$

In the data, this is `ret_type = 'default_evnt'`.

### Case 2: Trading under default (flat return)

This case applies when a bond remains in default.
- At $t$ the bond is in default.
- At $t+1$ the bond is in default.

A defaulted bond does not pay coupons. Returns should be based only on clean price changes, with no accrued interest component. The formula is

$$r_{i,t+1}^{\text{flat}} = \frac{P_{i,t+1}^{\text{end}}}{P_{i,t}^{\text{end}}} - 1$$

In the data, this is `ret_type = 'trad_in_def'`.

The trading-under-default return is also not allowed to exceed the standard return. This affects a small number of monthly observations where the flat price change would otherwise exceed the total return calculation.

## Examples with made-up data

### Example 1: Bond enters default

Consider Bond ABC with the following data.

| Month | P (Clean) | P + AI (Dirty) | sp_rat | mdy_rat |
|-------|-----------|----------------|--------|---------|
| $t$   | 98.50     | 100.25         | 15     | 14      |
| $t+1$ | 45.00     | 45.00          | 22     | 21      |

- In month $t$ the bond is not in default (sp_rat=15, mdy_rat=14).
- In month $t+1$ the bond is in default (sp_rat=22 or mdy_rat=21).
- This is a default event (`ret_type = 'default_evnt'`).

The return is
$$r_{t+1}^{\text{default}} = \frac{P_{t+1}^{\text{end}}}{P_{t}^{\text{end}} + AI_{t}^{\text{end}}} - 1 = \frac{45.00}{100.25} - 1 = -0.5511 = -55.11\%$$

### Example 2: Bond trading under default

Consider Bond DEF, which has been in default for several months.

| Month | P (Clean) | P + AI (Dirty) | sp_rat | mdy_rat |
|-------|-----------|----------------|--------|---------|
| $t$   | 35.00     | 35.00          | 22     | 21      |
| $t+1$ | 32.00     | 32.00          | 22     | 21      |
| $t+2$ | 38.00     | 38.00          | 22     | 21      |

- The bond is in default in every month.
- Each month is trading under default (`ret_type = 'trad_in_def'`).
- The dirty price equals the clean price because defaulted bonds do not accrue interest.

The month $t+1$ return is
$$r_{t+1}^{\text{flat}} = \frac{P_{t+1}^{\text{end}}}{P_{t}^{\text{end}}} - 1 = \frac{32.00}{35.00} - 1 = -0.0857 = -8.57\%$$

The month $t+2$ return is
$$r_{t+2}^{\text{flat}} = \frac{P_{t+2}^{\text{end}}}{P_{t+1}^{\text{end}}} - 1 = \frac{38.00}{32.00} - 1 = 0.1875 = +18.75\%$$

### Example 3: Full lifecycle of a defaulting bond

Consider Bond XYZ through a default cycle.

| Month | P | P + AI | sp_rat | mdy_rat | ret_type |
|-------|-----|--------|--------|---------|----------|
| $t$   | 95.00 | 96.50 | 10 | 9 | standard |
| $t+1$ | 92.00 | 94.00 | 10 | 9 | standard |
| $t+2$ | 40.00 | 40.00 | 22 | 21 | default_evnt |
| $t+3$ | 35.00 | 35.00 | 22 | 21 | trad_in_def |
| $t+4$ | 38.00 | 38.00 | 22 | 21 | trad_in_def |

The returns are as follows.

1. Month $t+1$ (standard)
   $$r = \frac{94.00}{96.50} - 1 = -0.0259 = -2.59\%$$

2. Month $t+2$ (default event)
   $$r^{\text{default}} = \frac{40.00}{94.00} - 1 = -0.5745 = -57.45\%$$

3. Month $t+3$ (trading under default)
   $$r^{\text{flat}} = \frac{35.00}{40.00} - 1 = -0.1250 = -12.50\%$$

4. Month $t+4$ (trading under default)
   $$r^{\text{flat}} = \frac{38.00}{35.00} - 1 = 0.0857 = +8.57\%$$

## Implementation details

### Detection logic

```python
# Current default status
in_default = (sp_rat == 22) | (mdy_rat == 21)

# Default event (transition INTO default)
is_default_event = (~in_default_lag) & (in_default)

# Trading under default (remains in default)
is_trading_under_default = (in_default_lag) & (in_default)
```

### Return type variable

The data includes a `ret_type` column that says which formula was used.
- `'standard'` is the normal bond return formula.
- `'default_evnt'` means the bond entered default this period.
- `'trad_in_def'` means the bond remained in default.

## Key assumptions

1. When a bond defaults, coupon payments stop immediately. Accrued interest is assumed to be zero for defaulted bonds.

2. The method assumes the rating data is accurate and reflects the true default or non-default status of bonds.

3. The trading-under-default return is capped at the standard return. This prevents cases where ignoring accrued interest would produce an artificially higher return.
