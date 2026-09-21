# Default Handling in Bond Return Calculations

## Overview

This document describes the methodology for calculating bond returns when bonds enter default or trade under default. The standard bond return formula is adjusted to account for the unique characteristics of defaulted bonds.

## Rating Variables

We use the raw agency rating variables to identify default status:

| Rating Agency | Variable | Default Level |
|--------------|----------|---------------|
| S&P | `sp_rat` | 22 |
| Moody's | `mdy_rat` | 21 |

**A bond is considered in default if:**
- `sp_rat == 22` **OR** `mdy_rat == 21`

## Return Formulas

### Standard Return (Non-Default)

For bonds not in default, we use the standard total return formula:

$$r_{i,t+1}^{\text{standard}} = \frac{P_{i,t+1}^{\text{end}} + AI_{i,t+1}^{\text{end}} + C_{i,t+1}}{P_{i,t}^{\text{end}} + AI_{i,t}^{\text{end}}} - 1$$

where $P$ is the clean price, $AI$ is accrued interest, and $C$ is the coupon payment.

In the data, this corresponds to `ret_type = 'standard'`.

---

### Case 1: Default Event Return

**Condition:** Bond transitions INTO default at time $t+1$
- At $t$: Bond is NOT in default (`sp_rat != 22` AND `mdy_rat != 21`)
- At $t+1$: Bond IS in default (`sp_rat == 22` OR `mdy_rat == 21`)

**Rationale:** When a bond defaults, coupon payments cease. The return should reflect only the price change, comparing the clean price at default to the dirty price before default.

**Formula:**

$$r_{i,t+1}^{\text{default}} = \frac{P_{i,t+1}^{\text{end}}}{P_{i,t}^{\text{end}} + AI_{i,t}^{\text{end}}} - 1$$

In the data, this corresponds to `ret_type = 'default_evnt'`.

---

### Case 2: Trading Under Default (Flat Return)

**Condition:** Bond remains in default
- At $t$: Bond IS in default
- At $t+1$: Bond IS in default

**Rationale:** A defaulted bond does not pay coupons. Returns should be based solely on clean price changes, with no accrued interest component.

**Formula:**

$$r_{i,t+1}^{\text{flat}} = \frac{P_{i,t+1}^{\text{end}}}{P_{i,t}^{\text{end}}} - 1$$

In the data, this corresponds to `ret_type = 'trad_in_def'`.

**Additional constraint:** The trading-under-default return is not allowed to exceed the standard return. This affects a small number of monthly observations where the flat price change would otherwise exceed the total return calculation.

---

## Examples with Made-Up Data

### Example 1: Bond Enters Default

Consider Bond ABC with the following data:

| Month | P (Clean) | P + AI (Dirty) | sp_rat | mdy_rat |
|-------|-----------|----------------|--------|---------|
| $t$   | 98.50     | 100.25         | 15     | 14      |
| $t+1$ | 45.00     | 45.00          | 22     | 21      |

**Analysis:**
- Month $t$: Bond is NOT in default (sp_rat=15, mdy_rat=14)
- Month $t+1$: Bond IS in default (sp_rat=22 OR mdy_rat=21)
- This is a **Default Event** (`ret_type = 'default_evnt'`)

**Return Calculation:**
$$r_{t+1}^{\text{default}} = \frac{P_{t+1}^{\text{end}}}{P_{t}^{\text{end}} + AI_{t}^{\text{end}}} - 1 = \frac{45.00}{100.25} - 1 = -0.5511 = -55.11\%$$

---

### Example 2: Bond Trading Under Default

Consider Bond DEF that has been in default for several months:

| Month | P (Clean) | P + AI (Dirty) | sp_rat | mdy_rat |
|-------|-----------|----------------|--------|---------|
| $t$   | 35.00     | 35.00          | 22     | 21      |
| $t+1$ | 32.00     | 32.00          | 22     | 21      |
| $t+2$ | 38.00     | 38.00          | 22     | 21      |

**Analysis:**
- All months: Bond IS in default
- Each month represents **Trading Under Default** (`ret_type = 'trad_in_def'`)
- Note: Dirty price equals clean price because defaulted bonds don't accrue interest

**Return Calculations:**

**Month $t+1$ return:**
$$r_{t+1}^{\text{flat}} = \frac{P_{t+1}^{\text{end}}}{P_{t}^{\text{end}}} - 1 = \frac{32.00}{35.00} - 1 = -0.0857 = -8.57\%$$

**Month $t+2$ return:**
$$r_{t+2}^{\text{flat}} = \frac{P_{t+2}^{\text{end}}}{P_{t+1}^{\text{end}}} - 1 = \frac{38.00}{32.00} - 1 = 0.1875 = +18.75\%$$

---

### Example 3: Full Lifecycle of a Defaulting Bond

Consider Bond XYZ through a default cycle:

| Month | P | P + AI | sp_rat | mdy_rat | ret_type |
|-------|-----|--------|--------|---------|----------|
| $t$   | 95.00 | 96.50 | 10 | 9 | standard |
| $t+1$ | 92.00 | 94.00 | 10 | 9 | standard |
| $t+2$ | 40.00 | 40.00 | 22 | 21 | **default_evnt** |
| $t+3$ | 35.00 | 35.00 | 22 | 21 | trad_in_def |
| $t+4$ | 38.00 | 38.00 | 22 | 21 | trad_in_def |

**Return Calculations:**

1. **Month $t+1$ (Standard):**
   $$r = \frac{94.00}{96.50} - 1 = -0.0259 = -2.59\%$$

2. **Month $t+2$ (Default Event):**
   $$r^{\text{default}} = \frac{40.00}{94.00} - 1 = -0.5745 = -57.45\%$$

3. **Month $t+3$ (Trading Under Default):**
   $$r^{\text{flat}} = \frac{35.00}{40.00} - 1 = -0.1250 = -12.50\%$$

4. **Month $t+4$ (Trading Under Default):**
   $$r^{\text{flat}} = \frac{38.00}{35.00} - 1 = 0.0857 = +8.57\%$$

---

## Implementation Details

### Detection Logic

```python
# Current default status
in_default = (sp_rat == 22) | (mdy_rat == 21)

# Default event (transition INTO default)
is_default_event = (~in_default_lag) & (in_default)

# Trading under default (remains in default)
is_trading_under_default = (in_default_lag) & (in_default)
```

### Return Type Variable

The data includes a `ret_type` column indicating which formula was used:
- `'standard'` - Normal bond return formula
- `'default_evnt'` - Bond entered default this period
- `'trad_in_def'` - Bond remained in default

---

## Key Assumptions

1. **Coupon Cessation on Default:** When a bond defaults, coupon payments stop immediately. Accrued interest is assumed to be zero for defaulted bonds.

2. **Rating Data Quality:** The methodology assumes rating data is accurate and reflects the true default/non-default status of bonds.

3. **Return Cap for Trading Under Default:** The trading-under-default return is capped at the standard return to prevent cases where ignoring accrued interest would produce an artificially higher return.
