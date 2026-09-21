# Value Signals for Corporate Bonds

This note documents the construction of corporate-bond **value signals** implemented in `compute_value()`.
The function produces a monthly panel of value signals (and optional risk overlays) that can be used to form
sorted portfolios (e.g., deciles) or as characteristics in cross-sectional asset pricing tests.

> **What the shipped panel actually contains.** This document describes the
> `compute_value()` function, which is more general than the panel that is published from
> it. `main_panel_<YYYY>.parquet` carries exactly four value columns:
>
> | column | model | overlay |
> |---|---|---|
> | `val_hz` | Houweling-van Zundert | none |
> | `val_hz_dts` | Houweling-van Zundert | duration-times-spread |
> | `val_ipr` | Israel-Palhares-Richardson | none |
> | `val_ipr_dts` | Israel-Palhares-Richardson | duration-times-spread |
>
> The **firm overlay** (`_wi`) described below is computed by the function but not kept in
> the published panel. See `DATA_DICTIONARY.md` for the panel's definitions.

## What `compute_value()` does

Given a bond-level panel with identifiers `i` (CUSIP) and months `t`, `compute_value()`:

1. Runs a **monthly cross-sectional regression** of spreads on controls.
2. Uses the fitted values to form a **value signal** (either a scaled deviation from “fair” spread, or the residual).
3. Optionally applies two overlays (activated when `dur_col` is supplied):
   - **DtS overlay (`_dts`)**: demean within month × quintile of duration-times-spread.
   - **Firm overlay (`_wi`)**: demean within month × issuing firm (single-issue firms are left unchanged).

The output columns are named by `model_type`, e.g. `val_hz`, `val_hz_dts`, `val_hz_wi`.

---

## Inputs and notation

For bond `i` in month `t`:

- `cs_{i,t}`: credit spread (column `cs_col`).
- `rating_{i,t}`: rating measure (column `rating_col`, default `spc_rat`).
- `z_{i,t}`: every other regressor, passed as the list `x_cols` (at least one is required),
  e.g. `dcs3` (3-month change in spread), `call`, `log_md_dur`, `vol12_x`.
- `ind_{i,t}`: industry group (column `industry_col`, e.g., FF17 industry code).
- `dur_{i,t}`: modified duration (column `dur_col`, used only for overlays).

Let `x_{i,t}` denote the full vector of regressors used at month `t` (including an intercept).

---

## Step 1: Monthly cross-sectional regression

### Baseline spread regression in levels

When `y_transform="level"`, for each month `t` we estimate:

\[
cs_{i,t} = x_{i,t}^\top \beta_t + \varepsilon_{i,t}.
\]

The regressor vector `x_{i,t}` is composed of:

- Rating controls, either:
  - **dummies** (`rating_mode="dummies"`): one-hot indicators for each rating (one base category omitted), or
  - **numeric** (`rating_mode="numeric"`): a single numeric rating regressor.
- Optional **industry dummies** (e.g., FF17) when `industry_col` is provided.
- The regressors `z_{i,t}` listed in `x_cols`. Nothing else enters, so a maturity control
  is present only if `x_cols` names one.

We run OLS each month (separately for each cross-section). The fitted “fair” spread is:

\[
\widehat{cs}_{i,t} = x_{i,t}^\top \widehat{\beta}_t.
\]

### Spread regression in logs (recommended when spreads are heteroskedastic)

When `y_transform="log"`, we estimate:

\[
\log(cs_{i,t}) = x_{i,t}^\top \beta_t + \varepsilon_{i,t},
\qquad cs_{i,t} > 0.
\]

This often stabilizes variance and reduces the influence of extreme spreads.

#### Retransformation to levels

To obtain a fitted spread in levels, we use a retransformation.
With `retransform="lognormal"` (default), we apply the standard lognormal correction:

\[
\widehat{cs}_{i,t} =
\exp\left(x_{i,t}^\top \widehat{\beta}_t + \tfrac{1}{2}\widehat{\sigma}_t^2\right),
\qquad
\widehat{\sigma}_t^2 = \frac{1}{N_t}\sum_{i\in\mathcal{I}_t} \widehat{\varepsilon}_{i,t}^2.
\]

This corrects the bias from Jensen’s inequality when exponentiating fitted log values.

As an alternative, `retransform="smearing"` implements Duan’s (1983) “smearing” estimator:

\[
\widehat{cs}_{i,t} = \exp(x_{i,t}^\top \widehat{\beta}_t)\cdot \widehat{m}_t,
\qquad
\widehat{m}_t = \frac{1}{N_t}\sum_{i\in\mathcal{I}_t} \exp(\widehat{\varepsilon}_{i,t}).
\]

**Citation:** Duan (1983), *JASA*, “Smearing estimate: A nonparametric retransformation method.”

---

## Step 2: Value signal definitions

`compute_value()` supports three ways to define the monthly value signal, controlled by `denom`:

### A) Scaled deviation from fitted spread (`denom="fitted"`)

\[
val_{i,t} = \frac{cs_{i,t}-\widehat{cs}_{i,t}}{\widehat{cs}_{i,t}}.
\]

Interpretation: the percentage deviation of the observed spread from “fair” spread.
A higher value implies a wider spread relative to fundamentals (a “cheaper” bond in spread space).

### B) Scaled deviation from actual spread (`denom="actual"`)

\[
val_{i,t} = \frac{cs_{i,t}-\widehat{cs}_{i,t}}{cs_{i,t}}.
\]

Interpretation: deviation scaled by the bond’s own spread.

### C) Regression residual (`denom="resid"`)

- If `y_transform="level"`:

\[
val_{i,t} = \varepsilon_{i,t} = cs_{i,t}-\widehat{cs}_{i,t}.
\]

- If `y_transform="log"`:

\[
val_{i,t} = \varepsilon_{i,t} = \log(cs_{i,t}) - x_{i,t}^\top\widehat{\beta}_t.
\]

Interpretation: the unexplained component of spreads after controlling for `x_{i,t}`.

---

## Step 3: DtS overlay (`_dts`)

When `dur_col` is provided, we compute **duration-times-spread**:

\[
DtS_{i,t} = dur_{i,t}\cdot cs_{i,t}.
\]

Each month, assign each bond to a DtS quintile:

\[
Q_{i,t} \in \{1,2,3,4,5\}
\quad \text{based on the cross-sectional quintiles of } DtS_{i,t}.
\]

Compute the equally-weighted mean value signal within each (month, quintile):

\[
\bar{val}_{t,q} = \frac{1}{N_{t,q}}\sum_{i:Q_{i,t}=q} val_{i,t}.
\]

Then the **DtS-demeaned** signal is:

\[
val^{dts}_{i,t} = val_{i,t} - \bar{val}_{t,Q_{i,t}}.
\]

**Intuition:** removes components of the signal that are common to bonds with similar “spread duration exposure” within the month.

Output column name: `val_{model_type}_dts` by default.

---

## Step 4: Firm overlay (`_wi`)

When `dur_col` is provided, the function also computes a **within-firm demeaned** signal.
Let `f(i,t)` denote the issuing firm for bond `i` in month `t`.

- If `firm_col` is a column (e.g., `permco`, `gvkey`, `issuer_id`), we use it directly.
- If `firm_col="issuer_cusip"`, we define:

\[
firm_{i,t} = \text{first 6 characters of CUSIP}_{i,t}.
\]

Within each (month, firm), compute the mean:

\[
\bar{val}_{f,t} = \frac{1}{N_{f,t}}\sum_{i \in f} val_{i,t}.
\]

Define the firm-level adjusted signal as:

\[
val^{wi}_{i,t} =
\begin{cases}
val_{i,t} - \bar{val}_{f,t}, & N_{f,t}\ge 2\\
val_{i,t}, & N_{f,t}=1
\end{cases}
\]

So **single-issue firms are left unchanged**.

**Intuition:** isolates within-issuer relative value (bond-to-bond cheapness) by removing issuer-level common components.

Output column name: `val_{model_type}_wi` by default (`suffix_firm`).

---

## Model-specific implementations

### 1) Houweling & van Zundert (2017) value signal (`model_type="hz"`)

**Reference:** Houweling and van Zundert (2017), “Factor Investing in the Corporate Bond Market,” *Financial Analysts Journal*, 73(2).

The production call regresses log spreads on:
- rating dummies (`rating_col` left at its default, `spc_rat`),
- FF17 industry dummies (`industry_col="ff17num"`),
- the 3-month spread change and a call dummy (`x_cols=["dcs3", "call"]`), the call dummy
  because spreads are **unadjusted** for embedded options,

with lognormal retransformation (`y_transform="log"`, `retransform="lognormal"`). There is
no maturity control.

The call as `lib/value.py` makes it (the month-end version; the adjusted version passes
`cs_adj`, `dcs3_adj` and `md_dur_adj`):

```python
val_end = compute_value(
    end,
    id_col="cusip",
    date_col="date",
    model_type="hz",
    cs_col="cs",
    industry_col="ff17num",
    x_cols=["dcs3", "call"],
    dur_col="md_dur",
    firm_col="issuer_cusip",
    y_transform="log",
    retransform="lognormal",
)
# columns: cusip, date, val_hz, val_hz_dts, val_hz_wi
```

### 2) Israel, Palhares & Richardson (2018) value signal (`model_type="ipr"`)

**Reference:** Israel, Palhares, and Richardson (2018), “Common factors in corporate bond returns,” *Journal of Investment Management*, 16(2).

This variant uses:
- numeric rating control (`rating_mode="numeric"`),
- FF17 industry dummies (`industry_col="ff17num"`),
- call dummy, log duration and bond excess-return volatility (`x_cols=["call", "log_md_dur", "vol12_x"]`),
- log spreads (`y_transform="log"`, `retransform="lognormal"`),
- and defines value as the **log-spread residual** (`denom="resid"`).

The call as `lib/value.py` makes it:

```python
end["log_md_dur"] = np.log(end["md_dur"])

val_end1 = compute_value(
    end,
    id_col="cusip",
    date_col="date",
    model_type="ipr",
    cs_col="cs",
    rating_mode="numeric",
    industry_col="ff17num",
    x_cols=["call", "log_md_dur", "vol12_x"],
    dur_col="md_dur",
    firm_col="issuer_cusip",
    denom="resid",
    y_transform="log",
    retransform="lognormal",
)
# columns: cusip, date, val_ipr, val_ipr_dts, val_ipr_wi
```

---

## Output schema

If `dur_col` is **not** provided, output is:

- `id_col`, `date_col`, `val_{model_type}`

If `dur_col` **is** provided, output is:

- `id_col`, `date_col`, `val_{model_type}`, `val_{model_type}_dts`, `val_{model_type}_wi`

If `return_overlay_details=True`, additional diagnostic columns are included from the DtS overlay:
- `dts`: duration-times-spread
- `Q_dts`: DtS quintile (1–5)
- `val_{model_type}_m_dtsQ`: mean signal within (date, Q_dts)

---

## Practical notes and recommendations

1. **Sample consistency matters.** Adding controls (e.g., `call`) can drop observations if the control has missing values.
   This changes the monthly regression coefficients and can materially change the signal.
2. **Log spreads require strictly positive spreads.** The function drops non-positive spreads when `y_transform="log"`.
3. **Interpretation of the sign.** In the scaled versions (`denom="fitted"` or `"actual"`), larger values mean spreads are wider than fitted (“cheaper”).
   In residual mode, positive residual means spreads are wider than predicted.
4. **Industry dummies.** Adding FF17 dummies often improves the “fair spread” benchmark by absorbing persistent cross-industry spread differences.

---

## References

- Duan, N. (1983). “Smearing estimate: A nonparametric retransformation method.” *Journal of the American Statistical Association*.
- Houweling, P., and J. van Zundert (2017). “Factor Investing in the Corporate Bond Market.” *Financial Analysts Journal*, 73(2), 100-115.
- Israel, R., D. Palhares, and S. Richardson (2018). “Common factors in corporate bond returns.” *Journal of Investment Management*, 16(2), 17-46.
