"""pin.py -- the Phase-0 daily projection: the streamlined daily panel, built ONCE, reused by every step.

Faithful DuckDB port of upstream `illiq_helper_functions.streamline_data()` (the step-2 prep), widened
into the shared base every step reads (the step-1 signal columns ride along so no step re-reads the
multi-GB daily input). Grain: one row per input row (the 210 duplicate (cusip, date) twins in the
golden input are KEPT, as upstream does; they are exact-price duplicates -- assumptions.md A4).

Upstream semantics reproduced exactly:
  - drop rows where BOTH sp_rating and mdy_rating are NULL (step 1.2b / streamline step 1)
  - fp = pr + accall;  ret_d = (fp - fp_lag) / prfull_lag;  ret_c = (pr - pr_lag) / pr_lag
  - day_gap = NYSE sessions in [prev_trd_dt, trd_exctn_dt)  (np.busday_count == cum_before diff)
  - returns AND dvol_lag are NULLed where day_gap > max_day_gap; the LAGGED returns are shifts of the
    ALREADY-gated series (no second gate)
  - Nret / Npair per (cusip, month) on ret_c; lst_txn = 1 unless last row of the (cusip, month) group
  - renames: prc_first->prc_1st, prc_last->prc_lst, dvolume->dvol, bond_amt_outstanding->ao

The pin is cached to output/_cache/pin_<mode>[_limitN]_<fp8>.parquet keyed by a content fingerprint
(input identity + the knobs that affect it + PIN_VERSION); a knob change rebuilds only this layer.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import _stage2_settings as cfg
from lib import nyse_calendar

# Bump when the pin SQL changes shape/semantics -- invalidates every cached pin.
# v2: + prc_vw_par, prc_ew (step-1 return types ret_vwp / ret_ew)
# v3: + frn (input file_row_number) carried through, and ALL per-cusip windows tie-break on it --
#     needed for the one bond whose whole tape is duplicated (dup (cusip,date) twins): pandas'
#     near-sorted sorts preserve raw file order on ties, so window lags/lst_txn must too (debug M10)
PIN_VERSION = 3


def pin_path(mode: str, limit_cusips: int | None = None) -> Path:
    """Deterministic cache path for the pin parquet under the current config."""
    daily = cfg.daily_input(mode)
    st = daily.stat()
    ident = {
        "pin_version": PIN_VERSION,
        "input": str(daily), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
        "max_day_gap": cfg.BUSINESS_DAY_GAP,
        "cal": [cfg.CAL_START, cfg.CAL_END, cfg.CALENDAR_NAME],
        "limit_cusips": limit_cusips,
    }
    fp8 = hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:8]
    tag = f"_limit{limit_cusips}" if limit_cusips else ""
    return cfg.CACHE_DIR / f"pin_{mode}{tag}_{fp8}.parquet"


def pin_sql(daily_path: Path, calendar_path: Path, max_day_gap: int,
            limit_cusips: int | None = None) -> str:
    """The one SELECT that builds the pin (see module docstring for the semantics)."""
    daily = Path(daily_path).as_posix()
    cal = Path(calendar_path).as_posix()
    cusip_filter = ""
    if limit_cusips:
        # deterministic dev universe: first N cusips by lexicographic order (house rule)
        cusip_filter = f"""
          AND cusip_id IN (SELECT DISTINCT cusip_id FROM read_parquet('{daily}')
                           ORDER BY cusip_id LIMIT {int(limit_cusips)})"""
    return f"""
WITH cal AS (
  SELECT "day"::DATE AS cday, cum_before FROM read_parquet('{cal}')
),
src AS (
  SELECT
    file_row_number AS frn,
    cusip_id, CAST(trd_exctn_dt AS DATE) AS dt,
    pr, prfull, acclast, accpmt, accall,
    prc_vw_par, prc_ew, prc_first, prc_last, prc_hi, prc_lo, prc_bid, prc_ask, bid_last,
    trade_count, bid_count, ask_count,
    qvolume, dvolume, bond_amt_outstanding,
    ytm, mod_dur, mac_dur, convexity, credit_spread, bond_maturity, bond_age,
    sp_rating, mdy_rating, spc_rating, mdc_rating,
    permno, permco, gvkey, ff17num, ff30num, db_type
  FROM read_parquet('{daily}', file_row_number=true)
  WHERE ((sp_rating IS NOT NULL) OR (mdy_rating IS NOT NULL)){cusip_filter}
),
lagged AS (
  SELECT *,
    pr + accall                       AS fp,
    lag(pr + accall) OVER win         AS fp_lag,
    lag(prfull)      OVER win         AS prfull_lag,
    lag(pr)          OVER win         AS pr_lag,
    lag(dt)          OVER win         AS prev_dt,
    lag(dvolume)     OVER win         AS dvol_lag_raw
  FROM src
  WINDOW win AS (PARTITION BY cusip_id ORDER BY dt, frn)
),
gapped AS (
  SELECT l.*, (ce.cum_before - cs.cum_before) AS day_gap
  FROM lagged l
  LEFT JOIN cal ce ON ce.cday = l.dt
  LEFT JOIN cal cs ON cs.cday = l.prev_dt
),
gated AS (
  -- day_gap > max NULLs the return; a NULL day_gap (first obs) passes through like upstream NaN>max=False
  SELECT *,
    CASE WHEN day_gap > {max_day_gap} THEN NULL ELSE (fp - fp_lag) / prfull_lag END AS ret_d,
    CASE WHEN day_gap > {max_day_gap} THEN NULL ELSE (pr - pr_lag) / pr_lag    END AS ret_c,
    CASE WHEN day_gap > {max_day_gap} THEN NULL ELSE dvol_lag_raw              END AS dvol_lag
  FROM gapped
),
shifted AS (
  SELECT *,
    lag(ret_d) OVER win AS ret_d_lag,
    lag(ret_c) OVER win AS ret_c_lag,
    date_trunc('month', dt)::DATE AS month_start
  FROM gated
  WINDOW win AS (PARTITION BY cusip_id ORDER BY dt, frn)
)
SELECT
  frn, cusip_id, dt, month_start, db_type,
  pr, prfull, acclast, accpmt, accall,
  prc_vw_par, prc_ew, prc_first AS prc_1st, prc_last AS prc_lst, prc_hi, prc_lo, prc_bid, prc_ask, bid_last,
  trade_count, bid_count, ask_count,
  qvolume, dvolume AS dvol, bond_amt_outstanding AS ao,
  ytm, mod_dur, mac_dur, convexity, credit_spread, bond_maturity, bond_age,
  sp_rating, mdy_rating, spc_rating, mdc_rating,
  permno, permco, gvkey, ff17num, ff30num,
  day_gap::SMALLINT AS day_gap,
  ret_d, ret_c, ret_d_lag, ret_c_lag, dvol_lag,
  sum((ret_c IS NOT NULL)::INT) OVER mwin ::SMALLINT AS "Nret",
  sum((ret_c IS NOT NULL AND ret_c_lag IS NOT NULL)::INT) OVER mwin ::SMALLINT AS "Npair",
  (row_number() OVER (PARTITION BY cusip_id, month_start ORDER BY dt, frn)
     < count(*) OVER mwin)::TINYINT AS lst_txn
FROM shifted
WINDOW mwin AS (PARTITION BY cusip_id, month_start)
"""


def build_pin(con, mode: str | None = None, limit_cusips: int | None = None,
              force: bool = False) -> Path:
    """Build (or reuse) the cached pin parquet; return its path. Logs wall time + rows."""
    mode = mode or cfg.INPUT_MODE
    out = pin_path(mode, limit_cusips)
    if out.exists() and not force:
        return out
    cfg.ensure_dirs()
    cal = nyse_calendar.ensure_calendar()
    sql = pin_sql(cfg.daily_input(mode), cal, cfg.BUSINESS_DAY_GAP, limit_cusips)
    t0 = time.time()
    tmp = out.with_suffix(".tmp.parquet")
    con.execute(f"COPY ({sql}) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    tmp.replace(out)  # atomic-ish: never leave a half-written pin under the final name
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    out.with_suffix(".json").write_text(json.dumps(
        {"rows": n, "wall_s": round(time.time() - t0, 2), "mode": mode,
         "limit_cusips": limit_cusips, "pin_version": PIN_VERSION,
         "input": str(cfg.daily_input(mode))}, indent=1))
    return out
