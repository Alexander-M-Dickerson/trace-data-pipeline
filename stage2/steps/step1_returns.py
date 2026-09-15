"""step1_returns.py -- DuckDB port of upstream step 1: monthly returns (6 price types), month-end
signals, adjusted signals, LIB, default handling, and duration-matched Treasury returns.

Ground truth: the reference implementation's monthly builder (the chunked wrapper is
semantics-neutral) + `normalize_default_returns` + the returns_alt slice of `wrangle_returns`. Each
stage below cites the upstream step it reproduces. Two-trap reminders:
  - R = (fp_t - fp_s) / prfull_s with fp = pr + accall (coupons retained), prfull = pr + acclast.
  - float32 fidelity: price arithmetic stays FLOAT end-to-end exactly like pandas float32 Series;
    mcap is BIGINT * float32 -> float64 (numpy promotion), reproduced with explicit DOUBLE casts.

Outputs (parquet blocks under output/blocks/<mode>/), grain one row per (cusip, calendar month-end):
  end_returns, bgn_returns, end_signals, adj_signals, all_returns, returns_alt
returns_alt is the returns validation target. all_returns keeps the PRE-normalization ret_vw
(the runner normalizes end/bgn AFTER all_returns is built inside process_bond_data).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _stage2_settings as cfg
from lib import duration_adjusted, month_boundaries, nyse_calendar, treasury
from lib import pin as pinlib

# price column (pin name) -> return column. Upstream PRICE_MAP; its 'last_bid' key never matches the
# golden input (whose column is bid_last), so ret_lbd never materializes -- 6 return types, not 7.
PRICE_MAP = {
    "pr": "ret_vw",
    "prc_vw_par": "ret_vwp",
    "prc_ew": "ret_ew",
    "prc_1st": "ret_1st",
    "prc_lst": "ret_lst",
    "prc_bid": "ret_bid",
}
ALT_RET_COLS = ("ret_vwp", "ret_ew", "ret_1st", "ret_lst", "ret_bid")


def _attach_tret(con, table: str, date_col: str, mode: str | None = None) -> None:
    """Upstream step 20: round mod_dur to 2dp (pandas float32 semantics), interpolate the CRSP
    fixed-term curve at it, and add `tret` to `table` keyed exactly by (cusip_id, {date_col})."""
    dftret = treasury.load_tret_wide(mode=mode)
    keys = con.execute(
        f"SELECT cusip_id, {date_col} AS date, mod_dur FROM {table}").df()
    keys["date"] = pd.to_datetime(keys["date"])
    keys["mod_dur"] = keys["mod_dur"].round(2)                     # float32 .round, as upstream
    pairs = keys[["date", "mod_dur"]].drop_duplicates().reset_index(drop=True)
    tret_map = treasury.interpolate_tret(pairs, dftret)
    keys = keys.merge(tret_map, on=["date", "mod_dur"], how="left")
    con.register("_tret_rows", keys[["cusip_id", "date", "tret"]])
    con.execute(f"""
CREATE OR REPLACE TEMP TABLE {table}_t AS
SELECT t.*, r.tret
FROM {table} t
LEFT JOIN _tret_rows r ON r.cusip_id = t.cusip_id AND r.date = t.{date_col}
""")
    con.unregister("_tret_rows")


def build(con, mode: str | None = None, limit_cusips: int | None = None) -> dict[str, Path]:
    """Build all step-1 blocks; returns {block_name: parquet_path}. Idempotent, ~O(pin) work."""
    mode = mode or cfg.INPUT_MODE
    t0 = time.time()
    pin_path = pinlib.build_pin(con, mode, limit_cusips)
    out_dir = cfg.BLOCKS_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    months = con.execute(
        f"SELECT DISTINCT month_start FROM read_parquet('{pin_path.as_posix()}')").df()
    con.register("month_bounds", month_boundaries.build_month_bounds(months["month_start"]))
    con.register("cal_lut", nyse_calendar.cal_lut_frame())

    fp_defs, lag_defs, ret_defs = [], [], []
    for px in PRICE_MAP:
        sfx = "" if px == "pr" else f"_{px}"
        fp_defs.append(f"{px} + accall AS fp{sfx}, {px} + acclast AS prfull{sfx}")
        lag_defs.append(f"lag(fp{sfx}) OVER w AS fp{sfx}_s, lag(prfull{sfx}) OVER w AS prfull{sfx}_s")
        ret_defs.append(
            f"(fp{sfx} - fp{sfx}_s) / prfull{sfx}_s AS {PRICE_MAP[px]}")

    # ============ upstream steps 3-7: dedup, pr filter, fp/prfull/mcap, month boundaries ==========
    con.execute(f"""
CREATE OR REPLACE TEMP TABLE t_src AS
SELECT s.cusip_id, s.dt, s.month_start,
       s.pr, s.prc_vw_par, s.prc_ew, s.prc_1st, s.prc_lst, s.prc_bid,
       s.accall, s.acclast, s.ao,
       s.ytm, s.mod_dur, s.convexity, s.credit_spread, s.bond_maturity, s.bond_age,
       s.sp_rating, s.mdy_rating, s.spc_rating, s.mdc_rating, s.ff17num, s.ff30num,
       {', '.join(fp_defs)},
       -- mcap is a FLOAT32 chain end-to-end: the golden's ao was float32 in memory, so every op
       -- rounds to float32 (empirically arbitrated -- f64 math lands 1 ulp off on 35% of rows, M10)
       CAST(s.ao AS FLOAT) * (s.pr + s.acclast) * CAST(10 AS FLOAT) / CAST(1e6 AS FLOAT) AS mcap,
       s.frn,
       mb.impl_floor, mb.cut_off_begin, mb.cut_off_end, mb.date_end_bus_lag, mb.month_end_cal
FROM (
  SELECT * FROM read_parquet('{pin_path.as_posix()}')
  WHERE pr IS NOT NULL
  QUALIFY row_number() OVER (PARTITION BY cusip_id, dt ORDER BY frn) = 1
) s
JOIN month_bounds mb USING (month_start)
""")

    # ============ upstream step 9: boundary windows, first/last pick, dummies, pair count =========
    con.execute("""
CREATE OR REPLACE TEMP TABLE t_bound AS
SELECT *,
  (dt >= cut_off_end)::TINYINT AS end_dummy,
  CASE WHEN dt >= cut_off_end THEN month_end_cal ELSE dt END AS date_end_ref,
  count(*) OVER (PARTITION BY cusip_id, month_start) AS cnt
FROM (
  SELECT * FROM (
    SELECT *, min(dt) OVER w AS min_dt, max(dt) OVER w AS max_dt
    FROM t_src
    WHERE (dt >= impl_floor AND dt <= cut_off_begin) OR dt >= cut_off_end
    WINDOW w AS (PARTITION BY cusip_id, month_start)
  )
  WHERE (dt = min_dt AND dt <= cut_off_begin) OR (dt = max_dt AND dt >= cut_off_end)
)
""")

    # ============ upstream step 11: month-end frame + per-cusip lags (PRE n<=31 filter) ===========
    con.execute(f"""
CREATE OR REPLACE TEMP TABLE t_end_full AS
SELECT *, {', '.join(ret_defs)}
FROM (
  SELECT *,
    date_diff('day', lag(date_end_ref) OVER w, date_end_ref) AS n_days,
    {', '.join(lag_defs)},
    lag(pr) OVER w AS pr_start, lag(accall) OVER w AS accall_start,
    lag(acclast) OVER w AS acclast_start, lag(dt) OVER w AS dt_s,
    lag(mcap) OVER w AS mcap_s
  FROM (SELECT * FROM t_bound WHERE end_dummy = 1)
  WINDOW w AS (PARTITION BY cusip_id ORDER BY date_end_ref)
)
""")

    # ============ upstream step 10: within-month (bgn) frame ======================================
    # cnt=2 months hold exactly [begin-row, end-row]; the end-row's cusip-lag is its month's begin row
    con.execute("""
CREATE OR REPLACE TEMP TABLE t_bgn_full AS
SELECT * FROM (
  SELECT *,
    lag(fp) OVER w AS fp_s, lag(prfull) OVER w AS prfull_s,
    lag(pr) OVER w AS pr_start, lag(accall) OVER w AS accall_start,
    lag(acclast) OVER w AS acclast_start, lag(mcap) OVER w AS mcap_s,
    min(dt) OVER m AS date_start
  FROM (SELECT * FROM t_bound WHERE cnt = 2)
  WINDOW w AS (PARTITION BY cusip_id ORDER BY dt),
         m AS (PARTITION BY cusip_id, month_start)
)
WHERE end_dummy = 1
""")

    # ============ upstream steps 10.5/11.5: default-return handling (event_based) =================
    # ratings lag AFTER the n<=31 filter (end) / on the bgn-returns frame, per pandas shift order.
    # pandas NaN==22 -> False is COALESCE(..., FALSE). ret_std snapshots the pre-adjustment ret_vw.
    default_case = """
  CASE WHEN NOT in_def_lag AND in_def THEN 'default_evnt'
       WHEN in_def_lag AND in_def THEN 'trad_in_def'
       ELSE 'standard' END AS ret_type,
  ret_vw AS ret_std,
  CASE WHEN NOT in_def_lag AND in_def THEN (pr - prfull_s) / prfull_s
       WHEN in_def_lag AND in_def THEN (pr - pr_start) / pr_start
       ELSE ret_vw END AS ret_vw_adj
"""
    in_def_defs = """
    COALESCE(sp_rating = 22, FALSE) OR COALESCE(mdy_rating = 21, FALSE) AS in_def,
    COALESCE(lag(sp_rating) OVER wd = 22, FALSE)
      OR COALESCE(lag(mdy_rating) OVER wd = 21, FALSE) AS in_def_lag
"""
    con.execute(f"""
CREATE OR REPLACE TEMP TABLE t_end_ret AS
SELECT *, {default_case}
FROM (
  SELECT *, {in_def_defs}
  FROM (SELECT * FROM t_end_full WHERE n_days <= 31)
  WINDOW wd AS (PARTITION BY cusip_id ORDER BY date_end_ref)
)
""")
    con.execute(f"""
CREATE OR REPLACE TEMP TABLE t_bgn_ret AS
SELECT *, {default_case}
FROM (
  SELECT *, (fp - fp_s) / prfull_s AS ret_vw, {in_def_defs}
  FROM t_bgn_full
  WINDOW wd AS (PARTITION BY cusip_id ORDER BY dt)
)
""")

    # ============ upstream steps 12-13: hprd / igap via the session LUT ===========================
    # end hprd: sessions in [prev month-end TRADE, calendar month-end); bgn hprd: [first, last trade);
    # igap: [previous business month-end, first trade)
    con.execute("""
CREATE OR REPLACE TEMP TABLE t_end_ret2 AS
SELECT e.*, (ce.cum_before - cs.cum_before) AS hprd
FROM t_end_ret e
LEFT JOIN cal_lut cs ON cs.cday = e.dt_s
LEFT JOIN cal_lut ce ON ce.cday = e.date_end_ref
""")
    con.execute("""
CREATE OR REPLACE TEMP TABLE t_bgn_ret2 AS
SELECT b.*, (ce.cum_before - cs.cum_before) AS hprd,
       (cs.cum_before - cl.cum_before) AS igap
FROM t_bgn_ret b
LEFT JOIN cal_lut cs ON cs.cday = b.date_start
LEFT JOIN cal_lut ce ON ce.cday = b.dt
LEFT JOIN cal_lut cl ON cl.cday = b.date_end_bus_lag
""")

    # ============ upstream step 15: LIB (bgn gap cost, re-dated to t-1, merged onto end) ==========
    # lib = (pr_bgn_start - pr_prev)/pr_prev ; libd = (fp_bgn_start - fp_prev)/prfull_prev, where
    # *_prev are the end frame's lagged month-end values at the SAME (cusip, month).
    con.execute("""
CREATE OR REPLACE TEMP TABLE t_lib AS
SELECT b.cusip_id,
       last_day((b.month_end_cal - INTERVAL 1 MONTH)::DATE) AS date_lib,
       (b.pr_start - e.pr_start) / e.pr_start AS lib,
       ((b.pr_start + b.accall_start) - (e.pr_start + e.accall_start))
         / (e.pr_start + e.acclast_start) AS libd
FROM t_bgn_ret2 b
JOIN t_end_ret2 e ON e.cusip_id = b.cusip_id AND e.date_end_ref = b.month_end_cal
""")

    # ============ upstream step 17 filters, then step 20 tret =====================================
    con.execute(f"""
CREATE OR REPLACE TEMP TABLE t_end_f AS
SELECT e.*, l.lib, l.libd
FROM (SELECT * FROM t_end_ret2 WHERE hprd > 0 AND date_end_ref >= DATE '{cfg.START_DATE}') e
LEFT JOIN t_lib l ON l.cusip_id = e.cusip_id AND l.date_lib = e.date_end_ref
""")
    con.execute(f"""
CREATE OR REPLACE TEMP TABLE t_bgn_f AS
SELECT * FROM t_bgn_ret2 WHERE hprd > 0 AND month_end_cal >= DATE '{cfg.START_DATE}'
""")
    _attach_tret(con, "t_end_f", "date_end_ref", mode=mode)
    _attach_tret(con, "t_bgn_f", "month_end_cal", mode=mode)

    # Five more Treasury benchmarks, built from each bond's own cash flows -- see
    # lib/duration_adjusted.py for the equations and citations.
    #
    # Attached to BOTH frames, exactly as `tret` is. It was the end frame only while these were
    # just panel columns -- the shipped ones are end-timed, so a `_bgn` twin had no reader. That
    # stopped being true when `all_returns` became the input to duration-adjusted rolling betas:
    # it unions the two frames, and 8.1% of its rows come from bgn alone. Leaving those without a
    # benchmark would estimate the alternative-benchmark betas on a non-random 92% subsample of
    # the rows the `tret` ones use, which is a confound in precisely the comparison the
    # alternatives exist to support. The bgn frame calls its window start `date_start`.
    _da = duration_adjusted.attach(
        con, "t_end_f_t", "date_end_ref", "t_end_f_td",
        treasury_mod=treasury, mode=mode)
    print(f"  duration-adjusted benchmarks [end]: {_da['tret_bns']:,}/{_da['rows']:,} rows "
          f"(tret_mat {_da['tret_mat']:,})")
    _dab = duration_adjusted.attach(
        con, "t_bgn_f_t", "month_end_cal", "t_bgn_f_td",
        treasury_mod=treasury, mode=mode, start_col="date_start")
    print(f"  duration-adjusted benchmarks [bgn]: {_dab['tret_bns']:,}/{_dab['rows']:,} rows "
          f"(tret_mat {_dab['tret_mat']:,})")

    # ============ upstream steps 16/16.6: end_signals (PRE n<=31 frame, point-in-time) ============
    # bbtm = 100/pr (float32); sze = mcap; fce_val = round(ao); short-name renames per SIGNAL_NAME_MAP
    con.execute(f"""
CREATE OR REPLACE TEMP TABLE t_sig AS
SELECT cusip_id AS cusip, date_end_ref AS date, dt_s, dt AS dt_e,
       ytm, mod_dur AS md_dur, convexity AS convx, credit_spread AS cs,
       sp_rating AS sp_rat, mdy_rating AS mdy_rat,
       spc_rating AS spc_rat, mdc_rating AS mdyc_rat,
       bond_maturity AS tmat, bond_age AS age,
       CAST(round(ao) AS BIGINT) AS fce_val, ff17num, ff30num,
       100 / pr AS bbtm, mcap AS sze, mcap_s, mcap AS mcap_e,
       fp_s, prfull_s, month_start, month_end_cal
FROM t_end_full
WHERE date_end_ref >= DATE '{cfg.START_DATE}'
QUALIFY row_number() OVER (PARTITION BY cusip_id, date_end_ref ORDER BY dt) = 1
""")

    # ============ upstream step 16 (adj machinery): pick the lagged in-month signal trade =========
    # candidates: same-month trades STRICTLY before the month-end trade, within adj_window sessions
    # of the CALENDAR month-end; pick the trade with min |calendar days to cut_off_adj| where
    # cut_off_adj = last session strictly before the month-end trade; ties -> earliest trade.
    # Lookup values are float32-downcast upstream (price_df_for_signals), hence the FLOAT casts.
    con.execute(f"""
CREATE OR REPLACE TEMP TABLE t_adj_best AS
SELECT * FROM (
  SELECT g.cusip_id, g.month_end_cal, g.dt AS sig_dt, g.dt_e,
         CAST(g.pr AS FLOAT) AS pr_adj, CAST(g.accall AS FLOAT) AS accall_adj,
         CAST(g.mcap AS FLOAT) AS mcap_adj,
         CAST(g.ytm AS FLOAT) AS ytm_adj, CAST(g.mod_dur AS FLOAT) AS md_dur_adj,
         CAST(g.convexity AS FLOAT) AS convx_adj, CAST(g.credit_spread AS FLOAT) AS cs_adj,
         abs(date_diff('day', ps.prev_session, g.dt)) AS ddiff
  FROM (
    SELECT p.*, t.dt_e
    FROM t_src p
    JOIN (SELECT cusip, month_start, dt_e FROM t_sig) t
      ON t.cusip = p.cusip_id AND t.month_start = p.month_start
    WHERE p.dt < t.dt_e
  ) g
  JOIN cal_lut cd ON cd.cday = g.dt
  JOIN cal_lut me ON me.cday = g.month_end_cal
  JOIN cal_lut ps ON ps.cday = g.dt_e
  WHERE (me.cum_before - cd.cum_before) <= {cfg.ADJ_WINDOW}
)
QUALIFY row_number() OVER (PARTITION BY cusip_id, month_end_cal ORDER BY ddiff, sig_dt) = 1
""")

    # str1_adj = (fp_adj - fp_s)/prfull_s (prev MONTH-END lags); str2_adj uses the BGN frame's lags;
    # bbtm_adj/sze_adj are float32 values stored float64 (np.asarray(..., 'float64') upstream);
    # sig_gap = sessions in [sig_dt, month-end trade)
    con.execute("""
CREATE OR REPLACE TEMP TABLE t_adj AS
SELECT s.cusip, s.date, a.sig_dt,
       CAST(cse.cum_before - css.cum_before AS DOUBLE) AS sig_gap,
       a.ytm_adj, a.md_dur_adj, a.convx_adj, a.cs_adj,
       ((a.pr_adj + a.accall_adj) - s.fp_s) / s.prfull_s AS str1_adj,
       ((a.pr_adj + a.accall_adj) - b.fp_s) / b.prfull_s AS str2_adj,
       CAST(100 / a.pr_adj AS DOUBLE) AS bbtm_adj,
       CAST(a.mcap_adj AS DOUBLE) AS sze_adj
FROM t_sig s
LEFT JOIN t_adj_best a ON a.cusip_id = s.cusip AND a.month_end_cal = s.date
LEFT JOIN t_bgn_ret2 b ON b.cusip_id = s.cusip AND b.month_end_cal = s.date
LEFT JOIN cal_lut css ON css.cday = a.sig_dt
LEFT JOIN cal_lut cse ON cse.cday = a.dt_e
""")

    # ============ final frames: normalize (runner) + block outputs ================================
    # end_returns: ret_vw = default-adjusted then capped at ret_std for trad_in_def; alt returns are
    # aligned to the normalized ret_vw on default rows (normalize_default_returns, is_end=True).
    alt_norm = ",\n       ".join(
        f"""CASE WHEN ret_type = 'default_evnt' THEN ret_vw_n
            WHEN ret_type = 'trad_in_def' AND {c} IS NOT NULL THEN ret_vw_n
            ELSE {c} END AS {c}""" for c in ALT_RET_COLS)
    con.execute(f"""
CREATE OR REPLACE TEMP TABLE end_returns AS
SELECT cusip, date, dt_s, dt_e, hprd, lib, libd, ret_std, ret_type,
       ret_vw_n AS ret_vw, {alt_norm},
       sp_rat, mdy_rat, spc_rat, mdyc_rat, tret,
       tret_bns, tret_cfm, tret_gprs, tret_cls, tret_mat, ret_vw_pre
FROM (
  SELECT cusip_id AS cusip, date_end_ref AS date, dt_s, dt AS dt_e, hprd, lib, libd,
         ret_std, ret_type, ret_vw_adj AS ret_vw_pre,
         CASE WHEN ret_type = 'trad_in_def' AND ret_vw_adj > ret_std THEN ret_std
              ELSE ret_vw_adj END AS ret_vw_n,
         ret_vwp, ret_ew, ret_1st, ret_lst, ret_bid,
         sp_rating AS sp_rat, mdy_rating AS mdy_rat,
         spc_rating AS spc_rat, mdc_rating AS mdyc_rat, tret,
         tret_bns, tret_cfm, tret_gprs, tret_cls, tret_mat
  FROM t_end_f_td
)
""")
    con.execute("""
CREATE OR REPLACE TEMP TABLE bgn_returns AS
SELECT cusip_id AS cusip, month_end_cal AS date, date_start AS dt_s, dt AS dt_e,
       CASE WHEN ret_type = 'trad_in_def' AND ret_vw_adj > ret_std THEN ret_std
            ELSE ret_vw_adj END AS ret_vw,
       ret_vw_adj AS ret_vw_pre,
       hprd, igap, ret_std, ret_type, mcap_s, mcap AS mcap_e,
       sp_rating AS sp_rat, mdy_rating AS mdy_rat,
       spc_rating AS spc_rat, mdc_rating AS mdyc_rat, tret
FROM t_bgn_f_td
""")

    # all_returns: PRE-normalization ret_vw, end rows take precedence over bgn on (cusip, date)
    # end_returns / bgn_returns are shipped blocks with fixed projections, so the benchmarks are
    # taken from the extended frames rather than by widening those two.
    con.execute("""
CREATE OR REPLACE TEMP TABLE t_end_f_td_ar AS
SELECT cusip_id AS cusip, date_end_ref AS date,
       CASE WHEN ret_type = 'trad_in_def' AND ret_vw_adj > ret_std THEN ret_std
            ELSE ret_vw_adj END AS ret_vw_x, ret_vw_adj AS ret_vw_pre,
       tret, tret_bns, tret_cls, ret_std, ret_type
FROM t_end_f_td
""")
    con.execute("""
CREATE OR REPLACE TEMP TABLE t_bgn_f_td_ar AS
SELECT cusip_id AS cusip, month_end_cal AS date,
       ret_vw_adj AS ret_vw_pre, tret, tret_bns, tret_cls, ret_std, ret_type
FROM t_bgn_f_td
""")
    con.execute("""
CREATE OR REPLACE TEMP TABLE all_returns AS
SELECT cusip, date, ret_vw, tret, tret_bns, tret_cls, ret_std, ret_type FROM (
  SELECT *, row_number() OVER (PARTITION BY cusip, date ORDER BY src) AS rn FROM (
    SELECT cusip, date, ret_vw_pre AS ret_vw, tret, tret_bns, tret_cls,
           ret_std, ret_type, 0 AS src FROM t_end_f_td_ar
    UNION ALL
    SELECT cusip, date, ret_vw_pre, tret, tret_bns, tret_cls,
           ret_std, ret_type, 1 FROM t_bgn_f_td_ar
  )
) WHERE rn = 1
""")

    # ============ write blocks =====================================================================
    blocks: dict[str, Path] = {}

    def _copy(name: str, sql: str) -> None:
        p = out_dir / f"{name}.parquet"
        con.execute(f"COPY ({sql}) TO '{p.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        blocks[name] = p

    _copy("end_returns", """
        SELECT cusip, date::TIMESTAMP AS date, dt_s::TIMESTAMP AS dt_s, dt_e::TIMESTAMP AS dt_e,
               hprd, lib, libd, ret_std, ret_type, ret_vw,
               ret_vwp, ret_ew, ret_1st, ret_lst, ret_bid,
               sp_rat, mdy_rat, spc_rat, mdyc_rat, tret,
               tret_bns, tret_cfm, tret_gprs, tret_cls, tret_mat
        FROM end_returns ORDER BY cusip, date""")
    _copy("bgn_returns", """
        SELECT cusip, date::TIMESTAMP AS date, dt_s::TIMESTAMP AS dt_s, dt_e::TIMESTAMP AS dt_e,
               ret_vw, hprd, igap, ret_std, ret_type,
               CAST(mcap_s AS DOUBLE) AS mcap_s, CAST(mcap_e AS DOUBLE) AS mcap_e,
               sp_rat, mdy_rat, spc_rat, mdyc_rat, tret
        FROM bgn_returns ORDER BY cusip, date""")
    _copy("end_signals", """
        SELECT cusip, date::TIMESTAMP AS date, dt_s::TIMESTAMP AS dt_s, dt_e::TIMESTAMP AS dt_e,
               ytm, md_dur, convx, cs, sp_rat, mdy_rat, spc_rat, mdyc_rat,
               tmat, age, fce_val, ff17num, ff30num,
               bbtm, sze, CAST(mcap_s AS DOUBLE) AS mcap_s, CAST(mcap_e AS DOUBLE) AS mcap_e
        FROM t_sig ORDER BY cusip, date""")
    _copy("adj_signals", """
        SELECT cusip, date::TIMESTAMP AS date, sig_dt::TIMESTAMP AS sig_dt, sig_gap,
               ytm_adj, md_dur_adj, convx_adj, cs_adj, str1_adj, str2_adj, bbtm_adj, sze_adj
        FROM t_adj ORDER BY cusip, date""")
    _copy("all_returns", """
        SELECT cusip, date::TIMESTAMP AS date, ret_vw, tret, tret_bns, tret_cls, ret_std, ret_type
        FROM all_returns ORDER BY cusip, date""")
    # firm_ids: Stage 1's permno/permco/gvkey collapsed to one row per bond-month, taken at
    # the same month-end trade t_sig uses. Stage 7 merges this instead of re-deriving the
    # identifiers from a separate issuer-level linker file.
    #
    # ! Read STRAIGHT FROM THE PIN and joined on (cusip_id, dt), rather than carried through
    #   t_src. Adding columns to t_src changes the physical row order, and several float32
    #   illiquidity kernels downstream are order-sensitive -- doing it that way moved six
    #   unrelated columns by ~1e-13 relative. Identifiers must not perturb the numeric path.
    _copy("firm_ids", f"""
        SELECT e.cusip_id AS cusip, e.date_end_ref::TIMESTAMP AS date,
               p.permno, p.permco, p.gvkey
        FROM (
            SELECT cusip_id, dt, date_end_ref FROM t_end_full
            WHERE date_end_ref >= DATE '{cfg.START_DATE}'
            QUALIFY row_number() OVER (PARTITION BY cusip_id, date_end_ref ORDER BY dt) = 1
        ) e
        JOIN (
            SELECT cusip_id, dt, permno, permco, gvkey
            FROM read_parquet('{pin_path.as_posix()}')
            WHERE pr IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY cusip_id, dt ORDER BY frn) = 1
        ) p ON p.cusip_id = e.cusip_id AND p.dt = e.dt
        ORDER BY cusip, date""")
    # returns_alt: the returns validation target (wrangle_returns step 4 slice, float32)
    _copy("returns_alt", """
        SELECT cusip, date::TIMESTAMP AS date,
               CAST(ret_vwp AS FLOAT) AS ret_vwp, CAST(ret_ew AS FLOAT) AS ret_ew,
               CAST(ret_1st AS FLOAT) AS ret_1st, CAST(ret_lst AS FLOAT) AS ret_lst,
               CAST(ret_bid AS FLOAT) AS ret_bid, CAST(tret AS FLOAT) AS tret
        FROM end_returns ORDER BY cusip, date""")

    (out_dir / "step1_meta.json").write_text(json.dumps({
        "wall_s": round(time.time() - t0, 2), "mode": mode, "limit_cusips": limit_cusips,
        "rows": {n: con.execute(f"SELECT count(*) FROM read_parquet('{p.as_posix()}')").fetchone()[0]
                 for n, p in blocks.items()},
    }, indent=1))
    return blocks
