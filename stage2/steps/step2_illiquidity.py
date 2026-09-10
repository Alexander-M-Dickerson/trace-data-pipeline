"""step2_illiquidity.py -- DuckDB port of upstream step 2: per-bond illiquidity/risk signals and the
market-level nontraded factor panel.

Ground truth: `stage2/illiq_helper_functions.py::run_illiquidity_pipeline` and its metric functions
(monthly_pi_fast, compute_monthly_amihud, compute_monthly_illiq_roll_fast, compute_hong_warga_spreads,
cs_ar_spreads, p_zeros, compute_bond_turnover, compute_vov, compute_within_month_risk). The pin IS the
streamline_data output (validated bit-faithful at G0), so every metric reads the pin.

Faithfulness notes (each mirrors an upstream quirk -- do not "fix"):
  - ilq/roll: n = the group's ROW count (pandas 'size'), while the sums skip NaN -- inconsistent on
    purpose; groups gated on >=5 valid PAIRS. roll = 0 when ilq <= 0 (CASE ELSE 0 == np.where).
  - amihud: one count gate (non-null _ac >= 5) NULLs ami/ami_v/lix together.
  - CS beta is gated at day_gap >= 5 (risk dvix at day_gap > 5 -- they differ upstream).
  - CS/AR lags/rolling pairs are computed WITHIN each analysis subset (full vs excl-last-trade), so
    the _adj variants re-pair neighbours, not just re-aggregate.
  - p_fht_adj = 2*sigma_adj*ppf((1+p_zro_FULL)/2) -- the adjusted version reuses the FULL p_zro.
  - pi: NULL when n < MIN_OBS or |detM| <= 1e-8 (np.isclose(detM, 0)).
  - factors: USA-domiciled bonds only (FISD), EW means of the NORMAL signals; first-difference all
    but PSB and RSJ; RSJ scaled /100; first (NaN-diff) row dropped.

Outputs under output/blocks/<mode>/: illiq_signals, illiq_signals_adj, illiq_factors.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import _stage2_settings as cfg
from lib import illiq_pandas, nyse_calendar
from lib import pin as pinlib
from lib import vix as vixlib
from lib.phase_timer import PhaseTimer

MIN_OBS = cfg.ILLIQ_MIN_OBS
CS_K = 3.0 - 2.0 * np.sqrt(2.0)

# upstream factor_map (iteration order = golden column order) + differencing set
FACTOR_MAP = [("pi", "PSB"), ("ami", "AMD"), ("lix", "LIX"), ("ilq", "ILLIQ"), ("roll", "ROLL"),
              ("spd_rel", "SPRD"), ("cs_sprd", "CSS"), ("ar_sprd", "ARS"), ("p_fht", "FHTS"),
              ("vov", "VOV"), ("rvol", "RVOL"), ("rsj", "RSJ")]
DIFF_FACTORS = {"AMD", "LIX", "ILLIQ", "ROLL", "SPRD", "CSS", "ARS", "FHTS", "VOV", "RVOL"}

# merged signal-column order (upstream merge chain order), used to split normal vs _adj
SIGNAL_ORDER = ["pi", "ami", "ami_v", "lix", "ilq", "roll", "spd_abs", "spd_rel",
                "cs_sprd", "ar_sprd", "p_zro", "p_fht", "trn", "vov",
                "dvol", "dskew", "dkurt", "db_mkt", "dvol_sys", "dvol_idio",
                "rvol", "rsj", "rsk", "rkt", "b_vix", "b_dvixd"]


def _both(sql_template: str, con, out_table: str) -> None:
    """Run a per-(cusip, month) metric on the full sample and the excl-last-trade subset, then
    left-join the _adj columns on. `sql_template` must contain {where} and produce columns
    cusip_id, month_start, <metrics...>."""
    con.execute(f"CREATE OR REPLACE TEMP TABLE _full AS {sql_template.format(where='TRUE')}")
    con.execute(f"CREATE OR REPLACE TEMP TABLE _adj AS {sql_template.format(where='lst_txn = 1')}")
    metric_cols = [r[0] for r in con.execute("DESCRIBE _full").fetchall()
                   if r[0] not in ("cusip_id", "month_start")]
    adj_sel = ", ".join(f"a.{c} AS {c}_adj" for c in metric_cols)
    con.execute(f"""
CREATE OR REPLACE TEMP TABLE {out_table} AS
SELECT f.*, {adj_sel}
FROM _full f LEFT JOIN _adj a USING (cusip_id, month_start)
""")


def build(con, mode: str | None = None, limit_cusips: int | None = None) -> dict[str, Path]:
    """Build illiq_signals / illiq_signals_adj / illiq_factors blocks; returns {name: path}."""
    mode = mode or cfg.INPUT_MODE
    t0 = time.time()
    pt = PhaseTimer()
    with pt("pin"):
        pin_path = pinlib.build_pin(con, mode, limit_cusips)
    out_dir = cfg.BLOCKS_DIR / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    with pt("load_t2"):
        con.execute(f"""
CREATE OR REPLACE TEMP TABLE t2 AS
SELECT frn, cusip_id, dt, month_start, pr, prc_1st, prc_lst, prc_hi, prc_lo, prc_bid, prc_ask,
       dvol, ao, day_gap, ret_d, ret_c, ret_d_lag, ret_c_lag, dvol_lag, "Nret", lst_txn
FROM read_parquet('{pin_path.as_posix()}')
""")

    # ---------------- float32-order-sensitive metrics: verbatim pandas ports ----------------------
    # pi / amihud / ilq-roll / the risk kernel are computed in pandas (lib/illiq_pandas) because
    # their upstream float32 accumulation order and near-singular ratios cannot be reproduced by
    # parallel double-precision SQL sums (debug.md M9). The frame is pulled in (cusip, dt) order --
    # the upstream streamline sort -- so groupby accumulation order matches bit-for-bit.
    with pt("pdf_pull"):
        pdf = con.execute("""
SELECT cusip_id, dt AS trd_exctn_dt, month_start, ret_c, ret_c_lag, ret_d,
       dvol, dvol_lag, prc_lst, prc_hi, prc_lo, day_gap, lst_txn
FROM t2 ORDER BY cusip_id, dt, frn
""").df()
        pdf["trd_exctn_dt"] = pd.to_datetime(pdf["trd_exctn_dt"])
        pdf["month_year"] = pdf["trd_exctn_dt"].dt.to_period("M")

    with pt("pandas_pi"):
        m_pi = illiq_pandas.monthly_pi_fast(pdf, min_obs=MIN_OBS)
    with pt("pandas_amihud"):
        m_am = illiq_pandas.compute_monthly_amihud(pdf, min_obs=MIN_OBS)
    with pt("pandas_ilq_roll"):
        m_ilq = illiq_pandas.compute_monthly_illiq_roll_fast(pdf, min_obs=MIN_OBS)
    with pt("pandas_risk"):
        m_rsk = illiq_pandas.compute_within_month_risk(pdf, vix_df=vixlib.load_vix(),
                                                       min_obs=MIN_OBS)
    del pdf
    with pt("register_pandas"):
        for name, frame in (("m_pi", m_pi), ("m_am", m_am), ("m_ilq", m_ilq), ("m_rsk", m_rsk)):
            frame["month_start"] = frame["date"].dt.to_period("M").dt.to_timestamp()
            con.register(name, frame.drop(columns=["date"]))

    # ---------------- Hong-Warga volume-weighted spreads (no min_obs) -----------------------------
    pt_sql = pt("sql_metrics")
    pt_sql.__enter__()
    hws_tpl = """
SELECT cusip_id, month_start, spd_abs,
       CASE WHEN isinf(spd_rel) THEN NULL ELSE spd_rel END AS spd_rel
FROM (
  SELECT cusip_id, month_start,
         (wask_s / wa_s) - (wbid_s / wb_s) AS spd_abs,
         ((wask_s / wa_s) - (wbid_s / wb_s)) / (((wask_s / wa_s) + (wbid_s / wb_s)) / 2) AS spd_rel
  FROM (
    SELECT cusip_id, month_start,
           sum(coalesce(prc_bid, 0) * _wb) AS wbid_s, sum(_wb) AS wb_s,
           sum(coalesce(prc_ask, 0) * _wa) AS wask_s, sum(_wa) AS wa_s
    FROM (
      SELECT cusip_id, month_start, prc_bid, prc_ask,
             CASE WHEN prc_bid IS NOT NULL THEN _w ELSE 0 END AS _wb,
             CASE WHEN prc_ask IS NOT NULL THEN _w ELSE 0 END AS _wa
      FROM (
        SELECT *, CASE WHEN dvol IS NULL OR isnan(dvol) OR isinf(dvol) OR dvol <= 0
                       THEN 0.0 ELSE CAST(dvol AS DOUBLE) END AS _w
        FROM t2 WHERE {where}
      )
    )
    GROUP BY 1, 2
  )
)"""
    _both(hws_tpl, con, "m_hws")

    # ---------------- Corwin-Schultz + Abdi-Ranaldo spreads ---------------------------------------
    # base: hi/lo/close non-null, clipped >= 1e-12; lags pair within the SUBSET (full vs lst=1)
    for tag, where in (("f", "TRUE"), ("a", "lst_txn = 1")):
        con.execute(f"""
CREATE OR REPLACE TEMP TABLE t_csar_{tag} AS
SELECT *,
  CASE WHEN day_gap >= 5 THEN NULL ELSE h2 + lag(h2) OVER w END AS beta,
  CASE WHEN lag(hi) OVER w IS NULL THEN NULL
       ELSE ln(greatest(hi, lag(hi) OVER w) / least(lo, lag(lo) OVER w)) ^ 2 END AS gamma,
  CASE WHEN day_gap >= 5 THEN NULL ELSE lag(eta) OVER w END AS eta_lag
FROM (
  SELECT frn, cusip_id, dt, month_start, day_gap,
         greatest(prc_hi, 1e-12) AS hi, greatest(prc_lo, 1e-12) AS lo,
         greatest(prc_lst, 1e-12) AS cl,
         ln(greatest(prc_hi, 1e-12) / greatest(prc_lo, 1e-12)) ^ 2 AS h2,
         0.5 * (ln(greatest(prc_hi, 1e-12)) + ln(greatest(prc_lo, 1e-12))) AS eta
  FROM t2
  WHERE prc_hi IS NOT NULL AND prc_lo IS NOT NULL AND prc_lst IS NOT NULL AND {where}
)
WINDOW w AS (PARTITION BY cusip_id ORDER BY dt, frn)
""")
        con.execute(f"""
CREATE OR REPLACE TEMP TABLE m_csar_{tag} AS
SELECT cusip_id, month_start, avg(cs) AS cs_sprd, avg(ar) AS ar_sprd, count(cl) AS nc
FROM (
  SELECT *,
    2.0 * (exp(alpha) - 1.0) / (1.0 + exp(alpha)) AS cs,
    -- explicit NULL guard: DuckDB greatest() IGNORES NULLs (perf_learnings_carryover), so a NULL
    -- eta_lag would silently clip to 0 instead of propagating NaN like pandas
    CASE WHEN eta_lag IS NULL THEN NULL
         ELSE sqrt(greatest(4.0 * (ln(cl) - 0.5 * (eta + eta_lag)) ^ 2 - (eta - eta_lag) ^ 2, 0.0))
         END AS ar
  FROM (
    SELECT *,
      CASE WHEN raw_alpha < 0 THEN 0.0 ELSE raw_alpha END AS alpha
    FROM (
      SELECT *, (sqrt(2.0 * beta) - sqrt(beta)) / {CS_K} - sqrt(gamma) AS raw_alpha
      FROM t_csar_{tag}
    )
  )
)
GROUP BY 1, 2
""")
    con.execute(f"""
CREATE OR REPLACE TEMP TABLE m_csar AS
SELECT f.cusip_id, f.month_start,
  CASE WHEN f.nc < {MIN_OBS} THEN NULL ELSE f.cs_sprd END AS cs_sprd,
  CASE WHEN f.nc < {MIN_OBS} THEN NULL ELSE f.ar_sprd END AS ar_sprd,
  CASE WHEN coalesce(a.nc, 0) < {MIN_OBS} THEN NULL ELSE a.cs_sprd END AS cs_sprd_adj,
  CASE WHEN coalesce(a.nc, 0) < {MIN_OBS} THEN NULL ELSE a.ar_sprd END AS ar_sprd_adj
FROM m_csar_f f LEFT JOIN m_csar_a a USING (cusip_id, month_start)
""")

    # ---------------- p_zro / p_fht (pandas tail for norm.ppf) ------------------------------------
    cal = nyse_calendar.build_calendar_frame()
    sess = cal.loc[cal["is_session"], "day"]
    bdays = sess.groupby(sess.dt.to_period("M")).size().rename("bdays").reset_index()
    bdays["month_start"] = bdays["day"].dt.to_timestamp()
    con.register("bdays_tbl", bdays[["month_start", "bdays"]])
    pz = con.execute(f"""
SELECT c.cusip_id, c.month_start,
       -- explicit NULL guards: DuckDB least/greatest IGNORE NULLs, so an unguarded clip turns a
       -- NULL count into 0.0 (bit us on p_zro_adj for single-trade months -- debug.md M10)
       CASE WHEN c.npr IS NULL THEN NULL
            ELSE least(greatest((b.bdays - c.npr) / b.bdays::DOUBLE, 0), 1) END AS p_zro,
       CASE WHEN c.npr_adj IS NULL THEN NULL
            ELSE least(greatest((b.bdays - c.npr_adj) / b.bdays::DOUBLE, 0), 1) END AS p_zro_adj,
       sf.sigma, sa.sigma_adj
FROM (
  SELECT cusip_id, month_start,
         sum((pr IS NOT NULL)::INT) AS npr,
         -- groups with NO excl-last-trade rows are ABSENT upstream -> NULL, not 0
         CASE WHEN sum((lst_txn = 1)::INT) = 0 THEN NULL
              ELSE sum((pr IS NOT NULL AND lst_txn = 1)::INT) END AS npr_adj
  FROM t2 GROUP BY 1, 2
) c
JOIN bdays_tbl b USING (month_start)
LEFT JOIN (
  SELECT cusip_id, month_start, stddev_samp(ret_c) AS sigma
  FROM t2 WHERE "Nret" >= {MIN_OBS} GROUP BY 1, 2
) sf USING (cusip_id, month_start)
LEFT JOIN (
  SELECT cusip_id, month_start, stddev_samp(ret_c) AS sigma_adj
  FROM t2 WHERE lst_txn = 1 GROUP BY 1, 2
  HAVING sum((ret_c IS NOT NULL)::INT) >= {MIN_OBS}
) sa USING (cusip_id, month_start)
""").df()
    z = ((1.0 + pz["p_zro"]) / 2.0).clip(lower=1e-12, upper=1 - 1e-12)
    ppf = norm.ppf(z)
    pz["p_fht"] = 2.0 * pz["sigma"] * ppf
    pz["p_fht_adj"] = 2.0 * pz["sigma_adj"] * ppf
    con.register("m_pz", pz[["cusip_id", "month_start", "p_zro", "p_zro_adj", "p_fht", "p_fht_adj"]])

    # ---------------- turnover --------------------------------------------------------------------
    trn_tpl = f"""
SELECT cusip_id, month_start,
       least(greatest(trn * 100.0, 0.0), 100.0) AS trn
FROM (
  SELECT cusip_id, month_start, avg(dvol / (ao / 1000.0)) AS trn
  FROM t2
  WHERE ao > 0 AND dvol > 0 AND {{where}}
  GROUP BY 1, 2 HAVING count(*) >= {MIN_OBS}
)"""
    _both(trn_tpl, con, "m_trn")

    # ---------------- vov -------------------------------------------------------------------------
    vov_tpl = f"""
SELECT cusip_id, month_start,
       CASE WHEN sigma > 0 AND V > 0 THEN 2.5 * sigma ^ 0.60 / V ^ 0.25 END AS vov
FROM (
  SELECT cusip_id, month_start, stddev_samp(ret_d) AS sigma, avg(dvol) AS V
  FROM t2
  WHERE ret_d IS NOT NULL AND dvol IS NOT NULL AND dvol > 0 AND "Nret" > 0 AND {{where}}
  GROUP BY 1, 2 HAVING count(ret_d) >= {MIN_OBS}
)"""
    _both(vov_tpl, con, "m_vov")

    # ---------------- merge all signals (upstream chain order), month-end date --------------------
    con.execute("""
CREATE OR REPLACE TEMP TABLE signals AS
SELECT * FROM m_pi
FULL JOIN m_am USING (cusip_id, month_start)
FULL JOIN m_ilq USING (cusip_id, month_start)
FULL JOIN m_hws USING (cusip_id, month_start)
FULL JOIN m_csar USING (cusip_id, month_start)
FULL JOIN m_pz USING (cusip_id, month_start)
FULL JOIN m_trn USING (cusip_id, month_start)
FULL JOIN m_vov USING (cusip_id, month_start)
FULL JOIN m_rsk USING (cusip_id, month_start)
""")
    pt_sql.__exit__(None, None, None)

    blocks: dict[str, Path] = {}

    def _copy(name: str, sql: str) -> None:
        p = out_dir / f"{name}.parquet"
        with pt("copy_blocks"):
            con.execute(f"COPY ({sql}) TO '{p.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        blocks[name] = p

    # NaN -> NULL at the block boundary: pandas NaN means "missing" everywhere upstream, while
    # DuckDB propagates NaN as a value through 0/0 divisions AND through avg() (the SPRD factor
    # was 100% NaN before this -- debug.md M9).
    def _n2n(c: str) -> str:
        return f"CASE WHEN isnan({c}) THEN NULL ELSE {c} END AS {c}"

    normal_sel = ", ".join(_n2n(c) for c in SIGNAL_ORDER)
    adj_sel = ", ".join(_n2n(f"{c}_adj") for c in SIGNAL_ORDER)
    _copy("illiq_signals", f"""
        SELECT cusip_id, last_day(month_start)::TIMESTAMP AS date, {normal_sel}
        FROM signals ORDER BY cusip_id, month_start""")
    _copy("illiq_signals_adj", f"""
        SELECT cusip_id, last_day(month_start)::TIMESTAMP AS date, {adj_sel}
        FROM signals ORDER BY cusip_id, month_start""")

    # ---------------- nontraded factors: USA bonds, EW means, differencing ------------------------
    # In PANDAS with the signals' native dtypes: upstream means accumulate float32 for the
    # float32 signals (AMD/LIX/ILLIQ/ROLL come out float32 in the golden factor file); a SQL
    # double-precision avg differs by ~3e-6 (debug.md M9).
    with pt("factors"):
        fisd = pd.read_parquet(cfg.AUX["fisd"], columns=["complete_cusip", "country_domicile"])
        usa = fisd.loc[fisd["country_domicile"] == "USA", ["complete_cusip"]].drop_duplicates()
        sig_pd = pd.read_parquet(blocks["illiq_signals"],
                                 columns=["cusip_id", "date"] + [s for s, _ in FACTOR_MAP])
        sig_usa = sig_pd.merge(usa, left_on="cusip_id", right_on="complete_cusip", how="inner")
        fac = None
        for sig_col, fac_name in FACTOR_MAP:
            f = sig_usa.groupby("date", observed=True)[sig_col].mean().rename(fac_name).reset_index()
            fac = f if fac is None else fac.merge(f, on="date", how="outer")
        fac = fac.sort_values("date").reset_index(drop=True)
        for _, name in FACTOR_MAP:
            if name in DIFF_FACTORS:
                fac[name] = fac[name].diff()
        fac["RSJ"] = fac["RSJ"] / 100
        fac = fac.iloc[1:].reset_index(drop=True)
        fac_path = out_dir / "illiq_factors.parquet"
        fac.to_parquet(fac_path, index=False)
        blocks["illiq_factors"] = fac_path

    (out_dir / "step2_meta.json").write_text(json.dumps({
        "wall_s": round(time.time() - t0, 2), "mode": mode, "limit_cusips": limit_cusips,
        "phases": pt.phases,
        "rows": {n: int(con.execute(f"SELECT count(*) FROM read_parquet('{p.as_posix()}')").fetchone()[0])
                 for n, p in blocks.items()},
    }, indent=1))
    return blocks
