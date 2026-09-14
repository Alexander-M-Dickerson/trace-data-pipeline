"""duration_adjusted.py -- Treasury benchmark returns built from each bond's own cash flows.

Adds four columns beside the incumbent `tret`, each a Treasury benchmark you can subtract from
`ret_vw` to get a duration-adjusted (credit) return:

    tret_bns    present-value (duration) weights on the bond's own cash flows
                van Binsbergen, J.H., Nozawa, Y. and Schwert, M., "Duration-Based Valuation of
                Corporate Bonds", SSRN 3914422, Eqs. (3)-(6).
    tret_cfm    the same ladder, future-value weights
                same paper, Internet Appendix A1, Eq. (19). Over-weights long-dated cash flows and
                so overstates interest-rate risk, which is why the authors prefer PV weighting.
    tret_gprs   the EXACT duration match
                Ghaderi, M., Plante, S., Roussanov, N. and Seo, S.B. (2026), "Reconstructing a
                Century of U.S. Corporate Bonds: Credit Risk in Historical Perspective", App. B.2.
    tret_mat    key-rate index returns interpolated at MATURITY rather than duration
                Bessembinder, H., Kahle, K., Maxwell, W. and Xu, D. (2009). The twin of `tret`,
                which interpolates the same series at modified duration (Andreani, M., Palhares, D.
                and Richardson, S. (2024), Rev Account Stud 29, 3887-3906, Eq. (1)).

THE METHOD. For bond i at t with cash flows CF_k at times t_k and own yield y compounded f times a
year, and z_t(.) the Gurkaynak-Sack-Wright zero curve:

    w_k         = PV_k / SUM_j PV_j ,     PV_k = CF_k / (1 + y/f)^(f*t_k)
    tret_bns    = SUM_k w_k * r_k
    r_k         = exp( t_k*z_t(t_k) - t'_k*z_{t+1}(t'_k) ) - 1 ,   t'_k = max(t_k - delta, 0)

t_k is measured from the START of the return window and t'_k from its END, so t'_k = t_k - delta
falls out of the dates. t'_k = 0 handles a cash flow paid inside the window: that zero matured, so
its leg is held to maturity. tret_cfm swaps w_k for CF_k / SUM_j CF_j. tret_gprs solves
SUM_k w_k*exp((y*-z_k)t_k) = 1 for the Treasury yield y* and reweights by w_k*exp((y*-z_k)t_k).

TWO INPUTS THIS REPO DID NOT HAVE, both in stage2/data/ beside crsp_treasury_returns.parquet:
  gsw_svensson.parquet        the Fed's curve parameters (feds200628), plus its published SVENYnn
                              columns, which tell us the longest tenor GSW actually ANCHORED each
                              day. Beyond that tenor the yield is held FLAT -- the convention of
                              Gurkaynak-Sack-Wright and of Ghaderi et al. (their App. A: "we
                              conservatively apply flat extrapolation when necessary"). This matters
                              only before 1986, when no Treasury long enough existed to price.
  fisd_cashflow_terms.parquet first_interest_date / last_interest_date / coupon_change_indicator,
                              which the stage0 FISD extract does not carry. Coupon, frequency and
                              maturity still come from the stage0 extract; this file supplies only
                              the schedule anchors, so the two vintages cannot disagree on terms.

TWO TRAPS IN THE GSW FILE, both load-bearing:
  1. TAU2 = -999.99 is a SENTINEL, not a parameter. BETA3 is exactly 0 on every such row, so the
     fourth Svensson term must be switched OFF, not evaluated: n/TAU2 with a negative TAU2 returns
     a finite, wrong number rather than raising.
  2. Recent rows carry BETA2 ~ +3475 against BETA3 ~ -3476 with TAU1 ~ TAU2 -- two huge terms that
     nearly cancel. float64 throughout.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import _stage2_settings as cfg

# Same pattern as treasury.TREASURY_CACHE: the module resolves its own inputs off cfg.DATA_DIR.
GSW_FILE = cfg.DATA_DIR / "gsw_svensson.parquet"
TERMS_FILE = cfg.DATA_DIR / "fisd_cashflow_terms.parquet"

# Day-count for time-to-cash-flow. Eq. (3) states 30/360 for the weights and Eq. (5) "actual time
# in years" for the Treasury leg. Measured against the authors' own published estimates on
# 1,227,867 bond-months: ACT/365 gives a median absolute error of 0.18 bps, 30/360 gives 0.50 bps.
YEAR_BASIS = 365.0

# A bond-month is usable only if its window is a real ~1 month. This also does the work of
# guaranteeing the lagged yield below belongs to the row it is used on.
WINDOW_DAYS_LO, WINDOW_DAYS_HI = 15, 45

NEW_COLS = ("tret_bns", "tret_cfm", "tret_gprs", "tret_mat")

# Newton steps for the tret_gprs root-find. The equation is strictly increasing in y*, so six is
# generous; measured max residual |SUM omega - 1| = 7.8e-16.
NEWTON_STEPS = 6

# Longest tenor GSW actually anchor, derived from the first non-null of each published SVENYnn.
_ANCHOR_STEPS = ((30.0, "SVENY30"), (25.0, "SVENY25"), (20.0, "SVENY20"),
                 (15.0, "SVENY15"), (10.0, "SVENY10"), (5.0, "SVENY05"))

_STEP_CASE = "CASE freq WHEN '1' THEN 12 WHEN '2' THEN 6 WHEN '4' THEN 3 WHEN '12' THEN 1 END"

# FISD's date tail is wild -- stale first_interest_dates reach back to 1930 and out past 2120.
_CF_LO, _CF_HI = "1960-01-01", "2130-01-01"

REQUIRED_COLS = ("cusip_id", "dt_s", "dt", "ytm", "mod_dur", "bond_maturity")


GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
_GSW_SKIP = 9   # the Fed prepends a provenance preamble before the real header row

# first_interest_date anchors the coupon grid on FISD's own dated cycle; last_interest_date caps it;
# coupon_change_indicator marks step-ups, whose dates are right but whose constant amount is not.
# Coupon / frequency / maturity come along only so a consumer can assert this file agrees with the
# stage0 extract it augments -- the extract remains the source of truth for a bond's terms.
_FISD_TERMS_SQL = """
    SELECT complete_cusip, first_interest_date, last_interest_date,
           coupon_change_indicator, coupon, interest_frequency, maturity
    FROM fisd.fisd_mergedissue
"""


def fetch_gsw(dest: Path | None = None) -> Path:
    """Download the Fed's GSW curve file. Public, no credentials."""
    import io
    import requests
    dest = Path(dest or GSW_FILE)
    r = requests.get(GSW_URL, timeout=300)
    r.raise_for_status()
    lines = r.text.split("\n")
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Date,BETA0"))
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), na_values=["NA"])
    keep = ["Date", "BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
    # The SVENYnn columns are kept because their first non-null date is what tells us the longest
    # tenor GSW actually ANCHORED, which is the whole basis of the flat-extrapolation policy.
    df = df[keep + [c for c in df.columns if c.startswith("SVENY")]]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["BETA0"].notna()]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False, compression="zstd")
    return dest


def fetch_fisd_terms(dest: Path | None = None, wrds_username: str | None = None) -> Path:
    """One-time WRDS fetch of the coupon-schedule anchors, cached. Same pattern as
    `treasury.fetch_treasury_returns`: the series is historical and stable."""
    import os

    import wrds
    dest = Path(dest or TERMS_FILE)
    user = wrds_username or os.environ.get("WRDS_USERNAME", "")
    if not user:
        raise RuntimeError(
            "WRDS_USERNAME is not set. duration_adjusted needs it only for its first run, to "
            f"fetch {dest.name}; afterwards the cached file is used.")
    db = wrds.Connection(wrds_username=user)
    try:
        df = db.raw_sql(_FISD_TERMS_SQL)
    finally:
        db.close()
    for c in ("first_interest_date", "last_interest_date", "maturity"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["complete_cusip"] = df["complete_cusip"].astype("string")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False, compression="zstd")
    return dest


def ensure_inputs(gsw_path: Path | None = None, terms_path: Path | None = None) -> None:
    """Fetch either cached input if it is missing, so a fresh clone builds without manual steps."""
    gsw_path = Path(gsw_path or GSW_FILE)
    terms_path = Path(terms_path or TERMS_FILE)
    if not gsw_path.exists():
        print(f"  duration_adjusted: fetching {gsw_path.name} from the Federal Reserve ...")
        fetch_gsw(gsw_path)
    if not terms_path.exists():
        print(f"  duration_adjusted: fetching {terms_path.name} from WRDS (one time) ...")
        fetch_fisd_terms(terms_path)


def sven_macro_sql() -> str:
    """Svensson zero yield, DECIMAL and continuously compounded, flat beyond `cap`.

    The internal alias is `_svx`, not `x`: a macro's inner names resolve in the CALLER's scope, so a
    plain `x` silently captures any caller whose own CTE or column is named `x`, and the binder
    error names neither side.
    """
    return """
    CREATE OR REPLACE TEMP MACRO sven(n, cap, b0, b1, b2, b3, t1, t2) AS (
      (WITH _sv AS (SELECT least(n, cap) AS _svx)
       SELECT (b0
             + b1 * (1 - exp(-_svx/t1)) / (_svx/t1)
             + b2 * ((1 - exp(-_svx/t1)) / (_svx/t1) - exp(-_svx/t1))
             + CASE WHEN t2 > 0
                    THEN b3 * ((1 - exp(-_svx/t2)) / (_svx/t2) - exp(-_svx/t2))
                    ELSE 0.0 END) / 100.0
       FROM _sv))
    """


def _anchor_case(gsw_path: Path) -> str:
    """Per-date longest anchored tenor, from the first date each published SVENYnn appears."""
    have = pd.read_parquet(gsw_path, columns=["Date"]).shape[0]
    if not have:
        raise RuntimeError(f"{gsw_path} is empty")
    cols = pd.read_parquet(gsw_path).columns
    whens = []
    for yrs, col in _ANCHOR_STEPS:
        if col in cols:
            whens.append(f'WHEN "{col}" IS NOT NULL THEN {yrs}')
    if not whens:
        raise RuntimeError(f"{gsw_path} carries no SVENYnn columns; cannot derive anchored tenor")
    return "CASE " + " ".join(whens) + " ELSE NULL END"


def build_curve(con, gsw_path: Path, table: str = "da_curve") -> None:
    """One row per fitted day: the Svensson parameters and the longest anchored tenor.

    Unfitted days are dropped rather than carried, so the ASOF join lands on the last day that
    actually HAS a curve -- necessary because most calendar month-ends are weekends or holidays.
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        SELECT CAST("Date" AS DATE) AS d,
               CAST(BETA0 AS DOUBLE) b0, CAST(BETA1 AS DOUBLE) b1,
               CAST(BETA2 AS DOUBLE) b2, CAST(BETA3 AS DOUBLE) b3,
               CAST(TAU1  AS DOUBLE) t1, CAST(TAU2  AS DOUBLE) t2,
               {_anchor_case(gsw_path)} AS anchor_yrs
        FROM read_parquet('{gsw_path.as_posix()}')
        WHERE BETA0 IS NOT NULL AND TAU1 > 0
    """)
    n = con.execute(f"SELECT COUNT(*) FROM {table} WHERE anchor_yrs IS NOT NULL").fetchone()[0]
    if n < 1000:
        raise RuntimeError(f"GSW curve has only {n} usable days -- check {gsw_path}")


def build_cashflows(con, fisd_path: Path, terms_path: Path, table: str = "da_cf") -> None:
    """Whole-life cash-flow ladder per cusip: coupons, the principal, and a zero-coupon branch.

    Terms (coupon, frequency, maturity) come from the stage0 FISD extract; only the schedule
    ANCHORS come from the WRDS terms file, so a vintage difference between the two cannot change a
    bond's economics. Two coupon grids, preferring the first:
      A  forward from first_interest_date, capped at coalesce(last_interest_date, maturity).
         Authoritative on cycle PHASE.
      B  backward from maturity, used where first_interest_date is missing (about 7% of the
         extract's bonds -- FISD did not record it for older issues).
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _da_bonds AS
        SELECT f.complete_cusip                              AS cusip,
               TRY_CAST(f.coupon AS DOUBLE)                  AS coupon,
               CAST(f.interest_frequency AS VARCHAR)         AS freq,
               TRY_CAST(f.maturity AS DATE)                  AS mat,
               TRY_CAST(f.dated_date AS DATE)                AS dated,
               TRY_CAST(f.offering_date AS DATE)             AS issued,
               f.coupon_type                                 AS ct,
               CAST(t.first_interest_date AS DATE)           AS fid,
               CAST(t.last_interest_date  AS DATE)           AS lid
        FROM read_parquet('{fisd_path.as_posix()}') f
        LEFT JOIN read_parquet('{terms_path.as_posix()}') t
               ON t.complete_cusip = f.complete_cusip
        WHERE TRY_CAST(f.maturity AS DATE) IS NOT NULL
          AND ( (f.coupon_type = 'F' AND TRY_CAST(f.coupon AS DOUBLE) > 0
                 AND CAST(f.interest_frequency AS VARCHAR) IN ('1','2','4','12'))
               OR f.coupon_type = 'Z' )
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        WITH bx AS (
            SELECT *, {_STEP_CASE} AS step,
                   coalesce(lid, mat) AS endd,
                   coalesce(dated, issued, CAST(mat - INTERVAL '40' YEAR AS DATE)) AS startd
            FROM _da_bonds
        ),
        ga AS (
            SELECT cusip, coupon / CAST(freq AS DOUBLE) AS amt, CAST(freq AS DOUBLE) AS f,
                   CAST(fid + (k * step) * INTERVAL '1' MONTH AS DATE) AS cfd
            FROM bx, unnest(range(0, CAST(datediff('month', fid, endd) / step AS INT) + 2)) AS t(k)
            WHERE ct = 'F' AND fid IS NOT NULL
        ),
        gb AS (
            SELECT cusip, coupon / CAST(freq AS DOUBLE) AS amt, CAST(freq AS DOUBLE) AS f,
                   CAST(mat - (k * step) * INTERVAL '1' MONTH AS DATE) AS cfd
            FROM bx, unnest(range(0, CAST(datediff('month', startd, mat) / step AS INT) + 2)) AS t(k)
            WHERE ct = 'F' AND fid IS NULL
        ),
        coup AS (
            SELECT g.* FROM ga g JOIN bx ON bx.cusip = g.cusip AND g.cfd <= bx.endd
            UNION ALL
            SELECT g.* FROM gb g JOIN bx ON bx.cusip = g.cusip AND g.cfd > bx.startd
        ),
        prin AS (
            SELECT cusip, 100.0 AS amt,
                   CASE WHEN freq IN ('1','2','4','12') THEN CAST(freq AS DOUBLE) ELSE 2.0 END AS f,
                   mat AS cfd
            FROM bx
        )
        SELECT cusip, cfd, amt, f
        FROM (SELECT * FROM coup UNION ALL SELECT * FROM prin)
        WHERE cfd BETWEEN DATE '{_CF_LO}' AND DATE '{_CF_HI}' AND amt > 0
    """)


def _ladder_sql(table: str, date_col: str, daily_path: Path) -> str:
    """(bond-month x remaining cash flow), materialised.

    The discount yield is the bond's own YTM observed on `dt_s` -- the day the return window opens,
    which is the day whose price starts the return. It is read from the DAILY stage-1 file at that
    exact date, not lagged off the monthly frame: `dt_s` is the previous row's `dt`, so `lag(ytm)`
    is usually the same number, but the monthly frame has already been filtered (hprd > 0, start
    date) and a dropped predecessor silently costs the row. Measured: the lag reaches 89.3% of
    bond-months, the direct lookup 99.996%. `y0_lag` remains as the fallback where the daily file
    has no row, and the window screen still guards the pairing.

    Materialising matters: a DuckDB macro re-evaluates the expression passed to an argument it
    references more than once, so tk/tk2 must be COLUMNS before `sven` sees them.
    """
    yb = YEAR_BASIS
    return f"""
    WITH pm AS (
        SELECT cusip_id, {date_col} AS d,
               CAST(dt_s AS DATE) AS a0, CAST(dt AS DATE) AS a1,
               lag(ytm) OVER w AS y0_lag,
               mod_dur, bond_maturity
        FROM {table}
        WINDOW w AS (PARTITION BY cusip_id ORDER BY {date_col})
    ),
    s1 AS (
        SELECT cusip_id, CAST(trd_exctn_dt AS DATE) AS a0, ytm AS y0_dts
        FROM read_parquet('{daily_path.as_posix()}')
        SEMI JOIN (SELECT DISTINCT cusip_id AS c, a0 AS d0 FROM pm) k
               ON k.c = cusip_id AND k.d0 = CAST(trd_exctn_dt AS DATE)
    ),
    pmf AS (
        SELECT pm.*, coalesce(s1.y0_dts, pm.y0_lag) AS y0
        FROM pm LEFT JOIN s1 USING (cusip_id, a0)
        WHERE pm.a0 IS NOT NULL AND pm.a1 IS NOT NULL
          AND coalesce(s1.y0_dts, pm.y0_lag) IS NOT NULL
          AND date_diff('day', pm.a0, pm.a1) BETWEEN {WINDOW_DAYS_LO} AND {WINDOW_DAYS_HI}
    ),
    j AS (
        SELECT pmf.*,
               c0.b0 ab0, c0.b1 ab1, c0.b2 ab2, c0.b3 ab3, c0.t1 at1, c0.t2 at2,
               c0.anchor_yrs acap,
               c1.b0 zb0, c1.b1 zb1, c1.b2 zb2, c1.b3 zb3, c1.t1 zt1, c1.t2 zt2,
               c1.anchor_yrs zcap
        FROM pmf
        ASOF LEFT JOIN da_curve c0 ON pmf.a0 >= c0.d
        ASOF LEFT JOIN da_curve c1 ON pmf.a1 >= c1.d
    ),
    leg AS (
        SELECT j.cusip_id, j.d, j.acap,
               date_diff('day', j.a0, cf.cfd) / {yb}                        AS tk,
               greatest(date_diff('day', j.a1, cf.cfd) / {yb}, 0.0)         AS tk2,
               cf.amt                                                        AS fv,
               cf.amt / pow(1 + j.y0 / cf.f,
                            cf.f * (date_diff('day', j.a0, cf.cfd) / {yb}))  AS pv,
               j.ab0, j.ab1, j.ab2, j.ab3, j.at1, j.at2,
               j.zb0, j.zb1, j.zb2, j.zb3, j.zt1, j.zt2, j.zcap
        FROM j JOIN da_cf cf ON cf.cusip = j.cusip_id AND cf.cfd > j.a0
        WHERE j.acap IS NOT NULL AND j.zcap IS NOT NULL
    ),
    z AS (
        SELECT *,
               sven(tk,  acap, ab0, ab1, ab2, ab3, at1, at2) AS z0,
               sven(tk2, zcap, zb0, zb1, zb2, zb3, zt1, zt2) AS z1
        FROM leg WHERE tk > 0
    )
    SELECT cusip_id, d, tk, fv, pv, z0,
           pv / SUM(pv) OVER (PARTITION BY cusip_id, d)                       AS w,
           exp(tk * z0 - CASE WHEN tk2 > 0 THEN tk2 * z1 ELSE 0.0 END) - 1.0   AS rk
    FROM z
    """


def _gprs_statements(ladder: str = "da_lad", steps: int = NEWTON_STEPS) -> list[str]:
    """The exact duration match, as a scalar root-find per bond-month.

    Ghaderi et al. give a recursion over the Treasury cash-flow vector. It collapses: substituting
    their update CF_k = w_k*P*exp(y*t_k) into P = SUM_k CF_k*exp(-z_k*t_k) cancels P and leaves

        SUM_k w_k * exp((y* - z_k) * t_k) = 1

    with weights omega_k = w_k*exp((y* - z_k)*t_k), which sum to 1 by that equation. Strictly
    increasing in y*, so Newton from the PV-weighted mean zero converges in a few steps. On a flat
    curve y* = z and omega_k = w_k, the degenerate case they identify.
    """
    out = [f"""
        CREATE OR REPLACE TEMP TABLE da_y AS
        SELECT cusip_id, d, SUM(w * z0) AS y FROM {ladder} GROUP BY cusip_id, d
    """]
    for _ in range(steps):
        out.append(f"""
        CREATE OR REPLACE TEMP TABLE da_y AS
        WITH g AS (
            SELECT l.cusip_id, l.d,
                   SUM(l.w * exp((y.y - l.z0) * l.tk))        AS s,
                   SUM(l.w * l.tk * exp((y.y - l.z0) * l.tk)) AS ds
            FROM {ladder} l JOIN da_y y USING (cusip_id, d)
            GROUP BY l.cusip_id, l.d
        )
        SELECT y.cusip_id, y.d, y.y - (g.s - 1.0) / g.ds AS y
        FROM da_y y JOIN g USING (cusip_id, d)
        """)
    return out


def _attach_tret_mat(con, table: str, date_col: str, treasury_mod, mode: str | None) -> None:
    """tret_mat: the incumbent `tret` machinery fed MATURITY instead of modified duration.

    Deliberately identical to `_attach_tret` in every other respect -- same function, same nodes,
    same flat ends, same 2dp rounding of the argument -- so the pair isolates the duration-vs-
    maturity choice and nothing else. Attached at the same point in the window as `tret`, not
    lagged, for the same reason.
    """
    dftret = treasury_mod.load_tret_wide(mode=mode)
    keys = con.execute(
        f"SELECT cusip_id, {date_col} AS date, bond_maturity FROM {table}").df()
    keys["date"] = pd.to_datetime(keys["date"])
    keys["mod_dur"] = keys["bond_maturity"].astype(float).round(2)
    pairs = keys[["date", "mod_dur"]].drop_duplicates().reset_index(drop=True)
    got = treasury_mod.interpolate_tret(pairs, dftret)
    keys = keys.merge(got, on=["date", "mod_dur"], how="left").rename(columns={"tret": "tret_mat"})
    con.register("_da_mat", keys[["cusip_id", "date", "tret_mat"]])
    con.execute("""
        CREATE OR REPLACE TEMP TABLE da_mat AS SELECT * FROM _da_mat
    """)
    con.unregister("_da_mat")


def attach(con, table: str, date_col: str, out_table: str, *,
           treasury_mod, mode: str | None = None,
           gsw_path: Path | None = None, fisd_path: Path | None = None,
           terms_path: Path | None = None, daily_path: Path | None = None) -> dict:
    """Add tret_bns / tret_cfm / tret_gprs / tret_mat to `table`, writing `out_table`.

    Mirrors `_attach_tret`: keyed exactly by (cusip_id, {date_col}), LEFT JOINed so no row is lost.
    Returns a small dict of counts for the build log.
    """
    gsw_path = Path(gsw_path or GSW_FILE)
    terms_path = Path(terms_path or TERMS_FILE)
    fisd_path = Path(fisd_path or cfg.AUX["fisd"])

    have = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()}
    missing = [c for c in REQUIRED_COLS if c not in have]
    if missing:
        raise RuntimeError(f"{table} is missing {missing}; duration_adjusted.attach needs {list(REQUIRED_COLS)}")
    if not Path(fisd_path).exists():
        raise RuntimeError(f"FISD extract not found: {fisd_path}")

    ensure_inputs(gsw_path, terms_path)
    con.execute(sven_macro_sql())
    build_curve(con, Path(gsw_path))
    build_cashflows(con, Path(fisd_path), Path(terms_path))
    con.execute(f"CREATE OR REPLACE TEMP TABLE da_lad AS {_ladder_sql(table, date_col, Path(daily_path or cfg.daily_input()))}")
    for stmt in _gprs_statements():
        con.execute(stmt)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE da_bench AS
        SELECT l.cusip_id, l.d,
               SUM(l.w * l.rk)                                AS tret_bns,
               SUM(l.fv * l.rk) / SUM(l.fv)                   AS tret_cfm,
               SUM(l.w * exp((y.y - l.z0) * l.tk) * l.rk)     AS tret_gprs
        FROM da_lad l JOIN da_y y USING (cusip_id, d)
        GROUP BY l.cusip_id, l.d
    """)
    _attach_tret_mat(con, table, date_col, treasury_mod, mode)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {out_table} AS
        SELECT t.*, b.tret_bns, b.tret_cfm, b.tret_gprs, m.tret_mat
        FROM {table} t
        LEFT JOIN da_bench b ON b.cusip_id = t.cusip_id AND b.d = t.{date_col}
        LEFT JOIN da_mat   m ON m.cusip_id = t.cusip_id AND m.date = t.{date_col}
    """)
    n, nb, nm = con.execute(
        f"SELECT COUNT(*), COUNT(tret_bns), COUNT(tret_mat) FROM {out_table}").fetchone()
    if nb == 0:
        raise RuntimeError("duration_adjusted produced no tret_bns at all -- check inputs")
    return {"rows": int(n), "tret_bns": int(nb), "tret_mat": int(nm)}
