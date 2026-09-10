"""check_external_data.py -- the external-data inventory CHECK for the monthly panel.

Every external series the monthly build consumes, in ONE registry: what it is, where it comes from
(WRDS table or public URL), which local cache holds it, and -- stamped live from the cache each run --
the LAST month we actually have. Run it to see whether any source has gone stale relative to the
panel, or before an extension to know what will (and won't) advance.

    check_external_data.py                 # table from local caches (fast, offline)
    check_external_data.py --live          # ALSO probe WRDS/DB for the current upstream frontier
    check_external_data.py --json-out inventory.json

The human-readable index (with provenance notes + refresh recipe) is context/external_data_inventory.md;
this script is what keeps its "last available" column honest. See _stage2_settings.MODE_PINS for how these
vintages become the panel's date frontier (debug.md M13 / assumptions.md A19).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import duckdb

import _stage2_settings as cfg

# --- the registry: one row per external series the monthly build relies on ------------------------
# kind: 'wrds' (live DB fetch, cached) | 'web' (public URL, cached) | 'pinned' (golden-vintage file)
# provider: the WRDS table or the config attr holding the URL. cache: local parquet we read.
FC = cfg.FACTOR_CACHE_DIR
D = cfg.DATA_DIR
SOURCES = [
    # name, kind, used_for, provider (WRDS table / URL attr), cache path, date column (None=rows only)
    ("CRSP treasury returns", "wrds", "tret / ret_vwx (duration-matched Treasury return)",
     "crsp.tfz_idx + crsp.tfz_mth_ft + ff.factors_monthly", D / "crsp_treasury_returns.parquet", "date"),
    ("FF5 factors + RF (pinned)", "wrds", "mktrf/smb/hml/rf; RF for BBW MKTB",
     "ff.fivefactors_monthly", D / "ff5_factors.parquet", "date"),
    ("VIX (daily, pinned)", "wrds", "dvix / vix risk factor",
     "cboe.cboe", D / "cboe_vix.parquet", "date"),
    ("FF5 factors (public path)", "web", "mktrf/smb/hml/rf when --factor-source public",
     "FF5_URL", FC / "ff5_french.parquet", "date"),
    ("VIX (monthly, public path)", "wrds", "vix/vxo when --factor-source public",
     "cboe.cboe", FC / "vix_monthly.parquet", "date"),
    ("FRED CPI + credit + yields", "web", "dcpi/cpi_vol6, credit(AAA/BAA), Treasury-yield level/slope",
     "FRED_CSV_URL (CPIAUCSL,AAA,BAA,DGS1..DGS30)", FC / "fred_cpi_credit.parquet", "date"),
    ("HKM intermediary capital", "web", "cptl/cptlt intermediary-capital factor",
     "HKM_URL", FC / "hkm_intermediary.parquet", "date"),
    ("Ludvigson uncertainty", "web", "unc/uncf/uncr macro-uncertainty factors",
     "LUDVIGSON_URL (via LUDVIGSON_INDEX)", FC / "ludvigson_uncertainty.parquet", "date"),
    ("EPU categorical", "web", "epu/epum/eput policy-uncertainty factors",
     "EPU_URL", FC / "epu_categorical.parquet", "date"),
    ("OSBAP extended BBW (pre-2002)", "web", "pre-2002-08 BBW factor backfill (published series)",
     "BBW_EXTENDED_URL", D / "bbw_factors_extended_1973_2023.parquet", "date"),
    ("Quote returns 1997-2002", "web", "pre-2002-07 quote-based returns for BBW VaR / value",
     "QUOTE_URL", D / "quote_returns_quantlib.parquet", "date"),
    ("NYSE session calendar", "web", "business-day gaps / month-end sessions (mcal-generated)",
     "pandas_market_calendars NYSE", D / "nyse_calendar.parquet", None),
    # golden-vintage pinned AUX files (consumed by step1 wrangle; stamp = filename vintage)
    ("OSBAP CUSIP-PERMNO linker", "pinned", "wrangle_returns identifier linker",
     "AUX['linker'] (golden vintage)", cfg.AUX["linker"], None),
    ("Moody's ratings", "pinned", "composite rating (mdy_rat)",
     "AUX['moody'] (golden vintage)", cfg.AUX["moody"], None),
    ("S&P ratings", "pinned", "composite rating (sp_rat)",
     "AUX['sp'] (golden vintage)", cfg.AUX["sp"], None),
    ("Call dummy", "pinned", "callable-bond flag",
     "AUX['call'] (golden vintage)", cfg.AUX["call"], None),
    ("FISD issue attributes", "pinned", "144a/country/call/sic bond attributes",
     "AUX['fisd'] (golden vintage)", cfg.AUX["fisd"], None),
    ("Pinned factor panel", "pinned", "the assembled factor time series (default --factor-source pinned)",
     "GOLDEN_OUTPUTS['factors']", cfg.GOLDEN_OUTPUTS["factors"], "date"),
]

# WRDS tables to probe with --live (table -> date column) for the CURRENT upstream frontier
_LIVE_WRDS = {
    "crsp.tfz_mth_ft": "mcaldt", "ff.factors_monthly": "dateff",
    "ff.fivefactors_monthly": "dateff", "cboe.cboe": "date",
}


def _cache_stamp(path: Path, date_col: str | None) -> dict:
    if not path.exists():
        return {"exists": False, "last": None, "rows": None, "mb": None}
    con = duckdb.connect()
    p = str(path).replace("\\", "/")
    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{p}')").fetchone()[0]
    last = None
    if date_col:
        cols = con.execute(f"SELECT * FROM read_parquet('{p}') LIMIT 0").df().columns.tolist()
        if date_col in cols:
            last = con.execute(f'SELECT MAX("{date_col}") FROM read_parquet(\'{p}\')').fetchone()[0]
            last = str(last)[:10] if last is not None else None
    return {"exists": True, "last": last, "rows": int(rows),
            "mb": round(path.stat().st_size / 1e6, 1)}


def _sidecar(path: Path) -> dict:
    """Read the factor_fetch sidecar meta (url + fetched_utc) if present."""
    meta = path.with_suffix(".meta.json")
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            return {"fetched_utc": m.get("fetched_utc"), "url": m.get("url")}
        except Exception:
            pass
    return {}


def build(live: bool = False) -> list[dict]:
    rows = []
    for name, kind, use, provider, cache, dcol in SOURCES:
        st = _cache_stamp(Path(cache), dcol)
        rows.append({"source": name, "kind": kind, "used_for": use, "provider": provider,
                     "cache": str(cache), **st, **_sidecar(Path(cache))})
    if live:
        import time
        t0 = time.time()
        try:
            import wrds
            _u = os.environ.get("WRDS_USERNAME", "")
            if not _u:
                raise RuntimeError("WRDS_USERNAME is not set. Stage 2 needs it only for its first run, to fetch and cache Treasury returns, Fama-French factors and VIX. Set it in config.py or as an environment variable.")
            db = wrds.Connection(wrds_username=_u)
            try:
                for tbl, dc in _LIVE_WRDS.items():
                    try:
                        mx = db.raw_sql(f"SELECT MAX({dc}) mx FROM {tbl}")["mx"].iloc[0]
                        rows.append({"source": f"[live] {tbl}", "kind": "wrds-live",
                                     "provider": tbl, "last": str(mx)[:10]})
                    except Exception as e:
                        rows.append({"source": f"[live] {tbl}", "last": f"ERR {str(e)[:40]}"})
            finally:
                db.close()
        except Exception as e:
            rows.append({"source": "[live] WRDS", "last": f"CONNECT_FAILED {str(e)[:60]}"})
        # local DB TRACE frontier (the ultimate daily-input source)
        try:
            dbp = str(cfg.REPO / "raw_data" / "wrds_trace.duckdb").replace("\\", "/")
            c = duckdb.connect(dbp, read_only=True)
            for t in ("trace_enhanced", "trace_btds144a"):
                mx = c.execute(f"SELECT MAX(CAST(trd_exctn_dt AS DATE)) FROM {t}").fetchone()[0]
                rows.append({"source": f"[live] DB {t}", "kind": "db", "last": str(mx)[:10]})
        except Exception as e:
            rows.append({"source": "[live] DB", "last": f"ERR {str(e)[:50]}"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="monthly external-data inventory check")
    ap.add_argument("--live", action="store_true", help="also probe WRDS + the local DB for the upstream frontier")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    rows = build(args.live)
    w = max(len(r["source"]) for r in rows)
    print(f"{'SOURCE':{w}}  {'KIND':9}  {'LAST':10}  {'ROWS':>10}  PROVIDER")
    for r in rows:
        print(f"{r['source']:{w}}  {r.get('kind',''):9}  {str(r.get('last')):10}  "
              f"{('' if r.get('rows') is None else format(r['rows'],',')):>10}  {r.get('provider','')}")
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=1, default=str))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
