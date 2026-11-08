#!/usr/bin/env python
"""Phase 2 — scan unflagged 2-min SPOC stars in a TESS sector for new transits.

* Builds or refreshes an exclusion list of known TOIs + confirmed planets
  (data/known_planet_tics.csv).
* Lists every 2-min SPOC light curve in the sector with one MAST query
  (cached in data/sector_targets_s<N>.csv).
* Removes known and already-scanned stars, draws a seeded random sample, and
  processes it in parallel: download PDCSAP → search → vet. Non-candidates' FITS
  files are deleted right away. Each candidate gets its TPF downloaded and a
  4-panel figure rendered.
* Every star goes to output/scan_log_s<N>.csv (so a re-run resumes where it
  stopped). Candidates go to output/new_candidates.csv, ranked by SNR.

Usage:
  .venv/bin/python src/hunt_new_planets.py --sector 14 --limit 25 --workers 6 --snr-threshold 7.5
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import (  # noqa: E402
    CACHE_DIR,
    CANDIDATES_CSV,
    DATA_DIR,
    DEFAULT,
    KNOWN_TICS_CSV,
    OUTPUT_DIR,
    PLOTS_CANDIDATES,
    Thresholds,
)

KNOWN_MAX_AGE_DAYS = 7

LOG_COLUMNS = [
    "tic", "sector", "obs_id", "status", "tmag", "teff", "r_star", "m_star",
    "period", "t0", "duration_hr", "depth_ppm", "depth_err", "snr", "sde", "beta", "n_transits",
    "rp_rs", "rp_earth", "odd_even_sigma", "odd_even_frac", "depth_odd", "depth_even",
    "secondary_depth", "secondary_sigma", "duration_max_hr", "duration_ratio",
    "single_event_frac", "harmonic_dll", "bls_power", "is_candidate", "fail_reasons", "flags",
    "plot", "runtime_s",
]


# =================================================================== exclusion list
def load_known_tics(refresh: bool = False) -> set[int]:
    fresh = KNOWN_TICS_CSV.exists() and (time.time() - KNOWN_TICS_CSV.stat().st_mtime) < KNOWN_MAX_AGE_DAYS * 86400
    if fresh and not refresh:
        return set(pd.read_csv(KNOWN_TICS_CSV)["tic"].astype(int))

    from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

    print("Refreshing known-planet exclusion list from the NASA Exoplanet Archive ...")
    toi = NasaExoplanetArchive.query_criteria(table="toi", select="tid").to_pandas()
    ps = NasaExoplanetArchive.query_criteria(table="pscomppars", select="tic_id").to_pandas()
    toi_tics = pd.to_numeric(toi["tid"], errors="coerce").dropna().astype(int)
    ps_tics = (
        ps["tic_id"].dropna().astype(str).str.replace("TIC", "", regex=False).str.strip()
    )
    ps_tics = pd.to_numeric(ps_tics, errors="coerce").dropna().astype(int)
    df = pd.concat([
        pd.DataFrame({"tic": toi_tics, "source": "toi"}),
        pd.DataFrame({"tic": ps_tics, "source": "pscomppars"}),
    ])
    df = df.groupby("tic", as_index=False)["source"].agg(lambda s: "+".join(sorted(set(s))))
    KNOWN_TICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(KNOWN_TICS_CSV, index=False)
    print(f"  {len(toi_tics)} TOI rows + {len(ps_tics)} confirmed-planet rows -> {len(df)} unique TICs")
    return set(df["tic"])


# ===================================================================== target list
def sector_targets(sector: int, refresh: bool = False) -> pd.DataFrame:
    path = DATA_DIR / f"sector_targets_s{sector}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)

    from astroquery.mast import Observations

    print(f"Querying MAST for all 2-min SPOC light curves in sector {sector} ...")
    obs = Observations.query_criteria(
        obs_collection="TESS",
        dataproduct_type="timeseries",
        sequence_number=sector,
        provenance_name="SPOC",
        t_exptime=[119, 121],
    )
    df = obs[["target_name", "obs_id"]].to_pandas()
    df["tic"] = pd.to_numeric(df["target_name"], errors="coerce")
    df = df.dropna(subset=["tic"]).astype({"tic": int}).drop_duplicates("tic")[["tic", "obs_id"]]
    df.to_csv(path, index=False)
    print(f"  {len(df)} targets")
    return df


# =========================================================================== worker
def process_star(tic: int, obs_id: str, sector: int, th: Thresholds, keep_cache: bool) -> dict:
    """Runs in a worker process. Never raises: errors are recorded in the row."""
    import warnings as _w

    _w.filterwarnings("ignore")
    import lightkurve as lk

    from src import pipeline_core as pc

    t_start = time.time()
    row = {"tic": tic, "sector": sector, "obs_id": obs_id, "status": "ok", "is_candidate": False}
    lc_path = tp_path = None
    try:
        lc_path = pc.download_spoc_product(obs_id, "lc")
        lc = pc.read_pdcsap(lc_path)
        n_valid = len(lc.remove_nans())
        if n_valid < 1000:
            raise ValueError(f"only {n_valid} valid PDCSAP cadences")
        star = pc.get_stellar_params(lc.meta, tic)
        out = pc.search(lc, th)
        m = pc.vet(out.result, star, th)
        row.update(m)
        row.update(tmag=star["tmag"], teff=star["teff"])

        if m["is_candidate"]:
            tpf, aperture = None, None
            try:
                tp_path = pc.download_spoc_product(obs_id, "tp")
                tpf = lk.read(str(tp_path), quality_bitmask="default")
                aperture = tpf.pipeline_mask
            except Exception as exc:  # still plot in LC-only mode
                row["status"] = f"ok (TPF unavailable: {type(exc).__name__})"
            row["plot"] = str(pc.plot_diagnostic(
                out, m, f"TIC {tic} — Sector {sector} PDCSAP candidate",
                PLOTS_CANDIDATES / f"TIC_{tic}_candidate.png", tpf=tpf, aperture=aperture,
            ))
    except Exception as exc:
        row["status"] = f"error: {type(exc).__name__}: {exc}"[:300]
        row["is_candidate"] = False
    finally:
        if not keep_cache and not row.get("is_candidate"):
            for p in (lc_path, tp_path):
                if p is not None:
                    Path(p).unlink(missing_ok=True)
    row["runtime_s"] = round(time.time() - t_start, 1)
    return row


# ============================================================================= main
def append_rows(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows).reindex(columns=LOG_COLUMNS)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def update_candidates(new_rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(new_rows).reindex(columns=LOG_COLUMNS)
    if CANDIDATES_CSV.exists():
        new = pd.concat([pd.read_csv(CANDIDATES_CSV), new], ignore_index=True)
    new = new.drop_duplicates(["tic", "sector"], keep="last").sort_values("snr", ascending=False)
    new.to_csv(CANDIDATES_CSV, index=False)
    return new


def dir_size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--limit", type=int, default=25, help="max number of unflagged stars to process")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--snr-threshold", type=float, default=DEFAULT.snr_min)
    ap.add_argument("--sde-threshold", type=float, default=DEFAULT.sde_min)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refresh-known", action="store_true", help="re-download the TOI/confirmed exclusion list")
    ap.add_argument("--refresh-targets", action="store_true", help="re-query MAST for the sector target list")
    ap.add_argument("--keep-cache", action="store_true", help="do not delete non-candidate FITS files")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    th = dataclasses.replace(DEFAULT, snr_min=args.snr_threshold, sde_min=args.sde_threshold)
    for d in (CACHE_DIR, PLOTS_CANDIDATES, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    known = load_known_tics(args.refresh_known)
    targets = sector_targets(args.sector, args.refresh_targets)
    log_path = OUTPUT_DIR / f"scan_log_s{args.sector}.csv"
    done = set(pd.read_csv(log_path)["tic"].astype(int)) if log_path.exists() else set()

    is_known = targets["tic"].isin(known)
    is_done = targets["tic"].isin(done)
    pool = targets[~is_known & ~is_done]
    sample = pool.sample(n=min(args.limit, len(pool)), random_state=args.seed)
    print(f"Sector {args.sector}: {len(targets)} targets | {int(is_known.sum())} known TOI/planet hosts excluded | "
          f"{int((is_done & ~is_known).sum())} already scanned | {len(pool)} unflagged remaining | processing {len(sample)}")
    print(f"Gates: SNR ≥ {th.snr_min}, SDE ≥ {th.sde_min}, odd/even ≤ {th.odd_even_sigma_max}σ, "
          f"Rp ≤ {th.rp_max_earth} R⊕, N_transits ≥ {th.min_n_transits}")

    t0 = time.time()
    rows, cands = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_star, int(r.tic), r.obs_id, args.sector, th, args.keep_cache)
                for r in sample.itertuples()]
        for fut in tqdm(as_completed(futs), total=len(futs), unit="star"):
            row = fut.result()
            rows.append(row)
            append_rows(log_path, [row])          # incremental: survives an interrupted run
            if row.get("is_candidate"):
                cands.append(row)
                tqdm.write(f"  ★ TIC {row['tic']}: P={row['period']:.4f} d, depth={row['depth_ppm']:.0f} ppm, "
                           f"SNR={row['snr']:.1f}, SDE={row['sde']:.1f}, Rp={row['rp_earth']:.1f} R⊕")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows).reindex(columns=LOG_COLUMNS)
    if cands:
        update_candidates(cands)

    n_err = int(df["status"].astype(str).str.startswith("error").sum())
    print(f"\nProcessed {len(df)} stars in {elapsed:.0f} s ({elapsed / max(len(df), 1):.1f} s/star wall, "
          f"{args.workers} workers). Errors: {n_err}. Candidates: {len(cands)}.")
    reasons = df.loc[df["status"] == "ok", "fail_reasons"].dropna().astype(str)
    counts = pd.Series([r for s in reasons for r in s.split(";") if r]).value_counts()
    if len(counts):
        print("Rejection reasons (a star can have several):")
        print(counts.to_string())
    if n_err:
        print("Errors:")
        print(df.loc[df["status"].astype(str).str.startswith("error"), ["tic", "status"]].to_string(index=False))
    show = ["tic", "tmag", "period", "depth_ppm", "snr", "sde", "n_transits", "rp_earth",
            "odd_even_sigma", "secondary_sigma", "fail_reasons", "flags"]
    print("\nTop 10 by SNR this run:")
    print(df.sort_values("snr", ascending=False)[show].head(10).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nScan log: {log_path}\nCandidates: {CANDIDATES_CSV if cands else '(none this run)'}")
    print(f"data/cache size now: {dir_size_mb(CACHE_DIR):.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
