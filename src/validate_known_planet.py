#!/usr/bin/env python
"""Phase 1 — rediscover WASP-126 b (TIC 25155310) from Sector 1 SPOC pixels.

1. Download the 2-min SPOC Target Pixel File and do aperture photometry with the
   SPOC pipeline aperture.
2. Run the shared search + vetting pipeline and render the 4-panel figure.
3. Repeat on the same sector's PDCSAP light curve, so the Phase 2 code path
   is also validated on a known planet.
4. Compare both against the NASA Exoplanet Archive. Exit code 1 if any
   acceptance criterion fails.

Usage:  .venv/bin/python src/validate_known_planet.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore", message=".*oktopus.*")

import lightkurve as lk  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import pipeline_core as pc  # noqa: E402
from src.config import CACHE_DIR, DEFAULT, PLOTS_VALIDATION  # noqa: E402

TIC = 25155310
SECTOR = 1
NAME = "WASP-126 b"

# Fallback if the archive is unreachable (pscomppars, checked 2026-11-07)
FALLBACK = {"period": 3.28883, "depth": 0.0070057, "rp_rs": 0.078, "duration_hr": 3.437, "rp_earth": 10.51}
# Values quoted in the original setup plan, shown for reference only
SPEC = {"period": 3.2888, "depth": 0.0078, "rp_rs": 0.088}


def fetch_catalog() -> tuple[dict, str]:
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

        tab = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select="pl_name,pl_orbper,pl_trandep,pl_ratror,pl_trandur,pl_rade",
            where=f"tic_id='TIC {TIC}'",
        )
        r = tab[0]
        v = lambda c: float(getattr(r[c], "value", r[c]))  # noqa: E731
        return {
            "period": v("pl_orbper"),
            "depth": v("pl_trandep") / 100.0,          # archive stores percent
            "rp_rs": v("pl_ratror"),
            "duration_hr": v("pl_trandur"),
            "rp_earth": v("pl_rade"),
        }, "NASA Exoplanet Archive (live)"
    except Exception as exc:
        return FALLBACK, f"fallback constants ({type(exc).__name__})"


def run_tpf(th):
    sr = lk.search_targetpixelfile(f"TIC {TIC}", sector=SECTOR, author="SPOC", exptime=120)
    if len(sr) == 0:
        sys.exit(f"No SPOC 120-s TPF found for TIC {TIC} sector {SECTOR}")
    tpf = sr[0].download(download_dir=str(CACHE_DIR), quality_bitmask="default")
    aperture = tpf.pipeline_mask
    lc = tpf.to_lightcurve(aperture_mask=aperture)
    star = pc.get_stellar_params(tpf.meta, TIC)
    out = pc.search(lc, th)
    m = pc.vet(out.result, star, th)
    path = pc.plot_diagnostic(
        out, m, f"TIC {TIC} ({NAME}) — Sector {SECTOR} TPF aperture photometry",
        PLOTS_VALIDATION / f"TIC_{TIC}_validation.png", tpf=tpf, aperture=aperture,
    )
    print(f"TPF: {tpf.shape[1]}x{tpf.shape[2]} stamp, {int(aperture.sum())}-px aperture, "
          f"{len(lc)} cadences, star = {star}")
    return out, m, path


def run_pdcsap(th):
    sr = lk.search_lightcurve(f"TIC {TIC}", sector=SECTOR, author="SPOC", exptime=120)
    lc = sr[0].download(download_dir=str(CACHE_DIR), flux_column="pdcsap_flux", quality_bitmask="default")
    star = pc.get_stellar_params(lc.meta, TIC)
    out = pc.search(lc, th)
    m = pc.vet(out.result, star, th)
    path = pc.plot_diagnostic(
        out, m, f"TIC {TIC} ({NAME}) — Sector {SECTOR} PDCSAP",
        PLOTS_VALIDATION / f"TIC_{TIC}_validation_pdcsap.png",
    )
    return out, m, path


def main() -> int:
    th = DEFAULT
    cat, cat_src = fetch_catalog()
    print(f"Catalog source: {cat_src}")

    _, m_tpf, p_tpf = run_tpf(th)
    _, m_pdc, p_pdc = run_pdcsap(th)

    def rel(x, ref):
        return abs(x - ref) / ref

    checks = [
        # label, key, catalog value, spec value, criterion text, test(metrics)
        ("Period [d]", "period", cat["period"], SPEC["period"], "|Δ| < 0.005 d",
         lambda m: abs(m["period"] - cat["period"]) < 0.005),
        ("Depth [%]", "depth", cat["depth"], SPEC["depth"], "±20 % of catalog",
         lambda m: rel(m["depth"], cat["depth"]) < 0.20),
        ("Rp/R*", "rp_rs", cat["rp_rs"], SPEC["rp_rs"], "±15 % of catalog",
         lambda m: rel(m["rp_rs"], cat["rp_rs"]) < 0.15),
        ("Odd-even frac", "odd_even_frac", None, None, f"< {th.odd_even_frac_max:.2f}",
         lambda m: m["odd_even_frac"] < th.odd_even_frac_max),
        ("Odd-even σ", "odd_even_sigma", None, None, f"< {th.odd_even_sigma_max:g}σ",
         lambda m: m["odd_even_sigma"] < th.odd_even_sigma_max),
        ("N transits", "n_transits", None, None, f"≥ {th.min_n_transits}",
         lambda m: m["n_transits"] >= th.min_n_transits),
        ("SDE", "sde", None, None, "≥ 8", lambda m: m["sde"] >= 8),
        ("All vetting gates", "is_candidate", None, None, "candidate", lambda m: bool(m["is_candidate"])),
        ("Duration [h]", "duration_hr", cat["duration_hr"], None, "info", None),
        ("Rp [R⊕]", "rp_earth", cat["rp_earth"], None, "info", None),
        ("SNR", "snr", None, None, "info", None),
        ("Secondary σ", "secondary_sigma", None, None, "info", None),
        ("t0 [BTJD]", "t0", None, None, "info", None),
    ]

    def fmt(key, v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "—"
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if key == "depth":
            return f"{100 * v:.3f}"
        if key == "n_transits":
            return str(int(v))
        if key in ("period", "t0"):
            return f"{v:.5f}"
        return f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"

    rows, all_pass = [], True
    for label, key, catv, specv, crit, test in checks:
        ok_t = ok_p = None
        if test is not None:
            ok_t, ok_p = bool(test(m_tpf)), bool(test(m_pdc))
            all_pass &= ok_t and ok_p
        rows.append({
            "quantity": label,
            "TPF (ours)": fmt(key, m_tpf[key]),
            "PDCSAP (ours)": fmt(key, m_pdc[key]),
            "catalog": fmt(key, catv),
            "spec": fmt(key, specv),
            "criterion": crit,
            "TPF": "—" if ok_t is None else ("PASS" if ok_t else "FAIL"),
            "PDCSAP": "—" if ok_p is None else ("PASS" if ok_p else "FAIL"),
        })
    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))
    table.to_csv(PLOTS_VALIDATION / f"TIC_{TIC}_validation.csv", index=False)
    print(f"\nFigures: {p_tpf}\n         {p_pdc}")
    print(f"Ephemeris for Phase 2b: --tic {TIC} --period {m_tpf['period']:.5f} "
          f"--t0 {m_tpf['t0']:.5f} --duration {m_tpf['duration_hr']/24:.4f} --sector {SECTOR}")
    print("\nPHASE 1 RESULT:", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
