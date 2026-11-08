#!/usr/bin/env python
"""Phase 2b — pixel-level false-positive vetting of a transit candidate.

A TESS pixel is 21″ across, so a faint eclipsing binary inside the aperture can
be diluted into a planet-sized dip. Given an ephemeris, this script checks
where on the sky the dip actually comes from:

1. In- vs out-of-transit flux-weighted centroid shift. Each transit is
   differenced against its own local baseline, so slow pointing drifts cancel.
   Flagged if > 3σ.
2. Difference image (mean local OOT − mean in-transit). Its centroid should
   sit on the target star. Flagged if it is > 1 px from the target.
3. A per-pixel phase-folded light-curve grid across the whole stamp.
4. Gaia DR3 sources within 60″, propagated to the TESS epoch and overlaid on
   the stamp. Neighbours bright enough to produce the observed depth are
   highlighted.

Usage:
  .venv/bin/python src/vet_candidate_tpf.py --tic 25155310 --period 3.28842 --t0 1327.52063 --sector 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import astropy.units as u  # noqa: E402
import lightkurve as lk  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.time import Time  # noqa: E402
from astropy.timeseries import BoxLeastSquares  # noqa: E402
from astropy.visualization import simple_norm  # noqa: E402

from src import pipeline_core as pc  # noqa: E402
from src.config import BTJD_OFFSET, CACHE_DIR, DEFAULT, PLOTS_CANDIDATES, TESS_PIXEL_ARCSEC  # noqa: E402

SIGMA_FLAG = 3.0          # centroid-shift significance threshold
DIFF_OFFSET_FLAG_PX = 1.0  # difference-image centroid vs target position


# ============================================================================ data
def load_tpf(tic: int, sector: int | None):
    pattern = f"*-s{sector:04d}-{tic:016d}-*_tp.fits" if sector else f"*-{tic:016d}-*_tp.fits"
    local = sorted(CACHE_DIR.rglob(pattern))
    if local:
        return lk.read(str(local[0]), quality_bitmask="default")
    sr = lk.search_targetpixelfile(f"TIC {tic}", author="SPOC", exptime=120, sector=sector)
    if len(sr) == 0:
        sys.exit(f"No SPOC 120-s TPF for TIC {tic} (sector {sector})")
    return sr[0].download(download_dir=str(CACHE_DIR), quality_bitmask="default")


def estimate_duration(t, f, period, t0) -> float:
    """Best box duration at a fixed ephemeris (maximum depth SNR)."""
    lc = lk.LightCurve(time=t, flux=f)
    flat, _, _ = pc.clean_and_flatten(lc, DEFAULT)
    bls = BoxLeastSquares(flat.time.value, flat.flux.value, np.full(len(flat), pc._robust_std(flat.flux.value)))
    best, best_snr = DEFAULT.durations[0], -np.inf
    for d in DEFAULT.durations:
        if d >= 0.5 * period:
            continue
        dep, err = bls.compute_stats(period, d, t0)["depth"]
        snr = float(dep) / float(err)
        if snr > best_snr:
            best, best_snr = d, snr
    return float(best)


# ========================================================================= analysis
def local_windows(t, period, t0, dur):
    ph = pc.phase_days(t, period, t0)
    epoch = np.round((t - t0) / period).astype(int)
    in_tr = np.abs(ph) < 0.4 * dur                               # central 80 % of transit
    oot_hi = min(2.5 * dur, 0.5 * period)
    oot = (np.abs(ph) >= 0.75 * dur) & (np.abs(ph) <= oot_hi)    # local baseline on both sides
    return ph, epoch, in_tr, oot


def centroid_test(col, row, epoch, in_tr, oot):
    """Per-transit local differencing of centroids. Returns a dict of results."""
    dxs, dys, resid_x, resid_y, n_in_tot = [], [], [], [], 0
    for e in np.unique(epoch[in_tr]):
        si, so = in_tr & (epoch == e), oot & (epoch == e)
        if si.sum() < 5 or so.sum() < 10:
            continue
        bx, by = np.nanmedian(col[so]), np.nanmedian(row[so])
        dxs.append(np.nanmean(col[si]) - bx)
        dys.append(np.nanmean(row[si]) - by)
        resid_x.append(col[so] - bx)
        resid_y.append(row[so] - by)
        n_in_tot += si.sum()
    n = len(dxs)
    if n == 0:
        return {"n_transits_used": 0, "flag": False, "note": "no usable transits"}
    dxs, dys = np.array(dxs), np.array(dys)
    rx, ry = np.concatenate(resid_x), np.concatenate(resid_y)
    # pointwise error from OOT scatter; empirical error from transit-to-transit scatter
    sx_pt = pc._robust_std(rx) * math.sqrt(1 / n_in_tot + 1 / rx.size)
    sy_pt = pc._robust_std(ry) * math.sqrt(1 / n_in_tot + 1 / ry.size)
    sx_emp = np.std(dxs, ddof=1) / math.sqrt(n) if n >= 3 else 0.0
    sy_emp = np.std(dys, ddof=1) / math.sqrt(n) if n >= 3 else 0.0
    sx, sy = max(sx_pt, sx_emp), max(sy_pt, sy_emp)
    dx, dy = float(np.mean(dxs)), float(np.mean(dys))
    sig = math.hypot(dx / sx, dy / sy)
    return {
        "n_transits_used": n,
        "dx_px": dx, "dy_px": dy, "sigma_x_px": sx, "sigma_y_px": sy,
        "shift_arcsec": math.hypot(dx, dy) * TESS_PIXEL_ARCSEC,
        "significance_sigma": sig,
        "flag": bool(sig > SIGMA_FLAG),
    }


def weighted_centroid(img, around, half=1):
    """Flux-weighted centroid of positive pixels in a (2*half+1)^2 box around (row, col)."""
    r0, c0 = around
    ny, nx = img.shape
    rs = slice(max(r0 - half, 0), min(r0 + half + 1, ny))
    cs = slice(max(c0 - half, 0), min(c0 + half + 1, nx))
    sub = np.clip(np.nan_to_num(img[rs, cs]), 0, None)
    if sub.sum() <= 0:
        return float(c0), float(r0)
    yy, xx = np.mgrid[rs, cs]
    return float((sub * xx).sum() / sub.sum()), float((sub * yy).sum() / sub.sum())


def difference_image(flux, in_tr, oot):
    oot_img = np.nanmean(flux[oot], axis=0)
    in_img = np.nanmean(flux[in_tr], axis=0)
    diff = oot_img - in_img
    sd = np.nanstd(flux[oot], axis=0)
    snr = diff / (sd * math.sqrt(1 / in_tr.sum() + 1 / oot.sum()))
    return oot_img, diff, snr


def per_pixel_folds(flux, ph, epoch, in_tr, oot, half, nbins=40):
    """Local-baseline normalised, phase-binned light curve for every pixel."""
    ny, nx = flux.shape[1:]
    win = np.abs(ph) <= half
    rel = np.full(flux.shape, np.nan)
    for e in np.unique(epoch[win]):
        sw, so = win & (epoch == e), oot & (epoch == e)
        if so.sum() < 5:
            continue
        base = np.nanmedian(flux[so], axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel[sw] = flux[sw] / base
    edges = np.linspace(-half, half, nbins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(ph, edges) - 1
    binned = np.full((nbins, ny, nx), np.nan)
    for b in range(nbins):
        sel = win & (idx == b)
        if sel.sum() >= 3:
            binned[b] = np.nanmean(rel[sel], axis=0)
    # per-pixel depth significance from in-transit vs OOT relative flux
    d_in = np.nanmean(rel[in_tr & win], axis=0)
    d_out = np.nanmean(rel[oot & win], axis=0)
    s_out = np.nanstd(rel[oot & win], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        depth_snr = (d_out - d_in) / (s_out * math.sqrt(1 / max((in_tr & win).sum(), 1) + 1 / max((oot & win).sum(), 1)))
    return centres, binned, depth_snr


def query_gaia(ra, dec, radius_arcsec, epoch_jyear):
    from astroquery.gaia import Gaia

    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    Gaia.ROW_LIMIT = -1
    job = Gaia.cone_search_async(SkyCoord(ra, dec, unit="deg"), radius=radius_arcsec * u.arcsec)
    tab = job.get_results()
    cols = {c.lower(): c for c in tab.colnames}
    g = lambda name: np.asarray(tab[cols[name]].filled(np.nan) if hasattr(tab[cols[name]], "filled") else tab[cols[name]], dtype=float)  # noqa: E731
    ra_g, dec_g, gmag = g("ra"), g("dec"), g("phot_g_mean_mag")
    pmra, pmdec = np.nan_to_num(g("pmra")), np.nan_to_num(g("pmdec"))
    dt = epoch_jyear - 2016.0
    dec_t = dec_g + pmdec * dt / 3.6e6
    ra_t = ra_g + pmra * dt / 3.6e6 / np.cos(np.radians(dec_g))
    sid = np.asarray(tab[cols["source_id"]])
    return ra_t, dec_t, gmag, sid


# ============================================================================= main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tic", type=int, required=True)
    ap.add_argument("--period", type=float, required=True, help="days")
    ap.add_argument("--t0", type=float, required=True, help="mid-transit time, BTJD")
    ap.add_argument("--duration", type=float, default=None, help="days (estimated if omitted)")
    ap.add_argument("--sector", type=int, default=None)
    ap.add_argument("--gaia-radius", type=float, default=60.0, help="arcsec")
    args = ap.parse_args()

    tpf = load_tpf(args.tic, args.sector)
    sector = int(tpf.meta.get("SECTOR", args.sector or 0))
    aperture = tpf.pipeline_mask
    if aperture.sum() == 0:
        aperture = tpf.create_threshold_mask(threshold=3)

    t_all = tpf.time.value
    flux_all = np.asarray(tpf.flux.value, dtype=float)
    ap_flux_all = np.nansum(flux_all[:, aperture], axis=1)
    good = np.isfinite(t_all) & np.isfinite(ap_flux_all) & (ap_flux_all > 0)
    t, flux, ap_flux = t_all[good], flux_all[good], ap_flux_all[good]
    col_c, row_c = (np.asarray(x.value, dtype=float)[good] for x in tpf.estimate_centroids(aperture_mask=aperture))
    col_c -= tpf.column      # absolute CCD -> stamp pixel coordinates
    row_c -= tpf.row

    P, t0 = args.period, args.t0
    dur = args.duration or estimate_duration(t, ap_flux / np.nanmedian(ap_flux), P, t0)
    ph, epoch, in_tr, oot = local_windows(t, P, t0, dur)
    if in_tr.sum() < 10 or oot.sum() < 20:
        sys.exit(f"Not enough in-transit ({in_tr.sum()}) or out-of-transit ({oot.sum()}) cadences in sector {sector}")

    # aperture depth from the same local baselines
    rel_ap = np.full_like(ap_flux, np.nan)
    for e in np.unique(epoch):
        so = oot & (epoch == e)
        if so.sum() >= 5:
            rel_ap[epoch == e] = ap_flux[epoch == e] / np.nanmedian(ap_flux[so])
    depth = float(np.nanmean(rel_ap[oot]) - np.nanmean(rel_ap[in_tr]))

    cen = centroid_test(col_c, row_c, epoch, in_tr, oot)
    if cen.get("n_transits_used") and depth > 0:
        cen["implied_source_offset_arcsec"] = cen["shift_arcsec"] / depth

    # target position in stamp pixels from the TPF WCS
    tx, ty = (float(v) for v in tpf.wcs.world_to_pixel(SkyCoord(tpf.ra, tpf.dec, unit="deg")))
    oot_img, diff, diff_snr = difference_image(flux, in_tr, oot)
    # Locate the dip source in FLUX units: per-pixel SNR is biased toward PSF wings
    # because pointing jitter inflates the noise of the bright core pixels.
    pk_r, pk_c = np.unravel_index(np.nanargmax(np.nan_to_num(diff, nan=-np.inf)), diff.shape)
    dcx, dcy = weighted_centroid(diff, (pk_r, pk_c))
    yy, xx = np.mgrid[: diff.shape[0], : diff.shape[1]]
    near = np.hypot(xx - tx, yy - ty) <= 3
    o_r, o_c = np.unravel_index(np.nanargmax(np.where(near, np.nan_to_num(oot_img, nan=-np.inf), -np.inf)), diff.shape)
    ocx, ocy = weighted_centroid(oot_img, (o_r, o_c))
    diff_off = math.hypot(dcx - ocx, dcy - ocy)          # vs OOT centroid: immune to WCS offsets
    diff_off_wcs = math.hypot(dcx - tx, dcy - ty)
    diffres = {
        "peak_pixel": [int(pk_c), int(pk_r)],
        "peak_snr": float(diff_snr[pk_r, pk_c]),
        "centroid_px": [dcx, dcy],
        "oot_centroid_px": [ocx, ocy],
        "target_wcs_px": [tx, ty],
        "offset_px": diff_off,
        "offset_arcsec": diff_off * TESS_PIXEL_ARCSEC,
        "offset_from_wcs_px": diff_off_wcs,
        "flag": bool(diff_off > DIFF_OFFSET_FLAG_PX),
    }

    half = min(3 * dur, 0.5 * P)
    centres, binned, pix_snr = per_pixel_folds(flux, ph, epoch, in_tr, oot, half)

    # Gaia DR3 neighbours
    gaia = {"queried": False}
    mid_jyear = Time(np.nanmedian(t) + BTJD_OFFSET, format="jd").jyear
    dg_max = -2.5 * math.log10(depth / (1 - depth)) if 0 < depth < 1 else float("nan")
    try:
        ra_g, dec_g, gmag, sid = query_gaia(tpf.ra, tpf.dec, args.gaia_radius, mid_jyear)
        gx, gy = tpf.wcs.world_to_pixel(SkyCoord(ra_g, dec_g, unit="deg"))
        sep = SkyCoord(ra_g, dec_g, unit="deg").separation(SkyCoord(tpf.ra, tpf.dec, unit="deg")).arcsec
        tmag = tpf.meta.get("TESSMAG")
        cand = np.where((sep < 10) & (gmag < (tmag + 2 if tmag else 99)))[0]
        it = int(cand[np.argmin(sep[cand])]) if cand.size else int(np.argmin(sep))
        dG = gmag - gmag[it]
        ap_rows, ap_cols = np.where(aperture)
        d_ap = np.min(np.hypot(gx[:, None] - ap_cols[None, :], gy[:, None] - ap_rows[None, :]), axis=1)
        risky = (np.arange(len(gmag)) != it) & (dG <= dg_max) & (d_ap <= 2.0)
        gaia = {
            "queried": True, "n_sources": int(len(gmag)), "target_source_id": int(sid[it]),
            "target_G": float(gmag[it]), "delta_G_max_for_depth": dg_max,
            "possible_contaminants": [
                {"source_id": int(sid[i]), "G": float(gmag[i]), "delta_G": float(dG[i]),
                 "sep_arcsec": float(sep[i]), "px_from_aperture": float(d_ap[i])}
                for i in np.where(risky)[0]
            ],
        }
    except Exception as exc:
        gaia["error"] = f"{type(exc).__name__}: {exc}"[:200]
        gx = gy = gmag = dG = None

    off_target = cen.get("flag", False) or diffres["flag"]
    verdict = "OFF-TARGET SUSPECT" if off_target else "ON-TARGET CONSISTENT"

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.35], hspace=0.25, wspace=0.28)
    axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ny, nx = oot_img.shape

    def overlay(ax):
        pc.outline_mask(ax, aperture, color="red", lw=1.3)
        ax.plot(tx, ty, "+", color="white", ms=14, mew=2, label="target (WCS)")
        if gaia.get("queried"):
            inside = (gx > -1) & (gx < nx) & (gy > -1) & (gy < ny)
            sizes = np.clip(80 * 10 ** (-0.25 * (gmag - gmag.min())), 6, 200)
            rk = np.zeros(len(gmag), dtype=bool)
            ids = {c["source_id"] for c in gaia["possible_contaminants"]}
            rk[[i for i, s in enumerate(sid) if int(s) in ids]] = True
            ax.scatter(gx[inside & ~rk], gy[inside & ~rk], s=sizes[inside & ~rk], facecolors="none",
                       edgecolors="cyan", lw=1.2, label="Gaia DR3")
            if rk.any():
                ax.scatter(gx[inside & rk], gy[inside & rk], s=sizes[inside & rk], facecolors="none",
                           edgecolors="orange", lw=2, label=f"Gaia ΔG ≤ {dg_max:.1f}")
        ax.set_xlim(-0.5, nx - 0.5)
        ax.set_ylim(-0.5, ny - 0.5)
        ax.set_xlabel("column [px]")
        ax.set_ylabel("row [px]")

    im = axA.imshow(oot_img, origin="lower", cmap="viridis",
                    norm=simple_norm(oot_img[np.isfinite(oot_img)], "sqrt", percent=99.5))
    fig.colorbar(im, ax=axA, fraction=0.046, label="e$^-$/s")
    overlay(axA)
    axA.set_title(f"Out-of-transit image + Gaia DR3 ({args.gaia_radius:.0f}″)")
    axA.legend(loc="upper right", fontsize=7, framealpha=0.6)

    lim = np.nanmax(np.abs(diff))
    im = axB.imshow(diff, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    fig.colorbar(im, ax=axB, fraction=0.046, label="OOT − in-transit [e$^-$/s]")
    overlay(axB)
    axB.plot(ocx, ocy, "o", mfc="none", mec="k", ms=12, mew=1.5, label="OOT centroid")
    axB.plot(dcx, dcy, "x", color="k", ms=12, mew=2.5, label="diff-image centroid")
    axB.legend(loc="upper right", fontsize=7, framealpha=0.6)
    axB.set_title(f"Difference image — source {diff_off:.2f} px ({diff_off*TESS_PIXEL_ARCSEC:.0f}″) from OOT centroid")

    win = np.abs(ph) <= half
    hrs = ph * 24
    for arr, lab, colr in ((col_c, "column", "C0"), (row_c, "row", "C3")):
        resid = np.full_like(arr, np.nan)
        for e in np.unique(epoch[win]):
            so = oot & (epoch == e)
            if so.sum() >= 5:
                resid[(epoch == e) & win] = arr[(epoch == e) & win] - np.nanmedian(arr[so])
        ok = win & np.isfinite(resid)
        bc, bm, be = pc.bin_series(hrs[ok], resid[ok] * TESS_PIXEL_ARCSEC * 1000, max(dur * 24 / 8, 10 / 60))
        axC.errorbar(bc, bm, be, fmt="o-", ms=3, color=colr, label=f"{lab} centroid")
    axC.axvspan(-0.4 * dur * 24, 0.4 * dur * 24, color="0.85", zorder=0, label="in-transit window")
    axC.axhline(0, color="k", lw=0.8)
    axC.set_xlabel("Hours from mid-transit")
    axC.set_ylabel("centroid offset [milli-arcsec]")
    if cen.get("n_transits_used"):
        axC.set_title(f"Centroid shift {cen['shift_arcsec']*1000:.1f} mas = {cen['significance_sigma']:.1f}σ "
                      f"({cen['n_transits_used']} transits)")
    axC.legend(fontsize=8)

    # per-pixel grid (rows drawn top-down so the grid matches origin='lower' images)
    sub = gs[1, 0:2].subgridspec(ny, nx, wspace=0.04, hspace=0.04)
    ylo, yhi = 1 - 2.2 * max(depth, 1e-4), 1 + 1.2 * max(depth, 1e-4)
    vmax = max(np.nanpercentile(np.abs(pix_snr), 99), 3)
    cmap = plt.get_cmap("RdBu_r")
    grid_top = 0.0
    for r in range(ny):
        for c in range(nx):
            ax = fig.add_subplot(sub[ny - 1 - r, c])
            grid_top = max(grid_top, ax.get_position().y1)
            s = pix_snr[r, c]
            if np.isfinite(s):
                ax.set_facecolor(cmap(0.5 + 0.5 * np.clip(s / vmax, -1, 1) * 0.6))
            ax.plot(centres * 24, binned[:, r, c], "k-", lw=0.8)
            ax.set_ylim(ylo, yhi)
            ax.set_xticks([])
            ax.set_yticks([])
            if aperture[r, c]:
                for sp in ax.spines.values():
                    sp.set_color("red")
                    sp.set_linewidth(1.6)
    fig.text(0.37, grid_top + 0.008, "Per-pixel phase-folded light curves (background colour = in-transit dip "
             "significance; red frame = aperture)", ha="center", va="bottom", fontsize=10)

    axE = fig.add_subplot(gs[1, 2])
    ok = win & np.isfinite(rel_ap)
    axE.plot(hrs[ok], rel_ap[ok], ".", ms=1.5, color="0.6", alpha=0.5, rasterized=True)
    bc, bm, be = pc.bin_series(hrs[ok], rel_ap[ok], max(dur * 24 / 10, 10 / 60))
    axE.errorbar(bc, bm, be, fmt="o", ms=3.5, color="k")
    axE.axvspan(-0.4 * dur * 24, 0.4 * dur * 24, color="0.85", zorder=0)
    axE.set_ylim(ylo, yhi)
    axE.set_xlabel("Hours from mid-transit")
    axE.set_ylabel("Aperture flux (local baseline)")
    axE.set_title(f"Aperture light curve — depth {depth*1e6:.0f} ppm")
    lines = [
        f"Centroid: {'FLAG' if cen.get('flag') else 'ok'}  ({cen.get('significance_sigma', float('nan')):.1f}σ)",
        f"Implied source offset: {cen.get('implied_source_offset_arcsec', float('nan')):.1f}″",
        f"Difference image: {'FLAG' if diffres['flag'] else 'ok'}  ({diff_off:.2f} px)",
    ]
    if gaia.get("queried"):
        lines.append(f"Gaia: {gaia['n_sources']} sources; {len(gaia['possible_contaminants'])} could mimic depth")
    else:
        lines.append("Gaia: query failed")
    axE.text(0.03, 0.04, "\n".join(lines), transform=axE.transAxes, fontsize=9, family="monospace",
             bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.6"))

    fig.suptitle(f"TIC {args.tic} — Sector {sector} deep pixel vetting — {verdict}\n"
                 f"P = {P:.5f} d, t0 = {t0:.5f} BTJD, duration = {dur*24:.2f} h", fontsize=13)
    out_png = PLOTS_CANDIDATES / f"TIC_{args.tic}_deep_vetting.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)

    report = {
        "tic": args.tic, "sector": sector, "period": P, "t0": t0, "duration_days": dur,
        "aperture_depth": depth, "n_in_transit": int(in_tr.sum()), "n_out_of_transit": int(oot.sum()),
        "centroid": cen, "difference_image": diffres, "gaia": gaia, "verdict": verdict,
        "figure": str(out_png),
    }
    out_json = out_png.with_suffix(".json")
    out_json.write_text(json.dumps(report, indent=2, default=float))

    print(f"TIC {args.tic} sector {sector}: depth {depth*1e6:.0f} ppm, duration {dur*24:.2f} h")
    if cen.get("n_transits_used"):
        print(f"  centroid shift   : {cen['shift_arcsec']*1000:.1f} mas, {cen['significance_sigma']:.2f}σ "
              f"(implied source offset {cen.get('implied_source_offset_arcsec', float('nan')):.1f}″) "
              f"-> {'FLAG' if cen['flag'] else 'ok'}")
    print(f"  difference image : source {diff_off:.2f} px ({diff_off*TESS_PIXEL_ARCSEC:.1f}″) from OOT centroid, "
          f"{diff_off_wcs:.2f} px from WCS target, peak px {diffres['peak_pixel']} -> "
          f"{'FLAG' if diffres['flag'] else 'ok'}")
    if gaia.get("queried"):
        print(f"  Gaia DR3         : {gaia['n_sources']} sources within {args.gaia_radius:.0f}″, target G = "
              f"{gaia['target_G']:.2f}, ΔG_max = {dg_max:.2f}, possible contaminants = {len(gaia['possible_contaminants'])}")
        for cnt in gaia["possible_contaminants"]:
            print(f"      Gaia {cnt['source_id']}: G={cnt['G']:.2f} (ΔG {cnt['delta_G']:+.2f}), "
                  f"{cnt['sep_arcsec']:.1f}″ away, {cnt['px_from_aperture']:.1f} px from aperture")
    else:
        print(f"  Gaia DR3         : {gaia.get('error')}")
    print(f"  VERDICT          : {verdict}")
    print(f"  figure: {out_png}\n  report: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
