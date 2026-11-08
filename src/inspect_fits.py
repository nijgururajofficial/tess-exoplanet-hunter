#!/usr/bin/env python
"""Read a TESS SPOC FITS file and explain what is inside it.

A FITS file is a stack of HDUs ("header/data units"). Each HDU has a header of
labelled values and, optionally, data. A SPOC 2-minute file contains:

  [0] PRIMARY     header only: the star (TIC ID, position, temperature, radius ...)
  [1] LIGHTCURVE  table, one row per 2-min cadence: TIME, SAP_FLUX, PDCSAP_FLUX, QUALITY ...
      or PIXELS   table, one row per cadence, each row holding a whole FLUX image (e.g. 11 x 11)
  [2] APERTURE    small integer image marking which pixels were summed into SAP_FLUX

This script opens files with astropy.io.fits and prints a plain-language
summary. It can also save a quick-look plot and export the per-cadence table to CSV.

Usage:
  .venv/bin/python src/inspect_fits.py --list                       # FITS files in data/cache/
  .venv/bin/python src/inspect_fits.py --tic 25155310               # cached files for one star
  .venv/bin/python src/inspect_fits.py path/to/file.fits --plot --csv
  .venv/bin/python src/inspect_fits.py --tic 25155310 --sector 1 --kind lc --header
  .venv/bin/python src/inspect_fits.py --tic 219900423 --sector 14 --download   # fetch from MAST first

In Python:
  from src.inspect_fits import table_to_dataframe
  df = table_to_dataframe("data/cache/.../..._lc.fits")
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.time import Time  # noqa: E402

from src.config import BTJD_OFFSET, CACHE_DIR, INSPECT_DIR, ROOT, TESS_PIXEL_ARCSEC  # noqa: E402

# Bits of the APERTURE image (TESS Science Data Products Description Document)
AP_OPTIMAL = 2      # pixel is in the photometric aperture summed into SAP_FLUX
AP_BACKGROUND = 4   # pixel was used to estimate the sky background

NAME_RE = re.compile(r"-s(\d{4})-(\d{16})-.*_(lc|tp)\.fits$")

KIND_TEXT = {
    "lc": "Light curve file: one brightness value per 2-minute cadence.",
    "tpf": "Target pixel file: a small image of the star for every 2-minute cadence.",
    "other": "Not a SPOC light curve or pixel file; showing its structure only.",
}

# (header keyword, unit, plain-language meaning)
STAR_KEYS = [
    ("OBJECT", "", "target name"),
    ("TICID", "", "TESS Input Catalog ID"),
    ("SECTOR", "", "observing sector (~27 days of one sky patch)"),
    ("CAMERA", "", "which of TESS's 4 cameras"),
    ("CCD", "", "which of that camera's 4 detectors"),
    ("RA_OBJ", "deg", "right ascension (sky longitude)"),
    ("DEC_OBJ", "deg", "declination (sky latitude)"),
    ("PMRA", "mas/yr", "proper motion in RA (how fast it drifts across the sky)"),
    ("PMDEC", "mas/yr", "proper motion in Dec"),
    ("TESSMAG", "mag", "brightness in the TESS band (lower = brighter)"),
    ("TEFF", "K", "surface temperature (Sun: 5772 K)"),
    ("LOGG", "cgs", "log surface gravity (Sun: 4.44; giant stars < 3.5)"),
    ("MH", "dex", "metallicity [M/H] (Sun: 0)"),
    ("RADIUS", "R_sun", "star radius in Suns"),
]


# ============================================================================ helpers
def _value(header, key):
    """Header value, or None when the keyword is missing, blank or NaN."""
    val = header.get(key)
    if val is None or isinstance(val, fits.card.Undefined):
        return None
    if isinstance(val, float) and not np.isfinite(val):
        return None
    return val


def _fmt(val) -> str:
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.5g}"
    return str(val)


def _native(arr: np.ndarray) -> np.ndarray:
    """FITS data are big-endian; pandas and some numpy routines want native byte order."""
    arr = np.asarray(arr)
    return arr.astype(arr.dtype.newbyteorder("="))


def file_kind(hdul) -> str:
    names = [h.name for h in hdul]
    if "PIXELS" in names:
        return "tpf"
    if "LIGHTCURVE" in names:
        return "lc"
    return "other"


def aperture_mask(hdul) -> np.ndarray:
    return (np.asarray(hdul["APERTURE"].data) & AP_OPTIMAL) > 0


def aperture_flux(hdul) -> np.ndarray:
    """Sum of the aperture pixels in every frame (NaN where any of them is NaN)."""
    flux = np.asarray(hdul["PIXELS"].data["FLUX"], dtype=float)
    return flux[:, aperture_mask(hdul)].sum(axis=1)


def table_to_dataframe(source):
    """Per-cadence table of a SPOC file as a pandas DataFrame.

    Light curves: every 1-D column. Pixel files: every 1-D column plus
    APERTURE_FLUX, the aperture-summed flux, in place of the image columns.
    `source` is a path or an open HDUList.
    """
    import pandas as pd

    if isinstance(source, (str, Path)):
        with fits.open(source) as hdul:
            return table_to_dataframe(hdul)
    hdul = source
    table = hdul[1]
    cols = {}
    for col in table.columns:
        arr = np.asarray(table.data[col.name])
        if arr.ndim == 1:
            cols[col.name] = _native(arr)
    if file_kind(hdul) == "tpf":
        cols["APERTURE_FLUX"] = aperture_flux(hdul)
    return pd.DataFrame(cols)


# ============================================================================ finding files
def find_files(tic: int | None = None, sector: int | None = None, kind: str = "both") -> list[Path]:
    pattern = f"*-{int(tic):016d}-*.fits" if tic is not None else "*.fits"
    files = sorted(CACHE_DIR.rglob(pattern)) if CACHE_DIR.exists() else []
    if sector is not None:
        files = [f for f in files if f"-s{int(sector):04d}-" in f.name]
    if kind != "both":
        files = [f for f in files if f.name.endswith(f"_{kind}.fits")]
    return files


def download(tic: int, sector: int | None, kind: str) -> list[Path]:
    """Fetch the SPOC 2-min file(s) for one star from MAST into data/cache/."""
    from astroquery.mast import Observations

    from src.pipeline_core import download_spoc_product

    crit = dict(obs_collection="TESS", dataproduct_type="timeseries", target_name=str(int(tic)),
                provenance_name="SPOC", t_exptime=[119, 121])
    if sector is not None:
        crit["sequence_number"] = int(sector)
    obs = Observations.query_criteria(**crit)
    if len(obs) == 0:
        sys.exit(f"MAST has no SPOC 2-min data for TIC {tic}" + (f" in sector {sector}" if sector else ""))
    obs_ids = sorted(set(obs["obs_id"]))
    if sector is None and len(obs_ids) > 1:
        sectors = sorted(int(NAME_RE.search(o + "_lc.fits").group(1)) for o in obs_ids)
        print(f"TIC {tic} has 2-min data in sectors {sectors}; downloading sector {sectors[0]}. "
              "Use --sector to choose another.")
        obs_ids = obs_ids[:1]
    kinds = ["lc", "tp"] if kind == "both" else [kind]
    paths = []
    for obs_id in obs_ids:
        for k in kinds:
            cached = sorted(CACHE_DIR.rglob(f"{obs_id}_{k}.fits"))
            if cached:
                paths.append(cached[0])
            else:
                print(f"Downloading {obs_id}_{k}.fits ...")
                paths.append(download_spoc_product(obs_id, kind=k))
    return paths


def list_cache() -> None:
    files = find_files()
    if not files:
        print(f"No FITS files in {CACHE_DIR.relative_to(ROOT)}/. "
              "Use --tic ID --download to fetch one.")
        return
    print(f"{len(files)} FITS file(s) in {CACHE_DIR.relative_to(ROOT)}/:\n")
    print(f"  {'TIC':>12}  {'sector':>6}  {'type':<11} {'size':>8}  path")
    for f in files:
        m = NAME_RE.search(f.name)
        tic, sector, kind = (int(m.group(2)), int(m.group(1)), m.group(3)) if m else ("?", "?", "?")
        label = {"lc": "light curve", "tp": "pixels"}.get(kind, "?")
        print(f"  {tic:>12}  {sector:>6}  {label:<11} {f.stat().st_size / 1e6:>6.1f} MB  {f.relative_to(ROOT)}")


# ============================================================================ printing
def describe_structure(hdul) -> None:
    print("\nWhat's in the file (HDUs):")
    for i, h in enumerate(hdul):
        if isinstance(h, fits.BinTableHDU):
            what = f"table, {h.header['NAXIS2']} rows x {len(h.columns)} columns"
        elif h.header.get("NAXIS", 0) == 0:
            what = "header only"
        else:
            shape = [h.header[f"NAXIS{n}"] for n in range(h.header["NAXIS"], 0, -1)]
            what = "image " + " x ".join(map(str, shape))
        print(f"  [{i}] {h.name:<18} {what}  ({len(h.header)} header cards)")


def describe_star(header) -> None:
    print("\nThe star (primary header):")
    for key, unit, meaning in STAR_KEYS:
        if key in header:
            print(f"  {key:<8} {_fmt(_value(header, key)):>14} {unit:<7} {meaning}")


def describe_columns(table) -> None:
    print(f"\nColumns of [{table.name}] (one row per cadence):")
    rows = []
    for i, col in enumerate(table.columns, start=1):
        desc = table.header.comments[f"TTYPE{i}"].replace("column title:", "").strip()
        rows.append((col.name, col.dim or "", col.unit or "", desc))
    widths = [max(len(r[k]) for r in rows) for k in range(3)]
    print(f"  {'name':<{widths[0]}}  {'shape':<{max(widths[1], 5)}}  {'unit':<{widths[2]}}  description")
    for name, shape, unit, desc in rows:
        print(f"  {name:<{widths[0]}}  {shape:<{max(widths[1], 5)}}  {unit:<{widths[2]}}  {desc}")


def describe_time(table) -> np.ndarray:
    t = np.asarray(table.data["TIME"], dtype=float)
    ok = np.isfinite(t)
    tt = t[ok]
    start, end = tt.min(), tt.max()
    dates = Time([start + BTJD_OFFSET, end + BTJD_OFFSET], format="jd", scale="tdb").utc.iso
    gaps = np.diff(tt)
    g = int(np.argmax(gaps))
    print("\nObserving window:")
    print(f"  BTJD {start:.3f} to {end:.3f}  =  {dates[0][:16]} to {dates[1][:16]} UTC  ({end - start:.1f} days)")
    print(f"  one cadence every {np.median(gaps) * 1440:.1f} min; {len(t)} cadences, {(~ok).sum()} without a timestamp")
    print(f"  longest gap: {gaps[g]:.2f} days from BTJD {tt[g]:.2f} (data downlink or a spacecraft event)")
    return t


def describe_quality(quality: np.ndarray, flux_ok: np.ndarray) -> None:
    from lightkurve.utils import TessQualityFlags

    q = _native(quality).astype(np.int64)
    good = (q == 0) & flux_ok
    print("\nData quality:")
    print(f"  usable cadences (QUALITY = 0 and flux present): {good.sum()} of {len(q)} ({100 * good.mean():.1f} %)")
    if (q != 0).any():
        print("  why cadences were flagged (one cadence can carry several flags):")
        for bit, desc in TessQualityFlags.STRINGS.items():
            n = int(((q & bit) != 0).sum())
            if n:
                print(f"    {n:>6}  {desc}")


def _scatter_ppm(f: np.ndarray) -> tuple[float, float, float]:
    """Median, overall robust scatter and point-to-point scatter (ppm of the median)."""
    med = float(np.median(f))
    overall = 1.4826 * np.median(np.abs(f - med)) / med * 1e6
    p2p = 1.4826 * np.median(np.abs(np.diff(f))) / np.sqrt(2) / med * 1e6
    return med, overall, p2p


def describe_lightcurve(hdul) -> None:
    table = hdul["LIGHTCURVE"]
    d, h = table.data, table.header
    describe_time(table)
    pdc_ok = np.isfinite(np.asarray(d["PDCSAP_FLUX"], dtype=float))
    describe_quality(d["QUALITY"], pdc_ok)

    print("\nBrightness (usable cadences):")
    for col, meaning in [("SAP_FLUX", "raw sum of the aperture pixels"),
                         ("PDCSAP_FLUX", "SAP with spacecraft systematics removed; used for the hunt")]:
        f = np.asarray(d[col], dtype=float)
        good = np.isfinite(f) & (_native(d["QUALITY"]) == 0)
        med, overall, p2p = _scatter_ppm(f[good])
        print(f"  {col:<12} median {med:>10,.0f} e-/s   scatter {overall:>6,.0f} ppm overall, "
              f"{p2p:>6,.0f} ppm point-to-point   ({meaning})")
    print("  (1 % = 10,000 ppm. 'Point-to-point' is the noise of a single 2-min value; "
          "'overall' also includes slow trends.)")

    print("\nPhotometry facts from the [LIGHTCURVE] header:")
    facts = [
        ("CDPP0_5", "ppm", "noise when averaged over 30 min"),
        ("CDPP1_0", "ppm", "noise when averaged over 1 hour"),
        ("CDPP2_0", "ppm", "noise when averaged over 2 hours (a transit must be several times deeper)"),
        ("CROWDSAP", "", "share of the aperture's light that comes from this star (1 = no contamination)"),
        ("FLFRCSAP", "", "share of this star's light that the aperture captures"),
        ("PDCMETHD", "", "method used to remove systematics"),
    ]
    for key, unit, meaning in facts:
        if key in h:
            print(f"  {key:<9} {_fmt(_value(h, key)):>10} {unit:<4} {meaning}")


def describe_pixels(hdul) -> None:
    table = hdul["PIXELS"]
    d = table.data
    describe_time(table)
    flux = np.asarray(d["FLUX"], dtype=float)            # (frames, rows, cols)
    ap_flux = aperture_flux(hdul)
    describe_quality(d["QUALITY"], np.isfinite(ap_flux))

    nfr, ny, nx = flux.shape
    ap = np.asarray(hdul["APERTURE"].data)
    optimal, bkg = (ap & AP_OPTIMAL) > 0, (ap & AP_BACKGROUND) > 0
    good = np.isfinite(ap_flux) & (_native(d["QUALITY"]) == 0)
    median_frame = np.nanmedian(flux[good], axis=0)
    r, c = np.unravel_index(np.nanargmax(median_frame), median_frame.shape)
    side = TESS_PIXEL_ARCSEC / 60

    print("\nPixels:")
    print(f"  {nfr} frames of {ny} x {nx} pixels; 1 pixel = {TESS_PIXEL_ARCSEC:.0f}\", "
          f"so the stamp covers {ny * side:.1f}' x {nx * side:.1f}' of sky")
    print(f"  aperture: {optimal.sum()} pixels are summed to make the light curve; "
          f"{bkg.sum()} pixels were used for the background")
    print(f"  brightest pixel in the median frame: row {r}, col {c} ({median_frame[r, c]:,.0f} e-/s); "
          f"stamp centre: row {ny // 2}, col {nx // 2}")
    corner_col, corner_row = _value(table.header, "1CRV4P"), _value(table.header, "2CRV4P")
    if corner_col is not None:
        print(f"  stamp's lower-left corner on the detector: column {corner_col}, row {corner_row}")
    med, overall, p2p = _scatter_ppm(ap_flux[good])
    print(f"  aperture-summed flux: median {med:,.0f} e-/s, scatter {overall:,.0f} ppm overall, "
          f"{p2p:,.0f} ppm point-to-point")


def dump_headers(hdul) -> None:
    for i, h in enumerate(hdul):
        print(f"\n{'-' * 30} header [{i}] {h.name} {'-' * 30}")
        print(repr(h.header))


# ============================================================================ outputs
def plot_quicklook(hdul, kind: str, path: Path, out_dir: Path = INSPECT_DIR) -> Path:
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    from src.pipeline_core import outline_mask  # also selects the Agg backend

    hdr = hdul[0].header
    title = f"TIC {_value(hdr, 'TICID')}  ·  Sector {_value(hdr, 'SECTOR')}"
    table = hdul[1]
    t = np.asarray(table.data["TIME"], dtype=float)
    q = _native(table.data["QUALITY"])

    if kind == "lc":
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for ax, col, label in [(axes[0], "SAP_FLUX", "SAP (raw)"),
                               (axes[1], "PDCSAP_FLUX", "PDCSAP (systematics removed)")]:
            f = np.asarray(table.data[col], dtype=float)
            good = np.isfinite(t) & np.isfinite(f) & (q == 0)
            ax.plot(t[good], f[good] / np.median(f[good]), ".", ms=1, color="k")
            ax.set_ylabel("relative flux")
            ax.set_title(label, loc="left", fontsize=10)
        axes[1].set_xlabel("time [BTJD = BJD - 2457000]")
        fig.suptitle(f"{title}  ·  light curve")
    else:
        flux = np.asarray(table.data["FLUX"], dtype=float)
        ap_flux = aperture_flux(hdul)
        good = np.isfinite(t) & np.isfinite(ap_flux) & (q == 0)
        median_frame = np.nanmedian(flux[good], axis=0)
        fig = plt.figure(figsize=(14, 4.8))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.6])
        ax_img, ax_lc = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        positive = median_frame[np.isfinite(median_frame) & (median_frame > 0)]
        norm = simple_norm(positive, "log", percent=99.5)
        im = ax_img.imshow(median_frame, origin="lower", cmap="viridis", norm=norm)
        outline_mask(ax_img, aperture_mask(hdul), color="red")
        cb = fig.colorbar(im, ax=ax_img, label="e-/s", fraction=0.046)
        lo, hi = max(float(norm.vmin), 1.0), float(norm.vmax)
        cb.set_ticks(sorted({float(f"{v:.2g}") for v in np.geomspace(lo, hi, 5)}))
        ax_img.set_title("median frame + aperture (red)", fontsize=10)
        ax_img.set_xlabel("column [px]")
        ax_img.set_ylabel("row [px]")
        ax_lc.plot(t[good], ap_flux[good] / np.median(ap_flux[good]), ".", ms=1, color="k")
        ax_lc.set_xlabel("time [BTJD = BJD - 2457000]")
        ax_lc.set_ylabel("relative flux")
        ax_lc.set_title("aperture-summed light curve", loc="left", fontsize=10)
        fig.suptitle(f"{title}  ·  target pixel file")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def export_csv(hdul, path: Path, out_dir: Path = INSPECT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{path.stem}.csv"
    table_to_dataframe(hdul).to_csv(out, index=False)
    return out


# ============================================================================ main
def inspect(path: Path, header: bool = False, plot: bool = False, csv: bool = False) -> None:
    path = Path(path)
    print("=" * 100)
    print(f"{path.name}  ({path.stat().st_size / 1e6:.1f} MB)")
    with fits.open(path) as hdul:
        kind = file_kind(hdul)
        print(KIND_TEXT[kind])
        describe_structure(hdul)
        describe_star(hdul[0].header)
        if kind != "other":
            describe_columns(hdul[1])
            (describe_lightcurve if kind == "lc" else describe_pixels)(hdul)
        if header:
            dump_headers(hdul)
        if plot and kind != "other":
            print(f"\nQuick-look plot: {plot_quicklook(hdul, kind, path).relative_to(ROOT)}")
        if csv and kind != "other":
            print(f"Per-cadence table: {export_csv(hdul, path).relative_to(ROOT)}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Read TESS SPOC FITS files (light curves and target pixel files) and explain their contents.",
        epilog="Plots and CSVs are written to output/fits_inspect/.",
    )
    ap.add_argument("paths", nargs="*", type=Path, help="FITS file(s) to read")
    ap.add_argument("--list", action="store_true", help="list the FITS files in data/cache/ and exit")
    ap.add_argument("--tic", type=int, help="read the cached files for this TIC ID")
    ap.add_argument("--sector", type=int, help="with --tic: only this sector")
    ap.add_argument("--kind", choices=["lc", "tp", "both"], default="both",
                    help="with --tic: light curve (lc), pixel file (tp) or both (default)")
    ap.add_argument("--download", action="store_true", help="with --tic: fetch missing files from MAST")
    ap.add_argument("--header", action="store_true", help="also print every header card")
    ap.add_argument("--plot", action="store_true", help="save a quick-look PNG")
    ap.add_argument("--csv", action="store_true", help="export the per-cadence table to CSV")
    args = ap.parse_args()

    if args.list:
        list_cache()
        return

    paths = list(args.paths)
    if args.tic is not None:
        found = find_files(args.tic, args.sector, args.kind)
        if args.download and (not found or (args.kind == "both" and len({f.name[-7:] for f in found}) < 2)):
            found = download(args.tic, args.sector, args.kind)
        if not found:
            sys.exit(f"No cached FITS file for TIC {args.tic}"
                     + (f" in sector {args.sector}" if args.sector else "")
                     + ". Add --download to fetch it from MAST.")
        paths += found
    if not paths:
        ap.print_help()
        sys.exit(1)

    for p in paths:
        if not Path(p).exists():
            sys.exit(f"File not found: {p}")
        inspect(p, header=args.header, plot=args.plot, csv=args.csv)


if __name__ == "__main__":
    main()
