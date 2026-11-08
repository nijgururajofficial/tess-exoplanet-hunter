# TESS Exoplanet Hunter

A validate-first pipeline for finding transiting-planet candidates in NASA TESS 2-minute light curves.
It searches each star for periodic dips with Box Least Squares (BLS), rejects common false positives
automatically, and checks the strongest candidates at the pixel level for background blends.

The pipeline is checked on a known planet before it hunts:

1. **Validate** – rediscover the hot Saturn WASP-126 b (TIC 25155310) from raw Sector 1 pixels.
2. **Hunt** – scan 2-minute stars in a sector that have no known planet or TOI, in parallel.
3. **Vet** – centroid, difference-image and Gaia DR3 checks on a candidate's Target Pixel File.

Built with Python 3.12, [lightkurve](https://github.com/lightkurve/lightkurve), [astropy](https://www.astropy.org/) and [astroquery](https://astroquery.readthedocs.io/).

## Results so far

**Phase 1 – WASP-126 b recovered** from both SPOC aperture photometry of the pixel file and the PDCSAP light curve:

| Quantity | Pixel file | PDCSAP | NASA Exoplanet Archive |
|---|---|---|---|
| Period [d] | 3.28842 | 3.28842 | 3.28883 |
| Depth | 0.60 % | 0.62 % | 0.70 % |
| Rp/R* | 0.077 | 0.079 | 0.078 |
| Rp [R⊕] | 10.5 | 10.7 | 10.5 |
| Odd/even difference | 2.7σ | 1.9σ | – |
| Transits observed | 8 | 8 | – |

A BLS box underestimates a limb-darkened transit's depth a little, so ~0.6 % against the catalogue's 0.70 % is expected.

**Phase 2 test** – 25 random stars from Sector 14 with no known planet or TOI: 0 candidates and 0 errors, in about 30–40 s with 6 workers.
The vetting rejected several eclipsing binaries and one stellar-variability signal (TIC 359587754).
Pixel-level vetting placed that signal on the star itself.

## Setup

Requires [uv](https://docs.astral.sh/uv/) (macOS: `brew install uv`).

```bash
git clone https://github.com/nijgururajofficial/tess-exoplanet-hunter.git
cd tess-exoplanet-hunter
uv sync                 # creates .venv with Python 3.12 and all dependencies
uv run pytest -q        # offline injection-recovery tests (~20 s)
```

All data is fetched on demand from MAST, the NASA Exoplanet Archive and the Gaia archive, so no manual downloads are needed.

## Usage

### Phase 1 – validate on a known planet

```bash
uv run python src/validate_known_planet.py
```

Downloads the Sector 1 pixel file for TIC 25155310 and runs the full pipeline on its aperture photometry and on the PDCSAP light curve.
It prints a comparison table against live NASA Exoplanet Archive values and exits non-zero if any acceptance check fails.
Acceptance checks: period within 0.005 d, depth within ±20 %, Rp/R* within ±15 %, odd/even < 3σ, ≥ 3 transits, SDE ≥ 8.

### Phase 2 – hunt for new candidates

```bash
uv run python src/hunt_new_planets.py --sector 14 --limit 25 --workers 6 --snr-threshold 7.5
```

| Flag | Default | Meaning |
|---|---|---|
| `--sector` | required | TESS sector to scan |
| `--limit` | 25 | maximum number of stars to process this run |
| `--workers` | 6 | parallel processes |
| `--snr-threshold` | 8.0 | minimum transit SNR (depth / depth error) |
| `--sde-threshold` | 7.0 | minimum BLS signal detection efficiency |
| `--seed` | 42 | random seed for choosing which stars to scan |
| `--refresh-known` | off | re-download the known-planet exclusion list (otherwise refreshed after 7 days) |
| `--refresh-targets` | off | re-query MAST for the sector's target list |
| `--keep-cache` | off | keep downloaded FITS files for rejected stars |

Runs are resumable. Every processed star is logged, and a re-run skips stars already scanned.
To scan a whole sector, raise `--limit` above its star count (Sector 14 has 19,626 stars after exclusions; roughly 6–9 h at the current speed).

### Phase 2b – pixel-level vetting

```bash
uv run python src/vet_candidate_tpf.py --tic 25155310 --period 3.28842 --t0 1327.52063 --sector 1
```

`--t0` is a mid-transit time in BTJD (BJD − 2457000). `--duration` (days) is estimated if omitted, and `--gaia-radius` defaults to 60″.
Copy the period and t0 from `output/new_candidates.csv`.

The script produces:
- **Centroid test** – in- vs out-of-transit flux-weighted centroid shift, with each transit compared against its own local baseline. Flagged above 3σ.
- **Difference image** – mean out-of-transit minus mean in-transit frame. Flagged if the dip's source is more than 1 px (21″) from the star.
- **Per-pixel grid** – a phase-folded light curve for every pixel in the stamp.
- **Gaia DR3 overlay** – neighbours within 60″, with proper motions propagated to the TESS epoch. Stars bright enough to fake the observed depth are highlighted.

### Inspecting FITS files

```bash
uv run python src/inspect_fits.py --list                    # FITS files in data/cache/
uv run python src/inspect_fits.py --tic 25155310 --plot --csv
uv run python src/inspect_fits.py --tic 219900423 --sector 14 --kind lc --download
```

Prints a plain-language summary of a light curve or pixel file:
- the file's structure (HDUs)
- the star's properties from the header
- every column, with its unit and meaning
- the observing window and largest gap
- how many cadences are usable, and why the others were flagged
- noise levels (scatter, CDPP) and aperture contamination

`--plot` saves a quick-look figure and `--csv` exports the per-cadence table, both to `output/fits_inspect/`. `--header` prints every header card.
In Python, `from src.inspect_fits import table_to_dataframe` loads a file's table as a pandas DataFrame.

## Vetting rules

A signal becomes a candidate only if it passes every hard rule:

| Rule | Rejects |
|---|---|
| SNR ≥ 8 and SDE ≥ 7 | noise and weak systematics |
| ≥ 3 transits, each with ≥ 50 % of its expected data points | one-off events |
| odd/even depth difference ≤ 3σ | eclipsing binaries found at half their true period |
| no secondary eclipse (> 3σ and > 5 % of primary depth) | eclipsing binaries |
| Rp ≤ 22 R⊕ (needs stellar radius) | stellar companions |
| duration ≤ 2 × the longest possible transit for the star (Kepler's third law) | variable stars and blends |
| a box fits better than a sine at P or P/2 | stellar variability |

Soft flags are reported but don't reject: one transit carrying most of the signal, a period near a harmonic of TESS's 13.7 d orbit, and a missing stellar radius.

How the error bars are estimated:
- The detrending runs twice. The second pass masks the transit so the filter can't flatten part of the dip away.
- The SNR and odd/even errors are scaled up by the measured transit-to-transit scatter, not the white-noise estimate alone.
- Odd and even transits are scaled separately, so a genuine binary's alternating depths can't hide inside its own error bar.

All thresholds live in [`src/config.py`](src/config.py).

## Outputs

| Path | Contents |
|---|---|
| `output/plots/validation/` | Phase 1 figures and comparison table |
| `output/plots/candidates/TIC_<id>_candidate.png` | figure for each candidate |
| `output/plots/candidates/TIC_<id>_deep_vetting.png` / `.json` | pixel-vetting figure and report |
| `output/scan_log_s<N>.csv` | metrics and verdict for every star scanned |
| `output/new_candidates.csv` | candidates ranked by SNR |
| `output/fits_inspect/` | quick-look plots and CSV exports from `inspect_fits.py` |
| `data/cache/` | downloaded FITS files (rejected stars' files deleted automatically) |
| `data/known_planet_tics.csv` | known TOI and confirmed-planet TIC IDs excluded from the hunt |

Each diagnostic figure has four panels:
- the pixel stamp with the photometric aperture
- the flattened light curve with predicted transit times marked
- the BLS periodogram
- the phase-folded transit, with odd and even transits shown separately

`data/` and `output/` are generated locally and are not tracked in git.

## Project layout

```
src/
  config.py                 paths, constants and all vetting thresholds
  pipeline_core.py          detrending, BLS search, vetting, plotting, MAST downloads
  validate_known_planet.py  Phase 1
  hunt_new_planets.py       Phase 2
  vet_candidate_tpf.py      Phase 2b
  inspect_fits.py           read and summarise any SPOC FITS file
tests/
  test_pipeline_core.py     synthetic transit, binary, variability and noise tests
```

## Limitations

- Each sector is searched on its own, so periods are effectively capped at about 13 days (3 transits in about 27 days). Multi-sector stitching isn't implemented yet.
- The shortest trial transit duration is about 1 hour, so very short transits are detected with diluted depth.
- The vetting rules have been checked on one known planet and synthetic data. A benchmark against the known TOIs in a sector would measure how often real planets are missed and false positives slip through.
- A candidate here is a signal worth follow-up, not a planet. Confirmation needs independent observations.

## Acknowledgements

This project uses data from the TESS mission, obtained from the Mikulski Archive for Space Telescopes (MAST); the NASA Exoplanet Archive, operated by Caltech under contract with NASA; and the ESA Gaia mission (DR3).
Analysis relies on lightkurve, astropy and astroquery.
