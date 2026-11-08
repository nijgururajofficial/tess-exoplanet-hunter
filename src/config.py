"""Single source of truth for paths, physical constants and vetting thresholds.

CLAUDE.md, the CLI defaults and pipeline_core all refer to these values, so a
threshold change happens here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"                     # all MAST downloads land here (never ~/.lightkurve)
KNOWN_TICS_CSV = DATA_DIR / "known_planet_tics.csv"
OUTPUT_DIR = ROOT / "output"
PLOTS_VALIDATION = OUTPUT_DIR / "plots" / "validation"
PLOTS_CANDIDATES = OUTPUT_DIR / "plots" / "candidates"
CANDIDATES_CSV = OUTPUT_DIR / "new_candidates.csv"
INSPECT_DIR = OUTPUT_DIR / "fits_inspect"          # quick-look plots / CSVs from inspect_fits.py

# ---------------------------------------------------------------- physical constants
R_SUN_IN_R_EARTH = 109.076          # R_sun / R_earth (equatorial)
R_JUP_IN_R_EARTH = 11.209
TESS_PIXEL_ARCSEC = 21.0            # TESS plate scale
TESS_ORBIT_DAYS = 13.7              # spacecraft orbit -> known systematic period
BTJD_OFFSET = 2457000.0             # BTJD = BJD - 2457000


@dataclass(frozen=True)
class Thresholds:
    # --- detrending
    flatten_window: int = 401               # cadences (401 x 2 min = 13.4 h); must be odd
    flatten_polyorder: int = 2
    transit_mask_pad: float = 1.5           # mask width = pad x BLS duration in the 2nd flatten pass
    sigma_upper: float = 3.0                # clip bright outliers only ...
    sigma_lower: float = 20.0               # ... never clip transit dips

    # --- BLS search grid
    period_min: float = 0.5                 # days
    period_max: float = 20.0                # days (also capped at baseline / (min_n_transits - 1))
    durations: tuple = field(default_factory=lambda: tuple(np.linspace(0.04, 0.25, 12)))
    frequency_factor: float = 1.0           # astropy autoperiod spacing; <1 = finer grid
    bls_oversample: int = 10

    # --- hard gates (all must pass to be a candidate)
    snr_min: float = 8.0                    # depth / depth_err
    sde_min: float = 7.0                    # signal detection efficiency of the BLS peak
    min_n_transits: int = 3                 # transits with >= half their expected cadences
    odd_even_sigma_max: float = 3.0         # |d_odd - d_even| / sigma  > this  -> eclipsing binary
    secondary_sigma_max: float = 3.0        # phase-0.5 dip significance ...
    secondary_ratio_max: float = 0.05       # ... AND depth_sec / depth > this -> eclipsing binary
    rp_max_earth: float = 22.0              # ~2 R_Jup; larger is stellar
    duration_ratio_max: float = 2.0         # observed / max central duration (Kepler 3rd law); 2x leaves
                                            # room for R* errors and eccentric orbits
    reject_sinusoid_preferred: bool = True  # a sine at P fits better than the box -> variability

    # --- soft flags (reported, never reject)
    single_event_frac_max: float = 0.5      # one transit carries > this share of the likelihood
    orbit_harmonic_tol: float = 0.01        # fractional distance to 13.7 d / n

    # --- Phase 1 reporting
    odd_even_frac_max: float = 0.10


DEFAULT = Thresholds()
