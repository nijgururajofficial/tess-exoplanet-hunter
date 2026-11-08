"""Shared TESS transit-search machinery.

Detrending -> Box Least Squares search -> automated false-positive vetting ->
4-panel diagnostic figure.  Used by Phase 1 (validation), Phase 2 (bulk hunt)
and Phase 2b (pixel vetting).

Conventions
-----------
* Times are BTJD (BJD - 2457000) floats in days.
* Light curves are normalised to a median of 1; depth is a positive fraction.
* FITS science arrays are never converted to 8-bit images.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=".*oktopus.*")

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import lightkurve as lk  # noqa: E402
from astropy.timeseries import BoxLeastSquares  # noqa: E402
from astropy.visualization import simple_norm  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402

from src.config import (  # noqa: E402
    CACHE_DIR,
    DEFAULT,
    R_SUN_IN_R_EARTH,
    TESS_ORBIT_DAYS,
    Thresholds,
)

G_SI = 6.674e-11
M_SUN = 1.989e30
R_SUN = 6.957e8


# ============================================================================ helpers
def transit_mask(t: np.ndarray, period: float, width: float, t0: float) -> np.ndarray:
    """True for cadences within +/- width/2 of any predicted mid-transit."""
    hp = 0.5 * period
    return np.abs((t - t0 + hp) % period - hp) < 0.5 * width


def phase_days(t: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Time from nearest mid-transit, in days, in [-P/2, P/2)."""
    return (t - t0 + 0.5 * period) % period - 0.5 * period


def _f(x) -> float:
    """Strip astropy units / numpy scalars to a plain float."""
    return float(getattr(x, "value", x))


def _robust_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def bin_series(x: np.ndarray, y: np.ndarray, width: float, min_count: int = 3):
    """Mean-bin y(x) in bins of `width`. Returns centres, means, standard errors."""
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])
    edges = np.arange(x.min(), x.max() + width, width)
    idx = np.digitize(x, edges) - 1
    cs, ms, es = [], [], []
    for i in np.unique(idx):
        sel = idx == i
        n = sel.sum()
        if n >= min_count:
            cs.append(x[sel].mean())
            ms.append(y[sel].mean())
            es.append(y[sel].std(ddof=1) / math.sqrt(n))
    return np.array(cs), np.array(ms), np.array(es)


# ======================================================================= detrending
def clean_and_flatten(lc: lk.LightCurve, th: Thresholds = DEFAULT, ephemeris=None):
    """Remove NaNs, normalise, flatten, clip bright outliers only.

    ephemeris : (period, t0, duration) or None. When given, in-transit cadences
        (pad x duration wide) are excluded from the Savitzky-Golay trend fit so the
        filter cannot eat the transit depth.

    Returns (flat, normalised_raw, trend).
    """
    norm = lc.remove_nans().normalize()
    mask = None
    if ephemeris is not None:
        period, t0, dur = ephemeris
        mask = transit_mask(norm.time.value, period, th.transit_mask_pad * dur, t0)
    flat, trend = norm.flatten(
        window_length=th.flatten_window,
        polyorder=th.flatten_polyorder,
        return_trend=True,
        mask=mask,
    )
    keep = np.isfinite(flat.flux.value)
    flat, trend = flat[keep], trend[keep]
    flat = flat.remove_outliers(sigma_lower=th.sigma_lower, sigma_upper=th.sigma_upper)
    return flat, norm, trend


# ============================================================================== BLS
@dataclass
class BLSResult:
    period: float
    t0: float
    duration: float
    depth: float
    depth_err: float          # white-noise error inflated by the red-noise factor beta
    snr: float                # depth / depth_err
    sde: float                # signal detection efficiency of the peak
    beta: float               # red-noise inflation factor (>= 1)
    max_power: float          # raw BLS log-likelihood power (NOT an SNR)
    n_transits: int           # transits with >= 50 % of their expected cadences
    baseline: float
    periods: np.ndarray = field(repr=False)
    power: np.ndarray = field(repr=False)
    stats: dict = field(repr=False, default_factory=dict)


def _noise_errors(lc: lk.LightCurve) -> np.ndarray:
    """Per-cadence errors rescaled so their median matches the empirical scatter."""
    y = lc.flux.value
    scatter = _robust_std(y)
    err = getattr(lc, "flux_err", None)
    err = None if err is None else np.asarray(err.value, dtype=float)
    if err is None or not np.isfinite(err).any() or np.nanmedian(err) <= 0:
        return np.full_like(y, scatter)
    err = np.where(np.isfinite(err) & (err > 0), err, np.nanmedian(err))
    return err * (scatter / np.median(err))


def _red_noise_beta(t, y, in_transit, duration) -> float:
    """Pont et al. (2006) beta: binned OOT scatter vs white-noise expectation."""
    t_o, y_o = t[~in_transit], y[~in_transit]
    if t_o.size < 50:
        return 1.0
    cadence = np.median(np.diff(t_o))
    n_per_bin = max(int(round(duration / cadence)), 2)
    _, means, _ = bin_series(t_o, y_o, duration, min_count=max(int(0.8 * n_per_bin), 2))
    if means.size < 5:
        return 1.0
    expected = _robust_std(y_o) / math.sqrt(n_per_bin)
    return float(max(1.0, np.std(means, ddof=1) / expected))


def run_bls(lc: lk.LightCurve, th: Thresholds = DEFAULT) -> BLSResult:
    t = np.asarray(lc.time.value, dtype=float)
    y = np.asarray(lc.flux.value, dtype=float)
    dy = _noise_errors(lc)
    ok = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy)
    t, y, dy = t[ok], y[ok], dy[ok]

    baseline = float(t.max() - t.min())
    durations = np.array([d for d in th.durations if d < th.period_min])
    pmax = min(th.period_max, baseline / max(th.min_n_transits - 1, 1))

    bls = BoxLeastSquares(t, y, dy)
    periods = bls.autoperiod(
        durations,
        minimum_period=th.period_min,
        maximum_period=pmax,
        minimum_n_transit=th.min_n_transits,
        frequency_factor=th.frequency_factor,
    )
    res = bls.power(periods, durations, objective="likelihood", oversample=th.bls_oversample)
    power = np.asarray(res.power, dtype=float)
    periods = np.asarray(_f_arr(res.period), dtype=float)
    i = int(np.nanargmax(power))
    period, t0, duration = float(periods[i]), _f(res.transit_time[i]), _f(res.duration[i])

    raw = bls.compute_stats(period, duration, t0)
    stats = {
        "depth": tuple(map(_f, raw["depth"])),
        "depth_odd": tuple(map(_f, raw["depth_odd"])),
        "depth_even": tuple(map(_f, raw["depth_even"])),
        "depth_phased": tuple(map(_f, raw["depth_phased"])),
        "depth_half": tuple(map(_f, raw["depth_half"])),
        "harmonic_delta_log_likelihood": _f(raw["harmonic_delta_log_likelihood"]),
        "transit_times": np.asarray(_f_arr(raw["transit_times"]), dtype=float),
        "per_transit_count": np.asarray(raw["per_transit_count"]),
        "per_transit_log_likelihood": np.asarray(_f_arr(raw["per_transit_log_likelihood"]), dtype=float),
    }
    depth, depth_err_white = stats["depth"]

    in_tr = transit_mask(t, period, duration, t0)
    cadence = float(np.median(np.diff(t)))
    expected_pts = duration / cadence

    # Per-transit depths: the empirical transit-to-transit scatter exposes noise that
    # white-noise errors miss (e.g. trend interpolation across each masked transit).
    ivar = 1.0 / dy ** 2
    y_out = np.sum(y[~in_tr] * ivar[~in_tr]) / np.sum(ivar[~in_tr])
    epoch = np.round((t - t0) / period).astype(int)
    deps, errs, eps = [], [], []
    for e in np.unique(epoch[in_tr]):
        sel = in_tr & (epoch == e)
        if sel.sum() < 0.5 * expected_pts:
            continue
        w = ivar[sel]
        deps.append(y_out - np.sum(y[sel] * w) / np.sum(w))
        errs.append(1.0 / math.sqrt(np.sum(w)))
        eps.append(e)
    deps, errs, eps = np.array(deps), np.array(errs), np.array(eps, dtype=int)
    stats["per_transit_depth"], stats["per_transit_depth_err"], stats["per_transit_epoch"] = deps, errs, eps
    n_transits = int(deps.size)

    # sqrt(reduced chi^2) of per-transit depths about their group mean (>= 1).
    # Odd/even scale uses within-parity scatter only, so a genuine odd/even
    # difference (eclipsing binary) cannot inflate its own error bar.
    k_all = _scatter_scale([(deps, errs)])
    odd = (eps % 2) == 1
    k_oe = _scatter_scale([(deps[odd], errs[odd]), (deps[~odd], errs[~odd])])
    stats["depth_err_scale"], stats["odd_even_err_scale"] = k_all, k_oe
    stats["harmonic_delta_log_likelihood"] = _sine_vs_box(t, y, ivar, period, duration, t0)

    beta = _red_noise_beta(t, y, in_tr, duration)
    depth_err = depth_err_white * max(beta, k_all)
    snr = depth / depth_err if depth_err > 0 and np.isfinite(depth_err) else 0.0

    # SDE on a running-median-detrended spectrum (BLS power rises toward long periods)
    size = max(101, (len(power) // 100) | 1)
    detr = power - median_filter(power, size=size, mode="nearest")
    sde = float((detr[i] - np.mean(detr)) / np.std(detr)) if np.std(detr) > 0 else 0.0

    return BLSResult(
        period=period, t0=t0, duration=duration, depth=float(depth), depth_err=float(depth_err),
        snr=float(snr), sde=sde, beta=beta, max_power=float(power[i]), n_transits=n_transits,
        baseline=baseline, periods=periods, power=power, stats=stats,
    )


def _f_arr(x):
    return getattr(x, "value", x)


def _sine_vs_box(t, y, ivar, period, duration, t0) -> float:
    """max over (P, P/2) of [lnL(sine + const) - lnL(box)]. > 0 means variability.

    astropy's compute_stats only tries a sine at P; checking P/2 as well catches
    sinusoidal variability that BLS has locked onto at twice its true period.
    """
    m_in = transit_mask(t, period, duration, t0)
    if not m_in.any() or m_in.all():
        return float("nan")
    y_in = np.sum(y[m_in] * ivar[m_in]) / np.sum(ivar[m_in])
    y_out = np.sum(y[~m_in] * ivar[~m_in]) / np.sum(ivar[~m_in])
    box_ll = -0.5 * (np.sum(ivar[m_in] * (y[m_in] - y_in) ** 2) + np.sum(ivar[~m_in] * (y[~m_in] - y_out) ** 2))
    best = -np.inf
    for p in (period, 0.5 * period):
        A = np.vstack((np.sin(2 * np.pi * t / p), np.cos(2 * np.pi * t / p), np.ones_like(t))).T
        w = np.linalg.solve(A.T @ (A * ivar[:, None]), A.T @ (y * ivar))
        best = max(best, -0.5 * np.sum(ivar * (y - A @ w) ** 2))
    return float(best - box_ll)


def _scatter_scale(groups) -> float:
    """max(1, sqrt(chi2/dof)) of values about each group's weighted mean."""
    chi2, dof = 0.0, 0
    for d, e in groups:
        if d.size < 2:
            continue
        w = 1.0 / e ** 2
        mu = np.sum(d * w) / np.sum(w)
        chi2 += float(np.sum(((d - mu) / e) ** 2))
        dof += d.size - 1
    return max(1.0, math.sqrt(chi2 / dof)) if dof > 0 else 1.0


@dataclass
class SearchOutput:
    result: BLSResult
    flat: lk.LightCurve
    norm: lk.LightCurve
    trend: lk.LightCurve
    n_passes: int


def search(lc: lk.LightCurve, th: Thresholds = DEFAULT, max_passes: int = 3) -> SearchOutput:
    """Iterative two-pass search.

    Pass 1 flattens blindly and runs BLS. Each further pass re-flattens the raw
    light curve with the current best transit masked and re-runs BLS, stopping
    once the period is stable to 0.1 %.
    """
    flat, norm, trend = clean_and_flatten(lc, th)
    res = run_bls(flat, th)
    n = 1
    while n < max_passes:
        flat, norm, trend = clean_and_flatten(lc, th, ephemeris=(res.period, res.t0, res.duration))
        new = run_bls(flat, th)
        n += 1
        stable = abs(new.period - res.period) / res.period < 1e-3
        res = new
        if stable:
            break
    return SearchOutput(result=res, flat=flat, norm=norm, trend=trend, n_passes=n)


# ================================================================== stellar params
def _clean_val(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def get_stellar_params(meta: dict, tic=None, allow_query: bool = True) -> dict:
    """R*, M*, Teff, logg, Tmag from the FITS header; TIC query fills the gaps."""
    star = {
        "radius": _clean_val(meta.get("RADIUS")),
        "teff": _clean_val(meta.get("TEFF")),
        "logg": _clean_val(meta.get("LOGG")),
        "tmag": _clean_val(meta.get("TESSMAG")),
        "mass": None,
        "source": "header",
    }
    if star["radius"] and star["logg"]:
        m = 10 ** (star["logg"] - 4.438) * star["radius"] ** 2
        star["mass"] = m if 0.08 <= m <= 10 else None

    if allow_query and tic is not None and (star["radius"] is None or star["mass"] is None):
        try:
            from astroquery.mast import Catalogs

            row = Catalogs.query_criteria(catalog="Tic", ID=int(tic))
            if len(row):
                row = row[0]
                for key, col in (("radius", "rad"), ("mass", "mass"), ("teff", "Teff"),
                                 ("logg", "logg"), ("tmag", "Tmag")):
                    if star[key] is None:
                        star[key] = _clean_val(row[col])
                star["source"] = "header+TIC"
        except Exception as exc:  # network trouble must not kill a bulk run
            star["source"] = f"header (TIC query failed: {type(exc).__name__})"
    return star


# ========================================================================== vetting
def max_transit_duration(period: float, r_star: float, m_star: float, rp_rs: float = 0.0) -> float:
    """Central-transit duration (days) for a circular orbit, via Kepler's 3rd law."""
    p_s = period * 86400.0
    a = (G_SI * m_star * M_SUN * p_s ** 2 / (4 * math.pi ** 2)) ** (1 / 3)
    x = min(1.0, r_star * R_SUN * (1 + rp_rs) / a)
    return period / math.pi * math.asin(x)


def vet(res: BLSResult, star: dict, th: Thresholds = DEFAULT) -> dict:
    """Automated false-positive vetting. Returns a flat dict of metrics + verdict."""
    s = res.stats
    depth = res.depth
    rp_rs = math.sqrt(depth) if depth > 0 else float("nan")
    r_star, m_star = star.get("radius"), star.get("mass")
    rp_earth = rp_rs * r_star * R_SUN_IN_R_EARTH if r_star else float("nan")

    d_odd, e_odd = s["depth_odd"]
    d_even, e_even = s["depth_even"]
    oe_den = math.hypot(e_odd, e_even) * s.get("odd_even_err_scale", 1.0)
    oe_sigma = abs(d_odd - d_even) / oe_den if np.isfinite(oe_den) and oe_den > 0 else float("nan")
    oe_max = max(d_odd, d_even)
    oe_frac = abs(d_odd - d_even) / oe_max if oe_max > 0 else float("nan")

    d_sec, e_sec = s["depth_phased"]
    sec_sigma = d_sec / e_sec if np.isfinite(e_sec) and e_sec > 0 and d_sec > 0 else 0.0
    sec_ratio = d_sec / depth if depth > 0 else float("nan")

    dur_max = float("nan")
    if r_star and m_star and np.isfinite(rp_rs):
        dur_max = max_transit_duration(res.period, r_star, m_star, rp_rs)
    dur_ratio = res.duration / dur_max if np.isfinite(dur_max) and dur_max > 0 else float("nan")
    # The BLS grid cannot fit a box shorter than its shortest trial duration, so a
    # short true transit (USP planet, small star) must not be rejected for that.
    dur_floor = max(dur_max, min(th.durations)) if np.isfinite(dur_max) else float("nan")
    too_long = bool(np.isfinite(dur_floor) and res.duration > th.duration_ratio_max * dur_floor)

    lls = s["per_transit_log_likelihood"]
    pos = lls[lls > 0]
    single_frac = float(pos.max() / pos.sum()) if pos.size else float("nan")

    harmonic_dist = min(abs(res.period - TESS_ORBIT_DAYS / n) / (TESS_ORBIT_DAYS / n) for n in range(1, 6))

    fail = []
    if not depth > 0:
        fail.append("non_positive_depth")
    if res.snr < th.snr_min:
        fail.append(f"snr<{th.snr_min:g}")
    if res.sde < th.sde_min:
        fail.append(f"sde<{th.sde_min:g}")
    if res.n_transits < th.min_n_transits:
        fail.append(f"n_transits<{th.min_n_transits}")
    if np.isfinite(oe_sigma) and oe_sigma > th.odd_even_sigma_max:
        fail.append("odd_even_mismatch")
    if sec_sigma > th.secondary_sigma_max and sec_ratio > th.secondary_ratio_max:
        fail.append("secondary_eclipse")
    if np.isfinite(rp_earth) and rp_earth > th.rp_max_earth:
        fail.append(f"rp>{th.rp_max_earth:g}Re")
    if too_long:
        fail.append("duration_too_long")
    sinusoid = s["harmonic_delta_log_likelihood"] > 0
    if sinusoid and th.reject_sinusoid_preferred:
        fail.append("sinusoid_preferred")

    flags = []
    if not r_star:
        flags.append("no_stellar_radius")
    if sinusoid and not th.reject_sinusoid_preferred:
        flags.append("sinusoid_preferred")
    if np.isfinite(single_frac) and single_frac > th.single_event_frac_max:
        flags.append("single_event_dominates")
    if harmonic_dist < th.orbit_harmonic_tol:
        flags.append("near_orbit_harmonic")

    return {
        "period": res.period,
        "t0": res.t0,
        "duration_hr": res.duration * 24,
        "depth": depth,
        "depth_err": res.depth_err,
        "depth_ppm": depth * 1e6,
        "snr": res.snr,
        "sde": res.sde,
        "beta": res.beta,
        "bls_power": res.max_power,
        "n_transits": res.n_transits,
        "rp_rs": rp_rs,
        "r_star": r_star if r_star else float("nan"),
        "m_star": m_star if m_star else float("nan"),
        "rp_earth": rp_earth,
        "depth_odd": d_odd,
        "depth_even": d_even,
        "odd_even_sigma": oe_sigma,
        "odd_even_frac": oe_frac,
        "secondary_depth": d_sec,
        "secondary_sigma": sec_sigma,
        "duration_max_hr": dur_max * 24 if np.isfinite(dur_max) else float("nan"),
        "duration_ratio": dur_ratio,
        "single_event_frac": single_frac,
        "harmonic_dll": s["harmonic_delta_log_likelihood"],
        "is_candidate": not fail,
        "fail_reasons": ";".join(fail),
        "flags": ";".join(flags),
    }


# ========================================================================= plotting
def outline_mask(ax, mask: np.ndarray, color: str = "red", lw: float = 1.5, x0: float = 0, y0: float = 0):
    """Draw the outer boundary of a boolean pixel mask (origin='lower' pixel coords)."""
    mask = np.asarray(mask, dtype=bool)
    ny, nx = mask.shape
    for r in range(ny):
        for c in range(nx):
            if not mask[r, c]:
                continue
            x, y = c + x0 - 0.5, r + y0 - 0.5
            if r == 0 or not mask[r - 1, c]:
                ax.plot([x, x + 1], [y, y], color=color, lw=lw)
            if r == ny - 1 or not mask[r + 1, c]:
                ax.plot([x, x + 1], [y + 1, y + 1], color=color, lw=lw)
            if c == 0 or not mask[r, c - 1]:
                ax.plot([x, x], [y, y + 1], color=color, lw=lw)
            if c == nx - 1 or not mask[r, c + 1]:
                ax.plot([x + 1, x + 1], [y, y + 1], color=color, lw=lw)


def median_image(tpf) -> np.ndarray:
    return np.nanmedian(np.asarray(tpf.flux.value, dtype=float), axis=0)


def plot_diagnostic(
    out: SearchOutput,
    metrics: dict,
    title: str,
    out_path: Path,
    tpf=None,
    aperture: np.ndarray | None = None,
) -> Path:
    res = out.result
    P, t0, dur = res.period, res.t0, res.duration
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    (ax1, ax2), (ax3, ax4) = axes

    # ---- Panel 1: pixel stamp + aperture (or raw vs flattened flux)
    if tpf is not None:
        img = median_image(tpf)
        norm = simple_norm(img[np.isfinite(img)], "sqrt", percent=99.5)
        im = ax1.imshow(img, origin="lower", cmap="viridis", norm=norm, interpolation="nearest")
        if aperture is not None:
            outline_mask(ax1, aperture)
        fig.colorbar(im, ax=ax1, label="median flux [e$^-$/s]", fraction=0.046)
        ny, nx = img.shape
        ax1.set_title(f"TPF median frame ({nx}x{ny} px), aperture = {int(np.sum(aperture)) if aperture is not None else 0} px")
        ax1.set_xlabel("column [px]")
        ax1.set_ylabel("row [px]")
    else:
        t = out.norm.time.value
        ax1.plot(t, out.norm.flux.value, ",", color="0.5", label="normalised raw")
        ax1.plot(out.trend.time.value, out.trend.flux.value, "r-", lw=1, label="trend")
        ax1.plot(out.flat.time.value, out.flat.flux.value - 3 * metrics["depth"] - 0.005, ",", color="C0",
                 label="flattened (offset)")
        ax1.legend(loc="lower left", fontsize=8, markerscale=20)
        ax1.set_title("Raw vs flattened flux")
        ax1.set_xlabel("Time [BTJD]")
        ax1.set_ylabel("Normalised flux")

    # ---- Panel 2: flattened light curve with predicted epochs
    t, f = out.flat.time.value, out.flat.flux.value
    ax2.plot(t, f, ".", ms=1, color="0.6", alpha=0.6, rasterized=True)
    bc, bm, _ = bin_series(t, f, 30 / 1440)
    ax2.plot(bc, bm, ".", ms=2.5, color="k", label="30-min bins")
    tt = res.stats["transit_times"]
    for ti in tt[(tt >= t.min() - dur) & (tt <= t.max() + dur)]:
        ax2.axvline(ti, color="red", alpha=0.35, lw=1)
    ax2.set_title(f"Flattened light curve — {res.n_transits} transits observed (red = t0 + nP)")
    ax2.set_xlabel("Time [BTJD]")
    ax2.set_ylabel("Normalised flux")
    lo = 1 - 2.5 * max(metrics["depth"], 5 * _robust_std(f))
    hi = 1 + 6 * _robust_std(f)
    ax2.set_ylim(lo, hi)
    ax2.legend(loc="lower right", fontsize=8)

    # ---- Panel 3: periodogram
    ax3.plot(res.periods, res.power, color="k", lw=0.6)
    ax3.axvline(P, color="red", alpha=0.5, lw=3, label=f"peak P = {P:.4f} d")
    for k, lab in ((0.5, "P/2"), (2.0, "2P")):
        if res.periods.min() <= k * P <= res.periods.max():
            ax3.axvline(k * P, color="red", ls=":", lw=1, alpha=0.6)
            ax3.text(k * P, res.power.max() * 0.95, lab, color="red", fontsize=8, ha="center")
    ax3.set_xscale("log")
    ax3.set_xlabel("Period [days]")
    ax3.set_ylabel("BLS power (Δ log-likelihood)")
    ax3.set_title(f"BLS periodogram — SDE = {res.sde:.1f}, SNR = {res.snr:.1f}")
    ax3.legend(loc="upper left", fontsize=8)

    # ---- Panel 4: phase fold, odd vs even
    ph = phase_days(t, P, t0)
    half = min(3 * dur, 0.5 * P)                       # never wrap past +/- P/2
    win = np.abs(ph) < half
    epoch = np.round((t - t0) / P).astype(int)
    odd = (epoch % 2) == 1
    hrs = ph * 24
    ax4.plot(hrs[win & ~odd], f[win & ~odd], ".", ms=1.5, color="C0", alpha=0.35, rasterized=True)
    ax4.plot(hrs[win & odd], f[win & odd], ".", ms=1.5, color="C1", alpha=0.35, rasterized=True)
    ax4.plot([], [], ".", ms=8, color="C0", label="even epochs")
    ax4.plot([], [], ".", ms=8, color="C1", label="odd epochs")
    bc, bm, be = bin_series(hrs[win], f[win], max(dur * 24 / 10, 10 / 60))
    ax4.errorbar(bc, bm, be, fmt="o", ms=3.5, color="k", label="binned", zorder=5)
    xm = np.linspace(-half * 24, half * 24, 2000)
    ax4.plot(xm, np.where(np.abs(xm) < dur * 12, 1 - res.depth, 1.0), "r-", lw=2, label="BLS box model", zorder=6)
    ax4.axhline(1 - metrics["depth_odd"], color="C1", ls="--", lw=1.2, label=f"odd depth {metrics['depth_odd']*1e6:.0f} ppm")
    ax4.axhline(1 - metrics["depth_even"], color="C0", ls="--", lw=1.2, label=f"even depth {metrics['depth_even']*1e6:.0f} ppm")
    ax4.set_xlim(-half * 24, half * 24)
    scat = _robust_std(f)
    ax4.set_ylim(1 - metrics["depth"] - 4 * scat, 1 + 4 * scat)
    ax4.set_xlabel("Hours from mid-transit")
    ax4.set_ylabel("Normalised flux")
    ax4.set_title(f"Phase-folded (duration {dur*24:.2f} h) — odd/even Δ = {metrics['odd_even_sigma']:.1f}σ")
    ax4.legend(loc="lower left", fontsize=7, ncol=2)

    verdict = "CANDIDATE" if metrics["is_candidate"] else f"REJECTED ({metrics['fail_reasons']})"
    rp = metrics["rp_earth"]
    rp_txt = f"{rp:.2f} R⊕" if np.isfinite(rp) else "n/a"
    sub = (f"P = {P:.5f} d | t0 = {t0:.4f} BTJD | depth = {metrics['depth_ppm']:.0f} ppm | "
           f"Rp/R* = {metrics['rp_rs']:.4f} | Rp = {rp_txt} | secondary {metrics['secondary_sigma']:.1f}σ")
    if metrics["flags"]:
        sub += f" | flags: {metrics['flags']}"
    fig.suptitle(f"{title} — {verdict}\n{sub}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ===================================================================== MAST helpers
def download_spoc_product(obs_id: str, kind: str = "lc", cache_dir: Path = CACHE_DIR, retries: int = 3) -> Path:
    """Fetch `<obs_id>_<kind>.fits` (kind = 'lc' or 'tp') straight from MAST."""
    from astroquery.mast import Observations

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{obs_id}_{kind}.fits"
    local = cache_dir / fname
    if local.exists() and local.stat().st_size > 0:
        return local
    last = ""
    for attempt in range(retries):
        try:
            status, msg, _ = Observations.download_file(
                f"mast:TESS/product/{fname}", local_path=str(local), cache=False, verbose=False
            )
            if status == "COMPLETE" and local.exists() and local.stat().st_size > 0:
                return local
            last = f"{status} {msg}"
        except Exception as exc:  # transient HTTP / connection errors
            last = f"{type(exc).__name__}: {exc}"
        local.unlink(missing_ok=True)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"MAST download failed for {fname} after {retries} tries: {last}")


def read_pdcsap(path: Path) -> lk.LightCurve:
    return lk.read(str(path), flux_column="pdcsap_flux", quality_bitmask="default")
