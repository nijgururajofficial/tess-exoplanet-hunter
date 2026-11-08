"""Offline injection-recovery tests for pipeline_core (no MAST access)."""

import numpy as np
import pytest
import lightkurve as lk

from src.pipeline_core import search, transit_mask, vet

STAR = {"radius": 1.0, "mass": 1.0}


def synthetic_lc(seed=0, noise=1e-3, var_amp=3e-3, var_period=5.0, transits=()):
    """27-day, 2-min-cadence sector with a 1-day mid-sector gap and slow variability.

    transits: iterable of (period, t0, duration, depth, depth_odd_or_None).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(1325.0, 1352.0, 2 / 1440)
    t = t[(t < 1338.0) | (t > 1339.0)]
    f = 1 + var_amp * np.sin(2 * np.pi * t / var_period) + rng.normal(0, noise, t.size)
    for period, t0, dur, depth, depth_odd in transits:
        m = transit_mask(t, period, dur, t0)
        if depth_odd is None:
            f[m] -= depth
        else:
            epoch = np.round((t - t0) / period).astype(int)
            f[m & (epoch % 2 == 0)] -= depth
            f[m & (epoch % 2 == 1)] -= depth_odd
    return lk.LightCurve(time=t, flux=f, flux_err=np.full_like(t, noise))


@pytest.fixture(scope="module")
def planet():
    lc = synthetic_lc(transits=[(3.2888, 1326.5, 0.143, 0.007, None)])
    out = search(lc)
    return out, vet(out.result, STAR)


def test_recovers_period(planet):
    out, _ = planet
    assert abs(out.result.period - 3.2888) < 0.005


def test_recovers_depth_within_10pct(planet):
    _, m = planet
    assert abs(m["depth"] - 0.007) / 0.007 < 0.10


def test_planet_passes_vetting(planet):
    _, m = planet
    assert m["is_candidate"], m["fail_reasons"]
    assert m["odd_even_frac"] < 0.10
    assert m["n_transits"] >= 6
    assert m["snr"] > 20 and m["sde"] > 8


def test_eclipsing_binary_fails_odd_even():
    # true binary period 4.2 d with unequal eclipses -> BLS locks onto 2.1 d
    lc = synthetic_lc(seed=1, transits=[(2.1, 1325.7, 0.12, 0.010, 0.005)])
    m = vet(search(lc).result, STAR)
    assert not m["is_candidate"]
    assert "odd_even_mismatch" in m["fail_reasons"]


def test_noise_is_not_a_candidate():
    lc = synthetic_lc(seed=2)
    m = vet(search(lc).result, STAR)
    assert not m["is_candidate"]


def test_stellar_variability_rejected():
    # 8.4-h sinusoid survives the 13.4-h flatten window; BLS finds a wide "box"
    lc = synthetic_lc(seed=4, var_amp=0.0, transits=())
    t = lc.time.value
    lc = lk.LightCurve(time=t, flux=lc.flux.value + 0.004 * np.sin(2 * np.pi * t / 0.35),
                       flux_err=lc.flux_err.value)
    m = vet(search(lc).result, STAR)
    assert not m["is_candidate"]
    assert "sinusoid_preferred" in m["fail_reasons"]


def test_short_usp_transit_not_rejected_as_too_long():
    # late-M dwarf, P = 0.55 d: true transit ~26 min, below the 58-min shortest BLS box
    star = {"radius": 0.15, "mass": 0.13}
    lc = synthetic_lc(seed=5, transits=[(0.55, 1325.2, 0.018, 0.006, None)])
    m = vet(search(lc).result, star)
    assert abs(m["period"] - 0.55) < 0.002
    assert "duration_too_long" not in m["fail_reasons"]
    assert m["is_candidate"], m["fail_reasons"]


def test_giant_companion_rejected_by_radius():
    # 9 % deep on a 1 R_sun star -> Rp ~ 33 R_earth -> stellar
    lc = synthetic_lc(seed=3, transits=[(4.0, 1326.0, 0.15, 0.09, None)])
    m = vet(search(lc).result, STAR)
    assert any(r.startswith("rp>") for r in m["fail_reasons"].split(";"))
