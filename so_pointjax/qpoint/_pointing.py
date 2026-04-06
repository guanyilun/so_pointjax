"""Core pointing pipeline: forward and inverse transformations.

Forward:  Az/El → ... → RA/Dec
Inverse:  RA/Dec → ... → Az/El

Two API levels:
  1. High-level (convenience): takes ctime, handles UTC conversions internally.
     NOT vmap-compatible (UTC→TAI uses leap-second tables).
  2. Low-level (_jit suffix): takes precomputed JD values.
     Fully JIT/vmap/grad compatible.

For vectorized use, precompute times outside vmap, then vmap the _jit functions.
"""

import jax.numpy as jnp

from so_pointjax.qpoint._quaternion import (
    identity, mul, inv, r3, quat2radecpa, quat2radec, radecpa2quat,
)
from so_pointjax.qpoint._corrections import (
    npb_quat, erot_quat, wobble_quat, lonlat_quat,
    azelpsi_quat,
    refraction_quat,
    aberration, earth_orbital_beta, diurnal_aberration_beta,
    det_offset_quat,
)
from so_pointjax.qpoint._time_utils import ctime2jd, ctime2jdtt, jdutc2jdut1


# ---------------------------------------------------------------------------
# Time precomputation helper
# ---------------------------------------------------------------------------

def precompute_times(ctime, dut1=0.0):
    """Precompute Julian Dates from Unix time (non-JIT).

    Call this outside vmap, then pass results to the _jit pipeline functions.

    Parameters
    ----------
    ctime : float or array
        Unix time(s) in seconds.
    dut1 : float or array
        UT1-UTC in seconds.

    Returns
    -------
    dict with keys: jd_utc1, jd_utc2, tt1, tt2, ut1_1, ut1_2
    """
    import numpy as np
    ctime_arr = np.atleast_1d(np.asarray(ctime, dtype=np.float64))
    dut1_arr = np.broadcast_to(np.asarray(dut1, dtype=np.float64), ctime_arr.shape)

    jd_utc1 = np.full_like(ctime_arr, 2440587.5)
    jd_utc2 = ctime_arr / 86400.0

    tt1 = np.empty_like(ctime_arr)
    tt2 = np.empty_like(ctime_arr)
    ut1_1 = np.empty_like(ctime_arr)
    ut1_2 = np.empty_like(ctime_arr)

    for i in range(len(ctime_arr)):
        t1, t2 = ctime2jdtt(float(ctime_arr[i]))
        tt1[i] = float(t1)
        tt2[i] = float(t2)
        u1, u2 = jdutc2jdut1(float(jd_utc1[i]), float(jd_utc2[i]),
                              float(dut1_arr[i]))
        ut1_1[i] = float(u1)
        ut1_2[i] = float(u2)

    # Squeeze back to scalar if input was scalar
    if np.ndim(ctime) == 0:
        return {
            'jd_utc1': jnp.float64(jd_utc1[0]),
            'jd_utc2': jnp.float64(jd_utc2[0]),
            'tt1': jnp.float64(tt1[0]),
            'tt2': jnp.float64(tt2[0]),
            'ut1_1': jnp.float64(ut1_1[0]),
            'ut1_2': jnp.float64(ut1_2[0]),
        }
    return {
        'jd_utc1': jnp.array(jd_utc1),
        'jd_utc2': jnp.array(jd_utc2),
        'tt1': jnp.array(tt1),
        'tt2': jnp.array(tt2),
        'ut1_1': jnp.array(ut1_1),
        'ut1_2': jnp.array(ut1_2),
    }


# ---------------------------------------------------------------------------
# Low-level JIT-compatible forward pipeline
# ---------------------------------------------------------------------------

def azelpsi2bore_jit(az, el, psi, lon, lat,
                     tt1, tt2, ut1_1, ut1_2,
                     pitch=0.0, roll=0.0,
                     weather_A=0.0, weather_B=0.0,
                     accuracy=1, mean_aber=True, fast_aber=False,
                     xp=0.0, yp=0.0):
    """JIT-compatible forward pipeline using precomputed times.

    Parameters
    ----------
    az, el, psi : float
        Azimuth, elevation, boresight rotation (degrees).
    lon, lat : float
        Observer lon/lat (degrees).
    tt1, tt2 : float
        TT as 2-part Julian Date.
    ut1_1, ut1_2 : float
        UT1 as 2-part Julian Date.
    pitch, roll : float
        Boresight pitch/roll (degrees).
    weather_A, weather_B : float
        Refraction coefficients from so_pointjax.erfa.refco. Set both to 0 to skip.
    accuracy : int
        0 = full, 1 = low (static for JIT).
    mean_aber : bool
        Apply annual aberration at boresight (static for JIT).
    fast_aber : bool
        Small-angle aberration (static for JIT).
    xp, yp : float
        IERS pole coordinates (arcseconds).

    Returns
    -------
    q_bore : array shape (4,)
    """
    # Handle zenith crossing
    el_eff = jnp.where(el > 90.0, 180.0 - el, el)
    az_eff = jnp.where(el > 90.0, az + 180.0, az)
    psi_eff = jnp.where(el > 90.0, psi - 180.0, psi)

    # Az/el quaternion (without psi)
    q = azelpsi_quat(az_eff, el_eff, 0.0, pitch, roll)

    # Refraction (right-applied) — using precomputed coefficients
    apply_ref = (weather_A != 0.0) | (weather_B != 0.0)
    tz = jnp.tan(jnp.pi / 2.0 - jnp.deg2rad(el_eff))
    ref_angle = tz * (weather_A + weather_B * tz * tz)
    from so_pointjax.qpoint._quaternion import r2
    q_ref = r2(-ref_angle)
    q = jnp.where(apply_ref, mul(q, q_ref), q)

    # Boresight rotation psi (right-applied)
    q_psi = r3(-jnp.deg2rad(psi_eff))
    q = mul(q, q_psi)

    # Diurnal aberration (left-applied)
    beta_rot = diurnal_aberration_beta(lat)
    q_daber = aberration(q, beta_rot, inv=False, fast=fast_aber)
    q = mul(q_daber, q)

    # Lon/lat → ITRS (left-applied)
    q_lonlat = lonlat_quat(lon, lat)
    q = mul(q_lonlat, q)

    # Wobble (left-applied)
    q_wobble = wobble_quat(tt1, tt2, xp, yp)
    q = mul(q_wobble, q)

    # Earth rotation (left-applied)
    q_erot = erot_quat(ut1_1, ut1_2)
    q = mul(q_erot, q)

    # NPB (left-applied)
    q_npb = npb_quat(tt1, tt2, accuracy=accuracy)
    q = mul(q_npb, q)

    # Annual aberration (left-applied)
    if mean_aber:
        beta_earth = earth_orbital_beta(tt1, tt2)
        q_aaber = aberration(q, beta_earth, inv=False, fast=fast_aber)
        q = mul(q_aaber, q)

    return q


# ---------------------------------------------------------------------------
# Precomputed slow corrections for vectorized use
# ---------------------------------------------------------------------------

def precompute_corrections(ctime, dut1=0.0, accuracy=1, mean_aber=True,
                           xp=0.0, yp=0.0, rate_npb=10.0, rate_aber=10.0):
    """Precompute slow-varying corrections at a coarse cadence.

    NPB and annual aberration (epv00) are expensive ERFA calls that change
    very slowly (timescale of hours). This function evaluates them at a coarse
    cadence (default: every 10 seconds, matching C QPoint) and returns arrays
    that can be indexed inside vmap.

    Parameters
    ----------
    ctime : array
        Unix times for the full scan.
    dut1 : float or array
        UT1-UTC in seconds.
    accuracy : int
        0 = full, 1 = low.
    mean_aber : bool
        Whether annual aberration will be applied.
    xp, yp : float
        IERS pole coordinates (arcseconds).
    rate_npb : float
        Cadence for NPB recomputation (seconds). 0 = every sample.
    rate_aber : float
        Cadence for aberration recomputation (seconds). 0 = every sample.

    Returns
    -------
    dict with keys: times (from precompute_times), q_npb (N_coarse, 4),
          beta_earth (N_coarse, 3), npb_idx (N,), aber_idx (N,)
    """
    import numpy as np

    times = precompute_times(ctime, dut1)
    ctime_arr = np.atleast_1d(np.asarray(ctime, dtype=np.float64))
    n = len(ctime_arr)

    # --- NPB at coarse cadence ---
    if rate_npb > 0 and n > 1:
        t_start = ctime_arr[0]
        t_end = ctime_arr[-1]
        n_npb = max(1, int(np.ceil((t_end - t_start) / rate_npb)) + 1)
        npb_ctimes = np.linspace(t_start, t_end, n_npb)
        npb_times = precompute_times(npb_ctimes, dut1)
        # Evaluate NPB at coarse cadence
        q_npb_arr = jnp.stack([
            npb_quat(float(npb_times['tt1'][i]), float(npb_times['tt2'][i]),
                     accuracy=accuracy)
            for i in range(n_npb)
        ])
        # Also wobble at coarse cadence (cheap but might as well)
        q_wobble_arr = jnp.stack([
            wobble_quat(float(npb_times['tt1'][i]), float(npb_times['tt2'][i]),
                        xp, yp)
            for i in range(n_npb)
        ])
        # Index mapping: which coarse sample is nearest to each fine sample
        npb_idx = jnp.array(np.round(
            (ctime_arr - t_start) / max(t_end - t_start, 1e-10) * (n_npb - 1)
        ).astype(np.int32).clip(0, n_npb - 1))
    else:
        # Evaluate per-sample (no caching)
        q_npb_arr = None
        q_wobble_arr = None
        npb_idx = None

    # --- Annual aberration at coarse cadence ---
    if mean_aber and rate_aber > 0 and n > 1:
        t_start = ctime_arr[0]
        t_end = ctime_arr[-1]
        n_aber = max(1, int(np.ceil((t_end - t_start) / rate_aber)) + 1)
        aber_ctimes = np.linspace(t_start, t_end, n_aber)
        aber_times = precompute_times(aber_ctimes, dut1)
        beta_arr = jnp.stack([
            earth_orbital_beta(float(aber_times['tt1'][i]),
                              float(aber_times['tt2'][i]))
            for i in range(n_aber)
        ])
        aber_idx = jnp.array(np.round(
            (ctime_arr - t_start) / max(t_end - t_start, 1e-10) * (n_aber - 1)
        ).astype(np.int32).clip(0, n_aber - 1))
    else:
        beta_arr = None
        aber_idx = None

    return {
        **times,
        'q_npb': q_npb_arr,
        'q_wobble': q_wobble_arr,
        'npb_idx': npb_idx,
        'beta_earth': beta_arr,
        'aber_idx': aber_idx,
    }


def azelpsi2bore_fast(az, el, psi, lon, lat,
                      tt1, tt2, ut1_1, ut1_2,
                      q_npb, q_wobble, beta_earth,
                      pitch=0.0, roll=0.0,
                      weather_A=0.0, weather_B=0.0,
                      fast_aber=False):
    """Fast forward pipeline with precomputed NPB, wobble, and aberration.

    Unlike azelpsi2bore_jit, this version takes the expensive corrections
    as precomputed inputs (looked up from coarse-cadence arrays outside this
    function). This avoids recomputing epv00 and xys00b/xys06a per sample.

    Parameters
    ----------
    az, el, psi : float
        Azimuth, elevation, boresight rotation (degrees).
    lon, lat : float
        Observer lon/lat (degrees).
    tt1, tt2, ut1_1, ut1_2 : float
        Precomputed Julian dates.
    q_npb : array (4,)
        Precomputed NPB quaternion for this sample.
    q_wobble : array (4,)
        Precomputed wobble quaternion for this sample.
    beta_earth : array (3,)
        Precomputed Earth orbital velocity / c for this sample.
    pitch, roll : float
        Boresight pitch/roll (degrees).
    weather_A, weather_B : float
        Refraction coefficients.
    fast_aber : bool
        Small-angle aberration.

    Returns
    -------
    q_bore : array shape (4,)
    """
    # Handle zenith crossing
    el_eff = jnp.where(el > 90.0, 180.0 - el, el)
    az_eff = jnp.where(el > 90.0, az + 180.0, az)
    psi_eff = jnp.where(el > 90.0, psi - 180.0, psi)

    # Az/el quaternion
    q = azelpsi_quat(az_eff, el_eff, 0.0, pitch, roll)

    # Refraction (right-applied)
    apply_ref = (weather_A != 0.0) | (weather_B != 0.0)
    tz = jnp.tan(jnp.pi / 2.0 - jnp.deg2rad(el_eff))
    ref_angle = tz * (weather_A + weather_B * tz * tz)
    from so_pointjax.qpoint._quaternion import r2
    q_ref = r2(-ref_angle)
    q = jnp.where(apply_ref, mul(q, q_ref), q)

    # Boresight rotation psi (right-applied)
    q_psi = r3(-jnp.deg2rad(psi_eff))
    q = mul(q, q_psi)

    # Diurnal aberration (left-applied)
    beta_rot = diurnal_aberration_beta(lat)
    q_daber = aberration(q, beta_rot, inv=False, fast=fast_aber)
    q = mul(q_daber, q)

    # Lon/lat �� ITRS (left-applied)
    q_lonlat = lonlat_quat(lon, lat)
    q = mul(q_lonlat, q)

    # Wobble (left-applied) — precomputed
    q = mul(q_wobble, q)

    # Earth rotation (left-applied) — still per-sample (cheap, varies fast)
    q_erot = erot_quat(ut1_1, ut1_2)
    q = mul(q_erot, q)

    # NPB (left-applied) — precomputed
    q = mul(q_npb, q)

    # Annual aberration (left-applied) — precomputed beta
    q_aaber = aberration(q, beta_earth, inv=False, fast=fast_aber)
    q = mul(q_aaber, q)

    return q


# ---------------------------------------------------------------------------
# High-level convenience forward pipeline
# ---------------------------------------------------------------------------

def azelpsi2bore(az, el, psi, lon, lat, ctime,
                 pitch=0.0, roll=0.0, dut1=0.0,
                 weather=None, accuracy=1, mean_aber=True, fast_aber=False,
                 xp=0.0, yp=0.0):
    """Convert az/el/psi to boresight quaternion (forward pipeline).

    Convenience wrapper that handles UTC time conversions internally.
    NOT vmap-compatible — use azelpsi2bore_jit + precompute_times for vmap.

    All angles in degrees. ctime in Unix seconds.

    Returns
    -------
    q_bore : array shape (4,)
    """
    times = precompute_times(ctime, dut1)

    # Compute refraction coefficients if weather provided
    weather_A, weather_B = 0.0, 0.0
    if weather is not None:
        import so_pointjax.erfa
        wavelength = 299792458.0 * 1e-3 / weather.get('frequency', 150e9)
        weather_A, weather_B = so_pointjax.erfa.refco(
            weather.get('pressure', 0.0),
            weather.get('temperature', 0.0),
            weather.get('humidity', 0.0),
            wavelength,
        )

    return azelpsi2bore_jit(
        az, el, psi, lon, lat,
        times['tt1'], times['tt2'], times['ut1_1'], times['ut1_2'],
        pitch=pitch, roll=roll,
        weather_A=weather_A, weather_B=weather_B,
        accuracy=accuracy, mean_aber=mean_aber, fast_aber=fast_aber,
        xp=xp, yp=yp,
    )


def azel2bore(az, el, lon, lat, ctime, **kwargs):
    """Convert az/el to boresight quaternion (psi=0)."""
    return azelpsi2bore(az, el, 0.0, lon, lat, ctime, **kwargs)


# ---------------------------------------------------------------------------
# Boresight → detector → RA/Dec
# ---------------------------------------------------------------------------

def bore2radecpa(q_off, ctime, q_bore, mean_aber=True, fast_aber=False,
                 accuracy=1):
    """Convert boresight quaternion + detector offset → (ra, dec, pa).

    Parameters
    ----------
    q_off : array shape (4,)
        Detector offset quaternion.
    ctime : float
        Unix time (seconds). Only used if mean_aber=False.
    q_bore : array shape (4,)
        Boresight quaternion (from azel2bore or azelpsi2bore).
    mean_aber : bool
        If True, annual aberration was already applied in bore.
    fast_aber : bool
        Use small-angle approximation for aberration.
    accuracy : int
        0 = full, 1 = low.

    Returns
    -------
    ra, dec, pa : floats in degrees
    """
    q_det = mul(q_bore, q_off)

    if not mean_aber:
        tt1, tt2 = ctime2jdtt(ctime)
        beta_earth = earth_orbital_beta(tt1, tt2)
        q_aaber = aberration(q_det, beta_earth, inv=False, fast=fast_aber)
        q_det = mul(q_aaber, q_det)

    return quat2radecpa(q_det)


def bore2radec(q_off, ctime, q_bore, mean_aber=True, fast_aber=False,
               accuracy=1):
    """Convert boresight + offset → (ra, dec, sin2psi, cos2psi)."""
    q_det = mul(q_bore, q_off)

    if not mean_aber:
        tt1, tt2 = ctime2jdtt(ctime)
        beta_earth = earth_orbital_beta(tt1, tt2)
        q_aaber = aberration(q_det, beta_earth, inv=False, fast=fast_aber)
        q_det = mul(q_aaber, q_det)

    return quat2radec(q_det)


# ---------------------------------------------------------------------------
# Complete forward: az/el + offset → RA/Dec
# ---------------------------------------------------------------------------

def azel2radecpa(delta_az, delta_el, delta_psi,
                 az, el, lon, lat, ctime,
                 psi=0.0, pitch=0.0, roll=0.0,
                 dut1=0.0, weather=None, accuracy=1,
                 fast_aber=False, xp=0.0, yp=0.0):
    """Complete az/el + detector offset → (ra, dec, pa).

    Annual aberration is applied at the detector level.
    """
    q_off = det_offset_quat(delta_az, delta_el, delta_psi)

    q_bore = azelpsi2bore(
        az, el, psi, lon, lat, ctime,
        pitch=pitch, roll=roll, dut1=dut1,
        weather=weather, accuracy=accuracy,
        mean_aber=True, fast_aber=fast_aber,
        xp=xp, yp=yp,
    )

    q_det = mul(q_bore, q_off)
    return quat2radecpa(q_det)


# ---------------------------------------------------------------------------
# Inverse pipeline: RA/Dec → az/el
# ---------------------------------------------------------------------------

def radec2azel(ra, dec, pa, lon, lat, ctime,
               dut1=0.0, weather=None, accuracy=1,
               mean_aber=True, fast_aber=False,
               xp=0.0, yp=0.0):
    """Inverse pipeline: (ra, dec, pa) → (az, el, pa_out).

    Applies corrections in reverse order.
    NOT vmap-compatible — use radec2azel_jit + precompute_times for vmap.
    """
    times = precompute_times(ctime, dut1)

    weather_A, weather_B = 0.0, 0.0
    if weather is not None:
        import so_pointjax.erfa
        wavelength = 299792458.0 * 1e-3 / weather.get('frequency', 150e9)
        weather_A, weather_B = so_pointjax.erfa.refco(
            weather.get('pressure', 0.0),
            weather.get('temperature', 0.0),
            weather.get('humidity', 0.0),
            wavelength,
        )

    return radec2azel_jit(
        ra, dec, pa, lon, lat,
        times['tt1'], times['tt2'], times['ut1_1'], times['ut1_2'],
        weather_A=weather_A, weather_B=weather_B,
        accuracy=accuracy, mean_aber=mean_aber, fast_aber=fast_aber,
        xp=xp, yp=yp,
    )


def radec2azel_jit(ra, dec, pa, lon, lat,
                   tt1, tt2, ut1_1, ut1_2,
                   weather_A=0.0, weather_B=0.0,
                   accuracy=1, mean_aber=True, fast_aber=False,
                   xp=0.0, yp=0.0):
    """JIT-compatible inverse pipeline using precomputed times.

    Returns (az, el, pa_out) in degrees.
    """
    q = radecpa2quat(ra, dec, pa)

    # Annual aberration inverse
    if mean_aber:
        beta_earth = earth_orbital_beta(tt1, tt2)
        q_aaber = aberration(q, beta_earth, inv=True, fast=fast_aber)
        q = mul(q_aaber, q)

    # NPB inverse
    q_npb = npb_quat(tt1, tt2, accuracy=accuracy)
    q = mul(inv(q_npb), q)

    # Earth rotation inverse
    q_erot = erot_quat(ut1_1, ut1_2)
    q = mul(inv(q_erot), q)

    # Wobble inverse
    q_wobble = wobble_quat(tt1, tt2, xp, yp)
    q = mul(inv(q_wobble), q)

    # Lon/lat inverse
    q_lonlat = lonlat_quat(lon, lat)
    q = mul(inv(q_lonlat), q)

    # Refraction inverse (right-applied)
    apply_ref = (weather_A != 0.0) | (weather_B != 0.0)
    if True:  # always compute, use jnp.where to apply
        # Extract elevation from current quaternion state
        _, el_raw, _ = quat2radecpa(q)
        tz = jnp.tan(jnp.pi / 2.0 - jnp.deg2rad(el_raw))
        ref_angle = tz * (weather_A + weather_B * tz * tz)
        from so_pointjax.qpoint._quaternion import r2
        q_ref = r2(ref_angle)  # inverse = positive
        q = jnp.where(apply_ref, mul(q, q_ref), q)

    # Diurnal aberration inverse
    beta_rot = diurnal_aberration_beta(lat)
    q_daber = aberration(q, beta_rot, inv=True, fast=fast_aber)
    q = mul(q_daber, q)

    # Extract az/el/pa
    az_out, el_out, pa_out = quat2radecpa(q)
    az_out = -az_out  # QPoint convention

    return az_out, el_out, pa_out
