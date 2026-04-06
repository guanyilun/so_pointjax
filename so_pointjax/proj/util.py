"""Scan pattern utilities, mirroring so3g.proj.util."""

import jax.numpy as jnp


def ces(el, az0, throw, v_scan, t):
    """Generate a constant-elevation scan (CES) pattern.

    Parameters
    ----------
    el : float
        Elevation angle.
    az0 : float
        Central azimuth angle.
    throw : float
        Azimuth half-scan amplitude.
    v_scan : float
        Scan speed (same units as angles/time).
    t : array
        Timestamps.

    Returns
    -------
    az, el : arrays
        Azimuth and elevation at each timestamp.

    Notes
    -----
    Units must be consistent (e.g. degrees, seconds, deg/s).
    """
    t = jnp.asarray(t, dtype=jnp.float64)
    phase = (t - t[0]) * v_scan % (4 * throw)
    az = jnp.where(phase > 2 * throw, 4 * throw - phase, phase)
    return az - throw + az0, el + jnp.zeros_like(t)
