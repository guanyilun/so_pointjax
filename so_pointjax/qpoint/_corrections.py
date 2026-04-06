"""Individual correction functions for the pointing pipeline.

Each correction produces a quaternion rotation. All functions are pure
and compatible with jax.jit / jax.grad / jax.vmap.
"""

import jax.numpy as jnp
import so_pointjax.erfa

from so_pointjax.qpoint._quaternion import (
    r1, r2, r3, r1_mul, r2_mul, r3_mul,
    rot, mul, to_col3,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Diurnal aberration constant (radians) — ~0.3191 arcsec
D_ABER_RAD = 1.54716541e-06

# Speed of light in AU/day
C_AUD = 173.14463269999999

# Speed of light in m/s
C_MS = 299792458.0


# ---------------------------------------------------------------------------
# Nutation / Precession / Frame bias (NPB)
# ---------------------------------------------------------------------------

def npb_quat(jd_tt1, jd_tt2, accuracy=0):
    """Compute the NPB correction quaternion.

    Parameters
    ----------
    jd_tt1, jd_tt2 : float
        TT as 2-part Julian Date.
    accuracy : int
        0 = full (xys06a), 1 = low (xys00b).

    Returns
    -------
    q : array shape (4,)
    """
    if accuracy == 0:
        X, Y, s = so_pointjax.erfa.xys06a(jd_tt1, jd_tt2)
    else:
        X, Y, s = so_pointjax.erfa.xys00b(jd_tt1, jd_tt2)

    Z = jnp.sqrt(1.0 - X * X - Y * Y)
    E = jnp.arctan2(Y, X)
    d = jnp.arccos(Z)

    q = r3(-E - s)
    q = r2_mul(d, q)
    q = r3_mul(E, q)
    return q


# ---------------------------------------------------------------------------
# Earth rotation
# ---------------------------------------------------------------------------

def erot_quat(jd_ut1_1, jd_ut1_2):
    """Compute the Earth rotation quaternion.

    Parameters
    ----------
    jd_ut1_1, jd_ut1_2 : float
        UT1 as 2-part Julian Date.

    Returns
    -------
    q : array shape (4,)
    """
    theta = so_pointjax.erfa.era00(jd_ut1_1, jd_ut1_2)
    return r3(theta)


# ---------------------------------------------------------------------------
# Polar motion (wobble)
# ---------------------------------------------------------------------------

def wobble_quat(jd_tt1, jd_tt2, xp, yp):
    """Compute the polar motion (wobble) correction quaternion.

    Parameters
    ----------
    jd_tt1, jd_tt2 : float
        TT as 2-part Julian Date.
    xp, yp : float
        IERS pole coordinates in arcseconds.

    Returns
    -------
    q : array shape (4,)
    """
    sprime = so_pointjax.erfa.sp00(jd_tt1, jd_tt2)
    DAS2R = jnp.pi / (180.0 * 3600.0)
    q = r1(-yp * DAS2R)
    q = r2_mul(-xp * DAS2R, q)
    q = r3_mul(sprime, q)
    return q


# ---------------------------------------------------------------------------
# Longitude / Latitude (observer to ITRS)
# ---------------------------------------------------------------------------

def lonlat_quat(lon, lat):
    """Compute the observer lon/lat transformation quaternion.

    Parameters
    ----------
    lon, lat : float
        Observer longitude and latitude in degrees.

    Returns
    -------
    q : array shape (4,)
    """
    q = r3(jnp.pi)
    q = r2_mul(jnp.pi / 2.0 - jnp.deg2rad(lat), q)
    q = r3_mul(jnp.deg2rad(lon), q)
    return q


# ---------------------------------------------------------------------------
# Azimuth / Elevation quaternion
# ---------------------------------------------------------------------------

def azel_quat(az, el, pitch=0.0, roll=0.0):
    """Construct quaternion from az/el (degrees), with optional pitch/roll.

    Parameters
    ----------
    az, el : float
        Azimuth and elevation in degrees.
    pitch, roll : float
        Boresight pitch and roll in degrees.

    Returns
    -------
    q : array shape (4,)
    """
    q = r3(jnp.pi)
    q = r2_mul(jnp.pi / 2.0 - jnp.deg2rad(el), q)
    q = r3_mul(-jnp.deg2rad(az), q)
    q = r2_mul(-jnp.deg2rad(pitch), q)
    q = r1_mul(-jnp.deg2rad(roll), q)
    return q


def azelpsi_quat(az, el, psi, pitch=0.0, roll=0.0):
    """Construct quaternion from az/el/psi (degrees), with optional pitch/roll.

    Parameters
    ----------
    az, el, psi : float
        Azimuth, elevation, and boresight rotation in degrees.
    pitch, roll : float
        Boresight pitch and roll in degrees.

    Returns
    -------
    q : array shape (4,)
    """
    q = r3(jnp.pi - jnp.deg2rad(psi))
    q = r2_mul(jnp.pi / 2.0 - jnp.deg2rad(el), q)
    q = r3_mul(-jnp.deg2rad(az), q)
    q = r2_mul(-jnp.deg2rad(pitch), q)
    q = r1_mul(-jnp.deg2rad(roll), q)
    return q


# ---------------------------------------------------------------------------
# Atmospheric refraction
# ---------------------------------------------------------------------------

def refraction(el, temperature=0.0, pressure=0.0, humidity=0.0, frequency=150e9):
    """Compute the atmospheric refraction correction angle.

    Parameters
    ----------
    el : float
        Elevation in degrees.
    temperature : float
        Temperature in Celsius.
    pressure : float
        Pressure in hPa (millibar).
    humidity : float
        Relative humidity (0-1).
    frequency : float
        Observation frequency in Hz.

    Returns
    -------
    ref_deg : float
        Refraction correction in degrees.
    """
    wavelength = C_MS * 1e-3 / frequency  # convert to micrometers
    A, B = so_pointjax.erfa.refco(pressure, temperature, humidity, wavelength)

    # Handle elevations above 90
    el_eff = jnp.where(el > 90.0, 180.0 - el, el)
    tz = jnp.tan(jnp.pi / 2.0 - jnp.deg2rad(el_eff))
    ref = tz * (A + B * tz * tz)
    return jnp.rad2deg(ref)


def refraction_quat(el, temperature=0.0, pressure=0.0, humidity=0.0,
                    frequency=150e9, inv=False):
    """Compute the refraction correction quaternion.

    Returns a rotation about Y-axis by the refraction angle.
    """
    delta = refraction(el, temperature, pressure, humidity, frequency)
    sign = jnp.where(inv, 1.0, -1.0)
    return r2(sign * jnp.deg2rad(delta))


# ---------------------------------------------------------------------------
# Aberration (annual and diurnal)
# ---------------------------------------------------------------------------

def aberration(q, beta, inv=False, fast=False):
    """Compute the aberration correction quaternion.

    Parameters
    ----------
    q : array shape (4,)
        Current pointing quaternion.
    beta : array shape (3,)
        Velocity vector as fraction of speed of light.
    inv : bool
        If True, compute inverse correction.
    fast : bool
        If True, use small-angle approximation.

    Returns
    -------
    qa : array shape (4,)
        Aberration correction quaternion (left-multiply onto q).
    """
    u = to_col3(q)

    # Cross product: n = u x beta (forward) or n = beta x u (inverse)
    n = jnp.where(inv, jnp.cross(beta, u), jnp.cross(u, beta))
    n_norm = jnp.sqrt(jnp.dot(n, n))

    if fast:
        # Small-angle approximation
        sa_2 = 0.5 * n_norm
        return jnp.array([
            1.0 - 0.5 * sa_2 * sa_2,
            -0.5 * n[0],
            -0.5 * n[1],
            -0.5 * n[2],
        ])
    else:
        ang = jnp.arcsin(n_norm)
        return rot(-ang, n)


def earth_orbital_beta(jd_tdb1, jd_tdb2):
    """Compute Earth's orbital velocity as a fraction of c.

    Parameters
    ----------
    jd_tdb1, jd_tdb2 : float
        TDB as 2-part Julian Date.

    Returns
    -------
    beta : array shape (3,)
        Velocity / c in AU/day units.
    """
    pvh, pvb = so_pointjax.erfa.epv00(jd_tdb1, jd_tdb2)
    # pvb[1] is barycentric velocity of Earth in AU/day
    return pvb[1] / C_AUD


def diurnal_aberration_beta(lat):
    """Compute diurnal aberration velocity vector.

    Parameters
    ----------
    lat : float
        Observer latitude in degrees.

    Returns
    -------
    beta : array shape (3,)
    """
    clat = jnp.cos(jnp.deg2rad(lat))
    return jnp.array([0.0, -clat * D_ABER_RAD, 0.0])


# ---------------------------------------------------------------------------
# Detector offset
# ---------------------------------------------------------------------------

def det_offset_quat(delta_az, delta_el, delta_psi):
    """Compute the detector offset quaternion.

    Parameters
    ----------
    delta_az, delta_el, delta_psi : float
        Detector offsets in degrees.

    Returns
    -------
    q : array shape (4,)
    """
    q = r3(-jnp.deg2rad(delta_psi))
    q = r2_mul(jnp.deg2rad(delta_el), q)
    q = r1_mul(-jnp.deg2rad(delta_az), q)
    return q


# ---------------------------------------------------------------------------
# Half-wave plate (HWP)
# ---------------------------------------------------------------------------

def hwp_quat(ang):
    """Compute the HWP rotation quaternion.

    Physical angle ang → polarization rotation by -2*ang.

    Parameters
    ----------
    ang : float
        HWP angle in degrees.

    Returns
    -------
    q : array shape (4,)
    """
    return r3(-2.0 * jnp.deg2rad(ang))
