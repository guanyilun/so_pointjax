"""Geodetic/geocentric coordinate transformations, ported from ERFA C library."""

import jax.numpy as jnp

from so_pointjax.erfa._core.constants import WGS84, GRS80, WGS72, DPI


# ============================================================================
# Reference ellipsoid parameters
# ============================================================================

# WGS84 ellipsoid parameters
_ELLIPSOIDS = {
    WGS84: (6378137.0, 1.0 / 298.257223563),
    GRS80: (6378137.0, 1.0 / 298.257222101),
    WGS72: (6378135.0, 1.0 / 298.26),
}


def eform(n):
    """Earth reference ellipsoids.

    Parameters
    ----------
    n : int
        Ellipsoid identifier (1=WGS84, 2=GRS80, 3=WGS72)

    Returns
    -------
    a : float
        Equatorial radius (m)
    f : float
        Flattening

    Raises
    ------
    ValueError
        If n is not a valid ellipsoid identifier.
    """
    if n not in _ELLIPSOIDS:
        raise ValueError(f"Invalid ellipsoid identifier: {n}")
    return _ELLIPSOIDS[n]


def gd2gce(a, f, elong, phi, height):
    """Geodetic to geocentric transformation for a general ellipsoid.

    Parameters
    ----------
    a : float
        Equatorial radius (m)
    f : float
        Flattening
    elong : float
        Longitude (radians, east +ve)
    phi : float
        Geodetic latitude (radians)
    height : float
        Height above ellipsoid (m, geodetic)

    Returns
    -------
    xyz : ndarray, shape (3,)
        Geocentric vector (m)
    """
    sp = jnp.sin(phi)
    cp = jnp.cos(phi)
    w = 1.0 - f
    w = w * w
    d = cp * cp + w * sp * sp
    ac = a / jnp.sqrt(d)
    a_s = w * ac

    r = (ac + height) * cp
    x = r * jnp.cos(elong)
    y = r * jnp.sin(elong)
    z = (a_s + height) * sp
    return jnp.array([x, y, z])


def gd2gc(n, elong, phi, height):
    """Geodetic to geocentric transformation using a standard ellipsoid.

    Parameters
    ----------
    n : int
        Ellipsoid identifier (1=WGS84, 2=GRS80, 3=WGS72)
    elong : float
        Longitude (radians, east +ve)
    phi : float
        Geodetic latitude (radians)
    height : float
        Height above ellipsoid (m, geodetic)

    Returns
    -------
    xyz : ndarray, shape (3,)
        Geocentric vector (m)
    """
    a, f = eform(n)
    return gd2gce(a, f, elong, phi, height)


def gc2gde(a, f, xyz):
    """Geocentric to geodetic transformation for a general ellipsoid.

    Parameters
    ----------
    a : float
        Equatorial radius (m)
    f : float
        Flattening
    xyz : ndarray, shape (3,)
        Geocentric vector (m)

    Returns
    -------
    elong : float
        Longitude (radians, east +ve)
    phi : float
        Geodetic latitude (radians)
    height : float
        Height above ellipsoid (m, geodetic)
    """
    # Functions of ellipsoid parameters.
    aeps2 = a * a * 1e-32
    e2 = (2.0 - f) * f
    e4t = e2 * e2 * 1.5
    ec2 = 1.0 - e2
    ec = jnp.sqrt(ec2)
    b_ax = a * ec

    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    # Distance from polar axis squared.
    p2 = x * x + y * y

    # Longitude.
    elong = jnp.where(p2 > 0.0, jnp.arctan2(y, x), 0.0)

    # Unsigned z-coordinate.
    absz = jnp.abs(z)

    # Distance from polar axis.
    p = jnp.sqrt(p2)

    # --- Non-polar case ---
    # Normalization.
    s0 = absz / a
    pn = p / a
    zc = ec * s0

    # Prepare Newton correction factors.
    c0 = ec * pn
    c02 = c0 * c0
    c03 = c02 * c0
    s02 = s0 * s0
    s03 = s02 * s0
    a02 = c02 + s02
    a0 = jnp.sqrt(a02)
    a03 = a02 * a0
    d0 = zc * a03 + e2 * s03
    f0 = pn * a03 - e2 * c03

    # Prepare Halley correction factor.
    b0 = e4t * s02 * c02 * pn * (a0 - ec)
    s1 = d0 * f0 - b0 * s0
    cc = ec * (f0 * f0 - b0 * c0)

    # Evaluate latitude and height (non-polar).
    phi_np = jnp.arctan2(s1, cc)
    s12 = s1 * s1
    cc2 = cc * cc
    height_np = (p * cc + absz * s1 - a * jnp.sqrt(ec2 * s12 + cc2)) / jnp.sqrt(s12 + cc2)

    # --- Polar case ---
    phi_polar = DPI / 2.0
    height_polar = absz - b_ax

    # Select based on whether we're near the pole.
    is_polar = p2 <= aeps2
    phi = jnp.where(is_polar, phi_polar, phi_np)
    height = jnp.where(is_polar, height_polar, height_np)

    # Restore sign of latitude.
    phi = jnp.where(z < 0, -phi, phi)

    return elong, phi, height


def gc2gd(n, xyz):
    """Geocentric to geodetic transformation using a standard ellipsoid.

    Parameters
    ----------
    n : int
        Ellipsoid identifier (1=WGS84, 2=GRS80, 3=WGS72)
    xyz : ndarray, shape (3,)
        Geocentric vector (m)

    Returns
    -------
    elong : float
        Longitude (radians, east +ve)
    phi : float
        Geodetic latitude (radians)
    height : float
        Height above ellipsoid (m, geodetic)
    """
    a, f = eform(n)
    return gc2gde(a, f, xyz)


__all__ = ["eform", "gd2gce", "gd2gc", "gc2gde", "gc2gd"]
