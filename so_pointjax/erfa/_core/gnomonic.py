"""Gnomonic (tangent-plane) projection functions, ported from ERFA C library."""

import jax.numpy as jnp

from so_pointjax.erfa._core.angles import anp


# ============================================================================
# Solve for xi, eta
# ============================================================================


def tpxes(a, b, a0, b0):
    """Tangent plane: celestial spherical to rectangular.

    Parameters
    ----------
    a, b : float
        Star's spherical coordinates (radians)
    a0, b0 : float
        Tangent point's spherical coordinates (radians)

    Returns
    -------
    xi, eta : float
        Rectangular coordinates of star image (radians at tangent point)
    j : int
        Status: 0=OK, 1=star too far, 2=antistar, 3=antistar too far
    """
    TINY = 1e-6

    sb0 = jnp.sin(b0)
    sb = jnp.sin(b)
    cb0 = jnp.cos(b0)
    cb = jnp.cos(b)
    da = a - a0
    sda = jnp.sin(da)
    cda = jnp.cos(da)

    # Reciprocal of star vector length to tangent plane.
    d = sb * sb0 + cb * cb0 * cda

    # Clamp d for edge cases using jnp.where for JIT compatibility.
    d_safe = jnp.where(d > TINY, d,
             jnp.where(d >= 0.0, TINY,
             jnp.where(d > -TINY, -TINY, d)))

    xi = cb * sda / d_safe
    eta = (sb * cb0 - cb * sb0 * cda) / d_safe

    # Status.
    j = jnp.where(d > TINY, 0,
        jnp.where(d >= 0.0, 1,
        jnp.where(d > -TINY, 2, 3)))

    return xi, eta, j


def tpxev(v, v0):
    """Tangent plane: direction cosines to rectangular.

    Parameters
    ----------
    v : ndarray, shape (3,)
        Direction cosines of star
    v0 : ndarray, shape (3,)
        Direction cosines of tangent point

    Returns
    -------
    xi, eta : float
        Tangent plane coordinates of star
    j : int
        Status: 0=OK, 1=star too far, 2=antistar, 3=antistar too far
    """
    TINY = 1e-6

    x, y, z = v[0], v[1], v[2]
    x0, y0, z0 = v0[0], v0[1], v0[2]

    # Deal with polar case.
    r2 = x0 * x0 + y0 * y0
    r = jnp.sqrt(r2)
    r_safe = jnp.where(r == 0.0, 1e-20, r)
    x0_safe = jnp.where(r == 0.0, 1e-20, x0)

    # Reciprocal of star vector length to tangent plane.
    w = x * x0_safe + y * y0
    d = w + z * z0

    # Clamp d for edge cases.
    d_safe = jnp.where(d > TINY, d,
             jnp.where(d >= 0.0, TINY,
             jnp.where(d > -TINY, -TINY, d)))

    d_r = d_safe * r_safe
    xi = (y * x0_safe - x * y0) / d_r
    r2_safe = jnp.where(r == 0.0, 1e-40, r2)
    eta = (z * r2_safe - z0 * w) / d_r

    j = jnp.where(d > TINY, 0,
        jnp.where(d >= 0.0, 1,
        jnp.where(d > -TINY, 2, 3)))

    return xi, eta, j


# ============================================================================
# Solve for star
# ============================================================================


def tpsts(xi, eta, a0, b0):
    """Tangent plane: rectangular to celestial spherical (star).

    Parameters
    ----------
    xi, eta : float
        Rectangular coordinates of star image
    a0, b0 : float
        Tangent point's spherical coordinates (radians)

    Returns
    -------
    a, b : float
        Star's spherical coordinates (radians)
    """
    sb0 = jnp.sin(b0)
    cb0 = jnp.cos(b0)
    d = cb0 - eta * sb0
    a_out = anp(jnp.arctan2(xi, d) + a0)
    b_out = jnp.arctan2(sb0 + eta * cb0, jnp.sqrt(xi * xi + d * d))
    return a_out, b_out


def tpstv(xi, eta, v0):
    """Tangent plane: rectangular to direction cosines (star).

    Parameters
    ----------
    xi, eta : float
        Rectangular coordinates of star image
    v0 : ndarray, shape (3,)
        Tangent point's direction cosines

    Returns
    -------
    v : ndarray, shape (3,)
        Star's direction cosines
    """
    x, y, z = v0[0], v0[1], v0[2]

    # Deal with polar case.
    r = jnp.sqrt(x * x + y * y)
    r_safe = jnp.where(r == 0.0, 1e-20, r)
    x_safe = jnp.where(r == 0.0, 1e-20, x)

    # Star vector length to tangent plane.
    f = jnp.sqrt(1.0 + xi * xi + eta * eta)

    vx = (x_safe - (xi * y + eta * x_safe * z) / r_safe) / f
    vy = (y + (xi * x_safe - eta * y * z) / r_safe) / f
    vz = (z + eta * r_safe) / f

    return jnp.array([vx, vy, vz])


# ============================================================================
# Solve for origin (tangent point)
# ============================================================================


def tpors(xi, eta, a, b):
    """Tangent plane: solve for origin (spherical).

    Parameters
    ----------
    xi, eta : float
        Rectangular coordinates of star image
    a, b : float
        Star's spherical coordinates (radians)

    Returns
    -------
    a01, b01 : float
        Tangent point solution 1 (radians)
    a02, b02 : float
        Tangent point solution 2 (radians)
    n : int
        Number of solutions (0, 1, or 2)
    """
    xi2 = xi * xi
    r = jnp.sqrt(1.0 + xi2 + eta * eta)
    sb = jnp.sin(b)
    cb = jnp.cos(b)
    rsb = r * sb
    rcb = r * cb
    w2 = rcb * rcb - xi2

    w = jnp.sqrt(jnp.maximum(w2, 0.0))

    # Solution 1.
    s = rsb - eta * w
    c = rsb * eta + w
    w1 = jnp.where((xi == 0.0) & (w == 0.0), 1.0, w)
    a01 = anp(a - jnp.arctan2(xi, w1))
    b01 = jnp.arctan2(s, c)

    # Solution 2 (w -> -w).
    s2 = rsb + eta * w
    c2 = rsb * eta - w
    w2_neg = jnp.where((xi == 0.0) & (w == 0.0), 1.0, -w)
    a02 = anp(a - jnp.arctan2(xi, w2_neg))
    b02 = jnp.arctan2(s2, c2)

    n = jnp.where(w2 >= 0.0,
                  jnp.where(jnp.abs(rsb) < 1.0, 1, 2),
                  0)

    return a01, b01, a02, b02, n


def tporv(xi, eta, v):
    """Tangent plane: solve for origin (direction cosines).

    Parameters
    ----------
    xi, eta : float
        Rectangular coordinates of star image
    v : ndarray, shape (3,)
        Star's direction cosines

    Returns
    -------
    v01 : ndarray, shape (3,)
        Tangent point solution 1
    v02 : ndarray, shape (3,)
        Tangent point solution 2
    n : int
        Number of solutions (0, 1, or 2)
    """
    x, y, z = v[0], v[1], v[2]
    rxy2 = x * x + y * y
    xi2 = xi * xi
    eta2p1 = eta * eta + 1.0
    r = jnp.sqrt(xi2 + eta2p1)
    rsb = r * z
    rcb = r * jnp.sqrt(rxy2)
    w2 = rcb * rcb - xi2

    w = jnp.sqrt(jnp.maximum(w2, 0.0))

    # Solution 1.
    denom = eta2p1 * jnp.sqrt(jnp.maximum(rxy2 * (w2 + xi2), 1e-300))
    c1 = (rsb * eta + w) / denom
    v01_x = c1 * (x * w + y * xi)
    v01_y = c1 * (y * w - x * xi)
    v01_z = (rsb - eta * w) / eta2p1

    # Solution 2 (w -> -w).
    c2 = (rsb * eta - w) / denom
    v02_x = c2 * (x * (-w) + y * xi)
    v02_y = c2 * (y * (-w) - x * xi)
    v02_z = (rsb + eta * w) / eta2p1

    v01 = jnp.array([v01_x, v01_y, v01_z])
    v02 = jnp.array([v02_x, v02_y, v02_z])

    n = jnp.where(w2 > 0.0,
                  jnp.where(jnp.abs(rsb) < 1.0, 1, 2),
                  0)

    return v01, v02, n


__all__ = ["tpxes", "tpxev", "tpsts", "tpstv", "tpors", "tporv"]
