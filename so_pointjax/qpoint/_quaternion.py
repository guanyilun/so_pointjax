"""Quaternion algebra for telescope pointing.

Quaternions are represented as jnp.ndarray of shape (4,) with layout [w, x, y, z].
All functions are pure (no mutation) and compatible with jax.jit, jax.grad, jax.vmap.
"""

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def identity():
    """Return the identity quaternion [1, 0, 0, 0]."""
    return jnp.array([1.0, 0.0, 0.0, 0.0])


def mul(a, b):
    """Quaternion multiplication: a * b."""
    return jnp.array([
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0],
    ])


def conj(q):
    """Quaternion conjugate: q*."""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def norm2(q):
    """Squared norm |q|^2."""
    return jnp.dot(q, q)


def norm(q):
    """Euclidean norm |q|."""
    return jnp.sqrt(norm2(q))


def normalize(q):
    """Normalize to unit quaternion."""
    return q / norm(q)


def inv(q):
    """Quaternion inverse: q* / |q|^2."""
    n2 = norm2(q)
    return jnp.array([q[0] / n2, -q[1] / n2, -q[2] / n2, -q[3] / n2])


# ---------------------------------------------------------------------------
# Rotation generators: R_i(angle) as quaternions
# ---------------------------------------------------------------------------

def r1(angle):
    """Rotation quaternion about X-axis by angle (radians)."""
    a2 = 0.5 * angle
    return jnp.array([jnp.cos(a2), jnp.sin(a2), 0.0, 0.0])


def r2(angle):
    """Rotation quaternion about Y-axis by angle (radians)."""
    a2 = 0.5 * angle
    return jnp.array([jnp.cos(a2), 0.0, jnp.sin(a2), 0.0])


def r3(angle):
    """Rotation quaternion about Z-axis by angle (radians)."""
    a2 = 0.5 * angle
    return jnp.array([jnp.cos(a2), 0.0, 0.0, jnp.sin(a2)])


def r1_mul(angle, q):
    """Left-multiply by R1: R1(angle) * q (optimized)."""
    a2 = 0.5 * angle
    c, s = jnp.cos(a2), jnp.sin(a2)
    return jnp.array([
        c*q[0] - s*q[1],
        c*q[1] + s*q[0],
        c*q[2] - s*q[3],
        c*q[3] + s*q[2],
    ])


def r2_mul(angle, q):
    """Left-multiply by R2: R2(angle) * q (optimized)."""
    a2 = 0.5 * angle
    c, s = jnp.cos(a2), jnp.sin(a2)
    return jnp.array([
        c*q[0] - s*q[2],
        c*q[1] + s*q[3],
        c*q[2] + s*q[0],
        c*q[3] - s*q[1],
    ])


def r3_mul(angle, q):
    """Left-multiply by R3: R3(angle) * q (optimized)."""
    a2 = 0.5 * angle
    c, s = jnp.cos(a2), jnp.sin(a2)
    return jnp.array([
        c*q[0] - s*q[3],
        c*q[1] - s*q[2],
        c*q[2] + s*q[1],
        c*q[3] + s*q[0],
    ])


def rot(angle, axis):
    """Rotation quaternion by angle around arbitrary axis vector.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.
    axis : array shape (3,)
        Rotation axis (will be normalized).

    Returns
    -------
    q : array shape (4,)
    """
    a2 = 0.5 * angle
    n = jnp.sqrt(jnp.dot(axis, axis))
    s = jnp.sin(a2) / n
    return jnp.array([jnp.cos(a2), s * axis[0], s * axis[1], s * axis[2]])


# ---------------------------------------------------------------------------
# Conversion: quaternion <-> rotation matrix
# ---------------------------------------------------------------------------

def to_matrix(q):
    """Convert unit quaternion to 3x3 rotation matrix."""
    u = normalize(q)
    w, x, y, z = u[0], u[1], u[2], u[3]
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    return jnp.array([
        [w2 + x2 - y2 - z2, 2.0*(x*y - w*z),     2.0*(x*z + w*y)],
        [2.0*(x*y + w*z),   w2 - x2 + y2 - z2,   2.0*(y*z - w*x)],
        [2.0*(x*z - w*y),   2.0*(y*z + w*x),      w2 - x2 - y2 + z2],
    ])


def to_col1(q):
    """First column of the rotation matrix (no normalization)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    return jnp.array([
        w2 + x2 - y2 - z2,
        2.0*(x*y + w*z),
        2.0*(x*z - w*y),
    ])


def to_col2(q):
    """Second column of the rotation matrix (no normalization)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    return jnp.array([
        2.0*(x*y - w*z),
        w2 - x2 + y2 - z2,
        2.0*(y*z + w*x),
    ])


def to_col3(q):
    """Third column of the rotation matrix (no normalization)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    w2, x2, y2, z2 = w*w, x*x, y*y, z*z
    return jnp.array([
        2.0*(x*z + w*y),
        2.0*(y*z - w*x),
        w2 - x2 - y2 + z2,
    ])


# ---------------------------------------------------------------------------
# Conversion: quaternion <-> (ra, dec, pa) in degrees
# ---------------------------------------------------------------------------

def quat2radecpa(q):
    """Extract (ra, dec, pa) in degrees from a pointing quaternion.

    Uses ZYZ Euler angle decomposition matching the C QPoint convention.

    Parameters
    ----------
    q : array shape (4,)

    Returns
    -------
    ra, dec, pa : floats in degrees
    """
    q00p33 = q[0]*q[0] + q[3]*q[3]
    q11p22 = q[1]*q[1] + q[2]*q[2]
    cosb2 = q00p33 * q11p22
    sinb_2 = 0.5 * (q00p33 - q11p22)

    q01 = q[0]*q[1]
    q02 = q[0]*q[2]
    q13 = q[1]*q[3]
    q23 = q[2]*q[3]

    sina_2 = q23 - q01
    cosa_2 = q02 + q13
    cosb_2 = jnp.sqrt(cosb2)

    ra = jnp.rad2deg(jnp.arctan2(sina_2, cosa_2))
    dec = jnp.rad2deg(jnp.arctan2(sinb_2, cosb_2))

    sing = q01 + q23
    cosg = q13 - q02
    pa = jnp.rad2deg(jnp.arctan2(sing, cosg))

    return ra, dec, pa


def radecpa2quat(ra, dec, pa):
    """Construct pointing quaternion from (ra, dec, pa) in degrees.

    Parameters
    ----------
    ra, dec, pa : floats in degrees

    Returns
    -------
    q : array shape (4,)
    """
    ra_rad = jnp.deg2rad(ra)
    dec_rad = jnp.deg2rad(dec)
    pa_rad = jnp.deg2rad(pa)

    q = r3(jnp.pi - pa_rad)
    q = r2_mul(jnp.pi / 2.0 - dec_rad, q)
    q = r3_mul(ra_rad, q)
    return q


# ---------------------------------------------------------------------------
# Conversion: quaternion <-> (ra, dec, sin2psi, cos2psi)
# ---------------------------------------------------------------------------

def quat2radec(q):
    """Extract (ra, dec, sin2psi, cos2psi) from a pointing quaternion.

    Parameters
    ----------
    q : array shape (4,)

    Returns
    -------
    ra, dec : floats in degrees
    sin2psi, cos2psi : floats (polarization angle)
    """
    q00p33 = q[0]*q[0] + q[3]*q[3]
    q11p22 = q[1]*q[1] + q[2]*q[2]
    cosb2 = q00p33 * q11p22
    sinb_2 = 0.5 * (q00p33 - q11p22)

    q01 = q[0]*q[1]
    q02 = q[0]*q[2]
    q13 = q[1]*q[3]
    q23 = q[2]*q[3]

    sina_2 = q23 - q01
    cosa_2 = q02 + q13
    cosb_2 = jnp.sqrt(cosb2)

    ra = jnp.rad2deg(jnp.arctan2(sina_2, cosa_2))
    dec = jnp.rad2deg(jnp.arctan2(sinb_2, cosb_2))

    sing = q01 + q23
    cosg = q13 - q02
    norm_val = 2.0 * cosg / cosb2

    sin2psi = norm_val * sing
    cos2psi = norm_val * cosg - 1.0

    return ra, dec, sin2psi, cos2psi


def radec2quat(ra, dec, sin2psi, cos2psi):
    """Construct pointing quaternion from (ra, dec, sin2psi, cos2psi).

    Parameters
    ----------
    ra, dec : floats in degrees
    sin2psi, cos2psi : floats (polarization angle)

    Returns
    -------
    q : array shape (4,)
    """
    ang = jnp.arctan2(sin2psi, cos2psi + 1.0)
    q = r3(jnp.pi - ang)
    q = r2_mul(jnp.pi / 2.0 - jnp.deg2rad(dec), q)
    q = r3_mul(jnp.deg2rad(ra), q)
    return q


# ---------------------------------------------------------------------------
# SLERP interpolation
# ---------------------------------------------------------------------------

def slerp(q0, q1, t):
    """Spherical linear interpolation between quaternions q0 and q1.

    Parameters
    ----------
    q0, q1 : array shape (4,)
        Unit quaternions.
    t : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    q : array shape (4,)
        Interpolated unit quaternion.
    """
    cos_alpha = jnp.dot(q0, q1)

    # Ensure shortest path: flip q1 if dot product is negative
    q1_adj = jnp.where(cos_alpha < 0.0, -q1, q1)
    cos_alpha = jnp.abs(cos_alpha)

    alpha = jnp.arccos(jnp.clip(cos_alpha, -1.0, 1.0))
    sin_alpha = jnp.sin(alpha)

    # When quaternions are very close, fall back to linear interpolation
    safe = sin_alpha > 1e-10
    s0 = jnp.where(safe, jnp.sin((1.0 - t) * alpha) / sin_alpha, 1.0 - t)
    s1 = jnp.where(safe, jnp.sin(t * alpha) / sin_alpha, t)

    return s0 * q0 + s1 * q1_adj
