"""VectorMatrix functions: vector ops, matrix ops, rotations, spherical/cartesian.

Ported from ERFA C library. All functions are pure and JIT-compatible.
"""

import jax.numpy as jnp
from so_pointjax.erfa._core.constants import D2PI


# ---------------------------------------------------------------------------
# VectorMatrix/VectorOps
# ---------------------------------------------------------------------------

def pdp(a, b):
    """Scalar product of two p-vectors."""
    return jnp.dot(a, b)


def pm(p):
    """Modulus of p-vector."""
    return jnp.sqrt(jnp.dot(p, p))


def pmp(a, b):
    """P-vector minus p-vector."""
    return a - b


def pn(p):
    """Decompose p-vector into modulus and direction.

    Returns (r, u) where r is the modulus and u is the unit vector.
    """
    r = pm(p)
    u = jnp.where(r != 0.0, p / r, jnp.zeros_like(p))
    return r, u


def ppp(a, b):
    """P-vector addition."""
    return a + b


def ppsp(a, s, b):
    """P-vector plus scaled p-vector: a + s*b."""
    return a + s * b


def pvdpv(a, b):
    """Inner product of two pv-vectors.

    Returns array of shape (2,): [p.p, p.v+v.p].
    """
    return jnp.array([
        jnp.dot(a[0], b[0]),
        jnp.dot(a[0], b[1]) + jnp.dot(a[1], b[0]),
    ])


def pvm(pv):
    """Modulus of pv-vector.

    Returns (r, s) where r is position modulus and s is velocity modulus.
    """
    return pm(pv[0]), pm(pv[1])


def pvmpv(a, b):
    """Subtract pv-vector b from pv-vector a."""
    return a - b


def pvppv(a, b):
    """Add two pv-vectors."""
    return a + b


def pvu(dt, pv):
    """Update a pv-vector: propagate position by dt.

    Returns pv-vector with position updated by p + dt*v,
    velocity unchanged.
    """
    return jnp.array([pv[0] + dt * pv[1], pv[1]])


def pvup(dt, pv):
    """Update a pv-vector, returning position only: p + dt*v."""
    return pv[0] + dt * pv[1]


def pvxpv(a, b):
    """Outer (cross) product of two pv-vectors.

    Returns pv-vector: position = a_p x b_p,
    velocity = a_p x b_v + a_v x b_p.
    """
    return jnp.array([
        pxp(a[0], b[0]),
        pxp(a[0], b[1]) + pxp(a[1], b[0]),
    ])


def pxp(a, b):
    """Cross product of two p-vectors."""
    return jnp.cross(a, b)


def s2xpv(s1, s2, pv):
    """Multiply a pv-vector by two scalars: s1*position, s2*velocity."""
    return jnp.array([s1 * pv[0], s2 * pv[1]])


def sxp(s, p):
    """Multiply a p-vector by a scalar."""
    return s * p


def sxpv(s, pv):
    """Multiply a pv-vector by a scalar."""
    return s * pv


# ---------------------------------------------------------------------------
# VectorMatrix/CopyExtendExtract
# ---------------------------------------------------------------------------

def cp(p):
    """Copy a p-vector (identity, for API compatibility)."""
    return jnp.array(p)


def cpv(pv):
    """Copy a pv-vector."""
    return jnp.array(pv)


def cr(r):
    """Copy an r-matrix."""
    return jnp.array(r)


def p2pv(p):
    """Extend a p-vector to a pv-vector by appending zero velocity."""
    return jnp.array([p, jnp.zeros(3)])


def pv2p(pv):
    """Discard velocity component of a pv-vector."""
    return pv[0]


# ---------------------------------------------------------------------------
# VectorMatrix/Initialization
# ---------------------------------------------------------------------------

def ir():
    """Initialize an r-matrix to the identity matrix."""
    return jnp.eye(3)


def zp():
    """Zero a p-vector."""
    return jnp.zeros(3)


def zpv():
    """Zero a pv-vector."""
    return jnp.zeros((2, 3))


def zr():
    """Zero an r-matrix."""
    return jnp.zeros((3, 3))


# ---------------------------------------------------------------------------
# VectorMatrix/MatrixOps
# ---------------------------------------------------------------------------

def rxr(a, b):
    """Product of two r-matrices: a @ b."""
    return a @ b


def tr(r):
    """Transpose an r-matrix."""
    return r.T


# ---------------------------------------------------------------------------
# VectorMatrix/MatrixVectorProducts
# ---------------------------------------------------------------------------

def rxp(r, p):
    """Product of r-matrix and p-vector."""
    return r @ p


def rxpv(r, pv):
    """Product of r-matrix and pv-vector."""
    return jnp.array([r @ pv[0], r @ pv[1]])


def trxp(r, p):
    """Product of transpose of r-matrix and p-vector."""
    return r.T @ p


def trxpv(r, pv):
    """Product of transpose of r-matrix and pv-vector."""
    return jnp.array([r.T @ pv[0], r.T @ pv[1]])


# ---------------------------------------------------------------------------
# VectorMatrix/BuildRotations
#
# NOTE: The C versions mutate r in-place. The JAX versions take (angle, r)
# and return a new matrix: r_new = R(angle) @ r.  This matches the C
# semantics (the ERFA C code applies the rotation as a left-multiply).
# ---------------------------------------------------------------------------

def rx(phi, r):
    """Rotate r-matrix about the x-axis."""
    s = jnp.sin(phi)
    c = jnp.cos(phi)
    row1 = c * r[1] + s * r[2]
    row2 = -s * r[1] + c * r[2]
    return jnp.array([r[0], row1, row2])


def ry(theta, r):
    """Rotate r-matrix about the y-axis."""
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    row0 = c * r[0] - s * r[2]
    row2 = s * r[0] + c * r[2]
    return jnp.array([row0, r[1], row2])


def rz(psi, r):
    """Rotate r-matrix about the z-axis."""
    s = jnp.sin(psi)
    c = jnp.cos(psi)
    row0 = c * r[0] + s * r[1]
    row1 = -s * r[0] + c * r[1]
    return jnp.array([row0, row1, r[2]])


# ---------------------------------------------------------------------------
# VectorMatrix/RotationVectors
# ---------------------------------------------------------------------------

def rm2v(r):
    """Express an r-matrix as an r-vector."""
    x = r[1, 2] - r[2, 1]
    y = r[2, 0] - r[0, 2]
    z = r[0, 1] - r[1, 0]
    s2 = jnp.sqrt(x * x + y * y + z * z)
    c2 = r[0, 0] + r[1, 1] + r[2, 2] - 1.0
    phi = jnp.arctan2(s2, c2)
    f = jnp.where(s2 > 0.0, phi / s2, 0.0)
    return jnp.array([x * f, y * f, z * f])


def rv2m(w):
    """Form the r-matrix corresponding to a given r-vector."""
    x, y, z = w[0], w[1], w[2]
    phi = jnp.sqrt(x * x + y * y + z * z)
    s = jnp.sin(phi)
    c = jnp.cos(phi)
    f = 1.0 - c
    # Normalize axis (safe division)
    x = jnp.where(phi > 0.0, x / phi, x)
    y = jnp.where(phi > 0.0, y / phi, y)
    z = jnp.where(phi > 0.0, z / phi, z)
    return jnp.array([
        [x * x * f + c,     x * y * f + z * s, x * z * f - y * s],
        [y * x * f - z * s, y * y * f + c,     y * z * f + x * s],
        [z * x * f + y * s, z * y * f - x * s, z * z * f + c    ],
    ])


# ---------------------------------------------------------------------------
# VectorMatrix/SeparationAndAngle
# ---------------------------------------------------------------------------

def pap(a, b):
    """Position-angle from two p-vectors."""
    am, au = pn(a)
    bm = pm(b)
    # Null vector case
    null = (am == 0.0) | (bm == 0.0)

    xa, ya, za = a[0], a[1], a[2]
    eta = jnp.array([-xa * za, -ya * za, xa * xa + ya * ya])
    xi = pxp(eta, au)
    a2b = pmp(b, a)
    st = pdp(a2b, xi)
    ct = pdp(a2b, eta)
    # Degenerate cases
    ct = jnp.where((st == 0.0) & (ct == 0.0), 1.0, ct)
    st = jnp.where(null, 0.0, st)
    ct = jnp.where(null, 1.0, ct)
    return jnp.arctan2(st, ct)


def pas(al, ap, bl, bp):
    """Position-angle from spherical coordinates."""
    dl = bl - al
    y = jnp.sin(dl) * jnp.cos(bp)
    x = jnp.sin(bp) * jnp.cos(ap) - jnp.cos(bp) * jnp.sin(ap) * jnp.cos(dl)
    pa = jnp.where((x != 0.0) | (y != 0.0), jnp.arctan2(y, x), 0.0)
    return pa


def sepp(a, b):
    """Angular separation between two p-vectors."""
    axb = pxp(a, b)
    ss = pm(axb)
    cs = pdp(a, b)
    s = jnp.where((ss != 0.0) | (cs != 0.0), jnp.arctan2(ss, cs), 0.0)
    return s


def seps(al, ap, bl, bp):
    """Angular separation between two sets of spherical coordinates."""
    ac = s2c(al, ap)
    bc = s2c(bl, bp)
    return sepp(ac, bc)


# ---------------------------------------------------------------------------
# VectorMatrix/SphericalCartesian
# ---------------------------------------------------------------------------

def c2s(p):
    """P-vector to spherical coordinates.

    Returns (theta, phi) in radians.
    """
    x, y, z = p[0], p[1], p[2]
    d2 = x * x + y * y
    theta = jnp.where(d2 == 0.0, 0.0, jnp.arctan2(y, x))
    phi = jnp.where(d2 + z * z == 0.0, 0.0, jnp.arctan2(z, jnp.sqrt(d2)))
    return theta, phi


def p2s(p):
    """P-vector to spherical polar coordinates.

    Returns (theta, phi, r).
    """
    theta, phi = c2s(p)
    r = pm(p)
    return theta, phi, r


def pv2s(pv):
    """Position/velocity to spherical coordinates.

    Returns (theta, phi, r, td, pd, rd).
    """
    x, y, z = pv[0, 0], pv[0, 1], pv[0, 2]
    xd, yd, zd = pv[1, 0], pv[1, 1], pv[1, 2]

    rxy2 = x * x + y * y
    r2 = rxy2 + z * z
    rtrue = jnp.sqrt(r2)

    # If null position, use velocity to determine direction
    rw = rtrue
    use_vel = rtrue == 0.0
    x_ = jnp.where(use_vel, xd, x)
    y_ = jnp.where(use_vel, yd, y)
    z_ = jnp.where(use_vel, zd, z)
    rxy2_ = jnp.where(use_vel, x_ * x_ + y_ * y_, rxy2)
    r2_ = jnp.where(use_vel, rxy2_ + z_ * z_, r2)
    rw = jnp.where(use_vel, jnp.sqrt(r2_), rw)

    rxy = jnp.sqrt(rxy2_)
    xyp = x * xd + y * yd

    has_xy = rxy2_ != 0.0
    theta = jnp.where(has_xy, jnp.arctan2(y_, x_), 0.0)
    phi = jnp.where(has_xy, jnp.arctan2(z_, rxy),
                    jnp.where(z_ != 0.0, jnp.arctan2(z_, rxy), 0.0))
    td = jnp.where(has_xy, (x * yd - y * xd) / rxy2, 0.0)
    pd = jnp.where(has_xy, (zd * rxy2 - z * xyp) / (r2 * rxy), 0.0)

    r = rtrue
    rd = jnp.where(rw != 0.0, (xyp + z * zd) / rw, 0.0)

    return theta, phi, r, td, pd, rd


def s2c(theta, phi):
    """Spherical to unit p-vector."""
    cp = jnp.cos(phi)
    return jnp.array([
        jnp.cos(theta) * cp,
        jnp.sin(theta) * cp,
        jnp.sin(phi),
    ])


def s2p(theta, phi, r):
    """Spherical polar to p-vector."""
    return r * s2c(theta, phi)


def s2pv(theta, phi, r, td, pd, rd):
    """Spherical to pv-vector."""
    st = jnp.sin(theta)
    ct = jnp.cos(theta)
    sp = jnp.sin(phi)
    cp = jnp.cos(phi)

    rcp = r * cp
    x = rcp * ct
    y = rcp * st
    rpd = r * pd
    w = rpd * sp - rd * cp

    pos = jnp.array([x, y, r * sp])
    vel = jnp.array([
        -y * td - w * ct,
         x * td - w * st,
        rpd * cp + rd * sp,
    ])
    return jnp.array([pos, vel])


__all__ = [
    # VectorOps
    "pdp", "pm", "pmp", "pn", "ppp", "ppsp",
    "pvdpv", "pvm", "pvmpv", "pvppv", "pvu", "pvup", "pvxpv",
    "pxp", "s2xpv", "sxp", "sxpv",
    # CopyExtendExtract
    "cp", "cpv", "cr", "p2pv", "pv2p",
    # Initialization
    "ir", "zp", "zpv", "zr",
    # MatrixOps
    "rxr", "tr",
    # MatrixVectorProducts
    "rxp", "rxpv", "trxp", "trxpv",
    # BuildRotations
    "rx", "ry", "rz",
    # RotationVectors
    "rm2v", "rv2m",
    # SeparationAndAngle
    "pap", "pas", "sepp", "seps",
    # SphericalCartesian
    "c2s", "p2s", "pv2s", "s2c", "s2p", "s2pv",
]
