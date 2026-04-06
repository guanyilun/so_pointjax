"""HEALPix pixelization functions in JAX.

Port of chealpix.c integer pixel indexing to JAX integer ops.
All functions are JIT-compatible.

Convention: theta = colatitude [0, pi], phi = longitude [0, 2*pi].
For astronomical use: theta = pi/2 - dec, phi = ra (both in radians).
"""

import jax
import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TWOTHIRD = 2.0 / 3.0
_PI = jnp.pi
_TWOPI = 2.0 * jnp.pi
_HALFPI = 0.5 * jnp.pi
_INV_HALFPI = 2.0 / jnp.pi

_JRLL = jnp.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=jnp.int32)
_JPLL = jnp.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=jnp.int32)

# ---------------------------------------------------------------------------
# Lookup tables for bit interleaving (NEST ordering)
# ---------------------------------------------------------------------------

def _build_utab():
    """Build spread-bits table: utab[i] spreads 8 bits of i into even positions."""
    t = np.zeros(256, dtype=np.int32)
    for i in range(256):
        t[i] = ((i & 0x1)) | ((i & 0x2) << 1) | ((i & 0x4) << 2) | ((i & 0x8) << 3) | \
               ((i & 0x10) << 4) | ((i & 0x20) << 5) | ((i & 0x40) << 6) | ((i & 0x80) << 7)
    return jnp.array(t, dtype=jnp.int32)

def _build_ctab():
    """Build compress-bits table: ctab[i] compresses alternating bits of i."""
    t = np.zeros(256, dtype=np.int32)
    for i in range(256):
        t[i] = ((i & 0x1)) | ((i & 0x2) >> 1) | ((i & 0x4) >> 1) | ((i & 0x8) >> 2) | \
               ((i & 0x10) >> 2) | ((i & 0x20) >> 3) | ((i & 0x40) >> 3) | ((i & 0x80) >> 4)
    # Actually, match the C code exactly: ctab compresses bits from positions 0,2,4,6,8,...
    # Let me recompute properly matching the C macro
    t2 = np.zeros(256, dtype=np.int32)
    for m in range(256):
        t2[m] = ((m & 0x1)) | ((m & 0x2) << 7) | ((m & 0x4) >> 1) | ((m & 0x8) << 6) | \
                ((m & 0x10) >> 2) | ((m & 0x20) << 5) | ((m & 0x40) >> 3) | ((m & 0x80) << 4)
    return jnp.array(t2, dtype=jnp.int32)

# Actually, let me just compute them the same way the C code does - by expanding the macros.
def _build_tables():
    """Build utab and ctab matching the C implementation exactly."""
    utab = np.zeros(256, dtype=np.int32)
    ctab = np.zeros(256, dtype=np.int32)

    for i in range(256):
        # utab: spread bits of i to even positions
        # bit k of i -> bit 2k of result
        v = 0
        for k in range(8):
            if i & (1 << k):
                v |= (1 << (2 * k))
        utab[i] = v

        # ctab: compress even-position bits
        # bit 2k of i -> bit k of result
        # But the C code's ctab is more complex - it takes interleaved input
        # and produces compressed output. Let me match the C usage exactly.
        pass

    # ctab: used in nest2xyf to compress bits
    # C code: raw = (pix&0x5555) | ((pix&0x55550000)>>15)
    #         ix = ctab[raw&0xff] | (ctab[raw>>8]<<4)
    # So ctab takes a byte where bits are at positions 0,1,8,9 (after the raw computation)
    # and compresses them to positions 0,1,2,3
    # Let me compute ctab from the C macro expansion
    for m in range(256):
        v = ((m & 0x1)) | ((m & 0x2) << 7) | ((m & 0x4) >> 1) | ((m & 0x8) << 6) | \
            ((m & 0x10) >> 2) | ((m & 0x20) << 5) | ((m & 0x40) >> 3) | ((m & 0x80) << 4)
        ctab[m] = v

    return jnp.array(utab, dtype=jnp.int32), jnp.array(ctab, dtype=jnp.int32)

_UTAB, _CTAB = _build_tables()


# ---------------------------------------------------------------------------
# Internal: (x, y, face) <-> pixel conversions
# ---------------------------------------------------------------------------

def _xyf2nest(nside, ix, iy, face_num):
    """Convert (x, y, face) to NEST pixel index."""
    return (face_num * nside * nside +
            (_UTAB[ix & 0xff] | (_UTAB[ix >> 8] << 16) |
             (_UTAB[iy & 0xff] << 1) | (_UTAB[iy >> 8] << 17)))


def _nest2xyf(nside, pix):
    """Convert NEST pixel to (ix, iy, face_num)."""
    npface = nside * nside
    face_num = pix // npface
    pix = pix & (npface - 1)
    raw = (pix & 0x5555) | ((pix & 0x55550000) >> 15)
    ix = _CTAB[raw & 0xff] | (_CTAB[raw >> 8] << 4)
    pix = pix >> 1
    raw = (pix & 0x5555) | ((pix & 0x55550000) >> 15)
    iy = _CTAB[raw & 0xff] | (_CTAB[raw >> 8] << 4)
    return ix, iy, face_num


def _xyf2ring(nside, ix, iy, face_num):
    """Convert (x, y, face) to RING pixel index."""
    nl4 = 4 * nside
    jr = _JRLL[face_num] * nside - ix - iy - 1

    # North polar cap
    nr_north = jr
    n_before_north = 2 * nr_north * (nr_north - 1)
    kshift_north = 0

    # South polar cap
    nr_south = nl4 - jr
    n_before_south = 12 * nside * nside - 2 * (nr_south + 1) * nr_south
    kshift_south = 0

    # Equatorial region
    ncap = 2 * nside * (nside - 1)
    nr_eq = nside
    n_before_eq = ncap + (jr - nside) * nl4
    kshift_eq = (jr - nside) & 1

    is_north = jr < nside
    is_south = jr > 3 * nside

    nr = jnp.where(is_north, nr_north, jnp.where(is_south, nr_south, nr_eq))
    n_before = jnp.where(is_north, n_before_north,
                         jnp.where(is_south, n_before_south, n_before_eq))
    kshift = jnp.where(is_north, kshift_north,
                       jnp.where(is_south, kshift_south, kshift_eq))

    jp = (_JPLL[face_num] * nr + ix - iy + 1 + kshift) // 2
    jp = jnp.where(jp > nl4, jp - nl4, jnp.where(jp < 1, jp + nl4, jp))

    return n_before + jp - 1


# ---------------------------------------------------------------------------
# Core: ang2pix (theta, phi -> pixel)
# ---------------------------------------------------------------------------

def _ang2pix_nest_z_phi(nside, z, phi):
    """Internal: z=cos(theta), phi -> NEST pixel."""
    za = jnp.abs(z)
    tt = jnp.mod(phi, _TWOPI) * _INV_HALFPI  # in [0, 4)

    # Equatorial region (za <= 2/3)
    temp1_eq = nside * (0.5 + tt)
    temp2_eq = nside * (z * 0.75)
    jp_eq = (temp1_eq - temp2_eq).astype(jnp.int32)
    jm_eq = (temp1_eq + temp2_eq).astype(jnp.int32)
    ifp_eq = jp_eq // nside
    ifm_eq = jm_eq // nside
    face_eq = jnp.where(ifp_eq == ifm_eq, ifp_eq | 4,
                        jnp.where(ifp_eq < ifm_eq, ifp_eq, ifm_eq + 8))
    ix_eq = jm_eq & (nside - 1)
    iy_eq = nside - (jp_eq & (nside - 1)) - 1

    # Polar regions (za > 2/3)
    ntt = jnp.minimum(tt.astype(jnp.int32), 3)
    tp = tt - ntt
    tmp = nside * jnp.sqrt(3 * (1 - za))
    jp_pol = jnp.minimum((tp * tmp).astype(jnp.int32), nside - 1)
    jm_pol = jnp.minimum(((1.0 - tp) * tmp).astype(jnp.int32), nside - 1)

    # North pole
    face_n = ntt
    ix_n = nside - jm_pol - 1
    iy_n = nside - jp_pol - 1
    # South pole
    face_s = ntt + 8
    ix_s = jp_pol
    iy_s = jm_pol

    face_pol = jnp.where(z >= 0, face_n, face_s)
    ix_pol = jnp.where(z >= 0, ix_n, ix_s)
    iy_pol = jnp.where(z >= 0, iy_n, iy_s)

    is_equatorial = za <= _TWOTHIRD
    face = jnp.where(is_equatorial, face_eq, face_pol)
    ix = jnp.where(is_equatorial, ix_eq, ix_pol)
    iy = jnp.where(is_equatorial, iy_eq, iy_pol)

    return _xyf2nest(nside, ix, iy, face)


def _ang2pix_ring_z_phi(nside, z, phi):
    """Internal: z=cos(theta), phi -> RING pixel."""
    za = jnp.abs(z)
    tt = jnp.mod(phi, _TWOPI) * _INV_HALFPI

    # Equatorial region
    temp1 = nside * (0.5 + tt)
    temp2 = nside * z * 0.75
    jp_eq = (temp1 - temp2).astype(jnp.int32)
    jm_eq = (temp1 + temp2).astype(jnp.int32)
    ir_eq = nside + 1 + jp_eq - jm_eq
    kshift_eq = 1 - (ir_eq & 1)
    ip_eq = (jp_eq + jm_eq - nside + kshift_eq + 1) // 2
    ip_eq = ip_eq % (4 * nside)
    pix_eq = nside * (nside - 1) * 2 + (ir_eq - 1) * 4 * nside + ip_eq

    # Polar regions
    tp = tt - tt.astype(jnp.int32)
    tmp = nside * jnp.sqrt(3 * (1 - za))
    jp_pol = (tp * tmp).astype(jnp.int32)
    jm_pol = ((1.0 - tp) * tmp).astype(jnp.int32)
    ir_pol = jp_pol + jm_pol + 1
    ip_pol = (tt * ir_pol).astype(jnp.int32)
    ip_pol = ip_pol % (4 * ir_pol)

    pix_north = 2 * ir_pol * (ir_pol - 1) + ip_pol
    pix_south = 12 * nside * nside - 2 * ir_pol * (ir_pol + 1) + ip_pol
    pix_pol = jnp.where(z > 0, pix_north, pix_south)

    return jnp.where(za <= _TWOTHIRD, pix_eq, pix_pol)


# ---------------------------------------------------------------------------
# Core: pix2ang (pixel -> theta, phi)
# ---------------------------------------------------------------------------

def _pix2ang_ring_z_phi(nside, pix):
    """Internal: RING pixel -> (z, phi)."""
    ncap = nside * (nside - 1) * 2
    npix = 12 * nside * nside
    fact2 = 4.0 / npix

    # North polar cap
    iring_n = (1 + jnp.sqrt(1 + 2 * pix).astype(jnp.int32)) >> 1
    iphi_n = pix + 1 - 2 * iring_n * (iring_n - 1)
    z_n = 1.0 - (iring_n * iring_n) * fact2
    phi_n = (iphi_n - 0.5) * _HALFPI / iring_n

    # Equatorial region
    fact1 = (nside << 1) * fact2
    ip_eq = pix - ncap
    iring_eq = ip_eq // (4 * nside) + nside
    iphi_eq = ip_eq % (4 * nside) + 1
    fodd_eq = jnp.where((iring_eq + nside) & 1, 1.0, 0.5)
    nl2 = 2 * nside
    z_eq = (nl2 - iring_eq) * fact1
    phi_eq = (iphi_eq - fodd_eq) * _PI / nl2

    # South polar cap
    ip_s = npix - pix
    iring_s = (1 + jnp.sqrt(2 * ip_s - 1).astype(jnp.int32)) >> 1
    iphi_s = 4 * iring_s + 1 - (ip_s - 2 * iring_s * (iring_s - 1))
    z_s = -1.0 + (iring_s * iring_s) * fact2
    phi_s = (iphi_s - 0.5) * _HALFPI / iring_s

    is_north = pix < ncap
    is_south = pix >= (npix - ncap)

    z = jnp.where(is_north, z_n, jnp.where(is_south, z_s, z_eq))
    phi = jnp.where(is_north, phi_n, jnp.where(is_south, phi_s, phi_eq))
    return z, phi


def _pix2ang_nest_z_phi(nside, pix):
    """Internal: NEST pixel -> (z, phi)."""
    nl4 = nside * 4
    npix = 12 * nside * nside
    fact2 = 4.0 / npix

    ix, iy, face_num = _nest2xyf(nside, pix)
    jr = _JRLL[face_num] * nside - ix - iy - 1

    # North polar cap
    nr_n = jr
    z_n = 1.0 - nr_n * nr_n * fact2
    kshift_n = 0

    # South polar cap
    nr_s = nl4 - jr
    z_s = nr_s * nr_s * fact2 - 1.0
    kshift_s = 0

    # Equatorial
    fact1 = (nside << 1) * fact2
    nr_eq = nside
    z_eq = (2 * nside - jr) * fact1
    kshift_eq = (jr - nside) & 1

    is_north = jr < nside
    is_south = jr > 3 * nside

    nr = jnp.where(is_north, nr_n, jnp.where(is_south, nr_s, nr_eq))
    z = jnp.where(is_north, z_n, jnp.where(is_south, z_s, z_eq))
    kshift = jnp.where(is_north, kshift_n, jnp.where(is_south, kshift_s, kshift_eq))

    jp = (_JPLL[face_num] * nr + ix - iy + 1 + kshift) // 2
    jp = jnp.where(jp > nl4, jp - nl4, jnp.where(jp < 1, jp + nl4, jp))

    phi = (jp - (kshift + 1) * 0.5) * (_HALFPI / nr)
    return z, phi


# ---------------------------------------------------------------------------
# Public API: angle-based
# ---------------------------------------------------------------------------

def ang2pix_nest(nside, theta, phi):
    """Convert (theta, phi) to NEST pixel index.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter (must be power of 2).
    theta : float
        Colatitude in radians [0, pi].
    phi : float
        Longitude in radians [0, 2*pi].

    Returns
    -------
    pix : int
    """
    return _ang2pix_nest_z_phi(nside, jnp.cos(theta), phi)


def ang2pix_ring(nside, theta, phi):
    """Convert (theta, phi) to RING pixel index."""
    return _ang2pix_ring_z_phi(nside, jnp.cos(theta), phi)


def pix2ang_nest(nside, pix):
    """Convert NEST pixel to (theta, phi).

    Returns
    -------
    theta : float
        Colatitude in radians.
    phi : float
        Longitude in radians.
    """
    z, phi = _pix2ang_nest_z_phi(nside, pix)
    return jnp.arccos(z), phi


def pix2ang_ring(nside, pix):
    """Convert RING pixel to (theta, phi)."""
    z, phi = _pix2ang_ring_z_phi(nside, pix)
    return jnp.arccos(z), phi


# ---------------------------------------------------------------------------
# Public API: vector-based
# ---------------------------------------------------------------------------

def vec2pix_nest(nside, vec):
    """Convert unit vector to NEST pixel.

    Parameters
    ----------
    nside : int
    vec : array shape (3,)
        [x, y, z] unit vector.
    """
    vlen = jnp.sqrt(jnp.dot(vec, vec))
    z = vec[2] / vlen
    phi = jnp.arctan2(vec[1], vec[0])
    phi = jnp.where(phi < 0, phi + _TWOPI, phi)
    return _ang2pix_nest_z_phi(nside, z, phi)


def vec2pix_ring(nside, vec):
    """Convert unit vector to RING pixel."""
    vlen = jnp.sqrt(jnp.dot(vec, vec))
    z = vec[2] / vlen
    phi = jnp.arctan2(vec[1], vec[0])
    phi = jnp.where(phi < 0, phi + _TWOPI, phi)
    return _ang2pix_ring_z_phi(nside, z, phi)


def pix2vec_nest(nside, pix):
    """Convert NEST pixel to unit vector [x, y, z]."""
    z, phi = _pix2ang_nest_z_phi(nside, pix)
    stheta = jnp.sqrt((1 - z) * (1 + z))
    return jnp.array([stheta * jnp.cos(phi), stheta * jnp.sin(phi), z])


def pix2vec_ring(nside, pix):
    """Convert RING pixel to unit vector [x, y, z]."""
    z, phi = _pix2ang_ring_z_phi(nside, pix)
    stheta = jnp.sqrt((1 - z) * (1 + z))
    return jnp.array([stheta * jnp.cos(phi), stheta * jnp.sin(phi), z])


# ---------------------------------------------------------------------------
# Public API: ordering conversions
# ---------------------------------------------------------------------------

def nest2ring(nside, ipnest):
    """Convert NEST pixel to RING pixel."""
    ix, iy, face_num = _nest2xyf(nside, ipnest)
    return _xyf2ring(nside, ix, iy, face_num)


def ring2nest(nside, ipring):
    """Convert RING pixel to NEST pixel."""
    ix, iy, face_num = _ring2xyf(nside, ipring)
    return _xyf2nest(nside, ix, iy, face_num)


def _ring2xyf(nside, pix):
    """Convert RING pixel to (ix, iy, face_num)."""
    ncap = 2 * nside * (nside - 1)
    npix = 12 * nside * nside
    nl2 = 2 * nside

    # North polar cap
    iring_n = (1 + _isqrt(1 + 2 * pix)) >> 1
    iphi_n = pix + 1 - 2 * iring_n * (iring_n - 1)
    kshift_n = 0
    nr_n = iring_n
    face_n = _special_div(iphi_n - 1, nr_n)

    # Equatorial region
    ip_eq = pix - ncap
    iring_eq = ip_eq // (4 * nside) + nside
    iphi_eq = ip_eq % (4 * nside) + 1
    kshift_eq = (iring_eq + nside) & 1
    nr_eq = nside
    ire_eq = iring_eq - nside + 1
    irm_eq = nl2 + 2 - ire_eq
    ifm_eq = (iphi_eq - ire_eq // 2 + nside - 1) // nside
    ifp_eq = (iphi_eq - irm_eq // 2 + nside - 1) // nside
    face_eq = jnp.where(ifp_eq == ifm_eq, ifp_eq | 4,
                        jnp.where(ifp_eq < ifm_eq, ifp_eq, ifm_eq + 8))

    # South polar cap
    ip_s = npix - pix
    iring_s = (1 + _isqrt(2 * ip_s - 1)) >> 1
    iphi_s = 4 * iring_s + 1 - (ip_s - 2 * iring_s * (iring_s - 1))
    kshift_s = 0
    nr_s = iring_s
    iring_s_full = 2 * nl2 - iring_s
    face_s = 8 + _special_div(iphi_s - 1, nr_s)

    is_north = pix < ncap
    is_south = pix >= (npix - ncap)

    iring = jnp.where(is_north, iring_n, jnp.where(is_south, iring_s_full, iring_eq))
    iphi = jnp.where(is_north, iphi_n, jnp.where(is_south, iphi_s, iphi_eq))
    kshift = jnp.where(is_north, kshift_n, jnp.where(is_south, kshift_s, kshift_eq))
    nr = jnp.where(is_north, nr_n, jnp.where(is_south, nr_s, nr_eq))
    face_num = jnp.where(is_north, face_n, jnp.where(is_south, face_s, face_eq))

    irt = iring - _JRLL[face_num] * nside + 1
    ipt = 2 * iphi - _JPLL[face_num] * nr - kshift - 1
    ipt = jnp.where(ipt >= nl2, ipt - 8 * nside, ipt)

    ix = (ipt - irt) >> 1
    iy = (-(ipt + irt)) >> 1
    return ix, iy, face_num


def _isqrt(v):
    """Integer square root."""
    v = jnp.asarray(v)
    return jnp.sqrt(v.astype(jnp.float64) + 0.5).astype(jnp.int32)


def _special_div(a, b):
    """Compute a // b for a in [0, 3*b), returning result in {0, 1, 2, 3}."""
    t = (a >= (b << 1)).astype(jnp.int32)
    a2 = a - t * (b << 1)
    return (t << 1) + (a2 >= b).astype(jnp.int32)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def nside2npix(nside):
    """Number of pixels for given nside."""
    return 12 * nside * nside


def npix2nside(npix):
    """Extract nside from pixel count. Returns -1 if invalid."""
    nside = jnp.sqrt(npix / 12.0).astype(jnp.int32)
    return jnp.where(nside * nside * 12 == npix, nside, -1)


# ---------------------------------------------------------------------------
# Astronomical convenience: RA/Dec interface
# ---------------------------------------------------------------------------

def radec2pix(nside, ra, dec, nest=True):
    """Convert (RA, Dec) in degrees to pixel index.

    Parameters
    ----------
    nside : int
    ra : float
        Right ascension (degrees).
    dec : float
        Declination (degrees).
    nest : bool
        If True, NEST ordering. If False, RING ordering.

    Returns
    -------
    pix : int
    """
    theta = _HALFPI - jnp.deg2rad(dec)
    phi = jnp.deg2rad(ra)
    if nest:
        return ang2pix_nest(nside, theta, phi)
    return ang2pix_ring(nside, theta, phi)


def pix2radec(nside, pix, nest=True):
    """Convert pixel index to (RA, Dec) in degrees.

    Returns
    -------
    ra, dec : float (degrees)
    """
    if nest:
        theta, phi = pix2ang_nest(nside, pix)
    else:
        theta, phi = pix2ang_ring(nside, pix)
    dec = jnp.rad2deg(_HALFPI - theta)
    ra = jnp.rad2deg(phi)
    return ra, dec


# ---------------------------------------------------------------------------
# Pointing pipeline integration
# ---------------------------------------------------------------------------

def quat2pix(q, nside, nest=True):
    """Convert quaternion to pixel index and polarization angle.

    Uses the "fast" method: extract Z-axis from quaternion rotation matrix.

    Parameters
    ----------
    q : array shape (4,)
        Quaternion [w, x, y, z].
    nside : int
    nest : bool

    Returns
    -------
    pix : int
    sin2psi : float
    cos2psi : float
    """
    from so_pointjax.qpoint._quaternion import to_col3

    vec = to_col3(q)
    if nest:
        pix = vec2pix_nest(nside, vec)
    else:
        pix = vec2pix_ring(nside, vec)

    # Compute polarization angle from quaternion
    # Matches qp_quat2pix in qp_pixel.c
    cosb2 = (1 - vec[2] * vec[2]) / 4.0

    # Near poles
    cosg_pole_n = q[3] * q[3] - q[0] * q[0]
    sing_pole_n = 2 * q[0] * q[3]
    cosg_pole_s = q[1] * q[1] - q[2] * q[2]
    sing_pole_s = 2 * q[1] * q[2]

    cosg_pole = jnp.where(vec[2] > 0, cosg_pole_n, cosg_pole_s)
    sing_pole = jnp.where(vec[2] > 0, sing_pole_n, sing_pole_s)
    norm_pole = 2 * cosg_pole

    # General case
    cosg_gen = q[1] * q[3] - q[0] * q[2]
    sing_gen = q[0] * q[1] + q[2] * q[3]
    norm_gen = 2.0 * cosg_gen / cosb2

    near_pole = cosb2 < 1e-15

    norm = jnp.where(near_pole, norm_pole, norm_gen)
    sing = jnp.where(near_pole, sing_pole, sing_gen)
    cosg = jnp.where(near_pole, cosg_pole, cosg_gen)

    sin2psi = norm * sing
    cos2psi = norm * cosg - 1

    return pix, sin2psi, cos2psi


def bore2pix(q_off, q_bore, nside, nest=True):
    """Convert boresight + detector offset to pixel and polarization.

    Parameters
    ----------
    q_off : array shape (4,)
        Detector offset quaternion.
    q_bore : array shape (4,)
        Boresight quaternion (output of azel2bore or similar).
    nside : int
    nest : bool

    Returns
    -------
    pix : int
    sin2psi : float
    cos2psi : float
    """
    from so_pointjax.qpoint._quaternion import mul
    q_det = mul(q_bore, q_off)
    return quat2pix(q_det, nside, nest=nest)
