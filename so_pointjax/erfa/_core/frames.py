"""Coordinate frame transforms, ported from ERFA C library.

Includes: horizon/equatorial, galactic/ICRS, ecliptic, and star catalog transforms.
"""

import jax.numpy as jnp

from so_pointjax.erfa._core.constants import D2PI, DPI, DAS2R, DD2R, DR2AS
from so_pointjax.erfa._core.angles import anp, anpm
from so_pointjax.erfa._core.vector import (
    s2c, c2s, rxp, trxp, rxr, pxp, pn, ppp, pmp, sxp, ppsp, ir, rx, rz,
    rv2m, pdp, pm, pvu, s2pv, pv2s, pvmpv, pvppv,
)


# ============================================================================
# Horizon / Equatorial
# ============================================================================


def ae2hd(az, el, phi):
    """Horizon to equatorial: azimuth, altitude to hour angle, declination.

    Parameters
    ----------
    az : float
        Azimuth (radians, N=0, E=+pi/2)
    el : float
        Altitude/elevation (radians)
    phi : float
        Site latitude (radians)

    Returns
    -------
    ha : float
        Hour angle (radians)
    dec : float
        Declination (radians)
    """
    sa = jnp.sin(az)
    ca = jnp.cos(az)
    se = jnp.sin(el)
    ce = jnp.cos(el)
    sp = jnp.sin(phi)
    cp = jnp.cos(phi)

    # HA,Dec unit vector.
    x = -ca * ce * sp + se * cp
    y = -sa * ce
    z = ca * ce * cp + se * sp

    # To spherical.
    r = jnp.sqrt(x * x + y * y)
    ha = jnp.where(r != 0.0, jnp.arctan2(y, x), 0.0)
    dec = jnp.arctan2(z, r)

    return ha, dec


def hd2ae(ha, dec, phi):
    """Equatorial to horizon: hour angle, declination to azimuth, altitude.

    Parameters
    ----------
    ha : float
        Hour angle (radians)
    dec : float
        Declination (radians)
    phi : float
        Site latitude (radians)

    Returns
    -------
    az : float
        Azimuth (radians, N=0, E=+pi/2)
    el : float
        Altitude/elevation (radians)
    """
    sh = jnp.sin(ha)
    ch = jnp.cos(ha)
    sd = jnp.sin(dec)
    cd = jnp.cos(dec)
    sp = jnp.sin(phi)
    cp = jnp.cos(phi)

    # Az,Alt unit vector.
    x = -ch * cd * sp + sd * cp
    y = -sh * cd
    z = ch * cd * cp + sd * sp

    # To spherical.
    r = jnp.sqrt(x * x + y * y)
    a = jnp.where(r != 0.0, jnp.arctan2(y, x), 0.0)
    az = jnp.where(a < 0.0, a + D2PI, a)
    el = jnp.arctan2(z, r)

    return az, el


def hd2pa(ha, dec, phi):
    """Parallactic angle for given hour angle and declination.

    Parameters
    ----------
    ha : float
        Hour angle (radians)
    dec : float
        Declination (radians)
    phi : float
        Site latitude (radians)

    Returns
    -------
    pa : float
        Parallactic angle (radians, range -pi to +pi)
    """
    cp = jnp.cos(phi)
    sqsz = cp * jnp.sin(ha)
    cqsz = jnp.sin(phi) * jnp.cos(dec) - cp * jnp.sin(dec) * jnp.cos(ha)
    return jnp.where((sqsz != 0.0) | (cqsz != 0.0),
                     jnp.arctan2(sqsz, cqsz), 0.0)


# ============================================================================
# Galactic / ICRS
# ============================================================================

# ICRS to Galactic rotation matrix (from Hipparcos Catalogue).
_R_ICRS2GAL = jnp.array([
    [-0.054875560416215368492398900454,
     -0.873437090234885048760383168409,
     -0.483835015548713226831774175116],
    [+0.494109427875583673525222371358,
     -0.444829629960011178146614061616,
     +0.746982244497218890527388004556],
    [-0.867666149019004701181616534570,
     -0.198076373431201528180486091412,
     +0.455983776175066922272100478348],
])


def icrs2g(dr, dd):
    """ICRS to Galactic coordinates.

    Parameters
    ----------
    dr : float
        ICRS right ascension (radians)
    dd : float
        ICRS declination (radians)

    Returns
    -------
    dl : float
        Galactic longitude (radians)
    db : float
        Galactic latitude (radians)
    """
    v1 = s2c(dr, dd)
    v2 = rxp(_R_ICRS2GAL, v1)
    dl, db = c2s(v2)
    dl = anp(dl)
    db = anpm(db)
    return dl, db


def g2icrs(dl, db):
    """Galactic to ICRS coordinates.

    Parameters
    ----------
    dl : float
        Galactic longitude (radians)
    db : float
        Galactic latitude (radians)

    Returns
    -------
    dr : float
        ICRS right ascension (radians)
    dd : float
        ICRS declination (radians)
    """
    v1 = s2c(dl, db)
    v2 = trxp(_R_ICRS2GAL, v1)
    dr, dd = c2s(v2)
    dr = anp(dr)
    dd = anpm(dd)
    return dr, dd


# ============================================================================
# Ecliptic coordinate transforms (IAU 2006)
# ============================================================================


def ecm06(date1, date2):
    """ICRS equatorial to ecliptic rotation matrix, IAU 2006.

    Parameters
    ----------
    date1, date2 : float
        TT as a 2-part Julian Date

    Returns
    -------
    rm : ndarray, shape (3, 3)
        ICRS to ecliptic rotation matrix
    """
    from so_pointjax.erfa._core.precnut import obl06, pmat06

    ob = obl06(date1, date2)
    bp = pmat06(date1, date2)
    e = rx(ob, ir())
    return rxr(e, bp)


def eqec06(date1, date2, dr, dd):
    """ICRS equatorial to ecliptic coordinates (IAU 2006).

    Parameters
    ----------
    date1, date2 : float
        TT as a 2-part Julian Date
    dr, dd : float
        ICRS right ascension and declination (radians)

    Returns
    -------
    dl : float
        Ecliptic longitude (radians)
    db : float
        Ecliptic latitude (radians)
    """
    v1 = s2c(dr, dd)
    rm = ecm06(date1, date2)
    v2 = rxp(rm, v1)
    a, b = c2s(v2)
    return anp(a), anpm(b)


def eceq06(date1, date2, dl, db):
    """Ecliptic to ICRS equatorial coordinates (IAU 2006).

    Parameters
    ----------
    date1, date2 : float
        TT as a 2-part Julian Date
    dl, db : float
        Ecliptic longitude and latitude (radians)

    Returns
    -------
    dr : float
        ICRS right ascension (radians)
    dd : float
        ICRS declination (radians)
    """
    v1 = s2c(dl, db)
    rm = ecm06(date1, date2)
    v2 = trxp(rm, v1)
    a, b = c2s(v2)
    return anp(a), anpm(b)


# ============================================================================
# Long-term precession (Vondrak et al. 2011, 2012)
# ============================================================================

# Ecliptic pole polynomial coefficients (arcsec).
_PQPOL = jnp.array([
    [5851.607687, -0.1189000, -0.00028913, 0.000000101],
    [-1600.886300, 1.1689818, -0.00000020, -0.000000437],
])

# Ecliptic pole periodic coefficients [period, p_cos, q_cos, p_sin, q_sin].
_PQPER = jnp.array([
    [708.15, -5486.751211, -684.661560, 667.666730, -5523.863691],
    [2309.00, -17.127623, 2446.283880, -2354.886252, -549.747450],
    [1620.00, -617.517403, 399.671049, -428.152441, -310.998056],
    [492.20, 413.442940, -356.652376, 376.202861, 421.535876],
    [1183.00, 78.614193, -186.387003, 184.778874, -36.776172],
    [622.00, -180.732815, -316.800070, 335.321713, -145.278396],
    [882.00, -87.676083, 198.296701, -185.138669, -34.744450],
    [547.00, 46.140315, 101.135679, -120.972830, 22.885731],
])

# Equator pole polynomial coefficients (arcsec).
_XYPOL = jnp.array([
    [5453.282155, 0.4252841, -0.00037173, -0.000000152],
    [-73750.930350, -0.7675452, -0.00018725, 0.000000231],
])

# Equator pole periodic coefficients [period, x_cos, y_cos, x_sin, y_sin].
_XYPER = jnp.array([
    [256.75, -819.940624, 75004.344875, 81491.287984, 1558.515853],
    [708.15, -8444.676815, 624.033993, 787.163481, 7774.939698],
    [274.20, 2600.009459, 1251.136893, 1251.296102, -2219.534038],
    [241.45, 2755.175630, -1102.212834, -1257.950837, -2523.969396],
    [2309.00, -167.659835, -2660.664980, -2966.799730, 247.850422],
    [492.20, 871.855056, 699.291817, 639.744522, -846.485643],
    [396.10, 44.769698, 153.167220, 131.600209, -1393.124055],
    [288.90, -512.313065, -950.865637, -445.040117, 368.526116],
    [231.10, -819.415595, 499.754645, 584.522874, 749.045012],
    [1610.00, -538.071099, -145.188210, -89.756563, 444.704518],
    [620.00, -189.793622, 558.116553, 524.429630, 235.934465],
    [157.87, -402.922932, -23.923029, -13.549067, 374.049623],
    [220.30, 179.516345, -165.405086, -210.157124, -171.330180],
    [1200.00, -9.814756, 9.344131, -44.919798, -22.899655],
])

# Obliquity at J2000.0 (radians).
_EPS0 = 84381.406 * DAS2R


def ltpecl(epj):
    """Long-term precession of the ecliptic pole.

    Parameters
    ----------
    epj : float
        Julian epoch (TT)

    Returns
    -------
    vec : ndarray, shape (3,)
        Ecliptic pole unit vector (J2000.0 mean equator and equinox)
    """
    t = (epj - 2000.0) / 100.0
    w = D2PI * t

    # Periodic terms.
    a = w / _PQPER[:, 0]
    s = jnp.sin(a)
    c = jnp.cos(a)
    p = jnp.sum(c * _PQPER[:, 1] + s * _PQPER[:, 3])
    q = jnp.sum(c * _PQPER[:, 2] + s * _PQPER[:, 4])

    # Polynomial terms.
    tw = jnp.array([1.0, t, t * t, t * t * t])
    p = p + jnp.sum(_PQPOL[0] * tw)
    q = q + jnp.sum(_PQPOL[1] * tw)

    # P_A and Q_A (radians).
    p = p * DAS2R
    q = q * DAS2R

    # Form the ecliptic pole vector.
    w2 = 1.0 - p * p - q * q
    w_val = jnp.where(w2 < 0.0, 0.0, jnp.sqrt(w2))
    s_eps = jnp.sin(_EPS0)
    c_eps = jnp.cos(_EPS0)

    return jnp.array([p, -q * c_eps - w_val * s_eps, -q * s_eps + w_val * c_eps])


def ltpequ(epj):
    """Long-term precession of the equator pole.

    Parameters
    ----------
    epj : float
        Julian epoch (TT)

    Returns
    -------
    veq : ndarray, shape (3,)
        Equator pole unit vector (J2000.0 mean equator and equinox)
    """
    t = (epj - 2000.0) / 100.0
    w = D2PI * t

    # Periodic terms.
    a = w / _XYPER[:, 0]
    s = jnp.sin(a)
    c = jnp.cos(a)
    x = jnp.sum(c * _XYPER[:, 1] + s * _XYPER[:, 3])
    y = jnp.sum(c * _XYPER[:, 2] + s * _XYPER[:, 4])

    # Polynomial terms.
    tw = jnp.array([1.0, t, t * t, t * t * t])
    x = x + jnp.sum(_XYPOL[0] * tw)
    y = y + jnp.sum(_XYPOL[1] * tw)

    # X and Y (direction cosines).
    x = x * DAS2R
    y = y * DAS2R

    # Form the equator pole vector.
    w2 = 1.0 - x * x - y * y
    z = jnp.where(w2 < 0.0, 0.0, jnp.sqrt(w2))

    return jnp.array([x, y, z])


# Frame bias constants (IERS Conventions 2010, Eqs. 5.21 and 5.33).
_DX = -0.016617 * DAS2R
_DE = -0.0068192 * DAS2R
_DR = -0.0146 * DAS2R


def ltecm(epj):
    """ICRS equatorial to ecliptic rotation matrix, long-term.

    Parameters
    ----------
    epj : float
        Julian epoch (TT)

    Returns
    -------
    rm : ndarray, shape (3, 3)
        ICRS to ecliptic rotation matrix
    """
    p = ltpequ(epj)
    z = ltpecl(epj)

    # Equinox (top row of matrix).
    w = pxp(p, z)
    _, x = pn(w)

    # Middle row.
    y = pxp(z, x)

    # Combine with frame bias.
    rm = jnp.array([
        [x[0] - x[1] * _DR + x[2] * _DX,
         x[0] * _DR + x[1] + x[2] * _DE,
         -x[0] * _DX - x[1] * _DE + x[2]],
        [y[0] - y[1] * _DR + y[2] * _DX,
         y[0] * _DR + y[1] + y[2] * _DE,
         -y[0] * _DX - y[1] * _DE + y[2]],
        [z[0] - z[1] * _DR + z[2] * _DX,
         z[0] * _DR + z[1] + z[2] * _DE,
         -z[0] * _DX - z[1] * _DE + z[2]],
    ])
    return rm


def ltp(epj):
    """Long-term precession matrix, J2000.0 to date.

    Parameters
    ----------
    epj : float
        Julian epoch (TT)

    Returns
    -------
    rp : ndarray, shape (3, 3)
        Precession matrix, J2000.0 to date
    """
    peqr = ltpequ(epj)
    pecl = ltpecl(epj)

    # Equinox (top row).
    v = pxp(peqr, pecl)
    _, eqx = pn(v)

    # Middle row.
    mid = pxp(peqr, eqx)

    return jnp.array([eqx, mid, peqr])


def ltpb(epj):
    """Long-term precession matrix, including ICRS frame bias.

    Parameters
    ----------
    epj : float
        Julian epoch (TT)

    Returns
    -------
    rpb : ndarray, shape (3, 3)
        Precession+bias matrix, J2000.0 to date
    """
    rp = ltp(epj)

    # Apply the bias.
    rpb = jnp.array([
        [rp[0, 0] - rp[0, 1] * _DR + rp[0, 2] * _DX,
         rp[0, 0] * _DR + rp[0, 1] + rp[0, 2] * _DE,
         -rp[0, 0] * _DX - rp[0, 1] * _DE + rp[0, 2]],
        [rp[1, 0] - rp[1, 1] * _DR + rp[1, 2] * _DX,
         rp[1, 0] * _DR + rp[1, 1] + rp[1, 2] * _DE,
         -rp[1, 0] * _DX - rp[1, 1] * _DE + rp[1, 2]],
        [rp[2, 0] - rp[2, 1] * _DR + rp[2, 2] * _DX,
         rp[2, 0] * _DR + rp[2, 1] + rp[2, 2] * _DE,
         -rp[2, 0] * _DX - rp[2, 1] * _DE + rp[2, 2]],
    ])
    return rpb


def lteqec(epj, dr, dd):
    """ICRS RA,Dec to ecliptic coordinates (long-term precession).

    Parameters
    ----------
    epj : float
        Julian epoch (TT)
    dr, dd : float
        ICRS right ascension and declination (radians)

    Returns
    -------
    dl : float
        Ecliptic longitude (radians)
    db : float
        Ecliptic latitude (radians)
    """
    v1 = s2c(dr, dd)
    rm = ltecm(epj)
    v2 = rxp(rm, v1)
    a, b = c2s(v2)
    return anp(a), anpm(b)


# ============================================================================
# Star catalog transforms: FK5 <-> Hipparcos
# ============================================================================

# FK5 wrt Hipparcos orientation and spin (radians, radians/year).
_EPX = -19.9e-3 * DAS2R
_EPY = -9.1e-3 * DAS2R
_EPZ = 22.9e-3 * DAS2R
_OMX = -0.30e-3 * DAS2R
_OMY = 0.60e-3 * DAS2R
_OMZ = 0.70e-3 * DAS2R


def fk5hip():
    """FK5 to Hipparcos rotation and spin.

    Returns
    -------
    r5h : ndarray, shape (3, 3)
        FK5 rotation wrt Hipparcos
    s5h : ndarray, shape (3,)
        FK5 spin wrt Hipparcos (rad/year)
    """
    v = jnp.array([_EPX, _EPY, _EPZ])
    r5h = rv2m(v)
    s5h = jnp.array([_OMX, _OMY, _OMZ])
    return r5h, s5h


def fk52h(r5, d5, dr5, dd5, px5, rv5):
    """FK5 (J2000.0) to Hipparcos star data.

    Parameters
    ----------
    r5, d5 : float
        FK5 RA, Dec (radians)
    dr5, dd5 : float
        FK5 proper motions (rad/Jyear)
    px5 : float
        Parallax (arcsec)
    rv5 : float
        Radial velocity (km/s, +ve = receding)

    Returns
    -------
    rh, dh, drh, ddh, pxh, rvh : float
        Hipparcos equivalents
    """
    from so_pointjax.erfa._core.astrometry import starpv, pvstar

    # FK5 barycentric pv-vector.
    pv5, _ = starpv(r5, d5, dr5, dd5, px5, rv5)

    # FK5 to Hipparcos orientation matrix and spin vector.
    r5h, s5h = fk5hip()

    # Make spin units per day instead of per year.
    s5h = s5h / 365.25

    # Orient the FK5 position into the Hipparcos system.
    ph = rxp(r5h, pv5[0])

    # Apply spin to the position giving extra space motion.
    wxp = pxp(pv5[0], s5h)

    # Add this component to the FK5 space motion.
    vv = ppp(wxp, pv5[1])

    # Orient the FK5 space motion into the Hipparcos system.
    vh = rxp(r5h, vv)

    pvh = jnp.array([ph, vh])

    # Hipparcos pv-vector to spherical.
    return pvstar(pvh)


def h2fk5(rh, dh, drh, ddh, pxh, rvh):
    """Hipparcos to FK5 (J2000.0) star data.

    Parameters
    ----------
    rh, dh : float
        Hipparcos RA, Dec (radians)
    drh, ddh : float
        Hipparcos proper motions (rad/Jyear)
    pxh : float
        Parallax (arcsec)
    rvh : float
        Radial velocity (km/s, +ve = receding)

    Returns
    -------
    r5, d5, dr5, dd5, px5, rv5 : float
        FK5 equivalents
    """
    from so_pointjax.erfa._core.astrometry import starpv, pvstar

    # Hipparcos barycentric pv-vector.
    pvh, _ = starpv(rh, dh, drh, ddh, pxh, rvh)

    # FK5 to Hipparcos orientation matrix and spin vector.
    r5h, s5h = fk5hip()

    # Make spin units per day instead of per year.
    s5h = s5h / 365.25

    # Orient the spin into the Hipparcos system.
    sh = rxp(r5h, s5h)

    # De-orient the Hipparcos position into FK5.
    p5 = trxp(r5h, pvh[0])

    # Apply spin to the position giving extra space motion.
    wxp = pxp(pvh[0], sh)

    # Subtract from Hipparcos space motion.
    vv = pmp(pvh[1], wxp)

    # De-orient the Hipparcos space motion into FK5.
    v5 = trxp(r5h, vv)

    pv5 = jnp.array([p5, v5])

    return pvstar(pv5)


def fk5hz(r5, d5, date1, date2):
    """FK5 to Hipparcos, assuming zero Hipparcos proper motion.

    Parameters
    ----------
    r5, d5 : float
        FK5 RA, Dec (radians), equinox J2000.0, at date
    date1, date2 : float
        TDB date (2-part JD)

    Returns
    -------
    rh : float
        Hipparcos RA (radians)
    dh : float
        Hipparcos Dec (radians)
    """
    from so_pointjax.erfa._core.constants import DJ00, DJY

    # Interval from given date to fundamental epoch J2000.0 (JY).
    t = -((date1 - DJ00) + date2) / DJY

    # FK5 barycentric position vector.
    p5e = s2c(r5, d5)

    # FK5 to Hipparcos orientation matrix and spin vector.
    r5h, s5h = fk5hip()

    # Accumulated Hipparcos wrt FK5 spin over that interval.
    vst = sxp(t, s5h)

    # Express the accumulated spin as a rotation matrix.
    rst = rv2m(vst)

    # Derotate the vector's FK5 axes back to date.
    p5 = trxp(rst, p5e)

    # Rotate the vector into the Hipparcos system.
    ph = rxp(r5h, p5)

    # Hipparcos vector to spherical.
    w, dh = c2s(ph)
    rh = anp(w)
    return rh, dh


def hfk5z(rh, dh, date1, date2):
    """Hipparcos to FK5, assuming zero Hipparcos proper motion.

    Parameters
    ----------
    rh, dh : float
        Hipparcos RA, Dec (radians)
    date1, date2 : float
        TDB date (2-part JD)

    Returns
    -------
    r5, d5 : float
        FK5 RA, Dec (radians), at date
    dr5, dd5 : float
        FK5 proper motions (rad/year)
    """
    from so_pointjax.erfa._core.constants import DJ00, DJY

    # Time interval from J2000.0 to given date (JY).
    t = ((date1 - DJ00) + date2) / DJY

    # Hipparcos barycentric position vector (normalized).
    ph = s2c(rh, dh)

    # FK5 to Hipparcos orientation matrix and spin vector.
    r5h, s5h = fk5hip()

    # Rotate the spin into the Hipparcos system.
    sh = rxp(r5h, s5h)

    # Accumulated Hipparcos wrt FK5 spin over that interval.
    vst = sxp(t, s5h)

    # Express the accumulated spin as a rotation matrix.
    rst = rv2m(vst)

    # Rotation matrix: accumulated spin, then FK5 to Hipparcos.
    r5ht = rxr(r5h, rst)

    # De-orient & de-spin the Hipparcos position into FK5.
    p5 = trxp(r5ht, ph)

    # Apply spin to the position giving a space motion.
    vv = pxp(sh, ph)

    # De-orient & de-spin the Hipparcos space motion into FK5.
    v5 = trxp(r5ht, vv)

    # FK5 position/velocity pv-vector to spherical.
    pv5e = jnp.array([p5, v5])
    w, d5, r, dr5, dd5, v = pv2s(pv5e)
    r5 = anp(w)

    return r5, d5, dr5, dd5


# ============================================================================
# Star catalog transforms: FK4 <-> FK5
# ============================================================================

# Radians per year to arcsec per century.
_PMF = 100.0 * DR2AS

# Km per sec to au per tropical century.
_VF = 21.095

# E-terms vectors A and Adot (Seidelmann 3.591-2).
_A_ETERM = jnp.array([
    [-1.62557e-6, -0.31919e-6, -0.13843e-6],
    [+1.245e-3, -1.580e-3, -0.659e-3],
])

# FK4->FK5 matrix M (Seidelmann 3.591-4), as [2][3][2][3].
_EM_FK425 = jnp.array([
    [[[+0.9999256782, -0.0111820611, -0.0048579477],
      [+0.00000242395018, -0.00000002710663, -0.00000001177656]],
     [[+0.0111820610, +0.9999374784, -0.0000271765],
      [+0.00000002710663, +0.00000242397878, -0.00000000006587]],
     [[+0.0048579479, -0.0000271474, +0.9999881997],
      [+0.00000001177656, -0.00000000006582, +0.00000242410173]]],
    [[[-0.000551, -0.238565, +0.435739],
      [+0.99994704, -0.01118251, -0.00485767]],
     [[+0.238514, -0.002667, -0.008541],
      [+0.01118251, +0.99995883, -0.00002718]],
     [[-0.435623, +0.012254, +0.002117],
      [+0.00485767, -0.00002714, +1.00000956]]]
])

# FK5->FK4 matrix M^-1 (Seidelmann 3.592-1), as [2][3][2][3].
_EM_FK524 = jnp.array([
    [[[+0.9999256795, +0.0111814828, +0.0048590039],
      [-0.00000242389840, -0.00000002710544, -0.00000001177742]],
     [[-0.0111814828, +0.9999374849, -0.0000271771],
      [+0.00000002710544, -0.00000242392702, +0.00000000006585]],
     [[-0.0048590040, -0.0000271557, +0.9999881946],
      [+0.00000001177742, +0.00000000006585, -0.00000242404995]]],
    [[[-0.000551, +0.238509, -0.435614],
      [+0.99990432, +0.01118145, +0.00485852]],
     [[-0.238560, -0.002667, +0.012254],
      [-0.01118145, +0.99991613, -0.00002717]],
     [[+0.435730, -0.008541, +0.002117],
      [-0.00485852, -0.00002716, +0.99996684]]]
])

# Simplified FK4->FK5 position-only matrix (for fk45z).
_EM_FK45Z = jnp.array([
    [[+0.9999256782, -0.0111820611, -0.0048579477],
     [+0.0111820610, +0.9999374784, -0.0000271765],
     [+0.0048579479, -0.0000271474, +0.9999881997]],
    [[-0.000551, -0.238565, +0.435739],
     [+0.238514, -0.002667, -0.008541],
     [-0.435623, +0.012254, +0.002117]],
])


def _apply_em(em, pv_in):
    """Apply 3x2 pv-vector matrix: pv_out[i][j] = sum_k,l em[i][j][k][l]*pv_in[k][l]."""
    # em shape: (2, 3, 2, 3), pv_in shape: (2, 3)
    return jnp.einsum('ijkl,kl->ij', em, pv_in)


def fk425(r1950, d1950, dr1950, dd1950, p1950, v1950):
    """Convert B1950.0 FK4 to J2000.0 FK5.

    Parameters
    ----------
    r1950, d1950 : float
        B1950.0 RA, Dec (radians)
    dr1950, dd1950 : float
        B1950.0 proper motions (rad/trop.yr)
    p1950 : float
        Parallax (arcsec)
    v1950 : float
        Radial velocity (km/s, +ve = receding)

    Returns
    -------
    r2000, d2000 : float
        J2000.0 RA, Dec (radians)
    dr2000, dd2000 : float
        J2000.0 proper motions (rad/Jul.yr)
    p2000 : float
        Parallax (arcsec)
    v2000 : float
        Radial velocity (km/s)
    """
    TINY = 1e-30

    # FK4 data in working units.
    ur = dr1950 * _PMF
    ud = dd1950 * _PMF
    px = p1950
    rv = v1950

    # Express as a pv-vector.
    pxvf = px * _VF
    w = rv * pxvf
    r0 = s2pv(r1950, d1950, 1.0, ur, ud, w)

    # Allow for E-terms (cf. Seidelmann 3.591-2).
    pv1 = pvmpv(r0, _A_ETERM)
    pv2_pos = sxp(pdp(r0[0], _A_ETERM[0]), r0[0])
    pv2_vel = sxp(pdp(r0[0], _A_ETERM[1]), r0[0])
    pv2 = jnp.array([pv2_pos, pv2_vel])
    pv1 = pvppv(pv1, pv2)

    # Convert pv-vector to Fricke system.
    pv2_out = _apply_em(_EM_FK425, pv1)

    # Revert to catalog form.
    r, d, w_r, ur_out, ud_out, rd = pv2s(pv2_out)
    rv_out = jnp.where(px > TINY, rd / pxvf, rv)
    px_out = jnp.where(px > TINY, px / w_r, px)

    return anp(r), d, ur_out / _PMF, ud_out / _PMF, px_out, rv_out


def fk524(r2000, d2000, dr2000, dd2000, p2000, v2000):
    """Convert J2000.0 FK5 to B1950.0 FK4.

    Parameters
    ----------
    r2000, d2000 : float
        J2000.0 RA, Dec (radians)
    dr2000, dd2000 : float
        J2000.0 proper motions (rad/Jul.yr)
    p2000 : float
        Parallax (arcsec)
    v2000 : float
        Radial velocity (km/s, +ve = receding)

    Returns
    -------
    r1950, d1950 : float
        B1950.0 RA, Dec (radians)
    dr1950, dd1950 : float
        B1950.0 proper motions (rad/trop.yr)
    p1950 : float
        Parallax (arcsec)
    v1950 : float
        Radial velocity (km/s)
    """
    TINY = 1e-30

    ur = dr2000 * _PMF
    ud = dd2000 * _PMF
    px = p2000
    rv = v2000

    pxvf = px * _VF
    w = rv * pxvf
    r0 = s2pv(r2000, d2000, 1.0, ur, ud, w)

    # Convert to Bessel-Newcomb system.
    r1 = _apply_em(_EM_FK524, r0)

    # Apply E-terms (equivalent to Seidelmann 3.592-3, one iteration).
    # Direction.
    w_len = pm(r1[0])
    p1 = pmp(sxp(w_len, _A_ETERM[0]), sxp(pdp(r1[0], _A_ETERM[0]), r1[0]))
    p1 = ppp(r1[0], p1)

    # Recompute length.
    w_len = pm(p1)

    # Direction (second pass).
    p1 = pmp(sxp(w_len, _A_ETERM[0]), sxp(pdp(r1[0], _A_ETERM[0]), r1[0]))
    pv_pos = ppp(r1[0], p1)

    # Derivative.
    p1 = pmp(sxp(w_len, _A_ETERM[1]), sxp(pdp(r1[0], _A_ETERM[1]), pv_pos))
    pv_vel = ppp(r1[1], p1)

    pv = jnp.array([pv_pos, pv_vel])

    # Revert to catalog form.
    r, d, w_r, ur_out, ud_out, rd = pv2s(pv)
    rv_out = jnp.where(px > TINY, rd / pxvf, rv)
    px_out = jnp.where(px > TINY, px / w_r, px)

    return anp(r), d, ur_out / _PMF, ud_out / _PMF, px_out, rv_out


def fk45z(r1950, d1950, bepoch):
    """FK4 (B1950.0) to FK5 (J2000.0), zero FK5 proper motion.

    Parameters
    ----------
    r1950, d1950 : float
        B1950.0 FK4 RA, Dec (radians)
    bepoch : float
        Besselian epoch (e.g. 1979.3)

    Returns
    -------
    r2000 : float
        J2000.0 FK5 RA (radians)
    d2000 : float
        J2000.0 FK5 Dec (radians)
    """
    from so_pointjax.erfa._core.calendar import epb2jd, epj

    # Spherical to Cartesian.
    r0 = s2c(r1950, d1950)

    # Adjust p-vector A to give zero proper motion in FK5.
    w = (bepoch - 1950.0) / _PMF
    p = ppsp(_A_ETERM[0], w, _A_ETERM[1])

    # Remove E-terms.
    p = ppsp(p, -pdp(r0, p), r0)
    p = pmp(r0, p)

    # Convert to Fricke system pv-vector (position-only matrix).
    pv = jnp.zeros((2, 3))
    for i in range(2):
        for j in range(3):
            pv = pv.at[i, j].set(jnp.sum(_EM_FK45Z[i, j] * p))

    # Allow for fictitious proper motion.
    djm0, djm = epb2jd(bepoch)
    w = (epj(djm0, djm) - 2000.0) / _PMF
    pv_upd = pvu(w, pv)

    # Revert to spherical.
    w_ra, d2000 = c2s(pv_upd[0])
    r2000 = anp(w_ra)
    return r2000, d2000


def fk54z(r2000, d2000, bepoch):
    """FK5 (J2000.0) to FK4 (B1950.0), zero FK5 proper motion.

    Parameters
    ----------
    r2000, d2000 : float
        J2000.0 FK5 RA, Dec (radians)
    bepoch : float
        Besselian epoch (e.g. 1950.0)

    Returns
    -------
    r1950, d1950 : float
        B1950.0 FK4 RA, Dec (radians)
    dr1950, dd1950 : float
        B1950.0 FK4 proper motions (rad/trop.yr)
    """
    # FK5 to FK4 with zero proper motion / parallax / RV.
    r, d, pr, pd, px, rv = fk524(r2000, d2000, 0.0, 0.0, 0.0, 0.0)

    # Spherical to Cartesian.
    p = s2c(r, d)

    # Fictitious proper motion (radians per year).
    v0 = -pr * p[1] - pd * jnp.cos(r) * jnp.sin(d)
    v1 = pr * p[0] - pd * jnp.sin(r) * jnp.sin(d)
    v2 = pd * jnp.cos(d)

    # Apply the motion.
    w = bepoch - 1950.0
    p = p + w * jnp.array([v0, v1, v2])

    # Cartesian to spherical.
    w_ra, d1950 = c2s(p)
    r1950 = anp(w_ra)

    return r1950, d1950, pr, pd


__all__ = [
    "ae2hd", "hd2ae", "hd2pa",
    "icrs2g", "g2icrs",
    "ecm06", "eqec06", "eceq06",
    "ltpecl", "ltpequ", "ltecm", "ltp", "ltpb", "lteqec",
    "fk5hip", "fk52h", "h2fk5", "fk5hz", "hfk5z",
    "fk425", "fk524", "fk45z", "fk54z",
]
