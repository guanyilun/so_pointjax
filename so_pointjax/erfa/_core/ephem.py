"""Ephemeris functions, ported from ERFA C library.

The epv00 coefficients are parsed from the original C source file at runtime.
"""

import os
import re

import jax.numpy as jnp

from so_pointjax.erfa._core.constants import DJ00, DJY, DJC, DJM, DD2R, DAS2R, DAU, D2PI


# ============================================================================
# Lazy-loaded coefficient tables for epv00
# ============================================================================

_EPV00_TABLES = None


def _load_epv00_tables():
    """Parse epv00 coefficient arrays from the C source file."""
    global _EPV00_TABLES
    if _EPV00_TABLES is not None:
        return _EPV00_TABLES

    src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'erfa', 'src', 'epv00.c')
    with open(src_path, 'r') as f:
        src = f.read()

    # Parse all 18 coefficient arrays
    array_names = [
        'e0x', 'e0y', 'e0z',
        'e1x', 'e1y', 'e1z',
        'e2x', 'e2y', 'e2z',
        's0x', 's0y', 's0z',
        's1x', 's1y', 's1z',
        's2x', 's2y', 's2z',
    ]

    tables = {}
    for name in array_names:
        # Match: static const double name[] = { ... };
        pattern = r'static\s+const\s+double\s+' + name + r'\[\]\s*=\s*\{([^}]+)\}'
        m = re.search(pattern, src)
        if m is None:
            raise RuntimeError(f"Could not find array {name} in epv00.c")
        body = m.group(1)
        # Extract all numbers
        nums = re.findall(r'[-+]?\d+\.\d+[eE][-+]?\d+|[-+]?\d+\.\d+', body)
        coeffs = [float(x) for x in nums]
        # Reshape into (n, 3) for (amplitude, phase, frequency) triplets
        n = len(coeffs) // 3
        tables[name] = jnp.array(coeffs).reshape(n, 3)

    _EPV00_TABLES = tables
    return tables


# Orientation matrix elements (ecliptic to BCRS)
_AM12 = 0.000000211284
_AM13 = -0.000000091603
_AM21 = -0.000000230286
_AM22 = 0.917482137087
_AM23 = -0.397776982902
_AM32 = 0.397776982902
_AM33 = 0.917482137087


def _eval_series_t0(coeffs, t):
    """Evaluate T^0 series: sum(a * cos(b + c*t)) and derivative."""
    a = coeffs[:, 0]
    b = coeffs[:, 1]
    c = coeffs[:, 2]
    p = b + c * t
    xyz = jnp.sum(a * jnp.cos(p))
    xyzd = -jnp.sum(a * c * jnp.sin(p))
    return xyz, xyzd


def _eval_series_t1(coeffs, t):
    """Evaluate T^1 series: sum(a * t * cos(b + c*t)) and derivative."""
    a = coeffs[:, 0]
    b = coeffs[:, 1]
    c = coeffs[:, 2]
    ct = c * t
    p = b + ct
    cp = jnp.cos(p)
    xyz = jnp.sum(a * t * cp)
    xyzd = jnp.sum(a * (cp - ct * jnp.sin(p)))
    return xyz, xyzd


def _eval_series_t2(coeffs, t):
    """Evaluate T^2 series: sum(a * t^2 * cos(b + c*t)) and derivative."""
    a = coeffs[:, 0]
    b = coeffs[:, 1]
    c = coeffs[:, 2]
    t2 = t * t
    ct = c * t
    p = b + ct
    cp = jnp.cos(p)
    xyz = jnp.sum(a * t2 * cp)
    xyzd = jnp.sum(a * t * (2.0 * cp - ct * jnp.sin(p)))
    return xyz, xyzd


def _rotate_ecliptic_to_bcrs(x, y, z):
    """Rotate from ecliptic to BCRS coordinates."""
    return (
        x + _AM12 * y + _AM13 * z,
        _AM21 * x + _AM22 * y + _AM23 * z,
        _AM32 * y + _AM33 * z,
    )


def epv00(date1, date2):
    """Earth position and velocity, heliocentric and barycentric, BCRS.

    Parameters
    ----------
    date1, date2 : float
        TDB date as a 2-part Julian Date

    Returns
    -------
    pvh : ndarray (2, 3)
        Heliocentric Earth position/velocity (au, au/day)
    pvb : ndarray (2, 3)
        Barycentric Earth position/velocity (au, au/day)
    """
    tables = _load_epv00_tables()

    # Time since reference epoch, Julian years
    t = ((date1 - DJ00) + date2) / DJY

    # Component names for x, y, z
    components = ['x', 'y', 'z']

    ph = jnp.zeros(3)
    vh = jnp.zeros(3)
    pb = jnp.zeros(3)
    vb = jnp.zeros(3)

    for i, comp in enumerate(components):
        xyz = 0.0
        xyzd = 0.0

        # Sun-to-Earth T^0
        v0, d0 = _eval_series_t0(tables[f'e0{comp}'], t)
        xyz += v0
        xyzd += d0

        # Sun-to-Earth T^1
        v1, d1 = _eval_series_t1(tables[f'e1{comp}'], t)
        xyz += v1
        xyzd += d1

        # Sun-to-Earth T^2
        v2, d2 = _eval_series_t2(tables[f'e2{comp}'], t)
        xyz += v2
        xyzd += d2

        # Heliocentric Earth position and velocity component
        ph = ph.at[i].set(xyz)
        vh = vh.at[i].set(xyzd / DJY)

        # SSB-to-Sun T^0
        v0s, d0s = _eval_series_t0(tables[f's0{comp}'], t)
        xyz += v0s
        xyzd += d0s

        # SSB-to-Sun T^1
        v1s, d1s = _eval_series_t1(tables[f's1{comp}'], t)
        xyz += v1s
        xyzd += d1s

        # SSB-to-Sun T^2
        v2s, d2s = _eval_series_t2(tables[f's2{comp}'], t)
        xyz += v2s
        xyzd += d2s

        # Barycentric Earth position and velocity component
        pb = pb.at[i].set(xyz)
        vb = vb.at[i].set(xyzd / DJY)

    # Rotate from ecliptic to BCRS coordinates
    pvh_px, pvh_py, pvh_pz = _rotate_ecliptic_to_bcrs(ph[0], ph[1], ph[2])
    pvh_vx, pvh_vy, pvh_vz = _rotate_ecliptic_to_bcrs(vh[0], vh[1], vh[2])
    pvb_px, pvb_py, pvb_pz = _rotate_ecliptic_to_bcrs(pb[0], pb[1], pb[2])
    pvb_vx, pvb_vy, pvb_vz = _rotate_ecliptic_to_bcrs(vb[0], vb[1], vb[2])

    pvh = jnp.array([[pvh_px, pvh_py, pvh_pz],
                      [pvh_vx, pvh_vy, pvh_vz]])
    pvb = jnp.array([[pvb_px, pvb_py, pvb_pz],
                      [pvb_vx, pvb_vy, pvb_vz]])

    return pvh, pvb


# ============================================================================
# moon98 — Approximate geocentric position and velocity of the Moon
# ============================================================================

# Fundamental argument coefficients (degrees): [c0, c1, c2, c3, c4]
# Moon's mean longitude
_ELP = jnp.array([218.31665436, 481267.88123421, -0.0015786,
                   1.0 / 538841.0, -1.0 / 65194000.0])
# Moon's mean elongation
_D = jnp.array([297.8501921, 445267.1114034, -0.0018819,
                1.0 / 545868.0, 1.0 / 113065000.0])
# Sun's mean anomaly
_EM = jnp.array([357.5291092, 35999.0502909, -0.0001536,
                 1.0 / 24490000.0, 0.0])
# Moon's mean anomaly
_EMP = jnp.array([134.9633964, 477198.8675055, 0.0087414,
                  1.0 / 69699.0, -1.0 / 14712000.0])
# Mean distance of the Moon from its ascending node
_F = jnp.array([93.2720950, 483202.0175233, -0.0036539,
                1.0 / 3526000.0, 1.0 / 863310000.0])

# Meeus further arguments: A1 (Venus), A2 (Jupiter), A3 (sidereal)
_A1 = jnp.array([119.75, 131.849])
_A2 = jnp.array([53.09, 479264.290])
_A3 = jnp.array([313.45, 481266.484])

# Additive term coefficients
_AL1, _AL2, _AL3 = 0.003958, 0.001962, 0.000318
_AB1, _AB2, _AB3 = -0.002235, 0.000382, 0.000175
_AB4, _AB5, _AB6 = 0.000175, 0.000127, -0.000115

# Fixed term in distance (m)
_R0 = 385000560.0

# E-factor coefficients
_E1, _E2 = -0.002516, -0.0000074

# Longitude/distance series: (nd, nem, nemp, nf, coefl, coefr)
_TLR = jnp.array([
    [0,  0,  1,  0,  6.288774, -20905355.0],
    [2,  0, -1,  0,  1.274027,  -3699111.0],
    [2,  0,  0,  0,  0.658314,  -2955968.0],
    [0,  0,  2,  0,  0.213618,   -569925.0],
    [0,  1,  0,  0, -0.185116,     48888.0],
    [0,  0,  0,  2, -0.114332,     -3149.0],
    [2,  0, -2,  0,  0.058793,    246158.0],
    [2, -1, -1,  0,  0.057066,   -152138.0],
    [2,  0,  1,  0,  0.053322,   -170733.0],
    [2, -1,  0,  0,  0.045758,   -204586.0],
    [0,  1, -1,  0, -0.040923,   -129620.0],
    [1,  0,  0,  0, -0.034720,    108743.0],
    [0,  1,  1,  0, -0.030383,    104755.0],
    [2,  0,  0, -2,  0.015327,     10321.0],
    [0,  0,  1,  2, -0.012528,         0.0],
    [0,  0,  1, -2,  0.010980,     79661.0],
    [4,  0, -1,  0,  0.010675,    -34782.0],
    [0,  0,  3,  0,  0.010034,    -23210.0],
    [4,  0, -2,  0,  0.008548,    -21636.0],
    [2,  1, -1,  0, -0.007888,     24208.0],
    [2,  1,  0,  0, -0.006766,     30824.0],
    [1,  0, -1,  0, -0.005163,     -8379.0],
    [1,  1,  0,  0,  0.004987,    -16675.0],
    [2, -1,  1,  0,  0.004036,    -12831.0],
    [2,  0,  2,  0,  0.003994,    -10445.0],
    [4,  0,  0,  0,  0.003861,    -11650.0],
    [2,  0, -3,  0,  0.003665,     14403.0],
    [0,  1, -2,  0, -0.002689,     -7003.0],
    [2,  0, -1,  2, -0.002602,         0.0],
    [2, -1, -2,  0,  0.002390,     10056.0],
    [1,  0,  1,  0, -0.002348,      6322.0],
    [2, -2,  0,  0,  0.002236,     -9884.0],
    [0,  1,  2,  0, -0.002120,      5751.0],
    [0,  2,  0,  0, -0.002069,         0.0],
    [2, -2, -1,  0,  0.002048,     -4950.0],
    [2,  0,  1, -2, -0.001773,      4130.0],
    [2,  0,  0,  2, -0.001595,         0.0],
    [4, -1, -1,  0,  0.001215,     -3958.0],
    [0,  0,  2,  2, -0.001110,         0.0],
    [3,  0, -1,  0, -0.000892,      3258.0],
    [2,  1,  1,  0, -0.000810,      2616.0],
    [4, -1, -2,  0,  0.000759,     -1897.0],
    [0,  2, -1,  0, -0.000713,     -2117.0],
    [2,  2, -1,  0, -0.000700,      2354.0],
    [2,  1, -2,  0,  0.000691,         0.0],
    [2, -1,  0, -2,  0.000596,         0.0],
    [4,  0,  1,  0,  0.000549,     -1423.0],
    [0,  0,  4,  0,  0.000537,     -1117.0],
    [4, -1,  0,  0,  0.000520,     -1571.0],
    [1,  0, -2,  0, -0.000487,     -1739.0],
    [2,  1,  0, -2, -0.000399,         0.0],
    [0,  0,  2, -2, -0.000381,     -4421.0],
    [1,  1,  1,  0,  0.000351,         0.0],
    [3,  0, -2,  0, -0.000340,         0.0],
    [4,  0, -3,  0,  0.000330,         0.0],
    [2, -1,  2,  0,  0.000327,         0.0],
    [0,  2,  1,  0, -0.000323,      1165.0],
    [1,  1, -1,  0,  0.000299,         0.0],
    [2,  0,  3,  0,  0.000294,         0.0],
    [2,  0, -1, -2,  0.000000,      8752.0],
])

# Latitude series: (nd, nem, nemp, nf, coefb)
_TB = jnp.array([
    [0,  0,  0,  1,  5.128122],
    [0,  0,  1,  1,  0.280602],
    [0,  0,  1, -1,  0.277693],
    [2,  0,  0, -1,  0.173237],
    [2,  0, -1,  1,  0.055413],
    [2,  0, -1, -1,  0.046271],
    [2,  0,  0,  1,  0.032573],
    [0,  0,  2,  1,  0.017198],
    [2,  0,  1, -1,  0.009266],
    [0,  0,  2, -1,  0.008822],
    [2, -1,  0, -1,  0.008216],
    [2,  0, -2, -1,  0.004324],
    [2,  0,  1,  1,  0.004200],
    [2,  1,  0, -1, -0.003359],
    [2, -1, -1,  1,  0.002463],
    [2, -1,  0,  1,  0.002211],
    [2, -1, -1, -1,  0.002065],
    [0,  1, -1, -1, -0.001870],
    [4,  0, -1, -1,  0.001828],
    [0,  1,  0,  1, -0.001794],
    [0,  0,  0,  3, -0.001749],
    [0,  1, -1,  1, -0.001565],
    [1,  0,  0,  1, -0.001491],
    [0,  1,  1,  1, -0.001475],
    [0,  1,  1, -1, -0.001410],
    [0,  1,  0, -1, -0.001344],
    [1,  0,  0, -1, -0.001335],
    [0,  0,  3,  1,  0.001107],
    [4,  0,  0, -1,  0.001021],
    [4,  0, -1,  1,  0.000833],
    [0,  0,  1, -3,  0.000777],
    [4,  0, -2,  1,  0.000671],
    [2,  0,  0, -3,  0.000607],
    [2,  0,  2, -1,  0.000596],
    [2, -1,  1, -1,  0.000491],
    [2,  0, -2,  1, -0.000451],
    [0,  0,  3, -1,  0.000439],
    [2,  0,  2,  1,  0.000422],
    [2,  0, -3, -1,  0.000421],
    [2,  1, -1,  1, -0.000366],
    [2,  1,  0,  1, -0.000351],
    [4,  0,  0,  1,  0.000331],
    [2, -1,  1,  1,  0.000315],
    [2, -2,  0, -1,  0.000302],
    [0,  0,  1,  3, -0.000283],
    [2,  1,  1, -1, -0.000229],
    [1,  1,  0, -1,  0.000223],
    [1,  1,  0,  1,  0.000223],
    [0,  1, -2, -1, -0.000220],
    [2,  1, -1, -1, -0.000220],
    [1,  0,  1,  1, -0.000185],
    [2, -1, -2, -1,  0.000181],
    [0,  1,  2,  1, -0.000177],
    [4,  0, -2, -1,  0.000176],
    [4, -1, -1, -1,  0.000166],
    [1,  0,  1, -1, -0.000164],
    [4,  0,  1, -1,  0.000132],
    [1,  0, -1, -1, -0.000119],
    [4, -1,  0, -1,  0.000115],
    [2, -2,  0,  1,  0.000107],
])


def _eval_poly4(coeffs, t):
    """Evaluate a 4th-degree polynomial and its derivative in degrees."""
    c0, c1, c2, c3, c4 = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]
    val = c0 + (c1 + (c2 + (c3 + c4 * t) * t) * t) * t
    dval = c1 + (2.0 * c2 + (3.0 * c3 + 4.0 * c4 * t) * t) * t
    return val, dval


def moon98(date1, date2):
    """Approximate geocentric position and velocity of the Moon.

    Parameters
    ----------
    date1, date2 : float
        TT date as a 2-part Julian Date.

    Returns
    -------
    pv : ndarray (2, 3)
        Moon position (au) and velocity (au/day), GCRS.
    """
    from so_pointjax.erfa._core.vector import s2pv, rx, rz, rxpv, ir
    from so_pointjax.erfa._core.precnut import pfw06

    t = ((date1 - DJ00) + date2) / DJC

    # Fundamental arguments (radians) and derivatives (rad/century)
    elp_deg, delp_deg = _eval_poly4(_ELP, t)
    elp = DD2R * (elp_deg % 360.0)
    delp = DD2R * delp_deg

    d_deg, dd_deg = _eval_poly4(_D, t)
    d = DD2R * (d_deg % 360.0)
    dd = DD2R * dd_deg

    em_deg, dem_deg = _eval_poly4(_EM, t)
    em = DD2R * (em_deg % 360.0)
    dem = DD2R * dem_deg

    emp_deg, demp_deg = _eval_poly4(_EMP, t)
    emp = DD2R * (emp_deg % 360.0)
    demp = DD2R * demp_deg

    f_deg, df_deg = _eval_poly4(_F, t)
    f = DD2R * (f_deg % 360.0)
    df = DD2R * df_deg

    # Meeus further arguments
    a1 = DD2R * (_A1[0] + _A1[1] * t)
    da1 = DD2R * _A1[1]
    a2 = DD2R * (_A2[0] + _A2[1] * t)
    da2 = DD2R * _A2[1]
    a3 = DD2R * (_A3[0] + _A3[1] * t)
    da3 = DD2R * _A3[1]

    # E-factor and square
    e = 1.0 + (_E1 + _E2 * t) * t
    de = _E1 + 2.0 * _E2 * t
    esq = e * e
    desq = 2.0 * e * de

    # Additive terms for longitude
    elpmf = elp - f
    delpmf = delp - df
    vel = _AL1 * jnp.sin(a1) + _AL2 * jnp.sin(elpmf) + _AL3 * jnp.sin(a2)
    vdel = (_AL1 * jnp.cos(a1) * da1 + _AL2 * jnp.cos(elpmf) * delpmf
            + _AL3 * jnp.cos(a2) * da2)

    vr = 0.0
    vdr = 0.0

    # Additive terms for latitude
    a1mf = a1 - f
    da1mf = da1 - df
    a1pf = a1 + f
    da1pf = da1 + df
    dlpmp = elp - emp
    slpmp = elp + emp
    vb = (_AB1 * jnp.sin(elp) + _AB2 * jnp.sin(a3)
          + _AB3 * jnp.sin(a1mf) + _AB4 * jnp.sin(a1pf)
          + _AB5 * jnp.sin(dlpmp) + _AB6 * jnp.sin(slpmp))
    vdb = (_AB1 * jnp.cos(elp) * delp + _AB2 * jnp.cos(a3) * da3
           + _AB3 * jnp.cos(a1mf) * da1mf + _AB4 * jnp.cos(a1pf) * da1pf
           + _AB5 * jnp.cos(dlpmp) * (delp - demp)
           + _AB6 * jnp.cos(slpmp) * (delp + demp))

    # Longitude/distance series
    nd = _TLR[:, 0]
    nem = _TLR[:, 1]
    nemp = _TLR[:, 2]
    nf = _TLR[:, 3]
    coefl = _TLR[:, 4]
    coefr = _TLR[:, 5]

    abs_nem = jnp.abs(nem)
    en = jnp.where(abs_nem == 1, e, jnp.where(abs_nem == 2, esq, 1.0))
    den = jnp.where(abs_nem == 1, de, jnp.where(abs_nem == 2, desq, 0.0))

    arg = nd * d + nem * em + nemp * emp + nf * f
    darg = nd * dd + nem * dem + nemp * demp + nf * df

    sin_arg = jnp.sin(arg)
    cos_arg = jnp.cos(arg)

    # Longitude: sum(coefl * sin(arg) * en)
    v_l = sin_arg * en
    dv_l = cos_arg * darg * en + sin_arg * den
    vel = vel + jnp.sum(coefl * v_l)
    vdel = vdel + jnp.sum(coefl * dv_l)

    # Distance: sum(coefr * cos(arg) * en)
    v_r = cos_arg * en
    dv_r = -sin_arg * darg * en + cos_arg * den
    vr = vr + jnp.sum(coefr * v_r)
    vdr = vdr + jnp.sum(coefr * dv_r)

    el = elp + DD2R * vel
    dl = (delp + DD2R * vdel) / DJC
    r = (vr + _R0) / DAU
    dr = vdr / DAU / DJC

    # Latitude series
    nd_b = _TB[:, 0]
    nem_b = _TB[:, 1]
    nemp_b = _TB[:, 2]
    nf_b = _TB[:, 3]
    coefb = _TB[:, 4]

    abs_nem_b = jnp.abs(nem_b)
    en_b = jnp.where(abs_nem_b == 1, e, jnp.where(abs_nem_b == 2, esq, 1.0))
    den_b = jnp.where(abs_nem_b == 1, de, jnp.where(abs_nem_b == 2, desq, 0.0))

    arg_b = nd_b * d + nem_b * em + nemp_b * emp + nf_b * f
    darg_b = nd_b * dd + nem_b * dem + nemp_b * demp + nf_b * df

    sin_arg_b = jnp.sin(arg_b)
    cos_arg_b = jnp.cos(arg_b)

    v_b = sin_arg_b * en_b
    dv_b = cos_arg_b * darg_b * en_b + sin_arg_b * den_b
    vb = vb + jnp.sum(coefb * v_b)
    vdb = vdb + jnp.sum(coefb * dv_b)

    b = vb * DD2R
    db = vdb * DD2R / DJC

    # Spherical to Cartesian
    pv = s2pv(el, b, r, dl, db, dr)

    # IAU 2006 bias+precession angles -> mean ecliptic to GCRS rotation
    gamb, phib, psib, epsa = pfw06(date1, date2)
    rm = ir()
    rm = rz(psib, rm)
    rm = rx(-phib, rm)
    rm = rz(-gamb, rm)

    # Rotate Moon position and velocity into GCRS
    pv = rxpv(rm, pv)

    return pv


# ============================================================================
# plan94 — Approximate heliocentric position and velocity of planets
# ============================================================================

# Gaussian constant
_GK = 0.017202098950

# Sin and cos of J2000.0 mean obliquity (IAU 1976)
_SINEPS = 0.3977771559319137
_COSEPS = 0.9174820620691818

# Maximum iterations for Kepler's equation
_KMAX = 10

# Planetary inverse masses
_AMAS = jnp.array([6023600.0, 408523.5, 328900.5, 3098710.0,
                    1047.355, 3498.5, 22869.0, 19314.0])

# Mean Keplerian elements [8 planets, 3 coefficients]
_A_PLAN = jnp.array([
    [0.3870983098, 0.0, 0.0],
    [0.7233298200, 0.0, 0.0],
    [1.0000010178, 0.0, 0.0],
    [1.5236793419, 3e-10, 0.0],
    [5.2026032092, 19132e-10, -39e-10],
    [9.5549091915, -0.0000213896, 444e-10],
    [19.2184460618, -3716e-10, 979e-10],
    [30.1103868694, -16635e-10, 686e-10],
])

_DLM = jnp.array([
    [252.25090552, 5381016286.88982, -1.92789],
    [181.97980085, 2106641364.33548, 0.59381],
    [100.46645683, 1295977422.83429, -2.04411],
    [355.43299958, 689050774.93988, 0.94264],
    [34.35151874, 109256603.77991, -30.60378],
    [50.07744430, 43996098.55732, 75.61614],
    [314.05500511, 15424811.93933, -1.75083],
    [304.34866548, 7865503.20744, 0.21103],
])

_E_PLAN = jnp.array([
    [0.2056317526, 0.0002040653, -28349e-10],
    [0.0067719164, -0.0004776521, 98127e-10],
    [0.0167086342, -0.0004203654, -0.0000126734],
    [0.0934006477, 0.0009048438, -80641e-10],
    [0.0484979255, 0.0016322542, -0.0000471366],
    [0.0555481426, -0.0034664062, -0.0000643639],
    [0.0463812221, -0.0002729293, 0.0000078913],
    [0.0094557470, 0.0000603263, 0.0],
])

_PI_PLAN = jnp.array([
    [77.45611904, 5719.11590, -4.83016],
    [131.56370300, 175.48640, -498.48184],
    [102.93734808, 11612.35290, 53.27577],
    [336.06023395, 15980.45908, -62.32800],
    [14.33120687, 7758.75163, 259.95938],
    [93.05723748, 20395.49439, 190.25952],
    [173.00529106, 3215.56238, -34.09288],
    [48.12027554, 1050.71912, 27.39717],
])

_DINC = jnp.array([
    [7.00498625, -214.25629, 0.28977],
    [3.39466189, -30.84437, -11.67836],
    [0.0, 469.97289, -3.35053],
    [1.84972648, -293.31722, -8.11830],
    [1.30326698, -71.55890, 11.95297],
    [2.48887878, 91.85195, -17.66225],
    [0.77319689, -60.72723, 1.25759],
    [1.76995259, 8.12333, 0.08135],
])

_OMEGA = jnp.array([
    [48.33089304, -4515.21727, -31.79892],
    [76.67992019, -10008.48154, -51.32614],
    [174.87317577, -8679.27034, 15.34191],
    [49.55809321, -10620.90088, -230.57416],
    [100.46440702, 6362.03561, 326.52178],
    [113.66550252, -9240.19942, -66.23743],
    [74.00595701, 2669.15033, 145.93964],
    [131.78405702, -221.94322, -0.78728],
])

# Trigonometric terms for semi-major axes
_KP = jnp.array([
    [69613, 75645, 88306, 59899, 15746, 71087, 142173, 3086, 0],
    [21863, 32794, 26934, 10931, 26250, 43725, 53867, 28939, 0],
    [16002, 21863, 32004, 10931, 14529, 16368, 15318, 32794, 0],
    [6345, 7818, 15636, 7077, 8184, 14163, 1107, 4872, 0],
    [1760, 1454, 1167, 880, 287, 2640, 19, 2047, 1454],
    [574, 0, 880, 287, 19, 1760, 1167, 306, 574],
    [204, 0, 177, 1265, 4, 385, 200, 208, 204],
    [0, 102, 106, 4, 98, 1367, 487, 204, 0],
], dtype=jnp.float64)

_CA = jnp.array([
    [4, -13, 11, -9, -9, -3, -1, 4, 0],
    [-156, 59, -42, 6, 19, -20, -10, -12, 0],
    [64, -152, 62, -8, 32, -41, 19, -11, 0],
    [124, 621, -145, 208, 54, -57, 30, 15, 0],
    [-23437, -2634, 6601, 6259, -1507, -1821, 2620, -2115, -1489],
    [62911, -119919, 79336, 17814, -24241, 12068, 8306, -4893, 8902],
    [389061, -262125, -44088, 8387, -22976, -2093, -615, -9720, 6633],
    [-412235, -157046, -31430, 37817, -9740, -13, -7449, 9644, 0],
], dtype=jnp.float64)

_SA = jnp.array([
    [-29, -1, 9, 6, -6, 5, 4, 0, 0],
    [-48, -125, -26, -37, 18, -13, -20, -2, 0],
    [-150, -46, 68, 54, 14, 24, -28, 22, 0],
    [-621, 532, -694, -20, 192, -94, 71, -73, 0],
    [-14614, -19828, -5869, 1881, -4372, -2255, 782, 930, 913],
    [139737, 0, 24667, 51123, -5102, 7429, -4095, -1976, -9566],
    [-138081, 0, 37205, -49039, -41901, -33872, -27037, -12474, 18797],
    [0, 28492, 133236, 69654, 52322, -49577, -26430, -3593, 0],
], dtype=jnp.float64)

# Trigonometric terms for mean longitudes
_KQ = jnp.array([
    [3086, 15746, 69613, 59899, 75645, 88306, 12661, 2658, 0, 0],
    [21863, 32794, 10931, 73, 4387, 26934, 1473, 2157, 0, 0],
    [10, 16002, 21863, 10931, 1473, 32004, 4387, 73, 0, 0],
    [10, 6345, 7818, 1107, 15636, 7077, 8184, 532, 10, 0],
    [19, 1760, 1454, 287, 1167, 880, 574, 2640, 19, 1454],
    [19, 574, 287, 306, 1760, 12, 31, 38, 19, 574],
    [4, 204, 177, 8, 31, 200, 1265, 102, 4, 204],
    [4, 102, 106, 8, 98, 1367, 487, 204, 4, 102],
], dtype=jnp.float64)

_CL = jnp.array([
    [21, -95, -157, 41, -5, 42, 23, 30, 0, 0],
    [-160, -313, -235, 60, -74, -76, -27, 34, 0, 0],
    [-325, -322, -79, 232, -52, 97, 55, -41, 0, 0],
    [2268, -979, 802, 602, -668, -33, 345, 201, -55, 0],
    [7610, -4997, -7689, -5841, -2617, 1115, -748, -607, 6074, 354],
    [-18549, 30125, 20012, -730, 824, 23, 1289, -352, -14767, -2062],
    [-135245, -14594, 4197, -4030, -5630, -2898, 2540, -306, 2939, 1986],
    [89948, 2103, 8963, 2695, 3682, 1648, 866, -154, -1963, -283],
], dtype=jnp.float64)

_SL = jnp.array([
    [-342, 136, -23, 62, 66, -52, -33, 17, 0, 0],
    [524, -149, -35, 117, 151, 122, -71, -62, 0, 0],
    [-105, -137, 258, 35, -116, -88, -112, -80, 0, 0],
    [854, -205, -936, -240, 140, -341, -97, -232, 536, 0],
    [-56980, 8016, 1012, 1448, -3024, -3710, 318, 503, 3767, 577],
    [138606, -13478, -4964, 1441, -1319, -1482, 427, 1236, -9167, -1918],
    [71234, -41116, 5334, -4935, -1848, 66, 434, -1748, 3780, -701],
    [-47645, 11647, 2166, 3194, 679, 0, -244, -419, -2531, 48],
], dtype=jnp.float64)


def _plan94_one(date1, date2, npi):
    """Compute plan94 for a single (valid, 0-indexed) planet index.

    This is the core computation, designed to be JIT-compatible.
    npi is 0-indexed (0=Mercury .. 7=Neptune).
    """
    from so_pointjax.erfa._core.angles import anpm

    t = ((date1 - DJ00) + date2) / DJM

    # Mean elements
    da = _A_PLAN[npi, 0] + (_A_PLAN[npi, 1] + _A_PLAN[npi, 2] * t) * t
    dl = (3600.0 * _DLM[npi, 0] + (_DLM[npi, 1] + _DLM[npi, 2] * t) * t) * DAS2R
    de = _E_PLAN[npi, 0] + (_E_PLAN[npi, 1] + _E_PLAN[npi, 2] * t) * t
    dp = anpm((3600.0 * _PI_PLAN[npi, 0] +
               (_PI_PLAN[npi, 1] + _PI_PLAN[npi, 2] * t) * t) * DAS2R)
    di = (3600.0 * _DINC[npi, 0] + (_DINC[npi, 1] + _DINC[npi, 2] * t) * t) * DAS2R
    dom = anpm((3600.0 * _OMEGA[npi, 0] +
                (_OMEGA[npi, 1] + _OMEGA[npi, 2] * t) * t) * DAS2R)

    # Trigonometric terms for semi-major axis
    dmu = 0.35953620 * t
    kp_row = _KP[npi]
    ca_row = _CA[npi]
    sa_row = _SA[npi]

    # First 8 terms for semi-major axis
    arga_8 = kp_row[:8] * dmu
    da = da + jnp.sum((ca_row[:8] * jnp.cos(arga_8) +
                        sa_row[:8] * jnp.sin(arga_8)) * 1e-7)

    # 9th term (index 8) multiplied by t
    arga_9 = kp_row[8] * dmu
    da = da + t * (ca_row[8] * jnp.cos(arga_9) +
                   sa_row[8] * jnp.sin(arga_9)) * 1e-7

    # Trigonometric terms for mean longitude
    kq_row = _KQ[npi]
    cl_row = _CL[npi]
    sl_row = _SL[npi]

    # First 8 terms
    argl_8 = kq_row[:8] * dmu
    dl = dl + jnp.sum((cl_row[:8] * jnp.cos(argl_8) +
                        sl_row[:8] * jnp.sin(argl_8)) * 1e-7)

    # Terms 9-10 (indices 8,9) multiplied by t
    argl_910 = kq_row[8:10] * dmu
    dl = dl + t * jnp.sum((cl_row[8:10] * jnp.cos(argl_910) +
                            sl_row[8:10] * jnp.sin(argl_910)) * 1e-7)

    dl = dl % D2PI

    # Kepler's equation: iterative solution
    am = dl - dp
    ae = am + de * jnp.sin(am)

    def kepler_step(carry, _):
        ae_prev, _ = carry
        dae = (am - ae_prev + de * jnp.sin(ae_prev)) / (1.0 - de * jnp.cos(ae_prev))
        ae_new = ae_prev + dae
        return (ae_new, dae), None

    import jax.lax as lax
    (ae, dae), _ = lax.scan(kepler_step, (ae, 1.0), None, length=_KMAX)

    # True anomaly
    ae2 = ae / 2.0
    at = 2.0 * jnp.arctan2(jnp.sqrt((1.0 + de) / (1.0 - de)) * jnp.sin(ae2),
                            jnp.cos(ae2))

    # Distance and speed
    r = da * (1.0 - de * jnp.cos(ae))
    v = _GK * jnp.sqrt((1.0 + 1.0 / _AMAS[npi]) / (da * da * da))

    si2 = jnp.sin(di / 2.0)
    xq = si2 * jnp.cos(dom)
    xp = si2 * jnp.sin(dom)
    tl = at + dp
    xsw = jnp.sin(tl)
    xcw = jnp.cos(tl)
    xm2 = 2.0 * (xp * xcw - xq * xsw)
    xf = da / jnp.sqrt(1.0 - de * de)
    ci2 = jnp.cos(di / 2.0)
    xms = (de * jnp.sin(dp) + xsw) * xf
    xmc = (de * jnp.cos(dp) + xcw) * xf
    xpxq2 = 2.0 * xp * xq

    # Position (ecliptic)
    x = r * (xcw - xm2 * xp)
    y = r * (xsw + xm2 * xq)
    z = r * (-xm2 * ci2)

    # Rotate to equatorial
    px = x
    py = y * _COSEPS - z * _SINEPS
    pz = y * _SINEPS + z * _COSEPS

    # Velocity (ecliptic)
    x = v * ((-1.0 + 2.0 * xp * xp) * xms + xpxq2 * xmc)
    y = v * ((1.0 - 2.0 * xq * xq) * xmc - xpxq2 * xms)
    z = v * (2.0 * ci2 * (xp * xms + xq * xmc))

    # Rotate to equatorial
    vx = x
    vy = y * _COSEPS - z * _SINEPS
    vz = y * _SINEPS + z * _COSEPS

    pv = jnp.array([[px, py, pz], [vx, vy, vz]])

    # Status: 0 if |t| <= 1, else 1 (year warning)
    jstat = jnp.where(jnp.abs(t) <= 1.0, 0, 1)

    # Check convergence (if dae still large after KMAX iterations)
    jstat = jnp.where(jnp.abs(dae) > 1e-12, 2, jstat)

    return pv, jstat


def plan94(date1, date2, np_planet):
    """Approximate heliocentric position and velocity of a planet.

    Parameters
    ----------
    date1, date2 : float
        TDB date as a 2-part Julian Date.
    np_planet : int
        Planet number (1=Mercury, 2=Venus, 3=EMB, 4=Mars,
        5=Jupiter, 6=Saturn, 7=Uranus, 8=Neptune).

    Returns
    -------
    pv : ndarray (2, 3)
        Planet position (au) and velocity (au/day), heliocentric J2000.0.
    j : int
        Status: -1 = illegal np, 0 = OK, +1 = year warning, +2 = convergence.
    """
    # Compute the result for valid planet (clamp index to valid range for tracing)
    npi = jnp.clip(np_planet - 1, 0, 7)
    pv_valid, jstat_valid = _plan94_one(date1, date2, npi)

    # Check for invalid planet number
    is_valid = (np_planet >= 1) & (np_planet <= 8)
    pv = jnp.where(is_valid, pv_valid, jnp.zeros((2, 3)))
    jstat = jnp.where(is_valid, jstat_valid, -1)

    return pv, jstat


__all__ = ["epv00", "moon98", "plan94"]
