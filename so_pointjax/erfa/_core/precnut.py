"""Precession, nutation, frame bias, and related rotation functions.

Ported from ERFA C library. All functions are differentiable and
JIT-compatible unless noted otherwise.

Phase 3 of the Differentiable ERFA project.
"""

import jax
import jax.numpy as jnp

from so_pointjax.erfa._core.constants import (
    DAS2R, D2PI, DJC, DJ00, DJM0, DJM00, TURNAS, DMAS2R,
)
from so_pointjax.erfa._core.angles import anp, anpm
from so_pointjax.erfa._core.vector import ir, rx, ry, rz, rxr, tr, cr


# ============================================================================
# Fundamental argument functions (IERS 2003)
# ============================================================================

def fal03(t):
    """Mean anomaly of the Moon (IERS 2003)."""
    return jnp.fmod(
        485868.249036
        + t * (1717915923.2178
        + t * (31.8792
        + t * (0.051635
        + t * (-0.00024470)))),
        TURNAS) * DAS2R


def falp03(t):
    """Mean anomaly of the Sun (IERS 2003)."""
    return jnp.fmod(
        1287104.793048
        + t * (129596581.0481
        + t * (-0.5532
        + t * (0.000136
        + t * (-0.00001149)))),
        TURNAS) * DAS2R


def faf03(t):
    """Mean argument of latitude of the Moon (IERS 2003)."""
    return jnp.fmod(
        335779.526232
        + t * (1739527262.8478
        + t * (-12.7512
        + t * (-0.001037
        + t * (0.00000417)))),
        TURNAS) * DAS2R


def fad03(t):
    """Mean elongation of the Moon from the Sun (IERS 2003)."""
    return jnp.fmod(
        1072260.703692
        + t * (1602961601.2090
        + t * (-6.3706
        + t * (0.006593
        + t * (-0.00003169)))),
        TURNAS) * DAS2R


def faom03(t):
    """Mean longitude of the Moon's ascending node (IERS 2003)."""
    return jnp.fmod(
        450160.398036
        + t * (-6962890.5431
        + t * (7.4722
        + t * (0.007702
        + t * (-0.00005939)))),
        TURNAS) * DAS2R


def fame03(t):
    """Mean longitude of Mercury (IERS 2003)."""
    return jnp.fmod(4.402608842 + 2608.7903141574 * t, D2PI)


def fave03(t):
    """Mean longitude of Venus (IERS 2003)."""
    return jnp.fmod(3.176146697 + 1021.3285546211 * t, D2PI)


def fae03(t):
    """Mean longitude of Earth (IERS 2003)."""
    return jnp.fmod(1.753470314 + 628.3075849991 * t, D2PI)


def fama03(t):
    """Mean longitude of Mars (IERS 2003)."""
    return jnp.fmod(6.203480913 + 334.0612426700 * t, D2PI)


def faju03(t):
    """Mean longitude of Jupiter (IERS 2003)."""
    return jnp.fmod(0.599546497 + 52.9690962641 * t, D2PI)


def fasa03(t):
    """Mean longitude of Saturn (IERS 2003)."""
    return jnp.fmod(0.874016757 + 21.3299104960 * t, D2PI)


def faur03(t):
    """Mean longitude of Uranus (IERS 2003)."""
    return jnp.fmod(5.481293872 + 7.4781598567 * t, D2PI)


def fane03(t):
    """Mean longitude of Neptune (IERS 2003)."""
    return jnp.fmod(5.311886287 + 3.8133035638 * t, D2PI)


def fapa03(t):
    """General accumulated precession in longitude (IERS 2003)."""
    return (0.024381750 + 0.00000538691 * t) * t


# ============================================================================
# Obliquity functions
# ============================================================================

def obl80(date1, date2):
    """Mean obliquity of the ecliptic, IAU 1980 model."""
    t = ((date1 - DJ00) + date2) / DJC
    return DAS2R * (84381.448
                    + (-46.8150
                    + (-0.00059
                    + 0.001813 * t) * t) * t)


def obl06(date1, date2):
    """Mean obliquity of the ecliptic, IAU 2006 precession."""
    t = ((date1 - DJ00) + date2) / DJC
    return (84381.406
            + (-46.836769
            + (-0.0001831
            + (0.00200340
            + (-0.000000576
            + (-0.0000000434) * t) * t) * t) * t) * t) * DAS2R


# ============================================================================
# Precession-rate adjustments
# ============================================================================

def pr00(date1, date2):
    """Precession-rate part of the IAU 2000 precession-nutation models."""
    t = ((date1 - DJ00) + date2) / DJC
    dpsipr = -0.29965 * DAS2R * t
    depspr = -0.02524 * DAS2R * t
    return dpsipr, depspr


# ============================================================================
# Frame bias
# ============================================================================

def bi00():
    """Frame bias components of IAU 2000 precession-nutation models."""
    dpsibi = -0.041775 * DAS2R
    depsbi = -0.0068192 * DAS2R
    dra = -0.0146 * DAS2R
    return dpsibi, depsbi, dra


# ============================================================================
# TIO locator s'
# ============================================================================

def sp00(date1, date2):
    """TIO locator s', positioning the Terrestrial Intermediate Origin."""
    t = ((date1 - DJ00) + date2) / DJC
    return -47e-6 * t * DAS2R


# ============================================================================
# Precession angles, IAU 1976 (Lieske)
# ============================================================================

def prec76(date01, date02, date11, date12):
    """IAU 1976 precession model (Euler angles)."""
    t0 = ((date01 - DJ00) + date02) / DJC
    t = ((date11 - date01) + (date12 - date02)) / DJC
    tas2r = t * DAS2R

    w = 2306.2181 + (1.39656 - 0.000139 * t0) * t0
    zeta = (w + ((0.30188 - 0.000344 * t0) + 0.017998 * t) * t) * tas2r
    z = (w + ((1.09468 + 0.000066 * t0) + 0.018203 * t) * t) * tas2r
    theta = ((2004.3109 + (-0.85330 - 0.000217 * t0) * t0)
             + ((-0.42665 - 0.000217 * t0) - 0.041833 * t) * t) * tas2r

    return zeta, z, theta


# ============================================================================
# Precession matrix, IAU 1976
# ============================================================================

def pmat76(date1, date2):
    """Precession matrix from J2000.0 to a specified date, IAU 1976 model."""
    zeta, z, theta = prec76(DJ00, 0.0, date1, date2)
    r = ir()
    r = rz(-zeta, r)
    r = ry(theta, r)
    r = rz(-z, r)
    return r


# ============================================================================
# Fukushima-Williams precession angles, IAU 2006
# ============================================================================

def pfw06(date1, date2):
    """Precession angles, IAU 2006 (Fukushima-Williams 4-angle formulation)."""
    t = ((date1 - DJ00) + date2) / DJC

    gamb = (-0.052928
            + (10.556378
            + (0.4932044
            + (-0.00031238
            + (-0.000002788
            + 0.0000000260 * t) * t) * t) * t) * t) * DAS2R

    phib = (84381.412819
            + (-46.811016
            + (0.0511268
            + (0.00053289
            + (-0.000000440
            + (-0.0000000176) * t) * t) * t) * t) * t) * DAS2R

    psib = (-0.041775
            + (5038.481484
            + (1.5584175
            + (-0.00018522
            + (-0.000026452
            + (-0.0000000148) * t) * t) * t) * t) * t) * DAS2R

    epsa = obl06(date1, date2)

    return gamb, phib, psib, epsa


# ============================================================================
# Fukushima-Williams angles to rotation matrix
# ============================================================================

def fw2m(gamb, phib, psi, eps):
    """Form rotation matrix given the Fukushima-Williams angles."""
    r = ir()
    r = rz(gamb, r)
    r = rx(phib, r)
    r = rz(-psi, r)
    r = rx(-eps, r)
    return r


# ============================================================================
# BPN matrix element extraction
# ============================================================================

def bpn2xy(rbpn):
    """Extract from the bias-precession-nutation matrix the X,Y of the CIP."""
    return rbpn[2, 0], rbpn[2, 1]


def fw2xy(gamb, phib, psi, eps):
    """CIP X,Y given Fukushima-Williams bias-precession-nutation angles."""
    r = fw2m(gamb, phib, psi, eps)
    return bpn2xy(r)


# ============================================================================
# Precession matrix, IAU 2006 (bias-precession)
# ============================================================================

def pmat06(date1, date2):
    """Precession matrix (including frame bias) from GCRS to a specified date, IAU 2006."""
    gamb, phib, psib, epsa = pfw06(date1, date2)
    return fw2m(gamb, phib, psib, epsa)


# ============================================================================
# Nutation matrix from obliquity and nutation angles
# ============================================================================

def numat(epsa, dpsi, deps):
    """Form nutation matrix from mean obliquity and nutation components."""
    r = ir()
    r = rx(epsa, r)
    r = rz(-dpsi, r)
    r = rx(-(epsa + deps), r)
    return r


# ============================================================================
# Frame bias and precession matrices, IAU 2000
# ============================================================================

def bp00(date1, date2):
    """Frame bias and precession, IAU 2000."""
    # J2000.0 obliquity (Lieske 1977)
    EPS0 = 84381.448 * DAS2R

    t = ((date1 - DJ00) + date2) / DJC

    # Frame bias
    dpsibi, depsbi, dra0 = bi00()

    # Precession angles (Lieske 1977)
    psia77 = (5038.7784 + (-1.07259 + (-0.001147) * t) * t) * t * DAS2R
    oma77 = EPS0 + ((0.05127 + (-0.007726) * t) * t) * t * DAS2R
    chia = (10.5526 + (-2.38064 + (-0.001125) * t) * t) * t * DAS2R

    # Apply IAU 2000 precession corrections
    dpsipr, depspr = pr00(date1, date2)
    psia = psia77 + dpsipr
    oma = oma77 + depspr

    # Frame bias matrix: GCRS to J2000.0
    rb = ir()
    rb = rz(dra0, rb)
    rb = ry(dpsibi * jnp.sin(EPS0), rb)
    rb = rx(-depsbi, rb)

    # Precession matrix: J2000.0 to mean of date
    rp = ir()
    rp = rx(EPS0, rp)
    rp = rz(-psia, rp)
    rp = rx(-oma, rp)
    rp = rz(chia, rp)

    # Bias-precession matrix: GCRS to mean of date
    rbp = rxr(rp, rb)

    return rb, rp, rbp


# ============================================================================
# Frame bias and precession matrices, IAU 2006
# ============================================================================

def bp06(date1, date2):
    """Frame bias and precession, IAU 2006."""
    # Bias matrix: evaluate F-W angles at J2000.0
    gamb, phib, psib, epsa = pfw06(DJM0, DJM00)
    rb = fw2m(gamb, phib, psib, epsa)

    # Bias-precession matrix: evaluate F-W angles at date
    rbpw = pmat06(date1, date2)

    # Precession matrix: rp = rbpw * rb^T
    rt = tr(rb)
    rp = rxr(rbpw, rt)

    rbp = rbpw
    return rb, rp, rbp


# ============================================================================
# Nutation, IAU 1980
# ============================================================================

# Coefficients: (nl, nlp, nf, nd, nom, sp, spt, ce, cet)
_NUT80_COEFFS = jnp.array([
    [ 0,  0,  0,  0,  1, -171996.0, -174.2,  92025.0,    8.9],
    [ 0,  0,  0,  0,  2,    2062.0,    0.2,   -895.0,    0.5],
    [-2,  0,  2,  0,  1,      46.0,    0.0,    -24.0,    0.0],
    [ 2,  0, -2,  0,  0,      11.0,    0.0,      0.0,    0.0],
    [-2,  0,  2,  0,  2,      -3.0,    0.0,      1.0,    0.0],
    [ 1, -1,  0, -1,  0,      -3.0,    0.0,      0.0,    0.0],
    [ 0, -2,  2, -2,  1,      -2.0,    0.0,      1.0,    0.0],
    [ 2,  0, -2,  0,  1,       1.0,    0.0,      0.0,    0.0],
    [ 0,  0,  2, -2,  2, -13187.0,   -1.6,   5736.0,   -3.1],
    [ 0,  1,  0,  0,  0,    1426.0,   -3.4,     54.0,   -0.1],
    [ 0,  1,  2, -2,  2,    -517.0,    1.2,    224.0,   -0.6],
    [ 0, -1,  2, -2,  2,     217.0,   -0.5,    -95.0,    0.3],
    [ 0,  0,  2, -2,  1,     129.0,    0.1,    -70.0,    0.0],
    [ 2,  0,  0, -2,  0,      48.0,    0.0,      1.0,    0.0],
    [ 0,  0,  2, -2,  0,     -22.0,    0.0,      0.0,    0.0],
    [ 0,  2,  0,  0,  0,      17.0,   -0.1,      0.0,    0.0],
    [ 0,  1,  0,  0,  1,     -15.0,    0.0,      9.0,    0.0],
    [ 0,  2,  2, -2,  2,     -16.0,    0.1,      7.0,    0.0],
    [ 0, -1,  0,  0,  1,     -12.0,    0.0,      6.0,    0.0],
    [-2,  0,  0,  2,  1,      -6.0,    0.0,      3.0,    0.0],
    [ 0, -1,  2, -2,  1,      -5.0,    0.0,      3.0,    0.0],
    [ 2,  0,  0, -2,  1,       4.0,    0.0,     -2.0,    0.0],
    [ 0,  1,  2, -2,  1,       4.0,    0.0,     -2.0,    0.0],
    [ 1,  0,  0, -1,  0,      -4.0,    0.0,      0.0,    0.0],
    [ 2,  1,  0, -2,  0,       1.0,    0.0,      0.0,    0.0],
    [ 0,  0, -2,  2,  1,       1.0,    0.0,      0.0,    0.0],
    [ 0,  1, -2,  2,  0,      -1.0,    0.0,      0.0,    0.0],
    [ 0,  1,  0,  0,  2,       1.0,    0.0,      0.0,    0.0],
    [-1,  0,  0,  1,  1,       1.0,    0.0,      0.0,    0.0],
    [ 0,  1,  2, -2,  0,      -1.0,    0.0,      0.0,    0.0],
    [ 0,  0,  2,  0,  2,   -2274.0,   -0.2,    977.0,   -0.5],
    [ 1,  0,  0,  0,  0,     712.0,    0.1,     -7.0,    0.0],
    [ 0,  0,  2,  0,  1,    -386.0,   -0.4,    200.0,    0.0],
    [ 1,  0,  2,  0,  2,    -301.0,    0.0,    129.0,   -0.1],
    [ 1,  0,  0, -2,  0,    -158.0,    0.0,     -1.0,    0.0],
    [-1,  0,  2,  0,  2,     123.0,    0.0,    -53.0,    0.0],
    [ 0,  0,  0,  2,  0,      63.0,    0.0,     -2.0,    0.0],
    [ 1,  0,  0,  0,  1,      63.0,    0.1,    -33.0,    0.0],
    [-1,  0,  0,  0,  1,     -58.0,   -0.1,     32.0,    0.0],
    [-1,  0,  2,  2,  2,     -59.0,    0.0,     26.0,    0.0],
    [ 1,  0,  2,  0,  1,     -51.0,    0.0,     27.0,    0.0],
    [ 0,  0,  2,  2,  2,     -38.0,    0.0,     16.0,    0.0],
    [ 2,  0,  0,  0,  0,      29.0,    0.0,     -1.0,    0.0],
    [ 1,  0,  2, -2,  2,      29.0,    0.0,    -12.0,    0.0],
    [ 2,  0,  2,  0,  2,     -31.0,    0.0,     13.0,    0.0],
    [ 0,  0,  2,  0,  0,      26.0,    0.0,     -1.0,    0.0],
    [-1,  0,  2,  0,  1,      21.0,    0.0,    -10.0,    0.0],
    [-1,  0,  0,  2,  1,      16.0,    0.0,     -8.0,    0.0],
    [ 1,  0,  0, -2,  1,     -13.0,    0.0,      7.0,    0.0],
    [-1,  0,  2,  2,  1,     -10.0,    0.0,      5.0,    0.0],
    [ 1,  1,  0, -2,  0,      -7.0,    0.0,      0.0,    0.0],
    [ 0,  1,  2,  0,  2,       7.0,    0.0,     -3.0,    0.0],
    [ 0, -1,  2,  0,  2,      -7.0,    0.0,      3.0,    0.0],
    [ 1,  0,  2,  2,  2,      -8.0,    0.0,      3.0,    0.0],
    [ 1,  0,  0,  2,  0,       6.0,    0.0,      0.0,    0.0],
    [ 2,  0,  2, -2,  2,       6.0,    0.0,     -3.0,    0.0],
    [ 0,  0,  0,  2,  1,      -6.0,    0.0,      3.0,    0.0],
    [ 0,  0,  2,  2,  1,      -7.0,    0.0,      3.0,    0.0],
    [ 1,  0,  2, -2,  1,       6.0,    0.0,     -3.0,    0.0],
    [ 0,  0,  0, -2,  1,      -5.0,    0.0,      3.0,    0.0],
    [ 1, -1,  0,  0,  0,       5.0,    0.0,      0.0,    0.0],
    [ 2,  0,  2,  0,  1,      -5.0,    0.0,      3.0,    0.0],
    [ 0,  1,  0, -2,  0,      -4.0,    0.0,      0.0,    0.0],
    [ 1,  0, -2,  0,  0,       4.0,    0.0,      0.0,    0.0],
    [ 0,  0,  0,  1,  0,      -4.0,    0.0,      0.0,    0.0],
    [ 1,  1,  0,  0,  0,      -3.0,    0.0,      0.0,    0.0],
    [ 1,  0,  2,  0,  0,       3.0,    0.0,      0.0,    0.0],
    [ 1, -1,  2,  0,  2,      -3.0,    0.0,      1.0,    0.0],
    [-1, -1,  2,  2,  2,      -3.0,    0.0,      1.0,    0.0],
    [-2,  0,  0,  0,  1,      -2.0,    0.0,      1.0,    0.0],
    [ 3,  0,  2,  0,  2,      -3.0,    0.0,      1.0,    0.0],
    [ 0, -1,  2,  2,  2,      -3.0,    0.0,      1.0,    0.0],
    [ 1,  1,  2,  0,  2,       2.0,    0.0,     -1.0,    0.0],
    [-1,  0,  2, -2,  1,      -2.0,    0.0,      1.0,    0.0],
    [ 2,  0,  0,  0,  1,       2.0,    0.0,     -1.0,    0.0],
    [ 1,  0,  0,  0,  2,      -2.0,    0.0,      1.0,    0.0],
    [ 3,  0,  0,  0,  0,       2.0,    0.0,      0.0,    0.0],
    [ 0,  0,  2,  1,  2,       2.0,    0.0,     -1.0,    0.0],
    [-1,  0,  0,  0,  2,       1.0,    0.0,     -1.0,    0.0],
    [ 1,  0,  0, -4,  0,      -1.0,    0.0,      0.0,    0.0],
    [-2,  0,  2,  2,  2,       1.0,    0.0,     -1.0,    0.0],
    [-1,  0,  2,  4,  2,      -2.0,    0.0,      1.0,    0.0],
    [ 2,  0,  0, -4,  0,      -1.0,    0.0,      0.0,    0.0],
    [ 1,  1,  2, -2,  2,       1.0,    0.0,     -1.0,    0.0],
    [ 1,  0,  2,  2,  1,      -1.0,    0.0,      1.0,    0.0],
    [-2,  0,  2,  4,  2,      -1.0,    0.0,      1.0,    0.0],
    [-1,  0,  4,  0,  2,       1.0,    0.0,      0.0,    0.0],
    [ 1, -1,  0, -2,  0,       1.0,    0.0,      0.0,    0.0],
    [ 2,  0,  2, -2,  1,       1.0,    0.0,     -1.0,    0.0],
    [ 2,  0,  2,  2,  2,      -1.0,    0.0,      0.0,    0.0],
    [ 1,  0,  0,  2,  1,      -1.0,    0.0,      0.0,    0.0],
    [ 0,  0,  4, -2,  2,       1.0,    0.0,      0.0,    0.0],
    [ 3,  0,  2, -2,  2,       1.0,    0.0,      0.0,    0.0],
    [ 1,  0,  2, -2,  0,      -1.0,    0.0,      0.0,    0.0],
    [ 0,  1,  2,  0,  1,       1.0,    0.0,      0.0,    0.0],
    [-1, -1,  0,  2,  1,       1.0,    0.0,      0.0,    0.0],
    [ 0,  0, -2,  0,  1,      -1.0,    0.0,      0.0,    0.0],
    [ 0,  0,  2, -1,  2,      -1.0,    0.0,      0.0,    0.0],
    [ 0,  1,  0,  2,  0,      -1.0,    0.0,      0.0,    0.0],
    [ 1,  0, -2, -2,  0,      -1.0,    0.0,      0.0,    0.0],
    [ 0, -1,  2,  0,  1,      -1.0,    0.0,      0.0,    0.0],
    [ 1,  1,  0, -2,  1,      -1.0,    0.0,      0.0,    0.0],
    [ 1,  0, -2,  2,  0,      -1.0,    0.0,      0.0,    0.0],
    [ 2,  0,  0,  2,  0,       1.0,    0.0,      0.0,    0.0],
    [ 0,  0,  2,  4,  2,      -1.0,    0.0,      0.0,    0.0],
    [ 0,  1,  0,  1,  0,       1.0,    0.0,      0.0,    0.0],
], dtype=jnp.float64)


def nut80(date1, date2):
    """Nutation, IAU 1980 model."""
    U2R = DAS2R / 1e4  # 0.1 milliarcsecond to radians

    t = ((date1 - DJ00) + date2) / DJC

    # Fundamental (Delaunay) arguments
    el = anpm((485866.733 + (715922.633 + (31.310 + 0.064 * t) * t) * t) * DAS2R
              + jnp.fmod(1325.0 * t, 1.0) * D2PI)
    elp = anpm((1287099.804 + (1292581.224 + (-0.577 - 0.012 * t) * t) * t) * DAS2R
               + jnp.fmod(99.0 * t, 1.0) * D2PI)
    f = anpm((335778.877 + (295263.137 + (-13.257 + 0.011 * t) * t) * t) * DAS2R
             + jnp.fmod(1342.0 * t, 1.0) * D2PI)
    d = anpm((1072261.307 + (1105601.328 + (-6.891 + 0.019 * t) * t) * t) * DAS2R
             + jnp.fmod(1236.0 * t, 1.0) * D2PI)
    om = anpm((450160.280 + (-482890.539 + (7.455 + 0.008 * t) * t) * t) * DAS2R
              + jnp.fmod(-5.0 * t, 1.0) * D2PI)

    fa = jnp.array([el, elp, f, d, om])

    # Sum nutation series
    nfa = _NUT80_COEFFS[:, :5]  # (106, 5) integer multipliers
    sp = _NUT80_COEFFS[:, 5]
    spt = _NUT80_COEFFS[:, 6]
    ce = _NUT80_COEFFS[:, 7]
    cet = _NUT80_COEFFS[:, 8]

    arg = nfa @ fa  # (106,)
    s = sp + spt * t
    c = ce + cet * t
    dp = jnp.sum(s * jnp.sin(arg))
    de = jnp.sum(c * jnp.cos(arg))

    dpsi = dp * U2R
    deps = de * U2R
    return dpsi, deps


# ============================================================================
# Nutation matrix, IAU 1980
# ============================================================================

def nutm80(date1, date2):
    """Nutation matrix, IAU 1980."""
    dpsi, deps = nut80(date1, date2)
    epsa = obl80(date1, date2)
    return numat(epsa, dpsi, deps)


# ============================================================================
# Nutation, IAU 2000B (simplified, 77 luni-solar terms)
# ============================================================================

# Coefficients: (nl, nlp, nf, nd, nom, ps, pst, pc, ec, ect, es)
_NUT00B_COEFFS = jnp.array([
    [ 0,  0,  0,  0,  1, -172064161.0, -174666.0,  33386.0,  92052331.0,   9086.0,  15377.0],
    [ 0,  0,  2, -2,  2,  -13170906.0,   -1675.0, -13696.0,   5730336.0,  -3015.0,  -4587.0],
    [ 0,  0,  2,  0,  2,   -2276413.0,    -234.0,   2796.0,    978459.0,   -485.0,   1374.0],
    [ 0,  0,  0,  0,  2,    2074554.0,     207.0,   -698.0,   -897492.0,    470.0,   -291.0],
    [ 0,  1,  0,  0,  0,    1475877.0,   -3633.0,  11817.0,     73871.0,   -184.0,  -1924.0],
    [ 0,  1,  2, -2,  2,    -516821.0,    1226.0,   -524.0,    224386.0,   -677.0,   -174.0],
    [ 1,  0,  0,  0,  0,     711159.0,      73.0,   -872.0,     -6750.0,      0.0,    358.0],
    [ 0,  0,  2,  0,  1,    -387298.0,    -367.0,    380.0,    200728.0,     18.0,    318.0],
    [ 1,  0,  2,  0,  2,    -301461.0,     -36.0,    816.0,    129025.0,    -63.0,    367.0],
    [ 0, -1,  2, -2,  2,     215829.0,    -494.0,    111.0,    -95929.0,    299.0,    132.0],
    [ 0,  0,  2, -2,  1,     128227.0,     137.0,    181.0,    -68982.0,     -9.0,     39.0],
    [-1,  0,  2,  0,  2,     123457.0,      11.0,     19.0,    -53311.0,     32.0,     -4.0],
    [-1,  0,  0,  2,  0,     156994.0,      10.0,   -168.0,     -1235.0,      0.0,     82.0],
    [ 1,  0,  0,  0,  1,      63110.0,      63.0,     27.0,    -33228.0,      0.0,     -9.0],
    [-1,  0,  0,  0,  1,     -57976.0,     -63.0,   -189.0,     31429.0,      0.0,    -75.0],
    [-1,  0,  2,  2,  2,     -59641.0,     -11.0,    149.0,     25543.0,    -11.0,     66.0],
    [ 1,  0,  2,  0,  1,     -51613.0,     -42.0,    129.0,     26366.0,      0.0,     78.0],
    [-2,  0,  2,  0,  1,      45893.0,      50.0,     31.0,    -24236.0,    -10.0,     20.0],
    [ 0,  0,  0,  2,  0,      63384.0,      11.0,   -150.0,     -1220.0,      0.0,     29.0],
    [ 0,  0,  2,  2,  2,     -38571.0,      -1.0,    158.0,     16452.0,    -11.0,     68.0],
    [ 0, -2,  2, -2,  2,      32481.0,       0.0,      0.0,    -13870.0,      0.0,      0.0],
    [-2,  0,  0,  2,  0,     -47722.0,       0.0,    -18.0,       477.0,      0.0,    -25.0],
    [ 2,  0,  2,  0,  2,     -31046.0,      -1.0,    131.0,     13238.0,    -11.0,     59.0],
    [ 1,  0,  2, -2,  2,      28593.0,       0.0,     -1.0,    -12338.0,     10.0,     -3.0],
    [-1,  0,  2,  0,  1,      20441.0,      21.0,     10.0,    -10758.0,      0.0,     -3.0],
    [ 2,  0,  0,  0,  0,      29243.0,       0.0,    -74.0,      -609.0,      0.0,     13.0],
    [ 0,  0,  2,  0,  0,      25887.0,       0.0,    -66.0,      -550.0,      0.0,     11.0],
    [ 0,  1,  0,  0,  1,     -14053.0,     -25.0,     79.0,      8551.0,     -2.0,    -45.0],
    [-1,  0,  0,  2,  1,      15164.0,      10.0,     11.0,     -8001.0,      0.0,     -1.0],
    [ 0,  2,  2, -2,  2,     -15794.0,      72.0,    -16.0,      6850.0,    -42.0,     -5.0],
    [ 0,  0, -2,  2,  0,      21783.0,       0.0,     13.0,      -167.0,      0.0,     13.0],
    [ 1,  0,  0, -2,  1,     -12873.0,     -10.0,    -37.0,      6953.0,      0.0,    -14.0],
    [ 0, -1,  0,  0,  1,     -12654.0,      11.0,     63.0,      6415.0,      0.0,     26.0],
    [-1,  0,  2,  2,  1,     -10204.0,       0.0,     25.0,      5222.0,      0.0,     15.0],
    [ 0,  2,  0,  0,  0,      16707.0,     -85.0,    -10.0,       168.0,     -1.0,     10.0],
    [ 1,  0,  2,  2,  2,      -7691.0,       0.0,     44.0,      3268.0,      0.0,     19.0],
    [-2,  0,  2,  0,  0,     -11024.0,       0.0,    -14.0,       104.0,      0.0,      2.0],
    [ 0,  1,  2,  0,  2,       7566.0,     -21.0,    -11.0,     -3250.0,      0.0,     -5.0],
    [ 0,  0,  2,  2,  1,      -6637.0,     -11.0,     25.0,      3353.0,      0.0,     14.0],
    [ 0, -1,  2,  0,  2,      -7141.0,      21.0,      8.0,      3070.0,      0.0,      4.0],
    [ 0,  0,  0,  2,  1,      -6302.0,     -11.0,      2.0,      3272.0,      0.0,      4.0],
    [ 1,  0,  2, -2,  1,       5800.0,      10.0,      2.0,     -3045.0,      0.0,     -1.0],
    [ 2,  0,  2, -2,  2,       6443.0,       0.0,     -7.0,     -2768.0,      0.0,     -4.0],
    [-2,  0,  0,  2,  1,      -5774.0,     -11.0,    -15.0,      3041.0,      0.0,     -5.0],
    [ 2,  0,  2,  0,  1,      -5350.0,       0.0,     21.0,      2695.0,      0.0,     12.0],
    [ 0, -1,  2, -2,  1,      -4752.0,     -11.0,     -3.0,      2719.0,      0.0,     -3.0],
    [ 0,  0,  0, -2,  1,      -4940.0,     -11.0,    -21.0,      2720.0,      0.0,     -9.0],
    [-1, -1,  0,  2,  0,       7350.0,       0.0,     -8.0,       -51.0,      0.0,      4.0],
    [ 2,  0,  0, -2,  1,       4065.0,       0.0,      6.0,     -2206.0,      0.0,      1.0],
    [ 1,  0,  0,  2,  0,       6579.0,       0.0,    -24.0,      -199.0,      0.0,      2.0],
    [ 0,  1,  2, -2,  1,       3579.0,       0.0,      5.0,     -1900.0,      0.0,      1.0],
    [ 1, -1,  0,  0,  0,       4725.0,       0.0,     -6.0,       -41.0,      0.0,      3.0],
    [-2,  0,  2,  0,  2,      -3075.0,       0.0,     -2.0,      1313.0,      0.0,     -1.0],
    [ 3,  0,  2,  0,  2,      -2904.0,       0.0,     15.0,      1233.0,      0.0,      7.0],
    [ 0, -1,  0,  2,  0,       4348.0,       0.0,    -10.0,       -81.0,      0.0,      2.0],
    [ 1, -1,  2,  0,  2,      -2878.0,       0.0,      8.0,      1232.0,      0.0,      4.0],
    [ 0,  0,  0,  1,  0,      -4230.0,       0.0,      5.0,       -20.0,      0.0,     -2.0],
    [-1, -1,  2,  2,  2,      -2819.0,       0.0,      7.0,      1207.0,      0.0,      3.0],
    [-1,  0,  2,  0,  0,      -4056.0,       0.0,      5.0,        40.0,      0.0,     -2.0],
    [ 0, -1,  2,  2,  2,      -2647.0,       0.0,     11.0,      1129.0,      0.0,      5.0],
    [-2,  0,  0,  0,  1,      -2294.0,       0.0,    -10.0,      1266.0,      0.0,     -4.0],
    [ 1,  1,  2,  0,  2,       2481.0,       0.0,     -7.0,     -1062.0,      0.0,     -3.0],
    [ 2,  0,  0,  0,  1,       2179.0,       0.0,     -2.0,     -1129.0,      0.0,     -2.0],
    [-1,  1,  0,  1,  0,       3276.0,       0.0,      1.0,        -9.0,      0.0,      0.0],
    [ 1,  1,  0,  0,  0,      -3389.0,       0.0,      5.0,        35.0,      0.0,     -2.0],
    [ 1,  0,  2,  0,  0,       3339.0,       0.0,    -13.0,      -107.0,      0.0,      1.0],
    [-1,  0,  2, -2,  1,      -1987.0,       0.0,     -6.0,      1073.0,      0.0,     -2.0],
    [ 1,  0,  0,  0,  2,      -1981.0,       0.0,      0.0,       854.0,      0.0,      0.0],
    [-1,  0,  0,  1,  0,       4026.0,       0.0,   -353.0,      -553.0,      0.0,   -139.0],
    [ 0,  0,  2,  1,  2,       1660.0,       0.0,     -5.0,      -710.0,      0.0,     -2.0],
    [-1,  0,  2,  4,  2,      -1521.0,       0.0,      9.0,       647.0,      0.0,      4.0],
    [-1,  1,  0,  1,  1,       1314.0,       0.0,      0.0,      -700.0,      0.0,      0.0],
    [ 0, -2,  2, -2,  1,      -1283.0,       0.0,      0.0,       672.0,      0.0,      0.0],
    [ 1,  0,  2,  2,  1,      -1331.0,       0.0,      8.0,       663.0,      0.0,      4.0],
    [-2,  0,  2,  2,  2,       1383.0,       0.0,     -2.0,      -594.0,      0.0,     -2.0],
    [-1,  0,  0,  0,  2,       1405.0,       0.0,      4.0,      -610.0,      0.0,      2.0],
    [ 1,  1,  2, -2,  2,       1290.0,       0.0,      0.0,      -556.0,      0.0,      0.0],
], dtype=jnp.float64)


def nut00b(date1, date2):
    """Nutation, IAU 2000B (truncated, for when accuracy of 1 mas is acceptable)."""
    U2R = DAS2R / 1e7  # 0.1 microarcsecond to radians
    DPPLAN = -0.135 * DMAS2R
    DEPLAN = 0.388 * DMAS2R

    t = ((date1 - DJ00) + date2) / DJC

    # Fundamental arguments (Simon et al. 1994)
    el = jnp.fmod(485868.249036 + 1717915923.2178 * t, TURNAS) * DAS2R
    elp = jnp.fmod(1287104.79305 + 129596581.0481 * t, TURNAS) * DAS2R
    f = jnp.fmod(335779.526232 + 1739527262.8478 * t, TURNAS) * DAS2R
    d = jnp.fmod(1072260.70369 + 1602961601.2090 * t, TURNAS) * DAS2R
    om = jnp.fmod(450160.398036 + (-6962890.5431) * t, TURNAS) * DAS2R

    fa = jnp.array([el, elp, f, d, om])

    nfa = _NUT00B_COEFFS[:, :5]
    ps = _NUT00B_COEFFS[:, 5]
    pst = _NUT00B_COEFFS[:, 6]
    pc = _NUT00B_COEFFS[:, 7]
    ec = _NUT00B_COEFFS[:, 8]
    ect = _NUT00B_COEFFS[:, 9]
    es = _NUT00B_COEFFS[:, 10]

    arg = jnp.fmod(nfa @ fa, D2PI)
    sarg = jnp.sin(arg)
    carg = jnp.cos(arg)

    dp = jnp.sum((ps + pst * t) * sarg + pc * carg)
    de = jnp.sum((ec + ect * t) * carg + es * sarg)

    dpsi = dp * U2R + DPPLAN
    deps = de * U2R + DEPLAN
    return dpsi, deps


# ============================================================================
# Nutation, IAU 2000A (full model, 678 luni-solar + 687 planetary terms)
# ============================================================================

def _load_nut00a_tables():
    """Load the IAU 2000A nutation coefficient tables from the ERFA C source.

    We read the tables directly from the C source file and cache them as JAX arrays.
    This avoids duplicating the ~1365 rows of coefficients in Python.
    """
    import re
    import os
    import numpy as np

    src_path = os.path.join(os.path.dirname(__file__), '..', '..', 'erfa', 'src', 'nut00a.c')
    src_path = os.path.normpath(src_path)

    with open(src_path, 'r') as f:
        source = f.read()

    # Parse luni-solar table (xls[])
    # Find the block between "xls[]" declaration and its closing "};"
    xls_match = re.search(r'}\s+xls\[\]\s*=\s*\{(.*?)\};', source, re.DOTALL)
    xls_text = xls_match.group(1)
    xls_rows = re.findall(r'\{([^}]+)\}', xls_text)

    xls_data = []
    for row in xls_rows:
        vals = [float(v.strip()) for v in row.split(',')]
        xls_data.append(vals)
    xls_arr = np.array(xls_data, dtype=np.float64)

    # Parse planetary table (xpl[])
    xpl_match = re.search(r'}\s+xpl\[\]\s*=\s*\{(.*?)\};', source, re.DOTALL)
    xpl_text = xpl_match.group(1)
    xpl_rows = re.findall(r'\{([^}]+)\}', xpl_text)

    xpl_data = []
    for row in xpl_rows:
        vals = [float(v.strip()) for v in row.split(',')]
        xpl_data.append(vals)
    xpl_arr = np.array(xpl_data, dtype=np.float64)

    return jnp.array(xls_arr), jnp.array(xpl_arr)


# Lazily loaded tables
_nut00a_xls = None
_nut00a_xpl = None


def _get_nut00a_tables():
    global _nut00a_xls, _nut00a_xpl
    if _nut00a_xls is None:
        _nut00a_xls, _nut00a_xpl = _load_nut00a_tables()
    return _nut00a_xls, _nut00a_xpl


def nut00a(date1, date2):
    """Nutation, IAU 2000A model (MHB2000 luni-solar and planetary nutation).

    This is the full model with 678 luni-solar and 687 planetary terms.
    """
    U2R = DAS2R / 1e7  # 0.1 microarcsecond to radians

    t = ((date1 - DJ00) + date2) / DJC

    xls, xpl = _get_nut00a_tables()

    # --- Luni-solar nutation ---

    # Fundamental arguments (IERS 2003 / MHB2000 mix, as in C source)
    el = fal03(t)
    elp = jnp.fmod(1287104.79305 + t * (129596581.0481 + t * (-0.5532 + t * (0.000136 + t * (-0.00001149)))), TURNAS) * DAS2R
    f = faf03(t)
    d = jnp.fmod(1072260.70369 + t * (1602961601.2090 + t * (-6.3706 + t * (0.006593 + t * (-0.00003169)))), TURNAS) * DAS2R
    om = faom03(t)

    fa_ls = jnp.array([el, elp, f, d, om])

    # xls columns: nl, nlp, nf, nd, nom, sp, spt, cp, ce, cet, se
    nfa_ls = xls[:, :5]
    sp = xls[:, 5]
    spt = xls[:, 6]
    cp = xls[:, 7]
    ce = xls[:, 8]
    cet = xls[:, 9]
    se = xls[:, 10]

    arg_ls = jnp.fmod(nfa_ls @ fa_ls, D2PI)
    sarg = jnp.sin(arg_ls)
    carg = jnp.cos(arg_ls)

    dpls = jnp.sum((sp + spt * t) * sarg + cp * carg)
    dels = jnp.sum((ce + cet * t) * carg + se * sarg)

    # --- Planetary nutation ---

    # Fundamental arguments for planetary terms (MHB2000 simplified Delaunay + planetary longitudes)
    al = jnp.fmod(2.35555598 + 8328.6914269554 * t, D2PI)
    af = jnp.fmod(1.627905234 + 8433.466158131 * t, D2PI)
    ad = jnp.fmod(5.198466741 + 7771.3771468121 * t, D2PI)
    aom = jnp.fmod(2.18243920 - 33.757045 * t, D2PI)
    apa = fapa03(t)
    alme = fame03(t)
    alve = fave03(t)
    alea = fae03(t)
    alma = fama03(t)
    alju = faju03(t)
    alsa = fasa03(t)
    alur = faur03(t)
    alne = jnp.fmod(5.321159000 + 3.8127774000 * t, D2PI)

    fa_pl = jnp.array([al, af, ad, aom, alme, alve, alea, alma, alju, alsa, alur, alne, apa])

    # xpl columns: nl, nf, nd, nom, nme, nve, nea, nma, nju, nsa, nur, nne, npa, sp, cp, se, ce
    nfa_pl = xpl[:, :13]
    sp_pl = xpl[:, 13]
    cp_pl = xpl[:, 14]
    se_pl = xpl[:, 15]
    ce_pl = xpl[:, 16]

    arg_pl = jnp.fmod(nfa_pl @ fa_pl, D2PI)
    sarg_pl = jnp.sin(arg_pl)
    carg_pl = jnp.cos(arg_pl)

    dppl = jnp.sum(sp_pl * sarg_pl + cp_pl * carg_pl)
    depl = jnp.sum(se_pl * sarg_pl + ce_pl * carg_pl)

    # Combine
    dpsi = (dpls + dppl) * U2R
    deps = (dels + depl) * U2R
    return dpsi, deps


# ============================================================================
# Nutation, IAU 2006/2000A
# ============================================================================

def nut06a(date1, date2):
    """Nutation, IAU 2006/2000A (MHB2000 adjusted for IAU 2006 precession)."""
    t = ((date1 - DJ00) + date2) / DJC
    fj2 = -2.7774e-6 * t

    dp, de = nut00a(date1, date2)

    dpsi = dp + dp * (0.4697e-6 + fj2)
    deps = de + de * fj2
    return dpsi, deps


# ============================================================================
# Nutation matrices (IAU 2000A, 2000B, 2006/2000A)
# ============================================================================

def num06a(date1, date2):
    """Form the nutation matrix, IAU 2006/2000A."""
    eps = obl06(date1, date2)
    dp, de = nut06a(date1, date2)
    return numat(eps, dp, de)


# ============================================================================
# Precession-nutation, IAU 2000 (base function)
# ============================================================================

def pn00(date1, date2, dpsi, deps):
    """Precession-nutation, IAU 2000, given nutation (dpsi, deps)."""
    dpsipr, depspr = pr00(date1, date2)
    epsa = obl80(date1, date2) + depspr

    rb, rp, rbp = bp00(date1, date2)
    rn = numat(epsa, dpsi, deps)
    rbpn = rxr(rn, rbp)

    return dpsi, deps, epsa, rb, rp, rbp, rn, rbpn


def pn00a(date1, date2):
    """Precession-nutation, IAU 2000A."""
    dpsi, deps = nut00a(date1, date2)
    return pn00(date1, date2, dpsi, deps)


def pn00b(date1, date2):
    """Precession-nutation, IAU 2000B."""
    dpsi, deps = nut00b(date1, date2)
    return pn00(date1, date2, dpsi, deps)


# ============================================================================
# Precession-nutation, IAU 2006 (base function)
# ============================================================================

def pn06(date1, date2, dpsi, deps):
    """Precession-nutation, IAU 2006, given nutation (dpsi, deps)."""
    # Frame bias at J2000.0
    gamb0, phib0, psib0, eps0 = pfw06(DJM0, DJM00)
    rb = fw2m(gamb0, phib0, psib0, eps0)

    # Bias-precession at date
    gamb, phib, psib, eps = pfw06(date1, date2)
    r2 = fw2m(gamb, phib, psib, eps)

    # Precession matrix: rp = r2 * rb^T
    rt = tr(rb)
    rp = rxr(r2, rt)

    rbp = r2

    # Bias-precession-nutation matrix (add nutation to F-W angles)
    rbpn = fw2m(gamb, phib, psib + dpsi, eps + deps)

    # Nutation matrix: rn = rbpn * rbp^T
    rt2 = tr(r2)
    rn = rxr(rbpn, rt2)

    epsa = eps
    return dpsi, deps, epsa, rb, rp, rbp, rn, rbpn


def pn06a(date1, date2):
    """Precession-nutation, IAU 2006/2000A."""
    dpsi, deps = nut06a(date1, date2)
    return pn06(date1, date2, dpsi, deps)


# ============================================================================
# Precession-nutation matrices (convenience wrappers)
# ============================================================================

def pnm00a(date1, date2):
    """Classical NPB matrix, IAU 2000A."""
    _, _, _, _, _, _, _, rbpn = pn00a(date1, date2)
    return rbpn


def pnm00b(date1, date2):
    """Classical NPB matrix, IAU 2000B."""
    _, _, _, _, _, _, _, rbpn = pn00b(date1, date2)
    return rbpn


def pnm06a(date1, date2):
    """Classical NPB matrix, IAU 2006/2000A."""
    gamb, phib, psib, epsa = pfw06(date1, date2)
    dp, de = nut06a(date1, date2)
    return fw2m(gamb, phib, psib + dp, epsa + de)


def pnm80(date1, date2):
    """Precession-nutation matrix, IAU 1976 precession, IAU 1980 nutation."""
    rmatp = pmat76(date1, date2)
    rmatn = nutm80(date1, date2)
    return rxr(rmatn, rmatp)


# ============================================================================
# Nutation matrices, IAU 2000A/B (via pn00a/b)
# ============================================================================

def num00a(date1, date2):
    """Nutation matrix, IAU 2000A."""
    _, _, _, _, _, _, rn, _ = pn00a(date1, date2)
    return rn


def num00b(date1, date2):
    """Nutation matrix, IAU 2000B."""
    _, _, _, _, _, _, rn, _ = pn00b(date1, date2)
    return rn


# ============================================================================
# Equation of the equinoxes complementary terms
# ============================================================================

# e0 coefficients: (8 fundamental arg multipliers, sin, cos) -- 33 terms
_EECT00_E0 = jnp.array([
    [ 0, 0, 0, 0, 1, 0, 0, 0,  2640.96e-6, -0.39e-6],
    [ 0, 0, 0, 0, 2, 0, 0, 0,    63.52e-6, -0.02e-6],
    [ 0, 0, 2,-2, 3, 0, 0, 0,    11.75e-6,  0.01e-6],
    [ 0, 0, 2,-2, 1, 0, 0, 0,    11.21e-6,  0.01e-6],
    [ 0, 0, 2,-2, 2, 0, 0, 0,    -4.55e-6,  0.00e-6],
    [ 0, 0, 2, 0, 3, 0, 0, 0,     2.02e-6,  0.00e-6],
    [ 0, 0, 2, 0, 1, 0, 0, 0,     1.98e-6,  0.00e-6],
    [ 0, 0, 0, 0, 3, 0, 0, 0,    -1.72e-6,  0.00e-6],
    [ 0, 1, 0, 0, 1, 0, 0, 0,    -1.41e-6, -0.01e-6],
    [ 0, 1, 0, 0,-1, 0, 0, 0,    -1.26e-6, -0.01e-6],
    [ 1, 0, 0, 0,-1, 0, 0, 0,    -0.63e-6,  0.00e-6],
    [ 1, 0, 0, 0, 1, 0, 0, 0,    -0.63e-6,  0.00e-6],
    [ 0, 1, 2,-2, 3, 0, 0, 0,     0.46e-6,  0.00e-6],
    [ 0, 1, 2,-2, 1, 0, 0, 0,     0.45e-6,  0.00e-6],
    [ 0, 0, 4,-4, 4, 0, 0, 0,     0.36e-6,  0.00e-6],
    [ 0, 0, 1,-1, 1,-8,12, 0,    -0.24e-6, -0.12e-6],
    [ 0, 0, 2, 0, 0, 0, 0, 0,     0.32e-6,  0.00e-6],
    [ 0, 0, 2, 0, 2, 0, 0, 0,     0.28e-6,  0.00e-6],
    [ 1, 0, 2, 0, 3, 0, 0, 0,     0.27e-6,  0.00e-6],
    [ 1, 0, 2, 0, 1, 0, 0, 0,     0.26e-6,  0.00e-6],
    [ 0, 0, 2,-2, 0, 0, 0, 0,    -0.21e-6,  0.00e-6],
    [ 0, 1,-2, 2,-3, 0, 0, 0,     0.19e-6,  0.00e-6],
    [ 0, 1,-2, 2,-1, 0, 0, 0,     0.18e-6,  0.00e-6],
    [ 0, 0, 0, 0, 0, 8,-13,-1,   -0.10e-6,  0.05e-6],
    [ 0, 0, 0, 2, 0, 0, 0, 0,     0.15e-6,  0.00e-6],
    [ 2, 0,-2, 0,-1, 0, 0, 0,    -0.14e-6,  0.00e-6],
    [ 1, 0, 0,-2, 1, 0, 0, 0,     0.14e-6,  0.00e-6],
    [ 0, 1, 2,-2, 2, 0, 0, 0,    -0.14e-6,  0.00e-6],
    [ 1, 0, 0,-2,-1, 0, 0, 0,     0.14e-6,  0.00e-6],
    [ 0, 0, 4,-2, 4, 0, 0, 0,     0.13e-6,  0.00e-6],
    [ 0, 0, 2,-2, 4, 0, 0, 0,    -0.11e-6,  0.00e-6],
    [ 1, 0,-2, 0,-3, 0, 0, 0,     0.11e-6,  0.00e-6],
    [ 1, 0,-2, 0,-1, 0, 0, 0,     0.11e-6,  0.00e-6],
], dtype=jnp.float64)

# e1 coefficients: 1 term
_EECT00_E1 = jnp.array([
    [ 0, 0, 0, 0, 1, 0, 0, 0,    -0.87e-6,  0.00e-6],
], dtype=jnp.float64)


def eect00(date1, date2):
    """Equation of the equinoxes complementary terms, IAU 2000."""
    t = ((date1 - DJ00) + date2) / DJC

    # Fundamental arguments
    fa = jnp.array([
        fal03(t), falp03(t), faf03(t), fad03(t), faom03(t),
        fave03(t), fae03(t), fapa03(t),
    ])

    # e0 terms (t^0)
    nfa0 = _EECT00_E0[:, :8]
    s0_sin = _EECT00_E0[:, 8]
    s0_cos = _EECT00_E0[:, 9]
    arg0 = nfa0 @ fa
    s0 = jnp.sum(s0_sin * jnp.sin(arg0) + s0_cos * jnp.cos(arg0))

    # e1 terms (t^1)
    nfa1 = _EECT00_E1[:, :8]
    s1_sin = _EECT00_E1[:, 8]
    s1_cos = _EECT00_E1[:, 9]
    arg1 = nfa1 @ fa
    s1 = jnp.sum(s1_sin * jnp.sin(arg1) + s1_cos * jnp.cos(arg1))

    return (s0 + s1 * t) * DAS2R


# ============================================================================
# Equation of the equinoxes
# ============================================================================

def ee00(date1, date2, epsa, dpsi):
    """Equation of the equinoxes, IAU 2000."""
    return dpsi * jnp.cos(epsa) + eect00(date1, date2)


def ee00a(date1, date2):
    """Equation of the equinoxes, IAU 2000A."""
    dpsipr, depspr = pr00(date1, date2)
    epsa = obl80(date1, date2) + depspr
    dpsi, deps = nut00a(date1, date2)
    return ee00(date1, date2, epsa, dpsi)


def ee00b(date1, date2):
    """Equation of the equinoxes, IAU 2000B."""
    dpsipr, depspr = pr00(date1, date2)
    epsa = obl80(date1, date2) + depspr
    dpsi, deps = nut00b(date1, date2)
    return ee00(date1, date2, epsa, dpsi)


def eqeq94(date1, date2):
    """Equation of the equinoxes, IAU 1994."""
    t = ((date1 - DJ00) + date2) / DJC
    om = anpm((450160.280 + (-482890.539 + (7.455 + 0.008 * t) * t) * t) * DAS2R
              + jnp.fmod(-5.0 * t, 1.0) * D2PI)
    dpsi, deps = nut80(date1, date2)
    eps0 = obl80(date1, date2)
    return dpsi * jnp.cos(eps0) + DAS2R * (0.00264 * jnp.sin(om) + 0.000063 * jnp.sin(om + om))


# ============================================================================
# CIO locator s (IAU 2000 and 2006)
# ============================================================================

# s00 and s06 share the same fundamental argument structure.
# s0: 33 terms, s1: 3 terms, s2: 25 terms, s3: 4 terms, s4: 1 term

def _s_series(t, x, y, sp, s0_table, s1_table, s2_table, s3_table, s4_table):
    """Evaluate the CIO locator s series."""
    fa = jnp.array([
        fal03(t), falp03(t), faf03(t), fad03(t), faom03(t),
        fave03(t), fae03(t), fapa03(t),
    ])

    def _eval_group(table):
        nfa = table[:, :8]
        sc = table[:, 8]
        cc = table[:, 9]
        arg = nfa @ fa
        return jnp.sum(sc * jnp.sin(arg) + cc * jnp.cos(arg))

    w0 = sp[0] + _eval_group(s0_table)
    w1 = sp[1] + _eval_group(s1_table)
    w2 = sp[2] + _eval_group(s2_table)
    w3 = sp[3] + _eval_group(s3_table)
    w4 = sp[4] + _eval_group(s4_table)
    w5 = sp[5]

    return (w0 + (w1 + (w2 + (w3 + (w4 + w5 * t) * t) * t) * t) * t) * DAS2R - x * y / 2.0


# --- S00 tables ---

_S00_SP = jnp.array([94.00e-6, 3808.35e-6, -119.94e-6, -72574.09e-6, 27.70e-6, 15.61e-6])

_S00_S0 = jnp.array([
    [ 0, 0, 0, 0, 1, 0, 0, 0, -2640.73e-6,   0.39e-6],
    [ 0, 0, 0, 0, 2, 0, 0, 0,   -63.53e-6,   0.02e-6],
    [ 0, 0, 2,-2, 3, 0, 0, 0,   -11.75e-6,  -0.01e-6],
    [ 0, 0, 2,-2, 1, 0, 0, 0,   -11.21e-6,  -0.01e-6],
    [ 0, 0, 2,-2, 2, 0, 0, 0,     4.57e-6,   0.00e-6],
    [ 0, 0, 2, 0, 3, 0, 0, 0,    -2.02e-6,   0.00e-6],
    [ 0, 0, 2, 0, 1, 0, 0, 0,    -1.98e-6,   0.00e-6],
    [ 0, 0, 0, 0, 3, 0, 0, 0,     1.72e-6,   0.00e-6],
    [ 0, 1, 0, 0, 1, 0, 0, 0,     1.41e-6,   0.01e-6],
    [ 0, 1, 0, 0,-1, 0, 0, 0,     1.26e-6,   0.01e-6],
    [ 1, 0, 0, 0,-1, 0, 0, 0,     0.63e-6,   0.00e-6],
    [ 1, 0, 0, 0, 1, 0, 0, 0,     0.63e-6,   0.00e-6],
    [ 0, 1, 2,-2, 3, 0, 0, 0,    -0.46e-6,   0.00e-6],
    [ 0, 1, 2,-2, 1, 0, 0, 0,    -0.45e-6,   0.00e-6],
    [ 0, 0, 4,-4, 4, 0, 0, 0,    -0.36e-6,   0.00e-6],
    [ 0, 0, 1,-1, 1,-8,12, 0,     0.24e-6,   0.12e-6],
    [ 0, 0, 2, 0, 0, 0, 0, 0,    -0.32e-6,   0.00e-6],
    [ 0, 0, 2, 0, 2, 0, 0, 0,    -0.28e-6,   0.00e-6],
    [ 1, 0, 2, 0, 3, 0, 0, 0,    -0.27e-6,   0.00e-6],
    [ 1, 0, 2, 0, 1, 0, 0, 0,    -0.26e-6,   0.00e-6],
    [ 0, 0, 2,-2, 0, 0, 0, 0,     0.21e-6,   0.00e-6],
    [ 0, 1,-2, 2,-3, 0, 0, 0,    -0.19e-6,   0.00e-6],
    [ 0, 1,-2, 2,-1, 0, 0, 0,    -0.18e-6,   0.00e-6],
    [ 0, 0, 0, 0, 0, 8,-13,-1,    0.10e-6,  -0.05e-6],
    [ 0, 0, 0, 2, 0, 0, 0, 0,    -0.15e-6,   0.00e-6],
    [ 2, 0,-2, 0,-1, 0, 0, 0,     0.14e-6,   0.00e-6],
    [ 0, 1, 2,-2, 2, 0, 0, 0,     0.14e-6,   0.00e-6],
    [ 1, 0, 0,-2, 1, 0, 0, 0,    -0.14e-6,   0.00e-6],
    [ 1, 0, 0,-2,-1, 0, 0, 0,    -0.14e-6,   0.00e-6],
    [ 0, 0, 4,-2, 4, 0, 0, 0,    -0.13e-6,   0.00e-6],
    [ 0, 0, 2,-2, 4, 0, 0, 0,     0.11e-6,   0.00e-6],
    [ 1, 0,-2, 0,-3, 0, 0, 0,    -0.11e-6,   0.00e-6],
    [ 1, 0,-2, 0,-1, 0, 0, 0,    -0.11e-6,   0.00e-6],
], dtype=jnp.float64)

_S00_S1 = jnp.array([
    [ 0, 0, 0, 0, 2, 0, 0, 0,    -0.07e-6,   3.57e-6],
    [ 0, 0, 0, 0, 1, 0, 0, 0,     1.71e-6,  -0.03e-6],
    [ 0, 0, 2,-2, 3, 0, 0, 0,     0.00e-6,   0.48e-6],
], dtype=jnp.float64)

_S00_S2 = jnp.array([
    [ 0, 0, 0, 0, 1, 0, 0, 0,   743.53e-6,  -0.17e-6],
    [ 0, 0, 2,-2, 2, 0, 0, 0,    56.91e-6,   0.06e-6],
    [ 0, 0, 2, 0, 2, 0, 0, 0,     9.84e-6,  -0.01e-6],
    [ 0, 0, 0, 0, 2, 0, 0, 0,    -8.85e-6,   0.01e-6],
    [ 0, 1, 0, 0, 0, 0, 0, 0,    -6.38e-6,  -0.05e-6],
    [ 1, 0, 0, 0, 0, 0, 0, 0,    -3.07e-6,   0.00e-6],
    [ 0, 1, 2,-2, 2, 0, 0, 0,     2.23e-6,   0.00e-6],
    [ 0, 0, 2, 0, 1, 0, 0, 0,     1.67e-6,   0.00e-6],
    [ 1, 0, 2, 0, 2, 0, 0, 0,     1.30e-6,   0.00e-6],
    [ 0, 1,-2, 2,-2, 0, 0, 0,     0.93e-6,   0.00e-6],
    [ 1, 0, 0,-2, 0, 0, 0, 0,     0.68e-6,   0.00e-6],
    [ 0, 0, 2,-2, 1, 0, 0, 0,    -0.55e-6,   0.00e-6],
    [ 1, 0,-2, 0,-2, 0, 0, 0,     0.53e-6,   0.00e-6],
    [ 0, 0, 0, 2, 0, 0, 0, 0,    -0.27e-6,   0.00e-6],
    [ 1, 0, 0, 0, 1, 0, 0, 0,    -0.27e-6,   0.00e-6],
    [ 1, 0,-2,-2,-2, 0, 0, 0,    -0.26e-6,   0.00e-6],
    [ 1, 0, 0, 0,-1, 0, 0, 0,    -0.25e-6,   0.00e-6],
    [ 1, 0, 2, 0, 1, 0, 0, 0,     0.22e-6,   0.00e-6],
    [ 2, 0, 0,-2, 0, 0, 0, 0,    -0.21e-6,   0.00e-6],
    [ 2, 0,-2, 0,-1, 0, 0, 0,     0.20e-6,   0.00e-6],
    [ 0, 0, 2, 2, 2, 0, 0, 0,     0.17e-6,   0.00e-6],
    [ 2, 0, 2, 0, 2, 0, 0, 0,     0.13e-6,   0.00e-6],
    [ 2, 0, 0, 0, 0, 0, 0, 0,    -0.13e-6,   0.00e-6],
    [ 1, 0, 2,-2, 2, 0, 0, 0,    -0.12e-6,   0.00e-6],
    [ 0, 0, 2, 0, 0, 0, 0, 0,    -0.11e-6,   0.00e-6],
], dtype=jnp.float64)

_S00_S3 = jnp.array([
    [ 0, 0, 0, 0, 1, 0, 0, 0,     0.30e-6, -23.51e-6],
    [ 0, 0, 2,-2, 2, 0, 0, 0,    -0.03e-6,  -1.39e-6],
    [ 0, 0, 2, 0, 2, 0, 0, 0,    -0.01e-6,  -0.24e-6],
    [ 0, 0, 0, 0, 2, 0, 0, 0,     0.00e-6,   0.22e-6],
], dtype=jnp.float64)

_S00_S4 = jnp.array([
    [ 0, 0, 0, 0, 1, 0, 0, 0,    -0.26e-6,  -0.01e-6],
], dtype=jnp.float64)


def s00(date1, date2, x, y):
    """The CIO locator s, positioning the Celestial Intermediate Origin on
    the equator of the CIP, given the CIP's X,Y coordinates. IAU 2000."""
    t = ((date1 - DJ00) + date2) / DJC
    return _s_series(t, x, y, _S00_SP, _S00_S0, _S00_S1, _S00_S2, _S00_S3, _S00_S4)


# --- S06 tables (slightly different polynomial and a few term coefficients) ---

_S06_SP = jnp.array([94.00e-6, 3808.65e-6, -122.68e-6, -72574.11e-6, 27.98e-6, 15.62e-6])

_S06_S1 = jnp.array([
    [ 0, 0, 0, 0, 2, 0, 0, 0,    -0.07e-6,   3.57e-6],
    [ 0, 0, 0, 0, 1, 0, 0, 0,     1.73e-6,  -0.03e-6],
    [ 0, 0, 2,-2, 3, 0, 0, 0,     0.00e-6,   0.48e-6],
], dtype=jnp.float64)

_S06_S2 = jnp.array([
    [ 0, 0, 0, 0, 1, 0, 0, 0,   743.52e-6,  -0.17e-6],
    [ 0, 0, 2,-2, 2, 0, 0, 0,    56.91e-6,   0.06e-6],
    [ 0, 0, 2, 0, 2, 0, 0, 0,     9.84e-6,  -0.01e-6],
    [ 0, 0, 0, 0, 2, 0, 0, 0,    -8.85e-6,   0.01e-6],
    [ 0, 1, 0, 0, 0, 0, 0, 0,    -6.38e-6,  -0.05e-6],
    [ 1, 0, 0, 0, 0, 0, 0, 0,    -3.07e-6,   0.00e-6],
    [ 0, 1, 2,-2, 2, 0, 0, 0,     2.23e-6,   0.00e-6],
    [ 0, 0, 2, 0, 1, 0, 0, 0,     1.67e-6,   0.00e-6],
    [ 1, 0, 2, 0, 2, 0, 0, 0,     1.30e-6,   0.00e-6],
    [ 0, 1,-2, 2,-2, 0, 0, 0,     0.93e-6,   0.00e-6],
    [ 1, 0, 0,-2, 0, 0, 0, 0,     0.68e-6,   0.00e-6],
    [ 0, 0, 2,-2, 1, 0, 0, 0,    -0.55e-6,   0.00e-6],
    [ 1, 0,-2, 0,-2, 0, 0, 0,     0.53e-6,   0.00e-6],
    [ 0, 0, 0, 2, 0, 0, 0, 0,    -0.27e-6,   0.00e-6],
    [ 1, 0, 0, 0, 1, 0, 0, 0,    -0.27e-6,   0.00e-6],
    [ 1, 0,-2,-2,-2, 0, 0, 0,    -0.26e-6,   0.00e-6],
    [ 1, 0, 0, 0,-1, 0, 0, 0,    -0.25e-6,   0.00e-6],
    [ 1, 0, 2, 0, 1, 0, 0, 0,     0.22e-6,   0.00e-6],
    [ 2, 0, 0,-2, 0, 0, 0, 0,    -0.21e-6,   0.00e-6],
    [ 2, 0,-2, 0,-1, 0, 0, 0,     0.20e-6,   0.00e-6],
    [ 0, 0, 2, 2, 2, 0, 0, 0,     0.17e-6,   0.00e-6],
    [ 2, 0, 2, 0, 2, 0, 0, 0,     0.13e-6,   0.00e-6],
    [ 2, 0, 0, 0, 0, 0, 0, 0,    -0.13e-6,   0.00e-6],
    [ 1, 0, 2,-2, 2, 0, 0, 0,    -0.12e-6,   0.00e-6],
    [ 0, 0, 2, 0, 0, 0, 0, 0,    -0.11e-6,   0.00e-6],
], dtype=jnp.float64)

_S06_S3 = jnp.array([
    [ 0, 0, 0, 0, 1, 0, 0, 0,     0.30e-6, -23.42e-6],
    [ 0, 0, 2,-2, 2, 0, 0, 0,    -0.03e-6,  -1.46e-6],
    [ 0, 0, 2, 0, 2, 0, 0, 0,    -0.01e-6,  -0.25e-6],
    [ 0, 0, 0, 0, 2, 0, 0, 0,     0.00e-6,   0.23e-6],
], dtype=jnp.float64)


def s06(date1, date2, x, y):
    """The CIO locator s, IAU 2006."""
    t = ((date1 - DJ00) + date2) / DJC
    # s06 uses the same s0 and s4 tables as s00, but different sp, s1, s2, s3
    return _s_series(t, x, y, _S06_SP, _S00_S0, _S06_S1, _S06_S2, _S06_S3, _S00_S4)


# ============================================================================
# CIO locator s convenience wrappers
# ============================================================================

def s00a(date1, date2):
    """CIO locator s, IAU 2000A, using IAU 2000A precession-nutation."""
    rbpn = pnm00a(date1, date2)
    x, y = bpn2xy(rbpn)
    return s00(date1, date2, x, y)


def s00b(date1, date2):
    """CIO locator s, IAU 2000B."""
    rbpn = pnm00b(date1, date2)
    x, y = bpn2xy(rbpn)
    return s00(date1, date2, x, y)


def s06a(date1, date2):
    """CIO locator s, IAU 2006/2000A."""
    rnpb = pnm06a(date1, date2)
    x, y = bpn2xy(rnpb)
    return s06(date1, date2, x, y)


# ============================================================================
# Equation of the origins
# ============================================================================

def eors(rnpb, s):
    """Equation of the origins, given NPB matrix and s."""
    x = rnpb[2, 0]
    ax = x / (1.0 + rnpb[2, 2])
    xs = 1.0 - ax * x
    ys = -ax * rnpb[2, 1]
    zs = -x
    p = rnpb[0, 0] * xs + rnpb[0, 1] * ys + rnpb[0, 2] * zs
    q = rnpb[1, 0] * xs + rnpb[1, 1] * ys + rnpb[1, 2] * zs
    return s - jnp.arctan2(q, p)


# ============================================================================
# Celestial-to-intermediate frame matrices
# ============================================================================

def c2ixys(x, y, s):
    """Form the celestial to intermediate-frame matrix given CIP X,Y and CIO locator s."""
    r2 = x * x + y * y
    e = jnp.where(r2 > 0.0, jnp.arctan2(y, x), 0.0)
    d = jnp.arctan(jnp.sqrt(r2 / (1.0 - r2)))
    r = ir()
    r = rz(e, r)
    r = ry(d, r)
    r = rz(-(e + s), r)
    return r


def c2ixy(date1, date2, x, y):
    """Celestial to intermediate-frame matrix, given X,Y and date (uses s00)."""
    return c2ixys(x, y, s00(date1, date2, x, y))


def c2ibpn(date1, date2, rbpn):
    """Celestial to intermediate-frame matrix, given bias-precession-nutation matrix."""
    x, y = bpn2xy(rbpn)
    return c2ixy(date1, date2, x, y)


def c2i00a(date1, date2):
    """Celestial to intermediate-frame matrix, IAU 2000A."""
    rbpn = pnm00a(date1, date2)
    return c2ibpn(date1, date2, rbpn)


def c2i00b(date1, date2):
    """Celestial to intermediate-frame matrix, IAU 2000B."""
    rbpn = pnm00b(date1, date2)
    return c2ibpn(date1, date2, rbpn)


def c2i06a(date1, date2):
    """Celestial to intermediate-frame matrix, IAU 2006/2000A."""
    rbpn = pnm06a(date1, date2)
    x, y = bpn2xy(rbpn)
    s = s06(date1, date2, x, y)
    return c2ixys(x, y, s)


# ============================================================================
# Polar motion matrix
# ============================================================================

def pom00(xp, yp, sp):
    """Form the matrix of polar motion for a given date, IAU 2000."""
    r = ir()
    r = rz(sp, r)
    r = ry(-xp, r)
    r = rx(-yp, r)
    return r


# ============================================================================
# Celestial-to-terrestrial matrices
# ============================================================================

def c2tcio(rc2i, era, rpom):
    """Form celestial-to-terrestrial matrix given CIO-based components."""
    r = rz(era, rc2i)
    return rxr(rpom, r)


def c2teqx(rbpn, gst, rpom):
    """Form celestial-to-terrestrial matrix given equinox-based components."""
    r = rz(gst, rbpn)
    return rxr(rpom, r)


def c2t00a(tta, ttb, uta, utb, xp, yp):
    """Form celestial-to-terrestrial matrix, IAU 2000A, CIO-based."""
    rc2i = c2i00a(tta, ttb)
    era = _era00_import(uta, utb)
    sp = sp00(tta, ttb)
    rpom = pom00(xp, yp, sp)
    return c2tcio(rc2i, era, rpom)


def c2t00b(tta, ttb, uta, utb, xp, yp):
    """Form celestial-to-terrestrial matrix, IAU 2000B, CIO-based."""
    rc2i = c2i00b(tta, ttb)
    era = _era00_import(uta, utb)
    rpom = pom00(xp, yp, 0.0)
    return c2tcio(rc2i, era, rpom)


def c2t06a(tta, ttb, uta, utb, xp, yp):
    """Form celestial-to-terrestrial matrix, IAU 2006/2000A, CIO-based."""
    rc2i = c2i06a(tta, ttb)
    era = _era00_import(uta, utb)
    sp = sp00(tta, ttb)
    rpom = pom00(xp, yp, sp)
    return c2tcio(rc2i, era, rpom)


def c2tpe(tta, ttb, uta, utb, dpsi, deps, xp, yp):
    """Form celestial-to-terrestrial matrix, given nutation, equinox-based, IAU 2000."""
    _, _, epsa, rb, rp, rbp, rn, rbpn = pn00(tta, ttb, dpsi, deps)
    gmst = _gmst00_import(uta, utb, tta, ttb)
    ee = ee00(tta, ttb, epsa, dpsi)
    sp = sp00(tta, ttb)
    rpom = pom00(xp, yp, sp)
    return c2teqx(rbpn, gmst + ee, rpom)


def c2txy(tta, ttb, uta, utb, x, y, xp, yp):
    """Form celestial-to-terrestrial matrix, given CIP X,Y, CIO-based, IAU 2000."""
    rc2i = c2ixy(tta, ttb, x, y)
    era = _era00_import(uta, utb)
    sp = sp00(tta, ttb)
    rpom = pom00(xp, yp, sp)
    return c2tcio(rc2i, era, rpom)


# ============================================================================
# X,Y coordinates of the CIP
# ============================================================================

def xys00a(date1, date2):
    """X,Y coordinates of the CIP and CIO locator s, IAU 2000A."""
    rbpn = pnm00a(date1, date2)
    x, y = bpn2xy(rbpn)
    s = s00(date1, date2, x, y)
    return x, y, s


def xys00b(date1, date2):
    """X,Y coordinates of the CIP and CIO locator s, IAU 2000B."""
    rbpn = pnm00b(date1, date2)
    x, y = bpn2xy(rbpn)
    s = s00(date1, date2, x, y)
    return x, y, s


def xys06a(date1, date2):
    """X,Y coordinates of the CIP and CIO locator s, IAU 2006/2000A."""
    rbpn = pnm06a(date1, date2)
    x, y = bpn2xy(rbpn)
    s = s06(date1, date2, x, y)
    return x, y, s


# ============================================================================
# Greenwich sidereal time functions (depend on precession-nutation)
# ============================================================================

def _era00_import(dj1, dj2):
    """Import era00 from time module (avoid circular import)."""
    from so_pointjax.erfa._core.time import era00
    return era00(dj1, dj2)


def _gmst00_import(uta, utb, tta, ttb):
    """Import gmst00 from time module."""
    from so_pointjax.erfa._core.time import gmst00
    return gmst00(uta, utb, tta, ttb)


def _gmst06_import(uta, utb, tta, ttb):
    """Import gmst06 from time module."""
    from so_pointjax.erfa._core.time import gmst06
    return gmst06(uta, utb, tta, ttb)


def _gmst82_import(dj1, dj2):
    """Import gmst82 from time module."""
    from so_pointjax.erfa._core.time import gmst82
    return gmst82(dj1, dj2)


def gst00a(uta, utb, tta, ttb):
    """Greenwich apparent sidereal time, IAU 2000A."""
    gmst = _gmst00_import(uta, utb, tta, ttb)
    ee = ee00a(tta, ttb)
    return anp(gmst + ee)


def gst00b(uta, utb):
    """Greenwich apparent sidereal time, IAU 2000B."""
    gmst = _gmst00_import(uta, utb, uta, utb)
    ee = ee00b(uta, utb)
    return anp(gmst + ee)


def gst06(uta, utb, tta, ttb, rnpb):
    """Greenwich apparent sidereal time, IAU 2006, given NPB matrix."""
    x, y = bpn2xy(rnpb)
    s = s06(tta, ttb, x, y)
    era = _era00_import(uta, utb)
    eo = eors(rnpb, s)
    return anp(era - eo)


def gst06a(uta, utb, tta, ttb):
    """Greenwich apparent sidereal time, IAU 2006/2000A."""
    rnpb = pnm06a(tta, ttb)
    return gst06(uta, utb, tta, ttb, rnpb)


def gst94(uta, utb):
    """Greenwich apparent sidereal time, IAU 1982/1994."""
    gmst = _gmst82_import(uta, utb)
    ee = eqeq94(uta, utb)
    return anp(gmst + ee)


# ============================================================================
# __all__ export list
# ============================================================================

__all__ = [
    # Fundamental arguments
    "fal03", "falp03", "faf03", "fad03", "faom03",
    "fame03", "fave03", "fae03", "fama03", "faju03", "fasa03", "faur03", "fane03",
    "fapa03",
    # Obliquity
    "obl80", "obl06",
    # Precession rate
    "pr00",
    # Frame bias
    "bi00",
    # TIO locator
    "sp00",
    # Precession angles and matrices
    "prec76", "pmat76", "pfw06", "pmat06",
    # Fukushima-Williams
    "fw2m", "fw2xy", "bpn2xy",
    # Nutation matrix builder
    "numat",
    # Bias-precession
    "bp00", "bp06",
    # Nutation models
    "nut80", "nut00b", "nut00a", "nut06a",
    # Nutation matrices
    "nutm80", "num00a", "num00b", "num06a",
    # Precession-nutation composites
    "pn00", "pn00a", "pn00b", "pn06", "pn06a",
    # P-N matrices
    "pnm00a", "pnm00b", "pnm06a", "pnm80",
    # Equation of equinoxes
    "eect00", "ee00", "ee00a", "ee00b", "eqeq94",
    # CIO locator s
    "s00", "s00a", "s00b", "s06", "s06a",
    # Equation of origins
    "eors",
    # Celestial-to-intermediate
    "c2ixys", "c2ixy", "c2ibpn", "c2i00a", "c2i00b", "c2i06a",
    # Polar motion
    "pom00",
    # Celestial-to-terrestrial
    "c2tcio", "c2teqx", "c2t00a", "c2t00b", "c2t06a", "c2tpe", "c2txy",
    # CIP coordinates
    "xys00a", "xys00b", "xys06a",
    # Greenwich sidereal time
    "gst00a", "gst00b", "gst06", "gst06a", "gst94",
]
