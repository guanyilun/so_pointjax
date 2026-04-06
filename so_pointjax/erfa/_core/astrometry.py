"""Astrometry functions, ported from ERFA C library.

Covers: stellar aberration, light deflection, proper motion/parallax,
refraction, observer position, ASTROM context setup, coordinate transforms
(ICRS <-> CIRS <-> observed), and star catalog <-> pv-vector conversions.
"""

import jax.numpy as jnp

from so_pointjax.erfa._core.constants import (
    SRS, AULT, DAYSEC, DAU, DJY, DJ00, DAS2R, D2PI, DJM, DC,
    WGS84,
)
from so_pointjax.erfa._core.vector import (
    pdp, pxp, pn, pm, pmp, ppp, ppsp, cp, sxp, ir, rx, ry, rz,
    rxp, trxp, trxpv, c2s, s2c, s2pv, pv2s, zp,
)
from so_pointjax.erfa._core.angles import anp, anpm
from so_pointjax.erfa._types import ASTROM, LDBODY


# ============================================================================
# Fundamental astrometric effects
# ============================================================================

def ab(pnat, v, s, bm1):
    """Apply aberration to transform natural direction into proper direction.

    Parameters
    ----------
    pnat : ndarray (3,)
        Natural direction to source (unit vector)
    v : ndarray (3,)
        Observer barycentric velocity in units of c
    s : float
        Distance from Sun to observer (au)
    bm1 : float
        sqrt(1-|v|^2): reciprocal of Lorentz factor

    Returns
    -------
    ppr : ndarray (3,)
        Proper direction to source (unit vector)
    """
    pdv = pdp(pnat, v)
    w1 = 1.0 + pdv / (1.0 + bm1)
    w2 = SRS / s
    p = pnat * bm1 + w1 * v + w2 * (v - pdv * pnat)
    r = jnp.sqrt(pdp(p, p))
    return p / r


def ld(bm, p, q, e, em, dlim):
    """Apply light deflection by a solar-system body.

    Parameters
    ----------
    bm : float
        Mass of gravitating body (solar masses)
    p : ndarray (3,)
        Direction from observer to source (unit vector)
    q : ndarray (3,)
        Direction from body to source (unit vector)
    e : ndarray (3,)
        Direction from body to observer (unit vector)
    em : float
        Distance from body to observer (au)
    dlim : float
        Deflection limiter

    Returns
    -------
    p1 : ndarray (3,)
        Observer to deflected source (unit vector)
    """
    qpe = q + e
    qdqpe = pdp(q, qpe)
    w = bm * SRS / em / jnp.maximum(qdqpe, dlim)
    eq = pxp(e, q)
    peq = pxp(p, eq)
    return p + w * peq


def ldn(n, b, ob, sc):
    """Apply light deflection by multiple solar-system bodies.

    Parameters
    ----------
    n : int
        Number of bodies
    b : list of LDBODY
        Data for each body
    ob : ndarray (3,)
        Barycentric position of observer (au)
    sc : ndarray (3,)
        Observer to star coord direction (unit vector)

    Returns
    -------
    sn : ndarray (3,)
        Observer to deflected star (unit vector)
    """
    # Light time for 1 au (days)
    CR = AULT / DAYSEC

    sn = cp(sc)
    for i in range(n):
        # Body to observer vector at epoch of observation (au)
        v = ob - b[i].pv[0]

        # Minus the time since the light passed the body (days)
        dt = pdp(sn, v) * CR

        # Neutralize if the star is "behind" the observer
        dt = jnp.minimum(dt, 0.0)

        # Backtrack the body to the time the light was passing the body
        ev = v + (-dt) * b[i].pv[1]

        # Body to observer vector as magnitude and direction
        em_val, e_vec = pn(ev)

        # Apply light deflection for this body
        sn = ld(b[i].bm, sn, sn, e_vec, em_val, b[i].dl)

    return sn


def ldsun(p, e, em):
    """Deflection of starlight by the Sun.

    Parameters
    ----------
    p : ndarray (3,)
        Direction from observer to star (unit vector)
    e : ndarray (3,)
        Direction from Sun to observer (unit vector)
    em : float
        Distance from Sun to observer (au)

    Returns
    -------
    p1 : ndarray (3,)
        Observer to deflected star (unit vector)
    """
    em2 = jnp.maximum(em * em, 1.0)
    dlim = 1e-6 / em2
    return ld(1.0, p, p, e, em, dlim)


def pmpx(rc, dc, pr, pd, px, rv, pmt, pob):
    """Proper motion and parallax.

    Parameters
    ----------
    rc, dc : float
        ICRS RA, Dec at catalog epoch (radians)
    pr : float
        RA proper motion (radians/year; dRA/dt, not cos(Dec)*dRA/dt)
    pd : float
        Dec proper motion (radians/year)
    px : float
        Parallax (arcsec)
    rv : float
        Radial velocity (km/s, +ve if receding)
    pmt : float
        Proper motion time interval (SSB, Julian years)
    pob : ndarray (3,)
        SSB to observer vector (au)

    Returns
    -------
    pco : ndarray (3,)
        Coordinate direction (BCRS unit vector)
    """
    # Km/s to au/year
    VF = DAYSEC * DJM / DAU

    # Light time for 1 au, Julian years
    AULTY = AULT / DAYSEC / DJY

    # Spherical coordinates to unit vector
    sr = jnp.sin(rc)
    cr = jnp.cos(rc)
    sd = jnp.sin(dc)
    cd = jnp.cos(dc)
    x = cr * cd
    y = sr * cd
    z = sd
    p = jnp.array([x, y, z])

    # Proper motion time interval (y) including Roemer effect
    dt = pmt + pdp(p, pob) * AULTY

    # Space motion (radians per year)
    pxr = px * DAS2R
    w = VF * rv * pxr
    pdz = pd * z
    pm_vec = jnp.array([
        -pr * y - pdz * cr + w * x,
         pr * x - pdz * sr + w * y,
         pd * cd + w * z,
    ])

    # Coordinate direction of star (unit vector, BCRS)
    p = p + dt * pm_vec - pxr * pob
    _, pco = pn(p)
    return pco


def refco(phpa, tc, rh, wl):
    """Determine the constants A and B in the atmospheric refraction model.

    dZ = A tan Z + B tan^3 Z

    Parameters
    ----------
    phpa : float
        Pressure at the observer (hPa = millibar)
    tc : float
        Ambient temperature at the observer (deg C)
    rh : float
        Relative humidity at the observer (range 0-1)
    wl : float
        Wavelength (micrometers)

    Returns
    -------
    refa : float
        tan Z coefficient (radians)
    refb : float
        tan^3 Z coefficient (radians)
    """
    optic = wl <= 100.0

    # Restrict parameters to safe values
    t = jnp.clip(tc, -150.0, 200.0)
    p = jnp.clip(phpa, 0.0, 10000.0)
    r = jnp.clip(rh, 0.0, 1.0)
    w = jnp.clip(wl, 0.1, 1e6)

    # Water vapour pressure at the observer
    ps = jnp.where(
        p > 0.0,
        jnp.power(10.0, (0.7859 + 0.03477 * t) / (1.0 + 0.00412 * t))
        * (1.0 + p * (4.5e-6 + 6e-10 * t * t)),
        0.0,
    )
    pw = jnp.where(p > 0.0, r * ps / (1.0 - (1.0 - r) * ps / jnp.maximum(p, 1e-30)), 0.0)

    # Refractive index minus 1 at the observer
    tk = t + 273.15
    wlsq = w * w
    gamma_optic = ((77.53484e-6 + (4.39108e-7 + 3.666e-9 / wlsq) / wlsq) * p
                   - 11.2684e-6 * pw) / tk
    gamma_radio = (77.6890e-6 * p - (6.3938e-6 - 0.375463 / tk) * pw) / tk
    gamma = jnp.where(optic, gamma_optic, gamma_radio)

    # Formula for beta from Stone, with empirical adjustments
    beta = 4.4474e-6 * tk
    beta = jnp.where(optic, beta, beta - 0.0074 * pw * beta)

    # Refraction constants from Green
    refa = gamma * (1.0 - beta)
    refb = -gamma * (beta - gamma / 2.0)

    return refa, refb


def pvtob(elong, phi, hm, xp, yp, sp, theta):
    """Position and velocity of a terrestrial observing station.

    Parameters
    ----------
    elong : float
        Longitude (radians, east +ve)
    phi : float
        Latitude (geodetic, radians)
    hm : float
        Height above ref. ellipsoid (geodetic, m)
    xp, yp : float
        Coordinates of the pole (radians)
    sp : float
        The TIO locator s' (radians)
    theta : float
        Earth rotation angle (radians)

    Returns
    -------
    pv : ndarray (2, 3)
        Position/velocity vector (m, m/s, CIRS)
    """
    from so_pointjax.erfa._core.geodetic import gd2gc
    from so_pointjax.erfa._core.precnut import pom00

    # Earth rotation rate in radians per UT1 second
    OM = 1.00273781191135448 * D2PI / DAYSEC

    # Geodetic to geocentric transformation (WGS84)
    xyzm = gd2gc(WGS84, elong, phi, hm)

    # Polar motion and TIO position
    rpm = pom00(xp, yp, sp)
    xyz = trxp(rpm, xyzm)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    # Functions of ERA
    s = jnp.sin(theta)
    c = jnp.cos(theta)

    # Position
    pos = jnp.array([c * x - s * y, s * x + c * y, z])

    # Velocity
    vel = jnp.array([OM * (-s * x - c * y), OM * (c * x - s * y), 0.0])

    return jnp.stack([pos, vel])


# ============================================================================
# ASTROM context-setup functions
# ============================================================================

def apcs(date1, date2, pv, ebpv, ehp):
    """For a space observer, prepare star-independent astrometry parameters.

    Parameters
    ----------
    date1, date2 : float
        TDB as a 2-part Julian Date
    pv : ndarray (2, 3)
        Observer's geocentric pos/vel (m, m/s)
    ebpv : ndarray (2, 3)
        Earth barycentric PV (au, au/day)
    ehp : ndarray (3,)
        Earth heliocentric P (au)

    Returns
    -------
    astrom : ASTROM
        Star-independent astrometry parameters
    """
    # au/d to m/s
    AUDMS = DAU / DAYSEC
    # Light time for 1 au (day)
    CR = AULT / DAYSEC

    # Time since reference epoch, years
    pmt = ((date1 - DJ00) + date2) / DJY

    # Adjust Earth ephemeris to observer
    dp = pv[0] / DAU
    dv = pv[1] / AUDMS
    pb = ebpv[0] + dp
    vb = ebpv[1] + dv
    ph = ehp + dp

    # Barycentric position of observer (au)
    eb = pb

    # Heliocentric direction and distance
    em, eh = pn(ph)

    # Barycentric vel. in units of c, and reciprocal of Lorentz factor
    v = vb * CR
    v2 = pdp(v, v)
    bm1 = jnp.sqrt(1.0 - v2)

    # Reset the NPB matrix to identity
    bpn = ir()

    return ASTROM(
        pmt=pmt, eb=eb, eh=eh, em=em, v=v, bm1=bm1, bpn=bpn,
        along=jnp.float64(0.0), phi=jnp.float64(0.0),
        xpl=jnp.float64(0.0), ypl=jnp.float64(0.0),
        sphi=jnp.float64(0.0), cphi=jnp.float64(0.0),
        diurab=jnp.float64(0.0), eral=jnp.float64(0.0),
        refa=jnp.float64(0.0), refb=jnp.float64(0.0),
    )


def apcg(date1, date2, ebpv, ehp):
    """For a geocentric observer, prepare star-independent astrometry parameters.

    Parameters
    ----------
    date1, date2 : float
        TDB as a 2-part Julian Date
    ebpv : ndarray (2, 3)
        Earth barycentric PV (au, au/day)
    ehp : ndarray (3,)
        Earth heliocentric P (au)

    Returns
    -------
    astrom : ASTROM
    """
    # Geocentric observer: zero geocentric pv
    pv = jnp.zeros((2, 3))
    return apcs(date1, date2, pv, ebpv, ehp)


def apci(date1, date2, ebpv, ehp, x, y, s):
    """For a terrestrial observer, prepare ICRS-to-CIRS astrometry parameters.

    Parameters
    ----------
    date1, date2 : float
        TDB as a 2-part Julian Date
    ebpv : ndarray (2, 3)
        Earth barycentric PV (au, au/day)
    ehp : ndarray (3,)
        Earth heliocentric P (au)
    x, y : float
        CIP X, Y (components of unit vector)
    s : float
        The CIO locator s (radians)

    Returns
    -------
    astrom : ASTROM
    """
    from so_pointjax.erfa._core.precnut import c2ixys

    # Geocentric ICRS-GCRS params
    astrom = apcg(date1, date2, ebpv, ehp)

    # CIO based BPN matrix
    bpn = c2ixys(x, y, s)

    return astrom._replace(bpn=bpn)


def apco(date1, date2, ebpv, ehp, x, y, s, theta,
         elong, phi, hm, xp, yp, sp_val, refa_val, refb_val):
    """For a terrestrial observer, prepare ICRS-to-observed astrometry parameters.

    Parameters
    ----------
    date1, date2 : float
        TDB as a 2-part Julian Date
    ebpv : ndarray (2, 3)
        Earth barycentric PV (au, au/day)
    ehp : ndarray (3,)
        Earth heliocentric P (au)
    x, y : float
        CIP X, Y
    s : float
        CIO locator s (radians)
    theta : float
        Earth rotation angle (radians)
    elong : float
        Longitude (radians, east +ve)
    phi : float
        Geodetic latitude (radians)
    hm : float
        Height above ellipsoid (m)
    xp, yp : float
        Polar motion coordinates (radians)
    sp_val : float
        TIO locator s' (radians)
    refa_val, refb_val : float
        Refraction constants A, B (radians)

    Returns
    -------
    astrom : ASTROM
    """
    from so_pointjax.erfa._core.precnut import c2ixys

    # Form the rotation matrix, CIRS to apparent [HA,Dec]
    r = ir()
    r = rz(theta + sp_val, r)
    r = ry(-xp, r)
    r = rx(-yp, r)
    r = rz(elong, r)

    # Solve for local Earth rotation angle
    a = r[0, 0]
    b = r[0, 1]
    eral = jnp.where((a != 0.0) | (b != 0.0), jnp.arctan2(b, a), 0.0)

    # Solve for polar motion [X,Y] with respect to local meridian
    c = r[0, 2]
    xpl_val = jnp.arctan2(c, jnp.sqrt(a * a + b * b))
    a2 = r[1, 2]
    b2 = r[2, 2]
    ypl_val = jnp.where((a2 != 0.0) | (b2 != 0.0), -jnp.arctan2(a2, b2), 0.0)

    # Adjusted longitude
    along = anpm(eral - theta)

    # Functions of latitude
    sphi = jnp.sin(phi)
    cphi = jnp.cos(phi)

    # CIO based BPN matrix
    bpn = c2ixys(x, y, s)

    # Observer's geocentric position and velocity (m, m/s, CIRS)
    pvc = pvtob(elong, phi, hm, xp, yp, sp_val, theta)

    # Rotate into GCRS
    pv = trxpv(bpn, pvc)

    # ICRS <-> GCRS parameters
    astrom = apcs(date1, date2, pv, ebpv, ehp)

    # Store all the terrestrial parameters and the BPN matrix
    return astrom._replace(
        bpn=bpn, along=along, phi=phi,
        xpl=xpl_val, ypl=ypl_val,
        sphi=sphi, cphi=cphi,
        diurab=jnp.float64(0.0),
        eral=eral,
        refa=refa_val, refb=refb_val,
    )


def aper(theta, astrom):
    """In an existing ASTROM, update the Earth rotation angle.

    Parameters
    ----------
    theta : float
        Earth rotation angle (radians)
    astrom : ASTROM
        Existing star-independent astrometry parameters

    Returns
    -------
    astrom : ASTROM
        Updated parameters (eral field)
    """
    return astrom._replace(eral=theta + astrom.along)


def apio(sp_val, theta, elong, phi, hm, xp, yp, refa_val, refb_val):
    """For a terrestrial observer, prepare CIRS-to-observed parameters.

    Parameters
    ----------
    sp_val : float
        TIO locator s' (radians)
    theta : float
        Earth rotation angle (radians)
    elong : float
        Longitude (radians, east +ve)
    phi : float
        Geodetic latitude (radians)
    hm : float
        Height above ellipsoid (m)
    xp, yp : float
        Polar motion coordinates (radians)
    refa_val, refb_val : float
        Refraction constants A, B (radians)

    Returns
    -------
    astrom : ASTROM
    """
    from so_pointjax.erfa._core.constants import CMPS

    # Form the rotation matrix, CIRS to apparent [HA,Dec]
    r = ir()
    r = rz(theta + sp_val, r)
    r = ry(-xp, r)
    r = rx(-yp, r)
    r = rz(elong, r)

    # Solve for local Earth rotation angle
    a = r[0, 0]
    b = r[0, 1]
    eral = jnp.where((a != 0.0) | (b != 0.0), jnp.arctan2(b, a), 0.0)

    # Solve for polar motion [X,Y] with respect to local meridian
    c = r[0, 2]
    xpl_val = jnp.arctan2(c, jnp.sqrt(a * a + b * b))
    a2 = r[1, 2]
    b2 = r[2, 2]
    ypl_val = jnp.where((a2 != 0.0) | (b2 != 0.0), -jnp.arctan2(a2, b2), 0.0)

    # Adjusted longitude
    along = anpm(eral - theta)

    # Functions of latitude
    sphi = jnp.sin(phi)
    cphi = jnp.cos(phi)

    # Observer's geocentric position and velocity (m, m/s, CIRS)
    pv_obs = pvtob(elong, phi, hm, xp, yp, sp_val, theta)

    # Magnitude of diurnal aberration vector
    diurab = jnp.sqrt(pv_obs[1, 0] ** 2 + pv_obs[1, 1] ** 2) / CMPS

    return ASTROM(
        pmt=jnp.float64(0.0),
        eb=jnp.zeros(3), eh=jnp.zeros(3), em=jnp.float64(0.0),
        v=jnp.zeros(3), bm1=jnp.float64(0.0), bpn=jnp.zeros((3, 3)),
        along=along, phi=phi,
        xpl=xpl_val, ypl=ypl_val,
        sphi=sphi, cphi=cphi,
        diurab=diurab, eral=eral,
        refa=refa_val, refb=refb_val,
    )


# ============================================================================
# Coordinate transform functions (quick, using pre-computed ASTROM)
# ============================================================================

def atciq(rc, dc, pr, pd, px, rv, astrom):
    """Quick ICRS to CIRS, given pre-computed ASTROM.

    Parameters
    ----------
    rc, dc : float
        ICRS RA, Dec at J2000.0 (radians)
    pr : float
        RA proper motion (radians/year)
    pd : float
        Dec proper motion (radians/year)
    px : float
        Parallax (arcsec)
    rv : float
        Radial velocity (km/s, +ve if receding)
    astrom : ASTROM

    Returns
    -------
    ri, di : float
        CIRS RA, Dec (radians)
    """
    # Proper motion and parallax, giving BCRS coordinate direction
    pco = pmpx(rc, dc, pr, pd, px, rv, astrom.pmt, astrom.eb)

    # Light deflection by the Sun, giving BCRS natural direction
    pnat = ldsun(pco, astrom.eh, astrom.em)

    # Aberration, giving GCRS proper direction
    ppr = ab(pnat, astrom.v, astrom.em, astrom.bm1)

    # Bias-precession-nutation, giving CIRS proper direction
    pi = rxp(astrom.bpn, ppr)

    # CIRS RA, Dec
    w, di = c2s(pi)
    ri = anp(w)
    return ri, di


def atciqz(rc, dc, astrom):
    """Quick ICRS to CIRS, zero parallax and proper motion.

    Parameters
    ----------
    rc, dc : float
        ICRS astrometric RA, Dec (radians)
    astrom : ASTROM

    Returns
    -------
    ri, di : float
        CIRS RA, Dec (radians)
    """
    # BCRS coordinate direction (unit vector)
    pco = s2c(rc, dc)

    # Light deflection by the Sun
    pnat = ldsun(pco, astrom.eh, astrom.em)

    # Aberration
    ppr = ab(pnat, astrom.v, astrom.em, astrom.bm1)

    # Bias-precession-nutation
    pi = rxp(astrom.bpn, ppr)

    # CIRS RA, Dec
    w, di = c2s(pi)
    ri = anp(w)
    return ri, di


def atciqn(rc, dc, pr, pd, px, rv, astrom, n, b):
    """Quick ICRS to CIRS with multiple light-deflecting bodies.

    Parameters
    ----------
    rc, dc : float
        ICRS RA, Dec at J2000.0 (radians)
    pr, pd : float
        Proper motions (radians/year)
    px : float
        Parallax (arcsec)
    rv : float
        Radial velocity (km/s)
    astrom : ASTROM
    n : int
        Number of bodies
    b : list of LDBODY

    Returns
    -------
    ri, di : float
        CIRS RA, Dec (radians)
    """
    # Proper motion and parallax
    pco = pmpx(rc, dc, pr, pd, px, rv, astrom.pmt, astrom.eb)

    # Light deflection by n bodies
    pnat = ldn(n, b, astrom.eb, pco)

    # Aberration
    ppr = ab(pnat, astrom.v, astrom.em, astrom.bm1)

    # Bias-precession-nutation
    pi = rxp(astrom.bpn, ppr)

    # CIRS RA, Dec
    w, di = c2s(pi)
    ri = anp(w)
    return ri, di


def _iterative_inversion(ppr, astrom, deflection_fn, n_iter):
    """Shared iterative inversion for aberration and light deflection.

    This implements the pattern: given forward result ppr, iteratively
    find the input that maps to ppr under the forward transform.
    """
    d = jnp.zeros(3)
    pnat = ppr  # initial guess

    for _ in range(n_iter):
        before = ppr - d
        _, before = pn(before)
        after = deflection_fn(before)
        d = after - before
        pnat = ppr - d
        _, pnat = pn(pnat)

    return pnat


def aticq(ri, di, astrom):
    """Quick CIRS to ICRS astrometric place, given pre-computed ASTROM.

    Parameters
    ----------
    ri, di : float
        CIRS RA, Dec (radians)
    astrom : ASTROM

    Returns
    -------
    rc, dc : float
        ICRS astrometric RA, Dec (radians)
    """
    # CIRS RA,Dec to Cartesian
    pi = s2c(ri, di)

    # Bias-precession-nutation, giving GCRS proper direction
    ppr = trxp(astrom.bpn, pi)

    # Iterative aberration inversion (2 iterations)
    def ab_fn(before):
        return ab(before, astrom.v, astrom.em, astrom.bm1)

    pnat = _iterative_inversion(ppr, astrom, ab_fn, 2)

    # Iterative light deflection inversion (5 iterations)
    def ld_fn(before):
        return ldsun(before, astrom.eh, astrom.em)

    pco = _iterative_inversion(pnat, astrom, ld_fn, 5)

    # ICRS astrometric RA, Dec
    w, dc = c2s(pco)
    rc = anp(w)
    return rc, dc


def aticqn(ri, di, astrom, n, b):
    """Quick CIRS to ICRS with multiple light-deflecting bodies.

    Parameters
    ----------
    ri, di : float
        CIRS RA, Dec (radians)
    astrom : ASTROM
    n : int
        Number of bodies
    b : list of LDBODY

    Returns
    -------
    rc, dc : float
        ICRS astrometric RA, Dec (radians)
    """
    # CIRS RA,Dec to Cartesian
    pi = s2c(ri, di)

    # Bias-precession-nutation, giving GCRS proper direction
    ppr = trxp(astrom.bpn, pi)

    # Iterative aberration inversion (2 iterations)
    def ab_fn(before):
        return ab(before, astrom.v, astrom.em, astrom.bm1)

    pnat = _iterative_inversion(ppr, astrom, ab_fn, 2)

    # Iterative light deflection inversion (5 iterations)
    def ldn_fn(before):
        return ldn(n, b, astrom.eb, before)

    pco = _iterative_inversion(pnat, astrom, ldn_fn, 5)

    # ICRS astrometric RA, Dec
    w, dc = c2s(pco)
    rc = anp(w)
    return rc, dc


def atioq(ri, di, astrom):
    """Quick CIRS to observed place transformation.

    Parameters
    ----------
    ri, di : float
        CIRS right ascension and declination (radians)
    astrom : ASTROM

    Returns
    -------
    aob : float
        Observed azimuth (radians: N=0, E=90)
    zob : float
        Observed zenith distance (radians)
    hob : float
        Observed hour angle (radians)
    dob : float
        Observed declination (radians)
    rob : float
        Observed right ascension (CIO-based, radians)
    """
    CELMIN = 1e-6
    SELMIN = 0.05

    # CIRS RA,Dec to Cartesian -HA,Dec
    v = s2c(ri - astrom.eral, di)
    x = v[0]
    y = v[1]
    z = v[2]

    # Polar motion
    sx = jnp.sin(astrom.xpl)
    cx = jnp.cos(astrom.xpl)
    sy = jnp.sin(astrom.ypl)
    cy = jnp.cos(astrom.ypl)
    xhd = cx * x + sx * z
    yhd = sx * sy * x + cy * y - cx * sy * z
    zhd = -sx * cy * x + sy * y + cx * cy * z

    # Diurnal aberration
    f = 1.0 - astrom.diurab * yhd
    xhdt = f * xhd
    yhdt = f * (yhd + astrom.diurab)
    zhdt = f * zhd

    # Cartesian -HA,Dec to Cartesian Az,El (S=0,E=90)
    xaet = astrom.sphi * xhdt - astrom.cphi * zhdt
    yaet = yhdt
    zaet = astrom.cphi * xhdt + astrom.sphi * zhdt

    # Azimuth (N=0,E=90)
    azobs = jnp.where(
        (xaet != 0.0) | (yaet != 0.0),
        jnp.arctan2(yaet, -xaet),
        0.0,
    )

    # Refraction
    r = jnp.sqrt(xaet * xaet + yaet * yaet)
    r = jnp.maximum(r, CELMIN)
    z_ref = jnp.maximum(zaet, SELMIN)

    # A*tan(z)+B*tan^3(z) model, with Newton-Raphson correction
    tz = r / z_ref
    w = astrom.refb * tz * tz
    delta = (astrom.refa + w) * tz / (1.0 + (astrom.refa + 3.0 * w) / (z_ref * z_ref))

    # Apply the change, giving observed vector
    cosdel = 1.0 - delta * delta / 2.0
    f = cosdel - delta * z_ref / r
    xaeo = xaet * f
    yaeo = yaet * f
    zaeo = cosdel * zaet + delta * r

    # Observed ZD
    zdobs = jnp.arctan2(jnp.sqrt(xaeo * xaeo + yaeo * yaeo), zaeo)

    # Az/El vector to HA,Dec vector (both right-handed)
    v_out = jnp.array([
        astrom.sphi * xaeo + astrom.cphi * zaeo,
        yaeo,
        -astrom.cphi * xaeo + astrom.sphi * zaeo,
    ])

    # To spherical -HA,Dec
    hmobs, dcobs = c2s(v_out)

    # Right ascension (with respect to CIO)
    raobs = astrom.eral + hmobs

    aob = anp(azobs)
    zob = zdobs
    hob = -hmobs
    dob = dcobs
    rob = anp(raobs)
    return aob, zob, hob, dob, rob


def atoiq(type_str, ob1, ob2, astrom):
    """Quick observed place to CIRS.

    Parameters
    ----------
    type_str : str
        Type of coordinates: "R" (RA,Dec), "H" (HA,Dec), or "A" (Az,ZD)
    ob1, ob2 : float
        Observed Az/RA/HA and ZD/Dec (radians)
    astrom : ASTROM

    Returns
    -------
    ri, di : float
        CIRS RA, Dec (radians)
    """
    SELMIN = 0.05

    sphi = astrom.sphi
    cphi = astrom.cphi

    c = type_str[0].upper()

    if c == 'A':
        # Az,ZD to Cartesian (S=0,E=90)
        ce = jnp.sin(ob2)
        xaeo = -jnp.cos(ob1) * ce
        yaeo = jnp.sin(ob1) * ce
        zaeo = jnp.cos(ob2)
    else:
        c1 = ob1
        if c == 'R':
            c1 = astrom.eral - ob1
        # To Cartesian -HA,Dec
        v = s2c(-c1, ob2)
        xmhdo = v[0]
        ymhdo = v[1]
        zmhdo = v[2]
        # To Cartesian Az,El (S=0,E=90)
        xaeo = sphi * xmhdo - cphi * zmhdo
        yaeo = ymhdo
        zaeo = cphi * xmhdo + sphi * zmhdo

    # Azimuth (S=0,E=90)
    az = jnp.where((xaeo != 0.0) | (yaeo != 0.0), jnp.arctan2(yaeo, xaeo), 0.0)

    # Sine of observed ZD, and observed ZD
    sz = jnp.sqrt(xaeo * xaeo + yaeo * yaeo)
    zdo = jnp.arctan2(sz, zaeo)

    # Refraction
    refa_v = astrom.refa
    refb_v = astrom.refb
    tz = sz / jnp.maximum(zaeo, SELMIN)
    dref = (refa_v + refb_v * tz * tz) * tz
    zdt = zdo + dref

    # To Cartesian Az,ZD
    ce = jnp.sin(zdt)
    xaet = jnp.cos(az) * ce
    yaet = jnp.sin(az) * ce
    zaet = jnp.cos(zdt)

    # Cartesian Az,ZD to Cartesian -HA,Dec
    xmhda = sphi * xaet + cphi * zaet
    ymhda = yaet
    zmhda = -cphi * xaet + sphi * zaet

    # Diurnal aberration
    f = 1.0 + astrom.diurab * ymhda
    xhd = f * xmhda
    yhd = f * (ymhda - astrom.diurab)
    zhd = f * zmhda

    # Polar motion
    sx = jnp.sin(astrom.xpl)
    cx = jnp.cos(astrom.xpl)
    sy = jnp.sin(astrom.ypl)
    cy = jnp.cos(astrom.ypl)
    v_out = jnp.array([
        cx * xhd + sx * sy * yhd - sx * cy * zhd,
        cy * yhd + sy * zhd,
        sx * xhd - cx * sy * yhd + cx * cy * zhd,
    ])

    # To spherical -HA,Dec
    hma, di = c2s(v_out)

    # Right ascension
    ri = anp(astrom.eral + hma)
    return ri, di


# ============================================================================
# Space motion: star catalog <-> pv-vector
# ============================================================================

def starpv(ra, dec, pmr, pmd, px, rv):
    """Convert star catalog coordinates to position+velocity vector.

    Parameters
    ----------
    ra, dec : float
        Right ascension, declination (radians)
    pmr, pmd : float
        RA, Dec proper motion (radians/year)
    px : float
        Parallax (arcseconds)
    rv : float
        Radial velocity (km/s, positive = receding)

    Returns
    -------
    pv : ndarray (2, 3)
        pv-vector (au, au/day)
    iwarn : int
        Status: 0=OK, 1=distance overridden, 2=excessive speed, 4=no convergence
    """
    PXMIN = 1e-7
    VMAX = 0.5
    IMAX = 100

    # Distance (au)
    w = jnp.where(px >= PXMIN, px, PXMIN)
    iwarn_dist = jnp.where(px >= PXMIN, 0, 1)
    r = DAS2R / w  # using DR2AS = DAS2R^-1... actually DR2AS/w

    # Correction: r = DR2AS / w = (1/DAS2R) / w... let me use the constant
    r = 1.0 / (w * DAS2R)  # equivalent to DR2AS / w

    # Radial speed (au/day)
    rd = DAYSEC * rv * 1e3 / DAU

    # Proper motion (radian/day)
    rad = pmr / DJY
    decd = pmd / DJY

    # To pv-vector (au, au/day)
    pv = s2pv(ra, dec, r, rad, decd, rd)

    # If excessive velocity, arbitrarily set it to zero
    v = pm(pv[1])
    excessive = v / DC > VMAX
    iwarn_vel = jnp.where(excessive, 2, 0)
    pv = jnp.where(excessive, pv.at[1].set(jnp.zeros(3)), pv)

    # Isolate the radial component of the velocity (au/day)
    _, pu = pn(pv[0])
    vsr = pdp(pu, pv[1])
    usr = sxp(vsr, pu)

    # Isolate the transverse component of the velocity (au/day)
    ust = pv[1] - usr
    vst = pm(ust)

    # Special-relativity dimensionless parameters
    betsr = vsr / DC
    betst = vst / DC

    # Determine the observed-to-inertial correction terms via iteration
    bett = betst
    betr = betsr

    d = 1.0 + betr
    w_val = betr * betr + bett * bett
    del_val = -w_val / (jnp.sqrt(1.0 - w_val) + 1.0)
    od = d
    odel = del_val

    # Unroll a fixed number of iterations (sufficient for convergence)
    for _ in range(IMAX):
        betr = d * betsr + del_val
        bett = d * betst
        d = 1.0 + betr
        w_val = betr * betr + bett * bett
        del_val = -w_val / (jnp.sqrt(jnp.maximum(1.0 - w_val, 1e-30)) + 1.0)
        od = d
        odel = del_val

    # Scale observed tangential velocity vector into inertial (au/d)
    ut = sxp(d, ust)

    # Compute inertial radial velocity vector (au/d)
    ur = sxp(DC * (d * betsr + del_val), pu)

    # Combine the two to obtain the inertial space velocity vector
    pv1 = ur + ut

    pv = jnp.stack([pv[0], pv1])

    iwarn = iwarn_dist + iwarn_vel
    return pv, iwarn


def pvstar(pv):
    """Convert star position+velocity vector to catalog coordinates.

    Parameters
    ----------
    pv : ndarray (2, 3)
        pv-vector (au, au/day)

    Returns
    -------
    ra, dec : float
        Right ascension, declination (radians)
    pmr, pmd : float
        RA, Dec proper motion (radians/year)
    px : float
        Parallax (arcsec)
    rv : float
        Radial velocity (km/s, positive = receding)
    """
    # Isolate the radial component of the velocity (au/day, inertial)
    r, pu = pn(pv[0])
    vr = pdp(pu, pv[1])
    ur = sxp(vr, pu)

    # Isolate the transverse component of the velocity (au/day, inertial)
    ut = pv[1] - ur
    vt = pm(ut)

    # Special-relativity dimensionless parameters
    bett = vt / DC
    betr = vr / DC

    # The observed-to-inertial correction terms
    d = 1.0 + betr
    w = betr * betr + bett * bett
    del_val = -w / (jnp.sqrt(1.0 - w) + 1.0)

    # Scale inertial tangential velocity vector into observed (au/d)
    ust = sxp(1.0 / d, ut)

    # Compute observed radial velocity vector (au/d)
    usr = sxp(DC * (betr - del_val) / d, pu)

    # Combine the two to obtain the observed velocity vector
    pv_obs = jnp.stack([pv[0], usr + ust])

    # Cartesian to spherical
    a, dec, r_out, rad, decd, rd = pv2s(pv_obs)

    # Return RA in range 0 to 2pi
    ra = anp(a)

    # Return proper motions in radians per year
    pmr = rad * DJY
    pmd = decd * DJY

    # Return parallax in arcsec
    px = DAS2R / jnp.maximum(r_out * DAS2R * DAS2R, 1e-30)
    # Correct: px = DR2AS / r = 1/(DAS2R * r)
    px = 1.0 / (DAS2R * r_out)

    # Return radial velocity in km/s
    rv = 1e-3 * rd * DAU / DAYSEC

    return ra, dec, pmr, pmd, px, rv


def starpm(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b):
    """Star proper motion: update star catalog data for space motion.

    Parameters
    ----------
    ra1, dec1 : float
        RA, Dec at epoch 1 (radians)
    pmr1, pmd1 : float
        RA, Dec proper motion (radians/year)
    px1 : float
        Parallax (arcsec)
    rv1 : float
        Radial velocity (km/s)
    ep1a, ep1b : float
        Epoch 1 as 2-part Julian Date (TDB)
    ep2a, ep2b : float
        Epoch 2 as 2-part Julian Date (TDB)

    Returns
    -------
    ra2, dec2 : float
        RA, Dec at epoch 2 (radians)
    pmr2, pmd2 : float
        RA, Dec proper motion at epoch 2 (radians/year)
    px2 : float
        Parallax at epoch 2 (arcsec)
    rv2 : float
        Radial velocity at epoch 2 (km/s)
    """
    from so_pointjax.erfa._core.vector import pvu

    # RA,Dec etc. at the "before" epoch to space motion pv-vector
    pv1, _ = starpv(ra1, dec1, pmr1, pmd1, px1, rv1)

    # Light time when observed (days)
    tl1 = pm(pv1[0]) / DC

    # Time interval, "before" to "after" (days)
    dt = (ep2a - ep1a) + (ep2b - ep1b)

    # Move star along track from the "before" observed position to the
    # "after" geometric position
    pv = pvu(dt + tl1, pv1)

    # From this geometric position, deduce the observed light time (days)
    r2 = pdp(pv[0], pv[0])
    rdv = pdp(pv[0], pv[1])
    v2 = pdp(pv[1], pv[1])
    c2mv2 = DC * DC - v2
    tl2 = (-rdv + jnp.sqrt(rdv * rdv + c2mv2 * r2)) / c2mv2

    # Move the position along track from the observed place at the
    # "before" epoch to the observed place at the "after" epoch
    pv2 = pvu(dt + (tl1 - tl2), pv1)

    # Space motion pv-vector to RA,Dec etc. at the "after" epoch
    ra2, dec2, pmr2, pmd2, px2, rv2 = pvstar(pv2)

    return ra2, dec2, pmr2, pmd2, px2, rv2


def pmsafe(ra1, dec1, pmr1, pmd1, px1, rv1, ep1a, ep1b, ep2a, ep2b):
    """Star proper motion: update star catalog data for space motion (safe version).

    Parameters
    ----------
    ra1, dec1 : float
        RA, Dec at epoch 1 (radians)
    pmr1, pmd1 : float
        RA, Dec proper motion (radians/year)
    px1 : float
        Parallax (arcsec)
    rv1 : float
        Radial velocity (km/s)
    ep1a, ep1b : float
        Epoch 1 as 2-part Julian Date (TDB)
    ep2a, ep2b : float
        Epoch 2 as 2-part Julian Date (TDB)

    Returns
    -------
    ra2, dec2 : float
        RA, Dec at epoch 2 (radians)
    pmr2, pmd2 : float
        RA, Dec proper motion at epoch 2 (radians/year)
    px2 : float
        Parallax at epoch 2 (arcsec)
    rv2 : float
        Radial velocity at epoch 2 (km/s)
    """
    from so_pointjax.erfa._core.vector import sepp

    # Minimum allowed parallax (arcsec)
    PXMIN = 5e-7
    # Factor giving maximum allowed transverse speed of about 1% c
    F = 326.0

    # Proper motion in one year (radians)
    pm_val = sepp(
        jnp.array([ra1, dec1]),
        jnp.array([ra1 + pmr1, dec1 + pmd1]),
    )
    # Actually sepp takes (al, ap, bl, bp) as separate args... let me use seps
    from so_pointjax.erfa._core.vector import seps
    pm_val = seps(ra1, dec1, ra1 + pmr1, dec1 + pmd1)

    # Override the parallax to reduce the chances of a warning status
    px1a = px1
    pm_scaled = pm_val * F
    px1a = jnp.where(px1a < pm_scaled, pm_scaled, px1a)
    px1a = jnp.where(px1a < PXMIN, PXMIN, px1a)

    # Carry out the transformation using the modified parallax
    ra2, dec2, pmr2, pmd2, px2, rv2 = starpm(
        ra1, dec1, pmr1, pmd1, px1a, rv1,
        ep1a, ep1b, ep2a, ep2b)

    return ra2, dec2, pmr2, pmd2, px2, rv2


# ============================================================================
# High-level "13" wrapper functions (auto-compute ephemerides)
# ============================================================================

def _epv00_import(date1, date2):
    """Lazy import of epv00 to avoid circular imports."""
    from so_pointjax.erfa._core.ephem import epv00
    return epv00(date1, date2)


def _era00_import(dj1, dj2):
    """Lazy import of era00."""
    from so_pointjax.erfa._core.time import era00
    return era00(dj1, dj2)


def apcg13(date1, date2):
    """For a geocentric observer, prepare ICRS-GCRS parameters (auto-computed).

    Parameters
    ----------
    date1, date2 : float
        TDB as a 2-part Julian Date

    Returns
    -------
    astrom : ASTROM
    """
    pvh, pvb = _epv00_import(date1, date2)
    return apcg(date1, date2, pvb, pvh[0])


def apcs13(date1, date2, pv):
    """For a space observer, prepare ICRS-GCRS parameters (auto-computed).

    Parameters
    ----------
    date1, date2 : float
        TDB as a 2-part Julian Date
    pv : ndarray (2, 3)
        Observer's geocentric pos/vel (m, m/s)

    Returns
    -------
    astrom : ASTROM
    """
    pvh, pvb = _epv00_import(date1, date2)
    return apcs(date1, date2, pv, pvb, pvh[0])


def apci13(date1, date2):
    """For a terrestrial observer, prepare ICRS-CIRS parameters (auto-computed).

    Parameters
    ----------
    date1, date2 : float
        TDB as a 2-part Julian Date

    Returns
    -------
    astrom : ASTROM
    eo : float
        Equation of the origins (ERA-GST)
    """
    from so_pointjax.erfa._core.precnut import pnm06a, bpn2xy, s06, eors

    # Earth barycentric & heliocentric position/velocity
    pvh, pvb = _epv00_import(date1, date2)

    # BPN matrix, IAU 2006/2000A
    r = pnm06a(date1, date2)

    # Extract CIP X, Y
    x, y = bpn2xy(r)

    # CIO locator
    s_val = s06(date1, date2, x, y)

    # ASTROM
    astrom = apci(date1, date2, pvb, pvh[0], x, y, s_val)

    # Equation of the origins
    eo = eors(r, s_val)

    return astrom, eo


def apco13(utc1, utc2, dut1, elong, phi, hm, xp, yp, phpa, tc, rh_val, wl):
    """For a terrestrial observer, prepare ICRS-to-observed parameters (auto-computed).

    Parameters
    ----------
    utc1, utc2 : float
        UTC as a 2-part quasi Julian Date
    dut1 : float
        UT1-UTC (seconds)
    elong : float
        Longitude (radians, east +ve)
    phi : float
        Geodetic latitude (radians)
    hm : float
        Height above ellipsoid (m)
    xp, yp : float
        Polar motion coordinates (radians)
    phpa : float
        Pressure at the observer (hPa)
    tc : float
        Ambient temperature (deg C)
    rh_val : float
        Relative humidity (range 0-1)
    wl : float
        Wavelength (micrometers)

    Returns
    -------
    astrom : ASTROM
    eo : float
        Equation of the origins
    """
    from so_pointjax.erfa._core.time import utctai, taitt, utcut1, era00
    from so_pointjax.erfa._core.precnut import pnm06a, bpn2xy, s06, sp00, eors

    # UTC to other timescales
    tai1, tai2 = utctai(utc1, utc2)
    tt1, tt2 = taitt(tai1, tai2)
    ut11, ut12 = utcut1(utc1, utc2, dut1)

    # Earth barycentric & heliocentric position/velocity
    pvh, pvb = _epv00_import(tt1, tt2)

    # BPN matrix, IAU 2006/2000A
    r = pnm06a(tt1, tt2)

    # Extract CIP X, Y
    x, y = bpn2xy(r)

    # CIO locator
    s_val = s06(tt1, tt2, x, y)

    # Earth rotation angle
    theta = era00(ut11, ut12)

    # TIO locator s'
    sp_val = sp00(tt1, tt2)

    # Refraction constants
    refa_val, refb_val = refco(phpa, tc, rh_val, wl)

    # ASTROM
    astrom = apco(tt1, tt2, pvb, pvh[0], x, y, s_val,
                  theta, elong, phi, hm, xp, yp, sp_val,
                  refa_val, refb_val)

    # Equation of the origins
    eo = eors(r, s_val)

    return astrom, eo


def aper13(ut11, ut12, astrom):
    """Update ASTROM for Earth rotation from UT1 (auto-computed).

    Parameters
    ----------
    ut11, ut12 : float
        UT1 as a 2-part Julian Date
    astrom : ASTROM

    Returns
    -------
    astrom : ASTROM
    """
    return aper(_era00_import(ut11, ut12), astrom)


def apio13(utc1, utc2, dut1, elong, phi, hm, xp, yp, phpa, tc, rh_val, wl):
    """For a terrestrial observer, prepare CIRS-to-observed parameters (auto-computed).

    Parameters
    ----------
    utc1, utc2 : float
        UTC as a 2-part quasi Julian Date
    dut1 : float
        UT1-UTC (seconds)
    elong : float
        Longitude (radians, east +ve)
    phi : float
        Geodetic latitude (radians)
    hm : float
        Height above ellipsoid (m)
    xp, yp : float
        Polar motion coordinates (radians)
    phpa : float
        Pressure (hPa)
    tc : float
        Temperature (deg C)
    rh_val : float
        Relative humidity (0-1)
    wl : float
        Wavelength (micrometers)

    Returns
    -------
    astrom : ASTROM
    """
    from so_pointjax.erfa._core.time import utctai, taitt, utcut1, era00
    from so_pointjax.erfa._core.precnut import sp00

    # UTC to other timescales
    tai1, tai2 = utctai(utc1, utc2)
    tt1, tt2 = taitt(tai1, tai2)
    ut11, ut12 = utcut1(utc1, utc2, dut1)

    # Earth rotation angle
    theta = era00(ut11, ut12)

    # TIO locator s'
    sp_val = sp00(tt1, tt2)

    # Refraction constants
    refa_val, refb_val = refco(phpa, tc, rh_val, wl)

    # CIRS-to-observed parameters
    return apio(sp_val, theta, elong, phi, hm, xp, yp, refa_val, refb_val)


# ============================================================================
# High-level convenience wrappers
# ============================================================================

def atci13(rc, dc, pr, pd, px, rv, date1, date2):
    """ICRS RA,Dec to CIRS (complete).

    Parameters
    ----------
    rc, dc : float
        ICRS RA, Dec at J2000.0 (radians)
    pr, pd : float
        RA, Dec proper motion (radians/year)
    px : float
        Parallax (arcsec)
    rv : float
        Radial velocity (km/s)
    date1, date2 : float
        TDB as a 2-part Julian Date

    Returns
    -------
    ri, di : float
        CIRS RA, Dec (radians)
    eo : float
        Equation of the origins
    """
    astrom, eo = apci13(date1, date2)
    ri, di = atciq(rc, dc, pr, pd, px, rv, astrom)
    return ri, di, eo


def atic13(ri, di, date1, date2):
    """CIRS RA,Dec to ICRS astrometric place (complete).

    Parameters
    ----------
    ri, di : float
        CIRS RA, Dec (radians)
    date1, date2 : float
        TDB as a 2-part Julian Date

    Returns
    -------
    rc, dc : float
        ICRS astrometric RA, Dec (radians)
    eo : float
        Equation of the origins
    """
    astrom, eo = apci13(date1, date2)
    rc, dc = aticq(ri, di, astrom)
    return rc, dc, eo


def atco13(rc, dc, pr, pd, px, rv, utc1, utc2, dut1,
           elong, phi, hm, xp, yp, phpa, tc, rh_val, wl):
    """ICRS RA,Dec to observed place (complete).

    Returns
    -------
    aob, zob, hob, dob, rob : float
        Observed coordinates
    eo : float
        Equation of the origins
    """
    astrom, eo = apco13(utc1, utc2, dut1, elong, phi, hm, xp, yp,
                        phpa, tc, rh_val, wl)
    ri, di = atciq(rc, dc, pr, pd, px, rv, astrom)
    aob, zob, hob, dob, rob = atioq(ri, di, astrom)
    return aob, zob, hob, dob, rob, eo


def atio13(ri, di, utc1, utc2, dut1, elong, phi, hm, xp, yp,
           phpa, tc, rh_val, wl):
    """CIRS RA,Dec to observed place (complete).

    Returns
    -------
    aob, zob, hob, dob, rob : float
        Observed coordinates
    """
    astrom = apio13(utc1, utc2, dut1, elong, phi, hm, xp, yp,
                    phpa, tc, rh_val, wl)
    return atioq(ri, di, astrom)


def atoc13(type_str, ob1, ob2, utc1, utc2, dut1,
           elong, phi, hm, xp, yp, phpa, tc, rh_val, wl):
    """Observed place to ICRS astrometric RA,Dec (complete).

    Returns
    -------
    rc, dc : float
        ICRS astrometric RA, Dec (radians)
    """
    astrom, eo = apco13(utc1, utc2, dut1, elong, phi, hm, xp, yp,
                        phpa, tc, rh_val, wl)
    ri, di = atoiq(type_str, ob1, ob2, astrom)
    rc, dc = aticq(ri, di, astrom)
    return rc, dc


def atoi13(type_str, ob1, ob2, utc1, utc2, dut1,
           elong, phi, hm, xp, yp, phpa, tc, rh_val, wl):
    """Observed place to CIRS (complete).

    Returns
    -------
    ri, di : float
        CIRS RA, Dec (radians)
    """
    astrom = apio13(utc1, utc2, dut1, elong, phi, hm, xp, yp,
                    phpa, tc, rh_val, wl)
    return atoiq(type_str, ob1, ob2, astrom)


__all__ = [
    # Fundamental effects
    "ab", "ld", "ldn", "ldsun", "pmpx", "refco", "pvtob",
    # Context setup
    "apcs", "apcg", "apci", "apco", "aper", "apio",
    # Context setup (auto-computed)
    "apcg13", "apcs13", "apci13", "apco13", "aper13", "apio13",
    # Transforms (quick)
    "atciq", "atciqz", "atciqn", "aticq", "aticqn", "atioq", "atoiq",
    # Transforms (complete)
    "atci13", "atic13", "atco13", "atio13", "atoc13", "atoi13",
    # Space motion
    "starpv", "pvstar", "starpm", "pmsafe",
]
