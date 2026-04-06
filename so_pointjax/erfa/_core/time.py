"""Time scale conversions, Earth rotation angle, and sidereal time.

Simple timescale conversions (TAI<->TT, TAI<->UT1, TT<->UT1, TT<->TDB,
TCG<->TT, TCB<->TDB) are differentiable and JIT-compatible (JAX).

UTC-related conversions (UTC<->TAI, UT1<->UTC) are non-differentiable
because they go through the leap-second table. They use plain Python.

Earth rotation angle (era00) and Greenwich Mean Sidereal Time (gmst00,
gmst06, gmst82) are differentiable and JIT-compatible.

GST and Equation of Equinoxes functions are deferred to Phase 3
(they depend on precession/nutation models).
"""

import math
import jax.numpy as jnp
from so_pointjax.erfa._core.constants import (
    DAYSEC, DJ00, DJC, DJM0, DJM00, DJM77, TTMTAI,
    ELG, ELB, TDB0, D2PI, DAS2R, DS2R,
)
from so_pointjax.erfa._core.angles import anp


# ---------------------------------------------------------------------------
# Helper: precision-safe split (add offset to smaller component)
# ---------------------------------------------------------------------------

def _split_add(a1, a2, offset):
    """Add offset to the (a1, a2) pair, preserving precision."""
    big1 = jnp.abs(a1) >= jnp.abs(a2)
    r1 = jnp.where(big1, a1, a1 + offset)
    r2 = jnp.where(big1, a2 + offset, a2)
    return r1, r2


def _split_sub(a1, a2, offset):
    """Subtract offset from the (a1, a2) pair, preserving precision."""
    return _split_add(a1, a2, -offset)


# ===========================================================================
# Differentiable timescale conversions (JAX, JIT-compatible)
# ===========================================================================

# --- TAI <-> TT ---

def taitt(tai1, tai2):
    """Time scale transformation: TAI to TT.

    Returns (tt1, tt2).
    """
    dtat = TTMTAI / DAYSEC
    return _split_add(tai1, tai2, dtat)


def tttai(tt1, tt2):
    """Time scale transformation: TT to TAI.

    Returns (tai1, tai2).
    """
    dtat = TTMTAI / DAYSEC
    return _split_sub(tt1, tt2, dtat)


# --- TAI <-> UT1 ---

def taiut1(tai1, tai2, dta):
    """Time scale transformation: TAI to UT1.

    dta: UT1-TAI in seconds (from observations).
    Returns (ut11, ut12).
    """
    dtad = dta / DAYSEC
    return _split_add(tai1, tai2, dtad)


def ut1tai(ut11, ut12, dta):
    """Time scale transformation: UT1 to TAI.

    dta: UT1-TAI in seconds.
    Returns (tai1, tai2).
    """
    dtad = dta / DAYSEC
    return _split_sub(ut11, ut12, dtad)


# --- TT <-> UT1 ---

def ttut1(tt1, tt2, dt):
    """Time scale transformation: TT to UT1.

    dt: TT-UT1 in seconds (Delta T).
    Returns (ut11, ut12).
    """
    dtd = dt / DAYSEC
    return _split_sub(tt1, tt2, dtd)


def ut1tt(ut11, ut12, dt):
    """Time scale transformation: UT1 to TT.

    dt: TT-UT1 in seconds (Delta T).
    Returns (tt1, tt2).
    """
    dtd = dt / DAYSEC
    return _split_add(ut11, ut12, dtd)


# --- TT <-> TDB ---

def tttdb(tt1, tt2, dtr):
    """Time scale transformation: TT to TDB.

    dtr: TDB-TT in seconds (quasi-periodic term, ~1.7ms amplitude).
    Returns (tdb1, tdb2).
    """
    dtrd = dtr / DAYSEC
    return _split_add(tt1, tt2, dtrd)


def tdbtt(tdb1, tdb2, dtr):
    """Time scale transformation: TDB to TT.

    dtr: TDB-TT in seconds.
    Returns (tt1, tt2).
    """
    dtrd = dtr / DAYSEC
    return _split_sub(tdb1, tdb2, dtrd)


# --- TCG <-> TT ---

def tcgtt(tcg1, tcg2):
    """Time scale transformation: TCG to TT.

    Returns (tt1, tt2).
    """
    # Reference epoch: 1977 Jan 1 00:00:32.184 TT as MJD
    t77t = DJM77 + TTMTAI / DAYSEC

    # TCG-TT
    big1 = jnp.abs(tcg1) >= jnp.abs(tcg2)
    t1 = jnp.where(big1, tcg1, tcg2)
    t2 = jnp.where(big1, tcg2, tcg1)
    elg_corr = ((t1 - DJM0) + (t2 - t77t)) * ELG
    tt1 = jnp.where(big1, tcg1, tcg1 - elg_corr)
    tt2 = jnp.where(big1, tcg2 - elg_corr, tcg2)
    return tt1, tt2


def tttcg(tt1, tt2):
    """Time scale transformation: TT to TCG.

    Returns (tcg1, tcg2).
    """
    t77t = DJM77 + TTMTAI / DAYSEC
    elgg = ELG / (1.0 - ELG)

    big1 = jnp.abs(tt1) >= jnp.abs(tt2)
    t1 = jnp.where(big1, tt1, tt2)
    t2 = jnp.where(big1, tt2, tt1)
    elg_corr = ((t1 - DJM0) + (t2 - t77t)) * elgg
    tcg1 = jnp.where(big1, tt1, tt1 + elg_corr)
    tcg2 = jnp.where(big1, tt2 + elg_corr, tt2)
    return tcg1, tcg2


# --- TCB <-> TDB ---

def tcbtdb(tcb1, tcb2):
    """Time scale transformation: TCB to TDB.

    Returns (tdb1, tdb2).
    """
    # 1977 Jan 1.0 TAI as two-part JD
    t77td = DJM0 + DJM77
    t77tf = TTMTAI / DAYSEC
    tdb0 = TDB0 / DAYSEC

    big1 = jnp.abs(tcb1) >= jnp.abs(tcb2)
    t1 = jnp.where(big1, tcb1, tcb2)
    t2 = jnp.where(big1, tcb2, tcb1)

    # TCB to TDB
    elb_corr = ((t1 - t77td) + (t2 - t77tf)) * ELB - tdb0
    tdb1 = jnp.where(big1, tcb1, tcb1 - elb_corr)
    tdb2 = jnp.where(big1, tcb2 - elb_corr, tcb2)
    return tdb1, tdb2


def tdbtcb(tdb1, tdb2):
    """Time scale transformation: TDB to TCB.

    Returns (tcb1, tcb2).
    """
    t77td = DJM0 + DJM77
    t77tf = TTMTAI / DAYSEC
    tdb0 = TDB0 / DAYSEC
    elbb = ELB / (1.0 - ELB)

    big1 = jnp.abs(tdb1) >= jnp.abs(tdb2)

    # Precision-preserving formulation matching C ERFA exactly:
    # d = t77td - (larger), f = (smaller) - tdb0
    # result = f - (d - (f - t77tf)) * elbb
    d = jnp.where(big1, t77td - tdb1, t77td - tdb2)
    f = jnp.where(big1, tdb2 - tdb0, tdb1 - tdb0)
    elb_corr = f - (d - (f - t77tf)) * elbb

    tcb1 = jnp.where(big1, tdb1, elb_corr)
    tcb2 = jnp.where(big1, elb_corr, tdb2)
    return tcb1, tcb2


# ===========================================================================
# Non-differentiable UTC conversions (plain Python, not JIT-compatible)
# ===========================================================================

def utctai(utc1, utc2):
    """Time scale transformation: UTC to TAI.

    Handles leap seconds and pre-1972 UTC drift.
    Returns (tai1, tai2).
    """
    from so_pointjax.erfa._core.calendar import jd2cal, cal2jd
    from so_pointjax.erfa._leapsec import dat

    utc1 = float(utc1)
    utc2 = float(utc2)

    # Put the two parts into big-first order
    if abs(utc1) >= abs(utc2):
        u1, u2 = utc1, utc2
        big1 = True
    else:
        u1, u2 = utc2, utc1
        big1 = False

    # Get TAI-UTC at 0h today
    iy, im, id, fd = jd2cal(u1, u2)
    dat0 = dat(iy, im, id, 0.0)

    # Get TAI-UTC at 12h today (to detect drift)
    dat12 = dat(iy, im, id, 0.5)

    # Get TAI-UTC at 0h tomorrow (to detect jumps)
    iyt, imt, idt, _ = jd2cal(u1 + 1.5, u2 - fd)
    dat24 = dat(iyt, imt, idt, 0.0)

    # Separate TAI-UTC change into per-day (DLOD) and any jump (DLEAP)
    dlod = 2.0 * (dat12 - dat0)
    dleap = dat24 - (dat0 + dlod)

    # Remove any scaling applied to spread leap into preceding day
    fd *= (DAYSEC + dleap) / DAYSEC

    # Scale from (pre-1972) UTC seconds to SI seconds
    fd *= (DAYSEC + dlod) / DAYSEC

    # Today's calendar date to 2-part JD
    z1, z2 = cal2jd(iy, im, id)

    # Assemble the TAI result, preserving the UTC split and order
    a2 = z1 - u1
    a2 += z2
    a2 += fd + dat0 / DAYSEC

    if big1:
        return u1, a2
    else:
        return a2, u1


def taiutc(tai1, tai2):
    """Time scale transformation: TAI to UTC.

    Uses iterative approach to invert the UTC->TAI transformation.
    Returns (utc1, utc2).
    """
    tai1 = float(tai1)
    tai2 = float(tai2)

    # Put the two parts into big-first order
    if abs(tai1) >= abs(tai2):
        a1, a2 = tai1, tai2
        big1 = True
    else:
        a1, a2 = tai2, tai1
        big1 = False

    # Initial guess for UTC
    u1, u2 = a1, a2

    # Iterate (usually converges in 1 iteration)
    for _ in range(3):
        g1, g2 = utctai(u1, u2)
        u2 += a1 - g1
        u2 += a2 - g2

    if big1:
        return u1, u2
    else:
        return u2, u1


def ut1utc(ut11, ut12, dut1):
    """Time scale transformation: UT1 to UTC.

    dut1: UT1-UTC in seconds (Delta UT1).
    Returns (utc1, utc2).
    """
    from so_pointjax.erfa._core.calendar import jd2cal, cal2jd
    from so_pointjax.erfa._leapsec import dat

    ut11 = float(ut11)
    ut12 = float(ut12)
    duts = float(dut1)

    # Put the two parts into big-first order
    if abs(ut11) >= abs(ut12):
        u1, u2 = ut11, ut12
        big1 = True
    else:
        u1, u2 = ut12, ut11
        big1 = False

    # See if the UT1 can possibly be in a leap-second day
    d1 = u1
    dats1 = 0.0
    for i in range(-1, 4):
        d2 = u2 + float(i)
        iy, im, id, fd = jd2cal(d1, d2)
        dats2 = dat(iy, im, id, 0.0)
        if i == -1:
            dats1 = dats2
        ddats = dats2 - dats1
        if abs(ddats) >= 0.5:
            # Leap second nearby: ensure UT1-UTC is "before" value
            if ddats * duts >= 0.0:
                duts -= ddats

            # UT1 for the start of the UTC day that ends in a leap
            d1_s, d2_s = cal2jd(iy, im, id)
            us1 = d1_s
            us2 = d2_s - 1.0 + duts / DAYSEC

            # Is the UT1 after this point?
            du = (u1 - us1) + (u2 - us2)
            if du > 0.0:
                # Fraction of the current UTC day that has elapsed
                fd_frac = du * DAYSEC / (DAYSEC + ddats)
                # Ramp UT1-UTC to bring about ERFA's JD(UTC) convention
                duts += ddats * (fd_frac if fd_frac <= 1.0 else 1.0)

            break
        dats1 = dats2

    # Subtract the (possibly adjusted) UT1-UTC from UT1 to give UTC
    u2 -= duts / DAYSEC

    if big1:
        return u1, u2
    else:
        return u2, u1


def utcut1(utc1, utc2, dut1):
    """Time scale transformation: UTC to UT1.

    dut1: UT1-UTC in seconds (Delta UT1).
    Returns (ut11, ut12).
    """
    from so_pointjax.erfa._core.calendar import jd2cal
    from so_pointjax.erfa._leapsec import dat

    utc1 = float(utc1)
    utc2 = float(utc2)

    # Look up TAI-UTC
    iy, im, id, _ = jd2cal(utc1, utc2)
    dat_val = dat(iy, im, id, 0.0)

    # Form UT1-TAI
    dta = dut1 - dat_val

    # UTC to TAI to UT1
    tai1, tai2 = utctai(utc1, utc2)
    ut11, ut12 = taiut1(tai1, tai2, dta)

    return ut11, ut12


# ===========================================================================
# Earth rotation angle and sidereal time (JAX, JIT-compatible)
# ===========================================================================

def era00(dj1, dj2):
    """Earth rotation angle (IAU 2000 model).

    Given UT1 as a 2-part Julian Date, returns the Earth Rotation Angle
    in radians (0 to 2pi).
    """
    # Days since J2000.0
    big1 = jnp.abs(dj1) >= jnp.abs(dj2)
    d1 = jnp.where(big1, dj1, dj2)
    d2 = jnp.where(big1, dj2, dj1)

    t = d1 + (d2 - DJ00)

    # Fractional part of T (days)
    f = jnp.fmod(d1, 1.0) + jnp.fmod(d2, 1.0)

    # Earth rotation angle
    theta = anp(D2PI * (f + 0.7790572732640 + 0.00273781191135448 * t))
    return theta


def gmst00(uta, utb, tta, ttb):
    """Greenwich mean sidereal time (IAU 2000 model, consistent with IAU 2000 precession).

    Given UT1 (uta, utb) and TT (tta, ttb) as 2-part Julian Dates,
    returns GMST in radians.
    """
    t = ((tta - DJ00) + ttb) / DJC

    gmst = anp(era00(uta, utb) +
               (0.014506 +
                (4612.15739966 +
                 (1.39667721 +
                  (-0.00009344 +
                   0.00001882 * t) * t) * t) * t) * DAS2R)
    return gmst


def gmst06(uta, utb, tta, ttb):
    """Greenwich mean sidereal time (IAU 2006 model, consistent with IAU 2006 precession).

    Given UT1 (uta, utb) and TT (tta, ttb) as 2-part Julian Dates,
    returns GMST in radians.
    """
    t = ((tta - DJ00) + ttb) / DJC

    gmst = anp(era00(uta, utb) +
               (0.014506 +
                (4612.156534 +
                 (1.3915817 +
                  (-0.00000044 +
                   (-0.000029956 +
                    (-0.0000000368) * t) * t) * t) * t) * t) * DAS2R)
    return gmst


def gmst82(dj1, dj2):
    """Greenwich mean sidereal time (IAU 1982 model).

    Given UT1 as a 2-part Julian Date, returns GMST in radians.
    """
    # Coefficients of IAU 1982 GMST-UT1 model
    A = 24110.54841 - DAYSEC / 2.0
    B = 8640184.812866
    C = 0.093104
    D = -6.2e-6

    # Note: the first constant, A, includes the 12h offset that converts
    # from Julian Date (noon) to calendar date (midnight).

    big1 = jnp.abs(dj1) >= jnp.abs(dj2)
    d1 = jnp.where(big1, dj1, dj2)
    d2 = jnp.where(big1, dj2, dj1)

    # Julian centuries since fundamental epoch
    t = (d1 + (d2 - DJ00)) / DJC

    # Fractional part of JD(UT1), in seconds
    f = DAYSEC * (jnp.fmod(d1, 1.0) + jnp.fmod(d2, 1.0))

    # GMST at this UT1
    gmst = anp(DS2R * ((A + (B + (C + D * t) * t) * t) + f))
    return gmst


__all__ = [
    # Differentiable timescale conversions
    "taitt", "tttai",
    "taiut1", "ut1tai",
    "ttut1", "ut1tt",
    "tttdb", "tdbtt",
    "tcgtt", "tttcg",
    "tcbtdb", "tdbtcb",
    # Non-differentiable UTC conversions
    "utctai", "taiutc",
    "ut1utc", "utcut1",
    # Earth rotation and sidereal time
    "era00",
    "gmst00", "gmst06", "gmst82",
]
