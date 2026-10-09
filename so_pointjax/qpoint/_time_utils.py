"""Time conversion utilities for QPoint.

Converts between Unix time (ctime), Julian Date, TT, UT1, and GMST.
Non-JIT functions (UTC conversions) use so_pointjax.erfa plain-Python routines.
"""

import jax.numpy as jnp
import so_pointjax.erfa


# Unix epoch as Julian Date (1970 Jan 1 0h UTC)
CTIME_JD_EPOCH = 2440587.5
DAYSEC = 86400.0


def ctime2jd(ctime):
    """Convert Unix time to 2-part Julian Date (UTC).

    Returns (jd1, jd2) where jd1 = CTIME_JD_EPOCH, jd2 = ctime/86400.
    """
    return CTIME_JD_EPOCH, ctime / DAYSEC


def jd2ctime(jd1, jd2):
    """Convert 2-part Julian Date (UTC) to Unix time."""
    return ((jd1 - CTIME_JD_EPOCH) + jd2) * DAYSEC


def ctime2jdtt(ctime):
    """Convert Unix time to 2-part Julian Date TT.

    Chains: ctime → JD(UTC) → TAI → TT.
    Non-JIT (goes through leap-second table).

    Returns (tt1, tt2).
    """
    jd1, jd2 = ctime2jd(ctime)
    tai1, tai2 = so_pointjax.erfa.utctai(jd1, jd2)
    tt1, tt2 = so_pointjax.erfa.taitt(tai1, tai2)
    return tt1, tt2


def jdutc2jdut1(jd1, jd2, dut1):
    """Convert JD(UTC) to JD(UT1) given DUT1 = UT1-UTC in seconds.

    Non-JIT (goes through leap-second table).

    Returns (ut1_1, ut1_2).
    """
    return so_pointjax.erfa.utcut1(jd1, jd2, dut1)


def _jd2cal_split(dj1, dj2):
    """Vectorized day/fraction split of ``erfa.jd2cal``.

    Same floating-point operations as the scalar routine, applied
    elementwise.  Returns (jd, f): the integer Julian Day Number and the
    fraction of the day since 0h.
    """
    import numpy as np
    dj1 = np.asarray(dj1, dtype=np.float64)
    dj2 = np.asarray(dj2, dtype=np.float64)

    d = np.round(dj1)
    f1 = dj1 - d
    jd = d.astype(np.int64)
    d = np.round(dj2)
    f2 = dj2 - d
    jd = jd + d.astype(np.int64)

    s = np.full_like(f1, 0.5)
    cs = np.zeros_like(f1)
    for x in (f1, f2):
        t = s + x
        cs = cs + np.where(np.abs(s) >= np.abs(x), (s - t) + x, (x - t) + s)
        s = t
        wrap = s >= 1.0
        jd = jd + wrap
        s = np.where(wrap, s - 1.0, s)

    f = s + cs
    cs = f - s

    neg = f < 0.0
    fn = s + 1.0
    cs = np.where(neg, cs + ((1.0 - fn) + s), cs)
    s = np.where(neg, fn, s)
    f = np.where(neg, s + cs, f)
    cs = np.where(neg, f - s, cs)
    jd = jd - neg

    eps = np.finfo(np.float64).eps
    big = (f - 1.0) >= -eps / 4.0
    t = s - 1.0
    cs = np.where(big, cs + ((s - t) - 1.0), cs)
    s = np.where(big, t, s)
    f = np.where(big, s + cs, f)
    up = big & (-eps / 2.0 < f)
    jd = jd + up
    f = np.where(up, np.maximum(f, 0.0), f)
    return jd, f


def utc2tt_ut1(utc1, utc2, dut1=0.0):
    """Vectorized UTC → TT and UTC → UT1 for arrays of 2-part JD(UTC).

    Elementwise equivalent of ``erfa.utctai`` + ``erfa.taitt`` and
    ``erfa.utcut1``.  The leap-second table is consulted once per distinct
    UTC day instead of once per sample, so this is fast for long time
    series and still handles leap seconds and pre-1972 drift.

    Returns (tt1, tt2, ut1_1, ut1_2) as numpy arrays.
    """
    import numpy as np
    from so_pointjax.erfa._core.calendar import jd2cal, cal2jd
    from so_pointjax.erfa._core.constants import TTMTAI
    from so_pointjax.erfa._leapsec import dat

    utc1 = np.asarray(utc1, dtype=np.float64)
    utc2 = np.asarray(utc2, dtype=np.float64)
    utc1, utc2 = np.broadcast_arrays(utc1, utc2)
    dut1 = np.broadcast_to(np.asarray(dut1, dtype=np.float64), utc1.shape)

    big1 = np.abs(utc1) >= np.abs(utc2)
    u1 = np.where(big1, utc1, utc2)
    u2 = np.where(big1, utc2, utc1)

    jd, fd = _jd2cal_split(u1, u2)

    days, inverse = np.unique(jd, return_inverse=True)
    inverse = inverse.reshape(jd.shape)
    per_day = np.empty((len(days), 5))
    for i, day in enumerate(days):
        iy, im, id, _ = jd2cal(float(day), 0.0)
        dat0 = dat(iy, im, id, 0.0)
        dat12 = dat(iy, im, id, 0.5)
        iyt, imt, idt, _ = jd2cal(float(day + 1), 0.0)
        dat24 = dat(iyt, imt, idt, 0.0)
        dlod = 2.0 * (dat12 - dat0)
        dleap = dat24 - (dat0 + dlod)
        z1, z2 = cal2jd(iy, im, id)
        per_day[i] = dat0, dlod, dleap, z1, z2
    dat0, dlod, dleap, z1, z2 = per_day[inverse].transpose(
        (jd.ndim,) + tuple(range(jd.ndim)))

    # utctai
    fd = fd * ((DAYSEC + dleap) / DAYSEC)
    fd = fd * ((DAYSEC + dlod) / DAYSEC)
    a2 = z1 - u1
    a2 = a2 + z2
    a2 = a2 + (fd + dat0 / DAYSEC)
    tai1 = np.where(big1, u1, a2)
    tai2 = np.where(big1, a2, u1)

    big = np.abs(tai1) >= np.abs(tai2)

    # taitt
    dtat = TTMTAI / DAYSEC
    tt1 = np.where(big, tai1, tai1 + dtat)
    tt2 = np.where(big, tai2 + dtat, tai2)

    # utcut1
    dtad = (dut1 - dat0) / DAYSEC
    ut1_1 = np.where(big, tai1, tai1 + dtad)
    ut1_2 = np.where(big, tai2 + dtad, tai2)

    return tt1, tt2, ut1_1, ut1_2


def ctime2gmst(ctime, dut1=0.0, accuracy=0):
    """Compute Greenwich Mean Sidereal Time from Unix time.

    Parameters
    ----------
    ctime : float
        Unix time in seconds.
    dut1 : float
        UT1-UTC in seconds (ignored if accuracy != 0).
    accuracy : int
        0 = full accuracy (uses UT1 and TT), 1 = low accuracy (UTC only).

    Returns
    -------
    gmst : float
        GMST in radians.
    """
    jd1, jd2 = ctime2jd(ctime)
    if accuracy == 0:
        ut1_1, ut1_2 = jdutc2jdut1(jd1, jd2, dut1)
        tt1, tt2 = ctime2jdtt(ctime)
        return so_pointjax.erfa.gmst00(ut1_1, ut1_2, tt1, tt2)
    else:
        return so_pointjax.erfa.gmst00(jd1, jd2, jd1, jd2)
