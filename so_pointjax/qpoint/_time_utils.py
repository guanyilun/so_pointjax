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
