"""Calendar and epoch conversion functions.

Ported from ERFA C library. Functions cal2jd, jd2cal, jdcalf are inherently
non-differentiable (integer calendar operations) and use plain NumPy.
Functions epb, epj, epb2jd, epj2jd are differentiable and use JAX.
"""

import numpy as np
import jax.numpy as jnp
from so_pointjax.erfa._core.constants import DJ00, DJM0, DJM00, DJY, DTY


# ---------------------------------------------------------------------------
# Non-differentiable calendar functions (plain NumPy)
# ---------------------------------------------------------------------------

def cal2jd(iy, im, id):
    """Gregorian calendar to Julian Date.

    Returns (djm0, djm) where djm0 is always 2400000.5 (MJD zero-point)
    and djm is the Modified Julian Date for 0 hrs.
    """
    IYMIN = -4799
    mtab = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    if iy < IYMIN:
        raise ValueError(f"bad year: {iy}")
    if im < 1 or im > 12:
        raise ValueError(f"bad month: {im}")

    # Leap year check for February
    ly = 1 if (im == 2 and iy % 4 == 0 and (iy % 100 != 0 or iy % 400 == 0)) else 0
    if id < 1 or id > mtab[im - 1] + ly:
        raise ValueError(f"bad day: {id}")

    # Algorithm from Explanatory Supplement, Section 12.92 (p604)
    # C integer division truncates toward zero
    my = int((im - 14) / 12)
    iypmy = iy + my
    djm0 = DJM0
    djm = float((1461 * (iypmy + 4800)) // 4
                + (367 * (im - 2 - 12 * my)) // 12
                - (3 * ((iypmy + 4900) // 100)) // 4
                + id - 2432076)

    return djm0, djm


def jd2cal(dj1, dj2):
    """Julian Date to Gregorian year, month, day, and fraction of a day.

    Uses Kahan-Neumaier compensated summation to preserve precision.
    """
    DJMIN = -68569.5
    DJMAX = 1e9

    dj = float(dj1) + float(dj2)
    if dj < DJMIN or dj > DJMAX:
        raise ValueError(f"unacceptable date: {dj}")

    # Separate day and fraction (where -0.5 <= fraction < 0.5)
    d = round(float(dj1))
    f1 = float(dj1) - d
    jd = int(d)
    d = round(float(dj2))
    f2 = float(dj2) - d
    jd += int(d)

    # Compute f1+f2+0.5 using compensated summation (Klein 2006)
    s = 0.5
    cs = 0.0
    for x in [f1, f2]:
        t = s + x
        if abs(s) >= abs(x):
            cs += (s - t) + x
        else:
            cs += (x - t) + s
        s = t
        if s >= 1.0:
            jd += 1
            s -= 1.0

    f = s + cs
    cs = f - s

    # Deal with negative f
    if f < 0.0:
        f = s + 1.0
        cs += (1.0 - f) + s
        s = f
        f = s + cs
        cs = f - s
        jd -= 1

    # Deal with f that is 1.0 or more (when rounded to double)
    DBL_EPSILON = np.finfo(np.float64).eps
    if (f - 1.0) >= -DBL_EPSILON / 4.0:
        t = s - 1.0
        cs += (s - t) - 1.0
        s = t
        f = s + cs
        if -DBL_EPSILON / 2.0 < f:
            jd += 1
            f = max(f, 0.0)

    # Express day in Gregorian calendar
    l = jd + 68569
    n = (4 * l) // 146097
    l -= (146097 * n + 3) // 4
    i = (4000 * (l + 1)) // 1461001
    l -= (1461 * i) // 4 - 31
    k = (80 * l) // 2447
    id = int(l - (2447 * k) // 80)
    l = k // 11
    im = int(k + 2 - 12 * l)
    iy = int(100 * (n - 49) + i + l)

    return iy, im, id, f


def jdcalf(ndp, dj1, dj2):
    """Julian Date to Gregorian calendar, rounded to a specified precision.

    ndp: number of decimal places of days in fraction (0-9).
    Returns (iy, im, id, ifd) where ifd is the rounded fraction.
    """
    if ndp < 0 or ndp > 9:
        raise ValueError(f"bad decimal places: {ndp}")

    denom = 10.0 ** ndp

    # Order date parts: big first
    dj1 = float(dj1)
    dj2 = float(dj2)
    if abs(dj1) >= abs(dj2):
        d1 = dj1
        d2 = dj2
    else:
        d1 = dj2
        d2 = dj1

    # Adjust to midnight
    d2 -= 0.5

    # Separate integer and fractional parts
    f1 = d1 % 1.0
    f2 = d2 % 1.0
    d1 = round(d1 - f1)
    d2 = round(d2 - f2)
    f = round((f1 + f2) * denom) / denom

    # Handle overflow of fractional part
    if f >= 1.0:
        f -= 1.0
        d2 += 1.0

    # Realign to noon
    d2 += 0.5

    # Convert to calendar
    iy, im, id, fd = jd2cal(d1, d2 + f)

    ifd = int(round(fd * denom))

    return iy, im, id, ifd


# ---------------------------------------------------------------------------
# Differentiable epoch functions (JAX)
# ---------------------------------------------------------------------------

def epb(dj1, dj2):
    """Julian Date to Besselian Epoch."""
    D1900 = 36524.68648
    return 1900.0 + ((dj1 - DJ00) + (dj2 + D1900)) / DTY


def epj(dj1, dj2):
    """Julian Date to Julian Epoch."""
    return 2000.0 + ((dj1 - DJ00) + dj2) / DJY


def epb2jd(epb_val):
    """Besselian Epoch to Julian Date.

    Returns (djm0, djm) where djm0 = 2400000.5.
    """
    return DJM0, 15019.81352 + (epb_val - 1900.0) * DTY


def epj2jd(epj_val):
    """Julian Epoch to Julian Date.

    Returns (djm0, djm) where djm0 = 2400000.5.
    """
    return DJM0, DJM00 + (epj_val - 2000.0) * 365.25


__all__ = [
    "cal2jd", "jd2cal", "jdcalf",
    "epb", "epj", "epb2jd", "epj2jd",
]
