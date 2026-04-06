"""Leap-second table and Delta(AT) = TAI-UTC lookup.

This module is non-differentiable by nature (discrete table lookup).
Uses plain Python/NumPy, not JAX-traced.
"""

from so_pointjax.erfa._core.calendar import cal2jd

# Release year for this version of dat
_IYV = 2023

# Leap-second / TAI-UTC table: (year, month, delta_AT)
_CHANGES = [
    (1960,  1,  1.4178180),
    (1961,  1,  1.4228180),
    (1961,  8,  1.3728180),
    (1962,  1,  1.8458580),
    (1963, 11,  1.9458580),
    (1964,  1,  3.2401300),
    (1964,  4,  3.3401300),
    (1964,  9,  3.4401300),
    (1965,  1,  3.5401300),
    (1965,  3,  3.6401300),
    (1965,  7,  3.7401300),
    (1965,  9,  3.8401300),
    (1966,  1,  4.3131700),
    (1968,  2,  4.2131700),
    (1972,  1, 10.0),
    (1972,  7, 11.0),
    (1973,  1, 12.0),
    (1974,  1, 13.0),
    (1975,  1, 14.0),
    (1976,  1, 15.0),
    (1977,  1, 16.0),
    (1978,  1, 17.0),
    (1979,  1, 18.0),
    (1980,  1, 19.0),
    (1981,  7, 20.0),
    (1982,  7, 21.0),
    (1983,  7, 22.0),
    (1985,  7, 23.0),
    (1988,  1, 24.0),
    (1990,  1, 25.0),
    (1991,  1, 26.0),
    (1992,  7, 27.0),
    (1993,  7, 28.0),
    (1994,  7, 29.0),
    (1996,  1, 30.0),
    (1997,  7, 31.0),
    (1999,  1, 32.0),
    (2006,  1, 33.0),
    (2009,  1, 34.0),
    (2012,  7, 35.0),
    (2015,  7, 36.0),
    (2017,  1, 37.0),
]

# Number of pre-leap-second entries (before 1972)
_NERA1 = 14

# Reference dates (MJD) and drift rates (s/day) for pre-1972 entries
_DRIFT = [
    (37300.0, 0.0012960),
    (37300.0, 0.0012960),
    (37300.0, 0.0012960),
    (37665.0, 0.0011232),
    (37665.0, 0.0011232),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (38761.0, 0.0012960),
    (39126.0, 0.0025920),
    (39126.0, 0.0025920),
]


def dat(iy, im, id, fd):
    """For a given UTC date, calculate Delta(AT) = TAI-UTC.

    Parameters
    ----------
    iy : int
        UTC year.
    im : int
        UTC month.
    id : int
        UTC day.
    fd : float
        Fraction of day (used only pre-1972).

    Returns
    -------
    deltat : float
        TAI minus UTC in seconds.

    Raises
    ------
    ValueError
        If the date is invalid.
    """
    if fd < 0.0 or fd > 1.0:
        raise ValueError(f"bad fraction of day: {fd}")

    # Convert date to MJD (also validates year/month/day)
    djm0, djm = cal2jd(iy, im, id)

    # Pre-UTC year warning
    if iy < _CHANGES[0][0]:
        raise ValueError(f"year {iy} predates UTC")

    # Combine year and month for ordered comparison
    m = 12 * iy + im

    # Find the preceding table entry
    i = -1
    for idx in range(len(_CHANGES) - 1, -1, -1):
        if m >= 12 * _CHANGES[idx][0] + _CHANGES[idx][1]:
            i = idx
            break

    if i < 0:
        raise ValueError("internal error in dat")

    # Get the Delta(AT)
    da = _CHANGES[i][2]

    # If pre-1972, adjust for drift
    if i < _NERA1:
        da += (djm + fd - _DRIFT[i][0]) * _DRIFT[i][1]

    return da


__all__ = ["dat"]
