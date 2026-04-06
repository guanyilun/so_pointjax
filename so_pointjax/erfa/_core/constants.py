"""ERFA constants, ported from erfam.h."""

import jax.numpy as jnp

# Pi
DPI = 3.141592653589793238462643

# 2Pi
D2PI = 6.283185307179586476925287

# Radians to degrees
DR2D = 57.29577951308232087679815

# Degrees to radians
DD2R = 1.745329251994329576923691e-2

# Radians to arcseconds
DR2AS = 206264.8062470963551564734

# Arcseconds to radians
DAS2R = 4.848136811095359935899141e-6

# Seconds of time to radians
DS2R = 7.272205216643039903848712e-5

# Arcseconds in a full circle
TURNAS = 1296000.0

# Milliarcseconds to radians
DMAS2R = DAS2R / 1e3

# Length of tropical year B1900 (days)
DTY = 365.242198781

# Seconds per day
DAYSEC = 86400.0

# Days per Julian year
DJY = 365.25

# Days per Julian century
DJC = 36525.0

# Days per Julian millennium
DJM = 365250.0

# Reference epoch (J2000.0), Julian Date
DJ00 = 2451545.0

# Julian Date of Modified Julian Date zero
DJM0 = 2400000.5

# Reference epoch (J2000.0), Modified Julian Date
DJM00 = 51544.5

# 1977 Jan 1.0 as MJD
DJM77 = 43144.0

# TT minus TAI (s)
TTMTAI = 32.184

# Astronomical unit (m, IAU 2012)
DAU = 149597870.7e3

# Speed of light (m/s)
CMPS = 299792458.0

# Light time for 1 au (s)
AULT = DAU / CMPS

# Speed of light (au per day)
DC = DAYSEC / AULT

# L_G = 1 - d(TT)/d(TCG)
ELG = 6.969290134e-10

# L_B = 1 - d(TDB)/d(TCB), and TDB (s) at TAI 1977/1/1.0
ELB = 1.550519768e-8
TDB0 = -6.55e-5

# Schwarzschild radius of the Sun (au)
SRS = 1.97412574336e-8

# Reference ellipsoids
WGS84 = 1
GRS80 = 2
WGS72 = 3


def dint(a):
    """Truncate to nearest whole number towards zero (like C's (int) cast)."""
    return jnp.where(a < 0.0, jnp.ceil(a), jnp.floor(a))


def dnint(a):
    """Round to nearest whole number."""
    return jnp.where(jnp.abs(a) < 0.5, 0.0,
                     jnp.where(a < 0.0, jnp.ceil(a - 0.5), jnp.floor(a + 0.5)))


def dsign(a, b):
    """Magnitude of A with sign of B."""
    return jnp.where(b < 0.0, -jnp.abs(a), jnp.abs(a))


__all__ = [
    "DPI", "D2PI", "DR2D", "DD2R", "DR2AS", "DAS2R", "DS2R", "TURNAS",
    "DMAS2R", "DTY", "DAYSEC", "DJY", "DJC", "DJM", "DJ00", "DJM0",
    "DJM00", "DJM77", "TTMTAI", "DAU", "CMPS", "AULT", "DC", "ELG",
    "ELB", "TDB0", "SRS", "WGS84", "GRS80", "WGS72",
    "dint", "dnint", "dsign",
]
