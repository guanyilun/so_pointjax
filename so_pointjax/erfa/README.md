# so_pointjax.erfa

Differentiable reimplementation of ERFA (Essential Routines for Fundamental
Astronomy) in JAX. All functions are JIT-compatible and most are differentiable
with `jax.grad`.

```python
import so_pointjax.erfa as erfa
```

Everything is exported at the top level, so `erfa.nut06a(...)` works directly.

## Constants

Standard ERFA constants are available as module-level variables:

```python
erfa.DPI          # pi
erfa.D2PI         # 2*pi
erfa.DD2R         # degrees to radians
erfa.DR2D         # radians to degrees
erfa.DAS2R        # arcseconds to radians
erfa.DR2AS        # radians to arcseconds
erfa.DS2R         # seconds of time to radians
erfa.DMAS2R       # milliarcseconds to radians
erfa.TURNAS       # arcseconds in a full circle

erfa.DJ00         # J2000.0 as Julian Date
erfa.DJM0         # MJD zero-point
erfa.DJM00        # J2000.0 as MJD
erfa.DJY          # days per Julian year
erfa.DJC          # days per Julian century
erfa.DJM          # days per Julian millennium
erfa.DAYSEC       # seconds per day

erfa.DAU          # astronomical unit (m)
erfa.CMPS         # speed of light (m/s)
erfa.AULT         # light time for 1 au (s)
erfa.DC           # speed of light (au/day)

erfa.ELG, erfa.ELB, erfa.TDB0   # relativistic parameters
erfa.SRS          # Schwarzschild radius of Sun (au)
erfa.TTMTAI       # TT - TAI (s)

erfa.WGS84, erfa.GRS80, erfa.WGS72   # reference ellipsoid IDs
```

Utility functions for numerical work:

```python
erfa.dint(a)       # truncate toward zero
erfa.dnint(a)      # round to nearest integer
erfa.dsign(a, b)   # |a| with sign of b
```

## Angles

```python
erfa.anp(a)        # normalize to [0, 2*pi)
erfa.anpm(a)       # normalize to [-pi, +pi)
```

## Vector and matrix operations

All operate on JAX arrays. Vectors are shape `(3,)`, matrices `(3, 3)`,
position-velocity vectors `(2, 3)`.

### Vector arithmetic

```python
erfa.ppp(a, b)         # a + b
erfa.pmp(a, b)         # a - b
erfa.sxp(s, p)         # s * p
erfa.ppsp(a, s, b)     # a + s*b
erfa.pdp(a, b)         # dot product
erfa.pxp(a, b)         # cross product
erfa.pm(p)             # modulus |p|
erfa.pn(p)             # -> (modulus, unit vector)
```

### Spherical-Cartesian conversions

```python
erfa.s2c(theta, phi)              # spherical -> unit vector
erfa.c2s(p)                       # unit vector -> (theta, phi)
erfa.s2p(theta, phi, r)           # spherical + distance -> vector
erfa.p2s(p)                       # vector -> (theta, phi, r)
erfa.s2pv(theta, phi, r, td, pd, rd)  # spherical -> pv-vector
erfa.pv2s(pv)                     # pv-vector -> spherical
```

### Angular separation and position angle

```python
erfa.sepp(a, b)              # separation between vectors
erfa.seps(al, ap, bl, bp)   # separation from spherical coords
erfa.pap(a, b)               # position angle from vectors
erfa.pas(al, ap, bl, bp)    # position angle from spherical coords
```

### Rotation matrices

```python
erfa.ir()              # identity matrix
erfa.rx(phi, r)        # rotate about x
erfa.ry(theta, r)      # rotate about y
erfa.rz(psi, r)        # rotate about z
erfa.rxr(a, b)         # matrix product a @ b
erfa.tr(r)             # transpose
erfa.rxp(r, p)         # r @ p
erfa.rxpv(r, pv)       # r @ pv
erfa.trxp(r, p)        # r^T @ p
erfa.trxpv(r, pv)      # r^T @ pv
erfa.rv2m(w)           # rotation vector -> matrix
erfa.rm2v(r)           # matrix -> rotation vector
```

### PV-vector operations

```python
erfa.p2pv(p)           # extend p to pv (zero velocity)
erfa.pv2p(pv)          # discard velocity
erfa.pvppv(a, b)       # pv addition
erfa.pvmpv(a, b)       # pv subtraction
erfa.sxpv(s, pv)       # scalar * pv
erfa.s2xpv(s1, s2, pv) # scale position by s1, velocity by s2
erfa.pvu(dt, pv)       # propagate pv by dt
erfa.pvup(dt, pv)      # propagate, return position only
erfa.pvdpv(a, b)       # inner product of pv-vectors
erfa.pvm(pv)           # -> (position modulus, velocity modulus)
erfa.pvxpv(a, b)       # outer product of pv-vectors
```

## Calendar and epochs

```python
# Gregorian <-> Julian Date (non-differentiable)
erfa.cal2jd(iy, im, id)        # -> (djm0, djm)
erfa.jd2cal(dj1, dj2)          # -> (iy, im, id, fraction)
erfa.jdcalf(ndp, dj1, dj2)     # -> (iy, im, id, ifd) rounded

# Epoch conversions (differentiable)
erfa.epb(dj1, dj2)             # JD -> Besselian epoch
erfa.epj(dj1, dj2)             # JD -> Julian epoch
erfa.epb2jd(epb)               # Besselian epoch -> JD
erfa.epj2jd(epj)               # Julian epoch -> JD
```

## Time scales

Two-part Julian Date representation `(jd1, jd2)` is used throughout for
numerical precision.

### Differentiable conversions (JIT-compatible)

```python
erfa.taitt(tai1, tai2)            # TAI -> TT
erfa.tttai(tt1, tt2)              # TT -> TAI
erfa.taiut1(tai1, tai2, dta)      # TAI -> UT1
erfa.ut1tai(ut11, ut12, dta)      # UT1 -> TAI
erfa.ttut1(tt1, tt2, dt)          # TT -> UT1
erfa.ut1tt(ut11, ut12, dt)        # UT1 -> TT
erfa.tttdb(tt1, tt2, dtr)         # TT -> TDB
erfa.tdbtt(tdb1, tdb2, dtr)       # TDB -> TT
erfa.tcgtt(tcg1, tcg2)            # TCG -> TT
erfa.tttcg(tt1, tt2)              # TT -> TCG
erfa.tcbtdb(tcb1, tcb2)           # TCB -> TDB
erfa.tdbtcb(tdb1, tdb2)           # TDB -> TCB
```

### UTC conversions (non-differentiable, involve leap seconds)

```python
erfa.utctai(utc1, utc2)           # UTC -> TAI
erfa.taiutc(tai1, tai2)           # TAI -> UTC
erfa.utcut1(utc1, utc2, dut1)     # UTC -> UT1
erfa.ut1utc(ut11, ut12, dut1)     # UT1 -> UTC
erfa.dat(iy, im, id, fd)          # Delta(AT) = TAI - UTC
```

### Earth rotation

```python
erfa.era00(dj1, dj2)                    # Earth Rotation Angle (IAU 2000)
erfa.gmst00(uta, utb, tta, ttb)          # GMST (IAU 2000)
erfa.gmst06(uta, utb, tta, ttb)          # GMST (IAU 2006)
erfa.gmst82(dj1, dj2)                   # GMST (IAU 1982)
erfa.gst00a(uta, utb, tta, ttb)          # GAST (IAU 2000A)
erfa.gst00b(uta, utb)                    # GAST (IAU 2000B)
erfa.gst06(uta, utb, tta, ttb, rnpb)     # GAST (IAU 2006)
erfa.gst06a(uta, utb, tta, ttb)          # GAST (IAU 2006/2000A)
erfa.gst94(uta, utb)                     # GAST (IAU 1982/1994)
```

## Precession and nutation

### Fundamental arguments (IERS 2003)

All take `t` (TDB Julian centuries from J2000.0):

```python
erfa.fal03(t)     # Moon mean anomaly
erfa.falp03(t)    # Sun mean anomaly
erfa.faf03(t)     # Moon argument of latitude
erfa.fad03(t)     # Moon mean elongation
erfa.faom03(t)    # Moon ascending node longitude
erfa.fame03(t)    # Mercury mean longitude
erfa.fave03(t)    # Venus mean longitude
erfa.fae03(t)     # Earth mean longitude
erfa.fama03(t)    # Mars mean longitude
erfa.faju03(t)    # Jupiter mean longitude
erfa.fasa03(t)    # Saturn mean longitude
erfa.faur03(t)    # Uranus mean longitude
erfa.fane03(t)    # Neptune mean longitude
erfa.fapa03(t)    # general accumulated precession
```

### Obliquity

```python
erfa.obl80(date1, date2)    # mean obliquity (IAU 1980)
erfa.obl06(date1, date2)    # mean obliquity (IAU 2006)
```

### Nutation models

```python
erfa.nut80(date1, date2)     # -> (dpsi, deps)  IAU 1980
erfa.nut00a(date1, date2)    # -> (dpsi, deps)  IAU 2000A
erfa.nut00b(date1, date2)    # -> (dpsi, deps)  IAU 2000B
erfa.nut06a(date1, date2)    # -> (dpsi, deps)  IAU 2006/2000A
erfa.numat(epsa, dpsi, deps) # nutation matrix from angles
```

### Precession

```python
erfa.pfw06(date1, date2)         # Fukushima-Williams angles -> (gamb, phib, psi, eps)
erfa.pmat06(date1, date2)        # precession matrix (IAU 2006)
erfa.pmat76(date1, date2)        # precession matrix (IAU 1976)
erfa.prec76(d01, d02, d11, d12)  # precession Euler angles (IAU 1976)
erfa.pr00(date1, date2)          # precession rate corrections -> (dpsipr, depspr)
```

### Bias-precession-nutation composites

```python
# Full decomposition -> (epsa, rb, rp, rbp, rn, rbpn)
erfa.pn00a(date1, date2)     # IAU 2000A
erfa.pn00b(date1, date2)     # IAU 2000B
erfa.pn06a(date1, date2)     # IAU 2006/2000A

# Combined matrix only
erfa.pnm00a(date1, date2)    # IAU 2000A
erfa.pnm00b(date1, date2)    # IAU 2000B
erfa.pnm06a(date1, date2)    # IAU 2006/2000A
erfa.pnm80(date1, date2)     # IAU 1980
```

### CIO and equation of origins

```python
erfa.s06a(date1, date2)            # CIO locator s (IAU 2006/2000A)
erfa.xys06a(date1, date2)          # CIP X,Y + s -> (x, y, s)
erfa.c2ixys(x, y, s)              # CIP -> celestial-intermediate matrix
erfa.eors(rnpb, s)                 # equation of origins
```

### Equation of equinoxes

```python
erfa.ee00a(date1, date2)     # IAU 2000A
erfa.ee00b(date1, date2)     # IAU 2000B
erfa.eqeq94(date1, date2)   # IAU 1994
```

### Celestial-to-terrestrial matrices

```python
erfa.c2t06a(tta, ttb, uta, utb, xp, yp)   # CIO-based (IAU 2006/2000A)
erfa.c2t00a(tta, ttb, uta, utb, xp, yp)   # CIO-based (IAU 2000A)
erfa.c2t00b(tta, ttb, uta, utb, xp, yp)   # CIO-based (IAU 2000B)
erfa.pom00(xp, yp, sp)                     # polar motion matrix
erfa.sp00(date1, date2)                     # TIO locator s'
```

## Ephemerides

```python
erfa.epv00(date1, date2)            # Earth pos/vel (barycentric + heliocentric)
erfa.moon98(date1, date2)           # Moon pos/vel (geocentric)
erfa.plan94(date1, date2, planet)   # planet pos/vel (1=Mercury .. 8=Neptune)
```

## Geodetic transforms

```python
erfa.eform(n)                          # ellipsoid params -> (a, f)
erfa.gd2gc(n, elong, phi, height)      # geodetic -> geocentric
erfa.gc2gd(n, xyz)                     # geocentric -> geodetic
erfa.gd2gce(a, f, elong, phi, height)  # geodetic -> geocentric (general)
erfa.gc2gde(a, f, xyz)                 # geocentric -> geodetic (general)
```

## Astrometry

### ASTROM context

The `ASTROM` named tuple holds star-independent astrometry parameters:

```python
# Auto-computed contexts
astrom = erfa.apci13(date1, date2)                  # ICRS -> CIRS
astrom = erfa.apco13(utc1, utc2, dut1, elong, phi,  # ICRS -> observed
                     hm, xp, yp, phpa, tc, rh, wl)
astrom = erfa.apcg13(date1, date2)                  # geocentric
astrom = erfa.apio13(utc1, utc2, dut1, elong, phi,  # CIRS -> observed
                     hm, xp, yp, phpa, tc, rh, wl)

# Manual contexts (for JIT)
astrom = erfa.apci(date1, date2, ebpv, ehp, x, y, s)
astrom = erfa.apco(date1, date2, ebpv, ehp, x, y, s,
                   theta, elong, phi, hm, xp, yp, sp, refa, refb)
```

### Quick transforms (given ASTROM)

```python
ri, di = erfa.atciq(rc, dc, pr, pd, px, rv, astrom)    # ICRS -> CIRS
ri, di = erfa.atciqz(rc, dc, astrom)                    # ICRS -> CIRS (zero pm)
rc, dc = erfa.aticq(ri, di, astrom)                     # CIRS -> ICRS
aob, zob, hob, dob, rob = erfa.atioq(ri, di, astrom)   # CIRS -> observed
ri, di = erfa.atoiq(type, ob1, ob2, astrom)             # observed -> CIRS
```

### Complete transforms

```python
ri, di = erfa.atci13(rc, dc, pr, pd, px, rv, date1, date2)
rc, dc = erfa.atic13(ri, di, date1, date2)
aob, zob, hob, dob, rob = erfa.atco13(rc, dc, pr, pd, px, rv,
                                        utc1, utc2, dut1, elong, phi,
                                        hm, xp, yp, phpa, tc, rh, wl)
```

### Refraction and aberration

```python
erfa.refco(phpa, tc, rh, wl)           # -> (refa, refb) refraction constants
erfa.ab(pnat, v, s, bm1)              # stellar aberration
erfa.ldsun(p, e, em)                   # light deflection by Sun
erfa.ldn(n, b, ob, sc)                # light deflection by multiple bodies
```

## Coordinate frames

### Horizon-equatorial

```python
erfa.ae2hd(az, el, phi)      # az/el -> hour angle/dec
erfa.hd2ae(ha, dec, phi)     # hour angle/dec -> az/el
erfa.hd2pa(ha, dec, phi)     # parallactic angle
```

### Galactic

```python
erfa.icrs2g(dr, dd)     # ICRS -> Galactic
erfa.g2icrs(dl, db)     # Galactic -> ICRS
```

### Ecliptic

```python
erfa.ecm06(date1, date2)              # ICRS -> ecliptic matrix
erfa.eqec06(date1, date2, dr, dd)     # ICRS -> ecliptic coords
erfa.eceq06(date1, date2, dl, db)     # ecliptic -> ICRS coords
```

### FK4/FK5/Hipparcos

```python
erfa.fk52h(r5, d5, dr5, dd5, px5, rv5)    # FK5 -> Hipparcos
erfa.h2fk5(rh, dh, drh, ddh, pxh, rvh)    # Hipparcos -> FK5
erfa.fk425(r, d, dr, dd, p, v)             # FK4 -> FK5
erfa.fk524(r, d, dr, dd, p, v)             # FK5 -> FK4
```

## Gnomonic (tangent-plane) projections

```python
erfa.tpxes(a, b, a0, b0)        # spherical -> tangent plane -> (xi, eta)
erfa.tpxev(v, v0)               # direction cosines -> tangent plane
erfa.tpsts(xi, eta, a0, b0)     # tangent plane -> spherical
erfa.tpstv(xi, eta, v0)         # tangent plane -> direction cosines
erfa.tpors(xi, eta, a, b)       # solve for tangent point (spherical)
erfa.tporv(xi, eta, v)          # solve for tangent point (vectors)
```

## Data types

```python
from so_pointjax.erfa import ASTROM, LDBODY

# ASTROM: star-independent astrometry parameters (NamedTuple)
# Fields: pmt, eb, eh, em, v, bm1, bpn, along, phi, xpl, ypl,
#         sphi, cphi, diurab, eral, refa, refb

# LDBODY: body parameters for light deflection (NamedTuple)
# Fields: bm (solar masses), dl (deflection limiter), pv (pos/vel)
```

## JAX compatibility

All differentiable functions work with `jax.jit`, `jax.grad`, and `jax.vmap`.
Functions involving UTC/leap-second lookups (e.g., `utctai`, `dat`) use plain
Python and are not JIT-compatible.

```python
import jax

# Differentiate nutation w.r.t. date
jax.grad(lambda d: erfa.nut06a(2451545.0, d)[0])(0.0)

# JIT-compile time conversion
fast_taitt = jax.jit(erfa.taitt)
```
