# so_pointjax.qpoint

Differentiable telescope pointing pipeline in JAX. Reimplements the
[qpoint](https://github.com/arahlin/qpoint) library: quaternion-based
az/el to RA/Dec conversion with atmospheric refraction, aberration,
precession-nutation, and HEALPix pixelization.

All functions are compatible with `jax.jit`, `jax.grad`, and `jax.vmap`.

```python
import so_pointjax.qpoint as qp
```

## Quaternion algebra

Quaternions are represented as JAX arrays of shape `(4,)` in `[w, x, y, z]` order.

### Core operations

```python
qp.identity()              # -> [1, 0, 0, 0]
qp.mul(a, b)               # quaternion product
qp.conj(q)                 # conjugate
qp.inv(q)                  # inverse
qp.norm(q)                 # |q|
qp.norm2(q)                # |q|^2
qp.normalize(q)            # q / |q|
```

### Rotation generators

```python
qp.r1(angle)               # rotation about x-axis
qp.r2(angle)               # rotation about y-axis
qp.r3(angle)               # rotation about z-axis
qp.rot(angle, axis)        # rotation about arbitrary axis (0, 1, or 2)

# Compose with existing quaternion (equivalent to mul(r_i(angle), q))
qp.r1_mul(angle, q)
qp.r2_mul(angle, q)
qp.r3_mul(angle, q)
```

### Quaternion-matrix conversions

```python
qp.to_matrix(q)            # -> (3, 3) rotation matrix
qp.to_col1(q)              # first column of rotation matrix
qp.to_col2(q)              # second column
qp.to_col3(q)              # third column (= rotated z-axis)
```

### Sky coordinate conversions

```python
# Quaternion <-> (RA, Dec, PA) in degrees
ra, dec, pa = qp.quat2radecpa(q)
q = qp.radecpa2quat(ra, dec, pa)

# Quaternion <-> (RA, Dec, sin2psi, cos2psi) -- avoids wrapping
ra, dec, sin2psi, cos2psi = qp.quat2radec(q)
q = qp.radec2quat(ra, dec, sin2psi, cos2psi)
```

### Interpolation

```python
qp.slerp(q0, q1, t)        # spherical linear interpolation, t in [0, 1]
```

## Corrections

Individual physical corrections, each returning a quaternion.

### Precession-nutation and frame bias

```python
qp.npb_quat(jd_tt1, jd_tt2, accuracy=0)
# accuracy: 0 = IAU 2006/2000A, 1 = IAU 2000B (faster, ~1 mas)
```

### Earth rotation

```python
qp.erot_quat(jd_ut1_1, jd_ut1_2)    # Earth Rotation Angle quaternion
```

### Polar motion

```python
qp.wobble_quat(jd_tt1, jd_tt2, xp, yp)   # xp, yp in radians
```

### Observer location

```python
qp.lonlat_quat(lon, lat)            # geodetic lon/lat in radians
```

### Telescope orientation

```python
qp.azel_quat(az, el, pitch=0., roll=0.)
qp.azelpsi_quat(az, el, psi, pitch=0., roll=0.)
```

### Atmospheric refraction

```python
# Refraction angle in degrees
delta = qp.refraction(el, temperature=0., pressure=0., humidity=0., frequency=150e9)

# As a quaternion correction
q_ref = qp.refraction_quat(el, temperature=0., pressure=0., humidity=0.,
                            frequency=150e9, inv=False)
```

### Aberration

```python
# Annual aberration (Earth orbital velocity)
beta = qp.earth_orbital_beta(jd_tdb1, jd_tdb2)     # -> (3,) velocity/c

# Diurnal aberration (observer rotation)
beta = qp.diurnal_aberration_beta(lat)              # -> (3,) velocity/c

# Apply aberration to pointing quaternion
q_corrected = qp.aberration(q, beta, inv=False, fast=False)
```

### Detector and HWP

```python
qp.det_offset_quat(delta_az, delta_el, delta_psi)   # focal plane offset
qp.hwp_quat(ang)                                     # half-wave plate angle
```

## Pointing pipeline

### Forward: az/el to RA/Dec

High-level convenience functions (handle time conversion internally):

```python
# Boresight quaternion from az/el + observer location + time
q_bore = qp.azel2bore(az, el, lon, lat, ctime,
                       dut1=0., weather=None, accuracy=1)

# With explicit boresight rotation angle psi
q_bore = qp.azelpsi2bore(az, el, psi, lon, lat, ctime,
                          pitch=0., roll=0., dut1=0.,
                          weather=None, accuracy=1)
```

Where:
- `az`, `el`, `psi`: radians
- `lon`, `lat`: radians
- `ctime`: Unix timestamp(s)
- `weather`: dict with keys `temperature` (C), `pressure` (mbar), `humidity` (0-1), `frequency` (Hz)
- `accuracy`: 0 = full (IAU 2006/2000A), 1 = fast (IAU 2000B)

### JIT-compatible variants

For use inside `jax.jit` (pre-convert times):

```python
# Pre-convert times once
times = qp.precompute_times(ctime, dut1=0.)
# -> dict: jd_utc1, jd_utc2, tt1, tt2, ut1_1, ut1_2

q_bore = qp.azelpsi2bore_jit(
    az, el, psi, lon, lat,
    times['tt1'], times['tt2'],
    times['ut1_1'], times['ut1_2'],
    weather_A=0., weather_B=0.,
    accuracy=1,
)
```

### Fast variant with precomputed corrections

For tight loops where corrections change slowly:

```python
corrections = qp.precompute_corrections(ctime, dut1=0., accuracy=1,
                                         rate_npb=10., rate_aber=10.)

q_bore = qp.azelpsi2bore_fast(
    az, el, psi, lon, lat,
    times['tt1'], times['tt2'],
    times['ut1_1'], times['ut1_2'],
    corrections['q_npb'], corrections['q_wobble'],
    corrections['beta_earth'],
    weather_A=0., weather_B=0.,
)
```

### Boresight to sky coordinates

```python
# With detector offset quaternion q_off
ra, dec, pa = qp.bore2radecpa(q_off, ctime, q_bore)
ra, dec, sin2psi, cos2psi = qp.bore2radec(q_off, ctime, q_bore)
```

### Complete forward pipeline

```python
ra, dec, pa = qp.azel2radecpa(
    delta_az, delta_el, delta_psi,    # detector offset
    az, el, lon, lat, ctime,          # boresight + observer
    psi=0., pitch=0., roll=0.,
    dut1=0., weather=None, accuracy=1,
)
```

### Inverse: RA/Dec to az/el

```python
az, el, pa = qp.radec2azel(ra, dec, pa, lon, lat, ctime,
                             dut1=0., weather=None, accuracy=1)

# JIT-compatible variant
az, el, pa = qp.radec2azel_jit(ra, dec, pa, lon, lat,
                                 tt1, tt2, ut1_1, ut1_2,
                                 weather_A=0., weather_B=0.)
```

## HEALPix pixelization

Pure-JAX HEALPix implementation (no healpy dependency). All functions are
JIT-compatible.

### Angle-based (theta = colatitude, phi = longitude, both in radians)

```python
pix = qp.ang2pix_nest(nside, theta, phi)
pix = qp.ang2pix_ring(nside, theta, phi)
theta, phi = qp.pix2ang_nest(nside, pix)
theta, phi = qp.pix2ang_ring(nside, pix)
```

### Vector-based

```python
pix = qp.vec2pix_nest(nside, vec)     # vec: (3,) unit vector
pix = qp.vec2pix_ring(nside, vec)
vec = qp.pix2vec_nest(nside, pix)
vec = qp.pix2vec_ring(nside, pix)
```

### Ordering conversion

```python
qp.nest2ring(nside, ipnest)
qp.ring2nest(nside, ipring)
qp.nside2npix(nside)
qp.npix2nside(npix)
```

### Astronomical interface (RA/Dec in degrees)

```python
pix = qp.radec2pix(nside, ra, dec, nest=True)
ra, dec = qp.pix2radec(nside, pix, nest=True)
```

### Pointing integration

```python
# Quaternion -> pixel + polarization angle
pix, sin2psi, cos2psi = qp.quat2pix(q, nside, nest=True)

# Boresight + detector offset -> pixel
pix, sin2psi, cos2psi = qp.bore2pix(q_off, q_bore, nside, nest=True)
```

## Time utilities

```python
qp.ctime2jd(ctime)                  # Unix time -> (jd1, jd2)
qp.jd2ctime(jd1, jd2)               # (jd1, jd2) -> Unix time
qp.ctime2jdtt(ctime)                # Unix time -> TT as (jd1, jd2)
qp.jdutc2jdut1(jd1, jd2, dut1)      # JD(UTC) -> JD(UT1)
qp.ctime2gmst(ctime, dut1=0., accuracy=0)  # -> GMST in radians
```

## IERS Bulletin A

Load Earth orientation parameters for high-accuracy pointing:

```python
# Download from IERS (requires internet)
iers_data = qp.update_bulletin_a(start_year=2000)

# Or load from local file
iers_data = qp.load_bulletin_a(filename)

# Interpolate for a given MJD
dut1, xp, yp = qp.interpolate_bulletin_a(iers_data, mjd)
```

## QPoint: stateful high-level API

`QPoint` wraps the functional API with persistent state for observatory
parameters:

```python
from so_pointjax.qpoint import QPoint

Q = QPoint(accuracy=1, mean_aber=True)
Q.set(weather={'temperature': 0., 'pressure': 550., 'humidity': 0.2})
Q.update_bulletin_a(start_year=2000)

q_bore = Q.azel2bore(az, el, pitch, roll, lon, lat, ctime)
ra, dec, pa = Q.bore2radecpa(q_off, ctime, q_bore)
```

`QPoint` methods accept the same keyword arguments as the functional API;
stored state provides defaults.

## Conventions

- Quaternions: `[w, x, y, z]` order
- Angles: radians internally, degrees for RA/Dec/PA user-facing outputs
- Time: Unix timestamps (seconds since 1970-01-01 UTC) for `ctime`,
  two-part Julian Dates internally
- Coordinate system: IAU conventions (ICRS, CIRS)
