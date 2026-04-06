# so_pointjax.proj

High-level API for differentiable telescope pointing, modeled after
`so3g.proj`. Provides the `Quat` class with operator overloading,
`CelestialSightLine` for boresight pointing, `FocalPlane` for detector
layouts, and observatory site/weather configuration.

All classes are JAX pytrees and work with `jax.jit`, `jax.grad`, and `jax.vmap`.

```python
from so_pointjax.proj import Quat, CelestialSightLine, FocalPlane, Assembly
from so_pointjax.proj import SITES, Weather, weather_factory
import jax.numpy as jnp
```

## Quat class

`Quat` wraps a JAX array of shape `(4,)` or `(N, 4)` with quaternion
arithmetic, automatic broadcasting, and operator overloading.

### Construction

```python
# Direct
q = Quat(1, 0, 0, 0)                              # (w, x, y, z)
q = Quat(jnp.array([1., 0., 0., 0.]))              # from array

# Named constructors
q = Quat.identity()                                 # (1, 0, 0, 0)
q = Quat.from_euler(axis, angle)                     # axis in {0, 1, 2}
q = Quat.from_iso(theta, phi, psi)                   # Rz(phi) Ry(theta) Rz(psi)
q = Quat.from_lonlat(lon, lat, psi=0.)               # celestial lon/lat
q = Quat.from_lonlat(az, el, azel=True)              # horizon az/el
q = Quat.from_xieta(xi, eta, gamma=0.)               # focal plane offset

# All constructors accept arrays for batch construction
q_arr = Quat.from_euler(2, jnp.linspace(0, 1, 1000))  # shape (1000, 4)
```

### Operators and broadcasting

```python
q1 * q2          # quaternion multiplication
~q               # conjugate (= inverse for unit quaternions)
abs(q)           # norm (scalar or array of norms)
-q               # negation
2 * q            # scalar multiplication

# Broadcasting: (4,) x (N, 4) -> (N, 4)
q_scalar * q_arr          # scalar x array
q_arr * q_scalar          # array x scalar
q_arr * ~q_arr            # element-wise -> all identity
q1_arr * q2_arr           # (N, 4) x (N, 4) element-wise

# Chaining
q1 * q2 * q3              # left-to-right associativity
```

### Decomposition

```python
theta, phi, psi = q.to_iso()              # ZYZ Euler angles
lon, lat, psi   = q.to_lonlat()           # celestial
az, el, psi     = q.to_lonlat(azel=True)  # horizon
xi, eta, gamma  = q.to_xieta()            # focal plane
```

### Other methods

```python
q.normalized()         # return unit quaternion
q.rotate(v)            # rotate vector(s) v by q: q v q*
q.numpy()              # convert to numpy array
np.array(q)            # also works
jnp.asarray(q)         # to JAX array
```

### Component access

```python
q.a, q.b, q.c, q.d    # (w, x, y, z) -- matches spt3g convention
q.w, q.x, q.y, q.z    # aliases
```

### Indexing and slicing

```python
q_arr[0]               # -> scalar Quat
q_arr[2:5]             # -> batch Quat (3 elements)
q_arr[-1]              # last element
len(q_arr)             # number of quaternions in batch
```

### JAX compatibility

`Quat` is registered as a JAX pytree and works transparently with
`jax.jit`, `jax.grad`, and `jax.vmap`:

```python
import jax

# JIT
@jax.jit
def compose(q1, q2):
    return q1 * q2

# Gradient
def loss(angle):
    q = Quat.from_euler(2, angle)
    return q.to_lonlat()[0]  # lon
jax.grad(loss)(1.0)

# Vectorized map
jax.vmap(lambda a: abs(Quat.from_euler(2, a)))(angles)
```

## Module-level quaternion functions

The `quat` module provides the same operations as free functions operating
on raw `jnp.ndarray` of shape `(4,)` or `(N, 4)`:

```python
from so_pointjax.proj import quat

# Construction
quat.euler(axis, angle)
quat.rotation_iso(theta, phi, psi)
quat.rotation_lonlat(lon, lat, psi=0., azel=False)
quat.rotation_xieta(xi, eta, gamma=0.)

# Decomposition
quat.decompose_iso(q)       # -> (theta, phi, psi)
quat.decompose_lonlat(q)    # -> (lon, lat, psi)
quat.decompose_xieta(q)     # -> (xi, eta, gamma)

# Arithmetic
quat.qmul(a, b)             # quaternion product
quat.qconj(q)               # conjugate
quat.qnorm(q)               # norm
quat.qnormalize(q)          # normalize
quat.qrotate(q, v)          # rotate vector
```

## Pointing pipeline

### CelestialSightLine

Carries a vector of celestial pointing quaternions (`self.Q`).

```python
from so_pointjax.proj import CelestialSightLine

# High-precision pointing (nutation, precession, aberration, refraction)
csl = CelestialSightLine.az_el(t, az, el, site='act', weather='toco')

# Fast approximate pointing (ERA-based, ~arcminute accuracy)
csl = CelestialSightLine.naive_az_el(t, az, el, site='act')

# From celestial coordinates directly
csl = CelestialSightLine.for_lonlat(lon, lat, psi=0.)

# From horizon coordinates (no Earth rotation applied)
csl = CelestialSightLine.for_horizon(t, az, el)
```

Arguments:
- `t`: Unix timestamps (float or array)
- `az`, `el`: azimuth and elevation in radians
- `site`: observatory name (`'act'`, `'so'`, `'so_lat'`, `'so_sat1'`,
  `'so_sat2'`, `'so_sat3'`) or `EarthlySite` object
- `weather`: atmosphere model (`'vacuum'`, `'toco'`, `'act'`, `'so'`,
  `'sa'`, `'typical'`) or `Weather` object

### Extracting sky coordinates

```python
# Boresight coordinates -> (N, 4) array [lon, lat, cos2psi, sin2psi]
coords = csl.coords()

# With focal plane -> (n_det, N, 4)
fp = FocalPlane.from_xieta(xi, eta)
coords = csl.coords(fplane=fp)
```

### FocalPlane

Describes detector positions and polarization responses in the focal plane.

```python
from so_pointjax.proj import FocalPlane

# Single boresight detector
fp = FocalPlane.boresight()

# From tangent plane coordinates
fp = FocalPlane.from_xieta(xi, eta, gamma=0., T=1., P=1.)

# Properties
fp.ndet              # number of detectors
fp.quats             # (n_det, 4) detector quaternions
fp.resps             # (n_det, 2) [T, P] responsivities
fp[1:]               # slice to subset of detectors
len(fp)              # same as fp.ndet
```

### Assembly

Groups boresight pointing with detector offsets.

```python
from so_pointjax.proj import Assembly

asm = Assembly.attach(csl, fp)        # full assembly
asm = Assembly.for_boresight(csl)     # boresight-only
```

## Weather

```python
from so_pointjax.proj import Weather, weather_factory

w = weather_factory('toco')           # preset: toco/act/so/sa/vacuum
w = Weather({'temperature': 10., 'pressure': 600., 'humidity': 0.5})
```

## Sites

```python
from so_pointjax.proj import SITES

SITES['act']        # EarthlySite(lon=-67.7876, lat=-22.9585, elev=5188)
SITES['so']         # alias for 'so_lat'
SITES['so_lat']     # SO Large Aperture Telescope
SITES['so_sat1']    # SO Small Aperture Telescopes 1-3
```

Each site has `lon` (degrees), `lat` (degrees), `elev` (meters), and
`typical_weather` (a `Weather` object).

## End-to-end differentiable example

```python
import jax
import jax.numpy as jnp
from so_pointjax.proj import Quat, CelestialSightLine, FocalPlane

def sky_lon(az, el):
    """Differentiable map from az/el to RA."""
    csl = CelestialSightLine.naive_az_el(
        jnp.array([1700000000.0]),
        jnp.array([az]),
        jnp.array([el]),
        site='act',
    )
    coords = csl.coords()
    return coords[0, 0]   # RA of first sample

# Gradient of RA w.r.t. az and el
dra_daz, dra_del = jax.grad(sky_lon, argnums=(0, 1))(1.0, 0.8)
```

## Precision

Validated against `so3g` to sub-milliarcsecond accuracy across all sites
and weather conditions (93 cross-validation tests):

| Layer                        | Agreement            |
|------------------------------|----------------------|
| Quaternion functions         | Bit-identical        |
| `naive_az_el`                | ~1e-12 (quat diff)   |
| `az_el` (all weather/sites)  | 0.0004--0.0005 arcsec |
| Detector projection          | ~1e-12               |

## Running tests

```bash
# Core tests (no so3g dependency needed)
python -m pytest so_pointjax/proj/tests/test_quat.py so_pointjax/proj/tests/test_coords.py -v

# Cross-validation against so3g (requires so3g)
python -m pytest so_pointjax/proj/tests/test_cross_validation.py -v -s

# All proj tests
python -m pytest so_pointjax/proj/tests/ -v
```
