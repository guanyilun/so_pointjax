# so_pointjax

Differentiable telescope pointing in JAX. A pure-Python, GPU-ready reimplementation
of the ERFA/qpoint/so3g pointing stack, fully compatible with `jax.jit`, `jax.grad`,
and `jax.vmap`.

## Architecture

```
so_pointjax
├── erfa     Low-level ERFA routines (time, precession, nutation, astrometry, ...)
├── qpoint   Mid-level pointing pipeline (az/el → RA/Dec, HEALPix, IERS)
└── proj     High-level API (Quat, CelestialSightLine, FocalPlane, Assembly)
```

Each layer builds on the one below. Use `proj` for most telescope work;
reach into `qpoint` or `erfa` when you need finer control.

## Installation

```bash
pip install so-pointjax          # from PyPI (when published)
pip install -e .                 # editable install from source
```

Dependencies: `jax >= 0.4.0`, `jaxlib >= 0.4.0`. Tests additionally need
`pytest >= 7.0` and `pyerfa >= 2.0`.

## Quick start

```python
import jax
import jax.numpy as jnp
from so_pointjax.proj import Quat, CelestialSightLine, FocalPlane

# Build a pointing sightline from az/el + timestamps
t  = jnp.array([1700000000.0, 1700000001.0])
az = jnp.array([1.0, 1.01])
el = jnp.array([0.8, 0.8])

csl = CelestialSightLine.az_el(t, az, el, site='act', weather='toco')

# Extract sky coordinates: (N, 4) → [lon, lat, cos2psi, sin2psi]
coords = csl.coords()

# With a focal plane of detectors
fp = FocalPlane.from_xieta(
    jnp.array([0.0, 0.01, -0.01]),
    jnp.array([0.0, 0.01,  0.01]),
)
det_coords = csl.coords(fplane=fp)   # (n_det, N, 4)
```

## End-to-end differentiable pointing

The entire pipeline is differentiable. Compute gradients of sky coordinates
with respect to any input:

```python
def sky_lon(az, el):
    csl = CelestialSightLine.naive_az_el(
        jnp.array([1700000000.0]),
        jnp.array([az]),
        jnp.array([el]),
        site='act',
    )
    return csl.coords()[0, 0]   # RA of first sample

dra_daz, dra_del = jax.grad(sky_lon, argnums=(0, 1))(1.0, 0.8)
```

## Quaternion algebra

The `Quat` class wraps JAX arrays with quaternion arithmetic, broadcasting,
and operator overloading:

```python
from so_pointjax.proj import Quat

q1 = Quat.from_lonlat(1.0, 0.5)
q2 = Quat.from_euler(2, 0.1)
q  = q1 * q2          # quaternion product
q_inv = ~q             # conjugate/inverse

# Batch operations with automatic broadcasting
q_arr = Quat.from_euler(2, jnp.linspace(0, 1, 1000))
rotated = q1 * q_arr   # (4,) x (1000, 4) -> (1000, 4)
```

`Quat` is a JAX pytree and works transparently with `jit`, `grad`, and `vmap`.

## Submodule guides

Each submodule has its own detailed README:

- **[`so_pointjax.erfa`](so_pointjax/erfa/README.md)** -- Differentiable ERFA:
  time scales, precession-nutation, astrometry, ephemerides, coordinate frames,
  geodetic transforms, and more (~200 functions).

- **[`so_pointjax.qpoint`](so_pointjax/qpoint/README.md)** -- Pointing pipeline:
  quaternion algebra, atmospheric/aberration corrections, az/el to RA/Dec
  conversion, HEALPix pixelization, IERS Bulletin A support.

- **[`so_pointjax.proj`](so_pointjax/proj/README.md)** -- High-level API:
  `Quat` class with operator overloading, `CelestialSightLine`, `FocalPlane`,
  `Assembly`, observatory sites, and weather models.

## Precision

Validated against the original C/Fortran implementations to sub-milliarcsecond
accuracy:

| Layer                        | Agreement              |
|------------------------------|------------------------|
| ERFA functions               | Matches pyerfa         |
| Quaternion functions         | Bit-identical to so3g  |
| `naive_az_el`                | ~1e-12 (quat diff)     |
| `az_el` (all weather/sites)  | 0.0004--0.0005 arcsec  |
| Detector projection          | ~1e-12                 |

## Running tests

```bash
# All tests
python -m pytest so_pointjax/ -v

# By submodule
python -m pytest so_pointjax/erfa/tests/ -v
python -m pytest so_pointjax/qpoint/tests/ -v
python -m pytest so_pointjax/proj/tests/ -v

# Cross-validation against so3g (requires so3g)
python -m pytest so_pointjax/proj/tests/test_cross_validation.py -v -s
```

## Performance

Key results (CPU, JIT-compiled):

- **Quaternion ops**: 1.3--7x faster than so3g at N >= 100K
- **Pointing pipeline**: 3--8x faster across all sizes
- **Bore x det composition**: 2--3x faster for realistic focal planes
- **Gradients**: ~1 us/sample (unique capability vs so3g)

```bash
python -m so_pointjax.proj.benchmarks.bench_so3g [--quick]
python -m so_pointjax.proj.benchmarks.bench_quat_array [--quick]
python -m so_pointjax.qpoint.benchmarks.bench_pointing [--quick]
```

## License

BSD-3-Clause
