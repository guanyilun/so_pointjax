"""so_pointjax: Differentiable telescope pointing in JAX.

Combines three layers:
  - so_pointjax.erfa    — differentiable ERFA routines
  - so_pointjax.qpoint  — pointing pipeline, quaternion ops, HEALPix
  - so_pointjax.proj    — high-level API (Quat, CelestialSightLine, FocalPlane, ...)
"""

import jax
jax.config.update("jax_enable_x64", True)

from so_pointjax import erfa    # noqa: F401
from so_pointjax import qpoint  # noqa: F401
from so_pointjax import proj    # noqa: F401
