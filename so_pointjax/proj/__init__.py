"""so_pointjax.proj: JAX reimplementation of so3g.proj for differentiable pointing.

Drop-in replacement for so3g.proj with JAX backend via so_pointjax.qpoint.
All functions are compatible with jax.jit, jax.grad, and jax.vmap.
"""

import jax
jax.config.update("jax_enable_x64", True)

from . import quat
from .quat import Quat
from . import util
from .weather import Weather, weather_factory
from .coords import (
    CelestialSightLine, EarthlySite, Assembly, FocalPlane,
    SITES, DEFAULT_SITE,
)

import jax.numpy as jnp
DEG = jnp.pi / 180.0
