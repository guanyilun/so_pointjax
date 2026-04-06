"""Differentiable ERFA (Essential Routines for Fundamental Astronomy) in JAX."""

import jax
jax.config.update("jax_enable_x64", True)

from so_pointjax.erfa._core.constants import *  # noqa: F401,F403
from so_pointjax.erfa._core.vector import *  # noqa: F401,F403
from so_pointjax.erfa._core.angles import *  # noqa: F401,F403
from so_pointjax.erfa._core.calendar import *  # noqa: F401,F403
from so_pointjax.erfa._core.time import *  # noqa: F401,F403
from so_pointjax.erfa._leapsec import *  # noqa: F401,F403
from so_pointjax.erfa._core.precnut import *  # noqa: F401,F403
from so_pointjax.erfa._core.geodetic import *  # noqa: F401,F403
from so_pointjax.erfa._core.ephem import *  # noqa: F401,F403
from so_pointjax.erfa._core.astrometry import *  # noqa: F401,F403
from so_pointjax.erfa._core.frames import *  # noqa: F401,F403
from so_pointjax.erfa._core.gnomonic import *  # noqa: F401,F403
from so_pointjax.erfa._types import ASTROM, LDBODY  # noqa: F401
