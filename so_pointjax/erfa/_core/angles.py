"""Angle normalization and conversion functions.

Ported from ERFA C library. String formatting functions (a2af, a2tf, d2tf)
are omitted as they are non-differentiable display-only operations.
"""

import jax.numpy as jnp
from so_pointjax.erfa._core.constants import D2PI, DPI


def anp(a):
    """Normalize angle into the range 0 <= a < 2pi."""
    w = jnp.fmod(a, D2PI)
    return jnp.where(w < 0.0, w + D2PI, w)


def anpm(a):
    """Normalize angle into the range -pi <= a < +pi."""
    w = jnp.fmod(a, D2PI)
    return jnp.where(jnp.abs(w) >= DPI,
                     w - jnp.sign(a) * D2PI,
                     w)


__all__ = ["anp", "anpm"]
