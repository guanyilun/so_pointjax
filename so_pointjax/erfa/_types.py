"""JAX-compatible data types for ERFA structs."""

from typing import NamedTuple
import jax
import jax.numpy as jnp


class ASTROM(NamedTuple):
    """Star-independent astrometry parameters.

    All vectors are with respect to BCRS axes.
    """
    pmt: jax.Array    # PM time interval (SSB, Julian years)
    eb: jax.Array     # SSB to observer (vector, au) shape (3,)
    eh: jax.Array     # Sun to observer (unit vector) shape (3,)
    em: jax.Array     # distance from Sun to observer (au)
    v: jax.Array      # barycentric observer velocity (vector, c) shape (3,)
    bm1: jax.Array    # sqrt(1-|v|^2): reciprocal of Lorentz factor
    bpn: jax.Array    # bias-precession-nutation matrix shape (3,3)
    along: jax.Array  # longitude + s' + dERA(DUT) (radians)
    phi: jax.Array    # geodetic latitude (radians)
    xpl: jax.Array    # polar motion xp wrt local meridian (radians)
    ypl: jax.Array    # polar motion yp wrt local meridian (radians)
    sphi: jax.Array   # sine of geodetic latitude
    cphi: jax.Array   # cosine of geodetic latitude
    diurab: jax.Array # magnitude of diurnal aberration vector
    eral: jax.Array   # "local" Earth rotation angle (radians)
    refa: jax.Array   # refraction constant A (radians)
    refb: jax.Array   # refraction constant B (radians)

    @staticmethod
    def empty():
        """Create a zero-initialized ASTROM."""
        return ASTROM(
            pmt=jnp.float64(0.0),
            eb=jnp.zeros(3),
            eh=jnp.zeros(3),
            em=jnp.float64(0.0),
            v=jnp.zeros(3),
            bm1=jnp.float64(0.0),
            bpn=jnp.zeros((3, 3)),
            along=jnp.float64(0.0),
            phi=jnp.float64(0.0),
            xpl=jnp.float64(0.0),
            ypl=jnp.float64(0.0),
            sphi=jnp.float64(0.0),
            cphi=jnp.float64(0.0),
            diurab=jnp.float64(0.0),
            eral=jnp.float64(0.0),
            refa=jnp.float64(0.0),
            refb=jnp.float64(0.0),
        )


class LDBODY(NamedTuple):
    """Body parameters for light deflection."""
    bm: jax.Array     # mass of the body (solar masses)
    dl: jax.Array     # deflection limiter (radians^2/2)
    pv: jax.Array     # barycentric PV of the body (au, au/day) shape (2,3)
