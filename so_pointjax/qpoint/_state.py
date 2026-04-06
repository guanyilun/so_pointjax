"""State management and high-level API for QPoint-JAX.

QPointState: immutable parameter container (JAX pytree-compatible).
QPoint: ergonomic stateful wrapper around the functional pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from so_pointjax.qpoint._quaternion import (
    identity, quat2radecpa, quat2radec,
)
from so_pointjax.qpoint._corrections import det_offset_quat, hwp_quat
from so_pointjax.qpoint._pointing import (
    azelpsi2bore, azel2bore, azelpsi2bore_jit, radec2azel_jit,
    bore2radecpa, bore2radec, azel2radecpa, radec2azel,
    precompute_times,
)
from so_pointjax.qpoint._time_utils import ctime2gmst


# ---------------------------------------------------------------------------
# Immutable state container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QPointState:
    """Immutable state for the QPoint pointing pipeline.

    Registered as a JAX pytree so it can be passed through jit/grad/vmap
    (though IERS data and weather are typically static).

    Parameters
    ----------
    accuracy : int
        0 = full (xys06a), 1 = low (xys00b). Default 1.
    mean_aber : bool
        Apply annual aberration at boresight level. Default True.
    fast_aber : bool
        Use small-angle aberration approximation. Default False.
    polconv : int
        Polarization convention: 0 = COSMO, 1 = IAU. Default 0.
    dut1 : float
        UT1-UTC in seconds. Default 0.0.
    weather : dict or None
        Weather parameters for refraction: {temperature, pressure, humidity, frequency}.
    xp : float
        IERS pole x coordinate (arcseconds). Default 0.0.
    yp : float
        IERS pole y coordinate (arcseconds). Default 0.0.
    ref_delta : float
        Additional refraction correction (degrees). Default 0.0.
    iers_data : dict or None
        IERS Bulletin A data: {mjd, dut1, x, y} arrays.
    """
    accuracy: int = 1
    mean_aber: bool = True
    fast_aber: bool = False
    polconv: int = 0
    dut1: float = 0.0
    weather: Optional[dict] = None
    xp: float = 0.0
    yp: float = 0.0
    ref_delta: float = 0.0
    iers_data: Optional[dict] = None

    def replace(self, **kwargs):
        """Return a new QPointState with updated fields."""
        current = {f.name: getattr(self, f.name)
                   for f in self.__dataclass_fields__.values()}
        current.update(kwargs)
        return QPointState(**current)

    def get_iers(self, ctime):
        """Look up IERS parameters for a given Unix time.

        Returns (dut1, xp, yp). Falls back to instance defaults if no
        IERS data is loaded.
        """
        if self.iers_data is None:
            return self.dut1, self.xp, self.yp

        from so_pointjax.qpoint._iers import interpolate_bulletin_a
        mjd = ctime / 86400.0 + 40587.0  # Unix time to MJD
        dut1, xp, yp = interpolate_bulletin_a(self.iers_data, mjd)
        return dut1, xp, yp


# ---------------------------------------------------------------------------
# High-level ergonomic wrapper
# ---------------------------------------------------------------------------

class QPoint:
    """High-level interface for the QPoint pointing pipeline.

    Wraps the functional pipeline with a stateful interface matching
    the original QPoint Python API.

    Examples
    --------
    >>> Q = QPoint(accuracy=1, mean_aber=True)
    >>> q_bore = Q.azel2bore(180.0, 45.0, 0.0, 0.0, -44.65, -89.99, 1700000000.0)
    >>> ra, dec, pa = Q.bore2radecpa(identity(), 1700000000.0, q_bore)

    >>> # With weather for refraction
    >>> Q.set(weather=dict(temperature=0.0, pressure=550.0, humidity=0.2, frequency=150e9))
    >>> q_bore = Q.azel2bore(180.0, 45.0, 0.0, 0.0, -44.65, -89.99, 1700000000.0)

    >>> # With IERS Bulletin A
    >>> Q.load_bulletin_a('finals2000A.data')
    >>> q_bore = Q.azel2bore(180.0, 45.0, 0.0, 0.0, -44.65, -89.99, 1700000000.0)
    """

    def __init__(self, **kwargs):
        """Initialize QPoint with given parameters.

        Parameters
        ----------
        accuracy : int or str
            0/'full'/'high' or 1/'low'. Default 1.
        mean_aber : bool
            Apply annual aberration at boresight. Default True.
        fast_aber : bool
            Small-angle aberration. Default False.
        polconv : int or str
            0/'cosmo' or 1/'iau'. Default 0.
        dut1 : float
            UT1-UTC in seconds. Default 0.0.
        weather : dict
            {temperature, pressure, humidity, frequency}.
        xp, yp : float
            IERS pole coordinates (arcseconds).
        ref_delta : float
            Additional refraction correction (degrees).
        update_iers : bool
            If True, fetch IERS data via astropy on init. Default False.
        """
        update_iers = kwargs.pop('update_iers', False)

        # Normalize convenience string values
        if 'accuracy' in kwargs:
            kwargs['accuracy'] = _normalize_accuracy(kwargs['accuracy'])
        if 'polconv' in kwargs:
            kwargs['polconv'] = _normalize_polconv(kwargs['polconv'])

        self._state = QPointState(**kwargs)

        if update_iers:
            self.update_bulletin_a()

    @property
    def state(self):
        """Current QPointState."""
        return self._state

    def set(self, **kwargs):
        """Update parameters. Returns self for chaining."""
        if 'accuracy' in kwargs:
            kwargs['accuracy'] = _normalize_accuracy(kwargs['accuracy'])
        if 'polconv' in kwargs:
            kwargs['polconv'] = _normalize_polconv(kwargs['polconv'])
        self._state = self._state.replace(**kwargs)
        return self

    def get(self, *args):
        """Get parameter values.

        With no args, returns dict of all parameters.
        With args, returns values for the named parameters.
        """
        if not args:
            return {f.name: getattr(self._state, f.name)
                    for f in self._state.__dataclass_fields__.values()
                    if f.name != 'iers_data'}
        if len(args) == 1:
            return getattr(self._state, args[0])
        return tuple(getattr(self._state, a) for a in args)

    # -----------------------------------------------------------------------
    # IERS Bulletin A
    # -----------------------------------------------------------------------

    def load_bulletin_a(self, filename, columns=None):
        """Load IERS Bulletin A from a text file."""
        from so_pointjax.qpoint._iers import load_bulletin_a
        data = load_bulletin_a(filename, columns=columns)
        self._state = self._state.replace(iers_data=data)
        return self

    def update_bulletin_a(self, start_year=2000):
        """Fetch IERS Bulletin A via astropy."""
        from so_pointjax.qpoint._iers import update_bulletin_a
        data = update_bulletin_a(start_year=start_year)
        self._state = self._state.replace(iers_data=data)
        return self

    def get_bulletin_a(self, mjd):
        """Query interpolated dut1, xp, yp at given MJD(s)."""
        if self._state.iers_data is None:
            raise ValueError("No IERS Bulletin A data loaded. "
                             "Call load_bulletin_a() or update_bulletin_a() first.")
        from so_pointjax.qpoint._iers import interpolate_bulletin_a
        return interpolate_bulletin_a(self._state.iers_data, mjd)

    def _get_iers_params(self, ctime):
        """Get dut1, xp, yp for a given ctime, using IERS data if available."""
        return self._state.get_iers(ctime)

    # -----------------------------------------------------------------------
    # Forward pipeline
    # -----------------------------------------------------------------------

    def azel2bore(self, az, el, pitch, roll, lon, lat, ctime, **kwargs):
        """Convert az/el to boresight quaternion.

        Parameters
        ----------
        az, el : float or array
            Azimuth and elevation (degrees).
        pitch, roll : float
            Boresight pitch and roll (degrees).
        lon, lat : float
            Observer longitude and latitude (degrees).
        ctime : float or array
            Unix time (seconds).

        Returns
        -------
        q_bore : array shape (4,) or (n, 4)
        """
        s = self._state
        dut1, xp, yp = self._get_iers_params(ctime)
        return azelpsi2bore(
            az, el, 0.0, lon, lat, ctime,
            pitch=pitch, roll=roll, dut1=dut1,
            weather=s.weather, accuracy=s.accuracy,
            mean_aber=s.mean_aber, fast_aber=s.fast_aber,
            xp=xp, yp=yp,
        )

    def azelpsi2bore(self, az, el, psi, lon, lat, ctime,
                     pitch=0.0, roll=0.0, **kwargs):
        """Convert az/el/psi to boresight quaternion."""
        s = self._state
        dut1, xp, yp = self._get_iers_params(ctime)
        return azelpsi2bore(
            az, el, psi, lon, lat, ctime,
            pitch=pitch, roll=roll, dut1=dut1,
            weather=s.weather, accuracy=s.accuracy,
            mean_aber=s.mean_aber, fast_aber=s.fast_aber,
            xp=xp, yp=yp,
        )

    def bore2radecpa(self, q_off, ctime, q_bore, **kwargs):
        """Convert boresight + offset → (ra, dec, pa).

        Parameters
        ----------
        q_off : array shape (4,)
            Detector offset quaternion.
        ctime : float
            Unix time (seconds).
        q_bore : array shape (4,)
            Boresight quaternion.

        Returns
        -------
        ra, dec, pa : floats in degrees
        """
        s = self._state
        return bore2radecpa(
            q_off, ctime, q_bore,
            mean_aber=s.mean_aber, fast_aber=s.fast_aber,
            accuracy=s.accuracy,
        )

    def bore2radec(self, q_off, ctime, q_bore, **kwargs):
        """Convert boresight + offset → (ra, dec, sin2psi, cos2psi)."""
        s = self._state
        return bore2radec(
            q_off, ctime, q_bore,
            mean_aber=s.mean_aber, fast_aber=s.fast_aber,
            accuracy=s.accuracy,
        )

    def azel2radecpa(self, delta_az, delta_el, delta_psi,
                     az, el, lon, lat, ctime,
                     psi=0.0, pitch=0.0, roll=0.0, **kwargs):
        """Complete az/el + detector offset → (ra, dec, pa)."""
        s = self._state
        dut1, xp, yp = self._get_iers_params(ctime)
        return azel2radecpa(
            delta_az, delta_el, delta_psi,
            az, el, lon, lat, ctime,
            psi=psi, pitch=pitch, roll=roll,
            dut1=dut1, weather=s.weather, accuracy=s.accuracy,
            fast_aber=s.fast_aber, xp=xp, yp=yp,
        )

    # -----------------------------------------------------------------------
    # Inverse pipeline
    # -----------------------------------------------------------------------

    def radec2azel(self, ra, dec, pa, lon, lat, ctime, **kwargs):
        """Inverse pipeline: (ra, dec, pa) → (az, el, pa_out)."""
        s = self._state
        dut1, xp, yp = self._get_iers_params(ctime)
        return radec2azel(
            ra, dec, pa, lon, lat, ctime,
            dut1=dut1, weather=s.weather, accuracy=s.accuracy,
            mean_aber=s.mean_aber, fast_aber=s.fast_aber,
            xp=xp, yp=yp,
        )

    # -----------------------------------------------------------------------
    # Vectorized pipeline (precompute + vmap)
    # -----------------------------------------------------------------------

    def precompute_times(self, ctime):
        """Precompute Julian Dates for vectorized use.

        Call this, then pass results to azelpsi2bore_jit via vmap.
        """
        dut1, _, _ = self._get_iers_params(ctime)
        return precompute_times(ctime, dut1)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def det_offset(self, delta_az, delta_el, delta_psi):
        """Create detector offset quaternion."""
        return det_offset_quat(delta_az, delta_el, delta_psi)

    def hwp_quat(self, theta):
        """Create half-wave plate rotation quaternion."""
        return hwp_quat(theta)

    def gmst(self, ctime, **kwargs):
        """Greenwich Mean Sidereal Time (degrees)."""
        return ctime2gmst(ctime, accuracy=self._state.accuracy)

    def lmst(self, ctime, lon, **kwargs):
        """Local Mean Sidereal Time (degrees)."""
        gmst_deg = ctime2gmst(ctime, accuracy=self._state.accuracy)
        return (gmst_deg + lon) % 360.0

    def bore_offset(self, q_bore, ang1, ang2, ang3, post=False):
        """Apply angular offset to boresight quaternion.

        Parameters
        ----------
        q_bore : array shape (4,)
            Boresight quaternion.
        ang1, ang2, ang3 : float
            Offset angles (degrees) — interpreted as (delta_az, delta_el, delta_psi).
        post : bool
            If True, right-multiply (post-apply). If False, left-multiply.

        Returns
        -------
        q : array shape (4,)
        """
        from so_pointjax.qpoint._quaternion import mul
        q_off = det_offset_quat(ang1, ang2, ang3)
        if post:
            return mul(q_bore, q_off)
        return mul(q_off, q_bore)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_accuracy(val):
    """Convert accuracy string/int to int."""
    if isinstance(val, str):
        if val in ('full', 'high', '0'):
            return 0
        if val in ('low', '1'):
            return 1
        raise ValueError(f"Unknown accuracy: {val!r}")
    return int(val)


def _normalize_polconv(val):
    """Convert polconv string/int to int."""
    if isinstance(val, str):
        if val.lower() in ('cosmo', '0'):
            return 0
        if val.lower() in ('iau', '1'):
            return 1
        raise ValueError(f"Unknown polconv: {val!r}")
    return int(val)
