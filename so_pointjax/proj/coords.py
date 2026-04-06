"""Coordinate classes mirroring so3g.proj.coords, backed by so_pointjax.qpoint.

Provides EarthlySite, CelestialSightLine, FocalPlane, and Assembly.
All pointing math uses JAX and is differentiable via jax.grad.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import quat
from .weather import Weather, weather_factory

DEG = jnp.pi / 180.0

# ERA polynomial for naive pointing (matches so3g)
ERA_EPOCH = 946684800 + 3600 * 12  # Noon on Jan 1 2000
ERA_POLY = jnp.array([6.300387486754831, 4.894961212823756])


# ---------------------------------------------------------------------------
# Site definitions
# ---------------------------------------------------------------------------

class EarthlySite:
    """Observatory location on Earth.

    Parameters
    ----------
    lon, lat : float
        Longitude (degrees E), latitude (degrees N).
    elev : float
        Elevation in meters above sea level.
    typical_weather : Weather or None
        Default weather for this site.
    """

    def __init__(self, lon, lat, elev, typical_weather=None):
        self.lon = lon
        self.lat = lat
        self.elev = elev
        self.typical_weather = typical_weather


def _debabyl(deg, arcmin, arcsec):
    return deg + arcmin / 60 + arcsec / 3600


SITES = {
    'act': EarthlySite(-67.7876, -22.9585, 5188.,
                       typical_weather=weather_factory('toco')),
    'so_lat': EarthlySite(-_debabyl(67, 47, 15.68), -_debabyl(22, 57, 39.47), 5188.,
                          typical_weather=weather_factory('toco')),
    'so_sat1': EarthlySite(-_debabyl(67, 47, 18.11), -_debabyl(22, 57, 36.38), 5188.,
                           typical_weather=weather_factory('toco')),
    'so_sat2': EarthlySite(-_debabyl(67, 47, 17.28), -_debabyl(22, 57, 36.35), 5188.,
                           typical_weather=weather_factory('toco')),
    'so_sat3': EarthlySite(-_debabyl(67, 47, 16.53), -_debabyl(22, 57, 35.97), 5188.,
                           typical_weather=weather_factory('toco')),
}
SITES['so'] = SITES['so_lat']
SITES['_default'] = SITES['so']
DEFAULT_SITE = 'so'


# ---------------------------------------------------------------------------
# JIT-compiled computation kernels
# ---------------------------------------------------------------------------

@jax.jit
def _naive_az_el_compute(t, az, el, roll, site_lon_rad, site_lat_rad):
    """Fused naive pointing: all quaternion ops compiled into one kernel."""
    J = (t - ERA_EPOCH) / 86400.0
    era = jnp.polyval(ERA_POLY, J)
    lst = era + site_lon_rad
    return quat.qmul(
        quat.qmul(
            quat.qmul(
                quat.euler(2, lst),
                quat.euler(1, jnp.pi / 2 - site_lat_rad)),
            quat.qmul(
                quat.euler(2, jnp.pi + jnp.zeros_like(t)),
                quat.euler(2, -az))),
        quat.qmul(
            quat.euler(1, jnp.pi / 2 - el),
            quat.euler(2, roll + jnp.zeros_like(t))))


@jax.jit
def _for_lonlat_compute(lon, lat, psi):
    """Fused lonlat→quaternion: all ops compiled into one kernel."""
    return quat.qmul(
        quat.qmul(
            quat.euler(2, lon),
            quat.euler(1, jnp.pi / 2 - lat)),
        quat.euler(2, psi + jnp.zeros_like(lon)))


@jax.jit
def _for_horizon_compute(az, el, roll):
    """Fused horizon→quaternion: all ops compiled into one kernel."""
    return quat.qmul(
        quat.qmul(
            quat.euler(2, -az),
            quat.euler(1, jnp.pi / 2 - el)),
        quat.euler(2, roll))


@jax.jit
def _compute_coords_jit(Q, det_quats):
    """Fused detector coordinate computation."""
    def _det_coords(q_det):
        q_total = quat.qmul(Q, q_det[None, :])
        return _quats_to_lonlat(q_total)
    return jax.vmap(_det_coords)(det_quats)


# ---------------------------------------------------------------------------
# CelestialSightLine
# ---------------------------------------------------------------------------

class CelestialSightLine:
    """Carries a vector of celestial pointing quaternions.

    The pointing is stored in self.Q as a jnp.ndarray of shape (N, 4) or (4,).
    All constructors produce JAX arrays compatible with jit/grad/vmap.
    """

    @staticmethod
    def decode_site(site=None):
        """Convert site argument to an EarthlySite.

        Parameters
        ----------
        site : EarthlySite, str, or None
            If None, uses the default site.
        """
        if site is None:
            site = DEFAULT_SITE
        if isinstance(site, EarthlySite):
            return site
        if site in SITES:
            return SITES[site]
        raise ValueError(f"Could not decode {site!r} as a Site.")

    @classmethod
    def naive_az_el(cls, t, az, el, roll=0., site=None, weather=None):
        """Construct from horizon coordinates using a simple ERA-based model.

        This is fast but only accurate to ~arcminutes (no nutation,
        precession, aberration, or refraction).

        Parameters
        ----------
        t : float or array
            Unix timestamps.
        az, el : float or array
            Azimuth and elevation in radians.
        roll : float or array
            Boresight roll in radians.
        site : EarthlySite, str, or None
        weather : ignored

        Returns
        -------
        CelestialSightLine
        """
        site = cls.decode_site(site)

        t = jnp.asarray(t, dtype=jnp.float64)
        az = jnp.asarray(az, dtype=jnp.float64)
        el = jnp.asarray(el, dtype=jnp.float64)

        self = cls()
        self.Q = _naive_az_el_compute(
            t, az, el, roll, site.lon * DEG, site.lat * DEG)
        return self

    @classmethod
    def az_el(cls, t, az, el, roll=None, site=None, weather=None, **kwargs):
        """Construct from horizon coordinates using high-precision pointing.

        Uses so_pointjax.qpoint for full corrections (nutation, precession, frame bias,
        aberration, polar motion, refraction).

        Parameters
        ----------
        t : float or array
            Unix timestamps.
        az, el : float or array
            Azimuth and elevation in radians.
        roll : float or array or None
            Boresight roll in radians.
        site : EarthlySite, str, or None
        weather : Weather, str, or None
            Weather data for refraction. Use 'typical', 'vacuum', 'toco', etc.
        **kwargs
            Additional keyword arguments passed to QPoint (accuracy, mean_aber, etc.).

        Returns
        -------
        CelestialSightLine
        """
        from so_pointjax.qpoint._state import QPoint as QP

        site = cls.decode_site(site)

        if isinstance(weather, str):
            if weather == 'typical':
                weather = site.typical_weather
            else:
                weather = weather_factory(weather)

        if weather is None:
            raise ValueError(
                "High-precision pointing requires weather data. "
                "Try 'toco', 'typical', or 'vacuum'."
            )

        from so_pointjax.qpoint._pointing import (
            precompute_corrections, azelpsi2bore_fast,
        )

        # Parse kwargs
        accuracy = kwargs.get('accuracy', 1)
        mean_aber = kwargs.get('mean_aber', True)

        t = jnp.asarray(t, dtype=jnp.float64)
        az = jnp.asarray(az, dtype=jnp.float64)
        el = jnp.asarray(el, dtype=jnp.float64)

        # so_pointjax.qpoint expects degrees
        az_deg = jnp.rad2deg(az)
        el_deg = jnp.rad2deg(el)

        # Use the fast vectorized path: precompute slow corrections, vmap the rest
        import numpy as _np
        corr = precompute_corrections(
            _np.asarray(t), accuracy=accuracy, mean_aber=mean_aber,
        )
        q_npb_per = corr['q_npb'][corr['npb_idx']]
        q_wobble_per = corr['q_wobble'][corr['npb_idx']]
        beta_per = corr['beta_earth'][corr['aber_idx']]

        # Compute refraction coefficients if weather provided
        weather_A, weather_B = 0.0, 0.0
        if weather is not None:
            import so_pointjax.erfa
            # c / f in micrometers: c(m/s) / f(GHz*1e9) * 1e6(m→µm)
            freq_ghz = weather.get('frequency', 150.0)  # GHz, matching qpoint default
            wavelength = 299792458.0 / (freq_ghz * 1e9) * 1e6  # micrometers
            weather_A, weather_B = so_pointjax.erfa.refco(
                weather.get('pressure', 0.0),
                weather.get('temperature', 0.0),
                weather.get('humidity', 0.0),
                wavelength,
            )

        @jax.jit
        def _forward(az_arr, el_arr, tt1, tt2, ut1_1, ut1_2, npb, wob, beta):
            def _single(az_i, el_i, t1, t2, u1, u2, n, w, b):
                return azelpsi2bore_fast(
                    az_i, el_i, 0.0, site.lon, site.lat,
                    t1, t2, u1, u2, n, w, b,
                    weather_A=weather_A, weather_B=weather_B,
                )
            return jax.vmap(_single)(az_arr, el_arr, tt1, tt2, ut1_1, ut1_2, npb, wob, beta)

        q_bore = _forward(
            az_deg, el_deg,
            corr['tt1'], corr['tt2'],
            corr['ut1_1'], corr['ut1_2'],
            q_npb_per, q_wobble_per, beta_per,
        )

        self = cls()

        # Apply boresight roll + pi rotation (matches so3g convention)
        if roll is None:
            self.Q = quat.qmul(q_bore, quat.euler(2, jnp.pi + jnp.zeros_like(t)))
        else:
            roll = jnp.asarray(roll, dtype=jnp.float64)
            self.Q = quat.qmul(q_bore, quat.euler(2, jnp.pi + roll))

        return self

    @classmethod
    def for_lonlat(cls, lon, lat, psi=0.):
        """Construct from celestial coordinates directly.

        Parameters
        ----------
        lon, lat : float or array
            RA and dec in radians.
        psi : float or array
            Parallactic rotation in radians. psi=0 means focal plane "up"
            is parallel to lines of longitude.

        Returns
        -------
        CelestialSightLine
        """
        self = cls()
        self.Q = _for_lonlat_compute(
            jnp.asarray(lon, dtype=jnp.float64),
            jnp.asarray(lat, dtype=jnp.float64),
            jnp.asarray(psi, dtype=jnp.float64))
        return self

    @classmethod
    def for_horizon(cls, t, az, el, roll=None, site=None, weather=None):
        """Construct trivial SightLine where celestial = horizon coordinates.

        Parameters
        ----------
        t : float or array
            Timestamps (ignored for pointing, kept for API compat).
        az, el : float or array
            Azimuth and elevation in radians.
        roll : float or array or None
            Boresight roll in radians.

        Returns
        -------
        CelestialSightLine
        """
        self = cls()
        az = jnp.asarray(az, dtype=jnp.float64)
        el = jnp.asarray(el, dtype=jnp.float64)

        if roll is None:
            roll_val = jnp.zeros_like(az)
        else:
            roll_val = jnp.asarray(roll, dtype=jnp.float64) + jnp.zeros_like(az)

        self.Q = _for_horizon_compute(az, el, roll_val)
        return self

    def coords(self, fplane=None, output=None):
        """Get celestial coordinates for each detector at each time.

        Parameters
        ----------
        fplane : FocalPlane or None
            Detector offsets. If None, returns boresight pointing.
        output : ignored
            Present for API compatibility.

        Returns
        -------
        coords : array shape (N, 4) or (n_det, N, 4)
            Columns are [lon, lat, cos2psi, sin2psi] in radians.
        """
        collapse = (fplane is None)
        if collapse:
            fplane = FocalPlane.boresight()
        result = self._compute_coords(fplane)
        if collapse:
            result = result[0]  # collapse single-detector dimension
        return result

    def _compute_coords(self, fplane):
        """Compute coordinates for all detectors."""
        Q = self.Q
        was_scalar = (Q.ndim == 1)
        Q = jnp.atleast_2d(Q)        # (N, 4)

        result = _compute_coords_jit(Q, fplane.quats)  # (n_det, N, 4)
        if was_scalar:
            result = result[:, 0, :]  # (n_det, 4) — squeeze time dimension
        return result


def _quats_to_lonlat(q):
    """Convert pointing quaternions to (lon, lat, cos2psi, sin2psi).

    Parameters
    ----------
    q : array shape (N, 4)

    Returns
    -------
    coords : array shape (N, 4)
        Columns: [lon, lat, cos2psi, sin2psi] in radians.
    """
    theta, phi, psi = quat.decompose_iso(q)
    lon = phi
    lat = jnp.pi / 2 - theta
    cos2psi = jnp.cos(2 * psi)
    sin2psi = jnp.sin(2 * psi)
    return jnp.stack([lon, lat, cos2psi, sin2psi], axis=-1)


# ---------------------------------------------------------------------------
# FocalPlane
# ---------------------------------------------------------------------------

class FocalPlane:
    """Detector positions and responses in the focal plane.

    Attributes
    ----------
    quats : array shape (n_det, 4)
        Detector pointing quaternions.
    resps : array shape (n_det, 2)
        Total intensity and polarization responsivity per detector.
    """

    def __init__(self, quats=None, resps=None):
        """Construct a FocalPlane.

        Parameters
        ----------
        quats : array-like shape (n_det, 4) or None
            Detector quaternions. None gives empty FocalPlane.
        resps : array-like shape (n_det, 2) or None
            [T_resp, P_resp] per detector. None defaults to all ones.
        """
        if quats is None:
            quats = jnp.zeros((0, 4))
        self.quats = jnp.asarray(quats, dtype=jnp.float64)
        if self.quats.ndim == 1:
            self.quats = self.quats[None, :]

        if resps is None:
            self.resps = jnp.ones((len(self.quats), 2))
        else:
            self.resps = jnp.asarray(resps, dtype=jnp.float64)

    @classmethod
    def boresight(cls):
        """FocalPlane with a single detector at the boresight."""
        return cls(quats=jnp.array([[1., 0., 0., 0.]]))

    @classmethod
    def from_xieta(cls, xi, eta, gamma=0., T=1., P=1., Q=1., U=0., hwp=False):
        """Construct from focal plane tangent coordinates.

        When looking at the sky along the boresight, xi is parallel to
        increasing azimuth and eta is parallel to increasing elevation.
        gamma is the polarization angle from the eta axis toward xi.

        Parameters
        ----------
        xi, eta : float or array
            Tangent plane positions in radians.
        gamma : float or array
            Polarization angle in radians.
        T, P : float or array
            Total intensity and polarization responsivity.
        Q, U : float or array
            Q- and U-polarization responsivity (alternative to P, gamma).
        hwp : bool
            If True, flip gamma sign for HWP convention.

        Returns
        -------
        FocalPlane
        """
        xi = jnp.atleast_1d(jnp.asarray(xi, dtype=jnp.float64))
        eta = jnp.atleast_1d(jnp.asarray(eta, dtype=jnp.float64))
        gamma = jnp.asarray(gamma, dtype=jnp.float64)
        T = jnp.asarray(T, dtype=jnp.float64)
        P = jnp.asarray(P, dtype=jnp.float64)
        Q = jnp.asarray(Q, dtype=jnp.float64)
        U = jnp.asarray(U, dtype=jnp.float64)

        # Combine gamma with Q/U specification
        gamma_total = gamma + jnp.arctan2(U, Q) / 2
        P_total = P * jnp.sqrt(Q**2 + U**2)

        if hwp:
            gamma_total = -gamma_total

        # Broadcast to 1d
        xi, eta, gamma_total, T, P_total = jnp.broadcast_arrays(
            xi, eta, gamma_total, T, P_total)

        quats = quat.rotation_xieta(xi, eta, gamma_total)
        resps = jnp.stack([T, P_total], axis=-1)
        return cls(quats=quats, resps=resps)

    def coeffs(self):
        """Return (n_det, 4) array of quaternion coefficients."""
        return self.quats

    @property
    def ndet(self):
        return len(self.quats)

    def __len__(self):
        return self.ndet

    def __getitem__(self, sel):
        """Slice the FocalPlane to get a subset of detectors."""
        return FocalPlane(quats=self.quats[sel], resps=self.resps[sel])


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

class Assembly:
    """Groups boresight pointing and detector offsets.

    Attributes
    ----------
    Q : array shape (N, 4)
        Boresight quaternions.
    fplane : FocalPlane
        Detector offsets and responses.
    collapse : bool
        If True, indicates a boresight-only assembly.
    """

    def __init__(self, collapse=False):
        self.collapse = collapse

    @classmethod
    def attach(cls, sight_line, fplane):
        """Create Assembly from a CelestialSightLine and FocalPlane.

        Parameters
        ----------
        sight_line : CelestialSightLine or array shape (N, 4)
            Boresight pointing.
        fplane : FocalPlane
            Detector offsets.
        """
        self = cls()
        if isinstance(sight_line, CelestialSightLine):
            self.Q = sight_line.Q
        else:
            self.Q = jnp.asarray(sight_line, dtype=jnp.float64)
        self.fplane = fplane
        return self

    @classmethod
    def for_boresight(cls, sight_line):
        """Assembly with a single dummy detector at the boresight."""
        self = cls(collapse=True)
        if isinstance(sight_line, CelestialSightLine):
            self.Q = sight_line.Q
        else:
            self.Q = jnp.asarray(sight_line, dtype=jnp.float64)
        self.fplane = FocalPlane.boresight()
        return self
