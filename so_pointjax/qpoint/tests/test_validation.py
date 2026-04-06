"""Phase 7: Validation of so_pointjax.qpoint against the C QPoint library.

Compares the full pointing pipeline output between the JAX and C implementations
for representative scans at different sites and configurations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

try:
    from qpoint.qpoint_class import QPoint as CQPoint
    HAS_CQPOINT = True
except ImportError:
    HAS_CQPOINT = False

from so_pointjax.qpoint._state import QPoint as JaxQPoint
from so_pointjax.qpoint._quaternion import identity, quat2radecpa, quat2radec
from so_pointjax.qpoint._corrections import det_offset_quat
from so_pointjax.qpoint._pointing import (
    azelpsi2bore, azel2bore, bore2radecpa, bore2radec,
    azel2radecpa, precompute_times, azelpsi2bore_jit,
)
from so_pointjax.qpoint._pixel import ang2pix_nest, ang2pix_ring


# Skip all tests if C QPoint not available
pytestmark = pytest.mark.skipif(not HAS_CQPOINT, reason="C QPoint not installed")


# ---------------------------------------------------------------------------
# Test sites
# ---------------------------------------------------------------------------

SITES = {
    'south_pole': {'lon': -44.65, 'lat': -89.99},
    'atacama': {'lon': -67.79, 'lat': -22.96},
    'mid_lat': {'lon': -118.0, 'lat': 34.0},
}

CTIMES = [
    1700000000.0,  # 2023-11-14
    1600000000.0,  # 2020-09-13
    1800000000.0,  # 2027-01-15
]


# ---------------------------------------------------------------------------
# Single-sample boresight comparison
# ---------------------------------------------------------------------------

class TestBoresightQuaternion:
    """Compare boresight quaternion between JAX and C."""

    @pytest.mark.parametrize('site_name,site', SITES.items())
    @pytest.mark.parametrize('ctime', CTIMES)
    def test_azel2bore(self, site_name, site, ctime):
        """Boresight quaternion should match for various az/el/sites."""
        az, el = 180.0, 45.0
        lon, lat = site['lon'], site['lat']

        # C QPoint
        cQ = CQPoint()
        q_c = cQ.azel2bore(az, el, 0.0, 0.0, lon, lat, ctime).flatten()

        # JAX
        q_j = np.array(azel2bore(az, el, lon, lat, ctime))

        # Quaternions may differ by sign (q and -q represent same rotation)
        if np.dot(q_c, q_j) < 0:
            q_j = -q_j

        assert_allclose(q_j, q_c, atol=1e-6,
                       err_msg=f"site={site_name}, ctime={ctime}")

    @pytest.mark.parametrize('az', [0.0, 45.0, 90.0, 180.0, 270.0])
    @pytest.mark.parametrize('el', [10.0, 30.0, 45.0, 60.0, 85.0])
    def test_azel_grid(self, az, el):
        """Test across az/el grid at South Pole."""
        lon, lat = -44.65, -89.99
        ctime = 1700000000.0

        cQ = CQPoint()
        q_c = cQ.azel2bore(az, el, 0.0, 0.0, lon, lat, ctime).flatten()

        q_j = np.array(azel2bore(az, el, lon, lat, ctime))
        if np.dot(q_c, q_j) < 0:
            q_j = -q_j

        assert_allclose(q_j, q_c, atol=1e-6,
                       err_msg=f"az={az}, el={el}")


# ---------------------------------------------------------------------------
# RA/Dec comparison
# ---------------------------------------------------------------------------

class TestRadec:
    """Compare (RA, Dec) output between JAX and C."""

    @pytest.mark.parametrize('site_name,site', SITES.items())
    def test_bore2radec(self, site_name, site):
        """RA/Dec from boresight should match."""
        az, el = 180.0, 45.0
        lon, lat = site['lon'], site['lat']
        ctime = 1700000000.0

        # C QPoint
        cQ = CQPoint()
        q_bore_c = cQ.azel2bore(az, el, 0.0, 0.0, lon, lat, ctime)
        q_off = np.array([1.0, 0.0, 0.0, 0.0])
        ra_c, dec_c, sin2psi_c, cos2psi_c = cQ.bore2radec(q_off, ctime, q_bore_c)

        # JAX
        q_bore_j = azel2bore(az, el, lon, lat, ctime)
        ra_j, dec_j, pa_j = bore2radecpa(identity(), ctime, q_bore_j)

        assert_allclose(float(ra_j) % 360, float(ra_c) % 360, atol=0.01,
                       err_msg=f"RA mismatch at {site_name}")
        assert_allclose(float(dec_j), float(dec_c), atol=0.01,
                       err_msg=f"Dec mismatch at {site_name}")

    def test_detector_offset(self):
        """Detector offset should produce consistent RA/Dec shift."""
        lon, lat = -44.65, -89.99
        ctime = 1700000000.0

        # C QPoint
        cQ = CQPoint()
        q_bore_c = cQ.azel2bore(180.0, 45.0, 0.0, 0.0, lon, lat, ctime)
        q_off_c = cQ.det_offset(1.0, 0.5, 0.0)
        ra_c, dec_c, _, _ = cQ.bore2radec(q_off_c, ctime, q_bore_c)

        # JAX
        q_bore_j = azel2bore(180.0, 45.0, lon, lat, ctime)
        q_off_j = det_offset_quat(1.0, 0.5, 0.0)
        ra_j, dec_j, pa_j = bore2radecpa(q_off_j, ctime, q_bore_j)

        assert_allclose(float(ra_j) % 360, float(ra_c) % 360, atol=0.01)
        assert_allclose(float(dec_j), float(dec_c), atol=0.01)


# ---------------------------------------------------------------------------
# Time series (vectorized) comparison
# ---------------------------------------------------------------------------

class TestTimeSeries:
    """Compare vectorized pipeline over time series."""

    def test_scan_south_pole(self):
        """Simulate a scan at the South Pole and compare RA/Dec."""
        n = 100
        lon, lat = -44.65, -89.99
        ctimes = np.linspace(1700000000.0, 1700000000.0 + 600.0, n)
        az = np.linspace(0, 360, n)
        el = np.full(n, 45.0)

        # C QPoint
        cQ = CQPoint()
        q_bore_c = cQ.azel2bore(az, el, np.zeros(n), np.zeros(n),
                                np.full(n, lon), np.full(n, lat), ctimes)
        q_off = np.array([1.0, 0.0, 0.0, 0.0])
        ra_c, dec_c, _, _ = cQ.bore2radec(q_off, ctimes, q_bore_c)

        # JAX (vectorized)
        times = precompute_times(ctimes)
        def forward(az_i, el_i, tt1, tt2, ut1_1, ut1_2):
            q = azelpsi2bore_jit(az_i, el_i, 0.0, lon, lat,
                                 tt1, tt2, ut1_1, ut1_2)
            ra, dec, pa = quat2radecpa(q)
            return ra, dec

        ra_j, dec_j = jax.vmap(forward)(
            jnp.array(az), jnp.array(el),
            times['tt1'], times['tt2'],
            times['ut1_1'], times['ut1_2'],
        )

        # Compare (allow small differences due to implementation details)
        ra_j_mod = np.array(ra_j) % 360
        ra_c_mod = np.array(ra_c) % 360
        # Handle wrap-around
        ra_diff = np.abs(ra_j_mod - ra_c_mod)
        ra_diff = np.minimum(ra_diff, 360 - ra_diff)

        assert np.all(ra_diff < 0.05), \
            f"Max RA diff: {np.max(ra_diff):.4f} deg"
        assert_allclose(np.array(dec_j), np.array(dec_c), atol=0.05)

    def test_scan_atacama(self):
        """Simulate a scan at Atacama."""
        n = 50
        lon, lat = -67.79, -22.96
        ctimes = np.linspace(1700000000.0, 1700000000.0 + 300.0, n)
        az = np.linspace(30, 330, n)
        el = np.full(n, 60.0)

        # C QPoint
        cQ = CQPoint()
        q_bore_c = cQ.azel2bore(az, el, np.zeros(n), np.zeros(n),
                                np.full(n, lon), np.full(n, lat), ctimes)
        q_off = np.array([1.0, 0.0, 0.0, 0.0])
        ra_c, dec_c, _, _ = cQ.bore2radec(q_off, ctimes, q_bore_c)

        # JAX
        times = precompute_times(ctimes)
        def forward(az_i, el_i, tt1, tt2, ut1_1, ut1_2):
            q = azelpsi2bore_jit(az_i, el_i, 0.0, lon, lat,
                                 tt1, tt2, ut1_1, ut1_2)
            ra, dec, pa = quat2radecpa(q)
            return ra, dec

        ra_j, dec_j = jax.vmap(forward)(
            jnp.array(az), jnp.array(el),
            times['tt1'], times['tt2'],
            times['ut1_1'], times['ut1_2'],
        )

        ra_j_mod = np.array(ra_j) % 360
        ra_c_mod = np.array(ra_c) % 360
        ra_diff = np.abs(ra_j_mod - ra_c_mod)
        ra_diff = np.minimum(ra_diff, 360 - ra_diff)

        assert np.all(ra_diff < 0.05), \
            f"Max RA diff: {np.max(ra_diff):.4f} deg"
        assert_allclose(np.array(dec_j), np.array(dec_c), atol=0.05)


# ---------------------------------------------------------------------------
# Gradient correctness (finite difference check)
# ---------------------------------------------------------------------------

class TestGradients:
    """Verify gradients via finite differences."""

    def _finite_diff(self, f, x, eps=1e-5):
        return (f(x + eps) - f(x - eps)) / (2 * eps)

    def test_dra_daz(self):
        """d(ra)/d(az) via grad vs finite difference."""
        lon, lat = -44.65, -89.99
        ctime = 1700000000.0

        def ra_from_az(az):
            q = azel2bore(az, 45.0, lon, lat, ctime)
            ra, _, _ = quat2radecpa(q)
            return ra

        g_jax = float(jax.grad(ra_from_az)(180.0))
        g_fd = float(self._finite_diff(ra_from_az, 180.0))
        assert_allclose(g_jax, g_fd, rtol=1e-4,
                       err_msg=f"jax={g_jax:.6f}, fd={g_fd:.6f}")

    def test_ddec_del(self):
        """d(dec)/d(el) via grad vs finite difference."""
        lon, lat = -44.65, -89.99
        ctime = 1700000000.0

        def dec_from_el(el):
            q = azel2bore(180.0, el, lon, lat, ctime)
            _, dec, _ = quat2radecpa(q)
            return dec

        g_jax = float(jax.grad(dec_from_el)(45.0))
        g_fd = float(self._finite_diff(dec_from_el, 45.0))
        assert_allclose(g_jax, g_fd, rtol=1e-4,
                       err_msg=f"jax={g_jax:.6f}, fd={g_fd:.6f}")

    def test_dra_dlon(self):
        """d(ra)/d(lon) via grad vs finite difference."""
        lat = -89.99
        ctime = 1700000000.0

        def ra_from_lon(lon):
            q = azel2bore(180.0, 45.0, lon, lat, ctime)
            ra, _, _ = quat2radecpa(q)
            return ra

        g_jax = float(jax.grad(ra_from_lon)(-44.65))
        g_fd = float(self._finite_diff(ra_from_lon, -44.65))
        assert_allclose(g_jax, g_fd, rtol=1e-4,
                       err_msg=f"jax={g_jax:.6f}, fd={g_fd:.6f}")

    def test_ddec_dlat(self):
        """d(dec)/d(lat) via grad vs finite difference."""
        lon = -44.65
        ctime = 1700000000.0

        def dec_from_lat(lat):
            q = azel2bore(180.0, 45.0, lon, lat, ctime)
            _, dec, _ = quat2radecpa(q)
            return dec

        g_jax = float(jax.grad(dec_from_lat)(-22.96))
        g_fd = float(self._finite_diff(dec_from_lat, -22.96))
        assert_allclose(g_jax, g_fd, rtol=1e-4,
                       err_msg=f"jax={g_jax:.6f}, fd={g_fd:.6f}")

    def test_full_jacobian(self):
        """Full Jacobian d(ra,dec)/d(az,el) at mid-latitude."""
        lon, lat = -67.79, -22.96
        ctime = 1700000000.0

        def radec_from_azel(azel):
            q = azel2bore(azel[0], azel[1], lon, lat, ctime)
            ra, dec, _ = quat2radecpa(q)
            return jnp.array([ra, dec])

        azel = jnp.array([45.0, 60.0])
        J = jax.jacobian(radec_from_azel)(azel)

        # Verify with finite differences
        eps = 1e-5
        for i in range(2):
            azel_plus = azel.at[i].set(azel[i] + eps)
            azel_minus = azel.at[i].set(azel[i] - eps)
            fd_col = (radec_from_azel(azel_plus) - radec_from_azel(azel_minus)) / (2 * eps)
            assert_allclose(np.array(J[:, i]), np.array(fd_col), rtol=1e-4,
                           err_msg=f"Jacobian column {i}")


# ---------------------------------------------------------------------------
# QPoint class API comparison
# ---------------------------------------------------------------------------

class TestQPointClassAPI:
    """Test the QPoint class API matches C behavior."""

    def test_set_get(self):
        Q = JaxQPoint(accuracy=0)
        assert Q.get('accuracy') == 0
        Q.set(accuracy=1)
        assert Q.get('accuracy') == 1

    def test_full_pipeline(self):
        """Full pipeline through QPoint class should match C."""
        lon, lat = -44.65, -89.99
        ctime = 1700000000.0

        cQ = CQPoint()
        q_c = cQ.azel2bore(180.0, 45.0, 0.0, 0.0, lon, lat, ctime).flatten()

        jQ = JaxQPoint()
        q_j = np.array(jQ.azel2bore(180.0, 45.0, 0.0, 0.0, lon, lat, ctime))

        if np.dot(q_c, q_j) < 0:
            q_j = -q_j
        assert_allclose(q_j, q_c, atol=1e-6)
