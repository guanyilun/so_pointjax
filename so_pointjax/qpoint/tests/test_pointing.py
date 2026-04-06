"""Tests for the core pointing pipeline."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from so_pointjax.qpoint._quaternion import (
    identity, norm, quat2radecpa, radecpa2quat,
)
from so_pointjax.qpoint._pointing import (
    azelpsi2bore, azel2bore, bore2radecpa, bore2radec,
    azel2radecpa, radec2azel,
    azelpsi2bore_jit, precompute_times,
)
from so_pointjax.qpoint._corrections import det_offset_quat


# Test parameters: South Pole Telescope approximate location
LON = -44.65  # degrees East
LAT = -89.99  # degrees (near South Pole)
CTIME = 1700000000.0  # ~2023-11-14


class TestForwardPipeline:

    def test_bore_unit_quaternion(self):
        """Boresight quaternion should be unit."""
        q = azel2bore(180.0, 45.0, LON, LAT, CTIME)
        assert_allclose(norm(q), 1.0, atol=1e-13)

    def test_different_azel_different_radec(self):
        """Different az/el should produce different RA/Dec."""
        q1 = azel2bore(0.0, 45.0, LON, LAT, CTIME)
        q2 = azel2bore(90.0, 45.0, LON, LAT, CTIME)
        ra1, dec1, _ = quat2radecpa(q1)
        ra2, dec2, _ = quat2radecpa(q2)
        assert not jnp.allclose(ra1, ra2) or not jnp.allclose(dec1, dec2)

    def test_elevation_affects_dec(self):
        """Higher elevation should produce different declination."""
        q1 = azel2bore(180.0, 30.0, LON, LAT, CTIME)
        q2 = azel2bore(180.0, 60.0, LON, LAT, CTIME)
        _, dec1, _ = quat2radecpa(q1)
        _, dec2, _ = quat2radecpa(q2)
        assert not jnp.allclose(dec1, dec2)


class TestBore2Radec:

    def test_boresight_offset_identity(self):
        """With identity offset, bore2radecpa should give same as
        directly extracting from q_bore."""
        q_bore = azel2bore(180.0, 45.0, LON, LAT, CTIME)
        q_off = identity()
        ra, dec, pa = bore2radecpa(q_off, CTIME, q_bore)
        ra2, dec2, pa2 = quat2radecpa(q_bore)
        assert_allclose(ra, ra2, atol=1e-12)
        assert_allclose(dec, dec2, atol=1e-12)
        assert_allclose(pa, pa2, atol=1e-12)

    def test_offset_shifts_pointing(self):
        """A detector offset should shift the pointing."""
        q_bore = azel2bore(180.0, 45.0, LON, LAT, CTIME)
        q_off_zero = identity()
        q_off_shift = det_offset_quat(1.0, 0.0, 0.0)  # 1 deg az offset

        ra0, dec0, _ = bore2radecpa(q_off_zero, CTIME, q_bore)
        ra1, dec1, _ = bore2radecpa(q_off_shift, CTIME, q_bore)
        # Should be different
        sep = jnp.sqrt((ra1 - ra0)**2 + (dec1 - dec0)**2)
        assert float(sep) > 0.1  # at least 0.1 degree shift


class TestAzel2Radecpa:

    def test_runs(self):
        """Complete pipeline should run without error."""
        ra, dec, pa = azel2radecpa(0.0, 0.0, 0.0,
                                   180.0, 45.0, LON, LAT, CTIME)
        assert jnp.all(jnp.isfinite(jnp.array([ra, dec, pa])))

    def test_dec_range(self):
        """Declination should be in [-90, 90]."""
        ra, dec, pa = azel2radecpa(0.0, 0.0, 0.0,
                                   180.0, 45.0, LON, LAT, CTIME)
        assert -90 <= float(dec) <= 90


class TestInversePipeline:

    def test_roundtrip_no_refraction(self):
        """Forward then inverse should recover original az/el (no refraction)."""
        az0, el0 = 180.0, 45.0
        q_bore = azelpsi2bore(az0, el0, 0.0, LON, LAT, CTIME, accuracy=1)
        ra, dec, pa = quat2radecpa(q_bore)

        az_out, el_out, _ = radec2azel(ra, dec, pa, LON, LAT, CTIME, accuracy=1)

        assert_allclose(float(az_out) % 360, az0 % 360, atol=0.1)
        assert_allclose(float(el_out), el0, atol=0.1)

    def test_roundtrip_mid_latitude(self):
        """Forward then inverse at mid-latitude site."""
        lon, lat = -67.79, -22.96  # Atacama
        az0, el0 = 45.0, 60.0
        q_bore = azelpsi2bore(az0, el0, 0.0, lon, lat, CTIME, accuracy=1)
        ra, dec, pa = quat2radecpa(q_bore)

        az_out, el_out, _ = radec2azel(ra, dec, pa, lon, lat, CTIME, accuracy=1)

        assert_allclose(float(az_out) % 360, az0 % 360, atol=0.1)
        assert_allclose(float(el_out), el0, atol=0.1)


class TestJaxTransforms:

    def test_jit_forward(self):
        """JIT-compatible pipeline should work with jit."""
        times = precompute_times(CTIME)

        @jax.jit
        def forward(az, el):
            q = azelpsi2bore_jit(az, el, 0.0, LON, LAT,
                                 times['tt1'], times['tt2'],
                                 times['ut1_1'], times['ut1_2'])
            return quat2radecpa(q)

        ra, dec, pa = forward(180.0, 45.0)
        assert jnp.all(jnp.isfinite(jnp.array([ra, dec, pa])))

    def test_grad_az(self):
        """Should be able to compute d(ra)/d(az)."""
        def ra_from_az(az):
            q = azel2bore(az, 45.0, LON, LAT, CTIME)
            ra, _, _ = quat2radecpa(q)
            return ra

        g = jax.grad(ra_from_az)(180.0)
        assert jnp.isfinite(g)
        # At the pole, d(ra)/d(az) should be approximately -1
        # (az increase → ra decrease for south pole observer)
        assert abs(float(g)) > 0.1

    def test_grad_el(self):
        """Should be able to compute d(dec)/d(el)."""
        def dec_from_el(el):
            q = azel2bore(180.0, el, LON, LAT, CTIME)
            _, dec, _ = quat2radecpa(q)
            return dec

        g = jax.grad(dec_from_el)(45.0)
        assert jnp.isfinite(g)

    def test_jacobian(self):
        """Should be able to compute full Jacobian d(ra,dec)/d(az,el)."""
        def radec_from_azel(azel):
            q = azel2bore(azel[0], azel[1], LON, LAT, CTIME)
            ra, dec, _ = quat2radecpa(q)
            return jnp.array([ra, dec])

        azel = jnp.array([180.0, 45.0])
        J = jax.jacobian(radec_from_azel)(azel)
        assert J.shape == (2, 2)
        assert jnp.all(jnp.isfinite(J))

    def test_vmap_forward(self):
        """Should be able to vmap over time samples using _jit API."""
        import numpy as np
        n = 5
        az = jnp.full(n, 180.0)
        el = jnp.full(n, 45.0)
        lon = jnp.full(n, LON)
        lat = jnp.full(n, LAT)
        ctimes = np.array([CTIME + i * 10.0 for i in range(n)])

        # Precompute times outside vmap (non-JIT)
        times = precompute_times(ctimes)

        # vmap the JIT-compatible pipeline
        def forward(az, el, lon, lat, tt1, tt2, ut1_1, ut1_2):
            return azelpsi2bore_jit(az, el, 0.0, lon, lat,
                                    tt1, tt2, ut1_1, ut1_2)

        quats = jax.vmap(forward)(az, el, lon, lat,
                                  times['tt1'], times['tt2'],
                                  times['ut1_1'], times['ut1_2'])
        assert quats.shape == (5, 4)
        ra, dec, pa = jax.vmap(quat2radecpa)(quats)
        assert ra.shape == (5,)
        assert jnp.all(jnp.isfinite(ra))
