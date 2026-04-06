"""Tests for correction functions."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from so_pointjax.qpoint._quaternion import (
    identity, mul, norm, normalize, to_matrix, to_col3,
    quat2radecpa, radecpa2quat,
)
from so_pointjax.qpoint._corrections import (
    npb_quat, erot_quat, wobble_quat, lonlat_quat,
    azel_quat, azelpsi_quat,
    refraction, refraction_quat,
    aberration, earth_orbital_beta, diurnal_aberration_beta,
    det_offset_quat, hwp_quat,
)
from so_pointjax.qpoint._time_utils import ctime2jdtt, ctime2jd, jdutc2jdut1


# ---------------------------------------------------------------------------
# NPB correction
# ---------------------------------------------------------------------------

class TestNPB:

    def test_unit_quaternion(self):
        """NPB quaternion should be unit."""
        tt1, tt2 = ctime2jdtt(1700000000.0)
        q = npb_quat(tt1, tt2, accuracy=0)
        assert_allclose(norm(q), 1.0, atol=1e-14)

    def test_low_accuracy(self):
        """Low accuracy NPB should also produce unit quaternion."""
        tt1, tt2 = ctime2jdtt(1700000000.0)
        q = npb_quat(tt1, tt2, accuracy=1)
        assert_allclose(norm(q), 1.0, atol=1e-14)

    def test_high_low_similar(self):
        """High and low accuracy should agree to ~arcsecond level."""
        tt1, tt2 = ctime2jdtt(1700000000.0)
        q_hi = npb_quat(tt1, tt2, accuracy=0)
        q_lo = npb_quat(tt1, tt2, accuracy=1)
        # Rotation matrices should be close
        m_hi = to_matrix(q_hi)
        m_lo = to_matrix(q_lo)
        assert_allclose(m_hi, m_lo, atol=1e-7)

    def test_rotation_is_small(self):
        """NPB is a small correction — rotation matrix close to identity-ish
        (but not identity since it includes precession over centuries)."""
        tt1, tt2 = ctime2jdtt(1700000000.0)
        q = npb_quat(tt1, tt2)
        # w component should be close to 1 (small rotation angle)
        # For J2000+23 years, precession is ~23*50"/yr ≈ 0.3 deg
        assert q[0] > 0.99

    def test_jit_compatible(self):
        """NPB should work with jit (the so_pointjax.erfa functions are jit-compatible)."""
        tt1, tt2 = ctime2jdtt(1700000000.0)
        q1 = npb_quat(tt1, tt2, accuracy=1)
        q2 = jax.jit(npb_quat, static_argnums=2)(tt1, tt2, 1)
        assert_allclose(q1, q2, atol=1e-15)


# ---------------------------------------------------------------------------
# Earth rotation
# ---------------------------------------------------------------------------

class TestErot:

    def test_unit_quaternion(self):
        jd1, jd2 = ctime2jd(1700000000.0)
        q = erot_quat(jd1, jd2)
        assert_allclose(norm(q), 1.0, atol=1e-15)

    def test_one_sidereal_day(self):
        """After one sidereal day, Earth rotation should be ~2pi."""
        # Sidereal day ≈ 86164.0905 seconds
        ctime0 = 1700000000.0
        ctime1 = ctime0 + 86164.0905
        jd0_1, jd0_2 = ctime2jd(ctime0)
        jd1_1, jd1_2 = ctime2jd(ctime1)
        q0 = erot_quat(jd0_1, jd0_2)
        q1 = erot_quat(jd1_1, jd1_2)
        # q0 and q1 should be very close (full rotation)
        # Their product q1 * q0^-1 should be close to R3(2pi) = identity
        from so_pointjax.qpoint._quaternion import inv
        dq = mul(q1, inv(q0))
        # w component should be close to ±1
        assert abs(abs(dq[0]) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# Wobble
# ---------------------------------------------------------------------------

class TestWobble:

    def test_zero_pole(self):
        """With zero pole coordinates, wobble should be nearly identity
        (only s' correction)."""
        tt1, tt2 = ctime2jdtt(1700000000.0)
        q = wobble_quat(tt1, tt2, 0.0, 0.0)
        # s' is very small (~microseconds of arc)
        assert_allclose(q[0], 1.0, atol=1e-10)

    def test_unit_quaternion(self):
        tt1, tt2 = ctime2jdtt(1700000000.0)
        q = wobble_quat(tt1, tt2, 0.1, 0.2)
        assert_allclose(norm(q), 1.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Lon/Lat
# ---------------------------------------------------------------------------

class TestLonLat:

    def test_unit_quaternion(self):
        q = lonlat_quat(45.0, -30.0)
        assert_allclose(norm(q), 1.0, atol=1e-15)

    def test_pole(self):
        """At the North Pole (lat=90), the lon/lat quaternion should produce
        specific rotation."""
        q = lonlat_quat(0.0, 90.0)
        assert_allclose(norm(q), 1.0, atol=1e-15)

    def test_equator_prime_meridian(self):
        """At (lon=0, lat=0), check structure."""
        q = lonlat_quat(0.0, 0.0)
        # R3(pi) * R2(pi/2) * R3(0) = R3(pi) * R2(pi/2)
        assert_allclose(norm(q), 1.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Az/El quaternion
# ---------------------------------------------------------------------------

class TestAzEl:

    def test_unit_quaternion(self):
        q = azel_quat(180.0, 45.0)
        assert_allclose(norm(q), 1.0, atol=1e-15)

    def test_azelpsi(self):
        """azelpsi with psi=0 should match azel."""
        q1 = azel_quat(120.0, 60.0, pitch=1.0, roll=0.5)
        q2 = azelpsi_quat(120.0, 60.0, 0.0, pitch=1.0, roll=0.5)
        assert_allclose(q1, q2, atol=1e-14)


# ---------------------------------------------------------------------------
# Refraction
# ---------------------------------------------------------------------------

class TestRefraction:

    def test_zero_pressure(self):
        """No atmosphere → no refraction."""
        ref = refraction(45.0, pressure=0.0)
        assert_allclose(ref, 0.0, atol=1e-15)

    def test_positive_at_low_elevation(self):
        """Refraction should be positive (bends light upward)."""
        ref = refraction(10.0, temperature=10.0, pressure=1013.25, humidity=0.5)
        assert float(ref) > 0.0

    def test_larger_at_lower_elevation(self):
        """Refraction is larger at lower elevations."""
        ref_low = refraction(5.0, temperature=10.0, pressure=1013.25)
        ref_high = refraction(45.0, temperature=10.0, pressure=1013.25)
        assert float(ref_low) > float(ref_high)


# ---------------------------------------------------------------------------
# Aberration
# ---------------------------------------------------------------------------

class TestAberration:

    def test_zero_velocity(self):
        """Zero velocity → no aberration (identity-like quaternion)."""
        q = radecpa2quat(45.0, 30.0, 0.0)
        beta = jnp.zeros(3)
        qa = aberration(q, beta)
        assert_allclose(qa[0], 1.0, atol=1e-10)

    def test_fast_matches_exact(self):
        """Fast and exact aberration should agree for small velocities."""
        q = radecpa2quat(45.0, 30.0, 0.0)
        beta = jnp.array([1e-4, 0.0, 0.0])
        qa_exact = aberration(q, beta, fast=False)
        qa_fast = aberration(q, beta, fast=True)
        assert_allclose(qa_exact, qa_fast, atol=1e-6)

    def test_earth_orbital_beta_reasonable(self):
        """Earth orbital beta should be ~1e-4."""
        tt1, tt2 = ctime2jdtt(1700000000.0)
        beta = earth_orbital_beta(tt1, tt2)
        speed = jnp.sqrt(jnp.dot(beta, beta))
        assert 5e-5 < float(speed) < 2e-4

    def test_diurnal_beta(self):
        """Diurnal aberration should be ~1.5e-6 at equator."""
        beta = diurnal_aberration_beta(0.0)  # equator
        assert_allclose(abs(beta[1]), D_ABER_RAD, atol=1e-15)
        assert_allclose(beta[0], 0.0)
        assert_allclose(beta[2], 0.0)

    def test_diurnal_beta_pole(self):
        """Diurnal aberration should be ~0 at the pole."""
        beta = diurnal_aberration_beta(90.0)
        assert_allclose(abs(beta[1]), 0.0, atol=1e-20)


# ---------------------------------------------------------------------------
# Detector offset
# ---------------------------------------------------------------------------

class TestDetOffset:

    def test_zero_offset(self):
        """Zero offset should give identity."""
        q = det_offset_quat(0.0, 0.0, 0.0)
        assert_allclose(q, identity(), atol=1e-15)

    def test_unit_quaternion(self):
        q = det_offset_quat(1.0, 2.0, 3.0)
        assert_allclose(norm(q), 1.0, atol=1e-15)


# ---------------------------------------------------------------------------
# HWP
# ---------------------------------------------------------------------------

class TestHWP:

    def test_zero_angle(self):
        q = hwp_quat(0.0)
        assert_allclose(q, identity(), atol=1e-15)

    def test_unit_quaternion(self):
        q = hwp_quat(45.0)
        assert_allclose(norm(q), 1.0, atol=1e-15)

    def test_double_angle(self):
        """HWP at 45 deg should produce R3(-90 deg) rotation."""
        from so_pointjax.qpoint._quaternion import r3
        q = hwp_quat(45.0)
        expected = r3(jnp.deg2rad(-90.0))
        assert_allclose(q, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------

class TestJaxTransforms:

    def test_jit_corrections(self):
        """All correction functions should be JIT-compatible."""
        tt1, tt2 = ctime2jdtt(1700000000.0)

        # NPB (accuracy must be static)
        jit_npb = jax.jit(npb_quat, static_argnums=2)
        q = jit_npb(tt1, tt2, 1)
        assert jnp.all(jnp.isfinite(q))

        # Erot
        jd1, jd2 = 2440587.5, 1700000000.0 / 86400.0
        q = jax.jit(erot_quat)(jd1, jd2)
        assert jnp.all(jnp.isfinite(q))

        # Wobble
        q = jax.jit(wobble_quat)(tt1, tt2, 0.1, 0.2)
        assert jnp.all(jnp.isfinite(q))

        # Lonlat
        q = jax.jit(lonlat_quat)(45.0, -30.0)
        assert jnp.all(jnp.isfinite(q))

    def test_grad_lonlat(self):
        """Should be able to differentiate lon/lat quaternion."""
        def lon_to_w(lon):
            q = lonlat_quat(lon, 45.0)
            return q[0]

        g = jax.grad(lon_to_w)(90.0)
        assert jnp.isfinite(g)

    def test_vmap_azel(self):
        """Should be able to vmap over az/el arrays."""
        az = jnp.linspace(0, 360, 10)
        el = jnp.full(10, 45.0)
        quats = jax.vmap(azel_quat)(az, el)
        assert quats.shape == (10, 4)
        norms = jax.vmap(norm)(quats)
        assert_allclose(norms, 1.0, atol=1e-15)


D_ABER_RAD = 1.54716541e-06
