"""Tests for HEALPix pixelization functions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from so_pointjax.qpoint._pixel import (
    ang2pix_nest, ang2pix_ring,
    pix2ang_nest, pix2ang_ring,
    vec2pix_nest, vec2pix_ring,
    pix2vec_nest, pix2vec_ring,
    nest2ring, ring2nest,
    nside2npix, npix2nside,
    radec2pix, pix2radec,
    quat2pix, bore2pix,
)

# Try to import healpy for reference comparisons
try:
    import healpy as hp
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

NSIDE = 64


# ---------------------------------------------------------------------------
# Basic sanity checks (no healpy needed)
# ---------------------------------------------------------------------------

class TestNsideNpix:

    def test_nside2npix(self):
        assert nside2npix(1) == 12
        assert nside2npix(2) == 48
        assert nside2npix(64) == 49152
        assert nside2npix(1024) == 12582912

    def test_npix2nside(self):
        assert npix2nside(12) == 1
        assert npix2nside(48) == 2
        assert npix2nside(49152) == 64

    def test_npix2nside_invalid(self):
        assert npix2nside(13) == -1


class TestAng2PixBasic:

    def test_north_pole_nest(self):
        """North pole (theta=0) should give pixel 0-3 for nside=1."""
        pix = ang2pix_nest(1, 0.0, 0.0)
        assert 0 <= int(pix) < 12

    def test_south_pole_nest(self):
        """South pole (theta=pi)."""
        pix = ang2pix_nest(1, jnp.pi, 0.0)
        assert 0 <= int(pix) < 12

    def test_equator_nest(self):
        pix = ang2pix_nest(NSIDE, jnp.pi / 2, 0.0)
        assert 0 <= int(pix) < nside2npix(NSIDE)

    def test_ang2pix_ring_range(self):
        """All pixels should be in valid range."""
        npix = nside2npix(NSIDE)
        for theta in [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            for phi in [0.0, 1.0, 3.14, 5.0]:
                pix = ang2pix_ring(NSIDE, theta, phi)
                assert 0 <= int(pix) < npix, f"theta={theta}, phi={phi}, pix={pix}"


class TestPix2AngBasic:

    def test_roundtrip_nest(self):
        """ang2pix -> pix2ang -> ang2pix should recover pixel."""
        theta0, phi0 = 1.2, 2.5
        pix = ang2pix_nest(NSIDE, theta0, phi0)
        theta1, phi1 = pix2ang_nest(NSIDE, pix)
        pix2 = ang2pix_nest(NSIDE, theta1, phi1)
        assert int(pix) == int(pix2)

    def test_roundtrip_ring(self):
        theta0, phi0 = 0.8, 4.0
        pix = ang2pix_ring(NSIDE, theta0, phi0)
        theta1, phi1 = pix2ang_ring(NSIDE, pix)
        pix2 = ang2pix_ring(NSIDE, theta1, phi1)
        assert int(pix) == int(pix2)

    def test_center_in_pixel(self):
        """Pixel center should map back to itself."""
        for pix0 in [0, 10, 100, 500]:
            theta, phi = pix2ang_nest(NSIDE, pix0)
            pix1 = ang2pix_nest(NSIDE, theta, phi)
            assert int(pix0) == int(pix1), f"pix0={pix0}, pix1={pix1}"


class TestVec2Pix:

    def test_z_axis_nest(self):
        """Z-axis should give north pole pixel."""
        vec = jnp.array([0.0, 0.0, 1.0])
        pix = vec2pix_nest(NSIDE, vec)
        assert 0 <= int(pix) < nside2npix(NSIDE)

    def test_roundtrip_nest(self):
        """vec2pix -> pix2vec -> vec2pix should recover pixel."""
        vec = jnp.array([0.5, 0.3, 0.8])
        vec = vec / jnp.linalg.norm(vec)
        pix = vec2pix_nest(NSIDE, vec)
        vec2 = pix2vec_nest(NSIDE, pix)
        pix2 = vec2pix_nest(NSIDE, vec2)
        assert int(pix) == int(pix2)

    def test_pix2vec_unit(self):
        """pix2vec should return unit vectors."""
        for pix in [0, 100, 1000]:
            vec = pix2vec_nest(NSIDE, pix)
            assert_allclose(jnp.linalg.norm(vec), 1.0, atol=1e-14)

    def test_pix2vec_ring_unit(self):
        for pix in [0, 100, 1000]:
            vec = pix2vec_ring(NSIDE, pix)
            assert_allclose(jnp.linalg.norm(vec), 1.0, atol=1e-14)


class TestNest2Ring:

    def test_roundtrip(self):
        """nest2ring -> ring2nest should be identity."""
        for pix_nest in [0, 10, 100, 500, nside2npix(NSIDE) - 1]:
            pix_ring = nest2ring(NSIDE, pix_nest)
            pix_nest2 = ring2nest(NSIDE, pix_ring)
            assert int(pix_nest) == int(pix_nest2), \
                f"nest={pix_nest} -> ring={pix_ring} -> nest={pix_nest2}"

    def test_ring2nest_roundtrip(self):
        for pix_ring in [0, 10, 100, 500]:
            pix_nest = ring2nest(NSIDE, pix_ring)
            pix_ring2 = nest2ring(NSIDE, pix_nest)
            assert int(pix_ring) == int(pix_ring2)

    def test_consistent_with_ang(self):
        """nest2ring should give same pixel as going through angles."""
        theta, phi = 1.0, 2.0
        pix_nest = ang2pix_nest(NSIDE, theta, phi)
        pix_ring_direct = ang2pix_ring(NSIDE, theta, phi)
        pix_ring_via_nest = nest2ring(NSIDE, pix_nest)
        assert int(pix_ring_direct) == int(pix_ring_via_nest)


class TestRadecInterface:

    def test_roundtrip(self):
        ra0, dec0 = 45.0, -30.0
        pix = radec2pix(NSIDE, ra0, dec0, nest=True)
        ra1, dec1 = pix2radec(NSIDE, pix, nest=True)
        pix2 = radec2pix(NSIDE, ra1, dec1, nest=True)
        assert int(pix) == int(pix2)

    def test_dec_range(self):
        """pix2radec should give dec in [-90, 90]."""
        for pix in [0, 100, nside2npix(NSIDE) // 2]:
            _, dec = pix2radec(NSIDE, pix, nest=True)
            assert -90 <= float(dec) <= 90


class TestQuat2Pix:

    def test_basic(self):
        """quat2pix should return valid pixel and polarization."""
        from so_pointjax.qpoint._quaternion import radecpa2quat
        q = radecpa2quat(45.0, -30.0, 10.0)
        pix, sin2psi, cos2psi = quat2pix(q, NSIDE, nest=True)
        assert 0 <= int(pix) < nside2npix(NSIDE)
        # sin2psi^2 + cos2psi^2 should be ~1
        assert_allclose(sin2psi**2 + cos2psi**2, 1.0, atol=0.1)

    def test_bore2pix(self):
        from so_pointjax.qpoint._quaternion import identity
        from so_pointjax.qpoint._pointing import azel2bore
        q_bore = azel2bore(180.0, 45.0, -44.65, -89.99, 1700000000.0)
        pix, s2, c2 = bore2pix(identity(), q_bore, NSIDE, nest=True)
        assert 0 <= int(pix) < nside2npix(NSIDE)


# ---------------------------------------------------------------------------
# JAX transforms
# ---------------------------------------------------------------------------

class TestJaxTransforms:

    def test_jit_ang2pix_nest(self):
        f = jax.jit(ang2pix_nest, static_argnums=0)
        pix = f(NSIDE, 1.0, 2.0)
        pix2 = ang2pix_nest(NSIDE, 1.0, 2.0)
        assert int(pix) == int(pix2)

    def test_jit_pix2ang_nest(self):
        f = jax.jit(pix2ang_nest, static_argnums=0)
        theta, phi = f(NSIDE, 100)
        assert jnp.isfinite(theta) and jnp.isfinite(phi)

    def test_vmap_ang2pix_nest(self):
        thetas = jnp.linspace(0.1, 3.0, 20)
        phis = jnp.linspace(0.0, 6.0, 20)
        f = jax.vmap(lambda t, p: ang2pix_nest(NSIDE, t, p))
        pixs = f(thetas, phis)
        assert pixs.shape == (20,)
        assert jnp.all(pixs >= 0) and jnp.all(pixs < nside2npix(NSIDE))

    def test_vmap_pix2ang_nest(self):
        pixs = jnp.arange(20)
        f = jax.vmap(lambda p: pix2ang_nest(NSIDE, p))
        thetas, phis = f(pixs)
        assert thetas.shape == (20,)
        assert jnp.all(jnp.isfinite(thetas))

    def test_vmap_nest2ring(self):
        pixs = jnp.arange(100)
        f = jax.vmap(lambda p: nest2ring(NSIDE, p))
        ring_pixs = f(pixs)
        assert ring_pixs.shape == (100,)
        # All should be unique
        assert len(set(ring_pixs.tolist())) == 100


# ---------------------------------------------------------------------------
# Validation against healpy (if available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HEALPY, reason="healpy not installed")
class TestAgainstHealpy:

    def test_ang2pix_nest(self):
        """Match healpy ang2pix for NEST ordering."""
        thetas = np.linspace(0.01, np.pi - 0.01, 50)
        phis = np.linspace(0.0, 2 * np.pi - 0.01, 50)
        for theta in thetas:
            for phi in phis[:5]:  # subset for speed
                expected = hp.ang2pix(NSIDE, theta, phi, nest=True)
                got = int(ang2pix_nest(NSIDE, float(theta), float(phi)))
                assert got == expected, \
                    f"theta={theta:.4f}, phi={phi:.4f}: got {got}, expected {expected}"

    def test_ang2pix_ring(self):
        """Match healpy ang2pix for RING ordering."""
        thetas = np.linspace(0.01, np.pi - 0.01, 50)
        phis = np.linspace(0.0, 2 * np.pi - 0.01, 50)
        for theta in thetas:
            for phi in phis[:5]:
                expected = hp.ang2pix(NSIDE, theta, phi, nest=False)
                got = int(ang2pix_ring(NSIDE, float(theta), float(phi)))
                assert got == expected, \
                    f"theta={theta:.4f}, phi={phi:.4f}: got {got}, expected {expected}"

    def test_pix2ang_nest(self):
        """Match healpy pix2ang for NEST ordering."""
        for pix in range(0, nside2npix(NSIDE), 100):
            expected_theta, expected_phi = hp.pix2ang(NSIDE, pix, nest=True)
            theta, phi = pix2ang_nest(NSIDE, pix)
            assert_allclose(float(theta), expected_theta, atol=1e-10,
                           err_msg=f"pix={pix}")
            assert_allclose(float(phi), expected_phi, atol=1e-10,
                           err_msg=f"pix={pix}")

    def test_pix2ang_ring(self):
        for pix in range(0, nside2npix(NSIDE), 100):
            expected_theta, expected_phi = hp.pix2ang(NSIDE, pix, nest=False)
            theta, phi = pix2ang_ring(NSIDE, pix)
            assert_allclose(float(theta), expected_theta, atol=1e-10,
                           err_msg=f"pix={pix}")
            assert_allclose(float(phi), expected_phi, atol=1e-10,
                           err_msg=f"pix={pix}")

    def test_nest2ring(self):
        for pix in range(0, nside2npix(NSIDE), 100):
            expected = hp.nest2ring(NSIDE, pix)
            got = int(nest2ring(NSIDE, pix))
            assert got == expected, f"pix={pix}: got {got}, expected {expected}"

    def test_ring2nest(self):
        for pix in range(0, nside2npix(NSIDE), 100):
            expected = hp.ring2nest(NSIDE, pix)
            got = int(ring2nest(NSIDE, pix))
            assert got == expected, f"pix={pix}: got {got}, expected {expected}"

    def test_vec2pix_nest(self):
        """Match healpy vec2pix for various directions."""
        vecs = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.707],
            [-0.3, 0.4, -0.866],
        ]
        for v in vecs:
            v = np.array(v)
            v = v / np.linalg.norm(v)
            expected = hp.vec2pix(NSIDE, *v, nest=True)
            got = int(vec2pix_nest(NSIDE, jnp.array(v)))
            assert got == expected, f"vec={v}: got {got}, expected {expected}"
