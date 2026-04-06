"""Tests for so_pointjax.proj.coords — verified via known values and roundtrips."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from so_pointjax.proj import quat
from so_pointjax.proj.coords import (
    EarthlySite, CelestialSightLine, FocalPlane, Assembly, SITES,
)

DEG = np.pi / 180.0


class TestEarthlySite:
    def test_sites_exist(self):
        assert 'so' in SITES
        assert 'act' in SITES
        assert SITES['so'].lon == SITES['so_lat'].lon

    def test_site_values(self):
        act = SITES['act']
        assert act.lon == -67.7876
        assert act.lat == -22.9585
        assert act.elev == 5188.

    def test_decode(self):
        site = CelestialSightLine.decode_site('act')
        assert isinstance(site, EarthlySite)
        assert site.lon == -67.7876

    def test_decode_default(self):
        site = CelestialSightLine.decode_site(None)
        assert isinstance(site, EarthlySite)

    def test_decode_object(self):
        s = EarthlySite(10.0, 20.0, 100.0)
        assert CelestialSightLine.decode_site(s) is s


class TestFocalPlane:
    def test_boresight(self):
        fp = FocalPlane.boresight()
        assert fp.ndet == 1
        np.testing.assert_allclose(np.array(fp.quats[0]), [1, 0, 0, 0])
        np.testing.assert_allclose(np.array(fp.resps[0]), [1, 1])

    def test_from_xieta_basic(self):
        xi = np.array([0.0, 0.01, -0.005])
        eta = np.array([0.0, -0.02, 0.01])
        fp = FocalPlane.from_xieta(xi, eta)
        assert fp.ndet == 3
        # First detector at boresight
        xi_out, eta_out, _ = quat.decompose_xieta(fp.quats[0])
        np.testing.assert_allclose(float(xi_out), 0.0, atol=1e-14)
        np.testing.assert_allclose(float(eta_out), 0.0, atol=1e-14)

    def test_from_xieta_roundtrip(self):
        xi = np.array([0.01, -0.005, 0.02])
        eta = np.array([-0.02, 0.01, -0.01])
        gamma = np.array([0.0, np.pi / 4, np.pi / 2])
        fp = FocalPlane.from_xieta(xi, eta, gamma)

        xi2, eta2, gamma2 = quat.decompose_xieta(fp.quats)
        np.testing.assert_allclose(np.array(xi2), xi, atol=1e-12)
        np.testing.assert_allclose(np.array(eta2), eta, atol=1e-12)
        np.testing.assert_allclose(np.array(gamma2), gamma, atol=1e-12)

    def test_from_xieta_response(self):
        fp = FocalPlane.from_xieta(
            np.array([0.01, -0.005]),
            np.array([-0.02, 0.01]),
            T=0.9, P=0.5,
        )
        np.testing.assert_allclose(np.array(fp.resps[:, 0]), 0.9, atol=1e-14)
        np.testing.assert_allclose(np.array(fp.resps[:, 1]), 0.5, atol=1e-14)

    def test_slice(self):
        fp = FocalPlane.from_xieta(
            np.array([0.01, -0.005, 0.02]),
            np.array([-0.02, 0.01, -0.01]),
        )
        fp2 = fp[1:]
        assert fp2.ndet == 2

    def test_len(self):
        fp = FocalPlane.from_xieta(np.zeros(5), np.zeros(5))
        assert len(fp) == 5


class TestNaiveAzEl:
    def test_shape(self):
        t = jnp.linspace(1700000000, 1700000600, 100)
        az = jnp.linspace(0, 2 * jnp.pi, 100)
        el = jnp.full(100, 45 * DEG)
        csl = CelestialSightLine.naive_az_el(t, az, el, site='act')
        assert csl.Q.shape == (100, 4)

    def test_unit_norm(self):
        t = jnp.linspace(1700000000, 1700000600, 50)
        az = jnp.linspace(0, jnp.pi, 50)
        el = jnp.full(50, 45 * DEG)
        csl = CelestialSightLine.naive_az_el(t, az, el, site='act')
        norms = jnp.sqrt(jnp.sum(csl.Q ** 2, axis=-1))
        np.testing.assert_allclose(np.array(norms), 1.0, atol=1e-14)

    def test_differentiable(self):
        def loss(az, el):
            csl = CelestialSightLine.naive_az_el(
                jnp.array([1700000000.0]),
                jnp.array([az]),
                jnp.array([el]),
                site='act',
            )
            c = csl.coords()
            return c[0, 0]  # lon of first sample

        g = jax.grad(loss, argnums=(0, 1))(1.0, 0.8)
        assert all(jnp.isfinite(gi) for gi in g)


class TestForLonlat:
    def test_roundtrip_via_coords(self):
        lon, lat = 1.5, -0.3
        csl = CelestialSightLine.for_lonlat(lon, lat)
        c = csl.coords()
        np.testing.assert_allclose(float(c[..., 0]), lon, atol=1e-10)
        np.testing.assert_allclose(float(c[..., 1]), lat, atol=1e-10)

    def test_with_psi(self):
        lon, lat, psi = 1.5, -0.3, 0.5
        csl = CelestialSightLine.for_lonlat(lon, lat, psi)
        c = csl.coords()
        np.testing.assert_allclose(float(c[..., 0]), lon, atol=1e-10)
        np.testing.assert_allclose(float(c[..., 1]), lat, atol=1e-10)


class TestForHorizon:
    def test_shape(self):
        t = jnp.linspace(1700000000, 1700000600, 50)
        az = jnp.linspace(0, jnp.pi, 50)
        el = jnp.full(50, 45 * DEG)
        csl = CelestialSightLine.for_horizon(t, az, el)
        assert csl.Q.shape == (50, 4)

    def test_unit_norm(self):
        t = jnp.array([1700000000.0])
        az = jnp.array([1.0])
        el = jnp.array([0.8])
        csl = CelestialSightLine.for_horizon(t, az, el)
        n = float(quat.qnorm(csl.Q.squeeze()))
        np.testing.assert_allclose(n, 1.0, atol=1e-14)


class TestCoords:
    def test_boresight_shape(self):
        """coords() without fplane should return (N, 4)."""
        t = jnp.linspace(1700000000, 1700000010, 10)
        az = jnp.linspace(0, 0.1, 10)
        el = jnp.full(10, 45 * DEG)
        csl = CelestialSightLine.for_horizon(t, az, el)
        c = csl.coords()
        assert c.shape == (10, 4)

    def test_with_focalplane_shape(self):
        """coords() with fplane should return (n_det, N, 4)."""
        csl = CelestialSightLine.for_lonlat(
            jnp.array([1.0, 1.1, 1.2]),
            jnp.array([0.5, 0.5, 0.5]),
        )
        fp = FocalPlane.from_xieta(
            np.array([0.0, 0.01]),
            np.array([0.0, -0.01]),
        )
        c = csl.coords(fplane=fp)
        assert c.shape == (2, 3, 4)

    def test_boresight_detector_matches(self):
        """Boresight detector should return same coords as no fplane."""
        csl = CelestialSightLine.for_lonlat(1.0, 0.5)
        c_bare = csl.coords()
        c_boresight = csl.coords(fplane=FocalPlane.boresight())
        np.testing.assert_allclose(
            np.array(c_bare).reshape(-1),
            np.array(c_boresight).reshape(-1),
            atol=1e-12,
        )

    def test_coords_differentiable(self):
        def loss(lon):
            csl = CelestialSightLine.for_lonlat(lon, 0.5)
            c = csl.coords()
            return c[..., 0].sum()

        g = jax.grad(loss)(1.0)
        assert jnp.isfinite(g)
        np.testing.assert_allclose(float(g), 1.0, atol=1e-8)

    def test_detector_offset(self):
        """Offset detector should point away from boresight."""
        csl = CelestialSightLine.for_lonlat(0.0, 0.0)
        fp = FocalPlane.from_xieta(
            np.array([0.0, 0.05]),
            np.array([0.0, 0.0]),
        )
        c = csl.coords(fplane=fp)
        # Scalar sightline → (n_det, 4)
        lon0 = float(c[0, 0])
        lon1 = float(c[1, 0])
        assert abs(lon1 - lon0) > 0.01


class TestAssembly:
    def test_attach(self):
        csl = CelestialSightLine.for_lonlat(1.0, 0.5)
        fp = FocalPlane.boresight()
        asm = Assembly.attach(csl, fp)
        assert asm.fplane.ndet == 1

    def test_for_boresight(self):
        csl = CelestialSightLine.for_lonlat(1.0, 0.5)
        asm = Assembly.for_boresight(csl)
        assert asm.collapse is True
        assert asm.fplane.ndet == 1

    def test_attach_array(self):
        """Should accept raw quaternion array."""
        q = jnp.array([[1., 0., 0., 0.]])
        fp = FocalPlane.boresight()
        asm = Assembly.attach(q, fp)
        np.testing.assert_allclose(np.array(asm.Q), np.array(q))


def _numerical_grad_scalar(f, x, eps=1e-7):
    """Central finite-difference gradient for scalar f, scalar or array x."""
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        grad[i] = (float(f(jnp.array(x_plus))) - float(f(jnp.array(x_minus)))) / (2 * eps)
    return grad


class TestCoordsGradNumerical:
    """Verify autodiff against finite differences for coord-level functions."""

    def test_for_lonlat_lon_grad(self):
        """d(lon_out)/d(lon_in) through for_lonlat → coords."""
        def f(lon):
            csl = CelestialSightLine.for_lonlat(lon, 0.5)
            c = csl.coords()
            return c[..., 0].sum()
        g_ad = float(jax.grad(f)(1.0))
        g_fd = float(_numerical_grad_scalar(f, np.array(1.0)))
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5)

    def test_for_lonlat_lat_grad(self):
        """d(lat_out)/d(lat_in) through for_lonlat → coords."""
        def f(lat):
            csl = CelestialSightLine.for_lonlat(1.0, lat)
            c = csl.coords()
            return c[..., 1].sum()
        g_ad = float(jax.grad(f)(0.5))
        g_fd = float(_numerical_grad_scalar(f, np.array(0.5)))
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5)

    def test_for_lonlat_with_focalplane_grad(self):
        """Gradient of coords through for_lonlat with focal plane offsets."""
        fp = FocalPlane.from_xieta(
            np.array([0.0, 0.01, -0.005]),
            np.array([0.0, -0.02, 0.01]),
        )
        def f(x):
            csl = CelestialSightLine.for_lonlat(x[0], x[1])
            c = csl.coords(fplane=fp)
            return c[..., 0].sum() + c[..., 1].sum()
        x0 = np.array([1.0, 0.5])
        g_ad = np.array(jax.grad(f)(jnp.array(x0)))
        g_fd = _numerical_grad_scalar(f, x0)
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-4, atol=1e-8)

    def test_naive_az_el_grad(self):
        """Gradient of naive_az_el output w.r.t. az and el."""
        def f_az(az):
            csl = CelestialSightLine.naive_az_el(
                jnp.array([1700000000.0]),
                jnp.array([az]),
                jnp.array([0.8]),
                site='act',
            )
            c = csl.coords()
            return c[0, 0]
        g_ad = float(jax.grad(f_az)(1.0))
        g_fd = float(_numerical_grad_scalar(f_az, np.array(1.0)))
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-4, atol=1e-8)

    def test_coords_psi_grad(self):
        """Gradient through psi (position angle) component."""
        def f(lon):
            csl = CelestialSightLine.for_lonlat(lon, 0.5, 0.3)
            c = csl.coords()
            return c[..., 2].sum()  # psi component
        g_ad = float(jax.grad(f)(1.0))
        g_fd = float(_numerical_grad_scalar(f, np.array(1.0)))
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-4, atol=1e-8)

    def test_batched_lonlat_grad(self):
        """Gradient with batched sightlines."""
        def f(lons):
            csl = CelestialSightLine.for_lonlat(lons, jnp.full(3, 0.5))
            c = csl.coords()
            return c[:, 0].sum()
        lons0 = np.array([0.5, 1.0, 1.5])
        g_ad = np.array(jax.grad(f)(jnp.array(lons0)))
        g_fd = _numerical_grad_scalar(f, lons0)
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5, atol=1e-8)
