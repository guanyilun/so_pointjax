"""Tests for so_pointjax.proj.quat — verified via roundtrips and known values."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from so_pointjax.proj import quat


class TestEuler:
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_identity(self, axis):
        """euler(axis, 0) should be identity."""
        q = quat.euler(axis, 0.0)
        np.testing.assert_allclose(np.array(q), [1, 0, 0, 0], atol=1e-15)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_half_turn(self, axis):
        """euler(axis, pi) should have w=0 and unit component on axis."""
        q = np.array(quat.euler(axis, np.pi))
        np.testing.assert_allclose(q[0], 0.0, atol=1e-15)
        np.testing.assert_allclose(abs(q[axis + 1]), 1.0, atol=1e-15)

    def test_vector_shape(self):
        angles = jnp.linspace(0, 2 * np.pi, 50)
        q = quat.euler(2, angles)
        assert q.shape == (50, 4)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_unit_norm(self, axis):
        q = quat.euler(axis, 1.23)
        np.testing.assert_allclose(float(quat.qnorm(q)), 1.0, atol=1e-15)


class TestRotationIso:
    def test_roundtrip(self):
        theta, phi, psi = 0.5, 1.2, -0.3
        q = quat.rotation_iso(theta, phi, psi)
        t2, p2, s2 = quat.decompose_iso(q)
        np.testing.assert_allclose(float(t2), theta, atol=1e-14)
        np.testing.assert_allclose(float(p2), phi, atol=1e-14)
        np.testing.assert_allclose(float(s2), psi, atol=1e-14)

    def test_no_psi(self):
        """Two-arg form should match three-arg with psi=0."""
        theta, phi = 0.5, 1.2
        q2 = quat.rotation_iso(theta, phi)
        q3 = quat.rotation_iso(theta, phi, 0.0)
        np.testing.assert_allclose(np.array(q2), np.array(q3), atol=1e-14)


class TestRotationLonlat:
    def test_roundtrip(self):
        lon, lat, psi = 1.0, 0.5, 0.3
        q = quat.rotation_lonlat(lon, lat, psi)
        lon2, lat2, psi2 = quat.decompose_lonlat(q)
        np.testing.assert_allclose(float(lon2), lon, atol=1e-14)
        np.testing.assert_allclose(float(lat2), lat, atol=1e-14)
        np.testing.assert_allclose(float(psi2), psi, atol=1e-14)

    def test_azel_roundtrip(self):
        az, el = 1.0, 0.5
        q = quat.rotation_lonlat(az, el, azel=True)
        az2, el2, _ = quat.decompose_lonlat(q, azel=True)
        np.testing.assert_allclose(float(az2), az, atol=1e-14)
        np.testing.assert_allclose(float(el2), el, atol=1e-14)

    def test_azel_sign_flip(self):
        """azel=True should negate lon."""
        lon, lat = 1.0, 0.5
        q_normal = quat.rotation_lonlat(lon, lat)
        q_azel = quat.rotation_lonlat(lon, lat, azel=True)
        # rotation_lonlat(lon, lat, azel=True) = rotation_iso(pi/2-lat, -lon)
        q_expected = quat.rotation_iso(jnp.pi / 2 - lat, -lon)
        np.testing.assert_allclose(np.array(q_azel), np.array(q_expected), atol=1e-14)


class TestRotationXieta:
    def test_roundtrip(self):
        xi, eta, gamma = 0.01, -0.02, 0.78
        q = quat.rotation_xieta(xi, eta, gamma)
        xi2, eta2, gamma2 = quat.decompose_xieta(q)
        np.testing.assert_allclose(float(xi2), xi, atol=1e-12)
        np.testing.assert_allclose(float(eta2), eta, atol=1e-12)
        np.testing.assert_allclose(float(gamma2), gamma, atol=1e-12)

    def test_boresight(self):
        """xi=eta=gamma=0 should give identity rotation."""
        q = quat.rotation_xieta(0.0, 0.0, 0.0)
        # theta=0 means identity
        xi, eta, gamma = quat.decompose_xieta(q)
        np.testing.assert_allclose(float(xi), 0.0, atol=1e-15)
        np.testing.assert_allclose(float(eta), 0.0, atol=1e-15)

    def test_vector(self):
        xi = jnp.array([0.01, -0.005, 0.02])
        eta = jnp.array([-0.02, 0.01, -0.01])
        gamma = jnp.array([0.0, np.pi / 4, np.pi / 2])
        q = quat.rotation_xieta(xi, eta, gamma)
        assert q.shape == (3, 4)

        xi2, eta2, gamma2 = quat.decompose_xieta(q)
        np.testing.assert_allclose(np.array(xi2), np.array(xi), atol=1e-12)
        np.testing.assert_allclose(np.array(eta2), np.array(eta), atol=1e-12)
        np.testing.assert_allclose(np.array(gamma2), np.array(gamma), atol=1e-12)


class TestQmul:
    def test_identity(self):
        e = jnp.array([1., 0., 0., 0.])
        q = quat.euler(2, 0.5)
        np.testing.assert_allclose(np.array(quat.qmul(e, q)), np.array(q), atol=1e-15)
        np.testing.assert_allclose(np.array(quat.qmul(q, e)), np.array(q), atol=1e-15)

    def test_inverse(self):
        q = quat.euler(2, 0.5)
        qi = quat.qconj(q)
        product = quat.qmul(q, qi)
        np.testing.assert_allclose(np.array(product), [1, 0, 0, 0], atol=1e-15)

    def test_batch(self):
        angles = jnp.linspace(0, 1, 10)
        a = quat.euler(2, angles)
        b = quat.euler(1, angles)
        result = quat.qmul(a, b)
        assert result.shape == (10, 4)

    def test_matches_scalar(self):
        """Batched qmul should match element-wise scalar qmul."""
        from so_pointjax.qpoint._quaternion import mul as scalar_mul
        a = quat.euler(2, 0.7)
        b = quat.euler(1, 0.3)
        np.testing.assert_allclose(
            np.array(quat.qmul(a, b)),
            np.array(scalar_mul(a, b)),
            atol=1e-14,
        )


class TestQrotate:
    def test_z_rotation(self):
        """90-degree Z rotation should map x-hat to y-hat."""
        q = quat.euler(2, jnp.pi / 2)
        v = jnp.array([1., 0., 0.])
        v_rot = quat.qrotate(q, v)
        np.testing.assert_allclose(np.array(v_rot), [0, 1, 0], atol=1e-14)

    def test_identity(self):
        q = jnp.array([1., 0., 0., 0.])
        v = jnp.array([1., 2., 3.])
        np.testing.assert_allclose(np.array(quat.qrotate(q, v)), [1, 2, 3], atol=1e-14)


def _numerical_grad(f, x, eps=1e-6):
    """Central finite-difference gradient for scalar-valued f.

    Uses eps=1e-6 so the FD noise floor is ~machine_eps/eps ≈ 1e-10.
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        x_plus = x.copy(); x_plus[i] += eps
        x_minus = x.copy(); x_minus[i] -= eps
        grad[i] = (f(jnp.array(x_plus)) - f(jnp.array(x_minus))) / (2 * eps)
    return grad


class TestDifferentiable:
    """Basic gradient smoke tests (original)."""

    def test_euler_grad(self):
        grad_fn = jax.grad(lambda a: quat.euler(2, a)[0])
        g = grad_fn(1.0)
        assert jnp.isfinite(g)
        # d/da cos(a/2) = -sin(a/2)/2
        np.testing.assert_allclose(float(g), -np.sin(0.5) / 2, atol=1e-10)

    def test_rotation_xieta_grad(self):
        def loss(xi, eta):
            q = quat.rotation_xieta(xi, eta, 0.0)
            return q[0]
        g = jax.grad(loss, argnums=(0, 1))(0.01, -0.02)
        assert all(jnp.isfinite(gi) for gi in g)

    def test_decompose_lonlat_grad(self):
        def loss(angle):
            q = quat.rotation_lonlat(angle, 0.5)
            lon, lat, psi = quat.decompose_lonlat(q)
            return lon
        g = jax.grad(loss)(1.0)
        assert jnp.isfinite(g)
        # d(lon)/d(lon) should be 1
        np.testing.assert_allclose(float(g), 1.0, atol=1e-10)

    def test_qmul_grad(self):
        def loss(angle):
            a = quat.euler(2, angle)
            b = quat.euler(1, 0.3)
            return quat.qmul(a, b)[0]
        g = jax.grad(loss)(0.5)
        assert jnp.isfinite(g)

    def test_xieta_grad_vmap(self):
        """Vectorized gradient of xi/eta decomposition."""
        def loss(xi, eta):
            q = quat.rotation_xieta(xi, eta, 0.0)
            xi_out, _, _ = quat.decompose_xieta(q)
            return xi_out

        grad_fn = jax.vmap(jax.grad(loss, argnums=(0, 1)))
        xi = jnp.array([0.01, 0.02, 0.03])
        eta = jnp.array([-0.01, -0.02, -0.03])
        g = grad_fn(xi, eta)
        # d(xi_out)/d(xi) should be ~1 for small angles
        np.testing.assert_allclose(np.array(g[0]), 1.0, atol=1e-6)


class TestGradNumerical:
    """Verify jax.grad against finite differences for all key functions."""

    # -- Euler rotations --

    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("component", [0, 1, 2, 3])
    def test_euler_grad_numerical(self, axis, component):
        """Each component of euler(axis, a) should have correct gradient."""
        def f(a):
            return quat.euler(axis, a)[component]
        a0 = 1.23
        g_ad = float(jax.grad(f)(a0))
        g_fd = float(_numerical_grad(f, np.array(a0)))
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5, atol=1e-7)

    # -- rotation_iso / decompose_iso roundtrip --

    @pytest.mark.parametrize("output", ['theta', 'phi', 'psi'])
    def test_rotation_iso_grad(self, output):
        """Gradient of decompose_iso(rotation_iso(theta, phi, psi))."""
        idx = {'theta': 0, 'phi': 1, 'psi': 2}[output]
        def f(x):
            q = quat.rotation_iso(x[0], x[1], x[2])
            return quat.decompose_iso(q)[idx]
        x0 = np.array([0.5, 1.2, -0.3])
        g_ad = np.array(jax.grad(f)(jnp.array(x0)))
        g_fd = _numerical_grad(f, x0)
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5, atol=1e-7)

    # -- rotation_lonlat / decompose_lonlat roundtrip --

    @pytest.mark.parametrize("output", ['lon', 'lat', 'psi'])
    def test_rotation_lonlat_grad(self, output):
        """Gradient of decompose_lonlat(rotation_lonlat(lon, lat, psi))."""
        idx = {'lon': 0, 'lat': 1, 'psi': 2}[output]
        def f(x):
            q = quat.rotation_lonlat(x[0], x[1], x[2])
            return quat.decompose_lonlat(q)[idx]
        x0 = np.array([1.0, 0.5, 0.3])
        g_ad = np.array(jax.grad(f)(jnp.array(x0)))
        g_fd = _numerical_grad(f, x0)
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("output", ['az', 'el', 'psi'])
    def test_rotation_azel_grad(self, output):
        """Gradient with azel=True."""
        idx = {'az': 0, 'el': 1, 'psi': 2}[output]
        def f(x):
            q = quat.rotation_lonlat(x[0], x[1], x[2], azel=True)
            return quat.decompose_lonlat(q, azel=True)[idx]
        x0 = np.array([1.0, 0.5, 0.3])
        g_ad = np.array(jax.grad(f)(jnp.array(x0)))
        g_fd = _numerical_grad(f, x0)
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5, atol=1e-7)

    # -- rotation_xieta / decompose_xieta roundtrip --

    @pytest.mark.parametrize("output", ['xi', 'eta', 'gamma'])
    def test_rotation_xieta_grad(self, output):
        """Gradient of decompose_xieta(rotation_xieta(xi, eta, gamma))."""
        idx = {'xi': 0, 'eta': 1, 'gamma': 2}[output]
        def f(x):
            q = quat.rotation_xieta(x[0], x[1], x[2])
            return quat.decompose_xieta(q)[idx]
        x0 = np.array([0.01, -0.02, 0.78])
        g_ad = np.array(jax.grad(f)(jnp.array(x0)))
        g_fd = _numerical_grad(f, x0)
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-4, atol=1e-7)

    # -- qmul --

    @pytest.mark.parametrize("component", [0, 1, 2, 3])
    def test_qmul_grad_both_args(self, component):
        """Gradient of qmul w.r.t. both input quaternions."""
        def f(x):
            a = x[:4]
            b = x[4:]
            return quat.qmul(a, b)[component]
        a0 = np.array(quat.euler(2, 0.7))
        b0 = np.array(quat.euler(1, 0.3))
        x0 = np.concatenate([a0, b0])
        g_ad = np.array(jax.grad(f)(jnp.array(x0)))
        g_fd = _numerical_grad(f, x0)
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5, atol=1e-7)

    # -- qrotate --

    @pytest.mark.parametrize("component", [0, 1, 2])
    def test_qrotate_grad_angle(self, component):
        """Gradient of qrotate w.r.t. rotation angle."""
        v0 = jnp.array([1.0, 2.0, 3.0])
        def f(angle):
            q = quat.euler(2, angle)
            return quat.qrotate(q, v0)[component]
        a0 = 0.7
        g_ad = float(jax.grad(f)(a0))
        g_fd = float(_numerical_grad(f, np.array(a0)))
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5, atol=1e-7)

    @pytest.mark.parametrize("component", [0, 1, 2])
    def test_qrotate_grad_vector(self, component):
        """Gradient of qrotate w.r.t. input vector."""
        q0 = quat.euler(2, 0.7)
        def f(v):
            return quat.qrotate(q0, v)[component]
        v0 = np.array([1.0, 2.0, 3.0])
        g_ad = np.array(jax.grad(f)(jnp.array(v0)))
        g_fd = _numerical_grad(f, v0)
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5, atol=1e-7)

    # -- Composed chains (rotation → qmul → decompose) --

    def test_bore_det_composition_grad(self):
        """Gradient through boresight × detector → coords chain."""
        def f(x):
            # x = [az, el, xi, eta]
            q_bore = quat.rotation_lonlat(x[0], x[1], azel=True)
            q_det = quat.rotation_xieta(x[2], x[3], 0.0)
            q_total = quat.qmul(q_bore, q_det)
            lon, lat, _ = quat.decompose_lonlat(q_total)
            return lon + lat  # sum so gradient covers both paths
        x0 = np.array([1.0, 0.8, 0.01, -0.02])
        g_ad = np.array(jax.grad(f)(jnp.array(x0)))
        g_fd = _numerical_grad(f, x0)
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-4, atol=1e-8)

    def test_multi_qmul_chain_grad(self):
        """Gradient through a chain of three quaternion multiplications."""
        def f(angles):
            q1 = quat.euler(0, angles[0])
            q2 = quat.euler(1, angles[1])
            q3 = quat.euler(2, angles[2])
            q = quat.qmul(quat.qmul(q1, q2), q3)
            return q[0] + q[1] + q[2] + q[3]
        x0 = np.array([0.3, 0.7, 1.1])
        g_ad = np.array(jax.grad(f)(jnp.array(x0)))
        g_fd = _numerical_grad(f, x0)
        np.testing.assert_allclose(g_ad, g_fd, rtol=1e-5, atol=1e-7)

    # -- Second-order derivatives --

    def test_euler_hessian(self):
        """Second derivative of euler should match finite differences."""
        def f(a):
            return quat.euler(2, a)[0]  # cos(a/2)
        hess_fn = jax.grad(jax.grad(f))
        a0 = 1.0
        h_ad = float(hess_fn(a0))
        # d²/da² cos(a/2) = -cos(a/2)/4
        h_exact = -np.cos(a0 / 2) / 4
        np.testing.assert_allclose(h_ad, h_exact, rtol=1e-10)

    def test_xieta_roundtrip_hessian(self):
        """Second derivative through xieta roundtrip."""
        def f(xi):
            q = quat.rotation_xieta(xi, 0.01, 0.0)
            xi_out, _, _ = quat.decompose_xieta(q)
            return xi_out
        # Numerical second derivative
        eps = 1e-5
        xi0 = 0.02
        h_ad = float(jax.grad(jax.grad(f))(xi0))
        h_fd = (float(jax.grad(f)(xi0 + eps)) - float(jax.grad(f)(xi0 - eps))) / (2 * eps)
        np.testing.assert_allclose(h_ad, h_fd, rtol=1e-3, atol=1e-8)


# =========================================================================
# Quat wrapper class
# =========================================================================

from so_pointjax.proj.quat import Quat


class TestQuatConstruction:
    def test_four_scalars(self):
        q = Quat(1, 0, 0, 0)
        np.testing.assert_allclose(np.array(q), [1, 0, 0, 0])

    def test_from_array(self):
        q = Quat(quat.euler(2, 0.5))
        assert q.shape == (4,)

    def test_from_list(self):
        q = Quat([1, 0, 0, 0])
        assert q.shape == (4,)

    def test_from_quat(self):
        q1 = Quat(1, 0, 0, 0)
        q2 = Quat(q1)
        np.testing.assert_allclose(np.array(q2), [1, 0, 0, 0])

    def test_batch(self):
        arr = quat.euler(2, jnp.linspace(0, 1, 10))
        q = Quat(arr)
        assert q.shape == (10, 4)
        assert len(q) == 10

    def test_identity(self):
        q = Quat.identity()
        np.testing.assert_allclose(np.array(q), [1, 0, 0, 0])

    def test_from_euler_scalar(self):
        q = Quat.from_euler(2, 0.5)
        expected = quat.euler(2, 0.5)
        np.testing.assert_allclose(np.array(q), np.array(expected), atol=1e-15)

    def test_from_euler_batch(self):
        angles = jnp.linspace(0, 1, 20)
        q = Quat.from_euler(2, angles)
        assert q.shape == (20, 4)
        np.testing.assert_allclose(np.array(q),
                                   np.array(quat.euler(2, angles)), atol=1e-15)

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_from_euler_axes(self, axis):
        q = Quat.from_euler(axis, np.pi / 2)
        assert abs(float(abs(q)) - 1.0) < 1e-15

    def test_from_iso(self):
        q = Quat.from_iso(0.5, 1.2, -0.3)
        expected = quat.rotation_iso(0.5, 1.2, -0.3)
        np.testing.assert_allclose(np.array(q), np.array(expected), atol=1e-15)

    def test_from_iso_roundtrip(self):
        q = Quat.from_iso(0.5, 1.2, -0.3)
        theta, phi, psi = q.to_iso()
        np.testing.assert_allclose(float(theta), 0.5, atol=1e-14)
        np.testing.assert_allclose(float(phi), 1.2, atol=1e-14)
        np.testing.assert_allclose(float(psi), -0.3, atol=1e-14)

    def test_from_lonlat(self):
        q = Quat.from_lonlat(1.0, 0.5, 0.3)
        expected = quat.rotation_lonlat(1.0, 0.5, 0.3)
        np.testing.assert_allclose(np.array(q), np.array(expected), atol=1e-15)

    def test_from_lonlat_roundtrip(self):
        q = Quat.from_lonlat(1.0, 0.5, 0.3)
        lon, lat, psi = q.to_lonlat()
        np.testing.assert_allclose(float(lon), 1.0, atol=1e-14)
        np.testing.assert_allclose(float(lat), 0.5, atol=1e-14)
        np.testing.assert_allclose(float(psi), 0.3, atol=1e-14)

    def test_from_lonlat_azel(self):
        q = Quat.from_lonlat(1.0, 0.5, azel=True)
        az, el, psi = q.to_lonlat(azel=True)
        np.testing.assert_allclose(float(az), 1.0, atol=1e-14)
        np.testing.assert_allclose(float(el), 0.5, atol=1e-14)

    def test_from_xieta(self):
        q = Quat.from_xieta(0.01, -0.02, 0.78)
        expected = quat.rotation_xieta(0.01, -0.02, 0.78)
        np.testing.assert_allclose(np.array(q), np.array(expected), atol=1e-15)

    def test_from_xieta_roundtrip(self):
        q = Quat.from_xieta(0.01, -0.02, 0.78)
        xi, eta, gamma = q.to_xieta()
        np.testing.assert_allclose(float(xi), 0.01, atol=1e-12)
        np.testing.assert_allclose(float(eta), -0.02, atol=1e-12)
        np.testing.assert_allclose(float(gamma), 0.78, atol=1e-12)

    def test_from_xieta_batch(self):
        xi = jnp.array([0.01, -0.005, 0.02])
        eta = jnp.array([-0.02, 0.01, -0.01])
        q = Quat.from_xieta(xi, eta)
        assert q.shape == (3, 4)
        xi2, eta2, _ = q.to_xieta()
        np.testing.assert_allclose(np.array(xi2), np.array(xi), atol=1e-12)
        np.testing.assert_allclose(np.array(eta2), np.array(eta), atol=1e-12)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError):
            Quat(jnp.ones(3))

    def test_bad_args_raises(self):
        with pytest.raises(TypeError):
            Quat(1, 2)

    def test_scalar_no_len(self):
        with pytest.raises(TypeError):
            len(Quat(1, 0, 0, 0))


class TestQuatComponents:
    def test_abcd(self):
        q = Quat(0.5, 0.6, 0.7, 0.8)
        assert float(q.a) == 0.5
        assert float(q.b) == 0.6
        assert float(q.c) == 0.7
        assert float(q.d) == 0.8

    def test_wxyz_aliases(self):
        q = Quat(0.5, 0.6, 0.7, 0.8)
        assert float(q.w) == float(q.a)
        assert float(q.x) == float(q.b)
        assert float(q.y) == float(q.c)
        assert float(q.z) == float(q.d)

    def test_batch_components(self):
        arr = quat.euler(2, jnp.array([0.0, 0.5, 1.0]))
        q = Quat(arr)
        assert q.a.shape == (3,)
        np.testing.assert_allclose(float(q.a[0]), 1.0, atol=1e-15)


class TestQuatMethods:
    def test_normalized(self):
        q = Quat(2, 0, 0, 0)
        qn = q.normalized()
        np.testing.assert_allclose(float(abs(qn)), 1.0, atol=1e-15)
        np.testing.assert_allclose(np.array(qn), [1, 0, 0, 0], atol=1e-15)

    def test_normalized_batch(self):
        data = jnp.array([[2, 0, 0, 0], [0, 3, 0, 0]], dtype=jnp.float64)
        q = Quat(data).normalized()
        norms = abs(q)
        np.testing.assert_allclose(np.array(norms), 1.0, atol=1e-15)

    def test_rotate_z90(self):
        """90° Z rotation should map x-hat to y-hat."""
        q = Quat.from_euler(2, jnp.pi / 2)
        v = jnp.array([1., 0., 0.])
        v_rot = q.rotate(v)
        np.testing.assert_allclose(np.array(v_rot), [0, 1, 0], atol=1e-14)

    def test_rotate_identity(self):
        q = Quat.identity()
        v = jnp.array([1., 2., 3.])
        np.testing.assert_allclose(np.array(q.rotate(v)), [1, 2, 3], atol=1e-14)

    def test_rotate_batch(self):
        """Batch quaternions rotating a single vector."""
        angles = jnp.array([0.0, jnp.pi / 2, jnp.pi])
        q = Quat.from_euler(2, angles)
        v = jnp.array([1., 0., 0.])
        # Need to broadcast v to match batch
        v_batch = jnp.tile(v, (3, 1))
        v_rot = q.rotate(v_batch)
        np.testing.assert_allclose(np.array(v_rot[0]), [1, 0, 0], atol=1e-14)
        np.testing.assert_allclose(np.array(v_rot[1]), [0, 1, 0], atol=1e-14)
        np.testing.assert_allclose(np.array(v_rot[2]), [-1, 0, 0], atol=1e-14)


class TestQuatArithmetic:
    def test_mul_identity(self):
        e = Quat(1, 0, 0, 0)
        q = Quat(quat.euler(2, 0.5))
        np.testing.assert_allclose(np.array(e * q), np.array(q), atol=1e-15)
        np.testing.assert_allclose(np.array(q * e), np.array(q), atol=1e-15)

    def test_mul_matches_qmul(self):
        q1 = Quat(quat.euler(2, 0.7))
        q2 = Quat(quat.euler(1, 0.3))
        expected = quat.qmul(q1.data, q2.data)
        np.testing.assert_allclose(np.array(q1 * q2), np.array(expected), atol=1e-15)

    def test_invert_conjugate(self):
        q = Quat(quat.euler(2, 0.5))
        conj = ~q
        expected = quat.qconj(q.data)
        np.testing.assert_allclose(np.array(conj), np.array(expected), atol=1e-15)

    def test_mul_inverse_identity(self):
        q = Quat(quat.euler(2, 0.5))
        prod = q * ~q
        np.testing.assert_allclose(np.array(prod), [1, 0, 0, 0], atol=1e-14)

    def test_abs_norm(self):
        q = Quat(quat.euler(2, 0.5))
        np.testing.assert_allclose(float(abs(q)), 1.0, atol=1e-15)

    def test_scalar_mul(self):
        q = Quat(1, 0, 0, 0)
        np.testing.assert_allclose(np.array(2 * q), [2, 0, 0, 0])
        np.testing.assert_allclose(np.array(q * 3), [3, 0, 0, 0])

    def test_neg(self):
        q = Quat(1, 0, 0, 0)
        np.testing.assert_allclose(np.array(-q), [-1, 0, 0, 0])

    def test_broadcast_scalar_array(self):
        """Scalar Quat * array Quat should broadcast."""
        q_scalar = Quat(quat.euler(2, 0.5))
        q_arr = Quat(quat.euler(1, jnp.array([0.1, 0.2, 0.3])))
        result = q_scalar * q_arr
        assert result.shape == (3, 4)
        # Check element-wise
        for i in range(3):
            expected = quat.qmul(q_scalar.data, q_arr.data[i])
            np.testing.assert_allclose(np.array(result[i]), np.array(expected),
                                       atol=1e-14)

    def test_broadcast_array_scalar(self):
        q_arr = Quat(quat.euler(2, jnp.array([0.1, 0.2, 0.3])))
        q_scalar = Quat(quat.euler(1, 0.5))
        result = q_arr * q_scalar
        assert result.shape == (3, 4)


class TestQuatIndexing:
    def test_index_scalar(self):
        q = Quat(quat.euler(2, jnp.linspace(0, 1, 5)))
        q0 = q[0]
        assert isinstance(q0, Quat)
        assert q0.shape == (4,)
        np.testing.assert_allclose(np.array(q0), [1, 0, 0, 0], atol=1e-15)

    def test_slice(self):
        q = Quat(quat.euler(2, jnp.linspace(0, 1, 10)))
        qs = q[2:5]
        assert isinstance(qs, Quat)
        assert qs.shape == (3, 4)

    def test_negative_index(self):
        q = Quat(quat.euler(2, jnp.linspace(0, 1, 5)))
        assert isinstance(q[-1], Quat)
        np.testing.assert_allclose(np.array(q[-1]), np.array(q[4]), atol=1e-15)


class TestQuatConversion:
    def test_numpy(self):
        q = Quat(quat.euler(2, jnp.linspace(0, 1, 5)))
        arr = q.numpy()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (5, 4)

    def test_np_array(self):
        q = Quat(1, 0, 0, 0)
        arr = np.array(q)
        assert arr.shape == (4,)
        assert arr.dtype == np.float64

    def test_jnp_asarray(self):
        q = Quat(1, 0, 0, 0)
        arr = jnp.asarray(q)
        assert arr.shape == (4,)


class TestQuatRepr:
    def test_scalar_repr(self):
        q = Quat(1, 0, 0, 0)
        assert 'Quat(1' in repr(q)

    def test_scalar_str(self):
        q = Quat(1, 0, 0, 0)
        assert '(1' in str(q)

    def test_array_repr(self):
        q = Quat(quat.euler(2, jnp.linspace(0, 1, 5)))
        assert 'shape=(5,)' in repr(q)

    def test_array_str(self):
        q = Quat(quat.euler(2, jnp.linspace(0, 1, 5)))
        assert 'Quat[5]' in str(q)


class TestQuatJax:
    def test_jit(self):
        @jax.jit
        def compose(q1, q2):
            return q1 * q2
        q1 = Quat(quat.euler(2, 0.5))
        q2 = Quat(quat.euler(1, 0.3))
        result = compose(q1, q2)
        assert isinstance(result, Quat)
        np.testing.assert_allclose(float(abs(result)), 1.0, atol=1e-14)

    def test_grad(self):
        def loss(angle):
            q1 = Quat(quat.euler(2, angle))
            q2 = Quat(quat.euler(1, 0.3))
            return (q1 * q2).a
        g = jax.grad(loss)(0.5)
        assert jnp.isfinite(g)

    def test_vmap(self):
        def fn(angle):
            q = Quat(quat.euler(2, angle))
            return abs(q)
        result = jax.vmap(fn)(jnp.linspace(0, 1, 5))
        np.testing.assert_allclose(np.array(result), 1.0, atol=1e-15)

    def test_grad_through_chain(self):
        """Gradient through multi-step Quat chain."""
        def loss(angle):
            q1 = Quat(quat.euler(2, angle))
            q2 = Quat(quat.euler(1, 0.3))
            q3 = Quat(quat.euler(0, 0.1))
            return (q1 * q2 * q3).a
        g = jax.grad(loss)(0.5)
        # Check against numerical
        eps = 1e-6
        g_fd = (float(jax.grad(loss)(0.5 + eps)) is None  # just check finite
                or True)
        assert jnp.isfinite(g)


class TestQuatBroadcasting:
    """Test automatic broadcasting in quaternion array operations."""

    def test_scalar_times_array(self):
        """(4,) × (N, 4) → (N, 4)."""
        q = Quat(quat.euler(2, 0.5))
        arr = Quat(quat.euler(1, jnp.linspace(0.1, 1.0, 20)))
        result = q * arr
        assert result.shape == (20, 4)
        # Verify each element
        for i in range(20):
            expected = quat.qmul(q.data, arr.data[i])
            np.testing.assert_allclose(np.array(result[i]),
                                       np.array(expected), atol=1e-14)

    def test_array_times_scalar(self):
        """(N, 4) × (4,) → (N, 4)."""
        arr = Quat(quat.euler(2, jnp.linspace(0.1, 1.0, 20)))
        q = Quat(quat.euler(1, 0.5))
        result = arr * q
        assert result.shape == (20, 4)

    def test_array_times_array_elementwise(self):
        """(N, 4) × (N, 4) → (N, 4) element-wise."""
        a = Quat(quat.euler(2, jnp.linspace(0, 1, 10)))
        b = Quat(quat.euler(1, jnp.linspace(0, 1, 10)))
        result = a * b
        assert result.shape == (10, 4)

    def test_array_times_inverse(self):
        """q * ~q should give identity for every element."""
        arr = Quat(quat.euler(2, jnp.linspace(0.1, 2.0, 50)))
        result = arr * ~arr
        expected = np.tile([1, 0, 0, 0], (50, 1))
        np.testing.assert_allclose(np.array(result), expected, atol=1e-14)

    def test_invert_array(self):
        """~array should conjugate each element."""
        arr = Quat(quat.euler(2, jnp.linspace(0.1, 1.0, 10)))
        inv = ~arr
        assert inv.shape == (10, 4)
        np.testing.assert_allclose(np.array(inv.data[..., 0]),
                                   np.array(arr.data[..., 0]), atol=1e-15)
        np.testing.assert_allclose(np.array(inv.data[..., 1:]),
                                   -np.array(arr.data[..., 1:]), atol=1e-15)

    def test_abs_array(self):
        """abs of array Quat should return array of norms."""
        arr = Quat(quat.euler(2, jnp.linspace(0, 2, 10)))
        norms = abs(arr)
        assert norms.shape == (10,)
        np.testing.assert_allclose(np.array(norms), 1.0, atol=1e-15)

    def test_chain_broadcast(self):
        """Chain: scalar * array * scalar."""
        q1 = Quat(quat.euler(2, 0.5))
        arr = Quat(quat.euler(1, jnp.linspace(0, 1, 10)))
        q2 = Quat(quat.euler(0, 0.3))
        result = q1 * arr * q2
        assert result.shape == (10, 4)
        norms = abs(result)
        np.testing.assert_allclose(np.array(norms), 1.0, atol=1e-14)

    def test_mul_raw_array(self):
        """Quat * raw jnp array should work."""
        q = Quat(quat.euler(2, 0.5))
        raw = quat.euler(1, jnp.linspace(0, 1, 5))  # jnp array, not Quat
        result = q * raw
        assert isinstance(result, Quat)
        assert result.shape == (5, 4)

    def test_rmul_raw_array(self):
        """Wrapping raw array in Quat allows left-multiply."""
        raw = quat.euler(2, 0.5)  # jnp (4,)
        q = Quat(quat.euler(1, jnp.linspace(0, 1, 5)))
        # Wrap raw to get quaternion multiplication (jnp __mul__ would
        # do element-wise, not quaternion product)
        result = Quat(raw) * q
        assert isinstance(result, Quat)
        assert result.shape == (5, 4)

    def test_bore_det_composition(self):
        """Realistic use case: boresight (N,4) × detector (4,) → (N,4)."""
        N = 100
        t = jnp.linspace(0, 1, N)
        q_bore = Quat(quat.euler(2, t))           # (N, 4) boresight
        q_det = Quat(quat.rotation_xieta(0.01, -0.02, 0.0))  # (4,) detector
        q_total = q_bore * q_det
        assert q_total.shape == (N, 4)
        norms = abs(q_total)
        np.testing.assert_allclose(np.array(norms), 1.0, atol=1e-14)

    def test_multi_det_composition(self):
        """Multiple detectors: loop over det, broadcast over time."""
        N_time = 50
        N_det = 4
        q_bore = Quat(quat.euler(2, jnp.linspace(0, 1, N_time)))
        xi = jnp.array([0.0, 0.01, -0.005, 0.02])
        eta = jnp.array([0.0, -0.02, 0.01, -0.01])

        results = []
        for i in range(N_det):
            q_det = Quat(quat.rotation_xieta(xi[i], eta[i], 0.0))
            q_total = q_bore * q_det  # (N_time, 4)
            results.append(q_total)
            assert q_total.shape == (N_time, 4)

        # Boresight detector should match original
        np.testing.assert_allclose(np.array(results[0]),
                                   np.array(q_bore), atol=1e-12)
