"""Tests for quaternion algebra."""

import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from so_pointjax.qpoint._quaternion import (
    identity, mul, conj, norm, norm2, normalize, inv,
    r1, r2, r3, r1_mul, r2_mul, r3_mul, rot,
    to_matrix, to_col1, to_col2, to_col3,
    quat2radecpa, radecpa2quat, quat2radec, radec2quat,
    slerp,
)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

class TestCoreOps:

    def test_identity(self):
        q = identity()
        assert_allclose(q, [1, 0, 0, 0])

    def test_mul_identity_left(self):
        q = jnp.array([0.5, 0.5, 0.5, 0.5])
        assert_allclose(mul(identity(), q), q, atol=1e-15)

    def test_mul_identity_right(self):
        q = jnp.array([0.5, 0.5, 0.5, 0.5])
        assert_allclose(mul(q, identity()), q, atol=1e-15)

    def test_mul_conjugate_gives_norm2(self):
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = mul(q, conj(q))
        expected_norm2 = norm2(q)
        assert_allclose(result[0], expected_norm2, atol=1e-13)
        assert_allclose(result[1:], 0.0, atol=1e-13)

    def test_mul_associative(self):
        a = normalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
        b = normalize(jnp.array([0.5, -1.0, 0.3, 0.7]))
        c = normalize(jnp.array([-0.2, 0.8, -0.5, 0.1]))
        assert_allclose(mul(mul(a, b), c), mul(a, mul(b, c)), atol=1e-14)

    def test_conj(self):
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        qc = conj(q)
        assert_allclose(qc, [1.0, -2.0, -3.0, -4.0])

    def test_norm(self):
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        assert_allclose(norm(q), jnp.sqrt(30.0), atol=1e-15)

    def test_normalize(self):
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        u = normalize(q)
        assert_allclose(norm(u), 1.0, atol=1e-15)

    def test_inv(self):
        q = normalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
        qi = inv(q)
        result = mul(q, qi)
        assert_allclose(result, identity(), atol=1e-14)

    def test_inv_non_unit(self):
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        qi = inv(q)
        result = mul(q, qi)
        assert_allclose(result, identity(), atol=1e-14)


# ---------------------------------------------------------------------------
# Rotation generators
# ---------------------------------------------------------------------------

class TestRotationGenerators:

    def test_r1_zero(self):
        assert_allclose(r1(0.0), identity(), atol=1e-15)

    def test_r2_zero(self):
        assert_allclose(r2(0.0), identity(), atol=1e-15)

    def test_r3_zero(self):
        assert_allclose(r3(0.0), identity(), atol=1e-15)

    def test_r1_pi(self):
        q = r1(jnp.pi)
        # cos(pi/2) = 0, sin(pi/2) = 1
        assert_allclose(q, [0, 1, 0, 0], atol=1e-15)

    def test_r2_pi(self):
        q = r2(jnp.pi)
        assert_allclose(q, [0, 0, 1, 0], atol=1e-15)

    def test_r3_pi(self):
        q = r3(jnp.pi)
        assert_allclose(q, [0, 0, 0, 1], atol=1e-15)

    def test_r1_mul_matches_mul(self):
        angle = 0.7
        q = normalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
        assert_allclose(r1_mul(angle, q), mul(r1(angle), q), atol=1e-14)

    def test_r2_mul_matches_mul(self):
        angle = -1.3
        q = normalize(jnp.array([0.5, -1.0, 0.3, 0.7]))
        assert_allclose(r2_mul(angle, q), mul(r2(angle), q), atol=1e-14)

    def test_r3_mul_matches_mul(self):
        angle = 2.1
        q = normalize(jnp.array([-0.2, 0.8, -0.5, 0.1]))
        assert_allclose(r3_mul(angle, q), mul(r3(angle), q), atol=1e-14)

    def test_rot_x_axis(self):
        angle = 1.2
        axis = jnp.array([1.0, 0.0, 0.0])
        assert_allclose(rot(angle, axis), r1(angle), atol=1e-15)

    def test_rot_y_axis(self):
        angle = -0.8
        axis = jnp.array([0.0, 1.0, 0.0])
        assert_allclose(rot(angle, axis), r2(angle), atol=1e-15)

    def test_rot_z_axis(self):
        angle = 0.5
        axis = jnp.array([0.0, 0.0, 1.0])
        assert_allclose(rot(angle, axis), r3(angle), atol=1e-15)

    def test_rot_unnormalized_axis(self):
        angle = 1.0
        axis = jnp.array([3.0, 0.0, 0.0])  # unnormalized
        assert_allclose(rot(angle, axis), r1(angle), atol=1e-14)

    def test_double_rotation(self):
        # R1(a) * R1(b) = R1(a+b)
        a, b = 0.3, 0.7
        assert_allclose(mul(r1(a), r1(b)), r1(a + b), atol=1e-14)


# ---------------------------------------------------------------------------
# Matrix conversion
# ---------------------------------------------------------------------------

class TestMatrixConversion:

    def test_identity_matrix(self):
        mat = to_matrix(identity())
        assert_allclose(mat, jnp.eye(3), atol=1e-15)

    def test_r3_90_matrix(self):
        # R3(90 degrees) rotates x->y, y->-x
        q = r3(jnp.pi / 2)
        mat = to_matrix(q)
        expected = jnp.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ], dtype=float)
        assert_allclose(mat, expected, atol=1e-14)

    def test_orthogonal(self):
        q = normalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
        mat = to_matrix(q)
        assert_allclose(mat @ mat.T, jnp.eye(3), atol=1e-14)
        assert_allclose(jnp.linalg.det(mat), 1.0, atol=1e-14)

    def test_columns_match_matrix(self):
        q = normalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
        mat = to_matrix(q)
        assert_allclose(to_col1(q), mat[:, 0], atol=1e-14)
        assert_allclose(to_col2(q), mat[:, 1], atol=1e-14)
        assert_allclose(to_col3(q), mat[:, 2], atol=1e-14)


# ---------------------------------------------------------------------------
# RA/Dec/PA conversions
# ---------------------------------------------------------------------------

class TestRadecpa:

    def test_roundtrip(self):
        ra, dec, pa = 45.0, 30.0, 10.0
        q = radecpa2quat(ra, dec, pa)
        ra2, dec2, pa2 = quat2radecpa(q)
        assert_allclose(ra2, ra, atol=1e-12)
        assert_allclose(dec2, dec, atol=1e-12)
        assert_allclose(pa2, pa, atol=1e-12)

    def test_roundtrip_negative_dec(self):
        ra, dec, pa = 200.0, -45.0, -30.0
        q = radecpa2quat(ra, dec, pa)
        ra2, dec2, pa2 = quat2radecpa(q)
        # RA wraps mod 360
        assert_allclose(ra2 % 360, ra % 360, atol=1e-12)
        assert_allclose(dec2, dec, atol=1e-12)
        assert_allclose(pa2, pa, atol=1e-12)

    def test_roundtrip_zero_pa(self):
        ra, dec, pa = 120.0, 60.0, 0.0
        q = radecpa2quat(ra, dec, pa)
        ra2, dec2, pa2 = quat2radecpa(q)
        assert_allclose(ra2, ra, atol=1e-12)
        assert_allclose(dec2, dec, atol=1e-12)
        assert_allclose(pa2, pa, atol=1e-11)

    def test_radec_sin2cos2_roundtrip(self):
        ra, dec, pa = 45.0, 30.0, 10.0
        q1 = radecpa2quat(ra, dec, pa)
        ra2, dec2, s2p, c2p = quat2radec(q1)
        q2 = radec2quat(ra2, dec2, s2p, c2p)
        # Both quaternions should represent the same rotation
        ra3, dec3, pa3 = quat2radecpa(q2)
        assert_allclose(ra3, ra, atol=1e-11)
        assert_allclose(dec3, dec, atol=1e-11)
        assert_allclose(pa3, pa, atol=1e-11)


# ---------------------------------------------------------------------------
# SLERP
# ---------------------------------------------------------------------------

class TestSlerp:

    def test_endpoints(self):
        q0 = normalize(jnp.array([1.0, 0.0, 0.0, 0.0]))
        q1 = normalize(jnp.array([0.0, 1.0, 0.0, 0.0]))
        assert_allclose(slerp(q0, q1, 0.0), q0, atol=1e-14)
        assert_allclose(slerp(q0, q1, 1.0), q1, atol=1e-14)

    def test_midpoint(self):
        q0 = r3(0.0)
        q1 = r3(jnp.pi / 2)
        qm = slerp(q0, q1, 0.5)
        expected = r3(jnp.pi / 4)
        assert_allclose(qm, expected, atol=1e-14)

    def test_same_quaternion(self):
        q = normalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
        result = slerp(q, q, 0.5)
        assert_allclose(result, q, atol=1e-14)

    def test_antipodal(self):
        q0 = r3(0.0)
        q1 = -r3(0.0)  # same rotation, opposite sign
        result = slerp(q0, q1, 0.5)
        assert_allclose(norm(result), 1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# JAX transformations: jit, grad, vmap
# ---------------------------------------------------------------------------

class TestJaxTransforms:

    def test_jit_mul(self):
        a = normalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
        b = normalize(jnp.array([0.5, -1.0, 0.3, 0.7]))
        jitted = jax.jit(mul)
        assert_allclose(jitted(a, b), mul(a, b), atol=1e-15)

    def test_vmap_r3(self):
        angles = jnp.linspace(0, jnp.pi, 10)
        quats = jax.vmap(r3)(angles)
        assert quats.shape == (10, 4)
        for i, angle in enumerate(angles):
            assert_allclose(quats[i], r3(angle), atol=1e-14)

    def test_grad_norm(self):
        # d/dq |q| evaluated at q = [3,4,0,0] should be q/|q| = [0.6, 0.8, 0, 0]
        q = jnp.array([3.0, 4.0, 0.0, 0.0])
        g = jax.grad(norm)(q)
        assert_allclose(g, q / 5.0, atol=1e-14)

    def test_grad_through_radecpa(self):
        # Verify we can differentiate through the full quaternion -> ra/dec chain
        def ra_from_quat(q):
            ra, _, _ = quat2radecpa(q)
            return ra

        q = radecpa2quat(45.0, 30.0, 10.0)
        g = jax.grad(ra_from_quat)(q)
        # Just check it doesn't NaN and has correct shape
        assert g.shape == (4,)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_through_slerp(self):
        def slerp_component(t):
            q0 = r3(0.0)
            q1 = r3(1.0)
            return slerp(q0, q1, t)[0]

        g = jax.grad(slerp_component)(0.5)
        assert jnp.isfinite(g)

    def test_vmap_radecpa_roundtrip(self):
        ras = jnp.array([0.0, 45.0, 90.0, 180.0, 270.0])
        decs = jnp.array([0.0, 30.0, -30.0, 60.0, -60.0])
        pas = jnp.array([0.0, 10.0, -10.0, 45.0, -45.0])
        quats = jax.vmap(radecpa2quat)(ras, decs, pas)
        ra2, dec2, pa2 = jax.vmap(quat2radecpa)(quats)
        assert_allclose(ra2 % 360, ras % 360, atol=1e-11)
        assert_allclose(dec2, decs, atol=1e-11)
        assert_allclose(pa2, pas, atol=1e-11)
