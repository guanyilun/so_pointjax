"""Tests for vector/matrix functions, validated against ERFA C test suite values."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import so_pointjax.erfa as era


def assert_close(a, b, atol=1e-12):
    assert jnp.allclose(a, b, atol=atol), f"Expected {b}, got {a}"


# ===== VectorOps =====

class TestPdp:
    def test_basic(self):
        a = jnp.array([2.0, 2.0, 3.0])
        b = jnp.array([1.0, 3.0, 4.0])
        assert_close(era.pdp(a, b), 20.0)


class TestPm:
    def test_basic(self):
        p = jnp.array([0.3, 1.2, -2.5])
        assert_close(era.pm(p), 2.789265136196270604)


class TestPmp:
    def test_basic(self):
        a = jnp.array([2.0, 2.0, 3.0])
        b = jnp.array([1.0, 3.0, 4.0])
        result = era.pmp(a, b)
        assert_close(result[0], 1.0)
        assert_close(result[1], -1.0)
        assert_close(result[2], -1.0)


class TestPn:
    def test_basic(self):
        p = jnp.array([0.3, 1.2, -2.5])
        r, u = era.pn(p)
        assert_close(r, 2.789265136196270604)
        assert_close(u[0], 0.1075552109073112058)
        assert_close(u[1], 0.4302208436292448232)
        assert_close(u[2], -0.8962934242275933816)


class TestPpp:
    def test_basic(self):
        a = jnp.array([2.0, 2.0, 3.0])
        b = jnp.array([1.0, 3.0, 4.0])
        result = era.ppp(a, b)
        assert_close(result[0], 3.0)
        assert_close(result[1], 5.0)
        assert_close(result[2], 7.0)


class TestPpsp:
    def test_basic(self):
        a = jnp.array([2.0, 2.0, 3.0])
        b = jnp.array([1.0, 3.0, 4.0])
        result = era.ppsp(a, 5.0, b)
        assert_close(result[0], 7.0)
        assert_close(result[1], 17.0)
        assert_close(result[2], 23.0)


class TestPvdpv:
    def test_basic(self):
        a = jnp.array([[2.0, 2.0, 3.0], [6.0, 0.0, 4.0]])
        b = jnp.array([[1.0, 3.0, 4.0], [0.0, 2.0, 8.0]])
        result = era.pvdpv(a, b)
        assert_close(result[0], 20.0)
        assert_close(result[1], 50.0)


class TestPvm:
    def test_basic(self):
        pv = jnp.array([[0.3, 1.2, -2.5], [0.45, -0.25, 1.1]])
        r, s = era.pvm(pv)
        assert_close(r, 2.789265136196270604)
        assert_close(s, 1.214495780149111922)


class TestPvmpv:
    def test_basic(self):
        a = jnp.array([[2.0, 2.0, 3.0], [5.0, 6.0, 3.0]])
        b = jnp.array([[1.0, 3.0, 4.0], [3.0, 2.0, 1.0]])
        result = era.pvmpv(a, b)
        assert_close(result[0, 0], 1.0)
        assert_close(result[0, 1], -1.0)
        assert_close(result[0, 2], -1.0)
        assert_close(result[1, 0], 2.0)
        assert_close(result[1, 1], 4.0)
        assert_close(result[1, 2], 2.0)


class TestPvppv:
    def test_basic(self):
        a = jnp.array([[2.0, 2.0, 3.0], [5.0, 6.0, 3.0]])
        b = jnp.array([[1.0, 3.0, 4.0], [3.0, 2.0, 1.0]])
        result = era.pvppv(a, b)
        assert_close(result[0, 0], 3.0)
        assert_close(result[0, 1], 5.0)
        assert_close(result[0, 2], 7.0)
        assert_close(result[1, 0], 8.0)
        assert_close(result[1, 1], 8.0)
        assert_close(result[1, 2], 4.0)


class TestPvu:
    def test_basic(self):
        pv = jnp.array([
            [126668.5912743160734, 2136.792716839935565, -245251.2339876830229],
            [-0.4051854035740713039e-2, -0.6253919754866175788e-2, 0.1189353719774107615e-1],
        ])
        result = era.pvu(2920.0, pv)
        assert_close(result[0, 0], 126656.7598605317105, atol=1e-6)
        assert_close(result[0, 1], 2118.531271155726332, atol=1e-8)
        assert_close(result[0, 2], -245216.5048590656190, atol=1e-6)
        assert_close(result[1, 0], -0.4051854035740713039e-2)
        assert_close(result[1, 1], -0.6253919754866175788e-2)
        assert_close(result[1, 2], 0.1189353719774107615e-1)


class TestPvup:
    def test_basic(self):
        pv = jnp.array([
            [126668.5912743160734, 2136.792716839935565, -245251.2339876830229],
            [-0.4051854035740713039e-2, -0.6253919754866175788e-2, 0.1189353719774107615e-1],
        ])
        result = era.pvup(2920.0, pv)
        assert_close(result[0], 126656.7598605317105, atol=1e-6)
        assert_close(result[1], 2118.531271155726332, atol=1e-8)
        assert_close(result[2], -245216.5048590656190, atol=1e-6)


class TestPvxpv:
    def test_basic(self):
        a = jnp.array([[2.0, 2.0, 3.0], [6.0, 0.0, 4.0]])
        b = jnp.array([[1.0, 3.0, 4.0], [0.0, 2.0, 8.0]])
        result = era.pvxpv(a, b)
        assert_close(result[0, 0], -1.0)
        assert_close(result[0, 1], -5.0)
        assert_close(result[0, 2], 4.0)
        assert_close(result[1, 0], -2.0)
        assert_close(result[1, 1], -36.0)
        assert_close(result[1, 2], 22.0)


class TestPxp:
    def test_basic(self):
        a = jnp.array([2.0, 2.0, 3.0])
        b = jnp.array([1.0, 3.0, 4.0])
        result = era.pxp(a, b)
        assert_close(result[0], -1.0)
        assert_close(result[1], -5.0)
        assert_close(result[2], 4.0)


class TestS2xpv:
    def test_basic(self):
        pv = jnp.array([[0.3, 1.2, -2.5], [0.5, 2.3, -0.4]])
        result = era.s2xpv(2.0, 3.0, pv)
        assert_close(result[0, 0], 0.6)
        assert_close(result[0, 1], 2.4)
        assert_close(result[0, 2], -5.0)
        assert_close(result[1, 0], 1.5)
        assert_close(result[1, 1], 6.9)
        assert_close(result[1, 2], -1.2)


class TestSxp:
    def test_basic(self):
        p = jnp.array([0.3, 1.2, -2.5])
        result = era.sxp(2.0, p)
        assert_close(result[0], 0.6, atol=0.0)
        assert_close(result[1], 2.4, atol=0.0)
        assert_close(result[2], -5.0, atol=0.0)


class TestSxpv:
    def test_basic(self):
        pv = jnp.array([[0.3, 1.2, -2.5], [0.5, 3.2, -0.7]])
        result = era.sxpv(2.0, pv)
        assert_close(result[0, 0], 0.6, atol=0.0)
        assert_close(result[0, 1], 2.4, atol=0.0)
        assert_close(result[0, 2], -5.0, atol=0.0)
        assert_close(result[1, 0], 1.0, atol=0.0)
        assert_close(result[1, 1], 6.4, atol=0.0)
        assert_close(result[1, 2], -1.4, atol=0.0)


# ===== CopyExtendExtract =====

class TestCp:
    def test_basic(self):
        p = jnp.array([0.3, 1.2, -2.5])
        result = era.cp(p)
        assert_close(result[0], 0.3, atol=0.0)
        assert_close(result[1], 1.2, atol=0.0)
        assert_close(result[2], -2.5, atol=0.0)


class TestCpv:
    def test_basic(self):
        pv = jnp.array([[0.3, 1.2, -2.5], [-0.5, 3.1, 0.9]])
        result = era.cpv(pv)
        assert_close(result[0, 0], 0.3, atol=0.0)
        assert_close(result[1, 2], 0.9, atol=0.0)


class TestCr:
    def test_basic(self):
        r = jnp.array([[2.0, 3.0, 2.0], [3.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        result = era.cr(r)
        assert_close(result[0, 0], 2.0, atol=0.0)
        assert_close(result[2, 2], 5.0, atol=0.0)


class TestP2pv:
    def test_basic(self):
        p = jnp.array([0.25, 1.2, 3.0])
        result = era.p2pv(p)
        assert_close(result[0, 0], 0.25, atol=0.0)
        assert_close(result[0, 1], 1.2, atol=0.0)
        assert_close(result[0, 2], 3.0, atol=0.0)
        assert_close(result[1, 0], 0.0, atol=0.0)
        assert_close(result[1, 1], 0.0, atol=0.0)
        assert_close(result[1, 2], 0.0, atol=0.0)


class TestPv2p:
    def test_basic(self):
        pv = jnp.array([[0.3, 1.2, -2.5], [-0.5, 3.1, 0.9]])
        result = era.pv2p(pv)
        assert_close(result[0], 0.3, atol=0.0)
        assert_close(result[1], 1.2, atol=0.0)
        assert_close(result[2], -2.5, atol=0.0)


# ===== Initialization =====

class TestIr:
    def test_basic(self):
        result = era.ir()
        assert_close(result[0, 0], 1.0, atol=0.0)
        assert_close(result[0, 1], 0.0, atol=0.0)
        assert_close(result[1, 1], 1.0, atol=0.0)
        assert_close(result[2, 2], 1.0, atol=0.0)


class TestZp:
    def test_basic(self):
        result = era.zp()
        assert_close(result[0], 0.0, atol=0.0)
        assert_close(result[1], 0.0, atol=0.0)
        assert_close(result[2], 0.0, atol=0.0)


class TestZpv:
    def test_basic(self):
        result = era.zpv()
        assert jnp.all(result == 0.0)


class TestZr:
    def test_basic(self):
        result = era.zr()
        assert jnp.all(result == 0.0)


# ===== MatrixOps =====

class TestRxr:
    def test_basic(self):
        a = jnp.array([[2.0, 3.0, 2.0], [3.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        b = jnp.array([[1.0, 2.0, 2.0], [4.0, 1.0, 1.0], [3.0, 0.0, 1.0]])
        result = era.rxr(a, b)
        assert_close(result[0, 0], 20.0)
        assert_close(result[0, 1], 7.0)
        assert_close(result[0, 2], 9.0)
        assert_close(result[1, 0], 20.0)
        assert_close(result[1, 1], 8.0)
        assert_close(result[1, 2], 11.0)
        assert_close(result[2, 0], 34.0)
        assert_close(result[2, 1], 10.0)
        assert_close(result[2, 2], 15.0)


class TestTr:
    def test_basic(self):
        r = jnp.array([[2.0, 3.0, 2.0], [3.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        result = era.tr(r)
        assert_close(result[0, 0], 2.0)
        assert_close(result[0, 1], 3.0)
        assert_close(result[0, 2], 3.0)
        assert_close(result[1, 0], 3.0)
        assert_close(result[1, 1], 2.0)
        assert_close(result[1, 2], 4.0)


# ===== MatrixVectorProducts =====

class TestRxp:
    def test_basic(self):
        r = jnp.array([[2.0, 3.0, 2.0], [3.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        p = jnp.array([0.2, 1.5, 0.1])
        result = era.rxp(r, p)
        assert_close(result[0], 5.1)
        assert_close(result[1], 3.9)
        assert_close(result[2], 7.1)


class TestRxpv:
    def test_basic(self):
        r = jnp.array([[2.0, 3.0, 2.0], [3.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        pv = jnp.array([[0.2, 1.5, 0.1], [1.5, 0.2, 0.1]])
        result = era.rxpv(r, pv)
        assert_close(result[0, 0], 5.1)
        assert_close(result[1, 0], 3.8)
        assert_close(result[0, 1], 3.9)
        assert_close(result[1, 1], 5.2)
        assert_close(result[0, 2], 7.1)
        assert_close(result[1, 2], 5.8)


class TestTrxp:
    def test_basic(self):
        r = jnp.array([[2.0, 3.0, 2.0], [3.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        p = jnp.array([0.2, 1.5, 0.1])
        result = era.trxp(r, p)
        assert_close(result[0], 5.2)
        assert_close(result[1], 4.0)
        assert_close(result[2], 5.4)


class TestTrxpv:
    def test_basic(self):
        r = jnp.array([[2.0, 3.0, 2.0], [3.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        pv = jnp.array([[0.2, 1.5, 0.1], [1.5, 0.2, 0.1]])
        result = era.trxpv(r, pv)
        assert_close(result[0, 0], 5.2)
        assert_close(result[0, 1], 4.0)
        assert_close(result[0, 2], 5.4)
        assert_close(result[1, 0], 3.9)
        assert_close(result[1, 1], 5.3)
        assert_close(result[1, 2], 4.1)


# ===== BuildRotations =====

class TestRx:
    def test_basic(self):
        r = jnp.array([[2.0, 3.0, 2.0], [3.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        result = era.rx(0.3456789, r)
        assert_close(result[0, 0], 2.0, atol=0.0)
        assert_close(result[0, 1], 3.0, atol=0.0)
        assert_close(result[0, 2], 2.0, atol=0.0)
        assert_close(result[1, 0], 3.839043388235612460)
        assert_close(result[1, 1], 3.237033249594111899)
        assert_close(result[1, 2], 4.516714379005982719)
        assert_close(result[2, 0], 1.806030415924501684)
        assert_close(result[2, 1], 3.085711545336372503)
        assert_close(result[2, 2], 3.687721683977873065)


class TestRy:
    def test_basic(self):
        r = jnp.array([[2.0, 3.0, 2.0], [3.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        result = era.ry(0.3456789, r)
        assert_close(result[0, 0], 0.8651847818978159930)
        assert_close(result[0, 1], 1.467194920539316554)
        assert_close(result[0, 2], 0.1875137911274457342)
        assert_close(result[1, 0], 3.0)
        assert_close(result[1, 1], 2.0)
        assert_close(result[1, 2], 3.0)
        assert_close(result[2, 0], 3.500207892850427330)
        assert_close(result[2, 1], 4.779889022262298150)
        assert_close(result[2, 2], 5.381899160903798712)


class TestRz:
    def test_basic(self):
        r = jnp.array([[2.0, 3.0, 2.0], [3.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        result = era.rz(0.3456789, r)
        assert_close(result[0, 0], 2.898197754208926769)
        assert_close(result[0, 1], 3.500207892850427330)
        assert_close(result[0, 2], 2.898197754208926769)
        assert_close(result[1, 0], 2.144865911309686813)
        assert_close(result[1, 1], 0.865184781897815993)
        assert_close(result[1, 2], 2.144865911309686813)
        assert_close(result[2, 0], 3.0)
        assert_close(result[2, 1], 4.0)
        assert_close(result[2, 2], 5.0)


# ===== RotationVectors =====

class TestRm2v:
    def test_basic(self):
        r = jnp.array([
            [0.00, -0.80, -0.60],
            [0.80, -0.36,  0.48],
            [0.60,  0.48, -0.64],
        ])
        result = era.rm2v(r)
        assert_close(result[0], 0.0)
        assert_close(result[1], 1.413716694115406957)
        assert_close(result[2], -1.884955592153875943)


class TestRv2m:
    def test_basic(self):
        w = jnp.array([0.0, 1.41371669, -1.88495559])
        result = era.rv2m(w)
        assert_close(result[0, 0], -0.7071067782221119905, atol=1e-14)
        assert_close(result[0, 1], -0.5656854276809129651, atol=1e-14)
        assert_close(result[0, 2], -0.4242640700104211225, atol=1e-14)
        assert_close(result[1, 0], 0.5656854276809129651, atol=1e-14)
        assert_close(result[1, 1], -0.0925483394532274246, atol=1e-14)
        assert_close(result[1, 2], -0.8194112531408833269, atol=1e-14)
        assert_close(result[2, 0], 0.4242640700104211225, atol=1e-14)
        assert_close(result[2, 1], -0.8194112531408833269, atol=1e-14)
        assert_close(result[2, 2], 0.3854415612311154341, atol=1e-14)


# ===== SeparationAndAngle =====

class TestPap:
    def test_basic(self):
        a = jnp.array([1.0, 0.1, 0.2])
        b = jnp.array([-3.0, 1e-3, 0.2])
        result = era.pap(a, b)
        assert_close(result, 0.3671514267841113674)


class TestPas:
    def test_basic(self):
        result = era.pas(1.0, 0.1, 0.2, -1.0)
        assert_close(result, -2.724544922932270424)


class TestSepp:
    def test_basic(self):
        a = jnp.array([1.0, 0.1, 0.2])
        b = jnp.array([-3.0, 1e-3, 0.2])
        result = era.sepp(a, b)
        assert_close(result, 2.860391919024660768)


class TestSeps:
    def test_basic(self):
        result = era.seps(1.0, 0.1, 0.2, -3.0)
        assert_close(result, 2.346722016996998842, atol=1e-14)


# ===== SphericalCartesian =====

class TestC2s:
    def test_basic(self):
        p = jnp.array([100.0, -50.0, 25.0])
        theta, phi = era.c2s(p)
        assert_close(theta, -0.4636476090008061162, atol=1e-14)
        assert_close(phi, 0.2199879773954594463, atol=1e-14)


class TestP2s:
    def test_basic(self):
        p = jnp.array([100.0, -50.0, 25.0])
        theta, phi, r = era.p2s(p)
        assert_close(theta, -0.4636476090008061162)
        assert_close(phi, 0.2199879773954594463)
        assert_close(r, 114.5643923738960002, atol=1e-9)


class TestPv2s:
    def test_basic(self):
        pv = jnp.array([
            [-0.4514964673880165, 0.03093394277342585, 0.05594668105108779],
            [1.292270850663260e-5, 2.652814182060692e-6, 2.568431853930293e-6],
        ])
        theta, phi, r, td, pd, rd = era.pv2s(pv)
        assert_close(theta, 3.073185307179586515)
        assert_close(phi, 0.1229999999999999992)
        assert_close(r, 0.4559999999999999757)
        assert_close(td, -0.7800000000000000364e-5, atol=1e-16)
        assert_close(pd, 0.9010000000000001639e-5, atol=1e-16)
        assert_close(rd, -0.1229999999999999832e-4, atol=1e-16)


class TestS2c:
    def test_basic(self):
        result = era.s2c(3.0123, -0.999)
        assert_close(result[0], -0.5366267667260523906)
        assert_close(result[1], 0.0697711109765145365)
        assert_close(result[2], -0.8409302618566214041)


class TestS2p:
    def test_basic(self):
        result = era.s2p(-3.21, 0.123, 0.456)
        assert_close(result[0], -0.4514964673880165228)
        assert_close(result[1], 0.0309339427734258688)
        assert_close(result[2], 0.0559466810510877933)


class TestS2pv:
    def test_basic(self):
        result = era.s2pv(-3.21, 0.123, 0.456, -7.8e-6, 9.01e-6, -1.23e-5)
        assert_close(result[0, 0], -0.4514964673880165228)
        assert_close(result[0, 1], 0.0309339427734258688)
        assert_close(result[0, 2], 0.0559466810510877933)
        assert_close(result[1, 0], 0.1292270850663260170e-4, atol=1e-16)
        assert_close(result[1, 1], 0.2652814182060691422e-5, atol=1e-16)
        assert_close(result[1, 2], 0.2568431853930292259e-5, atol=1e-16)


# ===== Differentiability checks =====

class TestDifferentiability:
    def test_grad_pdp(self):
        a = jnp.array([2.0, 2.0, 3.0])
        b = jnp.array([1.0, 3.0, 4.0])
        grad_a = jax.grad(lambda x: era.pdp(x, b))(a)
        assert jnp.allclose(grad_a, b)

    def test_grad_pm(self):
        p = jnp.array([3.0, 4.0, 0.0])
        grad_p = jax.grad(era.pm)(p)
        assert jnp.allclose(grad_p, jnp.array([0.6, 0.8, 0.0]))

    def test_grad_sepp(self):
        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([0.0, 1.0, 0.0])
        grad_a = jax.grad(lambda x: era.sepp(x, b))(a)
        assert jnp.all(jnp.isfinite(grad_a))

    def test_grad_s2c_through_sepp(self):
        """Test grad flows through s2c -> sepp chain."""
        def angular_dist(theta1, phi1, theta2, phi2):
            a = era.s2c(theta1, phi1)
            b = era.s2c(theta2, phi2)
            return era.sepp(a, b)
        grad_fn = jax.grad(angular_dist, argnums=0)
        result = grad_fn(0.0, 0.0, 1.0, 0.5)
        assert jnp.isfinite(result)

    def test_jit_rxp(self):
        r = jnp.eye(3)
        p = jnp.array([1.0, 2.0, 3.0])
        result = jax.jit(era.rxp)(r, p)
        assert jnp.allclose(result, p)

    def test_vmap_s2c(self):
        thetas = jnp.linspace(0, 6.0, 10)
        phis = jnp.linspace(-1.0, 1.0, 10)
        result = jax.vmap(era.s2c)(thetas, phis)
        assert result.shape == (10, 3)
