"""Tests for precession, nutation, frame bias, and related rotation functions.

Validated against ERFA C test suite reference values (t_erfa_c.c).
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import so_pointjax.erfa as era


def assert_close(a, b, atol=1e-12):
    assert jnp.allclose(jnp.asarray(a), jnp.asarray(b), atol=atol), f"Expected {b}, got {a}"


def assert_mat(name, mat, expected, atol=1e-12):
    """Check a 3x3 matrix against expected values."""
    for i in range(3):
        for j in range(3):
            assert abs(float(mat[i, j]) - expected[i][j]) < atol, (
                f"{name}[{i},{j}]: got {float(mat[i,j])}, expected {expected[i][j]}"
            )


# ===========================================================================
# Obliquity
# ===========================================================================

class TestObl80:
    def test_basic(self):
        assert_close(era.obl80(2400000.5, 54388.0), 0.4090751347643816218, atol=1e-14)


class TestObl06:
    def test_basic(self):
        assert_close(era.obl06(2400000.5, 54388.0), 0.4090749229387258204, atol=1e-14)


# ===========================================================================
# Frame bias and precession rate
# ===========================================================================

class TestBi00:
    def test_basic(self):
        dpsibi, depsbi, dra = era.bi00()
        assert_close(dpsibi, -0.2025309152835086613e-6)
        assert_close(depsbi, -0.3306041454222147847e-7)
        assert_close(dra, -0.7078279744199225506e-7)


class TestPr00:
    def test_basic(self):
        dpsipr, depspr = era.pr00(2400000.5, 53736)
        assert_close(dpsipr, -0.8716465172668347629e-7, atol=1e-22)
        assert_close(depspr, -0.7342018386722813087e-8, atol=1e-22)


class TestSp00:
    def test_basic(self):
        assert_close(era.sp00(2400000.5, 52541.0), -0.6216698469981019309e-11)


# ===========================================================================
# Precession angles and matrices
# ===========================================================================

class TestPfw06:
    def test_basic(self):
        gamb, phib, psib, epsa = era.pfw06(2400000.5, 50123.9999)
        assert_close(gamb, -0.2243387670997995690e-5)
        assert_close(phib, 0.4091014602391312808)
        assert_close(psib, -0.9501954178013031895e-3)
        assert_close(epsa, 0.4091014316587367491)


class TestFw2m:
    def test_basic(self):
        r = era.fw2m(-0.2243387670997992368e-5, 0.4091014602391312982,
                     -0.9501954178013015092e-3, 0.4091014316587367472)
        assert_close(r[0, 0], 0.9999995505176007047)
        assert_close(r[0, 1], 0.8695404617348192957e-3)
        assert_close(r[0, 2], 0.3779735201865582571e-3)


class TestFw2xy:
    def test_basic(self):
        x, y = era.fw2xy(-0.2243387670997992368e-5, 0.4091014602391312982,
                          -0.9501954178013015092e-3, 0.4091014316587367472)
        assert_close(x, -0.3779734957034082790e-3)
        assert_close(y, -0.1924880848087615651e-6)


# ===========================================================================
# Nutation models
# ===========================================================================

class TestNut80:
    def test_basic(self):
        dpsi, deps = era.nut80(2400000.5, 53736.0)
        assert_close(dpsi, -0.9643658353226563966e-5, atol=1e-13)
        assert_close(deps, 0.4060051006879713322e-4, atol=1e-13)


class TestNut00b:
    def test_basic(self):
        dpsi, deps = era.nut00b(2400000.5, 53736.0)
        assert_close(dpsi, -0.9632552291148362783e-5, atol=1e-13)
        assert_close(deps, 0.4063197106621159367e-4, atol=1e-13)


class TestNut00a:
    def test_basic(self):
        dpsi, deps = era.nut00a(2400000.5, 53736.0)
        assert_close(dpsi, -0.9630909107115518431e-5, atol=1e-13)
        assert_close(deps, 0.4063239174001678710e-4, atol=1e-13)


class TestNut06a:
    def test_basic(self):
        dpsi, deps = era.nut06a(2400000.5, 53736.0)
        assert_close(dpsi, -0.9630912025820308797e-5, atol=1e-13)
        assert_close(deps, 0.4063238496887249798e-4, atol=1e-13)


# ===========================================================================
# Nutation matrices
# ===========================================================================

class TestNutm80:
    def test_basic(self):
        rmatn = era.nutm80(2400000.5, 53736.0)
        assert_close(rmatn[0, 0], 0.9999999999534999268)
        assert_close(rmatn[0, 1], 0.8847935789636432161e-5)
        assert_close(rmatn[0, 2], 0.3835906502164019142e-5)
        assert_close(rmatn[1, 0], -0.8847780042583435924e-5)
        assert_close(rmatn[2, 2], 0.9999999991684415129)


class TestNum00a:
    def test_basic(self):
        rn = era.num00a(2400000.5, 53736.0)
        assert_close(rn[0, 0], 0.9999999999536227949)
        assert_close(rn[0, 1], 0.8836238544090873336e-5)


class TestNum00b:
    def test_basic(self):
        rn = era.num00b(2400000.5, 53736.0)
        assert_close(rn[0, 0], 0.9999999999536069682)


class TestNum06a:
    def test_basic(self):
        rn = era.num06a(2400000.5, 53736.0)
        assert_close(rn[0, 0], 0.9999999999536227668)
        assert_close(rn[0, 1], 0.8836241998111535233e-5)


# ===========================================================================
# Bias-precession matrices
# ===========================================================================

class TestBp00:
    def test_basic(self):
        rb, rp, rbp = era.bp00(2400000.5, 50123.9999)
        assert_close(rb[0, 0], 0.9999999999999942498)
        assert_close(rb[0, 1], -0.7078279744199196626e-7)
        assert_close(rp[0, 0], 0.9999995504864048241)
        assert_close(rp[0, 1], 0.8696113836207084411e-3)
        assert_close(rbp[0, 0], 0.9999995505175087260)


class TestBp06:
    def test_basic(self):
        rb, rp, rbp = era.bp06(2400000.5, 50123.9999)
        assert_close(rb[0, 0], 0.9999999999999942497)
        assert_close(rp[0, 0], 0.9999995504864960278)
        assert_close(rbp[0, 0], 0.9999995505176007047)


# ===========================================================================
# Precession-nutation composites
# ===========================================================================

class TestPnm00a:
    def test_basic(self):
        rbpn = era.pnm00a(2400000.5, 50123.9999)
        assert_close(rbpn[0, 0], 0.9999995832793134257)
        assert_close(rbpn[0, 1], 0.8372384254137809439e-3)
        assert_close(rbpn[0, 2], 0.3639684306407150645e-3)


class TestPnm00b:
    def test_basic(self):
        rbpn = era.pnm00b(2400000.5, 50123.9999)
        assert_close(rbpn[0, 0], 0.9999995832776208280)
        assert_close(rbpn[0, 1], 0.8372401264429654837e-3)


class TestPnm06a:
    def test_basic(self):
        rbpn = era.pnm06a(2400000.5, 50123.9999)
        assert_close(rbpn[0, 0], 0.9999995832794205484)
        assert_close(rbpn[0, 1], 0.8372382772630962111e-3)


class TestPnm80:
    def test_basic(self):
        rmatpn = era.pnm80(2400000.5, 50123.9999)
        assert_close(rmatpn[0, 0], 0.9999995831934611169)
        assert_close(rmatpn[0, 1], 0.8373654045728124011e-3)


# ===========================================================================
# Equation of the equinoxes
# ===========================================================================

class TestEe00:
    def test_basic(self):
        ee = era.ee00(2400000.5, 53736.0, 0.4090789763356509900, -0.9630909107115582393e-5)
        assert_close(ee, -0.8834193235367965479e-5)


class TestEe00a:
    def test_basic(self):
        assert_close(era.ee00a(2400000.5, 53736.0), -0.8834192459222588227e-5)


class TestEe00b:
    def test_basic(self):
        assert_close(era.ee00b(2400000.5, 53736.0), -0.8835700060003032831e-5)


class TestEect00:
    def test_basic(self):
        assert_close(era.eect00(2400000.5, 53736.0), 0.2046085004885125264e-8)


class TestEqeq94:
    def test_basic(self):
        assert_close(era.eqeq94(2400000.5, 41234.0), 0.5357758254609256894e-4)


# ===========================================================================
# CIO locator s
# ===========================================================================

class TestS00:
    def test_basic(self):
        assert_close(era.s00(2400000.5, 53736.0, 0.5791308486706011000e-3, 0.4020579816732961219e-4),
                     -0.1220036263270905693e-7)


class TestS00a:
    def test_basic(self):
        assert_close(era.s00a(2400000.5, 52541.0), -0.1340684448919163584e-7)


class TestS00b:
    def test_basic(self):
        assert_close(era.s00b(2400000.5, 52541.0), -0.1340695782951026584e-7)


class TestS06:
    def test_basic(self):
        assert_close(era.s06(2400000.5, 53736.0, 0.5791308486706011000e-3, 0.4020579816732961219e-4),
                     -0.1220032213076463117e-7)


class TestS06a:
    def test_basic(self):
        assert_close(era.s06a(2400000.5, 52541.0), -0.1340680437291812383e-7)


# ===========================================================================
# Greenwich sidereal time
# ===========================================================================

class TestGst00a:
    def test_basic(self):
        assert_close(era.gst00a(2400000.5, 53736.0, 2400000.5, 53736.0), 1.754166138018281369)


class TestGst00b:
    def test_basic(self):
        assert_close(era.gst00b(2400000.5, 53736.0), 1.754166136510680589)


class TestGst06a:
    def test_basic(self):
        assert_close(era.gst06a(2400000.5, 53736.0, 2400000.5, 53736.0), 1.754166137675019159)


class TestGst94:
    def test_basic(self):
        assert_close(era.gst94(2400000.5, 53736.0), 1.754166136020645203)


# ===========================================================================
# Celestial-to-intermediate frame
# ===========================================================================

class TestC2i00a:
    def test_basic(self):
        rc2i = era.c2i00a(2400000.5, 53736.0)
        assert_close(rc2i[0, 0], 0.9999998323037165557)
        assert_close(rc2i[0, 2], -0.5791308477073443415e-3)
        assert_close(rc2i[2, 2], 0.9999998314954572304)


class TestC2i00b:
    def test_basic(self):
        rc2i = era.c2i00b(2400000.5, 53736.0)
        assert_close(rc2i[0, 0], 0.9999998323040954356)


class TestC2i06a:
    def test_basic(self):
        rc2i = era.c2i06a(2400000.5, 53736.0)
        assert_close(rc2i[0, 0], 0.9999998323037159379)
        assert_close(rc2i[0, 2], -0.5791308487740529749e-3)


# ===========================================================================
# Celestial-to-terrestrial matrices
# ===========================================================================

class TestC2t00a:
    def test_basic(self):
        rc2t = era.c2t00a(2400000.5, 53736.0, 2400000.5, 53736.0, 2.55060238e-7, 1.860359247e-6)
        assert_close(rc2t[0, 0], -0.1810332128307182668)
        assert_close(rc2t[0, 1], 0.9834769806938457836)
        assert_close(rc2t[2, 2], 0.9999998325501692289)


class TestC2t00b:
    def test_basic(self):
        rc2t = era.c2t00b(2400000.5, 53736.0, 2400000.5, 53736.0, 2.55060238e-7, 1.860359247e-6)
        assert_close(rc2t[0, 0], -0.1810332128439678965)
        assert_close(rc2t[2, 2], 0.9999998325505635738)


class TestC2t06a:
    def test_basic(self):
        rc2t = era.c2t06a(2400000.5, 53736.0, 2400000.5, 53736.0, 2.55060238e-7, 1.860359247e-6)
        assert_close(rc2t[0, 0], -0.1810332128305897282)
        assert_close(rc2t[2, 2], 0.9999998325501747785)


class TestC2tpe:
    def test_basic(self):
        rc2t = era.c2tpe(2400000.5, 53736.0, 2400000.5, 53736.0,
                         -0.9630909107115582393e-5, 0.4090789763356509900,
                         2.55060238e-7, 1.860359247e-6)
        assert_close(rc2t[0, 0], -0.1813677995763029394)
        assert_close(rc2t[2, 2], 0.9174875068792735362)


class TestC2txy:
    def test_basic(self):
        rc2t = era.c2txy(2400000.5, 53736.0, 2400000.5, 53736.0,
                         0.5791308486706011000e-3, 0.4020579816732961219e-4,
                         2.55060238e-7, 1.860359247e-6)
        assert_close(rc2t[0, 0], -0.1810332128306279253)
        assert_close(rc2t[2, 2], 0.9999998325501746670)


# ===========================================================================
# CIP coordinates
# ===========================================================================

class TestXys00a:
    def test_basic(self):
        x, y, s = era.xys00a(2400000.5, 53736.0)
        assert_close(x, 0.5791308472168152904e-3)
        assert_close(y, 0.4020595661591500259e-4)
        assert_close(s, -0.1220040848471549623e-7)


class TestXys00b:
    def test_basic(self):
        x, y, s = era.xys00b(2400000.5, 53736.0)
        assert_close(x, 0.5791301929950208873e-3)
        assert_close(y, 0.4020553681373720832e-4)
        assert_close(s, -0.1220027377285083189e-7)


class TestXys06a:
    def test_basic(self):
        x, y, s = era.xys06a(2400000.5, 53736.0)
        assert_close(x, 0.5791308482835292617e-3)
        assert_close(y, 0.4020580099454020310e-4)
        assert_close(s, -0.1220032294164579896e-7)


# ===========================================================================
# Differentiability and JIT tests
# ===========================================================================

class TestDifferentiability:
    def test_jit_obl06(self):
        result = jax.jit(era.obl06)(2400000.5, 54388.0)
        assert_close(result, 0.4090749229387258204, atol=1e-14)

    def test_grad_obl06(self):
        """Obliquity is differentiable w.r.t. date."""
        grad_fn = jax.grad(era.obl06, argnums=1)
        g = grad_fn(2400000.5, 54388.0)
        assert jnp.abs(g) > 0.0

    def test_jit_nut00b(self):
        dpsi, deps = jax.jit(era.nut00b)(2400000.5, 53736.0)
        assert_close(dpsi, -0.9632552291148362783e-5, atol=1e-13)

    def test_grad_nut00b(self):
        """Nutation is differentiable w.r.t. date."""
        grad_fn = jax.grad(lambda d2: era.nut00b(2400000.5, d2)[0])
        g = grad_fn(53736.0)
        assert jnp.isfinite(g)
        assert jnp.abs(g) > 0.0

    def test_jit_nut80(self):
        dpsi, deps = jax.jit(era.nut80)(2400000.5, 53736.0)
        assert_close(dpsi, -0.9643658353226563966e-5, atol=1e-13)

    def test_jit_pnm06a(self):
        rbpn = jax.jit(era.pnm06a)(2400000.5, 50123.9999)
        assert_close(rbpn[0, 0], 0.9999995832794205484)

    def test_jit_gst00a(self):
        result = jax.jit(era.gst00a)(2400000.5, 53736.0, 2400000.5, 53736.0)
        assert_close(result, 1.754166138018281369)

    def test_jit_eect00(self):
        result = jax.jit(era.eect00)(2400000.5, 53736.0)
        assert_close(result, 0.2046085004885125264e-8)

    def test_vmap_obl06(self):
        dates = jnp.array([54388.0, 54389.0, 54390.0])
        results = jax.vmap(era.obl06, in_axes=(None, 0))(2400000.5, dates)
        assert results.shape == (3,)

    def test_grad_s06(self):
        """CIO locator s is differentiable."""
        grad_fn = jax.grad(era.s06, argnums=2)
        g = grad_fn(2400000.5, 53736.0, 0.5791308486706011000e-3, 0.4020579816732961219e-4)
        assert jnp.isfinite(g)

    def test_jit_ee00a(self):
        result = jax.jit(era.ee00a)(2400000.5, 53736.0)
        assert_close(result, -0.8834192459222588227e-5)

    def test_jit_c2i06a(self):
        rc2i = jax.jit(era.c2i06a)(2400000.5, 53736.0)
        assert_close(rc2i[0, 0], 0.9999998323037159379)
