"""Tests for time scale conversions and sidereal time, validated against ERFA C test suite."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import so_pointjax.erfa as era


def assert_close(a, b, atol=1e-12):
    assert jnp.allclose(jnp.asarray(a), jnp.asarray(b), atol=atol), f"Expected {b}, got {a}"


# ===========================================================================
# Simple (differentiable) timescale conversions
# ===========================================================================

class TestTaitt:
    def test_basic(self):
        tt1, tt2 = era.taitt(2453750.5, 0.892482639)
        assert_close(tt1, 2453750.5)
        assert_close(tt2, 0.892855139, atol=1e-12)


class TestTttai:
    def test_basic(self):
        tai1, tai2 = era.tttai(2453750.5, 0.892482639)
        assert_close(tai1, 2453750.5)
        assert_close(tai2, 0.892110139, atol=1e-12)


class TestTaiut1:
    def test_basic(self):
        ut11, ut12 = era.taiut1(2453750.5, 0.892482639, -32.6659)
        assert_close(ut11, 2453750.5)
        assert_close(ut12, 0.8921045614537037037, atol=1e-12)


class TestUt1tai:
    def test_basic(self):
        tai1, tai2 = era.ut1tai(2453750.5, 0.892104561, -32.6659)
        assert_close(tai1, 2453750.5)
        assert_close(tai2, 0.8924826385462962963, atol=1e-12)


class TestTtut1:
    def test_basic(self):
        ut11, ut12 = era.ttut1(2453750.5, 0.892855139, 64.8499)
        assert_close(ut11, 2453750.5)
        assert_close(ut12, 0.8921045614537037037, atol=1e-12)


class TestUt1tt:
    def test_basic(self):
        tt1, tt2 = era.ut1tt(2453750.5, 0.892104561, 64.8499)
        assert_close(tt1, 2453750.5)
        assert_close(tt2, 0.8928551385462962963, atol=1e-12)


class TestTttdb:
    def test_basic(self):
        tdb1, tdb2 = era.tttdb(2453750.5, 0.892855139, -0.000201)
        assert_close(tdb1, 2453750.5)
        assert_close(tdb2, 0.8928551366736111111, atol=1e-12)


class TestTdbtt:
    def test_basic(self):
        tt1, tt2 = era.tdbtt(2453750.5, 0.892855137, -0.000201)
        assert_close(tt1, 2453750.5)
        assert_close(tt2, 0.8928551393263888889, atol=1e-12)


class TestTcgtt:
    def test_basic(self):
        tt1, tt2 = era.tcgtt(2453750.5, 0.892862531)
        assert_close(tt1, 2453750.5)
        assert_close(tt2, 0.8928551387488816828, atol=1e-12)


class TestTttcg:
    def test_basic(self):
        tcg1, tcg2 = era.tttcg(2453750.5, 0.892482639)
        assert_close(tcg1, 2453750.5)
        assert_close(tcg2, 0.8924900312508587113, atol=1e-12)


class TestTcbtdb:
    def test_basic(self):
        tdb1, tdb2 = era.tcbtdb(2453750.5, 0.893019599)
        assert_close(tdb1, 2453750.5)
        assert_close(tdb2, 0.8928551362746343397, atol=1e-12)


class TestTdbtcb:
    def test_basic(self):
        tcb1, tcb2 = era.tdbtcb(2453750.5, 0.892855137)
        assert_close(tcb1, 2453750.5)
        assert_close(tcb2, 0.8930195997253656716, atol=1e-12)


# ===========================================================================
# Non-differentiable UTC conversions
# ===========================================================================

class TestUtctai:
    def test_basic(self):
        tai1, tai2 = era.utctai(2453750.5, 0.892100694)
        assert_close(tai1, 2453750.5)
        assert_close(tai2, 0.8924826384444444444, atol=1e-12)


class TestTaiutc:
    def test_basic(self):
        utc1, utc2 = era.taiutc(2453750.5, 0.892482639)
        assert_close(utc1, 2453750.5)
        assert_close(utc2, 0.8921006945555555556, atol=1e-12)


class TestUt1utc:
    def test_basic(self):
        utc1, utc2 = era.ut1utc(2453750.5, 0.892104561, 0.3341)
        assert_close(utc1, 2453750.5)
        assert_close(utc2, 0.8921006941018518519, atol=1e-12)


class TestUtcut1:
    def test_basic(self):
        ut11, ut12 = era.utcut1(2453750.5, 0.892100694, 0.3341)
        assert_close(ut11, 2453750.5)
        assert_close(ut12, 0.8921045608981481481, atol=1e-12)


# ===========================================================================
# Earth rotation angle and sidereal time
# ===========================================================================

class TestEra00:
    def test_basic(self):
        result = era.era00(2400000.5, 54388.0)
        assert_close(result, 0.4022837240028158102)


class TestGmst00:
    def test_basic(self):
        result = era.gmst00(2400000.5, 53736.0, 2400000.5, 53736.0)
        assert_close(result, 1.754174972210740592)


class TestGmst06:
    def test_basic(self):
        result = era.gmst06(2400000.5, 53736.0, 2400000.5, 53736.0)
        assert_close(result, 1.754174971870091203)


class TestGmst82:
    def test_basic(self):
        result = era.gmst82(2400000.5, 53736.0)
        assert_close(result, 1.754174981860675096)


# ===========================================================================
# Differentiability tests
# ===========================================================================

class TestTimeDifferentiability:
    def test_grad_taitt(self):
        """TAI->TT is a constant offset, so grad w.r.t. tai2 should be 1.0."""
        grad_fn = jax.grad(lambda x: era.taitt(2453750.5, x)[1])
        g = grad_fn(0.892482639)
        assert_close(g, 1.0)

    def test_jit_era00(self):
        result = jax.jit(era.era00)(2400000.5, 54388.0)
        assert_close(result, 0.4022837240028158102)

    def test_jit_gmst82(self):
        result = jax.jit(era.gmst82)(2400000.5, 53736.0)
        assert_close(result, 1.754174981860675096)

    def test_grad_era00(self):
        """ERA is differentiable w.r.t. the date."""
        grad_fn = jax.grad(era.era00, argnums=1)
        g = grad_fn(2400000.5, 54388.0)
        # Should be nonzero (ERA changes with date)
        assert jnp.abs(g) > 0.0

    def test_vmap_gmst00(self):
        utb_vals = jnp.array([53736.0, 53737.0, 53738.0])
        results = jax.vmap(era.gmst00, in_axes=(None, 0, None, 0))(
            2400000.5, utb_vals, 2400000.5, utb_vals
        )
        assert results.shape == (3,)

    def test_roundtrip_tai_tt(self):
        """TAI -> TT -> TAI should round-trip."""
        tt1, tt2 = era.taitt(2453750.5, 0.892482639)
        tai1, tai2 = era.tttai(tt1, tt2)
        assert_close(tai1, 2453750.5)
        assert_close(tai2, 0.892482639)

    def test_roundtrip_tcg_tt(self):
        """TCG -> TT -> TCG should approximately round-trip."""
        tt1, tt2 = era.tcgtt(2453750.5, 0.892862531)
        tcg1, tcg2 = era.tttcg(tt1, tt2)
        assert_close(tcg1, 2453750.5)
        assert_close(tcg2, 0.892862531, atol=1e-9)
