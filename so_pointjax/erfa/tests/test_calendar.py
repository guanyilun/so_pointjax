"""Tests for calendar and epoch functions, validated against ERFA C test suite values."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import so_pointjax.erfa as era


def assert_close(a, b, atol=1e-12):
    assert jnp.allclose(jnp.asarray(a), jnp.asarray(b), atol=atol), f"Expected {b}, got {a}"


# ---------------------------------------------------------------------------
# cal2jd
# ---------------------------------------------------------------------------

class TestCal2jd:
    def test_basic(self):
        djm0, djm = era.cal2jd(2003, 6, 1)
        assert_close(djm0, 2400000.5)
        assert_close(djm, 52791.0)

    def test_bad_year(self):
        with pytest.raises(ValueError, match="bad year"):
            era.cal2jd(-5000, 1, 1)

    def test_bad_month(self):
        with pytest.raises(ValueError, match="bad month"):
            era.cal2jd(2000, 13, 1)

    def test_bad_day(self):
        with pytest.raises(ValueError, match="bad day"):
            era.cal2jd(2000, 2, 30)

    def test_leap_year(self):
        # Feb 29 on a leap year should work
        djm0, djm = era.cal2jd(2000, 2, 29)
        assert djm0 == 2400000.5

    def test_non_leap_year(self):
        # Feb 29 on a non-leap year should fail
        with pytest.raises(ValueError, match="bad day"):
            era.cal2jd(1900, 2, 29)


# ---------------------------------------------------------------------------
# jd2cal
# ---------------------------------------------------------------------------

class TestJd2cal:
    def test_basic(self):
        iy, im, id, fd = era.jd2cal(2400000.5, 50123.9999)
        assert iy == 1996
        assert im == 2
        assert id == 10
        assert_close(fd, 0.9999)

    def test_roundtrip(self):
        """cal2jd -> jd2cal should round-trip."""
        djm0, djm = era.cal2jd(2023, 7, 15)
        iy, im, id, fd = era.jd2cal(djm0, djm)
        assert iy == 2023
        assert im == 7
        assert id == 15
        assert_close(fd, 0.0)

    def test_unacceptable_date(self):
        with pytest.raises(ValueError, match="unacceptable date"):
            era.jd2cal(0.0, -100000.0)


# ---------------------------------------------------------------------------
# jdcalf
# ---------------------------------------------------------------------------

class TestJdcalf:
    def test_basic(self):
        iy, im, id, ifd = era.jdcalf(4, 2400000.5, 50123.9999)
        assert iy == 1996
        assert im == 2
        assert id == 10
        assert ifd == 9999

    def test_bad_ndp(self):
        with pytest.raises(ValueError, match="bad decimal"):
            era.jdcalf(-1, 2400000.5, 50123.9999)


# ---------------------------------------------------------------------------
# epb
# ---------------------------------------------------------------------------

class TestEpb:
    def test_basic(self):
        result = era.epb(2415019.8135, 30103.18648)
        assert_close(result, 1982.418424159278580)


# ---------------------------------------------------------------------------
# epj
# ---------------------------------------------------------------------------

class TestEpj:
    def test_basic(self):
        result = era.epj(2451545.0, -7392.5)
        assert_close(result, 1979.760438056125941)


# ---------------------------------------------------------------------------
# epb2jd
# ---------------------------------------------------------------------------

class TestEpb2jd:
    def test_basic(self):
        djm0, djm = era.epb2jd(1957.3)
        assert_close(djm0, 2400000.5)
        assert_close(djm, 35948.1915101513)


# ---------------------------------------------------------------------------
# epj2jd
# ---------------------------------------------------------------------------

class TestEpj2jd:
    def test_basic(self):
        djm0, djm = era.epj2jd(1996.8)
        assert_close(djm0, 2400000.5)
        assert_close(djm, 50375.7)


# ---------------------------------------------------------------------------
# Differentiability tests for epoch functions
# ---------------------------------------------------------------------------

class TestDifferentiability:
    def test_grad_epj(self):
        """epj is linear -> gradient should be 1/DJY."""
        grad_fn = jax.grad(era.epj, argnums=1)
        g = grad_fn(2451545.0, 0.0)
        assert_close(g, 1.0 / era.DJY)

    def test_grad_epb(self):
        """epb is linear -> gradient should be 1/DTY."""
        grad_fn = jax.grad(era.epb, argnums=1)
        g = grad_fn(2451545.0, 0.0)
        assert_close(g, 1.0 / era.DTY)

    def test_jit_epj(self):
        result = jax.jit(era.epj)(2451545.0, -7392.5)
        assert_close(result, 1979.760438056125941)

    def test_jit_epb2jd(self):
        djm0, djm = jax.jit(era.epb2jd)(1957.3)
        assert_close(djm0, 2400000.5)
        assert_close(djm, 35948.1915101513)

    def test_vmap_epj(self):
        dj2_vals = jnp.array([0.0, -365.25, -730.5])
        results = jax.vmap(era.epj, in_axes=(None, 0))(2451545.0, dj2_vals)
        assert results.shape == (3,)
