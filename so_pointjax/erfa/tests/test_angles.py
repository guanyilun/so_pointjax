"""Tests for angle functions, validated against ERFA C test suite values."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import so_pointjax.erfa as era


def assert_close(a, b, atol=1e-12):
    assert jnp.allclose(a, b, atol=atol), f"Expected {b}, got {a}"


class TestAnp:
    def test_basic(self):
        result = era.anp(-0.1)
        # -0.1 + 2*pi
        assert_close(result, 6.183185307179586477)

    def test_positive(self):
        result = era.anp(3.0)
        assert_close(result, 3.0)

    def test_large(self):
        result = era.anp(10.0)
        assert result >= 0.0
        assert result < era.D2PI


class TestAnpm:
    def test_basic(self):
        result = era.anpm(-4.0)
        assert_close(result, -4.0 + era.D2PI)

    def test_in_range(self):
        result = era.anpm(1.0)
        assert_close(result, 1.0)

    def test_near_pi(self):
        result = era.anpm(4.0)
        assert result >= -era.DPI
        assert result < era.DPI
