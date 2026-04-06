"""Tests for ephem.py: moon98, plan94."""

import jax
import jax.numpy as jnp
import pytest

import so_pointjax.erfa


class TestMoon98:
    """Test eraMoon98 — reference from t_erfa_c.c."""

    def test_moon98(self):
        pv = so_pointjax.erfa.moon98(2400000.5, 43999.9)
        assert pv.shape == (2, 3)
        assert jnp.allclose(pv[0, 0], -0.2601295959971044180e-2, atol=1e-11)
        assert jnp.allclose(pv[0, 1],  0.6139750944302742189e-3, atol=1e-11)
        assert jnp.allclose(pv[0, 2],  0.2640794528229828909e-3, atol=1e-11)
        assert jnp.allclose(pv[1, 0], -0.1244321506649895021e-3, atol=1e-11)
        assert jnp.allclose(pv[1, 1], -0.5219076942678119398e-3, atol=1e-11)
        assert jnp.allclose(pv[1, 2], -0.1716132214378462047e-3, atol=1e-11)

    def test_moon98_jit(self):
        pv = jax.jit(so_pointjax.erfa.moon98)(2400000.5, 43999.9)
        assert jnp.allclose(pv[0, 0], -0.2601295959971044180e-2, atol=1e-11)

    def test_moon98_differentiable(self):
        def f(d2):
            pv = so_pointjax.erfa.moon98(2400000.5, d2)
            return pv[0, 0]
        g = jax.grad(f)(43999.9)
        assert jnp.isfinite(g)


class TestPlan94:
    """Test eraPlan94 — reference from t_erfa_c.c."""

    def test_invalid_planet_low(self):
        pv, j = so_pointjax.erfa.plan94(2400000.5, 1e6, 0)
        assert int(j) == -1
        assert jnp.allclose(pv, jnp.zeros((2, 3)))

    def test_invalid_planet_high(self):
        pv, j = so_pointjax.erfa.plan94(2400000.5, 1e6, 10)
        assert int(j) == -1

    def test_emb_remote_date(self):
        """EMB at remote date (year warning expected)."""
        pv, j = so_pointjax.erfa.plan94(2400000.5, -320000, 3)
        assert jnp.allclose(pv[0, 0],  0.9308038666832975759, atol=1e-11)
        assert jnp.allclose(pv[0, 1],  0.3258319040261346000, atol=1e-11)
        assert jnp.allclose(pv[0, 2],  0.1422794544481140560, atol=1e-11)
        assert jnp.allclose(pv[1, 0], -0.6429458958255170006e-2, atol=1e-11)
        assert jnp.allclose(pv[1, 1],  0.1468570657704237764e-1, atol=1e-11)
        assert jnp.allclose(pv[1, 2],  0.6406996426270981189e-2, atol=1e-11)
        assert int(j) == 1

    def test_mercury_normal(self):
        """Mercury at normal date."""
        pv, j = so_pointjax.erfa.plan94(2400000.5, 43999.9, 1)
        assert jnp.allclose(pv[0, 0],  0.2945293959257430832, atol=1e-11)
        assert jnp.allclose(pv[0, 1], -0.2452204176601049596, atol=1e-11)
        assert jnp.allclose(pv[0, 2], -0.1615427700571978153, atol=1e-11)
        assert jnp.allclose(pv[1, 0],  0.1413867871404614441e-1, atol=1e-11)
        assert jnp.allclose(pv[1, 1],  0.1946548301104706582e-1, atol=1e-11)
        assert jnp.allclose(pv[1, 2],  0.8929809783898904786e-2, atol=1e-11)
        assert int(j) == 0

    def test_plan94_jit(self):
        pv, j = jax.jit(so_pointjax.erfa.plan94, static_argnums=(2,))(2400000.5, 43999.9, 1)
        assert jnp.allclose(pv[0, 0], 0.2945293959257430832, atol=1e-11)

    def test_plan94_differentiable(self):
        def f(d2):
            pv, _ = so_pointjax.erfa.plan94(2400000.5, d2, 1)
            return pv[0, 0]
        g = jax.grad(f)(43999.9)
        assert jnp.isfinite(g)
