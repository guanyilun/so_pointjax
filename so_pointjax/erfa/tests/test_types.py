"""Tests for JAX-compatible data types."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from so_pointjax.erfa._types import ASTROM, LDBODY


class TestASTROM:
    def test_empty(self):
        a = ASTROM.empty()
        assert a.pmt == 0.0
        assert a.eb.shape == (3,)
        assert a.bpn.shape == (3, 3)

    def test_is_pytree(self):
        """ASTROM (NamedTuple) should be a valid JAX pytree."""
        a = ASTROM.empty()
        leaves, treedef = jax.tree.flatten(a)
        restored = treedef.unflatten(leaves)
        assert jnp.allclose(restored.pmt, a.pmt)

    def test_jit_compatible(self):
        @jax.jit
        def get_pmt(astrom):
            return astrom.pmt

        a = ASTROM.empty()
        assert get_pmt(a) == 0.0


class TestLDBODY:
    def test_create(self):
        b = LDBODY(
            bm=jnp.float64(1.0),
            dl=jnp.float64(0.001),
            pv=jnp.zeros((2, 3)),
        )
        assert b.bm == 1.0
        assert b.pv.shape == (2, 3)
