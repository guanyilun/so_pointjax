"""Ensure JAX 64-bit mode is enabled before any test imports."""
import jax
jax.config.update("jax_enable_x64", True)
