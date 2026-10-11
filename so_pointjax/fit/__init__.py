"""so_pointjax.fit: fits that differentiate through the pointing."""

import jax
jax.config.update("jax_enable_x64", True)

from .joint import (
    Template, offset_quat, realised_offset, group_offsets, fit_offsets_gains,
)
