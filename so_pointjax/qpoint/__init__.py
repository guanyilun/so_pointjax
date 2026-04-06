"""Differentiable QPoint: telescope pointing library in JAX."""

import jax
jax.config.update("jax_enable_x64", True)

# Quaternion algebra
from so_pointjax.qpoint._quaternion import (  # noqa: F401
    identity, mul, conj, norm, norm2, normalize, inv,
    r1, r2, r3, r1_mul, r2_mul, r3_mul, rot,
    to_matrix, to_col1, to_col2, to_col3,
    quat2radecpa, radecpa2quat, quat2radec, radec2quat,
    slerp,
)

# Time utilities
from so_pointjax.qpoint._time_utils import (  # noqa: F401
    ctime2jd, jd2ctime, ctime2jdtt, jdutc2jdut1, ctime2gmst,
)

# Correction functions
from so_pointjax.qpoint._corrections import (  # noqa: F401
    npb_quat, erot_quat, wobble_quat, lonlat_quat,
    azel_quat, azelpsi_quat,
    refraction, refraction_quat,
    aberration, earth_orbital_beta, diurnal_aberration_beta,
    det_offset_quat, hwp_quat,
)

# Pointing pipeline
from so_pointjax.qpoint._pointing import (  # noqa: F401
    azelpsi2bore, azel2bore,
    azelpsi2bore_jit, azelpsi2bore_fast, radec2azel_jit,
    bore2radecpa, bore2radec,
    azel2radecpa,
    radec2azel,
    precompute_times, precompute_corrections,
)

# HEALPix pixelization
from so_pointjax.qpoint._pixel import (  # noqa: F401
    ang2pix_nest, ang2pix_ring,
    pix2ang_nest, pix2ang_ring,
    vec2pix_nest, vec2pix_ring,
    pix2vec_nest, pix2vec_ring,
    nest2ring, ring2nest,
    nside2npix, npix2nside,
    radec2pix, pix2radec,
    quat2pix, bore2pix,
)

# IERS Bulletin A
from so_pointjax.qpoint._iers import (  # noqa: F401
    load_bulletin_a, update_bulletin_a, interpolate_bulletin_a,
)

# State management and high-level API
from so_pointjax.qpoint._state import QPointState, QPoint  # noqa: F401
