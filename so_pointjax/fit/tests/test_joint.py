"""Tests for so_pointjax.fit.joint on a synthetic, asymmetric nebula."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from so_pointjax.proj import quat
from so_pointjax.fit import (
    Template, offset_quat, realised_offset, group_offsets, fit_offsets_gains,
)

ARCMIN = np.pi / 180 / 60
LON0, LAT0 = 1.46, 0.38  # roughly the Crab


def _nebula(east, north):
    """Two offset Gaussians, so gain and position are not separable."""
    s = 1.2 * ARCMIN
    return (np.exp(-(east**2 + north**2) / (2 * s**2))
            + 0.5 * np.exp(-((east - 2 * ARCMIN)**2 + (north - ARCMIN)**2)
                           / (2 * (1.5 * s)**2)))


def _template():
    pix = 0.1 * ARCMIN
    x = (np.arange(501) - 250) * pix
    E, N = np.meshgrid(x, x)
    return Template(_nebula(E, N), LON0, LAT0, pix)


def _boresight(n_rows=30, per_row=500, half=12 * ARCMIN, roll=0.3):
    """Boustrophedon raster in (east, north) around the source."""
    frac = np.tile(np.linspace(0, 1, per_row), n_rows)
    row = np.repeat(np.arange(n_rows), per_row)
    east = half * (2 * frac - 1) * np.where(row % 2 == 0, 1, -1)
    north = half * (2 * (row + 0.5) / n_rows - 1)
    return quat.rotation_lonlat(LON0 + east / np.cos(LAT0), LAT0 + north,
                                roll)


def _focal_plane():
    xi, eta = np.meshgrid(np.linspace(-3, 3, 4) * ARCMIN,
                          np.linspace(-2, 2, 3) * ARCMIN)
    xi, eta = xi.ravel(), eta.ravel()
    groups = (xi > 0).astype(int)
    return quat.rotation_xieta(xi, eta), groups


def _simulate(true_offsets, noise, seed=0):
    rng = np.random.default_rng(seed)
    tmpl = _template()
    Q = _boresight()
    q_det, groups = _focal_plane()
    n_det = len(groups)
    gains = rng.uniform(0.5, 1.5, n_det)
    baselines = rng.normal(0, 1, n_det)
    data = np.empty((n_det, len(Q)))
    for i in range(n_det):
        off = true_offsets[groups[i]]
        q = quat.qmul(Q, quat.qmul(offset_quat(off[0], off[1]), q_det[i]))
        data[i] = gains[i] * tmpl.sample(*tmpl.sky_offsets(q)) + baselines[i]
    data += noise * rng.normal(size=data.shape)
    weights = np.full(data.shape, 1 / noise**2)
    return dict(data=data, weights=weights, Q_bore=Q, det_quats=q_det,
                template=tmpl, groups=groups, gains=gains,
                baselines=baselines)


TRUE = np.array([[1.5, -0.7], [-0.9, 1.1]]) * ARCMIN


def _fit(sim):
    return fit_offsets_gains(
        sim['data'], sim['weights'], sim['Q_bore'], sim['det_quats'],
        sim['template'], jnp.zeros((2, 2)), group_offsets(sim['groups']))


class TestTemplate:
    def test_orientation(self):
        tmpl = _template()
        eps = ARCMIN
        e, n = tmpl.sky_offsets(
            quat.rotation_lonlat(LON0 + eps / np.cos(LAT0), LAT0))
        np.testing.assert_allclose([e, n], [eps, 0], atol=1e-4 * eps)
        e, n = tmpl.sky_offsets(quat.rotation_lonlat(LON0, LAT0 + eps))
        np.testing.assert_allclose([e, n], [0, eps], atol=1e-4 * eps)

    def test_sample_on_grid(self):
        tmpl = _template()
        e, n = np.array([0.3, -1.2]) * ARCMIN, np.array([0.5, 2.0]) * ARCMIN
        np.testing.assert_allclose(tmpl.sample(e, n), _nebula(e, n),
                                   rtol=1e-12)


class TestOffsetQuat:
    def test_matches_rotation_xieta(self):
        dxi, deta = 1.5 * ARCMIN, -0.7 * ARCMIN
        xi, eta = realised_offset(dxi, deta)
        q = quat.rotation_xieta(xi, eta)
        np.testing.assert_allclose(offset_quat(dxi, deta), q, atol=1e-14)

    def test_gradient_at_zero(self):
        g = jax.jacfwd(lambda p: offset_quat(p[0], p[1]))(jnp.zeros(2))
        assert np.all(np.isfinite(g))


class TestFit:
    def test_noiseless_recovery(self):
        sim = _simulate(TRUE, noise=1e-6)
        res = _fit(sim)
        assert res['converged']
        np.testing.assert_allclose(res['params'], TRUE, atol=1e-5 * ARCMIN)
        np.testing.assert_allclose(res['gains'], sim['gains'], rtol=1e-5)
        np.testing.assert_allclose(res['baselines'], sim['baselines'],
                                   atol=1e-5)

    def test_noisy_errors_consistent(self):
        sim = _simulate(TRUE, noise=0.05, seed=1)
        res = _fit(sim)
        assert res['converged']
        err = np.sqrt(np.diag(res['cov']))
        pull = (np.ravel(res['params']) - TRUE.ravel()) / err
        assert np.all(np.abs(pull) < 5), pull
        gpull = (res['gains'] - sim['gains']) / res['gain_err']
        assert np.all(np.abs(gpull) < 5), gpull
        dof = sim['data'].size - 2 * len(sim['gains']) - 4
        assert abs(res['chi2'] / dof - 1) < 0.05
