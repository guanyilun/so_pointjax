"""Tests for so_pointjax.fit.joint on a synthetic, asymmetric nebula."""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from so_pointjax.proj import quat
from so_pointjax.fit import (
    Template, Component, offset_quat, realised_offset, group_offsets,
    fit_joint, fit_offsets_gains,
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


def _point_source(east=-2 * ARCMIN, north=-1.5 * ARCMIN):
    """Narrow Gaussian away from the nebula's centre."""
    pix = 0.1 * ARCMIN
    x = (np.arange(501) - 250) * pix
    E, N = np.meshgrid(x, x)
    s = 0.6 * ARCMIN
    return Template(np.exp(-((E - east)**2 + (N - north)**2) / (2 * s**2)),
                    LON0, LAT0, pix)


def _sky(true_offsets, sky):
    """Each detector's true sky pointing, and ``sky`` sampled along it."""
    Q = _boresight()
    q_det, groups = _focal_plane()
    S = np.empty((len(groups), len(Q)))
    for i in range(len(groups)):
        off = true_offsets[groups[i]]
        S[i] = sky(quat.qmul(Q, quat.qmul(offset_quat(off[0], off[1]),
                                          q_det[i])))
    return Q, q_det, groups, S


def _simulate(true_offsets, noise, seed=0):
    rng = np.random.default_rng(seed)
    tmpl = _template()
    Q, q_det, groups, S = _sky(true_offsets, tmpl)
    n_det = len(groups)
    gains = rng.uniform(0.5, 1.5, n_det)
    baselines = rng.normal(0, 1, n_det)
    data = gains[:, None] * S + baselines[:, None]
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
        assert res['residuals'].shape == sim['data'].shape
        np.testing.assert_allclose(np.sum(res['residuals']**2), res['chi2'],
                                   rtol=1e-8)

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


class TestSegments:
    def test_segment_gains_and_offsets(self):
        rng = np.random.default_rng(2)
        tmpl = _template()
        Q, q_det, groups, S = _sky(TRUE, tmpl)
        n_det, n_samp = S.shape
        # One segment per raster row, kept only where the detector sees the
        # nebula; samples outside every segment get zero weight.
        row = np.repeat(np.arange(30), n_samp // 30)
        seg = np.where(S > 0.02, row[None], -1)
        g = rng.uniform(0.5, 1.5, (n_det, 30))
        b = rng.normal(0, 1, (n_det, 30))
        lab = np.maximum(seg, 0)
        data = np.take_along_axis(g, lab, 1) * S + np.take_along_axis(b, lab, 1)
        weights = np.where(seg >= 0, 1e12, 0.)
        res = fit_joint(
            data, weights, Q, q_det,
            [Component('gain', tmpl, 'segment'), Component('offset', None,
                                                           'segment')],
            jnp.zeros((2, 2)), group_offsets(groups), segments=seg)
        assert res['converged']
        np.testing.assert_allclose(res['params'], TRUE, atol=1e-5 * ARCMIN)
        has = np.isfinite(res['amplitude_err']['gain'])
        assert has.sum() > 50
        g = g[:, :has.shape[1]]  # labels above the highest used are absent
        np.testing.assert_allclose(res['amplitudes']['gain'][has], g[has],
                                   rtol=1e-4)
        assert np.all(np.isnan(res['amplitude_err']['gain'][~has]))


class TestGlobal:
    def _sim(self, noise, seed):
        rng = np.random.default_rng(seed)
        tmpl, src = _template(), _point_source()
        Q, q_det, groups, S = _sky(TRUE, tmpl)
        P = _sky(TRUE, src)[3]
        n_det, n_samp = S.shape
        gains = rng.uniform(0.5, 1.5, n_det)
        # Four phase bins of a periodic signal; the source amplitude per bin
        # is shared by all detectors and scaled by each detector's gain.
        phase_bin = (np.floor(np.arange(n_samp) * 0.37) % 4).astype(int)
        amps = np.array([0.3, 0.0, 0.1, 0.2])
        data = gains[:, None] * (S + amps[phase_bin][None] * P)
        data += rng.normal(0, 1, n_det)[:, None]
        data += noise * rng.normal(size=data.shape)
        comps = [Component('gain', tmpl), Component('baseline')] + [
            Component(f'bin{k}', src, 'global',
                      gains[:, None] * (phase_bin == k)[None])
            for k in range(4)]
        res = fit_joint(data, np.full(data.shape, 1 / noise**2), Q, q_det,
                        comps, jnp.zeros((2, 2)), group_offsets(groups))
        return res, amps

    def test_noiseless(self):
        res, amps = self._sim(1e-6, 3)
        assert res['converged']
        np.testing.assert_allclose(res['params'], TRUE, atol=1e-5 * ARCMIN)
        got = [res['globals'][f'bin{k}'] for k in range(4)]
        np.testing.assert_allclose(got, amps, atol=1e-5)

    def test_noisy_errors_consistent(self):
        res, amps = self._sim(0.05, 4)
        got = np.array([res['globals'][f'bin{k}'] for k in range(4)])
        err = np.array([res['globals_err'][f'bin{k}'] for k in range(4)])
        assert np.all(np.abs((got - amps) / err) < 5), (got - amps) / err
        assert res['cov'].shape == (8, 8)


class TestSelectedSamples:
    def test_per_detector_selection(self):
        sim = _simulate(TRUE, noise=1e-6)
        S = _sky(TRUE, sim['template'])[3]
        keep = S > 0.01
        n_max = keep.sum(1).max()
        n_det = len(S)
        idx = np.zeros((n_det, n_max), int)
        w = np.zeros((n_det, n_max))
        for i in range(n_det):
            k = np.flatnonzero(keep[i])
            idx[i, :len(k)] = k
            w[i, :len(k)] = sim['weights'][i, k]
        data = np.take_along_axis(sim['data'], idx, 1)
        Q = np.asarray(sim['Q_bore'])[idx]
        res = fit_offsets_gains(data, w, Q, sim['det_quats'],
                                sim['template'], jnp.zeros((2, 2)),
                                group_offsets(sim['groups']))
        assert res['converged']
        assert n_max < 0.5 * S.shape[1]
        np.testing.assert_allclose(res['params'], TRUE, atol=1e-5 * ARCMIN)
        np.testing.assert_allclose(res['gains'], sim['gains'], rtol=1e-5)


class TestBlockSolve:
    def test_matches_dense(self):
        from so_pointjax.fit.joint import _solve_local
        rng = np.random.default_rng(5)
        n_s, nd, ns, n_seg, m = 300, 2, 3, 6, 4
        Cd = rng.normal(size=(n_s, nd))
        Cs = rng.normal(size=(n_s, ns))
        seg = rng.integers(-1, n_seg - 1, n_s)  # label n_seg-1 stays empty
        w = rng.uniform(0.5, 2, n_s)
        Y = rng.normal(size=(n_s, m))
        R, ad, ak, vd, vk = _solve_local(jnp.asarray(Cd), jnp.asarray(Cs),
                                         jnp.asarray(seg), n_seg,
                                         jnp.asarray(w), jnp.asarray(Y))
        R, ad, ak, vd, vk = (np.asarray(x) for x in (R, ad, ak, vd, vk))
        used = list(range(n_seg - 1))
        A = np.concatenate([Cd] + [Cs * (seg == k)[:, None] for k in used], 1)
        M = (A * w[:, None]).T @ A
        x = np.linalg.solve(M, (A * w[:, None]).T @ Y)
        np.testing.assert_allclose(ad, x[:nd], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(ak[used].reshape(-1, m), x[nd:],
                                   rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(R, np.sqrt(w)[:, None] * (Y - A @ x),
                                   rtol=1e-8, atol=1e-10)
        V = np.diag(np.linalg.inv(M))
        np.testing.assert_allclose(vd, V[:nd], rtol=1e-6)
        np.testing.assert_allclose(vk[used].ravel(), V[nd:], rtol=1e-6)
        assert np.all(np.isnan(vk[n_seg - 1]))
