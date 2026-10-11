"""Joint fit of pointing and linear amplitudes against sky components.

The model for detector i is d_i(t) = sum_k a_k m_k(t) S_k(q_i(t; p)), with
sky components S_k, optional modulations m_k, linear amplitudes a_k and
pointing parameters p entering through

    q_i(t) = Q_bore(t) * q_off(p, i, t) * q_det_i.

'det' and 'segment' amplitudes are solved in closed form per detector for
each trial p (variable projection); 'global' amplitudes join p in the
Levenberg-Marquardt vector.
"""

from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.ndimage import map_coordinates

from so_pointjax.proj import quat

SCOPES = ('det', 'segment', 'global')


@jax.tree_util.register_pytree_node_class
class Template:
    """Map on a flat grid centred on (lon0, lat0), usable as a sky component.

    ``data[iy, ix]`` is the value at east = x0 + ix*pix, north = y0 + iy*pix
    (radians), in the orthographic projection about (lon0, lat0).  The default
    x0, y0 put the centre of the grid on (lon0, lat0).  Sampling is bilinear
    and zero outside the grid.  The template should already carry the beam
    of the data being fitted.  Calling it on pointing quaternions samples it
    at their positions.
    """

    def __init__(self, data, lon0, lat0, pix, x0=None, y0=None):
        self.data = jnp.asarray(data, dtype=jnp.float64)
        ny, nx = self.data.shape
        self.lon0, self.lat0, self.pix = lon0, lat0, pix
        self.x0 = -(nx - 1) / 2 * pix if x0 is None else x0
        self.y0 = -(ny - 1) / 2 * pix if y0 is None else y0

    def tree_flatten(self):
        return (self.data,), (self.lon0, self.lat0, self.pix, self.x0, self.y0)

    @classmethod
    def tree_unflatten(cls, aux, children):
        lon0, lat0, pix, x0, y0 = aux
        return cls(children[0], lon0, lat0, pix, x0, y0)

    def sky_offsets(self, q):
        """(east, north) offsets from the centre of pointing quaternions q."""
        q_center = quat.rotation_lonlat(self.lon0, self.lat0)
        xi, eta, _ = quat.decompose_xieta(quat.qmul(quat.qconj(q_center), q))
        return -xi, eta

    def sample(self, east, north):
        coords = [(north - self.y0) / self.pix, (east - self.x0) / self.pix]
        return map_coordinates(self.data, coords, order=1, mode='constant',
                               cval=0.0)

    def __call__(self, q):
        return self.sample(*self.sky_offsets(q))


class Component(NamedTuple):
    """One linear term of the model.

    sky : callable q -> values, or None
        Evaluated on the detector's sky pointing quaternions (n_samp, 4),
        e.g. a `Template`.  None means a constant 1.
    scope : 'det', 'segment' or 'global'
    modulation : array (n_samp,) or (n_det, n_samp), optional
        Time weighting multiplied into the component.
    """
    name: str
    sky: Optional[Callable] = None
    scope: str = 'det'
    modulation: Any = None


def offset_quat(dxi, deta):
    """Focal-plane rotation that moves the boresight by (dxi, deta).

    Exact to first order; the realised offset is (dxi, deta) / (1 + r^2/4)
    with r^2 = dxi^2 + deta^2, which `realised_offset` recovers.  Unlike
    quat.rotation_xieta this is smooth at zero, so it can be differentiated
    there.
    """
    q = jnp.stack([jnp.ones_like(dxi), dxi / 2, -deta / 2,
                   jnp.zeros_like(dxi)], axis=-1)
    return quat.qnormalize(q)


def realised_offset(dxi, deta):
    """(xi, eta) offset that offset_quat(dxi, deta) applies."""
    xi, eta, _ = quat.decompose_xieta(offset_quat(dxi, deta))
    return xi, eta


def group_offsets(groups):
    """offset_fn for one (xi, eta) offset per detector group.

    ``groups[i]`` is the group index of detector i; the parameters are an
    array of shape (n_groups, 2).
    """
    groups = jnp.asarray(groups)

    def offset_fn(params, i):
        return params[groups[i]]
    return offset_fn


_RIDGE = 1e-12


def _scales(M):
    diag = jnp.diagonal(M, axis1=-2, axis2=-1)
    empty = diag <= 0
    return jnp.sqrt(jnp.where(empty, 1.0, diag)), empty


def _solve_local(Cd, Cs, seg, n_seg, w, Y):
    """Weighted least squares for one detector, for several right-hand sides.

    Model Cd @ a_d + Cs[t] @ a_seg[seg[t]], solved by eliminating the
    segment blocks and then the Schur complement.  Empty columns get
    amplitude 0 and NaN variance.  Returns sqrt(w) (Y - model), a_d, a_seg
    and their variances.
    """
    n_s, nd = Cd.shape
    ns, m = Cs.shape[1], Y.shape[1]
    model = jnp.zeros((n_s, m))
    if nd:
        D = (Cd * w[:, None]).T @ Cd
        sd, empty_d = _scales(D)
        S = D / jnp.outer(sd, sd) + _RIDGE * jnp.eye(nd)
        rhs = Cd.T @ (w[:, None] * Y) / sd[:, None]
    if ns:
        ok = seg >= 0
        lab = jnp.where(ok, seg, n_seg)
        wc = (w * ok)[:, None] * Cs
        M = jax.ops.segment_sum(wc[:, :, None] * Cs[:, None, :], lab, n_seg)
        sk, empty_k = _scales(M)
        M_inv = jnp.linalg.inv(M / (sk[:, :, None] * sk[:, None, :])
                               + _RIDGE * jnp.eye(ns))
        Yk = jax.ops.segment_sum(wc[:, :, None] * Y[:, None, :], lab, n_seg)
        MinvY = M_inv @ (Yk / sk[:, :, None])
    if nd and ns:
        B = jax.ops.segment_sum(wc[:, None, :] * Cd[:, :, None], lab, n_seg)
        B = B / (sd[None, :, None] * sk[:, None, :])
        MinvBt = M_inv @ jnp.swapaxes(B, 1, 2)
        S = S - jnp.einsum('kds,kse->de', B, MinvBt)
        rhs = rhs - jnp.einsum('kds,ksm->dm', B, MinvY)
    if nd:
        S_inv = jnp.linalg.inv(S)
        ad_n = S_inv @ rhs
        ad = ad_n / sd[:, None]
        vd = jnp.where(empty_d, jnp.nan, jnp.diag(S_inv) / sd**2)
        model = model + Cd @ ad
    else:
        ad, vd = jnp.zeros((0, m)), jnp.zeros(0)
    if ns:
        if nd:
            ak_n = MinvY - MinvBt @ ad_n
            V = M_inv + MinvBt @ S_inv @ jnp.swapaxes(MinvBt, 1, 2)
        else:
            ak_n, V = MinvY, M_inv
        ak = ak_n / sk[:, :, None]
        vk = jnp.where(empty_k, jnp.nan,
                       jnp.diagonal(V, axis1=1, axis2=2) / sk**2)
        model = model + ok[:, None] * jnp.einsum(
            'ts,tsm->tm', Cs, ak[jnp.clip(seg, 0)])
    else:
        ak, vk = jnp.zeros((n_seg, 0, m)), jnp.zeros((n_seg, 0))
    return jnp.sqrt(w)[:, None] * (Y - model), ad, ak, vd, vk


def _make_normal_equations(data, weights, Q_bore, det_quats, components,
                           offset_fn, unravel, n_point, segments, n_seg,
                           filt):
    data = jnp.asarray(data, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    det_quats = jnp.asarray(det_quats, dtype=jnp.float64)
    n_det, n_samp = data.shape
    Q_bore = jnp.asarray(Q_bore, dtype=jnp.float64)
    bore = (lambda i: Q_bore[i]) if Q_bore.ndim == 3 else (lambda i: Q_bore)
    mods = []
    for c in components:
        if c.modulation is None:
            mods.append(None)
        else:
            m = jnp.asarray(c.modulation, dtype=jnp.float64)
            mods.append((lambda i, m=m: m[i]) if m.ndim == 2
                        else (lambda i, m=m: m))
    segments = (jnp.full(data.shape, -1) if segments is None
                else jnp.asarray(segments))
    by_scope = {s: [k for k, c in enumerate(components) if c.scope == s]
                for s in SCOPES}

    def columns(point, i):
        off = offset_fn(unravel(point), i)
        q_det = quat.qmul(offset_quat(off[..., 0], off[..., 1]), det_quats[i])
        q = quat.qmul(bore(i), q_det)
        cols = []
        for c, mod in zip(components, mods):
            col = jnp.ones(n_samp) if c.sky is None else c.sky(q)
            if mod is not None:
                col = col * mod(i)
            cols.append(col if filt is None else filt(col))
        return [jnp.stack([cols[k] for k in by_scope[s]], axis=1)
                if by_scope[s] else jnp.zeros((n_samp, 0)) for s in SCOPES]

    def solve(point, i):
        Cd, Cs, G = columns(point, i)
        Y = jnp.concatenate([data[i][:, None], G], axis=1)
        return _solve_local(Cd, Cs, segments[i], n_seg, weights[i], Y)

    def det_terms(flat, i):
        point, b = flat[:n_point], flat[n_point:]

        def resid(pt):
            out = solve(pt, i)
            return out[0][:, 0] - out[0][:, 1:] @ b, out

        if n_point:
            Jp, (R, ad, ak, vd, vk) = jax.jacfwd(resid, has_aux=True)(point)
        else:
            R, ad, ak, vd, vk = solve(point, i)
            Jp = jnp.zeros((n_samp, 0))
        r = R[:, 0] - R[:, 1:] @ b
        J = jnp.concatenate([Jp, -R[:, 1:]], axis=1)
        return (r @ r, J.T @ J, J.T @ r, ad[:, 0] - ad[:, 1:] @ b,
                ak[..., 0] - ak[..., 1:] @ b, vd, vk)

    @jax.jit
    def normal_equations(flat):
        chi2, JtJ, Jtr, ad, ak, vd, vk = jax.lax.map(
            lambda i: det_terms(flat, i), jnp.arange(n_det))
        return chi2.sum(), JtJ.sum(0), Jtr.sum(0), (ad, ak, vd, vk)

    @jax.jit
    def residuals(flat):
        def one(i):
            R = solve(flat[:n_point], i)[0]
            return R[:, 0] - R[:, 1:] @ flat[n_point:]
        return jax.lax.map(one, jnp.arange(n_det))

    return normal_equations, residuals


def fit_joint(data, weights, Q_bore, det_quats, components, params0,
              offset_fn, segments=None, filt=None, max_iter=50, rtol=1e-10,
              lam=1e-3):
    """Fit pointing parameters and the amplitudes of linear sky components.

    Parameters
    ----------
    data : array (n_det, n_samp)
        Detector samples, already filtered with ``filt`` if one is given.
        May be each detector's own selection of samples, padded with zero
        weight to a common length.
    weights : array (n_det, n_samp)
        Inverse noise variance per sample; zero excludes a sample.
    Q_bore : array (n_samp, 4) or (n_det, n_samp, 4)
        Boresight quaternions, shared or per detector (for per-detector
        sample selections).
    det_quats : array (n_det, 4)
        Detector quaternions, e.g. FocalPlane.from_xieta(...).quats.
    components : list of Component
    params0 : pytree
        Starting pointing parameters, in the form ``offset_fn`` expects.
    offset_fn : callable (params, i) -> array (2,) or (n_samp, 2)
        (xi, eta) offset in radians for detector i, e.g. `group_offsets`.
    segments : int array (n_det, n_samp), optional
        Segment label of each sample, 0 .. n_seg-1, or -1 for none.  Needed
        when a component has scope 'segment'.
    filt : callable (n_samp,) -> (n_samp,), optional
        Linear filter applied to each model column, matching the data's.
        Only meaningful when the samples are contiguous.

    Returns
    -------
    dict with
        params : fitted pointing parameters, same structure as params0
        globals, globals_err : {name: value} for 'global' components
        amplitudes, amplitude_err : {name: array} for 'det' components,
            shape (n_det,), and 'segment' components, shape (n_det, n_seg);
            NaN error marks an amplitude with no data
        cov : covariance of the flat parameter vector, pointing first
            (flattened params0) then the global amplitudes in component
            order; local amplitudes are marginalized.  Scale by chi2/dof if
            the weights are not true inverse variances.
        n_pointing : length of the pointing part of the flat vector
        residuals : array (n_det, n_samp), sqrt(weights) * (data - model)
            at the solution
        chi2, n_iter, converged
    """
    for c in components:
        if c.scope not in SCOPES:
            raise ValueError(f"component {c.name!r}: scope must be one of "
                             f"{SCOPES}, not {c.scope!r}")
    if any(c.scope == 'segment' for c in components) and segments is None:
        raise ValueError("'segment' components need segments")
    n_seg = int(np.max(segments)) + 1 if segments is not None else 0

    point0, unravel = ravel_pytree(params0)
    n_point = len(point0)
    glob = [c for c in components if c.scope == 'global']
    flat = jnp.concatenate([jnp.asarray(point0, dtype=jnp.float64),
                            jnp.zeros(len(glob))])
    normal_equations, residuals = _make_normal_equations(
        data, weights, Q_bore, det_quats, components, offset_fn, unravel,
        n_point, segments, n_seg, filt)

    chi2, JtJ, Jtr, a = normal_equations(flat)
    converged = False
    for n_iter in range(1, max_iter + 1):
        A = JtJ + lam * jnp.diag(jnp.diag(JtJ))
        trial = flat - jnp.linalg.solve(A, Jtr)
        chi2_t, JtJ_t, Jtr_t, a_t = normal_equations(trial)
        if chi2_t < chi2:
            done = (chi2 - chi2_t) <= rtol * chi2
            flat, chi2, JtJ, Jtr, a = trial, chi2_t, JtJ_t, Jtr_t, a_t
            lam = max(lam / 10, 1e-12)
            if done:
                converged = True
                break
        else:
            lam *= 10
            if lam > 1e12:
                converged = True
                break

    cov = np.asarray(jnp.linalg.inv(JtJ))
    ad, ak, vd, vk = (np.asarray(x) for x in a)
    amps, amp_err = {}, {}
    det = [c for c in components if c.scope == 'det']
    seg = [c for c in components if c.scope == 'segment']
    for j, c in enumerate(det):
        amps[c.name], amp_err[c.name] = ad[:, j], np.sqrt(vd[:, j])
    for j, c in enumerate(seg):
        amps[c.name], amp_err[c.name] = ak[:, :, j], np.sqrt(vk[:, :, j])
    g = np.asarray(flat[n_point:])
    g_err = np.sqrt(np.diag(cov)[n_point:])
    return dict(params=unravel(flat[:n_point]),
                globals={c.name: g[k] for k, c in enumerate(glob)},
                globals_err={c.name: g_err[k] for k, c in enumerate(glob)},
                amplitudes=amps, amplitude_err=amp_err, cov=cov,
                n_pointing=n_point, residuals=np.asarray(residuals(flat)),
                chi2=float(chi2), n_iter=n_iter, converged=converged)


def fit_offsets_gains(data, weights, Q_bore, det_quats, template, params0,
                      offset_fn, filt=None, **kwargs):
    """`fit_joint` with one template, a gain and a baseline per detector.

    Returns the `fit_joint` result plus ``gains``, ``gain_err`` and
    ``baselines`` per detector.
    """
    res = fit_joint(data, weights, Q_bore, det_quats,
                    [Component('gain', template), Component('baseline')],
                    params0, offset_fn, filt=filt, **kwargs)
    res.update(gains=res['amplitudes']['gain'],
               gain_err=res['amplitude_err']['gain'],
               baselines=res['amplitudes']['baseline'])
    return res
