"""Joint fit of pointing and linear amplitudes against sky components.

The model for detector i is

    d_i(t) = sum_k  a_k * m_k(t) * S_k(q_i(t; p)),

where S_k is a sky component evaluated at the detector's sky pointing q_i,
m_k is an optional time modulation, a_k is a linear amplitude and p holds the
pointing parameters.  Each component has a scope saying how its amplitude is
shared:

    'det'      one amplitude per detector (e.g. a gain, or a baseline when
               S_k is None, meaning a constant)
    'segment'  one per detector per labelled segment of samples (e.g. per
               source crossing)
    'global'   one shared by all detectors

For each trial p, the 'det' and 'segment' amplitudes are solved in closed
form detector by detector (variable projection).  The 'global' amplitudes
join p in the Levenberg-Marquardt parameter vector: they enter linearly, so
Gauss-Newton solves them exactly, and keeping them there leaves the
detectors independent.  The Jacobian of the profiled residuals comes from
jax.jacfwd through the offset rotation, detector quaternion, boresight, the
sky components, an optional linear filter and the linear solve.

Pointing offsets are small rotations in the focal plane, applied between the
boresight and the detector:

    q_i(t) = Q_bore(t) * q_off(p, i, t) * q_det_i.

How p maps to offsets is set by an ``offset_fn``.  `group_offsets` gives one
(xi, eta) offset per detector group (e.g. per wafer, or one for the array);
returning per-sample offsets allows time-dependent terms.

Samples can be the full time stream or, faster, only the samples each
detector spends near the source: pass per-detector sample arrays padded to a
common length with zero weight, and a per-detector boresight.
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
        # xi grows toward decreasing longitude (west); eta toward north.
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


def _solve_local(A, w, y):
    """Weighted least squares y ~ A a; returns a, var(a), residual.

    Columns are normalized before solving, and a tiny ridge keeps empty or
    degenerate columns (e.g. a segment with no samples) from making the
    system singular; their amplitudes come out as 0 with NaN variance.
    """
    M = (A * w[:, None]).T @ A
    diag = jnp.diag(M)
    empty = diag <= 0
    scale = jnp.where(empty, 1.0, jnp.sqrt(jnp.where(empty, 1.0, diag)))
    Mn = M / jnp.outer(scale, scale) + 1e-12 * jnp.eye(len(diag))
    a = jnp.linalg.solve(Mn, (A * w[:, None]).T @ y / scale) / scale
    var = jnp.where(empty, jnp.nan, jnp.diag(jnp.linalg.inv(Mn)) / scale**2)
    return a, var, jnp.sqrt(w) * (y - A @ a)


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
    if segments is not None:
        segments = jnp.asarray(segments)
    local = [k for k, c in enumerate(components) if c.scope != 'global']
    glob = [k for k, c in enumerate(components) if c.scope == 'global']

    def columns(flat, i):
        off = offset_fn(unravel(flat[:n_point]), i)
        q_det = quat.qmul(offset_quat(off[..., 0], off[..., 1]), det_quats[i])
        q = quat.qmul(bore(i), q_det)
        cols = []
        for c, mod in zip(components, mods):
            col = jnp.ones(n_samp) if c.sky is None else c.sky(q)
            if mod is not None:
                col = col * mod(i)
            cols.append(col if filt is None else filt(col))
        A = []
        for k in local:
            if components[k].scope == 'det':
                A.append(cols[k][:, None])
            else:
                onehot = segments[i][:, None] == jnp.arange(n_seg)[None]
                A.append(cols[k][:, None] * onehot)
        A = jnp.concatenate(A, axis=1) if A else jnp.zeros((n_samp, 0))
        G = jnp.stack([cols[k] for k in glob], axis=1) if glob \
            else jnp.zeros((n_samp, 0))
        return A, G

    def solve(flat, i):
        A, G = columns(flat, i)
        y = data[i] - G @ flat[n_point:]
        if A.shape[1] == 0:
            return jnp.zeros(0), jnp.zeros(0), jnp.sqrt(weights[i]) * y
        return _solve_local(A, weights[i], y)

    def det_terms(flat, i):
        def resid(f):
            r = solve(f, i)[2]
            return r, r
        J, r = jax.jacfwd(resid, has_aux=True)(flat)
        a, var, _ = solve(flat, i)
        return r @ r, J.T @ J, J.T @ r, a, var

    @jax.jit
    def normal_equations(flat):
        chi2, JtJ, Jtr, a, var = jax.lax.map(
            lambda i: det_terms(flat, i), jnp.arange(n_det))
        return chi2.sum(), JtJ.sum(0), Jtr.sum(0), a, var

    return normal_equations


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
    normal_equations = _make_normal_equations(
        data, weights, Q_bore, det_quats, components, offset_fn, unravel,
        n_point, segments, n_seg, filt)

    chi2, JtJ, Jtr, a, var = normal_equations(flat)
    converged = False
    for n_iter in range(1, max_iter + 1):
        A = JtJ + lam * jnp.diag(jnp.diag(JtJ))
        trial = flat - jnp.linalg.solve(A, Jtr)
        chi2_t, JtJ_t, Jtr_t, a_t, var_t = normal_equations(trial)
        if chi2_t < chi2:
            done = (chi2 - chi2_t) <= rtol * chi2
            flat, chi2, JtJ, Jtr, a, var = trial, chi2_t, JtJ_t, Jtr_t, a_t, var_t
            lam = max(lam / 10, 1e-12)
            if done:
                converged = True
                break
        else:
            lam *= 10
            if lam > 1e12:
                converged = True  # no further decrease possible
                break

    cov = np.asarray(jnp.linalg.inv(JtJ))
    a, var = np.asarray(a), np.asarray(var)
    amps, amp_err, col = {}, {}, 0
    for c in components:
        if c.scope == 'global':
            continue
        if c.scope == 'det':
            amps[c.name], amp_err[c.name] = a[:, col], np.sqrt(var[:, col])
            col += 1
        else:
            amps[c.name] = a[:, col:col + n_seg]
            amp_err[c.name] = np.sqrt(var[:, col:col + n_seg])
            col += n_seg
    g = np.asarray(flat[n_point:])
    g_err = np.sqrt(np.diag(cov)[n_point:])
    return dict(params=unravel(flat[:n_point]),
                globals={c.name: g[k] for k, c in enumerate(glob)},
                globals_err={c.name: g_err[k] for k, c in enumerate(glob)},
                amplitudes=amps, amplitude_err=amp_err, cov=cov,
                n_pointing=n_point, chi2=float(chi2), n_iter=n_iter,
                converged=converged)


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
