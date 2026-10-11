"""Joint fit of pointing offsets and detector gains against a sky template.

The model for detector i is

    d_i(t) = g_i * F[T(x_i(t; p))] + b_i,

where T is a template map sampled at the detector's sky position x_i, F is
an optional linear filter, and p holds the pointing parameters.  The gains
g_i and baselines b_i enter linearly, so for each p they are solved in closed
form (variable projection) and the nonlinear problem is in p alone.  The
Jacobian of the profiled residuals comes from jax.jacfwd through the full
chain: offset rotation, detector quaternion, boresight, template sampling,
filter and the linear solve.  p is minimized by Levenberg-Marquardt.

Pointing offsets are small rotations in the focal plane, applied between the
boresight and the detector:

    q_det_sky(t) = Q_bore(t) * q_off(p, i, t) * q_det_i.

How p maps to offsets is set by an ``offset_fn``.  `group_offsets` gives one
(xi, eta) offset per detector group (e.g. per wafer, or one for the array);
returning per-sample offsets allows time-dependent terms such as per-crossing
shifts.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax.scipy.ndimage import map_coordinates

from so_pointjax.proj import quat


@jax.tree_util.register_pytree_node_class
class Template:
    """Map on a flat grid centred on (lon0, lat0).

    ``data[iy, ix]`` is the value at east = x0 + ix*pix, north = y0 + iy*pix
    (radians), in the orthographic projection about (lon0, lat0).  The default
    x0, y0 put the centre of the grid on (lon0, lat0).  Sampling is bilinear
    and zero outside the grid.  The template should already carry the beam
    of the data being fitted.
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


def _profile(d, w, s):
    """Weighted least squares for d ~ g*s + b; returns g, b, var(g), residual."""
    sw = jnp.sum(w)
    s1 = jnp.sum(w * s)
    ss = jnp.sum(w * s * s)
    d1 = jnp.sum(w * d)
    sd = jnp.sum(w * s * d)
    det = ss * sw - s1 * s1
    g = (sw * sd - s1 * d1) / det
    b = (ss * d1 - s1 * sd) / det
    r = jnp.sqrt(w) * (d - g * s - b)
    return g, b, sw / det, r


def _make_normal_equations(data, weights, Q_bore, det_quats, template,
                           offset_fn, unravel, filt):
    data = jnp.asarray(data, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    Q_bore = jnp.asarray(Q_bore, dtype=jnp.float64)
    det_quats = jnp.asarray(det_quats, dtype=jnp.float64)

    def det_model(flat, i):
        off = offset_fn(unravel(flat), i)
        q_det = quat.qmul(offset_quat(off[..., 0], off[..., 1]), det_quats[i])
        s = template.sample(*template.sky_offsets(quat.qmul(Q_bore, q_det)))
        return s if filt is None else filt(s)

    def det_terms(flat, i):
        def resid(f):
            r = _profile(data[i], weights[i], det_model(f, i))[3]
            return r, r
        J, r = jax.jacfwd(resid, has_aux=True)(flat)
        g, b, var_g, _ = _profile(data[i], weights[i], det_model(flat, i))
        return r @ r, J.T @ J, J.T @ r, g, b, var_g

    @jax.jit
    def normal_equations(flat):
        chi2, JtJ, Jtr, g, b, var_g = jax.lax.map(
            lambda i: det_terms(flat, i), jnp.arange(data.shape[0]))
        return (chi2.sum(), JtJ.sum(0), Jtr.sum(0),
                dict(gains=g, baselines=b, gain_err=jnp.sqrt(var_g)))

    return normal_equations


def fit_offsets_gains(data, weights, Q_bore, det_quats, template, params0,
                      offset_fn, filt=None, max_iter=50, rtol=1e-10,
                      lam=1e-3):
    """Fit pointing parameters, with per-detector gains and baselines profiled.

    Parameters
    ----------
    data : array (n_det, n_samp)
        Detector time streams, already filtered with ``filt`` if one is given.
    weights : array (n_det, n_samp)
        Inverse noise variance per sample; zero excludes a sample.  Restrict
        to the region around the source to keep the fit fast.
    Q_bore : array (n_samp, 4)
        Boresight quaternions, e.g. CelestialSightLine.az_el(...).Q.
    det_quats : array (n_det, 4)
        Detector quaternions, e.g. FocalPlane.from_xieta(...).quats.
    template : Template
    params0 : pytree
        Starting pointing parameters, in the form ``offset_fn`` expects.
    offset_fn : callable (params, i) -> array (2,) or (n_samp, 2)
        (xi, eta) offset in radians for detector i, e.g. `group_offsets`.
    filt : callable (n_samp,) -> (n_samp,), optional
        Linear filter applied to the model, matching the data's.

    Returns
    -------
    dict with
        params : fitted parameters, same structure as params0
        cov : covariance of the flattened parameters (with gains and
            baselines marginalized); scale by chi2/dof if the weights are
            not true inverse variances
        unravel : turns a flat parameter vector back into the pytree
        gains, baselines, gain_err : per detector, at the solution
        chi2, n_iter, converged
    """
    flat, unravel = ravel_pytree(params0)
    flat = jnp.asarray(flat, dtype=jnp.float64)
    normal_equations = _make_normal_equations(
        data, weights, Q_bore, det_quats, template, offset_fn, unravel, filt)

    chi2, JtJ, Jtr, lin = normal_equations(flat)
    converged = False
    for n_iter in range(1, max_iter + 1):
        A = JtJ + lam * jnp.diag(jnp.diag(JtJ))
        trial = flat - jnp.linalg.solve(A, Jtr)
        chi2_t, JtJ_t, Jtr_t, lin_t = normal_equations(trial)
        if chi2_t < chi2:
            done = (chi2 - chi2_t) <= rtol * chi2
            flat, chi2, JtJ, Jtr, lin = trial, chi2_t, JtJ_t, Jtr_t, lin_t
            lam = max(lam / 10, 1e-12)
            if done:
                converged = True
                break
        else:
            lam *= 10
            if lam > 1e12:
                converged = True  # no further decrease possible
                break

    return dict(params=unravel(flat), cov=np.asarray(jnp.linalg.inv(JtJ)),
                unravel=unravel, chi2=float(chi2), n_iter=n_iter,
                converged=converged,
                **{k: np.asarray(v) for k, v in lin.items()})
