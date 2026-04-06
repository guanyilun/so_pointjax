"""Quaternion utilities mirroring so3g.proj.quat, backed by JAX.

Quaternions are jnp.ndarray of shape (4,) or (N, 4) with layout [w, x, y, z].
All functions are compatible with jax.jit, jax.grad, jax.vmap.
"""

import jax
import jax.numpy as jnp
import numpy as np

from so_pointjax.qpoint._quaternion import (
    mul, conj, normalize, r1, r2, r3, r2_mul, r3_mul,
    quat2radecpa, radecpa2quat,
)

DEG = jnp.pi / 180.0


# ---------------------------------------------------------------------------
# Quat wrapper class
# ---------------------------------------------------------------------------

class Quat:
    """Quaternion with operator overloading and automatic broadcasting.

    Wraps a jnp.ndarray of shape ``(4,)`` (scalar) or ``(N, 4)`` (array).
    Arithmetic broadcasts following the same rules as jnp arrays:
    ``(4,)`` × ``(N, 4)`` → ``(N, 4)``, ``(N, 4)`` × ``(4,)`` → ``(N, 4)``.

    Operators:
        ``q1 * q2``  — quaternion multiplication (with broadcasting)
        ``~q``       — conjugate (= inverse for unit quaternions)
        ``abs(q)``   — norm
        ``2 * q``    — scalar multiplication
        ``q[i]``     — index into batch → scalar Quat
        ``q[2:5]``   — slice batch → Quat

    Registered as a JAX pytree so it works with jax.jit, jax.grad, jax.vmap.

    Parameters
    ----------
    data : array-like or four scalars
        ``Quat(array)`` where array has shape ``(..., 4)``, or
        ``Quat(w, x, y, z)`` for a scalar quaternion.

    Examples
    --------
    >>> q = Quat(1, 0, 0, 0)
    >>> q_arr = Quat(euler(2, jnp.linspace(0, 1, 100)))
    >>> q * q_arr          # (4,) × (100, 4) → (100, 4)
    >>> q_arr * ~q_arr     # element-wise inverse → all identity
    >>> q_arr[0]           # Quat scalar
    >>> q_arr[10:20]       # Quat batch of 10
    """

    __slots__ = ('_data',)

    def __init__(self, *args):
        if len(args) == 1:
            data = args[0]
            if isinstance(data, Quat):
                self._data = data._data
                return
            self._data = jnp.asarray(data, dtype=jnp.float64)
        elif len(args) == 4:
            self._data = jnp.array(args, dtype=jnp.float64)
        else:
            raise TypeError(
                f"Quat() takes 1 or 4 arguments ({len(args)} given)")

        if self._data.shape[-1] != 4:
            raise ValueError(
                f"Last dimension must be 4, got shape {self._data.shape}")

    # -- Raw array access --

    @property
    def data(self):
        """Underlying jnp.ndarray of shape (..., 4)."""
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def ndim(self):
        return self._data.ndim

    def __len__(self):
        if self._data.ndim == 1:
            raise TypeError("scalar Quat has no len()")
        return self._data.shape[0]

    # -- Component access (.a .b .c .d like spt3g, plus .w .x .y .z) --

    @property
    def a(self):
        """Scalar (w) component."""
        return self._data[..., 0]

    @property
    def b(self):
        """i (x) component."""
        return self._data[..., 1]

    @property
    def c(self):
        """j (y) component."""
        return self._data[..., 2]

    @property
    def d(self):
        """k (z) component."""
        return self._data[..., 3]

    # Aliases
    w = a
    x = b
    y = c
    z = d

    # -- Class constructors --

    @classmethod
    def identity(cls):
        """Identity quaternion (no rotation)."""
        return cls(1., 0., 0., 0.)

    @classmethod
    def from_euler(cls, axis, angle):
        """Quaternion for Euler rotation about a Cartesian axis.

        Parameters
        ----------
        axis : {0, 1, 2}
            Rotation axis (x, y, z).
        angle : float or array
            Rotation angle(s) in radians.
        """
        return cls(euler(axis, angle))

    @classmethod
    def from_iso(cls, theta, phi, psi=None):
        """Quaternion from ZYZ Euler angles: Rz(phi) Ry(theta) Rz(psi)."""
        return cls(rotation_iso(theta, phi, psi))

    @classmethod
    def from_lonlat(cls, lon, lat, psi=0., azel=False):
        """Quaternion from longitude/latitude (or az/el if azel=True)."""
        return cls(rotation_lonlat(lon, lat, psi, azel=azel))

    @classmethod
    def from_xieta(cls, xi, eta, gamma=0.):
        """Quaternion from focal plane tangent-plane coordinates."""
        return cls(rotation_xieta(xi, eta, gamma))

    # -- Decomposition methods --

    def to_iso(self):
        """Decompose into ZYZ Euler angles (theta, phi, psi)."""
        return decompose_iso(self._data)

    def to_lonlat(self, azel=False):
        """Decompose into (lon, lat, psi) or (az, el, psi) if azel=True."""
        return decompose_lonlat(self._data, azel=azel)

    def to_xieta(self):
        """Decompose into focal plane coordinates (xi, eta, gamma)."""
        return decompose_xieta(self._data)

    # -- Other operations --

    def normalized(self):
        """Return a copy normalized to unit quaternion."""
        return Quat(qnormalize(self._data))

    def rotate(self, v):
        """Rotate vector(s) v by this quaternion: q v q*.

        Parameters
        ----------
        v : array shape (3,) or (N, 3)

        Returns
        -------
        v_rot : array shape (3,) or (N, 3)
        """
        return qrotate(self._data, v)

    # -- Quaternion arithmetic --

    def _coerce(self, other):
        """Try to get a jnp array of shape (..., 4) from other."""
        if isinstance(other, Quat):
            return other._data
        if isinstance(other, (int, float)):
            return None
        other = jnp.asarray(other, dtype=jnp.float64)
        if other.ndim >= 1 and other.shape[-1] == 4:
            return other
        return None

    def __mul__(self, other):
        od = self._coerce(other)
        if od is not None:
            return Quat(qmul(self._data, od))
        if isinstance(other, (int, float)):
            return Quat(self._data * other)
        return NotImplemented

    def __rmul__(self, other):
        od = self._coerce(other)
        if od is not None:
            return Quat(qmul(od, self._data))
        if isinstance(other, (int, float)):
            return Quat(other * self._data)
        return NotImplemented

    def __invert__(self):
        """Quaternion conjugate (= inverse for unit quaternions)."""
        return Quat(qconj(self._data))

    def __abs__(self):
        """Quaternion norm."""
        return qnorm(self._data)

    def __neg__(self):
        return Quat(-self._data)

    # -- Indexing / slicing --

    def __getitem__(self, idx):
        result = self._data[idx]
        if result.ndim >= 1 and result.shape[-1] == 4:
            return Quat(result)
        return result

    # -- Conversion --

    def __array__(self, dtype=None, copy=None):
        arr = np.asarray(self._data)
        if dtype is not None:
            arr = arr.astype(dtype)
        if copy:
            arr = arr.copy()
        return arr

    def __jax_array__(self):
        return self._data

    def numpy(self):
        """Convert to numpy array."""
        return np.asarray(self._data)

    # -- Representation --

    def __repr__(self):
        if self._data.ndim == 1:
            w, x, y, z = [float(v) for v in self._data]
            return f"Quat({w:.6g}, {x:.6g}, {y:.6g}, {z:.6g})"
        return f"Quat(shape={self._data.shape[:-1]}, dtype={self._data.dtype})"

    def __str__(self):
        if self._data.ndim == 1:
            w, x, y, z = [float(v) for v in self._data]
            return f"({w:.6g},{x:.6g},{y:.6g},{z:.6g})"
        n = self._data.shape[0] if self._data.ndim == 2 else self._data.shape[:-1]
        return f"Quat[{n}]"

    # -- Comparison (for testing) --

    def __eq__(self, other):
        if isinstance(other, Quat):
            return jnp.array_equal(self._data, other._data)
        return NotImplemented


# Register Quat as a JAX pytree so jit/grad/vmap work
jax.tree_util.register_pytree_node(
    Quat,
    lambda q: ((q._data,), None),       # flatten
    lambda _, data: Quat(data[0]),       # unflatten
)


# ---------------------------------------------------------------------------
# Euler rotation
# ---------------------------------------------------------------------------

def euler(axis, angle):
    """Quaternion for an Euler rotation about a cartesian axis.

    Parameters
    ----------
    axis : {0, 1, 2}
        Rotation axis (x, y, z).
    angle : float or array
        Rotation angle in radians.

    Returns
    -------
    q : array shape (4,) or (N, 4)
    """
    angle = jnp.asarray(angle, dtype=jnp.float64)
    c = jnp.cos(angle / 2)
    s = jnp.sin(angle / 2)
    z = jnp.zeros_like(angle)

    # Build quaternion components based on axis
    # axis=0 → (c, s, 0, 0), axis=1 → (c, 0, s, 0), axis=2 → (c, 0, 0, s)
    components = [
        jnp.stack([c, s, z, z], axis=-1),  # axis 0
        jnp.stack([c, z, s, z], axis=-1),  # axis 1
        jnp.stack([c, z, z, s], axis=-1),  # axis 2
    ]
    return components[axis]


# ---------------------------------------------------------------------------
# Rotation constructors
# ---------------------------------------------------------------------------

def rotation_iso(theta, phi, psi=None):
    """Quaternion for composed Euler rotations Rz(phi) Ry(theta) Rz(psi).

    Parameters
    ----------
    theta, phi : float or array
        Rotation angles in radians.
    psi : float or array or None
        Third Euler angle. If None, omitted.

    Returns
    -------
    q : array shape (4,) or (N, 4)
    """
    q = qmul(euler(2, phi), euler(1, theta))
    if psi is not None:
        q = qmul(q, euler(2, psi))
    return q


def rotation_lonlat(lon, lat, psi=0., azel=False):
    """Quaternion for Rz(lon) Ry(pi/2 - lat) Rz(psi).

    Parameters
    ----------
    lon, lat : float or array
        Longitude and latitude in radians.
    psi : float or array
        Third Euler angle in radians.
    azel : bool
        If True, flip sign of lon (interpret as azimuth/elevation).

    Returns
    -------
    q : array shape (4,) or (N, 4)
    """
    if azel:
        return rotation_iso(jnp.pi / 2 - lat, -lon, psi)
    return rotation_iso(jnp.pi / 2 - lat, lon, psi)


def rotation_xieta(xi, eta, gamma=0.):
    """Quaternion rotating boresight center to focal plane position (xi, eta, gamma).

    Equivalent to Rz(phi) Ry(theta) Rz(psi) where:
        xi = -sin(theta) * sin(phi)
        eta = -sin(theta) * cos(phi)
        gamma = psi + phi

    Parameters
    ----------
    xi, eta : float or array
        Tangent plane coordinates in radians.
    gamma : float or array
        Polarization angle in radians.

    Returns
    -------
    q : array shape (4,) or (N, 4)
    """
    phi = jnp.arctan2(-xi, -eta)
    theta = jnp.arcsin(jnp.sqrt(xi**2 + eta**2))
    psi = gamma - phi
    return rotation_iso(theta, phi, psi)


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def decompose_iso(q):
    """Decompose quaternion into ZYZ Euler angles: q = Rz(phi) Ry(theta) Rz(psi).

    Parameters
    ----------
    q : array shape (4,) or (N, 4)

    Returns
    -------
    theta, phi, psi : floats or arrays in radians
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    a, b, c, d = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    psi = jnp.arctan2(a * b + c * d, a * c - b * d)
    phi = jnp.arctan2(c * d - a * b, a * c + b * d)
    theta = 2 * jnp.arctan2(jnp.sqrt(b**2 + c**2), jnp.sqrt(a**2 + d**2))

    return theta, phi, psi


def decompose_lonlat(q, azel=False):
    """Decompose quaternion into (lon, lat, psi) assuming rotation_lonlat construction.

    Parameters
    ----------
    q : array shape (4,) or (N, 4)
    azel : bool
        If True, returns (-lon, lat, psi) for azimuth/elevation interpretation.

    Returns
    -------
    lon, lat, psi : floats or arrays in radians
    """
    theta, phi, psi = decompose_iso(q)
    if azel:
        return -phi, jnp.pi / 2 - theta, psi
    return phi, jnp.pi / 2 - theta, psi


def decompose_xieta(q):
    """Decompose quaternion into (xi, eta, gamma) assuming rotation_xieta construction.

    Parameters
    ----------
    q : array shape (4,) or (N, 4)

    Returns
    -------
    xi, eta, gamma : floats or arrays in radians
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    a, b, c, d = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return (
        2 * (a * b - c * d),
        -2 * (a * c + b * d),
        jnp.arctan2(2 * a * d, a * a - d * d),
    )


# ---------------------------------------------------------------------------
# Quaternion arithmetic (array-friendly)
# ---------------------------------------------------------------------------

def qmul(a, b):
    """Quaternion multiplication, supporting batched (N, 4) arrays.

    Parameters
    ----------
    a, b : array shape (4,) or (N, 4)
        Quaternions to multiply.

    Returns
    -------
    q : array shape (4,) or (N, 4)
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)

    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    return jnp.stack([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0,
    ], axis=-1)


def qconj(q):
    """Quaternion conjugate, supporting batched arrays.

    Parameters
    ----------
    q : array shape (4,) or (N, 4)

    Returns
    -------
    q_conj : array shape (4,) or (N, 4)
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    return q.at[..., 1:].multiply(-1)


def qnorm(q):
    """Quaternion norm.

    Parameters
    ----------
    q : array shape (4,) or (N, 4)

    Returns
    -------
    norm : float or array
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    return jnp.sqrt(jnp.sum(q * q, axis=-1))


def qnormalize(q):
    """Normalize to unit quaternion.

    Parameters
    ----------
    q : array shape (4,) or (N, 4)

    Returns
    -------
    q_unit : array shape (4,) or (N, 4)
    """
    q = jnp.asarray(q, dtype=jnp.float64)
    return q / qnorm(q)[..., None]


def qrotate(q, v):
    """Rotate vector(s) v by quaternion(s) q: q v q*.

    Parameters
    ----------
    q : array shape (4,) or (N, 4)
    v : array shape (3,) or (N, 3)

    Returns
    -------
    v_rot : array shape (3,) or (N, 3)
    """
    v = jnp.asarray(v, dtype=jnp.float64)
    # Promote vector to quaternion: (0, vx, vy, vz)
    z = jnp.zeros(v.shape[:-1])
    v_q = jnp.stack([z, v[..., 0], v[..., 1], v[..., 2]], axis=-1)
    result = qmul(qmul(q, v_q), qconj(q))
    return result[..., 1:]
