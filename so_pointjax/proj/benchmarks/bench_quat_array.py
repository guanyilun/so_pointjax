"""Benchmark: so_pointjax.proj vs so3g quaternion array operations.

Focuses on G3VectorQuat-level operations that are typical in SO pipelines:
construction, arithmetic, decomposition, and composition chains.

Usage:
    python -m so_pointjax.proj.benchmarks.bench_quat_array [--quick]
"""

import time
import sys
import numpy as np
import jax
import jax.numpy as jnp

from so3g.proj import quat as so3g_quat
from spt3g.core import G3VectorQuat

from so_pointjax.proj import quat as jax_quat


def timeit(fn, n_iter=20, warmup=3):
    """Time a function, return median time in seconds."""
    for _ in range(warmup):
        result = fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = fn()
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, tuple):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times)


# =========================================================================
# Benchmarks
# =========================================================================

def bench_construct_euler(N):
    """Construct N quaternions from angles."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    angles_j = jnp.array(angles)

    t_so3g = timeit(lambda: so3g_quat.euler(2, angles))

    euler_jit = jax.jit(lambda a: jax_quat.euler(2, a))
    _ = euler_jit(angles_j)
    _.block_until_ready()
    t_jax = timeit(lambda: euler_jit(angles_j))

    return t_so3g, t_jax


def bench_construct_xieta(N):
    """Construct N detector quaternions from xi/eta/gamma."""
    xi = np.random.uniform(-0.05, 0.05, N)
    eta = np.random.uniform(-0.05, 0.05, N)
    gamma = np.random.uniform(0, np.pi, N)
    xi_j, eta_j, gamma_j = jnp.array(xi), jnp.array(eta), jnp.array(gamma)

    t_so3g = timeit(lambda: so3g_quat.rotation_xieta(xi, eta, gamma))

    f = jax.jit(jax_quat.rotation_xieta)
    _ = f(xi_j, eta_j, gamma_j)
    _.block_until_ready()
    t_jax = timeit(lambda: f(xi_j, eta_j, gamma_j))

    return t_so3g, t_jax


def bench_construct_lonlat(N):
    """Construct N quaternions from lon/lat."""
    lon = np.random.uniform(0, 2 * np.pi, N)
    lat = np.random.uniform(-np.pi / 2, np.pi / 2, N)
    lon_j, lat_j = jnp.array(lon), jnp.array(lat)

    t_so3g = timeit(lambda: so3g_quat.rotation_lonlat(lon, lat))

    f = jax.jit(jax_quat.rotation_lonlat)
    _ = f(lon_j, lat_j)
    _.block_until_ready()
    t_jax = timeit(lambda: f(lon_j, lat_j))

    return t_so3g, t_jax


def bench_multiply(N):
    """Multiply two N-element quaternion arrays."""
    angles1 = np.random.uniform(0, 2 * np.pi, N)
    angles2 = np.random.uniform(0, 2 * np.pi, N)

    a_so3g = so3g_quat.euler(2, angles1)
    b_so3g = so3g_quat.euler(1, angles2)

    a_jax = jax_quat.euler(2, jnp.array(angles1))
    b_jax = jax_quat.euler(1, jnp.array(angles2))

    t_so3g = timeit(lambda: a_so3g * b_so3g)

    f = jax.jit(jax_quat.qmul)
    _ = f(a_jax, b_jax)
    _.block_until_ready()
    t_jax = timeit(lambda: f(a_jax, b_jax))

    return t_so3g, t_jax


def bench_scalar_mul(N):
    """Multiply scalar quaternion by N-element array (broadcast)."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    a_so3g = so3g_quat.euler(2, angles)
    s_so3g = so3g_quat.euler(1, 0.5)  # scalar quat

    a_jax = jax_quat.euler(2, jnp.array(angles))
    s_jax = jax_quat.euler(1, 0.5)

    t_so3g = timeit(lambda: s_so3g * a_so3g)

    f = jax.jit(jax_quat.qmul)
    _ = f(s_jax, a_jax)
    _.block_until_ready()
    t_jax = timeit(lambda: f(s_jax, a_jax))

    return t_so3g, t_jax


def bench_chain3(N):
    """Compose 3 quaternion arrays: a * b * c."""
    a1 = np.random.uniform(0, 2 * np.pi, N)
    a2 = np.random.uniform(0, 2 * np.pi, N)
    a3 = np.random.uniform(0, 2 * np.pi, N)

    a_so3g = so3g_quat.euler(2, a1)
    b_so3g = so3g_quat.euler(1, a2)
    c_so3g = so3g_quat.euler(0, a3)

    a_jax = jax_quat.euler(2, jnp.array(a1))
    b_jax = jax_quat.euler(1, jnp.array(a2))
    c_jax = jax_quat.euler(0, jnp.array(a3))

    t_so3g = timeit(lambda: a_so3g * b_so3g * c_so3g)

    @jax.jit
    def chain(a, b, c):
        return jax_quat.qmul(jax_quat.qmul(a, b), c)
    _ = chain(a_jax, b_jax, c_jax)
    _.block_until_ready()
    t_jax = timeit(lambda: chain(a_jax, b_jax, c_jax))

    return t_so3g, t_jax


def bench_chain5(N):
    """Compose 5 quaternion arrays: a * b * c * d * e (typical pointing chain)."""
    arrays_np = [np.random.uniform(0, 2 * np.pi, N) for _ in range(5)]
    axes = [2, 1, 2, 1, 0]

    quats_so3g = [so3g_quat.euler(ax, a) for ax, a in zip(axes, arrays_np)]
    quats_jax = [jax_quat.euler(ax, jnp.array(a)) for ax, a in zip(axes, arrays_np)]

    def so3g_chain():
        q = quats_so3g[0]
        for qi in quats_so3g[1:]:
            q = q * qi
        return q

    @jax.jit
    def jax_chain(q0, q1, q2, q3, q4):
        q = jax_quat.qmul(q0, q1)
        q = jax_quat.qmul(q, q2)
        q = jax_quat.qmul(q, q3)
        q = jax_quat.qmul(q, q4)
        return q
    _ = jax_chain(*quats_jax)
    _.block_until_ready()

    t_so3g = timeit(so3g_chain)
    t_jax = timeit(lambda: jax_chain(*quats_jax))

    return t_so3g, t_jax


def bench_decompose_iso(N):
    """Decompose N quaternions into (theta, phi, psi)."""
    theta = np.random.uniform(0.1, np.pi - 0.1, N)
    phi = np.random.uniform(-np.pi, np.pi, N)
    psi = np.random.uniform(-np.pi, np.pi, N)
    q_so3g = so3g_quat.rotation_iso(theta, phi, psi)
    q_jax = jax_quat.rotation_iso(jnp.array(theta), jnp.array(phi), jnp.array(psi))

    t_so3g = timeit(lambda: so3g_quat.decompose_iso(q_so3g))

    f = jax.jit(jax_quat.decompose_iso)
    _ = f(q_jax)
    jax.tree.map(lambda x: x.block_until_ready(), _)
    t_jax = timeit(lambda: f(q_jax))

    return t_so3g, t_jax


def bench_decompose_xieta(N):
    """Decompose N quaternions into (xi, eta, gamma)."""
    xi = np.random.uniform(-0.05, 0.05, N)
    eta = np.random.uniform(-0.05, 0.05, N)
    gamma = np.random.uniform(0, np.pi, N)
    q_so3g = so3g_quat.rotation_xieta(xi, eta, gamma)
    q_jax = jax_quat.rotation_xieta(jnp.array(xi), jnp.array(eta), jnp.array(gamma))

    t_so3g = timeit(lambda: so3g_quat.decompose_xieta(q_so3g))

    f = jax.jit(jax_quat.decompose_xieta)
    _ = f(q_jax)
    jax.tree.map(lambda x: x.block_until_ready(), _)
    t_jax = timeit(lambda: f(q_jax))

    return t_so3g, t_jax


def bench_decompose_lonlat(N):
    """Decompose N quaternions into (lon, lat, psi)."""
    lon = np.random.uniform(0, 2 * np.pi, N)
    lat = np.random.uniform(-np.pi / 2, np.pi / 2, N)
    q_so3g = so3g_quat.rotation_lonlat(lon, lat)
    q_jax = jax_quat.rotation_lonlat(jnp.array(lon), jnp.array(lat))

    t_so3g = timeit(lambda: so3g_quat.decompose_lonlat(q_so3g))

    f = jax.jit(jax_quat.decompose_lonlat)
    _ = f(q_jax)
    jax.tree.map(lambda x: x.block_until_ready(), _)
    t_jax = timeit(lambda: f(q_jax))

    return t_so3g, t_jax


def bench_construct_decompose_roundtrip(N):
    """Full roundtrip: construct rotation_xieta → decompose_xieta."""
    xi = np.random.uniform(-0.05, 0.05, N)
    eta = np.random.uniform(-0.05, 0.05, N)
    gamma = np.random.uniform(0, np.pi, N)
    xi_j, eta_j, gamma_j = jnp.array(xi), jnp.array(eta), jnp.array(gamma)

    def so3g_fn():
        q = so3g_quat.rotation_xieta(xi, eta, gamma)
        return so3g_quat.decompose_xieta(q)

    @jax.jit
    def jax_fn(x, e, g):
        q = jax_quat.rotation_xieta(x, e, g)
        return jax_quat.decompose_xieta(q)
    _ = jax_fn(xi_j, eta_j, gamma_j)
    jax.tree.map(lambda x: x.block_until_ready(), _)

    t_so3g = timeit(so3g_fn)
    t_jax = timeit(lambda: jax_fn(xi_j, eta_j, gamma_j))

    return t_so3g, t_jax


def bench_bore_det_compose(N_time, N_det):
    """Boresight * detector offset — typical projection inner loop.

    For each time sample and detector, compute q_total = q_bore[t] * q_det[d].
    so3g does this with G3VectorQuat broadcasting.
    JAX does this with explicit broadcasting + qmul.
    """
    bore_angles = np.random.uniform(0, 2 * np.pi, N_time)
    xi = np.random.uniform(-0.03, 0.03, N_det)
    eta = np.random.uniform(-0.03, 0.03, N_det)

    q_bore_so3g = so3g_quat.rotation_lonlat(bore_angles, np.full(N_time, 0.5))
    q_det_so3g = [so3g_quat.rotation_xieta(xi[i], eta[i], 0.0) for i in range(N_det)]

    q_bore_jax = jax_quat.rotation_lonlat(jnp.array(bore_angles), jnp.full(N_time, 0.5))
    q_det_jax = jax_quat.rotation_xieta(jnp.array(xi), jnp.array(eta), jnp.zeros(N_det))

    # so3g: scalar_quat * G3VectorQuat for each detector
    def so3g_fn():
        results = []
        for d in range(N_det):
            results.append(q_bore_so3g * q_det_so3g[d])
        return results

    # JAX: broadcast qmul (N_time,4) * (N_det,1,4) → (N_det,N_time,4)
    @jax.jit
    def jax_fn(q_bore, q_det):
        return jax_quat.qmul(q_bore[None, :, :], q_det[:, None, :])
    _ = jax_fn(q_bore_jax, q_det_jax)
    _.block_until_ready()

    t_so3g = timeit(so3g_fn, n_iter=10, warmup=2)
    t_jax = timeit(lambda: jax_fn(q_bore_jax, q_det_jax), n_iter=10, warmup=2)

    return t_so3g, t_jax


# =========================================================================
# Main
# =========================================================================

def main():
    quick = '--quick' in sys.argv

    print("=" * 80)
    print("so_pointjax.proj vs so3g — Quaternion Array Operations")
    print("=" * 80)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"All JAX timings are jax.jit compiled (post-warmup).")
    print()

    if quick:
        sizes = [100, 1_000, 10_000, 100_000]
    else:
        sizes = [100, 1_000, 10_000, 100_000, 1_000_000]

    benchmarks = [
        ("euler(2, angles)", bench_construct_euler),
        ("rotation_xieta", bench_construct_xieta),
        ("rotation_lonlat", bench_construct_lonlat),
        ("quat multiply", bench_multiply),
        ("scalar * array", bench_scalar_mul),
        ("chain × 3", bench_chain3),
        ("chain × 5", bench_chain5),
        ("decompose_iso", bench_decompose_iso),
        ("decompose_xieta", bench_decompose_xieta),
        ("decompose_lonlat", bench_decompose_lonlat),
        ("xieta roundtrip", bench_construct_decompose_roundtrip),
    ]

    # Header
    header = f"  {'Operation':<22s}"
    for N in sizes:
        header += f"  {'N=' + f'{N:,}':>12s}"
    print(header)
    print(f"  {'-' * (22 + 14 * len(sizes))}")

    for name, fn in benchmarks:
        row_so3g = f"  {name:<22s}"
        row_jax = f"  {'':22s}"
        row_ratio = f"  {'':22s}"
        for N in sizes:
            t_s, t_j = fn(N)
            ratio = t_s / t_j if t_j > 0 else float('inf')
            row_so3g += f"  {t_s*1000:>10.3f}ms"
            row_jax += f"  {t_j*1000:>10.3f}ms"
            if ratio >= 1:
                row_ratio += f"  {ratio:>9.1f}x ✓"
            else:
                row_ratio += f"  {ratio:>9.2f}x  "
        print(row_so3g + "  ← so3g")
        print(row_jax + "  ← JAX JIT")
        print(row_ratio + "  ← speedup")
        print()

    # Bore * det composition
    print("-" * 80)
    print("Boresight × detector composition (q_bore[t] * q_det[d]):")
    print(f"  {'(N_time, N_det)':<22s}  {'so3g (ms)':>10s}  {'JAX (ms)':>10s}  {'speedup':>10s}")
    print(f"  {'-'*55}")

    if quick:
        cases = [(1000, 10), (1000, 100), (10000, 100)]
    else:
        cases = [(1000, 10), (1000, 100), (10000, 100), (10000, 1000), (100000, 100)]

    for nt, nd in cases:
        t_s, t_j = bench_bore_det_compose(nt, nd)
        ratio = t_s / t_j if t_j > 0 else float('inf')
        mark = "✓" if ratio >= 1 else ""
        print(f"  ({nt:,}, {nd:,}){'':<{22-len(f'({nt:,}, {nd:,})')}}  {t_s*1000:>10.3f}  {t_j*1000:>10.3f}  {ratio:>8.1f}x {mark}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
