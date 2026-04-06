"""Benchmark: so_pointjax.proj vs so3g for quaternion and pointing operations.

Usage:
    python -m so_pointjax.proj.benchmarks.bench_so3g [--quick]
"""

import time
import sys
import numpy as np
import jax
import jax.numpy as jnp

# --- so3g (reference) ---
from so3g.proj import quat as so3g_quat
from so3g.proj.coords import (
    CelestialSightLine as CSL_so3g,
    FocalPlane as FP_so3g,
)

# --- so_pointjax.proj ---
from so_pointjax.proj import quat as jax_quat
from so_pointjax.proj.coords import (
    CelestialSightLine as CSL_jax,
    FocalPlane as FP_jax,
)

DEG = np.pi / 180.0

# ERA constants for naive pointing JIT benchmark
ERA_EPOCH = 946684800 + 3600 * 12
ERA_POLY = jnp.array([6.300387486754831, 4.894961212823756])


def timeit(fn, n_iter=10, warmup=2):
    """Time a function, return median time in seconds."""
    for _ in range(warmup):
        result = fn()
        _block(result)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = fn()
        _block(result)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times)


def _block(result):
    """Block until JAX results are ready."""
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    elif isinstance(result, tuple):
        for r in result:
            if hasattr(r, 'block_until_ready'):
                r.block_until_ready()


# =========================================================================
# JIT-compiled JAX functions for fair comparison
# =========================================================================

_euler_jit = jax.jit(lambda angles: jax_quat.euler(2, angles))
_rotation_xieta_jit = jax.jit(jax_quat.rotation_xieta)
_decompose_xieta_jit = jax.jit(jax_quat.decompose_xieta)
_qmul_jit = jax.jit(jax_quat.qmul)
_rotation_lonlat_jit = jax.jit(jax_quat.rotation_lonlat)


# =========================================================================
# Quaternion benchmarks
# =========================================================================

def bench_euler(N):
    angles = np.random.uniform(0, 2 * np.pi, N)
    angles_j = jnp.array(angles)

    t_so3g = timeit(lambda: so3g_quat.euler(2, angles))
    t_jax_raw = timeit(lambda: jax_quat.euler(2, angles_j))
    t_jax_jit = timeit(lambda: _euler_jit(angles_j))

    return t_so3g, t_jax_raw, t_jax_jit


def bench_rotation_xieta(N):
    xi = np.random.uniform(-0.05, 0.05, N)
    eta = np.random.uniform(-0.05, 0.05, N)
    gamma = np.random.uniform(0, np.pi, N)
    xi_j, eta_j, gamma_j = jnp.array(xi), jnp.array(eta), jnp.array(gamma)

    t_so3g = timeit(lambda: so3g_quat.rotation_xieta(xi, eta, gamma))
    t_jax_raw = timeit(lambda: jax_quat.rotation_xieta(xi_j, eta_j, gamma_j))
    t_jax_jit = timeit(lambda: _rotation_xieta_jit(xi_j, eta_j, gamma_j))

    return t_so3g, t_jax_raw, t_jax_jit


def bench_decompose_xieta(N):
    xi = np.random.uniform(-0.05, 0.05, N)
    eta = np.random.uniform(-0.05, 0.05, N)
    gamma = np.random.uniform(0, np.pi, N)

    q_so3g = so3g_quat.rotation_xieta(xi, eta, gamma)
    q_jax = jax_quat.rotation_xieta(jnp.array(xi), jnp.array(eta), jnp.array(gamma))

    t_so3g = timeit(lambda: so3g_quat.decompose_xieta(q_so3g))
    t_jax_raw = timeit(lambda: jax_quat.decompose_xieta(q_jax))
    t_jax_jit = timeit(lambda: _decompose_xieta_jit(q_jax))

    return t_so3g, t_jax_raw, t_jax_jit


def bench_qmul(N):
    angles1 = np.random.uniform(0, 2 * np.pi, N)
    angles2 = np.random.uniform(0, 2 * np.pi, N)

    a_so3g = so3g_quat.euler(2, angles1)
    b_so3g = so3g_quat.euler(1, angles2)

    a_jax = jax_quat.euler(2, jnp.array(angles1))
    b_jax = jax_quat.euler(1, jnp.array(angles2))

    t_so3g = timeit(lambda: a_so3g * b_so3g)
    t_jax_raw = timeit(lambda: jax_quat.qmul(a_jax, b_jax))
    t_jax_jit = timeit(lambda: _qmul_jit(a_jax, b_jax))

    return t_so3g, t_jax_raw, t_jax_jit


def bench_rotation_lonlat(N):
    lon = np.random.uniform(0, 2 * np.pi, N)
    lat = np.random.uniform(-np.pi / 2, np.pi / 2, N)
    lon_j, lat_j = jnp.array(lon), jnp.array(lat)

    t_so3g = timeit(lambda: so3g_quat.rotation_lonlat(lon, lat))
    t_jax_raw = timeit(lambda: jax_quat.rotation_lonlat(lon_j, lat_j))
    t_jax_jit = timeit(lambda: _rotation_lonlat_jit(lon_j, lat_j))

    return t_so3g, t_jax_raw, t_jax_jit


# =========================================================================
# Pointing pipeline benchmarks
# =========================================================================

def bench_naive_az_el(N):
    t = np.linspace(1700000000, 1700000600, N)
    az = np.linspace(0, 2 * np.pi, N)
    el = np.full(N, 45 * DEG)

    t_j = jnp.array(t)
    az_j = jnp.array(az)
    el_j = jnp.array(el)

    site = CSL_jax.decode_site('act')

    t_so3g = timeit(lambda: CSL_so3g.naive_az_el(t, az, el, site='act'))
    t_jax = timeit(lambda: CSL_jax.naive_az_el(t_j, az_j, el_j, site='act'))

    # JIT-compiled version
    @jax.jit
    def naive_jit(t_arr, az_arr, el_arr):
        J = (t_arr - ERA_EPOCH) / 86400.0
        era = jnp.polyval(ERA_POLY, J)
        lst = era + site.lon * DEG
        return jax_quat.qmul(
            jax_quat.qmul(
                jax_quat.qmul(
                    jax_quat.euler(2, lst),
                    jax_quat.euler(1, jnp.pi / 2 - site.lat * DEG)),
                jax_quat.qmul(
                    jax_quat.euler(2, jnp.pi + jnp.zeros_like(t_arr)),
                    jax_quat.euler(2, -az_arr))),
            jax_quat.qmul(
                jax_quat.euler(1, jnp.pi / 2 - el_arr),
                jax_quat.euler(2, jnp.zeros_like(t_arr))))

    # Warmup
    _ = naive_jit(t_j, az_j, el_j)
    _.block_until_ready()

    t_jax_jit = timeit(lambda: naive_jit(t_j, az_j, el_j))

    # Verify agreement
    csl_so3g = CSL_so3g.naive_az_el(t, az, el, site='act')
    csl_jax = CSL_jax.naive_az_el(t_j, az_j, el_j, site='act')
    q_so3g = np.array(csl_so3g.Q)
    q_jax = np.array(csl_jax.Q)
    max_diff = np.max(np.abs(q_so3g - q_jax))

    return t_so3g, t_jax, t_jax_jit, max_diff


def _fix_qpoint_import():
    """Ensure the installed qpoint package is importable.

    The local qpoint/ source tree can shadow the installed package.
    Temporarily remove cwd-based entries and reimport.
    """
    import importlib
    cwd = str(__import__('pathlib').Path.cwd())
    saved = sys.path[:]
    sys.path = [p for p in sys.path if not p.startswith(cwd)]
    # Clear cached namespace-package import
    for mod in list(sys.modules):
        if mod == 'qpoint' or mod.startswith('qpoint.'):
            del sys.modules[mod]
    try:
        import qpoint
        importlib.reload(qpoint)
        return hasattr(qpoint, 'QPoint')
    except Exception:
        return False
    finally:
        sys.path = saved


def bench_az_el(N):
    """Benchmark high-precision pointing: so3g (C qpoint) vs so_pointjax.qpoint (fast path).

    We benchmark the so_pointjax.qpoint fast path directly (precompute + jit vmap)
    to show the actual runtime performance after compilation.
    """
    from so_pointjax.qpoint._pointing import (
        precompute_corrections, azelpsi2bore_fast,
    )
    from so_pointjax.qpoint._quaternion import quat2radecpa

    t = np.linspace(1700000000, 1700000600, N)
    az = np.linspace(0, 360, N)  # degrees for both
    el = np.full(N, 45.0)

    # --- so3g version (C qpoint) ---
    az_rad = az * DEG
    el_rad = el * DEG
    try:
        t_so3g = timeit(
            lambda: CSL_so3g.az_el(t, az_rad, el_rad, site='act', weather='toco'),
            n_iter=5, warmup=1)
    except Exception as e:
        print(f"    [so3g az_el failed: {e}]")
        t_so3g = None

    # --- so_pointjax.qpoint fast path ---
    site = CSL_jax.decode_site('act')
    corr = precompute_corrections(t, accuracy=1, mean_aber=True)
    q_npb_per = corr['q_npb'][corr['npb_idx']]
    q_wobble_per = corr['q_wobble'][corr['npb_idx']]
    beta_per = corr['beta_earth'][corr['aber_idx']]
    az_j = jnp.array(az)
    el_j = jnp.array(el)

    import so_pointjax.erfa
    weather_A, weather_B = so_pointjax.erfa.refco(550.0, 0.0, 0.2,
                                           299792458.0 * 1e-3 / 150e9)

    def forward(az_i, el_i, tt1, tt2, ut1_1, ut1_2, npb, wob, beta):
        q = azelpsi2bore_fast(
            az_i, el_i, 0.0, site.lon, site.lat,
            tt1, tt2, ut1_1, ut1_2, npb, wob, beta,
            weather_A=weather_A, weather_B=weather_B,
        )
        ra, dec, pa = quat2radecpa(q)
        return ra, dec

    forward_vmap = jax.jit(jax.vmap(forward))

    # Warmup / compile
    ra, dec = forward_vmap(
        az_j, el_j,
        corr['tt1'], corr['tt2'],
        corr['ut1_1'], corr['ut1_2'],
        q_npb_per, q_wobble_per, beta_per,
    )
    ra.block_until_ready()

    t_jax = timeit(
        lambda: forward_vmap(
            az_j, el_j,
            corr['tt1'], corr['tt2'],
            corr['ut1_1'], corr['ut1_2'],
            q_npb_per, q_wobble_per, beta_per,
        ),
        n_iter=5, warmup=0)

    return t_so3g, t_jax


def bench_coords(N, n_det):
    """Benchmark .coords() with a focal plane."""
    t = np.linspace(1700000000, 1700000600, N)
    az = np.linspace(0, 2 * np.pi, N)
    el = np.full(N, 45 * DEG)

    # so3g version
    csl_so3g = CSL_so3g.naive_az_el(t, az, el, site='act')
    xi = np.random.uniform(-0.03, 0.03, n_det)
    eta = np.random.uniform(-0.03, 0.03, n_det)
    gamma = np.random.uniform(0, np.pi, n_det)
    fp_so3g = FP_so3g.from_xieta(xi, eta, gamma)

    # JAX version
    t_j, az_j, el_j = jnp.array(t), jnp.array(az), jnp.array(el)
    csl_jax = CSL_jax.naive_az_el(t_j, az_j, el_j, site='act')
    fp_jax = FP_jax.from_xieta(xi, eta, gamma)

    t_so3g = timeit(lambda: csl_so3g.coords(fp_so3g), n_iter=5, warmup=1)
    t_jax = timeit(lambda: csl_jax.coords(fp_jax), n_iter=5, warmup=1)

    return t_so3g, t_jax


def bench_for_lonlat_coords(N):
    lon = np.random.uniform(0, 2 * np.pi, N)
    lat = np.random.uniform(-np.pi / 4, np.pi / 4, N)
    lon_j, lat_j = jnp.array(lon), jnp.array(lat)

    def so3g_fn():
        csl = CSL_so3g.for_lonlat(lon, lat)
        return csl.coords()

    def jax_fn():
        csl = CSL_jax.for_lonlat(lon_j, lat_j)
        return csl.coords()

    t_so3g = timeit(so3g_fn)
    t_jax = timeit(jax_fn)

    return t_so3g, t_jax


# =========================================================================
# Gradient benchmarks (JAX-only)
# =========================================================================

def bench_grad_xieta():
    def loss(xi, eta, gamma):
        q = jax_quat.rotation_xieta(xi, eta, gamma)
        xi_out, eta_out, _ = jax_quat.decompose_xieta(q)
        return xi_out**2 + eta_out**2

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))

    # Warmup
    g = grad_fn(0.01, -0.02, 0.5)
    jax.tree.map(lambda x: x.block_until_ready(), g)

    return timeit(lambda: grad_fn(0.01, -0.02, 0.5), n_iter=100)


def bench_grad_naive_pointing():
    def loss(az, el):
        csl = CSL_jax.naive_az_el(
            jnp.array([1700000000.0]),
            jnp.array([az]),
            jnp.array([el]),
            site='act',
        )
        c = csl.coords()
        return c[0, 0]

    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1)))

    # Warmup
    g = grad_fn(1.0, 0.8)
    jax.tree.map(lambda x: x.block_until_ready(), g)

    return timeit(lambda: grad_fn(1.0, 0.8), n_iter=100)


def bench_grad_vmap_xieta(N):
    def loss(xi, eta):
        q = jax_quat.rotation_xieta(xi, eta, 0.0)
        xi_out, _, _ = jax_quat.decompose_xieta(q)
        return xi_out

    grad_fn = jax.jit(jax.vmap(jax.grad(loss, argnums=(0, 1))))
    xi = jnp.linspace(-0.05, 0.05, N)
    eta = jnp.linspace(-0.05, 0.05, N)

    # Warmup
    g = grad_fn(xi, eta)
    jax.tree.map(lambda x: x.block_until_ready(), g)

    return timeit(lambda: grad_fn(xi, eta))


# =========================================================================
# Main
# =========================================================================

def print_row3(name, t_so3g, t_jax_raw, t_jax_jit, extra=""):
    r_raw = t_so3g / t_jax_raw if t_jax_raw > 0 else float('inf')
    r_jit = t_so3g / t_jax_jit if t_jax_jit > 0 else float('inf')
    print(f"  {name:<30s}  {t_so3g*1000:>9.3f}  {t_jax_raw*1000:>9.3f}  {t_jax_jit*1000:>9.3f}  {r_jit:>8.2f}x  {extra}")


def print_row2(name, t_so3g, t_jax, extra=""):
    if t_so3g is None:
        print(f"  {name:<30s}  {'N/A':>9s}  {t_jax*1000:>9.3f}  {'N/A':>8s}  {extra}")
    else:
        r = t_so3g / t_jax if t_jax > 0 else float('inf')
        print(f"  {name:<30s}  {t_so3g*1000:>9.3f}  {t_jax*1000:>9.3f}  {r:>8.2f}x  {extra}")


def main():
    quick = '--quick' in sys.argv

    # Fix qpoint import if local source tree shadows installed package
    _fix_qpoint_import()

    print("=" * 85)
    print("so_pointjax.proj vs so3g — Performance Comparison")
    print("=" * 85)
    print(f"JAX backend: {jax.default_backend()}")
    print()

    # --- Quaternion operations ---
    if quick:
        quat_sizes = [1_000, 10_000]
    else:
        quat_sizes = [1_000, 10_000, 100_000]

    print("1. Quaternion Operations")
    print(f"  {'Operation':<30s}  {'so3g (ms)':>9s}  {'JAX (ms)':>9s}  {'JIT (ms)':>9s}  {'JIT/so3g':>8s}")
    print(f"  {'-'*75}")

    for N in quat_sizes:
        print(f"  --- N = {N:,} ---")

        t_s, t_r, t_j = bench_euler(N)
        print_row3("euler(2, angle)", t_s, t_r, t_j)

        t_s, t_r, t_j = bench_rotation_xieta(N)
        print_row3("rotation_xieta", t_s, t_r, t_j)

        t_s, t_r, t_j = bench_decompose_xieta(N)
        print_row3("decompose_xieta", t_s, t_r, t_j)

        t_s, t_r, t_j = bench_qmul(N)
        print_row3("quat multiply", t_s, t_r, t_j)

        t_s, t_r, t_j = bench_rotation_lonlat(N)
        print_row3("rotation_lonlat", t_s, t_r, t_j)

    # --- Pointing pipeline ---
    print()
    print("2. Pointing Pipeline")
    print(f"  {'Operation':<30s}  {'so3g (ms)':>9s}  {'JAX (ms)':>9s}  {'JIT (ms)':>9s}  {'JIT/so3g':>8s}")
    print(f"  {'-'*75}")

    if quick:
        pipe_sizes = [1_000, 10_000]
    else:
        pipe_sizes = [1_000, 10_000, 100_000]

    for N in pipe_sizes:
        print(f"  --- N = {N:,} ---")

        t_s, t_j, t_jit, max_diff = bench_naive_az_el(N)
        print_row3(f"naive_az_el", t_s, t_j, t_jit, f"(qdiff: {max_diff:.1e})")

        t_s, t_j = bench_for_lonlat_coords(N)
        print_row2(f"for_lonlat + coords", t_s, t_j)

    # High-precision pointing
    print()
    print("  High-precision pointing (az_el: C qpoint vs so_pointjax.qpoint):")
    hp_sizes = [1_000, 10_000] if quick else [1_000, 10_000, 100_000]
    for N in hp_sizes:
        t_s, t_j = bench_az_el(N)
        print_row2(f"az_el N={N:,}", t_s, t_j)

    # Coords with focal plane
    print()
    print("  Detector projection (naive_az_el + coords with focal plane):")
    if quick:
        coord_cases = [(1_000, 10), (1_000, 100)]
    else:
        coord_cases = [(1_000, 10), (1_000, 100), (10_000, 100), (10_000, 1000)]
    for N, n_det in coord_cases:
        t_s, t_j = bench_coords(N, n_det)
        print_row2(f"coords N={N:,} det={n_det}", t_s, t_j)

    # --- Gradient benchmarks (JAX-only) ---
    print()
    print("3. Gradient Computation (JAX-only — no so3g equivalent)")
    print(f"  {'Operation':<50s}  {'time':>12s}")
    print(f"  {'-'*65}")

    t = bench_grad_xieta()
    print(f"  {'grad(xieta roundtrip), single':<50s}  {t*1e6:>10.1f} µs")

    t = bench_grad_naive_pointing()
    print(f"  {'grad(naive_az_el → lon), single':<50s}  {t*1e6:>10.1f} µs")

    grad_sizes = [100, 1_000] if quick else [100, 1_000, 10_000]
    for N in grad_sizes:
        t = bench_grad_vmap_xieta(N)
        print(f"  {'vmap(grad(xieta)), N=' + str(N):<50s}  {t*1000:>10.3f} ms  ({t/N*1e6:.1f} µs/sample)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
