"""Benchmark: so_pointjax.qpoint (JAX vmap+jit) vs C QPoint for pointing pipeline.

Usage:
    python -m so_pointjax.qpoint.benchmarks.bench_pointing
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

from qpoint.qpoint_class import QPoint as CQPoint

from so_pointjax.qpoint._pointing import (
    azelpsi2bore_jit, azelpsi2bore_fast, precompute_times, precompute_corrections,
)
from so_pointjax.qpoint._quaternion import quat2radecpa, identity
from so_pointjax.qpoint._corrections import det_offset_quat
from so_pointjax.qpoint._pixel import ang2pix_nest


def generate_scan(n, lon=-44.65, lat=-89.99, ctime0=1700000000.0, duration=600.0):
    """Generate a realistic constant-elevation scan."""
    ctimes = np.linspace(ctime0, ctime0 + duration, n)
    az = np.linspace(0, 360, n)
    el = np.full(n, 45.0)
    return az, el, ctimes, lon, lat


def bench_c_qpoint(az, el, ctimes, lon, lat, n_iter=5):
    """Benchmark C QPoint: azel2bore + bore2radec."""
    n = len(az)
    cQ = CQPoint()
    q_off = np.array([1.0, 0.0, 0.0, 0.0])

    # Warmup
    cQ.azel2bore(az[:10], el[:10], np.zeros(10), np.zeros(10),
                 np.full(10, lon), np.full(10, lat), ctimes[:10])

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        q_bore = cQ.azel2bore(az, el, np.zeros(n), np.zeros(n),
                              np.full(n, lon), np.full(n, lat), ctimes)
        ra, dec, sin2psi, cos2psi = cQ.bore2radec(q_off, ctimes, q_bore)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times), ra, dec


def bench_jax_vmap(az, el, ctimes, lon, lat, n_iter=5):
    """Benchmark JAX vmap+jit: azelpsi2bore_jit + quat2radecpa."""
    times_data = precompute_times(ctimes)
    az_j = jnp.array(az)
    el_j = jnp.array(el)

    def forward(az_i, el_i, tt1, tt2, ut1_1, ut1_2):
        q = azelpsi2bore_jit(az_i, el_i, 0.0, lon, lat,
                             tt1, tt2, ut1_1, ut1_2)
        ra, dec, pa = quat2radecpa(q)
        return ra, dec

    forward_vmap = jax.jit(jax.vmap(forward))

    # Warmup / compile
    ra_j, dec_j = forward_vmap(
        az_j, el_j,
        times_data['tt1'], times_data['tt2'],
        times_data['ut1_1'], times_data['ut1_2'],
    )
    ra_j.block_until_ready()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        ra_j, dec_j = forward_vmap(
            az_j, el_j,
            times_data['tt1'], times_data['tt2'],
            times_data['ut1_1'], times_data['ut1_2'],
        )
        ra_j.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times), np.array(ra_j), np.array(dec_j)


def bench_jax_map(az, el, ctimes, lon, lat, n_iter=5):
    """Benchmark JAX lax.map+jit: sequential loop, no intermediate arrays."""
    times_data = precompute_times(ctimes)
    az_j = jnp.array(az)
    el_j = jnp.array(el)

    # Stack inputs into a single array for lax.map
    inputs = jnp.stack([
        az_j, el_j,
        times_data['tt1'], times_data['tt2'],
        times_data['ut1_1'], times_data['ut1_2'],
    ], axis=-1)  # (N, 6)

    def forward(x):
        q = azelpsi2bore_jit(x[0], x[1], 0.0, lon, lat,
                             x[2], x[3], x[4], x[5])
        ra, dec, pa = quat2radecpa(q)
        return jnp.array([ra, dec])

    forward_map = jax.jit(lambda xs: jax.lax.map(forward, xs))

    # Warmup / compile
    result = forward_map(inputs)
    result.block_until_ready()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = forward_map(inputs)
        result.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    ra_j = np.array(result[:, 0])
    dec_j = np.array(result[:, 1])
    return np.median(times), ra_j, dec_j


def bench_jax_scan(az, el, ctimes, lon, lat, n_iter=5):
    """Benchmark JAX lax.scan+jit: sequential with carry (no intermediate arrays)."""
    times_data = precompute_times(ctimes)
    az_j = jnp.array(az)
    el_j = jnp.array(el)

    inputs = jnp.stack([
        az_j, el_j,
        times_data['tt1'], times_data['tt2'],
        times_data['ut1_1'], times_data['ut1_2'],
    ], axis=-1)  # (N, 6)

    def scan_fn(carry, x):
        q = azelpsi2bore_jit(x[0], x[1], 0.0, lon, lat,
                             x[2], x[3], x[4], x[5])
        ra, dec, pa = quat2radecpa(q)
        return carry, jnp.array([ra, dec])

    forward_scan = jax.jit(lambda xs: jax.lax.scan(scan_fn, None, xs)[1])

    # Warmup / compile
    result = forward_scan(inputs)
    result.block_until_ready()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = forward_scan(inputs)
        result.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    ra_j = np.array(result[:, 0])
    dec_j = np.array(result[:, 1])
    return np.median(times), ra_j, dec_j


def bench_jax_fast(az, el, ctimes, lon, lat, n_iter=5):
    """Benchmark JAX vmap+jit with precomputed slow corrections."""
    corr = precompute_corrections(ctimes)
    az_j = jnp.array(az)
    el_j = jnp.array(el)

    def forward(az_i, el_i, tt1, tt2, ut1_1, ut1_2, npb_i, wobble_i, beta_i):
        q = azelpsi2bore_fast(az_i, el_i, 0.0, lon, lat,
                              tt1, tt2, ut1_1, ut1_2,
                              npb_i, wobble_i, beta_i)
        ra, dec, pa = quat2radecpa(q)
        return ra, dec

    forward_vmap = jax.jit(jax.vmap(forward))

    # Look up precomputed corrections per sample
    q_npb_per = corr['q_npb'][corr['npb_idx']]
    q_wobble_per = corr['q_wobble'][corr['npb_idx']]
    beta_per = corr['beta_earth'][corr['aber_idx']]

    # Warmup / compile
    ra_j, dec_j = forward_vmap(
        az_j, el_j,
        corr['tt1'], corr['tt2'],
        corr['ut1_1'], corr['ut1_2'],
        q_npb_per, q_wobble_per, beta_per,
    )
    ra_j.block_until_ready()

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        ra_j, dec_j = forward_vmap(
            az_j, el_j,
            corr['tt1'], corr['tt2'],
            corr['ut1_1'], corr['ut1_2'],
            q_npb_per, q_wobble_per, beta_per,
        )
        ra_j.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return np.median(times), np.array(ra_j), np.array(dec_j)


def bench_jax_compile(az, el, ctimes, lon, lat):
    """Measure JAX compilation time."""
    times_data = precompute_times(ctimes)
    az_j = jnp.array(az)
    el_j = jnp.array(el)

    def forward(az_i, el_i, tt1, tt2, ut1_1, ut1_2):
        q = azelpsi2bore_jit(az_i, el_i, 0.0, lon, lat,
                             tt1, tt2, ut1_1, ut1_2)
        ra, dec, pa = quat2radecpa(q)
        return ra, dec

    forward_vmap = jax.jit(jax.vmap(forward))

    t0 = time.perf_counter()
    ra_j, dec_j = forward_vmap(
        az_j, el_j,
        times_data['tt1'], times_data['tt2'],
        times_data['ut1_1'], times_data['ut1_2'],
    )
    ra_j.block_until_ready()
    t1 = time.perf_counter()

    return t1 - t0


def bench_jax_grad(n=1000):
    """Benchmark JAX gradient computation."""
    lon, lat = -44.65, -89.99
    ctime = 1700000000.0

    # Use _jit variant so grad can trace through the full pipeline
    times_data = precompute_times(np.array([ctime]))

    def ra_from_azel(az, el):
        q = azelpsi2bore_jit(az, el, 0.0, lon, lat,
                             times_data['tt1'][0], times_data['tt2'][0],
                             times_data['ut1_1'][0], times_data['ut1_2'][0])
        ra, _, _ = quat2radecpa(q)
        return ra

    grad_fn = jax.jit(jax.grad(ra_from_azel, argnums=(0, 1)))

    # Warmup
    g = grad_fn(180.0, 45.0)
    jax.tree.map(lambda x: x.block_until_ready(), g)

    t0 = time.perf_counter()
    for _ in range(n):
        g = grad_fn(180.0, 45.0)
    jax.tree.map(lambda x: x.block_until_ready(), g)
    t1 = time.perf_counter()

    return (t1 - t0) / n


def bench_jax_grad_vmap(sizes, n_iter=5):
    """Benchmark vectorized gradient computation."""
    lon, lat = -44.65, -89.99

    results = {}
    for n in sizes:
        az, el, ctimes, _, _ = generate_scan(n, lon=lon, lat=lat)
        times_data = precompute_times(ctimes)
        az_j, el_j = jnp.array(az), jnp.array(el)

        def forward_ra(az_i, el_i, tt1, tt2, ut1_1, ut1_2):
            q = azelpsi2bore_jit(az_i, el_i, 0.0, lon, lat,
                                 tt1, tt2, ut1_1, ut1_2)
            ra, _, _ = quat2radecpa(q)
            return ra

        grad_forward = jax.jit(jax.vmap(jax.grad(forward_ra, argnums=(0, 1))))

        # Warmup
        g = grad_forward(az_j, el_j,
                         times_data['tt1'], times_data['tt2'],
                         times_data['ut1_1'], times_data['ut1_2'])
        jax.tree.map(lambda x: x.block_until_ready(), g)

        times = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            g = grad_forward(az_j, el_j,
                             times_data['tt1'], times_data['tt2'],
                             times_data['ut1_1'], times_data['ut1_2'])
            jax.tree.map(lambda x: x.block_until_ready(), g)
            t1 = time.perf_counter()
            times.append(t1 - t0)

        results[n] = np.median(times)

    return results


def main():
    print("=" * 70)
    print("so_pointjax.qpoint vs C QPoint — Pointing Pipeline Benchmark")
    print("=" * 70)
    print(f"JAX backend: {jax.default_backend()}")
    print()

    # Pre-warm: force lazy table loading before any JIT tracing
    from so_pointjax.qpoint._pointing import azel2bore
    _ = azel2bore(180.0, 45.0, -44.65, -89.99, 1700000000.0)

    import sys
    if '--quick' in sys.argv:
        sizes = [100, 1_000, 10_000]
    elif '--medium' in sys.argv:
        sizes = [100, 1_000, 10_000, 100_000]
    else:
        sizes = [100, 1_000, 10_000, 100_000, 1_000_000]

    # --- Forward pipeline ---
    header = f"{'N':>10s}  {'C (ms)':>10s}  {'vmap (ms)':>10s}  {'fast (ms)':>10s}  {'vmap/C':>7s}  {'fast/C':>7s}"
    print(header)
    print("-" * len(header))

    for n in sizes:
        az, el, ctimes, lon, lat = generate_scan(n)

        t_c, ra_c, dec_c = bench_c_qpoint(az, el, ctimes, lon, lat)
        t_v, ra_v, dec_v = bench_jax_vmap(az, el, ctimes, lon, lat)
        t_f, ra_f, dec_f = bench_jax_fast(az, el, ctimes, lon, lat)

        # Check fast accuracy
        ra_diff = np.abs((ra_f % 360) - (ra_c % 360))
        ra_diff = np.minimum(ra_diff, 360 - ra_diff)

        r_v = t_c / t_v if t_v > 0 else float('inf')
        r_f = t_c / t_f if t_f > 0 else float('inf')
        print(f"{n:>10d}  {t_c*1000:>10.2f}  {t_v*1000:>10.2f}  {t_f*1000:>10.2f}  {r_v:>6.2f}x  {r_f:>6.2f}x  (max RA diff: {np.max(ra_diff):.4f}°)")

    quick = '--quick' in sys.argv

    if not quick:
        # --- Compilation time ---
        print()
        print("Compilation time (first call, includes tracing + XLA compile):")
        for n in [1_000, 100_000]:
            az, el, ctimes, lon, lat = generate_scan(n)
            t_compile = bench_jax_compile(az, el, ctimes, lon, lat)
            print(f"  N={n:>7d}: {t_compile*1000:.0f} ms")

        # --- Gradient ---
        print()
        print("Gradient computation (JAX-only capability, no C equivalent):")
        t_grad = bench_jax_grad()
        print(f"  Single-sample d(ra)/d(az,el): {t_grad*1e6:.1f} µs/call")

        grad_sizes = [1_000, 10_000, 100_000]
        grad_results = bench_jax_grad_vmap(grad_sizes)
        print()
        print(f"  {'N':>10s}  {'vmap(grad) (ms)':>16s}  {'per sample (µs)':>16s}")
        print(f"  {'-'*50}")
        for n, t in grad_results.items():
            print(f"  {n:>10d}  {t*1000:>16.2f}  {t/n*1e6:>16.2f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
