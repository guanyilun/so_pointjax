"""Profile which correction steps are expensive in the JAX pipeline."""

import time
import numpy as np
import jax
import jax.numpy as jnp

from so_pointjax.qpoint._pointing import precompute_times, azelpsi2bore_jit
from so_pointjax.qpoint._quaternion import quat2radecpa, identity, mul, r3
from so_pointjax.qpoint._corrections import (
    npb_quat, erot_quat, wobble_quat, lonlat_quat,
    azelpsi_quat, aberration, earth_orbital_beta, diurnal_aberration_beta,
)

# Pre-warm tables
from so_pointjax.qpoint._pointing import azel2bore
_ = azel2bore(180.0, 45.0, -44.65, -89.99, 1700000000.0)


def bench_step(name, fn, args, n_iter=5):
    """Benchmark a single vmap'd step."""
    f = jax.jit(jax.vmap(fn))
    result = f(*args)
    jax.tree.map(lambda x: x.block_until_ready(), result)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        result = f(*args)
        jax.tree.map(lambda x: x.block_until_ready(), result)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def main():
    N = 10_000
    lon, lat = -44.65, -89.99
    ctimes = np.linspace(1700000000.0, 1700000600.0, N)
    az = jnp.linspace(0, 360, N)
    el = jnp.full(N, 45.0)
    times = precompute_times(ctimes)
    tt1, tt2 = times['tt1'], times['tt2']
    ut1_1, ut1_2 = times['ut1_1'], times['ut1_2']

    print(f"Step breakdown at N={N}")
    print(f"{'Step':<30s}  {'Time (ms)':>10s}  {'% of total':>10s}")
    print("-" * 55)

    results = {}

    # 1. azelpsi_quat
    t = bench_step("azelpsi_quat", lambda a, e: azelpsi_quat(a, e, 0.0, 0.0, 0.0), (az, el))
    results['azelpsi_quat'] = t

    # 2. lonlat_quat (scalar, but measure vmap overhead)
    t = bench_step("lonlat_quat", lambda _: lonlat_quat(lon, lat), (az,))  # dummy input
    results['lonlat_quat'] = t

    # 3. wobble_quat
    t = bench_step("wobble_quat", lambda t1, t2: wobble_quat(t1, t2, 0.0, 0.0), (tt1, tt2))
    results['wobble_quat'] = t

    # 4. erot_quat
    t = bench_step("erot_quat", lambda u1, u2: erot_quat(u1, u2), (ut1_1, ut1_2))
    results['erot_quat'] = t

    # 5. npb_quat (the expensive ERFA call)
    t = bench_step("npb_quat", lambda t1, t2: npb_quat(t1, t2, accuracy=1), (tt1, tt2))
    results['npb_quat'] = t

    # 6. earth_orbital_beta (epv00)
    t = bench_step("earth_orbital_beta", lambda t1, t2: earth_orbital_beta(t1, t2), (tt1, tt2))
    results['earth_orbital_beta'] = t

    # 7. diurnal aberration
    t = bench_step("diurnal_aber", lambda _: diurnal_aberration_beta(lat), (az,))
    results['diurnal_aber'] = t

    # 8. Full pipeline
    t = bench_step("FULL pipeline",
                   lambda a, e, t1, t2, u1, u2: azelpsi2bore_jit(a, e, 0.0, lon, lat, t1, t2, u1, u2),
                   (az, el, tt1, tt2, ut1_1, ut1_2))
    results['FULL pipeline'] = t

    # 9. Full pipeline + quat2radecpa
    def full_with_radec(a, e, t1, t2, u1, u2):
        q = azelpsi2bore_jit(a, e, 0.0, lon, lat, t1, t2, u1, u2)
        return quat2radecpa(q)
    t = bench_step("FULL + radecpa",
                   full_with_radec,
                   (az, el, tt1, tt2, ut1_1, ut1_2))
    results['FULL + radecpa'] = t

    total = results['FULL + radecpa']
    for name, t in results.items():
        pct = t / total * 100 if total > 0 else 0
        print(f"{name:<30s}  {t*1000:>10.2f}  {pct:>9.1f}%")


if __name__ == "__main__":
    main()
