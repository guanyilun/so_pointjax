"""Finite-difference gradient correctness checks for key differentiable chains.

Verifies that jax.grad produces gradients consistent with numerical
finite-difference approximations across the major function categories.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import so_pointjax.erfa


def check_grad(f, *args, eps=1e-7, atol=1e-5, rtol=1e-5):
    """Check jax.grad against centered finite differences for each arg."""
    for i in range(len(args)):
        # JAX analytical gradient
        grad_fn = jax.grad(f, argnums=i)
        g_jax = float(grad_fn(*args))

        # Centered finite difference
        args_plus = list(args)
        args_minus = list(args)
        args_plus[i] = args[i] + eps
        args_minus[i] = args[i] - eps
        g_fd = (f(*args_plus) - f(*args_minus)) / (2 * eps)

        np.testing.assert_allclose(
            g_jax, g_fd, atol=atol, rtol=rtol,
            err_msg=f"Gradient mismatch for arg {i}: jax={g_jax}, fd={g_fd}")


# ============================================================================
# Vector / Matrix operations
# ============================================================================


class TestVectorGrad:
    def test_grad_s2c(self):
        def f(theta, phi):
            return jnp.sum(so_pointjax.erfa.s2c(theta, phi))
        check_grad(f, 1.0, 0.5)

    def test_grad_c2s(self):
        def f(px, py, pz):
            theta, phi = so_pointjax.erfa.c2s(jnp.array([px, py, pz]))
            return theta + phi
        check_grad(f, 1.0, 2.0, 3.0)

    def test_grad_rxp(self):
        def f(angle):
            r = so_pointjax.erfa.rx(angle, so_pointjax.erfa.ir())
            p = jnp.array([1.0, 2.0, 3.0])
            return jnp.sum(so_pointjax.erfa.rxp(r, p))
        check_grad(f, 0.5)

    def test_grad_sepp(self):
        def f(ax, ay, az):
            a = jnp.array([ax, ay, az])
            b = jnp.array([0.0, 1.0, 0.0])
            return so_pointjax.erfa.sepp(a, b)
        check_grad(f, 1.0, 0.1, 0.1)

    def test_grad_rv2m(self):
        def f(wx, wy, wz):
            r = so_pointjax.erfa.rv2m(jnp.array([wx, wy, wz]))
            return jnp.sum(r)
        check_grad(f, 0.1, 0.2, 0.3)


# ============================================================================
# Angle operations
# ============================================================================


class TestAnglesGrad:
    def test_grad_anp(self):
        check_grad(so_pointjax.erfa.anp, 4.0)

    def test_grad_anpm(self):
        check_grad(so_pointjax.erfa.anpm, 4.0)


# ============================================================================
# Calendar / Epoch (differentiable subset)
# ============================================================================


class TestCalendarGrad:
    def test_grad_epb(self):
        def f(d2):
            return so_pointjax.erfa.epb(2451545.0, d2)
        check_grad(f, 0.0)

    def test_grad_epj(self):
        def f(d2):
            return so_pointjax.erfa.epj(2451545.0, d2)
        check_grad(f, 0.0)


# ============================================================================
# Time scales (differentiable subset)
# ============================================================================


class TestTimeGrad:
    def test_grad_era00(self):
        def f(d2):
            return so_pointjax.erfa.era00(2400000.5, d2)
        check_grad(f, 54388.0)

    def test_grad_gmst06(self):
        def f(d2):
            return so_pointjax.erfa.gmst06(2400000.5, d2, 2400000.5, d2)
        check_grad(f, 54388.0)

    def test_grad_taitt(self):
        def f(d2):
            _, t2 = so_pointjax.erfa.taitt(2453750.5, d2)
            return t2  # Only check d2 part to avoid FD precision loss
        check_grad(f, 0.892482639)

    def test_grad_tttdb(self):
        def f(dtr):
            _, t2 = so_pointjax.erfa.tttdb(2453750.5, 0.892855139, dtr)
            return t2
        check_grad(f, -0.000201)


# ============================================================================
# Precession / Nutation
# ============================================================================


class TestPrecNutGrad:
    def test_grad_obl06(self):
        def f(d2):
            return so_pointjax.erfa.obl06(2400000.5, d2)
        check_grad(f, 53736.0)

    def test_grad_nut00b(self):
        def f(d2):
            dpsi, deps = so_pointjax.erfa.nut00b(2400000.5, d2)
            return dpsi + deps
        check_grad(f, 53736.0)

    def test_grad_pnm06a(self):
        def f(d2):
            return jnp.sum(so_pointjax.erfa.pnm06a(2400000.5, d2))
        check_grad(f, 53736.0)

    def test_grad_s06a(self):
        def f(d2):
            return so_pointjax.erfa.s06a(2400000.5, d2)
        check_grad(f, 53736.0)

    def test_grad_gst06a(self):
        def f(d2):
            return so_pointjax.erfa.gst06a(2400000.5, d2, 2400000.5, d2)
        check_grad(f, 53736.0)

    def test_grad_fundamental_args(self):
        for name in ['fal03', 'falp03', 'faf03', 'fad03', 'faom03']:
            fn = getattr(so_pointjax.erfa, name)
            check_grad(fn, 0.8)


# ============================================================================
# Ephemerides
# ============================================================================


class TestEphemGrad:
    def test_grad_epv00(self):
        def f(d2):
            pvh, pvb = so_pointjax.erfa.epv00(2400000.5, d2)
            return jnp.sum(pvh) + jnp.sum(pvb)
        check_grad(f, 53411.525)

    def test_grad_moon98(self):
        def f(d2):
            pv = so_pointjax.erfa.moon98(2400000.5, d2)
            return jnp.sum(pv)
        check_grad(f, 43999.9)

    def test_grad_plan94_mercury(self):
        def f(d2):
            pv, _ = so_pointjax.erfa.plan94(2400000.5, d2, 1)
            return jnp.sum(pv)
        check_grad(f, 43999.9)

    def test_grad_plan94_jupiter(self):
        def f(d2):
            pv, _ = so_pointjax.erfa.plan94(2400000.5, d2, 5)
            return jnp.sum(pv)
        check_grad(f, 43999.9)


# ============================================================================
# Geodetic transforms
# ============================================================================


class TestGeodeticGrad:
    def test_grad_gd2gce(self):
        a, f = so_pointjax.erfa.eform(1)
        def fn(elong, phi, height):
            return jnp.sum(so_pointjax.erfa.gd2gce(a, f, elong, phi, height))
        # Use smaller eps for Earth-radius-scale values
        check_grad(fn, 0.5, 1.0, 1000.0, eps=1e-4, atol=1e-2, rtol=1e-2)

    def test_grad_gc2gde(self):
        a, f = so_pointjax.erfa.eform(1)
        xyz = so_pointjax.erfa.gd2gc(1, 0.5, 1.0, 1000.0)
        def fn(x, y, z):
            elong, phi, height = so_pointjax.erfa.gc2gde(a, f, jnp.array([x, y, z]))
            return elong + phi + height
        # Large input values (~6e6) need scaled eps
        check_grad(fn, float(xyz[0]), float(xyz[1]), float(xyz[2]),
                   eps=1.0, atol=1e-3, rtol=1e-3)

    def test_grad_gd2gc_gc2gd_roundtrip(self):
        """Gradient through gd2gc -> gc2gd roundtrip."""
        def f(elong, phi, height):
            xyz = so_pointjax.erfa.gd2gce(6378137.0, 1.0/298.257223563, elong, phi, height)
            e2, p2, h2 = so_pointjax.erfa.gc2gde(6378137.0, 1.0/298.257223563, xyz)
            return e2 + p2 + h2
        check_grad(f, 0.5, 1.0, 1000.0, eps=1e-4, atol=1e-2, rtol=1e-2)


# ============================================================================
# Coordinate frame transforms
# ============================================================================


class TestFramesGrad:
    def test_grad_ae2hd(self):
        def f(az, el, phi):
            ha, dec = so_pointjax.erfa.ae2hd(az, el, phi)
            return ha + dec
        check_grad(f, 1.0, 0.5, 0.8)

    def test_grad_hd2ae(self):
        def f(ha, dec, phi):
            az, el = so_pointjax.erfa.hd2ae(ha, dec, phi)
            return az + el
        check_grad(f, 1.0, 0.5, 0.8)

    def test_grad_icrs2g(self):
        def f(dr, dd):
            dl, db = so_pointjax.erfa.icrs2g(dr, dd)
            return dl + db
        check_grad(f, 1.0, 0.5)

    def test_grad_g2icrs(self):
        def f(dl, db):
            dr, dd = so_pointjax.erfa.g2icrs(dl, db)
            return dr + dd
        check_grad(f, 1.0, 0.5)

    def test_grad_eqec06(self):
        def f(dr, dd):
            dl, db = so_pointjax.erfa.eqec06(2400000.5, 53736.0, dr, dd)
            return dl + db
        check_grad(f, 1.0, 0.5)

    def test_grad_eceq06(self):
        def f(dl, db):
            dr, dd = so_pointjax.erfa.eceq06(2400000.5, 53736.0, dl, db)
            return dr + dd
        check_grad(f, 1.0, 0.5)

    def test_grad_ecm06(self):
        def f(d2):
            return jnp.sum(so_pointjax.erfa.ecm06(2400000.5, d2))
        check_grad(f, 53736.0)

    def test_grad_ltp(self):
        def f(epj):
            return jnp.sum(so_pointjax.erfa.ltp(epj))
        check_grad(f, 1500.0)


# ============================================================================
# Gnomonic projections
# ============================================================================


class TestGnomonicGrad:
    def test_grad_tpxes(self):
        def f(a, b):
            xi, eta, _ = so_pointjax.erfa.tpxes(a, b, 1.001, 0.501)
            return xi + eta
        check_grad(f, 1.0, 0.5)

    def test_grad_tpsts(self):
        def f(xi, eta):
            a, b = so_pointjax.erfa.tpsts(xi, eta, 1.001, 0.501)
            return a + b
        check_grad(f, 0.001, 0.002)


# ============================================================================
# Key end-to-end differentiable chains
# ============================================================================


class TestEndToEndGrad:
    """Test gradients through multi-step astronomical transform chains."""

    def test_grad_icrs_to_cirs(self):
        """Gradient of ICRS->CIRS transform w.r.t. catalog position."""
        def f(ra, dec):
            ri, di, _ = so_pointjax.erfa.atci13(ra, dec, 0.0, 0.0, 0.0, 0.0,
                                         2456165.5, 0.401182685)
            return ri + di
        check_grad(f, 2.71, 0.174)

    def test_grad_icrs_to_cirs_wrt_date(self):
        """Gradient of ICRS->CIRS transform w.r.t. date."""
        def f(d2):
            ri, di, _ = so_pointjax.erfa.atci13(2.71, 0.174, 0.0, 0.0, 0.0, 0.0,
                                         2456165.5, d2)
            return ri + di
        check_grad(f, 0.401182685)

    def test_grad_cirs_to_icrs(self):
        """Gradient of CIRS->ICRS inverse transform."""
        def f(ri, di):
            rc, dc, _ = so_pointjax.erfa.atic13(ri, di, 2456165.5, 0.401182685)
            return rc + dc
        check_grad(f, 2.71, 0.174)

    def test_grad_icrs_galactic_roundtrip(self):
        """Gradient through ICRS->Galactic->ICRS roundtrip."""
        def f(ra, dec):
            gl, gb = so_pointjax.erfa.icrs2g(ra, dec)
            ra2, dec2 = so_pointjax.erfa.g2icrs(gl, gb)
            return ra2 + dec2
        check_grad(f, 1.0, 0.5)

    def test_grad_horizon_roundtrip(self):
        """Gradient through horizon<->equatorial roundtrip."""
        def f(az, el):
            ha, dec = so_pointjax.erfa.ae2hd(az, el, 0.8)
            az2, el2 = so_pointjax.erfa.hd2ae(ha, dec, 0.8)
            return az2 + el2
        check_grad(f, 1.0, 0.5)

    def test_grad_ecliptic_roundtrip(self):
        """Gradient through equatorial->ecliptic->equatorial roundtrip."""
        def f(ra, dec):
            dl, db = so_pointjax.erfa.eqec06(2400000.5, 53736.0, ra, dec)
            ra2, dec2 = so_pointjax.erfa.eceq06(2400000.5, 53736.0, dl, db)
            return ra2 + dec2
        check_grad(f, 1.0, 0.5)

    def test_grad_earth_position_velocity(self):
        """Gradient of Earth ephemeris w.r.t. date."""
        def f(d2):
            pvh, pvb = so_pointjax.erfa.epv00(2400000.5, d2)
            return jnp.sum(pvh[0])  # Earth heliocentric position
        check_grad(f, 53411.525)

    def test_grad_precession_nutation_chain(self):
        """Gradient through full precession-nutation matrix computation."""
        def f(d2):
            rnpb = so_pointjax.erfa.pnm06a(2400000.5, d2)
            x, y = so_pointjax.erfa.bpn2xy(rnpb)
            s = so_pointjax.erfa.s06(2400000.5, d2, x, y)
            return s
        check_grad(f, 53736.0)

    def test_grad_gnomonic_roundtrip(self):
        """Gradient through tangent-plane projection roundtrip."""
        def f(a, b):
            xi, eta, _ = so_pointjax.erfa.tpxes(a, b, 1.001, 0.501)
            a2, b2 = so_pointjax.erfa.tpsts(xi, eta, 1.001, 0.501)
            return a2 + b2
        check_grad(f, 1.0, 0.5)
