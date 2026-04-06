"""Tests for coordinate frame transforms (Phase 5)."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import so_pointjax.erfa


# ============================================================================
# Horizon / Equatorial
# ============================================================================


class TestAe2hd:
    def test_basic(self):
        h, d = so_pointjax.erfa.ae2hd(5.5, 1.1, 0.7)
        assert abs(h - 0.5933291115507309663) < 1e-14
        assert abs(d - 0.9613934761647817620) < 1e-14


class TestHd2ae:
    def test_basic(self):
        az, el = so_pointjax.erfa.hd2ae(1.1, 1.2, 0.3)
        assert abs(az - 5.916889243730066194) < 1e-13
        assert abs(el - 0.4472186304990486228) < 1e-14


class TestHd2pa:
    def test_basic(self):
        pa = so_pointjax.erfa.hd2pa(1.1, 1.2, 0.3)
        assert abs(pa - 1.906227428001995580) < 1e-13


# ============================================================================
# Galactic / ICRS
# ============================================================================


class TestIcrs2g:
    def test_basic(self):
        dl, db = so_pointjax.erfa.icrs2g(5.9338074302227188048671087,
                                  -1.1784870613579944551540570)
        assert abs(dl - 5.5850536063818546461558) < 1e-14
        assert abs(db - (-0.7853981633974483096157)) < 1e-14


class TestG2icrs:
    def test_basic(self):
        dr, dd = so_pointjax.erfa.g2icrs(5.5850536063818546461558105,
                                  -0.7853981633974483096156608)
        assert abs(dr - 5.9338074302227188048671) < 1e-14
        assert abs(dd - (-1.1784870613579944551541)) < 1e-14


# ============================================================================
# Geodetic (gc2gd, gc2gde)
# ============================================================================


class TestGc2gde:
    def test_basic(self):
        xyz = jnp.array([2e6, 3e6, 5.244e6])
        e, p, h = so_pointjax.erfa.gc2gde(6378136.0, 0.0033528, xyz)
        assert abs(e - 0.9827937232473290680) < 1e-14
        assert abs(p - 0.9716018377570411532) < 1e-14
        assert abs(h - 332.36862495764397) < 1e-8


class TestGc2gd:
    def test_wgs84(self):
        xyz = jnp.array([2e6, 3e6, 5.244e6])
        e, p, h = so_pointjax.erfa.gc2gd(1, xyz)
        assert abs(e - 0.9827937232473290680) < 1e-14
        assert abs(p - 0.97160184819075459) < 1e-14
        assert abs(h - 331.4172461426059892) < 1e-8

    def test_grs80(self):
        xyz = jnp.array([2e6, 3e6, 5.244e6])
        e, p, h = so_pointjax.erfa.gc2gd(2, xyz)
        assert abs(e - 0.9827937232473290680) < 1e-14
        assert abs(p - 0.97160184820607853) < 1e-14
        assert abs(h - 331.41731754844348) < 1e-8

    def test_wgs72(self):
        xyz = jnp.array([2e6, 3e6, 5.244e6])
        e, p, h = so_pointjax.erfa.gc2gd(3, xyz)
        assert abs(e - 0.9827937232473290680) < 1e-14
        assert abs(p - 0.9716018181101511937) < 1e-14
        assert abs(h - 333.2770726130318123) < 1e-8


# ============================================================================
# Ecliptic coordinate transforms (IAU 2006)
# ============================================================================


class TestEcm06:
    def test_basic(self):
        rm = so_pointjax.erfa.ecm06(2456165.5, 0.401182685)
        assert abs(rm[0, 0] - 0.9999952427708701137) < 1e-14
        assert abs(rm[0, 1] - (-0.2829062057663042347e-2)) < 1e-14
        assert abs(rm[0, 2] - (-0.1229163741100017629e-2)) < 1e-14
        assert abs(rm[1, 0] - 0.3084546876908653562e-2) < 1e-14
        assert abs(rm[1, 1] - 0.9174891871550392514) < 1e-14
        assert abs(rm[1, 2] - 0.3977487611849338124) < 1e-14
        assert abs(rm[2, 0] - 0.2488512951527405928e-5) < 1e-14
        assert abs(rm[2, 1] - (-0.3977506604161195467)) < 1e-14
        assert abs(rm[2, 2] - 0.9174935488232863071) < 1e-14


class TestEqec06:
    def test_basic(self):
        dl, db = so_pointjax.erfa.eqec06(1234.5, 2440000.5, 1.234, 0.987)
        assert abs(dl - 1.342509918994654619) < 1e-14
        assert abs(db - 0.5926215259704608132) < 1e-14


class TestEceq06:
    def test_basic(self):
        dr, dd = so_pointjax.erfa.eceq06(2456165.5, 0.401182685, 5.1, -0.9)
        assert abs(dr - 5.533459733613627767) < 1e-14
        assert abs(dd - (-1.246542932554480576)) < 1e-14


# ============================================================================
# Long-term precession
# ============================================================================


class TestLtpecl:
    def test_basic(self):
        vec = so_pointjax.erfa.ltpecl(-1500.0)
        assert abs(vec[0] - 0.4768625676477096525e-3) < 1e-14
        assert abs(vec[1] - (-0.4052259533091875112)) < 1e-14
        assert abs(vec[2] - 0.9142164401096448012) < 1e-14


class TestLtpequ:
    def test_basic(self):
        veq = so_pointjax.erfa.ltpequ(-2500.0)
        assert abs(veq[0] - (-0.3586652560237326659)) < 1e-14
        assert abs(veq[1] - (-0.1996978910771128475)) < 1e-14
        assert abs(veq[2] - 0.9118552442250819624) < 1e-14


class TestLtecm:
    def test_basic(self):
        rm = so_pointjax.erfa.ltecm(-3000.0)
        assert abs(rm[0, 0] - 0.3564105644859788825) < 1e-14
        assert abs(rm[0, 1] - 0.8530575738617682284) < 1e-14
        assert abs(rm[0, 2] - 0.3811355207795060435) < 1e-14
        assert abs(rm[1, 0] - (-0.9343283469640709942)) < 1e-14
        assert abs(rm[1, 1] - 0.3247830597681745976) < 1e-14
        assert abs(rm[1, 2] - 0.1467872751535940865) < 1e-14
        assert abs(rm[2, 0] - 0.1431636191201167793e-2) < 1e-14
        assert abs(rm[2, 1] - (-0.4084222566960599342)) < 1e-14
        assert abs(rm[2, 2] - 0.9127919865189030899) < 1e-14


class TestLtp:
    def test_basic(self):
        rp = so_pointjax.erfa.ltp(1666.666)
        assert abs(rp[0, 0] - 0.9967044141159213819) < 1e-14
        assert abs(rp[0, 1] - 0.7437801893193210840e-1) < 1e-14
        assert abs(rp[0, 2] - 0.3237624409345603401e-1) < 1e-14
        assert abs(rp[1, 0] - (-0.7437802731819618167e-1)) < 1e-14
        assert abs(rp[1, 1] - 0.9972293894454533070) < 1e-14
        assert abs(rp[1, 2] - (-0.1205768842723593346e-2)) < 1e-14
        assert abs(rp[2, 0] - (-0.3237622482766575399e-1)) < 1e-14
        assert abs(rp[2, 1] - (-0.1206286039697609008e-2)) < 1e-14
        assert abs(rp[2, 2] - 0.9994750246704010914) < 1e-14


class TestLtpb:
    def test_basic(self):
        rpb = so_pointjax.erfa.ltpb(1666.666)
        assert abs(rpb[0, 0] - 0.9967044167723271851) < 1e-14
        assert abs(rpb[0, 1] - 0.7437794731203340345e-1) < 1e-14
        assert abs(rpb[0, 2] - 0.3237632684841625547e-1) < 1e-14
        assert abs(rpb[1, 0] - (-0.7437795663437177152e-1)) < 1e-14
        assert abs(rpb[1, 1] - 0.9972293947500013666) < 1e-14
        assert abs(rpb[1, 2] - (-0.1205741865911243235e-2)) < 1e-14
        assert abs(rpb[2, 0] - (-0.3237630543224664992e-1)) < 1e-14
        assert abs(rpb[2, 1] - (-0.1206316791076485295e-2)) < 1e-14
        assert abs(rpb[2, 2] - 0.9994750220222438819) < 1e-14


class TestLteqec:
    def test_basic(self):
        dl, db = so_pointjax.erfa.lteqec(-1500.0, 1.234, 0.987)
        assert abs(dl - 0.5039483649047114859) < 1e-14
        assert abs(db - 0.5848534459726224882) < 1e-14


# ============================================================================
# FK5 <-> Hipparcos
# ============================================================================


class TestFk5hip:
    def test_basic(self):
        r5h, s5h = so_pointjax.erfa.fk5hip()
        assert abs(r5h[0, 0] - 0.9999999999999928638) < 1e-14
        assert abs(r5h[0, 1] - 0.1110223351022919694e-6) < 1e-17
        assert abs(r5h[0, 2] - 0.4411803962536558154e-7) < 1e-17
        assert abs(r5h[1, 0] - (-0.1110223308458746430e-6)) < 1e-17
        assert abs(r5h[1, 1] - 0.9999999999999891830) < 1e-14
        assert abs(r5h[1, 2] - (-0.9647792498984142358e-7)) < 1e-17
        assert abs(r5h[2, 0] - (-0.4411805033656962252e-7)) < 1e-17
        assert abs(r5h[2, 1] - 0.9647792009175314354e-7) < 1e-17
        assert abs(r5h[2, 2] - 0.9999999999999943728) < 1e-14
        assert abs(s5h[0] - (-0.1454441043328607981e-8)) < 1e-17
        assert abs(s5h[1] - 0.2908882086657215962e-8) < 1e-17
        assert abs(s5h[2] - 0.3393695767766751955e-8) < 1e-17


class TestFk52h:
    def test_basic(self):
        rh, dh, drh, ddh, pxh, rvh = so_pointjax.erfa.fk52h(
            1.76779433, -0.2917517103,
            -1.91851572e-7, -5.8468475e-6,
            0.379210, -7.6)
        assert abs(rh - 1.767794226299947632) < 1e-14
        assert abs(dh - (-0.2917516070530391757)) < 1e-14
        assert abs(drh - (-0.1961874125605721270e-6)) < 1e-19
        assert abs(ddh - (-0.58459905176693911e-5)) < 1e-19
        assert abs(pxh - 0.37921) < 1e-14
        assert abs(rvh - (-7.6000000940000254)) < 1e-11


class TestH2fk5:
    def test_basic(self):
        r5, d5, dr5, dd5, px5, rv5 = so_pointjax.erfa.h2fk5(
            1.767794352, -0.2917512594,
            -2.76413026e-6, -5.92994449e-6,
            0.379210, -7.6)
        assert abs(r5 - 1.767794455700065506) < 1e-13
        assert abs(d5 - (-0.2917513626469638890)) < 1e-13
        assert abs(dr5 - (-0.27597945024511204e-5)) < 1e-18
        assert abs(dd5 - (-0.59308014093262838e-5)) < 1e-18
        assert abs(px5 - 0.37921) < 1e-13
        assert abs(rv5 - (-7.6000001309071126)) < 1e-11


class TestFk5hz:
    def test_basic(self):
        rh, dh = so_pointjax.erfa.fk5hz(1.76779433, -0.2917517103,
                                  2400000.5, 54479.0)
        assert abs(rh - 1.767794191464423978) < 1e-12
        assert abs(dh - (-0.2917516001679884419)) < 1e-12


class TestHfk5z:
    def test_basic(self):
        r5, d5, dr5, dd5 = so_pointjax.erfa.hfk5z(1.767794352, -0.2917512594,
                                             2400000.5, 54479.0)
        assert abs(r5 - 1.767794490535581026) < 1e-13
        assert abs(d5 - (-0.2917513695320114258)) < 1e-14
        assert abs(dr5 - 0.4335890983539243029e-8) < 1e-22
        assert abs(dd5 - (-0.8569648841237745902e-9)) < 1e-23


# ============================================================================
# FK4 <-> FK5
# ============================================================================


class TestFk425:
    def test_basic(self):
        r2000, d2000, dr2000, dd2000, p2000, v2000 = so_pointjax.erfa.fk425(
            0.07626899753879587532, -1.137405378399605780,
            0.1973749217849087460e-4, 0.5659714913272723189e-5,
            0.134, 8.7)
        assert abs(r2000 - 0.08757989933556446040) < 1e-14
        assert abs(d2000 - (-1.132279113042091895)) < 1e-12
        assert abs(dr2000 - 0.1953670614474396139e-4) < 1e-17
        assert abs(dd2000 - 0.5637686678659640164e-5) < 1e-18
        assert abs(p2000 - 0.1339919950582767871) < 1e-13
        assert abs(v2000 - 8.736999669183529069) < 1e-12


class TestFk524:
    def test_basic(self):
        r1950, d1950, dr1950, dd1950, p1950, v1950 = so_pointjax.erfa.fk524(
            0.8723503576487275595, -0.7517076365138887672,
            0.2019447755430472323e-4, 0.3541563940505160433e-5,
            0.1559, 86.87)
        assert abs(r1950 - 0.8636359659799603487) < 1e-13
        assert abs(d1950 - (-0.7550281733160843059)) < 1e-13
        assert abs(dr1950 - 0.2023628192747172486e-4) < 1e-17
        assert abs(dd1950 - 0.3624459754935334718e-5) < 1e-18
        assert abs(p1950 - 0.1560079963299390241) < 1e-13
        assert abs(v1950 - 86.79606353469163751) < 1e-11


class TestFk45z:
    def test_basic(self):
        r2000, d2000 = so_pointjax.erfa.fk45z(
            0.01602284975382960982, -0.1164347929099906024,
            1954.677617625256806)
        assert abs(r2000 - 0.02719295911606862303) < 1e-15
        assert abs(d2000 - (-0.1115766001565926892)) < 1e-13


class TestFk54z:
    def test_basic(self):
        r1950, d1950, dr1950, dd1950 = so_pointjax.erfa.fk54z(
            0.02719026625066316119, -0.1115815170738754813,
            1954.677308160316374)
        assert abs(r1950 - 0.01602015588390065476) < 1e-14
        assert abs(d1950 - (-0.1164397101110765346)) < 1e-13
        assert abs(dr1950 - (-0.1175712648471090704e-7)) < 1e-20
        assert abs(dd1950 - 0.2108109051316431056e-7) < 1e-20


# ============================================================================
# Gnomonic projections
# ============================================================================


class TestTpxes:
    def test_basic(self):
        xi, eta, j = so_pointjax.erfa.tpxes(1.3, 1.55, 2.3, 1.5)
        assert abs(xi - (-0.01753200983236980595)) < 1e-15
        assert abs(eta - 0.05962940005778712891) < 1e-15
        assert j == 0


class TestTpxev:
    def test_basic(self):
        v = so_pointjax.erfa.s2c(1.3, 1.55)
        v0 = so_pointjax.erfa.s2c(2.3, 1.5)
        xi, eta, j = so_pointjax.erfa.tpxev(v, v0)
        assert abs(xi - (-0.01753200983236980595)) < 1e-15
        assert abs(eta - 0.05962940005778712891) < 1e-15
        assert j == 0


class TestTpsts:
    def test_basic(self):
        a, b = so_pointjax.erfa.tpsts(-0.03, 0.07, 2.3, 1.5)
        assert abs(a - 0.7596127167359629775) < 1e-14
        assert abs(b - 1.540864645109263028) < 1e-13


class TestTpstv:
    def test_basic(self):
        v0 = so_pointjax.erfa.s2c(2.3, 1.5)
        v = so_pointjax.erfa.tpstv(-0.03, 0.07, v0)
        assert abs(v[0] - 0.02170030454907376677) < 1e-15
        assert abs(v[1] - 0.02060909590535367447) < 1e-15
        assert abs(v[2] - 0.9995520806583523804) < 1e-14


class TestTpors:
    def test_basic(self):
        a01, b01, a02, b02, n = so_pointjax.erfa.tpors(-0.03, 0.07, 1.3, 1.5)
        assert abs(a01 - 1.736621577783208748) < 1e-13
        assert abs(b01 - 1.436736561844090323) < 1e-13
        assert abs(a02 - 4.004971075806584490) < 1e-13
        assert abs(b02 - 1.565084088476417917) < 1e-13
        assert n == 2


class TestTporv:
    def test_basic(self):
        v = so_pointjax.erfa.s2c(1.3, 1.5)
        v01, v02, n = so_pointjax.erfa.tporv(-0.03, 0.07, v)
        assert abs(v01[0] - (-0.02206252822366888610)) < 1e-15
        assert abs(v01[1] - 0.1318251060359645016) < 1e-14
        assert abs(v01[2] - 0.9910274397144543895) < 1e-14
        assert abs(v02[0] - (-0.003712211763801968173)) < 1e-16
        assert abs(v02[1] - (-0.004341519956299836813)) < 1e-16
        assert abs(v02[2] - 0.9999836852110587012) < 1e-14
        assert n == 2


# ============================================================================
# Differentiability tests
# ============================================================================


class TestDifferentiability:
    def test_hd2ae_jit(self):
        f = jax.jit(so_pointjax.erfa.hd2ae)
        az, el = f(1.1, 1.2, 0.3)
        assert abs(az - 5.916889243730066194) < 1e-13

    def test_icrs2g_grad(self):
        def f(dr):
            dl, db = so_pointjax.erfa.icrs2g(dr, -1.0)
            return dl
        g = jax.grad(f)(1.0)
        assert jnp.isfinite(g)

    def test_eqec06_grad(self):
        def f(dr):
            dl, db = so_pointjax.erfa.eqec06(2451545.0, 0.0, dr, 0.5)
            return dl
        g = jax.grad(f)(1.0)
        assert jnp.isfinite(g)

    def test_ltpecl_jit(self):
        f = jax.jit(so_pointjax.erfa.ltpecl)
        vec = f(2000.0)
        assert jnp.all(jnp.isfinite(vec))

    def test_gc2gde_grad(self):
        def f(h):
            xyz = so_pointjax.erfa.gd2gce(6378137.0, 1.0/298.257223563, 0.5, 0.8, h)
            e, p, hh = so_pointjax.erfa.gc2gde(6378137.0, 1.0/298.257223563, xyz)
            return hh
        g = jax.grad(f)(1000.0)
        assert jnp.isfinite(g)
        assert abs(g - 1.0) < 1e-6  # round-trip should give gradient ~1

    def test_tpxes_jit(self):
        f = jax.jit(lambda a, b: so_pointjax.erfa.tpxes(a, b, 2.3, 1.5))
        xi, eta, j = f(1.3, 1.55)
        assert abs(xi - (-0.01753200983236980595)) < 1e-15
