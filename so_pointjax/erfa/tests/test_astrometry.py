"""Tests for astrometry functions, validated against ERFA C test suite."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import so_pointjax.erfa as era
from so_pointjax.erfa._types import LDBODY


def assert_close(a, b, atol=1e-12):
    assert jnp.allclose(jnp.asarray(a), jnp.asarray(b), atol=atol), f"Expected {b}, got {a}"


# ============================================================================
# Geodetic helpers
# ============================================================================

class TestEform:
    def test_wgs84(self):
        a, f = era.eform(era.WGS84)
        assert_close(a, 6378137.0, atol=1e-10)
        assert_close(f, 0.3352810664747480720e-2, atol=1e-18)

    def test_grs80(self):
        a, f = era.eform(era.GRS80)
        assert_close(a, 6378137.0, atol=1e-10)
        assert_close(f, 0.3352810681182318935e-2, atol=1e-18)

    def test_wgs72(self):
        a, f = era.eform(era.WGS72)
        assert_close(a, 6378135.0, atol=1e-10)
        assert_close(f, 0.3352779454167504862e-2, atol=1e-18)

    def test_invalid(self):
        with pytest.raises(ValueError):
            era.eform(0)


class TestGd2gc:
    def test_wgs84(self):
        xyz = era.gd2gc(era.WGS84, 3.1, -0.5, 2500.0)
        assert_close(xyz[0], -5599000.5577049947, atol=1e-7)
        assert_close(xyz[1], 233011.67223479203, atol=1e-7)
        assert_close(xyz[2], -3040909.4706983363, atol=1e-7)


# ============================================================================
# Fundamental astrometric effects
# ============================================================================

class TestAb:
    def test_basic(self):
        pnat = jnp.array([-0.76321968546737951, -0.60869453983060384, -0.21676408580639883])
        v = jnp.array([2.1044018893653786e-5, -8.9108923304429319e-5, -3.8633714797716569e-5])
        ppr = era.ab(pnat, v, 0.99980921395708788, 0.99999999506209258)
        assert_close(ppr[0], -0.7631631094219556269)
        assert_close(ppr[1], -0.6087553082505590832)
        assert_close(ppr[2], -0.2167926269368471279)


class TestLd:
    def test_basic(self):
        p = jnp.array([-0.763276255, -0.608633767, -0.216735543])
        q = jnp.array([-0.763276255, -0.608633767, -0.216735543])
        e = jnp.array([0.76700421, 0.605629598, 0.211937094])
        p1 = era.ld(0.00028574, p, q, e, 8.91276983, 3e-10)
        assert_close(p1[0], -0.7632762548968159627)
        assert_close(p1[1], -0.6086337670823762701)
        assert_close(p1[2], -0.2167355431320546947)


class TestLdn:
    def test_basic(self):
        b = [
            LDBODY(bm=jnp.float64(0.00028574), dl=jnp.float64(3e-10),
                   pv=jnp.array([[-7.81014427, -5.60956681, -1.98079819],
                                  [0.0030723249, -0.00406995477, -0.00181335842]])),
            LDBODY(bm=jnp.float64(0.00095435), dl=jnp.float64(3e-9),
                   pv=jnp.array([[0.738098796, 4.63658692, 1.9693136],
                                  [-0.00755816922, 0.00126913722, 0.000727999001]])),
            LDBODY(bm=jnp.float64(1.0), dl=jnp.float64(6e-6),
                   pv=jnp.array([[-0.000712174377, -0.00230478303, -0.00105865966],
                                  [6.29235213e-6, -3.30888387e-7, -2.96486623e-7]])),
        ]
        ob = jnp.array([-0.974170437, -0.2115201, -0.0917583114])
        sc = jnp.array([-0.763276255, -0.608633767, -0.216735543])
        sn = era.ldn(3, b, ob, sc)
        assert_close(sn[0], -0.7632762579693333866)
        assert_close(sn[1], -0.6086337636093002660)
        assert_close(sn[2], -0.2167355420646328159)


class TestLdsun:
    def test_basic(self):
        p = jnp.array([-0.763276255, -0.608633767, -0.216735543])
        e = jnp.array([-0.973644023, -0.20925523, -0.0907169552])
        p1 = era.ldsun(p, e, 0.999809214)
        assert_close(p1[0], -0.7632762580731413169)
        assert_close(p1[1], -0.6086337635262647900)
        assert_close(p1[2], -0.2167355419322321302)


class TestPmpx:
    def test_basic(self):
        pob = jnp.array([0.9, 0.4, 0.1])
        pco = era.pmpx(1.234, 0.789, 1e-5, -2e-5, 1e-2, 10.0, 8.75, pob)
        assert_close(pco[0], 0.2328137623960308438)
        assert_close(pco[1], 0.6651097085397855328)
        assert_close(pco[2], 0.7095257765896359837)


class TestRefco:
    def test_basic(self):
        refa, refb = era.refco(800.0, 10.0, 0.9, 0.4)
        assert_close(refa, 0.2264949956241415009e-3, atol=1e-15)
        assert_close(refb, -0.2598658261729343970e-6, atol=1e-18)


class TestPvtob:
    def test_basic(self):
        pv = era.pvtob(2.0, 0.5, 3000.0, 1e-6, -0.5e-6, 1e-8, 5.0)
        assert_close(pv[0][0], 4225081.367071159207, atol=1e-5)
        assert_close(pv[0][1], 3681943.215856198144, atol=1e-5)
        assert_close(pv[0][2], 3041149.399241260785, atol=1e-5)
        assert_close(pv[1][0], -268.4915389365998787, atol=1e-9)
        assert_close(pv[1][1], 308.0977983288903123, atol=1e-9)
        assert_close(pv[1][2], 0.0, atol=0.0)


# ============================================================================
# Ephemeris
# ============================================================================

class TestEpv00:
    def test_basic(self):
        pvh, pvb = era.epv00(2400000.5, 53411.52501161)
        assert_close(pvh[0][0], -0.7757238809297706813)
        assert_close(pvh[0][1], 0.5598052241363340596)
        assert_close(pvh[0][2], 0.2426998466481686993)
        assert_close(pvh[1][0], -0.1091891824147313846e-1)
        assert_close(pvh[1][1], -0.1247187268440845008e-1)
        assert_close(pvh[1][2], -0.5407569418065039061e-2)
        assert_close(pvb[0][0], -0.7714104440491111971)
        assert_close(pvb[0][1], 0.5598412061824171323)
        assert_close(pvb[0][2], 0.2425996277722452400)
        assert_close(pvb[1][0], -0.1091874268116823295e-1)
        assert_close(pvb[1][1], -0.1246525461732861538e-1)
        assert_close(pvb[1][2], -0.5404773180966231279e-2)


# ============================================================================
# Context-setup functions
# ============================================================================

class TestApcg:
    def test_basic(self):
        ebpv = jnp.array([[0.901310875, -0.417402664, -0.180982288],
                           [0.00742727954, 0.0140507459, 0.00609045792]])
        ehp = jnp.array([0.903358544, -0.415395237, -0.180084014])
        a = era.apcg(2456165.5, 0.401182685, ebpv, ehp)
        assert_close(a.pmt, 12.65133794027378508, atol=1e-11)
        assert_close(a.eb[0], 0.901310875)
        assert_close(a.eh[0], 0.8940025429324143045)
        assert_close(a.em, 1.010465295811013146)
        assert_close(a.v[0], 0.4289638913597693554e-4, atol=1e-16)
        assert_close(a.bm1, 0.9999999951686012981)
        assert_close(a.bpn[0, 0], 1.0, atol=0.0)


class TestApcs:
    def test_basic(self):
        pv = jnp.array([[-1836024.09, 1056607.72, -5998795.26],
                         [-77.0361767, -133.310856, 0.0971855934]])
        ebpv = jnp.array([[-0.974170438, -0.211520082, -0.0917583024],
                           [0.00364365824, -0.0154287319, -0.00668922024]])
        ehp = jnp.array([-0.973458265, -0.209215307, -0.0906996477])
        a = era.apcs(2456384.5, 0.970031644, pv, ebpv, ehp)
        assert_close(a.pmt, 13.25248468622587269, atol=1e-11)
        assert_close(a.eb[0], -0.9741827110629881886)
        assert_close(a.eh[0], -0.9736425571689454706)
        assert_close(a.em, 0.9998233241709796859)
        assert_close(a.v[0], 0.2078704993282685510e-4, atol=1e-16)
        assert_close(a.bm1, 0.9999999950277561237)


class TestApci:
    def test_basic(self):
        ebpv = jnp.array([[0.901310875, -0.417402664, -0.180982288],
                           [0.00742727954, 0.0140507459, 0.00609045792]])
        ehp = jnp.array([0.903358544, -0.415395237, -0.180084014])
        a = era.apci(2456165.5, 0.401182685, ebpv, ehp, 0.0013122272, -2.92808623e-5, 3.05749468e-8)
        assert_close(a.bpn[0, 0], 0.9999991390295159156)
        assert_close(a.bpn[2, 0], 0.1312227200000000000e-2)
        assert_close(a.bpn[2, 2], 0.9999991386008323373)


class TestApco:
    def test_basic(self):
        ebpv = jnp.array([[-0.974170438, -0.211520082, -0.0917583024],
                           [0.00364365824, -0.0154287319, -0.00668922024]])
        ehp = jnp.array([-0.973458265, -0.209215307, -0.0906996477])
        a = era.apco(2456384.5, 0.970031644, ebpv, ehp,
                     0.0013122272, -2.92808623e-5, 3.05749468e-8,
                     3.14540971, -0.527800806, -1.2345856, 2738.0,
                     2.47230737e-7, 1.82640464e-6, -3.01974337e-11,
                     0.000201418779, -2.36140831e-7)
        assert_close(a.pmt, 13.25248468622587269, atol=1e-11)
        assert_close(a.along, -0.5278008060295995734)
        assert_close(a.xpl, 0.1133427418130752958e-5, atol=1e-17)
        assert_close(a.ypl, 0.1453347595780646207e-5, atol=1e-17)
        assert_close(a.sphi, -0.9440115679003211329)
        assert_close(a.cphi, 0.3299123514971474711)
        assert_close(a.eral, 2.617608903970400427)
        assert_close(a.refa, 0.2014187790000000000e-3, atol=1e-15)
        assert_close(a.refb, -0.2361408310000000000e-6, atol=1e-18)


class TestAper:
    def test_basic(self):
        a = era.ASTROM.empty()._replace(along=jnp.float64(1.234))
        a = era.aper(5.678, a)
        assert_close(a.eral, 6.912000000000000000)


class TestApio:
    def test_basic(self):
        a = era.apio(-3.01974337e-11, 3.14540971, -0.527800806, -1.2345856,
                     2738.0, 2.47230737e-7, 1.82640464e-6, 0.000201418779, -2.36140831e-7)
        assert_close(a.along, -0.5278008060295995734)
        assert_close(a.xpl, 0.1133427418130752958e-5, atol=1e-17)
        assert_close(a.ypl, 0.1453347595780646207e-5, atol=1e-17)
        assert_close(a.sphi, -0.9440115679003211329)
        assert_close(a.cphi, 0.3299123514971474711)
        assert_close(a.diurab, 0.5135843661699913529e-6)
        assert_close(a.eral, 2.617608903970400427)
        assert_close(a.refa, 0.2014187790000000000e-3, atol=1e-15)


# ============================================================================
# Context-setup "13" wrappers
# ============================================================================

class TestApci13:
    def test_basic(self):
        a, eo = era.apci13(2456165.5, 0.401182685)
        assert_close(a.pmt, 12.65133794027378508, atol=1e-11)
        assert_close(a.eb[0], 0.9013108747340644755)
        assert_close(a.eh[0], 0.8940025429255499549)
        assert_close(a.em, 1.010465295964664178)
        assert_close(a.bpn[0, 0], 0.9999992060376761710)
        assert_close(eo, -0.2900618712657375647e-2)


class TestApco13:
    def test_basic(self):
        a, eo = era.apco13(2456384.5, 0.969254051, 0.1550675,
                           -0.527800806, -1.2345856, 2738.0,
                           2.47230737e-7, 1.82640464e-6,
                           731.0, 12.8, 0.59, 0.55)
        assert_close(a.pmt, 13.25248468622475727, atol=1e-11)
        assert_close(a.along, -0.5278008060295995733)
        assert_close(a.eral, 2.617608909189664000)
        assert_close(eo, -0.003020548354802412839, atol=1e-14)


class TestAper13:
    def test_basic(self):
        a = era.ASTROM.empty()._replace(along=jnp.float64(1.234))
        a = era.aper13(2456165.5, 0.401182685, a)
        assert_close(a.eral, 3.316236661789694933)


class TestApio13:
    def test_basic(self):
        a = era.apio13(2456384.5, 0.969254051, 0.1550675,
                       -0.527800806, -1.2345856, 2738.0,
                       2.47230737e-7, 1.82640464e-6,
                       731.0, 12.8, 0.59, 0.55)
        assert_close(a.along, -0.5278008060295995733)
        assert_close(a.diurab, 0.5135843661699913529e-6)
        assert_close(a.eral, 2.617608909189664000)


# ============================================================================
# Coordinate transforms
# ============================================================================

class TestAtci13:
    def test_basic(self):
        ri, di, eo = era.atci13(2.71, 0.174, 1e-5, 5e-6, 0.1, 55.0,
                                2456165.5, 0.401182685)
        assert_close(ri, 2.710121572968696744)
        assert_close(di, 0.1729371367219539137)
        assert_close(eo, -0.002900618712657375647, atol=1e-14)


class TestAtciq:
    def test_basic(self):
        astrom, eo = era.apci13(2456165.5, 0.401182685)
        ri, di = era.atciq(2.71, 0.174, 1e-5, 5e-6, 0.1, 55.0, astrom)
        assert_close(ri, 2.710121572968696744)
        assert_close(di, 0.1729371367219539137)


class TestAtciqz:
    def test_basic(self):
        astrom, eo = era.apci13(2456165.5, 0.401182685)
        ri, di = era.atciqz(2.71, 0.174, astrom)
        assert_close(ri, 2.709994899247256984)
        assert_close(di, 0.1728740720984931891)


class TestAtciqn:
    def test_basic(self):
        astrom, eo = era.apci13(2456165.5, 0.401182685)
        b = _make_ldbody_list()
        ri, di = era.atciqn(2.71, 0.174, 1e-5, 5e-6, 0.1, 55.0, astrom, 3, b)
        assert_close(ri, 2.710122008104983335)
        assert_close(di, 0.1729371916492767821)


class TestAtic13:
    def test_basic(self):
        rc, dc, eo = era.atic13(2.710121572969038991, 0.1729371367218230438,
                                2456165.5, 0.401182685)
        assert_close(rc, 2.710126504531716819)
        assert_close(dc, 0.1740632537627034482)
        assert_close(eo, -0.002900618712657375647, atol=1e-14)


class TestAticq:
    def test_basic(self):
        astrom, eo = era.apci13(2456165.5, 0.401182685)
        rc, dc = era.aticq(2.710121572969038991, 0.1729371367218230438, astrom)
        assert_close(rc, 2.710126504531716819)
        assert_close(dc, 0.1740632537627034482)


class TestAticqn:
    def test_basic(self):
        astrom, eo = era.apci13(2456165.5, 0.401182685)
        b = _make_ldbody_list()
        rc, dc = era.aticqn(2.709994899247599271, 0.1728740720983623469, astrom, 3, b)
        assert_close(rc, 2.709999575033027333)
        assert_close(dc, 0.1739999656316469990)


class TestAtco13:
    def test_basic(self):
        aob, zob, hob, dob, rob, eo = era.atco13(
            2.71, 0.174, 1e-5, 5e-6, 0.1, 55.0,
            2456384.5, 0.969254051, 0.1550675,
            -0.527800806, -1.2345856, 2738.0,
            2.47230737e-7, 1.82640464e-6,
            731.0, 12.8, 0.59, 0.55)
        assert_close(aob, 0.9251774485485515207e-1)
        assert_close(zob, 1.407661405256499357)
        assert_close(hob, -0.9265154431529724692e-1)
        assert_close(dob, 0.1716626560072526200)
        assert_close(rob, 2.710260453504961012)
        assert_close(eo, -0.003020548354802412839, atol=1e-14)


class TestAtio13:
    def test_basic(self):
        aob, zob, hob, dob, rob = era.atio13(
            2.710121572969038991, 0.1729371367218230438,
            2456384.5, 0.969254051, 0.1550675,
            -0.527800806, -1.2345856, 2738.0,
            2.47230737e-7, 1.82640464e-6,
            731.0, 12.8, 0.59, 0.55)
        assert_close(aob, 0.9233952224895122499e-1)
        assert_close(zob, 1.407758704513549991)
        assert_close(hob, -0.9247619879881698140e-1)
        assert_close(dob, 0.1717653435756234676)
        assert_close(rob, 2.710085107988480746)


class TestAtoc13:
    def test_R(self):
        rc, dc = era.atoc13('R', 2.710085107986886201, 0.1717653435758265198,
            2456384.5, 0.969254051, 0.1550675,
            -0.527800806, -1.2345856, 2738.0,
            2.47230737e-7, 1.82640464e-6, 731.0, 12.8, 0.59, 0.55)
        assert_close(rc, 2.709956744659136129)
        assert_close(dc, 0.1741696500898471362)

    def test_A(self):
        rc, dc = era.atoc13('A', 0.09233952224794989993, 1.407758704513722461,
            2456384.5, 0.969254051, 0.1550675,
            -0.527800806, -1.2345856, 2738.0,
            2.47230737e-7, 1.82640464e-6, 731.0, 12.8, 0.59, 0.55)
        assert_close(rc, 2.709956744659734086)
        assert_close(dc, 0.1741696500898471366)


class TestAtoi13:
    def test_R(self):
        ri, di = era.atoi13('R', 2.710085107986886201, 0.1717653435758265198,
            2456384.5, 0.969254051, 0.1550675,
            -0.527800806, -1.2345856, 2738.0,
            2.47230737e-7, 1.82640464e-6, 731.0, 12.8, 0.59, 0.55)
        assert_close(ri, 2.710121574447540810)
        assert_close(di, 0.1729371839116608778)

    def test_A(self):
        ri, di = era.atoi13('A', 0.09233952224794989993, 1.407758704513722461,
            2456384.5, 0.969254051, 0.1550675,
            -0.527800806, -1.2345856, 2738.0,
            2.47230737e-7, 1.82640464e-6, 731.0, 12.8, 0.59, 0.55)
        assert_close(ri, 2.710121574448138676)
        assert_close(di, 0.1729371839116608781)


# ============================================================================
# Space motion
# ============================================================================

class TestStarpv:
    def test_basic(self):
        pv, iwarn = era.starpv(0.01686756, -1.093989828, -1.78323516e-5,
                               2.336024047e-6, 0.74723, -21.6)
        assert_close(pv[0][0], 126668.5912743160601, atol=1e-10)
        assert_close(pv[0][1], 2136.792716839935195)
        assert_close(pv[0][2], -245251.2339876830091, atol=1e-10)
        assert_close(pv[1][0], -0.4051854008955659551e-2, atol=1e-13)
        assert_close(pv[1][1], -0.6253919754414777970e-2, atol=1e-15)
        assert_close(pv[1][2], 0.1189353714588109341e-1, atol=1e-13)


class TestPvstar:
    def test_basic(self):
        pv = jnp.array([[126668.5912743160601, 2136.792716839935195, -245251.2339876830091],
                         [-0.4051854035740712739e-2, -0.6253919754866173866e-2, 0.1189353719774107189e-1]])
        ra, dec, pmr, pmd, px, rv = era.pvstar(pv)
        assert_close(ra, 0.1686756e-1)
        assert_close(dec, -1.093989828)
        assert_close(pmr, -0.1783235160000472788e-4, atol=1e-16)
        assert_close(pmd, 0.2336024047000619347e-5, atol=1e-16)
        assert_close(px, 0.74723)
        assert_close(rv, -21.60000010107306010, atol=1e-11)


class TestPmsafe:
    def test_basic(self):
        ra2, dec2, pmr2, pmd2, px2, rv2 = era.pmsafe(
            1.234, 0.789, 1e-5, -2e-5, 1e-2, 10.0,
            2400000.5, 48348.5625, 2400000.5, 51544.5)
        assert_close(ra2, 1.234087484501017061)
        assert_close(dec2, 0.7888249982450468567)
        assert_close(pmr2, 0.9996457663586073988e-5)
        assert_close(pmd2, -0.2000040085106754565e-4, atol=1e-16)
        assert_close(px2, 0.9999997295356830666e-2)
        assert_close(rv2, 10.38468380293920069, atol=1e-10)


# ============================================================================
# Differentiability tests
# ============================================================================

class TestAstrometryDifferentiability:
    def test_jit_ab(self):
        pnat = jnp.array([-0.76321968546737951, -0.60869453983060384, -0.21676408580639883])
        v = jnp.array([2.1044018893653786e-5, -8.9108923304429319e-5, -3.8633714797716569e-5])
        ppr = jax.jit(era.ab)(pnat, v, 0.99980921395708788, 0.99999999506209258)
        assert_close(ppr[0], -0.7631631094219556269)

    def test_grad_refco(self):
        """Refraction constants should be differentiable w.r.t. pressure."""
        grad_fn = jax.grad(lambda p: era.refco(p, 10.0, 0.9, 0.4)[0])
        g = grad_fn(800.0)
        assert jnp.abs(g) > 0.0

    def test_jit_atciq(self):
        astrom, eo = era.apci13(2456165.5, 0.401182685)
        ri, di = jax.jit(era.atciq)(2.71, 0.174, 1e-5, 5e-6, 0.1, 55.0, astrom)
        assert_close(ri, 2.710121572968696744)

    def test_jit_pvtob(self):
        pv = jax.jit(era.pvtob)(2.0, 0.5, 3000.0, 1e-6, -0.5e-6, 1e-8, 5.0)
        assert_close(pv[0][0], 4225081.367071159207, atol=1e-5)


# ============================================================================
# Helper
# ============================================================================

def _make_ldbody_list():
    return [
        LDBODY(bm=jnp.float64(0.00028574), dl=jnp.float64(3e-10),
               pv=jnp.array([[-7.81014427, -5.60956681, -1.98079819],
                              [0.0030723249, -0.00406995477, -0.00181335842]])),
        LDBODY(bm=jnp.float64(0.00095435), dl=jnp.float64(3e-9),
               pv=jnp.array([[0.738098796, 4.63658692, 1.9693136],
                              [-0.00755816922, 0.00126913722, 0.000727999001]])),
        LDBODY(bm=jnp.float64(1.0), dl=jnp.float64(6e-6),
               pv=jnp.array([[-0.000712174377, -0.00230478303, -0.00105865966],
                              [6.29235213e-6, -3.30888387e-7, -2.96486623e-7]])),
    ]
