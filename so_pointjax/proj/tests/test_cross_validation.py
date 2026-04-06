"""Cross-validation: so_pointjax.proj vs so3g — precision agreement tests.

Every test compares so_pointjax.proj output to so3g output and asserts agreement
to a documented tolerance.  This file requires ``so3g`` to be installed.
"""

import sys
import os
import numpy as np
import jax.numpy as jnp
import pytest


def _fix_qpoint_import():
    """Ensure the installed qpoint package is importable.

    The editable install creates a namespace package at the outer qpoint/
    directory, shadowing the actual qpoint/qpoint/ subpackage. Fix by
    inserting the correct path.
    """
    qpoint_inner = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', '..', 'qpoint')
    qpoint_inner = os.path.normpath(qpoint_inner)
    if os.path.isdir(os.path.join(qpoint_inner, 'qpoint')):
        sys.path.insert(0, qpoint_inner)
    for m in list(sys.modules):
        if 'qpoint' in m:
            del sys.modules[m]


_fix_qpoint_import()

so3g_available = True
try:
    from so3g.proj import quat as sq
    from so3g.proj import coords as sc
except ImportError:
    so3g_available = False

from so_pointjax.proj import quat as jq
from so_pointjax.proj.coords import CelestialSightLine as CSL_jax, FocalPlane

pytestmark = pytest.mark.skipif(not so3g_available, reason="so3g not installed")

DEG = np.pi / 180.0


def _q2a(q):
    """Convert spt3g.core.quat to numpy (4,) float64 array."""
    return np.array([q.a, q.b, q.c, q.d], dtype=np.float64)


# ---------------------------------------------------------------------------
# Quaternion construction: euler
# ---------------------------------------------------------------------------

class TestEulerCrossVal:
    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("angle", [0.0, 0.5, 1.23, np.pi, -0.7, 2 * np.pi])
    def test_euler_scalar(self, axis, angle):
        q_so3g = _q2a(sq.euler(axis, angle))
        q_jax = np.array(jq.euler(axis, angle))
        np.testing.assert_allclose(q_jax, q_so3g, atol=1e-15,
                                   err_msg=f"euler({axis}, {angle})")

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_euler_vector(self, axis):
        angles = np.linspace(-np.pi, np.pi, 50)
        q_jax = np.array(jq.euler(axis, jnp.array(angles)))
        q_so3g = np.array([_q2a(sq.euler(axis, float(a))) for a in angles])
        np.testing.assert_allclose(q_jax, q_so3g, atol=1e-15)


# ---------------------------------------------------------------------------
# Quaternion construction: rotation_iso
# ---------------------------------------------------------------------------

class TestRotationIsoCrossVal:
    CASES = [
        (0.5, 1.2, -0.3),
        (0.0, 0.0, 0.0),
        (np.pi / 4, np.pi / 3, np.pi / 6),
        (1.5, -0.5, 0.8),
    ]

    @pytest.mark.parametrize("theta,phi,psi", CASES)
    def test_rotation_iso(self, theta, phi, psi):
        q_so3g = _q2a(sq.rotation_iso(theta, phi, psi))
        q_jax = np.array(jq.rotation_iso(theta, phi, psi))
        np.testing.assert_allclose(q_jax, q_so3g, atol=1e-14)

    @pytest.mark.parametrize("theta,phi,psi", CASES)
    def test_decompose_iso(self, theta, phi, psi):
        q = sq.rotation_iso(theta, phi, psi)
        t_so3g, p_so3g, s_so3g = sq.decompose_iso(q)

        q_jax = jnp.array(_q2a(q))
        t_jax, p_jax, s_jax = jq.decompose_iso(q_jax)

        np.testing.assert_allclose(float(t_jax), float(t_so3g), atol=1e-14)
        np.testing.assert_allclose(float(p_jax), float(p_so3g), atol=1e-14)
        np.testing.assert_allclose(float(s_jax), float(s_so3g), atol=1e-14)


# ---------------------------------------------------------------------------
# Quaternion construction: rotation_lonlat
# ---------------------------------------------------------------------------

class TestRotationLonlatCrossVal:
    CASES = [
        (1.0, 0.5, 0.3),
        (0.0, 0.0, 0.0),
        (-1.5, 0.8, -0.2),
        (np.pi, np.pi / 4, np.pi / 6),
    ]

    @pytest.mark.parametrize("lon,lat,psi", CASES)
    def test_rotation_lonlat(self, lon, lat, psi):
        q_so3g = _q2a(sq.rotation_lonlat(lon, lat, psi))
        q_jax = np.array(jq.rotation_lonlat(lon, lat, psi))
        np.testing.assert_allclose(q_jax, q_so3g, atol=1e-14)

    @pytest.mark.parametrize("lon,lat,psi", CASES)
    def test_decompose_lonlat(self, lon, lat, psi):
        q = sq.rotation_lonlat(lon, lat, psi)
        l_so3g, b_so3g, p_so3g = sq.decompose_lonlat(q)

        q_jax = jnp.array(_q2a(q))
        l_jax, b_jax, p_jax = jq.decompose_lonlat(q_jax)

        np.testing.assert_allclose(float(l_jax), float(l_so3g), atol=1e-14)
        np.testing.assert_allclose(float(b_jax), float(b_so3g), atol=1e-14)
        np.testing.assert_allclose(float(p_jax), float(p_so3g), atol=1e-14)

    @pytest.mark.parametrize("az,el,psi", CASES)
    def test_azel_mode(self, az, el, psi):
        q_so3g = _q2a(sq.rotation_lonlat(az, el, psi, azel=True))
        q_jax = np.array(jq.rotation_lonlat(az, el, psi, azel=True))
        np.testing.assert_allclose(q_jax, q_so3g, atol=1e-14)


# ---------------------------------------------------------------------------
# Quaternion construction: rotation_xieta
# ---------------------------------------------------------------------------

class TestRotationXietaCrossVal:
    CASES = [
        (0.01, -0.02, 0.78),
        (0.0, 0.0, 0.0),
        (-0.005, 0.01, np.pi / 4),
        (0.05, 0.03, -0.5),
    ]

    @pytest.mark.parametrize("xi,eta,gamma", CASES)
    def test_rotation_xieta(self, xi, eta, gamma):
        q_so3g = _q2a(sq.rotation_xieta(xi, eta, gamma))
        q_jax = np.array(jq.rotation_xieta(xi, eta, gamma))
        np.testing.assert_allclose(q_jax, q_so3g, atol=1e-14)

    @pytest.mark.parametrize("xi,eta,gamma", CASES)
    def test_decompose_xieta(self, xi, eta, gamma):
        q = sq.rotation_xieta(xi, eta, gamma)
        xi_s, eta_s, g_s = sq.decompose_xieta(q)

        q_jax = jnp.array(_q2a(q))
        xi_j, eta_j, g_j = jq.decompose_xieta(q_jax)

        np.testing.assert_allclose(float(xi_j), float(xi_s), atol=1e-14)
        np.testing.assert_allclose(float(eta_j), float(eta_s), atol=1e-14)
        np.testing.assert_allclose(float(g_j), float(g_s), atol=1e-14)

    def test_vectorized_xieta(self):
        xi = np.array([0.01, -0.005, 0.02, 0.0, 0.05])
        eta = np.array([-0.02, 0.01, -0.01, 0.0, 0.03])
        gamma = np.array([0.0, np.pi / 4, np.pi / 2, 0.0, -0.5])

        q_so3g = np.array(sq.rotation_xieta(xi, eta, gamma))
        q_jax = np.array(jq.rotation_xieta(jnp.array(xi), jnp.array(eta),
                                            jnp.array(gamma)))
        np.testing.assert_allclose(q_jax, q_so3g, atol=1e-14)


# ---------------------------------------------------------------------------
# Quaternion arithmetic: qmul, qconj
# ---------------------------------------------------------------------------

class TestQmulCrossVal:
    def _make_pairs(self):
        pairs = []
        for t1, p1, s1 in [(0.5, 1.2, -0.3), (0.0, 0.0, 0.0), (1.0, -0.5, 0.8)]:
            for t2, p2, s2 in [(0.3, 0.7, 0.1), (np.pi/4, np.pi/3, 0.0)]:
                pairs.append((
                    sq.rotation_iso(t1, p1, s1),
                    sq.rotation_iso(t2, p2, s2),
                ))
        return pairs

    def test_qmul(self):
        for q1_so3g, q2_so3g in self._make_pairs():
            prod_so3g = _q2a(q1_so3g * q2_so3g)
            q1_j = jnp.array(_q2a(q1_so3g))
            q2_j = jnp.array(_q2a(q2_so3g))
            prod_jax = np.array(jq.qmul(q1_j, q2_j))
            np.testing.assert_allclose(prod_jax, prod_so3g, atol=1e-14)

    def test_qmul_batch(self):
        angles1 = np.linspace(0.1, 2.0, 20)
        angles2 = np.linspace(-1.0, 1.0, 20)

        prods_so3g = np.array([_q2a(sq.euler(2, float(a1)) * sq.euler(1, float(a2)))
                               for a1, a2 in zip(angles1, angles2)])

        q1_j = jq.euler(2, jnp.array(angles1))
        q2_j = jq.euler(1, jnp.array(angles2))
        prods_jax = np.array(jq.qmul(q1_j, q2_j))

        np.testing.assert_allclose(prods_jax, prods_so3g, atol=1e-14)

    def test_qconj(self):
        for q_so3g, _ in self._make_pairs():
            conj_so3g = _q2a(~q_so3g)
            q_j = jnp.array(_q2a(q_so3g))
            conj_jax = np.array(jq.qconj(q_j))
            np.testing.assert_allclose(conj_jax, conj_so3g, atol=1e-15)


# ---------------------------------------------------------------------------
# Pointing: naive_az_el
# ---------------------------------------------------------------------------

class TestNaiveAzElCrossVal:
    def _make_scan(self, N=100):
        t = np.linspace(1700000000, 1700000600, N)
        az = np.linspace(0, 2 * np.pi, N)
        el = np.full(N, 50 * DEG)
        return t, az, el

    def test_naive_az_el_quaternions(self):
        """Boresight quaternions should agree to ~1e-14."""
        t, az, el = self._make_scan(200)
        csl_so3g = sc.CelestialSightLine.naive_az_el(t, az, el, site='act')
        csl_jax = CSL_jax.naive_az_el(jnp.array(t), jnp.array(az),
                                       jnp.array(el), site='act')
        q_so3g = np.array(csl_so3g.Q)
        q_jax = np.array(csl_jax.Q)
        np.testing.assert_allclose(q_jax, q_so3g, atol=1e-12,
                                   err_msg="naive_az_el quaternion disagreement")

    def test_naive_az_el_coords(self):
        """Sky coordinates from naive_az_el should agree."""
        t, az, el = self._make_scan(100)
        csl_so3g = sc.CelestialSightLine.naive_az_el(t, az, el, site='act')
        csl_jax = CSL_jax.naive_az_el(jnp.array(t), jnp.array(az),
                                       jnp.array(el), site='act')
        # so3g: decompose G3VectorQuat
        ra_so3g, dec_so3g, _ = sq.decompose_lonlat(csl_so3g.Q)
        # so_pointjax.proj coords
        c_jax = csl_jax.coords()
        ra_jax = np.array(c_jax[:, 0])
        dec_jax = np.array(c_jax[:, 1])

        np.testing.assert_allclose(ra_jax, ra_so3g, atol=1e-12,
                                   err_msg="RA disagreement")
        np.testing.assert_allclose(dec_jax, dec_so3g, atol=1e-12,
                                   err_msg="Dec disagreement")

    @pytest.mark.parametrize("site", ['act', 'so', 'so_lat', 'so_sat1', 'so_sat2', 'so_sat3'])
    def test_naive_all_sites(self, site):
        """naive_az_el should agree for every defined site."""
        t, az, el = self._make_scan(50)
        csl_so3g = sc.CelestialSightLine.naive_az_el(t, az, el, site=site)
        csl_jax = CSL_jax.naive_az_el(jnp.array(t), jnp.array(az),
                                       jnp.array(el), site=site)
        q_so3g = np.array(csl_so3g.Q)
        q_jax = np.array(csl_jax.Q)
        np.testing.assert_allclose(q_jax, q_so3g, atol=1e-12,
                                   err_msg=f"site={site}")


# ---------------------------------------------------------------------------
# Site coordinate cross-validation
# ---------------------------------------------------------------------------

class TestSitesCrossVal:
    """Verify all site definitions match between so3g and so_pointjax.proj."""

    def test_all_site_coordinates(self):
        from so_pointjax.proj.coords import SITES as jax_sites
        so3g_sites = sc.SITES
        for name in so3g_sites:
            assert name in jax_sites, f"Site {name!r} missing from so_pointjax.proj"
            s, j = so3g_sites[name], jax_sites[name]
            np.testing.assert_allclose(j.lon, s.lon, atol=1e-15,
                                       err_msg=f"site={name} lon")
            np.testing.assert_allclose(j.lat, s.lat, atol=1e-15,
                                       err_msg=f"site={name} lat")
            np.testing.assert_allclose(j.elev, s.elev, atol=0,
                                       err_msg=f"site={name} elev")

    def test_typical_weather_matches(self):
        from so_pointjax.proj.coords import SITES as jax_sites
        so3g_sites = sc.SITES
        for name in so3g_sites:
            s_w = so3g_sites[name].typical_weather
            j_w = jax_sites[name].typical_weather
            if s_w is None:
                assert j_w is None, f"site={name}: so3g has no weather, jax has {j_w}"
            else:
                assert j_w is not None, f"site={name}: so3g has weather, jax is None"
                for key in ['temperature', 'pressure', 'humidity']:
                    assert s_w[key] == j_w[key], \
                        f"site={name} weather[{key}]: so3g={s_w[key]}, jax={j_w[key]}"

    def test_default_site_matches(self):
        from so_pointjax.proj.coords import SITES as jax_sites
        assert '_default' in jax_sites
        s_def = sc.SITES['_default']
        j_def = jax_sites['_default']
        assert j_def.lon == s_def.lon
        assert j_def.lat == s_def.lat


# ---------------------------------------------------------------------------
# Weather cross-validation
# ---------------------------------------------------------------------------

class TestWeatherCrossVal:
    """Verify weather factory and refraction effects match so3g."""

    @pytest.mark.parametrize("preset", ['vacuum', 'toco', 'act', 'so', 'sa'])
    def test_weather_factory_values(self, preset):
        from so3g.proj.weather import weather_factory as wf_so3g
        from so_pointjax.proj.weather import weather_factory as wf_jax
        ws = dict(wf_so3g(preset))
        wj = dict(wf_jax(preset))
        for key in ['temperature', 'pressure', 'humidity']:
            assert ws[key] == wj[key], \
                f"weather_factory({preset!r})[{key}]: so3g={ws[key]}, jax={wj[key]}"

    def test_custom_weather_object(self):
        """Custom Weather object should work in az_el."""
        from so_pointjax.proj.weather import Weather
        w = Weather({'temperature': 10.0, 'pressure': 600.0, 'humidity': 0.5})
        t = np.linspace(1700000000, 1700000060, 10)
        az = np.full(10, 1.0)
        el = np.full(10, 50 * DEG)
        csl = CSL_jax.az_el(t, az, el, site='act', weather=w)
        assert csl.Q.shape == (10, 4)
        norms = np.sqrt(np.sum(np.array(csl.Q)**2, axis=-1))
        np.testing.assert_allclose(norms, 1.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Pointing: high-precision az_el (sites, weather, elevation)
# ---------------------------------------------------------------------------

def _sky_separation_arcsec(csl_so3g, csl_jax):
    """Compute max angular separation in arcsec between two sightlines."""
    ra_s, dec_s, _ = sq.decompose_lonlat(csl_so3g.Q)
    c_jax = csl_jax.coords()
    ra_j = np.array(c_jax[:, 0])
    dec_j = np.array(c_jax[:, 1])
    dra = (ra_j - ra_s) * np.cos(dec_s)
    ddec = dec_j - dec_s
    sep = np.sqrt(dra**2 + ddec**2) * 180 / np.pi * 3600
    return np.max(sep), np.mean(sep)


class TestAzElCrossVal:
    def _make_scan(self, N=50, el_deg=50):
        t = np.linspace(1700000000, 1700000600, N)
        az = np.linspace(0, 2 * np.pi, N)
        el = np.full(N, el_deg * DEG)
        return t, az, el

    # -- Weather presets --

    @pytest.mark.parametrize("weather", ['vacuum', 'toco', 'act', 'so', 'sa'])
    def test_az_el_weather_presets(self, weather):
        """All weather presets should agree to <1 arcsec."""
        t, az, el = self._make_scan(50)
        csl_so3g = sc.CelestialSightLine.az_el(t, az, el, site='act',
                                                 weather=weather)
        csl_jax = CSL_jax.az_el(t, az, el, site='act', weather=weather)
        max_sep, mean_sep = _sky_separation_arcsec(csl_so3g, csl_jax)
        print(f"\n  az_el (weather={weather}): max={max_sep:.4f}\", mean={mean_sep:.4f}\"")
        assert max_sep < 1.0, f"weather={weather}: {max_sep:.4f}\" > 1 arcsec"

    # -- All sites with atmosphere --

    @pytest.mark.parametrize("site", ['act', 'so', 'so_lat', 'so_sat1', 'so_sat2', 'so_sat3'])
    def test_az_el_all_sites(self, site):
        """az_el should agree for all sites with toco atmosphere."""
        t, az, el = self._make_scan(30)
        csl_so3g = sc.CelestialSightLine.az_el(t, az, el, site=site,
                                                 weather='toco')
        csl_jax = CSL_jax.az_el(t, az, el, site=site, weather='toco')
        max_sep, _ = _sky_separation_arcsec(csl_so3g, csl_jax)
        assert max_sep < 1.0, f"site={site}: {max_sep:.4f}\" > 1 arcsec"

    # -- typical_weather path --

    def test_az_el_typical_weather(self):
        """weather='typical' should use site.typical_weather."""
        t, az, el = self._make_scan(30)
        csl_so3g = sc.CelestialSightLine.az_el(t, az, el, site='act',
                                                 weather='typical')
        csl_jax = CSL_jax.az_el(t, az, el, site='act', weather='typical')
        max_sep, _ = _sky_separation_arcsec(csl_so3g, csl_jax)
        assert max_sep < 1.0, f"typical weather: {max_sep:.4f}\" > 1 arcsec"

    # -- Elevation dependence (refraction is larger near horizon) --

    @pytest.mark.parametrize("el_deg", [20, 30, 45, 60, 80])
    def test_az_el_elevation_range(self, el_deg):
        """Agreement should hold across the full elevation range."""
        t, az, el = self._make_scan(30, el_deg=el_deg)
        csl_so3g = sc.CelestialSightLine.az_el(t, az, el, site='act',
                                                 weather='toco')
        csl_jax = CSL_jax.az_el(t, az, el, site='act', weather='toco')
        max_sep, _ = _sky_separation_arcsec(csl_so3g, csl_jax)
        print(f"\n  az_el (el={el_deg}°, toco): max={max_sep:.4f}\"")
        assert max_sep < 1.0, f"el={el_deg}°: {max_sep:.4f}\" > 1 arcsec"

    # -- Custom Weather object cross-validated --

    def test_az_el_custom_weather(self):
        """Custom Weather object should give same result as so3g."""
        from so3g.proj.weather import Weather as Weather_so3g
        from so_pointjax.proj.weather import Weather as Weather_jax
        # Extreme weather: warm, humid, high pressure
        params = {'temperature': 20.0, 'pressure': 700.0, 'humidity': 0.8}
        w_so3g = Weather_so3g(params)
        w_jax = Weather_jax(params)

        t, az, el = self._make_scan(30)
        csl_so3g = sc.CelestialSightLine.az_el(t, az, el, site='act',
                                                 weather=w_so3g)
        csl_jax = CSL_jax.az_el(t, az, el, site='act', weather=w_jax)
        max_sep, _ = _sky_separation_arcsec(csl_so3g, csl_jax)
        print(f"\n  az_el (custom extreme weather): max={max_sep:.4f}\"")
        assert max_sep < 1.0, f"custom weather: {max_sep:.4f}\" > 1 arcsec"

    # -- Refraction makes a difference --

    def test_refraction_nonzero_effect(self):
        """Confirm atmosphere actually changes the pointing (not silently skipped)."""
        t, az, el = self._make_scan(30)
        csl_vac = CSL_jax.az_el(t, az, el, site='act', weather='vacuum')
        csl_atm = CSL_jax.az_el(t, az, el, site='act', weather='toco')
        q_vac = np.array(csl_vac.Q)
        q_atm = np.array(csl_atm.Q)
        max_diff = np.max(np.abs(q_vac - q_atm))
        # Refraction at 50° elevation, 550 mbar should shift pointing
        # by ~tens of arcsec — quaternion diff should be >1e-5
        assert max_diff > 1e-6, \
            f"Atmosphere had no effect: max |dq|={max_diff:.2e}"

    def test_refraction_larger_at_low_elevation(self):
        """Refraction effect should be larger at lower elevation."""
        t = np.linspace(1700000000, 1700000060, 10)
        az = np.full(10, 1.0)

        diffs = {}
        for el_deg in [25, 50, 75]:
            el = np.full(10, el_deg * DEG)
            csl_vac = CSL_jax.az_el(t, az, el, site='act', weather='vacuum')
            csl_atm = CSL_jax.az_el(t, az, el, site='act', weather='toco')
            diffs[el_deg] = np.max(np.abs(np.array(csl_vac.Q) - np.array(csl_atm.Q)))

        # Refraction should be larger at lower elevation (tan z dependence)
        assert diffs[25] > diffs[50], \
            f"Refraction at 25° ({diffs[25]:.2e}) should exceed 50° ({diffs[50]:.2e})"
        assert diffs[50] > diffs[75], \
            f"Refraction at 50° ({diffs[50]:.2e}) should exceed 75° ({diffs[75]:.2e})"

    def test_refraction_larger_with_higher_pressure(self):
        """Higher pressure should produce more refraction."""
        from so_pointjax.proj.weather import Weather
        t = np.linspace(1700000000, 1700000060, 10)
        az = np.full(10, 1.0)
        el = np.full(10, 40 * DEG)

        csl_vac = CSL_jax.az_el(t, az, el, site='act', weather='vacuum')
        w_low = Weather({'temperature': 0., 'pressure': 400., 'humidity': 0.2})
        w_high = Weather({'temperature': 0., 'pressure': 700., 'humidity': 0.2})

        csl_low = CSL_jax.az_el(t, az, el, site='act', weather=w_low)
        csl_high = CSL_jax.az_el(t, az, el, site='act', weather=w_high)

        diff_low = np.max(np.abs(np.array(csl_vac.Q) - np.array(csl_low.Q)))
        diff_high = np.max(np.abs(np.array(csl_vac.Q) - np.array(csl_high.Q)))

        assert diff_high > diff_low, \
            f"Higher pressure ({diff_high:.2e}) should refract more than lower ({diff_low:.2e})"


# ---------------------------------------------------------------------------
# Detector projection: focal plane offsets
# ---------------------------------------------------------------------------

class TestFocalPlaneCrossVal:
    def test_xieta_detector_coords(self):
        """Detector-projected sky coords should agree."""
        t = np.linspace(1700000000, 1700000060, 20)
        az = np.linspace(0, 0.5, 20)
        el = np.full(20, 50 * DEG)

        xi = np.array([0.0, 0.01, -0.005, 0.02])
        eta = np.array([0.0, -0.02, 0.01, -0.01])

        # so3g
        csl_so3g = sc.CelestialSightLine.naive_az_el(t, az, el, site='act')
        fp_so3g = sc.FocalPlane.from_xieta(xi, eta)

        # so_pointjax.proj
        csl_jax = CSL_jax.naive_az_el(jnp.array(t), jnp.array(az),
                                       jnp.array(el), site='act')
        fp_jax = FocalPlane.from_xieta(xi, eta)
        c_jax = csl_jax.coords(fplane=fp_jax)  # (ndet, N, 4)

        for idet in range(len(xi)):
            q_det = fp_so3g.quats[idet]
            ra_list, dec_list = [], []
            for it in range(len(t)):
                q_total = csl_so3g.Q[it] * q_det
                ra, dec, _ = sq.decompose_lonlat(q_total)
                ra_list.append(ra)
                dec_list.append(dec)
            ra_so3g = np.array(ra_list)
            dec_so3g = np.array(dec_list)

            ra_jax = np.array(c_jax[idet, :, 0])
            dec_jax = np.array(c_jax[idet, :, 1])

            np.testing.assert_allclose(ra_jax, ra_so3g, atol=1e-12,
                                       err_msg=f"det {idet} RA mismatch")
            np.testing.assert_allclose(dec_jax, dec_so3g, atol=1e-12,
                                       err_msg=f"det {idet} Dec mismatch")


# ---------------------------------------------------------------------------
# Precision summary (run with pytest -s to see printed output)
# ---------------------------------------------------------------------------

class TestPrecisionSummary:
    def test_summary(self):
        """Collect and print precision statistics across all functions."""
        results = {}

        # 1. Quaternion construction (scalar)
        for name, fn_name, args in [
            ("rotation_iso(0.5,1.2,-0.3)", "rotation_iso", (0.5, 1.2, -0.3)),
            ("rotation_lonlat(1.0,0.5,0.3)", "rotation_lonlat", (1.0, 0.5, 0.3)),
            ("rotation_xieta(0.01,-0.02,0.78)", "rotation_xieta", (0.01, -0.02, 0.78)),
        ]:
            q_so3g = _q2a(getattr(sq, fn_name)(*args))
            q_jax = np.array(getattr(jq, fn_name)(*args))
            results[name] = np.max(np.abs(q_so3g - q_jax))

        # 2. Vectorized xieta (100 random values)
        rng = np.random.default_rng(42)
        xi = rng.uniform(-0.05, 0.05, 100)
        eta = rng.uniform(-0.05, 0.05, 100)
        gamma = rng.uniform(-np.pi, np.pi, 100)
        q_so3g = np.array(sq.rotation_xieta(xi, eta, gamma))
        q_jax = np.array(jq.rotation_xieta(jnp.array(xi), jnp.array(eta),
                                            jnp.array(gamma)))
        results["rotation_xieta(100 random)"] = np.max(np.abs(q_so3g - q_jax))

        # 3. Decompose roundtrip
        for xi_v, eta_v, g_v in [(0.01, -0.02, 0.78), (0.05, 0.03, -0.5)]:
            q = sq.rotation_xieta(xi_v, eta_v, g_v)
            xi_s, eta_s, g_s = sq.decompose_xieta(q)
            q_j = jnp.array(_q2a(q))
            xi_j, eta_j, g_j = jq.decompose_xieta(q_j)
            max_d = max(abs(float(xi_j) - xi_s),
                        abs(float(eta_j) - eta_s),
                        abs(float(g_j) - g_s))
            results[f"decompose_xieta({xi_v},{eta_v})"] = max_d

        # 4. qmul (20 pairs)
        max_qmul_diff = 0
        for a1, a2 in zip(np.linspace(0.1, 2.0, 20), np.linspace(-1, 1, 20)):
            prod_so3g = _q2a(sq.euler(2, float(a1)) * sq.euler(1, float(a2)))
            prod_jax = np.array(jq.qmul(jq.euler(2, a1), jq.euler(1, a2)))
            max_qmul_diff = max(max_qmul_diff, np.max(np.abs(prod_so3g - prod_jax)))
        results["qmul(20 pairs)"] = max_qmul_diff

        # 5. naive_az_el (N=200)
        t = np.linspace(1700000000, 1700000600, 200)
        az = np.linspace(0, 2 * np.pi, 200)
        el = np.full(200, 50 * DEG)
        q_so3g = np.array(sc.CelestialSightLine.naive_az_el(
            t, az, el, site='act').Q)
        q_jax = np.array(CSL_jax.naive_az_el(
            jnp.array(t), jnp.array(az), jnp.array(el), site='act').Q)
        results["naive_az_el(N=200)"] = np.max(np.abs(q_so3g - q_jax))

        # 6. az_el high-precision (N=100)
        t100 = np.linspace(1700000000, 1700000600, 100)
        az100 = np.linspace(0, 2 * np.pi, 100)
        el100 = np.full(100, 50 * DEG)
        csl_so3g = sc.CelestialSightLine.az_el(t100, az100, el100,
                                                 site='act', weather='vacuum')
        csl_jax = CSL_jax.az_el(t100, az100, el100, site='act', weather='vacuum')
        q_so3g_hp = np.array(csl_so3g.Q)
        q_jax_hp = np.array(csl_jax.Q)
        results["az_el vacuum(N=100) |dq|"] = np.max(np.abs(q_so3g_hp - q_jax_hp))

        # Angular separation for az_el vacuum
        max_sep, _ = _sky_separation_arcsec(csl_so3g, csl_jax)
        results["az_el vacuum(N=100) arcsec"] = max_sep

        # 7. az_el with atmosphere (toco)
        csl_so3g_atm = sc.CelestialSightLine.az_el(t100, az100, el100,
                                                     site='act', weather='toco')
        csl_jax_atm = CSL_jax.az_el(t100, az100, el100, site='act', weather='toco')
        max_sep_atm, _ = _sky_separation_arcsec(csl_so3g_atm, csl_jax_atm)
        results["az_el toco(N=100) arcsec"] = max_sep_atm

        # 8. az_el with extreme custom weather
        from so3g.proj.weather import Weather as Weather_so3g
        from so_pointjax.proj.weather import Weather as Weather_jax
        w_extreme = {'temperature': 20.0, 'pressure': 700.0, 'humidity': 0.8}
        csl_so3g_ext = sc.CelestialSightLine.az_el(
            t100, az100, el100, site='act', weather=Weather_so3g(w_extreme))
        csl_jax_ext = CSL_jax.az_el(
            t100, az100, el100, site='act', weather=Weather_jax(w_extreme))
        max_sep_ext, _ = _sky_separation_arcsec(csl_so3g_ext, csl_jax_ext)
        results["az_el extreme weather arcsec"] = max_sep_ext

        # 9. az_el at low elevation (20°, where refraction is largest)
        el_low = np.full(100, 20 * DEG)
        csl_so3g_low = sc.CelestialSightLine.az_el(
            t100, az100, el_low, site='act', weather='toco')
        csl_jax_low = CSL_jax.az_el(
            t100, az100, el_low, site='act', weather='toco')
        max_sep_low, _ = _sky_separation_arcsec(csl_so3g_low, csl_jax_low)
        results["az_el toco el=20° arcsec"] = max_sep_low

        # Print summary table
        print("\n" + "=" * 65)
        print("  so_pointjax.proj vs so3g: precision agreement summary")
        print("=" * 65)
        print(f"  {'Function':<40s} {'max |diff|':>12s}")
        print("-" * 65)
        for name, diff in results.items():
            if 'arcsec' in name:
                print(f"  {name:<40s} {diff:>10.4f}\"")
            else:
                print(f"  {name:<40s} {diff:>12.2e}")
        print("=" * 65)

        # Assertions
        for name, diff in results.items():
            if 'arcsec' in name:
                assert diff < 1.0, f"{name}: {diff:.4f} arcsec > 1 arcsec"
            elif 'az_el' in name:
                assert diff < 1e-6, f"{name}: {diff:.2e} > 1e-6"
            else:
                assert diff < 1e-12, f"{name}: diff={diff:.2e} > 1e-12"
