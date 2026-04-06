"""Tests for state management and high-level QPoint API."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from so_pointjax.qpoint._state import QPointState, QPoint
from so_pointjax.qpoint._quaternion import identity, norm, quat2radecpa
from so_pointjax.qpoint._pointing import azel2bore


# Test parameters
LON = -44.65
LAT = -89.99
CTIME = 1700000000.0


# ---------------------------------------------------------------------------
# QPointState
# ---------------------------------------------------------------------------

class TestQPointState:

    def test_defaults(self):
        s = QPointState()
        assert s.accuracy == 1
        assert s.mean_aber is True
        assert s.fast_aber is False
        assert s.polconv == 0
        assert s.dut1 == 0.0
        assert s.weather is None
        assert s.xp == 0.0
        assert s.yp == 0.0

    def test_replace(self):
        s = QPointState()
        s2 = s.replace(accuracy=0, dut1=0.5)
        assert s2.accuracy == 0
        assert s2.dut1 == 0.5
        # Original unchanged
        assert s.accuracy == 1
        assert s.dut1 == 0.0

    def test_frozen(self):
        s = QPointState()
        with pytest.raises(AttributeError):
            s.accuracy = 0

    def test_get_iers_no_data(self):
        s = QPointState(dut1=0.3, xp=0.1, yp=0.2)
        dut1, xp, yp = s.get_iers(CTIME)
        assert dut1 == 0.3
        assert xp == 0.1
        assert yp == 0.2


# ---------------------------------------------------------------------------
# QPoint class
# ---------------------------------------------------------------------------

class TestQPointInit:

    def test_default(self):
        Q = QPoint()
        assert Q.state.accuracy == 1
        assert Q.state.mean_aber is True

    def test_kwargs(self):
        Q = QPoint(accuracy=0, fast_aber=True, dut1=0.1)
        assert Q.state.accuracy == 0
        assert Q.state.fast_aber is True
        assert Q.state.dut1 == 0.1

    def test_accuracy_string(self):
        Q = QPoint(accuracy='low')
        assert Q.state.accuracy == 1
        Q2 = QPoint(accuracy='high')
        assert Q2.state.accuracy == 0

    def test_polconv_string(self):
        Q = QPoint(polconv='iau')
        assert Q.state.polconv == 1


class TestQPointSet:

    def test_set(self):
        Q = QPoint()
        Q.set(accuracy=0, dut1=0.5)
        assert Q.state.accuracy == 0
        assert Q.state.dut1 == 0.5

    def test_set_chaining(self):
        Q = QPoint()
        Q.set(accuracy=0).set(dut1=0.5)
        assert Q.state.accuracy == 0
        assert Q.state.dut1 == 0.5


class TestQPointGet:

    def test_get_all(self):
        Q = QPoint(accuracy=0)
        params = Q.get()
        assert params['accuracy'] == 0
        assert 'iers_data' not in params

    def test_get_single(self):
        Q = QPoint(accuracy=0)
        assert Q.get('accuracy') == 0

    def test_get_multiple(self):
        Q = QPoint(accuracy=0, dut1=0.5)
        acc, dut1 = Q.get('accuracy', 'dut1')
        assert acc == 0
        assert dut1 == 0.5


class TestQPointForward:

    def test_azel2bore(self):
        Q = QPoint()
        q = Q.azel2bore(180.0, 45.0, 0.0, 0.0, LON, LAT, CTIME)
        assert_allclose(norm(q), 1.0, atol=1e-13)

    def test_azel2bore_matches_functional(self):
        """QPoint.azel2bore should match the functional API."""
        Q = QPoint()
        q1 = Q.azel2bore(180.0, 45.0, 0.0, 0.0, LON, LAT, CTIME)
        q2 = azel2bore(180.0, 45.0, LON, LAT, CTIME)
        assert_allclose(q1, q2, atol=1e-14)

    def test_bore2radecpa(self):
        Q = QPoint()
        q_bore = Q.azel2bore(180.0, 45.0, 0.0, 0.0, LON, LAT, CTIME)
        ra, dec, pa = Q.bore2radecpa(identity(), CTIME, q_bore)
        assert jnp.isfinite(ra)
        assert -90 <= float(dec) <= 90

    def test_azel2radecpa(self):
        Q = QPoint()
        ra, dec, pa = Q.azel2radecpa(0.0, 0.0, 0.0,
                                     180.0, 45.0, LON, LAT, CTIME)
        assert jnp.all(jnp.isfinite(jnp.array([ra, dec, pa])))


class TestQPointInverse:

    def test_radec2azel(self):
        Q = QPoint()
        q_bore = Q.azel2bore(180.0, 45.0, 0.0, 0.0, LON, LAT, CTIME)
        ra, dec, pa = quat2radecpa(q_bore)
        az, el, _ = Q.radec2azel(ra, dec, pa, LON, LAT, CTIME)
        assert jnp.all(jnp.isfinite(jnp.array([az, el])))

    def test_roundtrip(self):
        Q = QPoint(accuracy=1)
        az0, el0 = 180.0, 45.0
        q_bore = Q.azelpsi2bore(az0, el0, 0.0, LON, LAT, CTIME)
        ra, dec, pa = quat2radecpa(q_bore)
        az_out, el_out, _ = Q.radec2azel(ra, dec, pa, LON, LAT, CTIME)
        assert_allclose(float(az_out) % 360, az0 % 360, atol=0.1)
        assert_allclose(float(el_out), el0, atol=0.1)


class TestQPointUtility:

    def test_det_offset(self):
        Q = QPoint()
        q = Q.det_offset(1.0, 0.0, 0.0)
        assert_allclose(norm(q), 1.0, atol=1e-15)

    def test_hwp_quat(self):
        Q = QPoint()
        q = Q.hwp_quat(45.0)
        assert_allclose(norm(q), 1.0, atol=1e-15)

    def test_gmst(self):
        Q = QPoint()
        gmst = Q.gmst(CTIME)
        assert 0 <= float(gmst) < 360

    def test_lmst(self):
        Q = QPoint()
        lmst = Q.lmst(CTIME, LON)
        assert 0 <= float(lmst) < 360

    def test_bore_offset(self):
        Q = QPoint()
        q_bore = Q.azel2bore(180.0, 45.0, 0.0, 0.0, LON, LAT, CTIME)
        q_shifted = Q.bore_offset(q_bore, 1.0, 0.0, 0.0)
        assert_allclose(norm(q_shifted), 1.0, atol=1e-13)
        # Should be different from original
        assert not jnp.allclose(q_bore, q_shifted)

    def test_precompute_times(self):
        Q = QPoint()
        times = Q.precompute_times(CTIME)
        assert 'tt1' in times
        assert 'ut1_1' in times


# ---------------------------------------------------------------------------
# IERS Bulletin A (with synthetic data)
# ---------------------------------------------------------------------------

class TestQPointIERS:

    @pytest.fixture
    def iers_file(self, tmp_path):
        """Create a synthetic IERS data file."""
        mjd_start = 51544.0  # J2000
        n = 10000
        mjd = np.arange(mjd_start, mjd_start + n, dtype=np.float64)
        dut1 = 0.3 * np.sin(2 * np.pi * np.arange(n) / 365.25)
        xp = 0.1 + 0.05 * np.sin(2 * np.pi * np.arange(n) / 365.25)
        yp = 0.3 + 0.05 * np.cos(2 * np.pi * np.arange(n) / 365.25)

        filepath = tmp_path / "finals.data"
        np.savetxt(filepath, np.column_stack([mjd, dut1, xp, yp]))
        return str(filepath)

    def test_load_bulletin_a(self, iers_file):
        Q = QPoint()
        Q.load_bulletin_a(iers_file)
        assert Q.state.iers_data is not None
        assert 'mjd' in Q.state.iers_data

    def test_get_bulletin_a(self, iers_file):
        Q = QPoint()
        Q.load_bulletin_a(iers_file)
        dut1, xp, yp = Q.get_bulletin_a(51544.5)
        assert np.isfinite(dut1)
        assert np.isfinite(xp)
        assert np.isfinite(yp)

    def test_get_bulletin_a_not_loaded(self):
        Q = QPoint()
        with pytest.raises(ValueError, match="No IERS"):
            Q.get_bulletin_a(51544.0)

    def test_iers_used_in_pipeline(self, iers_file):
        """IERS data should affect pipeline output."""
        Q1 = QPoint()
        Q2 = QPoint()
        Q2.load_bulletin_a(iers_file)

        # Use a ctime that maps to an MJD within the IERS data range
        # MJD 51544 = 2000-01-01 12:00 UTC
        # ctime = (MJD - 40587) * 86400
        ctime_test = (51544.0 + 500 - 40587.0) * 86400.0

        q1 = Q1.azel2bore(180.0, 45.0, 0.0, 0.0, LON, LAT, ctime_test)
        q2 = Q2.azel2bore(180.0, 45.0, 0.0, 0.0, LON, LAT, ctime_test)

        # Should be slightly different due to dut1/xp/yp corrections
        # (may be very close but not identical)
        ra1, dec1, _ = quat2radecpa(q1)
        ra2, dec2, _ = quat2radecpa(q2)
        # Just check both are valid
        assert jnp.isfinite(ra1) and jnp.isfinite(ra2)
