"""Tests for time conversion utilities."""

import jax.numpy as jnp
from numpy.testing import assert_allclose

from so_pointjax.qpoint._time_utils import (
    ctime2jd, jd2ctime, ctime2jdtt, jdutc2jdut1, ctime2gmst,
    CTIME_JD_EPOCH,
)


class TestCtime2Jd:

    def test_epoch(self):
        """ctime=0 should give JD of Unix epoch."""
        jd1, jd2 = ctime2jd(0.0)
        assert_allclose(jd1, CTIME_JD_EPOCH)
        assert_allclose(jd2, 0.0)

    def test_one_day(self):
        jd1, jd2 = ctime2jd(86400.0)
        assert_allclose(jd1, CTIME_JD_EPOCH)
        assert_allclose(jd2, 1.0)

    def test_roundtrip(self):
        ctime = 1700000000.0  # approx 2023-11-14
        jd1, jd2 = ctime2jd(ctime)
        ctime2 = jd2ctime(jd1, jd2)
        assert_allclose(ctime2, ctime, atol=1e-6)


class TestCtime2Jdtt:

    def test_tt_ahead_of_utc(self):
        """TT should be ahead of UTC (by ~37+32.184 seconds in 2020s)."""
        ctime = 1700000000.0
        jd1_utc, jd2_utc = ctime2jd(ctime)
        tt1, tt2 = ctime2jdtt(ctime)
        # TT - UTC offset in days
        diff_days = (tt1 - jd1_utc) + (tt2 - jd2_utc)
        diff_secs = diff_days * 86400.0
        # Should be ~69.184 seconds (37 leap seconds + 32.184)
        assert 60 < diff_secs < 80

    def test_known_value(self):
        """Test a known UTC->TT conversion."""
        # 2020-01-01 00:00:00 UTC = ctime 1577836800
        ctime = 1577836800.0
        tt1, tt2 = ctime2jdtt(ctime)
        # TT should be 2458849.5 + 69.184/86400
        jd_utc = CTIME_JD_EPOCH + ctime / 86400.0
        jd_tt = tt1 + tt2
        assert_allclose(jd_tt - jd_utc, 69.184 / 86400.0, atol=1e-10)


class TestJdutc2Jdut1:

    def test_dut1_offset(self):
        """UT1 should differ from UTC by approximately dut1."""
        ctime = 1700000000.0
        jd1, jd2 = ctime2jd(ctime)
        dut1 = 0.1  # 100ms ahead
        ut1_1, ut1_2 = jdutc2jdut1(jd1, jd2, dut1)
        diff_secs = ((ut1_1 - jd1) + (ut1_2 - jd2)) * 86400.0
        assert_allclose(diff_secs, dut1, atol=1e-6)


class TestCtime2Gmst:

    def test_reasonable_range(self):
        """GMST should be in [0, 2pi)."""
        ctime = 1700000000.0
        gmst = ctime2gmst(ctime, dut1=0.0, accuracy=1)
        assert 0 <= float(gmst) < 2 * 3.14159265358979

    def test_low_accuracy_runs(self):
        """Low accuracy mode should not raise."""
        gmst = ctime2gmst(1700000000.0, accuracy=1)
        assert jnp.isfinite(gmst)

    def test_full_accuracy_runs(self):
        """Full accuracy mode should not raise."""
        gmst = ctime2gmst(1700000000.0, dut1=0.05, accuracy=0)
        assert jnp.isfinite(gmst)
