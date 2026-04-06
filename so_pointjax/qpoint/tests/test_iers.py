"""Tests for IERS Bulletin A loading and interpolation."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from so_pointjax.qpoint._iers import load_bulletin_a, interpolate_bulletin_a


@pytest.fixture
def iers_data(tmp_path):
    """Create synthetic IERS data and return both file path and dict."""
    mjd = np.arange(51544.0, 51644.0, 1.0)
    n = len(mjd)
    dut1 = 0.3 + 0.01 * np.arange(n)
    xp = 0.1 + 0.001 * np.arange(n)
    yp = 0.2 - 0.001 * np.arange(n)

    filepath = tmp_path / "test_iers.dat"
    np.savetxt(filepath, np.column_stack([mjd, dut1, xp, yp]))

    data = {'mjd': mjd, 'dut1': dut1, 'x': xp, 'y': yp}
    return str(filepath), data


class TestLoadBulletinA:

    def test_load(self, iers_data):
        filepath, _ = iers_data
        data = load_bulletin_a(filepath)
        assert 'mjd' in data
        assert 'dut1' in data
        assert 'x' in data
        assert 'y' in data
        assert len(data['mjd']) == 100

    def test_sorted(self, tmp_path):
        """Data should be sorted by MJD even if input is unsorted."""
        mjd = np.array([51546.0, 51544.0, 51545.0])
        dut1 = np.array([0.3, 0.1, 0.2])
        xp = np.array([0.13, 0.11, 0.12])
        yp = np.array([0.23, 0.21, 0.22])

        filepath = tmp_path / "unsorted.dat"
        np.savetxt(filepath, np.column_stack([mjd, dut1, xp, yp]))

        data = load_bulletin_a(str(filepath))
        assert_allclose(data['mjd'], [51544.0, 51545.0, 51546.0])
        assert_allclose(data['dut1'], [0.1, 0.2, 0.3])


class TestInterpolation:

    def test_exact_points(self, iers_data):
        """Interpolation at exact grid points should return exact values."""
        _, data = iers_data
        dut1, xp, yp = interpolate_bulletin_a(data, 51544.0)
        assert_allclose(dut1, 0.3, atol=1e-10)
        assert_allclose(xp, 0.1, atol=1e-10)
        assert_allclose(yp, 0.2, atol=1e-10)

    def test_midpoint(self, iers_data):
        """Interpolation at midpoint should give average."""
        _, data = iers_data
        dut1, xp, yp = interpolate_bulletin_a(data, 51544.5)
        assert_allclose(dut1, 0.305, atol=1e-10)
        assert_allclose(xp, 0.1005, atol=1e-10)
        assert_allclose(yp, 0.1995, atol=1e-10)

    def test_out_of_bounds(self, iers_data):
        """Out-of-bounds MJDs should return zeros."""
        _, data = iers_data
        dut1, xp, yp = interpolate_bulletin_a(data, 50000.0)
        assert_allclose(dut1, 0.0)
        assert_allclose(xp, 0.0)
        assert_allclose(yp, 0.0)

    def test_array_input(self, iers_data):
        """Should handle array inputs."""
        _, data = iers_data
        mjds = np.array([51544.0, 51544.5, 51545.0])
        dut1, xp, yp = interpolate_bulletin_a(data, mjds)
        assert dut1.shape == (3,)
        assert_allclose(dut1[0], 0.3, atol=1e-10)
        assert_allclose(dut1[2], 0.31, atol=1e-10)

    def test_leap_second_correction(self):
        """Leap second in dut1 should be handled."""
        data = {
            'mjd': np.array([51544.0, 51545.0, 51546.0]),
            'dut1': np.array([0.8, -0.2, -0.1]),  # leap at 51544-51545
            'x': np.array([0.1, 0.1, 0.1]),
            'y': np.array([0.2, 0.2, 0.2]),
        }
        # At midpoint, dut1 should interpolate smoothly through leap
        # dut1_hi - dut1_lo = -0.2 - 0.8 = -1.0, which triggers leap = -1
        # result = 0.5 * 0.8 + 0.5 * (-0.2 - (-1)) = 0.5 * 0.8 + 0.5 * 0.8 = 0.8
        dut1, _, _ = interpolate_bulletin_a(data, 51544.5)
        assert_allclose(dut1, 0.8, atol=1e-10)
