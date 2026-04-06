"""Tests for leap-second / Delta(AT) function, validated against ERFA C test suite."""

import pytest
import so_pointjax.erfa as era


def assert_close(a, b, atol=1e-12):
    assert abs(a - b) < atol, f"Expected {b}, got {a}"


class TestDat:
    def test_2003(self):
        result = era.dat(2003, 6, 1, 0.0)
        assert_close(result, 32.0)

    def test_2008(self):
        result = era.dat(2008, 1, 17, 0.0)
        assert_close(result, 33.0)

    def test_2017(self):
        result = era.dat(2017, 9, 1, 0.0)
        assert_close(result, 37.0)

    def test_pre_1972(self):
        # Pre-1972 should include drift adjustment
        result = era.dat(1970, 1, 1, 0.0)
        assert result > 0.0

    def test_bad_fraction(self):
        with pytest.raises(ValueError, match="bad fraction"):
            era.dat(2000, 1, 1, -0.1)

    def test_bad_month(self):
        with pytest.raises(ValueError, match="bad month"):
            era.dat(2000, 13, 1, 0.0)

    def test_pre_utc(self):
        with pytest.raises(ValueError, match="predates UTC"):
            era.dat(1950, 1, 1, 0.0)

    def test_1972_boundary(self):
        # Just before 1972: should use drift model
        result_before = era.dat(1971, 12, 31, 0.0)
        # At 1972: should be exactly 10.0
        result_at = era.dat(1972, 1, 1, 0.0)
        assert_close(result_at, 10.0)
        assert result_before != 10.0
