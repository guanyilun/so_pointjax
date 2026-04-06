"""IERS Bulletin A loading and interpolation.

Provides dut1, xp, yp polar motion parameters as a function of MJD.
"""

import numpy as np
import jax.numpy as jnp


def load_bulletin_a(filename, columns=None):
    """Load IERS Bulletin A data from a text file.

    Parameters
    ----------
    filename : str
        Path to whitespace-delimited text file.
    columns : list of str, optional
        Column names. Default: ['mjd', 'dut1', 'x', 'y'].
        Must include 'mjd', 'dut1', 'x', 'y'.

    Returns
    -------
    dict with keys 'mjd', 'dut1', 'x', 'y' as numpy arrays.
    """
    if columns is None:
        columns = ['mjd', 'dut1', 'x', 'y']

    data = np.loadtxt(filename)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    result = {}
    for i, col in enumerate(columns):
        if col in ('mjd', 'dut1', 'x', 'y'):
            result[col] = data[:, i].astype(np.float64)

    # Ensure sorted by MJD
    order = np.argsort(result['mjd'])
    for key in result:
        result[key] = result[key][order]

    return result


def update_bulletin_a(start_year=2000):
    """Fetch IERS Bulletin A data via astropy.

    Requires astropy to be installed.

    Parameters
    ----------
    start_year : int
        Start year for data range.

    Returns
    -------
    dict with keys 'mjd', 'dut1', 'x', 'y' as numpy arrays.
    """
    from astropy.utils.iers import IERS_Auto
    from astropy.time import Time

    iers = IERS_Auto.open()
    mjd = np.array(iers['MJD'].value, dtype=np.float64)
    dut1 = np.array(iers['UT1_UTC'].value, dtype=np.float64)
    xp = np.array(iers['PM_x'].value, dtype=np.float64)
    yp = np.array(iers['PM_y'].value, dtype=np.float64)

    # Filter by start year
    start_mjd = Time(f'{start_year}-01-01', scale='utc').mjd
    mask = mjd >= start_mjd
    return {
        'mjd': mjd[mask],
        'dut1': dut1[mask],
        'x': xp[mask],
        'y': yp[mask],
    }


def interpolate_bulletin_a(iers_data, mjd):
    """Interpolate IERS data at given MJD(s).

    Uses linear interpolation with leap-second correction for dut1.

    Parameters
    ----------
    iers_data : dict
        Output from load_bulletin_a or update_bulletin_a.
    mjd : float or array
        Modified Julian Date(s) to interpolate at.

    Returns
    -------
    dut1, xp, yp : float or arrays
        Interpolated values. Returns zeros for out-of-bounds MJDs.
    """
    mjd_arr = jnp.atleast_1d(jnp.asarray(mjd, dtype=jnp.float64))
    mjd_table = jnp.asarray(iers_data['mjd'])
    dut1_table = jnp.asarray(iers_data['dut1'])
    x_table = jnp.asarray(iers_data['x'])
    y_table = jnp.asarray(iers_data['y'])

    mjd_min = mjd_table[0]
    mjd_max = mjd_table[-1]
    n = len(mjd_table)

    # Fractional index
    idx_f = mjd_arr - mjd_min
    idx_lo = jnp.floor(idx_f).astype(jnp.int32)
    frac = idx_f - idx_lo

    # Clamp indices
    idx_lo = jnp.clip(idx_lo, 0, n - 2)
    idx_hi = idx_lo + 1

    # Linear interpolation
    dut1_lo = dut1_table[idx_lo]
    dut1_hi = dut1_table[idx_hi]

    # Leap second correction: if |dut1_hi - dut1_lo| > 0.5, adjust
    diff = dut1_hi - dut1_lo
    leap = jnp.where(diff > 0.5, 1.0, jnp.where(diff < -0.5, -1.0, 0.0))
    dut1_out = (1.0 - frac) * dut1_lo + frac * (dut1_hi - leap)

    x_out = (1.0 - frac) * x_table[idx_lo] + frac * x_table[idx_hi]
    y_out = (1.0 - frac) * y_table[idx_lo] + frac * y_table[idx_hi]

    # Zero out-of-bounds
    in_bounds = (mjd_arr >= mjd_min) & (mjd_arr < mjd_max)
    dut1_out = jnp.where(in_bounds, dut1_out, 0.0)
    x_out = jnp.where(in_bounds, x_out, 0.0)
    y_out = jnp.where(in_bounds, y_out, 0.0)

    # Squeeze back to scalar if input was scalar
    if jnp.ndim(mjd) == 0:
        return float(dut1_out[0]), float(x_out[0]), float(y_out[0])
    return dut1_out, x_out, y_out
