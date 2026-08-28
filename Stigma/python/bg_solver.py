"""
Shared solver used by all background plotting scripts.

solve_one(cpr, stigma, ups2, omega_m, h, z_max, npts)
    → runs rhopsol and returns arrays on a uniform redshift grid
"""

import numpy as np
from scipy.interpolate import interp1d
from equations import omega_r, rhopsol, validate_stigma_constraints, StigmaConstraintError


def H_lcdm(z, omega_m, h):
    """ΛCDM Hubble parameter in km/s/Mpc."""
    H0     = h * 100.0
    Omr    = omega_r(h)
    OmL    = 1.0 - omega_m - Omr
    return H0 * np.sqrt(Omr * (1+z)**4 + omega_m * (1+z)**3 + OmL)


def solve_one(cpr, stigma, ups2, omega_m, h, z_max=3.0, npts=500):
    """
    Solve the Stigma ODEs and return arrays on a uniform z grid.

    Parameters
    ----------
    cpr    : ϖ₀, initial value of the varpi field
    stigma : ς, the Stigma coupling parameter
    ups2   : υ₂, the upsilon2 parameter
    omega_m, h : standard cosmological parameters
    z_max, npts : redshift range and grid density

    Returns dict with keys: z, a, Hz, weff
    Raises StigmaConstraintError if parameters are invalid.
    """
    validate_stigma_constraints(stigma, cpr)

    Omr = omega_r(h)
    OmL = 1.0 - omega_m - Omr
    H0  = h * 100.0

    asol, _, hsol, _, _, P_DE, rho_DE = rhopsol(
        cpr, omega_m, h, OmL, stigma, ups2
    )

    # a goes from ~0 to 1 at today; z = 1/a - 1
    z_raw = 1.0 / asol - 1.0

    mask = (z_raw >= 0.0) & (z_raw <= z_max + 0.1)
    if mask.sum() < 10:
        raise RuntimeError(f"Too few points in z=[0,{z_max}]")

    z_s    = z_raw[mask]
    Hz_s   = hsol[mask]
    weff_s = (P_DE / rho_DE)[mask]

    idx = np.argsort(z_s)
    z_s, Hz_s, weff_s = z_s[idx], Hz_s[idx], weff_s[idx]

    z_grid = np.linspace(0.0, min(z_max, z_s.max()), npts)
    a_grid = 1.0 / (1.0 + z_grid)
    kw     = dict(bounds_error=False, fill_value=np.nan)

    return {
        "z":    z_grid,
        "a":    a_grid,
        "Hz":   interp1d(z_s, Hz_s,   **kw)(z_grid),
        "weff": interp1d(z_s, weff_s, **kw)(z_grid),
        "H0":   H0,
    }
