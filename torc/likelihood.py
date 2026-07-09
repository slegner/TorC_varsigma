"""DESI DR2 BAO likelihood for the TorC theory (noH0 anchoring).

    log L(theta) = -1/2 r^T C^{-1} r - 1/2 log|2 pi C|,   r = data - model

The model vector mixes D_M/r_d, D_H/r_d and D_V/r_d predictions built from
E(z) = H/H0 (torc.theory) and the single dimensionful combination
h0rd = H0 * r_d [km/s], sampled directly.

Defensive floor: a failed ODE solve surfaces as nan and is mapped to
logL = -1e30 ("impossible point"), so one bad particle cannot poison a
vmapped batch.  Out-of-prior points should additionally be routed through
torc.priors.loglikelihood_bounded, which avoids paying for doomed solves.
"""
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pandas import read_csv

from .theory import rhopsol_jax, OMEGA_R_DEFAULT

C_KMS  = 299792.458
OMEGAR = OMEGA_R_DEFAULT          # 8.24e-5, as everywhere in distances/

# ---------------------------------------------------------------------------
# Data (host side, once, at import).  Vendored in the repo at data/desidr2;
# override with $DESIDR2_DIR.
# ---------------------------------------------------------------------------
_here = os.environ.get(
    "DESIDR2_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "data", "desidr2"))
_data = read_csv(os.path.join(_here, "desidr2_mean.txt"),
                 header=None, index_col=None, sep=r"\s+", comment="#")
_cov  = np.loadtxt(os.path.join(_here, "desidr2_cov.txt"))

ZS        = jnp.array(_data.iloc[:, 0].to_numpy())     # 13 redshifts
DATA      = jnp.array(_data.iloc[:, 1].to_numpy())     # 13 measurements
_types    = _data.iloc[:, 2].to_numpy()                # what each row is
COV_INV   = jnp.array(np.linalg.inv(_cov))
LOGL_NORM = -0.5 * (len(_types) * np.log(2*np.pi) + np.linalg.slogdet(_cov)[1])

# one-hot masks: row i is exactly one of DM/DH/DV (no traced branching)
IS_DM = jnp.array(_types == "DM_over_rs", dtype=jnp.float64)
IS_DH = jnp.array(_types == "DH_over_rs", dtype=jnp.float64)
IS_DV = jnp.array(_types == "DV_over_rs", dtype=jnp.float64)

# fixed integration grid for D_M: fine, uniform, past the highest data z
Z_GRID = jnp.linspace(0.0, 2.6, 501)


def _cumtrapz(y, x):
    """Cumulative trapezoid, first entry 0 — the jit-able quad replacement."""
    dx = jnp.diff(x)
    steps = 0.5 * dx * (y[1:] + y[:-1])
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(steps)])


def model_vector(omega_m, h0rd, stigma, mups2, OmL):
    """Predict all 13 DESI numbers for one parameter point."""
    z_th, E_th, _ = rhopsol_jax(omega_m, OMEGAR, OmL, stigma, mups2)

    E_grid    = jnp.interp(Z_GRID, z_th, E_th)
    invE_grid = 1.0 / E_grid

    dm_grid = (C_KMS / h0rd) * _cumtrapz(invE_grid, Z_GRID)

    dm = jnp.interp(ZS, Z_GRID, dm_grid)
    dh = (C_KMS / h0rd) / jnp.interp(ZS, z_th, E_th)
    dv = (ZS * dm**2 * dh) ** (1.0/3.0)

    return IS_DM * dm + IS_DH * dh + IS_DV * dv


def loglikelihood(params):
    """params: dict with keys 'omega_m', 'h0rd', 'stigma', 'mups2', 'OmL'."""
    pred = model_vector(params["omega_m"], params["h0rd"],
                        params["stigma"],  params["mups2"], params["OmL"])
    r    = DATA - pred
    chi2 = r @ COV_INV @ r
    logl = -0.5 * chi2 + LOGL_NORM
    return jnp.where(jnp.isfinite(logl), logl, -1e30)
