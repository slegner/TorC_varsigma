"""DESI DR2 BAO + Type Ia supernovae, sharing a single background solve.

The naive combination calls rhopsol_jax twice -- once inside
torc.likelihood_lam and once inside torc.likelihood_sn -- which doubles the
cost of the only expensive step.  Here the ODE is solved ONCE and both data
sets are evaluated from the same (z, E).

The BAO data block (redshifts, measurements, covariance, normalisation) is
imported from torc.likelihood rather than restated, so the combined
likelihood cannot drift from the BAO-only one.

THE SUPERNOVAE ADD NO SAMPLED PARAMETERS: M and H0 are marginalised analytically
(see torc/likelihood_sn.py).  It constrains the SHAPE of E(z) only; h0rd
remains a BAO-only quantity.

    from torc.likelihood_combined import make_loglikelihood
    logl = make_loglikelihood(use_bao=True, use_sn=True)
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .likelihood import (ZS, DATA, COV_INV, LOGL_NORM, IS_DM, IS_DH, IS_DV,
                         Z_GRID, C_KMS, MUPS2_FIXED)
from .theory import OMEGA_R_DEFAULT
from . import theory_lam
from . import likelihood_sn as SN

OMEGAR = OMEGA_R_DEFAULT


def _cumtrapz(y, x):
    dx = jnp.diff(x)
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5*dx*(y[1:] + y[:-1]))])


def _bao_logl(z_th, E_th, h0rd):
    E_grid = jnp.interp(Z_GRID, z_th, E_th)
    dm_grid = (C_KMS / h0rd) * _cumtrapz(1.0 / E_grid, Z_GRID)
    dm = jnp.interp(ZS, Z_GRID, dm_grid)
    dh = (C_KMS / h0rd) / jnp.interp(ZS, z_th, E_th)
    dv = (ZS * dm**2 * dh) ** (1.0/3.0)
    pred = IS_DM*dm + IS_DH*dh + IS_DV*dv
    r = DATA - pred
    return -0.5 * (r @ COV_INV @ r) + LOGL_NORM


def _sn_logl(z_th, E_th, data):
    hdl = SN.h0_dl_over_c(z_th, E_th, data["zhd"], data["zhel"])
    y = 5.0*jnp.log10(hdl) - data["mb"]
    Y = data["onesigma"] @ y
    return (-0.5*(y @ data["invcov_tilde"] @ y)
            + jnp.log((data["b"]**(Y+1) - data["a"]**(Y+1)) / (Y+1))
            + data["lognorm"])


def make_loglikelihood(use_bao=True, use_sn=True, sn_data=None,
                       sn_dataset="pantheonplus"):
    """logL(params) over the requested data sets.

    sn_dataset: "pantheonplus" (vendored) or "desy5" (needs $DESY5_DIR).
    sn_data:    a preloaded dict from torc.likelihood_sn; loaded lazily from
                sn_dataset when omitted.
    """
    if not (use_bao or use_sn):
        raise ValueError("no data selected")
    if use_sn and sn_data is None:
        if sn_dataset == "pantheonplus":
            sn_data = SN.load_pantheonplus()
        elif sn_dataset == "desy5":
            sn_data = SN.load_desy5()
        else:
            raise ValueError(f"unknown sn_dataset {sn_dataset!r}")

    def loglikelihood(params):
        z_th, E_th, _ = theory_lam.rhopsol_jax(
            params["omega_m"], OMEGAR, params.get("OmL", 0.0),
            params["stigma"], params.get("mups2", MUPS2_FIXED),
            params.get("lam", 0.0))
        total = 0.0
        if use_bao:
            total = total + _bao_logl(z_th, E_th, params["h0rd"])
        if use_sn:
            total = total + _sn_logl(z_th, E_th, sn_data)
        return jnp.where(jnp.isfinite(total), total, -1e30)

    return loglikelihood
