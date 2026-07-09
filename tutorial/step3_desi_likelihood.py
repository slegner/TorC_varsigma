"""
STEP 3 — Building the DESI DR2 likelihood in JAX.
=================================================

Run me with:
    source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
    cd /Users/starfish/Desktop/PhD2025-26/distances/TorC
    python jax_tutorial/step3_desi_likelihood.py

-----------------------------------------------------------------------------
WHAT A LIKELIHOOD IS
-----------------------------------------------------------------------------
The likelihood is the probability of the DATA given the PARAMETERS:

    L(theta) = P(data | theta)

It is the meeting point of theory and experiment: theta goes in, the theory
predicts what the instrument should have seen, and L scores how plausible
the actual measurements are under that prediction. The sampler's whole job
is to explore how L varies over theta.

DESI reports its 13 BAO measurements with Gaussian errors and a covariance
matrix C (measurements from the same tracers are correlated!). For Gaussian
data the likelihood is the multivariate normal density:

    log L(theta) = -1/2 * r^T C^{-1} r  - 1/2 * log|2 pi C|
    with residuals r = data - model(theta)

The first term is -chi^2/2 — the only theta-dependent part. The second is
a constant: it does not move the posterior at all, but it DOES set the
absolute scale of the evidence log Z, so we keep it (it costs nothing).

-----------------------------------------------------------------------------
FROM E(z) TO THE MODEL VECTOR
-----------------------------------------------------------------------------
The theory (theory_jax.rhopsol_jax) hands us E(z) = H(z)/H0. DESI measures
three kinds of distance ratios (see ../distances/distances.py for the
LCDM version of the same formulas):

    D_H/r_d = c / (H0 r_d E(z))                       "Hubble distance"
    D_M/r_d = c / (H0 r_d) * INT_0^z dz'/E(z')        "transverse comoving"
    D_V/r_d = (z * (D_M/r_d)^2 * (D_H/r_d))^(1/3)     "volume average"

Everything dimensionful enters through the single combination
    h0rd = H0 * r_d   [km/s]
which we sample directly — the same convention as the distances/ folder
(no h, no separate r_d, no CAMB).

The integral INT dz'/E is scipy.quad territory, but quad is a Python loop
on data (rule 2!) — so we use a cumulative trapezoid on a fixed fine grid:
one jnp.cumsum, exact enough (validated below against quad to ~1e-5),
and it gives D_M at ALL redshifts in one shot.

-----------------------------------------------------------------------------
DEFENSIVE LIKELIHOODS
-----------------------------------------------------------------------------
A sampler explores the WHOLE prior, including corners where the ODE solver
fails (theory_jax passes throw=False, so failures surface as inf/nan, not
crashes — one bad particle must not kill a vmapped batch of 500). The last
line of the likelihood catches any non-finite chi^2 and returns -1e30:
"this parameter point is impossible", and the sampler moves on.
-----------------------------------------------------------------------------
"""

import sys, os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from pandas import read_csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from theory_jax import rhopsol_jax, OMEGA_R_DEFAULT

C_KMS  = 299792.458
OMEGAR = OMEGA_R_DEFAULT          # 8.24e-5, as everywhere in distances/

# ---------------------------------------------------------------------------
# 1. Load the data (host side, once, at import — numpy/pandas is fine here:
#    this is not traced, it just prepares constant arrays for jit)
# ---------------------------------------------------------------------------
_here = os.path.join(os.path.dirname(__file__), "..", "..", "desidr2")
_data = read_csv(os.path.join(_here, "desidr2_mean.txt"),
                 header=None, index_col=None, sep=r"\s+", comment="#")
_cov  = np.loadtxt(os.path.join(_here, "desidr2_cov.txt"))

ZS        = jnp.array(_data.iloc[:, 0].to_numpy())     # 13 redshifts
DATA      = jnp.array(_data.iloc[:, 1].to_numpy())     # 13 measurements
_types    = _data.iloc[:, 2].to_numpy()                # what each row is
COV_INV   = jnp.array(np.linalg.inv(_cov))
# constant part of log L: -1/2 log|2 pi C|
LOGL_NORM = -0.5 * (len(_types) * np.log(2*np.pi) + np.linalg.slogdet(_cov)[1])

# one-hot masks: row i is exactly one of DM/DH/DV. Precomputed as constants,
# they let us build the mixed model vector without any Python branching
# on traced values (rule 2 again).
IS_DM = jnp.array(_types == "DM_over_rs", dtype=jnp.float64)
IS_DH = jnp.array(_types == "DH_over_rs", dtype=jnp.float64)
IS_DV = jnp.array(_types == "DV_over_rs", dtype=jnp.float64)

# fixed integration grid for D_M: fine, uniform, past the highest data z
# (501 points -> trapezoid error ~1e-5 relative, ~1000x below DESI errors)
Z_GRID = jnp.linspace(0.0, 2.6, 501)


# ---------------------------------------------------------------------------
# 2. E(z) -> the 13-component model vector
# ---------------------------------------------------------------------------

def _cumtrapz(y, x):
    """Cumulative trapezoid, first entry 0 — the jit-able quad replacement."""
    dx = jnp.diff(x)
    steps = 0.5 * dx * (y[1:] + y[:-1])
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(steps)])


def model_vector(omega_m, h0rd, stigma, mups2, OmL):
    """Predict all 13 DESI numbers for one parameter point.

    OmL is a free parameter (NOT tied to flatness): the data get to decide
    how much cosmological constant they want alongside the stigma field.
    """
    z_th, E_th, _ = rhopsol_jax(omega_m, OMEGAR, OmL, stigma, mups2)

    # E on the fine distance grid (z_th is ascending, so interp is valid)
    E_grid    = jnp.interp(Z_GRID, z_th, E_th)
    invE_grid = 1.0 / E_grid

    dm_grid = (C_KMS / h0rd) * _cumtrapz(invE_grid, Z_GRID)   # D_M/r_d on grid

    dm = jnp.interp(ZS, Z_GRID, dm_grid)                      # at the data z's
    dh = (C_KMS / h0rd) / jnp.interp(ZS, z_th, E_th)
    dv = (ZS * dm**2 * dh) ** (1.0/3.0)

    return IS_DM * dm + IS_DH * dh + IS_DV * dv


# ---------------------------------------------------------------------------
# 3. The log-likelihood
# ---------------------------------------------------------------------------

def loglikelihood(params):
    """params: dict with keys 'omega_m', 'h0rd', 'stigma', 'mups2', 'OmL'.

    (A dict-of-scalars "pytree" — exactly the structure blackjax expects,
    and self-documenting: no remembering which column is which.)
    """
    pred = model_vector(params["omega_m"], params["h0rd"],
                        params["stigma"],  params["mups2"], params["OmL"])
    r    = DATA - pred
    chi2 = r @ COV_INV @ r
    logl = -0.5 * chi2 + LOGL_NORM
    # defensive floor: failed ODE / insane parameters -> "impossible", not nan
    return jnp.where(jnp.isfinite(logl), logl, -1e30)


loglikelihood_jit = jax.jit(loglikelihood)


# ===========================================================================
# VALIDATION + TIMING
# ===========================================================================
if __name__ == "__main__":
    import time
    from scipy.integrate import quad
    from scipy.interpolate import interp1d
    from test_theory_cpr1 import rhopsol as rhopsol_scipy, omega_r as omega_r_h

    # --- reference: same likelihood, all-scipy (quad + solve_ivp) ----------
    def loglike_scipy(omega_m, h0rd, stigma, mups2, OmL):
        # scipy theory still wants h; use the h whose omega_r matches OMEGAR
        h_eff = np.sqrt(omega_r_h(1.0) / OMEGAR)
        z, H, *_ = rhopsol_scipy(None, omega_m, h_eff, OmL, stigma, mups2)
        Efun = interp1d(z, H / H[0], kind="cubic")
        dm = np.array([C_KMS/h0rd * quad(lambda zp: 1/float(Efun(zp)), 0, zi,
                                         limit=200)[0] for zi in np.asarray(ZS)])
        dh = C_KMS / h0rd / np.array([float(Efun(zi)) for zi in np.asarray(ZS)])
        dv = (np.asarray(ZS) * dm**2 * dh) ** (1/3)
        pred = (np.asarray(IS_DM)*dm + np.asarray(IS_DH)*dh + np.asarray(IS_DV)*dv)
        r = np.asarray(DATA) - pred
        return -0.5 * r @ np.asarray(COV_INV) @ r + float(LOGL_NORM)

    flatL = lambda om: 1 - om - OMEGAR      # the flat value, for reference points
    print(f"{'omega_m':>8} {'h0rd':>8} {'stigma':>7} {'mups2':>6} {'OmL':>7} | "
          f"{'logL (jax)':>12} {'logL (scipy)':>12} {'diff':>9}")
    for om, h0rd, sig, mu, oml in [(0.2969, 10157.5, 1.0001, 4/3, flatL(0.2969)),
                                   (0.2969, 10157.5, 1.01,   4/3, flatL(0.2969)),
                                   (0.2969, 10157.5, 1.01,   4/3, 0.30),
                                   (0.2969, 10157.5, 1.01,   4/3, 0.0),
                                   (0.32,   10000.0, 1.3,    1.0, 0.60)]:
        p = dict(omega_m=om, h0rd=h0rd, stigma=sig, mups2=mu, OmL=oml)
        lj = float(loglikelihood_jit(p))
        ls = loglike_scipy(om, h0rd, sig, mu, oml)
        print(f"{om:8.4f} {h0rd:8.1f} {sig:7.4f} {mu:6.3f} {oml:7.3f} | "
              f"{lj:12.4f} {ls:12.4f} {lj-ls:9.2e}")

    # --- defensive floor: garbage parameters must give -1e30, not nan ------
    bad = dict(omega_m=0.3, h0rd=10000.0, stigma=0.2, mups2=1.0,  # stigma<1/4!
               OmL=0.7)
    print("\ngarbage params ->", float(loglikelihood_jit(bad)))

    # --- timing -------------------------------------------------------------
    p0 = dict(omega_m=0.3, h0rd=10150.0, stigma=1.2, mups2=1.2, OmL=0.7)
    _ = loglikelihood_jit(p0)
    t0 = time.perf_counter(); _ = jax.block_until_ready(loglikelihood_jit(p0))
    print(f"\nsingle logL (jitted)  : {(time.perf_counter()-t0)*1e3:7.2f} ms")

    n = 500
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    batch = dict(
        omega_m=jax.random.uniform(keys[0], (n,), minval=0.25, maxval=0.35),
        h0rd   =jax.random.uniform(keys[1], (n,), minval=9500., maxval=10800.),
        stigma =jax.random.uniform(keys[2], (n,), minval=1.001, maxval=2.0),
        mups2  =jax.random.uniform(keys[3], (n,), minval=0.6,   maxval=2.0),
        OmL    =jax.random.uniform(keys[3], (n,), minval=0.0,   maxval=0.9),
    )
    batched = jax.jit(jax.vmap(loglikelihood))
    _ = jax.block_until_ready(batched(batch))
    t0 = time.perf_counter(); _ = jax.block_until_ready(batched(batch))
    dt = time.perf_counter() - t0
    print(f"vmap over {n} logL     : {dt*1e3:7.1f} ms  ({dt/n*1e3:.2f} ms each)")
