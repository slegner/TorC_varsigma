"""Flat LCDM against DESI DR2 BAO and/or supernovae -- 1000 live points.

The reference model every TorC variant is measured against.  Same data,
covariance, distance formulas, priors on the shared parameters (omega_m,
h0rd) and sampler settings as scripts/run_ns_julia_1000_sn.py, so ln Z is
directly comparable.

    TORC_DATA=bao|sn|bao+sn      (default bao+sn)
    TORC_SN=pantheonplus|desy5   (default pantheonplus)

    python scripts/run_lcdm_1000_sn.py                      # BAO + Pantheon+
    TORC_SN=desy5 python scripts/run_lcdm_1000_sn.py        # BAO + DESY5
    TORC_DATA=bao python scripts/run_lcdm_1000_sn.py        # BAO alone

E(z) is analytic, so no ODE and no Julia backend -- these take seconds to
minutes rather than half an hour.

NOTE ON COMBINING SUPERNOVA SAMPLES: DESY5 and Pantheon+ share 335 objects
(18% of DESY5, 22% of Pantheon+ -- all 118 Foundation, 65 CfA and 145 DES),
so their likelihoods must NOT be summed; they are alternatives.  The
published analysis does the same (zenodo/chains has desidr2_des5y and
desidr2_pantheonplus separately, never combined).
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import blackjax
import tqdm

from torc.likelihood import (ZS, DATA, COV_INV, LOGL_NORM, IS_DM, IS_DH,
                             IS_DV, Z_GRID, OMEGAR, C_KMS, _cumtrapz)
from torc import likelihood_sn as SN

SN_SET = os.environ.get("TORC_SN", "pantheonplus").lower()
DATA_SEL = os.environ.get("TORC_DATA", "bao+sn").lower()
if SN_SET not in ("pantheonplus", "desy5"):
    raise SystemExit(f"TORC_SN must be pantheonplus or desy5 (got {SN_SET!r})")
if DATA_SEL not in ("bao", "sn", "bao+sn"):
    raise SystemExit(f"TORC_DATA must be bao, sn or bao+sn (got {DATA_SEL!r})")
USE_BAO, USE_SN = "bao" in DATA_SEL, "sn" in DATA_SEL

NUM_LIVE, NUM_DELETE, SEED = 1000, 100, 0
BOUNDS = {"omega_m": (0.01, 0.99)}
if USE_BAO:                      # h0rd only enters the BAO likelihood
    BOUNDS["h0rd"] = (3650.0, 18250.0)
NUM_INNER_STEPS = len(BOUNDS) * 5

_SN_TAG = {"pantheonplus": "pp", "desy5": "desy5"}[SN_SET]
_NAME = {"bao": "bao", "sn": _SN_TAG, "bao+sn": f"bao_{_SN_TAG}"}[DATA_SEL]
OUT = os.path.join(HERE, "..", "chains", f"lcdm_ns_1000_{_NAME}.npz")

# a fine z grid for the SN luminosity distance (SNe reach z ~ 2.26)
ZG_SN = jnp.linspace(0.0, 2.5, 4001)


def E_lcdm(z, om):
    return jnp.sqrt(om*(1+z)**3 + OMEGAR*(1+z)**4 + (1 - om - OMEGAR))


def make_loglikelihood(sn_data):
    def loglikelihood(params):
        om = params["omega_m"]
        total = 0.0
        if USE_BAO:
            h0rd = params["h0rd"]
            dm_grid = (C_KMS/h0rd) * _cumtrapz(1.0/E_lcdm(Z_GRID, om), Z_GRID)
            dm = jnp.interp(ZS, Z_GRID, dm_grid)
            dh = (C_KMS/h0rd) / E_lcdm(ZS, om)
            dv = (ZS * dm**2 * dh) ** (1.0/3.0)
            r = DATA - (IS_DM*dm + IS_DH*dh + IS_DV*dv)
            total = total + (-0.5*(r @ COV_INV @ r) + LOGL_NORM)
        if USE_SN:
            # same analytic M/H0 marginalisation as torc/likelihood_sn.py
            hdl = SN.h0_dl_over_c(ZG_SN, E_lcdm(ZG_SN, om),
                                  sn_data["zhd"], sn_data["zhel"])
            y = 5.0*jnp.log10(hdl) - sn_data["mb"]
            Y = sn_data["onesigma"] @ y
            total = total + (-0.5*(y @ sn_data["invcov_tilde"] @ y)
                             + jnp.log((sn_data["b"]**(Y+1)
                                        - sn_data["a"]**(Y+1))/(Y+1))
                             + sn_data["lognorm"])
        return jnp.where(jnp.isfinite(total), total, -1e30)
    return loglikelihood


def logprior_fn(p):
    return sum(jax.scipy.stats.uniform.logpdf(p[n], lo, hi-lo)
               for n, (lo, hi) in BOUNDS.items())


def sample_prior(key, num):
    keys = jax.random.split(key, len(BOUNDS))
    return {name: jax.random.uniform(k, (num,), minval=lo, maxval=hi)
            for k, (name, (lo, hi)) in zip(keys, BOUNDS.items())}


if __name__ == "__main__":
    print(f"LCDM   data: {DATA_SEL}" + (f" ({SN_SET})" if USE_SN else ""))
    print(f"ndim={len(BOUNDS)} ({', '.join(BOUNDS)})  NUM_LIVE={NUM_LIVE}")
    if os.path.exists(OUT):
        print(f"\nREFUSING to overwrite {OUT}\nmove or delete it first.")
        sys.exit(1)

    sn_data = None
    if USE_SN:
        sn_data = (SN.load_pantheonplus() if SN_SET == "pantheonplus"
                   else SN.load_desy5())
        print(f"{SN_SET}: {len(sn_data['zhd'])} supernovae")

    algo = blackjax.nss(logprior_fn=logprior_fn,
                        loglikelihood_fn=make_loglikelihood(sn_data),
                        num_delete=NUM_DELETE,
                        num_inner_steps=NUM_INNER_STEPS)
    rng = jax.random.PRNGKey(SEED)
    rng, k = jax.random.split(rng)
    t0 = time.perf_counter()
    live = jax.jit(algo.init)(sample_prior(k, NUM_LIVE))
    jax.block_until_ready(live)
    step = jax.jit(algo.step)
    dead = []
    with tqdm.tqdm(desc="dead points", unit=" pts") as pbar:
        while not live.logZ_live - live.logZ < -3:
            rng, k = jax.random.split(rng)
            live, d = step(k, live)
            dead.append(d)
            pbar.update(NUM_DELETE)
    run_t = time.perf_counter() - t0
    dead = blackjax.ns.utils.finalise(live, dead)

    from anesthetic import NestedSamples
    s = NestedSamples({n: np.asarray(v) for n, v in dead.particles.items()},
                      logL=np.asarray(dead.loglikelihood),
                      logL_birth=np.asarray(dead.loglikelihood_birth))
    lz = s.logZ(1000)
    print(f"\nrun time: {run_t:.1f} s")
    print(f"ln Z = {lz.mean():.4f} +- {lz.std():.4f}   D_KL = {s.D_KL():.2f}")
    for p in BOUNDS:
        print(f"  {p:8s} = {s[p].mean():11.4f} +- {s[p].std():.4f}")

    np.savez(OUT, logL=np.asarray(dead.loglikelihood),
             logL_birth=np.asarray(dead.loglikelihood_birth),
             logZ=float(lz.mean()), logZ_err=float(lz.std()),
             num_live=NUM_LIVE, num_delete=NUM_DELETE,
             num_inner_steps=NUM_INNER_STEPS, run_time=run_t,
             data=DATA_SEL, sn_dataset=SN_SET if USE_SN else "",
             **{k: np.asarray(v) for k, v in dead.particles.items()})
    print(f"saved -> {OUT}")
    print("RUNLCDM DONE")
