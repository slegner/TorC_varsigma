"""Flat LCDM fit to the same DESI DR2 BAO likelihood, for comparison with
TorC (same data vector, covariance, distance formulas, priors on omega_m
and h0rd).  E(z) is analytic, so nested sampling takes seconds.

Usage:  python scripts/run_lcdm.py       (from the repo root)
Writes chains/lcdm_ns.npz.
"""
import sys, os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import blackjax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.likelihood import (ZS, DATA, COV_INV, LOGL_NORM, IS_DM, IS_DH,
                             IS_DV, Z_GRID, OMEGAR, C_KMS, _cumtrapz)

def E_lcdm(z, om):
    return jnp.sqrt(om*(1+z)**3 + OMEGAR*(1+z)**4 + (1 - om - OMEGAR))

def loglikelihood(params):
    om, h0rd = params["omega_m"], params["h0rd"]
    dm_grid = (C_KMS/h0rd) * _cumtrapz(1.0/E_lcdm(Z_GRID, om), Z_GRID)
    dm = jnp.interp(ZS, Z_GRID, dm_grid)
    dh = (C_KMS/h0rd) / E_lcdm(ZS, om)
    dv = (ZS * dm**2 * dh) ** (1.0/3.0)
    r = DATA - (IS_DM*dm + IS_DH*dh + IS_DV*dv)
    logl = -0.5*(r @ COV_INV @ r) + LOGL_NORM
    return jnp.where(jnp.isfinite(logl), logl, -1e30)

BOUNDS = {"omega_m": (0.01, 0.99), "h0rd": (3650.0, 18250.0)}

def logprior_fn(p):
    return sum(jax.scipy.stats.uniform.logpdf(p[n], lo, hi-lo)
               for n, (lo, hi) in BOUNDS.items())

def sample_prior(key, num):
    keys = jax.random.split(key, len(BOUNDS))
    return {name: jax.random.uniform(k, (num,), minval=lo, maxval=hi)
            for k, (name, (lo, hi)) in zip(keys, BOUNDS.items())}

if __name__ == "__main__":
    algo = blackjax.nss(logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood,
                        num_delete=100, num_inner_steps=2*5)
    rng = jax.random.PRNGKey(0)
    rng, k = jax.random.split(rng)
    live = jax.jit(algo.init)(sample_prior(k, 500))
    step = jax.jit(algo.step)
    dead = []
    while not live.logZ_live - live.logZ < -3:
        rng, k = jax.random.split(rng)
        live, d = step(k, live)
        dead.append(d)
    dead = blackjax.ns.utils.finalise(live, dead)

    from anesthetic import NestedSamples
    s = NestedSamples({n: np.asarray(v) for n, v in dead.particles.items()},
                      logL=np.asarray(dead.loglikelihood),
                      logL_birth=np.asarray(dead.loglikelihood_birth))
    print("LCDM (DESI DR2 BAO):")
    print(f"  ln Z    = {s.logZ():.3f},  D_KL = {s.D_KL():.3f}")
    print(f"  omega_m = {s['omega_m'].mean():.4f} +- {s['omega_m'].std():.4f}")
    print(f"  h0rd    = {s['h0rd'].mean():.1f} +- {s['h0rd'].std():.1f} km/s")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "chains", "lcdm_ns.npz")
    np.savez(out,
             logL=np.asarray(dead.loglikelihood),
             logL_birth=np.asarray(dead.loglikelihood_birth),
             logZ=s.logZ(),
             **{k: np.asarray(v) for k, v in dead.particles.items()})
    print("saved ->", out)
