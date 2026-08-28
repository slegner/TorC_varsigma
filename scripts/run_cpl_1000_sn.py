"""Flat CPL (w0waCDM) against DESI DR2 BAO and/or supernovae -- 1000 live.

    w(a) = w0 + wa (1 - a)
    f_DE(z) = (1+z)^{3(1+w0+wa)} exp(-3 wa z / (1+z))

Companion to scripts/run_lcdm_1000_sn.py and run_ns_julia_1000_sn.py: same
data, covariance, distance formulas, priors on the shared parameters
(omega_m, h0rd), and sampler settings, so every ln Z in chains/ is directly
comparable.  E(z) is analytic, so no ODE and no Julia backend.

    TORC_DATA=bao|sn|bao+sn      (default bao+sn)
    TORC_SN=pantheonplus|desy5   (default pantheonplus)

    python scripts/run_cpl_1000_sn.py                    # DESI + Pantheon+
    TORC_SN=desy5 python scripts/run_cpl_1000_sn.py      # DESI + DESY5

PRIORS
------
omega_m ~ U[0.01, 0.99] and h0rd ~ U[3650, 18250] reproduce
torc/priors.py, so CPL sits in the same prior box as the TorC runs.
w0 ~ U[-3, 1] and wa ~ U[-3, 2] follow distances/common.py, with the hard
constraint w0 + wa < 0 (the equation of state accelerating at early times).

THAT CONSTRAINT IS NORMALISED HERE, and the existing cpl_nss.py does not do
this: it applies w0+wa<0 as a -inf gate on an otherwise unit-normalised
uniform density, leaving the prior integrating to 0.775 rather than 1.  For a
Bayes factor that is a constant +0.2549 nat offset in ln Z (= -log 0.775),
which is small but not negligible against ln Z errors of ~0.1.  Here the
(w0, wa) density carries 1/15.5 (the allowed area, verified analytically and
by Monte Carlo) and prior draws are rejection-sampled from the same region,
so the prior integrates to 1 and ln Z needs no correction.
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

W0_RANGE = (-3.0, 1.0)
WA_RANGE = (-3.0, 2.0)
# area of {(w0,wa) in box : w0 + wa < 0}; analytic 15.5, MC 15.5028 +- 0.0019
_W_AREA = 15.5

BOUNDS = {"omega_m": (0.01, 0.99)}
if USE_BAO:                       # h0rd only enters the BAO likelihood
    BOUNDS["h0rd"] = (3650.0, 18250.0)
BOUNDS["w0"] = W0_RANGE
BOUNDS["wa"] = WA_RANGE
NUM_INNER_STEPS = len(BOUNDS) * 5

_SN_TAG = {"pantheonplus": "pp", "desy5": "desy5"}[SN_SET]
_NAME = {"bao": "bao", "sn": _SN_TAG, "bao+sn": f"bao_{_SN_TAG}"}[DATA_SEL]
OUT = os.path.join(HERE, "..", "chains", f"cpl_ns_1000_{_NAME}.npz")

ZG_SN = jnp.linspace(0.0, 2.5, 4001)


def f_de(z, w0, wa):
    return (1.0 + z) ** (3.0*(1.0 + w0 + wa)) * jnp.exp(-3.0*wa*z/(1.0 + z))


def E_cpl(z, om, w0, wa):
    return jnp.sqrt(om*(1+z)**3 + OMEGAR*(1+z)**4
                    + (1.0 - om - OMEGAR)*f_de(z, w0, wa))


def make_loglikelihood(sn_data):
    def loglikelihood(params):
        om, w0, wa = params["omega_m"], params["w0"], params["wa"]
        total = 0.0
        if USE_BAO:
            h0rd = params["h0rd"]
            dm_grid = (C_KMS/h0rd)*_cumtrapz(1.0/E_cpl(Z_GRID, om, w0, wa),
                                             Z_GRID)
            dm = jnp.interp(ZS, Z_GRID, dm_grid)
            dh = (C_KMS/h0rd) / E_cpl(ZS, om, w0, wa)
            dv = (ZS * dm**2 * dh) ** (1.0/3.0)
            r = DATA - (IS_DM*dm + IS_DH*dh + IS_DV*dv)
            total = total + (-0.5*(r @ COV_INV @ r) + LOGL_NORM)
        if USE_SN:
            hdl = SN.h0_dl_over_c(ZG_SN, E_cpl(ZG_SN, om, w0, wa),
                                  sn_data["zhd"], sn_data["zhel"])
            y = 5.0*jnp.log10(hdl) - sn_data["mb"]
            Y = sn_data["onesigma"] @ y
            total = total + (-0.5*(y @ sn_data["invcov_tilde"] @ y)
                             + jnp.log((sn_data["b"]**(Y+1)
                                        - sn_data["a"]**(Y+1))/(Y+1))
                             + sn_data["lognorm"])
        # hard gate: an out-of-prior particle with a finite logL would be
        # chosen as a slice start with logslice = -inf and accept anything
        inb = (params["w0"] + params["wa"] < 0.0)
        for n, (lo, hi) in BOUNDS.items():
            inb = inb & (params[n] >= lo) & (params[n] <= hi)
        total = jnp.where(inb, total, -1e30)
        return jnp.where(jnp.isfinite(total), total, -1e30)
    return loglikelihood


def logprior_fn(p):
    lp = 0.0
    for n, (lo, hi) in BOUNDS.items():
        if n in ("w0", "wa"):
            continue
        lp = lp + jax.scipy.stats.uniform.logpdf(p[n], lo, hi - lo)
    inw = ((p["w0"] >= W0_RANGE[0]) & (p["w0"] <= W0_RANGE[1])
           & (p["wa"] >= WA_RANGE[0]) & (p["wa"] <= WA_RANGE[1])
           & (p["w0"] + p["wa"] < 0.0))
    return jnp.where(inw, lp - jnp.log(_W_AREA), -jnp.inf)


def sample_prior(key, num):
    """Draw from the prior, rejection-sampling (w0, wa) into w0 + wa < 0."""
    keys = jax.random.split(key, len(BOUNDS) + 2)
    out = {}
    i = 0
    for name, (lo, hi) in BOUNDS.items():
        if name in ("w0", "wa"):
            continue
        out[name] = jax.random.uniform(keys[i], (num,), minval=lo, maxval=hi)
        i += 1
    # oversample by 1/0.775 with headroom, then take the first `num` accepted
    m = int(num / 0.775 * 3) + 1000
    w0 = jax.random.uniform(keys[i], (m,), minval=W0_RANGE[0], maxval=W0_RANGE[1])
    wa = jax.random.uniform(keys[i+1], (m,), minval=WA_RANGE[0], maxval=WA_RANGE[1])
    ok = np.asarray(w0 + wa < 0.0)
    idx = np.flatnonzero(ok)[:num]
    if len(idx) < num:
        raise RuntimeError("prior rejection sampling came up short")
    out["w0"] = jnp.asarray(np.asarray(w0)[idx])
    out["wa"] = jnp.asarray(np.asarray(wa)[idx])
    return out


if __name__ == "__main__":
    print(f"CPL    data: {DATA_SEL}" + (f" ({SN_SET})" if USE_SN else ""))
    print(f"ndim={len(BOUNDS)} ({', '.join(BOUNDS)})  NUM_LIVE={NUM_LIVE}")
    print(f"w0+wa<0 imposed; (w0,wa) prior density 1/{_W_AREA}")
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
    print("RUNCPL DONE")
