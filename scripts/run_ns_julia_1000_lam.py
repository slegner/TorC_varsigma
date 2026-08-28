"""Production nested sampling of the LAMBDA model: 1000 live, Julia backend.

Mirrors scripts/run_ns_julia_1000.py exactly -- same NUM_LIVE, NUM_DELETE,
num_inner_steps rule, seed, convergence criterion and output format -- so the
resulting ln Z is comparable to the no-lambda chain produced by that script.

Two configurations, selected by the environment:

    python scripts/run_ns_julia_1000_lam.py                  # OmL and lam both sampled (5D)
    TORC_SAMPLE_OML=0 python scripts/run_ns_julia_1000_lam.py  # OmL = 0, lam alone (4D)

Output: chains/torc_ns_julia_1000_lam[_noOmL][_<stigmaprior>].npz

NOTE ON COMPARABILITY: chains/torc_ns_julia_1000_powerlaw.npz was produced
BEFORE the stigma sampling-bound fix in torc/priors.py (the power-law branch
used to floor logL above stigma = 3).  Compare against a no-lambda baseline
run after that fix, not against that file.

Run:
    caffeinate -ims python scripts/run_ns_julia_1000_lam.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import blackjax
import tqdm

from torc.priors_lam import (PRIOR_BOUNDS, logprior_fn, sample_prior,
                             loglikelihood_bounded, SAMPLE_OML,
                             STIGMA_PRIOR, STIGMA_LAMBDA, LAM_BOUNDS)
import torc.julia_ipc_lam as ji

NUM_LIVE = 1000
NUM_DELETE = NUM_LIVE // 10
NUM_INNER_STEPS = len(PRIOR_BOUNDS) * 5
SEED = 0

if STIGMA_PRIOR == "uniform":
    _TAG = ""
elif STIGMA_PRIOR == "powerlaw":
    _TAG = "_powerlaw"
else:
    _TAG = f"_{STIGMA_PRIOR}_lam{STIGMA_LAMBDA:g}".replace(".", "p")
_OML = "" if SAMPLE_OML else "_noOmL"
OUT = os.path.join(HERE, "..", "chains",
                   f"torc_ns_julia_1000_lam{_OML}{_TAG}.npz")

if __name__ == "__main__":
    print(f"ndim={len(PRIOR_BOUNDS)} ({', '.join(PRIOR_BOUNDS)})")
    print(f"SAMPLE_OML={SAMPLE_OML}   lam prior: uniform{LAM_BOUNDS}")
    print(f"stigma prior: {STIGMA_PRIOR}  bounds={PRIOR_BOUNDS['stigma']}")
    print(f"NUM_LIVE={NUM_LIVE} NUM_DELETE={NUM_DELETE} "
          f"NUM_INNER_STEPS={NUM_INNER_STEPS} seed={SEED}")
    if os.path.exists(OUT):
        print(f"\nREFUSING to overwrite {OUT}\nmove or delete it first.")
        sys.exit(1)

    ji.activate()

    rng = jax.random.PRNGKey(SEED)
    algo = blackjax.nss(logprior_fn=logprior_fn,
                        loglikelihood_fn=loglikelihood_bounded,
                        num_delete=NUM_DELETE,
                        num_inner_steps=NUM_INNER_STEPS)
    rng, k = jax.random.split(rng)
    t0 = time.perf_counter()
    live = jax.jit(algo.init)(sample_prior(k, NUM_LIVE))
    jax.block_until_ready(live)
    print(f"init done in {time.perf_counter()-t0:.1f}s", flush=True)

    step = jax.jit(algo.step)
    dead, n = [], 0
    t0 = time.perf_counter()
    with tqdm.tqdm(desc="dead points", unit=" pts") as pbar:
        while not (live.logZ_live - live.logZ < -3):
            rng, k = jax.random.split(rng)
            live, d = step(k, live)
            dead.append(d)
            n += 1
            pbar.update(NUM_DELETE)
            if n % 25 == 0:
                print(f"  call {n}: logZ_live-logZ="
                      f"{float(live.logZ_live-live.logZ):9.2f}  "
                      f"{time.perf_counter()-t0:6.0f}s", flush=True)
    run_t = time.perf_counter() - t0
    print(f"\nrun time: {run_t:.1f} s ({n} calls)", flush=True)

    dead = blackjax.ns.utils.finalise(live, dead)
    from anesthetic import NestedSamples
    s = NestedSamples({kk: np.asarray(v) for kk, v in dead.particles.items()},
                      logL=np.asarray(dead.loglikelihood),
                      logL_birth=np.asarray(dead.loglikelihood_birth))
    lz = s.logZ(1000)
    ess = int(blackjax.ns.utils.ess(jax.random.PRNGKey(1), dead))

    np.savez(OUT, logL=np.asarray(dead.loglikelihood),
             logL_birth=np.asarray(dead.loglikelihood_birth),
             logZ=float(lz.mean()), logZ_err=float(lz.std()),
             num_live=NUM_LIVE, num_delete=NUM_DELETE,
             num_inner_steps=NUM_INNER_STEPS, run_time=run_t,
             sample_OmL=bool(SAMPLE_OML), stigma_prior=STIGMA_PRIOR,
             **{kk: np.asarray(v) for kk, v in dead.particles.items()})

    print(f"ln Z = {lz.mean():.4f} +- {lz.std():.4f}   D_KL = {s.D_KL():.2f}")
    print(f"ESS  ~ {ess}")
    print("\nposterior means +- std:")
    for p in PRIOR_BOUNDS:
        print(f"  {p:8s} = {s[p].mean():11.4f} +- {s[p].std():.4f}")
    print(f"\nsaved -> {OUT}")
    print("RUN1000LAM DONE")
