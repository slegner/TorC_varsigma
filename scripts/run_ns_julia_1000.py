"""Production nested-sampling run: 1000 live points, Julia IPC backend.

NUM_LIVE=1000 is the publication setting (the README's 250 is exploration).
It roughly halves the ln Z uncertainty relative to 250:  sigma ~ sqrt(D_KL/N),
so ~0.20 -> ~0.09.

Backend: Julia Tsit5 in a separate process over a unix socket
(torc/julia_ipc.py), validated at 3.0x faster than diffrax with the posterior
unchanged (all marginals within 0.06 sigma, ln Z within 0.15 sigma).

Writes chains/torc_ns_julia_1000.npz -- a NEW filename, deliberately: it does
not clobber chains/torc_ns_noH0.npz the way run_ns.py does.

Run:
    caffeinate -ims python scripts/run_ns_julia_1000.py
(caffeinate stops idle sleep; note it does NOT survive closing the lid.)
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

from torc.priors import (PRIOR_BOUNDS, logprior_fn, sample_prior,
                         loglikelihood_bounded)
import torc.julia_ipc as ji

NUM_LIVE = 1000
NUM_DELETE = NUM_LIVE // 10          # 10% -- the blackjax-nss CPU guidance
NUM_INNER_STEPS = len(PRIOR_BOUNDS) * 5
SEED = 0
from torc.priors import STIGMA_PRIOR, STIGMA_LAMBDA
# separate filename per prior, so the uniform-prior chain survives
# tag by prior AND width, so lambda variants do not clobber each other
# lambda is only meaningful for the exponential; the power law is parameter-free
if STIGMA_PRIOR == "uniform":
    _TAG = ""
elif STIGMA_PRIOR == "powerlaw":
    _TAG = "_powerlaw"
else:
    _TAG = f"_{STIGMA_PRIOR}_lam{STIGMA_LAMBDA:g}".replace(".", "p")
# optional extra tag, e.g. TORC_TAG=_nostigmawall, so a re-run under changed
# prior bounds cannot overwrite an earlier chain made under the old ones
_SUFFIX = os.environ.get("TORC_TAG", "")
OUT = os.path.join(HERE, "..", "chains",
                   f"torc_ns_julia_1000{_TAG}{_SUFFIX}.npz")

if __name__ == "__main__":
    print(f"ndim={len(PRIOR_BOUNDS)} ({', '.join(PRIOR_BOUNDS)})")
    print(f"stigma prior: {STIGMA_PRIOR}  lambda={STIGMA_LAMBDA}")
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
             **{kk: np.asarray(v) for kk, v in dead.particles.items()})

    print(f"ln Z = {lz.mean():.4f} +- {lz.std():.4f}   D_KL = {s.D_KL():.2f}")
    print(f"ESS  ~ {ess}")
    print("\nposterior means +- std:")
    for p in PRIOR_BOUNDS:
        print(f"  {p:8s} = {s[p].mean():11.4f} +- {s[p].std():.4f}")
    print(f"\nsaved -> {OUT}")
    print("\n(250-live reference: ln Z = -8.14 +- 0.20; expect ~half that error)")
    print("RUN1000 DONE")
