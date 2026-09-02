"""Run the EXACT committed run_ns.py configuration at one seed.

Usage: python seed_scatter.py <seed>

Committed settings (git show HEAD:scripts/run_ns.py):
    NUM_LIVE=250, NUM_DELETE=50, NUM_INNER_STEPS=len(PRIOR_BOUNDS)*5=25
Measures ln Z so we can compare run-to-run scatter against the README's
reference value of -7.86 +- 0.21.
"""
import sys, os, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import blackjax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.priors import PRIOR_BOUNDS, logprior_fn, sample_prior, loglikelihood_bounded

SEED = int(sys.argv[1])
NUM_LIVE = 250
NUM_DELETE = 50
NUM_INNER_STEPS = len(PRIOR_BOUNDS) * 5

rng_key = jax.random.PRNGKey(SEED)
algo = blackjax.nss(logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_bounded,
                     num_delete=NUM_DELETE, num_inner_steps=NUM_INNER_STEPS)
rng_key, init_key = jax.random.split(rng_key)
live = jax.jit(algo.init)(sample_prior(init_key, NUM_LIVE))
step_fn = jax.jit(algo.step)

dead = []
n_calls = 0
t0 = time.perf_counter()
while not (live.logZ_live - live.logZ < -3):
    rng_key, subkey = jax.random.split(rng_key)
    live, dinfo = step_fn(subkey, live)
    dead.append(dinfo)
    n_calls += 1
    if n_calls % 25 == 0:
        print(f"  seed{SEED} call {n_calls}: {float(live.logZ_live-live.logZ):.2f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
run_t = time.perf_counter() - t0

dead = blackjax.ns.utils.finalise(live, dead)
from anesthetic import NestedSamples
s = NestedSamples({k: np.asarray(v) for k, v in dead.particles.items()},
                  logL=np.asarray(dead.loglikelihood),
                  logL_birth=np.asarray(dead.loglikelihood_birth))
lz = s.logZ(1000)
print(f"RESULT seed={SEED} lnZ={lz.mean():.4f} pm {lz.std():.4f} "
      f"D_KL={s.D_KL():.2f} calls={n_calls} time={run_t:.0f}s", flush=True)
