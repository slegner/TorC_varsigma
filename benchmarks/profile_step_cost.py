import sys, os, time
import jax
jax.config.update("jax_enable_x64", True)
import blackjax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.priors import PRIOR_BOUNDS, logprior_fn, sample_prior, loglikelihood_bounded

NUM_LIVE = 250
NUM_DELETE = 50
NUM_INNER_STEPS = len(PRIOR_BOUNDS) * 5
print(f"NUM_LIVE={NUM_LIVE} NUM_DELETE={NUM_DELETE} NUM_INNER_STEPS={NUM_INNER_STEPS}", flush=True)

rng_key = jax.random.PRNGKey(0)
algo = blackjax.nss(logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_bounded,
                     num_delete=NUM_DELETE, num_inner_steps=NUM_INNER_STEPS)

rng_key, init_key = jax.random.split(rng_key)
particles = sample_prior(init_key, NUM_LIVE)

print("compiling+running algo.init...", flush=True)
t0 = time.perf_counter()
live = jax.jit(algo.init)(particles)
jax.block_until_ready(live)
print(f"  init: {time.perf_counter()-t0:.2f}s", flush=True)

step_fn = jax.jit(algo.step)

print("compiling step_fn (first call)...", flush=True)
rng_key, subkey = jax.random.split(rng_key)
t0 = time.perf_counter()
live, dead_info = step_fn(subkey, live)
jax.block_until_ready(live)
print(f"  first step_fn call (incl. compile): {time.perf_counter()-t0:.2f}s", flush=True)

print("timing subsequent step_fn calls (already compiled)...", flush=True)
times = []
for i in range(10):
    rng_key, subkey = jax.random.split(rng_key)
    t0 = time.perf_counter()
    live, dead_info = step_fn(subkey, live)
    jax.block_until_ready(live)
    dt = time.perf_counter() - t0
    times.append(dt)
    print(f"  step {i}: {dt*1e3:.1f} ms   (logZ_live-logZ = {float(live.logZ_live - live.logZ):.3f})",
          flush=True)

import numpy as np
times = np.array(times)
print(f"\nmedian step_fn time: {np.median(times)*1e3:.1f} ms  (std {times.std()*1e3:.1f} ms)")
print(f"per-solve estimate (naive, 1 likelihood call/inner step): "
      f"{np.median(times)*1e3 / (NUM_DELETE*NUM_INNER_STEPS):.4f} ms/solve-equivalent")
print("(compare to the ~0.44-0.61 ms/solve measured in the pure-ODE batch benchmark "
      "-- if this per-solve-equivalent number is much larger, something other than "
      "the raw ODE solve dominates step_fn's cost.)")
