"""What fraction of likelihood calls during real nested sampling are
out-of-prior? That sets whether skipping the safe-ODE could ever pay.

Counts via jax.debug.callback, which fires inside jit/vmap/while_loop.
Slow, but we only need a handful of iterations.
"""
import sys, os, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import blackjax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.priors import PRIOR_BOUNDS, logprior_fn, sample_prior
from torc.likelihood import loglikelihood
from torc.priors import _SAFE

NAMES = list(PRIOR_BOUNDS)
COUNT = {"tot": 0, "oob": 0}


def _tally(inb):
    a = np.asarray(inb).ravel()
    COUNT["tot"] += a.size
    COUNT["oob"] += int((~a.astype(bool)).sum())


def counting_bounded(params):
    inb = jnp.bool_(True)
    for n, (lo, hi) in PRIOR_BOUNDS.items():
        inb = inb & (params[n] >= lo) & (params[n] <= hi)
    jax.debug.callback(_tally, inb)
    safe = {n: jnp.where(inb, params[n], _SAFE[n]) for n in NAMES}
    return jnp.where(inb, loglikelihood(safe), -1e30)


NUM_LIVE, NUM_DELETE, NUM_INNER = 250, 25, 13
algo = blackjax.nss(logprior_fn=logprior_fn, loglikelihood_fn=counting_bounded,
                     num_delete=NUM_DELETE, num_inner_steps=NUM_INNER)
rng = jax.random.PRNGKey(0)
rng, k = jax.random.split(rng)
live = jax.jit(algo.init)(sample_prior(k, NUM_LIVE))
step = jax.jit(algo.step)

print("iter   cumulative OOB fraction   logZ_live-logZ", flush=True)
COUNT["tot"] = COUNT["oob"] = 0
marks = []
t0 = time.perf_counter()
for i in range(30):
    rng, k = jax.random.split(rng)
    live, _ = step(k, live)
    jax.block_until_ready(live)
    if (i + 1) % 5 == 0:
        f = COUNT["oob"] / max(COUNT["tot"], 1)
        marks.append(f)
        print(f"{i+1:4d}   {100*f:20.2f}%   {float(live.logZ_live-live.logZ):14.2f}",
              flush=True)

frac = COUNT["oob"] / max(COUNT["tot"], 1)
print(f"\ntotal likelihood calls counted : {COUNT['tot']:,}")
print(f"out-of-prior calls             : {COUNT['oob']:,}")
print(f"OOB FRACTION                   : {100*frac:.2f}%")
print(f"(elapsed {time.perf_counter()-t0:.0f}s -- inflated by the debug callback)")
print()
print("break-even for lax.map+cond was ~50% OOB.")
print("VERDICT:", "worth it" if frac > 0.5 else
      f"NOT worth it -- skipping would save at most ~{100*frac:.0f}% of calls, "
      f"but sequential mapping costs ~2x on the other {100*(1-frac):.0f}%")
print("DONE")
