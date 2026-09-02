"""mups2-fixed (4 params) at one seed, current code.

For an EXACT gauge direction the evidence factorises:
    Z_5d = Z_4d x \int prior(mups2) dmups2 = Z_4d
so lnZ must match the 5-parameter runs (-7.880 +- 0.053). Any systematic
offset would mean mups2 is not a pure gauge parameter after all.
"""
import sys, os, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import blackjax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.likelihood import loglikelihood
from torc.priors import _SAFE

SEED = int(sys.argv[1])
MUPS2_FIXED = 4.0 / 3.0
BOUNDS = {"omega_m": (0.01, 0.99), "h0rd": (3650.0, 18250.0),
          "stigma": (1.0001, 3.0), "OmL": (0.1, 1.5)}
NUM_LIVE, NUM_DELETE = 250, 50
NUM_INNER_STEPS = len(BOUNDS) * 5          # 20, vs 25 for 5 params


def logprior_fn(p):
    return sum(jax.scipy.stats.uniform.logpdf(p[n], lo, hi - lo)
               for n, (lo, hi) in BOUNDS.items())


def sample_prior(key, num):
    keys = jax.random.split(key, len(BOUNDS))
    return {name: jax.random.uniform(k, (num,), minval=lo, maxval=hi)
            for k, (name, (lo, hi)) in zip(keys, BOUNDS.items())}


def loglikelihood_bounded(params):
    inb = jnp.bool_(True)
    for name, (lo, hi) in BOUNDS.items():
        inb = inb & (params[name] >= lo) & (params[name] <= hi)
    safe = {name: jnp.where(inb, params[name], _SAFE[name]) for name in BOUNDS}
    safe = dict(safe, mups2=MUPS2_FIXED)
    return jnp.where(inb, loglikelihood(safe), -1e30)


rng = jax.random.PRNGKey(SEED)
algo = blackjax.nss(logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_bounded,
                     num_delete=NUM_DELETE, num_inner_steps=NUM_INNER_STEPS)
rng, k = jax.random.split(rng)
live = jax.jit(algo.init)(sample_prior(k, NUM_LIVE))
step = jax.jit(algo.step)

dead = []
n = 0
t0 = time.perf_counter()
while not (live.logZ_live - live.logZ < -3):
    rng, k = jax.random.split(rng)
    live, d = step(k, live)
    dead.append(d)
    n += 1
    if n % 25 == 0:
        print(f"  seed{SEED} call {n}: {float(live.logZ_live-live.logZ):.2f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
rt = time.perf_counter() - t0

dead = blackjax.ns.utils.finalise(live, dead)
from anesthetic import NestedSamples
s = NestedSamples({k2: np.asarray(v) for k2, v in dead.particles.items()},
                  logL=np.asarray(dead.loglikelihood),
                  logL_birth=np.asarray(dead.loglikelihood_birth))
lz = s.logZ(1000)
print(f"RESULT seed={SEED} lnZ={lz.mean():.4f} pm {lz.std():.4f} "
      f"D_KL={s.D_KL():.2f} calls={n} time={rt:.0f}s", flush=True)
