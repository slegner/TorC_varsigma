"""Can we skip the safe-ODE for out-of-prior points?

Under vmap: no (lax.cond -> select, both branches run). Established.

But vmap may be buying nothing here anyway (in-sampler 2.02 ms/solve vs
~0.34 ms for a clean homogeneous batch -- heterogeneous lockstep eats it).
If so, a SEQUENTIAL map costs little and DOES allow lax.cond to branch,
which would let out-of-bounds points skip the ODE entirely.

Compare, on a realistic heterogeneous batch with a given OOB fraction:
  1. vmap + current bounded likelihood        (status quo)
  2. vmap + lax.cond                          (expected: no better)
  3. lax.map + lax.cond  (skip ODE when OOB)  (the candidate)
"""
import sys, os, time, functools
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.likelihood import loglikelihood
from torc.priors import PRIOR_BOUNDS, _SAFE, sample_prior

NAMES = list(PRIOR_BOUNDS)


def in_bounds(p):
    inb = jnp.bool_(True)
    for n, (lo, hi) in PRIOR_BOUNDS.items():
        inb = inb & (p[n] >= lo) & (p[n] <= hi)
    return inb


def bounded_where(p):                      # status quo
    inb = in_bounds(p)
    safe = {n: jnp.where(inb, p[n], _SAFE[n]) for n in NAMES}
    return jnp.where(inb, loglikelihood(safe), -1e30)


def bounded_cond(p):                       # real branch when not vmapped
    inb = in_bounds(p)
    safe = {n: jnp.where(inb, p[n], _SAFE[n]) for n in NAMES}
    return jax.lax.cond(inb, lambda q: loglikelihood(q),
                        lambda q: jnp.asarray(-1e30), safe)


def make_batch(n, oob_frac, seed=0):
    """Realistic heterogeneous live points; a fraction pushed out of bounds
    along stigma (into the toxic band, as a real slice step would)."""
    p = sample_prior(jax.random.PRNGKey(seed), n)
    p = {k: np.array(v, copy=True) for k, v in p.items()}
    k = int(n * oob_frac)
    if k:
        p["stigma"][:k] = 0.8
    return {k2: jnp.asarray(v) for k2, v in p.items()}


def timed(f, p, reps=6):
    jax.block_until_ready(f(p))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); jax.block_until_ready(f(p))
        ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1e3


N = 25            # num_delete in the sampler
f_vmap_where = jax.jit(jax.vmap(bounded_where))
f_vmap_cond  = jax.jit(jax.vmap(bounded_cond))
f_map_cond   = jax.jit(lambda p: jax.lax.map(bounded_cond, p))

print(f"batch of {N} heterogeneous points (prior draws), varying OOB fraction")
print(f"{'OOB frac':>9s} {'vmap+where':>12s} {'vmap+cond':>11s} {'lax.map+cond':>14s} "
      f"{'best vs status quo':>19s}")
for frac in (0.0, 0.1, 0.25, 0.5, 1.0):
    p = make_batch(N, frac)
    a = timed(f_vmap_where, p)
    b = timed(f_vmap_cond, p)
    c = timed(f_map_cond, p)
    best = min(a, b, c)
    print(f"{frac:9.2f} {a:10.2f}ms {b:9.2f}ms {c:12.2f}ms {a/best:18.2f}x")

# correctness: all three must agree
p = make_batch(N, 0.3)
va, vb, vc = [np.asarray(f(p)) for f in (f_vmap_where, f_vmap_cond, f_map_cond)]
print(f"\ncorrectness: vmap+where == vmap+cond: {np.allclose(va, vb)}, "
      f"== lax.map+cond: {np.allclose(va, vc)}")
print("DONE")
