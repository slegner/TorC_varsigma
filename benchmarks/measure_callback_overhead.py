"""Could we swap ONLY the ODE solve out to Julia, keeping the rest in JAX?

That needs jax.pure_callback inside jit+vmap+while_loop.  The decisive
question is what a host round-trip costs there, because the sampler makes
~216,000 solves per run in batches of ~25.

Budget: Julia would do 25 solves in 25 x 0.17 = ~4.3 ms.  If the callback
overhead is comparable to or larger than that, the 12x solver advantage is
eaten and the idea is dead.
"""
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

N = 25          # num_delete: the batch handed to one callback
NOUT = 500      # E(z) values returned per particle


def host_fn(params):
    """Stand-in for 'call Julia and get E(z) back'.  Does no real work, so
    whatever this costs is PURE overhead."""
    p = np.asarray(params)
    return np.ones((p.shape[0], NOUT), dtype=np.float64)


def via_callback(params):
    return jax.pure_callback(
        host_fn,
        jax.ShapeDtypeStruct((params.shape[0], NOUT), jnp.float64),
        params, vmap_method="expand_dims")


def timeit(f, *a, reps=20):
    jax.block_until_ready(f(*a))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*a))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


print("=" * 68)
print("1. bare pure_callback, batch of 25, no work done on the host")
print("=" * 68)
p = jnp.ones((N, 5))
f1 = jax.jit(via_callback)
ms1 = timeit(f1, p)
print(f"   one callback round-trip : {ms1:8.3f} ms")
print(f"   Julia would need        :    4.250 ms for 25 solves")
print(f"   -> overhead is {ms1/4.25:.2f}x the useful work")

print()
print("=" * 68)
print("2. the REAL structure: callback inside lax.while_loop inside vmap")
print("=" * 68)


def slice_like(params, n_iter):
    """Mimic the slice sampler: a while_loop whose body needs a likelihood
    (hence the ODE, hence the callback) on every iteration."""
    def body(carry):
        i, acc = carry
        E = via_callback(params)
        return i + 1, acc + E[:, -1].sum()

    def cond(carry):
        return carry[0] < n_iter

    return jax.lax.while_loop(cond, body, (0, 0.0))


for n_iter in (1, 5, 10):
    f2 = jax.jit(lambda p, n=n_iter: slice_like(p, n))
    ms2 = timeit(f2, p, reps=10)
    print(f"   {n_iter:2d} while-loop iterations : {ms2:8.3f} ms "
          f"({ms2/n_iter:6.3f} ms per callback)")

print()
print("=" * 68)
print("3. what this implies for a full run")
print("=" * 68)
per_cb = ms1
n_callbacks = 216_000 / N
print(f"   ~{n_callbacks:.0f} callbacks per run (216k solves / batch of {N})")
print(f"   overhead alone : {n_callbacks*per_cb/1000:8.1f} s")
print(f"   + Julia solves : {216_000*0.17/1000:8.1f} s")
print(f"   = total        : {(n_callbacks*per_cb + 216_000*0.17)/1000:8.1f} s")
print(f"   current diffrax pipeline (measured) :   435.5 s")
print()
print("   NOTE: this ignores that pure_callback also prevents XLA from")
print("   fusing step_fn, which costs more on top.")
print("DONE")
