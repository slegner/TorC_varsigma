"""Cost cascade for the LCDM fit, measured the same way as the TorC one."""
import sys, os, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import blackjax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.abspath("."), "scripts"))
from run_lcdm import loglikelihood, logprior_fn, sample_prior, BOUNDS

NUM_LIVE = 500
NUM_DELETE = 100
NUM_INNER_STEPS = 2 * 5

# ---- 1. single likelihood call cost (jit, no vmap) ----
f = jax.jit(loglikelihood)
p = {"omega_m": jnp.asarray(0.3), "h0rd": jnp.asarray(10000.0)}
jax.block_until_ready(f(p))
ts = []
for _ in range(200):
    t0 = time.perf_counter()
    jax.block_until_ready(f(p))
    ts.append(time.perf_counter() - t0)
single_ms = np.median(ts) * 1e3
print(f"single loglikelihood call (jit, no vmap): {single_ms:.4f} ms", flush=True)

# ---- 2. batched rate, for the batching-efficiency comparison ----
fb = jax.jit(jax.vmap(loglikelihood))
pb = {"omega_m": jnp.linspace(0.1, 0.9, 250), "h0rd": jnp.linspace(4000., 18000., 250)}
jax.block_until_ready(fb(pb))
ts = []
for _ in range(50):
    t0 = time.perf_counter()
    jax.block_until_ready(fb(pb))
    ts.append(time.perf_counter() - t0)
batch_ms = np.median(ts) * 1e3 / 250
print(f"batched (vmap 250) per-call:              {batch_ms:.4f} ms  "
      f"({single_ms/batch_ms:.1f}x faster than single)", flush=True)

# ---- 3. full NS run ----
algo = blackjax.nss(logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood,
                    num_delete=NUM_DELETE, num_inner_steps=NUM_INNER_STEPS)
rng = jax.random.PRNGKey(0)
rng, k = jax.random.split(rng)
live = jax.jit(algo.init)(sample_prior(k, NUM_LIVE))
step = jax.jit(algo.step)

rng, k = jax.random.split(rng)
t0 = time.perf_counter()
live, d = step(k, live)
jax.block_until_ready(live)
print(f"first step (incl. compile): {time.perf_counter()-t0:.2f}s", flush=True)

dead = [d]
n_calls = 1
nstep_all, nshrink_all = [], []
t0 = time.perf_counter()
while not (live.logZ_live - live.logZ < -3):
    rng, k = jax.random.split(rng)
    live, d = step(k, live)
    dead.append(d)
    n_calls += 1
    ik = d.inner_kernel_info.info
    nstep_all.append(np.asarray(ik.num_steps).mean())
    nshrink_all.append(np.asarray(ik.num_shrink).mean())
run_t = time.perf_counter() - t0
print(f"run: {n_calls} step_fn calls, {run_t:.2f}s  ({run_t/n_calls*1e3:.1f} ms/call)", flush=True)

dead_f = blackjax.ns.utils.finalise(live, dead)
from anesthetic import NestedSamples
s = NestedSamples({n: np.asarray(v) for n, v in dead_f.particles.items()},
                  logL=np.asarray(dead_f.loglikelihood),
                  logL_birth=np.asarray(dead_f.loglikelihood_birth))
lz = s.logZ(1000)
D_KL = s.D_KL()

mean_nstep = np.mean(nstep_all)
mean_nshrink = np.mean(nshrink_all)
calls_per_slice = mean_nstep + mean_nshrink
n_dead = n_calls * NUM_DELETE
solves_per_call = NUM_INNER_STEPS * NUM_DELETE * calls_per_slice
total_solves = n_calls * solves_per_call
eff_ms = run_t * 1e3 / total_solves

print("\n" + "=" * 60)
print("LCDM COST CASCADE")
print("=" * 60)
print(f"  params sampled              : 2 (omega_m, h0rd)")
print(f"  num_live / num_delete / inner: {NUM_LIVE} / {NUM_DELETE} / {NUM_INNER_STEPS}")
print(f"  D_KL                        : {D_KL:.2f} nats")
print(f"  ln Z                        : {lz.mean():.3f} +- {lz.std():.3f}")
print(f"  slice: num_steps={mean_nstep:.2f}, num_shrink={mean_nshrink:.2f} "
      f"-> {calls_per_slice:.2f} likelihood calls/slice step")
print(f"  1 likelihood call           : {eff_ms*1e3:.1f} us (effective, in-sampler)")
print(f"  1 slice step                : {eff_ms*calls_per_slice:.3f} ms")
print(f"  1 replacement (= 1 dead pt) : {eff_ms*calls_per_slice*NUM_INNER_STEPS:.2f} ms")
print(f"  dead points                 : {n_dead:,}")
print(f"  total likelihood calls      : {total_solves:,.0f}")
print(f"  TOTAL RUN                   : {run_t:.2f} s")
print(f"\n  standalone single-call rate : {single_ms:.4f} ms")
print(f"  in-sampler effective rate   : {eff_ms:.4f} ms  "
      f"({eff_ms/single_ms:.2f}x standalone)")
print("DONE")
