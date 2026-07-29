import sys, os, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import blackjax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.priors import PRIOR_BOUNDS, logprior_fn, sample_prior, loglikelihood_bounded

NUM_LIVE = 250
NUM_INNER_STEPS = len(PRIOR_BOUNDS) * 5   # unchanged, per skill guidance (5*ndim), independent of num_delete
N_STEPS_TO_TIME = 8

def run_config(num_delete, seed=0):
    print(f"\n=== num_delete={num_delete} ({100*num_delete/NUM_LIVE:.0f}% of num_live) ===", flush=True)
    rng_key = jax.random.PRNGKey(seed)
    algo = blackjax.nss(logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_bounded,
                         num_delete=num_delete, num_inner_steps=NUM_INNER_STEPS)

    rng_key, init_key = jax.random.split(rng_key)
    particles = sample_prior(init_key, NUM_LIVE)
    live = jax.jit(algo.init)(particles)
    step_fn = jax.jit(algo.step)

    # compile (first call, discard timing)
    rng_key, subkey = jax.random.split(rng_key)
    t0 = time.perf_counter()
    live, dead_info = step_fn(subkey, live)
    jax.block_until_ready(live)
    print(f"  first call (compile+run): {time.perf_counter()-t0:.2f}s", flush=True)

    times = []
    num_steps_all, num_shrink_all = [], []
    for i in range(N_STEPS_TO_TIME):
        rng_key, subkey = jax.random.split(rng_key)
        t0 = time.perf_counter()
        live, dead_info = step_fn(subkey, live)
        jax.block_until_ready(live)
        dt = time.perf_counter() - t0
        times.append(dt)
        ss = np.asarray(dead_info.inner_kernel_info.info.num_steps)
        sh = np.asarray(dead_info.inner_kernel_info.info.num_shrink)
        num_steps_all.append(ss.mean())
        num_shrink_all.append(sh.mean())
        print(f"  step {i}: {dt*1e3:7.1f} ms  ({dt/num_delete*1e3:.2f} ms/dead-point)  "
              f"[mean num_steps={ss.mean():.2f}, num_shrink={sh.mean():.2f}]  "
              f"logZ_live-logZ={float(live.logZ_live - live.logZ):.2f}", flush=True)

    times = np.array(times)
    print(f"  -> median: {np.median(times)*1e3:.1f} ms/call, "
          f"{np.median(times)/num_delete*1e3:.3f} ms/dead-point  "
          f"(mean num_steps={np.mean(num_steps_all):.2f}, num_shrink={np.mean(num_shrink_all):.2f})",
          flush=True)
    return dict(times=times, per_dead=times/num_delete,
                num_steps=np.mean(num_steps_all), num_shrink=np.mean(num_shrink_all))


results = {}
for nd in (50, 25, 10):
    results[nd] = run_config(nd)

print("\n" + "=" * 70)
print("SUMMARY (median over {} step_fn calls after compile, same seed=0 init)".format(N_STEPS_TO_TIME))
print("=" * 70)
base = None
for nd in (50, 25, 10):
    r = results[nd]
    per_dead_ms = np.median(r["per_dead"]) * 1e3
    if base is None:
        base = per_dead_ms
    print(f"num_delete={nd:3d} ({100*nd/NUM_LIVE:2.0f}% of num_live): "
          f"{np.median(r['times'])*1e3:8.1f} ms/call, {per_dead_ms:7.3f} ms/dead-point "
          f"({per_dead_ms/base:.2f}x num_delete=50's per-dead-point cost)  "
          f"[num_steps={r['num_steps']:.2f}, num_shrink={r['num_shrink']:.2f}]")
