import sys, os, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import blackjax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.priors import PRIOR_BOUNDS, logprior_fn, sample_prior, loglikelihood_bounded

NUM_LIVE = 250
NUM_DELETE = 25   # 10% of num_live, per skill guidance + validated in the previous test


def run_to_convergence(num_inner_steps, seed=0, label=""):
    print(f"\n=== NUM_INNER_STEPS={num_inner_steps}  ({label}) ===", flush=True)
    rng_key = jax.random.PRNGKey(seed)
    algo = blackjax.nss(logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_bounded,
                         num_delete=NUM_DELETE, num_inner_steps=num_inner_steps)

    rng_key, init_key = jax.random.split(rng_key)
    particles = sample_prior(init_key, NUM_LIVE)

    t0 = time.perf_counter()
    live = jax.jit(algo.init)(particles)
    jax.block_until_ready(live)
    print(f"  init: {time.perf_counter()-t0:.1f}s", flush=True)

    step_fn = jax.jit(algo.step)
    dead = []
    n_calls = 0
    t0 = time.perf_counter()
    while not (live.logZ_live - live.logZ < -3):
        rng_key, subkey = jax.random.split(rng_key)
        live, dead_info = step_fn(subkey, live)
        dead.append(dead_info)
        n_calls += 1
        if n_calls % 5 == 0:
            print(f"    call {n_calls}: logZ_live-logZ={float(live.logZ_live-live.logZ):.2f}  "
                  f"elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    run_time = time.perf_counter() - t0
    print(f"  run done: {n_calls} step_fn calls, {run_time:.1f}s total "
          f"({run_time/n_calls*1e3:.1f} ms/call, {run_time/(n_calls*NUM_DELETE)*1e3:.2f} ms/dead-point)",
          flush=True)

    dead = blackjax.ns.utils.finalise(live, dead)
    logZ = float(live.logZ)

    from anesthetic import NestedSamples
    samples = NestedSamples(
        {k: np.asarray(v) for k, v in dead.particles.items()},
        logL=np.asarray(dead.loglikelihood),
        logL_birth=np.asarray(dead.loglikelihood_birth))

    means = {name: float(samples[name].mean()) for name in PRIOR_BOUNDS}
    stds = {name: float(samples[name].std()) for name in PRIOR_BOUNDS}

    print(f"  logZ = {logZ:.3f}")
    for name in PRIOR_BOUNDS:
        print(f"    {name:8s} = {means[name]:10.4f} +- {stds[name]:.4f}")

    return dict(n_calls=n_calls, run_time=run_time, logZ=logZ, means=means, stds=stds)


results = {}
results[25] = run_to_convergence(25, label="baseline, current run_ns.py setting")
results[13] = run_to_convergence(13, label="halved")

print("\n" + "=" * 72)
print("COMPARISON")
print("=" * 72)
r25, r13 = results[25], results[13]
print(f"NUM_INNER_STEPS=25: {r25['n_calls']} calls, {r25['run_time']:.1f}s total, logZ={r25['logZ']:.3f}")
print(f"NUM_INNER_STEPS=13: {r13['n_calls']} calls, {r13['run_time']:.1f}s total, logZ={r13['logZ']:.3f}")
print(f"speedup (total wall time): {r25['run_time']/r13['run_time']:.2f}x")
print(f"logZ difference: {r13['logZ']-r25['logZ']:+.3f}  (nats)")
print("\nposterior mean shift, in units of the NUM_INNER_STEPS=25 std dev:")
for name in PRIOR_BOUNDS:
    shift = (r13["means"][name] - r25["means"][name]) / r25["stds"][name]
    std_ratio = r13["stds"][name] / r25["stds"][name]
    print(f"  {name:8s}  mean shift = {shift:+.3f} sigma   std ratio (13/25) = {std_ratio:.3f}")
