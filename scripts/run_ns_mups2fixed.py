"""TorC nested sampling with mups2 FIXED to 4/3 (exact background gauge
parameter -- see torc/theory.py docstring; the likelihood is provably flat
along it, so fixing it must reproduce the 5-parameter run's evidence and
posteriors while sampling one dimension fewer).

Usage:  python scripts/run_ns_mups2fixed.py     (from the repo root)
Writes chains/torc_ns_noH0_mups2fixed.npz.
"""
import sys, os, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import blackjax
import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.likelihood import loglikelihood

MUPS2_FIXED = 4.0 / 3.0

BOUNDS = {
    "omega_m": (0.01,   0.99),
    "h0rd":    (3650.0, 18250.0),
    "stigma":  (1.0001, 3.0),
    "OmL":     (0.1,    1.5),
}
_SAFE = dict(omega_m=0.3, h0rd=10000.0, stigma=1.5, OmL=0.7)

NUM_LIVE        = 250
NUM_DELETE      = 50
NUM_INNER_STEPS = len(BOUNDS) * 5      # 20


def logprior_fn(p):
    return sum(jax.scipy.stats.uniform.logpdf(p[n], lo, hi - lo)
               for n, (lo, hi) in BOUNDS.items())


def sample_prior(key, num):
    keys = jax.random.split(key, len(BOUNDS))
    return {name: jax.random.uniform(k, (num,), minval=lo, maxval=hi)
            for k, (name, (lo, hi)) in zip(keys, BOUNDS.items())}


def loglikelihood_bounded(params):
    """4-parameter clamp; injects the fixed mups2 before the solve."""
    inb = jnp.bool_(True)
    for name, (lo, hi) in BOUNDS.items():
        inb = inb & (params[name] >= lo) & (params[name] <= hi)
    safe = {name: jnp.where(inb, params[name], _SAFE[name]) for name in BOUNDS}
    safe = dict(safe, mups2=MUPS2_FIXED)
    return jnp.where(inb, loglikelihood(safe), -1e30)


if __name__ == "__main__":
    rng_key = jax.random.PRNGKey(0)

    algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_bounded,
        num_delete=NUM_DELETE,
        num_inner_steps=NUM_INNER_STEPS,
    )

    rng_key, init_key = jax.random.split(rng_key)
    particles = sample_prior(init_key, NUM_LIVE)

    print("initialising live points (compiles the likelihood)...", flush=True)
    t0 = time.perf_counter()
    live = jax.jit(algo.init)(particles)
    print(f"  init done in {time.perf_counter()-t0:.1f} s", flush=True)

    step_fn = jax.jit(algo.step)
    dead = []
    t0 = time.perf_counter()
    with tqdm.tqdm(desc="dead points", unit=" pts") as pbar:
        while not live.logZ_live - live.logZ < -3:
            rng_key, subkey = jax.random.split(rng_key)
            live, dead_info = step_fn(subkey, live)
            dead.append(dead_info)
            pbar.update(NUM_DELETE)
    print(f"run time: {time.perf_counter()-t0:.1f} s", flush=True)

    dead = blackjax.ns.utils.finalise(live, dead)
    logZ = float(live.logZ)
    ess  = int(blackjax.ns.utils.ess(jax.random.PRNGKey(1), dead))
    print(f"\nlog Z = {logZ:.3f}   (5-param run gave -8.097; should agree)")
    print(f"effective posterior sample size ~ {ess}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "chains")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "torc_ns_noH0_mups2fixed.npz")
    np.savez(out,
             logL=np.asarray(dead.loglikelihood),
             logL_birth=np.asarray(dead.loglikelihood_birth),
             logZ=logZ,
             mups2_fixed=MUPS2_FIXED,
             **{k: np.asarray(v) for k, v in dead.particles.items()})
    print(f"saved dead points -> {out}", flush=True)

    from anesthetic import NestedSamples
    samples = NestedSamples(
        {k: np.asarray(v) for k, v in dead.particles.items()},
        logL=np.asarray(dead.loglikelihood),
        logL_birth=np.asarray(dead.loglikelihood_birth))
    print(f"\nlogZ (anesthetic) = {samples.logZ():.3f},  "
          f"D_KL = {samples.D_KL():.3f}")
    print("posterior means +- std:")
    for name in BOUNDS:
        print(f"  {name:8s} = {samples[name].mean():10.4f} "
              f"+- {samples[name].std():.4f}")
