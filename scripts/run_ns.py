"""Nested sampling of the TorC model (noH0 anchoring) against DESI DR2 BAO.

Usage:  python scripts/run_ns.py            (from the repo root)
Writes chains/torc_ns_noH0.npz and prints logZ / ESS / posterior summary.

Knobs (see the blackjax-nss guidance): NUM_LIVE 250 = exploration,
1000+ = publication.  mups2 is an exact background gauge parameter and can
be fixed without loss for BAO-only runs (drop it from PRIOR_BOUNDS in
torc/priors.py and from the theory call to sample 4 parameters).
"""
import sys, os, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import blackjax
import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.priors import (PRIOR_BOUNDS, logprior_fn, sample_prior,
                         loglikelihood_bounded)

NUM_LIVE        = 250
NUM_DELETE      = 50
NUM_INNER_STEPS = len(PRIOR_BOUNDS) * 5

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
    print(f"\nlog Z = {logZ:.3f}")
    print(f"effective posterior sample size ~ {ess}")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "chains")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "torc_ns_noH0.npz")
    np.savez(out,
             logL=np.asarray(dead.loglikelihood),
             logL_birth=np.asarray(dead.loglikelihood_birth),
             logZ=logZ,
             **{k: np.asarray(v) for k, v in dead.particles.items()})
    print(f"saved dead points -> {out}", flush=True)

    from anesthetic import NestedSamples
    samples = NestedSamples(
        {k: np.asarray(v) for k, v in dead.particles.items()},
        logL=np.asarray(dead.loglikelihood),
        logL_birth=np.asarray(dead.loglikelihood_birth))
    print("\nposterior means +- std:")
    for name in PRIOR_BOUNDS:
        print(f"  {name:8s} = {samples[name].mean():10.4f} "
              f"+- {samples[name].std():.4f}")
