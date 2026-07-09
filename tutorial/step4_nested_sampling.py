"""
STEP 4 — Nested sampling with BlackJAX NSS.
===========================================

Run me with:
    source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
    cd /Users/starfish/Desktop/PhD2025-26/distances/TorC
    python jax_tutorial/step4_nested_sampling.py

Writes the dead points to chains/torc_ns.npz (loaded by step 5 for the
corner plot and constraints).

-----------------------------------------------------------------------------
WHAT NESTED SAMPLING DOES
-----------------------------------------------------------------------------
We want two things from Bayes' theorem:

    posterior(theta) = L(theta) * prior(theta) / Z,      Z = INT L dprior

  * the POSTERIOR tells us what the data say about (Om, H0rd, stigma, ...)
  * the EVIDENCE Z lets us compare whole models (Stigma vs LCDM).

Nested sampling computes Z (and gets the posterior for free) by a simple
scheme: hold `num_live` points drawn from the prior; repeatedly delete the
worst (lowest-L) ones and replace them with new prior draws constrained to
have HIGHER likelihood. Each deletion shrinks the enclosed prior volume by
a known factor (~ 1/num_live), so the sequence of dead points traces out
how likelihood builds up as volume shrinks: Z = sum(L_dead * dVolume).
Posterior samples are the dead points re-weighted by L*dVolume.

The hard part is "draw from the prior above a likelihood floor". NSS does
it with slice sampling: from a surviving live point, take `num_inner_steps`
randomized slice steps that stay above the floor — enough steps and the
new point is an independent draw. That inner kernel is where jit/vmap pay:
blackjax replaces `num_delete` points simultaneously as one batched
computation.

-----------------------------------------------------------------------------
THE PRIOR IS PART OF THE MODEL
-----------------------------------------------------------------------------
Two functions below MUST agree: logprior_fn (the density we claim) and
sample_prior (the draws we initialise from). For uniform boxes both are
trivial, but if they ever disagree the evidence is silently wrong — keep
them next to each other.

Prior ranges (h0rd, omega_m follow distances/desidr2.py):
    omega_m ~ U(0.01, 0.99)
    h0rd    ~ U(3650, 18250) km/s
    stigma  ~ U(1.0001, 3)
    mups2   ~ U(0.5, 3)
    OmL     ~ U(0.1, 1.5)

-----------------------------------------------------------------------------
KNOBS (see blackjax-nss guidance)
-----------------------------------------------------------------------------
  num_live        : posterior/evidence resolution. 500 = exploration,
                    1000-2000 = publication.
  num_delete      : how many deaths per step, batched together. ~10% of
                    num_live on CPU.
  num_inner_steps : slice-sampling steps per replacement = reliability.
                    5 * ndims is the standard start; halve/double it and
                    check the posterior doesn't move.
  stop            : logZ_live - logZ < -3, i.e. the live points can at
                    most add e^-3 ~ 5% to the evidence.
-----------------------------------------------------------------------------
"""

import sys, os, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import blackjax
import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from step3_desi_likelihood import loglikelihood     # the whole of step 3

# ---------------------------------------------------------------------------
# 1. Prior: density and sampler, side by side
# ---------------------------------------------------------------------------
PRIOR_BOUNDS = {
    "omega_m": (0.01,   0.99),
    "h0rd":    (3650.0, 18250.0),
    "stigma":  (1.0001, 3.0),
    "mups2":   (0.5,    3.0),
    "OmL":     (0.1,    1.5),
}

def logprior_fn(params):
    lp = 0.0
    for name, (lo, hi) in PRIOR_BOUNDS.items():
        # note jax.scipy convention: uniform.logpdf(x, loc, scale=width)
        lp += jax.scipy.stats.uniform.logpdf(params[name], lo, hi - lo)
    return lp

def sample_prior(rng_key, n):
    keys = jax.random.split(rng_key, len(PRIOR_BOUNDS))
    return {name: jax.random.uniform(k, (n,), minval=lo, maxval=hi)
            for k, (name, (lo, hi)) in zip(keys, PRIOR_BOUNDS.items())}

# ---------------------------------------------------------------------------
# 2. The sampler
# ---------------------------------------------------------------------------
NUM_LIVE        = 250                   # exploration run; 1000+ for publication
NUM_DELETE      = 50                    # 20% of num_live, batched together
NUM_DIMS        = len(PRIOR_BOUNDS)
NUM_INNER_STEPS = NUM_DIMS * 5          # 25

# The slice sampler proposes points outside the prior box; the prior rejects
# them, but an out-of-bounds ODE solve burns the full max_steps budget and --
# because vmapped lanes run in lockstep -- stalls its entire batch (~20x
# slowdown measured). Clamp such points to a safe fiducial before the solve
# and floor their logL directly: results are identical, batches stay fast.
_SAFE = dict(omega_m=0.3, h0rd=10000.0, stigma=1.5, mups2=1.5, OmL=0.7)

def loglikelihood_bounded(params):
    inb = jnp.bool_(True)
    for name, (lo, hi) in PRIOR_BOUNDS.items():
        inb = inb & (params[name] >= lo) & (params[name] <= hi)
    safe = {name: jnp.where(inb, params[name], _SAFE[name])
            for name in PRIOR_BOUNDS}
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

    print("initialising live points (compiles the likelihood)...")
    t0 = time.perf_counter()
    live = jax.jit(algo.init)(particles)
    print(f"  init done in {time.perf_counter()-t0:.1f} s")

    step_fn = jax.jit(algo.step)

    # --- the nested sampling loop ------------------------------------------
    # Each iteration kills the NUM_DELETE worst live points (they become
    # "dead points" carrying the evidence integral) and replaces them by
    # slice sampling above the death contour.
    dead = []
    t0 = time.perf_counter()
    with tqdm.tqdm(desc="dead points", unit=" pts") as pbar:
        while not live.logZ_live - live.logZ < -3:
            rng_key, subkey = jax.random.split(rng_key)
            live, dead_info = step_fn(subkey, live)
            dead.append(dead_info)
            pbar.update(NUM_DELETE)
    print(f"run time: {time.perf_counter()-t0:.1f} s")

    # --- finalise: dead + remaining live points = the full run --------------
    dead = blackjax.ns.utils.finalise(live, dead)

    logZ = float(live.logZ)
    kidx = jax.random.PRNGKey(1)
    ess  = int(blackjax.ns.utils.ess(kidx, dead))
    print(f"\nlog Z = {logZ:.3f}")
    print(f"effective posterior sample size ~ {ess}")

    # --- save for step 5 -----------------------------------------------------
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "chains"), exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), "..", "chains", "torc_ns.npz")
    np.savez(
        out,
        logL=np.asarray(dead.loglikelihood),
        logL_birth=np.asarray(dead.loglikelihood_birth),
        logZ=logZ,
        **{k: np.asarray(v) for k, v in dead.particles.items()},
    )
    print(f"saved dead points -> {out}")

    # --- quick posterior summary (proper plots in step 5) --------------------
    from anesthetic import NestedSamples
    samples = NestedSamples(
        {k: np.asarray(v) for k, v in dead.particles.items()},
        logL=np.asarray(dead.loglikelihood),
        logL_birth=np.asarray(dead.loglikelihood_birth),
    )
    print("\nposterior means +- std:")
    for name in PRIOR_BOUNDS:
        print(f"  {name:8s} = {samples[name].mean():10.4f} +- {samples[name].std():.4f}")
