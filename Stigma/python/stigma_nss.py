"""
BlackJAX Nested Slice Sampling for Stigma Q-torsion vs DESI DR2 BAO.

Converts the DESI DR2 likelihood from polychord/numpy (desidr2.py) to JAX,
using the Stigma ODE (equations_jax.py) to replace the FlexKnot w(z) model.

Parameters sampled:
  h0rd    : H0 * r_d  [km/s]         – BAO scale (prior ≈ Uniform[3000, 20000])
  omega_m : Ω_m                       – matter density (prior ≈ Uniform[0.1, 0.6])
  OmL     : Ω_Λ                       – cosmological constant (prior ≈ Uniform[0, 1−Ω_m])
  stigma  : Stigma theory parameter   – (prior ≈ Uniform[1.001, 2.0], > cpr²)
  mups2   : −υ₂ > 0                   – (prior ≈ Uniform[0.5, 2.5])
  cpr     : ϖ₀ (field IC ≈ 0.99)      – (prior ≈ Uniform[0.6, 0.999])

OmL is sampled so the data determines whether a cosmological constant is needed
on top of the Stigma field (pure modified gravity = OmL→0 region of posterior).
h0rd is sampled to marginalise over the sound-horizon / H0 degeneracy.

Usage
-----
    source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
    cd /Users/starfish/Desktop/PhD2025-26/distances/TorC/Stigma/python
    python stigma_nss.py

Output
------
    output/stigma_desi_dead.pkl  – dead points (NSInfo object)
    output/stigma_desi_live.pkl  – live state  (NSState object)
"""

import os, pickle, sys
import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import tqdm

jax.config.update("jax_enable_x64", True)

# ── Add python/ dir to path for equations_jax ────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from equations_jax import (
    solve_stigma_jax, compute_desi_distances,
    constraints_satisfied, C_KMS,
    OMEGA_R, H_NORM, RD_PLANCK
)

# ── DESI DR2 data ─────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__),
                         "../../../desidr2")

_mean_raw = np.array([
    7.944,   # DV/rs @ z=0.295
    13.587,  # DM/rs @ z=0.510
    21.863,  # DH/rs @ z=0.510
    17.347,  # DM/rs @ z=0.706
    19.458,  # DH/rs @ z=0.706
    21.574,  # DM/rs @ z=0.934
    17.641,  # DH/rs @ z=0.934
    27.605,  # DM/rs @ z=1.321
    14.178,  # DH/rs @ z=1.321
    30.519,  # DM/rs @ z=1.484
    12.816,  # DH/rs @ z=1.484
    38.988,  # DM/rs @ z=2.330
     8.632,  # DH/rs @ z=2.330
])

_cov_raw = np.array([
    [0.005626, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.028561, -0.034277425, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, -0.034277425, 0.182329, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.0324, -0.02527848, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, -0.02527848, 0.110224, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.023409, -0.012549825, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, -0.012549825, 0.037249, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.1024, -0.03034528, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.03034528, 0.047089, 0.000, 0.000, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.574564, -0.190149606, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.190149606, 0.263169, 0.000, 0.000],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.281961, -0.023114961],
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.023114961, 0.010201],
])

# Pre-compute Cholesky precision (for fast log-likelihood)
_mean_jax = jnp.array(_mean_raw)
_cov_jax  = jnp.array(_cov_raw)
_cov_inv  = jnp.linalg.inv(_cov_jax)
_log_det  = jnp.log(jnp.linalg.det(_cov_jax))
_N_data   = 13


# ── Log-prior ─────────────────────────────────────────────────────────────────

def logprior_fn(params):
    """
    Uniform priors with hard boundary enforcement.
    Returns log-prior value for a SINGLE particle dict.

    6 parameters: h0rd, omega_m, OmL, stigma, mups2, cpr.
    OmL sampled so the posterior reveals whether a Λ is needed.
    """
    h0rd    = params["h0rd"]
    omega_m = params["omega_m"]
    OmL     = params["OmL"]
    stigma  = params["stigma"]
    mups2   = params["mups2"]
    cpr     = params["cpr"]

    lp = jnp.array(0.0)

    # Uniform priors (loc, scale = width)
    lp += jax.scipy.stats.uniform.logpdf(h0rd,    3000.0, 17000.0)  # [3000, 20000]
    lp += jax.scipy.stats.uniform.logpdf(omega_m, 0.1,    0.5)      # [0.1, 0.6]
    lp += jax.scipy.stats.uniform.logpdf(OmL,     0.0,    1.0)      # [0.0, 1.0]
    lp += jax.scipy.stats.uniform.logpdf(stigma,  1.001,  0.999)    # [1.001, 2.0]
    lp += jax.scipy.stats.uniform.logpdf(mups2,   0.5,    2.0)      # [0.5, 2.5]
    lp += jax.scipy.stats.uniform.logpdf(cpr,     0.6,    0.399)    # [0.6, 0.999]

    # Flat universe: omega_m + OmL + omega_r < 1  (Stigma provides the rest)
    lp = jnp.where(omega_m + OmL + OMEGA_R < 1.0, lp, -jnp.inf)

    # Stigma theory hard constraints
    c1, c2 = constraints_satisfied(stigma, cpr)
    lp = jnp.where(c1 > 0.0, lp, -jnp.inf)
    lp = jnp.where(c2 > 0.0, lp, -jnp.inf)

    return lp


# ── Log-likelihood ────────────────────────────────────────────────────────────

def loglikelihood_fn(params):
    """
    DESI DR2 BAO Gaussian log-likelihood for a SINGLE particle dict.

    Solves the Stigma ODE with the sampled OmL, uses H0_actual from the ODE
    and the sampled h0rd (= H0_actual * rd) for BAO distances.

    IMPORTANT: also enforces the prior as a hard gate.  Without this, particles
    that violate the prior but happen to produce a finite ODE result keep a
    finite logL.  BlackJAX NSS then uses them as slice-sampling start points,
    where logslice = -inf + log(u) = -inf, making ANY proposal valid and
    causing parameters to drift into the millions.  Gating on logprior here
    ensures all out-of-prior particles have logL = -1e30, so they are always
    killed first and never become start points.
    """
    # ── Prior gate (must come before ODE to catch out-of-bounds cheaply) ────
    lp = logprior_fn(params)

    h0rd    = params["h0rd"]
    omega_m = params["omega_m"]
    OmL     = params["OmL"]
    stigma  = params["stigma"]
    mups2   = params["mups2"]
    cpr     = params["cpr"]

    # Solve Stigma ODE with sampled OmL.
    # rtol/atol=1e-6 is accurate enough for DESI (~1-5% data precision)
    # and ~2x faster than 1e-7.
    asol, hsol, a0, H0_actual = solve_stigma_jax(
        cpr, omega_m, stigma, mups2,
        OmL=OmL,
        h=H_NORM, omr=OMEGA_R,
        rtol=1e-6, atol=1e-6,
        n_save=500, tau_max=1e3,
    )

    # BAO distances: h0rd = H0 * rd (sampled); rd = h0rd / H0_actual
    rd = h0rd / H0_actual          # effective sound horizon [Mpc]
    theory = compute_desi_distances(asol, hsol, a0, H0_actual, rd=rd)

    # Gaussian log-likelihood:  -0.5*(x-mu)^T C^{-1} (x-mu) - 0.5*log|2pi C|
    residual = theory - _mean_jax
    chi2     = residual @ _cov_inv @ residual
    logl     = -0.5 * chi2 - 0.5 * (_N_data * jnp.log(2*jnp.pi) + _log_det)

    # Return -1e30 if ODE failed OR if prior is violated
    logl = jnp.where(jnp.isfinite(logl), logl, jnp.array(-1e30))
    return jnp.where(jnp.isfinite(lp), logl, jnp.array(-1e30))


# ── NSS setup ─────────────────────────────────────────────────────────────────

def run_nss(num_live=500, seed=0, output_dir="output",
            num_inner_steps=None, max_slice_steps=5, max_shrinkage=20):
    """
    Run NSS.

    Parameters
    ----------
    num_live          : Number of live points (500 for production, 200 exploration).
    num_inner_steps   : Slice moves per replacement (default: 3*ndim = 18).
                        The blackjax-nss skill recommends 5*ndim for production;
                        3*ndim is fine for an exploration run.
    max_slice_steps   : Max bracket *expansions* per slice move (default 5).
                        Blackjax default is 10; 5 halves the expansion cost.
    max_shrinkage     : Max bracket *contractions* per slice move (default 20).
                        Blackjax default is 100; 20 gives a 5x speedup.
    """
    os.makedirs(output_dir, exist_ok=True)

    rng_key = jax.random.PRNGKey(seed)
    num_dims = 6   # h0rd, omega_m, OmL, stigma, mups2, cpr
    num_delete = max(1, num_live // 10)
    if num_inner_steps is None:
        num_inner_steps = num_dims * 3   # 18  (use 5*ndim=30 for publication)

    print(f"NSS: num_live={num_live}, num_delete={num_delete}, "
          f"num_inner_steps={num_inner_steps}, "
          f"max_slice_steps={max_slice_steps}, max_shrinkage={max_shrinkage}")

    # ── Build algorithm ──────────────────────────────────────────────────────
    algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
        max_steps=max_slice_steps,
        max_shrinkage=max_shrinkage,
    )

    # ── Initialise live particles from prior (rejection sampling) ────────────
    # Sample more than needed and keep only those that pass the prior hard
    # constraints (flatness + Stigma viability).  This avoids wasting the
    # first ~30% of NS iterations killing guaranteed-invalid initial points,
    # and—crucially—ensures no initial particle has logprior = -inf, which
    # would otherwise cause the slice sampler to drift outside the prior.
    import numpy as np

    print("Sampling valid prior particles …")
    valid = {k: [] for k in ("h0rd", "omega_m", "OmL", "stigma", "mups2", "cpr")}
    n_found = 0
    batch = 0
    while n_found < num_live:
        rng_key, k1, k2, k3, k4, k5, k6 = jax.random.split(rng_key, 7)
        n_try = max(num_live * 4, 800)
        h0rd_b    = np.array(jax.random.uniform(k1, (n_try,), minval=3000.0,  maxval=20000.0))
        omega_m_b = np.array(jax.random.uniform(k2, (n_try,), minval=0.1,     maxval=0.6))
        OmL_b     = np.array(jax.random.uniform(k3, (n_try,), minval=0.0,     maxval=1.0))
        stigma_b  = np.array(jax.random.uniform(k4, (n_try,), minval=1.001,   maxval=2.0))
        mups2_b   = np.array(jax.random.uniform(k5, (n_try,), minval=0.5,     maxval=2.5))
        cpr_b     = np.array(jax.random.uniform(k6, (n_try,), minval=0.6,     maxval=0.999))
        # Apply hard constraints:
        flat_ok = (omega_m_b + OmL_b + float(OMEGA_R)) < 1.0
        c1 = stigma_b - cpr_b**2
        c2 = cpr_b**2 - stigma_b * (5 + 4 * stigma_b * (-2 + cpr_b**2))
        ok = flat_ok & (c1 > 0) & (c2 > 0)
        for k, arr in zip(("h0rd","omega_m","OmL","stigma","mups2","cpr"),
                          (h0rd_b, omega_m_b, OmL_b, stigma_b, mups2_b, cpr_b)):
            valid[k].extend(arr[ok].tolist())
        n_found = len(valid["h0rd"])
        batch += 1
        if batch > 50:
            raise RuntimeError("Could not sample enough valid prior particles")

    particles = {k: jnp.array(valid[k][:num_live]) for k in valid}
    print(f"  Found {n_found} valid prior particles in {batch} batch(es)")
    print("Initialising live points …")
    live = jax.jit(algo.init)(particles)
    print(f"  logZ = {live.logZ:.3f},  logZ_live = {live.logZ_live:.3f}")

    # ── Run ──────────────────────────────────────────────────────────────────
    step_fn = jax.jit(algo.step)
    dead = []

    with tqdm.tqdm(desc="Dead points", unit=" pts") as pbar:
        while not (live.logZ_live - live.logZ < -3):
            rng_key, subkey = jax.random.split(rng_key)
            live, dead_info = step_fn(subkey, live)
            dead.append(dead_info)
            pbar.update(num_delete)

    # ── Finalise ─────────────────────────────────────────────────────────────
    dead = blackjax.ns.utils.finalise(live, dead)
    print(f"\nDone.  logZ = {live.logZ:.4f} ± {jnp.exp(live.logZ_live - live.logZ):.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    with open(os.path.join(output_dir, "stigma_desi_dead.pkl"), "wb") as f:
        pickle.dump(dead, f)
    with open(os.path.join(output_dir, "stigma_desi_live.pkl"), "wb") as f:
        pickle.dump(live, f)

    print(f"Saved to {output_dir}/stigma_desi_dead.pkl")
    return live, dead


# ── Anesthetic helper ─────────────────────────────────────────────────────────

def load_as_nested_samples(output_dir="output"):
    """Load saved dead points and return an anesthetic NestedSamples object."""
    from anesthetic import NestedSamples

    with open(os.path.join(output_dir, "stigma_desi_dead.pkl"), "rb") as f:
        dead = pickle.load(f)

    labels = {
        "h0rd":    r"$H_0 r_d$ [km/s]",
        "omega_m": r"$\Omega_m$",
        "OmL":     r"$\Omega_\Lambda$",
        "stigma":  r"$\varsigma$",
        "mups2":   r"$-\upsilon_2$",
        "cpr":     r"$\varpi_0$",
    }

    return NestedSamples(
        dead.particles,
        logL=dead.loglikelihood,
        logL_birth=dead.loglikelihood_birth,
        labels=labels,
    )


# ── Quick likelihood test ─────────────────────────────────────────────────────

def test_likelihood(stigma=1.1, mups2=4/3, cpr=0.99, omega_m=0.3, OmL=0.7, h0rd=9900.0):
    """Run a single likelihood evaluation and print timing."""
    import time
    params = {
        "h0rd":    jnp.array(h0rd),
        "omega_m": jnp.array(omega_m),
        "OmL":     jnp.array(OmL),
        "stigma":  jnp.array(stigma),
        "mups2":   jnp.array(mups2),
        "cpr":     jnp.array(cpr),
    }

    print("Prior:", logprior_fn(params))

    # First call (includes JIT compilation)
    t0 = time.time()
    logl_fn = jax.jit(loglikelihood_fn)
    logl = logl_fn(params)
    logl.block_until_ready()
    t1 = time.time()
    print(f"log L = {float(logl):.4f}  (first call including JIT: {t1-t0:.2f}s)")

    # Second call (cached)
    t0 = time.time()
    logl = logl_fn(params)
    logl.block_until_ready()
    t1 = time.time()
    print(f"log L = {float(logl):.4f}  (cached JIT: {t1-t0:.4f}s)")
    return float(logl)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Stigma Q-torsion vs DESI DR2: BlackJAX nested slice sampling")
    parser.add_argument("--test", action="store_true",
                        help="Run a single likelihood test instead of full NSS")
    parser.add_argument("--live", type=int, default=200,
                        help="Number of live points (default 200 for exploration)")
    parser.add_argument("--inner", type=int, default=None,
                        help="Slice moves per replacement (default 3*ndim=18)")
    parser.add_argument("--max-slice", type=int, default=5,
                        help="Max bracket expansions per slice move (default 5)")
    parser.add_argument("--max-shrink", type=int, default=20,
                        help="Max bracket contractions per slice move (default 20)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="output")
    args = parser.parse_args()

    if args.test:
        test_likelihood()
    else:
        run_nss(
            num_live=args.live,
            seed=args.seed,
            output_dir=args.output,
            num_inner_steps=args.inner,
            max_slice_steps=args.max_slice,
            max_shrinkage=args.max_shrink,
        )
