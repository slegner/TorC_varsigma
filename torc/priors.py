"""Sampling priors and the bounds-clamped likelihood wrapper.

The prior is a uniform box.  IMPORTANT: the slice sampler (blackjax nss)
proposes points outside the box; the prior density rejects them, but an
out-of-bounds ODE solve burns the full max_steps budget and — because
vmapped lanes run in lockstep — stalls its entire batch (~20x slowdown,
measured at the committed max_steps=8192; ~3x at the current 1024).
loglikelihood_bounded clamps such points to a safe fiducial before the
solve and floors their logL at -1e30 directly: results are identical,
batches stay fast.  Always hand THIS to the sampler.

mups2 IS NOT SAMPLED.  It is an exact background gauge parameter: it
rescales the field's clock (tau -> sqrt(mups2)*tau) and cancels out of
E(z), verified to 1e-9 by tests/test_theory.py::test_mups2_gauge.  Its
posterior was flat across the whole prior and, for a flat direction with a
normalised prior, the evidence factorises so ln Z is unchanged -- confirmed
empirically at 1.29 sigma over four seeds each (sampled -7.880 +- 0.106 vs
fixed -8.110 +- 0.338).  Fixing it drops the dimension 5 -> 4, which takes
num_inner_steps from 25 to 20 and buys ~1.17x.

Set SAMPLE_MUPS2 = True to restore it -- necessary for any perturbation-level
observable, where mups2 becomes physical.
"""
import jax
import jax.numpy as jnp

from .likelihood import loglikelihood, MUPS2_FIXED

SAMPLE_MUPS2 = False

_ALL_BOUNDS = {
    "omega_m": (0.01,   0.99),
    "h0rd":    (3650.0, 18250.0),
    "stigma":  (1.0001, 3.0),
    "mups2":   (0.5,    3.0),   # exact background gauge parameter
    "OmL":     (0.1,    1.5),
}

PRIOR_BOUNDS = {k: v for k, v in _ALL_BOUNDS.items()
                if SAMPLE_MUPS2 or k != "mups2"}

_SAFE = {k: v for k, v in
         dict(omega_m=0.3, h0rd=10000.0, stigma=1.5, mups2=1.5, OmL=0.7).items()
         if k in PRIOR_BOUNDS}


def logprior_fn(params):
    lp = 0.0
    for name, (lo, hi) in PRIOR_BOUNDS.items():
        lp += jax.scipy.stats.uniform.logpdf(params[name], lo, hi - lo)
    return lp


def sample_prior(rng_key, n):
    keys = jax.random.split(rng_key, len(PRIOR_BOUNDS))
    return {name: jax.random.uniform(k, (n,), minval=lo, maxval=hi)
            for k, (name, (lo, hi)) in zip(keys, PRIOR_BOUNDS.items())}


def loglikelihood_bounded(params):
    """loglikelihood, safe and fast for out-of-prior slice proposals."""
    inb = jnp.bool_(True)
    for name, (lo, hi) in PRIOR_BOUNDS.items():
        inb = inb & (params[name] >= lo) & (params[name] <= hi)
    safe = {name: jnp.where(inb, params[name], _SAFE[name])
            for name in PRIOR_BOUNDS}
    if not SAMPLE_MUPS2:
        safe["mups2"] = MUPS2_FIXED   # the likelihood always needs the key
    return jnp.where(inb, loglikelihood(safe), -1e30)
