"""Sampling priors and the bounds-clamped likelihood wrapper.

The prior is a uniform box.  IMPORTANT: the slice sampler (blackjax nss)
proposes points outside the box; the prior density rejects them, but an
out-of-bounds ODE solve burns the full max_steps budget and — because
vmapped lanes run in lockstep — stalls its entire batch (~20x slowdown
measured).  loglikelihood_bounded clamps such points to a safe fiducial
before the solve and floors their logL at -1e30 directly: results are
identical, batches stay fast.  Always hand THIS to the sampler.
"""
import jax
import jax.numpy as jnp

from .likelihood import loglikelihood

PRIOR_BOUNDS = {
    "omega_m": (0.01,   0.99),
    "h0rd":    (3650.0, 18250.0),
    "stigma":  (1.0001, 3.0),
    "mups2":   (0.5,    3.0),   # exact background gauge parameter: fixable
    "OmL":     (0.1,    1.5),
}

_SAFE = dict(omega_m=0.3, h0rd=10000.0, stigma=1.5, mups2=1.5, OmL=0.7)


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
    return jnp.where(inb, loglikelihood(safe), -1e30)
