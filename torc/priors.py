"""Sampling priors and the bounds-clamped likelihood wrapper.

IMPORTANT: the slice sampler (blackjax nss) proposes points outside the
prior support; the prior density rejects them, but an out-of-support ODE
solve burns the full max_steps budget and — because vmapped lanes run in
lockstep — stalls its entire batch (~20x slowdown, measured at the
committed max_steps=8192; ~3x at the current 1024).  loglikelihood_bounded
clamps such points to a safe fiducial before the solve and floors their
logL at -1e30 directly: results are identical, batches stay fast.  Always
hand THIS to the sampler.

STIGMA PRIOR (STIGMA_PRIOR)
---------------------------
"exponential" (default) is an Occam prior anchored at the LCDM limit:

    pi(s) = exp(-(s - s0)/lam) / lam,   s >= s0 = 1.0001,  lam = 2.0

Prior belief decays smoothly with distance from LCDM rather than being flat
out to an arbitrary wall.  s = 1 exactly is a coordinate singularity (the
series expansions carry (-1 + stigma) in denominators and fail to solve), so
the support starts a hair above it.

With lam = 2.0 the prior is deliberately broad: the density at s = 3 is
exp(-2) = 37% of its peak, and 37% of the prior mass lies above s = 3 (for
an exponential those two numbers coincide).  So s = 3 -- the old uniform
upper wall -- is comfortably supported rather than excluded, while departures
from LCDM are still penalised in proportion to their size.

STIGMA_MAX is a SOLVER GUARD, not a prior bound.  The prior is unbounded
above in intent; truncating at 50 removes prior mass exp(-49/2) ~ 2e-11,
unreachable in any finite run, while ensuring the sampler can never hand the
ODE a wild value.  The normalisation below accounts for the truncation
exactly anyway.  Solves were verified over s in [1.0001, 50] (~70 steps
throughout, no failures), so the guard sits exactly at the verified edge.

"uniform" restores the original flat prior on [1.0001, 3.0].

CHANGING THIS CHANGES THE EVIDENCE.  ln Z under different priors answers
different questions and the values must never be swapped for one another.
At 1000 live points against LCDM (ln Z = -7.739 +- 0.096):

    exponential lam=0.5   -7.521 +- 0.097   +0.22   1.24:1 TorC
    uniform [1, 3]        -8.073 +- 0.096   -0.33   1.40:1 LCDM
    exponential lam=2.0   -8.461 +- 0.102   -0.72   2.06:1 LCDM

All inconclusive (|Delta ln Z| < 1), and the sign flips with the prior --
the data do not distinguish the models.  ln Z falls monotonically as the
prior widens: the posterior sits at s ~ 1.3-1.4 in every case, so a broader
prior only adds volume where the likelihood is not.  That is the Occam
factor, and it is the reason lam is a reported choice, not a free knob.

MUPS2 IS NOT SAMPLED.  It is an exact background gauge parameter: it
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
import numpy as np

from .likelihood import loglikelihood, MUPS2_FIXED

SAMPLE_MUPS2 = False

STIGMA_PRIOR = "powerlaw"         # "powerlaw" | "exponential" | "uniform"
STIGMA_LAMBDA = 1.0               # decay scale of the exponential
STIGMA_FLOOR = 1.0001             # s = 1 exactly is singular
STIGMA_MAX = 1.0e4                # solver guard; see note on truncation below
STIGMA_UNIFORM_MAX = 3.0          # only used when STIGMA_PRIOR == "uniform"

# Only the "uniform" prior has a genuine wall; the exponential and the power
# law are both defined out to STIGMA_MAX and their sampling bounds must match
# their support.  The powerlaw branch used to fall through to
# STIGMA_UNIFORM_MAX = 3, which made loglikelihood_bounded floor everything
# above 3 at -1e30: in chains/torc_ns_julia_1000_powerlaw.npz that is 2.6% of
# the dead points (max stigma 1668), so that chain's ln Z = -8.0264 carries an
# unintended truncation penalty and is NOT comparable to runs made after this
# fix.  Solves were re-verified finite and well behaved over the whole range
# stigma in [1.0001, 1e4] (E(z=1) -> 2.0 smoothly), so there is nothing to
# guard against here.
_STIGMA_HI = STIGMA_UNIFORM_MAX if STIGMA_PRIOR == "uniform" else STIGMA_MAX

# normalisation of the TRUNCATED exponential on [STIGMA_FLOOR, STIGMA_MAX]:
#   int pi = lam * (1 - exp(-(hi - lo)/lam))
# the bracket is 1 - 2e-11, but carry it so the density integrates to 1 exactly
_STIGMA_NORM = STIGMA_LAMBDA * (
    1.0 - np.exp(-(STIGMA_MAX - STIGMA_FLOOR) / STIGMA_LAMBDA))

# power law: int_{s0}^{smax} s^-2 ds = 1/s0 - 1/smax
_PL_NORM = 1.0 / STIGMA_FLOOR - 1.0 / STIGMA_MAX

_ALL_BOUNDS = {
    "omega_m": (0.01,   0.99),
    "h0rd":    (3650.0, 18250.0),
    "stigma":  (STIGMA_FLOOR, _STIGMA_HI),
    "mups2":   (0.5,    3.0),   # exact background gauge parameter
    "OmL":     (0.1,    1.5),
}

PRIOR_BOUNDS = {k: v for k, v in _ALL_BOUNDS.items()
                if SAMPLE_MUPS2 or k != "mups2"}

_SAFE = {k: v for k, v in
         dict(omega_m=0.3, h0rd=10000.0, stigma=1.5, mups2=1.5, OmL=0.7).items()
         if k in PRIOR_BOUNDS}


def _logprior_stigma(s):
    """log pi(stigma), normalised over [STIGMA_FLOOR, _STIGMA_HI]."""
    if STIGMA_PRIOR == "powerlaw":
        inside = (s >= STIGMA_FLOOR) & (s <= STIGMA_MAX)
        lp = -2.0 * jnp.log(jnp.where(inside, s, 1.0)) - jnp.log(_PL_NORM)
        return jnp.where(inside, lp, -jnp.inf)
    if STIGMA_PRIOR == "uniform":
        return jax.scipy.stats.uniform.logpdf(
            s, STIGMA_FLOOR, _STIGMA_HI - STIGMA_FLOOR)
    inside = (s >= STIGMA_FLOOR) & (s <= STIGMA_MAX)
    lp = -(s - STIGMA_FLOOR) / STIGMA_LAMBDA - jnp.log(_STIGMA_NORM)
    return jnp.where(inside, lp, -jnp.inf)


def _sample_stigma(key, n):
    """Exact inverse-CDF draw from the prior."""
    if STIGMA_PRIOR == "powerlaw":
        # CDF(s) = (1/s0 - 1/s) / (1/s0 - 1/smax)
        #   =>  s = 1 / (1/s0 - u * (1/s0 - 1/smax))
        u = jax.random.uniform(key, (n,))
        s = 1.0 / (1.0 / STIGMA_FLOOR - u * _PL_NORM)
        return jnp.clip(s, STIGMA_FLOOR, STIGMA_MAX)
    if STIGMA_PRIOR == "uniform":
        return jax.random.uniform(key, (n,), minval=STIGMA_FLOOR,
                                  maxval=_STIGMA_HI)
    # CDF(s) = (1 - exp(-(s-lo)/lam)) / (1 - exp(-(hi-lo)/lam))
    # => s = lo - lam * log(1 - u * (1 - exp(-(hi-lo)/lam)))
    u = jax.random.uniform(key, (n,))
    frac = 1.0 - np.exp(-(STIGMA_MAX - STIGMA_FLOOR) / STIGMA_LAMBDA)
    s = STIGMA_FLOOR - STIGMA_LAMBDA * jnp.log1p(-u * frac)
    return jnp.clip(s, STIGMA_FLOOR, STIGMA_MAX)


def logprior_fn(params):
    lp = 0.0
    for name, (lo, hi) in PRIOR_BOUNDS.items():
        if name == "stigma":
            lp += _logprior_stigma(params[name])
        else:
            lp += jax.scipy.stats.uniform.logpdf(params[name], lo, hi - lo)
    return lp


def sample_prior(rng_key, n):
    keys = jax.random.split(rng_key, len(PRIOR_BOUNDS))
    out = {}
    for k, (name, (lo, hi)) in zip(keys, PRIOR_BOUNDS.items()):
        out[name] = (_sample_stigma(k, n) if name == "stigma"
                     else jax.random.uniform(k, (n,), minval=lo, maxval=hi))
    return out


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
