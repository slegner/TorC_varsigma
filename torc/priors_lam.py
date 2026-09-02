"""Priors for the Lambda model, built to be directly comparable to torc/priors.py.

The stigma prior machinery (`_logprior_stigma`, `_sample_stigma`) and the
bounds on the shared parameters are IMPORTED from torc.priors rather than
restated, so the two cannot drift apart: any ln Z difference between a
lambda chain and the existing no-lambda chain is then attributable to the
model, not to a prior that quietly changed.  In particular STIGMA_PRIOR is
whatever torc/priors.py currently says (at the time of writing:
"powerlaw", matching chains/torc_ns_julia_1000_powerlaw.npz).

TWO CONFIGURATIONS, selected by SAMPLE_OML:

  SAMPLE_OML = True   omega_m, h0rd, stigma, OmL, lam        (5 parameters)
      The general model.  It nests TorC (lam = 0) and the Lambda-only case
      (OmL = 0), so the posterior shows directly whether the data want a
      lam term on top of the fluid.

  SAMPLE_OML = False  omega_m, h0rd, stigma, lam             (4 parameters)
      OmL is pinned to 0 and lam alone sources the late-time acceleration.
      Same dimension as the current TorC run, so ln Z is comparable to it
      on equal footing.

LAM PRIOR: uniform on [0, 1.5].  It matches the width of the OmL box but
reaches down to 0, so the no-lambda model sits INSIDE the prior support
rather than on an excluded edge -- a lam posterior that piles up against 0
is then meaningful evidence that the data do not want the term, instead of
an artefact of where the prior was cut.

MUPS2 IS NOT SAMPLED, exactly as in torc/priors.py: it is an exact
background gauge parameter that cancels out of E(z).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from .likelihood_lam import loglikelihood, MUPS2_FIXED
from .priors import (_ALL_BOUNDS, _logprior_stigma, _sample_stigma,
                     STIGMA_PRIOR, STIGMA_LAMBDA, STIGMA_FLOOR)

# set TORC_SAMPLE_OML=0 in the environment to pin OmL = 0 (lambda-only model)
SAMPLE_OML = os.environ.get("TORC_SAMPLE_OML", "1").lower() not in ("0", "false", "no")
SAMPLE_MUPS2 = False       # exact gauge parameter; see torc/priors.py
# set TORC_SAMPLE_LAM=0 to drop lam entirely (recovers the no-lambda model,
# so one run script covers both with and without the term)
SAMPLE_LAM = os.environ.get("TORC_SAMPLE_LAM", "1").lower() not in ("0", "false", "no")
LAM_FIXED = 0.0

LAM_BOUNDS = (0.0, 1.5)
OML_FIXED = 0.0            # value used when SAMPLE_OML is False

_ORDER = ["omega_m", "h0rd", "stigma", "OmL", "lam"]

_BOUNDS_ALL = dict(_ALL_BOUNDS)
_BOUNDS_ALL["lam"] = LAM_BOUNDS

PRIOR_BOUNDS = {k: _BOUNDS_ALL[k] for k in _ORDER
                if (k != "OmL" or SAMPLE_OML) and (k != "lam" or SAMPLE_LAM)}

_SAFE = {"omega_m": 0.3, "h0rd": 10000.0, "stigma": 1.5,
         "OmL": 0.7, "lam": 0.5}
_SAFE = {k: v for k, v in _SAFE.items() if k in PRIOR_BOUNDS}


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


def make_bounded(logl_fn):
    """Wrap any logL(params) so out-of-prior slice proposals are safe and fast.

    Always hand the RESULT to the sampler -- see the note in torc/priors.py:
    an out-of-support ODE solve burns the full step budget and stalls its
    whole vmapped batch.
    """
    def bounded(params):
        inb = jnp.bool_(True)
        for name, (lo, hi) in PRIOR_BOUNDS.items():
            inb = inb & (params[name] >= lo) & (params[name] <= hi)
        safe = {name: jnp.where(inb, params[name], _SAFE[name])
                for name in PRIOR_BOUNDS}
        if not SAMPLE_MUPS2:
            safe["mups2"] = MUPS2_FIXED
        if not SAMPLE_OML:
            safe["OmL"] = OML_FIXED
        if not SAMPLE_LAM:
            safe["lam"] = LAM_FIXED
        return jnp.where(inb, logl_fn(safe), -1e30)
    return bounded


# BAO-only wrapper, kept under its original name for existing callers
loglikelihood_bounded = make_bounded(loglikelihood)
