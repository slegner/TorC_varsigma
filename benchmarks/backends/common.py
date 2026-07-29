"""Shared setup for every ODE backend: identical problem, identical grid.

Every backend file in this folder imports from here, so they are guaranteed
to be solving the same initial-value problem on the same output grid.  The
only thing that differs between them is the integrator.

Problem: TorC background equations in x = ln(a), state y = (cp, da, dcp),
integrated from the early-universe series ICs up to x = 0 (today).
Parameters: the _SAFE point from torc/priors.py.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..")
sys.path.insert(0, REPO)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from torc.theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                          dda, ddcp, Z_SPLIT, N_EARLY, N_LATE, OMEGA_R_DEFAULT)
from torc.priors import _SAFE

XSPLIT = -np.log(1.0 + Z_SPLIT)


def setup(params=None):
    """(x0, y0, args, xs) for the given parameter dict (default: _SAFE)."""
    p = params or _SAFE
    om, st, mu, ol = p["omega_m"], p["stigma"], p["mups2"], p["OmL"]
    omr = OMEGA_R_DEFAULT
    tau = small(om, omr, st, mu)
    x0 = float(jnp.log(aseries(tau, om, omr, ol, st, mu)))
    y0 = (float(cpseries(tau, om, omr, ol, st, mu)),
          float(aseriesp(tau, om, omr, ol, st, mu)),
          float(cpseriesp(tau, om, omr, ol, st, mu)))
    xs = np.concatenate([np.linspace(x0, XSPLIT, N_EARLY, endpoint=False),
                         np.linspace(XSPLIT, 0.0, N_LATE)])
    return x0, y0, (om, omr, ol, st, mu), xs


def rhs_np(x, y, args):
    """Pure-numpy RHS. dda/ddcp are plain arithmetic (no jnp), so they run
    natively on floats -- this is what scipy and CVODE call, with no JAX
    dispatch overhead."""
    cp, da, dcp = y
    om, omr, ol, st, mu = args
    a = np.exp(x)
    fac = a / da
    return [dcp * fac,
            dda(a, da, cp, dcp, om, omr, ol, st, mu) * fac,
            ddcp(a, da, cp, dcp, om, omr, ol, st, mu) * fac]


def norm_E(da_vals, xs):
    """E(z) = H/H(a=1), normalised to 1 today."""
    E = np.asarray(da_vals) * np.exp(-2.0 * xs)
    return E / E[-1]


def timeit(fn, reps=7):
    """Median wall time in ms, discarding the first (compiling) call."""
    import time
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def report(name, results):
    print(f"--- {name} ---")
    print(f"{'solver':24s} {'tol':>7s} {'ms':>10s} {'ok':>5s}")
    for r in results:
        ms = f"{r['ms']:10.3f}" if r["ms"] is not None else f"{'FAILED':>10s}"
        print(f"{r['solver']:24s} {r['tol']:7.0e} {ms} {str(r['ok']):>5s}")
