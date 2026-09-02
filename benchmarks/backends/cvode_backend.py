"""SUNDIALS CVODE backend (via scikits.odes).

Same pure-numpy RHS as scipy.  Requires the SUNDIALS C library:
    brew install sundials
    SUNDIALS_INST=$(brew --prefix sundials) pip install scikits-odes

NOTE: the first CVODE call in a process costs ~250-800 ms of one-time
initialisation.  timeit() discards it, but do not benchmark a single call.

Standalone:  python benchmarks/backends/cvode_backend.py
"""
import numpy as np

from common import setup, norm_E, timeit, report
from torc.theory import dda, ddcp

SOLVERS = {"CVODE Adams": "ADAMS", "CVODE BDF [stiff]": "BDF"}


def solve(solver="CVODE Adams", tol=1e-8, params=None):
    try:
        from scikits.odes import ode
    except Exception as e:
        return dict(backend="cvode", solver=solver, tol=tol, ms=None,
                    ok=False, E=None, steps=None, note="not installed")
    x0, y0, args, xs = setup(params)
    y0 = np.asarray(y0)

    def cv_rhs(x, y, ydot):
        cp, da, dcp = y
        om, omr, ol, st, mu = args
        a = np.exp(x)
        fac = a / da
        ydot[0] = dcp * fac
        ydot[1] = dda(a, da, cp, dcp, om, omr, ol, st, mu) * fac
        ydot[2] = ddcp(a, da, cp, dcp, om, omr, ol, st, mu) * fac

    def run():
        return ode("cvode", cv_rhs, lmm_type=SOLVERS[solver],
                   rtol=tol, atol=tol, old_api=False).solve(xs, y0)

    try:
        out = run()
    except Exception as e:
        return dict(backend="cvode", solver=solver, tol=tol, ms=None, ok=False,
                    E=None, steps=None, note=type(e).__name__)
    if out.flag != 0:
        return dict(backend="cvode", solver=solver, tol=tol, ms=None, ok=False,
                    E=None, steps=None, note=f"flag={out.flag}")
    ms = timeit(run, reps=5)
    return dict(backend="cvode", solver=solver, tol=tol, ms=ms, ok=True,
                E=norm_E(np.asarray(out.values.y)[:, 1], xs), steps=None,
                note="")


def run_all(tols=(1e-5, 1e-8), params=None):
    return [solve(s, t, params) for t in tols for s in SOLVERS]


if __name__ == "__main__":
    report("SUNDIALS CVODE", run_all())
