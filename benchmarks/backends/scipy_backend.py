"""scipy.integrate.solve_ivp backend.

Uses the pure-numpy RHS from common.py, which calls the SAME dda/ddcp
imported from torc/theory.py.  Not JAX-traceable, so this cannot be used
inside the nested-sampling pipeline -- reference only.

Standalone:  python benchmarks/backends/scipy_backend.py
"""
from scipy.integrate import solve_ivp

from common import setup, rhs_np, norm_E, timeit, report

SOLVERS = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
STIFF = {"Radau", "BDF"}


def solve(solver="DOP853", tol=1e-8, params=None):
    x0, y0, args, xs = setup(params)
    label = f"{solver}{' [stiff]' if solver in STIFF else ''}"

    def run():
        return solve_ivp(rhs_np, [x0, 0.0], list(y0), t_eval=xs, args=(args,),
                         method=solver, rtol=tol, atol=tol)

    try:
        s = run()
    except Exception as e:
        return dict(backend="scipy", solver=label, tol=tol, ms=None, ok=False,
                    E=None, steps=None, note=type(e).__name__)
    if not s.success:
        return dict(backend="scipy", solver=label, tol=tol, ms=None, ok=False,
                    E=None, steps=None, note="failed")
    ms = timeit(run, reps=5)
    return dict(backend="scipy", solver=label, tol=tol, ms=ms, ok=True,
                E=norm_E(s.y[1], xs), steps=int(s.nfev), note="")


def run_all(tols=(1e-5, 1e-8), params=None):
    return [solve(s, t, params) for t in tols for s in SOLVERS]


if __name__ == "__main__":
    report("scipy solve_ivp", run_all())
