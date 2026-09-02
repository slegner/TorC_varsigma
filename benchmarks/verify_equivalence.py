"""Are all the backends solving the SAME equations?

Timing comparisons are meaningless if the backends disagree on the physics.
This solves the identical problem with every available backend at a tight
tolerance and compares E(z) against one common reference
(scipy DOP853 @ 1e-13).  Anything above ~1e-9 means a transcription bug,
not a solver-accuracy difference.

Backends:
  diffrax        -- imports the vector field straight from torc/theory.py
  scipy          -- pure-numpy RHS calling the SAME dda/ddcp from theory.py
  SUNDIALS CVODE -- same numpy RHS again (optional: needs scikits.odes)
  Julia          -- RHS auto-generated from theory.py by gen_julia_rhs.py;
                    read from julia_E.tsv if benchmark_julia.jl has been run

Run:
    python benchmarks/gen_julia_rhs.py        # writes torc_rhs.jl + reference_E.npz
    julia   benchmarks/benchmark_julia.jl     # writes julia_E.tsv   (optional)
    python  benchmarks/verify_equivalence.py
"""
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

from torc.theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                          dda, ddcp, _TERM_LNA, Z_SPLIT, N_EARLY, N_LATE,
                          OMEGA_R_DEFAULT)

XSPLIT = -np.log(1.0 + Z_SPLIT)
POINTS = {
    "fiducial":    dict(omega_m=0.30,   stigma=1.50,   mups2=1.50,   OmL=0.70),
    "slow ridge":  dict(omega_m=0.9555, stigma=1.1926, mups2=0.6794, OmL=0.4868),
    "stigma low":  dict(omega_m=0.50,   stigma=1.0001, mups2=1.50,   OmL=0.70),
    "stigma high": dict(omega_m=0.20,   stigma=2.90,   mups2=2.50,   OmL=1.20),
}
TOL = 1e-12


def setup(p):
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
    cp, da, dcp = y
    om, omr, ol, st, mu = args
    a = np.exp(x)
    fac = a / da
    return [dcp * fac,
            dda(a, da, cp, dcp, om, omr, ol, st, mu) * fac,
            ddcp(a, da, cp, dcp, om, omr, ol, st, mu) * fac]


def norm_E(da_vals, xs):
    E = np.asarray(da_vals) * np.exp(-2.0 * xs)
    return E / E[-1]


def E_scipy(x0, y0, args, xs, tol, method="DOP853"):
    s = solve_ivp(rhs_np, [x0, 0.0], list(y0), t_eval=xs, args=(args,),
                  method=method, rtol=tol, atol=tol)
    return norm_E(s.y[1], xs) if s.success else None


def E_diffrax(x0, y0, args, xs, tol):
    sol = diffrax.diffeqsolve(
        _TERM_LNA, diffrax.Dopri8(), t0=x0, t1=0.0, dt0=None, y0=y0, args=args,
        stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol),
        saveat=diffrax.SaveAt(ts=jnp.asarray(xs)), max_steps=1_000_000,
        throw=False, adjoint=diffrax.ForwardMode())
    if not bool(sol.result == diffrax.RESULTS.successful):
        return None
    return norm_E(np.asarray(sol.ys[1]), xs)


def E_cvode(x0, y0, args, xs, tol):
    try:
        from scikits.odes import ode
    except Exception:
        return "unavailable"

    def cv_rhs(x, y, ydot):
        cp, da, dcp = y
        om, omr, ol, st, mu = args
        a = np.exp(x)
        fac = a / da
        ydot[0] = dcp * fac
        ydot[1] = dda(a, da, cp, dcp, om, omr, ol, st, mu) * fac
        ydot[2] = ddcp(a, da, cp, dcp, om, omr, ol, st, mu) * fac

    try:
        out = ode("cvode", cv_rhs, lmm_type="ADAMS", rtol=tol, atol=tol,
                  old_api=False).solve(xs, np.asarray(y0))
        if out.flag != 0:
            return None
        return norm_E(np.asarray(out.values.y)[:, 1], xs)
    except Exception:
        return None


def E_julia(name, xs):
    f = os.path.join(HERE, "julia_E.tsv")
    if not os.path.exists(f):
        return "unavailable"
    rows = [l.split("\t") for l in open(f) if l.startswith(name + "\t")]
    if not rows:
        return None
    x = np.array([float(r[1]) for r in rows])
    E = np.array([float(r[2]) for r in rows])
    if len(x) != len(xs) or not np.allclose(x, xs, rtol=0, atol=1e-9):
        return np.interp(xs, x[::-1], E[::-1]) if x[0] > x[-1] else np.interp(xs, x, E)
    return E


def main():
    print(f"Cross-backend equivalence check (all solvers at rtol=atol={TOL:.0e})")
    print("reference: scipy DOP853 @ 1e-13.\n")
    print("NOTE: stigma=1.0001 sits immediately beside the stigma=1 coordinate")
    print("singularity, where the REFERENCE itself fails to converge.  For such")
    print("cases the tolerance below is relaxed and the result is reported as")
    print("'reference unconverged' rather than blamed on the backends.\n")
    print(f"{'case':13s} {'backend':16s} {'max |dE/E| vs reference':>25s}")
    worst = {}
    for name, p in POINTS.items():
        x0, y0, args, xs = setup(p)
        ref = E_scipy(x0, y0, args, xs, 1e-13)
        # is the reference itself converged?  compare against a looser one:
        # if they already disagree, no backend can be judged against it.
        ref2 = E_scipy(x0, y0, args, xs, 1e-11)
        ref_err = float(np.nanmax(np.abs((ref2 - ref) / ref))) \
            if ref is not None and ref2 is not None else np.inf
        converged = ref_err < 1e-9
        thresh = 1e-8 if converged else max(1e-8, 100 * ref_err)
        if not converged:
            print(f"{name:13s} *** REFERENCE UNCONVERGED "
                  f"(self-consistency {ref_err:.1e}); threshold relaxed to "
                  f"{thresh:.1e} ***")
        results = {
            "diffrax Dopri8": E_diffrax(x0, y0, args, xs, TOL),
            "scipy RK45":     E_scipy(x0, y0, args, xs, TOL, "RK45"),
            "scipy Radau":    E_scipy(x0, y0, args, xs, TOL, "Radau"),
            "SUNDIALS CVODE": E_cvode(x0, y0, args, xs, TOL),
            "Julia Vern9":    E_julia(name, xs),
        }
        for bname, E in results.items():
            if isinstance(E, str):
                print(f"{name:13s} {bname:16s} {E:>25s}")
                continue
            if E is None:
                print(f"{name:13s} {bname:16s} {'FAILED':>25s}")
                continue
            d = float(np.nanmax(np.abs((E - ref) / ref)))
            if converged:
                worst[bname] = max(worst.get(bname, 0.0), d)
            flag = "" if d < thresh else "   <-- CHECK"
            print(f"{name:13s} {bname:16s} {d:25.3e}{flag}")
        print()

    print("worst case per backend (cases with a CONVERGED reference only):")
    for b, d in sorted(worst.items(), key=lambda kv: kv[1]):
        verdict = "same equations" if d < 1e-8 else "DISCREPANCY"
        print(f"  {b:16s} {d:10.3e}   {verdict}")


if __name__ == "__main__":
    main()
