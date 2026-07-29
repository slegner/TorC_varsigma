"""SUNDIALS CVODE (via scikits.odes) on the TorC background ODE.

Uses the SAME pure-numpy RHS as the scipy benchmark (dda/ddcp imported from
torc/theory.py), so the only thing that differs is the integrator.
CVODE is a compiled C solver, so unlike scipy it does not pay a Python
callback per step... except that the RHS itself is still Python here, which
is the honest limitation to note.
"""
import sys, os, time
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.theory import dda, ddcp, Z_SPLIT, N_EARLY, N_LATE, OMEGA_R_DEFAULT
from scikits.odes import ode

XSPLIT = -np.log(1.0 + Z_SPLIT)
POINTS = {
    "fiducial":    dict(omega_m=0.30,   stigma=1.50,   mups2=1.50,   OmL=0.70),
    "slow ridge":  dict(omega_m=0.9555, stigma=1.1926, mups2=0.6794, OmL=0.4868),
    "stigma low":  dict(omega_m=0.50,   stigma=1.0001, mups2=1.50,   OmL=0.70),
    "stigma high": dict(omega_m=0.20,   stigma=2.90,   mups2=2.50,   OmL=1.20),
}

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from torc.theory import aseries, cpseries, aseriesp, cpseriesp, small


def setup(p):
    om, st, mu, ol = p["omega_m"], p["stigma"], p["mups2"], p["OmL"]
    omr = OMEGA_R_DEFAULT
    tau = small(om, omr, st, mu)
    x0 = float(jnp.log(aseries(tau, om, omr, ol, st, mu)))
    y0 = np.array([float(cpseries(tau, om, omr, ol, st, mu)),
                   float(aseriesp(tau, om, omr, ol, st, mu)),
                   float(cpseriesp(tau, om, omr, ol, st, mu))])
    args = (om, omr, ol, st, mu)
    xs = np.concatenate([np.linspace(x0, XSPLIT, N_EARLY, endpoint=False),
                         np.linspace(XSPLIT, 0.0, N_LATE)])
    return x0, y0, args, xs


def rhs_np(x, y, args):
    cp, da, dcp = y
    om, omr, ol, st, mu = args
    a = np.exp(x)
    fac = a / da
    return [dcp * fac,
            dda(a, da, cp, dcp, om, omr, ol, st, mu) * fac,
            ddcp(a, da, cp, dcp, om, omr, ol, st, mu) * fac]


def E_from(da_vals, xs):
    E = da_vals * np.exp(-2.0 * xs)
    return E / E[-1]


REPS = 5
print(f"{'case':13s} {'solver':22s} {'tol':>6s} {'ms':>9s} {'nfev':>8s} {'maxrelerr':>11s}")
for pname, p in POINTS.items():
    x0, y0, args, xs = setup(p)
    ref = solve_ivp(rhs_np, [x0, 0.0], y0, t_eval=xs, args=(args,),
                    method="DOP853", rtol=1e-13, atol=1e-13)
    Eref = E_from(ref.y[1], xs)

    def cv_rhs(x, y, ydot):
        cp, da, dcp = y
        om, omr, ol, st, mu = args
        a = np.exp(x)
        fac = a / da
        ydot[0] = dcp * fac
        ydot[1] = dda(a, da, cp, dcp, om, omr, ol, st, mu) * fac
        ydot[2] = ddcp(a, da, cp, dcp, om, omr, ol, st, mu) * fac

    for tol in (1e-5, 1e-8):
        for label, lmm in (("cvode ADAMS", "adams"), ("cvode BDF [stiff]", "bdf")):
            try:
                def run():
                    s = ode("cvode", cv_rhs, lmm_type=lmm.upper(),
                            rtol=tol, atol=tol, old_api=False)
                    return s.solve(xs, y0)
                out = run()
                if not out.flag == 0:
                    print(f"{pname:13s} {label:22s} {tol:6.0e} {'-':>9s} {'-':>8s}  "
                          f"FAILED(flag={out.flag})")
                    continue
                ts = []
                for _ in range(REPS):
                    t0 = time.perf_counter(); run(); ts.append(time.perf_counter() - t0)
                ms = np.median(ts) * 1e3
                E = E_from(np.asarray(out.values.y)[:, 1], xs)
                err = np.nanmax(np.abs((E - Eref) / Eref))
                print(f"{pname:13s} {label:22s} {tol:6.0e} {ms:9.3f} {'n/a':>8s} {err:11.3e}")
            except Exception as e:
                print(f"{pname:13s} {label:22s} {tol:6.0e} {'-':>9s} {'-':>8s}  "
                      f"ERROR: {type(e).__name__}: {str(e)[:80]}")
    sys.stdout.flush()
print("CVODE BENCHMARK DONE")
