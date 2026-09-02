"""Benchmark ODE solvers on the TorC background system.

Two families:
  A. diffrax (JAX-native)  -- these can DROP IN to the NSS pipeline
  B. scipy solve_ivp       -- reference only; NOT jit/vmap-able, so adopting
                              one means leaving JAX. Included to answer
                              "would a stiff solver (Radau/BDF) do better?"

Metrics: wall time per solve, work done (steps / RHS evals), and accuracy of
E(z) against a high-precision reference. A fast-but-wrong solver is useless,
so accuracy is reported alongside speed.
"""
import sys, os, time, warnings
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import diffrax
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                          dda, ddcp, _TERM_LNA, Z_SPLIT, N_EARLY, N_LATE,
                          OMEGA_R_DEFAULT)
warnings.filterwarnings("ignore")

# --- test points: fiducial, the slow ridge, and two others ---
POINTS = {
    "fiducial":   dict(omega_m=0.30, stigma=1.50, mups2=1.50, OmL=0.70),
    "slow ridge": dict(omega_m=0.9555, stigma=1.1926, mups2=0.6794, OmL=0.4868),
    "stigma low": dict(omega_m=0.50, stigma=1.0001, mups2=1.50, OmL=0.70),
    "stigma high": dict(omega_m=0.20, stigma=2.90, mups2=2.50, OmL=1.20),
}


def setup(p):
    om, st, mu, ol = p["omega_m"], p["stigma"], p["mups2"], p["OmL"]
    omr = OMEGA_R_DEFAULT
    tau = small(om, omr, st, mu)
    x0 = float(jnp.log(aseries(tau, om, omr, ol, st, mu)))
    y0 = (float(cpseries(tau, om, omr, ol, st, mu)),
          float(aseriesp(tau, om, omr, ol, st, mu)),
          float(cpseriesp(tau, om, omr, ol, st, mu)))
    args = (om, omr, ol, st, mu)
    xsplit = -np.log(1.0 + Z_SPLIT)
    xs = np.concatenate([np.linspace(x0, xsplit, N_EARLY, endpoint=False),
                          np.linspace(xsplit, 0.0, N_LATE)])
    return x0, y0, args, xs


def rhs_np(x, y, args):
    """Pure-numpy RHS: dda/ddcp are plain arithmetic, so no JAX dispatch
    overhead -- this makes the scipy timings a fair comparison."""
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


# --- reference: scipy DOP853 at 1e-13 ---
def reference(p):
    x0, y0, args, xs = setup(p)
    s = solve_ivp(rhs_np, [x0, 0.0], y0, t_eval=xs, args=(args,),
                  method="DOP853", rtol=1e-13, atol=1e-13)
    return E_from(s.y[1], xs)


DIFFRAX_SOLVERS = {
    "Tsit5 (current)": diffrax.Tsit5(), "Dopri5": diffrax.Dopri5(),
    "Dopri8": diffrax.Dopri8(), "Bosh3": diffrax.Bosh3(),
    "Heun": diffrax.Heun(), "Ralston": diffrax.Ralston(),
    "Kvaerno3 [stiff]": diffrax.Kvaerno3(), "Kvaerno5 [stiff]": diffrax.Kvaerno5(),
    "KenCarp4 [stiff]": diffrax.KenCarp4(),
}
SCIPY_METHODS = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
TOLS = [1e-5, 1e-8]
REPS = 7


def bench_diffrax(name, solver, p, tol):
    x0, y0, args, xs = setup(p)
    xs_j = jnp.asarray(xs)
    ctrl = diffrax.PIDController(rtol=tol, atol=tol)
    root = diffrax.VeryChord(rtol=tol, atol=tol)

    def run():
        kw = {}
        if isinstance(solver, (diffrax.Kvaerno3, diffrax.Kvaerno5, diffrax.KenCarp4)):
            kw = {}
        return diffrax.diffeqsolve(
            _TERM_LNA, solver, t0=x0, t1=0.0, dt0=None, y0=y0, args=args,
            stepsize_controller=ctrl, saveat=diffrax.SaveAt(ts=xs_j),
            max_steps=100000, throw=False, adjoint=diffrax.ForwardMode(), **kw)

    f = jax.jit(run)
    try:
        sol = jax.block_until_ready(f())
    except Exception as e:
        return None, None, None, f"compile fail: {type(e).__name__}"
    if not bool(sol.result == diffrax.RESULTS.successful):
        return None, None, None, f"FAILED ({sol.result})"
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); jax.block_until_ready(f()); ts.append(time.perf_counter()-t0)
    E = np.asarray(sol.ys[1]) * np.exp(-2.0*xs); E = E/E[-1]
    return np.median(ts)*1e3, int(sol.stats["num_steps"]), E, None


def bench_scipy(method, p, tol):
    x0, y0, args, xs = setup(p)
    try:
        s = solve_ivp(rhs_np, [x0, 0.0], y0, t_eval=xs, args=(args,),
                      method=method, rtol=tol, atol=tol)
    except Exception as e:
        return None, None, None, f"{type(e).__name__}"
    if not s.success:
        return None, None, None, "FAILED"
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        solve_ivp(rhs_np, [x0, 0.0], y0, t_eval=xs, args=(args,),
                  method=method, rtol=tol, atol=tol)
        ts.append(time.perf_counter()-t0)
    return np.median(ts)*1e3, s.nfev, E_from(s.y[1], xs), None


for pname, p in POINTS.items():
    ref = reference(p)
    print("\n" + "=" * 86)
    print(f"POINT: {pname}   {p}")
    print("=" * 86)
    print(f"{'solver':22s} {'tol':>6s} {'ms':>9s} {'steps/nfev':>11s} {'max|dE/E|':>11s}  note")
    for tol in TOLS:
        for name, sv in DIFFRAX_SOLVERS.items():
            ms, st, E, err = bench_diffrax(name, sv, p, tol)
            if err:
                print(f"{'  dfx '+name:22s} {tol:6.0e} {'-':>9s} {'-':>11s} {'-':>11s}  {err}")
            else:
                rel = np.nanmax(np.abs(E - ref) / ref)
                print(f"{'  dfx '+name:22s} {tol:6.0e} {ms:9.3f} {st:11d} {rel:11.2e}")
        for m in SCIPY_METHODS:
            ms, nf, E, err = bench_scipy(m, p, tol)
            if err:
                print(f"{'  spy '+m:22s} {tol:6.0e} {'-':>9s} {'-':>11s} {'-':>11s}  {err}")
            else:
                rel = np.nanmax(np.abs(E - ref) / ref)
                print(f"{'  spy '+m:22s} {tol:6.0e} {ms:9.3f} {nf:11d} {rel:11.2e}")
    sys.stdout.flush()
print("\nBENCHMARK DONE")
