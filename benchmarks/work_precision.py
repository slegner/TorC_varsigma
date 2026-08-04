"""Work-precision diagram: runtime vs ACHIEVED accuracy.

WHY THIS REPLACES THE BAR CHART
-------------------------------
plot_speed.py compares runtime at a REQUESTED tolerance.  That is not a fair
comparison: rtol/atol control LOCAL error, every library estimates it with a
different norm and error estimator, and what we care about is GLOBAL error in
E(z).  Two solvers asked for 1e-8 routinely deliver global errors differing by
an order of magnitude, so "fastest at rtol=1e-8" can just mean "loosest error
estimator at rtol=1e-8".

The standard fix (Hairer & Wanner) is a work-precision diagram: run each
solver over a RANGE of tolerances, measure the error actually achieved against
a high-accuracy reference, and plot runtime against that.  A solver is better
if its curve lies lower and to the left.  Reading horizontally answers the
question that matters: at the accuracy I need, which is fastest?

REFERENCE
---------
Computed by two independent solvers (diffrax Dopri8 and scipy DOP853) at
rtol=atol=1e-13 and cross-checked; if they disagree the reference is not
trustworthy and the script says so rather than quoting errors against it.

TIMING HYGIENE (same as backends/common.py)
-------------------------------------------
  * the whole diffrax solve sits inside ONE outer jax.jit
  * compiled once per (solver, tol), reused for every repetition
  * warmed up before the timing loop
  * jax.block_until_ready() inside the timed region
  * identical output grid (500 points), rtol == atol, no events, for every
    backend

Run:  python benchmarks/work_precision.py [--no-julia]
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "backends"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

from common import setup, rhs_np, norm_E
from torc.theory import _TERM_LNA

PLOT_DIR = os.path.join(HERE, "..", "plots")
TOLS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12]
REPS = 7

# diffrax's implicit solvers default to VeryChord with tolerances
# "taken from the stepsize_controller".  That inheritance misbehaves on this
# RHS: Kvaerno5 then needs 13376 steps and ~1022 ms, versus 61 steps and
# ~7 ms when the root-finder tolerances are given EXPLICITLY -- a 128x
# benchmarking artifact, not a property of implicit methods.  Raising
# root_find_max_steps (10 -> 200) changes nothing; only explicit tolerances
# fix it.  Always configure the root finder before benchmarking these.
def _implicit(cls, tol):
    import optimistix as optx
    return cls(root_finder=optx.Newton(rtol=tol, atol=tol),
               root_find_max_steps=100)


DIFFRAX_EXPLICIT = {"Tsit5": diffrax.Tsit5, "Dopri5": diffrax.Dopri5,
                    "Dopri8": diffrax.Dopri8, "Bosh3": diffrax.Bosh3}
DIFFRAX_IMPLICIT = {"Kvaerno5 [stiff]": diffrax.Kvaerno5,
                    "Kvaerno3 [stiff]": diffrax.Kvaerno3}
DIFFRAX = {**DIFFRAX_EXPLICIT, **DIFFRAX_IMPLICIT}


def make_solver(name, tol):
    return (_implicit(DIFFRAX_IMPLICIT[name], tol)
            if name in DIFFRAX_IMPLICIT else DIFFRAX_EXPLICIT[name]())
SCIPY = ["RK45", "DOP853", "Radau", "LSODA"]

STYLE = {
    "diffrax": dict(color="#785EF0", marker="o"),
    "julia":   dict(color="#FFB000", marker="s"),
    "cvode":   dict(color="#009E73", marker="^"),
    "scipy":   dict(color="#DC267F", marker="v"),
}


def timeit(fn, reps=REPS):
    fn()                                    # warm up / compile
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def err_vs(E, ref):
    m = np.isfinite(E) & np.isfinite(ref)
    if not m.any():
        return np.nan
    return float(np.nanmax(np.abs((E[m] - ref[m]) / ref[m])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-julia", action="store_true")
    a = ap.parse_args()

    x0, y0, args, xs = setup()          # the _SAFE point
    xs_j = jnp.asarray(xs)

    # ---- reference, cross-validated between two independent solvers --------
    def dfx_E(solver, tol):
        f = jax.jit(lambda: diffrax.diffeqsolve(
            _TERM_LNA, solver, t0=x0, t1=0.0, dt0=None, y0=y0, args=args,
            stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol),
            saveat=diffrax.SaveAt(ts=xs_j), max_steps=10_000_000,
            throw=False, adjoint=diffrax.ForwardMode()))
        s = jax.block_until_ready(f())
        if not bool(s.result == diffrax.RESULTS.successful):
            return None, None
        return norm_E(np.asarray(s.ys[1]), xs), f

    ref_a, _ = dfx_E(diffrax.Dopri8(), 1e-13)
    sref = solve_ivp(rhs_np, [x0, 0.0], list(y0), t_eval=xs, args=(args,),
                     method="DOP853", rtol=1e-13, atol=1e-13)
    ref_b = norm_E(sref.y[1], xs) if sref.success else None
    if ref_a is None or ref_b is None:
        print("reference solve FAILED -- cannot proceed"); return
    ref_gap = err_vs(ref_b, ref_a)
    print(f"reference cross-check (diffrax Dopri8 vs scipy DOP853 @1e-13): "
          f"{ref_gap:.2e}")
    if ref_gap > 1e-10:
        print("  WARNING: references disagree; errors below are only "
              "meaningful above this floor")
    REF = ref_a
    floor = max(ref_gap, 1e-14)

    rows = []

    # ---- diffrax -----------------------------------------------------------
    for name in DIFFRAX:
        for tol in TOLS:
            E, f = dfx_E(make_solver(name, tol), tol)
            if E is None:
                print(f"  diffrax {name:18s} {tol:.0e}  FAILED"); continue
            ms = timeit(lambda: jax.block_until_ready(f()))
            e = err_vs(E, REF)
            rows.append(("diffrax", name, tol, ms, e))
            print(f"  diffrax {name:18s} {tol:.0e}  {ms:8.3f} ms  err {e:.2e}",
                  flush=True)

    # ---- scipy -------------------------------------------------------------
    for m in SCIPY:
        for tol in TOLS:
            run = lambda: solve_ivp(rhs_np, [x0, 0.0], list(y0), t_eval=xs,
                                    args=(args,), method=m, rtol=tol, atol=tol)
            s = run()
            if not s.success:
                print(f"  scipy   {m:18s} {tol:.0e}  FAILED"); continue
            ms = timeit(run, reps=3)
            e = err_vs(norm_E(s.y[1], xs), REF)
            rows.append(("scipy", m, tol, ms, e))
            print(f"  scipy   {m:18s} {tol:.0e}  {ms:8.3f} ms  err {e:.2e}",
                  flush=True)

    # ---- SUNDIALS CVODE ----------------------------------------------------
    try:
        from scikits.odes import ode

        def cv_rhs(x, y, ydot):
            ydot[:] = rhs_np(x, y, args)

        for lmm in ("ADAMS", "BDF"):
            for tol in TOLS:
                run = lambda: ode("cvode", cv_rhs, lmm_type=lmm, rtol=tol,
                                  atol=tol, old_api=False).solve(xs,
                                                                 np.asarray(y0))
                out = run()
                if out.flag != 0:
                    print(f"  cvode   {lmm:18s} {tol:.0e}  FAILED"); continue
                ms = timeit(run, reps=3)
                e = err_vs(norm_E(np.asarray(out.values.y)[:, 1], xs), REF)
                rows.append(("cvode", f"CVODE {lmm}", tol, ms, e))
                print(f"  cvode   {lmm:18s} {tol:.0e}  {ms:8.3f} ms  "
                      f"err {e:.2e}", flush=True)
    except Exception as ex:
        print(f"  cvode unavailable: {type(ex).__name__}")

    # ---- Julia, through the IPC server (same one the pipeline uses) --------
    if not a.no_julia:
        try:
            import torc.julia_ipc as ji
            ji.start()
            P = np.asarray(args).reshape(1, -1)
            X0 = np.asarray([x0])
            Y0 = np.asarray(y0).reshape(1, -1)
            for tol in TOLS:
                run = lambda: ji.solve_batch(P, X0, Y0, tol)
                E = run()[0]
                if not np.isfinite(E).all():
                    print(f"  julia   {'Tsit5':18s} {tol:.0e}  FAILED"); continue
                ms = timeit(run, reps=REPS)
                e = err_vs(E, REF)
                rows.append(("julia", "Tsit5", tol, ms, e))
                print(f"  julia   {'Tsit5':18s} {tol:.0e}  {ms:8.3f} ms  "
                      f"err {e:.2e}  (incl. IPC round-trip)", flush=True)
        except Exception as ex:
            print(f"  julia unavailable: {type(ex).__name__}: {ex}")

    # ---- plot --------------------------------------------------------------
    plt.rcParams.update({
        "text.usetex": True, "font.family": "STIXGeneral",
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 11,
        "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    keys = sorted({(b, s) for b, s, *_ in rows})
    for b, s in keys:
        pts = sorted([(e, ms) for bb, ss, t, ms, e in rows
                      if bb == b and ss == s and np.isfinite(e) and e > 0])
        if not pts:
            continue
        e, ms = zip(*pts)
        st = dict(STYLE.get(b, {}))
        ax.plot(e, ms, marker=st.pop("marker", "o"), ms=4, lw=1.1,
                label=f"{b}: {s}", alpha=0.85, **st)
    ax.axvspan(1e-18, floor, color="grey", alpha=0.15)
    ax.text(floor, ax.get_ylim()[1], "  reference floor", fontsize=6.5,
            va="top", color="grey")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(r"achieved accuracy: $\max |\Delta E/E|$ vs reference "
                  r"(better $\rightarrow$)")
    ax.set_ylabel("median wall time per solve [ms]")
    ax.set_title(r"Work--precision: runtime at \emph{equal achieved accuracy}"
                 "\n" r"TorC background ODE at the \texttt{\_SAFE} point, "
                 r"tolerances $10^{-4}$ to $10^{-12}$", fontsize=10)
    ax.grid(alpha=0.3, lw=0.5, which="both")
    ax.legend(ncol=2, loc="upper left")
    fig.text(0.5, -0.02,
             "Lower-left is better.  Read HORIZONTALLY: at the accuracy you "
             "need, which solver is fastest?\nEqual requested tolerance is "
             "NOT equal accuracy -- that is exactly what this diagram "
             "corrects for.", ha="center", va="top", fontsize=7)
    fig.tight_layout()
    out = os.path.join(PLOT_DIR, "work_precision")
    for ext in ("pdf", "png"):
        fig.savefig(out + "." + ext, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"\nsaved plots/work_precision.{{pdf,png}}")

    # ---- the answer the diagram is for -------------------------------------
    print("\nfastest solver at each achieved-accuracy target:")
    for target in (1e-4, 1e-6, 1e-8, 1e-10):
        ok = [(ms, b, s, t, e) for b, s, t, ms, e in rows
              if np.isfinite(e) and e <= target]
        if ok:
            ms, b, s, t, e = min(ok)
            print(f"  err <= {target:.0e} : {b:8s} {s:18s} "
                  f"{ms:8.3f} ms  (asked {t:.0e}, got {e:.1e})")
        else:
            print(f"  err <= {target:.0e} : none reached this accuracy")


if __name__ == "__main__":
    main()
