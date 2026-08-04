"""Do ALL the ODE solvers produce the same solution?

plot_backend_comparison.py shows six representative solvers.  This shows
every one available across all four backends -- ~20 curves -- at the _SAFE
point, all requesting the same tolerance.

Top panel: the comoving Hubble horizon H0/(aH) = 1/(aE).  Expected to look
like a single line; that is the point.

Bottom panel: residual against a cross-validated reference.  This is where
disagreement would actually show, and where the solvers separate into those
that honour the requested tolerance and those that do not.

Run:  python benchmarks/plot_backend_comparison_all.py [--tol 1e-10] [--no-julia]
"""
import argparse
import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

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

DIFFRAX = {
    "Tsit5": diffrax.Tsit5(), "Dopri5": diffrax.Dopri5(),
    "Dopri8": diffrax.Dopri8(), "Bosh3": diffrax.Bosh3(),
    "Heun": diffrax.Heun(), "Ralston": diffrax.Ralston(),
    "Midpoint": diffrax.Midpoint(),
}
IMPLICIT = {"Kvaerno3 [stiff]": diffrax.Kvaerno3,
            "Kvaerno5 [stiff]": diffrax.Kvaerno5}

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

SCIPY = ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
CVODE = {"CVODE Adams": "ADAMS", "CVODE BDF": "BDF"}

# one colour family per backend so ~20 curves stay readable
FAMILY = {"diffrax": cm.Purples, "scipy": cm.RdPu,
          "cvode": cm.Greens, "julia": cm.Oranges}

plt.rcParams.update({
    "text.usetex": True, "font.family": "STIXGeneral",
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 11,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 6,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--no-julia", action="store_true")
    a = ap.parse_args()
    TOL = a.tol

    x0, y0, args, xs = setup()
    xs_j = jnp.asarray(xs)

    def dfx(solver, tol):
        s = diffrax.diffeqsolve(
            _TERM_LNA, solver, t0=x0, t1=0.0, dt0=None, y0=y0, args=args,
            stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol),
            saveat=diffrax.SaveAt(ts=xs_j), max_steps=10_000_000,
            throw=False, adjoint=diffrax.ForwardMode())
        if not bool(s.result == diffrax.RESULTS.successful):
            return None
        return norm_E(np.asarray(s.ys[1]), xs)

    # ---- cross-validated reference ----------------------------------------
    ref_a = dfx(diffrax.Dopri8(), 1e-13)
    sref = solve_ivp(rhs_np, [x0, 0.0], list(y0), t_eval=xs, args=(args,),
                     method="DOP853", rtol=1e-13, atol=1e-13)
    ref_b = norm_E(sref.y[1], xs) if sref.success else None
    gap = float(np.nanmax(np.abs((ref_b - ref_a) / ref_a)))
    print(f"reference cross-check (diffrax Dopri8 vs scipy DOP853 @1e-13): "
          f"{gap:.2e}")
    REF = ref_a

    results = {}          # (backend, name) -> E

    print(f"\nsolving at the _SAFE point, all requesting rtol=atol={TOL:.0e}")
    for nm, sv in list(DIFFRAX.items()) + \
            [(k, _implicit(v, TOL)) for k, v in IMPLICIT.items()]:
        E = dfx(sv, TOL)
        results[("diffrax", nm)] = E
        print(f"  diffrax {nm:20s} {'FAILED' if E is None else 'ok'}")

    for m in SCIPY:
        s = solve_ivp(rhs_np, [x0, 0.0], list(y0), t_eval=xs, args=(args,),
                      method=m, rtol=TOL, atol=TOL)
        results[("scipy", m)] = norm_E(s.y[1], xs) if s.success else None
        print(f"  scipy   {m:20s} {'ok' if s.success else 'FAILED'}")

    try:
        from scikits.odes import ode

        def cv_rhs(x, y, ydot):
            ydot[:] = rhs_np(x, y, args)

        for nm, lmm in CVODE.items():
            out = ode("cvode", cv_rhs, lmm_type=lmm, rtol=TOL, atol=TOL,
                      old_api=False).solve(xs, np.asarray(y0))
            results[("cvode", nm)] = (norm_E(np.asarray(out.values.y)[:, 1], xs)
                                      if out.flag == 0 else None)
            print(f"  cvode   {nm:20s} {'ok' if out.flag == 0 else 'FAILED'}")
    except Exception as e:
        print(f"  cvode unavailable ({type(e).__name__})")

    if not a.no_julia:
        try:
            import torc.julia_ipc as ji
            ji.start()
            E = ji.solve_batch(np.asarray(args).reshape(1, -1),
                               np.asarray([x0]),
                               np.asarray(y0).reshape(1, -1), TOL)[0]
            results[("julia", "Tsit5 (IPC)")] = E if np.isfinite(E).all() else None
            print(f"  julia   {'Tsit5 (IPC)':20s} ok")
        except Exception as e:
            print(f"  julia unavailable ({type(e).__name__}: {e})")

    ok = {k: v for k, v in results.items() if v is not None}
    print(f"\n{len(ok)}/{len(results)} solvers succeeded")

    # ---- plot --------------------------------------------------------------
    aa = np.exp(xs)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.4, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1.3], "hspace": 0.07})

    counts = {}
    for b, _ in ok:
        counts[b] = counts.get(b, 0) + 1
    idx = {b: 0 for b in counts}

    errs = []
    for (b, nm), E in sorted(ok.items()):
        cmap = FAMILY.get(b, cm.Greys)
        frac = 0.35 + 0.55 * (idx[b] / max(counts[b] - 1, 1))
        idx[b] += 1
        col = cmap(frac)
        ax1.plot(aa, 1.0 / (aa * E), color=col, lw=0.9, alpha=0.8)
        e = np.abs((E - REF) / REF)
        ax2.plot(aa, np.maximum(e, 1e-17), color=col, lw=0.9, alpha=0.85,
                 label=f"{b}: {nm}")
        errs.append((float(np.nanmax(e)), b, nm))

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_ylabel(r"$H_0/(aH(a))$")
    ax1.set_title(r"All ODE solvers at the \texttt{\_SAFE} point "
                  rf"(rtol$=$atol$=10^{{{int(np.log10(TOL))}}}$)"
                  "\n" r"top: every solver overlaid --- they coincide; "
                  r"bottom: residual vs reference", fontsize=9.5)

    ax2.axhline(TOL, color="k", ls="--", lw=0.8)
    ax2.text(aa[2], TOL * 1.5, "requested tolerance", fontsize=6.5)
    ax2.axhspan(1e-17, max(gap, 1e-14), color="grey", alpha=0.15)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"$a$")
    ax2.set_ylabel(r"$|\Delta E/E|$ vs reference")
    ax2.legend(ncol=3, loc="upper left", fontsize=5.8)

    fig.text(0.5, -0.01,
             "Solvers below the dashed line met the tolerance they were "
             "asked for; those above did not.\nGrey band is the reference "
             "floor -- nothing can be resolved below it.",
             ha="center", va="top", fontsize=7)
    out = os.path.join(PLOT_DIR, "backend_comparison_all")
    for ext in ("pdf", "png"):
        fig.savefig(out + "." + ext, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"saved plots/backend_comparison_all.{{pdf,png}}\n")

    print(f"max |dE/E| vs reference (requested {TOL:.0e}):")
    for e, b, nm in sorted(errs):
        flag = "" if e <= TOL else "   <-- did NOT meet tolerance"
        print(f"  {b:8s} {nm:20s} {e:10.2e}{flag}")


if __name__ == "__main__":
    main()
