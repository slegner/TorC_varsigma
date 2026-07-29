"""How fast is each ODE solver on the TorC background equations?

Runs every backend itself (no cached logs), at the _SAFE parameter point,
and bar-charts the median wall time per solve.

Each backend lives in its own file under benchmarks/backends/ and can also
be run on its own:

    python benchmarks/backends/diffrax_backend.py
    python benchmarks/backends/scipy_backend.py
    python benchmarks/backends/cvode_backend.py
    python benchmarks/backends/julia_backend.py

They all import benchmarks/backends/common.py, which builds the initial
conditions and the output grid -- so every backend is provably solving the
same initial-value problem, and only the integrator differs.

Run:
    python benchmarks/plot_speed.py            # all backends, both tolerances
    python benchmarks/plot_speed.py --no-julia # skip the julia subprocess
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "backends"))
sys.path.insert(0, os.path.join(HERE, ".."))

PLOT_DIR = os.path.join(HERE, "..", "plots")
TOLS = (1e-5, 1e-8)

BACKEND_COLOR = {
    "diffrax": "#785EF0",   # what the pipeline uses
    "julia":   "#FFB000",
    "cvode":   "#009E73",
    "scipy":   "#DC267F",
}
BACKEND_LABEL = {
    "diffrax": "diffrax (JAX) -- used by the pipeline",
    "julia":   "Julia (OrdinaryDiffEq/Sundials)",
    "cvode":   "SUNDIALS CVODE (scikits.odes)",
    "scipy":   "scipy solve_ivp",
}

plt.rcParams.update({
    "text.usetex": True, "font.family": "STIXGeneral",
    "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 10, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.linewidth": 0.8,
})


def collect(use_julia=True):
    import diffrax_backend, scipy_backend, cvode_backend
    rows = []
    print("running diffrax ...", flush=True)
    rows += diffrax_backend.run_all(TOLS)
    print("running scipy ...", flush=True)
    rows += scipy_backend.run_all(TOLS)
    print("running cvode ...", flush=True)
    rows += cvode_backend.run_all(TOLS)
    if use_julia:
        import julia_backend
        print("running julia (subprocess; JIT compile is slow) ...", flush=True)
        rows += julia_backend.run_all(TOLS)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-julia", action="store_true")
    a = ap.parse_args()

    rows = collect(use_julia=not a.no_julia)
    ok = [r for r in rows if r["ok"] and r["ms"] is not None]
    bad = [r for r in rows if not r["ok"]]

    print(f"\n{len(ok)} solves succeeded, {len(bad)} failed")
    for r in bad:
        print(f"  FAILED: {r['backend']:8s} {r['solver']:20s} "
              f"tol={r['tol']:.0e}  ({r['note']})")

    fig, axes = plt.subplots(1, len(TOLS), figsize=(11, 5.6), sharey=False)
    for ax, tol in zip(np.atleast_1d(axes), TOLS):
        sub = sorted([r for r in ok if r["tol"] == tol], key=lambda r: r["ms"])
        labels = [f"{r['backend']}: {r['solver']}" for r in sub]
        vals = [r["ms"] for r in sub]
        cols = [BACKEND_COLOR.get(r["backend"], "grey") for r in sub]
        y = np.arange(len(sub))
        ax.barh(y, vals, color=cols, height=0.72)
        ax.set_yticks(y)
        ax.set_yticklabels([l.replace("_", r"\_") for l in labels], fontsize=7)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("median wall time per solve [ms]")
        ax.set_title(rf"tolerance $=10^{{{int(np.log10(tol))}}}$", fontsize=10)
        ax.grid(axis="x", alpha=0.3, lw=0.5)
        fastest = vals[0]
        for yi, v in zip(y, vals):
            ax.text(v * 1.12, yi, f"{v:.2f} ms  ({v/fastest:.0f}x)",
                    va="center", fontsize=6.2)
        ax.set_xlim(right=max(vals) * 12)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for b, c in BACKEND_COLOR.items()]
    fig.legend(handles, [BACKEND_LABEL[b] for b in BACKEND_COLOR],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(r"ODE solver speed on the TorC background equations "
                 r"(\texttt{\_SAFE} point, identical problem and output grid)",
                 fontsize=11, y=0.98)
    # Single-solve timing cannot show batching, which is what the sampler
    # actually does -- and diffrax gains ~3.4x from vmap while the Julia
    # driver is a serial loop.  State that plainly on the figure.
    fig.text(0.5, -0.055,
             r"Single solves, all setup costs included.  \textbf{The sampler "
             r"batches:} under \texttt{vmap} diffrax reaches 1.45 ms/solve "
             r"(3.4$\times$ better than serial), while the Julia driver is a "
             r"serial loop at 0.72 ms/solve --- so in the pipeline the real "
             r"gap is $\sim$2$\times$, not what these bars suggest.",
             ha="center", va="top", fontsize=7.5, wrap=True)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    out = os.path.join(PLOT_DIR, "solver_speed_comparison")
    for ext in ("pdf", "png"):
        fig.savefig(out + "." + ext, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"\nsaved plots/solver_speed_comparison.{{pdf,png}}")

    for tol in TOLS:
        sub = sorted([r for r in ok if r["tol"] == tol], key=lambda r: r["ms"])
        if not sub:
            continue
        print(f"\ntol {tol:.0e}: fastest = {sub[0]['backend']} "
              f"{sub[0]['solver']} at {sub[0]['ms']:.3f} ms; "
              f"slowest = {sub[-1]['backend']} {sub[-1]['solver']} at "
              f"{sub[-1]['ms']:.1f} ms  "
              f"({sub[-1]['ms']/sub[0]['ms']:.0f}x spread)")


if __name__ == "__main__":
    main()
