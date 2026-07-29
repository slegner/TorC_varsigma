"""Do all the ODE backends actually produce the same solution?

Solves the SAME problem at the _SAFE parameter point with every available
backend and overlays the comoving Hubble horizon H0/(aH(a)) = 1/(a E).

The overlay is expected to look like a single curve -- that is the point.
The lower panel shows the residual against a common reference, which is
where any real disagreement would show up.

    diffrax Dopri8   -- vector field imported from torc/theory.py
    diffrax Tsit5    -- same field, different integrator
    scipy DOP853     -- pure-numpy RHS calling the SAME dda/ddcp
    scipy Radau      -- ditto (implicit)
    SUNDIALS CVODE   -- ditto, via scikits.odes   (optional)
    Julia Vern9      -- RHS auto-generated from theory.py; read from
                        julia_E.tsv if benchmark_julia.jl has been run

Run:
    python benchmarks/plot_backend_comparison.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torc.theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                          dda, ddcp, _TERM_LNA, Z_SPLIT, N_EARLY, N_LATE,
                          OMEGA_R_DEFAULT)
from torc.priors import _SAFE

PLOT_DIR = os.path.join(HERE, "..", "plots")
XSPLIT = -np.log(1.0 + Z_SPLIT)
TOL = 1e-10
CASE = "fiducial"          # == the _SAFE point

plt.rcParams.update({
    "text.usetex": True, "font.family": "STIXGeneral",
    "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 11, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.linewidth": 0.8, "lines.linewidth": 1.2,
})


def setup():
    om, st = _SAFE["omega_m"], _SAFE["stigma"]
    mu, ol = _SAFE["mups2"], _SAFE["OmL"]
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


def run_all():
    x0, y0, args, xs = setup()
    out = {}

    for lbl, solver in (("diffrax Dopri8", diffrax.Dopri8()),
                        ("diffrax Tsit5", diffrax.Tsit5())):
        s = diffrax.diffeqsolve(
            _TERM_LNA, solver, t0=x0, t1=0.0, dt0=None, y0=y0, args=args,
            stepsize_controller=diffrax.PIDController(rtol=TOL, atol=TOL),
            saveat=diffrax.SaveAt(ts=jnp.asarray(xs)), max_steps=1_000_000,
            throw=False, adjoint=diffrax.ForwardMode())
        if bool(s.result == diffrax.RESULTS.successful):
            out[lbl] = norm_E(np.asarray(s.ys[1]), xs)

    for lbl, m in (("scipy DOP853", "DOP853"), ("scipy Radau", "Radau")):
        s = solve_ivp(rhs_np, [x0, 0.0], list(y0), t_eval=xs, args=(args,),
                      method=m, rtol=TOL, atol=TOL)
        if s.success:
            out[lbl] = norm_E(s.y[1], xs)

    try:
        from scikits.odes import ode

        def cv_rhs(x, y, ydot):
            cp, da, dcp = y
            om, omr, ol, st, mu = args
            a = np.exp(x); fac = a / da
            ydot[0] = dcp * fac
            ydot[1] = dda(a, da, cp, dcp, om, omr, ol, st, mu) * fac
            ydot[2] = ddcp(a, da, cp, dcp, om, omr, ol, st, mu) * fac

        r = ode("cvode", cv_rhs, lmm_type="ADAMS", rtol=TOL, atol=TOL,
                old_api=False).solve(xs, np.asarray(y0))
        if r.flag == 0:
            out["SUNDIALS CVODE"] = norm_E(np.asarray(r.values.y)[:, 1], xs)
    except Exception as e:
        print(f"  (CVODE unavailable: {type(e).__name__})")

    f = os.path.join(HERE, "julia_E.tsv")
    if os.path.exists(f):
        rows = [l.split("\t") for l in open(f) if l.startswith(CASE + "\t")]
        if rows:
            jx = np.array([float(r[1]) for r in rows])
            jE = np.array([float(r[2]) for r in rows])
            order = np.argsort(jx)
            out["Julia Vern9"] = np.interp(xs, jx[order], jE[order])
    else:
        print("  (Julia unavailable: run benchmarks/benchmark_julia.jl first)")

    return xs, out


def main():
    xs, res = run_all()
    a = np.exp(xs)
    ref_key = "diffrax Dopri8"
    ref = res[ref_key]

    print(f"backends solved at _SAFE: {list(res)}")
    print(f"\nmax |dE/E| vs {ref_key}:")
    for k, E in res.items():
        print(f"  {k:18s} {np.nanmax(np.abs((E - ref) / ref)):.3e}")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.4, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})

    colors = ["#785EF0", "#DC267F", "#FFB000", "#648FFF", "#009E73", "#000000"]
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    for i, (k, E) in enumerate(res.items()):
        ax1.plot(a, 1.0 / (a * E), linestyle=styles[i % len(styles)],
                 color=colors[i % len(colors)], label=k, alpha=0.85)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_ylabel(r"$H_0/(aH(a))$")
    ax1.legend(loc="upper left", ncol=2, frameon=True)
    ax1.set_title(r"Comoving Hubble horizon at the \texttt{\_SAFE} point "
                  r"($\Omega_m{=}0.3,\ \varsigma{=}1.5,\ -\upsilon_2{=}1.5,\ "
                  r"\Omega_\Lambda{=}0.7$)", fontsize=9)

    for i, (k, E) in enumerate(res.items()):
        if k == ref_key:
            continue
        ax2.plot(a, np.abs((E - ref) / ref), linestyle=styles[i % len(styles)],
                 color=colors[i % len(colors)], label=k, alpha=0.85)
    ax2.axhline(TOL, color="grey", lw=0.8, ls=":",
                label=f"solver tolerance ({TOL:.0e})")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"$a$")
    ax2.set_ylabel(r"$|\Delta E / E|$ vs " + ref_key.replace("_", r"\_"))
    ax2.legend(loc="upper left", ncol=2, fontsize=6.5)

    out = os.path.join(PLOT_DIR, "backend_comparison_safe")
    for ext in ("pdf", "png"):
        fig.savefig(out + "." + ext, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"\nsaved plots/backend_comparison_safe.{{pdf,png}}")


if __name__ == "__main__":
    main()
