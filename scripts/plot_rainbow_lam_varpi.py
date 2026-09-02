"""Rainbow: varpi(a) for the LAMBDA model at Omega_Lambda = 0, varying varsigma.

Companion to scripts/plot_rainbow_field.py (same colour map, layout and
noH0 anchoring), but the background is torc.theory_lam with OmL pinned to 0,
so lambda alone sources the late-time evolution.

    python scripts/plot_rainbow_lam_varpi.py [--lam 1.0]

Left  : varpi(a) coloured by varsigma
Right : the lambda source term along each trajectory, normalised to its
        early-time value -- every lambda term carries varpi^2 or varpi^4
        (unlike Omega_Lambda, which has a varpi-independent floor), so the
        source decays exactly as varpi does.  That is why lambda cannot
        replace the fluid; see scripts/plot_why_lam_fails.py.

The varsigma range covers the DESY5 posterior (1.73 +- 0.32) and the
BAO/Pantheon+ posteriors (~1.3).
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

from torc.theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                         N_EARLY, N_LATE, Z_SPLIT, OMEGA_R_DEFAULT,
                         _SOLVER, _CTRL)
from torc import theory_lam as TL

OMR = OMEGA_R_DEFAULT
OMEGA_M = 0.3
MUPS2 = 4.0 / 3.0
OML = 0.0                     # the point of this figure

plt.rcParams.update({
    "text.usetex": True, "font.family": "cmtt", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "lines.linewidth": 1.0,
})
CMAP_COLORS = ["#785EF0", "#DC267F", "#FFB000"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_cmap", CMAP_COLORS, N=256)


def run(stigma, lam, OmL=OML):
    """noH0 solve of the lambda model; returns a, varpi."""
    tau = small(OMEGA_M, OMR, stigma, MUPS2)
    x0 = float(jnp.log(aseries(tau, OMEGA_M, OMR, OmL, stigma, MUPS2)))
    if not np.isfinite(x0):
        raise ValueError("non-finite initial condition")
    y0 = (cpseries(tau, OMEGA_M, OMR, OmL, stigma, MUPS2),
          aseriesp(tau, OMEGA_M, OMR, OmL, stigma, MUPS2),
          cpseriesp(tau, OMEGA_M, OMR, OmL, stigma, MUPS2))
    xs = jnp.concatenate([
        jnp.linspace(x0, -np.log(1 + Z_SPLIT), N_EARLY, endpoint=False),
        jnp.linspace(-np.log(1 + Z_SPLIT), 0.0, N_LATE)])
    sol = diffrax.diffeqsolve(
        TL._TERM_LNA, _SOLVER, t0=x0, t1=0.0, dt0=None, y0=y0,
        args=(OMEGA_M, OMR, OmL, stigma, MUPS2, lam),
        stepsize_controller=_CTRL, saveat=diffrax.SaveAt(ts=xs),
        max_steps=8192, throw=False, adjoint=diffrax.ForwardMode())
    a = np.exp(np.asarray(xs))
    varpi = np.asarray(sol.ys[0])
    if not np.all(np.isfinite(varpi)):
        raise ValueError("solve failed")
    return a, varpi


def lam_source(cp, stigma, lam):
    """The lambda part of the a^4 source bracket (see plot_why_lam_fails.py)."""
    return cp**4*lam*(-1 + 4*stigma**2) + cp**2*stigma*(-7*lam + 4*lam*stigma)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--amin", type=float, default=None,
                    help="left x limit; default is the earliest solved a "
                         "(~1.3e-9, where the series initial conditions start)")
    args = ap.parse_args()

    stigmas = np.linspace(1.05, 2.60, args.n)
    norm = Normalize(vmin=stigmas.min(), vmax=stigmas.max())
    sm = ScalarMappable(cmap=CUSTOM_CMAP, norm=norm)
    sm.set_array([])

    runs = []
    for st in stigmas:
        try:
            a, varpi = run(st, args.lam)
            runs.append((st, a, varpi))
            print(f"  stigma={st:.3f}: varpi {varpi[0]:.4f} -> {varpi[-1]:.4f}",
                  flush=True)
        except Exception as e:
            print(f"  stigma={st:.3f}: SKIPPED ({e})", flush=True)
    if not runs:
        raise SystemExit("no successful solves")

    a_start = min(a[0] for _, a, _ in runs)
    amin = args.amin if args.amin is not None else a_start
    print(f"\nearliest solved a = {a_start:.4e}  (plot x-limit {amin:.1e})")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 3.9),
                                        gridspec_kw={"wspace": 0.32})

    for st, a, varpi in runs:
        ax1.plot(a, varpi, color=CUSTOM_CMAP(norm(st)))
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$a$")
    ax1.set_ylabel(r"$\varpi(a)$")
    ax1.set_xlim(amin, 1)
    ax1.axhline(1.0, color="k", ls=":", lw=0.8)
    ax1.set_title(r"$\Omega_\Lambda = 0$, $\lambda = %g$" % args.lam,
                  fontsize=9)

    for st, a, varpi in runs:
        s = np.abs(lam_source(varpi, st, args.lam))
        ax2.plot(a, s/s[0], color=CUSTOM_CMAP(norm(st)))
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$a$")
    ax2.set_ylabel(r"$|\lambda$ source$|$ / early value")
    ax2.set_xlim(amin, 1)
    ax2.axhline(1.0, color="k", ls=":", lw=0.8)
    ax2.set_title(r"the $\lambda$ source decays with $\varpi$", fontsize=9)

    # third panel: how flat IS varpi early?  The series gives varpi = 1 + v1*tau
    # with v1 < 0, and a ~ tau in radiation domination, so 1 - varpi should grow
    # LINEARLY in a at early times: a straight line of slope 1 on log-log.
    for st, a, varpi in runs:
        ax3.plot(a, 1.0 - varpi, color=CUSTOM_CMAP(norm(st)))
    ref_a = np.array([amin, 1e-4])
    ax3.plot(ref_a, (1.0 - runs[len(runs)//2][2][0] + 1e-12)
             + (ref_a/ref_a[0])*1e-4, color="k", ls="--", lw=0.9,
             label=r"slope 1 ($1-\varpi \propto a$)")
    ax3.set_xscale("log"); ax3.set_yscale("log")
    ax3.set_xlim(amin, 1)
    ax3.set_xlabel(r"$a$")
    ax3.set_ylabel(r"$1 - \varpi$")
    ax3.legend(loc="upper left")
    ax3.set_title(r"early-time departure from $\varpi=1$", fontsize=9)

    cbar = fig.colorbar(sm, ax=ax3, orientation="vertical",
                        fraction=0.045, pad=0.02)
    cbar.set_label(r"$\varsigma$")

    out = os.path.join(HERE, "..", "plots",
                       f"rainbow_lam_varpi_OmL0_lam{args.lam:g}".replace(".", "p"))
    for ext in ("pdf", "png"):
        fig.savefig(out + "." + ext, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {os.path.relpath(out)}.{{pdf,png}}")

    print(f"\n{'stigma':>8} {'varpi(a=1)':>11} {'source ratio':>13}")
    for st, a, varpi in runs:
        sv = np.abs(lam_source(varpi, st, args.lam))
        print(f"{st:>8.3f} {varpi[-1]:>11.5f} {sv[-1]/sv[0]:>13.5f}")

    print("\nHOW CONSTANT IS varpi EARLY?  varpi at fixed a:")
    probes = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    hdr = "".join(f"{p:>12.0e}" for p in probes)
    print(f"{'stigma':>8}{hdr}")
    for st, a, varpi in runs[::max(1, len(runs)//5)]:
        vals = [np.interp(p, a, varpi, left=np.nan) for p in probes]
        print(f"{st:>8.3f}" + "".join(
            f"{v:>12.6f}" if np.isfinite(v) else f"{'-':>12}" for v in vals))
    print("\nlocal power-law slope d ln(1-varpi)/d ln a (1 = linear in a):")
    for st, a, varpi in runs[::max(1, len(runs)//5)]:
        m = (a > 2*a[0]) & (a < 1e-6)
        sl = np.gradient(np.log(1-varpi[m]), np.log(a[m]))
        print(f"{st:>8.3f}   slope over a in (2a_ini, 1e-6) = "
              f"{np.median(sl):.4f}")


if __name__ == "__main__":
    main()
