"""
Background cosmology plots — fixed stigma (ς) and ϖ₀ (cpr), varying υ₂ (ups2)
===============================================================================

Produces H(z)/H_ΛCDM − 1,  w(z),  and w(a)  for a range of
υ₂ values, with all other parameters fixed.

Run from the python/ directory:
    python background_vary_ups2.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equations import StigmaConstraintError
from bg_solver import solve_one, H_lcdm

# =============================================================================
# Fixed parameters
# =============================================================================
stigma  = 1.00          # ς fixed
cpr     = 0.90          # ϖ₀ fixed
omega_m = 0.30
h       = 0.70

# υ₂ values to sweep
# Must keep the expression under the sqrt positive:
#   omega_r * stigma * (stigma - cpr²)  /  (ups2 * (5*stigma - cpr² + 4*stigma²*(-2+cpr²)))  > 0
# With stigma=1, cpr=0.9 the bracket in the denominator ≈ -0.57, so ups2 must be negative.
UPS2_CANDIDATES = [-0.50, -0.80, -1.00, -4/3, -1.50, -2.00, -2.50]

Z_MAX  = 3.0
NPTS   = 500
H0     = h * 100.0

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Plots")
os.makedirs(OUTDIR, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"), bbox_inches="tight")
    print(f"  saved → {name}.pdf")


def main():
    print(f"Fixed:  ς={stigma},  ϖ₀={cpr},  Ωm={omega_m},  h={h}")
    print("Solving for each υ₂ value...\n")

    results = {}
    colors  = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(UPS2_CANDIDATES)))

    for ups2, col in zip(UPS2_CANDIDATES, colors):
        label = f"{ups2:.2f}" if ups2 != -4/3 else r"-4/3"
        print(f"  υ₂ = {label} ...", end=" ", flush=True)
        try:
            res = solve_one(cpr, stigma, ups2, omega_m, h, Z_MAX, NPTS)
            results[ups2] = (res, col, label)
            print("done")
        except StigmaConstraintError as e:
            print(f"constraint violated — {e}")
        except Exception as e:
            print(f"FAILED — {e}")

    if not results:
        print("No runs succeeded.")
        return

    # ΛCDM reference
    z_ref         = np.linspace(0, Z_MAX, 600)
    Hz_ref        = H_lcdm(z_ref, omega_m, h)
    Hz_ref_interp = interp1d(z_ref, Hz_ref, bounds_error=False, fill_value=np.nan)

    # ── Plot 1: H(z) absolute — Stigma + ΛCDM
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(z_ref, Hz_ref, "k--", lw=2, label=r"$\Lambda$CDM")
    for ups2, (res, col, label) in results.items():
        ax.plot(res["z"], res["Hz"], color=col, lw=1.8,
                label=rf"$\upsilon_2 = {label}$")
    ax.set_xlabel(r"Redshift $z$", fontsize=13)
    ax.set_ylabel(r"$H(z)$ [km/s/Mpc]", fontsize=13)
    ax.set_xlim(0, Z_MAX)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "Hz_absolute_vary_ups2")

    # ── Plot 2: H(z) - H_ΛCDM difference
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="k", lw=1.5, ls="--", label=r"$\Lambda$CDM")
    for ups2, (res, col, label) in results.items():
        diff = res["Hz"] - Hz_ref_interp(res["z"])
        ax.plot(res["z"], diff, color=col, lw=1.8,
                label=rf"$\upsilon_2 = {label}$")
    ax.set_xlabel(r"Redshift $z$", fontsize=13)
    ax.set_ylabel(r"$H(z) - H_{\Lambda\mathrm{CDM}}(z)$ [km/s/Mpc]", fontsize=12)
    ax.set_xlim(0, Z_MAX)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "Hz_diff_vary_ups2")

    # ── Plot 2: w(z) ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(-1,   color="k",    lw=1.5, ls="--", label=r"$w=-1$ ($\Lambda$CDM)")
    ax.axhline(-1/3, color="gray", lw=1.0, ls=":",  alpha=0.7,
               label=r"$w=-1/3$ (acceleration threshold)")
    for ups2, (res, col, label) in results.items():
        ax.plot(res["z"], res["weff"], color=col, lw=1.8,
                label=rf"$\upsilon_2 = {label}$")
    ax.set_xlabel(r"Redshift $z$", fontsize=13)
    ax.set_ylabel(r"$w_\mathrm{eff}(z)$", fontsize=13)
    ax.set_title(rf"Equation of state  ($\varsigma={stigma}$, $\varpi_0={cpr}$)",
                 fontsize=12)
    ax.set_xlim(0, Z_MAX)
    ax.set_ylim(-2.5, 1.0)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "weff_z_vary_ups2")

    # ── Plot 3: w(a) ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(-1,   color="k",    lw=1.5, ls="--", label=r"$w=-1$ ($\Lambda$CDM)")
    ax.axhline(-1/3, color="gray", lw=1.0, ls=":",  alpha=0.7,
               label=r"$w=-1/3$ (acceleration threshold)")
    for ups2, (res, col, label) in results.items():
        ax.plot(res["a"], res["weff"], color=col, lw=1.8,
                label=rf"$\upsilon_2 = {label}$")
    ax.set_xlabel(r"Scale factor $a$", fontsize=13)
    ax.set_ylabel(r"$w_\mathrm{eff}(a)$", fontsize=13)
    ax.set_title(rf"Equation of state  ($\varsigma={stigma}$, $\varpi_0={cpr}$)",
                 fontsize=12)
    ax.set_xlim(1.0 / (1.0 + Z_MAX), 1.0)
    ax.set_ylim(-2.5, 1.0)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "weff_a_vary_ups2")

    print("\nAll plots saved to", OUTDIR)
    plt.show()


if __name__ == "__main__":
    main()
