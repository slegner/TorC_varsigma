"""
Background cosmology plots — fixed stigma (ς), varying ϖ₀ (cpr)
================================================================

Produces H(z)/H_ΛCDM − 1,  w(z),  and w(a)  for a range of
initial varpi field values ϖ₀, with all other parameters fixed.

Run from the python/ directory:
    python background_vary_cpr.py
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
stigma  = 1.00          # ς fixed  (1.0 = GR-like limit of Stigma theory)
ups2    = -4.0 / 3.0    # υ₂ fixed
omega_m = 0.30
h       = 0.70

# ϖ₀ values to sweep — must satisfy stigma > cpr² and the second constraint
# For stigma=1 both constraints reduce to  0 < cpr < 1
CPR_CANDIDATES = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

Z_MAX  = 3.0
NPTS   = 500
H0     = h * 100.0

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Plots")
os.makedirs(OUTDIR, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"), bbox_inches="tight")
    print(f"  saved → {name}.pdf")


def main():
    print(f"Fixed:  ς={stigma},  υ₂={ups2:.4f},  Ωm={omega_m},  h={h}")
    print("Solving for each ϖ₀ value...\n")

    results = {}
    colors  = plt.cm.viridis(np.linspace(0.1, 0.9, len(CPR_CANDIDATES)))

    for cpr, col in zip(CPR_CANDIDATES, colors):
        print(f"  ϖ₀ = {cpr:.2f} ...", end=" ", flush=True)
        try:
            res = solve_one(cpr, stigma, ups2, omega_m, h, Z_MAX, NPTS)
            results[cpr] = (res, col)
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
    for cpr, (res, col) in results.items():
        ax.plot(res["z"], res["Hz"], color=col, lw=1.8,
                label=rf"$\varpi_0 = {cpr:.2f}$")
    ax.set_xlabel(r"Redshift $z$", fontsize=13)
    ax.set_ylabel(r"$H(z)$ [km/s/Mpc]", fontsize=13)
    ax.set_xlim(0, Z_MAX)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "Hz_absolute_vary_cpr")

    # ── Plot 2: H(z) - H_ΛCDM difference
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="k", lw=1.5, ls="--", label=r"$\Lambda$CDM")
    for cpr, (res, col) in results.items():
        diff = res["Hz"] - Hz_ref_interp(res["z"])
        ax.plot(res["z"], diff, color=col, lw=1.8,
                label=rf"$\varpi_0 = {cpr:.2f}$")
    ax.set_xlabel(r"Redshift $z$", fontsize=13)
    ax.set_ylabel(r"$H(z) - H_{\Lambda\mathrm{CDM}}(z)$ [km/s/Mpc]", fontsize=12)
    ax.set_xlim(0, Z_MAX)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "Hz_diff_vary_cpr")

    # ── Plot 2: w(z) ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(-1,   color="k",    lw=1.5, ls="--", label=r"$w=-1$ ($\Lambda$CDM)")
    ax.axhline(-1/3, color="gray", lw=1.0, ls=":",  alpha=0.7,
               label=r"$w=-1/3$ (acceleration threshold)")
    for cpr, (res, col) in results.items():
        ax.plot(res["z"], res["weff"], color=col, lw=1.8,
                label=rf"$\varpi_0 = {cpr:.2f}$")
    ax.set_xlabel(r"Redshift $z$", fontsize=13)
    ax.set_ylabel(r"$w_\mathrm{eff}(z)$", fontsize=13)
    ax.set_title(rf"Equation of state  ($\varsigma={stigma}$, $\upsilon_2={ups2:.3f}$)",
                 fontsize=12)
    ax.set_xlim(0, Z_MAX)
    ax.set_ylim(-2.5, 1.0)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "weff_z_vary_cpr")

    # ── Plot 3: w(a) ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(-1,   color="k",    lw=1.5, ls="--", label=r"$w=-1$ ($\Lambda$CDM)")
    ax.axhline(-1/3, color="gray", lw=1.0, ls=":",  alpha=0.7,
               label=r"$w=-1/3$ (acceleration threshold)")
    for cpr, (res, col) in results.items():
        ax.plot(res["a"], res["weff"], color=col, lw=1.8,
                label=rf"$\varpi_0 = {cpr:.2f}$")
    ax.set_xlabel(r"Scale factor $a$", fontsize=13)
    ax.set_ylabel(r"$w_\mathrm{eff}(a)$", fontsize=13)
    ax.set_title(rf"Equation of state  ($\varsigma={stigma}$, $\upsilon_2={ups2:.3f}$)",
                 fontsize=12)
    ax.set_xlim(1.0 / (1.0 + Z_MAX), 1.0)
    ax.set_ylim(-2.5, 1.0)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "weff_a_vary_cpr")

    print("\nAll plots saved to", OUTDIR)
    plt.show()


if __name__ == "__main__":
    main()
