"""
Background cosmology plots in redshift space  (z = 0 to ~3)
============================================================

For a range of varsigma (ς) values, this script produces four plots:

  1. H(z)/H0          — expansion history vs ΛCDM
  2. ΔH/H_ΛCDM        — fractional deviation from ΛCDM
  3. w_eff(z)         — effective dark-energy equation of state
  4. Ω components(z)  — matter / radiation / dark-energy fractions

How the physics works
---------------------
rhopsol() (from equations.py) integrates the coupled Stigma ODEs for
a(τ) and ϖ(τ) in conformal time, starting from power-series initial
conditions deep in radiation domination. Different varsigma values change
the coupling between ϖ and gravity, shifting H(z) away from ΛCDM.

The key outputs from rhopsol are:
  asol    — scale factor a(τ),  goes from ~0 to 1 at today
  hsol    — physical H(τ) in km/s/Mpc
  P_DE, rho_DE  — effective dark-energy pressure and density

We then convert a → z via  z = 1/a − 1  and plot everything vs z.

Run from the python/ directory:
    python background_hz.py
"""

import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── import the ODE solver from equations.py (same directory) ─────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equations import omega_r, rhopsol, validate_stigma_constraints, StigmaConstraintError


# =============================================================================
# Parameters — edit these to explore different cosmologies
# =============================================================================

h          = 0.70          # dimensionless Hubble constant (H0 = h × 100 km/s/Mpc)
omega_m    = 0.30          # matter density fraction today
ups2       = -4.0 / 3.0   # υ₂ parameter (fixed)
cpr        = 0.90          # ϖ₀ = initial value of the varpi field

Omega_r    = omega_r(h)                    # radiation density fraction (computed)
OmL        = 1.0 - omega_m - Omega_r      # dark-energy density fraction (flat universe)
H0         = h * 100.0                    # H0 in km/s/Mpc

# varsigma values to try — invalid ones are automatically skipped
VARSIGMA_CANDIDATES = [0.70, 0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.00, 1.03, 1.06, 1.10, 1.15, 1.20]

Z_MAX  = 3.0          # maximum redshift to plot
NPTS   = 500          # number of points on z grid

# output folder (../Plots/ relative to this script)
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Plots")
os.makedirs(OUTDIR, exist_ok=True)


# =============================================================================
# ΛCDM reference  H(z) = H0 √[Ωr(1+z)⁴ + Ωm(1+z)³ + ΩΛ]
# =============================================================================

def H_lcdm(z):
    return H0 * np.sqrt(Omega_r * (1+z)**4 + omega_m * (1+z)**3 + OmL)


# =============================================================================
# Solve the Stigma ODEs for one varsigma value and return redshift-space arrays
# =============================================================================

def solve_one(stigma):
    """
    Returns arrays interpolated onto a uniform z grid [0, Z_MAX].

    rhopsol outputs:
      asol    : scale factor (0 → 1 at today)
      hsol    : H(τ) in km/s/Mpc  (equals H0 when asol=1 by construction)
      P_DE    : effective dark-energy pressure  (normalised)
      rho_DE  : effective dark-energy density   (normalised)

    We compute:
      z        = 1/a − 1
      w_eff(z) = P_DE / rho_DE
      Ωi(z)    = ρi / ρ_crit(z)  where ρ_crit ∝ H(z)²
    """
    asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE = rhopsol(
        cpr, omega_m, h, OmL, stigma, ups2
    )

    # ── convert to redshift ───────────────────────────────────────────────────
    # asol goes from small value to 1 today, so z = 1/a − 1 gives z=0 today
    z_raw = 1.0 / asol - 1.0

    # keep only the late-time range we care about (z = 0 to Z_MAX)
    # and sort ascending in z (ODE runs forward in time, so z decreases)
    mask = (z_raw >= 0.0) & (z_raw <= Z_MAX + 0.1)
    if mask.sum() < 10:
        raise RuntimeError(f"Too few points in z=[0,{Z_MAX}] for ς={stigma}")

    z_s    = z_raw[mask]
    Hz_s   = hsol[mask]           # H(z) in km/s/Mpc
    weff_s = (P_DE / rho_DE)[mask]

    idx  = np.argsort(z_s)        # sort by increasing z
    z_s, Hz_s, weff_s = z_s[idx], Hz_s[idx], weff_s[idx]

    # ── density fractions  Ωi(z) ─────────────────────────────────────────────
    # E²(z) = (H(z)/H0)²  normalised Hubble squared
    E2   = (Hz_s / H0) ** 2
    Om_m = omega_m * (1 + z_s)**3 / E2      # matter
    Om_r = Omega_r * (1 + z_s)**4 / E2      # radiation
    Om_L = OmL             / E2              # cosmological constant
    Om_x = 1.0 - Om_m - Om_r - Om_L         # residual (Stigma modification)

    # ── interpolate onto a uniform z grid ────────────────────────────────────
    z_grid = np.linspace(0.0, min(Z_MAX, z_s.max()), NPTS)
    kw     = dict(bounds_error=False, fill_value=np.nan)

    # also keep a grid (a = 1/(1+z)) for w(a) plot
    a_grid = 1.0 / (1.0 + z_grid)

    return {
        "z":      z_grid,
        "a":      a_grid,
        "Hz":     interp1d(z_s, Hz_s,   **kw)(z_grid),
        "weff":   interp1d(z_s, weff_s, **kw)(z_grid),
        "Om_m":   interp1d(z_s, Om_m,   **kw)(z_grid),
        "Om_r":   interp1d(z_s, Om_r,   **kw)(z_grid),
        "Om_L":   interp1d(z_s, Om_L,   **kw)(z_grid),
        "Om_x":   interp1d(z_s, Om_x,   **kw)(z_grid),
    }


# =============================================================================
# Main: solve for each valid varsigma, then produce plots
# =============================================================================

def main():

    # ── check which varsigma values satisfy the theory constraints ────────────
    print("Checking theory constraints for each ς candidate...")
    valid_stigmas = []
    for s in VARSIGMA_CANDIDATES:
        try:
            validate_stigma_constraints(s, cpr)
            valid_stigmas.append(s)
            print(f"  ς = {s:.2f}  ✓")
        except StigmaConstraintError as e:
            print(f"  ς = {s:.2f}  ✗  ({e})")

    if not valid_stigmas:
        print("\nNo valid ς values found — adjust VARSIGMA_CANDIDATES or cpr.")
        return

    # ── solve the ODEs for each valid varsigma ────────────────────────────────
    print("\nRunning ODE solver...")
    results = {}    # stigma → dict of arrays
    colors  = plt.cm.plasma(np.linspace(0.1, 0.85, len(valid_stigmas)))

    for stigma, col in zip(valid_stigmas, colors):
        print(f"  Solving ς = {stigma:.2f} ...", end=" ", flush=True)
        try:
            results[stigma] = (solve_one(stigma), col)
            print("done")
        except Exception as e:
            print(f"FAILED — {e}")

    if not results:
        print("No runs succeeded.")
        return

    # ΛCDM reference on the same z grid
    z_ref   = np.linspace(0, Z_MAX, 600)
    Hz_ref  = H_lcdm(z_ref)
    Hz_ref_interp = interp1d(z_ref, Hz_ref, bounds_error=False, fill_value=np.nan)

    # ── helper to save figure ─────────────────────────────────────────────────
    def save(fig, name):
        path = os.path.join(OUTDIR, f"{name}.pdf")
        fig.savefig(path, bbox_inches="tight")
        print(f"  saved → {name}.pdf")

    # =========================================================================
    # Plot 1: H(z) — Stigma curves + ΛCDM on same axes
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(z_ref, Hz_ref, "k--", lw=2, label=r"$\Lambda$CDM")
    for stigma, (res, col) in results.items():
        ax.plot(res["z"], res["Hz"], color=col, lw=1.8,
                label=rf"$\varsigma = {stigma:.2f}$")

    ax.set_xlabel(r"Redshift $z$", fontsize=13)
    ax.set_ylabel(r"$H(z)$ [km/s/Mpc]", fontsize=13)
    ax.set_title(r"Expansion history — Stigma vs $\Lambda$CDM", fontsize=12)
    ax.set_xlim(0, Z_MAX)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "Hz_absolute_varsigma")

    # =========================================================================
    # Plot 2: H(z) - H_ΛCDM — absolute difference
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(0, color="k", lw=1.5, ls="--", label=r"$\Lambda$CDM")
    for stigma, (res, col) in results.items():
        diff = res["Hz"] - Hz_ref_interp(res["z"])
        ax.plot(res["z"], diff, color=col, lw=1.8,
                label=rf"$\varsigma = {stigma:.2f}$")

    ax.set_xlabel(r"Redshift $z$", fontsize=13)
    ax.set_ylabel(r"$H(z) - H_{\Lambda\mathrm{CDM}}(z)$ [km/s/Mpc]", fontsize=12)
    ax.set_title(r"Difference from $\Lambda$CDM", fontsize=12)
    ax.set_xlim(0, Z_MAX)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "Hz_diff_varsigma")

    # =========================================================================
    # Plot 3: effective equation of state  w_eff(z) = P_DE / ρ_DE
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(-1,    color="k",    lw=1.5, ls="--", label=r"$w=-1$ ($\Lambda$CDM)")
    ax.axhline(-1/3,  color="gray", lw=1.0, ls=":",  alpha=0.7,
               label=r"$w=-1/3$ (acceleration threshold)")

    for stigma, (res, col) in results.items():
        ax.plot(res["z"], res["weff"], color=col, lw=1.8,
                label=rf"$\varsigma = {stigma:.2f}$")

    ax.set_xlabel(r"Redshift $z$", fontsize=13)
    ax.set_ylabel(r"$w_\mathrm{eff}(z)$", fontsize=13)
    ax.set_title(r"Effective dark-energy equation of state $w_\mathrm{eff}(z)$", fontsize=12)
    ax.set_xlim(0, Z_MAX)
    ax.set_ylim(-2.5, 1.0)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "weff_z_varsigma")

    # =========================================================================
    # Plot 3b: w(a) — same but vs scale factor a, no log scale
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axhline(-1,   color="k",    lw=1.5, ls="--", label=r"$w=-1$ ($\Lambda$CDM)")
    ax.axhline(-1/3, color="gray", lw=1.0, ls=":",  alpha=0.7,
               label=r"$w=-1/3$ (acceleration threshold)")

    for stigma, (res, col) in results.items():
        ax.plot(res["a"], res["weff"], color=col, lw=1.8,
                label=rf"$\varsigma = {stigma:.2f}$")

    ax.set_xlabel(r"Scale factor $a$", fontsize=13)
    ax.set_ylabel(r"$w_\mathrm{eff}(a)$", fontsize=13)
    ax.set_title(r"Effective dark-energy equation of state $w_\mathrm{eff}(a)$", fontsize=12)
    ax.set_xlim(1.0 / (1.0 + Z_MAX), 1.0)
    ax.set_ylim(-2.5, 1.0)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    save(fig, "weff_a_varsigma")

    # =========================================================================
    # Plot 4: density fractions Ωi(z) — one panel per varsigma
    # =========================================================================
    n     = len(results)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                              sharex=True, sharey=True)
    axes_flat = np.array(axes).flatten()

    for ax, (stigma, (res, _)) in zip(axes_flat, results.items()):
        z = res["z"]
        ax.stackplot(
            z,
            res["Om_r"], res["Om_m"], res["Om_L"], res["Om_x"],
            labels=[r"$\Omega_r$", r"$\Omega_m$", r"$\Omega_\Lambda$",
                    r"$\Omega_\mathrm{Stigma}$"],
            colors=["#FFA500", "#4488CC", "#CC3333", "#55AA55"],
            alpha=0.80,
        )
        ax.set_title(rf"$\varsigma = {stigma:.2f}$", fontsize=10)
        ax.set_xlim(0, Z_MAX)
        ax.set_ylim(0, 1.3)
        ax.grid(ls="--", alpha=0.3)
        if ax is axes_flat[0]:
            ax.legend(fontsize=7, loc="upper right")

    for ax in axes_flat[n:]:
        ax.set_visible(False)
    for ax in axes_flat[(nrows - 1) * ncols: n]:
        ax.set_xlabel(r"Redshift $z$", fontsize=11)
    for ax in axes_flat[::ncols]:
        ax.set_ylabel(r"$\Omega_i(z)$", fontsize=11)

    fig.suptitle(
        r"Density fractions $\Omega_i(z)$ for different $\varsigma$"
        f"\n($\\varpi_0={cpr}$, $\\Omega_m={omega_m}$, $h={h}$)",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    save(fig, "Omega_z_varsigma")

    print("\nAll plots saved to", OUTDIR)
    plt.show()


if __name__ == "__main__":
    main()
