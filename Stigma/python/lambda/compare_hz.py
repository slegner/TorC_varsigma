"""
Compare H(z) of the two models: reference (python/equations_Hz.py, lam = 0)
vs Lambda (equations_lambda.py).

Both are started from the SAME initial values -- the script checks that
explicitly -- so any difference in H(z) comes purely from the different
second derivatives dda / ddcp.

Run:
    python compare_hz.py
Until dda/ddcp are filled in, it plots the reference model only and says so.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import equations_lambda as lam_model
from reference_model import ref

# --- parameters -----------------------------------------------------------
cpr     = 0.99      # varpi_0
omega_m = 0.3
h       = 0.67
OmL     = 0.7
stigma  = 1.5
mups2   = 4/3
lam     = 1.0       # the Lambda parameter of the second model

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hz_comparison.pdf")


def check_same_initial_values():
    """Both models must start from the same [a, cp, a', cp'] at the same tau_ini."""
    tau_ref = ref.small(cpr, omega_m, h, stigma, mups2)
    tau_lam = lam_model.small(cpr, omega_m, h, stigma, mups2)
    y_ref = [ref.aseries(tau_ref, cpr, omega_m, h, OmL, 1, stigma, mups2),
             ref.cpseries(tau_ref, cpr, omega_m, h, OmL, 1, stigma, mups2),
             ref.aseriesp(tau_ref, cpr, omega_m, h, OmL, 1, stigma, mups2),
             ref.cpseriesp(tau_ref, cpr, omega_m, h, OmL, 1, stigma, mups2)]
    y_lam = lam_model.initial_state(tau_lam, cpr, omega_m, h, OmL, 1, stigma, mups2)

    assert np.isclose(tau_ref, tau_lam, rtol=0, atol=0), "tau_ini differs between models"
    assert np.allclose(y_ref, y_lam, rtol=0, atol=0), "initial values differ between models"
    print(f"initial values match:  tau_ini = {tau_ref:.6e}")
    print(f"  a  = {y_ref[0]:.6e}   cp  = {y_ref[1]:.6e}")
    print(f"  a' = {y_ref[2]:.6e}   cp' = {y_ref[3]:.6e}")
    return tau_ref, y_ref


def main():
    check_same_initial_values()

    if lam == 0:
        print("\nNOTE: at lam = 0 the Lambda model is bit-identical to the reference\n"
              "      (verified to 0 ulp), so the two curves will lie exactly on top of\n"
              "      each other.  Set lam != 0 -- or OmL = 0 with lam sourcing the\n"
              "      acceleration -- to get a genuinely different H(z).")

    print(f"\nsolving reference model (lam = 0) ...")
    z_ref, H_ref, _, _, _ = ref.rhopsol(cpr, omega_m, h, OmL, stigma, mups2)
    print(f"  H0 = {H_ref[0]:.4f} km/s/Mpc,  z_max = {z_ref[-1]:.3e}")

    z_lam = H_lam = None
    try:
        print(f"solving Lambda model (lam = {lam}) ...")
        z_lam, H_lam, _, _, _ = lam_model.rhopsol(cpr, omega_m, h, OmL, stigma, mups2, lam)
        print(f"  H0 = {H_lam[0]:.4f} km/s/Mpc,  z_max = {z_lam[-1]:.3e}")
    except NotImplementedError as exc:
        print(f"\n  [skipped] {exc}\n")

    # --- plot -------------------------------------------------------------
    if H_lam is None:
        fig, ax0 = plt.subplots(figsize=(7, 5))
        ax1 = None
    else:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [2, 1]})

    ax0.plot(z_ref, H_ref, "k--", lw=2, label=r"reference ($\lambda=0$)")
    if H_lam is not None:
        ax0.plot(z_lam, H_lam, "-", lw=1.5, label=rf"$\Lambda$ model ($\lambda={lam}$)")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_ylabel(r"$H(z)\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$")
    ax0.set_title(rf"$\varsigma={stigma}$, $\varpi_0={cpr}$, $m\upsilon_2={mups2:.4f}$")
    ax0.legend()

    if ax1 is not None:
        zgrid = np.logspace(np.log10(max(z_ref[0], z_lam[0], 1e-3)),
                            np.log10(min(z_ref[-1], z_lam[-1])), 2000)
        Hr = np.interp(zgrid, z_ref, H_ref)
        Hl = np.interp(zgrid, z_lam, H_lam)
        ax1.plot(zgrid, 100*(Hl - Hr)/Hr, "-", lw=1.5)
        ax1.axhline(0, color="k", ls="--", lw=1)
        ax1.set_xscale("log")
        ax1.set_ylabel(r"$\Delta H/H$ [%]")
    (ax1 or ax0).set_xlabel(r"$z$")

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"saved {OUT}")

    # --- table ------------------------------------------------------------
    if H_lam is not None:
        print(f"\n{'z':>10} {'H_ref':>12} {'H_lam':>12} {'dH/H [%]':>10}")
        for zz in [0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0]:
            if zz < z_ref[0] or zz > z_ref[-1]:
                continue
            hr = np.interp(zz, z_ref, H_ref)
            hl = np.interp(zz, z_lam, H_lam)
            print(f"{zz:>10.2f} {hr:>12.4f} {hl:>12.4f} {100*(hl-hr)/hr:>10.4f}")


if __name__ == "__main__":
    main()
