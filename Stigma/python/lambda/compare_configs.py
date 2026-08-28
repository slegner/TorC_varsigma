"""
Compare the three ways of sourcing the late-time acceleration:

    OmL only    OmL = 0.7, lam = 0     (the original model -- Lambda as a fluid)
    lam only    OmL = 0,   lam = 0.7   (Lambda from the theory's own term)
    OmL + lam   OmL = 0.7, lam = 0.7   (both)

Run:
    python3 compare_configs.py

Produces hz_configs.pdf (H(z), ratio to the OmL-only run, and varpi(a))
and prints a table.  In every case the a0 shooting normalises H(z=0) = 100h,
so the curves are compared at fixed H0.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import equations_lambda as L

# --- parameters -----------------------------------------------------------
cpr     = 0.99
omega_m = 0.3
h       = 0.67
stigma  = 1.5
mups2   = 4/3

OmL_val = 0.7
lam_val = 0.7

CONFIGS = [
    ("OmL only",  OmL_val, 0.0,     "k",  "-"),
    ("lam only",  0.0,     lam_val, "C0", "-"),
    ("OmL + lam", OmL_val, lam_val, "C3", "-"),
]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hz_configs.pdf")


def main():
    runs = []
    for label, OmL, lam, color, ls in CONFIGS:
        print(f"solving {label:<10}  (OmL = {OmL}, lam = {lam}) ...")
        z, H, P_DE, rho_DE, asol = L.rhopsol(cpr, omega_m, h, OmL, stigma, mups2, lam)
        print(f"   H(z=0) = {H[0]:.4f} km/s/Mpc,  z_max = {z[-1]:.4e}")
        runs.append(dict(label=label, OmL=OmL, lam=lam, color=color, ls=ls,
                         z=z, H=H, a=asol))

    base = runs[0]

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(7.5, 10),
                                        gridspec_kw={"height_ratios": [2, 1, 1.3]})

    for r in runs:
        ax0.plot(r["z"], r["H"], r["ls"], color=r["color"], lw=1.6, label=r["label"])
    ax0.set_xscale("log"); ax0.set_yscale("log")
    ax0.set_xlabel(r"$z$")
    ax0.set_ylabel(r"$H(z)\;[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$")
    ax0.set_title(rf"$\varsigma={stigma}$, $\varpi_0={cpr}$, $m\upsilon_2={mups2:.4f}$,"
                  rf"  $\Omega_\Lambda={OmL_val}$, $\lambda={lam_val}$")
    ax0.legend()

    zgrid = np.geomspace(max(min(r["z"][1] for r in runs), 1e-3),
                         min(r["z"][-1] for r in runs), 3000)
    Hb = np.interp(zgrid, base["z"], base["H"])
    for r in runs[1:]:
        ax1.plot(zgrid, 100*(np.interp(zgrid, r["z"], r["H"])/Hb - 1),
                 r["ls"], color=r["color"], lw=1.6, label=r["label"])
    ax1.axhline(0, color="k", ls="--", lw=1)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$z$")
    ax1.set_ylabel(r"$\Delta H/H$ vs OmL only [%]")
    ax1.legend()

    for r in runs:
        # w_eff from the expansion history:  w = -1 + (2/3)(1+z) dlnH/dz
        z, H = r["z"], r["H"]
        lnH = np.log(H)
        dlnHdz = np.gradient(lnH, z)
        w = -1 + (2/3)*(1 + z)*dlnHdz
        m = z < 3
        ax2.plot(z[m], w[m], r["ls"], color=r["color"], lw=1.6, label=r["label"])
    ax2.axhline(-1, color="0.6", ls=":", lw=1)
    ax2.set_xlabel(r"$z$")
    ax2.set_ylabel(r"$w_{\rm eff}(z)$  (from $H$)")
    ax2.set_xlim(0, 3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"\nsaved {OUT}")

    # --- table ------------------------------------------------------------
    zs = [0.0, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0]
    print(f"\n{'z':>8}" + "".join(f"{r['label']:>14}" for r in runs)
          + "".join(f"{'d%: '+r['label']:>16}" for r in runs[1:]))
    for zz in zs:
        vals = [np.interp(zz, r["z"], r["H"]) for r in runs]
        row = f"{zz:>8.1f}" + "".join(f"{v:>14.4f}" for v in vals)
        row += "".join(f"{100*(v/vals[0]-1):>15.4f}%" for v in vals[1:])
        print(row)


if __name__ == "__main__":
    main()
