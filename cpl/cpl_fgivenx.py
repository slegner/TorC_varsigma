"""
fgivenx functional posteriors for the CPL fit to DESI DR2 BAO.

Reads the nested-sampling chains produced by `cpl_nss.py` and plots the
functional posterior (fgivenx sigma-contours) of

  1. w(z) = w0 + wa z/(1+z)                      -> output/cpl_fgivenx_wz.pdf
  2. E(z) = H(z)/H0, as a ratio to LCDM          -> output/cpl_fgivenx_hz.pdf
  3. DM/rd and DH/rd, as a ratio to LCDM, with
     the DESI DR2 data points overplotted        -> output/cpl_fgivenx_bao.pdf

The LCDM reference curve is the posterior median of the `lcdm` chain, so the
ratio panels show the CPL departure from the best-fit LCDM background rather
than from an arbitrary fiducial.

Usage
-----
    source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
    cd distances/TorC/cpl
    python cpl_fgivenx.py                 # all three figures
    python cpl_fgivenx.py --output output_mock
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fgivenx import plot_contours

from cpl_nss import (
    load_as_nested_samples, PARAMS, OMEGA_R, C_KMS,
    Z_DATA, MEAN_DATA, TYPE_DATA, COV_DATA,
)

# Plot grid: DESI DR2 spans 0.295 <= z <= 2.330.
Z_MIN, Z_MAX = 0.0, 2.5
_Z_FINE = np.linspace(0.0, Z_MAX, 2001)   # internal integration grid


# -- Numpy background (fgivenx calls f once per posterior sample) -------------

def _f_de(z, w0, wa):
    return (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * z / (1.0 + z))


def _E(z, omega_m, w0, wa):
    om_de = 1.0 - omega_m - OMEGA_R
    return np.sqrt(omega_m * (1.0 + z) ** 3
                   + omega_r_term(z)
                   + om_de * _f_de(z, w0, wa))


def omega_r_term(z):
    return OMEGA_R * (1.0 + z) ** 4


def _cum_inv_E(omega_m, w0, wa):
    """int_0^z dz'/E(z') on the internal fine grid."""
    inv_E = 1.0 / _E(_Z_FINE, omega_m, w0, wa)
    return np.concatenate([[0.0], np.cumsum(
        0.5 * (inv_E[:-1] + inv_E[1:]) * np.diff(_Z_FINE))])


# -- Functions of (z, theta) handed to fgivenx --------------------------------
# theta is ordered as PARAMS["cpl"] = [h0rd, omega_m, w0, wa]; for the LCDM
# chain theta = [h0rd, omega_m] and (w0, wa) are pinned to (-1, 0).

def _unpack(theta):
    if len(theta) == 4:
        return theta[0], theta[1], theta[2], theta[3]
    return theta[0], theta[1], -1.0, 0.0


def w_of_z(z, theta):
    _, _, w0, wa = _unpack(theta)
    return w0 + wa * z / (1.0 + z)


def E_of_z(z, theta):
    _, omega_m, w0, wa = _unpack(theta)
    return _E(np.asarray(z, dtype=float), omega_m, w0, wa)


def dm_over_rd(z, theta):
    h0rd, omega_m, w0, wa = _unpack(theta)
    integral = np.interp(z, _Z_FINE, _cum_inv_E(omega_m, w0, wa))
    return C_KMS / h0rd * integral


def dh_over_rd(z, theta):
    h0rd, omega_m, w0, wa = _unpack(theta)
    return C_KMS / (h0rd * _E(np.asarray(z, dtype=float), omega_m, w0, wa))


# -- Helpers ------------------------------------------------------------------

def _theta_and_weights(model, output_dir):
    ns = load_as_nested_samples(model, output_dir)
    theta = ns[PARAMS[model]].to_numpy()
    return theta, ns.get_weights()


def _lcdm_reference(output_dir):
    """Posterior-median LCDM parameters, used as the denominator of the ratios."""
    ns = load_as_nested_samples("lcdm", output_dir)
    h0rd = float(np.asarray(ns["h0rd"].quantile(0.5)))
    om   = float(np.asarray(ns["omega_m"].quantile(0.5)))
    return np.array([h0rd, om])


def _ratio(f, ref_theta):
    """Wrap f(z, theta) into f(z, theta) / f(z, ref_theta)."""
    def g(z, theta):
        return f(z, theta) / f(z, ref_theta)
    return g


def _contour(f, z, theta, weights, ax, cache, colors=plt.cm.Blues_r,
             lines=True):
    cbar = plot_contours(f, z, theta, weights=weights, ax=ax,
                         colors=colors, lines=lines, cache=cache,
                         ny=200, parallel=False)
    return cbar


# -- Figures ------------------------------------------------------------------

def plot_wz(output_dir="output"):
    theta, w = _theta_and_weights("cpl", output_dir)
    z = np.linspace(Z_MIN, Z_MAX, 200)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cbar = _contour(w_of_z, z, theta, w, ax,
                    cache=os.path.join(output_dir, "cache_wz"))
    cbar = plt.colorbar(cbar, ax=ax, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])

    ax.axhline(-1.0, color="k", ls="--", lw=1.2, label=r"$\Lambda$CDM ($w=-1$)")
    ax.axhline(-1.0 / 3.0, color="0.5", ls=":", lw=1.0,
               label=r"$w=-1/3$ (acceleration threshold)")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$w(z)$")
    ax.set_xlim(Z_MIN, Z_MAX)
    ax.set_title("CPL equation of state, DESI DR2 BAO")
    ax.legend(loc="lower left", fontsize=9)
    out = os.path.join(output_dir, "cpl_fgivenx_wz.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_hz(output_dir="output"):
    ref = _lcdm_reference(output_dir)
    theta, w = _theta_and_weights("cpl", output_dir)
    z = np.linspace(Z_MIN, Z_MAX, 200)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cbar = _contour(_ratio(E_of_z, ref), z, theta, w, ax,
                    cache=os.path.join(output_dir, "cache_hz"))
    cbar = plt.colorbar(cbar, ax=ax, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])

    ax.axhline(1.0, color="k", ls="--", lw=1.2, label=r"$\Lambda$CDM")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$E(z)\,/\,E_{\Lambda\rm CDM}(z)$")
    ax.set_xlim(Z_MIN, Z_MAX)
    ax.set_title("Expansion rate relative to best-fit $\\Lambda$CDM")
    ax.legend(loc="best", fontsize=9)
    out = os.path.join(output_dir, "cpl_fgivenx_hz.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


def plot_bao(output_dir="output"):
    """DM/rd and DH/rd relative to LCDM, with the DESI DR2 points overplotted."""
    ref = _lcdm_reference(output_dir)
    theta, w = _theta_and_weights("cpl", output_dir)
    z = np.linspace(0.05, Z_MAX, 200)   # DM/rd -> 0/0 at z = 0
    sigma = np.sqrt(np.diag(COV_DATA))

    fig, axes = plt.subplots(2, 1, figsize=(6.2, 7.0), sharex=True)

    for ax, f, name, label in [
        (axes[0], dm_over_rd, "DM_over_rs", r"$(D_M/r_d)\,/\,\Lambda$CDM"),
        (axes[1], dh_over_rd, "DH_over_rs", r"$(D_H/r_d)\,/\,\Lambda$CDM"),
    ]:
        cbar = _contour(_ratio(f, ref), z, theta, w, ax,
                        cache=os.path.join(output_dir, f"cache_{name}"))
        cbar = plt.colorbar(cbar, ax=ax, ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['', r'$1\sigma$', r'$2\sigma$', r'$3\sigma$'])
        ax.axhline(1.0, color="k", ls="--", lw=1.2)

        # DESI points of this type, divided by the same LCDM reference
        sel = [i for i, t in enumerate(TYPE_DATA) if t == name]
        if sel:
            zz  = Z_DATA[sel]
            den = f(zz, ref)
            ax.errorbar(zz, MEAN_DATA[sel] / den, yerr=sigma[sel] / den,
                        fmt="o", color="crimson", ms=4, capsize=2,
                        label="DESI DR2", zorder=5)
            ax.legend(loc="best", fontsize=9)
        ax.set_ylabel(label)

    # DV/rs point (z = 0.295) has no DM/DH panel of its own; note it.
    axes[1].set_xlabel(r"$z$")
    axes[0].set_title("BAO observables relative to best-fit $\\Lambda$CDM")
    axes[0].set_xlim(0.0, Z_MAX)
    out = os.path.join(output_dir, "cpl_fgivenx_bao.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="fgivenx plots for the CPL DESI fit")
    p.add_argument("--output", default="output")
    p.add_argument("--only", choices=["wz", "hz", "bao"], default=None)
    args = p.parse_args()

    if args.only in (None, "wz"):
        plot_wz(args.output)
    if args.only in (None, "hz"):
        plot_hz(args.output)
    if args.only in (None, "bao"):
        plot_bao(args.output)
