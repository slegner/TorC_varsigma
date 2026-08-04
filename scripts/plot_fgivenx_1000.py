"""fgivenx posterior-predictive plot for the 1000-live TorC fit vs DESI DR2.

Two panels (same observables as the rainbow plots):
    D_V/(r_d z^{2/3}) - LCDM best fit      (7 points)
    D_M/(z D_H)       - LCDM best fit      (6 points)
with the DESI data + errors over fgivenx contours of the posterior, and the
max-likelihood curve dashed.

HOW THE MAX-LIKELIHOOD CURVE IS OBTAINED
----------------------------------------
The older plot_fgivenx.py used argmax(logL) over the dead points, i.e. the
single best SAMPLE the sampler happened to visit.  That is not a maximum --
nested sampling never optimises, and with ~12500 dead points in 4 dimensions
the best sample can sit well short of the true peak.

Here we do both and report the difference:
  1. best dead point         -- argmax over the chain (the old behaviour)
  2. Nelder-Mead from there  -- an actual optimisation of logL, bounded to
                                the prior box, matching how the LCDM
                                reference curve is already fitted
The plot is drawn with (2) and annotated with the gain in logL over (1), so
it is clear what was optimised and by how much.

Model curves are precomputed for every dead point with jax (vmap, chunked);
fgivenx then interpolates rows weighted by the posterior weights.
"""
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from pandas import read_csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from fgivenx import plot_contours
from anesthetic import NestedSamples

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, ".."))
from torc.theory import rhopsol_jax as rhopsol_noH0, OMEGA_R_DEFAULT
from torc.likelihood import loglikelihood as torc_logl, MUPS2_FIXED
from torc.priors import PRIOR_BOUNDS

CHAIN = os.path.join(here, "..", "chains", "torc_ns_julia_1000.npz")
omegar = OMEGA_R_DEFAULT
c_kms = 299792.458
COLOR = "#DC267F"


def cmap_from_hex(hex_color):
    return LinearSegmentedColormap.from_list(
        "custom", [(1, 1, 1), to_rgb(hex_color)], N=256)


plt.rcParams.update({
    "text.usetex": True, "font.family": "cmtt", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "axes.linewidth": 0.8,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "lines.linewidth": 0.9,
})

# --- DESI DR2 data -----------------------------------------------------------
_dd = os.environ.get("DESIDR2_DIR", os.path.join(here, "..", "data", "desidr2"))
data = read_csv(os.path.join(_dd, "desidr2_mean.txt"), header=None,
                index_col=None, sep=r"\s+", comment="#")
cov = np.loadtxt(os.path.join(_dd, "desidr2_cov.txt"))
zs = data.iloc[:, 0].to_numpy()
mean_data = data.iloc[:, 1].to_numpy()
types = data.iloc[:, 2].to_numpy()
cov_inv = np.linalg.inv(cov)


# --- LCDM best-fit reference (optimised, unchanged) --------------------------
def E_l(z, om):
    return np.sqrt(om*(1+z)**3 + omegar*(1+z)**4 + (1-om-omegar))


def dm_l(z, h0rd, om):
    return c_kms/h0rd*quad(lambda x: 1/E_l(x, om), 0, z)[0]


def dh_l(z, h0rd, om):
    return c_kms/h0rd/E_l(z, om)


def dv_l(z, h0rd, om):
    return (z*dm_l(z, h0rd, om)**2*dh_l(z, h0rd, om))**(1/3)


def chi2_l(p):
    h0rd, om = p
    if om <= 0 or om >= 1 or h0rd <= 0:
        return 1e10
    fns = {"DM_over_rs": dm_l, "DH_over_rs": dh_l, "DV_over_rs": dv_l}
    pred = [fns[t](z, h0rd, om) for z, t in zip(zs, types)]
    return (mean_data-pred) @ cov_inv @ (mean_data-pred)


h0rd_bf, om_bf = minimize(chi2_l, x0=[10145, 0.298], method="Nelder-Mead",
                          options={"xatol": 1e-6, "fatol": 1e-8,
                                   "maxiter": 10000}).x
print(f"LCDM best fit (Nelder-Mead): H0rd={h0rd_bf:.1f}, Om={om_bf:.4f}")

# --- data residuals ----------------------------------------------------------
np.random.seed(42)
samps = multivariate_normal(mean_data, cov).rvs(10_000)
dmx, dhx, dvx = (types == t for t in ("DM_over_rs", "DH_over_rs", "DV_over_rs"))
dm_s, dh_s, dv_s = samps[:, dmx], samps[:, dhx], samps[:, dvx]
zm, zv_bgs = zs[dmx], zs[dvx]
dv_der = (zm*dm_s**2*dh_s)**(1/3)
zv_all = np.concatenate([zv_bgs, zm])
dv_all = np.hstack([dv_s, dv_der])
o = np.argsort(zv_all)
zv_all, dv_all = zv_all[o], dv_all[:, o]
dv_mean, dv_std = (dv_all/zv_all**(2/3)).mean(0), (dv_all/zv_all**(2/3)).std(0)
dv_res = dv_mean - np.array([dv_l(z, h0rd_bf, om_bf)/z**(2/3) for z in zv_all])
dm_rat = dm_s/(zm*dh_s)
dm_mean, dm_std = dm_rat.mean(0), dm_rat.std(0)
dm_res = dm_mean - np.array(
    [dm_l(z, h0rd_bf, om_bf)/(z*dh_l(z, h0rd_bf, om_bf)) for z in zm])

# --- model curves ------------------------------------------------------------
A_PLOT = np.linspace(0.22, 0.995, 150)
Z_PLOT = 1.0/A_PLOT - 1.0
ZG = jnp.linspace(0.0, 4.0, 801)


def _cumtrapz(y, x):
    dx = jnp.diff(x)
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5*dx*(y[1:]+y[:-1]))])


def _ratios_from_E(E_grid, h0rd):
    dm = (c_kms/h0rd)*_cumtrapz(1.0/E_grid, ZG)
    dh = (c_kms/h0rd)/E_grid
    dv_ratio = (ZG*dm**2*dh)**(1.0/3.0)/ZG**(2.0/3.0)
    dm_ratio = dm/(ZG*dh)
    zp = jnp.array(Z_PLOT)
    return jnp.interp(zp, ZG, dv_ratio), jnp.interp(zp, ZG, dm_ratio)


@jax.jit
def torc_ratios(omega_m, h0rd, stigma, mups2, OmL):
    z_th, E_th, _ = rhopsol_noH0(omega_m, omegar, OmL, stigma, mups2)
    return _ratios_from_E(jnp.interp(ZG, z_th, E_th), h0rd)


@jax.jit
def lcdm_ratios(omega_m, h0rd):
    E = jnp.sqrt(omega_m*(1+ZG)**3 + omegar*(1+ZG)**4 + (1-omega_m-omegar))
    return _ratios_from_E(E, h0rd)


dv_ref, dm_ref = (np.asarray(v) for v in lcdm_ratios(om_bf, h0rd_bf))

d = np.load(CHAIN)
PARAMS = [p for p in ["omega_m", "h0rd", "stigma", "mups2", "OmL"]
          if p in d.files]                      # mups2 absent when fixed
print(f"chain: {os.path.basename(CHAIN)}  params={PARAMS}")
ns = NestedSamples({k: d[k] for k in PARAMS},
                   logL=d["logL"], logL_birth=d["logL_birth"])
w = np.asarray(ns.get_weights())
w = w/w.max()
n = len(d["logL"])

batched = jax.jit(jax.vmap(torc_ratios))
dv_rows, dm_rows = [], []
print(f"computing curves for {n} dead points...", flush=True)
for i in range(0, n, 512):
    sl = slice(i, min(i+512, n))
    args = []
    for k in ["omega_m", "h0rd", "stigma", "mups2", "OmL"]:
        args.append(jnp.array(d[k][sl]) if k in d.files
                    else jnp.full(sl.stop-sl.start, MUPS2_FIXED))
    rv, rm = batched(*args)
    dv_rows.append(np.asarray(rv))
    dm_rows.append(np.asarray(rm))
DV_all = np.vstack(dv_rows) - dv_ref
DM_all = np.vstack(dm_rows) - dm_ref

# --- MAX LIKELIHOOD: best sample, then an actual optimisation ----------------
ibf = int(np.argmax(d["logL"]))
best_sample = {k: float(d[k][ibf]) for k in PARAMS}
logl_sample = float(d["logL"][ibf])
print("\nmax-likelihood:")
print(f"  1. best dead point      logL = {logl_sample:.4f}")
print("     " + ", ".join(f"{k}={v:.5g}" for k, v in best_sample.items()))

names = list(PARAMS)
bounds = [PRIOR_BOUNDS[k] for k in names]


def neg_logl(x):
    p = {k: jnp.asarray(v) for k, v in zip(names, x)}
    v = float(torc_logl(p))
    return -v if np.isfinite(v) else 1e10


res = minimize(neg_logl, x0=[best_sample[k] for k in names], bounds=bounds,
               method="Nelder-Mead",
               options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000,
                        "maxfev": 20000})
best_opt = dict(zip(names, res.x))
logl_opt = -res.fun
gain = logl_opt - logl_sample
print(f"  2. Nelder-Mead          logL = {logl_opt:.4f}   "
      f"(gain {gain:+.4f}, {res.nfev} evaluations)")
print("     " + ", ".join(f"{k}={v:.5g}" for k, v in best_opt.items()))

opt_args = [best_opt.get(k, MUPS2_FIXED)
            for k in ["omega_m", "h0rd", "stigma", "mups2", "OmL"]]
rv, rm = torc_ratios(*[jnp.asarray(v) for v in opt_args])
DV_opt = np.asarray(rv) - dv_ref
DM_opt = np.asarray(rm) - dm_ref

# --- plot ---------------------------------------------------------------------
cmap = cmap_from_hex(COLOR)
fig, (ax_dv, ax_dm) = plt.subplots(2, 1, figsize=(6, 7.4), sharex=True)

F = lambda x, row: np.interp(x, A_PLOT, row)
for ax, rows, ylim in [(ax_dv, DV_all, (-0.75, 0.75)),
                       (ax_dm, DM_all, (-0.15, 0.15))]:
    y = np.linspace(*ylim, 300)
    plot_contours(F, A_PLOT, rows, ax=ax, weights=w, y=y, lines=False,
                  colors=cmap)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_ylim(*ylim)

# both curves, so the effect of optimising is visible
ax_dv.plot(A_PLOT, DV_all[ibf], color="grey", ls=":", lw=1.0,
           label=r"best sample ($\ln L=%.2f$)" % logl_sample)
ax_dv.plot(A_PLOT, DV_opt, color="k", ls="--", lw=1.2,
           label=r"max likelihood ($\ln L=%.2f$)" % logl_opt)
ax_dm.plot(A_PLOT, DM_all[ibf], color="grey", ls=":", lw=1.0)
ax_dm.plot(A_PLOT, DM_opt, color="k", ls="--", lw=1.2)

labels_dv = ["BGS", "LRG", "LRG", "LRG+ELG", "ELG", "QSO", r"Ly-$\alpha$"]
labels_dm = ["LRG", "LRG", "LRG+ELG", "ELG", "QSO", r"Ly-$\alpha$"]
dv_cols = [f"C{i}" for i in range(7)]
dm_cols = dv_cols[1:]
for a, r_, e, lb, c in zip(1/(1+zv_all), dv_res, dv_std, labels_dv, dv_cols):
    ax_dv.errorbar(a, r_, yerr=e, marker="+", markersize=7, linewidth=1.2,
                   color=c, label=lb, capsize=0, zorder=5)
for a, r_, e, lb, c in zip(1/(1+zm), dm_res, dm_std, labels_dm, dm_cols):
    ax_dm.errorbar(a, r_, yerr=e, marker="+", markersize=7, linewidth=1.2,
                   color=c, label=lb, capsize=0, zorder=5)

ax_dv.set_ylabel(r"$D_\mathrm{V}/(r_d z^{2/3}) - (\cdot)_{\Lambda\mathrm{CDM}}$")
ax_dm.set_ylabel(r"$D_\mathrm{M}/(z D_H) - (\cdot)_{\Lambda\mathrm{CDM}}$")
ax_dm.set_xlabel(r"$a$")
ax_dv.set_xlim(A_PLOT[0], 1.0)
ax_dv.legend(fontsize=6.5, ncol=2)
ax_dm.legend(fontsize=6.5, ncol=2)

note = (r"\textbf{max-likelihood curve:} Nelder-Mead on $\ln L$, started from "
        r"the best of %d dead points, bounded to the prior box." "\n"
        r"$\ln L$: %.3f (best sample) $\rightarrow$ %.3f (optimised), "
        r"gain %+.3f in %d evaluations." "\n"
        r"$\Lambda$CDM reference curve is fitted the same way "
        r"($\chi^2$, Nelder-Mead)."
        % (n, logl_sample, logl_opt, gain, res.nfev))
fig.text(0.5, -0.015, note, ha="center", va="top", fontsize=6.5)
fig.suptitle(r"TorC posterior predictive vs DESI DR2 BAO (1000 live)",
             fontsize=11, y=0.995)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(here, "..", "plots",
                             f"torc_fgivenx_1000.{ext}"), bbox_inches="tight")
print("\nsaved plots/torc_fgivenx_1000.{pdf,png}")
