"""fgivenx posterior-predictive plot for the noH0 TorC fit vs DESI DR2.

Two panels (same observables as the rainbow plots):
    D_V/(r_d z^{2/3}) - LCDM best fit      (7 points)
    D_M/(z D_H)       - LCDM best fit      (6 points)
with the DESI data + errors on top of fgivenx contours of the posterior
(1/2 sigma), and the max-likelihood curve dashed.

Model curves are precomputed for every dead point with jax (vmap, chunked);
fgivenx then just interpolates rows, weighted by the posterior weights.
"""
import os, sys
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

omegar = OMEGA_R_DEFAULT
c_kms  = 299792.458
COLOR  = "#DC267F"                       # noH0 pink

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
_dd = os.environ.get("DESIDR2_DIR",
                     os.path.join(here, "..", "data", "desidr2"))
data      = read_csv(os.path.join(_dd, "desidr2_mean.txt"),
                     header=None, index_col=None, sep=r"\s+", comment="#")
cov       = np.loadtxt(os.path.join(_dd, "desidr2_cov.txt"))
zs        = data.iloc[:, 0].to_numpy()
mean_data = data.iloc[:, 1].to_numpy()
types     = data.iloc[:, 2].to_numpy()
cov_inv   = np.linalg.inv(cov)

# --- LCDM best-fit reference (quad, as in the rainbow scripts) ---------------
def E_l(z, om): return np.sqrt(om*(1+z)**3 + omegar*(1+z)**4 + (1-om-omegar))
def dm_l(z, h0rd, om): return c_kms/h0rd*quad(lambda x: 1/E_l(x, om), 0, z)[0]
def dh_l(z, h0rd, om): return c_kms/h0rd/E_l(z, om)
def dv_l(z, h0rd, om): return (z*dm_l(z, h0rd, om)**2*dh_l(z, h0rd, om))**(1/3)

def chi2_l(p):
    h0rd, om = p
    if om <= 0 or om >= 1 or h0rd <= 0: return 1e10
    fns = {"DM_over_rs": dm_l, "DH_over_rs": dh_l, "DV_over_rs": dv_l}
    pred = [fns[t](z, h0rd, om) for z, t in zip(zs, types)]
    return (mean_data-pred) @ cov_inv @ (mean_data-pred)

h0rd_bf, om_bf = minimize(chi2_l, x0=[10145, 0.298], method="Nelder-Mead",
                          options={"xatol": 1e-6, "fatol": 1e-8,
                                   "maxiter": 10000}).x
print(f"LCDM best fit: H0rd={h0rd_bf:.1f}, Om={om_bf:.4f}")

# --- data residuals (identical construction to the rainbow scripts) ----------
np.random.seed(42)
samps = multivariate_normal(mean_data, cov).rvs(10_000)
dmx, dhx, dvx = (types == t for t in ("DM_over_rs", "DH_over_rs", "DV_over_rs"))
dm_s, dh_s, dv_s = samps[:, dmx], samps[:, dhx], samps[:, dvx]
zm, zv_bgs = zs[dmx], zs[dvx]
dv_der  = (zm*dm_s**2*dh_s)**(1/3)
zv_all  = np.concatenate([zv_bgs, zm])
dv_all  = np.hstack([dv_s, dv_der])
o       = np.argsort(zv_all); zv_all, dv_all = zv_all[o], dv_all[:, o]
dv_mean, dv_std = (dv_all/zv_all**(2/3)).mean(0), (dv_all/zv_all**(2/3)).std(0)
dv_res  = dv_mean - np.array([dv_l(z, h0rd_bf, om_bf)/z**(2/3) for z in zv_all])
dm_rat  = dm_s/(zm*dh_s)
dm_mean, dm_std = dm_rat.mean(0), dm_rat.std(0)
dm_res  = dm_mean - np.array(
    [dm_l(z, h0rd_bf, om_bf)/(z*dh_l(z, h0rd_bf, om_bf)) for z in zm])

# --- model curves on the plot grid, for every dead point ----------------------
A_PLOT = np.linspace(0.22, 0.995, 150)          # data live at a ~ 0.30-0.77
Z_PLOT = 1.0/A_PLOT - 1.0
ZG     = jnp.linspace(0.0, 4.0, 801)            # integration grid

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

d = np.load(os.path.join(here, "..", "chains", "torc_ns_noH0.npz"))
PARAMS = ["omega_m", "h0rd", "stigma", "mups2", "OmL"]
ns = NestedSamples({k: d[k] for k in PARAMS},
                   logL=d["logL"], logL_birth=d["logL_birth"])
w = np.asarray(ns.get_weights()); w = w/w.max()
n = len(d["logL"])
print(f"computing curves for {n} dead points...")

batched = jax.jit(jax.vmap(torc_ratios))
dv_rows, dm_rows = [], []
for i in range(0, n, 512):
    sl = slice(i, min(i+512, n))
    rv, rm = batched(*(jnp.array(d[k][sl]) for k in PARAMS))
    dv_rows.append(np.asarray(rv)); dm_rows.append(np.asarray(rm))
    print(f"  {sl.stop}/{n}", flush=True)
DV = np.vstack(dv_rows) - dv_ref                # residual curves
DM = np.vstack(dm_rows) - dm_ref
ibf = int(np.argmax(d["logL"]))
print(f"best fit (max logL={d['logL'][ibf]:.2f}): " +
      ", ".join(f"{k}={float(d[k][ibf]):.4g}" for k in PARAMS))

# --- the plot -----------------------------------------------------------------
cmap = cmap_from_hex(COLOR)
fig, (ax_dv, ax_dm) = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

F = lambda x, row: np.interp(x, A_PLOT, row)
for ax, rows, ylim in [(ax_dv, DV, (-0.75, 0.75)), (ax_dm, DM, (-0.15, 0.15))]:
    y = np.linspace(*ylim, 300)
    plot_contours(F, A_PLOT, rows, ax=ax, weights=w, y=y, lines=False,
                  colors=cmap)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_ylim(*ylim)

ax_dv.plot(A_PLOT, DV[ibf], color="k", ls="--", lw=1.0, label="max-likelihood")
ax_dm.plot(A_PLOT, DM[ibf], color="k", ls="--", lw=1.0)

labels_dv = ["BGS", "LRG", "LRG", "LRG+ELG", "ELG", "QSO", r"Ly-$\alpha$"]
labels_dm = ["LRG", "LRG", "LRG+ELG", "ELG", "QSO", r"Ly-$\alpha$"]
dv_cols = [f"C{i}" for i in range(7)]
dm_cols = dv_cols[1:]        # same tracer, same colour in both panels
                             # (BGS is DV-only, so the bottom starts at C1)
for a, r_, e, lb, c in zip(1/(1+zv_all), dv_res, dv_std, labels_dv, dv_cols):
    ax_dv.errorbar(a, r_, yerr=e, marker="+", markersize=7, linewidth=1.2,
                   color=c, label=lb, capsize=0, zorder=5)
for a, r_, e, lb, c in zip(1/(1+zm), dm_res, dm_std, labels_dm, dm_cols):
    ax_dm.errorbar(a, r_, yerr=e, marker="+", markersize=7, linewidth=1.2,
                   color=c, label=lb, capsize=0, zorder=5)

ax_dv.set_ylabel(r"$D_\mathrm{V}/(r_d z^{2/3}) - (D_\mathrm{V}/(r_d z^{2/3}))_{\Lambda\mathrm{CDM}}$")
ax_dm.set_ylabel(r"$D_\mathrm{M}/(z D_H) - (D_\mathrm{M}/(z D_H))_{\Lambda\mathrm{CDM}}$")
ax_dm.set_xlabel(r"$a$")
ax_dv.set_xlim(A_PLOT[0], 1.0)
ax_dv.legend(fontsize=7, ncol=2)
ax_dm.legend(fontsize=7, ncol=2)
fig.suptitle(r"TorC noH0 posterior predictive vs DESI DR2 BAO",
             fontsize=11, y=0.995)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(here, "..", "plots", f"torc_noH0_fgivenx_desi.{ext}"),
                bbox_inches="tight")
print("saved plots/torc_noH0_fgivenx_desi.{pdf,png}")
