"""
CPL vs TorC (1/stigma^2 prior) vs LCDM against DESI DR2 BAO -- distance fits.

Overlays the CPL chains from `cpl_nss.py` on the TorC power-law-prior chain
from ../chains/torc_ns_julia_1000_powerlaw.npz, producing:

    output/compare_corner.pdf      shared-parameter posterior (h0rd, omega_m)
    output/compare_distances.pdf   fgivenx D_V/r_d, D_M/r_d, D_H/r_d relative
                                   to best-fit LCDM, with the DESI DR2 points

Only the distance observables are plotted -- these are what DESI measures and
what the likelihood sees.  No w(z) reconstruction.

IMPORTANT -- prior boxes
------------------------
log Z is only comparable between runs sharing a prior box, so this script
uses the CPL/LCDM chains run with `--prior-box torc`, which reproduces
../torc/priors.py (omega_m ~ U[0.01, 0.99], h0rd ~ U[3650, 18250]).
Generate them first:

    python cpl_nss.py --model lcdm --live 1000 --prior-box torc
    python cpl_nss.py --model cpl  --live 1000 --prior-box torc

Usage
-----
    source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
    cd distances/TorC/cpl
    python cpl_torc_compare.py
    python cpl_torc_compare.py --only distances --nsub 3000
"""

import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from fgivenx import plot_contours
from anesthetic import NestedSamples, make_2d_axes

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from cpl_nss import (
    load_as_nested_samples, set_prior_box, PARAMS, OMEGA_R, C_KMS,
    Z_DATA, MEAN_DATA, TYPE_DATA, COV_DATA,
)

# CPL/LCDM chains must come from the TorC-matched prior box.
set_prior_box("torc")

COLORS = {"cpl": "#0072B2", "torc": "#DC267F", "lcdm": "#111111"}
LEGEND = {"cpl": "CPL ($w_0w_a$CDM)",
          "torc": r"TorC ($1/\varsigma^2$ prior)",
          "lcdm": r"$\Lambda$CDM"}

Z_MAX = 2.5
ZG    = np.linspace(0.0, 4.0, 801)          # integration / E(z) grid
Z_BAO = np.linspace(0.05, Z_MAX, 120)       # fgivenx x-grid (DM/rd is 0/0 at 0)


# -- Locate the TorC repo -----------------------------------------------------

def _find_torc():
    here = os.path.dirname(os.path.abspath(__file__))
    # This package normally lives at <TorC>/cpl, so try the parent first.
    cands = [os.path.dirname(here)]
    d = here
    for _ in range(6):
        d = os.path.dirname(d)
        cands.append(os.path.join(d, "distances", "TorC"))
    cands.append("/Users/starfish/Desktop/PhD2025-26/distances/TorC")
    for c in cands:
        if os.path.isdir(os.path.join(c, "torc")):
            return os.path.abspath(c)
    raise FileNotFoundError("Could not locate distances/TorC")


TORC_DIR = _find_torc()
sys.path.insert(0, TORC_DIR)
from torc.theory import rhopsol_jax, OMEGA_R_DEFAULT      # noqa: E402
from torc.likelihood import MUPS2_FIXED                    # noqa: E402

assert abs(OMEGA_R_DEFAULT - OMEGA_R) < 1e-12, (
    f"radiation mismatch: TorC {OMEGA_R_DEFAULT} vs cpl_nss {OMEGA_R}")

TORC_CHAIN = os.path.join(TORC_DIR, "chains",
                          "torc_ns_julia_1000_powerlaw.npz")


# -- Distances ----------------------------------------------------------------

def _E_lcdm(z, om):
    return np.sqrt(om * (1 + z) ** 3 + OMEGA_R * (1 + z) ** 4
                   + (1 - om - OMEGA_R))


def _distances_from_E(E_grid, h0rd):
    """(DM/rd, DH/rd, DV/rd) on ZG.  DV is 0/0 at z=0; that point is unused."""
    inv_E = 1.0 / E_grid
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (inv_E[:-1] + inv_E[1:]) * np.diff(ZG))])
    dm = C_KMS / h0rd * cum
    dh = C_KMS / (h0rd * E_grid)
    with np.errstate(invalid="ignore", divide="ignore"):
        dv = (ZG * dm ** 2 * dh) ** (1.0 / 3.0)
        # The two combinations the TorC scripts plot.  Both are 0/0 at z = 0;
        # substitute their analytic limits (DM -> (c/h0rd) z, DH -> c/h0rd).
        dvn = dv / ZG ** (2.0 / 3.0)
        dmz = dm / (ZG * dh)
    dvn[0] = dh[0]
    dmz[0] = 1.0
    return dm, dh, dv, dvn, dmz


def _lcdm_best_fit():
    """(h0rd, omega_m) minimising chi2 -- the reference all ratios divide by."""
    cov_inv = np.linalg.inv(COV_DATA)
    types = np.array(TYPE_DATA)

    def chi2(p):
        h0rd, om = p
        if om <= 0 or om >= 1 or h0rd <= 0:
            return 1e10
        dm, dh, dv = _distances_from_E(_E_lcdm(ZG, om), h0rd)[:3]
        pred = np.where(types == "DM_over_rs", np.interp(Z_DATA, ZG, dm),
                        np.where(types == "DH_over_rs",
                                 np.interp(Z_DATA, ZG, dh),
                                 np.interp(Z_DATA, ZG, dv)))
        r = MEAN_DATA - pred
        return r @ cov_inv @ r

    res = minimize(chi2, x0=[10145.0, 0.298], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 10000})
    return float(res.x[0]), float(res.x[1])


# -- Per-model curve construction ---------------------------------------------

def _assemble(E, h0rd):
    n = E.shape[0]
    dm = np.empty((n, len(ZG)))
    dh, dv, dvn, dmz = (np.empty_like(dm) for _ in range(4))
    for i in range(n):
        dm[i], dh[i], dv[i], dvn[i], dmz[i] = _distances_from_E(E[i], h0rd[i])
    return {"DM": dm, "DH": dh, "DV": dv, "DVn": dvn, "DMz": dmz}


def cpl_curves(theta):
    """theta: (n, 4) array of [h0rd, omega_m, w0, wa]."""
    h0rd, om, w0, wa = (theta[:, i] for i in range(4))
    z = ZG[None, :]
    f_de = (1 + z) ** (3 * (1 + w0[:, None] + wa[:, None])) * \
        np.exp(-3 * wa[:, None] * z / (1 + z))
    E = np.sqrt(om[:, None] * (1 + z) ** 3 + OMEGA_R * (1 + z) ** 4
                + (1 - om[:, None] - OMEGA_R) * f_de)
    return _assemble(E, h0rd)


def lcdm_curves(theta):
    """theta: (n, 2) array of [h0rd, omega_m]."""
    h0rd, om = theta[:, 0], theta[:, 1]
    z = ZG[None, :]
    E = np.sqrt(om[:, None] * (1 + z) ** 3 + OMEGA_R * (1 + z) ** 4
                + (1 - om[:, None] - OMEGA_R))
    return _assemble(E, h0rd)


def torc_curves(theta, chunk=512):
    """theta: (n, 4) array of [h0rd, omega_m, OmL, stigma]; solves the ODE."""
    h0rd, om, OmL, stigma = (theta[:, i] for i in range(4))

    @jax.jit
    def one(omega_m, OmL_, stigma_):
        z_th, E_th, _ = rhopsol_jax(omega_m, OMEGA_R, OmL_, stigma_,
                                    MUPS2_FIXED)
        return z_th, E_th          # native log-a grid, fixed length

    batched = jax.jit(jax.vmap(one))
    n = len(h0rd)
    z_rows, E_rows = [], []
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        zz, EE = batched(jnp.array(om[sl]), jnp.array(OmL[sl]),
                         jnp.array(stigma[sl]))
        z_rows.append(np.asarray(zz))
        E_rows.append(np.asarray(EE))
        print(f"    TorC solves {sl.stop}/{n}", flush=True)
    z_nat, E_nat = np.vstack(z_rows), np.vstack(E_rows)

    E = np.empty((n, len(ZG)))
    for i in range(n):
        E[i] = np.interp(ZG, z_nat[i], E_nat[i])

    bad = ~np.isfinite(E).all(axis=1)
    if bad.any():
        print(f"    dropped {bad.sum()}/{n} TorC samples with a failed solve")
        E, h0rd = E[~bad], h0rd[~bad]
    return _assemble(E, h0rd)


# -- Chain loading ------------------------------------------------------------

def _resample(n, weights, nsub, seed=0):
    """Indices of `nsub` equally-weighted posterior draws (fgivenx speed)."""
    p = np.asarray(weights, dtype=float)
    p = p / p.sum()
    return np.random.default_rng(seed).choice(n, size=min(nsub, n),
                                              replace=True, p=p)


def load_all(output_dir="output", nsub=2000):
    out = {}

    for model in ("cpl", "lcdm"):
        ns = load_as_nested_samples(model, output_dir)
        theta = ns[PARAMS[model]].to_numpy()
        w = np.asarray(ns.get_weights())
        print(f"  {model}: {len(w)} dead points")
        idx = _resample(len(w), w, nsub)
        curves = (cpl_curves if model == "cpl" else lcdm_curves)(theta[idx])
        out[model] = (curves, None, ns)

    d = np.load(TORC_CHAIN)
    tp = ["h0rd", "omega_m", "OmL", "stigma"]
    ns_t = NestedSamples({k: d[k] for k in tp},
                         logL=d["logL"], logL_birth=d["logL_birth"],
                         labels={"h0rd": r"$H_0 r_d$ [km/s]",
                                 "omega_m": r"$\Omega_m$",
                                 "OmL": r"$\Omega_\Lambda$",
                                 "stigma": r"$\varsigma$"})
    w_t = np.asarray(ns_t.get_weights())
    print(f"  torc: {len(w_t)} dead points  (logZ = {float(d['logZ']):.3f}"
          f" +/- {float(d['logZ_err']):.3f})")
    # Subsample the parameters FIRST -- the ODE solve is the expensive part.
    idx = _resample(len(w_t), w_t, nsub)
    out["torc"] = (torc_curves(ns_t[tp].to_numpy()[idx]), None, ns_t)
    return out


# -- Plot helpers -------------------------------------------------------------

def _cmap(hex_color):
    return LinearSegmentedColormap.from_list(
        "c", [(1, 1, 1), to_rgb(hex_color)], N=256)


def _band(ax, rows, weights, ylim, color, x=Z_BAO, ny=200, alpha=0.65,
          in_a=False):
    """
    Overlay one model's fgivenx band.  Rows are always tabulated on ZG; `in_a`
    says the plot's x-axis is the scale factor, so the lookup converts
    a -> z = 1/a - 1 inside the function handed to fgivenx.  (Passing a z-grid
    as x while labelling the axis `a` silently plots the band against the wrong
    abscissa -- the data points and the bands then live on different scales.)
    """
    y = np.linspace(*ylim, ny)
    if in_a:
        F = lambda xx, row: np.interp(1.0 / xx - 1.0, ZG, row)   # noqa: E731
    else:
        F = lambda xx, row: np.interp(xx, ZG, row)               # noqa: E731
    plot_contours(F, x, rows, ax=ax, weights=weights, y=y,
                  colors=_cmap(color), alpha=alpha, lines=False,
                  rasterize_contours=True)


# -- Figures ------------------------------------------------------------------

def plot_corner(data, output_dir="output"):
    """Shared-parameter posterior: all three models on (h0rd, omega_m)."""
    params = ["h0rd", "omega_m"]
    fig, axes = make_2d_axes(params, figsize=(5.2, 5.2), facecolor="w")
    for m in ("lcdm", "torc", "cpl"):
        data[m][2].plot_2d(axes, alpha=0.75, label=LEGEND[m], color=COLORS[m])
    handles, labels = axes.iloc[1, 0].get_legend_handles_labels()
    axes.iloc[-1, 0].legend(handles, labels, bbox_to_anchor=(1.0, 2.35),
                            loc="lower center", fontsize=8)
    out = os.path.join(output_dir, "compare_corner.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_distances(data, ref, output_dir="output"):
    """D_V, D_M, D_H over r_d, as ratios to best-fit LCDM, with the DESI data."""
    h0rd_bf, om_bf = ref
    dm_ref, dh_ref, dv_ref = _distances_from_E(_E_lcdm(ZG, om_bf), h0rd_bf)[:3]
    sigma = np.sqrt(np.diag(COV_DATA))
    types = np.array(TYPE_DATA)

    specs = [("DV", dv_ref, "DV_over_rs", r"$(D_V/r_d)\,/\,\Lambda$CDM",
              (0.95, 1.06)),
             ("DM", dm_ref, "DM_over_rs", r"$(D_M/r_d)\,/\,\Lambda$CDM",
              (0.95, 1.08)),
             ("DH", dh_ref, "DH_over_rs", r"$(D_H/r_d)\,/\,\Lambda$CDM",
              (0.93, 1.08))]

    fig, axes = plt.subplots(3, 1, figsize=(6.6, 9.6), sharex=True)

    for ax, (key, ref_curve, dtype, ylabel, ylim) in zip(axes, specs):
        for m in ("cpl", "torc"):
            rows, w, _ = data[m]
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = rows[key] / ref_curve
            _band(ax, ratio, w, ylim, COLORS[m])
        ax.axhline(1.0, color="k", ls="--", lw=1.2)

        sel = types == dtype
        if sel.any():
            den = np.interp(Z_DATA[sel], ZG, ref_curve)
            ax.errorbar(Z_DATA[sel], MEAN_DATA[sel] / den,
                        yerr=sigma[sel] / den, fmt="o", color="k", ms=4,
                        capsize=2, lw=1.2, zorder=6)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)

    axes[0].set_xlim(0.05, Z_MAX)
    axes[-1].set_xlabel(r"$z$")
    axes[0].set_title(r"DESI DR2 BAO distances relative to best-fit "
                      r"$\Lambda$CDM")
    handles = [Patch(facecolor=COLORS[m], alpha=0.65, label=LEGEND[m])
               for m in ("cpl", "torc")]
    handles.append(Line2D([], [], color="k", marker="o", ls="none",
                          label="DESI DR2"))
    handles.append(Line2D([], [], color="k", ls="--",
                          label=r"$\Lambda$CDM best fit"))
    axes[0].legend(handles=handles, fontsize=8, loc="upper right", ncol=2)

    out = os.path.join(output_dir, "compare_distances.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _max_likelihood_curves(data, optimise=True):
    """
    Curves at each model's maximum-likelihood point.

    The argmax over nested-sampling dead points is only an approximation to
    the true maximum -- NS samples the posterior, it does not hill-climb, and
    along a flat degeneracy the best dead point can sit well away from the
    peak in parameter space even when its logL is close.  With `optimise` the
    dead point is used as the start of a Nelder-Mead climb on the same
    likelihood, so the dashed curve really is the max-likelihood one.

    For CPL the climb moves the parameters appreciably (the w0-wa degeneracy
    is flat) but buys only dlogL ~ 0.003; for LCDM it changes nothing.
    """
    from scipy.optimize import minimize as _minimize
    from cpl_nss import make_loglikelihood

    out = {}
    for m, curves_fn, names in (
            ("cpl", cpl_curves, PARAMS["cpl"]),
            ("torc", torc_curves, ["h0rd", "omega_m", "OmL", "stigma"])):
        ns = data[m][2]
        theta = ns[names].to_numpy()
        logL = np.asarray(ns.logL)
        best = theta[int(np.argmax(logL))]
        ll_best = float(logL.max())

        if optimise and m == "cpl":
            ll_fn = make_loglikelihood("cpl")

            def negll(x):
                p_ = {n: jnp.array(float(v)) for n, v in zip(names, x)}
                v = float(ll_fn(p_))
                return 1e10 if not np.isfinite(v) else -v

            r = _minimize(negll, best, method="Nelder-Mead",
                          options={"xatol": 1e-8, "fatol": 1e-10,
                                   "maxiter": 40000, "maxfev": 40000})
            if -r.fun > ll_best:
                best, ll_best = r.x, -r.fun

        print(f"    {m} max logL = {ll_best:.4f}  ("
              + ", ".join(f"{n}={v:.4g}" for n, v in zip(names, best)) + ")")
        out[m] = {k: v[0] for k, v in curves_fn(best[None, :]).items()}
        out[m]["logL"] = ll_best
    return out


def plot_torc_convention(data, ref, output_dir="output"):
    """
    The distance combination the TorC scripts plot
    (distances/TorC/scripts/plot_fgivenx.py), with CPL overlaid:

        D_V/(r_d z^{2/3})  -  LCDM best fit      (7 points)
        D_M/(z D_H)        -  LCDM best fit      (6 points)

    Residuals, not ratios, plotted against the scale factor a.  The data
    construction below is copied from that script so the points, error bars and
    tracer colours match figure-for-figure: the 6 derived D_V values come from
    Monte-Carlo sampling the full covariance rather than propagating by hand.
    """
    h0rd_bf, om_bf = ref
    a_plot = np.linspace(0.22, 0.995, 150)
    z_plot = 1.0 / a_plot - 1.0

    _, _, _, dvn_ref, dmz_ref = _distances_from_E(_E_lcdm(ZG, om_bf), h0rd_bf)

    # --- data residuals (identical construction to the TorC script) ----------
    types = np.array(TYPE_DATA)
    rng_state = np.random.get_state()
    np.random.seed(42)
    samps = multivariate_normal(MEAN_DATA, COV_DATA).rvs(10_000)
    np.random.set_state(rng_state)

    dmx, dhx, dvx = (types == t for t in
                     ("DM_over_rs", "DH_over_rs", "DV_over_rs"))
    dm_s, dh_s, dv_s = samps[:, dmx], samps[:, dhx], samps[:, dvx]
    zm, zv_bgs = Z_DATA[dmx], Z_DATA[dvx]

    dv_der = (zm * dm_s ** 2 * dh_s) ** (1 / 3)
    zv_all = np.concatenate([zv_bgs, zm])
    dv_all = np.hstack([dv_s, dv_der])
    o = np.argsort(zv_all)
    zv_all, dv_all = zv_all[o], dv_all[:, o]

    dv_mean = (dv_all / zv_all ** (2 / 3)).mean(0)
    dv_std = (dv_all / zv_all ** (2 / 3)).std(0)
    dv_res = dv_mean - np.interp(zv_all, ZG, dvn_ref)

    dm_rat = dm_s / (zm * dh_s)
    dm_mean, dm_std = dm_rat.mean(0), dm_rat.std(0)
    dm_res = dm_mean - np.interp(zm, ZG, dmz_ref)

    # --- figure ---------------------------------------------------------------
    fig, (ax_dv, ax_dm) = plt.subplots(2, 1, figsize=(6.4, 7.4), sharex=True)

    ml = _max_likelihood_curves(data)

    for ax, key, ref_curve, ylim in [
        (ax_dv, "DVn", dvn_ref, (-0.75, 0.75)),
        (ax_dm, "DMz", dmz_ref, (-0.15, 0.15)),
    ]:
        for m in ("cpl", "torc"):
            rows, w, _ = data[m]
            _band(ax, rows[key] - ref_curve, w, ylim, COLORS[m], x=a_plot,
                  in_a=True)
        for m in ("cpl", "torc"):
            resid = ml[m][key] - ref_curve
            ax.plot(a_plot, np.interp(z_plot, ZG, resid), color=COLORS[m],
                    ls="--", lw=1.1, zorder=7)
        ax.axhline(0.0, color="k", ls="--", lw=0.8)
        ax.set_ylim(*ylim)

    labels_dv = ["BGS", "LRG", "LRG", "LRG+ELG", "ELG", "QSO", r"Ly-$\alpha$"]
    labels_dm = labels_dv[1:]         # BGS is DV-only, so DM starts at C1
    cols = [f"C{i}" for i in range(7)]
    for a_, r_, e_, lb, c in zip(1 / (1 + zv_all), dv_res, dv_std,
                                 labels_dv, cols):
        ax_dv.errorbar(a_, r_, yerr=e_, marker="+", markersize=7, linewidth=1.2,
                       color=c, label=lb, capsize=0, zorder=5)
    for a_, r_, e_, lb, c in zip(1 / (1 + zm), dm_res, dm_std,
                                 labels_dm, cols[1:]):
        ax_dm.errorbar(a_, r_, yerr=e_, marker="+", markersize=7, linewidth=1.2,
                       color=c, label=lb, capsize=0, zorder=5)

    ax_dv.set_ylabel(r"$D_\mathrm{V}/(r_d z^{2/3}) - "
                     r"(D_\mathrm{V}/(r_d z^{2/3}))_{\Lambda\mathrm{CDM}}$")
    ax_dm.set_ylabel(r"$D_\mathrm{M}/(z D_H) - "
                     r"(D_\mathrm{M}/(z D_H))_{\Lambda\mathrm{CDM}}$")
    ax_dm.set_xlabel(r"$a$")
    ax_dv.set_xlim(a_plot[0], 1.0)

    model_handles = [Patch(facecolor=COLORS[m], alpha=0.65, label=LEGEND[m])
                     for m in ("cpl", "torc")]
    for m in ("cpl", "torc"):
        model_handles.append(
            Line2D([], [], color=COLORS[m], ls="--", lw=1.1,
                   label=rf"max likelihood ($\ln L = {ml[m]['logL']:.2f}$)"))
    ax_dv.legend(handles=model_handles
                 + ax_dv.get_legend_handles_labels()[0], fontsize=7, ncol=2)
    ax_dm.legend(handles=model_handles
                 + ax_dm.get_legend_handles_labels()[0], fontsize=7, ncol=2)
    fig.suptitle("CPL vs TorC posterior predictive, DESI DR2 BAO",
                 fontsize=11, y=0.995)
    fig.tight_layout()

    out = os.path.join(output_dir, "compare_torc_convention.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_evidences(data, nsamples=500):
    """
    Evidence table for all three runs, split into its two pieces.

    Nested sampling decomposes the evidence exactly as

        log Z = <log L>_P  -  D_KL

    i.e. how well the model can fit (the posterior-averaged log-likelihood)
    minus what it had to pay to do so (the information gained from prior to
    posterior -- the Occam factor, in nats).  A model can lose on evidence
    either by fitting worse or by needing more prior compression; splitting
    the two says which.  d_G is the Bayesian model dimensionality: the number
    of parameters the data actually constrains.

    All three runs share the TorC prior box, so the numbers are comparable.
    """
    stats = {}
    for m in ("lcdm", "torc", "cpl"):
        stats[m] = data[m][2].stats(nsamples=nsamples)

    print("\n=== Evidence decomposition: log Z = <log L>_P - D_KL ===")
    print("    (TorC prior box: omega_m U[0.01,0.99], h0rd U[3650,18250])\n")
    hdr = f"  {'model':<26} {'log Z':>16} {'<log L>_P':>16} {'D_KL':>15} {'d_G':>13}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    plain = {"lcdm": "LCDM", "torc": "TorC (1/stigma^2)", "cpl": "CPL (w0waCDM)"}
    for m in ("lcdm", "torc", "cpl"):
        st = stats[m]
        def pm(key):
            return f"{st[key].mean():7.2f} +/- {st[key].std():.2f}"
        print(f"  {plain[m]:<26} {pm('logZ'):>16} {pm('logL_P'):>16} "
              f"{pm('D_KL'):>15} {pm('d_G'):>13}")

    print("\n  Differences vs LCDM (positive favours the extension):")
    base = stats["lcdm"]
    for m in ("torc", "cpl"):
        st = stats[m]
        d_logz = st["logZ"] - base["logZ"]
        d_logl = st["logL_P"] - base["logL_P"]
        d_kl = st["D_KL"] - base["D_KL"]
        print(f"    {plain[m]:<24} "
              f"D logZ = {d_logz.mean():+6.2f} +/- {d_logz.std():.2f}   "
              f"D <logL> = {d_logl.mean():+6.2f} +/- {d_logl.std():.2f}   "
              f"D D_KL = {d_kl.mean():+6.2f} +/- {d_kl.std():.2f}")

    # Sanity: the decomposition must close on itself.
    print("\n  Check (log Z + D_KL - <log L>_P, should be 0):")
    for m in ("lcdm", "torc", "cpl"):
        st = stats[m]
        resid = (st["logZ"] + st["D_KL"] - st["logL_P"]).mean()
        print(f"    {plain[m]:<24} {resid:+.2e}")

    # The TorC chain also ships its own stored logZ; confirm we reproduce it.
    d = np.load(TORC_CHAIN)
    print(f"\n  TorC stored logZ = {float(d['logZ']):.3f} "
          f"+/- {float(d['logZ_err']):.3f}  "
          f"(recomputed here: {stats['torc']['logZ'].mean():.3f})")
    return stats


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="CPL vs TorC vs LCDM: DESI DR2 distance fits")
    p.add_argument("--output", default="output")
    p.add_argument("--nsub", type=int, default=2000,
                   help="posterior draws per model handed to fgivenx")
    p.add_argument("--only",
                   choices=["corner", "distances", "torc"], default=None)
    args = p.parse_args()

    print("Loading chains and precomputing curves ...")
    data = load_all(args.output, nsub=args.nsub)
    ref = _lcdm_best_fit()
    print(f"LCDM best fit: h0rd = {ref[0]:.1f}, omega_m = {ref[1]:.4f}")

    if args.only in (None, "corner"):
        plot_corner(data, args.output)
    if args.only in (None, "distances"):
        plot_distances(data, ref, args.output)
    if args.only in (None, "torc"):
        plot_torc_convention(data, ref, args.output)

    print_evidences(data)
