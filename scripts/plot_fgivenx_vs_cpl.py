"""fgivenx: the lambda fits vs CPL (w0waCDM) against DESI DR2 BAO.

    python scripts/plot_fgivenx_vs_cpl.py [--nsub 2000]

Overlays posterior-predictive bands for

    CPL (w0 wa CDM)                  cpl_torcbox_desi_dead.pkl
    TorC, no lambda      (4D)        torc_ns_julia_1000_powerlaw_nostigmawall.npz
    TorC + lambda        (5D)        torc_ns_julia_1000_lam_powerlaw.npz
    TorC + lambda, OmL=0 (4D)        torc_ns_julia_1000_lam_noOmL_powerlaw.npz

on D_V/r_d, D_M/r_d and D_H/r_d, each as a ratio to the best-fit LCDM, with
the DESI DR2 points.  Max-likelihood curves are OPTIMISED (Nelder-Mead from
the best dead point), not taken as the argmax over samples.

REUSE: the CPL machinery, the DESI data block and the distance/band helpers
are imported from the cpl-desi worktree's cpl_torc_compare.py rather than
duplicated, so the two comparisons cannot drift apart.  That module is
imported READ-ONLY -- nothing there is modified.

PRIOR BOXES: the CPL and LCDM chains used here are the `--prior-box torc`
variants, which reproduce distances/TorC/torc/priors.py.  The TorC baseline is
the POST-FIX chain (stigma sampling bound opened from 3 to STIGMA_MAX); the
older torc_ns_julia_1000_powerlaw.npz must not be mixed in.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import minimize
from anesthetic import NestedSamples

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
TORC_DIR = os.path.join(HERE, "..")
CPL_DIR = os.path.join(
    "/Users/starfish/Desktop/PhD2025-26/Stigma/.claude/worktrees/cpl-desi",
    "python")
if not os.path.isdir(CPL_DIR):
    raise SystemExit(f"CPL worktree not found at {CPL_DIR}")
sys.path.insert(0, TORC_DIR)
sys.path.insert(0, CPL_DIR)

import cpl_torc_compare as ctc          # noqa: E402  (runs set_prior_box("torc"))
from cpl_nss import load_as_nested_samples, PARAMS  # noqa: E402
from torc.theory_lam import rhopsol_jax as rhopsol_lam   # noqa: E402
from torc.likelihood import MUPS2_FIXED                  # noqa: E402
from torc.likelihood_lam import loglikelihood as logl_lam  # noqa: E402
from torc.likelihood import loglikelihood as logl_nolam    # noqa: E402

CPL_OUT = os.path.join(CPL_DIR, "output")
CHAINS = os.path.join(TORC_DIR, "chains")
PLOTS = os.path.join(TORC_DIR, "plots")

ZG = ctc.ZG
OMEGA_R = ctc.OMEGA_R

# label, chain file, colour, has_lam, has_OmL
TORC_RUNS = [
    (r"TorC, no $\lambda$",              "torc_ns_julia_1000_powerlaw_nostigmawall.npz",
     "#785EF0", False, True),
    (r"TorC $+\,\lambda$",               "torc_ns_julia_1000_lam_powerlaw.npz",
     "#DC267F", True, True),
    (r"TorC $+\,\lambda$, $\Omega_\Lambda=0$", "torc_ns_julia_1000_lam_noOmL_powerlaw.npz",
     "#FE6100", True, False),
]
CPL_COLOR = "#0072B2"


def torc_lam_curves(theta, chunk=512):
    """theta: (n, 5) columns [h0rd, omega_m, OmL, stigma, lam]."""
    h0rd, om, OmL, stigma, lam = (theta[:, i] for i in range(5))

    @jax.jit
    def one(omega_m, OmL_, stigma_, lam_):
        z_th, E_th, _ = rhopsol_lam(omega_m, OMEGA_R, OmL_, stigma_,
                                    MUPS2_FIXED, lam_)
        return z_th, E_th

    batched = jax.jit(jax.vmap(one))
    n = len(h0rd)
    z_rows, E_rows = [], []
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        zz, EE = batched(jnp.array(om[sl]), jnp.array(OmL[sl]),
                         jnp.array(stigma[sl]), jnp.array(lam[sl]))
        z_rows.append(np.asarray(zz))
        E_rows.append(np.asarray(EE))
        print(f"    solves {sl.stop}/{n}", flush=True)
    z_nat, E_nat = np.vstack(z_rows), np.vstack(E_rows)

    E = np.empty((n, len(ZG)))
    for i in range(n):
        E[i] = np.interp(ZG, z_nat[i], E_nat[i])
    bad = ~np.isfinite(E).all(axis=1)
    if bad.any():
        print(f"    dropped {bad.sum()}/{n} samples with a failed solve")
        E, h0rd = E[~bad], h0rd[~bad]
    return ctc._assemble(E, h0rd)


def load_torc(fname, has_lam, has_OmL, nsub):
    path = os.path.join(CHAINS, fname)
    if not os.path.exists(path):
        print(f"  [missing] {fname}")
        return None
    d = np.load(path)
    cols = [c for c in ("h0rd", "omega_m", "OmL", "stigma", "lam")
            if c in d.files]
    ns = NestedSamples({k: d[k] for k in cols}, logL=d["logL"],
                       logL_birth=d["logL_birth"])
    w = np.asarray(ns.get_weights())
    lz = ns.logZ(1000)
    print(f"  {fname}: {len(w)} dead points, lnZ = {lz.mean():.4f} "
          f"+- {lz.std():.4f}")
    idx = ctc._resample(len(w), w, nsub)
    th = ns[cols].to_numpy()[idx]
    # widen to the full [h0rd, omega_m, OmL, stigma, lam] signature
    full = np.zeros((th.shape[0], 5))
    full[:, 0] = th[:, cols.index("h0rd")]
    full[:, 1] = th[:, cols.index("omega_m")]
    full[:, 2] = th[:, cols.index("OmL")] if has_OmL else 0.0
    full[:, 3] = th[:, cols.index("stigma")]
    full[:, 4] = th[:, cols.index("lam")] if has_lam else 0.0
    return dict(curves=torc_lam_curves(full), ns=ns, cols=cols, d=d,
                has_lam=has_lam, has_OmL=has_OmL)


def optimised_best_fit(entry):
    """Nelder-Mead on logL from the best dead point.  Returns (curves, logL)."""
    ns, cols = entry["ns"], entry["cols"]
    theta = ns[cols].to_numpy()
    logL = np.asarray(ns.logL)
    x0 = theta[int(np.argmax(logL))]
    ll_start = float(logL.max())
    ll_fn = logl_lam if entry["has_lam"] else logl_nolam

    def negll(x):
        p = {n: jnp.asarray(float(v)) for n, v in zip(cols, x)}
        v = float(ll_fn(p))
        return 1e10 if not np.isfinite(v) else -v

    from torc.priors_lam import _BOUNDS_ALL
    bounds = [_BOUNDS_ALL[c] for c in cols]
    r = minimize(negll, x0, method="Nelder-Mead", bounds=bounds,
                 options={"xatol": 1e-8, "fatol": 1e-10,
                          "maxiter": 40000, "maxfev": 40000})
    best, ll = (r.x, -r.fun) if -r.fun > ll_start else (x0, ll_start)
    print(f"    max logL {ll_start:.4f} -> {ll:.4f} "
          f"(gain {ll - ll_start:+.4f}): "
          + ", ".join(f"{n}={v:.5g}" for n, v in zip(cols, best)))
    full = np.zeros((1, 5))
    full[0, 0] = best[cols.index("h0rd")]
    full[0, 1] = best[cols.index("omega_m")]
    full[0, 2] = best[cols.index("OmL")] if entry["has_OmL"] else 0.0
    full[0, 3] = best[cols.index("stigma")]
    full[0, 4] = best[cols.index("lam")] if entry["has_lam"] else 0.0
    return {k: v[0] for k, v in torc_lam_curves(full).items()}, ll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsub", type=int, default=2000,
                    help="posterior draws per model (ODE cost)")
    args = ap.parse_args()

    print("loading CPL (torc prior box):")
    ns_cpl = load_as_nested_samples("cpl", CPL_OUT)
    w_cpl = np.asarray(ns_cpl.get_weights())
    lz_cpl = ns_cpl.logZ(1000)
    print(f"  cpl: {len(w_cpl)} dead points, lnZ = {lz_cpl.mean():.4f} "
          f"+- {lz_cpl.std():.4f}")
    idx = ctc._resample(len(w_cpl), w_cpl, args.nsub)
    cpl_curves = ctc.cpl_curves(ns_cpl[PARAMS["cpl"]].to_numpy()[idx])

    print("loading TorC chains:")
    entries = []
    for label, fname, colour, has_lam, has_OmL in TORC_RUNS:
        e = load_torc(fname, has_lam, has_OmL, args.nsub)
        if e is not None:
            e.update(label=label, colour=colour)
            entries.append(e)

    print("optimised max-likelihood points:")
    for e in entries:
        e["bf"], e["bf_logL"] = optimised_best_fit(e)

    h0rd_bf, om_bf = ctc._lcdm_best_fit()
    print(f"LCDM reference best fit: h0rd={h0rd_bf:.1f}, Om={om_bf:.4f}")
    dm_ref, dh_ref, dv_ref = ctc._distances_from_E(
        ctc._E_lcdm(ZG, om_bf), h0rd_bf)[:3]

    sigma = np.sqrt(np.diag(ctc.COV_DATA))
    types = np.array(ctc.TYPE_DATA)
    specs = [("DV", dv_ref, "DV_over_rs", r"$(D_V/r_d)\,/\,\Lambda$CDM", (0.95, 1.06)),
             ("DM", dm_ref, "DM_over_rs", r"$(D_M/r_d)\,/\,\Lambda$CDM", (0.95, 1.08)),
             ("DH", dh_ref, "DH_over_rs", r"$(D_H/r_d)\,/\,\Lambda$CDM", (0.93, 1.08))]

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 10.0), sharex=True)
    for ax, (key, ref_curve, dtype, ylabel, ylim) in zip(axes, specs):
        with np.errstate(invalid="ignore", divide="ignore"):
            ctc._band(ax, cpl_curves[key] / ref_curve, None, ylim, CPL_COLOR)
            for e in entries:
                ctc._band(ax, e["curves"][key] / ref_curve, None, ylim,
                          e["colour"], alpha=0.45)
            for e in entries:
                ax.plot(ctc.Z_BAO,
                        np.interp(ctc.Z_BAO, ZG, e["bf"][key] / ref_curve),
                        color=e["colour"], ls="--", lw=1.3)
        ax.axhline(1.0, color="k", ls="--", lw=1.2)
        sel = types == dtype
        if sel.any():
            den = np.interp(ctc.Z_DATA[sel], ZG, ref_curve)
            ax.errorbar(ctc.Z_DATA[sel], ctc.MEAN_DATA[sel] / den,
                        yerr=sigma[sel] / den, fmt="o", color="k", ms=4,
                        capsize=2, lw=1.2, zorder=6)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)

    axes[0].set_xlim(0.05, ctc.Z_MAX)
    axes[-1].set_xlabel(r"$z$")
    axes[0].set_title(r"DESI DR2 BAO distances relative to best-fit "
                      r"$\Lambda$CDM --- CPL vs TorC $\pm\,\lambda$")
    handles = [Patch(facecolor=CPL_COLOR, alpha=0.65, label=r"CPL ($w_0w_a$CDM)")]
    handles += [Patch(facecolor=e["colour"], alpha=0.45, label=e["label"])
                for e in entries]
    handles += [Line2D([], [], color="k", ls="--", label="optimised best fit"),
                Line2D([], [], color="k", marker="o", ls="none", label="DESI DR2")]
    axes[0].legend(handles=handles, fontsize=7.5, loc="upper right", ncol=2)

    note = ("max-likelihood curves are Nelder-Mead optima started from the best "
            "dead point, bounded to the prior box\n"
            + ";  ".join(f"{e['label']}: $\\ln L={e['bf_logL']:.3f}$"
                         for e in entries))
    fig.text(0.5, 0.005, note, ha="center", va="top", fontsize=6.5)

    out = os.path.join(PLOTS, "fgivenx_lam_vs_cpl")
    for ext in ("pdf", "png"):
        fig.savefig(out + "." + ext, bbox_inches="tight",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"saved plots/fgivenx_lam_vs_cpl.{{pdf,png}}")

    print("\nlog-evidences (all in the TorC prior box):")
    print(f"  CPL (4D)                    {lz_cpl.mean():9.4f} +- {lz_cpl.std():.4f}")
    for e in entries:
        lz = e["ns"].logZ(1000)
        print(f"  {e['label']:28s}{lz.mean():9.4f} +- {lz.std():.4f}")


if __name__ == "__main__":
    main()
