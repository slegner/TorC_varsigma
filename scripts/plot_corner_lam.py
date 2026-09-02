"""Corner plots for the Lambda fits vs the no-lambda baseline -- DESI DR2 BAO.

    python scripts/plot_corner_lam.py

All chains use the same data, covariance, distance formulas, sampler settings
and priors on the shared parameters, so the evidences are directly comparable.
Chains that are missing are skipped with a note.

IMPORTANT: the baseline here is chains/torc_ns_julia_1000_powerlaw_nostigmawall.npz,
NOT torc_ns_julia_1000_powerlaw.npz -- the latter predates the stigma
sampling-bound fix in torc/priors.py (it floored logL above stigma = 3) and
must not be differenced against these runs.

Produces:
    plots/lam_corner_vs_baseline.{pdf,png}   all three overlaid
    plots/lam_corner_lam_marginal.{pdf,png}  the lam posterior vs its prior
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from anesthetic import NestedSamples, make_2d_axes

here = os.path.dirname(os.path.abspath(__file__))
CHAINS = os.path.join(here, "..", "chains")
PLOT_DIR = os.path.join(here, "..", "plots")

_ALL = ["omega_m", "h0rd", "stigma", "OmL", "lam"]
LABELS = {
    "omega_m": r"$\Omega_\mathrm{m}$",
    "h0rd":    r"$H_0 r_\mathrm{d}$ [km/s]",
    "stigma":  r"$\varsigma$",
    "OmL":     r"$\Omega_\Lambda$",
    "lam":     r"$\lambda$",
}
LAM_PRIOR = (0.0, 1.5)

# key -> (legend label, chain file, colour)
# BAO-only, BAO+Pantheon+ and BAO+DESY5 variants of each model.  ln Z is only
# comparable WITHIN one data set, so never overlay across data sets to read a
# Bayes factor -- those overlays are for comparing PARAMETER constraints.
ALL_RUNS = {
    # --- DESI DR2 BAO alone ---
    "nolam":     ("no-$\\lambda$, BAO",
                  "torc_ns_julia_1000_powerlaw_nostigmawall.npz", "#648FFF"),
    "lam_OmL":   ("$\\lambda + \\Omega_\\Lambda$, BAO",
                  "torc_ns_julia_1000_lam_powerlaw.npz", "#DC267F"),
    "lam_noOmL": ("$\\lambda$ only, BAO",
                  "torc_ns_julia_1000_lam_noOmL_powerlaw.npz", "#FFB000"),
    # --- DESI DR2 BAO + Pantheon+ ---
    "pp_nolam":     ("no-$\\lambda$, BAO+Pantheon+",
                     "torc_ns_julia_1000_bao_pp_powerlaw.npz", "#785EF0"),
    "pp_lam_OmL":   ("$\\lambda + \\Omega_\\Lambda$, BAO+Pantheon+",
                     "torc_ns_julia_1000_bao_pp_lam_powerlaw.npz", "#DC267F"),
    "pp_lam_noOmL": ("$\\lambda$ only, BAO+Pantheon+",
                     "torc_ns_julia_1000_bao_pp_lam_noOmL_powerlaw.npz", "#FFB000"),
    # --- DESI DR2 BAO + DESY5 ---
    "d5_nolam":     ("no-$\\lambda$, BAO+DESY5",
                     "torc_ns_julia_1000_bao_desy5_powerlaw.npz", "#FE6100"),
    "d5_lam_OmL":   ("$\\lambda + \\Omega_\\Lambda$, BAO+DESY5",
                     "torc_ns_julia_1000_bao_desy5_lam_powerlaw.npz", "#DC267F"),
    "d5_lam_noOmL": ("$\\lambda$ only, BAO+DESY5",
                     "torc_ns_julia_1000_bao_desy5_lam_noOmL_powerlaw.npz", "#FFB000"),
}
DEFAULT_SELECT = ["nolam", "lam_OmL", "lam_noOmL"]

plt.rcParams.update({
    "text.usetex": True, "font.family": "cmtt", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 11, "axes.titlesize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "lines.linewidth": 0.8,
})

KDE_KW = dict(
    kinds=dict(diagonal="hist_1d", lower="kde_2d", upper="scatter_2d"),
    diagonal_kwargs=dict(bins=20),
    upper_kwargs=dict(rasterized=True, zorder=-1),
    alpha=0.6,
)


def load(fname):
    path = os.path.join(CHAINS, fname)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    ps = [p for p in _ALL if p in d.files]
    s = NestedSamples({k: d[k] for k in ps}, logL=d["logL"],
                      logL_birth=d["logL_birth"],
                      labels=[LABELS[k] for k in ps])
    return d, s, ps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--select", default=",".join(DEFAULT_SELECT),
                    help="comma-separated subset of " + "/".join(ALL_RUNS)
                         + ".  The FIRST one is the baseline that Delta ln Z "
                           "is measured against.")
    ap.add_argument("--tag", default=None,
                    help="filename suffix; defaults to the selection when it "
                         "is not the full set")
    args = ap.parse_args()

    keys = [k.strip() for k in args.select.split(",") if k.strip()]
    bad = [k for k in keys if k not in ALL_RUNS]
    if bad:
        raise SystemExit(f"unknown run(s) {bad}; choose from {list(ALL_RUNS)}")
    runs = [ALL_RUNS[k] for k in keys]
    if args.tag is not None:
        tag = args.tag
    elif keys == DEFAULT_SELECT:
        tag = ""
    else:
        tag = "_" + "_".join(keys)

    loaded = []
    for label, fname, colour in runs:
        got = load(fname)
        if got is None:
            print(f"[missing, skipped] {fname}")
            continue
        d, s, ps = got
        lz = s.logZ(1000)
        dset = ("BAO+DESY5" if "desy5" in fname else
                "BAO+Pantheon+" if "_pp" in fname else "BAO")
        loaded.append(dict(label=label, colour=colour, d=d, s=s, ps=ps,
                           dataset=dset,
                           lnZ=float(lz.mean()), err=float(lz.std())))
        print(f"loaded {fname}: params={ps} lnZ={lz.mean():.4f}+-{lz.std():.4f}")
    if not loaded:
        print("nothing to plot")
        return

    params = [p for p in _ALL if any(p in r["ps"] for r in loaded)]

    # ---- overlay corner ---------------------------------------------------
    fig, axes = make_2d_axes(params, figsize=(9.5, 9.5), upper=True)
    for r in loaded:
        r["s"].plot_2d(axes, c=r["colour"], label=r["label"], **KDE_KW)

    base = loaded[0]
    # ln Z is only comparable within one data set: the likelihood
    # normalisation differs between BAO, BAO+Pantheon+ and BAO+DESY5, so a
    # difference across them is meaningless (it would read as hundreds of
    # nats).  Only annotate Delta when every run shares a data set.
    same_data = len({r["dataset"] for r in loaded}) == 1
    lines = []
    for r in loaded:
        lines.append(f"{r['label']}: $\\ln Z = {r['lnZ']:.3f} \\pm {r['err']:.3f}$"
                     + ("" if (r is base or not same_data) else
                        "  ($\\Delta = %+.3f \\pm %.3f$)"
                        % (r["lnZ"] - base["lnZ"],
                           float(np.hypot(r["err"], base["err"])))))
    if same_data:
        lines.append(r"$|\Delta\ln Z| < 1$: \textbf{inconclusive} (Jeffreys)")
    else:
        lines.append(r"different data sets: $\ln Z$ \textbf{not} comparable")
    fig.text(0.98, 0.02, "\n".join(lines), transform=fig.transFigure,
             fontsize=7.5, va="bottom", ha="right", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))
    h, l = axes.iloc[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", bbox_to_anchor=(1.24, 0.5), fontsize=8)
    fig.suptitle(r"TorC with and without the $\lambda$ term "
                 r"--- DESI DR2 BAO, 1000 live points each", fontsize=11, y=0.99)
    out = os.path.join(PLOT_DIR, f"lam_corner_vs_baseline{tag}")
    for ext in ("pdf", "png"):
        fig.savefig(out + "." + ext, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"saved plots/lam_corner_vs_baseline{tag}.{{pdf,png}}")

    # ---- lam marginal vs its prior ---------------------------------------
    withlam = [r for r in loaded if "lam" in r["ps"]]
    if withlam:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        lo, hi = LAM_PRIOR
        ax.axhline(1.0 / (hi - lo), color="k", ls="--", lw=1.0,
                   label=r"prior $\mathcal{U}[0, 1.5]$")
        for r in withlam:
            lam = np.asarray(r["s"]["lam"])
            w = np.asarray(r["s"].get_weights())
            ax.hist(lam, bins=40, range=(lo, hi), weights=w, density=True,
                    histtype="step", lw=1.4, color=r["colour"], label=r["label"])
            m, sd = float(r["s"]["lam"].mean()), float(r["s"]["lam"].std())
            print(f"{r['label']}: lam = {m:.4f} +- {sd:.4f}"
                  f"   (prior mean {(lo+hi)/2:.4f} +- {(hi-lo)/np.sqrt(12):.4f})")
        ax.set_xlabel(LABELS["lam"])
        ax.set_ylabel("posterior density")
        ax.set_xlim(lo, hi)
        ax.legend(fontsize=7.5)
        ax.set_title(r"Is $\lambda$ constrained by the data?", fontsize=10)
        fig.tight_layout()
        out = os.path.join(PLOT_DIR, f"lam_corner_lam_marginal{tag}")
        for ext in ("pdf", "png"):
            fig.savefig(out + "." + ext, bbox_inches="tight")
        plt.close(fig)
        print(f"saved plots/lam_corner_lam_marginal{tag}.{{pdf,png}}")


if __name__ == "__main__":
    main()
