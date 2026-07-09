"""
STEP 5 — Corner plot and constraints from the nested sampling run.
==================================================================

Run me with:
    source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
    cd /Users/starfish/Desktop/PhD2025-26/distances/TorC
    python jax_tutorial/step5_corner_plot.py

Loads the dead points written by step 4 (chains/torc_ns.npz) into
anesthetic, prints the evidence and constraints, and saves the corner
plot to plots/torc_ns_corner.{pdf,png}.
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from anesthetic import NestedSamples, make_2d_axes

here     = os.path.dirname(__file__)
CHAIN    = os.path.join(here, "..", "chains", "torc_ns.npz")
PLOT_DIR = os.path.join(here, "..", "plots")
COLOR    = "#FFB000"                      # TorC colour (matches /corner-plot)

PARAMS = ["omega_m", "h0rd", "stigma", "mups2", "OmL"]
LABELS = {
    "omega_m": r"$\Omega_\mathrm{m}$",
    "h0rd":    r"$H_0 r_\mathrm{d}$ [km/s]",
    "stigma":  r"$\varsigma$",
    "mups2":   r"$-\upsilon_2$",
    "OmL":     r"$\Omega_\Lambda$",
}

plt.rcParams.update({
    "text.usetex": True, "font.family": "cmtt", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 11, "axes.titlesize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "lines.linewidth": 0.8,
})

d = np.load(CHAIN)
samples = NestedSamples({k: d[k] for k in PARAMS},
                        logL=d["logL"], logL_birth=d["logL_birth"],
                        labels=[LABELS[k] for k in PARAMS])

logZ, D_KL, logL_P = samples.logZ(), samples.D_KL(), samples.logL_P()
print(f"ln Z    = {logZ:.3f}")
print(f"D_KL    = {D_KL:.3f}")
print(f"<lnL>_P = {logL_P:.3f}")
print("\nposterior means +- std:")
for n in PARAMS:
    print(f"  {n:8s} = {samples[n].mean():10.4f} +- {samples[n].std():.4f}")

fig, axes = make_2d_axes(PARAMS, figsize=(9, 9), upper=True)
samples.plot_2d(
    axes, c=COLOR, label="TorC (DESI DR2 BAO)",
    kinds=dict(diagonal="hist_1d", lower="kde_2d", upper="scatter_2d"),
    diagonal_kwargs=dict(bins=15),
    upper_kwargs=dict(rasterized=True, zorder=-1),
    alpha=0.6,
)

fig.text(0.98, 0.02,
         (f"$\\ln Z = {logZ:.1f}$,  $D_{{\\mathrm{{KL}}}} = {D_KL:.1f}$,  "
          f"$\\langle\\ln\\mathcal{{L}}\\rangle_{{\\mathcal{{P}}}} = {logL_P:.1f}$"),
         transform=fig.transFigure, fontsize=7, va="bottom", ha="right",
         family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85))

handles, labels_leg = axes.iloc[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="center right",
           bbox_to_anchor=(1.12, 0.5), fontsize=8)
fig.suptitle("TorC nested sampling — DESI DR2 BAO", fontsize=11, y=1.0)

os.makedirs(PLOT_DIR, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(PLOT_DIR, f"torc_ns_corner.{ext}"),
                bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print(f"\nsaved -> {os.path.abspath(os.path.join(PLOT_DIR, 'torc_ns_corner'))}.{{pdf,png}}")
