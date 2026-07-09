"""Corner plots for the noH0 nested sampling run (chains/torc_ns_noH0.npz):
one solo, one overlaid on the calibrated TorC run for direct comparison of
the two anchoring conventions.  Style matches jax_tutorial/step5_corner_plot.py.
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from anesthetic import NestedSamples, make_2d_axes

here     = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(here, "..", "plots")

PARAMS = ["omega_m", "h0rd", "stigma", "mups2", "OmL"]
LABELS = {
    "omega_m": r"$\Omega_\mathrm{m}$",
    "h0rd":    r"$H_0 r_\mathrm{d}$ [km/s]",
    "stigma":  r"$\varsigma$",
    "mups2":   r"$-\upsilon_2$",
    "OmL":     r"$\Omega_\Lambda$",
}
# note: under the noH0 anchoring omega_m/OmL are NOT today's fractions —
# same symbols kept for axis alignment, interpretation differs.

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

def load(chain_file):
    d = np.load(os.path.join(here, "..", "chains", chain_file))
    return NestedSamples({k: d[k] for k in PARAMS},
                         logL=d["logL"], logL_birth=d["logL_birth"],
                         labels=[LABELS[k] for k in PARAMS])

KDE_KW = dict(
    kinds=dict(diagonal="hist_1d", lower="kde_2d", upper="scatter_2d"),
    diagonal_kwargs=dict(bins=15),
    upper_kwargs=dict(rasterized=True, zorder=-1),
    alpha=0.6,
)

runs = [
    ("torc_ns_noH0.npz", r"TorC noH0 ($a_0=1$, today $:= a{=}1$)", "#DC267F"),
    ("torc_ns.npz",      r"TorC calibrated ($E{=}1$ at input fractions)", "#FFB000"),
]
samples = {f: load(f) for f, _, _ in runs}

def stats_line(label, s):
    return (f"{label}:  $\\ln Z = {s.logZ():.2f}$,  "
            f"$D_{{\\mathrm{{KL}}}} = {s.D_KL():.1f}$")

# --- plot 1: noH0 solo -------------------------------------------------------
fig, axes = make_2d_axes(PARAMS, figsize=(9, 9), upper=True)
f, lbl, c = runs[0]
samples[f].plot_2d(axes, c=c, label=lbl, **KDE_KW)
fig.text(0.98, 0.02, stats_line(lbl, samples[f]), transform=fig.transFigure,
         fontsize=7, va="bottom", ha="right", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85))
h, l = axes.iloc[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="center right", bbox_to_anchor=(1.14, 0.5), fontsize=8)
fig.suptitle("TorC noH0 nested sampling — DESI DR2 BAO", fontsize=11, y=1.0)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(PLOT_DIR, f"torc_ns_noH0_corner.{ext}"),
                bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("saved torc_ns_noH0_corner.{pdf,png}")

# --- plot 2: overlay of the two anchorings ----------------------------------
fig, axes = make_2d_axes(PARAMS, figsize=(9, 9), upper=True)
for f, lbl, c in runs:
    samples[f].plot_2d(axes, c=c, label=lbl, **KDE_KW)
lines = [stats_line(lbl, samples[f]) for f, lbl, _ in runs]
fig.text(0.98, 0.02, "\n".join(lines), transform=fig.transFigure,
         fontsize=7, va="bottom", ha="right", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85))
h, l = axes.iloc[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="center right", bbox_to_anchor=(1.22, 0.5), fontsize=8)
fig.suptitle("TorC anchoring comparison — DESI DR2 BAO", fontsize=11, y=1.0)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(PLOT_DIR, f"torc_ns_vs_noH0_corner.{ext}"),
                bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("saved torc_ns_vs_noH0_corner.{pdf,png}")

for f, lbl, _ in runs:
    s = samples[f]
    print(f"\n{lbl}: lnZ = {s.logZ():.3f}, D_KL = {s.D_KL():.2f}")
    for n in PARAMS:
        print(f"  {n:8s} = {s[n].mean():10.4f} +- {s[n].std():.4f}")
