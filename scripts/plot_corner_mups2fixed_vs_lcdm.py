"""Corner plot: 4-parameter TorC (noH0, mups2 fixed to 4/3) vs flat LCDM,
same DESI DR2 BAO likelihood.  LCDM has only (omega_m, h0rd), so it appears
in those panels only.  Style matches plot_corner_vs_lcdm.py.
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from anesthetic import NestedSamples, make_2d_axes

here     = os.path.dirname(os.path.abspath(__file__))
CHAINS   = os.path.join(here, "..", "chains")
PLOT_DIR = os.path.join(here, "..", "plots")

PARAMS = ["omega_m", "h0rd", "stigma", "OmL"]
LABELS = {
    "omega_m": r"$\Omega_\mathrm{m}$",
    "h0rd":    r"$H_0 r_\mathrm{d}$ [km/s]",
    "stigma":  r"$\varsigma$",
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

def load(chain_file, params):
    d = np.load(os.path.join(CHAINS, chain_file))
    return NestedSamples({k: d[k] for k in params},
                         logL=d["logL"], logL_birth=d["logL_birth"],
                         labels=[LABELS[k] for k in params])

torc = load("torc_ns_noH0_mups2fixed.npz", PARAMS)
lcdm = load("lcdm_ns.npz", ["omega_m", "h0rd"])

KDE_KW = dict(
    kinds=dict(diagonal="hist_1d", lower="kde_2d", upper="scatter_2d"),
    diagonal_kwargs=dict(bins=15),
    upper_kwargs=dict(rasterized=True, zorder=-1),
    alpha=0.6,
)

fig, axes = make_2d_axes(PARAMS, figsize=(8, 8), upper=True)
torc.plot_2d(axes, c="#DC267F",
             label=r"TorC noH0, $-\upsilon_2 = 4/3$ (4 params)", **KDE_KW)
lcdm.plot_2d(axes, c="#648FFF", label=r"$\Lambda$CDM", **KDE_KW)

lines = [
    f"TorC:  $\\ln Z = {torc.logZ():.2f}$,  $D_{{\\mathrm{{KL}}}} = {torc.D_KL():.1f}$",
    f"$\\Lambda$CDM:  $\\ln Z = {lcdm.logZ():.2f}$,  $D_{{\\mathrm{{KL}}}} = {lcdm.D_KL():.1f}$",
]
fig.text(0.98, 0.02, "\n".join(lines), transform=fig.transFigure,
         fontsize=7, va="bottom", ha="right", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85))
h, l = axes.iloc[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="center right", bbox_to_anchor=(1.25, 0.5), fontsize=8)
fig.suptitle(r"TorC ($-\upsilon_2$ fixed) vs $\Lambda$CDM — DESI DR2 BAO",
             fontsize=11, y=1.0)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(PLOT_DIR, f"torc_mups2fixed_vs_lcdm_corner.{ext}"),
                bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("saved plots/torc_mups2fixed_vs_lcdm_corner.{pdf,png}")
print(f"ln B (LCDM vs TorC 4-param) = {lcdm.logZ() - torc.logZ():+.2f}")
