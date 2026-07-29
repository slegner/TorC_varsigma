"""Corner plot: mups2 sampled (5 params) vs mups2 fixed at 4/3 (4 params),
same DESI DR2 BAO likelihood.

mups2 is an exact background gauge parameter -- it rescales conformal time
(tau -> sqrt(mups2)*tau) and cancels out of E(z), verified to 1e-9 by
tests/test_theory.py.  So fixing it should leave every other marginal
unchanged, and the evidence should be identical (Z factorises over a flat
direction with a normalised prior).  This plot is that check.

The fixed run has no mups2 column, so it simply does not appear in the
mups2 panels -- same convention as plot_corner_vs_lcdm.py.
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from anesthetic import NestedSamples, make_2d_axes

here     = os.path.dirname(os.path.abspath(__file__))
CHAINS   = os.path.join(here, "..", "chains")
PLOT_DIR = os.path.join(here, "..", "plots")

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


def load(chain_file, params):
    d = np.load(os.path.join(CHAINS, chain_file))
    return NestedSamples({k: d[k] for k in params},
                         logL=d["logL"], logL_birth=d["logL_birth"],
                         labels=[LABELS[k] for k in params])


sampled = load("torc_ns_noH0.npz", PARAMS)
fixed   = load("torc_ns_noH0_mups2fixed.npz",
               ["omega_m", "h0rd", "stigma", "OmL"])   # no mups2 column

KDE_KW = dict(
    kinds=dict(diagonal="hist_1d", lower="kde_2d", upper="scatter_2d"),
    diagonal_kwargs=dict(bins=15),
    upper_kwargs=dict(rasterized=True, zorder=-1),
    alpha=0.6,
)

fig, axes = make_2d_axes(PARAMS, figsize=(9, 9), upper=True)
sampled.plot_2d(axes, c="#785EF0", label=r"$-\upsilon_2$ sampled (5 params)", **KDE_KW)
fixed.plot_2d(axes,   c="#FFB000", label=r"$-\upsilon_2$ fixed $=4/3$ (4 params)", **KDE_KW)

lz_s, lz_f = sampled.logZ(1000), fixed.logZ(1000)
lines = [
    f"sampled:  $\\ln Z = {lz_s.mean():.2f} \\pm {lz_s.std():.2f}$,  "
    f"$D_{{\\mathrm{{KL}}}} = {sampled.D_KL():.1f}$",
    f"fixed:    $\\ln Z = {lz_f.mean():.2f} \\pm {lz_f.std():.2f}$,  "
    f"$D_{{\\mathrm{{KL}}}} = {fixed.D_KL():.1f}$",
    r"gauge $\Rightarrow$ $\ln Z$ should be identical",
]
fig.text(0.98, 0.02, "\n".join(lines), transform=fig.transFigure,
         fontsize=7, va="bottom", ha="right", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85))
h, l = axes.iloc[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="center right", bbox_to_anchor=(1.22, 0.5), fontsize=8)
fig.suptitle(r"TorC noH0: $-\upsilon_2$ sampled vs fixed at $4/3$ — DESI DR2 BAO",
             fontsize=11, y=1.0)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(PLOT_DIR, f"torc_mups2_fixed_vs_sampled_corner.{ext}"),
                bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("saved plots/torc_mups2_fixed_vs_sampled_corner.{pdf,png}")

print(f"\nln Z sampled = {lz_s.mean():+.3f} +- {lz_s.std():.3f}")
print(f"ln Z fixed   = {lz_f.mean():+.3f} +- {lz_f.std():.3f}")
d = lz_f.mean() - lz_s.mean()
s = np.sqrt(lz_s.std() ** 2 + lz_f.std() ** 2)
print(f"difference   = {d:+.3f} +- {s:.3f}  ({abs(d)/s:.2f} sigma; expect 0)")

print("\nmarginals (mean +- std):")
print(f"{'param':10s} {'sampled':>22s} {'fixed':>22s} {'shift/sigma':>12s}")
for p in ["omega_m", "h0rd", "stigma", "OmL"]:
    ms, ss = sampled[p].mean(), sampled[p].std()
    mf, sf = fixed[p].mean(), fixed[p].std()
    print(f"{p:10s} {ms:12.4f} +- {ss:7.4f} {mf:12.4f} +- {sf:7.4f} "
          f"{(mf-ms)/ss:12.3f}")
