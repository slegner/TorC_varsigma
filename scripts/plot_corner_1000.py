"""Corner plot of the 1000-live production posterior (Julia IPC backend),
overlaid on the 250-live exploration run.

More live points should sharpen ln Z without moving the physics, so the
contours are expected to sit on top of each other while the evidence error
roughly halves (sigma ~ sqrt(D_KL/N)).

Chains:
    chains/torc_ns_julia_1000.npz   scripts/run_ns_julia_1000.py
    chains/torc_ns_noH0.npz         scripts/run_ns.py (250 live)
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from anesthetic import NestedSamples, make_2d_axes

here = os.path.dirname(os.path.abspath(__file__))
CHAINS = os.path.join(here, "..", "chains")
PLOT_DIR = os.path.join(here, "..", "plots")

_ALL = ["omega_m", "h0rd", "stigma", "mups2", "OmL"]
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
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "lines.linewidth": 0.8,
})


def load(fname):
    d = np.load(os.path.join(CHAINS, fname))
    ps = [p for p in _ALL if p in d.files]
    return NestedSamples({k: d[k] for k in ps}, logL=d["logL"],
                         logL_birth=d["logL_birth"],
                         labels=[LABELS[k] for k in ps]), ps


big, PARAMS = load("torc_ns_julia_1000.npz")
small, _ = load("torc_ns_noH0.npz")

KDE_KW = dict(
    kinds=dict(diagonal="hist_1d", lower="kde_2d", upper="scatter_2d"),
    diagonal_kwargs=dict(bins=20),
    upper_kwargs=dict(rasterized=True, zorder=-1),
    alpha=0.6,
)

fig, axes = make_2d_axes(PARAMS, figsize=(8.5, 8.5), upper=True)
small.plot_2d(axes, c="#785EF0", label=r"250 live (exploration)", **KDE_KW)
big.plot_2d(axes, c="#DC267F", label=r"\textbf{1000 live (production)}", **KDE_KW)

lzb, lzs = big.logZ(1000), small.logZ(1000)
lines = [
    f"250 live:   $\\ln Z = {lzs.mean():.3f} \\pm {lzs.std():.3f}$",
    f"1000 live:  $\\ln Z = {lzb.mean():.3f} \\pm {lzb.std():.3f}$",
    f"$D_{{\\mathrm{{KL}}}} = {big.D_KL():.2f}$ nats",
    r"error $\to$ %.0f\%% of the 250-live value" % (100*lzb.std()/lzs.std()),
]
fig.text(0.98, 0.02, "\n".join(lines), transform=fig.transFigure,
         fontsize=7.5, va="bottom", ha="right", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85))
h, l = axes.iloc[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="center right", bbox_to_anchor=(1.20, 0.5), fontsize=8)
fig.suptitle(r"TorC noH0 posterior --- DESI DR2 BAO, 1000 live points",
             fontsize=11, y=0.99)

out = os.path.join(PLOT_DIR, "torc_1000live_corner")
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, bbox_inches="tight",
                dpi=300 if ext == "png" else None)
plt.close(fig)
print("saved plots/torc_1000live_corner.{pdf,png}\n")

print(f"ln Z  250 live = {lzs.mean():+.4f} +- {lzs.std():.4f}")
print(f"ln Z 1000 live = {lzb.mean():+.4f} +- {lzb.std():.4f}")
print(f"  error ratio  = {lzb.std()/lzs.std():.2f} (expect ~0.5 for 4x live)")
d = lzb.mean() - lzs.mean()
sd = float(np.sqrt(lzb.std()**2 + lzs.std()**2))
print(f"  shift        = {d:+.4f} +- {sd:.4f}  ({abs(d)/sd:.2f} sigma)\n")

print(f"{'param':10s} {'250 live':>22s} {'1000 live':>22s} {'shift/sig':>10s}")
for p in PARAMS:
    m1, s1 = small[p].mean(), small[p].std()
    m2, s2 = big[p].mean(), big[p].std()
    print(f"{p:10s} {m1:12.4f} +- {s1:7.4f} {m2:12.4f} +- {s2:7.4f} "
          f"{(m2-m1)/s1:+10.3f}")
