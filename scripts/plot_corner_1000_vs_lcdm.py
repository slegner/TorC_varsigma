"""TorC vs flat LCDM at MATCHED 1000 live points -- DESI DR2 BAO.

Both chains use the same data, covariance, distance formulas and priors on
the shared parameters, and the same sampler settings, so the evidences are
directly comparable.  LCDM has only (omega_m, h0rd), so it appears in those
panels only -- same convention as plot_corner_vs_lcdm.py.

Chains:
    chains/torc_ns_julia_1000.npz   scripts/run_ns_julia_1000.py  (1183 s)
    chains/lcdm_ns_1000.npz         LCDM at 1000 live             (9.5 s)
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


def load(fname, keep=None):
    d = np.load(os.path.join(CHAINS, fname))
    ps = [p for p in _ALL if p in d.files and (keep is None or p in keep)]
    return NestedSamples({k: d[k] for k in ps}, logL=d["logL"],
                         logL_birth=d["logL_birth"],
                         labels=[LABELS[k] for k in ps]), ps


torc, PARAMS = load("torc_ns_julia_1000.npz")
lcdm, _ = load("lcdm_ns_1000.npz", keep=["omega_m", "h0rd"])

KDE_KW = dict(
    kinds=dict(diagonal="hist_1d", lower="kde_2d", upper="scatter_2d"),
    diagonal_kwargs=dict(bins=20),
    upper_kwargs=dict(rasterized=True, zorder=-1),
    alpha=0.6,
)

fig, axes = make_2d_axes(PARAMS, figsize=(8.5, 8.5), upper=True)
torc.plot_2d(axes, c="#DC267F", label=r"TorC (4 params)", **KDE_KW)
lcdm.plot_2d(axes, c="#648FFF", label=r"$\Lambda$CDM (2 params)", **KDE_KW)

lzt, lzl = torc.logZ(1000), lcdm.logZ(1000)
d = lzl.mean() - lzt.mean()
sd = float(np.sqrt(lzt.std() ** 2 + lzl.std() ** 2))
lines = [
    f"TorC:          $\\ln Z = {lzt.mean():.3f} \\pm {lzt.std():.3f}$, "
    f"$D_{{\\mathrm{{KL}}}}={torc.D_KL():.2f}$",
    f"$\\Lambda$CDM: $\\ln Z = {lzl.mean():.3f} \\pm {lzl.std():.3f}$, "
    f"$D_{{\\mathrm{{KL}}}}={lcdm.D_KL():.2f}$",
    f"$\\Delta \\ln Z = {d:+.3f} \\pm {sd:.3f}$  "
    f"(Bayes factor ${np.exp(d):.2f}$ : 1)",
    r"$|\Delta\ln Z| < 1$: \textbf{inconclusive} (Jeffreys)",
]
fig.text(0.98, 0.02, "\n".join(lines), transform=fig.transFigure,
         fontsize=7.5, va="bottom", ha="right", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))
h, l = axes.iloc[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="center right", bbox_to_anchor=(1.20, 0.5), fontsize=8)
fig.suptitle(r"TorC vs $\Lambda$CDM --- DESI DR2 BAO, 1000 live points each",
             fontsize=11, y=0.99)

out = os.path.join(PLOT_DIR, "torc_vs_lcdm_1000live_corner")
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, bbox_inches="tight",
                dpi=300 if ext == "png" else None)
plt.close(fig)
print("saved plots/torc_vs_lcdm_1000live_corner.{pdf,png}\n")

print(f"ln Z  TorC = {lzt.mean():+.4f} +- {lzt.std():.4f}   (4 params)")
print(f"ln Z  LCDM = {lzl.mean():+.4f} +- {lzl.std():.4f}   (2 params)")
print(f"  Delta ln Z (LCDM - TorC) = {d:+.4f} +- {sd:.4f}  ({abs(d)/sd:.2f} sigma)")
print(f"  Bayes factor             = {np.exp(d):.2f} : 1 favouring LCDM")
scale = ("inconclusive ('bare mention')" if abs(d) < 1 else
         "substantial" if abs(d) < 2.5 else "strong")
print(f"  Jeffreys                 = {scale}\n")

print("shared-parameter constraints:")
print(f"{'param':10s} {'TorC':>24s} {'LCDM':>24s} {'ratio':>8s}")
for p in ("omega_m", "h0rd"):
    m1, s1 = torc[p].mean(), torc[p].std()
    m2, s2 = lcdm[p].mean(), lcdm[p].std()
    print(f"{p:10s} {m1:13.4f} +- {s1:8.4f} {m2:13.4f} +- {s2:8.4f} "
          f"{s1/s2:7.1f}x")
print("\n(TorC's extra freedom in varsigma/OmL loosens omega_m; it pays an")
print(" Occam penalty for 2 extra parameters without fitting the 13 BAO")
print(" points any better.)")
