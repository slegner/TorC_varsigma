"""Prior vs posterior for the 1000-live TorC run -- how much did DESI learn?

anesthetic's NestedSamples.prior() re-weights the same dead points at
infinite temperature (beta = 0), recovering the prior the run started from.
Overlaying it on the posterior makes the information gain visible: where the
two coincide the data said nothing, where the posterior is narrower the data
constrained the parameter.

D_KL is exactly that gain in nats, so the plot and the number say the same
thing in different languages.

LCDM is shown for reference on the two shared parameters.

Chains:
    chains/torc_ns_julia_1000.npz
    chains/lcdm_ns_1000.npz
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


post, PARAMS = load("torc_ns_julia_1000.npz")
prior = post.prior()                      # beta = 0: the prior it started from
lcdm, _ = load("lcdm_ns_1000.npz", keep=["omega_m", "h0rd"])

KDE_KW = dict(
    kinds=dict(diagonal="hist_1d", lower="kde_2d", upper="scatter_2d"),
    diagonal_kwargs=dict(bins=20),
    upper_kwargs=dict(rasterized=True, zorder=-1),
    alpha=0.55,
)

fig, axes = make_2d_axes(PARAMS, figsize=(8.5, 8.5), upper=True)
prior.plot_2d(axes, c="#999999", label=r"prior (\texttt{.prior()}, $\beta=0$)",
              **KDE_KW)
post.plot_2d(axes, c="#DC267F", label=r"TorC posterior (1000 live)", **KDE_KW)
lcdm.plot_2d(axes, c="#648FFF", label=r"$\Lambda$CDM posterior", **KDE_KW)

lz = post.logZ(1000)
lines = [
    f"$\\ln Z = {lz.mean():.3f} \\pm {lz.std():.3f}$",
    f"$D_{{\\mathrm{{KL}}}} = {post.D_KL():.2f}$ nats "
    r"$=$ prior $\rightarrow$ posterior compression",
    r"grey $=$ what we assumed, pink $=$ what we learnt",
]
fig.text(0.98, 0.02, "\n".join(lines), transform=fig.transFigure,
         fontsize=7.5, va="bottom", ha="right", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.9))
h, l = axes.iloc[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="center right", bbox_to_anchor=(1.24, 0.5), fontsize=8)
fig.suptitle(r"TorC: prior vs posterior --- DESI DR2 BAO, 1000 live points",
             fontsize=11, y=0.99)

out = os.path.join(PLOT_DIR, "torc_prior_posterior_corner")
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, bbox_inches="tight",
                dpi=300 if ext == "png" else None)
plt.close(fig)
print("saved plots/torc_prior_posterior_corner.{pdf,png}\n")

print(f"D_KL = {post.D_KL():.3f} nats  (total information gained)\n")
print("per-parameter: how much did the data narrow each one?")
print(f"{'param':10s} {'prior std':>14s} {'posterior std':>16s} {'shrink':>9s}")
for p in PARAMS:
    sp, spo = prior[p].std(), post[p].std()
    print(f"{p:10s} {sp:14.4f} {spo:16.4f} {sp/spo:8.2f}x")
print("\n(a shrink factor near 1 means DESI BAO said essentially nothing "
      "about that parameter)")
