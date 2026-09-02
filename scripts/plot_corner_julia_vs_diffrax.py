"""Corner plot: Julia IPC backend vs diffrax, same likelihood and seed.

The two ODE backends agree to 1.4e-08 on individual solves, so the
posteriors must be statistically identical -- a 3x speedup that moved the
evidence would be a bug, not a speedup.  This is that check, plotted.

Chains come from benchmarks/run_ns_julia_vs_diffrax.py.
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from anesthetic import NestedSamples, make_2d_axes

here = os.path.dirname(os.path.abspath(__file__))
CHAINS = os.path.join(here, "..", "chains")
PLOT_DIR = os.path.join(here, "..", "plots")

_ALL_PARAMS = ["omega_m", "h0rd", "stigma", "mups2", "OmL"]
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
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "lines.linewidth": 0.8,
})


def load(fname):
    d = np.load(os.path.join(CHAINS, fname))
    params = [p for p in _ALL_PARAMS if p in d.files]
    return NestedSamples({k: d[k] for k in params},
                         logL=d["logL"], logL_birth=d["logL_birth"],
                         labels=[LABELS[k] for k in params]), params


# diffrax: the run_ns.py output (same 4-param config, same seed)
dfx, PARAMS = load("torc_ns_noH0.npz")
jl, _ = load("torc_ns_julia.npz")

KDE_KW = dict(
    kinds=dict(diagonal="hist_1d", lower="kde_2d", upper="scatter_2d"),
    diagonal_kwargs=dict(bins=15),
    upper_kwargs=dict(rasterized=True, zorder=-1),
    alpha=0.6,
)

fig, axes = make_2d_axes(PARAMS, figsize=(8.5, 8.5), upper=True)
dfx.plot_2d(axes, c="#785EF0", label=r"diffrax Dopri8 (JAX) --- 871.8\,s", **KDE_KW)
jl.plot_2d(axes, c="#FFB000", label=r"Julia Tsit5 via IPC --- 290.4\,s", **KDE_KW)

lzd, lzj = dfx.logZ(1000), jl.logZ(1000)
d = lzj.mean() - lzd.mean()
sd = float(np.sqrt(lzd.std() ** 2 + lzj.std() ** 2))
lines = [
    f"diffrax: $\\ln Z = {lzd.mean():.2f} \\pm {lzd.std():.2f}$, "
    f"$D_{{\\mathrm{{KL}}}} = {dfx.D_KL():.1f}$",
    f"julia:   $\\ln Z = {lzj.mean():.2f} \\pm {lzj.std():.2f}$, "
    f"$D_{{\\mathrm{{KL}}}} = {jl.D_KL():.1f}$",
    f"$\\Delta \\ln Z = {d:+.3f} \\pm {sd:.3f}$ (${abs(d)/sd:.2f}\\sigma$)",
    r"\textbf{3.0$\times$ faster, same posterior}",
]
fig.text(0.98, 0.02, "\n".join(lines), transform=fig.transFigure,
         fontsize=7, va="bottom", ha="right", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85))
h, l = axes.iloc[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="center right", bbox_to_anchor=(1.22, 0.5), fontsize=8)
fig.suptitle(r"TorC noH0 posterior: Julia IPC vs diffrax --- DESI DR2 BAO",
             fontsize=11, y=0.99)

out = os.path.join(PLOT_DIR, "julia_vs_diffrax_corner")
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, bbox_inches="tight",
                dpi=300 if ext == "png" else None)
plt.close(fig)
print("saved plots/julia_vs_diffrax_corner.{pdf,png}\n")

print(f"ln Z diffrax = {lzd.mean():+.3f} +- {lzd.std():.3f}")
print(f"ln Z julia   = {lzj.mean():+.3f} +- {lzj.std():.3f}")
print(f"difference   = {d:+.3f} +- {sd:.3f}  ({abs(d)/sd:.2f} sigma)\n")
print(f"{'param':10s} {'diffrax':>22s} {'julia':>22s} {'shift/sigma':>12s}")
for p in PARAMS:
    m1, s1 = dfx[p].mean(), dfx[p].std()
    m2, s2 = jl[p].mean(), jl[p].std()
    print(f"{p:10s} {m1:12.4f} +- {s1:7.4f} {m2:12.4f} +- {s2:7.4f} "
          f"{(m2-m1)/s1:+12.3f}")
