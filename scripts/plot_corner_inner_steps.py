"""Corner plot: NUM_INNER_STEPS=25 (baseline) vs 13 (halved), both at
num_delete=25, same DESI DR2 BAO likelihood -- checks that halving
num_inner_steps doesn't change the posterior (per the blackjax-nss skill's
recommended validation) before adopting it as the default in run_ns.py.
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from anesthetic import NestedSamples, make_2d_axes

here     = os.path.dirname(os.path.abspath(__file__))
CHAINS   = os.path.join(here, "..", "chains")
PLOT_DIR = os.path.join(here, "..", "plots")

# mups2 is absent from chains produced with SAMPLE_MUPS2 = False;
# keep only the parameters the chain actually contains.
_ALL_PARAMS = ["omega_m", "h0rd", "stigma", "mups2", "OmL"]


def _available(chain_file):
    """Parameters actually present in a chain (mups2 is missing from runs
    made with SAMPLE_MUPS2 = False)."""
    d = np.load(os.path.join(CHAINS, chain_file))
    return [p for p in _ALL_PARAMS if p in d.files]


PARAMS = _available("torc_ns_inner25.npz")
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


inner25 = load("torc_ns_inner25.npz", PARAMS)
inner13 = load("torc_ns_inner13.npz", PARAMS)

KDE_KW = dict(
    kinds=dict(diagonal="hist_1d", lower="kde_2d", upper="scatter_2d"),
    diagonal_kwargs=dict(bins=15),
    upper_kwargs=dict(rasterized=True, zorder=-1),
    alpha=0.6,
)

fig, axes = make_2d_axes(PARAMS, figsize=(9, 9), upper=True)
inner25.plot_2d(axes, c="#785EF0", label=r"NUM\_INNER\_STEPS$=25$ (baseline)", **KDE_KW)
inner13.plot_2d(axes, c="#FFB000", label=r"NUM\_INNER\_STEPS$=13$ (halved)", **KDE_KW)

# bootstrap logZ uncertainties from each run's own samples -- the matched-
# condition way to judge whether the two logZ values actually differ, rather
# than comparing against a number quoted for a different configuration.
lz25 = inner25.logZ(1000)
lz13 = inner13.logZ(1000)
d_logZ = float(lz13.mean() - lz25.mean())
sig_diff = float(np.sqrt(lz25.std() ** 2 + lz13.std() ** 2))

lines = [
    f"inner=25:  $\\ln Z = {lz25.mean():.2f} \\pm {lz25.std():.2f}$,  "
    f"$D_{{\\mathrm{{KL}}}} = {inner25.D_KL():.1f}$",
    f"inner=13:  $\\ln Z = {lz13.mean():.2f} \\pm {lz13.std():.2f}$,  "
    f"$D_{{\\mathrm{{KL}}}} = {inner13.D_KL():.1f}$",
    f"$\\Delta \\ln Z = {d_logZ:+.2f} \\pm {sig_diff:.2f}$  "
    f"(${abs(d_logZ) / sig_diff:.1f}\\sigma$)",
]
fig.text(0.98, 0.02, "\n".join(lines), transform=fig.transFigure,
         fontsize=7, va="bottom", ha="right", family="monospace",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85))
h, l = axes.iloc[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="center right", bbox_to_anchor=(1.20, 0.5), fontsize=8)
fig.suptitle(r"TorC noH0 posterior: NUM\_INNER\_STEPS 25 vs 13 (num\_delete=25) — DESI DR2 BAO",
             fontsize=10, y=1.0)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(PLOT_DIR, f"torc_inner_steps_corner.{ext}"),
                bbox_inches="tight", dpi=300 if ext == "png" else None)
plt.close(fig)
print("saved plots/torc_inner_steps_corner.{pdf,png}")

print(f"\nln Z (inner=25) = {lz25.mean():.3f} +- {lz25.std():.3f}")
print(f"ln Z (inner=13) = {lz13.mean():.3f} +- {lz13.std():.3f}")
print(f"difference      = {d_logZ:+.3f} +- {sig_diff:.3f}  "
      f"({abs(d_logZ) / sig_diff:.2f} sigma)")
print("-> consistent with sampling noise" if abs(d_logZ) < 2 * sig_diff
      else "-> DIFFERENCE EXCEEDS 2 sigma, investigate before adopting inner=13")

print("\nposterior means +- std:")
print(f"{'param':10s} {'inner=25':>22s} {'inner=13':>22s} {'shift/sigma25':>14s}")
for name in PARAMS:
    m25, s25 = inner25[name].mean(), inner25[name].std()
    m13, s13 = inner13[name].mean(), inner13[name].std()
    print(f"{name:10s} {m25:12.4f} +- {s25:7.4f} {m13:12.4f} +- {s13:7.4f} "
          f"{(m13 - m25) / s25:+14.3f}")
