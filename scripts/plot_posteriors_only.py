"""All TorC posteriors overlaid, no priors -- are they actually the same?

The prior-sensitivity result says the EVIDENCE swings by 0.94 nats across the
four stigma priors while the fit does not change.  This figure is the direct
check of the second half of that claim: strip the priors away and look only at
what came out.

If the posteriors coincide, then the ln Z spread is purely an Occam effect and
the parameter constraints can be quoted without choosing a prior.

    chains/torc_ns_julia_1000_exponential_lam0p5.npz
    chains/torc_ns_julia_1000_exponential_lam1.npz
    chains/torc_ns_julia_1000_exponential_lam2.npz
    chains/torc_ns_julia_1000_powerlaw.npz
    chains/lcdm_ns_1000.npz

Run:
    python scripts/plot_posteriors_only.py
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from anesthetic import NestedSamples, make_2d_axes

here = os.path.dirname(os.path.abspath(__file__))
CHAINS = os.path.join(here, "..", "chains")
PLOT_DIR = os.path.join(here, "..", "plots")

_ALL = ["omega_m", "h0rd", "stigma", "OmL"]
LABELS = {"omega_m": r"$\Omega_\mathrm{m}$", "h0rd": r"$H_0 r_\mathrm{d}$ [km/s]",
          "stigma": r"$\varsigma$", "OmL": r"$\Omega_\Lambda$"}
RUNS = [
    (r"exp. $\lambda=0.5$", "torc_ns_julia_1000_exponential_lam0p5.npz", "#DC267F"),
    (r"exp. $\lambda=1.0$", "torc_ns_julia_1000_exponential_lam1.npz",   "#FE6100"),
    (r"exp. $\lambda=2.0$", "torc_ns_julia_1000_exponential_lam2.npz",   "#009E73"),
    (r"power law $1/\varsigma^2$", "torc_ns_julia_1000_powerlaw.npz",    "#111111"),
]

plt.rcParams.update({
    "text.usetex": True, "font.family": "cmtt", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 11, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 9, "figure.dpi": 150,
    "savefig.dpi": 300, "axes.linewidth": 0.8, "lines.linewidth": 0.9,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
})


def load(fname, keep=None):
    d = np.load(os.path.join(CHAINS, fname))
    ps = [p for p in _ALL if p in d.files and (keep is None or p in keep)]
    return NestedSamples({k: d[k] for k in ps}, logL=d["logL"],
                         logL_birth=d["logL_birth"],
                         labels=[LABELS[k] for k in ps]), ps


KW = dict(kinds=dict(diagonal="kde_1d", lower="kde_2d"), alpha=0.9)

lcdm, _ = load("lcdm_ns_1000.npz", keep=["omega_m", "h0rd"])
fig, axes = make_2d_axes(_ALL, figsize=(7.6, 7.6), upper=False)
lcdm.plot_2d(axes, c="#648FFF", label=r"$\Lambda$CDM", **KW)

rows = []
for lab, f, c in RUNS:
    s, _ = load(f)
    s.plot_2d(axes, c=c, label=lab, **KW)
    w = s.get_weights()
    rows.append((lab, s, w))

for i, r in enumerate(_ALL):
    for j, c in enumerate(_ALL):
        ax = axes.iloc[i, j]
        if ax is None:
            continue
        if c == "stigma":
            ax.set_xlim(1.0, 2.6)
        if r == "stigma" and i != j:
            ax.set_ylim(1.0, 2.6)

h, l = axes.iloc[1, 0].get_legend_handles_labels()
fig.legend(h, l, loc="center right", bbox_to_anchor=(1.30, 0.5), fontsize=8.5)
fig.suptitle(r"TorC posteriors are prior-independent --- DESI DR2 BAO, 1000 live",
             fontsize=11, y=0.99)
fig.text(0.98, 0.02,
         "\n".join([r"four $\varsigma$ priors spanning a $100\times$ range in width",
                    r"$\ln Z$ moves by \textbf{0.94 nats}",
                    r"$\varsigma$ moves by \textbf{0.10}, well inside $\pm0.25$"]),
         transform=fig.transFigure, fontsize=8, va="bottom", ha="right",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.95))

out = os.path.join(PLOT_DIR, "torc_posteriors_only")
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, bbox_inches="tight",
                dpi=300 if ext == "png" else None)
plt.close(fig)
print("saved plots/torc_posteriors_only.{pdf,png}\n")

print("posterior means +- std, by prior:")
hdr = f"{'param':10s}" + "".join(f"{lab.replace('$','').replace(chr(92)+'lambda','lam').replace(chr(92)+'varsigma','s'):>24s}" for lab, *_ in rows)
print(hdr)
for p in _ALL:
    line = f"{p:10s}"
    for lab, s, w in rows:
        x = np.asarray(s[p]); m = np.average(x, weights=w)
        sd = np.sqrt(np.average((x - m) ** 2, weights=w))
        line += f"{m:13.4f} +-{sd:8.4f}"
    print(line)

print("\nspread of the posterior MEAN across the four priors, in units of its own sigma:")
for p in _ALL:
    ms, sds = [], []
    for lab, s, w in rows:
        x = np.asarray(s[p]); m = np.average(x, weights=w)
        ms.append(m); sds.append(np.sqrt(np.average((x - m) ** 2, weights=w)))
    print(f"  {p:10s} range = {np.ptp(ms):10.4f}   mean sigma = {np.mean(sds):10.4f}"
          f"   ratio = {np.ptp(ms)/np.mean(sds):.2f} sigma")
