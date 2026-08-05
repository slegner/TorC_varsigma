"""Prior sensitivity: does the TorC-vs-LCDM verdict survive the stigma prior?

Two 1000-live TorC runs, identical except for the stigma prior:

    uniform      pi(s) = const on [1.0001, 3]
    exponential  pi(s) = exp(-(s-1.0001)/0.5)/0.5  on [1.0001, inf)

The exponential is an Occam prior anchored at the LCDM limit: it penalises
departures from LCDM in proportion to their size, rather than treating every
departure out to an arbitrary wall as equally likely.

The point of the figure: the Bayes factor CHANGES SIGN between them.  That is
not a contradiction -- it is what |Delta ln Z| < 1 means.  When the evidence
is this weak the verdict is set by the prior, not by the data, and quoting
either number alone would be misleading.

Chains:
    chains/torc_ns_julia_1000.npz              uniform
    chains/torc_ns_julia_1000_exponential.npz  exponential
    chains/lcdm_ns_1000.npz
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from anesthetic import NestedSamples, make_2d_axes

here = os.path.dirname(os.path.abspath(__file__))
CHAINS = os.path.join(here, "..", "chains")
PLOT_DIR = os.path.join(here, "..", "plots")

_ALL = ["omega_m", "h0rd", "stigma", "OmL"]
LABELS = {
    "omega_m": r"$\Omega_\mathrm{m}$",
    "h0rd":    r"$H_0 r_\mathrm{d}$ [km/s]",
    "stigma":  r"$\varsigma$",
    "OmL":     r"$\Omega_\Lambda$",
}
LAM, FLOOR = 0.5, 1.0001

plt.rcParams.update({
    "text.usetex": True, "font.family": "cmtt", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 11, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150,
    "savefig.dpi": 300, "axes.linewidth": 0.8,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
})


def load(fname, keep=None):
    d = np.load(os.path.join(CHAINS, fname))
    ps = [p for p in _ALL if p in d.files and (keep is None or p in keep)]
    return NestedSamples({k: d[k] for k in ps}, logL=d["logL"],
                         logL_birth=d["logL_birth"],
                         labels=[LABELS[k] for k in ps]), ps


uni, PARAMS = load("torc_ns_julia_1000.npz")
exp_, _ = load("torc_ns_julia_1000_exponential.npz")
lcdm, _ = load("lcdm_ns_1000.npz", keep=["omega_m", "h0rd"])

lzu, lze, lzl = uni.logZ(1000), exp_.logZ(1000), lcdm.logZ(1000)
du = lzu.mean() - lzl.mean()
de = lze.mean() - lzl.mean()
sdu = float(np.sqrt(lzu.std()**2 + lzl.std()**2))
sde = float(np.sqrt(lze.std()**2 + lzl.std()**2))

fig = plt.figure(figsize=(11.5, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.28)

# ---- left: the stigma marginal under each prior, with the priors ----------
axL = fig.add_subplot(gs[0, 0])
s_grid = np.linspace(FLOOR, 3.0, 400)
axL.plot(s_grid, np.exp(-(s_grid - FLOOR) / LAM) / LAM, color="#DC267F",
         ls="--", lw=1.2, label=r"prior: exponential ($\lambda=0.5$)")
axL.plot(s_grid, np.full_like(s_grid, 1 / (3.0 - FLOOR)), color="#785EF0",
         ls="--", lw=1.2, label=r"prior: uniform $[1,3]$")
for smp, c, lab in ((uni, "#785EF0", "posterior (uniform)"),
                    (exp_, "#DC267F", "posterior (exponential)")):
    w = np.asarray(smp.get_weights())
    axL.hist(np.asarray(smp["stigma"]), bins=60, range=(FLOOR, 3.0),
             weights=w, density=True, histtype="stepfilled", alpha=0.45,
             color=c, label=lab)
axL.set_xlabel(r"$\varsigma$")
axL.set_ylabel("density")
axL.set_xlim(1.0, 3.0)
axL.legend(fontsize=7.5)
axL.set_title(r"$\varsigma$: prior (dashed) vs posterior (filled)", fontsize=10)

# ---- right: the evidence comparison --------------------------------------
axR = fig.add_subplot(gs[0, 1])
names = [r"TorC" "\n" r"uniform prior", r"$\Lambda$CDM", r"TorC" "\n" r"exponential prior"]
vals = [lzu.mean(), lzl.mean(), lze.mean()]
errs = [lzu.std(), lzl.std(), lze.std()]
cols = ["#785EF0", "#648FFF", "#DC267F"]
x = np.arange(3)
axR.errorbar(x, vals, yerr=errs, fmt="o", ms=9, capsize=5, lw=1.6,
             color="k", zorder=3, ls="none")
for xi, v, e, c in zip(x, vals, errs, cols):
    axR.plot([xi], [v], "o", ms=9, color=c, zorder=4)
axR.axhline(lzl.mean(), color="#648FFF", ls=":", lw=1.0, zorder=1)
axR.set_xticks(x)
axR.set_xticklabels(names, fontsize=8)
axR.set_ylabel(r"$\ln Z$")
axR.set_title(r"Evidence --- the verdict flips with the prior", fontsize=10)
axR.grid(axis="y", alpha=0.3, lw=0.5)

# annotate inside the axes (annotate() with data coords escaped the frame)
axR.text(0.03, 0.06,
         rf"uniform: $\Delta\ln Z={du:+.2f}\pm{sdu:.2f}$"
         "\n" rf"{np.exp(abs(du)):.2f}:1 for $\Lambda$CDM",
         transform=axR.transAxes, ha="left", va="bottom", fontsize=8,
         color="#785EF0",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#785EF0", alpha=0.9))
axR.text(0.97, 0.94,
         rf"exponential: $\Delta\ln Z={de:+.2f}\pm{sde:.2f}$"
         "\n" rf"{np.exp(abs(de)):.2f}:1 for TorC",
         transform=axR.transAxes, ha="right", va="top", fontsize=8,
         color="#DC267F",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DC267F", alpha=0.9))

fig.suptitle(r"TorC vs $\Lambda$CDM is prior-dominated --- DESI DR2 BAO, "
             r"1000 live points", fontsize=11.5, y=0.99)
fig.text(0.5, -0.03,
         r"Both $|\Delta\ln Z|<1$: \textbf{inconclusive} on the Jeffreys scale. "
         r"The data do not distinguish the models, so the sign of the Bayes "
         r"factor is set by the prior on $\varsigma$, not by DESI."
         "\n"
         r"Neither number should be quoted alone as ``the'' Bayes factor.",
         ha="center", va="top", fontsize=8)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])

out = os.path.join(PLOT_DIR, "prior_sensitivity")
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, bbox_inches="tight",
                dpi=300 if ext == "png" else None)
plt.close(fig)
print("saved plots/prior_sensitivity.{pdf,png}\n")

print(f"{'run':28s} {'ln Z':>18s} {'D_KL':>7s}")
for tag, lz, s in (("TorC (uniform prior)", lzu, uni),
                   ("TorC (exponential prior)", lze, exp_),
                   ("LCDM", lzl, lcdm)):
    print(f"{tag:28s} {lz.mean():+11.4f} +- {lz.std():.4f} {s.D_KL():7.2f}")
print()
print(f"uniform     : Delta lnZ = {du:+.3f} +- {sdu:.3f}  "
      f"({np.exp(abs(du)):.2f}:1 for {'TorC' if du>0 else 'LCDM'})")
print(f"exponential : Delta lnZ = {de:+.3f} +- {sde:.3f}  "
      f"({np.exp(abs(de)):.2f}:1 for {'TorC' if de>0 else 'LCDM'})")
print(f"\nprior change moved TorC ln Z by {lze.mean()-lzu.mean():+.3f} nats "
      f"and flipped the preferred model")
print(f"\nstigma posterior: uniform {uni['stigma'].mean():.4f} +- "
      f"{uni['stigma'].std():.4f}   exponential {exp_['stigma'].mean():.4f} "
      f"+- {exp_['stigma'].std():.4f}")
