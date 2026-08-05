"""Prior sensitivity: does the TorC-vs-LCDM verdict survive the stigma prior?

Three 1000-live TorC runs, identical except for the prior on stigma:

    exponential lam=0.5   pi(s) = exp(-(s-s0)/0.5)/0.5     mean s = 1.5
    uniform               pi(s) = const on [1.0001, 3]     mean s = 2.0
    exponential lam=2.0   pi(s) = exp(-(s-s0)/2.0)/2.0     mean s = 3.0

The exponentials are Occam priors anchored at the LCDM limit (s -> 1): they
penalise departures from LCDM in proportion to their size rather than
treating every departure out to an arbitrary wall as equally likely.

Two things the figure shows:

1. The Bayes factor CHANGES SIGN across the three.  That is not a
   contradiction -- it is what |Delta ln Z| < 1 means.  When the evidence is
   this weak the verdict is set by the prior, not by the data.

2. ln Z falls monotonically as the prior widens (-7.52, -8.07, -8.46).  That
   is the Occam factor, uncontaminated: the likelihood is identical in all
   three runs and the posterior lands at s ~ 1.3-1.4 every time, so the only
   thing that changed is how much prior volume sits where the likelihood is
   not.  At lam=2 the typical prior draw is s = 3 and 37% of the mass lies
   above it, so most of the prior is wasted and TorC pays for it.

Chains:
    chains/torc_ns_julia_1000.npz                        uniform
    chains/torc_ns_julia_1000_exponential_lam0p5.npz     exponential, lam=0.5
    chains/torc_ns_julia_1000_exponential_lam2.npz       exponential, lam=2.0
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
FLOOR = 1.0001

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


RUNS = [                      # (label, lambda or None for uniform, chain, colour)
    (r"exponential $\lambda=0.5$", 0.5,
     "torc_ns_julia_1000_exponential_lam0p5.npz", "#DC267F"),
    (r"uniform $[1,3]$", None, "torc_ns_julia_1000.npz", "#785EF0"),
    (r"exponential $\lambda=2.0$", 2.0,
     "torc_ns_julia_1000_exponential_lam2.npz", "#009E73"),
]
loaded = [(lab, lam, load(f)[0], c) for lab, lam, f, c in RUNS]
lcdm, _ = load("lcdm_ns_1000.npz", keep=["omega_m", "h0rd"])
uni = [s for lab, lam, s, c in loaded if lam is None][0]
PARAMS = _ALL

lzl = lcdm.logZ(1000)
stats = []
for lab, lam, smp, c in loaded:
    lz = smp.logZ(1000)
    d = lz.mean() - lzl.mean()
    sd = float(np.sqrt(lz.std()**2 + lzl.std()**2))
    stats.append(dict(lab=lab, lam=lam, s=smp, c=c, lz=lz, d=d, sd=sd))

fig = plt.figure(figsize=(11.5, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.28)

# ---- left: the stigma marginal under each prior, with the priors ----------
axL = fig.add_subplot(gs[0, 0])
s_grid = np.linspace(FLOOR, 4.0, 500)
for st in stats:
    if st["lam"] is None:
        pr = np.where(s_grid <= 3.0, 1 / (3.0 - FLOOR), 0.0)
    else:
        pr = np.exp(-(s_grid - FLOOR) / st["lam"]) / st["lam"]
    axL.plot(s_grid, pr, color=st["c"], ls="--", lw=1.3,
             label=f"prior: {st['lab']}")
for st in stats:
    w = np.asarray(st["s"].get_weights())
    axL.hist(np.asarray(st["s"]["stigma"]), bins=60, range=(FLOOR, 4.0),
             weights=w, density=True, histtype="step", lw=1.6,
             color=st["c"])
axL.text(0.97, 0.60, "dashed = prior\nsolid = posterior", transform=axL.transAxes,
         ha="right", va="top", fontsize=7.5)
axL.set_xlabel(r"$\varsigma$")
axL.set_ylabel("density")
axL.set_xlim(1.0, 4.0)
axL.legend(fontsize=7.5)
axL.set_title(r"$\varsigma$: prior (dashed) vs posterior (filled)", fontsize=10)

# ---- right: the evidence comparison --------------------------------------
axR = fig.add_subplot(gs[0, 1])
x = np.arange(len(stats))
vals = [st["lz"].mean() for st in stats]
errs = [st["lz"].std() for st in stats]
axR.errorbar(x, vals, yerr=errs, fmt="none", capsize=5, lw=1.6, color="k", zorder=3)
for xi, st in zip(x, stats):
    axR.plot([xi], [st["lz"].mean()], "o", ms=10, color=st["c"], zorder=4)
axR.axhline(lzl.mean(), color="#648FFF", ls="--", lw=1.2, zorder=1,
            label=rf"$\Lambda$CDM $= {lzl.mean():.3f}\pm{lzl.std():.3f}$")
axR.axhspan(lzl.mean() - lzl.std(), lzl.mean() + lzl.std(),
            color="#648FFF", alpha=0.15, zorder=0)
axR.set_xticks(x)
axR.set_xticklabels([st["lab"] for st in stats], fontsize=8)
axR.set_ylabel(r"$\ln Z$")
axR.set_title(r"Evidence vs prior width", fontsize=10)
axR.grid(axis="y", alpha=0.3, lw=0.5)
axR.legend(loc="lower left", fontsize=7.5)
for xi, st in zip(x, stats):
    fav = "TorC" if st["d"] > 0 else r"$\Lambda$CDM"
    axR.text(xi, st["lz"].mean() + st["lz"].std() + 0.06,
             rf"$\Delta={st['d']:+.2f}$" "\n" rf"{np.exp(abs(st['d'])):.2f}:1 {fav}",
             ha="center", va="bottom", fontsize=7.5, color=st["c"])
axR.set_ylim(min(vals) - 0.30, max(vals) + 0.55)

fig.suptitle(r"TorC vs $\Lambda$CDM is prior-dominated --- DESI DR2 BAO, "
             r"1000 live points", fontsize=11.5, y=0.99)
fig.text(0.5, -0.03,
         r"All three have $|\Delta\ln Z|<1$: \textbf{inconclusive} on the Jeffreys "
         r"scale.  The data do not distinguish the models, so the Bayes factor "
         r"is set by the prior on $\varsigma$ --- it even changes sign."
         "\n"
         r"Widening the prior \emph{lowers} the evidence: that is the Occam "
         r"factor.  At $\lambda=2$ the typical $\varsigma$ is 3 while the "
         r"posterior wants $\varsigma\approx1.4$, so most prior volume is wasted."
         "\n"
         r"No single number here is ``the'' Bayes factor.",
         ha="center", va="top", fontsize=8)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])

out = os.path.join(PLOT_DIR, "prior_sensitivity")
for ext in ("pdf", "png"):
    fig.savefig(out + "." + ext, bbox_inches="tight",
                dpi=300 if ext == "png" else None)
plt.close(fig)
print("saved plots/prior_sensitivity.{pdf,png}\n")

print(f"{'TorC prior':24s} {'ln Z':>18s} {'D_KL':>6s} "
      f"{'Delta lnZ':>16s} {'Bayes':>18s}")
print(f"{'LCDM':24s} {lzl.mean():+11.4f} +- {lzl.std():.4f} "
      f"{lcdm.D_KL():6.2f}")
for st in stats:
    fav = "TorC" if st["d"] > 0 else "LCDM"
    print(f"{st['lab']:24s} {st['lz'].mean():+11.4f} +- {st['lz'].std():.4f} "
          f"{st['s'].D_KL():6.2f} {st['d']:+8.3f} +- {st['sd']:.3f} "
          f"{np.exp(abs(st['d'])):8.2f}:1 {fav}")
    print(f"{'':24s}   stigma = {st['s']['stigma'].mean():.4f} +- "
          f"{st['s']['stigma'].std():.4f}")
print("\nwidening the prior LOWERS the evidence -- the Occam factor: prior "
      "volume\nplaced where the likelihood is not costs evidence.")
