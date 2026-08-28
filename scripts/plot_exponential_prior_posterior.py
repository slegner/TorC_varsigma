"""Prior vs posterior for the three exponential-stigma-prior runs.

pi(s) = exp(-(s-s0)/lam)/lam  on [1.0001, 50],  for lam = 0.5, 1.0 and 2.0.

For each lam this makes a corner plot overlaying

    prior      -- NestedSamples.prior(), i.e. the same dead points reweighted
                  at beta = 0, so it is the prior the run actually sampled
    posterior  -- the run itself

Where the two coincide the data said nothing; where the posterior is narrower
the data constrained the parameter.  D_KL is that gain in nats.

The three runs differ ONLY in lam.  Same data, same likelihood, same seed,
same sampler settings.  So any difference between the posteriors is caused by
the prior alone -- and the point of the figure is how little difference there
is (stigma = 1.28, 1.36, 1.38) even though ln Z moves by 0.94 nats across
them.  The fit is robust; the evidence is not.

lam = 1.0 is the one that lands on exact parity with LCDM
(Delta ln Z = -0.023 +- 0.135, Bayes factor 1.02:1).

Outputs:
    plots/torc_prior_posterior_exp_lam0p5.{pdf,png}
    plots/torc_prior_posterior_exp_lam1.{pdf,png}
    plots/torc_prior_posterior_exp_lam2.{pdf,png}
    plots/torc_prior_posterior_exp_powerlaw.{pdf,png}

Run:
    python scripts/plot_exponential_prior_posterior.py
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
RUNS = [   # (short label, pi(s) in TeX, chain file, tag, colour)
    (r"$\lambda=0.5$", r"$\pi(\varsigma)\propto e^{-(\varsigma-1)/0.5}$",
     "torc_ns_julia_1000_exponential_lam0p5.npz", "lam0p5", "#DC267F"),
    (r"$\lambda=1.0$", r"$\pi(\varsigma)\propto e^{-(\varsigma-1)/1.0}$",
     "torc_ns_julia_1000_exponential_lam1.npz", "lam1", "#FE6100"),
    (r"$\lambda=2.0$", r"$\pi(\varsigma)\propto e^{-(\varsigma-1)/2.0}$",
     "torc_ns_julia_1000_exponential_lam2.npz", "lam2", "#009E73"),
    (r"power law", r"$\pi(\varsigma)=1/\varsigma^{2}$ (no free scale)",
     "torc_ns_julia_1000_powerlaw.npz", "powerlaw", "#111111"),
]

plt.rcParams.update({
    "text.usetex": True, "font.family": "cmtt", "mathtext.fontset": "cm",
    "font.size": 9, "axes.labelsize": 11, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
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


KDE_KW = dict(
    kinds=dict(diagonal="hist_1d", lower="kde_2d", upper="scatter_2d"),
    diagonal_kwargs=dict(bins=25),
    upper_kwargs=dict(rasterized=True, zorder=-1),
    alpha=0.55,
)

lcdm, _ = load("lcdm_ns_1000.npz", keep=["omega_m", "h0rd"])
lzl = lcdm.logZ(1000)

rows = []
for lab, pitex, fname, tag, colour in RUNS:
    post, PARAMS = load(fname)
    prior = post.prior()                 # beta = 0 reweighting
    lz = post.logZ(1000)
    d = lz.mean() - lzl.mean()
    sd = float(np.sqrt(lz.std()**2 + lzl.std()**2))

    # The heavy-tailed prior runs to stigma ~ 1e3.  Binning/KDE-ing over that
    # full range puts the whole plotted window inside the first bin, so the
    # prior renders as a flat top-hat instead of 1/stigma^2.  Restrict the
    # prior sample to the window BEFORE plotting -- the prior is separable, so
    # conditioning on stigma < SMAX leaves the other marginals untouched -- and
    # report the excluded mass in the caption.
    SMAX = 4.0
    pw = np.asarray(prior.get_weights()); ps = np.asarray(prior["stigma"])
    frac_out = 100.0 * pw[ps > SMAX].sum() / pw.sum()
    prior_win = prior[ps <= SMAX]

    fig, axes = make_2d_axes(PARAMS, figsize=(7.8, 7.8), upper=True)
    prior_win.plot_2d(axes, c="#999999",
                      label=r"prior (\texttt{.prior()}, $\beta=0$)", **KDE_KW)
    post.plot_2d(axes, c=colour,
                 label=rf"posterior, {lab}", **KDE_KW)
    lcdm.plot_2d(axes, c="#648FFF", label=r"$\Lambda$CDM posterior", **KDE_KW)

    for i, r in enumerate(PARAMS):
        for j, c in enumerate(PARAMS):
            ax = axes.iloc[i, j]
            if ax is None:
                continue
            if c == "stigma":
                ax.set_xlim(1.0, SMAX)
            if r == "stigma" and i != j:
                ax.set_ylim(1.0, SMAX)

    fav = "TorC" if d > 0 else r"$\Lambda$CDM"
    lines = [
        pitex,
        rf"$\ln Z = {lz.mean():.3f} \pm {lz.std():.3f}$",
        rf"$\Lambda$CDM: $\ln Z = {lzl.mean():.3f} \pm {lzl.std():.3f}$",
        rf"$\Delta\ln Z = {d:+.3f} \pm {sd:.3f}$",
        rf"Bayes factor ${np.exp(abs(d)):.2f}\!:\!1$ {fav} (inconclusive)",
        rf"$D_{{\mathrm{{KL}}}} = {post.D_KL():.2f}$ nats",
        rf"prior mass at $\varsigma>{SMAX:.0f}$: {frac_out:.1f}\%  (axis clipped)",
    ]
    fig.text(0.98, 0.02, "\n".join(lines), transform=fig.transFigure,
             fontsize=8, va="bottom", ha="right",
             bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="grey",
                       alpha=0.95))
    h, l = axes.iloc[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", bbox_to_anchor=(1.26, 0.5), fontsize=8)
    fig.suptitle(rf"TorC, $\varsigma$ prior {lab}: "
                 r"prior vs posterior --- DESI DR2 BAO, 1000 live",
                 fontsize=10.5, y=0.99)

    out = os.path.join(PLOT_DIR, f"torc_prior_posterior_exp_{tag}")
    for ext in ("pdf", "png"):
        fig.savefig(out + "." + ext, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"saved plots/torc_prior_posterior_exp_{tag}.{{pdf,png}}")

    rows.append((lab, post, prior, PARAMS, lz, d, sd))

# ---- numbers ------------------------------------------------------------
print(f"\nLCDM  ln Z = {lzl.mean():+.4f} +- {lzl.std():.4f}   "
      f"D_KL = {lcdm.D_KL():.2f}\n")
print(f"{'prior':>12s} {'ln Z':>20s} {'D_KL':>7s} {'Delta ln Z vs LCDM':>22s} {'Bayes':>18s}")
for lab, post, prior, PARAMS, lz, d, sd in rows:
    fav = "TorC" if d > 0 else "LCDM"
    lab_p = lab.replace("$", "").replace("\\lambda", "lam")
    print(f"{lab_p:>12s} {lz.mean():+11.4f} +- {lz.std():.4f} {post.D_KL():7.2f} "
          f"{d:+13.3f} +- {sd:.3f} {np.exp(abs(d)):11.2f}:1 {fav}")

print("\nhow much did DESI narrow each parameter?  (prior std / posterior std)")
hdr = f"{'param':10s}" + "".join(
    f"{lab.replace(chr(36),chr(32)).replace(chr(92)+'lambda','lam'):>26s}" for lab, *_ in rows)
print(hdr)
print(f"{'':10s}" + "".join(f"{'prior':>9s}{'post':>9s}{'shrink':>8s}" for _ in rows))
for p in rows[0][3]:
    line = f"{p:10s}"
    for lab, post, prior, PARAMS, *_ in rows:
        sp, spo = prior[p].std(), post[p].std()
        line += f"{sp:9.4f}{spo:9.4f}{sp/spo:7.2f}x"
    print(line)
print("\n(shrink ~ 1 means DESI BAO said essentially nothing about that parameter)")
