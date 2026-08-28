"""
CPL prior vs posterior -- how much did DESI DR2 BAO actually learn?

anesthetic's NestedSamples.prior() re-weights the same dead points at infinite
temperature (beta = 0), recovering the prior the run started from.  Overlaying
it on the posterior makes the information gain visible: where the two coincide
the data said nothing; where the posterior is narrower the data constrained it.
D_KL is that gain in nats, so the plot and the number say the same thing.

Same idea as ../scripts/plot_corner_prior_posterior.py, for the CPL run.

The CPL prior (TorC-matched box, `cpl_nss.py --prior-box torc`):

    h0rd    ~ U[3650, 18250]   km/s
    omega_m ~ U[0.01, 0.99]
    w0      ~ U[-3, 1]
    wa      ~ U[-3, 2]
    subject to the hard cut  w0 + wa < 0   (as in ../../common.py)

The cut is what makes the (w0, wa) prior panel a triangle rather than a
rectangle; it removes 22.5% of that rectangle.  Note w0 and wa are each
allowed positive on their own -- only their sum is bounded above.

Outputs
-------
    output/cpl_prior_posterior.pdf    all four CPL parameters
    output/cpl_prior_posterior_w.pdf  the (w0, wa) plane alone, with the cut

Usage
-----
    source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
    cd distances/TorC/cpl
    python cpl_prior_posterior.py
    python cpl_prior_posterior.py --box default
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from anesthetic import make_2d_axes

from cpl_nss import (load_as_nested_samples, set_prior_box, PARAMS,
                     PRIOR_BOXES, LABELS)

PRIOR_C = "#999999"
POST_C = "#0072B2"

KDE_KW = dict(
    kinds=dict(diagonal="hist_1d", lower="kde_2d", upper="scatter_2d"),
    diagonal_kwargs=dict(bins=20),
    upper_kwargs=dict(rasterized=True, zorder=-1, ncompress=2000),
    alpha=0.55,
)


def _prior_edges(ax, name, box, axis="x"):
    """Dotted lines at the uniform prior boundaries."""
    lo, width = PRIOR_BOXES[box][name]
    for v in (lo, lo + width):
        (ax.axvline if axis == "x" else ax.axhline)(
            v, color=PRIOR_C, ls=":", lw=0.8, zorder=0)


def plot_corner(box="torc", output_dir="output"):
    set_prior_box(box)
    post = load_as_nested_samples("cpl", output_dir)
    prior = post.prior()                 # beta = 0: the prior it started from

    params = PARAMS["cpl"]
    fig, axes = make_2d_axes(params, figsize=(8.2, 8.2), upper=True,
                             facecolor="w")
    prior.plot_2d(axes, c=PRIOR_C, label="prior ($\\beta=0$)", **KDE_KW)
    post.plot_2d(axes, c=POST_C, label="CPL posterior", **KDE_KW)

    handles, labels = axes.iloc[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)
    fig.suptitle(f"CPL prior vs posterior, DESI DR2 BAO "
                 f"({box} prior box)", y=1.05, fontsize=11)

    out = os.path.join(output_dir, "cpl_prior_posterior.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_w_plane(box="torc", output_dir="output"):
    """The (w0, wa) plane on its own, with the w0 + wa < 0 cut drawn."""
    set_prior_box(box)
    post = load_as_nested_samples("cpl", output_dir)
    prior = post.prior()

    w0lo, w0w = PRIOR_BOXES[box]["w0"]
    walo, waw = PRIOR_BOXES[box]["wa"]

    # NestedSamples rows are DEAD POINTS -- scattering them raw ignores the
    # nested-sampling weights and would show neither the prior nor the
    # posterior.  compress() draws equally-weighted samples from each.
    pri = prior.compress(4000)
    pos = post.compress(4000)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.scatter(pri["w0"], pri["wa"], s=1.5, color=PRIOR_C, alpha=0.30,
               rasterized=True, zorder=1)
    ax.scatter(pos["w0"], pos["wa"], s=1.5, color=POST_C, alpha=0.45,
               rasterized=True, zorder=2)

    # prior rectangle and the w0 + wa < 0 cut
    ax.add_patch(plt.Rectangle((w0lo, walo), w0w, waw, fill=False,
                               ec="k", ls="--", lw=1.0))
    xs = np.array([w0lo, w0lo + w0w])
    ax.plot(xs, -xs, color="crimson", lw=1.4, label="$w_0 + w_a = 0$")

    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    ax.plot(-1, 0, marker="*", ms=13, color="k", zorder=6)
    ax.annotate(r"$\Lambda$CDM", (-1, 0), textcoords="offset points",
                xytext=(8, 6), fontsize=9)

    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$w_a$")
    ax.set_xlim(w0lo - 0.2, w0lo + w0w + 0.2)
    ax.set_ylim(walo - 0.2, walo + waw + 0.2)
    ax.set_title("CPL prior support and posterior in the $(w_0, w_a)$ plane")

    handles = [Line2D([], [], color=PRIOR_C, marker="o", ls="none", ms=5,
                      label="prior ($\\beta=0$)"),
               Line2D([], [], color=POST_C, marker="o", ls="none", ms=5,
                      label="CPL posterior"),
               Line2D([], [], color="crimson", lw=1.4, label="$w_0+w_a=0$ cut"),
               Line2D([], [], color="k", ls="--", lw=1.0,
                      label="prior box")]
    ax.legend(handles=handles, fontsize=8, loc="lower left")

    out = os.path.join(output_dir, "cpl_prior_posterior_w.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="CPL prior vs posterior")
    p.add_argument("--box", default="torc", choices=sorted(PRIOR_BOXES))
    p.add_argument("--output", default="output")
    args = p.parse_args()
    plot_corner(args.box, args.output)
    plot_w_plane(args.box, args.output)
