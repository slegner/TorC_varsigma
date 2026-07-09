"""
Rainbow plots in the style of Stigma/python/Plot_rainbow/Plot_rainbow_stigma_mups2.py
(left: H0/(aH(a)) log-log with residual inset; right: field trajectory
varpi(a)), for the noH0 anchoring of the TorC equations:
a0 = 1 pinned, today := a = 1, E normalised by H(a=1).  No CAMB.

Varies varsigma, OmL and mups2 (one figure each) over the sampling priors.
Writes plots/rainbow_noH0_{stigma,OmL,mups2}.{pdf,png}.
"""
import sys, os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

TORC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, TORC)
from torc.scipy_reference import (aseries, cpseries, aseriesp, cpseriesp, coupdiff,
                                   small, omega_r)

# --- baseline params (conventions of the noH0 pipeline) ---
h       = 0.7
omega_m = 0.2969
omegar  = omega_r(h)

STIGMA_FID = 1.0001
MUPS2_FID  = 4.0/3.0
OML_FID    = 1 - omega_m - omegar

SCALE_FACTORS_A = np.logspace(-6, 0, 1000)

SWEEPS = {
    "stigma": (np.unique(np.concatenate([np.linspace(1.0001, 1.3, 6),
                                         np.linspace(1.3, 3.0, 10)])),
               r'$\varsigma$'),
    "OmL":    (np.linspace(0.1, 1.5, 15), r'$\Omega_\Lambda$'),
    "mups2":  (np.linspace(0.5, 3.0, 15), r'$-\upsilon_2$'),
}
FID = dict(stigma=STIGMA_FID, mups2=MUPS2_FID, OmL=OML_FID)
FID_LABEL = (r'Fiducial ($\varsigma=1.0001,\;-\upsilon_2=4/3,'
             r'\;\Omega_\Lambda=%.3f$)' % OML_FID)

# --- style (matches the Plot_rainbow reference) ---
plt.rcParams.update({
    "text.usetex": True, "font.family": "STIXGeneral",
    "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 11, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "lines.linewidth": 1.0,
})
CMAP_COLORS = ["#785EF0", "#DC267F", "#FFB000"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_cmap", CMAP_COLORS, N=256)


def run_noH0(stigma, mups2, OmL):
    """noH0 solve: a0 = 1, integrate to a = 1.  Returns asol, varpi, hr
    with hr = H0_eff/(a H(a)) and H0_eff = H(a=1)."""
    a0 = 1
    tauini = small(omega_m, h, stigma, mups2)
    y0 = [aseries(tauini, omega_m, h, OmL, a0, stigma, mups2),
          cpseries(tauini, omega_m, h, OmL, a0, stigma, mups2),
          aseriesp(tauini, omega_m, h, OmL, a0, stigma, mups2),
          cpseriesp(tauini, omega_m, h, OmL, a0, stigma, mups2)]
    def astop(t, y): return y[0] - 1.0
    astop.terminal = True
    rhs = lambda t, y: coupdiff(y, t, omega_m, h, OmL, a0, stigma, mups2)
    sol = solve_ivp(rhs, [tauini, np.inf], y0, events=astop,
                    rtol=1e-8, atol=1e-8)
    tsol  = interp1d(sol.y[0], sol.t, bounds_error=False)
    avals = np.logspace(np.log10(sol.y[0][0]), 0, 5000)
    tvals = tsol(avals); tvals[0] = tauini
    sol2 = solve_ivp(rhs, [tauini, np.inf], y0, t_eval=tvals,
                     events=astop, rtol=1e-8, atol=1e-8)
    asol, varpi, hsol, _ = sol2.y
    aend, _, hend, _ = sol2.y_events[0][0]
    if aend > asol[-1]:
        asol  = np.append(asol,  aend)
        varpi = np.append(varpi, sol2.y_events[0][0][1])
        hsol  = np.append(hsol,  hend)
    H0_eff = hend / aend**2
    hr = H0_eff * asol / hsol          # = H0/(a H(a)),  H = hsol/a^2
    return asol, varpi, hr


def on_common(asol, y):
    return np.interp(np.log10(SCALE_FACTORS_A), np.log10(asol), y,
                     left=np.nan, right=np.nan)


def make_rainbow(param_name, param_values, cbar_label):
    norm = Normalize(vmin=param_values.min(), vmax=param_values.max())
    sm   = ScalarMappable(cmap=CUSTOM_CMAP, norm=norm)
    sm.set_array([])

    print(f"[{param_name}] fiducial", flush=True)
    p = dict(FID)
    asol_f, varpi_f, hr_f = run_noH0(p["stigma"], p["mups2"], p["OmL"])
    hr_f_common = on_common(asol_f, hr_f)

    runs = []
    for val in param_values:
        p = dict(FID); p[param_name] = val
        print(f"  {param_name}={val:.4f}", flush=True)
        try:
            asol, varpi, hr = run_noH0(p["stigma"], p["mups2"], p["OmL"])
            runs.append((val, asol, varpi, hr, on_common(asol, hr)))
        except Exception as e:
            print(f"    SKIPPED: {e}", flush=True)

    fig_width_in  = (510 * 0.99) / 72.27
    fig_height_in = fig_width_in * (6 / 8) / 2.4
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(fig_width_in, fig_height_in),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.30})

    # left: H0/(aH(a))
    for val, asol, varpi, hr, _ in runs:
        ax1.plot(asol, hr, color=CUSTOM_CMAP(norm(val)))
    ax1.plot(asol_f, hr_f, '--', color='black', label='Fiducial', zorder=5)
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlim(1e-6, 1); ax1.set_ylim(1e-4, 2)
    ax1.legend(loc='upper left')
    ax1.set_xlabel(r'$a$')
    ax1.set_ylabel(r'$H_0/(aH(a))$')

    # inset: residual vs fiducial
    axins = ax1.inset_axes([0.53, 0.04, 0.44, 0.36])
    for val, _, _, _, hr_c in runs:
        axins.plot(SCALE_FACTORS_A, hr_c - hr_f_common,
                   color=CUSTOM_CMAP(norm(val)))
    axins.axhline(0, c='k', ls='--', lw=0.8)
    axins.set_xlim(0, 1)
    axins.tick_params(labelsize=6)
    axins.set_xlabel(r'$a$', fontsize=7)
    axins.set_ylabel(r'$\Delta(H_0/aH)$', fontsize=7)

    # right: varpi(a)
    for val, asol, varpi, _, _ in runs:
        ax2.plot(asol, varpi, color=CUSTOM_CMAP(norm(val)))
    ax2.plot(asol_f, varpi_f, '--', color='black', label='Fiducial', zorder=5)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$a$')
    ax2.set_ylabel(r'$\varpi$')
    ax2.legend(loc='best')

    cbar = fig.colorbar(sm, ax=ax2, orientation='vertical',
                        fraction=0.04, pad=0.02)
    cbar.set_label(cbar_label)

    out = os.path.join(TORC, "plots", f"rainbow_noH0_{param_name}")
    fig.savefig(out + ".pdf", bbox_inches='tight')
    fig.savefig(out + ".png", bbox_inches='tight', dpi=200)
    print(f"Saved {out}.pdf", flush=True)
    plt.close()


if __name__ == "__main__":
    for pname, (values, lbl) in SWEEPS.items():
        make_rainbow(pname, values, lbl)
