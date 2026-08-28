"""
Plot Hubble horizon and varpi(a) for a single parameter set,
split into three epochs with piecewise attractor lines.

Epochs are separated by:
  - radiation-matter equality:  a_eq  = Omega_r / Omega_m
  - matter-Lambda equality:     a_mL  = (Omega_m / Omega_L)^{1/3}

wi values for cp_plus / cp_minus attractors:
  Epoch 1 (early):         wi = -1/3
  Epoch 2 (matter):        wi =  0
  Epoch 3 (dark energy):   wi = -1
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equations_noa0 import rhopsol, omega_r
from equations_piecewise import cp_plus, cp_minus

# ── Parameters ─────────────────────────────────────────────────────────
stigma  = 1.01
mups2   = 4.0 / 3.0
h       = 0.67
omega_m = 0.3
OmL     = 0.7
H0      = h * 100

# ── Initial conditions: cp± at radiation epoch (wi = -1/3) ─────────────
cpr_plus  = cp_plus(stigma,  -1/3)
cpr_minus = cp_minus(stigma, -1/3)
print(f"cp+(stigma={stigma}, wi=-1/3) = {cpr_plus:.6f}  → used as cpr for '+' run")
print(f"cp-(stigma={stigma}, wi=-1/3) = {cpr_minus:.6f}  → used as cpr for '-' run")

# ── Solve ODE for both initial conditions ───────────────────────────────
branches = {}
for label, cpr_val in [('+', cpr_plus), ('-', cpr_minus)]:
    print(f"\nSolving ODE ({label} branch, cpr={cpr_val:.6f})...")
    try:
        asol, varpisol, hsol, _, a0, P, rho, _ = rhopsol(
            cpr_val, omega_m, h, OmL, stigma, mups2)
        hr = H0 * asol / (a0 * hsol)
        branches[label] = dict(asol=asol, varpisol=varpisol, hsol=hsol,
                                a0=a0, hr=hr, cpr=cpr_val)
        print(f"  ✓ completed, a0={a0:.6f}")
    except Exception as e:
        print(f"  ✗ SKIPPED: {e}")

if not branches:
    raise RuntimeError("Both branches failed — cannot plot.")

# convenience aliases (fall back gracefully if one branch missing)
a_start = min(b['asol'][0] for b in branches.values())

# ── Epoch boundaries ───────────────────────────────────────────────────
omr   = omega_r(h)
a_eq  = omr / omega_m              # radiation-matter equality
a_mL  = (omega_m / OmL) ** (1/3)  # matter-Lambda equality

print(f"Radiation-matter equality:  a_eq  = {a_eq:.3e}  (z = {1/a_eq - 1:.0f})")
print(f"Matter-Lambda equality:     a_mL  = {a_mL:.3f}  (z = {1/a_mL - 1:.2f})")

# ── Piecewise attractor values ─────────────────────────────────────────
epochs = [
    dict(wi=-1/3, label=r'$w_i=-1/3$',           a_lo=0,       a_hi=a_eq,  color='#FFF3E0'),
    dict(wi= 0,   label=r'$w_i=0$',              a_lo=a_eq,    a_hi=a_mL,  color='#E8F5E9'),
    dict(wi=-1,   label=r'$w_i=-1$',             a_lo=a_mL,    a_hi=1.0,   color='#E8EAF6'),
]

for ep in epochs:
    ep['a_lo'] = max(ep['a_lo'], a_start)   # clamp to actual solution range
    ep['cp+']  = cp_plus(stigma,  ep['wi'])
    ep['cp-']  = cp_minus(stigma, ep['wi'])
    print(f"  wi={ep['wi']:+.4f}:  cp+ = {ep['cp+']:.6f},  cp- = {ep['cp-']:.6f}")

# ── Style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True, "font.family": "STIXGeneral",
    "mathtext.fontset": "stix", "font.size": 10,
    "axes.labelsize": 12, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 200,
    "savefig.dpi": 200, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "lines.linewidth": 1.2,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

EPOCH_COLORS = ['#FFF3E0', '#E8F5E9', '#E8EAF6']
VLINE_KW     = dict(color='#888888', ls=':', lw=0.9, zorder=1)
COLORS_PM    = ('steelblue', 'tomato')


def shade_epochs(ax):
    for ep in epochs:
        ax.axvspan(ep['a_lo'], ep['a_hi'], alpha=0.6,
                   color=ep['color'], zorder=0, lw=0)
    ax.axvline(a_eq, **VLINE_KW)
    ax.axvline(a_mL, **VLINE_KW)


def epoch_labels(ax, y_axes=0.97):
    """Write epoch labels (wi) near the top of the axes."""
    centres = [
        np.sqrt(max(a_start, epochs[0]['a_lo']) * a_eq),
        np.sqrt(a_eq * a_mL),
        np.sqrt(a_mL * 1.0),
    ]
    for ep, xc in zip(epochs, centres):
        ax.text(xc, y_axes, ep['label'],
                transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8.5,
                color='#333333')


BRANCH_COLORS = {'+': 'steelblue', '-': 'tomato'}

# ── Left panel: Hubble horizon H0/(aH) ────────────────────────────────
shade_epochs(ax1)

for lbl, b in branches.items():
    ax1.plot(b['asol'], b['hr'], color=BRANCH_COLORS[lbl], lw=1.5,
             label=rf"$\varpi_0={b['cpr']:.4f}$ (${lbl}$ branch)", zorder=3)

# Reference power-law slopes in each epoch: H0/(aH) ~ a^{(1+3wi)/2}
ref_branch = branches.get('+') or branches.get('-')
for ep, ls in zip(epochs, ['--', '-.', ':']):
    a_lo = max(ep['a_lo'], a_start)
    a_hi = min(ep['a_hi'], 1.0)
    if a_lo >= a_hi:
        continue
    a_mid  = np.sqrt(a_lo * a_hi)
    hr_mid = np.interp(a_mid, ref_branch['asol'], ref_branch['hr'])
    a_ref  = np.array([a_lo, a_hi])
    slope  = (1 + 3 * ep['wi']) / 2
    hr_ref = hr_mid * (a_ref / a_mid) ** slope
    ax1.plot(a_ref, hr_ref, ls=ls, lw=1.0, color='#444444', zorder=2,
             label=fr'slope $= {slope:+.1f}$, {ep["label"]}')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(a_start, 1.0)
ax1.set_xlabel(r'$a$')
ax1.set_ylabel(r'$H_0\,/\,(aH(a))$')
ax1.set_title(r'Hubble horizon')
ax1.legend(loc='upper left', fontsize=7.5)
epoch_labels(ax1)

# ── Right panel: varpi(a) with piecewise attractors ────────────────────
shade_epochs(ax2)

for lbl, b in branches.items():
    ax2.plot(b['asol'], b['varpisol'], color=BRANCH_COLORS[lbl], lw=1.5,
             label=rf"$\varpi_0={b['cpr']:.4f}$ (${lbl}$ branch)", zorder=3)

for ep in epochs:
    a_lo = max(ep['a_lo'], a_start)
    a_hi = min(ep['a_hi'], 1.0)
    if a_lo >= a_hi:
        continue
    # cp+
    ax2.hlines(ep['cp+'], a_lo, a_hi,
               colors=COLORS_PM[0], ls='--', lw=1.4, zorder=2)
    ax2.annotate(fr"$\varpi_+={ep['cp+']:.3f}$",
                 xy=(np.sqrt(a_lo * a_hi), ep['cp+']),
                 xytext=(0, 5), textcoords='offset points',
                 ha='center', fontsize=7, color=COLORS_PM[0])
    # cp-
    ax2.hlines(ep['cp-'], a_lo, a_hi,
               colors=COLORS_PM[1], ls='--', lw=1.4, zorder=2)
    ax2.annotate(fr"$\varpi_-={ep['cp-']:.3f}$",
                 xy=(np.sqrt(a_lo * a_hi), ep['cp-']),
                 xytext=(0, -10), textcoords='offset points',
                 ha='center', fontsize=7, color=COLORS_PM[1])

# Legend
ax2.plot([], [], color=COLORS_PM[0], ls='--', lw=1.4, label=r'$\varpi_+$ (attractor)')
ax2.plot([], [], color=COLORS_PM[1], ls='--', lw=1.4, label=r'$\varpi_-$ (attractor)')
ax2.legend(loc='lower left', fontsize=7.5)

ax2.set_xscale('log')
ax2.set_xlim(a_start, 1.0)
ax2.set_xlabel(r'$a$')
ax2.set_ylabel(r'$\varpi$')
ax2.set_title(r'Scalar field $\varpi(a)$')
epoch_labels(ax2)

# ── Super-title ────────────────────────────────────────────────────────
fig.suptitle(
    rf'Stigma: $\varsigma={stigma}$, '
    rf'$m\upsilon_2={mups2:.3f}$, '
    rf'init. from $\varpi_{{\pm}}(w_i=-1/3)$',
    fontsize=12, y=1.01)

plt.tight_layout()

outfile = 'plot_epochs.pdf'
plt.savefig(outfile, bbox_inches='tight')
print(f"\nSaved {outfile}")
plt.show()
