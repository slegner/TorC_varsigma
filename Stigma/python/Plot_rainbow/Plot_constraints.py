import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from equations_double import rhopsol

# --- Same parameter values as the rainbow plots ---
cpr           = 0.99
STIGMA_FID    = 1.0
MUPS2_FID     = 4.0 / 3.0
stigma_values = np.linspace(1.001, 2.0, 10)
mups2_values  = np.linspace(2.0 / 3.0, 2.0, 10)

# --- Same style and colormap as rainbow plots ---
plt.rcParams.update({
    "text.usetex": True, "font.family": "STIXGeneral",
    "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 11, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "axes.linewidth": 0.8,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.minor.size": 2, "ytick.minor.size": 2,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
})

CMAP_COLORS = ["#785EF0", "#DC267F", "#FFB000"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_cmap", CMAP_COLORS, N=256)

# --- Constraint grid ---
varsigma = np.linspace(0, 2.5, 1000)
varpi    = np.linspace(0, 2.0, 1000)
S, P     = np.meshgrid(varsigma, varpi)

c1    = S > P**2
c2    = S * (5 + 4 * S * (-2 + P**2)) < P**2
valid = c1 & c2

# ======================================================================
# Figure 1: stigma rainbow values on the constraint plane
# ======================================================================
norm_s = Normalize(vmin=stigma_values.min(), vmax=stigma_values.max())
sm_s   = ScalarMappable(cmap=CUSTOM_CMAP, norm=norm_s)
sm_s.set_array([])

fig1, ax = plt.subplots(figsize=(5.5, 4.5))

ax.contourf(S, P, valid.astype(float), levels=[0.5, 1.5],
            colors=["#4488CC"], alpha=0.35, zorder=0)
ax.contour(S, P, (S - P**2),
           levels=[0], colors=["#CC3333"], linewidths=1.4, zorder=1)
ax.contour(S, P, (S * (5 + 4 * S * (-2 + P**2)) - P**2),
           levels=[0], colors=["#FF8800"], linewidths=1.4, zorder=1)

# Vertical line at cpr used in runs
ax.axvline(x=cpr, color='black', ls=':', lw=1.0, zorder=2,
           label=fr'$\varpi_0 = {cpr}$')

# Rainbow stigma dots at fixed cpr
for s in stigma_values:
    ax.plot(cpr, s, 'o', color=CUSTOM_CMAP(norm_s(s)),
            ms=5, zorder=3, markeredgewidth=0.3, markeredgecolor='k')

# Fiducial point
ax.plot(cpr, STIGMA_FID, '*', color='black', ms=9, zorder=4,
        label=fr'Fiducial $\varsigma={STIGMA_FID}$')

# Legend entries for constraint lines
ax.plot([], [], color="#4488CC", lw=8, alpha=0.35, label="Valid region")
ax.plot([], [], color="#CC3333", lw=1.4, label=r"$\varsigma = \varpi_0^2$")
ax.plot([], [], color="#FF8800", lw=1.4,
        label=r"$\varsigma(5+4\varsigma(-2+\varpi_0^2))=\varpi_0^2$")

ax.set_xlabel(r"$\varpi_0$")
ax.set_ylabel(r"$\varsigma$")
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 2.5)
ax.legend(loc="upper right", fontsize=7)
ax.grid(ls="--", alpha=0.25)

cbar = fig1.colorbar(sm_s, ax=ax, orientation='vertical',
                     fraction=0.04, pad=0.02)
cbar.set_label(r'$\varsigma$ (rainbow values)')

fig1.tight_layout()
fig1.savefig('constraints_stigma.pdf', bbox_inches='tight')
print("Saved constraints_stigma.pdf")
plt.close()

# ======================================================================
# Figure 2: mups2 rainbow values — note mups2 does not appear in the
# constraints, so all mups2 values sit at the same point on this plane.
# Show the valid region + fixed point (cpr, stigma=1), then a colour
# strip of mups2 values in an inset.
# ======================================================================
norm_m = Normalize(vmin=mups2_values.min(), vmax=mups2_values.max())
sm_m   = ScalarMappable(cmap=CUSTOM_CMAP, norm=norm_m)
sm_m.set_array([])

fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))

ax2.contourf(S, P, valid.astype(float), levels=[0.5, 1.5],
             colors=["#4488CC"], alpha=0.35, zorder=0)
ax2.contour(S, P, (S - P**2),
            levels=[0], colors=["#CC3333"], linewidths=1.4, zorder=1)
ax2.contour(S, P, (S * (5 + 4 * S * (-2 + P**2)) - P**2),
            levels=[0], colors=["#FF8800"], linewidths=1.4, zorder=1)

ax2.axvline(x=cpr, color='black', ls=':', lw=1.0, zorder=2,
            label=fr'$\varpi_0 = {cpr}$')

# All mups2 values map to the same point — draw a stacked dot
ax2.plot(cpr, STIGMA_FID, 'o', color='black', ms=8, zorder=3,
         label=fr'All $m\upsilon_2$ runs: $\varsigma={STIGMA_FID}$')

ax2.plot([], [], color="#4488CC", lw=8, alpha=0.35, label="Valid region")
ax2.plot([], [], color="#CC3333", lw=1.4, label=r"$\varsigma = \varpi_0^2$")
ax2.plot([], [], color="#FF8800", lw=1.4,
         label=r"$\varsigma(5+4\varsigma(-2+\varpi_0^2))=\varpi_0^2$")

ax2.set_xlabel(r"$\varpi_0$")
ax2.set_ylabel(r"$\varsigma$")
ax2.set_xlim(0, 2.0)
ax2.set_ylim(0, 2.5)
ax2.legend(loc="upper right", fontsize=7)
ax2.grid(ls="--", alpha=0.25)

# Inset: colour strip of mups2 values
axins2 = ax2.inset_axes([0.05, 0.72, 0.40, 0.08])
for m in mups2_values:
    axins2.axvline(x=m, color=CUSTOM_CMAP(norm_m(m)), lw=3)
axins2.axvline(x=MUPS2_FID, color='k', lw=1.5, ls='--')
axins2.set_xlim(mups2_values.min(), mups2_values.max())
axins2.set_yticks([])
axins2.set_xlabel(r'$m\upsilon_2$', fontsize=7)
axins2.tick_params(labelsize=6)
axins2.set_title(r'$m\upsilon_2$ range (not in constraints)', fontsize=6, pad=2)

cbar2 = fig2.colorbar(sm_m, ax=ax2, orientation='vertical',
                      fraction=0.04, pad=0.02)
cbar2.set_label(r'$m\upsilon_2$ (rainbow values)')

fig2.tight_layout()
fig2.savefig('constraints_mups2.pdf', bbox_inches='tight')
print("Saved constraints_mups2.pdf")
plt.close()

# ======================================================================
# Helper: constraint / highest-derivative (H²) term
# ======================================================================
# Same cosmological params as the rainbow plots
h       = 0.67
omega_m = 0.3
OmL     = 0.7


def constraint_ratio(asol, hsol, varpisol, a0, stigma, mups2):
    """
    |C| / |T_hd|  where T_hd is the H² term in the (0,0) Friedmann constraint.

        T_hd = (3 a a0 / 4ς) · υ₂ · (da/a)² · (4ς − ϖ²)

    The ratio should sit at the level of rtol (~1e-8) everywhere.
    """
    T_hd = (3.0 * asol * a0 / (4.0 * stigma)) * \
           (hsol**2 * mups2 * (-varpisol**2 + 4.0 * stigma)) / asol**2
    return T_hd   # denominator only; caller divides cval by this


# ======================================================================
# Figure 3: Constraint ratio — vary stigma, fix mups2
# ======================================================================
# Use a safe reduced range to avoid ODE hangs at high stigma
stigma_vals_dyn = np.linspace(1.001, 1.5, 8)
mups2_vals_dyn  = np.linspace(2.0/3.0, 1.5, 8)

norm_s3 = Normalize(vmin=stigma_vals_dyn.min(), vmax=stigma_vals_dyn.max())
sm_s3   = ScalarMappable(cmap=CUSTOM_CMAP, norm=norm_s3)
sm_s3.set_array([])

fig3, ax3 = plt.subplots(figsize=(6, 4))

for s in stigma_vals_dyn:
    try:
        asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE, cval = rhopsol(
            cpr, omega_m, h, OmL, s, MUPS2_FID)
        T_hd  = constraint_ratio(asol, hsol, varpisol, a0, s, MUPS2_FID)
        ratio = np.abs(cval) / np.abs(T_hd)
        color = CUSTOM_CMAP(norm_s3(s))
        ax3.plot(asol, ratio, color=color, lw=1.0)
    except Exception as e:
        print(f"  stigma={s:.3f} SKIPPED: {e}")

ax3.axhline(1e-8, color='grey', ls=':', lw=0.8, label=r'$\mathtt{rtol}=10^{-8}$')
ax3.set_xlabel(r'$a$')
ax3.set_ylabel(r'$|\mathcal{C}|\,/\,|T_{\mathrm{hd}}|$')
ax3.set_title(fr'Constraint ratio: $m\upsilon_2={MUPS2_FID:.3f}$, vary $\varsigma$')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.legend(loc='upper right')
ax3.grid(ls='--', alpha=0.25)
cbar3 = fig3.colorbar(sm_s3, ax=ax3, fraction=0.04, pad=0.02)
cbar3.set_label(r'$\varsigma$')
fig3.tight_layout()
fig3.savefig('constraint_residual_stigma.pdf', bbox_inches='tight')
print("Saved constraint_residual_stigma.pdf")
plt.close()

# ======================================================================
# Figure 4: Constraint ratio — vary mups2, fix stigma
# ======================================================================
norm_m4 = Normalize(vmin=mups2_vals_dyn.min(), vmax=mups2_vals_dyn.max())
sm_m4   = ScalarMappable(cmap=CUSTOM_CMAP, norm=norm_m4)
sm_m4.set_array([])

fig4, ax4 = plt.subplots(figsize=(6, 4))

for m in mups2_vals_dyn:
    try:
        asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE, cval = rhopsol(
            cpr, omega_m, h, OmL, STIGMA_FID, m)
        T_hd  = constraint_ratio(asol, hsol, varpisol, a0, STIGMA_FID, m)
        ratio = np.abs(cval) / np.abs(T_hd)
        color = CUSTOM_CMAP(norm_m4(m))
        ax4.plot(asol, ratio, color=color, lw=1.0)
    except Exception as e:
        print(f"  mups2={m:.3f} SKIPPED: {e}")

ax4.axhline(1e-8, color='grey', ls=':', lw=0.8, label=r'$\mathtt{rtol}=10^{-8}$')
ax4.set_xlabel(r'$a$')
ax4.set_ylabel(r'$|\mathcal{C}|\,/\,|T_{\mathrm{hd}}|$')
ax4.set_title(fr'Constraint ratio: $\varsigma={STIGMA_FID}$, vary $m\upsilon_2$')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.legend(loc='upper right')
ax4.grid(ls='--', alpha=0.25)
cbar4 = fig4.colorbar(sm_m4, ax=ax4, fraction=0.04, pad=0.02)
cbar4.set_label(r'$m\upsilon_2$')
fig4.tight_layout()
fig4.savefig('constraint_residual_mups2.pdf', bbox_inches='tight')
print("Saved constraint_residual_mups2.pdf")
plt.close()
