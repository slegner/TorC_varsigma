"""Compare Hubble horizon from analytic g_i (hsol_de.py) vs full numerical ODE (equations.py)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hsol_de import omega_r, Hplus, Hminus, solve_a0
from equations import rhopsol

# Fixed cosmological parameters
h = 0.67
omega_m = 0.3
cpr = 0.8
wi = -1
H0 = 100 * h
OmR = omega_r(h)
OmL = 1 - omega_m - OmR

# Analytic a range (DE dominated, late times)
a_ana = np.linspace(0.77, 1, 1000)

def gi_value(stigma, wi, v2, sign):
    """Analytic g_i constant. v2 uses old (negative) convention from hsol_de.py."""
    return (4*(stigma*(3 - 9*wi) - stigma**3*(1 + 3*wi)**2 + (stigma +
            3*stigma*wi)**2 + sign*(1 + 3*wi)*((-1 + stigma)**2*stigma**2*(2 - 6*wi +
            (stigma + 3*stigma*wi)**2))**(1/2)))/(stigma*v2*(1 + 2*stigma +
            6*(-1 + stigma)*wi)*(-7 - 3*wi + 4*stigma*(1 + 3*wi)))

# GR Hubble horizon: H0/(H_GR * a)
def horizon_GR(a):
    H = H0 * np.sqrt(OmR / a**4 + omega_m / a**3 + OmL)
    return H0 / (H * a)

# --- Plot 1: Fix ups2 = -4/3, vary stigma ---
# cpr=0.8 => constraint 1: stigma > 0.64; constraint 2: applies above that.
# Use range starting just above the boundary.
ups2_fixed = -4.0 / 3.0
stigma_values = np.arange(1.001, 2.0, 0.1).tolist()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
ax.plot(a_ana, horizon_GR(a_ana), 'k--', lw=2, label='GR')

cmap1 = plt.cm.viridis
norm1 = Normalize(vmin=min(stigma_values), vmax=max(stigma_values))

for stigma in stigma_values:
    color = cmap1(norm1(stigma))
    v2 = ups2_fixed  # ups2 is already negative (old convention for hsol_de)

    # Analytic: H0/(H * a), using hsol_de convention (v2 negative)
    try:
        gi = gi_value(stigma, wi, v2, -1)
        a0 = solve_a0(gi, h, omega_m, OmL)
        H2 = Hplus(h, stigma, wi, OmL, omega_m, v2, a_ana, a0)
        mask = H2 > 0
        if np.any(mask):
            H_ana = np.sqrt(H2[mask])
            ax.plot(a_ana[mask], H0 / (H_ana * a_ana[mask]), '-', color=color, lw=1.5)
    except Exception as e:
        print(f"Analytic failed stigma={stigma:.2f}: {e}")

    # Numerical: H0 * a / (da/dτ) = H0 * asol / hsol
    try:
        asol, varpisol, hsol, dvarpisol, a0_num, P_DE, rho_DE = rhopsol(
            cpr, omega_m, h, OmL, stigma, ups2_fixed)
        mask = asol >= 0.77
        horizon_num = H0 * asol[mask] / hsol[mask]
        ax.plot(asol[mask], horizon_num, '--', color=color, lw=2, alpha=0.8)
    except Exception as e:
        print(f"Numerical failed stigma={stigma:.2f}: {e}")

ax.set_xlabel('Scale factor $a$')
ax.set_ylabel('$H_0 / (H \\cdot a)$')
ax.set_title(f'Hubble horizon: $\\upsilon_2 = -4/3$, vary $\\varsigma$')
sm1 = ScalarMappable(cmap=cmap1, norm=norm1)
cb1 = fig.colorbar(sm1, ax=ax)
cb1.set_label('$\\varsigma$')
ax.plot([], [], 'k-', lw=1.5, label='$g_i^+$ analytic')
ax.plot([], [], 'k--', lw=2, label='numerical ODE')
ax.legend(fontsize=8, loc='best')
ax.set_xlim(0.77, 1)

# --- Plot 2: Fix stigma = 1, vary ups2 ---
stigma_fixed = 1.0
ups2_values = np.arange(-2.0, -2.0/3.0, 0.2).tolist()

ax = axes[1]
ax.plot(a_ana, horizon_GR(a_ana), 'k--', lw=2, label='GR')

cmap2 = plt.cm.plasma
norm2 = Normalize(vmin=min(ups2_values), vmax=max(ups2_values))

for ups2 in ups2_values:
    color = cmap2(norm2(ups2))
    v2 = ups2  # ups2 is already negative (old convention for hsol_de)

    # Analytic
    try:
        gi = gi_value(stigma_fixed, wi, v2, -1)
        a0 = solve_a0(gi, h, omega_m, OmL)
        H2 = Hplus(h, stigma_fixed, wi, OmL, omega_m, v2, a_ana, a0)
        mask = H2 > 0
        if np.any(mask):
            H_ana = np.sqrt(H2[mask])
            ax.plot(a_ana[mask], H0 / (H_ana * a_ana[mask]), '-', color=color, lw=1.5)
    except Exception as e:
        print(f"Analytic failed ups2={ups2:.2f}: {e}")

    # Numerical
    try:
        asol, varpisol, hsol, dvarpisol, a0_num, P_DE, rho_DE = rhopsol(
            cpr, omega_m, h, OmL, stigma_fixed, ups2)
        mask = asol >= 0.77
        horizon_num = H0 * asol[mask] / hsol[mask]
        ax.plot(asol[mask], horizon_num, '--', color=color, lw=2, alpha=0.8)
    except Exception as e:
        print(f"Numerical failed ups2={ups2:.2f}: {e}")

ax.set_xlabel('Scale factor $a$')
ax.set_ylabel('$H_0 / (H \\cdot a)$')
ax.set_title(f'Hubble horizon: $\\varsigma = 1$, vary $\\upsilon_2$')
sm2 = ScalarMappable(cmap=cmap2, norm=norm2)
cb2 = fig.colorbar(sm2, ax=ax)
cb2.set_label('$\\upsilon_2$')
ax.plot([], [], 'k-', lw=1.5, label='$g_i^+$ analytic')
ax.plot([], [], 'k--', lw=2, label='numerical ODE')
ax.legend(fontsize=8, loc='best')
ax.set_xlim(0.77, 1)

plt.tight_layout()
plt.savefig('../Plots/hsol_de_numerical_overlay.pdf', bbox_inches='tight')
plt.show()
print("Saved to Plots/hsol_de_numerical_overlay.pdf")
