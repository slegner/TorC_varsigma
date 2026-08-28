import numpy as np
import matplotlib.pyplot as plt
import sys
from hsol_de import omega_r, Hplus, Hminus, solve_a0

# Fixed cosmological parameters
h = 0.67
omega_m = 0.3
wi = -1

# DE dominated epoch: a > 0.77 (z < 0.3)
a = np.linspace(0.77, 1, 1000)
OmR = omega_r(h)
OmL = 1 - omega_m - OmR

def gi_value(stigma, wi, v2, sign):
    """Extract gi from the Hplus/Hminus prefactor.
    sign=-1 for plus branch, sign=+1 for minus branch."""
    return (4*(stigma*(3 - 9*wi) - stigma**3*(1 + 3*wi)**2 + (stigma + \
            3*stigma*wi)**2 + sign*(1 + 3*wi)*((-1 + stigma)**2*stigma**2*(2 - 6*wi + \
            (stigma + 3*stigma*wi)**2))**(1/2)))/(stigma*v2*(1 + 2*stigma + \
            6*(-1 + stigma)*wi)*(-7 - 3*wi + 4*stigma*(1 + 3*wi)))

# Standard GR: H^2 = H0^2 (Om_r a^-4 + Om_m a^-3 + Om_L)
def H2_GR(a):
    H0_2 = (100 * h)**2
    return H0_2 * (OmR / a**4 + omega_m / a**3 + OmL)

# --- Plot 1: Fix v2 = -4/3, vary stigma ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

stigma_values = [-0.5, 0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
stigma_values = [s for s in stigma_values if abs(s) > 0.05]
v2_fixed = -4/3

ax = axes[0]
ax.plot(a, np.sqrt(H2_GR(a)), 'k--', lw=2, label='GR')

print("=== Vary stigma (v2 = -4/3, wi = -1) ===")
for stigma in stigma_values:
    for sign, branch, ls in [(-1, 'plus', '-'), (1, 'minus', '--')]:
        try:
            gi = gi_value(stigma, wi, v2_fixed, sign)
            a0 = solve_a0(gi, h, omega_m, OmL)
            print(f"  stigma={stigma:6.2f}, g_i^{branch}: gi={gi:.4f}, a0={a0:.6f}")
            if branch == 'plus':
                H2 = Hplus(h, stigma, wi, OmL, omega_m, v2_fixed, a, a0)
            else:
                H2 = Hminus(h, stigma, wi, OmL, omega_m, v2_fixed, a, a0)
            mask = H2 > 0
            if np.any(mask):
                ax.plot(a[mask], np.sqrt(H2[mask]), ls,
                        label=f'$g_i^{{{branch[0]}}}$, $\\varsigma$={stigma:.1f}')
        except Exception as e:
            print(f"  stigma={stigma:6.2f}, g_i^{branch}: FAILED ({e})")

ax.set_xlabel('Scale factor $a$')
ax.set_ylabel('$H$ [km/s/Mpc]')
ax.set_title(f'$\\upsilon_2 = -4/3$, $w_i = {wi}$, vary $\\varsigma$')
ax.set_yscale('log')
ax.legend(fontsize=7, loc='upper right')
ax.set_xlim(0.77, 1)

# --- Plot 2: Fix stigma = 1, vary v2 ---
stigma_fixed = 1.0
v2_values = [-2.5, -2.0, -1.5, -4/3, -1.0, -0.5, -0.2]

ax = axes[1]
ax.plot(a, np.sqrt(H2_GR(a)), 'k--', lw=2, label='GR')

print("\n=== Vary v2 (stigma = 1, wi = -1) ===")
for v2 in v2_values:
    label_v2 = '-4/3' if abs(v2 - (-4/3)) < 0.01 else f'{v2:.2f}'
    for sign, branch, ls in [(-1, 'plus', '-'), (1, 'minus', '--')]:
        try:
            gi = gi_value(stigma_fixed, wi, v2, sign)
            a0 = solve_a0(gi, h, omega_m, OmL)
            print(f"  v2={label_v2:>6s}, g_i^{branch}: gi={gi:.4f}, a0={a0:.6f}")
            if branch == 'plus':
                H2 = Hplus(h, stigma_fixed, wi, OmL, omega_m, v2, a, a0)
            else:
                H2 = Hminus(h, stigma_fixed, wi, OmL, omega_m, v2, a, a0)
            mask = H2 > 0
            if np.any(mask):
                ax.plot(a[mask], np.sqrt(H2[mask]), ls,
                        label=f'$g_i^{{{branch[0]}}}$, $\\upsilon_2$={label_v2}')
        except Exception as e:
            print(f"  v2={label_v2:>6s}, g_i^{branch}: FAILED ({e})")

ax.set_xlabel('Scale factor $a$')
ax.set_ylabel('$H$ [km/s/Mpc]')
ax.set_title(f'$\\varsigma = 1$, $w_i = {wi}$, vary $\\upsilon_2$')
ax.set_yscale('log')
ax.legend(fontsize=7, loc='upper right')
ax.set_xlim(0.77, 1)

plt.tight_layout()
plt.savefig('../Plots/hsol_de_comparison.pdf', bbox_inches='tight')
plt.show()
print("\nSaved to Plots/hsol_de_comparison.pdf")
