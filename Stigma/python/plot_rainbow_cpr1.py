import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_varpi_cpr1 import rhopsol, omega_r

# --- Parameters ---
omega_m = 0.3
h       = 0.7
OmL     = 0.7
mups2   = 4/3

stigma_values = np.concatenate([
    np.linspace(0.6, 0.97, 15),
    np.linspace(1.03, 1.4, 15),
])

# --- Colormap ---
CMAP_COLORS = ["#785EF0", "#DC267F", "#FFB000"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_cmap", CMAP_COLORS, N=256)
norm = Normalize(vmin=0.6, vmax=1.4)
sm   = ScalarMappable(cmap=CUSTOM_CMAP, norm=norm)
sm.set_array([])

fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)

for stigma in stigma_values:
    print(f"Running stigma={stigma:.4f}")
    try:
        asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE, constrain, ddasol, Gi, T_hd = \
            rhopsol(omega_m, h, OmL, stigma, mups2)
    except Exception as e:
        print(f"  SKIPPED: {e}")
        continue

    color  = CUSTOM_CMAP(norm(stigma))
    weff   = 1/3 - (2*asol*ddasol)/(3*hsol**2)
    Gsol   = interp1d(asol, Gi, bounds_error=False, fill_value="extrapolate")

    def H_func(a, Gsol=Gsol, a0=a0):
        return (h*100)*np.sqrt(Gsol(a)*(omega_m*(a0*a)**-3 + omega_r(h)*(a0*a)**-4 + OmL))

    w_de = P_DE / rho_DE

    axes[0].plot(asol, varpisol**2,           color=color)
    axes[1].plot(asol, Gi,                    color=color)
    axes[2].plot(asol, weff,                  color=color)
    axes[3].plot(asol, w_de,                  color=color)
    axes[4].plot(asol, h*100*asol/(a0*hsol),  color=color)

# --- Reference lines ---
for w in [1/3, 0, -1]:
    axes[2].axhline(w, color='grey', linestyle=':', linewidth=0.8)
    axes[3].axhline(w, color='grey', linestyle=':', linewidth=0.8)

for ax in axes:
    ax.axvline(1, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xscale('log')

axes[0].set_ylabel(r'$\varpi^2$')
axes[1].set_ylabel(r'$G_N$')
axes[2].set_ylabel(r'$w_{\rm eff}$')
axes[3].set_ylabel(r'$w_{\rm DE}$')
axes[3].set_ylim(-2, 2)
axes[4].set_ylabel('Hubble horizon')
axes[4].set_xlabel(r'$a$')
axes[4].set_yscale('log')

cbar = fig.colorbar(sm, ax=axes.tolist(), orientation='vertical',
                    fraction=0.02, pad=0.02)
cbar.set_label(r'$\varsigma$')

plt.savefig('rainbow_cpr1_stigma.pdf', bbox_inches='tight')
plt.show()
