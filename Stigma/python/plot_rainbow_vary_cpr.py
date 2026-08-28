import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equations_double import (omega_r, validate_stigma_constraints, small,
                               aseries, cpseries, aseriesp, cpseriesp,
                               coupdiff, hstop,
                               rho_deriveda0, P_deriveda0)

def astop100(t, y):
    return y[0] - 100

astop100.terminal  = True
astop100.direction = 1

def rhopsol(cpr, omega_m, h, OmL, stigma, mups2):
    validate_stigma_constraints(stigma, cpr)

    tauini = small(cpr, omega_m, h, stigma, mups2)

    y0 = [aseries(tauini, cpr, omega_m, h, OmL, 1, stigma, mups2),
          cpseries(tauini, cpr, omega_m, h, OmL, 1, stigma, mups2),
          aseriesp(tauini, cpr, omega_m, h, OmL, 1, stigma, mups2),
          cpseriesp(tauini, cpr, omega_m, h, OmL, 1, stigma, mups2)]

    sol = solve_ivp(lambda tau, y: coupdiff(y, tau, omega_m, h, OmL, 1, stigma, mups2),
                    [tauini, np.inf], y0, events=hstop(h), rtol=1e-8, atol=1e-8)
    a0 = sol.y_events[0][0][0]

    y02 = [aseries(tauini, cpr, omega_m, h, OmL, a0, stigma, mups2),
           cpseries(tauini, cpr, omega_m, h, OmL, a0, stigma, mups2),
           aseriesp(tauini, cpr, omega_m, h, OmL, a0, stigma, mups2),
           cpseriesp(tauini, cpr, omega_m, h, OmL, a0, stigma, mups2)]

    sola0 = solve_ivp(lambda tau, y: coupdiff(y, tau, omega_m, h, OmL, a0, stigma, mups2),
                      [tauini, np.inf], y02, events=astop100, rtol=1e-8, atol=1e-8)

    tsol  = interp1d(sola0.y[0], sola0.t, bounds_error=False)
    avals = np.logspace(np.log10(sola0.y[0][0]), 2, 10000)   # up to a=100
    tvals = tsol(avals)
    tvals[0] = tauini
    avals[0] = sola0.y[0][0]

    sola02 = solve_ivp(lambda tau, y: coupdiff(y, tau, omega_m, h, OmL, a0, stigma, mups2),
                       [tauini, np.inf], y02, t_eval=tvals, events=astop100,
                       rtol=1e-8, atol=1e-8)

    asol, varpisol, hsol, dvarpisol = sola02.y
    aend, vpend, hend, dvpend = sola02.y_events[0][0]

    if aend > asol[-1]:
        asol      = np.append(asol,      aend)
        hsol      = np.append(hsol,      hend)
        varpisol  = np.append(varpisol,  vpend)
        dvarpisol = np.append(dvarpisol, dvpend)

    rhoderiv = rho_deriveda0(asol, hsol, varpisol, dvarpisol, omega_m, h, OmL, a0, stigma, mups2)
    P_DE     = P_deriveda0(  asol, hsol, varpisol, dvarpisol, omega_m, h, OmL, a0, stigma, mups2) / rhoderiv[-1]
    rho_DE   = rho_deriveda0(asol, hsol, varpisol, dvarpisol, omega_m, h, OmL, a0, stigma, mups2) / rhoderiv[-1]

    return asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE

# --- Fixed parameters ---
stigma  = 1.01
mups2   = 4/3
omega_m = 0.3
h       = 0.7
OmL     = 0.7

cpr_values = np.linspace(0.7, 0.999, 20)

# --- Predicted asymptotic varpi^2 values ---
def cpsq_minus(stigma, wi):
    return ((stigma*(1 + 2*(-3 + stigma)*stigma + (-3 + 6*stigma*(1 + stigma))*wi)
             + 2*((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + 3*stigma*wi)**2))**(1/2))
            / ((-1 + 4*stigma**2)*(-1 + 3*wi)))

varpi2_wi0  = cpsq_minus(stigma, 0)
varpi2_wim1 = cpsq_minus(stigma, -1)

# --- Colormap ---
CMAP_COLORS = ["#785EF0", "#DC267F", "#FFB000"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_cmap", CMAP_COLORS, N=256)
norm = Normalize(vmin=cpr_values.min(), vmax=cpr_values.max())
sm   = ScalarMappable(cmap=CUSTOM_CMAP, norm=norm)
sm.set_array([])

fig, (ax_hub, ax_vp) = plt.subplots(1, 2, figsize=(10, 5), sharey=False)

for cpr in cpr_values:
    print(f"Running cpr={cpr:.4f}")
    try:
        asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE = \
            rhopsol(cpr, omega_m, h, OmL, stigma, mups2)
    except Exception as e:
        print(f"  SKIPPED: {e}")
        continue

    color = CUSTOM_CMAP(norm(cpr))
    hubble_horizon = h * 100 * asol / (a0 * hsol)

    ax_hub.plot(asol, hubble_horizon, color=color)
    ax_vp.plot(asol,  varpisol**2,    color=color)

# --- Reference lines ---
for ax in (ax_hub, ax_vp):
    ax.axvline(1, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xscale('log')
    ax.set_xlabel(r'$a$')

ax_hub.set_yscale('log')
ax_hub.set_ylabel('Hubble horizon')
ax_hub.set_title(r'Hubble horizon  ($\varsigma=1.01$)')

ax_vp.axhline(varpi2_wi0,  color='C1', linestyle='-',  linewidth=1.2,
              label=r'$\varpi^2_-$, $w_i=0$')
ax_vp.axhline(varpi2_wim1, color='C2', linestyle='--', linewidth=1.2,
              label=r'$\varpi^2_-$, $w_i=-1$')
ax_vp.set_ylabel(r'$\varpi^2$')
ax_vp.set_title(r'$\varpi^2$ evolution  ($\varsigma=1.01$)')
ax_vp.legend(fontsize=8)

cbar = fig.colorbar(sm, ax=[ax_hub, ax_vp], orientation='vertical',
                    fraction=0.02, pad=0.02)
cbar.set_label(r'$\varpi_0$')

plt.tight_layout()
plt.savefig('rainbow_vary_cpr_stigma1p01.pdf', bbox_inches='tight')
print("Saved rainbow_vary_cpr_stigma1p01.pdf")
plt.show()
