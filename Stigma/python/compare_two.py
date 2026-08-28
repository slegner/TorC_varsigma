import os, sys
import numpy as np
import matplotlib.pyplot as plt
import camb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equations_double import rhopsol

# --- Fixed cosmological parameters ---
cpr     = 0.99
omega_m = 0.3
h       = 0.67
OmL     = 0.7
H0      = h * 100

def solve_and_camb(stigma, mups2):
    asol, _, hsol, _, a0, P_DE, rho_DE, _ = rhopsol(
        cpr, omega_m, h, OmL, stigma, mups2)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=0.02242, omch2=0.11933, mnu=0.06, omk=0, tau=0.0544)
    pars.set_dark_energy_rho_p_a(a_p=asol, a_rho=asol, p=P_DE, rho=rho_DE)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    pars.InitPower.set_params(As=2.1e-9, ns=0.965, r=0)
    results = camb.get_results(pars)
    cls = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
    ell = np.arange(cls.shape[0])
    horizon = H0 * asol / (a0 * hsol)
    return asol, horizon, ell, cls[:, 0]

# ======================================================
# CELL 1: run once to save baseline
# ======================================================
base_a, base_horizon, base_ell, base_cls = solve_and_camb(stigma=1.0, mups2=4/3)

# ======================================================
# CELL 2: change these and re-run to compare
# ======================================================
other_a, other_horizon, other_ell, other_cls = solve_and_camb(stigma=1.5, mups2=4/3)

fig1, ax1 = plt.subplots()
ax1.plot(base_a,  base_horizon,  'k--', label='Baseline')
ax1.plot(other_a, other_horizon, '-',   label='Other')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel(r'$a$'); ax1.set_ylabel(r'$H_0/(aH(a))$')
ax1.legend(); fig1.tight_layout()

fig2, ax2 = plt.subplots()
ax2.plot(base_ell,  base_cls,  'k--', label='Baseline')
ax2.plot(other_ell, other_cls, '-',   label='Other')
ax2.set_xlabel(r'$\ell$'); ax2.set_ylabel(r'$D_\ell\;[\mu\mathrm{K}^2]$')
ax2.legend(); fig2.tight_layout()

plt.show()
