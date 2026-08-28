import os, sys
import numpy as np
import matplotlib.pyplot as plt
import camb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from equations_double import rhopsol

# --- Fixed parameters ---
cpr     = 0.99
omega_m = 0.3
h       = 0.67
OmL     = 0.7
H0      = h * 100
mups2   = 4/3

fig1, ax1 = plt.subplots()   # Hubble horizon
fig2, ax2 = plt.subplots()   # CMB

for stigma in [1.0, 1.2, 1.5, 2.0]:
    asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE, constrain = rhopsol(
        cpr, omega_m, h, OmL, stigma, mups2)

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=0.02242, omch2=0.11933, mnu=0.06, omk=0, tau=0.0544)
    pars.set_dark_energy_rho_p_a(a_p=asol, a_rho=asol, p=P_DE, rho=rho_DE)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)
    pars.InitPower.set_params(As=2.1e-9, ns=0.965, r=0)
    results = camb.get_results(pars)
    cls = results.get_cmb_power_spectra(pars, CMB_unit='muK')['total']
    ell = np.arange(cls.shape[0])

    label = fr'$\varsigma={stigma}$'
    ax1.plot(asol, H0 * asol / (a0 * hsol), label=label)
    ax2.plot(ell, cls[:, 0], label=label)

ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_xlabel(r'$a$'); ax1.set_ylabel(r'$H_0/(aH(a))$')
ax1.legend(); fig1.tight_layout()

ax2.set_xlabel(r'$\ell$'); ax2.set_ylabel(r'$D_\ell\;[\mu\mathrm{K}^2]$')
ax2.legend(); fig2.tight_layout()

plt.show()
