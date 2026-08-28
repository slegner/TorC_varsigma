"""
H(z) and calcresult = H0*a/(a0*a') for the Stigma background, varpi_0 = 1.

Uses the ORIGINAL aseries/cpseries seeds (varpi_0 = 1 at 0th order).
a0 is pure bookkeeping: the rhs only ever sees a_phys = a0*a and a0*da, so we
integrate with a0 = 1 (a is then a_phys) and fix the normalisation afterwards
by declaring "today" to be where the conformal rate cal-H = a'/a equals H0.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy import constants
from stigma_rhs import coupdiff, dda, ddcp


def omega_r(h):
    ab = (8*constants.pi**5*constants.k**4)/(15*constants.h**3*constants.c**3)
    rhor = ab*2.7255**4/(constants.c**2)
    rhov = 3.046*(7/8)*(4/11)**(4/3)*rhor
    return((rhor+rhov)/((3*(100*h)**2/(8*constants.pi*constants.G))*(1000)**2/(10**6*constants.parsec)**2))


# ------------- original varpi_0 = 1 series (unchanged) -------------
def aseries(tau, omega_m, h, OmL, a0, stigma, mups2):
    return ((200*h*(omega_r(h))**(1/2)*(stigma)**(1/2)*tau)/(a0**2*mups2*(-1+4*stigma))**(1/2)
        + (2500*h**2*tau**2*(omega_m))/(a0*mups2)
        + (15625*h**3*(stigma)**(1/2)*(-1+4*stigma)**(1/2)*(-2+stigma*(5+24*stigma))*tau**3*(omega_m)**2)
          /(a0*mups2**(3/2)*(omega_r(h))**(1/2)*(-1+stigma)*(-3+stigma+8*stigma**2))
        + (390625*h**4*(-1+4*stigma)*(-6+stigma*(31+stigma*(357+2*stigma*(-123+5*stigma*(-151+24*stigma)))))*tau**4*(omega_m)**3)
          /(2*a0*mups2**2*omega_r(h)*(-1+stigma)**2*(-3+stigma+8*stigma**2)*(-6+stigma+20*stigma**2)))

def cpseries(tau, omega_m, h, OmL, a0, stigma, mups2):
    return (1 - (25*h*(-1+4*stigma)**(1/2)*tau*(omega_m))/(2*(mups2)**(1/2)*(omega_r(h))**(1/2)*(stigma)**(1/2))
        + (1875*h**2*(-1+4*stigma)*(2+stigma*(-2+3*stigma))*tau**2*(omega_m)**2)
          /(8*mups2*omega_r(h)*(-1+stigma)*stigma*(-3+stigma+8*stigma**2))
        + (15625*h**3*(-1+4*stigma)**(3/2)*(-36+stigma*(72+stigma*(175+stigma*(-225+4*stigma*(-180+stigma*(95+48*stigma))))))*tau**3*(omega_m)**3)
          /(16*mups2**(3/2)*omega_r(h)**(3/2)*(-1+stigma)**2*stigma**(3/2)*(-3+stigma+8*stigma**2)*(-6+stigma+20*stigma**2)))

def aseriesp(tau, omega_m, h, OmL, a0, stigma, mups2):
    return ((200*h*(omega_r(h))**(1/2)*(stigma)**(1/2))/(a0**2*mups2*(-1+4*stigma))**(1/2)
        + (5000*h**2*tau*(omega_m))/(a0*mups2)
        + (46875*h**3*(stigma)**(1/2)*(-1+4*stigma)**(1/2)*(-2+stigma*(5+24*stigma))*tau**2*(omega_m)**2)
          /(a0*mups2**(3/2)*(omega_r(h))**(1/2)*(-1+stigma)*(-3+stigma+8*stigma**2))
        + (781250*h**4*(-1+4*stigma)*(-6+stigma*(31+stigma*(357+2*stigma*(-123+5*stigma*(-151+24*stigma)))))*tau**3*(omega_m)**3)
          /(a0*mups2**2*omega_r(h)*(-1+stigma)**2*(-3+stigma+8*stigma**2)*(-6+stigma+20*stigma**2)))

def cpseriesp(tau, omega_m, h, OmL, a0, stigma, mups2):
    return ((-25*h*(-1+4*stigma)**(1/2)*(omega_m))/(2*(mups2)**(1/2)*(omega_r(h))**(1/2)*(stigma)**(1/2))
        + (1875*h**2*(-1+4*stigma)*(2+stigma*(-2+3*stigma))*tau*(omega_m)**2)
          /(4*mups2*omega_r(h)*(-1+stigma)*stigma*(-3+stigma+8*stigma**2))
        + (46875*h**3*(-1+4*stigma)**(3/2)*(-36+stigma*(72+stigma*(175+stigma*(-225+4*stigma*(-180+stigma*(95+48*stigma))))))*tau**2*(omega_m)**3)
          /(16*mups2**(3/2)*omega_r(h)**(3/2)*(-1+stigma)**2*stigma**(3/2)*(-3+stigma+8*stigma**2)*(-6+stigma+20*stigma**2)))

def small(omega_m, h, stigma, mups2):
    v1_abs = 25*h*np.sqrt(4*stigma-1)*omega_m/(2*np.sqrt(mups2)*np.sqrt(omega_r(h))*np.sqrt(stigma))
    return min(1e-8, 1e-2*abs(1-stigma))/v1_abs


