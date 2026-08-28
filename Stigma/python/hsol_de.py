import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy import constants
import camb.constants

def omega_r(h, nnu=camb.constants.default_nnu):
    ab = (8*constants.pi**5*constants.k**4)/(15*constants.h**3*constants.c**3)
    rhor = ab*camb.constants.COBE_CMBTemp**4/(constants.c**2)
    rhov = nnu*(7/8)*(4/11)**(4/3)*rhor
    return((rhor+rhov)/((3*(100*h)**2/(8*constants.pi*constants.G))*(1000)**2/(10**6*constants.parsec)**2))


def solve_a0(gi, h, omega_m, OmL):
    """Solve 1 = gi * (Omega_r/a0^4 + Omega_m/a0^3 + OmL) for a0."""
    OmR = omega_r(h)
    def f(a0):
        return gi * (OmR/a0**4 + omega_m/a0**3 + OmL) - 1
    return brentq(f, 0.01, 10.0)


def Hplus(h, stigma, wi, OmL, omega_m, v2, a, a0):
    return((40000*h**2*(stigma*(3 - 9*wi) - stigma**3*(1 + 3*wi)**2 + (stigma + \
            3*stigma*wi)**2 - (1 + 3*wi)*((-1 + stigma)**2*stigma**2*(2 - 6*wi + \
            (stigma + 3*stigma*wi)**2))**(1/2))*(a**4*a0**4*OmL + omega_r(h) + \
            a*a0*(omega_m)))/(a**4*a0**4*stigma*v2*(1 + 2*stigma + 6*(-1 + \
            stigma)*wi)*(-7 - 3*wi + 4*stigma*(1 + 3*wi))))

def Hminus(h, stigma, wi, OmL, omega_m, v2, a, a0):
    return((40000*h**2*(stigma*(3 - 9*wi) - stigma**3*(1 + 3*wi)**2 + (stigma + \
            3*stigma*wi)**2 + (1 + 3*wi)*((-1 + stigma)**2*stigma**2*(2 - 6*wi + \
            (stigma + 3*stigma*wi)**2))**(1/2))*(a**4*a0**4*OmL + omega_r(h) + \
            a*a0*(omega_m)))/(a**4*a0**4*stigma*v2*(1 + 2*stigma + 6*(-1 + \
            stigma)*wi)*(-7 - 3*wi + 4*stigma*(1 + 3*wi))))
