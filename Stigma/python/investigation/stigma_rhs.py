import sys
import numpy as np
from scipy.integrate import solve_ivp
import upward_seed as u
from upward_seed import omega_r

def ddcp(a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2):
    return ((-120000*a**4*a0**4*h**2*OmL*stigma*(-cp**2 + stigma)**2*(stigma - 4*stigma**2 + cp**2*(-1 + 4*stigma**2)) + 40000*h**2*omega_r(h)*stigma*(-cp**2 + stigma)**2*(stigma - 4*stigma**2 + cp**2*(-1 + 4*stigma**2)) - a0**2*cp**2*da**2*mups2*((5 - 8*stigma)*stigma + cp**2*(-1 + 4*stigma**2))*((7 - 4*stigma)*stigma**2 + 2*cp**2*stigma*(1 + 2*(-3 + stigma)*stigma) + cp**4*(-1 + 4*stigma**2)) + a**2*a0**2*dcp**2*mups2*(stigma**3 + 4*stigma**4 - 32*stigma**5 - cp**4*stigma*(-1 + 2*stigma)*(1 + 2*stigma)*(3 + 4*(-3 + stigma)*stigma) - cp**6*(1 - 4*stigma**2)**2 + cp**2*stigma**2*(-3 + 8*stigma + 16*stigma**3)) - 2*a*a0**2*cp*da*dcp*mups2*(cp**4*stigma*(-1 + 2*stigma)*(1 + 2*stigma)*(11 + 4*(-4 + stigma)*stigma) + cp**6*(1 - 4*stigma**2)**2 + stigma**3*(-1 + 16*stigma*(-2 + 3*stigma)) + cp**2*stigma**2*(11 - 16*stigma*(1 + stigma*(-1 + 2*stigma)))))/(8*a**2*a0**2*cp*mups2*stigma*(-cp**2 + stigma)**2*(-1 + 4*stigma**2)))

def dda(a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2):
    return (a**2*a0**2*dcp**2*mups2*(3*cp**2*stigma**2 + cp**4*(1 - 2*stigma)**2*stigma*(-3 + 2*stigma)*(1 + 2*stigma) - stigma**3*(1 + 8*stigma) + cp**6*(1 - 4*stigma**2)**2) + 120000*a**4*a0**4*h**2*OmL*stigma*(-cp**2 + stigma)**2*(stigma + cp**2*(-1 + 4*stigma**2)) - 40000*h**2*omega_r(h)*stigma*(-cp**2 + stigma)**2*(stigma + cp**2*(-1 + 4*stigma**2)) + 2*a*a0**2*cp*da*dcp*mups2*(stigma + cp**2*(-1 + 4*stigma**2))*((7 - 4*stigma)*stigma**2 + 2*cp**2*stigma*(1 + 2*(-3 + stigma)*stigma) + cp**4*(-1 + 4*stigma**2)) + a0**2*cp**2*da**2*mups2*(cp**4*(1 - 2*stigma)**2*stigma*(-7 + 2*stigma)*(1 + 2*stigma) + stigma**3*(3 + 32*(-1 + stigma)*stigma) + cp**6*(1 - 4*stigma**2)**2 + cp**2*stigma**2*(3 - 16*(-1 + stigma)*stigma*(-1 + 2*stigma))))/(8*a*a0**2*cp**2*mups2*stigma*(-cp**2 + stigma)**2*(-1 + 4*stigma**2))

def coupdiff(tau, y, omega_m, h, OmL, a0, stigma, mups2):
    a, cp, da, dcp = y
    return [da, dcp, dda(a,da,cp,dcp,omega_m,h,OmL,a0,stigma,mups2),
                     ddcp(a,da,cp,dcp,omega_m,h,OmL,a0,stigma,mups2)]

