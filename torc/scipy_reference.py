"""
Updated theory equations (from test_equations.py) packaged as an importable
module with the same interface as TorC_cpr1.py, for use by the rainbow
plotting scripts.

Differences from TorC_cpr1.py:
  - aseries / cpseries / aseriesp taken from test_equations.py (series
    extended to higher order in tau).
  - cpseriesp is the term-by-term tau-derivative of the new cpseries
    (test_equations.py defines cpseries twice instead of cpseriesp).
  - dda, ddcp, P_deriveda0, rho_deriveda0 are identical in both files.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy import constants

MP2 = 1


def omega_r(h):
    ab = (8*constants.pi**5*constants.k**4)/(15*constants.h**3*constants.c**3)
    rhor = ab*2.7255**4/(constants.c**2)
    rhov = 3.046*(7/8)*(4/11)**(4/3)*rhor
    return((rhor+rhov)/((3*(100*h)**2/(8*constants.pi*constants.G))*(1000)**2/(10**6*constants.parsec)**2))


def aseries(tau, omega_m, h, OmL, a0, stigma, mups2):
    val = (
          (200*h*(omega_r(h))**(1/2)*(stigma)**(1/2)*tau)/(a0**2*mups2*(-1 + \
            4*stigma))**(1/2) + (2500*h**2*tau**2*(omega_m))/(a0*mups2) + \
            (15625*h**3*(stigma)**(1/2)*(-1 + 4*stigma)**(1/2)*(-2 + stigma*(5 + \
            24*stigma))*tau**3*(omega_m)**2)/(a0*mups2**(3/2)*(omega_r(h))**(1/2)*(-1 + \
            stigma)*(-3 + stigma + 8*stigma**2)) + (390625*h**4*(-1 + \
            4*stigma)*(-6 + stigma*(31 + stigma*(357 + 2*stigma*(-123 + \
            5*stigma*(-151 + \
            24*stigma)))))*tau**4*(omega_m)**3)/(2*a0*mups2**2*omega_r(h)*(-1 + \
            stigma)**2*(-3 + stigma + 8*stigma**2)*(-6 + stigma + 20*stigma**2))
    )
    return val


def cpseries(tau, omega_m, h, OmL, a0, stigma, mups2):
    val = (
       1 - (25*h*(-1 + \
            4*stigma)**(1/2)*tau*(omega_m))/(2*(mups2)**(1/2)*(omega_r(h))**(1/2)*(stigma)**(1/2)) + (1875*h**2*(-1 + \
            4*stigma)*(2 + stigma*(-2 + \
            3*stigma))*tau**2*(omega_m)**2)/(8*mups2*omega_r(h)*(-1 + stigma)*stigma*(-3 \
            + stigma + 8*stigma**2)) + (15625*h**3*(-1 + 4*stigma)**(3/2)*(-36 + \
            stigma*(72 + stigma*(175 + stigma*(-225 + 4*stigma*(-180 + stigma*(95 \
            + 48*stigma))))))*tau**3*(omega_m)**3)/(16*mups2**(3/2)*omega_r(h)**(3/2)*(-1 + stigma)**2*stigma**(3/2)*(-3 \
            + stigma + 8*stigma**2)*(-6 + stigma + 20*stigma**2))
    )
    return val


def aseriesp(tau, omega_m, h, OmL, a0, stigma, mups2):
    val = (
          (200*h*(omega_r(h))**(1/2)*(stigma)**(1/2))/(a0**2*mups2*(-1 + \
            4*stigma))**(1/2) + (5000*h**2*tau*(omega_m))/(a0*mups2) + \
            (46875*h**3*(stigma)**(1/2)*(-1 + 4*stigma)**(1/2)*(-2 + stigma*(5 + \
            24*stigma))*tau**2*(omega_m)**2)/(a0*mups2**(3/2)*(omega_r(h))**(1/2)*(-1 + \
            stigma)*(-3 + stigma + 8*stigma**2)) + (781250*h**4*(-1 + \
            4*stigma)*(-6 + stigma*(31 + stigma*(357 + 2*stigma*(-123 + \
            5*stigma*(-151 + \
            24*stigma)))))*tau**3*(omega_m)**3)/(a0*mups2**2*omega_r(h)*(-1 + \
            stigma)**2*(-3 + stigma + 8*stigma**2)*(-6 + stigma + 20*stigma**2))
    )
    return val


def cpseriesp(tau, omega_m, h, OmL, a0, stigma, mups2):
    # d/dtau of cpseries above: -A + 2*B*tau + 3*C*tau**2
    val = (
       - (25*h*(-1 + \
            4*stigma)**(1/2)*(omega_m))/(2*(mups2)**(1/2)*(omega_r(h))**(1/2)*(stigma)**(1/2)) + (1875*h**2*(-1 + \
            4*stigma)*(2 + stigma*(-2 + \
            3*stigma))*tau*(omega_m)**2)/(4*mups2*omega_r(h)*(-1 + stigma)*stigma*(-3 \
            + stigma + 8*stigma**2)) + (46875*h**3*(-1 + 4*stigma)**(3/2)*(-36 + \
            stigma*(72 + stigma*(175 + stigma*(-225 + 4*stigma*(-180 + stigma*(95 \
            + 48*stigma))))))*tau**2*(omega_m)**3)/(16*mups2**(3/2)*omega_r(h)**(3/2)*(-1 + stigma)**2*stigma**(3/2)*(-3 \
            + stigma + 8*stigma**2)*(-6 + stigma + 20*stigma**2))
    )
    return val


def P_deriveda0(a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2):
    return((MP2*(10000*cp**9*da*h**2*(-1 + 4*stigma**2)**3*(4*a**4*a0**4*OmL + \
            mups2*omega_r(h) + a*a0*(omega_m)) + 2*a*dcp*stigma**3*(-1 + 2*stigma)*(1 + \
            2*stigma)*(-7 + 4*stigma)*(a**2*a0**2*dcp**2*mups2 + \
            40000*h**2*stigma*(a**4*a0**4*OmL + omega_r(h) + a*a0*(omega_m))) + \
            2*a*cp**6*dcp*(1 - 4*stigma**2)**2*(a**2*a0**2*dcp**2*mups2*(1 - \
            4*stigma**2) + 40000*h**2*stigma*(a**4*a0**4*OmL + omega_r(h) + \
            a*a0*(omega_m))) - 2*a*cp**2*dcp*stigma**2*(-1 + \
            4*stigma**2)*(-(a**2*a0**2*dcp**2*mups2*(5 + 8*stigma*(1 + 2*(-2 + \
            stigma)*stigma))) + 40000*h**2*stigma*(-5 + 2*stigma)*(1 + \
            2*stigma)*(a**4*a0**4*OmL + omega_r(h) + a*a0*(omega_m))) + \
            2*a*cp**4*dcp*stigma*(-(a**2*a0**2*dcp**2*mups2*(3 + 4*(-3 + \
            stigma)*stigma)*(1 - 4*stigma**2)**2) - 120000*h**2*stigma*(-1 + \
            2*stigma)*(1 + 2*stigma)*(-1 + 4*stigma)*(a**4*a0**4*OmL + omega_r(h) + \
            a*a0*(omega_m))) + cp*da*stigma**3*(-(a**2*a0**2*dcp**2*mups2*(-1 + \
            2*stigma)*(1 + 2*stigma)*(43 + 8*stigma*(-19 + 8*stigma))) - \
            10000*h**2*stigma*(-5 + 8*stigma)*(8*a**4*a0**4*OmL*(-1 + stigma)*(1 + \
            2*stigma) + 4*omega_r(h)*(3 + 2*stigma*(-5 + 2*stigma)) + a*a0*(-7 + \
            4*stigma)*(-1 + 4*stigma)*(omega_m))) - cp**7*da*(1 - \
            4*stigma**2)**2*(3*a**2*a0**2*dcp**2*mups2*(-1 + 4*stigma**2) + \
            10000*h**2*stigma*(4*a**4*a0**4*OmL*(-11 + 2*stigma*(9 + stigma)) + \
            omega_r(h)*(4 - 8*(-1 + stigma)*stigma + mups2*(-11 + 4*stigma*(4 + stigma))) \
            - 4*a*a0*(2 + (-6 + stigma)*stigma)*(omega_m))) + \
            cp**3*da*stigma**2*(10000*h**2*omega_r(h)*stigma*(-92 - mups2*(5 - \
            8*stigma)**2*(-1 + 4*stigma**2) + 8*stigma*(21 + stigma*(69 + \
            4*stigma*(-41 + 18*stigma)))) + a*a0*(a*a0*dcp**2*mups2*(-1 + \
            2*stigma)*(1 + 2*stigma)*(43 + 8*stigma*(-1 + stigma*(-31 + 12*stigma))) \
            + 40000*h**2*stigma*(a**3*a0**3*OmL*(37 + 2*stigma*(-67 + 7*stigma*(9 \
            + 4*stigma - 8*stigma**2))) + (-2 + 5*stigma)*(4 + stigma*(11 + \
            8*stigma*(-5 + 2*stigma)))*(omega_m)))) + cp**5*da*stigma*(-1 + \
            4*stigma**2)*(10000*h**2*omega_r(h)*stigma*(mups2*(-5 + 8*stigma)*(-7 + \
            8*stigma*(1 + stigma)) - 4*(9 + 2*stigma*(1 + 2*stigma*(-13 + \
            8*stigma)))) + a*a0*(3*a*a0*dcp**2*mups2*(1 - 16*(stigma - \
            4*stigma**3 + stigma**4)) + 20000*h**2*stigma*(2*a**3*a0**3*OmL*(37 + \
            2*stigma*(-57 + 4*stigma*(9 + 4*stigma))) + (5 - 2*stigma*(30 + \
            stigma*(-57 + \
            16*stigma)))*(omega_m))))))/(a**4*cp**3*da*mups2*(-cp**2 + \
            stigma)*(-1 + 4*stigma**2)*((5 - 8*stigma)*stigma + cp**2*(-1 + \
            4*stigma**2))**2))


def rho_deriveda0(a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2):
    return((3*MP2*(-10000*h**2*omega_r(h) - 10000*a*h**2*(omega_m) + \
            (-(a**2*a0**2*dcp**2*mups2*(-stigma + cp**2*(1 - 4*stigma**2))) - \
            2*a*a0**2*cp*da*dcp*mups2*(stigma*(-1 + 4*stigma) + cp**2*(1 - \
            4*stigma**2)) + 40000*h**2*stigma*(-cp**2 + stigma)*(a**4*a0**4*OmL \
            + omega_r(h) + a*a0*(omega_m)))/(-(cp**2*mups2*(5 - 8*stigma)*stigma) - \
            cp**4*mups2*(-1 + 4*stigma**2))))/a**4)


def ddcp(a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2):
    val = (-1/8*(120000*a**4*a0**4*h**2*OmL*stigma*(-cp**2 + \
            stigma)**2*(stigma - 4*stigma**2 + cp**2*(-1 + 4*stigma**2)) - \
            40000*h**2*omega_r(h)*stigma*(-cp**2 + stigma)**2*(stigma - 4*stigma**2 + \
            cp**2*(-1 + 4*stigma**2)) + a0**2*cp**2*da**2*mups2*((5 - \
            8*stigma)*stigma + cp**2*(-1 + 4*stigma**2))*((7 - \
            4*stigma)*stigma**2 + 2*cp**2*stigma*(1 + 2*(-3 + stigma)*stigma) + \
            cp**4*(-1 + 4*stigma**2)) + a**2*a0**2*dcp**2*mups2*(stigma**3*(-1 + \
            4*stigma)*(1 + 8*stigma) + cp**4*stigma*(-1 + 2*stigma)*(1 + \
            2*stigma)*(3 + 4*(-3 + stigma)*stigma) + cp**6*(1 - 4*stigma**2)**2 + \
            cp**2*stigma**2*(3 - 8*(stigma + 2*stigma**3))) + \
            2*a*a0**2*cp*da*dcp*mups2*(cp**4*stigma*(-1 + 2*stigma)*(1 + \
            2*stigma)*(11 + 4*(-4 + stigma)*stigma) + cp**6*(1 - 4*stigma**2)**2 + \
            stigma**3*(-1 + 16*stigma*(-2 + 3*stigma)) + cp**2*stigma**2*(11 - \
            16*stigma*(1 + stigma*(-1 + \
            2*stigma)))))/(a**2*a0**2*cp*mups2*stigma*(-cp**2 + stigma)**2*(-1 + \
            4*stigma**2)))
    return val


def dda(a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2):
    val = (-1/8*(-(a**2*a0**2*dcp**2*mups2*(3*cp**2*stigma**2 + cp**4*(1 - \
            2*stigma)**2*stigma*(-3 + 2*stigma)*(1 + 2*stigma) - stigma**3*(1 + \
            8*stigma) + cp**6*(1 - 4*stigma**2)**2)) - \
            120000*a**4*a0**4*h**2*OmL*stigma*(-cp**2 + stigma)**2*(stigma + \
            cp**2*(-1 + 4*stigma**2)) + 40000*h**2*omega_r(h)*stigma*(-cp**2 + \
            stigma)**2*(stigma + cp**2*(-1 + 4*stigma**2)) - \
            2*a*a0**2*cp*da*dcp*mups2*(stigma + cp**2*(-1 + 4*stigma**2))*((7 - \
            4*stigma)*stigma**2 + 2*cp**2*stigma*(1 + 2*(-3 + stigma)*stigma) + \
            cp**4*(-1 + 4*stigma**2)) - a0**2*cp**2*da**2*mups2*(cp**4*(1 - \
            2*stigma)**2*stigma*(-7 + 2*stigma)*(1 + 2*stigma) + stigma**3*(3 + \
            32*(-1 + stigma)*stigma) + cp**6*(1 - 4*stigma**2)**2 + \
            cp**2*stigma**2*(3 - 16*(-1 + stigma)*stigma*(-1 + \
            2*stigma))))/(a*a0**2*cp**2*mups2*stigma*(-cp**2 + stigma)**2*(-1 + \
            4*stigma**2)))
    return val


def coupdiff(y, tau, omega_m, h, OmL, a0, stigma, mups2):
    a, cp, da, dcp = y
    return [da, dcp,
            dda( a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2),
            ddcp(a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2)]


def small(omega_m, h, stigma, mups2):
    return 1e-6*(2*(mups2)**(1/2)*(omega_r(h))**(1/2)*(stigma)**(1/2)) / \
           (25*h*(-1 + 4*stigma)**(1/2)*(omega_m))


def hstop(h):
    def event(t, y):
        return y[2] - h*100
    event.terminal = True
    event.direction = 1
    return event


def astop(t, y):
    return y[0] - 1   # stop at a = 1 (today) for correct H0 normalisation

astop.terminal = True
astop.direction = 1


def rhopsol(cpr, omega_m, h, OmL, stigma, mups2):
    """
    cpr is accepted for interface compatibility but ignored (always cpr=1 here).

    Returns
    -------
    z      : 1-D array, ascending redshift
    H      : 1-D array, H(z) = (a0*hsol)/asol^2, ascending-z order
    P_DE   : 1-D array, effective DE pressure, indexed by asol (ascending a)
    rho_DE : 1-D array, effective DE density,   indexed by asol (ascending a)
    asol   : 1-D array, ascending scale factor  (for CAMB a_p / a_rho)
    """
    tauini = small(omega_m, h, stigma, mups2)

    y0 = [aseries( tauini, omega_m, h, OmL, 1, stigma, mups2),
          cpseries( tauini, omega_m, h, OmL, 1, stigma, mups2),
          aseriesp( tauini, omega_m, h, OmL, 1, stigma, mups2),
          cpseriesp(tauini, omega_m, h, OmL, 1, stigma, mups2)]

    sol = solve_ivp(lambda tau, y: coupdiff(y, tau, omega_m, h, OmL, 1, stigma, mups2),
                    [tauini, np.inf], y0, events=hstop(h), rtol=1e-8, atol=1e-8)
    a0 = sol.y_events[0][0][0]

    y02 = [aseries( tauini, omega_m, h, OmL, a0, stigma, mups2),
           cpseries( tauini, omega_m, h, OmL, a0, stigma, mups2),
           aseriesp( tauini, omega_m, h, OmL, a0, stigma, mups2),
           cpseriesp(tauini, omega_m, h, OmL, a0, stigma, mups2)]

    sola0 = solve_ivp(lambda tau, y: coupdiff(y, tau, omega_m, h, OmL, a0, stigma, mups2),
                      [tauini, np.inf], y02, events=astop, rtol=1e-8, atol=1e-8)

    tsol  = interp1d(sola0.y[0], sola0.t, bounds_error=False)
    avals = np.logspace(np.log10(sola0.y[0][0]), 0, 10000)  # a_min → 1
    tvals = tsol(avals)
    tvals[0] = tauini
    avals[0] = sola0.y[0][0]

    sola02 = solve_ivp(lambda tau, y: coupdiff(y, tau, omega_m, h, OmL, a0, stigma, mups2),
                       [tauini, np.inf], y02, t_eval=tvals, events=astop,
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
    rho_DE   = rhoderiv / rhoderiv[-1]

    hubble = (a0 * hsol) / asol**2
    z      = 1.0 / asol - 1.0

    return z[::-1], hubble[::-1], P_DE, rho_DE, asol
