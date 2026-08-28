import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy import constants
import matplotlib.pyplot as plt
import camb
MP2 = 1

def omega_r(h):
    ab = (8*constants.pi**5*constants.k**4)/(15*constants.h**3*constants.c**3)
    rhor = ab*2.7255**4/(constants.c**2)
    rhov = 3.046*(7/8)*(4/11)**(4/3)*rhor
    return((rhor+rhov)/((3*(100*h)**2/(8*constants.pi*constants.G))*(1000)**2/(10**6*constants.parsec)**2))

#power series for a and cp(varpi)
def aseries(tau, omega_m, h, OmL, a0, stigma, mups2):
    val = (
        (200*h*(omega_r(h))**(1/2)*(stigma)**(1/2)*tau)/(a0*(mups2)**(1/2)*(-1 + \
        4*stigma)**(1/2)) + (2500*h**2*tau**2*(omega_m))/(a0*mups2) + \
        (15625*h**3*(stigma)**(1/2)*(-1 + 4*stigma)**(1/2)*(-2 + stigma*(5 + \
        24*stigma))*tau**3*(omega_m)**2)/(a0*mups2**(3/2)*(omega_r(h))**(1/2)*(-1 + \
        stigma)*(-3 + stigma + 8*stigma**2))
        )
    return val

def cpseries(tau, omega_m, h, OmL, a0, stigma, mups2):
    val = (
       1 - (25*h*(-1 + \
        4*stigma)**(1/2)*tau*(omega_m))/(2*(mups2)**(1/2)*(omega_r(h))**(1/2)*(stigma)**(1/2)) + (1875*h**2*(-1 + \
        4*stigma)*(2 + stigma*(-2 + \
        3*stigma))*tau**2*(omega_m)**2)/(8*mups2*omega_r(h)*(-1 + stigma)*stigma*(-3 \
        + stigma + 8*stigma**2))
    )
    return val


def aseriesp(tau, omega_m, h, OmL, a0, stigma, mups2):
    val = (
           (200*h*(omega_r(h))**(1/2)*(stigma)**(1/2))/(a0*(mups2)**(1/2)*(-1 + \
            4*stigma)**(1/2)) + (5000*h**2*tau*(omega_m))/(a0*mups2) + \
            (46875*h**3*(stigma)**(1/2)*(-1 + 4*stigma)**(1/2)*(-2 + stigma*(5 + \
            24*stigma))*tau**2*(omega_m)**2)/(a0*mups2**(3/2)*(omega_r(h))**(1/2)*(-1 + \
            stigma)*(-3 + stigma + 8*stigma**2))
    )
    return val


def cpseriesp(tau, omega_m, h, OmL, a0, stigma, mups2):
    val = (
        (-25*h*(-1 + \
        4*stigma)**(1/2)*(omega_m))/(2*(mups2)**(1/2)*(omega_r(h))**(1/2)*(stigma)**(1/2)) + (1875*h**2*(-1 + \
        4*stigma)*(2 + stigma*(-2 + \
        3*stigma))*tau*(omega_m)**2)/(4*mups2*omega_r(h)*(-1 + stigma)*stigma*(-3 + \
        stigma + 8*stigma**2))
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
    dydt = [da, dcp, dda(a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2), ddcp(a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2)]
    return dydt

#Initial conditions
def small(omega_m, h, stigma, mups2):
    val = 0.001*(np.abs((1*(stigma**3*(5 + stigma)*(-1 + 2*stigma) + 1**6*(-1 + 
			2*stigma)*(1 + 2*stigma)*(-2 + 3*(-1 + stigma)*stigma) + 
			3*1**2*stigma**2*(8 + stigma*(-17 + 3*stigma)) - 1**4*stigma*(11 + 
			stigma + 2*stigma**2*(-30 + stigma*(14 + 
			stigma))))*((omega_r(h)*stigma*(-1**2 + stigma)*-mups2)/(-1**2 + 
			5*stigma + 4*(-2 + 1**2)*stigma**2))**(1/2))/(h*(2*1**2*(-4 + 
			stigma)*(-1 + stigma)*stigma**3 + stigma**4 - 2*stigma**5 + 
			2*1**4*stigma**2*(2 + stigma)*(-3 + 4*stigma) + 2*1**6*stigma*(2 + 
			stigma - 7*stigma**2) + 1**8*(-1 + 
			4*stigma**2))*(omega_m)))/250000)
    return val

#Stops for Solve_ivp
def hstop(h):
    def event(t, y):
        return y[2]-h*100
    event.terminal = True
    event.direction = 1
    return event

def astop(t, y):
    return y[0]-100

astop.terminal = True
astop.direction = 1



def constraint(a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2):
    val = ((3*a*a0*MP2*((-2*cp*da*dcp*mups2)/a - dcp**2*mups2 - (4*(((-1 + \
            cp**2)*da)/a + cp*dcp)**2*mups2*stigma**2)/(-cp**2 + stigma) + \
            (da**2*mups2*(-cp**2 + 4*stigma))/a**2 - \
            (40000*h**2*stigma*(a**4*a0**4*OmL + omega_r(h) + \
            a*a0*(omega_m)))/(a**2*a0**2)))/(4*stigma))
    # val = ((-2*cp*da*dcp*mups2)/a - dcp**2*mups2 - (4*(((-1 + cp**2)*da)/a + \
    #         cp*dcp)**2*mups2*stigma**2)/(-cp**2 + stigma) + (da**2*mups2*(-cp**2 + \
    #         4*stigma))/a**2 - (40000*h**2*stigma*(a**4*a0**4*OmL + omega_r(h) + \
    #         a*a0*(omega_m)))/(a**2*a0**2))
    return val

def constraint_ratio(a, da, cp, a0, stigma, mups2):
    T_hd = 3*a*a0*MP2/(4*stigma) * \
           (da**2*mups2*(-cp**2 + 4*stigma))/a**2
    return T_hd

def G(a, da, cp, dcp, stigma, mups, a0):
    val = (-4*a0**4*da**2*stigma*(-cp**2 + stigma))/(mups*(-(cp**4*da**2) - 2*a*cp**3*da*dcp - a**2*cp**2*dcp**2 + 5*cp**2*da**2*stigma + 2*a*cp*da*dcp*stigma + a**2*dcp**2*stigma - 8*cp**2*da**2*stigma**2 + 4*cp**4*da**2*stigma**2 - 8*a*cp*da*dcp*stigma**2 + 8*a*cp**3*da*dcp*stigma**2 + 4*a**2*cp**2*dcp**2*stigma**2))
    return val

def rhopsol(omega_m, h, OmL, stigma, mups2):

    
    tauini = small(omega_m, h, stigma, mups2)
    
    y0 = [aseries(tauini, omega_m, h, OmL, 1, stigma, mups2), cpseries(tauini, omega_m, h, OmL, 1, stigma, mups2),
          aseriesp(tauini, omega_m, h, OmL, 1, stigma, mups2), cpseriesp(tauini, omega_m, h, OmL, 1, stigma, mups2)]
    sol = solve_ivp(lambda tau, y: coupdiff(y, tau, omega_m, h, OmL, 1, stigma, mups2),
                    [tauini, np.inf], y0,  events = hstop(h), method='Radau', rtol=1e-8, atol=1e-8)
    print(f"  solver status={sol.status}: {sol.message}")
    print(f"  tauini={tauini:.3e}  y0={[f'{v:.4g}' for v in y0]}")
    print(f"  da/dtau range: {sol.y[2].min():.4g} → {sol.y[2].max():.4g}  (need {h*100:.1f})")
    #Find a0.
    a0 = sol.y_events[0][0][0]
    
    y02 = [aseries(tauini, omega_m, h, OmL, a0, stigma, mups2), cpseries(tauini, omega_m, h, OmL, a0, stigma, mups2),
          aseriesp(tauini, omega_m, h, OmL, a0, stigma, mups2), cpseriesp(tauini, omega_m, h, OmL, a0, stigma, mups2)]
    
    sola0 = solve_ivp(lambda tau, y: coupdiff(y, tau, omega_m, h, OmL, a0, stigma, mups2),
                    [tauini, np.inf], y02, events = astop, method='Radau', rtol=1e-8, atol=1e-8)
    
    tsol = interp1d(sola0.y[0], sola0.t, bounds_error = False)
    avals = np.logspace(np.log10(sola0.y[0][0]),2,10000)
    tvals = tsol(avals)
    tvals[0] = tauini
    avals[0] = sola0.y[0][0]

    sola02 = solve_ivp(lambda tau, y: coupdiff(y, tau, omega_m, h, OmL, a0, stigma, mups2),
                    [tauini, np.inf], y02, t_eval = tvals, events = astop, method='Radau', rtol=1e-8, atol=1e-8)
    
    asol, varpisol, hsol, dvarpisol = sola02.y
    aend, vpend, hend, dvpend = sola02.y_events[0][0]
    
    if aend > asol[-1]:
        asol = np.append(asol, aend)
        hsol = np.append(hsol, hend)
        varpisol = np.append(varpisol, vpend)
        dvarpisol = np.append(dvarpisol, dvpend)

    # Calculate effective dark energy density and pressure
    rhoderiv = rho_deriveda0(asol, hsol, varpisol, dvarpisol, omega_m, h, OmL, a0, stigma, mups2)
    P_DE = P_deriveda0(asol, hsol, varpisol, dvarpisol, omega_m, h, OmL, a0, stigma, mups2) / rhoderiv[-1]
    rho_DE = rho_deriveda0(asol, hsol, varpisol, dvarpisol, omega_m, h, OmL, a0, stigma, mups2) / rhoderiv[-1]


    hubble = (a0* hsol)/asol**2
    z = 1/asol-1
    
    rhoderiv = rho_deriveda0(asol, hsol, varpisol, dvarpisol, omega_m, h, OmL, a0, stigma, mups2)
    P_DE = P_deriveda0(asol, hsol, varpisol, dvarpisol, omega_m, h, OmL, a0, stigma, mups2) / rhoderiv[-1]
    rho_DE = rho_deriveda0(asol, hsol, varpisol, dvarpisol, omega_m, h, OmL, a0, stigma, mups2) / rhoderiv[-1]
    constrain = constraint(asol, hsol, varpisol, dvarpisol, omega_m, h, OmL, a0, stigma, mups2)
    ddasol = dda(asol, hsol, varpisol, dvarpisol, omega_m, h, OmL, a0, stigma, mups2) 
    Gi = G(asol, hsol, varpisol, dvarpisol, stigma, mups2, a0)
    T_hd = constraint_ratio(asol, hsol, varpisol, a0, stigma, mups2)

    return asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE, constrain, ddasol, Gi,T_hd

def cpsq_plus(stigma, wi):
    val = ((stigma*(1 + 2*(-3 + stigma)*stigma + (-3 + 6*stigma*(1 + stigma))*wi) - 2*((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + 3*stigma*wi)**2))**(1/2))/((-1 + 4*stigma**2)*(-1 + 3*wi)))
    return val

def cpsq_minus(stigma, wi):
    val = ((stigma*(1 + 2*(-3 + stigma)*stigma + (-3 + 6*stigma*(1 + stigma))*wi) + 2*((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + 3*stigma*wi)**2))**(1/2))/((-1 + 4*stigma**2)*(-1 + 3*wi)))
    return val

def gi_sol(stigma, wi, mups2, a0):
    return ((-4*a0**4*stigma*(-1 + 3*wi))/(mups2*(stigma*(3 - 9*wi) - stigma**3*(1 + 3*wi)**2 + (stigma + 3*stigma*wi)**2 + (1 + 3*wi)*((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + 3*stigma*wi)**2))**(1/2))))

def gi_sol_2(stigma, wi, mups2, a0):
    return (-4*a0**4*stigma*(1 - 3*wi))/(mups2*(stigma**3*(1 + 3*wi)**2 + stigma*(-3 + 9*wi) - (stigma + 3*stigma*wi)**2 + (1 + 3*wi)*((-1 + stigma)**2*stigma**2*(2 - 6*wi + (stigma + 3*stigma*wi)**2))**(1/2)))

def H(a, omega_m, h):
    val = (h*100)*np.sqrt(Gsol(a)*(omega_m*(a0*a)**-3+omega_r(h)*(a0*a)**-4+OmL))
    return val

if __name__ == "__main__":
    stigma  = 1.8
    mups2   = 4/3
    omega_m = 0.3
    h       = 0.7
    OmL     = 0.7

    asol, varpisol, hsol, dvarpisol, a0, P_DE, rho_DE, constrain, ddasol, Gi, T_hd = \
        rhopsol(omega_m, h, OmL, stigma, mups2)

    weff = 1/3 - (2*asol*ddasol)/(3*hsol**2)
    Gsol = interp1d(asol, Gi, bounds_error=False, fill_value="extrapolate")

    fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)

    # Panel 1: varpi^2
    ax = axes[0]
    ax.plot(asol, varpisol**2, color='C0', label=r'$\varpi^2$')
    ax.axhline(cpsq_minus(stigma, 0),  color='C1', linestyle='-',  label=r'$\varpi^2_-$, $w_i=0$')
    ax.axhline(cpsq_minus(stigma, -1), color='C2', linestyle='--', label=r'$\varpi^2_-$, $w_i=-1$')
    ax.set_ylabel(r'$\varpi^2$')
    ax.set_xscale('log')
    ax.legend(fontsize=8)

    # Panel 2: G_Newton
    ax = axes[1]
    ax.plot(asol, Gi, color='C0', label=r'$G_N$')
    ax.axhline(gi_sol_2(stigma, 0,  mups2, a0), color='C1', linestyle='-',  label=r'$G_{N,-}$, $w_i=0$')
    ax.axhline(gi_sol_2(stigma, -1, mups2, a0), color='C2', linestyle='--', label=r'$G_{N,-}$, $w_i=-1$')
    ax.set_ylabel(r'$G_N$')
    ax.set_xscale('log')
    ax.legend(fontsize=8)

    # Panel 3: w_eff
    ax = axes[2]
    ax.plot(asol, weff, color='C0')
    for w in [1/3, 0, -1]:
        ax.axhline(w, color='grey', linestyle=':', linewidth=0.8)
    ax.set_ylabel(r'$w_{\rm eff}$')
    ax.set_xscale('log')

    # Panel 4: w_DE
    ax = axes[3]
    ax.plot(asol, P_DE / rho_DE, color='C0')
    for w in [1/3, 0, -1]:
        ax.axhline(w, color='grey', linestyle=':', linewidth=0.8)
    ax.set_ylabel(r'$w_{\rm DE}$')
    ax.set_ylim(-2, 2)
    ax.set_xscale('log')

    # Panel 5: Hubble horizon
    ax = axes[4]
    ax.plot(asol, h*100/(asol*H(asol, omega_m, h)))
    ax.plot(asol, h*100*asol/(a0*hsol), "--")
    ax.plot(asol, h*100/(asol*((a0*hsol)/asol**2)), "-.")
    ax.set_ylabel('Hubble horizon')
    ax.set_xlabel(r'$a$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    for ax in axes:
        ax.axvline(1, color='grey', linestyle=':', linewidth=0.8)

    plt.tight_layout()
    plt.savefig(f'varpi_cpr1_stigma{stigma}_mups2{mups2}.pdf', bbox_inches='tight')
    plt.show()
