"""
Upward (scaling) branch seed for stigma < 1 -- drop-in companion to
aseries / cpseries / aseriesp / cpseriesp.

Those four seed the varpi_0 = 1 Taylor branch, which has varpi' < 0 for every
stigma > 1/4.  For stigma < 1 that runs down into varpi**2 = stigma, where
(-cp**2 + stigma) in the denominators of dda/ddcp vanishes.

The functions here seed the OTHER root of the (quadratic) constraint.  In a
pure radiation background the full system has the EXACT scaling solution

    a_phys(tau) = A * tau**stigma
    varpi(tau)  = W * tau**(1 - stigma)
    A * W       = 2*H0*sqrt( omega_r*stigma / (mups2*(4*stigma**2 - 1)) )

(both the constraint and the varpi equation vanish identically on it).
stigma = 1 gives a ~ tau, varpi = const, i.e. the ordinary radiation era --
which is why stigma = 1 is the special point.

Conventions match the existing code: H0 = 100*h, a_phys = a0 * a, conformal
time tau, mups2 = -ups2 > 0.

REQUIRES stigma > 1/2  (else 4*stigma**2 - 1 < 0 and A*W is imaginary).
"""

import numpy as np
from scipy import constants


def omega_r(h):
    ab = (8*constants.pi**5*constants.k**4)/(15*constants.h**3*constants.c**3)
    rhor = ab*2.7255**4/(constants.c**2)
    rhov = 3.046*(7/8)*(4/11)**(4/3)*rhor
    return((rhor+rhov)/((3*(100*h)**2/(8*constants.pi*constants.G))*(1000)**2/(10**6*constants.parsec)**2))


# ------------------------------------------------------------------ amplitude
def AW(h, stigma, mups2):
    """The fixed product A*W. Only this is determined; the split is free."""
    if stigma <= 0.5:
        raise ValueError("upward scaling branch needs stigma > 1/2")
    return 200*h*np.sqrt(omega_r(h)*stigma/(mups2*(4*stigma**2 - 1)))


def amplitudes(tau, varpi_ini, h, stigma, mups2):
    """
    Pick the split from the value you want varpi to have at tau.
    Returns (A, W) with A physical (a_phys = A*tau**stigma).

    varpi_ini < sqrt(stigma) -> starts below the lower wall, grows into it
    varpi_ini > sqrt(stigma) -> starts above it, grows toward 2*sqrt(stigma)
    """
    W = varpi_ini / tau**(1 - stigma)
    return AW(h, stigma, mups2)/W, W


# ------------------------------------------------------- matter correction
def _c1d1(A, omega_m, h, stigma):
    """
    First matter correction, expanding in u = tau**stigma:
        a_phys = A*tau**s     * (1 + c1*u)
        varpi  = W*tau**(1-s) * (1 + d1*u)
    """
    s = stigma
    Q = 1 + s - 6*s**2 + 2*s**3
    P = 1 + 3*s - 12*s**2 + 2*s**3
    den = 4*omega_r(h)*s*(1 + s)*(3*s - 1)
    return A*omega_m*Q/den, -A*omega_m*P/den


# ------------------------------------------------------------------ series
def aseries_up(tau, omega_m, h, OmL, a0, stigma, mups2, varpi_ini=1.0):
    A, W = amplitudes(tau, varpi_ini, h, stigma, mups2)
    c1, _ = _c1d1(A, omega_m, h, stigma)
    return (A/a0)*(tau**stigma + c1*tau**(2*stigma))


def cpseries_up(tau, omega_m, h, OmL, a0, stigma, mups2, varpi_ini=1.0):
    A, W = amplitudes(tau, varpi_ini, h, stigma, mups2)
    _, d1 = _c1d1(A, omega_m, h, stigma)
    return W*(tau**(1 - stigma) + d1*tau)


def aseriesp_up(tau, omega_m, h, OmL, a0, stigma, mups2, varpi_ini=1.0):
    A, W = amplitudes(tau, varpi_ini, h, stigma, mups2)
    c1, _ = _c1d1(A, omega_m, h, stigma)
    s = stigma
    return (A/a0)*(s*tau**(s - 1) + 2*s*c1*tau**(2*s - 1))


def cpseriesp_up(tau, omega_m, h, OmL, a0, stigma, mups2, varpi_ini=1.0):
    A, W = amplitudes(tau, varpi_ini, h, stigma, mups2)
    _, d1 = _c1d1(A, omega_m, h, stigma)
    return W*((1 - stigma)*tau**(-stigma) + d1)


# --------------------------------------------------------------- tau_ini
def small_up(omega_m, h, stigma, mups2, varpi_ini=1.0, eps=1e-3):
    """
    tau_ini such that the matter correction is a fraction eps of the
    leading term. Analogue of small() for this branch.
    """
    tau = 1e-6
    for _ in range(200):                     # A depends on tau, so iterate
        A, _W = amplitudes(tau, varpi_ini, h, stigma, mups2)
        c1, d1 = _c1d1(A, omega_m, h, stigma)
        m = max(abs(c1), abs(d1))
        new = (eps/m)**(1.0/stigma) if m > 0 else tau
        if abs(new - tau) < 1e-14*abs(new):
            break
        tau = new
    return tau


# ----------------------------------------------------------------- events
def wall_lower(stigma):
    """varpi**2 = stigma : the Phi-elimination singularity (dda/ddcp blow up)."""
    def event(t, y, *args):
        return y[1]**2 - stigma
    event.terminal = True
    return event


def wall_upper(stigma):
    """varpi**2 = 4*stigma : effective Planck mass changes sign. Crossed SILENTLY."""
    def event(t, y, *args):
        return y[1]**2 - 4*stigma
    event.terminal = False          # not a singularity; watch, don't stop
    return event


def constraint_residual(y, omega_m, h, OmL, a0, stigma, mups2):
    """Relative violation of the modified Friedmann constraint. Monitor this."""
    a, cp, da, dcp = y
    s, H0 = stigma, 100*h
    ap, ap_a = a0*da, a0*a
    rho = omega_r(h) + omega_m*ap_a + OmL*ap_a**4
    lhs = 4*H0**2*s*(cp**2 - s)*rho
    rhs = mups2*(cp**2*((4*s**2 - 1)*cp**2 + s*(5 - 8*s))*ap**2
                 + 2*ap_a*cp*((4*s**2 - 1)*cp**2 - s*(4*s - 1))*ap*dcp
                 + ap_a**2*(s + (4*s**2 - 1)*cp**2)*dcp**2)
    return (lhs - rhs)/max(abs(lhs), abs(rhs))


# ------------------------------------------------------------------- usage
if __name__ == "__main__":
    import numpy as np
    from scipy.integrate import solve_ivp
    from stigma_rhs import coupdiff          # your dda/ddcp, signature (tau, y, *args)

    h, omega_m, OmL, a0, mups2 = 0.67, 0.31, 0.69, 1.0, 4/3
    stigma = 0.6

    print(f"stigma={stigma}  lower wall sqrt(s)={np.sqrt(stigma):.4f} "
          f" upper wall 2sqrt(s)={2*np.sqrt(stigma):.4f}")

    for varpi_ini in (0.5*np.sqrt(stigma), 1.5*np.sqrt(stigma)):
        tau_ini = small_up(omega_m, h, stigma, mups2, varpi_ini)
        y0 = [aseries_up(tau_ini, omega_m, h, OmL, a0, stigma, mups2, varpi_ini),
              cpseries_up(tau_ini, omega_m, h, OmL, a0, stigma, mups2, varpi_ini),
              aseriesp_up(tau_ini, omega_m, h, OmL, a0, stigma, mups2, varpi_ini),
              cpseriesp_up(tau_ini, omega_m, h, OmL, a0, stigma, mups2, varpi_ini)]

        sol = solve_ivp(coupdiff, [tau_ini, 1e-2], y0,
                        args=(omega_m, h, OmL, a0, stigma, mups2),
                        events=[wall_lower(stigma), wall_upper(stigma)],
                        rtol=1e-9, atol=1e-14, method="Radau")

        print(f"\n  varpi_ini={varpi_ini:.4f}  tau_ini={tau_ini:.3e}")
        print(f"    seed |F1|rel = {abs(constraint_residual(y0,omega_m,h,OmL,a0,stigma,mups2)):.2e}")
        print(f"    ended tau={sol.t[-1]:.3e}  a={sol.y[0,-1]:.4e}  varpi^2={sol.y[1,-1]**2:.5f}")
        print(f"    lower wall hit: {len(sol.t_events[0])>0}"
              f"   upper wall crossed: {len(sol.t_events[1])>0}")
