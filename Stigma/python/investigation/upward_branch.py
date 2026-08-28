"""
Upward (scaling) branch of the Stigma Q-torsion background, for stigma < 1.

This is NOT the varpi_0 = 1 Taylor branch. That branch has varpi' < 0 for all
stigma > 1/4 and, for stigma < 1, runs into the surface varpi^2 = stigma where
the Phi elimination is singular.

The branch here is the second root of the (quadratic) modified Friedmann
constraint. In a pure radiation background the full system admits the EXACT
scaling solution

    a(tau)    = A * tau**stigma
    varpi(tau)= W * tau**(1 - stigma)
    A * W     = 2*H0*sqrt( OmR*stigma / (mups2*(4*stigma**2 - 1)) )

verified symbolically: both the constraint and the varpi equation vanish
identically on it.  Requires stigma > 1/2 (else 4*stigma^2-1 < 0).

Note stigma = 1 recovers a ~ tau, varpi = const, i.e. the standard radiation
era -- which is exactly why stigma = 1 is the special point.

Only the product A*W is fixed; the split is the overall normalisation of a.

Conventions: conformal time tau (dtau/dt = 1/a), mups2 = -ups2 > 0.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ---------------------------------------------------------------- background
def amplitude(stigma, OmR, H0=1.0, mups2=4.0 / 3.0):
    """A*W for the exact radiation scaling solution."""
    if stigma <= 0.5:
        raise ValueError("scaling branch needs stigma > 1/2 (4*stigma^2-1 > 0)")
    return 2.0 * H0 * np.sqrt(OmR * stigma / (mups2 * (4.0 * stigma**2 - 1.0)))


def matter_correction(stigma, A, OmM, OmR):
    """
    First matter correction, expanding in u = tau**stigma:
        a     = A*tau**s     * (1 + c1*tau**s)
        varpi = W*tau**(1-s) * (1 + d1*tau**s)
    c1*P + d1*Q = 0 with P,Q below (verified symbolically).
    """
    s = stigma
    Q = 1.0 + s - 6.0 * s**2 + 2.0 * s**3
    P = 1.0 + 3.0 * s - 12.0 * s**2 + 2.0 * s**3
    den = 4.0 * OmR * s * (1.0 + s) * (3.0 * s - 1.0)
    c1 = A * OmM * Q / den
    d1 = -A * OmM * P / den
    return c1, d1


def seed(tau0, stigma, A, OmR, OmM=0.0, H0=1.0, mups2=4.0 / 3.0, matter=True):
    """Initial (a, varpi, a', varpi') on the scaling branch at tau0."""
    s = stigma
    W = amplitude(s, OmR, H0, mups2) / A
    c1, d1 = matter_correction(s, A, OmM, OmR) if matter else (0.0, 0.0)
    u = tau0**s
    a = A * tau0**s * (1.0 + c1 * u)
    ap = A * (s * tau0 ** (s - 1.0) + 2.0 * s * c1 * tau0 ** (2.0 * s - 1.0))
    w = W * tau0 ** (1.0 - s) * (1.0 + d1 * u)
    wp = W * ((1.0 - s) * tau0 ** (-s) + d1 * tau0 ** 0.0 * 1.0)
    # d/dtau [ W tau^(1-s) (1 + d1 tau^s) ] = W[(1-s)tau^-s + d1 tau^0]
    return np.array([a, w, ap, wp])


# ------------------------------------------------------- constraint (exact)
def constraint(y, stigma, OmR, OmM, OmL, H0=1.0, mups2=4.0 / 3.0):
    """
    Modified Friedmann constraint, conformal time. Should stay ~0.
      4 H0^2 s (w^2 - s) * rho~
        = mups2 { w^2[(4s^2-1)w^2 + s(5-8s)] a'^2
                + 2 a w [(4s^2-1)w^2 - s(4s-1)] a' w'
                + a^2 [s + (4s^2-1)w^2] w'^2 }
    """
    a, w, ap, wp = y
    s = stigma
    rho = OmR + OmM * a + OmL * a**4
    lhs = 4.0 * H0**2 * s * (w**2 - s) * rho
    rhs = mups2 * (
        w**2 * ((4 * s**2 - 1) * w**2 + s * (5 - 8 * s)) * ap**2
        + 2 * a * w * ((4 * s**2 - 1) * w**2 - s * (4 * s - 1)) * ap * wp
        + a**2 * (s + (4 * s**2 - 1) * w**2) * wp**2
    )
    return lhs - rhs


def wp_roots(a, w, ap, stigma, OmR, OmM, OmL, H0=1.0, mups2=4.0 / 3.0):
    """Both roots of the constraint, solved exactly for varpi'."""
    s = stigma
    rho = OmR + OmM * a + OmL * a**4
    c2 = -mups2 * a**2 * (s + (4 * s**2 - 1) * w**2)
    c1 = -2 * mups2 * a * w * ((4 * s**2 - 1) * w**2 - s * (4 * s - 1))
    c0 = 4 * H0**2 * s * (w**2 - s) * rho - mups2 * w**2 * (
        (4 * s**2 - 1) * w**2 + s * (5 - 8 * s)
    ) * ap**2
    disc = c1**2 - 4 * c2 * c0
    if disc < 0:
        return None
    r = np.sqrt(disc)
    return np.array([(-c1 + r) / (2 * c2), (-c1 - r) / (2 * c2)])


def constraint_terms(y, stigma, OmR, OmM, OmL, H0=1.0, mups2=4.0 / 3.0):
    """Return (lhs, rhs) separately so the residual can be scaled sensibly."""
    a, w, ap, wp = y
    s = stigma
    rho = OmR + OmM * a + OmL * a**4
    lhs = 4.0 * H0**2 * s * (w**2 - s) * rho
    rhs = mups2 * (
        w**2 * ((4 * s**2 - 1) * w**2 + s * (5 - 8 * s)) * ap**2
        + 2 * a * w * ((4 * s**2 - 1) * w**2 - s * (4 * s - 1)) * ap * wp
        + a**2 * (s + (4 * s**2 - 1) * w**2) * wp**2
    )
    return lhs, rhs


if __name__ == "__main__":
    stigma, OmR, OmM, OmL = 0.6, 9e-5, 0.31, 0.69
    A = 1.0

    print(f"stigma = {stigma}:   a ~ tau^{stigma},   varpi ~ tau^{1-stigma:.3f}")
    print(f"A*W = {amplitude(stigma, OmR):.8e}")
    c1, d1 = matter_correction(stigma, A, OmM, OmR)
    print(f"c1 = {c1:.6e}   d1 = {d1:.6e}")

    print("\nRELATIVE constraint residual of the seed, |lhs-rhs|/max(|lhs|,|rhs|):")
    print(f"  {'tau0':>8}  {'leading only':>14}  {'with matter':>14}")
    for tau0 in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
        row = []
        for m in (False, True):
            y0 = seed(tau0, stigma, A, OmR, OmM, matter=m)
            L, R = constraint_terms(y0, stigma, OmR, OmM, OmL)
            row.append(abs(L - R) / max(abs(L), abs(R)))
        print(f"  {tau0:8.0e}  {row[0]:14.3e}  {row[1]:14.3e}")

    print("\nExact scaling law check (pure radiation, OmM=OmL=0):")
    for tau0 in [1e-6, 1e-8]:
        y0 = seed(tau0, stigma, A, OmR, 0.0, matter=False)
        L, R = constraint_terms(y0, stigma, OmR, 0.0, 0.0)
        print(f"  tau0={tau0:.0e}  |lhs-rhs|/|lhs| = {abs(L-R)/abs(L):.3e}   (should be ~machine eps)")

    print("\nBoth constraint roots for varpi' at the seed (tau0=1e-6):")
    y0 = seed(1e-6, stigma, A, OmR, OmM)
    rts = wp_roots(y0[0], y0[1], y0[2], stigma, OmR, OmM, OmL)
    print(f"  series varpi' = {y0[3]:.6e}")
    if rts is None:
        print("  exact roots   = none (discriminant < 0 at this a')")
        print("  -> seed a' and varpi' must be varied together; use the series pair")
    else:
        print(f"  exact roots   = {rts[0]:.6e}, {rts[1]:.6e}")

    print("\n(a*varpi)' should be constant = A*W:")
    for tau0 in [1e-6, 1e-5, 1e-4]:
        y0 = seed(tau0, stigma, A, OmR, 0.0, matter=False)
        a, w, ap, wp = y0
        print(f"  tau={tau0:.0e}   (a*w)' = {ap*w + a*wp:.8e}")
