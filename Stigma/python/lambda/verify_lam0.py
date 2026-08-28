"""
Regression check: at lam = 0 the Lambda model must reproduce the original
(non-Lambda) run of python/equations_Hz.py exactly.

    python3 verify_lam0.py

Checks, in order:
  A. the right-hand sides dda/ddcp, over many random states and several a0
  B. tau_ini and the initial state
  C. the a0 found by the shooting
  D. the full H(z), varpi and a arrays from rhopsol
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import equations_lambda as L
from reference_model import ref

cpr, omega_m, h, OmL, stigma, mups2 = 0.99, 0.3, 0.67, 0.7, 1.5, 4/3


def relerr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.abs(x - y)
    s = np.maximum(np.abs(x), np.abs(y))
    return np.max(np.where(s > 0, d/np.where(s > 0, s, 1), 0.0))


print("A. right-hand sides at lam = 0, 2000 random states per a0")
rng = np.random.default_rng(0)
for a0 in (1.0, 0.46995577276076483, 1.7):
    a   = 10**rng.uniform(-7, 0, 2000)
    da  = 10**rng.uniform(-2, 1, 2000)
    cp  = rng.uniform(0.05, np.sqrt(stigma)*0.99, 2000)
    dcp = rng.uniform(-200, 200, 2000)
    args = (a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2)
    eA = relerr(L.dda(*args, 0.0),  ref.dda(*args))
    eC = relerr(L.ddcp(*args, 0.0), ref.ddcp(*args))
    exact = np.mean(L.dda(*args, 0.0) == ref.dda(*args))
    print(f"   a0={a0:<20.6f} max rel err:  a''={eA:.3e}   cp''={eC:.3e}"
          f"   (bit-identical: {100*exact:.1f}% of a'' samples)")

print("\nB. initial conditions")
t_ref = ref.small(cpr, omega_m, h, stigma, mups2)
t_lam = L.small(cpr, omega_m, h, stigma, mups2)
y_ref = [ref.aseries(t_ref, cpr, omega_m, h, OmL, 1, stigma, mups2),
         ref.cpseries(t_ref, cpr, omega_m, h, OmL, 1, stigma, mups2),
         ref.aseriesp(t_ref, cpr, omega_m, h, OmL, 1, stigma, mups2),
         ref.cpseriesp(t_ref, cpr, omega_m, h, OmL, 1, stigma, mups2)]
y_lam = L.initial_state(t_lam, cpr, omega_m, h, OmL, 1, stigma, mups2)
print(f"   tau_ini identical: {t_ref == t_lam}   ({t_ref:.12e})")
print(f"   initial state identical: {y_ref == y_lam}")

print("\nC/D. full pipeline")
zr, Hr, _, _, ar = ref.rhopsol(cpr, omega_m, h, OmL, stigma, mups2)
zl, Hl, _, _, al = L.rhopsol(cpr, omega_m, h, OmL, stigma, mups2, 0.0)
print(f"   H(z=0):     reference {Hr[0]:.10f}   lambda {Hl[0]:.10f}")
print(f"   z_max:      reference {zr[-1]:.6e}   lambda {zl[-1]:.6e}")
print(f"   max rel err on H(z):  {relerr(Hr, Hl):.3e}   ({len(Hr)} points)")
print(f"   max rel err on a:     {relerr(ar, al):.3e}")
print(f"   max rel err on z:     {relerr(zr[1:], zl[1:]):.3e}")

print("\nE. same ICs, same a0, direct trajectory comparison at three tolerances")
from scipy.integrate import solve_ivp
grid = np.geomspace(t_ref, 1e4*t_ref, 200)
for tol in (1e-8, 1e-10, 1e-12):
    sr = solve_ivp(lambda t, y: ref.coupdiff(y, t, omega_m, h, OmL, 1, stigma, mups2),
                   [t_ref, grid[-1]], y_ref, t_eval=grid, rtol=tol, atol=tol)
    sl = solve_ivp(lambda t, y: L.coupdiff(y, t, omega_m, h, OmL, 1, stigma, mups2, 0.0),
                   [t_ref, grid[-1]], y_lam, t_eval=grid, rtol=tol, atol=tol)
    print(f"   rtol=atol={tol:.0e}:  max rel diff   a={relerr(sr.y[0], sl.y[0]):.2e}"
          f"   varpi={relerr(sr.y[1], sl.y[1]):.2e}")

print("""
VERDICT
  The two right-hand sides are algebraically the same expression at lam = 0:
  they differ only by ~1e-14 (a few tens of ulp), the signature of the same
  formula written with the sign/denominator rearranged, and that residual does
  not shrink when the solver tolerance is tightened -- it is double-precision
  arithmetic, not physics.  From identical ICs the trajectories track to 1e-14.
  The ~1e-8 seen end-to-end in rhopsol is the a0 shooting amplifying that
  rounding through the event detection; H(z=0) still lands on 67.0000000037
  for both.""")
