"""
STEP 2 — Replacing solve_ivp with diffrax.
==========================================

Run me with:
    source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
    cd /Users/starfish/Desktop/PhD2025-26/distances/TorC
    python jax_tutorial/step2_ode_diffrax.py

This file re-implements rhopsol() from test_theory_cpr1.py in JAX.
The physics equations (dda, ddcp, series ICs, P/rho) are IMPORTED UNCHANGED
— step 1 showed they are already JAX-compatible. Only the solver changes.

-----------------------------------------------------------------------------
THE TWO IDEAS
-----------------------------------------------------------------------------

IDEA 1: integrate in x = ln(a) instead of conformal time tau.

    The scipy version needed THREE solve_ivp calls:
      (a) solve until da = 100h to calibrate a0          [hstop event]
      (b) solve until a = 1, to learn the map tau(a)      [astop event]
      (c) re-solve with t_eval on tau(a-grid)             [astop event]
    Calls (b)+(c) exist only because we want output on a nice grid in a,
    but tau is the integration variable and we don't know tau(a) upfront.

    Fix: make a the integration variable. Since da/dtau > 0 always, a is
    monotonic in tau and we may use x = ln(a) as "time". Chain rule:
        dx/dtau = (da/dtau)/a          =>   d(.)/dx = d(.)/dtau * a/da
    The state shrinks from (a, cp, da, dcp) to (cp, da, dcp) — a IS the
    clock now — and the a=1 stop becomes simply "integrate until x = 0".
    One solve, outputs saved exactly on the log-spaced a-grid we want.

IDEA 2: the hstop calibration becomes a diffrax Event.

    Pass 1 still needs an event ("stop when da = 100h, report where").
    diffrax.Event(cond_fn) stops integration when cond_fn crosses zero,
    and an optimistix root-finder polishes the crossing location — the
    equivalent of scipy's event.terminal = True + root refinement.
-----------------------------------------------------------------------------
"""

import sys, os
import jax
jax.config.update("jax_enable_x64", True)   # BEFORE anything else
import jax.numpy as jnp
import diffrax
import optimistix as optx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_theory_cpr1 import (aseries, cpseries, aseriesp, cpseriesp,
                              dda, ddcp, P_deriveda0, rho_deriveda0,
                              omega_r, small)

# Number of output points on the log-spaced a-grid (scipy version used 10000).
# Keep it FIXED: jit recompiles if array shapes change (see step 1, demo on
# retracing). It is a compile-time constant, so plain python int.
N_GRID = 1000


# ---------------------------------------------------------------------------
# The vector field in x = ln(a)
# ---------------------------------------------------------------------------
# diffrax wants f(t, y, args) -> dy/dt. Here t is x, y = (cp, da, dcp).
# dda/ddcp are YOUR functions, imported, untouched.

def vector_field(x, y, args):
    cp, da, dcp = y
    omega_m, h, OmL, a0, stigma, mups2 = args
    a = jnp.exp(x)
    fac = a / da                       # = dtau/dx, from the chain rule
    return (dcp * fac,
            dda(a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2) * fac,
            ddcp(a, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2) * fac)


TERM   = diffrax.ODETerm(vector_field)
SOLVER = diffrax.Tsit5()               # adaptive RK, the analogue of scipy RK45
CTRL   = diffrax.PIDController(rtol=1e-8, atol=1e-8)   # = rtol/atol in solve_ivp


def initial_conditions(omega_m, h, OmL, a0, stigma, mups2):
    """Series ICs at tau_ini, exactly as in rhopsol, then switch to x=ln(a)."""
    tauini = small(omega_m, h, stigma, mups2)
    a_ini  = aseries(  tauini, omega_m, h, OmL, a0, stigma, mups2)
    y0 = (cpseries( tauini, omega_m, h, OmL, a0, stigma, mups2),
          aseriesp( tauini, omega_m, h, OmL, a0, stigma, mups2),
          cpseriesp(tauini, omega_m, h, OmL, a0, stigma, mups2))
    return jnp.log(a_ini), y0


# ---------------------------------------------------------------------------
# Pass 1 — calibrate a0: integrate with a0=1 until da = 100h  (was: hstop)
# ---------------------------------------------------------------------------

def calibrate_a0(omega_m, h, OmL, stigma, mups2):
    x0, y0 = initial_conditions(omega_m, h, OmL, 1.0, stigma, mups2)
    args = (omega_m, h, OmL, 1.0, stigma, mups2)

    # cond_fn returns a float; the event fires where it crosses zero.
    # The Newton root-finder then locates the crossing to high precision
    # (scipy did the same internally with brentq).
    def cond_fn(t, y, args, **kwargs):
        return y[1] - 100.0 * h

    event = diffrax.Event(cond_fn, root_finder=optx.Newton(rtol=1e-8, atol=1e-8))

    sol = diffrax.diffeqsolve(
        TERM, SOLVER,
        t0=x0, t1=x0 + 45.0,       # generous upper bound; event stops us long before
        dt0=None,                  # PIDController picks the first step
        y0=y0, args=args,
        stepsize_controller=CTRL,
        event=event,
        max_steps=65536,
    )
    # With an event, the last saved (t, y) is AT the event. a0 = a(event) = e^x.
    return jnp.exp(sol.ts[-1])


# ---------------------------------------------------------------------------
# Pass 2 — the real solve: ICs with calibrated a0, integrate to x=0 (a=1),
#           output on a fixed log-spaced a-grid                (was: astop +
#           interpolate tau(a) + third solve with t_eval)
# ---------------------------------------------------------------------------

def rhopsol_jax(omega_m, h, OmL, stigma, mups2):
    """JAX/diffrax version of test_theory_cpr1.rhopsol.

    Returns (z, H, P_DE, rho_DE, asol) with identical conventions:
    z and H ascending in z; P_DE/rho_DE/asol ascending in a.
    """
    a0 = calibrate_a0(omega_m, h, OmL, stigma, mups2)

    x0, y0 = initial_conditions(omega_m, h, OmL, a0, stigma, mups2)
    args = (omega_m, h, OmL, a0, stigma, mups2)

    xs = jnp.linspace(x0, 0.0, N_GRID)          # log-spaced in a, ends AT a=1

    sol = diffrax.diffeqsolve(
        TERM, SOLVER,
        t0=x0, t1=0.0, dt0=None,
        y0=y0, args=args,
        stepsize_controller=CTRL,
        saveat=diffrax.SaveAt(ts=xs),           # dense output on our grid
        max_steps=65536,
    )

    asol = jnp.exp(xs)
    cp, da, dcp = sol.ys

    hubble = a0 * da / asol**2                  # same as scipy version
    z      = 1.0 / asol - 1.0

    rhoderiv = rho_deriveda0(asol, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2)
    P_DE     = P_deriveda0(  asol, da, cp, dcp, omega_m, h, OmL, a0, stigma, mups2) / rhoderiv[-1]
    rho_DE   = rhoderiv / rhoderiv[-1]

    return z[::-1], hubble[::-1], P_DE, rho_DE, asol


# ===========================================================================
# VALIDATION + TIMING (runs when you execute this file)
# ===========================================================================
if __name__ == "__main__":
    import time
    import numpy as np
    from test_theory_cpr1 import rhopsol as rhopsol_scipy

    h = 0.7
    omega_m = 0.2969                    # DESI DR2 LCDM best fit
    OmL = 1 - omega_m - omega_r(h)

    # --- correctness: compare E(z) and w(z) against the scipy version ------
    print(f"{'stigma':>8} {'mups2':>6} | {'a0 rel diff':>12} {'max|dE/E|':>12} {'max|dw|':>10}")
    for stigma, mups2 in [(1.0001, 4/3), (1.01, 4/3), (1.5, 4/3),
                          (1.1, 0.8),   (1.3, 1.8)]:
        zs_j, H_j, P_j, rho_j, a_j = rhopsol_jax(omega_m, h, OmL, stigma, mups2)
        zs_s, H_s, P_s, rho_s, a_s = rhopsol_scipy(None, omega_m, h, OmL, stigma, mups2)

        # compare on a common z grid (each code has its own output grid)
        ztest = np.linspace(0.0, 10.0, 200)
        E_j = np.interp(ztest, np.asarray(zs_j), np.asarray(H_j / H_j[0]))
        E_s = np.interp(ztest, zs_s, H_s / H_s[0])

        w_j = np.interp(ztest, np.asarray(zs_j), np.asarray((P_j / rho_j)[::-1]))
        w_s = np.interp(ztest, zs_s, (P_s / rho_s)[::-1])

        # a0 comparison: scipy stores it only implicitly; recompute both ways
        a0_j = float(calibrate_a0(omega_m, h, OmL, stigma, mups2))
        H0_s = H_s[0]      # both normalise the same way; compare via E instead

        print(f"{stigma:8.4f} {mups2:6.3f} | {abs(H_j[0]/H0_s - 1):12.2e} "
              f"{np.max(np.abs(E_j/E_s - 1)):12.2e} {np.max(np.abs(w_j - w_s)):10.2e}")

    # --- speed: jit + vmap ---------------------------------------------------
    # E(z=0..) is all the likelihood will need; jit the whole pipeline.
    @jax.jit
    def get_E(stigma, mups2):
        z, H, P, rho, a = rhopsol_jax(omega_m, h, OmL, stigma, mups2)
        return H / H[0]

    t0 = time.perf_counter()
    _ = get_E(1.2, 1.1).block_until_ready()
    print(f"\njit compile + first solve : {time.perf_counter()-t0:7.2f} s   (one-off cost)")

    t0 = time.perf_counter()
    _ = get_E(1.3, 0.9).block_until_ready()
    t_single = time.perf_counter() - t0
    print(f"compiled single solve     : {t_single*1e3:7.1f} ms   (scipy was ~150 ms)")

    # vmap: 64 parameter points, ONE call — this is how the sampler will run
    stigmas = jnp.linspace(1.001, 1.9, 64)
    mupss   = jnp.linspace(0.7,   1.9, 64)
    get_E_batch = jax.jit(jax.vmap(get_E))
    _ = get_E_batch(stigmas, mupss).block_until_ready()      # compile
    t0 = time.perf_counter()
    _ = get_E_batch(stigmas, mupss).block_until_ready()
    t_batch = time.perf_counter() - t0
    print(f"vmap over 64 solves       : {t_batch*1e3:7.1f} ms   "
          f"({t_batch/64*1e3:.2f} ms per solve)")
