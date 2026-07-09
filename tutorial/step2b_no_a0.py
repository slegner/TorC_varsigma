"""
STEP 2b — Eliminating a0 from the dynamics (a0 = 1 everywhere).
===============================================================

Run me with:
    source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
    cd /Users/starfish/Desktop/PhD2025-26/distances/TorC
    python jax_tutorial/step2b_no_a0.py

-----------------------------------------------------------------------------
WHY a0 CAN BE REMOVED: A SCALING SYMMETRY
-----------------------------------------------------------------------------
Your equations have an exact symmetry (verified numerically to 1e-16):

    a  -> a / lam,   da -> da / lam,   a0 -> a0 * lam,   (cp, dcp unchanged)

leaves the system invariant:
    * aseries, aseriesp are exactly proportional to 1/a0,
    * cpseries, cpseriesp do not contain a0 at all,
    * dda scales as 1/lam (as d^2(a/lam)/dtau^2 must),
    * ddcp is invariant.

Consequence: the old two-pass procedure
    pass 1:  solve with a0=1, stop where da = 100h, call that point a*
    pass 2:  REDO everything with a0 = a* in the equations
was mathematically redundant — by the symmetry (lam = a*), the pass-2
trajectory is EXACTLY the pass-1 trajectory divided by a*:

    a_2(tau) = a_1(tau)/a*,   cp_2 = cp_1,   da_2 = da_1/a*,   dcp_2 = dcp_1.

So we never need a0 in the dynamics. Solve ONCE with a0 = 1, and read
everything off afterwards:

    "today"      = the point where da = 100h   (the old hstop event)
    a*           = a at that point
    a_norm       = a/a*                        (scale factor, = 1 today)
    z            = a*/a - 1
    H(z)         = a*^2 da / a^2               (H today = 100h EXACTLY)

We only need E(z) = H/H0 for the BAO likelihood (P_DE/rho_DE/w were legacy
outputs of the old pipeline — test_equations.py never actually computed
them — and they are dropped here), so the result is fully a0-free.

Bonus: H0 = 100h exactly by construction. The old calibrated version gave
H0 = 70.08 for h = 0.7 — that 0.1% wobble was pass-2 re-solve error.
-----------------------------------------------------------------------------
"""

import sys, os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax
import optimistix as optx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_theory_cpr1 import (aseries, cpseries, aseriesp, cpseriesp,
                              dda, ddcp, omega_r, small)

N_GRID = 1000


# ---------------------------------------------------------------------------
# The a0-free equations: just the imported ones evaluated at a0 = 1.
# (Rewriting the algebra to delete the symbol would risk transcription
#  errors in those enormous expressions; a0=1 is the same thing.)
# ---------------------------------------------------------------------------

def dda_noa0(a, da, cp, dcp, omega_m, h, OmL, stigma, mups2):
    return dda(a, da, cp, dcp, omega_m, h, OmL, 1.0, stigma, mups2)

def ddcp_noa0(a, da, cp, dcp, omega_m, h, OmL, stigma, mups2):
    return ddcp(a, da, cp, dcp, omega_m, h, OmL, 1.0, stigma, mups2)


def vector_field(x, y, args):
    """RHS in x = ln(a); state y = (cp, da, dcp). No a0."""
    cp, da, dcp = y
    omega_m, h, OmL, stigma, mups2 = args
    a = jnp.exp(x)
    fac = a / da
    return (dcp * fac,
            dda_noa0(a, da, cp, dcp, omega_m, h, OmL, stigma, mups2) * fac,
            ddcp_noa0(a, da, cp, dcp, omega_m, h, OmL, stigma, mups2) * fac)


TERM   = diffrax.ODETerm(vector_field)
SOLVER = diffrax.Tsit5()
CTRL   = diffrax.PIDController(rtol=1e-8, atol=1e-8)


def rhopsol_jax(omega_m, h, OmL, stigma, mups2):
    """Fully a0-free solve. Returns (z ascending, H ascending-in-z, a_norm
    ascending). H(z=0) = 100h exactly; the likelihood uses E(z) = H/(100h)."""

    # --- series ICs at tau_ini, with a0 = 1, once ---------------------------
    tauini = small(omega_m, h, stigma, mups2)
    x0 = jnp.log(aseries(tauini, omega_m, h, OmL, 1.0, stigma, mups2))
    y0 = (cpseries( tauini, omega_m, h, OmL, 1.0, stigma, mups2),
          aseriesp( tauini, omega_m, h, OmL, 1.0, stigma, mups2),
          cpseriesp(tauini, omega_m, h, OmL, 1.0, stigma, mups2))
    args = (omega_m, h, OmL, stigma, mups2)

    # --- find "today": da = 100h  (the old hstop event) ---------------------
    def cond_fn(t, y, args, **kwargs):
        return y[1] - 100.0 * h

    sol_ev = diffrax.diffeqsolve(
        TERM, SOLVER, t0=x0, t1=x0 + 45.0, dt0=None, y0=y0, args=args,
        stepsize_controller=CTRL,
        event=diffrax.Event(cond_fn, root_finder=optx.Newton(rtol=1e-8, atol=1e-8)),
        max_steps=65536,
    )
    xstar = sol_ev.ts[-1]            # ln(a*): where H = 100h, i.e. today

    # --- the same trajectory again, saved on a grid ending exactly today ----
    # (Same ICs, same equations — nothing is recalibrated. This solve exists
    #  only because SaveAt needs to know its grid, and the grid ends at
    #  xstar, which we learn from the event.)
    xs = jnp.linspace(x0, xstar, N_GRID)
    sol = diffrax.diffeqsolve(
        TERM, SOLVER, t0=x0, t1=xstar, dt0=None, y0=y0, args=args,
        stepsize_controller=CTRL, saveat=diffrax.SaveAt(ts=xs),
        max_steps=65536,
    )
    cp, da, dcp = sol.ys

    # --- physical quantities via the symmetry -------------------------------
    a_norm = jnp.exp(xs - xstar)             # = a/a*, equals 1 today
    z      = 1.0 / a_norm - 1.0
    hubble = da * jnp.exp(2.0 * (xstar - xs))  # = a*^2 da / a^2 ; = 100h today

    return z[::-1], hubble[::-1], a_norm


# ===========================================================================
# VALIDATION: must reproduce step 2 (calibrated-a0) and scipy exactly-ish
# ===========================================================================
if __name__ == "__main__":
    import time
    import numpy as np
    from step2_ode_diffrax import rhopsol_jax as rhopsol_calibrated
    from test_theory_cpr1 import rhopsol as rhopsol_scipy

    h = 0.7
    omega_m = 0.2969
    OmL = 1 - omega_m - omega_r(h)

    print("H0 check (should be exactly 70):",
          float(rhopsol_jax(omega_m, h, OmL, 1.2, 1.1)[1][0]))
    print()
    print(f"{'stigma':>8} {'mups2':>6} | {'vs step2: max|dE/E|':>20} | {'vs scipy: max|dE/E|':>20}")
    for stigma, mups2 in [(1.01, 4/3), (1.5, 4/3), (1.1, 0.8), (1.3, 1.8)]:
        zb, Hb, ab = rhopsol_jax(omega_m, h, OmL, stigma, mups2)
        zc, Hc, *_ = rhopsol_calibrated(omega_m, h, OmL, stigma, mups2)
        zs, Hs, *_ = rhopsol_scipy(None, omega_m, h, OmL, stigma, mups2)

        ztest = np.linspace(0.0, 10.0, 200)
        Eb = np.interp(ztest, np.asarray(zb), np.asarray(Hb / Hb[0]))
        Ec = np.interp(ztest, np.asarray(zc), np.asarray(Hc / Hc[0]))
        Es = np.interp(ztest, zs, Hs / Hs[0])

        print(f"{stigma:8.4f} {mups2:6.3f} | {np.max(np.abs(Eb/Ec-1)):20.2e} "
              f"| {np.max(np.abs(Eb/Es-1)):20.2e}")

    # timing
    @jax.jit
    def get_E(stigma, mups2):
        z, H, a = rhopsol_jax(omega_m, h, OmL, stigma, mups2)
        return H / H[0]
    _ = get_E(1.2, 1.1).block_until_ready()
    t0 = time.perf_counter(); _ = get_E(1.3, 0.9).block_until_ready()
    print(f"\ncompiled single solve: {(time.perf_counter()-t0)*1e3:.1f} ms")
