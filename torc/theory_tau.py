"""
theory_tau.py — the TorC theory with NO H0-epoch calibration.
==================================================================

Alternative anchoring to torc/theory.py, same interface:

    z, E, a = rhopsol_jax(omega_m, omegar, OmL, stigma, mups2)

Anchoring convention of THIS module:
    a0 = 1 is pinned in the equations, and "today" is DEFINED as the moment
    the scale factor reaches a = 1.  No reference to H0 anywhere: E(z) is
    normalised by whatever the expansion rate happens to be at a = 1.

Contrast with torc/theory.py (the standard convention):
    there, "today" is the epoch at which the input density fractions
    (omega_m, omegar, OmL) hold — located dynamically via the da = 100
    event (E = 1 with the labels true) — and a is normalised afterwards.

Interpretation warnings for this module:
  * omega_m / omegar / OmL are NOT today's density fractions here; they are
    the fractions at the (never-visited) epoch where H_code = 100.  Fitted
    values are not comparable to DESI's Omega_m.
  * E(z) differs from the calibrated version by a z-dependent factor that
    tends to a constant at high z (this is exactly the flat ~0.25 offset
    seen when comparing an uncalibrated a0 = 1 solve against a calibrated
    one at omega_m=0.289, OmL=0.9, stigma=1.1, mups2=3).
  * Physically, this convention ties the theory's internal a0 scale to the
    observation epoch.

Solver identical to torc/theory.py's tau version otherwise (diffrax,
event + steps for the tau(ln a) map, second solve on the a-grid, guarded
for sampler safety).  Self-test:  python theory_tau.py
"""

import jax
jax.config.update("jax_enable_x64", True)   # must precede any jax call
import jax.numpy as jnp
import diffrax
import optimistix as optx

# equations, series ICs and solver constants are shared with the parent
# module — only the anchoring differs, so import rather than duplicate.
from .theory import (OMEGA_R_DEFAULT, N_EARLY, N_LATE, Z_SPLIT, TAU_SPAN,
                     MP2, omega_r, aseries, cpseries, aseriesp, cpseriesp,
                     dda, ddcp, small, _TERM_TAU, _SOLVER, _CTRL)


def _a_eq_1_event():
    # "today" declared to be a = 1 in a0 = 1 units (no H0 anywhere).
    # tau state is (a, cp, da, dcp): a is y[0]
    def cond_fn(t, y, args, **kwargs):
        return y[0] - 1.0
    return diffrax.Event(cond_fn, root_finder=optx.Newton(rtol=1e-8, atol=1e-8))


def rhopsol_jax_tau(omega_m, omegar, OmL, stigma, mups2):
    """Solve the theory with a0 = 1, today := (a = 1); return (z, E, a_norm).

    Same return convention as theory_jax.rhopsol_jax: z ascending with
    z[0] = 0, E(0) = 1 exactly, a_norm ascending and ending at 1.
    """
    tauini = small(omega_m, omegar, stigma, mups2)
    # keep t0/t1 finite even for garbage parameters (nan ICs still poison E)
    tau0 = jnp.where(jnp.isfinite(tauini) & (tauini > 0), tauini, 1e-8)
    y0 = (aseries(  tauini, omega_m, omegar, OmL, stigma, mups2),
          cpseries( tauini, omega_m, omegar, OmL, stigma, mups2),
          aseriesp( tauini, omega_m, omegar, OmL, stigma, mups2),
          cpseriesp(tauini, omega_m, omegar, OmL, stigma, mups2))
    args = (omega_m, omegar, OmL, stigma, mups2)

    # --- solve 1: integrate to a = 1, keeping the accepted steps (they
    # provide the tau(ln a) map for the output grid) --------------------------
    sol_ev = diffrax.diffeqsolve(
        _TERM_TAU, _SOLVER, t0=tau0, t1=tau0 + TAU_SPAN, dt0=0.01 * tau0,
        y0=y0, args=args, stepsize_controller=_CTRL, event=_a_eq_1_event(),
        saveat=diffrax.SaveAt(t0=True, steps=True),
        max_steps=8192,   # ample for good solves; caps the cost of FAILED ones
        throw=False,      # failed solve -> inf/nan output, not a crash (sampler-safe)
    )
    a_st, _, da_st, _ = sol_ev.ys
    mask    = jnp.isfinite(sol_ev.ts) & jnp.isfinite(a_st) & (a_st > 0)
    idx     = jnp.maximum(jnp.sum(mask) - 1, 0)
    taustar = sol_ev.ts[idx]
    astar   = a_st[idx]                        # = 1 when the event fired
    dastar  = da_st[idx]                       # H(a=1) in code units
    x0        = jnp.log(a_st[0])               # = ln a_ini (t0 is saved)
    xstar_raw = jnp.log(astar)                  # ~ 0

    # failed solve, event never reached, or a pathological corner: substitute
    # a valid dummy grid and poison E (one bad particle must not crash a
    # vmapped batch — same guard as the parent module).
    ok = (jnp.isfinite(taustar) & jnp.isfinite(xstar_raw) & jnp.isfinite(x0)
          & (jnp.abs(astar - 1.0) < 1e-3)                # event really fired
          & (dastar > 0)
          & (xstar_raw > x0 + jnp.log(1.0 + Z_SPLIT)))
    xstar = jnp.where(ok, xstar_raw, x0 + 45.0)

    # log-spaced a-grid ending exactly today: sparse before z = Z_SPLIT,
    # dense after (that's where the BAO data are)
    xsplit = xstar - jnp.log(1.0 + Z_SPLIT)
    xs = jnp.concatenate([
        jnp.linspace(x0, xsplit, N_EARLY, endpoint=False),
        jnp.linspace(xsplit, xstar, N_LATE),
    ])
    lna_st = jnp.where(mask, jnp.log(jnp.where(mask, a_st, 1.0)), xstar_raw)
    tau_st = jnp.where(mask, sol_ev.ts, taustar)
    tau_targets = jnp.interp(xs, lna_st, tau_st)
    dummy = tau0 * (1.0 + jnp.linspace(0.0, 1.0, N_EARLY + N_LATE))
    tau_targets = jnp.where(ok, tau_targets, dummy)
    tau_end     = jnp.where(ok, taustar, 2.0 * tau0)

    # --- solve 2: same trajectory, saved on the a-grid -----------------------
    sol = diffrax.diffeqsolve(
        _TERM_TAU, _SOLVER, t0=tau0, t1=tau_end, dt0=0.01 * tau0, y0=y0,
        args=args, stepsize_controller=_CTRL,
        saveat=diffrax.SaveAt(ts=tau_targets),
        max_steps=8192,
        throw=False,
    )
    a_ev, cp_ev, da_ev, dcp_ev = sol.ys

    a_norm = a_ev / astar                        # = 1 today (a = 1)
    z      = 1.0 / a_norm - 1.0
    # E = H/H(a=1) with H = da/a^2 in a0 = 1 code units — no "100" anywhere
    E      = (da_ev / dastar) * (astar / a_ev) ** 2
    E      = jnp.where(ok, E, jnp.nan)   # failed solve -> nan -> logL floor

    return z[::-1], E[::-1], a_norm
