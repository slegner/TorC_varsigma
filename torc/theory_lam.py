"""
theory_lam.py — the TorC background with the Lambda (lam) term switched on.
==========================================================================

Same theory, same anchoring, same initial conditions as torc/theory.py; the
ONLY difference is that dda/ddcp carry the extra `lam` source terms coming
from Notebooktrt2lam.nb.  At lam = 0 these reduce to torc.theory exactly
(verified numerically by tests/test_theory_lam.py).

The Lambda term enters the a**4 source block alongside OmL, with a different
stigma structure and always multiplying cp**2 or cp**4 — i.e. it couples
through the field sector rather than as a pure a**4 constant:

    ... - 40000*a**4*(-cp**2 + stigma)**2*(
              3*OmL*(1 - 4*stigma)*stigma**2
            + cp**4*lam*(-1 + 4*stigma**2)
            + cp**2*stigma*(-7*lam - 3*OmL + 4*lam*stigma + 12*OmL*stigma**2))

Setting lam = 0 collapses that bracket to 3*OmL*(...), which is TorC's
120000*a**4*OmL*stigma*(...) term.

INITIAL CONDITIONS are inherited unchanged from torc.theory.  That is exact
to double precision, not an approximation: every lam term carries a**4, and
a(tau_ini) ~ 1e-9, so the lam contribution to the series is ~1e-36 of the
leading term — below the floor of float64.  Both models therefore start from
literally the same numbers.

CONVENTION: identical to torc/theory.py — a0 = 1 pinned, today is a = 1,
E(z) = H/H(a=1), omegar an input, h absent, mups2 an exact gauge parameter.
Expressions were converted from Stigma/lambda/equations_lambda.py
mechanically (h**2*omega_r(h) -> omegar, h**2 -> 1), not retyped.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

from .theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                     omega_r, MP2, N_EARLY, N_LATE, Z_SPLIT,
                     OMEGA_R_DEFAULT, _SOLVER, _CTRL)

ARGS_DOC = "a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2, lam"


def dda(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2, lam):
    """a'' with the Lambda term.  lam = 0 reproduces torc.theory.dda."""
    val = (a**2*dcp**2*mups2*(-3*cp**2*stigma**2 - cp**4*(1 - 2*stigma)**2*stigma*(-3 + \
            2*stigma)*(1 + 2*stigma) + stigma**3*(1 + 8*stigma) - cp**6*(1 - \
            4*stigma**2)**2) + 40000*omegar*stigma*(-cp**2 + stigma)**2*(stigma + \
            cp**2*(-1 + 4*stigma**2)) - 2*a*cp*da*dcp*mups2*(stigma + cp**2*(-1 + \
            4*stigma**2))*((7 - 4*stigma)*stigma**2 + 2*cp**2*stigma*(1 + 2*(-3 + \
            stigma)*stigma) + cp**4*(-1 + 4*stigma**2)) - 40000*a**4*(-cp**2 + \
            stigma)**2*(3*OmL*stigma**2 + cp**4*lam*(-1 + 4*stigma**2) + \
            cp**2*stigma*(lam - 3*OmL + 8*lam*stigma + 12*OmL*stigma**2)) - \
            cp**2*da**2*mups2*(cp**4*(1 - 2*stigma)**2*stigma*(-7 + 2*stigma)*(1 + \
            2*stigma) + stigma**3*(3 + 32*(-1 + stigma)*stigma) + cp**6*(1 - \
            4*stigma**2)**2 + cp**2*stigma**2*(3 - 16*(-1 + stigma)*stigma*(-1 + \
            2*stigma))))/(8*a*cp**2*mups2*stigma*(-cp**2 + stigma)**2*(1 - 4*stigma**2))
    return val


def ddcp(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2, lam):
    """varpi'' with the Lambda term.  lam = 0 reproduces torc.theory.ddcp."""
    val = (40000*omegar*stigma*(-cp**2 + stigma)**2*(stigma - 4*stigma**2 + cp**2*(-1 + \
            4*stigma**2)) - cp**2*da**2*mups2*((5 - 8*stigma)*stigma + cp**2*(-1 + \
            4*stigma**2))*((7 - 4*stigma)*stigma**2 + 2*cp**2*stigma*(1 + 2*(-3 + \
            stigma)*stigma) + cp**4*(-1 + 4*stigma**2)) - 40000*a**4*(-cp**2 + \
            stigma)**2*(3*OmL*(1 - 4*stigma)*stigma**2 + cp**4*lam*(-1 + 4*stigma**2) + \
            cp**2*stigma*(-7*lam - 3*OmL + 4*lam*stigma + 12*OmL*stigma**2)) + \
            a**2*dcp**2*mups2*(stigma**3 + 4*stigma**4 - 32*stigma**5 - cp**4*stigma*(-1 \
            + 2*stigma)*(1 + 2*stigma)*(3 + 4*(-3 + stigma)*stigma) - cp**6*(1 - \
            4*stigma**2)**2 + cp**2*stigma**2*(-3 + 8*stigma + 16*stigma**3)) - \
            2*a*cp*da*dcp*mups2*(cp**4*stigma*(-1 + 2*stigma)*(1 + 2*stigma)*(11 + 4*(-4 \
            + stigma)*stigma) + cp**6*(1 - 4*stigma**2)**2 + stigma**3*(-1 + \
            16*stigma*(-2 + 3*stigma)) + cp**2*stigma**2*(11 - 16*stigma*(1 + stigma*(-1 \
            + 2*stigma)))))/(8*a**2*cp*mups2*stigma*(-cp**2 + stigma)**2*(-1 + \
            4*stigma**2))
    return val


def _vector_field_lna(x, y, args):
    cp, da, dcp = y
    omega_m, omegar, OmL, stigma, mups2, lam = args
    a = jnp.exp(x)
    fac = a / da                              # dtau/dx
    return (dcp * fac,
            dda(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2, lam) * fac,
            ddcp(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2, lam) * fac)


_TERM_LNA = diffrax.ODETerm(_vector_field_lna)


def rhopsol_jax(omega_m, omegar, OmL, stigma, mups2, lam):
    """Solve the Lambda theory; return (z, E, a).  Mirrors torc.theory.rhopsol_jax."""
    tauini = small(omega_m, omegar, stigma, mups2)
    x0_raw = jnp.log(aseries(tauini, omega_m, omegar, OmL, stigma, mups2))
    ok = jnp.isfinite(x0_raw) & (x0_raw < -jnp.log(1.0 + Z_SPLIT))
    x0 = jnp.where(ok, x0_raw, -45.0)
    y0 = (cpseries( tauini, omega_m, omegar, OmL, stigma, mups2),
          aseriesp( tauini, omega_m, omegar, OmL, stigma, mups2),
          cpseriesp(tauini, omega_m, omegar, OmL, stigma, mups2))
    args = (omega_m, omegar, OmL, stigma, mups2, lam)

    xsplit = -jnp.log(1.0 + Z_SPLIT)
    xs = jnp.concatenate([
        jnp.linspace(x0, xsplit, N_EARLY, endpoint=False),
        jnp.linspace(xsplit, 0.0, N_LATE),
    ])
    sol = diffrax.diffeqsolve(
        _TERM_LNA, _SOLVER, t0=x0, t1=0.0, dt0=None, y0=y0, args=args,
        stepsize_controller=_CTRL, saveat=diffrax.SaveAt(ts=xs),
        max_steps=1024, throw=False,
        adjoint=diffrax.ForwardMode(),
    )
    cp, da, dcp = sol.ys

    a_grid = jnp.exp(xs)
    z      = 1.0 / a_grid - 1.0
    E      = da * jnp.exp(-2.0 * xs) / da[-1]
    E      = jnp.where(ok, E, jnp.nan)

    return z[::-1], E[::-1], a_grid
