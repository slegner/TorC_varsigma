"""
theory_jax.py — the updated TorC theory, self-contained, in JAX.
================================================================

Everything in one file: the theory equations + the diffrax solver.
This is the module the nested-sampling pipeline imports.

Interface (matches the conventions of the distances/ folder):

    z, E, a = rhopsol_jax(omega_m, omegar, OmL, stigma, mups2)

with omegar a plain input number (the distances/ scripts use 8.24e-5),
E(z) = H(z)/H0 normalised so E(0) = 1 exactly.

ANCHORING CONVENTION (noH0): a0 = 1 is pinned in the equations and "today"
is DEFINED as the moment the scale factor reaches a = 1; E is normalised
by whatever the expansion rate is there.  No reference to H0 anywhere.
NOTE: with this convention omega_m / omegar / OmL are the density fractions
at the epoch where H_code = 100 (never visited), NOT today's fractions —
fitted values are not directly comparable to e.g. DESI's Omega_m.
The previous convention (today located dynamically as the epoch where the
input fractions hold, via a da = 100 event) is archived in
legacy/H0_renorm/theory_jax.py.

Unit-fixing parameters eliminated by exact scaling symmetries (each
verified numerically):

  a0 (scale of a):        solve with a0 = 1  (and here a0 = 1 also anchors
                          today, per the noH0 convention).
  h  (scale of tau):      equations evaluated at h = 1 with omegar an input;
                          E = H/H0 carries no h.
  mups2 (field's clock):  rescales tau -> sqrt(mups2)*tau only; the
                          trajectory in a-space and E(z) are exactly
                          independent of it.  It is kept as an argument for
                          interface stability (and for future perturbation-
                          level observables) but does nothing here.
                          WHY: mups2 multiplies every term quadratic in
                          tau-derivatives in dda/ddcp but none of the
                          matter/radiation/Lambda sources, so it is absorbed
                          by tau = sqrt(mups2)*sigma; the series ICs scale
                          the same way (tau_ini ~ sqrt(mups2), term n ~
                          tau^n mups2^(-n/2)).  H then carries one common
                          factor mups2^(-1/2) at every epoch, which cancels
                          in the ratio E = H/H(a=1).  Derived and checked
                          step by step in scripts/explain_mups2_gauge.py
                          (-> plots/mups2_gauge.pdf).

Solver: x = ln(a) is the integration variable.  Because today is the FIXED
endpoint x = 0, the whole solve is ONE diffrax call on a static save grid —
no event, no calibration pass.  State y = (cp, da, dcp).
"""

import jax
jax.config.update("jax_enable_x64", True)   # must precede any jax call
import jax.numpy as jnp
import diffrax
from scipy import constants

MP2 = 1
# output grid: dense where the BAO data live (z < Z_SPLIT), sparse earlier.
# sizes are fixed => jit compiles once
N_EARLY = 200            # points covering a_ini .. a(Z_SPLIT)
N_LATE  = 300            # points covering z < Z_SPLIT
Z_SPLIT = 2.7

# radiation density used across the distances/ folder scripts
OMEGA_R_DEFAULT = 8.24e-5


def omega_r(h):
    """Physical radiation density parameter (photons + neutrinos) for a
    given h — kept only as a convenience for choosing a value of omegar
    (e.g. omega_r(0.7) = 8.47e-5). Not used by the equations below."""
    ab = (8*constants.pi**5*constants.k**4)/(15*constants.h**3*constants.c**3)
    rhor = ab*2.7255**4/(constants.c**2)
    rhov = 3.046*(7/8)*(4/11)**(4/3)*rhor
    return((rhor+rhov)/((3*(100*h)**2/(8*constants.pi*constants.G))*(1000)**2/(10**6*constants.parsec)**2))


# ---------------------------------------------------------------------------
# Early-time series solutions (initial conditions), a0- and h-free
# ---------------------------------------------------------------------------

def aseries(tau, omega_m, omegar, OmL, stigma, mups2):
    val = (
          (200*(omegar)**(1/2)*(stigma)**(1/2)*tau)/(mups2*(-1 + \
            4*stigma))**(1/2) + (2500*tau**2*(omega_m))/(mups2) + \
            (15625*(stigma)**(1/2)*(-1 + 4*stigma)**(1/2)*(-2 + stigma*(5 + \
            24*stigma))*tau**3*(omega_m)**2)/(mups2**(3/2)*(omegar)**(1/2)*(-1 + \
            stigma)*(-3 + stigma + 8*stigma**2)) + (390625*(-1 + \
            4*stigma)*(-6 + stigma*(31 + stigma*(357 + 2*stigma*(-123 + \
            5*stigma*(-151 + \
            24*stigma)))))*tau**4*(omega_m)**3)/(2*mups2**2*omegar*(-1 + \
            stigma)**2*(-3 + stigma + 8*stigma**2)*(-6 + stigma + 20*stigma**2))
    )
    return val


def cpseries(tau, omega_m, omegar, OmL, stigma, mups2):
    val = (
       1 - (25*(-1 + \
            4*stigma)**(1/2)*tau*(omega_m))/(2*(mups2)**(1/2)*(omegar)**(1/2)*(stigma)**(1/2)) + (1875*(-1 + \
            4*stigma)*(2 + stigma*(-2 + \
            3*stigma))*tau**2*(omega_m)**2)/(8*mups2*omegar*(-1 + stigma)*stigma*(-3 \
            + stigma + 8*stigma**2)) + (15625*(-1 + 4*stigma)**(3/2)*(-36 + \
            stigma*(72 + stigma*(175 + stigma*(-225 + 4*stigma*(-180 + stigma*(95 \
            + 48*stigma))))))*tau**3*(omega_m)**3)/(16*mups2**(3/2)*omegar**(3/2)*(-1 + stigma)**2*stigma**(3/2)*(-3 \
            + stigma + 8*stigma**2)*(-6 + stigma + 20*stigma**2))
    )
    return val


def aseriesp(tau, omega_m, omegar, OmL, stigma, mups2):
    val = (
          (200*(omegar)**(1/2)*(stigma)**(1/2))/(mups2*(-1 + \
            4*stigma))**(1/2) + (5000*tau*(omega_m))/(mups2) + \
            (46875*(stigma)**(1/2)*(-1 + 4*stigma)**(1/2)*(-2 + stigma*(5 + \
            24*stigma))*tau**2*(omega_m)**2)/(mups2**(3/2)*(omegar)**(1/2)*(-1 + \
            stigma)*(-3 + stigma + 8*stigma**2)) + (781250*(-1 + \
            4*stigma)*(-6 + stigma*(31 + stigma*(357 + 2*stigma*(-123 + \
            5*stigma*(-151 + \
            24*stigma)))))*tau**3*(omega_m)**3)/(mups2**2*omegar*(-1 + \
            stigma)**2*(-3 + stigma + 8*stigma**2)*(-6 + stigma + 20*stigma**2))
    )
    return val


def cpseriesp(tau, omega_m, omegar, OmL, stigma, mups2):
    # d/dtau of cpseries above: -A + 2*B*tau + 3*C*tau**2
    val = (
       - (25*(-1 + \
            4*stigma)**(1/2)*(omega_m))/(2*(mups2)**(1/2)*(omegar)**(1/2)*(stigma)**(1/2)) + (1875*(-1 + \
            4*stigma)*(2 + stigma*(-2 + \
            3*stigma))*tau*(omega_m)**2)/(4*mups2*omegar*(-1 + stigma)*stigma*(-3 \
            + stigma + 8*stigma**2)) + (46875*(-1 + 4*stigma)**(3/2)*(-36 + \
            stigma*(72 + stigma*(175 + stigma*(-225 + 4*stigma*(-180 + stigma*(95 \
            + 48*stigma))))))*tau**2*(omega_m)**3)/(16*mups2**(3/2)*omegar**(3/2)*(-1 + stigma)**2*stigma**(3/2)*(-3 \
            + stigma + 8*stigma**2)*(-6 + stigma + 20*stigma**2))
    )
    return val


# ---------------------------------------------------------------------------
# The theory's ODEs (second order, conformal time), a0- and h-free
# ---------------------------------------------------------------------------

def dda(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2):
    val = (-1/8*(-(a**2*dcp**2*mups2*(3*cp**2*stigma**2 + cp**4*(1 - \
            2*stigma)**2*stigma*(-3 + 2*stigma)*(1 + 2*stigma) - stigma**3*(1 + \
            8*stigma) + cp**6*(1 - 4*stigma**2)**2)) - \
            120000*a**4*OmL*stigma*(-cp**2 + stigma)**2*(stigma + \
            cp**2*(-1 + 4*stigma**2)) + 40000*omegar*stigma*(-cp**2 + \
            stigma)**2*(stigma + cp**2*(-1 + 4*stigma**2)) - \
            2*a*cp*da*dcp*mups2*(stigma + cp**2*(-1 + 4*stigma**2))*((7 - \
            4*stigma)*stigma**2 + 2*cp**2*stigma*(1 + 2*(-3 + stigma)*stigma) + \
            cp**4*(-1 + 4*stigma**2)) - cp**2*da**2*mups2*(cp**4*(1 - \
            2*stigma)**2*stigma*(-7 + 2*stigma)*(1 + 2*stigma) + stigma**3*(3 + \
            32*(-1 + stigma)*stigma) + cp**6*(1 - 4*stigma**2)**2 + \
            cp**2*stigma**2*(3 - 16*(-1 + stigma)*stigma*(-1 + \
            2*stigma))))/(a*cp**2*mups2*stigma*(-cp**2 + stigma)**2*(-1 + \
            4*stigma**2)))
    return val


def ddcp(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2):
    val = (-1/8*(120000*a**4*OmL*stigma*(-cp**2 + \
            stigma)**2*(stigma - 4*stigma**2 + cp**2*(-1 + 4*stigma**2)) - \
            40000*omegar*stigma*(-cp**2 + stigma)**2*(stigma - 4*stigma**2 + \
            cp**2*(-1 + 4*stigma**2)) + cp**2*da**2*mups2*((5 - \
            8*stigma)*stigma + cp**2*(-1 + 4*stigma**2))*((7 - \
            4*stigma)*stigma**2 + 2*cp**2*stigma*(1 + 2*(-3 + stigma)*stigma) + \
            cp**4*(-1 + 4*stigma**2)) + a**2*dcp**2*mups2*(stigma**3*(-1 + \
            4*stigma)*(1 + 8*stigma) + cp**4*stigma*(-1 + 2*stigma)*(1 + \
            2*stigma)*(3 + 4*(-3 + stigma)*stigma) + cp**6*(1 - 4*stigma**2)**2 + \
            cp**2*stigma**2*(3 - 8*(stigma + 2*stigma**3))) + \
            2*a*cp*da*dcp*mups2*(cp**4*stigma*(-1 + 2*stigma)*(1 + \
            2*stigma)*(11 + 4*(-4 + stigma)*stigma) + cp**6*(1 - 4*stigma**2)**2 + \
            stigma**3*(-1 + 16*stigma*(-2 + 3*stigma)) + cp**2*stigma**2*(11 - \
            16*stigma*(1 + stigma*(-1 + \
            2*stigma)))))/(a**2*cp*mups2*stigma*(-cp**2 + stigma)**2*(-1 + \
            4*stigma**2)))
    return val


def small(omega_m, omegar, stigma, mups2):
    """tau_ini: small enough that the leading series term is ~1e-6 of cp."""
    return 1e-6*(2*(mups2)**(1/2)*(omegar)**(1/2)*(stigma)**(1/2)) / \
           (25*(-1 + 4*stigma)**(1/2)*(omega_m))


# ---------------------------------------------------------------------------
# diffrax solver.  Primary: x = ln(a), state y = (cp, da, dcp) — one solve
# from x0 to 0.  The tau vector field is kept for noH0/theory_jax_noH0.py.
# ---------------------------------------------------------------------------

def _vector_field_lna(x, y, args):
    cp, da, dcp = y
    omega_m, omegar, OmL, stigma, mups2 = args
    a = jnp.exp(x)
    fac = a / da                              # dtau/dx
    return (dcp * fac,
            dda(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2) * fac,
            ddcp(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2) * fac)


def _vector_field_tau(t, y, args):
    a, cp, da, dcp = y
    omega_m, omegar, OmL, stigma, mups2 = args
    return (da, dcp,
            dda(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2),
            ddcp(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2))


_TERM_LNA = diffrax.ODETerm(_vector_field_lna)
_TERM_TAU = diffrax.ODETerm(_vector_field_tau)
_SOLVER   = diffrax.Dopri8()
_CTRL     = diffrax.PIDController(rtol=1e-8, atol=1e-8)
# Benchmarked against Tsit5/Dopri5/Bosh3/Heun/Ralston, the Kvaerno+KenCarp
# implicit family, and scipy RK45/RK23/DOP853/Radau/BDF/LSODA.  The system is
# NOT stiff: implicit solvers cost 76-480x more and fail outright at 1e-8, so
# an explicit high-order RK is right.  Dopri8 at 1e-8 needs ~72 steps/solve --
# fewer than Tsit5 needed at 1e-5 (~68) for ~1000x better accuracy, at equal
# or lower wall time.  Batch cost is set by the slowest vmap lane and by fixed
# per-solve overhead, not by mean step count, so the in-sampler gain is ~12%.

TAU_SPAN  = 10.0        # only used by the tau-based variant in noH0/


def rhopsol_jax(omega_m, omegar, OmL, stigma, mups2):
    """Solve the theory (noH0 anchoring, ln(a) variable); return (z, E, a).

    z  : redshift, ascending, z[0] = 0
    E  : H(z)/H(a=1), ascending in z; E[0] = 1 exactly
    a  : scale factor grid (ascending, ends at 1) = 1/(1+z) reversed

    Today is the fixed endpoint x = ln(a) = 0, so this is a single
    diffrax solve on a static save grid — no event, no calibration.
    Failed solves (nan blow-up, max_steps) propagate nan into E, which the
    likelihood floors at -1e30; the save grid never depends on the solution,
    so a bad point cannot crash a vmapped batch.
    """
    tauini = small(omega_m, omegar, stigma, mups2)
    x0_raw = jnp.log(aseries(tauini, omega_m, omegar, OmL, stigma, mups2))
    # keep the grid finite for garbage parameters (nan ICs still poison E)
    ok = jnp.isfinite(x0_raw) & (x0_raw < -jnp.log(1.0 + Z_SPLIT))
    x0 = jnp.where(ok, x0_raw, -45.0)
    y0 = (cpseries( tauini, omega_m, omegar, OmL, stigma, mups2),
          aseriesp( tauini, omega_m, omegar, OmL, stigma, mups2),
          cpseriesp(tauini, omega_m, omegar, OmL, stigma, mups2))
    args = (omega_m, omegar, OmL, stigma, mups2)

    xsplit = -jnp.log(1.0 + Z_SPLIT)
    xs = jnp.concatenate([
        jnp.linspace(x0, xsplit, N_EARLY, endpoint=False),
        jnp.linspace(xsplit, 0.0, N_LATE),
    ])
    sol = diffrax.diffeqsolve(
        _TERM_LNA, _SOLVER, t0=x0, t1=0.0, dt0=None, y0=y0, args=args,
        stepsize_controller=_CTRL, saveat=diffrax.SaveAt(ts=xs),
        max_steps=1024,   # Dopri8@1e-8 good solves need ~72-105 steps (measured);
                          # ~10x margin, and caps the cost of FAILED ones
        throw=False,      # failed solve -> inf/nan output, not a crash (sampler-safe)
        adjoint=diffrax.ForwardMode(),  # forward-mode autodiff (few params); as in recfast
    )
    cp, da, dcp = sol.ys

    a_grid = jnp.exp(xs)                         # ends exactly at 1
    z      = 1.0 / a_grid - 1.0
    # E = H/H(a=1) with H prop. da * e^{-2x}; no "100", no h, no H0 anywhere
    E      = da * jnp.exp(-2.0 * xs) / da[-1]
    E      = jnp.where(ok, E, jnp.nan)   # garbage params -> nan -> logL floor

    return z[::-1], E[::-1], a_grid
