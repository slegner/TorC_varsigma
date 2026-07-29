"""The benchmark ranks solvers on a single solve. What matters is the
vmapped batch cost inside the sampler. Test Tsit5 vs Dopri5 vs Dopri8 there.
"""
import sys, os, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import diffrax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                          _TERM_LNA, Z_SPLIT, N_EARLY, N_LATE, OMEGA_R_DEFAULT)
from torc.likelihood import (ZS, DATA, COV_INV, LOGL_NORM, IS_DM, IS_DH,
                              IS_DV, Z_GRID, C_KMS, _cumtrapz)
from torc.priors import PRIOR_BOUNDS, _SAFE, sample_prior

XSPLIT = -np.log(1.0 + Z_SPLIT)


def make_logl(solver, tol):
    ctrl = diffrax.PIDController(rtol=tol, atol=tol)
    def rhopsol(omega_m, omegar, OmL, stigma, mups2):
        tau = small(omega_m, omegar, stigma, mups2)
        x0_raw = jnp.log(aseries(tau, omega_m, omegar, OmL, stigma, mups2))
        ok = jnp.isfinite(x0_raw) & (x0_raw < XSPLIT)
        x0 = jnp.where(ok, x0_raw, -45.0)
        y0 = (cpseries(tau, omega_m, omegar, OmL, stigma, mups2),
              aseriesp(tau, omega_m, omegar, OmL, stigma, mups2),
              cpseriesp(tau, omega_m, omegar, OmL, stigma, mups2))
        xs = jnp.concatenate([jnp.linspace(x0, XSPLIT, N_EARLY, endpoint=False),
                              jnp.linspace(XSPLIT, 0.0, N_LATE)])
        sol = diffrax.diffeqsolve(
            _TERM_LNA, solver, t0=x0, t1=0.0, dt0=None, y0=y0,
            args=(omega_m, omegar, OmL, stigma, mups2),
            stepsize_controller=ctrl, saveat=diffrax.SaveAt(ts=xs),
            max_steps=1024, throw=False, adjoint=diffrax.ForwardMode())
        cp, da, dcp = sol.ys
        z = 1.0 / jnp.exp(xs) - 1.0
        E = da * jnp.exp(-2.0 * xs) / da[-1]
        E = jnp.where(ok, E, jnp.nan)
        return z[::-1], E[::-1], sol.stats["num_steps"]

    def logl(p):
        z_th, E_th, ns = rhopsol(p["omega_m"], OMEGA_R_DEFAULT, p["OmL"],
                                  p["stigma"], p["mups2"])
        E_grid = jnp.interp(Z_GRID, z_th, E_th)
        dm_grid = (C_KMS / p["h0rd"]) * _cumtrapz(1.0 / E_grid, Z_GRID)
        dm = jnp.interp(ZS, Z_GRID, dm_grid)
        dh = (C_KMS / p["h0rd"]) / jnp.interp(ZS, z_th, E_th)
        dv = (ZS * dm ** 2 * dh) ** (1.0 / 3.0)
        r = DATA - (IS_DM * dm + IS_DH * dh + IS_DV * dv)
        out = -0.5 * (r @ COV_INV @ r) + LOGL_NORM
        return jnp.where(jnp.isfinite(out), out, -1e30), ns
    return logl


N = 250
pts = sample_prior(jax.random.PRNGKey(0), N)

SOLVERS = {"Tsit5 (current)": diffrax.Tsit5(), "Dopri5": diffrax.Dopri5(),
           "Dopri8": diffrax.Dopri8()}
print(f"{'solver':18s} {'tol':>7s} {'batch(250)':>12s} {'ms/solve':>10s} "
      f"{'mean steps':>11s} {'max logL diff vs Tsit5@1e-5':>28s}")
ref = None
for tol in (1e-5, 1e-8):
    for name, sv in SOLVERS.items():
        f = jax.jit(jax.vmap(make_logl(sv, tol)))
        v, ns = jax.block_until_ready(f(pts))
        v = np.asarray(v); ns = np.asarray(ns)
        if ref is None:
            ref = v
        ts = []
        for _ in range(10):
            t0 = time.perf_counter(); jax.block_until_ready(f(pts))
            ts.append(time.perf_counter() - t0)
        ms = np.median(ts) * 1e3
        fin = np.isfinite(v) & np.isfinite(ref)
        d = np.abs(v[fin] - ref[fin]).max()
        print(f"{name:18s} {tol:7.0e} {ms:10.2f}ms {ms/N:9.4f}ms {ns.mean():11.1f} "
              f"{d:28.2e}")
print("\nDONE")
