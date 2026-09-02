"""Does max_steps explain the ~20x figure quoted in priors.py?

Committed value is max_steps=8192; the working tree lowered it to 1024.
A failed solve burns the FULL budget, so the batch penalty from one toxic
lane should scale roughly with max_steps.
"""
import sys, os, time, functools
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import diffrax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                          _TERM_LNA, _SOLVER, _CTRL, Z_SPLIT, N_EARLY, N_LATE,
                          OMEGA_R_DEFAULT)
from torc.likelihood import (ZS, DATA, COV_INV, LOGL_NORM, IS_DM, IS_DH,
                              IS_DV, Z_GRID, C_KMS, _cumtrapz)


def make_logl(max_steps):
    def rhopsol(omega_m, omegar, OmL, stigma, mups2):
        tauini = small(omega_m, omegar, stigma, mups2)
        x0_raw = jnp.log(aseries(tauini, omega_m, omegar, OmL, stigma, mups2))
        ok = jnp.isfinite(x0_raw) & (x0_raw < -jnp.log(1.0 + Z_SPLIT))
        x0 = jnp.where(ok, x0_raw, -45.0)
        y0 = (cpseries(tauini, omega_m, omegar, OmL, stigma, mups2),
              aseriesp(tauini, omega_m, omegar, OmL, stigma, mups2),
              cpseriesp(tauini, omega_m, omegar, OmL, stigma, mups2))
        xsplit = -jnp.log(1.0 + Z_SPLIT)
        xs = jnp.concatenate([jnp.linspace(x0, xsplit, N_EARLY, endpoint=False),
                              jnp.linspace(xsplit, 0.0, N_LATE)])
        sol = diffrax.diffeqsolve(
            _TERM_LNA, _SOLVER, t0=x0, t1=0.0, dt0=None, y0=y0,
            args=(omega_m, omegar, OmL, stigma, mups2),
            stepsize_controller=_CTRL, saveat=diffrax.SaveAt(ts=xs),
            max_steps=max_steps, throw=False, adjoint=diffrax.ForwardMode())
        cp, da, dcp = sol.ys
        a_grid = jnp.exp(xs)
        z = 1.0 / a_grid - 1.0
        E = da * jnp.exp(-2.0 * xs) / da[-1]
        E = jnp.where(ok, E, jnp.nan)
        return z[::-1], E[::-1], a_grid

    def logl(p):
        z_th, E_th, _ = rhopsol(p["omega_m"], OMEGA_R_DEFAULT, p["OmL"],
                                 p["stigma"], p["mups2"])
        E_grid = jnp.interp(Z_GRID, z_th, E_th)
        dm_grid = (C_KMS / p["h0rd"]) * _cumtrapz(1.0 / E_grid, Z_GRID)
        dm = jnp.interp(ZS, Z_GRID, dm_grid)
        dh = (C_KMS / p["h0rd"]) / jnp.interp(ZS, z_th, E_th)
        dv = (ZS * dm ** 2 * dh) ** (1.0 / 3.0)
        r = DATA - (IS_DM * dm + IS_DH * dh + IS_DV * dv)
        out = -0.5 * (r @ COV_INV @ r) + LOGL_NORM
        return jnp.where(jnp.isfinite(out), out, -1e30)
    return logl


N = 50
def mk(n_toxic):
    st = np.full(N, 1.5); st[:n_toxic] = 0.8      # toxic band 0.6-0.99
    return dict(omega_m=jnp.full(N, 0.3), h0rd=jnp.full(N, 10000.),
                stigma=jnp.asarray(st), mups2=jnp.full(N, 1.5),
                OmL=jnp.full(N, 0.7))

print(f"{'max_steps':>10s} {'clean batch':>13s} {'1 toxic lane':>14s} {'penalty':>9s}")
res = {}
for ms in (1024, 8192):
    f = jax.jit(jax.vmap(make_logl(ms)))
    def t(p, reps=12):
        jax.block_until_ready(f(p)); ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); jax.block_until_ready(f(p))
            ts.append(time.perf_counter() - t0)
        return np.median(ts) * 1e3
    a, b = t(mk(0)), t(mk(1))
    res[ms] = (a, b, b / a)
    print(f"{ms:10d} {a:11.2f}ms {b:12.2f}ms {b/a:8.1f}x", flush=True)

print(f"\ncommitted max_steps=8192 gives a {res[8192][2]:.0f}x batch penalty")
print(f"working-tree max_steps=1024 gives {res[1024][2]:.0f}x")
print("priors.py documents '~20x' -- consistent with the 8192 era it was written in.")
print("DONE")
