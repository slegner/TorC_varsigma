"""Where does the pathological (max_steps-exhausting) solve actually live?

priors.py documents a ~20x slowdown for unclamped out-of-bounds proposals,
but stigma=0.4 solves normally. Scan out-of-bounds regions and find what
actually triggers it. num_steps is the robust metric (immune to CPU load).

Note the algebra: the dda/ddcp denominators contain (-1+4*stigma**2), which
vanishes at stigma = 0.5 exactly; and small()/aseries contain
(-1+4*stigma)**(1/2), which is imaginary for stigma < 0.25.
"""
import sys, os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import diffrax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                          _TERM_LNA, _SOLVER, _CTRL, Z_SPLIT, N_EARLY, N_LATE,
                          OMEGA_R_DEFAULT)

def solve_stats(omega_m, stigma, mups2, OmL, omegar=OMEGA_R_DEFAULT):
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
        max_steps=1024, throw=False, adjoint=diffrax.ForwardMode())
    return sol.stats["num_steps"], sol.result == diffrax.RESULTS.successful, ok

F = jax.jit(jax.vmap(solve_stats))

print("=== scan over stigma (others at fiducial 0.3/1.5/0.7) ===")
print("prior box for stigma is [1.0001, 3.0]; below that is out-of-bounds")
sts = np.array([0.05, 0.1, 0.2, 0.24, 0.2501, 0.3, 0.4, 0.45, 0.49, 0.499,
                0.5, 0.501, 0.51, 0.55, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0001,
                1.5, 3.0])
n = len(sts)
ns, okres, icok = F(jnp.full(n, 0.3), jnp.asarray(sts), jnp.full(n, 1.5), jnp.full(n, 0.7))
ns, okres, icok = np.asarray(ns), np.asarray(okres), np.asarray(icok)
for s, a, b, c in zip(sts, ns, okres, icok):
    flag = ""
    if not c: flag += " IC-nonfinite"
    if not b: flag += "  <-- SOLVER FAILED (hit max_steps)"
    elif a > 200: flag += "  <-- expensive"
    print(f"  stigma={s:<8.4f} num_steps={a:5d} ok={bool(b)}{flag}")

print("\n=== other out-of-bounds directions ===")
cases = {
    "omega_m=-0.5":      (-0.5, 1.5, 1.5, 0.7),
    "omega_m=0 (exact)": (0.0, 1.5, 1.5, 0.7),
    "omega_m=5":         (5.0, 1.5, 1.5, 0.7),
    "mups2=-1":          (0.3, 1.5, -1.0, 0.7),
    "mups2=0 (exact)":   (0.3, 1.5, 0.0, 0.7),
    "OmL=-3":            (0.3, 1.5, 1.5, -3.0),
    "OmL=50":            (0.3, 1.5, 1.5, 50.0),
    "stigma=1 (exact)":  (0.3, 1.0, 1.5, 0.7),
    "all extreme":       (-1.0, 0.5, -2.0, 100.0),
}
k = list(cases)
arr = np.array([cases[x] for x in k])
ns, okres, icok = F(jnp.asarray(arr[:,0]), jnp.asarray(arr[:,1]),
                     jnp.asarray(arr[:,2]), jnp.asarray(arr[:,3]))
ns, okres, icok = np.asarray(ns), np.asarray(okres), np.asarray(icok)
for name, a, b, c in zip(k, ns, okres, icok):
    flag = ""
    if not c: flag += " IC-nonfinite"
    if not b: flag += "  <-- SOLVER FAILED (hit max_steps)"
    elif a > 200: flag += "  <-- expensive"
    print(f"  {name:20s} num_steps={a:5d} ok={bool(b)}{flag}")

print(f"\nfiducial reference: num_steps={int(np.asarray(F(jnp.array([0.3]),jnp.array([1.5]),jnp.array([1.5]),jnp.array([0.7]))[0])[0])}")
print("DONE")
