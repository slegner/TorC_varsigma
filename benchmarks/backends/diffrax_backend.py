"""diffrax (JAX) backend.  This is the one the real pipeline uses.

The vector field is imported straight from torc/theory.py -- not
reimplemented -- so this measures exactly what rhopsol_jax does.

Standalone:  python benchmarks/backends/diffrax_backend.py
"""
import jax
import jax.numpy as jnp
import diffrax

from common import setup, norm_E, timeit, report
from torc.theory import _TERM_LNA

SOLVERS = {
    "Tsit5":            diffrax.Tsit5(),
    "Dopri5":           diffrax.Dopri5(),
    "Dopri8":           diffrax.Dopri8(),
    "Bosh3":            diffrax.Bosh3(),
    "Heun":             diffrax.Heun(),
    "Kvaerno3 [stiff]": diffrax.Kvaerno3(),
    "Kvaerno5 [stiff]": diffrax.Kvaerno5(),
}


def solve(solver="Dopri8", tol=1e-8, params=None):
    x0, y0, args, xs = setup(params)
    ctrl = diffrax.PIDController(rtol=tol, atol=tol)
    xs_j = jnp.asarray(xs)

    def run():
        return diffrax.diffeqsolve(
            _TERM_LNA, SOLVERS[solver], t0=x0, t1=0.0, dt0=None, y0=y0,
            args=args, stepsize_controller=ctrl,
            saveat=diffrax.SaveAt(ts=xs_j), max_steps=1_000_000,
            throw=False, adjoint=diffrax.ForwardMode())

    f = jax.jit(run)
    try:
        sol = jax.block_until_ready(f())
    except Exception as e:
        return dict(backend="diffrax", solver=solver, tol=tol, ms=None,
                    ok=False, E=None, steps=None, note=type(e).__name__)
    ok = bool(sol.result == diffrax.RESULTS.successful)
    if not ok:
        return dict(backend="diffrax", solver=solver, tol=tol, ms=None,
                    ok=False, E=None, steps=None, note="max_steps")
    ms = timeit(lambda: jax.block_until_ready(f()))
    return dict(backend="diffrax", solver=solver, tol=tol, ms=ms, ok=True,
                E=norm_E(sol.ys[1], xs), steps=int(sol.stats["num_steps"]),
                note="")


def run_all(tols=(1e-5, 1e-8), params=None):
    return [solve(s, t, params) for t in tols for s in SOLVERS]


if __name__ == "__main__":
    report("diffrax (JAX)", run_all())
