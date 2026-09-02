"""Is the TorC background system actually stiff?

Stiffness is not "hard to solve" -- it is a specific thing: the Jacobian of
the RHS has eigenvalues spanning a huge range of magnitudes, so the fastest
decaying mode forces an explicit solver to take tiny steps for STABILITY
long after that mode has died and stopped mattering for ACCURACY.

The standard diagnostic is the stiffness ratio

    S = max|Re(lambda)| / min|Re(lambda)|

over the eigenvalues of the Jacobian dF/dy along the trajectory.  Rules of
thumb: S < 1e3 is non-stiff, S > 1e5-1e6 means an implicit solver should
start to pay off.

Run:  python benchmarks/measure_stiffness.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "backends"))
sys.path.insert(0, os.path.join(HERE, ".."))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

from common import setup
from torc.theory import _vector_field_lna, _TERM_LNA


def main():
    x0, y0, args, xs = setup()

    # solve once to get the trajectory
    sol = diffrax.diffeqsolve(
        _TERM_LNA, diffrax.Dopri8(), t0=x0, t1=0.0, dt0=None, y0=y0, args=args,
        stepsize_controller=diffrax.PIDController(rtol=1e-10, atol=1e-10),
        saveat=diffrax.SaveAt(ts=jnp.asarray(xs)), max_steps=1_000_000,
        throw=False, adjoint=diffrax.ForwardMode())
    cp, da, dcp = [np.asarray(v) for v in sol.ys]

    # Jacobian of the vector field w.r.t. the state, along the trajectory
    def field(x, y):
        return jnp.stack(_vector_field_lna(x, y, args))

    jac = jax.jit(jax.jacobian(field, argnums=1))

    ratios, spectra = [], []
    for i in range(0, len(xs), 5):
        y = (jnp.asarray(cp[i]), jnp.asarray(da[i]), jnp.asarray(dcp[i]))
        J = np.asarray(jac(jnp.asarray(xs[i]), y))
        J = np.squeeze(J)
        if J.shape != (3, 3) or not np.all(np.isfinite(J)):
            continue
        ev = np.linalg.eigvals(J)
        re = np.abs(ev.real)
        re = re[re > 0]
        if len(re) < 2:
            continue
        ratios.append(re.max() / re.min())
        spectra.append((float(xs[i]), re.max(), re.min()))

    ratios = np.array(ratios)
    print(f"Jacobian sampled at {len(ratios)} points along the trajectory")
    print(f"(state y = (cp, da, dcp); integration variable x = ln a)\n")
    print(f"  stiffness ratio S = max|Re L| / min|Re L|")
    print(f"    median : {np.median(ratios):12.2f}")
    print(f"    mean   : {ratios.mean():12.2f}")
    print(f"    MAX    : {ratios.max():12.2f}   <-- the number that matters")
    print()
    i = int(np.argmax(ratios))
    x, hi, lo = spectra[i]
    print(f"  worst point: x = ln a = {x:.3f}  (a = {np.exp(x):.3e})")
    print(f"    fastest |Re lambda| = {hi:.4g}")
    print(f"    slowest |Re lambda| = {lo:.4g}")
    print()
    S = ratios.max()
    if S < 1e3:
        verdict = "NOT STIFF -- explicit solvers are the right choice"
    elif S < 1e5:
        verdict = "mildly stiff -- explicit still likely wins"
    else:
        verdict = "STIFF -- an implicit solver should pay off"
    print(f"  VERDICT (S = {S:.1f}): {verdict}")
    print()
    print("  For reference, classic stiff test problems:")
    print("    Van der Pol (mu=1000)  S ~ 1e6")
    print("    ROBER (Robertson)      S ~ 1e11")
    print("    combustion / chemistry S ~ 1e8-1e13")


if __name__ == "__main__":
    main()
