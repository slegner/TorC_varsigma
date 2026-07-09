"""Validation of the jax theory against the scipy reference, plus the
mups2 gauge invariance.  Run:  python tests/test_theory.py   (or pytest)."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
jax.config.update("jax_enable_x64", True)
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from torc.theory import rhopsol_jax, omega_r
from torc.theory_tau import rhopsol_jax_tau
from torc.scipy_reference import (aseries, cpseries, aseriesp, cpseriesp,
                                  coupdiff, small)

H_CMP  = 0.7
OMEGAR = omega_r(H_CMP)
ZTEST  = np.linspace(0.0, 2.6, 200)          # the BAO data region

SETS = [(0.2969, 1.0001, 4/3, 0.703),
        (0.2969, 1.5,    4/3, 0.703),
        (0.2893, 1.1,    3.0, 0.9),
        (0.01,   1.0001, 3.0, 1.5),
        (0.99,   2.9999, 0.5, 0.1),
        (0.5,    2.0,    0.6, 1.2)]


def rhopsol_scipy_a0eq1(om, h, OmL, stigma, mups2):
    """scipy solve_ivp with the same anchoring (a0 = 1, today := a = 1)."""
    tauini = small(om, h, stigma, mups2)
    y0 = [aseries(tauini, om, h, OmL, 1, stigma, mups2),
          cpseries(tauini, om, h, OmL, 1, stigma, mups2),
          aseriesp(tauini, om, h, OmL, 1, stigma, mups2),
          cpseriesp(tauini, om, h, OmL, 1, stigma, mups2)]
    def astop(t, y): return y[0] - 1.0
    astop.terminal = True
    rhs = lambda t, y: coupdiff(y, t, om, h, OmL, 1, stigma, mups2)
    sol = solve_ivp(rhs, [tauini, np.inf], y0, events=astop,
                    rtol=1e-8, atol=1e-8)
    tsol  = interp1d(sol.y[0], sol.t, bounds_error=False)
    avals = np.logspace(np.log10(sol.y[0][0]), 0, 5000)
    tvals = tsol(avals); tvals[0] = tauini
    sol2 = solve_ivp(rhs, [tauini, np.inf], y0, t_eval=tvals,
                     events=astop, rtol=1e-8, atol=1e-8)
    a, cp, da, dcp = sol2.y
    aend, _, daend, _ = sol2.y_events[0][0]
    return 1/a - 1, (da/a**2) / (daend/aend**2)


def _E(fn, om, sg, mp, ol):
    z, E, _ = fn(om, OMEGAR, ol, sg, mp)
    return np.interp(ZTEST, np.asarray(z), np.asarray(E))


def test_normalisation():
    _, E, _ = rhopsol_jax(0.2969, OMEGAR, 0.7, 1.2, 1.1)
    assert float(E[0]) == 1.0


def test_vs_scipy():
    for om, sg, mp, ol in SETS:
        Ej = _E(rhopsol_jax, om, sg, mp, ol)
        zs, Es = rhopsol_scipy_a0eq1(om, H_CMP, ol, sg, mp)
        Es_i = np.interp(ZTEST, zs[::-1], Es[::-1])
        d = np.max(np.abs(Ej/Es_i - 1))
        # stiff stigma -> 1.0001 corners sit at solver-tolerance level
        tol = 5e-4 if sg < 1.01 else 1e-5
        assert d < tol, f"({om},{sg},{mp},{ol}): {d:.2e} >= {tol}"
        print(f"  vs scipy ({om},{sg},{mp},{ol}): {d:.2e}")


def test_tau_vs_lna():
    for om, sg, mp, ol in SETS[:4]:
        d = np.max(np.abs(_E(rhopsol_jax, om, sg, mp, ol) /
                          _E(rhopsol_jax_tau, om, sg, mp, ol) - 1))
        assert d < 5e-4
        print(f"  tau vs lna ({om},{sg},{mp},{ol}): {d:.2e}")


def test_mups2_gauge():
    """mups2 rescales the field's clock only: E(z) must not depend on it."""
    for mu1, mu2 in [(0.5, 3.0), (1.0, 2.0)]:
        d = np.max(np.abs(_E(rhopsol_jax, 0.2969, 1.4, mu1, 0.9) /
                          _E(rhopsol_jax, 0.2969, 1.4, mu2, 0.9) - 1))
        assert d < 1e-4
        print(f"  mups2 gauge {mu1} vs {mu2}: {d:.2e}")


if __name__ == "__main__":
    test_normalisation(); print("normalisation OK")
    test_vs_scipy();      print("vs scipy OK")
    test_tau_vs_lna();    print("tau vs lna OK")
    test_mups2_gauge();   print("mups2 gauge OK")
    print("ALL THEORY TESTS PASSED")
