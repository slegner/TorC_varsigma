"""Validation of the Lambda theory.  Run:  python tests/test_theory_lam.py

The Lambda model must reduce EXACTLY to the no-lambda theory at lam = 0 --
that is the one property that makes a Delta ln Z between their chains
interpretable.  These tests pin it down at three levels: the right-hand
sides, the solved E(z), and the DESI likelihood.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax
jax.config.update("jax_enable_x64", True)

from torc import theory as T
from torc import theory_lam as TL

OM, OMR, OML, MU = 0.3, 8.24e-5, 0.7, 4.0/3.0
SETS = [(0.2969, 1.0001, 4/3, 0.703),
        (0.2969, 1.5,    4/3, 0.703),
        (0.2893, 1.1,    3.0, 0.9),
        (0.5,    2.0,    0.6, 1.2),
        (0.3,    50.0,   1.5, 0.7),
        (0.3,    5000.0, 1.5, 0.7)]


def test_rhs_reduces_at_zero_lam():
    """dda/ddcp at lam = 0 must equal torc.theory to double-precision noise."""
    rng = np.random.default_rng(0)
    n = 3000
    a   = np.exp(rng.uniform(-20, 0, n))
    da  = np.exp(rng.uniform(-1, 1, n))
    cp  = rng.uniform(0.05, 0.999, n)
    dcp = rng.uniform(-3000, 3000, n)
    for name in ("dda", "ddcp"):
        f0 = getattr(T,  name)(a, da, cp, dcp, OM, OMR, OML, 1.5, MU)
        f1 = getattr(TL, name)(a, da, cp, dcp, OM, OMR, OML, 1.5, MU, 0.0)
        rel = np.max(np.abs(f1 / f0 - 1))
        assert rel < 1e-12, f"{name}: max rel err {rel:.3e}"
        print(f"  {name:5s} lam=0 vs TorC: max rel err {rel:.3e}")


def test_Ez_reduces_at_zero_lam():
    """The solved E(z) at lam = 0 must match, over the whole parameter box.

    TOLERANCE: 1e-4, not 1e-8, and the reason is worth knowing.  The two
    right-hand sides are the same formula written with the sign and
    denominator rearranged, so they agree to ~1e-14 but not bit-for-bit
    (test_rhs_reduces_at_zero_lam).  Adaptive stepping turns that into
    different step sequences, and how much the endpoint moves depends
    entirely on how well conditioned the solve is:

        stigma = 1.0001   max |dE/E| = 5.2e-05      d logL = 6e-03
        stigma = 1.001    max |dE/E| = 1.9e-06      d logL = 6e-03
        stigma = 1.01     max |dE/E| = 3.8e-08      d logL = 8e-05
        stigma >= 1.1     max |dE/E| < 1.7e-08      d logL < 4e-07

    The sensitivity is confined to a sliver at the stigma = 1 coordinate
    singularity and is a property of the SOLVER, not of the lambda terms:
    torc.theory compared against ITSELF at rtol 1e-8 vs 1e-12 fails to
    solve at all at stigma = 1.0001, and agrees to 2e-8 by stigma = 1.01.
    Under the power-law prior ~1% of the prior mass sits below stigma =
    1.01, where logL wobbles by <= 0.006 -- negligible against ln Z
    uncertainties of ~0.1.
    """
    for om, st, mu, ol in SETS:
        _, E0, _ = T.rhopsol_jax(om, OMR, ol, st, mu)
        _, E1, _ = TL.rhopsol_jax(om, OMR, ol, st, mu, 0.0)
        E0, E1 = np.asarray(E0), np.asarray(E1)
        ok = np.isfinite(E0) & np.isfinite(E1)
        rel = np.max(np.abs(E1[ok] / E0[ok] - 1))
        tol = 1e-4 if st < 1.01 else 1e-7
        assert rel < tol, f"om={om} stigma={st}: max rel err {rel:.3e}"
        print(f"  om={om:<7g} stigma={st:<8g} E(z) max rel err {rel:.3e}")


def test_lam_actually_does_something():
    """A non-zero lam must change E(z) well above solver noise, and stay finite."""
    _, E0, _ = TL.rhopsol_jax(OM, OMR, OML, 1.5, MU, 0.0)
    for lam in (0.1, 0.5, 1.0, 1.5):
        _, E1, _ = TL.rhopsol_jax(OM, OMR, OML, 1.5, MU, lam)
        E0a, E1a = np.asarray(E0), np.asarray(E1)
        assert np.all(np.isfinite(E1a)), f"lam={lam} produced non-finite E"
        d = np.max(np.abs(E1a / E0a - 1))
        assert d > 1e-4, f"lam={lam} changed E(z) by only {d:.3e}"
        print(f"  lam={lam:<4g} max |dE/E| = {100*d:6.3f}%")


def test_solver_survives_large_stigma():
    """The power-law prior reaches stigma ~ 1e4; solves must not fail there."""
    for st in (3.0, 10.0, 50.0, 200.0, 1e3, 1e4):
        for lam in (0.0, 0.7):
            _, E, _ = TL.rhopsol_jax(OM, OMR, OML, st, MU, lam)
            assert np.all(np.isfinite(np.asarray(E))), f"stigma={st} lam={lam}"
        print(f"  stigma={st:<8g} finite for lam = 0 and 0.7")


def test_likelihood_reduces_at_zero_lam():
    """The DESI likelihood at lam = 0 must match the no-lambda likelihood."""
    import torc.priors as P0
    from torc.priors_lam import loglikelihood_bounded as ll_lam
    for om, st, mu, ol in SETS[:4]:
        p0 = dict(omega_m=om, h0rd=10000.0, stigma=st, OmL=ol)
        a = float(P0.loglikelihood_bounded(p0))
        b = float(ll_lam(dict(p0, lam=0.0)))
        # absolute, not relative: what matters for ln Z is the shift in logL,
        # and 0.01 is already two orders below the ln Z uncertainty (~0.1)
        assert abs(a - b) < 0.01, f"{a} vs {b}  (d = {abs(a-b):.3e})"
        print(f"  om={om:<7g} stigma={st:<8g} logL {a:14.6f} vs {b:14.6f}")


if __name__ == "__main__":
    for fn in (test_rhs_reduces_at_zero_lam, test_Ez_reduces_at_zero_lam,
               test_lam_actually_does_something, test_solver_survives_large_stigma,
               test_likelihood_reduces_at_zero_lam):
        print(f"\n{fn.__name__}:")
        fn()
    print("\nAll Lambda theory tests passed.")
