"""
Sanity checks to run right after pasting the expressions into dda / ddcp.

    python check_lambda.py

1. evaluates dda / ddcp at the shared initial state (finite? sensible size?)
2. compares them against the reference model with lam = 0 -- if the Lambda
   model is supposed to reduce to the reference at lam -> 0, these must agree
   to machine precision.  A mismatch means the pasted expression is off.
3. checks that a short integration from the initial state survives.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import equations_lambda as lam_model
from reference_model import ref

cpr     = 0.99
omega_m = 0.3
h       = 0.67
OmL     = 0.7
stigma  = 1.5
mups2   = 4/3
lam     = 1.0

REDUCES_TO_REFERENCE_AT_ZERO_LAM = True   # set False if that is not expected


def main():
    tauini = lam_model.small(cpr, omega_m, h, stigma, mups2)
    a, cp, da, dcp = lam_model.initial_state(tauini, cpr, omega_m, h, OmL, 1, stigma, mups2)
    args = (a, da, cp, dcp, omega_m, h, OmL, 1, stigma, mups2)

    print(f"tau_ini = {tauini:.6e}")
    print(f"state:  a = {a:.6e}  cp = {cp:.6e}  a' = {da:.6e}  cp' = {dcp:.6e}\n")

    # 1. finite?
    try:
        dda_l = lam_model.dda(*args, lam)
        ddcp_l = lam_model.ddcp(*args, lam)
    except NotImplementedError as exc:
        print(f"nothing to check yet: {exc}")
        return
    print(f"lambda model:  a'' = {dda_l: .8e}   cp'' = {ddcp_l: .8e}")
    for name, v in (("dda", dda_l), ("ddcp", ddcp_l)):
        if not np.isfinite(v):
            print(f"  !! {name} is not finite")

    # 2. lam -> 0 limit
    dda_r = ref.dda(*args)
    ddcp_r = ref.ddcp(*args)
    print(f"reference   :  a'' = {dda_r: .8e}   cp'' = {ddcp_r: .8e}")

    if REDUCES_TO_REFERENCE_AT_ZERO_LAM:
        dda_0 = lam_model.dda(*args, 0.0)
        ddcp_0 = lam_model.ddcp(*args, 0.0)
        ok_a = np.isclose(dda_0, dda_r, rtol=1e-10, atol=0)
        ok_c = np.isclose(ddcp_0, ddcp_r, rtol=1e-10, atol=0)
        print(f"\nlam -> 0 limit:  a''  {'MATCHES' if ok_a else 'DIFFERS'} "
              f"(rel {abs(dda_0/dda_r - 1):.3e})")
        print(f"                 cp'' {'MATCHES' if ok_c else 'DIFFERS'} "
              f"(rel {abs(ddcp_0/ddcp_r - 1):.3e})")

    # 3. short integration, does it survive?
    from scipy.integrate import solve_ivp
    y0 = [a, cp, da, dcp]
    sol = solve_ivp(lambda t, y: lam_model.coupdiff(y, t, omega_m, h, OmL, 1, stigma, mups2, lam),
                    [tauini, 1e3*tauini], y0, rtol=1e-8, atol=1e-8)
    print(f"\nshort integration: status={sol.status}, reached tau = {sol.t[-1]:.4e} "
          f"(target {1e3*tauini:.4e}), a = {sol.y[0][-1]:.4e}, cp = {sol.y[1][-1]:.6f}")
    if sol.status != 0:
        print(f"  message: {sol.message}")


if __name__ == "__main__":
    main()
