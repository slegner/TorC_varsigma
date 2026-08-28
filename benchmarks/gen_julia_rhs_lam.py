"""Generate the Julia RHS for the LAMBDA theory from torc/theory_lam.py.

Same mechanical extraction as gen_julia_rhs.py (the only Python -> Julia
transformation is ** -> ^), applied to the lam-carrying dda/ddcp, and with
the parameter vector widened from 5 to 6 entries:

    p = (omega_m, omegar, OmL, stigma, mups2, lam)

Writes benchmarks/torc_rhs_lam.jl, then verifies the emitted expressions
against the Python originals numerically.

    python benchmarks/gen_julia_rhs_lam.py
"""
import ast
import os
import re
import sys

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                         Z_SPLIT, N_EARLY, N_LATE, OMEGA_R_DEFAULT)
from torc.theory_lam import dda, ddcp

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "..", "torc", "theory_lam.py")).read()
OUT = os.path.join(HERE, "torc_rhs_lam.jl")

tree = ast.parse(SRC)
bodies = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in ("dda", "ddcp"):
        for st in node.body:
            if isinstance(st, ast.Assign) and getattr(st.targets[0], "id", None) == "val":
                bodies[node.name] = ast.get_source_segment(SRC, st.value)
assert set(bodies) == {"dda", "ddcp"}, bodies.keys()


def to_julia(expr: str) -> str:
    e = expr.replace("\\\n", " ")
    e = re.sub(r"\s+", " ", e)
    e = e.replace("**", "^")
    return e.strip()


ARGS = "a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2, lam"
jl = ['# AUTO-GENERATED from torc/theory_lam.py by gen_julia_rhs_lam.py -- do not edit.',
      '# Only transformation applied: Python ** -> Julia ^',
      '']
for name in ("dda", "ddcp"):
    jl.append(f"function {name}({ARGS})")
    jl.append(f"    return {to_julia(bodies[name])}")
    jl.append("end")
    jl.append("")

jl += [
    "function torc_rhs!(du, u, p, x)",
    "    cp, da, dcp = u",
    "    omega_m, omegar, OmL, stigma, mups2, lam = p",
    "    a = exp(x)",
    "    fac = a / da",
    "    du[1] = dcp * fac",
    f"    du[2] = dda({ARGS}) * fac",
    f"    du[3] = ddcp({ARGS}) * fac",
    "    nothing",
    "end",
    "",
]

POINTS = {
    "fiducial":    dict(omega_m=0.30,   stigma=1.50,   mups2=1.50,   OmL=0.70, lam=0.0),
    "lam mid":     dict(omega_m=0.30,   stigma=1.50,   mups2=1.50,   OmL=0.70, lam=0.75),
    "lam high":    dict(omega_m=0.30,   stigma=1.50,   mups2=1.50,   OmL=0.00, lam=1.50),
    "slow ridge":  dict(omega_m=0.9555, stigma=1.1926, mups2=0.6794, OmL=0.4868, lam=0.3),
    "stigma low":  dict(omega_m=0.50,   stigma=1.0001, mups2=1.50,   OmL=0.70, lam=1.0),
    "stigma high": dict(omega_m=0.20,   stigma=2.90,   mups2=2.50,   OmL=1.20, lam=0.5),
}
XSPLIT = -np.log(1.0 + Z_SPLIT)

jl.append("# name, x0, y0(cp,da,dcp), params(omega_m,omegar,OmL,stigma,mups2,lam)")
jl.append("const CASES = [")
for nm, p in POINTS.items():
    om, st, mu, ol, lm = p["omega_m"], p["stigma"], p["mups2"], p["OmL"], p["lam"]
    omr = OMEGA_R_DEFAULT
    tau = small(om, omr, st, mu)
    x0 = float(jnp.log(aseries(tau, om, omr, ol, st, mu)))
    y0 = (float(cpseries(tau, om, omr, ol, st, mu)),
          float(aseriesp(tau, om, omr, ol, st, mu)),
          float(cpseriesp(tau, om, omr, ol, st, mu)))
    jl.append(f'    ("{nm}", {x0!r}, [{y0[0]!r}, {y0[1]!r}, {y0[2]!r}], '
              f'[{om!r}, {omr!r}, {ol!r}, {st!r}, {mu!r}, {lm!r}]),')
jl.append("]")
jl.append(f"const XSPLIT = {float(XSPLIT)!r}")
jl.append(f"const N_EARLY = {N_EARLY}")
jl.append(f"const N_LATE  = {N_LATE}")

open(OUT, "w").write("\n".join(jl) + "\n")
print(f"wrote {OUT}  ({len(jl)} lines)")

# --- numeric cross-check of the emitted expressions ------------------------
print("\nverification: Python dda/ddcp vs the emitted expression evaluated with")
print("Julia's ^ mapped back to ** (identical arithmetic):")
ns_expr = {n: to_julia(bodies[n]).replace("^", "**") for n in bodies}
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(500):
    v = dict(a=rng.uniform(1e-6, 1.0), da=rng.uniform(0.1, 5.0),
             cp=rng.uniform(0.1, 2.0), dcp=rng.uniform(-1.0, 1.0),
             omega_m=rng.uniform(0.05, 0.9), omegar=OMEGA_R_DEFAULT,
             OmL=rng.uniform(0.0, 1.4), stigma=rng.uniform(1.01, 2.9),
             mups2=rng.uniform(0.6, 2.9), lam=rng.uniform(0.0, 1.5))
    for name, fn in (("dda", dda), ("ddcp", ddcp)):
        ref = float(fn(v["a"], v["da"], v["cp"], v["dcp"], v["omega_m"],
                       v["omegar"], v["OmL"], v["stigma"], v["mups2"], v["lam"]))
        got = float(eval(ns_expr[name], {"__builtins__": {}}, dict(v)))
        worst = max(worst, abs(got - ref) / max(abs(ref), 1e-300))
print(f"  max relative difference over 500 random points: {worst:.3e}")
assert worst < 1e-12, "emitted expression does not match the Python original"
print("  OK")
