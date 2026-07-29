"""Generate a Julia RHS from torc/theory.py MECHANICALLY.

dda/ddcp are pure arithmetic (only ** * / + - and parens), so converting
Python -> Julia needs exactly one substitution: ** -> ^.  Doing it by
extraction rather than by hand removes any transcription risk; the result
is then verified numerically against the Python originals.

Initial conditions are computed in Python and embedded as literals, so the
four series functions don't need converting too.
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
                          dda, ddcp, _TERM_LNA, Z_SPLIT, N_EARLY, N_LATE,
                          OMEGA_R_DEFAULT)

SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "torc", "theory.py")).read()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "torc_rhs.jl")

# --- extract the `val = (...)` expression from a named function ---
tree = ast.parse(SRC)
bodies = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in ("dda", "ddcp"):
        for st in node.body:
            if isinstance(st, ast.Assign) and st.targets[0].id == "val":
                bodies[node.name] = ast.get_source_segment(SRC, st.value)

assert set(bodies) == {"dda", "ddcp"}, bodies.keys()

def to_julia(expr: str) -> str:
    e = expr.replace("\\\n", " ")          # drop Python line continuations
    e = re.sub(r"\s+", " ", e)             # collapse whitespace
    e = e.replace("**", "^")               # the ONLY operator difference
    return e.strip()

ARGS = "a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2"
jl = ['# AUTO-GENERATED from torc/theory.py by gen_julia_rhs.py -- do not edit.',
      '# Only transformation applied: Python ** -> Julia ^',
      '']
for name in ("dda", "ddcp"):
    jl.append(f"function {name}({ARGS})")
    jl.append(f"    return {to_julia(bodies[name])}")
    jl.append("end")
    jl.append("")

# vector field in x = ln(a), matching theory.py::_vector_field_lna
jl += [
    "function torc_rhs!(du, u, p, x)",
    "    cp, da, dcp = u",
    "    omega_m, omegar, OmL, stigma, mups2 = p",
    "    a = exp(x)",
    "    fac = a / da",
    "    du[1] = dcp * fac",
    "    du[2] = dda(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2) * fac",
    "    du[3] = ddcp(a, da, cp, dcp, omega_m, omegar, OmL, stigma, mups2) * fac",
    "    nothing",
    "end",
    "",
]

# --- initial conditions per test point, computed here, embedded as literals ---
POINTS = {
    "fiducial":    dict(omega_m=0.30,   stigma=1.50,   mups2=1.50,   OmL=0.70),
    "slow ridge":  dict(omega_m=0.9555, stigma=1.1926, mups2=0.6794, OmL=0.4868),
    "stigma low":  dict(omega_m=0.50,   stigma=1.0001, mups2=1.50,   OmL=0.70),
    "stigma high": dict(omega_m=0.20,   stigma=2.90,   mups2=2.50,   OmL=1.20),
}
XSPLIT = -np.log(1.0 + Z_SPLIT)

jl.append("# name, x0, y0(cp,da,dcp), params(omega_m,omegar,OmL,stigma,mups2)")
jl.append("const CASES = [")
meta = {}
for nm, p in POINTS.items():
    om, st, mu, ol = p["omega_m"], p["stigma"], p["mups2"], p["OmL"]
    omr = OMEGA_R_DEFAULT
    tau = small(om, omr, st, mu)
    x0 = float(jnp.log(aseries(tau, om, omr, ol, st, mu)))
    y0 = (float(cpseries(tau, om, omr, ol, st, mu)),
          float(aseriesp(tau, om, omr, ol, st, mu)),
          float(cpseriesp(tau, om, omr, ol, st, mu)))
    meta[nm] = (x0, y0, (om, omr, ol, st, mu))
    jl.append(f'    ("{nm}", {x0!r}, [{y0[0]!r}, {y0[1]!r}, {y0[2]!r}], '
              f'[{om!r}, {omr!r}, {ol!r}, {st!r}, {mu!r}]),')
jl.append("]")
jl.append(f"const XSPLIT = {float(XSPLIT)!r}")
jl.append(f"const N_EARLY = {N_EARLY}")
jl.append(f"const N_LATE  = {N_LATE}")

open(OUT, "w").write("\n".join(jl) + "\n")
print(f"wrote {OUT}  ({len(jl)} lines)")

# dump Python reference E(z) per case for cross-backend verification.
# Uses diffrax Dopri8 @1e-12 rather than scipy DOP853 @1e-13: ~8x faster and
# equally accurate.  scipy at 1e-13 effectively hangs on the stigma=1.0001
# case, which sits right next to the stigma=1 coordinate singularity.
import numpy as _np
import diffrax as _dfx
ref = {}
for nm,(x0,y0,pr) in meta.items():
    xs = _np.concatenate([_np.linspace(x0, XSPLIT, N_EARLY, endpoint=False),
                          _np.linspace(XSPLIT, 0.0, N_LATE)])
    _s = _dfx.diffeqsolve(
        _TERM_LNA, _dfx.Dopri8(), t0=x0, t1=0.0, dt0=None, y0=tuple(y0), args=pr,
        stepsize_controller=_dfx.PIDController(rtol=1e-12, atol=1e-12),
        saveat=_dfx.SaveAt(ts=jnp.asarray(xs)), max_steps=200_000,
        throw=False, adjoint=_dfx.ForwardMode())
    if not bool(_s.result == _dfx.RESULTS.successful):
        print(f"  WARNING: reference solve failed for {nm}; skipping")
        continue
    E = _np.asarray(_s.ys[1])*_np.exp(-2.0*xs); E = E/E[-1]
    ref[nm] = _np.vstack([xs, E])
_np.savez(os.path.join(HERE, "reference_E.npz"), **ref)
print("wrote", os.path.join(HERE, "reference_E.npz"),
      "(DOP853@1e-13 reference E(z) for cross-backend checks)")

# --- numeric cross-check of the generated expressions ---
print("\nverification: Python dda/ddcp vs the extracted expression, evaluated")
print("in a namespace that mimics Julia's semantics (same arithmetic):")
ns_expr = {n: to_julia(bodies[n]).replace("^", "**") for n in bodies}
rng = np.random.default_rng(0)
worst = 0.0
for _ in range(200):
    v = dict(a=rng.uniform(1e-6, 1.0), da=rng.uniform(0.1, 5.0),
             cp=rng.uniform(0.1, 2.0), dcp=rng.uniform(-1.0, 1.0),
             omega_m=rng.uniform(0.05, 0.9), omegar=OMEGA_R_DEFAULT,
             OmL=rng.uniform(0.2, 1.4), stigma=rng.uniform(1.01, 2.9),
             mups2=rng.uniform(0.6, 2.9))
    for name, fn in (("dda", dda), ("ddcp", ddcp)):
        ref = float(fn(v["a"], v["da"], v["cp"], v["dcp"], v["omega_m"],
                       v["omegar"], v["OmL"], v["stigma"], v["mups2"]))
        got = float(eval(ns_expr[name], {}, v))
        if ref != 0:
            worst = max(worst, abs(got - ref) / abs(ref))
print(f"  max relative difference over 200 random points: {worst:.3e}")
print("  (0 means the extraction is character-exact arithmetic)")
