"""Julia (OrdinaryDiffEq + Sundials) backend, driven from Python.

The Julia RHS is AUTO-GENERATED from torc/theory.py by gen_julia_rhs.py --
the only transformation is Python ** -> Julia ^, verified to 0.000e+00
relative difference against the Python originals.  So Julia is provably
integrating the same equations, not a hand-transcribed copy.

This shells out to `julia`, runs one solve per (solver, tol), and parses
the timings back.  Julia JIT-compiles on first use, so the helper script
discards the first call, exactly like timeit() does for the other backends.

Requires:  brew install --cask julia-app
           julia -e 'using Pkg; Pkg.add(["OrdinaryDiffEq","Sundials"])'
           python benchmarks/gen_julia_rhs.py     # writes torc_rhs.jl

Standalone:  python benchmarks/backends/julia_backend.py
"""
import os
import shutil
import subprocess
import tempfile

import numpy as np

from common import setup, report

HERE = os.path.dirname(os.path.abspath(__file__))
RHS_JL = os.path.join(HERE, "..", "torc_rhs.jl")

SOLVERS = ["Tsit5", "Vern8", "Vern9", "Rodas5P [stiff]",
           "CVODE_Adams", "CVODE_BDF [stiff]"]
_ALG = {"Tsit5": "Tsit5()", "Vern8": "Vern8()", "Vern9": "Vern9()",
        "Rodas5P [stiff]": "Rodas5P()", "CVODE_Adams": "CVODE_Adams()",
        "CVODE_BDF [stiff]": "CVODE_BDF()"}

_SCRIPT = r'''
using OrdinaryDiffEq, Sundials, Statistics, Printf
include(ARGS[1])
x0 = parse(Float64, ARGS[2])
y0 = [parse(Float64, s) for s in split(ARGS[3], ",")]
p  = [parse(Float64, s) for s in split(ARGS[4], ",")]
xs = [parse(Float64, s) for s in split(ARGS[5], ",")]
for spec in split(ARGS[6], ";")
    parts = split(spec, "|")
    alg = eval(Meta.parse(parts[1])); tol = parse(Float64, parts[2])
    # ODEProblem construction is INSIDE the timed function, matching what
    # diffrax's diffeqsolve call pays.  Hoisting it out (as an earlier
    # version did) made Julia look ~1.8x faster than it really is.
    f() = solve(ODEProblem(torc_rhs!, copy(y0), (x0, 0.0), p), alg,
                reltol=tol, abstol=tol, saveat=xs, maxiters=1_000_000)
    local s
    try
        s = f()
    catch e
        @printf("%s\t%.0e\tNA\tERR\n", parts[1], tol); continue
    end
    if s.retcode != ReturnCode.Success
        @printf("%s\t%.0e\tNA\tFAIL\n", parts[1], tol); continue
    end
    ts = Float64[]
    for _ in 1:7
        t0 = time_ns(); f(); push!(ts, (time_ns()-t0)/1e6)
    end
    da = [s(x)[2] for x in xs]
    E = da .* exp.(-2 .* xs); E = E ./ E[end]
    @printf("%s\t%.0e\t%.4f\tOK\t%s\n", parts[1], tol, median(ts),
            join([@sprintf("%.17g", v) for v in E], ","))
end
'''


def run_all(tols=(1e-5, 1e-8), params=None):
    if shutil.which("julia") is None:
        return [dict(backend="julia", solver=s, tol=t, ms=None, ok=False,
                     E=None, steps=None, note="julia not on PATH")
                for t in tols for s in SOLVERS]
    if not os.path.exists(RHS_JL):
        return [dict(backend="julia", solver=s, tol=t, ms=None, ok=False,
                     E=None, steps=None, note="run gen_julia_rhs.py first")
                for t in tols for s in SOLVERS]

    x0, y0, args, xs = setup(params)
    spec = ";".join(f"{_ALG[s]}|{t:.0e}" for t in tols for s in SOLVERS)
    with tempfile.NamedTemporaryFile("w", suffix=".jl", delete=False) as fh:
        fh.write(_SCRIPT)
        script = fh.name
    try:
        out = subprocess.run(
            ["julia", script, os.path.abspath(RHS_JL), repr(float(x0)),
             ",".join(repr(float(v)) for v in y0),
             ",".join(repr(float(v)) for v in args),
             ",".join(repr(float(v)) for v in xs), spec],
            capture_output=True, text=True, timeout=1800)
    finally:
        os.unlink(script)

    inv = {v: k for k, v in _ALG.items()}
    res = []
    for line in out.stdout.splitlines():
        f = line.split("\t")
        if len(f) < 4:
            continue
        name = inv.get(f[0], f[0])
        tol = float(f[1])
        if f[3] != "OK":
            res.append(dict(backend="julia", solver=name, tol=tol, ms=None,
                            ok=False, E=None, steps=None, note=f[3]))
        else:
            E = np.array([float(v) for v in f[4].split(",")]) if len(f) > 4 else None
            res.append(dict(backend="julia", solver=name, tol=tol,
                            ms=float(f[2]), ok=True, E=E, steps=None, note=""))
    if not res:
        return [dict(backend="julia", solver=s, tol=t, ms=None, ok=False,
                     E=None, steps=None, note="no output: " + out.stderr[-120:])
                for t in tols for s in SOLVERS]
    return res


if __name__ == "__main__":
    report("Julia (OrdinaryDiffEq + Sundials)", run_all())
