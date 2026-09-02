"""Julia batch prototype, FIXED: each parameter point gets its own save grid.

x0 = ln(a_ini) depends on the parameters, so the output grid xs differs per
point.  The first version passed a single shared grid (taken from the last
point in the loop); diffrax rejected it, Julia silently accepted it and
produced numbers for save times outside its own integration range.

Correct version: XS is an (n, 500) matrix, one row per point -- matching
what rhopsol_jax does per parameter point.
"""
import os
import sys
import time

os.environ.setdefault("PYTHON_JULIAPKG_EXE",
                      "/Applications/Julia-1.11.app/Contents/Resources/julia/bin/julia")
os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")

import numpy as np

REPO = "/Users/starfish/Desktop/PhD2025-26/distances/TorC"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "benchmarks", "backends"))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

from common import setup, norm_E
from torc.theory import _TERM_LNA
from torc.priors import PRIOR_BOUNDS

from juliacall import Main as jl
jl.seval("import Pkg"); jl.seval('Pkg.add(["OrdinaryDiffEq"]; io=devnull)')
jl.seval("using OrdinaryDiffEq")
jl.seval(f'include("{os.path.join(REPO, "benchmarks", "torc_rhs.jl")}")')

# NOTE: XS is now (n, nout) -- one save grid per point.
jl.seval(r"""
function solve_batch(P, X0, Y0, XS)
    n, nout = size(XS)
    out = Matrix{Float64}(undef, n, nout)
    for i in 1:n
        xs = @view XS[i, :]
        y0 = [Y0[i,1], Y0[i,2], Y0[i,3]]
        prob = ODEProblem(torc_rhs!, y0, (X0[i], 0.0), collect(@view P[i, :]))
        s = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=xs,
                  maxiters=100_000)
        if s.retcode != ReturnCode.Success
            out[i, :] .= NaN
        else
            da = [s(x)[2] for x in xs]
            E = da .* exp.(-2 .* xs)
            out[i, :] = E ./ E[end]
        end
    end
    return out
end
""")

N = 25
rng = np.random.default_rng(0)
P, X0, Y0, XS = [], [], [], []
for _ in range(N):
    p = {k: rng.uniform(*PRIOR_BOUNDS[k]) for k in PRIOR_BOUNDS}
    x0, y0, args, xs = setup(p)
    P.append(args); X0.append(x0); Y0.append(y0); XS.append(xs)
P, X0, Y0, XS = np.array(P), np.array(X0), np.array(Y0), np.array(XS)
print(f"batch of {N} points, per-row grids, XS shape {XS.shape}\n")


def dfx(i):
    s = diffrax.diffeqsolve(
        _TERM_LNA, diffrax.Tsit5(), t0=float(X0[i]), t1=0.0, dt0=None,
        y0=tuple(float(v) for v in Y0[i]),
        args=tuple(float(v) for v in P[i]),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-8),
        saveat=diffrax.SaveAt(ts=jnp.asarray(XS[i])), max_steps=1_000_000,
        throw=False, adjoint=diffrax.ForwardMode())
    ok = bool(s.result == diffrax.RESULTS.successful)
    return norm_E(np.asarray(s.ys[1]), XS[i]) if ok else np.full(XS.shape[1], np.nan)


E_dfx = np.array([dfx(i) for i in range(N)])
E_jl = np.asarray(jl.solve_batch(P, X0, Y0, XS))

print("TEST 1: each Julia row vs diffrax for the SAME parameters")
ds = []
for i in range(N):
    m = np.isfinite(E_jl[i]) & np.isfinite(E_dfx[i])
    ds.append(np.nanmax(np.abs((E_jl[i][m] - E_dfx[i][m]) / E_dfx[i][m]))
              if m.any() else np.nan)
ds = np.array(ds)
print(f"  max |dE/E| over rows : {np.nanmax(ds):.3e}")
print(f"  -> {'PASS' if np.nanmax(ds) < 1e-6 else 'FAIL'}")

print("\nTEST 2: are rows scrambled?")
mism = 0
for i in range(N):
    m = np.isfinite(E_jl[i]) & np.isfinite(E_dfx[i])
    if not m.any():
        continue
    self_d = np.nanmax(np.abs((E_jl[i][m] - E_dfx[i][m]) / E_dfx[i][m]))
    other = []
    for j in range(N):
        if j == i:
            continue
        mm = np.isfinite(E_jl[i]) & np.isfinite(E_dfx[j])
        if mm.any():
            other.append(np.nanmax(np.abs((E_jl[i][mm] - E_dfx[j][mm])
                                          / E_dfx[j][mm])))
    if other and self_d >= min(other):
        mism += 1
print(f"  rows better matched by ANOTHER row's solution: {mism}")
print(f"  -> {'PASS' if mism == 0 else 'FAIL'}")

print("\nTEST 3: stale results after changing input?")
E1 = np.asarray(jl.solve_batch(P, X0, Y0, XS))
P2, X02, Y02, XS2 = P.copy(), X0.copy(), Y0.copy(), XS.copy()
rng2 = np.random.default_rng(1)
for i in range(N):
    p = {k: rng2.uniform(*PRIOR_BOUNDS[k]) for k in PRIOR_BOUNDS}
    x0, y0, args, xs = setup(p)
    P2[i], X02[i], Y02[i], XS2[i] = args, x0, y0, xs
E2 = np.asarray(jl.solve_batch(P2, X02, Y02, XS2))
changed = sum(not np.allclose(E1[i], E2[i], equal_nan=True) for i in range(N))
print(f"  rows that changed with new parameters: {changed}/{N}")
print(f"  -> {'PASS (no staleness)' if changed == N else 'FAIL'}")

print("\nTIMING (correct per-row grids):")
jl.solve_batch(P, X0, Y0, XS)
ts = []
for _ in range(12):
    t0 = time.perf_counter(); jl.solve_batch(P, X0, Y0, XS)
    ts.append(time.perf_counter() - t0)
ms = float(np.median(ts) * 1e3)
print(f"  julia {N} solves/call : {ms:8.2f} ms  ({ms/N:.3f} ms/solve)")
print(f"  diffrax in-sampler    :    2.020 ms/solve  -> {2.020/(ms/N):.2f}x")
tot = (216_000 / N) * (ms + 0.18) / 1000
print(f"  projected full run    : {tot:7.1f} s  vs 435.5 s -> {435.5/tot:.1f}x")
print("DONE")
