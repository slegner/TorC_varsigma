"""DEAD END -- kept as documentation.  DO NOT USE.

Embeds Julia in the Python process (juliacall) and calls it from inside
jax.pure_callback.  This FAILS in every configuration tested:

    THREADS=auto + HANDLE_SIGNALS=yes  -> hang
    THREADS=1    + HANDLE_SIGNALS=no   -> hang
    THREADS=auto + HANDLE_SIGNALS=no   -> SIGBUS

Cause: JAX dispatches pure_callback from a runtime-controlled thread,
and the embedded Julia runtime is thread-sensitive.  Isolating the
variables showed it is NOT Threads.@threads (serial hangs too), NOT
marshalling, NOT allocation, and NOT JIT compilation -- a bare ODE solve
under jit works fine, the real one does not.

Use torc/julia_ipc.py instead: Julia in a SEPARATE PROCESS over a unix
socket.  Same speedup, none of this.
"""
import os
import sys

import numpy as np

import jax
import jax.numpy as jnp

from .theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                     Z_SPLIT, N_EARLY, N_LATE, OMEGA_R_DEFAULT)

HERE = os.path.dirname(os.path.abspath(__file__))
RHS_JL = os.path.join(HERE, "..", "benchmarks", "torc_rhs.jl")
XSPLIT = -np.log(1.0 + Z_SPLIT)
NOUT = N_EARLY + N_LATE

_jl = None


def _boot(tol=1e-8):
    """Start Julia once and define the threaded batch solver."""
    global _jl
    if _jl is not None:
        return _jl
    os.environ.setdefault("PYTHON_JULIACALL_THREADS", "auto")
    os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
    # juliapkg resolves libjulia relative to the binary's parent, so a
    # symlink such as /usr/local/bin/julia breaks it ("could not load
    # library .../lib/julia/sys.dylib").  Point at a real install.
    if "PYTHON_JULIAPKG_EXE" not in os.environ:
        import glob as _glob
        import re as _re

        def _ver(path):
            m = _re.search(r"Julia-(\d+)\.(\d+)", path)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

        # sort by VERSION, not lexically: "Julia-1.5" sorts above "Julia-1.11"
        # as a string, and juliacall needs >= 1.10.
        _c = sorted(_glob.glob(
            "/Applications/Julia-*.app/Contents/Resources/julia/bin/julia"),
            key=_ver, reverse=True)
        _c = [c for c in _c if _ver(c) >= (1, 10)]
        if _c:
            os.environ["PYTHON_JULIAPKG_EXE"] = _c[0]
            os.environ.setdefault("PYTHON_JULIAPKG_OFFLINE", "yes")
    if not os.path.exists(RHS_JL):
        raise FileNotFoundError(
            f"{RHS_JL} missing -- run: python benchmarks/gen_julia_rhs.py")
    from juliacall import Main as jl
    jl.seval("import Pkg")
    jl.seval('Pkg.add(["OrdinaryDiffEq"]; io=devnull)')
    jl.seval("using OrdinaryDiffEq, Base.Threads")
    jl.seval(f'include("{os.path.abspath(RHS_JL)}")')
    jl.seval(r"""
    function _one!(out, i, P, X0, Y0, XS, tol)
        xs = @view XS[i, :]
        prob = ODEProblem(torc_rhs!, [Y0[i,1],Y0[i,2],Y0[i,3]],
                          (X0[i], 0.0), collect(@view P[i, :]))
        s = solve(prob, Tsit5(), reltol=tol, abstol=tol, saveat=xs,
                  maxiters=100_000)
        n = length(xs)
        if s.retcode != ReturnCode.Success || length(s.u) != n
            @inbounds out[i, :] .= NaN
        else
            @inbounds for k in 1:n
                out[i,k] = s.u[k][2]*exp(-2*xs[k])
            end
            e = out[i,n]
            @inbounds for k in 1:n; out[i,k] /= e; end
        end
    end

    function torc_batch(P, X0, Y0, XS, tol)
        n, nout = size(XS)
        out = Matrix{Float64}(undef, n, nout)
        Threads.@threads for i in 1:n
            _one!(out, i, P, X0, Y0, XS, tol)
        end
        out
    end
    """)
    print(f"[theory_julia] Julia ready, {jl.seval('Threads.nthreads()')} threads",
          file=sys.stderr)
    _jl = jl
    return jl


def _host_solve(P, X0, Y0, XS, tol=1e-8):
    """Host side of the callback: batch of parameter rows -> E(z) rows.

    Under vmap the callback receives a leading batch axis; unbatched it does
    not.  The declared ShapeDtypeStruct must match exactly, so drop the axis
    we added when the caller did not supply one.
    """
    jl = _boot()
    P = np.asarray(P, dtype=np.float64)
    batched = P.ndim > 1
    P = np.ascontiguousarray(np.atleast_2d(P))
    X0 = np.ascontiguousarray(np.asarray(X0, dtype=np.float64).ravel())
    Y0 = np.ascontiguousarray(np.atleast_2d(np.asarray(Y0, dtype=np.float64)))
    XS = np.ascontiguousarray(np.atleast_2d(np.asarray(XS, dtype=np.float64)))
    out = np.asarray(jl.torc_batch(P, X0, Y0, XS, float(tol)),
                     dtype=np.float64)
    return out if batched else out[0]


def rhopsol_julia(omega_m, omegar, OmL, stigma, mups2, tol=1e-8):
    """Same signature and return convention as theory.rhopsol_jax."""
    tauini = small(omega_m, omegar, stigma, mups2)
    x0_raw = jnp.log(aseries(tauini, omega_m, omegar, OmL, stigma, mups2))
    ok = jnp.isfinite(x0_raw) & (x0_raw < XSPLIT)
    x0 = jnp.where(ok, x0_raw, -45.0)
    y0 = jnp.stack([cpseries(tauini, omega_m, omegar, OmL, stigma, mups2),
                    aseriesp(tauini, omega_m, omegar, OmL, stigma, mups2),
                    cpseriesp(tauini, omega_m, omegar, OmL, stigma, mups2)])
    xs = jnp.concatenate([jnp.linspace(x0, XSPLIT, N_EARLY, endpoint=False),
                          jnp.linspace(XSPLIT, 0.0, N_LATE)])
    P = jnp.stack([omega_m, jnp.asarray(omegar, dtype=x0.dtype), OmL,
                   stigma, mups2])

    E = jax.pure_callback(
        lambda p, a, b, c: _host_solve(p, a, b, c, tol),
        jax.ShapeDtypeStruct((NOUT,), jnp.float64),
        P, x0, y0, xs, vmap_method="expand_dims")

    E = jnp.where(ok, E, jnp.nan)
    a_grid = jnp.exp(xs)
    z = 1.0 / a_grid - 1.0
    return z[::-1], E[::-1], a_grid


def activate(tol=1e-8):
    """Point torc.likelihood at the Julia solver (call before sampling)."""
    import torc.likelihood as L
    _boot(tol)
    L.rhopsol_jax = lambda om, omr, ol, st, mu: rhopsol_julia(
        om, omr, ol, st, mu, tol)
    print("[theory_julia] torc.likelihood now uses the Julia backend",
          file=sys.stderr)


def deactivate():
    """Restore the diffrax solver."""
    import torc.likelihood as L
    from .theory import rhopsol_jax
    L.rhopsol_jax = rhopsol_jax
    print("[theory_julia] restored the diffrax backend", file=sys.stderr)
