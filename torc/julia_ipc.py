"""Julia ODE backend over IPC -- drop-in for torc.theory.rhopsol_jax.

Embedding Julia in-process (juliacall) and calling it from inside
jax.pure_callback deadlocks or segfaults in every configuration tested,
because JAX dispatches the callback from a runtime-controlled thread and
the embedded Julia runtime is thread-sensitive.

Here Julia runs as a SEPARATE PROCESS.  The Python side only ever writes
bytes to a socket, which is thread-safe, so the failure class disappears.
Julia keeps its own threads internally (Threads.@threads over the batch),
which is where the speedup comes from -- unlike vmap, those threads are
independent, so there is no lockstep waste.

    export PYTHON_JULIA_EXE=/Applications/Julia-1.11.app/.../julia   # optional
    python benchmarks/gen_julia_rhs.py      # writes benchmarks/torc_rhs.jl
    import torc.julia_ipc as ji; ji.activate()

activate() starts the server on first use and shuts it down at exit.
"""
import atexit
import glob
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

import jax
import jax.numpy as jnp

from .theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                     Z_SPLIT, N_EARLY, N_LATE, OMEGA_R_DEFAULT)

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER_JL = os.path.join(HERE, "..", "benchmarks", "torc_server.jl")
RHS_JL = os.path.join(HERE, "..", "benchmarks", "torc_rhs.jl")
XSPLIT = -np.log(1.0 + Z_SPLIT)
NOUT = N_EARLY + N_LATE

_proc = None
_sock = None
_lock = threading.Lock()      # JAX may call the callback from several threads
_sockpath = None


def _find_julia():
    exe = os.environ.get("PYTHON_JULIA_EXE")
    if exe:
        return exe

    def ver(p):
        m = re.search(r"Julia-(\d+)\.(\d+)", p)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    # sort by VERSION: "Julia-1.5" beats "Julia-1.11" lexically but is older
    c = sorted(glob.glob(
        "/Applications/Julia-*.app/Contents/Resources/julia/bin/julia"),
        key=ver, reverse=True)
    c = [p for p in c if ver(p) >= (1, 10)]
    if c:
        return c[0]
    from shutil import which
    return which("julia") or "julia"


def start(timeout=600):
    """Launch the Julia server and connect.  Idempotent."""
    global _proc, _sock, _sockpath
    if _sock is not None:
        return
    if not os.path.exists(RHS_JL):
        raise FileNotFoundError(
            f"{RHS_JL} missing -- run: python benchmarks/gen_julia_rhs.py")

    _sockpath = os.path.join(tempfile.gettempdir(), f"torc_{os.getpid()}.sock")
    if os.path.exists(_sockpath):
        os.unlink(_sockpath)

    julia = _find_julia()
    _proc = subprocess.Popen(
        [julia, "-t", "auto", os.path.abspath(SERVER_JL), _sockpath],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1)

    t0 = time.time()
    while True:
        line = _proc.stdout.readline()
        if line:
            if "TORC_SERVER_READY" in line:
                print(f"[julia_ipc] {line.strip()} "
                      f"(startup {time.time()-t0:.1f}s)", file=sys.stderr)
                break
            if "ERROR" in line or "error" in line:
                print(f"[julia_ipc] {line.rstrip()}", file=sys.stderr)
        if _proc.poll() is not None:
            raise RuntimeError("julia server exited during startup")
        if time.time() - t0 > timeout:
            raise TimeoutError("julia server did not become ready")

    _sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    _sock.connect(_sockpath)
    atexit.register(stop)


def stop():
    global _proc, _sock
    try:
        if _sock is not None:
            _sock.close()
    except Exception:
        pass
    _sock = None
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except Exception:
            _proc.kill()
    _proc = None
    if _sockpath and os.path.exists(_sockpath):
        try:
            os.unlink(_sockpath)
        except Exception:
            pass


def _recv_exact(n):
    buf = bytearray()
    while len(buf) < n:
        chunk = _sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("julia server closed the connection")
        buf += chunk
    return bytes(buf)


def solve_batch(P, X0, Y0, tol=1e-8):
    """P (n,5), X0 (n,), Y0 (n,3)  ->  E (n, NOUT).  Thread-safe."""
    start()
    P = np.ascontiguousarray(np.atleast_2d(P), dtype=np.float64)
    X0 = np.ascontiguousarray(np.ravel(X0), dtype=np.float64)
    Y0 = np.ascontiguousarray(np.atleast_2d(Y0), dtype=np.float64)
    n = P.shape[0]
    # julia is column-major: send P as (5,n) and Y0 as (3,n)
    req = (struct.pack("<i", n) + struct.pack("<d", float(tol))
           + P.T.tobytes(order="F") + X0.tobytes()
           + Y0.T.tobytes(order="F"))
    with _lock:
        _sock.sendall(req)
        raw = _recv_exact(NOUT * n * 8)
    return np.frombuffer(raw, dtype=np.float64).reshape(NOUT, n, order="F").T


def rhopsol_ipc(omega_m, omegar, OmL, stigma, mups2, tol=1e-8):
    """Same signature and return convention as theory.rhopsol_jax."""
    tauini = small(omega_m, omegar, stigma, mups2)
    x0_raw = jnp.log(aseries(tauini, omega_m, omegar, OmL, stigma, mups2))
    ok = jnp.isfinite(x0_raw) & (x0_raw < XSPLIT)
    x0 = jnp.where(ok, x0_raw, -45.0)
    y0 = jnp.stack([cpseries(tauini, omega_m, omegar, OmL, stigma, mups2),
                    aseriesp(tauini, omega_m, omegar, OmL, stigma, mups2),
                    cpseriesp(tauini, omega_m, omegar, OmL, stigma, mups2)])
    P = jnp.stack([omega_m, jnp.asarray(omegar, dtype=x0.dtype), OmL,
                   stigma, mups2])

    def host(p, a, b):
        p = np.asarray(p)
        batched = p.ndim > 1
        E = solve_batch(p, np.asarray(a), np.asarray(b), tol)
        return E if batched else E[0]

    E = jax.pure_callback(
        host, jax.ShapeDtypeStruct((NOUT,), jnp.float64), P, x0, y0,
        vmap_method="expand_dims")

    E = jnp.where(ok, E, jnp.nan)
    xs = jnp.concatenate([jnp.linspace(x0, XSPLIT, N_EARLY, endpoint=False),
                          jnp.linspace(XSPLIT, 0.0, N_LATE)])
    a_grid = jnp.exp(xs)
    z = 1.0 / a_grid - 1.0
    return z[::-1], E[::-1], a_grid


def activate(tol=1e-8):
    import torc.likelihood as L
    start()
    L.rhopsol_jax = lambda om, omr, ol, st, mu: rhopsol_ipc(
        om, omr, ol, st, mu, tol)
    print("[julia_ipc] torc.likelihood now uses the Julia IPC backend",
          file=sys.stderr)


def deactivate():
    import torc.likelihood as L
    from .theory import rhopsol_jax
    L.rhopsol_jax = rhopsol_jax
