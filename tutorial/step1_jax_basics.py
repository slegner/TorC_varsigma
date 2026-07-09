"""
STEP 1 — JAX basics, demonstrated on your own TorC theory equations.
====================================================================

Run me with:
    source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
    cd /Users/starfish/Desktop/PhD2025-26/distances/TorC
    python jax_tutorial/step1_jax_basics.py

JAX in one sentence: numpy's API + a compiler (jit) + function
transformations (grad, vmap) — in exchange, your functions must be pure
(no side effects) and must not use Python if/while on array values.

Key insight for our project: the equation functions in test_theory_cpr1.py
(dda, ddcp, aseries, cpseries, ...) only use  + - * / ** , so they work on
JAX arrays UNCHANGED. Only the solver machinery around them (solve_ivp,
quad) needs replacing — that is what steps 2 and 3 do.
"""

import time
import numpy as np

# JAX uses float32 by default (a deep-learning habit). Cosmology needs
# float64, so this line must come BEFORE any other jax call, in every file.
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_theory_cpr1 import dda, cpseries, cpseriesp, omega_r

# Fixed cosmology for all demos
h, omega_m = 0.7, 0.2969
OmL = 1 - omega_m - omega_r(h)
ARGS = (omega_m, h, OmL, 1.0, 1.01, 4/3)   # (omega_m, h, OmL, a0, stigma, mups2)


# ---------------------------------------------------------------------------
# Demo 1 — jax.numpy: same code, different array type
# ---------------------------------------------------------------------------
# dda(a, da, cp, dcp, ...) is YOUR second-order ODE right-hand side.
# Call it with python floats (numpy path) and with jnp arrays (JAX path):

x_np  = dda(0.5, 40.0, 0.9, -0.1, *ARGS)
x_jax = dda(jnp.array(0.5), jnp.array(40.0), jnp.array(0.9), jnp.array(-0.1), *ARGS)

print("Demo 1 | dda with floats    :", x_np)
print("Demo 1 | dda with jnp arrays:", float(x_jax), f"  (type: {type(x_jax).__name__})")
print("        -> your equations are already JAX-compatible; nothing to port.\n")


# ---------------------------------------------------------------------------
# Demo 2 — jax.grad: exact automatic derivatives
# ---------------------------------------------------------------------------
# You derived cpseriesp = d(cpseries)/dtau by hand (it was the function
# MISSING from test_equations.py). JAX differentiates the code for you.
# grad differentiates w.r.t. argument 0 (tau) by default:

tau = 1e-4
dcp_hand = cpseriesp(tau, *ARGS)
dcp_auto = jax.grad(cpseries)(tau, *ARGS)

print("Demo 2 | hand-derived cpseriesp    :", dcp_hand)
print("Demo 2 | jax.grad(cpseries)        :", float(dcp_auto))

# argnums picks WHICH argument to differentiate w.r.t.
# Argument 5 of cpseries is stigma -> sensitivity of cp to the theory param:
dcp_dstigma = jax.grad(cpseries, argnums=5)(tau, *ARGS)
print("Demo 2 | d(cpseries)/d(stigma)     :", float(dcp_dstigma))
print("        -> no more hand-derived *p functions, ever.\n")


# ---------------------------------------------------------------------------
# Demo 3 — jax.vmap: batch any function over any argument
# ---------------------------------------------------------------------------
# Write physics for ONE parameter point; vmap turns it into a batched
# version. Nested sampling will use this to evaluate many live points at
# once. Here: cpseries at one tau, five stigmas, no for-loop.

def cp_of_stigma(stigma):
    return cpseries(1e-4, omega_m, h, OmL, 1.0, stigma, 4/3)

stigmas = jnp.array([1.001, 1.01, 1.1, 1.5, 2.0])
print("Demo 3 | vmap over stigma:", jax.vmap(cp_of_stigma)(stigmas))
print("        -> the -64.9 at stigma=1.001 is real: tau=1e-4 is outside the")
print("           series' convergence radius when (stigma-1) is small. The")
print("           sampler only uses the series at tau_ini ~ 1e-9, where it's fine.\n")


# ---------------------------------------------------------------------------
# Demo 4 — jax.jit: trace once, compile, run at machine speed
# ---------------------------------------------------------------------------
big = jnp.linspace(0.1, 1.0, 2_000_000)

def f_slow(a):
    return dda(a, 40.0, 0.9, -0.1, *ARGS)

f_fast = jax.jit(f_slow)

_ = f_fast(big).block_until_ready()   # first call compiles (slow, once)
# .block_until_ready() is needed for honest timing: JAX dispatches work
# asynchronously and returns before the computation finishes.

t0 = time.perf_counter(); r1 = f_slow(big).block_until_ready(); t1 = time.perf_counter()
t2 = time.perf_counter(); r2 = f_fast(big).block_until_ready(); t3 = time.perf_counter()

print(f"Demo 4 | un-jitted: {(t1-t0)*1e3:6.1f} ms   jitted: {(t3-t2)*1e3:6.1f} ms   "
      f"(2M points, same answer: {bool(jnp.allclose(r1, r2))})\n")


# ---------------------------------------------------------------------------
# Demo 5 — the price you pay: the two rules
# ---------------------------------------------------------------------------
# Rule 1: purity. In-place mutation is forbidden on jax arrays:
x = jnp.zeros(3)
try:
    x[0] = 1.0
except TypeError as e:
    print("Demo 5 | rule 1 (purity):", str(e)[:70], "...")
y = x.at[0].set(1.0)          # the functional replacement
print("        functional update: x.at[0].set(1.0) ->", y)

# Rule 2: no Python control flow on traced values. Inside jit, JAX traces
# with abstract values — it knows shapes, not numbers, so `if x > 0` is
# unanswerable at trace time:
@jax.jit
def bad(x):
    if x > 0:                 # <- python bool of a traced value: boom
        return x
    return -x

try:
    bad(jnp.array(1.0))
except jax.errors.TracerBoolConversionError as e:
    print("Demo 5 | rule 2 (control flow): TracerBoolConversionError (as expected)")

good = jax.jit(lambda x: jnp.where(x > 0, x, -x))   # branchless replacement
print("        jnp.where version works:", good(jnp.array(-3.0)))
print()
print("This is exactly why scipy's solve_ivp / quad cannot be jitted (they are")
print("Python while-loops on data) and why step 2 uses diffrax instead.")
