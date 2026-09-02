"""Does the uncommitted rtol/atol 1e-7 -> 1e-5 loosening shift logL (and hence ln Z)?

A uniform shift D in logL shifts ln Z by exactly D, so measuring the logL
offset on a fixed set of posterior samples isolates the tolerance effect
from sampling noise -- no need for extra full NS runs.
"""
import sys, os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import diffrax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torc.theory as T
from torc.likelihood import loglikelihood

CH = os.path.join(os.path.abspath("."), "chains")
d = np.load(os.path.join(CH, "torc_ns_inner25.npz"))
keys = ["omega_m", "h0rd", "stigma", "mups2", "OmL"]
n = len(d["logL"])
# use the posterior-weighted region: take the last 1500 dead points (highest L)
sl = slice(max(0, n - 1500), n)
pts = {k: jnp.asarray(d[k][sl]) for k in keys}
print(f"evaluating {len(d['logL'][sl])} high-likelihood dead points", flush=True)

def logl_at(rtol, atol):
    T._CTRL = diffrax.PIDController(rtol=rtol, atol=atol)
    f = jax.jit(jax.vmap(loglikelihood))
    out = np.asarray(jax.block_until_ready(f(pts)))
    return out

l5 = logl_at(1e-5, 1e-5)     # current working tree
l7 = logl_at(1e-7, 1e-7)     # committed value (README reference)
l9 = logl_at(1e-9, 1e-9)     # near-exact reference

ok = np.isfinite(l5) & np.isfinite(l7) & np.isfinite(l9)
print(f"finite in all three: {ok.sum()}/{len(ok)}")
l5, l7, l9 = l5[ok], l7[ok], l9[ok]

for name, a, b in [("1e-5 vs 1e-7", l5, l7),
                   ("1e-7 vs 1e-9", l7, l9),
                   ("1e-5 vs 1e-9", l5, l9)]:
    dd = a - b
    print(f"\n{name}:")
    print(f"  mean dlogL   = {dd.mean():+.4f}   (a uniform shift moves ln Z by this)")
    print(f"  median dlogL = {np.median(dd):+.4f}")
    print(f"  std          = {dd.std():.4f}")
    print(f"  max |dlogL|  = {np.abs(dd).max():.4f}")
print("\nDONE")
