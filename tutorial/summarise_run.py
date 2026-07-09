"""Post-run summary of chains/torc_ns.npz: posterior constraints + a census
of the 'impossible' (logL = -1e30) points — the ones whose failed ODE solves
crashed the pre-guard theory_jax."""
import os
import numpy as np
from anesthetic import NestedSamples

here = os.path.dirname(__file__)
d = np.load(os.path.join(here, "..", "chains", "torc_ns.npz"))
params = ["omega_m", "h0rd", "stigma", "mups2", "OmL"]

print(f"logZ = {float(d['logZ']):.3f}   |   dead points: {len(d['logL'])}")

samples = NestedSamples({k: d[k] for k in params},
                        logL=d["logL"], logL_birth=d["logL_birth"])
print("\nposterior means +- std:")
for n in params:
    print(f"  {n:8s} = {samples[n].mean():10.4f} +- {samples[n].std():.4f}")

# --- the failure census ----------------------------------------------------
bad = d["logL"] <= -1e29
print(f"\nfloored points (logL = -1e30): {bad.sum()} of {len(bad)}")
if bad.any():
    print(f"{'param':>8} {'failing range':>22}   {'prior range':>22}")
    for n in params:
        v = d[n][bad]
        print(f"{n:>8} [{v.min():9.4f}, {v.max():9.4f}]", end="")
        print(f"   [{d[n].min():9.4f}, {d[n].max():9.4f}]")
    print("\nfirst 10 failing points:")
    print(" ".join(f"{n:>9}" for n in params))
    for i in np.where(bad)[0][:10]:
        print(" ".join(f"{d[n][i]:9.4f}" for n in params))
