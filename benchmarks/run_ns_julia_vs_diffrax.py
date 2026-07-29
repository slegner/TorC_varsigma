"""End-to-end: does the Julia ODE backend actually make nested sampling faster?

Every projection so far has been arithmetic over microbenchmarks.  This runs
the real sampler to convergence with each backend and compares wall time and
ln Z.  Same seed, same settings, same everything except which solver the
likelihood calls.

The backends must agree on ln Z -- a speedup that changes the evidence is
not a speedup, it is a bug.

Run:
    python benchmarks/run_ns_julia_vs_diffrax.py            # both
    python benchmarks/run_ns_julia_vs_diffrax.py --diffrax  # one at a time
    python benchmarks/run_ns_julia_vs_diffrax.py --julia
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import blackjax

from torc.priors import (PRIOR_BOUNDS, logprior_fn, sample_prior,
                         loglikelihood_bounded)

NUM_LIVE = 250
NUM_DELETE = 50
NUM_INNER_STEPS = len(PRIOR_BOUNDS) * 5
SEED = 0


def run(label):
    rng = jax.random.PRNGKey(SEED)
    algo = blackjax.nss(logprior_fn=logprior_fn,
                        loglikelihood_fn=loglikelihood_bounded,
                        num_delete=NUM_DELETE,
                        num_inner_steps=NUM_INNER_STEPS)
    rng, k = jax.random.split(rng)

    t0 = time.perf_counter()
    live = jax.jit(algo.init)(sample_prior(k, NUM_LIVE))
    jax.block_until_ready(live)
    t_init = time.perf_counter() - t0

    step = jax.jit(algo.step)
    dead, n = [], 0
    t0 = time.perf_counter()
    while not (live.logZ_live - live.logZ < -3):
        rng, k = jax.random.split(rng)
        live, d = step(k, live)
        dead.append(d)
        n += 1
        if n % 20 == 0:
            print(f"    [{label}] call {n}: "
                  f"{float(live.logZ_live - live.logZ):8.2f}  "
                  f"{time.perf_counter() - t0:6.0f}s", flush=True)
    t_run = time.perf_counter() - t0

    dead = blackjax.ns.utils.finalise(live, dead)
    from anesthetic import NestedSamples
    s = NestedSamples({kk: np.asarray(v) for kk, v in dead.particles.items()},
                      logL=np.asarray(dead.loglikelihood),
                      logL_birth=np.asarray(dead.loglikelihood_birth))
    lz = s.logZ(1000)
    return dict(label=label, init=t_init, run=t_run, calls=n,
                lnZ=float(lz.mean()), err=float(lz.std()),
                D_KL=float(s.D_KL()),
                means={p: float(s[p].mean()) for p in PRIOR_BOUNDS},
                stds={p: float(s[p].std()) for p in PRIOR_BOUNDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diffrax", action="store_true")
    ap.add_argument("--julia", action="store_true")
    a = ap.parse_args()
    both = not (a.diffrax or a.julia)

    print(f"NUM_LIVE={NUM_LIVE} NUM_DELETE={NUM_DELETE} "
          f"NUM_INNER_STEPS={NUM_INNER_STEPS} ndim={len(PRIOR_BOUNDS)} seed={SEED}\n")

    res = {}
    if a.diffrax or both:
        print("=== diffrax (Dopri8, vmapped) ===", flush=True)
        res["diffrax"] = run("diffrax")
        print(f"  -> {res['diffrax']['run']:.1f}s, "
              f"lnZ={res['diffrax']['lnZ']:.3f}\n", flush=True)

    if a.julia or both:
        print("=== julia (Tsit5, threaded, separate process over IPC) ===", flush=True)
        import torc.julia_ipc as tj
        tj.activate()
        res["julia"] = run("julia")
        tj.deactivate()
        print(f"  -> {res['julia']['run']:.1f}s, "
              f"lnZ={res['julia']['lnZ']:.3f}\n", flush=True)

    print("=" * 70)
    for r in res.values():
        print(f"{r['label']:10s} init {r['init']:6.1f}s  run {r['run']:7.1f}s  "
              f"({r['calls']} calls)  lnZ {r['lnZ']:+.3f} +- {r['err']:.3f}  "
              f"D_KL {r['D_KL']:.2f}")
    if len(res) == 2:
        d, j = res["diffrax"], res["julia"]
        print()
        print(f"  SPEEDUP (run only)      : {d['run']/j['run']:.2f}x")
        print(f"  SPEEDUP (incl. init)    : "
              f"{(d['run']+d['init'])/(j['run']+j['init']):.2f}x")
        dz = j["lnZ"] - d["lnZ"]
        sz = (d["err"] ** 2 + j["err"] ** 2) ** 0.5
        print(f"  ln Z difference         : {dz:+.3f} +- {sz:.3f} "
              f"({abs(dz)/sz:.2f} sigma)")
        print(f"  -> {'CONSISTENT' if abs(dz) < 2*sz else 'INCONSISTENT -- investigate'}")
        print()
        print(f"  {'param':10s} {'diffrax':>22s} {'julia':>22s} {'shift/sig':>10s}")
        for p in PRIOR_BOUNDS:
            m1, s1 = d["means"][p], d["stds"][p]
            m2, s2 = j["means"][p], j["stds"][p]
            print(f"  {p:10s} {m1:12.4f} +- {s1:7.4f} {m2:12.4f} +- {s2:7.4f} "
                  f"{(m2-m1)/s1:+10.3f}")
    print("DONE")


if __name__ == "__main__":
    main()
