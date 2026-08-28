"""Nested sampling with DESI DR2 BAO and/or DESY5 supernovae: 1000 live, Julia.

Same settings as scripts/run_ns_julia_1000_lam.py (live points, delete
fraction, inner-step rule, seed, convergence test), so ln Z is comparable
across every run in chains/.

Data set and model are chosen from the environment:

    TORC_DATA=bao|sn|bao+sn     (default bao+sn)
    TORC_SN=pantheonplus|desy5  which supernovae   (default pantheonplus)
    TORC_SAMPLE_LAM=0|1         include the lambda term      (default 1)
    TORC_SAMPLE_OML=0|1         include the Omega_Lambda fluid (default 1)

    # DESI + DESY5, lambda + OmL
    DESY5_DIR=... python scripts/run_ns_julia_1000_sn.py
    # DESI + DESY5, no lambda (the baseline for those two)
    DESY5_DIR=... TORC_SAMPLE_LAM=0 python scripts/run_ns_julia_1000_sn.py
    # DESY5 alone
    DESY5_DIR=... TORC_DATA=sn python scripts/run_ns_julia_1000_sn.py

Output: chains/torc_ns_julia_1000_<data>[_lam][_noOmL]_<stigmaprior>.npz

NOTE: DESY5 adds NO sampled parameters -- M and H0 are marginalised
analytically (torc/likelihood_sn.py).  It constrains the shape of E(z);
h0rd is a BAO-only quantity, so with TORC_DATA=sn it is dropped.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import blackjax
import tqdm

import torc.priors_lam as P
from torc.priors_lam import (logprior_fn, sample_prior, make_bounded,
                             SAMPLE_OML, SAMPLE_LAM, STIGMA_PRIOR,
                             STIGMA_LAMBDA)
from torc.likelihood_combined import make_loglikelihood
import torc.julia_ipc_lam as ji

SN_SET = os.environ.get("TORC_SN", "pantheonplus").lower()
if SN_SET not in ("pantheonplus", "desy5"):
    raise SystemExit(f"TORC_SN must be pantheonplus or desy5 (got {SN_SET!r})")
DATA = os.environ.get("TORC_DATA", "bao+sn").lower()
if DATA not in ("bao", "sn", "bao+sn"):
    raise SystemExit(f"TORC_DATA must be bao, sn or bao+sn (got {DATA!r})")
USE_BAO, USE_SN = "bao" in DATA, "sn" in DATA

# h0rd only enters the BAO likelihood; sampling it against SN alone would be
# a flat direction costing evidence for nothing.
PRIOR_BOUNDS = dict(P.PRIOR_BOUNDS)
if not USE_BAO:
    PRIOR_BOUNDS.pop("h0rd", None)
    P.PRIOR_BOUNDS = PRIOR_BOUNDS

NUM_LIVE = 1000
NUM_DELETE = NUM_LIVE // 10
NUM_INNER_STEPS = len(PRIOR_BOUNDS) * 5
SEED = 0

if STIGMA_PRIOR == "uniform":
    _TAG = ""
elif STIGMA_PRIOR == "powerlaw":
    _TAG = "_powerlaw"
else:
    _TAG = f"_{STIGMA_PRIOR}_lam{STIGMA_LAMBDA:g}".replace(".", "p")
_SN_TAG = {"pantheonplus": "pp", "desy5": "desy5"}[SN_SET]
_NAME = {"bao": "bao", "sn": _SN_TAG, "bao+sn": f"bao_{_SN_TAG}"}[DATA]
OUT = os.path.join(
    HERE, "..", "chains",
    f"torc_ns_julia_1000_{_NAME}"
    f"{'_lam' if SAMPLE_LAM else ''}{'' if SAMPLE_OML else '_noOmL'}{_TAG}.npz")

if __name__ == "__main__":
    print(f"data: {DATA} ({SN_SET})   lambda: {SAMPLE_LAM}   OmL: {SAMPLE_OML}")
    print(f"ndim={len(PRIOR_BOUNDS)} ({', '.join(PRIOR_BOUNDS)})")
    print(f"stigma prior: {STIGMA_PRIOR}  bounds={PRIOR_BOUNDS['stigma']}")
    print(f"NUM_LIVE={NUM_LIVE} NUM_DELETE={NUM_DELETE} "
          f"NUM_INNER_STEPS={NUM_INNER_STEPS} seed={SEED}")
    if os.path.exists(OUT):
        print(f"\nREFUSING to overwrite {OUT}\nmove or delete it first.")
        sys.exit(1)

    sn_data = None
    if USE_SN:
        from torc import likelihood_sn as SN
        sn_data = (SN.load_pantheonplus() if SN_SET == "pantheonplus"
                   else SN.load_desy5())
        print(f"{SN_SET}: {len(sn_data['zhd'])} supernovae, "
              f"z = {float(sn_data['zhd'].min()):.4f} .. "
              f"{float(sn_data['zhd'].max()):.4f}")

    ji.activate_combined()
    logl = make_bounded(make_loglikelihood(use_bao=USE_BAO, use_sn=USE_SN,
                                           sn_data=sn_data,
                                           sn_dataset=SN_SET))

    rng = jax.random.PRNGKey(SEED)
    algo = blackjax.nss(logprior_fn=logprior_fn, loglikelihood_fn=logl,
                        num_delete=NUM_DELETE, num_inner_steps=NUM_INNER_STEPS)
    rng, k = jax.random.split(rng)
    t0 = time.perf_counter()
    live = jax.jit(algo.init)(sample_prior(k, NUM_LIVE))
    jax.block_until_ready(live)
    print(f"init done in {time.perf_counter()-t0:.1f}s", flush=True)

    step = jax.jit(algo.step)
    dead, n = [], 0
    t0 = time.perf_counter()
    with tqdm.tqdm(desc="dead points", unit=" pts") as pbar:
        while not (live.logZ_live - live.logZ < -3):
            rng, k = jax.random.split(rng)
            live, d = step(k, live)
            dead.append(d)
            n += 1
            pbar.update(NUM_DELETE)
            if n % 25 == 0:
                print(f"  call {n}: logZ_live-logZ="
                      f"{float(live.logZ_live-live.logZ):9.2f}  "
                      f"{time.perf_counter()-t0:6.0f}s", flush=True)
    run_t = time.perf_counter() - t0
    print(f"\nrun time: {run_t:.1f} s ({n} calls)", flush=True)

    dead = blackjax.ns.utils.finalise(live, dead)
    from anesthetic import NestedSamples
    s = NestedSamples({kk: np.asarray(v) for kk, v in dead.particles.items()},
                      logL=np.asarray(dead.loglikelihood),
                      logL_birth=np.asarray(dead.loglikelihood_birth))
    lz = s.logZ(1000)
    ess = int(blackjax.ns.utils.ess(jax.random.PRNGKey(1), dead))

    np.savez(OUT, logL=np.asarray(dead.loglikelihood),
             logL_birth=np.asarray(dead.loglikelihood_birth),
             logZ=float(lz.mean()), logZ_err=float(lz.std()),
             num_live=NUM_LIVE, num_delete=NUM_DELETE,
             num_inner_steps=NUM_INNER_STEPS, run_time=run_t,
             data=DATA, sn_dataset=SN_SET, sample_OmL=bool(SAMPLE_OML),
             sample_lam=bool(SAMPLE_LAM), stigma_prior=STIGMA_PRIOR,
             **{kk: np.asarray(v) for kk, v in dead.particles.items()})

    print(f"ln Z = {lz.mean():.4f} +- {lz.std():.4f}   D_KL = {s.D_KL():.2f}")
    print(f"ESS  ~ {ess}")
    print("\nposterior means +- std:")
    for p in PRIOR_BOUNDS:
        print(f"  {p:8s} = {s[p].mean():11.4f} +- {s[p].std():.4f}")
    print(f"\nsaved -> {OUT}")
    print("RUN1000SN DONE")
