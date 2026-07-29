"""Is the sampler paying a tax for proposing into 'difficult' regions?

Directly measures, per NS iteration:
  - fraction of inner-kernel slice proposals landing OUTSIDE the prior box
    (these get clamped to _SAFE, still cost a full ODE solve, return -1e30,
     and FAIL the slice constraint -> force extra shrinkage)
  - stepping-out / shrinkage counts, early vs late in the run
  - correlation of shrinkage count with each parameter's position
  - where the live points actually sit relative to the prior bounds
"""
import sys, os, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import blackjax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from torc.priors import PRIOR_BOUNDS, logprior_fn, sample_prior, loglikelihood_bounded

NUM_LIVE = 250
NUM_DELETE = 25
NUM_INNER_STEPS = 13
N_CALLS = 60
PARAMS = list(PRIOR_BOUNDS.keys())

rng_key = jax.random.PRNGKey(0)
algo = blackjax.nss(logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_bounded,
                     num_delete=NUM_DELETE, num_inner_steps=NUM_INNER_STEPS)
rng_key, init_key = jax.random.split(rng_key)
live = jax.jit(algo.init)(sample_prior(init_key, NUM_LIVE))
step_fn = jax.jit(algo.step)

print(f"{'call':>5s} {'s':>6s} {'OOB%':>6s} {'nstep':>6s} {'nshrink':>8s} {'acc%':>6s} "
      f"{'logZl-logZ':>11s}   live-point ranges (frac of prior width)", flush=True)

recs = []
t_start = time.perf_counter()
for i in range(N_CALLS):
    rng_key, subkey = jax.random.split(rng_key)
    t0 = time.perf_counter()
    live, dinfo = step_fn(subkey, live)
    jax.block_until_ready(live)
    dt = time.perf_counter() - t0

    ik = dinfo.inner_kernel_info
    pos = {k: np.asarray(v) for k, v in ik.position.items()}   # (num_delete, inner_steps)
    nstep = np.asarray(ik.info.num_steps)
    nshrink = np.asarray(ik.info.num_shrink)
    acc = np.asarray(ik.info.is_accepted)

    # fraction of ACCEPTED positions outside the box is ~0 by construction;
    # what matters is how often the sampler *probes* outside. We can only see
    # the accepted endpoint per inner step, so use it as a lower bound, and
    # additionally record how close accepted points sit to each boundary.
    oob = np.zeros_like(nstep, dtype=bool)
    for k in PARAMS:
        lo, hi = PRIOR_BOUNDS[k]
        oob |= (pos[k] < lo) | (pos[k] > hi)

    lp = {k: np.asarray(v) for k, v in live.particles.items()}
    rng_txt = " ".join(
        f"{k[:6]}:{(lp[k].min()-PRIOR_BOUNDS[k][0])/(PRIOR_BOUNDS[k][1]-PRIOR_BOUNDS[k][0]):.2f}-"
        f"{(lp[k].max()-PRIOR_BOUNDS[k][0])/(PRIOR_BOUNDS[k][1]-PRIOR_BOUNDS[k][0]):.2f}"
        for k in PARAMS)

    recs.append(dict(call=i, dt=dt, oob=oob.mean(), nstep=nstep.mean(),
                     nshrink=nshrink.mean(), acc=acc.mean(),
                     pos={k: pos[k].ravel() for k in PARAMS},
                     nshrink_flat=nshrink.ravel(), nstep_flat=nstep.ravel()))
    if i % 5 == 0 or i == N_CALLS - 1:
        print(f"{i:5d} {dt:6.2f} {100*oob.mean():6.2f} {nstep.mean():6.2f} "
              f"{nshrink.mean():8.2f} {100*acc.mean():6.1f} "
              f"{float(live.logZ_live-live.logZ):11.2f}   {rng_txt}", flush=True)

print(f"\ntotal {time.perf_counter()-t_start:.1f}s over {N_CALLS} calls")

early = recs[:10]
late = recs[-10:]
print("\n=== early (calls 0-9) vs late (last 10) ===")
for nm, grp in [("early", early), ("late", late)]:
    print(f"  {nm:6s}  s/call={np.mean([r['dt'] for r in grp]):.2f}  "
          f"num_steps={np.mean([r['nstep'] for r in grp]):.2f}  "
          f"num_shrink={np.mean([r['nshrink'] for r in grp]):.2f}  "
          f"accept%={100*np.mean([r['acc'] for r in grp]):.1f}  "
          f"OOB%={100*np.mean([r['oob'] for r in grp]):.2f}")

# correlate slice cost with position, pooled over the whole run
print("\n=== correlation of slice cost with parameter value (all proposals) ===")
allpos = {k: np.concatenate([r["pos"][k] for r in recs]) for k in PARAMS}
allshrink = np.concatenate([r["nshrink_flat"] for r in recs])
allstep = np.concatenate([r["nstep_flat"] for r in recs])
from scipy.stats import pearsonr
for k in PARAMS:
    r_sh, _ = pearsonr(allpos[k], allshrink)
    r_st, _ = pearsonr(allpos[k], allstep)
    lo, hi = PRIOR_BOUNDS[k]
    frac = (allpos[k] - lo) / (hi - lo)
    print(f"  {k:8s} r(num_shrink)={r_sh:+.3f}  r(num_steps)={r_st:+.3f}   "
          f"proposals within 5% of a bound: {100*np.mean((frac<0.05)|(frac>0.95)):.1f}%")

# how much does the expensive tail cost?
print("\n=== distribution of per-inner-step cost (likelihood calls = num_steps+num_shrink) ===")
tot = allstep + allshrink
for p in (50, 75, 90, 99, 100):
    print(f"  {p:3d}th pct: {np.percentile(tot, p):.1f} likelihood calls")
print(f"  mean: {tot.mean():.2f}   (a 'free' step would be ~2)")
frac_exp = np.mean(tot > np.percentile(tot, 90))
print(f"  share of ALL likelihood calls consumed by the worst 10% of inner steps: "
      f"{100*tot[tot>np.percentile(tot,90)].sum()/tot.sum():.1f}%")
print("DONE")
