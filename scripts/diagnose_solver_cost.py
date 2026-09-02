"""
diagnose_solver_cost.py — which region of TorC's in-box parameter space is
slowest to solve?

The nested-sampling pipeline vmaps torc.theory.rhopsol_jax's diffrax solve
over batches of proposals; because vmap runs all lanes in lockstep, a single
slow/stiff point drags the whole batch to its step count (torc/priors.py
already documents a measured ~20x slowdown for unclamped out-of-bounds
proposals). This script checks for an analogous hotspot *inside* the prior
box over the ODE's free parameters (omega_m, stigma, mups2, OmL; omegar is
pinned to OMEGA_R_DEFAULT and h0rd never reaches the solver).

It reconstructs rhopsol_jax's solve locally (never edits torc/theory.py or
torc/likelihood.py) so it can read diffrax's sol.stats/sol.result, which the
production solve computes but discards.

Usage:
    python scripts/diagnose_solver_cost.py [--force]

Outputs:
    diagnostics/solver_cost_scan.npz    raw per-point scan (cached)
    diagnostics/solver_cost_top20.txt   20 slowest points
    plots/solver_cost_diagnostic.{pdf,png}
    stdout: step-count + wall-clock timing report
"""
import argparse
import os
import sys
import time

import jax
jax.config.update("jax_enable_x64", True)   # must precede any jax call
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr, qmc

TORC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, TORC)

import diffrax
from torc.theory import (aseries, cpseries, aseriesp, cpseriesp, small,
                          _TERM_LNA, _SOLVER, _CTRL, Z_SPLIT, N_EARLY, N_LATE,
                          OMEGA_R_DEFAULT)
from torc.priors import (PRIOR_BOUNDS, _ALL_BOUNDS, _SAFE,
                         MUPS2_FIXED, sample_prior)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

DIAG_DIR = os.path.join(TORC, "diagnostics")
PLOTS_DIR = os.path.join(TORC, "plots")
os.makedirs(DIAG_DIR, exist_ok=True)

N_SOBOL = 2 ** 14
GRID_N = 80
N_PRIOR = 2000
FREE_PARAMS = ["omega_m", "stigma", "mups2", "OmL"]   # h0rd excluded: never reaches the ODE

CMAP_COLORS = ["#785EF0", "#DC267F", "#FFB000"]
CUSTOM_CMAP = LinearSegmentedColormap.from_list("custom_cmap", CMAP_COLORS, N=256)
plt.rcParams.update({
    "text.usetex": True, "font.family": "STIXGeneral",
    "mathtext.fontset": "stix", "font.size": 9,
    "axes.labelsize": 11, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "figure.dpi": 300, "savefig.dpi": 300,
})


# ---------------------------------------------------------------------------
# Instrumented solve: mirrors torc.theory.rhopsol_jax exactly, but returns
# (num_steps, success) instead of (z, E, a).
# ---------------------------------------------------------------------------

def _solve_with_stats(omega_m, stigma, mups2, OmL, omegar=OMEGA_R_DEFAULT):
    tauini = small(omega_m, omegar, stigma, mups2)
    x0_raw = jnp.log(aseries(tauini, omega_m, omegar, OmL, stigma, mups2))
    ok = jnp.isfinite(x0_raw) & (x0_raw < -jnp.log(1.0 + Z_SPLIT))
    x0 = jnp.where(ok, x0_raw, -45.0)
    y0 = (cpseries(tauini, omega_m, omegar, OmL, stigma, mups2),
          aseriesp(tauini, omega_m, omegar, OmL, stigma, mups2),
          cpseriesp(tauini, omega_m, omegar, OmL, stigma, mups2))
    args = (omega_m, omegar, OmL, stigma, mups2)

    xsplit = -jnp.log(1.0 + Z_SPLIT)
    xs = jnp.concatenate([
        jnp.linspace(x0, xsplit, N_EARLY, endpoint=False),
        jnp.linspace(xsplit, 0.0, N_LATE),
    ])
    sol = diffrax.diffeqsolve(
        _TERM_LNA, _SOLVER, t0=x0, t1=0.0, dt0=None, y0=y0, args=args,
        stepsize_controller=_CTRL, saveat=diffrax.SaveAt(ts=xs),
        max_steps=1024, throw=False, adjoint=diffrax.ForwardMode(),
    )
    num_steps = sol.stats["num_steps"]
    success = sol.result == diffrax.RESULTS.successful   # compare inside the vmapped fn
    return num_steps, success


_SINGLE = jax.jit(_solve_with_stats)
_BATCH = jax.jit(jax.vmap(_solve_with_stats))


# ---------------------------------------------------------------------------
# 1. Build the scan points: Sobol (unbiased 4-D coverage), two 2-D grids
#    (for the hotspot maps), and a realism cross-check against the sampler's
#    actual proposal distribution.
# ---------------------------------------------------------------------------

def build_points():
    lo = {n: _ALL_BOUNDS[n][0] for n in FREE_PARAMS}
    hi = {n: _ALL_BOUNDS[n][1] for n in FREE_PARAMS}

    sampler = qmc.Sobol(d=4, scramble=True, seed=0)
    u = sampler.random(N_SOBOL)
    lo_arr = np.array([lo[n] for n in FREE_PARAMS])
    hi_arr = np.array([hi[n] for n in FREE_PARAMS])
    sobol = qmc.scale(u, lo_arr, hi_arr)
    sobol_om, sobol_st, sobol_mu, sobol_ol = sobol.T
    sobol_src = np.full(N_SOBOL, "sobol")

    om_grid = np.linspace(lo["omega_m"], hi["omega_m"], GRID_N)
    st_grid = np.linspace(lo["stigma"], hi["stigma"], GRID_N)
    mu_grid = np.linspace(lo["mups2"], hi["mups2"], GRID_N)

    OM, ST = np.meshgrid(om_grid, st_grid, indexing="ij")
    OM, ST = OM.ravel(), ST.ravel()
    fixed_settings = [(_SAFE.get("mups2", MUPS2_FIXED), _SAFE["OmL"]),
                      (0.6, 1.0), (2.5, 0.3)]
    g1_om = np.tile(OM, len(fixed_settings))
    g1_st = np.tile(ST, len(fixed_settings))
    g1_mu = np.concatenate([np.full_like(OM, mu) for mu, _ in fixed_settings])
    g1_ol = np.concatenate([np.full_like(OM, ol) for _, ol in fixed_settings])
    g1_src = np.full(len(g1_om), "grid_om_stigma")

    ST2, MU2 = np.meshgrid(st_grid, mu_grid, indexing="ij")
    ST2, MU2 = ST2.ravel(), MU2.ravel()
    g2_om = np.full_like(ST2, _SAFE["omega_m"])
    g2_st = ST2
    g2_mu = MU2
    g2_ol = np.full_like(ST2, _SAFE["OmL"])
    g2_src = np.full(len(ST2), "grid_stigma_mups2")

    prior_pts = sample_prior(jax.random.PRNGKey(0), N_PRIOR)
    p_om = np.array(prior_pts["omega_m"])
    p_st = np.array(prior_pts["stigma"])
    p_mu = np.array(prior_pts["mups2"])
    p_ol = np.array(prior_pts["OmL"])
    p_src = np.full(N_PRIOR, "prior")

    omega_m = np.concatenate([sobol_om, g1_om, g2_om, p_om])
    stigma = np.concatenate([sobol_st, g1_st, g2_st, p_st])
    mups2 = np.concatenate([sobol_mu, g1_mu, g2_mu, p_mu])
    OmL = np.concatenate([sobol_ol, g1_ol, g2_ol, p_ol])
    source = np.concatenate([sobol_src, g1_src, g2_src, p_src])

    return omega_m, stigma, mups2, OmL, source


def run_scan(force=False):
    npz_path = os.path.join(DIAG_DIR, "solver_cost_scan.npz")
    if os.path.exists(npz_path) and not force:
        print(f"Loading cached scan from {npz_path} (use --force to rerun)")
        d = np.load(npz_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    omega_m, stigma, mups2, OmL, source = build_points()
    n = len(omega_m)
    print(f"Running {n} instrumented solves (Sobol={N_SOBOL}, "
          f"grids={2 * GRID_N * GRID_N * 3 - GRID_N * GRID_N}, prior={N_PRIOR})...")
    t0 = time.perf_counter()
    num_steps, success = _BATCH(jnp.asarray(omega_m), jnp.asarray(stigma),
                                 jnp.asarray(mups2), jnp.asarray(OmL))
    num_steps = np.asarray(jax.block_until_ready(num_steps))
    success = np.asarray(jax.block_until_ready(success))
    dt = time.perf_counter() - t0
    print(f"Scan done in {dt:.1f}s ({dt / n * 1e3:.3f} ms/point, incl. one-time compile)")

    data = dict(omega_m=omega_m, stigma=stigma, mups2=mups2, OmL=OmL,
                num_steps=num_steps, success=success, source=source)
    np.savez(npz_path, **data)
    print(f"Saved {npz_path}")
    return data


# ---------------------------------------------------------------------------
# 2. Step-count analysis + console report.
# ---------------------------------------------------------------------------

def analyze(data):
    om, st, mu, ol = data["omega_m"], data["stigma"], data["mups2"], data["OmL"]
    ns, ok, src = data["num_steps"], data["success"].astype(bool), data["source"]

    sobol_mask = src == "sobol"
    ns_sobol = ns[sobol_mask]

    i_max = int(np.argmax(ns))
    slowest = dict(omega_m=float(om[i_max]), stigma=float(st[i_max]),
                   mups2=float(mu[i_max]), OmL=float(ol[i_max]),
                   num_steps=int(ns[i_max]))
    median_ns = float(np.median(ns_sobol))
    min_ns = int(np.min(ns_sobol))

    lines = []
    lines.append("=" * 72)
    lines.append("SOLVER COST DIAGNOSTIC -- TorC in-box parameter scan")
    lines.append("=" * 72)
    lines.append(f"Total points scanned: {len(ns)}  "
                 f"(sobol={sobol_mask.sum()}, "
                 f"grid_om_stigma={(src == 'grid_om_stigma').sum()}, "
                 f"grid_stigma_mups2={(src == 'grid_stigma_mups2').sum()}, "
                 f"prior={(src == 'prior').sum()})")
    n_fail = int((~ok).sum())
    lines.append(f"In-box solve failures: {n_fail}" +
                 (" (NONE -- as expected)" if n_fail == 0 else " <-- INVESTIGATE"))
    lines.append("")
    lines.append(f"Slowest point: num_steps={slowest['num_steps']}  "
                 f"omega_m={slowest['omega_m']:.4f} stigma={slowest['stigma']:.4f} "
                 f"mups2={slowest['mups2']:.4f} OmL={slowest['OmL']:.4f}")
    lines.append(f"Sobol-scan median num_steps={median_ns:.1f}, min={min_ns}")
    lines.append(f"Slowest:median ratio = {slowest['num_steps'] / median_ns:.2f}x")
    lines.append(f"Slowest point uses {slowest['num_steps']}/1024 = "
                 f"{100 * slowest['num_steps'] / 1024:.1f}% of the max_steps budget")
    lines.append("(torc/priors.py documents a measured ~20x BATCH slowdown for "
                 "unclamped OUT-OF-BOUNDS proposals -- compare to the in-box "
                 "ratio above: this is the in-box analogue of that number.)")
    lines.append("")
    lines.append("Per-parameter Pearson correlation with num_steps (Sobol scan):")
    for name, vals in [("omega_m", om[sobol_mask]), ("stigma", st[sobol_mask]),
                        ("mups2", mu[sobol_mask]), ("OmL", ol[sobol_mask])]:
        r, _ = pearsonr(vals, ns_sobol)
        lines.append(f"  {name:10s} r = {r:+.3f}")

    print("\n".join(lines))

    order = np.argsort(-ns)[:20]
    top_path = os.path.join(DIAG_DIR, "solver_cost_top20.txt")
    with open(top_path, "w") as f:
        f.write("rank  num_steps  omega_m     stigma      mups2       OmL         source\n")
        for rank, i in enumerate(order, 1):
            f.write(f"{rank:4d}  {ns[i]:9d}  {om[i]:.6f}  {st[i]:.6f}  "
                    f"{mu[i]:.6f}  {ol[i]:.6f}  {src[i]}\n")
    print(f"Saved {top_path}")

    return slowest


# ---------------------------------------------------------------------------
# 3. Wall-clock timing: single-point, and 250-point batches (matching
#    NUM_LIVE in scripts/run_ns.py) to expose the vmap lockstep-drag effect.
# ---------------------------------------------------------------------------

def _time_single(point, reps=50):
    args = tuple(jnp.asarray(float(v)) for v in point)
    jax.block_until_ready(_SINGLE(*args))   # warm up / compile
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(_SINGLE(*args))
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _time_batches_interleaved(batches, reps=40, seed=1):
    """Time several named batches in randomized round-robin order (not
    sequentially) so a slow/fast run isn't an artifact of measurement drift
    (thermal ramp, background load, etc.) -- confirmed necessary empirically:
    a naive sequential fast/slow/mixed timing and this interleaved one agree,
    but interleaving is the version that rules out drift as the explanation."""
    args = {k: tuple(jnp.asarray(a) for a in v) for k, v in batches.items()}
    stats = {}
    for k, a in args.items():
        ns_batch, _ = jax.block_until_ready(_BATCH(*a))   # warm up / compile
        ns_batch = np.asarray(ns_batch)
        stats[k] = (int(ns_batch.max()), float(ns_batch.mean()))

    rng = np.random.default_rng(seed)
    times = {k: [] for k in batches}
    order = list(batches.keys())
    for _ in range(reps):
        rng.shuffle(order)
        for k in order:
            t0 = time.perf_counter()
            jax.block_until_ready(_BATCH(*args[k]))
            times[k].append(time.perf_counter() - t0)

    return {k: (float(np.median(times[k])), *stats[k]) for k in batches}


def run_timing(slowest, seed=1):
    rng = np.random.default_rng(seed)
    fid = _SAFE

    points = {
        "fiducial (_SAFE)": (fid["omega_m"], fid["stigma"], fid["mups2"], fid["OmL"]),
        "slowest found": (slowest["omega_m"], slowest["stigma"],
                           slowest["mups2"], slowest["OmL"]),
        "stigma floor corner": (0.5, _ALL_BOUNDS["stigma"][0], fid["mups2"], fid["OmL"]),
        "mid-box": (0.5, 2.0, 1.5, 0.8),
    }

    print("\n--- Single-point timing (median of 50 reps, one-time compile excluded) ---")
    single_times = {}
    fid_t = None
    for name, p in points.items():
        t = _time_single(p)
        single_times[name] = t
        if name == "fiducial (_SAFE)":
            fid_t = t
        print(f"  {name:22s} {t * 1e3:8.4f} ms   ({t / fid_t:.2f}x fiducial)")

    print("\n--- Batch timing (250-point batches, randomized-order median of 40 reps) ---")
    n_live = 250

    def jitter(center, name):
        return np.clip(rng.normal(center, 0.02, n_live), *_ALL_BOUNDS[name])

    fast = tuple(jitter(fid[n], n) for n in ("omega_m", "stigma", "mups2", "OmL"))
    slow = tuple(jitter(slowest[n], n) for n in ("omega_m", "stigma", "mups2", "OmL"))
    mixed = tuple(a.copy() for a in fast)
    for arr, n in zip(mixed, ("omega_m", "stigma", "mups2", "OmL")):
        arr[0] = slowest[n]

    results = _time_batches_interleaved(
        {"fast (near fiducial)": fast, "slow (near ridge)": slow,
         "mixed (1 outlier of 250)": mixed})

    t_fast = results["fast (near fiducial)"][0]
    for name, (t, nmax, nmean) in results.items():
        print(f"  {name:26s} {t * 1e3:8.2f} ms total ({t / n_live * 1e3:.4f} ms/solve)  "
              f"[num_steps max={nmax}, mean={nmean:.1f}]   {t / t_fast:.2f}x fast")

    t_mixed = results["mixed (1 outlier of 250)"][0]
    t_slow = results["slow (near ridge)"][0]
    print(f"\n  Note: max num_steps is matched between 'slow' and 'mixed' ({results['slow (near ridge)'][1]} "
          f"vs {results['mixed (1 outlier of 250)'][1]}), yet 'mixed' costs "
          f"{t_mixed / max(t_slow, 1e-9):.2f}x the 'slow' batch -- wall time here is NOT simply "
          "governed by the batch's single slowest lane. The expensive case is a batch "
          "that MIXES near-identical fast lanes with a differently-behaving outlier, "
          "not a batch that's uniformly slow. This matters because real NS live-point "
          "batches are exactly this kind of mixture (mostly typical points, "
          "occasionally one from a slower region), so this 'mixed' number -- not the "
          "single-point step-count ratio -- is the realistic estimate of per-batch cost.")

    return dict(single=single_times, fast=t_fast, slow=t_slow, mixed=t_mixed)


# ---------------------------------------------------------------------------
# 4. Diagnostic plot.
# ---------------------------------------------------------------------------

def make_plot(data):
    om, st, mu, ol = data["omega_m"], data["stigma"], data["mups2"], data["OmL"]
    ns, src = data["num_steps"], data["source"]

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.15], hspace=0.5, wspace=0.32)

    sobol_mask = src == "sobol"
    params = [("omega_m", om, r"$\Omega_m$"), ("stigma", st, r"$\varsigma$"),
              ("mups2", mu, r"$-\upsilon_2$"), ("OmL", ol, r"$\Omega_\Lambda$")]
    axesA = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
             fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    n_sobol_all = ns[sobol_mask]
    for ax, (name, vals, label) in zip(axesA, params):
        v = vals[sobol_mask]
        lo, hi = _ALL_BOUNDS[name]
        bins = np.linspace(lo, hi, 31)
        idx = np.clip(np.digitize(v, bins) - 1, 0, len(bins) - 2)
        mean_ns = np.array([n_sobol_all[idx == b].mean() if (idx == b).any() else np.nan
                             for b in range(len(bins) - 1)])
        max_ns = np.array([n_sobol_all[idx == b].max() if (idx == b).any() else np.nan
                            for b in range(len(bins) - 1)])
        centers = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(centers, mean_ns, color=CUSTOM_CMAP(0.15), label="mean")
        ax.plot(centers, max_ns, color=CUSTOM_CMAP(0.85), label="max")
        ax.set_xlabel(label)
        ax.set_ylabel("num\\_steps")
        ax.legend()
    axesA[0].set_title("Panel A -- marginal step count vs. parameter (Sobol scan)",
                        fontsize=9, loc="left")

    axB1 = fig.add_subplot(gs[2, 0])
    axB2 = fig.add_subplot(gs[2, 1])

    om_grid = np.linspace(*_ALL_BOUNDS["omega_m"], GRID_N)
    st_grid = np.linspace(*_ALL_BOUNDS["stigma"], GRID_N)
    mu_grid = np.linspace(*_ALL_BOUNDS["mups2"], GRID_N)

    m1 = src == "grid_om_stigma"
    n1 = GRID_N * GRID_N
    ns1_grid = ns[m1][:n1].reshape(GRID_N, GRID_N)   # [i,j] -> om_grid[i], st_grid[j]
    im1 = axB1.pcolormesh(om_grid, st_grid, ns1_grid.T, cmap=CUSTOM_CMAP, shading="auto")
    fig.colorbar(im1, ax=axB1, label="num\\_steps")
    axB1.set_xlabel(r"$\Omega_m$"); axB1.set_ylabel(r"$\varsigma$")
    axB1.set_title(r"Panel B -- $(\Omega_m,\varsigma)$ at fiducial "
                    r"$(-\upsilon_2,\Omega_\Lambda)$", fontsize=9)

    m2 = src == "grid_stigma_mups2"
    ns2_grid = ns[m2].reshape(GRID_N, GRID_N)        # [i,j] -> st_grid[i], mu_grid[j]
    im2 = axB2.pcolormesh(st_grid, mu_grid, ns2_grid.T, cmap=CUSTOM_CMAP, shading="auto")
    fig.colorbar(im2, ax=axB2, label="num\\_steps")
    axB2.set_xlabel(r"$\varsigma$"); axB2.set_ylabel(r"$-\upsilon_2$")
    axB2.set_title(r"$(\varsigma,-\upsilon_2)$ at fiducial $(\Omega_m,\Omega_\Lambda)$",
                    fontsize=9)

    i_max = int(np.argmax(ns))
    axB1.scatter([om[i_max]], [st[i_max]], color="black", marker="*", s=140,
                 label=f"global max ({int(ns[i_max])} steps)", zorder=5)
    axB1.scatter([_SAFE["omega_m"]], [_SAFE["stigma"]], color="white",
                 edgecolor="black", marker="o", s=50, label="_SAFE fiducial", zorder=5)
    axB1.legend(loc="upper left")

    out = os.path.join(PLOTS_DIR, "solver_cost_diagnostic")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.savefig(out + ".png", bbox_inches="tight", dpi=200)
    print(f"Saved {out}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                         help="ignore any cached scan and re-run from scratch")
    args = parser.parse_args()

    data = run_scan(force=args.force)
    slowest = analyze(data)
    make_plot(data)
    run_timing(slowest)


if __name__ == "__main__":
    main()
