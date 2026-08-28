"""
BlackJAX Nested Slice Sampling for CPL (w0waCDM) vs DESI DR2 BAO.

Companion to `Stigma/python/stigma_nss.py`: same data, same likelihood conventions, same
sampler settings, but the dark-energy sector is the CPL parametrisation

    w(a) = w0 + wa (1 - a)   <=>   w(z) = w0 + wa z / (1 + z)

whose density evolution is analytic,

    f_DE(z) = rho_DE(z)/rho_DE(0)
            = (1+z)^{3(1 + w0 + wa)} exp(-3 wa z / (1+z)),

so no ODE solve is needed and the likelihood is ~microseconds rather than
~milliseconds.  Because the priors on the shared parameters (h0rd, omega_m)
match `stigma_nss.py`, the returned log-evidences are directly comparable and
give a Bayes factor between CPL and the Stigma Q-torsion model.

Models
------
    cpl   : flat w0waCDM, 4 params  (h0rd, omega_m, w0, wa)
    lcdm  : flat LCDM,    2 params  (h0rd, omega_m)   [w0 = -1, wa = 0]

Parameters sampled
------------------
  h0rd    : H0 * r_d  [km/s]   - BAO scale        (Uniform[3000, 20000])
  omega_m : Omega_m            - matter density   (Uniform[0.1, 0.6])
  w0      : w(z=0)             -                  (Uniform[-3, 1])
  wa      : -dw/da             -                  (Uniform[-3, 2])
Hard prior constraint (as in ../../common.py): w0 + wa < 0, i.e. the
equation of state is in the accelerating regime at early times.

Radiation Omega_r = 8.24e-5 is held fixed, matching equations_jax.OMEGA_R.

Usage
-----
    source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
    python cpl_nss.py --test                 # single likelihood + chi2 check
    python cpl_nss.py --model cpl  --live 500
    python cpl_nss.py --model lcdm --live 500
    python cpl_nss.py --plot                 # corner plot + logZ table
    python cpl_nss.py --model cpl --mock --mock-w0 -0.9 --mock-wa -0.5

`--mock` replaces the measured DESI DR2 data vector by a synthetic realisation
drawn from a CPL fiducial with the real DESI DR2 covariance, for pipeline
validation / forecasting.  The covariance (and hence the error bars) is always
the real one.

Output
------
    output/cpl_desi_dead.pkl / cpl_desi_live.pkl      (--model cpl)
    output/lcdm_desi_dead.pkl / lcdm_desi_live.pkl    (--model lcdm)
"""

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
import tqdm

jax.config.update("jax_enable_x64", True)

# -- Constants (kept identical to equations_jax.py so runs are comparable) ----
OMEGA_R = 8.24e-5        # radiation density parameter (fixed)
C_KMS   = 299792.458     # speed of light [km/s]

# Redshift grid for the comoving-distance integral.  1/E(z) is smooth, so
# trapezium on 4000 points over [0, 2.5] is accurate to ~1e-9 relative.
_Z_GRID = jnp.linspace(0.0, 2.5, 4000)


# -- DESI DR2 data ------------------------------------------------------------

def _find_desi_dir():
    """Locate distances/desidr2 by walking up from this file, then fall back."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    d = here
    for _ in range(6):
        d = os.path.dirname(d)
        candidates.append(os.path.join(d, "distances", "desidr2"))
    candidates.append(os.path.join(here, "..", "..", "distances", "desidr2"))
    candidates.append("/Users/starfish/Desktop/PhD2025-26/distances/desidr2")
    for c in candidates:
        if os.path.isfile(os.path.join(c, "desidr2_mean.txt")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Could not locate distances/desidr2/desidr2_mean.txt; tried:\n  "
        + "\n  ".join(candidates))


DESI_DIR = _find_desi_dir()

_rows = [l.split() for l in open(os.path.join(DESI_DIR, "desidr2_mean.txt"))
         if l.strip() and not l.lstrip().startswith("#")]
Z_DATA    = np.array([float(r[0]) for r in _rows])
MEAN_DATA = np.array([float(r[1]) for r in _rows])
TYPE_DATA = [r[2] for r in _rows]
COV_DATA  = np.loadtxt(os.path.join(DESI_DIR, "desidr2_cov.txt"))
N_DATA    = len(MEAN_DATA)

# Integer type codes for a branch-free JAX assembly of the theory vector:
#   0 = DM_over_rs, 1 = DH_over_rs, 2 = DV_over_rs
_CODE = {"DM_over_rs": 0, "DH_over_rs": 1, "DV_over_rs": 2}
TYPE_CODE = jnp.array([_CODE[t] for t in TYPE_DATA])

_z_jax    = jnp.array(Z_DATA)
_mean_jax = jnp.array(MEAN_DATA)
_cov_jax  = jnp.array(COV_DATA)
_cov_inv  = jnp.linalg.inv(_cov_jax)
_sign, _log_det = jnp.linalg.slogdet(_cov_jax)
_norm     = -0.5 * (N_DATA * jnp.log(2 * jnp.pi) + _log_det)


# -- CPL background -----------------------------------------------------------

def f_de(z, w0, wa):
    """rho_DE(z) / rho_DE(0) for w(a) = w0 + wa (1 - a)."""
    return (1.0 + z) ** (3.0 * (1.0 + w0 + wa)) * jnp.exp(-3.0 * wa * z / (1.0 + z))


def E_cpl(z, omega_m, w0, wa, omega_r=OMEGA_R):
    """H(z) / H0 for flat w0waCDM."""
    om_de = 1.0 - omega_m - omega_r
    return jnp.sqrt(omega_m * (1.0 + z) ** 3
                    + omega_r * (1.0 + z) ** 4
                    + om_de * f_de(z, w0, wa))


def desi_distances_cpl(h0rd, omega_m, w0, wa):
    """
    DESI DR2 BAO observables for flat CPL, in the row order of desidr2_mean.txt.

        DH(z)/rd = c / (h0rd * E(z))
        DM(z)/rd = (c / h0rd) * int_0^z dz'/E(z')
        DV(z)/rd = (z * (DM/rd)^2 * (DH/rd))^(1/3)

    h0rd = H0 * rd in km/s, so c/h0rd is dimensionless.
    """
    inv_E = 1.0 / E_cpl(_Z_GRID, omega_m, w0, wa)

    dz    = jnp.diff(_Z_GRID)
    trap  = 0.5 * (inv_E[:-1] + inv_E[1:]) * dz
    cum   = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(trap)])

    integral   = jnp.interp(_z_jax, _Z_GRID, cum)
    DM_over_rd = C_KMS / h0rd * integral
    DH_over_rd = C_KMS / (h0rd * E_cpl(_z_jax, omega_m, w0, wa))
    DV_over_rd = (_z_jax * DM_over_rd ** 2 * DH_over_rd) ** (1.0 / 3.0)

    return jnp.where(TYPE_CODE == 0, DM_over_rd,
                     jnp.where(TYPE_CODE == 1, DH_over_rd, DV_over_rd))


# -- Priors -------------------------------------------------------------------

# (loc, scale) for jax.scipy.stats.uniform, i.e. support [loc, loc+scale].
#
# Two prior boxes are available, because log Z is only comparable between runs
# that share one.  `--prior-box torc` reproduces the box used by the TorC runs
# in ../torc/priors.py, so that CPL can be compared directly
# against the TorC 1/stigma^2 evidence (and so that LCDM here reproduces the
# TorC LCDM value of -7.734 +/- 0.094 as a cross-pipeline validation).
PRIOR_BOXES = {
    "default": {
        "h0rd":    (3000.0, 17000.0),   # [3000, 20000]  - matches stigma_nss.py
        "omega_m": (0.1,    0.5),       # [0.1,  0.6]    - matches stigma_nss.py
        "w0":      (-3.0,   4.0),       # [-3,   1]      - matches ../../common.py
        "wa":      (-3.0,   5.0),       # [-3,   2]      - matches ../../common.py
    },
    "torc": {
        "h0rd":    (3650.0, 14600.0),   # [3650, 18250]  - TorC priors.py
        "omega_m": (0.01,   0.98),      # [0.01, 0.99]   - TorC priors.py
        "w0":      (-3.0,   4.0),       # [-3,   1]      - unchanged
        "wa":      (-3.0,   5.0),       # [-3,   2]      - unchanged
    },
}

# Mutable module-level box, switched by set_prior_box().
PRIOR_RANGE = dict(PRIOR_BOXES["default"])
PRIOR_BOX = "default"


def set_prior_box(name):
    """Select a prior box; affects priors, sampling and output file tags."""
    global PRIOR_RANGE, PRIOR_BOX
    if name not in PRIOR_BOXES:
        raise ValueError(f"unknown prior box {name!r}; "
                         f"choose from {sorted(PRIOR_BOXES)}")
    PRIOR_RANGE = dict(PRIOR_BOXES[name])
    PRIOR_BOX = name
    return PRIOR_RANGE


def chain_tag(model, box=None):
    """File stem for a chain: `<model>_desi`, or `<model>_torcbox_desi`."""
    box = PRIOR_BOX if box is None else box
    return f"{model}_desi" if box == "default" else f"{model}_{box}box_desi"

PARAMS = {
    "cpl":  ["h0rd", "omega_m", "w0", "wa"],
    "lcdm": ["h0rd", "omega_m"],
}


def _unpack(params, model):
    """Return (h0rd, omega_m, w0, wa), fixing w0=-1, wa=0 for LCDM."""
    h0rd    = params["h0rd"]
    omega_m = params["omega_m"]
    if model == "cpl":
        return h0rd, omega_m, params["w0"], params["wa"]
    return h0rd, omega_m, jnp.array(-1.0), jnp.array(0.0)


def make_logprior(model):
    names = PARAMS[model]

    def logprior_fn(params):
        lp = jnp.array(0.0)
        for n in names:
            loc, scale = PRIOR_RANGE[n]
            lp += jax.scipy.stats.uniform.logpdf(params[n], loc, scale)
        if model == "cpl":
            # w0 + wa < 0, as imposed in ../../common.py
            lp = jnp.where(params["w0"] + params["wa"] < 0.0, lp, -jnp.inf)
        # Dark energy must not be a negative fraction of the budget.
        lp = jnp.where(params["omega_m"] + OMEGA_R < 1.0, lp, -jnp.inf)
        return lp

    return logprior_fn


def make_loglikelihood(model, mean_vector=None):
    """
    Gaussian DESI DR2 BAO log-likelihood.

    As in stigma_nss.py the prior is re-applied as a hard gate: an out-of-prior
    particle with a finite logL would be picked as a slice-sampling start point
    with logslice = -inf, accepting any proposal and letting the chain drift.

    mean_vector : override the measured data (used by --mock).
    """
    logprior_fn = make_logprior(model)
    mu = _mean_jax if mean_vector is None else jnp.asarray(mean_vector)

    def loglikelihood_fn(params):
        lp = logprior_fn(params)
        h0rd, omega_m, w0, wa = _unpack(params, model)

        theory   = desi_distances_cpl(h0rd, omega_m, w0, wa)
        residual = theory - mu
        chi2     = residual @ _cov_inv @ residual
        logl     = -0.5 * chi2 + _norm

        logl = jnp.where(jnp.isfinite(logl), logl, jnp.array(-1e30))
        return jnp.where(jnp.isfinite(lp), logl, jnp.array(-1e30))

    return loglikelihood_fn


# -- Mock data ----------------------------------------------------------------

def make_mock_data(h0rd, omega_m, w0, wa, seed=0, noiseless=False):
    """Synthetic DESI DR2 data vector from a CPL fiducial + real DR2 covariance."""
    theory = np.array(desi_distances_cpl(h0rd, omega_m, w0, wa))
    if noiseless:
        return theory
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(theory, COV_DATA)


# -- NSS driver ---------------------------------------------------------------

def run_nss(model="cpl", num_live=500, seed=0, output_dir="output",
            num_inner_steps=None, max_slice_steps=5, max_shrinkage=20,
            mock=None):
    """
    Run nested slice sampling.  Settings mirror stigma_nss.py so the evidences
    are computed under the same sampler configuration.
    """
    os.makedirs(output_dir, exist_ok=True)
    names    = PARAMS[model]
    num_dims = len(names)
    num_delete = max(1, num_live // 10)
    if num_inner_steps is None:
        num_inner_steps = num_dims * 5   # 5*ndim: publication setting

    print(f"model={model}  ndim={num_dims}  num_live={num_live}  "
          f"num_delete={num_delete}  num_inner_steps={num_inner_steps}  "
          f"prior_box={PRIOR_BOX}")
    print(f"DESI DR2 data: {DESI_DIR}  ({N_DATA} points)"
          + ("  [MOCK]" if mock is not None else ""))

    logprior_fn = make_logprior(model)
    loglike_fn  = make_loglikelihood(model, mean_vector=mock)

    algo = blackjax.nss(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglike_fn,
        num_delete=num_delete,
        num_inner_steps=num_inner_steps,
        max_steps=max_slice_steps,
        max_shrinkage=max_shrinkage,
    )

    # -- Initialise live points inside the hard prior boundary ---------------
    rng_key = jax.random.PRNGKey(seed)
    rng_key, init_key = jax.random.split(rng_key)
    particles = _sample_prior(init_key, model, num_live)
    print("Initialising live points ...")
    live = jax.jit(algo.init)(particles)
    print(f"  logZ = {live.logZ:.3f},  logZ_live = {live.logZ_live:.3f}")

    step_fn = jax.jit(algo.step)
    dead = []
    with tqdm.tqdm(desc="Dead points", unit=" pts") as pbar:
        while not (live.logZ_live - live.logZ < -3):
            rng_key, subkey = jax.random.split(rng_key)
            live, dead_info = step_fn(subkey, live)
            dead.append(dead_info)
            pbar.update(num_delete)

    dead = blackjax.ns.utils.finalise(live, dead)
    print(f"\nDone.  logZ = {live.logZ:.4f} "
          f"+/- {jnp.exp(live.logZ_live - live.logZ):.4f}")

    tag = chain_tag(model)
    with open(os.path.join(output_dir, f"{tag}_dead.pkl"), "wb") as f:
        pickle.dump(dead, f)
    with open(os.path.join(output_dir, f"{tag}_live.pkl"), "wb") as f:
        pickle.dump(live, f)
    print(f"Saved to {output_dir}/{tag}_dead.pkl")
    return live, dead


def _sample_prior(rng_key, model, num_live):
    """Rejection-sample live points satisfying the hard prior constraints."""
    names = PARAMS[model]
    kept = {n: [] for n in names}
    n_found, batch = 0, 0
    while n_found < num_live:
        keys = jax.random.split(rng_key, len(names) + 1)
        rng_key, keys = keys[0], keys[1:]
        n_try = max(num_live * 4, 800)
        draw = {}
        for n, k in zip(names, keys):
            loc, scale = PRIOR_RANGE[n]
            draw[n] = np.array(jax.random.uniform(
                k, (n_try,), minval=loc, maxval=loc + scale))
        ok = np.ones(n_try, dtype=bool)
        if model == "cpl":
            ok &= (draw["w0"] + draw["wa"]) < 0.0
        ok &= (draw["omega_m"] + float(OMEGA_R)) < 1.0
        for n in names:
            kept[n].extend(draw[n][ok].tolist())
        n_found = len(kept[names[0]])
        batch += 1
        if batch > 50:
            raise RuntimeError("Could not sample enough valid prior particles")
    print(f"  Found {n_found} valid prior particles in {batch} batch(es)")
    return {n: jnp.array(kept[n][:num_live]) for n in names}


# -- Analysis helpers ---------------------------------------------------------

LABELS = {
    "h0rd":    r"$H_0 r_d$ [km/s]",
    "omega_m": r"$\Omega_m$",
    "w0":      r"$w_0$",
    "wa":      r"$w_a$",
}


def load_as_nested_samples(model="cpl", output_dir="output", box=None):
    """Load saved dead points as an anesthetic NestedSamples object."""
    from anesthetic import NestedSamples
    fn = os.path.join(output_dir, f"{chain_tag(model, box)}_dead.pkl")
    with open(fn, "rb") as f:
        dead = pickle.load(f)
    return NestedSamples(
        dead.particles,
        logL=dead.loglikelihood,
        logL_birth=dead.loglikelihood_birth,
        labels={k: v for k, v in LABELS.items() if k in PARAMS[model]},
    )


def summarise(models=("cpl", "lcdm"), output_dir="output", plot=True, box=None):
    """Print posterior summaries + log-evidences; optionally make a corner plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from anesthetic import make_2d_axes

    available = [m for m in models
                 if os.path.isfile(os.path.join(
                     output_dir, f"{chain_tag(m, box)}_dead.pkl"))]
    if not available:
        raise FileNotFoundError(f"No chains found in {output_dir}/")

    samples, stats = {}, {}
    for m in available:
        s = load_as_nested_samples(m, output_dir, box)
        samples[m] = s
        st = s.stats(nsamples=200)
        stats[m] = st
        print(f"\n=== {m.upper()} ===")
        print(f"  logZ  = {st['logZ'].mean():.3f} +/- {st['logZ'].std():.3f}")
        print(f"  D_KL  = {st['D_KL'].mean():.3f} +/- {st['D_KL'].std():.3f}")
        # WeightedSeries.quantile already accounts for the nested-sampling
        # weights carried in the index, so they must not be passed again.
        for p in PARAMS[m]:
            lo, med, hi = np.asarray(s[p].quantile([0.16, 0.5, 0.84]))
            print(f"  {p:8s} = {med:10.4f}  (+{hi - med:.4f} / -{med - lo:.4f})")
        chi2_min = -2 * (float(s.logL.max()) - float(_norm))
        print(f"  chi2_min = {chi2_min:.3f}  (N_data = {N_DATA})")

    if len(available) == 2:
        a, b = available
        dlogZ = stats[a]['logZ'] - stats[b]['logZ']
        print(f"\nlog B({a}/{b}) = {dlogZ.mean():.3f} +/- {dlogZ.std():.3f}")

    if plot:
        params = PARAMS["cpl"] if "cpl" in available else PARAMS[available[0]]
        fig, axes = make_2d_axes(params, figsize=(7.5, 7.5), facecolor="w")
        colors = {"cpl": "C0", "lcdm": "C1"}
        for m in available:
            samples[m].plot_2d(axes, alpha=0.8, label=m.upper(),
                               color=colors.get(m, None))
        # Collect the legend handles from a panel every model appears in:
        # LCDM only spans (h0rd, omega_m), so the bottom-left panel misses it.
        handles, labels = axes.iloc[1, 0].get_legend_handles_labels()
        axes.iloc[-1, 0].legend(handles, labels,
                                bbox_to_anchor=(len(axes) / 2, len(axes)),
                                loc="lower center", ncols=len(available))
        suffix = "" if (box or PRIOR_BOX) == "default" else f"_{box or PRIOR_BOX}box"
        out = os.path.join(output_dir, f"cpl_desi_corner{suffix}.pdf")
        fig.savefig(out, bbox_inches="tight")
        print(f"\nSaved corner plot to {out}")
        plt.close(fig)

    return samples, stats


# -- Quick test ---------------------------------------------------------------

def test_likelihood(model="cpl", h0rd=10145.0, omega_m=0.30, w0=-1.0, wa=0.0):
    """Single likelihood evaluation + timing, and a chi2 sanity check."""
    import time
    params = {"h0rd": jnp.array(h0rd), "omega_m": jnp.array(omega_m)}
    if model == "cpl":
        params["w0"] = jnp.array(w0)
        params["wa"] = jnp.array(wa)

    logprior_fn = make_logprior(model)
    loglike_fn  = jax.jit(make_loglikelihood(model))
    print(f"DESI DR2 dir : {DESI_DIR}")
    print(f"log prior    : {float(logprior_fn(params)):.4f}")

    t0 = time.time(); logl = loglike_fn(params); logl.block_until_ready()
    print(f"log L = {float(logl):.4f}   (first call inc. JIT: {time.time() - t0:.2f}s)")
    t0 = time.time()
    for _ in range(100):
        logl = loglike_fn(params)
    logl.block_until_ready()
    print(f"log L = {float(logl):.4f}   "
          f"(cached: {(time.time() - t0) / 100 * 1e6:.1f} us/call)")

    theory = np.array(desi_distances_cpl(h0rd, omega_m, w0, wa))
    resid  = theory - MEAN_DATA
    chi2   = resid @ np.linalg.inv(COV_DATA) @ resid
    print(f"\nchi2 = {chi2:.3f} for {N_DATA} data points")
    print(f"{'z':>7} {'type':>12} {'data':>9} {'theory':>9} {'sigma':>7} {'pull':>7}")
    sig = np.sqrt(np.diag(COV_DATA))
    for i in range(N_DATA):
        print(f"{Z_DATA[i]:7.3f} {TYPE_DATA[i]:>12} {MEAN_DATA[i]:9.3f} "
              f"{theory[i]:9.3f} {sig[i]:7.3f} {resid[i] / sig[i]:7.2f}")
    return float(logl)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="CPL (w0waCDM) vs DESI DR2 BAO: BlackJAX nested slice sampling")
    p.add_argument("--model", default="cpl", choices=["cpl", "lcdm"])
    p.add_argument("--test", action="store_true", help="single likelihood test")
    p.add_argument("--plot", action="store_true",
                   help="summarise saved chains + corner plot (no sampling)")
    p.add_argument("--live", type=int, default=500)
    p.add_argument("--inner", type=int, default=None,
                   help="slice moves per replacement (default 5*ndim)")
    p.add_argument("--max-slice", type=int, default=5)
    p.add_argument("--max-shrink", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="output")
    p.add_argument("--mock", action="store_true",
                   help="fit a synthetic CPL realisation instead of the real data")
    p.add_argument("--mock-w0", type=float, default=-0.9)
    p.add_argument("--mock-wa", type=float, default=-0.5)
    p.add_argument("--mock-omegam", type=float, default=0.30)
    p.add_argument("--mock-h0rd", type=float, default=10145.0)
    p.add_argument("--mock-noiseless", action="store_true")
    p.add_argument("--prior-box", default="default", choices=sorted(PRIOR_BOXES),
                   help="prior box; 'torc' matches distances/TorC/torc/priors.py")
    args = p.parse_args()

    set_prior_box(args.prior_box)

    if args.test:
        test_likelihood(args.model)
    elif args.plot:
        summarise(output_dir=args.output)
    else:
        mock = None
        if args.mock:
            mock = make_mock_data(args.mock_h0rd, args.mock_omegam,
                                  args.mock_w0, args.mock_wa,
                                  seed=args.seed, noiseless=args.mock_noiseless)
            print(f"Mock fiducial: h0rd={args.mock_h0rd}, "
                  f"omega_m={args.mock_omegam}, w0={args.mock_w0}, "
                  f"wa={args.mock_wa}, "
                  + ("noiseless" if args.mock_noiseless else f"seed={args.seed}"))
        run_nss(model=args.model, num_live=args.live, seed=args.seed,
                output_dir=args.output, num_inner_steps=args.inner,
                max_slice_steps=args.max_slice, max_shrinkage=args.max_shrink,
                mock=mock)
