# TorC vs DESI DR2 BAO

JAX implementation of the TorC (stigma-field) background cosmology, with a
nested-sampling pipeline (blackjax NSS) against the DESI DR2 BAO distance
measurements, and the plotting suite used to explore it.

## Layout

```
torc/                    the package
  theory.py              canonical solver (noH0 anchoring, single ln(a) solve)
  theory_tau.py          conformal-time variant of the same anchoring (cross-check)
  scipy_reference.py     original solve_ivp equations (validation reference)
  likelihood.py          DESI DR2 BAO likelihood (13 points, full covariance)
  priors.py              prior box + bounds-clamped likelihood wrapper
scripts/                 entry points — run from the repo root
  run_ns.py              nested sampling of TorC  -> chains/torc_ns_noH0.npz
  run_lcdm.py            flat-LCDM comparison fit -> chains/lcdm_ns.npz
  plot_corner.py         corner plot of the TorC posterior
  plot_corner_vs_lcdm.py TorC vs LCDM overlay
  plot_fgivenx.py        posterior-predictive bands over the DESI data
  plot_rainbow_desi.py   parameter sweeps vs DESI residuals (3-panel rainbows)
  plot_rainbow_field.py  H0/(aH) + field-trajectory rainbows
tests/                   validation vs scipy + sampler-safety stress tests
tutorial/                step-by-step construction of the pipeline (step1..step5)
legacy/                  archived experiments: pre-JAX scripts, CAMB-based
                         rainbows, and H0_renorm/ (the calibrated anchoring)
chains/  plots/          results
theory_jax.py, test_theory_cpr1.py    back-compat shims -> torc.*
```

## Anchoring convention (important)

The canonical solver uses the **noH0** convention: a0 = 1 is pinned in the
equations and *today* is defined as the moment a = 1; E(z) = H/H(a=1).
No value of H0 is assumed anywhere — the single dimensionful quantity is
the sampled product `h0rd` = H0·r_d [km/s].

Consequences:
- `omega_m` / `OmL` are **not** today's density fractions under this
  convention (they are the fractions at the never-visited epoch where
  H_code = 100); fitted values are not directly comparable to DESI's Ωm.
- The previous convention (today located dynamically where the input
  fractions hold, via a da = 100 event) is archived in
  `legacy/H0_renorm/theory_jax.py`; `chains/torc_ns.npz` was produced with
  it. The two anchorings coincide exactly for flat parameter combinations.
- `mups2` (−υ₂) is an **exact background gauge parameter** (it rescales the
  field's internal clock, τ → √mups2·τ); E(z) is provably independent of
  it and background data cannot constrain it. It is kept in the interface
  for perturbation-level work. NOTE: the `P_deriveda0`/`rho_deriveda0`
  expressions in `torc/scipy_reference.py` are *not* mups2-invariant even
  though w = P/ρ should be — pending re-derivation; treat w_DE panels and
  anything handing that fluid to CAMB with care.

## Quickstart

```bash
source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
cd distances/TorC
python tests/test_theory.py            # validate the solver (~1 min)
python scripts/run_ns.py               # exploration run, ~15 min on CPU
python scripts/plot_corner.py          # corner plot -> plots/
```

Data are read from `../../desidr2` (override with `$DESIDR2_DIR`).
Dependencies: jax, diffrax, optimistix, blackjax (handley-lab fork with
`nss`), anesthetic, fgivenx, scipy, pandas, matplotlib.

## Sampler notes

- Always hand `torc.priors.loglikelihood_bounded` to the sampler, not the
  raw likelihood: the slice sampler proposes outside the prior box, and an
  out-of-bounds ODE solve burns the full `max_steps` budget while vmapped
  lanes run in lockstep (~20× slowdown without the clamp).
- Exploration settings (250 live / 50 delete / 5·ndim inner steps) give
  ln Z to ±0.2; use 1000+ live points for publication numbers.
- Reference results (DESI DR2 BAO alone): LCDM ln Z ≈ −7.73 ± 0.14,
  TorC-noH0 ln Z ≈ −7.86 ± 0.21 — a statistical dead heat.
