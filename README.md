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

## Quickstart

```bash
source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
cd distances/TorC
python tests/test_theory.py            # validate the solver (~1 min)
python scripts/run_ns.py               # exploration run, ~15 min on CPU
python scripts/plot_corner.py          # corner plot -> plots/
```

Data are vendored in `data/desidr2/` (override with `$DESIDR2_DIR`).
Dependencies: jax, diffrax, optimistix, blackjax (handley-lab fork with
`nss`), anesthetic, fgivenx, scipy, pandas, matplotlib.

## Sampler notes

- Always hand `torc.priors.loglikelihood_bounded` to the sampler, not the
  raw likelihood: the slice sampler proposes outside the prior box, and an
  out-of-bounds ODE solve burns the full `max_steps` budget while vmapped
  lanes run in lockstep (~20× slowdown without the clamp).
- Exploration settings (250 live / 50 delete / 5·ndim inner steps) give
  ln Z to ±0.2; use 1000+ live points for publication numbers.
- Reference results (DESI DR2 BAO alone, **1000 live points**, Julia IPC
  backend). LCDM: ln Z = −7.739 ± 0.096. TorC depends on the ς prior:

  | ς prior | ln Z | Δ ln Z vs LCDM | Bayes factor |
  |---|---|---|---|
  | exponential λ=0.5 | −7.521 ± 0.097 | +0.22 ± 0.14 | 1.24:1 TorC |
  | uniform [1, 3]    | −8.073 ± 0.096 | −0.33 ± 0.14 | 1.40:1 LCDM |
  | exponential λ=2.0 | −8.461 ± 0.102 | −0.72 ± 0.14 | 2.06:1 LCDM |

  All three are |Δ ln Z| < 1 — inconclusive on the Jeffreys scale, a
  statistical dead heat. The sign of the Bayes factor is set by the prior,
  not by the data, so **no single number here is "the" Bayes factor**.
  ln Z falls monotonically as the prior widens: that is the Occam factor
  (the posterior lands at ς ≈ 1.3–1.4 regardless, so a wider prior only
  adds volume where the likelihood is not). See
  `scripts/plot_prior_sensitivity.py` → `plots/prior_sensitivity.pdf`.
