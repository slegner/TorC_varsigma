# Which file makes which plot

Provenance map for the CPL and TorC figures: every plot, the script that draws
it, the chain it reads, and the command to regenerate it.

Everything lives in the TorC repo, `distances/TorC/`:

- **`cpl/`** (this directory) — the CPL work, and the CPL↔TorC overlays.
- **`../scripts/`, `../chains/`, `../plots/`** — the existing TorC pipeline and its own
  plotting suite.

Both use the same environment:

```bash
source /Users/starfish/Desktop/PhD2025-26/env_camb_new/bin/activate
```

---

## 1. CPL — this directory (`distances/TorC/cpl/`)

### Chains

| Chain | Made by | Prior box |
|---|---|---|
| `output/cpl_desi_{dead,live}.pkl` | `cpl_nss.py --model cpl --live 500` | default (Ω_m ∈ [0.1,0.6]) |
| `output/lcdm_desi_{dead,live}.pkl` | `cpl_nss.py --model lcdm --live 500` | default |
| `output/cpl_torcbox_desi_{dead,live}.pkl` | `cpl_nss.py --model cpl --live 1000 --prior-box torc` | TorC-matched |
| `output/lcdm_torcbox_desi_{dead,live}.pkl` | `cpl_nss.py --model lcdm --live 1000 --prior-box torc` | TorC-matched |
| `output_mock/cpl_desi_dead.pkl` | `cpl_nss.py --model cpl --mock --mock-noiseless` | default |

Chain pickles are gitignored — deterministic in `--seed`, ~1 min each.

### Corner plots

| Figure | Script | Chains | Command |
|---|---|---|---|
| `output/cpl_desi_corner.pdf` | **`cpl_nss.py`** → `summarise()` | cpl + lcdm (default box) | `python cpl_nss.py --plot` |
| `output/compare_corner.pdf` | **`cpl_torc_compare.py`** → `plot_corner()` | cpl + lcdm (torc box) + TorC power-law | `python cpl_torc_compare.py --only corner` |
| `output/cpl_prior_posterior.pdf` | **`cpl_prior_posterior.py`** → `plot_corner()` | cpl (torc box), prior via `.prior()` | `python cpl_prior_posterior.py` |
| `output/cpl_prior_posterior_w.pdf` | **`cpl_prior_posterior.py`** → `plot_w_plane()` | same | `python cpl_prior_posterior.py` |

`cpl_desi_corner.pdf` is CPL vs ΛCDM over all four CPL parameters.
`compare_corner.pdf` is the three-way overlay, restricted to the parameters the
models share — (H₀r_d, Ω_m) — since CPL has (w₀, w_a) and TorC has (Ω_Λ, ς).

### fgivenx plots

| Figure | Script | Chains | Command |
|---|---|---|---|
| `output/cpl_fgivenx_wz.pdf` | **`cpl_fgivenx.py`** → `plot_wz()` | cpl (default box) | `python cpl_fgivenx.py --only wz` |
| `output/cpl_fgivenx_hz.pdf` | **`cpl_fgivenx.py`** → `plot_hz()` | cpl + lcdm (default box) | `python cpl_fgivenx.py --only hz` |
| `output/cpl_fgivenx_bao.pdf` | **`cpl_fgivenx.py`** → `plot_bao()` | cpl + lcdm (default box) | `python cpl_fgivenx.py --only bao` |
| `output/compare_distances.pdf` | **`cpl_torc_compare.py`** → `plot_distances()` | cpl + lcdm (torc box) + TorC power-law | `python cpl_torc_compare.py --only distances` |
| `output/compare_torc_convention.pdf` | **`cpl_torc_compare.py`** → `plot_torc_convention()` | same | `python cpl_torc_compare.py --only torc` |

`cpl_fgivenx.py` plots CPL alone; `cpl_torc_compare.py` overlays CPL on TorC.
`compare_distances.pdf` uses ratios to ΛCDM against z; `compare_torc_convention.pdf`
uses the TorC convention (residuals in D_V/(r_d z^{2/3}) and D_M/(z D_H) against
scale factor a) so it can be laid beside the TorC figures.

`cpl_fgivenx.py` caches the expensive step in `output/cache_*` — the first run
takes ~40 min, reruns are instant. `cpl_torc_compare.py` instead subsamples to
`--nsub` equally-weighted draws (default 2000), which keeps it to a few minutes
with no cache.

### Evidence table

```bash
python -c "from cpl_torc_compare import *; print_evidences(load_all(nsub=200))"
```

---

## 2. TorC — the parent directory (`distances/TorC/`)

Run from the TorC repo root (one level up from here), not from `scripts/`:

```bash
cd /Users/starfish/Desktop/PhD2025-26/distances/TorC
```

### Chains

| Chain | Made by |
|---|---|
| `../chains/torc_ns_noH0.npz` | `../scripts/run_ns.py` (250 live, exploration) |
| `../chains/torc_ns_julia_1000.npz` | `../scripts/run_ns_julia_1000.py` (uniform ς prior) |
| `../chains/torc_ns_julia_1000_powerlaw.npz` | same, `STIGMA_PRIOR = "powerlaw"` (**the 1/ς² run**) |
| `../chains/torc_ns_julia_1000_exponential_lam{0p5,1,2}.npz` | same, exponential ς priors |
| `../chains/lcdm_ns_1000.npz` | `../scripts/run_lcdm.py` (1000 live) |
| `../chains/lcdm_ns.npz` | `../scripts/run_lcdm.py` (250 live) |

The ς prior is selected by `STIGMA_PRIOR` in `../torc/priors.py`.

### fgivenx plots

| Figure | Script | Chain |
|---|---|---|
| `../plots/torc_noH0_fgivenx_desi.{pdf,png}` | **`../scripts/plot_fgivenx.py`** | `torc_ns_noH0.npz` |
| `../plots/torc_fgivenx_1000.{pdf,png}` | **`../scripts/plot_fgivenx_1000.py`** | `torc_ns_julia_1000.npz` |

**Note:** the TorC reference figure is `plot_fgivenx.py`, which reads
`torc_ns_noH0.npz` — *not* the power-law chain. `compare_torc_convention.pdf`
overlays `torc_ns_julia_1000_powerlaw.npz`, so the two agree in shape and in
their a→1 limits but not point-for-point.

### Corner plots

| Figure | Script | Chains |
|---|---|---|
| (noH0 corner) | **`../scripts/plot_corner.py`** | `torc_ns_noH0.npz`, `torc_ns.npz` |
| `../plots/torc_noH0_vs_lcdm_corner.{pdf,png}` | **`../scripts/plot_corner_vs_lcdm.py`** | `torc_ns_noH0.npz`, `lcdm_ns.npz` |
| `../plots/torc_1000live_corner.{pdf,png}` | **`../scripts/plot_corner_1000.py`** | `torc_ns_julia_1000.npz` |
| `../plots/torc_vs_lcdm_1000live_corner.{pdf,png}` | **`../scripts/plot_corner_1000_vs_lcdm.py`** | `torc_ns_julia_1000.npz`, `lcdm_ns_1000.npz` |
| `../plots/torc_mups2fixed_vs_lcdm_corner.{pdf,png}` | **`../scripts/plot_corner_mups2fixed_vs_lcdm.py`** | mups2-fixed run |
| `../plots/torc_mups2_fixed_vs_sampled_corner.{pdf,png}` | **`../scripts/plot_corner_mups2_fixed_vs_sampled.py`** | mups2 fixed vs sampled |
| `../plots/torc_inner_steps_corner.{pdf,png}` | **`../scripts/plot_corner_inner_steps.py`** | `torc_ns_inner13/25.npz` |
| `../plots/julia_vs_diffrax_corner.{pdf,png}` | **`../scripts/plot_corner_julia_vs_diffrax.py`** | Julia vs diffrax backends |
| `../plots/torc_prior_posterior_corner.{pdf,png}` | **`../scripts/plot_corner_prior_posterior.py`** | prior vs posterior |

### Prior-comparison plots (these are where the 1/ς² chain is used)

| Figure | Script | Chains |
|---|---|---|
| `../plots/torc_posteriors_only.{pdf,png}` | **`../scripts/plot_posteriors_only.py`** | all four ς priors + LCDM — **includes the power-law 1/ς² chain** |
| `../plots/prior_sensitivity.{pdf,png}` | **`../scripts/plot_prior_sensitivity.py`** | uniform, exp λ=0.5, exp λ=2.0, LCDM |
| `../plots/torc_prior_posterior_exp_powerlaw.{pdf,png}` | **`../scripts/plot_exponential_prior_posterior.py`** | power-law prior vs its posterior |

---

## Quick answers

- **"Which file makes the CPL fgivenx plot?"** → `cpl/cpl_fgivenx.py` (CPL
  alone), or `cpl/cpl_torc_compare.py` (CPL overlaid on TorC).
- **"Which file makes the TorC fgivenx plot?"** → `../scripts/plot_fgivenx.py`
  (noH0 chain) or `plot_fgivenx_1000.py` (1000-live uniform-prior chain).
- **"Which file makes the CPL corner plot?"** → `cpl/cpl_nss.py --plot`, or
  `python/cpl_torc_compare.py --only corner` for the three-way overlay.
- **"Which file makes the TorC corner plot?"** → `../scripts/plot_corner_1000.py`
  and friends; for the 1/ς² prior specifically, `plot_posteriors_only.py`.
