# Stigma — Q-torsion modified gravity

Symbolic derivation (Mathematica) and numerical cosmology (Python) for the Stigma
Q-torsion model.

Lives in the TorC_varsigma repo at `PhD2025-26/distances/TorC/Stigma/`. It was
briefly absorbed into the Qtorsion repo (Aug 2026) before being relocated here,
next to the related CPL work. Its own pre-absorb git history is preserved as a
bundle in `archive/` — see `archive/RESTORING_HISTORY.md`.

**Related work nearby:** the CPL/w0waCDM DESI DR2 comparison is a sibling at
`../cpl/`. DESI distance data is read from `PhD2025-26/distances/desidr2/` via
`../../../desidr2/` — Stigma now sits inside `distances/`, so that path is one
segment shorter than it used to be.

## Layout

| Path | Contents |
|---|---|
| `mathematica/` | xAct derivations and notebooks. `StigmaEquationsOfMotion.m` → `ModifiedEinstein.m` → `ModifiedFriedmann.m` → `EffectivePandRho.m`; `EarlyTimePowerSeries.m` for the early-time series. `mups2/` is the parallel μ·υ² parameterisation of the same module set. |
| `python/` | Numerical background/perturbation code. **Flat by design** — modules import each other by bare name (`from equations import ...`), so keep new modules at this level. |
| `python/lambda/` | Λ-extension study (self-contained, see its own `README.md`). |
| `python/investigation/` | Aug 2026 branch-structure investigation (self-contained, see `LOG.md`). |
| `jupyter/` | Exploratory notebooks (JAX, DESI fits). |
| `Plots/` | Published figures. Several `python/` scripts write here via `Plots/` or `../Plots/` — don't rename this directory. |
| `latex/` | Analysis write-ups. `mathematica/ModifiedEinstein.m` exports `gisolution_*.pdf` here. |
| `paper/` | The paper itself (`paper_Stigma.tex`). |
| `docs/` | `todo.md` and derivation logs. |
| `archive/` | Quarantined during the Aug 2026 cleanup — stale duplicate PDFs, two-line stubs, and the pre-absorb git history bundle. |

## Running things

Mathematica modules are standalone; run from `mathematica/`:

```
wolframscript -file StigmaEquationsOfMotion.m
```

Python scripts run from `python/`, which is also where imports resolve:

```
cd python && python analysis.py
```
