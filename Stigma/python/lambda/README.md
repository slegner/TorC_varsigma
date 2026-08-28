# `lambda/` — second H(z) model (Λ switched on)

A second background model that produces `H(z)`, set up so that it starts from
**the same initial values** as the reference model in `python/equations_Hz.py`
and differs **only in the second derivatives** `a''` and `varpi''`.

Those two derivatives are the only thing left blank — everything else
(constraint check, `tau_ini`, the `a0` shooting, the H(z) output, the plots) is
already wired up and tested.

## Files

| file | what it is |
|---|---|
| `equations_lambda.py` | the Λ model. **Fill in `dda` and `ddcp`.** |
| `reference_model.py` | loads `python/equations_Hz.py` as the reference (λ = 0) model, unmodified |
| `check_lambda.py` | quick sanity checks on `dda`/`ddcp` |
| `verify_lam0.py` | **regression check: at `lam = 0` the model must reproduce the original run exactly** |
| `compare_hz.py` | reference vs Λ model, one `lam` → `hz_comparison.pdf` |
| `compare_configs.py` | the three sources of acceleration: OmL only / lam only / both → `hz_configs.pdf` |

## Status of the equations

`dda` / `ddcp` are filled in from `Notebooktrt2lam.nb`.  Two things were needed
to make them work with the rest of the pipeline:

**1. a0 convention.**  The pasted expressions contain no `a0` — they are written
for the *physical* scale factor.  The series ICs and `rhopsol` use the reference
convention where the physical scale factor is `a*a0`.  The pasted bodies are
kept verbatim as `_dda_phys` / `_ddcp_phys`, and thin wrappers convert:

```python
dda (a, da, ...) = _dda_phys (a*a0, da*a0, ...) / a0
ddcp(a, da, ...) = _ddcp_phys(a*a0, da*a0, ...)
```

Without this the a0 shooting in `rhopsol` is inconsistent and H(z=0) comes out
as 1587 instead of 67 km/s/Mpc.  If you re-export from Mathematica with
`a -> a*a0` substituted, delete the wrappers and rename `_dda_phys`/`_ddcp_phys`
back to `dda`/`ddcp`.

**2. `P_deriveda0` / `rho_deriveda0`** were lost when the expressions were pasted
in and have been restored.  They are still the **lam = 0** expressions — they
accept `lam` and ignore it.  Λ enters `dda`/`ddcp`, so it almost certainly
enters the effective DE split too: re-derive them before trusting `P_DE` /
`rho_DE` at `lam != 0` (e.g. before feeding them to CAMB).

Checks that pass right now:

* at `lam = 0`, `dda`/`ddcp` reproduce `python/equations_Hz.py` to 0 ulp, and
  the whole `rhopsol` pipeline reproduces the reference H(z) to 1e-11;
* at `tau_ini` the Λ terms are below double precision (they carry `a**4`, and
  `a ~ 1e-7` there), so the two models genuinely do start from the same initial
  values — the shared series ICs are justified;
* at `lam = 1` the H(z) difference is ~0.5–0.7 %.

## Choosing `lam`

`lam` and `OmL` sit side by side in the same `a**4` source block of `dda`/`ddcp`,
with different `stigma` structure, so they are two independent Λ-like sources.

**At `lam = 0` the Λ model is bit-identical to the reference** — the second H(z)
would just retrace the first.  Measured through the full pipeline
(`cpr=0.99, omega_m=0.3, h=0.67, stigma=1.5, mups2=4/3`; `a0` shooting pins
`H0 = 67` in every case):

| OmL | lam | ΔH/H at z=1 | ΔH/H at z=1000 |
|---|---|---|---|
| 0.7 | 0.0 | 0 (identical) | 0 (identical) |
| 0.7 | 1.0 | −0.66 % | −0.65 % |
| 0.0 | 0.7 | +2.66 % | +2.67 % |
| 0.0 | 1.0 | +2.34 % | +2.35 % |

`OmL = 0` with `lam` alone gives a perfectly well-behaved cosmology, which is
what you would use if Λ is meant to *replace* the fluid rather than add to it.
Note the difference is nearly z-independent in every case — mostly a
normalisation shift, with only a small change of shape.

## The three configurations

`compare_configs.py` runs, at fixed `H(z=0) = 100h`:

| | OmL | lam | |
|---|---|---|---|
| **OmL only** | 0.7 | 0 | the original model — Λ as a fluid |
| **lam only** | 0 | 0.7 | Λ from the theory's own term |
| **OmL + lam** | 0.7 | 0.7 | both |

Measured (`cpr=0.99, omega_m=0.3, h=0.67, stigma=1.5, mups2=4/3`), as ΔH/H
against the OmL-only run:

| z | lam only | OmL + lam |
|---|---|---|
| 0.5 | +2.11 % | −0.38 % |
| 1 | +2.66 % | −0.47 % |
| 2 | +2.94 % | −0.51 % |
| 10 | +3.04 % | −0.53 % |
| 1000 | +2.67 % | −0.46 % |

Both differences are nearly flat in z: at fixed H0 the `lam` term acts mostly
like a rescaling of the expansion history rather than a change of its shape,
and it pushes H *up* on its own but *down* when added on top of OmL.

## The lam = 0 regression check

`verify_lam0.py` is the check that the Λ equations reduce to the original run:

* right-hand sides over 2000 random states at three values of `a0`: max relative
  difference ~1e-14, i.e. a few tens of ulp — the same formula with the
  sign/denominator rearranged, not a different one;
* `tau_ini` and the initial state: bit-identical;
* from identical ICs the trajectories track to 1e-14 in `a` and `varpi`, and
  that residual does **not** shrink when the solver tolerance is tightened from
  1e-8 to 1e-12, which is what proves it is arithmetic and not physics;
* end-to-end `rhopsol` agrees to 2e-8 (the a0 shooting amplifying that rounding
  through the event detection), with `H(z=0) = 67.0000000037` for both.

## Running

```sh
cd lambda
python3 verify_lam0.py      # lam = 0 must reproduce the original run
python3 compare_configs.py  # OmL only vs lam only vs both  -> hz_configs.pdf
python3 compare_hz.py       # reference vs one lam value    -> hz_comparison.pdf
python3 check_lambda.py     # quick spot check of dda/ddcp
```

Parameters live at the top of each script:

```python
cpr = 0.99; omega_m = 0.3; h = 0.67; stigma = 1.5; mups2 = 4/3
OmL_val = 0.7; lam_val = 0.7      # compare_configs.py
```

`check_lambda.py` has one switch, `REDUCES_TO_REFERENCE_AT_ZERO_LAM = True` —
set it to `False` if the Λ model is *not* expected to reduce to
`python/equations_Hz.py` at `lam = 0`.

## Using it from a notebook

```python
import sys; sys.path.insert(0, '/Users/starfish/Desktop/PhD2025-26/distances/TorC/Stigma/lambda')
from equations_lambda import rhopsol
z, hubble, P_DE, rho_DE, asol = rhopsol(cpr, omega_m, h, OmL, stigma, mups2, lam)
```

Same return signature as `equations_Hz.rhopsol`, plus the trailing `lam`
argument.
