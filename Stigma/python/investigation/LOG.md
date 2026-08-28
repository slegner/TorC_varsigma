# Investigation log — why full dynamical equations give ς>1-type evolution when ς<1

Started 2026-08-25. Working copy: `investigation/Notebooktrt3_WORKING.nb` (original untouched).

## Task
From the Lagrangian, re-derive Φ, lapse, scale-factor, ϖ equations. Audit every
elimination step and every division by expressions like (ς − ϖ²). Compare full
equations against the algebraic constant-ϖ correspondence roots.

## Hypotheses tested
(table below, appended as work proceeds)

| # | Hypothesis | Status | Evidence |
|---|---|---|---|

## Running notes

### Notebook map (Notebooktrt3.nb, 149 Input cells)

- IN[2-4]  : xAct setup, FLRW metric diag(N²,-a²,-a²,-a²), fields φ,ψ,ϖ; current
             `J[a] = 4 sig1 ψ³ ∇^a(φ/ψ) − Mp² ups2 ∇^a φ`;
             `JJ = J·J`; **`sqJJ = PowerExpand[Sqrt[JJ]]`**  ← branch choice #1
- IN[5]    : L = √−g ( (½Mp²υ₂ − ½σ₂ψ²)R − 6σ₂X_ψ + √(J·J) + ¾Mp²(υ₂φ²+4υ₁ψ²) − 3σ₂φ²ψ² )
- IN[6]    : sub sig1→sig2·stigma, ups1→0, ψ→(Mp√υ₂/√(4σ₂ς))ϖ.
             Euler–Lagrange wrt N, a, φ, ϖ → Eqn, Eqa, EqPhi, EqVarpi.
             Then N→1, and **`Phis[t] = Phis[t] /. Flatten[DSolve[EqPhi==0,...]]`** ← branch/root choice #2
- IN[7]    : `EqnFinal /. sign -> 1`   ← a symbol literally called `sign` survives into EqnFinal
- IN[9]    : ups2 = −mups2
- IN[15-19]: Friedmann constraint solved for H². Denominators `(ς − ϖ²)` and `(4ς − ϖ²)`
- IN[21-24]: t → conformal-ish time Tau; E1 (ϖ eq), F1 (lapse/Friedmann), F2 (scale factor)
- IN[50-55]: **algebraic constant-ϖ correspondence**: EqVarpiFinal with ϖ'=ϖ''=0,
             H = 2/(3(1+w_i)t), x = ϖ². `xRoots = Solve[Eqsolve==0, x]`,
             xL = xRoots[[2]], xR = xRoots[[3]],
             `xPhys := Piecewise[{{xR, ς<1},{xL, ς>1}}]`   ← **hand-assigned branch**
- IN[66-67]: xPlus series (1/2<ς<1): x ≈ 1 − 3(ς−1)ε/(8ς²)
- IN[80,136]: `Diffs = SolveValues[{F2==0, E1==0}, {a'', ϖ''}]`  ← division-by-zero risk
- IN[89]   : r = ς(4ς−1)/((2ς−1)(2ς+1));  q = r−1 = (1−ς)/(4ς²−1)
- IN[91,137]: E11 — explicit ϖ equation, denominator `8 mups2 ς (4ς²−1) a² ϖ (ς−ϖ²)²`
- IN[124-149]: early-time power series a=Σa_i τ^i, **ϖ = 1 + Σ ϖ_i τ^i (i.e. ϖ₀ ≡ 1)**

### Immediate suspects (before running anything)
1. `PowerExpand[Sqrt[JJ]]` — Sqrt[J·J] is |J|; PowerExpand commits to one sign branch.
2. The `sign` symbol in IN[7] confirms a sign ambiguity was known and hard-set to +1.
3. `DSolve[EqPhi==0, Phis[t], t]` + `Flatten[...]` takes whichever solution comes first.
4. `xRoots[[2]]`/`[[3]]` indexing + the `Piecewise` on ς<1 vs ς>1 — assigns branches by hand.
5. Constraint (memory) `ς > ϖ₀²`; with the series normalisation ϖ₀=1 this *requires ς>1*.
   So denominators (ς−ϖ²) change sign between the ς<1 and ς>1 setups.

---
## Run 1 — Lagrangian reproduced (wolframscript + xAct)

`JJ = J·J = K²/N²` with

    K = (Mp²υ₂ − 4σ₁ψ²)·φ' + 4σ₁ φ ψ ψ'

`PowerExpand[Sqrt[JJ]]` returns **K/N**, but √(J·J) is **|K|/N**.
So the Lagrangian actually used is the |K| = +K branch. Verified `sqJJ^2 - JJ == 0`
(i.e. Mathematica only guarantees the square, not the sign).

After the notebook's substitutions σ₁ = σ₂ς and ψ = Mp√υ₂ ϖ /√(4σ₂ς):

    Mp²υ₂ − 4σ₁ψ² = Mp²υ₂(1 − ϖ²)
    4σ₁ φ ψ ψ'     = Mp²υ₂ φ ϖ ϖ'
  ⇒ K = Mp²υ₂ [ (1 − ϖ²) φ' + φ ϖ ϖ' ]

and the potential terms collapse to  (3/4)Mp²υ₂ φ² (ς − ϖ²)/ς.

**H2 (hypothesis): the sign of K flips at ς = 1**, because the early-time series gives
(IN[144])  ϖ₁ ∝ (ς − 1).  With ϖ = 1 + ϖ₁τ + …:
    K ≈ Mp²υ₂ ϖ₁ (φ − 2τφ')  →  sign(K) ∝ sign(υ₂ ϖ₁ φ) ∝ sign(ς − 1).
If so, `sign→1` hard-wires ONE side of ς=1 and running ς<1 integrates the ς>1 branch.

Next: redo the EL derivation with the sign kept symbolic (`sgn`), confirm sgn→1
reproduces the notebook, then determine sign(K) on-shell for ς<1 and ς>1.

---
## Runs 2-4 — independent re-derivation from the Lagrangian

Reproduced end-to-end with xAct in a standalone script (`scratchpad/master.wl`,
`stageB/C/D.wl`), NOT reusing the notebook's stored F1/E1/E11.

### Result A — the Φ sector (confirms user item 9, sharpens it)
`EqPhi` is **purely algebraic in Φ**: coefficient of Φ′ in EqPhi is *identically zero*
(the Φ′ terms cancel in the EL variation). Coefficient of Φ is
`(3Mp²υ₂ a³ N (ς−ϖ²))/(2ς)`.

    Φ = −2·sgn·ς [ (ϖ²−1)a′ + a ϖ ϖ′ ] / ( a (ς − ϖ²) )

So the pole is created *only* by this elimination. Matches user's item 9 exactly.

### Result B — square-root sign is irrelevant (confirms user item 6)
Carried `sgn` symbolically through the whole derivation. In EqnFinal / EqaFinal /
EqVarpiFinal, `Exponent[..., sgn] == 2` in every case, and
`Eq(sgn=1) − Eq(sgn=−1) == 0` for all three. **Ruled out.**
*Reason (structural, not luck):* Φ ∝ sgn, and L = (quadratic in Φ) + sgn·(linear in Φ);
back-substituting a Φ ∝ sgn can only ever produce sgn².

### Result C — leading orders all carry powers of (ς−1)
Conformal time, a = Σ aᵢτⁱ (i≥1), ϖ = 1 + Σ vᵢτⁱ:

  F1 leading (T⁰):  −3Mp²**(ς−1)**·[ 4H₀²Ω_r0ς + mυ₂a₁²(1−4ς) ]
  E1 leading (T⁰):  −3Mp²mυ₂**(ς−1)²**·a₁·( a₂ + a₁v₁ )

These are *numerator* factors — the (ς−ϖ²) denominators were already cleared — so they
are genuine. The whole leading-order system degenerates at ς=1.

### Result D — v₁ is unique; the notebook contains a wrong duplicate
Both order-(T⁰,T¹) equations are **linear** in {a₂,v₁} ⇒ unique solution, no ± branch:

    a₁ = 2H₀√(Ω_r0 ς)/√(mυ₂(4ς−1))          (a₁>0 branch chosen = expanding)
    a₂ = H₀²Ω_m0/(4 mυ₂)
    v₁ = −H₀Ω_m0√(4ς−1)/(8√(mυ₂ Ω_r0 ς))   < 0  for ALL ς > 1/4

✔ agrees with `knownRules` (IN[31]/IN[121]).
✘ **contradicts IN[144]**, which reports ϖ₁ ∝ (ς−1)·… .  IN[144] is wrong: it descends
from IN[125] `ap = SolveValues[{F1==0},{a′}]` (picks one root of a *quadratic*, with
PowerExpand) and then IN[131-132] `aplist2 = Drop[aplist,{2}]; aplist2[[1]] += na[1]`,
which is an unjustified hand-patch. **Discard the IN[144]–IN[149] chain.**

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| H1 | Missing IC sign | ruled out a priori (user instruction) + v₁ unique | Result D |
| H2 | PowerExpand[Sqrt[JJ]] hides a ς-dependent sign branch | **RULED OUT** | Result B: sgn² only |
| H3 | Notebook has two inconsistent v₁'s, IN[144] is right | **RULED OUT** | Result D: knownRules is right, IN[144] is the bad one |
| H4 | ϖ₀=1 is an unjustified normalisation; true radiation ϖ₀ is ς-dependent | TESTING | — |

---
## Runs 5-9 — the answer

### Result E — no algebra bug anywhere (all three checks negative)
- `dF1/dT` on the {F2=0, E1=0} surface is **identically 0** — not merely ∝ F1. The
  constraint is an exact first integral; the Φ elimination is fully consistent.
- `E11(notebook IN[91]) − (my ϖ″ from SolveValues[{F2,E1}]) ≡ 0`. The notebook's
  dynamical equations are correct.
- F1 stays satisfied numerically along the ς=0.9 run (1e-15 → 1e-11, tracking term scale).

### Result F — the crash reproduced from my own equations
NDSolve on my F2/E1 with the series ICs (mυ₂=4/3, Ω_r0=9e-5, Ω_m0=0.31, Ω_L0=0.69):
  ς=0.9 → ends τ=0.00505, ϖ²=0.90017 (= ς)   **crash at ϖ²=ς**
  ς=0.6 → ends τ=0.01119, ϖ²=0.60115 (= ς)   **crash at ϖ²=ς**
  ς=1.5 → survives to τ=3.45, ϖ² falls 1→0.885→0.45 (pole at 1.5 is ABOVE, so safe)
**ς=0.9 crashes at a=4.2e-5, i.e. BEFORE equality (a_eq=Ω_r0/Ω_m0=2.9e-4)** — deep in
radiation domination, where ϖ should still be ≈1.
⇒ ς>1 is not "working"; it is failing in a safe direction. ϖ drifts fast in BOTH regimes.

### Result G — the correspondence, rebuilt independently
Regular root (w: 1/3→0):  ς=0.9: 1→0.982 | ς=0.6: 1→0.888 | ς=1.5: 1→1.043 | ς=2: 1→1.053
- Confirms the notebook's hand-assigned Piecewise (xR regular for ς<1, xL for ς>1):
  the ε^{-1} coefficient of the series vanishes for exactly one root on each side.
- Confirms x_reg = 1 − 3(ς−1)ε/(8ς²) + …
- **For ς<1 the correspondence says ϖ DECREASES.** So v₁<0 is the correct direction;
  the expectation (item 10) that the ς<1 branch should move to larger ϖ is not what the
  correspondence says.
- Equilibrium never reaches the pole (0.888 > 0.6). But the dynamical solution blows
  through it: v₁^corr/v₁^dyn = −(ς−1)/(ς(4ς−1)) = 0.043 at ς=0.9 (23× too fast),
  −0.067 at ς=1.5 (opposite sign). **The trajectory never tracks the correspondence.**

### Result H — ϖ₀=±1 is the unique regular radiation fixed point (proved)
Pure radiation, ϖ=const, constraint imposed, a ∝ τ; the ϖ-equation factors to
    −48H₀²Mp²Ω_r0(ς−1)ς³·**(w₀−1)(w₀+1)**·(ς−w₀²) / [w₀(5ς−8ς²−w₀²+4ς²w₀²)]
Roots: w₀=±1, or w₀²=ς (which IS the singular surface, not an interior solution).
Also, the correspondence polynomial at w=1/3 factors to 8Mp²mυ₂(ς−1)ς²√x(√x−1)(√x+1)a³
⇒ x=1 uniquely. **ϖ₀=±1 confirmed rigorously.**

### Result I — THE ANSWER: the two no-ghost conditions, and why ς<1 hides its own violation
From the Lagrangian (ups2 = −mups2, mups2>0):
  coeff of Φ² in L   = −3Mp²mυ₂a³(ς−ϖ²)/(4ς)          healthy ⟺ **ς > ϖ²**
  coeff of a′² in F1 = −3Mp²mυ₂ϖ²[(4ς²−1)ϖ²+ς(5−8ς)]  healthy ⟺ **bracket < 0**
  coeff of ϖ′² in L  = +3Mp²mυ₂a³/(4ς)                 always fine
These are *verbatim* `validate_stigma_constraints`, python/equations.py:22-23.
**At ϖ₀=1 both reduce to ς > 1**  (bracket|_{ϖ=1} = −(ς−1)(4ς−1)).

The exact constraint root is
    a′ = ± 2i H₀√Ω_r0 √ς √(ς−w₀²) / √(mυ₂w₀²[5ς−w₀²+4ς²(−2+w₀²)])
| ς | ς−ϖ₀² | bracket | √(ς−1) | √(bracket) | a′ exact | a₁ notebook |
|---|---|---|---|---|---|---|
|0.6| −0.4 ✗| +0.56 ✗| imaginary | real | −0.010757 | +0.010757 |
|0.9| −0.10 ✗| +0.26 ✗| imaginary | real | −0.009668 | +0.009668 |
|1.5| +0.5 ✓| −2.5 ✓| real | imaginary | +0.009000 | +0.009000 |
|2.0| +1.0 ✓| −7.0 ✓| real | imaginary | +0.008783 | +0.008783 |

**Exactly one of the two square roots is imaginary on each side of ς=1**, so a′ (and every
series coefficient built from it) comes out REAL on both sides — via a *different* pairing
of imaginary factors. `PowerExpand` in IN[125]/knownRules bakes in the ς>1 pairing and then
evaluates it happily at ς<1. Nothing in the algebra ever signals the crossing.
(cf. the memory note "the I in a1 comes from TWO sources that cancel" — they cancel on BOTH
sides, which is precisely the trap.)

### CONCLUSION
For ς<1 at ϖ₀=±1, **both** no-ghost conditions are violated **simultaneously**, and the two
sign flips cancel inside every square root. So the ς<1 run is not a different branch — it is
the ς>1 branch's formulas continued across the singular surface into the ghost sector. That
is exactly why it "produces ς>1-type evolution". The universal downward drift (v₁<0 for all
ς>1/4) then carries ϖ into ϖ²=ς, the boundary where the Φ elimination is singular; for ς>1
the identical drift moves away from √ς, so it survives.

**With ϖ₀=±1 there is no healthy ς<1 cosmology. The theory as written requires ς>1.**
The "valid ς<1 branch" visible in the ϖ field equation (root x_R) is real but sits at
ϖ²>ς — inside the forbidden sector.

| # | Hypothesis | Status |
|---|---|---|
| H1 | Missing IC sign | ruled out — v₁ unique (Result D) |
| H2 | PowerExpand[Sqrt[JJ]] sign branch | **ruled out** — sgn² only (Result B) |
| H3 | IN[144] v₁ is the right one | **ruled out** — knownRules is right (Result D) |
| H4 | ϖ₀ is a free parameter, true value ς-dependent | **ruled out** — ϖ₀=±1 forced (Result H) |
| H5 | Algebra/elimination bug in notebook | **ruled out** — dF1/dT≡0, E11 matches (Result E) |
| H6 | ς<1 is the ghost sector; both constraints flip together | **CONFIRMED** (Result I) |
