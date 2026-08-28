(* ::Package:: *)

(* ============================================================================
   Notebooktrt2.m
   Early-time (radiation-era) power series for the Stigma model, extending
   Notebooktrt.nb to ScaleFactor4, ScaleFactor5 and VarPi3, VarPi4.

       a(tau)     = a1 tau + a2 tau^2 + a3 tau^3 + a4 tau^4 + a5 tau^5
       varpi(tau) = 1 + w1 tau + w2 tau^2 + w3 tau^3 + w4 tau^4

   varpi0 = 1 is NOT an assumption: Section 2 shows the 1/tau pole of the
   varpi equation of motion is proportional to varpi0 (varpi0^2 - 1)(stigma - 1),
   so regularity at the bang pins varpi0 = +-1 (we take +1; the -1 case is the
   exact mirror varpi -> -varpi of the equations).

   Equations of motion: transcribed from Notebooktrt.nb chapter 6, i.e. the
   conformal-time system after rhorules and ups2 -> -mups2:
     CC : the N-constraint (Friedmann).  Equals the notebook's F1 up to the
          nonzero overall factor 3 alp Mp^2 a[tau]/(4 stigma), so same roots.
     E1 : the varpi equation of motion  (notebook's constraintE / E1).
     F2 : the scale-factor equation      (notebook's constraintF2), used only
          as an independent redundancy check via the Bianchi identity.
   These three were cross-validated numerically against the notebook's
   Diffs (dda/ddcp) export: mutual consistency at 1e-19 relative, and the
   constraint is conserved along the {F2, E1} flow.

   ROOT-CHOICE POLICY (replaces PowerExpand):
     The ONLY branch choice in the whole computation is the sign of a1
     (expanding vs contracting).  a1^2 = 4 H0^2 OmR stigma /
     (alp^2 mups2 (4 stigma - 1)) has a manifestly positive radicand for
     stigma > 1/4, so the expanding root is written down explicitly.
     Every later stage is a LINEAR 2x2 solve for {w_k, a_(k+1)} -- unique,
     no radicals introduced, hence nothing for PowerExpand to mangle.
     (This is where the old "mixed branch" went wrong: PowerExpand /
     Abs-stripping on Sqrt[stigma - varpi0^2]-type radicals, which are only
     jointly -- not separately -- real through the (stigma - 1) cancellation
     at varpi0 = 1.)

   Run:   << "Notebooktrt2.m"     (from the mathematica/ directory)
   or:    wolframscript -file Notebooktrt2.m
   ==========================================================================*)

ClearAll["Global`*"];

(* ---------------------------------------------------------------------------
   0.  Symbols and assumptions
       H0  : 100 h  (km/s/Mpc units as in the notebook, 40000 h^2 = 4 H0^2)
       OmR, OmM, OmL : Omega_r0, Omega_m0, Omega_Lambda0
       alp : the alpha scale normalisation (a0 in the python export)
   -------------------------------------------------------------------------*)
$Assumptions = And[
   stigma > 1/4, stigma != 1,
   H0 > 0, OmR > 0, OmM > 0, OmL > 0,
   mups2 > 0, alp > 0, Mp > 0, tau > 0];

(* ---------------------------------------------------------------------------
   1.  Equations of motion (a, cp are formal heads replaced by series later)
   -------------------------------------------------------------------------*)
CC = -2 mups2 cp[tau] a'[tau] cp'[tau]/a[tau] - mups2 cp'[tau]^2 -
     4 mups2 stigma^2 ((cp[tau]^2 - 1) a'[tau]/a[tau] +
        cp[tau] cp'[tau])^2/(stigma - cp[tau]^2) +
     mups2 (4 stigma - cp[tau]^2) a'[tau]^2/a[tau]^2 -
     4 H0^2 stigma (OmR + alp OmM a[tau] +
        alp^4 OmL a[tau]^4)/(alp^2 a[tau]^2);

E1 = (-3 alp Mp^2 mups2 (
      a[tau] (2 a'[tau] cp'[tau] + a[tau] cp''[tau]) stigma^2 +
      cp[tau] stigma^2 (a[tau] a''[tau] (1 - 4 stigma) +
         4 a'[tau]^2 (-1 + stigma) + 4 a[tau]^2 cp'[tau]^2 stigma) +
      a[tau] cp[tau]^5 a''[tau] (1 - 4 stigma^2) +
      2 a[tau] cp[tau]^2 (2 a'[tau] cp'[tau] + a[tau] cp''[tau]) stigma (-1 + 2 stigma^2) -
      a[tau] cp[tau]^4 (2 a'[tau] cp'[tau] + a[tau] cp''[tau]) (-1 + 4 stigma^2) +
      2 cp[tau]^3 stigma (-2 a'[tau]^2 (-1 + stigma) stigma +
         a[tau] a''[tau] (-1 + 2 stigma (1 + stigma)))
      ))/(2 a[tau] stigma (stigma - cp[tau]^2)^2);

F2 = (3 Mp^2 (
      -12 a[tau]^2 alp^2 H0^2 OmL +
      ((4 H0^2 OmR)/alp^2 +
        (cp[tau]^2 a'[tau]^2 mups2 ((5 - 8 stigma) stigma +
           cp[tau]^2 (-1 + 4 stigma^2)))/(stigma (stigma - cp[tau]^2)))/a[tau]^2 +
      (2 cp[tau] mups2 (cp[tau] a''[tau] (cp[tau]^2 - stigma) ((5 - 8 stigma) stigma +
           cp[tau]^2 (-1 + 4 stigma^2)) +
         a'[tau] cp'[tau] (3 stigma^2 (-3 + 4 stigma) +
           2 cp[tau]^2 stigma (1 + 2 stigma - 6 stigma^2) +
           cp[tau]^4 (-1 + 4 stigma^2))))/(a[tau] stigma (stigma - cp[tau]^2)^2) +
      (2 cp[tau] cp''[tau] mups2 (cp[tau]^2 - stigma) (stigma - 4 stigma^2 +
           cp[tau]^2 (-1 + 4 stigma^2)) +
        cp'[tau]^2 mups2 (stigma^2 (1 + 8 stigma) + cp[tau]^4 (1 - 4 stigma^2) -
           2 cp[tau]^2 stigma (1 - 4 stigma + 6 stigma^2)))/(stigma (stigma - cp[tau]^2)^2)
      ))/4;

(* ---------------------------------------------------------------------------
   2.  Check: regularity pins varpi0 = +-1
       (leading 1/tau pole of the varpi EOM with general varpi0)
   -------------------------------------------------------------------------*)
Print["--- Section 2: varpi0 pinning check ---"];
pinning = Factor@Simplify@SeriesCoefficient[
     E1 /. {a -> Function[t, ac1 t], cp -> Function[t, p0 + p1 t]},
     {tau, 0, -1}];
Print["1/tau coefficient of varpi EOM  =  ", pinning];
Print["  -> vanishes only for varpi0 = +-1 (given stigma != 1, a1 != 0).  \
Proceeding with varpi0 = +1."];

(* ---------------------------------------------------------------------------
   3.  Series ansatz and stagewise solve
       Orders used (verified: each stage's pair is linear in its 2 unknowns
       and free of higher ones):
         constraint at tau^(k-2)  &  varpi EOM at tau^(k-1)
                                  ->  {w_k, a_(k+1)},   k = 1..4
   -------------------------------------------------------------------------*)
nA = 5; nP = 4;
aserF = Function[t, Evaluate[Sum[ac[k] t^k, {k, 1, nA}]]];
pserF = Function[t, Evaluate[1 + Sum[pc[k] t^k, {k, 1, nP}]]];

CCs = Together[CC /. {a -> aserF, cp -> pserF}];
E1s = Together[E1 /. {a -> aserF, cp -> pserF}];
F2s = Together[F2 /. {a -> aserF, cp -> pserF}];

Print["--- Section 3: stagewise solve ---"];

(* stage 0 : a1 from the tau^-2 pole of the constraint *)
c2 = Simplify@SeriesCoefficient[CCs, {tau, 0, -2}];
a1sq = Simplify[x /. First@Solve[(c2 /. ac[1]^2 -> x) == 0, x]];
Print["a1^2 = ", a1sq];
rules = {ac[1] -> 2 H0 Sqrt[OmR stigma/(mups2 (4 stigma - 1))]/alp};
Print["explicit expanding root a1 = ", ac[1] /. rules,
      "   (root check: ", Simplify[(ac[1]^2 - a1sq) /. rules], " should be 0)"];

(* stages 1..4 : linear solves *)
Do[
  Module[{unk, eqC, eqE, sol, newrules},
   unk = {pc[k], ac[k + 1]};
   eqC = Together@SeriesCoefficient[CCs /. rules, {tau, 0, k - 2}];
   eqE = Together@SeriesCoefficient[E1s /. rules, {tau, 0, k - 1}];
   sol = Solve[{eqC == 0, eqE == 0}, unk];
   Which[
    sol === {}, Print["!! stage ", k, ": no solution -- pairing broke down"]; Abort[],
    Length[sol] > 1, Print["!! stage ", k, ": ", Length[sol],
      " solutions -- NOT unique, inspect!"]; Abort[]];
   newrules = Simplify[First[sol], TimeConstraint -> 120];
   rules = Join[rules, newrules];
   Print["stage ", k, " solved for ", unk,
     "  (LeafCount ", LeafCount[newrules], ")"];
   ], {k, 1, 4}];

(* ---------------------------------------------------------------------------
   4.  Double checks
   -------------------------------------------------------------------------*)
Print["--- Section 4: checks ---"];

(* 4a. residual orders: with a through tau^5 and varpi through tau^4 the
       constraint must vanish through tau^2, the varpi EOM through tau^3,
       and (Bianchi) the scale-factor EOM through its matching order.     *)
firstNonzero[expr_, name_, lo_, hi_] := Module[{c, m},
   For[m = lo, m <= hi, m++,
    c = Simplify[Together@SeriesCoefficient[expr, {tau, 0, m}],
        TimeConstraint -> 60];
    If[c =!= 0,
     Print[name, ": residual starts at tau^", m,
       "  (expected: first order beyond truncation)"];
     Return[m]]];
   Print[name, ": residual zero through tau^", hi]; hi + 1];

firstNonzero[CCs /. rules, "constraint      ", -2, 3];
firstNonzero[E1s /. rules, "varpi EOM       ", -1, 4];
firstNonzero[F2s /. rules, "scale-factor EOM", -3, 3];

(* 4b. cross-check the low orders against the validated Notebooktrt series
       (the cpseries/aseries used in Untitled1-Copy1.ipynb, cell 1)        *)
Print["w1 check (0 expected): ",
  Simplify[(pc[1] /. rules) + H0 OmM Sqrt[4 stigma - 1]/(8 Sqrt[mups2 OmR stigma])]];
Print["a2 check (0 expected): ",
  Simplify[(ac[2] /. rules) - H0^2 OmM/(4 alp mups2)]];
Print["w2 check (0 expected): ",
  Simplify[(pc[2] /. rules) -
    3 H0^2 (4 stigma - 1) (2 + stigma (-2 + 3 stigma)) OmM^2/
     (128 mups2 OmR (stigma - 1) stigma (8 stigma^2 + stigma - 3))]];
Print["a3 check (0 expected): ",
  Simplify[(ac[3] /. rules) -
    H0^3 Sqrt[stigma] Sqrt[4 stigma - 1] (-2 + stigma (5 + 24 stigma)) OmM^2/
     (64 alp mups2^(3/2) Sqrt[OmR] (stigma - 1) (8 stigma^2 + stigma - 3))]];

(* ---------------------------------------------------------------------------
   5.  Results
   -------------------------------------------------------------------------*)
Print["--- Section 5: results ---"];
{ScaleFactor1, ScaleFactor2, ScaleFactor3, ScaleFactor4, ScaleFactor5} =
  Table[ac[k] /. rules, {k, 1, 5}];
{VarPi1, VarPi2, VarPi3, VarPi4} = Table[pc[k] /. rules, {k, 1, 4}];

Print["VarPi3       = ", VarPi3];
Print["VarPi4       = ", VarPi4];
Print["ScaleFactor4 = ", ScaleFactor4];
Print["ScaleFactor5 = ", ScaleFactor5];
Print["(note: OmL first enters at this order -- ScaleFactor5/VarPi4 \
contain OmL; everything below them is radiation+matter only)"];

aseries[tau_]     = Sum[{ScaleFactor1, ScaleFactor2, ScaleFactor3,
     ScaleFactor4, ScaleFactor5}[[k]] tau^k, {k, 1, 5}];
varpiseries[tau_] = 1 + Sum[{VarPi1, VarPi2, VarPi3, VarPi4}[[k]] tau^k, {k, 1, 4}];
aseriesp[tau_]     = D[aseries[tau], tau];
varpiseriesp[tau_] = D[varpiseries[tau], tau];

resultrules = rules;

(* python export, same replacement conventions as the notebook: *)
topython = {OmL -> OmL, OmM -> omega_m, OmR -> OmR, alp -> a0,
    H0 -> h*100, Mp -> 1};
(* << Wolfpy`;  aseries[tau] /. topython // ToPythonString  *)

Print["Done.  Series available as aseries[tau], varpiseries[tau] \
(+ derivatives), coefficients as ScaleFactor1..5, VarPi1..4, all in \
resultrules."];
