(* ::Package:: *)


<<xAct`xTensor`;
<<xAct`xPerm`;
<<xAct`xTras`;

(*-------------------------------------*)
(*  Definition of Riemannian manifold  *)
(*-------------------------------------*)

Get@FileNameJoin@{DirectoryName[$InputFileName],"DefManifold.m"};

(*------------------*)
(*  FLRW spacetime  *)
(*------------------*)

DefChart[ch, M, {0, 1, 2, 3}, {t[], x[], y[], z[]}, ChartColor -> Purple]

DefScalarFunction[nn,PrintAs->"N"];
DefScalarFunction[aa,PrintAs->"a"];

metricmatrix = {{nn[t[]]^2, 0, 0, 0}, {0, -aa[t[]]^2, 0, 0}, {0, 0, -aa[t[]]^2, 0}, {0, 0, 0, -aa[t[]]^2}};
metric = CTensor[metricmatrix, {-ch, -ch}]
SetCMetric[metric, ch, SignatureOfMetric -> {1, 3, 0}]
cd = CovDOfMetric[metric]
detg = Det[metricmatrix]
Riemanncd = Riemann[cd];
Riccicd = Ricci[cd];
RicciScalarcd = RicciScalar[cd];

Print@(RicciScalarcd//Simplify);

DefScalarFunction[Psis,PrintAs->"\[Psi]"];
DefScalarFunction[Phis,PrintAs->"\[Phi]"];
DefScalarFunction[Varpis,PrintAs->"\[CurlyPi]"];
DefScalarFunction[Rhos,PrintAs->"\[Rho]"];

phi=CTensor[Phis[t[]],{}];
psi = CTensor[Psis[t[]],{}];
varpi = CTensor[Varpis[t[]],{}];

Xphi=1/2 cd[-a]@phi[] cd[a]@phi[]//ContractBasis//Simplification;
Xpsi = 1/2 cd[-a]@psi[] cd[a]@psi[]//ContractBasis //FullSimplify;

DefTensor[J[a], M];

J[a_] := 4 sig1 psi[]^3 cd[a]@(phi[]/psi[]) -Mp^2 (-mups2) cd[a]@phi[];
JJ = J[-a] J[a] //ContractBasis//FullSimplify;

sqJJ = Sqrt[JJ] // PowerExpand;
 

L=Sqrt[-detg]((1/2 Mp^2 (-mups2)-1/2 (-msig2) psi[]^2)RicciScalarcd[] - 12(1/2 (-msig2) Xpsi)+sqJJ+3/4 Mp^2 ((-mups2) phi[]^2 +4 ups1 psi[]^2)-3/2 2 (-msig2) phi[]^2 psi[]^2);

Print@"The full Lagrangian we are working with is"
Print@(L);

LTemp := FullSimplify[L /. {t[] -> t}, Assumptions -> {aa[t] > 0, nn[t] > 0}];

Lnew := FullSimplify[LTemp /. {sig1 -> (-msig2) stigma, ups1 -> 0, 
  Psis[t] -> (Mp Sqrt[(-mups2)] Varpis[t])/Sqrt[4 (-msig2) stigma],
  Derivative[1][Psis][t] -> (Mp Sqrt[(-mups2)] Derivative[1][Varpis][t])/Sqrt[4 (-msig2) stigma]}, 
  Assumptions -> {aa[t] > 0, nn[t] > 0}];

Print@"With the substitution for psi[] and sig1, the Lagrangian becomes"
Print@(Lnew);

Eqn = Sum[(-1)^it D[D[Lnew, Derivative[it][nn][t]], {t, it}], {it, 0, 2}] - Rhos[t] aa[t]^3 // FullSimplify;
Eqa = Sum[(-1)^it D[D[Lnew, Derivative[it][aa][t]], {t, it}], {it, 0, 2}] + 3 P[t] aa[t]^2 // FullSimplify;
EqPhi = Sum[(-1)^it D[D[Lnew, Derivative[it][Phis][t]], {t, it}], {it, 0, 2}] // FullSimplify;
EqVarpi = Sum[(-1)^it D[D[Lnew, Derivative[it][Varpis][t]], {t, it}], {it, 0, 2}] // FullSimplify;

nn[t_] := 1;
DefScalarFunction[rhom,PrintAs->"\(\*SubscriptBox[\(\[Rho]\), \(m\)]\)"];
DefScalarFunction[rhor,PrintAs->"\(\*SubscriptBox[\(\[Rho]\), \(r\)]\)"];
DefScalarFunction[rhoL,PrintAs->"\(\*SubscriptBox[\(\[Rho]\), \(\[CapitalLambda]\)]\)"];
Rhos[t_] := rhom[t] + rhor[t] + rhoL[t];
P[t_] := 1/3 rhor[t] - rhoL[t];

Phis[t_] = Phis[t] /. Flatten[DSolve[EqPhi == 0, Phis[t], t]];

EqnFinal = Eqn // FullSimplify;
EqaFinal = Eqa // FullSimplify;
EqVarpiFinal = EqVarpi // FullSimplify;

Print@"After applying constraints and solving phi equation:";
Print@"Final Equation for N (lapse function):";
Print@(EqnFinal);
Print@"Final Equation for a (scale factor):";
Print@(EqaFinal);
Print@"Final Equation for Varpi field:";
Print@(EqVarpiFinal);
