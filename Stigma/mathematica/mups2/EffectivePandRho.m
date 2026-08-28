(* ::Package:: *)

Get@FileNameJoin@{DirectoryName[$InputFileName], "StigmaEquationsofMotion.m"}

DefScalarFunction[Tau, PrintAs -> "\[Tau]"];
DefConstantSymbol[alps, PrintAs -> "\[Alpha]"];
DefConstantSymbol[H0, 
  PrintAs -> "\(\*SubscriptBox[\(\[ScriptCapitalH]\), \(0\)]\)"];
DefConstantSymbol[OmegaM0, 
  PrintAs -> "\(\*SubscriptBox[\(\[CapitalOmega]\), \(m,0\)]\)"];
DefConstantSymbol[OmegaR0, 
  PrintAs -> "\(\*SubscriptBox[\(\[CapitalOmega]\), \(r,0\)]\)"];
DefConstantSymbol[OmegaL0, 
  PrintAs -> 
   "\(\*SubscriptBox[\(\[CapitalOmega]\), \(\[CapitalLambda],0\)]\)"];


MakeSeries[InputExpr_] := 
  Module[{rule, rule2, intermediate1, intermediate2}, 
   rule = {Varpis :> (Varpis[Tau[#]] &), 
     aa :> (alps aa[Tau[#]] &)};
   rule2 = {rhom :> (rhom[Tau[#]] &), rhor :> 
     (rhor[Tau[#]] &), rhoL :> (rhoL[Tau[#]] &)};
   intermediate1 = InputExpr /. rule /. rule2;
   intermediate2 = intermediate1 /. Tau''[t] -> 
     D[1/(alps aa[Tau[t]]), t] /. 
    Derivative[1][Tau][t] -> 1/(alps aa[Tau]) /. Tau[
     t] -> Tau;
   PowerExpand[FullSimplify[intermediate2]]];


ConstraintEqn = MakeSeries[EqnFinal];
ScaleEqn = MakeSeries[EqaFinal];
VarPiEqn = MakeSeries[EqVarpiFinal];

Print["Constraint Equation:", ConstraintEqn, "\n"];
Print["Scale Factor Equation:", ScaleEqn, "\n"];
Print["\nVarpi Equation:", VarPiEqn, "\n"];

(*------------------------------------------*)
(*  Define additional symbols *)
(*------------------------------------------*)

DefScalarFunction[rhomGR, PrintAs -> "\!\(\*SubscriptBox[\(\[Rho]\), \(m, GR\)]\)"];
DefScalarFunction[rhorGR, PrintAs -> "\!\(\*SubscriptBox[\(\[Rho]\), \(r, GR\)]\)"];
DefScalarFunction[rhoLGR, PrintAs -> "\!\(\*SubscriptBox[\(\[Rho]\), \(\[CapitalLambda], GR\)]\)"];
DefScalarFunction[rhoLeff, PrintAs -> "\!\(\*SubscriptBox[\(\[Rho]\), \(\[CapitalLambda], eff\)]\)"];
DefConstantSymbol[HH, PrintAs -> "\[ScriptCapitalH]"];
DefConstantSymbol[HH2, PrintAs -> "\!\(\*SuperscriptBox[\(\[ScriptCapitalH]\), \(2\)]\)"];
DefConstantSymbol[H00, PrintAs -> "\!\(\*SubscriptBox[\(H\), \(0\)]\)"];

(*------------------------------------------*)
(*  Replacement Rules for GR vs Stigma     *)
(*------------------------------------------*)

rhoGRrules = {
  rhom[Tau] :> rhomGR[Tau], 
  rhor[Tau] :> rhorGR[Tau], 
  rhoL[Tau] :> rhoLGR[Tau]
};

(*------------------------------------------*)
(*  GR Limit Analysis *)
(*------------------------------------------*)

F1 = ConstraintEqn;

F1GRtau = F1 /. Derivative[1][Varpis][Tau] -> 0 /. 
  Varpis[Tau] -> 1 /. 
  Derivative[1][aa][Tau] -> HH/alps /. 
  stigma -> 1 /. 
  (-mups2) -> -4/3 /. 
  HH^2 -> HH2 /. 
  alps -> 1 /. 
  rhoGRrules // FullSimplify;

F1GRH2 = HH2 /. Flatten[Solve[F1GRtau == 0, HH2]] /. 
  rhoLGR[Tau] -> rhoLeff[Tau] // FullSimplify;

Print["F1GRtau:", F1GRtau];
Print["F1GRH2:", F1GRH2];

(*------------------------------------------*)
(*  Define density and pressure rules       *)
(*------------------------------------------*)

(* Define pressure functions *)
DefScalarFunction[pm, PrintAs -> "\!\(\*SubscriptBox[\(p\), \(m\)]\)"];
DefScalarFunction[pr, PrintAs -> "\!\(\*SubscriptBox[\(p\), \(r\)]\)"];
DefScalarFunction[pL, PrintAs -> "\!\(\*SubscriptBox[\(p\), \(\[CapitalLambda]\)]\)"];
DefScalarFunction[pmGR, PrintAs -> "\!\(\*SubscriptBox[\(p\), \(m, GR\)]\)"];
DefScalarFunction[prGR, PrintAs -> "\!\(\*SubscriptBox[\(p\), \(r, GR\)]\)"];
DefScalarFunction[pLGR, PrintAs -> "\!\(\*SubscriptBox[\(p\), \(\[CapitalLambda], GR\)]\)"];

(* Density rules with our defined symbols *)
rhorules = {
  rhom[Tau] -> 3 Mp^2 H00^2 OmegaM0 (alps aa[Tau])^(-3), 
  rhor[Tau] -> 3 Mp^2 H00^2 OmegaR0 (alps aa[Tau])^(-4), 
  rhoL[Tau] -> 3 Mp^2 H00^2 OmegaL0
};

(* GR density rules *)
rhorulesGR = {
  rhomGR[Tau] -> 3 Mp^2 H00^2 OmegaM0 (aa[Tau])^(-3), 
  rhorGR[Tau] -> 3 Mp^2 H00^2 OmegaR0 (aa[Tau])^(-4), 
  rhoLGR[Tau] -> 3 Mp^2 H00^2 OmegaL0
};

(* Pressure rules *)
prules = {
  pm[Tau] -> 0, 
  pr[Tau] -> (1/3) rhor[Tau], 
  pL[Tau] -> -rhoL[Tau]
};

(* GR pressure rules *)
prulesGR = {
  pmGR[Tau] -> 0, 
  prGR[Tau] -> -(1/3) rhorGR[Tau], 
  pLGR[Tau] -> -rhoLGR[Tau]
};


(*------------------------------------------*)
(*  Full Stigma analysis and effective density *)
(*------------------------------------------*)

(* Expand the full constraint equation *)
F1expand = F1 // Expand;

(* Replace derivative terms with H^2 *)
F1tau = F1expand /. Derivative[1][aa][Tau]^2 -> HH2/alps^2 // FullSimplify;

(* Solve for H^2 in full Stigma theory *)
F1Htau = HH2 /. Flatten[Solve[F1tau == 0, HH2]] // FullSimplify;

(* Find effective dark energy density by equating GR and Stigma Hubble parameters *)
rhoe = rhoLeff[Tau] /. 
  Flatten[Solve[F1GRH2 == F1Htau, rhoLeff[Tau]]] /. 
  rhorules /. rhorulesGR // FullSimplify;

Print["Effective dark energy density (rhoe):", rhoe];

(*------------------------------------------*)
(*  Second derivatives and effective pressure *)
(*------------------------------------------*)

(* Solve for second derivatives *)
secdiv = Flatten[Solve[ScaleEqn == 0 && VarPiEqn == 0, {Derivative[2][aa][Tau], Derivative[2][Varpis][Tau]}]] // FullSimplify;

(* Define effective pressure function *)
DefScalarFunction[Peff, PrintAs -> "\!\(\*SubscriptBox[\(P\), \(eff\)]\)"];

(* Calculate effective pressure from continuity equation *)
Pefftau = Peff /. Flatten[Solve[D[rhoe, Tau] == -3 Derivative[1][aa][Tau]/aa[Tau] (rhoe + Peff), Peff]] /. secdiv /. prules /. rhorules // FullSimplify;

(* Expand and substitute to get final effective pressure *)
Peffexpand = Pefftau // Expand;
Pe = Peffexpand /. prules /. rhorules /. Derivative[1][aa][Tau]^2 -> HH2/alps^2 /. HH2 -> F1Htau /. prules /. rhorules // FullSimplify;

Print["Final effective pressure (Pe):", Pe];

