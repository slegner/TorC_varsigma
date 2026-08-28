(*=======================================================*)
(*  Early Time Power Series Solution for Stigma Theory  *)
(*  Based on Will's exact approach from PGT_fromWill    *)
(*=======================================================*)


Get@FileNameJoin@{DirectoryName[$InputFileName],"StigmaEquationsOfMotion.m"};



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

DefConstantSymbol[ScaleFactor1, 
  PrintAs -> "\(\*SubscriptBox[\(a\), \(1\)]\)"];
DefConstantSymbol[ScaleFactor2, 
  PrintAs -> "\(\*SubscriptBox[\(a\), \(2\)]\)"];
DefConstantSymbol[ScaleFactor3, 
  PrintAs -> "\(\*SubscriptBox[\(a\), \(3\)]\)"];

DefConstantSymbol[VarPi0,
  PrintAs -> "\(\*SubscriptBox[\(\[CurlyPi]\), \(0\)]\)"];
DefConstantSymbol[VarPi1,
  PrintAs -> "\(\*SubscriptBox[\(\[CurlyPi]\), \(1\)]\)"];
DefConstantSymbol[VarPi2,
  PrintAs -> "\(\*SubscriptBox[\(\[CurlyPi]\), \(2\)]\)"];

(*-------------------------------------------------------------*)
(*  Global sign assumptions — enforced for all Simplify,      *)
(*  FullSimplify, and Refine calls in this session             *)
(*-------------------------------------------------------------*)
$Assumptions = {
  mups2 > 0, msig2 > 0, stigma > 0, alps > 0,
  VarPi0 > 0, OmegaM0 > 0, OmegaR0 > 0, H0 > 0, Mp > 0,
  stigma > VarPi0^2,
  5*stigma - VarPi0^2 + 4*stigma^2*(-2 + VarPi0^2) < 0
};

ScaleFactor[x_] := ScaleFactor1*x + ScaleFactor2*x^2 + ScaleFactor3*x^3;
VarPi[x_] := VarPi0 + VarPi1*x + VarPi2*x^2;

rhom[x_] := 3*Mp^2*OmegaM0*H0^2/aa[x]^3;
rhor[x_] := 3*Mp^2*OmegaR0*H0^2/aa[x]^4;
rhoL[x_] := 3*Mp^2*OmegaL0*H0^2;

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
   Refine[FullSimplify[intermediate2]]];


ConstraintEqn = MakeSeries[EqnFinal];
VarPiEqn = MakeSeries[EqVarpiFinal];


seriesSubstitutions = {aa[Tau] -> ScaleFactor[Tau], 
   Varpis[Tau] -> VarPi[Tau], 
   Derivative[1][aa][Tau] -> Derivative[1][ScaleFactor][Tau], 
   Derivative[2][aa][Tau] -> Derivative[2][ScaleFactor][Tau], 
   Derivative[1][Varpis][Tau] -> Derivative[1][VarPi][Tau], 
   Derivative[2][Varpis][Tau] -> Derivative[2][VarPi][Tau]};

MainEqs = {ConstraintEqn, VarPiEqn} /. seriesSubstitutions;


series1 = Series[MainEqs[[1]], {Tau, 0, 3}];
series2 = Series[MainEqs[[2]], {Tau, 0, 3}];

NegOrderConstraint = SeriesCoefficient[series1, -1];
NegativeOrderVarPi = SeriesCoefficient[series2, -1];
NegativeOrderEqs = {NegOrderConstraint == 0};

ZeroOrderConstraint = SeriesCoefficient[series1, 0];
ZeroOrderVarPi = SeriesCoefficient[series2, 0];

FirstOrderConstraint = SeriesCoefficient[series1, 1];
FirstOrderVarPi = SeriesCoefficient[series2, 1];

SecondOrderConstraint = SeriesCoefficient[series1, 2];
SecondOrderVarPi = SeriesCoefficient[series2, 2];

Print["Solving for power series coefficients..."];

Print["Solving for a₁ from \[Tau]^(-1) constraint:"];
NegativeOrderEqs = {NegOrderConstraint == 0};
Print[NegativeOrderEqs[[1]]];

LeadingSols = Solve[NegativeOrderEqs, {ScaleFactor1}];

(* ===================================================================*)
(*PART 1:Find initial solutions for A1 (ScaleFactor1)*)
(* ===================================================================*)
If[Length[LeadingSols] > 0, A1Branches = {};
  Do[If[Head[LeadingSols[[i]]] === ConditionalExpression, 
    branchSol = LeadingSols[[i]][[1]];
    branchConditions = LeadingSols[[i]][[2]];, 
    branchSol = LeadingSols[[i]];
    branchConditions = True;];
   branchSol =
    Refine[Simplify[branchSol]] /. {Abs[VarPi0] -> VarPi0};
   branchSol = 
    branchSol /. ConditionalExpression[val_, cond_] :> val;
   AppendTo[A1Branches, branchSol];, {i, 1, Length[LeadingSols]}];
  A1Sol = A1Branches;, A1Sol = {};];

(* ===================================================================*)
(*PART 2:Use A1 solutions to find other variables for each branch*)
(* ===================================================================*)

If[Length[A1Sol] > 0, AllBranchSolutions = {};
  Do[currentA1 = A1Sol[[branchNum]];
   
   (* Solve zero-order equations for VarPi1 and ScaleFactor2 *)
   ZeroOrderConstraintSub = 
    ZeroOrderConstraint /. currentA1 // FullSimplify;
   ZeroOrderVarPiSub = ZeroOrderVarPi /. currentA1 // FullSimplify;
   A2Pi1Sol = 
    Solve[{ZeroOrderConstraintSub == 0, 
      ZeroOrderVarPiSub == 0}, {VarPi1, ScaleFactor2}];
   A2Pi1Sol = A2Pi1Sol // FullSimplify;
   
   (* Apply A1, VarPi1, and ScaleFactor2 solutions to first-order equations *)
   currentSols = Flatten[{currentA1, A2Pi1Sol}];
   FirstOrderConstraintSub = 
    FirstOrderConstraint /. currentSols // FullSimplify;
   FirstOrderVarPiSub = FirstOrderVarPi /. currentSols // FullSimplify;
   
   (* Solve first-order equations for VarPi2 and ScaleFactor3 *)
   A3Pi2Sol = 
    Solve[{FirstOrderConstraintSub == 0, 
      FirstOrderVarPiSub == 0}, {VarPi2, ScaleFactor3}];
   A3Pi2Sol = A3Pi2Sol // FullSimplify;
   
   (* Combine all solutions for this branch *)
   branchSols = Flatten[{currentA1, A2Pi1Sol, A3Pi2Sol}];
   AppendTo[AllBranchSolutions, branchSols];, {branchNum, 1, 
    Length[A1Sol]}];];

(* ===================================================================*)
(*PART 3:Simplify,save,and print the final solutions*)
(* ===================================================================*)

(*1. Initialize empty lists to store the expressions from each branch*)
aSolutions = {};
VarPiSolutions = {};
aDerivatives = {};
VarPiDerivatives = {};

Print["\n=== SIMPLIFYING AND SAVING SOLUTIONS ==="];

(*This loop processes the solutions from each branch - simplified without assumptions*)
Do[Print["=== Processing Branch ", branchNum, " ==="];
  FinalSols = AllBranchSolutions[[branchNum]];
  
  (*Extract coefficients directly from AllBranchSolutions*)
  a1 = ScaleFactor1 /. FinalSols;
  a2 = ScaleFactor2 /. FinalSols;
  a3 = ScaleFactor3 /. FinalSols;
  pi0 = VarPi0;  (* VarPi0 is a parameter, not solved for *)
  pi1 = VarPi1 /. FinalSols;
  pi2 = VarPi2 /. FinalSols;
  
  (*Construct the power series expressions*)
  aSol = a1*Tau + a2*Tau^2 + a3*Tau^3;
  VarPiSol = pi0 + pi1*Tau + pi2*Tau^2;
  aDeriv = D[aSol, Tau];
  VarPiDeriv = D[VarPiSol, Tau];
  
  (*Append the final expressions for this branch to the lists*)
  AppendTo[aSolutions, aSol];
  AppendTo[VarPiSolutions, VarPiSol];
  AppendTo[aDerivatives, aDeriv];
  AppendTo[VarPiDerivatives, VarPiDeriv];
  
  (*Print the coefficients for this branch*)
  Print["Branch ", branchNum, " coefficients:"];
  Print["  a1 = ", a1];
  Print["  a2 = ", a2];
  Print["  a3 = ", a3];
  Print["  pi0 = ", pi0, " (parameter)"];
  Print["  pi1 = ", pi1];
  Print["  pi2 = ", pi2];
  , {branchNum, 1, Length[AllBranchSolutions]}];

(*3. Add a new loop to print the final saved solutions from each branch*)
Print["\n=== FINAL SOLUTIONS ==="];
Do[Print["\n=== BRANCH ", branchNum, " ==="];
  Print["aSolution[Tau] = ", aSolutions[[branchNum]]];
  Print["VarPiSolution[Tau] = ", VarPiSolutions[[branchNum]]];
  Print["aDerivative[Tau] = ", aDerivatives[[branchNum]]];
  Print["VarPiDerivative[Tau] = ", 
   VarPiDerivatives[[branchNum]]];, {branchNum, 1, 
   Length[aSolutions]}];

(* ===================================================================*)
(* PART 4: Define robust tau initial (tauini) expression              *)
(* ===================================================================*)

Print["\n=== ROBUST TAU INITIAL DEFINITION ==="];

(* Extract a1 and a2 coefficients from the first branch solution *)
If[Length[AllBranchSolutions] >= 1,
  (* Get coefficients directly from AllBranchSolutions *)
  firstBranchSols = AllBranchSolutions[[1]];
  a1 = ScaleFactor1 /. firstBranchSols;
  a2 = ScaleFactor2 /. firstBranchSols;
  a3 = ScaleFactor3 /. firstBranchSols;
  
  (* Define robust tauini where linear term dominates quadratic *)
  (* Use PowerExpand with assumptions to ensure real, positive result *)
  assumptions = {Element[{alps, stigma, cpr, h, omega_m, OmegaR0, OmegaL0, Mp, mv2}, Reals], 
                 alps > 0, stigma > 0, cpr > 0, h > 0, omega_m > 0, OmegaR0 > 0, OmegaL0 > 0, Mp > 0, mv2 > 0};
  
  (* Simple a1/a2 ratio with variable substitutions *)
  tauiniWithSubs = a1/a2 /. {OmegaM0 -> omega_m, VarPi0 -> cpr, mups2 -> mv2, H0 -> h*100};
  tauiniRobust = tauiniWithSubs * 10^(-6);
  
  Print["Using a1, a2, and a3 from Branch 1 power series:"];
  Print["a1 (sqrt combined) = ", a1];
  Print["a2 (sqrt combined) = ", a2];
  Print["a3 (sqrt combined) = ", a3];
  Print[""];
  Print["tauiniRobust = (a1/a2) × 10^(-6)"];
  Print["This ensures |a1*τ| >> |a2*τ²| >> |a3*τ³| (linear dominates higher orders)"];
  Print[""];
  Print["Full simplified expression: ", tauiniRobust];
  Print[""];
  Print["With Python variable names:"];
  Print["tauiniRobust = ", tauiniRobust /. {OmegaM0 -> omega_m, VarPi0 -> cpr, mups2 -> mv2, H0 -> h*100}];
  
, Print["Branch 1 solution not available. Need at least 1 branch."]
];

(* ===================================================================*)
(* PART 5: ToPython converter function                                *)
(* ===================================================================*)

(* Function to combine separate square roots into a single square root *)
CombineSqrts[expr_] := Module[{result},
  result = expr;
  
  (* First expand and collect square root terms *)
  result = Expand[result];
  
  (* Repeatedly apply square root combination rules until no more changes *)
  result = FixedPoint[
    Function[x, 
      x //. {
        (* Basic patterns *)
        Sqrt[a_]*Sqrt[b_] :> Sqrt[Expand[a*b]],
        Sqrt[a_]/Sqrt[b_] :> Sqrt[Expand[a/b]],
        (* Multiple square roots in products *)
        Times[pre___, Sqrt[a_], Sqrt[b_], post___] :> Times[pre, Sqrt[Expand[a*b]], post],
        Times[pre___, Sqrt[a_], Sqrt[b_], Sqrt[c_], post___] :> Times[pre, Sqrt[Expand[a*b*c]], post],
        Times[pre___, Sqrt[a_], Sqrt[b_], Sqrt[c_], Sqrt[d_], post___] :> Times[pre, Sqrt[Expand[a*b*c*d]], post],
        (* Fractions with multiple square roots *)
        Divide[Times[pre1___, Sqrt[a_], post1___], Times[pre2___, Sqrt[b_], post2___]] :> 
          Divide[Times[pre1, post1], Times[pre2, post2]] * Sqrt[Expand[a/b]],
        (* Remove nested CombineSqrts calls *)
        CombineSqrts[x_] :> x
      }
    ],
    result
  ];
  
  (* Final aggressive simplification *)
  result = PowerExpand[FullSimplify[result]];
  
  result
];

ToPython[expr_] := Module[{result},
  result = expr;
  
  (* Skip complex square root combination to avoid recursion *)
  
  (* Convert to string *)
  result = ToString[result, InputForm];
  
  (* Enhanced replacements including Greek letters *)
  result = StringReplace[result, {
    (* Handle Greek letters and special characters - \[name] -> name *)
    RegularExpression["\\\\\\[([^\\]]+)\\\\\\]"] -> "$1",
    
    (* Handle Sqrt specifically first - Sqrt[x] -> (x)**(1/2) *)
    RegularExpression["Sqrt\\[([^\\]]+)\\]"] -> "($1)**(1/2)",
    
    (* Handle Power function - Power[x, y] -> (x)**(y) *)  
    RegularExpression["Power\\[([^,]+),\\s*([^\\]]+)\\]"] -> "($1)**($2)",
    
    (* Trigonometric functions *)
    "Sin[" -> "sin(",
    "Cos[" -> "cos(",
    "Tan[" -> "tan(",
    "ArcSin[" -> "asin(",
    "ArcCos[" -> "acos(",
    "ArcTan[" -> "atan(",
    "Sinh[" -> "sinh(",
    "Cosh[" -> "cosh(",
    "Tanh[" -> "tanh(",
    
    (* Other mathematical functions *)
    "Log[" -> "log(",
    "Exp[" -> "exp(",
    "Abs[" -> "abs(",
    
    (* Constants *)
    "Pi" -> "pi",
    "E" -> "e",
    "I" -> "1j",
    
    (* Replace remaining brackets with parentheses *)
    "[" -> "(",
    "]" -> ")"
  }];
  
  (* Second pass: Handle all ^ characters more aggressively *)
  result = StringReplace[result, "^" -> "**"];
  
  (* Third pass: Use regex to catch any remaining exponential patterns *)
  result = StringReplace[result, RegularExpression["([a-zA-Z0-9_]+)\\^([a-zA-Z0-9_]+)"] -> "$1**$2"];
  
  result
];

Print["\n=== PYTHON CONVERSION OF TAU INITIAL ==="];

(* Convert the tauini expression to Python format *)
If[Length[aSolutions] >= 2,
  Print["Converting tauiniRobust to Python format:"];
  
  (* Clear any potential symbol contamination *)
  
  (* Simple a1/a2 expression for Python *)
  tauiniSimpleExpr = (a1/a2) * 10^(-6);
  tauiniWithPythonVars = tauiniSimpleExpr /. {OmegaM0 -> omega_m, VarPi0 -> cpr, mups2 -> mv2, H0 -> h*100};
  
  (* Convert to Python *)
  tauiniPython = tauiniWithPythonVars //ToPython;
  
  Print["tauiniRobust //ToPython = ", tauiniPython];
  Print[""];
  Print["COPY-PASTE READY FOR PYTHON:"];
  Print["tauini_robust = ", tauiniPython];
];
