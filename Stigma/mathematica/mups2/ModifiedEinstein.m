(* ::Package:: *)

(*==============================================*)
(*  Modified Einstein Equations for Stigma    *)
(*==============================================*)

Print@"Loading Modified Einstein equations for Stigma Q-torsion theory...";

Get@FileNameJoin@{DirectoryName[$InputFileName],"StigmaEquationsOfMotion.m"};

DefConstantSymbol[varpiConst,PrintAs->"\[CurlyPi]"];

(* Compute F1 constant by setting varpi derivative to zero and using steady state assumptions *)
F1const = Eqn /. Derivative[1][Varpis][t] -> 0 /. (Derivative[1][aa][t] -> H[t] aa[t]) /. Varpis[t] -> varpiConst /. H[t] -> Sqrt[H2] // FullSimplify;

Print@"First Friedmann equation analogue with constant \[CurlyPi]:";
Print@(F1const);

(* GR limit for comparison *)
Print@"Constructing GR limit...";
F1GR = FullSimplify[F1const /. varpiConst -> 1] /. stigma -> 1 /. mups2 -> 4/3;
Print@"First Friedmann equation in GR limit:";
Print@(F1GR);

DefConstantSymbol[kappa,PrintAs->"\[Kappa]"];
Print@"Solving F1const == 0 for H2...";
F1H2 = H2 /. Flatten[Solve[F1const == 0, H2]] /. Mp -> 1/Sqrt[kappa];
Print@"F1H2 computed successfully";

DefConstantSymbol[gi,PrintAs->"\(\*SubscriptBox[\(g\), \(i\)]\)"];

Print@"Trying Solve with TimeConstrained on F1GR == 0 for H2...";

solveResult = Solve[F1GR == 0, H2]
F1GRH2 = H2 /. Flatten[solveResult] /. Mp -> 1/Sqrt[gi kappa];
Print@"F1GRH2 computed successfully";
Print@(F1GRH2);

(* Find the effective gravitational constant ratio *)
givalue = gi /. Flatten[Solve[F1H2 == F1GRH2, gi]];

Print@"Effective gravitational constant ratio G_eff/G_GR:";
Print@(givalue);

DefConstantSymbol[wi,PrintAs->"\(\*SubscriptBox[\(w\), \(i\)]\)"];

Print@"Substituting H as a function of wi"
Hi = 2/(3 (1 + wi) t);

HH[t_] := aa'[t]/aa[t];

Esolve = EqVarpiFinal /. Derivative[1][Varpis][t] -> 0 /. Derivative[2][Varpis][t] -> 0 /. Varpis[t] -> varpiConst /. Flatten[Solve[HH'[t] == D[Hi, t], Derivative[2][aa][t]]] /. Derivative[1][aa][t] -> Hi aa[t] /. varpiConst -> Sqrt[varpiConst2] // FullSimplify;

gisolution = givalue /. varpiConst -> Sqrt[varpiConst2] /. Solve[Esolve == 0, varpiConst2] // FullSimplify

Print@"Effective gravitational constant ratio G_eff/G_GR as a function of equation of state parameter wi:";
Print@(gisolution);

(* Define plotting functions for the non-infinite solutions *)
giup2minus[wi_, stigma_] := Evaluate[gisolution[[2]] * (-mups2) /. {wi -> wi, stigma -> stigma}];
giup2plus[wi_, stigma_] := Evaluate[gisolution[[3]] * (-mups2) /. {wi -> wi, stigma -> stigma}];

Print@"Creating density plots...";

plot1 = DensityPlot[giup2minus[wi, stigma], {wi, -1, 1}, {stigma, -1, 2}, 
 FrameLabel -> {"wi", "stigma"}, 
 ColorFunction -> "BlueGreenYellow", 
 PlotLegends -> BarLegend[Automatic, LegendLabel -> "gi mups2"],
 Epilog -> {Red, Dashed, Line[{{1/3, -1}, {1/3, 2}}]}, PlotLabel -> "gi minus solution"];

plot2 = DensityPlot[giup2plus[wi, stigma], {wi, -1, 1}, {stigma, -1, 2},
 FrameLabel -> {"wi", "stigma"},
 ColorFunction -> "BlueGreenYellow",
 PlotLegends -> BarLegend[Automatic, LegendLabel -> "gi mups2"], 
 Epilog -> {Red, Dashed, Line[{{1/3, -1}, {1/3, 2}}]}, PlotLabel -> "gi plus solution"];

(* Conditional display - only in notebook environment *)
If[$FrontEnd =!= Null,
    Print["Running in Notebook mode, displaying plots."];
    Print@plot1;
    Print@plot2;
    ,
    Print["Running in Terminal mode, skipping plot display."];
];

(* Always export to files *)
outputDir = FileNameJoin[{ParentDirectory[ParentDirectory[DirectoryName[$InputFileName]]], "latex"}];
If[!DirectoryQ[outputDir], CreateDirectory[outputDir]];

Export[FileNameJoin[{outputDir, "gisolution_minus.pdf"}], plot1];
Export[FileNameJoin[{outputDir, "gisolution_plus.pdf"}], plot2];

Print@"Plots saved to latex folder.";
