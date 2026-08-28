(* ::Package:: *)

(*=======================================================*)
(*  Extract Differential Equations for Stigma Theory    *)
(*  Solves constraint equations for second derivatives  *)
(*=======================================================*)

Get@FileNameJoin@{DirectoryName[$InputFileName],"StigmaEquationsOfMotion.m"};

Print["=== EXTRACTING DIFFERENTIAL EQUATIONS ==="];

(* Solve the constraint equations for the second derivatives *)
Print["Solving constraint equations for aa''[\[Tau]] and \[CurlyPi]''[\[Tau]]..."];

Diffs = Flatten[
    Solve[{F2a == 0, 
      E1a == 0}, {aa''[\[Tau]], \[CurlyPi]''[\[Tau]]}]] // 
   FullSimplify;

Print["=== DIFFERENTIAL EQUATIONS ==="];
Print["aa''[\[Tau]] = ", aa''[\[Tau]] /. Diffs];
Print["\[CurlyPi]''[\[Tau]] = ", \[CurlyPi]''[\[Tau]] /. Diffs];

Print["\n=== COMPLETE DIFFERENTIAL EQUATION SYSTEM ==="];
Print["The differential equations are:"];
Do[
  Print[Diffs[[i]]], 
  {i, 1, Length[Diffs]}
];