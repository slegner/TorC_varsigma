(* ::Package:: *)

(*==========================================================*)
(*  Modified Friedmann Equation with Effective kappa gi   *)
(*==========================================================*)

Print@"Loading Modified Friedmann equation with effective gravitational coupling...";

(* Load the gi definition from ModifiedEinstein.m *)
Get@FileNameJoin@{DirectoryName[$InputFileName],"ModifiedEinstein.m"};

Print@"Constructing modified Friedmann equation with effective kappa...";

(* Define the modified gravitational constant *)
kappaEff = gi * kappa;

Print@"Effective gravitational constant: \[Kappa]_eff = g_i * \[Kappa] where \[Kappa] = 8\[Pi]G/c^4";
Print@"gi = G_eff/G_GR = " <> ToString[givalue];

(* Standard Friedmann equation: H^2 = (8πG/3c^4) * ρ = (κ/3) * ρ *)
(* Modified version: H^2 = (κ_eff/3) * ρ = (gi * κ/3) * ρ *)

Print@"Standard Friedmann equation: H^2 = (\[Kappa]/3) \[Rho]";
Print@"Modified Friedmann equation: H^2 = (\[Kappa]_eff/3) \[Rho] = (g_i \[Kappa]/3) \[Rho]";

(* Define the modified Friedmann equation *)
FriedmannModified[rho_] := (gi * kappa * rho)/3;

Print@"Where the effective gravitational parameter gi depends on:";
Print@"- Equation of state parameter wi";
Print@"- Stigma coupling parameter stigma"; 
Print@"- Upsilon parameter (-mups2)";

(* Express in terms of Hubble parameter *)
HubbleModified[rho_] := Sqrt[FriedmannModified[rho]];

Print@"Modified Hubble parameter: H = Sqrt[(g_i \[Kappa]/3) \[Rho]]";

(* For comparison with standard cosmology *)
HubbleStandard[rho_] := Sqrt[(kappa * rho)/3];

Print@"Ratio of modified to standard Hubble parameter:";
Print@"H_modified/H_standard = Sqrt[g_i]";

(* The gi solutions from ModifiedEinstein.m are available as gisolution *)
Print@"The gi parameter has multiple solutions depending on the field configuration:";
Print@"gi solutions = " <> ToString[gisolution];

Print@"Modified Friedmann equation construction complete.";