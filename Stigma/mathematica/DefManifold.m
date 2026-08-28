(* ::Package:: *)

(*===============*)
(*  DefManifold  *)
(*===============*)

DefManifold[M,4,IndexRange[{a,n}]];
AddIndices[TangentM,{a1,b1,c1,d1,e1,f1,g1,h1,i1,j1,k1,l1,n1,m1,o1,p1,q1,r1,s1,t1,u1,v1,w1,x1,y1,z1}];

(*-----------------------------------------------------------------------*)
(*  Relabeling of the indices so that we can type Roman and look Greek!  *)
(*-----------------------------------------------------------------------*)

StandardIndices=ToString/@Alphabet[];
ExtendedIndices=ToExpression@(ToString@#<>"1")&/@StandardIndices;

GeoStandardIndicesSymb=(ToString@#)&/@Evaluate@((#[[2]])&/@{
	{a,"\[Alpha]"},
	{b,"\[Beta]"},
	{c,"\[Chi]"},
	{d,"\[Delta]"},
	{e,"\[Epsilon]"},
	{f,"\[Phi]"},
	{g,"\[Gamma]"},
	{h,"\[Eta]"},
	{i,"\[Iota]"},
	{j,"\[Theta]"},
	{k,"\[Kappa]"},
	{l,"\[Lambda]"},
	{m,"\[Mu]"},
	{n,"\[Nu]"},
	{o,"\[Omicron]"},
	{p,"\[Pi]"},
	{q,"\[Omega]"},
	{r,"\[Rho]"},
	{s,"\[Sigma]"},
	{t,"\[Tau]"},
	{u,"\[Upsilon]"},
	{v,"\[Psi]"},
	{w,"\[Omega]"},
	{x,"\[Xi]"},
	{y,"\[CurlyPhi]"},
	{z,"\[Zeta]"}});
GeoExtendedIndicesSymb=ToString@ToExpression@(ToString@#<>"'")&/@GeoStandardIndicesSymb;

(PrintAs@Evaluate@#1^=Evaluate@#2)&~MapThread~{ToExpression/@StandardIndices,GeoStandardIndicesSymb};
(PrintAs@Evaluate@#1^=Evaluate@#2)&~MapThread~{ToExpression/@ExtendedIndices,GeoExtendedIndicesSymb};

(*----------------------------------------------------*)
(*  Constants and functions used in Stigma theory    *)
(*----------------------------------------------------*)

DefConstantSymbol[Mp,PrintAs->"\(\*SubscriptBox[\(\[ScriptCapitalM]\), \(p\)]\)"];

DefConstantSymbol[stigma,PrintAs->"\[Stigma]"];
DefConstantSymbol[ups2,PrintAs->"\(\*SubscriptBox[\(\[Upsilon]\), \(2\)]\)"];
DefConstantSymbol[ups1,PrintAs->"\(\*SubscriptBox[\(\[Upsilon]\), \(1\)]\)"];
DefConstantSymbol[sig1,PrintAs->"\(\*SubscriptBox[\(\[Sigma]\), \(1\)]\)"];
DefConstantSymbol[sig2,PrintAs->"\(\*SubscriptBox[\(\[Sigma]\), \(2\)]\)"];

Print@"We will define the conditions which restrict to the theory of interest:";

ToTheory={ups1->0};

Print@ToTheory;
