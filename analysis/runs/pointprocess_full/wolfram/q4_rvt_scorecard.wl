(* Re-analysis of stored test results. Inputs are aggregate estimates and standard errors only. *)
b = {0.09486133363146312, 0.04698824749997441, 0.24183999191973884, 0.12354240265513214, 0.2490472839469133};
s = {0.01011990793350723, 0.010220759365283636, 0.007906271889043913, 0.010211830070918535, 0.00860780413561371};
w = 1/s^2;
q[idx_] := Module[{ww = w[[idx]], bb = b[[idx]], m}, m = Total[ww bb]/Total[ww]; Total[ww (bb - m)^2]];
pool[idx_] := {Total[w[[idx]] b[[idx]]]/Total[w[[idx]]], 1/Sqrt[Total[w[[idx]]]]};
pres = {3, 5}; cham = {1, 2, 4};
qTot = q[Range[5]]; qPres = q[pres]; qCham = q[cham]; qWithin = qPres + qCham; qBetween = qTot - qWithin;
pv[x_, df_] := N[SurvivalFunction[ChiSquareDistribution[df], x], 6];
Print["Q total (df 4): ", qTot, "  p ", pv[qTot, 4]];
Print["Q between institution lost (df 1): ", qBetween, "  share of Q ", 100 qBetween/qTot];
Print["Q within presidency losses (df 1): ", qPres, "  p ", pv[qPres, 1]];
Print["Q within chamber losses (df 2): ", qCham, "  p ", pv[qCham, 2], "  I2 ", 100 (qCham - 2)/qCham];
Print["Q within, both (df 3): ", qWithin, "  p ", pv[qWithin, 3]];
Print["pooled presidency: ", pool[pres], "  pooled chamber: ", pool[cham]];
Print["ratio presidency/chamber: ", pool[pres][[1]]/pool[cham][[1]]];
(* chamber losses with a between-year shock variance taken from years without a change (placebo variance) *)
tau2 = 0.0010098404360146653;
w2 = 1/(s[[cham]]^2 + tau2); m2 = Total[w2 b[[cham]]]/Total[w2]; q2 = Total[w2 (b[[cham]] - m2)^2];
Print["chamber losses allowing placebo-year variance: Q ", q2, " (df 2)  p ", pv[q2, 2]];

(* steps in the losing party's daily series at the three dates on which control changed, and at the 2020 election *)
st = {0.03554644518713992, 0.040165530119182224, 0.04378492703221707}; se = {0.0108792427256351, 0.013251222518835133, 0.013415188050713846};
ws = 1/se^2; ms = Total[ws st]/Total[ws];
Print["losing-party steps pooled: ", ms, "  se ", 1/Sqrt[Total[ws]], "  Q ", Total[ws (st - ms)^2], "  p ", pv[Total[ws (st - ms)^2], 2]];
p3 = {6/51, 4/51, 2/51}; p4 = Append[p3, 35/51];
fisher[p_] := {-2 Total[Log[p]], pv[-2 Total[Log[p]], 2 Length[p]]};
stouffer[p_] := N[1 - CDF[NormalDistribution[], Total[InverseCDF[NormalDistribution[], 1 - p]]/Sqrt[Length[p]]], 6];
Print["placebo p values (one-sided, 50 placebo dates): ", N[p4]];
Print["Fisher, three change dates: ", N[fisher[p3]], "   Stouffer ", stouffer[p3]];
Print["Fisher, four prespecified losing-party series: ", N[fisher[p4]], "   Stouffer ", stouffer[p4]];
Print["chance that all three steps are positive under a fair coin: ", N[1/8]];

(* bound on the history effect relative to the size of the response itself *)
lo = -0.033; hi = 0.021; re = 0.15141443073095606;
Print["largest history effect not excluded, as share of the mean response: ", 100 Max[Abs[lo], Abs[hi]]/re, " percent; of a presidency loss: ", 100 Max[Abs[lo], Abs[hi]]/pool[pres][[1]]];

(* resource combinations: R2 of additive and tree models under the two hold-out schemes *)
Print["member hold-out R2 additive, trees: ", {0.48490083475857526, 0.5252957574333958}, "  new-change hold-out: ", {0.23578948223620055, 0.17867661387292733}];
