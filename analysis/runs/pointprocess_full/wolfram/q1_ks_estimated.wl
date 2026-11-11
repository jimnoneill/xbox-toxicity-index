(* Time-rescaling KS test when parameters are estimated.
   Known parameters: sqrt(n) (F_n - u) converges to a Brownian bridge, covariance min(s,t) - s t, and the 5 percent
   critical value of sqrt(n) D is 1.358.
   One estimated rate (the simplest fitted model, a constant-rate Poisson process): the limit loses the direction of
   the score, covariance min(s,t) - s t - phi(s) phi(t) with phi(u) = (1-u) Log[1-u]. *)
kolm[x_] := 1 - 2 Sum[(-1)^(k - 1) Exp[-2 k^2 x^2], {k, 1, 100}];
critKnown = x /. FindRoot[kolm[x] == 0.95, {x, 1.3}];
meanKnown = Sqrt[Pi/2] Log[2];
phi[u_] := (1 - u) Log[1 - u];
varKnown[u_] := u (1 - u);
varEst[u_] := u (1 - u) - phi[u]^2;
maxVarEst = NMaximize[{varEst[u], 0 < u < 1}, u];
(* Monte Carlo of the sup of the limiting process on a grid *)
SeedRandom[20261110];
m = 400; grid = N[Range[m]/(m + 1)];
cov = Table[Min[s, t] - s t - phi[s] phi[t], {s, grid}, {t, grid}];
L = CholeskyDecomposition[cov + 10^-10 IdentityMatrix[m]];
sups = Table[Max[Abs[RandomVariate[NormalDistribution[], m] . L]], {20000}];
critEst = Quantile[sups, 0.95];
(* direct finite-sample check: exponential sample, rate estimated, n = 2890 *)
n = 2890;
dstat[] := Module[{x = Sort[RandomVariate[ExponentialDistribution[1], n]], u},
   u = 1 - Exp[-x/Mean[x]]; Sqrt[n] Max[Max[Range[n]/n - u], Max[u - (Range[n] - 1)/n]]];
fin = Table[dstat[], {4000}];
{"critKnown" -> critKnown, "meanKnown sqrt(n) E[D]" -> N[meanKnown], "maxVar known" -> 0.25, "maxVar est" -> maxVarEst,
 "critEst (limit process, 5 pct)" -> critEst, "critEst finite n=2890" -> Quantile[fin, 0.95],
 "rejection rate of the 1.358 rule when one rate is estimated" -> N[Mean[Boole[# > critKnown] & /@ fin]],
 "median sqrt(n) D, estimated" -> Median[fin], "mean sup, estimated" -> Mean[sups]}
