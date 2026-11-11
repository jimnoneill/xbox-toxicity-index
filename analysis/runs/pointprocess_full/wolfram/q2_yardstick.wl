(* A size-free reading of the KS distance D.
   If the true rate of an interval model is (1 + eps) times the fitted rate, the rescaled intervals are exponential
   with rate (1 + eps) where the model expects rate 1. The KS distance between the two distributions is
   sup_x | Exp[-x] - Exp[-(1 + eps) x] |. Find it in closed form, its small-eps limit, and the eps equivalent to a given D. *)
f[x_, e_] := Exp[-x] - Exp[-(1 + e) x];
xs = x /. First[Solve[D[f[x, e], x] == 0 && x > 0, x, Reals] /. C[1] -> 0 // Quiet];
dOfEps = FullSimplify[f[x, e] /. x -> Log[1 + e]/e, e > 0];
ser = Normal[Series[dOfEps, {e, 0, 2}]];
epsOf[d_] := e /. FindRoot[(dOfEps) == d, {e, 0.1}];
{"argmax x" -> Log[1 + e]/e, "D(eps)" -> dOfEps, "small eps" -> ser,
 "eps equivalent to D = 0.024, 0.038, 0.047, 0.05, 0.10" -> (epsOf /@ {0.024, 0.038, 0.047, 0.05, 0.10}),
 "D for eps = 5, 10, 20, 30 percent" -> N[dOfEps /. e -> {0.05, 0.10, 0.20, 0.30}],
 "noise floor: E[D] with known parameters at n = 757, 2890" -> N[Sqrt[Pi/2] Log[2]/Sqrt[{757, 2890}]]}
