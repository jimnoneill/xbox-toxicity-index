(* When does BIC prefer the larger model?
   The larger model has dk more parameters and gains g nats of log-likelihood per event, so its total gain is g n.
   BIC prefers it when 2 g n > dk Log[n]; AIC when g n > dk. Solve for the stream length at which BIC switches. *)
sol = Solve[2 g n == dk Log[n] && n > E, n, Reals] // Quiet;
nstar[g_, dk_] := -(dk/(2 g)) ProductLog[-1, -2 g/dk];
gmin[n_, dk_] := dk Log[n]/(2 n);
check = Table[{g, N[nstar[g, 5]], N[2 g nstar[g, 5] - 5 Log[nstar[g, 5]]]}, {g, {0.001, 0.002, 0.005, 0.01, 0.02}}];
{"closed form" -> HoldForm[n == -(dk/(2 g)) ProductLog[-1, -2 g/dk]],
 "largest gain per event for which no stream length suffices" -> "none: for any g > 0 a long enough stream switches",
 "switch length for dk = 5 at g = 0.001, 0.002, 0.005, 0.01, 0.02 nats per event" -> check,
 "gain per event needed at n = 2890 (dk = 5): BIC, AIC" -> {N[gmin[2890, 5]], N[5/2890]},
 "gain per event needed at n = 500, 2890, 20000 (dk = 5, BIC)" -> N[{gmin[500, 5], gmin[2890, 5], gmin[20000, 5]}],
 "total gain needed at n = 2890 (dk = 5): BIC, AIC" -> {N[5 Log[2890]/2], 5.}}
