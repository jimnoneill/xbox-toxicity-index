# RVT-A DESIGN (frozen 2026-10-13 before any H1/H2 quantity was estimated). Never edited afterward; deviations go in the report.

## Disclosure
Seen before freezing: the earlier Kistner opposition-effect results (association ~ +0.10 share/yr out of presidency; forecast skill),
and member-year counts per year of the Twitter panel (n>=50): 2010 214, 2011 364, 2012 403, 2013 458, 2014 464, 2015 471, 2016 466,
2017 491, 2018 496, 2019 513, 2020 511, 2021 527, 2022 521 (855 members, 5,899 member-years). NOT seen: any share by experience group,
any transition-specific estimate, any veteran/newcomer comparison. Code: scripts/v7/rvt_a_*.py, outputs output/v7/rvt_a/.

## Data and common definitions
Panel: xbox.kistner_ext.build_panel(("Twitter",)), member-years 2010-2022 with n>=50, share = NegativePartisan share. Chamber from the
panel (Voteview join). Service history, first-congress start, party-at-time, |NOMINATE dim1| from data/voteview_all_members.csv
(House/Senate rows; start_year(c)=1789+2(c-1); a member served calendar years start..start+1 of each congress listed).
Seniority at T = T - start_year(first congress ever listed) (Voteview, any chamber). Panel-based first observed year is a sensitivity.
|nominate_dim1| = value in the congress of year T-1 (else nearest other congress of the member; members with none are dropped and counted).
Power tables (calendar year, used also for pre-2010): President D 1993-2000, R 2001-08, D 2009-16, R 2017-20, D 2021-22 (R 1989-92);
House D to 1994, R 1995-2006, D 2007-10, R 2011-18, D 2019-22; Senate D to 1994, R 1995-2000, D 2001-02 (107th contested; coded D, flagged),
R 2003-06, D 2007-14, R 2015-20, D 2021-22. "Out of presidency" = own party != President; "out of own chamber" = own party != majority of the
chamber the member served in during that congress.
Primary outcome (all hypotheses): D1 = share_T - share_{T-1} (both years n>=50). Robustness: D2 = mean(T,T+1) - mean(T-2,T-1) (>=1 year per side).
Equivalence margin (for "supports fixed landscape"): +/-0.03 share (30% of the established ~0.10 opposition effect). "Supports recursive"
requires the CI to exclude 0 AND |estimate| >= 0.02.
Inference toolkit: (i) member-clustered OLS SEs; (ii) wild cluster bootstrap by member, Rademacher, null-imposed (restricted) residuals,
B=1999, seed 20261013; (iii) party-year-cell inference as specified per hypothesis (cell-level meta-analysis / placebo-year calibration);
(iv) leave-one-year-out (or leave-one-transition-out) stability. Two-sided 95%.

## H1a Path dependence
Transitions and losers: 2011 House Democrats; 2015 Senate Democrats; 2017 all Democrats; 2019 House Republicans; 2021 all Republicans.
Sample: losers with D1 defined at T. Outcome: D1.
Key regressor (PRIMARY) E_match: served >=1 calendar year earlier than T (>=1989) in which own party lacked the body lost at T
(presidency for 2017/2021; own-chamber majority, chamber of that congress, for 2011/2015/2019). Voteview-defined (includes pre-2010 service).
Secondary regressors: E_any (lacked presidency or own-chamber majority in any earlier year); dose = number of such earlier matched years;
E_panel = same as E_match but counting only years 2010..T-1 in which the member is in the n>=50 panel (expected: little variation, reported).
Controls (PRIMARY model): seniority (linear+quadratic), pre-level share_{T-1}, |nominate dim1|, Senate indicator, transition FE.
Models: M1 pooled (T FE); M2 Democrats only; M3 Republicans only; M4 pooled without pre-level control.
Separating seniority from experience (stated honestly): within a transition E_match is a deterministic step in first-congress start year,
hence in seniority; it is not separable from seniority by cell comparison. Identification then comes only from (a) differing step locations
across transitions under a common smooth seniority function (checked by the quadratic and by a cubic variant), and (b) a local-cohort
comparison: per transition, members whose first-congress start is within 4 years either side of the step (L_T = last matched opposition year),
E coefficient with linear seniority, pooled with T FE.
Placebo-cutoff test: re-estimate the primary coefficient with the E step shifted by k years (k in -8..+8, excluding |k|<=1); report where the
true-cutoff estimate falls in this placebo distribution (rank). A true-experience effect should stand out; a pure seniority effect will not.
Cell-level inference: per-transition coefficient (same controls except T FE) combined by DerSimonian-Laird random-effects meta-analysis with
Cochran Q; leave-one-transition-out of the pooled coefficient.
Decision: RECURSIVE if pooled M1 CI excludes 0 under member-cluster AND wild bootstrap AND random-effects meta CI, |b|>=0.02, same sign in
both parties, sign stable under leave-one-transition-out, AND the true cutoff estimate lies outside the central 90% of the placebo-cutoff
distribution. FIXED if M1 90% CIs under all three inference methods lie inside +/-0.03. Otherwise CANNOT DISTINGUISH.
Report minimum detectable effect (2.8*SE).

## H1b Hysteresis
Windows A = 2015-16, B = 2021-22. Member-window level = mean share over window years with n>=50 (>=1 year).
Veterans: observed in A and in B, first-congress start <= 2011. Newcomers-A: first-congress start in {2013, 2015}, observed in A.
Newcomers-B: first-congress start in {2019, 2021}, observed in B (lack the 2017-20 spell in their role; 2019 entrants partly exposed).
Estimands, by party and pooled (party x window and party x veteran fixed effects), stacked member-window regression
y = a + b1 Vet + b2 B + tau (Vet x B) + g'X + e, X = |nominate dim1|, Senate, log n; cluster by member.
 (1) Veteran own change (B-A) mean (descriptive). (2) Level gap vets-newcomers in B. (3) PRIMARY tau = (vetB - vetA) - (newB - newA)
 (difference-in-differences: veteran change net of common drift as measured by newcomer cohorts facing the same regimes).
Assumptions: (a) the cohort/selection difference between veterans and newcomers is stable across windows A and B (placebo check below);
(b) the A and B regimes differ in chamber control (Ds: A = President in, House+Senate out; B = all in; Rs the mirror), so the regime is NOT
identical across windows; the DiD handles this only to the extent veterans and newcomers respond to a regime alike, which a fixed landscape
implies; (c) no differential selection of who stays. What counts: tau != 0 stable across leave-one-year-out and placebo. What does not count:
a veteran change alone, or a level gap alone (confounded by cohort and seniority).
Placebo DiDs (no intervening opposite spell): P1 windows 2011-12 vs 2013-14 (vets start<=2009; newcomers start 2011 / 2013), P2 windows
2017-18 vs 2019-20 (vets start<=2015; newcomers start 2017 / 2019). Their spread calibrates cohort-instability.
Leave-one-year-out: all four single-year combos (2015|2016) x (2021|2022). Inference: member-cluster, wild bootstrap by member, plus the two
party estimates reported separately as the cell-level check.
Decision: RECURSIVE if pooled tau CI excludes 0 (cluster and wild), |tau|>=0.02, same sign in both parties, all four leave-one-year-out signs
agree, and |tau| exceeds the largest |placebo tau| (pooled). FIXED if pooled 90% CIs (both methods) within +/-0.03 and placebo taus within
+/-0.03. Otherwise CANNOT DISTINGUISH.

## H2 Heterogeneous transition effects
For each transition T in {2011,2015,2017,2019,2021}: affected sample = members of the affected chamber(s) (2011 House, 2015 Senate, 2019 House,
2017 and 2021 both chambers), D1 outcome. PRIMARY estimator E2: beta_T = mean D1 (loser party) - mean D1 (gaining party) in the affected sample,
via OLS D1 ~ Loser + Senate (chamber control where both chambers present); positive = losers become more negative relative to gainers.
Member FE are implicit in the first difference. Secondary E1: loser-only mean D1 (no control). Secondary E3 (chamber transitions): loser party in
affected chamber vs same party in unaffected chamber. Robustness: D2 windows.
Homogeneity: Cochran Q and I^2 across the five E2 estimates (SE from member-cluster); placebo-year calibration: the same contrast (D minus R,
within the same chamber set) for placebo years T in {2012,2013,2014,2016,2018,2020,2022} with no power change at T (D1 outcome);
F = [sum_T (beta_T - mean)^2/4] / [variance of placebo betas] referred to F(4, 6) (party-year shock calibration). Party separated from
time: within-party Q (Democrats 2011, 2015, 2017; Republicans 2019, 2021) and a within-party weighted meta-regression slope on T.
Panel model (secondary): member FE, share ~ outPres + outOwnChamber + outOtherChamber + party-specific linear trends + outPres x Late +
outOwnChamber x Late (Late = year>=2017), member-clustered and wild bootstrap, leave-one-year-out of the Late interactions.
Decision: RECURSIVE if member-cluster Q p<.05 AND placebo-calibrated p<.05 AND at least one within-party Q p<.05 AND the within-party
slopes on T have the same sign in both parties. FIXED if Q p>.20 and I^2<25% and calibrated p>.20. Otherwise CANNOT DISTINGUISH.
