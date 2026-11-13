# RVT-E: precision of the nulls, specificity of the content result, dating of the change, and own history at short time scales

Date written: 2026-11-12, before any RVT-E estimate was computed. This file is not edited after estimation; deviations are logged in the report.

## Disclosure

Known when this was written: all results of RVT-A, RVT-B, RVT-C and RVT-D. For E1, approximate 95 percent intervals for the two C3 coefficients had already been worked out by hand from the rounded published values (TI7 0.003, SE 0.002; log event rate -0.006, SE 0.031), so the direction of the E1 verdicts for C3 was foreseeable when the margins below were fixed; the margins follow the rule RVT-A fixed on 2026-10-13 (30 percent of the established environmental effect). For E4, the H4a result for the 2020 election date was known (no step in the losing party's TI7 series, placebo p = 0.69), as were the C5 volume steps at the three seating dates for both parties. Looked at for construction only: file schemas, row counts, the values of the post-type column and the date range of the single-account archive. No split-half quantity, no placebo-outcome contrast, no election-date estimate for 2018 or 2022, no feedback or persistence estimate had been looked at.

The congressional corpus carries no engagement counts (post identifier, time, text and client only). A feedback test in the manner of the reinforcement literature therefore cannot be run on the corpus. It is run on the one account for which counts were collected (E5), and the corpus is used for a description of how long a member's own deviations last (E6).

## Common construction

Member-day table as in RVT-C: members of the two major parties in office, 2017-06-21 to 2023-07-11, with post count and TI7 sum per day (non-repost posts) and event count per day. State of a member over a set of days: TI7 = post-weighted mean (minimum number of posts stated per test); log event rate = ln((events + 0.5)/days). Standard errors clustered by member; two-sided 0.05; Holm within a test across the coordinates or features named. Seed 20261112.

## E1 Bounds on the tests of individual history (no new estimation)

Inputs are the stored estimates and standard errors: H1a pooled coefficient of earlier matched time out of power (M1), H1b pooled difference-in-differences, C3 coefficient of the previous margin for TI7 and for log event rate.

- For each: 90 and 95 percent intervals (as stored, or estimate +/- 1.645 and 1.96 SE), the largest absolute effect inside the 95 percent interval, and that bound as a share of the benchmark environmental effect in the same units.
- Equivalence margin: 30 percent of the benchmark. Panel tests: +/-0.03 share, as fixed in RVT-A. C3, TI7: 0.3 x the C4 contrast in TI7 between members who lost and gained power. C3, log event rate: 0.3 x the C1 difference in log event rate between members on and off the ballot.
- Verdict per test: "bounded" if the 90 percent interval lies inside the margin (two one-sided tests at 0.05), otherwise "not bounded at this margin". Minimum detectable effect reported as 2.8 SE.
- The correlations of successive election responses get Fisher-z 95 percent intervals from the stored r and number of pairs.

## E2 Reliability of a member's single response

The correlation of a member's responses to successive elections (C3: 0.07 for TI7, 0.08 for log event rate) can be low because responses are unstable or because one response is measured with much error. E2 measures the error.

- Sample and periods as in RVT-C (pre = 56 days before Labor Day; window = Labor Day to election day; coverage rules unchanged).
- Split: within each period, days are numbered in date order and divided into odd and even days. Half-state computed as the state, with at least 5 non-repost posts per half-period for TI7 (sensitivity: 10). Half-response R_h = half-state(window) - half-state(pre).
- Split-half correlation r_h of R_odd with R_even across member-cycles of members on the ballot, after subtracting cycle means. Reliability of the full response by Spearman-Brown: rel = 2 r_h / (1 + r_h).
- Cross-cycle correlation r_x recomputed as in C3 (raw) and after subtracting cycle means.
- Reported: rel; r_x; r_x / rel (correlation of true responses under classical error, with rel the largest r_x that perfectly stable responses could show). The same three quantities for the level of the state in the window, as a reference for what a stable trait looks like in these data.
- Uncertainty: 2,000 bootstrap resamples of members, percentile 95 percent intervals.
- Reading fixed in advance, per coordinate: if the upper bound of rel is below 0.2, a single response is too noisy for r_x to say anything about stability. If the lower bound of rel is at least 0.5 and the upper bound of r_x / rel is below 0.5, responses are measured well and do not persist. Anything else is reported as intermediate, with the numbers.
- Secondary: the same split-half reliability for the displacement at changes in party control (C4: 180 days after minus 180 days before), which bears on every member-level analysis of that displacement.

## E3 Placebo outcomes at changes in party control

C4 found that TI7 rose for members who lost power relative to members who gained. E3 asks whether an arbitrary property of the text shows a contrast of similar size in the same model.

- Post-level features from the text of non-repost posts, after lowercasing and removing links, account mentions and HTML entities; posts with at least 20 letters a to z. Four arbitrary letter-class shares (share of letters): P1 x, y or z; P2 the letters n to z; P3 vowels a, e, i, o, u; P4 letters at odd positions of the alphabet (a, c, e, ...).
- Scored outcomes run through the same model for comparison: TI7, each of its seven components, and the scores outside the index (advocacy, genuine anger, emotion anger, hate, epistemic, engagement bait, agency). One style measure, log number of letters, is reported and not treated as a placebo.
- Member-period value: post-weighted mean over the period, at least 10 posts with the feature defined. Displacement, groups, change indicators and clustering exactly as C4, at the real dates and one year before.
- Each contrast is reported in raw units and standardized by the standard deviation of that outcome's displacement across members (real dates).
- Check: the TI7 contrast must reproduce the stored C4 value.
- Reading fixed in advance: the content result is specific if no arbitrary feature has Holm p < 0.05 (four features) at the real dates and the absolute standardized TI7 contrast exceeds the largest absolute standardized contrast among the four. If an arbitrary feature does show Holm p < 0.05, the TI7 contrast is reported as a multiple of the largest arbitrary contrast and the paper's wording is limited accordingly.

## E4 Is the change in place at the election or at the seating?

The outcome of an election is known about eight weeks before control changes hands (eleven weeks in 2020 to 2021). A response to the conditions of office predicts a shift at the seating. A response to known future conditions predicts a shift at the election.

- Pairs (election, seating): (2018-11-06, 2019-01-03), (2020-11-03, 2021-01-20), (2022-11-08, 2023-01-03).
- Groups by member, from the calendar table at the seating date as in C4: lost power, gained power. Members in office on every day from 120 days before the election to 120 days after the seating (stayers), so that the series contain the same members throughout.
- Daily series per group: TI7 = mean over members posting that day of the member-day TI7; volume = mean over all stayers of ln(1 + events). Primary series: lost minus gained, which removes what the calendar does to both groups (campaign, holidays, opening of a Congress). The two group series are reported as secondary.
- Model on days from election - 120 to seating + 120, the two event days excluded: constant, linear trend, day-of-week indicators, step at the election, step at the seating. Newey-West standard errors with 7 lags. Also reported: BIC of the four models (neither step, election only, seating only, both) and the three period means.
- Placebo: for each pair, 50 anchor dates drawn at random such that the whole structure (same spacing, same window) lies inside the data and both placebo event dates are at least 21 days from all six real dates; same members, stayers recomputed for the placebo window. Two-sided empirical p for each step = (1 + number of placebo |step| at least as large)/(number of placebos + 1).
- Pooled over the three pairs: inverse-variance mean of each step, and Fisher's combination of the three placebo p values for each step.
- Reading fixed in advance, per coordinate: "at the seating" if the seating step has pooled Fisher p < 0.05 and the election step does not; "at the election" for the reverse; "both" or "neither" otherwise. Caveats fixed in advance: in 2020 the presidency was called four days after election day and control of the Senate was not settled until January 2021, and in 2022 control of the House was called about eight days after election day, so the election step is dated early.

## E5 Feedback and the next day's posting, single account (exploratory)

The reinforcement literature finds that feedback on a user's posts predicts the user's later posting. E5 asks this of the one account with engagement counts, the personal account of the case-study senator (posts scored as in the paper; archive from July 2022).

- Posts: non-repost posts with scores. Feedback of a post f = ln(1 + likes + reposts), counts as collected. Residual feedback = f minus the mean f of posts of the same type (original, reply, quote) in the same calendar month, so that neither the growth of the account nor its mix of post types enters.
- Day = America/New_York date. Per day: n posts, TI7 = mean, mean residual feedback F. Premium of provocation P = mean residual feedback of that day's posts above the account's median TI7 over the previous 28 days minus that of the posts at or below it, on days with at least 2 posts on each side.
- Two coefficients, fixed in advance: (i) content: TI7 on day t+1 on P of day t; (ii) volume: ln(1 + posts) on day t+1 on the feedback surprise of day t, S = F of day t minus the mean of F over days t-7 to t-1. Controls in both: the outcome on day t and its mean over days t-7 to t-1, ln(1 + posts) on day t, day-of-week indicators, calendar-month indicators. Days t and t+1 must both have at least 2 posts for (i). Newey-West, 7 lags. Holm across the two.
- Lead placebo for each: the same model with the predictor taken from day t+2, after the outcome. Learning predicts the lag coefficient and not the lead.
- Reported per SD of the predictor and as a share of the SD of the outcome. One account: whatever is found describes this account and no other.

## E6 How long does a member's own deviation last? (descriptive)

- Member-week state from the corpus: TI7 (at least 5 non-repost posts in the week) and ln(1 + events) for weeks fully in office. From each, subtract the mean of the member's party in that week (the shared environment, including party control) and then the member's own mean over all weeks.
- Autocorrelation of the residual at lags of 1, 2, 4, 8, 13, 26, 52 and 104 weeks, pooled over members, with 95 percent intervals from 500 bootstrap resamples of members.
- Measurement error: split-half reliability of the weekly TI7 residual from odd and even days of the week (Spearman-Brown); autocorrelations are shown raw and divided by that reliability.
- No hypothesis is tested. The lag at which the interval first includes zero is reported as the length of own-state memory. Persistence of a member's own deviations bounds from above how long any dependence of the state on its own past can last; it also contains slow circumstances particular to a member, and so cannot by itself show that experience changed anything.
