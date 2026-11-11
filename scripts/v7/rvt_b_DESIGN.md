# RVT-B pre-registered design (written before any estimation; never edited afterward)

Date written: 2026-10-13. Only data-construction checks (file schemas, date ranges, account types, package availability) were looked at before this file. No TI7 outcome estimate was inspected.

## Common construction (rvt_b_01_build.py)
- Posts: post_probs_full.parquet, post_type != 'retweet'. TI7 = mean of the 7 p_* columns. Map screen_name to bioguide via account_member_map.csv (unmatched dropped). Date = America/New_York calendar date.
- Member-day TI7 = mean TI7 of that member's posts that day (also keep post count and campaign-account post count). Join calendar_member_day.parquet on (date, bioguide); keep in_office==1 and party in {D,R}.
- Cached at output/v7/rvt_b/member_day.parquet and member_week.parquet (weeks start Monday, Eastern).
- Seed 20261013. Level 0.05, two-sided unless "rising" is stated.

## H4a Step vs ramp at power transitions (rvt_b_02_h4a.py)
Events (4): 2019-01-03 (House to D), 2020-11-03 (election), 2021-01-20 (inauguration; presidency and Senate to D), 2023-01-03 (House to R). Parties D, R, so 8 event-party series.
Predicted direction (opposition effect): the party that loses power goes UP, the party that gains goes DOWN. Loser: 2019 R, 2020-11 R, 2021 R, 2023 D. s = +1 for loser, -1 for gainer; signed step = s * step.
Primary series ("in-office"): per day, unweighted mean over in-office party members who posted that day of member-day TI7. Robustness series ("stayers"): same, restricted to members in office on every day of the window (removes composition jumps at 3 Jan swearing-ins). A verdict is robust only if both agree.
Window: 120 calendar days each side of the event, event day excluded; days with no posts dropped. (Windows of 2020-11-03 and 2021-01-20 overlap; stated, not corrected.)
Models (OLS on the daily series, day-of-week dummies in all): (i) trend; (ii) trend + step 1[t>=0]; (iii) trend + ramp t*1[t>=0]; (iv) trend + step + ramp. t in days from event. Compare AIC and BIC (Gaussian OLS likelihood). Step and ramp inference: Newey-West HAC, 7 lags.
Placebo: per party, 50 random dates drawn without replacement from dates with a full +-120 day window inside the data and at least 21 days from any of the 4 event dates (windows may overlap real events; conservative). Same four models; record step (model ii), ramp slope (model iii), dBIC = BIC(iii) - BIC(ii) (positive = step preferred). Real step placed in the placebo distribution by percentile of |step| and by empirical one-sided p for the signed step in the predicted direction = (1 + #placebo with s_event * step >= real signed value)/(51).
Decision rules:
- A series "changes" if min(BIC(ii),BIC(iii),BIC(iv)) < BIC(i) - 6 AND the real |step| (model ii) or |ramp slope x 120| (model iii) exceeds the 95th percentile of the matching placebo statistic.
- Among changing series: "step-dominant" if BIC(ii) < BIC(iii) - 2; "ramp-dominant" if BIC(iii) < BIC(ii) - 2; else ambiguous.
- Also compare the share of real series with step preferred (dBIC > 2) with the placebo share.
- Supports RVT if at least 3 series change and at least 2/3 of changing series are step-dominant and a majority of signed steps are in the predicted direction. Against if the majority of changing series are ramp-dominant, or fewer than 2 series change. Otherwise cannot distinguish. Primary and stayers must agree for supports/against; disagreement means cannot distinguish.

## H4b Critical slowing down (EXPLORATORY) (rvt_b_03_h4b.py)
Series per member: (a) weekly mean TI7 over non-retweet posts, requiring >= 5 posts in the week; (b) weekly log(1+post count) over in-office weeks. Eligible: members with a contiguous run (gaps <= 2 weeks linearly interpolated, longer gaps break the run) of >= 60 weeks.
Changepoints: own PELT implementation, L2 mean-shift cost, min segment 8 weeks, penalty = 4 * sigma^2 * ln(n), sigma from MAD of first differences / (sqrt2 * 0.6745). Keep a changepoint if |mean of next segment - mean of previous segment| > 1 within-member SD (SD of the run).
Exclusion: drop changepoints within 3 weeks of 2018-11-06, 2019-01-03, 2020-11-03, 2021-01-05, 2021-01-20, 2022-11-08, 2023-01-03, 2024-11-05, 2025-01-03, 2025-01-20. Qualifying also requires the 26 weeks before the changepoint to lie inside the run and contain no other changepoint.
Indicators: the 26 weeks c-26..c-1; remove a linear fit within the window; rolling window 13 weeks; rolling variance and lag-1 autocorrelation (14 positions); Kendall tau against time for each.
Null 1 (primary): 1000 AR(1) surrogates per event matched to the detrended window's phi and variance (re-detrended the same way); p = share of surrogate taus >= observed (one-sided, rising). Null 2: 1000 phase-randomized surrogates of the window, same procedure.
Matched non-events: up to 5 random control windows per event from the same member and series, where the 26-week window plus following 8 weeks contain no detected changepoint and no excluded date; same statistics.
Report per series type and indicator: n qualifying events; share with p<0.05 (rising) versus the 5% null rate (exact one-sided binomial) and versus the control windows' share; mean tau of events minus controls (bootstrap CI resampling members). Four primary tests (2 indicators x TI7/volume), Holm.
Decision: Supports CSD if, for the TI7 series, both indicators have significant share > 5% (Holm p<0.05) AND the mean-tau difference vs controls has CI above 0. Against if n >= 30 and neither indicator is elevated (Holm p>0.05 and tau-difference CIs include 0 or are negative). Fewer than 30 qualifying events, or mixed results: cannot distinguish (descriptive).

## H3 Configuration (rvt_b_04_h3.py)
Transitions: 2019-01-03, 2021-01-20, 2023-01-03. Unit: member x transition; member in office on day t-1 and day t+179, with >= 30 non-retweet posts in each window.
Outcome: dTI7 = mean post-level TI7 over [t, t+179] minus over [t-180, t-1].
Exposure (calendar flags): dpower = (aligned_with_president + in_chamber_majority) at t minus at t-1 (range -2..+2).
Features measured before t: seniority (years since first term start, terms_all.csv); margin (last general-election margin on or before the transition: own GENERAL % minus best other GENERAL % from FEC official results workbooks 2018/2020/2022, matched by FEC candidate id in elections.csv; uncontested = 1.0; members with no FEC result in those workbooks, e.g. senators not up and any race before 2018, get missing => train-fold median imputation + missing indicator; missingness reported); chamber (senate=1); majority_before (in_chamber_majority at t-1); extremity (|nom_dim1|, fallback |np_dim1|); campaign share (share of the member's pre-window non-retweet posts from account_type=='campaign'). Controls in every model: rep (party), pre-window TI7 level (regression to the mean), dpower, two transition dummies. Features standardized inside each training fold.
Models (sklearn pipelines): A additive ridge; B ridge on additive + all pairwise products of the continuous/binary substantive features (seniority, margin, chamber, majority_before, extremity, campaign share, rep, pre TI7, dpower; dummies and missing indicator stay additive); C GradientBoostingRegressor (depth 3, 300 trees, lr 0.03, subsample 0.7, min_samples_leaf 20, no tuning). Ridge alpha by inner 5-fold CV over logspace(-2,4,25).
CV: primary leave-one-transition-out (3 folds); secondary 10-fold GroupKFold by member, 5 repeats with shuffled groups. Skill = out-of-fold MSE and R2 against the training-fold mean predictor. Skill difference = MSE(A) - MSE(B or C), 2000 cluster-bootstrap resamples of members on the pooled out-of-fold predictions, 95% percentile CI.
Interaction terms: ridge-B standardized coefficients over 200 member-bootstrap refits: top 10 by |mean coef| with 95% CI and sign agreement; plus OLS (A + all products) with member-cluster robust SEs, Holm over the interaction family. "Large and stable" = bootstrap CI excludes 0, |coef| >= 0.1 SD of dTI7, and Holm p < 0.05.
Additive associations: OLS (member-cluster SE) of dTI7 on standardized features; focal pre-specified: among members with dpower < 0 (lost power) the coefficient of margin (do safer members respond more?).
Decision: Supports RVT if, in BOTH CV schemes, at least one of B, C has skill difference CI > 0 versus A. Against if A has OOS R2 > 0.005 and in neither scheme does B or C beat A with CI > 0. Cannot distinguish if no model has OOS R2 > 0.005, or the schemes disagree.

## Reporting
Deviations from this file are logged in the final report, never by editing this file.
