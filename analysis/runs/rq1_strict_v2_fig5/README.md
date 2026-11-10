# RQ1 Figure 5, strict-v2 (corrected design)

Isolated directory. Everything outside it was read only. Code: `fig5_v2.py` (imports `../rq1_gbt_v2/run_gbt.py` and the P1 recipe in `scripts/v7/p1_models.py` without changing them).

## Commands

```bash
cd ~/x_box/pipeline/manuscript/epjds_submission_work/rq1_strict_v2_fig5
./run_all.sh            # monthly, final, lee --mode archived, lee --mode strict, lmm (logs: log_*.log)
PYTHONPATH=$HOME/x_box/pyextra ~/x_box/venv/bin/python -u fig5_v2.py --stage report   # results.json
PYTHONPATH=$HOME/x_box/pyextra ~/x_box/venv/bin/python -u fig5_v2.py --stage figure   # fig5_strict_v2.png/.pdf
PYTHONPATH=$HOME/x_box/pyextra ~/x_box/venv/bin/python figS2_timing_sensitivity.py
cd fig8 && PYTHONPATH=$HOME/x_box/pyextra ~/x_box/venv/bin/python render_fig8.py
```

Wall time of `run_all.sh`: 1,571 s (monthly 894, final 138, Lee 19 + 55, mixed models 452).

## Design

- Features and cohort: `run_gbt.aligned_repaired_frames` and `select_aligned_origin` (strict-v2 lags with 7,200 s closure, prior-date vote clock, Jan 3 noon seat change, Congress lags without hindsight imputation, at least 10 closed prior events, training labels closed 7,200 s before origin). The previous-event multi-post flag uses `strict_prev_multi` (closure-aware). Previous reply flag, reply share of last 10 and own first-post reply flag are start-observable and identical in both feature files (checked, 0 differences). Own timing, scheduler share and regime flags are the P1 columns, as in the aligned design.
- Learner: P1 LightGBM, frozen `hp.json`, seed 20261010, 600,000-row cap. Single-block ablations use the `reg_R` hyperparameters, as archived `p1_final.py` did. No tuning.
- Origins: 13 monthly origins July 2022 to July 2023, 422,529 test events, 923 accounts.
- Gate: P and R refits equal `../rq1_gbt_v2/predictions_aligned` exactly (max abs difference 0 in every month, `monthly_checks.json`), and the account bootstrap returns +1.471370% (1.274712 to 1.708672), identical to `aligned_evaluation.json`.
- CIs: account-blocked bootstrap, 2,000 draws, seed 20261010, all months of an account resampled together (aligned_summary scheme). Paired differences use the same draws.

## Files and what the numbers mean (results.json)

- `panel_a`: `m_R_vs_m_P` rhythm over content history; `m_RT_vs_m_PT` both models also get previous-event type controls (reply status, multi-post status, reply share of last 10); `m_RTO_vs_m_PTO` both also get the target's own first-post reply flag; `lmm_R_vs_lmm_P` mixed model; `rhythm_gain_change_with_prev_type_controls_pp` paired change in rhythm gain; `m_P_vs_roll50` content reference over the corrected 50-event rolling mean; `orig_r_vs_orig_p` archived predictions on the same rows.
- `panel_b`: rhythm over content by party, chamber, account type, scheduled-looking (`flag_strong`).
- `panel_c_monthly`: persistence plus one block versus persistence, 13 origins (plotted). `panel_c_single_final_origin`: same models fitted once at 2022-07-01, the archived procedure (412,347 test events, account filter at that origin).
- `risk_curves`: partial dependence of P(TI7 above training-window account 80th percentile) from the corrected final-origin rhythm classifier (`pdp_strict_v2.csv`, archived grid and bootstrap), plus empirical bins (`empirical_bins_strict_v2.csv`); boolean checks of the four text statements for corrected and archived curves.
- `lee_archived_inputs` / `lee_strict_inputs` (also `lee_archived.json`, `lee_strict.json`): Lee transfer with models trained on all congressional rows before 2023-07-12, no Congress features, calendar-week block bootstrap B=1000 (archived procedure). The archived mode reproduces `output/v7/p1/lee.json` exactly (4.651395% personal, CI 3.884735 to 5.401864). Strict mode rebuilds Lee inputs: 7,200 s closure lags and multi-post flag, prior-date vote clock (Utah zone, VoteView through 2026-09-30), account encoding from closed events only, eligibility at least 10 closed prior events; training uses the strict cohort.
- `fig5_strict_v2.png/.pdf`: new Figure 5 from `panel_a`, `panel_b`, `panel_c_monthly`. Own post type is drawn open and labelled diagnostic.
- `figS2_timing_sensitivity.png/.pdf`: relabelled supplementary S2 (values from `../rq1_gbt_v2/aligned_evaluation.json`).
- `fig8_lee.png/.pdf`: Figure 8 with "non-retweet" changed to "non-repost"; `fig8/figs_04_lee_temporal_nonrepost.py` differs from the original by that one line.

Limits: same retrospective event universe, outage mask and thread reconstruction as the aligned run; associations, not causal effects.
