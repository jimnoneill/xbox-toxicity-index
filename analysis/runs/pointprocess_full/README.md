# Timing models for member event streams

Code behind Section 4.2 and Supplement S3 of the manuscript: comparison of timing models on each member's stream of events, the shared-activity term, the calibrated goodness-of-fit test, the diagnostic of leftover dependence, and the exploratory analysis of Congress as coupled on/off units.

The scripts read the scored corpus and the event tables of the main pipeline, which are not public. They are released as a record of the methods. The aggregate results they write are in the data archive (see RELEASES.md at the repository root).

| File | Purpose |
| --- | --- |
| `run_full.py` | Builds post-level member streams and fits the model set of `xbox/pointprocess_models.py` to every member with enough posts. |
| `run_events.py` | The same comparison on event streams (threads combined), with the validation on simulated streams and the parametric bootstrap of the time-rescaling test. |
| `shared_models.py`, `run_shared.py` | Hawkes model whose background rate also follows the number of events other members started in the same hour; fits, likelihood-ratio tests, refits by calendar year, bootstrap and the bias simulation. |
| `scale_misfit.py` | Observed and model-predicted shares of gaps between events, by length of gap. |
| `run_memory.py` | Detrended fluctuation analysis and autocorrelations of the normal scores of the rescaled intervals, with simulated streams as the reference. |
| `run_spin.py` | Members as on/off units by hour: schedule model, one collective term, party-split collective terms and member-by-member couplings (pseudo-likelihood, ridge), fitted on 2018 to 2021 and scored on 2022; simulation of the number of members on at once. |
| `summarize_full.py`, `summarize_shared.py` | Write the summary files read by the manuscript's number checks. |
| `figS1_v40.py` | Draws Figure S1. |
| `chain_events.sh`, `chain_shared.sh` | Run order. |
| `wolfram/*.wl` | Wolfram Language checks of three analytic quantities: the critical value of the Kolmogorov-Smirnov test with an estimated rate, the size of rate error that a given test distance corresponds to, and the stream length at which BIC changes its choice. |

Result files in the data archive: `summary_temporal_full.json` (posts), `summary_temporal_events.json` (events), `summary_shared.json` (shared-activity term, bootstrap, misfit and dependence), `spin_summary.json` (coupled units), `figS1_v40.json` (values drawn in Figure S1) and `wolfram/wolfram_results.json`. Member-level tables are not released.
