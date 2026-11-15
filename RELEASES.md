# Versioned research archives

The submission analysis extensions are archived with the original analysis source. The software is MIT licensed, including the added analysis extensions, as authorized by the copyright owner.

Software archive: https://doi.org/10.5281/zenodo.23968306
Derived results and annotation metadata: https://doi.org/10.5281/zenodo.23968308

The data archive contains selected aggregate numerical results and numerical training/annotation labels, not the original post text or a complete retrainable corpus. The private source dataset remains private. Both archive READMEs and manifests state exclusions, provenance and reproducibility boundaries. Existing numerical analysis results were not recomputed for this release.

## Version 2026.11.09

Software archive: https://doi.org/10.5281/zenodo.24095873 (previous version 10.5281/zenodo.23968306)
Derived results and annotation metadata: https://doi.org/10.5281/zenodo.24095876 (previous version 10.5281/zenodo.23968308)

Added code: the Figure 5 refit on the corrected event-timing design with Figure S2 and Figure 8 rendering (analysis/runs/rq1_strict_v2_fig5), scripts that redraw Figures 5 and 7 from their stored results with the manuscript's terms (analysis/runs/fig_relabel_v30), and the builder of the thread-joining audit sheet (analysis/annotation/thread_audit).

Added data (Zenodo only): the 200-pair thread-joining audit with an author's review, a summary of training-post sources and date ranges, the 100 curated political events with public sources, aggregate volume and burst forecast scores, Figure 5 results and risk curves, case-study descriptors, and the Figure 7 inputs. Files of version 2026.11.04 are unchanged in both archives.

## Version 2026.11.10

Software archive: https://doi.org/10.5281/zenodo.24128150 (previous version 10.5281/zenodo.24095873)
Derived results and annotation metadata: https://doi.org/10.5281/zenodo.24128151 (previous version 10.5281/zenodo.24095876)

Added code: timing models for member post and event streams, with the shared-activity term, the calibrated goodness-of-fit test, the diagnostic of leftover dependence, the coupled-units analysis and Figure S1 (analysis/runs/pointprocess_full, see its README), and scripts that redraw Figures 2, 4, 6 and 7 and the external-panel figure from their stored results (analysis/runs/fig_relabel_v36, fig_relabel_v37, fig_relabel_v42).

Added data (Zenodo only): aggregate results of the timing models, the shared-activity term, the leftover-dependence diagnostic and the coupled-units analysis, and the values plotted in Figure 2, Figure 4, Figure S1 and the external-panel figure. Member-level tables are not released. Files of version 2026.11.09 are unchanged in both archives.

## Version 2026.11.11

Software archive: https://doi.org/10.5281/zenodo.24144555 (previous version 10.5281/zenodo.24128150)
Derived results and annotation metadata: https://doi.org/10.5281/zenodo.24144556 (previous version 10.5281/zenodo.24128151)

Added code and designs: the written designs of the tests of the viability framework (scripts/v7/rvt_a_DESIGN.md, rvt_b_DESIGN.md, rvt_c_DESIGN.md, rvt_d_DESIGN.md), the code of the second set of tests and of the release and width tests (scripts/v7/rvt_c_01_state.py, rvt_c_02_c5_deviation.py, rvt_d_01_release_width.py), the comparison of one timing rule with yearly rules (analysis/runs/pointprocess_full/run_drift.py), and the re-analysis of the stored test estimates (analysis/runs/pointprocess_full/wolfram/q4_rvt_scorecard.wl).

Added data (Zenodo only): aggregate results of the second set of tests and of the release and width tests, the timing-rule comparison, and an updated file of analytic check values. Member-level tables are not released. Other files of version 2026.11.10 are unchanged in both archives.

## Correction of 15 November 2026: class order of the general-purpose emotion model

What was wrong. The outputs of cardiffnlp/twitter-roberta-base-emotion are in the order of the TweetEval mapping file: anger, joy, optimism, sadness. The id2label table in that model's config.json lists joy, optimism, anger, sadness. The code looked the anger class up by name in the config, so the value it stored and analyzed as the emotion model's anger score was the optimism probability.

What is not affected. The Toxicity Index (TI7) and its seven components, which do not use the emotion model; the purpose-trained classifiers and their training labels; the timing analyses; the tests of the viability framework. The output positions of the other general-purpose models (sentiment, offensive language, irony, hate, toxicity) were checked and are correct.

What changed in this repository.

- xbox/config.py (CLASS_ORDER_OVERRIDES) and xbox/classifiers.py fix the class order in one place; scripts/v7/full_corpus_score.py, ti_classify_probs.py and t6_score_official.py take the anger class by position; scripts/classify_all_congress.py names the classes in the correct order. xbox/classifiers.py also accepts either spelling of the no-hate class.
- tests/test_emotion_class_order.py checks the order, including on sentences of known emotion when the model is available.
- scripts/v7/emotion4_score.py scores the emotion model again with all four outputs, emotion4_apply.py writes them into the stored score tables, and emotion_anger_stats.py analyzes the two anger scores (the emotion model's anger class and the purpose-trained genuine-anger classifier).
- scripts/v7/stats_anger.py is marked superseded. It examined an early draft claim about the class it took for anger.
- The fifth round of tests of the viability framework is added: scripts/v7/rvt_e_DESIGN.md (frozen before estimation), rvt_e_01 to rvt_e_07 and rvt_e_run.sh. Scripts rvt_e_08 and rvt_e_09 are the diagnostics that found the error.

What changed on Hugging Face. The model card no longer says that the emotion model's anger score ranks human-coded anger inversely or labels mostly advocacy; those statements described its optimism class. The xbox package copy is updated. In the dataset, the class column of labels_advocacy_anger_1k.parquet is renamed to the correct classes, and the columns of the earlier index tables that held the optimism flag are named accordingly.

Archived versions. The Zenodo versions up to 2026.11.11 hold the earlier code. Two files in the data archive carry the earlier naming: in annotation/classifier_vs_human.csv the measures "offshelf P(anger)" and "offshelf anger argmax" are the optimism probability and class, and "offshelf joy" is the anger probability; in output/v7/stats_full/numbers_stats_full.json the keys that begin anger_ots and the key p_emotion_anger describe the optimism class. A corrected archive version will follow this entry.
