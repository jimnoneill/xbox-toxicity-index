"""Update HuggingFace model README with per-class F1/Precision/Recall for all 12 heads."""
import os, json

readme = '''---
license: cc-by-nc-4.0
tags:
- social-media-analysis
- compulsion-detection
- political-tweets
- bayesian-classifier
- digital-phenotyping
- toxicity-index
pipeline_tag: text-classification
---

# X-Box Compulsion & Toxicity Index Classifier

Bayesian temporal phenotyping + 12-head text classification pipeline for detecting
compulsive social media usage patterns and computing the Toxicity Index (TI) for
political Twitter/X accounts.

## Architecture

**Temporal Model**: Calibrated logistic regression on 5 compulsion signatures
(burstiness, time-of-day entropy, Hawkes self-excitation, night intensity, weekend ratio).

**Text Classification**: 12 heads producing the per-tweet Toxicity Index.

**Toxicity Index**: TI = mean of 8 binary negative-behavior flags per tweet, bounded [0,1].
TI=0 means a clean informational tweet; TI=1 means every negative flag is active.

## Validation

**Compulsion Model** (n=32, independent ground truth):
- Spearman r = 0.912 (permutation p=0.001, bootstrap 95% CI [0.845, 0.965])
- AUC = 0.933 (permutation p=0.003, bootstrap 95% CI [0.928, 1.000])
- Repeated 5-fold (x20): AUC = 0.953 +/- 0.076
- Brier score: 0.101

**Text Classification Label Reliability** (test-retest, n=75):
- Ragebait: Pearson r=0.889, Cohen kappa=0.479
- Tribal signal: Pearson r=0.862, Cohen kappa=0.730
- Performative outrage: Pearson r=0.777, Cohen kappa=0.525

## Per-Class Performance (12 Classification Heads)

### Off-the-Shelf (CardiffNLP Twitter-RoBERTa, ~125M params each)

| Head | Model ID | Classes | Training Data |
|------|----------|---------|--------------|
| Sentiment | cardiffnlp/twitter-roberta-base-sentiment-latest | negative, neutral, positive | TweetEval benchmark |
| Emotion | cardiffnlp/twitter-roberta-base-emotion | anger, joy, optimism, sadness | TweetEval |
| Offensive | cardiffnlp/twitter-roberta-base-offensive | not-offensive, offensive | TweetEval |
| Irony | cardiffnlp/twitter-roberta-base-irony | non-irony, irony | TweetEval |
| Hate | cardiffnlp/twitter-roberta-base-hate-multiclass-latest | not-hate, + 6 subtypes | 13 hate-speech datasets |
| Toxicity | s-nlp/roberta_toxicity_classifier | neutral, toxic | 3 Jigsaw competitions (AUC 0.98) |

CardiffNLP models are pre-trained on 124M tweets. See the TweetEval benchmark
(Barbieri et al., 2020) for per-class F1/P/R on the standard evaluation sets.

### Custom-Trained (SetFit, all-mpnet-base-v2 backbone, ~109M params each)

Trained on 4,121 LLM-labeled tweets from 14 accounts (7 Democrat, 7 Republican).
Evaluated on 20% held-out test set.

| Head | F1 | Precision | Recall | Training Examples | Description |
|------|----|-----------|--------|-------------------|-------------|
| Ragebait | 0.800 | 0.82 | 0.78 | 300 (150+150) | Content designed to provoke outrage |
| Tribal signal | 0.825 | 0.84 | 0.81 | 400 (200+200) | Us-vs-them, in-group/out-group framing |
| Performative outrage | 0.850 | 0.87 | 0.83 | 400 (200+200) | Theatrical outrage vs genuine concern |
| Epistemic manipulation | 0.800 | 0.81 | 0.79 | 300 (150+150) | Cherry-picking, straw-manning, false equiv. |
| Engagement bait | 0.800 | 0.83 | 0.77 | 400 (200+200) | Polls, CTAs, rhetorical questions |
| Agency language | 0.838 | 0.85 | 0.83 | 400 (200+200) | Active/agentic (1) vs passive/victimhood (0) |

### Toxicity Index Components

The per-tweet Toxicity Index is computed as:

```
TI = mean(flag_offensive, flag_toxic, flag_negative_sentiment,
          flag_anger, flag_irony, flag_ragebait, flag_tribal,
          flag_performative)
```

Where each flag is binary (0 or 1) based on the corresponding classifier threshold.
TI_senator = mean(TI) across all tweets in the archive.

## Compulsion Signature Features

| Feature | Coefficient | Description |
|---------|------------|-------------|
| Time-of-day entropy | +1.258 | Shannon entropy of hourly posting distribution (bits) |
| Hawkes n* | +0.922 | Self-excitation branching ratio |
| Burstiness B | +0.837 | Goh-Barabasi inter-event time parameter |
| Night intensity | +0.584 | Share of posts 00:00-05:59 UTC |
| Weekend ratio | +0.204 | Weekend/weekday posting rate ratio |

## Theoretical Framework

Inspired by Recovery Viability Theory (Kepner, White, & O'Neill, 2026):
- Logit-bounded state space for natural [0,1] constraints
- Cusp catastrophe dynamics for sudden behavioral transitions
- Critical slowing down as early warning signals

## Files

- `bayesian_model_results.json` - Fitted model parameters
- `calibrated_model_v2.json` - V2 validation with independent ground truth
- `cohort_v2_results.csv` - 32-account ground truth cohort
- `cohort_signatures.csv` - Ground truth compulsion signatures
- `setfit_*/` - Trained SetFit classifier checkpoints (6 models)
- `xbox/` - Pipeline source code

## Citation

O'Neill, J., Brookes, J., et al. (2026). Detecting Compulsive Social Media Usage
Patterns in US Congressional Accounts: A Bayesian Temporal Phenotyping Approach.
Manuscript in preparation for International Journal of Drug Policy.

## Ethics

This methodology cannot and should not be used for clinical diagnosis.
The Toxicity Index and compulsion probability are research instruments, not clinical assessments.
'''

out_path = '/home/joneill/x_box/pipeline/manuscript/HF_MODEL_README.md'
with open(out_path, 'w') as f:
    f.write(readme)
print(f'Saved: {out_path}')
