"""
Bayesian compulsion classifier — calibrated posterior P(compulsive | signatures).

Fits class-conditional distributions from ground truth cohort, then computes
posterior probabilities with proper uncertainty quantification.

Uses the Kepner framework:
- Logit-bounded state space for natural [0,1] constraints
- Prior distributions fitted from known compulsive vs strategic accounts
- Posterior updates as new evidence (temporal features) is observed
- Viability boundary maps to the posterior decision threshold
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import logit, expit
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import LeaveOneOut
from dataclasses import dataclass, field

sys.path.insert(0, '/home/joneill/x_box/pipeline')


@dataclass
class BayesianCompulsionResult:
    """Result of Bayesian compulsion classification for one account."""
    account: str
    posterior_compulsive: float  # P(compulsive | features)
    posterior_strategic: float   # P(strategic | features)
    log_likelihood_ratio: float  # log P(features | compulsive) / P(features | strategic)
    classification: str          # "compulsive", "strategic", "uncertain"
    credible_interval: tuple = (0.0, 1.0)  # 95% credible interval
    feature_contributions: dict = field(default_factory=dict)  # per-feature log-likelihood ratios
    z_state: float = 0.0  # logit(posterior) — Kepner framework


class BayesianCompulsionClassifier:
    """
    Naive Bayes classifier with Gaussian class-conditional distributions,
    fitted on ground truth cohort.

    P(compulsive | x) = P(x | compulsive) * P(compulsive) / P(x)

    Where P(x | class) = product of P(x_i | class) for each feature
    (independence assumption — validated by AUC analysis).
    """

    FEATURES = [
        'burstiness_B', 'time_entropy', 'night_intensity',
        'hawkes_n', 'weekend_ratio',
    ]

    # Feature weights based on AUC from ground truth analysis
    FEATURE_WEIGHTS = {
        'burstiness_B': 1.0,
        'time_entropy': 1.2,     # AUC=1.000, strongest
        'night_intensity': 1.2,  # AUC=1.000
        'hawkes_n': 1.0,         # AUC=0.984
        'weekend_ratio': 0.3,    # AUC=0.625, weak — downweight
    }

    def __init__(self):
        self.class_params = {}  # {class: {feature: {mean, std}}}
        self.prior = {}         # {class: prior_probability}
        self.fitted = False

    def fit(self, df: pd.DataFrame, label_col: str = 'category'):
        """
        Fit class-conditional Gaussian distributions from ground truth data.

        Args:
            df: DataFrame with columns for each feature + label_col
            label_col: Column containing 'compulsive' or 'strategic'
        """
        classes = df[label_col].unique()

        for cls in classes:
            subset = df[df[label_col] == cls]
            self.prior[cls] = len(subset) / len(df)
            self.class_params[cls] = {}

            for feat in self.FEATURES:
                if feat in subset.columns:
                    values = subset[feat].dropna()
                    self.class_params[cls][feat] = {
                        'mean': float(values.mean()),
                        'std': max(float(values.std()), 0.01),  # floor to avoid division by zero
                        'n': len(values),
                    }

        self.fitted = True
        return self

    def predict(self, features: dict) -> BayesianCompulsionResult:
        """
        Compute posterior P(compulsive | features) for a single account.
        """
        if not self.fitted:
            raise RuntimeError("Must call fit() first")

        log_posteriors = {}
        feature_contributions = {}

        for cls in self.class_params:
            log_posterior = np.log(self.prior[cls])

            for feat in self.FEATURES:
                if feat in features and feat in self.class_params[cls]:
                    x = features[feat]
                    mu = self.class_params[cls][feat]['mean']
                    sigma = self.class_params[cls][feat]['std']
                    weight = self.FEATURE_WEIGHTS.get(feat, 1.0)

                    # Gaussian log-likelihood (weighted)
                    ll = stats.norm.logpdf(x, loc=mu, scale=sigma) * weight
                    log_posterior += ll

                    # Track per-feature contribution
                    if feat not in feature_contributions:
                        feature_contributions[feat] = {}
                    feature_contributions[feat][cls] = float(ll)

            log_posteriors[cls] = log_posterior

        # Normalize via log-sum-exp
        max_log = max(log_posteriors.values())
        log_evidence = max_log + np.log(
            sum(np.exp(lp - max_log) for lp in log_posteriors.values())
        )
        posteriors = {cls: np.exp(lp - log_evidence) for cls, lp in log_posteriors.items()}

        p_comp = posteriors.get('compulsive', 0)
        p_strat = posteriors.get('strategic', 0)

        # Log-likelihood ratio
        llr = log_posteriors.get('compulsive', -np.inf) - log_posteriors.get('strategic', -np.inf)

        # Per-feature LLR
        feat_llr = {}
        for feat in feature_contributions:
            comp_ll = feature_contributions[feat].get('compulsive', 0)
            strat_ll = feature_contributions[feat].get('strategic', 0)
            feat_llr[feat] = round(comp_ll - strat_ll, 3)

        # Classification with uncertainty zone
        if p_comp > 0.8:
            classification = 'compulsive'
        elif p_comp < 0.2:
            classification = 'strategic'
        else:
            classification = 'uncertain'

        # Approximate credible interval via Beta distribution
        # Use effective sample size from features
        n_eff = sum(self.class_params['compulsive'][f]['n']
                     for f in self.FEATURES if f in self.class_params['compulsive'])
        alpha = p_comp * n_eff + 1
        beta_param = (1 - p_comp) * n_eff + 1
        ci_low = stats.beta.ppf(0.025, alpha, beta_param)
        ci_high = stats.beta.ppf(0.975, alpha, beta_param)

        # Kepner z-state
        z = float(logit(np.clip(p_comp, 0.001, 0.999)))

        return BayesianCompulsionResult(
            account=features.get('account', 'unknown'),
            posterior_compulsive=round(p_comp, 4),
            posterior_strategic=round(p_strat, 4),
            log_likelihood_ratio=round(float(llr), 3),
            classification=classification,
            credible_interval=(round(ci_low, 4), round(ci_high, 4)),
            feature_contributions=feat_llr,
            z_state=round(z, 3),
        )

    def predict_batch(self, df: pd.DataFrame) -> list:
        """Predict for multiple accounts."""
        results = []
        for _, row in df.iterrows():
            features = {f: row[f] for f in self.FEATURES if f in row.index}
            features['account'] = row.get('account', row.get('username', 'unknown'))
            results.append(self.predict(features))
        return results

    def leave_one_out_cv(self, df: pd.DataFrame, label_col: str = 'category'):
        """
        Leave-one-out cross-validation — the gold standard for small cohorts.
        Refit the model on N-1 samples, predict the held-out sample, repeat.
        """
        loo = LeaveOneOut()
        y_true = []
        y_pred = []
        y_prob = []
        details = []

        for train_idx, test_idx in loo.split(df):
            train = df.iloc[train_idx]
            test = df.iloc[test_idx]

            # Refit on training fold
            model = BayesianCompulsionClassifier()
            model.fit(train, label_col=label_col)

            # Predict held-out
            row = test.iloc[0]
            features = {f: row[f] for f in self.FEATURES if f in row.index}
            features['account'] = row.get('account', 'unknown')
            result = model.predict(features)

            true_label = 1 if row[label_col] == 'compulsive' else 0
            pred_label = 1 if result.classification == 'compulsive' else 0

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_prob.append(result.posterior_compulsive)
            details.append({
                'account': features['account'],
                'true': row[label_col],
                'pred': result.classification,
                'posterior': result.posterior_compulsive,
                'correct': true_label == pred_label,
            })

        # Metrics
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0

        return {
            'accuracy': round(accuracy, 4),
            'f1': round(f1, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'auc': round(auc, 4),
            'n_samples': len(y_true),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'details': details,
        }

    def get_params(self) -> dict:
        """Return fitted parameters for inspection/serialization."""
        return {
            'prior': self.prior,
            'class_params': self.class_params,
            'features': self.FEATURES,
            'feature_weights': self.FEATURE_WEIGHTS,
        }


# ── Main: fit, validate, and apply ──────────────────────────────

if __name__ == '__main__':
    # Load ground truth
    cohort = pd.read_csv('/home/joneill/x_box/pipeline/data/ground_truth/cohort_signatures.csv')
    print(f'Ground truth cohort: {len(cohort)} accounts')
    print(f'  Compulsive: {(cohort.category == "compulsive").sum()}')
    print(f'  Strategic:  {(cohort.category == "strategic").sum()}')

    # ── Fit the model ─────────────────────────────────
    print('\n=== FITTING BAYESIAN MODEL ===')
    model = BayesianCompulsionClassifier()
    model.fit(cohort, label_col='category')

    params = model.get_params()
    print('\nClass-conditional distributions:')
    for cls in ['compulsive', 'strategic']:
        print(f'\n  P({cls}) = {params["prior"][cls]:.3f}')
        for feat in model.FEATURES:
            p = params['class_params'][cls][feat]
            print(f'    {feat:20s}: mu={p["mean"]:+8.3f}  sigma={p["std"]:6.3f}')

    # ── Leave-one-out cross-validation ────────────────
    print('\n=== LEAVE-ONE-OUT CROSS-VALIDATION ===')
    cv = model.leave_one_out_cv(cohort, label_col='category')

    print(f'\n  Accuracy:  {cv["accuracy"]:.1%}')
    print(f'  F1 Score:  {cv["f1"]:.3f}')
    print(f'  Precision: {cv["precision"]:.3f}')
    print(f'  Recall:    {cv["recall"]:.3f}')
    print(f'  AUC:       {cv["auc"]:.3f}')
    print(f'  Confusion: {cv["confusion_matrix"]}')

    print('\n  Per-account LOO results:')
    for d in cv['details']:
        correct = 'OK' if d['correct'] else 'MISS'
        print(f'    {d["account"]:22s}  true={d["true"]:10s}  pred={d["pred"]:12s}  P={d["posterior"]:.3f}  [{correct}]')

    # ── Score full cohort with fitted model ───────────
    print('\n=== FULL COHORT POSTERIOR SCORES ===')
    results = model.predict_batch(cohort)

    print(f'\n{"Account":22s} {"P(comp)":>8s} {"95% CI":>16s} {"LLR":>8s} {"z-state":>8s} {"Class":>12s}')
    print('-' * 80)
    for r in sorted(results, key=lambda x: x.posterior_compulsive, reverse=True):
        ci = f'[{r.credible_interval[0]:.3f}, {r.credible_interval[1]:.3f}]'
        print(f'{r.account:22s} {r.posterior_compulsive:8.4f} {ci:>16s} {r.log_likelihood_ratio:+8.1f} {r.z_state:+8.3f} {r.classification:>12s}')

    # ── Feature importance (mean |LLR| per feature) ──
    print('\n=== FEATURE IMPORTANCE (mean |LLR| contribution) ===')
    feat_importance = {}
    for r in results:
        for feat, llr in r.feature_contributions.items():
            feat_importance.setdefault(feat, []).append(abs(llr))
    for feat, vals in sorted(feat_importance.items(), key=lambda x: -np.mean(x[1])):
        print(f'  {feat:20s}  mean_|LLR|={np.mean(vals):.3f}')

    # ── Apply to all 98 senators from HF data ────────
    print('\n=== SCORING ALL 98 SENATORS ===')
    from xbox.compulsion_signatures import CompulsionSignatureAnalyzer

    hf = pd.read_parquet('/home/joneill/x_box/pipeline/data/senator_tweets_hf.parquet')
    hf['created_at'] = pd.to_datetime(hf['date'], errors='coerce', utc=True)

    sig_analyzer = CompulsionSignatureAnalyzer()
    senator_results = []

    for username in hf['username'].unique():
        sub = hf[hf['username'] == username]
        if len(sub) < 100:
            continue
        sig = sig_analyzer.analyze(sub, senator_name=username)
        features = {
            'account': username,
            'burstiness_B': sig.burstiness_B,
            'time_entropy': sig.time_entropy,
            'night_intensity': sig.night_intensity,
            'hawkes_n': sig.self_excitation_ratio,
            'weekend_ratio': sig.weekend_ratio,
        }
        result = model.predict(features)
        senator_results.append({
            'account': username,
            'posterior_compulsive': result.posterior_compulsive,
            'classification': result.classification,
            'ci_low': result.credible_interval[0],
            'ci_high': result.credible_interval[1],
            'llr': result.log_likelihood_ratio,
            'z_state': result.z_state,
        })

    senator_df = pd.DataFrame(senator_results).sort_values('posterior_compulsive', ascending=False)
    senator_df.to_csv('/home/joneill/x_box/pipeline/output/senator_bayesian_scores.csv', index=False)

    print(f'\n{"Senator":22s} {"P(comp)":>8s} {"95% CI":>16s} {"Class":>12s}')
    print('-' * 62)
    for _, r in senator_df.head(15).iterrows():
        ci = f'[{r.ci_low:.3f}, {r.ci_high:.3f}]'
        print(f'{r.account:22s} {r.posterior_compulsive:8.4f} {ci:>16s} {r.classification:>12s}')
    print('...')
    for _, r in senator_df.tail(10).iterrows():
        ci = f'[{r.ci_low:.3f}, {r.ci_high:.3f}]'
        print(f'{r.account:22s} {r.posterior_compulsive:8.4f} {ci:>16s} {r.classification:>12s}')

    # Save everything
    output = {
        'model_params': model.get_params(),
        'cv_results': {k: v for k, v in cv.items() if k != 'details'},
        'cv_details': cv['details'],
    }
    with open('/home/joneill/x_box/pipeline/output/bayesian_model_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f'\nSaved:')
    print(f'  output/bayesian_model_results.json')
    print(f'  output/senator_bayesian_scores.csv')
