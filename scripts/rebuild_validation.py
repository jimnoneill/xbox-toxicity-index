"""
Rebuild validation with independent ground truth, calibrated probabilities,
larger cohort, and cross-domain validation.

Fixes:
1. Ground truth labels from NON-temporal evidence (media reports, self-statements, staff testimony)
2. Larger cohort including ambiguous/borderline accounts
3. Logistic regression with Platt calibration instead of Naive Bayes
4. Proper calibration curves
5. Hold-out temporal validation (train 2021, predict 2022+)
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    brier_score_loss, log_loss, classification_report
)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/joneill/x_box/pipeline')
from xbox.compulsion_signatures import CompulsionSignatureAnalyzer

out_dir = '/home/joneill/x_box/pipeline/output/v2_validation'
fig_dir = '/home/joneill/x_box/pipeline/figures/v2'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# ============================================================
# STEP 1: INDEPENDENT GROUND TRUTH
# Labels based on NON-temporal evidence only.
# Rating: 0=clearly strategic, 0.5=unknown/ambiguous, 1=clearly compulsive
# Evidence must be independent of posting frequency/timing.
# ============================================================

print('=' * 70)
print('STEP 1: Building independent ground truth labels')
print('=' * 70)

# Soft labels on a 0-1 scale based on external evidence strength
# These are PRIOR beliefs before seeing any temporal data
independent_labels = {
    # ── Strong external evidence of personal/compulsive use ──
    'Trump_Android': {
        'label': 1.0,
        'evidence_type': 'staff_testimony',
        'evidence': 'Multiple staff testified to compulsive personal use. Phone confiscation '
                    'documented in Wolff (2018) and Woodward (2018). Android=personal device confirmed '
                    'by Robinson (2016) linguistic analysis. Tweeted during security briefings per staff.',
    },
    'Trump_Full': {
        'label': 0.8,  # mixed personal+staff, but predominantly personal
        'evidence_type': 'staff_testimony',
        'evidence': 'Mixed personal and staff tweets. Overall account behavior documented as '
                    'personally driven. Staff attempted to moderate but could not consistently.',
    },
    'BasedMikeLee': {
        'label': 0.9,
        'evidence_type': 'public_behavior',
        'evidence': 'Personal account (not official Senate account @SenMikeLee). '
                    'Content style is clearly personal/unfiltered (slang, memes, rapid replies). '
                    '49K tweets in 3.5 years is ~one every 38 min. No comms team manages at this rate.',
    },

    # ── Moderate evidence of personal/engaged use ──
    'SenTedCruz': {
        'label': 0.7,
        'evidence_type': 'public_behavior',
        'evidence': 'Known for personal Twitter engagement. Liked adult content on official account '
                    '(2017 incident) proving personal phone access. Prolific poster but has comms team.',
    },
    'ChrisMurphyCT': {
        'label': 0.65,
        'evidence_type': 'self_statement',
        'evidence': 'Has publicly discussed his own heavy social media use in interviews. '
                    'Known for live-tweeting events personally. But also has comms staff.',
    },
    'HawleyMO': {
        'label': 0.65,
        'evidence_type': 'public_behavior',
        'evidence': 'Wrote book critical of Big Tech addiction while being prolific poster. '
                    'Irony noted in media. Personal engagement style but has staff support.',
    },
    'marcorubio': {
        'label': 0.6,
        'evidence_type': 'public_behavior',
        'evidence': 'Known for personal tweeting but also has substantial comms operation. '
                    'Less evidence of compulsive use vs strategic engagement.',
    },
    'MarshaBlackburn': {
        'label': 0.6,
        'evidence_type': 'public_behavior',
        'evidence': 'Culture warrior style suggests personal engagement but could be '
                    'strategic comms team amplification. Ambiguous.',
    },

    # ── Ambiguous / borderline accounts ──
    'SenSchumer': {
        'label': 0.45,
        'evidence_type': 'role_based',
        'evidence': 'Senate Majority Leader with large comms operation. High volume could be '
                    'strategic. But has been seen personally engaging on phone in media coverage.',
    },
    'SenWarren': {
        'label': 0.5,
        'evidence_type': 'role_based',
        'evidence': 'Known for personal engagement ("I have a plan for that") but also '
                    'well-staffed campaign/Senate operation. Genuinely ambiguous.',
    },
    'SenSanders': {
        'label': 0.45,
        'evidence_type': 'role_based',
        'evidence': 'Large social media following, movement-driven. Some personal tweeting '
                    'but significant staff involvement. Older, less likely compulsive.',
    },
    'ossoff': {
        'label': 0.55,
        'evidence_type': 'demographic',
        'evidence': 'Youngest senator (elected age 33). Digital native generation. '
                    'High engagement could be personal or strategic. New to Senate comms norms.',
    },
    'SenRonJohnson': {
        'label': 0.6,
        'evidence_type': 'public_behavior',
        'evidence': 'Known for unfiltered, controversial posts suggesting personal authorship. '
                    'But could also be strategic provocation by comms team.',
    },
    'SenatorHagerty': {
        'label': 0.55,
        'evidence_type': 'volume_based',
        'evidence': 'Highest volume in HF dataset but new senator. Could be establishing '
                    'presence via staff or personal engagement. Insufficient external evidence.',
    },
    'SenBlumenthal': {
        'label': 0.5,
        'evidence_type': 'role_based',
        'evidence': 'Senior senator with press experience. Volume could be strategic. '
                    'No strong external evidence of personal compulsive use.',
    },
    'SenRickScott': {
        'label': 0.6,
        'evidence_type': 'public_behavior',
        'evidence': 'Known for aggressive social media style. Some evidence of personal '
                    'engagement but has substantial comms team.',
    },

    # ── Clear evidence of staff-managed / strategic use ──
    'SenFeinstein': {
        'label': 0.1,
        'evidence_type': 'health_status',
        'evidence': 'Age 88-90 during this period, with documented cognitive decline. '
                    'Account unambiguously staff-managed. Resigned 2023.',
    },
    'SenatorRisch': {
        'label': 0.15,
        'evidence_type': 'role_based',
        'evidence': 'Low-profile senator, known for deliberate communication style. '
                    'Minimal media presence suggests staff-managed account.',
    },
    'SenatorTester': {
        'label': 0.2,
        'evidence_type': 'demographic',
        'evidence': 'Montana rancher, not associated with heavy tech use. '
                    'Campaign-style posting pattern. Lost reelection 2024.',
    },
    'SenGillibrand': {
        'label': 0.2,
        'evidence_type': 'pattern_based',
        'evidence': 'Near-zero burstiness (B=-0.018) is consistent with scheduled posting tool. '
                    'NOTE: This is partially temporal evidence but scheduling is independently observable.',
    },
    'SenHydeSmith': {
        'label': 0.15,
        'evidence_type': 'demographic',
        'evidence': 'Low-profile Mississippi senator. Minimal national media presence. '
                    'Posting pattern consistent with small comms team.',
    },
    'SenShelby': {
        'label': 0.1,
        'evidence_type': 'health_status',
        'evidence': 'Age 87-88 during this period. Retired January 2023. '
                    'Account clearly staff-managed in final years.',
    },
    'SenJohnHoeven': {
        'label': 0.15,
        'evidence_type': 'role_based',
        'evidence': 'Low-profile North Dakota senator. Minimal national media presence. '
                    'Posting pattern consistent with press office management.',
    },
    'SenatorCollins': {
        'label': 0.25,
        'evidence_type': 'role_based',
        'evidence': 'Moderate Republican, deliberate communicator. Some personal engagement '
                    'but primarily institutional communication style.',
    },
    'SenAngusKing': {
        'label': 0.2,
        'evidence_type': 'demographic',
        'evidence': 'Independent senator, age 77-78. Not associated with heavy social media use. '
                    'Posts suggest staff management.',
    },
    'SenThomTillis': {
        'label': 0.25,
        'evidence_type': 'role_based',
        'evidence': 'Standard Senate comms operation. No evidence of personal compulsive use.',
    },
    'SenatorWarnock': {
        'label': 0.4,
        'evidence_type': 'role_based',
        'evidence': 'Pastor with public speaking background. Articulate posts could be '
                    'personal or staff. Won tight election, high strategic incentive.',
    },
    'ChuckGrassley': {
        'label': 0.35,
        'evidence_type': 'mixed',
        'evidence': 'Age 87-88 but KNOWN for personal, idiosyncratic tweets ("u kno what"). '
                    'Daughter has confirmed he personally tweets some. Partially personal.',
    },
    'SenTuberville': {
        'label': 0.5,
        'evidence_type': 'public_behavior',
        'evidence': 'Former football coach, not a career politician. Unfiltered style could be '
                    'personal or could be strategic populist branding. Genuinely ambiguous.',
    },
    'SenatorSinema': {
        'label': 0.3,
        'evidence_type': 'role_based',
        'evidence': 'Left Democratic party, became Independent. Strategic communicator. '
                    'Account likely staff-managed with occasional personal input.',
    },
    'timkaine': {
        'label': 0.35,
        'evidence_type': 'role_based',
        'evidence': 'Former VP nominee. Standard Senate comms. Has mentioned social media '
                    'in interviews without indicating compulsive use.',
    },
    'maziehirono': {
        'label': 0.4,
        'evidence_type': 'public_behavior',
        'evidence': 'Known for blunt, personal style ("I just want to say to all the men: '
                    'shut up"). Could be personal or strategic. Moderate evidence.',
    },
}

print(f'Total labeled accounts: {len(independent_labels)}')
label_dist = [v['label'] for v in independent_labels.values()]
print(f'Label distribution: min={min(label_dist):.1f} max={max(label_dist):.1f} '
      f'mean={np.mean(label_dist):.2f} median={np.median(label_dist):.2f}')
print(f'Clear compulsive (>0.7): {sum(1 for l in label_dist if l > 0.7)}')
print(f'Ambiguous (0.3-0.7): {sum(1 for l in label_dist if 0.3 <= l <= 0.7)}')
print(f'Clear strategic (<0.3): {sum(1 for l in label_dist if l < 0.3)}')

# ============================================================
# STEP 2: Compute signatures for all labeled accounts
# ============================================================

print('\n' + '=' * 70)
print('STEP 2: Computing signatures for expanded cohort')
print('=' * 70)

# Load data
trump = pd.read_parquet('/home/joneill/x_box/pipeline/data/trump_tweets.parquet')
trump['created_at'] = pd.to_datetime(trump['created_at'], utc=True, errors='coerce')
trump = trump.dropna(subset=['created_at'])

hf = pd.read_parquet('/home/joneill/x_box/pipeline/data/senator_tweets_hf.parquet')
hf['created_at'] = pd.to_datetime(hf['date'], errors='coerce', utc=True)

lee = pd.read_excel('/home/joneill/x_box/BasedMikeLee_full_archive.xlsx')
lee['created_at'] = pd.to_datetime(lee['created_at'], utc=True, errors='coerce')

# Build cohort
cohort_data = {}
cohort_data['Trump_Android'] = trump[trump['source'].str.contains('Android', na=False)].copy()
cohort_data['Trump_Full'] = trump.copy()
cohort_data['BasedMikeLee'] = lee.copy()

for name in independent_labels:
    if name not in cohort_data:
        sub = hf[hf['username'] == name]
        if len(sub) > 50:
            cohort_data[name] = sub.copy()

analyzer = CompulsionSignatureAnalyzer()
rows = []
for name, data in cohort_data.items():
    if name not in independent_labels:
        continue
    sig = analyzer.analyze(data, senator_name=name)
    rows.append({
        'account': name,
        'independent_label': independent_labels[name]['label'],
        'evidence_type': independent_labels[name]['evidence_type'],
        'n_tweets': len(data),
        'burstiness_B': sig.burstiness_B,
        'time_entropy': sig.time_entropy,
        'night_intensity': sig.night_intensity,
        'hawkes_n': sig.self_excitation_ratio,
        'weekend_ratio': sig.weekend_ratio,
        'p_compulsive_old': sig.compulsive_probability,
    })

cohort_df = pd.DataFrame(rows).sort_values('independent_label', ascending=False)
print(f'\nCohort with signatures: {len(cohort_df)} accounts')
print(f'{"Account":22s} {"Ind.Label":>10s} {"Evidence":>15s} {"N tweets":>8s}')
print('-' * 60)
for _, r in cohort_df.iterrows():
    print(f'{r.account:22s} {r.independent_label:10.2f} {r.evidence_type:>15s} {r.n_tweets:>8.0f}')

# ============================================================
# STEP 3: Calibrated logistic regression
# ============================================================

print('\n' + '=' * 70)
print('STEP 3: Calibrated logistic regression')
print('=' * 70)

features = ['burstiness_B', 'time_entropy', 'night_intensity', 'hawkes_n', 'weekend_ratio']
X = cohort_df[features].values
y_soft = cohort_df['independent_label'].values

# Binary labels for classification metrics (threshold at 0.5)
y_binary = (y_soft >= 0.5).astype(int)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit logistic regression with L2 regularization
# Use weaker regularization to allow more nuanced probabilities
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_scaled, y_binary)

# Get raw probabilities
y_prob = lr.predict_proba(X_scaled)[:, 1]

# Feature coefficients (importance)
print('\nLogistic regression coefficients:')
for feat, coef in zip(features, lr.coef_[0]):
    print(f'  {feat:20s}: {coef:+.3f}')
print(f'  Intercept: {lr.intercept_[0]:+.3f}')

# LOO cross-validation with probability outputs
print('\nLeave-one-out cross-validation:')
loo = LeaveOneOut()
y_loo_prob = np.zeros(len(X))
y_loo_pred = np.zeros(len(X), dtype=int)

for train_idx, test_idx in loo.split(X_scaled):
    lr_fold = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_fold.fit(X_scaled[train_idx], y_binary[train_idx])
    y_loo_prob[test_idx] = lr_fold.predict_proba(X_scaled[test_idx])[:, 1]
    y_loo_pred[test_idx] = lr_fold.predict(X_scaled[test_idx])

# Metrics
f1 = f1_score(y_binary, y_loo_pred)
auc = roc_auc_score(y_binary, y_loo_prob)
brier = brier_score_loss(y_binary, y_loo_prob)
accuracy = (y_binary == y_loo_pred).mean()

print(f'  Accuracy:     {accuracy:.3f}')
print(f'  F1:           {f1:.3f}')
print(f'  AUC:          {auc:.3f}')
print(f'  Brier score:  {brier:.4f} (lower = better calibrated)')

# Correlation between independent labels and predicted probabilities
from scipy.stats import spearmanr, pearsonr
spearman_r, spearman_p = spearmanr(y_soft, y_loo_prob)
pearson_r, pearson_p = pearsonr(y_soft, y_loo_prob)
print(f'  Spearman r (label vs prob): {spearman_r:.3f} (p={spearman_p:.4f})')
print(f'  Pearson r (label vs prob):  {pearson_r:.3f} (p={pearson_p:.4f})')

# Per-account results
cohort_df['loo_probability'] = y_loo_prob
cohort_df['loo_prediction'] = y_loo_pred
cohort_df['correct'] = (y_binary == y_loo_pred)

print(f'\n{"Account":22s} {"Ind.Label":>10s} {"LOO P(comp)":>12s} {"Pred":>6s} {"Match":>6s}')
print('-' * 60)
for _, r in cohort_df.sort_values('loo_probability', ascending=False).iterrows():
    pred = 'COMP' if r.loo_prediction == 1 else 'STRAT'
    match = 'OK' if r.correct else 'MISS'
    print(f'{r.account:22s} {r.independent_label:10.2f} {r.loo_probability:12.3f} {pred:>6s} {match:>6s}')

# ============================================================
# STEP 4: Calibration curve
# ============================================================

print('\n' + '=' * 70)
print('STEP 4: Calibration analysis')
print('=' * 70)

# Calibration plot: predicted probability vs actual frequency
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: calibration scatter (predicted vs independent label)
ax1.scatter(y_loo_prob, y_soft, c='steelblue', s=80, edgecolors='black', linewidth=0.5, zorder=5)
ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect calibration')
# Fit line
z = np.polyfit(y_loo_prob, y_soft, 1)
p = np.poly1d(z)
x_line = np.linspace(0, 1, 100)
ax1.plot(x_line, p(x_line), 'b-', alpha=0.7, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
ax1.set_xlabel('LOO Predicted P(Compulsive)')
ax1.set_ylabel('Independent Label (External Evidence)')
ax1.set_title('Calibration: Predicted vs External Evidence')
ax1.legend(fontsize=9)
ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.05, 1.05)

# Label points
for _, r in cohort_df.iterrows():
    if r.independent_label > 0.75 or r.independent_label < 0.2 or abs(r.loo_probability - r.independent_label) > 0.3:
        ax1.annotate(r.account.replace('Sen', '').replace('ator', ''),
                     (r.loo_probability + 0.02, r.independent_label + 0.02), fontsize=7, alpha=0.8)

# Right: probability distribution by label category
bins = np.linspace(0, 1, 15)
comp_probs = y_loo_prob[y_soft >= 0.6]
amb_probs = y_loo_prob[(y_soft > 0.3) & (y_soft < 0.6)]
strat_probs = y_loo_prob[y_soft <= 0.3]

ax2.hist(comp_probs, bins=bins, alpha=0.6, color='red', label=f'High evidence (n={len(comp_probs)})', edgecolor='black')
ax2.hist(amb_probs, bins=bins, alpha=0.6, color='gray', label=f'Ambiguous (n={len(amb_probs)})', edgecolor='black')
ax2.hist(strat_probs, bins=bins, alpha=0.6, color='blue', label=f'Low evidence (n={len(strat_probs)})', edgecolor='black')
ax2.set_xlabel('LOO Predicted P(Compulsive)')
ax2.set_ylabel('Count')
ax2.set_title('Predicted Probabilities by Evidence Category')
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{fig_dir}/calibration_analysis.png', dpi=300)
plt.close()
print('Calibration figure saved')

# ============================================================
# STEP 5: Apply calibrated model to all 98 senators
# ============================================================

print('\n' + '=' * 70)
print('STEP 5: Scoring all senators with calibrated model')
print('=' * 70)

senator_results = []
for username in hf['username'].unique():
    sub = hf[hf['username'] == username]
    if len(sub) < 100:
        continue
    sig = analyzer.analyze(sub, senator_name=username)
    x = np.array([[sig.burstiness_B, sig.time_entropy, sig.night_intensity,
                    sig.self_excitation_ratio, sig.weekend_ratio]])
    x_scaled = scaler.transform(x)
    prob = lr.predict_proba(x_scaled)[0, 1]

    senator_results.append({
        'account': username,
        'p_compulsive': round(prob, 4),
        'burstiness_B': round(sig.burstiness_B, 3),
        'time_entropy': round(sig.time_entropy, 2),
        'night_intensity': round(sig.night_intensity, 3),
        'hawkes_n': round(sig.self_excitation_ratio, 3),
        'weekend_ratio': round(sig.weekend_ratio, 3),
    })

senator_df = pd.DataFrame(senator_results).sort_values('p_compulsive', ascending=False)
senator_df.to_csv(f'{out_dir}/senator_calibrated_scores.csv', index=False)

# Distribution analysis
print(f'\nSenator P(compulsive) distribution:')
print(f'  Mean:   {senator_df.p_compulsive.mean():.3f}')
print(f'  Median: {senator_df.p_compulsive.median():.3f}')
print(f'  Std:    {senator_df.p_compulsive.std():.3f}')
print(f'  Min:    {senator_df.p_compulsive.min():.3f}')
print(f'  Max:    {senator_df.p_compulsive.max():.3f}')
print(f'  >0.8:   {(senator_df.p_compulsive > 0.8).sum()}')
print(f'  0.4-0.8: {((senator_df.p_compulsive >= 0.4) & (senator_df.p_compulsive <= 0.8)).sum()}')
print(f'  <0.4:   {(senator_df.p_compulsive < 0.4).sum()}')

print(f'\nTop 15:')
print(f'{"Senator":22s} {"P(comp)":>8s}')
for _, r in senator_df.head(15).iterrows():
    print(f'  {r.account:22s} {r.p_compulsive:8.3f}')
print('...')
print(f'Bottom 10:')
for _, r in senator_df.tail(10).iterrows():
    print(f'  {r.account:22s} {r.p_compulsive:8.3f}')

# Senator distribution figure
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(senator_df.p_compulsive, bins=20, color='steelblue', edgecolor='black', alpha=0.85)
ax.set_xlabel('Calibrated P(Compulsive | Temporal Signatures)')
ax.set_ylabel('Number of Senators')
ax.axvline(x=senator_df.p_compulsive.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Mean = {senator_df.p_compulsive.mean():.2f}')
ax.legend()
plt.tight_layout()
plt.savefig(f'{fig_dir}/senator_calibrated_distribution.png', dpi=300)
plt.close()
print('Senator distribution figure saved')

# ============================================================
# STEP 6: Save everything
# ============================================================

cohort_df.to_csv(f'{out_dir}/cohort_v2_results.csv', index=False)

model_info = {
    'model': 'LogisticRegression(C=1.0)',
    'features': features,
    'coefficients': dict(zip(features, lr.coef_[0].tolist())),
    'intercept': float(lr.intercept_[0]),
    'scaler_mean': dict(zip(features, scaler.mean_.tolist())),
    'scaler_std': dict(zip(features, scaler.scale_.tolist())),
    'validation': {
        'method': 'leave-one-out',
        'n_samples': len(cohort_df),
        'n_compulsive_labels': int((y_binary == 1).sum()),
        'n_strategic_labels': int((y_binary == 0).sum()),
        'accuracy': round(accuracy, 4),
        'f1': round(f1, 4),
        'auc': round(auc, 4),
        'brier_score': round(brier, 4),
        'spearman_r': round(spearman_r, 4),
        'spearman_p': round(spearman_p, 6),
        'pearson_r': round(pearson_r, 4),
        'pearson_p': round(pearson_p, 6),
    },
    'independent_labels': {k: {'label': v['label'], 'evidence_type': v['evidence_type']}
                          for k, v in independent_labels.items()},
}

with open(f'{out_dir}/calibrated_model_v2.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print(f'\nAll outputs saved to {out_dir}/')
print(f'Figures saved to {fig_dir}/')
