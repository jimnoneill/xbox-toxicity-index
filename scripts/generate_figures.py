"""Generate all publication figures for the IJDP manuscript."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
from math import pi

sys.path.insert(0, '/home/joneill/x_box/pipeline')
from xbox.compulsion_signatures import CompulsionSignatureAnalyzer

os.makedirs('/home/joneill/x_box/pipeline/figures', exist_ok=True)

cohort = pd.read_csv('/home/joneill/x_box/pipeline/data/ground_truth/cohort_signatures.csv')
senators = pd.read_csv('/home/joneill/x_box/pipeline/output/senator_bayesian_scores.csv')

plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# ── Figure 1: Ground truth separation ──
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
comp = cohort[cohort.category == 'compulsive']
strat = cohort[cohort.category == 'strategic']
ax.scatter(comp.time_entropy, comp.night_intensity, c='#d32f2f', s=120, zorder=5,
           label='Known Compulsive (n=8)', edgecolors='darkred', linewidth=1.5, marker='o')
ax.scatter(strat.time_entropy, strat.night_intensity, c='#1565c0', s=120, zorder=5,
           label='Known Strategic (n=8)', edgecolors='darkblue', linewidth=1.5, marker='s')
for _, r in cohort.iterrows():
    dx = 0.04
    dy = 0.012 if r.category == 'compulsive' else -0.012
    ax.annotate(r.account.replace('Sen', '').replace('ator', ''),
                (r.time_entropy + dx, r.night_intensity + dy), fontsize=7, alpha=0.85)
ax.set_xlabel('Time-of-Day Entropy (bits)')
ax.set_ylabel('Night Posting Intensity (00:00-05:59 UTC share)')
ax.legend(fontsize=10, loc='upper left')
ax.set_xlim(3.1, 4.55)
# Decision boundary region
ax.axhspan(0.05, 0.12, alpha=0.08, color='gray')
ax.axvspan(3.7, 3.85, alpha=0.08, color='gray')
plt.tight_layout()
plt.savefig('/home/joneill/x_box/pipeline/figures/fig1_ground_truth_separation.png', dpi=300)
plt.close()
print('Figure 1: Ground truth separation')

# ── Figure 2: Feature importance ──
fig, ax = plt.subplots(1, 1, figsize=(7, 4))
features = ['Night Intensity', 'Time-of-Day Entropy', 'Burstiness (B)', 'Hawkes n*', 'Weekend Ratio']
importance = [28.077, 7.952, 4.783, 4.630, 0.053]
colors = ['#d32f2f', '#e64a19', '#f57c00', '#ffa000', '#bdbdbd']
bars = ax.barh(features[::-1], importance[::-1], color=colors[::-1], edgecolor='black', linewidth=0.5)
ax.set_xlabel('Mean |Log-Likelihood Ratio| Contribution')
for bar, val in zip(bars, importance[::-1]):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('/home/joneill/x_box/pipeline/figures/fig2_feature_importance.png', dpi=300)
plt.close()
print('Figure 2: Feature importance')

# ── Figure 3: Senator posterior distribution ──
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
bins = np.linspace(0, 1, 21)
ax.hist(senators.posterior_compulsive, bins=bins, color='steelblue', edgecolor='black', alpha=0.85)
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision threshold')
n_comp = (senators.posterior_compulsive > 0.5).sum()
n_strat = (senators.posterior_compulsive <= 0.5).sum()
ax.text(0.75, ax.get_ylim()[1]*0.85, f'Compulsive: {n_comp}', fontsize=11, color='red')
ax.text(0.05, ax.get_ylim()[1]*0.85, f'Strategic: {n_strat}', fontsize=11, color='blue')
ax.set_xlabel('Posterior P(Compulsive | Temporal Signatures)')
ax.set_ylabel('Number of Senators')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('/home/joneill/x_box/pipeline/figures/fig3_senator_posterior_distribution.png', dpi=300)
plt.close()
print('Figure 3: Senator posterior distribution')

# ── Figure 4: Mike Lee temporal trajectory ──
lee = pd.read_excel('/home/joneill/x_box/BasedMikeLee_full_archive.xlsx')
lee['created_at'] = pd.to_datetime(lee['created_at'], utc=True, errors='coerce')
lee = lee.dropna(subset=['created_at']).sort_values('created_at')
weekly = lee.set_index('created_at').resample('W').size()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax1.plot(weekly.index, weekly.values, color='steelblue', linewidth=0.6, alpha=0.5)
ax1.plot(weekly.index, weekly.rolling(8).mean(), color='#1565c0', linewidth=2.5, label='8-week MA')
ax1.set_ylabel('Tweets per Week')
ax1.legend(loc='upper left')
ax1.annotate('Escalation onset', xy=(pd.Timestamp('2023-06-01', tz='UTC'), 200),
             fontsize=9, color='red', style='italic')

lee['is_night'] = lee['created_at'].dt.hour.between(0, 5)
night_w = lee.set_index('created_at')['is_night'].resample('W').mean()
ax2.plot(night_w.index, night_w.values, color='purple', linewidth=0.6, alpha=0.5)
ax2.plot(night_w.index, night_w.rolling(8).mean(), color='#6a1b9a', linewidth=2.5, label='8-week MA')
ax2.axhline(y=0.25, color='red', linestyle='--', alpha=0.6, label='Uniform baseline')
ax2.set_ylabel('Night Posting Share (00-05h UTC)')
ax2.set_xlabel('Date')
ax2.legend(loc='upper left')
plt.tight_layout()
plt.savefig('/home/joneill/x_box/pipeline/figures/fig4_mikelee_trajectory.png', dpi=300)
plt.close()
print('Figure 4: Mike Lee trajectory')

# ── Figure 5: Radar comparison ──
categories_r = ['Burstiness\n(B)', 'Time\nEntropy', 'Night\nIntensity', 'Hawkes\nn*', 'Weekend\nRatio']
N = len(categories_r)
mins = [-0.1, 3.0, 0.0, 0.7, 0.3]
maxs = [0.8, 4.5, 0.45, 1.0, 1.1]

def norm(vals):
    return [(v - mn) / (mx - mn) for v, mn, mx in zip(vals, mins, maxs)]

profiles = {
    'Mike Lee': (norm([0.755, 4.21, 0.402, 0.984, 0.838]), '#d32f2f'),
    'Trump (Android)': (norm([0.525, 4.28, 0.397, 0.980, 1.033]), '#ff6f00'),
    'Strategic Mean': (norm([0.105, 3.54, 0.037, 0.816, 0.664]), '#1565c0'),
}

angles = [n / float(N) * 2 * pi for n in range(N)] + [0]
fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(polar=True))
for label, (vals, color) in profiles.items():
    vals_plot = vals + vals[:1]
    ax.plot(angles, vals_plot, linewidth=2.5, label=label, color=color)
    ax.fill(angles, vals_plot, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_r, fontsize=10)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
plt.tight_layout()
plt.savefig('/home/joneill/x_box/pipeline/figures/fig5_radar_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print('Figure 5: Radar comparison')

# ── Figure 6: Trump escalation by year ──
trump = pd.read_parquet('/home/joneill/x_box/pipeline/data/trump_tweets.parquet')
trump['created_at'] = pd.to_datetime(trump['created_at'], utc=True, errors='coerce')
trump = trump.dropna(subset=['created_at'])

analyzer = CompulsionSignatureAnalyzer()
years, p_comps, nights, burstinesses = [], [], [], []
for year in range(2011, 2018):
    sub = trump[trump['created_at'].dt.year == year]
    if len(sub) < 50: continue
    sig = analyzer.analyze(sub)
    years.append(year)
    p_comps.append(sig.compulsive_probability)
    nights.append(sig.night_intensity)
    burstinesses.append(sig.burstiness_B)

fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
ax1.plot(years, p_comps, 'o-', color='#d32f2f', linewidth=2.5, markersize=10, label='P(Compulsive)')
ax1.set_ylabel('P(Compulsive)', color='#d32f2f')
ax1.set_xlabel('Year')
ax1.set_ylim(0.4, 1.0)
ax1.tick_params(axis='y', labelcolor='#d32f2f')
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4)

ax2 = ax1.twinx()
ax2.plot(years, nights, 's--', color='#6a1b9a', linewidth=2, markersize=8, label='Night intensity')
ax2.set_ylabel('Night Posting Share', color='#6a1b9a')
ax2.tick_params(axis='y', labelcolor='#6a1b9a')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)

ax1.annotate('Campaign\nannouncement', xy=(2015, 0.882), xytext=(2014.2, 0.92),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'), color='gray')
plt.tight_layout()
plt.savefig('/home/joneill/x_box/pipeline/figures/fig6_trump_escalation.png', dpi=300)
plt.close()
print('Figure 6: Trump escalation')

# ── Figure 7: ROC curves per feature ──
from sklearn.metrics import roc_curve, auc as sk_auc

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
y_true = [1 if c == 'compulsive' else 0 for c in cohort.category]
feature_labels = {
    'night_intensity': ('Night Intensity', '#d32f2f'),
    'time_entropy': ('Time Entropy', '#e64a19'),
    'hawkes_n': ('Hawkes n*', '#f57c00'),
    'burstiness_B': ('Burstiness B', '#ffa000'),
    'weekend_ratio': ('Weekend Ratio', '#bdbdbd'),
}
for feat, (label, color) in feature_labels.items():
    fpr, tpr, _ = roc_curve(y_true, cohort[feat])
    roc_auc = sk_auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{label} (AUC={roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=9, loc='lower right')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.savefig('/home/joneill/x_box/pipeline/figures/fig7_roc_curves.png', dpi=300)
plt.close()
print('Figure 7: ROC curves')

print('\nAll figures generated.')
