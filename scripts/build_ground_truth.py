"""Build ground truth cohort and run compulsion signatures on all accounts."""
import pandas as pd
import numpy as np
import os, json, sys, time

sys.path.insert(0, '/home/joneill/x_box/pipeline')
from xbox.compulsion_signatures import CompulsionSignatureAnalyzer
from sklearn.metrics import roc_auc_score
from scipy import stats

os.makedirs('/home/joneill/x_box/pipeline/data/ground_truth', exist_ok=True)

# Load data sources
trump = pd.read_parquet('/home/joneill/x_box/pipeline/data/trump_tweets.parquet')
trump['created_at'] = pd.to_datetime(trump['created_at'], utc=True, errors='coerce')
trump = trump.dropna(subset=['created_at'])
print(f'Trump: {len(trump)} tweets')

hf = pd.read_parquet('/home/joneill/x_box/pipeline/data/senator_tweets_hf.parquet')
hf['created_at'] = pd.to_datetime(hf['date'], errors='coerce', utc=True)
print(f'HF Senators: {len(hf)} tweets from {hf["username"].nunique()} senators')

lee = pd.read_excel('/home/joneill/x_box/BasedMikeLee_full_archive.xlsx')
lee['created_at'] = pd.to_datetime(lee['created_at'], utc=True, errors='coerce')
print(f'Mike Lee: {len(lee)} tweets')

# Define cohort
compulsive_accounts = [
    'Trump_Android', 'Trump_Full', 'BasedMikeLee',
    'SenTedCruz', 'MarshaBlackburn', 'HawleyMO', 'marcorubio', 'ChrisMurphyCT',
]
strategic_accounts = [
    'SenFeinstein', 'SenatorRisch', 'SenatorTester', 'SenHydeSmith',
    'SenShelby', 'SenJohnHoeven', 'SenGillibrand', 'SenatorCollins',
]

# Build cohort data
cohort = {}
cohort['Trump_Android'] = trump[trump['source'].str.contains('Android', na=False)].copy()
cohort['Trump_Full'] = trump.copy()
cohort['BasedMikeLee'] = lee.copy()

for name in compulsive_accounts + strategic_accounts:
    if name not in cohort:
        sub = hf[hf['username'] == name]
        if len(sub) > 50:
            cohort[name] = sub.copy()

print(f'\nCohort: {len(cohort)} accounts')

# Run signatures
analyzer = CompulsionSignatureAnalyzer()
results = []

for name, data in cohort.items():
    category = 'compulsive' if name in compulsive_accounts else 'strategic'
    sig = analyzer.analyze(data, senator_name=name, twitter_handle=name)
    results.append({
        'account': name,
        'category': category,
        'n_tweets': len(data),
        'p_compulsive': sig.compulsive_probability,
        'classification': sig.classification,
        'burstiness_B': sig.burstiness_B,
        'time_entropy': sig.time_entropy,
        'weekend_ratio': sig.weekend_ratio,
        'night_intensity': sig.night_intensity,
        'hawkes_n': sig.self_excitation_ratio,
        'gap_shape': sig.gap_distribution_shape,
        'z_state': sig.z_state,
    })
    print(f'  {name:22s} [{category:10s}] P={sig.compulsive_probability:.3f} B={sig.burstiness_B:+.3f}')

rdf = pd.DataFrame(results).sort_values('p_compulsive', ascending=False)
rdf.to_csv('/home/joneill/x_box/pipeline/data/ground_truth/cohort_signatures.csv', index=False)

# Print results
print('\n' + '=' * 80)
print('GROUND TRUTH COHORT')
print('=' * 80)
print(f'\n{"Account":22s} {"Label":>10s} {"P(comp)":>8s} {"B":>7s} {"Entropy":>8s} {"Night":>7s} {"n*":>6s} {"Weekend":>8s}')
print('-' * 75)
for _, r in rdf.iterrows():
    label = 'COMPULSIVE' if r.category == 'compulsive' else 'strategic'
    print(f'{r.account:22s} {label:>10s} {r.p_compulsive:8.3f} {r.burstiness_B:+7.3f} {r.time_entropy:8.2f} {r.night_intensity:7.1%} {r.hawkes_n:6.3f} {r.weekend_ratio:8.3f}')

# Separation analysis
comp = rdf[rdf.category == 'compulsive']
strat = rdf[rdf.category == 'strategic']

print(f'\n\nSEPARATION ANALYSIS (compulsive vs strategic):')
print(f'{"Feature":20s} {"Comp Mean":>10s} {"Strat Mean":>11s} {"t-stat":>8s} {"p-value":>9s} {"AUC":>6s}')
print('-' * 68)
for col in ['p_compulsive', 'burstiness_B', 'time_entropy', 'night_intensity', 'hawkes_n', 'weekend_ratio']:
    c = comp[col].dropna()
    s = strat[col].dropna()
    if len(c) > 1 and len(s) > 1:
        t_stat, p_val = stats.ttest_ind(c, s)
        labels = [1]*len(c) + [0]*len(s)
        scores = list(c) + list(s)
        auc = roc_auc_score(labels, scores)
        print(f'{col:20s} {c.mean():10.3f} {s.mean():11.3f} {t_stat:+8.2f} {p_val:9.4f} {auc:6.3f}')

print(f'\nSaved to pipeline/data/ground_truth/cohort_signatures.csv')
