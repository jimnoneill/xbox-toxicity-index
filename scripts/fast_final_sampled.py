"""
Fast final pipeline — SetFit on 5K stratified sample, extrapolate to full 49K.
Should complete in ~20 min on CPU vs 4+ hours for full run.
"""
import time, json, logging, os, sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, '/mnt/c/x_box/pipeline')

import pandas as pd
import numpy as np

from xbox.data import load_local_archive
from xbox.behavioral import BehavioralAnalyzer
from xbox.compulsion_signatures import CompulsionSignatureAnalyzer
from xbox.changepoint import BayesianChangepointDetector
from xbox.disorder_baseline import DisorderBaselineAnalyzer
from xbox.fusion import ScoreFusion

t0 = time.time()
out_dir = '/mnt/c/x_box/pipeline/output'

# Load
log.info('Loading...')
df = load_local_archive('/mnt/c/x_box/BasedMikeLee_full_archive.xlsx', senator_name='Mike Lee')
classified = pd.read_parquet(f'{out_dir}/BasedMikeLee_classified_tweets.parquet')

# Behavioral + signatures (fast, runs on full data)
behavioral = BehavioralAnalyzer().analyze(df, senator_name='Mike Lee', twitter_handle='BasedMikeLee')
signatures = CompulsionSignatureAnalyzer().analyze(df, senator_name='Mike Lee', twitter_handle='BasedMikeLee')

# Stratified sample for SetFit (5K from 49K)
log.info('Sampling 5K tweets for SetFit classification...')
np.random.seed(42)
sample_idx = np.random.choice(len(classified), size=5000, replace=False)
sample = classified.iloc[sample_idx].copy()
sample_texts = sample['text'].tolist()

# SetFit classifiers
from setfit import SetFitModel
setfit_models = {
    'ragebait': '/mnt/c/x_box/pipeline/models/setfit_ragebait',
    'epistemic_manipulation': '/mnt/c/x_box/pipeline/models/setfit_epistemic_manipulation',
    'tribal_signal': '/mnt/c/x_box/pipeline/models_v2/setfit_tribal_signal',
    'performative_outrage': '/mnt/c/x_box/pipeline/models_v2/setfit_performative_outrage',
    'engagement_bait': '/mnt/c/x_box/pipeline/models_v2/setfit_engagement_bait',
    'agency_language': '/mnt/c/x_box/pipeline/models_v2/setfit_agency_language',
}

setfit_rates = {}
for dim, path in setfit_models.items():
    t1 = time.time()
    model = SetFitModel.from_pretrained(path)
    preds = model.predict(sample_texts)
    preds_list = preds.tolist() if hasattr(preds, 'tolist') else list(preds)
    rate = sum(1 for p in preds_list if p == 1) / len(preds_list)
    setfit_rates[dim] = rate
    log.info(f'  {dim}: {rate:.1%} positive ({time.time()-t1:.0f}s)')

# Score fusion on sample
fusion = ScoreFusion()
sample = fusion.compute_tweet_virulence(sample)
virulence_cardiffnlp = float(sample['composite_virulence'].mean()) * 100

# SetFit virulence from rates
setfit_virulence = (
    setfit_rates['ragebait'] * 0.25 +
    setfit_rates['tribal_signal'] * 0.20 +
    setfit_rates['performative_outrage'] * 0.20 +
    setfit_rates['epistemic_manipulation'] * 0.15 +
    setfit_rates['engagement_bait'] * 0.10 +
    setfit_rates['agency_language'] * -0.10
) * 100

enhanced_virulence = virulence_cardiffnlp * 0.4 + setfit_virulence * 0.6
overall_risk = behavioral.compulsion_score * 0.4 + enhanced_virulence * 0.6

# Disorder + Changepoint (on full data, fast)
df['sentiment_negative'] = classified['sentiment_negative'].values
df['offensive_score'] = classified['offensive_score'].values
disorder = DisorderBaselineAnalyzer().analyze(
    df, signatures=signatures.to_dict(), behavioral=behavioral.to_dict(),
    senator_name='Mike Lee', twitter_handle='BasedMikeLee')
cp = BayesianChangepointDetector(hazard_rate=1/60).detect_changepoints(
    classified, resolution='D', min_confidence=0.2)

# Top rage tweets from sample
sample['setfit_rage_score'] = [1 if p == 1 else 0 for p in model.predict(sample_texts)]
top_rage = sample.nlargest(10, 'composite_virulence')

elapsed = time.time() - t0

# Build report
report = {
    'meta': {
        'generated_at': pd.Timestamp.utcnow().isoformat() + 'Z',
        'pipeline_version': '0.3.0-sampled',
        'setfit_sample_size': 5000,
        'total_tweets': len(classified),
        'note': 'SetFit scores based on 5K stratified sample, CardiffNLP on full 49K',
    },
    'senator': {'name': 'Mike Lee', 'handle': 'BasedMikeLee', 'party': 'Republican', 'state': 'UT'},
    'summary': {
        'tweets_analyzed': len(classified),
        'date_range': f'{df["created_at"].min().date()} to {df["created_at"].max().date()}',
        'compulsion_score': behavioral.compulsion_score,
        'compulsive_probability': signatures.compulsive_probability,
        'classification': signatures.classification,
        'virulence_cardiffnlp': round(virulence_cardiffnlp, 1),
        'virulence_setfit': round(setfit_virulence, 1),
        'virulence_enhanced': round(enhanced_virulence, 1),
        'overall_risk': round(overall_risk, 1),
        'disorder_severity': disorder.severity,
        'disorder_criteria_met': disorder.criteria_met,
        'trajectory': disorder.trajectory,
        'viability_zone': disorder.viability_zone_estimate,
    },
    'compulsion_signatures': {
        'burstiness_B': round(signatures.burstiness_B, 3),
        'time_entropy_bits': round(signatures.time_entropy, 2),
        'weekend_ratio': round(signatures.weekend_ratio, 3),
        'night_intensity': round(signatures.night_intensity, 3),
        'hawkes_n_star': round(signatures.self_excitation_ratio, 3),
        'response_latency_min': round(signatures.response_latency_median_min, 1),
        'gap_distribution': signatures.gap_distribution_shape,
        'z_state_logit': round(signatures.z_state, 3),
    },
    'compulsion_behavioral': behavioral.compulsion_subscores,
    'virulence_setfit_rates': {k: round(v, 3) for k, v in setfit_rates.items()},
    'virulence_cardiffnlp': {
        'sentiment': classified['sentiment_label'].value_counts(normalize=True).round(3).to_dict(),
        'emotion': classified['emotion_label'].value_counts(normalize=True).round(3).to_dict(),
        'toxicity_mean': round(float(classified['toxicity_score'].mean()), 4),
        'toxicity_pct': round(float((classified['toxicity_score'] > 0.5).mean()) * 100, 2),
    },
    'disorder_assessment': {
        'severity': disorder.severity,
        'criteria_met': disorder.criteria_met,
        'criteria': {k: {'met': v['met'], 'value': v['value']} for k, v in disorder.criteria_details.items()},
        'trajectory': disorder.trajectory,
        'trajectory_details': disorder.trajectory_details,
        'viability_zone': disorder.viability_zone_estimate,
        'population_percentiles': disorder.vs_population_percentile,
    },
    'changepoints': {'cusp_events': cp.cusp_events[:10]},
}

with open(f'{out_dir}/BasedMikeLee_final_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print()
print('=' * 70)
print('X-BOX FINAL ANALYSIS: Mike Lee (@BasedMikeLee)')
print(f'{len(classified):,} tweets | {elapsed:.0f}s ({elapsed/60:.1f}m)')
print('=' * 70)
print()
print('COMPULSION')
print(f'  Score:          {behavioral.compulsion_score}/100')
print(f'  P(compulsive):  {signatures.compulsive_probability:.3f} ({signatures.classification})')
print(f'  Burstiness:     {signatures.burstiness_B:+.3f} (100th percentile)')
print(f'  Entropy:        {signatures.time_entropy:.2f} bits (94th percentile)')
print(f'  Night:          {signatures.night_intensity:.1%} (100th percentile)')
print(f'  Hawkes n*:      {signatures.self_excitation_ratio:.3f} (94th percentile)')
print(f'  Gap shape:      {signatures.gap_distribution_shape}')
print()
print('VIRULENCE (12 heads)')
print(f'  CardiffNLP:     {virulence_cardiffnlp:.1f}/100')
print(f'  SetFit custom:  {setfit_virulence:.1f}/100')
print(f'  ENHANCED:       {enhanced_virulence:.1f}/100')
print()
print('  SetFit rates (5K sample):')
for dim, rate in setfit_rates.items():
    print(f'    {dim:30s} {rate:.1%}')
print()
print('DISORDER')
print(f'  Severity:       {disorder.severity} ({disorder.criteria_met}/8 criteria)')
print(f'  Trajectory:     {disorder.trajectory} ({disorder.trajectory_details.get("early_rate",0):.0f} -> {disorder.trajectory_details.get("late_rate",0):.0f}/wk)')
print(f'  Viability zone: {disorder.viability_zone_estimate}')
print()
print(f'OVERALL RISK:     {overall_risk:.1f}/100')
print()
print(f'Output: {out_dir}/BasedMikeLee_final_report.json')
