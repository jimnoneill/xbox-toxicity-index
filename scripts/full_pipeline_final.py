"""
Full enhanced pipeline — Mike Lee final analysis.
Uses: CardiffNLP (6 heads) + SetFit v2 mpnet (6 heads) + compulsion signatures + disorder baseline + changepoints.
"""
import time, json, logging, os, sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()

sys.path.insert(0, '/mnt/c/x_box/pipeline')

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from xbox.data import load_local_archive
from xbox.behavioral import BehavioralAnalyzer
from xbox.compulsion_signatures import CompulsionSignatureAnalyzer
from xbox.changepoint import BayesianChangepointDetector
from xbox.disorder_baseline import DisorderBaselineAnalyzer
from xbox.fusion import ScoreFusion

t0 = time.time()
out_dir = '/mnt/c/x_box/pipeline/output'
os.makedirs(out_dir, exist_ok=True)

# ── Load data + existing CardiffNLP classifications ───────────
log.info('Loading data...')
df = load_local_archive('/mnt/c/x_box/BasedMikeLee_full_archive.xlsx', senator_name='Mike Lee')
classified = pd.read_parquet(f'{out_dir}/BasedMikeLee_classified_tweets.parquet')
log.info(f'{len(df):,} tweets loaded, {len(classified)} pre-classified')

# ── Behavioral + Compulsion Signatures ────────────────────────
log.info('Behavioral analysis...')
behavioral = BehavioralAnalyzer().analyze(df, senator_name='Mike Lee', twitter_handle='BasedMikeLee')
signatures = CompulsionSignatureAnalyzer().analyze(df, senator_name='Mike Lee', twitter_handle='BasedMikeLee')

# ── SetFit custom classifiers (v1 + v2, mpnet backbone) ──────
log.info('Running SetFit classifiers...')
from setfit import SetFitModel

setfit_models = {
    'ragebait': '/mnt/c/x_box/pipeline/models/setfit_ragebait',
    'epistemic_manipulation': '/mnt/c/x_box/pipeline/models/setfit_epistemic_manipulation',
    'tribal_signal': '/mnt/c/x_box/pipeline/models_v2/setfit_tribal_signal',
    'performative_outrage': '/mnt/c/x_box/pipeline/models_v2/setfit_performative_outrage',
    'engagement_bait': '/mnt/c/x_box/pipeline/models_v2/setfit_engagement_bait',
    'agency_language': '/mnt/c/x_box/pipeline/models_v2/setfit_agency_language',
}

texts = classified['text'].tolist()
for dim, path in setfit_models.items():
    t1 = time.time()
    model = SetFitModel.from_pretrained(path)
    preds = model.predict(texts)
    classified[f'setfit_{dim}'] = preds.tolist() if hasattr(preds, 'tolist') else list(preds)
    rate = float(pd.Series(classified[f'setfit_{dim}']).astype(float).mean())
    log.info(f'  {dim}: {rate:.1%} positive ({time.time()-t1:.0f}s)')

# ── Score fusion ──────────────────────────────────────────────
log.info('Score fusion...')
fusion = ScoreFusion()
classified = fusion.compute_tweet_virulence(classified)

# SetFit virulence
weights = {
    'setfit_ragebait': 0.25, 'setfit_tribal_signal': 0.20,
    'setfit_performative_outrage': 0.20, 'setfit_epistemic_manipulation': 0.15,
    'setfit_engagement_bait': 0.10, 'setfit_agency_language': -0.10,
}
classified['setfit_virulence'] = sum(
    classified[col].astype(float) * w for col, w in weights.items() if col in classified.columns
)
classified['enhanced_virulence'] = (
    classified['composite_virulence'] * 0.4 + classified['setfit_virulence'] * 0.6
).clip(0, 1)

# ── Changepoint detection ────────────────────────────────────
log.info('Changepoint detection...')
cp = BayesianChangepointDetector(hazard_rate=1/60).detect_changepoints(
    classified, resolution='D', min_confidence=0.2
)

# ── Disorder baseline ────────────────────────────────────────
log.info('Disorder baseline analysis...')
# Merge sentiment/offensive into main df for disorder analysis
df['sentiment_negative'] = classified['sentiment_negative'].values
df['offensive_score'] = classified['offensive_score'].values
disorder = DisorderBaselineAnalyzer().analyze(
    df, signatures=signatures.to_dict(), behavioral=behavioral.to_dict(),
    senator_name='Mike Lee', twitter_handle='BasedMikeLee',
)

# ── Build comprehensive report ────────────────────────────────
log.info('Generating report...')

virulence_cardiffnlp = float(classified['composite_virulence'].mean()) * 100
virulence_setfit = float(classified['setfit_virulence'].mean()) * 100
virulence_enhanced = float(classified['enhanced_virulence'].mean()) * 100

report = {
    'meta': {
        'generated_at': pd.Timestamp.utcnow().isoformat() + 'Z',
        'pipeline_version': '0.3.0',
        'total_classification_heads': 12,
        'backbone_setfit': 'sentence-transformers/all-mpnet-base-v2 (v1/v2)',
        'backbone_cardiffnlp': 'cardiffnlp/twitter-roberta-base',
        'note': 'GTE-large backbone queued for GPU retraining',
    },
    'senator': {'name': 'Mike Lee', 'handle': 'BasedMikeLee', 'party': 'Republican', 'state': 'UT'},
    'summary': {
        'tweets_analyzed': len(classified),
        'date_range': f'{df["created_at"].min().date()} to {df["created_at"].max().date()}',
        'compulsion_score': behavioral.compulsion_score,
        'compulsive_probability': signatures.compulsive_probability,
        'compulsion_classification': signatures.classification,
        'virulence_cardiffnlp': round(virulence_cardiffnlp, 1),
        'virulence_setfit': round(virulence_setfit, 1),
        'virulence_enhanced': round(virulence_enhanced, 1),
        'overall_risk': round(behavioral.compulsion_score * 0.4 + virulence_enhanced * 0.6, 1),
        'disorder_severity': disorder.severity,
        'disorder_criteria_met': disorder.criteria_met,
        'trajectory': disorder.trajectory,
        'viability_zone': disorder.viability_zone_estimate,
    },
    'compulsion_signatures': {
        'burstiness_B': signatures.burstiness_B,
        'time_entropy_bits': signatures.time_entropy,
        'weekend_ratio': signatures.weekend_ratio,
        'night_intensity': signatures.night_intensity,
        'hawkes_self_excitation': signatures.self_excitation_ratio,
        'response_latency_min': signatures.response_latency_median_min,
        'gap_distribution': signatures.gap_distribution_shape,
        'z_state_logit': signatures.z_state,
    },
    'compulsion_behavioral': behavioral.compulsion_subscores,
    'virulence_setfit': {
        dim: {'positive_rate': float(classified[f'setfit_{dim}'].astype(float).mean())}
        for dim in setfit_models
    },
    'virulence_cardiffnlp': {
        'sentiment': classified['sentiment_label'].value_counts(normalize=True).to_dict(),
        'emotion': classified['emotion_label'].value_counts(normalize=True).to_dict(),
        'toxicity_mean': float(classified['toxicity_score'].mean()),
        'toxicity_pct': float((classified['toxicity_score'] > 0.5).mean()) * 100,
    },
    'disorder_assessment': {
        'severity': disorder.severity,
        'criteria_met': disorder.criteria_met,
        'criteria': {k: {'met': v['met'], 'value': v['value']}
                    for k, v in disorder.criteria_details.items()},
        'trajectory': disorder.trajectory,
        'trajectory_details': disorder.trajectory_details,
        'viability_zone': disorder.viability_zone_estimate,
        'vs_baseline': disorder.vs_strategic_baseline,
        'population_percentiles': disorder.vs_population_percentile,
    },
    'changepoints': {
        'cusp_events': cp.cusp_events[:10],
        'early_warnings': [str(e) for e in cp.early_warnings],
    },
    'top_enhanced_rage_tweets': [],
}

# Top rage by enhanced virulence
top = classified.nlargest(10, 'enhanced_virulence')
for _, row in top.iterrows():
    report['top_enhanced_rage_tweets'].append({
        'text': str(row.get('text', ''))[:200],
        'enhanced_virulence': round(float(row.get('enhanced_virulence', 0)), 4),
        'setfit_ragebait': int(row.get('setfit_ragebait', 0)),
        'setfit_tribal': int(row.get('setfit_tribal_signal', 0)),
        'setfit_performative': int(row.get('setfit_performative_outrage', 0)),
        'sentiment': str(row.get('sentiment_label', '')),
        'created_at': str(row.get('created_at', '')),
    })

# Save
with open(f'{out_dir}/BasedMikeLee_final_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

for col in classified.columns:
    if classified[col].dtype == object:
        classified[col] = classified[col].astype(str)
classified.to_parquet(f'{out_dir}/BasedMikeLee_final_classified.parquet', index=False)

elapsed = time.time() - t0

# ── Print summary ─────────────────────────────────────────────
print()
print('=' * 70)
print('X-BOX ENHANCED PIPELINE — FINAL RESULTS')
print(f'Mike Lee (@BasedMikeLee) — {len(classified):,} tweets')
print('=' * 70)
print()
print('COMPULSION')
print(f'  Behavioral score:      {behavioral.compulsion_score}/100')
print(f'  Compulsive probability:{signatures.compulsive_probability:.3f} ({signatures.classification})')
print(f'  Burstiness B:          {signatures.burstiness_B:+.3f}')
print(f'  Time entropy:          {signatures.time_entropy:.2f} bits')
print(f'  Night intensity:       {signatures.night_intensity:.1%}')
print(f'  Hawkes n*:             {signatures.self_excitation_ratio:.3f}')
print(f'  Response latency:      {signatures.response_latency_median_min:.1f} min')
print()
print('VIRULENCE (12 heads)')
print(f'  CardiffNLP (6 heads):  {virulence_cardiffnlp:.1f}/100')
print(f'  SetFit custom (6):     {virulence_setfit:.1f}/100')
print(f'  ENHANCED (combined):   {virulence_enhanced:.1f}/100')
print()
print('  SetFit breakdown:')
for dim in setfit_models:
    rate = float(classified[f'setfit_{dim}'].astype(float).mean())
    print(f'    {dim:30s} {rate:.1%}')
print()
print('DISORDER ASSESSMENT')
print(f'  Severity:              {disorder.severity} ({disorder.criteria_met}/8 criteria)')
print(f'  Trajectory:            {disorder.trajectory}')
print(f'  Viability zone:        {disorder.viability_zone_estimate}')
print(f'  Early: {disorder.trajectory_details.get("early_rate", 0):.0f}/wk → Late: {disorder.trajectory_details.get("late_rate", 0):.0f}/wk')
print()
print('POPULATION PERCENTILES')
for k, p in disorder.vs_population_percentile.items():
    print(f'  {k:25s} z={p["z_score"]:+5.1f}  P={p["percentile"]:5.1f}%')
print()
print(f'OVERALL RISK: {report["summary"]["overall_risk"]}/100')
print(f'Time: {elapsed:.0f}s ({elapsed/60:.1f}m)')
print()
print(f'Output: {out_dir}/BasedMikeLee_final_report.json')
print(f'        {out_dir}/BasedMikeLee_final_classified.parquet')
