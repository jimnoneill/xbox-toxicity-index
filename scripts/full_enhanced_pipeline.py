"""Run full enhanced pipeline on Mike Lee — all 12 classification heads + behavioral + changepoint."""
import time, json, logging, os, sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()

sys.path.insert(0, '/mnt/c/x_box/pipeline')

import pandas as pd
import numpy as np
from xbox.data import load_local_archive
from xbox.classifiers import MultiHeadClassifier
from xbox.behavioral import BehavioralAnalyzer
from xbox.compulsion_signatures import CompulsionSignatureAnalyzer
from xbox.changepoint import BayesianChangepointDetector
from xbox.fusion import ScoreFusion
from xbox.report import generate_json_report, generate_markdown_report

t0 = time.time()
out_dir = '/mnt/c/x_box/pipeline/output'
os.makedirs(out_dir, exist_ok=True)

# ── Step 1: Load data ──────────────────────────────
log.info('STEP 1: Loading Mike Lee archive...')
df = load_local_archive('/mnt/c/x_box/BasedMikeLee_full_archive.xlsx', senator_name='Mike Lee')
log.info(f'Loaded {len(df):,} tweets')

# ── Step 2: Behavioral analysis ────────────────────
log.info('STEP 2: Behavioral analysis...')
behavioral_analyzer = BehavioralAnalyzer()
behavioral = behavioral_analyzer.analyze(df, senator_name='Mike Lee', twitter_handle='BasedMikeLee')
log.info(f'Compulsion score: {behavioral.compulsion_score}/100')

# ── Step 3: Compulsion signatures ──────────────────
log.info('STEP 3: Compulsion signatures (burstiness, entropy, Hawkes)...')
sig_analyzer = CompulsionSignatureAnalyzer()
signatures = sig_analyzer.analyze(df, senator_name='Mike Lee', twitter_handle='BasedMikeLee')
log.info(f'Compulsive probability: {signatures.compulsive_probability:.3f} ({signatures.classification})')

# ── Step 4: Load existing CardiffNLP classifications ──
log.info('STEP 4: Loading existing CardiffNLP classifications...')
classified_path = f'{out_dir}/BasedMikeLee_classified_tweets.parquet'
if os.path.exists(classified_path):
    classified = pd.read_parquet(classified_path)
    log.info(f'Loaded {len(classified)} pre-classified tweets')
else:
    log.info('No existing classifications found, running fresh...')
    classifier = MultiHeadClassifier()
    classified = classifier.classify_tweets(df, batch_size=32)

# ── Step 5: Custom SetFit classifiers ──────────────
log.info('STEP 5: Running custom SetFit classifiers (6 heads)...')
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

for dim, model_path in setfit_models.items():
    t1 = time.time()
    log.info(f'  Loading {dim}...')
    model = SetFitModel.from_pretrained(model_path)

    # Predict in batches
    batch_size = 256
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        preds = model.predict(batch)
        all_preds.extend(preds.tolist() if hasattr(preds, 'tolist') else list(preds))

    classified[f'setfit_{dim}'] = all_preds
    elapsed = time.time() - t1
    pos_rate = sum(1 for p in all_preds if p == 1) / len(all_preds)
    log.info(f'  {dim}: {pos_rate:.1%} positive, {elapsed:.0f}s')

# ── Step 6: Score fusion (enhanced) ────────────────
log.info('STEP 6: Enhanced score fusion...')
fusion = ScoreFusion()
classified = fusion.compute_tweet_virulence(classified)

# Compute enhanced virulence using SetFit scores
setfit_cols = [c for c in classified.columns if c.startswith('setfit_')]
classified['setfit_virulence'] = 0.0
weights = {
    'setfit_ragebait': 0.25,
    'setfit_tribal_signal': 0.20,
    'setfit_performative_outrage': 0.20,
    'setfit_epistemic_manipulation': 0.15,
    'setfit_engagement_bait': 0.10,
    'setfit_agency_language': -0.10,  # negative: agentic language REDUCES virulence
}
for col, w in weights.items():
    if col in classified.columns:
        classified['setfit_virulence'] += classified[col].astype(float) * w

# Combined virulence: CardiffNLP + SetFit
classified['enhanced_virulence'] = (
    classified['composite_virulence'] * 0.4 +
    classified['setfit_virulence'] * 0.6
).clip(0, 1)

profile = fusion.aggregate_senator_profile(
    classified, behavioral,
    senator_name='Mike Lee',
    twitter_handle='BasedMikeLee',
    party='Republican',
    state='UT',
)

# Override virulence with enhanced score
enhanced_virulence_mean = float(classified['enhanced_virulence'].mean()) * 100
profile.virulence_score = round(enhanced_virulence_mean, 1)
profile.overall_risk_score = round(
    behavioral.compulsion_score * 0.4 + enhanced_virulence_mean * 0.6, 1
)

# ── Step 7: Changepoint detection ──────────────────
log.info('STEP 7: Changepoint detection...')
detector = BayesianChangepointDetector(hazard_rate=1/60)
cp_profile = detector.detect_changepoints(classified, resolution='D', min_confidence=0.2)

# ── Step 8: Save everything ────────────────────────
log.info('STEP 8: Saving results...')

# Enhanced report
report = {
    'meta': {
        'generated_at': pd.Timestamp.utcnow().isoformat() + 'Z',
        'pipeline_version': '0.2.0-enhanced',
        'classification_heads': {
            'cardiffnlp': ['sentiment', 'emotion', 'offensive', 'irony', 'hate', 'toxicity'],
            'setfit_custom': list(setfit_models.keys()),
        },
    },
    'senator': {
        'name': 'Mike Lee',
        'handle': 'BasedMikeLee',
        'party': 'Republican',
        'state': 'UT',
    },
    'summary': {
        'tweets_analyzed': len(classified),
        'date_range': f'{classified["created_at"].min()} to {classified["created_at"].max()}',
        'compulsion_score': behavioral.compulsion_score,
        'compulsive_probability': signatures.compulsive_probability,
        'compulsion_classification': signatures.classification,
        'virulence_score_cardiffnlp': round(float(classified['composite_virulence'].mean()) * 100, 1),
        'virulence_score_setfit': round(float(classified['setfit_virulence'].mean()) * 100, 1),
        'virulence_score_enhanced': enhanced_virulence_mean,
        'overall_risk_score': profile.overall_risk_score,
    },
    'compulsion_signatures': signatures.to_dict(),
    'compulsion_behavioral': {
        'score': behavioral.compulsion_score,
        'subscores': behavioral.compulsion_subscores,
        'tweets_per_day': behavioral.tweets_per_day,
        'median_gap_minutes': behavioral.median_gap_minutes,
        'night_share': behavioral.night_share,
        'sessions_per_day': behavioral.sessions_per_day,
    },
    'virulence_setfit': {
        dim: {
            'positive_rate': float(classified[f'setfit_{dim}'].astype(float).mean()),
            'count_positive': int(classified[f'setfit_{dim}'].astype(float).sum()),
        }
        for dim in setfit_models.keys()
    },
    'virulence_cardiffnlp': {
        'sentiment': profile.sentiment_distribution,
        'emotion': profile.emotion_distribution,
        'toxicity': profile.toxicity_stats,
    },
    'changepoints': cp_profile.to_dict(),
    'top_rage_tweets': profile.top_rage_tweets[:10],
}

with open(f'{out_dir}/BasedMikeLee_enhanced_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

# Save classified tweets
for col in classified.columns:
    if classified[col].dtype == object:
        classified[col] = classified[col].astype(str)
classified.to_parquet(f'{out_dir}/BasedMikeLee_enhanced_classified.parquet', index=False)

t_total = time.time() - t0

print()
print('=' * 70)
print('ENHANCED PIPELINE COMPLETE')
print('=' * 70)
print(f'Time: {t_total:.0f}s ({t_total/60:.1f}m)')
print(f'Tweets: {len(classified):,}')
print()
print(f'COMPULSION:')
print(f'  Behavioral score:       {behavioral.compulsion_score}/100')
print(f'  Compulsive probability: {signatures.compulsive_probability:.3f} ({signatures.classification})')
print(f'  Burstiness B:           {signatures.burstiness_B:+.3f}')
print(f'  Time entropy:           {signatures.time_entropy:.2f} bits')
print(f'  Hawkes self-excitation: {signatures.self_excitation_ratio:.3f}')
print()
print(f'VIRULENCE (12 heads):')
print(f'  CardiffNLP composite:   {float(classified["composite_virulence"].mean()) * 100:.1f}/100')
print(f'  SetFit custom:          {float(classified["setfit_virulence"].mean()) * 100:.1f}/100')
print(f'  ENHANCED (combined):    {enhanced_virulence_mean:.1f}/100')
print()
print(f'  SetFit breakdown:')
for dim in setfit_models.keys():
    rate = float(classified[f'setfit_{dim}'].astype(float).mean())
    print(f'    {dim:30s} {rate:.1%} positive')
print()
print(f'OVERALL RISK: {profile.overall_risk_score}/100')
print()
print(f'CHANGEPOINTS: {len(cp_profile.cusp_events)} cusp events')
for evt in cp_profile.cusp_events[:5]:
    print(f'  {evt["date"]} | {evt["direction"]:8s} | z={evt["z_score"]:.1f}')
print()
print(f'Saved: {out_dir}/BasedMikeLee_enhanced_report.json')
print(f'       {out_dir}/BasedMikeLee_enhanced_classified.parquet')
