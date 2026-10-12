"""
Timezone correction v2: Use actual metadata instead of estimating.

Congresstweets: `time` column has Eastern time with DST offsets.
We strip the offset and use the local time directly (it's already local).
Mike Lee X archive: UTC -> America/Denver with proper DST.
"""
import os, sys
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

sys.path.insert(0, '/home/joneill/x_box/pipeline')
from xbox.compulsion_signatures import CompulsionSignatureAnalyzer

out_dir = '/home/joneill/x_box/pipeline/output'
ti_dir = f'{out_dir}/toxicity_index_v2'

print('='*70)
print('Loading congresstweets with native timezone offsets...')
ct = pd.read_parquet('/home/joneill/x_box/pipeline/data/congresstweets_2017_2023.parquet')
print(f'Tweets: {len(ct):,}')

# `time` is like "2017-06-22T15:08:29-04:00" - the datetime part IS local Eastern time.
# Strip the offset and parse just the local component.
ct['local_time_str'] = ct['time'].str[:19]
ct['local_time'] = pd.to_datetime(ct['local_time_str'])
ct['local_hour'] = ct['local_time'].dt.hour
ct['tz_offset'] = ct['time'].str[-6:]

print(f'Offset distribution:')
print(ct['tz_offset'].value_counts())
print(f'\nSample local times:')
for _, row in ct.head(5).iterrows():
    print(f'  {row["screen_name"]:20s} raw: {row["time"]}  local: {row["local_time"]}')

ct['is_night'] = ct['local_hour'].isin([0,1,2,3,4,5])
print(f'\nOverall night posting rate (00-05 Eastern): {ct["is_night"].mean():.1%}')

member_night = ct.groupby('screen_name').agg(
    n_tweets=('text', 'count'),
    night_rate=('is_night', 'mean'),
    mean_hour=('local_hour', 'mean'),
).query('n_tweets >= 50').sort_values('night_rate', ascending=False)

print(f'\nTop 10 night posters (Eastern time):')
for _, r in member_night.head(10).iterrows():
    print(f'  {r.name:25s} night={r.night_rate:.1%} mean_hour={r.mean_hour:.1f} ({int(r.n_tweets):,} tweets)')

# ============================================================
# Mike Lee: UTC -> Mountain time (DST-aware)
# ============================================================
print('\n' + '='*70)
print('Mike Lee: UTC -> Mountain time (America/Denver, DST-aware)')

ml = pd.read_parquet(f'{out_dir}/BasedMikeLee_classified_tweets.parquet')
print(f'Mike Lee tweets: {len(ml):,}')

mountain = ZoneInfo('America/Denver')
ml['created_at_utc'] = pd.to_datetime(ml['created_at'], utc=True)
# Convert each timestamp individually for proper DST
ml['local_time'] = ml['created_at_utc'].apply(lambda x: x.to_pydatetime().astimezone(mountain))
ml['local_time'] = pd.to_datetime(ml['local_time'].apply(lambda x: x.replace(tzinfo=None)))
ml['local_hour'] = ml['local_time'].dt.hour

ml_night = ml['local_hour'].isin([0,1,2,3,4,5]).mean()
print(f'Night posting rate (00-05 Mountain): {ml_night:.1%}')
print(f'Mean posting hour (Mountain): {ml["local_hour"].mean():.1f}')

print('\nPosting hour distribution (Mountain):')
for h in range(24):
    pct = (ml['local_hour'] == h).mean()
    bar = '#' * int(pct * 200)
    print(f'  {h:02d}:00  {pct:5.1%} {bar}')

# ============================================================
# Recompute compulsion signatures
# ============================================================
print('\n' + '='*70)
print('Recomputing compulsion signatures with corrected local times...')

analyzer = CompulsionSignatureAnalyzer()

results = []
for username in ct['screen_name'].unique():
    sub = ct[ct['screen_name'] == username].copy()
    if len(sub) < 100:
        continue
    sub['created_at'] = sub['local_time']
    try:
        sig = analyzer.analyze(sub, senator_name=username)
        results.append({
            'account': username,
            'n_tweets': len(sub),
            'p_compulsive': sig.compulsive_probability,
            'night_intensity': sig.night_intensity,
            'time_entropy': sig.time_entropy,
            'burstiness_B': sig.burstiness_B,
            'hawkes_n': sig.self_excitation_ratio,
        })
    except Exception as e:
        pass

results_df = pd.DataFrame(results).sort_values('p_compulsive', ascending=False)
results_df.to_csv(f'{ti_dir}/compulsion_signatures_tz_corrected_v2.csv', index=False)
print(f'Computed signatures for {len(results_df)} members')

print(f'\nTop 10 compulsive (tz-corrected):')
for _, r in results_df.head(10).iterrows():
    print(f'  {r.account:25s} p={r.p_compulsive:.3f} night={r.night_intensity:.3f} burst={r.burstiness_B:.3f} ({int(r.n_tweets):,})')

# Mike Lee
print(f'\nMike Lee compulsion (Mountain time):')
ml_for_sig = ml.copy()
ml_for_sig['created_at'] = ml_for_sig['local_time']
try:
    ml_sig = analyzer.analyze(ml_for_sig, senator_name='Mike Lee')
    print(f'  p_compulsive: {ml_sig.compulsive_probability:.3f}')
    print(f'  night_intensity: {ml_sig.night_intensity:.3f}')
    print(f'  burstiness: {ml_sig.burstiness_B:.3f}')
    print(f'  hawkes_n: {ml_sig.self_excitation_ratio:.3f}')
    print(f'  time_entropy: {ml_sig.time_entropy:.3f}')
except Exception as e:
    print(f'  Error: {e}')

# Compare
print(f'\nOld tz-corrected results:')
try:
    old = pd.read_csv(f'{out_dir}/v2_validation/senator_tz_corrected.csv')
    ml_old = old[old['account'].str.contains('BasedMikeLee|SenMikeLee|Mike', case=False, na=False)]
    if len(ml_old) > 0:
        print(ml_old[['account','p_compulsive','night_intensity','burstiness_B','hawkes_n']].to_string(index=False))
except Exception as e:
    print(f'  {e}')

print('\nDONE')
