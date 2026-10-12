"""
Collect House + celebrity data via direct downloads (no HF datasets API).
"""
import os, sys, json, time
import pandas as pd
import numpy as np
import requests
from io import StringIO

sys.path.insert(0, '/home/joneill/x_box/pipeline')
from xbox.compulsion_signatures import CompulsionSignatureAnalyzer

data_dir = '/home/joneill/x_box/pipeline/data'
out_dir = '/home/joneill/x_box/pipeline/output/v2_validation'
os.makedirs(f'{data_dir}/crossdomain', exist_ok=True)

analyzer = CompulsionSignatureAnalyzer()

# ============================================================
# PART 1: congresstweets — daily JSON from GitHub
# ============================================================
print('=' * 70)
print('PART 1: congresstweets daily archives')
print('=' * 70)

# alexlitel/congresstweets stores daily files at data/YYYY-MM-DD.json
# Try both the main branch and the raw data
all_tweets = []
found = 0

for year in [2024, 2023]:
    for month in range(12, 0, -1):
        for day in range(1, 29, 7):  # every 7th day
            date_str = f'{year}-{month:02d}-{day:02d}'
            url = f'https://raw.githubusercontent.com/alexlitel/congresstweets/master/data/{date_str}.json'
            try:
                resp = requests.get(url, timeout=8)
                if resp.status_code == 200:
                    tweets = resp.json()
                    for t in tweets:
                        t['_date'] = date_str
                    all_tweets.extend(tweets)
                    found += 1
                    if found % 5 == 0:
                        print(f'  {found} dates found, {len(all_tweets)} tweets')
            except:
                pass
        if found >= 30:
            break
    if found >= 30:
        break

print(f'congresstweets: {found} dates, {len(all_tweets)} total tweets')

if all_tweets:
    ct = pd.DataFrame(all_tweets)
    print(f'Columns: {list(ct.columns)}')

    # Parse timestamps
    for ts_col in ['time', 'date', 'created_at', '_date']:
        if ts_col in ct.columns:
            ct['created_at'] = pd.to_datetime(ct[ts_col], utc=True, errors='coerce')
            break

    ct = ct.dropna(subset=['created_at'])

    # Identify screen names
    sn_col = None
    for c in ['screen_name', 'user_screen_name', 'user']:
        if c in ct.columns:
            sn_col = c
            break

    if sn_col:
        # Load house handles
        house = pd.read_parquet(f'{data_dir}/house_handles.parquet')
        house_set = set(house['twitter_handle'].str.lower())

        ct['screen_name_lower'] = ct[sn_col].str.lower()
        house_tweets = ct[ct['screen_name_lower'].isin(house_set)]
        senate_tweets = ct[~ct['screen_name_lower'].isin(house_set)]

        print(f'House tweets: {len(house_tweets)} from {house_tweets[sn_col].nunique()} members')
        print(f'Senate tweets: {len(senate_tweets)}')

        if len(house_tweets) > 0:
            # Run signatures on top House tweeters
            print('\nHouse member signatures (top 30 by volume):')
            house_results = []
            top_house = house_tweets.groupby(sn_col).size().nlargest(30)

            for username, count in top_house.items():
                sub = house_tweets[house_tweets[sn_col] == username]
                if len(sub) < 30:
                    continue
                sig = analyzer.analyze(sub, senator_name=username)
                house_results.append({
                    'account': username,
                    'chamber': 'house',
                    'n_tweets': len(sub),
                    'p_compulsive': sig.compulsive_probability,
                    'burstiness_B': round(sig.burstiness_B, 3),
                    'time_entropy': round(sig.time_entropy, 2),
                    'night_intensity': round(sig.night_intensity, 3),
                    'hawkes_n': round(sig.self_excitation_ratio, 3),
                    'weekend_ratio': round(sig.weekend_ratio, 3),
                })
                print(f'  {username:25s} P={sig.compulsive_probability:.3f} '
                      f'B={sig.burstiness_B:+.3f} N={sig.night_intensity:.1%} '
                      f'({len(sub)} tweets)')

            if house_results:
                pd.DataFrame(house_results).to_csv(f'{out_dir}/house_signatures.csv', index=False)
                print(f'\nSaved {len(house_results)} House signatures')

    ct.to_parquet(f'{data_dir}/congresstweets_sample.parquet', index=False)

# ============================================================
# PART 2: Celebrity tweets via Kaggle/GitHub CSV downloads
# ============================================================
print('\n' + '=' * 70)
print('PART 2: Cross-domain celebrity data')
print('=' * 70)

celebrity_sources = {
    'elon_musk': [
        'https://raw.githubusercontent.com/kyleskom/Elon-Musk-Tweets/master/data.csv',
    ],
}

crossdomain_results = []

for name, urls in celebrity_sources.items():
    for url in urls:
        try:
            print(f'  Trying {name}: {url[:60]}...')
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                df = pd.read_csv(StringIO(resp.text))
                print(f'    Got {len(df)} rows, cols={list(df.columns)[:6]}')

                # Find timestamp column
                ts_col = None
                for c in ['created_at', 'date', 'datetime', 'Datetime', 'Date', 'Time']:
                    if c in df.columns:
                        ts_col = c
                        break

                if ts_col:
                    df['created_at'] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
                    df = df.dropna(subset=['created_at'])
                    print(f'    Parsed {len(df)} tweets with timestamps')

                    if len(df) >= 100:
                        sig = analyzer.analyze(df, senator_name=name)
                        result = {
                            'account': name,
                            'domain': 'celebrity',
                            'n_tweets': len(df),
                            'date_range': f'{df.created_at.min().date()} to {df.created_at.max().date()}',
                            'p_compulsive': sig.compulsive_probability,
                            'burstiness_B': round(sig.burstiness_B, 3),
                            'time_entropy': round(sig.time_entropy, 2),
                            'night_intensity': round(sig.night_intensity, 3),
                            'hawkes_n': round(sig.self_excitation_ratio, 3),
                            'weekend_ratio': round(sig.weekend_ratio, 3),
                        }
                        crossdomain_results.append(result)
                        print(f'    RESULT: P(comp)={sig.compulsive_probability:.3f} '
                              f'B={sig.burstiness_B:+.3f} N={sig.night_intensity:.1%}')
                        df.to_parquet(f'{data_dir}/crossdomain/{name}.parquet', index=False)
                        break
        except Exception as e:
            print(f'    Error: {e}')

# Also add Trump as cross-domain reference
trump = pd.read_parquet(f'{data_dir}/trump_tweets.parquet')
trump['created_at'] = pd.to_datetime(trump['created_at'], utc=True, errors='coerce')
sig = analyzer.analyze(trump.dropna(subset=['created_at']), senator_name='Trump_Full')
crossdomain_results.append({
    'account': 'Trump_Full',
    'domain': 'political_reference',
    'n_tweets': len(trump),
    'date_range': f'{trump.created_at.min().date()} to {trump.created_at.max().date()}',
    'p_compulsive': sig.compulsive_probability,
    'burstiness_B': round(sig.burstiness_B, 3),
    'time_entropy': round(sig.time_entropy, 2),
    'night_intensity': round(sig.night_intensity, 3),
    'hawkes_n': round(sig.self_excitation_ratio, 3),
    'weekend_ratio': round(sig.weekend_ratio, 3),
})

# Mike Lee as reference
lee = pd.read_excel('/home/joneill/x_box/BasedMikeLee_full_archive.xlsx')
lee['created_at'] = pd.to_datetime(lee['created_at'], utc=True, errors='coerce')
sig = analyzer.analyze(lee.dropna(subset=['created_at']), senator_name='BasedMikeLee')
crossdomain_results.append({
    'account': 'BasedMikeLee',
    'domain': 'political_reference',
    'n_tweets': len(lee),
    'date_range': f'{lee.created_at.min().date()} to {lee.created_at.max().date()}',
    'p_compulsive': sig.compulsive_probability,
    'burstiness_B': round(sig.burstiness_B, 3),
    'time_entropy': round(sig.time_entropy, 2),
    'night_intensity': round(sig.night_intensity, 3),
    'hawkes_n': round(sig.self_excitation_ratio, 3),
    'weekend_ratio': round(sig.weekend_ratio, 3),
})

if crossdomain_results:
    cd_df = pd.DataFrame(crossdomain_results)
    cd_df.to_csv(f'{out_dir}/crossdomain_signatures.csv', index=False)
    print(f'\n  Cross-domain results: {len(cd_df)}')
    for _, r in cd_df.iterrows():
        print(f'    {r.account:25s} P={r.p_compulsive:.3f} ({r.domain})')

print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'congresstweets collected: {len(all_tweets)}')
print(f'Cross-domain accounts: {len(crossdomain_results)}')
