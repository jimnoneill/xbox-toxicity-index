"""
Collect House tweet data + cross-domain celebrity validation.

1. House of Representatives: Use congresstweets GitHub archive for House members
2. Cross-domain: Pull celebrity/public figure tweet datasets for accounts with
   documented compulsive use, addiction, or mental health crises
"""
import os, sys, json, time
import pandas as pd
import numpy as np
import requests

sys.path.insert(0, '/home/joneill/x_box/pipeline')
from xbox.compulsion_signatures import CompulsionSignatureAnalyzer

out_dir = '/home/joneill/x_box/pipeline/output/v2_validation'
data_dir = '/home/joneill/x_box/pipeline/data'
os.makedirs(out_dir, exist_ok=True)

# ============================================================
# PART 1: House of Representatives tweets
# ============================================================
print('=' * 70)
print('PART 1: House of Representatives Data Collection')
print('=' * 70)

# The alexlitel/congresstweets repo has daily JSON archives
# Try to get recent data from their GitHub
base_url = 'https://raw.githubusercontent.com/alexlitel/congresstweets/master/data'

# Try a range of recent dates
house_handles = pd.read_parquet(f'{data_dir}/house_handles.parquet')
print(f'House handles: {len(house_handles)}')

# Try to get congresstweets data
all_congress_tweets = []
dates_tried = 0
dates_found = 0

# Try last 90 days of data (March 2026 back to Dec 2025)
for year in [2026, 2025]:
    for month in range(12, 0, -1):
        for day in [1, 8, 15, 22]:  # Sample 4 days per month
            date_str = f'{year}-{month:02d}-{day:02d}'
            url = f'{base_url}/{date_str}.json'
            dates_tried += 1
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    tweets = resp.json()
                    all_congress_tweets.extend(tweets)
                    dates_found += 1
                    if dates_found % 10 == 0:
                        print(f'  Found {dates_found} dates, {len(all_congress_tweets)} tweets so far...')
            except:
                pass
            if dates_found >= 50 or dates_tried >= 100:
                break
        if dates_found >= 50 or dates_tried >= 100:
            break
    if dates_found >= 50 or dates_tried >= 100:
        break

print(f'  Tried {dates_tried} dates, found {dates_found}')
print(f'  Total congress tweets: {len(all_congress_tweets)}')

if all_congress_tweets:
    congress_df = pd.DataFrame(all_congress_tweets)
    print(f'  Columns: {list(congress_df.columns)[:10]}')

    # Try to identify House members
    if 'screen_name' in congress_df.columns:
        house_handles_lower = set(house_handles['twitter_handle'].str.lower())
        congress_df['is_house'] = congress_df['screen_name'].str.lower().isin(house_handles_lower)
        house_tweets = congress_df[congress_df['is_house']]
        print(f'  House tweets: {len(house_tweets)}')
        print(f'  House members found: {house_tweets["screen_name"].nunique()}')
    elif 'user' in congress_df.columns:
        print(f'  User column structure: {type(congress_df["user"].iloc[0]) if len(congress_df) > 0 else "empty"}')
else:
    print('  No congresstweets data available from GitHub.')
    print('  Trying HuggingFace datasets...')

    from datasets import load_dataset
    for ds_name in ['alexlitel/us-congressional-tweets', 'congresstweets/us-congress']:
        try:
            ds = load_dataset(ds_name, trust_remote_code=True)
            key = 'train' if 'train' in ds else list(ds.keys())[0]
            congress_df = ds[key].to_pandas()
            print(f'  {ds_name}: {len(congress_df)} tweets')
            print(f'  Columns: {list(congress_df.columns)[:10]}')
            break
        except Exception as e:
            print(f'  {ds_name}: {e}')

# ============================================================
# PART 2: Cross-domain celebrity validation
# ============================================================
print('\n' + '=' * 70)
print('PART 2: Cross-Domain Celebrity Validation')
print('=' * 70)

# Search for celebrity tweet datasets on HuggingFace
from datasets import load_dataset

celebrity_datasets = {}

# Try various celebrity/public figure datasets
datasets_to_try = [
    ('elonmusk-tweets', 'Elon Musk - documented compulsive poster post-acquisition'),
    ('kanye-tweets', 'Kanye/Ye - documented bipolar episodes visible in posting'),
]

for search_term, description in datasets_to_try:
    print(f'\n  Searching for {search_term}...')
    for prefix in ['', 'amanmadaan/', 'sahilnayak/', 'shaikhafsar/', 'virattt/']:
        try:
            ds = load_dataset(f'{prefix}{search_term}', trust_remote_code=True)
            key = 'train' if 'train' in ds else list(ds.keys())[0]
            df = ds[key].to_pandas()
            if len(df) > 100:
                celebrity_datasets[search_term] = df
                print(f'    Found: {prefix}{search_term} ({len(df)} tweets)')
                print(f'    Columns: {list(df.columns)[:8]}')
                break
        except:
            continue
    else:
        print(f'    Not found on HuggingFace')

# Try direct search for Elon Musk tweets
for ds_name in ['alexmk/elon-musk-tweets', 'mithilparmar/elon-musk-tweets',
                'JoeyBernstein/elon-musk-tweets-dataset',
                'ahmedkhan81/elon-musk-tweets', 'rguo123/trump_all_tweets']:
    try:
        print(f'  Trying {ds_name}...')
        ds = load_dataset(ds_name, trust_remote_code=True)
        key = 'train' if 'train' in ds else list(ds.keys())[0]
        df = ds[key].to_pandas()
        if len(df) > 100:
            short_name = ds_name.split('/')[-1]
            celebrity_datasets[short_name] = df
            print(f'    Found: {len(df)} tweets, cols={list(df.columns)[:8]}')
    except Exception as e:
        pass

# Also try the full Trump dataset we couldn't get before
for ds_name in ['rguo123/trump_all_tweets', 'Hellisotherpeople/trump-tweets']:
    try:
        print(f'  Trying {ds_name}...')
        ds = load_dataset(ds_name, trust_remote_code=True)
        key = 'train' if 'train' in ds else list(ds.keys())[0]
        df = ds[key].to_pandas()
        if len(df) > 33000:  # More than what we already have
            celebrity_datasets['trump_extended'] = df
            print(f'    Found extended Trump: {len(df)} tweets')
            print(f'    Columns: {list(df.columns)[:8]}')
    except:
        pass

print(f'\n  Celebrity datasets collected: {len(celebrity_datasets)}')

# ============================================================
# PART 3: Run compulsion signatures on whatever we collected
# ============================================================
print('\n' + '=' * 70)
print('PART 3: Running Compulsion Signatures on Collected Data')
print('=' * 70)

analyzer = CompulsionSignatureAnalyzer()
crossdomain_results = []

for name, df in celebrity_datasets.items():
    # Try to find timestamp and text columns
    ts_col = None
    for candidate in ['created_at', 'date', 'datetime', 'timestamp', 'Date', 'Datetime']:
        if candidate in df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        print(f'  {name}: no timestamp column found ({list(df.columns)[:5]})')
        continue

    df['created_at'] = pd.to_datetime(df[ts_col], utc=True, errors='coerce')
    df = df.dropna(subset=['created_at'])

    if len(df) < 100:
        print(f'  {name}: too few tweets after parsing ({len(df)})')
        continue

    sig = analyzer.analyze(df, senator_name=name, twitter_handle=name)
    result = {
        'account': name,
        'n_tweets': len(df),
        'date_range': f'{df.created_at.min().date()} to {df.created_at.max().date()}',
        'p_compulsive': sig.compulsive_probability,
        'classification': sig.classification,
        'burstiness_B': round(sig.burstiness_B, 3),
        'time_entropy': round(sig.time_entropy, 2),
        'night_intensity': round(sig.night_intensity, 3),
        'hawkes_n': round(sig.self_excitation_ratio, 3),
        'weekend_ratio': round(sig.weekend_ratio, 3),
    }
    crossdomain_results.append(result)
    print(f'  {name:30s}: P(comp)={sig.compulsive_probability:.3f} '
          f'B={sig.burstiness_B:+.3f} H={sig.time_entropy:.2f} '
          f'N={sig.night_intensity:.1%} n*={sig.self_excitation_ratio:.3f} '
          f'({len(df)} tweets)')

if crossdomain_results:
    cd_df = pd.DataFrame(crossdomain_results)
    cd_df.to_csv(f'{out_dir}/crossdomain_signatures.csv', index=False)
    print(f'\n  Saved {len(cd_df)} cross-domain results')

# ============================================================
# SUMMARY
# ============================================================
print('\n' + '=' * 70)
print('SUMMARY')
print('=' * 70)
print(f'House handles collected: {len(house_handles)}')
print(f'Congress tweets found: {len(all_congress_tweets) if all_congress_tweets else 0}')
print(f'Celebrity datasets: {len(celebrity_datasets)}')
print(f'Cross-domain signatures: {len(crossdomain_results)}')
