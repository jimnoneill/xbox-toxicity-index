"""Pull full 2017-2023 congresstweets archive."""
import requests, json, os, time
import pandas as pd

data_dir = '/home/joneill/x_box/pipeline/data'
all_tweets = []
dates_found = 0

print('Collecting congresstweets 2017-2023...')

for year in range(2017, 2024):
    for month in range(1, 13):
        for day in [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]:
            date_str = f'{year}-{month:02d}-{day:02d}'
            url = f'https://raw.githubusercontent.com/alexlitel/congresstweets/master/data/{date_str}.json'
            try:
                resp = requests.get(url, timeout=8)
                if resp.status_code == 200:
                    tweets = resp.json()
                    for t in tweets:
                        t['_date'] = date_str
                    all_tweets.extend(tweets)
                    dates_found += 1
                    if dates_found % 20 == 0:
                        print(f'  {dates_found} dates, {len(all_tweets):,} tweets ({date_str})')
            except:
                pass
            time.sleep(0.05)

print(f'\nTotal: {dates_found} dates, {len(all_tweets):,} tweets')

if all_tweets:
    df = pd.DataFrame(all_tweets)
    print(f'Columns: {list(df.columns)}')

    for c in ['time', 'date', 'created_at']:
        if c in df.columns:
            df['created_at'] = pd.to_datetime(df[c], utc=True, errors='coerce')
            break

    df = df.dropna(subset=['created_at'])
    df['year'] = df['created_at'].dt.year

    print('\nBy year:')
    for y in sorted(df['year'].unique()):
        sub = df[df['year'] == y]
        n = len(sub)
        members = sub['screen_name'].nunique() if 'screen_name' in sub.columns else 0
        print(f'  {y}: {n:>7,} tweets from {members} members')

    if 'screen_name' in df.columns:
        print(f'\nTotal unique members: {df["screen_name"].nunique()}')

    # Save
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
    df.to_parquet(f'{data_dir}/congresstweets_2017_2023.parquet', index=False)
    sz = os.path.getsize(f'{data_dir}/congresstweets_2017_2023.parquet') / 1e6
    print(f'Saved: congresstweets_2017_2023.parquet ({sz:.1f} MB)')
