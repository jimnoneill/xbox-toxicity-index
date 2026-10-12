"""
ti_account_member_map.py  (spec A)

Builds output/v7/ti/account_member_map.csv: one row per congresstweets screen_name, mapped to a
member (bioguide) with party, chamber, state, icpsr, and account type (office / campaign).

Sources (downloaded to data/ext_dl_ti/, untrusted data, read with json only):
  - alexlitel/congresstweets-automator data/historical-users.json   (bioguide ids, office/campaign accounts)
  - unitedstates/congress-legislators legislators-current/historical.json (terms, party, icpsr)
  - data/voteview_all_members.csv (Nokken-Poole dim1 by congress, keyed by icpsr)
  - data/congresstweets_2017_2023.parquet (screen_name, user_id, counts)

Matching order: (1) user_id equals an account id in the metadata, (2) screen_name equals a metadata
screen_name or prev_names entry, case-insensitive. Entity types other than member (committee, caucus,
party) are kept in the file with bioguide empty and entity_type set, then reported as unmatched to members.

Party and chamber come from the legislator terms overlapping the study window (2017-06-01 to 2023-07-31),
choosing the term with the longest overlap; party switchers are flagged. Nokken-Poole dim1 is the mean over
congresses 115-118 in which the member has a VoteView row.
Deterministic, no randomness.
"""
import json, sys, os
from collections import Counter
import numpy as np
import pandas as pd

ROOT = os.path.expanduser('~/x_box/pipeline')
DL = f'{ROOT}/data/ext_dl_ti'
OUT = f'{ROOT}/output/v7/ti'
os.makedirs(OUT, exist_ok=True)
W0, W1 = pd.Timestamp('2017-06-01'), pd.Timestamp('2023-07-31')

ct = pd.read_parquet(f'{ROOT}/data/congresstweets_2017_2023.parquet', columns=['screen_name', 'user_id', 'created_at'])
acc = (ct.groupby('screen_name')
         .agg(n_tweets=('user_id', 'size'), n_user_ids=('user_id', 'nunique'),
              user_id=('user_id', lambda s: s.value_counts().index[0]),
              first=('created_at', 'min'), last=('created_at', 'max')).reset_index())

users = json.load(open(f'{DL}/historical-users.json'))
by_id, by_sn = {}, {}
for e in users:
    for a in e['accounts']:
        by_id[str(a['id'])] = (e, a)
        by_sn[a['screen_name'].lower()] = (e, a)
        for pn in a.get('prev_names', []) or []:
            if isinstance(pn, dict):
                pn = pn.get('screen_name') or pn.get('name') or ''
            if pn:
                by_sn.setdefault(str(pn).lower(), (e, a))

# fallback: current-member handle tables (account type unknown, so labelled 'unknown')
fb_id, fb_sn = {}, {}
for e in json.load(open(f'{DL}/legislators-social-media.json')):
    sc = e['social']
    if sc.get('twitter_id'):
        fb_id[str(sc['twitter_id'])] = e['id']['bioguide']
    if sc.get('twitter'):
        fb_sn[sc['twitter'].lower()] = e['id']['bioguide']
for fn in ['senator_handles', 'house_handles']:
    h = pd.read_parquet(f'{ROOT}/data/{fn}.parquet')
    for x in h.itertuples():
        fb_sn.setdefault(str(x.twitter_handle).lower(), x.bioguide_id)
        tid = str(getattr(x, 'twitter_id', '') or '').strip()
        if tid.isdigit():
            fb_id.setdefault(tid, x.bioguide_id)

rows = []
for r in acc.itertuples():
    hit, how = by_id.get(str(r.user_id)), 'user_id'
    if hit is None:
        hit, how = by_sn.get(r.screen_name.lower()), 'screen_name'
    if hit is None:
        bg = fb_id.get(str(r.user_id)) or fb_sn.get(r.screen_name.lower())
        if bg:
            rows.append(dict(screen_name=r.screen_name, user_id=r.user_id, bioguide=bg, name='', meta_party='',
                             meta_chamber='', meta_state='', account_type='unknown', entity_type='member',
                             match_method='fallback_handle_table'))
        else:
            rows.append(dict(screen_name=r.screen_name, user_id=r.user_id, entity_type='unmatched', match_method='none'))
        continue
    e, a = hit
    idd = e.get('id') if isinstance(e.get('id'), dict) else {}
    rows.append(dict(screen_name=r.screen_name, user_id=r.user_id, bioguide=idd.get('bioguide', ''),
                     name=e['name'], meta_party=e.get('party', ''), meta_chamber=e.get('chamber', ''),
                     meta_state=e.get('state', ''), account_type=a['account_type'],
                     entity_type=e['type'], match_method=how))
m = pd.DataFrame(rows)
m = m.merge(acc[['screen_name', 'n_tweets', 'n_user_ids']], on='screen_name')

# legislators terms
leg = json.load(open(f'{DL}/legislators-current.json')) + json.load(open(f'{DL}/legislators-historical.json'))
info = {}
for p in leg:
    bg = p['id'].get('bioguide')
    best, bo, parties = None, -1, set()
    for t in p['terms']:
        s, e = pd.Timestamp(t['start']), pd.Timestamp(t['end'])
        ov = (min(e, W1) - max(s, W0)).days
        if ov > 0:
            parties.add(t.get('party'))
            if ov > bo:
                best, bo = t, ov
    last = p['terms'][-1]
    t = best or last
    info[bg] = dict(icpsr=p['id'].get('icpsr'), leg_party=t.get('party'),
                    leg_chamber={'sen': 'senate', 'rep': 'house'}[t['type']], leg_state=t['state'],
                    party_switch=int(len(parties) > 1), in_window=int(best is not None),
                    full_name=p['name'].get('official_full') or f"{p['name']['first']} {p['name']['last']}")
inf = pd.DataFrame.from_dict(info, orient='index').rename_axis('bioguide').reset_index()
m = m.merge(inf, on='bioguide', how='left')

# VoteView Nokken-Poole 115-118
vv = pd.read_csv(f'{ROOT}/data/voteview_all_members.csv')
vv = vv[(vv.congress >= 115) & (vv.congress <= 118) & vv.nokken_poole_dim1.notna()]
vvb = vv.groupby('bioguide_id').agg(vv_icpsr=('icpsr', 'first'), np_dim1=('nokken_poole_dim1', 'mean'),
                                    nom_dim1=('nominate_dim1', 'mean'), vv_congresses=('congress', 'nunique')).reset_index()
vvb = vvb.rename(columns={'bioguide_id': 'bioguide'})
m = m.merge(vvb, on='bioguide', how='left')
m['icpsr'] = m['vv_icpsr'].fillna(m['icpsr'])
m['icpsr'] = m['icpsr'].astype('Int64')
pm = {'Democrat': 'D', 'Republican': 'R', 'Independent': 'I'}
m['party'] = m['leg_party'].map(pm).fillna(m['meta_party'].replace({'N/A': ''}))
m['chamber'] = m['leg_chamber'].fillna(m['meta_chamber'])
m['state'] = m['leg_state'].fillna(m['meta_state'])
m['name'] = m['full_name'].fillna(m['name'])
m.loc[m.entity_type != 'member', ['bioguide']] = ''
cols = ['screen_name', 'user_id', 'bioguide', 'icpsr', 'name', 'party', 'chamber', 'state', 'account_type',
        'entity_type', 'match_method', 'n_tweets', 'n_user_ids', 'party_switch', 'np_dim1', 'nom_dim1', 'vv_congresses']
m = m[cols].sort_values('screen_name')
m.to_csv(f'{OUT}/account_member_map.csv', index=False)

# coverage table
mem = m[m.entity_type == 'member']
cov = [
    ('accounts_in_congresstweets', len(m)),
    ('tweets_total', int(m.n_tweets.sum())),
    ('accounts_matched_any_entity', int((m.entity_type != 'unmatched').sum())),
    ('accounts_matched_to_member', len(mem)),
    ('tweets_matched_to_member', int(mem.n_tweets.sum())),
    ('members_matched', int(mem.bioguide.nunique())),
    ('accounts_committee', int((m.entity_type == 'committee').sum())),
    ('accounts_caucus', int((m.entity_type == 'caucus').sum())),
    ('accounts_party', int((m.entity_type == 'party').sum())),
    ('accounts_unmatched', int((m.entity_type == 'unmatched').sum())),
    ('tweets_unmatched', int(m[m.entity_type == 'unmatched'].n_tweets.sum())),
    ('member_accounts_office', int((mem.account_type == 'office').sum())),
    ('member_accounts_campaign', int((mem.account_type == 'campaign').sum())),
    ('members_with_both_office_and_campaign', int((mem.groupby('bioguide').account_type.nunique() == 2).sum())),
    ('members_party_D', int(mem.drop_duplicates('bioguide').party.eq('D').sum())),
    ('members_party_R', int(mem.drop_duplicates('bioguide').party.eq('R').sum())),
    ('members_party_I', int(mem.drop_duplicates('bioguide').party.eq('I').sum())),
    ('members_senate', int(mem.drop_duplicates('bioguide').chamber.eq('senate').sum())),
    ('members_house', int(mem.drop_duplicates('bioguide').chamber.eq('house').sum())),
    ('members_with_icpsr', int(mem.drop_duplicates('bioguide').icpsr.notna().sum())),
    ('members_with_nokken_poole', int(mem.drop_duplicates('bioguide').np_dim1.notna().sum())),
    ('members_party_switch_in_window', int(mem.drop_duplicates('bioguide').party_switch.fillna(0).sum())),
    ('screen_names_with_multiple_user_ids', int((m.n_user_ids > 1).sum())),
]
pd.DataFrame(cov, columns=['item', 'value']).to_csv(f'{OUT}/account_member_coverage.csv', index=False)
print(pd.DataFrame(cov, columns=['item', 'value']).to_string())
print(m[m.entity_type != 'member'].groupby('entity_type').n_tweets.agg(['size', 'sum']))
print(m[m.entity_type == 'unmatched'].sort_values('n_tweets', ascending=False).head(25)[['screen_name', 'user_id', 'n_tweets']].to_string())
