import numpy as np, pandas as pd, itertools, warnings
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
warnings.filterwarnings('ignore')
V7 = '/home/joneill/x_box/pipeline/output/v7'
O = V7 + '/rvt_b'
rng = np.random.default_rng(20261013)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)

# ---------------- margins from FEC official results
def col(cols, key):
    h = [c for c in cols if key in str(c).upper()]
    return h[0] if h else None

mrows = []
for y in (2018, 2020, 2022):
    xl = pd.ExcelFile(f'{V7}/t3/src/fecres/fe{y}.xlsx')
    for sh in xl.sheet_names:
        if ('HOUSE RESULTS' in sh.upper() or 'SENATE RESULTS' in sh.upper()) and 'NC 09' not in sh.upper():
            d = xl.parse(sh, header=0)
            cid = col(d.columns, 'FEC ID')
            st = col(d.columns, 'STATE ABBREVIATION')
            dist = col(d.columns, 'DISTRICT')
            gp = [c for c in d.columns if str(c).strip().upper() == 'GENERAL %'][0]
            gv = [c for c in d.columns if str(c).strip().upper() == 'GENERAL VOTES'][0]
            win = [c for c in d.columns if 'GE WINNER' in str(c).upper()][0]
            q = pd.DataFrame({'id': d[cid].astype(str).str.strip().str.upper(), 'state': d[st], 'dist': d[dist].astype(str), 'pct': pd.to_numeric(d[gp], errors='coerce'),
                              'gv': pd.to_numeric(d[gv], errors='coerce'), 'win': d[win].astype(str).str.upper().str.startswith('W')})
            q = q[q.id.str.match(r'^[HSP]\d[A-Z]{2}\d{5}$') & q.state.notna()]
            q['kind'] = 'S' if 'SENATE' in sh.upper() else 'H'
            q['year'] = y
            mrows.append(q)
Q = pd.concat(mrows)
out = []
for (y, k, s, dst), g in Q.groupby(['year', 'kind', 'state', 'dist']):
    gg = g[g.pct.notna() & (g.pct > 0)]
    nwin = g.win.sum()
    if k == 'S' and nwin > 1:
        continue  # two-seat states (GA 2020): ambiguous, leave missing
    if len(gg) == 0:
        if nwin == 1:
            for i in g[g.win].id:
                out.append((y, i, 1.0, 1))
        continue
    if len(gg) == 1:
        out.append((y, gg.id.iloc[0], 1.0, 1))
        continue
    for i, p in zip(gg.id, gg.pct):
        others = gg.pct[gg.id != i].max()
        out.append((y, i, p - others, 0))
M = pd.DataFrame(out, columns=['year', 'fec', 'margin', 'uncontested']).drop_duplicates(['year', 'fec'])
el = pd.read_csv(V7 + '/t3/elections.csv', usecols=['bioguide', 'cycle', 'fec_candidate_id'])
el['fec'] = el.fec_candidate_id.astype(str).str.upper()
el = el.merge(M, left_on=['cycle', 'fec'], right_on=['year', 'fec'], how='left')
marg = el.dropna(subset=['margin']).groupby(['bioguide', 'cycle']).margin.first().unstack()
print('margins by cycle: members with margin', marg.notna().sum().to_dict(), 'of', el.bioguide.nunique())

# ---------------- member features
md = pd.read_parquet(O + '/member_day.parquet', columns=['date', 'bioguide', 'ti_sum', 'n', 'n_camp'])
cal = pd.read_parquet(V7 + '/t3/calendar_member_day.parquet', columns=['date', 'bioguide', 'in_office', 'party', 'term_chamber', 'aligned_with_president', 'in_chamber_majority'])
cal['bioguide'] = cal.bioguide.astype(str)
cal['term_chamber'] = cal.term_chamber.astype(str)
cal['party'] = cal.party.astype(str)
terms = pd.read_csv(V7 + '/t3/src/terms_all.csv', parse_dates=['start'])
first = terms.groupby('bioguide').start.min()
am = pd.read_csv(V7 + '/ti/account_member_map.csv').dropna(subset=['bioguide']).drop_duplicates('bioguide').set_index('bioguide')
ext = am.nom_dim1.abs().where(am.nom_dim1.notna(), am.np_dim1.abs())
T = {'2019-01-03': 2018, '2021-01-20': 2020, '2023-01-03': 2022}
rows = []
for ts, cyc in T.items():
    t = pd.Timestamp(ts)
    pre = md[(md.date >= t - pd.Timedelta(days=180)) & (md.date <= t - pd.Timedelta(days=1))].groupby('bioguide')[['ti_sum', 'n', 'n_camp']].sum()
    post = md[(md.date >= t) & (md.date <= t + pd.Timedelta(days=179))].groupby('bioguide')[['ti_sum', 'n']].sum()
    c0 = cal[cal.date == t - pd.Timedelta(days=1)].set_index('bioguide')
    c1 = cal[cal.date == t].set_index('bioguide')
    c2 = cal[cal.date == t + pd.Timedelta(days=179)].set_index('bioguide')
    X = pre.join(post, lsuffix='_pre', rsuffix='_post', how='inner')
    X = X.join(c0[['in_office', 'party', 'term_chamber', 'aligned_with_president', 'in_chamber_majority']].add_suffix('_0'), how='inner')
    X = X.join(c1[['aligned_with_president', 'in_chamber_majority']].add_suffix('_1'), how='inner')
    X = X.join(c2[['in_office']].add_suffix('_2'), how='inner')
    X = X[(X.in_office_0 == 1) & (X.in_office_2 == 1) & (X.n_pre >= 30) & (X.n_post >= 30) & X.party_0.isin(['D', 'R'])].copy()
    X['dti'] = X.ti_sum_post / X.n_post - X.ti_sum_pre / X.n_pre
    X['pre_ti7'] = X.ti_sum_pre / X.n_pre
    X['camp'] = X.n_camp / X.n_pre
    X['dpower'] = (X.aligned_with_president_1 + X.in_chamber_majority_1) - (X.aligned_with_president_0 + X.in_chamber_majority_0)
    X['majority_before'] = X.in_chamber_majority_0
    X['senate'] = (X.term_chamber_0 == 'senate').astype(int)
    X['rep'] = (X.party_0 == 'R').astype(int)
    X['seniority'] = [(t - first.get(b, t)).days / 365.25 for b in X.index]
    mm = []
    for b in X.index:
        v = np.nan
        if b in marg.index:
            for cy in range(cyc, 2017, -2):
                if cy in marg.columns and pd.notna(marg.at[b, cy]):
                    v = marg.at[b, cy]
                    break
        mm.append(v)
    X['margin'] = mm
    X['extremity'] = ext.reindex(X.index).values
    X['trans'] = ts
    X['n_post_posts'] = X.n_post
    rows.append(X.reset_index().rename(columns={'index': 'bioguide'}))
D = pd.concat(rows, ignore_index=True)
D['margin_missing'] = D.margin.isna().astype(int)
D['tr2021'] = (D.trans == '2021-01-20').astype(int)
D['tr2023'] = (D.trans == '2023-01-03').astype(int)
D = D.dropna(subset=['extremity'])
print('N rows', len(D), D.groupby('trans').size().to_dict(), 'members', D.bioguide.nunique())
print('margin missing share by trans/chamber:\n', D.groupby(['trans', 'senate']).margin_missing.mean().round(3).to_string())
print('dpower counts\n', pd.crosstab(D.trans, D.dpower).to_string())
print('dti mean/sd by trans', D.groupby('trans').dti.agg(['mean', 'std']).round(4).to_string())
D.to_csv(O + '/h3_data.csv', index=False)

CONT = ['seniority', 'margin', 'senate', 'majority_before', 'extremity', 'camp', 'rep', 'pre_ti7', 'dpower']
DUM = ['tr2021', 'tr2023', 'margin_missing']
PAIRS = list(itertools.combinations(CONT, 2))


def prep(tr, te, inter):
    A = tr[CONT + DUM].copy().astype(float)
    B = te[CONT + DUM].copy().astype(float)
    med = A.margin.median()
    A['margin'] = A.margin.fillna(med)
    B['margin'] = B.margin.fillna(med)
    mu = A[CONT].mean()
    sd = A[CONT].std().replace(0, 1)
    for Z in (A, B):
        Z[CONT] = (Z[CONT] - mu) / sd
    if inter:
        for a, b in PAIRS:
            A[a + 'x' + b] = A[a] * A[b]
            B[a + 'x' + b] = B[a] * B[b]
    m2 = A.mean()
    s2 = A.std().replace(0, 1)
    return A, B, (A - m2) / s2, (B - m2) / s2


ALPH = np.logspace(-2, 4, 25)


def fitpred(tr, te, which):
    if which == 'mean':
        return np.full(len(te), tr.dti.mean())
    if which == 'A':
        _, _, Xa, Xb = prep(tr, te, False)
    elif which == 'B':
        _, _, Xa, Xb = prep(tr, te, True)
    else:
        Xa, Xb, _, _ = prep(tr, te, False)
    if which in ('A', 'B'):
        gs = GridSearchCV(Ridge(), {'alpha': ALPH}, cv=GroupKFold(5), scoring='neg_mean_squared_error')
        gs.fit(Xa, tr.dti.values, groups=tr.bioguide.values)
        return gs.predict(Xb)
    g = GradientBoostingRegressor(max_depth=3, n_estimators=300, learning_rate=0.03, subsample=0.7, min_samples_leaf=20, random_state=1)
    g.fit(Xa, tr.dti.values)
    return g.predict(Xb)


MODELS = ['mean', 'A', 'B', 'C']


def cv_loto():
    P = {m: np.zeros(len(D)) for m in MODELS}
    for ts in D.trans.unique():
        te = D.trans == ts
        for m in MODELS:
            P[m][te.values] = fitpred(D[~te], D[te], m)
    return P


def cv_group(seed):
    P = {m: np.zeros(len(D)) for m in MODELS}
    ug = D.bioguide.unique()
    r = np.random.default_rng(seed)
    perm = r.permutation(len(ug))
    fold = {g: i % 10 for g, i in zip(ug[perm], range(len(ug)))}
    f = D.bioguide.map(fold).values
    for k in range(10):
        te = f == k
        for m in MODELS:
            P[m][te] = fitpred(D[~te], D[te], m)
    return P


def summarize(label, Ps):
    # Ps list of prediction dicts (1 for loto, 5 for group)
    y = D.dti.values
    se = {m: np.mean([(y - P[m]) ** 2 for P in Ps], axis=0) for m in MODELS}
    mem = D.bioguide.values
    ug = np.unique(mem)
    idx = {g: np.where(mem == g)[0] for g in ug}
    res = {}
    for m in MODELS:
        res[m] = dict(mse=se[m].mean(), r2=1 - se[m].sum() / se['mean'].sum())
    print(label, {m: (round(v['mse'] * 1e4, 4), round(v['r2'], 4)) for m, v in res.items()}, '(MSE x1e4, R2 vs mean)')
    B = 2000
    sums = {m: np.array([se[m][idx[g]].sum() for g in ug]) for m in MODELS}
    cnt = np.array([len(idx[g]) for g in ug])
    rec = []
    for a, b in (('A', 'B'), ('A', 'C'), ('mean', 'A')):
        d = []
        for _ in range(B):
            s = rng.integers(0, len(ug), len(ug))
            d.append((sums[a][s].sum() - sums[b][s].sum()) / cnt[s].sum())
        d = np.array(d)
        pt = se[a].mean() - se[b].mean()
        rec.append(dict(scheme=label, comparison=f'MSE({a})-MSE({b})', diff_e4=pt * 1e4, lo_e4=np.percentile(d, 2.5) * 1e4, hi_e4=np.percentile(d, 97.5) * 1e4))
        print(label, rec[-1])
    return res, rec


PL = cv_loto()
r1, c1 = summarize('LOTO', [PL])
PGs = [cv_group(s) for s in range(5)]
r2, c2 = summarize('GROUP10x5', PGs)
pd.DataFrame(c1 + c2).to_csv(O + '/h3_skill_diffs.csv', index=False)
pd.DataFrame({k + '_' + m: v[m] for k, v in (('loto', r1), ('grp', r2)) for m in MODELS}).to_csv(O + '/h3_skill.csv')
# per-transition R2 (LOTO)
for ts in D.trans.unique():
    te = (D.trans == ts).values
    y = D.dti.values[te]
    print('LOTO held-out', ts, 'n', te.sum(), {m: round(1 - ((y - PL[m][te]) ** 2).sum() / ((y - PL['mean'][te]) ** 2).sum(), 4) for m in ('A', 'B', 'C')})

# ---------------- full-data inference
A, _, Xs, _ = prep(D, D, True)
sdy = D.dti.std()
gs = GridSearchCV(Ridge(), {'alpha': ALPH}, cv=GroupKFold(5), scoring='neg_mean_squared_error').fit(Xs, D.dti.values, groups=D.bioguide.values)
alpha = gs.best_params_['alpha']
print('ridge-B alpha (full data)', alpha)
ints = [a + 'x' + b for a, b in PAIRS]
coefs = []
ug = D.bioguide.unique()
for _ in range(200):
    pick = rng.choice(ug, len(ug))
    Db = pd.concat([D[D.bioguide == g] for g in pick], ignore_index=True)
    _, _, Xb, _ = prep(Db, Db, True)
    r = Ridge(alpha=alpha).fit(Xb, Db.dti.values)
    coefs.append(pd.Series(r.coef_ / sdy, index=Xb.columns))
Cf = pd.DataFrame(coefs)[ints]
base = Ridge(alpha=alpha).fit(Xs, D.dti.values)
b0 = pd.Series(base.coef_ / sdy, index=Xs.columns)[ints]
# OLS with cluster SE on A + products (unscaled products of standardized features; y in raw units)
Xo = sm.add_constant(A)
o = sm.OLS(D.dti.values, Xo).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.bioguide)[0]})
po = o.pvalues[ints]
ordr = np.argsort(po.values)
holm = np.empty(len(po))
run = 0
for r_, i in enumerate(ordr):
    run = max(run, min(1, (len(po) - r_) * po.values[i]))
    holm[i] = run
tab = pd.DataFrame({'ridgeB_coef_sdy': b0, 'boot_lo': Cf.quantile(.025), 'boot_hi': Cf.quantile(.975), 'sign_agree': Cf.apply(lambda c: (np.sign(c) == np.sign(c.mean())).mean()),
                    'ols_coef': o.params[ints], 'ols_p': po, 'holm_p': pd.Series(holm, index=po.index)})
tab['coef_sd_ols'] = tab.ols_coef / sdy
tab['large_stable'] = ((tab.boot_lo > 0) | (tab.boot_hi < 0)) & (tab.ridgeB_coef_sdy.abs() >= 0.1) & (tab.holm_p < 0.05)
tab = tab.reindex(tab.ridgeB_coef_sdy.abs().sort_values(ascending=False).index)
tab.to_csv(O + '/h3_interactions.csv')
print(tab.head(10).round(4).to_string())
print('n large&stable', int(tab.large_stable.sum()), ' n holm<0.05', int((tab.holm_p < 0.05).sum()), ' min holm p', tab.holm_p.min().round(4))

# additive associations
Xa_ = sm.add_constant(prep(D, D, False)[0])
oa = sm.OLS(D.dti.values, Xa_).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(D.bioguide)[0]})
ci = oa.conf_int()
add = pd.DataFrame({'coef': oa.params, 'lo': ci[0], 'hi': ci[1], 'p': oa.pvalues})
add.to_csv(O + '/h3_additive.csv')
print('ADDITIVE (dTI7 per 1 SD; SD of dTI7 =', round(sdy, 4), ')\n', add.round(4).to_string())
L = D[D.dpower < 0]
Al = prep(L, L, False)[0].drop(columns=['dpower', 'tr2021', 'tr2023'], errors='ignore')
Al = Al.loc[:, Al.std() > 0]
ol = sm.OLS(L.dti.values, sm.add_constant(Al)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(L.bioguide)[0]})
cil = ol.conf_int()
print('LOSERS (dpower<0) n', len(L), 'margin coef', round(ol.params['margin'], 5), 'CI', cil.loc['margin'].round(5).tolist(), 'p', round(ol.pvalues['margin'], 4))
Lc = L[L.margin_missing == 0]
Al2 = prep(Lc, Lc, False)[0].drop(columns=['dpower', 'tr2021', 'tr2023', 'margin_missing'], errors='ignore')
Al2 = Al2.loc[:, Al2.std() > 0]
ol2 = sm.OLS(Lc.dti.values, sm.add_constant(Al2)).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(Lc.bioguide)[0]})
print('LOSERS complete-margin n', len(Lc), 'margin coef', round(ol2.params['margin'], 5), 'CI', ol2.conf_int().loc['margin'].round(5).tolist(), 'p', round(ol2.pvalues['margin'], 4))
