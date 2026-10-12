"""P3 forecasting library: member-year transitions, baselines, penalized regression, gradient-boosted trees,
rolling yearly origins, skill with member-cluster bootstrap CIs, calibration, permutation importance, SHAP.

Rules enforced here (MATH_SPEC section H):
  * features of the transition at origin t come only from panel rows of years t and t-1 plus scheduled calendar
    facts (election cycle, party, chamber); targets come from year t+1;
  * a model for origin t is fit only on transitions s -> s+1 with s+1 <= t (target year already observed);
  * hyperparameters are chosen by an inner rolling split inside those training transitions (fit on all but the last
    training transition, validate on the last); with a single training transition fixed defaults are used;
  * seeded (20261010).
Predictive associations only.
"""
import warnings

import numpy as np
import pandas as pd

SEED = 20261010
warnings.filterwarnings("ignore")

LEVELS = ["log_rate", "ti7", "tribal", "performative", "ragebait", "irony", "neg_sentiment", "offensive", "toxic",
          "genuine_anger", "advocacy"]
DESC = ["A_n", "rapid_share_300", "sessions_per_day_30", "events_per_session_30", "entropy", "night_00_06",
        "weekend_share"]
MIX = ["campaign_share", "sched_share", "tool_share", "phone_share"]
COMOVE = ["comove_beta", "momentum_beta"]
TRENDS = ["ti7_slope", "vol_trend"]
STATIC = ["party_R", "chamber_senate", "np_dim1", "abs_np_dim1", "seniority_congresses"]
CAL = ["aligned_with_president", "in_chamber_majority", "seat_up_general_cur", "seat_up_primary_cur",
       "seat_up_general_next", "seat_up_primary_next", "election_year_next"]
LAGS = ["lag_d_log_rate", "lag_d_ti7", "lag_d_tribal", "lag_d_performative", "lag_d_ragebait", "has_lag"]
HIST = ["dev_hist_log_rate", "dev_hist_ti7"]
FEATURES_FULL = LEVELS + DESC + MIX + COMOVE + TRENDS + STATIC + CAL + LAGS + HIST
FEATURES_CORE = LEVELS + DESC + TRENDS + STATIC + CAL + LAGS + HIST  # no account-mix, no co-movement (Lee-compatible)
GROUPS = {
    "volume level": ["log_rate", "lag_d_log_rate", "dev_hist_log_rate", "vol_trend"],
    "TI dimensions": ["ti7", "tribal", "performative", "ragebait", "irony", "neg_sentiment", "offensive", "toxic",
                      "genuine_anger", "advocacy", "lag_d_ti7", "lag_d_tribal", "lag_d_performative", "lag_d_ragebait",
                      "dev_hist_ti7", "ti7_slope"],
    "temporal descriptors": DESC,
    "account mix": MIX,
    "co-movement/momentum": COMOVE,
    "party/chamber/ideology/seniority": STATIC,
    "electoral calendar": ["seat_up_general_cur", "seat_up_primary_cur", "seat_up_general_next",
                           "seat_up_primary_next", "election_year_next"],
    "alignment/majority": ["aligned_with_president", "in_chamber_majority"],
}
TARGETS = {"d_logvol": ("log_rate", "volume"), "d_ti7": ("ti7", "TI7"), "d_tribal": ("tribal", "tribal"),
           "d_performative": ("performative", "performative"), "d_ragebait": ("ragebait", "ragebait")}
ESC = {"esc_ti7": "d_ti7", "esc_vol": "d_logvol"}


# ------------------------------------------------------------------ transitions

def seat_lookup(el):
    el = el.copy()
    el["seat_up"] = np.where(el.ballot_reason == "senate_seat_not_up", 0, 1)
    return el[el.cycle.isin([2018, 2020, 2022])].set_index(["bioguide", "cycle"]).seat_up.to_dict()


def seat_flags(su, bg, t):
    """Scheduled seat-up flags for the PY ending at origin t (cur) and the PY being forecast (next). Known in advance:
    the House is up every even year, Senate by class. General election of cycle c is in PY c, primaries in PY c-1."""
    f = {}
    f["seat_up_general_cur"] = su.get((bg, t), np.nan) if t % 2 == 0 else 0
    f["seat_up_primary_cur"] = su.get((bg, t + 1), np.nan) if (t + 1) % 2 == 0 else 0
    f["seat_up_general_next"] = su.get((bg, t + 1), np.nan) if (t + 1) % 2 == 0 else 0
    f["seat_up_primary_next"] = su.get((bg, t + 2), np.nan) if (t + 2) % 2 == 0 else 0
    f["election_year_next"] = int((t + 1) % 2 == 0)
    return f


def quality(panel, min_events=50, min_office_days=150, min_scored=25):
    ok = (panel.n_events >= min_events) & (panel.in_office_days >= min_office_days) & (panel.n_scored.fillna(0) >= min_scored)
    return ok


def make_transitions(panel, su, min_scored=25, min_events=50, min_office_days=150):
    """One row per member and origin t (features from t, t-1; target from t+1). Returns (T, attrition)."""
    p = panel.copy()
    p["ok"] = quality(p, min_events, min_office_days, min_scored)
    p["party_R"] = (p.party == "R").astype(float)
    p.loc[~p.party.isin(["R", "D"]), "party_R"] = np.nan
    p["chamber_senate"] = (p.chamber == "senate").astype(float)
    p["aligned_with_president"] = p.aligned_with_president.astype(float)
    p["in_chamber_majority"] = p.in_chamber_majority.astype(float)
    key = p.set_index(["bioguide", "py"])
    rows, att = [], []
    for t in sorted(p.py.unique()):
        if t + 1 > p.py.max():
            continue
        cur = p[(p.py == t) & p.ok]
        nxt = key
        n_cur = len(cur)
        left = retained = fail = 0
        for _, r in cur.iterrows():
            bg = r.bioguide
            if (bg, t + 1) not in nxt.index:
                left += 1
                continue
            n1 = nxt.loc[(bg, t + 1)]
            if not bool(n1.ok):
                fail += 1
                continue
            row = {c: r.get(c, np.nan) for c in LEVELS + DESC + MIX + COMOVE + TRENDS + STATIC +
                   ["aligned_with_president", "in_chamber_majority"]}
            row.update(seat_flags(su, bg, t))
            row.update(bioguide=bg, origin=t, party=r.party, chamber=r.chamber,
                       ti7_se=r.ti7_se, ti7_se_next=n1.ti7_se)
            # lag deltas (year t minus t-1) and history deviation: panel rows up to year t only
            hist = p[(p.bioguide == bg) & (p.py <= t) & p.ok]
            prev = hist[hist.py == t - 1]
            has = len(prev) == 1
            row["has_lag"] = float(has)
            for v, nm in (("log_rate", "lag_d_log_rate"), ("ti7", "lag_d_ti7"), ("tribal", "lag_d_tribal"),
                          ("performative", "lag_d_performative"), ("ragebait", "lag_d_ragebait")):
                row[nm] = float(r[v] - prev[v].iloc[0]) if has else 0.0
            for v in ("log_rate", "ti7"):
                row[f"dev_hist_{v}"] = float(hist[v].mean() - r[v])
            # targets
            row["d_logvol"] = float(n1.log_rate - r.log_rate)
            for tg in ("ti7", "tribal", "performative", "ragebait"):
                row[f"d_{tg}"] = float(n1[tg] - r[tg])
            rows.append(row)
            retained += 1
        att.append(dict(origin=t, target_year=t + 1, members_eligible_t=n_cur, left_or_missing_next=left,
                        fail_quality_next=fail, retained=retained,
                        attrition_share=1 - retained / n_cur if n_cur else np.nan))
    T = pd.DataFrame(rows)
    # within-target-year top-quintile escalation (outcome definition; uses only that year's retained members)
    for e, d in ESC.items():
        T[e] = 0
        for t, g in T.groupby("origin"):
            thr = g[d].quantile(0.8)
            T.loc[g.index, e] = (g[d] > thr).astype(int)
    return T, pd.DataFrame(att)


# ------------------------------------------------------------------ models

def _prep(Xtr, Xte, feats):
    med = Xtr[feats].median().fillna(0.0)
    a = Xtr[feats].fillna(med)
    b = Xte[feats].fillna(med)
    mu, sd = a.mean(), a.std().replace(0, 1).fillna(1)
    return ((a - mu) / sd).values, ((b - mu) / sd).values


def _ridge(Xa, y, Xb, alpha):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=alpha, random_state=SEED).fit(Xa, y).predict(Xb)


def _enet(Xa, y, Xb, alpha):
    from sklearn.linear_model import ElasticNet
    return ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=SEED, max_iter=5000).fit(Xa, y).predict(Xb)


GRID = {
    "ridge": [1.0, 10.0, 100.0, 1000.0],
    "enet": [0.0005, 0.002, 0.01, 0.05],
    "gbt": [dict(n_estimators=60, num_leaves=4), dict(n_estimators=150, num_leaves=4),
            dict(n_estimators=60, num_leaves=8), dict(n_estimators=150, num_leaves=8)],
}
DEFAULT = {"ridge": 100.0, "enet": 0.01, "gbt": dict(n_estimators=100, num_leaves=4)}


def _gbt(Xa, y, Xb, hp, binary=False, return_model=False):
    import lightgbm as lgb
    kw = dict(learning_rate=0.05, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=5.0, random_state=SEED, n_jobs=8, verbose=-1, **hp)
    m = lgb.LGBMClassifier(**kw) if binary else lgb.LGBMRegressor(**kw)
    m.fit(Xa, y)
    p = m.predict_proba(Xb)[:, 1] if binary else m.predict(Xb)
    return (p, m) if return_model else p


def _logit(Xa, y, Xb, C):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=C, max_iter=2000, random_state=SEED).fit(Xa, y).predict_proba(Xb)[:, 1]


GRID["logit"] = [0.003, 0.03, 0.3]
DEFAULT["logit"] = 0.03


def _loss_cont(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _loss_bin(y, p):
    return float(np.mean((y - p) ** 2))


def fit_predict(kind, Ttr, Tte, feats, target, binary=False, return_model=False):
    """Fit `kind` on Ttr (a set of earlier transitions) and predict Tte. Hyperparameter = inner rolling split.
    Returns (pred, chosen_hp, model_or_None)."""
    origins = sorted(Ttr.origin.unique())
    ytr = Ttr[target].values.astype(float)

    def run(A, yA, B, hp, ret=False):
        if kind == "gbt":
            return _gbt(A[feats], yA, B[feats], hp, binary, ret)
        Xa, Xb = _prep(A, B, feats)
        if kind == "ridge":
            return _ridge(Xa, yA, Xb, hp)
        if kind == "enet":
            return _enet(Xa, yA, Xb, hp)
        if kind == "logit":
            return _logit(Xa, yA, Xb, hp)
        raise ValueError(kind)

    hp = DEFAULT[kind]
    if len(origins) >= 2:
        A = Ttr[Ttr.origin < origins[-1]]
        V = Ttr[Ttr.origin == origins[-1]]
        best, bl = None, np.inf
        for cand in GRID[kind]:
            pv = run(A, A[target].values.astype(float), V, cand)
            lv = (_loss_bin if binary else _loss_cont)(V[target].values.astype(float), pv)
            if lv < bl:
                best, bl = cand, lv
        hp = best
    if kind == "gbt" and return_model:
        p, m = run(Ttr, ytr, Tte, hp, True)
        return p, hp, m
    return run(Ttr, ytr, Tte, hp), hp, None


# ------------------------------------------------------------------ baselines

def baselines_reg(Ttr, Tte, target, level_col):
    """Deployable baselines: persistence (0), regression to the mean on own level, member history mean, party x chamber
    mean change in training transitions, small OLS (own level + party + chamber + election-year flag). Plus the NON-deployable party-year oracle (target-year group mean)."""
    out = {"persistence": np.zeros(len(Tte))}
    a = np.polyfit(Ttr[level_col].values, Ttr[target].values, 1)
    out["regression_to_mean"] = np.polyval(a, Tte[level_col].values)
    hist = {"d_logvol": "dev_hist_log_rate", "d_ti7": "dev_hist_ti7"}.get(target)
    if hist:
        out["member_history_mean"] = Tte[hist].values
    cols = [level_col, "party_R", "chamber_senate", "election_year_next"]
    A = np.column_stack([np.ones(len(Ttr))] + [Ttr[c].fillna(Ttr[c].median()).values for c in cols])
    beta, *_ = np.linalg.lstsq(A, Ttr[target].values, rcond=None)
    B = np.column_stack([np.ones(len(Tte))] + [Tte[c].fillna(Ttr[c].median()).values for c in cols])
    out["small_linear"] = B @ beta
    gm = Ttr.groupby(["party", "chamber"])[target].mean()
    out["party_mean_change"] = np.array([gm.get((p, c), Ttr[target].mean()) for p, c in zip(Tte.party, Tte.chamber)])
    oracle = Tte.groupby(["party", "chamber"])[target].transform("mean")
    out["party_year_ORACLE"] = oracle.values
    return out


# ------------------------------------------------------------------ rolling origins

def rolling(T, origins, targets=TARGETS, feats_full=FEATURES_FULL, feats_core=FEATURES_CORE, final_origin=None):
    """Predict each origin from earlier transitions. Returns long DataFrame of predictions."""
    recs = []
    hps = []
    for t in origins:
        Ttr = T[T.origin <= t - 1].reset_index(drop=True)
        Tte = T[T.origin == t].reset_index(drop=True)
        if len(Ttr) == 0 or len(Tte) == 0:
            continue
        for tg, (lvl, _) in targets.items():
            preds = baselines_reg(Ttr, Tte, tg, lvl)
            for kind, fs, tag in (("ridge", feats_full, "ridge"), ("enet", feats_full, "enet"),
                                  ("gbt", feats_full, "gbt"), ("ridge", feats_core, "ridge_core"),
                                  ("gbt", feats_core, "gbt_core")):
                p, hp, _ = fit_predict(kind, Ttr, Tte, fs, tg)
                preds[tag] = p
                hps.append(dict(origin=t, target=tg, model=tag, hp=str(hp)))
            for m, p in preds.items():
                recs.append(pd.DataFrame(dict(bioguide=Tte.bioguide, origin=t, target=tg, model=m, pred=p,
                                              y=Tte[tg].values, party=Tte.party, chamber=Tte.chamber)))
        for e, d in ESC.items():
            base = float(Ttr[e].mean())
            score_b = {"base_rate": np.full(len(Tte), base),
                       "persistence_score": Tte[{"esc_ti7": "lag_d_ti7", "esc_vol": "lag_d_log_rate"}[e]].values,
                       "rtm_score": -Tte[{"esc_ti7": "ti7", "esc_vol": "log_rate"}[e]].values}
            for kind, fs, tag in (("logit", feats_full, "logit"), ("gbt", feats_full, "gbt"),
                                  ("logit", feats_core, "logit_core"), ("gbt", feats_core, "gbt_core")):
                p, hp, _ = fit_predict(kind, Ttr, Tte, fs, e, binary=True)
                score_b[tag] = p
                hps.append(dict(origin=t, target=e, model=tag, hp=str(hp)))
            for m, p in score_b.items():
                recs.append(pd.DataFrame(dict(bioguide=Tte.bioguide, origin=t, target=e, model=m, pred=p,
                                              y=Tte[e].values, party=Tte.party, chamber=Tte.chamber)))
    return pd.concat(recs, ignore_index=True), pd.DataFrame(hps)


# ------------------------------------------------------------------ metrics

def _member_sums(df, kind, loss_kind="mse"):
    d = df.assign(se=(df.y - df.pred) ** 2, ae=(df.y - df.pred).abs())
    g = d.groupby("bioguide").agg(se=("se", "sum"), ae=("ae", "sum"), n=("se", "size"))
    return g


def skill_ci(df, target, model, base_model, B=2000, seed=SEED, metric="rmse"):
    """Skill = 1 - score_model / score_base (RMSE or MAE, or Brier for binary) pooled over rows, cluster bootstrap over
    members. df is the long prediction table for the rows to pool."""
    a = _member_sums(df[(df.target == target) & (df.model == model)], None)
    b = _member_sums(df[(df.target == target) & (df.model == base_model)], None)
    j = a.join(b, lsuffix="_m", rsuffix="_b", how="inner")
    col = "se" if metric in ("rmse", "brier") else "ae"
    n = j.n_m.values.astype(float)
    sm, sb = j[f"{col}_m"].values, j[f"{col}_b"].values
    root = metric == "rmse"

    def sk(w):
        nm, nb = (w * sm).sum() / (w * n).sum(), (w * sb).sum() / (w * n).sum()
        return 1 - (np.sqrt(nm) / np.sqrt(nb) if root else nm / nb)

    pt = sk(np.ones(len(j)))
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(j), np.ones(len(j)) / len(j), size=B).astype(float)
    num_m, num_b, den = W @ sm, W @ sb, W @ n
    nm, nb = num_m / den, num_b / den
    bs = 1 - (np.sqrt(nm) / np.sqrt(nb) if root else nm / nb)
    return dict(skill=float(pt), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                score_model=float(np.sqrt(sm.sum() / n.sum()) if root else sm.sum() / n.sum()),
                score_base=float(np.sqrt(sb.sum() / n.sum()) if root else sb.sum() / n.sum()), n_rows=int(n.sum()),
                n_members=int(len(j)))


def auc_ci(df, target, model, B=500, seed=SEED):
    from sklearn.metrics import roc_auc_score
    d = df[(df.target == target) & (df.model == model)]
    y, p, mb = d.y.values, d.pred.values, d.bioguide.values
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return dict(auc=0.5, lo=0.5, hi=0.5)
    auc = roc_auc_score(y, p)
    idx = pd.Series(np.arange(len(d))).groupby(mb).apply(np.array)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(idx), len(idx))
        ii = np.concatenate([idx.iloc[k] for k in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        vals.append(roc_auc_score(y[ii], p[ii]))
    return dict(auc=float(auc), lo=float(np.percentile(vals, 2.5)), hi=float(np.percentile(vals, 97.5)))


def reliability(y, p, bins=10):
    """Equal-frequency reliability table and ECE."""
    q = pd.qcut(pd.Series(p).rank(method="first"), bins, labels=False)
    t = pd.DataFrame(dict(y=y, p=p, q=q)).groupby("q").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    ece = float((t.n * (t.pred - t.obs).abs()).sum() / t.n.sum())
    return t.reset_index(), ece


def calib_slope(y, p):
    """OLS slope of realised on predicted change (1 = calibrated spread) and intercept."""
    if np.std(p) < 1e-12:
        return dict(slope=np.nan, intercept=float(np.mean(y)))
    b, a = np.polyfit(p, y, 1)
    return dict(slope=float(b), intercept=float(a))


# ------------------------------------------------------------------ importance

def perm_importance(model_fn, Tte, feats, target, groups=GROUPS, n_rep=20, seed=SEED, binary=False):
    """model_fn(df)->predictions. Importance = mean increase in RMSE (Brier for binary) when a feature (or group) is
    permuted across test rows."""
    rng = np.random.default_rng(seed)
    y = Tte[target].values.astype(float)
    loss = _loss_bin if binary else _loss_cont
    base = loss(y, model_fn(Tte))
    rows = []
    items = [(f, [f]) for f in feats] + [("GROUP: " + g, [c for c in cols if c in feats]) for g, cols in groups.items()]
    for name, cols in items:
        if not cols:
            continue
        inc = []
        for _ in range(n_rep):
            X = Tte.copy()
            perm = rng.permutation(len(X))
            for c in cols:
                X[c] = X[c].values[perm]
            inc.append(loss(y, model_fn(X)) - base)
        rows.append(dict(feature=name, importance=float(np.mean(inc)), sd=float(np.std(inc)), base=base))
    return pd.DataFrame(rows)
