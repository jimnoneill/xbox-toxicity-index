"""Kistner et al. (2026) external-validation helpers: panel, power config, features, models, skill, bootstrap.
Standalone (does not import or modify any existing xbox module)."""
import numpy as np
import pandas as pd

DATA = "/home/joneill/x_box/pipeline/data/kistner"
SEED = 20261012
MIN_N = 50  # min messages in a member-year (primary measure) to enter the panel

# Power configuration by calendar year: who holds each office for (essentially) the whole year.
# Known after the previous November election (2021 Senate: D control from 20 Jan 2021 after the Georgia runoffs).
PRES = {y: ("D" if (y <= 2016 or y >= 2021) else "R") for y in range(2009, 2023)}
HOUSE = {y: ("D" if (y <= 2010 or y >= 2019) else "R") for y in range(2009, 2023)}
SENATE = {y: ("D" if (y <= 2014 or y >= 2021) else "R") for y in range(2009, 2023)}


def congress_of(year):
    return (year - 1789) // 2 + 1  # 2011 -> 112


def load_voteview():
    v = pd.read_csv(f"{DATA}/raw/Voteview.tab", sep="\t", low_memory=False)
    v = v[v.congress.between(111, 117) & v.chamber.isin(["House", "Senate"])]
    v = v[["congress", "chamber", "icpsr"]].drop_duplicates(["congress", "icpsr"], keep="last")
    return v.rename(columns={"icpsr": "MemberICPSR"})


def build_panel(modes=("Twitter",)):
    """Member-year panel of the negative-partisan share, pooled over `modes`."""
    d = pd.read_csv(f"{DATA}/member_year_mode.csv")
    d = d[d.Mode.isin(modes)]
    g = d.groupby(["MemberICPSR", "Year", "MemberParty"], as_index=False)[["n", "neg", "pos", "adv", "cc", "cs", "bip"]].sum()
    g["share"] = g.neg / g.n
    for c in ["pos", "adv", "cc", "cs", "bip"]:
        g[c + "_sh"] = g[c] / g.n
    g["congress"] = g.Year.map(congress_of)
    g = g.merge(load_voteview(), on=["MemberICPSR", "congress"], how="left")
    return g.rename(columns={"MemberParty": "party"})


def power_features(year, party, chamber):
    """Power-status features for forecast year `year` (all known after the November election of year-1).
    `party`, `chamber` are arrays. Features are own-party relative."""
    party = np.asarray(party)
    def own(tab, y):
        return (party == tab[y]).astype(float)
    pres = own(PRES, year); house = own(HOUSE, year); sen = own(SENATE, year)
    pres0 = own(PRES, year - 1); house0 = own(HOUSE, year - 1); sen0 = own(SENATE, year - 1)
    isH = (np.asarray(chamber) == "House").astype(float)
    ch = np.where(isH == 1, house, sen); ch0 = np.where(isH == 1, house0, sen0)
    oth = np.where(isH == 1, sen, house); oth0 = np.where(isH == 1, sen0, house0)
    R = (np.asarray(party) == "R").astype(float)
    tri = pres + house + sen; tri0 = pres0 + house0 + sen0
    f = {
        "p_pres": pres, "p_ch": ch, "p_oth": oth, "p_tri": tri,
        "p_dpres": pres - pres0, "p_dch": ch - ch0, "p_doth": oth - oth0, "p_dtri": tri - tri0,
        "p_dpres_xR": (pres - pres0) * R, "p_dch_xR": (ch - ch0) * R,
        "p_lost_any": ((pres - pres0 < 0) | (ch - ch0 < 0) | (oth - oth0 < 0)).astype(float),
        "p_gain_any": ((pres - pres0 > 0) | (ch - ch0 > 0) | (oth - oth0 > 0)).astype(float),
    }
    return pd.DataFrame(f)


POWER = ["p_pres", "p_ch", "p_oth", "p_tri", "p_dpres", "p_dch", "p_doth", "p_dtri", "p_dpres_xR", "p_dch_xR",
         "p_lost_any", "p_gain_any"]
STATIC = ["isR", "isSen"]  # party, chamber (scheduled facts)
PROFILE = ["lag_share", "lag_dshare", "lag_dshare_miss", "lag_logn", "lag_pos", "lag_adv", "lag_cc", "lag_cs", "lag_bip",
           "lag_fb_share", "lag_fb_miss"]


def build_transitions(panel, fb=None, target_years=range(2011, 2023)):
    """One row per member and target year Y: needs rows in Y-1 and Y (both n >= MIN_N).
    Features use panel rows of years <= Y-1 only (+ scheduled facts + power config of Y).
    `target` = share_Y - share_{Y-1}."""
    p = panel.set_index(["MemberICPSR", "Year"])
    rows = []
    for Y in target_years:
        cur = panel[(panel.Year == Y) & (panel.n >= MIN_N)]
        prev = panel[(panel.Year == Y - 1) & (panel.n >= MIN_N)]
        m = prev.merge(cur[["MemberICPSR", "share"]], on="MemberICPSR", suffixes=("", "_tgt"))
        if m.empty:
            continue
        pp = panel[panel.Year == Y - 2][["MemberICPSR", "share"]].rename(columns={"share": "share_l2"})
        m = m.merge(pp, on="MemberICPSR", how="left")
        out = pd.DataFrame({
            "MemberICPSR": m.MemberICPSR.values, "Y": Y, "party": m.party.values, "chamber": m.chamber.values,
            "lag_share": m.share.values, "lag_dshare": (m.share - m.share_l2).fillna(0).values,
            "lag_dshare_miss": m.share_l2.isna().astype(float).values, "lag_logn": np.log(m.n.values),
            "lag_pos": m.pos_sh.values, "lag_adv": m.adv_sh.values, "lag_cc": m.cc_sh.values, "lag_cs": m.cs_sh.values,
            "lag_bip": m.bip_sh.values, "share_tgt": m.share_tgt.values,
        })
        out["target"] = out.share_tgt - out.lag_share
        rows.append(out)
    t = pd.concat(rows, ignore_index=True)
    t["chamber"] = t.chamber.fillna("House")  # unmatched ids (few) default to House; flagged in log
    t["isR"] = (t.party == "R").astype(float)
    t["isSen"] = (t.chamber == "Senate").astype(float)
    if fb is not None:
        f = fb[["MemberICPSR", "Year", "share"]].copy(); f["Year"] += 1
        t = t.merge(f.rename(columns={"Year": "Y", "share": "lag_fb_share"}), on=["MemberICPSR", "Y"], how="left")
    else:
        t["lag_fb_share"] = np.nan
    t["lag_fb_miss"] = t.lag_fb_share.isna().astype(float)
    t["lag_fb_share"] = t.lag_fb_share.fillna(t.lag_share)  # impute with own twitter lag; flag carries missingness
    pw = []
    for Y, g in t.groupby("Y"):
        pf = power_features(Y, g.party.values, g.chamber.values); pf.index = g.index; pw.append(pf)
    t = pd.concat([t, pd.concat(pw)], axis=1)
    return t


def feats_only_lags(t):
    """Return the feature matrix columns that depend on the panel (for the perturbation test)."""
    return t[PROFILE + STATIC + POWER].copy()


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def boot_skill(y, yhat, yb, members, B=1000, seed=SEED):
    """Skill = 1 - RMSE_model/RMSE_base (base predictions yb), member-cluster bootstrap. Returns (point, lo, hi, draws)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); yhat = np.asarray(yhat); yb = np.asarray(yb)
    codes, uniq = pd.factorize(members)
    idx = [np.where(codes == k)[0] for k in range(len(uniq))]
    e1 = (y - yhat) ** 2; e0 = (y - yb) ** 2
    s1 = np.array([e1[i].sum() for i in idx]); s0 = np.array([e0[i].sum() for i in idx]); c = np.array([len(i) for i in idx])
    pt = 1 - np.sqrt(e1.mean()) / np.sqrt(e0.mean())
    draws = np.empty(B)
    K = len(idx)
    for b in range(B):
        j = rng.integers(0, K, K)
        draws[b] = 1 - np.sqrt(s1[j].sum() / c[j].sum()) / np.sqrt(s0[j].sum() / c[j].sum())
    return pt, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), draws


# ---------------------------------------------------------------- models
SETS = {"ctrl": STATIC + PROFILE, "power": STATIC + POWER, "both": STATIC + PROFILE + POWER}
ALPHAS = [1, 10, 100, 1000, 10000]
ITERS = [100, 300]
MODELS = {"R_ctrl": ("ridge", "ctrl"), "R_power": ("ridge", "power"), "R_both": ("ridge", "both"),
          "G_ctrl": ("gbt", "ctrl"), "G_both": ("gbt", "both")}


def fit_predict(kind, cols, train, test, hp):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    Xtr = train[cols].values.astype(float); Xte = test[cols].values.astype(float)
    if kind == "ridge":
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=hp).fit(sc.transform(Xtr), train.target.values)
        return m.predict(sc.transform(Xte))
    m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=hp, min_samples_leaf=40,
                                      l2_regularization=1.0, random_state=SEED).fit(Xtr, train.target.values)
    return m.predict(Xte)


def rtm_predict(train, test):
    x = train.lag_share.values; y = train.target.values
    b = np.cov(x, y, bias=True)[0, 1] / x.var(); a = y.mean() - b * x.mean()
    return a + b * test.lag_share.values


def grid(kind):
    return ALPHAS if kind == "ridge" else ITERS


def oracle_pred(test):
    return test.groupby(["Y", "party", "chamber"]).target.transform("mean").values


def direction_table(test, preds):
    """Per target year: realized vs predicted mean change per party (D,R) and D-R gap. preds: name -> array aligned to test rows."""
    test = test.reset_index(drop=True)
    rows = []
    for Y, g in test.groupby("Y"):
        r = {"Y": int(Y)}
        for p in ["D", "R"]:
            s = (g.party == p).values
            r[f"act_{p}"] = float(g.target.values[s].mean())
            for k, v in preds.items():
                r[f"{k}_{p}"] = float(np.asarray(v)[g.index.values][s].mean())
        r["act_gap"] = r["act_D"] - r["act_R"]
        for k in preds:
            r[f"{k}_gap"] = r[f"{k}_D"] - r[f"{k}_R"]
            r[f"{k}_hit_D"] = bool(np.sign(r[f"{k}_D"]) == np.sign(r["act_D"]))
            r[f"{k}_hit_R"] = bool(np.sign(r[f"{k}_R"]) == np.sign(r["act_R"]))
            r[f"{k}_hit_gap"] = bool(np.sign(r[f"{k}_gap"]) == np.sign(r["act_gap"]))
        rows.append(r)
    return pd.DataFrame(rows)


def with_power(t, cfg_year_map=None):
    """Copy of t with power features recomputed; cfg_year_map maps target year -> year whose configuration is used (placebo)."""
    t = t.copy()
    for Y, g in t.groupby("Y"):
        yy = Y if cfg_year_map is None else cfg_year_map[Y]
        pf = power_features(yy, g.party.values, g.chamber.values)
        t.loc[g.index, POWER] = pf[POWER].values
    return t
