"""RVT-E follow-up (after the E3 results were seen): why the off-the-shelf anger probability falls for members who lost power
while the purpose-trained genuine-anger score rises. Descriptive. Uses the E3 member-day cache and member sample, and the
four-class emotion labels stored for the 2017 to 2023 sample of the corpus."""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/x_box/pipeline/output/v7")
BASE = os.path.expanduser("~/x_box/pipeline/output")
O = ROOT + "/rvt_e"
D7 = ["offensive", "toxic", "neg_sentiment", "irony", "ragebait", "tribal", "performative"]
SHOW = ["p_emotion_anger", "p_genuine_anger", "p_neg_sentiment", "p_advocacy", "p_agency", "p_performative", "p_tribal", "ti7"]
TR = ["2019-01-03", "2021-01-20", "2023-01-03"]
RES = {}
am = pd.read_csv(ROOT + "/ti/account_member_map.csv")[["screen_name", "bioguide"]].dropna(subset=["bioguide"])
M = pd.read_csv(O + "/e3_displacement_real.csv", usecols=["bioguide", "trans", "lost"])
M["bioguide"] = M.bioguide.astype(str)

# ---- A. post-level correlations, full corpus, non-repost posts of members
pp = pd.read_parquet(ROOT + "/full/post_probs_full.parquet", columns=["screen_name", "post_type", "argmax_emotion_anger"] + [f"p_{d}" for d in D7] +
                     ["p_emotion_anger", "p_genuine_anger", "p_advocacy", "p_agency"])
pp = pp[pp.post_type != "retweet"].merge(am, on="screen_name")
pp["ti7"] = pp[[f"p_{d}" for d in D7]].mean(axis=1)
RES["A_post_level"] = dict(
    n_posts=int(len(pp)),
    corr_pooled={c: float(np.corrcoef(pp.p_emotion_anger, pp[c])[0, 1]) for c in SHOW if c != "p_emotion_anger"},
    corr_within_member={c: float(np.corrcoef(pp.p_emotion_anger - pp.groupby("bioguide").p_emotion_anger.transform("mean"),
                                             pp[c] - pp.groupby("bioguide")[c].transform("mean"))[0, 1]) for c in SHOW if c != "p_emotion_anger"},
    mean_emotion_anger=float(pp.p_emotion_anger.mean()), share_argmax_anger=float(pp.argmax_emotion_anger.mean()),
    mean_genuine_anger=float(pp.p_genuine_anger.mean()), share_genuine_anger_above_half=float((pp.p_genuine_anger > 0.5).mean()))
ga = pp.p_genuine_anger > 0.5
RES["A_post_level"]["emotion_anger_among_genuine_anger_posts"] = dict(mean_prob=float(pp[ga].p_emotion_anger.mean()), share_argmax=float(pp[ga].argmax_emotion_anger.mean()))
RES["A_post_level"]["emotion_anger_among_other_posts"] = dict(mean_prob=float(pp[~ga].p_emotion_anger.mean()), share_argmax=float(pp[~ga].argmax_emotion_anger.mean()))
RES["A_post_level"]["by_argmax_anger"] = {str(k): {c: float(v[c].mean()) for c in SHOW} for k, v in pp.groupby("argmax_emotion_anger")}
del pp

# ---- B. levels before and after each change, by group (member means, then mean over members)
G = pd.read_parquet(O + "/e3_member_day_features.parquet")
G["bioguide"] = G.bioguide.astype(str)
rows = []
for tr in TR:
    t = pd.Timestamp(tr)
    for per, lo, hi in (("before", t - pd.Timedelta(days=180), t), ("after", t, t + pd.Timedelta(days=180))):
        x = G[(G.date >= lo) & (G.date < hi)].groupby("bioguide").sum(numeric_only=True)
        d = pd.DataFrame({c: np.where(x["n_" + c] >= 10, x["s_" + c] / x["n_" + c].where(x["n_" + c] > 0), np.nan) for c in SHOW}, index=x.index)
        d = d.join(M[M.trans == tr].set_index("bioguide").lost, how="inner")
        d["trans"], d["period"] = tr, per
        rows.append(d.reset_index())
L = pd.concat(rows, ignore_index=True)
piv = L.groupby(["trans", "lost", "period"])[SHOW].mean()
RES["B_levels"] = {}
for tr in TR:
    RES["B_levels"][tr] = {}
    for lost, nm in ((1.0, "lost"), (0.0, "gained")):
        b, a = piv.loc[(tr, lost, "before")], piv.loc[(tr, lost, "after")]
        RES["B_levels"][tr][nm] = {c: dict(before=float(b[c]), after=float(a[c]), change=float(a[c] - b[c])) for c in SHOW}

# ---- C. four-class labels in the 2017 to 2023 sample: where the probability mass goes
s = pd.read_parquet(ROOT + "/ti/post_probs.parquet", columns=["screen_name", "created_at", "post_type", "p_genuine_anger", "p_advocacy", "p_neg_sentiment"] + [f"p_{d}" for d in D7 if d != "neg_sentiment"])
v1 = pd.read_parquet(BASE + "/toxicity_index/congress_classified_2017_2023.parquet", columns=["emotion_label"])
assert len(s) == len(v1)
s["emotion"] = v1.emotion_label.astype(str).values
s["ti7"] = s[[f"p_{d}" for d in D7]].mean(axis=1)
s = s[s.post_type != "retweet"].merge(am, on="screen_name")
s["bioguide"] = s.bioguide.astype(str)
s["date"] = pd.to_datetime(s.created_at, utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
LAB = sorted(s.emotion.unique())
RES["C_labels"] = dict(n_posts=int(len(s)), share={k: float(v) for k, v in s.emotion.value_counts(normalize=True).items()},
                       by_label={k: dict(n=int(len(v)), ti7=float(v.ti7.mean()), genuine_anger=float(v.p_genuine_anger.mean()),
                                         genuine_anger_above_half=float((v.p_genuine_anger > 0.5).mean()), neg_sentiment=float(v.p_neg_sentiment.mean()),
                                         advocacy=float(v.p_advocacy.mean()), tribal=float(v.p_tribal.mean()), performative=float(v.p_performative.mean()))
                                 for k, v in s.groupby("emotion")},
                       labels_of_genuine_anger_posts={k: float(v) for k, v in s[s.p_genuine_anger > 0.5].emotion.value_counts(normalize=True).items()},
                       labels_of_top_decile_ti7_posts={k: float(v) for k, v in s[s.ti7 >= s.ti7.quantile(0.9)].emotion.value_counts(normalize=True).items()})
for k in LAB:
    s["is_" + k] = (s.emotion == k).astype(float)
rows = []
for tr in TR:
    t = pd.Timestamp(tr)
    for per, lo, hi in (("before", t - pd.Timedelta(days=180), t), ("after", t, t + pd.Timedelta(days=180))):
        x = s[(s.date >= lo) & (s.date < hi)].groupby("bioguide")
        d = x[["is_" + k for k in LAB]].mean()[x.size() >= 10]
        d = d.join(M[M.trans == tr].set_index("bioguide").lost, how="inner")
        d["trans"], d["period"] = tr, per
        rows.append(d.reset_index())
S = pd.concat(rows, ignore_index=True)
W = S.pivot_table(index=["bioguide", "trans", "lost"], columns="period", values=["is_" + k for k in LAB]).dropna()
RES["C_labels"]["share_change_by_group"] = {}
RES["C_labels"]["lost_minus_gained"] = {}
for tr in TR + ["pooled"]:
    X = W if tr == "pooled" else W[W.index.get_level_values("trans") == tr]
    lost = X.index.get_level_values("lost") == 1
    RES["C_labels"]["share_change_by_group"][tr] = {}
    RES["C_labels"]["lost_minus_gained"][tr] = {}
    for k in LAB:
        d = X[("is_" + k, "after")] - X[("is_" + k, "before")]
        RES["C_labels"]["share_change_by_group"][tr][k] = dict(lost=float(d[lost].mean()), gained=float(d[~lost].mean()),
                                                              lost_before=float(X[("is_" + k, "before")][lost].mean()),
                                                              gained_before=float(X[("is_" + k, "before")][~lost].mean()))
        RES["C_labels"]["lost_minus_gained"][tr][k] = float(d[lost].mean() - d[~lost].mean())
    RES["C_labels"]["share_change_by_group"][tr]["n_members"] = dict(lost=int(lost.sum()), gained=int((~lost).sum()))
json.dump(RES, open(O + "/e8_anger_scores.json", "w"), indent=1)
print(json.dumps(RES, indent=1))
