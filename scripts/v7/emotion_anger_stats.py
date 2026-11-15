"""emotion_anger_stats.py: the two anger scores on the full corpus.

General-purpose score: anger class and anger probability of the four-class emotion model, scored by emotion4_score.py
with the class order of the benchmark mapping file. Purpose-trained score: the genuine-anger classifier.

Record level (all scored records, and non-repost records): class shares, share of posts each score marks as angry,
overlap of the two, Spearman rank correlations with performative outrage and with each other.
Member level (members of the two major parties with at least 50 posts on the basis, as in ti_full_run.py): party
contrasts, Republican minus Democratic, Hedges g with bootstrap intervals (stats_common.contrast_table, same seed).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_common import contrast_table

SEED = 20261010
ROOT = os.path.expanduser("~/x_box/pipeline")
V7 = f"{ROOT}/output/v7"
O = f"{V7}/emotion4"
D7 = ["offensive", "toxic", "neg_sentiment", "irony", "ragebait", "tribal", "performative"]
EM = ["anger", "joy", "optimism", "sadness"]

pp = pd.read_parquet(f"{V7}/full/post_probs_full.parquet", columns=["id", "screen_name", "post_type", "p_genuine_anger", "p_advocacy"] + [f"p_{d}" for d in D7])
e4 = pd.read_parquet(f"{O}/full.parquet")
assert len(pp) == len(e4) and (pp.id.astype(str).values == e4.id.values).all(), "emotion scores are not row-aligned with the corpus table"
for c in EM:
    pp["em_" + c] = e4["p_emotion_" + c].values
pp["em_class"] = e4.emotion_class.astype(str).values
pp["ti7"] = pp[[f"p_{d}" for d in D7]].mean(axis=1)
pp["em_is_anger"] = (pp.em_class == "anger")
pp["ga_flag"] = pp.p_genuine_anger > 0.5
pp["adv_flag"] = pp.p_advocacy > 0.5
pp["primary"] = pp.post_type != "retweet"
RES = {}


def rho(a, b):
    return float(stats.spearmanr(a, b).statistic)


def record_block(x):
    d = dict(n=int(len(x)), class_share={c: float((x.em_class == c).mean()) for c in EM},
             mean_prob={c: float(x["em_" + c].mean()) for c in EM},
             genuine_anger_above_half=float(x.ga_flag.mean()),
             both=float((x.em_is_anger & x.ga_flag).mean()), either=float((x.em_is_anger | x.ga_flag).mean()),
             genuine_anger_above_half_within_anger_class=float(x.ga_flag[x.em_is_anger].mean()),
             anger_class_within_genuine_anger_above_half=float(x.em_is_anger[x.ga_flag].mean()),
             advocacy_above_half=float(x.adv_flag.mean()),
             advocacy_above_half_within_anger_class=float(x.adv_flag[x.em_is_anger].mean()),
             advocacy_above_half_within_genuine_anger_above_half=float(x.adv_flag[x.ga_flag].mean()),
             spearman=dict(emotion_anger_vs_performative=rho(x.em_anger, x.p_performative),
                           genuine_anger_vs_performative=rho(x.p_genuine_anger, x.p_performative),
                           emotion_anger_vs_genuine_anger=rho(x.em_anger, x.p_genuine_anger),
                           emotion_anger_vs_ti7=rho(x.em_anger, x.ti7), genuine_anger_vs_ti7=rho(x.p_genuine_anger, x.ti7),
                           emotion_anger_vs_advocacy=rho(x.em_anger, x.p_advocacy), genuine_anger_vs_advocacy=rho(x.p_genuine_anger, x.p_advocacy),
                           emotion_optimism_vs_advocacy=rho(x.em_optimism, x.p_advocacy)),
             by_class={c: dict(n=int(len(g)), ti7=float(g.ti7.mean()), genuine_anger_above_half=float(g.ga_flag.mean()),
                               advocacy_above_half=float(g.adv_flag.mean()), performative=float(g.p_performative.mean()),
                               neg_sentiment=float(g.p_neg_sentiment.mean())) for c, g in x.groupby("em_class")})
    return d


RES["records_all"] = record_block(pp)
RES["records_nonrepost"] = record_block(pp[pp.primary])
print(json.dumps({k: {a: b for a, b in v.items() if a != "by_class"} for k, v in RES.items()}, indent=1), flush=True)

# ---- member level, as in ti_full_run.py
u = pp.drop_duplicates("id", keep="first")
amap = pd.read_csv(f"{V7}/ti/account_member_map.csv", dtype={"user_id": str})[["screen_name", "bioguide", "entity_type", "party"]]
u = u.merge(amap, on="screen_name", how="left")
mem = u[u.entity_type == "member"].copy()
for c in EM:
    mem["cls_" + c] = (mem.em_class == c).astype(float)
mem["ga"] = mem.ga_flag.astype(float)
mem["anger_cls_nonadv"] = np.where(~mem.adv_flag, mem.cls_anger, np.nan)
mem["ga_nonadv"] = np.where(~mem.adv_flag, mem.ga, np.nan)
MEAS = ["cls_anger", "em_anger", "ga", "p_genuine_anger", "cls_optimism", "em_optimism", "cls_joy", "cls_sadness", "anger_cls_nonadv", "ga_nonadv", "ti7"]
RES["member"] = {}
for basis, s in (("nonrt", mem[mem.primary]), ("allposts", mem)):
    g = s.groupby("bioguide")[MEAS].mean()
    g["n_posts"] = s.groupby("bioguide").size()
    g["party"] = s.groupby("bioguide").party.first()
    dr = g[(g.n_posts >= 50) & g.party.isin(["D", "R"])].dropna(subset=MEAS)
    ct = pd.DataFrame(contrast_table(dr, MEAS, seed=SEED))
    ct.to_csv(f"{O}/anger_party_contrasts_{basis}.csv", index=False)
    RES["member"][basis] = dict(n_R=int((dr.party == "R").sum()), n_D=int((dr.party == "D").sum()),
                                contrasts={r.measure: dict(mean_R=float(r.mean1), mean_D=float(r.mean2), g=float(r.hedges_g), g_lo=float(r.g_lo),
                                                           g_hi=float(r.g_hi), p=float(r.p)) for r in ct.itertuples()},
                                member_corr=dict(anger_class_rate_vs_genuine_anger_rate=float(np.corrcoef(dr.cls_anger, dr.ga)[0, 1]),
                                                 anger_class_rate_vs_ti7=float(np.corrcoef(dr.cls_anger, dr.ti7)[0, 1]),
                                                 genuine_anger_rate_vs_ti7=float(np.corrcoef(dr.ga, dr.ti7)[0, 1])))
json.dump(RES, open(f"{O}/anger_stats.json", "w"), indent=1)
print(json.dumps(RES["member"], indent=1))
print("DONE")
