"""RVT-A H1a path dependence. Follows output/v7/rvt_a/DESIGN.md."""
import sys, json
sys.path.insert(0, "/home/joneill/x_box/pipeline/scripts/v7")
from rvt_a_common import *

panel = load_panel()
v, first_start = load_service(panel)
svc = years_service(v)
share = panel.set_index(["MemberICPSR", "Year"]).share
pinfo = panel.set_index(["MemberICPSR", "Year"])[["party", "chamber"]]

TR = {2011: ("D", "House", "ch"), 2015: ("D", "Senate", "ch"), 2017: ("D", None, "pres"),
      2019: ("R", "House", "ch"), 2021: ("R", None, "pres")}


def out_year(party, chamber_y, y, kind):
    if kind == "pres":
        return party != PRES[y]
    return party != MAJ[chamber_y][y]


rows = []
drops = {"no_nominate": 0}
for T, (party, ch, kind) in TR.items():
    cur = panel[(panel.Year == T) & (panel.party == party)]
    for r in cur.itertuples():
        i = r.MemberICPSR
        if (i, T - 1) not in share.index:
            continue
        pc = pinfo.loc[(i, T - 1)]
        if pc.party != party or (ch is not None and pc.chamber != ch):
            continue
        if ch is not None and r.chamber != ch:
            continue
        s = svc.get(i, [])
        em = [y for (y, c, p) in s if y < T and out_year(p, c, y, kind)]
        ea = [y for (y, c, p) in s if y < T and (out_year(p, c, y, "pres") or out_year(p, c, y, "ch"))]
        pan = panel[(panel.MemberICPSR == i) & (panel.Year < T)]
        ep = [y for y, c in zip(pan.Year, pan.chamber) if out_year(party, c, y, kind)]
        nm = nominate_at(v, i, T)
        d2pre = [share.get((i, y)) for y in (T - 2, T - 1) if (i, y) in share.index]
        d2post = [share.get((i, y)) for y in (T, T + 1) if (i, y) in share.index]
        rows.append(dict(id=i, T=T, party=party, kind=kind, senate=float(pc.chamber == "Senate"), d1=share[(i, T)] - share[(i, T - 1)],
                         d2=np.mean(d2post) - np.mean(d2pre), pre=share[(i, T - 1)], pre2=np.mean(d2pre),
                         first_start=first_start.get(i, np.nan), sen=T - first_start.get(i, np.nan), nom=nm,
                         E=float(len(em) > 0), dose=len(em), Eany=float(len(ea) > 0), Epanel=float(len(ep) > 0),
                         panel_first=int(pan.Year.min()) if len(pan) else T - 1))
df = pd.DataFrame(rows)
drops["no_nominate"] = int(df.nom.isna().sum())
drops["no_first"] = int(df.first_start.isna().sum())
df = df.dropna(subset=["nom", "first_start"]).reset_index(drop=True)
df["sen2"] = df.sen ** 2 / 10
df["sen3"] = df.sen ** 3 / 100
df["panel_sen"] = df["T"] - df.panel_first
res = {"sample": {}, "drops": drops}
for T in TR:
    g = df[df["T"] == T]
    res["sample"][T] = dict(n=len(g), E1=int(g.E.sum()), E0=int((1 - g.E).sum()), Epanel1=int(g.Epanel.sum()),
                            corr_E_sen=float(np.corrcoef(g.E, g.sen)[0, 1]) if g.E.std() > 0 else None,
                            max_start_E1=float(g[g.E == 1].first_start.max()) if g.E.sum() else None)
res["sample"]["total_rows"] = len(df); res["sample"]["members"] = int(df.id.nunique())
print(json.dumps(res["sample"], indent=1))
sstar = {T: df[(df["T"] == T) & (df.E == 1)].first_start.max() for T in TR}


def design(d, E, outcome, party_sub=None, controls=("sen", "sen2", "pre", "nom", "senate"), fe=True):
    cols = [np.ones(len(d)), d[E].values]
    names = ["const", E]
    for c in controls:
        cols.append(d[c].values); names.append(c)
    if fe:
        Ts = sorted(d["T"].unique())[1:]
        for T in Ts:
            cols.append((d["T"] == T).values.astype(float)); names.append(f"T{T}")
    X = np.column_stack(cols)
    return X, d[outcome].values, names


def run(d, E="E", outcome="d1", controls=("sen", "sen2", "pre", "nom", "senate"), fe=True, wild=True):
    if d[E].std() == 0 or len(d) < 15:
        return None
    ctr = tuple(c for c in controls if d[c].std() > 0)
    if "pre" in controls and outcome == "d2":
        d = d.assign(pre=d.pre2)
    X, y, names = design(d, E, outcome, controls=ctr, fe=fe)
    cl = codes(d.id)
    if wild:
        r = wild_boot_t(X, y, cl, 1)
    else:
        b, se, e, V = ols_cluster(X, y, cl)
        r = dict(b=float(b[1]), se=float(se[1]))
    r["n"] = int(len(d)); r["members"] = int(d.id.nunique())
    return r


out = {}
# primary and variants
out["M1_primary"] = run(df)
out["M2_D"] = run(df[df.party == "D"])
out["M3_R"] = run(df[df.party == "R"])
out["M4_nopre"] = run(df, controls=("sen", "sen2", "nom", "senate"))
out["M1_cubic"] = run(df, controls=("sen", "sen2", "sen3", "pre", "nom", "senate"))
out["M1_Eany"] = run(df, E="Eany")
out["M1_dose"] = run(df, E="dose")
out["M1_Epanel"] = run(df, E="Epanel")
out["M1_D2"] = run(df, outcome="d2")
out["M1_panel_seniority"] = run(df.assign(sen=df.panel_sen, sen2=df.panel_sen ** 2 / 10))
out["M1_minPrimary_noSenateNom"] = run(df, controls=("sen", "sen2", "pre"))
# pooled without wild for speed variants: already wild; leave-one-transition-out
loto = {}
for T in TR:
    loto[T] = run(df[df["T"] != T], wild=False)
out["leave_one_transition_out"] = loto
# per-transition + RE meta
pt = {}
bs, ss = [], []
for T in TR:
    g = df[df["T"] == T]
    r = run(g, fe=False, wild=False)
    pt[T] = r
    if r:
        bs.append(r["b"]); ss.append(r["se"])
out["per_transition"] = pt
out["meta_RE"] = dl_meta(bs, ss) if len(bs) >= 2 else None
# local cohort
loc = df[df.apply(lambda r: abs(r.first_start - sstar[int(r["T"])]) <= 4, axis=1)]
out["local_cohort"] = run(loc, controls=("sen", "pre", "nom", "senate"))
out["local_cohort"]["sample_by_T"] = {int(T): int(((loc["T"] == T)).sum()) for T in TR}
# placebo cutoffs
plc = {}
for k in list(range(-8, -1)) + list(range(2, 9)):
    d = df.copy()
    d["Ek"] = (d.first_start <= d["T"].map(sstar) + k).astype(float)
    r = run(d, E="Ek", wild=False)
    if r:
        plc[k] = dict(b=r["b"], se=r["se"], n1=int(d.Ek.sum()))
out["placebo_cutoff"] = plc
bt = out["M1_primary"]["b"]
pb = np.array([x["b"] for x in plc.values()])
out["placebo_summary"] = dict(true_b=bt, placebo_n=len(pb), placebo_p5=float(np.percentile(pb, 5)), placebo_p95=float(np.percentile(pb, 95)),
                              rank_abs=float((np.abs(pb) >= abs(bt)).mean()), true_outside_central90=bool(bt < np.percentile(pb, 5) or bt > np.percentile(pb, 95)))
# MDE
out["MDE_primary"] = 2.8 * out["M1_primary"]["se"]
# E means
out["raw_means_d1"] = {f"{T}_E{int(e)}": dict(n=int(((df["T"] == T) & (df.E == e)).sum()), mean=float(df[(df["T"] == T) & (df.E == e)].d1.mean()) if ((df["T"] == T) & (df.E == e)).sum() else None)
                       for T in TR for e in (0, 1)}
json.dump({"sample": res["sample"], "drops": drops, **out}, open(f"{OUT}/h1a_results.json", "w"), indent=1, default=float)
df.to_csv(f"{OUT}/h1a_data.csv", index=False)
print(json.dumps(out, indent=1, default=float))
